"""ResNet-18 + HyperLSTM 自回归笔画提取器（对齐 seq_extract 设计）。

关键改动相比上一版：
1. 输出激活遵循 seq_extract:get_mixture_coef
   - x1y1 = sigmoid           ∈ [0, 1]   patch 内绝对位置
   - x2y2 = tanh              ∈ [-1, 1]  相对 cursor 的 offset
   - width = sigmoid * (1-min) + min  ∈ [min_width, 1]
   - scaling = sigmoid * max_scaling   ∈ [0, max_scaling]
   - pen = softmax(logits) → 取 P(pen=1) 的 softmax 软值（差分 argmax）
2. forward 现在是真正的可微 rollout：caller 给出 step 数，模型自己迭代更新 cursor/canvas/window，
   全过程 torch 可微。
3. closed-loop 训练：不用 prev_stroke 特征、不做 scheduled sampling；训练时可注入随机 init_cursor。

被 train.py 和 inference.py 共享。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from diffable_state import (
    RolloutState, init_rollout_state, crop_patch_around_cursor, step_with_renderer,
    MIN_WIDTH, MAX_SCALING, MIN_WINDOW_SIZE,
)
from rnn import HyperLSTMCell


class ResNetFeatureBackbone(nn.Module):
    """ResNet-18 stem + body，返回 token + 全局特征。"""

    def __init__(self, image_size: int = 256, d_model: int = 256, in_chans: int = 1):
        super().__init__()
        resnet = resnet18(weights=None)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if in_chans == 1:
            with torch.no_grad():
                self.conv1.weight.copy_(resnet.conv1.weight.data.mean(dim=1, keepdim=True))
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.proj = nn.Conv2d(512, d_model, kernel_size=1)
        num_tokens = (image_size // 32) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)

    def forward_features(self, image: torch.Tensor):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)
        tokens = x.flatten(2).transpose(1, 2)  # (N, T, d)
        n_tokens = tokens.shape[1]
        if n_tokens != self.pos_embed.shape[1]:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=n_tokens, mode='linear', align_corners=False).transpose(1, 2)
        else:
            pos = self.pos_embed
        return tokens + pos


class PatchEncoder(nn.Module):
    """编码 (target_patch ∥ canvas_patch) 的局部 CNN。"""

    def __init__(self, patch_size: int = 64, d_model: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * (patch_size // 8) ** 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, target_patch: torch.Tensor, canvas_patch: torch.Tensor) -> torch.Tensor:
        x = torch.cat([target_patch, canvas_patch], dim=1)
        x = self.conv(x)
        x = x.flatten(1)
        return self.proj(x)


class StrokeHead(nn.Module):
    """把 GRU hidden 解出 (pen_logits, x1y1, x2y2, width, scaling)。
    输出激活严格对齐 seq_extract.get_mixture_coef。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.pen_head = nn.Linear(hidden_dim, 2)        # softmax logits for [pen=0, pen=1]
        self.x1y1_head = nn.Linear(hidden_dim, 2)       # sigmoid → [0, 1]
        self.x2y2_head = nn.Linear(hidden_dim, 2)       # tanh    → [-1, 1]
        self.width_head = nn.Linear(hidden_dim, 1)      # sigmoid * (1 - min) + min
        self.scaling_head = nn.Linear(hidden_dim, 1)    # sigmoid * max_scaling
        self.soft_beta = 10.0

    def soft_argmax(self, logits: torch.Tensor) -> torch.Tensor:
        """对齐 seq_extract.differentiable_argmax，β=10。
        返回 P(pen=1) 的可微近似 ∈ [0, 1]。"""
        N, C = logits.shape
        ar = torch.cumsum(torch.ones_like(logits), dim=1) - 1.0  # [0, 1]
        return (F.softmax(logits * self.soft_beta, dim=1) * ar).sum(dim=1)

    def forward(self, hidden: torch.Tensor) -> dict:
        pen_logits = self.pen_head(hidden)
        pen_soft = self.soft_argmax(pen_logits)  # (N,)
        pen_hard = pen_logits.argmax(dim=-1).float()  # (N,)

        x1y1 = torch.sigmoid(self.x1y1_head(hidden))
        x2y2 = torch.tanh(self.x2y2_head(hidden))
        width = torch.sigmoid(self.width_head(hidden)) * (1.0 - MIN_WIDTH) + MIN_WIDTH
        scaling = torch.sigmoid(self.scaling_head(hidden)) * MAX_SCALING

        return {
            'pen_logits': pen_logits,
            'pen_state_soft': pen_soft,
            'pen_state_hard': pen_hard,
            'x1y1': x1y1,
            'x2y2': x2y2,
            'width': width,
            'scaling': scaling,
        }


def stroke7_to_step_pred(stroke7: torch.Tensor) -> dict:
    """把 GT seq7 的一步转换成 step_with_renderer 可消费的 pred dict。"""
    pen = stroke7[:, 0]
    return {
        'pen_state_soft': pen,
        'pen_state_hard': pen,
        'x1y1': stroke7[:, 1:3],
        'x2y2': stroke7[:, 3:5],
        'width': stroke7[:, 5:6],
        'scaling': stroke7[:, 6:7],
    }


class ResNetAutoregressiveExtractor7D(nn.Module):
    """对齐 seq_extract VirtualSketchingModel 的简化 PyTorch 版本。

    输入：target image (B, 1, H, W) ∈ [0, 1]，1=BG / 0=stroke
    输出：模型自己 rollout T 步的 (B, T, 7) 序列 + rollout 中间产物。"""

    def __init__(self,
                 image_size: int = 256,
                 max_seq_len: int = 100,
                 d_model: int = 256,
                 hidden_dim: int = 256,
                 num_heads: int = None,
                 patch_size: int = 64,
                 raster_size: int = 128,
                 init_window_size: float = None):
        super().__init__()
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.raster_size = raster_size
        self.init_window_size = init_window_size or float(min(128, image_size))

        if num_heads is None:
            for n in [16, 8, 4, 2, 1]:
                if d_model % n == 0:
                    num_heads = n
                    break
        assert d_model % num_heads == 0
        self.num_heads = num_heads

        # 全局图像 token + 全局 pooling
        self.target_backbone = ResNetFeatureBackbone(image_size=image_size, d_model=d_model)
        self.global_norm = nn.LayerNorm(d_model)

        # 局部 patch 编码（target_patch ∥ canvas_patch）
        self.patch_encoder = PatchEncoder(patch_size=patch_size, d_model=d_model)

        # canvas 全局编码（用于跟踪已画区域）
        self.canvas_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(d_model),
        )

        # 标量 / 短向量 embedding
        self.cursor_mlp = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.LayerNorm(d_model))
        # 注意：故意不要 prev_stroke MLP——它在 closed-loop 训练里会成为 cheat sheet，
        #      让模型学到"下一步 ≈ 函数(上一步)"而忽略 target image。
        self.window_mlp = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.step_mlp = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.LayerNorm(d_model))

        # patch ↔ target 全局 attention，对齐 seq_extract 的 combine encoder
        self.patch_target_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True)

        self.gru_input_proj = nn.Sequential(
            nn.Linear(d_model * 5, hidden_dim),  # patch_attn + target_global + canvas + cursor + (win+step)
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = HyperLSTMCell(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            forget_bias=1.0,
            use_recurrent_dropout=True,
            dropout_keep_prob=0.9,
            use_layer_norm=True,
            hyper_num_units=256,
            hyper_embedding_size=32,
            hyper_use_recurrent_dropout=False
        )

        self.head = StrokeHead(hidden_dim)

    # --------------------------------------------------------------------- #
    # 编码相关
    # --------------------------------------------------------------------- #
    def encode_target(self, target_mask: torch.Tensor):
        """target_mask: (N, 1, H, W) ∈ [0, 1], 1=stroke"""
        tokens = self.target_backbone.forward_features(target_mask)
        global_feat = self.global_norm(tokens.mean(dim=1))
        return tokens, global_feat

    def encode_step(self, target_tokens, target_global, target_mask, state: RolloutState,
                    step_index: torch.Tensor, hidden):
        """target_mask: (N, 1, H, W), state: 当前 rollout 状态, step_index: (N, 1) ∈ [0, 1]"""
        curr_window = state.prev_scaling * state.prev_window_size
        curr_window = torch.clamp(curr_window, MIN_WINDOW_SIZE, float(state.img_size))

        # 局部 patch
        target_patch = crop_patch_around_cursor(target_mask, state.cursor,
                                                curr_window, self.patch_size)
        canvas_patch = crop_patch_around_cursor(state.canvas, state.cursor,
                                                curr_window, self.patch_size)
        patch_feat = self.patch_encoder(target_patch, canvas_patch)  # (N, d)

        # patch ↔ target 全局 attention
        patch_query = patch_feat.unsqueeze(1)
        patch_attn, _ = self.patch_target_attn(patch_query, target_tokens, target_tokens)
        patch_attn = patch_attn.squeeze(1)

        # canvas 全局
        canvas_feat = self.canvas_encoder(state.canvas)

        # 标量 embeddings (不再用 prev_stroke——见 __init__ 中的注释)
        cursor_feat = self.cursor_mlp(state.cursor)
        win_top = curr_window / float(state.img_size)
        win_bot = curr_window / MIN_WINDOW_SIZE
        window_feat = self.window_mlp(torch.cat([win_top, win_bot], dim=-1))
        step_feat = self.step_mlp(step_index)

        gru_input = self.gru_input_proj(torch.cat([
            patch_attn, target_global, canvas_feat, cursor_feat,
            window_feat + step_feat
        ], dim=-1))

        # HyperLSTM returns (h, new_state)
        h, new_state = self.gru(gru_input, hidden)
        return h, new_state

    # --------------------------------------------------------------------- #
    # rollout
    # --------------------------------------------------------------------- #
    def forward(self, *args, **kwargs):
        """nn.Module.forward 调 rollout，使 DDP 能拦截到 gradient reduce。"""
        return self.rollout(*args, **kwargs)

    def rollout(self,
                target_image: torch.Tensor,
                neural_renderer,
                seq_len: int = None,
                gt_strokes: torch.Tensor = None,
                scheduled_sampling_prob: float = 0.0,  # noqa: ARG002 — 保留旧参数兼容
                teacher_forcing_prob: float = 0.0,
                detach_canvas_for_encoder: bool = True,
                init_state: RolloutState = None,
                init_hidden: torch.Tensor = None,
                init_cursor: torch.Tensor = None) -> dict:
        """从 target_image 出发，模型自闭环 unroll 一段序列。

        Args:
            target_image: (N, 1, H, W) ∈ [0, 1], 1=BG / 0=stroke（与 dataset 输出一致）
            neural_renderer: 预训练 NeuralRasterizorStep
            seq_len: rollout 步数；默认 self.max_seq_len
            gt_strokes: 保留签名兼容旧调用；rollout 内不再使用
            scheduled_sampling_prob: 已废弃，保留签名兼容旧 shell
            init_cursor: (N, 2) 可选初始 cursor；训练时由 train.py 从 stroke 像素采样
            detach_canvas_for_encoder: 把传给 encoder 的 canvas detach，避免长链 RNN 梯度

        Returns:
            dict {
              'seq':                 (N, T, 7) 模型预测的 7D 序列（合 pen_soft + x2y2 + x1y1 + width + scaling）
              'pen_logits':          (N, T, 2)
              'rendered':            (N, H, W) ∈ [0, 1] 最终 soft canvas
              'pos_before_max_min':  (N, T, 2)  pre-clip 像素 cursor，给 outside loss
              'win_size_before_max_min': (N, T, 1)
              'cursors':             (N, T, 2)  [0, 1) 归一化坐标（post-clip）
              'window_sizes':        (N, T, 1)
            }
        """
        if seq_len is None:
            seq_len = self.max_seq_len

        # target_image 是 [0=stroke, 1=BG]（PIL 灰度图）；模型需要 mask = 1-image
        target_mask = 1.0 - target_image

        N = target_image.shape[0]
        device = target_image.device
        dtype = target_image.dtype

        target_tokens, target_global = self.encode_target(target_mask)

        if init_state is None:
            state = init_rollout_state(N, self.image_size, device, dtype)
            state.prev_window_size = torch.full_like(state.prev_window_size,
                                                     self.init_window_size)
            if init_cursor is not None:
                # init_cursor: (N, 2) ∈ [0, 1)，由 train.py 从 stroke 像素采样
                state.cursor = init_cursor.to(device=device, dtype=dtype)
        else:
            state = init_state

        if init_hidden is not None:
            hidden = init_hidden
        else:
            # HyperLSTM initial state: (total_h, total_c)
            hidden = self.gru.get_initial_state(N, device)

        seqs = []
        pen_logits_list = []
        pos_list = []
        win_size_list = []
        cursor_list = []
        window_list = []

        for t in range(seq_len):
            step_index = torch.full((N, 1), t / max(seq_len, 1), device=device, dtype=dtype)

            # encoder 用 detach 后的 canvas，避免 RNN 长链
            state_for_enc = state
            if detach_canvas_for_encoder:
                state_for_enc = RolloutState(
                    cursor=state.cursor,
                    canvas=state.canvas.detach(),
                    prev_width=state.prev_width,
                    prev_scaling=state.prev_scaling,
                    prev_window_size=state.prev_window_size,
                    prev_stroke=state.prev_stroke,
                    img_size=state.img_size,
                )
            h, hidden = self.encode_step(target_tokens, target_global, target_mask,
                                      state_for_enc, step_index, hidden)
            pred = self.head(h)

            # 记录中间值（pre-step）
            pen_logits_list.append(pred['pen_logits'])

            # phase1 warmup: 可用 GT 当前笔推进状态，让下一步输入状态与 GT 序列对齐。
            step_pred = pred
            if gt_strokes is not None and t < gt_strokes.shape[1] and teacher_forcing_prob > 0.0:
                use_gt = teacher_forcing_prob >= 1.0
                if not use_gt and self.training:
                    use_gt = bool(torch.rand((), device=device) < teacher_forcing_prob)
                if use_gt:
                    step_pred = stroke7_to_step_pred(gt_strokes[:, t].to(device=device, dtype=dtype))

            state, info = step_with_renderer(state, step_pred, neural_renderer,
                                             raster_size=self.raster_size)

            # 组装 7D stroke 输出（与原 seq_extract 的 pred_params 对齐：
            #   [pen, x1, y1, x2, y2, r, s]，其中 x1y1 是 sigmoid 后，x2y2 是 tanh 后）
            stroke7 = torch.cat([
                pred['pen_state_soft'].view(N, 1),
                pred['x1y1'],
                pred['x2y2'],
                pred['width'],
                pred['scaling'],
            ], dim=-1)
            seqs.append(stroke7)

            pos_list.append(info['pos_before_max_min'])
            win_size_list.append(info['win_size_before_max_min'])
            cursor_list.append(state.cursor)
            window_list.append(info['curr_window_size'])

            # 不做 scheduled sampling: 训练时让模型用自己的 prediction 推进 cursor/canvas，
            # 这样 phase1 才是真正的 closed-loop 训练（与 inference 一致）。
            # 否则模型会学会"prev_stroke ≈ GT 上一步 → 复制 + 微调"的 cheat，
            # 而忽略 target image。

        return {
            'seq': torch.stack(seqs, dim=1),
            'pen_logits': torch.stack(pen_logits_list, dim=1),
            'rendered': state.canvas.squeeze(1),  # (N, H, W)
            'pos_before_max_min': torch.stack(pos_list, dim=1),
            'win_size_before_max_min': torch.stack(win_size_list, dim=1),
            'cursors': torch.stack(cursor_list, dim=1),
            'window_sizes': torch.stack(window_list, dim=1),
            'final_state': state,
            'final_hidden': hidden,
        }


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f'Total: {total/1e6:.2f}M, Trainable: {trainable/1e6:.2f}M'


if __name__ == '__main__':
    from neural_renderer import NeuralRasterizorStep
    m = ResNetAutoregressiveExtractor7D(image_size=256, max_seq_len=20, d_model=128)
    r = NeuralRasterizorStep(raster_size=128)
    img = torch.rand(2, 1, 256, 256)
    out = m.rollout(img, r)
    print('seq', out['seq'].shape, 'rendered', out['rendered'].shape)
    print(count_parameters(m))

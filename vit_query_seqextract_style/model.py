"""ViT + HyperLSTM (seq_extract style) 自回归笔画提取器。

ViT 全局编码 + HyperLSTMCell 解码器，与 seq_extract 完全对齐。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

from diffable_state import (
    RolloutState, init_rollout_state, crop_patch_around_cursor, step_with_renderer,
    MIN_WIDTH, MAX_SCALING, MIN_WINDOW_SIZE,
)
from rnn import HyperLSTMCell


class ViTBackbone(nn.Module):
    """torchvision vit_b_16 预训练；输入 (N, 1, H, W) → tokens (N, T, d_model)。"""

    def __init__(self, img_size: int = 224, d_model: int = 256, pretrained: bool = True):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        vit = vit_b_16(weights=weights)

        old_proj = vit.conv_proj
        new_proj = nn.Conv2d(
            1,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
        )
        with torch.no_grad():
            new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        vit.conv_proj = new_proj

        self.vit = vit
        self.img_size = img_size
        self.embed_dim = vit.hidden_dim
        self.patch_size = vit.patch_size
        self.feat_proj = nn.Linear(self.embed_dim, d_model)
        self.d_model = d_model

        if img_size != 224:
            self._interpolate_pos_embed(target_size=img_size)

    def _interpolate_pos_embed(self, target_size: int):
        pe = self.vit.encoder.pos_embedding
        cls_pe, patch_pe = pe[:, :1], pe[:, 1:]
        old_grid = int(round((patch_pe.shape[1]) ** 0.5))
        new_grid = target_size // self.patch_size
        patch_pe = patch_pe.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(
            patch_pe, size=(new_grid, new_grid),
            mode='bicubic', align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
        self.vit.encoder.pos_embedding = nn.Parameter(torch.cat([cls_pe, patch_pe], dim=1))
        self.vit.image_size = target_size

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        n = image.shape[0]
        x = self.vit._process_input(image)
        cls_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.vit.encoder(x)
        return self.feat_proj(x[:, 1:])


class AddCoordChannels(nn.Module):
    """Add normalized x/y coord channels, matching seq_extract add_coordconv."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing='ij',
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(n, -1, -1, -1)
        return torch.cat([x, coords], dim=1)


class Conv13C3Stem(nn.Module):
    """conv13_c3 风格 CNN stem.

    文档中的主体是 5 组 stride-2/stride-1 的 3x3 Conv + IN + ReLU，
    输入额外拼接 CoordConv 的 x/y 两个通道。
    """

    def __init__(self, in_chans: int):
        super().__init__()
        channels = [
            (32, 2), (32, 1),
            (64, 2), (64, 1),
            (128, 2), (128, 1),
            (256, 2), (256, 1),
            (256, 2), (256, 1),
        ]
        layers = [AddCoordChannels()]
        curr = in_chans + 2
        for out_ch, stride in channels:
            layers.extend([
                nn.Conv2d(curr, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.ReLU(inplace=True),
            ])
            curr = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Conv13C3Backbone(nn.Module):
    """Global image encoder; returns spatial tokens for attention."""

    def __init__(self, in_chans: int = 1, d_model: int = 256):
        super().__init__()
        self.stem = Conv13C3Stem(in_chans)
        self.proj = nn.Conv2d(256, d_model, kernel_size=1)

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.proj(self.stem(image))
        return feat.flatten(2).transpose(1, 2)


class Conv13C3VectorEncoder(nn.Module):
    """conv13_c3 风格 CNN，输出单个向量，用于 canvas/global 状态编码。"""

    def __init__(self, in_chans: int = 1, d_model: int = 256):
        super().__init__()
        self.stem = Conv13C3Stem(in_chans)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(self.stem(image))


class PatchEncoder(nn.Module):
    """Local target/canvas patch encoder using conv13_c3 style CNN."""

    def __init__(self, patch_size: int = 64, d_model: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.stem = Conv13C3Stem(in_chans=2)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, target_patch: torch.Tensor, canvas_patch: torch.Tensor) -> torch.Tensor:
        x = torch.cat([target_patch, canvas_patch], dim=1)
        return self.proj(self.stem(x))


class StrokeHead(nn.Module):
    """与 lightweight 同名结构，输出激活对齐 seq_extract.get_mixture_coef。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.pen_head = nn.Linear(hidden_dim, 2)
        self.x1y1_head = nn.Linear(hidden_dim, 2)
        self.x2y2_head = nn.Linear(hidden_dim, 2)
        self.width_head = nn.Linear(hidden_dim, 1)
        self.scaling_head = nn.Linear(hidden_dim, 1)
        self.soft_beta = 10.0

    def soft_argmax(self, logits: torch.Tensor) -> torch.Tensor:
        ar = torch.cumsum(torch.ones_like(logits), dim=1) - 1.0
        return (F.softmax(logits * self.soft_beta, dim=1) * ar).sum(dim=1)

    def forward(self, hidden: torch.Tensor) -> dict:
        pen_logits = self.pen_head(hidden)
        pen_soft = self.soft_argmax(pen_logits)
        pen_hard = pen_logits.argmax(dim=-1).float()

        x1y1 = torch.sigmoid(self.x1y1_head(hidden))
        x2y2 = torch.tanh(self.x2y2_head(hidden))
        width = torch.sigmoid(self.width_head(hidden)) * (1.0 - MIN_WIDTH) + MIN_WIDTH
        scaling = torch.sigmoid(self.scaling_head(hidden)) * MAX_SCALING

        return {
            'pen_logits': pen_logits,
            'pen_state_soft': pen_soft,
            'pen_state_hard': pen_hard,
            'x1y1': x1y1, 'x2y2': x2y2,
            'width': width, 'scaling': scaling,
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


class ViTAutoregressiveExtractor7D(nn.Module):
    """ViT + conv13_c3 自回归提取器。

    ViT 负责全局目标图像 token；conv13_c3 风格 CNN 负责局部 patch/canvas 编码。
    """

    def __init__(self,
                 image_size: int = 224,
                 max_seq_len: int = 48,
                 d_model: int = 256,
                 hidden_dim: int = 256,
                 num_heads: int = None,
                 patch_size: int = 64,
                 raster_size: int = 128,
                 init_window_size: float = None,
                 pretrained: bool = True):
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

        self.target_backbone = ViTBackbone(
            img_size=image_size, d_model=d_model, pretrained=pretrained)
        self.global_norm = nn.LayerNorm(d_model)

        self.patch_encoder = PatchEncoder(patch_size=patch_size, d_model=d_model)

        self.canvas_encoder = Conv13C3VectorEncoder(in_chans=1, d_model=d_model)

        self.cursor_mlp = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.LayerNorm(d_model))
        # 故意不要 prev_stroke MLP——它在 closed-loop 训练里会成为 cheat sheet，
        # 让模型学到"下一步 ≈ 函数(上一步)"而忽略 target image。
        self.window_mlp = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.step_mlp = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.LayerNorm(d_model))

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

    def encode_target(self, target_mask: torch.Tensor):
        tokens = self.target_backbone.forward_features(target_mask)
        global_feat = self.global_norm(tokens.mean(dim=1))
        return tokens, global_feat

    def encode_step(self, target_tokens, target_global, target_mask,
                    state: RolloutState, step_index: torch.Tensor,
                    hidden):
        curr_window = state.prev_scaling * state.prev_window_size
        curr_window = torch.clamp(curr_window, MIN_WINDOW_SIZE, float(state.img_size))

        target_patch = crop_patch_around_cursor(target_mask, state.cursor,
                                                curr_window, self.patch_size)
        canvas_patch = crop_patch_around_cursor(state.canvas, state.cursor,
                                                curr_window, self.patch_size)
        patch_feat = self.patch_encoder(target_patch, canvas_patch)

        patch_query = patch_feat.unsqueeze(1)
        patch_attn, _ = self.patch_target_attn(patch_query, target_tokens, target_tokens)
        patch_attn = patch_attn.squeeze(1)

        canvas_feat = self.canvas_encoder(state.canvas)
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

    def forward(self, *args, **kwargs):
        """nn.Module.forward 调 rollout，使 DDP 能拦截到 gradient reduce。"""
        return self.rollout(*args, **kwargs)

    def rollout(self,
                target_image: torch.Tensor,
                neural_renderer,
                seq_len: int = None,
                gt_strokes: torch.Tensor = None,
                scheduled_sampling_prob: float = 0.0,  # noqa: ARG002 — 已废弃
                teacher_forcing_prob: float = 0.0,
                detach_canvas_for_encoder: bool = True,
                init_state: RolloutState = None,
                init_hidden: torch.Tensor = None,
                init_cursor: torch.Tensor = None,
                force_pen_down_until_jump: bool = False,
                pen_jump_threshold: float = 0.25) -> dict:
        if seq_len is None:
            seq_len = self.max_seq_len
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

        seqs, pen_logits_list, pos_list, win_size_list, cursor_list, window_list = \
            [], [], [], [], [], []

        for t in range(seq_len):
            step_index = torch.full((N, 1), t / max(seq_len, 1), device=device, dtype=dtype)
            state_for_enc = state
            if detach_canvas_for_encoder:
                state_for_enc = RolloutState(
                    cursor=state.cursor, canvas=state.canvas.detach(),
                    prev_width=state.prev_width, prev_scaling=state.prev_scaling,
                    prev_window_size=state.prev_window_size,
                    prev_stroke=state.prev_stroke, img_size=state.img_size,
                )
            h, hidden = self.encode_step(target_tokens, target_global, target_mask,
                                      state_for_enc, step_index, hidden)
            pred = self.head(h)
            pen_logits_list.append(pred['pen_logits'])

            if force_pen_down_until_jump:
                curr_window = state.prev_scaling * state.prev_window_size
                curr_window = torch.clamp(curr_window, MIN_WINDOW_SIZE, float(state.img_size))
                endpoint_delta = pred['x2y2'] * curr_window / (2.0 * float(state.img_size))
                jump = endpoint_delta.norm(dim=-1)
                forced_pen = (jump > float(pen_jump_threshold)).to(dtype=dtype)
                pred = {
                    **pred,
                    'pen_state_soft': forced_pen,
                    'pen_state_hard': forced_pen,
                }

            step_pred = pred
            if gt_strokes is not None and t < gt_strokes.shape[1] and teacher_forcing_prob > 0.0:
                use_gt = teacher_forcing_prob >= 1.0
                if not use_gt and self.training:
                    use_gt = bool(torch.rand((), device=device) < teacher_forcing_prob)
                if use_gt:
                    step_pred = stroke7_to_step_pred(gt_strokes[:, t].to(device=device, dtype=dtype))

            state, info = step_with_renderer(state, step_pred, neural_renderer,
                                             raster_size=self.raster_size)
            stroke7 = torch.cat([
                pred['pen_state_soft'].view(N, 1),
                pred['x1y1'], pred['x2y2'],
                pred['width'], pred['scaling'],
            ], dim=-1)
            seqs.append(stroke7)
            pos_list.append(info['pos_before_max_min'])
            win_size_list.append(info['win_size_before_max_min'])
            cursor_list.append(state.cursor)
            window_list.append(info['curr_window_size'])

            # 不做 scheduled sampling: 训练时让模型用自己的 prediction 推进 cursor/canvas，
            # 这样 phase1 才是真正的 closed-loop 训练（与 inference 一致）。

        return {
            'seq': torch.stack(seqs, dim=1),
            'pen_logits': torch.stack(pen_logits_list, dim=1),
            'rendered': state.canvas.squeeze(1),
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
    m = ViTAutoregressiveExtractor7D(image_size=224, max_seq_len=12, d_model=256,
                                     hidden_dim=256, patch_size=64,
                                     pretrained=False)
    r = NeuralRasterizorStep(raster_size=128)
    img = torch.rand(2, 1, 224, 224)
    out = m.rollout(img, r)
    print('seq', out['seq'].shape, 'rendered', out['rendered'].shape)
    print(count_parameters(m))

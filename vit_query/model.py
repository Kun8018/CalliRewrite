"""ViT-B/16 (ImageNet-pretrained) + GRU 自回归笔画提取器。

与 lightweight/ 的 ResNet 版本结构一致；唯一区别在 backbone：
torchvision.vit_b_16 的预训练权重 + 单通道输入适配 + 224 输入。

closed-loop 训练：不用 prev_stroke 特征、不做 scheduled sampling；训练时可注入 init_cursor。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

from diffable_state import (
    RolloutState, init_rollout_state, crop_patch_around_cursor, step_with_renderer,
    MIN_WIDTH, MAX_SCALING, MIN_WINDOW_SIZE,
)


class ViTBackbone(nn.Module):
    """torchvision vit_b_16 预训练；输入 (N, 1, H, W) → tokens (N, T, d_model)。

    适配：
    - 把 conv proj 第一个 in_channels=3 改成 1（用 mean of RGB 权重做 init）。
    - 把 image_size 改成 args.img_size（vit_b_16 默认 224，pos_embed 需要 interpolate）。
    - 输出 patch token，不带 cls。
    """

    def __init__(self, img_size: int = 224, d_model: int = 256, pretrained: bool = True):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        vit = vit_b_16(weights=weights)

        # 替换 conv proj 适配单通道输入
        old_proj = vit.conv_proj  # in=3, out=768, k=patch_size, stride=patch_size
        new_proj = nn.Conv2d(1, old_proj.out_channels,
                              kernel_size=old_proj.kernel_size,
                              stride=old_proj.stride, padding=old_proj.padding)
        with torch.no_grad():
            new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        vit.conv_proj = new_proj

        # 关键：torchvision vit 的 image_size 是 build 时固定的；通过 interpolate
        # pos_embed 即可兼容更小输入（224 默认就 ok；如果 img_size != 224 见 interpolate_pos_embed）
        self.vit = vit
        self.img_size = img_size
        self.embed_dim = vit.hidden_dim  # 768
        self.patch_size = vit.patch_size  # 16

        self.feat_proj = nn.Linear(self.embed_dim, d_model)
        self.d_model = d_model

        # 如果 img_size 与预训练 224 不一致，需 interpolate pos_embed
        if img_size != 224:
            self._interpolate_pos_embed(target_size=img_size)

    def _interpolate_pos_embed(self, target_size: int):
        pe = self.vit.encoder.pos_embedding  # (1, 197, 768)
        cls_pe, patch_pe = pe[:, :1], pe[:, 1:]
        old_grid = int(round((patch_pe.shape[1]) ** 0.5))
        new_grid = target_size // self.patch_size
        patch_pe = patch_pe.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(patch_pe, size=(new_grid, new_grid),
                                 mode='bicubic', align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
        new_pe = torch.cat([cls_pe, patch_pe], dim=1)
        self.vit.encoder.pos_embedding = nn.Parameter(new_pe)
        # vit.image_size 用于内部 reshape，更新它
        self.vit.image_size = target_size

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        """image: (N, 1, H, W)
        returns: (N, num_patches, d_model)"""
        N = image.shape[0]
        x = self.vit._process_input(image)  # (N, num_patches, embed_dim)
        cls_token = self.vit.class_token.expand(N, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.vit.encoder(x)
        # 去掉 cls
        x = x[:, 1:]
        x = self.feat_proj(x)  # (N, num_patches, d_model)
        return x


class PatchEncoder(nn.Module):
    """与 lightweight 同名结构。"""

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


class ViTAutoregressiveExtractor7D(nn.Module):
    """ViT 版自回归提取器。接口与 lightweight ResNetAutoregressiveExtractor7D 完全一致。"""

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

        self.target_backbone = ViTBackbone(img_size=image_size, d_model=d_model,
                                           pretrained=pretrained)
        self.global_norm = nn.LayerNorm(d_model)

        self.patch_encoder = PatchEncoder(patch_size=patch_size, d_model=d_model)

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
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = StrokeHead(hidden_dim)

    def encode_target(self, target_mask: torch.Tensor):
        tokens = self.target_backbone.forward_features(target_mask)
        global_feat = self.global_norm(tokens.mean(dim=1))
        return tokens, global_feat

    def encode_step(self, target_tokens, target_global, target_mask,
                    state: RolloutState, step_index: torch.Tensor,
                    hidden: torch.Tensor) -> torch.Tensor:
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

        return self.gru(gru_input, hidden)

    def forward(self, *args, **kwargs):
        """nn.Module.forward 调 rollout，使 DDP 能拦截到 gradient reduce。"""
        return self.rollout(*args, **kwargs)

    def rollout(self,
                target_image: torch.Tensor,
                neural_renderer,
                seq_len: int = None,
                gt_strokes: torch.Tensor = None,  # noqa: ARG002 — 保留签名兼容旧 train.py
                scheduled_sampling_prob: float = 0.0,  # noqa: ARG002 — 已废弃
                detach_canvas_for_encoder: bool = True,
                init_state: RolloutState = None,
                init_hidden: torch.Tensor = None,
                init_cursor: torch.Tensor = None) -> dict:
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

        hidden = init_hidden if init_hidden is not None else \
            torch.zeros(N, self.hidden_dim, device=device, dtype=dtype)

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
            hidden = self.encode_step(target_tokens, target_global, target_mask,
                                      state_for_enc, step_index, hidden)
            pred = self.head(hidden)
            pen_logits_list.append(pred['pen_logits'])

            state, info = step_with_renderer(state, pred, neural_renderer,
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

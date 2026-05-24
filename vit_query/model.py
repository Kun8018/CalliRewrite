"""
ViT + Trajectory Queries 架构
从图像直接回归密集点轨迹
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTTinyPatch16X16(nn.Module):
    """极简 ViT-Tiny，兼容 timm 接口风格"""

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=192, depth=12, num_heads=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))  # +1 for cls token

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # We don't need the head for feature extraction
        self.num_classes = 0

    def forward_features(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Return patch features without cls token
        return x[:, 1:, :]  # (B, num_patches, embed_dim)


class ViTTinyBackbone(nn.Module):
    """使用 torchvision 的 ViT-Base 简化到 tiny 规模，或直接用极简实现"""

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=192, pretrained=False):
        super().__init__()

        # 使用我们自己的极简实现
        self.vit = ViTTinyPatch16X16(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=12,
            num_heads=3
        )

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

    def forward_features(self, x):
        return self.vit.forward_features(x)  # (B, num_patches, embed_dim)


class ViTTrajectoryExtractor(nn.Module):
    """
    ViT + Trajectory Queries 架构

    输入: 书法图像 (1, 224, 224)
    输出: 密集点轨迹 (num_points, 2) 归一化坐标 [0,1]
    """

    def __init__(self, img_size=224, num_points=100, embed_dim=192, num_queries=None):
        super().__init__()

        self.img_size = img_size
        self.num_points = num_points
        self.embed_dim = embed_dim

        if num_queries is None:
            num_queries = num_points

        # 1. ViT 骨干网络
        self.vit_backbone = ViTTinyBackbone(
            img_size=img_size,
            patch_size=16,
            in_chans=1,
            embed_dim=embed_dim
        )

        # 2. Trajectory Queries
        self.traj_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.traj_queries, std=0.02)

        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # 4. 查询投影（如果 queries 数量 != 输出点数）
        self.query_proj = None
        if num_queries != num_points:
            self.query_proj = nn.Linear(num_queries, num_points)

        # 5. 回归头
        self.coord_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2)
        )

        # 可选：预测笔状态（用于和二阶段兼容）
        self.pen_head = None  # 可以扩展

    def forward(self, stroke_mask):
        """
        stroke_mask: (B, 1, H, W) 灰度图像 [0,1] 或 [-1,1]
        返回: (B, num_points, 2) 归一化坐标 [0,1]
        """
        B = stroke_mask.shape[0]

        # 1. 提取图像特征
        features = self.vit_backbone.forward_features(stroke_mask)  # (B, num_patches, embed_dim)

        # 2. 准备 queries
        queries = self.traj_queries.expand(B, -1, -1)  # (B, num_queries, embed_dim)

        # 3. Transformer Decoder
        out = self.decoder(tgt=queries, memory=features)  # (B, num_queries, embed_dim)

        # 4. 投影到目标点数（如有需要）
        if self.query_proj is not None:
            out = out.transpose(1, 2)  # (B, embed_dim, num_queries)
            out = self.query_proj(out)  # (B, embed_dim, num_points)
            out = out.transpose(1, 2)  # (B, num_points, embed_dim)

        # 5. 回归坐标
        trajectory_points = torch.sigmoid(self.coord_head(out))  # (B, num_points, 2)

        return trajectory_points


class ViTTrajectoryExtractor7D(nn.Module):
    """
    扩展版本：输出 7 维格式，兼容 seq_extract

    输出: (B, seq_len, 7)
         [pen_state, x1, y1, x2, y2, r, s]
    """

    def __init__(self, img_size=224, seq_len=100, embed_dim=192, num_queries=None):
        super().__init__()

        self.img_size = img_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        if num_queries is None:
            num_queries = seq_len

        # ViT 骨干网络
        self.vit_backbone = ViTTinyBackbone(
            img_size=img_size,
            patch_size=16,
            in_chans=1,
            embed_dim=embed_dim
        )

        # Trajectory Queries
        self.traj_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.traj_queries, std=0.02)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # 查询投影
        self.query_proj = None
        if num_queries != seq_len:
            self.query_proj = nn.Linear(num_queries, seq_len)

        # 输出头
        self.pen_head = nn.Linear(embed_dim, 1)
        self.coord_head = nn.Linear(embed_dim, 4)  # x1, y1, x2, y2
        self.param_head = nn.Linear(embed_dim, 2)  # r, s

    def forward(self, stroke_mask):
        B = stroke_mask.shape[0]

        features = self.vit_backbone.forward_features(stroke_mask)

        queries = self.traj_queries.expand(B, -1, -1)
        out = self.decoder(tgt=queries, memory=features)

        if self.query_proj is not None:
            out = out.transpose(1, 2)
            out = self.query_proj(out)
            out = out.transpose(1, 2)

        pen_logits = self.pen_head(out)
        coords = self.coord_head(out)
        params = self.param_head(out)

        predictions = torch.cat([
            torch.sigmoid(pen_logits),  # pen_state [0,1]
            torch.tanh(coords),         # x1,y1,x2,y2 [-1,1]
            torch.sigmoid(params)       # r,s [0,1]
        ], dim=-1)

        return predictions


class ViTAutoregressiveExtractor7D(nn.Module):
    """自回归版：target image + current canvas/cursor -> next 7D stroke."""

    def __init__(self, img_size=224, seq_len=100, embed_dim=192, hidden_dim=256):
        super().__init__()
        self.img_size = img_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.vit_backbone = ViTTinyBackbone(
            img_size=img_size,
            patch_size=16,
            in_chans=1,
            embed_dim=embed_dim
        )
        self.target_pool = nn.LayerNorm(embed_dim)

        self.canvas_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(embed_dim),
        )

        self.cursor_mlp = nn.Sequential(nn.Linear(2, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.prev_stroke_mlp = nn.Sequential(nn.Linear(7, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.step_mlp = nn.Sequential(nn.Linear(1, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.state_query_norm = nn.LayerNorm(embed_dim)
        self.target_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

        self.gru_input = nn.Sequential(
            nn.Linear(embed_dim * 5, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.pen_head = nn.Linear(hidden_dim, 1)
        self.coord_head = nn.Linear(hidden_dim, 4)
        self.param_head = nn.Linear(hidden_dim, 2)

    def encode_target(self, target_mask):
        tokens = self.vit_backbone.forward_features(target_mask)
        global_feat = self.target_pool(tokens.mean(dim=1))
        return tokens, global_feat

    def encode_step(self, target_tokens, target_global, canvas, cursor, prev_stroke, step_index, hidden=None):
        canvas_feat = self.canvas_encoder(canvas)
        cursor_feat = self.cursor_mlp(cursor)
        prev_feat = self.prev_stroke_mlp(prev_stroke)
        step_feat = self.step_mlp(step_index)
        query = self.state_query_norm(canvas_feat + cursor_feat + prev_feat + step_feat).unsqueeze(1)
        attn_feat, _ = self.target_attn(query, target_tokens, target_tokens)
        attn_feat = attn_feat.squeeze(1)
        gru_input = self.gru_input(torch.cat([
            attn_feat, target_global, canvas_feat, cursor_feat, prev_feat
        ], dim=-1))
        if hidden is None:
            hidden = torch.zeros(canvas.shape[0], self.hidden_dim, device=canvas.device, dtype=canvas.dtype)
        hidden = self.gru(gru_input, hidden)
        return hidden

    def decode_hidden(self, hidden):
        pen_logits = self.pen_head(hidden).squeeze(-1)
        coords = torch.tanh(self.coord_head(hidden))
        params = torch.sigmoid(self.param_head(hidden))
        seq = torch.cat([torch.sigmoid(pen_logits).unsqueeze(-1), coords, params], dim=-1)
        return {'seq': seq, 'pen_logits': pen_logits}

    def forward_step(self, target_tokens, target_global, canvas, cursor, prev_stroke, step_index, hidden=None):
        hidden = self.encode_step(target_tokens, target_global, canvas, cursor, prev_stroke, step_index, hidden)
        output = self.decode_hidden(hidden)
        return output, hidden

    def forward_teacher_forcing(self, target_mask, canvases, cursors, prev_strokes, step_indices):
        target_tokens, target_global = self.encode_target(target_mask)
        hidden = None
        seq_outputs = []
        pen_logits_outputs = []
        for i in range(canvases.shape[1]):
            output, hidden = self.forward_step(
                target_tokens,
                target_global,
                canvases[:, i],
                cursors[:, i],
                prev_strokes[:, i],
                step_indices[:, i],
                hidden
            )
            seq_outputs.append(output['seq'])
            pen_logits_outputs.append(output['pen_logits'])
        return {
            'seq': torch.stack(seq_outputs, dim=1),
            'pen_logits': torch.stack(pen_logits_outputs, dim=1),
        }


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.2f}M, Trainable: {trainable/1e6:.2f}M"


if __name__ == "__main__":
    print("Testing ViT Trajectory Extractor (2D points)...")
    model_2d = ViTTrajectoryExtractor(img_size=224, num_points=100, embed_dim=192)
    print(f"Model 2D: {count_parameters(model_2d)}")

    x = torch.randn(2, 1, 224, 224)
    out = model_2d(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    print("✓ 2D model OK")

    print("\nTesting ViT Trajectory Extractor (7D seq)...")
    model_7d = ViTTrajectoryExtractor7D(img_size=224, seq_len=100, embed_dim=192)
    print(f"Model 7D: {count_parameters(model_7d)}")

    out = model_7d(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Pen state range: [{out[..., 0].min():.3f}, {out[..., 0].max():.3f}]")
    print(f"Coords range: [{out[..., 1:5].min():.3f}, {out[..., 1:5].max():.3f}]")
    print(f"Params range: [{out[..., 5:7].min():.3f}, {out[..., 5:7].max():.3f}]")
    print("✓ 7D model OK")

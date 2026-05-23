"""
ViT + 彩色标注笔画
输入: RGB 图片 (红色标注当前笔画)
输出: 7D 笔画序列
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTTinyPatch16X16RGB(nn.Module):
    """极简 ViT-Tiny，支持 RGB 输入"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192,
                 depth=12, num_heads=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)

        return x[:, 1:, :]


class ViTTinyDualChannel(nn.Module):
    """双输入: 灰度图 + 红色mask"""

    def __init__(self, img_size=224, patch_size=16, embed_dim=192,
                 depth=12, num_heads=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 两个分支
        self.gray_patch_embed = nn.Conv2d(1, embed_dim // 2, kernel_size=patch_size, stride=patch_size)
        self.mask_patch_embed = nn.Conv2d(1, embed_dim // 2, kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward_features(self, gray_img, red_mask):
        """
        gray_img: (B, 1, 224, 224)
        red_mask: (B, 1, 224, 224)
        """
        B = gray_img.shape[0]

        # 分别编码
        gray_feat = self.gray_patch_embed(gray_img)
        mask_feat = self.mask_patch_embed(red_mask)

        # 拼接
        x = torch.cat([gray_feat, mask_feat], dim=1)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)

        return x[:, 1:, :]


class ViTColorTrajectoryExtractor7D(nn.Module):
    """
    ViT + 彩色标注笔画 (RGB 输入版本)

    输入: RGB 图片，红色标注当前笔画
    输出: 7D 笔画序列
    """

    def __init__(self, img_size=224, seq_len=100, embed_dim=192, num_queries=None):
        super().__init__()

        self.img_size = img_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        if num_queries is None:
            num_queries = seq_len

        # RGB ViT 编码器
        self.vit_backbone = ViTTinyPatch16X16RGB(
            img_size=img_size,
            patch_size=16,
            in_chans=3,
            embed_dim=embed_dim,
            depth=12,
            num_heads=3
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

        self.query_proj = None
        if num_queries != seq_len:
            self.query_proj = nn.Linear(num_queries, seq_len)

        # 输出头
        self.pen_head = nn.Linear(embed_dim, 1)
        self.coord_head = nn.Linear(embed_dim, 4)
        self.param_head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        """
        x: (B, 3, 224, 224) - RGB 图片
        返回: (B, seq_len, 7)
        """
        B = x.shape[0]

        features = self.vit_backbone.forward_features(x)
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
            torch.sigmoid(pen_logits),
            torch.tanh(coords),
            torch.sigmoid(params)
        ], dim=-1)

        return predictions


class ViTDualTrajectoryExtractor7D(nn.Module):
    """
    ViT + 双输入版本 (灰度 + 红色mask)

    输入: (gray_img, red_mask)
    输出: 7D 笔画序列
    """

    def __init__(self, img_size=224, seq_len=100, embed_dim=192, num_queries=None):
        super().__init__()

        self.img_size = img_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        if num_queries is None:
            num_queries = seq_len

        # 双输入 ViT 编码器
        self.vit_backbone = ViTTinyDualChannel(
            img_size=img_size,
            patch_size=16,
            embed_dim=embed_dim,
            depth=12,
            num_heads=3
        )

        # Trajectory Queries
        self.traj_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.traj_queries, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.query_proj = None
        if num_queries != seq_len:
            self.query_proj = nn.Linear(num_queries, seq_len)

        self.pen_head = nn.Linear(embed_dim, 1)
        self.coord_head = nn.Linear(embed_dim, 4)
        self.param_head = nn.Linear(embed_dim, 2)

    def forward(self, gray_img, red_mask):
        """
        gray_img: (B, 1, 224, 224)
        red_mask: (B, 1, 224, 224)
        """
        B = gray_img.shape[0]

        features = self.vit_backbone.forward_features(gray_img, red_mask)
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
            torch.sigmoid(pen_logits),
            torch.tanh(coords),
            torch.sigmoid(params)
        ], dim=-1)

        return predictions


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.2f}M, Trainable: {trainable/1e6:.2f}M"


if __name__ == "__main__":
    print("Testing ViT + Color Stroke models...")

    # 测试 RGB 版本
    print("\n" + "="*60)
    print("Test 1: RGB Input Version")
    print("="*60)
    model_rgb = ViTColorTrajectoryExtractor7D(
        img_size=224, seq_len=100, embed_dim=192
    )
    print(f"Model RGB: {count_parameters(model_rgb)}")

    x_rgb = torch.randn(2, 3, 224, 224)
    print(f"Input: {x_rgb.shape}")
    out_rgb = model_rgb(x_rgb)
    print(f"Output: {out_rgb.shape}")
    print("✓ RGB version OK!")

    # 测试双输入版本
    print("\n" + "="*60)
    print("Test 2: Dual Input Version (Gray + Red Mask)")
    print("="*60)
    model_dual = ViTDualTrajectoryExtractor7D(
        img_size=224, seq_len=100, embed_dim=192
    )
    print(f"Model Dual: {count_parameters(model_dual)}")

    gray = torch.randn(2, 1, 224, 224)
    mask = torch.randn(2, 1, 224, 224)
    print(f"Input gray: {gray.shape}, mask: {mask.shape}")
    out_dual = model_dual(gray, mask)
    print(f"Output: {out_dual.shape}")
    print("✓ Dual version OK!")

    print("\n" + "="*60)
    print("✓ All models OK!")
    print("="*60)

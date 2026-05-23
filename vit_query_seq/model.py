"""
ViT + Trajectory Queries (支持多图输入: 渐进式笔画)
输入: 图片序列 (num_strokes, 1, 224, 224)
输出: 完整笔画序列 (seq_len, 7)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTTinyPatch16X16(nn.Module):
    """极简 ViT-Tiny，单图处理"""

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=192, depth=12, num_heads=3):
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

        return x[:, 1:, :]  # (B, num_patches, embed_dim)


class MultiImageViTEncoder(nn.Module):
    """多图编码器: 处理渐进式笔画序列"""

    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=192, num_images=10, depth=6, num_heads=4):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_images = num_images

        # 单图 ViT 编码器
        self.single_vit = ViTTinyPatch16X16(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        )

        self.num_patches = self.single_vit.num_patches

        # 图像序列的位置编码
        self.image_pos_embed = nn.Parameter(torch.randn(1, num_images, embed_dim))

        # 可选: 跨图像注意力层
        cross_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.cross_image_transformer = nn.TransformerEncoder(cross_layer, num_layers=2)

    def forward(self, x):
        """
        x: (B, num_images, 1, 224, 224)
        返回: (B, num_images * num_patches, embed_dim)
        """
        B, num_img, C, H, W = x.shape

        # 展平 batch 和 image 维度
        x_flat = x.reshape(B * num_img, C, H, W)

        # 每张图单独编码
        features_flat = self.single_vit.forward_features(x_flat)  # (B*num_img, num_patches, embed_dim)

        # 恢复维度
        features = features_flat.reshape(B, num_img, self.num_patches, self.embed_dim)

        # 添加图像位置编码
        features = features + self.image_pos_embed.unsqueeze(2)

        # 展平 patch 维度
        features = features.reshape(B, num_img * self.num_patches, self.embed_dim)

        # 跨图像注意力
        features = self.cross_image_transformer(features)

        return features


class ViTSeqTrajectoryExtractor7D(nn.Module):
    """
    ViT + Trajectory Queries (多图输入版本)

    输入: 图片序列 (num_images, 1, 224, 224)
    输出: 7D 笔画序列 (seq_len, 7)
    """

    def __init__(self, img_size=224, num_images=10, seq_len=100,
                 embed_dim=192, num_queries=None):
        super().__init__()

        self.img_size = img_size
        self.num_images = num_images
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        if num_queries is None:
            num_queries = seq_len

        # 多图 ViT 编码器
        self.vit_encoder = MultiImageViTEncoder(
            img_size=img_size,
            patch_size=16,
            in_chans=1,
            embed_dim=embed_dim,
            num_images=num_images,
            depth=6,
            num_heads=4
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

    def forward(self, x):
        """
        x: (B, num_images, 1, 224, 224)
        返回: (B, seq_len, 7)
        """
        B = x.shape[0]

        # 编码图片序列
        features = self.vit_encoder(x)  # (B, num_images * num_patches, embed_dim)

        # 准备 queries
        queries = self.traj_queries.expand(B, -1, -1)  # (B, num_queries, embed_dim)

        # Decoder
        out = self.decoder(tgt=queries, memory=features)

        # 投影到目标长度
        if self.query_proj is not None:
            out = out.transpose(1, 2)
            out = self.query_proj(out)
            out = out.transpose(1, 2)

        # 预测输出
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
    print("Testing ViT + Sequential Images model...")

    model = ViTSeqTrajectoryExtractor7D(
        img_size=224,
        num_images=10,
        seq_len=100,
        embed_dim=192
    )

    print(f"\nModel: {count_parameters(model)}")

    # 测试输入: (batch, num_images, 1, 224, 224)
    x = torch.randn(2, 10, 1, 224, 224)
    print(f"\nInput shape: {x.shape}")

    out = model(x)
    print(f"Output shape: {out.shape}")

    pen_state = out[..., 0]
    coords = out[..., 1:5]
    params = out[..., 5:7]

    print(f"\nPen state range: [{pen_state.min():.3f}, {pen_state.max():.3f}]")
    print(f"Coords range: [{coords.min():.3f}, {coords.max():.3f}]")
    print(f"Params range: [{params.min():.3f}, {params.max():.3f}]")

    print("\n✓ Model OK!")

#!/usr/bin/env python3
"""
测试 phase2 训练流程：NeuralRenderer + VGG Perceptual Loss
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# 确保能导入 vit_query 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ViTTrajectoryExtractor7D
from neural_renderer import NeuralRasterizorStep, seq7_to_absolute
from vgg_loss import VGG16PerceptualLoss
from dataset import QuickDrawCleanDatasetViT


def test_neural_renderer():
    """测试 NeuralRenderer 基本功能"""
    print("=" * 60)
    print("测试 NeuralRenderer")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 128  # 小一点测试更快

    # 创建 renderer
    renderer = NeuralRasterizorStep(img_size).to(device)

    # 创建一些测试笔画：(batch, seq_len, 8) - [x0, y0, x1, y1, x2, y2, r0, r2]
    batch_size = 2
    seq_len = 5
    strokes_abs = torch.rand(batch_size, seq_len, 8, device=device) * 0.8 + 0.1  # 在中间区域

    print(f"输入笔画形状: {strokes_abs.shape}")

    # 渲染
    with torch.no_grad():
        rendered = renderer(strokes_abs)

    print(f"渲染输出形状: {rendered.shape}")
    print(f"渲染值范围: [{rendered.min():.3f}, {rendered.max():.3f}]")
    print("✓ NeuralRenderer 基本测试通过\n")
    return True


def test_seq7_conversion():
    """测试 seq7 到绝对坐标的转换"""
    print("=" * 60)
    print("测试 seq7 坐标转换")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 128

    # 创建一些 seq7 格式的笔画
    batch_size = 2
    seq_len = 5
    strokes_seq7 = torch.zeros(batch_size, seq_len, 7, device=device)

    # 第一笔：pen=0（绘制）
    strokes_seq7[:, 0, 0] = 0.0
    strokes_seq7[:, 0, 1] = 0.5  # dx1
    strokes_seq7[:, 0, 2] = 0.0  # dy1
    strokes_seq7[:, 0, 3] = 1.0  # dx2
    strokes_seq7[:, 0, 4] = 0.0  # dy2
    strokes_seq7[:, 0, 5] = 0.1  # r
    strokes_seq7[:, 0, 6] = 1.0  # s

    # 第二笔：pen=1（不绘制）
    strokes_seq7[:, 1, 0] = 1.0

    print(f"seq7 输入形状: {strokes_seq7.shape}")

    # 转换
    strokes_abs = seq7_to_absolute(strokes_seq7, img_size)

    print(f"转换后形状: {strokes_abs.shape}")
    print(f"绝对坐标范围: [{strokes_abs.min():.3f}, {strokes_abs.max():.3f}]")
    print("✓ seq7 转换测试通过\n")
    return True


def test_unsupervised_loss():
    """测试无监督损失函数"""
    print("=" * 60)
    print("测试无监督损失函数")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 128

    # 创建模型
    model = ViTTrajectoryExtractor7D(
        img_size=img_size,
        seq_len=20,
        embed_dim=128
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建损失函数
    from train import UnsupervisedLoss
    criterion = UnsupervisedLoss(img_size=img_size).to(device)

    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 创建一些测试图像
    batch_size = 2
    # 简单的目标图像：中间有个方块
    target_images = torch.ones(batch_size, img_size, img_size, device=device)
    s = img_size // 4
    target_images[:, s:-s, s:-s] = 0.0  # [0.0-stroke, 1.0-BG]

    print(f"目标图像形状: {target_images.shape}")

    # 一步训练
    model.train()
    optimizer.zero_grad()

    # 前向传播
    strokes_seq7 = model(target_images.unsqueeze(1))  # 需要 channel 维度

    print(f"模型输出形状: {strokes_seq7.shape}")

    # 计算损失
    losses = criterion(strokes_seq7, target_images)

    print(f"总损失: {losses['total'].item():.4f}")
    print(f"  渲染损失: {losses['render'].item():.4f}")
    print(f"  感知损失: {losses['perceptual'].item():.4f}")

    # 反向传播
    losses['total'].backward()

    # 检查梯度
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5

    print(f"梯度范数: {grad_norm:.4f}")

    optimizer.step()

    print("✓ 无监督损失和梯度回流测试通过\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("Phase2 集成测试")
    print("=" * 60 + "\n")

    try:
        test_neural_renderer()
        test_seq7_conversion()
        test_unsupervised_loss()

        print("=" * 60)
        print("所有测试通过！ ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

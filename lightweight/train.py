#!/usr/bin/env python3
"""
ResNet-18 + Transformer 训练脚本
使用 QuickDraw 数据集训练轻量模型
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from model import StrokeTransformer, count_parameters
from dataset import StrokeDataset, QuickDrawConverter, create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Train Stroke Transformer')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录，包含 npz 文件')
    parser.add_argument('--quickdraw_npz', type=str, default=None,
                        help='QuickDraw npz 文件路径')
    parser.add_argument('--quickdraw_save_dir', type=str, default='./qd_data',
                        help='QuickDraw 转换后的保存目录')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--max_seq_len', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                        help='Teacher forcing 概率')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--use_wandb', action='store_true',
                        help='使用 wandb 记录')
    parser.add_argument('--wandb_project', type=str, default='stroke-transformer')
    parser.add_argument('--save_every', type=int, default=10,
                        help='每多少 epoch 保存一次')
    return parser.parse_args()


class StrokeLoss(nn.Module):
    """
    组合损失函数
    - pen_state: 二分类损失
    - coordinates: L1 损失
    - parameters: L1 损失
    """
    def __init__(self, weight_pen=1.0, weight_coord=1.0, weight_param=1.0):
        super().__init__()
        self.weight_pen = weight_pen
        self.weight_coord = weight_coord
        self.weight_param = weight_param

        self.bce_loss = nn.BCELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')

    def forward(self, predictions, targets, mask=None):
        """
        predictions: (batch, seq_len, 7)
        targets: (batch, seq_len, 7)
        mask: (batch, seq_len) - 有效性掩码
        """
        if mask is None:
            mask = torch.ones_like(targets[..., 0])

        # Pen state 损失
        pen_pred = predictions[..., 0]
        pen_target = targets[..., 0]
        pen_loss = self.bce_loss(pen_pred, pen_target)
        pen_loss = (pen_loss * mask).sum() / mask.sum()

        # Coordinates 损失 (x1, y1, x2, y2)
        coord_pred = predictions[..., 1:5]
        coord_target = targets[..., 1:5]
        coord_loss = self.l1_loss(coord_pred, coord_target)
        coord_loss = (coord_loss.mean(dim=-1) * mask).sum() / mask.sum()

        # Parameters 损失 (r, s)
        param_pred = predictions[..., 5:7]
        param_target = targets[..., 5:7]
        param_loss = self.l1_loss(param_pred, param_target)
        param_loss = (param_loss.mean(dim=-1) * mask).sum() / mask.sum()

        total_loss = (
            self.weight_pen * pen_loss +
            self.weight_coord * coord_loss +
            self.weight_param * param_loss
        )

        return {
            'total': total_loss,
            'pen': pen_loss,
            'coord': coord_loss,
            'param': param_loss
        }


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, teacher_forcing_ratio=0.5):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {'pen': 0.0, 'coord': 0.0, 'param': 0.0}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        images = batch['image'].to(device)
        strokes = batch['strokes'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()

        # 前向传播
        predictions = model(images, strokes, teacher_forcing_ratio=teacher_forcing_ratio)

        # 计算损失
        losses = criterion(predictions, strokes, mask)
        loss = losses['total']

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += losses[k].item()

        pbar.set_postfix({'loss': loss.item()})

    # 平均损失
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for k in loss_components:
        loss_components[k] /= num_batches

    return avg_loss, loss_components


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    """验证"""
    model.eval()
    total_loss = 0.0
    loss_components = {'pen': 0.0, 'coord': 0.0, 'param': 0.0}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    for batch in pbar:
        images = batch['image'].to(device)
        strokes = batch['strokes'].to(device)
        mask = batch['mask'].to(device)

        # 前向传播 (不使用 teacher forcing)
        predictions = model(images, strokes, teacher_forcing_ratio=0.0)

        # 计算损失
        losses = criterion(predictions, strokes, mask)

        # 统计
        total_loss += losses['total'].item()
        for k in loss_components:
            loss_components[k] += losses[k].item()

        pbar.set_postfix({'loss': losses['total'].item()})

    # 平均损失
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for k in loss_components:
        loss_components[k] /= num_batches

    return avg_loss, loss_components


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f'Checkpoint saved to {save_path}')


def load_quickdraw_data(quickdraw_npz, save_dir, image_size=256, max_items=10000):
    """加载并转换 QuickDraw 数据"""
    print(f'Loading QuickDraw data from {quickdraw_npz}...')

    # 加载 QuickDraw 原始数据
    sketches = QuickDrawConverter.load_quickdraw_npz(quickdraw_npz, max_items=max_items)

    # 转换并保存
    os.makedirs(save_dir, exist_ok=True)
    pairs = QuickDrawConverter.create_pairs(sketches, save_dir=save_dir, image_size=image_size)

    print(f'Created {len(pairs)} image-stroke pairs')
    return save_dir


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设备
    device = torch.device(args.device)
    print(f'Using device: {device}')

    # 初始化 wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=vars(args))
    elif args.use_wandb and not WANDB_AVAILABLE:
        print('Warning: wandb not installed, continuing without logging')
        args.use_wandb = False

    # 准备数据
    if args.quickdraw_npz is not None:
        # 使用 QuickDraw 数据
        data_dir = load_quickdraw_data(
            args.quickdraw_npz,
            args.quickdraw_save_dir,
            image_size=args.image_size,
            max_items=50000  # 可以调整
        )
        args.data_dir = data_dir

    if args.data_dir is None:
        raise ValueError('需要指定 --data_dir 或 --quickdraw_npz')

    # 创建数据集
    full_dataset = StrokeDataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        max_seq_len=args.max_seq_len
    )

    if len(full_dataset) == 0:
        raise ValueError('未找到数据，请检查 data_dir')

    # 划分训练/验证集
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f'Train dataset: {len(train_dataset)} samples')
    print(f'Val dataset: {len(val_dataset)} samples')

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 创建模型
    model = StrokeTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        max_seq_len=args.max_seq_len
    ).to(device)

    print(f'Model created: {count_parameters(model)}')

    # 损失函数和优化器
    criterion = StrokeLoss(weight_pen=1.0, weight_coord=5.0, weight_param=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    best_val_loss = float('inf')

    print('Starting training...')
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_comp = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            teacher_forcing_ratio=args.teacher_forcing_ratio
        )

        # 验证
        val_loss, val_comp = validate(
            model, val_loader, criterion, device, epoch
        )

        # 打印
        print(f'\nEpoch {epoch}/{args.epochs}')
        print(f'  Train Loss: {train_loss:.4f} (pen={train_comp["pen"]:.4f}, coord={train_comp["coord"]:.4f}, param={train_comp["param"]:.4f})')
        print(f'  Val Loss: {val_loss:.4f} (pen={val_comp["pen"]:.4f}, coord={val_comp["coord"]:.4f}, param={val_comp["param"]:.4f})')

        # 记录到 wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/pen_loss': train_comp['pen'],
                'train/coord_loss': train_comp['coord'],
                'train/param_loss': train_comp['param'],
                'val/loss': val_loss,
                'val/pen_loss': val_comp['pen'],
                'val/coord_loss': val_comp['coord'],
                'val/param_loss': val_comp['param'],
            })

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, 'model_best.pth')
            )
            print(f'  New best val loss: {best_val_loss:.4f}')

        # 定期保存
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, f'model_epoch_{epoch}.pth')
            )

    # 保存最终模型
    save_checkpoint(
        model, optimizer, args.epochs, val_loss,
        os.path.join(args.output_dir, 'model_final.pth')
    )

    print('\nTraining complete!')
    print(f'Best val loss: {best_val_loss:.4f}')

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

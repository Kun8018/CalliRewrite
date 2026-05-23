#!/usr/bin/env python3
"""
ViT + 彩色标注笔画 训练脚本
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

from model import ViTColorTrajectoryExtractor7D, ViTDualTrajectoryExtractor7D, count_parameters
from dataset import StrokeColorDatasetViT, MultiCharColorDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TrajectoryLoss7D(nn.Module):
    """7D 序列损失"""

    def __init__(self, weight_pen=1.0, weight_coord=5.0, weight_param=1.0):
        super().__init__()
        self.weight_pen = weight_pen
        self.weight_coord = weight_coord
        self.weight_param = weight_param

        self.bce_loss = nn.BCELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')

    def forward(self, predictions, targets, mask=None):
        if mask is None:
            mask = torch.ones_like(targets[..., 0])

        pen_pred = predictions[..., 0]
        pen_target = targets[..., 0]
        pen_loss = self.bce_loss(pen_pred, pen_target)
        pen_loss = (pen_loss * mask).sum() / mask.sum()

        coord_pred = predictions[..., 1:5]
        coord_target = targets[..., 1:5]
        coord_loss = self.l1_loss(coord_pred, coord_target)
        coord_loss = (coord_loss.mean(dim=-1) * mask).sum() / mask.sum()

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


def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT Color Stroke')

    parser.add_argument('--data_dir', type=str, default=None, help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--mode', type=str, default='rgb', choices=['rgb', 'dual'],
                        help='rgb=RGB输入, dual=灰度+mask输入')
    parser.add_argument('--multi_char', action='store_true', help='多字符数据集格式')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--use_wandb', action='store_true', help='使用 wandb')
    parser.add_argument('--wandb_project', type=str, default='vit_color_stroke')
    parser.add_argument('--save_every', type=int, default=10)

    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, mode='rgb'):
    model.train()
    total_loss = 0.0
    loss_components = {}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        if mode == 'rgb':
            images = batch['image'].to(device)
        else:
            gray = batch['gray_image'].to(device)
            mask = batch['red_mask'].to(device)

        targets = batch['strokes'].to(device)
        mask_loss = batch['mask'].to(device)

        optimizer.zero_grad()

        if mode == 'rgb':
            predictions = model(images)
        else:
            predictions = model(gray, mask)

        losses = criterion(predictions, targets, mask_loss)
        loss = losses['total']
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k, v in losses.items():
            if k != 'total':
                loss_components[k] = loss_components.get(k, 0.0) + v.item()

        pbar.set_postfix({'loss': loss.item()})

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for k in loss_components:
        loss_components[k] /= num_batches

    return avg_loss, loss_components


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, mode='rgb'):
    model.eval()
    total_loss = 0.0
    loss_components = {}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    for batch in pbar:
        if mode == 'rgb':
            images = batch['image'].to(device)
        else:
            gray = batch['gray_image'].to(device)
            mask = batch['red_mask'].to(device)

        targets = batch['strokes'].to(device)
        mask_loss = batch['mask'].to(device)

        if mode == 'rgb':
            predictions = model(images)
        else:
            predictions = model(gray, mask)

        losses = criterion(predictions, targets, mask_loss)
        total_loss += losses['total'].item()
        for k, v in losses.items():
            if k != 'total':
                loss_components[k] = loss_components.get(k, 0.0) + v.item()

        pbar.set_postfix({'loss': losses['total'].item()})

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for k in loss_components:
        loss_components[k] /= num_batches

    return avg_loss, loss_components


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f'Checkpoint saved to {save_path}')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f'Using device: {device}')

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=vars(args))
    elif args.use_wandb and not WANDB_AVAILABLE:
        print('Warning: wandb not installed')
        args.use_wandb = False

    if args.data_dir is None:
        raise ValueError('Need --data_dir')

    # 数据集
    if args.multi_char:
        full_dataset = MultiCharColorDataset(
            data_dir=args.data_dir,
            img_size=args.img_size,
            seq_len=args.seq_len,
            mode=args.mode
        )
    else:
        full_dataset = StrokeColorDatasetViT(
            data_dir=args.data_dir,
            img_size=args.img_size,
            seq_len=args.seq_len,
            mode=args.mode
        )

    if len(full_dataset) == 0:
        raise ValueError('No data found!')

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # 模型
    if args.mode == 'rgb':
        model = ViTColorTrajectoryExtractor7D(
            img_size=args.img_size,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim
        ).to(device)
    else:
        model = ViTDualTrajectoryExtractor7D(
            img_size=args.img_size,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim
        ).to(device)

    print(f'Model: {count_parameters(model)}')

    criterion = TrajectoryLoss7D()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')

    print('Starting training...')
    for epoch in range(1, args.epochs + 1):
        train_loss, train_comp = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args.mode)
        val_loss, val_comp = validate(model, val_loader, criterion, device, epoch, args.mode)

        scheduler.step()

        print(f'\nEpoch {epoch}/{args.epochs}')
        print(f'  Train: {train_loss:.4f} {train_comp}')
        print(f'  Val: {val_loss:.4f} {val_comp}')

        if args.use_wandb and WANDB_AVAILABLE:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
            }
            for k, v in train_comp.items():
                log_dict[f'train/{k}'] = v
            for k, v in val_comp.items():
                log_dict[f'val/{k}'] = v
            wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                          os.path.join(args.output_dir, 'model_best.pth'))
            print(f'  New best!')

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss,
                          os.path.join(args.output_dir, f'model_epoch_{epoch}.pth'))

    save_checkpoint(model, optimizer, args.epochs, val_loss,
                  os.path.join(args.output_dir, 'model_final.pth'))

    print(f'\nDone! Best val loss: {best_val_loss:.4f}')

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
ViT + Trajectory Queries 训练脚本
支持 seq_extract 风格两阶段：
phase1: QuickDraw-clean 预训练
phase2: 书法监督数据 fine-tune（.png + .npz）
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

from model import ViTTrajectoryExtractor, ViTTrajectoryExtractor7D, ViTAutoregressiveExtractor7D, count_parameters
from dataset import StrokeDatasetViT, QuickDrawCleanDatasetViT

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TrajectoryLoss2D(nn.Module):
    """2D 密集点回归损失"""

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_points, target_points):
        l1 = self.l1_loss(pred_points, target_points)
        mse = self.mse_loss(pred_points, target_points)
        total_loss = l1 + 0.1 * mse
        return {'total': total_loss, 'l1': l1, 'mse': mse}


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

        mask_sum = mask.sum().clamp_min(1.0)
        pen_loss = self.bce_loss(predictions[..., 0], targets[..., 0])
        pen_loss = (pen_loss * mask).sum() / mask_sum

        coord_loss = self.l1_loss(predictions[..., 1:5], targets[..., 1:5])
        coord_loss = (coord_loss.mean(dim=-1) * mask).sum() / mask_sum

        param_loss = self.l1_loss(predictions[..., 5:7], targets[..., 5:7])
        param_loss = (param_loss.mean(dim=-1) * mask).sum() / mask_sum

        total_loss = (
            self.weight_pen * pen_loss +
            self.weight_coord * coord_loss +
            self.weight_param * param_loss
        )
        return {'total': total_loss, 'pen': pen_loss, 'coord': coord_loss, 'param': param_loss}


class AutoregressiveTrajectoryLoss7D(nn.Module):
    def __init__(self, weight_pen=1.0, weight_coord=5.0, weight_param=1.0):
        super().__init__()
        self.weight_pen = weight_pen
        self.weight_coord = weight_coord
        self.weight_param = weight_param
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')

    def forward(self, outputs, targets, mask=None):
        predictions = outputs['seq']
        pen_logits = outputs['pen_logits']
        if mask is None:
            mask = torch.ones_like(targets[..., 0])
        mask_sum = mask.sum().clamp_min(1.0)

        pen_loss = self.bce_loss(pen_logits, targets[..., 0])
        pen_loss = (pen_loss * mask).sum() / mask_sum

        coord_loss = self.l1_loss(predictions[..., 1:5], targets[..., 1:5])
        coord_loss = (coord_loss.mean(dim=-1) * mask).sum() / mask_sum

        param_loss = self.l1_loss(predictions[..., 5:7], targets[..., 5:7])
        param_loss = (param_loss.mean(dim=-1) * mask).sum() / mask_sum

        total_loss = (
            self.weight_pen * pen_loss +
            self.weight_coord * coord_loss +
            self.weight_param * param_loss
        )
        return {'total': total_loss, 'pen': pen_loss, 'coord': coord_loss, 'param': param_loss}


def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT Trajectory Extractor')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='1=QuickDraw-clean 预训练, 2=书法数据 fine-tune')
    parser.add_argument('--dataset_root', type=str, default='../seq_extract/datasets',
                        help='seq_extract 数据根目录，phase1 默认读取 QuickDraw-clean')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='phase2 监督数据目录，要求 .png/.jpg + 同名 .npz')
    parser.add_argument('--phase1_checkpoint', type=str, default=None,
                        help='phase2 从 phase1 checkpoint 初始化')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_points', type=int, default=100, help='2D 模式的点数')
    parser.add_argument('--seq_len', type=int, default=100, help='7D 模式的序列长度')
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--mode', type=str, default='seq7', choices=['seq7', 'points'],
                        help='输出模式: seq7 或 points；两阶段建议用 seq7')
    parser.add_argument('--arch', type=str, default='autoregressive', choices=['autoregressive', 'oneshot'],
                        help='autoregressive=canvas/cursor 自回归主路径；oneshot=旧的一次性序列预测')
    parser.add_argument('--chunk_len', type=int, default=8, help='autoregressive teacher-forcing chunk 长度')
    parser.add_argument('--chunks_per_sample', type=int, default=4, help='每个样本每轮 epoch 的随机 chunk 数')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_items_per_category', type=int, default=None,
                        help='phase1 调试时限制每个 QuickDraw 类别样本数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--use_wandb', action='store_true', help='使用 wandb')
    parser.add_argument('--wandb_project', type=str, default='vit-query-stroke')
    parser.add_argument('--save_every', type=int, default=10)
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    loss_components = {}
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch in pbar:
        optimizer.zero_grad()

        if model.__class__.__name__ == 'ViTAutoregressiveExtractor7D':
            outputs = model.forward_teacher_forcing(
                batch['target_mask'].to(device),
                batch['canvases'].to(device),
                batch['cursors'].to(device),
                batch['prev_strokes'].to(device),
                batch['step_indices'].to(device)
            )
            targets = batch['strokes'].to(device)
            mask = batch['mask'].to(device)
            losses = criterion(outputs, targets, mask)
        elif model.__class__.__name__ == 'ViTTrajectoryExtractor7D':
            images = batch['image'].to(device)
            targets = batch['strokes'].to(device)
            mask = batch['mask'].to(device)
            predictions = model(images)
            losses = criterion(predictions, targets, mask)
        else:
            images = batch['image'].to(device)
            targets = batch['points'].to(device)
            predictions = model(images)
            losses = criterion(predictions, targets)

        loss = losses['total']
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k, v in losses.items():
            if k != 'total':
                loss_components[k] = loss_components.get(k, 0.0) + v.item()
        pbar.set_postfix({'loss': loss.item()})

    num_batches = max(len(dataloader), 1)
    for k in loss_components:
        loss_components[k] /= num_batches
    return total_loss / num_batches, loss_components


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    loss_components = {}
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')

    for batch in pbar:
        if model.__class__.__name__ == 'ViTAutoregressiveExtractor7D':
            outputs = model.forward_teacher_forcing(
                batch['target_mask'].to(device),
                batch['canvases'].to(device),
                batch['cursors'].to(device),
                batch['prev_strokes'].to(device),
                batch['step_indices'].to(device)
            )
            targets = batch['strokes'].to(device)
            mask = batch['mask'].to(device)
            losses = criterion(outputs, targets, mask)
        elif model.__class__.__name__ == 'ViTTrajectoryExtractor7D':
            images = batch['image'].to(device)
            targets = batch['strokes'].to(device)
            mask = batch['mask'].to(device)
            predictions = model(images)
            losses = criterion(predictions, targets, mask)
        else:
            images = batch['image'].to(device)
            targets = batch['points'].to(device)
            predictions = model(images)
            losses = criterion(predictions, targets)

        total_loss += losses['total'].item()
        for k, v in losses.items():
            if k != 'total':
                loss_components[k] = loss_components.get(k, 0.0) + v.item()
        pbar.set_postfix({'loss': losses['total'].item()})

    num_batches = max(len(dataloader), 1)
    for k in loss_components:
        loss_components[k] /= num_batches
    return total_loss / num_batches, loss_components


def save_checkpoint(model, optimizer, epoch, loss, save_path, args):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'phase': args.phase,
        'args': vars(args),
    }, save_path)
    print(f'Checkpoint saved to {save_path}')


def load_phase1_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint['model_state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Loaded phase1 checkpoint: {checkpoint_path}')
    if missing:
        print(f'  Missing keys: {len(missing)}')
    if unexpected:
        print(f'  Unexpected keys: {len(unexpected)}')


def build_dataset(args):
    if args.phase == 1:
        if args.mode != 'seq7':
            raise ValueError('phase1 currently supports --mode seq7')
        train_dataset = QuickDrawCleanDatasetViT(
            dataset_root=args.dataset_root,
            split='train',
            img_size=args.img_size,
            seq_len=args.seq_len,
            max_items_per_category=args.max_items_per_category,
            arch=args.arch,
            chunk_len=args.chunk_len,
            chunks_per_sample=args.chunks_per_sample
        )
        val_dataset = QuickDrawCleanDatasetViT(
            dataset_root=args.dataset_root,
            split='test',
            img_size=args.img_size,
            seq_len=args.seq_len,
            max_items_per_category=args.max_items_per_category,
            arch=args.arch,
            chunk_len=args.chunk_len,
            chunks_per_sample=1
        )
        return train_dataset, val_dataset

    if args.data_dir is None:
        possible_dirs = [
            '../seq_extract/outputs/__new_train_phase_2',
            '../rl_finetune/data/train_data',
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                args.data_dir = d
                print(f'Using phase2 data dir: {d}')
                break

    if args.data_dir is None:
        raise ValueError('phase2 needs --data_dir with .png/.jpg + same-name .npz labels')

    full_dataset = StrokeDatasetViT(
        data_dir=args.data_dir,
        img_size=args.img_size,
        num_points=args.num_points,
        seq_len=args.seq_len,
        mode=args.mode,
        arch=args.arch,
        chunk_len=args.chunk_len,
        chunks_per_sample=args.chunks_per_sample
    )
    if len(full_dataset) == 0:
        raise ValueError('No data found!')

    val_size = max(1, int(len(full_dataset) * args.val_split)) if len(full_dataset) > 1 else 0
    train_size = len(full_dataset) - val_size
    if val_size == 0:
        return full_dataset, full_dataset
    return random_split(full_dataset, [train_size, val_size])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f'Using device: {device}')
    print(f'Phase: {args.phase}')

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=vars(args))
    elif args.use_wandb and not WANDB_AVAILABLE:
        print('Warning: wandb not installed')
        args.use_wandb = False

    train_dataset, val_dataset = build_dataset(args)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError('No data found!')
    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    if args.arch == 'autoregressive':
        if args.mode != 'seq7':
            raise ValueError('autoregressive currently supports --mode seq7')
        model = ViTAutoregressiveExtractor7D(
            img_size=args.img_size,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim
        ).to(device)
        criterion = AutoregressiveTrajectoryLoss7D()
    elif args.mode == 'seq7':
        model = ViTTrajectoryExtractor7D(
            img_size=args.img_size,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim
        ).to(device)
        criterion = TrajectoryLoss7D()
    else:
        model = ViTTrajectoryExtractor(
            img_size=args.img_size,
            num_points=args.num_points,
            embed_dim=args.embed_dim
        ).to(device)
        criterion = TrajectoryLoss2D()

    if args.phase == 2 and args.phase1_checkpoint:
        load_phase1_checkpoint(model, args.phase1_checkpoint, device)

    print(f'Model: {count_parameters(model)}')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val_loss = float('inf')

    print('Starting training...')
    for epoch in range(1, args.epochs + 1):
        train_loss, train_comp = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_comp = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        print(f'\nEpoch {epoch}/{args.epochs}')
        print(f'  Train: {train_loss:.4f} {train_comp}')
        print(f'  Val: {val_loss:.4f} {val_comp}')

        if args.use_wandb and WANDB_AVAILABLE:
            log_dict = {'epoch': epoch, 'train/loss': train_loss, 'val/loss': val_loss}
            for k, v in train_comp.items():
                log_dict[f'train/{k}'] = v
            for k, v in val_comp.items():
                log_dict[f'val/{k}'] = v
            wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(args.output_dir, 'model_best.pth'), args)
            print('  New best!')

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(args.output_dir, f'model_epoch_{epoch}.pth'), args)

    save_checkpoint(model, optimizer, args.epochs, val_loss,
                    os.path.join(args.output_dir, 'model_final.pth'), args)
    print(f'\nDone! Best val loss: {best_val_loss:.4f}')

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main()

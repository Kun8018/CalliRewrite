#!/usr/bin/env python3
"""
ViT + Trajectory Queries 训练脚本
支持 seq_extract 风格两阶段：
phase1: QuickDraw-clean 预训练
phase2: 书法监督数据 fine-tune（.png + .npz）
"""
import os
import sys
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

from model import ViTTrajectoryExtractor, ViTTrajectoryExtractor7D, ViTAutoregressiveExtractor7D, count_parameters
from dataset import StrokeDatasetViT, QuickDrawCleanDatasetViT, ImageOnlyDatasetViT, initial_seq7_state, apply_seq7_step
from neural_renderer import NeuralRasterizorStep, seq7_to_absolute
from vgg_loss import VGG16PerceptualLoss

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


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


class UnsupervisedLoss(nn.Module):
    """无监督训练损失：渲染损失 + 感知损失"""

    def __init__(self, render_weight=1.0, perceptual_weight=1.0, img_size=224):
        super().__init__()
        self.render_weight = render_weight
        self.perceptual_weight = perceptual_weight
        self.l1_loss = nn.L1Loss()
        self.renderer = NeuralRasterizorStep(img_size)
        self.perceptual_loss = VGG16PerceptualLoss()
        self.img_size = img_size

    def forward(self, strokes_seq7, target_images):
        """
        strokes_seq7: (N, seq_len, 7) - 模型输出的 seq7 格式
        target_images: (N, 1, img_size, img_size) or (N, img_size, img_size) - 目标图像
        """
        # 将 seq7 转换为绝对坐标格式
        strokes_abs = seq7_to_absolute(strokes_seq7, self.img_size)

        # 渲染图像
        rendered = self.renderer(strokes_abs)  # (N, img_size, img_size) [0.0-BG, 1.0-stroke]

        # 处理 target_images 的维度和格式
        if target_images.dim() == 4 and target_images.size(1) == 1:
            target_images = target_images.squeeze(1)  # (N, H, W)

        # 确保 target_images 是 [0.0-BG, 1.0-stroke] 格式
        # 如果输入是 [0.0-stroke, 1.0-BG]，需要反转
        if target_images.min() >= 0.0 and target_images.max() <= 1.0:
            # 检查是否需要反转：假设目标图像中笔画是深色（低值）
            if target_images.mean() > 0.5:
                target_images = 1.0 - target_images

        render_loss = self.l1_loss(rendered, target_images)
        perc_loss = self.perceptual_loss(
            rendered.unsqueeze(1),
            target_images.unsqueeze(1)
        )
        total_loss = self.render_weight * render_loss + self.perceptual_weight * perc_loss
        return {
            'total': total_loss,
            'render': render_loss,
            'perceptual': perc_loss
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT Trajectory Extractor')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='1=QuickDraw-clean 预训练, 2=书法数据无监督 fine-tune')
    parser.add_argument('--dataset_root', type=str, default='../seq_extract/datasets',
                        help='seq_extract 数据根目录，phase1 默认读取 QuickDraw-clean')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='phase2 数据目录，只需要 .png/.jpg 图片')
    parser.add_argument('--phase1_checkpoint', type=str, default=None,
                        help='phase2 从 phase1 checkpoint 初始化')
    parser.add_argument('--output_dir', type=str, default='./output_vit_query', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_points', type=int, default=100, help='2D 模式的点数')
    parser.add_argument('--seq_len', type=int, default=100, help='7D 模式的序列长度')
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--num_heads', type=int, default=None, help='transformer attention heads, auto-selected if None')
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
    parser.add_argument('--use-wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--use-tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--wandb_project', type=str, default='vit-query-stroke')
    parser.add_argument('--save_every', type=int, default=10)
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, unsupervised=False):
    model.train()
    total_loss = 0.0
    loss_components = {}
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch in pbar:
        optimizer.zero_grad()

        if unsupervised:
            images = batch['image'].to(device)
            if model.__class__.__name__ == 'ViTAutoregressiveExtractor7D':
                # 注意：autoregressive 模型的完整推理循环不可微，
                # phase2 推荐使用 ViTTrajectoryExtractor7D (oneshot)
                # 这里作为 fallback，仍然使用 no_grad
                with torch.no_grad():
                    target_mask = 1.0 - images
                    target_tokens, target_global = model.encode_target(target_mask)
                    state = initial_seq7_state(model.img_size)
                    hidden = None
                    strokes_list = []
                    for i in range(model.seq_len):
                        canvas = torch.tensor(state['canvas'], dtype=torch.float32, device=device).unsqueeze(0)
                        cursor = torch.tensor(state['cursor'], dtype=torch.float32, device=device).unsqueeze(0)
                        prev_stroke = torch.zeros(1, 7, dtype=torch.float32, device=device) if i == 0 else strokes_list[-1].unsqueeze(0)
                        step_index = torch.tensor([[i / model.seq_len]], dtype=torch.float32, device=device)
                        output, hidden = model.forward_step(target_tokens, target_global, canvas, cursor, prev_stroke, step_index, hidden)
                        stroke = output['seq'].squeeze(0).detach()
                        strokes_list.append(stroke)
                        state = apply_seq7_step(state, stroke.cpu().numpy(), model.img_size)
                    strokes = torch.stack(strokes_list, dim=0).unsqueeze(0)
                losses = criterion(strokes, images)
            else:
                # oneshot 模型可以端到端训练
                strokes = model(images)
                losses = criterion(strokes, images)
        else:
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
def validate(model, dataloader, criterion, device, epoch, unsupervised=False):
    model.eval()
    total_loss = 0.0
    loss_components = {}
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')

    for batch in pbar:
        if unsupervised:
            images = batch['image'].to(device)
            if model.__class__.__name__ == 'ViTAutoregressiveExtractor7D':
                target_mask = 1.0 - images
                target_tokens, target_global = model.encode_target(target_mask)
                state = initial_seq7_state(model.img_size)
                hidden = None
                strokes_list = []
                for i in range(model.seq_len):
                    canvas = torch.tensor(state['canvas'], dtype=torch.float32, device=device).unsqueeze(0)
                    cursor = torch.tensor(state['cursor'], dtype=torch.float32, device=device).unsqueeze(0)
                    prev_stroke = torch.zeros(1, 7, dtype=torch.float32, device=device) if i == 0 else strokes_list[-1].unsqueeze(0)
                    step_index = torch.tensor([[i / model.seq_len]], dtype=torch.float32, device=device)
                    output, hidden = model.forward_step(target_tokens, target_global, canvas, cursor, prev_stroke, step_index, hidden)
                    stroke = output['seq'].squeeze(0).detach()
                    strokes_list.append(stroke)
                    state = apply_seq7_step(state, stroke.cpu().numpy(), model.img_size)
                strokes = torch.stack(strokes_list, dim=0).unsqueeze(0)
            else:
                strokes = model(images)
            losses = criterion(strokes, images)
        else:
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

    # Phase2: 无监督训练，只需要图片
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
        raise ValueError('phase2 needs --data_dir with .png/.jpg images')

    # Phase2 使用 ImageOnlyDatasetViT，不需要 npz
    full_dataset = ImageOnlyDatasetViT(
        data_dir=args.data_dir,
        img_size=args.img_size,
        mode=args.mode
    )
    if len(full_dataset) == 0:
        raise ValueError('No images found in data_dir!')

    val_size = max(1, int(len(full_dataset) * args.val_split)) if len(full_dataset) > 1 else 0
    train_size = len(full_dataset) - val_size
    if val_size == 0:
        return full_dataset, full_dataset
    return random_split(full_dataset, [train_size, val_size])


class Tee:
    """同时输出到终端和文件"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', buffering=1)

    def write(self, message):
        try:
            self.terminal.write(message)
        except OSError:
            pass  # SSH 断开时忽略 terminal 错误
        self.log.write(message)

    def flush(self):
        try:
            self.terminal.flush()
        except OSError:
            pass  # SSH 断开时忽略 terminal 错误
        self.log.flush()

    def close(self):
        self.log.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 全程日志
    training_log = os.path.join(args.output_dir, 'training.log')
    tee = Tee(training_log)
    sys.stdout = tee
    sys.stderr = tee
    print(f'Training log saved to: {training_log}')

    device = torch.device(args.device)
    print(f'Using device: {device}')
    print(f'Phase: {args.phase}')

    # CSV logger
    csv_path = os.path.join(args.output_dir, 'train_log.csv')
    csv_file = open(csv_path, 'w', buffering=1)  # line-buffered
    csv_writer = csv.writer(csv_file)
    # Write header based on mode/arch
    header = ['epoch', 'train_loss', 'val_loss', 'best_val_loss']
    # Add loss components
    if args.phase == 2:
        header.extend(['train_render', 'train_perceptual', 'val_render', 'val_perceptual'])
    elif args.mode == 'seq7' or args.arch == 'autoregressive':
        header.extend(['train_pen', 'train_coord', 'train_param', 'val_pen', 'val_coord', 'val_param'])
    else:
        header.extend(['train_l1', 'train_mse', 'val_l1', 'val_mse'])
    csv_writer.writerow(header)
    print(f'Training log saved to: {csv_path}')

    # TensorBoard
    writer = None
    if args.use_tensorboard and TENSORBOARD_AVAILABLE:
        tb_dir = os.path.join(args.output_dir, 'tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        print(f'TensorBoard logs saved to: {tb_dir}')

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
            embed_dim=args.embed_dim,
            num_heads=args.num_heads
        ).to(device)
        if args.phase == 2:
            criterion = UnsupervisedLoss(img_size=args.img_size).to(device)
        else:
            criterion = AutoregressiveTrajectoryLoss7D()
    elif args.mode == 'seq7':
        model = ViTTrajectoryExtractor7D(
            img_size=args.img_size,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads
        ).to(device)
        if args.phase == 2:
            criterion = UnsupervisedLoss(img_size=args.img_size).to(device)
        else:
            criterion = TrajectoryLoss7D()
    else:
        model = ViTTrajectoryExtractor(
            img_size=args.img_size,
            num_points=args.num_points,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads
        ).to(device)
        criterion = TrajectoryLoss2D()

    if args.phase == 2 and args.phase1_checkpoint:
        load_phase1_checkpoint(model, args.phase1_checkpoint, device)

    print(f'Model: {count_parameters(model)}')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val_loss = float('inf')

    print('Starting training...')
    unsupervised = (args.phase == 2)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_comp = train_epoch(model, train_loader, optimizer, criterion, device, epoch, unsupervised)
        val_loss, val_comp = validate(model, val_loader, criterion, device, epoch, unsupervised)
        scheduler.step()

        print(f'\nEpoch {epoch}/{args.epochs}')
        print(f'  Train: {train_loss:.4f} {train_comp}')
        print(f'  Val: {val_loss:.4f} {val_comp}')

        # CSV logging
        csv_row = [epoch, train_loss, val_loss, best_val_loss]
        if args.phase == 2:
            csv_row.extend([
                train_comp.get('render', 0.0), train_comp.get('perceptual', 0.0),
                val_comp.get('render', 0.0), val_comp.get('perceptual', 0.0)
            ])
        elif args.mode == 'seq7' or args.arch == 'autoregressive':
            csv_row.extend([
                train_comp.get('pen', 0.0), train_comp.get('coord', 0.0), train_comp.get('param', 0.0),
                val_comp.get('pen', 0.0), val_comp.get('coord', 0.0), val_comp.get('param', 0.0)
            ])
        else:
            csv_row.extend([
                train_comp.get('l1', 0.0), train_comp.get('mse', 0.0),
                val_comp.get('l1', 0.0), val_comp.get('mse', 0.0)
            ])
        csv_writer.writerow(csv_row)

        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            for k, v in train_comp.items():
                writer.add_scalar(f'Loss/train_{k}', v, epoch)
            for k, v in val_comp.items():
                writer.add_scalar(f'Loss/val_{k}', v, epoch)
            writer.flush()

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

    csv_file.close()
    if writer:
        writer.close()

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    # 恢复 stdout/stderr 并关闭 tee
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    tee.close()


if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception as e:
        # 记录报错到文件
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--output_dir', type=str, default='./output_vit_query')
        temp_args, _ = parser.parse_known_args()
        os.makedirs(temp_args.output_dir, exist_ok=True)
        error_log = os.path.join(temp_args.output_dir, 'error.log')
        with open(error_log, 'w') as f:
            f.write(f'Error: {e}\n')
            f.write(traceback.format_exc())
        print(f'Error logged to {error_log}')
        traceback.print_exc()
        sys.exit(1)

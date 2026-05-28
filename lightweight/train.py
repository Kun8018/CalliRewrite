#!/usr/bin/env python3
"""
ResNet-18 lightweight stroke extraction training.
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

from model import StrokeTransformer, ResNetAutoregressiveExtractor7D, count_parameters
from dataset import StrokeDataset, QuickDrawCleanDataset, QuickDrawConverter, ImageOnlyDataset, initial_seq7_state, apply_seq7_step
from neural_renderer import NeuralRasterizorStep, seq7_to_absolute
from vgg_loss import VGG16PerceptualLoss


class StrokeLoss(nn.Module):
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

    def __init__(self, render_weight=1.0, perceptual_weight=1.0, image_size=256):
        super().__init__()
        self.render_weight = render_weight
        self.perceptual_weight = perceptual_weight
        self.l1_loss = nn.L1Loss()
        self.renderer = NeuralRasterizorStep(image_size)
        self.perceptual_loss = VGG16PerceptualLoss()
        self.image_size = image_size

    def forward(self, strokes_seq7, target_images):
        """
        strokes_seq7: (N, seq_len, 7) - 模型输出的 seq7 格式
        target_images: (N, 1, H, W) or (N, H, W) - 目标图像
        """
        # 将 seq7 转换为绝对坐标格式
        strokes_abs = seq7_to_absolute(strokes_seq7, self.image_size)

        # 渲染图像
        rendered = self.renderer(strokes_abs)  # (N, H, W) [0.0-BG, 1.0-stroke]

        # 处理 target_images 的维度和格式
        if target_images.dim() == 4 and target_images.size(1) == 1:
            target_images = target_images.squeeze(1)

        # 确保 target_images 是 [0.0-BG, 1.0-stroke] 格式
        if target_images.min() >= 0.0 and target_images.max() <= 1.0:
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
    parser = argparse.ArgumentParser(description='Train lightweight stroke extractor')
    parser.add_argument('--arch', type=str, default='autoregressive', choices=['autoregressive', 'oneshot'])
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
    parser.add_argument('--dataset_root', type=str, default='../seq_extract/datasets')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--quickdraw_npz', type=str, default=None)
    parser.add_argument('--quickdraw_save_dir', type=str, default='./qd_data_lightweight')
    parser.add_argument('--phase1_checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./output_lightweight')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--max_seq_len', type=int, default=100)
    parser.add_argument('--chunk_len', type=int, default=8)
    parser.add_argument('--chunks_per_sample', type=int, default=4)
    parser.add_argument('--max_items_per_category', type=int, default=None)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=None)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--use-wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--use-tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--wandb_project', type=str, default='lightweight-stroke')
    parser.add_argument('--save_every', type=int, default=10)
    return parser.parse_args()


def load_quickdraw_data(quickdraw_npz, save_dir, image_size=256, max_items=10000):
    print(f'Loading QuickDraw data from {quickdraw_npz}...')
    sketches = QuickDrawConverter.load_quickdraw_npz(quickdraw_npz, max_items=max_items)
    os.makedirs(save_dir, exist_ok=True)
    QuickDrawConverter.create_pairs(sketches, save_dir=save_dir, image_size=image_size)
    return save_dir


def build_dataset(args):
    if args.phase == 1:
        train_dataset = QuickDrawCleanDataset(
            dataset_root=args.dataset_root,
            split='train',
            image_size=args.image_size,
            max_seq_len=args.max_seq_len,
            max_items_per_category=args.max_items_per_category,
            arch=args.arch,
            chunk_len=args.chunk_len,
            chunks_per_sample=args.chunks_per_sample
        )
        val_dataset = QuickDrawCleanDataset(
            dataset_root=args.dataset_root,
            split='test',
            image_size=args.image_size,
            max_seq_len=args.max_seq_len,
            max_items_per_category=args.max_items_per_category,
            arch=args.arch,
            chunk_len=args.chunk_len,
            chunks_per_sample=1
        )
        return train_dataset, val_dataset

    if args.quickdraw_npz is not None:
        args.data_dir = load_quickdraw_data(
            args.quickdraw_npz,
            args.quickdraw_save_dir,
            image_size=args.image_size,
            max_items=args.max_items_per_category or 50000
        )
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

    # Phase2 用 ImageOnlyDataset，不需要 npz
    full_dataset = ImageOnlyDataset(
        data_dir=args.data_dir,
        image_size=args.image_size
    )
    if len(full_dataset) == 0:
        raise ValueError('No images found!')
    val_size = max(1, int(len(full_dataset) * args.val_split)) if len(full_dataset) > 1 else 0
    train_size = len(full_dataset) - val_size
    if val_size == 0:
        return full_dataset, full_dataset
    return random_split(full_dataset, [train_size, val_size])


def build_model_and_loss(args, device):
    if args.arch == 'autoregressive':
        model = ResNetAutoregressiveExtractor7D(
            image_size=args.image_size,
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            num_heads=args.num_heads
        ).to(device)
        if args.phase == 2:
            criterion = UnsupervisedLoss(image_size=args.image_size).to(device)
        else:
            criterion = AutoregressiveTrajectoryLoss7D()
    else:
        model = StrokeTransformer(
            d_model=args.d_model,
            nhead=args.nhead,
            num_decoder_layers=args.num_decoder_layers,
            max_seq_len=args.max_seq_len
        ).to(device)
        if args.phase == 2:
            criterion = UnsupervisedLoss(image_size=args.image_size).to(device)
        else:
            criterion = StrokeLoss()
    return model, criterion


def load_phase1_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f'Loaded phase1 checkpoint: {checkpoint_path}')
    if missing:
        print(f'  Missing keys: {len(missing)}')
    if unexpected:
        print(f'  Unexpected keys: {len(unexpected)}')


def run_batch(model, criterion, batch, device, arch, teacher_forcing_ratio=0.0):
    if arch == 'autoregressive':
        outputs = model.forward_teacher_forcing(
            batch['target_mask'].to(device),
            batch['canvases'].to(device),
            batch['cursors'].to(device),
            batch['prev_strokes'].to(device),
            batch['step_indices'].to(device)
        )
        targets = batch['strokes'].to(device)
        mask = batch['mask'].to(device)
        return criterion(outputs, targets, mask)

    images = batch['image'].to(device)
    strokes = batch['strokes'].to(device)
    mask = batch['mask'].to(device)
    predictions = model(images, strokes, teacher_forcing_ratio=teacher_forcing_ratio)
    return criterion(predictions, strokes, mask)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, args):
    model.train()
    total_loss = 0.0
    if args.phase == 2:
        loss_components = {'render': 0.0, 'perceptual': 0.0}
    else:
        loss_components = {'pen': 0.0, 'coord': 0.0, 'param': 0.0}
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch in pbar:
        optimizer.zero_grad()
        if args.phase == 2:
            images = batch['image'].to(device)
            if args.arch == 'autoregressive':
                # autoregressive 模型完整推理不可微，用 no_grad
                with torch.no_grad():
                    target_mask = 1.0 - images
                    target_tokens, target_global = model.encode_target(target_mask)
                    state = initial_seq7_state(model.image_size)
                    hidden = None
                    strokes_list = []
                    window_size = torch.full((1, 1), model.init_window_size, dtype=torch.float32, device=device)
                    for i in range(model.max_seq_len):
                        canvas = torch.tensor(state['canvas'], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                        cursor = torch.tensor(state['cursor'], dtype=torch.float32, device=device).unsqueeze(0)
                        prev_stroke = torch.zeros(1, 7, dtype=torch.float32, device=device) if i == 0 else strokes_list[-1].unsqueeze(0)
                        step_index = torch.tensor([[i / model.max_seq_len]], dtype=torch.float32, device=device)
                        output, hidden = model.forward_step(target_tokens, target_global, target_mask, canvas, cursor, prev_stroke, step_index, hidden, window_size)
                        stroke = output['seq'].squeeze(0).detach()
                        strokes_list.append(stroke)
                        state = apply_seq7_step(state, stroke.cpu().numpy(), model.image_size)
                    strokes_seq7 = torch.stack(strokes_list, dim=0).unsqueeze(0)
                losses = criterion(strokes_seq7, images)
            else:
                # oneshot 模型可以端到端训练
                strokes_seq7 = model(images)
                losses = criterion(strokes_seq7, images)
        else:
            losses = run_batch(model, criterion, batch, device, args.arch, args.teacher_forcing_ratio)
        loss = losses['total']
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        for k in loss_components:
            if k in losses:
                loss_components[k] += losses[k].item()
        pbar.set_postfix({'loss': loss.item()})

    num_batches = max(len(dataloader), 1)
    for k in loss_components:
        loss_components[k] /= num_batches
    return total_loss / num_batches, loss_components


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, args):
    model.eval()
    total_loss = 0.0
    if args.phase == 2:
        loss_components = {'render': 0.0, 'perceptual': 0.0}
    else:
        loss_components = {'pen': 0.0, 'coord': 0.0, 'param': 0.0}
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')

    for batch in pbar:
        if args.phase == 2:
            images = batch['image'].to(device)
            if args.arch == 'autoregressive':
                target_mask = 1.0 - images
                target_tokens, target_global = model.encode_target(target_mask)
                state = initial_seq7_state(model.image_size)
                hidden = None
                strokes_list = []
                window_size = torch.full((1, 1), model.init_window_size, dtype=torch.float32, device=device)
                for i in range(model.max_seq_len):
                    canvas = torch.tensor(state['canvas'], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                    cursor = torch.tensor(state['cursor'], dtype=torch.float32, device=device).unsqueeze(0)
                    prev_stroke = torch.zeros(1, 7, dtype=torch.float32, device=device) if i == 0 else strokes_list[-1].unsqueeze(0)
                    step_index = torch.tensor([[i / model.max_seq_len]], dtype=torch.float32, device=device)
                    output, hidden = model.forward_step(target_tokens, target_global, target_mask, canvas, cursor, prev_stroke, step_index, hidden, window_size)
                    stroke = output['seq'].squeeze(0).detach()
                    strokes_list.append(stroke)
                    state = apply_seq7_step(state, stroke.cpu().numpy(), model.image_size)
                strokes_seq7 = torch.stack(strokes_list, dim=0).unsqueeze(0)
                losses = criterion(strokes_seq7, images)
            else:
                strokes_seq7 = model(images)
                losses = criterion(strokes_seq7, images)
        else:
            losses = run_batch(model, criterion, batch, device, args.arch, 0.0)
        total_loss += losses['total'].item()
        for k in loss_components:
            if k in losses:
                loss_components[k] += losses[k].item()
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
        'arch': args.arch,
        'args': vars(args),
    }, save_path)
    print(f'Checkpoint saved to {save_path}')


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
    print(f'Arch: {args.arch}, Phase: {args.phase}')

    # CSV logger
    csv_path = os.path.join(args.output_dir, 'train_log.csv')
    csv_file = open(csv_path, 'w', buffering=1)  # line-buffered
    csv_writer = csv.writer(csv_file)
    # Write header based on phase
    if args.phase == 2:
        csv_writer.writerow([
            'epoch',
            'train_loss', 'train_render', 'train_perceptual',
            'val_loss', 'val_render', 'val_perceptual',
            'best_val_loss'
        ])
    else:
        csv_writer.writerow([
            'epoch',
            'train_loss', 'train_pen', 'train_coord', 'train_param',
            'val_loss', 'val_pen', 'val_coord', 'val_param',
        'best_val_loss'
    ])
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
        print('Warning: wandb not installed, continuing without logging')
        args.use_wandb = False

    train_dataset, val_dataset = build_dataset(args)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError('No data found!')
    print(f'Train dataset: {len(train_dataset)} samples')
    print(f'Val dataset: {len(val_dataset)} samples')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model, criterion = build_model_and_loss(args, device)
    if args.phase == 2 and args.phase1_checkpoint:
        load_phase1_checkpoint(model, args.phase1_checkpoint, device)
    print(f'Model created: {count_parameters(model)}')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val_loss = float('inf')

    print('Starting training...')
    for epoch in range(1, args.epochs + 1):
        train_loss, train_comp = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args)
        val_loss, val_comp = validate(model, val_loader, criterion, device, epoch, args)
        scheduler.step()

        print(f'\nEpoch {epoch}/{args.epochs}')
        if args.phase == 2:
            print(f'  Train Loss: {train_loss:.4f} (render={train_comp["render"]:.4f}, perceptual={train_comp["perceptual"]:.4f})')
            print(f'  Val Loss: {val_loss:.4f} (render={val_comp["render"]:.4f}, perceptual={val_comp["perceptual"]:.4f})')
        else:
            print(f'  Train Loss: {train_loss:.4f} (pen={train_comp["pen"]:.4f}, coord={train_comp["coord"]:.4f}, param={train_comp["param"]:.4f})')
            print(f'  Val Loss: {val_loss:.4f} (pen={val_comp["pen"]:.4f}, coord={val_comp["coord"]:.4f}, param={val_comp["param"]:.4f})')

        # CSV logging
        if args.phase == 2:
            csv_writer.writerow([
                epoch,
                train_loss, train_comp.get('render', 0.0), train_comp.get('perceptual', 0.0),
                val_loss, val_comp.get('render', 0.0), val_comp.get('perceptual', 0.0),
                best_val_loss
            ])
        else:
            csv_writer.writerow([
                epoch,
                train_loss, train_comp.get('pen', 0.0), train_comp.get('coord', 0.0), train_comp.get('param', 0.0),
                val_loss, val_comp.get('pen', 0.0), val_comp.get('coord', 0.0), val_comp.get('param', 0.0),
                best_val_loss
            ])

        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            if args.phase == 2:
                writer.add_scalar('Loss/train_render', train_comp.get('render', 0.0), epoch)
                writer.add_scalar('Loss/train_perceptual', train_comp.get('perceptual', 0.0), epoch)
                writer.add_scalar('Loss/val_render', val_comp.get('render', 0.0), epoch)
                writer.add_scalar('Loss/val_perceptual', val_comp.get('perceptual', 0.0), epoch)
            else:
                writer.add_scalar('Loss/train_pen', train_comp.get('pen', 0.0), epoch)
                writer.add_scalar('Loss/train_coord', train_comp.get('coord', 0.0), epoch)
                writer.add_scalar('Loss/train_param', train_comp.get('param', 0.0), epoch)
                writer.add_scalar('Loss/val_pen', val_comp.get('pen', 0.0), epoch)
                writer.add_scalar('Loss/val_coord', val_comp.get('coord', 0.0), epoch)
                writer.add_scalar('Loss/val_param', val_comp.get('param', 0.0), epoch)
            writer.flush()

        # WandB logging
        if args.use_wandb and WANDB_AVAILABLE:
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(args.output_dir, 'model_best.pth'), args)
            print(f'  New best val loss: {best_val_loss:.4f}')

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(args.output_dir, f'model_epoch_{epoch}.pth'), args)

    save_checkpoint(model, optimizer, args.epochs, val_loss, os.path.join(args.output_dir, 'model_final.pth'), args)
    print('\nTraining complete!')
    print(f'Best val loss: {best_val_loss:.4f}')

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
        parser.add_argument('--output_dir', type=str, default='./output_lightweight')
        temp_args, _ = parser.parse_known_args()
        os.makedirs(temp_args.output_dir, exist_ok=True)
        error_log = os.path.join(temp_args.output_dir, 'error.log')
        with open(error_log, 'w') as f:
            f.write(f'Error: {e}\n')
            f.write(traceback.format_exc())
        print(f'Error logged to {error_log}')
        traceback.print_exc()
        sys.exit(1)

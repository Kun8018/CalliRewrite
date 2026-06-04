#!/usr/bin/env python3
"""conv13_c3 自回归笔画提取器 — 尽量对齐 seq_extract 的训练策略。

phase 1: 在 QuickDraw-clean 上学 stroke 序列；
  - 完全闭环 rollout，没有 teacher forcing；
  - loss = raster(L1+VGG perceptual) + 7 项辅助 loss；
  - stroke_num_loss curriculum: 从 0 增加到 0.5；
  - 随机初始 cursor。

预训练 NeuralRasterizor 通过 --renderer_ckpt 加载，强烈建议 freeze（默认）。
"""
import os
import sys
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split, DataLoader, DistributedSampler
from tqdm import tqdm
from PIL import Image, ImageDraw

from model import ViTAutoregressiveExtractor7D, count_parameters
from neural_renderer import NeuralRasterizorStep
from dataset import QuickDrawCleanDataset, ImageOnlyDataset
from losses import CombinedRolloutLoss
from visualize import generate_order_image


# --------------------------------------------------------------------- #
# DDP helpers
# --------------------------------------------------------------------- #

def setup_ddp():
    """初始化 DDP 环境（若用 torchrun 启动）。
    返回 (local_rank, world_size, is_main)。
    单卡时返回 (0, 1, True)。"""
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        world_size = dist.get_world_size()
        is_main = local_rank == 0
        if is_main:
            print(f'[DDP] world_size={world_size}, local_rank={local_rank}')
        return local_rank, world_size, is_main
    return 0, 1, True


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value: float, world_size: int) -> float:
    """跨 rank 平均一个标量。"""
    if world_size <= 1:
        return value
    t = torch.tensor([value], dtype=torch.float64, device='cuda')
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world_size).item()


# --------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description='Train ViT stroke extractor (seqextract-style)')
    p.add_argument('--phase', type=int, default=1, choices=[1, 2])
    p.add_argument('--dataset_root', type=str, default='../seq_extract/datasets')
    p.add_argument('--data_dir', type=str, default=None,
                   help='phase2 image-only data dir')
    p.add_argument('--phase1_checkpoint', type=str, default=None)
    p.add_argument('--renderer_ckpt', type=str, required=True,
                   help='预训练 RasterUnit 权重 (.pth)，必须提供')
    p.add_argument('--freeze_renderer', action='store_true', default=True)
    p.add_argument('--no_freeze_renderer', dest='freeze_renderer', action='store_false')
    p.add_argument('--output_dir', type=str, default='./output_vit_seqextract')

    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--max_seq_len', type=int, default=48)
    p.add_argument('--patch_size', type=int, default=64)
    p.add_argument('--raster_size', type=int, default=128)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--num_heads', type=int, default=None)
    p.add_argument('--no_pretrained_vit', action='store_true',
                   help='不加载 torchvision ViT ImageNet 预训练权重（默认加载）')

    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--epochs', type=int, default=None,
                   help='兼容旧参数；如果不设 --num_steps，则作为 num_steps 使用')
    p.add_argument('--num_steps', type=int, default=90040,
                   help='seq_extract phase1 默认 90040 steps，phase2 默认 30020 steps')
    p.add_argument('--eval_every', type=int, default=5000,
                   help='每多少 step 跑一次验证')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--decay_power', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--amp', action='store_true', default=False,
                   help='bf16 混合精度训练')
    p.add_argument('--no_amp', dest='amp', action='store_false')
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--max_items_per_category', type=int, default=5000,
                   help='Phase1 每个 QuickDraw 类别取多少样本')
    p.add_argument('--cache_size', type=int, default=50000,
                   help='dataset 内存 cache 多少处理过的样本，0 关闭')

    # 随机初始 cursor
    p.add_argument('--random_init_cursor', action='store_true', default=True,
                   help='训练时从 stroke 像素随机采初始 cursor')
    p.add_argument('--no_random_init_cursor', dest='random_init_cursor', action='store_false')
    p.add_argument('--use_gt_init_cursor', action='store_true', default=False,
                   help='phase1 使用 GT 起笔点（seqextract 默认关闭）')
    p.add_argument('--no_gt_init_cursor', dest='use_gt_init_cursor', action='store_false')
    p.add_argument('--init_cursor_low', type=float, default=0.2)
    p.add_argument('--init_cursor_high', type=float, default=0.8)

    # loss weights（对齐 seqextract hyper_parameters.py）
    p.add_argument('--w_raster', type=float, default=1.0)
    p.add_argument('--w_stroke_num', type=float, default=0.5)
    p.add_argument('--w_stroke_num_end', type=float, default=0.0)
    p.add_argument('--sn_loss_type', type=str, default='increasing',
                   choices=['fixed', 'increasing'], help='seqextract: increasing from 0 to 0.5')
    p.add_argument('--w_smoothness', type=float, default=None,
                   help='默认 phase1=0, phase2=0.5')
    p.add_argument('--w_angle', type=float, default=None,
                   help='默认 phase1=0, phase2=1.0')
    p.add_argument('--w_outside', type=float, default=10.0)
    p.add_argument('--w_win_outside', type=float, default=10.0)
    p.add_argument('--w_early_pen', type=float, default=0.1)
    p.add_argument('--early_pen_length', type=int, default=7)
    p.add_argument('--early_pen_loss_type', type=str, default='move',
                   choices=['move'], help='seqextract: move')

    # Seqextract 没有监督 loss
    p.add_argument('--w_supervised', type=float, default=0.0)
    p.add_argument('--use_perceptual', action='store_true', default=True)
    p.add_argument('--no_perceptual', dest='use_perceptual', action='store_false')
    p.add_argument('--use_l1_raster', action='store_true', default=True)
    p.add_argument('--no_l1_raster', dest='use_l1_raster', action='store_false')
    p.add_argument('--perc_loss_fuse_type', type=str, default='add',
                   choices=['raw_add', 'add'])

    p.add_argument('--fail_on_nonfinite', action='store_true', default=True)
    p.add_argument('--skip_nonfinite', dest='fail_on_nonfinite', action='store_false')

    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--use_tensorboard', action='store_true')
    p.add_argument('--save_every', type=int, default=15000)
    return p.parse_args()


def setdefault_weights_by_phase(args):
    if args.phase == 1:
        if args.w_smoothness is None: args.w_smoothness = 0.0
        if args.w_angle is None:      args.w_angle = 0.0
    else:
        if args.w_smoothness is None: args.w_smoothness = 0.5
        if args.w_angle is None:      args.w_angle = 1.0


# --------------------------------------------------------------------- #
# Tee logger
# --------------------------------------------------------------------- #

class Tee:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', buffering=1)

    def write(self, msg):
        try:
            self.terminal.write(msg)
        except OSError:
            pass
        self.log.write(msg)

    def flush(self):
        try:
            self.terminal.flush()
        except OSError:
            pass
        self.log.flush()

    def close(self):
        self.log.close()


def create_summary_writer(output_dir):
    """Create a TensorBoard writer with clear fallback."""
    tb_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(tb_dir, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(tb_dir)
        print(f'TensorBoard: {tb_dir} (torch.utils.tensorboard)')
        return writer
    except ImportError as torch_tb_error:
        try:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(tb_dir)
            print(f'TensorBoard: {tb_dir} (tensorboardX)')
            return writer
        except ImportError:
            print('[warn] TensorBoard is not installed.')
            return None


# --------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------- #

def build_dataset(args):
    if args.phase == 1:
        train_ds = QuickDrawCleanDataset(
            dataset_root=args.dataset_root, split='train',
            image_size=args.image_size, max_seq_len=args.max_seq_len,
            max_items_per_category=args.max_items_per_category,
            cache_size=args.cache_size)
        val_ds = QuickDrawCleanDataset(
            dataset_root=args.dataset_root, split='test',
            image_size=args.image_size, max_seq_len=args.max_seq_len,
            max_items_per_category=max(500, args.max_items_per_category // 10),
            cache_size=args.cache_size // 5 if args.cache_size > 0 else 0)
        return train_ds, val_ds

    if args.data_dir is None:
        candidates = [
            '../seq_extract/outputs/__new_train_phase_2',
        ]
        for d in candidates:
            if os.path.exists(d):
                args.data_dir = d
                print(f'Using phase2 data dir: {d}')
                break
    if args.data_dir is None:
        raise ValueError('phase2 needs --data_dir')
    full = ImageOnlyDataset(data_dir=args.data_dir, image_size=args.image_size)
    if len(full) == 0:
        raise ValueError('No images found')
    val_size = max(1, int(len(full) * args.val_split)) if len(full) > 1 else 0
    train_size = len(full) - val_size
    if val_size == 0:
        return full, full
    return random_split(full, [train_size, val_size])


def build_model_renderer_loss(args, device):
    renderer = NeuralRasterizorStep(raster_size=args.raster_size,
                                    pretrained_path=args.renderer_ckpt,
                                    freeze=args.freeze_renderer).to(device)

    model = ViTAutoregressiveExtractor7D(
        image_size=args.image_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        raster_size=args.raster_size,
        pretrained=not args.no_pretrained_vit).to(device)

    # Seqextract style: use raw_add for perceptual, no normalization
    use_perceptual_norm = args.perc_loss_fuse_type == 'add'
    loss_fn = CombinedRolloutLoss(
        raster_weight=args.w_raster,
        stroke_num_weight=args.w_stroke_num,
        smoothness_weight=args.w_smoothness,
        angle_weight=args.w_angle,
        outside_weight=args.w_outside,
        win_outside_weight=args.w_win_outside,
        early_pen_weight=args.w_early_pen,
        early_pen_length=args.early_pen_length,
        supervised_weight=args.w_supervised,
        use_perceptual=args.use_perceptual,
        use_l1_raster=args.use_l1_raster,
        use_perceptual_norm=use_perceptual_norm,
        phase=args.phase,
    ).to(device)
    return model, renderer, loss_fn


def load_phase1_checkpoint(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(sd['model_state_dict'], strict=False)
    print(f'[phase1 init] loaded {ckpt_path}')
    if missing:
        print(f'  missing: {len(missing)} (e.g. {missing[:3]})')
    if unexpected:
        print(f'  unexpected: {len(unexpected)} (e.g. {unexpected[:3]})')


def schedule_lr(args, global_step):
    """对齐 seqextract 的多项式衰减 lr schedule."""
    total_steps = args.num_steps
    progress = min(global_step / total_steps, 1.0)
    lr_diff = args.lr - args.min_lr
    lr = args.min_lr + lr_diff * (1 - progress) ** args.decay_power
    return lr


def schedule_stroke_num_weight(args, global_step):
    """对齐 seqextract 的 stroke_num_loss curriculum：从 0 增加到 0.5。"""
    if args.sn_loss_type == 'fixed':
        return args.w_stroke_num
    elif args.sn_loss_type == 'increasing':
        # from 0 to w_stroke_num
        progress = min(global_step / args.num_steps, 1.0)
        return args.w_stroke_num_end + (args.w_stroke_num - args.w_stroke_num_end) * progress
    return args.w_stroke_num


# --------------------------------------------------------------------- #
# random init cursor (对齐 seqextract)
# --------------------------------------------------------------------- #

def sample_init_cursors_from_stroke(target_stroke_img: torch.Tensor,
                                    lo: float = 0.2, hi: float = 0.8,
                                    stroke_thresh: float = 0.5) -> torch.Tensor:
    N, H, W = target_stroke_img.shape
    device = target_stroke_img.device
    dtype = target_stroke_img.dtype
    fallback = torch.rand(N, 2, device=device, dtype=dtype) * (hi - lo) + lo
    init_cursor = fallback.clone()
    for i in range(N):
        ys, xs = torch.where(target_stroke_img[i] > stroke_thresh)
        if ys.numel() == 0:
            continue
        x_norm = xs.to(dtype) / W
        y_norm = ys.to(dtype) / H
        in_bounds = ((x_norm >= lo) & (x_norm <= hi) & (y_norm >= lo) & (y_norm <= hi))
        if in_bounds.any():
            xs, ys = xs[in_bounds], ys[in_bounds]
        pick = torch.randint(0, xs.numel(), (1,), device=device).item()
        init_cursor[i, 0] = xs[pick].to(dtype) / W
        init_cursor[i, 1] = ys[pick].to(dtype) / H
    return init_cursor


# --------------------------------------------------------------------- #
# step (完全闭环 rollout)
# --------------------------------------------------------------------- #

def run_step(model, renderer, loss_fn, batch, device, args, stroke_num_weight, training: bool):
    target_image = batch['target_image'].to(device, non_blocking=True)
    target_stroke_img = batch['target_stroke_img'].to(device, non_blocking=True)

    gt_strokes = batch.get('gt_strokes')
    gt_mask = batch.get('gt_mask')
    if gt_strokes is not None:
        gt_strokes = gt_strokes.to(device, non_blocking=True)
        gt_mask = gt_mask.to(device, non_blocking=True)

    # 初始 cursor
    gt_init_cursor = batch.get('init_cursor')
    if args.use_gt_init_cursor and gt_init_cursor is not None:
        init_cursor = gt_init_cursor.to(device, non_blocking=True)
    elif training and args.random_init_cursor:
        init_cursor = sample_init_cursors_from_stroke(
            target_stroke_img, lo=args.init_cursor_low, hi=args.init_cursor_high)
    else:
        init_cursor = None

    use_amp = args.amp and device.type == 'cuda'
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        # 完全闭环 rollout，没有 teacher forcing
        rollout = model(
            target_image, renderer,
            seq_len=args.max_seq_len,
            gt_strokes=gt_strokes,
            scheduled_sampling_prob=0.0,
            teacher_forcing_prob=0.0,  # NO TEACHER FORCING
            init_cursor=init_cursor,
        )
        # 更新 loss_fn 的 stroke_num_weight
        losses = loss_fn(rollout, target_stroke_img, args.image_size,
                         gt_strokes=gt_strokes, gt_mask=gt_mask,
                         stroke_num_weight_override=stroke_num_weight)
    return losses, rollout


def format_loss_components(losses):
    parts = []
    for k, v in losses.items():
        if torch.is_tensor(v):
            if v.numel() == 1:
                parts.append(f'{k}={v.detach().float().item():.6g}')
            else:
                parts.append(f'{k}=tensor{tuple(v.shape)}')
        else:
            parts.append(f'{k}={float(v):.6g}')
    return ', '.join(parts)


def find_first_nonfinite_grad(model):
    named_params = model.module.named_parameters() if isinstance(model, DDP) else model.named_parameters()
    max_name = None
    max_abs = -1.0
    for name, param in named_params:
        grad = param.grad
        if grad is None:
            continue
        finite = torch.isfinite(grad)
        if not finite.all():
            bad = grad[~finite]
            bad_value = bad.flatten()[0].detach().float().item()
            return f'{name}: first_bad_grad={bad_value}'
        current_max = grad.detach().abs().max().float().item()
        if current_max > max_abs:
            max_abs = current_max
            max_name = name
    if max_name is not None:
        return f'all_grads_finite, max_abs_grad={max_abs:.6g} at {max_name}'
    return 'no_grads'


def train_epoch(model, renderer, loss_fn, loader, optimizer, device, epoch, args, stroke_num_weight):
    model.train()
    if hasattr(renderer.raster_unit, 'eval') and args.freeze_renderer:
        renderer.raster_unit.eval()
    total = 0.0
    comp_acc = {}
    num_updates = 0
    is_main = (getattr(args, 'local_rank', 0) == 0)
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]', disable=not is_main)
    for step, batch in enumerate(pbar, start=1):
        optimizer.zero_grad()
        losses, _ = run_step(model, renderer, loss_fn, batch, device, args, stroke_num_weight, training=True)
        loss = losses['total']
        if not torch.isfinite(loss):
            msg = (f'Non-finite train loss at epoch={epoch}, step={step}: {format_loss_components(losses)}')
            if args.fail_on_nonfinite:
                raise FloatingPointError(msg)
            if is_main:
                print(f'[skip] {msg}')
            continue
        loss.backward()
        if args.grad_clip > 0:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=False)
            if not torch.isfinite(grad_norm):
                msg = (f'Non-finite grad norm at epoch={epoch}, step={step}: '
                       f'grad_norm={grad_norm.item()}, bad_grad={find_first_nonfinite_grad(model)}, {format_loss_components(losses)}')
                if args.fail_on_nonfinite:
                    raise FloatingPointError(msg)
                if is_main:
                    print(f'[skip] {msg}')
                optimizer.zero_grad(set_to_none=True)
                continue
        optimizer.step()
        total += loss.item()
        num_updates += 1
        for k, v in losses.items():
            if k == 'total':
                continue
            comp_acc[k] = comp_acc.get(k, 0.0) + float(v)
        if is_main:
            pbar.set_postfix(loss=loss.item(), sn_weight=f'{stroke_num_weight:.3f}')
    if num_updates == 0:
        raise RuntimeError(f'No training batches processed at epoch={epoch}.')
    comp_acc = {k: v / num_updates for k, v in comp_acc.items()}
    return total / num_updates, comp_acc


@torch.no_grad()
def validate(model, renderer, loss_fn, loader, device, epoch, args, stroke_num_weight):
    model.eval()
    total = 0.0
    comp_acc = {}
    num_batches = 0
    is_main = (getattr(args, 'local_rank', 0) == 0)
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]', disable=not is_main)
    for step, batch in enumerate(pbar, start=1):
        losses, _ = run_step(model, renderer, loss_fn, batch, device, args, stroke_num_weight, training=False)
        if not torch.isfinite(losses['total']):
            raise FloatingPointError(f'Non-finite val loss at epoch={epoch}, step={step}: {format_loss_components(losses)}')
        total += losses['total'].item()
        num_batches += 1
        for k, v in losses.items():
            if k == 'total':
                continue
            comp_acc[k] = comp_acc.get(k, 0.0) + float(v)
        if is_main:
            pbar.set_postfix(loss=losses['total'].item())
    if num_batches == 0:
        raise RuntimeError(f'No validation batches processed at epoch={epoch}.')
    comp_acc = {k: v / num_batches for k, v in comp_acc.items()}
    return total / num_batches, comp_acc


def next_train_batch(loader, iterator, sampler, data_epoch):
    try:
        return next(iterator), iterator, data_epoch
    except StopIteration:
        data_epoch += 1
        if sampler is not None:
            sampler.set_epoch(data_epoch)
        iterator = iter(loader)
        return next(iterator), iterator, data_epoch


def train_one_step(model, renderer, loss_fn, batch, optimizer, device,
                   step, args, stroke_num_weight):
    model.train()
    if hasattr(renderer.raster_unit, 'eval') and args.freeze_renderer:
        renderer.raster_unit.eval()
    optimizer.zero_grad()
    losses, _ = run_step(
        model, renderer, loss_fn, batch, device, args, stroke_num_weight, training=True)
    loss = losses['total']
    is_main = (getattr(args, 'local_rank', 0) == 0)
    if not torch.isfinite(loss):
        msg = f'Non-finite train loss at step={step}: {format_loss_components(losses)}'
        if args.fail_on_nonfinite:
            raise FloatingPointError(msg)
        if is_main:
            print(f'[skip] {msg}')
        return None, None
    loss.backward()
    if args.grad_clip > 0:
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip, error_if_nonfinite=False)
        if not torch.isfinite(grad_norm):
            msg = (f'Non-finite grad norm at step={step}: '
                   f'grad_norm={grad_norm.item()}, '
                   f'bad_grad={find_first_nonfinite_grad(model)}, '
                   f'{format_loss_components(losses)}')
            if args.fail_on_nonfinite:
                raise FloatingPointError(msg)
            if is_main:
                print(f'[skip] {msg}')
            optimizer.zero_grad(set_to_none=True)
            return None, None
    optimizer.step()
    comp = {k: float(v) for k, v in losses.items() if k != 'total'}
    return loss.item(), comp


def save_checkpoint(model, optim, epoch, loss, save_path, args):
    sd = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': sd,
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss,
        'args': vars(args),
    }, save_path)
    print(f'  ↳ saved {save_path}')


# --------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------- #

def main():
    args = parse_args()
    if args.epochs is not None:
        args.num_steps = args.epochs
    setdefault_weights_by_phase(args)
    local_rank, world_size, is_main = setup_ddp()
    args.world_size = world_size
    args.local_rank = local_rank

    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device)
        if args.device.startswith('cuda'):
            idx = int(args.device.split(':')[-1]) if ':' in args.device else 0
            torch.cuda.set_device(idx)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        tee = Tee(os.path.join(args.output_dir, 'training.log'))
        sys.stdout = tee
        sys.stderr = tee
        print(f'Args: {vars(args)}')
        print(f'Device: {device}, phase={args.phase}, world_size={world_size}')

    csv_writer = None
    csv_file = None
    writer = None
    if is_main:
        csv_path = os.path.join(args.output_dir, 'train_log.csv')
        csv_file = open(csv_path, 'w', buffering=1)
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['step', 'lr', 'sn_weight', 'train_loss', 'val_loss', 'best_val_loss'])
        if args.use_tensorboard:
            writer = create_summary_writer(args.output_dir)

    train_ds, val_ds = build_dataset(args)
    if is_main:
        print(f'Train: {len(train_ds)}, Val: {len(val_ds)}')

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError('Empty dataloader.')

    model, renderer, loss_fn = build_model_renderer_loss(args, device)
    if args.phase == 2 and args.phase1_checkpoint:
        load_phase1_checkpoint(model, args.phase1_checkpoint, device)
    if is_main:
        print(f'Model params: {count_parameters(model)}')

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float('inf')
    last_val_loss = float('inf')

    if is_main:
        print('=' * 100)
        print('TRAINING (seqextract-style):')
        print('  - NO teacher forcing')
        print('  - NO supervised loss')
        print('  - raster (L1 + perceptual) + 7 aux losses')
        print(f'  - num_steps={args.num_steps}, eval_every={args.eval_every}, save_every={args.save_every}')
        print('  - stroke_num weight curriculum: 0 → 0.5')
        print('  - random init cursor from stroke')
        print('=' * 100)

    train_iter = iter(train_loader)
    data_epoch = 0
    running_loss = 0.0
    running_comp = {}
    running_count = 0

    for step in range(1, args.num_steps + 1):
        batch, train_iter, data_epoch = next_train_batch(
            train_loader, train_iter, train_sampler, data_epoch)

        lr = schedule_lr(args, step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        stroke_num_weight = schedule_stroke_num_weight(args, step)

        step_loss, step_comp = train_one_step(
            model, renderer, loss_fn, batch, optimizer, device,
            step, args, stroke_num_weight)
        if step_loss is None:
            continue

        running_loss += step_loss
        running_count += 1
        for k, v in step_comp.items():
            running_comp[k] = running_comp.get(k, 0.0) + v

        should_log = (step % args.eval_every == 0) or (step == args.num_steps)
        if not should_log:
            continue

        train_loss = running_loss / max(running_count, 1)
        train_comp = {k: v / max(running_count, 1) for k, v in running_comp.items()}
        running_loss = 0.0
        running_comp = {}
        running_count = 0

        val_loss, val_comp = validate(
            model, renderer, loss_fn, val_loader, device,
            step, args, stroke_num_weight)

        train_loss = all_reduce_mean(train_loss, world_size)
        val_loss = all_reduce_mean(val_loss, world_size)
        last_val_loss = val_loss
        train_comp = {k: all_reduce_mean(v, world_size) for k, v in train_comp.items()}
        val_comp = {k: all_reduce_mean(v, world_size) for k, v in val_comp.items()}

        if is_main:
            print(f'Step {step}/{args.num_steps}: lr={lr:.6g}, '
                  f'sn_weight={stroke_num_weight:.3g}, train={train_loss:.4g}, '
                  f'val={val_loss:.4g}, best={best:.4g}')
            print(f'  train: {train_comp}')
            print(f'  val: {val_comp}')
            csv_writer.writerow([step, lr, stroke_num_weight, train_loss, val_loss, best] +
                                [f'{v:.6g}' for v in train_comp.values()] +
                                [f'{v:.6g}' for v in val_comp.values()])
            if writer:
                writer.add_scalar('loss/train', train_loss, step)
                writer.add_scalar('loss/val', val_loss, step)
                writer.add_scalar('lr', lr, step)
                writer.add_scalar('sn_weight', stroke_num_weight, step)
                for k, v in train_comp.items():
                    writer.add_scalar(f'train/{k}', v, step)
                for k, v in val_comp.items():
                    writer.add_scalar(f'val/{k}', v, step)
                writer.flush()

            if val_loss < best:
                best = val_loss
                save_checkpoint(model, optimizer, step, val_loss, os.path.join(args.output_dir, 'model_best.pth'), args)
            if step % args.save_every == 0:
                save_checkpoint(model, optimizer, step, val_loss, os.path.join(args.output_dir, f'model_step_{step}.pth'), args)

    if is_main:
        save_checkpoint(model, optimizer, args.num_steps, last_val_loss, os.path.join(args.output_dir, 'model_final.pth'), args)
        print(f'\nBest val loss: {best:.4f}')
        csv_file.close()
        if writer:
            writer.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        tee.close()

    cleanup_ddp()


if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception as e:
        print(f'Fatal: {e}')
        traceback.print_exc()
        cleanup_ddp()
        sys.exit(1)

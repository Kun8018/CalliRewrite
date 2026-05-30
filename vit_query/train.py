#!/usr/bin/env python3
"""ViT 自回归笔画提取器 — phase 1 / phase 2 训练入口（v2，完全对齐 seq_extract）。

phase 1: 在 QuickDraw-clean 上学 stroke 序列；
  - 模型自闭环 rollout 全序列，全程可微；
  - loss = raster(L1+VGG) + 7 项辅助 + 监督 (pen/coord/param)
  - 支持 scheduled sampling: prob 从 1.0 退火到 0.0

phase 2: 在书法图像无监督训练；
  - rollout 同上，监督权重置 0；
  - 打开 smoothness / angle loss。

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

from model import ViTAutoregressiveExtractor7D, count_parameters
from neural_renderer import NeuralRasterizorStep
from dataset import QuickDrawCleanDataset, ImageOnlyDataset
from losses import CombinedRolloutLoss


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
    p = argparse.ArgumentParser(description='Train ViT stroke extractor (v2)')
    p.add_argument('--phase', type=int, default=1, choices=[1, 2])
    p.add_argument('--dataset_root', type=str, default='../seq_extract/datasets')
    p.add_argument('--data_dir', type=str, default=None,
                   help='phase2 image-only data dir')
    p.add_argument('--phase1_checkpoint', type=str, default=None)
    p.add_argument('--renderer_ckpt', type=str, required=True,
                   help='预训练 RasterUnit 权重 (.pth)，必须提供')
    p.add_argument('--freeze_renderer', action='store_true', default=True)
    p.add_argument('--no_freeze_renderer', dest='freeze_renderer', action='store_false')
    p.add_argument('--output_dir', type=str, default='./output_vit_v2')

    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--max_seq_len', type=int, default=48)  # 原版 phase1/2 都是 48
    p.add_argument('--patch_size', type=int, default=64)
    p.add_argument('--raster_size', type=int, default=128)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--num_heads', type=int, default=None)
    p.add_argument('--no_pretrained_vit', action='store_true',
                   help='不加载 torchvision ViT ImageNet 预训练权重（默认加载）')

    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--max_items_per_category', type=int, default=5000,
                   help='Phase1 每个 QuickDraw 类别取多少样本（原版 ~5万/类 → 这里 5k 已够）')
    p.add_argument('--cache_size', type=int, default=50000,
                   help='dataset 内存 cache 多少处理过的样本，0 关闭')

    # scheduled sampling
    p.add_argument('--ss_prob_start', type=float, default=1.0,
                   help='phase1 起始 teacher forcing 概率（1=纯 TF, 0=纯 free run）')
    p.add_argument('--ss_prob_end', type=float, default=0.0)

    # loss weights（对齐 hyper_parameters.py phase 1/2 默认）
    p.add_argument('--w_raster', type=float, default=1.0)
    p.add_argument('--w_stroke_num', type=float, default=0.5)
    p.add_argument('--w_smoothness', type=float, default=None,
                   help='默认 phase1=0, phase2=0.5')
    p.add_argument('--w_angle', type=float, default=None,
                   help='默认 phase1=0, phase2=1.0')
    p.add_argument('--w_outside', type=float, default=10.0)
    p.add_argument('--w_win_outside', type=float, default=10.0)
    p.add_argument('--w_early_pen', type=float, default=0.1)
    p.add_argument('--w_supervised', type=float, default=None,
                   help='默认 phase1=1.0, phase2=0.0')

    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--use_tensorboard', action='store_true')
    p.add_argument('--save_every', type=int, default=5)
    return p.parse_args()


def setdefault_weights_by_phase(args):
    if args.phase == 1:
        if args.w_smoothness is None: args.w_smoothness = 0.0
        if args.w_angle is None:      args.w_angle = 0.0
        if args.w_supervised is None: args.w_supervised = 1.0
    else:
        if args.w_smoothness is None: args.w_smoothness = 0.5
        if args.w_angle is None:      args.w_angle = 1.0
        if args.w_supervised is None: args.w_supervised = 0.0


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

    # phase 2
    if args.data_dir is None:
        candidates = [
            '../seq_extract/outputs/__new_train_phase_2',
            '../rl_finetune/data/train_data',
        ]
        for d in candidates:
            if os.path.exists(d):
                args.data_dir = d
                print(f'Using phase2 data dir: {d}')
                break
    if args.data_dir is None:
        raise ValueError('phase2 needs --data_dir with .png/.jpg images')
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

    loss_fn = CombinedRolloutLoss(
        raster_weight=args.w_raster,
        stroke_num_weight=args.w_stroke_num,
        smoothness_weight=args.w_smoothness,
        angle_weight=args.w_angle,
        outside_weight=args.w_outside,
        win_outside_weight=args.w_win_outside,
        early_pen_weight=args.w_early_pen,
        supervised_weight=args.w_supervised,
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


def schedule_ss_prob(args, epoch):
    if args.phase != 1 or args.epochs <= 1:
        return 0.0  # phase 2 不需要 GT 笔画 scheduled sampling
    frac = (epoch - 1) / (args.epochs - 1)
    return args.ss_prob_start + frac * (args.ss_prob_end - args.ss_prob_start)


# --------------------------------------------------------------------- #
# step
# --------------------------------------------------------------------- #

def run_step(model, renderer, loss_fn, batch, device, args, ss_prob):
    target_image = batch['target_image'].to(device, non_blocking=True)            # (N, 1, H, W) 1=BG
    target_stroke_img = batch['target_stroke_img'].to(device, non_blocking=True)  # (N, H, W) 1=stroke

    gt_strokes = batch.get('gt_strokes')
    gt_mask = batch.get('gt_mask')
    if gt_strokes is not None:
        gt_strokes = gt_strokes.to(device, non_blocking=True)
        gt_mask = gt_mask.to(device, non_blocking=True)

    # DDP 包装后 model(*) 会拦截 forward 做 gradient reduce；
    # forward 内部就是 rollout（见 model.py）
    rollout = model(
        target_image, renderer,
        seq_len=args.max_seq_len,
        gt_strokes=gt_strokes,
        scheduled_sampling_prob=ss_prob,
    )

    losses = loss_fn(rollout, target_stroke_img, args.image_size,
                     gt_strokes=gt_strokes, gt_mask=gt_mask)
    return losses, rollout


def train_epoch(model, renderer, loss_fn, loader, optimizer, device, epoch, args):
    model.train()
    if hasattr(renderer.raster_unit, 'eval') and args.freeze_renderer:
        renderer.raster_unit.eval()
    ss_prob = schedule_ss_prob(args, epoch)
    total = 0.0
    comp_acc = {}
    is_main = (getattr(args, 'local_rank', 0) == 0)
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train] ss={ss_prob:.2f}',
                disable=not is_main)
    for batch in pbar:
        optimizer.zero_grad()
        losses, _ = run_step(model, renderer, loss_fn, batch, device, args, ss_prob)
        loss = losses['total']
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += loss.item()
        for k, v in losses.items():
            if k == 'total':
                continue
            comp_acc[k] = comp_acc.get(k, 0.0) + float(v)
        if is_main:
            pbar.set_postfix(loss=loss.item())
    n = max(len(loader), 1)
    comp_acc = {k: v / n for k, v in comp_acc.items()}
    return total / n, comp_acc, ss_prob


@torch.no_grad()
def validate(model, renderer, loss_fn, loader, device, epoch, args):
    model.eval()
    total = 0.0
    comp_acc = {}
    is_main = (getattr(args, 'local_rank', 0) == 0)
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]', disable=not is_main)
    for batch in pbar:
        losses, _ = run_step(model, renderer, loss_fn, batch, device, args, 0.0)
        total += losses['total'].item()
        for k, v in losses.items():
            if k == 'total':
                continue
            comp_acc[k] = comp_acc.get(k, 0.0) + float(v)
        if is_main:
            pbar.set_postfix(loss=losses['total'].item())
    n = max(len(loader), 1)
    comp_acc = {k: v / n for k, v in comp_acc.items()}
    return total / n, comp_acc


def save_checkpoint(model, optim, epoch, loss, save_path, args):
    """save model.module.state_dict() if DDP, else model.state_dict()."""
    sd = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': sd,
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss,
        'phase': args.phase,
        'args': vars(args),
    }, save_path)
    print(f'  ↳ saved {save_path}')


# --------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------- #

def main():
    args = parse_args()
    setdefault_weights_by_phase(args)

    # 1) DDP 初始化（torchrun 自动设环境变量；单进程时返回单卡 stub）
    local_rank, world_size, is_main = setup_ddp()
    args.world_size = world_size
    args.local_rank = local_rank

    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    elif torch.cuda.is_available():
        # 单卡：尊重用户 --device cuda:N
        device = torch.device(args.device)
        if args.device.startswith('cuda'):
            idx = int(args.device.split(':')[-1]) if ':' in args.device else 0
            torch.cuda.set_device(idx)
    else:
        device = torch.device('cpu')

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        tee = Tee(os.path.join(args.output_dir, 'training.log'))
        sys.stdout = tee
        sys.stderr = tee
        print(f'Args: {vars(args)}')
        print(f'Device: {device}, phase={args.phase}, world_size={world_size}')

    # CSV / TB 只在 rank 0
    csv_writer = None
    csv_file = None
    writer = None
    if is_main:
        csv_path = os.path.join(args.output_dir, 'train_log.csv')
        csv_file = open(csv_path, 'w', buffering=1)
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'ss_prob', 'train_loss', 'val_loss', 'best_val_loss'])
        if args.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(args.output_dir, 'tensorboard')
                os.makedirs(tb_dir, exist_ok=True)
                writer = SummaryWriter(tb_dir)
                print(f'TensorBoard: {tb_dir}')
            except ImportError:
                print('TensorBoard not available')

    # 2) 数据
    train_ds, val_ds = build_dataset(args)
    if is_main:
        print(f'Train {len(train_ds)} / Val {len(val_ds)}')

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # 3) 模型 + DDP 包装
    model, renderer, loss_fn = build_model_renderer_loss(args, device)
    if args.phase == 2 and args.phase1_checkpoint:
        load_phase1_checkpoint(model, args.phase1_checkpoint, device)
    if is_main:
        print(f'Model: {count_parameters(model)}')

    if world_size > 1:
        # find_unused_parameters=True 因为 scheduled_sampling 等可能某些参数没走梯度
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    optim_ = optim.AdamW(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=args.epochs)
    best = float('inf')

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_comp, ss_prob = train_epoch(
            model, renderer, loss_fn, train_loader, optim_, device, epoch, args)
        val_loss, val_comp = validate(
            model, renderer, loss_fn, val_loader, device, epoch, args)
        scheduler.step()

        # 跨 rank 平均 loss 显示
        train_loss = all_reduce_mean(train_loss, world_size)
        val_loss = all_reduce_mean(val_loss, world_size)
        train_comp = {k: all_reduce_mean(v, world_size) for k, v in train_comp.items()}
        val_comp = {k: all_reduce_mean(v, world_size) for k, v in val_comp.items()}

        if is_main:
            print(f'Epoch {epoch}/{args.epochs}  ss={ss_prob:.2f}  '
                  f'train={train_loss:.4f}  val={val_loss:.4f}  best={best:.4f}')
            print('  train_comp:', {k: round(v, 4) for k, v in train_comp.items()})
            print('  val_comp:',   {k: round(v, 4) for k, v in val_comp.items()})

            csv_writer.writerow([epoch, ss_prob, train_loss, val_loss, best,
                                 *[round(v, 6) for v in train_comp.values()],
                                 *[round(v, 6) for v in val_comp.values()]])
            if writer:
                writer.add_scalar('loss/train', train_loss, epoch)
                writer.add_scalar('loss/val', val_loss, epoch)
                writer.add_scalar('ss_prob', ss_prob, epoch)
                for k, v in train_comp.items():
                    writer.add_scalar(f'train/{k}', v, epoch)
                for k, v in val_comp.items():
                    writer.add_scalar(f'val/{k}', v, epoch)
                writer.flush()

            if val_loss < best:
                best = val_loss
                save_checkpoint(model, optim_, epoch, val_loss,
                                os.path.join(args.output_dir, 'model_best.pth'), args)
            if epoch % args.save_every == 0:
                save_checkpoint(model, optim_, epoch, val_loss,
                                os.path.join(args.output_dir, f'model_epoch_{epoch}.pth'), args)

    if is_main:
        save_checkpoint(model, optim_, args.epochs, val_loss,
                        os.path.join(args.output_dir, 'model_final.pth'), args)
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
        print(f'Fatal error: {e}')
        traceback.print_exc()
        sys.exit(1)

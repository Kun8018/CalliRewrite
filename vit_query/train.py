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
    p.add_argument('--amp', action='store_true', default=False,
                   help='bf16 混合精度训练（5090 推荐开启，提速并省显存）')
    p.add_argument('--no_amp', dest='amp', action='store_false')
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--max_items_per_category', type=int, default=5000,
                   help='Phase1 每个 QuickDraw 类别取多少样本（原版 ~5万/类 → 这里 5k 已够）')
    p.add_argument('--cache_size', type=int, default=50000,
                   help='dataset 内存 cache 多少处理过的样本，0 关闭')

    # scheduled sampling (已废弃，保留参数以免老 shell 报错)
    p.add_argument('--ss_prob_start', type=float, default=0.0,
                   help='[已废弃] scheduled sampling 起始概率')
    p.add_argument('--ss_prob_end', type=float, default=0.0,
                   help='[已废弃] scheduled sampling 终点概率')
    p.add_argument('--teacher_forcing_prob', type=float, default=None,
                   help='phase1 训练时用 GT 当前笔推进 rollout 状态的概率；默认 phase1=1, phase2=0')
    p.add_argument('--teacher_forcing_end', type=float, default=None,
                   help='phase1 teacher forcing 线性衰减到的终值；不设则不衰减')
    p.add_argument('--teacher_forcing_decay_epochs', type=int, default=0,
                   help='phase1 teacher forcing 从起值衰减到终值所用 epoch 数；0 表示关闭衰减')
    p.add_argument('--teacher_forcing_warmup_epochs', type=int, default=0,
                   help='phase1 前多少个 epoch 保持 teacher forcing 起值不衰减')
    p.add_argument('--best_metric', type=str, default='val_tf',
                   choices=['val_free', 'val_tf'],
                   help='兼容旧参数；主流程会同时保存 model_best_tf/free.pth')
    p.add_argument('--viz_every', type=int, default=0,
                   help='每隔多少 epoch 保存固定样本的 TF100/Free 对比图；0 表示关闭')
    p.add_argument('--viz_category', type=str, default='duck',
                   help='phase1 可视化优先使用的 QuickDraw 类别；不存在则退回验证集首样本')
    p.add_argument('--viz_index', type=int, default=0,
                   help='可视化样本在类别文件或验证集中的索引')

    # 随机初始 cursor（关键：迫使模型必须看 target image）
    p.add_argument('--random_init_cursor', action='store_true', default=True,
                   help='训练时从 stroke 像素随机采初始 cursor（防止固化路径）')
    p.add_argument('--no_random_init_cursor', dest='random_init_cursor', action='store_false')
    p.add_argument('--use_gt_init_cursor', action='store_true', default=True,
                   help='phase1 有 GT 序列时使用数据里的真实起笔 cursor，对齐监督坐标')
    p.add_argument('--no_gt_init_cursor', dest='use_gt_init_cursor', action='store_false')
    p.add_argument('--init_cursor_low', type=float, default=0.2)
    p.add_argument('--init_cursor_high', type=float, default=0.8)

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
    p.add_argument('--early_pen_start_idx', type=int, default=0)
    p.add_argument('--early_pen_end_idx', type=int, default=None)
    p.add_argument('--normalize_pos_outside', action='store_true',
                   help='将 pos_outside 除以 image_size；默认关闭以对齐 seq_extract 原版')
    p.add_argument('--w_supervised', type=float, default=None,
                   help='默认 phase1=0.1, phase2=0.0')
    p.add_argument('--w_sup_pen', type=float, default=1.0)
    p.add_argument('--w_sup_coord', type=float, default=5.0)
    p.add_argument('--w_sup_param', type=float, default=1.0)
    p.add_argument('--w_sup_tail_pen', type=float, default=0.5)
    p.add_argument('--w_sup_pen_up', type=float, default=1.0,
                   help='监督 pen loss 中 GT pen-up 类别的额外权重')
    p.add_argument('--use_perceptual', action='store_true', default=True)
    p.add_argument('--no_perceptual', dest='use_perceptual', action='store_false',
                   help='关闭 VGG perceptual loss；phase1 数值不稳定时建议关闭')
    p.add_argument('--use_l1_raster', action='store_true', default=True)
    p.add_argument('--no_l1_raster', dest='use_l1_raster', action='store_false')
    p.add_argument('--fail_on_nonfinite', action='store_true', default=True,
                   help='loss/grad 出现 NaN/Inf 时立即报错并打印 loss 分量')
    p.add_argument('--skip_nonfinite', dest='fail_on_nonfinite', action='store_false',
                   help='遇到 NaN/Inf batch 时跳过该 batch（不推荐，只用于临时抢救长训练）')

    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--use_tensorboard', action='store_true')
    p.add_argument('--save_every', type=int, default=5)
    return p.parse_args()


def setdefault_weights_by_phase(args):
    if args.phase == 1:
        if args.w_smoothness is None: args.w_smoothness = 0.0
        if args.w_angle is None:      args.w_angle = 0.0
        # 监督权重降到 0.1：random_init_cursor 之后 GT cursor 路径不再与模型一致，
        # 强监督 coord/param 会拖累训练。只保留弱信号约束 pen state 节奏。
        if args.w_supervised is None: args.w_supervised = 0.1
        if args.teacher_forcing_prob is None: args.teacher_forcing_prob = 1.0
    else:
        if args.w_smoothness is None: args.w_smoothness = 0.5
        if args.w_angle is None:      args.w_angle = 1.0
        if args.w_supervised is None: args.w_supervised = 0.0
        if args.teacher_forcing_prob is None: args.teacher_forcing_prob = 0.0


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
    """Create a TensorBoard writer with a clear fallback path."""
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
            print('[warn] TensorBoard is not installed in this Python env.')
            print(f'[warn] Install with: {sys.executable} -m pip install tensorboard')
            print(f'[warn] Original import error: {torch_tb_error}')
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
        early_pen_start_idx=args.early_pen_start_idx,
        early_pen_end_idx=args.early_pen_end_idx,
        normalize_pos_outside=args.normalize_pos_outside,
        supervised_weight=args.w_supervised,
        sup_pen_weight=args.w_sup_pen,
        sup_coord_weight=args.w_sup_coord,
        sup_param_weight=args.w_sup_param,
        sup_tail_pen_weight=args.w_sup_tail_pen,
        sup_pen_up_weight=args.w_sup_pen_up,
        use_perceptual=args.use_perceptual,
        use_l1_raster=args.use_l1_raster,
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
    """scheduled sampling 已废弃，始终返回 0（保留供 CSV 列兼容）。"""
    return 0.0


def schedule_teacher_forcing_prob(args, epoch):
    """Phase1 先用 GT 对齐状态学习单步预测，再逐步切到闭环 rollout。"""
    start = float(args.teacher_forcing_prob or 0.0)
    if args.phase != 1 or args.teacher_forcing_end is None or args.teacher_forcing_decay_epochs <= 1:
        return start

    end = float(args.teacher_forcing_end)
    warmup = max(int(args.teacher_forcing_warmup_epochs), 0)
    if epoch <= warmup:
        return start
    decay_epoch = epoch - warmup
    progress = min(max((decay_epoch - 1) / float(args.teacher_forcing_decay_epochs - 1), 0.0), 1.0)
    return start + (end - start) * progress


def _tensor_mask_to_pil(mask: torch.Tensor) -> Image.Image:
    arr = mask.detach().float().cpu().clamp(0, 1).numpy()
    arr = ((1.0 - arr) * 255.0).astype('uint8')
    return Image.fromarray(arr, mode='L').convert('RGB')


def _draw_title(img: Image.Image, title: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, min(out.width, 420), 22], fill='white')
    draw.text((6, 5), title, fill=(0, 0, 0))
    return out


def _make_compare_panel(images):
    w, h = images[0].size
    panel = Image.new('RGB', (w * len(images), h), 'white')
    for i, img in enumerate(images):
        panel.paste(img.resize((w, h)), (i * w, 0))
    return panel


def _rollout_for_viz(model, renderer, batch, device, args, teacher_forcing_prob: float):
    losses, rollout = run_step(
        model, renderer, lambda *a, **k: {'total': torch.zeros((), device=device)},
        batch, device, args, 0.0, training=False,
        teacher_forcing_prob=teacher_forcing_prob,
    )
    return losses, rollout


def _pen_stats(seq: torch.Tensor) -> str:
    pen = seq.detach().float().cpu()[0, :, 0]
    pen_up = int((pen >= 0.5).sum().item())
    pen_down = int((pen < 0.5).sum().item())
    return f'len={pen.numel()} pen_down={pen_down} pen_up={pen_up} mean_pen_up={pen.mean().item():.4f}'


def _seq_to_thin_pil(seq: torch.Tensor, init_cursor: torch.Tensor,
                     image_size: int, title: str) -> Image.Image:
    import numpy as np

    strokes = seq.detach().float().cpu()[0].numpy().astype(np.float32)
    strokes[:, 0] = (strokes[:, 0] >= 0.5).astype(np.float32)
    cursor = init_cursor.detach().float().cpu()[0].numpy().astype(np.float32)
    img = generate_order_image(
        strokes,
        image_size=image_size,
        line_width=2,
        init_cursors=np.asarray([cursor], dtype=np.float32),
        round_lengths=np.asarray([len(strokes)], dtype=np.int64),
    )
    return _draw_title(img, title)


def build_viz_batch(args, val_ds):
    if args.phase != 1 or args.viz_every <= 0:
        return None, 'disabled'

    sample = None
    label = None
    npz_path = os.path.join(
        args.dataset_root, 'QuickDraw-clean', 'test', f'{args.viz_category}.npz')
    if os.path.exists(npz_path):
        try:
            import numpy as np
            from dataset import render_stroke3_tensor, stroke3_to_normalized_xy
            from dataset import quickdraw_stroke3_to_7d, pad_strokes

            data = np.load(npz_path, allow_pickle=True, encoding='latin1')
            strokes = data['stroke3'].tolist()
            if strokes:
                idx = min(max(int(args.viz_index), 0), len(strokes) - 1)
                stroke3 = np.asarray(strokes[idx], dtype=np.float32)
                target_image = render_stroke3_tensor(stroke3, args.image_size)
                target_stroke = 1.0 - target_image.squeeze(0)
                points = stroke3_to_normalized_xy(stroke3)
                gt, mask, seq_len = pad_strokes(
                    quickdraw_stroke3_to_7d(stroke3, args.image_size),
                    args.max_seq_len,
                )
                sample = {
                    'target_image': target_image,
                    'target_stroke_img': target_stroke,
                    'gt_strokes': torch.from_numpy(gt),
                    'gt_mask': torch.from_numpy(mask),
                    'init_cursor': torch.from_numpy(points[0].astype(np.float32)),
                    'seq_len': seq_len,
                }
                label = f'{args.viz_category}[{idx}]'
        except Exception as exc:
            label = f'{args.viz_category} unavailable: {exc}'

    if sample is None:
        sample = val_ds[min(max(int(args.viz_index), 0), len(val_ds) - 1)]
        label = label or f'val[{args.viz_index}]'

    batch = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
    return batch, label


@torch.no_grad()
def save_epoch_visualization(model, renderer, viz_batch, viz_label, device, epoch, args):
    if viz_batch is None or args.viz_every <= 0 or epoch % args.viz_every != 0:
        return

    model.eval()
    rollout_model = model.module if isinstance(model, DDP) else model
    out_dir = os.path.join(args.output_dir, 'viz_epoch')
    os.makedirs(out_dir, exist_ok=True)

    batch = {k: v.to(device, non_blocking=True) for k, v in viz_batch.items()}
    _, rollout_tf = _rollout_for_viz(rollout_model, renderer, batch, device, args, 1.0)
    _, rollout_free = _rollout_for_viz(rollout_model, renderer, batch, device, args, 0.0)

    original = _draw_title(_tensor_mask_to_pil(batch['target_stroke_img'][0]), f'Original {viz_label}')
    tf_img = _seq_to_thin_pil(rollout_tf['seq'], batch['init_cursor'], args.image_size, 'Generated TF100 Thin')
    free_img = _seq_to_thin_pil(rollout_free['seq'], batch['init_cursor'], args.image_size, 'Generated Free Thin')
    panel = _make_compare_panel([original, tf_img, free_img])
    image_path = os.path.join(out_dir, f'epoch_{epoch:04d}_compare.png')
    panel.save(image_path)

    tf_canvas = _draw_title(_tensor_mask_to_pil(rollout_tf['rendered'][0]), 'TF100 Soft Canvas')
    free_canvas = _draw_title(_tensor_mask_to_pil(rollout_free['rendered'][0]), 'Free Soft Canvas')
    canvas_panel = _make_compare_panel([original, tf_canvas, free_canvas])
    canvas_path = os.path.join(out_dir, f'epoch_{epoch:04d}_canvas_compare.png')
    canvas_panel.save(canvas_path)

    stats_path = os.path.join(out_dir, f'epoch_{epoch:04d}_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f'sample={viz_label}\n')
        f.write(f'tf100: {_pen_stats(rollout_tf["seq"])}\n')
        f.write(f'free:  {_pen_stats(rollout_free["seq"])}\n')
    print(f'  ↳ saved visualization {image_path}')


def sample_init_cursors_from_stroke(target_stroke_img: torch.Tensor,
                                    lo: float = 0.2,
                                    hi: float = 0.8,
                                    stroke_thresh: float = 0.5) -> torch.Tensor:
    """从 target 笔画像素随机采初始 cursor，对齐 seq_extract gen_init_cursors。

    target_stroke_img: (N, H, W) ∈ [0,1], 1=stroke
    Returns: (N, 2) cursor (x, y) ∈ [0, 1)
    """
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
        in_bounds = ((x_norm >= lo) & (x_norm <= hi) &
                     (y_norm >= lo) & (y_norm <= hi))
        if in_bounds.any():
            xs, ys = xs[in_bounds], ys[in_bounds]

        pick = torch.randint(0, xs.numel(), (1,), device=device).item()
        init_cursor[i, 0] = xs[pick].to(dtype) / W
        init_cursor[i, 1] = ys[pick].to(dtype) / H

    return init_cursor


# --------------------------------------------------------------------- #
# step
# --------------------------------------------------------------------- #

def run_step(model, renderer, loss_fn, batch, device, args, ss_prob,
             training: bool, teacher_forcing_prob: float = None):
    target_image = batch['target_image'].to(device, non_blocking=True)            # (N, 1, H, W) 1=BG
    target_stroke_img = batch['target_stroke_img'].to(device, non_blocking=True)  # (N, H, W) 1=stroke

    gt_strokes = batch.get('gt_strokes')
    gt_mask = batch.get('gt_mask')
    if gt_strokes is not None:
        gt_strokes = gt_strokes.to(device, non_blocking=True)
        gt_mask = gt_mask.to(device, non_blocking=True)

    gt_init_cursor = batch.get('init_cursor')
    if args.use_gt_init_cursor and gt_init_cursor is not None:
        # phase1 的 seq7 监督坐标是相对真实起笔点编码的，必须用同一个起点 rollout。
        init_cursor = gt_init_cursor.to(device, non_blocking=True)
    elif training and args.random_init_cursor:
        init_cursor = sample_init_cursors_from_stroke(
            target_stroke_img,
            lo=args.init_cursor_low,
            hi=args.init_cursor_high,
        )
    else:
        init_cursor = None

    use_amp = args.amp and device.type == 'cuda'
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        # DDP 包装后 model(*) 会拦截 forward 做 gradient reduce；
        # forward 内部就是 rollout（见 model.py）
        if teacher_forcing_prob is None:
            teacher_forcing_prob = args.teacher_forcing_prob if training else 0.0
        rollout = model(
            target_image, renderer,
            seq_len=args.max_seq_len,
            gt_strokes=gt_strokes,
            scheduled_sampling_prob=ss_prob,
            teacher_forcing_prob=teacher_forcing_prob,
            init_cursor=init_cursor,
        )
        losses = loss_fn(rollout, target_stroke_img, args.image_size,
                         gt_strokes=gt_strokes, gt_mask=gt_mask)
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
            return f'{name}: shape={tuple(grad.shape)}, first_bad_grad={bad_value}'
        current_max = grad.detach().abs().max().float().item()
        if current_max > max_abs:
            max_abs = current_max
            max_name = name
    if max_name is not None:
        return f'all individual grads finite; max_abs_grad={max_abs:.6g} at {max_name}'
    return 'no gradients'


def train_epoch(model, renderer, loss_fn, loader, optimizer, device, epoch, args):
    model.train()
    if hasattr(renderer.raster_unit, 'eval') and args.freeze_renderer:
        renderer.raster_unit.eval()
    ss_prob = schedule_ss_prob(args, epoch)
    tf_prob = schedule_teacher_forcing_prob(args, epoch)
    total = 0.0
    comp_acc = {}
    num_updates = 0
    is_main = (getattr(args, 'local_rank', 0) == 0)
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]',
                disable=not is_main)
    for step, batch in enumerate(pbar, start=1):
        optimizer.zero_grad()
        losses, _ = run_step(model, renderer, loss_fn, batch, device, args, ss_prob,
                             training=True, teacher_forcing_prob=tf_prob)
        loss = losses['total']
        if not torch.isfinite(loss):
            msg = (f'Non-finite train loss at epoch={epoch}, step={step}: '
                   f'{format_loss_components(losses)}')
            if args.fail_on_nonfinite:
                raise FloatingPointError(msg)
            if is_main:
                print(f'[skip] {msg}')
            continue
        loss.backward()
        if args.grad_clip > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip, error_if_nonfinite=False)
            if not torch.isfinite(grad_norm):
                msg = (f'Non-finite grad norm at epoch={epoch}, step={step}: '
                       f'grad_norm={grad_norm.item()}, '
                       f'bad_grad={find_first_nonfinite_grad(model)}, '
                       f'{format_loss_components(losses)}')
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
            pbar.set_postfix(loss=loss.item())
    if num_updates == 0:
        raise RuntimeError(
            f'No training batches were processed at epoch={epoch}. '
            'Check dataset path, max_items_per_category, world_size, batch_size, and drop_last.')
    comp_acc = {k: v / num_updates for k, v in comp_acc.items()}
    return total / num_updates, comp_acc, ss_prob, tf_prob


@torch.no_grad()
def validate(model, renderer, loss_fn, loader, device, epoch, args,
             teacher_forcing_prob: float = 0.0, tag: str = 'Val'):
    model.eval()
    total = 0.0
    comp_acc = {}
    num_batches = 0
    is_main = (getattr(args, 'local_rank', 0) == 0)
    pbar = tqdm(loader, desc=f'Epoch {epoch} [{tag}]', disable=not is_main)
    for step, batch in enumerate(pbar, start=1):
        losses, _ = run_step(model, renderer, loss_fn, batch, device, args, 0.0,
                             training=False, teacher_forcing_prob=teacher_forcing_prob)
        if not torch.isfinite(losses['total']):
            raise FloatingPointError(
                f'Non-finite val loss at epoch={epoch}, step={step}: '
                f'{format_loss_components(losses)}')
        total += losses['total'].item()
        num_batches += 1
        for k, v in losses.items():
            if k == 'total':
                continue
            comp_acc[k] = comp_acc.get(k, 0.0) + float(v)
        if is_main:
            pbar.set_postfix(loss=losses['total'].item())
    if num_batches == 0:
        raise RuntimeError(
            f'No validation batches were processed at epoch={epoch}. '
            'Check dataset path, val split/test files, world_size, and batch_size.')
    comp_acc = {k: v / num_batches for k, v in comp_acc.items()}
    return total / num_batches, comp_acc


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

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        tee = Tee(os.path.join(args.output_dir, 'training.log'))
        sys.stdout = tee
        sys.stderr = tee
        print(f'Args: {vars(args)}')
        print(f'Device: {device}, phase={args.phase}, world_size={world_size}, '
              f'amp={"bf16" if args.amp and device.type == "cuda" else "off"}')

    # CSV / TB 只在 rank 0
    csv_writer = None
    csv_file = None
    writer = None
    if is_main:
        csv_path = os.path.join(args.output_dir, 'train_log.csv')
        csv_file = open(csv_path, 'w', buffering=1)
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'epoch', 'ss_prob', 'tf_prob',
            'train_loss', 'val_tf100_loss', 'val_free_loss', 'selected_metric_loss',
        ])
        if args.use_tensorboard:
            writer = create_summary_writer(args.output_dir)

    # 2) 数据
    train_ds, val_ds = build_dataset(args)
    if is_main:
        print(f'Train {len(train_ds)} / Val {len(val_ds)}')
    viz_batch, viz_label = build_viz_batch(args, val_ds)
    if is_main and args.viz_every > 0:
        print(f'Visualization sample: {viz_label}')

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
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError(
            'Empty DataLoader: '
            f'train_ds={len(train_ds)}, val_ds={len(val_ds)}, '
            f'train_loader={len(train_loader)}, val_loader={len(val_loader)}, '
            f'batch_size={args.batch_size}, world_size={world_size}. '
            'For phase1, verify --dataset_root points to QuickDraw-clean and enough samples are loaded.')

    # 3) 模型 + DDP 包装
    model, renderer, loss_fn = build_model_renderer_loss(args, device)
    if args.phase == 2 and args.phase1_checkpoint:
        load_phase1_checkpoint(model, args.phase1_checkpoint, device)
    if is_main:
        print(f'Model: {count_parameters(model)}')

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    optim_ = optim.AdamW(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=args.epochs)
    best_tf = float('inf')
    best_free = float('inf')

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_comp, ss_prob, tf_prob = train_epoch(
            model, renderer, loss_fn, train_loader, optim_, device, epoch, args)
        val_tf_loss, val_tf_comp = validate(
            model, renderer, loss_fn, val_loader, device, epoch, args,
            teacher_forcing_prob=1.0, tag='ValTF100')
        val_free_loss, val_free_comp = validate(
            model, renderer, loss_fn, val_loader, device, epoch, args,
            teacher_forcing_prob=0.0, tag='ValFree')
        metric_loss = val_free_loss if args.best_metric == 'val_free' else val_tf_loss
        scheduler.step()

        # 跨 rank 平均 loss 显示
        train_loss = all_reduce_mean(train_loss, world_size)
        val_tf_loss = all_reduce_mean(val_tf_loss, world_size)
        val_free_loss = all_reduce_mean(val_free_loss, world_size)
        metric_loss = all_reduce_mean(metric_loss, world_size)
        train_comp = {k: all_reduce_mean(v, world_size) for k, v in train_comp.items()}
        val_tf_comp = {k: all_reduce_mean(v, world_size) for k, v in val_tf_comp.items()}
        val_free_comp = {k: all_reduce_mean(v, world_size) for k, v in val_free_comp.items()}

        if is_main:
            print(f'Epoch {epoch}/{args.epochs}  '
                  f'train={train_loss:.4f}  val_tf100={val_tf_loss:.4f}  '
                  f'val_free={val_free_loss:.4f}  best_tf={best_tf:.4f}  '
                  f'best_free={best_free:.4f}  '
                  f'tf_prob={tf_prob:.3f}')
            print('  train_comp:', {k: round(v, 4) for k, v in train_comp.items()})
            print('  val_tf100_comp:', {k: round(v, 4) for k, v in val_tf_comp.items()})
            print('  val_free_comp:', {k: round(v, 4) for k, v in val_free_comp.items()})

            csv_writer.writerow([epoch, ss_prob, tf_prob, train_loss,
                                 val_tf_loss, val_free_loss, metric_loss,
                                 *[round(v, 6) for v in train_comp.values()],
                                 *[round(v, 6) for v in val_tf_comp.values()],
                                 *[round(v, 6) for v in val_free_comp.values()]])
            if writer:
                writer.add_scalar('loss/train', train_loss, epoch)
                writer.add_scalar('loss/val_tf100', val_tf_loss, epoch)
                writer.add_scalar('loss/val_free', val_free_loss, epoch)
                writer.add_scalar('ss_prob', ss_prob, epoch)
                writer.add_scalar('tf_prob', tf_prob, epoch)
                for k, v in train_comp.items():
                    writer.add_scalar(f'train/{k}', v, epoch)
                for k, v in val_tf_comp.items():
                    writer.add_scalar(f'val_tf100/{k}', v, epoch)
                for k, v in val_free_comp.items():
                    writer.add_scalar(f'val_free/{k}', v, epoch)
                writer.flush()

            if val_tf_loss < best_tf:
                best_tf = val_tf_loss
                save_checkpoint(model, optim_, epoch, val_tf_loss,
                                os.path.join(args.output_dir, 'model_best_tf.pth'), args)
                # 兼容旧推理脚本默认 ckpt 名称：phase1 以 TF100 best 作为主模型。
                save_checkpoint(model, optim_, epoch, val_tf_loss,
                                os.path.join(args.output_dir, 'model_best.pth'), args)
            if val_free_loss < best_free:
                best_free = val_free_loss
                save_checkpoint(model, optim_, epoch, val_free_loss,
                                os.path.join(args.output_dir, 'model_best_free.pth'), args)
            if epoch % args.save_every == 0:
                save_checkpoint(model, optim_, epoch, metric_loss,
                                os.path.join(args.output_dir, f'model_epoch_{epoch}.pth'), args)
            save_epoch_visualization(model, renderer, viz_batch, viz_label, device, epoch, args)

    if is_main:
        save_checkpoint(model, optim_, args.epochs, metric_loss,
                        os.path.join(args.output_dir, 'model_final.pth'), args)
        print(f'\nBest val_tf100 loss: {best_tf:.4f}')
        print(f'Best val_free loss: {best_free:.4f}')
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
        cleanup_ddp()
        sys.exit(1)

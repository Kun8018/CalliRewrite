#!/usr/bin/env python3
"""离线预训练 NeuralRasterizor.RasterUnit。

输入参数空间：(N, 10) [x0, y0, x1, y1, x2, y2, r0, r2, w0, w2]，全部 ∈ [0, 1]
GT 渲染：PIL 画二次贝塞尔曲线 + 端点圆，width 用 r0/r2 线性插值。

训练完得到的 .pth 直接喂给 NeuralRasterizorStep(pretrained_path=...) 即可。

用法：
    python pretrain_renderer.py \
        --output_path output_renderer/raster_unit_pretrained.pth \
        --steps 100000 --batch_size 64
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from neural_renderer import RasterUnit


RASTER_SIZE = 128


def sample_quadratic_bezier_pixels(N: int, raster_size: int = RASTER_SIZE,
                                    rng: random.Random = None):
    """合成 N 个贝塞尔笔画 (params, gt_image)。"""
    if rng is None:
        rng = random
    params = np.zeros((N, 10), dtype=np.float32)
    images = np.zeros((N, raster_size, raster_size), dtype=np.float32)

    for i in range(N):
        # 在 [0.05, 0.95] 范围采点，避免完全贴边
        x0, y0 = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
        x2, y2 = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
        # control 点可以略微出框
        x1, y1 = rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)
        # width / radius 范围与原版 hps 一致：min_width=0.01，max=1.0
        r0 = rng.uniform(0.01, 0.18)
        r2 = rng.uniform(0.01, 0.18)
        w0 = rng.uniform(0.01, 0.18)
        w2 = rng.uniform(0.01, 0.18)

        params[i] = [x0, y0, x1, y1, x2, y2, r0, r2, w0, w2]

        # 渲染 GT：PIL 沿贝塞尔曲线 sample 64 个点，每个点画 width 为局部插值的圆
        img = Image.new('L', (raster_size, raster_size), 0)
        draw = ImageDraw.Draw(img)
        S = 64
        ts = np.linspace(0.0, 1.0, S)
        xs = (1 - ts) ** 2 * x0 + 2 * (1 - ts) * ts * x1 + ts ** 2 * x2
        ys = (1 - ts) ** 2 * y0 + 2 * (1 - ts) * ts * y1 + ts ** 2 * y2
        rs = (1 - ts) * r0 + ts * r2  # 半径线性插值
        ws = (1 - ts) * w0 + ts * w2  # width 同 r（原版基本只用 r）
        radii_px = (np.maximum(rs, ws) * raster_size).astype(np.float32)
        xs_px = xs * raster_size
        ys_px = ys * raster_size

        for j in range(S):
            r_px = max(1.0, float(radii_px[j]))
            x_px = float(xs_px[j])
            y_px = float(ys_px[j])
            draw.ellipse([x_px - r_px, y_px - r_px,
                          x_px + r_px, y_px + r_px], fill=255)

        arr = np.array(img, dtype=np.float32) / 255.0  # 1=stroke
        images[i] = arr

    return params, images


class BezierDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size: int, raster_size: int = RASTER_SIZE, seed: int = 0):
        self.batch_size = batch_size
        self.raster_size = raster_size
        self.seed = seed

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        seed = self.seed + (worker.id if worker else 0)
        rng = random.Random(seed)
        while True:
            params, images = sample_quadratic_bezier_pixels(
                self.batch_size, self.raster_size, rng)
            yield (torch.from_numpy(params), torch.from_numpy(images))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output_path', type=str, required=True)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--log_every', type=int, default=200)
    p.add_argument('--save_every', type=int, default=10000)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--resume', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    device = torch.device(args.device)
    model = RasterUnit(raster_size=RASTER_SIZE).to(device)
    if args.resume and os.path.isfile(args.resume):
        sd = torch.load(args.resume, map_location=device)
        if 'raster_unit' in sd:
            sd = sd['raster_unit']
        model.load_state_dict(sd)
        print(f'Resumed from {args.resume}')

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    ds = BezierDataset(batch_size=args.batch_size, raster_size=RASTER_SIZE)
    loader = torch.utils.data.DataLoader(ds, batch_size=None,
                                          num_workers=args.num_workers)
    it = iter(loader)

    model.train()
    t0 = time.time()
    ema_loss = None
    for step in range(1, args.steps + 1):
        params, gt = next(it)
        params = params.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        pred = model(params)  # (N, R, R), 1=stroke
        # L1 + binary cross entropy 双损失，更利于收敛到锐利边缘
        l1 = F.l1_loss(pred, gt)
        bce = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), gt)
        loss = l1 + 0.1 * bce

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f'step {step:6d}/{args.steps}  loss={loss.item():.4f}  ema={ema_loss:.4f}  '
                  f'l1={l1.item():.4f}  bce={bce.item():.4f}  ({dt:.1f}s)',
                  flush=True)

        if step % args.save_every == 0 or step == args.steps:
            ckpt = {'raster_unit': model.state_dict(),
                    'step': step, 'loss': ema_loss}
            torch.save(ckpt, args.output_path)
            print(f'  ↳ saved to {args.output_path}', flush=True)


if __name__ == '__main__':
    main()

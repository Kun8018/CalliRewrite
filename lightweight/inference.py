#!/usr/bin/env python3
"""Lightweight 推理脚本 — 适配 v2 可微 rollout。

逻辑：
1. 读图，预处理成 (1, 1, H, W) ∈ [0, 1], 1=BG;
2. model.rollout 自闭环 unroll；
3. 后处理：pen 二值化，截掉末尾连续抬笔；
4. 多轮（max_rounds）：每轮跑完用 find_undrawn_cursor 在 residual 找下一个起点，
   保留 canvas + hidden state，再 rollout。
5. 输出 seq_extract 兼容的 .npz + 可视化。
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from model import ResNetAutoregressiveExtractor7D
from neural_renderer import NeuralRasterizorStep
from diffable_state import init_rollout_state, RolloutState
from dataset import find_undrawn_cursor
from visualize import visualize_strokes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--input', type=str, required=True,
                   help='image path or directory')
    p.add_argument('--output_dir', type=str, default='./inference_output_lightweight')
    p.add_argument('--renderer_ckpt', type=str, required=True)
    p.add_argument('--image_size', type=int, default=None)
    p.add_argument('--max_seq_len', type=int, default=None)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--pen_threshold', type=float, default=0.5)
    p.add_argument('--max_consecutive_lifts', type=int, default=3)
    p.add_argument('--max_consecutive_downs', type=int, default=24,
                   help='连续落笔超过该步数时截断；0 表示关闭')
    p.add_argument('--max_rounds', type=int, default=4)
    p.add_argument('--init_cursor_strategy', choices=['center', 'stroke'], default='stroke',
                   help='第一轮起笔位置：center=图像中心，stroke=最大未绘制笔画区域')
    return p.parse_args()


def load_model(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get('args', {})

    def pick(name, default):
        v = getattr(args, name, None)
        return v if v is not None else saved_args.get(name, default)

    image_size = pick('image_size', 256)
    max_seq_len = pick('max_seq_len', 48)
    d_model = saved_args.get('d_model', 256)
    hidden_dim = saved_args.get('hidden_dim', 256)
    patch_size = saved_args.get('patch_size', 64)
    raster_size = saved_args.get('raster_size', 128)
    num_heads = saved_args.get('num_heads', None)

    model = ResNetAutoregressiveExtractor7D(
        image_size=image_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        patch_size=patch_size,
        raster_size=raster_size).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded {args.checkpoint}  (epoch={ckpt.get("epoch")})')

    renderer = NeuralRasterizorStep(raster_size=raster_size,
                                    pretrained_path=args.renderer_ckpt,
                                    freeze=True).to(device)
    renderer.eval()
    return model, renderer, image_size, max_seq_len


def preprocess(image_path, image_size):
    img = Image.open(image_path).convert('L')
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) 1=BG


def trim_trailing_lifts(strokes: np.ndarray, max_consec: int):
    """seq_extract 风格：连续 max_consec 次 pen=1 就截断（这一轮结束）。"""
    if max_consec <= 0:
        return strokes
    lifts = 0
    for i, s in enumerate(strokes):
        if s[0] >= 0.5:
            lifts += 1
            if lifts >= max_consec:
                return strokes[:i + 1]
        else:
            lifts = 0
    return strokes


def trim_long_pen_down_run(strokes: np.ndarray, max_consec: int):
    if max_consec <= 0:
        return strokes
    downs = 0
    for i, s in enumerate(strokes):
        if s[0] < 0.5:
            downs += 1
            if downs >= max_consec:
                return strokes[:i + 1]
        else:
            downs = 0
    return strokes


def make_state_with_cursor(cursor: np.ndarray, image_size: int, device) -> RolloutState:
    state = init_rollout_state(1, image_size, device)
    state.cursor = torch.from_numpy(cursor.astype(np.float32)).to(device).unsqueeze(0)
    return state


@torch.no_grad()
def infer_image(model, renderer, image_path, image_size, max_seq_len,
                pen_threshold=0.5, max_consec=3, max_rounds=4,
                init_cursor_strategy='stroke', max_consecutive_downs=24, device='cuda'):
    img_tensor = preprocess(image_path, image_size).to(device)  # (1, 1, H, W) 1=BG
    state = None
    hidden = None
    all_strokes = []
    init_cursors = []
    round_lengths = []

    target_np = (1.0 - img_tensor[0, 0]).cpu().numpy()  # (H, W) 1=stroke
    first_cursor = None
    if init_cursor_strategy == 'stroke':
        first_cursor = find_undrawn_cursor(target_np, np.zeros_like(target_np))
        if first_cursor is not None:
            state = make_state_with_cursor(first_cursor, image_size, device)

    for round_idx in range(max_rounds):
        if round_idx == 0:
            init_cursor_round = first_cursor if first_cursor is not None else np.array([0.5, 0.5], dtype=np.float32)
        else:
            init_cursor_round = state.cursor[0].cpu().numpy()

        out = model.rollout(img_tensor, renderer, seq_len=max_seq_len,
                            init_state=state, init_hidden=hidden,
                            detach_canvas_for_encoder=True)
        strokes = out['seq'][0].cpu().numpy().astype(np.float32)  # (T, 7)
        strokes[:, 0] = (strokes[:, 0] > pen_threshold).astype(np.float32)
        strokes = trim_trailing_lifts(strokes, max_consec)
        strokes = trim_long_pen_down_run(strokes, max_consecutive_downs)

        if len(strokes) == 0:
            break

        init_cursors.append(init_cursor_round.copy())
        round_lengths.append(len(strokes))
        all_strokes.append(strokes)

        # 更新 state 以便下一轮继续（canvas 累积）
        state = out['final_state']
        hidden = out['final_hidden']

        if round_idx + 1 >= max_rounds:
            break

        # 找下一个未画区域作为新起点
        canvas_np = state.canvas[0, 0].cpu().numpy()
        next_cursor = find_undrawn_cursor(target_np, canvas_np)
        if next_cursor is None:
            break
        # reset cursor / hidden / prev_stroke，保留 canvas / window
        state = RolloutState(
            cursor=torch.from_numpy(next_cursor).to(device).unsqueeze(0),
            canvas=state.canvas,
            prev_width=state.prev_width,
            prev_scaling=torch.ones_like(state.prev_scaling),
            prev_window_size=state.prev_window_size,
            prev_stroke=torch.zeros_like(state.prev_stroke),
            img_size=state.img_size,
        )
        hidden = None  # 新一轮 GRU hidden 重置

    if not all_strokes:
        strokes_final = np.array([[1.0, 0.5, 0.5, 0.0, 0.0, 0.1, 1.0]], dtype=np.float32)
        round_lengths = [1]
        init_cursors = [np.array([0.5, 0.5], dtype=np.float32)]
    else:
        strokes_final = np.concatenate(all_strokes, axis=0)

    return strokes_final, init_cursors, round_lengths


def save_seq_extract_npz(output_path, strokes_data, image_size,
                         init_cursors=None, round_lengths=None):
    if init_cursors is None:
        init_cursors = [[0.5, 0.5]]
    if round_lengths is None:
        round_lengths = [len(strokes_data)]
    init_width = float(strokes_data[0, 5]) if len(strokes_data) > 0 else 0.1
    np.savez_compressed(
        output_path,
        strokes_data=strokes_data.astype(np.float64),
        init_cursors=np.asarray(init_cursors, dtype=np.float32),
        image_size=np.array(image_size, dtype=np.int64),
        round_length=np.asarray(round_lengths, dtype=np.int64),
        init_width=np.array(init_width, dtype=np.float64),
    )


def process_one(model, renderer, image_path, output_dir, image_size, max_seq_len,
                args, device):
    strokes, init_cursors, round_lengths = infer_image(
        model, renderer, image_path, image_size, max_seq_len,
        pen_threshold=args.pen_threshold,
        max_consec=args.max_consecutive_lifts,
        max_rounds=args.max_rounds,
        init_cursor_strategy=args.init_cursor_strategy,
        max_consecutive_downs=args.max_consecutive_downs,
        device=device,
    )
    os.makedirs(output_dir, exist_ok=True)
    base = Path(image_path).stem
    npz_path = os.path.join(output_dir, f'{base}.npz')
    save_seq_extract_npz(npz_path, strokes, image_size, init_cursors, round_lengths)

    vis_dir = os.path.join(output_dir, base)
    visualize_strokes(npz_path, original_img_path=image_path, output_dir=vis_dir)

    print(f'{image_path}: {len(strokes)} strokes  rounds={round_lengths}')
    print(f'  -> {npz_path}')


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f'Using device: {device}')
    model, renderer, image_size, max_seq_len = load_model(args, device)

    if os.path.isfile(args.input):
        process_one(model, renderer, args.input, args.output_dir,
                    image_size, max_seq_len, args, device)
    elif os.path.isdir(args.input):
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        paths = sorted(p for p in Path(args.input).iterdir()
                       if p.suffix.lower() in exts)
        print(f'Found {len(paths)} images in {args.input}')
        for p in paths:
            try:
                process_one(model, renderer, str(p), args.output_dir,
                            image_size, max_seq_len, args, device)
            except Exception as e:
                print(f'  ERROR on {p}: {e}')
    else:
        print(f'Input not found: {args.input}')
        return
    print(f'\nResults: {args.output_dir}')


if __name__ == '__main__':
    main()

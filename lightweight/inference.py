#!/usr/bin/env python3
"""
轻量模型推理脚本，输出 seq_extract 兼容 npz。
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from model import StrokeTransformer, ResNetAutoregressiveExtractor7D
from dataset import initial_seq7_state, apply_seq7_step, find_undrawn_cursor
from visualize import visualize_strokes


def parse_args():
    parser = argparse.ArgumentParser(description='Lightweight Stroke Inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./inference_output')
    parser.add_argument('--arch', type=str, default=None, choices=['autoregressive', 'oneshot'])
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--max_seq_len', type=int, default=None)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--nhead', type=int, default=None)
    parser.add_argument('--num_decoder_layers', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pen_threshold', type=float, default=0.5)
    parser.add_argument('--max_consecutive_lifts', type=int, default=3)
    parser.add_argument('--max_rounds', type=int, default=4)
    return parser.parse_args()


def checkpoint_arg(checkpoint_args, cli_value, name, default):
    if cli_value is not None:
        return cli_value
    return checkpoint_args.get(name, default)


def load_model(checkpoint_path, device, arch=None, image_size=None, max_seq_len=None,
               d_model=None, nhead=None, num_decoder_layers=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get('args', {})
    arch = arch or checkpoint.get('arch') or checkpoint_args.get('arch', 'oneshot')
    image_size = checkpoint_arg(checkpoint_args, image_size, 'image_size', 256)
    max_seq_len = checkpoint_arg(checkpoint_args, max_seq_len, 'max_seq_len', 100)
    d_model = checkpoint_arg(checkpoint_args, d_model, 'd_model', 256)
    nhead = checkpoint_arg(checkpoint_args, nhead, 'nhead', 4)
    num_decoder_layers = checkpoint_arg(checkpoint_args, num_decoder_layers, 'num_decoder_layers', 3)

    if arch == 'autoregressive':
        model = ResNetAutoregressiveExtractor7D(
            image_size=image_size,
            max_seq_len=max_seq_len,
            d_model=d_model
        ).to(device)
    else:
        model = StrokeTransformer(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            max_seq_len=max_seq_len
        ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'Model loaded from {checkpoint_path} (epoch {checkpoint["epoch"]}, arch {arch})')
    return model, arch, image_size, max_seq_len


def preprocess_image(image_path, image_size=256, normalize='zero_one'):
    img = Image.open(image_path).convert('L')
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size))
    img_np = np.array(img, dtype=np.float32) / 255.0
    if normalize == 'minus_one_one':
        img_np = (img_np - 0.5) / 0.5
    return torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)


def preprocess_target_mask(image_path, image_size=256):
    return 1.0 - preprocess_image(image_path, image_size, normalize='zero_one')


def postprocess_strokes(strokes, pen_threshold=0.5, max_consecutive_lifts=3):
    strokes_np = strokes.cpu().numpy()
    strokes_np[:, 0] = (strokes_np[:, 0] > pen_threshold).astype(np.float32)

    if max_consecutive_lifts > 0:
        valid_len = len(strokes_np)
        lift_count = 0
        for i in range(len(strokes_np)):
            if strokes_np[i, 0] == 1:
                lift_count += 1
                if lift_count >= max_consecutive_lifts:
                    valid_len = i + 1
                    break
            else:
                lift_count = 0
        strokes_np = strokes_np[:valid_len]

    if len(strokes_np) == 0:
        strokes_np = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0]], dtype=np.float32)
    return strokes_np


def save_seq_extract_npz(output_path, strokes_data, image_size=256, init_cursors=None, round_lengths=None):
    if init_cursors is None:
        init_cursors = [[0.5, 0.5]]
    if round_lengths is None:
        round_lengths = [len(strokes_data)]
    init_width = np.array(float(strokes_data[0, 5]) if len(strokes_data) > 0 else 0.1, dtype=np.float64)

    np.savez_compressed(
        output_path,
        strokes_data=strokes_data.astype(np.float64),
        init_cursors=np.asarray(init_cursors, dtype=np.float32),
        image_size=np.array(image_size, dtype=np.int64),
        round_length=np.asarray(round_lengths, dtype=np.int64),
        init_width=init_width
    )


def save_inference_outputs(image_path, output_dir, strokes_data, image_size, init_cursors=None, round_lengths=None):
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem
    npz_path = os.path.join(output_dir, f'{base_name}.npz')
    save_seq_extract_npz(npz_path, strokes_data, image_size, init_cursors, round_lengths)

    vis_output_dir = os.path.join(output_dir, base_name)
    visualize_strokes(npz_path, original_img_path=image_path, output_dir=vis_output_dir)

    print(f'Processed {image_path}: {len(strokes_data)} strokes')
    print(f'  -> {npz_path}')
    print(f'  -> Visualization: {vis_output_dir}')
    return strokes_data


def infer_single_image_oneshot(model, image_path, output_dir, image_size=256, max_seq_len=100,
                               pen_threshold=0.5, max_consecutive_lifts=3, device='cuda'):
    img_tensor = preprocess_image(image_path, image_size, normalize='minus_one_one').to(device)
    with torch.no_grad():
        strokes = model.generate(img_tensor, max_len=max_seq_len)
    strokes_processed = postprocess_strokes(strokes, pen_threshold, max_consecutive_lifts)
    return save_inference_outputs(image_path, output_dir, strokes_processed, image_size)


def infer_single_image_autoregressive(model, image_path, output_dir, image_size=256, max_seq_len=100,
                                      pen_threshold=0.5, max_consecutive_lifts=3,
                                      max_rounds=4, device='cuda'):
    target_mask = preprocess_target_mask(image_path, image_size).to(device)
    state = initial_seq7_state(image_size)
    init_cursors = [state['cursor'].copy()]
    round_lengths = []
    current_round_len = 0
    consecutive_lifts = 0
    hidden = None
    prev_stroke = torch.zeros(1, 7, dtype=torch.float32, device=device)
    strokes_out = []

    with torch.no_grad():
        target_tokens, target_global = model.encode_target(target_mask)
        for step_idx in range(max_seq_len):
            canvas = torch.tensor(state['canvas'], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            cursor = torch.tensor(state['cursor'], dtype=torch.float32, device=device).unsqueeze(0)
            step = torch.tensor([[step_idx / max(max_seq_len, 1)]], dtype=torch.float32, device=device)
            output, hidden = model.forward_step(
                target_tokens, target_global, canvas, cursor, prev_stroke, step, hidden
            )
            stroke = output['seq'][0].detach().cpu().numpy().astype(np.float32)
            stroke[0] = 1.0 if stroke[0] > pen_threshold else 0.0
            strokes_out.append(stroke.copy())
            current_round_len += 1

            state = apply_seq7_step(state, stroke, image_size)
            prev_stroke = torch.tensor(stroke, dtype=torch.float32, device=device).unsqueeze(0)

            if stroke[0] == 1.0:
                consecutive_lifts += 1
            else:
                consecutive_lifts = 0

            if max_consecutive_lifts > 0 and consecutive_lifts >= max_consecutive_lifts:
                round_lengths.append(current_round_len)
                if len(round_lengths) >= max_rounds:
                    break
                target_np = target_mask[0, 0].detach().cpu().numpy()
                next_cursor = find_undrawn_cursor(target_np, state['canvas'])
                if next_cursor is None:
                    break
                state['cursor'] = next_cursor
                state['prev_stroke'] = np.zeros(7, dtype=np.float32)
                prev_stroke = torch.zeros(1, 7, dtype=torch.float32, device=device)
                hidden = None
                init_cursors.append(next_cursor.copy())
                current_round_len = 0
                consecutive_lifts = 0

    if current_round_len > 0 and (not round_lengths or sum(round_lengths) < len(strokes_out)):
        round_lengths.append(current_round_len)
    if not strokes_out:
        strokes_out = [np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0], dtype=np.float32)]
        round_lengths = [1]

    strokes_processed = np.asarray(strokes_out, dtype=np.float32)
    return save_inference_outputs(image_path, output_dir, strokes_processed, image_size, init_cursors, round_lengths)


def infer_single_image(model, image_path, output_dir, arch='autoregressive', image_size=256, max_seq_len=100,
                       pen_threshold=0.5, max_consecutive_lifts=3, max_rounds=4, device='cuda'):
    if arch == 'autoregressive':
        return infer_single_image_autoregressive(
            model, image_path, output_dir,
            image_size=image_size,
            max_seq_len=max_seq_len,
            pen_threshold=pen_threshold,
            max_consecutive_lifts=max_consecutive_lifts,
            max_rounds=max_rounds,
            device=device
        )
    return infer_single_image_oneshot(
        model, image_path, output_dir,
        image_size=image_size,
        max_seq_len=max_seq_len,
        pen_threshold=pen_threshold,
        max_consecutive_lifts=max_consecutive_lifts,
        device=device
    )


def infer_directory(model, input_dir, output_dir, arch='autoregressive', image_size=256, max_seq_len=100,
                    pen_threshold=0.5, max_consecutive_lifts=3, max_rounds=4, device='cuda'):
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(input_dir).glob(f'*{ext}'))
        image_paths.extend(Path(input_dir).glob(f'*{ext.upper()}'))

    image_paths = sorted(image_paths)
    print(f'Found {len(image_paths)} images in {input_dir}')

    all_strokes = []
    for img_path in image_paths:
        try:
            strokes = infer_single_image(
                model, str(img_path), output_dir,
                arch=arch,
                image_size=image_size,
                max_seq_len=max_seq_len,
                pen_threshold=pen_threshold,
                max_consecutive_lifts=max_consecutive_lifts,
                max_rounds=max_rounds,
                device=device
            )
            all_strokes.append((str(img_path), strokes))
        except Exception as e:
            print(f'Error processing {img_path}: {e}')
    return all_strokes


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f'Using device: {device}')
    model, arch, image_size, max_seq_len = load_model(
        args.checkpoint,
        device,
        arch=args.arch,
        image_size=args.image_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers
    )

    if os.path.isfile(args.input):
        print(f'Processing single image: {args.input}')
        infer_single_image(
            model, args.input, args.output_dir,
            arch=arch,
            image_size=image_size,
            max_seq_len=max_seq_len,
            pen_threshold=args.pen_threshold,
            max_consecutive_lifts=args.max_consecutive_lifts,
            max_rounds=args.max_rounds,
            device=device
        )
    elif os.path.isdir(args.input):
        print(f'Processing directory: {args.input}')
        infer_directory(
            model, args.input, args.output_dir,
            arch=arch,
            image_size=image_size,
            max_seq_len=max_seq_len,
            pen_threshold=args.pen_threshold,
            max_consecutive_lifts=args.max_consecutive_lifts,
            max_rounds=args.max_rounds,
            device=device
        )
    else:
        print(f'Input not found: {args.input}')
        return

    print(f'\nResults saved to {args.output_dir}')


if __name__ == '__main__':
    main()

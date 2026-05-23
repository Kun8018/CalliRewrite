#!/usr/bin/env python3
"""
ViT + 彩色标注笔画 推理脚本
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from model import ViTColorTrajectoryExtractor7D, ViTDualTrajectoryExtractor7D
from dataset import extract_red_mask


def load_model(checkpoint_path, device, mode='rgb',
               img_size=224, seq_len=100, embed_dim=192):
    """加载模型"""
    if mode == 'rgb':
        model = ViTColorTrajectoryExtractor7D(
            img_size=img_size,
            seq_len=seq_len,
            embed_dim=embed_dim
        ).to(device)
    else:
        model = ViTDualTrajectoryExtractor7D(
            img_size=img_size,
            seq_len=seq_len,
            embed_dim=embed_dim
        ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f'Model loaded from {checkpoint_path} (epoch {checkpoint["epoch"]})')
    return model


def preprocess_rgb(img_path, img_size=224):
    """预处理 RGB 图片"""
    img = Image.open(img_path).convert('RGB')
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
    return torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)


def preprocess_dual(img_path, img_size=224):
    """预处理为灰度+mask"""
    img = Image.open(img_path).convert('RGB')
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size))
    img_np = np.array(img)

    gray = np.mean(img_np, axis=2, keepdims=True).astype(np.float32) / 255.0
    mask = extract_red_mask(img_np)

    gray_tensor = torch.tensor(gray.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.tensor(mask[np.newaxis, np.newaxis, :, :], dtype=torch.float32)

    return gray_tensor, mask_tensor


def postprocess_strokes(strokes, pen_threshold=0.5):
    """后处理"""
    strokes_np = strokes.cpu().numpy()
    strokes_np[:, 0] = (strokes_np[:, 0] > pen_threshold).astype(np.float32)

    valid_len = len(strokes_np)
    move_count = 0
    for i in range(len(strokes_np)):
        if strokes_np[i, 0] == 1:
            move_count += 1
            if move_count >= 3:
                valid_len = i + 1
                break
        else:
            move_count = 0

    strokes_np = strokes_np[:valid_len]

    if len(strokes_np) == 0:
        strokes_np = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0]], dtype=np.float32)

    return strokes_np


def save_seq_extract_npz(output_path, strokes_data, img_size=224):
    """保存兼容格式"""
    init_cursors = np.array([[0.5, 0.5]], dtype=np.float32)
    round_length = np.array([len(strokes_data)], dtype=np.int32)
    init_width = float(strokes_data[0, 5]) if len(strokes_data) > 0 else 0.1

    np.savez_compressed(
        output_path,
        strokes_data=strokes_data.astype(np.float32),
        init_cursors=init_cursors,
        image_size=img_size,
        round_length=round_length,
        init_width=init_width
    )


def parse_args():
    parser = argparse.ArgumentParser(description='ViT Color Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重')
    parser.add_argument('--input', type=str, required=True, help='输入图片/目录')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='输出目录')
    parser.add_argument('--mode', type=str, default='rgb', choices=['rgb', 'dual'],
                        help='输入模式')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--pen_threshold', type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f'Using device: {device}')

    model = load_model(
        args.checkpoint, device, mode=args.mode,
        img_size=args.img_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim
    )

    # 处理输入
    if os.path.isfile(args.input):
        input_files = [args.input]
    else:
        input_files = []
        for f in sorted(os.listdir(args.input)):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                input_files.append(os.path.join(args.input, f))

    os.makedirs(args.output_dir, exist_ok=True)

    for img_path in input_files:
        try:
            if args.mode == 'rgb':
                img_tensor = preprocess_rgb(img_path, args.img_size).to(device)
                with torch.no_grad():
                    strokes = model(img_tensor)[0]
            else:
                gray_tensor, mask_tensor = preprocess_dual(img_path, args.img_size)
                gray_tensor = gray_tensor.to(device)
                mask_tensor = mask_tensor.to(device)
                with torch.no_grad():
                    strokes = model(gray_tensor, mask_tensor)[0]

            strokes_processed = postprocess_strokes(strokes, args.pen_threshold)
            base_name = Path(img_path).stem
            npz_path = os.path.join(args.output_dir, f'{base_name}.npz')
            save_seq_extract_npz(npz_path, strokes_processed, args.img_size)

            print(f'Processed: {img_path}')
            print(f'  Strokes: {len(strokes_processed)}')
            print(f'  Output: {npz_path}')

        except Exception as e:
            print(f'Error processing {img_path}: {e}')
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

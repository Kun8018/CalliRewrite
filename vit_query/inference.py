#!/usr/bin/env python3
"""
ViT + Trajectory Queries 推理脚本
生成兼容二阶段的 npz 文件
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from model import ViTTrajectoryExtractor7D


def load_model(checkpoint_path, device, img_size=224, seq_len=100, embed_dim=192):
    """加载训练好的模型"""
    model = ViTTrajectoryExtractor7D(
        img_size=img_size,
        seq_len=seq_len,
        embed_dim=embed_dim
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f'Model loaded from {checkpoint_path} (epoch {checkpoint["epoch"]})')
    return model


def preprocess_image(image_path, img_size=224):
    """预处理输入图像"""
    img = Image.open(image_path).convert('L')
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size))

    # ViT 常用 [0, 1] 归一化
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    return img_tensor


def postprocess_strokes(strokes, pen_threshold=0.5):
    """后处理生成的笔画序列"""
    strokes_np = strokes.cpu().numpy()

    # 二值化 pen_state
    strokes_np[:, 0] = (strokes_np[:, 0] > pen_threshold).astype(np.float32)

    # 简单的启发式截断（连续多个 move 就停止）
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
    """保存为 seq_extract 兼容的 npz 格式"""
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


def infer_single_image(model, image_path, output_dir, img_size=224, seq_len=100,
                       pen_threshold=0.5, device='cuda'):
    """推理单张图像"""
    img_tensor = preprocess_image(image_path, img_size).to(device)

    with torch.no_grad():
        strokes = model(img_tensor)[0]  # (seq_len, 7)

    strokes_processed = postprocess_strokes(strokes, pen_threshold)

    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem
    npz_path = os.path.join(output_dir, f'{base_name}.npz')
    save_seq_extract_npz(npz_path, strokes_processed, img_size)

    print(f'Processed {image_path}: {len(strokes_processed)} strokes → {npz_path}')
    return strokes_processed


def infer_directory(model, input_dir, output_dir, img_size=224, seq_len=100,
                    pen_threshold=0.5, device='cuda'):
    """推理整个目录"""
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
                img_size=img_size, seq_len=seq_len,
                pen_threshold=pen_threshold, device=device
            )
            all_strokes.append((str(img_path), strokes))
        except Exception as e:
            print(f'Error processing {img_path}: {e}')

    return all_strokes


def parse_args():
    parser = argparse.ArgumentParser(description='ViT Trajectory Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重')
    parser.add_argument('--input', type=str, required=True, help='输入图像/目录')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='输出目录')
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
        args.checkpoint, device,
        img_size=args.img_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim
    )

    if os.path.isfile(args.input):
        infer_single_image(model, args.input, args.output_dir,
                           img_size=args.img_size, seq_len=args.seq_len,
                           pen_threshold=args.pen_threshold, device=device)
    elif os.path.isdir(args.input):
        infer_directory(model, args.input, args.output_dir,
                        img_size=args.img_size, seq_len=args.seq_len,
                        pen_threshold=args.pen_threshold, device=device)
    else:
        print(f'Input not found: {args.input}')
        return

    print(f'\nResults saved to {args.output_dir}')


if __name__ == '__main__':
    main()

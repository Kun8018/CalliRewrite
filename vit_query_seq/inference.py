#!/usr/bin/env python3
"""
ViT + 渐进式图片输入 推理脚本
输入: img_00.png ~ img_09.png (或任意数量)
输出: 兼容二阶段的 npz
"""
import os
import argparse
import re
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from model import ViTSeqTrajectoryExtractor7D
from visualize import visualize_strokes


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def load_model(checkpoint_path, device, img_size=224, num_images=10,
               seq_len=100, embed_dim=192):
    """加载模型"""
    model = ViTSeqTrajectoryExtractor7D(
        img_size=img_size,
        num_images=num_images,
        seq_len=seq_len,
        embed_dim=embed_dim
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f'Model loaded from {checkpoint_path} (epoch {checkpoint["epoch"]})')
    return model


def load_image_sequence(input_path, num_images=10, img_size=224):
    """
    加载图片序列

    input_path: 单张图片或目录
    """
    if os.path.isfile(input_path):
        # 单张图片: 复制 num_images 份
        img = Image.open(input_path).convert('L')
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size))
        img_np = np.array(img, dtype=np.float32) / 255.0
        images = [img_np] * num_images
    else:
        # 目录: 排序加载
        image_files = []
        for f in sorted(os.listdir(input_path), key=natural_sort_key):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(input_path, f))

        # 不足的话补最后一张
        while len(image_files) < num_images:
            image_files.append(image_files[-1] if image_files else input_path)

        # 加载图片
        images = []
        for img_path in image_files[:num_images]:
            img = Image.open(img_path).convert('L')
            if img.size != (img_size, img_size):
                img = img.resize((img_size, img_size))
            img_np = np.array(img, dtype=np.float32) / 255.0
            images.append(img_np)

    # 堆叠: (num_images, 1, H, W)
    images_np = np.stack(images, axis=0)
    images_np = np.expand_dims(images_np, axis=1)

    return torch.tensor(images_np, dtype=torch.float32)


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
    parser = argparse.ArgumentParser(description='ViT Sequential Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重')
    parser.add_argument('--input', type=str, required=True, help='输入图片/目录')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='输出目录')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_images', type=int, default=10, help='图片序列长度')
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
        num_images=args.num_images,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim
    )

    # 加载图片序列
    images_tensor = load_image_sequence(
        args.input,
        num_images=args.num_images,
        img_size=args.img_size
    ).unsqueeze(0).to(device)  # (1, num_images, 1, 224, 224)

    # 推理
    with torch.no_grad():
        strokes = model(images_tensor)[0]  # (seq_len, 7)

    # 后处理
    strokes_processed = postprocess_strokes(strokes, args.pen_threshold)

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = Path(args.input).stem if os.path.isfile(args.input) else 'output'
    npz_path = os.path.join(args.output_dir, f'{base_name}.npz')
    save_seq_extract_npz(npz_path, strokes_processed, args.img_size)

    # 确定原始图片路径（用于 compare 图）
    original_img_path = None
    if os.path.isfile(args.input):
        original_img_path = args.input
    else:
        # 目录的话，取第一张图
        image_files = []
        for f in sorted(os.listdir(args.input), key=natural_sort_key):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(args.input, f))
        if image_files:
            original_img_path = image_files[0]

    # 生成可视化（order/color/compare 图）
    vis_output_dir = os.path.join(args.output_dir, base_name)
    visualize_strokes(npz_path, original_img_path=original_img_path, output_dir=vis_output_dir)

    print(f'\nProcessed: {args.input}')
    print(f'Strokes: {len(strokes_processed)}')
    print(f'Output: {npz_path}')
    print(f'Visualization: {vis_output_dir}')


if __name__ == '__main__':
    main()

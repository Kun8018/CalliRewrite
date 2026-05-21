#!/usr/bin/env python3
"""
推理脚本：使用训练好的模型生成笔画序列
输出与 seq_extract 兼容的 npz 文件，可直接用于二阶段
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from model import StrokeTransformer


def parse_args():
    parser = argparse.ArgumentParser(description='Stroke Transformer Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像路径或目录')
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                        help='输出目录')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--max_seq_len', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pen_threshold', type=float, default=0.5,
                        help='pen_state 二值化阈值')
    return parser.parse_args()


def load_model(checkpoint_path, device, d_model=256, nhead=4, num_decoder_layers=3, max_seq_len=100):
    """加载训练好的模型"""
    model = StrokeTransformer(
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        max_seq_len=max_seq_len
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f'Model loaded from {checkpoint_path} (epoch {checkpoint["epoch"]})')
    return model


def preprocess_image(image_path, image_size=256):
    """预处理输入图像"""
    img = Image.open(image_path).convert('L')  # 灰度图
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size))

    # 归一化到 [-1, 1]
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5

    # 转换为 tensor: (1, 1, H, W)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    return img_tensor


def postprocess_strokes(strokes, pen_threshold=0.5):
    """
    后处理生成的笔画序列
    strokes: (seq_len, 7) - [pen_state, x1, y1, x2, y2, r, s]
    """
    strokes_np = strokes.cpu().numpy()

    # 二值化 pen_state
    strokes_np[:, 0] = (strokes_np[:, 0] > pen_threshold).astype(np.float32)

    # 移除连续的移动指令，提前终止
    # 找到有效的笔画长度
    valid_len = len(strokes_np)

    # 简单策略：如果出现连续多个 pen_state=1 就截断
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

    # 确保至少有一个笔画
    if len(strokes_np) == 0:
        strokes_np = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0]], dtype=np.float32)

    return strokes_np


def save_seq_extract_npz(output_path, strokes_data, image_size=256):
    """
    保存为与 seq_extract 兼容的 npz 格式
    包含二阶段需要的所有字段
    """
    # 构造完整数据
    # 注意：这里的 init_cursors, round_length, init_width 是简化版本
    # 实际 seq_extract 输出的这些字段更复杂，但二阶段主要使用 strokes_data

    # 简单的初始光标位置
    init_cursors = np.array([[0.5, 0.5]], dtype=np.float32)

    # 每轮步数（简化为整个序列长度）
    round_length = np.array([len(strokes_data)], dtype=np.int32)

    # 初始宽度（从第一个笔画获取）
    init_width = float(strokes_data[0, 5]) if len(strokes_data) > 0 else 0.1

    np.savez_compressed(
        output_path,
        strokes_data=strokes_data.astype(np.float32),
        init_cursors=init_cursors,
        image_size=image_size,
        round_length=round_length,
        init_width=init_width
    )


def infer_single_image(model, image_path, output_dir, image_size=256, max_seq_len=100, pen_threshold=0.5, device='cuda'):
    """推理单张图像"""
    # 预处理
    img_tensor = preprocess_image(image_path, image_size).to(device)

    # 推理
    with torch.no_grad():
        strokes = model.generate(img_tensor, max_len=max_seq_len)

    # 后处理
    strokes_processed = postprocess_strokes(strokes, pen_threshold)

    # 准备输出
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem

    # 保存 npz（兼容 seq_extract 格式）
    npz_path = os.path.join(output_dir, f'{base_name}.npz')
    save_seq_extract_npz(npz_path, strokes_processed, image_size)

    # 同时保存原始笔画数据（用于调试）
    raw_path = os.path.join(output_dir, f'{base_name}_raw.npy')
    np.save(raw_path, strokes.cpu().numpy())

    print(f'Processed {image_path}: {len(strokes_processed)} strokes')
    print(f'  -> {npz_path}')

    return strokes_processed


def infer_directory(model, input_dir, output_dir, image_size=256, max_seq_len=100, pen_threshold=0.5, device='cuda'):
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
                image_size=image_size,
                max_seq_len=max_seq_len,
                pen_threshold=pen_threshold,
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

    # 加载模型
    model = load_model(
        args.checkpoint, device,
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        max_seq_len=args.max_seq_len
    )

    # 推理
    if os.path.isfile(args.input):
        print(f'Processing single image: {args.input}')
        infer_single_image(
            model, args.input, args.output_dir,
            image_size=args.image_size,
            max_seq_len=args.max_seq_len,
            pen_threshold=args.pen_threshold,
            device=device
        )
    elif os.path.isdir(args.input):
        print(f'Processing directory: {args.input}')
        infer_directory(
            model, args.input, args.output_dir,
            image_size=args.image_size,
            max_seq_len=args.max_seq_len,
            pen_threshold=args.pen_threshold,
            device=device
        )
    else:
        print(f'Input not found: {args.input}')
        return

    print(f'\nResults saved to {args.output_dir}')


if __name__ == '__main__':
    main()

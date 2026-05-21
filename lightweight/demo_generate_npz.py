#!/usr/bin/env python3
"""
演示脚本：生成 npz 文件 (使用随机权重，仅作格式演示)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image
from model import StrokeTransformer
from inference import preprocess_image, postprocess_strokes, save_seq_extract_npz


def demo_generate():
    """演示生成 npz"""
    print('=' * 60)
    print('Demo: Generate npz files (untrained model)')
    print('=' * 60)

    # 创建模型 (随机权重)
    model = StrokeTransformer(
        d_model=256,
        nhead=4,
        num_decoder_layers=3,
        max_seq_len=100
    )
    model.eval()

    print(f'\nModel created (random weights)')

    # 找测试图像
    test_images = []
    possible_dirs = [
        '../seq_extract/outputs/__new_train_phase_2',
        '../rl_finetune/data/train_data',
        '../rl_finetune/data/test_data'
    ]

    for d in possible_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.png'):
                    test_images.append(os.path.join(d, f))
                    if len(test_images) >= 3:
                        break
        if len(test_images) >= 3:
            break

    if not test_images:
        print('\nNo test images found, creating dummy image...')
        # 创建 dummy 图像
        os.makedirs('demo_output', exist_ok=True)
        dummy_img = Image.new('L', (256, 256), 255)
        dummy_path = 'demo_output/dummy.png'
        dummy_img.save(dummy_path)
        test_images = [dummy_path]

    print(f'\nFound {len(test_images)} test images')

    # 输出目录
    output_dir = 'demo_output'
    os.makedirs(output_dir, exist_ok=True)

    # 处理每张图像
    for img_path in test_images:
        print(f'\nProcessing: {img_path}')

        # 预处理
        img_tensor = preprocess_image(img_path, image_size=256)

        # 生成笔画
        with torch.no_grad():
            strokes = model.generate(img_tensor, max_len=50)

        # 后处理
        strokes_processed = postprocess_strokes(strokes, pen_threshold=0.5)

        # 保存 npz
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        npz_path = os.path.join(output_dir, f'{base_name}.npz')
        save_seq_extract_npz(npz_path, strokes_processed, image_size=256)

        print(f'  Saved: {npz_path}')
        print(f'  Strokes: {len(strokes_processed)}')

        # 显示前几个笔画
        print(f'  Sample strokes:')
        for i in range(min(3, len(strokes_processed))):
            s = strokes_processed[i]
            ps = 'MOVE' if s[0] > 0.5 else 'DRAW'
            print(f'    [{i}] {ps}: (x1={s[1]:.3f}, y1={s[2]:.3f}) -> (x2={s[3]:.3f}, y2={s[4]:.3f}), r={s[5]:.3f}')

    # 验证生成的 npz
    print('\n' + '=' * 60)
    print('Verifying generated npz files...')
    print('=' * 60)

    for f in os.listdir(output_dir):
        if f.endswith('.npz'):
            npz_path = os.path.join(output_dir, f)
            print(f'\n{npz_path}:')

            data = np.load(npz_path, allow_pickle=True, encoding='latin1')
            print(f'  Keys: {list(data.keys())}')
            print(f'  strokes_data shape: {data["strokes_data"].shape}')
            print(f'  image_size: {data["image_size"]}')

    print('\n' + '=' * 60)
    print(f'✓ Demo complete! Outputs in: {os.path.abspath(output_dir)}')
    print('=' * 60)


if __name__ == '__main__':
    demo_generate()

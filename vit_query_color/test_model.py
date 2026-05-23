#!/usr/bin/env python3
"""
测试 ViT + 彩色标注笔画
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import (
    ViTColorTrajectoryExtractor7D,
    ViTDualTrajectoryExtractor7D,
    count_parameters
)


def test_rgb_model():
    """测试 RGB 输入模型"""
    print('=' * 60)
    print('Testing RGB Input Model')
    print('=' * 60)

    model = ViTColorTrajectoryExtractor7D(
        img_size=224,
        seq_len=100,
        embed_dim=192
    )

    print(f'\nModel: {count_parameters(model)}')

    x = torch.randn(2, 3, 224, 224)
    print(f'\nInput shape: {x.shape}')

    out = model(x)
    print(f'Output shape: {out.shape}')

    pen_state = out[..., 0]
    coords = out[..., 1:5]
    params = out[..., 5:7]

    print(f'\nPen state range: [{pen_state.min():.3f}, {pen_state.max():.3f}]')
    print(f'Coords range: [{coords.min():.3f}, {coords.max():.3f}]')
    print(f'Params range: [{params.min():.3f}, {params.max():.3f}]')

    assert out.shape == (2, 100, 7)
    assert (pen_state >= 0).all() and (pen_state <= 1).all()
    assert (coords >= -1).all() and (coords <= 1).all()
    assert (params >= 0).all() and (params <= 1).all()

    print('\n✓ RGB model OK!')


def test_dual_model():
    """测试双输入模型"""
    print('\n' + '=' * 60)
    print('Testing Dual Input Model (Gray + Red Mask)')
    print('=' * 60)

    model = ViTDualTrajectoryExtractor7D(
        img_size=224,
        seq_len=100,
        embed_dim=192
    )

    print(f'\nModel: {count_parameters(model)}')

    gray = torch.randn(2, 1, 224, 224)
    mask = torch.randn(2, 1, 224, 224)
    print(f'\nInput gray: {gray.shape}, mask: {mask.shape}')

    out = model(gray, mask)
    print(f'Output shape: {out.shape}')

    pen_state = out[..., 0]
    coords = out[..., 1:5]
    params = out[..., 5:7]

    print(f'\nPen state range: [{pen_state.min():.3f}, {pen_state.max():.3f}]')
    print(f'Coords range: [{coords.min():.3f}, {coords.max():.3f}]')
    print(f'Params range: [{params.min():.3f}, {params.max():.3f}]')

    assert out.shape == (2, 100, 7)
    assert (pen_state >= 0).all() and (pen_state <= 1).all()
    assert (coords >= -1).all() and (coords <= 1).all()
    assert (params >= 0).all() and (params <= 1).all()

    print('\n✓ Dual model OK!')


def test_mask_extraction():
    """测试红色mask提取"""
    print('\n' + '=' * 60)
    print('Testing Red Mask Extraction')
    print('=' * 60)

    from dataset import extract_red_mask
    import numpy as np

    # 创建一个测试RGB图
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    # 加一个红方块
    rgb[25:75, 25:75, 0] = 255
    # 加一些灰色
    rgb[10:20, 10:20, :] = 128

    mask = extract_red_mask(rgb)
    print(f'Input shape: {rgb.shape}')
    print(f'Mask shape: {mask.shape}')
    print(f'Mask sum: {mask.sum()}')

    # 验证mask里有值
    assert mask.sum() > 0, 'Mask should have non-zero values'

    print('\n✓ Mask extraction OK!')


def main():
    print('Testing ViT + Color Stroke models...\n')

    try:
        test_mask_extraction()
        test_rgb_model()
        test_dual_model()

        print('\n' + '=' * 60)
        print('✓ All tests passed!')
        print('=' * 60)
        return 0

    except Exception as e:
        print(f'\n✗ Test failed: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

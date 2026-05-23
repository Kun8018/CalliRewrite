#!/usr/bin/env python3
"""
测试 ViT + 渐进式图片序列
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import ViTSeqTrajectoryExtractor7D, count_parameters


def test_model():
    """测试多图输入模型"""
    print('=' * 60)
    print('Testing ViT + Sequential Images')
    print('=' * 60)

    model = ViTSeqTrajectoryExtractor7D(
        img_size=224,
        num_images=10,
        seq_len=100,
        embed_dim=192
    )

    print(f'\nModel: {count_parameters(model)}')

    # 测试输入: (batch, num_images, 1, 224, 224)
    x = torch.randn(2, 10, 1, 224, 224)
    print(f'\nInput shape: {x.shape}')

    out = model(x)
    print(f'Output shape: {out.shape}')

    pen_state = out[..., 0]
    coords = out[..., 1:5]
    params = out[..., 5:7]

    print(f'\nPen state range: [{pen_state.min():.3f}, {pen_state.max():.3f}]')
    print(f'Coords range: [{coords.min():.3f}, {coords.max():.3f}]')
    print(f'Params range: [{params.min():.3f}, {params.max():.3f}]')

    assert out.shape == (2, 100, 7), f'Expected (2, 100, 7), got {out.shape}'
    assert (pen_state >= 0).all() and (pen_state <= 1).all()
    assert (coords >= -1).all() and (coords <= 1).all()
    assert (params >= 0).all() and (params <= 1).all()

    print('\n✓ Model OK!')


def test_encoder_only():
    """测试编码器部分"""
    print('\n' + '=' * 60)
    print('Testing Multi-Image Encoder Only')
    print('=' * 60)

    from model import MultiImageViTEncoder

    encoder = MultiImageViTEncoder(
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=192,
        num_images=10,
        depth=6,
        num_heads=4
    )

    print(f'\nEncoder params: {count_parameters(encoder)}')

    x = torch.randn(2, 10, 1, 224, 224)
    print(f'\nInput shape: {x.shape}')

    features = encoder(x)
    print(f'Features shape: {features.shape}')

    assert features.shape == (2, 10 * encoder.num_patches, 192), 'Unexpected shape'

    print('\n✓ Encoder OK!')


def main():
    print('Testing ViT + Sequential Images...\n')

    try:
        test_encoder_only()
        test_model()

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

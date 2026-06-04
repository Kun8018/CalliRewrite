#!/usr/bin/env python3
"""
测试 ViT + Trajectory Queries 模型
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import ViTTrajectoryExtractor, ViTTrajectoryExtractor7D, count_parameters


def test_model_2d():
    """测试 2D 密集点输出模型"""
    print('=' * 60)
    print('Testing ViT Trajectory Extractor (2D points)')
    print('=' * 60)

    model = ViTTrajectoryExtractor(
        img_size=224,
        num_points=100,
        embed_dim=192
    )

    print(f'\nModel: {count_parameters(model)}')

    x = torch.randn(2, 1, 224, 224)
    print(f'\nInput shape: {x.shape}')

    out = model(x)
    print(f'Output shape: {out.shape}')
    print(f'Output range: [{out.min():.3f}, {out.max():.3f}]')

    assert out.shape == (2, 100, 2), f'Expected (2, 100, 2), got {out.shape}'
    assert (out >= 0).all() and (out <= 1).all(), 'Output should be in [0, 1]'

    print('\n✓ 2D model OK!')


def test_model_7d():
    """测试 7D 序列输出模型"""
    print('\n' + '=' * 60)
    print('Testing ViT Trajectory Extractor (7D seq)')
    print('=' * 60)

    model = ViTTrajectoryExtractor7D(
        img_size=224,
        seq_len=100,
        embed_dim=192
    )

    print(f'\nModel: {count_parameters(model)}')

    x = torch.randn(2, 1, 224, 224)
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
    assert (pen_state >= 0).all() and (pen_state <= 1).all(), 'pen_state should be in [0,1]'
    assert (coords >= -1).all() and (coords <= 1).all(), 'coords should be in [-1,1]'
    assert (params >= 0).all() and (params <= 1).all(), 'params should be in [0,1]'

    print('\n✓ 7D model OK!')


def test_backbone():
    """测试 ViT backbone"""
    print('\n' + '=' * 60)
    print('Testing ViT Backbone')
    print('=' * 60)

    from model import ViTTinyBackbone

    backbone = ViTTinyBackbone(
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=192
    )

    print(f'\nBackbone params: {count_parameters(backbone)}')

    x = torch.randn(2, 1, 224, 224)
    features = backbone.forward_features(x)
    print(f'\nInput shape: {x.shape}')
    print(f'Features shape: {features.shape}')

    assert features.shape == (2, backbone.num_patches, backbone.embed_dim), 'Unexpected feature shape'

    print('\n✓ Backbone OK!')


def main():
    print('Testing ViT + Trajectory Queries models...\n')

    try:
        test_backbone()
        test_model_2d()
        test_model_7d()

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

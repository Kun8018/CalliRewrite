#!/usr/bin/env python3
"""
简单测试脚本：验证模型能正常前向传播
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from model import StrokeTransformer, ResNetAutoregressiveExtractor7D, count_parameters
from dataset import StrokeDataset, initial_seq7_state, make_autoregressive_item


def test_model_forward():
    """测试模型前向传播"""
    print('=' * 60)
    print('Testing Model Forward Pass')
    print('=' * 60)

    # 创建模型
    model = StrokeTransformer(
        d_model=256,
        nhead=4,
        num_decoder_layers=3,
        max_seq_len=100
    )

    print(f'\nModel created: {count_parameters(model)}')

    # 测试输入
    batch_size = 2
    image = torch.randn(batch_size, 1, 256, 256)
    target_seq = torch.randn(batch_size, 50, 7)  # 7维格式

    print(f'\nInput image shape: {image.shape}')
    print(f'Target sequence shape: {target_seq.shape}')

    # 训练模式前向 (teacher forcing 可能输出与 target 相同长度)
    output = model(image, target_seq, teacher_forcing_ratio=1.0)
    print(f'\nTraining output shape (with teacher forcing): {output.shape}')
    assert output.shape == (batch_size, 50, 7), f'Expected (2, 50, 7), got {output.shape}'

    # 无 teacher forcing 时输出 max_seq_len 长度
    output = model(image, target_seq, teacher_forcing_ratio=0.0)
    print(f'Training output shape (without teacher forcing): {output.shape}')
    assert output.shape == (batch_size, 100, 7), f'Expected (2, 100, 7), got {output.shape}'

    # 检查输出范围
    pen_state = output[..., 0]
    coords = output[..., 1:5]
    params = output[..., 5:7]

    print(f'\nOutput ranges:')
    print(f'  pen_state: [{pen_state.min():.3f}, {pen_state.max():.3f}] (should be [0, 1])')
    print(f'  coords: [{coords.min():.3f}, {coords.max():.3f}] (should be [-1, 1])')
    print(f'  params: [{params.min():.3f}, {params.max():.3f}] (should be [0, 1])')

    assert (pen_state >= 0).all() and (pen_state <= 1).all()
    assert (coords >= -1).all() and (coords <= 1).all()
    assert (params >= 0).all() and (params <= 1).all()

    print('\n✓ Training forward pass OK!')

    # 测试生成模式
    print('\n' + '=' * 60)
    print('Testing Generation Mode')
    print('=' * 60)

    single_image = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        generated = model.generate(single_image, max_len=30)

    print(f'\nGenerated strokes shape: {generated.shape}')
    assert generated.shape == (30, 7), f'Expected (30, 7), got {generated.shape}'

    print(f'\nGenerated sample (first 5 strokes):')
    for i in range(min(5, len(generated))):
        stroke = generated[i]
        ps = 'MOVE' if stroke[0] > 0.5 else 'DRAW'
        print(f'  [{i}] {ps}: (x1={stroke[1]:.3f}, y1={stroke[2]:.3f}) -> (x2={stroke[3]:.3f}, y2={stroke[4]:.3f}), r={stroke[5]:.3f}, s={stroke[6]:.3f}')

    print('\n✓ Generation mode OK!')


def test_dataset():
    """测试数据集加载"""
    print('\n' + '=' * 60)
    print('Testing Dataset')
    print('=' * 60)

    # 检查是否有可用数据
    data_dirs = [
        '../seq_extract/outputs/__new_train_phase_2',
        '../rl_finetune/data/train_data',
    ]

    found_data = False
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f'\nTrying data dir: {data_dir}')

            # 尝试创建数据集 (可能只有 npz 没有 png)
            try:
                dataset = StrokeDataset(
                    data_dir=data_dir,
                    image_size=256,
                    max_seq_len=100
                )
                print(f'Dataset size: {len(dataset)}')

                if len(dataset) > 0:
                    sample = dataset[0]
                    print(f'\nSample:')
                    print(f'  image shape: {sample["image"].shape}')
                    print(f'  strokes shape: {sample["strokes"].shape}')
                    print(f'  mask shape: {sample["mask"].shape}')
                    print(f'  seq_len: {sample["seq_len"]}')

                    found_data = True
                    print('\n✓ Dataset loading OK!')
                    break
            except Exception as e:
                print(f'Warning: {e}')

    if not found_data:
        print('\n⚠ No complete data found (need paired png+npz).')
        print('  This is OK - the model code is still valid.')
        print('  You can use QuickDraw data or your own data for training.')


def test_loss():
    """测试损失函数"""
    print('\n' + '=' * 60)
    print('Testing Loss Function')
    print('=' * 60)

    # 复制损失函数定义，避免导入 wandb
    import torch.nn as nn

    class StrokeLoss(nn.Module):
        def __init__(self, weight_pen=1.0, weight_coord=1.0, weight_param=1.0):
            super().__init__()
            self.weight_pen = weight_pen
            self.weight_coord = weight_coord
            self.weight_param = weight_param
            self.bce_loss = nn.BCELoss(reduction='none')
            self.l1_loss = nn.L1Loss(reduction='none')

        def forward(self, predictions, targets, mask=None):
            if mask is None:
                mask = torch.ones_like(targets[..., 0])
            pen_pred = predictions[..., 0]
            pen_target = targets[..., 0]
            pen_loss = self.bce_loss(pen_pred, pen_target)
            pen_loss = (pen_loss * mask).sum() / mask.sum()
            coord_pred = predictions[..., 1:5]
            coord_target = targets[..., 1:5]
            coord_loss = self.l1_loss(coord_pred, coord_target)
            coord_loss = (coord_loss.mean(dim=-1) * mask).sum() / mask.sum()
            param_pred = predictions[..., 5:7]
            param_target = targets[..., 5:7]
            param_loss = self.l1_loss(param_pred, param_target)
            param_loss = (param_loss.mean(dim=-1) * mask).sum() / mask.sum()
            total_loss = self.weight_pen * pen_loss + self.weight_coord * coord_loss + self.weight_param * param_loss
            return {'total': total_loss, 'pen': pen_loss, 'coord': coord_loss, 'param': param_loss}

    criterion = StrokeLoss()

    batch_size = 2
    seq_len = 50

    # 模拟预测和目标
    predictions = torch.rand(batch_size, seq_len, 7)
    predictions[..., 0] = torch.sigmoid(predictions[..., 0])  # pen_state [0,1]
    predictions[..., 1:5] = torch.tanh(predictions[..., 1:5])  # coords [-1,1]
    predictions[..., 5:7] = torch.sigmoid(predictions[..., 5:7])  # params [0,1]

    targets = torch.rand(batch_size, seq_len, 7)
    targets[..., 0] = (targets[..., 0] > 0.5).float()  # binary pen_state

    mask = torch.ones(batch_size, seq_len)
    mask[:, -10:] = 0  # 最后10个位置mask掉

    losses = criterion(predictions, targets, mask)

    print(f'\nLoss values:')
    print(f'  total: {losses["total"].item():.4f}')
    print(f'  pen: {losses["pen"].item():.4f}')
    print(f'  coord: {losses["coord"].item():.4f}')
    print(f'  param: {losses["param"].item():.4f}')

    assert not torch.isnan(losses['total'])
    assert losses['total'] >= 0

    print('\n✓ Loss function OK!')


def test_autoregressive_model():
    """测试 autoregressive 模型前向传播"""
    print('\n' + '=' * 60)
    print('Testing Autoregressive Model (ResNet)')
    print('=' * 60)

    model = ResNetAutoregressiveExtractor7D(
        image_size=256,
        max_seq_len=100,
        d_model=256
    )
    print(f'\nModel created: {count_parameters(model)}')

    B = 2
    image = torch.randn(B, 1, 256, 256)
    target_mask = 1.0 - image

    print(f'\nInput target_mask shape: {target_mask.shape}')

    # 测试 encode_target
    target_tokens, target_global = model.encode_target(target_mask)
    print(f'\nEncode target:')
    print(f'  target_tokens shape: {target_tokens.shape}')
    print(f'  target_global shape: {target_global.shape}')

    # 测试 forward_step
    canvas = torch.zeros(B, 1, 256, 256)
    cursor = torch.tensor([[0.5, 0.5]], dtype=torch.float32).repeat(B, 1)
    prev_stroke = torch.zeros(B, 7, dtype=torch.float32)
    step = torch.tensor([[0.0]], dtype=torch.float32).repeat(B, 1)
    hidden = None

    output, hidden = model.forward_step(
        target_tokens, target_global, canvas, cursor, prev_stroke, step, hidden
    )
    print(f'\nForward step (single):')
    print(f'  seq shape: {output["seq"].shape}')
    print(f'  pen_logits shape: {output["pen_logits"].shape}')
    print(f'  hidden shape: {hidden.shape}')

    # 测试输出范围
    seq = output['seq']
    pen_state = seq[..., 0]
    coords = seq[..., 1:5]
    params = seq[..., 5:7]
    print(f'\nStep output ranges:')
    print(f'  pen_state (after sigmoid): [{pen_state.min():.3f}, {pen_state.max():.3f}]')
    print(f'  coords (after tanh): [{coords.min():.3f}, {coords.max():.3f}]')
    print(f'  params (after sigmoid): [{params.min():.3f}, {params.max():.3f}]')

    # 测试 teacher forcing
    chunk_len = 8
    canvases = torch.zeros(B, chunk_len, 1, 256, 256)
    cursors = torch.zeros(B, chunk_len, 2)
    prev_strokes = torch.zeros(B, chunk_len, 7)
    step_indices = torch.rand(B, chunk_len, 1)

    output = model.forward_teacher_forcing(target_mask, canvases, cursors, prev_strokes, step_indices)
    print(f'\nTeacher forcing output:')
    print(f'  seq shape: {output["seq"].shape}')
    print(f'  pen_logits shape: {output["pen_logits"].shape}')

    assert output['seq'].shape == (B, chunk_len, 7), f'Expected (B, 8, 7), got {output["seq"].shape}'
    assert output['pen_logits'].shape == (B, chunk_len), f'Expected (B, 8), got {output["pen_logits"].shape}'

    print('\n✓ Autoregressive model OK!')


def main():
    print('Testing Lightweight Stroke Extractor...\n')

    try:
        test_model_forward()
        test_autoregressive_model()
        test_loss()
        test_dataset()

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

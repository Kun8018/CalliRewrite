#!/usr/bin/env python3
"""
分析一阶段输出的 npy/npz 文件内容
"""
import os
import sys
import numpy as np

def analyze_seq_extract_npz(npz_path):
    """分析 seq_extract 阶段保存的 npz 文件"""
    print(f"\n{'='*60}")
    print(f"分析 seq_extract 输出的 npz 文件: {npz_path}")
    print(f"{'='*60}")

    data = np.load(npz_path, encoding='latin1', allow_pickle=True)

    print("\n文件包含的键:")
    for key in data.keys():
        print(f"  - {key}")

    strokes_data = data['strokes_data']
    init_cursors = data['init_cursors']
    image_size = data['image_size']
    round_length = data['round_length']
    init_width = data['init_width']

    print(f"\n1. strokes_data 形状: {strokes_data.shape}")
    print(f"   dtype: {strokes_data.dtype}")
    print(f"   数据格式: [pen_state, x1, y1, x2, y2, radius, scaling]")
    print(f"\n   前 5 个stroke示例:")
    for i in range(min(5, len(strokes_data))):
        ps, x1, y1, x2, y2, r, s = strokes_data[i]
        print(f"     [{i}] 笔状态={ps:1.0f}, ({x1:.3f},{y1:.3f})->({x2:.3f},{y2:.3f}), r={r:.3f}, s={s:.3f}")

    # 统计笔状态
    draw_count = np.sum(strokes_data[:, 0] == 0)
    move_count = np.sum(strokes_data[:, 0] == 1)
    print(f"\n   统计: {draw_count} 个绘画 stroke, {move_count} 个移动 stroke")

    print(f"\n2. init_cursors: {init_cursors}")
    print(f"   形状: {init_cursors.shape if hasattr(init_cursors, 'shape') else type(init_cursors)}")
    print(f"   说明: 每个 round 的初始光标位置")

    print(f"\n3. image_size: {image_size}")
    print(f"   说明: 输入图像大小")

    print(f"\n4. round_length: {round_length}")
    print(f"   说明: 每轮采样的步数")
    print(f"   总步数: {sum(round_length)}")

    print(f"\n5. init_width: {init_width}")
    print(f"   说明: 初始笔画宽度")

    return strokes_data, init_cursors, image_size, round_length, init_width

def analyze_rl_finetune_npy(npy_path):
    """分析 rl_finetune 阶段保存的 npy 文件"""
    print(f"\n{'='*60}")
    print(f"分析 rl_finetune 输出的 npy 文件: {npy_path}")
    print(f"{'='*60}")

    data = np.load(npy_path)

    print(f"\n数据形状: {data.shape}")
    print(f"dtype: {data.dtype}")
    print(f"\n数据格式: [pen_state, x, y, r, ...] (可能包括更多列)")

    print(f"\n前 10 个点示例:")
    for i in range(min(10, len(data))):
        ps, x, y, r = data[i, :4]
        print(f"  [{i}] 笔状态={ps:1.0f}, (x={x:.3f}, y={y:.3f}), r={r:.3f}")

    # 统计
    draw_count = np.sum(data[:, 0] == 0)
    new_stroke_count = np.sum(data[:, 0] == 1)
    print(f"\n统计: {draw_count} 个绘画点, {new_stroke_count} 个新笔画开始")

    # 分割笔画
    stroke_indices = np.where(data[:, 0] == 1)[0]
    stroke_indices = np.append(stroke_indices, len(data))
    strokes = []
    for i in range(1, len(stroke_indices)):
        stroke = data[stroke_indices[i-1]:stroke_indices[i]]
        if len(stroke) > 1:
            strokes.append(stroke)

    print(f"\n共有 {len(strokes)} 个完整笔画")
    print(f"笔画长度统计:")
    for i, stroke in enumerate(strokes[:5]):
        print(f"  笔画 {i}: {len(stroke)} 个点, r均值={stroke[:, 3].mean():.3f}")

    return data

def find_example_files(base_dir):
    """在目录中查找示例文件"""
    npz_files = []
    npy_files = []

    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.npz'):
                npz_files.append(os.path.join(root, f))
            elif f.endswith('.npy'):
                npy_files.append(os.path.join(root, f))

    return npz_files, npy_files

def main():
    # 查找示例文件
    npz_files, npy_files = find_example_files('seq_extract')
    npz_files2, npy_files2 = find_example_files('rl_finetune')

    npz_files.extend(npz_files2)
    npy_files.extend(npy_files2)

    print(f"找到 {len(npz_files)} 个 npz 文件, {len(npy_files)} 个 npy 文件")

    # 分析第一个 npz
    if npz_files:
        analyze_seq_extract_npz(npz_files[0])

    # 分析第一个 npy
    if npy_files:
        analyze_rl_finetune_npy(npy_files[0])

    # 如果有命令行参数，直接分析指定文件
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.exists(target):
            if target.endswith('.npz'):
                analyze_seq_extract_npz(target)
            elif target.endswith('.npy'):
                analyze_rl_finetune_npy(target)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
检查 seq_extract 兼容 npz 的结构、坐标范围和可视化风险。
"""
import argparse
import numpy as np


def print_array_info(data):
    print('--- NPZ 数据文件核检 ---')
    print(f'Keys: {list(data.keys())}')
    for key in data.keys():
        value = data[key]
        print(f'键名: {key}, 形状: {value.shape}, 类型: {value.dtype}')
        if value.ndim == 0:
            print(f'  value: {value.item()}')
        elif value.ndim == 1 and len(value) <= 20:
            print(f'  values: {value}')
    print()


def check_required_keys(data):
    required = ['strokes_data', 'init_cursors', 'image_size', 'round_length', 'init_width']
    missing = [key for key in required if key not in data]
    if missing:
        print(f'[WARN] 缺少 seq_extract 关键字段: {missing}')
    else:
        print('[OK] seq_extract 关键字段齐全')
    print()


def check_strokes(strokes_data):
    print('1. 笔画状态 pen_state 统计')
    if strokes_data.ndim != 2 or strokes_data.shape[1] != 7:
        print(f'[ERROR] strokes_data 应为 (N,7)，当前是 {strokes_data.shape}')
        return

    pen_states = strokes_data[:, 0]
    unique, counts = np.unique(pen_states, return_counts=True)
    print(f'  总步数: {len(pen_states)}')
    print(f'  unique/counts: {list(zip(unique.tolist(), counts.tolist()))}')
    print(f'  draw(0.0) 步数: {int(np.sum(pen_states == 0.0))}')
    print(f'  lift/quit(1.0) 步数: {int(np.sum(pen_states == 1.0))}')
    non_binary = np.sum((pen_states != 0.0) & (pen_states != 1.0))
    if non_binary:
        print(f'  [WARN] 非二值 pen_state 数量: {int(non_binary)}')
    if np.sum(pen_states == 0.0) == 0:
        print('  [WARN] 没有 draw(0.0)，可视化会几乎为空')
    print()

    print('2. 相对坐标 x1,y1,x2,y2 范围')
    names = ['x1/ctrl_x', 'y1/ctrl_y', 'x2/end_x', 'y2/end_y']
    coords = strokes_data[:, 1:5]
    for i, name in enumerate(names):
        col = coords[:, i]
        print(f'  {name}: min={col.min():.6f}, max={col.max():.6f}, mean={col.mean():.6f}, std={col.std():.6f}')
    print(f'  coords mean abs: {np.abs(coords).mean():.6f}')
    print(f'  coords abs max: {np.abs(coords).max():.6f}')
    if np.abs(coords).mean() < 0.03:
        print('  [WARN] 坐标平均幅度很小，轨迹容易缩在中心；通常是模型没学会移动或 checkpoint 太早')
    if np.abs(coords).max() > 1.05:
        print('  [WARN] 坐标超过 [-1,1]，可能和 tanh 输出/训练标签定义不一致')
    print()

    print('3. r,s 参数范围')
    params = strokes_data[:, 5:7]
    print(f'  r: min={params[:,0].min():.6f}, max={params[:,0].max():.6f}, mean={params[:,0].mean():.6f}')
    print(f'  s: min={params[:,1].min():.6f}, max={params[:,1].max():.6f}, mean={params[:,1].mean():.6f}')
    print()

    print('4. 前 20 行 strokes_data')
    np.set_printoptions(precision=4, suppress=True)
    print(strokes_data[:20])
    print()


def check_rounds(data, stroke_count):
    print('5. round_length / init_cursors 一致性')
    if 'round_length' not in data or 'init_cursors' not in data:
        print('[WARN] 缺少 round_length 或 init_cursors')
        print()
        return

    round_length = np.asarray(data['round_length']).reshape(-1)
    init_cursors = np.asarray(data['init_cursors'])
    print(f'  round_length: {round_length}, shape={round_length.shape}, sum={int(round_length.sum())}')
    print(f'  init_cursors shape: {init_cursors.shape}, values={init_cursors}')
    if int(round_length.sum()) != stroke_count:
        print(f'  [WARN] sum(round_length)={int(round_length.sum())} != len(strokes_data)={stroke_count}')
    if init_cursors.ndim != 2 or init_cursors.shape[1] != 2:
        print('  [WARN] init_cursors 应为 (num_rounds,2)')
    elif len(init_cursors) != len(round_length):
        print(f'  [WARN] init_cursors 数量 {len(init_cursors)} 和 round_length 数量 {len(round_length)} 不一致')
    if init_cursors.size and (init_cursors.min() < 0 or init_cursors.max() > 1):
        print('  [WARN] init_cursors 超出 [0,1]')
    print()


def check_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True, encoding='latin1')
    print(f'File: {npz_path}')
    print_array_info(data)
    check_required_keys(data)

    if 'strokes_data' not in data:
        return
    strokes_data = np.asarray(data['strokes_data'])
    check_strokes(strokes_data)
    check_rounds(data, len(strokes_data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='检查 seq_extract/vit_query 生成的 npz')
    parser.add_argument('npz_file', help='要检查的 npz 文件路径')
    args = parser.parse_args()
    check_npz(args.npz_file)

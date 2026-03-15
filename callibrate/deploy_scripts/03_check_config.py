#!/usr/bin/env python3
"""
阶段0 · Step 3 ── 配置文件检查 + 坐标预览（无需机器人）

检查 franka_config.yaml 中 workspace_center 是否合理，
并预览：给定配置后，NPZ 坐标将被映射到机器人的哪个物理位置。

⚠️ workspace_center 必须是你实际测量的纸面中心坐标（Franka 基坐标系），
   不能用默认值直接上机！
"""

import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CFG_PATH = os.path.join(ROOT, 'franka_config.yaml')

try:
    import yaml
except ImportError:
    print('❌ pyyaml 未安装 → pip install pyyaml')
    sys.exit(1)


def load_config():
    if not os.path.exists(CFG_PATH):
        print(f'❌ 找不到配置文件: {CFG_PATH}')
        sys.exit(1)
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def check_config(cfg):
    print('=' * 60)
    print('配置文件检查 · 阶段0 Step 3')
    print('=' * 60)
    print(f'\n配置路径: {CFG_PATH}\n')

    wc = cfg.get('workspace_center', {})
    x = wc.get('x')
    y = wc.get('y')
    z = wc.get('z')

    print(f'  workspace_center:')
    print(f'    x = {x} m  （机器人基坐标系，前方）')
    print(f'    y = {y} m  （机器人基坐标系，左/右）')
    print(f'    z = {z} m  （纸面高度）')

    issues = []
    if x is None or y is None or z is None:
        issues.append('❌ workspace_center 字段不完整，请填写 x/y/z')
    else:
        # Franka 典型可达范围检查
        if not (0.2 <= x <= 0.8):
            issues.append(f'⚠️ x={x} 超出 Franka 典型范围 [0.2, 0.8]m，请核实')
        if not (-0.4 <= y <= 0.4):
            issues.append(f'⚠️ y={y} 超出 Franka 典型范围 [-0.4, 0.4]m，请核实')
        if not (0.0 <= z <= 0.6):
            issues.append(f'⚠️ z={z} 超出 Franka 典型范围 [0.0, 0.6]m，请核实')

        # 安全边界预览
        safety = cfg.get('safety', {})
        max_xy = safety.get('max_xy_range', 0.15)
        max_z  = safety.get('max_z_range', 0.10)
        print(f'\n  安全边界 (相对于 workspace_center):')
        print(f'    X: [{x-max_xy:.3f}, {x+max_xy:.3f}] m  (±{max_xy}m)')
        print(f'    Y: [{y-max_xy:.3f}, {y+max_xy:.3f}] m  (±{max_xy}m)')
        print(f'    Z: [{z-max_z:.3f}, {z+max_z:.3f}] m   (±{max_z}m)')

    other_keys = ['robot_ip', 'default_speed', 'default_acceleration']
    print(f'\n  其他配置:')
    for k in other_keys:
        print(f'    {k}: {cfg.get(k, "（未设置）")}')

    print()
    if issues:
        for iss in issues:
            print(f'  {iss}')
        return False
    else:
        print('  ✅ 配置文件格式正常')
        return True


def preview_mapping(cfg, npz_path):
    """预览 NPZ 坐标映射到机器人坐标后的结果"""
    if not os.path.exists(npz_path):
        return

    wc = cfg['workspace_center']
    wx, wy, wz = wc['x'], wc['y'], wc['z']

    d = np.load(npz_path)
    x, y, z = d['pos_3d_x'], d['pos_3d_y'], d['pos_3d_z']

    # 坐标映射（v2 NPZ 格式：相对坐标，z相对于纸面）
    # x/y: 中心对齐到 workspace_center
    x_center = (x.min() + x.max()) / 2
    y_center = (y.min() + y.max()) / 2
    x_robot = wx + (x - x_center)
    y_robot = wy + (y - y_center)
    z_robot = wz + z  # z=0 → 纸面, z>0 → 抬笔, z<0 → 按压

    print(f'\n  坐标映射预览（文件: {os.path.basename(npz_path)}）:')
    print(f'    NPZ X [{x.min():.4f}, {x.max():.4f}] → 机器人 X [{x_robot.min():.4f}, {x_robot.max():.4f}] m')
    print(f'    NPZ Y [{y.min():.4f}, {y.max():.4f}] → 机器人 Y [{y_robot.min():.4f}, {y_robot.max():.4f}] m')
    print(f'    NPZ Z [{z.min():.4f}, {z.max():.4f}] → 机器人 Z [{z_robot.min():.4f}, {z_robot.max():.4f}] m')

    # 检查映射后是否在安全范围内
    safety = cfg.get('safety', {})
    max_xy = safety.get('max_xy_range', 0.15)
    max_z  = safety.get('max_z_range', 0.10)
    out_x = np.sum((x_robot < wx - max_xy) | (x_robot > wx + max_xy))
    out_y = np.sum((y_robot < wy - max_xy) | (y_robot > wy + max_xy))
    out_z = np.sum((z_robot < wz - max_z)  | (z_robot > wz + max_z))
    if out_x or out_y or out_z:
        print(f'\n    ⚠️ 部分点超出安全边界: X={out_x}点 Y={out_y}点 Z={out_z}点')
        print(f'       → 这些点会被 RoboControl 自动截断到安全范围内')
    else:
        print(f'    ✅ 映射后所有点均在安全范围内')


if __name__ == '__main__':
    cfg = load_config()
    ok = check_config(cfg)

    # 预览一个示例 NPZ 的映射结果
    sample = os.path.join(ROOT, 'examples', 'example_永.npz')
    if os.path.exists(sample):
        preview_mapping(cfg, sample)

    print('\n' + '=' * 60)
    if not ok:
        print('❌ 配置存在问题，请修正后再继续。')
        print(f'   编辑: {CFG_PATH}')
    else:
        print('✅ 配置检查通过。')
        print('\n下一步: 阶段2 —— 用 Franka Desk 手动教点，测量纸面坐标')
        print('        然后更新 franka_config.yaml，再运行 03_check_config.py 复验')
    print('=' * 60)

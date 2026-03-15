#!/usr/bin/env python3
"""
阶段4 · Step 8 ── 单点接触测试（轻触纸面中心一次）

验证:
  1. workspace_center 的 Z 值与实际纸面一致
  2. 笔尖能正确接触纸面（轻微印记）
  3. 接触力合理，不会戳穿纸或打滑

路径: Home → 纸面上方5cm → 纸面轻触 → 纸面上方5cm → Home

⚠️ 运行前请确认 Step 7 已通过。
"""

import sys
import os
import time

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CFG_PATH = os.path.join(ROOT, 'franka_config.yaml')

try:
    import yaml
    import numpy as np
except ImportError as e:
    print(f'❌ 缺少依赖: {e}')
    sys.exit(1)

sys.path.insert(0, ROOT)


def load_config():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    wc = cfg['workspace_center']
    return cfg, (wc['x'], wc['y'], wc['z'])


def main():
    print('=' * 60)
    print('单点接触测试 · 阶段4 Step 8')
    print('⚠️  机器人将轻触纸面，请确认安全')
    print('=' * 60)

    cfg, (wx, wy, wz) = load_config()
    robot_ip = cfg.get('robot_ip', '172.16.0.2')
    speed = 0.015  # 极慢

    # 接触深度（轻触，可根据实际调整）
    touch_z = wz - 0.001   # 纸面以下 1mm（极轻）

    print(f'\n配置:')
    print(f'  机器人 IP:  {robot_ip}')
    print(f'  纸面中心:   ({wx:.4f}, {wy:.4f}, {wz:.4f})')
    print(f'  接触目标:   ({wx:.4f}, {wy:.4f}, {touch_z:.4f})  ← 轻触深度 1mm')
    print(f'  速度:       {speed}')
    print('\n执行路径:')
    print(f'  1. 移到纸面上方 5cm  ({wx:.3f}, {wy:.3f}, {wz+0.05:.3f})')
    print(f'  2. 缓慢下降，轻触纸面 ({wx:.3f}, {wy:.3f}, {touch_z:.3f})')
    print(f'  3. 停留 1 秒，观察接触情况')
    print(f'  4. 抬起到上方 5cm')
    print(f'  5. 返回 Home')

    confirm = input('\n确认后输入 YES 继续: ')
    if confirm.strip() != 'YES':
        print('已取消。')
        sys.exit(0)

    from RoboControl import FrankaCalligraphyController
    ctrl = FrankaCalligraphyController(
        robot_ip=robot_ip,
        default_speed=speed,
        workspace_center=(wx, wy, wz),
    )

    if not ctrl.connect():
        print('❌ 连接失败')
        sys.exit(1)

    try:
        # 1. 移到上方
        print('\n[1/5] 移到纸面上方 5cm...')
        ctrl.move_cartesian(wx, wy, wz + 0.05, speed=speed)
        time.sleep(0.5)

        # 2. 轻触
        print('[2/5] 轻触纸面（极慢下降）...')
        print('  → 随时准备按急停！')
        ctrl.move_cartesian(wx, wy, touch_z, speed=0.01)

        # 3. 停留观察
        print('[3/5] 停留 2 秒，请观察接触情况...')
        time.sleep(2)
        print('  问题1: 笔尖是否碰到纸面？（应该是）')
        print('  问题2: 接触力是否太大/太小？')
        obs = input('  接触情况正常吗？(y/N): ')

        # 4. 抬起
        print('[4/5] 抬起...')
        ctrl.move_cartesian(wx, wy, wz + 0.05, speed=speed)
        time.sleep(0.5)

        # 5. Home
        print('[5/5] 返回 Home...')
        ctrl.move_to_home()

        if obs.strip().lower() == 'y':
            print('\n✅ 单点接触测试通过！')
            print('\n如果接触深度不合适，调整 franka_config.yaml 中的 workspace_center.z:')
            print('  - 笔尖没碰到纸 → 减小 z（让机器人降低）')
            print('  - 压力太大     → 增大 z（让机器人升高）')
        else:
            print('\n⚠️  接触不正常。请根据观察结果调整 workspace_center.z，重新运行本步骤。')

    finally:
        ctrl.disconnect()

    print(f'\n{"=" * 60}')
    print('下一步: Step 9 —— 标定线段（验证完整坐标映射）')
    print('=' * 60)


if __name__ == '__main__':
    main()

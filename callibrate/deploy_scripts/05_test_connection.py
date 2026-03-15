#!/usr/bin/env python3
"""
阶段1 · Step 5 ── 网络连通 + 连接测试（需要机器人通电联网）

仅测试 TCP 连接，不发送任何运动指令，机器人不会动。

前置条件:
  - 机器人通电并解锁（Franka Desk 中显示 Ready）
  - 电脑与机器人在同一网络
  - franka_config.yaml 中 robot_ip 已填写

用法:
    python 05_test_connection.py
    python 05_test_connection.py --ip 172.16.0.2
"""

import sys
import os
import argparse
import subprocess

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CFG_PATH = os.path.join(ROOT, 'franka_config.yaml')

try:
    import yaml
    cfg = yaml.safe_load(open(CFG_PATH))
    DEFAULT_IP = cfg.get('robot_ip', '172.16.0.2')
except Exception:
    DEFAULT_IP = '172.16.0.2'


def ping_test(ip):
    print(f'\n[1] Ping 测试: {ip}')
    result = subprocess.run(
        ['ping', '-c', '3', '-W', '1', ip],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        # 提取 RTT
        for line in result.stdout.splitlines():
            if 'avg' in line or 'rtt' in line:
                print(f'  ✅ 可达  {line.strip()}')
                break
        return True
    else:
        print(f'  ❌ 无法 ping 通 {ip}')
        print(f'     请检查:')
        print(f'       1. 机器人是否通电并联网')
        print(f'       2. 网线/WiFi 是否连接正确')
        print(f'       3. IP 地址是否正确（Franka Desk → Settings → Network）')
        return False


def franka_connect_test(ip):
    print(f'\n[2] Franka 库连接测试: {ip}')
    sys.path.insert(0, ROOT)

    # 尝试 franky
    try:
        import franky
        print('  使用 franky...')
        robot = franky.Robot(ip)
        print(f'  ✅ franky 连接成功！')
        print(f'     机器人状态: 已连接')
        del robot
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f'  ❌ franky 连接失败: {e}')
        return False

    # 尝试 frankx
    try:
        import frankx
        print('  使用 frankx...')
        robot = frankx.Robot(ip)
        robot.set_default_behavior()
        print(f'  ✅ frankx 连接成功！')
        del robot
        return True
    except ImportError:
        print('  ❌ 没有 franky 也没有 frankx')
        print('     请运行: pip install frankx')
        return False
    except Exception as e:
        print(f'  ❌ frankx 连接失败: {e}')
        print(f'     常见原因:')
        print(f'       1. 机器人在 Franka Desk 中未解锁')
        print(f'       2. Franka Desk 仍然打开且占用连接（请关闭浏览器标签页）')
        print(f'       3. IP 地址错误')
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Franka 连接测试（仅网络，不运动）')
    parser.add_argument('--ip', type=str, default=DEFAULT_IP, help=f'机器人 IP（默认: {DEFAULT_IP}）')
    args = parser.parse_args()

    print('=' * 60)
    print('连接测试 · 阶段1 Step 5')
    print('⚠️  机器人不会运动，仅测试网络连接')
    print('=' * 60)

    p = ping_test(args.ip)
    if not p:
        print('\n❌ 网络不通，请先解决网络问题再继续。')
        sys.exit(1)

    c = franka_connect_test(args.ip)

    print(f'\n{"=" * 60}')
    if c:
        print('✅ 连接测试通过！')
        print('\n下一步: 阶段2 —— 用 Franka Desk 手动教点，测量纸面坐标')
        print('        更新 franka_config.yaml → 运行 06_teach_workspace.py')
    else:
        print('❌ 连接失败，请排查后重试。')
    print('=' * 60)

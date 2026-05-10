#!/usr/bin/env python3
"""
简化版测试脚本：直接使用现有的示例数据用于仿真
"""
import os
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="准备仿真数据")
    parser.add_argument("--example", "-e", type=str, default="永",
                        choices=["永"], help="选择示例字符")
    parser.add_argument("--output", "-o", type=str,
                        default="outputs/simulation_永.npz", help="输出文件")

    args = parser.parse_args()

    # 使用现有的示例数据
    example_file = f"callibrate/examples/example_{args.example}.npz"

    if os.path.exists(example_file):
        print(f"使用现有的示例文件: {example_file}")

        # 加载并复制到输出目录
        data = np.load(example_file)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        np.savez(args.output, **data)

        print(f"\n{'='*60}")
        print(f"✅ 准备完成！")
        print(f"   输出文件: {args.output}")
        print(f"\n现在可以在 mujoco_sim 中运行:")
        print(f"   cd mujoco_sim")
        print(f"   python mujoco_simulator.py ../{args.output} --speed 0.05")
        print(f"{'='*60}\n")
    else:
        print(f"示例文件不存在: {example_file}")
        print("请检查项目是否完整。")

        # 列出可用的示例
        if os.path.exists("callibrate/examples/"):
            print("\n可用的示例:")
            for f in os.listdir("callibrate/examples/"):
                if f.endswith(".npz"):
                    print(f"  - {f}")


if __name__ == "__main__":
    main()
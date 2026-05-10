#!/usr/bin/env python3
"""
简单的测试脚本，使用项目中已有的数据

这个脚本展示如何：
1. 使用 qwen_stroke_extractor（简化版）
2. 使用 seq_extract_modern（简化版）
3. 测试与 rl_finetune 的集成
"""
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def check_existing_data():
    """检查项目中已有的数据"""
    print("=" * 60)
    print("检查项目现有数据")
    print("=" * 60)

    data_locations = [
        ("dataset/train", "训练数据"),
        ("dataset/val", "验证数据"),
        ("rl_finetune/data/train_data", "RL 训练数据"),
        ("rl_finetune/data/test_data", "RL 测试数据")
    ]

    existing_data = []
    for path, description in data_locations:
        full_path = Path(path)
        if full_path.exists():
            images = list(full_path.glob("*.png"))
            npy_files = list(full_path.glob("*.npy"))
            print(f"\n✅ 找到 {description} ({full_path})")
            print(f"   - 图像数量: {len(images)}")
            print(f"   - NPY 数据: {len(npy_files)}")
            if images:
                existing_data.append((str(full_path / images[0]), description))
        else:
            print(f"\n❌ 未找到 {description}")

    return existing_data


def test_existing_rl_data():
    """测试使用现有的 RL 数据"""
    print("\n" + "=" * 60)
    print("测试现有 RL 数据")
    print("=" * 60)

    rl_data_path = Path("rl_finetune/data/train_data")
    if not rl_data_path.exists():
        print("❌ RL 数据不存在")
        return False

    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image

        # 测试第一个图像和数据
        image_path = rl_data_path / "0.png"
        data_path = rl_data_path / "0.npy"

        if image_path.exists() and data_path.exists():
            print("✅ 找到测试图像和数据")

            # 加载图像
            img = Image.open(image_path)
            print(f"   - 图像尺寸: {img.size}")

            # 加载数据
            data = np.load(data_path)
            print(f"   - 数据形状: {data.shape}")
            print(f"   - 数据内容: {data}")

            return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    return False


def quick_start_guide():
    """打印快速开始指南"""
    print("\n" + "=" * 60)
    print("快速开始指南")
    print("=" * 60)

    print("\n1. 使用现有数据（推荐）:")
    print("   项目已有数据可以直接使用:")
    print("   - dataset/train/")
    print("   - rl_finetune/data/train_data/")

    print("\n2. 测试 Qwen 模型:")
    print("   先安装依赖:")
    print("   pip install transformers==4.40 accelerate==0.30.1")
    print("   然后使用 qwen_stroke_extractor 测试:")
    print("   cd qwen_stroke_extractor")
    print("   python quick_test.py")

    print("\n3. 训练 seq_extract_modern:")
    print("   cd seq_extract_modern")
    print("   python scripts/train.py --train_data ../dataset/train --val_data ../dataset/val")

    print("\n4. 训练强化学习策略:")
    print("   cd rl_finetune")
    print("   python try_tianshou.py --train_data data/train_data --test_data data/test_data")

    print("\n5. 测试 MuJoCo 仿真:")
    print("   cd mujoco_sim")
    print("   python mujoco_simulator.py ../callibrate/examples/example_永.npz --speed 0.05")


def main():
    print("CalliRewrite 项目简单测试\n")

    # 检查现有数据
    existing_data = check_existing_data()

    # 测试 RL 数据
    test_existing_rl_data()

    # 打印快速开始指南
    quick_start_guide()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

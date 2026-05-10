#!/usr/bin/env python3
"""
集成测试脚本：测试整个项目的功能流程

测试流程：
1. 使用 qwen_stroke_extractor 从图像中提取笔画顺序
2. 验证 ViT + Transformer 模型的功能
3. 准备数据用于 Tianshou 强化学习精调
"""
import os
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies():
    """检查是否有必要的依赖包"""
    print("检查依赖包...")
    dependencies = {
        "torch": "torch",
        "transformers": "transformers",
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "cv2": "opencv-python"
    }

    missing = []
    for import_name, pip_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"✅ 已安装: {pip_name}")
        except ImportError:
            print(f"❌ 未安装: {pip_name}")
            missing.append(pip_name)

    if missing:
        print("\n需要先安装依赖包:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        print("\n或者使用完整命令:")
        print("  pip install torch torchvision transformers accelerate matplotlib opencv-python")
        return False

    return True


def test_qwen_extractor(input_image, output_dir):
    """
    测试 qwen_stroke_extractor 的功能
    """
    print("\n" + "=" * 60)
    print("测试 1: qwen_stroke_extractor")
    print("=" * 60)

    try:
        from qwen_stroke_extractor.extractor import create_extractor

        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # 直接使用 API 模式（简单测试）
        try:
            extractor = create_extractor(use_api=False)
            print("✅ 成功创建 qwen_stroke_extractor（本地模式）")
        except Exception as e:
            print(f"⚠️  本地模式加载失败，使用 API 模式: {e}")
            extractor = create_extractor(
                use_api=True,
                api_key="YOUR_API_KEY"  # 请替换为您的实际 API Key
            )
            print("✅ 成功创建 qwen_stroke_extractor（API 模式）")

        # 测试提取
        print(f"\n正在提取图像 {input_image} 的笔画...")
        result = extractor.extract(input_image)
        print(f"✅ 提取成功，获取 {len(result.strokes)} 个笔画")

        # 保存结果
        output_path = output_dir / "qwen_extracted.json"
        extractor.save_result(result, str(output_path), format='json')
        print(f"✅ 结果已保存至 {output_path}")

        # 可视化
        viz_path = output_dir / "qwen_visualization.png"
        extractor.visualize_result(result, str(viz_path), background_image=input_image)
        print(f"✅ 可视化已保存至 {viz_path}")

        return True

    except Exception as e:
        print(f"❌ qwen_stroke_extractor 测试失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_seq_extract_modern(input_image, output_dir):
    """
    测试 seq_extract_modern 的功能
    """
    print("\n" + "=" * 60)
    print("测试 2: seq_extract_modern")
    print("=" * 60)

    # 检查是否有预训练模型
    model_available = False
    model_path = Path("outputs/checkpoints/best_model.ckpt")
    if model_path.exists():
        model_available = True

    if not model_available:
        print("⚠️  seq_extract_modern 模型需要先训练")
        print("   训练命令: python seq_extract_modern/scripts/train.py --train_data data/train --val_data data/val")
        return False

    try:
        from seq_extract_modern.inference.predictor import Predictor
        from seq_extract_modern.configs.model_config import get_default_config

        config = get_default_config()
        predictor = Predictor(
            model_path=str(model_path),
            config=config
        )

        print(f"✅ 成功创建 seq_extract_modern 预测器")

        # 预测
        result = predictor.predict(input_image, num_strokes=100)
        print(f"✅ 提取到 {len(result['stroke_params'])} 个笔画")

        # 保存结果
        output_dir = Path(output_dir)
        import numpy as np
        np.save(str(output_dir / "seq_extract_modern_result.npy"), result['stroke_params'])
        print("✅ 结果已保存")

        return True

    except Exception as e:
        print(f"❌ seq_extract_modern 测试失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def prepare_for_rl(input_image, output_dir, source="qwen"):
    """
    为强化学习准备数据
    """
    print("\n" + "=" * 60)
    print("测试 3: 数据准备")
    print("=" * 60)

    # 检查是否有提取结果
    qwen_result = Path(output_dir) / "qwen_extracted.json"
    seq_result = Path(output_dir) / "seq_extract_modern_result.npy"

    if source == "qwen" and qwen_result.exists():
        data_path = qwen_result
        print(f"✅ 使用 qwen_stroke_extractor 结果")
    elif source == "seq" and seq_result.exists():
        data_path = seq_result
        print(f"✅ 使用 seq_extract_modern 结果")
    else:
        print("❌ 未找到有效的提取结果")
        return False

    # 创建 RL 数据格式
    output_dir = Path(output_dir)
    train_dir = output_dir / "rl_train_data"
    train_dir.mkdir(exist_ok=True)

    try:
        # 简单的转换（这里只做演示）
        if data_path.suffix == ".json":
            import json
            with open(data_path, 'r') as f:
                data = json.load(f)

            # 保存简化版本
            import numpy as np
            strokes = []
            for i, stroke in enumerate(data['strokes']):
                for point in stroke['points']:
                    strokes.append([point['x'], point['y'], i])

            strokes = np.array(strokes)
            np.save(str(train_dir / "strokes.npy"), strokes)
            print(f"✅ RL 训练数据已保存至 {train_dir / 'strokes.npy'}")
        else:
            import shutil
            shutil.copy(data_path, train_dir / "strokes.npy")
            print(f"✅ RL 训练数据已保存至 {train_dir / 'strokes.npy'}")

        return True

    except Exception as e:
        print(f"⚠️  数据准备失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="集成测试脚本")
    parser.add_argument("input_image", nargs='?', help="输入图像路径")
    parser.add_argument("-o", "--output-dir", default="outputs/integrated_test",
                      help="输出目录")
    parser.add_argument("-s", "--source", choices=["qwen", "seq"], default="qwen",
                      help="使用的笔画提取来源: qwen 或 seq")

    args = parser.parse_args()

    # 检查依赖
    if not check_dependencies():
        return 1

    # 如果没有提供输入图像，使用项目中的示例
    if not args.input_image:
        print("\n没有指定输入图像，将使用项目示例图像...")
        possible_paths = [
            "dataset/train/elephant.png",
            "rl_finetune/data/test_data/0.png",
            "seq_extract/sample_inputs/clean_line_drawings/duck.png"
        ]
        for path in possible_paths:
            if Path(path).exists():
                args.input_image = path
                print(f"使用示例图像: {args.input_image}")
                break
        else:
            print("❌ 没有找到可用的示例图像，请提供输入图像路径")
            return 1

    # 验证输入图像
    input_image = Path(args.input_image)
    if not input_image.exists():
        print(f"❌ 输入图像不存在: {input_image}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n开始测试，输入图像: {input_image}")

    # 1. 测试 qwen_stroke_extractor
    qwen_success = test_qwen_extractor(str(input_image), str(output_dir))

    # 2. 测试 seq_extract_modern（可选）
    seq_success = test_seq_extract_modern(str(input_image), str(output_dir))

    # 3. 数据准备
    rl_success = prepare_for_rl(str(input_image), str(output_dir), args.source)

    # 打印总结
    print("\n" + "=" * 60)
    print("集成测试总结")
    print("=" * 60)

    tests = [
        ("qwen_stroke_extractor", qwen_success),
        ("seq_extract_modern", seq_success),
        ("RL 数据准备", rl_success)
    ]

    all_passed = True
    for test_name, status in tests:
        status_str = "✅ 通过" if status else "❌ 失败"
        print(f"{test_name:20} | {status_str}")
        if not status:
            all_passed = False

    print("\n" + "=" * 60)
    print(f"测试结果: {'全部通过' if all_passed else '部分失败'}")
    print("=" * 60)

    if all_passed:
        print("\n🎉 所有测试通过！可以开始使用 Tianshou 进行强化学习精调。")
        print("\n下一步命令:")
        print("   python rl_finetune/try_tianshou.py --train_data outputs/integrated_test/rl_train_data")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

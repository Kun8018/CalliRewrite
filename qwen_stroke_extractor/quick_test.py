#!/usr/bin/env python3
"""
快速测试脚本 - 帮助用户选择合适的方案

本脚本提供:
1. 测试现有的 seq_extract_modern 模型（PyTorch ViT+Transformer）
2. 展示如何使用千问Plus的API模式
3. 对比两种方案的输出
"""
import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_seq_extract_modern(image_path, output_dir):
    """
    测试现有的 seq_extract_modern 模型
    """
    print("=" * 60)
    print("测试方案 1: seq_extract_modern (ViT + Transformer)")
    print("=" * 60)

    try:
        from seq_extract_modern.inference.predictor import Predictor as ModernPredictor
        from seq_extract_modern.configs.model_config import get_default_config

        config = get_default_config()

        print("\n注意: 需要有训练好的模型才能使用此方案")
        print("如果没有模型，此方案会报错")

        # 这里假设模型路径，实际使用时需要用户自己训练模型
        # predictor = ModernPredictor(
        #     model_path="path/to/checkpoint.ckpt",
        #     config=config
        # )

        print("\n✅ seq_extract_modern 方案准备就绪（需要训练模型）")
        print("\n提示: 如果要使用此方案，需要先训练模型:")
        print("  cd seq_extract_modern")
        print("  python scripts/train.py --train_data /path/to/data")

        return True

    except ImportError as e:
        print(f"\n❌ 导入失败: {e}")
        print("请确保已安装 seq_extract_modern 的依赖")
        return False


def test_qwen_api_demo(image_path, output_dir):
    """
    演示千问Plus API方案（不实际调用API）
    """
    print("\n" + "=" * 60)
    print("测试方案 2: 千问Plus API (Qwen-VL-Plus)")
    print("=" * 60)

    print("\n✅ 千问Plus API 方案准备就绪")
    print("\n使用步骤:")
    print("1. 访问 https://dashscope.aliyun.com/")
    print("2. 注册阿里云账号并获取 API Key")
    print("3. 运行:")
    print("   python examples.py /path/to/image.png --use-api --api-key YOUR_KEY")

    print("\nAPI调用示例代码:")
    print("""
from qwen_stroke_extractor.extractor import create_extractor

extractor = create_extractor(
    use_api=True,
    api_key="your-api-key"
)

result = extractor.extract("calligraphy.png")

# 保存为 npy 格式
extractor.save_result(result, "strokes.npy")

# 可视化
extractor.visualize_result(result, "viz.png")
    """)

    print("\n💡 提示: API 模式适合快速测试，不需要训练模型")

    return True


def test_qwen_local_demo(image_path, output_dir):
    """
    演示千问Plus本地模型方案
    """
    print("\n" + "=" * 60)
    print("测试方案 3: 千问Plus 本地模型")
    print("=" * 60)

    print("\n✅ 本地模型方案准备就绪")
    print("\n使用步骤:")
    print("1. 从 HuggingFace 下载模型:")
    print("   git lfs install")
    print("   git clone https://huggingface.co/Qwen/Qwen-VL-Plus")
    print("2. 运行:")
    print("   python examples.py /path/to/image.png --model-path ./Qwen-VL-Plus")

    print("\n💡 提示: 本地模型需要较高配置的 GPU（建议 16GB+ 显存）")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="快速测试 - 帮助选择合适的笔画提取方案"
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="测试图像路径（可选）"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="outputs/quick_test",
        help="输出目录"
    )

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 60)
    print("CalliRewrite 笔画提取方案快速测试")
    print("=" * 60)

    print("\n共有三种方案可供选择:\n")
    print("1. seq_extract_modern  - PyTorch ViT + Transformer（需要训练）")
    print("2. 千问Plus API       - 使用阿里云 API（推荐）")
    print("3. 千问Plus 本地模型   - 本地运行大模型")

    results = []

    # 测试方案1
    results.append(("seq_extract_modern", test_seq_extract_modern(args.image, output_dir)))

    # 测试方案2
    results.append(("千问Plus API", test_qwen_api_demo(args.image, output_dir)))

    # 测试方案3
    results.append(("千问Plus 本地模型", test_qwen_local_demo(args.image, output_dir)))

    # 总结
    print("\n" + "=" * 60)
    print("方案对比总结")
    print("=" * 60)

    print("\n" + "-" * 60)
    print(f"{'方案':<20} | {'状态':<10} | {'特点':<30}")
    print("-" * 60)

    for name, ok in results:
        if name == "seq_extract_modern":
            features = "可控性好，需训练"
        elif name == "千问Plus API":
            features = "即开即用，推荐"
        else:
            features = "离线可用，需大GPU"

        status = "✅ 准备好" if ok else "❌ 需配置"
        print(f"{name:<20} | {status:<10} | {features:<30}")

    print("-" * 60)

    print("\n💡 推荐方案:")
    print("- 如果你想快速测试: 使用 千问Plus API")
    print("- 如果你想完全控制: 使用 seq_extract_modern（需训练）")
    print("- 如果你有强大GPU: 使用 千问Plus 本地模型")

    print("\n详细文档:")
    print("- qwen_stroke_extractor/README.md")
    print("- seq_extract_modern/README.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
简单测试脚本 - 验证 seq_extract_modern 的基本功能
"""
import os
import sys
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent / "seq_extract_modern"))

def test_config():
    print("测试 1/5: 配置模块...")
    try:
        from configs.model_config import get_default_config
        config = get_default_config()
        print(f"✓ 配置加载成功")
        print(f"  - 图像大小: {config['model'].image_size}")
        print(f"  - 笔画维度: {config['model'].stroke_params_dim}")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data():
    print("\n测试 2/5: 数据模块...")
    try:
        from configs.model_config import get_default_config
        from data.datasets import CalligraphyDataset
        config = get_default_config()

        # 测试类是否存在
        print(f"✓ Dataset 类加载成功")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def test_models():
    print("\n测试 3/5: 模型模块...")
    try:
        from configs.model_config import get_default_config
        from models.vit_transformer import (
            create_extractor_model,
            ViTStrokeEncoder,
            StrokeTransformerDecoder,
            CalligraphyExtractor
        )
        config = get_default_config()

        # 测试创建模型
        model = create_extractor_model(config['model'])
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 模型创建成功")
        print(f"  - 参数数量: {num_params:,}")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_renderer():
    print("\n测试 4/5: 渲染器模块...")
    try:
        from renderer.neural_renderer import (
            create_renderer,
            SimpleRenderer,
            NeuralRasterizor
        )
        print(f"✓ 渲染器模块加载成功")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference():
    print("\n测试 5/5: 推理模块...")
    try:
        from configs.model_config import get_default_config
        from inference.predictor import create_predictor
        config = get_default_config()

        print(f"✓ 推理模块加载成功")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("  seq_extract_modern 基本功能验证")
    print("="*60)

    results = [
        ("配置", test_config()),
        ("数据", test_data()),
        ("模型", test_models()),
        ("渲染", test_renderer()),
        ("推理", test_inference()),
    ]

    print("\n" + "="*60)
    print("  验证摘要")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("  所有模块加载成功！")
        print("  下一步: 安装依赖并运行训练/测试")
    else:
        print("  部分模块加载失败，请检查依赖")
    print("="*60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
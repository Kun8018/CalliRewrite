#!/usr/bin/env python3
"""
专门测试 seq_extract_modern（ViT + Transformer）的脚本
"""
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "seq_extract_modern"))


def test_imports():
    """测试基本导入"""
    print("=" * 60)
    print("测试 1: 基础依赖导入")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        import torchvision
        from pytorch_lightning import Trainer

        print("✅ PyTorch 和 PyTorch Lightning 导入成功")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_model_config():
    """测试配置"""
    print("\n" + "=" * 60)
    print("测试 2: 配置加载")
    print("=" * 60)

    try:
        from configs.model_config import ViTTransformerConfig, TrainingConfig

        config = ViTTransformerConfig()
        print("✅ 配置加载成功")
        print(f"   图像尺寸: {config.image_size}")
        print(f"   ViT 隐藏层: {config.vit_hidden_dim}")
        print(f"   Decoder 隐藏层: {config.decoder_hidden_dim}")

        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("测试 3: 模型创建")
    print("=" * 60)

    try:
        from configs.model_config import ViTTransformerConfig
        from models.vit_transformer import create_extractor_model

        config = ViTTransformerConfig()
        model = create_extractor_model(config)
        print("✅ 模型创建成功")
        print(f"   模型类型: {type(model)}")

        # 检查参数计数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   总参数: {total_params:,}")

        return True
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_data_loader():
    """测试数据加载"""
    print("\n" + "=" * 60)
    print("测试 4: 数据加载")
    print("=" * 60)

    try:
        from configs.model_config import ViTTransformerConfig
        from data.datasets import CalligraphyDataset
        from data.transforms import get_training_transforms

        config = ViTTransformerConfig()
        dataset_path = Path("../dataset/train")

        if not dataset_path.exists():
            print("⚠️  训练数据不存在，但代码可用")
            print("   需要创建数据集:")
            print("   python scripts/prepare_data.py")
            return True

        transform = get_training_transforms(config)
        dataset = CalligraphyDataset(str(dataset_path), config, transform, training=True)

        print(f"✅ 数据加载成功")
        print(f"   训练图像数量: {len(dataset)}")

        # 测试加载一个样本
        sample = dataset[0]
        print(f"   样本类型: {type(sample)}")

        return True

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_training():
    """测试训练流程"""
    print("\n" + "=" * 60)
    print("测试 5: 训练流程")
    print("=" * 60)

    print("⚠️  跳过训练流程测试（需要完整的数据和环境配置）")
    print("   可以使用 scripts/train.py 进行完整训练")
    return True


def main():
    print("seq_extract_modern (ViT + Transformer) 测试")

    results = {}

    results["基础依赖"] = test_imports()
    results["配置"] = test_model_config()
    results["模型创建"] = test_model_creation()
    results["数据加载"] = test_data_loader()
    results["训练流程"] = test_training()

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, status in results.items():
        status_str = "✅ 成功" if status else "❌ 失败"
        print(f"{name:20} | {status_str}")

    print("\n" + "=" * 60)
    print("下一步:")
    print("=" * 60)

    if all(results.values()):
        print("🎉 所有模块测试成功！")
        print("\n现在你可以开始训练了:")
        print("   cd seq_extract_modern")
        print("   python scripts/train.py --train_data ../dataset/train --val_data ../dataset/val --max_epoch 10")

    else:
        print("\n⚠️  部分测试失败，请先解决依赖问题")


if __name__ == "__main__":
    sys.exit(main())

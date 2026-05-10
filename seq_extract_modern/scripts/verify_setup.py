"""
验证环境和模块安装是否正确
"""
import os
import sys
from pathlib import Path


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_python_version():
    print_header("Python 版本")
    print(f"Python 版本: {sys.version}")
    assert sys.version_info >= (3, 10), "需要 Python 3.10 或更高版本"
    print("✓ Python 版本满足要求")


def test_imports():
    print_header("测试核心依赖导入")

    modules = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
        ('matplotlib', 'Matplotlib'),
    ]

    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            module = sys.modules[module_name]
            if hasattr(module, '__version__'):
                print(f"✓ {display_name}: {module.__version__}")
            else:
                print(f"✓ {display_name}: 已安装")
        except ImportError as e:
            print(f"✗ {display_name}: 未安装 - {e}")
            all_ok = False

    return all_ok


def test_project_structure():
    print_header("检查项目结构")

    base_dir = Path(__file__).parent.parent
    expected_files = [
        "configs/model_config.py",
        "models/vit_transformer.py",
        "data/datasets.py",
        "trainer/training_module.py",
        "renderer/neural_renderer.py",
        "inference/predictor.py",
        "scripts/train.py",
        "scripts/test.py",
        "requirements.txt",
        "README.md",
    ]

    all_exist = True
    for file_path in expected_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}: 缺失")
            all_exist = False

    return all_exist


def test_model_creation():
    print_header("测试模型创建")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from configs.model_config import get_default_config
        from models.vit_transformer import create_extractor_model

        config = get_default_config()
        model = create_extractor_model(config.model)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 模型创建成功")
        print(f"✓ 参数数量: {num_params:,}")

        return True

    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predictor():
    print_header("测试预测器")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from configs.model_config import get_default_config
        from inference.predictor import Predictor

        config = get_default_config()
        predictor = Predictor(model_path=None, config=config, device='cpu')

        print(f"✓ 预测器创建成功")

        return True

    except Exception as e:
        print(f"✗ 预测器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("  seq_extract_modern 安装验证")
    print("="*60)

    results = []

    try:
        test_python_version()
    except AssertionError as e:
        print(f"✗ {e}")
        return 1

    results.append(("核心依赖", test_imports()))
    results.append(("项目结构", test_project_structure()))
    results.append(("模型创建", test_model_creation()))
    results.append(("预测器", test_predictor()))

    print_header("验证摘要")

    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("  所有验证通过！准备开始使用。")
        print("="*60 + "\n")
        return 0
    else:
        print("  部分验证失败，请检查上述错误。")
        print("="*60 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
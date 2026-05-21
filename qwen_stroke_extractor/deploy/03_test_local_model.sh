#!/bin/bash
# 服务器部署脚本 3/5 - 测试本地模型
# 使用方法: bash 03_test_local_model.sh

set -e

echo "========================================"
echo "步骤 3: 测试本地模型"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 检查虚拟环境
if [ -d "$PROJECT_ROOT/calli_train_env" ]; then
    source "$PROJECT_ROOT/calli_train_env/bin/activate"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "未找到虚拟环境，请先运行 01_install_dependencies.sh"
    exit 1
fi

cd "$PROJECT_ROOT/qwen_stroke_extractor"

# 检查配置
if [ -f "$SCRIPT_DIR/config.sh" ]; then
    source "$SCRIPT_DIR/config.sh"
else
    # 尝试自动检测
    if [ -d "models/Qwen-VL-Plus" ]; then
        QWEN_MODEL_PATH="$PROJECT_ROOT/qwen_stroke_extractor/models/Qwen-VL-Plus"
        QWEN_MODEL_NAME="Qwen-VL-Plus"
    elif [ -d "models/Qwen-VL" ]; then
        QWEN_MODEL_PATH="$PROJECT_ROOT/qwen_stroke_extractor/models/Qwen-VL"
        QWEN_MODEL_NAME="Qwen-VL"
    elif [ -d "../Qwen-VL-Plus" ]; then
        QWEN_MODEL_PATH="$PROJECT_ROOT/Qwen-VL-Plus"
        QWEN_MODEL_NAME="Qwen-VL-Plus"
    elif [ -d "../Qwen-VL" ]; then
        QWEN_MODEL_PATH="$PROJECT_ROOT/Qwen-VL"
        QWEN_MODEL_NAME="Qwen-VL"
    else
        echo "未找到模型！"
        echo "请先运行 bash 02_download_model.sh"
        exit 1
    fi
fi

echo "项目根目录: $PROJECT_ROOT"
echo "模型路径: $QWEN_MODEL_PATH"
echo "模型名称: $QWEN_MODEL_NAME"

# 检查模型路径是否存在
if [ ! -d "$QWEN_MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $QWEN_MODEL_PATH"
    exit 1
fi

# 检查是否有测试图像
TEST_IMAGE="$PROJECT_ROOT/rl_finetune/data/train_data/0.png"
if [ ! -f "$TEST_IMAGE" ]; then
    echo "未找到测试图像，正在查找替代图像..."

    # 尝试在项目中查找其他 PNG 图像
    TEST_IMAGE=$(find "$PROJECT_ROOT" -name "*.png" -o -name "*.jpg" | head -1)
    if [ -z "$TEST_IMAGE" ]; then
        echo "错误: 项目中未找到任何图像文件"
        exit 1
    fi
    echo "使用找到的图像: $TEST_IMAGE"
fi

# 创建输出目录
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen_test"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "测试图像: $TEST_IMAGE"
echo "输出目录: $OUTPUT_DIR"

# 运行测试脚本
echo ""
echo "========================================"
echo "运行本地模型测试..."
echo "========================================"

python3 <<END
import sys
import os
sys.path.insert(0, "$PROJECT_ROOT")
from qwen_stroke_extractor.extractor import create_extractor

print("初始化提取器...")
extractor = create_extractor(
    model_path="$QWEN_MODEL_PATH",
    use_api=False
)

print("提取笔画...")
result = extractor.extract("$TEST_IMAGE")
print(f"识别到 {len(result.strokes)} 个笔画")

# 保存结果
result_dir = "$OUTPUT_DIR"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 保存为各种格式
print("保存结果...")
extractor.save_result(result, os.path.join(result_dir, "strokes.npy"), format="npy")
extractor.save_result(result, os.path.join(result_dir, "strokes.json"), format="json")

# 可视化
try:
    extractor.visualize_result(
        result,
        os.path.join(result_dir, "visualization.png"),
        background_image="$TEST_IMAGE"
    )
    print("可视化成功")
except Exception as e:
    print(f"可视化时出错: {e}")
    print("继续执行，跳过可视化")

print("-" * 60)
print("测试完成！")
print("-" * 60)
print(f"结果保存在: {result_dir}")
print()
print("检查输出文件:")
print(f"  - 笔画参数: {result_dir}/strokes.npy")
print(f"  - JSON格式: {result_dir}/strokes.json")
print(f"  - 可视化: {result_dir}/visualization.png")
print()
print("测试统计信息:")
print(f"  笔画数量: {len(result.strokes)}")
if result.strokes:
    avg_width = sum(stroke.width for stroke in result.strokes) / len(result.strokes)
    print(f"  平均宽度: {avg_width:.4f}")
END

echo ""
echo "========================================"
echo "✅ 本地模型测试完成！"
echo "========================================"

# 显示输出结果
echo ""
echo "检查生成的文件:"
ls -la "$OUTPUT_DIR"

echo ""
echo "下一步: 运行 bash 04_batch_process.sh 进行批量处理"

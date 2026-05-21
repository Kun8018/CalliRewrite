#!/bin/bash
# 服务器部署脚本 4/5 - 批量处理脚本
# 使用方法: bash 04_batch_process.sh [输入目录]

set -e

echo "========================================"
echo "步骤 4: 批量处理书法图像"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 获取输入目录
if [ "$#" -eq 0 ]; then
    # 查找项目中的图像目录
    if [ -d "$PROJECT_ROOT/rl_finetune/data/test_data" ]; then
        INPUT_DIR="$PROJECT_ROOT/rl_finetune/data/test_data"
    elif [ -d "$PROJECT_ROOT/dataset/val" ]; then
        INPUT_DIR="$PROJECT_ROOT/dataset/val"
    else
        echo "未找到图像目录"
        echo "使用方法: bash 04_batch_process.sh [输入目录]"
        exit 1
    fi
    echo "使用默认图像目录: $INPUT_DIR"
else
    INPUT_DIR="$1"
fi

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

echo "项目根目录: $PROJECT_ROOT"
echo "输入目录: $INPUT_DIR"

# 激活虚拟环境
if [ -d "$PROJECT_ROOT/calli_train_env" ]; then
    source "$PROJECT_ROOT/calli_train_env/bin/activate"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "未找到虚拟环境，请先运行 01_install_dependencies.sh"
    exit 1
fi

# 检查配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.sh" ]; then
    source "$SCRIPT_DIR/config.sh"
else
    # 尝试自动检测模型
    if [ -d "$PROJECT_ROOT/qwen_stroke_extractor/models/Qwen-VL-Plus" ]; then
        QWEN_MODEL_PATH="$PROJECT_ROOT/qwen_stroke_extractor/models/Qwen-VL-Plus"
        QWEN_MODEL_NAME="Qwen-VL-Plus"
    elif [ -d "$PROJECT_ROOT/qwen_stroke_extractor/models/Qwen-VL" ]; then
        QWEN_MODEL_PATH="$PROJECT_ROOT/qwen_stroke_extractor/models/Qwen-VL"
        QWEN_MODEL_NAME="Qwen-VL"
    elif [ -d "$PROJECT_ROOT/Qwen-VL-Plus" ]; then
        QWEN_MODEL_PATH="$PROJECT_ROOT/Qwen-VL-Plus"
        QWEN_MODEL_NAME="Qwen-VL-Plus"
    elif [ -d "$PROJECT_ROOT/Qwen-VL" ]; then
        QWEN_MODEL_PATH="$PROJECT_ROOT/Qwen-VL"
        QWEN_MODEL_NAME="Qwen-VL"
    else
        echo "未找到模型！"
        echo "请先运行 bash 02_download_model.sh"
        exit 1
    fi
fi

echo "模型路径: $QWEN_MODEL_PATH"
echo "模型名称: $QWEN_MODEL_NAME"

# 创建输出目录
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen_batch"
mkdir -p "$OUTPUT_DIR"

echo "输出目录: $OUTPUT_DIR"

# 查找图像文件
echo ""
echo "查找图像文件..."
IMAGE_EXT="*.png *.jpg *.jpeg *.gif *.bmp"
IMAGE_FILES=()

for ext in $IMAGE_EXT; do
    found=($(find "$INPUT_DIR" -iname "$ext" -type f))
    IMAGE_FILES+=("${found[@]}")
done

if [ ${#IMAGE_FILES[@]} -eq 0 ]; then
    echo "未找到图像文件"
    echo "支持的格式: $IMAGE_EXT"
    exit 1
fi

echo "找到 ${#IMAGE_FILES[@]} 个图像文件"

# 运行批量处理
echo ""
echo "========================================"
echo "开始批量处理..."
echo "========================================"

python3 <<END
import sys
import os
sys.path.insert(0, "$PROJECT_ROOT")
from qwen_stroke_extractor.extractor import create_extractor

# 初始化提取器
print("初始化提取器...")
try:
    extractor = create_extractor(
        model_path="$QWEN_MODEL_PATH",
        use_api=False
    )
except Exception as e:
    print(f"初始化失败: {e}")
    sys.exit(1)

input_dir = "$INPUT_DIR"
output_dir = "$OUTPUT_DIR"
image_files = ${IMAGE_FILES[@]}

os.makedirs(output_dir, exist_ok=True)

total_files = len(image_files)
success_count = 0

for i, image_path in enumerate(image_files):
    filename = os.path.basename(image_path)
    name, _ = os.path.splitext(filename)

    print()
    print(f"正在处理 {i+1}/{total_files}: {filename}")

    try:
        # 提取笔画
        result = extractor.extract(image_path)

        # 保存结果
        output_path = os.path.join(output_dir, f"{name}_strokes.npy")
        extractor.save_result(result, output_path, format="npy")

        # 保存 JSON
        json_path = os.path.join(output_dir, f"{name}_strokes.json")
        extractor.save_result(result, json_path, format="json")

        # 可视化（可选）
        viz_path = os.path.join(output_dir, f"{name}_viz.png")
        try:
            extractor.visualize_result(
                result,
                viz_path,
                background_image=image_path
            )
        except Exception as e:
            print(f"可视化失败: {e}")

        success_count += 1
        print(f"成功: {len(result.strokes)} 个笔画")

    except Exception as e:
        print(f"失败: {e}")

print()
print("-" * 60)
print("批量处理完成")
print("-" * 60)
print(f"成功: {success_count}/{total_files}")
print(f"失败: {total_files - success_count}/{total_files}")
print()
print(f"结果保存在: {output_dir}")
END

echo ""
echo "========================================"
echo "✅ 批量处理完成！"
echo "========================================"

echo ""
echo "检查输出目录:"
ls -la "$OUTPUT_DIR" | head -30

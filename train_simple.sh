#!/bin/bash

# 简化的训练脚本
# 直接使用原始项目的 sample_inputs 作为训练数据

set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== CalliRewrite seq_extract_modern 训练 ==="
echo "项目根目录: $PROJECT_ROOT"

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "请激活虚拟环境: source calli_train_env_gpu/bin/activate"
    exit 1
fi

# 创建训练目录
TRAIN_DATA_DIR="$PROJECT_ROOT/dataset/train"
VAL_DATA_DIR="$PROJECT_ROOT/dataset/val"

mkdir -p "$TRAIN_DATA_DIR"
mkdir -p "$VAL_DATA_DIR"

# 复制数据（使用原始项目的 sample_inputs/clean_line_drawings 作为训练数据）
ORIGINAL_DATA="$PROJECT_ROOT/seq_extract/sample_inputs/clean_line_drawings"

if [ ! -d "$ORIGINAL_DATA" ]; then
    echo "原始数据集不存在: $ORIGINAL_DATA"
    exit 1
fi

# 复制所有图像
cp "$ORIGINAL_DATA"/*.png "$TRAIN_DATA_DIR/"

# 简单的验证集划分（取前2张作为验证）
cp "$TRAIN_DATA_DIR/puppy.png" "$VAL_DATA_DIR/"
cp "$TRAIN_DATA_DIR/elephant.png" "$VAL_DATA_DIR/"

# 检查数据
echo -e "\n--- 数据检查 ---"
TRAIN_COUNT=$(ls -1 "$TRAIN_DATA_DIR"/*.png 2>/dev/null | wc -l)
VAL_COUNT=$(ls -1 "$VAL_DATA_DIR"/*.png 2>/dev/null | wc -l)

echo "训练数据: $TRAIN_COUNT 张图像"
echo "验证数据: $VAL_COUNT 张图像"

if [ $TRAIN_COUNT -eq 0 ] || [ $VAL_COUNT -eq 0 ]; then
    echo "数据复制失败!"
    exit 1
fi

# 运行训练
echo -e "\n--- 开始训练 ---"
cd "$PROJECT_ROOT/seq_extract_modern"

python scripts/train.py \
    --train_data "$TRAIN_DATA_DIR" \
    --val_data "$VAL_DATA_DIR" \
    --batch_size 8 \
    --lr 1e-4 \
    --max_epochs 50 \
    --gpus 0 \
    --save_dir "$PROJECT_ROOT/outputs" \
    --project_name "calli_extract_mac_cpu"

echo -e "\n=== 训练完成 ==="
echo "结果保存在: $PROJECT_ROOT/outputs"

# 测试训练后的模型
echo -e "\n--- 测试训练后的模型 ---"
python scripts/test.py \
    --input "$PROJECT_ROOT/dataset/val" \
    --model "$PROJECT_ROOT/outputs/checkpoints/best_model.ckpt" \
    --output "$PROJECT_ROOT/outputs/inference"

echo -e "\n=== 所有完成! ==="
echo "1. 训练数据: $TRAIN_DATA_DIR"
echo "2. 验证数据: $VAL_DATA_DIR"
echo "3. 检查点: $PROJECT_ROOT/outputs/checkpoints/"
echo "4. 推理结果: $PROJECT_ROOT/outputs/inference/"
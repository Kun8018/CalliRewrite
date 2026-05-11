#!/bin/bash
# 最简单的下载方式 - 直接 git clone
# 使用方法: bash download_git.sh [Qwen-VL | Qwen-VL-Plus]

set -e

echo "========================================"
echo "Git 下载（最简单）"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/qwen_stroke_extractor/models"

# 选择模型
if [ "$#" -eq 0 ]; then
    echo "未指定模型，默认下载 Qwen-VL"
    MODEL_NAME="Qwen-VL"
else
    MODEL_NAME="$1"
fi

echo "项目根目录: $PROJECT_ROOT"
echo "模型存储路径: $MODEL_DIR"
echo "准备下载: $MODEL_NAME"

# 创建目录
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

# 检查是否已经下载
if [ -d "$MODEL_NAME" ]; then
    echo "模型 $MODEL_NAME 已存在"
    exit 0
fi

# 使用镜像
echo ""
echo "使用 HuggingFace 镜像..."
export HF_ENDPOINT=https://hf-mirror.com

# 使用 git clone
echo ""
echo "开始 git clone..."
git lfs install
echo ""
echo "克隆模型: https://hf-mirror.com/Qwen/$MODEL_NAME"
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/Qwen/$MODEL_NAME

# 下载 LFS 文件
cd "$MODEL_NAME"
git lfs pull

# 更新配置文件
echo ""
echo "创建配置文件..."
if [ ! -f "$SCRIPT_DIR/config.sh" ]; then
    cat > "$SCRIPT_DIR/config.sh" <<EOL
# qwen_stroke_extractor 配置文件
export QWEN_MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
export QWEN_MODEL_NAME="$MODEL_NAME"
EOL
fi

echo ""
echo "========================================"
echo "下载完成！"
echo "========================================"
echo "模型位置: $MODEL_DIR/$MODEL_NAME"
ls -la "$MODEL_DIR/$MODEL_NAME"
echo ""
echo "下一步: 运行 bash 03_test_local_model.sh"
echo ""

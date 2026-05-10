#!/bin/bash
# 服务器部署脚本 2/5 - 下载模型
# 使用方法: bash 02_download_model.sh [model_name]
# 可选模型: Qwen-VL-Plus, Qwen-VL

set -e

echo "========================================"
echo "步骤 2: 下载千问模型"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/qwen_stroke_extractor/models"

# 检查参数
if [ "$#" -eq 0 ]; then
    echo "未指定模型名称，使用默认模型: Qwen-VL"
    MODEL_NAME="Qwen-VL"
else
    MODEL_NAME="$1"
fi

echo "项目根目录: $PROJECT_ROOT"
echo "模型存储路径: $MODEL_DIR"
echo "准备下载: $MODEL_NAME"

# 创建模型目录
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

# 检查是否已经下载
if [ -d "$MODEL_NAME" ]; then
    echo "模型 $MODEL_NAME 已经存在于 $MODEL_DIR 中"
    echo "如果需要重新下载，请先删除该目录"
    ls -la "$MODEL_NAME"
    echo "----------------------------------------"
    echo "如果已存在的模型不完整，请运行: rm -rf $MODEL_DIR/$MODEL_NAME"
    exit 0
fi

# 检查虚拟环境
if [[ -z "${VIRTUAL_ENV}" ]]; then
    if [ -d "$PROJECT_ROOT/calli_train_env" ]; then
        echo "未激活虚拟环境，正在激活..."
        source "$PROJECT_ROOT/calli_train_env/bin/activate"
    fi
fi

# 检查 git-lfs 是否安装
if ! command -v git-lfs &> /dev/null; then
    echo "git-lfs 未安装，正在安装..."

    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y git-lfs
    elif command -v yum &> /dev/null; then
        yum install -y git-lfs
    elif command -v brew &> /dev/null; then
        brew install git-lfs
    else
        echo "无法自动安装 git-lfs，请手动安装"
        echo "MacOS: brew install git-lfs"
        echo "Ubuntu: apt-get install git-lfs"
        echo "CentOS: yum install git-lfs"
        exit 1
    fi

    # 初始化 git-lfs
    git lfs install
fi

echo ""
echo "开始从 HuggingFace 下载 $MODEL_NAME..."
echo "模型大小: ~10GB"
echo "预计时间取决于网络速度..."

# 使用 git-lfs 克隆模型
GIT_CMD="git clone https://huggingface.co/Qwen/$MODEL_NAME"

if $GIT_CMD; then
    echo ""
    echo "========================================"
    echo "✅ 模型 $MODEL_NAME 下载成功！"
    echo "========================================"

    # 检查模型完整性
    echo ""
    echo "检查模型文件..."
    ls -la "$MODEL_DIR/$MODEL_NAME" | head -30

    # 更新配置文件
    if [ ! -f "$PROJECT_ROOT/qwen_stroke_extractor/deploy/config.sh" ]; then
        cat > "$PROJECT_ROOT/qwen_stroke_extractor/deploy/config.sh" <<EOL
# qwen_stroke_extractor 配置文件
export QWEN_MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
export QWEN_MODEL_NAME="$MODEL_NAME"
EOL
        echo "配置文件已创建"
    fi

else
    echo ""
    echo "========================================"
    echo "❌ 模型下载失败！"
    echo "========================================"
    echo "可能的原因:"
    echo "1. 网络连接问题"
    echo "2. HuggingFace 访问限制"
    echo "3. 磁盘空间不足"
    echo ""
    echo "解决方法:"
    echo "1. 检查网络连接"
    echo "2. 使用 ModelScope 下载: bash 02_download_model.sh --modelscope"
    echo "3. 检查磁盘空间: df -h"
    echo "4. 尝试直接在 HuggingFace 下载 zip 文件"
    exit 1
fi

echo ""
echo "下一步: 运行 bash 03_test_local_model.sh"

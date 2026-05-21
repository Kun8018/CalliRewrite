#!/bin/bash
# 服务器部署脚本 1/5 - 安装依赖
# 使用方法: bash 01_install_dependencies.sh

set -e

echo "========================================"
echo "步骤 1: 安装 qwen_stroke_extractor 依赖"
echo "========================================"

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "项目根目录: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# 检查是否有虚拟环境
if [ -d "calli_train_env" ]; then
    echo "检测到现有的虚拟环境，正在激活..."
    source calli_train_env/bin/activate
elif [ -d ".venv" ]; then
    echo "检测到现有的虚拟环境，正在激活..."
    source .venv/bin/activate
else
    echo "未找到虚拟环境，请先创建虚拟环境"
    echo "可以运行: python3 -m venv calli_train_env"
    exit 1
fi

echo "当前 Python: $(which python3)"
echo "Python 版本: $(python3 --version)"

# 升级 pip
echo ""
echo "升级 pip..."
pip install --upgrade pip

# 安装依赖
echo ""
echo "安装核心依赖..."
cd "$PROJECT_ROOT/qwen_stroke_extractor"

# 检查 requirements.txt 是否存在
if [ ! -f "requirements.txt" ]; then
    echo "错误: 找不到 requirements.txt"
    exit 1
fi

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装额外的依赖
echo ""
echo "安装额外的依赖..."
pip install dashscope -i https://pypi.tuna.tsinghua.edu.cn/simple  # 阿里云 SDK
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple  # ModelScope 下载工具（可选）

echo ""
echo "========================================"
echo "依赖安装完成！"
echo "========================================"

# 验证安装
echo ""
echo "验证安装..."
python3 -c "import torch; print('torch OK:', torch.__version__)"
python3 -c "import transformers; print('transformers OK:', transformers.__version__)"
python3 -c "import numpy; print('numpy OK:', numpy.__version__)"
python3 -c "import PIL; print('PIL OK:', PIL.__version__)"

echo ""
echo "下一步: 运行 bash 02_download_model.sh"

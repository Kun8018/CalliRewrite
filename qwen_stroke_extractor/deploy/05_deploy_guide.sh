#!/bin/bash
# 服务器部署脚本 5/5 - 快速部署指南和检查清单
# 使用方法: bash 05_deploy_guide.sh

echo "========================================"
echo "CalliRewrite - qwen_stroke_extractor 部署指南"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo ""
echo "📋 完整部署步骤"
echo "========================================"

cat <<END
步骤 1: 环境准备
  1. 确认服务器有 NVIDIA GPU（建议 16GB+ 显存）
  2. 确认已安装 CUDA
  3. 确认有足够的磁盘空间（至少 80GB）
  4. 确认项目已克隆到服务器上

步骤 2: 安装依赖
  cd $PROJECT_ROOT
  cd qwen_stroke_extractor/deploy
  bash 01_install_dependencies.sh

步骤 3: 下载模型
  bash 02_download_model.sh [Qwen-VL | Qwen-VL-Plus]
  # 推荐先下载 Qwen-VL 较小的模型测试

步骤 4: 测试本地模型
  bash 03_test_local_model.sh

步骤 5: 批量处理图像
  bash 04_batch_process.sh [图像目录]

END

echo ""
echo "⚡ 快速一键部署（推荐）"
echo "========================================"

cat <<END
cd $PROJECT_ROOT/qwen_stroke_extractor/deploy
bash 01_install_dependencies.sh
bash 02_download_model.sh Qwen-VL  # 先下载小模型测试
bash 03_test_local_model.sh       # 测试是否正常工作
bash 04_batch_process.sh          # 批量处理
END

echo ""
echo "🔍 环境检查清单"
echo "========================================"

# 检查 CUDA
echo ""
echo "检查 CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU 检测到:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "❌ 未检测到 NVIDIA GPU，将使用 CPU（很慢）"
fi

# 检查 Python 环境
echo ""
echo "检查 Python 环境..."
if [ -d "$PROJECT_ROOT/calli_train_env" ]; then
    echo "✅ 虚拟环境存在: calli_train_env"
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "  ⚠️ 虚拟环境未激活，运行: source calli_train_env/bin/activate"
    else
        echo "  ✅ 虚拟环境已激活"
    fi
else
    echo "❌ 虚拟环境不存在，运行: python3 -m venv calli_train_env"
fi

# 检查磁盘空间
echo ""
echo "检查磁盘空间..."
if command -v df &> /dev/null; then
    echo "磁盘使用情况:"
    df -h "$PROJECT_ROOT" | tail -1
fi

echo ""
echo "💡 常见问题"
echo "========================================"

cat <<END
Q: 模型下载太慢怎么办？
A: 使用 ModelScope 国内镜像
   - 修改 02_download_model.sh 使用 ModelScope
   - 或使用阿里云 OSS 加速器

Q: 显存不足怎么办？
A: 1. 使用更小的模型（Qwen-VL 而非 Qwen-VL-Plus）
   2. 使用 CPU 模式（很慢但可行）
   3. 使用 API 模式（推荐）

Q: 如何使用 API 模式？
A: 1. 注册阿里云账号，获取 API Key
   2. 修改 examples.py 使用 --use-api
   3. 参考 qwen_stroke_extractor/README.md

END

echo ""
echo "📚 更多信息"
echo "========================================"

cat <<END
- 详细文档: $PROJECT_ROOT/qwen_stroke_extractor/README.md
- 项目主页: $PROJECT_ROOT
- 测试脚本: $PROJECT_ROOT/qwen_stroke_extractor/quick_test.py

END

echo ""
echo "🚀 准备好部署了？现在开始执行步骤 1 吧！"

#!/bin/bash
# qwen_stroke_extractor 一键部署脚本
# 使用方法: bash deploy_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "CalliRewrite qwen_stroke_extractor 一键部署"
echo "========================================"

cd "$SCRIPT_DIR"

# 检查是否已经运行过
if [ -f ".deployed" ]; then
    echo ""
    echo "检测到已经部署过"
    echo "重新部署前，运行: rm -f .deployed"
    echo "或者直接运行: bash 03_test_local_model.sh 测试"
    exit 0
fi

echo ""
echo "📋 部署前检查..."

# 运行检查脚本
bash 05_deploy_guide.sh

echo ""
read -p "确认开始部署吗？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消部署"
    exit 1
fi

echo ""
echo "========================================"
echo "步骤 1/5: 安装依赖..."
echo "========================================"
bash 01_install_dependencies.sh

echo ""
echo "========================================"
echo "步骤 2/5: 下载模型..."
echo "========================================"
bash 02_download_model.sh Qwen-VL

echo ""
echo "========================================"
echo "步骤 3/5: 测试本地模型..."
echo "========================================"
bash 03_test_local_model.sh

echo ""
echo "========================================"
echo "步骤 4/5: 批量处理测试（可选）"
echo "========================================"
read -p "是否进行批量处理测试？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash 04_batch_process.sh
fi

echo ""
echo "========================================"
echo "✅ 部署完成！"
echo "========================================"

# 标记已部署
touch .deployed

echo ""
echo "📝 使用提示"
echo "----------------------------------------"
echo "1. 测试单张图像:"
echo "   cd $PROJECT_ROOT/qwen_stroke_extractor"
echo "   python examples.py /path/to/image.png --model-path models/Qwen-VL"

echo ""
echo "2. 批量处理图像:"
echo "   bash deploy/04_batch_process.sh /path/to/images"

echo ""
echo "3. 查看配置:"
echo "   cat deploy/config.sh"

echo ""
echo "4. 参考文档:"
echo "   cat deploy/README.md"
echo "   cat ../README.md"

echo ""
echo "🎉 部署完成！现在可以开始使用了！"

#!/bin/bash
# 使用 ModelScope 快速下载模型
# 使用方法: bash download_modelscope.sh [Qwen-VL | Qwen-VL-Plus]

set -e

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

echo "========================================"
echo "ModelScope 快速下载"
echo "========================================"
echo "项目根目录: $PROJECT_ROOT"
echo "模型存储路径: $MODEL_DIR"
echo "准备下载: $MODEL_NAME"

# 检查虚拟环境
if [ -d "$PROJECT_ROOT/calli_train_env" ]; then
    source "$PROJECT_ROOT/calli_train_env/bin/activate"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# 确保 modelscope 已安装
echo ""
echo "检查依赖..."
python3 -c "import modelscope" 2>/dev/null || {
    echo "正在安装 modelscope..."
    pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
}

# 创建目录
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

# 检查是否已经下载
if [ -d "$MODEL_NAME" ]; then
    echo "模型 $MODEL_NAME 已存在"
    exit 0
fi

# 使用 Python 下载
echo ""
echo "开始下载..."

python3 <<END
from modelscope import snapshot_download
import os

model_id = "qwen/$MODEL_NAME"
print(f"正在从 ModelScope 下载: {model_id}")
print()

try:
    model_dir = snapshot_download(
        model_id,
        cache_dir="$MODEL_DIR",
    )
    print()
    print("✅ 下载成功！")
    print(f"模型位置: {model_dir}")
    print()

    # 创建软链接
    target_link = os.path.join("$MODEL_DIR", "$MODEL_NAME")
    if not os.path.exists(target_link):
        import glob
        # 查找实际的下载位置
        cache_path = os.path.join("$MODEL_DIR", "qwen", "$MODEL_NAME")
        if os.path.exists(cache_path):
            versions = sorted(os.listdir(cache_path))
            if versions:
                latest_dir = os.path.join(cache_path, versions[-1])
                os.symlink(latest_dir, target_link)
                print(f"✅ 已创建软链接: {target_link}")

    # 更新配置文件
    config_path = "$SCRIPT_DIR/config.sh"
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write("# qwen_stroke_extractor 配置文件\n")
            f.write(f"export QWEN_MODEL_PATH=\"$MODEL_DIR/$MODEL_NAME\"\n")
            f.write(f"export QWEN_MODEL_NAME=\"$MODEL_NAME\"\n")
        print(f"✅ 配置文件已创建: {config_path}")

    print()
    print("========================================")
    print("下载完成！")
    print("========================================")
    print()
    print("下一步: 运行 bash 03_test_local_model.sh")
    print()

except Exception as e:
    print(f"❌ 下载失败: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)
END

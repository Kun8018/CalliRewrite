#!/bin/bash
# 简单下载脚本 - 使用 HuggingFace Hub（不依赖 ModelScope）
# 使用方法: bash download_simple.sh [--mirror {hf-mirror|byte-trust|fastgit|hf.co}] [Qwen-VL | Qwen-VL-Plus]

set -e

echo "========================================"
echo "简单下载（HuggingFace Hub + 可切换镜像）"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/qwen_stroke_extractor/models"

# 默认参数
MIRROR="hf-mirror"
MODEL_NAME="Qwen-VL"

# 解析参数
while [ "$#" -gt 0 ]; do
    case "$1" in
        --mirror)
            if [ "$#" -gt 1 ]; then
                case "$2" in
                    hf-mirror)
                        MIRROR="hf-mirror"
                        ;;
                    byte-trust)
                        MIRROR="byte-trust"
                        ;;
                    fastgit)
                        MIRROR="fastgit"
                        ;;
                    hf-co)
                        MIRROR="hf.co"
                        ;;
                    *)
                        echo "警告: 未知的镜像源 $2，使用默认的 hf-mirror"
                        MIRROR="hf-mirror"
                        ;;
                esac
                shift
            else
                echo "警告: --mirror 需要参数"
            fi
            shift
            ;;
        Qwen-VL|Qwen-VL-Plus)
            MODEL_NAME="$1"
            shift
            ;;
        *)
            echo "警告: 未知参数 $1，忽略"
            shift
            ;;
    esac
done

# 镜像配置
declare -A MIRROR_CONFIGS
MIRROR_CONFIGS["hf-mirror"]="https://hf-mirror.com"
MIRROR_CONFIGS["byte-trust"]="https://hf-mirror.byte-trust.com"
MIRROR_CONFIGS["fastgit"]="https://huggingface.co.fastgit.org"
MIRROR_CONFIGS["hf-co"]="https://huggingface.co"

echo "项目根目录: $PROJECT_ROOT"
echo "模型存储路径: $MODEL_DIR"
echo "准备下载: $MODEL_NAME"
echo "使用镜像源: $MIRROR (${MIRROR_CONFIGS[$MIRROR]})"

# 检查虚拟环境
if [ -d "$PROJECT_ROOT/calli_train_env" ]; then
    source "$PROJECT_ROOT/calli_train_env/bin/activate"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# 确保 huggingface_hub 已安装
echo ""
echo "检查依赖..."
python3 -c "from huggingface_hub import snapshot_download" 2>/dev/null || {
    echo "正在安装 huggingface_hub..."
    pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
}

# 创建目录
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

# 检查是否已经下载
if [ -d "$MODEL_NAME" ]; then
    echo "模型 $MODEL_NAME 已存在"
    exit 0
fi

# 设置镜像加速
export HF_ENDPOINT="${MIRROR_CONFIGS[$MIRROR]}"

# 使用 Python 下载
echo ""
echo "开始下载..."

python3 <<END
from huggingface_hub import snapshot_download
import os

model_id = "Qwen/$MODEL_NAME"
print(f"正在从 $MIRROR 下载: {model_id}")
print()

try:
    model_dir = snapshot_download(
        model_id,
        local_dir="$MODEL_NAME",
        local_dir_use_symlinks=False,
        max_workers=4,
    )
    print()
    print("✅ 下载成功！")
    print(f"模型位置: {os.path.abspath(model_dir)}")
    print()

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

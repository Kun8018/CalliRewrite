#!/bin/bash
# 服务器部署脚本 2/5 - 下载模型
# 使用方法: bash 02_download_model.sh [--modelscope] [model_name]
# 可选模型: Qwen-VL-Plus, Qwen-VL
# 示例: bash 02_download_model.sh --modelscope Qwen-VL

set -e

echo "========================================"
echo "步骤 2: 下载千问模型"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/qwen_stroke_extractor/models"

# 默认参数
USE_MODELSCOPE=false
MODEL_NAME="Qwen-VL"

# 解析参数
while [ "$#" -gt 0 ]; do
    case "$1" in
        --modelscope)
            USE_MODELSCOPE=true
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

echo "项目根目录: $PROJECT_ROOT"
echo "模型存储路径: $MODEL_DIR"
echo "准备下载: $MODEL_NAME"
echo "下载方式: $([ "$USE_MODELSCOPE" = true ] && echo "ModelScope" || echo "HuggingFace")"

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

if [ "$USE_MODELSCOPE" = true ]; then
    echo ""
    echo "========================================"
    echo "使用 ModelScope 下载..."
    echo "========================================"

    # 检查 modelscope 是否已安装
    if ! python3 -c "import modelscope" 2>/dev/null; then
        echo "正在安装 modelscope..."
        pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi

    echo "开始从 ModelScope 下载 $MODEL_NAME..."
    echo "模型大小: ~10GB"
    echo "预计时间取决于网络速度..."

    # 使用 Python 下载
    python3 <<END
from modelscope import snapshot_download
import os

model_id = "qwen/$MODEL_NAME"
print(f"正在下载: {model_id}")

try:
    model_dir = snapshot_download(model_id, cache_dir="$MODEL_DIR")
    print(f"\n✅ 下载成功！")
    print(f"模型位置: {model_dir}")

    # 创建软链接
    target_link = os.path.join("$MODEL_DIR", "$MODEL_NAME")
    if not os.path.exists(target_link):
        import glob
        # 查找实际的下载位置
        cache_dir = os.path.join("$MODEL_DIR", "qwen", "$MODEL_NAME")
        if os.path.exists(cache_dir):
            versions = sorted(os.listdir(cache_dir))
            if versions:
                latest_dir = os.path.join(cache_dir, versions[-1])
                os.symlink(latest_dir, target_link)
                print(f"已创建软链接: {target_link}")

except Exception as e:
    print(f"❌ 下载失败: {e}")
    import sys
    sys.exit(1)
END

else
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
            echo "或者使用 ModelScope: bash 02_download_model.sh --modelscope"
            exit 1
        fi

        git lfs install
    fi

    echo ""
    echo "开始从 HuggingFace 下载 $MODEL_NAME..."
    echo "模型大小: ~10GB"
    echo "如果太慢，使用 ModelScope: bash 02_download_model.sh --modelscope"
    echo "预计时间取决于网络速度..."

    # 使用 git-lfs 克隆模型
    GIT_CMD="git clone https://huggingface.co/Qwen/$MODEL_NAME"

    if $GIT_CMD; then
        echo ""
        echo "========================================"
        echo "✅ 模型 $MODEL_NAME 下载成功！"
        echo "========================================"
    else
        echo ""
        echo "========================================"
        echo "❌ HuggingFace 下载失败！"
        echo "========================================"
        echo "推荐使用 ModelScope 下载:"
        echo "  bash 02_download_model.sh --modelscope"
        exit 1
    fi
fi

# 检查模型完整性
echo ""
echo "检查模型文件..."
if [ -d "$MODEL_NAME" ]; then
    ls -la "$MODEL_DIR/$MODEL_NAME" | head -30
else
    # 检查 ModelScope 的下载位置
    echo "查找 ModelScope 下载的模型..."
    if [ -d "$MODEL_DIR/qwen/$MODEL_NAME" ]; then
        latest_version=$(ls -1 "$MODEL_DIR/qwen/$MODEL_NAME" | tail -1)
        if [ -n "$latest_version" ]; then
            actual_path="$MODEL_DIR/qwen/$MODEL_NAME/$latest_version"
            echo "找到模型在: $actual_path"
            ln -sf "$actual_path" "$MODEL_DIR/$MODEL_NAME"
            echo "已创建软链接: $MODEL_DIR/$MODEL_NAME"
        fi
    fi
fi

# 更新配置文件
if [ ! -f "$PROJECT_ROOT/qwen_stroke_extractor/deploy/config.sh" ]; then
    cat > "$PROJECT_ROOT/qwen_stroke_extractor/deploy/config.sh" <<EOL
# qwen_stroke_extractor 配置文件
export QWEN_MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
export QWEN_MODEL_NAME="$MODEL_NAME"
EOL
    echo "配置文件已创建"
fi

echo ""
echo "下一步: 运行 bash 03_test_local_model.sh"

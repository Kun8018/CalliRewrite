#!/bin/bash
# Lightweight 模型推理 — v2
# 用法:
#   bash inference.sh                                          # 用默认 ckpt/input
#   bash inference.sh --input path/to/image.png
#   bash inference.sh --input ../seq_extract/sample_inputs/clean_line_drawings/duck.png
#   bash inference.sh --checkpoint output_ar_phase1_v2/model_best.pth --input ../some_img.png
set -e

# 显式指定 conda env 的 python，避免被 /home/<user>/.local 里的旧 PyTorch/numpy 抢走
CONDA_ENV=/data1/Calliwrite/kun/CalliRewrite/calli_train_env
PY=$CONDA_ENV/bin/python
export PYTHONNOUSERSITE=1

cd "$(dirname "$0")"

# 默认参数
CHECKPOINT="output_ar_phase2_v2/model_best.pth"
RENDERER_CKPT="output_renderer/raster_unit_pretrained.pth"
INPUT="../seq_extract/outputs/__new_train_phase_2/0.png"
OUTPUT_DIR="inference_output_v2"
DEVICE="cuda:0"
MAX_CONSECUTIVE_LIFTS=3
MAX_ROUNDS=1
INIT_CURSOR_STRATEGY="stroke"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint)        CHECKPOINT="$2"; shift 2;;
    --renderer_ckpt)     RENDERER_CKPT="$2"; shift 2;;
    --input)             INPUT="$2"; shift 2;;
    --output_dir)        OUTPUT_DIR="$2"; shift 2;;
    --device)            DEVICE="$2"; shift 2;;
    --max_consecutive_lifts) MAX_CONSECUTIVE_LIFTS="$2"; shift 2;;
    --max_rounds)        MAX_ROUNDS="$2"; shift 2;;
    --init_cursor_strategy) INIT_CURSOR_STRATEGY="$2"; shift 2;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

# 自动 fallback：如果 phase2 ckpt 不存在，用 phase1 best
if [ ! -f "$CHECKPOINT" ]; then
  echo "[warn] $CHECKPOINT 不存在，回退到 output_ar_phase1_v2/model_best.pth"
  CHECKPOINT="output_ar_phase1_v2/model_best.pth"
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "[error] 没有可用的 model checkpoint。请先训练 phase1 或 phase2。"
  exit 1
fi

if [ ! -f "$RENDERER_CKPT" ]; then
  echo "[error] 没有 renderer ckpt: $RENDERER_CKPT"
  echo "请先运行 bash pretrain_renderer.sh"
  exit 1
fi

echo "Checkpoint:     $CHECKPOINT"
echo "Renderer ckpt:  $RENDERER_CKPT"
echo "Input:          $INPUT"
echo "Output dir:     $OUTPUT_DIR"

$PY inference.py \
  --checkpoint "$CHECKPOINT" \
  --renderer_ckpt "$RENDERER_CKPT" \
  --input "$INPUT" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --max_consecutive_lifts "$MAX_CONSECUTIVE_LIFTS" \
  --max_rounds "$MAX_ROUNDS" \
  --init_cursor_strategy "$INIT_CURSOR_STRATEGY"

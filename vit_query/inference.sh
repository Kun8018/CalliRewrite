#!/bin/bash
# ViT Query 模型推理脚本

# Activate conda environment
conda activate /data1/Calliwrite/kun/CalliRewrite/calli_train_env

cd "$(dirname "$0")"

# 默认参数
CHECKPOINT="output_ar_phase2/model_best.pth"
INPUT="../seq_extract/outputs/__new_train_phase_2/0.png"
OUTPUT_DIR="inference_output_phase2"
ARCH="autoregressive"
DEVICE="cuda:0"
MAX_CONSECUTIVE_LIFTS=10
MAX_ROUNDS=8

# 解析命令行参数（如果有）
while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --input)
      INPUT="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --max_consecutive_lifts)
      MAX_CONSECUTIVE_LIFTS="$2"
      shift 2
      ;;
    --max_rounds)
      MAX_ROUNDS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 运行推理
python inference.py \
  --checkpoint "$CHECKPOINT" \
  --arch "$ARCH" \
  --input "$INPUT" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --max_consecutive_lifts "$MAX_CONSECUTIVE_LIFTS" \
  --max_rounds "$MAX_ROUNDS"

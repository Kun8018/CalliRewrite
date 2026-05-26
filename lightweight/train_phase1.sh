#!/bin/bash
# Lightweight (ResNet) Phase 1 Training (QuickDraw)

cd "$(dirname "$0")"

# Download data first if not present
if [ ! -d "datasets/QuickDraw-clean" ]; then
  echo "Downloading QuickDraw data..."
  python download_quickdraw_clean.py
fi

python train.py \
  --arch autoregressive \
  --phase 1 \
  --dataset_root datasets \
  --output_dir output_ar_phase1 \
  --image_size 256 \
  --max_seq_len 100 \
  --chunk_len 8 \
  --chunks_per_sample 4 \
  --batch_size 8 \
  --epochs 50 \
  --device cuda:1 \
  --use-tensorboard

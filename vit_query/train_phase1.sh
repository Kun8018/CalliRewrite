#!/bin/bash
# ViT Query Phase 1 Training (QuickDraw)

cd "$(dirname "$0")"

python train.py \
  --arch autoregressive \
  --phase 1 \
  --dataset_root ../seq_extract/datasets \
  --output_dir output_ar_phase1 \
  --mode seq7 \
  --img_size 224 \
  --seq_len 100 \
  --chunk_len 8 \
  --chunks_per_sample 4 \
  --batch_size 8 \
  --epochs 50 \
  --device cuda:0 \
  --use-tensorboard

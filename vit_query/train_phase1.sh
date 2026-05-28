#!/bin/bash
# ViT Query Phase 1 Training (QuickDraw)

# Activate conda environment
conda activate /data1/Calliwrite/kun/CalliRewrite/calli_train_env

cd "$(dirname "$0")"

python train.py \
  --arch autoregressive \
  --phase 1 \
  --dataset_root ../seq_extract/datasets \
  --output_dir output_ar_phase1 \
  --mode seq7 \
  --img_size 224 \
  --seq_len 100 \
  --embed_dim 192 \
  --chunk_len 16 \
  --chunks_per_sample 4 \
  --batch_size 64 \
  --epochs 50 \
  --num_workers 16 \
  --device cuda:0 \
  --use-tensorboard

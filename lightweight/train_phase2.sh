#!/bin/bash
# Lightweight (ResNet) Phase 2 Finetuning — v2 可微 rollout
# 需要：raster_unit_pretrained.pth + phase1 best ckpt
set -e

conda activate /data1/Calliwrite/kun/CalliRewrite/calli_train_env

cd "$(dirname "$0")"

if [ ! -f "output_renderer/raster_unit_pretrained.pth" ]; then
  echo "Renderer ckpt not found, run pretrain_renderer.sh first."
  exit 1
fi

if [ ! -f "output_ar_phase1_v2/model_best.pth" ]; then
  echo "Phase 1 ckpt not found, run train_phase1.sh first."
  exit 1
fi

python train.py \
  --phase 2 \
  --data_dir ../seq_extract/outputs/__new_train_phase_2 \
  --renderer_ckpt output_renderer/raster_unit_pretrained.pth \
  --phase1_checkpoint output_ar_phase1_v2/model_best.pth \
  --output_dir output_ar_phase2_v2 \
  --image_size 256 \
  --max_seq_len 48 \
  --patch_size 64 \
  --raster_size 128 \
  --d_model 256 \
  --hidden_dim 256 \
  --batch_size 12 \
  --epochs 100 \
  --lr 1e-4 \
  --num_workers 8 \
  --device cuda:2 \
  --use_tensorboard

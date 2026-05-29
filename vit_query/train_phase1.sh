#!/bin/bash
# ViT-B/16 (ImageNet pretrained) Phase 1 Training — v2 可微 rollout
set -e

conda activate /data1/Calliwrite/kun/CalliRewrite/calli_train_env

cd "$(dirname "$0")"

if [ ! -d "datasets/QuickDraw-clean" ] && [ ! -d "../seq_extract/datasets/QuickDraw-clean" ]; then
  echo "QuickDraw-clean not found, please prepare datasets first."
  exit 1
fi

if [ ! -f "output_renderer/raster_unit_pretrained.pth" ]; then
  echo "Renderer ckpt not found, run pretrain_renderer.sh first."
  exit 1
fi

python train.py \
  --phase 1 \
  --dataset_root ../seq_extract/datasets \
  --renderer_ckpt output_renderer/raster_unit_pretrained.pth \
  --output_dir output_ar_phase1_v2 \
  --image_size 224 \
  --max_seq_len 48 \
  --patch_size 64 \
  --raster_size 128 \
  --d_model 256 \
  --hidden_dim 256 \
  --batch_size 8 \
  --epochs 50 \
  --lr 1e-4 \
  --ss_prob_start 1.0 \
  --ss_prob_end 0.0 \
  --num_workers 8 \
  --device cuda:0 \
  --use_tensorboard

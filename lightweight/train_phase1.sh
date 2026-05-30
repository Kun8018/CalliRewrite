#!/bin/bash
# Lightweight (ResNet) Phase 1 Training — v2 可微 rollout + DDP 4 卡
# 需要预先运行 pretrain_renderer.sh 得到 raster_unit_pretrained.pth
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

# 4 卡 DDP，单卡 batch=12 → 全局 batch=48
# max_items_per_category=5000 → 共 5 万样本（10 类）/ epoch
# cache_size=50000 → 全 cache 到内存
torchrun --standalone --nproc_per_node=4 train.py \
  --phase 1 \
  --dataset_root ../seq_extract/datasets \
  --renderer_ckpt output_renderer/raster_unit_pretrained.pth \
  --output_dir output_ar_phase1_v2 \
  --image_size 256 \
  --max_seq_len 48 \
  --patch_size 64 \
  --raster_size 128 \
  --d_model 256 \
  --hidden_dim 256 \
  --max_items_per_category 5000 \
  --cache_size 50000 \
  --batch_size 12 \
  --epochs 50 \
  --lr 1e-4 \
  --ss_prob_start 1.0 \
  --ss_prob_end 0.0 \
  --num_workers 8 \
  --use_tensorboard

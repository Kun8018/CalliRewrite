#!/bin/bash
# ViT Query Phase 2 Finetuning (Calligraphy)
# 使用 NeuralRenderer 进行无监督 fine-tune

# Activate conda environment
conda activate /data1/Calliwrite/kun/CalliRewrite/calli_train_env

cd "$(dirname "$0")"

# 使用 autoregressive 模式（带 patch encoder 的新架构）
python train.py \
  --arch autoregressive \
  --phase 2 \
  --data_dir ../seq_extract/outputs/__new_train_phase_2 \
  --phase1_checkpoint output_ar_phase1/model_best.pth \
  --output_dir output_ar_phase2 \
  --mode seq7 \
  --img_size 224 \
  --seq_len 100 \
  --embed_dim 192 \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-4 \
  --num_workers 16 \
  --device cuda:0 \
  --use-tensorboard

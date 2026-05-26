#!/bin/bash
# Lightweight (ResNet) Phase 2 Finetuning (Calligraphy)

# Activate conda environment
conda activate /data1/Calliwrite/kun/CalliRewrite/calli_train_env

cd "$(dirname "$0")"

python train.py \
  --arch autoregressive \
  --phase 2 \
  --data_dir ../seq_extract/outputs/__new_train_phase_2 \
  --phase1_checkpoint output_ar_phase1/model_best.pth \
  --output_dir output_ar_phase2 \
  --image_size 256 \
  --max_seq_len 100 \
  --chunk_len 16 \
  --chunks_per_sample 4 \
  --batch_size 128 \
  --epochs 50 \
  --num_workers 16 \
  --device cuda:1 \
  --use-tensorboard

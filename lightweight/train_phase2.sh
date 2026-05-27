#!/bin/bash
# Lightweight (ResNet) Phase 2 Finetuning (Calligraphy)
# 使用 NeuralRenderer 进行无监督 fine-tune
#
# 注意：
#   - 推荐使用 --arch=oneshot 模式，这样可以通过 NeuralRenderer 进行端到端可微训练
#   - 如果使用 --arch=autoregressive，则无法通过渲染损失进行梯度回流（推理循环不可微）

# Activate conda environment
conda activate /data1/Calliwrite/kun/CalliRewrite/calli_train_env

cd "$(dirname "$0")"

# 方案 1：使用 oneshot 模式（推荐用于 phase2，可以端到端训练）
python train.py \
    --arch oneshot \
    --phase 2 \
    --data_dir ../seq_extract/outputs/__new_train_phase_2 \
    --phase1_checkpoint output_ar_phase1/model_best.pth \
    --output_dir output_ar_phase2 \
    --image_size 256 \
    --max_seq_len 100 \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --num_workers 16 \
    --device cuda:1 \
    --use-tensorboard

# 方案 2：使用 autoregressive 模式（不推荐用于 phase2，无法通过渲染损失训练）
# python train.py \
#     --arch autoregressive \
#     --phase 2 \
#     --data_dir ../seq_extract/outputs/__new_train_phase_2 \
#     --phase1_checkpoint output_ar_phase1/model_best.pth \
#     --output_dir output_ar_phase2 \
#     --image_size 256 \
#     --max_seq_len 100 \
#     --chunk_len 8 \
#     --chunks_per_sample 4 \
#     --batch_size 64 \
#     --epochs 100 \
#     --num_workers 16 \
#     --device cuda:1 \
#     --use-tensorboard

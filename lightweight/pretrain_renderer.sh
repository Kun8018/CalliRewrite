#!/bin/bash
# 预训练 NeuralRasterizor.RasterUnit
# 用法: bash pretrain_renderer.sh
# 输出: output_renderer/raster_unit_pretrained.pth
set -e

conda activate /data1/Calliwrite/kun/CalliRewrite/calli_train_env

cd "$(dirname "$0")"

mkdir -p output_renderer

python pretrain_renderer.py \
  --output_path output_renderer/raster_unit_pretrained.pth \
  --steps 100000 \
  --batch_size 64 \
  --lr 1e-4 \
  --num_workers 4 \
  --log_every 200 \
  --save_every 10000 \
  --device cuda:0

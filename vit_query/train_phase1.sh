#!/bin/bash
# ViT-B/16 (ImageNet pretrained) Phase 1 Training — v2 可微 rollout
# 需要预先运行 pretrain_renderer.sh
# 单卡训练；若要多卡 DDP，把 $PY train.py 改成:
#   $PY -m torch.distributed.run --standalone --nproc_per_node=4 train.py ...
# 同时去掉 --device 参数。
set -e

# 显式指定 conda env 的 python，避免被 /home/<user>/.local 里的旧 PyTorch/numpy 抢走
CONDA_ENV=/data1/Calliwrite/kun/CalliRewrite/calli_train_env
PY=$CONDA_ENV/bin/python
export PYTHONNOUSERSITE=1

cd "$(dirname "$0")"

DATASET_ROOT=""
if compgen -G "datasets/QuickDraw-clean/train/*.npz" > /dev/null && \
   compgen -G "datasets/QuickDraw-clean/test/*.npz" > /dev/null; then
  DATASET_ROOT="datasets"
elif compgen -G "../seq_extract/datasets/QuickDraw-clean/train/*.npz" > /dev/null && \
     compgen -G "../seq_extract/datasets/QuickDraw-clean/test/*.npz" > /dev/null; then
  DATASET_ROOT="../seq_extract/datasets"
else
  echo "QuickDraw-clean npz files not found."
  echo "Expected one of:"
  echo "  vit_query/datasets/QuickDraw-clean/{train,test}/*.npz"
  echo "  seq_extract/datasets/QuickDraw-clean/{train,test}/*.npz"
  exit 1
fi

if [ ! -f "output_renderer/raster_unit_pretrained.pth" ]; then
  echo "Renderer ckpt not found, run pretrain_renderer.sh first."
  exit 1
fi

# 5090 32G 单卡：batch=24 + bf16 AMP
# max_items_per_category=5000 → 5 万样本 / epoch
# cache_size=0 → 关闭 cache（worker fork 会复制 cache，易 OOM）
# num_workers=8 → 配合 cache=0，减轻 CPU 数据瓶颈
# phase1 先做稳定的序列监督 warmup；raster/perceptual 放到后续微调再打开。
$PY train.py \
  --phase 1 \
  --dataset_root "$DATASET_ROOT" \
  --renderer_ckpt output_renderer/raster_unit_pretrained.pth \
  --output_dir output_ar_phase1_v2 \
  --image_size 224 \
  --max_seq_len 48 \
  --patch_size 64 \
  --raster_size 128 \
  --d_model 256 \
  --hidden_dim 256 \
  --max_items_per_category 5000 \
  --cache_size 0 \
  --batch_size 24 \
  --epochs 50 \
  --lr 3e-5 \
  --grad_clip 0.5 \
  --teacher_forcing_prob 1.0 \
  --w_supervised 1.0 \
  --w_sup_pen 5.0 \
  --w_sup_coord 2.0 \
  --w_sup_param 0.5 \
  --w_sup_tail_pen 1.0 \
  --w_raster 0.0 \
  --w_stroke_num 0.0 \
  --w_outside 0.0 \
  --w_win_outside 0.0 \
  --w_early_pen 0.0 \
  --no_perceptual \
  --no_l1_raster \
  --no_random_init_cursor \
  --num_workers 8 \
  --device cuda:0 \
  --use_tensorboard


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

ensure_tensorboard() {
  if ! "$PY" - <<'PY' >/dev/null 2>&1
from torch.utils.tensorboard import SummaryWriter  # noqa: F401
PY
  then
    echo "TensorBoard not found in $PY, installing..."
    "$PY" -m pip install tensorboard
  fi
}
ensure_tensorboard

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

# SeqExtract-style phase1:
# - ViT global encoder + conv13_c3 local/canvas CNN + closed-loop rollout
# - 90040 training steps, eval every 5000, save every 15000
# - stroke_num weight increases from 0 to 0.5
$PY train.py \
  --phase 1 \
  --dataset_root "$DATASET_ROOT" \
  --renderer_ckpt output_renderer/raster_unit_pretrained.pth \
  --output_dir output_ar_phase1_v2 \
  --image_size 278 \
  --max_seq_len 48 \
  --patch_size 128 \
  --raster_size 128 \
  --d_model 128 \
  --hidden_dim 256 \
  --max_items_per_category 50000 \
  --cache_size 0 \
  --batch_size 12 \
  --num_steps 90040 \
  --eval_every 5000 \
  --save_every 15000 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --decay_power 0.9 \
  --weight_decay 0.0 \
  --grad_clip 1.0 \
  --w_raster 1.0 \
  --w_stroke_num 0.5 \
  --w_stroke_num_end 0.0 \
  --sn_loss_type increasing \
  --w_smoothness 0.0 \
  --w_angle 0.0 \
  --w_outside 10.0 \
  --w_win_outside 10.0 \
  --w_early_pen 0.1 \
  --early_pen_length 7 \
  --w_supervised 0.0 \
  --use_perceptual \
  --use_l1_raster \
  --num_workers 8 \
  --device cuda:0 \
  --use_tensorboard


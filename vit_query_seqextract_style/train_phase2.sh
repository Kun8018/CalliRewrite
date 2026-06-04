#!/bin/bash
# ViT-B/16 Phase 2 Finetuning — v2 可微 rollout，单卡
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

if [ ! -f "output_renderer/raster_unit_pretrained.pth" ]; then
  echo "Renderer ckpt not found, run pretrain_renderer.sh first."
  exit 1
fi

if [ ! -f "output_ar_phase1_v2/model_best.pth" ]; then
  echo "Phase 1 ckpt not found, run train_phase1.sh first."
  exit 1
fi

$PY train.py \
  --phase 2 \
  --data_dir ../seq_extract/outputs/__new_train_phase_2 \
  --renderer_ckpt output_renderer/raster_unit_pretrained.pth \
  --phase1_checkpoint output_ar_phase1_v2/model_best.pth \
  --output_dir output_ar_phase2_v2 \
  --image_size 256 \
  --max_seq_len 48 \
  --patch_size 128 \
  --raster_size 128 \
  --d_model 128 \
  --hidden_dim 256 \
  --batch_size 12 \
  --num_steps 30020 \
  --eval_every 5000 \
  --save_every 5000 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --decay_power 0.9 \
  --weight_decay 0.0 \
  --grad_clip 1.0 \
  --w_raster 1.0 \
  --w_stroke_num 0.5 \
  --sn_loss_type fixed \
  --w_smoothness 0.5 \
  --w_angle 1.0 \
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


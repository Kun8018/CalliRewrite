#!/bin/bash
# Lightweight (ResNet) Phase 1 Training — v2 可微 rollout + DDP 4 卡
# 需要预先运行 pretrain_renderer.sh 得到 raster_unit_pretrained.pth
set -e

# 显式指定 conda env 的 python，避免被 /home/<user>/.local 里的旧 PyTorch/numpy 抢走
CONDA_ENV=/data1/Calliwrite/kun/CalliRewrite/calli_train_env
PY=$CONDA_ENV/bin/python

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
  echo "  lightweight/datasets/QuickDraw-clean/{train,test}/*.npz"
  echo "  seq_extract/datasets/QuickDraw-clean/{train,test}/*.npz"
  exit 1
fi

if [ ! -f "output_renderer/raster_unit_pretrained.pth" ]; then
  echo "Renderer ckpt not found, run pretrain_renderer.sh first."
  exit 1
fi

# 屏蔽用户级 site-packages（避免被 /home/<user>/.local 污染）
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

# 4 卡 DDP，单卡 batch=12 → 全局 batch=48
# max_items_per_category=5000 → 共 5 万样本（10 类）/ epoch
# cache_size=0 → 关闭 dataset 内存 cache（避免 4 rank × 8 worker × 12GB = OOM）
# num_workers=2 → 4 rank × 2 = 8 worker 已够 GPU 不饿，且内存占用可控
# phase1 先做稳定的序列监督 warmup；raster/perceptual 放到后续微调再打开。
$PY -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
  --phase 1 \
  --dataset_root "$DATASET_ROOT" \
  --renderer_ckpt output_renderer/raster_unit_pretrained.pth \
  --output_dir output_ar_phase1_v2 \
  --image_size 256 \
  --max_seq_len 48 \
  --patch_size 64 \
  --raster_size 128 \
  --d_model 256 \
  --hidden_dim 256 \
  --max_items_per_category 5000 \
  --cache_size 0 \
  --batch_size 12 \
  --epochs 50 \
  --lr 3e-5 \
  --grad_clip 0.5 \
  --teacher_forcing_prob 1.0 \
  --teacher_forcing_end 0.3 \
  --teacher_forcing_warmup_epochs 10 \
  --teacher_forcing_decay_epochs 30 \
  --best_metric val_tf \
  --viz_every 10 \
  --viz_category duck \
  --w_supervised 1.0 \
  --w_sup_pen 3.0 \
  --w_sup_pen_up 8.0 \
  --w_sup_coord 2.0 \
  --w_sup_param 0.5 \
  --w_sup_tail_pen 0.5 \
  --w_raster 0.0 \
  --w_stroke_num 0.0 \
  --w_outside 0.0 \
  --w_win_outside 0.0 \
  --w_early_pen 0.0 \
  --no_perceptual \
  --no_l1_raster \
  --no_random_init_cursor \
  --num_workers 2 \
  --use_tensorboard


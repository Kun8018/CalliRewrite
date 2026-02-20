# CalliRewrite 完整训练指南

## 🎯 训练概览

**两阶段训练流程:**

```
阶段1: seq_extract (TensorFlow LSTM)
├─ Phase 1-1: QuickDraw 预训练 (3-5天, V100)
└─ Phase 1-2: 书法微调 (2-3天, V100)
     ↓ 输出: .npy 文件
阶段2: rl_finetune (PyTorch SAC)
└─ RL 优化笔画参数 (1-2天, V100)
     ↓ 输出: 优化后 .npy 文件
```

---

## 📋 准备工作

### 1. 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **GPU** | NVIDIA GTX 1080 Ti (11GB) | V100/A100 (16GB+) |
| **CUDA** | 11.2+ | 11.8 |
| **内存** | 32GB | 64GB+ |
| **存储** | 50GB | 100GB+ |
| **OS** | Ubuntu 18.04+ | Ubuntu 20.04 |

### 2. 检查 GPU

```bash
# 在远程服务器上运行
nvidia-smi

# 应该看到类似输出:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  Tesla V100-SXM2...  Off  | 00000000:00:1E.0 Off |                    0 |
# | N/A   32C    P0    40W / 300W |      0MiB / 16384MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

---

## 🔧 阶段1: seq_extract 训练

### 步骤 1: 创建 Conda 环境

```bash
# 在远程服务器上
cd /path/to/CalliRewrite/seq_extract

# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate CalliRewrite

# 验证 TensorFlow 安装
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

# 应该输出:
# 2.10.0
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**如果 environment.yml 有问题，手动安装:**

```bash
conda create -n CalliRewrite python=3.8 -y
conda activate CalliRewrite

# 安装 CUDA 工具链
conda install cudatoolkit=11.8 cudnn=8.8 -c conda-forge -y

# 安装 TensorFlow
pip install tensorflow-gpu==2.10.0

# 安装其他依赖
pip install numpy==1.24.4 scipy matplotlib opencv-python pillow
pip install cairocffi gizeh
pip install tensorboard
```

### 步骤 2: 准备数据

**选项 A: 使用已有数据 (推荐)**

```bash
# 检查训练数据
ls -lh /path/to/CalliRewrite/seq_extract/data/

# 应该有:
# clean_line_drawings/  - QuickDraw 数据 (Phase 1)
# gb/                   - 书法数据 (Phase 2)
```

**选项 B: 下载 QuickDraw 数据**

```bash
# Phase 1 需要 QuickDraw 数据
mkdir -p data/clean_line_drawings

# 下载脚本 (示例)
# 从 https://github.com/googlecreativelab/quickdraw-dataset
# 下载 .npy 格式数据，解压到 data/clean_line_drawings/
```

### 步骤 3: Phase 1-1 训练 (QuickDraw)

**修改超参数:**

编辑 `hyper_parameters.py`:

```python
hparams = {
    'program_name': 'my_train_phase_1',  # 修改为你的实验名
    'data_set': 'clean_line_drawings',

    'num_steps': 90040,      # 总步数 (可根据GPU减少)
    'batch_size': 12,        # 根据GPU显存调整 (16GB GPU可用12)
    'save_every': 15000,
    'eval_every': 5000,

    'learning_rate': 0.0001,
    'gpus': [0],             # 使用第一块 GPU
}
```

**启动训练:**

```bash
cd /path/to/CalliRewrite/seq_extract

# 方式1: 直接运行
python train_phase_1.py

# 方式2: 后台运行 (推荐)
nohup python train_phase_1.py > train_phase1.log 2>&1 &

# 方式3: 使用 tmux/screen (推荐)
tmux new -s train_seq
python train_phase_1.py
# 按 Ctrl+B 然后 D 分离会话

# 查看日志
tail -f train_phase1.log
```

**监控训练:**

```bash
# 在本地电脑上,通过 SSH 端口转发访问 TensorBoard
ssh -L 6006:localhost:6006 user@remote_server

# 在远程服务器上启动 TensorBoard
conda activate CalliRewrite
tensorboard --logdir=outputs/log/my_train_phase_1

# 本地浏览器打开: http://localhost:6006
```

**检查输出:**

```bash
# 训练过程中会生成:
outputs/
├── snapshot/my_train_phase_1/
│   ├── model-15000.ckpt      # 检查点 15000
│   ├── model-30000.ckpt
│   └── model-90000.ckpt      # 最终模型
├── log/my_train_phase_1/
│   └── events.out.tfevents   # TensorBoard 日志
└── log_img/my_train_phase_1/
    ├── res_128/              # 不同分辨率的预测结果
    ├── res_170/
    └── res_278/
```

### 步骤 4: Phase 1-2 微调 (书法数据)

**修改超参数:**

编辑 `hyper_parameters.py`:

```python
hparams = {
    'program_name': 'my_train_phase_2',
    'data_set': 'gb',  # 书法数据集

    'num_steps': 60000,
    'batch_size': 12,

    # ⚠️ 重要: 指定预训练模型路径
    'pretrained_model': 'outputs/snapshot/my_train_phase_1/model-90000',
}
```

**启动训练:**

```bash
# 后台运行
nohup python train_phase_2.py > train_phase2.log 2>&1 &

# TensorBoard 监控
tensorboard --logdir=outputs/log/my_train_phase_2
```

### 步骤 5: 推理生成 .npy

**训练完成后,使用模型生成笔画序列:**

```bash
cd /path/to/CalliRewrite/seq_extract

# 推理单张图像
python test.py \
  --input_image sample_inputs/calligraphy/永.png \
  --output_npy outputs/永.npy \
  --model_path outputs/snapshot/my_train_phase_2/model-60000

# 批量推理
python test.py \
  --input_dir sample_inputs/calligraphy/ \
  --output_dir outputs/npy_results/ \
  --model_path outputs/snapshot/my_train_phase_2/model-60000
```

---

## 🎮 阶段2: rl_finetune 训练

### 步骤 1: 创建 PyTorch 环境

```bash
cd /path/to/CalliRewrite/rl_finetune

# 创建新环境
conda env create -f environment.yml

# 激活环境
conda activate rl_finetune

# 验证安装
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 应该输出:
# 2.1.1
# True
```

**如果 environment.yml 有问题,手动安装:**

```bash
conda create -n rl_finetune python=3.9 -y
conda activate rl_finetune

# PyTorch (根据你的 CUDA 版本选择)
# CUDA 11.8
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
# pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121

# RL 框架
pip install tianshou==0.5.0
pip install gymnasium==0.28.1

# 其他依赖
pip install numpy scipy matplotlib opencv-python pillow
pip install tensorboard
```

### 步骤 2: 准备数据

**数据格式要求:**

```
data/
├── train_data/
│   ├── 0.png        # 书法图像 (256x256)
│   ├── 0.npy        # 对应的笔画序列 (来自 seq_extract)
│   ├── 1.png
│   ├── 1.npy
│   └── ...
└── test_data/
    ├── 0.png
    ├── 0.npy
    └── ...
```

**转换 seq_extract 输出到 rl_finetune 输入:**

```bash
# 假设 seq_extract 输出在 /path/to/seq_extract/outputs/npy_results/

# 复制到 rl_finetune 数据目录
cp /path/to/seq_extract/outputs/npy_results/*.npy \
   /path/to/rl_finetune/data/train_data/

# 确保对应的 .png 也在同一目录
cp /path/to/calligraphy_images/*.png \
   /path/to/rl_finetune/data/train_data/
```

### 步骤 3: 训练 RL 模型

**选择工具类型:**

项目支持 3 种工具:

1. **brush** (毛笔) - 圆形笔头 + 三角形笔尖
2. **fude** (日式笔) - 椭圆形笔头
3. **marker** (马克笔) - 方形笔头

**使用训练脚本:**

```bash
cd /path/to/CalliRewrite/rl_finetune

# 训练毛笔模型
bash scripts/train_brush.sh

# 或手动运行
python try_tianshou.py \
  --folder_path data/train_data/ \
  --output_path result/brush/ \
  --tool brush \
  --env_num 8 \
  --step_per_epoch 5000 \
  --epoch 20 \
  --gpu 0
```

**脚本内容 (scripts/train_brush.sh):**

```bash
#!/bin/bash

python try_tianshou.py \
  --folder_path data/train_data/ \
  --output_path result/brush/ \
  --visualize_path result/brush/demo/ \
  --tool brush \
  --tool_property tool_property/brush.json \
  --env_num 8 \
  --step_per_epoch 5000 \
  --epoch 20 \
  --batch_size 256 \
  --actor_lr 3e-5 \
  --critic_lr 1e-4 \
  --alpha_lr 1e-4 \
  --tau 0.005 \
  --gamma 0.9 \
  --gpu 0
```

**后台运行:**

```bash
# 使用 nohup
nohup bash scripts/train_brush.sh > train_rl.log 2>&1 &

# 或 tmux
tmux new -s train_rl
bash scripts/train_brush.sh
# Ctrl+B, D 分离

# 查看日志
tail -f train_rl.log
```

**TensorBoard 监控:**

```bash
tensorboard --logdir=result/brush/log

# 本地通过 SSH 访问:
# ssh -L 6007:localhost:6006 user@remote_server
# 浏览器打开: http://localhost:6007
```

### 步骤 4: 训练输出

```bash
result/brush/
├── models/
│   ├── epoch_5_actor.pth      # Actor 网络权重
│   ├── epoch_5_critic1.pth    # Critic1 网络
│   ├── epoch_5_critic2.pth    # Critic2 网络
│   └── epoch_20_*.pth         # 最终模型
├── demo/
│   ├── 0_epoch_5.png          # 训练过程可视化
│   └── ...
├── arrays/
│   ├── 0.npy                  # 优化后的笔画序列
│   └── ...
└── log/
    └── events.out.tfevents    # TensorBoard 日志
```

---

## 🚀 远程训练最佳实践

### 1. 使用 tmux 管理会话

```bash
# 创建新会话
tmux new -s train_session

# 分离会话 (训练在后台继续)
Ctrl+B, D

# 重新连接
tmux attach -t train_session

# 列出所有会话
tmux ls

# 杀死会话
tmux kill-session -t train_session
```

### 2. 监控 GPU 使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用 gpustat (更友好)
pip install gpustat
watch -n 1 gpustat -cpu
```

### 3. 自动恢复训练

创建 `auto_resume.sh`:

```bash
#!/bin/bash

# 检查是否有之前的检查点
if [ -d "outputs/snapshot/my_train_phase_1" ]; then
    LATEST_CKPT=$(ls -t outputs/snapshot/my_train_phase_1/model-*.ckpt | head -1)
    echo "Resume from: $LATEST_CKPT"
    python train_phase_1.py --resume_from=$LATEST_CKPT
else
    echo "Start from scratch"
    python train_phase_1.py
fi
```

### 4. 自动备份检查点

```bash
# 定时备份到云存储 (如 Google Drive, AWS S3)
# 添加到 crontab
crontab -e

# 每6小时备份一次
0 */6 * * * rsync -avz /path/to/outputs/ /backup/location/
```

---

## 📊 训练时间估算

### seq_extract (TensorFlow)

| GPU 型号 | Phase 1 (90k steps) | Phase 2 (60k steps) |
|---------|---------------------|---------------------|
| **V100 (32GB)** | ~4天 | ~2.5天 |
| **RTX 3090 (24GB)** | ~5天 | ~3天 |
| **RTX 4090 (24GB)** | ~3.5天 | ~2天 |
| **A100 (40GB)** | ~3天 | ~1.5天 |

### rl_finetune (PyTorch SAC)

| GPU 型号 | 20 epochs (8 envs) |
|---------|---------------------|
| **V100** | ~30小时 |
| **RTX 3090** | ~36小时 |
| **RTX 4090** | ~24小时 |
| **A100** | ~20小时 |

---

## ⚠️ 常见问题

### 问题 1: CUDA Out of Memory

```bash
# 解决方案: 减小 batch_size
# seq_extract/hyper_parameters.py
'batch_size': 8,  # 从 12 降到 8

# rl_finetune/try_tianshou.py
--batch_size 128  # 从 256 降到 128
--env_num 4       # 从 8 降到 4
```

### 问题 2: TensorFlow 找不到 GPU

```bash
# 检查 CUDA 版本匹配
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"

# 应该看到:
# cuda_version: "11.8"
# cudnn_version: "8"

# 如果不匹配,重新安装:
pip uninstall tensorflow-gpu
pip install tensorflow-gpu==2.10.0
```

### 问题 3: 训练中断后恢复

```bash
# seq_extract 会自动从最新检查点恢复
python train_phase_1.py  # 自动检测并恢复

# rl_finetune 需要手动指定
python try_tianshou.py --resume_path result/brush/models/epoch_5_actor.pth
```

---

## 📦 传输文件 (本地 ↔ 远程)

### 上传数据到远程服务器

```bash
# 从本地传输到远程
scp -r /local/path/CalliRewrite/ user@remote_server:/remote/path/

# 或使用 rsync (增量传输,更快)
rsync -avz --progress /local/path/CalliRewrite/ user@remote_server:/remote/path/
```

### 下载训练结果

```bash
# 下载模型和日志
scp -r user@remote_server:/remote/path/CalliRewrite/seq_extract/outputs/ ./outputs_seq/
scp -r user@remote_server:/remote/path/CalliRewrite/rl_finetune/result/ ./result_rl/
```

---

## 🎓 快速开始 Checklist

### Phase 1: seq_extract

- [ ] SSH 登录远程 GPU 服务器
- [ ] 上传 CalliRewrite 代码到服务器
- [ ] 创建 conda 环境 `CalliRewrite`
- [ ] 验证 TensorFlow GPU 可用
- [ ] 准备 QuickDraw 数据
- [ ] 修改 `hyper_parameters.py`
- [ ] 启动 Phase 1-1 训练 (后台运行)
- [ ] SSH 端口转发 TensorBoard
- [ ] 等待训练完成 (~4天)
- [ ] 启动 Phase 1-2 微调 (~2.5天)
- [ ] 使用 `test.py` 生成 .npy 文件
- [ ] 下载 .npy 文件到本地

### Phase 2: rl_finetune

- [ ] 创建 conda 环境 `rl_finetune`
- [ ] 验证 PyTorch GPU 可用
- [ ] 准备数据 (图像 + .npy 配对)
- [ ] 运行 `scripts/train_brush.sh`
- [ ] TensorBoard 监控
- [ ] 等待训练完成 (~30小时)
- [ ] 下载优化后的 .npy 文件

### 本地使用

- [ ] 将优化后的 .npy 传到 `callibrate/` 或 `mujoco_sim/`
- [ ] 使用 `calibrate.py` 转换为 .npz
- [ ] MuJoCo 仿真测试
- [ ] 真实机器人执行

---

## 💡 小技巧

1. **多GPU训练** (如果有多块GPU):
   ```python
   # hyper_parameters.py
   'gpus': [0, 1, 2, 3],  # 使用4块GPU
   'loop_per_gpu': 1,
   ```

2. **降低训练步数快速测试**:
   ```python
   'num_steps': 5000,  # 从 90040 降到 5000
   'save_every': 1000,
   ```

3. **只训练几个样本验证流程**:
   ```bash
   # 只保留 10 张图像在 data/train_data/
   ls data/train_data/ | head -20 | tail -10  # 保留 5-15.png/npy
   ```

---

**祝训练顺利！有问题随时问我 🚀**
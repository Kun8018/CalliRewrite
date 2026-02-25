# CalliRewrite 完整训练指南

## 🎯 训练概览

**三种训练方案:**

### 🚀 方案 1: 快速开始（推荐，无需训练）

```
书法图像
   ↓
骨架提取 (extract_dense_trajectory.py)
   ↓ 输出: .npz 文件 (秒级)
MuJoCo 仿真 / 机器人控制
```

**优势:** ⚡ 无需 GPU | ⚡ 即时生成 | ⚡ 节省资源

---

### ⚡ 方案 2: 跳过 Phase 1（推荐，需要 GPU）

```
预训练 Phase 1 模型
   ↓
Phase 1-2: 书法微调 (2-3天, V100)
   ↓ 输出: .npy 文件
rl_finetune: RL 优化 (1-2天, V100)
   ↓ 输出: 优化后 .npy 文件
转换为 .npz → MuJoCo / 机器人
```

**优势:** ✅ 节省 5 天训练时间 | ✅ 无需 QuickDraw 数据 | ✅ 高质量笔画

---

### 🔬 方案 3: 完整训练（研究用途）

```
阶段1: seq_extract (TensorFlow LSTM)
├─ Phase 1-1: QuickDraw 预训练 (3-5天, V100)
└─ Phase 1-2: 书法微调 (2-3天, V100)
     ↓ 输出: .npy 文件
阶段2: rl_finetune (PyTorch SAC)
└─ RL 优化笔画参数 (1-2天, V100)
     ↓ 输出: 优化后 .npy 文件
转换为 .npz → MuJoCo / 机器人
```

**优势:** 🎓 完全可控 | 🎓 适合研究改进 | 🎓 自定义数据集

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

## ⚡ 快速开始：跳过 Phase 1（推荐）

**如果你只想生成书法笔画，不需要从头训练整个模型！**

### 为什么跳过 Phase 1？

| 方案 | 时间 | 需要数据 | 优势 |
|------|------|----------|------|
| **完整训练** | ~7-8 天 | QuickDraw (几十GB) + 书法数据 | 完全可控 |
| **跳过 Phase 1** | ~2-3 天 | 只需书法数据 | 节省时间和存储 |

### 步骤 1: 获取预训练模型

**选项 A: 从项目作者获取**

联系项目作者或查找 GitHub Release：

```bash
# 假设下载了预训练模型
# pretrained_phase1_model.zip (约 500MB)

# 解压到 seq_extract/outputs/snapshot/
cd /path/to/CalliRewrite/seq_extract
mkdir -p outputs/snapshot/pretrained_phase_1
unzip pretrained_phase1_model.zip -d outputs/snapshot/pretrained_phase_1/
```

**选项 B: 使用第三方预训练模型**

如果项目没有提供，可以：
1. 在 GitHub Issues 中询问作者
2. 在相关论文的页面查找模型链接
3. 使用类似项目的预训练模型（风险：可能不兼容）

**选项 C: 使用骨架提取方法（无需训练）**

如果找不到预训练模型，使用 `mujoco_sim/extract_dense_trajectory.py`：

```bash
cd /path/to/CalliRewrite/mujoco_sim

# 直接从图像生成 NPZ（不需要 LSTM）
python extract_dense_trajectory.py \
  sample_inputs/calligraphy/永.png \
  --output demo_outputs/永_dense.npz \
  --size 0.12 \
  --depth -0.002
```

---

### 步骤 2: 准备书法数据集

创建 `seq_extract/datasets/2000/` 目录：

```bash
cd /path/to/CalliRewrite/seq_extract
mkdir -p datasets/2000

# 目录结构:
# datasets/2000/
# ├── 0.png     # 书法图像 (256x256)
# ├── 1.png
# ├── 2.png
# └── ...
```

**数据要求：**
- 图像尺寸：256×256 像素
- 格式：PNG/JPG
- 内容：黑底白字 或 白底黑字（代码会自动处理）
- 数量：建议 500-2000 张

**数据来源：**
1. 手写扫描（拍照后裁剪缩放）
2. 书法字体库（字体转图像）
3. 在线书法作品（爬虫获取）

---

### 步骤 3: 配置 Phase 2 训练

编辑 `seq_extract/hyper_parameters.py`：

```python
def get_default_hparams_phase_2():
    hparams = dict(
        program_name='my_train_phase_2',      # 你的实验名
        data_set='gb',                         # 使用书法数据集

        num_steps=30020,                       # 训练步数
        batch_size=12,                         # 根据 GPU 调整
        save_every=5000,

        # ⚠️ 关键：指定预训练模型路径
        pretrained_model='outputs/snapshot/pretrained_phase_1/model-90000',

        # 其他参数保持默认
        ...
    )
```

**如果没有预训练模型**，可以修改 `train_phase_2.py` 从头训练（但效果会差）：

```python
# 在 train_phase_2.py 中注释掉加载预训练模型的代码
# 或设置 pretrained_model=None
```

---

### 步骤 4: 安装环境

**推荐配置（兼容性更好）：**

```bash
cd /path/to/CalliRewrite/seq_extract

# 创建 Python 3.9 环境
conda create -n CalliRewrite python=3.9 -y
conda activate CalliRewrite

# 使用清华镜像安装 TensorFlow 2.12（更容易获取）
pip install tensorflow==2.12.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装其他依赖
pip install numpy==1.24.4 scipy matplotlib opencv-python pillow \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install cairocffi gizeh tensorboard \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证 GPU
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

---

### 步骤 5: 启动 Phase 2 训练

```bash
cd /path/to/CalliRewrite/seq_extract
conda activate CalliRewrite

# 后台运行
nohup python train_phase_2.py > train_phase2.log 2>&1 &

# 或使用 tmux（推荐）
tmux new -s train_phase2
python train_phase_2.py
# Ctrl+B, D 分离

# 查看日志
tail -f train_phase2.log
```

**TensorBoard 监控：**

```bash
# 在远程服务器
tensorboard --logdir=outputs/log/my_train_phase_2

# 本地电脑（SSH 端口转发）
ssh -L 6006:localhost:6006 user@remote_server

# 浏览器打开: http://localhost:6006
```

---

### 步骤 6: 推理生成 .npy

训练完成后（~2-3 天），生成笔画序列：

```bash
cd /path/to/CalliRewrite/seq_extract

# 单张图像推理
python test.py \
  --input_image sample_inputs/calligraphy/永.png \
  --output_npy outputs/永.npy \
  --model_path outputs/snapshot/my_train_phase_2/model-30000

# 批量推理
python test.py \
  --input_dir sample_inputs/calligraphy/ \
  --output_dir outputs/npy_results/ \
  --model_path outputs/snapshot/my_train_phase_2/model-30000
```

---

### 步骤 7: RL 微调（可选）

如果想进一步优化，继续训练 rl_finetune：

```bash
cd /path/to/CalliRewrite/rl_finetune

# 复制 Phase 2 生成的 .npy 到 rl_finetune 数据目录
cp /path/to/seq_extract/outputs/npy_results/*.npy data/train_data/

# 确保有对应的图像
cp /path/to/calligraphy_images/*.png data/train_data/

# 训练
bash scripts/train_brush.sh
```

---

## 📊 时间对比

| 阶段 | 完整训练 | 跳过 Phase 1 |
|------|----------|--------------|
| Phase 1-1 (QuickDraw) | ~5 天 | ❌ 跳过 |
| Phase 1-2 (书法微调) | ~3 天 | ✅ 只训练这个 (~2.5 天) |
| Phase 2 (RL 优化) | ~1.5 天 | ~1.5 天 |
| **总计** | **~9.5 天** | **~4 天** |

---

## 🎓 快速开始检查清单

- [ ] 获取预训练 Phase 1 模型（或使用骨架提取）
- [ ] 准备 500-2000 张书法图像（256×256）
- [ ] 创建 conda 环境 `CalliRewrite`
- [ ] 验证 TensorFlow GPU 可用
- [ ] 修改 `hyper_parameters.py` 指定预训练模型
- [ ] 启动 Phase 2 训练（后台运行）
- [ ] TensorBoard 监控
- [ ] 等待训练完成 (~2.5 天)
- [ ] 使用 `test.py` 生成 .npy 文件

---

## 🔧 阶段1: seq_extract 完整训练（仅当需要时）

**⚠️ 如果你已经跳过 Phase 1，可以忽略下面的内容！**

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

### 问题 4: 找不到预训练的 Phase 1 模型

**解决方案 A: 联系作者**

```bash
# 在 GitHub 上开 Issue 询问
# 标题: Request for Pre-trained Phase 1 Model
# 说明你想跳过 QuickDraw 训练，只做书法微调
```

**解决方案 B: 使用骨架提取（推荐，无需训练）**

```bash
cd /path/to/CalliRewrite/mujoco_sim

# 从图像直接生成 NPZ
python extract_dense_trajectory.py \
  your_calligraphy_image.png \
  --output result.npz \
  --size 0.12 \
  --depth -0.002
```

**解决方案 C: 使用公开数据集预训练模型**

- 搜索类似项目（如 SketchRNN, Pix2Seq）
- 检查 Papers with Code 是否有相关模型
- 尝试 Hugging Face Model Hub

### 问题 5: conda install cudatoolkit 报错

**错误信息：**
```
CondaHTTPError: HTTP 000 CONNECTION FAILED
```

**解决方案：使用清华镜像源**

```bash
# 添加清华镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes

# 重新安装
conda install cudatoolkit=11.8 cudnn=8.8 -y
```

**或者跳过 cudatoolkit（推荐）：**

```bash
# TensorFlow 2.12+ 已内置 CUDA，不需要单独安装
pip install tensorflow==2.12.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 6: pip 下载 TensorFlow 很慢或中断

**解决方案：使用国内镜像**

```bash
# 清华源（推荐）
pip install tensorflow==2.12.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn

# 增加超时和重试
pip install tensorflow==2.12.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 600 --retries 10
```

**或者手动下载后安装：**

```bash
# 在本地下载 .whl 文件
# 访问 https://pypi.org/project/tensorflow/2.12.0/#files

# 上传到远程服务器
scp tensorflow-2.12.0-*.whl user@server:~/

# 在服务器安装
pip install ~/tensorflow-2.12.0-*.whl
```

### 问题 7: 没有 QuickDraw 数据怎么办

**方案 1: 跳过 Phase 1-1（推荐）**

直接使用预训练模型或骨架提取方法，见上面的"快速开始"章节。

**方案 2: 下载 QuickDraw 数据**

```bash
# 从 Google 官方下载（需要翻墙）
# https://github.com/googlecreativelab/quickdraw-dataset

# 下载特定类别的 .npy 格式
wget https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/airplane.npy

# 需要下载的类别（参考 dataset_utils.py）:
# airplane, bus, car, sailboat, bird, cat, dog, tree, flower, zigzag
```

**数据量大小：** 每个类别约 1-5 GB，总计约 20-50 GB

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
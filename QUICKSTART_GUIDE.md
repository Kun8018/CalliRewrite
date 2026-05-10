# CalliRewrite 2025 快速入门指南

基于大模型的机器人书法系统

## 📋 目录

- [项目概述](#项目概述)
- [环境配置](#环境配置)
- [快速测试](#快速测试)
- [模块详解](#模块详解)
- [训练流程](#训练流程)
- [常见问题](#常见问题)

## 🚀 项目概述

本项目使用大模型（千问/Qwen）结合 ViT + Transformer 提取笔画顺序，通过 Tianshou 强化学习框架精调后控制机器人书写。

### 技术架构

```
书法图像 
    ↓
Qwen-VL-Plus 大模型/API
    ↓ 提取笔画顺序
ViT + Transformer (seq_extract_modern)
    ↓ 输出坐标
Tianshou SAC 强化学习精调
    ↓
MuJoCo 仿真 / 真实机器人
```

## 💻 环境配置

### 1. 基础依赖

使用 RTX 5090/4090 显卡，推荐配置：

```bash
# 创建 conda 环境
conda create -n calli2025 python=3.9
conda activate calli2025

# 安装 PyTorch (根据你的 CUDA 版本)
conda install pytorch==2.1.1 torchvision==0.16.2 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装其他依赖
pip install -r seq_extract_modern/requirements.txt

# 安装 Tianshou
pip install tianshou gymnasium

# 安装大模型相关
pip install transformers accelerate
```

### 2. 千问模型配置

#### 选项 A: 使用 API（推荐，快速测试）

1. 注册阿里云账号并获取 API Key
2. 修改代码中的 API Key
3. 开始使用

#### 选项 B: 使用本地模型

```bash
# 下载模型
git lfs install
git clone https://huggingface.co/Qwen/Qwen-VL-Plus
```

更新 `qwen_stroke_extractor/config.py` 中的路径

### 3. MuJoCo 仿真环境

```bash
cd mujoco_sim
pip install -r requirements.txt
```

## 🏃 快速测试

### 步骤 1: 使用集成测试脚本

```bash
# 1. 准备一张书法图像
# 可以使用项目中的示例图像
mkdir -p data/test_images

# 2. 运行集成测试
python integrated_test.py data/test_images/永.png

# 或使用 qwen 直接测试
cd qwen_stroke_extractor
python quick_test.py
```

### 步骤 2: 测试 qwen_stroke_extractor

```bash
# 使用 API
python examples.py ../imgs/永.png --use-api --api-key YOUR_API_KEY --visualize

# 使用本地模型
python examples.py ../imgs/永.png --model-path /path/to/Qwen-VL-Plus --visualize
```

### 步骤 3: 测试 MuJoCo 仿真

```bash
cd mujoco_sim

# 使用示例数据
python mujoco_simulator.py ../callibrate/examples/example_永.npz --speed 0.05

# 使用自己的输出
python mujoco_simulator.py ../outputs/integrated_test/rl_simulation.npz --speed 0.05 --record video.mp4
```

## 📦 模块详解

### 1. qwen_stroke_extractor

大模型驱动的笔画提取器，支持 API 和本地部署两种模式。

```python
from qwen_stroke_extractor.extractor import create_extractor

# 创建提取器
extractor = create_extractor(
    use_api=True,  # 使用 API 或本地
    api_key="your_key"
)

# 提取笔画
result = extractor.extract("calligraphy.png")

# 可视化
extractor.visualize_result(result, "viz.png", background_image="calligraphy.png")

# 保存结果
extractor.save_result(result, "output.npz", format='npy')
```

**关键文件**:
- `extractor.py`: 核心提取器
- `config.py`: 配置文件
- `examples.py`: 使用示例

### 2. seq_extract_modern

ViT + Transformer 架构的现代化笔画提取模型。

```python
from seq_extract_modern.inference.predictor import Predictor
from seq_extract_modern.configs.model_config import get_default_config

# 加载配置和模型
config = get_default_config()
predictor = Predictor(model_path="path/to/model.ckpt", config=config)

# 预测
result = predictor.predict("calligraphy.png", num_strokes=100)

# 获取笔画参数
stroke_params = result["stroke_params"]
```

**训练命令**:
```bash
cd seq_extract_modern
python scripts/train.py \
    --train_data data/train \
    --val_data data/val \
    --batch_size 32 \
    --max_epochs 100
```

### 3. rl_finetune (Tianshou)

强化学习精调模块，使用 Tianshou 框架的 SAC 算法。

```bash
cd rl_finetune
python try_tianshou.py \
    --train_data data/train \
    --test_data data/test \
    --which_tool brush \
    --tool_property_dir tool_property/brush.json \
    --save_video_dir outputs/videos \
    --save_model_dir outputs/models
```

**关键文件**:
- `try_tianshou.py`: Tianshou 训练脚本
- `Callienv/envs/Callienv.py`: 书法 Gym 环境
- `MLP/model.py`: 策略网络

### 4. mujoco_sim

MuJoCo 物理仿真环境。

```bash
cd mujoco_sim
python mujoco_simulator.py \
    trajectory.npz \
    --speed 0.05 \
    --record video.mp4
```

### 5. callibrate

机器人控制和校准模块。

```bash
cd callibrate

# 生成校准测试
python calibrate.py --mode generate --tool brush

# 执行真实机器人控制
python RoboControl.py trajectory.npz <robot_ip> 0.05
```

## 🎯 训练流程完整路径

### 阶段 1: 准备数据（当前 - 5月15日）

```bash
# 1. 收集/准备书法图像
mkdir -p data/train_images data/val_images

# 2. 使用 qwen_stroke_extractor 批量提取
cd qwen_stroke_extractor
python examples.py --batch ../data/train_images --output ../data/raw_strokes

# 3. 转换为训练格式
cd ..
python -m data_utils.prepare_training_data \
    --source_dir data/raw_strokes \
    --target_dir data/train \
    --seed 42
```

### 阶段 2: 训练 ViT 模型（5月16日 - 6月20日）

```bash
cd seq_extract_modern

# 开始训练
python scripts/train.py \
    --train_data ../data/train \
    --val_data ../data/val \
    --batch_size 32 \
    --max_epochs 100 \
    --save_dir outputs/checkpoints \
    --use_wandb  # 可选，使用 WandB 监控
```

### 阶段 3: 强化学习精调（6月21日 - 7月20日）

```bash
cd rl_finetune

# 准备 RL 数据（使用训练好的 ViT 提取）
python prepare_rl_data.py \
    --vit_model ../seq_extract_modern/outputs/checkpoints/best.ckpt \
    --input_dir ../data/train_images \
    --output_dir data/train_data

# 训练 RL 策略
python try_tianshou.py \
    --train_data data/train_data \
    --test_data data/test_data \
    --which_tool brush \
    --max_epoch 150 \
    --actor_lr 3e-5
```

### 阶段 4: 仿真验证（7月21日 - 8月31日）

```bash
# 批量测试
cd mujoco_sim
python batch_test.py \
    --rl_model ../rl_finetune/outputs/models/best.pth \
    --input_dir data/test_characters \
    --output_dir outputs/batch_results

# 人工评估
python evaluate.py --results_dir outputs/batch_results
```

### 阶段 5: 论文撰写与投稿（9月1日 - 9月15日）

```bash
# 收集实验结果
python analysis.py --results_dir outputs/all_results

# 生成图表
python plot_results.py --data analysis/results.csv --output paper/figures

# 论文 LaTeX
cd paper
pdflatex main.tex
```

## 📁 项目结构

```
CalliRewrite/
├── qwen_stroke_extractor/       # 大模型笔画提取
├── seq_extract_modern/          # ViT + Transformer 模型
├── seq_extract/                 # 旧版 LSTM 提取（兼容）
├── rl_finetune/                 # Tianshou 强化学习
├── mujoco_sim/                  # 物理仿真
├── callibrate/                  # 机器人控制
├── integrated_test.py           # 集成测试
├── README.md                    # 项目说明
└── QUICKSTART_GUIDE.md          # 本指南
```

## 🔧 常见问题

### Q: RTX 5090 支持吗？
A: 完全支持！5090 具有更强的计算能力，比 4090 更适合训练 ViT 和运行大模型。

### Q: 大模型和 ViT 如何选择？
A: 
- 快速测试/零样本：优先用 qwen_stroke_extractor
- 可控性/训练优化：用 seq_extract_modern
- 最佳效果：两者结合（qwen 提供粗顺序 -> ViT 优化 -> RL 精调）

### Q: 数据不足怎么办？
A:
1. 使用数据增强（旋转、缩放、变形）
2. 使用 qwen 生成合成数据
3. 使用 QuickDraw 预训练 + 书法微调

### Q: 如何结合千问和 ViT？
A: 有两个方案：
```python
# 方案1: qwen 提供标注 -> 训练 ViT
extract_result = extractor.extract(image)
save_for_training(extract_result)

# 方案2: qwen 作为模块 -> 与 ViT 融合（推荐！）
class CombinedModel:
    def __init__(self, qwen_extractor, vit_model):
        self.qwen = qwen_extractor
        self.vit = vit_model
    
    def forward(self, image):
        stroke_order = self.qwen.extract(image)  # 提取顺序
        stroke_coords = self.vit.predict(image, stroke_order)  # 使用顺序优化
        return stroke_coords
```

### Q: MuJoCo 仿真有问题？
A: 检查：
- MuJoCo 安装正确
- Python 版本兼容
- 运行系统测试：`python mujoco_sim/quick_test.py`

## 📞 获取帮助

1. 查看各模块 README
2. 查看项目文档：`ARCHITECTURE.md`
3. 运行示例代码
4. 查看原项目 ICRA 2024 论文

---

**祝使用愉快！** 🎉

如需更多帮助，欢迎交流讨论

# 使用 seq_extract_modern 运行仿真的快速指南

## 📋 概述

这个指南将帮助你使用新的 `seq_extract_modern` 模块与仿真系统配合使用。

---

## 🎯 工作流程

```
书法图像 → seq_extract_modern → .npy → 转换 → .npz → mujoco_sim → 仿真演示
```

---

## 📦 第一步：安装依赖

### 1. 安装 seq_extract_modern 的依赖

```bash
# 创建虚拟环境
python3 -m venv calli_env
source calli_env/bin/activate

# 安装核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow matplotlib pytorch-lightning torchmetrics opencv-python
```

### 2. 安装 MuJoCo 仿真依赖

MuJoCo 的安装比较复杂，推荐以下方法：

#### 方法 A：使用 Conda（推荐）
```bash
conda install -c conda-forge mujoco
```

#### 方法 B：使用 Homebrew（Mac）
```bash
brew install mujoco
```

#### 方法 C：手动安装

**重要**：MuJoCo 3.0+ 需要先下载并设置环境变量。
请参考 [MuJoCo 官方文档](https://mujoco.readthedocs.io/)

---

## 🚀 第二步：快速测试（使用现有示例数据

### 1. 检查是否有示例 NPZ 文件：

```bash
ls -la outputs/
# 你应该看到 simple_square.npz
```

### 2. 如果没有，运行我们提供的简单示例：

```bash
source calli_env/bin/activate
python3 make_simple_npz.py
```

### 3. 运行仿真：

```bash
cd mujoco_sim
python mujoco_simulator.py ../outputs/simple_square.npz --speed 0.05
```

---

## 🎨 第三步：使用 seq_extract_modern 提取笔画

### 方法 A：使用原始 seq_extract 的输出（推荐先用这个）

由于新的 seq_extract_modern 模型还没有预训练权重，所以我们先用原始的 seq_extract 模块：

```bash
cd seq_extract
conda activate CalliRewrite

# 先使用原始模块提取笔画
python test.py --input sample_inputs/clean_line_drawings/mouse.png \
    --model pretrain_clean_line_drawings \
    --sample 1
```

这将在 `seq_extract/outputs/` 中生成 `.npy` 文件。

### 方法 B：使用转换脚本

```bash
# 使用原始输出目录中找到生成的 .npy 文件
ls seq_extract/outputs/

# 然后使用桥接脚本转换为仿真可用的 .npz
source calli_env/bin/activate
python3 seq_extract_modern_to_simulation.py \
    --input seq_extract/outputs/mouse_strokes.npy \
    --output outputs/mouse_simulation.npz \
    --use_old_format
```

### 方法 C：直接使用原始的 calibrate.py

```bash
cd callibrate

# 如果原始项目已经有转换功能
python calibrate.py --mode convert --tool brush \
    --input ../seq_extract/outputs/mouse_strokes.npy \
    --output ../outputs/mouse.npz \
    --alpha 0.04 --beta 0.5
```

---

## 🤖 第四步：运行仿真

```bash
cd mujoco_sim

# 基础仿真
python mujoco_simulator.py ../outputs/mouse_simulation.npz --speed 0.05

# 录制视频
python mujoco_simulator.py ../outputs/mouse_simulation.npz \
    --record outputs/video.mp4 --speed 0.05
```

---

## 📝 第五步：完整的流程（一旦 seq_extract_modern 有预训练权重后

### 1. 训练新模型

```bash
cd seq_extract_modern
source calli_env/bin/activate

# 训练（需要数据）
python scripts/train.py --help
```

### 2. 推理和直接转换为仿真

```bash
source calli_env/bin/activate
python3 seq_extract_modern_to_simulation.py \
    --input seq_extract/sample_inputs/clean_line_drawings/mouse.png \
    --output outputs/mouse_from_new.npz
```

---

## 🔍 排查问题

### 问题 1：MuJoCo 安装失败

**解决方案**：
- 尝试使用 conda-forge 安装：`conda install -c conda-forge mujoco`
- 或者先用原始的 seq_extract 模块，它不需要 MuJoCo

### 问题 2：找不到 .npy 文件

**解决方案**：
- 检查 `seq_extract/outputs/` 目录
- 使用正确的文件路径

### 问题 3：找不到 `callibrate` 目录不存在

**解决方案**：
- 使用 `ls /Users/kun/CalliRewrite/` 查看实际目录名
- 原始项目中可能叫 `callibrate`（不是 `calibrate`

---

## 📊 文件位置汇总

| 文件/目录 | 说明 |
|----------|------|
| `seq_extract_modern/` | 新的现代化代码 |
| `seq_extract/` | 原始代码（有预训练权重） |
| `callibrate/` | 校准和转换模块 |
| `mujoco_sim/` | 仿真模块 |
| `outputs/` | 输出目录（我们创建的） |

---

## 💡 推荐：先用这个！

**建议：先用原始的 seq_extract 模块工作流程（有预训练权重）：

```bash
# 1. 激活原始环境
conda activate CalliRewrite

# 2. 提取笔画
cd seq_extract
python test.py --input sample_inputs/clean_line_drawings/elephant.png --model pretrain_clean_line_drawings

# 3. 找到输出文件
ls outputs/

# 4. 用原始 calibrate 转换
cd ../callibrate
python calibrate.py --mode convert --tool brush --input ../seq_extract/outputs/elephant_strokes.npy --output ../outputs/elephant.npz --alpha 0.04 --beta 0.5

# 5. 运行仿真
cd ../mujoco_sim
python mujoco_simulator.py ../outputs/elephant.npz --speed 0.05
```

这样最简单！
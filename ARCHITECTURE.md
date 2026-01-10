# CalliRewrite 代码整体框架说明

## 📚 项目概览

**CalliRewrite** 是一个端到端的机器人书法系统，能够从书法图像中恢复笔画行为，并在机器人上重现书法作品。

**核心特点**:
- 🎨 无监督学习（无需人工标注笔画顺序）
- 🤖 支持低成本机器人臂（Dobot, Franka等）
- 🖌️ 支持多种书写工具（毛笔、钢笔、马克笔）
- 📄 ICRA 2024 论文项目

---

## 🏗️ 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                      CalliRewrite 系统架构                         │
└──────────────────────────────────────────────────────────────────┘

输入: 书法图像 (PNG/JPG)
   ↓
┌─────────────────────────────────┐
│  阶段 1: 粗笔画序列提取          │  seq_extract/
│  (Coarse Sequence Extraction)   │
│  • LSTM 编码-解码器              │
│  • 两阶段训练                    │
│  • QuickDraw → 书法数据          │
└─────────────────────────────────┘
   ↓ output: .npy (虚拟笔画序列)
┌─────────────────────────────────┐
│  阶段 2: 工具感知精调            │  rl_finetune/
│  (Tool-Aware Fine-tuning)       │
│  • 强化学习 (SAC)                │
│  • 虚拟书法环境                  │
│  • 笔刷物理建模                  │
└─────────────────────────────────┘
   ↓ output: .npy (优化后笔画序列)
┌─────────────────────────────────┐
│  阶段 3: 机器人校准              │  callibrate/
│  (Robot Calibration)            │
│  • r-z 关系拟合                  │
│  • Sim2Real 转换                │
│  • 书法风格处理                  │
└─────────────────────────────────┘
   ↓ output: .npz (机器人控制点)
┌─────────────────────────────────┐
│  阶段 4: 机器人执行              │  callibrate/RoboControl.py
│  (Robot Execution)              │
│  • Franka/Dobot 控制             │
│  • 轨迹执行                      │
└─────────────────────────────────┘
   ↓
输出: 实物书法作品 ✍️
```

---

## 📂 目录结构详解

```
CalliRewrite/
│
├── seq_extract/                    # 阶段1: 粗笔画序列提取
│   ├── train_phase_1.py            # 训练阶段1 (QuickDraw)
│   ├── train_phase_2.py            # 训练阶段2 (书法数据)
│   ├── test.py                     # 测试/推理脚本
│   ├── hyper_parameters.py         # 超参数配置
│   ├── model_common_train.py       # LSTM 模型定义
│   ├── dataset_utils.py            # 数据加载工具
│   │
│   ├── rasterization_utils/        # 渲染工具
│   │   ├── NeuralRenderer.py       # 神经渲染器
│   │   └── RealRenderer.py         # 真实笔画渲染
│   │
│   ├── vgg_utils/                  # 感知损失
│   │   └── VGG16.py                # VGG16 特征提取
│   │
│   ├── tools/                      # 可视化工具
│   │   ├── visualize_drawing.py    # 笔画可视化
│   │   ├── gif_making.py           # GIF 生成
│   │   └── svg_conversion.py       # SVG 转换
│   │
│   └── sample_inputs/              # 示例输入图像
│       ├── clean_line_drawings/    # 清晰线稿
│       ├── rough_sketches/         # 粗糙草图
│       └── faces/                  # 人脸素描
│
├── rl_finetune/                    # 阶段2: 强化学习精调
│   ├── try_tianshou.py             # 主训练脚本 (Tianshou库)
│   ├── utils.py                    # 工具函数
│   │
│   ├── Callienv/                   # 书法环境定义
│   │   └── envs/
│   │       ├── Callienv.py         # Gym 环境主类
│   │       ├── tools.py            # 笔刷工具模型
│   │       └── skel_utils.py       # 骨架处理工具
│   │
│   ├── MLP/                        # 策略网络
│   │   ├── mlp.py                  # MLP 定义
│   │   └── model.py                # 完整模型
│   │
│   ├── tool_property/              # 工具物理属性
│   │   ├── brush_property.json     # 毛笔参数
│   │   ├── fude_property.json      # 日式笔参数
│   │   └── marker_property.json    # 马克笔参数
│   │
│   ├── data/                       # 训练数据
│   │   ├── train_data/             # 训练集 (.npy)
│   │   └── test_data/              # 测试集 (.npy)
│   │
│   └── scripts/                    # 训练脚本
│       ├── train_brush.sh          # 毛笔训练
│       ├── train_fude.sh           # 日式笔训练
│       └── train_marker.sh         # 马克笔训练
│
├── callibrate/                     # 阶段3&4: 校准与机器人控制
│   ├── calibrate.py                # 主校准脚本 ⭐
│   ├── RoboControl.py              # Franka 机器人控制 ⭐ (新增)
│   ├── DobotDllType.py             # Dobot API 封装
│   ├── load_data_try.py            # 数据加载
│   ├── generate_example_npz.py     # 生成示例文件 (新增)
│   │
│   ├── callibrate.ipynb            # Jupyter 校准教程
│   ├── franka_config.yaml          # Franka 配置 (新增)
│   ├── FRANKA_SETUP.md             # Franka 设置指南 (新增)
│   ├── NPZ_FILE_EXPLAINED.md       # NPZ 文件说明 (新增)
│   │
│   ├── examples/                   # 示例文件 (新增)
│   │   ├── simple_line.npz         # 简单直线
│   │   ├── test_calibration.npz    # 校准测试
│   │   └── example_永.npz          # "永"字示例
│   │
│   └── *.dll, *.pdf                # Dobot 相关文件
│
├── demo/                           # 演示材料
│   └── teaser.png                  # 项目展示图
│
├── README.md                       # 项目说明
└── LICENSE                         # MIT 许可证
```

---

## 🔑 核心模块详解

### 1️⃣ **seq_extract** - 粗笔画序列提取

**目标**: 从书法图像中提取笔画的粗略顺序和位置

**核心技术**:
- **编码器-解码器 LSTM**: 学习笔画序列表示
- **两阶段训练**:
  - 阶段1: 在 QuickDraw 数据集上预训练
  - 阶段2: 在书法数据上微调
- **神经渲染器**: 将笔画序列渲染为图像
- **感知损失**: 使用 VGG16 提取特征计算相似度

**关键文件**:
```python
# 训练
train_phase_1.py  # QuickDraw 预训练
train_phase_2.py  # 书法微调

# 模型
model_common_train.py  # LSTM 编码器-解码器
├── Encoder: 图像 → 隐藏状态
└── Decoder: 隐藏状态 → 笔画序列

# 推理
test.py  # 从图像生成 .npy 笔画文件
```

**数据格式**:
```python
# 输入: 书法图像 (256×256)
# 输出: .npy 笔画序列
data.shape = (N, 7)
# [p_t, x, y, r, col1, col2, col3]
# p_t: 0=继续, 1=新笔画
# x, y: 虚拟坐标 (0-255)
# r: 笔画半径 (0-255)
```

---

### 2️⃣ **rl_finetune** - 工具感知精调

**目标**: 使用强化学习优化笔画序列，适应特定书写工具的物理特性

**核心技术**:
- **强化学习算法**: SAC (Soft Actor-Critic)
- **自定义 Gym 环境**: `CalliEnv`
- **笔刷物理建模**: 模拟不同工具的动力学特性
- **奖励函数**: 图像相似度 + 笔画平滑度

**关键文件**:
```python
# 主训练脚本
try_tianshou.py
├── 加载粗笔画序列 (.npy)
├── 初始化 CalliEnv 环境
├── 训练 SAC 策略
└── 输出优化后序列 (.npy)

# 环境定义
Callienv/envs/Callienv.py
├── __init__(): 初始化环境
├── reset(): 加载新图像
├── step(action): 执行动作，计算奖励
└── render(): 渲染当前状态

# 工具建模
Callienv/envs/tools.py
├── BrushTool        # 毛笔
├── FudeTool         # 日式笔
└── MarkerTool       # 马克笔
```

**状态空间** (8维):
```python
state = [
    period,      # 当前笔画进度 [0,1]
    r,           # 笔画半径 [0,1]
    l,           # 笔刷长度 [0,1]
    theta,       # 旋转角度 [0,1]
    curvature,   # 曲率 [0,1]
    r_prime,     # 移动距离 [0,1]
    vec_x,       # X方向向量 [-1,1]
    vec_y        # Y方向向量 [-1,1]
]
```

**动作空间** (2维):
```python
action = [
    r_prime,     # 移动距离 [-1,1]
    theta_prime  # 移动角度 [-1,1] (×π)
]
```

**奖励函数**:
```python
reward = -2 * abs(r_prime) * curvature / K
         + alpha * cos_similarity(theta, new_theta)
         + beta
# K: 归一化常数
# alpha, beta: 权重参数
```

---

### 3️⃣ **callibrate** - 校准与机器人控制

**目标**: 将虚拟笔画参数转换为真实机器人控制指令

#### 子模块 A: 校准 (calibrate.py)

**功能**:
1. 生成校准测试轨迹
2. 测量笔画宽度
3. 拟合 r-z 映射函数
4. 转换 RL 输出为机器人坐标

**核心函数**:
```python
# 1. 生成校准数据
generate_calibration_data(tool_type)
├── 设置 max_z, min_z, its
├── 生成 z 高度序列
└── 保存为 test.npz

# 2. 拟合校准函数
fit_calibration_function(widths, zs, tool_type)
├── 使用分段线性函数拟合
├── piecewise_linear3/4()
└── 返回 r → z 映射函数

# 3. RL 到机器人转换 ⭐⭐⭐
convert_rl_to_npz(npy_path, output_path, calibration_func,
                  alpha, beta, style_type)
├── 加载 .npy (虚拟坐标)
├── 缩放到真实尺寸 (α参数)
├── 半径 → 高度 (校准函数)
├── 书法风格处理 (起笔/收笔)
└── 保存为 .npz (真实坐标)
```

**关键参数**:
| 参数 | 含义 | 示例 |
|------|------|------|
| `alpha` | 字符大小(米) | 0.04 = 4cm |
| `beta` | 笔画宽度系数 | 0.5 = 正常 |
| `style_type` | 书法风格 | 0=隶书, 1=楷书 |

**校准原理**:
```python
# 分段线性拟合 r-z 关系
def func_brush(r):
    if r <= 0.00077:
        z = k0 * (r - x0) + y0
    elif r <= 0.00178:
        z = k1 * (r - x1) + y1
    elif r <= 0.00246:
        z = k2 * (r - x2) + y2
    else:
        z = k3 * (r - x3) + y3
    return z

# 实际测量
实验: 机器人在不同 z 高度画线
测量: 实际笔画宽度 w
拟合: r = w/2 → z 的映射
```

#### 子模块 B: 机器人控制 (RoboControl.py)

**功能**: 在 Franka 机器人上执行书法轨迹

**核心类**:
```python
class FrankaCalligraphyController:
    def __init__(robot_ip, workspace_center, ...):
        # 初始化机器人连接

    def connect():
        # 连接 franky/frankx

    def move_cartesian(x, y, z, speed):
        # 笛卡尔空间运动

    def execute_trajectory(x, y, z, speed, wait_time):
        # 执行完整轨迹

    def move_to_home():
        # 返回安全位置

# 主接口函数
def Control(npz_path, robot_ip, speed):
    """CalliRewrite 兼容接口"""
    # 加载 .npz
    # 执行轨迹
    # 返回结果
```

**支持的机器人**:
- ✅ Franka Emika (Panda, FR3) - 通过 franky/frankx
- ✅ Dobot Magician - 通过 DobotDllType.py
- 🔄 其他机器人 - 可扩展

---

## 🔄 完整数据流

### 数据类型演变

```
┌────────────────┐
│  书法图像       │  永.png (256×256)
│  PNG/JPG       │
└────────┬───────┘
         │ seq_extract/test.py
         ↓
┌────────────────┐
│  粗笔画序列     │  永_coarse.npy
│  (N, 7) array  │  [p_t, x, y, r, ...]
│  虚拟坐标       │  x, y, r ∈ [0, 255]
└────────┬───────┘
         │ rl_finetune/try_tianshou.py
         ↓
┌────────────────┐
│  精细笔画序列   │  永.npy
│  (M, 7) array  │  优化后的笔画参数
│  RL 优化后      │  更平滑、更符合工具特性
└────────┬───────┘
         │ callibrate/calibrate.py
         │ convert_rl_to_npz()
         ↓
┌────────────────┐
│  机器人控制点   │  永.npz
│  {x, y, z}     │  pos_3d_x, pos_3d_y, pos_3d_z
│  真实米制坐标   │  x, y, z ∈ 真实空间 (米)
└────────┬───────┘
         │ callibrate/RoboControl.py
         │ Control()
         ↓
┌────────────────┐
│  Franka 机器人  │
│  逐点运动       │  move_cartesian(x[i], y[i], z[i])
└────────┬───────┘
         ↓
┌────────────────┐
│  实物书法作品   │  ✍️
└────────────────┘
```

---

## 🎯 关键算法

### 1. LSTM 编码-解码器 (seq_extract)

```python
# 编码器
class Encoder(tf.keras.Model):
    def __init__(self):
        self.conv_layers = [Conv2D(...), ...]
        self.lstm = LSTM(hidden_size)

    def forward(self, image):
        # image: (B, 256, 256, 1)
        features = self.conv_layers(image)
        hidden = self.lstm(features)
        return hidden  # (B, hidden_size)

# 解码器
class Decoder(tf.keras.Model):
    def __init__(self):
        self.lstm_cell = LSTMCell(hidden_size)
        self.fc = Dense(7)  # [p_t, x, y, r, ...]

    def forward(self, hidden, max_len):
        outputs = []
        for t in range(max_len):
            hidden = self.lstm_cell(hidden)
            output = self.fc(hidden)
            outputs.append(output)
        return outputs  # (max_len, 7)
```

### 2. SAC 强化学习 (rl_finetune)

```python
# 策略网络
class ActorNetwork(nn.Module):
    def forward(self, state):
        # state: [period, r, l, theta, curvature, r_prime, vec_x, vec_y]
        mu, sigma = self.mlp(state)
        action = mu + sigma * noise  # [r_prime, theta_prime]
        return action

# 训练循环
for epoch in range(num_epochs):
    state = env.reset()
    for step in range(max_steps):
        action = actor(state)
        next_state, reward, done = env.step(action)
        # 更新 Actor-Critic
        critic_loss = ...
        actor_loss = ...
```

### 3. 分段线性校准 (callibrate)

```python
def piecewise_linear4(x, x0, x1, x2, y0, y1, y2, k0, k1):
    """4段分段线性函数"""
    if x <= x0:
        return k0 * (x - x0) + y0
    elif x0 < x <= x1:
        return (x - x0) * (y1 - y0) / (x1 - x0) + y0
    elif x1 < x <= x2:
        return (x - x1) * (y2 - y1) / (x2 - x1) + y1
    else:
        return k1 * (x - x2) + y2

# 使用 scipy.optimize.curve_fit 拟合参数
params, _ = curve_fit(piecewise_linear4, r_data, z_data, bounds=...)
```

---

## 🛠️ 使用工作流

### 快速开始

```bash
# 1. 安装依赖
cd seq_extract && conda env create -f environment.yml
cd ../rl_finetune && conda env create -f environment.yml

# 2. 粗笔画提取
conda activate calli_ext
cd seq_extract
python test.py --input imgs/永.png --output ../rl_finetune/data/train_data/0.npy

# 3. RL 精调
conda activate calli_rl
cd ../rl_finetune
bash scripts/train_brush.sh

# 4. 校准转换
cd ../callibrate
python calibrate.py --mode generate --tool brush  # 生成校准数据
python calibrate.py --mode convert --tool brush \
  --input ../rl_finetune/results/永.npy \
  --output ./永.npz --alpha 0.04

# 5. 机器人执行
python RoboControl.py ./永.npz 172.16.0.2 0.05
```

---

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 图像输入尺寸 | 256×256 |
| 粗提取速度 | ~2秒/字符 |
| RL 训练时间 | ~30分钟/字符 |
| 校准精度 | ±0.5mm |
| 机器人执行时间 | ~2-5分钟/字符 |

---

## 🔬 技术创新点

1. **无监督笔画恢复**: 无需标注笔画顺序
2. **工具感知 RL**: 适应不同书写工具的物理特性
3. **Sim2Real 校准**: 精确的虚拟-真实转换
4. **模块化设计**: 各阶段解耦，易于扩展

---

## 📚 相关论文

```bibtex
@inproceedings{luo2024callirewrite,
  title={CalliRewrite: Recovering Handwriting Behaviors from Calligraphy Images without Supervision},
  author={Luo, Yuxuan and Wu, Zekun and Lian, Zhouhui},
  booktitle={ICRA 2024},
  year={2024}
}
```

---

**总结**: CalliRewrite 通过 **无监督学习 + 强化学习 + 机器人校准** 三阶段流水线，实现了从书法图像到机器人实物重现的完整系统！

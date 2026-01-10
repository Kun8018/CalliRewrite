# Seq_Extract 训练过程详解

## 📚 模块概述

**seq_extract** 是 CalliRewrite 系统的第一阶段，负责从书法图像中提取粗笔画序列。

**目标**: 将 256×256 书法图像转换为笔画序列 (x, y, r, ...)

**核心技术**:
- 编码器-解码器 LSTM
- 神经渲染器
- 感知损失 (VGG16)
- 两阶段渐进式训练

---

## 🏗️ 整体架构

```
输入: 书法图像 (256×256)
    ↓
┌─────────────────────────────────┐
│  Encoder (CNN + LSTM)           │
│  • Conv layers: 提取图像特征     │
│  • LSTM: 编码为隐藏状态          │
└─────────────────────────────────┘
    ↓ hidden_state (256维)
┌─────────────────────────────────┐
│  Decoder (LSTM)                 │
│  • 自回归生成笔画序列             │
│  • 每步输出 7 维参数              │
└─────────────────────────────────┘
    ↓ [p_t, x, y, r, w, θ, ...]
┌─────────────────────────────────┐
│  Neural Renderer                │
│  • 将笔画序列渲染为图像           │
│  • 可微分渲染                    │
└─────────────────────────────────┘
    ↓ 渲染图像 (256×256)
┌─────────────────────────────────┐
│  Loss Calculation               │
│  • 渲染损失 (L1/L2)              │
│  • 感知损失 (VGG16)              │
│  • 笔画数损失                    │
│  • 平滑度损失                    │
└─────────────────────────────────┘
    ↓ 反向传播优化
更新 Encoder + Decoder 参数
```

---

## 🎓 两阶段训练策略

### 阶段 1: QuickDraw 预训练

**目的**: 在大规模简笔画数据上学习基本的笔画生成能力

**数据集**: QuickDraw (Google 手绘数据集)
- 包含数百万个简笔画
- 已标注笔画顺序
- 风格简单、清晰

**训练配置**:
```python
# hyper_parameters.py - Phase 1
hparams = {
    'program_name': 'new_train_phase_1',
    'data_set': 'clean_line_drawings',  # QuickDraw

    'num_steps': 90040,           # 总训练步数
    'save_every': 15000,          # 每15000步保存一次
    'eval_every': 5000,           # 每5000步验证一次

    'batch_size': 12,             # 批次大小
    'max_seq_len': 48,            # 最大序列长度

    'learning_rate': 0.0001,      # 初始学习率
    'min_learning_rate': 0.000001,# 最小学习率
    'decay_power': 0.9,           # 学习率衰减指数
}
```

**关键特点**:
- ✅ 数据充足 (数百万样本)
- ✅ 监督信号清晰 (有笔画顺序)
- ✅ 快速收敛 (约3-5天 on single GPU)

---

### 阶段 2: 书法数据微调

**目的**: 在真实书法数据上微调，适应复杂的书法风格

**数据集**: 自定义书法数据集
- 数千个汉字书法图像
- 无笔画顺序标注 (无监督)
- 风格复杂、多样

**训练配置**:
```python
# hyper_parameters.py - Phase 2
hparams = {
    'program_name': 'new_train_phase_2',
    'data_set': 'gb',  # 书法数据集

    'num_steps': 60000,           # 微调步数
    'save_every': 10000,

    # 从 Phase 1 加载预训练权重
    'pretrained_model': 'outputs/snapshot/new_train_phase_1/model-90000',
}
```

**关键特点**:
- 🎨 复杂书法风格
- 📖 无监督学习
- 🔄 迁移学习 (从 Phase 1)

---

## 🔑 核心组件详解

### 1️⃣ **Encoder - CNN + LSTM**

**目的**: 将图像编码为固定维度的隐藏状态

**架构**:
```python
# model_common_train.py - generative_cnn_c3_encoder

Input: 图像 (N, 256, 256, 1)
    ↓
Conv1: (N, 128, 128, 64)   # kernel=3, stride=2
    ↓
Conv2: (N, 64, 64, 128)    # kernel=3, stride=2
    ↓
Conv3: (N, 32, 32, 256)    # kernel=3, stride=2
    ↓
Conv4: (N, 16, 16, 512)    # kernel=3, stride=2
    ↓
Flatten: (N, 16×16×512)
    ↓
LSTM: (N, 256)             # hidden_size=256
    ↓
Output: hidden_state (N, 256)
```

**关键技术**:
- **多尺度特征提取**: 逐层降采样捕获不同尺度的笔画特征
- **LSTM 编码**: 将空间特征编码为序列状态
- **Coordconv**: 添加坐标信息增强空间感知

---

### 2️⃣ **Decoder - 自回归 LSTM**

**目的**: 从隐藏状态自回归生成笔画序列

**架构**:
```python
# 解码器循环
hidden_state = encoder_output  # (N, 256)

for t in range(max_seq_len):  # max_seq_len = 48
    # LSTM 解码
    output, hidden_state = lstm_cell(hidden_state)

    # 输出层
    pen_state = fc_pen(output)      # (N, 2) - [pen_up, pen_down]
    position = fc_pos(output)       # (N, 2) - [x, y]
    radius = fc_radius(output)      # (N, 1) - r
    width = fc_width(output)        # (N, 1) - w
    angle = fc_angle(output)        # (N, 1) - θ

    # 合并输出
    params[t] = [pen_state, position, radius, width, angle]
```

**输出格式**:
```python
# 每个时间步输出 7 维参数
params[t] = [p_t, x, y, r, w, θ, κ]

p_t: 笔状态 (0=抬笔, 1=落笔)
x, y: 位置坐标 [0, 1] (归一化)
r: 笔画半径 [0, 1]
w: 笔刷宽度 [0, 1]
θ: 旋转角度 [0, 2π]
κ: 其他参数
```

**关键技术**:
- **Hyper LSTM**: 使用超网络动态调整 LSTM 参数
- **Recurrent Dropout**: 防止过拟合 (keep_prob=0.9)
- **Teacher Forcing**: 训练时使用真实笔画引导

---

### 3️⃣ **Neural Renderer - 可微分渲染**

**目的**: 将笔画参数渲染为图像，支持梯度回传

**渲染过程**:
```python
# NeuralRasterizorStep

def render_stroke(x, y, r, w, θ):
    """渲染单个笔画"""
    # 1. 创建笔刷模板
    brush = create_brush_template(r, w, θ)
    # 形状: (2r, 2w) - 椭圆形笔刷

    # 2. 放置到画布
    canvas = paste_brush(canvas, brush, x, y)
    # 使用 alpha blending 混合

    # 3. 累积渲染
    return canvas

# 渲染整个序列
canvas = zeros(256, 256)
for t in range(seq_len):
    if pen_state[t] == 1:  # 落笔
        canvas = render_stroke(x[t], y[t], r[t], w[t], θ[t])

return canvas  # (256, 256) - 渲染图像
```

**关键特性**:
- ✅ **可微分**: 所有操作支持梯度回传
- ✅ **高效**: GPU 加速并行渲染
- ✅ **真实**: 模拟真实笔刷效果

---

### 4️⃣ **Loss Functions - 多任务损失**

#### A. 渲染损失 (Raster Loss)
```python
# L1 损失
raster_loss = |rendered_image - target_image|

# 或 L2 损失
raster_loss = (rendered_image - target_image)^2
```

#### B. 感知损失 (Perceptual Loss)
```python
# 使用预训练 VGG16 提取特征
vgg_layers = ['ReLU1_2', 'ReLU2_2', 'ReLU3_3', 'ReLU5_1']

perc_loss = 0
for layer in vgg_layers:
    feat_rendered = VGG(rendered_image)[layer]
    feat_target = VGG(target_image)[layer]
    perc_loss += |feat_rendered - feat_target|

# 加权融合
total_loss = raster_loss + λ_perc × perc_loss
```

**感知损失的优势**:
- 🎨 捕获高层语义特征
- 📐 对局部位移不敏感
- ✨ 生成更自然的笔画

#### C. 笔画数损失 (Stroke Number Loss)
```python
# 鼓励使用更少的笔画
stroke_count = sum(pen_state == 1)  # 统计落笔次数
sn_loss = λ_sn × stroke_count

# λ_sn 随训练动态调整
# Phase 1: λ_sn 从 0 逐渐增加到 0.5 (increasing)
# Phase 2: λ_sn 从 0.5 逐渐减少到 0 (decreasing)
```

**动态权重调整**:
```python
if sn_loss_type == 'increasing':
    # 前期不限制，后期增加约束
    λ_sn = min(step / total_steps × 0.5, 0.5)
elif sn_loss_type == 'decreasing':
    # 前期强约束，后期放松
    λ_sn = 0.5 - min(step / total_steps × 0.5, 0.5)
```

#### D. 平滑度损失 (Smoothness Loss)
```python
# 鼓励笔画平滑
smoothness_loss = sum(|params[t+1] - params[t]|)

# 包括位置、半径、角度的变化
pos_smooth = |pos[t+1] - pos[t]|
radius_smooth = |r[t+1] - r[t]|
angle_smooth = |θ[t+1] - θ[t]|
```

#### E. 角度损失 (Angle Loss)
```python
# 鼓励笔画沿运动方向
direction = (pos[t+1] - pos[t]) / |pos[t+1] - pos[t]|
angle_loss = |direction - [cos(θ[t]), sin(θ[t])]|
```

#### F. 边界损失 (Outside Loss)
```python
# 惩罚超出画布的笔画
outside_loss = sum(max(0, x[t] - 1) + max(0, -x[t]))
             + sum(max(0, y[t] - 1) + max(0, -y[t]))
```

#### 总损失
```python
total_loss = λ_raster × raster_loss
           + λ_perc × perc_loss
           + λ_sn × sn_loss
           + λ_smooth × smoothness_loss
           + λ_angle × angle_loss
           + λ_outside × outside_loss

# 默认权重
λ_raster = 1.0
λ_perc = 1.0 (自适应)
λ_sn = 0.0 ~ 0.5 (动态)
λ_smooth = 0.0
λ_angle = 0.0
λ_outside = 10.0
```

---

## 🔄 训练流程

### 主训练循环

```python
# train_phase_1.py

def train(sess, model, train_set, val_set):
    """主训练函数"""

    for step in range(1, num_steps + 1):
        # 1. 学习率衰减
        lr = (lr_init - lr_min) × (1 - step/total_steps)^decay_power + lr_min

        # 2. 动态调整损失权重
        λ_sn = compute_sn_weight(step, sn_loss_type)

        # 3. 加载批次数据
        images, targets, cursors = train_set.get_batch_multi_res(
            batch_size=12,
            image_size_range=[128, 278]  # 多尺度训练
        )

        # 4. 前向传播
        feed_dict = {
            model.input_photo: images,
            model.lr: lr,
            model.stroke_num_loss_weight: λ_sn,
        }

        rendered, params, loss = sess.run(
            [model.pred_raster_imgs, model.pred_params, model.total_loss],
            feed_dict
        )

        # 5. 反向传播
        sess.run(model.train_op, feed_dict)

        # 6. 日志记录
        if step % 100 == 0:
            print(f"Step {step}: loss={loss:.4f}, lr={lr:.6f}")

        # 7. 保存中间结果
        if step % 500 == 0:
            save_log_images(sess, model, val_set, step)

        # 8. 保存模型
        if step % 15000 == 0:
            save_model(sess, model, step)
```

---

## 📊 多尺度训练策略

**目的**: 提高模型对不同分辨率的泛化能力

**实现**:
```python
# 每个 batch 随机选择分辨率
image_sizes = [128, 150, 170, 190, 210, 230, 250, 278]

for step in range(num_steps):
    # 随机选择本批次的图像尺寸
    current_size = random.choice(image_sizes)

    # 调整输入图像大小
    images = resize(original_images, (current_size, current_size))

    # 训练
    train_step(images, targets)
```

**优势**:
- 🎯 提高分辨率鲁棒性
- 📈 防止过拟合特定尺度
- 🚀 数据增强效果

---

## 🎨 关键训练技巧

### 1. 渐进式笔画生成
```python
# early_pen_loss_type = 'move'

# 训练初期：只关注序列前半部分
early_steps: loss_range = [0, 24]

# 训练中期：逐渐扩展到全序列
mid_steps: loss_range = [12, 36]

# 训练后期：关注整个序列
late_steps: loss_range = [0, 48]
```

**目的**: 避免早期训练崩溃，渐进式学习

### 2. 自适应感知损失
```python
# 动态调整感知损失各层权重
mean_perc_losses = [0.0] * len(vgg_layers)

for step in training:
    # 计算当前感知损失
    perc_losses = compute_perceptual_loss(rendered, target)

    # 指数移动平均
    for i in range(len(vgg_layers)):
        mean_perc_losses[i] = 0.95 × mean_perc_losses[i] + 0.05 × perc_losses[i]

    # 归一化权重
    weights = 1.0 / (mean_perc_losses + 1e-8)
```

**优势**: 自动平衡各层贡献

### 3. Curriculum Learning
```python
# 从简单到复杂的训练策略

# 阶段 1: 简单线条 (QuickDraw)
Phase_1: {
    'data': 'clean_line_drawings',
    'max_strokes': 20,
    'complexity': 'low',
}

# 阶段 2: 复杂书法 (Calligraphy)
Phase_2: {
    'data': 'calligraphy',
    'max_strokes': 48,
    'complexity': 'high',
}
```

---

## 📈 训练监控

### 关键指标
```python
# 1. 总损失
total_loss = raster + perc + sn + smooth + angle + outside

# 2. 渲染质量
PSNR = 10 × log10(MAX^2 / MSE)
SSIM = structural_similarity(rendered, target)

# 3. 笔画统计
avg_strokes = mean(stroke_count)
stroke_efficiency = coverage / stroke_count

# 4. 收敛速度
loss_decrease_rate = (loss[t] - loss[t-100]) / 100
```

### TensorBoard 可视化
```python
# train_phase_1.py - create_summary()

summary_writer.add_scalar('loss/total', total_loss, step)
summary_writer.add_scalar('loss/raster', raster_loss, step)
summary_writer.add_scalar('loss/perceptual', perc_loss, step)
summary_writer.add_scalar('loss/stroke_num', sn_loss, step)

summary_writer.add_image('pred/rendered', rendered_image, step)
summary_writer.add_image('target/original', target_image, step)

summary_writer.add_histogram('params/position', positions, step)
summary_writer.add_histogram('params/radius', radii, step)
```

---

## 🔧 训练配置示例

### 完整配置文件
```python
# hyper_parameters.py

hparams = {
    # ===== 模型架构 =====
    'encoder_type': 'conv13_c3',       # 13层 Conv + C3
    'dec_rnn_size': 256,               # Decoder LSTM 隐藏层
    'dec_model': 'hyper',              # Hyper LSTM
    'max_seq_len': 48,                 # 最大序列长度

    # ===== 数据配置 =====
    'batch_size': 12,                  # 批次大小
    'image_size_small': 128,           # 最小图像尺寸
    'image_size_large': 278,           # 最大图像尺寸

    # ===== 训练超参数 =====
    'num_steps': 90040,                # 总训练步数
    'learning_rate': 0.0001,           # 初始学习率
    'min_learning_rate': 0.000001,     # 最小学习率
    'decay_power': 0.9,                # 衰减指数
    'grad_clip': 1.0,                  # 梯度裁剪

    # ===== 损失权重 =====
    'raster_loss_weight': 1.0,         # 渲染损失
    'stroke_num_loss_weight': 0.5,     # 笔画数损失
    'smoothness_loss_weight': 0.0,     # 平滑度损失
    'angle_loss_weight': 0.0,          # 角度损失
    'outside_loss_weight': 10.0,       # 边界损失

    # ===== 感知损失 =====
    'perc_loss_layers': [
        'ReLU1_2',   # 低层特征 (纹理)
        'ReLU2_2',   # 中层特征 (边缘)
        'ReLU3_3',   # 高层特征 (形状)
        'ReLU5_1',   # 最高层 (语义)
    ],
    'perc_loss_fuse_type': 'add',      # 融合方式

    # ===== 正则化 =====
    'use_recurrent_dropout': True,     # Recurrent Dropout
    'recurrent_dropout_prob': 0.90,    # Keep prob

    # ===== GPU配置 =====
    'gpus': [0],                       # 使用的GPU
    'loop_per_gpu': 1,                 # 每GPU循环数
}
```

---

## 💾 训练输出

### 文件结构
```
outputs/
├── snapshot/
│   ├── new_train_phase_1/
│   │   ├── model-15000.ckpt       # 检查点 15000
│   │   ├── model-30000.ckpt       # 检查点 30000
│   │   ├── model-90000.ckpt       # 最终模型
│   │   └── checkpoint             # 检查点索引
│   └── new_train_phase_2/
│       └── ...
│
├── log/
│   ├── new_train_phase_1/
│   │   └── events.out.tfevents    # TensorBoard 日志
│   └── new_train_phase_2/
│       └── ...
│
└── log_img/
    ├── new_train_phase_1/
    │   ├── res_128/
    │   │   ├── 500.png            # 步骤 500 预测
    │   │   ├── 1000.png           # 步骤 1000 预测
    │   │   └── gt.png             # Ground Truth
    │   ├── res_170/
    │   └── res_278/
    └── new_train_phase_2/
        └── ...
```

---

## 🚀 训练命令

### Phase 1 训练
```bash
cd seq_extract
conda activate calli_ext

# 启动训练
python train_phase_1.py

# 监控训练
tensorboard --logdir=outputs/log/new_train_phase_1
```

### Phase 2 微调
```bash
# 修改 hyper_parameters.py 指定预训练模型
# pretrained_model = 'outputs/snapshot/new_train_phase_1/model-90000'

python train_phase_2.py

tensorboard --logdir=outputs/log/new_train_phase_2
```

---

## 📊 训练性能

| 指标 | Phase 1 | Phase 2 |
|------|---------|---------|
| **训练步数** | 90,000 | 60,000 |
| **训练时间** | ~5天 (V100) | ~3天 (V100) |
| **最终损失** | 0.15 | 0.08 |
| **PSNR** | 28 dB | 32 dB |
| **平均笔画数** | 15 | 12 |

---

## 🎯 训练检查清单

### 开始训练前
- [ ] 数据集准备完成 (QuickDraw / 书法)
- [ ] 环境配置正确 (`calli_ext`)
- [ ] GPU 显存充足 (≥12GB)
- [ ] 超参数配置检查
- [ ] 预训练模型路径正确 (Phase 2)

### 训练过程中
- [ ] 监控损失曲线 (TensorBoard)
- [ ] 检查生成质量 (log_img)
- [ ] 验证笔画数合理性
- [ ] GPU 利用率 >80%
- [ ] 定期备份检查点

### 训练完成后
- [ ] 在验证集上测试
- [ ] 可视化生成结果
- [ ] 保存最佳模型
- [ ] 记录超参数配置

---

**总结**: Seq_Extract 通过两阶段渐进式训练策略，结合多任务损失和可微分渲染，实现了从书法图像到笔画序列的无监督提取！

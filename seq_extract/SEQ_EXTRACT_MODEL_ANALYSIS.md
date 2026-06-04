# Seq_Extract 模型分析文档

## 目录
1. [概述](#概述)
2. [数据集](#数据集)
3. [训练方式](#训练方式)
4. [模型架构](#模型架构)
5. [参数分析](#参数分析)
6. [损失函数](#损失函数)

---

## 概述

Seq_Extract 是一个基于神经光栅化的草图向量化模型，采用两阶段训练策略。该模型通过逐步绘制的方式将光栅图像转换为矢量笔画序列。

**关键特性：**
- 可微分神经光栅化器
- 多分辨率训练
- HyperLSTM 解码器
- 感知损失函数

---

## 数据集

### 1. QuickDraw-Clean (Phase 1)

#### 数据集信息
- **来源**: Google QuickDraw 数据集
- **位置**: `datasets/QuickDraw-clean/`
- **类别**: 10个类别
  ```
  airplane, bus, car, sailboat, bird, 
  cat, dog, tree, flower, zigzag
  ```
- **格式**: npz 压缩文件，包含 stroke3 格式数据
  - 训练集: `train/{category}.npz`
  - 测试集: `test/{category}.npz`

#### 数据格式 (Stroke3)
```
stroke3: [dx, dy, pen_state]
- dx, dy: 相对坐标偏移
- pen_state: 0=继续绘制, 1=结束当前笔画
```

#### 数据加载器 (GeneralMultiObjectDataLoader)

**功能特性：**
1. **多物体合成**: 随机组合多个物体到一个画布
2. **多分辨率**: 支持 128x128 到 278x278 的随机分辨率
3. **笔画粗细变化**: 可选随机笔画粗细

**数据生成流程：**
```
1. 从 stroke3 数据随机采样物体
2. 生成随机分辨率 [128, 278]
3. 确定物体数量和位置：
   - 分辨率 ≤172: 1个物体
   - 172 < 分辨率 ≤225: 1-2个物体
   - 225 < 分辨率 ≤278: 2个物体
4. 使用 RealRenderer 将笔画光栅化
5. 将多个物体粘贴到一个画布
6. 生成初始光标位置（随机在未绘制像素上）
```

#### 下载脚本
```bash
python seq_extract/download_quickdraw_clean.py
```
支持多种来源：
- Google Cloud Storage (默认)
- Hugging Face Datasets
- ModelScope
- Google Creative Lab

### 2. GB Dataset (Phase 2, 可选)
- 用于照片到草图的迁移学习
- 图像尺寸: 256x256

---

## 训练方式

### 两阶段训练策略

#### Phase 1: QuickDraw 预训练

**训练脚本**: `train_phase_1.py`

**关键超参数** (`hyper_parameters.py`):

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_steps` | 90040 | 总训练步数 |
| `save_every` | 15000 | 保存间隔 |
| `batch_size` | 12 | 批次大小 |
| `max_seq_len` | 48 | 最大笔画序列长度 |
| `learning_rate` | 0.0001 | 初始学习率 |
| `min_learning_rate` | 0.000001 | 最小学习率 |
| `decay_power` | 0.9 | 学习率衰减指数 |
| `image_size_small` | 128 | 最小训练分辨率 |
| `image_size_large` | 278 | 最大训练分辨率 |
| `raster_size` | 128 | 光栅化输出尺寸 |
| `dec_rnn_size` | 256 | 解码器 RNN 隐藏层大小 |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |

**学习率调度**:
```python
# 多项式衰减
curr_learning_rate = (init_lr - min_lr) * (1 - step/total_steps)^power + min_lr
```

**笔画数损失权重调度**:
```python
# 递增模式
sn_loss_type = 'increasing'
stroke_num_loss_weight = 0.5  # 初始
# 从 0 开始递增到 0.5
```

**优化器**: Adam
**损失函数**: 见 [损失函数](#损失函数) 章节

**训练流程**:
```
1. 加载多分辨率数据 (128-278)
2. 每个分辨率随机采样
3. 模型逐步生成 48 个笔画
4. 计算总损失
5. 反向传播，梯度裁剪
6. 定期保存和记录
```

#### Phase 2: GB 数据集微调 (可选)

**训练脚本**: `train_phase_2.py`

**关键区别**:
- `data_set`: 'gb'
- `image_size_large`: 256
- `smoothness_loss_weight`: 0.5
- `angle_loss_weight`: 1.0

---

## 模型架构

### 整体架构 (VirtualSketchingModel)

```
输入图像 → CNN 编码器 → HyperLSTM 解码器 → 笔画参数 → 神经光栅化器 → 重建图像
                              ↑
                          光标位置
```

### 1. CNN 编码器

**类型**: `conv13_c3` (默认)

**架构**:
```
输入: (N, 128, 128, C)  C=通道数
  ├─ 本地图像块 + 全局图像
  ├─ 添加坐标卷积 (CoordConv)
  │
  ├─ Conv 32×3×3, stride=2, IN, ReLU
  ├─ Conv 32×3×3, stride=1, IN, ReLU
  ├─ Conv 64×3×3, stride=2, IN, ReLU
  ├─ Conv 64×3×3, stride=1, IN, ReLU
  ├─ Conv 128×3×3, stride=2, IN, ReLU
  ├─ Conv 128×3×3, stride=1, IN, ReLU
  ├─ Conv 256×3×3, stride=2, IN, ReLU
  ├─ Conv 256×3×3, stride=1, IN, ReLU
  ├─ Conv 256×3×3, stride=2, IN, ReLU
  ├─ Conv 256×3×3, stride=1, IN, ReLU
  │
  ├─ Reshape: (N, 256×4×4) = (N, 4096)
  └─ Linear: 4096 → 128
输出: (N, 128) 特征向量
```

**输入组成**:
```
encoder_input = [
    目标图像块,
    当前画布块,
    目标图像(缩放),
    当前画布(缩放),
    光标位置,
    上一笔宽度,
    窗口大小信息
]
```

**其他编码器选项**:
- `conv10`: 10层网络
- `conv10_deep`: 深层10层
- `conv13_c3_attn`: 带注意力机制
- `combine33/43/53/FC`: 双通路（本地+全局）编码器

### 2. HyperLSTM 解码器

**类型**: HyperLSTMCell (默认)

**架构**:
```
主 LSTM: 256 units
  ├─ 输入: CNN特征 + 光标位置
  ├─ 隐藏状态: (c, h)
  └─ 输出: 笔画参数

Hyper LSTM: 256 units
  ├─ 输入: 主 LSTM 的输入 + 隐藏状态
  └─ 输出: 动态调制主 LSTM 的权重

层数: 1层
循环 dropout: 0.9 keep prob
```

**HyperLSTM 工作原理**:
```
1. HyperLSTM 接收 [x_t, h_t]
2. 生成动态权重 α, β
3. 主 LSTM 权重 = α * W + β
4. 实现动态权重调制
```

**输入组成**:
```python
decoder_input = [
    cursor_position,        # 当前光标 (2D)
    cnn_feature,            # CNN 编码 (128D)
    prev_width,             # 上一笔宽度 (1D)
    window_size_info,       # 窗口大小 (2D)
]
```

**输出参数**:
```
output = [
    pen_state,              # 笔状态 (2D, softmax)
    x1, y1,                 # 控制点 (2D, sigmoid [0,1])
    x2, y2,                 # 结束点 (2D, tanh [-1,1] → 归一化)
    width,                  # 笔画宽度 (1D, sigmoid)
    scaling                 # 缩放因子 (1D, sigmoid)
]
```

### 3. 神经光栅化器 (NeuralRenderer)

**RasterUnit 架构**:
```
输入: (N, 10) 笔画参数
  ├─ FC 10 → 512, ReLU
  ├─ FC 512 → 1024, ReLU
  ├─ FC 1024 → 2048, ReLU
  ├─ FC 2048 → 4096, ReLU
  │
  ├─ Reshape: 4096 → (16, 16, 16)
  ├─ Conv 16→32×3×3, stride=1, ReLU
  ├─ Conv 32→32×3×3, stride=1, ReLU
  ├─ PixelShuffle (upscale ×2): 32 → 8
  ├─ Output: (32, 32, 8)
  │
  ├─ Conv 8→16×3×3, stride=1, ReLU
  ├─ Conv 16→16×3×3, stride=1, ReLU
  ├─ PixelShuffle (upscale ×2): 16 → 4
  ├─ Output: (64, 64, 4)
  │
  ├─ Conv 4→8×3×3, stride=1, ReLU
  ├─ Conv 8→4×3×3, stride=1, ReLU
  ├─ PixelShuffle (upscale ×2): 4 → 1
  ├─ Output: (128, 128, 1)
  │
  └─ Sigmoid → (N, 128, 128)
输出: (N, 128, 128) 光栅图像 [0,1]
```

**输入参数格式**:
```
stroke_params = [x0, y0, x1, y1, x2, y2, r0, r2, w0, w2]
- x0,y0: 起始点
- x1,y1: 控制点
- x2,y2: 结束点
- r0,r2: 预留参数
- w0,w2: 起始/结束宽度
```

**光栅化流程**:
```
1. 每步生成一个笔画
2. 使用 RasterUnit 渲染
3. 可微分粘贴到大画布
4. 累加所有笔画
5. Clip 到 [0,1]
```

### 4. 逐步绘制循环

**伪代码**:
```python
canvas = zeros(image_size)
cursor = init_cursor
prev_width = init_width
prev_window = init_window

for t in range(max_seq_len):
    # 裁剪当前图像块
    patch = crop(canvas, cursor, prev_window)
    
    # CNN 编码
    feature = cnn_encoder(patch, target_image, cursor)
    
    # RNN 解码
    stroke_params, next_state = lstm_decoder(feature, prev_state)
    
    # 光栅化
    stroke_image = neural_rasterizer(stroke_params)
    
    # 粘贴到画布
    canvas = paste(canvas, stroke_image, cursor, prev_window)
    
    # 更新状态
    cursor = cursor + stroke_params.offset
    prev_width = stroke_params.width
    prev_window = prev_window * stroke_params.scaling

return canvas
```

---

## 参数分析

### 模型参数量估算

#### 1. CNN 编码器 (conv13_c3)
```
Conv 层:
  - CNN_ENC_1: (3×3×C+1)×32 ≈ 1K
  - CNN_ENC_1_2: (3×3×32+1)×32 ≈ 9K
  - CNN_ENC_2: (3×3×32+1)×64 ≈ 18K
  - CNN_ENC_2_2: (3×3×64+1)×64 ≈ 37K
  - CNN_ENC_3: (3×3×64+1)×128 ≈ 74K
  - CNN_ENC_3_2: (3×3×128+1)×128 ≈ 147K
  - CNN_ENC_4: (3×3×128+1)×256 ≈ 295K
  - CNN_ENC_4_2: (3×3×256+1)×256 ≈ 590K
  - CNN_ENC_5: (3×3×256+1)×256 ≈ 590K
  - CNN_ENC_5_2: (3×3×256+1)×256 ≈ 590K
FC 层:
  - CNN_ENC_FC: 4096×128 = 524K

小计: ≈ 2.87M 参数
```

#### 2. HyperLSTM 解码器
```
主 LSTM:
  - W_xh: (input_size × 4×256) ≈ (135 × 1024) = 138K
  - W_hh: (256 × 4×256) = 262K
  - Bias: 4×256 = 1K

Hyper LSTM:
  - W_xh: ((135+256) × 4×256) = 391 × 1024 = 400K
  - W_hh: (256 × 4×256) = 262K
  - Bias: 4×256 = 1K
  - Hyper 投影: 多个小线性层 ≈ 10K

输出层:
  - DEC_RNN_out_pen: 256×2 = 512
  - DEC_RNN_out_params: 256×6 = 1.5K

小计: ≈ 1.08M 参数
```

#### 3. 神经光栅化器 (RasterUnit)
```
FC 层:
  - fc1: 10×512 = 5K
  - fc2: 512×1024 = 524K
  - fc3: 1024×2048 = 2M
  - fc4: 2048×4096 = 8M

Conv 层:
  - conv1: (3×3×16+1)×32 = 4.6K
  - conv2: (3×3×32+1)×32 = 9.2K
  - conv3: (3×3×8+1)×16 = 1.2K
  - conv4: (3×3×16+1)×16 = 2.3K
  - conv5: (3×3×4+1)×8 = 296
  - conv6: (3×3×8+1)×4 = 292

小计: ≈ 10.55M 参数
```

#### 4. VGG16 (感知损失, 冻结)
```
仅用于特征提取，不训练
参数量: ≈ 138M (但不计入训练参数)
```

### 总参数量总结

| 组件 | 参数量 | 是否训练 |
|------|--------|----------|
| CNN 编码器 | ~2.87M | 是 |
| HyperLSTM 解码器 | ~1.08M | 是 |
| 神经光栅化器 | ~10.55M | 否 (预训练冻结) |
| VGG16 | ~138M | 否 (冻结) |
| **总计 (可训练)** | **~3.95M** | - |

### 显存估算

**训练时 (batch_size=12, max_seq_len=48)**:
```
激活值:
  - CNN 中间特征: ≈ 12×128×128×256 ≈ 50MB
  - RNN 序列: 12×48×256 ≈ 1.5MB
  - 画布历史: 12×48×278×278 ≈ 420MB
  - 光栅化输出: 12×48×128×128 ≈ 90MB

梯度:
  - 参数梯度: ≈ 3.95M × 4 × 2 ≈ 32MB (forward+backward)
  - 优化器状态 (Adam): ≈ 3.95M × 8 ≈ 32MB

总计: ≈ 650MB (使用单 GPU)
```

---

## 损失函数

### 总损失

```
total_loss = (
    raster_loss_weight * raster_loss
    + early_pen_loss_weight * early_pen_loss
    + stroke_num_loss_weight * stroke_num_loss
    + smoothness_loss_weight * smoothness_loss
    + angle_loss_weight * angle_loss
    + outside_loss_weight * cursor_outside_loss
    + win_size_outside_loss_weight * window_outside_loss
)
```

### 各损失详解

#### 1. 光栅损失 (Raster Loss)

**类型**: 感知损失 (Perceptual Loss)

**VGG16 层**:
```
ReLU1_2, ReLU2_2, ReLU3_3, ReLU5_1
```

**计算方式**:
```python
# 对每个 VGG 层
for layer in perc_loss_layers:
    pred_feat = vgg16(pred_image)
    target_feat = vgg16(target_image)
    
    # L1 距离
    layer_loss_raw = mean(|pred_feat - target_feat|)
    
    # 归一化 (使用运行均值)
    layer_loss_norm = layer_loss_raw / running_mean[layer]

# 融合方式: 'add' (均值)
raster_loss = mean(layer_loss_norm for all layers)
```

**替代方案**:
- `l1`: 像素级 L1 距离
- `mse`: 像素级 MSE
- `triplet`: 三元组损失 (未实现)

#### 2. 笔画数损失 (Stroke Number Loss)

**目标**: 鼓励尽早结束绘制

**计算方式**:
```python
# pen_state[:, 0] = 继续绘制的概率
# pen_state[:, 1] = 结束的概率
mean_end_prob = mean(pen_state[:, 1] for all steps)
stroke_num_loss = 1 - mean_end_prob  # 越小越好
```

**权重调度**:
- Phase 1: `increasing`, 从 0 增加到 0.5
- Phase 2: `fixed`, 固定 0.5

#### 3. 提前结束损失 (Early Penalty Loss)

**目标**: 防止过早结束

**类型**: `move` (默认)

**计算方式**:
```python
# 动态移动有效区间
start_idx = dynamic_start
end_idx = dynamic_end

# 最小结束概率 (应接近 0)
min_end_prob = min(pen_state[start_idx:end_idx, 1])
early_pen_loss = min_end_prob  # 越小越好
```

#### 4. 平滑损失 (Smoothness Loss)

**目标**: 鼓励笔画方向连续

**计算方式**:
```python
for t in 1..T:
    # 当前和上一个偏移向量
    v_prev = stroke_params[t-1].offset
    v_curr = stroke_params[t].offset
    
    # 归一化
    v_prev = normalize(v_prev)
    v_curr = normalize(v_curr)
    
    # 余弦相似度
    cos_sim = dot(v_prev, v_curr)
    
    # 仅在两笔都在绘制时计算
    if pen_state[t-1, 0] and pen_state[t, 0]:
        smoothness_loss += (1 - cos_sim)  # 越小越好
```

**权重**:
- Phase 1: 0.0
- Phase 2: 0.5

#### 5. 角度损失 (Angle Loss)

**目标**: 防止急转弯

**计算方式**:
```python
# 与上一方向的夹角
# 仅当连续绘制时惩罚大角度变化
```

**权重**:
- Phase 1: 0.0
- Phase 2: 1.0

#### 6. 光标越界损失 (Cursor Outside Loss)

**目标**: 防止光标移出画布

**计算方式**:
```python
cursor_pos_clipped = clip(cursor_pos, 0, image_size)
cursor_outside_loss = mean(|cursor_pos - cursor_pos_clipped|)
```

**权重**: 10.0

#### 7. 窗口大小越界损失 (Window Size Outside Loss)

**目标**: 保持窗口大小在合理范围

**计算方式**:
```python
# 上界
over_size = max(window_size - image_size, 0)
# 下界
under_size = max(min_window_size - window_size, 0)

window_outside_loss = mean(over_size / image_size + under_size / min_window_size)
```

**权重**: 10.0

**min_window_size**: 32

### 默认损失权重

| 损失项 | Phase 1 | Phase 2 |
|--------|---------|---------|
| `raster_loss_weight` | 1.0 | 1.0 |
| `early_pen_loss_weight` | 0.1 | 0.1 |
| `stroke_num_loss_weight` | 0-0.5 (递增) | 0.5 (固定) |
| `smoothness_loss_weight` | 0.0 | 0.5 |
| `angle_loss_weight` | 0.0 | 1.0 |
| `outside_loss_weight` | 10.0 | 10.0 |
| `win_size_outside_loss_weight` | 10.0 | 10.0 |

---

## 推理流程

### 推理模式

**配置**:
```python
model_mode = 'sample'
batch_size = 1
use_input_dropout = 0
use_recurrent_dropout = 0
use_output_dropout = 0
```

**采样策略**:
```
1. 初始光标: 随机在黑色像素上
2. 每步预测 pen_state
3. 使用 softargmax (可微分) 或 argmax
4. 直到 pen_state[1] > 0.5 或达到 max_seq_len
```

### 可微分 Argmax (Softargmax)

```python
def softargmax(logits, beta=10):
    # softmax with high temperature
    weights = softmax(logits * beta)
    # 期望
    argmax_val = sum(i * weights[i] for i)
    return argmax_val
```

---

## 复现指南

### 环境配置

```bash
# 使用 conda
conda env create -f seq_extract/environment.yml
conda activate seq_extract

# 或使用 pip
pip install tensorflow-gpu==1.x numpy scipy pillow
```

### 数据准备

```bash
# 下载 QuickDraw-Clean
cd seq_extract
python download_quickdraw_clean.py --source gcs
```

### 预训练模型

需要准备:
1. **神经光栅化器**: `outputs/snapshot/pretrain_neural_renderer/renderer_300000.tfmodel`
2. **VGG16**: `outputs/snapshot/pretrain_perceptual_model/`

### 开始训练

```bash
# Phase 1
python train_phase_1.py

# Phase 2 (可选)
python train_phase_2.py
```

### 监控训练

```bash
# TensorBoard
tensorboard --logdir outputs/log
```

---

## 总结

### 关键设计特点

1. **可微分光栅化**: 端到端训练
2. **多分辨率训练**: 增强泛化
3. **HyperLSTM**: 动态建模能力
4. **感知损失**: 感知对齐
5. **渐进式生成**: 模拟人类绘制

### 性能指标

- **推理速度**: ~10-50ms/图像 (取决于分辨率)
- **显存占用**: ~650MB (训练)
- **模型大小**: ~15MB (可训练参数)
- **总参数量**: ~3.95M (可训练)

### 扩展方向

1. 更长序列 (max_seq_len > 48)
2. 更多类别
3. 更高分辨率
4. 交互式编辑
5. 风格迁移

---

## 参考资料

1. **QuickDraw Dataset**: https://quickdraw.withgoogle.com/
2. **HyperLSTM**: https://arxiv.org/abs/1609.09106
3. **Perceptual Losses**: https://arxiv.org/abs/1603.08155
4. **Neural Renderer**: (内部预训练)

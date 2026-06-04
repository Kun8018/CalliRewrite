# ViT Query 模型参数量分析

基于 `vit_query/model.py` 手动计算。

---

## 一、ViT Backbone (ViT-B/16)

### 整体架构

```
输入: (N, 1, 224, 224)
  ↓
Conv Proj: 1 → 768, k=16, s=16
  ↓
12 × Transformer Encoder Layer
  ↓
Linear: 768 → 256
  ↓
输出: (N, 196, 256)
```

### 详细参数量

#### 1. Conv Proj (patch embedding)

```
Conv2d: in=1, out=768, k=16×16, s=16
  - Weight: 16×16×1×768 = 196,608
  - Bias: 768
  - 小计: 197,376 ≈ 0.20M
```

#### 2. Positional Embedding

```
pos_embedding: (1, 197, 768)
cls_token: (1, 1, 768)
  - 小计: (197 + 1) × 768 = 152,064 ≈ 0.15M
```

#### 3. Transformer Encoder Layer (×12)

**单层参数**:

```
Multi-head Attention (12 heads):
  - Q_proj: 768×768 = 589,824
  - K_proj: 768×768 = 589,824
  - V_proj: 768×768 = 589,824
  - out_proj: 768×768 = 589,824
  - Attention 小计: 2,359,296 ≈ 2.36M

MLP:
  - fc1: 768×3072 = 2,359,296
  - fc2: 3072×768 = 2,359,296
  - MLP 小计: 4,718,592 ≈ 4.72M

Layer Norms:
  - ln1: 768×2 = 1,536
  - ln2: 768×2 = 1,536
  - LN 小计: 3,072 ≈ 0.003M

单层总计: ~7.08M
```

**12层总计**:
```
12 × 7.08M = 84.96M
```

#### 4. Final Projection

```
Linear: 768 → 256
  - Weight: 768×256 = 196,608
  - Bias: 256
  - 小计: 196,864 ≈ 0.20M
```

### ViT Backbone 总计

| 组件 | 参数量 |
|------|--------|
| Conv Proj | 0.20M |
| Pos/Cls Tokens | 0.15M |
| 12×Encoder Layers | 84.96M |
| Final Proj | 0.20M |
| **总计** | **85.51M** |

---

## 二、Patch Encoder (小 CNN)

### 架构

```
输入: (N, 2, 64, 64)  [target_patch + canvas_patch]
  ↓
Conv2d: 2→32, k=5×5, s=2, p=2
  ↓ GELU
Conv2d: 32→64, k=3×3, s=2, p=1
  ↓ GELU
Conv2d: 64→128, k=3×3, s=2, p=1
  ↓ GELU
Flatten: 128×8×8 = 8,192
  ↓
Linear: 8,192 → 256
  ↓
LayerNorm: 256
  ↓
输出: (N, 256)
```

### 详细参数量

```
Conv1:
  - Weight: 5×5×2×32 = 1,600
  - Bias: 32
  - 小计: 1,632

Conv2:
  - Weight: 3×3×32×64 = 18,432
  - Bias: 64
  - 小计: 18,496

Conv3:
  - Weight: 3×3×64×128 = 73,728
  - Bias: 128
  - 小计: 73,856

Linear Proj:
  - Weight: 8192×256 = 2,097,152
  - Bias: 256
  - 小计: 2,097,408

LayerNorm:
  - Weight: 256
  - Bias: 256
  - 小计: 512

Patch Encoder 总计:
  1,632 + 18,496 + 73,856 + 2,097,408 + 512
  = 2,191,904 ≈ 2.19M
```

---

## 三、Canvas Encoder (CNN)

### 架构

```
输入: (N, 1, H, W)
  ↓
Conv2d: 1→32, k=5×5, s=2, p=2
  ↓ GELU
Conv2d: 32→64, k=3×3, s=2, p=1
  ↓ GELU
Conv2d: 64→256, k=3×3, s=2, p=1
  ↓ GELU
AdaptiveAvgPool2d: (1,1)
  ↓ Flatten
LayerNorm: 256
  ↓
输出: (N, 256)
```

### 详细参数量

```
Conv1:
  - Weight: 5×5×1×32 = 800
  - Bias: 32
  - 小计: 832

Conv2:
  - Weight: 3×3×32×64 = 18,432
  - Bias: 64
  - 小计: 18,496

Conv3:
  - Weight: 3×3×64×256 = 147,456
  - Bias: 256
  - 小计: 147,712

LayerNorm:
  - Weight: 256
  - Bias: 256
  - 小计: 512

Canvas Encoder 总计:
  832 + 18,496 + 147,712 + 512
  = 167,552 ≈ 0.17M
```

---

## 四、其他小模块

### 1. Cursor MLP

```
Linear: 2 → 256
Linear: 256 → 256
LayerNorm: 256

参数量:
  (2×256 + 256) + (256×256 + 256) + (256+256)
  = 768 + 65,792 + 512
  = 67,072 ≈ 0.07M
```

### 2. Window MLP

```
同 Cursor MLP: ≈ 0.07M
```

### 3. Step MLP

```
同 Cursor MLP: ≈ 0.07M
```

### 4. Patch-Target Attention

```
MultiheadAttention: embed_dim=256, num_heads=16
  - in_proj_weight: 256×3×256 = 196,608
  - in_proj_bias: 3×256 = 768
  - out_proj_weight: 256×256 = 65,536
  - out_proj_bias: 256
  - 小计: 263,168 ≈ 0.26M
```

### 5. GRU Input Projection

```
Linear: 256×5 = 1280 → 256
LayerNorm: 256

参数量:
  (1280×256 + 256) + (256+256)
  = 327,936 + 512
  = 328,448 ≈ 0.33M
```

### 6. GRU Cell

```
GRUCell: input_size=256, hidden_size=256

参数量:
  - Weight_ih: 3×256 × 256 = 196,608
  - Weight_hh: 3×256 × 256 = 196,608
  - Bias_ih: 3×256 = 768
  - Bias_hh: 3×256 = 768
  - 小计: 394,752 ≈ 0.39M
```

### 7. Stroke Head

```
Linear(256→2) + Linear(256→2) + Linear(256→2) + Linear(256→1) + Linear(256→1)

参数量:
  (256×2+2) + (256×2+2) + (256×2+2) + (256×1+1) + (256×1+1)
  = 514 + 514 + 514 + 257 + 257
  = 2,056 ≈ 0.002M
```

---

## 五、Neural Renderer (来自 seq_extract)

### 架构

```
输入: (N, 10)
  ↓
FC: 10→512 → ReLU
FC: 512→1024 → ReLU
FC: 1024→2048 → ReLU
FC: 2048→4096 → ReLU
  ↓ Reshape
(1,16,16,16) → Transpose → (1,16,16,16)
  ↓
Conv: 16→32, k=3×3, s=1
  ↓ ReLU
Conv: 32→32, k=3×3, s=1
  ↓ PixelShuffle ×2
(32,32,8)
  ↓
Conv: 8→16, k=3×3, s=1
  ↓ ReLU
Conv: 16→16, k=3×3, s=1
  ↓ PixelShuffle ×2
(64,64,4)
  ↓
Conv: 4→8, k=3×3, s=1
  ↓ ReLU
Conv: 8→4, k=3×3, s=1
  ↓ PixelShuffle ×2
(128,128,1)
  ↓ Sigmoid
输出: (N, 128, 128)
```

### 详细参数量

```
FC 层:
  - fc1: 10×512 + 512 = 5,632
  - fc2: 512×1024 + 1024 = 525,312
  - fc3: 1024×2048 + 2048 = 2,099,200
  - fc4: 2048×4096 + 4096 = 8,392,704
  - FC 小计: 11,022,848 ≈ 11.02M

Conv 层:
  - conv1: 3×3×16×32 + 32 = 4,640
  - conv2: 3×3×32×32 + 32 = 9,248
  - conv3: 3×3×8×16 + 16 = 1,168
  - conv4: 3×3×16×16 + 16 = 2,320
  - conv5: 3×3×4×8 + 8 = 296
  - conv6: 3×3×8×4 + 4 = 292
  - Conv 小计: 17,964 ≈ 0.02M

Neural Renderer 总计:
  11.02M + 0.02M ≈ 11.04M
```

---

## 六、总参数量汇总

### ViT Query 完整模型

| 组件 | 参数量 | 是否训练 |
|------|--------|----------|
| ViT Backbone | 85.51M | 是 (可选冻结) |
| Patch Encoder CNN | 2.19M | 是 |
| Canvas Encoder CNN | 0.17M | 是 |
| Cursor/Window/Step MLPs | 0.21M | 是 |
| Patch-Target Attention | 0.26M | 是 |
| GRU Input Proj | 0.33M | 是 |
| GRU Cell | 0.39M | 是 |
| Stroke Head | 0.002M | 是 |
| **主要模型总计** | **89.06M** | - |
| Neural Renderer | 11.04M | 否 (冻结) |
| **全部总计** | **100.10M** | - |

### 可训练参数量 (不同配置)

| 配置 | 可训练参数量 |
|------|-------------|
| ViT全部微调 | 89.06M |
| ViT冻结，只训其余 | 3.55M |

---

## 七、与 Seq-Extract 对比

| 组件 | seq_extract | vit_query |
|------|-------------|-----------|
| Backbone | CNN (~2.87M) | ViT (~85.51M) |
| Decoder | HyperLSTM (~1.08M) | GRU (~0.39M) |
| CNN 小模块 | - | Patch+Canvas (~2.36M) |
| **主要模型总计** | **3.95M** | **89.06M** |
| Neural Renderer | 10.55M (冻结) | 11.04M (冻结) |

**参数量比值**: **22.5x** (89.06M / 3.95M)

---

## 八、显存估算 (推理)

| 组件 | 显存 (batch=8, seq_len=48) |
|------|---------------------------|
| ViT 特征 | ~8×196×768×4 ≈ 4.7MB |
| Canvas 历史 | ~8×48×224×224×4 ≈ 77MB |
| 激活值 | ~50MB |
| **总计** | **~132MB** |

---

## 总结

### ViT Query 的特点

1. **ViT Backbone**: ~85.51M (占比 ~96%)
2. **CNN 模块**: ~2.36M (Patch+Canvas Encoder)
3. **GRU + Head**: ~0.72M
4. **Neural Renderer**: ~11.04M (冻结)

### 训练策略建议

- **Phase 1**: 冻结 ViT，只训其余 (~3.55M)，快速迭代
- **Phase 2**: 放开 ViT 微调 (~89.06M)，精细调整

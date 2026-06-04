# ViT + Trajectory Queries 架构

使用 Vision Transformer + Trajectory Queries 的轻量模型，从图像直接并行输出完整笔画序列。

## 架构特点

```
输入图像 (1, 224, 224)
      ↓
ViT-Tiny Backbone → Patch Features (196 tokens)
      ↓
Trajectory Queries (可学习的点查询)
      ↓
Transformer Decoder (交叉注意力)
      ↓
输出: 完整笔画序列 (一次性输出，非自回归)
```

## 两种输出模式

### 1. `mode='seq7'` (默认，推荐)
- 输出: `(seq_len, 7)` 格式
- `[pen_state, x1, y1, x2, y2, r, s]`
- **完全兼容 seq_extract，可直接用于二阶段**

### 2. `mode='points'`
- 输出: `(num_points, 2)` 格式
- `[x, y]` 密集点坐标
- 用于直接回归轨迹点

## 文件说明

- `model.py`: 模型定义
  - `ViTTinyBackbone`: 极简 ViT 骨干网络
  - `ViTTrajectoryExtractor`: 2D 点输出模型
  - `ViTTrajectoryExtractor7D`: 7D 序列输出模型（兼容二阶段）
- `dataset.py`: 数据加载
- `train.py`: 训练脚本
- `inference.py`: 推理脚本，生成 npz
- `test_model.py`: 模型测试

## 快速开始

### 测试模型

```bash
cd vit_query
python test_model.py
```

### 训练模型

```bash
# 使用 7D 序列模式（推荐）
python train.py \
    --data_dir ../seq_extract/outputs/__new_train_phase_2 \
    --output_dir ./output \
    --mode seq7 \
    --img_size 224 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4
```

### 推理生成 npz

```bash
python inference.py \
    --checkpoint ./output/model_best.pth \
    --input ../rl_finetune/data/test_data \
    --output_dir ./inference_output
```

## 模型对比

| 特性 | lightweight/ (ResNet + Transformer) | vit_query/ (ViT + Queries) |
|------|-------------------------------------|---------------------------|
| 骨干网络 | ResNet-18 | ViT-Tiny |
| 解码方式 | 自回归（逐步生成） | 并行（一次性输出） |
| 训练方式 | Teacher forcing | 直接回归 |
| 参数量 | ~13.68M | ~15M (可调整) |
| 输出格式 | 7D 序列 | 7D 序列 or 2D 点 |
| 推理速度 | 慢（逐点生成） | 快（并行输出） |

## 架构细节

### Trajectory Queries

使用可学习的 query 向量，每个 query 负责预测一个输出点：

```python
self.traj_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
```

### 交叉注意力

Decoder 中 queries 去关注图像特征：

```python
out = self.decoder(tgt=queries, memory=features)
```

### ViT 输入归一化

```
图像 → [0, 1] 归一化
```

## 与二阶段集成

生成的 npz 文件格式和 seq_extract 完全一致，可以直接喂给 rl_finetune：

```
vit_query 输出 npz
      ↓
rl_finetune 直接使用
```

## 超参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| img_size | 224 | ViT 标准输入 |
| embed_dim | 192 | ViT-Tiny 维度 |
| seq_len | 100 | 输出序列长度 |
| batch_size | 32-64 | 批次大小 |
| lr | 1e-4 | 学习率 |
| epochs | 100-200 | 训练轮数 |

## 进阶使用

### 使用不同的 ViT 配置

```python
# 修改 model.py 中的 ViTTinyBackbone
self.vit = ViTTinyPatch16X16(
    img_size=224,
    patch_size=16,
    in_chans=1,
    embed_dim=256,  # 加大维度
    depth=12,
    num_heads=4      # 增加头数
)
```

### 调整 Decoder 层数

```python
self.decoder = nn.TransformerDecoder(
    decoder_layer,
    num_layers=6  # 从 2 改为 6
)
```

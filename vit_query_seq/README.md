# ViT + 渐进式图片序列

从**逐步添加笔画**的图片序列中学习生成完整笔画轨迹。

## 原理

```
输入:
  img_00.png (只有第一笔)
  img_01.png (第一笔+第二笔)
  img_02.png (第一笔+第二笔+第三笔)
  ...
  img_09.png (完整字)
  ↓
ViT 编码器 (处理每一张图)
  ↓
跨图注意力 (组合所有图的信息)
  ↓
Trajectory Queries + Decoder
  ↓
输出: 完整 7D 笔画序列
```

## 数据目录结构

### 格式 A: 多字符目录 (推荐)
```
data_dir/
  character_001/
    img_00.png
    img_01.png
    ...
    img_09.png
    data.npz
  character_002/
    img_00.png
    ...
```

### 格式 B: 简化单目录
```
data_dir/
  img_00.png
  img_01.png
  ...
  img_09.png
  data.npz
```

## 文件说明

- `model.py`: 核心模型
  - `MultiImageViTEncoder`: 多图编码器
  - `ViTSeqTrajectoryExtractor7D`: 完整模型
- `dataset.py`: 数据加载
- `train.py`: 训练脚本
- `inference.py`: 推理脚本
- `test_model.py`: 测试

## 快速开始

### 测试模型
```bash
cd vit_query_seq
python test_model.py
```

### 训练
```bash
# 格式 A (推荐)
python train.py --data_dir /path/to/data --output_dir ./output --epochs 100

# 格式 B (简化)
python train.py --data_dir /path/to/simple_data --simple_dataset
```

### 推理
```bash
# 单张图片 (会复制成序列)
python inference.py --checkpoint ./output/model_best.pth --input char.png

# 图片序列目录
python inference.py --checkpoint ./output/model_best.pth --input ./char_images/
```

## 千问生成数据的使用流程

1. **千问生成渐进式图片**
   ```
   输入: "写一个'永'字，显示每一步"
   输出: 10张图片，每加一笔保存一张
   ```

2. **准备数据目录**
   ```
   qwen_data/
     yong/
       img_00.png (第一笔)
       img_01.png (第二笔)
       ...
       data.npz (真实笔画，或用现有模型生成)
   ```

3. **训练**
   ```bash
   python train.py --data_dir ./qwen_data --epochs 100
   ```

## 模型架构

### MultiImageViTEncoder
```
输入: (B, num_images, 1, 224, 224)
  ↓
每张图单独用 ViT 编码
  ↓
添加图像位置编码
  ↓
跨图 Transformer 注意力
  ↓
输出: (B, num_images * num_patches, embed_dim)
```

### 完整模型参数量
- **约 12-15M** (可调)

## 和其他版本对比

| 版本 | 输入 | 特点 |
|------|------|------|
| `lightweight/` | 单张图 | ResNet + 自回归 |
| `vit_query/` | 单张图 | ViT + Queries (并行输出) |
| `vit_query_seq/` | 图片序列 | ViT + 跨图注意力 |

## 超参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--num_images` | 10 | 图片序列长度 |
| `--img_size` | 224 | 图片大小 |
| `--embed_dim` | 192 | 特征维度 |
| `--seq_len` | 100 | 输出序列长度 |

## 输出格式

完全兼容 `seq_extract`，可以直接喂给二阶段强化学习：

```
output.npz
  ├── strokes_data: (seq_len, 7)
  ├── init_cursors
  ├── image_size
  ├── round_length
  └── init_width
```

# ViT + 彩色标注笔画

从**红色标注当前笔画**的图片中学习生成笔画序列。

## 数据示例
```
一张图里有多个字，红色标注当前要写的那一笔
  ↓
模型学习: 看红色部分，输出这个笔画的参数
```

## 两种输入模式

### 模式 1: RGB 输入 (`--mode rgb`)
```
输入: (3, 224, 224) - RGB图片
红色通道标注当前笔画
```

### 模式 2: 灰度 + Mask (`--mode dual`)
```
输入: 
  gray: (1, 224, 224) - 灰度图
  mask: (1, 224, 224) - 红色提取的二值mask
```

## 数据目录结构

### 格式 A: 单字符目录
```
data_dir/
  img_00.png (红色标第1笔)
  img_01.png (红色标第2笔)
  ...
  data.npz (完整笔画)
```

### 格式 B: 多字符目录
```
data_dir/
  char_001/
    img_00.png
    ...
    data.npz
  char_002/
    ...
```

## 文件说明

- `model.py`: 核心模型
  - `ViTColorTrajectoryExtractor7D`: RGB输入版本
  - `ViTDualTrajectoryExtractor7D`: 灰度+mask版本
- `dataset.py`: 数据加载
- `train.py`: 训练脚本
- `inference.py`: 推理脚本
- `test_model.py`: 测试

## 快速开始

### 测试模型
```bash
cd vit_query_color
python test_model.py
```

### 训练
```bash
# RGB 模式 (推荐)
python train.py --data_dir ./my_data --mode rgb --epochs 100

# Dual 模式 (灰度+mask)
python train.py --data_dir ./my_data --mode dual --epochs 100

# 多字符格式
python train.py --data_dir ./multi_char_data --multi_char
```

### 推理
```bash
# RGB 模式
python inference.py --checkpoint ./output/model_best.pth --input char.png --mode rgb

# Dual 模式
python inference.py --checkpoint ./output/model_best.pth --input char.png --mode dual
```

## 模型参数量

| 模型 | 参数量 |
|------|--------|
| RGB 版本 | ~6.2M |
| Dual 版本 | ~6.1M |

## 四个版本对比

| 目录 | 输入 | 特点 |
|------|------|------|
| `lightweight/` | 单张图 | ResNet + 自回归 |
| `vit_query/` | 单张图 | ViT + Queries |
| `vit_query_seq/` | 图片序列 | 渐进式笔画 |
| `vit_query_color/` | RGB/dual | 红色标注当前笔 |

## 千问生成数据

1. **千问生成彩色图**
   ```
   输入: "写一个'永'字，每一步用红色标注"
   输出: 多张图，每张图红色标一笔
   ```

2. **准备数据目录**
   ```
   qwen_color_data/
     yong/
       img_00.png (红色标第1笔)
       img_01.png (红色标第2笔)
       ...
       data.npz (真实笔画)
   ```

3. **训练**
   ```bash
   python train.py --data_dir ./qwen_color_data --multi_char --mode rgb
   ```

## 输出格式

完全兼容 `seq_extract`：

```
output.npz
  ├── strokes_data: (seq_len, 7)
  ├── init_cursors
  ├── image_size
  ├── round_length
  └── init_width
```

可以直接喂给二阶段强化学习！

## 超参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--mode` | rgb | rgb/dual |
| `--img_size` | 224 | 图片大小 |
| `--embed_dim` | 192 | 特征维度 |
| `--seq_len` | 100 | 输出序列长度 |
| `--multi_char` | False | 多字符格式 |

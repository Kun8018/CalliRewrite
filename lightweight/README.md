# ResNet-18 + Transformer 轻量模型

用于替换 seq_extract 一阶段的轻量模型，使用 ResNet-18 编码器 + Transformer 解码器架构。

## 架构说明

```
输入: 书法图像 (1, 256, 256)
    ↓
ResNet-18 编码器 (修改为单通道输入)
    ↓
特征投影 (512 → d_model)
    ↓
Transformer 解码器 (自回归生成)
    ↓
输出: 笔画序列 (seq_len, 7)
       [pen_state, x1, y1, x2, y2, r, s]
```

## 文件说明

- `model.py`: 模型定义 (StrokeTransformer)
- `dataset.py`: 数据加载模块
- `train.py`: 训练脚本
- `inference.py`: 推理脚本，生成兼容二阶段的 npz 文件

## 安装依赖

```bash
pip install torch torchvision numpy pillow tqdm wandb
```

## 快速开始

### 1. 准备数据

#### 选项 A: 使用 QuickDraw 数据集

下载 QuickDraw npz 文件 (例如 `full_numpy_bitmap_apple.npz`):
```bash
# 从 Google QuickDraw 下载
wget https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/apple.npy
# 注意：需要转换格式，或使用我们提供的转换器
```

或者使用现有的 seq_extract 格式数据:
```
data_dir/
  ├── image1.png
  ├── image1.npz
  ├── image2.png
  ├── image2.npz
  └── ...
```

### 2. 训练

```bash
# 使用 QuickDraw 数据训练
python train.py \
    --quickdraw_npz path/to/quickdraw.npz \
    --quickdraw_save_dir ./qd_data \
    --output_dir ./output \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4

# 或使用现有数据训练
python train.py \
    --data_dir ../rl_finetune/data \
    --output_dir ./output
```

### 3. 推理

```bash
python inference.py \
    --checkpoint ./output/model_best.pth \
    --input path/to/image.png \
    --output_dir ./inference_output

# 或批量处理目录
python inference.py \
    --checkpoint ./output/model_best.pth \
    --input path/to/images_dir \
    --output_dir ./inference_output
```

## 输出格式

推理生成的 npz 文件与 seq_extract 输出格式完全兼容:

```python
data = np.load('output.npz', allow_pickle=True)
print(data.keys())
# ['strokes_data', 'init_cursors', 'image_size', 'round_length', 'init_width']

# strokes_data 形状: (seq_len, 7)
# 格式: [pen_state, x1, y1, x2, y2, r, s]
```

## 与二阶段集成

生成的 npz 文件可以直接用于 rl_finetune 阶段:

```bash
# 1. 用轻量模型生成 npz
python lightweight/inference.py \
    --checkpoint lightweight/output/model_best.pth \
    --input calligraphy_images/ \
    --output_dir rl_finetune/data/train_data/

# 2. 运行二阶段强化学习训练
cd rl_finetune
python try_tianshou.py \
    --train_data_dir ./data/train_data/ \
    --test_data_dir ./data/test_data/ \
    --which_tool brush
```

## 模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 256 | 模型维度 |
| nhead | 4 | Transformer 注意力头数 |
| num_decoder_layers | 3 | 解码器层数 |
| max_seq_len | 100 | 最大序列长度 |
| image_size | 256 | 输入图像尺寸 |

## 模型大小

- 总参数量: ~15M (ResNet-18: ~11M + Transformer: ~4M)
- 推理速度: ~10ms/张 (GPU)

## 对比 seq_extract

| 特性 | seq_extract (原) | lightweight (新) |
|------|-----------------|------------------|
| 架构 | LSTM + Diff Renderer | ResNet-18 + Transformer |
| 参数量 | ~50M | ~15M |
| 训练数据 | QuickDraw + 书法 | QuickDraw + 书法 |
| 训练时间 | ~5天 | ~1天 |
| 输出格式 | npz (7维) | npz (7维, 兼容) |

## 训练技巧

1. **Teacher Forcing**: 训练时使用 `teacher_forcing_ratio=0.5` 平衡稳定性和泛化
2. **损失函数**: 坐标损失权重更高 (weight_coord=5.0)
3. **数据增强**: 可以在 dataset.py 中添加图像增强

## 常见问题

**Q: 如何获取 QuickDraw 数据?**

A: 访问 https://quickdraw.withgoogle.com/data 下载 npz 文件，或使用 `gsutil` 从 Google Cloud Storage 下载。

**Q: 生成的笔画质量不好怎么办?**

A: 1) 增加训练数据 2) 调整模型大小 3) 延长训练时间 4) 使用书法数据微调

**Q: 可以只训练部分层吗?**

A: 可以，修改 train.py 冻结 ResNet 部分层，只训练 Transformer 解码器。

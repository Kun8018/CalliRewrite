# seq_extract_modern

CalliRewrite 项目的现代化 seq_extract 模块，基于 PyTorch 2.x 和 Transformer 架构。

## 项目结构

```
seq_extract_modern/
├── configs/              # 配置文件
│   └── model_config.py   # 模型和训练配置
├── models/               # 模型定义
│   └── vit_transformer.py # ViT + Transformer 架构
├── data/                 # 数据处理
│   ├── datasets.py       # 数据集定义
│   └── transforms.py     # 数据增强
├── trainer/              # 训练器
│   └── training_module.py # PyTorch Lightning 实现
├── renderer/             # 神经渲染
│   └── neural_renderer.py # 现代化渲染器
├── inference/            # 推理代码
│   └── predictor.py      # 预测器
├── scripts/              # 运行脚本
│   ├── train.py          # 训练入口
│   └── test.py           # 测试入口
├── requirements.txt      # 依赖列表
└── README.md            # 说明文档
```

## 快速开始

### 1. 安装依赖

```bash
cd seq_extract_modern
conda create -n calli_modern python=3.10
conda activate calli_modern
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python scripts/train.py \
    --train_data /path/to/training/images \
    --val_data /path/to/validation/images \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --gpus 1
```

### 3. 测试模型

```bash
python scripts/test.py \
    --input /path/to/image/or/directory \
    --model /path/to/checkpoint.ckpt \
    --output outputs \
    --visualize \
    --num_strokes 100
```

## 使用示例

### Python API

```python
from inference.predictor import Predictor
from configs.model_config import get_default_config

# 加载配置
config = get_default_config()

# 创建预测器
predictor = Predictor(
    model_path='path/to/checkpoint.ckpt',
    config=config,
    device='cuda'  # 或 'cpu'
)

# 预测
result = predictor.predict('path/to/image.png', num_strokes=100)

# 保存结果
predictor.save_result(result, 'output/strokes.npy')

# 可视化
predictor.visualize_result(result, 'output/strokes.png')
```

### 与原项目集成

输出的 `.npy` 文件格式与原 seq_extract 模块兼容，可以直接用于后续的强化学习精调模块：

```python
# 与原项目 rl_finetune 模块集成
import numpy as np

# 加载新模块输出的笔画
strokes = np.load('output/strokes.npy')

# 直接传递给 rl_finetune
# ... 使用原项目的代码 ...
```

## 主要改进

相比原始 TensorFlow 1.x 实现，这个现代化版本提供：

1. **技术栈更新**
   - PyTorch 2.x 替代 TensorFlow 1.x
   - ViT + Transformer 架构替代 LSTM
   - 支持混合精度训练

2. **代码质量**
   - 更简洁、可维护的代码
   - 模块化设计
   - 完整的类型提示

3. **训练效率**
   - PyTorch Lightning 简化训练流程
   - 自动分布式训练支持
   - 更好的日志和可视化

4. **推理优化**
   - 简化的推理 API
   - 支持 ONNX 导出
   - 更好的性能优化

## 与原项目对比

| 特性 | 原 seq_extract | seq_extract_modern |
|-----|--------------|-------------------|
| 框架 | TensorFlow 1.x | PyTorch 2.x |
| 模型架构 | LSTM | ViT + Transformer |
| 训练框架 | 自定义 | PyTorch Lightning |
| 混合精度 | ❌ | ✅ |
| 代码可维护性 | 中等 | 高 |
| 社区支持 | 下降 | 活跃 |

## 迁移指南

要从原项目迁移，你可以：

1. **保持输出兼容**：新模块输出的 `.npy` 格式与原模块完全兼容
2. **渐进式替换**：先在开发环境测试，再逐步替换生产环境
3. **重训练**：使用新架构需要重新训练模型

## 贡献指南

欢迎贡献！请确保：

1. 遵循 PEP 8 代码风格
2. 添加类型提示
3. 编写单元测试
4. 更新文档

## 许可证

与 CalliRewrite 项目保持一致。

## 引用

如果你使用这个代码，请引用原 CalliRewrite 论文。
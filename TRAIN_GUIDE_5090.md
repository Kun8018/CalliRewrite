# CalliRewrite 5090 显卡训练指南

## 🚀 快速开始训练

### 1. 激活 GPU 虚拟环境

```bash
cd /Users/kun/CalliRewrite
source calli_train_env_gpu/bin/activate
```

### 2. 验证 CUDA 和 PyTorch

```bash
python3 -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

应该会看到：
```
PyTorch 版本: 2.x.x
CUDA 可用: True
GPU: NVIDIA GeForce RTX 5090
```

### 3. 运行训练

#### 方法 A：使用简化脚本（推荐）

```bash
cd /Users/kun/CalliRewrite
chmod +x train_simple.sh
./train_simple.sh
```

#### 方法 B：自定义参数训练

```bash
cd /Users/kun/CalliRewrite/seq_extract_modern
source calli_train_env_gpu/bin/activate

python scripts/train.py \
    --train_data ../dataset/train \
    --val_data ../dataset/val \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 200 \
    --gpus 1 \
    --save_dir ../outputs \
    --project_name calli_extract_5090
```

## 📊 训练参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--train_data` | 训练数据目录 | `../dataset/train` |
| `--val_data` | 验证数据目录 | `../dataset/val` |
| `--batch_size` | 批次大小 | 32-64（5090 可以用 64）|
| `--lr` | 学习率 | 1e-4 |
| `--max_epochs` | 最大训练轮数 | 200 |
| `--gpus` | GPU 数量 | 1 |
| `--save_dir` | 保存目录 | `../outputs` |
| `--project_name` | 项目名称 | `calli_extract_5090` |

## 🎯 训练过程监控

训练会在以下位置生成：

1. **TensorBoard 日志**：`outputs/calli_extract_5090/lightning_logs/`
   ```bash
   tensorboard --logdir outputs/calli_extract_5090/lightning_logs/
   ```

2. **模型检查点**：`outputs/checkpoints/best_model.ckpt`

3. **示例图像**：每个 epoch 都会保存验证图像

## 📈 预期训练效果

- **初期（1-50 epochs）**：模型学习基本笔画识别
- **中期（50-150 epochs）**：笔画提取质量显著提升
- **后期（150-200 epochs）**：细节优化，PSNR 稳定

## 🔧 常见问题

### Q1: CUDA 显存不足？

**A**: 减小 `batch_size` 到 16 或 8：
```bash
--batch_size 16
```

### Q2: 训练速度慢？

**A**: 5090 应该很快，如果慢检查：
- 确认使用了 CUDA 版本的 PyTorch
- 检查数据加载是否是瓶颈

### Q3: 如何恢复训练？

**A**: 使用 `--resume_from_checkpoint` 参数：
```bash
--resume_from_checkpoint ../outputs/checkpoints/last.ckpt
```

## 🎨 训练后使用

训练完成后，使用模型进行推理和仿真：

```bash
# 测试训练好的模型
python seq_extract_modern/scripts/test.py \
    --input ../dataset/val \
    --model ../outputs/checkpoints/best_model.ckpt \
    --output ../outputs/inference

# 转换为仿真格式
python seq_extract_modern_to_simulation.py \
    --input seq_extract/sample_inputs/clean_line_drawings/elephant.png \
    --output outputs/elephant_simulation.npz

# 运行仿真
cd mujoco_sim
python mujoco_simulator.py ../outputs/elephant_simulation.npz --speed 0.05
```
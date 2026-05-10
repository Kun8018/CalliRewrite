#!/usr/bin/env python3
"""
检查 GPU 环境
"""
import sys
import os

print("=== 检查 Python 环境 ===")
print(f"Python 版本: {sys.version}")

print("\n=== 检查 PyTorch ===")
try:
    import torch
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 设备数: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError as e:
    print(f"PyTorch 未安装: {e}")

print("\n=== 检查训练模块 ===")
try:
    sys.path.insert(0, '/Users/kun/CalliRewrite/seq_extract_modern')
    from configs.model_config import get_default_config
    print("✅ 配置模块加载成功")

    config = get_default_config()
    print(f"✅ 配置创建成功")

except Exception as e:
    print(f"❌ 模块加载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 检查数据目录 ===")
data_dir = '/Users/kun/CalliRewrite/dataset/train'
if os.path.exists(data_dir):
    files = os.listdir(data_dir)
    print(f"✅ 训练数据目录存在，包含 {len(files)} 个文件")
else:
    print(f"❌ 训练数据目录不存在: {data_dir}")
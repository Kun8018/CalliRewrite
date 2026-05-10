#!/usr/bin/env python3
"""
检查旧版 CNN+LSTM 架构的输出数据格式
"""

import numpy as np
from pathlib import Path

# 检查 rl_finetune 数据目录
data_dir = Path("rl_finetune/data/train_data")

print("=" * 60)
print("检查旧版数据格式")
print("=" * 60)

if data_dir.exists():
    np_files = list(data_dir.glob("*.npy"))
    print(f"\n找到 {len(np_files)} 个 .npy 文件")

    if np_files:
        sample_file = np_files[0]
        print(f"\n读取示例文件: {sample_file}")

        data = np.load(sample_file)
        print(f"数据形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"数据范围: [{np.min(data)}, {np.max(data)}]")

        print(f"\n前 3 行数据:")
        print(data[:3])

        print(f"\n完整数据内容:")
        print(data)

print("\n" + "=" * 60)
print("检查 seq_extract 旧版代码中的数据格式定义")
print("=" * 60)

seq_extract_dir = Path("seq_extract")
if seq_extract_dir.exists():
    print("\n找到 seq_extract 目录，检查模型定义...")

    # 检查 rnn.py
    rnn_file = seq_extract_dir / "rnn.py"
    if rnn_file.exists():
        print(f"\n读取 {rnn_file}")
        with open(rnn_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 查找输出相关的内容
            import re
            output_matches = re.findall(r'(def.*output|output.*=|shape.*=|\.Dense|\.Linear)', content)
            if output_matches:
                print("找到输出相关代码片段:")
                print("\n".join(output_matches[:10]))

    # 检查 hyper_parameters.py
    hp_file = seq_extract_dir / "hyper_parameters.py"
    if hp_file.exists():
        print(f"\n读取 {hp_file}")
        with open(hp_file, 'r', encoding='utf-8') as f:
            hp_content = f.read()
            print("超参数内容:")
            print(hp_content)

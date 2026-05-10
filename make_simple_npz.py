#!/usr/bin/env python3
"""
创建一个简单的方形轨迹用于仿真演示
"""
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)

# 简单方形轨迹
x = []
y = []
z = []

paper_center_x = 0.25
paper_center_y = 0.25
square_size = 0.04  # 4cm
max_z = 0.01  # 笔抬起高度
min_z = -0.003  # 笔接触高度

# 从起点开始，抬起
x.append(paper_center_x - square_size/2 - 0.01)
y.append(paper_center_y - square_size/2)
z.append(max_z + 0.03)

# 画方形
# 左到右
x.append(paper_center_x - square_size/2)
y.append(paper_center_y - square_size/2)
z.append(min_z)

x.append(paper_center_x + square_size/2)
y.append(paper_center_y - square_size/2)
z.append(min_z)

# 抬起
x.append(paper_center_x + square_size/2)
y.append(paper_center_y - square_size/2)
z.append(max_z + 0.03)

# 右到左
x.append(paper_center_x + square_size/2)
y.append(paper_center_y + square_size/2)
z.append(min_z)

x.append(paper_center_x - square_size/2)
y.append(paper_center_y + square_size/2)
z.append(min_z)

# 抬起
x.append(paper_center_x - square_size/2)
y.append(paper_center_y + square_size/2)
z.append(max_z + 0.03)

# 保存
np.savez(
    "outputs/simple_square.npz",
    pos_3d_x=np.array(x),
    pos_3d_y=np.array(y),
    pos_3d_z=np.array(z)
)

print("✅ 创建完成: outputs/simple_square.npz")
print("\n现在可以运行仿真:")
print("  cd mujoco_sim")
print("  python mujoco_simulator.py ../outputs/simple_square.npz --speed 0.05")
import numpy as np

# 📥 加载你用 inference.py 预测出来的那个 npz 文件
data = np.load("./inference_ar_phase1_debug/0.npz", allow_pickle=True)

print("--- 📂 NPZ 预测数据核心核检 ---")
strokes_data = data['strokes_data']

# 🎯 核心指标一：检查动作开关 (pen_state)
pen_states = strokes_data[:, 0]
print(f"1. 🖋️ 笔画状态统计:")
print(f"   - 值为 1.0 (落笔) 的步数: {np.sum(pen_states == 1.0)}")
print(f"   - 值为 0.0 (抬笔) 的步数: {np.sum(pen_states == 0.0)}")

# 🎯 核心指标二：检查相对坐标的绝对数值范围
dx1 = strokes_data[:, 1]
dy1 = strokes_data[:, 2]
dx2 = strokes_data[:, 3]
dy2 = strokes_data[:, 4]

print(f"\n2. 🗺️ 相对坐标位移范围:")
print(f"   - dx1 (控制点1) 范围: [{dx1.min():.6f}, {dx1.max():.6f}]")
print(f"   - dy1 (控制点1) 范围: [{dy1.min():.6f}, {dy1.max():.6f}]")
print(f"   - dx2 (终点)    范围: [{dx2.min():.6f}, {dx2.max():.6f}]")
print(f"   - dy2 (终点)    范围: [{dy2.min():.6f}, {dy2.max():.6f}]")
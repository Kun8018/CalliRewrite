#!/usr/bin/env python3
"""
从书法图像提取密集轨迹 - 生成用于MuJoCo仿真的NPZ文件
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize
import sys
import os


def extract_skeleton_trajectory(image_path, output_npz,
                               char_size=0.12, pen_depth=-0.002):
    """
    从书法图像提取骨架轨迹

    Args:
        image_path: 输入图像路径
        output_npz: 输出NPZ路径
        char_size: 字符实际大小（米）
        pen_depth: 笔压深度（米，负值表示按压）
    """
    print("=" * 70)
    print("从书法图像提取密集轨迹")
    print("=" * 70)
    print(f"输入图像: {image_path}")
    print(f"字符大小: {char_size*100:.1f} cm")
    print(f"笔压深度: {abs(pen_depth)*1000:.1f} mm\n")

    # 1. 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    h, w = img.shape
    print(f"✅ 图像尺寸: {w}x{h}")

    # 2. 二值化（反转：黑字->白字）
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 3. 骨架提取
    print("骨架提取中...")
    skeleton = skeletonize(binary > 0)

    # 4. 找到所有骨架点
    y_coords, x_coords = np.where(skeleton)
    print(f"✅ 骨架点数: {len(x_coords)}")

    if len(x_coords) == 0:
        raise ValueError("未检测到骨架点，请检查图像")

    # 5. 使用连通组件分析分离笔画
    from skimage.measure import label
    labeled = label(skeleton, connectivity=2)
    num_strokes = labeled.max()
    print(f"✅ 检测到笔画数: {num_strokes}")

    # 6. 按笔画提取轨迹
    all_x, all_y, all_z = [], [], []

    for stroke_id in range(1, num_strokes + 1):
        # 提取当前笔画的点
        stroke_mask = (labeled == stroke_id)
        sy, sx = np.where(stroke_mask)

        if len(sx) < 3:  # 跳过太短的笔画（噪点）
            continue

        # 排序：从一端到另一端（简化：使用最近邻排序）
        points = np.column_stack([sx, sy])
        sorted_points = sort_stroke_points(points)

        stroke_x = sorted_points[:, 0]
        stroke_y = sorted_points[:, 1]

        # 归一化到[0, char_size]
        x_min, x_max = stroke_x.min(), stroke_x.max()
        y_min, y_max = stroke_y.min(), stroke_y.max()

        # 全局归一化（使用图像边界）
        x_norm = (stroke_x / w) * char_size
        y_norm = (stroke_y / h) * char_size

        # 每笔开始前抬笔
        if len(all_x) > 0:  # 不是第一笔
            all_x.append(all_x[-1])
            all_y.append(all_y[-1])
            all_z.append(0.05)  # 抬笔到5cm高

        # 移动到起点上方
        all_x.append(x_norm[0])
        all_y.append(y_norm[0])
        all_z.append(0.05)

        # 下笔
        all_x.append(x_norm[0])
        all_y.append(y_norm[0])
        all_z.append(pen_depth)

        # 书写笔画
        for i in range(len(stroke_x)):
            all_x.append(x_norm[i])
            all_y.append(y_norm[i])
            all_z.append(pen_depth)

        # 笔画结束
        print(f"  笔画 {stroke_id}: {len(stroke_x)} 个点")

    # 7. 最后抬笔
    all_x.append(all_x[-1])
    all_y.append(all_y[-1])
    all_z.append(0.05)

    # 8. 转为numpy数组
    pos_3d_x = np.array(all_x)
    pos_3d_y = np.array(all_y)
    pos_3d_z = np.array(all_z)

    print(f"\n✅ 总控制点数: {len(pos_3d_x)}")
    print(f"   X 范围: [{pos_3d_x.min():.4f}, {pos_3d_x.max():.4f}] m")
    print(f"   Y 范围: [{pos_3d_y.min():.4f}, {pos_3d_y.max():.4f}] m")
    print(f"   Z 范围: [{pos_3d_z.min():.4f}, {pos_3d_z.max():.4f}] m")

    # 9. 保存NPZ
    os.makedirs(os.path.dirname(output_npz) or '.', exist_ok=True)
    np.savez(output_npz,
             pos_3d_x=pos_3d_x,
             pos_3d_y=pos_3d_y,
             pos_3d_z=pos_3d_z)

    print(f"\n✅ NPZ文件已保存: {output_npz}")
    print("=" * 70)

    return output_npz


def sort_stroke_points(points):
    """
    使用最近邻排序笔画点（从一端到另一端）

    Args:
        points: (N, 2) 数组

    Returns:
        sorted_points: (N, 2) 排序后的数组
    """
    if len(points) == 0:
        return points

    # 找起点：选择最左上角的点
    start_idx = np.argmin(points[:, 0] + points[:, 1])

    sorted_points = [points[start_idx]]
    remaining = list(range(len(points)))
    remaining.remove(start_idx)

    current = points[start_idx]

    # 贪心最近邻
    while remaining:
        # 计算到所有剩余点的距离
        dists = np.linalg.norm(points[remaining] - current, axis=1)
        nearest_idx = np.argmin(dists)

        actual_idx = remaining[nearest_idx]
        sorted_points.append(points[actual_idx])
        current = points[actual_idx]
        remaining.pop(nearest_idx)

    return np.array(sorted_points)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从书法图像提取密集轨迹")
    parser.add_argument("image", type=str, help="输入图像路径")
    parser.add_argument("--output", "-o", type=str,
                       default="demo_outputs/dense_trajectory.npz",
                       help="输出NPZ路径")
    parser.add_argument("--size", type=float, default=0.12,
                       help="字符大小（米），默认12cm")
    parser.add_argument("--depth", type=float, default=-0.002,
                       help="笔压深度（米），默认-2mm")

    args = parser.parse_args()

    extract_skeleton_trajectory(args.image, args.output, args.size, args.depth)

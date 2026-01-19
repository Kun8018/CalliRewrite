#!/usr/bin/env python3
"""
通用轨迹转换和可视化工具
支持从RL输出(.npy)转换为机器人控制点(.npz)并生成轨迹可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import sys
import argparse

# ============================================================================
# 校准函数（从calibrate.py复制）
# ============================================================================

def func_brush_precalibrated(radii):
    """
    预校准的毛笔函数
    如需重新校准，请使用calibrate.py
    """
    if radii >= 0 and radii <= 7.72667536e-04:
        z = -5.9701493 * (radii - 7.72667536e-04) + 4.34974603e-03
    elif radii > 7.72667536e-04 and radii <= 1.78125854e-03:
        z = -1.538473 * (radii - 7.72667536e-04) + 4.34974603e-03
    elif radii > 1.78125854e-03 and radii <= 2.45866277e-03:
        z = -6.028019 * (radii - 1.78125854e-03) + 2.79805600e-03
    elif radii > 2.45866277e-03 and radii < 0.0045:
        z = -2.37843574 * (radii - 2.45866277e-03) - 1.28534957e-03
    else:
        return -0.006
    return z


# ============================================================================
# RL数据转NPZ
# ============================================================================

def convert_rl_to_npz(npy_path, output_npz_path,
                     calibration_func=func_brush_precalibrated,
                     alpha=0.04, beta=0.5, style_type=0):
    """
    将RL输出的.npy文件转换为机器人控制点.npz文件

    Args:
        npy_path: RL输出的.npy文件路径
        output_npz_path: 输出的.npz文件路径
        calibration_func: 校准函数（r -> z）
        alpha: 字符尺寸（米），例如0.04 = 4cm宽
        beta: 笔画宽度对比度调整（0.5 = 正常）
        style_type: 0=隶书（clerical），1=楷书（regular）

    Returns:
        x, y, z: 控制点列表（米）
    """
    # 加载RL状态
    data = np.load(npy_path)
    print(f"加载RL状态: {npy_path}")
    print(f"  数据形状: {data.shape}")
    print(f"  字符尺寸: {alpha*100} cm")
    print(f"  笔画宽度调整: {beta}")

    record_x = []
    record_y = []
    record_z = []

    for i in range(data.shape[0]):
        # 只使用前4列：p_t, x, y, r（忽略后面的颜色等信息）
        p_t, x, y, r = data[i, :4]

        # 归一化坐标（假设原始数据在0-255范围）
        # 归一化到0-1范围
        x = x / 255.0
        y = y / 255.0
        r = r / 255.0

        # 缩放到实际尺寸
        x_ = x * alpha
        y_ = y * alpha
        r_ = r * alpha * beta

        # 使用校准函数将r转换为z
        h = calibration_func(r_)
        h = h - 0.09  # 调整桌面高度

        if p_t == 0:
            # 正常笔画点
            record_x.append(x_)
            record_y.append(y_)
            record_z.append(h)

        elif p_t == 1:
            # 新笔画开始
            if i == data.shape[0] - 1:
                continue  # 如果是最后一个点则跳过

            if style_type == 1:
                # 楷书：从左上方进入
                record_x.append(x_ - 2 * r_)
                record_y.append(y_ - 2 * r_)
                record_z.append(0.05)  # 抬笔
            else:
                # 隶书：从上方进入
                record_x.append(x_)
                record_y.append(y_)
                record_z.append(0.05)

            # 添加当前点（落笔）
            record_x.append(x_)
            record_y.append(y_)
            record_z.append(h)

            if style_type == 0:
                # 隶书：笔画起始处的回锋
                nxt_vec = data[i + 1, 1:3] / 255.0 - np.array([x, y])  # 归一化后计算方向
                vec_norm = np.linalg.norm(nxt_vec)
                if vec_norm > 0:
                    nxt_vec = nxt_vec / vec_norm
                    record_x.append(x_ - 2 * r_ * nxt_vec[0])
                    record_y.append(y_ - 2 * r_ * nxt_vec[1])
                    record_z.append(h)
                record_x.append(x_)
                record_y.append(y_)
                record_z.append(h)

    # 结束时逐渐抬笔
    for _ in range(5):
        record_x.append(record_x[-1])
        record_y.append(record_y[-1])
        record_z.append(record_z[-1] + 0.015)

    # 保存为npz
    np.savez(output_npz_path,
             pos_3d_x=record_x,
             pos_3d_y=record_y,
             pos_3d_z=record_z)

    print(f"✅ 已转换 {len(record_x)} 个控制点")
    print(f"  X范围: [{np.min(record_x)*1000:.1f}, {np.max(record_x)*1000:.1f}] mm")
    print(f"  Y范围: [{np.min(record_y)*1000:.1f}, {np.max(record_y)*1000:.1f}] mm")
    print(f"  Z范围: [{np.min(record_z)*1000:.1f}, {np.max(record_z)*1000:.1f}] mm")
    print(f"✅ 已保存到: {output_npz_path}")

    return record_x, record_y, record_z


# ============================================================================
# NPZ可视化
# ============================================================================

def visualize_npz_trajectory(npz_file, output_image):
    """可视化NPZ文件中的轨迹"""

    # 加载数据
    data = np.load(npz_file)
    x = data['pos_3d_x']
    y = data['pos_3d_y']
    z = data['pos_3d_z']

    print(f"\n可视化轨迹: {npz_file}")
    print(f"  控制点数: {len(x)}")
    print(f"  X范围: [{x.min()*1000:.1f}, {x.max()*1000:.1f}] mm")
    print(f"  Y范围: [{y.min()*1000:.1f}, {y.max()*1000:.1f}] mm")
    print(f"  Z范围: [{z.min()*1000:.1f}, {z.max()*1000:.1f}] mm")

    # 创建图形
    fig = plt.figure(figsize=(12, 10))

    # 1. XY平面轨迹（俯视图）
    ax1 = plt.subplot(2, 2, 1)
    # 根据Z值判断是否接触（Z < 0表示接触纸面）
    contact_mask = z < 0

    # 画非接触部分（虚线）
    if np.any(~contact_mask):
        ax1.plot(x[~contact_mask]*1000, y[~contact_mask]*1000, 'gray',
                linestyle='--', linewidth=1, alpha=0.5, label='Pen Up')

    # 画接触部分（实线）
    if np.any(contact_mask):
        ax1.plot(x[contact_mask]*1000, y[contact_mask]*1000, 'black',
                linewidth=2, label='Writing')

    # 标记起点和终点
    ax1.plot(x[0]*1000, y[0]*1000, 'go', markersize=8, label='Start')
    ax1.plot(x[-1]*1000, y[-1]*1000, 'ro', markersize=8, label='End')

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title('XY Plane (Top View)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    ax1.invert_yaxis()  # Y轴反转，使其与纸面方向一致

    # 2. XZ平面轨迹（侧视图）
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(x*1000, z*1000, 'b-', linewidth=1.5)
    ax2.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Paper')
    ax2.plot(x[0]*1000, z[0]*1000, 'go', markersize=8, label='Start')
    ax2.plot(x[-1]*1000, z[-1]*1000, 'ro', markersize=8, label='End')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.set_title('XZ Plane (Side View)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. YZ平面轨迹（前视图）
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(y*1000, z*1000, 'r-', linewidth=1.5)
    ax3.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Paper')
    ax3.plot(y[0]*1000, z[0]*1000, 'go', markersize=8, label='Start')
    ax3.plot(y[-1]*1000, z[-1]*1000, 'ro', markersize=8, label='End')
    ax3.set_xlabel('Y (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('YZ Plane (Front View)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Z坐标随时间变化
    ax4 = plt.subplot(2, 2, 4)
    time_points = np.arange(len(z))
    ax4.plot(time_points, z*1000, 'purple', linewidth=1.5)
    ax4.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Paper')
    ax4.fill_between(time_points, z*1000, 0, where=(z<0), alpha=0.3, color='blue', label='Contact')
    ax4.set_xlabel('Control Point Index')
    ax4.set_ylabel('Z (mm)')
    ax4.set_title('Pen Height Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"✅ 轨迹可视化已保存: {output_image}")
    plt.close()

    # 统计信息
    contact_points = np.sum(contact_mask)
    print(f"\n统计信息:")
    print(f"  接触点数: {contact_points}/{len(z)} ({contact_points/len(z)*100:.1f}%)")
    print(f"  轨迹长度: {np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))*1000:.1f} mm")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='将RL输出转换为NPZ并生成轨迹可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从npy文件转换并可视化
  python convert_and_visualize.py input.npy

  # 指定输出路径
  python convert_and_visualize.py input.npy -o output.npz --viz output_viz.png

  # 自定义参数
  python convert_and_visualize.py input.npy --alpha 0.05 --beta 0.7

  # 只可视化已有的npz文件
  python convert_and_visualize.py --npz existing.npz --viz output.png
        """
    )

    parser.add_argument('input', nargs='?', help='输入的.npy文件路径')
    parser.add_argument('-o', '--output', help='输出的.npz文件路径（默认：与输入同名）')
    parser.add_argument('--viz', '--visualize', dest='viz_output',
                       help='可视化输出图片路径（默认：npz文件名_trajectory.png）')
    parser.add_argument('--npz', help='直接可视化已有的npz文件（跳过转换步骤）')
    parser.add_argument('--alpha', type=float, default=0.04,
                       help='字符尺寸（米），默认0.04（4cm）')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='笔画宽度调整，默认0.5')
    parser.add_argument('--style', type=int, default=0, choices=[0, 1],
                       help='书法风格：0=隶书（默认），1=楷书')
    parser.add_argument('--no-viz', action='store_true',
                       help='不生成可视化图片')

    args = parser.parse_args()

    # 模式1：只可视化已有的npz文件
    if args.npz:
        npz_path = args.npz
        viz_path = args.viz_output or npz_path.replace('.npz', '_trajectory.png')

        print("=" * 70)
        print("可视化NPZ轨迹")
        print("=" * 70)
        visualize_npz_trajectory(npz_path, viz_path)
        print("=" * 70)
        return

    # 模式2：从npy转换
    if not args.input:
        parser.print_help()
        return

    npy_path = args.input

    if not os.path.exists(npy_path):
        print(f"❌ 错误：找不到输入文件 {npy_path}")
        return

    # 确定输出路径
    if args.output:
        npz_path = args.output
    else:
        # 默认：替换扩展名为.npz
        base_name = os.path.splitext(npy_path)[0]
        npz_path = base_name + '.npz'

    # 确定可视化输出路径
    if args.viz_output:
        viz_path = args.viz_output
    else:
        viz_path = npz_path.replace('.npz', '_trajectory.png')

    print("=" * 70)
    print("RL轨迹转换和可视化")
    print("=" * 70)
    print(f"输入文件: {npy_path}")
    print(f"输出NPZ: {npz_path}")
    if not args.no_viz:
        print(f"可视化图: {viz_path}")
    print("=" * 70)
    print()

    # 转换
    try:
        convert_rl_to_npz(
            npy_path,
            npz_path,
            alpha=args.alpha,
            beta=args.beta,
            style_type=args.style
        )
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 可视化
    if not args.no_viz:
        print()
        try:
            visualize_npz_trajectory(npz_path, viz_path)
        except Exception as e:
            print(f"❌ 可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return

    print()
    print("=" * 70)
    print("✅ 完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

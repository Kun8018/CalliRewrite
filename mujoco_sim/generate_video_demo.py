#!/usr/bin/env python3
"""
生成视频演示 - 从俯视角度录制书法轨迹
"""

import sys
import os
import numpy as np
import mujoco
import imageio

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mujoco_simulator import FrankaCalligraphySimulator

def generate_video(npz_path, output_video, speed=0.05, fps=30):
    """
    生成书法轨迹视频（俯视角度）

    Args:
        npz_path: NPZ文件路径
        output_video: 输出视频路径
        speed: 运动速度 (m/s)
        fps: 视频帧率
    """
    print("=" * 70)
    print("MuJoCo 书法仿真 - 视频录制")
    print("=" * 70)
    print(f"\n输入文件: {npz_path}")
    print(f"输出视频: {output_video}")
    print(f"速度: {speed} m/s")
    print(f"帧率: {fps} FPS\n")

    # 加载 NPZ 数据
    data = np.load(npz_path)
    x = data['pos_3d_x']
    y = data['pos_3d_y']
    z = data['pos_3d_z']

    num_points = len(x)
    print(f"✅ 加载轨迹数据:")
    print(f"   控制点数: {num_points}")
    print(f"   X 范围: [{x.min():.4f}, {x.max():.4f}] m")
    print(f"   Y 范围: [{y.min():.4f}, {y.max():.4f}] m")
    print(f"   Z 范围: [{z.min():.4f}, {z.max():.4f}] m\n")

    # 创建仿真器（离屏模式）
    print("初始化 MuJoCo 仿真器...")
    sim = FrankaCalligraphySimulator(render_mode="rgb_array")

    # 创建离屏渲染器
    print("创建离屏渲染器 (1280x720)...")
    renderer = mujoco.Renderer(sim.model, height=720, width=1280)

    # 获取俯视相机ID
    camera_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_view")
    print(f"✅ 使用相机: top_view (ID={camera_id})\n")

    # 初始化视频录制
    frames = []
    print("开始执行轨迹并录制视频...\n")

    # 执行轨迹并录制
    for i in range(num_points):
        # NPZ坐标是相对于纸张的，需要转换到世界坐标系
        # 纸张中心在世界坐标 (0.5, 0, 0.01)
        target_pos = np.array([
            x[i] + sim.paper_offset[0],  # 加上X偏移
            y[i] + sim.paper_offset[1],  # 加上Y偏移
            z[i]  # Z坐标保持不变
        ])

        # 移动到目标位置
        sim.move_to_position(target_pos, speed=speed, wait_time=0.0)

        # 渲染当前帧（俯视角度）
        renderer.update_scene(sim.data, camera=camera_id)
        pixels = renderer.render()
        frames.append(pixels)

        # 显示进度
        if i % 10 == 0 or i == num_points - 1:
            progress = (i + 1) / num_points * 100
            print(f"  进度: {i+1}/{num_points} ({progress:.1f}%)")

    print(f"\n✅ 轨迹执行完成!")
    print(f"   总帧数: {len(frames)}")
    print(f"   视频时长: {len(frames)/fps:.2f} 秒")

    # 检查笔迹统计
    if len(sim.ink_traces) > 0:
        contact_points = sum(1 for _, _, _, c in sim.ink_traces if c)
        print(f"   记录点数: {len(sim.ink_traces)}")
        print(f"   接触点数: {contact_points}")
        print(f"   接触率: {contact_points/len(sim.ink_traces)*100:.1f}%")

    # 保存视频
    print(f"\n保存视频到: {output_video}")
    os.makedirs(os.path.dirname(output_video) if os.path.dirname(output_video) else ".", exist_ok=True)
    imageio.mimsave(output_video, frames, fps=fps)

    # 获取视频文件大小
    video_size_kb = os.path.getsize(output_video) / 1024
    video_size_mb = video_size_kb / 1024

    if video_size_mb >= 1:
        print(f"✅ 视频已保存! 大小: {video_size_mb:.2f} MB")
    else:
        print(f"✅ 视频已保存! 大小: {video_size_kb:.1f} KB")

    # 保存画布图像
    import cv2
    canvas_path = output_video.replace('.mp4', '_canvas.png')
    cv2.imwrite(canvas_path, sim.paper_canvas)
    canvas_size = os.path.getsize(canvas_path) / 1024
    print(f"✅ 画布已保存: {canvas_path} ({canvas_size:.1f} KB)")

    # 清理
    sim.close()

    print("\n" + "=" * 70)
    print("视频生成完成!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成书法轨迹视频（俯视角度）")
    parser.add_argument("npz_file", type=str, help="NPZ 轨迹文件路径")
    parser.add_argument("--output", "-o", type=str, default="outputs/calligraphy_demo.mp4",
                        help="输出视频路径 (默认: outputs/calligraphy_demo.mp4)")
    parser.add_argument("--speed", "-s", type=float, default=0.05,
                        help="运动速度 (m/s, 默认: 0.05)")
    parser.add_argument("--fps", "-f", type=int, default=30,
                        help="视频帧率 (默认: 30)")

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.npz_file):
        print(f"❌ 错误: 文件不存在: {args.npz_file}")
        sys.exit(1)

    # 生成视频
    generate_video(args.npz_file, args.output, args.speed, args.fps)

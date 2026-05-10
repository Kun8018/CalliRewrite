#!/usr/bin/env python3
"""
桥接脚本：将 seq_extract_modern 的输出转换为仿真系统可用的格式

使用流程：
1. 从书法图像提取笔画（使用新的 seq_extract_modern）
2. 转换为仿真可用的 .npz 格式
3. 直接在 mujoco_sim 中运行

Example:
    python seq_extract_modern_to_simulation.py \
        --input seq_extract/sample_inputs/永.png \
        --output outputs/simulation_永.npz
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent / "seq_extract_modern"))


def convert_strokes_to_3d(
    strokes: np.ndarray,
    character_size: float = 0.04,  # 4cm
    beta: float = 0.5,  # 笔画宽度调整
    paper_offset: tuple = (0.25, 0.25),  # 纸张中心位置 (米)
    max_z: float = 0.01,  # 笔抬起高度
    min_z: float = -0.006,  # 最大压力高度
) -> tuple:
    """
    将新的 seq_extract_modern 输出的笔画参数转换为 3D 坐标

    Args:
        strokes: (N, 7) 数组 - [x1, y1, x2, y2, width, pressure, eos]
                 其中 x, y 在 [-1, 1] 范围内
        character_size: 字符的真实大小（米）
        beta: 笔画宽度调整系数
        paper_offset: 纸张在机器人坐标系中的中心位置
        max_z: 笔抬起高度
        min_z: 最大压力高度

    Returns:
        pos_3d_x, pos_3d_y, pos_3d_z: 3D 坐标列表
    """
    record_x = []
    record_y = []
    record_z = []

    # 简化的笔画宽度到高度的映射（暂用，实际需要校准）
    def r_to_z(radius: float) -> float:
        """简化的 r->z 映射函数"""
        if radius <= 0.001:
            return max_z - 0.001
        elif radius <= 0.003:
            return max_z - (radius / 0.003) * (max_z - min_z)
        else:
            return min_z

    for i in range(len(strokes)):
        x1, y1, x2, y2, width, pressure, eos = strokes[i]

        # 检查是否是结束笔画
        if eos > 0.5:
            # 抬起笔
            if len(record_x) > 0:
                record_x.append(record_x[-1])
                record_y.append(record_y[-1])
                record_z.append(max_z)
            continue

        # 转换坐标：从 [-1, 1] 到 [0, character_size]
        # 注意：原始 seq_extract 中的坐标是 [0, 1]，但新的是 [-1, 1]
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0

        # 归一化到 [0, 1]，然后乘以字符大小
        x_norm = (x_center + 1.0) / 2.0
        y_norm = (y_center + 1.0) / 2.0

        # 真实世界坐标
        x_real = x_norm * character_size - character_size / 2.0 + paper_offset[0]
        y_real = y_norm * character_size - character_size / 2.0 + paper_offset[1]

        # 从宽度计算 z 高度
        radius_meters = width * character_size * beta
        z_real = r_to_z(radius_meters)

        # 添加起点（如果是新笔画）
        if i == 0 or strokes[i-1, 6] > 0.5:
            # 先抬起笔
            record_x.append(x_real)
            record_y.append(y_real)
            record_z.append(max_z + 0.03)

            # 然后放下笔
            record_x.append(x_real)
            record_y.append(y_real)
            record_z.append(z_real)
        else:
            # 继续当前笔画
            record_x.append(x_real)
            record_y.append(y_real)
            record_z.append(z_real)

    # 最后抬起笔
    if len(record_x) > 0:
        record_x.append(record_x[-1])
        record_y.append(record_y[-1])
        record_z.append(max_z + 0.03)

    return np.array(record_x), np.array(record_y), np.array(record_z)


def convert_old_style_to_3d(
    npy_path: str,
    output_path: str,
    alpha: float = 0.04,
    beta: float = 0.5,
):
    """
    兼容原始 seq_extract 的格式（N, 7）数组，但用新的方式处理

    原始格式：[eos, x1, y1, x2, y2, width, scaling] 或类似

    这是一个简化版本，实际应该使用 calibrate.py 中的完整功能
    """
    data = np.load(npy_path)
    print(f"Loaded strokes: {data.shape}")

    # 这里使用简化的转换
    # 实际生产中应该使用 callibrate.py 的 convert_rl_to_npz() 函数
    # 这里我们只是做一个简单的演示转换

    record_x = []
    record_y = []
    record_z = []

    # 简化的 r->z 映射
    def simple_r_to_z(r):
        max_z = 0.01
        min_z = -0.006
        return max_z - r * (max_z - min_z)

    paper_offset_x = 0.25
    paper_offset_y = 0.25

    for i in range(data.shape[0]):
        # 尝试解析不同的数据格式
        if data.shape[1] == 4:
            # 旧格式: [p_t, x, y, r]
            p_t, x, y, r = data[i]
            eos = p_t
            x_norm = x
            y_norm = y
            width = r
        elif data.shape[1] == 7:
            # 新格式: [eos, x1, y1, x2, y2, width, scaling]
            eos = data[i, 0]
            x_norm = (data[i, 1] + data[i, 3]) / 2.0
            y_norm = (data[i, 2] + data[i, 4]) / 2.0
            width = data[i, 5]
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        x_real = x_norm * alpha - alpha / 2.0 + paper_offset_x
        y_real = y_norm * alpha - alpha / 2.0 + paper_offset_y
        z_real = simple_r_to_z(width * beta)

        if eos == 0:
            record_x.append(x_real)
            record_y.append(y_real)
            record_z.append(z_real)
        else:
            if i < data.shape[0] - 1:
                record_x.append(x_real)
                record_y.append(y_real)
                record_z.append(0.05)

    np.savez(
        output_path,
        pos_3d_x=np.array(record_x),
        pos_3d_y=np.array(record_y),
        pos_3d_z=np.array(record_z)
    )
    print(f"Saved to: {output_path}")

    return record_x, record_y, record_z


def main():
    parser = argparse.ArgumentParser(description="将 seq_extract_modern 输出转换为仿真格式")

    parser.add_argument("--input", "-i", type=str, required=True,
                        help="输入图像或 .npy 文件路径")
    parser.add_argument("--output", "-o", type=str, default="outputs/simulation.npz",
                        help="输出 .npz 文件路径")
    parser.add_argument("--size", "-s", type=float, default=0.04,
                        help="字符大小（米），默认 0.04m = 4cm")
    parser.add_argument("--beta", "-b", type=float, default=0.5,
                        help="笔画宽度调整系数")
    parser.add_argument("--use_old_format", action="store_true",
                        help="输入是原始 seq_extract 的 .npy 格式")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备: auto, cpu, cuda")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # 判断输入类型
    input_ext = os.path.splitext(args.input)[1].lower()

    if input_ext in ['.png', '.jpg', '.jpeg']:
        print("从图像提取笔画...")

        # 配置设备
        if args.device == "auto":
            device = "cuda" if sys.platform != "darwin" and torch.cuda.is_available() else "cpu"
        else:
            device = args.device

        # 导入新模块
        import torch
        from configs.model_config import get_default_config
        from inference.predictor import Predictor

        config = get_default_config()
        predictor = Predictor(model_path=None, config=config, device=device)

        # 提取笔画
        result = predictor.predict(args.input, num_strokes=100)
        strokes = result["stroke_params"]
        print(f"提取到 {len(strokes)} 个笔画点")

        # 转换为 3D 坐标
        print("转换为 3D 坐标...")
        pos_x, pos_y, pos_z = convert_strokes_to_3d(
            strokes,
            character_size=args.size,
            beta=args.beta
        )

    elif input_ext == ".npy":
        print("从 .npy 文件加载笔画...")

        if args.use_old_format:
            # 使用旧格式处理
            pos_x, pos_y, pos_z = convert_old_style_to_3d(
                args.input,
                args.output,
                alpha=args.size,
                beta=args.beta
            )
            return  # 已保存文件
        else:
            # 加载新的格式
            strokes = np.load(args.input)
            print(f"加载到 {len(strokes)} 个笔画点")

            # 转换为 3D 坐标
            pos_x, pos_y, pos_z = convert_strokes_to_3d(
                strokes,
                character_size=args.size,
                beta=args.beta
            )
    else:
        raise ValueError(f"不支持的输入格式: {input_ext}")

    # 保存 .npz 文件
    np.savez(
        args.output,
        pos_3d_x=np.array(pos_x),
        pos_3d_y=np.array(pos_y),
        pos_3d_z=np.array(pos_z)
    )

    print(f"\n{'='*60}")
    print(f"✅ 转换完成！")
    print(f"   输出文件: {args.output}")
    print(f"   控制点数量: {len(pos_x)}")
    print(f"\n现在可以在 mujoco_sim 中运行:")
    print(f"   cd mujoco_sim")
    print(f"   python mujoco_simulator.py ../{args.output} --speed 0.05")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import torch
    main()
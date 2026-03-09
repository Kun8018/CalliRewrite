#!/usr/bin/env python3
"""
CalliRewrite + LingBot-VLA 端到端书法流水线

集成 LingBot-VLA 上层规划器与 CalliRewrite 执行系统，实现自然语言驱动的书法机器人控制。

流程:
    用户指令 → 解析汉字 → LingBot-VLA 场景规划 → 生成 NPZ → 仿真/执行

用法:
    # 仿真（使用 seq_extract 精确数据）
    python -m lingbot_planner.pipeline "写一个昨字" --simulate \\
        --data-dir ../seq_extract/outputs/__new_train_phase_2

    # 仿真（直接指定字符图像，快速模式）
    python -m lingbot_planner.pipeline "写龙" --image dragon.png --simulate

    # 真实机器人执行
    python -m lingbot_planner.pipeline "写龙" --image dragon.png \\
        --execute --robot-ip 172.16.0.2

    # 启用 LingBot-VLA 服务器（需在 GPU 机器单独启动）
    python -m lingbot_planner.pipeline "写昨" --simulate \\
        --planner-url http://gpu-server:8080 --camera workspace.jpg
"""

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np

# 项目路径
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
_MUJOCO_SIM_DIR = os.path.join(_ROOT_DIR, "mujoco_sim")
_CALLIBRATE_DIR = os.path.join(_ROOT_DIR, "callibrate")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_calligraphy_pipeline(
    instruction: str,
    camera_image_path: Optional[str] = None,
    image_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    robot_ip: str = "172.16.0.2",
    planner_url: str = "http://localhost:8080",
    output_dir: str = "lingbot_outputs",
    simulate: bool = True,
    execute: bool = False,
    speed: float = 0.05,
    char_size: float = 0.12,
    pen_depth: float = -0.002,
) -> dict:
    """
    端到端书法流水线，集成 LingBot-VLA 上层规划。

    执行步骤：
    1. 解析汉字 — 从自然语言指令提取目标汉字
    2. 场景规划 — LingBot-VLA 理解工作区，获取纸张位置和预备动作
    3. 生成 NPZ  — 精确笔画提取（order+seq_data）或快速骨架提取（图像）
    4. 执行      — MuJoCo 仿真 或 Franka FR3 真实执行

    Args:
        instruction:        自然语言书写指令，如 "写一个龙字"
        camera_image_path:  工作区相机图像路径（供 LingBot-VLA 场景理解，可选）
        image_path:         直接指定字符图像路径（绕过 data_dir 查找，可选）
        data_dir:           seq_extract 输出根目录（含 order/ 和 seq_data/，可选）
        robot_ip:           Franka 机器人 IP 地址
        planner_url:        LingBot-VLA 推理服务器 URL
        output_dir:         输出文件目录
        simulate:           是否在 MuJoCo 中仿真并保存画布图像
        execute:            是否在真实机器人上执行
        speed:              运动速度系数（0.0~1.0）
        char_size:          字符实际大小（米，默认 0.12 m = 12 cm）
        pen_depth:          笔压深度（米，负值表示按压，默认 -0.002 m）

    Returns:
        dict，包含：
        - char (str):          解析出的目标汉字
        - plan (dict):         LingBot-VLA 规划结果
        - npz_path (str):      生成的 NPZ 轨迹文件路径
        - canvas_path (str):   画布图像路径（仿真时）

    Raises:
        ValueError:       指令无法解析，或缺少必要参数
        FileNotFoundError: 找不到字符数据
    """
    from .planner import LingBotVLAPlanner
    from .character_selector import CharacterSelector

    os.makedirs(output_dir, exist_ok=True)
    result = {}

    # ── 步骤 1：解析汉字 ─────────────────────────────────────────────
    selector = CharacterSelector()
    char = selector.parse_character(instruction)
    if char is None:
        raise ValueError(f"无法从指令中解析汉字: '{instruction}'")
    logger.info(f"[1/4] 解析汉字: '{char}'")
    result["char"] = char

    # ── 步骤 2：LingBot-VLA 场景理解（可选，服务器不可用时自动跳过）──────
    planner = LingBotVLAPlanner(server_url=planner_url)
    camera_image = _load_camera_image(camera_image_path)

    plan = planner.plan_setup(instruction, camera_image)
    status_emoji = "✅" if plan["status"] == "ok" else "⚠️"
    logger.info(f"[2/4] LingBot-VLA 规划: {status_emoji} {plan['message']}")
    if plan["setup_actions"]:
        logger.info(f"       返回 {len(plan['setup_actions'])} 个预备动作")
    result["plan"] = plan

    # ── 步骤 3：生成 NPZ 轨迹文件 ────────────────────────────────────
    npz_path = os.path.join(output_dir, f"{char}.npz")

    if image_path is not None:
        # 用户直接提供字符图像 → 快速骨架提取
        logger.info(f"[3/4] 快速骨架提取: {image_path}")
        npz_path = selector.generate_npz_fast(image_path, npz_path, char_size, pen_depth)

    elif data_dir is not None:
        # 在 seq_extract 输出中查找精确数据
        order_dir = os.path.join(data_dir, "order")
        seq_dir = os.path.join(data_dir, "seq_data")
        found = selector.find_in_data(char, seq_dir, order_dir)

        if found:
            order_path, seq_path = found
            logger.info(f"[3/4] 精确笔画提取 (order={os.path.basename(order_path)})")
            npz_path = selector.generate_npz_accurate(
                order_path, seq_path, npz_path, char_size, pen_depth
            )
        else:
            # 精确数据不存在，尝试快速骨架提取
            img_path = selector.find_image_in_data(char, data_dir)
            if img_path:
                logger.info(f"[3/4] 快速骨架提取（未找到精确数据）: {img_path}")
                npz_path = selector.generate_npz_fast(img_path, npz_path, char_size, pen_depth)
            else:
                raise FileNotFoundError(
                    f"在 '{data_dir}' 中找不到 '{char}' 的数据。\n"
                    f"  期望文件: order/{char}.png 和 seq_data/{char}.npz\n"
                    f"  或使用 --image 参数直接指定字符图像"
                )
    else:
        raise ValueError(
            "缺少数据来源。请提供以下之一：\n"
            "  --image   字符图像路径（快速骨架提取）\n"
            "  --data-dir seq_extract 输出根目录（精确笔画提取）"
        )

    logger.info(f"       NPZ 已生成: {npz_path}")
    result["npz_path"] = npz_path

    # ── 步骤 4a：MuJoCo 仿真 ─────────────────────────────────────────
    if simulate:
        logger.info("[4/4] MuJoCo 仿真中...")
        canvas_path = _run_simulation(npz_path, output_dir, char, speed)
        logger.info(f"       画布已保存: {canvas_path}")
        result["canvas_path"] = canvas_path

    # ── 步骤 4b：真实机器人执行 ──────────────────────────────────────
    if execute:
        logger.info(f"[4/4] Franka FR3 执行 (IP: {robot_ip})...")
        _run_on_robot(npz_path, robot_ip, speed)

    return result


def _load_camera_image(camera_image_path: Optional[str]) -> Optional[np.ndarray]:
    """读取相机图像并转为 RGB numpy 数组。"""
    if camera_image_path is None:
        return None
    try:
        import cv2
        img = cv2.imread(camera_image_path)
        if img is None:
            logger.warning(f"无法读取相机图像: {camera_image_path}")
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.warning(f"读取相机图像失败: {e}")
        return None


def _run_simulation(npz_path: str, output_dir: str, char: str, speed: float) -> str:
    """
    在 MuJoCo 中执行轨迹，保存画布图像（离屏渲染，不弹出窗口）。

    逻辑与 generate_composite_video.py 保持一致：
    - 使用 render_mode="rgb_array" 离屏模式
    - 降低接触力阈值到 0.001（与 generate_composite_video.py 一致）
    - 最终保存 sim.paper_canvas 为 PNG
    """
    import cv2

    if _MUJOCO_SIM_DIR not in sys.path:
        sys.path.insert(0, _MUJOCO_SIM_DIR)
    from mujoco_simulator import FrankaCalligraphySimulator

    sim = FrankaCalligraphySimulator(render_mode="rgb_array")

    # 降低接触力阈值（与 generate_composite_video.py 保持一致）
    def patched_update():
        brush_pos, contact_force = sim.get_brush_contact()
        is_contact = contact_force > 0.001
        sim.ink_traces.append((*brush_pos, is_contact))
        if is_contact:
            sim._draw_on_canvas(brush_pos)
        else:
            if hasattr(sim, "_last_canvas_pos"):
                delattr(sim, "_last_canvas_pos")

    sim._update_ink_trace = patched_update

    # 加载并执行轨迹
    data = np.load(npz_path)
    x_pts = data["pos_3d_x"]
    y_pts = data["pos_3d_y"]
    z_pts = data["pos_3d_z"]
    n = len(x_pts)

    logger.info(f"       执行 {n} 个控制点...")
    for i in range(n):
        # Z 坐标变换（与 generate_composite_video.py 一致）
        z_world = z_pts[i] + 0.011       # 纸张表面偏移
        z_clamped = max(z_world, 0.008)  # 最低 0.008 m 防止 IK 失败
        target = np.array([
            x_pts[i] + sim.paper_offset[0],
            y_pts[i] + sim.paper_offset[1],
            z_clamped,
        ])
        sim.move_to_position(target, speed=speed, wait_time=0.0)

    # 保存画布
    canvas_path = os.path.join(output_dir, f"{char}_canvas.png")
    cv2.imwrite(canvas_path, sim.paper_canvas)
    sim.close()
    return canvas_path


def _run_on_robot(npz_path: str, robot_ip: str, speed: float):
    """在真实 Franka 机器人上执行轨迹（调用 callibrate/RoboControl.py）。"""
    if _CALLIBRATE_DIR not in sys.path:
        sys.path.insert(0, _CALLIBRATE_DIR)
    from RoboControl import Control
    Control(npz_path, robot_ip, speed)


def main():
    parser = argparse.ArgumentParser(
        description="CalliRewrite + LingBot-VLA 书法流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仿真 — 使用 seq_extract 精确笔画数据
  python -m lingbot_planner.pipeline "写一个昨字" --simulate \\
      --data-dir ../seq_extract/outputs/__new_train_phase_2

  # 仿真 — 直接提供字符图像（快速骨架提取）
  python -m lingbot_planner.pipeline "写龙" --image dragon.png --simulate

  # 真实机器人执行
  python -m lingbot_planner.pipeline "写龙" --image dragon.png \\
      --execute --robot-ip 172.16.0.2

  # 同时仿真并执行（先仿真验证再上机）
  python -m lingbot_planner.pipeline "写昨" \\
      --data-dir ../seq_extract/outputs/__new_train_phase_2 \\
      --simulate --execute --robot-ip 172.16.0.2

  # 启用 LingBot-VLA 场景理解（需在 GPU 机器单独启动服务器）
  python -m lingbot_planner.pipeline "写昨" --simulate \\
      --planner-url http://gpu-server:8080 --camera workspace.jpg
        """,
    )

    parser.add_argument("instruction", type=str, help="书写指令，如 '写一个龙字'")
    parser.add_argument(
        "--camera", type=str, default=None,
        help="工作区相机图像路径（供 LingBot-VLA 场景理解，可选）",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="字符图像路径（直接指定，使用快速骨架提取）",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="seq_extract 输出根目录（含 order/ 和 seq_data/，使用精确笔画提取）",
    )
    parser.add_argument(
        "--robot-ip", type=str, default="172.16.0.2",
        help="Franka 机器人 IP（默认: 172.16.0.2）",
    )
    parser.add_argument(
        "--planner-url", type=str, default="http://localhost:8080",
        help="LingBot-VLA 推理服务器 URL（默认: http://localhost:8080）",
    )
    parser.add_argument(
        "--output-dir", type=str, default="lingbot_outputs",
        help="输出目录（默认: lingbot_outputs）",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="在 MuJoCo 中仿真并保存画布图像",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="在真实 Franka FR3 机器人上执行（需连接机器人）",
    )
    parser.add_argument(
        "--speed", type=float, default=0.05,
        help="运动速度系数（默认: 0.05）",
    )
    parser.add_argument(
        "--char-size", type=float, default=0.12,
        help="字符大小，单位米（默认: 0.12 = 12 cm）",
    )
    parser.add_argument(
        "--pen-depth", type=float, default=-0.002,
        help="笔压深度，单位米（默认: -0.002 = -2 mm）",
    )

    args = parser.parse_args()

    if not args.simulate and not args.execute:
        parser.error("请至少指定 --simulate 或 --execute 之一")

    try:
        result = run_calligraphy_pipeline(
            instruction=args.instruction,
            camera_image_path=args.camera,
            image_path=args.image,
            data_dir=args.data_dir,
            robot_ip=args.robot_ip,
            planner_url=args.planner_url,
            output_dir=args.output_dir,
            simulate=args.simulate,
            execute=args.execute,
            speed=args.speed,
            char_size=args.char_size,
            pen_depth=args.pen_depth,
        )

        print(f"\n✅ 完成! 汉字: '{result['char']}'")
        if "npz_path" in result:
            print(f"   NPZ:  {result['npz_path']}")
        if "canvas_path" in result:
            print(f"   画布: {result['canvas_path']}")
        plan_status = result.get("plan", {}).get("status", "unavailable")
        if plan_status == "ok":
            print("   LingBot-VLA: ✅ 规划成功")
        else:
            print(f"   LingBot-VLA: ⚠️  {result.get('plan', {}).get('message', '未连接')}")

    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"流水线意外失败: {e}")
        raise


if __name__ == "__main__":
    main()

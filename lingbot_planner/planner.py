"""
LingBot-VLA 上层规划器 HTTP 客户端

通过 HTTP 与 LingBot-VLA 推理服务器通信，完成：
- 场景理解：检测纸张位置、毛笔状态
- 预动作规划：引导机器人到书写就绪状态

LingBot-VLA 服务器需单独在 GPU 机器上启动：
    cd lingbot-vla
    python -m deploy.lingbot_robotwin_policy --model_path <model_path> --port 8080

架构参考: arxiv:2601.18692 "A Pragmatic VLA Foundation Model"
  - 骨干网络: Qwen2.5-VL-3B-Instruct
  - 动作头: Flow Matching (14~75 维动作向量)
  - 训练数据: 20,000 小时，9 种机器人配置
"""

import base64
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests 未安装: pip install requests")

try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class LingBotVLAPlanner:
    """
    LingBot-VLA 上层规划器 HTTP 客户端

    连接到独立运行的 LingBot-VLA 推理服务器，获取：
    1. 场景理解结果（纸张位置、毛笔状态）
    2. 预备动作序列（定位、对齐等）

    服务器不可用时自动回退到 franka_config.yaml 默认配置。

    示例:
        planner = LingBotVLAPlanner("http://gpu-server:8080")
        if planner.is_available():
            plan = planner.plan_setup("写一个龙字", camera_img)
            # plan["paper_position"] → [0.4, 0.0, 0.2]
            # plan["setup_actions"] → list of action vectors
    """

    # 默认纸张位置（来自 callibrate/franka_config.yaml: workspace_center）
    _DEFAULT_PAPER_POSITION = [0.4, 0.0, 0.2]

    def __init__(self, server_url: str = "http://localhost:8080"):
        """
        Args:
            server_url: LingBot-VLA 推理服务器 URL（含端口）
        """
        self.server_url = server_url.rstrip("/")
        self._last_plan: Optional[dict] = None

    def is_available(self) -> bool:
        """
        检查 LingBot-VLA 服务器是否可用。

        Returns:
            True 表示服务器在线且可响应
        """
        if not HAS_REQUESTS:
            return False
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def plan_setup(
        self,
        instruction: str,
        camera_image: Optional[np.ndarray] = None,
    ) -> dict:
        """
        向 LingBot-VLA 服务器发送书写指令和相机图像，获取场景规划结果。

        LingBot-VLA 使用 Qwen2.5-VL 理解场景并输出：
        - 纸张在机器人坐标系中的位置
        - 预备动作序列（Flow Matching 生成的 action vectors）

        Args:
            instruction:    自然语言书写指令，如 "写一个龙字"
            camera_image:   工作区相机图像 (H, W, 3) RGB numpy 数组，可为 None

        Returns:
            dict，包含以下键：
            - status (str):           "ok" | "unavailable" | "error"
            - paper_position (list):  [x, y, z] 纸张中心（米，机器人基坐标系）
            - setup_actions (list):   预备动作向量列表（可能为空）
            - message (str):          状态说明
        """
        if not self.is_available():
            logger.info("LingBot-VLA 服务器不可用，使用默认纸张位置")
            return {
                "status": "unavailable",
                "paper_position": self._DEFAULT_PAPER_POSITION.copy(),
                "setup_actions": [],
                "message": "服务器不可用，使用默认配置（workspace_center from franka_config.yaml）",
            }

        try:
            payload = {"instruction": instruction}

            # 编码相机图像（JPEG base64）
            if camera_image is not None and HAS_PIL:
                img_pil = Image.fromarray(camera_image.astype(np.uint8))
                buf = io.BytesIO()
                img_pil.save(buf, format="JPEG", quality=85)
                payload["image_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            elif camera_image is not None and not HAS_PIL:
                logger.warning("Pillow 未安装，跳过图像编码: pip install Pillow")

            resp = requests.post(
                f"{self.server_url}/plan",
                json=payload,
                timeout=10.0,
            )
            resp.raise_for_status()

            result = resp.json()
            self._last_plan = result

            paper_pos = result.get("paper_position", self._DEFAULT_PAPER_POSITION.copy())
            actions = result.get("actions", [])

            return {
                "status": "ok",
                "paper_position": paper_pos,
                "setup_actions": actions,
                "message": f"规划成功，返回 {len(actions)} 个预备动作",
            }

        except Exception as e:
            logger.warning(f"LingBot-VLA 请求失败: {e}，回退到默认配置")
            return {
                "status": "error",
                "paper_position": self._DEFAULT_PAPER_POSITION.copy(),
                "setup_actions": [],
                "message": f"请求失败: {e}",
            }

    def get_default_paper_position(self) -> list:
        """返回默认纸张位置（来自 franka_config.yaml workspace_center）。"""
        return self._DEFAULT_PAPER_POSITION.copy()

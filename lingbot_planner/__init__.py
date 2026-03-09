"""
LingBot-VLA 上层规划器 — CalliRewrite 集成模块

通过自然语言指令驱动书法机器人：
  "写一个龙字" → LingBot-VLA 场景理解 → CalliRewrite 轨迹生成 → Franka FR3 执行

用法:
    python -m lingbot_planner.pipeline "写一个昨字" --simulate \\
        --data-dir ../seq_extract/outputs/__new_train_phase_2
"""

from .planner import LingBotVLAPlanner
from .character_selector import CharacterSelector

__all__ = ["LingBotVLAPlanner", "CharacterSelector"]

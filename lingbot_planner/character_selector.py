"""
汉字选择器：从自然语言指令解析目标汉字，并找到或生成对应 NPZ 轨迹文件。

两种 NPZ 生成路径：
- 精确路径 (generate_npz_accurate):  需要 order PNG + seq_data NPZ
                                     使用 extract_from_order_v2.py，笔画顺序正确
- 快速路径 (generate_npz_fast):      仅需字符图像
                                     使用 extract_dense_trajectory.py，秒级生成
"""

import logging
import os
import re
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# 项目路径
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
_MUJOCO_SIM_DIR = os.path.join(_ROOT_DIR, "mujoco_sim")


class CharacterSelector:
    """
    汉字选择器

    1. parse_character()  — 从自然语言指令提取目标汉字
    2. find_in_data()     — 在 seq_extract 输出中查找 order PNG + seq_data NPZ
    3. find_image_in_data() — 在数据目录中查找字符图像
    4. generate_npz_accurate() — 精确笔画提取（调用 extract_from_order_v2）
    5. generate_npz_fast()    — 快速骨架提取（调用 extract_dense_trajectory）

    示例:
        selector = CharacterSelector()
        char = selector.parse_character("写一个龙字")  # → "龙"
        found = selector.find_in_data("龙", seq_dir, order_dir)
        if found:
            npz = selector.generate_npz_accurate(*found, "out/龙.npz")
    """

    # 常见书写指令停用字（不是要书写的目标字）
    _STOP_CHARS = set("写帮我一个字请你来把到给为了的也和在就")

    def parse_character(self, instruction: str) -> Optional[str]:
        """
        从自然语言书写指令中提取目标汉字。

        支持格式（按优先级）：
        1. "X字" 模式：   "写一个龙字"  → "龙"
        2. 动词+字 模式： "帮我写龙"    → "龙"
        3. 单汉字指令：   "龙"          → "龙"
        4. 最后一个有效汉字（多字时）

        Args:
            instruction: 自然语言指令

        Returns:
            单个汉字，或 None（无法解析）
        """
        # 优先：匹配 "X字" 模式，X 为单个汉字
        m = re.search(r"([\u4e00-\u9fff])字", instruction)
        if m:
            candidate = m.group(1)
            if candidate not in self._STOP_CHARS:
                return candidate

        # 次优先：动词"写/帮"后紧跟的汉字
        m = re.search(r"[写帮]([\u4e00-\u9fff])", instruction)
        if m:
            candidate = m.group(1)
            if candidate not in self._STOP_CHARS:
                return candidate

        # 提取所有汉字，过滤停用字
        all_chars = re.findall(r"[\u4e00-\u9fff]", instruction)
        valid = [c for c in all_chars if c not in self._STOP_CHARS]

        if len(valid) == 1:
            return valid[0]
        if len(valid) > 1:
            # 多个有效汉字：取最后一个（通常是目标字）
            logger.debug(f"多个候选汉字 {valid}，取最后一个: {valid[-1]}")
            return valid[-1]

        return None

    def find_in_data(
        self,
        char: str,
        seq_data_dir: str,
        order_dir: str,
    ) -> Optional[tuple]:
        """
        在 seq_extract 输出中查找字符的 order PNG 和 seq_data NPZ。

        文件命名约定：
        - order/     {char}.png   — 彩色笔段顺序图
        - seq_data/  {char}.npz   — 包含 strokes_data 的序列数据

        Args:
            char:          目标汉字
            seq_data_dir:  seq_data/ 目录路径
            order_dir:     order/ 目录路径

        Returns:
            (order_png_path, seq_npz_path) 元组，或 None（未找到）
        """
        order_path = os.path.join(order_dir, f"{char}.png")
        seq_path = os.path.join(seq_data_dir, f"{char}.npz")
        if os.path.exists(order_path) and os.path.exists(seq_path):
            return order_path, seq_path
        return None

    def find_image_in_data(self, char: str, data_dir: str) -> Optional[str]:
        """
        在数据目录中查找字符图像（供快速骨架提取）。

        搜索顺序：imgs/, images/, samples/, order/, 根目录

        Args:
            char:      目标汉字
            data_dir:  数据根目录

        Returns:
            图像路径，或 None（未找到）
        """
        subdirs = ["imgs", "images", "samples", "order", "."]
        for subdir in subdirs:
            p = os.path.join(data_dir, subdir, f"{char}.png")
            if os.path.exists(p):
                return p
        return None

    def generate_npz_accurate(
        self,
        order_path: str,
        seq_path: str,
        output_path: str,
        char_size: float = 0.12,
        pen_depth: float = -0.002,
    ) -> str:
        """
        精确笔画提取：使用颜色匹配恢复正确笔画顺序。

        调用 mujoco_sim/extract_from_order_v2.py:extract_ordered_strokes_v2()
        - 优点：笔画顺序与 LSTM 预测完全一致，16~30 笔画正确分离
        - 缺点：需要 order PNG + seq_data NPZ（seq_extract 输出）

        Args:
            order_path:   order/ 目录中的 PNG 文件路径
            seq_path:     seq_data/ 目录中的 NPZ 文件路径
            output_path:  输出 NPZ 路径
            char_size:    字符大小（米，默认 0.12 m）
            pen_depth:    笔压深度（米，默认 -0.002 m）

        Returns:
            output_path
        """
        if _MUJOCO_SIM_DIR not in sys.path:
            sys.path.insert(0, _MUJOCO_SIM_DIR)
        from extract_from_order_v2 import extract_ordered_strokes_v2

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        extract_ordered_strokes_v2(
            order_path,
            seq_path,
            output_path,
            char_size=char_size,
            pen_depth=pen_depth,
        )
        return output_path

    def generate_npz_fast(
        self,
        image_path: str,
        output_path: str,
        char_size: float = 0.12,
        pen_depth: float = -0.002,
    ) -> str:
        """
        快速骨架提取：从字符图像直接生成 NPZ。

        调用 mujoco_sim/extract_dense_trajectory.py:extract_skeleton_trajectory()
        - 优点：秒级完成，只需字符图像
        - 缺点：笔画顺序由连通组件决定，不保证与书写顺序完全一致

        Args:
            image_path:   字符图像路径（PNG，黑字白底）
            output_path:  输出 NPZ 路径
            char_size:    字符大小（米，默认 0.12 m）
            pen_depth:    笔压深度（米，默认 -0.002 m）

        Returns:
            output_path
        """
        if _MUJOCO_SIM_DIR not in sys.path:
            sys.path.insert(0, _MUJOCO_SIM_DIR)
        from extract_dense_trajectory import extract_skeleton_trajectory

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        extract_skeleton_trajectory(image_path, output_path, char_size, pen_depth)
        return output_path

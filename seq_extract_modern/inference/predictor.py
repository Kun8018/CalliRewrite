"""
推理和预测接口
"""
import os
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from models.vit_transformer import CalligraphyExtractor, create_extractor_model
from configs.model_config import get_default_config


@dataclass
class StrokeParams:
    """
    笔画参数数据结构
    """
    x1: float  # 起点 x 坐标
    y1: float  # 起点 y 坐标
    x2: float  # 终点 x 坐标
    y2: float  # 终点 y 坐标
    width: float  # 笔画宽度
    pressure: float  # 压力
    eos: float  # 是否结束笔画的标志


class Predictor:
    """
    书法笔画提取预测器
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 config=None,
                 device: str = None):
        """
        Args:
            model_path: 模型权重路径
            config: 配置对象
            device: 设备 ('cpu', 'cuda', 'cuda:0' 等)
        """
        # 加载配置
        if config is None:
            config = get_default_config()
        self.config = config

        # 设置设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # 初始化模型
        self.model = create_extractor_model(config['model'])
        self.model.to(self.device)
        self.model.eval()

        # 加载权重
        if model_path is not None and os.path.exists(model_path):
            self._load_checkpoint(model_path)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.model.image_size, config.model.image_size)),
            transforms.ToTensor(),
        ])

    def _load_checkpoint(self, checkpoint_path: str):
        """
        加载模型权重
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"Model loaded from {checkpoint_path}")

    def _preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        预处理图像
        Args:
            image_path: 图像路径

        Returns:
            (image_tensor, original_size)
        """
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        original_size = img.size

        # 转换为灰度图
        img_np = np.array(img)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_np = np.mean(img_np, axis=2).astype(np.uint8)

        # 填充为正方形
        height, width = img_np.shape[:2]
        max_dim = max(height, width)
        if height != width:
            pad_height = max_dim - height
            pad_width = max_dim - width
            img_np = np.pad(
                img_np,
                ((0, pad_height), (0, pad_width)),
                mode='constant',
                constant_values=255
            )

        # 归一化：[0.0, 1.0]，0.0表示笔画，1.0表示背景
        img_np = img_np.astype(np.float32) / 255.0
        img_np = 1.0 - img_np  # 反转

        # 转换为张量
        img_tensor = self.transform(img_np)
        img_tensor = img_tensor.unsqueeze(0)  # 添加 batch 维度

        return img_tensor, original_size

    def predict(self,
                image_path: str,
                num_strokes: int = 100,
                return_raw: bool = False) -> dict:
        """
        预测书法图像的笔画序列
        Args:
            image_path: 输入图像路径
            num_strokes: 提取的笔画数量
            return_raw: 是否返回原始参数

        Returns:
            预测结果字典
        """
        # 预处理图像
        img_tensor, original_size = self._preprocess_image(image_path)
        img_tensor = img_tensor.to(self.device)

        # 预测
        with torch.no_grad():
            stroke_params = self.model.extract_strokes(
                img_tensor,
                sequence_length=num_strokes
            )

        # 处理结果
        stroke_params_np = stroke_params[0].cpu().numpy()

        result = {
            'image_path': image_path,
            'original_size': original_size,
            'num_strokes': num_strokes,
            'stroke_params': stroke_params_np,
        }

        if not return_raw:
            # 转换为更友好的格式
            result['strokes'] = self._format_strokes(stroke_params_np)

        return result

    def _format_strokes(self, stroke_params: np.ndarray) -> list:
        """
        格式化笔画参数
        """
        strokes = []
        for i in range(stroke_params.shape[0]):
            param = stroke_params[i]
            stroke = StrokeParams(
                x1=float(param[0]),
                y1=float(param[1]),
                x2=float(param[2]),
                y2=float(param[3]),
                width=float(param[4]),
                pressure=float(param[5]),
                eos=float(param[6])
            )
            strokes.append(stroke)

        return strokes

    def save_result(self, result: dict, output_path: str):
        """
        保存预测结果
        """
        # 保存 .npy 格式（与原始项目兼容）
        np.save(output_path, result['stroke_params'])
        print(f"Result saved to {output_path}")

    def visualize_result(self, result: dict, output_image_path: str):
        """
        可视化预测结果
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        # 加载原始图像
        img = Image.open(result['image_path']).convert('RGB')
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # 显示原始图像
        ax[0].imshow(img)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        # 显示笔画
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(1, 0)  # 反转 y 轴
        ax[1].set_aspect('equal')
        ax[1].set_title('Extracted Strokes')

        stroke_params = result['stroke_params']
        for i in range(stroke_params.shape[0]):
            x1, y1, x2, y2, width, pressure, eos = stroke_params[i]

            if eos < 0.5:  # 只绘制非结束笔画
                # 转换坐标
                x1 = (x1 + 1) / 2
                y1 = (y1 + 1) / 2
                x2 = (x2 + 1) / 2
                y2 = (y2 + 1) / 2

                # 绘制箭头
                arrow = FancyArrowPatch(
                    (x1, y1), (x2, y2),
                    arrowstyle='->',
                    linewidth=width * 5,
                    alpha=0.7 + pressure * 0.3
                )
                ax[1].add_patch(arrow)

        plt.tight_layout()
        plt.savefig(output_image_path, dpi=150)
        plt.close()

        print(f"Visualization saved to {output_image_path}")


def create_predictor(model_path: Optional[str] = None,
                     config=None,
                     device: str = None) -> Predictor:
    """
    创建预测器实例
    """
    return Predictor(model_path=model_path, config=config, device=device)
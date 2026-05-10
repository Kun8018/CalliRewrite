"""
数据加载和处理模块
"""
import os
import random
from typing import List, Tuple, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from configs.model_config import ViTTransformerConfig, TrainingConfig


class CalligraphyDataset(Dataset):
    """
    书法图像数据集
    """

    def __init__(self,
                 image_dir: str,
                 config: Union[ViTTransformerConfig, TrainingConfig],
                 transform: transforms.Compose = None,
                 training: bool = True):
        """
        Args:
            image_dir: 图像目录
            config: 配置对象
            transform: 数据增强变换
            training: 是否用于训练
        """
        self.image_dir = image_dir
        self.config = config
        self.transform = transform
        self.training = training

        # 获取所有图像文件
        self.image_files = self._get_image_files()

    def _get_image_files(self) -> List[str]:
        """获取目录中的图像文件"""
        supported_extensions = ['.png', '.jpg', '.jpeg']
        image_files = []

        if os.path.isdir(self.image_dir):
            for filename in os.listdir(self.image_dir):
                if any(filename.lower().endswith(ext) for ext in supported_extensions):
                    image_files.append(os.path.join(self.image_dir, filename))
        elif os.path.isfile(self.image_dir) and any(self.image_dir.lower().endswith(ext) for ext in supported_extensions):
            image_files = [self.image_dir]

        return sorted(image_files)

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """
        预处理图像：
        1. 转换为灰度图
        2. 填充为正方形
        3. 调整大小
        4. 归一化
        """
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img, dtype=np.uint8)

        # 转换为灰度图
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_np = np.mean(img_np, axis=2).astype(np.uint8)

        height, width = img_np.shape[:2]

        # 填充为正方形
        max_dim = max(height, width)
        if height != width:
            pad_height = max_dim - height
            pad_width = max_dim - width

            # 白色背景填充
            img_np = np.pad(
                img_np,
                ((0, pad_height), (0, pad_width)),
                mode='constant',
                constant_values=255
            )

        # 调整大小
        target_size = self.config.image_size
        if img_np.shape[0] != target_size:
            img = Image.fromarray(img_np)
            img = img.resize((target_size, target_size), Image.BILINEAR)
            img_np = np.array(img)

        # 归一化：[0.0, 1.0]，其中0.0表示笔画，1.0表示背景
        img_np = img_np.astype(np.float32) / 255.0
        img_np = 1.0 - img_np  # 反转，使0.0表示笔画

        return img_np

    def _generate_initial_cursor(self, sketch_image: np.ndarray) -> np.ndarray:
        """
        生成初始光标位置
        """
        image_size = sketch_image.shape[0]
        raster_size = self.config.raster_size

        # 找到第一个笔画像素
        stroke_indices = np.where(sketch_image < 0.1)
        if len(stroke_indices[0]) > 0:
            # 选择第一个笔画像素作为初始光标
            i, j = stroke_indices[0][0], stroke_indices[1][0]
            center = np.array([i, j], dtype=np.float32)
        else:
            # 如果没有笔画，返回中心
            center = np.array([image_size // 2, image_size // 2], dtype=np.float32)

        return center / float(image_size)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取数据项
        Returns:
            (image, initial_cursor) - 预处理后的图像和初始光标位置
        """
        image_path = self.image_files[idx]
        image_np = self._preprocess_image(image_path)

        # 生成初始光标位置
        initial_cursor = self._generate_initial_cursor(image_np)

        # 应用数据增强
        if self.transform:
            image_np = self.transform(image_np)

        return image_np, initial_cursor


class TestDataset(CalligraphyDataset):
    """
    测试数据集
    """

    def __init__(self,
                 image_dir: str,
                 config: Union[ViTTransformerConfig, TrainingConfig],
                 transform: transforms.Compose = None):
        super().__init__(image_dir, config, transform, training=False)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        获取测试数据项
        Returns:
            (image, initial_cursor, filename)
        """
        image_np, initial_cursor = super().__getitem__(idx)
        filename = os.path.basename(self.image_files[idx])
        return image_np, initial_cursor, filename


def create_data_loader(dataset: Dataset,
                       batch_size: int,
                       shuffle: bool = True,
                       num_workers: int = 4) -> torch.utils.data.DataLoader:
    """
    创建数据加载器
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def get_test_dataset(image_path: str,
                     config: Union[ViTTransformerConfig, TrainingConfig]
                     ) -> TestDataset:
    """
    获取测试数据集
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    return TestDataset(image_path, config, transform)
"""
数据增强和预处理变换
"""
import random

import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RandomCrop:
    """
    随机裁剪增强
    """

    def __init__(self, crop_size: int = 224):
        self.crop_size = crop_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.shape[0] <= self.crop_size or image.shape[1] <= self.crop_size:
            return image

        height, width = image.shape[:2]
        y = random.randint(0, height - self.crop_size)
        x = random.randint(0, width - self.crop_size)

        return image[y:y + self.crop_size, x:x + self.crop_size]


class RandomRotation:
    """
    随机旋转增强
    """

    def __init__(self, max_angle: int = 15):
        self.max_angle = max_angle

    def __call__(self, image: np.ndarray) -> np.ndarray:
        angle = random.randint(-self.max_angle, self.max_angle)

        # 使用PIL旋转
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        img_pil = img_pil.rotate(angle, Image.BILINEAR, expand=True, fillcolor=255)
        img_np = np.array(img_pil) / 255.0

        return img_np


class RandomContrast:
    """
    随机对比度增强
    """

    def __init__(self, contrast_range: Tuple[float, float] = (0.8, 1.2)):
        self.contrast_range = contrast_range

    def __call__(self, image: np.ndarray) -> np.ndarray:
        contrast_factor = random.uniform(*self.contrast_range)

        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(contrast_factor)
        img_np = np.array(img_pil) / 255.0

        return img_np


class RandomBrightness:
    """
    随机亮度增强
    """

    def __init__(self, brightness_range: Tuple[float, float] = (0.8, 1.2)):
        self.brightness_range = brightness_range

    def __call__(self, image: np.ndarray) -> np.ndarray:
        brightness_factor = random.uniform(*self.brightness_range)

        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(brightness_factor)
        img_np = np.array(img_pil) / 255.0

        return img_np


class GaussianBlur:
    """
    高斯模糊增强
    """

    def __init__(self, sigma_range: Tuple[float, float] = (0.1, 0.5)):
        self.sigma_range = sigma_range

    def __call__(self, image: np.ndarray) -> np.ndarray:
        sigma = random.uniform(*self.sigma_range)

        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
        img_np = np.array(img_pil) / 255.0

        return img_np


def get_training_transforms(config):
    """
    获取训练数据增强变换
    """
    transforms_list = [
        transforms.ToPILImage(),
    ]

    if config.use_augmentation:
        if config.random_rotate:
            transforms_list.append(RandomRotation(max_angle=15))

        if config.random_crop and config.image_size > 224:
            transforms_list.append(RandomCrop(crop_size=224))

        transforms_list.extend([
            RandomContrast(contrast_range=(0.8, 1.2)),
            RandomBrightness(brightness_range=(0.8, 1.2)),
        ])

    transforms_list.extend([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])

    return transforms.Compose(transforms_list)


def get_validation_transforms(config):
    """
    获取验证数据增强变换（不使用随机增强）
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])


def get_test_transforms(config):
    """
    获取测试数据增强变换
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])


def get_albumentations_transforms(config):
    """
    使用 Albumentations 库的增强变换（推荐用于高性能增强）
    """
    transforms_list = [
        A.Resize(config.image_size, config.image_size),
    ]

    if config.use_augmentation:
        transforms_list.extend([
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.GridDistortion(p=0.3),
        ])

    transforms_list.extend([
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)
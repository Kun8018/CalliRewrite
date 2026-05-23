"""
数据加载模块 (彩色标注笔画)
支持两种输入格式:
1. RGB 图片 (红色标注当前笔画)
2. 灰度图 + 红色mask (分离)
"""
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def extract_red_mask(rgb_img):
    """从 RGB 图片提取红色mask"""
    r = rgb_img[:, :, 0].astype(np.float32)
    g = rgb_img[:, :, 1].astype(np.float32)
    b = rgb_img[:, :, 2].astype(np.float32)

    # 红色通道明显比其他通道高
    mask = (r > g + 30) & (r > b + 30)
    return mask.astype(np.float32)


class StrokeColorDatasetViT(Dataset):
    """
    ViT 轨迹提取数据集 (彩色标注)

    目录结构:
        data_dir/
            img_00.png (RGB, 红色标第1笔)
            img_01.png (红色标第2笔)
            ...
            data.npz (完整笔画)
    """

    def __init__(self, data_dir=None, img_size=224, seq_len=100,
                 mode='rgb'):
        """
        参数:
            data_dir: 数据目录
            img_size: 输入图像大小
            seq_len: 输出序列长度
            mode: 'rgb' (直接用RGB) 或 'dual' (分离mask)
        """
        self.img_size = img_size
        self.seq_len = seq_len
        self.mode = mode

        self.samples = []

        if data_dir is not None:
            self._load_from_dir(data_dir)

        print(f"Loaded {len(self.samples)} samples (mode={mode})")

    def _load_from_dir(self, data_dir):
        """从目录加载"""
        if not os.path.exists(data_dir):
            print(f"Warning: {data_dir} not found")
            return

        # 找 npz
        npz_path = None
        for f in os.listdir(data_dir):
            if f.endswith('.npz'):
                npz_path = os.path.join(data_dir, f)
                break

        if npz_path is None:
            return

        # 加载 strokes
        try:
            data = np.load(npz_path, allow_pickle=True, encoding='latin1')
            strokes_data = data['strokes_data']
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            return

        # 找图片
        image_files = []
        for f in sorted(os.listdir(data_dir)):
            if f.endswith(('.png', '.jpg', '.jpeg')) and not f.endswith('.npz'):
                image_files.append(os.path.join(data_dir, f))

        for img_path in image_files:
            self.samples.append((img_path, strokes_data))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, strokes_data = self.samples[idx]

        # 加载 RGB 图片
        img = Image.open(img_path).convert('RGB')
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size))
        img_np = np.array(img)

        if self.mode == 'rgb':
            # RGB 模式: (3, H, W)
            img_np = img_np.astype(np.float32) / 255.0
            img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
            image_tensor = torch.tensor(img_np, dtype=torch.float32)
        else:
            # Dual 模式: 分离灰度和mask
            gray_np = np.mean(img_np, axis=2, keepdims=True).astype(np.float32) / 255.0
            mask_np = extract_red_mask(img_np)
            gray_tensor = torch.tensor(gray_np.transpose(2, 0, 1), dtype=torch.float32)
            mask_tensor = torch.tensor(mask_np[np.newaxis, :, :], dtype=torch.float32)
            image_tensor = (gray_tensor, mask_tensor)

        # 处理 strokes
        if len(strokes_data.shape) == 2 and strokes_data.shape[1] == 7:
            strokes = strokes_data
        else:
            strokes = np.zeros((self.seq_len, 7), dtype=np.float32)

        seq_len = min(len(strokes), self.seq_len)
        padded_strokes = np.zeros((self.seq_len, 7), dtype=np.float32)
        padded_strokes[:seq_len] = strokes[:seq_len]
        mask = np.zeros(self.seq_len, dtype=np.float32)
        mask[:seq_len] = 1.0

        if self.mode == 'rgb':
            return {
                'image': image_tensor,  # (3, 224, 224)
                'strokes': torch.tensor(padded_strokes, dtype=torch.float32),
                'mask': torch.tensor(mask, dtype=torch.float32),
                'seq_len': seq_len
            }
        else:
            return {
                'gray_image': image_tensor[0],  # (1, 224, 224)
                'red_mask': image_tensor[1],  # (1, 224, 224)
                'strokes': torch.tensor(padded_strokes, dtype=torch.float32),
                'mask': torch.tensor(mask, dtype=torch.float32),
                'seq_len': seq_len
            }


class MultiCharColorDataset(Dataset):
    """
    多字符彩色标注数据集

    目录结构:
        data_dir/
            char_001/
                img_00.png
                img_01.png
                ...
                data.npz
            char_002/
                ...
    """

    def __init__(self, data_dir=None, img_size=224, seq_len=100, mode='rgb'):
        self.img_size = img_size
        self.seq_len = seq_len
        self.mode = mode
        self.samples = []

        if data_dir is not None:
            self._load_from_dir(data_dir)

        print(f"Loaded {len(self.samples)} samples (mode={mode})")

    def _load_from_dir(self, data_dir):
        if not os.path.exists(data_dir):
            return

        for item in sorted(os.listdir(data_dir)):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                self._load_char_dir(item_path)

    def _load_char_dir(self, char_dir):
        npz_path = None
        for f in os.listdir(char_dir):
            if f.endswith('.npz'):
                npz_path = os.path.join(char_dir, f)
                break

        if npz_path is None:
            return

        try:
            data = np.load(npz_path, allow_pickle=True, encoding='latin1')
            strokes_data = data['strokes_data']
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            return

        for f in sorted(os.listdir(char_dir)):
            if f.endswith(('.png', '.jpg', '.jpeg')) and not f.endswith('.npz'):
                self.samples.append((os.path.join(char_dir, f), strokes_data))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, strokes_data = self.samples[idx]

        img = Image.open(img_path).convert('RGB')
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size))
        img_np = np.array(img)

        if self.mode == 'rgb':
            img_np = img_np.astype(np.float32) / 255.0
            img_np = img_np.transpose(2, 0, 1)
            image_tensor = torch.tensor(img_np, dtype=torch.float32)
        else:
            gray_np = np.mean(img_np, axis=2, keepdims=True).astype(np.float32) / 255.0
            mask_np = extract_red_mask(img_np)
            gray_tensor = torch.tensor(gray_np.transpose(2, 0, 1), dtype=torch.float32)
            mask_tensor = torch.tensor(mask_np[np.newaxis, :, :], dtype=torch.float32)
            image_tensor = (gray_tensor, mask_tensor)

        if len(strokes_data.shape) == 2 and strokes_data.shape[1] == 7:
            strokes = strokes_data
        else:
            strokes = np.zeros((self.seq_len, 7), dtype=np.float32)

        seq_len = min(len(strokes), self.seq_len)
        padded_strokes = np.zeros((self.seq_len, 7), dtype=np.float32)
        padded_strokes[:seq_len] = strokes[:seq_len]
        mask = np.zeros(self.seq_len, dtype=np.float32)
        mask[:seq_len] = 1.0

        if self.mode == 'rgb':
            return {
                'image': image_tensor,
                'strokes': torch.tensor(padded_strokes, dtype=torch.float32),
                'mask': torch.tensor(mask, dtype=torch.float32),
                'seq_len': seq_len
            }
        else:
            return {
                'gray_image': image_tensor[0],
                'red_mask': image_tensor[1],
                'strokes': torch.tensor(padded_strokes, dtype=torch.float32),
                'mask': torch.tensor(mask, dtype=torch.float32),
                'seq_len': seq_len
            }

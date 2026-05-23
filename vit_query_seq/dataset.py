"""
数据加载模块 (支持渐进式图片序列)
输入: 图片序列 (img_0.png, img_1.png, ...) - 每加一笔
输出: 7D 笔画序列 (seq_len, 7)
"""
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def natural_sort_key(s):
    """自然排序: img_1.png, img_10.png → 正确排序"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


class StrokeSeqDatasetViT(Dataset):
    """
    ViT 轨迹提取数据集 (渐进式图片序列)

    目录结构:
        data_dir/
            character_001/
                img_00.png  (第一笔)
                img_01.png  (第一笔+第二笔)
                ...
                img_09.png  (完整字)
                data.npz    (对应的完整笔画)
            character_002/
                ...
    """

    def __init__(self, data_dir=None, img_size=224, seq_len=100,
                 num_images=10, mode='seq7'):
        """
        参数:
            data_dir: 包含子目录的数据目录
            img_size: 输入图像大小
            seq_len: 输出序列长度
            num_images: 每个字符的图片数量
            mode: 'seq7' (目前只支持这个)
        """
        self.img_size = img_size
        self.seq_len = seq_len
        self.num_images = num_images
        self.mode = mode

        self.samples = []

        if data_dir is not None:
            self._load_from_dir(data_dir)

        print(f"Loaded {len(self.samples)} samples (num_images={num_images})")

    def _load_from_dir(self, data_dir):
        """从目录加载数据"""
        if not os.path.exists(data_dir):
            print(f"Warning: {data_dir} not found")
            return

        # 遍历子目录
        for item in sorted(os.listdir(data_dir)):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                self._load_sample_dir(item_path)

    def _load_sample_dir(self, sample_dir):
        """加载单个样本目录"""
        # 找 npz 文件
        npz_path = None
        for f in os.listdir(sample_dir):
            if f.endswith('.npz'):
                npz_path = os.path.join(sample_dir, f)
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

        # 找图片文件
        image_files = []
        for f in os.listdir(sample_dir):
            if f.endswith(('.png', '.jpg', '.jpeg')) and not f.endswith('.npz'):
                image_files.append(os.path.join(sample_dir, f))

        # 自然排序
        image_files = sorted(image_files, key=natural_sort_key)

        # 如果图片数量不对，跳过
        if len(image_files) != self.num_images:
            return

        self.samples.append((image_files, strokes_data))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_files, strokes_data = self.samples[idx]

        # 加载图片序列
        images = []
        for img_path in image_files:
            img = Image.open(img_path).convert('L')
            if img.size != (self.img_size, self.img_size):
                img = img.resize((self.img_size, self.img_size))
            img_np = np.array(img, dtype=np.float32) / 255.0
            images.append(img_np)

        # 堆叠成 (num_images, 1, H, W)
        images_np = np.stack(images, axis=0)
        images_np = np.expand_dims(images_np, axis=1)  # 加通道维度

        # 处理 strokes
        if len(strokes_data.shape) == 2 and strokes_data.shape[1] == 7:
            strokes = strokes_data
        else:
            strokes = np.zeros((self.seq_len, 7), dtype=np.float32)

        # 截取或填充
        seq_len = min(len(strokes), self.seq_len)
        padded_strokes = np.zeros((self.seq_len, 7), dtype=np.float32)
        padded_strokes[:seq_len] = strokes[:seq_len]

        mask = np.zeros(self.seq_len, dtype=np.float32)
        mask[:seq_len] = 1.0

        return {
            'images': torch.tensor(images_np, dtype=torch.float32),  # (num_images, 1, 224, 224)
            'strokes': torch.tensor(padded_strokes, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'seq_len': seq_len
        }


class SimpleStrokeSeqDatasetViT(Dataset):
    """
    简化版本: 单目录下 img_00.png ~ img_09.png + data.npz

    目录结构:
        data_dir/
            img_00.png
            img_01.png
            ...
            img_09.png
            data.npz
    """

    def __init__(self, data_dir=None, img_size=224, seq_len=100,
                 num_images=10):
        self.img_size = img_size
        self.seq_len = seq_len
        self.num_images = num_images
        self.samples = []

        if data_dir is not None:
            self._load_data(data_dir)

        print(f"Loaded {len(self.samples)} samples")

    def _load_data(self, data_dir):
        if not os.path.exists(data_dir):
            return

        # 找 npz
        npz_path = None
        for f in sorted(os.listdir(data_dir)):
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
        for f in sorted(os.listdir(data_dir), key=natural_sort_key):
            if f.endswith(('.png', '.jpg', '.jpeg')) and not f.endswith('.npz'):
                image_files.append(os.path.join(data_dir, f))

        if len(image_files) == self.num_images:
            self.samples.append((image_files, strokes_data))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_files, strokes_data = self.samples[idx]

        # 加载图片
        images = []
        for img_path in image_files:
            img = Image.open(img_path).convert('L')
            if img.size != (self.img_size, self.img_size):
                img = img.resize((self.img_size, self.img_size))
            img_np = np.array(img, dtype=np.float32) / 255.0
            images.append(img_np)

        images_np = np.stack(images, axis=0)
        images_np = np.expand_dims(images_np, axis=1)

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

        return {
            'images': torch.tensor(images_np, dtype=torch.float32),
            'strokes': torch.tensor(padded_strokes, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'seq_len': seq_len
        }

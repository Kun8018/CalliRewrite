"""
数据加载模块
支持两种输出格式：
1. 2D 密集点坐标 (num_points, 2)
2. 7D 贝塞尔曲线序列 (seq_len, 7)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class StrokeDatasetViT(Dataset):
    """
    ViT 轨迹提取数据集

    支持两种模式：
    - mode='points': 输出 (num_points, 2) 密集点
    - mode='seq7': 输出 (seq_len, 7) 贝塞尔序列
    """

    def __init__(self, data_dir=None, npz_files=None, image_stroke_pairs=None,
                 img_size=224, num_points=100, seq_len=100, mode='seq7'):
        """
        参数:
            data_dir: 包含 npz 和对应图像的目录
            npz_files: npz 文件路径列表
            image_stroke_pairs: (image_path, stroke_data) 列表
            img_size: 输入图像大小
            num_points: 输出点数量（mode='points'）
            seq_len: 输出序列长度（mode='seq7'）
            mode: 'points' 或 'seq7'
        """
        self.img_size = img_size
        self.num_points = num_points
        self.seq_len = seq_len
        self.mode = mode

        self.samples = []

        if npz_files is not None:
            for npz_file in npz_files:
                self._load_npz(npz_file)

        if data_dir is not None:
            self._load_from_dir(data_dir)

        if image_stroke_pairs is not None:
            self.samples.extend(image_stroke_pairs)

        print(f"Loaded {len(self.samples)} samples (mode: {mode})")

    def _load_npz(self, npz_path):
        try:
            data = np.load(npz_path, allow_pickle=True, encoding='latin1')
            strokes_data = data['strokes_data']

            image_path = npz_path.replace('.npz', '.png')
            if not os.path.exists(image_path):
                image_path = npz_path.replace('.npz', '.jpg')

            if os.path.exists(image_path):
                self.samples.append((image_path, strokes_data))
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")

    def _load_from_dir(self, data_dir):
        npz_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.npz'):
                    npz_files.append(os.path.join(root, f))

        for npz_file in npz_files:
            self._load_npz(npz_file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, strokes_data = self.samples[idx]

        # 加载并预处理图像
        image = Image.open(image_path).convert('L')
        if image.size != (self.img_size, self.img_size):
            image = image.resize((self.img_size, self.img_size))

        # 归一化到 [0, 1]，ViT 常用这个范围
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)  # (1, H, W)

        if self.mode == 'seq7':
            return self._get_item_seq7(image, strokes_data)
        else:
            return self._get_item_points(image, strokes_data)

    def _get_item_seq7(self, image, strokes_data):
        """返回 7D 序列格式"""
        if len(strokes_data.shape) == 2 and strokes_data.shape[1] == 7:
            strokes = strokes_data
        else:
            strokes = self._convert_to_7d(strokes_data)

        # 截取或填充
        seq_len = min(len(strokes), self.seq_len)
        padded_strokes = np.zeros((self.seq_len, 7), dtype=np.float32)
        padded_strokes[:seq_len] = strokes[:seq_len]

        mask = np.zeros(self.seq_len, dtype=np.float32)
        mask[:seq_len] = 1.0

        return {
            'image': image,
            'strokes': torch.tensor(padded_strokes, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'seq_len': seq_len
        }

    def _get_item_points(self, image, strokes_data):
        """返回 2D 密集点格式（需要转换）"""
        # 先转换到 7D，再采样成密集点
        if not (len(strokes_data.shape) == 2 and strokes_data.shape[1] == 7):
            strokes_data = self._convert_to_7d(strokes_data)

        # 采样贝塞尔曲线为密集点
        points = self._sample_strokes_to_points(strokes_data, self.num_points)

        return {
            'image': image,
            'points': torch.tensor(points, dtype=torch.float32),
        }

    def _convert_to_7d(self, strokes_data):
        """简单格式转换占位符"""
        # 实际项目中需要根据你的数据格式实现
        if len(strokes_data.shape) == 2 and strokes_data.shape[1] == 3:
            return self._quickdraw_to_7d(strokes_data)
        return strokes_data

    def _quickdraw_to_7d(self, qd_strokes):
        """QuickDraw 到 7D 格式的简化转换"""
        result = []
        x, y = 0.0, 0.0

        points = []
        for dx, dy, ps in qd_strokes:
            x += dx
            y += dy
            points.append((x, y, ps))

        strokes = []
        current_stroke = []
        for p in points:
            if p[2] == 1 and current_stroke:
                strokes.append(current_stroke)
                current_stroke = []
            current_stroke.append(p)
        if current_stroke:
            strokes.append(current_stroke)

        for stroke in strokes:
            if len(stroke) < 2:
                continue
            result.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0])
            for i in range(len(stroke) - 1):
                x0, y0, _ = stroke[i]
                x1, y1, _ = stroke[i + 1]
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                result.append([0.0, 0.5, 0.5, 1.0, 1.0, 0.1, 1.0])

        return np.array(result, dtype=np.float32)

    def _sample_strokes_to_points(self, strokes_7d, num_points):
        """
        将 7D 贝塞尔曲线采样成密集点

        这里是简化实现，实际项目可以用 rl_finetune 中的 skel_utils
        """
        # 简单实现：假设 strokes_7d 已经是某种点的格式
        # 实际中需要解析贝塞尔曲线并采样

        # 占位实现：生成一些示例点
        t = np.linspace(0, 1, num_points)
        points = np.stack([t, t], axis=1)  # 占位
        return points.astype(np.float32)


def create_dataloader(data_dir=None, npz_files=None, batch_size=32,
                      img_size=224, num_points=100, seq_len=100, mode='seq7',
                      num_workers=4, shuffle=True):
    dataset = StrokeDatasetViT(
        data_dir=data_dir,
        npz_files=npz_files,
        img_size=img_size,
        num_points=num_points,
        seq_len=seq_len,
        mode=mode
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return loader


if __name__ == "__main__":
    print("Testing dataset...")

    # 先把 seq_extract 的 png 和 npz 配对
    test_dirs = []
    possible_dirs = [
        '../seq_extract/outputs/__new_train_phase_2',
        '../rl_finetune/data/train_data',
    ]

    for d in possible_dirs:
        if os.path.exists(d):
            test_dirs.append(d)

    if test_dirs:
        dataset_seq7 = StrokeDatasetViT(data_dir=test_dirs[0], img_size=224, mode='seq7')
        print(f"\nSeq7 dataset: {len(dataset_seq7)} samples")

        if len(dataset_seq7) > 0:
            sample = dataset_seq7[0]
            print(f"Image shape: {sample['image'].shape}")
            print(f"Strokes shape: {sample['strokes'].shape}")
            print(f"Mask shape: {sample['mask'].shape}")

        dataset_points = StrokeDatasetViT(data_dir=test_dirs[0], img_size=224, mode='points')
        print(f"\nPoints dataset: {len(dataset_points)} samples")

        if len(dataset_points) > 0:
            sample = dataset_points[0]
            print(f"Image shape: {sample['image'].shape}")
            print(f"Points shape: {sample['points'].shape}")

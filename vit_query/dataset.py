"""
数据加载模块
支持：
1. seq_extract 输出的 .png + .npz 监督数据
2. QuickDraw-clean/stroke3 phase1 预训练数据
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


QUICKDRAW_CATEGORIES = [
    'airplane', 'bus', 'car', 'sailboat', 'bird', 'cat', 'dog',
    'tree', 'flower', 'zigzag'
]


class StrokeDatasetViT(Dataset):
    """ViT 轨迹提取数据集：读取同名 .png/.jpg + .npz。"""

    def __init__(self, data_dir=None, npz_files=None, image_stroke_pairs=None,
                 img_size=224, num_points=100, seq_len=100, mode='seq7'):
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
        for npz_file in sorted(npz_files):
            self._load_npz(npz_file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, strokes_data = self.samples[idx]
        image = load_grayscale_tensor(image_path, self.img_size)

        if self.mode == 'seq7':
            return self._get_item_seq7(image, strokes_data)
        return self._get_item_points(image, strokes_data)

    def _get_item_seq7(self, image, strokes_data):
        strokes = ensure_7d(strokes_data)
        return make_seq7_item(image, strokes, self.seq_len)

    def _get_item_points(self, image, strokes_data):
        t = np.linspace(0, 1, self.num_points)
        points = np.stack([t, t], axis=1).astype(np.float32)
        return {'image': image, 'points': torch.tensor(points, dtype=torch.float32)}


class QuickDrawCleanDatasetViT(Dataset):
    """phase1：读取 seq_extract/datasets/QuickDraw-clean/{train,test}/*.npz 的 stroke3。"""

    def __init__(self, dataset_root, split='train', img_size=224, seq_len=100,
                 categories=None, max_items_per_category=None):
        self.img_size = img_size
        self.seq_len = seq_len
        self.samples = []
        categories = categories or QUICKDRAW_CATEGORIES

        base_dir = os.path.join(dataset_root, 'QuickDraw-clean', split)
        for category in categories:
            npz_path = os.path.join(base_dir, f'{category}.npz')
            if not os.path.exists(npz_path):
                print(f"Warning: missing {npz_path}")
                continue
            data = np.load(npz_path, allow_pickle=True, encoding='latin1')
            stroke3_list = data['stroke3'].tolist()
            if max_items_per_category is not None:
                stroke3_list = stroke3_list[:max_items_per_category]
            self.samples.extend(stroke3_list)

        print(f"Loaded {len(self.samples)} QuickDraw-clean samples ({split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stroke3 = np.asarray(self.samples[idx], dtype=np.float32)
        image = render_stroke3_tensor(stroke3, self.img_size)
        strokes = quickdraw_stroke3_to_7d(stroke3)
        return make_seq7_item(image, strokes, self.seq_len)


def load_grayscale_tensor(image_path, img_size):
    image = Image.open(image_path).convert('L')
    if image.size != (img_size, img_size):
        image = image.resize((img_size, img_size))
    image = np.array(image, dtype=np.float32) / 255.0
    return torch.tensor(image).unsqueeze(0)


def make_seq7_item(image, strokes, seq_len):
    seq_len_actual = min(len(strokes), seq_len)
    padded_strokes = np.zeros((seq_len, 7), dtype=np.float32)
    if seq_len_actual > 0:
        padded_strokes[:seq_len_actual] = strokes[:seq_len_actual]
    mask = np.zeros(seq_len, dtype=np.float32)
    mask[:seq_len_actual] = 1.0
    return {
        'image': image,
        'strokes': torch.tensor(padded_strokes, dtype=torch.float32),
        'mask': torch.tensor(mask, dtype=torch.float32),
        'seq_len': seq_len_actual
    }


def ensure_7d(strokes_data):
    strokes_data = np.asarray(strokes_data, dtype=np.float32)
    if strokes_data.ndim == 2 and strokes_data.shape[1] == 7:
        return strokes_data
    if strokes_data.ndim == 2 and strokes_data.shape[1] == 3:
        return quickdraw_stroke3_to_7d(strokes_data)
    return strokes_data


def normalize_xy(points):
    xy = points[:, :2].astype(np.float32)
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    scale = np.maximum(max_xy - min_xy, 1e-6)
    return (xy - min_xy) / scale


def render_stroke3_tensor(stroke3, img_size):
    xy = normalize_xy(stroke3)
    pts = xy * (img_size - 1)
    img = Image.new('L', (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)
    for i in range(len(pts) - 1):
        if stroke3[i, 2] > 0.5:
            continue
        draw.line([tuple(pts[i]), tuple(pts[i + 1])], fill=0, width=3)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr).unsqueeze(0)


def quickdraw_stroke3_to_7d(stroke3):
    if len(stroke3) < 2:
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0]], dtype=np.float32)

    xy = normalize_xy(stroke3)
    result = []
    current = xy[0]

    for i in range(1, len(xy)):
        prev = xy[i - 1]
        point = xy[i]
        if stroke3[i - 1, 2] > 0.5:
            result.append([1.0, 0.0, 0.0, point[0] - current[0], point[1] - current[1], 0.1, 1.0])
            current = point
            continue

        delta = point - current
        ctrl = (prev + point) / 2 - current
        result.append([0.0, ctrl[0], ctrl[1], delta[0], delta[1], 0.1, 1.0])
        current = point

    if not result:
        result.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0])
    return np.asarray(result, dtype=np.float32)


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

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


if __name__ == "__main__":
    print("Testing dataset...")
    possible_dirs = [
        '../seq_extract/outputs/__new_train_phase_2',
        '../rl_finetune/data/train_data',
    ]
    for d in possible_dirs:
        if os.path.exists(d):
            dataset = StrokeDatasetViT(data_dir=d, img_size=224, mode='seq7')
            print(f"Seq7 dataset: {len(dataset)} samples")
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"Image shape: {sample['image'].shape}")
                print(f"Strokes shape: {sample['strokes'].shape}")
                print(f"Mask shape: {sample['mask'].shape}")
            break

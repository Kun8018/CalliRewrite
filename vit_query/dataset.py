"""数据加载模块。

接口约定：
- target_image:      (1, H, W) ∈ [0, 1], 1=BG  (PIL 灰度归一化)
- target_stroke_img: (H, W)    ∈ [0, 1], 1=stroke  (raster loss 直接用)
- gt_strokes:        (T, 7),  padded 到 max_seq_len, 末尾置 0
- gt_mask:           (T,)     0/1, 标记 GT 有效长度
- image_path:        str（phase 2 仅有图像时使用）
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


# ===================================================================== #
# 数据集类
# ===================================================================== #

class QuickDrawCleanDataset(Dataset):
    """读取 seq_extract/datasets/QuickDraw-clean/{train,test}/*.npz 中的 stroke3。
    Phase1 监督数据来源。"""

    def __init__(self, dataset_root, split='train', image_size=256, max_seq_len=100,
                 categories=None, max_items_per_category=None):
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.samples = []
        categories = categories or QUICKDRAW_CATEGORIES

        base_dir = os.path.join(dataset_root, 'QuickDraw-clean', split)
        for category in categories:
            npz_path = os.path.join(base_dir, f'{category}.npz')
            if not os.path.exists(npz_path):
                print(f'Warning: missing {npz_path}')
                continue
            data = np.load(npz_path, allow_pickle=True, encoding='latin1')
            stroke3_list = data['stroke3'].tolist()
            if max_items_per_category is not None:
                stroke3_list = stroke3_list[:max_items_per_category]
            self.samples.extend(stroke3_list)

        print(f'Loaded {len(self.samples)} QuickDraw-clean samples ({split})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stroke3 = np.asarray(self.samples[idx], dtype=np.float32)
        target_image = render_stroke3_tensor(stroke3, self.image_size)        # (1, H, W) ∈ [0,1], 1=BG
        target_stroke = 1.0 - target_image.squeeze(0)                          # (H, W), 1=stroke
        strokes = quickdraw_stroke3_to_7d(stroke3, self.image_size)
        gt, mask, seq_len = pad_strokes(strokes, self.max_seq_len)
        return {
            'target_image': target_image,
            'target_stroke_img': target_stroke,
            'gt_strokes': torch.from_numpy(gt),
            'gt_mask': torch.from_numpy(mask),
            'seq_len': seq_len,
        }


class ImageOnlyDataset(Dataset):
    """Phase 2 无监督：只读图片。"""

    def __init__(self, data_dir=None, image_files=None, image_size=256):
        self.image_size = image_size
        self.samples = list(image_files or [])
        if data_dir is not None:
            for root, _, files in os.walk(data_dir):
                for f in sorted(files):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append(os.path.join(root, f))
        print(f'Loaded {len(self.samples)} image-only samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.samples[idx]
        target_image = load_grayscale_tensor(image_path, self.image_size)  # (1, H, W) 1=BG
        target_stroke = 1.0 - target_image.squeeze(0)
        return {
            'target_image': target_image,
            'target_stroke_img': target_stroke,
            'image_path': image_path,
        }


# ===================================================================== #
# Helpers
# ===================================================================== #

def load_grayscale_tensor(image_path, img_size):
    """读 PIL 灰度图 → (1, H, W) ∈ [0,1], 1=BG / 0=stroke。"""
    image = Image.open(image_path).convert('L')
    if image.size != (img_size, img_size):
        image = image.resize((img_size, img_size))
    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def pad_strokes(strokes: np.ndarray, max_seq_len: int):
    """(L, 7) → (max_seq_len, 7), mask (max_seq_len,), L"""
    L = min(len(strokes), max_seq_len)
    padded = np.zeros((max_seq_len, 7), dtype=np.float32)
    if L > 0:
        padded[:L] = strokes[:L]
    mask = np.zeros(max_seq_len, dtype=np.float32)
    mask[:L] = 1.0
    return padded, mask, L


# ---- stroke3 → seq7 转换（与原 v1 等价，保留供 phase 1 数据预处理使用）---- #

def stroke3_to_normalized_xy(stroke3):
    stroke3 = np.asarray(stroke3, dtype=np.float32)
    xy = np.cumsum(stroke3[:, :2], axis=0)
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    scale = np.maximum(max_xy - min_xy, 1e-6)
    return (xy - min_xy) / scale


def quickdraw_stroke3_to_7d(stroke3, img_size=256):
    """stroke3 (L, 3) [dx, dy, pen_lift] → seq7 (T, 7)。
    跟 cursor/window 的逻辑一致：每点都按当前 cursor / window 求相对偏移。"""
    if len(stroke3) < 2:
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0]], dtype=np.float32)
    points = stroke3_to_normalized_xy(stroke3)
    return _normalized_points_to_seq7(points, stroke3[:, 2], img_size=img_size)


def _normalized_points_to_seq7(points, pen_lifts, img_size=256):
    cursor = points[0].copy()
    prev_window = float(min(128, img_size))
    prev_scaling = 1.0
    result = []
    for i in range(1, len(points)):
        curr_window = max(32.0, prev_scaling * prev_window)
        prev = points[i - 1]
        cur = points[i]
        pen = 1.0 if pen_lifts[i - 1] > 0.5 else 0.0
        if pen == 1.0:
            stroke = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0], dtype=np.float32)
        else:
            ctrl = (prev + cur) / 2.0
            scale = 2.0 * img_size / max(curr_window, 1e-6)
            ctrl_rel = (ctrl - cursor) * scale
            end_rel = (cur - cursor) * scale
            stroke = np.array([
                0.0,
                np.clip(ctrl_rel[0], -1.0, 1.0),
                np.clip(ctrl_rel[1], -1.0, 1.0),
                np.clip(end_rel[0], -1.0, 1.0),
                np.clip(end_rel[1], -1.0, 1.0),
                0.1,
                1.0
            ], dtype=np.float32)
        result.append(stroke)
        # cursor 走到终点
        if pen == 1.0:
            cursor = cur.copy()
        else:
            cursor = cur.copy()
        prev_window = curr_window
        prev_scaling = stroke[6]
    return np.asarray(result, dtype=np.float32)


def render_stroke3_tensor(stroke3, img_size):
    """stroke3 → PIL 渲染 (1, H, W) ∈ [0, 1], 1=BG / 0=stroke。"""
    xy = stroke3_to_normalized_xy(stroke3)
    pts = xy * (img_size - 1)
    img = Image.new('L', (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)
    for i in range(len(pts) - 1):
        if stroke3[i, 2] > 0.5:
            continue
        draw.line([tuple(pts[i]), tuple(pts[i + 1])], fill=0, width=3)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ---- 推理时 cursor 重启策略保留 ---- #

def find_undrawn_cursor(target_mask, canvas, patch_size=32, threshold=10.0):
    """在 residual 图中找最大未画区域的中心，返回归一化坐标 (2,) 或 None。"""
    target = np.asarray(target_mask, dtype=np.float32)
    drawn = np.asarray(canvas, dtype=np.float32)
    residual = np.clip(target - drawn, 0.0, 1.0)
    h, w = residual.shape
    best_score = 0.0
    best_center = None
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = residual[y:min(y + patch_size, h), x:min(x + patch_size, w)]
            score = float(patch.sum())
            if score > best_score:
                best_score = score
                best_center = (x + patch.shape[1] / 2.0, y + patch.shape[0] / 2.0)
    if best_center is None or best_score < threshold:
        return None
    return np.array([best_center[0] / w, best_center[1] / h], dtype=np.float32)


if __name__ == '__main__':
    print('Testing dataset module...')
    # 简单冒烟测试：构造一个小 QuickDraw-clean dataset 看 __getitem__
    import sys
    if len(sys.argv) > 1:
        ds = QuickDrawCleanDataset(dataset_root=sys.argv[1], split='train',
                                    max_items_per_category=2)
        for i in range(min(2, len(ds))):
            s = ds[i]
            print(i, 'image', s['target_image'].shape, 'stroke_img',
                  s['target_stroke_img'].shape, 'gt', s['gt_strokes'].shape,
                  'seq_len', s['seq_len'])

"""
数据加载模块
支持：
1. seq_extract 输出的 .png + .npz 监督数据
2. QuickDraw-clean/stroke3 phase1 预训练数据
"""
import os
import random
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
                 img_size=224, num_points=100, seq_len=100, mode='seq7',
                 arch='oneshot', chunk_len=8, chunks_per_sample=4):
        self.img_size = img_size
        self.num_points = num_points
        self.seq_len = seq_len
        self.mode = mode
        self.arch = arch
        self.chunk_len = chunk_len
        self.chunks_per_sample = chunks_per_sample
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
        if self.arch == 'autoregressive':
            return len(self.samples) * self.chunks_per_sample
        return len(self.samples)

    def __getitem__(self, idx):
        if self.arch == 'autoregressive':
            idx = idx // self.chunks_per_sample
        image_path, strokes_data = self.samples[idx]
        image = load_grayscale_tensor(image_path, self.img_size)

        if self.arch == 'autoregressive':
            strokes = ensure_7d(strokes_data)
            return make_autoregressive_item(image, strokes, self.seq_len, self.chunk_len, self.img_size)
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
                 categories=None, max_items_per_category=None,
                 arch='oneshot', chunk_len=8, chunks_per_sample=4):
        self.img_size = img_size
        self.seq_len = seq_len
        self.arch = arch
        self.chunk_len = chunk_len
        self.chunks_per_sample = chunks_per_sample
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
        if self.arch == 'autoregressive':
            return len(self.samples) * self.chunks_per_sample
        return len(self.samples)

    def __getitem__(self, idx):
        if self.arch == 'autoregressive':
            idx = idx // self.chunks_per_sample
        stroke3 = np.asarray(self.samples[idx], dtype=np.float32)
        image = render_stroke3_tensor(stroke3, self.img_size)
        strokes = quickdraw_stroke3_to_7d(stroke3)
        if self.arch == 'autoregressive':
            return make_autoregressive_item(image, strokes, self.seq_len, self.chunk_len, self.img_size)
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


def initial_seq7_state(img_size, cursor=None):
    if cursor is None:
        cursor = np.array([0.5, 0.5], dtype=np.float32)
    return {
        'canvas': np.zeros((img_size, img_size), dtype=np.float32),
        'cursor': np.asarray(cursor, dtype=np.float32),
        'prev_stroke': np.zeros(7, dtype=np.float32),
        'prev_width': 0.1,
        'prev_scaling': 1.0,
        'prev_window_size': min(128.0, float(img_size)),
    }


def seq7_window_size(state, img_size, min_window_size=32.0):
    curr_window_size = state['prev_scaling'] * state['prev_window_size']
    return float(np.clip(curr_window_size, min_window_size, img_size))


def render_seq7_step_on_canvas(canvas, cursor, stroke, img_size, curr_window_size=None, line_width=3):
    if stroke[0] >= 0.5:
        return canvas
    if curr_window_size is None:
        curr_window_size = min(128.0, float(img_size))

    x0 = float(cursor[0]) * img_size
    y0 = float(cursor[1]) * img_size
    x1 = x0 + float(stroke[1]) * curr_window_size / 2.0
    y1 = y0 + float(stroke[2]) * curr_window_size / 2.0
    x2 = x0 + float(stroke[3]) * curr_window_size / 2.0
    y2 = y0 + float(stroke[4]) * curr_window_size / 2.0

    pil = Image.fromarray((canvas * 255).astype(np.uint8), mode='L')
    draw = ImageDraw.Draw(pil)
    points = []
    for t in np.linspace(0.0, 1.0, 16):
        x = (1 - t) * (1 - t) * x0 + 2 * (1 - t) * t * x1 + t * t * x2
        y = (1 - t) * (1 - t) * y0 + 2 * (1 - t) * t * y1 + t * t * y2
        points.append((float(np.clip(x, 0, img_size - 1)), float(np.clip(y, 0, img_size - 1))))
    if len(points) >= 2:
        draw.line(points, fill=255, width=line_width)
    return np.array(pil, dtype=np.float32) / 255.0


def update_seq7_cursor_state(state, stroke, img_size, min_window_size=32.0):
    curr_window_size = seq7_window_size(state, img_size, min_window_size)
    cursor = state['cursor'].copy()
    delta = np.array([stroke[3], stroke[4]], dtype=np.float32) * curr_window_size / 2.0 / float(img_size)
    cursor = np.clip(cursor + delta, 0.0, (img_size - 1) / float(img_size)).astype(np.float32)

    next_scaling = float(np.clip(stroke[6], 0.05, 2.0))
    next_window_size = float(np.clip(next_scaling * curr_window_size, min_window_size, img_size))
    state['cursor'] = cursor
    state['prev_width'] = float(stroke[5]) * curr_window_size / max(next_window_size, 1e-6)
    state['prev_scaling'] = next_scaling
    state['prev_window_size'] = curr_window_size
    state['prev_stroke'] = stroke.astype(np.float32)
    return state


def apply_seq7_step(state, stroke, img_size):
    curr_window_size = seq7_window_size(state, img_size)
    state['canvas'] = render_seq7_step_on_canvas(
        state['canvas'], state['cursor'], stroke, img_size, curr_window_size
    )
    return update_seq7_cursor_state(state, stroke, img_size)


def simulate_seq7_state_until(strokes, end_idx, img_size):
    state = initial_seq7_state(img_size)
    for i in range(max(0, min(end_idx, len(strokes)))):
        state = apply_seq7_step(state, strokes[i].astype(np.float32), img_size)
    return state


def make_target_mask(image):
    arr = image.squeeze(0).numpy().astype(np.float32)
    return torch.tensor(1.0 - arr, dtype=torch.float32).unsqueeze(0)


def make_autoregressive_item(image, strokes, seq_len, chunk_len, img_size):
    strokes = ensure_7d(strokes)
    seq_len_actual = min(len(strokes), seq_len)
    if seq_len_actual > 0:
        max_start = max(seq_len_actual - 1, 0)
        start_idx = random.randint(0, max_start)
    else:
        start_idx = 0

    state = simulate_seq7_state_until(strokes, start_idx, img_size)
    canvases = np.zeros((chunk_len, 1, img_size, img_size), dtype=np.float32)
    cursors = np.zeros((chunk_len, 2), dtype=np.float32)
    prev_strokes = np.zeros((chunk_len, 7), dtype=np.float32)
    step_indices = np.zeros((chunk_len, 1), dtype=np.float32)
    targets = np.zeros((chunk_len, 7), dtype=np.float32)
    mask = np.zeros(chunk_len, dtype=np.float32)

    for j in range(chunk_len):
        stroke_idx = start_idx + j
        canvases[j, 0] = state['canvas']
        cursors[j] = state['cursor']
        prev_strokes[j] = state['prev_stroke']
        step_indices[j, 0] = min(stroke_idx / max(seq_len, 1), 1.0)
        if stroke_idx < seq_len_actual:
            stroke = strokes[stroke_idx].astype(np.float32)
            targets[j] = stroke
            mask[j] = 1.0
            state = apply_seq7_step(state, stroke, img_size)

    return {
        'target_mask': make_target_mask(image),
        'canvases': torch.tensor(canvases, dtype=torch.float32),
        'cursors': torch.tensor(cursors, dtype=torch.float32),
        'prev_strokes': torch.tensor(prev_strokes, dtype=torch.float32),
        'step_indices': torch.tensor(step_indices, dtype=torch.float32),
        'strokes': torch.tensor(targets, dtype=torch.float32),
        'mask': torch.tensor(mask, dtype=torch.float32),
        'seq_len': seq_len_actual
    }


def find_undrawn_cursor(target_mask, canvas, patch_size=32, threshold=10.0):
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


def stroke3_to_normalized_xy(stroke3):
    stroke3 = np.asarray(stroke3, dtype=np.float32)
    xy = np.cumsum(stroke3[:, :2], axis=0)
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    scale = np.maximum(max_xy - min_xy, 1e-6)
    return (xy - min_xy) / scale


def render_stroke3_tensor(stroke3, img_size):
    xy = stroke3_to_normalized_xy(stroke3)
    pts = xy * (img_size - 1)
    img = Image.new('L', (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)
    for i in range(len(pts) - 1):
        if stroke3[i, 2] > 0.5:
            continue
        draw.line([tuple(pts[i]), tuple(pts[i + 1])], fill=0, width=3)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr).unsqueeze(0)


def normalized_points_to_seq7(points, pen_lifts, img_size=224):
    if len(points) < 2:
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0]], dtype=np.float32)

    state = initial_seq7_state(img_size, cursor=points[0])
    result = []
    for i in range(1, len(points)):
        curr_window_size = seq7_window_size(state, img_size)
        prev = points[i - 1]
        point = points[i]
        pen = 1.0 if pen_lifts[i - 1] > 0.5 else 0.0
        if pen == 1.0:
            stroke = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0], dtype=np.float32)
        else:
            cursor = state['cursor']
            ctrl = (prev + point) / 2.0
            scale = 2.0 * img_size / max(curr_window_size, 1e-6)
            ctrl_rel = (ctrl - cursor) * scale
            end_rel = (point - cursor) * scale
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
        state = apply_seq7_step(state, stroke, img_size)
        if pen == 1.0:
            state['cursor'] = point.astype(np.float32)

    return np.asarray(result, dtype=np.float32)


def quickdraw_stroke3_to_7d(stroke3):
    if len(stroke3) < 2:
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0]], dtype=np.float32)
    xy = stroke3_to_normalized_xy(stroke3)
    return normalized_points_to_seq7(xy, stroke3[:, 2], img_size=224)


def create_dataloader(data_dir=None, npz_files=None, batch_size=32,
                      img_size=224, num_points=100, seq_len=100, mode='seq7',
                      num_workers=4, shuffle=True, arch='oneshot',
                      chunk_len=8, chunks_per_sample=4):
    dataset = StrokeDatasetViT(
        data_dir=data_dir,
        npz_files=npz_files,
        img_size=img_size,
        num_points=num_points,
        seq_len=seq_len,
        mode=mode,
        arch=arch,
        chunk_len=chunk_len,
        chunks_per_sample=chunks_per_sample
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

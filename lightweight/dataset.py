"""
数据加载模块
支持:
1. 现有seq_extract格式的npz数据
2. QuickDraw数据格式
3. 自定义数据格式
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json


class StrokeDataset(Dataset):
    """
    笔画数据集
    支持从npz加载，或直接用图像+stroke pairs
    """
    def __init__(
        self,
        data_dir=None,
        npz_files=None,
        image_stroke_pairs=None,
        image_size=256,
        max_seq_len=100,
        transform=None
    ):
        """
        参数:
            data_dir: 包含npz和对应图像的目录
            npz_files: npz文件路径列表
            image_stroke_pairs: (image_path, stroke_data)列表
            image_size: 图像尺寸
            max_seq_len: 最大序列长度
            transform: 图像变换
        """
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.transform = transform

        self.samples = []

        # 从npz文件加载
        if npz_files is not None:
            for npz_file in npz_files:
                self._load_npz(npz_file)

        # 从数据目录加载
        if data_dir is not None:
            self._load_from_dir(data_dir)

        # 从pairs加载
        if image_stroke_pairs is not None:
            self.samples.extend(image_stroke_pairs)

        print(f"Loaded {len(self.samples)} samples")

    def _load_npz(self, npz_path):
        """从seq_extract格式的npz加载"""
        try:
            data = np.load(npz_path, allow_pickle=True, encoding='latin1')
            strokes_data = data['strokes_data']

            # 查找对应的图像（假设同名）
            image_path = npz_path.replace('.npz', '.png')
            if not os.path.exists(image_path):
                image_path = npz_path.replace('.npz', '.jpg')

            if os.path.exists(image_path):
                self.samples.append((image_path, strokes_data))
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")

    def _load_from_dir(self, data_dir):
        """从目录自动查找npz和图像"""
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

        # 加载图像
        image = Image.open(image_path).convert('L')  # 灰度图
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size))

        # 归一化
        image = np.array(image, dtype=np.float32) / 255.0
        image = (image - 0.5) / 0.5  # 归一化到 [-1, 1]

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image).unsqueeze(0)  # (1, H, W)

        # 处理stroke数据
        if len(strokes_data.shape) == 2 and strokes_data.shape[1] == 7:
            # 已经是 (seq_len, 7) 格式
            strokes = strokes_data
        else:
            # 其他格式需要转换
            strokes = self._convert_format(strokes_data)

        # 截取或填充到max_seq_len
        seq_len = min(len(strokes), self.max_seq_len)
        padded_strokes = np.zeros((self.max_seq_len, 7), dtype=np.float32)
        padded_strokes[:seq_len] = strokes[:seq_len]

        # 有效性mask
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:seq_len] = 1.0

        return {
            'image': image,
            'strokes': torch.tensor(padded_strokes, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'seq_len': seq_len
        }

    def _convert_format(self, strokes_data):
        """转换其他格式到7维"""
        # 这里可以根据实际数据格式调整
        # 假设是 (seq_len, 3) 的QuickDraw格式: [dx, dy, pen_state]
        if len(strokes_data.shape) == 2 and strokes_data.shape[1] == 3:
            return self._quickdraw_to_our_format(strokes_data)

        return strokes_data

    def _quickdraw_to_our_format(self, qd_strokes):
        """
        QuickDraw格式 -> 我们的7维格式
        QuickDraw: [dx, dy, pen_state]
        我们: [pen_state, x1, y1, x2, y2, r, s]
        """
        result = []

        # 首先将dx/dy转换为绝对坐标
        x, y = 0.0, 0.0
        absolute_points = []

        for dx, dy, ps in qd_strokes:
            x += dx
            y += dy
            absolute_points.append((x, y, ps))

        # 现在转换为贝塞尔曲线表示
        # 这里做简化：两点组成一条线
        for i in range(len(absolute_points) - 1):
            x0, y0, ps0 = absolute_points[i]
            x1, y1, ps1 = absolute_points[i + 1]

            if ps0 == 0 and ps1 == 0:  # 笔画中间
                # 控制点在中间
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2

                result.append([
                    0.0,  # pen_state: 绘画
                    0.5, 0.5,  # 控制点相对位置
                    1.0, 1.0,  # 终点相对位置
                    0.1,  # r (默认值)
                    1.0   # s (默认值)
                ])
            elif ps0 == 1:  # 开始新笔画
                result.append([
                    1.0,  # pen_state: 移动
                    0.0, 0.0,
                    0.0, 0.0,
                    0.1,
                    1.0
                ])

        return np.array(result, dtype=np.float32)


class QuickDrawConverter:
    """QuickDraw数据格式转换器"""

    @staticmethod
    def load_quickdraw_npz(npz_path, max_items=1000):
        """加载QuickDraw npz文件"""
        data = np.load(npz_path, allow_pickle=True, encoding='latin1')
        sketches = data['train'][:max_items]  # 用训练集
        return sketches

    @staticmethod
    def render_strokes(strokes, image_size=256):
        """将QuickDraw笔画渲染成图像"""
        from PIL import Image, ImageDraw

        # 创建空白图像
        img = Image.new('L', (image_size, image_size), 255)
        draw = ImageDraw.Draw(img)

        # 规范化坐标
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')

        x, y = 0, 0
        points = []
        for dx, dy, _ in strokes:
            x += dx
            y += dy
            points.append((x, y))
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        # 缩放和居中
        scale = min(image_size * 0.8 / (x_max - x_min + 1),
                    image_size * 0.8 / (y_max - y_min + 1))
        offset_x = (image_size - (x_max - x_min) * scale) / 2 - x_min * scale
        offset_y = (image_size - (y_max - y_min) * scale) / 2 - y_min * scale

        # 绘制
        x, y = 0, 0
        pen_down = False
        line_points = []

        for dx, dy, pen_state in strokes:
            x += dx
            y += dy

            # 变换坐标
            px = x * scale + offset_x
            py = y * scale + offset_y

            if pen_state == 0:
                if not pen_down:
                    pen_down = True
                    line_points = [(px, py)]
                else:
                    line_points.append((px, py))
            elif pen_state == 1 and pen_down:
                pen_down = False
                if len(line_points) >= 2:
                    draw.line(line_points, fill=0, width=2)
                line_points = []

        # 绘制最后一条线
        if pen_down and len(line_points) >= 2:
            draw.line(line_points, fill=0, width=2)

        return np.array(img)

    @staticmethod
    def create_pairs(sketches, save_dir=None, image_size=256):
        """创建(image, strokes) pairs"""
        pairs = []

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for idx, sketch in enumerate(sketches):
            # 渲染图像
            img = QuickDrawConverter.render_strokes(sketch, image_size)

            # 转换格式
            strokes = QuickDrawConverter._convert_qd_to_7d(sketch)

            if save_dir:
                # 保存
                img_pil = Image.fromarray(img)
                img_path = os.path.join(save_dir, f'qd_{idx:06d}.png')
                npz_path = os.path.join(save_dir, f'qd_{idx:06d}.npz')
                img_pil.save(img_path)
                np.savez(npz_path, strokes_data=strokes)
                pairs.append((img_path, strokes))
            else:
                # 仅在内存中
                pairs.append((img, strokes))

        return pairs

    @staticmethod
    def _convert_qd_to_7d(sketch):
        """
        QuickDraw格式转7维格式
        [dx, dy, ps] -> [ps, x1, y1, x2, y2, r, s]
        """
        result = []
        x, y = 0.0, 0.0

        # 先收集绝对坐标
        points = []
        for dx, dy, ps in sketch:
            x += dx
            y += dy
            points.append((x, y, ps))

        # 分组为笔画
        strokes = []
        current_stroke = []
        for p in points:
            if p[2] == 1 and current_stroke:
                strokes.append(current_stroke)
                current_stroke = []
            current_stroke.append(p)
        if current_stroke:
            strokes.append(current_stroke)

        # 转换每一画
        for stroke in strokes:
            if len(stroke) < 2:
                continue

            # 第一个点是移动到起点
            result.append([
                1.0,  # pen_state: move
                0.0, 0.0,
                0.0, 0.0,
                0.1, 1.0
            ])

            # 后续点是贝塞尔曲线
            for i in range(len(stroke) - 1):
                x0, y0, _ = stroke[i]
                x1, y1, _ = stroke[i + 1]

                # 控制点在中间
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2

                result.append([
                    0.0,  # pen_state: draw
                    0.5, 0.5,  # 控制点（相对）
                    1.0, 1.0,  # 终点（相对）
                    0.1, 1.0
                ])

        return np.array(result, dtype=np.float32)


def create_dataloader(data_dir=None, npz_files=None, batch_size=32,
                      image_size=256, max_seq_len=100, num_workers=4, shuffle=True):
    """创建DataLoader"""
    dataset = StrokeDataset(
        data_dir=data_dir,
        npz_files=npz_files,
        image_size=image_size,
        max_seq_len=max_seq_len
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return loader


if __name__ == "__main__":
    # 测试
    print("Testing data module...")

    # 1. 测试数据集创建
    dataset = StrokeDataset(data_dir="../rl_finetune/data")
    print(f"Dataset size: {len(dataset)}")

    # 2. 测试获取样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Strokes shape: {sample['strokes'].shape}")
        print(f"Seq len: {sample['seq_len']}")

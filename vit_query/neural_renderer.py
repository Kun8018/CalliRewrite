
import torch
import torch.nn as nn
import torch.nn.functional as F


class RasterUnit(nn.Module):
    def __init__(self, raster_size):
        super().__init__()
        self.raster_size = raster_size

        self.fc1 = nn.Linear(10, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)

        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(8, 4, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = x.view(-1, 16, 16, 16)
        x = x.permute(0, 2, 3, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.pixel_shuffle(x, upscale_factor=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.pixel_shuffle(x, upscale_factor=2)

        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        x = F.pixel_shuffle(x, upscale_factor=2)

        x = torch.sigmoid(x)
        stroke_image = 1.0 - x.view(-1, self.raster_size, self.raster_size)
        return stroke_image


class NeuralRasterizorStep(nn.Module):
    def __init__(self, raster_size, position_format='abs'):
        super().__init__()
        self.raster_size = raster_size
        self.position_format = position_format
        self.raster_unit = RasterUnit(raster_size)

    def raster_func_stroke_abs(self, input_data):
        """
        input_data: (N, 8) [x0, y0, x1, y1, x2, y2, r0, r2] in [0, 1]
        returns: (N, raster_size, raster_size) [0.0-BG, 1.0-stroke]
        """
        w_in = torch.ones(input_data.shape[0], 2, device=input_data.device)
        raster_params = torch.cat([input_data, w_in], dim=-1)
        stroke_image = self.raster_unit(raster_params)
        stroke_image = 1.0 - stroke_image
        return stroke_image

    def forward(self, strokes):
        """
        strokes: (N, seq_len, 10) - absolute format [x0, y0, x1, y1, x2, y2, r0, r2, w0, w2]
                 or (N, seq_len, 8) - absolute format [x0, y0, x1, y1, x2, y2, r0, r2]
        returns: (N, raster_size, raster_size) [0.0-BG, 1.0-stroke]
        """
        N, seq_len, D = strokes.shape
        canvas = torch.zeros(N, self.raster_size, self.raster_size, device=strokes.device)

        for i in range(seq_len):
            stroke = strokes[:, i, :]
            if D == 10:
                stroke_params = stroke[:, :8]
            else:
                stroke_params = stroke
            stroke_img = self.raster_func_stroke_abs(stroke_params)
            canvas = torch.clamp(canvas + stroke_img, 0.0, 1.0)

        return canvas


def seq7_to_absolute(strokes_seq7, img_size):
    """
    将 seq7 相对坐标序列转换为 NeuralRenderer 需要的绝对坐标格式
    strokes_seq7: (N, seq_len, 7) - [pen, dx1, dy1, dx2, dy2, r, s]
    returns: (N, seq_len, 8) - [x0, y0, x1, y1, x2, y2, r0, r2] in [0, 1]
    """
    N, seq_len, _ = strokes_seq7.shape
    device = strokes_seq7.device

    # 初始化状态
    cursor = torch.full((N, 2), 0.5, device=device)  # 从中心开始
    prev_scaling = torch.ones(N, device=device)
    prev_window_size = torch.full((N,), float(min(128, img_size)), device=device)

    abs_strokes = []

    for i in range(seq_len):
        stroke = strokes_seq7[:, i, :]
        pen = stroke[:, 0]

        # 计算当前窗口大小
        curr_window_size = prev_scaling * prev_window_size
        curr_window_size = torch.clamp(curr_window_size, 32.0, float(img_size))

        # 计算绝对坐标
        x0 = cursor[:, 0]
        y0 = cursor[:, 1]

        # 控制点和终点
        dx1 = stroke[:, 1]
        dy1 = stroke[:, 2]
        dx2 = stroke[:, 3]
        dy2 = stroke[:, 4]

        x1 = x0 + dx1 * curr_window_size / 2.0 / img_size
        y1 = y0 + dy1 * curr_window_size / 2.0 / img_size
        x2 = x0 + dx2 * curr_window_size / 2.0 / img_size
        y2 = y0 + dy2 * curr_window_size / 2.0 / img_size

        # 半径
        r = stroke[:, 5]
        r_abs = r * curr_window_size / img_size  # 转换为 [0,1] 范围

        # 只在 pen < 0.5 时绘制
        mask = (pen < 0.5).float()

        # 构造绝对坐标格式，非绘制时用 (0,0,...)
        abs_stroke = torch.stack([
            x0 * mask, y0 * mask,
            x1 * mask, y1 * mask,
            x2 * mask, y2 * mask,
            r_abs * mask, r_abs * mask
        ], dim=1)
        abs_strokes.append(abs_stroke)

        # 更新状态
        cursor = torch.stack([x2, y2], dim=1)
        cursor = torch.clamp(cursor, 0.0, 1.0)

        next_scaling = torch.clamp(stroke[:, 6], 0.05, 2.0)
        prev_scaling = next_scaling
        prev_window_size = curr_window_size

    return torch.stack(abs_strokes, dim=1)

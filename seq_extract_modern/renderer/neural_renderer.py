"""
神经渲染器模块
将抽象的笔画参数转换为真实感图像
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RasterUnit(nn.Module):
    """
    神经渲染单元
    接受笔画参数并生成 raster_size x raster_size 的图像
    """

    def __init__(self, raster_size: int = 128):
        super().__init__()
        self.raster_size = raster_size

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
        )

        # 卷积和上采样层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 32x32

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 64x64

            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 128x128
            nn.Sigmoid(),
        )

    def forward(self, input_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_params: (B, 10) - [x0, y0, x1, y1, x2, y2, r0, r2, w0, w2]

        Returns:
            stroke_image: (B, raster_size, raster_size) - [0.0-stroke, 1.0-BG]
        """
        # 全连接网络
        x = self.fc_layers(input_params)

        # 重塑和卷积
        x = x.view(-1, 16, 16, 16)  # (B, 16, 16, 16)
        x = x.permute(0, 3, 1, 2)  # (B, 16, 16, 16)

        x = self.conv_layers(x)

        # 重塑为输出大小
        x = x.view(-1, self.raster_size, self.raster_size)

        # 确保输出在 [0, 1] 范围内
        x = x.clamp(0.0, 1.0)

        # 反转：[0.0-stroke, 1.0-BG]
        stroke_image = 1.0 - x

        return stroke_image


class NeuralRasterizor(nn.Module):
    """
    神经渲染器，将笔画序列转换为图像
    """

    def __init__(self, raster_size: int = 128, seq_len: int = 20):
        super().__init__()
        self.raster_size = raster_size
        self.seq_len = seq_len

        self.raster_unit = RasterUnit(raster_size=raster_size)

    def raster_func_abs(self, input_data: torch.Tensor,
                        raster_seq_len: int = None) -> torch.Tensor:
        """
        绝对坐标下的笔画渲染
        Args:
            input_data: (B, seq_len, 10) - [x0, y0, x1, y1, x2, y2, r0, r2, w0, w2]，所有值在 [0.0, 1.0] 范围内
            raster_seq_len: 渲染的序列长度

        Returns:
            stroke_images: (B, raster_size, raster_size) - [0.0-BG, 1.0-stroke]
        """
        seq_len = raster_seq_len if raster_seq_len is not None else self.seq_len

        batch_size = input_data.size(0)

        # 渲染每个笔画
        seq_stroke_images = []
        for i in range(seq_len):
            stroke_params = input_data[:, i]
            stroke_image = self.raster_unit(stroke_params)  # (B, raster_size, raster_size)
            seq_stroke_images.append(stroke_image)

        # (seq_len, B, raster_size, raster_size) -> (B, seq_len, raster_size, raster_size)
        seq_stroke_images = torch.stack(seq_stroke_images, dim=1)

        # 合并所有笔画
        stroke_images = 1.0 - seq_stroke_images  # 反转，使0表示背景，1表示笔画
        stroke_images = stroke_images.sum(dim=1)  # 相加所有笔画
        stroke_images = stroke_images.clamp(0.0, 1.0)  # 确保值在 [0, 1] 范围内

        return stroke_images


class StrokeRenderer(nn.Module):
    """
    笔画渲染器，支持各种渲染模式
    """

    def __init__(self, raster_size: int = 128):
        super().__init__()
        self.raster_size = raster_size
        self.neural_rasterizor = NeuralRasterizor(raster_size=raster_size)

    def render_stroke_sequence(self, stroke_params_list: torch.Tensor,
                               seq_len: int = None) -> torch.Tensor:
        """
        渲染完整的笔画序列
        Args:
            stroke_params_list: (B, seq_len, 10)
            seq_len: 序列长度

        Returns:
            rendered_images: (B, 1, raster_size, raster_size)
        """
        if seq_len is None:
            seq_len = stroke_params_list.size(1)

        rendered_images = self.neural_rasterizor.raster_func_abs(
            stroke_params_list,
            raster_seq_len=seq_len
        )

        return rendered_images.unsqueeze(1)

    def render_single_stroke(self, stroke_params: torch.Tensor) -> torch.Tensor:
        """
        渲染单个笔画
        Args:
            stroke_params: (B, 10)

        Returns:
            stroke_images: (B, 1, raster_size, raster_size)
        """
        stroke_params_expanded = stroke_params.unsqueeze(1)  # (B, 1, 10)
        return self.render_stroke_sequence(stroke_params_expanded, seq_len=1)


class SimpleRenderer(nn.Module):
    """
    简单的笔画渲染器（不使用神经网络，用于调试和快速原型）
    """

    def __init__(self, raster_size: int = 64):
        super().__init__()
        self.raster_size = raster_size

    def forward(self, stroke_params: torch.Tensor,
                cursor_pos: torch.Tensor,
                window_size: torch.Tensor) -> torch.Tensor:
        """
        简单的笔画渲染
        Args:
            stroke_params: (B, 7) - [x1, y1, x2, y2, width, pressure, eos]
            cursor_pos: (B, 2) - 光标位置
            window_size: (B,) - 窗口大小

        Returns:
            stroke_images: (B, 1, raster_size, raster_size)
        """
        batch_size = stroke_params.size(0)

        stroke_images = torch.zeros(
            batch_size, 1, self.raster_size, self.raster_size,
            device=stroke_params.device
        )

        # 简单的笔画绘制
        for i in range(batch_size):
            x1, y1, x2, y2 = stroke_params[i, :4]
            width = stroke_params[i, 4] * 10  # 宽度范围 0-10 像素
            pressure = stroke_params[i, 5]

            if stroke_params[i, 6] < 0.5:  # 如果不是结束笔画
                # 转换坐标到 [0, raster_size]
                x1 = (x1 + 1) * 0.5 * self.raster_size
                y1 = (y1 + 1) * 0.5 * self.raster_size
                x2 = (x2 + 1) * 0.5 * self.raster_size
                y2 = (y2 + 1) * 0.5 * self.raster_size

                # 绘制线条
                self._draw_line(
                    stroke_images[i, 0],
                    (x1.item(), y1.item()),
                    (x2.item(), y2.item()),
                    width.item(),
                    pressure.item()
                )

        return stroke_images

    def _draw_line(self, img: torch.Tensor,
                   start: Tuple[float, float],
                   end: Tuple[float, float],
                   width: float,
                   pressure: float):
        """
        绘制线条的Bresenham算法实现
        """
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while x0 != x1 or y0 != y1:
            if 0 <= x0 < self.raster_size and 0 <= y0 < self.raster_size:
                intensity = pressure
                img[y0, x0] = max(img[y0, x0], intensity)

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        if 0 <= x0 < self.raster_size and 0 <= y0 < self.raster_size:
            img[y0, x0] = max(img[y0, x0], pressure)


def create_renderer(raster_size: int,
                    neural: bool = True) -> nn.Module:
    """
    创建笔画渲染器
    Args:
        raster_size: 渲染图像大小
        neural: 是否使用神经渲染器

    Returns:
        renderer: 渲染器实例
    """
    if neural:
        return StrokeRenderer(raster_size=raster_size)
    else:
        return SimpleRenderer(raster_size=raster_size)
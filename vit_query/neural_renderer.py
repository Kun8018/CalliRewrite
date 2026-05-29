"""可微的神经渲染器 (RasterUnit) — 与 seq_extract 原版 raster_unit 同构。

输入: (N, 10) [x0, y0, x1, y1, x2, y2, r0, r2, w0, w2]，全部 ∈ [0, 1]
输出: (N, 128, 128) ∈ [0, 1]，1=stroke / 0=BG
（注意：原版 TF RasterUnit forward 末尾是 `1 - sigmoid`，这里我们直接返回 sigmoid，
 即 1=stroke。配套的 load_from_tf_ckpt 工具会在权重转换时处理这个翻转。）

预训练流程见 pretrain_renderer.py。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class RasterUnit(nn.Module):
    def __init__(self, raster_size: int = 128):
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

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """params: (N, 10) ∈ [0, 1]
        returns: (N, 128, 128) ∈ [0, 1], 1=stroke"""
        N = params.shape[0]
        x = F.relu(self.fc1(params))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(N, 16, 16, 16)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.pixel_shuffle(x, upscale_factor=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.pixel_shuffle(x, upscale_factor=2)

        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        x = F.pixel_shuffle(x, upscale_factor=2)

        x = torch.sigmoid(x).view(N, self.raster_size, self.raster_size)
        return x


class NeuralRasterizorStep(nn.Module):
    """对齐 seq_extract.rasterization_utils.NeuralRenderer.NeuralRasterizorStep。

    raster_unit 始终输出 128，外层需要别的分辨率就 interpolate。"""

    def __init__(self, raster_size: int = 128, pretrained_path: str = None,
                 freeze: bool = True):
        super().__init__()
        self.raster_size = raster_size
        self.raster_unit = RasterUnit(128)

        if pretrained_path is not None and os.path.isfile(pretrained_path):
            self.load_pretrained(pretrained_path)
            if freeze:
                for p in self.raster_unit.parameters():
                    p.requires_grad = False
                self.raster_unit.eval()

    def load_pretrained(self, path: str):
        sd = torch.load(path, map_location='cpu')
        if 'raster_unit' in sd:
            sd = sd['raster_unit']
        elif 'state_dict' in sd:
            sd = sd['state_dict']
        self.raster_unit.load_state_dict(sd, strict=True)
        print(f'[NeuralRasterizorStep] loaded raster_unit from {path}')

    def forward_stroke(self, params: torch.Tensor) -> torch.Tensor:
        """单步渲染：(N, 8 or 10) → (N, raster_size, raster_size) ∈ [0, 1], 1=stroke"""
        if params.shape[-1] == 8:
            w_in = torch.ones(params.shape[0], 2, device=params.device, dtype=params.dtype)
            params = torch.cat([params, w_in], dim=-1)
        img = self.raster_unit(params)
        if self.raster_size != 128:
            img = F.interpolate(img.unsqueeze(1),
                                size=(self.raster_size, self.raster_size),
                                mode='bilinear', align_corners=False).squeeze(1)
        return img

    def forward(self, strokes: torch.Tensor) -> torch.Tensor:
        """整段序列渲染（兼容旧接口）。
        strokes: (N, seq_len, 8 or 10) ∈ [0, 1]
        returns: (N, raster_size, raster_size) ∈ [0, 1], 1=stroke。"""
        N, T, D = strokes.shape
        canvas = torch.zeros(N, self.raster_size, self.raster_size,
                             device=strokes.device, dtype=strokes.dtype)
        for t in range(T):
            stroke_img = self.forward_stroke(strokes[:, t])  # (N, raster_size, raster_size)
            canvas = torch.clamp(canvas + stroke_img, 0.0, 1.0)
        return canvas


def seq7_to_absolute(strokes_seq7: torch.Tensor, img_size: int) -> torch.Tensor:
    """旧接口，保留以兼容旧 inference 路径。新版 rollout 不再使用。

    strokes_seq7: (N, seq_len, 7) — [pen, x1, y1, x2, y2, r, s]
    returns: (N, seq_len, 8) ∈ [0, 1] — [x0, y0, x1, y1, x2, y2, r0, r2]"""
    N, T, _ = strokes_seq7.shape
    device = strokes_seq7.device

    cursor = torch.full((N, 2), 0.5, device=device, dtype=strokes_seq7.dtype)
    prev_scaling = torch.ones(N, device=device, dtype=strokes_seq7.dtype)
    prev_window_size = torch.full((N,), float(min(128, img_size)),
                                  device=device, dtype=strokes_seq7.dtype)

    abs_strokes = []
    for i in range(T):
        stroke = strokes_seq7[:, i]
        pen = stroke[:, 0]
        curr_window_size = torch.clamp(prev_scaling * prev_window_size, 32.0, float(img_size))
        x0, y0 = cursor[:, 0], cursor[:, 1]
        x1 = x0 + stroke[:, 1] * curr_window_size / 2.0 / img_size
        y1 = y0 + stroke[:, 2] * curr_window_size / 2.0 / img_size
        x2 = x0 + stroke[:, 3] * curr_window_size / 2.0 / img_size
        y2 = y0 + stroke[:, 4] * curr_window_size / 2.0 / img_size
        r_abs = stroke[:, 5] * curr_window_size / img_size
        mask = (pen < 0.5).float()
        abs_stroke = torch.stack([
            x0 * mask, y0 * mask, x1 * mask, y1 * mask, x2 * mask, y2 * mask,
            r_abs * mask, r_abs * mask
        ], dim=1)
        abs_strokes.append(abs_stroke)
        cursor = torch.clamp(torch.stack([x2, y2], dim=1), 0.0, 1.0)
        prev_scaling = torch.clamp(stroke[:, 6], 0.05, 2.0)
        prev_window_size = curr_window_size
    return torch.stack(abs_strokes, dim=1)

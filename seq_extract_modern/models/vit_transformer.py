"""
ViT + Transformer 书法笔画提取模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class StrokePredictor(nn.Module):
    """
    笔画参数预测器
    接收 Transformer 输出并预测笔画参数
    """

    def __init__(self, hidden_dim: int = 768, output_dim: int = 7):
        """
        Args:
            hidden_dim: 输入特征维度
            output_dim: 输出维度（x1, y1, x2, y2, width, pressure, eos）
        """
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, hidden_dim)

        Returns:
            stroke_params: (B, seq_len, output_dim)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # 对输出进行适当的激活函数
        # x1, y1, x2, y2 应该在 [-1, 1] 范围内
        # width 和 pressure 应该在 [0, 1] 范围内
        # eos 应该是 sigmoid 激活

        x[:, :, :4] = torch.tanh(x[:, :, :4])  # x1, y1, x2, y2
        x[:, :, 4:6] = torch.sigmoid(x[:, :, 4:6])  # width, pressure
        x[:, :, 6] = torch.sigmoid(x[:, :, 6])  # eos

        return x


class NeuralRenderer(nn.Module):
    """
    神经渲染器
    将笔画参数转换为图像
    """

    def __init__(self, raster_size: int = 64):
        super().__init__()
        self.raster_size = raster_size

    def forward(self, stroke_params: torch.Tensor,
                cursor_pos: torch.Tensor,
                window_size: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stroke_params: (B, 7) - [x1, y1, x2, y2, width, pressure, eos]
            cursor_pos: (B, 2) - 光标位置
            window_size: (B,) - 窗口大小

        Returns:
            stroke_images: (B, 1, raster_size, raster_size)
        """
        batch_size = stroke_params.size(0)

        # 创建空白图像
        stroke_images = torch.zeros(
            batch_size, 1, self.raster_size, self.raster_size,
            device=stroke_params.device
        )

        # 计算笔画坐标（相对于光标位置）
        # stroke_params[:, :4] 在 [-1, 1] 范围内，需要转换到 [0, raster_size] 范围内
        stroke_coords = (stroke_params[:, :4] + 1.0) * 0.5  # 转换到 [0, 1]
        stroke_coords = stroke_coords * window_size.unsqueeze(1)  # 转换到窗口坐标
        stroke_coords = stroke_coords + cursor_pos.unsqueeze(1) * self.raster_size  # 加上光标偏移

        # 简单的笔画渲染（这里实现一个简化的版本）
        for i in range(batch_size):
            x1, y1, x2, y2 = stroke_coords[i]
            width = stroke_params[i, 4] * 5.0  # 宽度范围 0-5 像素
            pressure = stroke_params[i, 5]

            # 绘制线条
            if stroke_params[i, 6] < 0.5:  # 如果不是结束笔画
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
        绘制一条简单的线条
        """
        # 使用 Bresenham 算法或其他线条绘制算法
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while x0 != x1 or y0 != y1:
            if 0 <= x0 < self.raster_size and 0 <= y0 < self.raster_size:
                # 绘制像素，考虑压力
                intensity = pressure * 0.8  # 压力影响线条强度
                img[y0, x0] = max(img[y0, x0], intensity)

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        # 绘制终点
        if 0 <= x0 < self.raster_size and 0 <= y0 < self.raster_size:
            img[y0, x0] = max(img[y0, x0], pressure)


class ViTStrokeEncoder(nn.Module):
    """
    使用 ViT 作为图像编码器
    """

    def __init__(self, config):
        super().__init__()
        # 加载预训练的 ViT-B/16
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)
        # 移除分类头，保留特征提取部分
        self.vit.heads = nn.Identity()

        # 调整输入通道（支持灰度图）
        if config.num_channels == 1:
            self.vit.conv_proj = nn.Conv2d(
                in_channels=1,
                out_channels=self.vit.conv_proj.out_channels,
                kernel_size=self.vit.conv_proj.kernel_size,
                stride=self.vit.conv_proj.stride,
                padding=self.vit.conv_proj.padding
            )

        # 图像特征压缩
        self.feature_compressor = nn.Sequential(
            nn.Linear(768, config.vit_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.vit_hidden_dim, config.vit_hidden_dim)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 1, H, W)

        Returns:
            features: (B, vit_hidden_dim)
        """
        features = self.vit(images)
        features = self.feature_compressor(features)
        return features


class StrokeTransformerDecoder(nn.Module):
    """
    Transformer 解码器，用于预测笔画序列
    """

    def __init__(self, config):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(
            d_model=config.decoder_hidden_dim,
            nhead=config.decoder_num_heads,
            dim_feedforward=config.decoder_mlp_dim,
            dropout=0.1,
            activation='relu'
        )

        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_num_layers
        )

        # 位置编码
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_sequence_length, config.decoder_hidden_dim)
        )

        # 笔画参数预测器
        self.predictor = StrokePredictor(
            config.decoder_hidden_dim,
            config.stroke_params_dim
        )

    def forward(self, encoder_features: torch.Tensor,
                target_sequence: torch.Tensor = None,
                sequence_length: int = None) -> torch.Tensor:
        """
        Args:
            encoder_features: (B, vit_hidden_dim)
            target_sequence: (B, seq_len, decoder_hidden_dim)
            sequence_length: 生成的序列长度

        Returns:
            predictions: (B, seq_len, stroke_params_dim)
        """
        batch_size = encoder_features.size(0)

        if target_sequence is not None:
            # 训练模式
            seq_len = target_sequence.size(1)
            pos_encoding = self.pos_encoding[:, :seq_len]

            # 解码
            decoder_output = self.decoder(
                target_sequence,
                encoder_features.unsqueeze(1).repeat(1, seq_len, 1)
            )

            predictions = self.predictor(decoder_output)

        else:
            # 推理模式
            seq_len = sequence_length if sequence_length is not None else 100
            pos_encoding = self.pos_encoding[:, :seq_len]

            # 使用位置编码作为输入
            decoder_input = pos_encoding.repeat(batch_size, 1, 1)

            # 解码
            decoder_output = self.decoder(
                decoder_input,
                encoder_features.unsqueeze(1).repeat(1, seq_len, 1)
            )

            predictions = self.predictor(decoder_output)

        return predictions


class CalligraphyExtractor(nn.Module):
    """
    书法笔画序列提取模型
    包含：
    1. ViT 图像编码器
    2. Transformer 解码器
    3. 笔画参数预测器
    4. 神经渲染器
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = ViTStrokeEncoder(config)
        self.decoder = StrokeTransformerDecoder(config)
        self.renderer = NeuralRenderer(raster_size=config.raster_size)

    def forward(self, images: torch.Tensor,
                target_sequence: torch.Tensor = None,
                sequence_length: int = None) -> torch.Tensor:
        """
        Args:
            images: (B, 1, H, W)
            target_sequence: (B, seq_len, decoder_hidden_dim) - 训练时使用
            sequence_length: 推理时的序列长度

        Returns:
            stroke_params: (B, seq_len, stroke_params_dim)
        """
        # 编码图像特征
        encoder_features = self.encoder(images)

        # 解码笔画序列
        stroke_params = self.decoder(
            encoder_features,
            target_sequence,
            sequence_length
        )

        return stroke_params

    def extract_strokes(self, images: torch.Tensor,
                        sequence_length: int = 100) -> torch.Tensor:
        """
        简化的笔画提取接口
        Args:
            images: (B, 1, H, W)
            sequence_length: 要提取的笔画数量

        Returns:
            stroke_params: (B, sequence_length, 7)
        """
        self.eval()

        with torch.no_grad():
            stroke_params = self.forward(
                images,
                target_sequence=None,
                sequence_length=sequence_length
            )

        return stroke_params


def create_extractor_model(config) -> CalligraphyExtractor:
    """
    创建笔画提取模型
    """
    model = CalligraphyExtractor(config)

    # 初始化权重
    model.apply(_init_weights)

    return model


def _init_weights(module):
    """
    权重初始化
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
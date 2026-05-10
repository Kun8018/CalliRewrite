"""
PyTorch Lightning 训练模块
"""
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from models.vit_transformer import CalligraphyExtractor, create_extractor_model
from renderer.neural_renderer import create_renderer
from data.datasets import CalligraphyDataset, create_data_loader
from data.transforms import get_training_transforms, get_validation_transforms


class PerceptualLoss(nn.Module):
    """
    感知损失
    使用预训练模型提取特征进行对比
    """

    def __init__(self, layers=['conv1', 'conv2', 'conv3']):
        super().__init__()
        self.layers = layers
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                features_pred: Dict[str, torch.Tensor] = None,
                features_target: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """
        计算感知损失
        Args:
            pred: 预测图像 (B, 1, H, W)
            target: 目标图像 (B, 1, H, W)
            features_pred: 预测图像特征
            features_target: 目标图像特征

        Returns:
            loss: 感知损失
        """
        # 像素损失
        pixel_loss = self.mse_loss(pred, target)

        # 如果没有提供特征，只返回像素损失
        if features_pred is None or features_target is None:
            return pixel_loss

        # 感知损失
        perceptual_loss = 0.0
        for layer in self.layers:
            if layer in features_pred and layer in features_target:
                perceptual_loss += self.mse_loss(
                    features_pred[layer],
                    features_target[layer]
                )

        total_loss = pixel_loss + 0.1 * perceptual_loss

        return total_loss


class CalligraphyTrainer(pl.LightningModule):
    """
    书法笔画提取训练器
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # 初始化模型
        model_config = config['model'] if isinstance(config['model'], dict) else config.model
        self.model = create_extractor_model(model_config)

        # 初始化渲染器
        raster_size = config['model']['raster_size'] if isinstance(config['model'], dict) else config.model.raster_size
        self.renderer = create_renderer(
            raster_size,
            neural=False  # 初期使用简单渲染器，便于调试
        )

        # 损失函数
        self.pixel_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()

        # 指标
        self.train_psnr = PeakSignalNoiseRatio()
        self.val_psnr = PeakSignalNoiseRatio()
        self.train_ssim = StructuralSimilarityIndexMeasure()
        self.val_ssim = StructuralSimilarityIndexMeasure()

        # 自动混合精度
        use_amp = config['training']['use_amp'] if isinstance(config['training'], dict) else config.training.use_amp
        self.use_amp = use_amp

    def forward(self, images: torch.Tensor,
                target_sequence: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        """
        stroke_params = self.model(images, target_sequence)
        return stroke_params

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """
        训练步骤
        """
        images, initial_cursors = batch

        # 前向传播
        stroke_params = self.model(images)

        # 渲染笔画
        rendered_images = self._render_strokes(stroke_params, initial_cursors)

        # 计算损失
        loss = self._compute_loss(rendered_images, images)

        # 更新指标
        self.train_psnr(rendered_images, images)
        self.train_ssim(rendered_images, images)

        # 记录日志
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_psnr', self.train_psnr, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        """
        验证步骤
        """
        images, initial_cursors = batch

        # 前向传播
        stroke_params = self.model.extract_strokes(images, sequence_length=100)

        # 渲染笔画
        rendered_images = self._render_strokes(stroke_params, initial_cursors)

        # 计算损失
        loss = self._compute_loss(rendered_images, images)

        # 更新指标
        self.val_psnr(rendered_images, images)
        self.val_ssim(rendered_images, images)

        # 记录日志
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_psnr', self.val_psnr, prog_bar=True)

        # 保存示例图像
        if batch_idx == 0:
            self._save_example_images(images, rendered_images, 'val')

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, str],
                  batch_idx: int):
        """
        测试步骤
        """
        images, initial_cursors, filenames = batch

        # 提取笔画
        stroke_params = self.model.extract_strokes(images, sequence_length=100)

        # 渲染笔画
        rendered_images = self._render_strokes(stroke_params, initial_cursors)

        # 计算指标
        psnr = self.val_psnr(rendered_images, images)
        ssim = self.val_ssim(rendered_images, images)

        # 保存结果
        self._save_test_results(stroke_params, images, rendered_images, filenames)

        return {'psnr': psnr, 'ssim': ssim, 'filenames': filenames}

    def configure_optimizers(self) -> Tuple[list, list]:
        """
        配置优化器和学习率调度器
        """
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training'].learning_rate,
            betas=self.config['training'].betas,
            weight_decay=self.config['training'].weight_decay
        )

        # 余弦退火学习率调度器
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config['training'].max_epochs
        )

        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        """
        获取训练数据加载器
        """
        transform = get_training_transforms(self.config['training'])
        dataset = CalligraphyDataset(
            self.config['training'].train_data_dir,
            self.config['model'],
            transform,
            training=True
        )

        return create_data_loader(
            dataset,
            self.config['training'].batch_size,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        获取验证数据加载器
        """
        transform = get_validation_transforms(self.config['training'])
        dataset = CalligraphyDataset(
            self.config['training'].val_data_dir,
            self.config['model'],
            transform,
            training=False
        )

        return create_data_loader(
            dataset,
            self.config['training'].batch_size,
            shuffle=False
        )

    def _render_strokes(self, stroke_params: torch.Tensor,
                        initial_cursors: torch.Tensor) -> torch.Tensor:
        """
        渲染笔画序列
        """
        batch_size = stroke_params.size(0)
        rendered_images = []

        # 简化的渲染：渲染第一个笔画
        for i in range(batch_size):
            cursor_pos = initial_cursors[i]
            window_size = torch.tensor([self.config['model']['raster_size']],
                                       device=self.device).float()

            # 渲染每个笔画（这里简化为渲染第一个笔画）
            if stroke_params.size(1) > 0:
                stroke_param = stroke_params[i, 0:1]
                rendered_image = self.renderer(
                    stroke_param,
                    cursor_pos.unsqueeze(0),
                    window_size
                )
                rendered_images.append(rendered_image)

        if rendered_images:
            return torch.cat(rendered_images, dim=0)
        else:
            return torch.zeros(batch_size, 1, self.config['model']['raster_size'],
                               self.config['model']['raster_size'], device=self.device)

    def _compute_loss(self, rendered_images: torch.Tensor,
                      target_images: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        """
        # 确保尺寸一致
        if rendered_images.size()[2:] != target_images.size()[2:]:
            rendered_images = F.interpolate(
                rendered_images,
                size=target_images.size()[2:],
                mode='bilinear',
                align_corners=False
            )

        # 像素损失
        pixel_loss = self.pixel_loss(rendered_images, target_images)

        return pixel_loss

    def _save_example_images(self, target_images: torch.Tensor,
                             rendered_images: torch.Tensor,
                             prefix: str):
        """
        保存示例图像
        """
        import torchvision.utils as vutils

        save_dir = os.path.join(self.logger.log_dir, 'images')
        os.makedirs(save_dir, exist_ok=True)

        # 拼接图像
        combined = torch.cat([target_images[:4], rendered_images[:4]], dim=0)
        grid = vutils.make_grid(combined, nrow=4, normalize=True)

        vutils.save_image(
            grid,
            os.path.join(save_dir, f'{prefix}_epoch_{self.current_epoch}.png')
        )

    def _save_test_results(self, stroke_params: torch.Tensor,
                           target_images: torch.Tensor,
                           rendered_images: torch.Tensor,
                           filenames: list):
        """
        保存测试结果
        """
        save_dir = os.path.join(self.logger.log_dir, 'test_results')
        os.makedirs(save_dir, exist_ok=True)

        # 保存笔画参数
        for i, filename in enumerate(filenames):
            base_name = os.path.splitext(filename)[0]
            np.save(
                os.path.join(save_dir, f'{base_name}_strokes.npy'),
                stroke_params[i].cpu().numpy()
            )


def create_trainer(config) -> CalligraphyTrainer:
    """
    创建训练器实例
    """
    return CalligraphyTrainer(config)
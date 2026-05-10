"""
模型配置文件
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ViTTransformerConfig:
    """ViT + Transformer 模型配置"""

    # 图像参数
    image_size: int = 256
    patch_size: int = 16
    num_channels: int = 1

    # ViT 编码器参数
    vit_hidden_dim: int = 768
    vit_num_heads: int = 12
    vit_num_layers: int = 12
    vit_mlp_dim: int = 3072

    # Transformer 解码器参数
    decoder_num_layers: int = 6
    decoder_num_heads: int = 12
    decoder_hidden_dim: int = 768
    decoder_mlp_dim: int = 3072

    # 笔画参数
    stroke_params_dim: int = 7  # x1, y1, x2, y2, width, pressure, eos
    max_sequence_length: int = 500

    # 渲染参数
    raster_size: int = 64
    min_window_size: int = 64


@dataclass
class TrainingConfig:
    """训练配置"""

    # 基础训练参数
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    gradient_clip_val: float = 1.0

    # 优化器
    optimizer_type: str = "AdamW"
    betas: Tuple[float, float] = (0.9, 0.999)
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000

    # 数据增强
    use_augmentation: bool = True
    random_crop: bool = True
    random_rotate: bool = True

    # 损失函数权重
    pixel_loss_weight: float = 1.0
    perceptual_loss_weight: float = 100.0

    # 训练策略
    use_amp: bool = True  # 混合精度训练
    accumulate_grad_batches: int = 1
    precision: int = 16  # 16位混合精度

    # 检查点
    save_top_k: int = 3
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class InferenceConfig:
    """推理配置"""

    # 生成策略
    num_samples: int = 1
    temperature: float = 0.0  # 0.0 = 确定性

    # 停止条件
    round_stop_state_num: int = 12
    stroke_acc_threshold: float = 0.95
    max_rounds: int = 10

    # 输出
    save_images: bool = True
    visualize_strokes: bool = True


def get_default_config():
    """获取默认配置"""
    return {
        "model": ViTTransformerConfig(),
        "training": TrainingConfig(),
        "inference": InferenceConfig(),
    }
"""
训练脚本
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from configs.model_config import get_default_config
from trainer.training_module import create_trainer


def parse_args():
    parser = argparse.ArgumentParser(description='训练书法笔画提取模型')

    # 数据路径
    parser.add_argument('--train_data', type=str, required=True,
                        help='训练数据目录')
    parser.add_argument('--val_data', type=str, required=True,
                        help='验证数据目录')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='最大训练轮数')
    parser.add_argument('--gpus', type=int, default=1,
                        help='使用的 GPU 数量')

    # 保存和日志
    parser.add_argument('--save_dir', type=str, default='outputs',
                        help='保存目录')
    parser.add_argument('--project_name', type=str, default='calli_extract',
                        help='项目名称（用于日志）')
    parser.add_argument('--use_wandb', action='store_true',
                        help='使用 WandB 记录日志')

    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    config = get_default_config()

    # 更新配置
    config['training'].train_data_dir = args.train_data
    config['training'].val_data_dir = args.val_data
    config['training'].batch_size = args.batch_size
    config['training'].learning_rate = args.lr
    config['training'].max_epochs = args.max_epochs

    # 创建保存目录
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 初始化训练器
    model = create_trainer(config)

    # 回调函数
    callbacks = []

    # 模型检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, 'checkpoints'),
        filename='best_model',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_weights_only=True,
    )
    callbacks.append(checkpoint_callback)

    # 早停
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min',
        verbose=True
    )
    callbacks.append(early_stop_callback)

    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # 日志记录器
    if args.use_wandb:
        logger = WandbLogger(
            project=args.project_name,
            save_dir=save_dir,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=save_dir,
            name='logs'
        )

    # PyTorch Lightning 训练器
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=args.gpus if args.gpus > 0 else 'auto',
        callbacks=callbacks,
        logger=logger,
        precision=16 if config['training'].use_amp else 32,
        gradient_clip_val=config['training'].gradient_clip_val,
        accumulate_grad_batches=config['training'].accumulate_grad_batches,
        log_every_n_steps=10,
        val_check_interval=1.0,
    )

    # 开始训练
    print("开始训练...")
    trainer.fit(model)

    print(f"训练完成！最佳模型保存在: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()
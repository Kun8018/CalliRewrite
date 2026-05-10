"""
测试脚本
用于对单张图像或整个目录进行推理
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model_config import get_default_config
from inference.predictor import Predictor, create_predictor


def parse_args():
    parser = argparse.ArgumentParser(description='测试书法笔画提取模型')

    # 输入
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入图像路径或目录')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='模型权重路径')

    # 输出
    parser.add_argument('--output', '-o', type=str, default='outputs',
                        help='输出目录')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='保存可视化结果')

    # 参数
    parser.add_argument('--num_strokes', type=int, default=100,
                        help='提取的笔画数量')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备: auto, cpu, cuda, cuda:0 等')

    return parser.parse_args()


def main():
    args = parse_args()

    # 配置设备
    if args.device == 'auto':
        device = 'cuda' if sys.platform != 'darwin' and torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 初始化预测器
    config = get_default_config()

    predictor = create_predictor(
        model_path=args.model,
        config=config,
        device=device
    )

    # 处理输入
    if os.path.isdir(args.input):
        # 处理目录中的所有图像
        process_directory(args.input, output_dir, predictor, args)
    else:
        # 处理单个图像
        process_image(args.input, output_dir, predictor, args)

    print(f"处理完成！结果保存在: {output_dir}")


def process_image(image_path: str, output_dir: str,
                  predictor: Predictor, args):
    """
    处理单个图像
    """
    filename = os.path.basename(image_path)
    filename_without_ext = os.path.splitext(filename)[0]

    print(f"处理图像: {filename}")

    # 预测
    result = predictor.predict(image_path, args.num_strokes)

    # 保存结果
    output_npy = os.path.join(output_dir, f'{filename_without_ext}_strokes.npy')
    predictor.save_result(result, output_npy)

    # 保存可视化
    if args.visualize:
        output_png = os.path.join(output_dir, f'{filename_without_ext}_strokes.png')
        predictor.visualize_result(result, output_png)

    print(f"完成: {filename}")


def process_directory(input_dir: str, output_dir: str,
                      predictor: Predictor, args):
    """
    处理目录中的所有图像
    """
    supported_extensions = ['.png', '.jpg', '.jpeg']

    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            image_path = os.path.join(input_dir, filename)
            process_image(image_path, output_dir, predictor, args)


if __name__ == '__main__':
    import torch

    main()
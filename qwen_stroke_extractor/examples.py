#!/usr/bin/env python3
"""
基于千问Plus的图像到笔画提取示例
"""
import os
import argparse
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from qwen_stroke_extractor.extractor import (
    QwenStrokeExtractor,
    create_extractor,
    StrokeExtractionResult
)


def visualize_result(
    image_path: str,
    result: StrokeExtractionResult,
    output_path: str
):
    """
    可视化笔画提取结果
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    img = Image.open(image_path).convert('RGB')

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 显示原始图像
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # 显示笔画
    ax[1].imshow(img)
    ax[1].set_xlim(0, img.width)
    ax[1].set_ylim(img.height, 0)  # 反转 y 轴
    ax[1].set_aspect('equal')
    ax[1].set_title(f'Extracted Strokes ({len(result.strokes)})')

    # 绘制笔画
    for i, stroke in enumerate(result.strokes):
        # 坐标转换：归一化坐标 -> 图像坐标
        x1 = stroke.points[0].x * img.width
        y1 = stroke.points[0].y * img.height
        x2 = stroke.points[-1].x * img.width
        y2 = stroke.points[-1].y * img.height
        width = stroke.width * img.width * 0.5

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->',
            linewidth=width * 3,
            alpha=0.7,
            facecolor='red',
            edgecolor='black',
            label=f'Stroke {i+1}'
        )
        ax[1].add_patch(arrow)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"可视化结果保存到: {output_path}")


def run_example(args):
    """
    运行示例
    """
    # 初始化提取器
    try:
        extractor = create_extractor(
            model_path=args.model_path,
            use_api=args.use_api,
            api_key=args.api_key
        )
    except Exception as e:
        print(f"初始化提取器失败: {e}")
        return False

    # 提取笔画
    try:
        print(f"正在分析图像: {args.input_image}")
        result = extractor.extract(args.input_image)
        print(f"成功识别 {len(result.strokes)} 个笔画")

        # 保存结果
        save_dir = Path(args.output_dir)
        save_dir.mkdir(exist_ok=True)

        # 保存为 numpy 格式
        stroke_params = extractor.to_numpy(result, output_format='params')
        np_output_path = save_dir / 'strokes_params.npy'
        stroke_params.tofile(str(np_output_path))
        print(f"笔画参数保存到: {np_output_path}")

        # 保存为 stroke3 格式（与原项目兼容）
        stroke3_data = extractor.to_numpy(result, output_format='stroke3')
        stroke3_output_path = save_dir / 'strokes_stroke3.npy'
        stroke3_data.tofile(str(stroke3_output_path))
        print(f"Stroke3 格式保存到: {stroke3_output_path}")

        # 保存为 JSON 格式
        json_output_path = save_dir / 'strokes.json'
        extractor.save_result(result, str(json_output_path), format='json')
        print(f"JSON 格式保存到: {json_output_path}")

        # 可视化（如果需要）
        if args.visualize:
            viz_path = save_dir / 'visualization.png'
            visualize_result(args.input_image, result, str(viz_path))

        # 打印一些统计信息
        print("\n统计信息:")
        print(f"笔画数量: {len(result.strokes)}")
        if result.strokes:
            avg_width = sum(stroke.width for stroke in result.strokes) / len(result.strokes)
            print(f"平均笔画宽度: {avg_width:.4f}")

        print("\n提取完成！")
        return True

    except Exception as e:
        print(f"提取过程中出错: {e}")
        if args.verbose:
            import traceback
            print(f"详细错误信息:\n{traceback.format_exc()}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="基于千问Plus的图像到笔画提取工具"
    )

    # 输入参数
    parser.add_argument(
        "input_image",
        help="输入图像路径"
    )

    # 输出参数
    parser.add_argument(
        "-o", "--output-dir",
        default="outputs/qwen_results",
        help="输出目录 (默认: outputs/qwen_results)"
    )

    # 模型参数
    parser.add_argument(
        "--model-path",
        help="本地模型路径 (如果未指定，将自动下载)"
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="使用API模式(需要配置API密钥)"
    )
    parser.add_argument(
        "--api-key",
        help="API密钥 (使用API模式时必需)"
    )

    # 可视化参数
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="可视化提取结果"
    )

    # 其他参数
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细信息"
    )

    args = parser.parse_args()

    # 参数验证
    if args.use_api and not args.api_key:
        parser.error("使用API模式时需要提供 --api-key 参数")

    # 运行示例
    if run_example(args):
        return 0
    else:
        return 1


if __name__ == "__main__":
    main()

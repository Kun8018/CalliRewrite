#!/usr/bin/env python3
"""
可视化模块：生成 order、color、compare 等图
"""
import os
import numpy as np
from PIL import Image, ImageDraw


def get_colors(n):
    """生成渐变颜色"""
    colors = []
    for i in range(n):
        ratio = i / max(n-1, 1)
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        b = 0
        colors.append((r, g, b))
    return colors


def draw_stroke(draw, x0, y0, x1, y1, x2, y2, r, color, width=3):
    """绘制贝塞尔曲线"""
    # 二次贝塞尔曲线采样
    points = []
    for t in np.linspace(0, 1, 20):
        x = (1-t)**2 * x0 + 2*(1-t)*t * x1 + t**2 * x2
        y = (1-t)**2 * y0 + 2*(1-t)*t * y1 + t**2 * y2
        points.append((x, y))

    # 绘制线段
    for i in range(len(points)-1):
        draw.line([points[i], points[i+1]], fill=color, width=width)


def strokes_to_absolute(strokes_data):
    """把相对坐标转为绝对坐标（简化版）"""
    # 这个是简化版，实际需要更复杂的逻辑
    # 这里只是为了可视化演示
    points = []
    current_pos = np.array([0.5, 0.5])

    for stroke in strokes_data:
        pen_state = stroke[0]
        x1, y1 = stroke[1], stroke[2]
        x2, y2 = stroke[3], stroke[4]

        if pen_state == 0:
            # 绘画：绘制贝塞尔曲线
            points.append({
                'type': 'draw',
                'x0': current_pos[0],
                'y0': current_pos[1],
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
            current_pos = np.array([x2, y2])
        else:
            # 移动
            points.append({
                'type': 'move',
                'x0': current_pos[0],
                'y0': current_pos[1],
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
            current_pos = np.array([x2, y2])

    return points


def generate_order_image(strokes_data, image_size=256, line_width=3):
    """生成 order 图：每笔用不同颜色显示顺序"""
    img = Image.new('RGB', (image_size, image_size), 'white')
    draw = ImageDraw.Draw(img)

    colors = get_colors(len(strokes_data))
    current_pos = np.array([image_size/2, image_size/2])
    color_idx = 0

    for stroke in strokes_data:
        pen_state = stroke[0]
        x1_rel, y1_rel = stroke[1], stroke[2]
        x2_rel, y2_rel = stroke[3], stroke[4]

        # 转换到像素坐标（简化）
        x0, y0 = current_pos
        x1 = x0 + x1_rel * image_size/2
        y1 = y0 + y1_rel * image_size/2
        x2 = x0 + x2_rel * image_size/2
        y2 = y0 + y2_rel * image_size/2

        if pen_state == 0:
            # 绘制
            color = colors[color_idx % len(colors)]
            # 用线代替贝塞尔
            draw.line([(x0, y0), (x2, y2)], fill=color, width=line_width)
            color_idx += 1

        current_pos = np.array([x2, y2])

    return img


def generate_color_image(strokes_data, current_stroke_idx, image_size=256, line_width=3):
    """生成 color 图：红色标注当前笔"""
    img = Image.new('RGB', (image_size, image_size), 'white')
    draw = ImageDraw.Draw(img)

    current_pos = np.array([image_size/2, image_size/2])

    for i, stroke in enumerate(strokes_data):
        pen_state = stroke[0]
        x1_rel, y1_rel = stroke[1], stroke[2]
        x2_rel, y2_rel = stroke[3], stroke[4]

        x0, y0 = current_pos
        x1 = x0 + x1_rel * image_size/2
        y1 = y0 + y1_rel * image_size/2
        x2 = x0 + x2_rel * image_size/2
        y2 = y0 + y2_rel * image_size/2

        if pen_state == 0:
            # 当前笔用红色，其他用灰色
            if i == current_stroke_idx:
                color = (255, 0, 0)
            else:
                color = (128, 128, 128)
            draw.line([(x0, y0), (x2, y2)], fill=color, width=line_width)

        current_pos = np.array([x2, y2])

    return img


def generate_compare_image(original_img, generated_img, output_size=(1000, 500)):
    """生成 compare 图：左右对比"""
    w, h = output_size
    img_w, img_h = w//2, h

    # 调整大小
    original_resized = original_img.resize((img_w, img_h))
    generated_resized = generated_img.resize((img_w, img_h))

    # 拼接
    compare_img = Image.new('RGB', output_size, 'white')
    compare_img.paste(original_resized, (0, 0))
    compare_img.paste(generated_resized, (img_w, 0))

    # 添加文字
    draw = ImageDraw.Draw(compare_img)
    draw.text((10, 10), "Original", fill=(0, 0, 0))
    draw.text((img_w + 10, 10), "Generated", fill=(0, 0, 0))

    return compare_img


def visualize_strokes(npz_path, original_img_path=None, output_dir=None):
    """完整的可视化流程"""
    if output_dir is None:
        base_dir = os.path.dirname(npz_path)
        base_name = os.path.basename(npz_path)[:-4]
        output_dir = os.path.join(base_dir, base_name)

    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    data = np.load(npz_path, encoding='latin1', allow_pickle=True)
    strokes_data = data['strokes_data']

    # 生成 order 图
    order_img = generate_order_image(strokes_data, image_size=256)
    order_path = os.path.join(output_dir, 'order.png')
    order_img.save(order_path)
    print(f'Saved: {order_path}')

    # 生成 color 图（每一步）
    color_dir = os.path.join(output_dir, 'color')
    os.makedirs(color_dir, exist_ok=True)

    for i in range(len(strokes_data)):
        color_img = generate_color_image(strokes_data, i, image_size=256)
        color_path = os.path.join(color_dir, f'{i:03d}.png')
        color_img.save(color_path)
    print(f'Saved: {len(strokes_data)} color images')

    # 生成 compare 图（如果有原图）
    if original_img_path and os.path.exists(original_img_path):
        original_img = Image.open(original_img_path).convert('RGB')
        generated_img = generate_order_image(strokes_data, image_size=256)
        compare_img = generate_compare_image(original_img, generated_img)
        compare_path = os.path.join(output_dir, 'compare.png')
        compare_img.save(compare_path)
        print(f'Saved: {compare_path}')

    return output_dir


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, required=True, help='npz 文件路径')
    parser.add_argument('--original', type=str, default=None, help='原始图像路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    args = parser.parse_args()

    visualize_strokes(args.npz, args.original, args.output)

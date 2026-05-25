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


def normalize_round_lengths(round_lengths, stroke_count):
    round_lengths = np.asarray(round_lengths, dtype=np.int64).reshape(-1)
    round_lengths = round_lengths[round_lengths > 0]
    if len(round_lengths) == 0:
        return np.array([stroke_count], dtype=np.int64)

    normalized = []
    remaining = int(stroke_count)
    for length in round_lengths:
        if remaining <= 0:
            break
        current = min(int(length), remaining)
        normalized.append(current)
        remaining -= current
    if remaining > 0:
        normalized.append(remaining)
    return np.asarray(normalized, dtype=np.int64)


def iter_stroke_points(strokes_data, init_cursors, round_lengths, image_size):
    stroke_idx = 0
    for round_idx, round_length in enumerate(round_lengths):
        cursor = np.array(init_cursors[min(round_idx, len(init_cursors) - 1)], dtype=np.float32)
        prev_scaling = 1.0
        prev_window_size = min(128.0, float(image_size))
        for _ in range(int(round_length)):
            if stroke_idx >= len(strokes_data):
                return
            stroke = strokes_data[stroke_idx]
            curr_window_size = float(np.clip(prev_scaling * prev_window_size, 32.0, image_size))
            x0 = float(cursor[0]) * image_size
            y0 = float(cursor[1]) * image_size
            x1 = x0 + float(stroke[1]) * curr_window_size / 2.0
            y1 = y0 + float(stroke[2]) * curr_window_size / 2.0
            x2 = x0 + float(stroke[3]) * curr_window_size / 2.0
            y2 = y0 + float(stroke[4]) * curr_window_size / 2.0
            points = []
            for t in np.linspace(0.0, 1.0, 16):
                x = (1 - t) * (1 - t) * x0 + 2 * (1 - t) * t * x1 + t * t * x2
                y = (1 - t) * (1 - t) * y0 + 2 * (1 - t) * t * y1 + t * t * y2
                points.append((float(np.clip(x, 0, image_size - 1)), float(np.clip(y, 0, image_size - 1))))
            yield stroke_idx, stroke, points
            cursor = np.clip(
                cursor + np.array([stroke[3], stroke[4]], dtype=np.float32) * curr_window_size / 2.0 / float(image_size),
                0.0,
                (image_size - 1) / float(image_size)
            )
            prev_scaling = float(np.clip(stroke[6], 0.05, 2.0))
            prev_window_size = curr_window_size
            stroke_idx += 1


def generate_order_image(strokes_data, image_size=256, line_width=3, init_cursors=None, round_lengths=None):
    """生成 order 图：每笔用不同颜色显示顺序"""
    img = Image.new('RGB', (image_size, image_size), 'white')
    draw = ImageDraw.Draw(img)
    init_cursors = np.asarray(init_cursors if init_cursors is not None else [[0.5, 0.5]], dtype=np.float32)
    round_lengths = normalize_round_lengths(round_lengths if round_lengths is not None else [len(strokes_data)], len(strokes_data))
    colors = get_colors(len(strokes_data))
    color_idx = 0

    for _, stroke, points in iter_stroke_points(strokes_data, init_cursors, round_lengths, image_size):
        if stroke[0] == 0:
            color = colors[color_idx % len(colors)]
            draw.line(points, fill=color, width=line_width)
            color_idx += 1

    return img


def generate_color_image(strokes_data, current_stroke_idx, image_size=256, line_width=3,
                         init_cursors=None, round_lengths=None):
    """生成 color 图：红色标注当前笔"""
    img = Image.new('RGB', (image_size, image_size), 'white')
    draw = ImageDraw.Draw(img)
    init_cursors = np.asarray(init_cursors if init_cursors is not None else [[0.5, 0.5]], dtype=np.float32)
    round_lengths = normalize_round_lengths(round_lengths if round_lengths is not None else [len(strokes_data)], len(strokes_data))

    for i, stroke, points in iter_stroke_points(strokes_data, init_cursors, round_lengths, image_size):
        if stroke[0] == 0:
            color = (255, 0, 0) if i == current_stroke_idx else (128, 128, 128)
            draw.line(points, fill=color, width=line_width)

    return img


def generate_compare_image(original_img, generated_img, output_size=(1000, 500)):
    """生成 compare 图：左右对比"""
    w, h = output_size
    img_w, img_h = w//2, h

    original_resized = original_img.resize((img_w, img_h))
    generated_resized = generated_img.resize((img_w, img_h))

    compare_img = Image.new('RGB', output_size, 'white')
    compare_img.paste(original_resized, (0, 0))
    compare_img.paste(generated_resized, (img_w, 0))

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

    data = np.load(npz_path, encoding='latin1', allow_pickle=True)
    strokes_data = data['strokes_data']
    init_cursors = data['init_cursors'] if 'init_cursors' in data else np.array([[0.5, 0.5]], dtype=np.float32)
    round_lengths = data['round_length'] if 'round_length' in data else np.array([len(strokes_data)], dtype=np.int64)

    # Order 图
    order_img = generate_order_image(strokes_data, image_size=256, init_cursors=init_cursors, round_lengths=round_lengths)
    order_path = os.path.join(output_dir, 'order.png')
    order_img.save(order_path)
    print(f'Saved: {order_path}')

    # Color 图（每一步）
    color_dir = os.path.join(output_dir, 'color')
    os.makedirs(color_dir, exist_ok=True)

    for i in range(len(strokes_data)):
        color_img = generate_color_image(
            strokes_data, i, image_size=256,
            init_cursors=init_cursors, round_lengths=round_lengths
        )
        color_path = os.path.join(color_dir, f'{i:03d}.png')
        color_img.save(color_path)
    print(f'Saved: {len(strokes_data)} color images')

    # Compare 图
    if original_img_path and os.path.exists(original_img_path):
        original_img = Image.open(original_img_path).convert('RGB')
        generated_img = generate_order_image(
            strokes_data, image_size=256,
            init_cursors=init_cursors, round_lengths=round_lengths
        )
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

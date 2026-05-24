#!/usr/bin/env python3
"""
Download QuickDraw SketchRNN data and arrange local train/test files for seq_extract phase 1.
"""
import argparse
import os
import urllib.error
import urllib.parse
import urllib.request

import numpy as np


CATEGORIES = [
    'airplane', 'bus', 'car', 'sailboat', 'bird', 'cat', 'dog',
    'tree', 'flower', 'zigzag'
]

GCS_BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/sketchrnn'
HF_DATASET = 'quick_draw'
HF_CONFIG = 'sketchrnn'
MODELSCOPE_DATASET = 'quick_draw'
MODELSCOPE_CONFIG = 'sketchrnn'


def parse_args():
    parser = argparse.ArgumentParser(description='Download QuickDraw-clean data for seq_extract')
    default_output_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'QuickDraw-clean')
    parser.add_argument('--source', type=str, default='gcs', choices=['gcs', 'hf', 'modelscope', 'googlecreativelab'],
                        help='下载来源：gcs=Google Storage 直链，hf=Hugging Face datasets，modelscope=ModelScope datasets，googlecreativelab=官方项目直链')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                        help='输出目录，默认是 seq_extract/datasets/QuickDraw-clean')
    parser.add_argument('--categories', nargs='*', default=CATEGORIES,
                        help='要下载的类别名')
    parser.add_argument('--splits', nargs='*', default=['train', 'test'], choices=['train', 'test', 'valid'],
                        help='要下载的数据划分')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在文件')
    parser.add_argument('--hf_streaming', action='store_true', help='Hugging Face 使用流式加载')
    parser.add_argument('--hf_max_items', type=int, default=None,
                        help='Hugging Face 每个 split 最多保存多少条，调试用')
    parser.add_argument('--modelscope_dataset', type=str, default=MODELSCOPE_DATASET,
                        help='ModelScope 数据集 ID，默认 quick_draw')
    parser.add_argument('--modelscope_config', type=str, default=MODELSCOPE_CONFIG,
                        help='ModelScope 配置名，默认 sketchrnn')
    parser.add_argument('--modelscope_max_items', type=int, default=None,
                        help='ModelScope 每个 split 最多保存多少条，调试用')
    parser.add_argument('--invert_pen_state', action='store_true',
                        help='如果第三方数据源使用 1=draw, 0=end，则打开此选项转换为 seq_extract 的 0=draw, 1=end')
    return parser.parse_args()


def download_file(url, output_path, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        print(f'Skip existing: {output_path}')
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + '.tmp'
    print(f'Downloading: {url}')

    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request) as response, open(tmp_path, 'wb') as f:
            total = response.headers.get('Content-Length')
            total = int(total) if total else None
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 / total
                    print(f'  {downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB ({percent:.1f}%)', end='\r')
            if total:
                print()
        os.replace(tmp_path, output_path)
        print(f'Saved: {output_path}')
    except urllib.error.HTTPError as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f'Failed: {url} ({e.code})')
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_official_split_npz(output_path, stroke3_data, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        print(f'Skip existing: {output_path}')
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, stroke3=stroke3_data)
    print(f'Saved: {output_path} ({len(stroke3_data)} samples)')


def save_official_sketchrnn_npz(raw_path, category, splits, output_dir, overwrite=False):
    data = np.load(raw_path, allow_pickle=True, encoding='latin1')
    for split in splits:
        source_key = 'valid' if split == 'valid' else split
        if source_key not in data:
            print(f'Skip {category}/{split}: missing key {source_key} in {raw_path}')
            continue
        output_path = os.path.join(output_dir, split, f'{category}.npz')
        save_official_split_npz(output_path, data[source_key], overwrite=overwrite)


def download_from_gcs(args):
    raw_dir = os.path.join(args.output_dir, '_raw_sketchrnn')
    os.makedirs(raw_dir, exist_ok=True)
    for category in args.categories:
        filename = f'{category}.npz'
        url = f'{GCS_BASE_URL}/{urllib.parse.quote(filename)}'
        raw_path = os.path.join(raw_dir, filename)
        download_file(url, raw_path, overwrite=args.overwrite)
        if os.path.exists(raw_path):
            save_official_sketchrnn_npz(raw_path, category, args.splits, args.output_dir, args.overwrite)


def download_from_googlecreativelab(args):
    print('Using googlecreativelab/quickdraw-dataset documented storage URLs.')
    print('The official project points to the same public QuickDraw Google Storage bucket.')
    download_from_gcs(args)


def normalize_hf_split(split):
    if split == 'test':
        return 'test'
    if split == 'valid':
        return 'validation'
    return 'train'


def get_example_category(example):
    for key in ('word', 'category', 'class', 'label'):
        if key in example:
            value = example[key]
            if isinstance(value, str):
                return value.replace(' ', '_')
            return str(value)
    return None


def get_example_strokes(example):
    for key in ('drawing', 'stroke3', 'strokes', 'ink'):
        if key in example:
            return example[key]
    raise KeyError(f'Cannot find strokes field in example keys: {list(example.keys())}')


def normalize_pen_state(stroke3, invert_pen_state=False):
    stroke3 = np.asarray(stroke3, dtype=np.float32)
    if stroke3.ndim == 2 and stroke3.shape[1] == 3 and invert_pen_state:
        stroke3 = stroke3.copy()
        stroke3[:, 2] = 1.0 - stroke3[:, 2]
    return stroke3


def convert_strokes_to_stroke3(strokes, invert_pen_state=False):
    arr = np.asarray(strokes, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return normalize_pen_state(arr, invert_pen_state)

    result = []
    last_x = 0.0
    last_y = 0.0

    for stroke in strokes:
        if len(stroke) != 2:
            continue
        xs, ys = stroke
        for i, (x, y) in enumerate(zip(xs, ys)):
            pen_state = 1.0 if i == len(xs) - 1 else 0.0
            result.append([float(x) - last_x, float(y) - last_y, pen_state])
            last_x = float(x)
            last_y = float(y)

    if not result:
        return np.zeros((0, 3), dtype=np.float32)
    return normalize_pen_state(np.asarray(result, dtype=np.float32), invert_pen_state)


def save_stroke3_npz(output_path, stroke3_list, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        print(f'Skip existing: {output_path}')
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = np.asarray(stroke3_list, dtype=object)
    np.savez_compressed(output_path, stroke3=data, train=data, test=data)
    print(f'Saved: {output_path} ({len(stroke3_list)} samples)')


def save_examples_by_category(examples, categories, max_items, output_dir, split, overwrite, invert_pen_state=False):
    category_set = set(categories)
    grouped = {category: [] for category in categories}

    for example in examples:
        category = get_example_category(example)
        if category not in category_set:
            continue
        grouped[category].append(convert_strokes_to_stroke3(get_example_strokes(example), invert_pen_state))
        if max_items is not None and all(len(v) >= max_items for v in grouped.values()):
            break

    split_dir = os.path.join(output_dir, split)
    for category, stroke3_list in grouped.items():
        if max_items is not None:
            stroke3_list = stroke3_list[:max_items]
        output_path = os.path.join(split_dir, f'{category}.npz')
        save_stroke3_npz(output_path, stroke3_list, overwrite=overwrite)


def download_from_hf(args):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError('Hugging Face source requires: pip install datasets') from e

    for split in args.splits:
        hf_split = normalize_hf_split(split)
        print(f'Loading Hugging Face dataset: {HF_DATASET}/{HF_CONFIG} split={hf_split}')
        dataset = load_dataset(HF_DATASET, HF_CONFIG, split=hf_split, streaming=args.hf_streaming)
        save_examples_by_category(
            dataset, args.categories, args.hf_max_items,
            args.output_dir, split, args.overwrite, args.invert_pen_state
        )


def normalize_modelscope_split(split):
    if split == 'valid':
        return 'validation'
    return split


def iter_modelscope_dataset(dataset):
    if isinstance(dataset, dict):
        for value in dataset.values():
            yield from iter_modelscope_dataset(value)
        return
    for example in dataset:
        yield example


def download_from_modelscope(args):
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as e:
        raise ImportError('ModelScope source requires: pip install modelscope') from e

    for split in args.splits:
        ms_split = normalize_modelscope_split(split)
        print(f'Loading ModelScope dataset: {args.modelscope_dataset}/{args.modelscope_config} split={ms_split}')
        dataset = MsDataset.load(
            args.modelscope_dataset,
            subset_name=args.modelscope_config,
            split=ms_split
        )
        save_examples_by_category(
            iter_modelscope_dataset(dataset), args.categories, args.modelscope_max_items,
            args.output_dir, split, args.overwrite, args.invert_pen_state
        )


def main():
    args = parse_args()

    if args.source == 'gcs':
        download_from_gcs(args)
    elif args.source == 'googlecreativelab':
        download_from_googlecreativelab(args)
    elif args.source == 'hf':
        download_from_hf(args)
    else:
        download_from_modelscope(args)

    print('\nDone.')
    print(f'Output: {args.output_dir}')
    print('Expected by seq_extract locally: datasets/QuickDraw-clean/train/*.npz and test/*.npz')


if __name__ == '__main__':
    main()

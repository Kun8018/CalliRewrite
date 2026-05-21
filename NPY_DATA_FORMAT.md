# 一阶段（seq_extract）输出数据格式详解

## 概述

本项目中，一阶段（序列提取）生成的 `npz` 文件是连接一阶段和二阶段（强化学习微调）的关键数据格式。

## 文件类型说明

| 阶段 | 文件扩展名 | 说明 |
|-----|-----------|------|
| 一阶段（seq_extract）输出 | `.npz` | 原始笔画数据，包含7列信息 |
| 二阶段（rl_finetune）输入/输出 | `.npy` | 优化后的笔画数据，包含工具属性 |

## 一阶段 .npz 文件详解

### 文件内容

使用 `np.load(filename, allow_pickle=True)` 加载后，文件包含以下键：

```python
data = np.load('xxxx.npz', encoding='latin1', allow_pickle=True)
data.keys()  # ['strokes_data', 'init_cursors', 'image_size', 'round_length', 'init_width']
```

### 1. strokes_data - 核心笔画数据

**形状**: `(N_strokes, 7)` 或 `(N_strokes, 9)`

**数据格式**:
```
[pen_state, x1, y1, x2, y2, radius, scaling]
或
[pen_state, x1, y1, x2, y2, width2, scaling2]
```

**字段说明**:

| 列索引 | 名称 | 范围 | 说明 |
|-------|-----|------|------|
| 0 | pen_state | 0 或 1 | 笔状态: 0=绘画, 1=移动 |
| 1-2 | (x1, y1) | [-1, 1] | 贝塞尔曲线控制点1（相对偏移） |
| 3-4 | (x2, y2) | [-1, 1] | 贝塞尔曲线终点（相对偏移） |
| 5 | radius/width2 | float | 终点笔刷宽度 |
| 6 | scaling2 | float | 下一个窗口缩放因子 |

**注意**: 这里的 (x1, y1) 是相对于起点和终点的偏移量，不是绝对坐标。计算方式如下：

```python
# 相对坐标转绝对坐标（简化版）
x0y0 = np.array([0, 0])  # 起点归一化位置
x2y2 = stroke_params[3:5]  # 终点
x1y1 = x0y0 + (x2y2 - x0y0) * stroke_params[1:3]  # 控制点
```

### 2. init_cursors - 初始光标位置

**形状**: `(M_rounds, 2)` 或类似

**说明**: 每轮绘画开始时的光标位置，归一化到 [0, 1] 范围

### 3. image_size - 图像尺寸

**类型**: int

**说明**: 原始图像尺寸，通常 256

### 4. round_length - 每轮步数

**形状**: `(M_rounds,)`

**说明**: 每轮采样的步数

### 5. init_width - 初始宽度

**类型**: float

**说明**: 初始笔画宽度

## 数据处理流程

### 一阶段 → 二阶段转换过程

在 `rl_finetune/Callienv/envs/skel_utils.py` 中进行处理：

1. **加载原始数据**:
```python
npz_data = np.load('xxxx.npz', encoding='latin1', allow_pickle=True)
```

2. **全局坐标转换**: `make_global_nplist()`
- 把相对坐标转换为绝对图像坐标
- 计算出实际的 (x, y) 位置

3. **采样成密集点**: `parse_skel()`
- 把贝塞尔曲线采样成连续的点
- 格式: `[pen_state, x, y]`

4. **二阶段内部处理**: `add_beg_end_seq()`
- 添加笔画首尾的额外控制点
- 用于笔刷效果渲染

## 二阶段处理后的格式

在二阶段的 `CalliEnv` 中，数据被转换为以下格式:

**形状**: `(N_points, 7)`

```python
[pen_state, x, y, r, color1, color2, color3]
# pen_state: 0=继续, 1=新笔画开始
# x, y: 像素坐标 [0, 256] (或其他图像尺寸)
# r: 笔刷半径，像素单位
```

**注意**: 经过 `add_beg_end_seq()` 处理后，坐标从归一化的 [0,1] 被缩放回了像素值 [0, 256]！


## 实际数据示例

假设我们有一个简单的L形笔画：

```python
# 原始 npz 中的 strokes_data 示例
strokes_data = np.array([
    # 笔画1: 横线
    [0, 0.5, 0.5, 1.0, 0.0, 0.05, 1.0],  # 从左向右画
    [1, 0.0, 0.0, 0.0, 1.0, 0.05, 1.0],  # 移动到下一个笔画起点
    # 笔画2: 竖线
    [0, 0.5, 0.5, 0.0, 1.0, 0.05, 1.0],
])
```

经过处理后变成密集点:

```python
# 处理后的格式 (N_points, 3) -> [pen_state, x, y]
processed = np.array([
    [1, 0.1, 0.5],  # 新笔画开始
    [0, 0.2, 0.5],  # 绘画
    [0, 0.3, 0.5],
    ...
    [0, 0.9, 0.5],
    [1, 0.9, 0.5],  # 新笔画开始
    [0, 0.9, 0.6],
    ...
])
```

## 相关代码位置

| 功能 | 文件位置 | 函数名 |
|-----|---------|-------|
| 保存npz | `seq_extract/utils.py` | `save_seq_data()` |
| 加载和转换 | `rl_finetune/Callienv/envs/skel_utils.py` | `transfer_data()` |
| 全局坐标转换 | `rl_finetune/Callienv/envs/skel_utils.py` | `make_global_nplist()` |
| 采样贝塞尔 | `rl_finetune/Callienv/envs/skel_utils.py` | `parse_skel()` |

## 如何在代码中使用

### 加载并查看数据

```python
import numpy as np

# 加载一阶段输出
npz_path = 'path/to/output.npz'
data = np.load(npz_path, encoding='latin1', allow_pickle=True)

# 获取笔画数据
strokes = data['strokes_data']

# 遍历所有笔画
for stroke in strokes:
    pen_state = stroke[0]
    if pen_state == 0:
        print(f"绘画: 控制点={stroke[1:3]}, 终点={stroke[3:5]}")
    else:
        print(f"移动到下一笔")
```

## 可视化数据

可以使用项目中的工具可视化:

```python
from seq_extract.utils import draw_strokes
# 需要 tf 和相关组件...
```

## 总结

一阶段的输出是用相对坐标表示的贝塞尔笔画序列，
二阶段把它转换成工具可以操作的绝对坐标密集点，并强化学习优化工具参数。

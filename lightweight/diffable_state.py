"""可微的 stroke rollout 状态更新。

对齐 seq_extract/model_common_train.py 的 get_points_and_raster_image：
- cursor / window_size / canvas / prev_width / prev_scaling 全部是 torch tensor，可反传；
- 每步用预训练 NeuralRasterizor 渲染单步贝塞尔笔画，paste 到 canvas；
- pos_before_max_min / win_size_before_max_min 暴露出来供 outside_loss 用。
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F


MIN_WINDOW_SIZE = 32.0
MAX_SCALING = 2.0
MIN_WIDTH = 0.01


@dataclass
class RolloutState:
    cursor: torch.Tensor          # (N, 2)   in [0, 1)
    canvas: torch.Tensor          # (N, 1, H, W) in [0, 1], 1=stroke
    prev_width: torch.Tensor      # (N, 1)   in [min_width, 1]
    prev_scaling: torch.Tensor    # (N, 1)   in [0, max_scaling]
    prev_window_size: torch.Tensor  # (N, 1) in [min_window_size, image_size]
    prev_stroke: torch.Tensor     # (N, 7)
    img_size: int


def init_rollout_state(batch_size: int, img_size: int, device, dtype=torch.float32) -> RolloutState:
    """对齐 seq_extract: init_cursor=(0.5,0.5), init_width=0.1, init_scaling=1.0, init_window=raster_size."""
    return RolloutState(
        cursor=torch.full((batch_size, 2), 0.5, device=device, dtype=dtype),
        canvas=torch.zeros((batch_size, 1, img_size, img_size), device=device, dtype=dtype),
        prev_width=torch.full((batch_size, 1), 0.1, device=device, dtype=dtype),
        prev_scaling=torch.ones((batch_size, 1), device=device, dtype=dtype),
        prev_window_size=torch.full((batch_size, 1), float(min(128.0, img_size)),
                                    device=device, dtype=dtype),
        prev_stroke=torch.zeros((batch_size, 7), device=device, dtype=dtype),
        img_size=img_size,
    )


def crop_patch_around_cursor(image: torch.Tensor, cursor: torch.Tensor,
                             window_size: torch.Tensor, patch_size: int) -> torch.Tensor:
    """以 cursor 为中心从 image 抠 window_size 大小的 patch 并 resize 到 patch_size。

    image: (N, C, H, W)  in [0, 1]
    cursor: (N, 2)       in [0, 1), (x, y)
    window_size: (N, 1)  pixel units (float, with grad)
    返回: (N, C, patch_size, patch_size)

    使用 grid_sample，相当于 seq_extract 的 image_cropping_v3。"""
    N, C, H, W = image.shape
    device = image.device

    cursor_px = cursor * float(H)  # (N, 2)
    half = window_size / 2.0  # (N, 1)

    y = torch.linspace(-1.0, 1.0, patch_size, device=device)
    x = torch.linspace(-1.0, 1.0, patch_size, device=device)
    yv, xv = torch.meshgrid(y, x, indexing='ij')
    grid_base = torch.stack([xv, yv], dim=-1).unsqueeze(0)  # (1, P, P, 2)

    scale = (half / (float(H) / 2.0)).view(N, 1, 1, 1)
    offset_x = (cursor_px[:, 0:1] - float(H) / 2.0) / (float(H) / 2.0)
    offset_y = (cursor_px[:, 1:2] - float(H) / 2.0) / (float(H) / 2.0)
    offset = torch.stack([offset_x, offset_y], dim=-1).view(N, 1, 1, 2)

    grid = grid_base * scale + offset
    patch = F.grid_sample(image, grid, mode='bilinear', align_corners=False, padding_mode='zeros')
    return patch


def paste_patch_to_canvas(patch: torch.Tensor, cursor: torch.Tensor,
                          window_size: torch.Tensor, canvas_size: int) -> torch.Tensor:
    """把 (N, 1, P, P) 的 raster patch 贴到 (N, 1, canvas_size, canvas_size) 的位置。

    用 grid_sample 反向做：对目标 canvas 的每个像素，算它对应 patch 的归一化坐标。
    支持任意 window_size，可微。"""
    N, _, P, _ = patch.shape
    device = patch.device

    y = torch.arange(canvas_size, device=device, dtype=patch.dtype)
    x = torch.arange(canvas_size, device=device, dtype=patch.dtype)
    yv, xv = torch.meshgrid(y, x, indexing='ij')
    grid_pix = torch.stack([xv, yv], dim=-1).unsqueeze(0).expand(N, -1, -1, -1)  # (N, H, W, 2)

    cursor_px = cursor * float(canvas_size)  # (N, 2)
    half = window_size / 2.0  # (N, 1)
    half = half.view(N, 1, 1, 1)

    centered = grid_pix - cursor_px.view(N, 1, 1, 2)
    grid = centered / half  # 归一化到 [-1, 1] 范围 (相对于 patch)
    pasted = F.grid_sample(patch, grid, mode='bilinear',
                           align_corners=False, padding_mode='zeros')
    return pasted


def step_with_renderer(state: RolloutState,
                       pred: dict,
                       neural_renderer,
                       raster_size: int,
                       use_pen_soft: bool = True) -> tuple:
    """用模型输出 pred 推进 state 一步，返回 (next_state, info)。

    pred 是 model.decode_hidden 返回的字典，包含：
      - 'pen_state_soft': (N,)    softmax 后的 penUp 概率（用于可微 mask）
      - 'pen_state_hard': (N,)    argmax (0/1) 用于 hard canvas
      - 'x1y1': (N, 2)   sigmoid 后 [0,1]，patch 内 ctrl 点绝对位置
      - 'x2y2': (N, 2)   tanh 后 [-1,1]，相对 cursor 的 offset
      - 'width': (N, 1)  sigmoid * (1-min_width) + min_width
      - 'scaling': (N, 1) sigmoid * max_scaling

    info 包含 pos_before_max_min / win_size_before_max_min，供 outside_loss 用。
    """
    N = state.cursor.shape[0]
    img_size = state.img_size
    device = state.cursor.device

    pen_soft = pred['pen_state_soft'].view(N, 1)  # (N, 1)
    pen_hard = pred['pen_state_hard'].view(N, 1)  # (N, 1)
    x1y1 = pred['x1y1']            # (N, 2) ∈ [0, 1]
    x2y2 = pred['x2y2']            # (N, 2) ∈ [-1, 1]
    next_width = pred['width']     # (N, 1) ∈ [min_width, 1]
    next_scaling = pred['scaling'] # (N, 1) ∈ [0, max_scaling]

    curr_window_size = state.prev_scaling * state.prev_window_size  # (N, 1)
    curr_window_size = torch.clamp(curr_window_size,
                                   min=MIN_WINDOW_SIZE,
                                   max=float(img_size))

    # ---- 渲染 ----
    # 原版送给 raster_unit 的是绝对坐标，已经在 patch 局部空间归一化到 [0,1]
    # x0y0 = (0.5, 0.5)（patch 中心，即 cursor 处）
    # x1y1 ∈ [0,1] 是 control point 在 patch 内坐标
    # x2y2 ∈ [-1,1] → 转 [0,1]
    x0y0 = torch.full_like(x1y1, 0.5)
    x2y2_unit = (x2y2 + 1.0) / 2.0
    w0 = state.prev_width
    w2 = next_width
    raster_params = torch.cat([x0y0, x1y1, x2y2_unit, w0, w2, w0, w2], dim=-1)  # (N, 10)
    stroke_img = neural_renderer.raster_unit(raster_params)  # (N, raster_size, raster_size)
    if stroke_img.shape[-1] != raster_size:
        stroke_img = F.interpolate(stroke_img.unsqueeze(1),
                                   size=(raster_size, raster_size),
                                   mode='bilinear', align_corners=False).squeeze(1)
    stroke_img = stroke_img.unsqueeze(1)  # (N, 1, R, R), [0, 1] (1=stroke)

    # 贴回 canvas
    stroke_on_canvas = paste_patch_to_canvas(stroke_img,
                                             state.cursor,
                                             curr_window_size,
                                             img_size)  # (N, 1, H, W)

    # 用 (1 - pen_soft) 作可微 mask（pen=1 是抬笔，不画）
    pen_mask_soft = (1.0 - pen_soft).view(N, 1, 1, 1)
    pen_mask_hard = (1.0 - pen_hard).view(N, 1, 1, 1)
    canvas_soft_delta = stroke_on_canvas * pen_mask_soft
    canvas_hard_delta = stroke_on_canvas * pen_mask_hard

    # 软 canvas 用于梯度反传；hard canvas 用于送入 encoder
    new_canvas_soft = torch.clamp(state.canvas + canvas_soft_delta, 0.0, 1.0)
    new_canvas_hard = torch.clamp(state.canvas.detach() + canvas_hard_delta.detach(), 0.0, 1.0)

    # ---- cursor 更新（沿用原版 stop_accu_grad 策略：cursor 累加只走 stop_gradient）----
    # delta = x2y2 * curr_window / 2 （像素单位）→ 归一化到 [0,1]
    cursor_px = state.cursor.detach() * float(img_size)  # 不让 cursor 长链梯度累加
    delta_px = x2y2 * curr_window_size / 2.0  # 注意：x2y2 是 (dx, dy)，对应 cursor 的 x/y
    new_cursor_px = cursor_px + delta_px
    pos_before_clip = new_cursor_px.clone()  # 暴露给 outside_loss
    new_cursor_px = torch.clamp(new_cursor_px, 0.0, float(img_size - 1))
    new_cursor = new_cursor_px / float(img_size)  # 仍带 x2y2 的梯度

    # ---- window_size 更新 ----
    next_window_size_raw = next_scaling * curr_window_size.detach()  # 让 scaling 梯度回传
    win_size_before_clip = next_window_size_raw.clone()
    next_window_size = torch.clamp(next_window_size_raw,
                                   min=MIN_WINDOW_SIZE,
                                   max=float(img_size))

    # 原版 prev_width 也按窗口缩放校正
    new_prev_width = next_width * curr_window_size / torch.clamp(next_window_size, min=1e-6)
    new_prev_width = torch.clamp(new_prev_width, MIN_WIDTH, 1.0)

    new_prev_stroke = torch.cat([
        pen_hard, x2y2, x1y1, next_width, next_scaling
    ], dim=-1).view(N, 7).detach()  # prev_stroke 只作为下一步的输入特征，不参与梯度

    next_state = RolloutState(
        cursor=new_cursor,
        canvas=new_canvas_soft,
        prev_width=new_prev_width,
        prev_scaling=next_scaling,
        prev_window_size=curr_window_size,
        prev_stroke=new_prev_stroke,
        img_size=img_size,
    )

    info = {
        'pos_before_max_min': pos_before_clip,        # (N, 2), pre-clip pixel coord
        'win_size_before_max_min': win_size_before_clip,  # (N, 1)
        'curr_window_size': curr_window_size,          # (N, 1)
        'curr_canvas_hard': new_canvas_hard,
        'stroke_on_canvas': stroke_on_canvas,
    }
    return next_state, info

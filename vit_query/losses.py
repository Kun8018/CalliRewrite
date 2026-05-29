"""与 seq_extract 原版完全对齐的 loss 集合。

原版 build_losses (seq_extract/model_common_train.py:1069-1245) 的 7 项：
- raster_cost           : VGG perceptual / L1 (rendered vs target)
- stroke_num_cost       : 1 - mean(pen_state)，鼓励下笔（pen=0）
- smoothness_cost       : 相邻 (x2,y2) 向量余弦相似度
- angle_cost            : (x2,y2) 与 (-1, -1) 的余弦相似度（鼓励笔画走向多样）
- pos_outside_cost      : cursor pre-clip 像素与 clip 后的差 → 惩罚出界
- win_size_outside_cost : window pre-clip 与边界的差
- early_pen_states_cost : 前 K 步不能抬笔（pen min）

phase1 / phase2 区别只是 weights 不同（见 hyper_parameters.py）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vgg_loss import VGG16PerceptualLoss
from diffable_state import MIN_WINDOW_SIZE


# --------------------------------------------------------------------- #
# 各分量
# --------------------------------------------------------------------- #

def raster_l1_cost(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """rendered: (N, H, W) ∈ [0,1] 1=stroke
       target:   (N, H, W) ∈ [0,1] 1=stroke (caller 负责把 PIL [1=BG] 翻成 [1=stroke])"""
    return F.l1_loss(rendered, target)


class PerceptualCost(nn.Module):
    """对齐原版 perc_loss 的 raw_add 模式：直接把多层 L1 取均值。"""

    def __init__(self):
        super().__init__()
        self.vgg = VGG16PerceptualLoss()

    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # vgg_loss 接受 (N, 1, H, W)
        return self.vgg(rendered.unsqueeze(1), target.unsqueeze(1))


def stroke_num_cost(pred_seq: torch.Tensor) -> torch.Tensor:
    """pred_seq: (N, T, 7) — 注意 [:, :, 0] 是 pen_soft ∈ [0, 1]
       原版 stroke_num_loss = 1 - mean(pen_state)，pen=0 表示落笔，所以
       1 - E[pen] 越小 → 落笔越多 → loss 越大；反过来，越大越好。
       等价：loss = mean(pen)（越大表示越多抬笔，惩罚）→ 与原版语义一致。"""
    return pred_seq[..., 0].mean()


def smoothness_cost(pred_seq: torch.Tensor) -> torch.Tensor:
    """相邻 (x2, y2) 向量余弦相似度（仅对相邻都落笔的步），返回 1 - mean(cos)。"""
    x2y2 = pred_seq[..., 3:5]
    pen = pred_seq[..., 0]
    drawing = 1.0 - pen  # 1=落笔
    a1 = x2y2[:, :-1]
    a2 = x2y2[:, 1:]
    norm = a1.norm(dim=-1) * a2.norm(dim=-1) + 1e-8
    cos = (a1 * a2).sum(dim=-1) / norm  # (N, T-1)
    w = drawing[:, :-1] * drawing[:, 1:]
    if w.sum() < 1.0:
        return torch.tensor(0.0, device=pred_seq.device, dtype=pred_seq.dtype)
    return 1.0 - (cos * w).sum() / w.sum().clamp_min(1.0)


def angle_cost(pred_seq: torch.Tensor) -> torch.Tensor:
    """seq_extract get_angle_loss：(x2,y2) vs (-1,-1) 的余弦相似度，落笔且 cos > 0.5 时算。
       注意原版返回 -final * 10 的总系数是 angle_loss_weight=1.0 自带的；这里返回原值，
       train.py 中按 weights 加权。"""
    x2y2 = pred_seq[:, 1:, 3:5]
    pen_curr = pred_seq[:, 1:, 0]
    pen_prev = pred_seq[:, :-1, 0]
    ref = torch.full_like(x2y2, -1.0)
    norm = x2y2.norm(dim=-1) * ref.norm(dim=-1) + 1e-8
    cos = (x2y2 * ref).sum(dim=-1) / norm
    mask = (1.0 - pen_curr) * pen_prev  # 当前落笔 且 上一步抬笔 → 笔画起点
    # 原版用 where(rst > 0.5, rst, 0)，等价惩罚 cos > 0.5 的情形
    final = (cos * mask).clamp(min=0.5)
    final = (final - 0.5) * (mask)  # 把 < 0.5 的清零
    return final.mean() * 10.0


def pos_outside_cost(pos_before: torch.Tensor, img_size: int) -> torch.Tensor:
    """pos_before: (N, T, 2) pixel-space pre-clip cursor 位置"""
    pos_after = torch.clamp(pos_before, 0.0, float(img_size - 1))
    return (pos_before - pos_after).abs().mean()


def win_size_outside_cost(win_before: torch.Tensor, img_size: int) -> torch.Tensor:
    """win_before: (N, T, 1) pre-clip window size."""
    top = torch.clamp(win_before - float(img_size), min=0.0) / float(img_size)
    bot = torch.clamp(MIN_WINDOW_SIZE - win_before, min=0.0) / MIN_WINDOW_SIZE
    return (top + bot).mean()


def early_pen_states_cost(pred_seq: torch.Tensor, early_len: int = 7) -> torch.Tensor:
    """前 early_len 步：取每个 batch 的 pen 最小值再平均。原版越低越好。"""
    early = pred_seq[:, :early_len, 0]  # (N, K)
    return early.min(dim=1).values.mean()


# --------------------------------------------------------------------- #
# 监督 loss（phase 1 GT 7D 序列对齐用）
# --------------------------------------------------------------------- #

class SupervisedSeqLoss(nn.Module):
    """对 (N, T, 7) 预测 vs (N, T, 7) GT 做：
       - pen: BCE-with-logits（用 pen_logits）
       - x1y1/x2y2: L1
       - width/scaling: L1
       带 mask（GT 的有效长度）。"""

    def __init__(self, w_pen=1.0, w_coord=5.0, w_param=1.0):
        super().__init__()
        self.w_pen = w_pen
        self.w_coord = w_coord
        self.w_param = w_param

    def forward(self, pred_seq: torch.Tensor, pen_logits: torch.Tensor,
                gt: torch.Tensor, mask: torch.Tensor) -> dict:
        mask_sum = mask.sum().clamp_min(1.0)

        # pen_logits: (N, T, 2)，gt[..., 0] ∈ {0, 1}
        gt_pen = gt[..., 0].long().clamp(0, 1)
        # 用 cross_entropy；先 flatten
        pen_logits_flat = pen_logits.reshape(-1, 2)
        gt_pen_flat = gt_pen.reshape(-1)
        pen_loss_pix = F.cross_entropy(pen_logits_flat, gt_pen_flat, reduction='none')
        pen_loss_pix = pen_loss_pix.reshape(gt_pen.shape)
        pen_loss = (pen_loss_pix * mask).sum() / mask_sum

        coord_diff = (pred_seq[..., 1:5] - gt[..., 1:5]).abs().mean(dim=-1)
        coord_loss = (coord_diff * mask).sum() / mask_sum

        param_diff = (pred_seq[..., 5:7] - gt[..., 5:7]).abs().mean(dim=-1)
        param_loss = (param_diff * mask).sum() / mask_sum

        total = self.w_pen * pen_loss + self.w_coord * coord_loss + self.w_param * param_loss
        return {'sup_total': total, 'sup_pen': pen_loss,
                'sup_coord': coord_loss, 'sup_param': param_loss}


# --------------------------------------------------------------------- #
# 组合 loss
# --------------------------------------------------------------------- #

class CombinedRolloutLoss(nn.Module):
    """对齐 seq_extract 的 build_training_op_split，从 rollout 产物 + GT 计算全部 loss。"""

    def __init__(self,
                 raster_weight: float = 1.0,
                 stroke_num_weight: float = 0.5,
                 smoothness_weight: float = 0.0,
                 angle_weight: float = 0.0,
                 outside_weight: float = 10.0,
                 win_outside_weight: float = 10.0,
                 early_pen_weight: float = 0.1,
                 early_pen_length: int = 7,
                 supervised_weight: float = 1.0,
                 use_perceptual: bool = True,
                 use_l1_raster: bool = True,
                 phase: int = 1):
        super().__init__()
        self.raster_weight = raster_weight
        self.stroke_num_weight = stroke_num_weight
        self.smoothness_weight = smoothness_weight
        self.angle_weight = angle_weight
        self.outside_weight = outside_weight
        self.win_outside_weight = win_outside_weight
        self.early_pen_weight = early_pen_weight
        self.early_pen_length = early_pen_length
        self.supervised_weight = supervised_weight
        self.use_perceptual = use_perceptual
        self.use_l1_raster = use_l1_raster
        self.phase = phase

        if use_perceptual:
            self.perceptual = PerceptualCost()
        self.supervised = SupervisedSeqLoss()

    def forward(self,
                rollout_out: dict,
                target_stroke_img: torch.Tensor,
                img_size: int,
                gt_strokes: torch.Tensor = None,
                gt_mask: torch.Tensor = None) -> dict:
        """target_stroke_img: (N, H, W) ∈ [0,1], 1=stroke"""
        pred_seq = rollout_out['seq']
        pen_logits = rollout_out['pen_logits']
        rendered = rollout_out['rendered']  # (N, H, W), 1=stroke
        pos_before = rollout_out['pos_before_max_min']
        win_before = rollout_out['win_size_before_max_min']

        comp = {}
        total = torch.zeros((), device=pred_seq.device, dtype=pred_seq.dtype)

        # raster
        if self.use_l1_raster:
            l1 = raster_l1_cost(rendered, target_stroke_img)
            comp['raster_l1'] = l1.detach().float().item()
            total = total + self.raster_weight * l1
        if self.use_perceptual:
            perc = self.perceptual(rendered, target_stroke_img)
            comp['perceptual'] = perc.detach().float().item()
            total = total + self.raster_weight * perc

        # stroke_num
        sn = stroke_num_cost(pred_seq)
        comp['stroke_num'] = sn.detach().float().item()
        total = total + self.stroke_num_weight * sn

        # smoothness / angle（按原版 phase 2 才打开）
        if self.smoothness_weight > 0:
            sm = smoothness_cost(pred_seq)
            comp['smoothness'] = sm.detach().float().item()
            # 原版乘以 stroke_num_loss_weight；这里用 stroke_num_weight 模拟
            total = total + self.smoothness_weight * self.stroke_num_weight * sm
        if self.angle_weight > 0:
            ag = angle_cost(pred_seq)
            comp['angle'] = ag.detach().float().item()
            total = total + self.angle_weight * self.stroke_num_weight * ag

        # outside
        pos_out = pos_outside_cost(pos_before, img_size)
        comp['pos_outside'] = pos_out.detach().float().item()
        total = total + self.outside_weight * pos_out

        win_out = win_size_outside_cost(win_before, img_size)
        comp['win_outside'] = win_out.detach().float().item()
        total = total + self.win_outside_weight * win_out

        # early pen
        epc = early_pen_states_cost(pred_seq, self.early_pen_length)
        comp['early_pen'] = epc.detach().float().item()
        total = total + self.early_pen_weight * epc

        # supervised（phase 1 默认开启；phase 2 可关）
        if gt_strokes is not None and self.supervised_weight > 0:
            T_pred = pred_seq.shape[1]
            T_gt = gt_strokes.shape[1]
            T = min(T_pred, T_gt)
            sup = self.supervised(pred_seq[:, :T], pen_logits[:, :T],
                                  gt_strokes[:, :T], gt_mask[:, :T])
            comp.update({k: v.detach().float().item() for k, v in sup.items()})
            total = total + self.supervised_weight * sup['sup_total']

        comp['total'] = total
        return comp

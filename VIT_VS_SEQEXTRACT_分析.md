# 为什么 `lightweight` / `vit_query` 比原版 `seq_extract` 效果差 — 详细诊断

> 阅读对象：本仓库 ICRA 2024 论文 CalliRewrite 一阶段的改写工作（用 ViT/ResNet+Transformer 替换原 CNN+LSTM 编解码器）。
>
> 结论先行：**当前 `lightweight` 和 `vit_query` 的效果差，不是因为"ViT 比 CNN+LSTM 弱"，而是因为复刻过程中引入了若干阻断训练的实现 bug，并且把原版做对的若干关键设计省略掉了。**

下文按"致命 bug → 严重设计缺陷 → 一般实现细节"三档列出问题，每条都给出文件/行号、原版做法、新版做法、影响。

---

## 0. 原版 `seq_extract` 的核心设计要点（对照基准）

在 [seq_extract/model_common_train.py:824-1035](seq_extract/model_common_train.py#L824-L1035) 的 `get_points_and_raster_image` 里，原版每一步做的事情是：

1. **以 cursor 为中心** 用 `image_cropping_v3` 在 target image 和 canvas 上抠一个 `window_size × window_size` 的 patch（[seq_extract/model_common_train.py:522-548](seq_extract/model_common_train.py#L522-L548)）；
2. 同时把 entire image / canvas 缩到 `raster_size` 作为全局上下文；
3. **patch+entire+cursor+window_size 一起送入 `combined encoder`**（[seq_extract/model_common_train.py:290-392](seq_extract/model_common_train.py#L290-L392)），产生一个特征 `z`；
4. `z` + prev_width + window_size + cursor 送入 LSTM，得到这一步的 `(pen, x1y1, x2y2, width, scaling)`；
5. 用一个**预训练好的可微 RasterUnit** 把这一步的笔画段渲染成 raster_size 大小的小图，再 `image_pasting_v3` 贴回 entire canvas；
6. **canvas、cursor、window_size 都用模型自己刚刚预测的结果更新**，再进入下一步；
7. 训练时整个 `max_seq_len=48` 的循环展开在 graph 里，**全过程可微**，loss 直接通过 raster/perc 路径反传到所有 cursor/window 决策。

损失包含 8 项（[seq_extract/model_common_train.py:1069-1245](seq_extract/model_common_train.py#L1069-L1245)）：
- `raster_cost`（VGG perceptual on rendered vs target）
- `stroke_num_cost` 鼓励不要早停
- `smoothness_cost`、`angle_cost`
- `pos_outside_cost`、`win_size_outside_cost`（权重 10.0，强约束 cursor / window 不出界）
- `early_pen_states_cost`
- 以上加权和

两阶段都会调用 `load_checkpoint(sess, FLAGS.neural_renderer_path, ras_only=True)` ([seq_extract/train_phase_1.py:296](seq_extract/train_phase_1.py#L296), [seq_extract/train_phase_2.py:309](seq_extract/train_phase_2.py#L309))，加载 30 万步预训练的 RasterUnit。

记住这套设计，下文每个问题都是相对它的偏离。

---

## 1. 致命 bug：直接阻断了 phase 2 的训练

### 1.1 phase 2 整个 rollout 被包在 `torch.no_grad()` 里

文件：
- [lightweight/train.py:312-335](lightweight/train.py#L312-L335)
- [vit_query/train.py:212-237](vit_query/train.py#L212-L237)

```python
if args.arch == 'autoregressive':
    # autoregressive 模型完整推理不可微，用 no_grad
    with torch.no_grad():
        ...
        for i in range(model.max_seq_len):
            ...
            output, hidden = model.forward_step(...)
            stroke = output['seq'].squeeze(0).detach()   # ← detach
            strokes_list.append(stroke)
            state = apply_seq7_step(state, stroke.cpu().numpy(), model.image_size)  # ← numpy
        ...
        strokes_seq7 = torch.stack(batch_strokes, dim=0)
    losses = criterion(strokes_seq7, images)   # ← 此 loss 完全没有梯度
loss.backward()                                 # ← 等于 noop
```

**影响**：phase 2 调一次 `loss.backward()` 时，autoregressive 模型的所有参数都没有梯度。换句话说，无论你跑多少 epoch，**`autoregressive` 模型的 phase 2 训练根本没在更新参数**。模型质量等于 phase 1 终止时的状态。

注释里写"autoregressive 模型完整推理不可微"，**这个判断是错的**：原版 seq_extract 把 LSTM rollout 完整展开在计算图里，全过程可微（[seq_extract/model_common_train.py:873-1026](seq_extract/model_common_train.py#L873-L1026)）。我们这边不可微的原因不是模型架构，而是代码里 `.detach()` + `.cpu().numpy()` + numpy `apply_seq7_step` 主动断了梯度。

**修复方向**：
1. 删掉 `torch.no_grad()` 和 `.detach()`；
2. 把 `apply_seq7_step` 用 torch ops 重写（参考下面 1.2）；
3. canvas 更新改成可微 — 即每步用可微的 RasterUnit 渲染当前 stroke，再 `clamp(canvas + stroke_img, 0, 1)`，避免 numpy 路径。

### 1.2 canvas / cursor 更新断梯度

[lightweight/dataset.py:377-423](lightweight/dataset.py#L377-L423) 中 `render_seq7_step_on_canvas` 直接用 `PIL.ImageDraw.line` 画线，整个过程在 CPU + numpy。`apply_seq7_step` 同样是纯 numpy。

phase 2 的 rollout 用了它，phase 1 训练数据的预 unroll 也用了它（[lightweight/dataset.py:438-476](lightweight/dataset.py#L438-L476)）。

**影响**：
- **梯度不可达**：即使去掉 1.1 的 `no_grad`，cursor、canvas 仍然是 numpy 路径产生的，回传不到模型；
- **训练/推理 mismatch**：phase 1 训练时 dataset 用 PIL 直线画 canvas，phase 2 渲染 loss 用的是 NeuralRasterizorStep，两者的笔画形状/线宽分布完全不一致；
- 模型从未看到自己预测出来的 canvas 长什么样（exposure bias 1）。

**修复方向**：把这条路径换成纯 torch 实现，并直接调用 `NeuralRasterizorStep` 渲染单步笔画后累加到 canvas。让数据集预生成的 canvas 和推理时一致。

### 1.3 `NeuralRasterizorStep` 从未加载预训练权重

文件：[lightweight/neural_renderer.py:49-95](lightweight/neural_renderer.py#L49-L95)、[vit_query/neural_renderer.py:50-96](vit_query/neural_renderer.py#L50-L96)、[lightweight/train.py:105](lightweight/train.py#L105)、[vit_query/train.py:125](vit_query/train.py#L125)。

```python
class NeuralRasterizorStep(nn.Module):
    def __init__(self, raster_size, position_format='abs'):
        super().__init__()
        ...
        self.raster_unit = RasterUnit(128)   # ← 随机初始化，没有 load_state_dict
```

对比原版 [seq_extract/train_phase_1.py:296](seq_extract/train_phase_1.py#L296)：
```python
load_checkpoint(sess, FLAGS.neural_renderer_path, ras_only=True)
# renderer_300000.tfmodel：30 万步预训练的 RasterUnit
```

**影响**：
- 原版的 RasterUnit 是把 `(x0,y0,x1,y1,x2,y2,r0,r2,w0,w2)` 这类参数渲染成贝塞尔笔画的小图，**必须经过预训练**才能输出像样的 stroke；
- 我们这边的 RasterUnit 是从头随机权重，输出是噪声；
- phase 2 的 `UnsupervisedLoss = L1(rendered, target) + perceptual(rendered, target)` 里 `rendered` 一直是噪声 → loss 没有任何有意义的梯度可以反传到模型；
- 即使修了 1.1 的 `no_grad`，loss 也基本是噪声梯度，模型不会收敛到正确解。

**修复方向**：
- 选项 A（推荐）：**把 TF 原版的 `renderer_300000.tfmodel` 权重迁移到 PyTorch**（架构是同一个：4 个 FC + 6 个 conv + 3 个 pixel_shuffle），然后 freeze；
- 选项 B：用同样的（笔画参数, 渲染图像）pair 在 PyTorch 端预训练一遍 RasterUnit，达到 reasonable IoU 后再 freeze；
- 选项 C：完全放弃 NeuralRasterizor，phase 2 改用**纯几何的可微渲染器**（如 DiffVG 或 soft splatting），代价是要重新调超参。

> ⚠️ 如果只修 1.1 而不修 1.3，phase 2 还是会失败。两条 bug 是合并发生的。

---

## 2. 严重设计缺陷

### 2.1 cursor 闭环断裂 + Exposure Bias

原版每一步的 cursor、prev_width、window_size 都由模型上一步的输出 **forward 自闭环** 产生（[seq_extract/model_common_train.py:996-1025](seq_extract/model_common_train.py#L996-L1025)）。
新版 phase 1 训练时用 `make_autoregressive_item` 预先按 GT 笔画 unroll 出 `(canvases, cursors, prev_strokes)` 序列（[lightweight/dataset.py:438-476](lightweight/dataset.py#L438-L476)），**teacher forcing 完全替换掉了模型自己的预测路径**。

```python
# lightweight/dataset.py:455-465
for j in range(chunk_len):
    canvases[j, 0] = state['canvas']   # ← 由 GT 笔画产生
    cursors[j] = state['cursor']
    prev_strokes[j] = state['prev_stroke']
    ...
    if stroke_idx < seq_len_actual:
        stroke = strokes[stroke_idx].astype(np.float32)
        state = apply_seq7_step(state, stroke, img_size)   # ← 按 GT 推进
```

**影响**：
- 模型在 phase 1 从来没见过"自己预测出来的 cursor 是什么样的"，推理时一旦笔画稍偏，cursor 就脱离训练分布 → **典型 exposure bias**；
- 你在两张推理图里看到的"画到右上角不停转圈"正是这个症状：模型预测了一次坏 stroke，cursor 飘出去后，后续状态已经在它的训练分布之外，模型只能输出噪声。

**修复方向**：
- 短期：phase 1 即引入 scheduled sampling — 以一定概率把上一步换成模型自己的预测；
- 中期：参考原版，把整段 (canvas, cursor) 用模型自身预测可微地 unroll，至少做几步内的 BPTT；
- 长期：phase 2 必须接通 1.1 的可微 rollout，让 perceptual loss 反过来纠正错的 cursor。

### 2.2 损失函数把原版的关键正则全删了

原版 8 项 loss 加权和（参数权重在 [seq_extract/hyper_parameters.py:39-244](seq_extract/hyper_parameters.py#L39-L244)）：

| Loss | 原版权重 (phase2) | 新版 phase1 | 新版 phase2 |
|---|---|---|---|
| `raster_cost` (VGG perceptual) | 1.0 | — | 1.0 (render+perc) |
| `stroke_num_cost` | 0.5 | — | — |
| `smoothness_cost` | 0.5 (× sn_w) | — | — |
| `angle_cost` | 1.0 (× sn_w) | — | — |
| `pos_outside_cost` | **10.0** | — | — |
| `win_size_outside_cost` | **10.0** | — | — |
| `early_pen_states_cost` | 0.1 | — | — |
| pen BCE / coord L1 / param L1 | — | (1, 5, 1) | — |

**影响**：
- `pos_outside_cost` 和 `win_size_outside_cost` 是 cursor 不飞出去、窗口不变成负数的硬保险。新版完全没有 → 模型可以输出任意疯狂的 `dx2/dy2`、`scaling`，反正没人惩罚。**这是你在推理图里看到 cursor 在右上角乱转、笔画完全不在字形上的直接原因**；
- `stroke_num_cost` 鼓励模型尽可能下笔，没它的话模型最容易学到的捷径就是**全部 `pen=1`（笔抬起）+ 端到端 loss 不太差**（因为没有 raster 损失能拉回来）；
- `smoothness_cost / angle_cost` 是 phase 2 才打开的两项，没有它们 phase 2 的输出会非常折线、抖动；
- 新版 phase 1 三项 `pen + coord + param` 都是直接对 GT 7D 序列做监督，本质上是把 phase 1 改成纯 imitation learning，并不和原版"先在 QuickDraw 上学渲染对齐"的目标对齐。

**修复方向**：把原版 6 个辅助 loss 全部移植过来，权重保留原值。其中 `outside` 类两项只需要把 cursor / window 中间变量从 rollout 里 expose 出来即可计算。

### 2.3 输出激活与原版的语义不一致

原版输出（[seq_extract/model_common_train.py:418-436](seq_extract/model_common_train.py#L418-L436)）：

```python
x1y1   = sigmoid(...)                          # ∈ [0, 1], "patch 内绝对位置"
x2y2   = tanh(...)                             # ∈ [-1, 1], "相对 cursor 的 offset"
widths = sigmoid * (1 - min_width) + min_width # ∈ [min_width, 1.0]
scaling = sigmoid * max_scaling                # ∈ [0, 2.0]  ← 关键：可放大
```

新版（[lightweight/model.py:152-160, 442-444](lightweight/model.py#L152-L160) 等）：

```python
coords = torch.tanh(...)        # x1,y1,x2,y2 全部 tanh ∈ [-1, 1]
params = torch.sigmoid(...)     # r, s ∈ [0, 1]  ← scaling 上限只能到 1
```

**影响**：
- x1y1 的含义被改成了 [-1,1]（vs 原版 [0,1]），而 `seq7_to_absolute` 里 (`lightweight/neural_renderer.py:122-135`) 又把它当 offset 用，**单位都对不上**；
- scaling 上限只能 1.0 意味着**模型无法把窗口放大**，只能不变或缩小；原版 phase 2 经常需要放大窗口去画长笔画 → 新版做不到；
- `min_width` 没了，模型可以一直输出 `r=0`（无可见笔画），训练初期最容易掉进这个 trivial 解。

**修复方向**：完全照搬原版的输出激活与缩放系数，包括 `min_width=0.01` 和 `max_scaling=2.0`。

### 2.4 推理时 `window_size` 是固定常数

新版 `ViTAutoregressiveExtractor7D.forward_teacher_forcing` 中（[vit_query/model.py:474-475](vit_query/model.py#L474-L475)、[lightweight/model.py:464](lightweight/model.py#L464)）：

```python
window_size = torch.full((batch_size, 1), self.init_window_size, ...)
for i in range(canvases.shape[1]):
    output, hidden = self.forward_step(..., window_size)   # ← 永不更新
```

`init_window_size = patch_size * 2 = 128`，并且每一步都用同一个 `window_size` 当输入。模型预测的 `r/s`（scaling）从未真正作用回 patch 裁剪。

**影响**：
- 原版的 multi-scale 注意力（先用大窗口找位置，再用小窗口画细节）整个机制失效；
- patch 永远是 128×128 的固定大小，对小笔画/大笔画一视同仁；
- 模型也学不出"我应该放大/缩小窗口"。

**修复方向**：参考 [seq_extract/model_common_train.py:876-878, 987-998](seq_extract/model_common_train.py#L876-L998)，每步算 `curr_window_size = prev_scaling * prev_window_size`，clamp 到 `[min_window_size, image_size]`，并传给 `crop_patch`。

### 2.5 ViT 没有预训练，且 patch_size=16 在 224 输入上只有 196 个 token

[vit_query/model.py:11-69](vit_query/model.py#L11-L69) 的 `ViTTinyPatch16X16` 是**完全从随机初始化**的 12 层 ViT。
对比原版的 `conv13_c3` encoder：

| 特性 | 原版 conv13_c3 | 新版 ViTTiny |
|---|---|---|
| Inductive bias | 强（卷积+coordconv+小感受野堆叠） | 弱（全局 attention） |
| 预训练 | RasterUnit 部分预训练，encoder 同时训练大数据集 | 无 |
| 训练数据量级 | QuickDraw 10 类 × N + 书法数据 | 同上 |
| Token 数 / 局部窗 | 局部多尺度 | 14×14=196 个全局 token |

**影响**：ViT 在数据量这种规模下，缺少 inductive bias 又没有预训练初始化，**很难学到笔画局部结构**。新版即使把上面 bug 都修好，也仍然会显著弱于原版。

**修复方向**：
- 选项 A：换成 `ViT-S/16` 预训练（DINO / MAE），加载权重再微调；
- 选项 B：直接用 ResNet 作 patch encoder + 一个浅层 transformer decoder（即 `lightweight` 路线），但是先把 1.1/1.3/2.1/2.2 修好；
- 选项 C：保留 ViT 但加 patch 级 conv stem（hybrid ViT），增加局部 bias。

### 2.6 phase 1 用 chunk_len=8/16 训练，丢失全局时序结构

[lightweight/dataset.py:438-476](lightweight/dataset.py#L438-L476) `make_autoregressive_item` 每次随机选 `start_idx ∈ [0, seq_len-1]`，只 unroll `chunk_len` 步（默认 8 或 16）。

**影响**：
- 模型每个 batch 只看到一段中间，没有看到笔画的"开始 → 结束"全过程；
- LSTM/GRU 的 hidden state 在 chunk 起点是 zero init，**等于丢掉了前 start_idx 步的隐藏信息**，但 canvas/cursor 又是从 GT 推到中间，造成 hidden state 和 canvas 不一致；
- 原版 phase 1 max_seq_len=48 全展开 BPTT。

**修复方向**：要么 unroll 全长序列，要么至少从 `start_idx=0` 起步固定长度，避免 hidden state 错配。

---

## 3. 一般实现细节

### 3.1 `vit_query` 把图像 resize 到 224，丢信息

phase 2 数据集图像是 256 的（[vit_query/train_phase2.sh:18](vit_query/train_phase2.sh#L18) `--img_size 224`），ViT 输入只有 224。书法细节本来就重要，再 down-sample 损失更大。建议 `img_size=256`，patch_size 改为 16 → 16×16=256 个 token。

### 3.2 `find_undrawn_cursor`（[lightweight/dataset.py:479-495](lightweight/dataset.py#L479-L495)）在训练里没有被用到

只有 inference 时被引用。原版有 `init_cursor_on_undrawn_pixel` 选项可以训练时也用 residual 图找新起点；当前实现等于在 phase 2 推理时让 cursor 起点合理，但训练时模型从未学过这种 init 分布。

### 3.3 phase 1 / phase 2 的 `image` 归一化方式不一致

- phase 1 `make_seq7_item` 用 `normalize='zero_one'`（[lightweight/dataset.py:341](lightweight/dataset.py#L341)）；
- `StrokeDataset.__getitem__` 的 oneshot 分支又用 `normalize='minus_one_one'`；
- `ImageOnlyDataset` 也是 `zero_one`。

`UnsupervisedLoss.forward` 里靠 `if target_images.mean() > 0.5: target_images = 1.0 - target_images` 来动态推断方向（[lightweight/train.py:124-127](lightweight/train.py#L124-L127)），不稳定 — 如果一张图本来就接近全白，会被错误反转。

**修复方向**：dataset 端就统一成 `[0=BG, 1=stroke]` 单一格式；loss 端不要做 mean 判断。

### 3.4 `crop_patch` 在 ViT 流程里仅用于局部 patch encoder，target 输入到 ViT 时仍然是全图

意味着 ViT global feature 和 patch encoder feature 之间没有显式的 cursor 引导。原版的 `add_coords` (CoordConv) 给 encoder 输入显式提供了 cursor / window 信息；新版只用 MLP embed 后 concat，**位置信号弱很多**。

---

## 4. 建议的优先修复顺序

1. **修 1.3 NeuralRasterizor 预训练**（不修这个，phase 2 永远没法 unsupervised 训）；
2. **修 1.1 + 1.2 让 phase 2 rollout 可微**（去掉 `no_grad`、把 `apply_seq7_step` 改成可微 torch 版）；
3. **加回 2.2 的 6 个辅助 loss**（重点是 `pos_outside` 和 `win_size_outside`，它们是当前推理失败的直接原因）；
4. **修 2.3 输出激活语义**（让 x1y1/x2y2/scaling 含义和原版一致）；
5. **修 2.4 让 window_size 真正动态化**；
6. **修 2.1 + 2.6 phase 1 训练目标**（scheduled sampling，至少 unroll 完整 chunk，hidden state 别错配）；
7. 最后再上 2.5（ViT 预训练 / hybrid 化）。

修完前 4 项，理论上就能复现接近原版的 phase 2 渲染效果；7 完整修完后，才可能讨论"ViT 比 CNN+LSTM 强还是弱"这件事。当前的实验**还没真正在比 ViT vs LSTM**，比的是"有 bug 的 ViT pipeline vs 没 bug 的 LSTM pipeline"。

---

## 5. 一句话总结给老板看

> 当前 `lightweight` 和 `vit_query` 主要不是输给了"ViT"，而是输给了两个实现层面的 bug（phase 2 的 `no_grad` rollout、未预训练的 NeuralRasterizor）和六项原版关键正则的缺失（cursor 出界、窗口出界、stroke_num 等）。先把这些修了，再讨论架构创新点是否有效。

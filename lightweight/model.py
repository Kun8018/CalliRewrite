"""
ResNet-18 + Transformer 轻量模型
用于替换一阶段的 seq_extract
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class StrokeTransformer(nn.Module):
    """
    ResNet18 编码器 + Transformer 解码器
    输入: 书法图像 (1, 256, 256)
    输出: 笔画序列 (N, 7) - [pen_state, x1, y1, x2, y2, r, s]
    """
    def __init__(
        self,
        d_model=256,
        nhead=None,
        num_decoder_layers=3,
        max_seq_len=100,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 自动选择合适的 nhead（必须能整除 d_model）
        if nhead is None:
            for n in [16, 8, 4, 2, 1]:
                if d_model % n == 0:
                    nhead = n
                    break
        assert d_model % nhead == 0, f"d_model {d_model} must be divisible by nhead {nhead}"
        self.nhead = nhead

        # ========== 1. ResNet-18 编码器 ==========
        resnet = resnet18(weights=None)

        # 修改第一个卷积层接受单通道输入
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)  # RGB → 灰度

        # ResNet层
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 特征投影到d_model
        self.feature_proj = nn.Linear(512, d_model)

        # ========== 2. 位置编码 ==========
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        # ========== 3. Transformer 解码器 ==========
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # ========== 4. 输出头 ==========
        # 7个输出维度: [pen_state, x1, y1, x2, y2, r, s]
        self.pen_head = nn.Linear(d_model, 1)  # pen_state (0或1)
        self.coord_head = nn.Linear(d_model, 4)  # x1, y1, x2, y2
        self.param_head = nn.Linear(d_model, 2)  # r, s

        # 开始 token 嵌入
        self.sos_embedding = nn.Parameter(torch.randn(1, 1, d_model))

        # 目标序列嵌入（用于teacher forcing训练）
        self.target_embedding = nn.Linear(7, d_model)

    def encode_image(self, image):
        """
        编码图像到特征向量
        image: (batch, 1, 256, 256)
        返回: (batch, seq_len=1, d_model)
        """
        # ResNet前向
        x = self.conv1(image)          # (batch, 64, 128, 128)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)            # (batch, 64, 64, 64)

        x = self.layer1(x)             # (batch, 64, 64, 64)
        x = self.layer2(x)             # (batch, 128, 32, 32)
        x = self.layer3(x)             # (batch, 256, 16, 16)
        x = self.layer4(x)             # (batch, 512, 8, 8)

        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (batch, 512, 1, 1)
        x = x.flatten(1)                       # (batch, 512)

        # 投影到 d_model
        x = self.feature_proj(x)              # (batch, d_model)

        # 增加序列维度
        x = x.unsqueeze(1)                     # (batch, 1, d_model)

        return x

    def forward(self, image, target_seq=None, teacher_forcing_ratio=0.5):
        """
        训练模式下使用 teacher forcing

        参数:
            image: (batch, 1, 256, 256)
            target_seq: (batch, seq_len, 7) - 用于teacher forcing
            teacher_forcing_ratio: 使用teacher forcing的概率

        返回:
            predictions: (batch, seq_len, 7)
        """
        batch_size = image.shape[0]

        # 编码图像
        memory = self.encode_image(image)  # (batch, 1, d_model)

        # 准备解码器输入
        if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # Teacher forcing: 使用真实的前一步
            decoder_input = self.target_embedding(target_seq)  # (batch, seq_len, d_model)
            # 添加 SOS token
            sos = self.sos_embedding.repeat(batch_size, 1, 1)
            decoder_input = torch.cat([sos, decoder_input[:, :-1]], dim=1)
        else:
            # 自回归: 从 SOS 开始
            decoder_input = self.sos_embedding.repeat(batch_size, self.max_seq_len, 1)

        # 添加位置编码
        decoder_input = self.pos_encoder(decoder_input)

        tgt_mask = self._causal_mask(decoder_input.shape[1], decoder_input.device)
        output = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=tgt_mask
        )  # (batch, seq_len, d_model)

        # 输出预测
        pen_logits = self.pen_head(output)       # (batch, seq_len, 1)
        coords = self.coord_head(output)         # (batch, seq_len, 4)
        params = self.param_head(output)         # (batch, seq_len, 2)

        # 合并输出
        predictions = torch.cat([
            torch.sigmoid(pen_logits),  # pen_state [0,1]
            torch.tanh(coords),         # x1,y1,x2,y2 [-1,1]
            torch.sigmoid(params)       # r,s [0,1]
        ], dim=-1)

        return predictions

    def _causal_mask(self, seq_len, device):
        return torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)

    @torch.no_grad()
    def generate(self, image, max_len=100):
        """
        推理模式，自回归生成笔画序列

        参数:
            image: (1, 1, 256, 256)

        返回:
            strokes: (seq_len, 7)
        """
        self.eval()

        batch_size = image.shape[0]

        # 编码图像
        memory = self.encode_image(image)  # (batch, 1, d_model)

        # 从 SOS 开始
        generated = []
        current_input = self.sos_embedding.repeat(batch_size, 1, 1)  # (batch, 1, d_model)

        for _ in range(max_len):
            decoder_input = self.pos_encoder(current_input)
            tgt_mask = self._causal_mask(decoder_input.shape[1], decoder_input.device)
            output = self.decoder(
                tgt=decoder_input,
                memory=memory,
                tgt_mask=tgt_mask
            )

            # 预测最后一步
            last_output = output[:, -1:]
            pen_logits = self.pen_head(last_output)
            coords = self.coord_head(last_output)
            params = self.param_head(last_output)

            # 组合
            pred = torch.cat([
                torch.sigmoid(pen_logits),
                torch.tanh(coords),
                torch.sigmoid(params)
            ], dim=-1)

            # 保存
            generated.append(pred.squeeze(1))

            # 二值化pen_state
            pred_for_input = pred.clone()
            pred_for_input[..., 0] = (pred_for_input[..., 0] > 0.5).float()

            # 作为下一步输入
            next_embedding = self.target_embedding(pred_for_input)
            current_input = torch.cat([current_input, next_embedding], dim=1)

        # 合并所有预测
        strokes = torch.stack(generated, dim=1)  # (batch, seq_len, 7)

        return strokes[0]


class ResNetFeatureBackbone(nn.Module):
    def __init__(self, image_size=256, d_model=256):
        super().__init__()
        resnet = resnet18(weights=None)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.proj = nn.Conv2d(512, d_model, kernel_size=1)
        num_tokens = (image_size // 32) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)

    def forward_features(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)
        tokens = x.flatten(2).transpose(1, 2)
        return tokens + self.pos_embed[:, :tokens.shape[1]]


class PatchEncoder(nn.Module):
    """
    编码图像局部 Patch（参考 seq_extract 的 cropping_func）
    """
    def __init__(self, patch_size=64, d_model=256):
        super().__init__()
        self.patch_size = patch_size
        # 简单的 CNN 来编码 patch
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),  # target + canvas
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * (patch_size // 8) * (patch_size // 8), d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, target_patch, canvas_patch):
        """
        target_patch: (N, 1, P, P) - 目标图像的 patch
        canvas_patch: (N, 1, P, P) - 当前 canvas 的 patch
        """
        x = torch.cat([target_patch, canvas_patch], dim=1)  # (N, 2, P, P)
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x


def crop_patch(image, center, patch_size, image_size):
    """
    从 image 中裁剪 center 周围的 patch（可微）
    image: (N, 1, H, W)
    center: (N, 2), [0, 1] 归一化坐标
    patch_size: int
    image_size: int
    """
    N = image.shape[0]
    # 转换到像素坐标
    center_px = center * image_size

    # 计算边界：让 pytorch 处理越界情况（自动 padding）
    half_patch = patch_size // 2
    start = center_px - half_patch
    end = center_px + half_patch

    # 用 F.grid_sample 来实现可微的裁剪（参考 spatial transformer）
    # 先创建采样网格
    y = torch.linspace(-1, 1, patch_size, device=image.device)
    x = torch.linspace(-1, 1, patch_size, device=image.device)
    yv, xv = torch.meshgrid(y, x, indexing='ij')

    # 每个样本的网格
    grid = torch.stack([xv, yv], dim=-1)  # (P, P, 2)
    grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)  # (N, P, P, 2)

    # 缩放和平移网格到 center
    scale = half_patch / (image_size / 2)
    offset = (center_px - image_size / 2) / (image_size / 2)

    grid = grid * scale + offset.view(N, 1, 1, 2)

    # 采样
    patch = F.grid_sample(image, grid, align_corners=False, padding_mode='border')
    return patch


class ResNetAutoregressiveExtractor7D(nn.Module):
    def __init__(self, image_size=256, max_seq_len=100, d_model=256, hidden_dim=256, num_heads=None, patch_size=64):
        super().__init__()
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # 自动选择合适的 num_heads（必须能整除 d_model）
        if num_heads is None:
            for n in [16, 8, 4, 2, 1]:
                if d_model % n == 0:
                    num_heads = n
                    break
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads

        # ========== 1. 全局特征编码 ==========
        self.target_backbone = ResNetFeatureBackbone(image_size=image_size, d_model=d_model)
        self.target_pool = nn.LayerNorm(d_model)

        # ========== 2. 局部 Patch 编码 (关键新增！) ==========
        self.patch_encoder = PatchEncoder(patch_size=patch_size, d_model=d_model)

        # ========== 3. 其他状态编码 ==========
        self.canvas_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(d_model),
        )
        self.cursor_mlp = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.prev_stroke_mlp = nn.Sequential(nn.Linear(7, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.step_mlp = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.window_size_mlp = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.LayerNorm(d_model))

        # ========== 4. 特征融合和注意力 ==========
        self.state_query_norm = nn.LayerNorm(d_model * 2)  # patch + canvas + ...
        self.target_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.patch_target_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)

        self.gru_input = nn.Sequential(
            nn.Linear(d_model * 5, hidden_dim),  # patch_feat + target_global + canvas_feat + cursor_feat + prev_feat
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.pen_head = nn.Linear(hidden_dim, 1)
        self.coord_head = nn.Linear(hidden_dim, 4)
        self.param_head = nn.Linear(hidden_dim, 2)

        # ========== 5. 初始 window size ==========
        self.init_window_size = patch_size * 2  # 初始窗口大小

    def encode_target(self, target_mask):
        tokens = self.target_backbone.forward_features(target_mask)
        global_feat = self.target_pool(tokens.mean(dim=1))
        return tokens, global_feat

    def encode_step(self, target_tokens, target_global, target_mask, canvas, cursor,
                   prev_stroke, step_index, hidden=None, window_size=None):
        """
        每次生成前的编码，包含局部 Patch！
        """
        batch_size = canvas.shape[0]
        device = canvas.device

        if window_size is None:
            window_size = torch.full((batch_size, 1), self.init_window_size,
                                   dtype=torch.float32, device=device)

        # ========== 1. 裁剪并编码局部 Patch (关键！) ==========
        # 从 target 和 canvas 裁剪 patch
        target_patch = crop_patch(target_mask, cursor, self.patch_size, self.image_size)
        canvas_patch = crop_patch(canvas, cursor, self.patch_size, self.image_size)

        # 编码 patch
        patch_feat = self.patch_encoder(target_patch, canvas_patch)  # (N, d_model)

        # ========== 2. 编码其他状态 ==========
        canvas_feat = self.canvas_encoder(canvas)
        cursor_feat = self.cursor_mlp(cursor)
        prev_feat = self.prev_stroke_mlp(prev_stroke)
        step_feat = self.step_mlp(step_index)
        win_feat = self.window_size_mlp(window_size / self.image_size)

        # ========== 3. Patch 特征和 Target 特征做注意力 ==========
        patch_query = patch_feat.unsqueeze(1)  # (N, 1, d_model)
        patch_attn_feat, _ = self.patch_target_attn(patch_query, target_tokens, target_tokens)
        patch_attn_feat = patch_attn_feat.squeeze(1)  # (N, d_model)

        # ========== 4. 全局融合和 GRU ==========
        # 组合：使用 patch_attn_feat 代替原来的 attn_feat
        gru_input = self.gru_input(torch.cat([
            patch_attn_feat, target_global, canvas_feat, cursor_feat, prev_feat
        ], dim=-1))

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=canvas.dtype)
        hidden = self.gru(gru_input, hidden)
        return hidden

    def decode_hidden(self, hidden):
        pen_logits = self.pen_head(hidden).squeeze(-1)
        coords = torch.tanh(self.coord_head(hidden))
        params = torch.sigmoid(self.param_head(hidden))
        seq = torch.cat([torch.sigmoid(pen_logits).unsqueeze(-1), coords, params], dim=-1)
        return {'seq': seq, 'pen_logits': pen_logits}

    def forward_step(self, target_tokens, target_global, target_mask, canvas, cursor,
                   prev_stroke, step_index, hidden=None, window_size=None):
        hidden = self.encode_step(target_tokens, target_global, target_mask, canvas,
                                  cursor, prev_stroke, step_index, hidden, window_size)
        output = self.decode_hidden(hidden)
        return output, hidden

    def forward_teacher_forcing(self, target_mask, canvases, cursors, prev_strokes, step_indices):
        """
        Teacher forcing 训练
        """
        target_tokens, target_global = self.encode_target(target_mask)
        hidden = None
        seq_outputs = []
        pen_logits_outputs = []

        batch_size = target_mask.shape[0]
        device = target_mask.device
        window_size = torch.full((batch_size, 1), self.init_window_size, dtype=torch.float32, device=device)

        for i in range(canvases.shape[1]):
            output, hidden = self.forward_step(
                target_tokens,
                target_global,
                target_mask,
                canvases[:, i],
                cursors[:, i],
                prev_strokes[:, i],
                step_indices[:, i],
                hidden,
                window_size
            )
            seq_outputs.append(output['seq'])
            pen_logits_outputs.append(output['pen_logits'])

        return {
            'seq': torch.stack(seq_outputs, dim=1),
            'pen_logits': torch.stack(pen_logits_outputs, dim=1),
        }


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.2f}M, Trainable: {trainable/1e6:.2f}M"


if __name__ == "__main__":
    # 测试模型
    model = StrokeTransformer(d_model=256, nhead=4, num_decoder_layers=3)
    print(f"Model created: {count_parameters(model)}")

    # 测试前向
    image = torch.randn(1, 1, 256, 256)
    target_seq = torch.randn(1, 20, 7)

    # 训练模式
    output = model(image, target_seq, teacher_forcing_ratio=0.5)
    print(f"Training output shape: {output.shape}")

    # 推理模式
    strokes = model.generate(image, max_len=50)
    print(f"Generated strokes shape: {strokes.shape}")
    print("Generated strokes sample:")
    print(strokes[:5])

    # 测试 autoregressive 模型
    print("\nTesting ResNetAutoregressiveExtractor7D...")
    ar_model = ResNetAutoregressiveExtractor7D(image_size=256, max_seq_len=100, d_model=256)
    print(f"AR Model created: {count_parameters(ar_model)}")

    # 测试 forward_step
    target_mask = torch.randn(2, 1, 256, 256)
    canvas = torch.zeros(2, 1, 256, 256)
    cursor = torch.rand(2, 2)
    prev_stroke = torch.zeros(2, 7)
    step_index = torch.tensor([[0.0], [0.0]])

    tokens, global_feat = ar_model.encode_target(target_mask)
    output, hidden = ar_model.forward_step(tokens, global_feat, target_mask, canvas, cursor, prev_stroke, step_index)
    print(f"Step output seq shape: {output['seq'].shape}")

    # 测试 forward_teacher_forcing
    seq_len = 10
    canvases = torch.zeros(2, seq_len, 1, 256, 256)
    cursors = torch.rand(2, seq_len, 2)
    prev_strokes = torch.zeros(2, seq_len, 7)
    step_indices = torch.rand(2, seq_len, 1)

    tf_output = ar_model.forward_teacher_forcing(target_mask, canvases, cursors, prev_strokes, step_indices)
    print(f"Teacher forcing output seq shape: {tf_output['seq'].shape}")

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

            # 检查是否结束（这里用简单启发式：连续几个move就结束）
            # 实际应用可以设计更智能的停止机制

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


class ResNetAutoregressiveExtractor7D(nn.Module):
    def __init__(self, image_size=256, max_seq_len=100, d_model=256, hidden_dim=256, num_heads=None):
        super().__init__()
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # 自动选择合适的 num_heads（必须能整除 d_model）
        if num_heads is None:
            for n in [16, 8, 4, 2, 1]:
                if d_model % n == 0:
                    num_heads = n
                    break
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads

        self.target_backbone = ResNetFeatureBackbone(image_size=image_size, d_model=d_model)
        self.target_pool = nn.LayerNorm(d_model)
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
        self.state_query_norm = nn.LayerNorm(d_model)
        self.target_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.gru_input = nn.Sequential(
            nn.Linear(d_model * 5, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.pen_head = nn.Linear(hidden_dim, 1)
        self.coord_head = nn.Linear(hidden_dim, 4)
        self.param_head = nn.Linear(hidden_dim, 2)

    def encode_target(self, target_mask):
        tokens = self.target_backbone.forward_features(target_mask)
        global_feat = self.target_pool(tokens.mean(dim=1))
        return tokens, global_feat

    def encode_step(self, target_tokens, target_global, canvas, cursor, prev_stroke, step_index, hidden=None):
        canvas_feat = self.canvas_encoder(canvas)
        cursor_feat = self.cursor_mlp(cursor)
        prev_feat = self.prev_stroke_mlp(prev_stroke)
        step_feat = self.step_mlp(step_index)
        query = self.state_query_norm(canvas_feat + cursor_feat + prev_feat + step_feat).unsqueeze(1)
        attn_feat, _ = self.target_attn(query, target_tokens, target_tokens)
        attn_feat = attn_feat.squeeze(1)
        gru_input = self.gru_input(torch.cat([
            attn_feat, target_global, canvas_feat, cursor_feat, prev_feat
        ], dim=-1))
        if hidden is None:
            hidden = torch.zeros(canvas.shape[0], self.hidden_dim, device=canvas.device, dtype=canvas.dtype)
        return self.gru(gru_input, hidden)

    def decode_hidden(self, hidden):
        pen_logits = self.pen_head(hidden).squeeze(-1)
        coords = torch.tanh(self.coord_head(hidden))
        params = torch.sigmoid(self.param_head(hidden))
        seq = torch.cat([torch.sigmoid(pen_logits).unsqueeze(-1), coords, params], dim=-1)
        return {'seq': seq, 'pen_logits': pen_logits}

    def forward_step(self, target_tokens, target_global, canvas, cursor, prev_stroke, step_index, hidden=None):
        hidden = self.encode_step(target_tokens, target_global, canvas, cursor, prev_stroke, step_index, hidden)
        output = self.decode_hidden(hidden)
        return output, hidden

    def forward_teacher_forcing(self, target_mask, canvases, cursors, prev_strokes, step_indices):
        target_tokens, target_global = self.encode_target(target_mask)
        hidden = None
        seq_outputs = []
        pen_logits_outputs = []
        for i in range(canvases.shape[1]):
            output, hidden = self.forward_step(
                target_tokens,
                target_global,
                canvases[:, i],
                cursors[:, i],
                prev_strokes[:, i],
                step_indices[:, i],
                hidden
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

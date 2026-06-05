"""
PyTorch port of seq_extract's RNN cells:
- LayerNormLSTMCell
- HyperLSTMCell

Based on: seq_extract/rnn.py (Magenta)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_init(shape, scale=1.0):
    """Orthogonal initialization for LSTM weights."""
    if len(shape) < 2:
        raise ValueError('Orthogonal init requires at least 2D shape')
    flat_shape = (shape[0], shape[1] if len(shape) == 2 else shape[1] * shape[2])
    a = torch.randn(flat_shape, dtype=torch.float32)
    u, _, v = torch.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q * scale


def lstm_ortho_init(shape, scale=1.0):
    """Orthogonal init for LSTM (split into 4 gates)."""
    size_x = shape[0]
    size_h = shape[1] // 4
    t = torch.zeros(shape, dtype=torch.float32)
    t[:, :size_h] = orthogonal_init((size_x, size_h), scale)
    t[:, size_h:size_h*2] = orthogonal_init((size_x, size_h), scale)
    t[:, size_h*2:size_h*3] = orthogonal_init((size_x, size_h), scale)
    t[:, size_h*3:] = orthogonal_init((size_x, size_h), scale)
    return t


class LayerNorm(nn.Module):
    """Layer normalization for single tensor."""
    def __init__(self, num_units, gamma_start=1.0, epsilon=1e-3, use_bias=True):
        super().__init__()
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.full((num_units,), gamma_start))
        if use_bias:
            self.beta = nn.Parameter(torch.zeros(num_units))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        inv_std = 1.0 / torch.sqrt(var + self.epsilon)
        normalized = (x - mean) * inv_std * self.gamma
        if self.use_bias:
            normalized = normalized + self.beta
        return normalized


class LayerNormLSTMCell(nn.Module):
    """
    LSTM with Layer Norm, Orthogonal Init, Recurrent Dropout (no memory loss).
    PyTorch port of seq_extract's LayerNormLSTMCell.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 forget_bias=1.0,
                 use_recurrent_dropout=False,
                 dropout_keep_prob=0.9):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.use_recurrent_dropout = use_recurrent_dropout
        self.dropout_keep_prob = dropout_keep_prob

        # Weights: W_xh (input to hidden), W_hh (hidden to hidden)
        # F.linear expects weight shape (out_dim, in_dim)
        self.W_xh = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

        # Layer Norms
        self.ln_all = LayerNorm(4 * hidden_size, gamma_start=1.0)
        self.ln_c = LayerNorm(hidden_size, gamma_start=1.0)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        # W_xh: uniform init
        nn.init.xavier_uniform_(self.W_xh)
        # W_hh: orthogonal init (per gate)
        with torch.no_grad():
            self.W_hh.copy_(lstm_ortho_init((4 * self.hidden_size, self.hidden_size), 1.0))

    def forward(self, x, state):
        """
        Args:
            x: (batch_size, input_size)
            state: tuple (h, c) each (batch_size, hidden_size)
        Returns:
            h: new hidden (batch_size, hidden_size)
            new_state: (new_h, new_c)
        """
        h_prev, c_prev = state

        # Linear projections
        xh = F.linear(x, self.W_xh, None)
        hh = F.linear(h_prev, self.W_hh, None)
        pre_activation = xh + hh + self.bias

        # Layer norm for all gates together
        pre_activation = self.ln_all(pre_activation)

        # Split into gates
        i, j, f, o = pre_activation.chunk(4, dim=-1)

        # Apply forget bias
        f = f + self.forget_bias

        # Recurrent dropout (only on j, no memory loss)
        if self.use_recurrent_dropout and self.training:
            dropout_rate = 1.0 - self.dropout_keep_prob
            j = F.dropout(j, dropout_rate, training=True)

        # LSTM update
        g = torch.tanh(j)
        c = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * g
        c = self.ln_c(c)
        h = torch.tanh(c) * torch.sigmoid(o)

        return h, (h, c)


class HyperLSTMCell(nn.Module):
    """
    HyperLSTM with Layer Norm, Orthogonal Init, Recurrent Dropout.
    PyTorch port of seq_extract's HyperLSTMCell.

    Key idea: small "Hyper" LSTM generates dynamic weights for main LSTM.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 forget_bias=1.0,
                 use_recurrent_dropout=False,
                 dropout_keep_prob=0.90,
                 use_layer_norm=True,
                 hyper_num_units=256,
                 hyper_embedding_size=32,
                 hyper_use_recurrent_dropout=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.use_recurrent_dropout = use_recurrent_dropout
        self.dropout_keep_prob = dropout_keep_prob
        self.use_layer_norm = use_layer_norm
        self.hyper_num_units = hyper_num_units
        self.hyper_embedding_size = hyper_embedding_size
        self.hyper_use_recurrent_dropout = hyper_use_recurrent_dropout

        total_hidden = hidden_size + hyper_num_units
        self.total_hidden = total_hidden

        # Hyper cell
        if use_layer_norm:
            self.hyper_cell = LayerNormLSTMCell(
                input_size + hidden_size,  # input = [x, h_prev]
                hyper_num_units,
                forget_bias=forget_bias,
                use_recurrent_dropout=hyper_use_recurrent_dropout,
                dropout_keep_prob=dropout_keep_prob
            )
        else:
            self.hyper_cell = LayerNormLSTMCell(
                input_size + hidden_size,
                hyper_num_units,
                forget_bias=forget_bias,
                use_recurrent_dropout=hyper_use_recurrent_dropout,
                dropout_keep_prob=dropout_keep_prob
            )

        # Main LSTM weights
        self.W_xh = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

        # Hyper projections
        self.zw = nn.Linear(hyper_num_units, hyper_embedding_size)
        self.alpha_ix = nn.Linear(hyper_embedding_size, hidden_size, bias=False)
        self.alpha_jx = nn.Linear(hyper_embedding_size, hidden_size, bias=False)
        self.alpha_fx = nn.Linear(hyper_embedding_size, hidden_size, bias=False)
        self.alpha_ox = nn.Linear(hyper_embedding_size, hidden_size, bias=False)
        self.alpha_ih = nn.Linear(hyper_embedding_size, hidden_size, bias=False)
        self.alpha_jh = nn.Linear(hyper_embedding_size, hidden_size, bias=False)
        self.alpha_fh = nn.Linear(hyper_embedding_size, hidden_size, bias=False)
        self.alpha_oh = nn.Linear(hyper_embedding_size, hidden_size, bias=False)
        self.zb = nn.Linear(hyper_num_units, hyper_embedding_size)
        self.beta_ih = nn.Linear(hyper_embedding_size, hidden_size, bias=False)
        self.beta_jh = nn.Linear(hyper_embedding_size, hidden_size, bias=False)
        self.beta_fh = nn.Linear(hyper_embedding_size, hidden_size, bias=False)
        self.beta_oh = nn.Linear(hyper_embedding_size, hidden_size, bias=False)

        # Layer Norms
        self.ln_all = LayerNorm(4 * hidden_size, gamma_start=1.0)
        self.ln_c = LayerNorm(hidden_size, gamma_start=1.0)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_xh)
        with torch.no_grad():
            self.W_hh.copy_(lstm_ortho_init((4 * self.hidden_size, self.hidden_size), 1.0))
        nn.init.constant_(self.zw.weight, 0.0)
        nn.init.constant_(self.zw.bias, 1.0)
        init_gamma = 0.10
        nn.init.constant_(self.alpha_ix.weight, init_gamma / self.hyper_embedding_size)
        nn.init.constant_(self.alpha_jx.weight, init_gamma / self.hyper_embedding_size)
        nn.init.constant_(self.alpha_fx.weight, init_gamma / self.hyper_embedding_size)
        nn.init.constant_(self.alpha_ox.weight, init_gamma / self.hyper_embedding_size)
        nn.init.constant_(self.alpha_ih.weight, init_gamma / self.hyper_embedding_size)
        nn.init.constant_(self.alpha_jh.weight, init_gamma / self.hyper_embedding_size)
        nn.init.constant_(self.alpha_fh.weight, init_gamma / self.hyper_embedding_size)
        nn.init.constant_(self.alpha_oh.weight, init_gamma / self.hyper_embedding_size)
        nn.init.normal_(self.zb.weight, 0.0, 0.01)
        nn.init.constant_(self.zb.bias, 0.0)
        nn.init.constant_(self.beta_ih.weight, 0.0)
        nn.init.constant_(self.beta_jh.weight, 0.0)
        nn.init.constant_(self.beta_fh.weight, 0.0)
        nn.init.constant_(self.beta_oh.weight, 0.0)

    def hyper_norm(self, layer, alpha_layer, beta_layer, use_bias=True):
        """Apply hyper normalization."""
        zw = torch.tanh(self.zw(self.hyper_output))
        alpha = alpha_layer(zw)
        result = alpha * layer
        if use_bias:
            zb = torch.tanh(self.zb(self.hyper_output))
            beta = beta_layer(zb)
            result = result + beta
        return result

    def forward(self, x, state):
        """
        Args:
            x: (batch_size, input_size)
            state: tuple (total_h, total_c)
                total_h: concat([h, hyper_h]) (batch_size, hidden_size + hyper_num_units)
                total_c: concat([c, hyper_c]) (batch_size, hidden_size + hyper_num_units)
        Returns:
            h: new hidden (batch_size, hidden_size)
            new_state: (new_total_h, new_total_c)
        """
        total_h_prev, total_c_prev = state
        h_prev = total_h_prev[:, :self.hidden_size]
        c_prev = total_c_prev[:, :self.hidden_size]
        hyper_h_prev = total_h_prev[:, self.hidden_size:]
        hyper_c_prev = total_c_prev[:, self.hidden_size:]
        hyper_state_prev = (hyper_h_prev, hyper_c_prev)

        # Hyper LSTM forward
        hyper_input = torch.cat([x, h_prev], dim=-1)
        hyper_h, hyper_state = self.hyper_cell(hyper_input, hyper_state_prev)
        self.hyper_output = hyper_h

        # Main LSTM linear projections
        xh = F.linear(x, self.W_xh, None)
        hh = F.linear(h_prev, self.W_hh, None)

        # Split into gates
        ix, jx, fx, ox = xh.chunk(4, dim=-1)
        ih, jh, fh, oh = hh.chunk(4, dim=-1)
        ib, jb, fb, ob = self.bias.chunk(4, dim=-1)

        # Hyper normalize (dynamic modulation)
        ix = self.hyper_norm(ix, self.alpha_ix, None, use_bias=False)
        jx = self.hyper_norm(jx, self.alpha_jx, None, use_bias=False)
        fx = self.hyper_norm(fx, self.alpha_fx, None, use_bias=False)
        ox = self.hyper_norm(ox, self.alpha_ox, None, use_bias=False)
        ih = self.hyper_norm(ih, self.alpha_ih, self.beta_ih, use_bias=True)
        jh = self.hyper_norm(jh, self.alpha_jh, self.beta_jh, use_bias=True)
        fh = self.hyper_norm(fh, self.alpha_fh, self.beta_fh, use_bias=True)
        oh = self.hyper_norm(oh, self.alpha_oh, self.beta_oh, use_bias=True)

        # Gates
        i = ix + ih + ib
        j = jx + jh + jb
        f = fx + fh + fb
        o = ox + oh + ob

        # Concat and layer norm
        pre_activation = torch.cat([i, j, f, o], dim=-1)
        pre_activation = self.ln_all(pre_activation)
        i, j, f, o = pre_activation.chunk(4, dim=-1)

        # Apply forget bias
        f = f + self.forget_bias

        # Recurrent dropout
        if self.use_recurrent_dropout and self.training:
            dropout_rate = 1.0 - self.dropout_keep_prob
            j = F.dropout(j, dropout_rate, training=True)

        # LSTM update
        g = torch.tanh(j)
        c = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * g
        c = self.ln_c(c)
        h = torch.tanh(c) * torch.sigmoid(o)

        # New total state
        new_hyper_h, new_hyper_c = hyper_state
        new_total_h = torch.cat([h, new_hyper_h], dim=-1)
        new_total_c = torch.cat([c, new_hyper_c], dim=-1)
        new_total_state = (new_total_h, new_total_c)

        return h, new_total_state

    def get_initial_state(self, batch_size, device):
        """Get initial state for HyperLSTM."""
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        hyper_h = torch.zeros(batch_size, self.hyper_num_units, device=device)
        hyper_c = torch.zeros(batch_size, self.hyper_num_units, device=device)
        total_h = torch.cat([h, hyper_h], dim=-1)
        total_c = torch.cat([c, hyper_c], dim=-1)
        return (total_h, total_c)

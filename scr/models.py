import math
import torch
from torch import nn
from typing import Sequence, List, Optional, Tuple


class FeatureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalConvGRU(nn.Module):
    def __init__(self, state_dim: int, conv_channels: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(state_dim, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gru = nn.GRU(input_size=conv_channels, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, state_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


class TemporalDilatedCNN(nn.Module):
    def __init__(
        self,
        state_dim: int,
        channels: int = 96,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = state_dim
        dilation = 1
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=dilation * (kernel_size - 1) // 2,
                    dilation=dilation,
                )
            )
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_channels = channels
            dilation *= 2
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels, state_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.backbone(x)
        x = self.head(x)
        x = x.permute(0, 2, 1)
        return x[:, -1, :]


# --- НОВАЯ АРХИТЕКТУРА: ResNet-1D ---
class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, 
            padding=dilation * (kernel_size - 1) // 2, 
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, 
            padding=dilation * (kernel_size - 1) // 2, 
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # Skip connection
        out = self.act2(out)
        return out

# --- Pure PyTorch Mamba Implementation ---

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state

        # Входная проекция (сразу на x и z)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Conv (как в Mamba - depthwise)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = nn.SiLU()

        # Проекции для SSM параметров
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D параметры (A и D)
        A = torch.repeat_interleave(
            torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0),
            self.d_inner,
            dim=0
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in real parameter
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Выходная проекция
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch, seq_len, d = x.shape
        
        # 1. Проекция
        xz = self.in_proj(x) # [B, L, 2*d_inner]
        x, z = xz.chunk(2, dim=-1) # Разделяем на сигнал и гейт

        # 2. Свертка
        x = x.transpose(1, 2) # [B, d_inner, L]
        x = self.conv1d(x)[:, :, :seq_len] # Causal padding trick
        x = x.transpose(1, 2) # [B, L, d_inner]
        x = self.activation(x)

        # 3. SSM (Selective Scan) - Pure PyTorch implementation
        # Вычисляем параметры delta, B, C
        x_dbl = self.x_proj(x) # [B, L, dt_rank + 2*d_state]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(dt) # [B, L, d_inner]
        dt = torch.nn.functional.softplus(dt) # Delta всегда > 0

        A = -torch.exp(self.A_log.to(dtype=x.dtype))  # [d_inner, d_state]
        
        # Дискретизация (Zero-Order Hold)
        # DA = exp(delta * A)
        # Это упрощенная версия сканирования (медленная, но работает везде)
        # Для окна 32-64 это быстро.
        
        y = self.selective_scan(x, dt, A, B, C, self.D)

        # 4. Gating и выход
        y = y * self.activation(z)
        y = self.out_proj(y)
        return self.dropout(y)

    def selective_scan(self, u, dt, A, B, C, D):
        # u: [B, L, D_in]
        # dt: [B, L, D_in]
        # A: [D_in, N]
        # B: [B, L, N]
        # C: [B, L, N]
        # D: [D_in]

        device = u.device
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        # Приводим тензоры к общему dtype/устройству
        A = A.to(device=device, dtype=u.dtype)
        D = D.to(device=device, dtype=u.dtype)
        dt = dt.to(device=device, dtype=u.dtype)
        B = B.to(device=device, dtype=u.dtype)
        C = C.to(device=device, dtype=u.dtype)

        # Дискретизация матриц перехода и управляющих сигналов
        # dA: (B, L, D_in, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # dB: (B, L, D_in, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)

        # Источник: (B, L, D_in, d_state)
        source = dB * u.unsqueeze(-1)

        # Решение линейного рекуррентного уравнения через накопление сумм/произведений
        # p_t = prod_{i<=t} dA_i
        p = torch.cumprod(dA, dim=1)
        eps = torch.finfo(u.dtype).eps
        p_safe = p.clamp_min(eps)

        # g_t = sum_{k<=t} source_k / p_k
        g = torch.cumsum(source / p_safe, dim=1)

        # h_t = p_t * g_t
        h = g * p

        # Выход: y_t = sum_n h_{t,:,n} * C_{t,n} + D * u_t
        y = (h * C.unsqueeze(2)).sum(dim=-1) + D * u
        return y

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos/sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int, device: torch.device):
        return (
            self.cos_cached[:seq_len].to(device),
            self.sin_cached[:seq_len].to(device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # cos, sin: [seq_len, head_dim]
    # q, k: [batch, heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEMultiHeadAttention(nn.Module):
    """Multi-Head Attention с RoPE."""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=max_seq_len)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(seq_len, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class ImprovedTransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block с RoPE."""
    
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, nhead, dropout, max_seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class TemporalTransformerV2(nn.Module):
    """Улучшенный Transformer с RoPE и Pre-LayerNorm."""
    
    def __init__(
        self,
        state_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_window: int = 128,
    ):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, d_model)
        
        self.blocks = nn.ModuleList([
            ImprovedTransformerBlock(d_model, nhead, dim_feedforward, dropout, max_window)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Голова с дополнительным слоем
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model, state_dim),
        )
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mask = self._causal_mask(seq_len, x.device)
        
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x[:, -1, :])
        return self.head(x)

class TemporalMamba(nn.Module):
    def __init__(
        self,
        state_dim: int,
        d_model: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        expand: int = 2,
        d_state: int = 16,
    ):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, d_model)
        
        layers = []
        for _ in range(num_layers):
            layers.append(MambaBlock(d_model, d_state=d_state, expand=expand, dropout=dropout))
            layers.append(nn.LayerNorm(d_model)) # Mamba любит Norm между блоками
            
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d_model, state_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, state_dim]
        x = self.input_proj(x)
        x = self.backbone(x)
        # Берем последний шаг
        return self.head(x[:, -1, :])

class TemporalDeepGRU(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        
        # GRU с LayerNorm работает намного стабильнее и позволяет делать сеть глубокой
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False 
        )
        
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, state_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, state_dim]
        x = self.input_proj(x)
        
        # GRU
        out, _ = self.gru(x)
        
        # Берем последний шаг
        last = out[:, -1, :]
        
        # Нормализация перед выходом (стабилизирует)
        last = self.ln(last)
        
        return self.head(last)

class TemporalResNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        channels: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(state_dim, channels, kernel_size=1)
        
        layers = []
        dilation = 1
        for _ in range(num_blocks):
            layers.append(ResBlock(channels, kernel_size, dilation, dropout))
            dilation *= 2
            
        self.backbone = nn.Sequential(*layers)
        
        # УБРАЛИ AdaptiveAvgPool1d
        # УБРАЛИ Flatten
        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, state_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, state_dim]
        x = x.permute(0, 2, 1) 
        x = self.input_proj(x)
        x = self.backbone(x)
        
        # БЕРЕМ ПОСЛЕДНИЙ ШАГ: [batch, channels, seq_len] -> [batch, channels]
        last_step = x[:, :, -1]
        
        return self.head(last_step)
# ------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_window: Optional[int] = None,
        pe_type: str = "learned",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.d_model = d_model
        self.pe_type = pe_type
        
        self.input_proj = nn.Linear(state_dim, d_model)
        
        # Используем единое имя для обоих типов PE, чтобы избежать конфликтов при загрузке
        if self.pe_type == "sinusoidal":
            self.pos_encoder = PositionalEncoding(d_model, max_len=max_window or 500)
            # Создаём dummy parameter для совместимости с learned PE
            self.positional = nn.Parameter(
                torch.zeros(1, 1, dtype=torch.float32), requires_grad=False
            )
        else:
            self.positional = nn.Parameter(
                torch.zeros(max_window or 128, d_model, dtype=torch.float32)
            )
            # Создаём dummy module для совместимости с sinusoidal PE
            self.pos_encoder = nn.Identity()
        
        activation = "gelu" if self.pe_type == "sinusoidal" else "relu"
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, state_dim)
        
        if self.pe_type == "sinusoidal":
            self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        device = x.device
        
        token = self.input_proj(x)
        
        if self.pe_type == "sinusoidal":
            token = self.pos_encoder(token)
        else:
            pos_embedding = self.positional[:seq_len].unsqueeze(0).to(device)
            token = token + pos_embedding
            
        mask = self._causal_mask(seq_len, device)
        encoded = self.encoder(token, mask=mask)
        last = encoded[:, -1, :]
        last = self.norm(last)
        return self.head(last)
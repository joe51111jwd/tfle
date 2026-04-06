import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .bitlinear import BitLinear, RMSNorm
from .config import NovaConfig


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 500000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = positions[:, None] * freqs[None, :]
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    return (
        torch.repeat_interleave(cos_vals, 2, dim=-1),
        torch.repeat_interleave(sin_vals, 2, dim=-1),
    )


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos_half = cos[..., ::2]
    sin_half = sin[..., ::2]
    o1 = x1 * cos_half - x2 * sin_half
    o2 = x2 * cos_half + x1 * sin_half
    return torch.stack([o1, o2], dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: NovaConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_groups = config.n_heads // config.n_kv_heads
        self.gradient_checkpointing = config.gradient_checkpointing

        self.norm = RMSNorm(config.hidden_dim)
        self.q_proj = BitLinear(config.hidden_dim, config.n_heads * self.head_dim)
        self.k_proj = BitLinear(config.hidden_dim, config.n_kv_heads * self.head_dim)
        self.v_proj = BitLinear(config.hidden_dim, config.n_kv_heads * self.head_dim)
        self.o_proj = BitLinear(config.n_heads * self.head_dim, config.hidden_dim)

        cos, sin = precompute_rope_freqs(self.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("_rope_cos", cos)
        self.register_buffer("_rope_sin", sin)

    def _attn_forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        B, L, _ = x.shape
        residual = x
        x = self.norm(x)

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        q = apply_rope(q, self._rope_cos.to(q.device), self._rope_sin.to(q.device))
        k = apply_rope(k, self._rope_cos.to(k.device), self._rope_sin.to(k.device))

        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        if mask is not None:
            attn = attn + mask
        attn = torch.softmax(attn, dim=-1)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out) + residual

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._attn_forward, x, mask, use_reentrant=False
            )
        return self._attn_forward(x, mask)

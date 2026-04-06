import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear, RMSNorm
from .config import NovaConfig


@torch.jit.script
def selective_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    batch, seq_len, d_inner = x.shape
    d_state = B_mat.shape[-1]

    dt_expanded = dt.unsqueeze(-1)
    A_expanded = A.unsqueeze(0).unsqueeze(0)
    dA = torch.exp(A_expanded * dt_expanded)

    x_expanded = x.unsqueeze(-1)
    B_expanded = B_mat.unsqueeze(2)
    dBx = dt_expanded * B_expanded * x_expanded

    h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
    outputs: list[torch.Tensor] = []
    for t in range(seq_len):
        h = dA[:, t] * h + dBx[:, t]
        y_t = torch.sum(h * C_mat[:, t].unsqueeze(1), dim=-1)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1)
    return y + x * D


class MambaBlock(nn.Module):
    def __init__(self, config: NovaConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.d_state = config.mamba_d_state
        self.d_conv = config.mamba_d_conv
        self.d_inner = config.mamba_d_inner
        self.dt_rank = config.dt_rank

        self.norm = RMSNorm(config.hidden_dim)
        self.in_proj = BitLinear(config.hidden_dim, self.d_inner * 2)

        self.conv_weight = nn.Parameter(torch.randn(self.d_inner, self.d_conv) * 0.1)
        self.conv_bias = nn.Parameter(torch.zeros(self.d_inner))

        self.dt_proj = BitLinear(self.d_inner, self.dt_rank)
        self.dt_proj_down = nn.Linear(self.dt_rank, self.d_inner)
        self.B_proj = BitLinear(self.d_inner, self.d_state)
        self.C_proj = BitLinear(self.d_inner, self.d_state)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = BitLinear(self.d_inner, config.hidden_dim)

    def _causal_conv1d(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        pad_len = self.d_conv - 1
        x_padded = F.pad(x, (0, 0, pad_len, 0))
        out = torch.zeros_like(x)
        for k in range(self.d_conv):
            out = out + x_padded[:, pad_len - k: pad_len - k + L, :] * self.conv_weight[:, k]
        return out + self.conv_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_path, z = torch.chunk(xz, 2, dim=-1)

        x_path = self._causal_conv1d(x_path)
        x_path = F.silu(x_path)

        dt = F.softplus(self.dt_proj_down(self.dt_proj(x_path)))
        B_mat = self.B_proj(x_path)
        C_mat = self.C_proj(x_path)

        A = -torch.exp(self.A_log)
        y = selective_scan(x_path, dt, A, B_mat, C_mat, self.D)

        y = y * F.silu(z)
        return self.out_proj(y) + residual

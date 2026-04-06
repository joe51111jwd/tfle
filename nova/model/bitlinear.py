import math

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class BitLinear(nn.Module):
    """Ternary linear layer with STE pass-through.

    Weights stored in float32, quantized to {-1, 0, +1} * alpha in forward.
    Activations quantized to INT8 range via absmax with STE.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        scale = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.norm = RMSNorm(in_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        alpha = self.weight.abs().mean().clamp(min=1e-10)
        w_q = torch.clamp(torch.round(self.weight / alpha), -1, 1)
        w_q = self.weight + (w_q - self.weight).detach()

        gamma = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-10)
        x_q = torch.clamp(torch.round(x * 127.0 / gamma), -128, 127)
        x_q = x + (x_q - x).detach()

        y = x_q @ w_q.T * (alpha * gamma / 127.0)
        if self.bias is not None:
            y = y + self.bias
        return y

    def ternary_weights(self) -> torch.Tensor:
        alpha = self.weight.abs().mean().clamp(min=1e-10)
        return torch.clamp(torch.round(self.weight / alpha), -1, 1)

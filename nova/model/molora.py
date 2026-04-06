import torch
import torch.nn as nn

from .config import NovaConfig


class LoRAAdapter(nn.Module):
    """Single LoRA expert. A and B are float32 (not BitLinear)."""

    def __init__(self, dim: int, rank: int, alpha: float):
        super().__init__()
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.randn(dim, rank) * (1.0 / rank))
        self.B = nn.Parameter(torch.zeros(rank, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A @ self.B) * self.scaling


class MoLoRA(nn.Module):
    """Mixture of LoRA with top-k routing over named experts."""

    EXPERT_NAMES = ["math", "code", "planning", "self_eval", "tool_use"]

    def __init__(self, config: NovaConfig):
        super().__init__()
        self.n_experts = config.molora_experts
        self.top_k = config.molora_top_k
        dim = config.hidden_dim

        self.experts = nn.ModuleList([
            LoRAAdapter(dim, config.molora_rank, config.molora_alpha)
            for _ in range(self.n_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.n_experts),
        )

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        routing_logits = self.router(x)
        routing_weights = torch.softmax(routing_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        expert_outputs = torch.stack([e(x) for e in self.experts], dim=-2)

        combined = torch.zeros_like(base_output)
        for k in range(self.top_k):
            idx = top_k_indices[..., k : k + 1]
            weight = top_k_weights[..., k : k + 1]
            expert_out = torch.gather(
                expert_outputs, -2,
                idx.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]),
            ).squeeze(-2)
            combined = combined + expert_out * weight

        return base_output + combined

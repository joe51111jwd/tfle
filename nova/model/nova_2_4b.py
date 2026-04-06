import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import NovaConfig, NOVA_2_4B
from .bitlinear import BitLinear, RMSNorm
from .mamba_block import MambaBlock
from .attention import GroupedQueryAttention
from .molora import MoLoRA


class FFN(nn.Module):
    def __init__(self, config: NovaConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_dim)
        self.up = BitLinear(config.hidden_dim, config.d_ff)
        self.down = BitLinear(config.d_ff, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.up(x)
        x = F.relu(x) ** 2
        x = self.down(x)
        return x + residual


class TransformerBlock(nn.Module):
    def __init__(self, config: NovaConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.ffn = FFN(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.attention(x, mask=mask)
        x = self.ffn(x)
        return x


class Nova2_4B(nn.Module):
    """NOVA 2.4B: 32 layers (24 Mamba + 8 Attention), hidden=2048, vocab=50261.

    MMMA pattern with MoLoRA on attention layers, BitLinear everywhere,
    ReLU-squared FFN, and RMSNorm.
    """

    def __init__(self, config: NovaConfig):
        super().__init__()
        self.config = config
        self._gradient_checkpointing = config.gradient_checkpointing

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layer_pattern = config.layer_pattern

        layers: list[nn.Module] = []
        for lt in self.layer_pattern:
            if lt == "A":
                layers.append(TransformerBlock(config))
            else:
                layers.append(MambaBlock(config))
        self.layers = nn.ModuleList(layers)

        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        n_attn = sum(1 for lt in self.layer_pattern if lt == "A")
        self.moloras = nn.ModuleList([MoLoRA(config) for _ in range(n_attn)])

    @classmethod
    def from_config(cls, config: NovaConfig) -> "Nova2_4B":
        return cls(config)

    @property
    def gradient_checkpointing(self) -> bool:
        return self._gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value: bool) -> None:
        self._gradient_checkpointing = value
        self.config.gradient_checkpointing = value
        for layer in self.layers:
            if isinstance(layer, TransformerBlock):
                layer.attention.gradient_checkpointing = value

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)

        if mask is None:
            seq_len = input_ids.shape[1]
            mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        attn_idx = 0
        for layer, lt in zip(self.layers, self.layer_pattern):
            if lt == "A":
                base_out = layer(x, mask=mask)
                x = self.moloras[attn_idx](x, base_out)
                attn_idx += 1
            else:
                if self._gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=False
                    )
                else:
                    x = layer(x)

        x = self.norm(x)
        return self.lm_head(x)

    def count_parameters(self) -> dict[str, int | float]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_M": total / 1e6,
            "total_B": total / 1e9,
        }

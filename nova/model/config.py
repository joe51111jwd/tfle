from dataclasses import dataclass


@dataclass
class NovaConfig:
    n_layers: int = 32
    hidden_dim: int = 2560
    vocab_size: int = 50261
    n_heads: int = 20
    n_kv_heads: int = 4
    d_ff: int = 6912
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    molora_experts: int = 5
    molora_top_k: int = 2
    molora_rank: int = 16
    molora_alpha: float = 32.0
    max_seq_len: int = 4096
    rope_theta: float = 500000.0
    gradient_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.n_heads

    @property
    def mamba_d_inner(self) -> int:
        return self.hidden_dim * self.mamba_expand

    @property
    def dt_rank(self) -> int:
        import math
        return math.ceil(self.hidden_dim / 16)

    @property
    def n_mamba_layers(self) -> int:
        return self.n_layers - self.n_attn_layers

    @property
    def n_attn_layers(self) -> int:
        return self.n_layers // 4

    @property
    def layer_pattern(self) -> list[str]:
        pattern = ["M", "M", "M", "A"] * (self.n_layers // 4)
        remaining = self.n_layers - len(pattern)
        pattern.extend(["M"] * remaining)
        return pattern


NOVA_2_4B = NovaConfig()

NOVA_10M = NovaConfig(
    n_layers=12,
    hidden_dim=640,
    vocab_size=32000,
    n_heads=10,
    n_kv_heads=2,
    d_ff=1728,
    mamba_d_state=16,
    mamba_d_conv=4,
    mamba_expand=2,
    molora_experts=5,
    molora_top_k=2,
    molora_rank=8,
    molora_alpha=16.0,
    max_seq_len=512,
    rope_theta=500000.0,
)

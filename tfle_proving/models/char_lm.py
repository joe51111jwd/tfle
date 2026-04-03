"""Character-level language model: float32 embedding + ternary TFLE hidden layers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from existing TFLE codebase
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tfle.layers import TFLELayer, generate_k_proposals, ternary_matmul
from tfle.config import TFLEConfig, FitnessType, CoolingSchedule, resolve_device
from tfle.baseline import STETernaryLinear, ste_ternary


class CharLM:
    """Character-level LM with float32 embedding + ternary hidden layers.

    Architecture:
        Embedding(256 -> embed_dim) [float32, backprop]
        Flatten(context_len * embed_dim)
        TFLELayer(flat_dim -> hidden[0]) [ternary, TFLE]
        TFLELayer(hidden[0] -> hidden[1]) [ternary, TFLE]
        ...
        TFLELayer(hidden[-1] -> vocab_size) [ternary, TFLE]
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 32,
        context_len: int = 128,
        hidden_sizes: list[int] | None = None,
        config: TFLEConfig | None = None,
        device: torch.device | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [512, 512, 256]
        if config is None:
            config = TFLEConfig()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.flat_dim = context_len * embed_dim
        self.device = device or torch.device("cpu")

        # Float32 embedding (trained with backprop)
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)

        # Ternary layers: [flat_dim, *hidden_sizes, vocab_size]
        layer_dims = [self.flat_dim] + hidden_sizes + [vocab_size]
        self.layers: list[TFLELayer] = []
        for i, (in_f, out_f) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.layers.append(
                TFLELayer(in_f, out_f, config, layer_idx=i, device=self.device)
            )

        self.config = config

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed and flatten. x: (B, context_len) -> (B, flat_dim)."""
        emb = self.embedding(x)
        return emb.reshape(emb.shape[0], -1)

    def ternary_forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward through ternary layers only. h: (B, flat_dim) -> (B, vocab_size)."""
        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.layer_norm(h, h.shape[-1:])
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward. x: (B, context_len) -> (B, vocab_size) logits."""
        return self.ternary_forward(self.embed(x))

    @torch.no_grad()
    def generate(
        self, prompt_tokens: list[int], max_length: int, temperature: float = 0.8
    ) -> list[int]:
        """Auto-regressive text generation."""
        tokens = list(prompt_tokens)
        for _ in range(max_length):
            ctx = tokens[-self.context_len :]
            if len(ctx) < self.context_len:
                ctx = [0] * (self.context_len - len(ctx)) + ctx
            x = torch.tensor([ctx], dtype=torch.long, device=self.device)
            logits = self.forward(x)
            if temperature > 0:
                probs = F.softmax(logits[0] / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
            else:
                next_tok = logits[0].argmax().item()
            tokens.append(next_tok)
        return tokens

    def generate_text(self, prompt: str, length: int = 200, temperature: float = 0.8) -> str:
        tokens = [ord(c) % 256 for c in prompt]
        out_tokens = self.generate(tokens, length, temperature)
        return "".join(chr(t) for t in out_tokens)

    def get_ternary_param_count(self) -> int:
        return sum(l.in_features * l.out_features for l in self.layers)

    def get_embed_param_count(self) -> int:
        return self.vocab_size * self.embed_dim

    def save_checkpoint(self, path: str):
        state = {
            "embedding": self.embedding.state_dict(),
            "weights": [l.weights.cpu().clone() for l in self.layers],
            "traces": [],
            "arch": {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "context_len": self.context_len,
                "hidden_sizes": [
                    l.out_features
                    for l in self.layers[:-1]
                ],
            },
        }
        # Save traces for resuming
        for l in self.layers:
            if self.config.separate_pos_neg_traces:
                state["traces"].append({
                    "success": l.success_traces.cpu().clone(),
                    "error": l.error_traces.cpu().clone(),
                })
            else:
                state["traces"].append({"traces": l.traces.cpu().clone()})
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        state = torch.load(path, weights_only=False, map_location="cpu")
        self.embedding.load_state_dict(state["embedding"])
        self.embedding.to(self.device)
        for i, w in enumerate(state["weights"]):
            self.layers[i].weights = w.to(self.layers[i].device)
        if state.get("traces"):
            for i, t in enumerate(state["traces"]):
                if "success" in t:
                    self.layers[i].success_traces = t["success"].to(self.layers[i].device)
                    self.layers[i].error_traces = t["error"].to(self.layers[i].device)
                else:
                    self.layers[i].traces = t["traces"].to(self.layers[i].device)


class STECharLM(nn.Module):
    """Same architecture as CharLM but trained with backprop + STE."""

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 32,
        context_len: int = 128,
        hidden_sizes: list[int] | None = None,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 512, 256]

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.flat_dim = context_len * embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        layer_dims = [self.flat_dim] + hidden_sizes + [vocab_size]
        self.ternary_layers = nn.ModuleList()
        for in_f, out_f in zip(layer_dims[:-1], layer_dims[1:]):
            self.ternary_layers.append(STETernaryLinear(in_f, out_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        h = emb.reshape(emb.shape[0], -1)
        for i, layer in enumerate(self.ternary_layers):
            h = layer(h)
            if i < len(self.ternary_layers) - 1:
                h = F.relu(h)
                h = F.layer_norm(h, h.shape[-1:])
        return h

    @torch.no_grad()
    def generate(
        self, prompt_tokens: list[int], max_length: int, temperature: float = 0.8
    ) -> list[int]:
        tokens = list(prompt_tokens)
        for _ in range(max_length):
            ctx = tokens[-self.context_len :]
            if len(ctx) < self.context_len:
                ctx = [0] * (self.context_len - len(ctx)) + ctx
            x = torch.tensor([ctx], dtype=torch.long, device=next(self.parameters()).device)
            logits = self.forward(x)
            if temperature > 0:
                probs = F.softmax(logits[0] / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
            else:
                next_tok = logits[0].argmax().item()
            tokens.append(next_tok)
        return tokens

    def generate_text(self, prompt: str, length: int = 200, temperature: float = 0.8) -> str:
        tokens = [ord(c) % 256 for c in prompt]
        out_tokens = self.generate(tokens, length, temperature)
        return "".join(chr(t) for t in out_tokens)

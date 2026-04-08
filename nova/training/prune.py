"""Prune-and-Ternarize: model pruning for NOVA (CONDITIONAL).

CONDITIONAL: Only used if pruning tests pass. Do not activate in production
without verifying that accuracy remains acceptable after pruning.

Three pruning strategies:
- Uniform depth pruning: remove layers, keeping every Nth
- Importance-based width pruning: prune attention heads by L1 norm,
  FFN neurons by activation magnitude
- Qwen-to-NOVA transfer: load a pretrained Qwen model and map weights
  into the NOVA hybrid architecture (Phase 0 entry point)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from nova.model.config import NovaConfig
from nova.model.nova_2_4b import Nova2_4B

logger = logging.getLogger(__name__)


@dataclass
class PruneConfig:
    strategy: str = "depth"  # "depth" or "width"
    keep_every_n: int = 2
    target_ratio: float = 0.5
    calibration_batches: int = 50


class ModelPruner:
    """CONDITIONAL: Prune model layers or heads/neurons by importance.

    This module is experimental. Only use after validating that pruned
    model accuracy stays within acceptable bounds.
    """

    def __init__(self, config: PruneConfig | None = None):
        self.config = config or PruneConfig()
        self.head_importance: dict[str, torch.Tensor] = {}
        self.neuron_importance: dict[str, torch.Tensor] = {}
        self._calibrated = False

    def uniform_depth_prune(
        self,
        model: nn.Module,
        keep_every_n: int | None = None,
    ) -> nn.Module:
        """Remove layers, keeping every Nth layer.

        For a model with layers [0,1,2,...,N-1] and keep_every_n=2,
        keeps layers [0, 2, 4, ...]. Adjusts layer pattern for
        Mamba/Attention hybrid models.
        """
        n = keep_every_n or self.config.keep_every_n

        if not hasattr(model, "layers"):
            logger.warning("Model has no 'layers' attribute, skipping depth prune")
            return model

        original_count = len(model.layers)
        keep_indices = list(range(0, original_count, n))

        # always keep the first and last layer
        if 0 not in keep_indices:
            keep_indices.insert(0, 0)
        if (original_count - 1) not in keep_indices:
            keep_indices.append(original_count - 1)

        kept_layers = nn.ModuleList([model.layers[i] for i in keep_indices])
        model.layers = kept_layers

        # update layer pattern if the model tracks it
        if hasattr(model, "layer_pattern"):
            model.layer_pattern = [model.layer_pattern[i] for i in keep_indices]

        # re-index MoLoRA if present
        if hasattr(model, "moloras") and hasattr(model, "layer_pattern"):
            n_attn = sum(1 for lt in model.layer_pattern if lt == "A")
            if len(model.moloras) > n_attn:
                model.moloras = nn.ModuleList(list(model.moloras)[:n_attn])

        logger.info(
            f"Depth prune: {original_count} -> {len(kept_layers)} layers "
            f"(keep_every_n={n})"
        )
        return model

    def calibrate(
        self,
        model: nn.Module,
        data_loader,
        num_batches: int | None = None,
    ):
        """Run data through the model to compute importance scores.

        Head importance: L1 norm of attention output weights per head.
        Neuron importance: mean activation magnitude in FFN intermediate layers.
        """
        num_batches = num_batches or self.config.calibration_batches
        self.head_importance = {}
        self.neuron_importance = {}

        # collect activation hooks
        hooks = []
        activation_sums: dict[str, torch.Tensor] = {}
        activation_counts: dict[str, int] = {}

        def _make_hook(name: str):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor) and output.dim() >= 2:
                    # average over all dims except the last (feature dim)
                    dims = tuple(range(output.dim() - 1))
                    act = output.detach().abs().mean(dim=dims)
                    if name not in activation_sums:
                        activation_sums[name] = torch.zeros_like(act)
                        activation_counts[name] = 0
                    activation_sums[name] += act
                    activation_counts[name] += 1
            return hook_fn

        for name, module in model.named_modules():
            if hasattr(module, "in_features") and hasattr(module, "out_features"):
                hooks.append(module.register_forward_hook(_make_hook(name)))

        # run calibration data
        model.eval()
        n = 0
        with torch.no_grad():
            for batch in data_loader:
                if n >= num_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                if hasattr(x, "to"):
                    x = x.to(next(model.parameters()).device)
                model(x)
                n += 1

        # remove hooks
        for h in hooks:
            h.remove()

        # compute head importance via L1 norm of attention output weights
        for name, module in model.named_modules():
            if hasattr(module, "weight") and "attn" in name.lower():
                w = module.weight.data
                if w.dim() == 2:
                    self.head_importance[name] = w.abs().sum(dim=1)

        # neuron importance from activation magnitudes
        for name, act_sum in activation_sums.items():
            count = activation_counts[name]
            self.neuron_importance[name] = act_sum / max(count, 1)

        self._calibrated = True
        logger.info(
            f"Calibration complete: {n} batches, "
            f"{len(self.head_importance)} head layers, "
            f"{len(self.neuron_importance)} neuron layers"
        )

    def importance_width_prune(
        self,
        model: nn.Module,
        target_ratio: float | None = None,
    ) -> nn.Module:
        """Prune heads by L1 norm, neurons by activation magnitude.

        target_ratio is the fraction of heads/neurons to KEEP (e.g. 0.5 = keep half).
        Requires calibrate() to have been called first.
        """
        if not self._calibrated:
            logger.warning("Model not calibrated. Call calibrate() first.")
            return model

        ratio = target_ratio or self.config.target_ratio

        pruned_count = 0
        for name, module in model.named_modules():
            if not hasattr(module, "weight") or module.weight.dim() != 2:
                continue

            importance = None
            if name in self.head_importance:
                importance = self.head_importance[name]
            elif name in self.neuron_importance:
                importance = self.neuron_importance[name]

            if importance is None:
                continue

            if importance.dim() == 0:
                continue

            n_total = importance.shape[0]
            n_keep = max(1, int(n_total * ratio))

            if n_keep >= n_total:
                continue

            # zero out least-important rows (heads/neurons)
            _, indices = importance.sort()
            prune_indices = indices[:n_total - n_keep]

            with torch.no_grad():
                module.weight.data[prune_indices] = 0.0
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data[prune_indices] = 0.0

            pruned_count += len(prune_indices)

        logger.info(
            f"Width prune: zeroed {pruned_count} rows/neurons "
            f"(keep_ratio={ratio:.2f})"
        )
        return model

    def prune(
        self,
        model: nn.Module,
        strategy: str | None = None,
        target_ratio: float | None = None,
    ) -> nn.Module:
        """Main entry point for pruning.

        Args:
            model: The model to prune.
            strategy: "depth" for uniform layer removal,
                      "width" for importance-based head/neuron pruning.
            target_ratio: For depth, treated as keep_every_n (int cast).
                          For width, fraction of neurons to keep.
        """
        strategy = strategy or self.config.strategy

        if strategy == "depth":
            keep_n = int(target_ratio) if target_ratio and target_ratio > 1 else self.config.keep_every_n
            return self.uniform_depth_prune(model, keep_every_n=keep_n)
        elif strategy == "width":
            return self.importance_width_prune(model, target_ratio=target_ratio)
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}. Use 'depth' or 'width'.")


def _select_layers_uniform(source_n: int, target_n: int) -> list[int]:
    """Select target_n layers from source_n using uniform spacing.

    Always includes the first and last layer.
    """
    if target_n >= source_n:
        return list(range(source_n))
    if target_n == 1:
        return [0]
    if target_n == 2:
        return [0, source_n - 1]

    indices = [round(i * (source_n - 1) / (target_n - 1)) for i in range(target_n)]
    return sorted(set(indices))


def _prune_linear_width(
    src_weight: torch.Tensor,
    target_out: int,
    target_in: int,
    src_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Prune a weight matrix to target dimensions using L1 importance.

    Keeps the most important output rows and input columns.
    """
    w = src_weight
    # prune output dim (rows)
    if w.shape[0] > target_out:
        importance = w.abs().sum(dim=1)
        _, keep_idx = importance.topk(target_out)
        keep_idx = keep_idx.sort().values
        w = w[keep_idx]
        if src_bias is not None:
            src_bias = src_bias[keep_idx]
    # prune input dim (cols)
    if w.shape[1] > target_in:
        importance = w.abs().sum(dim=0)
        _, keep_idx = importance.topk(target_in)
        keep_idx = keep_idx.sort().values
        w = w[:, keep_idx]
    return w, src_bias


def _copy_to_bitlinear(
    target_module: nn.Module,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    """Copy a weight tensor into a BitLinear layer."""
    with torch.no_grad():
        target_module.weight.copy_(weight)
        if bias is not None and target_module.bias is not None:
            target_module.bias.copy_(bias)


def load_qwen_as_nova(model_name: str, nova_config: NovaConfig) -> Nova2_4B:
    """Load a Qwen model from HuggingFace and map its weights into NOVA.

    Phase 0 entry point for weight transfer. Handles:
    - Loading Qwen-2.5-3B (or 1.5B) via AutoModelForCausalLM
    - Structural pruning: drop layers to match nova_config.n_layers
    - Width pruning: reduce hidden_dim, heads, FFN to match config
    - Weight mapping: Qwen transformer -> NOVA MMMA hybrid
      - Attention positions get Qwen attention + MLP weights
      - Mamba positions get Qwen MLP weights as linear proj approximation,
        Mamba-specific params (A_log, conv, dt) initialized from scratch
    - Returns a NOVA model ready for ternarization
    """
    from transformers import AutoModelForCausalLM

    logger.info(f"Loading Qwen model: {model_name}")
    qwen = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True,
    )
    qwen.eval()

    qwen_layers = qwen.model.layers
    qwen_n_layers = len(qwen_layers)
    qwen_hidden = qwen.config.hidden_size
    qwen_n_heads = qwen.config.num_attention_heads
    qwen_n_kv_heads = getattr(qwen.config, "num_key_value_heads", qwen_n_heads)
    qwen_head_dim = qwen_hidden // qwen_n_heads
    qwen_d_ff = qwen.config.intermediate_size

    logger.info(
        f"Qwen: {qwen_n_layers} layers, hidden={qwen_hidden}, "
        f"heads={qwen_n_heads}, kv_heads={qwen_n_kv_heads}, d_ff={qwen_d_ff}"
    )

    # 1. Select which Qwen layers to keep (uniform spacing)
    keep_indices = _select_layers_uniform(qwen_n_layers, nova_config.n_layers)
    logger.info(f"Keeping {len(keep_indices)}/{qwen_n_layers} layers: {keep_indices}")

    # 2. Build empty NOVA model
    nova = Nova2_4B(nova_config)

    # 3. Map embedding weights (prune vocab rows if needed, prune hidden cols)
    with torch.no_grad():
        qwen_embed = qwen.model.embed_tokens.weight.data
        # take first nova_config.vocab_size rows, prune hidden dim
        vocab_rows = min(qwen_embed.shape[0], nova_config.vocab_size)
        embed_w = qwen_embed[:vocab_rows, :nova_config.hidden_dim]
        # pad if nova vocab > qwen vocab
        if vocab_rows < nova_config.vocab_size:
            pad = torch.zeros(
                nova_config.vocab_size - vocab_rows, nova_config.hidden_dim,
            )
            nn.init.normal_(pad, std=0.02)
            embed_w = torch.cat([embed_w, pad], dim=0)
        nova.embed_tokens.weight.copy_(embed_w)

        # LM head
        qwen_lm = qwen.lm_head.weight.data
        lm_rows = min(qwen_lm.shape[0], nova_config.vocab_size)
        lm_w = qwen_lm[:lm_rows, :nova_config.hidden_dim]
        if lm_rows < nova_config.vocab_size:
            pad = torch.zeros(
                nova_config.vocab_size - lm_rows, nova_config.hidden_dim,
            )
            nn.init.normal_(pad, std=0.02)
            lm_w = torch.cat([lm_w, pad], dim=0)
        nova.lm_head.weight.copy_(lm_w)

    # 4. Map layer weights
    hidden = nova_config.hidden_dim
    n_heads = nova_config.n_heads
    n_kv_heads = nova_config.n_kv_heads
    head_dim = nova_config.head_dim
    d_ff = nova_config.d_ff
    d_inner = nova_config.mamba_d_inner

    attn_idx = 0
    for nova_idx, (nova_layer, layer_type) in enumerate(
        zip(nova.layers, nova.layer_pattern)
    ):
        qwen_idx = keep_indices[nova_idx]
        qwen_layer = qwen_layers[qwen_idx]

        if layer_type == "A":
            _map_qwen_to_attention(
                qwen_layer, nova_layer, nova.moloras[attn_idx],
                hidden, n_heads, n_kv_heads, head_dim, d_ff, nova_config,
            )
            attn_idx += 1
        else:
            _map_qwen_to_mamba(
                qwen_layer, nova_layer, hidden, d_inner, nova_config,
            )

    # 5. Final norm
    with torch.no_grad():
        qwen_norm_w = qwen.model.norm.weight.data[:hidden]
        nova.norm.weight.copy_(qwen_norm_w)

    total = sum(p.numel() for p in nova.parameters())
    logger.info(f"NOVA model built: {total/1e6:.1f}M params")

    del qwen
    return nova


def _map_qwen_to_attention(
    qwen_layer: nn.Module,
    nova_block: nn.Module,
    molora: nn.Module,
    hidden: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    d_ff: int,
    config: NovaConfig,
):
    """Map a Qwen transformer layer to a NOVA TransformerBlock + MoLoRA."""
    attn_mod = nova_block.attention
    qwen_attn = qwen_layer.self_attn

    with torch.no_grad():
        # Q projection: prune to n_heads * head_dim output, hidden input
        q_w, _ = _prune_linear_width(
            qwen_attn.q_proj.weight.data,
            n_heads * head_dim, hidden,
        )
        _copy_to_bitlinear(attn_mod.q_proj, q_w)

        # K projection: prune to n_kv_heads * head_dim
        k_w, _ = _prune_linear_width(
            qwen_attn.k_proj.weight.data,
            n_kv_heads * head_dim, hidden,
        )
        _copy_to_bitlinear(attn_mod.k_proj, k_w)

        # V projection
        v_w, _ = _prune_linear_width(
            qwen_attn.v_proj.weight.data,
            n_kv_heads * head_dim, hidden,
        )
        _copy_to_bitlinear(attn_mod.v_proj, v_w)

        # O projection
        o_w, _ = _prune_linear_width(
            qwen_attn.o_proj.weight.data,
            hidden, n_heads * head_dim,
        )
        _copy_to_bitlinear(attn_mod.o_proj, o_w)

        # Attention RMSNorm
        attn_mod.norm.weight.copy_(
            qwen_layer.input_layernorm.weight.data[:hidden]
        )

        # FFN: Qwen uses gate_proj (up_gate), up_proj, down_proj with SiLU gating
        # NOVA FFN has up (hidden->d_ff) and down (d_ff->hidden)
        # Approximate: use gate_proj as the up projection
        ffn = nova_block.ffn
        gate_w, _ = _prune_linear_width(
            qwen_layer.mlp.gate_proj.weight.data, d_ff, hidden,
        )
        _copy_to_bitlinear(ffn.up, gate_w)

        down_w, _ = _prune_linear_width(
            qwen_layer.mlp.down_proj.weight.data, hidden, d_ff,
        )
        _copy_to_bitlinear(ffn.down, down_w)

        # FFN RMSNorm
        ffn.norm.weight.copy_(
            qwen_layer.post_attention_layernorm.weight.data[:hidden]
        )


def _map_qwen_to_mamba(
    qwen_layer: nn.Module,
    nova_block: nn.Module,
    hidden: int,
    d_inner: int,
    config: NovaConfig,
):
    """Map a Qwen transformer layer to a NOVA MambaBlock.

    Mamba-specific params (A_log, conv, dt) are initialized from scratch.
    Linear projections approximated from Qwen MLP weights.
    """
    with torch.no_grad():
        # RMSNorm from Qwen input layernorm
        nova_block.norm.weight.copy_(
            qwen_layer.input_layernorm.weight.data[:hidden]
        )

        # in_proj: hidden -> d_inner*2
        # Use gate_proj (hidden->d_ff) and up_proj (hidden->d_ff) as approximation
        # Prune both to d_inner output, hidden input, then concatenate
        gate_w, _ = _prune_linear_width(
            qwen_layer.mlp.gate_proj.weight.data, d_inner, hidden,
        )
        up_w, _ = _prune_linear_width(
            qwen_layer.mlp.up_proj.weight.data, d_inner, hidden,
        )
        in_proj_w = torch.cat([gate_w, up_w], dim=0)  # (d_inner*2, hidden)
        _copy_to_bitlinear(nova_block.in_proj, in_proj_w)

        # out_proj: d_inner -> hidden
        # Use down_proj (d_ff->hidden), prune to (hidden, d_inner)
        down_w, _ = _prune_linear_width(
            qwen_layer.mlp.down_proj.weight.data, hidden, d_inner,
        )
        _copy_to_bitlinear(nova_block.out_proj, down_w)

        # Mamba-specific params: initialize from scratch
        # A_log: already initialized in MambaBlock.__init__
        # conv_weight: small random init (already done)
        # dt_proj, B_proj, C_proj: keep default random init
        # D: already ones

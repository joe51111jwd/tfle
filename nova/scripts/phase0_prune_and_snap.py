#!/usr/bin/env python3
"""
Phase 0: Download Qwen-2.5-3B -> Prune to 1B -> Snap to ternary
=============================================================
Cost: $0 (runs on local machine or minimal instance)
Time: ~1 hour
Output: nova_1b_ternary_init.pt checkpoint

Usage:
  python nova/scripts/phase0_prune_and_snap.py \
    --source Qwen/Qwen2.5-3B \
    --output checkpoints/nova_1b_ternary_init.pt \
    --calibration_data wikitext
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from nova.model.config import NovaConfig, NOVA_1B_QWEN
from nova.model.nova_2_4b import Nova2_4B
from nova.model.mamba_block import MambaBlock
from nova.model.attention import GroupedQueryAttention
from nova.model.bitlinear import BitLinear, RMSNorm
from nova.training.ternarize import Ternarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QWEN_N_LAYERS = 36
QWEN_HIDDEN = 2048
QWEN_N_HEADS = 16
QWEN_N_KV_HEADS = 2
QWEN_D_FF = 5632  # intermediate_size
QWEN_HEAD_DIM = QWEN_HIDDEN // QWEN_N_HEADS  # 128
QWEN_VOCAB = 151646

PPL_GATE = 5_000_000  # quality gate: PPL must be < 5M after snap


# ===================================================================
# Step 1: Load source model
# ===================================================================
def load_qwen(model_name: str, device: str) -> tuple:
    """Load Qwen-2.5-3B from HuggingFace and return (model, tokenizer)."""
    logger.info(f"[1/8] Loading {model_name} from HuggingFace...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error(
            "transformers not installed. Run: pip install transformers accelerate"
        )
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"  Loaded {model_name}: {total_params / 1e9:.2f}B params on CPU"
        )
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)


# ===================================================================
# Step 2: Load calibration data
# ===================================================================
def load_calibration_data(
    source: str,
    tokenizer,
    n_samples: int = 256,
    seq_len: int = 512,
) -> list[torch.Tensor]:
    """Load calibration sequences from WikiText-103 or FineWeb."""
    logger.info(f"[2/8] Loading calibration data ({source})...")
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Run: pip install datasets")
        sys.exit(1)

    if source == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")
        text_key = "text"
    elif source == "fineweb":
        ds = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        text_key = "text"
    else:
        logger.error(f"Unknown calibration source: {source}. Use 'wikitext' or 'fineweb'")
        sys.exit(1)

    # Concatenate text and tokenize
    texts = []
    char_budget = n_samples * seq_len * 4  # ~4 chars per token estimate
    total_chars = 0
    for row in ds:
        t = row[text_key]
        if not t or len(t.strip()) < 50:
            continue
        texts.append(t)
        total_chars += len(t)
        if total_chars >= char_budget:
            break

    full_text = "\n".join(texts)
    tokens = tokenizer.encode(full_text, return_tensors="pt")[0]

    # Chunk into sequences
    sequences = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        sequences.append(tokens[i : i + seq_len].unsqueeze(0))
        if len(sequences) >= n_samples:
            break

    logger.info(f"  Prepared {len(sequences)} calibration sequences of length {seq_len}")
    return sequences


# ===================================================================
# Step 3: Run calibration for importance scores
# ===================================================================
def compute_layer_importance(
    model,
    calibration_seqs: list[torch.Tensor],
    n_batches: int = 32,
) -> dict[str, torch.Tensor]:
    """Forward pass through Qwen to compute per-layer importance scores.

    Importance = mean L2 norm of each layer's output delta (residual contribution).
    """
    logger.info("[3/8] Running calibration to compute importance scores...")

    layer_importance = {}
    hooks = []
    layer_inputs = {}
    layer_outputs = {}

    def _make_input_hook(name: str):
        def hook_fn(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                x = inp[0]
            else:
                x = inp
            if isinstance(x, torch.Tensor):
                layer_inputs[name] = x.detach()
        return hook_fn

    def _make_output_hook(name: str):
        def hook_fn(module, inp, out):
            if isinstance(out, tuple) and len(out) > 0:
                x = out[0]
            else:
                x = out
            if isinstance(x, torch.Tensor):
                layer_outputs[name] = x.detach()
        return hook_fn

    # Register hooks on decoder layers
    decoder_layers = None
    for name, module in model.named_modules():
        if hasattr(module, "__len__") and hasattr(module, "__getitem__"):
            # Find the main layer list (model.model.layers for Qwen)
            parent_name = name
            if len(module) == QWEN_N_LAYERS:
                decoder_layers = module
                break

    if decoder_layers is None:
        # Fallback: try model.model.layers
        try:
            decoder_layers = model.model.layers
        except AttributeError:
            logger.error("Cannot find decoder layers in the model")
            sys.exit(1)

    for i, layer in enumerate(decoder_layers):
        name = f"layer_{i}"
        hooks.append(layer.register_forward_hook(_make_output_hook(name)))

    # Also hook attention and MLP submodules for head/neuron importance
    head_importance = {}
    neuron_importance = {}

    for i, layer in enumerate(decoder_layers):
        # Attention output norm
        attn_name = f"attn_{i}"
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            if hasattr(attn, "o_proj"):
                w = attn.o_proj.weight.data
                # Per-head importance: reshape [hidden, hidden] -> [n_heads, head_dim, hidden]
                # and compute L1 norm per head
                try:
                    head_w = w.reshape(QWEN_HIDDEN, QWEN_N_HEADS, QWEN_HEAD_DIM)
                    head_importance[i] = head_w.abs().sum(dim=(0, 2))  # [n_heads]
                except RuntimeError:
                    head_importance[i] = torch.ones(QWEN_N_HEADS)

        # MLP neuron importance from gate/up weights
        mlp_name = f"mlp_{i}"
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            if hasattr(mlp, "gate_proj"):
                w = mlp.gate_proj.weight.data
                neuron_importance[i] = w.abs().mean(dim=1)  # [d_ff]

    # Forward pass for layer output importance
    importance_accum = {}
    n = 0
    model.eval()
    with torch.no_grad():
        for seq in calibration_seqs[:n_batches]:
            layer_inputs.clear()
            layer_outputs.clear()
            try:
                model(seq)
            except Exception:
                continue
            n += 1

            # Compute residual contribution norm per layer
            prev_out = None
            for i in range(len(decoder_layers)):
                name = f"layer_{i}"
                if name in layer_outputs:
                    out = layer_outputs[name]
                    if prev_out is not None and out.shape == prev_out.shape:
                        delta = (out - prev_out).float()
                        importance = delta.norm(dim=-1).mean().item()
                    else:
                        importance = out.float().norm(dim=-1).mean().item()
                    if name not in importance_accum:
                        importance_accum[name] = 0.0
                    importance_accum[name] += importance
                    prev_out = out

    for h in hooks:
        h.remove()

    # Average
    for k in importance_accum:
        importance_accum[k] /= max(n, 1)

    # Convert to tensor
    layer_scores = torch.tensor(
        [importance_accum.get(f"layer_{i}", 0.0) for i in range(len(decoder_layers))]
    )

    logger.info(
        f"  Calibrated on {n} batches. Layer importance range: "
        f"[{layer_scores.min():.4f}, {layer_scores.max():.4f}]"
    )

    return {
        "layer_scores": layer_scores,
        "head_importance": head_importance,
        "neuron_importance": neuron_importance,
    }


# ===================================================================
# Step 4: Prune Qwen layers and width to ~1B target
# ===================================================================
def select_layers_to_keep(
    layer_scores: torch.Tensor,
    target_layers: int,
    total_layers: int,
) -> list[int]:
    """Select which Qwen layers to keep based on importance.

    Always keeps first and last layer. Selects remaining by importance.
    """
    if target_layers >= total_layers:
        return list(range(total_layers))

    # Always keep first and last
    must_keep = {0, total_layers - 1}
    remaining_budget = target_layers - len(must_keep)

    # Score the middle layers
    middle_indices = list(range(1, total_layers - 1))
    middle_scores = layer_scores[1:-1]

    # Select top-scoring middle layers
    _, sorted_idx = middle_scores.sort(descending=True)
    selected = sorted_idx[:remaining_budget].tolist()
    selected_indices = sorted([middle_indices[i] for i in selected])

    keep = sorted(must_keep | set(selected_indices))
    return keep


def prune_linear_width(
    weight: torch.Tensor,
    target_out: int,
    target_in: int,
    neuron_scores: torch.Tensor | None = None,
    dim: str = "both",
) -> torch.Tensor:
    """Prune a weight matrix to target dimensions.

    If neuron_scores provided, keeps highest-scoring neurons.
    Otherwise uses L1 norm of rows/columns.
    """
    out_dim, in_dim = weight.shape

    # Prune output dimension (rows)
    if out_dim > target_out:
        if neuron_scores is not None and len(neuron_scores) == out_dim:
            _, keep_idx = neuron_scores.sort(descending=True)
            keep_idx = keep_idx[:target_out].sort().values
        else:
            row_importance = weight.abs().sum(dim=1)
            _, keep_idx = row_importance.sort(descending=True)
            keep_idx = keep_idx[:target_out].sort().values
        weight = weight[keep_idx]

    # Prune input dimension (columns)
    if in_dim > target_in:
        col_importance = weight.abs().sum(dim=0)
        _, keep_idx = col_importance.sort(descending=True)
        keep_idx = keep_idx[:target_in].sort().values
        weight = weight[:, keep_idx]

    # If we need to pad (unlikely but handle vocab mismatch)
    if weight.shape[0] < target_out:
        pad = torch.zeros(target_out - weight.shape[0], weight.shape[1], device=weight.device)
        weight = torch.cat([weight, pad], dim=0)
    if weight.shape[1] < target_in:
        pad = torch.zeros(weight.shape[0], target_in - weight.shape[1], device=weight.device)
        weight = torch.cat([weight, pad], dim=1)

    return weight


def extract_pruned_weights(
    qwen_model,
    keep_layers: list[int],
    importance: dict[str, torch.Tensor],
    config: NovaConfig,
) -> dict[str, torch.Tensor]:
    """Extract and prune Qwen weights to match NOVA_1B dimensions.

    Returns a flat dict of tensors ready to map into Nova2_4B.
    """
    logger.info("[4/8] Pruning Qwen weights to NOVA_1B dimensions...")

    head_importance = importance.get("head_importance", {})
    neuron_importance = importance.get("neuron_importance", {})

    try:
        decoder_layers = qwen_model.model.layers
        embed_weight = qwen_model.model.embed_tokens.weight.data.clone()
        lm_head_weight = qwen_model.lm_head.weight.data.clone()
        final_norm_weight = qwen_model.model.norm.weight.data.clone()
    except AttributeError as e:
        logger.error(f"Unexpected model structure: {e}")
        sys.exit(1)

    pruned = {}

    # --- Embeddings ---
    # Qwen vocab = 151646, NOVA_1B_QWEN vocab = 151650 (+4 special tokens)
    # Hidden: Qwen=2048, NOVA_1B=1728 -- prune hidden dim
    pruned["embed_tokens"] = prune_linear_width(
        embed_weight, config.vocab_size, config.hidden_dim,
    )

    # --- LM head ---
    pruned["lm_head"] = prune_linear_width(
        lm_head_weight, config.vocab_size, config.hidden_dim,
    )

    # --- Final norm ---
    if len(final_norm_weight) > config.hidden_dim:
        norm_importance = final_norm_weight.abs()
        _, keep_idx = norm_importance.sort(descending=True)
        keep_idx = keep_idx[: config.hidden_dim].sort().values
        pruned["final_norm"] = final_norm_weight[keep_idx]
    elif len(final_norm_weight) < config.hidden_dim:
        pruned["final_norm"] = F.pad(
            final_norm_weight, (0, config.hidden_dim - len(final_norm_weight)), value=1.0
        )
    else:
        pruned["final_norm"] = final_norm_weight.clone()

    # --- Per-layer weights ---
    nova_layer_pattern = config.layer_pattern  # MMMA MMMA ...
    n_nova_layers = config.n_layers

    for nova_idx, qwen_idx in enumerate(keep_layers):
        layer = decoder_layers[qwen_idx]
        layer_type = nova_layer_pattern[nova_idx]

        # Pre-attention norm (input_layernorm)
        if hasattr(layer, "input_layernorm"):
            ln_w = layer.input_layernorm.weight.data
            if len(ln_w) > config.hidden_dim:
                ln_importance = ln_w.abs()
                _, keep = ln_importance.sort(descending=True)
                keep = keep[: config.hidden_dim].sort().values
                pruned[f"layer_{nova_idx}_attn_norm"] = ln_w[keep]
            else:
                pruned[f"layer_{nova_idx}_attn_norm"] = ln_w[: config.hidden_dim].clone()

        # Post-attention norm
        if hasattr(layer, "post_attention_layernorm"):
            ln_w = layer.post_attention_layernorm.weight.data
            if len(ln_w) > config.hidden_dim:
                ln_importance = ln_w.abs()
                _, keep = ln_importance.sort(descending=True)
                keep = keep[: config.hidden_dim].sort().values
                pruned[f"layer_{nova_idx}_ffn_norm"] = ln_w[keep]
            else:
                pruned[f"layer_{nova_idx}_ffn_norm"] = ln_w[: config.hidden_dim].clone()

        attn = layer.self_attn
        mlp = layer.mlp

        if layer_type == "A":
            # ---- Attention layer: copy Q/K/V/O projections ----
            nova_q_dim = config.n_heads * config.head_dim
            nova_k_dim = config.n_kv_heads * config.head_dim
            nova_v_dim = config.n_kv_heads * config.head_dim

            # Q projection: [n_heads*head_dim, hidden] -> [nova_q_dim, nova_hidden]
            q_w = attn.q_proj.weight.data
            pruned[f"layer_{nova_idx}_q_proj"] = prune_linear_width(
                q_w, nova_q_dim, config.hidden_dim,
            )

            # K projection
            k_w = attn.k_proj.weight.data
            pruned[f"layer_{nova_idx}_k_proj"] = prune_linear_width(
                k_w, nova_k_dim, config.hidden_dim,
            )

            # V projection
            v_w = attn.v_proj.weight.data
            pruned[f"layer_{nova_idx}_v_proj"] = prune_linear_width(
                v_w, nova_v_dim, config.hidden_dim,
            )

            # O projection: [hidden, n_heads*head_dim]
            o_w = attn.o_proj.weight.data
            pruned[f"layer_{nova_idx}_o_proj"] = prune_linear_width(
                o_w, config.hidden_dim, nova_q_dim,
            )

            # FFN for attention blocks
            gate_w = mlp.gate_proj.weight.data
            up_w = mlp.up_proj.weight.data
            down_w = mlp.down_proj.weight.data

            n_scores = neuron_importance.get(qwen_idx)

            # FFN up: [d_ff, hidden] -> [nova_d_ff, nova_hidden]
            pruned[f"layer_{nova_idx}_ffn_up"] = prune_linear_width(
                up_w, config.d_ff, config.hidden_dim, n_scores,
            )
            # FFN down: [hidden, d_ff] -> [nova_hidden, nova_d_ff]
            pruned[f"layer_{nova_idx}_ffn_down"] = prune_linear_width(
                down_w, config.hidden_dim, config.d_ff,
            )

        else:
            # ---- Mamba layer: adapt MLP weights into Mamba projections ----
            d_inner = config.mamba_d_inner  # hidden * expand = 1728 * 2 = 3456

            # in_proj: [hidden, d_inner*2] -- use gate_proj and up_proj
            gate_w = mlp.gate_proj.weight.data
            up_w = mlp.up_proj.weight.data

            # gate_proj: [d_ff, hidden], up_proj: [d_ff, hidden]
            # Prune both to [d_inner, hidden] and stack for in_proj [d_inner*2, hidden]
            gate_pruned = prune_linear_width(gate_w, d_inner, config.hidden_dim)
            up_pruned = prune_linear_width(up_w, d_inner, config.hidden_dim)
            pruned[f"layer_{nova_idx}_in_proj"] = torch.cat(
                [gate_pruned, up_pruned], dim=0
            )

            # out_proj: [hidden, d_inner] -- use down_proj
            down_w = mlp.down_proj.weight.data
            pruned[f"layer_{nova_idx}_out_proj"] = prune_linear_width(
                down_w, config.hidden_dim, d_inner,
            )

    total_pruned_params = sum(t.numel() for t in pruned.values())
    logger.info(f"  Extracted {len(pruned)} weight tensors, {total_pruned_params / 1e6:.1f}M params")
    return pruned


# ===================================================================
# Step 5 & 6: Build Nova2_4B and map weights
# ===================================================================
def build_and_map_nova(
    pruned_weights: dict[str, torch.Tensor],
    config: NovaConfig,
) -> Nova2_4B:
    """Build a Nova2_4B model with NOVA_1B config and load pruned weights."""
    logger.info("[5/8] Building Nova2_4B with NOVA_1B config...")
    model = Nova2_4B(config)
    logger.info(f"  Model created: {model.count_parameters()['total_M']:.1f}M params")

    logger.info("[6/8] Mapping pruned Qwen weights into NOVA model...")
    mapped = 0
    failed = 0

    def _safe_copy(target: torch.Tensor, source: torch.Tensor, name: str) -> bool:
        nonlocal mapped, failed
        if target.shape != source.shape:
            logger.warning(
                f"  Shape mismatch for {name}: "
                f"target={list(target.shape)}, source={list(source.shape)}"
            )
            # Try to fit what we can
            slices_t = tuple(slice(0, min(t, s)) for t, s in zip(target.shape, source.shape))
            slices_s = tuple(slice(0, min(t, s)) for t, s in zip(target.shape, source.shape))
            try:
                target[slices_t].copy_(source[slices_s])
                mapped += 1
                return True
            except Exception as e:
                logger.warning(f"  Failed to partially copy {name}: {e}")
                failed += 1
                return False
        target.copy_(source)
        mapped += 1
        return True

    with torch.no_grad():
        # Embeddings
        if "embed_tokens" in pruned_weights:
            _safe_copy(
                model.embed_tokens.weight.data,
                pruned_weights["embed_tokens"],
                "embed_tokens",
            )

        # LM head
        if "lm_head" in pruned_weights:
            _safe_copy(
                model.lm_head.weight.data,
                pruned_weights["lm_head"],
                "lm_head",
            )

        # Final norm
        if "final_norm" in pruned_weights:
            _safe_copy(
                model.norm.weight.data,
                pruned_weights["final_norm"],
                "final_norm",
            )

        # Per-layer weights
        attn_idx = 0
        for i, (layer, lt) in enumerate(zip(model.layers, model.layer_pattern)):
            if lt == "A":
                # Attention layers
                attn_block = layer  # TransformerBlock
                attn = attn_block.attention

                # Attention norm
                key = f"layer_{i}_attn_norm"
                if key in pruned_weights:
                    _safe_copy(attn.norm.weight.data, pruned_weights[key], key)

                # Q/K/V/O projections (BitLinear has .weight attribute)
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    key = f"layer_{i}_{proj_name}"
                    if key in pruned_weights:
                        proj = getattr(attn, proj_name)
                        _safe_copy(proj.weight.data, pruned_weights[key], key)

                # FFN weights
                ffn = attn_block.ffn
                key = f"layer_{i}_ffn_norm"
                if key in pruned_weights:
                    _safe_copy(ffn.norm.weight.data, pruned_weights[key], key)

                key = f"layer_{i}_ffn_up"
                if key in pruned_weights:
                    _safe_copy(ffn.up.weight.data, pruned_weights[key], key)

                key = f"layer_{i}_ffn_down"
                if key in pruned_weights:
                    _safe_copy(ffn.down.weight.data, pruned_weights[key], key)

                attn_idx += 1

            else:
                # Mamba layers
                mamba = layer  # MambaBlock

                # Norm
                key = f"layer_{i}_attn_norm"
                if key in pruned_weights:
                    _safe_copy(mamba.norm.weight.data, pruned_weights[key], key)

                # in_proj: from gate+up stacked
                key = f"layer_{i}_in_proj"
                if key in pruned_weights:
                    _safe_copy(mamba.in_proj.weight.data, pruned_weights[key], key)

                # out_proj: from down
                key = f"layer_{i}_out_proj"
                if key in pruned_weights:
                    _safe_copy(mamba.out_proj.weight.data, pruned_weights[key], key)

                # Mamba-specific params: standard initialization
                _init_mamba_specific(mamba, config)

    logger.info(f"  Mapped {mapped} tensors, {failed} partial/failed")
    return model


def _init_mamba_specific(mamba: MambaBlock, config: NovaConfig):
    """Initialize Mamba-specific parameters with standard values.

    These have no Qwen equivalent, so we use the same init as MambaBlock.__init__.
    """
    d_inner = config.mamba_d_inner
    d_state = config.mamba_d_state
    d_conv = config.mamba_d_conv

    with torch.no_grad():
        # A_log: log of diagonal SSM matrix
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_inner, -1)
        mamba.A_log.data.copy_(torch.log(A))

        # D: skip connection
        mamba.D.data.fill_(1.0)

        # Conv weights: small random
        mamba.conv_weight.data.normal_(0, 0.1)
        mamba.conv_bias.data.zero_()

        # dt_proj, B_proj, C_proj: Xavier uniform (BitLinear default)
        for proj in [mamba.dt_proj, mamba.B_proj, mamba.C_proj]:
            if hasattr(proj, "weight"):
                scale = 1.0 / math.sqrt(proj.weight.shape[1])
                proj.weight.data.normal_(0, scale)

        # dt_proj_down: regular Linear
        scale = 1.0 / math.sqrt(mamba.dt_proj_down.weight.shape[1])
        mamba.dt_proj_down.weight.data.normal_(0, scale)
        if mamba.dt_proj_down.bias is not None:
            mamba.dt_proj_down.bias.data.zero_()


# ===================================================================
# Step 7: Snap to ternary
# ===================================================================
def snap_to_ternary(model: Nova2_4B) -> Nova2_4B:
    """Apply ternary quantization, preserving float shadow weights."""
    logger.info("[7/8] Snapping weights to ternary...")
    ternarizer = Ternarizer()
    model = ternarizer.ternarize_model(model)

    stats = ternarizer.get_ternary_stats(model)
    logger.info(
        f"  Ternarized: {stats['n_ternary_layers']} layers, "
        f"zero fraction: {stats['avg_zero_fraction']:.3f}"
    )
    return model


# ===================================================================
# Step 8: Verify
# ===================================================================
@torch.no_grad()
def verify_model(
    model: Nova2_4B,
    calibration_seqs: list[torch.Tensor],
    config: NovaConfig,
    n_eval: int = 16,
) -> float:
    """Run forward pass on calibration data and compute perplexity."""
    logger.info("[8/8] Verifying model...")

    # Parameter counts
    total = sum(p.numel() for p in model.parameters())
    ternary = 0
    float_params = 0
    for name, module in model.named_modules():
        if not hasattr(module, "weight") or module.weight.dim() != 2:
            continue
        if hasattr(module, "_shadow_weight"):
            ternary += module.weight.numel()
        elif isinstance(module, (nn.Embedding, nn.Linear)):
            float_params += module.weight.numel()

    # Count norms and other float params
    for name, p in model.named_parameters():
        if "norm" in name.lower() or "embed" in name.lower() or "lm_head" in name.lower():
            float_params += p.numel()

    print("\n" + "=" * 60)
    print("NOVA 1B Ternary Init -- Parameter Report")
    print("=" * 60)
    print(f"  Total params:   {total:>14,} ({total / 1e9:.3f}B)")
    print(f"  Ternary params: {ternary:>14,}")
    print(f"  Float params:   {float_params:>14,}")
    print(f"  Ternary ratio:  {ternary / max(total, 1) * 100:>13.1f}%")
    print("=" * 60)

    # Compute perplexity
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    for seq in calibration_seqs[:n_eval]:
        input_ids = seq[:, :-1].to(device)
        targets = seq[:, 1:].to(device)

        try:
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM during verification, reducing batch count")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                break
            logger.warning(f"Forward pass error: {e}")
            continue

    if total_tokens == 0:
        logger.error("No tokens processed during verification!")
        return float("inf")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 30))  # cap to avoid overflow

    print(f"\n  Calibration PPL: {ppl:,.2f}")
    print(f"  Avg loss:        {avg_loss:.4f}")

    if ppl < PPL_GATE:
        print(f"  Quality gate:    PASS (PPL {ppl:,.0f} < {PPL_GATE:,})")
    else:
        print(f"  Quality gate:    FAIL (PPL {ppl:,.0f} >= {PPL_GATE:,})")

    print("=" * 60 + "\n")
    return ppl


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: Prune Qwen-2.5-3B to NOVA 1B ternary"
    )
    parser.add_argument(
        "--source",
        default="Qwen/Qwen2.5-3B",
        help="HuggingFace model name (default: Qwen/Qwen2.5-3B)",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/nova_1b_ternary_init.pt",
        help="Output checkpoint path",
    )
    parser.add_argument(
        "--calibration_data",
        default="wikitext",
        choices=["wikitext", "fineweb"],
        help="Calibration dataset (default: wikitext)",
    )
    parser.add_argument(
        "--n_calibration",
        type=int,
        default=256,
        help="Number of calibration sequences (default: 256)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Calibration sequence length (default: 512)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for pruning (default: cpu)",
    )
    args = parser.parse_args()

    start_time = time.time()
    print("\n" + "=" * 60)
    print("Phase 0: Qwen-2.5-3B -> NOVA 1B Ternary Init")
    print("=" * 60 + "\n")

    # Step 1: Load Qwen
    qwen_model, tokenizer = load_qwen(args.source, args.device)

    # Step 2: Load calibration data
    calibration_seqs = load_calibration_data(
        args.calibration_data, tokenizer, args.n_calibration, args.seq_len,
    )

    # Step 3: Compute importance scores
    importance = compute_layer_importance(qwen_model, calibration_seqs)

    # Step 4: Select layers and prune
    config = NOVA_1B_QWEN
    keep_layers = select_layers_to_keep(
        importance["layer_scores"],
        target_layers=config.n_layers,
        total_layers=QWEN_N_LAYERS,
    )
    logger.info(f"  Keeping Qwen layers: {keep_layers}")

    pruned_weights = extract_pruned_weights(qwen_model, keep_layers, importance, config)

    # Free Qwen to save memory
    del qwen_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc
    gc.collect()

    # Step 5 & 6: Build NOVA model and map weights
    nova_model = build_and_map_nova(pruned_weights, config)
    del pruned_weights
    gc.collect()

    # Step 7: Snap to ternary
    nova_model = snap_to_ternary(nova_model)

    # Step 8: Verify
    ppl = verify_model(nova_model, calibration_seqs, config)

    # Save checkpoint
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": nova_model.state_dict(),
        "config": {
            "n_layers": config.n_layers,
            "hidden_dim": config.hidden_dim,
            "vocab_size": config.vocab_size,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "d_ff": config.d_ff,
            "mamba_d_state": config.mamba_d_state,
            "mamba_d_conv": config.mamba_d_conv,
            "mamba_expand": config.mamba_expand,
            "max_seq_len": config.max_seq_len,
        },
        "source_model": args.source,
        "kept_qwen_layers": keep_layers,
        "calibration_ppl": ppl,
        "phase": 0,
    }

    torch.save(checkpoint, str(output_path))
    logger.info(f"  Saved checkpoint to {output_path}")

    elapsed = time.time() - start_time
    print(f"\nPhase 0 complete in {elapsed / 60:.1f} minutes")
    print(f"Checkpoint: {output_path}")
    print(f"PPL: {ppl:,.2f}")

    if ppl >= PPL_GATE:
        logger.warning(
            f"PPL {ppl:,.0f} exceeds gate {PPL_GATE:,}. "
            "Consider adjusting pruning strategy."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

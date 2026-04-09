"""Liger-kernel monkey-patching for NOVA.

Liger (https://github.com/linkedin/Liger-Kernel) provides fused Triton kernels
that can cut step time 15-30% on H100/A100 for the parts of a transformer that
aren't already ternary. We patch the parts of NOVA that are NOT wrapped in
BitLinear — if you replace a BitLinear-internal RMSNorm with a dense Liger
RMSNorm you lose the whole point of the ternary model.

Safe targets:
    - Top-level RMSNorm on Nova2_4B (`model.norm`).
    - Any RoPE implementation — ours lives inline in attention.py; we leave
      it alone unless the attention layer exposes a nn.Module that matches
      Liger's expected interface.
    - SwiGLU / GeGLU MLPs — NOVA uses ReLU-squared, so nothing to patch here.

Unsafe targets (do NOT touch):
    - RMSNorm fields inside BitLinear / MambaBlock / GroupedQueryAttention
      (they precede the ternary quantization and are handled by the fused
      BitLinear kernel in production).
"""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm  # type: ignore
    _HAS_LIGER_RMS = True
except Exception:
    LigerRMSNorm = None  # type: ignore
    _HAS_LIGER_RMS = False

try:
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP  # type: ignore
    _HAS_LIGER_SWIGLU = True
except Exception:
    LigerSwiGLUMLP = None  # type: ignore
    _HAS_LIGER_SWIGLU = False

try:
    from liger_kernel.transformers.rope import LigerRotaryPositionalEmbedding as LigerRoPE  # type: ignore
    _HAS_LIGER_ROPE = True
except Exception:
    LigerRoPE = None  # type: ignore
    _HAS_LIGER_ROPE = False

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (  # type: ignore
        LigerFusedLinearCrossEntropyLoss,
    )
    _HAS_LIGER_FLCE = True
except Exception:
    LigerFusedLinearCrossEntropyLoss = None  # type: ignore
    _HAS_LIGER_FLCE = False


HAS_LIGER = _HAS_LIGER_RMS or _HAS_LIGER_SWIGLU or _HAS_LIGER_ROPE or _HAS_LIGER_FLCE


def _is_bitlinear_owned_norm(parent: nn.Module) -> bool:
    """Return True if the parent module is a BitLinear (so its .norm is ternary-adjacent)."""
    cls_name = type(parent).__name__
    return cls_name == "BitLinear"


def _is_component_norm(parent_path: str) -> bool:
    """Return True for norms inside attention / mamba / ffn that precede BitLinear."""
    ternary_parents = ("attention", "ffn", "layers", "mamba", "in_proj", "out_proj")
    return any(p in parent_path for p in ternary_parents)


def _iter_named_modules_with_parents(
    model: nn.Module,
) -> list[tuple[str, str, str, nn.Module, nn.Module]]:
    """Yield (full_name, parent_name, attr_name, parent, child) for every submodule."""
    result: list[tuple[str, str, str, nn.Module, nn.Module]] = []
    for name, module in list(model.named_modules()):
        if name == "":
            continue
        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
        else:
            parent_name, attr_name = "", name
        parent = model
        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)
        result.append((name, parent_name, attr_name, parent, module))
    return result


def _replace_rmsnorm(
    model: nn.Module,
    patched: list[str],
) -> None:
    if not _HAS_LIGER_RMS:
        return
    from nova.model.bitlinear import RMSNorm as NovaRMSNorm

    for full_name, parent_name, attr_name, parent, module in _iter_named_modules_with_parents(model):
        if not isinstance(module, NovaRMSNorm):
            continue
        if _is_bitlinear_owned_norm(parent):
            continue
        if _is_component_norm(parent_name):
            continue

        dim = module.weight.shape[0]
        eps = getattr(module, "eps", 1e-6)
        try:
            liger = LigerRMSNorm(dim, eps=eps)
        except TypeError:
            liger = LigerRMSNorm(dim)
        with torch.no_grad():
            liger.weight.copy_(module.weight)
        liger = liger.to(module.weight.device, dtype=module.weight.dtype)
        setattr(parent, attr_name, liger)
        patched.append(f"RMSNorm -> LigerRMSNorm: {full_name}")


def _replace_swiglu(model: nn.Module, patched: list[str]) -> None:
    if not _HAS_LIGER_SWIGLU:
        return
    for full_name, _parent_name, attr_name, parent, module in _iter_named_modules_with_parents(model):
        cls_name = type(module).__name__
        if cls_name not in ("SwiGLU", "SwiGLUMLP", "GeGLU", "GeGLUMLP"):
            continue
        if not hasattr(module, "gate_proj") or not hasattr(module, "up_proj"):
            continue
        try:
            liger = LigerSwiGLUMLP(module.config) if hasattr(module, "config") else None
            if liger is None:
                continue
            setattr(parent, attr_name, liger)
            patched.append(f"{cls_name} -> LigerSwiGLUMLP: {full_name}")
        except Exception as e:
            logger.debug(f"Skipping SwiGLU swap at {full_name}: {e}")


def _replace_rope(model: nn.Module, patched: list[str]) -> None:
    if not _HAS_LIGER_ROPE:
        return
    for full_name, _parent_name, attr_name, parent, module in _iter_named_modules_with_parents(model):
        cls_name = type(module).__name__
        if cls_name not in ("RotaryEmbedding", "RoPE", "LlamaRotaryEmbedding"):
            continue
        try:
            setattr(parent, attr_name, LigerRoPE())
            patched.append(f"{cls_name} -> LigerRoPE: {full_name}")
        except Exception as e:
            logger.debug(f"Skipping RoPE swap at {full_name}: {e}")


def patch_with_liger(model: nn.Module) -> tuple[nn.Module, list[str]]:
    """Monkey-patch a NOVA model to use Liger fused kernels where safe.

    Replaces (if liger-kernel is installed):
        - Top-level RMSNorm (outside BitLinear / attention / mamba) -> LigerRMSNorm
        - SwiGLU / GeGLU MLP -> LigerSwiGLUMLP (NOVA doesn't have one today)
        - RoPE module -> LigerRoPE (NOVA's RoPE is inline and will be skipped)

    Does NOT patch RMSNorms that live inside BitLinear — those are paired with
    the ternary quantization and get their own fused kernel upstream.

    Returns (model, patched_descriptions). `patched_descriptions` is a
    (possibly empty) list of human-readable strings describing every swap.
    If liger-kernel is not installed, the function logs a warning and
    returns the model unchanged with an empty list.
    """
    if not HAS_LIGER:
        logger.warning(
            "liger-kernel not installed; skipping patch_with_liger. "
            "Install with `pip install liger-kernel` for ~20% step-time speedup."
        )
        return model, []

    patched: list[str] = []
    _replace_rmsnorm(model, patched)
    _replace_swiglu(model, patched)
    _replace_rope(model, patched)

    if patched:
        logger.info(f"Liger patched {len(patched)} modules:")
        for desc in patched:
            logger.info(f"  {desc}")
    else:
        logger.info(
            "Liger available but no safe targets found in this model "
            "(BitLinear-internal norms are skipped by design)."
        )
    return model, patched


def get_liger_loss() -> Callable[..., torch.Tensor]:
    """Return a fused-linear-cross-entropy loss if Liger is installed, else F.cross_entropy.

    The Liger fused loss takes (lm_head_weight, hidden_states, labels) and
    skips the materialization of the full [batch*seq, vocab] logit tensor —
    on a 50K-vocab 2.4B model that is a 4-6GB activation savings.

    The fallback mirrors the signature so calling code can stay the same:

        loss_fn = get_liger_loss()
        loss = loss_fn(lm_head.weight, hidden_states, labels)
    """
    if _HAS_LIGER_FLCE:
        liger_loss_module = LigerFusedLinearCrossEntropyLoss()

        def liger_loss_fn(
            lm_head_weight: torch.Tensor,
            hidden_states: torch.Tensor,
            labels: torch.Tensor,
            ignore_index: int = -100,
        ) -> torch.Tensor:
            flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
            flat_labels = labels.reshape(-1)
            return liger_loss_module(lm_head_weight, flat_hidden, flat_labels)

        return liger_loss_fn

    def fallback_loss_fn(
        lm_head_weight: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        logits = hidden_states @ lm_head_weight.T
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=ignore_index,
        )

    return fallback_loss_fn

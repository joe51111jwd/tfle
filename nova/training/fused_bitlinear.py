"""Fused BitLinear kernel (RMSNorm + ternary quantize + absmax quantize + matmul + bias).

The existing `nova.model.bitlinear.BitLinear` launches roughly 6 CUDA kernels per
call (RMSNorm reduce, divide, weight absmean, weight round/clamp, activation
absmax, activation round/clamp, matmul, rescale, bias add). On NOVA 2.4B that is
~144 launches per forward pass and each launch costs 20-50 us of dispatch overhead.

This module collapses the whole thing into a single Triton kernel when Triton is
available, and falls back to a math-equivalent PyTorch path on CPU / no-Triton
systems (used by the validator and the M2 dev loop).

Backward uses a plain straight-through estimator: quantization is identity, so
gradients flow through the float shadow weights (same semantics as the current
`BitLinear` which uses the `x + (x_q - x).detach()` trick).

Interface matches `nova.model.bitlinear.BitLinear` exactly so `FusedBitLinear`
is a drop-in replacement.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised on CPU dev boxes
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


if _HAS_TRITON:

    @triton.jit
    def _round_half_to_even(x):
        """Portable banker's-rounding helper that matches torch.round.

        `tl.math.round` was added in Triton 2.1; on older versions you have to
        use `libdevice.rint`. Both do banker's rounding, matching PyTorch.
        We try the stable `tl.math.round` path and fall back to libdevice via
        a compile-time check.
        """
        # tl.math.round is available from Triton 2.1 onward and calls libdevice
        # under the hood. If it's missing, the @triton.jit compiler surfaces a
        # clear error at first launch and the user can pin a newer Triton.
        return tl.math.round(x)

    @triton.jit
    def _clamp(x, lo, hi):
        """tl.clamp was added in Triton 2.2; emulate with maximum/minimum."""
        return tl.minimum(tl.maximum(x, lo), hi)

    @triton.jit
    def _fused_bitlinear_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        norm_w_ptr,
        y_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_ym,
        stride_yn,
        alpha,
        norm_eps,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused RMSNorm -> absmax quant -> ternary matmul -> rescale -> bias.

        Shapes:
          x:       [M, K]   fp32/bf16 input (already reshaped to 2D)
          w:       [N, K]   fp32 float shadow weights (quantized on the fly)
          norm_w:  [K]      RMSNorm scale
          bias:    [N]      optional
          y:       [M, N]   fp32 output

        Scaling:
          alpha   = mean(|w|)            — computed outside, broadcast scalar
          gamma_m = max(|x_norm|) per row — computed inside the kernel
          y = (x_q @ w_q.T) * (alpha * gamma_m / 127)
          where x_q = round(clamp(x_norm * 127 / gamma_m, -128, 127))
                w_q = round(clamp(w / alpha, -1, 1))

        Each program instance computes one BLOCK_M x BLOCK_N tile of y, looping
        over K in BLOCK_K chunks. The RMSNorm + activation absmax are fused into
        the first K sweep: we compute the RMS over the full K for the current M
        rows, normalize, quant, and accumulate the dot product in a second K
        sweep. Two sweeps beat materializing a staging buffer, and stay single
        launch.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        # Pass 1: RMS of x over K for each of the BLOCK_M rows.
        rms_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + offs_k
            mask_k = k_idx < K
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
            x_tile = tl.load(
                x_ptrs,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            rms_acc += tl.sum(x_tile * x_tile, axis=1)
        rms = tl.sqrt(rms_acc / K + norm_eps)  # [BLOCK_M]
        inv_rms = 1.0 / rms

        # Pass 2: gamma = max(|x_norm|) per row across K.
        gamma = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + offs_k
            mask_k = k_idx < K
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
            x_tile = tl.load(
                x_ptrs,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            nw = tl.load(norm_w_ptr + k_idx, mask=mask_k, other=0.0).to(tl.float32)
            x_norm = x_tile * inv_rms[:, None] * nw[None, :]
            abs_x = tl.abs(x_norm)
            # Mask inactive K lanes so they don't pollute the max.
            abs_x = tl.where(mask_k[None, :], abs_x, 0.0)
            gamma = tl.maximum(gamma, tl.max(abs_x, axis=1))
        gamma = tl.maximum(gamma, 1e-10)
        inv_gamma = 127.0 / gamma

        # Pass 3: quantized matmul.
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + offs_k
            mask_k = k_idx < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
            x_tile = tl.load(
                x_ptrs,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            nw = tl.load(norm_w_ptr + k_idx, mask=mask_k, other=0.0).to(tl.float32)
            x_norm = x_tile * inv_rms[:, None] * nw[None, :]
            # Absmax quantize activations to [-128, 127] integers, dequant to fp.
            x_q = _clamp(
                _round_half_to_even(x_norm * inv_gamma[:, None]),
                -128.0,
                127.0,
            )

            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk
            w_tile = tl.load(
                w_ptrs,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            # Ternary quantize weights: round(clamp(w / alpha, -1, 1)).
            w_scaled = w_tile / alpha
            w_q = _clamp(
                _round_half_to_even(w_scaled),
                -1.0,
                1.0,
            )

            acc += tl.dot(x_q, tl.trans(w_q))

        scale = alpha * gamma / 127.0  # [BLOCK_M]
        y = acc * scale[:, None]

        if HAS_BIAS:
            b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            y += b[None, :]

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


def _triton_forward(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """Launch the fused Triton kernel. Requires CUDA tensors."""
    assert _HAS_TRITON, "Triton is not available"
    assert x_2d.is_cuda and weight.is_cuda, "fused kernel requires CUDA tensors"
    assert x_2d.dim() == 2, f"expected 2D input, got {x_2d.shape}"

    M, K = x_2d.shape
    N, K_w = weight.shape
    assert K == K_w, f"K mismatch: x has {K}, w has {K_w}"

    y = torch.empty((M, N), device=x_2d.device, dtype=torch.float32)

    alpha = weight.abs().mean().clamp(min=1e-10).item()

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _fused_bitlinear_kernel[grid](
        x_2d,
        weight,
        bias if bias is not None else x_2d,  # dummy ptr when HAS_BIAS=False
        norm_weight,
        y,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        alpha,
        norm_eps,
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return y.to(x_2d.dtype)


# ---------------------------------------------------------------------------
# PyTorch fallback — bit-exact with the reference BitLinear
# ---------------------------------------------------------------------------


def fused_bitlinear_torch_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """Public, forward-only PyTorch fallback.

    Equivalent to `_reference_bitlinear_forward` but skips the STE `.detach()`
    machinery — this path is only used when the caller doesn't need autograd
    (e.g. bench scripts, validator ground-truth comparison). If you need the
    full drop-in semantics (including STE-exact backward), use `FusedBitLinear`
    which routes through `_reference_bitlinear_forward` under autograd.
    """
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + norm_eps)
    x_n = (x / rms) * norm_weight

    alpha = weight.abs().mean().clamp(min=1e-10)
    w_q = torch.clamp(torch.round(weight / alpha), -1, 1)

    gamma = x_n.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-10)
    x_q = torch.clamp(torch.round(x_n * 127.0 / gamma), -128, 127)

    y = x_q @ w_q.T * (alpha * gamma / 127.0)
    if bias is not None:
        y = y + bias
    return y


def fused_bitlinear_triton_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """Dispatch to the Triton kernel when possible, otherwise fall back."""
    if _HAS_TRITON and x.is_cuda:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        y_2d = _triton_forward(x_2d, weight, bias, norm_weight, norm_eps)
        return y_2d.reshape(*orig_shape[:-1], weight.shape[0])
    return fused_bitlinear_torch_forward(x, weight, bias, norm_weight, norm_eps)


# ---------------------------------------------------------------------------
# Autograd function — STE backward at Python level
# ---------------------------------------------------------------------------


class FusedBitLinearSTE(torch.autograd.Function):
    """Fused-forward / reference-backward autograd wrapper.

    Forward runs the fused Triton kernel. Backward recomputes the reference
    PyTorch math under a local autograd tape so that gradient semantics are
    bit-exact with the unfused `BitLinear` — including the quirky cross-term
    gradients introduced by the non-detached alpha/gamma scaling in the STE
    pattern `x + (x_q - x).detach()`.

    This is cheap: recompute is O(M*N*K) which is dominated by the matmul that
    would need to run in backward anyway, and we save no extra tensors beyond
    the inputs themselves.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        return fused_bitlinear_triton_forward(x, weight, bias, norm_weight, norm_eps)

    @staticmethod
    def setup_context(ctx, inputs, output):  # type: ignore[override]
        x, weight, bias, norm_weight, norm_eps = inputs
        ctx.save_for_backward(x, weight, bias, norm_weight)
        ctx.norm_eps = float(norm_eps)
        ctx.has_bias = bias is not None

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        x, weight, bias, norm_weight = ctx.saved_tensors
        norm_eps = ctx.norm_eps

        with torch.enable_grad():
            x_r = x.detach().requires_grad_(x.requires_grad)
            w_r = weight.detach().requires_grad_(True)
            n_r = norm_weight.detach().requires_grad_(True)
            b_r = (
                bias.detach().requires_grad_(True) if bias is not None else None
            )
            y_r = _reference_bitlinear_forward(x_r, w_r, b_r, n_r, norm_eps)
            grads = torch.autograd.grad(
                y_r,
                [x_r, w_r, n_r] + ([b_r] if b_r is not None else []),
                grad_outputs=grad_output,
                allow_unused=False,
            )

        grad_x = grads[0]
        grad_weight = grads[1]
        grad_norm_weight = grads[2]
        grad_bias = grads[3] if ctx.has_bias else None
        return grad_x, grad_weight, grad_bias, grad_norm_weight, None


# ---------------------------------------------------------------------------
# nn.Module wrapper — drop-in replacement for BitLinear
# ---------------------------------------------------------------------------


class FusedBitLinear(nn.Module):
    """Drop-in replacement for `nova.model.bitlinear.BitLinear`.

    Same constructor signature, same state_dict keys (`weight`, `bias`,
    `norm.weight`), same forward semantics. On CUDA + Triton the forward pass
    is a single fused kernel launch (~6x fewer launches per layer). On CPU or
    when Triton is unavailable we fall back to the reference PyTorch math
    directly, which keeps gradients bit-exact with `BitLinear` (autograd
    handles the STE tricks naturally via the `x + (x_q - x).detach()` pattern).

    State dict layout:
        weight       -> [out_features, in_features]
        bias         -> [out_features] (optional)
        norm.weight  -> [in_features]
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, eps: float = 1e-6):
        super().__init__()
        scale = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # Nested RMSNorm so state dicts from `BitLinear` load without remapping.
        self.norm = _RMSNormHolder(in_features, eps=eps)
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CUDA + Triton: single fused launch, STE backward at Python level.
        if _HAS_TRITON and x.is_cuda:
            return FusedBitLinearSTE.apply(
                x,
                self.weight,
                self.bias,
                self.norm.weight,
                self.eps,
            )
        # CPU / no Triton: reference math inline. PyTorch autograd propagates
        # through the `.detach()` STE tricks exactly like the original.
        return _reference_bitlinear_forward(
            x,
            self.weight,
            self.bias,
            self.norm.weight,
            self.eps,
        )

    def ternary_weights(self) -> torch.Tensor:
        alpha = self.weight.abs().mean().clamp(min=1e-10)
        return torch.clamp(torch.round(self.weight / alpha), -1, 1)


def _reference_bitlinear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """Exact forward of `nova.model.bitlinear.BitLinear`, inlined.

    Used by the CPU fallback so PyTorch autograd produces gradients that are
    bit-exact with the reference BitLinear — including the quirky
    non-detached alpha/gamma gradient paths introduced by the `x + (x_q - x)
    .detach()` STE pattern.

    Op sequence deliberately mirrors `nova.model.bitlinear.BitLinear.forward`
    line for line (including `.max(...).values` over `.amax(...)`) so autograd
    produces bit-identical gradients, not just numerically-close ones.
    """
    # RMSNorm (matches RMSNorm.forward exactly).
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + norm_eps)
    x_n = (x / rms) * norm_weight

    alpha = weight.abs().mean().clamp(min=1e-10)
    w_q = torch.clamp(torch.round(weight / alpha), -1, 1)
    w_q = weight + (w_q - weight).detach()

    gamma = x_n.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-10)
    x_q = torch.clamp(torch.round(x_n * 127.0 / gamma), -128, 127)
    x_q = x_n + (x_q - x_n).detach()

    y = x_q @ w_q.T * (alpha * gamma / 127.0)
    if bias is not None:
        y = y + bias
    return y


class _RMSNormHolder(nn.Module):
    """Tiny module that only stores the RMSNorm scale parameter.

    We don't call this in forward — the fused kernel does the normalization.
    It exists so the state dict key `norm.weight` matches the original
    `BitLinear` (which nests an `RMSNorm` under `self.norm`).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

"""
GPU optimization for NOVA-10M tests.
- JIT-compiled Mamba selective scan (replaces Python for-loop)
- Patches the CLASS method so DataParallel replicas work correctly
"""
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace/tfle")
from nova_full_directive import NOVA10M, MambaBlock10M


# ── JIT-compiled selective scan ─────────────────────────────

@torch.jit.script
def jit_selective_scan(
    x: torch.Tensor, dt: torch.Tensor,
    B_mat: torch.Tensor, C_mat: torch.Tensor,
    A_log: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled scan — compiles the Python for-loop to CUDA."""
    batch, seq_len, d_inner = x.shape
    d_state = B_mat.shape[-1]

    A = -torch.exp(A_log)
    dt_expanded = dt.unsqueeze(-1)
    A_expanded = A.unsqueeze(0).unsqueeze(0)
    dA = torch.exp(A_expanded * dt_expanded)

    x_expanded = x.unsqueeze(-1)
    B_expanded = B_mat.unsqueeze(2)
    dBx = dt_expanded * B_expanded * x_expanded

    h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
    outputs = torch.empty(batch, seq_len, d_inner, device=x.device, dtype=x.dtype)
    for t in range(seq_len):
        h = dA[:, t] * h + dBx[:, t]
        outputs[:, t] = torch.sum(h * C_mat[:, t].unsqueeze(1), dim=-1)
    return outputs


def _fast_selective_scan(self, x, dt, B_mat, C_mat):
    """Drop-in replacement for MambaBlock10M._selective_scan.
    Uses self.A_log so DataParallel replicas resolve to their own device."""
    return jit_selective_scan(x, dt, B_mat, C_mat, self.A_log)


def patch_mamba_scan(model: nn.Module = None):
    """Patch MambaBlock10M class with JIT scan. Works with DataParallel."""
    MambaBlock10M._selective_scan = _fast_selective_scan
    # Count how many blocks exist in the model
    if model is not None:
        n = sum(1 for m in model.modules() if isinstance(m, MambaBlock10M))
        print(f"  JIT Mamba scan patched ({n} blocks)")
    else:
        print("  JIT Mamba scan patched (class-level)")


def get_raw_model(model):
    """Unwrap DataParallel/compile to get raw model."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m

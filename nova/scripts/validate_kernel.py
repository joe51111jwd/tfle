"""Validate that FusedBitLinear matches BitLinear over a full training loop.

Runs 100 training steps twice:
  1. A model built with the reference `nova.model.bitlinear.BitLinear`
  2. The same model with `nova.training.fused_bitlinear.FusedBitLinear` swapped in

Same seed, same data, same optimizer state. Losses are compared step by step
and must stay within 0.01% of each other. If they diverge the script prints
the first diverging step along with the maximum per-layer parameter drift and
exits non-zero so CI / the developer notices.

Usage:
    python -m nova.scripts.validate_kernel
    python -m nova.scripts.validate_kernel --steps 100 --tol 1e-4 --device cpu
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nova.model.bitlinear import BitLinear, RMSNorm
from nova.training.fused_bitlinear import FusedBitLinear


# ---------------------------------------------------------------------------
# Tiny model that exercises BitLinear exactly how NOVA does it
# ---------------------------------------------------------------------------


class TinyFFNBlock(nn.Module):
    """Mimics `nova.model.nova_2_4b.FFN` shape and activation exactly."""

    def __init__(self, hidden_dim: int, ff_dim: int, linear_cls: type[nn.Module]):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.up = linear_cls(hidden_dim, ff_dim)
        self.down = linear_cls(ff_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.up(x)
        x = F.relu(x) ** 2
        x = self.down(x)
        return x + residual


class TinyModel(nn.Module):
    """Small stack of FFN blocks + LM head. Enough to exercise gradient flow
    through multiple BitLinear layers per step, which is what the validator
    cares about.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        ff_dim: int,
        n_layers: int,
        linear_cls: type[nn.Module],
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList(
            [TinyFFNBlock(hidden_dim, ff_dim, linear_cls) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    step: int
    loss: float


def _make_batches(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    n_steps: int,
    device: torch.device,
    seed: int,
) -> list[torch.Tensor]:
    """Produce a deterministic sequence of token batches."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.randint(
            0, vocab_size, (batch_size, seq_len), generator=gen, device="cpu"
        ).to(device)
        for _ in range(n_steps)
    ]


def _run_training(
    model: nn.Module,
    batches: list[torch.Tensor],
    lr: float,
    device: torch.device,
    seed: int,
) -> list[StepResult]:
    """Run `len(batches)` training steps and return per-step losses.

    Uses a causal LM objective: predict token t+1 from tokens [0..t].
    """
    torch.manual_seed(seed)  # ensures any non-determinism in ops starts the same
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    results: list[StepResult] = []
    for step, batch in enumerate(batches):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
        results.append(StepResult(step=step, loss=loss.item()))
    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _clone_state_into(dst: nn.Module, src: nn.Module) -> None:
    """Copy `src` state dict into `dst`. Works because FusedBitLinear keeps
    the same parameter keys as BitLinear (`weight`, `bias`, `norm.weight`).
    """
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch: missing={missing}, unexpected={unexpected}. "
            "FusedBitLinear should be a drop-in replacement for BitLinear."
        )


def _max_param_drift(ref: nn.Module, fused: nn.Module) -> tuple[str, float]:
    """Return (name, max abs diff) of the largest per-parameter drift."""
    worst_name = ""
    worst_val = 0.0
    ref_params = dict(ref.named_parameters())
    for name, p_fused in fused.named_parameters():
        p_ref = ref_params[name]
        diff = (p_ref.detach() - p_fused.detach()).abs().max().item()
        if diff > worst_val:
            worst_val = diff
            worst_name = name
    return worst_name, worst_val


def validate(
    n_steps: int,
    rel_tol: float,
    device: torch.device,
    seed: int,
) -> int:
    """Returns 0 on success, 1 on failure."""
    vocab_size = 256
    hidden_dim = 64
    ff_dim = 128
    n_layers = 3
    batch_size = 4
    seq_len = 32
    lr = 1e-2

    torch.manual_seed(seed)
    ref_model = TinyModel(vocab_size, hidden_dim, ff_dim, n_layers, BitLinear).to(device)

    # Same init then swap BitLinear -> FusedBitLinear via state_dict.
    torch.manual_seed(seed)
    fused_model = TinyModel(vocab_size, hidden_dim, ff_dim, n_layers, FusedBitLinear).to(
        device
    )
    _clone_state_into(fused_model, ref_model)

    batches = _make_batches(vocab_size, batch_size, seq_len, n_steps, device, seed)

    print(
        f"[validate_kernel] {n_steps} steps, device={device}, "
        f"hidden={hidden_dim}, layers={n_layers}, rel_tol={rel_tol:.1e}"
    )

    ref_results = _run_training(ref_model, batches, lr, device, seed=seed)
    fused_results = _run_training(fused_model, batches, lr, device, seed=seed)

    first_div_step: int | None = None
    max_rel_diff = 0.0

    for ref_r, fused_r in zip(ref_results, fused_results):
        denom = max(abs(ref_r.loss), 1e-12)
        rel_diff = abs(ref_r.loss - fused_r.loss) / denom
        max_rel_diff = max(max_rel_diff, rel_diff)
        if rel_diff > rel_tol and first_div_step is None:
            first_div_step = ref_r.step

    if first_div_step is None:
        print(
            f"[validate_kernel] PASS — max rel loss diff = {max_rel_diff:.3e} "
            f"over {n_steps} steps"
        )
        # Final param drift is also printed for diagnostic purposes.
        name, drift = _max_param_drift(ref_model, fused_model)
        print(f"[validate_kernel] max param drift: {name} = {drift:.3e}")
        return 0

    # ---- Divergence path: print diagnostics and exit non-zero. ----
    ref_loss = ref_results[first_div_step].loss
    fused_loss = fused_results[first_div_step].loss
    print(
        f"[validate_kernel] FAIL — loss diverged at step {first_div_step}: "
        f"ref={ref_loss:.8f}, fused={fused_loss:.8f}, "
        f"rel_diff={abs(ref_loss - fused_loss) / max(abs(ref_loss), 1e-12):.3e}"
    )

    # Rerun up to diverge point with layer-level diagnostics to narrow down.
    print("[validate_kernel] reprobing layer outputs at the diverge step...")
    torch.manual_seed(seed)
    r2 = TinyModel(vocab_size, hidden_dim, ff_dim, n_layers, BitLinear).to(device)
    torch.manual_seed(seed)
    f2 = TinyModel(vocab_size, hidden_dim, ff_dim, n_layers, FusedBitLinear).to(device)
    _clone_state_into(f2, r2)
    o2 = torch.optim.SGD(r2.parameters(), lr=lr)
    o3 = torch.optim.SGD(f2.parameters(), lr=lr)
    for step, batch in enumerate(batches):
        if step == first_div_step:
            break
        for m, o in ((r2, o2), (f2, o3)):
            m.train()
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            lg = m(inp)
            ls = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))
            o.zero_grad()
            ls.backward()
            o.step()

    # Now forward once and report per-layer deltas.
    r2.eval()
    f2.eval()
    with torch.no_grad():
        b = batches[first_div_step][:, :-1]
        x_ref = r2.embed(b)
        x_fused = f2.embed(b)
        for i, (br, bf) in enumerate(zip(r2.blocks, f2.blocks)):
            x_ref = br(x_ref)
            x_fused = bf(x_fused)
            diff = (x_ref - x_fused).abs().max().item()
            print(f"  layer {i}: max|ref - fused| = {diff:.3e}")

    name, drift = _max_param_drift(r2, f2)
    print(f"[validate_kernel] pre-diverge max param drift: {name} = {drift:.3e}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4, help="relative loss tolerance, default 0.01 percent")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[validate_kernel] CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")

    return validate(
        n_steps=args.steps,
        rel_tol=args.tol,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())

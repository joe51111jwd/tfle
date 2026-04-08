"""Post-prune ternarization for NOVA (CONDITIONAL).

CONDITIONAL: Only used if pruning tests pass. Do not activate in production
without verifying accuracy after the full prune-ternarize-recover pipeline.

Converts pruned float weights back to ternary via absmean quantization
(same method as BitLinear). Keeps embeddings, LayerNorm/RMSNorm weights,
and biases as float32. Stores original floats as shadow weights for
STE recovery training.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Ternarizer:
    """CONDITIONAL: Convert float model weights to ternary after pruning.

    Uses BitLinear's existing quantization scheme:
        alpha = w.abs().mean()
        w_q = sign(w) * alpha  (after rounding w/alpha to {-1, 0, +1})

    Float32 originals are stored as _shadow_weight attributes for
    STE recovery training.
    """

    def __init__(self):
        self.ternarized_layers: list[str] = []
        self.skipped_layers: list[str] = []

    def _should_skip(self, name: str, module: nn.Module) -> bool:
        """Determine if a layer should stay float32.

        Skip: embeddings, LayerNorm/RMSNorm, biases, lm_head.
        """
        if isinstance(module, (nn.Embedding, nn.LayerNorm)):
            return True
        if "embed" in name.lower():
            return True
        if "norm" in name.lower():
            return True
        if "lm_head" in name.lower():
            return True
        # RMSNorm from bitlinear.py
        if type(module).__name__ == "RMSNorm":
            return True
        return False

    def ternarize_model(self, model: nn.Module) -> nn.Module:
        """Convert eligible float weights to ternary via absmean quantization.

        For each eligible linear layer:
        1. Store original float weights as shadow weights (_shadow_weight)
        2. Compute alpha = w.abs().mean()
        3. Quantize: w_q = clamp(round(w / alpha), -1, 1)
        4. Write back: w = w_q * alpha

        Biases within ternarized layers are kept as float32.
        """
        self.ternarized_layers = []
        self.skipped_layers = []

        for name, module in model.named_modules():
            if self._should_skip(name, module):
                self.skipped_layers.append(name)
                continue

            if not hasattr(module, "weight") or module.weight.dim() != 2:
                continue

            with torch.no_grad():
                w = module.weight.data
                alpha = w.abs().mean().clamp(min=1e-10)
                w_q = torch.clamp(torch.round(w / alpha), -1, 1)

                # store original floats as shadow weights for STE recovery
                module.register_buffer(
                    "_shadow_weight", w.clone(), persistent=False
                )

                # write ternary weights
                module.weight.data.copy_(w_q * alpha)

            self.ternarized_layers.append(name)

        logger.info(
            f"Ternarized {len(self.ternarized_layers)} layers, "
            f"skipped {len(self.skipped_layers)} (embeddings/norms/biases)"
        )
        return model

    def restore_from_shadow(self, model: nn.Module) -> nn.Module:
        """Restore original float weights from shadow copies.

        Used if ternarization fails quality checks and you need to revert.
        """
        restored = 0
        for name, module in model.named_modules():
            if hasattr(module, "_shadow_weight"):
                with torch.no_grad():
                    module.weight.data.copy_(module._shadow_weight)
                restored += 1

        logger.info(f"Restored {restored} layers from shadow weights")
        return model

    def get_ternary_stats(self, model: nn.Module) -> dict:
        """Report ternarization statistics for the model."""
        total_params = 0
        ternary_params = 0
        zero_fraction = 0.0
        n_layers = 0

        for name, module in model.named_modules():
            if not hasattr(module, "weight") or module.weight.dim() != 2:
                continue
            if self._should_skip(name, module):
                continue

            w = module.weight.data
            total_params += w.numel()

            alpha = w.abs().mean().clamp(min=1e-10)
            w_q = torch.clamp(torch.round(w / alpha), -1, 1)
            ternary_params += w.numel()
            zero_fraction += (w_q == 0).float().mean().item()
            n_layers += 1

        return {
            "total_params": total_params,
            "ternary_params": ternary_params,
            "avg_zero_fraction": zero_fraction / max(n_layers, 1),
            "n_ternary_layers": n_layers,
        }

"""Prune-and-Ternarize: model pruning for NOVA (CONDITIONAL).

CONDITIONAL: Only used if pruning tests pass. Do not activate in production
without verifying that accuracy remains acceptable after pruning.

Two pruning strategies:
- Uniform depth pruning: remove layers, keeping every Nth
- Importance-based width pruning: prune attention heads by L1 norm,
  FFN neurons by activation magnitude
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

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

"""CDLL: Compression-Driven Layer Learning.

Local fitness function that rewards layers for compressing information
while preserving mutual information with inputs. Deeper layers compress
more aggressively (alpha scales with depth).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TFLEConfig


class CDLLFitness:
    """Compression-Driven Layer Learning fitness for a single layer.

    Fitness = -alpha * entropy(output) + beta * mutual_info(input, output)

    Deeper layers get higher alpha, encouraging progressive compression.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_idx: int,
        config: TFLEConfig,
        device: torch.device,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.layer_idx = layer_idx
        self.config = config
        self.device = device
        self.n_bins = config.cdll_n_bins

        # Alpha scales with depth: deeper layers compress more
        self.alpha = config.cdll_alpha * (config.cdll_depth_alpha_scale ** layer_idx)
        self.beta = config.cdll_beta

        # Optional reconstruction decoder
        self.decoder: nn.Module | None = None
        if config.cdll_reconstruction:
            self.decoder = nn.Sequential(
                nn.Linear(out_features, min(out_features * 2, in_features)),
                nn.ReLU(),
                nn.Linear(min(out_features * 2, in_features), in_features),
            ).to(device)
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(), lr=1e-3
            )

    def _compute_entropy(self, activations: torch.Tensor) -> torch.Tensor:
        """Estimate entropy of activation patterns via binned histogram.

        Args:
            activations: (B, out_features) layer outputs.

        Returns:
            Scalar entropy estimate.
        """
        # Bin each neuron's activations and compute per-neuron entropy, then average
        act = activations.detach()
        if act.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        # Clamp to avoid extreme outliers in histogram
        act_min = act.min()
        act_max = act.max()
        if act_max - act_min < 1e-8:
            return torch.tensor(0.0, device=self.device)

        # Normalize to [0, 1] for binning
        act_norm = (act - act_min) / (act_max - act_min + 1e-8)
        bin_indices = (act_norm * (self.n_bins - 1)).long().clamp(0, self.n_bins - 1)

        # Per-neuron histogram -> entropy, averaged across neurons
        total_entropy = torch.tensor(0.0, device=self.device)
        B = act.shape[0]
        for j in range(act.shape[1]):
            counts = torch.bincount(bin_indices[:, j], minlength=self.n_bins).float()
            probs = counts / B
            probs = probs[probs > 0]
            total_entropy -= (probs * probs.log()).sum()

        return total_entropy / act.shape[1]

    def _compute_mutual_info(
        self, layer_input: torch.Tensor, activations: torch.Tensor
    ) -> torch.Tensor:
        """Estimate mutual information via variance-based cross-covariance.

        Uses trace of cross-covariance matrix as a proxy for MI.

        Args:
            layer_input: (B, in_features).
            activations: (B, out_features).

        Returns:
            Scalar MI estimate.
        """
        x = layer_input.detach()
        y = activations.detach()

        if x.shape[0] < 2:
            return torch.tensor(0.0, device=self.device)

        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)

        # Cross-covariance: (out, in)
        B = x.shape[0]
        cross_cov = (y_centered.T @ x_centered) / (B - 1)

        # Frobenius norm of cross-covariance as MI proxy
        mi = cross_cov.norm() / (x.shape[1] * y.shape[1]) ** 0.5

        return mi

    def _compute_reconstruction(
        self, layer_input: torch.Tensor, activations: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruction proxy: tiny decoder tries to recover input from output.

        Returns negative reconstruction error (higher = better).
        """
        if self.decoder is None:
            return torch.tensor(0.0, device=self.device)

        x = layer_input.detach()
        y = activations.detach()

        # Train the decoder for one step
        self.decoder.train()
        reconstructed = self.decoder(y)
        loss = F.mse_loss(reconstructed, x)
        self.decoder_optimizer.zero_grad()
        loss.backward()
        self.decoder_optimizer.step()

        # Evaluate
        self.decoder.eval()
        with torch.no_grad():
            reconstructed = self.decoder(y)
            recon_error = F.mse_loss(reconstructed, x)

        return -recon_error

    def compute(
        self, layer_input: torch.Tensor, activations: torch.Tensor
    ) -> float:
        """Compute CDLL fitness.

        fitness = -alpha * entropy + beta * mutual_info [+ reconstruction]

        Returns:
            Scalar fitness value (higher is better).
        """
        entropy = self._compute_entropy(activations)
        mi = self._compute_mutual_info(layer_input, activations)
        fitness = -self.alpha * entropy + self.beta * mi

        if self.config.cdll_reconstruction and self.decoder is not None:
            recon = self._compute_reconstruction(layer_input, activations)
            fitness = fitness + recon

        return fitness.item()

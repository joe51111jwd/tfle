"""CDLL: Compression-Driven Layer Learning.

Local fitness: L = alpha * H(output) - beta * I(output; input)

Each layer's quality measured by compression (low entropy) + preservation
(high mutual information with input). Deeper layers compress harder.

Alpha scales linearly with depth: 0.3 (layer 0) → 0.8 (deepest hidden).
Beta = 1.0. Entropy via 32-bin histogram. MI via squared cross-covariance trace.

This is the ONLY fitness function for TFLE. It's local — no full-model
forward pass needed — so layers can train in parallel across GPUs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TFLEConfig


class CDLLFitness:
    """Compression-Driven Layer Learning fitness for a single layer.

    fitness = -alpha * entropy(output) + beta * mutual_info(input, output)
    Higher = better (less entropy, more MI).
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

        # Alpha scales linearly with depth: start → end
        n_layers = len(config.layer_sizes) - 1
        if n_layers > 1:
            depth_frac = layer_idx / (n_layers - 1)
        else:
            depth_frac = 0.0
        self.alpha = config.cdll_alpha_start + (config.cdll_alpha_end - config.cdll_alpha_start) * depth_frac
        self.beta = config.cdll_beta

        # Optional reconstruction decoder (legacy support)
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
        """Histogram entropy of activations — fully vectorized on GPU.

        For batch of activations (B, D), computes per-neuron entropy via
        one-hot binning as a single batched operation. No Python loops.
        """
        act = activations.detach()
        if act.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        act_min = act.min()
        act_max = act.max()
        if act_max - act_min < 1e-8:
            return torch.tensor(0.0, device=self.device)

        B, D = act.shape
        sample_d = min(D, 256)
        act_sampled = act[:, :sample_d]

        # Normalize to [0, 1] and bin
        act_norm = (act_sampled - act_min) / (act_max - act_min + 1e-8)
        bin_indices = (act_norm * (self.n_bins - 1)).long().clamp(0, self.n_bins - 1)

        # One-hot encode bins: (B, sample_d) -> (B, sample_d, n_bins)
        one_hot = torch.zeros(B, sample_d, self.n_bins, device=self.device)
        one_hot.scatter_(2, bin_indices.unsqueeze(2), 1.0)

        # Sum over batch to get histograms per neuron: (sample_d, n_bins)
        counts = one_hot.sum(dim=0)
        probs = counts / B

        # Entropy per neuron: -sum(p * log(p)) for p > 0
        log_probs = torch.where(probs > 0, probs.log(), torch.zeros_like(probs))
        entropy_per_neuron = -(probs * log_probs).sum(dim=1)  # (sample_d,)

        return entropy_per_neuron.mean()

    def _compute_mutual_info(
        self, layer_input: torch.Tensor, activations: torch.Tensor
    ) -> torch.Tensor:
        """Variance-based MI proxy: trace of squared cross-covariance matrix.

        Higher = more information preserved from input.
        """
        x = layer_input.detach()
        y = activations.detach()

        if x.shape[0] < 2:
            return torch.tensor(0.0, device=self.device)

        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)

        B = x.shape[0]
        # Sample dimensions if large
        x_dim = min(x.shape[1], 256)
        y_dim = min(y.shape[1], 256)
        cross_cov = (y_centered[:, :y_dim].T @ x_centered[:, :x_dim]) / (B - 1)

        # Frobenius norm normalized by dimensionality
        mi = cross_cov.norm() / (x_dim * y_dim) ** 0.5
        return mi

    @staticmethod
    def _stabilize(t: torch.Tensor) -> torch.Tensor:
        """Standardize to zero mean, unit variance. Prevents overflow in deep layers."""
        std = t.std()
        if std < 1e-8:
            return t - t.mean()
        return (t - t.mean()) / std

    def compute(
        self, layer_input: torch.Tensor, activations: torch.Tensor
    ) -> float:
        """Compute CDLL fitness. Higher = better.

        fitness = -alpha * entropy + beta * mutual_info
        """
        # Stabilize to prevent overflow in deep layers
        act_stable = self._stabilize(activations)
        inp_stable = self._stabilize(layer_input)

        entropy = self._compute_entropy(act_stable)
        mi = self._compute_mutual_info(inp_stable, act_stable)
        fitness = -self.alpha * entropy + self.beta * mi

        if self.config.cdll_reconstruction and self.decoder is not None:
            recon = self._compute_reconstruction(layer_input, activations)
            fitness = fitness + recon

        return fitness.item()

    def compute_batch(self, layer_input: torch.Tensor, activations_k: torch.Tensor) -> torch.Tensor:
        """Compute CDLL fitness for K proposals at once. No Python loops.

        Args:
            layer_input: (B, in_features) — same for all proposals
            activations_k: (K, B, out_features) — K outputs

        Returns:
            (K,) fitness values
        """
        K, B, D = activations_k.shape
        act = activations_k.detach()

        # Stabilize each proposal independently
        act_mean = act.mean(dim=(1, 2), keepdim=True)
        act_std = act.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        act = (act - act_mean) / act_std

        sample_d = min(D, 256)

        # --- Batched entropy ---
        act_flat = act.reshape(K, -1)
        a_min = act_flat.min(dim=1).values.view(K, 1, 1)
        a_max = act_flat.max(dim=1).values.view(K, 1, 1)
        spread = a_max - a_min + 1e-8
        zero_mask = (spread.squeeze() < 1e-7)

        act_norm = (act[:, :, :sample_d] - a_min) / spread
        bin_idx = (act_norm * (self.n_bins - 1)).long().clamp(0, self.n_bins - 1)

        one_hot = torch.zeros(K, B, sample_d, self.n_bins, device=self.device)
        one_hot.scatter_(3, bin_idx.unsqueeze(3), 1.0)
        counts = one_hot.sum(dim=1)  # (K, sample_d, n_bins)
        probs = counts / B
        log_p = torch.where(probs > 0, probs.log(), torch.zeros_like(probs))
        entropy_k = -(probs * log_p).sum(dim=2).mean(dim=1)  # (K,)
        if zero_mask.any():
            entropy_k = entropy_k.masked_fill(zero_mask, 0.0)

        # --- Batched MI ---
        x = layer_input.detach()
        x_std = x.std()
        if x_std > 1e-8:
            x = (x - x.mean()) / x_std
        x_c = x - x.mean(dim=0, keepdim=True)
        x_dim = min(x.shape[1], 256)
        y_dim = sample_d
        x_sub = x_c[:, :x_dim]  # (B, x_dim)
        y_sub = act[:, :, :y_dim]  # (K, B, y_dim)
        y_c = y_sub - y_sub.mean(dim=1, keepdim=True)
        x_exp = x_sub.unsqueeze(0).expand(K, -1, -1)
        cc = torch.bmm(y_c.transpose(1, 2), x_exp) / max(B - 1, 1)  # (K, y_dim, x_dim)
        mi_k = cc.reshape(K, -1).norm(dim=1) / (x_dim * y_dim) ** 0.5

        return -self.alpha * entropy_k + self.beta * mi_k

    def _compute_reconstruction(
        self, layer_input: torch.Tensor, activations: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruction proxy (optional, legacy)."""
        if self.decoder is None:
            return torch.tensor(0.0, device=self.device)
        x = layer_input.detach()
        y = activations.detach()
        self.decoder.train()
        reconstructed = self.decoder(y)
        loss = F.mse_loss(reconstructed, x)
        self.decoder_optimizer.zero_grad()
        loss.backward()
        self.decoder_optimizer.step()
        self.decoder.eval()
        with torch.no_grad():
            recon_error = F.mse_loss(self.decoder(y), x)
        return -recon_error

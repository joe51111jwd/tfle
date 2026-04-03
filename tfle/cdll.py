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
        self.gamma = config.cdll_gamma  # class separation weight

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

    def _compute_class_separation(self, activations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Vectorized Fisher discriminant — no Python loops over classes.

        Uses scatter_mean to compute all class means at once.
        Approximates I(representation; labels).
        """
        act = activations.detach()
        B, D = act.shape
        sample_d = min(D, 64)  # smaller sample for speed (100 classes is heavy)
        act = act[:, :sample_d]

        global_mean = act.mean(dim=0)  # (sample_d,)

        # Compute class means via scatter
        num_classes = int(labels.max().item()) + 1
        # One-hot encode labels: (B, num_classes)
        one_hot = torch.zeros(B, num_classes, device=self.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        class_counts = one_hot.sum(dim=0).clamp(min=1)  # (num_classes,)

        # Class means: (num_classes, sample_d) = (num_classes, B) @ (B, sample_d)
        class_sums = one_hot.T @ act  # (num_classes, sample_d)
        class_means = class_sums / class_counts.unsqueeze(1)

        # Between-class variance: sum_c n_c * (mean_c - global_mean)^2
        diffs = class_means - global_mean.unsqueeze(0)  # (num_classes, sample_d)
        between = (class_counts.unsqueeze(1) * diffs ** 2).sum()

        # Within-class variance: sum of (x - class_mean)^2 for each x
        # Expand class means to each sample: (B, sample_d)
        sample_class_means = class_means[labels]  # (B, sample_d)
        within = ((act - sample_class_means) ** 2).sum()

        if within < 1e-8:
            return torch.tensor(1.0 if between > 0 else 0.0, device=self.device)

        ratio = between / (within + 1e-8)
        return torch.sigmoid(ratio - 1.0)

    @staticmethod
    def _stabilize(t: torch.Tensor) -> torch.Tensor:
        """Standardize to zero mean, unit variance. Prevents overflow in deep layers."""
        std = t.std()
        if std < 1e-8:
            return t - t.mean()
        return (t - t.mean()) / std

    def compute(
        self, layer_input: torch.Tensor, activations: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> float:
        """Compute CDLL fitness. Higher = better.

        Full Information Bottleneck:
          fitness = -alpha * H(T) + beta * I(T;X) + gamma * I(T;Y)
        Where T=representation, X=input, Y=labels
        """
        act_stable = self._stabilize(activations)
        inp_stable = self._stabilize(layer_input)

        entropy = self._compute_entropy(act_stable)
        mi = self._compute_mutual_info(inp_stable, act_stable)
        fitness = -self.alpha * entropy + self.beta * mi

        # The missing IB term: I(T;Y)
        if labels is not None and self.gamma > 0:
            class_sep = self._compute_class_separation(activations, labels)
            fitness = fitness + self.gamma * class_sep

        result = fitness.item()
        if result != result:
            return 0.0
        return result

    def compute_batch(
        self, layer_input: torch.Tensor, activations_k: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute CDLL fitness for K proposals. Vectorized + NaN-safe.

        Stabilizes per-proposal, computes entropy via one-hot, MI via bmm,
        class separation via Fisher discriminant.
        """
        K, B, D = activations_k.shape
        dev = self.device

        # Stabilize each proposal: zero mean, unit variance
        act = activations_k.detach().clone()
        act_mean = act.mean(dim=(1, 2), keepdim=True)
        act_std = act.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        act = (act - act_mean) / act_std

        # Stabilize shared input
        inp = layer_input.detach()
        inp_std = inp.std().clamp(min=1e-6)
        inp = (inp - inp.mean()) / inp_std

        sample_d = min(D, 256)

        # --- Entropy (K,) ---
        act_s = act[:, :, :sample_d]  # (K, B, sample_d)
        # Per-proposal min/max for binning
        flat = act_s.reshape(K, -1)
        a_min = flat.min(dim=1).values.view(K, 1, 1)
        a_max = flat.max(dim=1).values.view(K, 1, 1)
        spread = (a_max - a_min).clamp(min=1e-6)

        normed = (act_s - a_min) / spread
        bin_idx = (normed * (self.n_bins - 1)).long().clamp(0, self.n_bins - 1)

        # One-hot: (K, B, sample_d) -> (K, B, sample_d, n_bins)
        one_hot = torch.zeros(K, B, sample_d, self.n_bins, device=dev)
        one_hot.scatter_(3, bin_idx.unsqueeze(3), 1.0)
        counts = one_hot.sum(dim=1)  # (K, sample_d, n_bins)
        probs = counts / B
        log_p = torch.where(probs > 0, probs.log(), torch.zeros_like(probs))
        entropy_k = -(probs * log_p).sum(dim=2).mean(dim=1)  # (K,)

        # --- MI (K,) ---
        x_dim = min(inp.shape[1], 256)
        y_dim = sample_d
        x_c = inp[:, :x_dim] - inp[:, :x_dim].mean(dim=0, keepdim=True)  # (B, x_dim)
        y_c = act_s - act_s.mean(dim=1, keepdim=True)  # (K, B, sample_d)
        x_exp = x_c.unsqueeze(0).expand(K, -1, -1)  # (K, B, x_dim)
        cc = torch.bmm(y_c.transpose(1, 2), x_exp) / max(B - 1, 1)  # (K, y_dim, x_dim)
        mi_k = cc.reshape(K, -1).norm(dim=1) / max((x_dim * y_dim) ** 0.5, 1e-6)

        result = -self.alpha * entropy_k + self.beta * mi_k

        # Class separation: I(T;Y) — computed per proposal via loop (small K)
        if labels is not None and self.gamma > 0:
            for k in range(K):
                cs = self._compute_class_separation(activations_k[k], labels)
                result[k] = result[k] + self.gamma * cs

        return torch.nan_to_num(result, nan=0.0)

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

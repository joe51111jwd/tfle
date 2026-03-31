"""Fitness functions: contrastive and predictive coding."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import TFLEConfig
from .layers import compute_goodness


class ContrastiveFitness:
    """Forward-Forward style contrastive goodness fitness."""

    def __init__(self, config: TFLEConfig):
        self.config = config

    def compute(
        self,
        out_real: torch.Tensor,
        out_corrupted: torch.Tensor,
    ) -> float:
        goodness_real = compute_goodness(out_real, self.config.goodness_metric).mean()
        goodness_fake = compute_goodness(out_corrupted, self.config.goodness_metric).mean()
        return (goodness_real - goodness_fake).item()


class PredictiveCodingFitness:
    """Predictive coding fitness: each layer predicts what the next layer expects.

    Layer N's output should match Layer N+1's prediction of its input.
    The prediction is based on a simple linear predictor learned via
    exponential moving average of recent (input, output) pairs.
    """

    def __init__(self, in_features: int, out_features: int, config: TFLEConfig):
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        # Running statistics for prediction
        self._mean_input = torch.zeros(in_features)
        self._mean_output = torch.zeros(out_features)
        self._covariance = torch.zeros(out_features, in_features)
        self._n_updates = 0
        self._ema_decay = 0.99

    def update_statistics(self, layer_input: torch.Tensor, layer_output: torch.Tensor):
        """Update running statistics with new observations."""
        # EMA of input/output means and covariance
        batch_mean_in = layer_input.mean(dim=0).detach()
        batch_mean_out = layer_output.mean(dim=0).detach()

        if self._n_updates == 0:
            self._mean_input = batch_mean_in
            self._mean_output = batch_mean_out
        else:
            d = self._ema_decay
            self._mean_input = d * self._mean_input + (1 - d) * batch_mean_in
            self._mean_output = d * self._mean_output + (1 - d) * batch_mean_out

        # Update covariance estimate
        centered_in = layer_input - self._mean_input.unsqueeze(0)
        centered_out = layer_output - self._mean_output.unsqueeze(0)
        batch_cov = (centered_out.T @ centered_in) / max(layer_input.size(0), 1)

        if self._n_updates == 0:
            self._covariance = batch_cov.detach()
        else:
            self._covariance = (
                self._ema_decay * self._covariance + (1 - self._ema_decay) * batch_cov.detach()
            )

        self._n_updates += 1

    def predict_output(self, layer_input: torch.Tensor) -> torch.Tensor:
        """Predict what this layer's output should look like."""
        centered = layer_input - self._mean_input.unsqueeze(0)
        predicted = centered @ self._covariance.T + self._mean_output.unsqueeze(0)
        return predicted

    def compute(
        self,
        layer_input: torch.Tensor,
        actual_output: torch.Tensor,
    ) -> float:
        """Compute predictive fitness: negative prediction error."""
        if self._n_updates < 5:
            # Not enough data for meaningful prediction yet
            self.update_statistics(layer_input, actual_output)
            return 0.0

        predicted = self.predict_output(layer_input)
        # Negative MSE (higher = better prediction = better fitness)
        prediction_error = F.mse_loss(actual_output, predicted).item()
        self.update_statistics(layer_input, actual_output)
        return -prediction_error


class HybridFitness:
    """Combines contrastive and predictive fitness."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: TFLEConfig,
        contrastive_weight: float = 0.5,
    ):
        self.contrastive = ContrastiveFitness(config)
        self.predictive = PredictiveCodingFitness(in_features, out_features, config)
        self.contrastive_weight = contrastive_weight

    def compute(
        self,
        layer_input: torch.Tensor,
        out_real: torch.Tensor,
        out_corrupted: torch.Tensor,
    ) -> float:
        c_fitness = self.contrastive.compute(out_real, out_corrupted)
        p_fitness = self.predictive.compute(layer_input, out_real)
        return self.contrastive_weight * c_fitness + (1 - self.contrastive_weight) * p_fitness

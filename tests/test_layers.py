"""Tests for ternary layers."""

import pytest
import torch

from tfle.config import GoodnessMetric, InitMethod, TFLEConfig
from tfle.layers import (
    TFLELayer,
    compute_goodness,
    random_ternary,
    ternary_matmul,
)


class TestRandomTernary:
    def test_values_are_ternary(self):
        config = TFLEConfig(init_method=InitMethod.BALANCED_RANDOM)
        w = random_ternary(100, 50, config)
        unique = set(w.unique().tolist())
        assert unique.issubset({-1, 0, 1})

    def test_shape(self):
        config = TFLEConfig()
        w = random_ternary(64, 32, config)
        assert w.shape == (64, 32)

    def test_dtype_is_int8(self):
        config = TFLEConfig()
        w = random_ternary(10, 10, config)
        assert w.dtype == torch.int8

    def test_sparse_init_has_more_zeros(self):
        config = TFLEConfig(init_method=InitMethod.SPARSE_RANDOM, init_seed=42)
        w = random_ternary(1000, 100, config)
        zero_ratio = (w == 0).float().mean().item()
        assert zero_ratio > 0.5  # sparse should have ~60% zeros

    def test_kaiming_init(self):
        config = TFLEConfig(init_method=InitMethod.KAIMING_TERNARY, init_seed=42)
        w = random_ternary(1000, 100, config)
        unique = set(w.unique().tolist())
        assert unique.issubset({-1, 0, 1})

    def test_seed_reproducibility(self):
        config = TFLEConfig(init_seed=123)
        w1 = random_ternary(50, 50, config)
        w2 = random_ternary(50, 50, config)
        assert torch.equal(w1, w2)

    def test_zero_bias(self):
        config = TFLEConfig(init_zero_bias=0.8, init_seed=42)
        w = random_ternary(1000, 100, config)
        zero_ratio = (w == 0).float().mean().item()
        assert zero_ratio > 0.7  # should be close to 0.8


class TestTernaryMatmul:
    def test_basic_multiplication(self):
        x = torch.ones(2, 3)
        w = torch.tensor([[1, -1], [0, 1], [-1, 0]], dtype=torch.int8)
        result = ternary_matmul(x, w)
        # x @ w = [[1+0-1, -1+1+0]] = [[0, 0]]
        expected = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        assert torch.allclose(result, expected)

    def test_output_shape(self):
        x = torch.randn(8, 64)
        w = torch.randint(-1, 2, (64, 32), dtype=torch.int8)
        result = ternary_matmul(x, w)
        assert result.shape == (8, 32)


class TestComputeGoodness:
    def test_sum_of_squares(self):
        a = torch.tensor([[1.0, 2.0, 3.0]])
        g = compute_goodness(a, GoodnessMetric.SUM_OF_SQUARES)
        assert g.item() == pytest.approx(14.0)

    def test_mean_activation(self):
        a = torch.tensor([[-1.0, 2.0, -3.0]])
        g = compute_goodness(a, GoodnessMetric.MEAN_ACTIVATION)
        assert g.item() == pytest.approx(2.0)

    def test_max_activation(self):
        a = torch.tensor([[-1.0, 5.0, -3.0]])
        g = compute_goodness(a, GoodnessMetric.MAX_ACTIVATION)
        assert g.item() == pytest.approx(5.0)

    def test_entropy(self):
        a = torch.tensor([[1.0, 1.0, 1.0]])
        g = compute_goodness(a, GoodnessMetric.ENTROPY)
        # Uniform distribution should have max entropy
        assert g.item() > 0

    def test_batch_dimension(self):
        a = torch.randn(16, 64)
        g = compute_goodness(a, GoodnessMetric.SUM_OF_SQUARES)
        assert g.shape == (16,)


class TestTFLELayer:
    def test_creation(self):
        config = TFLEConfig()
        layer = TFLELayer(784, 512, config, layer_idx=0)
        assert layer.weights.shape == (784, 512)
        assert layer.in_features == 784
        assert layer.out_features == 512

    def test_forward(self):
        config = TFLEConfig()
        layer = TFLELayer(64, 32, config)
        x = torch.randn(8, 64)
        out = layer.forward(x)
        assert out.shape == (8, 32)

    def test_train_step_returns_metrics(self):
        config = TFLEConfig(flip_rate=0.05)
        layer = TFLELayer(64, 32, config)
        x_real = torch.randn(16, 64)
        x_corrupt = torch.randn(16, 64)
        metrics = layer.train_step(x_real, x_corrupt, temperature=5.0)
        assert "accepted" in metrics
        assert "fitness_before" in metrics
        assert "fitness_after" in metrics
        assert "delta" in metrics
        assert "n_candidates" in metrics

    def test_train_step_changes_weights_sometimes(self):
        config = TFLEConfig(flip_rate=0.1, initial_temperature=100.0)
        layer = TFLELayer(32, 16, config)
        original_weights = layer.weights.clone()
        x_real = torch.randn(32, 32)
        x_corrupt = torch.randn(32, 32)

        # Run many steps — at high temperature, some should be accepted
        changed = False
        for _ in range(50):
            layer.train_step(x_real, x_corrupt, temperature=100.0)
            if not torch.equal(layer.weights, original_weights):
                changed = True
                break
        assert changed, "Weights should change after many steps at high temperature"

    def test_weight_distribution(self):
        config = TFLEConfig()
        layer = TFLELayer(100, 50, config)
        dist = layer.get_weight_distribution()
        assert "neg1" in dist
        assert "zero" in dist
        assert "pos1" in dist
        assert dist["neg1"] + dist["zero"] + dist["pos1"] == pytest.approx(1.0)

    def test_trace_statistics(self):
        config = TFLEConfig(separate_pos_neg_traces=True)
        layer = TFLELayer(32, 16, config)
        stats = layer.get_trace_statistics()
        assert "error_trace_mean" in stats
        assert "success_trace_mean" in stats

    def test_acceptance_rate_starts_zero(self):
        config = TFLEConfig()
        layer = TFLELayer(32, 16, config)
        assert layer.get_acceptance_rate() == 0.0

    def test_separate_traces(self):
        config = TFLEConfig(separate_pos_neg_traces=True)
        layer = TFLELayer(32, 16, config)
        assert hasattr(layer, "success_traces")
        assert hasattr(layer, "error_traces")
        assert layer.success_traces.shape == (32, 16)

    def test_single_trace(self):
        config = TFLEConfig(separate_pos_neg_traces=False)
        layer = TFLELayer(32, 16, config)
        assert hasattr(layer, "traces")
        assert layer.traces.shape == (32, 16)

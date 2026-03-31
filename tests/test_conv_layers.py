"""Tests for ternary convolutional layers."""

import pytest
import torch

from tfle.config import TFLEConfig
from tfle.conv_layers import TFLEConvLayer, ternary_conv2d


class TestTernaryConv2d:
    def test_output_shape(self):
        x = torch.randn(4, 3, 32, 32)
        w = torch.randint(-1, 2, (16, 3, 3, 3), dtype=torch.int8)
        out = ternary_conv2d(x, w, padding=1)
        assert out.shape == (4, 16, 32, 32)

    def test_no_padding_reduces_spatial(self):
        x = torch.randn(2, 1, 8, 8)
        w = torch.randint(-1, 2, (4, 1, 3, 3), dtype=torch.int8)
        out = ternary_conv2d(x, w, padding=0)
        assert out.shape == (2, 4, 6, 6)

    def test_stride(self):
        x = torch.randn(2, 1, 8, 8)
        w = torch.randint(-1, 2, (4, 1, 3, 3), dtype=torch.int8)
        out = ternary_conv2d(x, w, stride=2, padding=1)
        assert out.shape == (2, 4, 4, 4)


class TestTFLEConvLayer:
    def test_creation(self):
        config = TFLEConfig()
        layer = TFLEConvLayer(3, 16, 3, config, padding=1)
        assert layer.weights.shape == (16, 3, 3, 3)
        assert layer.n_weights == 16 * 3 * 3 * 3

    def test_forward(self):
        config = TFLEConfig()
        layer = TFLEConvLayer(3, 16, 3, config, padding=1)
        x = torch.randn(4, 3, 32, 32)
        out = layer.forward(x)
        assert out.shape == (4, 16, 32, 32)

    def test_weights_are_ternary(self):
        config = TFLEConfig()
        layer = TFLEConvLayer(3, 16, 3, config)
        unique = set(layer.weights.unique().tolist())
        assert unique.issubset({-1, 0, 1})

    def test_train_step(self):
        config = TFLEConfig(flip_rate=0.05)
        layer = TFLEConvLayer(3, 8, 3, config, padding=1)
        x_real = torch.randn(4, 3, 8, 8)
        x_corrupt = torch.randn(4, 3, 8, 8)
        metrics = layer.train_step(x_real, x_corrupt, temperature=5.0)
        assert "accepted" in metrics
        assert "fitness_before" in metrics
        assert "n_candidates" in metrics

    def test_train_step_modifies_weights(self):
        config = TFLEConfig(flip_rate=0.1, initial_temperature=100.0)
        layer = TFLEConvLayer(3, 8, 3, config, padding=1)
        original = layer.weights.clone()
        x_real = torch.randn(8, 3, 8, 8)
        x_corrupt = torch.randn(8, 3, 8, 8)
        changed = False
        for _ in range(50):
            layer.train_step(x_real, x_corrupt, temperature=100.0)
            if not torch.equal(layer.weights, original):
                changed = True
                break
        assert changed

    def test_weight_distribution(self):
        config = TFLEConfig()
        layer = TFLEConvLayer(3, 16, 3, config)
        dist = layer.get_weight_distribution()
        assert dist["neg1"] + dist["zero"] + dist["pos1"] == pytest.approx(1.0)

    def test_acceptance_rate(self):
        config = TFLEConfig()
        layer = TFLEConvLayer(3, 8, 3, config, padding=1)
        assert layer.get_acceptance_rate() == 0.0
        # Run a few steps
        x = torch.randn(4, 3, 8, 8)
        for _ in range(10):
            layer.train_step(x, torch.randn_like(x), temperature=10.0)
        rate = layer.get_acceptance_rate()
        assert 0.0 <= rate <= 1.0

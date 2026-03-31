"""Tests for TFLE model."""

import tempfile

import torch

from tfle.config import TFLEConfig
from tfle.model import TFLEModel


class TestTFLEModel:
    def test_creation(self):
        config = TFLEConfig(layer_sizes=[64, 32, 16, 10])
        model = TFLEModel(config)
        assert len(model.layers) == 3

    def test_forward_shape(self):
        config = TFLEConfig(layer_sizes=[64, 32, 10])
        model = TFLEModel(config)
        x = torch.randn(8, 64)
        out = model.forward(x)
        assert out.shape == (8, 10)

    def test_predict(self):
        config = TFLEConfig(layer_sizes=[64, 32, 10])
        model = TFLEModel(config)
        x = torch.randn(8, 64)
        preds = model.predict(x)
        assert preds.shape == (8,)
        assert preds.min() >= 0
        assert preds.max() < 10

    def test_train_step(self):
        config = TFLEConfig(layer_sizes=[32, 16, 10], flip_rate=0.1)
        model = TFLEModel(config)
        x = torch.randn(16, 32)
        labels = torch.randint(0, 10, (16,))
        metrics = model.train_step(x, temperature=5.0, labels=labels)
        assert len(metrics) == 2  # 2 layers
        for m in metrics:
            assert "accepted" in m

    def test_evaluate(self):
        config = TFLEConfig(layer_sizes=[32, 16, 10])
        model = TFLEModel(config)
        x = torch.randn(16, 32)
        labels = torch.randint(0, 10, (16,))
        result = model.evaluate(x, labels)
        assert "accuracy" in result
        assert "loss" in result
        assert 0 <= result["accuracy"] <= 1

    def test_total_params(self):
        config = TFLEConfig(layer_sizes=[784, 512, 256, 10])
        model = TFLEModel(config)
        expected = 784 * 512 + 512 * 256 + 256 * 10
        assert model.get_total_params() == expected

    def test_memory_usage(self):
        config = TFLEConfig(layer_sizes=[784, 512, 256, 10])
        model = TFLEModel(config)
        mem = model.get_memory_usage_bytes()
        assert mem["total_bytes"] > 0
        assert mem["total_mb"] > 0

    def test_checkpoint_save_load(self):
        config = TFLEConfig(layer_sizes=[32, 16, 10], init_seed=42)
        model = TFLEModel(config)
        original_weights = [layer.weights.clone() for layer in model.layers]

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            model.save_checkpoint(path)

        # Modify weights
        for layer in model.layers:
            layer.weights = torch.zeros_like(layer.weights, dtype=torch.int8)

        # Load and verify restoration
        model.load_checkpoint(path)
        for orig, loaded in zip(original_weights, [layer.weights for layer in model.layers]):
            assert torch.equal(orig, loaded)

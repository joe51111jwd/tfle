"""Tests for convolutional TFLE model."""

import torch

from tfle.config import TFLEConfig
from tfle.conv_model import TFLEConvModel


class TestTFLEConvModel:
    def test_creation(self):
        config = TFLEConfig()
        model = TFLEConvModel(config, n_classes=10)
        assert len(model.conv_layers) == 3
        assert len(model.fc_layers) == 2

    def test_forward_shape(self):
        config = TFLEConfig()
        model = TFLEConvModel(config, n_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model.forward(x)
        assert out.shape == (4, 10)

    def test_predict(self):
        config = TFLEConfig()
        model = TFLEConvModel(config, n_classes=10)
        x = torch.randn(4, 3, 32, 32)
        preds = model.predict(x)
        assert preds.shape == (4,)
        assert preds.min() >= 0
        assert preds.max() < 10

    def test_train_step(self):
        config = TFLEConfig(flip_rate=0.05)
        model = TFLEConvModel(config, n_classes=10)
        x = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        metrics = model.train_step(x, temperature=5.0, labels=labels)
        assert len(metrics) == 5  # 3 conv + 2 fc
        for m in metrics:
            assert "accepted" in m

    def test_evaluate(self):
        config = TFLEConfig()
        model = TFLEConvModel(config, n_classes=10)
        x = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        result = model.evaluate(x, labels)
        assert "accuracy" in result
        assert "loss" in result

    def test_total_params(self):
        config = TFLEConfig()
        model = TFLEConvModel(config, n_classes=10)
        params = model.get_total_params()
        assert params > 0
        # Conv: 3*32*3*3 + 32*64*3*3 + 64*128*3*3
        # FC: 2048*256 + 256*10
        expected_conv = 3*32*3*3 + 32*64*3*3 + 64*128*3*3
        expected_fc = 2048*256 + 256*10
        assert params == expected_conv + expected_fc

    def test_memory_usage(self):
        config = TFLEConfig()
        model = TFLEConvModel(config, n_classes=10)
        mem = model.get_memory_usage_bytes()
        assert mem["total_bytes"] > 0
        assert mem["total_mb"] > 0

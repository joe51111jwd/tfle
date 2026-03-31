"""Tests for the training loop."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from tfle.config import TFLEConfig
from tfle.model import TFLEModel
from tfle.training import TFLETrainer


def make_dummy_loaders(n_train=200, n_val=50, in_features=32, n_classes=5, batch_size=16):
    """Create simple dummy dataloaders for testing."""
    x_train = torch.randn(n_train, in_features)
    y_train = torch.randint(0, n_classes, (n_train,))
    x_val = torch.randn(n_val, in_features)
    y_val = torch.randint(0, n_classes, (n_val,))

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)
    return train_loader, val_loader


class TestTFLETrainer:
    def test_training_runs(self):
        config = TFLEConfig(
            layer_sizes=[32, 16, 5],
            total_training_steps=50,
            eval_interval=10,
            early_stopping_patience=1000,
        )
        model = TFLEModel(config)
        train_loader, val_loader = make_dummy_loaders(in_features=32, n_classes=5)
        trainer = TFLETrainer(model, config, train_loader, val_loader)
        result = trainer.train(verbose=False)
        assert result.total_steps > 0
        assert result.training_time_seconds > 0
        assert len(result.val_accuracies) > 0

    def test_early_stopping(self):
        config = TFLEConfig(
            layer_sizes=[32, 16, 5],
            total_training_steps=10_000,
            eval_interval=5,
            early_stopping_patience=3,
        )
        model = TFLEModel(config)
        train_loader, val_loader = make_dummy_loaders(in_features=32, n_classes=5)
        trainer = TFLETrainer(model, config, train_loader, val_loader)
        result = trainer.train(verbose=False)
        # Should stop well before 10,000 steps
        assert result.total_steps < 10_000

    def test_training_without_val_loader(self):
        config = TFLEConfig(
            layer_sizes=[32, 16, 5],
            total_training_steps=30,
            eval_interval=10,
            early_stopping_patience=1000,
        )
        model = TFLEModel(config)
        train_loader, _ = make_dummy_loaders(in_features=32, n_classes=5)
        trainer = TFLETrainer(model, config, train_loader, val_loader=None)
        result = trainer.train(verbose=False)
        assert result.total_steps > 0

    def test_memory_usage_reported(self):
        config = TFLEConfig(layer_sizes=[32, 16, 5], total_training_steps=10, eval_interval=5)
        model = TFLEModel(config)
        train_loader, val_loader = make_dummy_loaders(in_features=32, n_classes=5)
        trainer = TFLETrainer(model, config, train_loader, val_loader)
        result = trainer.train(verbose=False)
        assert result.memory_usage["total_bytes"] > 0

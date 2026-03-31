"""Tests for ternary transformer."""

import torch

from tfle.config import TFLEConfig
from tfle.transformer import TFLEAttention, TFLETransformerBlock, TFLETransformerModel


class TestTFLEAttention:
    def test_creation(self):
        config = TFLEConfig()
        attn = TFLEAttention(64, 4, config)
        assert attn.embed_dim == 64
        assert attn.n_heads == 4
        assert attn.head_dim == 16

    def test_forward_shape(self):
        config = TFLEConfig()
        attn = TFLEAttention(64, 4, config)
        x = torch.randn(2, 8, 64)
        out = attn.forward(x)
        assert out.shape == (2, 8, 64)

    def test_with_mask(self):
        config = TFLEConfig()
        attn = TFLEAttention(64, 4, config)
        x = torch.randn(2, 8, 64)
        mask = torch.tril(torch.ones(8, 8)).unsqueeze(0).unsqueeze(0)
        out = attn.forward(x, mask)
        assert out.shape == (2, 8, 64)

    def test_projection_layers(self):
        config = TFLEConfig()
        attn = TFLEAttention(64, 4, config)
        layers = attn.get_projection_layers()
        assert len(layers) == 4  # Q, K, V, Output


class TestTFLETransformerBlock:
    def test_forward_shape(self):
        config = TFLEConfig()
        block = TFLETransformerBlock(64, 4, 128, config)
        x = torch.randn(2, 8, 64)
        out = block.forward(x)
        assert out.shape == (2, 8, 64)

    def test_residual_connection(self):
        config = TFLEConfig()
        block = TFLETransformerBlock(64, 4, 128, config)
        x = torch.randn(2, 8, 64)
        out = block.forward(x)
        # Output should not be identical to input (residual adds something)
        assert not torch.allclose(x, out)

    def test_get_all_layers(self):
        config = TFLEConfig()
        block = TFLETransformerBlock(64, 4, 128, config)
        layers = block.get_all_layers()
        # 4 attention projections + 2 FF layers = 6
        assert len(layers) == 6


class TestTFLETransformerModel:
    def test_creation(self):
        config = TFLEConfig()
        model = TFLETransformerModel(
            vocab_size=100, embed_dim=64, n_heads=4,
            n_layers=2, ff_dim=128, max_seq_len=32, config=config,
        )
        assert len(model.blocks) == 2

    def test_forward_shape(self):
        config = TFLEConfig()
        model = TFLETransformerModel(
            vocab_size=100, embed_dim=64, n_heads=4,
            n_layers=2, ff_dim=128, max_seq_len=32, config=config,
        )
        tokens = torch.randint(0, 100, (2, 16))
        logits = model.forward(tokens)
        assert logits.shape == (2, 16, 100)

    def test_total_params(self):
        config = TFLEConfig()
        model = TFLETransformerModel(
            vocab_size=100, embed_dim=64, n_heads=4,
            n_layers=1, ff_dim=128, max_seq_len=32, config=config,
        )
        params = model.get_total_params()
        assert params > 0
        # Should include embedding params + ternary layer params
        all_layers = model.get_all_layers()
        ternary_params = sum(
            layer.in_features * layer.out_features for layer in all_layers
        )
        embedding_params = 100 * 64 + 32 * 64
        assert params == ternary_params + embedding_params

    def test_train_step(self):
        config = TFLEConfig(flip_rate=0.05)
        model = TFLETransformerModel(
            vocab_size=50, embed_dim=32, n_heads=2,
            n_layers=1, ff_dim=64, max_seq_len=16, config=config,
        )
        tokens = torch.randint(0, 50, (4, 8))
        metrics = model.train_step(tokens, temperature=5.0)
        assert len(metrics) > 0
        for m in metrics:
            assert "accepted" in m

    def test_evaluate(self):
        config = TFLEConfig()
        model = TFLETransformerModel(
            vocab_size=50, embed_dim=32, n_heads=2,
            n_layers=1, ff_dim=64, max_seq_len=16, config=config,
        )
        tokens = torch.randint(0, 50, (4, 8))
        result = model.evaluate(tokens, tokens)
        assert "loss" in result
        assert "accuracy" in result
        assert result["loss"] > 0
        assert 0.0 <= result["accuracy"] <= 1.0

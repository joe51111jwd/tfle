"""Tests for STE baseline model."""

import torch

from tfle.baseline import STEBaselineModel, ste_ternary


class TestSTETernary:
    def test_output_is_ternary(self):
        x = torch.randn(100)
        out = ste_ternary(x)
        # Output should be approximately ternary
        unique = out.unique()
        for v in unique:
            assert v.item() in {-1.0, 0.0, 1.0} or abs(v.item()) < 0.01

    def test_gradient_flows(self):
        x = torch.randn(10, requires_grad=True)
        out = ste_ternary(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        # STE: gradient should be 1 (pass-through)
        assert (x.grad == 1.0).all()


class TestSTEBaselineModel:
    def test_forward(self):
        model = STEBaselineModel([64, 32, 10])
        x = torch.randn(8, 64)
        out = model(x)
        assert out.shape == (8, 10)

    def test_backward(self):
        model = STEBaselineModel([32, 16, 10])
        x = torch.randn(4, 32)
        labels = torch.randint(0, 10, (4,))
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, labels)
        loss.backward()
        for layer in model.layers:
            assert layer.weight.grad is not None

    def test_parameter_count(self):
        model = STEBaselineModel([784, 512, 256, 10])
        n_params = sum(p.numel() for p in model.parameters())
        expected = 784 * 512 + 512 * 256 + 256 * 10
        assert n_params == expected

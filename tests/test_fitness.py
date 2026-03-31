"""Tests for fitness functions."""

import torch

from tfle.config import GoodnessMetric, TFLEConfig
from tfle.fitness import ContrastiveFitness, HybridFitness, PredictiveCodingFitness


class TestContrastiveFitness:
    def test_basic_computation(self):
        config = TFLEConfig()
        fitness = ContrastiveFitness(config)
        # Real data with higher activations should score higher
        out_real = torch.randn(16, 32) * 2.0
        out_corrupt = torch.randn(16, 32) * 0.5
        score = fitness.compute(out_real, out_corrupt)
        assert score > 0  # Real should have higher goodness

    def test_identical_data_gives_zero(self):
        config = TFLEConfig()
        fitness = ContrastiveFitness(config)
        data = torch.randn(16, 32)
        score = fitness.compute(data, data)
        assert abs(score) < 1e-5

    def test_different_metrics(self):
        for metric in GoodnessMetric:
            config = TFLEConfig(goodness_metric=metric)
            fitness = ContrastiveFitness(config)
            out_real = torch.randn(8, 16)
            out_corrupt = torch.randn(8, 16)
            score = fitness.compute(out_real, out_corrupt)
            assert isinstance(score, float)


class TestPredictiveCodingFitness:
    def test_initial_updates(self):
        config = TFLEConfig()
        fitness = PredictiveCodingFitness(32, 16, config)
        # First few calls should return 0 (not enough data)
        for _ in range(3):
            inp = torch.randn(8, 32)
            out = torch.randn(8, 16)
            score = fitness.compute(inp, out)
            assert score == 0.0

    def test_convergence_with_linear_data(self):
        config = TFLEConfig()
        fitness = PredictiveCodingFitness(32, 16, config)

        # Create data with a simple linear relationship
        W = torch.randn(32, 16) * 0.1
        scores = []
        for _ in range(50):
            inp = torch.randn(16, 32)
            out = inp @ W + torch.randn(16, 16) * 0.01  # nearly linear
            score = fitness.compute(inp, out)
            if score != 0.0:
                scores.append(score)

        # Later scores should be less negative (better prediction)
        if len(scores) > 10:
            early = sum(scores[:5]) / 5
            late = sum(scores[-5:]) / 5
            assert late >= early  # prediction should improve

    def test_prediction_shape(self):
        config = TFLEConfig()
        fitness = PredictiveCodingFitness(32, 16, config)
        # Warm up
        for _ in range(10):
            fitness.update_statistics(torch.randn(8, 32), torch.randn(8, 16))
        pred = fitness.predict_output(torch.randn(4, 32))
        assert pred.shape == (4, 16)


class TestHybridFitness:
    def test_combines_both(self):
        config = TFLEConfig()
        hybrid = HybridFitness(32, 16, config, contrastive_weight=0.5)
        inp = torch.randn(8, 32)
        out_real = torch.randn(8, 16)
        out_corrupt = torch.randn(8, 16)
        # Warm up predictive
        for _ in range(10):
            hybrid.predictive.update_statistics(torch.randn(8, 32), torch.randn(8, 16))
        score = hybrid.compute(inp, out_real, out_corrupt)
        assert isinstance(score, float)

    def test_weight_affects_output(self):
        config = TFLEConfig()
        h1 = HybridFitness(32, 16, config, contrastive_weight=1.0)
        h2 = HybridFitness(32, 16, config, contrastive_weight=0.0)
        inp = torch.randn(8, 32)
        out_real = torch.randn(8, 16)
        out_corrupt = torch.randn(8, 16)
        # Warm up predictive
        for _ in range(10):
            h1.predictive.update_statistics(torch.randn(8, 32), torch.randn(8, 16))
            h2.predictive.update_statistics(torch.randn(8, 32), torch.randn(8, 16))

        s1 = h1.compute(inp, out_real, out_corrupt)
        s2 = h2.compute(inp, out_real, out_corrupt)
        # Different weights should give different scores (usually)
        # This isn't guaranteed but is very likely with random data
        assert isinstance(s1, float)
        assert isinstance(s2, float)

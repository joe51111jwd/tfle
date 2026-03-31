"""Tests for data corruption strategies."""

import torch

from tfle.config import CorruptionMethod, TFLEConfig
from tfle.corruption import corrupt_data, generate_negative_samples


class TestCorruptData:
    def test_label_shuffle_preserves_shape(self):
        config = TFLEConfig(corruption_method=CorruptionMethod.LABEL_SHUFFLE)
        x = torch.randn(32, 64)
        corrupted = corrupt_data(x, config)
        assert corrupted.shape == x.shape

    def test_label_shuffle_is_permutation(self):
        config = TFLEConfig(corruption_method=CorruptionMethod.LABEL_SHUFFLE)
        x = torch.arange(10).float().unsqueeze(1)
        corrupted = corrupt_data(x, config)
        # Same elements, different order (usually)
        assert set(corrupted.flatten().tolist()) == set(x.flatten().tolist())

    def test_gaussian_noise(self):
        config = TFLEConfig(
            corruption_method=CorruptionMethod.GAUSSIAN_NOISE,
            corruption_strength=0.5,
        )
        x = torch.zeros(32, 64)
        corrupted = corrupt_data(x, config)
        assert corrupted.shape == x.shape
        assert not torch.equal(corrupted, x)

    def test_input_mask(self):
        config = TFLEConfig(
            corruption_method=CorruptionMethod.INPUT_MASK,
            corruption_strength=0.5,
        )
        x = torch.ones(32, 64)
        corrupted = corrupt_data(x, config)
        # Some values should be zero
        assert (corrupted == 0).any()
        # Some should still be 1
        assert (corrupted == 1).any()

    def test_feature_permute(self):
        config = TFLEConfig(
            corruption_method=CorruptionMethod.FEATURE_PERMUTE,
            corruption_strength=0.5,
        )
        x = torch.randn(32, 64)
        corrupted = corrupt_data(x, config)
        assert corrupted.shape == x.shape

    def test_mixup(self):
        config = TFLEConfig(
            corruption_method=CorruptionMethod.MIXUP,
            corruption_strength=0.5,
        )
        x = torch.randn(32, 64)
        corrupted = corrupt_data(x, config)
        assert corrupted.shape == x.shape


class TestGenerateNegativeSamples:
    def test_correct_count(self):
        config = TFLEConfig(num_negative_samples=3)
        x = torch.randn(16, 32)
        samples = generate_negative_samples(x, config)
        assert len(samples) == 3

    def test_each_sample_has_correct_shape(self):
        config = TFLEConfig(num_negative_samples=2)
        x = torch.randn(16, 32)
        samples = generate_negative_samples(x, config)
        for s in samples:
            assert s.shape == x.shape

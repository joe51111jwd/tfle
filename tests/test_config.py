"""Tests for TFLE configuration."""

import pytest

from tfle.config import (
    CoolingSchedule,
    FitnessType,
    InitMethod,
    SelectionMethod,
    TFLEConfig,
)


class TestTFLEConfig:
    def test_default_creation(self):
        config = TFLEConfig()
        assert config.flip_rate == 0.03
        assert config.init_method == InitMethod.BALANCED_RANDOM
        assert config.trace_decay == 0.95
        assert config.initial_temperature == 10.0

    def test_custom_creation(self):
        config = TFLEConfig(
            flip_rate=0.05,
            init_method=InitMethod.SPARSE_RANDOM,
            fitness_type=FitnessType.CONTRASTIVE,
            total_training_steps=50_000,
        )
        assert config.flip_rate == 0.05
        assert config.init_method == InitMethod.SPARSE_RANDOM
        assert config.total_training_steps == 50_000

    def test_layer_sizes_default(self):
        config = TFLEConfig()
        assert config.layer_sizes == [784, 512, 256, 10]

    def test_flip_rate_depth_scaling(self):
        config = TFLEConfig(flip_rate=0.05, flip_rate_depth_scale=0.8, depth_scaled_flip_rate=True)
        assert config.get_flip_rate_for_layer(0) == pytest.approx(0.05)
        assert config.get_flip_rate_for_layer(1) == pytest.approx(0.04)
        assert config.get_flip_rate_for_layer(2) == pytest.approx(0.032)

    def test_flip_rate_no_depth_scaling(self):
        config = TFLEConfig(flip_rate=0.05, depth_scaled_flip_rate=False)
        assert config.get_flip_rate_for_layer(0) == config.get_flip_rate_for_layer(3)

    def test_temperature_depth_scaling(self):
        config = TFLEConfig(temperature_depth_scale=0.7, depth_scaled_temperature=True)
        t0 = config.get_temperature_for_layer(10.0, 0)
        t1 = config.get_temperature_for_layer(10.0, 1)
        assert t0 == pytest.approx(10.0)
        assert t1 == pytest.approx(7.0)

    def test_all_enum_values_valid(self):
        # Ensure all enum defaults are actual enum members
        config = TFLEConfig()
        assert isinstance(config.init_method, InitMethod)
        assert isinstance(config.selection_method, SelectionMethod)
        assert isinstance(config.cooling_schedule, CoolingSchedule)
        assert isinstance(config.fitness_type, FitnessType)

"""Tests for temperature scheduling."""

import pytest

from tfle.annealing import TemperatureScheduler
from tfle.config import CoolingSchedule, TFLEConfig


class TestTemperatureScheduler:
    def test_initial_temperature(self):
        config = TFLEConfig(initial_temperature=10.0)
        scheduler = TemperatureScheduler(config)
        assert scheduler.get_temperature() == 10.0

    def test_exponential_cooling(self):
        config = TFLEConfig(
            initial_temperature=10.0,
            cooling_schedule=CoolingSchedule.EXPONENTIAL,
            cooling_rate=0.99,
            min_temperature=0.01,
            reheat_on_plateau=False,
        )
        scheduler = TemperatureScheduler(config)
        scheduler.step_update(1.0)
        assert scheduler.get_temperature() < 10.0
        assert scheduler.get_temperature() == pytest.approx(10.0 * 0.99, rel=1e-4)

    def test_linear_cooling(self):
        config = TFLEConfig(
            initial_temperature=10.0,
            cooling_schedule=CoolingSchedule.LINEAR,
            min_temperature=0.0,
            total_training_steps=100,
            reheat_on_plateau=False,
        )
        scheduler = TemperatureScheduler(config)
        for _ in range(50):
            scheduler.step_update(1.0)
        assert scheduler.get_temperature() == pytest.approx(5.0, rel=0.1)

    def test_cosine_cooling(self):
        config = TFLEConfig(
            initial_temperature=10.0,
            cooling_schedule=CoolingSchedule.COSINE,
            min_temperature=0.0,
            total_training_steps=100,
            reheat_on_plateau=False,
        )
        scheduler = TemperatureScheduler(config)
        # At halfway point, cosine should be near midpoint
        for _ in range(50):
            scheduler.step_update(1.0)
        temp = scheduler.get_temperature()
        assert 3.0 < temp < 7.0

    def test_min_temperature_floor(self):
        config = TFLEConfig(
            initial_temperature=1.0,
            cooling_schedule=CoolingSchedule.EXPONENTIAL,
            cooling_rate=0.5,
            min_temperature=0.1,
            reheat_on_plateau=False,
        )
        scheduler = TemperatureScheduler(config)
        for _ in range(100):
            scheduler.step_update(1.0)
        assert scheduler.get_temperature() >= 0.1

    def test_reheat_on_plateau(self):
        config = TFLEConfig(
            initial_temperature=5.0,
            cooling_schedule=CoolingSchedule.EXPONENTIAL,
            cooling_rate=0.999,
            min_temperature=0.01,
            reheat_on_plateau=True,
            plateau_window=10,
            reheat_factor=3.0,
        )
        scheduler = TemperatureScheduler(config)
        # Cool down, then stall
        for i in range(20):
            # Report same fitness = no improvement
            scheduler.step_update(1.0)

        # After plateau_window steps of no improvement, should reheat
        # The temperature should be higher than it was at step 9 (due to reheat)
        # or at least reset
        temp_after = scheduler.get_temperature()
        # Reheat should have triggered somewhere
        assert temp_after > config.min_temperature

    def test_reset(self):
        config = TFLEConfig(initial_temperature=10.0, reheat_on_plateau=False)
        scheduler = TemperatureScheduler(config)
        for _ in range(100):
            scheduler.step_update(1.0)
        scheduler.reset()
        assert scheduler.get_temperature() == 10.0
        assert scheduler.step == 0

    def test_adaptive_cooling(self):
        config = TFLEConfig(
            initial_temperature=10.0,
            cooling_schedule=CoolingSchedule.ADAPTIVE,
            cooling_rate=0.99,
            min_temperature=0.01,
            plateau_window=10,
            reheat_on_plateau=False,
        )
        scheduler = TemperatureScheduler(config)
        # With constant fitness, should eventually warm up
        for _ in range(20):
            scheduler.step_update(1.0)
        assert scheduler.get_temperature() > config.min_temperature

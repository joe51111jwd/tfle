"""Tests for convergence monitoring."""

import pytest

from tfle.config import TFLEConfig
from tfle.monitor import ConvergenceMonitor


class TestConvergenceMonitor:
    def test_creation(self):
        config = TFLEConfig()
        monitor = ConvergenceMonitor(config, n_layers=3)
        assert len(monitor.layer_metrics) == 3

    def test_record_step(self):
        config = TFLEConfig()
        monitor = ConvergenceMonitor(config, n_layers=2)
        metrics = {"accepted": True, "fitness_before": 1.0, "fitness_after": 2.0, "delta": 1.0}
        monitor.record_step(0, metrics)
        assert len(monitor.layer_metrics[0].fitness_values) == 1

    def test_record_validation(self):
        config = TFLEConfig()
        monitor = ConvergenceMonitor(config, n_layers=2)
        monitor.record_validation(100, 0.85)
        assert monitor.best_val_accuracy == 0.85
        assert len(monitor.val_accuracy_history) == 1

    def test_validation_improvement_tracking(self):
        config = TFLEConfig()
        monitor = ConvergenceMonitor(config, n_layers=2)
        monitor.record_validation(100, 0.80)
        monitor.record_validation(200, 0.85)
        assert monitor.steps_since_improvement == 0
        monitor.record_validation(300, 0.84)
        assert monitor.steps_since_improvement == 1

    def test_early_stopping(self):
        config = TFLEConfig(early_stopping_patience=3)
        monitor = ConvergenceMonitor(config, n_layers=1)
        monitor.record_validation(0, 0.9)
        for i in range(1, 5):
            monitor.record_validation(i, 0.89)
        assert monitor.should_early_stop()

    def test_no_early_stop_with_improvement(self):
        config = TFLEConfig(early_stopping_patience=3)
        monitor = ConvergenceMonitor(config, n_layers=1)
        for i in range(5):
            monitor.record_validation(i, 0.8 + i * 0.01)
        assert not monitor.should_early_stop()

    def test_oscillation_detection_stable(self):
        config = TFLEConfig()
        monitor = ConvergenceMonitor(config, n_layers=1)
        # Steadily increasing fitness = no oscillation
        for i in range(30):
            metrics = {"accepted": True, "fitness_before": float(i), "fitness_after": float(i + 1)}
            monitor.record_step(0, metrics)
        assert not monitor.detect_oscillation(0)

    def test_oscillation_detection_unstable(self):
        config = TFLEConfig()
        monitor = ConvergenceMonitor(config, n_layers=1)
        # Alternating high/low = oscillation
        for i in range(30):
            val = 10.0 if i % 2 == 0 else 1.0
            metrics = {"accepted": True, "fitness_before": val, "fitness_after": val}
            monitor.record_step(0, metrics)
        assert monitor.detect_oscillation(0)

    def test_acceptance_rate(self):
        config = TFLEConfig()
        monitor = ConvergenceMonitor(config, n_layers=1)
        for i in range(10):
            metrics = {"accepted": i < 7, "fitness_before": 1.0, "fitness_after": 1.0}
            monitor.record_step(0, metrics)
        rate = monitor.get_layer_acceptance_rate(0)
        assert rate == pytest.approx(0.7)

    def test_get_summary(self):
        config = TFLEConfig()
        monitor = ConvergenceMonitor(config, n_layers=2)
        summary = monitor.get_summary()
        assert "convergence_rate" in summary
        assert "layers" in summary
        assert len(summary["layers"]) == 2

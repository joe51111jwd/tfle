"""Tests for analysis and visualization tools."""

import tempfile
from pathlib import Path

from tfle.analysis import (
    generate_summary_report,
    plot_acceptance_rate,
    plot_memory_comparison,
    plot_training_curves,
    plot_weight_distribution_over_time,
)


class TestPlotTrainingCurves:
    def test_generates_figure(self):
        curves = {
            "method_a": [(0, 0.1), (100, 0.5), (200, 0.8)],
            "method_b": [(0, 0.2), (100, 0.6), (200, 0.9)],
        }
        fig = plot_training_curves(curves)
        assert fig is not None

    def test_saves_to_file(self):
        curves = {"test": [(0, 0.5), (100, 0.7)]}
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        plot_training_curves(curves, save_path=path)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_dict_format(self):
        curves = {
            "test": [{"step": 0, "accuracy": 0.1}, {"step": 100, "accuracy": 0.5}],
        }
        fig = plot_training_curves(curves)
        assert fig is not None


class TestPlotMemoryComparison:
    def test_generates_figure(self):
        mem_data = {
            "n_params": 100000,
            "tfle": {
                "weight_bytes": 100000, "trace_bytes": 200000,
                "total_bytes": 300000, "total_mb": 0.3,
            },
            "ste_backprop": {
                "weight_bytes": 400000, "gradient_bytes": 400000,
                "optimizer_bytes": 800000, "total_bytes": 1600000, "total_mb": 1.6,
            },
        }
        fig = plot_memory_comparison(mem_data)
        assert fig is not None


class TestPlotWeightDistribution:
    def test_generates_figure(self):
        dists = [
            {"neg1": 0.3, "zero": 0.4, "pos1": 0.3},
            {"neg1": 0.25, "zero": 0.5, "pos1": 0.25},
            {"neg1": 0.2, "zero": 0.6, "pos1": 0.2},
        ]
        fig = plot_weight_distribution_over_time(dists)
        assert fig is not None

    def test_empty_returns_none(self):
        fig = plot_weight_distribution_over_time([])
        assert fig is None


class TestPlotAcceptanceRate:
    def test_generates_figure(self):
        rates = [0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
        fig = plot_acceptance_rate(rates)
        assert fig is not None

    def test_empty_returns_none(self):
        fig = plot_acceptance_rate([])
        assert fig is None


class TestSummaryReport:
    def test_generates_report(self):
        results = {
            "tfle": {"final_accuracy": 0.85, "training_time": 100.0, "memory_mb": 2.5},
            "baseline": {"final_accuracy": 0.95, "training_time": 50.0, "memory_mb": 8.0},
        }
        report = generate_summary_report(results)
        assert "TFLE" in report
        assert "0.85" in report
        assert "Memory" in report

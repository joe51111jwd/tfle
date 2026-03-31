"""Tests for benchmarking utilities."""

import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from tfle.benchmarks import (
    BenchmarkRun,
    BenchmarkSuite,
    benchmark_memory_comparison,
    get_hardware_info,
)


def make_loaders(n=100, features=32, classes=5, batch_size=16):
    x = torch.randn(n, features)
    y = torch.randint(0, classes, (n,))
    ds = TensorDataset(x, y)
    train = DataLoader(ds, batch_size=batch_size, shuffle=True)
    val = DataLoader(ds, batch_size=batch_size)
    return train, val


class TestBenchmarkSuite:
    def test_add_and_save_load(self):
        suite = BenchmarkSuite()
        run = BenchmarkRun(
            name="test_run",
            config={"flip_rate": 0.03},
            result={"accuracy": 0.85},
            hardware_info={"platform": "test"},
            timestamp="2026-03-30",
        )
        suite.add_run(run)
        assert len(suite.runs) == 1

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        suite.save(path)

        loaded = BenchmarkSuite.load(path)
        assert len(loaded.runs) == 1
        assert loaded.runs[0].name == "test_run"
        assert loaded.runs[0].result["accuracy"] == 0.85

    def test_multiple_runs(self):
        suite = BenchmarkSuite()
        for i in range(5):
            suite.add_run(BenchmarkRun(name=f"run_{i}", result={"accuracy": i * 0.1}))
        assert len(suite.runs) == 5


class TestHardwareInfo:
    def test_returns_dict(self):
        info = get_hardware_info()
        assert isinstance(info, dict)
        assert "platform" in info
        assert "torch_version" in info
        assert "python_version" in info


class TestMemoryComparison:
    def test_basic_comparison(self):
        result = benchmark_memory_comparison([784, 512, 256, 10])
        assert "tfle" in result
        assert "ste_backprop" in result
        assert "memory_ratio" in result
        assert result["memory_ratio"] > 1  # STE should use more memory
        assert result["tfle"]["total_mb"] < result["ste_backprop"]["total_mb"]

    def test_scaling(self):
        small = benchmark_memory_comparison([64, 32, 10])
        large = benchmark_memory_comparison([784, 512, 256, 10])
        assert large["tfle"]["total_mb"] > small["tfle"]["total_mb"]

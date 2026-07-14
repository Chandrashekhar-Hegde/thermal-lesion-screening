import pytest

from src.benchmark_edge import benchmark


def test_benchmark_requires_an_existing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        benchmark(str(tmp_path / "missing.onnx"))


def test_benchmark_validates_run_configuration_before_loading_runtime(tmp_path):
    model = tmp_path / "model.onnx"
    model.touch()

    with pytest.raises(ValueError, match="runs and threads must be positive"):
        benchmark(str(model), runs=0)

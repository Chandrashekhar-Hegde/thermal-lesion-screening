import importlib
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

torch = pytest.importorskip("torch")
nn = torch.nn
train = importlib.import_module("src.train")
refit = importlib.import_module("src.refit")
benchmark = importlib.import_module("src.benchmark_edge").benchmark


def _tiny_model(_backbone, pretrained=True):
    del pretrained
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 1))


def _write_dataset(root: Path) -> None:
    for label, class_name in ((0, "healthy"), (1, "sick")):
        for index in range(8):
            patient = root / class_name / f"{class_name[0].upper()}{index:02d}"
            patient.mkdir(parents=True)
            rng = np.random.default_rng(label * 100 + index)
            pixels = rng.integers(0, 128, size=(16, 16, 3), dtype=np.uint8)
            pixels[:, :, label] += 100
            Image.fromarray(pixels).save(patient / "view.png")


def _write_config(path: Path, data_root: Path, artifact_root: Path) -> None:
    config = {
        "data": {
            "root": str(data_root),
            "classes": {"healthy": 0, "sick": 1},
            "image_size": [16, 16],
        },
        "split": {
            "strategy": "repeated_patient_cv",
            "n_folds": 2,
            "n_repeats": 1,
            "inner_val_frac": 0.25,
            "seed": 7,
        },
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "num_workers": 0,
            "lr": 0.001,
            "weight_decay": 0.0,
            "backbone": "smoke-test",
            "pretrained": False,
            "early_stopping_patience": 1,
            "save_fold_models": False,
        },
        "eval": {
            "target_sensitivity": 0.9,
            "n_bootstrap": 10,
            "checkpoint_reducer": "mean",
            "aggregation_candidates": [
                {"name": "mean"},
                {"name": "max"},
                {"name": "top_k_mean", "top_k": 3},
            ],
        },
        "edge": {
            "onnx_path": str(artifact_root / "model.onnx"),
            "benchmark_runs": 2,
        },
        "artifacts": {"root": str(artifact_root)},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_evaluation_refit_export_and_inference(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    artifact_root = tmp_path / "artifacts"
    config_path = tmp_path / "config.yaml"
    _write_dataset(data_root)
    _write_config(config_path, data_root, artifact_root)
    monkeypatch.setattr(train, "build_model", _tiny_model)
    monkeypatch.setattr(refit, "build_model", _tiny_model)

    report = train.run(str(config_path))
    metadata = refit.run(str(config_path), str(artifact_root / "cv_report.json"))
    timing = benchmark(
        str(artifact_root / "model.onnx"),
        runs=2,
        warmup=0,
        threads=1,
    )

    assert report["status"] == "completed"
    assert report["primary_nested_selection"]["n_patients"] == 16
    assert metadata["status"] == "refit on all clean patients; not independently evaluated"
    assert timing["runs"] == 2
    assert (artifact_root / "RESULTS.md").is_file()
    assert len(list((artifact_root / "figures").glob("*.png"))) == 3

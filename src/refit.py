"""Refit one deployment model after evaluation without making a new metric claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.dicom_io import manifest_from_folders
from src.integrity import audit_manifest
from src.model import build_model, export_onnx
from src.train import (
    ThermalDataset,
    _training_weights,
    evaluation_configuration,
    seed_everything,
    seed_worker,
    transforms_for,
)


def _load_inputs(config_path: str, cv_report_path: str):
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with Path(cv_report_path).open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "completed":
        raise ValueError("a completed CV report is required before refitting")
    return config, report


def run(config_path: str, cv_report_path: str) -> dict:
    config, report = _load_inputs(config_path, cv_report_path)
    if report.get("configuration") != evaluation_configuration(config):
        raise ValueError("configuration does not match the completed CV report")
    raw_manifest = manifest_from_folders(
        config["data"]["root"],
        config["data"]["classes"],
        validate_patient_labels=False,
    )
    integrity = audit_manifest(raw_manifest)
    manifest = integrity.manifest
    expected_patients = int(report["integrity"]["after"]["patients"])
    expected_fingerprint = report["integrity"]["clean_manifest_sha256"]
    if (
        manifest["patient_id"].nunique() != expected_patients
        or integrity.report["clean_manifest_sha256"] != expected_fingerprint
    ):
        raise ValueError("current clean dataset does not match the completed CV report")

    best_epochs = [int(fold["best_epoch"]) for fold in report["cross_validation"]["folds"]]
    epochs = max(1, int(round(float(np.median(best_epochs)))))
    seed = int(config["split"]["seed"])
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    indices = np.arange(len(manifest))
    weights = _training_weights(manifest, indices)
    dataset = ThermalDataset(
        manifest,
        indices,
        transforms_for("train", tuple(config["data"]["image_size"])),
        weights,
    )
    num_workers = int(config["train"].get("num_workers", 2))
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        pin_memory=device == "cuda",
    )

    model = build_model(
        config["train"]["backbone"],
        pretrained=bool(config["train"].get("pretrained", True)),
    ).to(device)
    patient_labels = manifest[["patient_id", "label"]].drop_duplicates()
    n_positive = int(patient_labels["label"].sum())
    n_negative = len(patient_labels) - n_positive
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_negative / n_positive], device=device),
        reduction="none",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    for _ in range(epochs):
        model.train()
        for images, labels, sample_weights in loader:
            optimizer.zero_grad()
            losses = criterion(
                model(images.to(device)).squeeze(1),
                labels.to(device),
            )
            batch_weights = sample_weights.to(device)
            loss = (losses * batch_weights).sum() / batch_weights.sum()
            loss.backward()
            optimizer.step()

    artifact_dir = Path(config.get("artifacts", {}).get("root", "artifacts"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    onnx_path = export_onnx(
        model,
        config["edge"]["onnx_path"],
        tuple(config["data"]["image_size"]),
    )
    metadata = {
        "status": "refit on all clean patients; not independently evaluated",
        "patients": int(patient_labels["patient_id"].nunique()),
        "images": int(len(manifest)),
        "epochs": epochs,
        "epoch_rule": "rounded median best epoch across outer CV folds",
        "checkpoint": str(model_path),
        "onnx": str(onnx_path),
    }
    with (artifact_dir / "refit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cv-report", default="artifacts/cv_report.json")
    args = parser.parse_args()
    result = run(args.config, args.cv_report)
    print(json.dumps(result, indent=2, sort_keys=True))

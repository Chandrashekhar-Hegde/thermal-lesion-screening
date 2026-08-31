"""Repeated patient-level nested cross-validation for thermal screening."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.dicom_io import manifest_from_folders
from src.integrity import audit_manifest, write_integrity_artifacts
from src.metrics import (
    aggregate_by_patient,
    repeated_cv_report,
    select_aggregation,
)
from src.model import build_model
from src.reporting import write_evaluation_figures, write_results_markdown
from src.splits import RepeatedFold, repeated_patient_level_cv

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class ThermalDataset(Dataset):
    """Image rows with optional inverse patient-frequency weights."""

    def __init__(self, manifest, indices, transform, sample_weights=None):
        self.rows = manifest.iloc[indices].reset_index(drop=True)
        self.transform = transform
        if sample_weights is None:
            sample_weights = np.ones(len(self.rows), dtype=np.float32)
        self.sample_weights = np.asarray(sample_weights, dtype=np.float32)
        if len(self.sample_weights) != len(self.rows):
            raise ValueError("sample_weights must align with dataset rows")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row["path"]) as source:
            image = source.convert("RGB")
        return (
            self.transform(image),
            torch.tensor(float(row["label"])),
            torch.tensor(self.sample_weights[index]),
        )


def transforms_for(split: str, size):
    normalization = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=7, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                normalization,
            ]
        )
    return transforms.Compose([transforms.Resize(size), transforms.ToTensor(), normalization])


def seed_everything(seed: int) -> None:
    """Seed training and request deterministic kernels where available."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _training_weights(manifest: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Give each patient equal total loss weight regardless of image count."""
    patients = manifest.iloc[indices]["patient_id"]
    counts = patients.value_counts()
    weights = patients.map(lambda patient_id: 1.0 / counts[patient_id]).to_numpy()
    return (weights / weights.mean()).astype(np.float32)


def evaluation_configuration(config: dict) -> dict:
    """Return the path-free configuration that defines an evaluation run."""
    return {
        "data": {
            "classes": config["data"]["classes"],
            "image_size": config["data"]["image_size"],
        },
        "split": config["split"],
        "train": config["train"],
        "eval": config["eval"],
    }


def _make_loaders(manifest, split, config, device, seed):
    image_size = tuple(config["data"]["image_size"])
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"].get("num_workers", 2))
    options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device == "cuda",
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    return {
        "train": DataLoader(
            ThermalDataset(
                manifest,
                split.train,
                transforms_for("train", image_size),
                _training_weights(manifest, split.train),
            ),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            **options,
        ),
        "val": DataLoader(
            ThermalDataset(
                manifest,
                split.val,
                transforms_for("val", image_size),
            ),
            **options,
        ),
        "test": DataLoader(
            ThermalDataset(
                manifest,
                split.test,
                transforms_for("test", image_size),
            ),
            **options,
        ),
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for images, batch_labels, _ in loader:
        logits = model(images.to(device)).squeeze(1)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def _candidate_key(candidate: dict) -> str:
    if candidate["name"] == "top_k_mean":
        return f"top_k_mean_{candidate['top_k']}"
    return candidate["name"]


def _aggregate_with_candidate(labels, scores, patients, candidate):
    return aggregate_by_patient(
        labels,
        scores,
        patients,
        reducer=candidate["name"],
        top_k=candidate.get("top_k"),
    )


def _prediction_rows(
    repeated_fold: RepeatedFold,
    labels: np.ndarray,
    scores: np.ndarray,
    patients: np.ndarray,
    candidate: dict,
    threshold: float,
) -> list[dict]:
    return [
        {
            "repeat": repeated_fold.repeat,
            "fold": repeated_fold.fold,
            "patient_id": patient_id,
            "label": int(label),
            "score": float(score),
            "prediction": int(score >= threshold),
            "reducer": candidate["name"],
            "top_k": candidate.get("top_k"),
            "threshold": float(threshold),
        }
        for label, score, patient_id in zip(labels, scores, patients, strict=True)
    ]


def train_fold(
    manifest: pd.DataFrame,
    repeated_fold: RepeatedFold,
    config: dict,
    device: str,
    artifact_dir: Path,
) -> tuple[list[dict], list[dict], dict]:
    """Train one inner model and evaluate one untouched outer test fold."""
    base_seed = int(config["split"]["seed"])
    fold_seed = base_seed + 10_000 * repeated_fold.repeat + repeated_fold.fold
    seed_everything(fold_seed)
    split = repeated_fold.split
    loaders = _make_loaders(manifest, split, config, device, fold_seed)

    model = build_model(
        config["train"]["backbone"],
        pretrained=bool(config["train"].get("pretrained", True)),
    ).to(device)
    train_patients = manifest.iloc[split.train][["patient_id", "label"]].drop_duplicates()
    n_positive = int(train_patients["label"].sum())
    n_negative = len(train_patients) - n_positive
    positive_weight = torch.tensor([n_negative / n_positive], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight, reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    epochs = int(config["train"]["epochs"])
    patience_limit = int(config["train"]["early_stopping_patience"])
    checkpoint_reducer = config["eval"].get("checkpoint_reducer", "mean")
    validation_patients = manifest.iloc[split.val]["patient_id"].to_numpy()
    best_auc, stale_epochs, best_epoch, best_state = -1.0, 0, -1, None
    for epoch in range(epochs):
        model.train()
        for images, labels, sample_weights in loaders["train"]:
            optimizer.zero_grad()
            losses = criterion(
                model(images.to(device)).squeeze(1),
                labels.to(device),
            )
            weights = sample_weights.to(device)
            loss = (losses * weights).sum() / weights.sum()
            loss.backward()
            optimizer.step()

        validation_labels, validation_scores = predict(model, loaders["val"], device)
        validation_labels, validation_scores, _ = aggregate_by_patient(
            validation_labels,
            validation_scores,
            validation_patients,
            reducer=checkpoint_reducer,
        )
        validation_auc = roc_auc_score(validation_labels, validation_scores)
        log.info(
            "repeat %d fold %d epoch %d  patient-level val AUROC %.4f",
            repeated_fold.repeat + 1,
            repeated_fold.fold + 1,
            epoch + 1,
            validation_auc,
        )
        if validation_auc > best_auc:
            best_auc = validation_auc
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience_limit:
                break

    if best_state is None:
        raise RuntimeError("training finished without a model checkpoint")
    model.load_state_dict(best_state)
    if config["train"].get("save_fold_models", False):
        checkpoint = (
            artifact_dir
            / "checkpoints"
            / f"repeat_{repeated_fold.repeat + 1}"
            / f"fold_{repeated_fold.fold + 1}.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint)

    validation_image_labels, validation_image_scores = predict(model, loaders["val"], device)
    candidates = config["eval"]["aggregation_candidates"]
    target_sensitivity = float(config["eval"]["target_sensitivity"])
    selected, validation_comparison = select_aggregation(
        validation_image_labels,
        validation_image_scores,
        validation_patients,
        candidates,
        target_sensitivity,
    )

    test_image_labels, test_image_scores = predict(model, loaders["test"], device)
    test_patients = manifest.iloc[split.test]["patient_id"].to_numpy()
    selected_candidate = {"name": selected.name, "top_k": selected.top_k}
    labels, scores, patients = _aggregate_with_candidate(
        test_image_labels,
        test_image_scores,
        test_patients,
        selected_candidate,
    )
    primary_rows = _prediction_rows(
        repeated_fold,
        labels,
        scores,
        patients,
        selected_candidate,
        selected.threshold,
    )

    comparison_by_key = {
        _candidate_key(candidate): candidate for candidate in validation_comparison
    }
    exploratory_rows = []
    for candidate in candidates:
        candidate_result = comparison_by_key[_candidate_key(candidate)]
        labels, scores, patients = _aggregate_with_candidate(
            test_image_labels,
            test_image_scores,
            test_patients,
            candidate,
        )
        rows = _prediction_rows(
            repeated_fold,
            labels,
            scores,
            patients,
            candidate,
            candidate_result["threshold"],
        )
        for row in rows:
            row["candidate_key"] = _candidate_key(candidate)
        exploratory_rows.extend(rows)

    fold_report = {
        "repeat": repeated_fold.repeat,
        "fold": repeated_fold.fold,
        "seed": fold_seed,
        "patients": {
            "train": int(train_patients["patient_id"].nunique()),
            "validation": int(manifest.iloc[split.val]["patient_id"].nunique()),
            "test": int(manifest.iloc[split.test]["patient_id"].nunique()),
        },
        "best_epoch": best_epoch,
        "validation_auroc_at_selection": round(float(best_auc), 4),
        "selected_aggregation": selected.as_dict(),
        "validation_aggregation_comparison": validation_comparison,
    }
    return primary_rows, exploratory_rows, fold_report


def run(config_path: str) -> dict:
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if config["split"].get("strategy") != "repeated_patient_cv":
        raise ValueError("split.strategy must be repeated_patient_cv")
    artifact_dir = Path(config.get("artifacts", {}).get("root", "artifacts"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_manifest = manifest_from_folders(
        config["data"]["root"],
        config["data"]["classes"],
        validate_patient_labels=False,
    )
    integrity = audit_manifest(raw_manifest)
    write_integrity_artifacts(integrity, artifact_dir)
    manifest = integrity.manifest
    log.info(
        "integrity pass: %d -> %d patients, %d -> %d images",
        integrity.report["before"]["patients"],
        integrity.report["after"]["patients"],
        integrity.report["before"]["images"],
        integrity.report["after"]["images"],
    )

    repeated_folds = repeated_patient_level_cv(
        manifest,
        n_folds=int(config["split"]["n_folds"]),
        n_repeats=int(config["split"]["n_repeats"]),
        val_frac=float(config["split"]["inner_val_frac"]),
        seed=int(config["split"]["seed"]),
    )
    primary_rows, exploratory_rows, fold_reports = [], [], []
    for repeated_fold in repeated_folds:
        primary, exploratory, fold_report = train_fold(
            manifest,
            repeated_fold,
            config,
            device,
            artifact_dir,
        )
        primary_rows.extend(primary)
        exploratory_rows.extend(exploratory)
        fold_reports.append(fold_report)

    primary_predictions = pd.DataFrame(primary_rows)
    exploratory_predictions = pd.DataFrame(exploratory_rows)
    expected = int(config["split"]["n_repeats"]) * manifest["patient_id"].nunique()
    if len(primary_predictions) != expected:
        raise RuntimeError("outer-fold predictions are incomplete")
    if primary_predictions.duplicated(["repeat", "patient_id"]).any():
        raise RuntimeError("a patient has multiple predictions within one repeat")

    n_bootstrap = int(config["eval"]["n_bootstrap"])
    bootstrap_seed = int(config["split"]["seed"])
    primary_report = repeated_cv_report(
        primary_predictions,
        n_boot=n_bootstrap,
        seed=bootstrap_seed,
    )
    reducer_reports = {}
    for candidate_key, candidate_frame in exploratory_predictions.groupby("candidate_key"):
        reducer_reports[candidate_key] = repeated_cv_report(
            candidate_frame,
            n_boot=n_bootstrap,
            seed=bootstrap_seed,
        )

    report = {
        "status": "completed",
        "unit_of_analysis": "patient",
        "configuration": evaluation_configuration(config),
        "target_sensitivity": float(config["eval"]["target_sensitivity"]),
        "integrity": integrity.report,
        "cross_validation": {
            "n_folds": int(config["split"]["n_folds"]),
            "n_repeats": int(config["split"]["n_repeats"]),
            "inner_validation_fraction": float(config["split"]["inner_val_frac"]),
            "folds": fold_reports,
        },
        "primary_nested_selection": primary_report,
        "fixed_reducer_comparison": reducer_reports,
        "comparison_note": (
            "Fixed reducers are paired exploratory estimates. The primary estimate "
            "selects the reducer using validation patients inside each outer fold."
        ),
    }
    primary_predictions.to_csv(artifact_dir / "cv_predictions.csv", index=False)
    exploratory_predictions.to_csv(artifact_dir / "cv_reducer_predictions.csv", index=False)
    with (artifact_dir / "cv_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_evaluation_figures(
        primary_predictions,
        artifact_dir / "figures",
        seed=bootstrap_seed,
        n_boot=n_bootstrap,
        target_sensitivity=float(config["eval"]["target_sensitivity"]),
    )
    write_results_markdown(report, artifact_dir / "RESULTS.md")
    log.info("wrote repeated-CV report and figures to %s", artifact_dir)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    run(parser.parse_args().config)

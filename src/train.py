"""End-to-end training with patient-level model selection and evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.dicom_io import manifest_from_folders
from src.metrics import (
    aggregate_by_patient,
    format_report,
    full_report,
    threshold_at_sensitivity,
)
from src.model import build_model, export_onnx
from src.splits import describe_split, patient_level_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class ThermalDataset(Dataset):
    def __init__(self, manifest, indices, transform):
        self.rows = manifest.iloc[indices].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row["path"]) as source:
            image = source.convert("RGB")
        return self.transform(image), torch.tensor(float(row["label"]))


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


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for images, batch_labels in loader:
        logits = model(images.to(device)).squeeze(1)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def main(config_path: str) -> None:
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    seed = int(config["split"]["seed"])
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)

    manifest = manifest_from_folders(
        config["data"]["root"],
        config["data"]["classes"],
    )
    split = patient_level_split(
        manifest,
        test_frac=config["split"]["test_frac"],
        val_frac=config["split"]["val_frac"],
        seed=seed,
    )
    summary = describe_split(manifest, split)
    log.info("split summary:\n%s", summary.to_string(index=False))
    summary.to_csv(artifact_dir / "split_summary.csv", index=False)

    image_size = tuple(config["data"]["image_size"])
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"].get("num_workers", 2))
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device == "cuda",
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    loaders = {
        "train": DataLoader(
            ThermalDataset(
                manifest,
                split.train,
                transforms_for("train", image_size),
            ),
            shuffle=True,
            generator=generator,
            **loader_options,
        ),
        "val": DataLoader(
            ThermalDataset(manifest, split.val, transforms_for("val", image_size)),
            **loader_options,
        ),
        "test": DataLoader(
            ThermalDataset(manifest, split.test, transforms_for("test", image_size)),
            **loader_options,
        ),
    }

    model = build_model(
        config["train"]["backbone"],
        pretrained=bool(config["train"].get("pretrained", True)),
    ).to(device)

    n_positive = int(manifest.iloc[split.train]["label"].sum())
    n_negative = len(split.train) - n_positive
    positive_weight = torch.tensor([n_negative / max(n_positive, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    epochs = int(config["train"]["epochs"])
    if epochs <= 0:
        raise ValueError("train.epochs must be positive")

    reducer = config["eval"].get("patient_score_reducer", "mean")
    validation_patient_ids = manifest.iloc[split.val]["patient_id"].to_numpy()
    best_auc, stale_epochs, best_epoch, best_state = -1.0, 0, -1, None
    for epoch in range(epochs):
        model.train()
        for images, labels in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(
                model(images.to(device)).squeeze(1),
                labels.to(device),
            )
            loss.backward()
            optimizer.step()

        validation_labels, validation_scores = predict(model, loaders["val"], device)
        validation_labels, validation_scores, _ = aggregate_by_patient(
            validation_labels,
            validation_scores,
            validation_patient_ids,
            reducer=reducer,
        )
        validation_auc = roc_auc_score(validation_labels, validation_scores)
        log.info("epoch %2d  patient-level val AUROC %.4f", epoch + 1, validation_auc)

        if validation_auc > best_auc:
            best_auc = validation_auc
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config["train"]["early_stopping_patience"]:
                log.info("early stop after epoch %d", epoch + 1)
                break

    if best_state is None:
        raise RuntimeError("training finished without a model checkpoint")
    model.load_state_dict(best_state)
    torch.save(best_state, artifact_dir / "model.pt")

    validation_labels, validation_scores = predict(model, loaders["val"], device)
    validation_labels, validation_scores, validation_patients = aggregate_by_patient(
        validation_labels,
        validation_scores,
        validation_patient_ids,
        reducer=reducer,
    )
    target_sensitivity = float(config["eval"]["target_sensitivity"])
    threshold = threshold_at_sensitivity(
        validation_labels,
        validation_scores,
        target_sensitivity,
    )

    test_image_labels, test_image_scores = predict(model, loaders["test"], device)
    test_labels, test_scores, test_patients = aggregate_by_patient(
        test_image_labels,
        test_image_scores,
        manifest.iloc[split.test]["patient_id"].to_numpy(),
        reducer=reducer,
    )
    report = full_report(
        test_labels,
        test_scores,
        threshold=threshold,
        target_sensitivity=target_sensitivity,
        n_boot=int(config["eval"]["n_bootstrap"]),
        sample_unit="patients",
    )
    report.update(
        {
            "best_epoch": best_epoch,
            "val_auroc_at_selection": round(float(best_auc), 4),
            "validation_patients": int(len(validation_patients)),
            "test_patients": int(len(test_patients)),
            "test_images": int(len(test_image_labels)),
            "patient_score_reducer": reducer,
            "threshold_selected_on": "validation",
            "seed": seed,
        }
    )

    print("\n" + "=" * 62)
    print("HELD-OUT TEST SET  (patient-level, fixed validation threshold)")
    print("=" * 62)
    print(format_report(report))

    with (artifact_dir / "test_report.json").open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.write("\n")

    onnx_path = export_onnx(model, config["edge"]["onnx_path"], image_size)
    log.info("wrote %s and %s", artifact_dir / "test_report.json", onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)

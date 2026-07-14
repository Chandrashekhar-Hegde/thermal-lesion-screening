"""
End-to-end training run.

Order of operations matters and is enforced here:

  1. Build the manifest (patient_id comes from DICOM header or directory level).
  2. Split by PATIENT, before a single image is loaded.
  3. Train, selecting on the validation fold only.
  4. Touch the test set exactly ONCE, at the end.

Step 4 is a discipline, not a suggestion. If you evaluate on test, adjust
something, and evaluate again, the test set has become a validation set and your
headline number is no longer an out-of-sample estimate. Re-split with a new seed
if you need to iterate.

Usage:
    python -m src.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.dicom_io import manifest_from_folders
from src.metrics import full_report, format_report
from src.model import build_model, export_onnx
from src.splits import patient_level_split, describe_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class ThermalDataset(Dataset):
    def __init__(self, manifest, indices, tfm):
        self.rows = manifest.iloc[indices].reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tfm(img), torch.tensor(float(r["label"]))


def transforms_for(split: str, size):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if split == "train":
        # Geometric only. Do NOT colour-jitter thermal imagery — the pixel
        # values ARE the temperatures. Randomising them destroys the signal.
        return transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=7, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([transforms.Resize(size), transforms.ToTensor(), norm])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device)).squeeze(1)
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path("artifacts").mkdir(exist_ok=True)

    # 1 & 2 — manifest, then split by patient BEFORE anything else touches data.
    man = manifest_from_folders(cfg["data"]["root"], cfg["data"]["classes"])
    sp = patient_level_split(
        man,
        test_frac=cfg["split"]["test_frac"],
        val_frac=cfg["split"]["val_frac"],
        seed=cfg["split"]["seed"],
    )
    summary = describe_split(man, sp)
    log.info("split summary:\n%s", summary.to_string(index=False))
    summary.to_csv("artifacts/split_summary.csv", index=False)

    size = tuple(cfg["data"]["image_size"])
    bs = cfg["train"]["batch_size"]
    loaders = {
        "train": DataLoader(ThermalDataset(man, sp.train, transforms_for("train", size)),
                            batch_size=bs, shuffle=True, num_workers=2),
        "val": DataLoader(ThermalDataset(man, sp.val, transforms_for("val", size)),
                          batch_size=bs, num_workers=2),
        "test": DataLoader(ThermalDataset(man, sp.test, transforms_for("test", size)),
                           batch_size=bs, num_workers=2),
    }

    model = build_model(cfg["train"]["backbone"]).to(device)

    # Class imbalance handled in the loss, not by resampling — resampling a
    # patient's images across the batch is another quiet route to leakage.
    n_pos = int(man.iloc[sp.train]["label"].sum())
    n_neg = len(sp.train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])

    # 3 — train, selecting on val only.
    best_auc, patience, best_state = -1.0, 0, None
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for x, y in loaders["train"]:
            opt.zero_grad()
            loss = crit(model(x.to(device)).squeeze(1), y.to(device))
            loss.backward()
            opt.step()

        yv, pv = predict(model, loaders["val"], device)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yv, pv)
        log.info("epoch %2d  val AUROC %.4f", epoch, auc)

        if auc > best_auc:
            best_auc, best_state, patience = auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= cfg["train"]["early_stopping_patience"]:
                log.info("early stop at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    torch.save(best_state, "artifacts/model.pt")

    # 4 — the test set, once.
    yt, pt = predict(model, loaders["test"], device)
    rep = full_report(yt, pt,
                      target_sensitivity=cfg["eval"]["target_sensitivity"],
                      n_boot=cfg["eval"]["n_bootstrap"])
    rep["val_auroc_at_selection"] = round(float(best_auc), 4)
    rep["test_patients"] = int(man.iloc[sp.test]["patient_id"].nunique())

    print("\n" + "=" * 62)
    print("HELD-OUT TEST SET  (patient-level split, evaluated once)")
    print("=" * 62)
    print(f"{rep['test_patients']} patients\n")
    print(format_report(rep))

    json.dump(rep, open("artifacts/test_report.json", "w"), indent=2)
    export_onnx(model, cfg["edge"]["onnx_path"], size)
    log.info("wrote artifacts/test_report.json and %s", cfg["edge"]["onnx_path"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    main(ap.parse_args().config)

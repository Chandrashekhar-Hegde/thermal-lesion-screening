"""
Patient-level data splitting.

This is the most important module in the repository.

The most common — and most damaging — error in medical imaging ML is splitting
data at the *image* or *lesion* level. If two thermograms of the same patient
land on opposite sides of a train/test boundary, the model can memorise that
patient rather than learn the pathology. Reported sensitivity goes up. Real
clinical performance does not. The number becomes meaningless.

Every split produced here is grouped by patient_id and stratified by label, and
`assert_no_patient_leakage` is called before any split is returned. There is no
code path in this repository that can produce a patient-leaking split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class PatientLeakageError(AssertionError):
    """Raised when the same patient appears in more than one split partition."""


@dataclass(frozen=True)
class Split:
    """A single train/val/test partition, indexed into the source manifest."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def assert_no_patient_leakage(
    manifest: pd.DataFrame,
    split: Split,
    patient_col: str = "patient_id",
) -> None:
    """Hard-fail if any patient appears in more than one partition.

    Called automatically by every split function in this module. Also exercised
    directly by tests/test_splits.py.
    """
    parts = {
        "train": set(manifest.iloc[split.train][patient_col]),
        "val": set(manifest.iloc[split.val][patient_col]),
        "test": set(manifest.iloc[split.test][patient_col]),
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = parts[a] & parts[b]
        if overlap:
            raise PatientLeakageError(
                f"{len(overlap)} patient(s) appear in both '{a}' and '{b}': "
                f"{sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}. "
                "This split is invalid and any metric computed from it is meaningless."
            )


def patient_level_split(
    manifest: pd.DataFrame,
    label_col: str = "label",
    patient_col: str = "patient_id",
    test_frac: float = 0.2,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Split:
    """Stratified, patient-grouped train/val/test split.

    Stratification keeps the positive rate comparable across partitions;
    grouping guarantees a patient's images never straddle a boundary. Because
    whole patients move together, realised fractions will not land exactly on
    the requested ones — that is expected and correct.
    """
    if not 0 < test_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("test_frac and val_frac must each lie strictly in (0, 1)")

    y = manifest[label_col].to_numpy()
    groups = manifest[patient_col].to_numpy()
    idx = np.arange(len(manifest))

    # Carve out the test set first.
    n_test_folds = max(2, round(1 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(sgkf.split(idx, y, groups))

    # Then split the remainder into train/val, again grouped by patient.
    rel_val = val_frac / (1.0 - test_frac)
    n_val_folds = max(2, round(1 / rel_val))
    sgkf_val = StratifiedGroupKFold(
        n_splits=n_val_folds, shuffle=True, random_state=seed + 1
    )
    sub_train, sub_val = next(
        sgkf_val.split(dev_idx, y[dev_idx], groups[dev_idx])
    )

    split = Split(
        train=dev_idx[sub_train],
        val=dev_idx[sub_val],
        test=test_idx,
    )
    assert_no_patient_leakage(manifest, split, patient_col)
    return split


def patient_level_cv(
    manifest: pd.DataFrame,
    n_folds: int = 5,
    label_col: str = "label",
    patient_col: str = "patient_id",
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Patient-grouped, stratified K-fold cross-validation.

    Use for model selection only. Report your headline number on the held-out
    test set from `patient_level_split`, never on cross-validation folds — a
    CV score is an estimate you tuned against, not an independent result.
    """
    y = manifest[label_col].to_numpy()
    groups = manifest[patient_col].to_numpy()
    idx = np.arange(len(manifest))

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_i, val_i in sgkf.split(idx, y, groups):
        if set(groups[train_i]) & set(groups[val_i]):
            raise PatientLeakageError("StratifiedGroupKFold produced a leaking fold")
        yield idx[train_i], idx[val_i]


def describe_split(manifest: pd.DataFrame, split: Split, label_col: str = "label",
                   patient_col: str = "patient_id") -> pd.DataFrame:
    """Summary table to paste straight into a README or a paper's methods section."""
    rows = []
    for name, ix in (("train", split.train), ("val", split.val), ("test", split.test)):
        sub = manifest.iloc[ix]
        rows.append(
            {
                "split": name,
                "images": len(sub),
                "patients": sub[patient_col].nunique(),
                "positive": int(sub[label_col].sum()),
                "positive_rate": round(float(sub[label_col].mean()), 3),
            }
        )
    return pd.DataFrame(rows)

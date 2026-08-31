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

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


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


@dataclass(frozen=True)
class RepeatedFold:
    """One outer test fold with an inner validation partition."""

    repeat: int
    fold: int
    split: Split


def _validate_manifest(
    manifest: pd.DataFrame,
    label_col: str,
    patient_col: str,
) -> None:
    required = {label_col, patient_col}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing required column(s): {sorted(missing)}")
    if manifest.empty:
        raise ValueError("manifest cannot be empty")
    if manifest[list(required)].isna().any().any():
        raise ValueError("labels and patient identifiers cannot be empty")

    classes = set(manifest[label_col].unique().tolist())
    if classes != {0, 1}:
        raise ValueError("manifest labels must contain both binary classes: 0 and 1")

    labels_per_patient = manifest.groupby(patient_col)[label_col].nunique()
    conflicts = labels_per_patient[labels_per_patient > 1]
    if not conflicts.empty:
        raise ValueError(f"patient(s) have conflicting labels: {conflicts.index.tolist()[:5]}")


def assert_no_patient_leakage(
    manifest: pd.DataFrame,
    split: Split,
    patient_col: str = "patient_id",
) -> None:
    """Hard-fail if any patient appears in more than one partition.

    Called automatically by every split function in this module. Also exercised
    directly by tests/test_splits.py.
    """
    allocated = np.concatenate([split.train, split.val, split.test])
    if (allocated < 0).any() or (allocated >= len(manifest)).any():
        raise PatientLeakageError("split contains an out-of-range image index")
    if len(allocated) != len(manifest) or len(np.unique(allocated)) != len(manifest):
        raise PatientLeakageError("each image must appear in exactly one split partition")

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
    _validate_manifest(manifest, label_col, patient_col)
    if not 0 < test_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("test_frac and val_frac must each lie strictly in (0, 1)")
    if test_frac + val_frac >= 1:
        raise ValueError("test_frac + val_frac must be less than 1")

    patient_table = manifest[[patient_col, label_col]].drop_duplicates()
    try:
        development, test = train_test_split(
            patient_table,
            test_size=test_frac,
            stratify=patient_table[label_col],
            random_state=seed,
        )
        relative_val_fraction = val_frac / (1.0 - test_frac)
        train, val = train_test_split(
            development,
            test_size=relative_val_fraction,
            stratify=development[label_col],
            random_state=seed + 1,
        )
    except ValueError as error:
        raise ValueError(
            "unable to create stratified patient splits; add more patients per class "
            "or adjust the requested fractions"
        ) from error

    split = Split(
        train=np.flatnonzero(manifest[patient_col].isin(train[patient_col])),
        val=np.flatnonzero(manifest[patient_col].isin(val[patient_col])),
        test=np.flatnonzero(manifest[patient_col].isin(test[patient_col])),
    )
    assert_no_patient_leakage(manifest, split, patient_col)
    for name, indices in (
        ("train", split.train),
        ("val", split.val),
        ("test", split.test),
    ):
        if set(manifest.iloc[indices][label_col].unique()) != {0, 1}:
            raise ValueError(f"{name} split does not contain both classes; use more patients")
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
    _validate_manifest(manifest, label_col, patient_col)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    patient_labels = manifest[[patient_col, label_col]].drop_duplicates()
    class_counts = patient_labels[label_col].value_counts()
    if int(class_counts.min()) < n_folds:
        raise ValueError("each class must have at least n_folds patients")

    patient_table = manifest[[patient_col, label_col]].drop_duplicates().reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_patient_indices, val_patient_indices in splitter.split(
        patient_table[patient_col], patient_table[label_col]
    ):
        train_patients = patient_table.iloc[train_patient_indices][patient_col]
        val_patients = patient_table.iloc[val_patient_indices][patient_col]
        train_i = np.flatnonzero(manifest[patient_col].isin(train_patients))
        val_i = np.flatnonzero(manifest[patient_col].isin(val_patients))
        if set(manifest.iloc[train_i][patient_col]) & set(manifest.iloc[val_i][patient_col]):
            raise PatientLeakageError("patient-level cross-validation produced a leaking fold")
        yield train_i, val_i


def repeated_patient_level_cv(
    manifest: pd.DataFrame,
    n_folds: int = 5,
    n_repeats: int = 5,
    val_frac: float = 0.20,
    label_col: str = "label",
    patient_col: str = "patient_id",
    seed: int = 42,
) -> Iterator[RepeatedFold]:
    """Yield repeated outer test folds with an inner validation split.

    Every patient is used in exactly one outer test fold per repeat. The inner
    validation patients are drawn only from that fold's development partition and
    are used for checkpoint and operating-threshold selection.
    """
    _validate_manifest(manifest, label_col, patient_col)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must lie strictly in (0, 1)")

    patient_table = manifest[[patient_col, label_col]].drop_duplicates().reset_index(drop=True)
    class_counts = patient_table[label_col].value_counts()
    if int(class_counts.min()) < n_folds:
        raise ValueError("each class must have at least n_folds patients")

    for repeat in range(n_repeats):
        outer = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed + repeat,
        )
        for fold, (development_indices, test_indices) in enumerate(
            outer.split(patient_table[patient_col], patient_table[label_col])
        ):
            development = patient_table.iloc[development_indices]
            test = patient_table.iloc[test_indices]
            try:
                train, val = train_test_split(
                    development,
                    test_size=val_frac,
                    stratify=development[label_col],
                    random_state=seed + 10_000 * repeat + fold,
                )
            except ValueError as error:
                raise ValueError(
                    "unable to create an inner stratified validation split; add more "
                    "patients per class or adjust val_frac"
                ) from error

            split = Split(
                train=np.flatnonzero(manifest[patient_col].isin(train[patient_col])),
                val=np.flatnonzero(manifest[patient_col].isin(val[patient_col])),
                test=np.flatnonzero(manifest[patient_col].isin(test[patient_col])),
            )
            assert_no_patient_leakage(manifest, split, patient_col)
            for name, indices in (
                ("train", split.train),
                ("val", split.val),
                ("test", split.test),
            ):
                if set(manifest.iloc[indices][label_col].unique()) != {0, 1}:
                    raise ValueError(
                        f"repeat {repeat}, fold {fold}: {name} does not contain both classes"
                    )
            yield RepeatedFold(repeat=repeat, fold=fold, split=split)


def describe_split(
    manifest: pd.DataFrame,
    split: Split,
    label_col: str = "label",
    patient_col: str = "patient_id",
) -> pd.DataFrame:
    """Summary table to paste straight into a README or a paper's methods section."""
    rows = []
    for name, ix in (("train", split.train), ("val", split.val), ("test", split.test)):
        sub = manifest.iloc[ix]
        patient_rows = sub[[patient_col, label_col]].drop_duplicates()
        rows.append(
            {
                "split": name,
                "images": len(sub),
                "patients": len(patient_rows),
                "positive_patients": int(patient_rows[label_col].sum()),
                "patient_positive_rate": round(float(patient_rows[label_col].mean()), 3),
            }
        )
    return pd.DataFrame(rows)

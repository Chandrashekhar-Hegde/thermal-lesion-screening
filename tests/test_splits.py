"""
Tests that make the central claim of this repository falsifiable.

If `test_no_patient_appears_in_two_partitions` passes, no patient's data can
straddle a train/test boundary. That is the whole point. Everything else in the
repo is ordinary engineering; this is the part that decides whether the reported
numbers mean anything.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.splits import (  # noqa: E402
    PatientLeakageError,
    Split,
    assert_no_patient_leakage,
    describe_split,
    patient_level_cv,
    patient_level_split,
    repeated_patient_level_cv,
)


def make_manifest(n_patients=60, imgs_per_patient=4, pos_rate=0.35, seed=7):
    """Synthetic manifest: several images per patient, label constant per patient."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        label = int(rng.random() < pos_rate)
        for k in range(imgs_per_patient):
            rows.append(
                {
                    "image_id": f"P{p:03d}_{k}",
                    "patient_id": f"P{p:03d}",
                    "label": label,
                    "path": f"data/P{p:03d}_{k}.png",
                }
            )
    return pd.DataFrame(rows)


def test_no_patient_appears_in_two_partitions():
    m = make_manifest()
    s = patient_level_split(m, seed=1)

    tr = set(m.iloc[s.train]["patient_id"])
    va = set(m.iloc[s.val]["patient_id"])
    te = set(m.iloc[s.test]["patient_id"])

    assert tr & va == set(), "patient leaked between train and val"
    assert tr & te == set(), "patient leaked between train and test"
    assert va & te == set(), "patient leaked between val and test"


def test_every_image_used_exactly_once():
    m = make_manifest()
    s = patient_level_split(m, seed=2)
    allocated = np.concatenate([s.train, s.val, s.test])
    assert len(allocated) == len(m)
    assert len(np.unique(allocated)) == len(m), "an image was allocated twice"


def test_requested_fractions_are_applied_to_patients():
    manifest = make_manifest(n_patients=100)
    split = patient_level_split(manifest, test_frac=0.2, val_frac=0.1, seed=2)

    assert manifest.iloc[split.train]["patient_id"].nunique() == 70
    assert manifest.iloc[split.val]["patient_id"].nunique() == 10
    assert manifest.iloc[split.test]["patient_id"].nunique() == 20


def test_both_classes_present_in_every_partition():
    m = make_manifest()
    s = patient_level_split(m, seed=3)
    for name, ix in (("train", s.train), ("val", s.val), ("test", s.test)):
        labels = set(m.iloc[ix]["label"])
        assert labels == {0, 1}, f"partition '{name}' is missing a class: {labels}"


def test_leakage_detector_actually_fires():
    """A deliberately corrupted split must be rejected. A guard that never
    fires is not a guard."""
    m = make_manifest()
    bad = Split(
        train=np.arange(0, 100),
        val=np.arange(90, 140),  # overlaps train -> shares patients
        test=np.arange(140, len(m)),
    )
    with pytest.raises(PatientLeakageError):
        assert_no_patient_leakage(m, bad)


def test_cv_folds_are_patient_grouped():
    m = make_manifest()
    for train_i, val_i in patient_level_cv(m, n_folds=5, seed=4):
        assert not (set(m.iloc[train_i]["patient_id"]) & set(m.iloc[val_i]["patient_id"])), (
            "CV fold leaked a patient"
        )


def test_repeated_cv_has_every_patient_in_one_test_fold_per_repeat():
    manifest = make_manifest(n_patients=100)
    folds = list(repeated_patient_level_cv(manifest, n_folds=5, n_repeats=3, seed=4))

    assert len(folds) == 15
    all_patients = set(manifest["patient_id"])
    for repeat in range(3):
        test_patients = []
        for repeated_fold in folds:
            if repeated_fold.repeat != repeat:
                continue
            split = repeated_fold.split
            train = set(manifest.iloc[split.train]["patient_id"])
            val = set(manifest.iloc[split.val]["patient_id"])
            test = set(manifest.iloc[split.test]["patient_id"])
            assert not train & val
            assert not train & test
            assert not val & test
            test_patients.extend(test)

        assert set(test_patients) == all_patients
        assert len(test_patients) == len(all_patients)


def test_repeated_cv_is_deterministic():
    manifest = make_manifest(n_patients=100)
    first = list(repeated_patient_level_cv(manifest, n_folds=5, n_repeats=2, seed=11))
    second = list(repeated_patient_level_cv(manifest, n_folds=5, n_repeats=2, seed=11))

    for index, left in enumerate(first):
        right = second[index]
        assert left.repeat == right.repeat
        assert left.fold == right.fold
        assert np.array_equal(left.split.train, right.split.train)
        assert np.array_equal(left.split.val, right.split.val)
        assert np.array_equal(left.split.test, right.split.test)


def test_split_is_deterministic_under_seed():
    m = make_manifest()
    a = patient_level_split(m, seed=11)
    b = patient_level_split(m, seed=11)
    assert np.array_equal(a.test, b.test), "same seed produced different splits"


def test_describe_split_reports_patient_counts():
    m = make_manifest()
    s = patient_level_split(m, seed=5)
    d = describe_split(m, s)
    assert set(d["split"]) == {"train", "val", "test"}
    assert d["patients"].sum() == m["patient_id"].nunique()
    assert "positive_patients" in d
    assert "patient_positive_rate" in d


def test_split_rejects_invalid_total_fraction():
    with pytest.raises(ValueError, match="less than 1"):
        patient_level_split(make_manifest(), test_frac=0.6, val_frac=0.4)


def test_split_rejects_conflicting_patient_labels():
    manifest = make_manifest()
    manifest.loc[1, "label"] = 1 - manifest.loc[0, "label"]
    with pytest.raises(ValueError, match="conflicting labels"):
        patient_level_split(manifest)

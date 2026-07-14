import numpy as np
import pytest

from src.metrics import (
    aggregate_by_patient,
    full_report,
    threshold_at_sensitivity,
)


def test_aggregate_by_patient_returns_one_mean_score_per_patient():
    labels, scores, patients = aggregate_by_patient(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.3, 0.7, 0.9]),
        np.array(["P1", "P1", "P2", "P2"]),
    )

    assert labels.tolist() == [0, 1]
    assert scores == pytest.approx([0.2, 0.8])
    assert patients.tolist() == ["P1", "P2"]


def test_aggregate_by_patient_rejects_conflicting_labels():
    with pytest.raises(ValueError, match="conflicting labels"):
        aggregate_by_patient(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            np.array(["P1", "P1"]),
        )


def test_threshold_uses_highest_value_that_reaches_target_sensitivity():
    threshold = threshold_at_sensitivity(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.35, 0.8]),
        target_sensitivity=0.5,
    )
    assert threshold == pytest.approx(0.8)


def test_full_report_uses_the_supplied_threshold():
    report = full_report(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.4, 0.9]),
        threshold=0.5,
        target_sensitivity=0.9,
        n_boot=100,
    )

    point = report["operating_point"]
    assert point["threshold"] == 0.5
    assert point["sensitivity"] == 0.5
    assert point["specificity"] == 1.0
    assert report["sample_unit"] == "patients"

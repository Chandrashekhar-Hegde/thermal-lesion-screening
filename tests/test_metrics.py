import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    aggregate_by_patient,
    full_report,
    repeated_cv_report,
    select_aggregation,
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


def test_top_k_mean_uses_available_highest_scores():
    labels, scores, _ = aggregate_by_patient(
        np.array([0, 0, 0, 1]),
        np.array([0.1, 0.8, 0.6, 0.9]),
        np.array(["P1", "P1", "P1", "P2"]),
        reducer="top_k_mean",
        top_k=2,
    )

    assert labels.tolist() == [0, 1]
    assert scores == pytest.approx([0.7, 0.9])


def test_top_k_mean_requires_positive_k():
    with pytest.raises(ValueError, match="positive integer"):
        aggregate_by_patient(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            np.array(["P1", "P2"]),
            reducer="top_k_mean",
            top_k=0,
        )


def test_aggregation_is_selected_on_validation_specificity():
    choice, comparison = select_aggregation(
        np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        np.array([0.1, 0.9, 0.2, 0.2, 0.8, 0.7, 0.6, 0.5]),
        np.array(["N1", "N1", "N2", "N2", "P1", "P1", "P2", "P2"]),
        candidates=[{"name": "mean"}, {"name": "max"}],
        target_sensitivity=1.0,
    )

    assert choice.name == "mean"
    assert len(comparison) == 2


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


def test_repeated_cv_report_requires_one_prediction_per_patient_per_repeat():
    predictions = pd.DataFrame(
        {
            "repeat": [0, 0],
            "patient_id": ["P1", "P1"],
            "label": [0, 0],
            "score": [0.1, 0.2],
            "prediction": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="once per repeat"):
        repeated_cv_report(predictions, n_boot=10)


def test_repeated_cv_report_summarizes_patient_clustered_repeats():
    rows = []
    for repeat in range(2):
        rows.extend(
            [
                {
                    "repeat": repeat,
                    "patient_id": "N1",
                    "label": 0,
                    "score": 0.1,
                    "prediction": 0,
                },
                {
                    "repeat": repeat,
                    "patient_id": "N2",
                    "label": 0,
                    "score": 0.2,
                    "prediction": 0,
                },
                {
                    "repeat": repeat,
                    "patient_id": "P1",
                    "label": 1,
                    "score": 0.8,
                    "prediction": 1,
                },
                {
                    "repeat": repeat,
                    "patient_id": "P2",
                    "label": 1,
                    "score": 0.9,
                    "prediction": 1,
                },
            ]
        )

    report = repeated_cv_report(pd.DataFrame(rows), n_boot=50, seed=3)

    assert report["n_patients"] == 4
    assert report["n_repeats"] == 2
    assert report["summary"]["auroc"]["mean"] == 1.0
    assert report["summary"]["sensitivity"]["mean"] == 1.0

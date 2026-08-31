import pandas as pd

from src.reporting import write_evaluation_figures, write_results_markdown


def _predictions():
    rows = []
    for repeat in range(2):
        rows.extend(
            [
                {"repeat": repeat, "patient_id": "N1", "label": 0, "score": 0.1, "prediction": 0},
                {"repeat": repeat, "patient_id": "N2", "label": 0, "score": 0.2, "prediction": 0},
                {"repeat": repeat, "patient_id": "P1", "label": 1, "score": 0.8, "prediction": 1},
                {"repeat": repeat, "patient_id": "P2", "label": 1, "score": 0.9, "prediction": 1},
            ]
        )
    return pd.DataFrame(rows)


def test_write_evaluation_figures_creates_three_pngs(tmp_path):
    write_evaluation_figures(_predictions(), tmp_path, seed=2)

    assert sorted(path.name for path in tmp_path.glob("*.png")) == [
        "confusion_matrix.png",
        "roc_curve.png",
        "sensitivity_specificity.png",
    ]


def test_write_results_markdown_uses_completed_values(tmp_path):
    report = {
        "target_sensitivity": 0.9,
        "integrity": {
            "before": {"patients": 20},
            "after": {"patients": 18},
            "excluded": {"patients": 1},
            "merged_patient_aliases": 1,
        },
        "cross_validation": {
            "n_folds": 5,
            "n_repeats": 2,
            "folds": [
                {
                    "patients": {"train": 11, "validation": 3, "test": 4},
                    "selected_aggregation": {"name": "mean"},
                }
            ],
        },
        "primary_nested_selection": {
            "summary": {
                metric: {"mean": 0.8, "sd": 0.1}
                for metric in ("auroc", "auprc", "sensitivity", "specificity")
            },
            "patient_cluster_bootstrap_95ci": {
                metric: [0.6, 0.9] for metric in ("auroc", "auprc", "sensitivity", "specificity")
            },
        },
    }

    output = tmp_path / "RESULTS.md"
    write_results_markdown(report, output)

    content = output.read_text()
    assert "5 folds × 2 repeats" in content
    assert "Patients excluded | 1" in content
    assert "0.800" in content

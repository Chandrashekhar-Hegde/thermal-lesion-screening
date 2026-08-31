"""Figures and concise Markdown generated from repeated-CV predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve


def _validate_predictions(predictions: pd.DataFrame) -> None:
    required = {"repeat", "patient_id", "label", "score", "prediction"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing required column(s): {sorted(missing)}")
    if predictions.empty:
        raise ValueError("predictions cannot be empty")


def _repeat_roc(frame: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    false_positive_rate, true_positive_rate, _ = roc_curve(frame["label"], frame["score"])
    return np.interp(grid, false_positive_rate, true_positive_rate)


def _roc_band(predictions: pd.DataFrame, seed: int, n_boot: int = 500):
    grid = np.linspace(0, 1, 201)
    repeats = sorted(predictions["repeat"].unique())
    indexed = {
        repeat: predictions[predictions["repeat"] == repeat].set_index("patient_id")
        for repeat in repeats
    }
    patients = sorted(indexed[repeats[0]].index.tolist(), key=str)
    observed = np.mean(
        [_repeat_roc(indexed[repeat].reset_index(), grid) for repeat in repeats],
        axis=0,
    )

    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        repeat_curves = []
        for repeat in repeats:
            sample = indexed[repeat].loc[sampled]
            if sample["label"].nunique() < 2:
                repeat_curves = []
                break
            repeat_curves.append(_repeat_roc(sample, grid))
        if repeat_curves:
            curves.append(np.mean(repeat_curves, axis=0))
    if not curves:
        return grid, observed, np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    low, high = np.percentile(np.asarray(curves), [2.5, 97.5], axis=0)
    return grid, observed, low, high


def _operating_points(predictions: pd.DataFrame) -> list[tuple[float, float]]:
    points = []
    for _, frame in predictions.groupby("repeat"):
        tn, fp, fn, tp = confusion_matrix(
            frame["label"], frame["prediction"], labels=[0, 1]
        ).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        points.append((specificity, sensitivity))
    return points


def write_evaluation_figures(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    seed: int = 0,
    n_boot: int = 500,
    target_sensitivity: float = 0.90,
) -> None:
    """Write the three predeclared patient-level evaluation figures."""
    _validate_predictions(predictions)
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0 < target_sensitivity <= 1:
        raise ValueError("target_sensitivity must be in (0, 1]")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    colors = {"line": "#155e75", "band": "#67e8f9", "point": "#b91c1c"}

    grid, mean_roc, low_roc, high_roc = _roc_band(predictions, seed, n_boot)
    figure, axis = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    axis.plot(grid, mean_roc, color=colors["line"], linewidth=2, label="Mean across repeats")
    axis.fill_between(
        grid,
        low_roc,
        high_roc,
        color=colors["band"],
        alpha=0.35,
        label="95% patient-cluster bootstrap band",
    )
    axis.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1)
    for specificity, sensitivity in _operating_points(predictions):
        axis.scatter(1 - specificity, sensitivity, color=colors["point"], s=24, zorder=3)
    axis.set(xlabel="False-positive rate", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("Patient-level ROC")
    axis.legend(loc="lower right", frameon=False)
    figure.savefig(destination / "roc_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    for repeat, frame in predictions.groupby("repeat"):
        false_positive_rate, sensitivity, _ = roc_curve(frame["label"], frame["score"])
        axis.plot(
            1 - false_positive_rate,
            sensitivity,
            linewidth=1.2,
            alpha=0.55,
            label=f"Repeat {int(repeat) + 1}",
        )
    for specificity, sensitivity in _operating_points(predictions):
        axis.scatter(specificity, sensitivity, color=colors["point"], s=24, zorder=3)
    axis.axhline(
        target_sensitivity,
        color="#475569",
        linestyle="--",
        linewidth=1,
        label=f"{target_sensitivity:.0%} target",
    )
    axis.set(xlabel="Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("Sensitivity–specificity trade-off")
    axis.legend(loc="lower left", frameon=False, fontsize=8)
    figure.savefig(destination / "sensitivity_specificity.png", dpi=180)
    plt.close(figure)

    matrices = np.asarray(
        [
            confusion_matrix(frame["label"], frame["prediction"], labels=[0, 1])
            for _, frame in predictions.groupby("repeat")
        ],
        dtype=float,
    )
    means = matrices.mean(axis=0)
    standard_deviations = matrices.std(axis=0, ddof=1) if len(matrices) > 1 else np.zeros((2, 2))
    figure, axis = plt.subplots(figsize=(5.4, 4.8), constrained_layout=True)
    image = axis.imshow(means, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{means[row, column]:.1f}\n± {standard_deviations[row, column]:.1f}",
                ha="center",
                va="center",
                color="black",
            )
    axis.set_xticks([0, 1], labels=["Predicted negative", "Predicted positive"])
    axis.set_yticks([0, 1], labels=["Actual negative", "Actual positive"])
    axis.set_title("Patient-level confusion matrix\nmean ± SD across repeats")
    figure.colorbar(image, ax=axis, shrink=0.8, label="Patients per repeat")
    figure.savefig(destination / "confusion_matrix.png", dpi=180)
    plt.close(figure)


def _format_interval(interval: list[float]) -> str:
    return f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def write_results_markdown(report: dict, output_path: str | Path) -> None:
    """Write a short, reviewable results page using only completed-run values."""
    primary = report["primary_nested_selection"]
    summary = primary["summary"]
    intervals = primary["patient_cluster_bootstrap_95ci"]
    integrity = report["integrity"]
    folds = report["cross_validation"]["folds"]
    target = report["target_sensitivity"]

    split_ranges = {}
    for name in ("train", "validation", "test"):
        values = [fold["patients"][name] for fold in folds]
        split_ranges[name] = f"{min(values)}–{max(values)}"

    rows = []
    for name, label in (
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
    ):
        rows.append(
            f"| {label} | {summary[name]['mean']:.3f} | {summary[name]['sd']:.3f} | "
            f"{_format_interval(intervals[name])} |"
        )

    selected = pd.Series([fold["selected_aggregation"]["name"] for fold in folds]).value_counts()
    reducer_text = ", ".join(f"{name}: {count}" for name, count in selected.items())
    cv_description = (
        f"{report['cross_validation']['n_folds']} folds × "
        f"{report['cross_validation']['n_repeats']} repeats"
    )
    content = f"""# Results

These values come from a completed repeated patient-level nested cross-validation run.
They are not clinical performance claims.

| Protocol item | Value |
|---|---:|
| Patients before integrity checks | {integrity["before"]["patients"]} |
| Patients after integrity checks | {integrity["after"]["patients"]} |
| Patients excluded | {integrity["excluded"]["patients"]} |
| Patient aliases merged | {integrity["merged_patient_aliases"]} |
| Repeated CV | {cv_description} |
| Patients per inner train fold | {split_ranges["train"]} |
| Patients per validation fold | {split_ranges["validation"]} |
| Patients per outer test fold | {split_ranges["test"]} |
| Target validation sensitivity | {target:.0%} |

| Patient-level metric | Mean | Repeat SD | 95% patient-cluster bootstrap CI |
|---|---:|---:|---:|
{chr(10).join(rows)}

Reducers were selected inside each outer fold using validation specificity at the fixed
sensitivity target. Selection counts: {reducer_text}.

The operating threshold and reducer were never selected on an outer test fold. The
confusion matrix reports mean patient counts per repeat because every patient is tested
once in each repeat.
"""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")

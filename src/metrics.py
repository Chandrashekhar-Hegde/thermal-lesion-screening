"""Patient-level diagnostic accuracy metrics for screening models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


@dataclass
class OperatingPoint:
    """Performance at one fixed decision threshold."""

    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"threshold={self.threshold:.3f}  "
            f"sens={self.sensitivity:.3f}  spec={self.specificity:.3f}  "
            f"PPV={self.ppv:.3f}  NPV={self.npv:.3f}  "
            f"(TP={self.tp} FP={self.fp} TN={self.tn} FN={self.fn})"
        )


def _as_binary_arrays(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    require_both_classes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_score, dtype=float).reshape(-1)

    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("labels and scores must be non-empty and have equal length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")

    classes = set(np.unique(labels).tolist())
    if not classes.issubset({0, 1}):
        raise ValueError("labels must be binary values: 0 or 1")
    if require_both_classes and classes != {0, 1}:
        raise ValueError("labels must contain both classes")
    return labels.astype(int), scores


def aggregate_by_patient(
    y_true: np.ndarray,
    y_score: np.ndarray,
    patient_ids: np.ndarray,
    reducer: Literal["mean", "max"] = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse image predictions into one score and label per patient."""
    labels, scores = _as_binary_arrays(y_true, y_score, require_both_classes=False)
    patients = np.asarray(patient_ids, dtype=object).reshape(-1)
    if len(patients) != len(labels):
        raise ValueError("patient_ids must have the same length as labels and scores")
    if reducer not in {"mean", "max"}:
        raise ValueError("reducer must be 'mean' or 'max'")

    grouped: dict[object, list[int]] = {}
    for index, patient_id in enumerate(patients):
        if patient_id is None or str(patient_id).strip() == "":
            raise ValueError("patient_ids cannot contain empty values")
        grouped.setdefault(patient_id, []).append(index)

    patient_labels: list[int] = []
    patient_scores: list[float] = []
    ordered_ids: list[object] = []
    for patient_id, indices in grouped.items():
        patient_classes = np.unique(labels[indices])
        if len(patient_classes) != 1:
            raise ValueError(f"patient {patient_id!r} has conflicting labels")
        patient_labels.append(int(patient_classes[0]))
        values = scores[indices]
        patient_scores.append(float(values.mean() if reducer == "mean" else values.max()))
        ordered_ids.append(patient_id)

    return (
        np.asarray(patient_labels, dtype=int),
        np.asarray(patient_scores, dtype=float),
        np.asarray(ordered_ids, dtype=object),
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def operating_point(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> OperatingPoint:
    """Return the confusion-matrix summary at a fixed threshold."""
    labels, scores = _as_binary_arrays(y_true, y_score, require_both_classes=False)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, y_pred, labels=[0, 1]).ravel()
    return OperatingPoint(
        threshold=float(threshold),
        sensitivity=_safe_div(tp, tp + fn),
        specificity=_safe_div(tn, tn + fp),
        ppv=_safe_div(tp, tp + fp),
        npv=_safe_div(tn, tn + fn),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )


def threshold_at_sensitivity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_sensitivity: float = 0.90,
) -> float:
    """Return the highest threshold that reaches the requested sensitivity."""
    if not 0 < target_sensitivity <= 1:
        raise ValueError("target_sensitivity must be in (0, 1]")
    labels, scores = _as_binary_arrays(y_true, y_score)
    _, true_positive_rate, thresholds = roc_curve(labels, scores)
    candidates = thresholds[(true_positive_rate >= target_sensitivity) & np.isfinite(thresholds)]
    if len(candidates) == 0:
        raise ValueError("no finite threshold reaches the requested sensitivity")
    return float(np.max(candidates))


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval over independent observations."""
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    labels, scores = _as_binary_arrays(y_true, y_score)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) < 2:
            continue
        stats.append(metric_fn(labels[indices], scores[indices]))
    if not stats:
        return (float("nan"), float("nan"))
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def full_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
    target_sensitivity: float,
    n_boot: int = 2000,
    sample_unit: str = "patients",
) -> dict:
    """Build a diagnostic report using an externally selected threshold."""
    labels, scores = _as_binary_arrays(y_true, y_score)
    if not np.logical_and(scores >= 0, scores <= 1).all():
        raise ValueError("scores must be probabilities in [0, 1]")

    auc = roc_auc_score(labels, scores)
    auc_low, auc_high = bootstrap_ci(labels, scores, roc_auc_score, n_boot=n_boot)
    point = operating_point(labels, scores, threshold)
    sensitivity_low, sensitivity_high = bootstrap_ci(
        labels,
        scores,
        lambda a, b: operating_point(a, b, threshold).sensitivity,
        n_boot=n_boot,
    )
    specificity_low, specificity_high = bootstrap_ci(
        labels,
        scores,
        lambda a, b: operating_point(a, b, threshold).specificity,
        n_boot=n_boot,
    )

    return {
        "n": int(len(labels)),
        "sample_unit": sample_unit,
        "n_positive": int(labels.sum()),
        "prevalence": round(float(labels.mean()), 4),
        "auroc": round(float(auc), 4),
        "auroc_95ci": [round(auc_low, 4), round(auc_high, 4)],
        "auprc": round(float(average_precision_score(labels, scores)), 4),
        "brier": round(float(brier_score_loss(labels, scores)), 4),
        "operating_point": point.as_dict(),
        "sensitivity_95ci": [round(sensitivity_low, 4), round(sensitivity_high, 4)],
        "specificity_95ci": [round(specificity_low, 4), round(specificity_high, 4)],
        "target_sensitivity": target_sensitivity,
    }


def format_report(report: dict) -> str:
    """Format a report for terminal output."""
    point = report["operating_point"]
    return (
        f"n = {report['n']} {report['sample_unit']}  |  positives = {report['n_positive']}  "
        f"(prevalence {report['prevalence']:.1%})\n"
        f"AUROC  {report['auroc']:.3f}  "
        f"95% CI [{report['auroc_95ci'][0]:.3f}, {report['auroc_95ci'][1]:.3f}]\n"
        f"AUPRC  {report['auprc']:.3f}   Brier  {report['brier']:.3f}\n"
        f"\nAt threshold {point['threshold']:.3f} "
        f"(selected on validation data for >= {report['target_sensitivity']:.0%} sensitivity):\n"
        f"  Sensitivity  {point['sensitivity']:.3f}  "
        f"95% CI [{report['sensitivity_95ci'][0]:.3f}, "
        f"{report['sensitivity_95ci'][1]:.3f}]\n"
        f"  Specificity  {point['specificity']:.3f}  "
        f"95% CI [{report['specificity_95ci'][0]:.3f}, "
        f"{report['specificity_95ci'][1]:.3f}]\n"
        f"  PPV {point['ppv']:.3f}   NPV {point['npv']:.3f}\n"
        f"  TP {point['tp']}  FP {point['fp']}  TN {point['tn']}  FN {point['fn']}"
    )

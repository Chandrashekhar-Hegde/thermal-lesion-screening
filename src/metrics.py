"""Patient-level diagnostic accuracy metrics for screening models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
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


@dataclass(frozen=True)
class AggregationChoice:
    """A validation-selected patient score reducer and operating threshold."""

    name: str
    top_k: int | None
    threshold: float
    sensitivity: float
    specificity: float
    auroc: float

    def as_dict(self) -> dict:
        return asdict(self)


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
    reducer: Literal["mean", "max", "top_k_mean"] = "mean",
    top_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse image predictions into one score and label per patient."""
    labels, scores = _as_binary_arrays(y_true, y_score, require_both_classes=False)
    patients = np.asarray(patient_ids, dtype=object).reshape(-1)
    if len(patients) != len(labels):
        raise ValueError("patient_ids must have the same length as labels and scores")
    if reducer not in {"mean", "max", "top_k_mean"}:
        raise ValueError("reducer must be 'mean', 'max', or 'top_k_mean'")
    if reducer == "top_k_mean" and (not isinstance(top_k, int) or top_k <= 0):
        raise ValueError("top_k must be a positive integer for top_k_mean")
    if reducer != "top_k_mean" and top_k is not None:
        raise ValueError("top_k is only valid with top_k_mean")

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
        if reducer == "mean":
            reduced_score = values.mean()
        elif reducer == "max":
            reduced_score = values.max()
        else:
            reduced_score = np.sort(values)[-min(top_k, len(values)) :].mean()
        patient_scores.append(float(reduced_score))
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


def select_aggregation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    patient_ids: np.ndarray,
    candidates: list[dict],
    target_sensitivity: float,
) -> tuple[AggregationChoice, list[dict]]:
    """Choose a reducer on validation data by specificity at target sensitivity."""
    if not candidates:
        raise ValueError("at least one aggregation candidate is required")

    choices: list[AggregationChoice] = []
    for candidate in candidates:
        unknown = set(candidate) - {"name", "top_k"}
        if unknown:
            raise ValueError(f"unknown aggregation setting(s): {sorted(unknown)}")
        name = candidate.get("name")
        top_k = candidate.get("top_k")
        labels, scores, _ = aggregate_by_patient(
            y_true,
            y_score,
            patient_ids,
            reducer=name,
            top_k=top_k,
        )
        threshold = threshold_at_sensitivity(labels, scores, target_sensitivity)
        point = operating_point(labels, scores, threshold)
        choices.append(
            AggregationChoice(
                name=name,
                top_k=top_k,
                threshold=threshold,
                sensitivity=point.sensitivity,
                specificity=point.specificity,
                auroc=float(roc_auc_score(labels, scores)),
            )
        )

    best = choices[0]
    for choice in choices[1:]:
        if (choice.specificity, choice.auroc) > (best.specificity, best.auroc):
            best = choice
    return best, [choice.as_dict() for choice in choices]


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


def _decision_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "ppv": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _repeat_metrics(frame: pd.DataFrame) -> dict:
    labels, scores = _as_binary_arrays(frame["label"], frame["score"])
    predictions = np.asarray(frame["prediction"], dtype=int)
    if not set(np.unique(predictions).tolist()).issubset({0, 1}):
        raise ValueError("prediction must contain only 0 and 1")
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        **_decision_metrics(labels, predictions),
    }


def repeated_cv_report(
    predictions: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Summarize repeated OOF predictions with a patient-cluster bootstrap."""
    required = {"repeat", "patient_id", "label", "score", "prediction"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing required column(s): {sorted(missing)}")
    if predictions.empty:
        raise ValueError("predictions cannot be empty")
    if predictions.duplicated(["repeat", "patient_id"]).any():
        raise ValueError("each patient must appear once per repeat")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    frame = predictions.copy()
    if frame.groupby("patient_id")["label"].nunique().max() != 1:
        raise ValueError("a patient has conflicting labels across repeats")
    if not frame["score"].between(0, 1).all():
        raise ValueError("score must contain probabilities in [0, 1]")
    repeats = sorted(frame["repeat"].unique().tolist())
    patient_sets = [set(frame.loc[frame["repeat"] == repeat, "patient_id"]) for repeat in repeats]
    if any(patients != patient_sets[0] for patients in patient_sets[1:]):
        raise ValueError("every repeat must contain the same patients")

    per_repeat = []
    for repeat in repeats:
        metrics = _repeat_metrics(frame[frame["repeat"] == repeat])
        per_repeat.append({"repeat": int(repeat), **metrics})

    summary = {}
    spread_metrics = ("auroc", "auprc", "brier", "sensitivity", "specificity", "ppv", "npv")
    for name in spread_metrics:
        values = np.asarray([row[name] for row in per_repeat], dtype=float)
        summary[name] = {
            "mean": round(float(values.mean()), 4),
            "sd": round(float(values.std(ddof=1)), 4) if len(values) > 1 else 0.0,
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
        }

    ordered_patients = sorted(patient_sets[0], key=str)
    indexed = {
        repeat: frame[frame["repeat"] == repeat].set_index("patient_id") for repeat in repeats
    }
    rng = np.random.default_rng(seed)
    bootstrap_values = {name: [] for name in ("auroc", "auprc", "sensitivity", "specificity")}
    for _ in range(n_boot):
        sampled = rng.choice(ordered_patients, size=len(ordered_patients), replace=True)
        sampled_metrics = []
        valid = True
        for repeat in repeats:
            repeat_sample = indexed[repeat].loc[sampled]
            if repeat_sample["label"].nunique() < 2:
                valid = False
                break
            sampled_metrics.append(_repeat_metrics(repeat_sample))
        if not valid:
            continue
        for name in bootstrap_values:
            bootstrap_values[name].append(
                float(np.mean([metrics[name] for metrics in sampled_metrics]))
            )

    intervals = {}
    for name, values in bootstrap_values.items():
        if not values:
            intervals[name] = [float("nan"), float("nan")]
        else:
            low, high = np.percentile(values, [2.5, 97.5])
            intervals[name] = [round(float(low), 4), round(float(high), 4)]

    return {
        "protocol": "repeated patient-level nested cross-validation",
        "n_patients": int(len(ordered_patients)),
        "n_repeats": int(len(repeats)),
        "decision_rule": "fold-specific reducer and threshold selected on validation patients",
        "per_repeat": per_repeat,
        "summary": summary,
        "patient_cluster_bootstrap_95ci": intervals,
        "n_bootstrap": int(n_boot),
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

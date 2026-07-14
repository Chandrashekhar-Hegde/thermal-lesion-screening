"""
Diagnostic-accuracy metrics for screening models.

A screening claim is not a number, it is a *pair* of numbers at a *stated*
operating point, with an interval around it. A model that reports "94.7%
sensitivity" and nothing else is indistinguishable from a model that shouts
"positive" at every patient — that one also scores 100% sensitivity.

Everything in this module therefore reports sensitivity and specificity
together, names the threshold they were computed at, and attaches a bootstrap
confidence interval. Point estimates on a few hundred images are noisy; saying
so is not weakness, it is competence.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)


@dataclass
class OperatingPoint:
    """Performance at one specific decision threshold."""

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


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def operating_point(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> OperatingPoint:
    """Full confusion-matrix summary at an explicitly chosen threshold."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return OperatingPoint(
        threshold=float(threshold),
        sensitivity=_safe_div(tp, tp + fn),
        specificity=_safe_div(tn, tn + fp),
        ppv=_safe_div(tp, tp + fp),
        npv=_safe_div(tn, tn + fn),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )


def threshold_at_sensitivity(
    y_true: np.ndarray, y_score: np.ndarray, target_sensitivity: float = 0.90
) -> float:
    """Lowest threshold achieving at least `target_sensitivity`.

    For a screening tool this is usually the right way to pick an operating
    point: fix the miss rate you can clinically tolerate, then report whatever
    specificity that costs you. Choosing the threshold that flatters your
    headline number instead is how screening tools get abandoned in the field.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ok = np.where(tpr >= target_sensitivity)[0]
    if len(ok) == 0:
        return float(np.min(thr))
    return float(thr[ok[0]])


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for any metric of the form fn(y_true, y_score)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        if len(np.unique(y_true[ix])) < 2:   # degenerate resample
            continue
        stats.append(metric_fn(y_true[ix], y_score[ix]))
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def full_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_sensitivity: float = 0.90,
    n_boot: int = 2000,
) -> dict:
    """Everything an interviewer or a reviewer will ask for, in one call."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    auc = roc_auc_score(y_true, y_score)
    auc_lo, auc_hi = bootstrap_ci(y_true, y_score, roc_auc_score, n_boot=n_boot)

    thr = threshold_at_sensitivity(y_true, y_score, target_sensitivity)
    op = operating_point(y_true, y_score, thr)

    sens_lo, sens_hi = bootstrap_ci(
        y_true, y_score,
        lambda a, b: operating_point(a, b, thr).sensitivity, n_boot=n_boot,
    )
    spec_lo, spec_hi = bootstrap_ci(
        y_true, y_score,
        lambda a, b: operating_point(a, b, thr).specificity, n_boot=n_boot,
    )

    return {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "prevalence": round(float(y_true.mean()), 4),
        "auroc": round(float(auc), 4),
        "auroc_95ci": [round(auc_lo, 4), round(auc_hi, 4)],
        "auprc": round(float(average_precision_score(y_true, y_score)), 4),
        "brier": round(float(brier_score_loss(y_true, y_score)), 4),
        "operating_point": op.as_dict(),
        "sensitivity_95ci": [round(sens_lo, 4), round(sens_hi, 4)],
        "specificity_95ci": [round(spec_lo, 4), round(spec_hi, 4)],
        "target_sensitivity": target_sensitivity,
    }


def format_report(rep: dict) -> str:
    """Human-readable block. Paste into a README; do not paste a bare number."""
    op = rep["operating_point"]
    return (
        f"n = {rep['n']} images  |  positives = {rep['n_positive']}  "
        f"(prevalence {rep['prevalence']:.1%})\n"
        f"AUROC  {rep['auroc']:.3f}  95% CI [{rep['auroc_95ci'][0]:.3f}, {rep['auroc_95ci'][1]:.3f}]\n"
        f"AUPRC  {rep['auprc']:.3f}   Brier  {rep['brier']:.3f}\n"
        f"\nAt threshold {op['threshold']:.3f} "
        f"(chosen for >= {rep['target_sensitivity']:.0%} sensitivity):\n"
        f"  Sensitivity  {op['sensitivity']:.3f}  "
        f"95% CI [{rep['sensitivity_95ci'][0]:.3f}, {rep['sensitivity_95ci'][1]:.3f}]\n"
        f"  Specificity  {op['specificity']:.3f}  "
        f"95% CI [{rep['specificity_95ci'][0]:.3f}, {rep['specificity_95ci'][1]:.3f}]\n"
        f"  PPV {op['ppv']:.3f}   NPV {op['npv']:.3f}\n"
        f"  TP {op['tp']}  FP {op['fp']}  TN {op['tn']}  FN {op['fn']}"
    )

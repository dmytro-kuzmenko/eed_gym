"""Utility metrics for EED-Gym evaluations.

The helpers in this module avoid heavy dependencies (e.g., scikit-learn) while
covering the handful of metrics we report in the paper:

* ROC-AUC                         via the Mann–Whitney formulation
* PR-AUC                          via sorted precision/recall deltas
* Brier score                     (optionally using bin-averaged probabilities)
* Expected Calibration Error      given either raw samples or per-bin stats

All inputs are NumPy arrays; functions try to mirror scikit-learn semantics
when it is sensible (e.g., returning ``nan`` if only one class is present).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BinStats:
    """Container for calibration bin statistics."""

    counts: np.ndarray
    pred_rates: np.ndarray
    true_rates: np.ndarray

    def ece(self) -> float:
        total = int(self.counts.sum())
        if total == 0:
            return 0.0
        weights = self.counts / total
        return float(np.sum(weights * np.abs(self.pred_rates - self.true_rates)))


def _count_pos_neg(labels: np.ndarray) -> tuple[int, int]:
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    return pos, neg


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve (Mann–Whitney U)."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos, neg = _count_pos_neg(y_true)
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    # average ranks for ties
    values, inverse, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    for idx, c in enumerate(counts):
        if c > 1:
            mask = inverse == idx
            ranks[mask] = ranks[mask].mean()

    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision–recall curve."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos, neg = _count_pos_neg(y_true)
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tp = fp = 0
    prev_recall = 0.0
    auc = 0.0
    for truth in y_sorted:
        if truth == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / pos
        precision = tp / max(1, tp + fp)
        auc += precision * (recall - prev_recall)
        prev_recall = recall
    return float(auc)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Standard Brier score on probabilities."""

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def brier_score_binned(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> float:
    """Brier score using empirical bin means as probabilities."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    probs = np.zeros_like(y_score)
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == bins - 1:
            mask = (y_score >= left) & (y_score <= right)
        else:
            mask = (y_score >= left) & (y_score < right)
        if mask.any():
            probs[mask] = y_true[mask].mean()
    return brier_score(y_true, probs)


def calibration_bins(y_prob: np.ndarray, y_true: np.ndarray, bins: int = 10) -> BinStats:
    """Compute per-bin predicted/true rates for calibration plots."""

    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts = np.zeros(bins, dtype=int)
    pred_rates = np.zeros(bins, dtype=float)
    true_rates = np.zeros(bins, dtype=float)

    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (y_prob >= left) & (y_prob <= right if idx == bins - 1 else y_prob < right)
        counts[idx] = int(mask.sum())
        if counts[idx] == 0:
            continue
        pred_rates[idx] = float(np.clip(y_prob[mask].mean(), 0.0, 1.0))
        true_rates[idx] = float(y_true[mask].mean())

    return BinStats(counts=counts, pred_rates=pred_rates, true_rates=true_rates)


def expected_calibration_error(y_prob: np.ndarray, y_true: np.ndarray, bins: int = 10) -> float:
    """ECE computed from raw samples."""

    return calibration_bins(y_prob, y_true, bins=bins).ece()


def expected_calibration_error_from_bins(counts: Iterable[int], pred_rates: Iterable[float], true_rates: Iterable[float]) -> float:
    """ECE computed from already-aggregated bin statistics."""

    stats = BinStats(
        counts=np.asarray(list(counts), dtype=int),
        pred_rates=np.asarray(list(pred_rates), dtype=float),
        true_rates=np.asarray(list(true_rates), dtype=float),
    )
    return stats.ece()

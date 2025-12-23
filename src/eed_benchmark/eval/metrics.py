"""Lightweight metric helpers for the EED benchmark.

The original implementation avoided heavy dependencies such as scikit-learn
while still providing the small collection of metrics reported in the paper.
This module mirrors that behaviour closely:

* ROC-AUC via the Mann-Whitney formulation (with tie handling)
* PR-AUC via accumulated precision deltas
* Brier score, optionally using bin-averaged probabilities
* Expected Calibration Error from either raw samples or bin statistics

All functions operate on NumPy arrays and return ``nan`` when a metric is
ill-defined (for example, AUROC with a single class present).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _safe_counts(labels: np.ndarray) -> Tuple[int, int]:
    """Return the number of positive and negative labels."""

    labels = np.asarray(labels, dtype=int)
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    return pos, neg


# Use the Mann–Whitney formulation and average ranks for tied scores so AUROC stays stable without sklearn
def roc_auc_score_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve using the Mann-Whitney U equivalence."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos, neg = _safe_counts(y_true)
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    _, inverse, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            mask = inverse == idx
            ranks[mask] = ranks[mask].mean()

    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


# Sweep thresholds by sorting descending scores and accumulate [precision * delta_recall] to approximate PR-AUC.
def pr_auc_score_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve by cumulative deltas."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos, neg = _safe_counts(y_true)
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    truth_sorted = y_true[order]
    tp = fp = 0
    prev_recall = 0.0
    auc = 0.0
    for truth in truth_sorted:
        if truth == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / pos
        precision = tp / max(1, tp + fp)
        auc += precision * (recall - prev_recall)
        prev_recall = recall
    return float(auc)


def brier_score_prob(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Classic Brier score between labels and probability estimates."""

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def brier_score_binned(
    y_true: np.ndarray, y_score: np.ndarray, bins: int = 10
) -> float:
    """Brier score using empirical bin means as probability estimates."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    y_prob = np.zeros_like(y_score, dtype=float)

    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == bins - 1:
            mask = (y_score >= left) & (y_score <= right)
        else:
            mask = (y_score >= left) & (y_score < right)
        if mask.any():
            y_prob[mask] = y_true[mask].mean()

    return brier_score_prob(y_true, y_prob)


# Equal-width bins over [0,1]; last bin is inclusive on the right edge so perfect scores aren't dropped.
def calibration_ece_binned(
    y_true: np.ndarray, y_score: np.ndarray, bins: int = 10
) -> float:
    """Expected Calibration Error from raw samples via equal-width bins."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = y_true.size
    ece = 0.0
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == bins - 1:
            mask = (y_score >= left) & (y_score <= right)
        else:
            mask = (y_score >= left) & (y_score < right)
        count = mask.sum()
        if count == 0:
            continue
        true_rate = float(y_true[mask].mean())
        pred_rate = float(np.clip(y_score[mask].mean(), 0.0, 1.0))
        ece += (count / total) * abs(pred_rate - true_rate)
    return float(ece)


def ece_from_bin_rates(
    bin_counts: np.ndarray, pred_rates: np.ndarray, true_rates: np.ndarray
) -> float:
    """Expected Calibration Error given pre-computed bin statistics."""

    bin_counts = np.asarray(bin_counts, dtype=float)
    pred_rates = np.asarray(pred_rates, dtype=float)
    true_rates = np.asarray(true_rates, dtype=float)

    total = bin_counts.sum()
    if total == 0:
        return 0.0
    weights = bin_counts / total
    return float(np.sum(weights * np.abs(pred_rates - true_rates)))

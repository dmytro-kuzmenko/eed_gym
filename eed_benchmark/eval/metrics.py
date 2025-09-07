#!/usr/bin/env python3
"""
Lightweight evaluation metrics without external deps:
- AUROC (rank-based)
- PR-AUC (precision-recall curve area)
- Expected Calibration Error (ECE) via binning
- Brier score with optional bin-averaged probabilities

We assume a binary label y_true in {0,1} and continuous scores y_score.
For calibration in this project, we compare refusal rates to should-refuse
rates per risk bin; for Brier we offer a bin-conditional probability variant.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np


def _safe_counts(labels: np.ndarray) -> Tuple[int, int]:
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    return pos, neg


def roc_auc_score_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC via Mann-Whitney U equivalence (no ties correction).
    Returns nan if only one class present.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos, neg = _safe_counts(y_true)
    if pos == 0 or neg == 0:
        return float("nan")
    # ranks of scores (average ranks for ties)
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # tie correction: average ranks for ties
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    for c in np.unique(inv):
        mask = inv == c
        if counts[c] > 1:
            ranks[mask] = ranks[mask].mean()
    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def pr_auc_score_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC by sorting by score and summing precision deltas.
    Returns nan if only one class present.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos, neg = _safe_counts(y_true)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    tp = 0
    fp = 0
    prev_recall = 0.0
    auc = 0.0
    for i in range(y_true_sorted.size):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / pos
        precision = tp / max(1, tp + fp)
        d_recall = recall - prev_recall
        auc += precision * d_recall
        prev_recall = recall
    return float(auc)


def calibration_ece_binned(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> float:
    """Expected Calibration Error (ECE) using equal-width bins over score.
    For each bin, compares predicted rate (mean of y_pred_proxy) to empirical
    true rate. Here we use refusal rate as the predicted prob and should-refuse
    rate as the true prob when passed appropriately.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = y_true.size
    for i in range(bins):
        mask = (y_score >= bin_edges[i]) & (y_score < bin_edges[i + 1] if i < bins - 1 else y_score <= bin_edges[i + 1])
        m = int(mask.sum())
        if m == 0:
            continue
        true_rate = float(y_true[mask].mean())
        # The caller is expected to pass as y_score the binning variable (e.g., risk)
        # and compute predicted rate separately; however to keep this standalone,
        # we define ECE over a pair of rates supplied externally if needed.
        # In common use below, we recompute ECE from bin-wise rates.
        # Here we treat y_score as the probability estimate for y_true: fall back.
        pred_rate = float(np.clip(y_score[mask].mean(), 0.0, 1.0))
        ece += (m / n) * abs(pred_rate - true_rate)
    return float(ece)


def ece_from_bin_rates(bin_counts: np.ndarray, pred_rates: np.ndarray, true_rates: np.ndarray) -> float:
    """ECE when bin-wise predicted and true rates are available."""
    n = int(bin_counts.sum())
    if n == 0:
        return 0.0
    weights = bin_counts / n
    return float(np.sum(weights * np.abs(pred_rates - true_rates)))


def brier_score_prob(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def brier_score_binned(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> float:
    """Brier score using bin-average probabilities as predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    y_prob = np.zeros_like(y_score)
    for i in range(bins):
        mask = (y_score >= bin_edges[i]) & (y_score < bin_edges[i + 1] if i < bins - 1 else y_score <= bin_edges[i + 1])
        if mask.any():
            y_prob[mask] = y_true[mask].mean()
    return brier_score_prob(y_true, y_prob)
#!/usr/bin/env python3
"""Minimal metrics for binary labels and continuous scores."""

from __future__ import annotations
from typing import Tuple
import numpy as np


def _safe_counts(labels: np.ndarray) -> Tuple[int, int]:
    labels = np.asarray(labels).astype(int)
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    return pos, neg


def _rankdata_tieavg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, x.size + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for j, c in enumerate(counts):
        if c > 1:
            m = inv == j
            ranks[m] = ranks[m].mean()
    return ranks


def roc_auc_score_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos, neg = _safe_counts(y_true)
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = _rankdata_tieavg(y_score)
    rank_sum_pos = float(ranks[y_true == 1].sum())
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def pr_auc_score_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos, neg = _safe_counts(y_true)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    tp = fp = 0
    prev_recall = 0.0
    auc = 0.0
    for yi in y:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / pos
        precision = tp / max(1, tp + fp)
        auc += precision * (recall - prev_recall)
        prev_recall = recall
    return float(auc)


def calibration_ece_binned(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    n = y_true.size
    ece = 0.0
    for i in range(bins):
        if i < bins - 1:
            m = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        else:
            m = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        cnt = int(m.sum())
        if cnt == 0:
            continue
        true_rate = float(y_true[m].mean())
        pred_rate = float(np.clip(y_prob[m].mean(), 0.0, 1.0))
        ece += (cnt / n) * abs(pred_rate - true_rate)
    return float(ece)


def ece_from_bin_rates(bin_counts: np.ndarray, pred_rates: np.ndarray, true_rates: np.ndarray) -> float:
    bin_counts = np.asarray(bin_counts, dtype=float)
    pred_rates = np.asarray(pred_rates, dtype=float)
    true_rates = np.asarray(true_rates, dtype=float)
    n = float(bin_counts.sum())
    if n <= 0:
        return 0.0
    weights = bin_counts / n
    return float(np.sum(weights * np.abs(pred_rates - true_rates)))


def brier_score_prob(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def brier_score_binned(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    y_prob = np.zeros_like(y_score, dtype=float)
    for i in range(bins):
        if i < bins - 1:
            m = (y_score >= edges[i]) & (y_score < edges[i + 1])
        else:
            m = (y_score >= edges[i]) & (y_score <= edges[i + 1])
        if m.any():
            y_prob[m] = float(y_true[m].mean())
    return brier_score_prob(y_true, y_prob)

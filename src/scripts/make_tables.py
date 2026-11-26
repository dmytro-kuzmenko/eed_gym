"""Render the JSON written by ``reproduce.sh`` as Markdown tables.

    python src/scripts/make_tables.py artifacts/results > src/results/v1.1/tables.md
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t

LEARNED = [
    ("ppo", "Vanilla PPO"),
    ("ppo_lstm", "PPO-LSTM"),
    ("ppo_masked", "Masked PPO"),
    ("ppo_lagrangian", "Lagrangian PPO"),
]
ABLATIONS = [
    ("ppo", "Vanilla PPO"),
    ("ppo_noaffect", "No Affect"),
    ("ppo_noclarifyalt", "No Clarify/Alt"),
    ("ppo_nocurriculum", "No Curriculum"),
    ("ppo_notrustpenalty", "No Trust Penalty"),
]
HEURISTICS = [
    ("always_comply", "Always-Comply"),
    ("refuse_risky", "Risk-Refusal"),
    ("threshold", "Valence-Threshold"),
    ("vignette_gate", "Vignette-Gate"),
]
MAIN = [
    ("unsafe_rate", "Unsafe %", 100.0, 1),
    ("mean_refusals", "Refusals/ep", 1.0, 1),
    ("f1", "F1", 1.0, 2),
    ("mean_trust", "Trust", 1.0, 2),
    ("trust_in_band", "Trust in band", 1.0, 2),
]
CALIBRATION = [
    ("calibration_rho", "Spearman rho", 1.0, 3),
    ("brier", "Brier", 1.0, 3),
    ("auroc", "AUROC", 1.0, 3),
    ("pr_auc", "PR-AUC", 1.0, 3),
]
HEURISTIC_COLS = [
    ("mean_reward", "Mean reward", 1.0, 1),
    ("unsafe_rate", "Unsafe %", 100.0, 1),
    ("mean_refusals", "Refusals/ep", 1.0, 1),
    ("justified_ratio", "Justified ratio", 1.0, 2),
    ("f1", "F1", 1.0, 2),
    ("calibration_rho", "Spearman rho", 1.0, 2),
    ("mean_trust", "Mean trust", 1.0, 2),
]


def mean_ci(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return float("nan"), 0.0
    half = t.ppf(0.975, n - 1) * arr.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    return float(arr.mean()), float(half)


def load_id(results: Path, name: str):
    """ID results: metric -> (mean, 95% CI half-width across seeds)."""

    path = results / f"id_{name}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if "model" in data:  # single summary (heuristic)
        out = {}
        for key, value in data.items():
            if isinstance(value, (int, float)):
                lo, hi = data.get("ci95", {}).get(key, (value, value))
                out[key] = (value, (hi - lo) / 2.0)
        return out
    return {key: (stats["mean"], stats["ci95"]) for key, stats in data.items()}


def load_st(results: Path, name: str):
    """ST results: mean over the persona x stressor grid, then across seeds."""

    path = results / f"st_{name}.json"
    if not path.is_file():
        return None
    ckpts = json.loads(path.read_text())["checkpoints"]
    keys = ckpts[0]["mean_metrics"].keys()
    return {key: mean_ci([c["mean_metrics"][key] for c in ckpts]) for key in keys}


def table(title, rows, cols, loader):
    lines = [f"### {title}", "", "| Model | " + " | ".join(c[1] for c in cols) + " |"]
    lines.append("|---|" + "---|" * len(cols))
    for name, label in rows:
        data = loader(name)
        if data is None:
            continue
        cells = []
        for key, _, scale, digits in cols:
            mean, half = data[key]
            cells.append(f"{mean * scale:.{digits}f} ± {half * scale:.{digits}f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/results")

    def id_(name):
        return load_id(results, name)

    def st_(name):
        return load_st(results, name)

    print("# EED Gym v1.1 results\n")
    print(
        "Learned policies: mean ± 95% CI half-width across 5 training seeds, "
        "100 deterministic episodes per evaluation. Heuristics: one run, "
        "± is the bootstrap 95% half-width over episodes (ID only).\n"
    )
    print(table("ID, baselines", LEARNED, MAIN, id_))
    print(table("Stress test, baselines", LEARNED, MAIN, st_))
    print(table("ID, ablations", ABLATIONS, MAIN, id_))
    print(table("Stress test, ablations", ABLATIONS, MAIN, st_))
    print(table("ID, calibration", LEARNED + ABLATIONS[1:], CALIBRATION, id_))
    print(table("ID, heuristics", HEURISTICS + LEARNED[:1], HEURISTIC_COLS, id_))
    print(table("Stress test, heuristics", HEURISTICS, MAIN, st_))


if __name__ == "__main__":
    main()

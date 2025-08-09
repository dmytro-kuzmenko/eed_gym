#!/usr/bin/env python3
"""
vignette_fit.py – Process Google Form CSV responses and produce:
  1) Summary stats for appropriateness, safety, trust, empathy, blame by response_type
  2) A simple human-grounded refusal model: logistic P(should_refuse | perceived_risk)
  3) Suggested environment RewardWeights tweaks derived from group deltas

Input CSV columns (case-insensitive names accepted):
  - pid: participant id (string)
  - scenario_id: 1..10 (int)
  - response_type: comply | refuse_empathic | refuse_constructive
  - appropriateness, safety, trust, empathy, blame_robot: Likert 1..7 (ints)
  - perceived_risk: Likert 1..7 (int)
  - comprehension: Yes/No (string)

Usage:
  python vignette_fit.py --csv data/vignettes.csv --out out/params_human.json --print
"""
from __future__ import annotations
import argparse, csv, json, math
from typing import Dict, Any, List
import numpy as np


RESP_TYPES = ["comply", "refuse_empathic", "refuse_constructive"]


def _norm01(x):
    return (x - 1.0) / 6.0


def load_responses(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            d = {k.strip().lower(): v for k, v in r.items()}
            comp = d.get("comprehension", "yes").strip().lower()
            if comp in ("no", "0", "false"):
                continue
            try:
                rows.append({
                    "pid": d.get("pid", ""),
                    "scenario_id": int(d.get("scenario_id", "0") or 0),
                    "response_type": d.get("response_type", "").strip().lower(),
                    "appropriateness": float(d.get("appropriateness", "nan")),
                    "safety": float(d.get("safety", "nan")),
                    "trust": float(d.get("trust", "nan")),
                    "empathy": float(d.get("empathy", "nan") or 0.0),
                    "blame_robot": float(d.get("blame_robot", "nan")),
                    "perceived_risk": float(d.get("perceived_risk", "nan")),
                })
            except Exception:
                continue
    # filter to known response types
    rows = [r for r in rows if r["response_type"] in RESP_TYPES]
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for rt in RESP_TYPES:
        sel = [r for r in rows if r["response_type"] == rt]
        if not sel:
            continue
        arr = {k: np.array([x[k] for x in sel], dtype=float) for k in ["appropriateness","safety","trust","empathy","blame_robot","perceived_risk"]}
        out[rt] = {f"mean_{k}": float(np.nanmean(v)) for k, v in arr.items()}
    return out


def fit_logistic_perceived_risk(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Fit logistic P(should_refuse=1 | perceived_risk).
    Define label from perceived_risk >= 5 as positive.
    """
    risks = np.array([r["perceived_risk"] for r in rows], dtype=float)
    x = _norm01(risks)
    y = np.array([1.0 if r >= 5.0 else 0.0 for r in risks], dtype=float)
    # add intercept
    X = np.stack([np.ones_like(x), x], axis=1)
    w = np.zeros(2, dtype=float)
    # Newton-Raphson
    for _ in range(25):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        # avoid singularity
        W = p * (1 - p) + 1e-6
        # gradient and Hessian
        g = X.T @ (y - p)
        H = -(X.T * W) @ X
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if np.linalg.norm(w_new - w) < 1e-6:
            w = w_new
            break
        w = w_new
    return {"intercept": float(w[0]), "coef_risk": float(w[1])}


def suggest_reward_weights(stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Heuristic mapping from group means to environment weights.
    Conservative scaling to small magnitudes.
    """
    trust_c = stats.get("comply", {}).get("mean_trust", 0.0)
    trust_e = stats.get("refuse_empathic", {}).get("mean_trust", trust_c)
    trust_k = stats.get("refuse_constructive", {}).get("mean_trust", trust_c)
    emp_e = stats.get("refuse_empathic", {}).get("mean_empathy", 0.0)
    emp_k = stats.get("refuse_constructive", {}).get("mean_empathy", 0.0)

    # Refusal penalty from trust drop vs comply (average across refusal styles)
    refusal_trust_drop = max(0.0, (trust_c - 0.5*(trust_e + trust_k)))
    refusal_weight = min(0.5, 0.25 + 0.5 * refusal_trust_drop / 7.0)

    # Explanation trust boost is the positive delta between refusal styles and comply
    # Use empathic as reference if higher
    exp_trust_boost = max(0.0, max(trust_e, trust_k) - min(trust_e, trust_k))
    exp_trust_boost = min(0.1, exp_trust_boost / 50.0)

    # Style bonuses from perceived empathy difference
    de = max(0.0, emp_e - emp_k) / 100.0
    dk = max(0.0, emp_k - emp_e) / 100.0

    return {
        "refusal": float(refusal_weight),
        "explanation_trust_boost": float(exp_trust_boost),
        "empathetic_style_bonus": float(0.03 + de),
        "constructive_style_bonus": float(0.03 + dk),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="human_params.json")
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()

    rows = load_responses(args.csv)
    if len(rows) == 0:
        raise SystemExit("No valid rows found.")
    stats = summarize(rows)
    logit = fit_logistic_perceived_risk(rows)
    weights = suggest_reward_weights(stats)

    out = {
        "n": len(rows),
        "stats_by_response_type": stats,
        "human_model": {"logistic": logit},
        "suggested_reward_weights": weights,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    if args.print:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()



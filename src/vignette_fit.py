#!/usr/bin/env python3
"""
vignette_fit.py – Process Google Form CSV responses and produce:
  1) Summary stats for appropriateness, safety, trust, empathy, blame by response_type
  2) A simple human-grounded refusal model: logistic P(appropriate>=5 | perceived_risk, response_style)
  3) Suggested environment RewardWeights tweaks derived from group deltas

Accepts EITHER:
  • Tidy/long CSV (e.g., vignette_clean_130.csv) with columns:
      participant, scenario, Response_clean_inferred, Appropriateness, Perceived safety,
      Trust in the robot, Perceived empathy, Who is to blame if something goes wrong?,
      I perceived the task as risky, I understood the scenario
  • Raw Google Forms wide export (repeated blocks with .1, .2, ... suffixes and
    'Select any option to your liking' containing the 'Response' / 'Response​' / 'Response​​' variants)

Usage:
  python vignette_fit.py --csv data/vignettes.csv --out out/params_human.json --print
  # optional:
  python vignette_fit.py --csv data/vignettes_raw.csv --save-long out/vignette_long.csv
"""
from __future__ import annotations
import argparse, csv, json, math, sys, re
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# ------------------------------
# Config / canonical names
# ------------------------------
RESP_TYPES = ["comply", "refuse_empathic", "refuse_constructive"]

# Canonical column names we will output/use internally
COL_MAP_CANON = {
    "participant": "participant",
    "scenario": "scenario",
    "response": "Response",  # raw response string (may contain zero-width chars)
    "response_clean": "Response_clean_inferred_inferred",
    "appropriateness": "Appropriateness",
    "safety": "Perceived safety",
    "trust": "Trust in the robot",
    "empathy": "Perceived empathy",
    "blame": "Who is to blame if something goes wrong?",
    "risk": "I perceived the task as risky",
    "comprehension": "I understood the scenario",
}

# Google Forms question stems we look for (case-insensitive match)
GF_Q_BASES = [
    "Appropriateness",
    "Perceived safety",
    "Trust in the robot",
    "Perceived empathy",
    "Who is to blame if something goes wrong?",
    "I perceived the task as risky",
    "I understood the scenario",
    "Select any option to your liking",
]

# Map the weird Response strings to semantic classes
def map_response_to_type(s: str) -> str | None:
    """
    Map 'Response' variants from Google Forms (with invisible ZWSP chars) to canonical response types.
      'Response'            -> comply
      'Response\u200b'      -> refuse_empathic
      'Response\u200b\u200b'-> refuse_constructive
    """
    if s is None:
        return None
    s = str(s).strip()
    # Normalize by removing regular spaces but keep zero-width detection
    # Count ZERO-WIDTH SPACE (\u200b)
    zw = s.count("\u200b")
    base = s.replace("\u200b", "")
    if base != "Response":
        # Some tidy files store already-clean labels
        s_lower = s.lower()
        if "construct" in s_lower:
            return "refuse_constructive"
        if "empath" in s_lower:
            return "refuse_empathic"
        if "comply" in s_lower:
            return "comply"
        return None
    if zw == 0:
        return "comply"
    elif zw == 1:
        return "refuse_empathic"
    elif zw >= 2:
        return "refuse_constructive"
    return None

def _norm01(x):
    return (x - 1.0) / 6.0

# ------------------------------
# Loaders
# ------------------------------
def detect_is_tidy(df: pd.DataFrame) -> bool:
    """Heuristic: tidy if it already has 'participant' and 'scenario' and 'Response_clean_inferred' columns."""
    cols = {c.lower() for c in df.columns}
    return (
        "participant" in cols and
        "scenario" in cols and
        ("response_clean" in cols or "response" in cols)
    )

def load_tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical labels if needed and return a tidy long DataFrame."""
    # Make a lowercase map for robustness
    lower_map = {c.lower(): c for c in df.columns}
    def getcol(name):
        # return actual column name by case-insensitive lookup; fall back to given
        return lower_map.get(name.lower(), name)

    # Ensure Response_clean_inferred exists
    if "response_clean" not in {c.lower() for c in df.columns}:
        rc = df[getcol("Response")].map(map_response_to_type)
        df["Response_clean_inferred"] = rc
    else:
        # Normalize any weird casing/spacing
        df["Response_clean_inferred"] = df[getcol("Response_clean_inferred")].map(map_response_to_type)

    # Standardize comprehension to Yes/No where possible (not strictly needed)
    if "I understood the scenario" in df.columns:
        pass

    # Keep only relevant columns if present
    keep = []
    for k in ["participant","scenario","Response","Response_clean_inferred",
              "Appropriateness","Perceived safety","Trust in the robot",
              "Perceived empathy","Who is to blame if something goes wrong?",
              "I perceived the task as risky","I understood the scenario"]:
        if k in df.columns:
            keep.append(k)
    return df[keep].copy()

def load_from_google_forms_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse raw Google Forms export where each vignette appears as a repeated block with optional .1, .2 suffixes.
    Returns a tidy long DataFrame with one row per (participant, scenario).
    """
    # Build map: scenario index -> dict(question_base -> column_name)
    vignette_blocks: Dict[int, Dict[str, str]] = {}
    for col in df.columns:
        m = re.match(r"^(.*?)(?:\.(\d+))?$", col)
        if not m:
            continue
        base, idx = m.groups()
        base = base.strip()
        if any(base.lower() == b.lower() for b in GF_Q_BASES):
            vidx = int(idx) if idx else 0
            vignette_blocks.setdefault(vidx, {})[base] = col

    # Reshape
    records = []
    for pid, row in df.iterrows():
        for vidx in sorted(vignette_blocks.keys()):
            cols = vignette_blocks[vidx]
            rec = {
                "participant": pid,
                "scenario": vidx,
            }
            # Pull values (if missing, leave None)
            for q in GF_Q_BASES:
                # find case-insensitive match in cols
                target = None
                for k in cols:
                    if k.lower() == q.lower():
                        target = cols[k]
                        break
                if target and target in df.columns:
                    rec[q] = row[target]
                else:
                    rec[q] = None
            records.append(rec)

    long_df = pd.DataFrame(records)

    # Map response to clean type
    long_df["Response_clean_inferred"] = long_df["Select any option to your liking"].map(map_response_to_type)

    # Drop vignettes with no ratings (all NaN Likert)
    if "Appropriateness" in long_df.columns:
        long_df = long_df.dropna(subset=["Appropriateness"], how="all")

    return long_df

def load_any(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if detect_is_tidy(df):
        return load_tidy(df)
    return load_from_google_forms_wide(df)

# ------------------------------
# Analytics
# ------------------------------
def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with a valid response type (A/B/C mapped)."""
    df = df.copy()
    df = df[~df["Response_clean_inferred"].isna()]
    return df

def summarize(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    # Coerce numeric
    for k in ["Appropriateness","Perceived safety","Trust in the robot",
              "Perceived empathy","Who is to blame if something goes wrong?",
              "I perceived the task as risky"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    for label, resp_type in [("comply","comply"),
                             ("refuse_empathic","refuse_empathic"),
                             ("refuse_constructive","refuse_constructive")]:
        sel = df[df["Response_clean_inferred"] == resp_type]
        if sel.empty:
            continue
        out[label] = {
            "mean_appropriateness": float(sel["Appropriateness"].mean()),
            "mean_safety":          float(sel["Perceived safety"].mean()),
            "mean_trust":           float(sel["Trust in the robot"].mean()),
            "mean_empathy":         float(sel["Perceived empathy"].mean()),
            "mean_blame_robot":     float(sel["Who is to blame if something goes wrong?"].mean()),
            "mean_perceived_risk":  float(sel["I perceived the task as risky"].mean()),
            "n": int(len(sel)),
        }
    return out

def fit_logistic(df: pd.DataFrame) -> Dict[str, float]:
    """
    Logistic P(appropriate>=5 | risk_norm, response_style).
    style encoded via two dummies: empathic, constructive (baseline=comply).
    Newton-Raphson with L2 ridge to stabilize on small N.
    """
    # Numeric prep
    y = (pd.to_numeric(df["Appropriateness"], errors="coerce") >= 5).astype(float).values
    risk = pd.to_numeric(df["I perceived the task as risky"], errors="coerce").values
    # Normalize risk
    mu = np.nanmean(risk); sd = np.nanstd(risk) + 1e-8
    x_risk = (risk - mu) / sd

    # Style dummies
    style = df["Response_clean_inferred"].values
    empathic = (style == "refuse_empathic").astype(float)
    constructive = (style == "refuse_constructive").astype(float)

    X = np.stack([np.ones_like(x_risk), x_risk, empathic, constructive], axis=1)
    # Remove rows with NaNs
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]; y = y[mask]

    # Newton-Raphson with ridge
    w = np.zeros(X.shape[1])
    ridge = 1e-3
    for _ in range(50):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        W = p * (1 - p) + 1e-9
        g = X.T @ (y - p) - ridge * w
        H = -(X.T * W) @ X - ridge * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if np.linalg.norm(w_new - w) < 1e-6:
            w = w_new
            break
        w = w_new

    return {
        "intercept": float(w[0]),
        "coef_risk_norm": float(w[1]),
        "coef_empathic": float(w[2]),
        "coef_constructive": float(w[3]),
        "n": int(len(y)),
        "risk_norm_mean": float(mu),
        "risk_norm_std": float(sd),
    }

def suggest_reward_weights(stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Heuristic mapping from group means to environment weights (conservative magnitudes)."""
    trust_c = stats.get("comply", {}).get("mean_trust", 0.0)
    trust_e = stats.get("refuse_empathic", {}).get("mean_trust", trust_c)
    trust_k = stats.get("refuse_constructive", {}).get("mean_trust", trust_c)
    emp_e   = stats.get("refuse_empathic", {}).get("mean_empathy", 0.0)
    emp_k   = stats.get("refuse_constructive", {}).get("mean_empathy", 0.0)

    # Refusal penalty from trust drop vs comply (average across refusal styles)
    refusal_trust_drop = max(0.0, (trust_c - 0.5*(trust_e + trust_k)))
    refusal_weight = min(0.5, 0.25 + 0.5 * refusal_trust_drop / 7.0)

    # Explanation trust boost: difference between the two refusal styles (very small)
    exp_trust_boost = max(0.0, abs(trust_e - trust_k)) / 50.0
    exp_trust_boost = float(min(0.1, exp_trust_boost))

    # Style bonuses from perceived empathy differences
    de = max(0.0, emp_e - emp_k) / 100.0
    dk = max(0.0, emp_k - emp_e) / 100.0

    return {
        "refusal": float(refusal_weight),
        "explanation_trust_boost": float(exp_trust_boost),
        "empathetic_style_bonus": float(0.03 + de),
        "constructive_style_bonus": float(0.03 + dk),
    }

# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to raw Google Forms CSV or tidy vignette_clean CSV")
    ap.add_argument("--out", default="human_params.json", help="Path to write summary JSON")
    ap.add_argument("--print", action="store_true", help="Print JSON to stdout")
    ap.add_argument("--save-long", default="", help="Optional path to save normalized long CSV")
    args = ap.parse_args()

    df_raw = pd.read_csv(args.csv)
    df_long = load_any(args.csv)  # auto-detects format and reshapes if needed

    # Filter to valid responses (A/B/C mapped)
    df_long = filter_valid_rows(df_long)

    # Summaries
    stats = summarize(df_long)
    logit = fit_logistic(df_long)
    weights = suggest_reward_weights(stats)

    out = {
        "n_rows": int(len(df_long)),
        "stats_by_response_type": stats,
        "human_model": {"logistic": logit},
        "suggested_reward_weights": weights,
    }

    if args.save_long:
        df_long.to_csv(args.save_long, index=False)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    if args.print:
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vignette_effects.py
Compute ANOVA, pairwise Cohen's d/Hedges' g with BCa CIs, and post-hoc power
for the vignette study (e.g., trust across Compliance vs Empathic vs Constructive).

Usage examples:
  # Long-format CSV (one row per vignette rating)
  python vignette_effects.py --csv vignette.csv --dv trust --between condition --id participant_id --export results_trust.csv

  # Wide-format CSV: multiple trust_* columns (one row per participant)
  python vignette_effects.py --csv vignette_wide.csv --dv trust --between condition --id participant_id \
      --wide-dv-regex '^trust_' --wide-cond-map 'trust_Compliance:Compliance,trust_Empathic:Empathic,trust_Constructive:Constructive' \
      --export results_trust.csv
"""
from __future__ import annotations
import argparse, re, sys, json
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.power import TTestIndPower

# ----------------------------
# Helpers: effect sizes & CIs
# ----------------------------

def cohens_d(x: np.ndarray, y: np.ndarray, use_unbiased: bool = False) -> float:
    """Cohen's d for independent samples (pooled SD). If use_unbiased=True, returns Hedges' g."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = x.size, y.size
    mx, my = np.nanmean(x), np.nanmean(y)
    sx, sy = np.nanvar(x, ddof=1), np.nanvar(y, ddof=1)
    # pooled SD
    s_p2 = ((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2)
    d = (mx - my) / np.sqrt(s_p2)
    if use_unbiased:
        # Hedges' correction J
        J = 1 - (3 / (4*(nx + ny) - 9))
        d = d * J
    return d

def _bca_ci(samples: np.ndarray, stat_fn, alpha=0.05, n_boot=10000, random_state=0) -> Tuple[float, float]:
    """BCa CI for a statistic via bootstrap on provided data rows (2-group resampling).
    samples: list of (group_label, value) rows; we’ll resample within each group to preserve n.
    """
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame(samples, columns=["grp", "val"])
    groups = [g for g, _ in df.groupby("grp")]
    # original stat
    d0 = stat_fn(df)
    # bootstrap
    boot_stats = []
    grouped = dict(tuple(df.groupby("grp")))
    for _ in range(n_boot):
        resampled = []
        for g in groups:
            vals = grouped[g]["val"].to_numpy()
            res_idx = rng.integers(0, len(vals), size=len(vals))
            resampled.extend([(g, v) for v in vals[res_idx]])
        boot_stats.append(stat_fn(pd.DataFrame(resampled, columns=["grp", "val"])))
    boot_stats = np.array(boot_stats)
    # Bias-correction z0
    prop = np.mean(boot_stats < d0)
    prop = np.clip(prop, 1e-6, 1-1e-6)
    z0 = stats.norm.ppf(prop)
    # Acceleration (jackknife)
    jack = []
    for i in range(len(df)):
        jack_df = df.drop(df.index[i])
        jack.append(stat_fn(jack_df))
    jack = np.array(jack)
    jack_mean = np.mean(jack)
    num = np.sum((jack_mean - jack)**3)
    den = 6.0 * (np.sum((jack_mean - jack)**2) ** 1.5 + 1e-12)
    a = num / den if den != 0 else 0.0
    # Adjusted quantiles
    z_alpha1 = stats.norm.ppf(alpha/2)
    z_alpha2 = stats.norm.ppf(1 - alpha/2)
    def adj(qz):  # adjusted percentile
        return stats.norm.cdf(z0 + (z0 + qz) / (1 - a*(z0 + qz)))
    lo = np.quantile(boot_stats, adj(z_alpha1))
    hi = np.quantile(boot_stats, adj(z_alpha2))
    return float(lo), float(hi)

def effect_with_ci(x: np.ndarray, y: np.ndarray, alpha=0.05, n_boot=10000, unbiased=True, random_state=0):
    """Return dict with d, g, BCa CI for chosen metric (Hedges' g by default)."""
    def d_stat(df_xy):
        x_vals = df_xy.loc[df_xy["grp"] == "X", "val"].to_numpy()
        y_vals = df_xy.loc[df_xy["grp"] == "Y", "val"].to_numpy()
        return cohens_d(x_vals, y_vals, use_unbiased=unbiased)
    d_or_g = cohens_d(x, y, use_unbiased=unbiased)
    samples = np.concatenate([np.c_[np.full(x.size, "X"), x],
                              np.c_[np.full(y.size, "Y"), y]])
    lo, hi = _bca_ci(samples, d_stat, alpha=alpha, n_boot=n_boot, random_state=random_state)
    return {"hedges_g" if unbiased else "cohens_d": float(d_or_g),
            "ci_low": lo, "ci_high": hi}

# ----------------------------
# ANOVA + pairwise comparisons
# ----------------------------

def one_way_anova(df: pd.DataFrame, dv: str, between: str) -> Dict:
    model = ols(f"{dv} ~ C({between})", data=df).fit()
    anova = anova_lm(model, typ=2)
    # eta-squared
    ss_between = anova.loc[f"C({between})", "sum_sq"]
    ss_total = ss_between + anova.loc["Residual", "sum_sq"]
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
    return {
        "anova_table": anova.reset_index().rename(columns={"index":"term"}).to_dict(orient="records"),
        "eta_squared": float(eta_sq)
    }

def pairwise_effects(df: pd.DataFrame, dv: str, between: str, alpha=0.05, n_boot=20000, unbiased=True, seed=0):
    levels = df[between].dropna().unique().tolist()
    out = []
    for i in range(len(levels)):
        for j in range(i+1, len(levels)):
            g1, g2 = levels[i], levels[j]
            x = df.loc[df[between]==g1, dv].dropna().to_numpy()
            y = df.loc[df[between]==g2, dv].dropna().to_numpy()
            eff = effect_with_ci(x, y, alpha=alpha, n_boot=n_boot, unbiased=unbiased, random_state=seed)
            # two-sample Welch t-test for p-value (robust to unequal variances)
            t, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
            out.append({
                "dv": dv, "group_a": g1, "n_a": int(x.size), "mean_a": float(np.mean(x)), "sd_a": float(np.std(x, ddof=1)),
                "group_b": g2, "n_b": int(y.size), "mean_b": float(np.mean(y)), "sd_b": float(np.std(y, ddof=1)),
                "t": float(t), "p_val": float(p),
                **eff
            })
    return pd.DataFrame(out)

# ----------------------------
# Power analysis (post-hoc)
# ----------------------------

def power_two_sample_t(effect_size: float, n1: int, n2: int, alpha=0.05, ratio=None, alternative="two-sided") -> float:
    """Post-hoc power for independent t-test using Cohen's d/Hedges' g as effect_size."""
    ratio = ratio if ratio is not None else n2 / max(n1, 1)
    analysis = TTestIndPower()
    return float(analysis.power(effect_size=abs(effect_size), nobs1=n1, alpha=alpha, ratio=ratio, alternative=alternative))

# ----------------------------
# Wide -> long reshaping support
# ----------------------------

def maybe_melt_wide(df: pd.DataFrame, dv: str, id_col: str, wide_dv_regex: str, wide_cond_map: str) -> pd.DataFrame:
    """
    If wide_dv_regex provided, melt columns matching it into long format.
    wide_cond_map maps column_name:ConditionName pairs, comma-separated.
    Example: 'trust_Compliance:Compliance,trust_Empathic:Empathic,trust_Constructive:Constructive'
    """
    pattern = re.compile(wide_dv_regex)
    dv_cols = [c for c in df.columns if pattern.search(c)]
    if not dv_cols:
        raise ValueError(f"No columns matched regex: {wide_dv_regex}")
    long = df[[id_col] + dv_cols].melt(id_vars=[id_col], var_name="__col", value_name=dv)
    # map to condition
    mapping = {}
    if wide_cond_map:
        for pair in wide_cond_map.split(","):
            k, v = pair.split(":")
            mapping[k.strip()] = v.strip()
    else:
        # default: everything after first underscore is the condition
        for c in dv_cols:
            parts = c.split("_", 1)
            mapping[c] = parts[1] if len(parts) > 1 else c
    long["condition"] = long["__col"].map(mapping)
    long = long.drop(columns="__col")
    return long

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to vignette CSV")
    ap.add_argument("--dv", required=True, help="Dependent variable column name (e.g., trust)")
    ap.add_argument("--between", default="condition", help="Between-subject factor column (default: condition)")
    ap.add_argument("--id", default="participant_id", help="Participant ID column")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--nboot", type=int, default=20000)
    ap.add_argument("--unbiased", action="store_true", help="Report Hedges' g instead of Cohen's d")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", default="", help="Optional: path to save pairwise results CSV")
    # Wide support
    ap.add_argument("--wide-dv-regex", default="", help="Regex to pick DV columns in wide data (if provided, melt to long)")
    ap.add_argument("--wide-cond-map", default="", help="Mapping of wide columns to condition labels, e.g. 'trust_Compliance:Compliance,...'")
    args = ap.parse_args()

    COL_RENAME = {
        "Appropriateness": "appropriateness",
        "Perceived safety": "safety",
        "Trust in the robot": "trust",
        "Perceived empathy": "empathy",
        "Who is to blame if something goes wrong?": "blame",
        "I perceived the task as risky": "risk",
        "I understood the scenario": "understood"
    }

    df = pd.read_csv(args.csv)
    df = df.rename(columns=COL_RENAME)

    if args.wide_dv_regex:
        df = maybe_melt_wide(df, dv=args.dv, id_col=args.id, wide_dv_regex=args.wide_dv_regex, wide_cond_map=args.wide_cond_map)

    # sanity
    for col in [args.dv, args.between, args.id]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Keep only id, between, dv; drop NA dv
    df = df[[args.id, args.between, args.dv]].dropna(subset=[args.dv]).copy()

    # ANOVA
    anova_res = one_way_anova(df, dv=args.dv, between=args.between)

    # Pairwise effects + power
    pw = pairwise_effects(df, dv=args.dv, between=args.between, alpha=args.alpha,
                          n_boot=args.nboot, unbiased=args.unbiased, seed=args.seed)

    # Power per contrast (post-hoc, based on Hedges' g/Cohen's d and actual n)
    powers = []
    for _, row in pw.iterrows():
        eff = row["hedges_g"] if args.unbiased else row["cohens_d"]
        power = power_two_sample_t(eff, int(row["n_a"]), int(row["n_b"]), alpha=args.alpha)
        powers.append(power)
    pw["posthoc_power"] = powers

    # Print summary to stdout (JSON blocks for easy capture)
    print("ANOVA:", json.dumps({
        "dv": args.dv,
        "between": args.between,
        "eta_squared": anova_res["eta_squared"],
        "anova_table": anova_res["anova_table"]
    }, indent=2))
    print("\nPAIRWISE:", pw.to_json(orient="records", indent=2))

    if args.export:
        pw.to_csv(args.export, index=False)
        print(f"\nSaved pairwise results to: {args.export}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

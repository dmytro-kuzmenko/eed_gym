#!/usr/bin/env python3
"""Analyse vignette study outcomes (ANOVA, pairwise effects, power)."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.power import TTestIndPower

DEFAULT_BETWEEN = "condition"
DEFAULT_ID = "participant_id"


@dataclass
class PairwiseEffect:
    dv: str
    group_a: str
    n_a: int
    mean_a: float
    sd_a: float
    group_b: str
    n_b: int
    mean_b: float
    sd_b: float
    t: float
    p_val: float
    hedges_g: float
    ci_low: float
    ci_high: float
    posthoc_power: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_x, n_y = x.size, y.size
    mean_diff = float(np.nanmean(x) - np.nanmean(y))
    var_x = float(np.nanvar(x, ddof=1))
    var_y = float(np.nanvar(y, ddof=1))
    pooled = ((n_x - 1) * var_x + (n_y - 1) * var_y) / max(n_x + n_y - 2, 1)
    return mean_diff / np.sqrt(max(pooled, 1e-12))


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    g = cohens_d(x, y)
    n = x.size + y.size
    correction = 1 - (3.0 / (4 * n - 9)) if n > 2 else 1.0
    return g * correction


def bootstrap_bca(
    samples: pd.DataFrame,
    stat_fn,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = samples.groupby("grp")
    stat0 = stat_fn(samples)
    boot = []
    for _ in range(n_boot):
        resampled_rows = []
        for name, group in groups:
            resampled = group.sample(n=len(group), replace=True, random_state=rng)
            resampled_rows.append(resampled)
        boot.append(stat_fn(pd.concat(resampled_rows, ignore_index=True)))
    boot = np.asarray(boot)

    prop = np.clip(np.mean(boot < stat0), 1e-6, 1 - 1e-6)
    z0 = stats.norm.ppf(prop)

    jack = []
    for i in range(len(samples)):
        jack_df = samples.drop(samples.index[i])
        jack.append(stat_fn(jack_df))
    jack = np.asarray(jack)
    jack_mean = jack.mean()
    numerator = np.sum((jack_mean - jack) ** 3)
    denominator = 6 * (np.sum((jack_mean - jack) ** 2) ** 1.5 + 1e-12)
    acc = numerator / denominator if denominator != 0 else 0.0

    def adjusted_quantile(z: float) -> float:
        return stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))

    z_low = stats.norm.ppf(alpha / 2)
    z_high = stats.norm.ppf(1 - alpha / 2)
    lo = np.quantile(boot, adjusted_quantile(z_low))
    hi = np.quantile(boot, adjusted_quantile(z_high))
    return float(lo), float(hi)


def pairwise_effects(
    df: pd.DataFrame,
    dv: str,
    between: str,
    alpha: float,
    n_boot: int,
    seed: int,
) -> List[PairwiseEffect]:
    levels = df[between].dropna().unique().tolist()
    effects: List[PairwiseEffect] = []
    power_calc = TTestIndPower()

    for i, group_a in enumerate(levels):
        for group_b in levels[i + 1 :]:
            x = df.loc[df[between] == group_a, dv].dropna().to_numpy(dtype=float)
            y = df.loc[df[between] == group_b, dv].dropna().to_numpy(dtype=float)
            if x.size == 0 or y.size == 0:
                continue

            g = hedges_g(x, y)
            samples = pd.concat(
                [
                    pd.DataFrame({"grp": "A", "val": x}),
                    pd.DataFrame({"grp": "B", "val": y}),
                ]
            )
            ci_low, ci_high = bootstrap_bca(samples, lambda df_xy: hedges_g(
                df_xy.loc[df_xy["grp"] == "A", "val"].to_numpy(),
                df_xy.loc[df_xy["grp"] == "B", "val"].to_numpy(),
            ), alpha=alpha, n_boot=n_boot, seed=seed)

            t_stat, p_val = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
            power = float(
                power_calc.power(
                    effect_size=abs(g),
                    nobs1=x.size,
                    alpha=alpha,
                    ratio=y.size / x.size if x.size else 1.0,
                )
            )

            effects.append(
                PairwiseEffect(
                    dv=dv,
                    group_a=group_a,
                    n_a=int(x.size),
                    mean_a=float(np.mean(x)),
                    sd_a=float(np.std(x, ddof=1)),
                    group_b=group_b,
                    n_b=int(y.size),
                    mean_b=float(np.mean(y)),
                    sd_b=float(np.std(y, ddof=1)),
                    t=float(t_stat),
                    p_val=float(p_val),
                    hedges_g=float(g),
                    ci_low=ci_low,
                    ci_high=ci_high,
                    posthoc_power=power,
                )
            )
    return effects


def run_anova(df: pd.DataFrame, dv: str, between: str) -> Dict[str, object]:
    model = ols(f"{dv} ~ C({between})", data=df).fit()
    table = anova_lm(model, typ=2)
    ss_between = table.loc[f"C({between})", "sum_sq"]
    ss_total = ss_between + table.loc["Residual", "sum_sq"]
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else float("nan")
    return {
        "anova": table.reset_index().rename(columns={"index": "term"}).to_dict(orient="records"),
        "eta_squared": eta_sq,
    }


def maybe_melt_wide(
    df: pd.DataFrame,
    dv: str,
    id_col: str,
    regex: str,
    mapping: str,
) -> pd.DataFrame:
    pattern = re.compile(regex)
    dv_cols = [c for c in df.columns if pattern.search(c)]
    if not dv_cols:
        raise ValueError(f"No columns matched regex '{regex}'")

    mapping_dict: Dict[str, str] = {}
    if mapping:
        for pair in mapping.split(","):
            key, value = pair.split(":", 1)
            mapping_dict[key.strip()] = value.strip()
    else:
        for col in dv_cols:
            mapping_dict[col] = col.split("_", 1)[-1]

    long_df = (
        df[[id_col] + dv_cols]
        .melt(id_vars=id_col, var_name="_tmp", value_name=dv)
        .assign(**{DEFAULT_BETWEEN: lambda d: d["_tmp"].map(mapping_dict)})
        .drop(columns="_tmp")
    )
    return long_df


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.csv)
    if args.wide_dv_regex:
        df = maybe_melt_wide(df, args.dv, args.id, args.wide_dv_regex, args.wide_cond_map)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute vignette ANOVA and pairwise effect sizes")
    parser.add_argument("--csv", required=True, help="Path to vignette CSV")
    parser.add_argument("--dv", required=True, help="Dependent variable column (e.g., trust)")
    parser.add_argument("--between", default=DEFAULT_BETWEEN, help="Between-subject factor column")
    parser.add_argument("--id", default=DEFAULT_ID, help="Participant ID column")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--nboot", type=int, default=20_000, help="Number of bootstrap samples for CIs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export", type=Path, help="Optional path to export pairwise CSV")
    parser.add_argument("--json-out", type=Path, help="Optional JSON output path")
    parser.add_argument("--wide-dv-regex", default="", help="Regex to melt wide-format columns")
    parser.add_argument("--wide-cond-map", default="", help="Mapping for wide columns, e.g. 'trust_emp:Empathic'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args)

    anova = run_anova(df, args.dv, args.between)
    effects = pairwise_effects(df, args.dv, args.between, args.alpha, args.nboot, args.seed)

    if args.export:
        out_df = pd.DataFrame([effect.as_dict() for effect in effects])
        out_df.to_csv(args.export, index=False)
        print(f"Saved pairwise results to {args.export}")

    payload = {
        "anova": anova,
        "pairwise": [effect.as_dict() for effect in effects],
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"Saved JSON summary to {args.json_out}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
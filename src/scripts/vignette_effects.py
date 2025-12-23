#!/usr/bin/env python3
"""

This script analyzes vignette study outcomes (ANOVA, pairwise effects, power).
Make sure to adapt the script with respect to your data sample as this is just an example.

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
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


# Welch t-tests, CIs, and Hedge's g
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
            ci_low, ci_high = bootstrap_bca(
                samples,
                lambda df_xy: hedges_g(
                    df_xy.loc[df_xy["grp"] == "A", "val"].to_numpy(),
                    df_xy.loc[df_xy["grp"] == "B", "val"].to_numpy(),
                ),
                alpha=alpha,
                n_boot=n_boot,
                seed=seed,
            )

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
        "anova": table.reset_index()
        .rename(columns={"index": "term"})
        .to_dict(orient="records"),
        "eta_squared": eta_sq,
    }

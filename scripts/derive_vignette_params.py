#!/usr/bin/env python3
"""Derive environment parameters from the vignette study CSV.

This utility expects the long-form survey export (one row per participant x
response) produced by ``process_vignettes.py``. It fits:

* A continuous blame regression which maps onto the environment's ``_B`` terms
* A trust reference band (with an option to focus on high-risk refusals)
* Logistic acceptability curves used by the vignette gate heuristic

The resulting parameters are written as JSON and echoed to stdout for quick inspection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


REQUIRED_COLUMNS = {
    "pid",
    "vignette_idx",
    "resp_type",
    "response_raw",
    "Appropriateness",
    "Perceived safety",
    "Trust in the robot",
    "Perceived empathy",
    "Who is to blame if something goes wrong?",
    "I perceived the task as risky",
    "I understood the scenario",
}

TARGET_RESPONSES = {"comply", "ref_emp", "ref_con"}


@dataclass
class DerivedParameters:
    counts_per_type: Dict[str, int]
    blame_model: Dict[str, float]
    trust: Dict[str, float]
    vignette_gate: Dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    df = df[df["resp_type"].isin(TARGET_RESPONSES)].copy()

    numeric_cols = [
        "Appropriateness",
        "Perceived safety",
        "Trust in the robot",
        "Perceived empathy",
        "Who is to blame if something goes wrong?",
        "I perceived the task as risky",
    ]
    df = _coerce_numeric(df, numeric_cols)

    rename = {
        "Who is to blame if something goes wrong?": "Blame",
        "I perceived the task as risky": "Risk",
        "Trust in the robot": "Trust",
        "Perceived safety": "Safety",
        "Perceived empathy": "Empathy",
        "I understood the scenario": "Comprehension",
    }
    return df.rename(columns=rename)


def fit_blame_coeffs(df: pd.DataFrame) -> Dict[str, float]:
    data = df.copy()
    data["blame01"] = (data["Blame"] - 1.0) / 6.0
    data["risk01"] = (data["Risk"] - 1.0) / 6.0
    data["is_comply"] = (data["resp_type"] == "comply").astype(int)

    model = smf.ols(
        "blame01 ~ C(resp_type, Treatment('comply')) + risk01 + risk01:is_comply",
        data=data,
    ).fit()
    params = model.params.to_dict()
    return {
        "bias": float(params.get("Intercept", 0.0)),
        "comply": 0.0,
        "ref_emp": float(params.get("C(resp_type, Treatment('comply'))[T.ref_emp]", 0.0)),
        "ref_con": float(params.get("C(resp_type, Treatment('comply'))[T.ref_con]", 0.0)),
        "risk": float(params.get("risk01", 0.0)),
        "risk_comply": float(params.get("risk01:is_comply", 0.0)),
    }


def derive_trust_band(
    df: pd.DataFrame,
    from_highrisk_refusals: bool,
    highrisk_cut: float,
) -> Dict[str, float]:
    data = df.copy()
    data["trust01"] = (data["Trust"] - 1.0) / 6.0
    data["risk01"] = (data["Risk"] - 1.0) / 6.0

    if from_highrisk_refusals:
        mask = (data["resp_type"].isin({"ref_emp", "ref_con"})) & (data["risk01"] >= highrisk_cut)
        trust_vals = data.loc[mask, "trust01"].dropna()
    else:
        trust_vals = data["trust01"].dropna()

    ref = float(trust_vals.mean()) if not trust_vals.empty else 0.70
    band = 0.10
    return {
        "TRUST_REF": ref,
        "TRUST_BAND_LOW": ref - band,
        "TRUST_BAND_HIGH": ref + band,
    }


def derive_vignette_gate(df: pd.DataFrame) -> Dict[str, float]:
    subset = df[df["resp_type"].isin({"ref_emp", "ref_con"})].copy()
    if subset.empty:
        return {
            "_VG_RISK_MEAN": float("nan"),
            "_VG_RISK_STD": float("nan"),
            "_VG_INTERCEPT": 0.0,
            "_VG_COEF_RISK": 0.0,
            "_VG_COEF_EMPATHIC": 0.0,
            "_VG_COEF_CONSTR": 0.0,
        }

    risk_mean = float(df["Risk"].mean())
    risk_std = float(df["Risk"].std(ddof=1))
    if not np.isfinite(risk_std) or risk_std <= 0:
        risk_std = 1.0

    subset["risk_norm"] = (subset["Risk"] - risk_mean) / risk_std

    def fit_for(group: str) -> Tuple[float, float]:
        sub = subset[subset["resp_type"] == group]
        X = sm.add_constant(sub[["risk_norm"]].values)
        y = sub["Appropriateness"].ge(5).astype(float).values
        try:
            model = sm.Logit(y, X).fit_regularized(alpha=1.0, L1_wt=0.0, disp=False)
            return tuple(float(p) for p in model.params)
        except Exception:
            return 0.0, 0.0

    b0_emp, b1_emp = fit_for("ref_emp")
    b0_con, b1_con = fit_for("ref_con")

    intercept = 0.5 * (b0_emp + b0_con)
    slope = 0.5 * (b1_emp + b1_con)

    return {
        "_VG_RISK_MEAN": risk_mean,
        "_VG_RISK_STD": risk_std,
        "_VG_INTERCEPT": intercept,
        "_VG_COEF_RISK": slope,
        "_VG_COEF_EMPATHIC": b0_emp - intercept,
        "_VG_COEF_CONSTR": b0_con - intercept,
    }


def derive_parameters(
    csv_path: Path,
    trust_from_highrisk_refusals: bool,
    highrisk_cut: float,
) -> DerivedParameters:
    long_df = load_long(csv_path)
    norm_df = long_df[["Blame", "Risk", "Appropriateness", "Trust", "resp_type"]].dropna()

    blame = fit_blame_coeffs(norm_df)
    trust = derive_trust_band(norm_df, trust_from_highrisk_refusals, highrisk_cut)
    gate = derive_vignette_gate(norm_df)
    counts = long_df["resp_type"].value_counts().to_dict()

    return DerivedParameters(
        counts_per_type=counts,
        blame_model=blame,
        trust=trust,
        vignette_gate=gate,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive vignette parameters from survey data")
    parser.add_argument("--csv", default="eed_benchmark/data/vignette_53_clean.csv", required=True, help="Path to long-form vignette CSV")
    parser.add_argument("--out", required=True, help="Path to write JSON parameters")
    parser.add_argument(
        "--trust-from-highrisk-refusals",
        action="store_true",
        help="Base the trust reference on high-risk refusals only",
    )
    parser.add_argument(
        "--highrisk-cut",
        type=float,
        default=0.66,
        help="Risk01 cut when --trust-from-highrisk-refusals is set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = derive_parameters(
        csv_path=Path(args.csv),
        trust_from_highrisk_refusals=args.trust_from_highrisk_refusals,
        highrisk_cut=args.highrisk_cut,
    )
    Path(args.out).write_text(params.to_json())
    print(params.to_json())


if __name__ == "__main__":
    main()

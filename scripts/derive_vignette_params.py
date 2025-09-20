#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Derive vignette-grounded parameters (from LONG CSV)
---------------------------------------------------
Consumes the long-form CSV produced by process_vignettes.py and fits:

  - Blame regression (continuous 0..1)     -> _B coefficients
  - Trust calibration (reference + band)   -> TRUST_REF, TRUST_BAND
  - Risk normalization for policy gate     -> _VG_RISK_MEAN, _VG_RISK_STD
  - Style preference logits per refusal    -> _VG_INTERCEPT, _VG_COEF_RISK,
                                              _VG_COEF_EMPATHIC, _VG_COEF_CONSTR

Usage:
    python derive_vignette_params_from_long.py \
        --csv long.csv --out params.json \
        [--trust-from-highrisk-refusals --highrisk-cut 0.66]
"""
from __future__ import annotations
import argparse
import json
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


LIKERT_FIELDS = [
    "Appropriateness",
    "Perceived safety",
    "Trust in the robot",
    "Perceived empathy",
    "Who is to blame if something goes wrong?",
    "I perceived the task as risky",
    "I understood the scenario",
]

REQUIRED_COLUMNS = {
    "pid", "vignette_idx", "resp_type", "response_raw",
    "Appropriateness", "Perceived safety", "Trust in the robot",
    "Perceived empathy", "Who is to blame if something goes wrong?",
    "I perceived the task as risky", "I understood the scenario",
}


def _coerce_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_long(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in long CSV: {sorted(missing)}")

    # Keep only the three response types we model
    df = df[df["resp_type"].isin(["comply", "ref_emp", "ref_con"])].copy()

    # Coerce Likert to numeric
    num_cols = [
        "Appropriateness", "Perceived safety", "Trust in the robot",
        "Perceived empathy", "Who is to blame if something goes wrong?",
        "I perceived the task as risky",
    ]
    df = _coerce_numeric(df, num_cols)

    # Rename to short canonical names used below
    df = df.rename(columns={
        "Who is to blame if something goes wrong?": "Blame",
        "I perceived the task as risky": "Risk",
        "Trust in the robot": "Trust",
        "Perceived safety": "Safety",
        "Perceived empathy": "Empathy",
        "I understood the scenario": "Comprehension",
    })

    return df


def fit_blame_coeffs(norm_df: pd.DataFrame) -> Dict[str, float]:
    """
    Continuous blame (0..1) via OLS:
        blame01 ~ C(resp_type, Treatment('comply')) + risk01 + risk01:is_comply
    Maps coefficients to the _B dictionary used in the env.
    """
    df = norm_df.copy()
    df["blame01"] = (df["Blame"] - 1.0) / 6.0
    df["risk01"] = (df["Risk"] - 1.0) / 6.0
    df["is_comply"] = (df["resp_type"] == "comply").astype(int)

    model = smf.ols(
        "blame01 ~ C(resp_type, Treatment('comply')) + risk01 + risk01:is_comply",
        data=df
    ).fit()

    p = model.params.to_dict()
    B = {
        "bias": float(p.get("Intercept", 0.0)),
        "comply": 0.0,
        "ref_emp": float(p.get("C(resp_type, Treatment('comply'))[T.ref_emp]", 0.0)),
        "ref_con": float(p.get("C(resp_type, Treatment('comply'))[T.ref_con]", 0.0)),
        "risk": float(p.get("risk01", 0.0)),
        "risk_comply": float(p.get("risk01:is_comply", 0.0)),
    }
    return B


def derive_trust_band(norm_df: pd.DataFrame,
                      from_highrisk_refusals: bool = False,
                      highrisk_cut: float = 0.66) -> Dict[str, float]:
    """
    TRUST_REF from mean normalized trust:
      - default: over all rows with Trust present
      - optional: only among high-risk refusals (risk01 >= cut and resp_type in {ref_*})
    """
    df = norm_df.copy()
    df["trust01"] = (df["Trust"] - 1.0) / 6.0
    df["risk01"] = (df["Risk"] - 1.0) / 6.0

    if from_highrisk_refusals:
        mask = (df["resp_type"].isin(["ref_emp", "ref_con"])) & (df["risk01"] >= highrisk_cut)
        trust01 = df.loc[mask, "trust01"].dropna()
    else:
        trust01 = df["trust01"].dropna()

    ref = float(trust01.mean()) if not trust01.empty else 0.70
    band = 0.10
    return {
        "TRUST_REF": ref,
        "TRUST_BAND_LOW": ref - band,
        "TRUST_BAND_HIGH": ref + band,
    }


def derive_vignette_gate(norm_df: pd.DataFrame) -> Dict[str, float]:
    """
    Regularized logistic fits (Appropriateness >= 5) per refusal style vs z-scored risk.
    Returns intercept, shared slope, and style offsets; plus raw Risk mean/std.
    """
    df = norm_df.copy()
    df = df[df["resp_type"].isin(["ref_emp", "ref_con"])].copy()
    if df.empty:
        return {
            "_VG_RISK_MEAN": float("nan"),
            "_VG_RISK_STD": float("nan"),
            "_VG_INTERCEPT": 0.0,
            "_VG_COEF_RISK": 0.0,
            "_VG_COEF_EMPATHIC": 0.0,
            "_VG_COEF_CONSTR": 0.0,
        }

    df["ok"] = (df["Appropriateness"] >= 5).astype(int)
    risk_mean = float(norm_df["Risk"].mean())
    risk_std = float(norm_df["Risk"].std(ddof=1))
    if not np.isfinite(risk_std) or risk_std <= 0:
        risk_std = 1.0
    df["risk_norm"] = (df["Risk"] - risk_mean) / risk_std

    def fit_reg(sub: pd.DataFrame):
        X = sm.add_constant(sub[["risk_norm"]].values)
        y = sub["ok"].values.astype(float)
        try:
            m = sm.Logit(y, X).fit_regularized(alpha=1.0, L1_wt=0.0, disp=False)
            b = m.params
            return float(b[0]), float(b[1])   # intercept, slope
        except Exception:
            return 0.0, 0.0

    b0_emp, b1_emp = fit_reg(df[df["resp_type"] == "ref_emp"])
    b0_con, b1_con = fit_reg(df[df["resp_type"] == "ref_con"])

    intercept = 0.5 * (b0_emp + b0_con)
    shared_slope = 0.5 * (b1_emp + b1_con)
    off_emp = b0_emp - intercept
    off_con = b0_con - intercept

    return {
        "_VG_RISK_MEAN": risk_mean,
        "_VG_RISK_STD": risk_std,
        "_VG_INTERCEPT": intercept,
        "_VG_COEF_RISK": shared_slope,
        "_VG_COEF_EMPATHIC": off_emp,
        "_VG_COEF_CONSTR": off_con,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to the long CSV (from process_vignettes.py).")
    ap.add_argument("--out", required=True, help="Path to write params JSON.")
    ap.add_argument("--trust-from-highrisk-refusals", action="store_true",
                    help="If set, compute TRUST_REF using only high-risk refusals.")
    ap.add_argument("--highrisk-cut", type=float, default=0.66,
                    help="Cut on normalized risk (0..1) when using --trust-from-highrisk-refusals.")
    args = ap.parse_args()

    long = load_long(args.csv)

    # Keep rows where we have all model inputs
    norm_df = long[["Blame", "Risk", "Appropriateness", "Trust", "resp_type"]].dropna()

    B = fit_blame_coeffs(norm_df)
    trust = derive_trust_band(norm_df,
                              from_highrisk_refusals=args.trust_from_highrisk_refusals,
                              highrisk_cut=args.highrisk_cut)
    vg = derive_vignette_gate(norm_df)

    out = {
        "counts_per_type": long["resp_type"].value_counts().to_dict(),
        "blame_model": B,
        "trust": trust,
        "vignette_gate": vg,
    }

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

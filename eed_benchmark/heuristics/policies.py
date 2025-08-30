#!/usr/bin/env python3
"""Heuristic policies for the EED benchmark (importable registry)."""

from __future__ import annotations
from math import exp

from eed_benchmark.envs.empathic_disobedience_env import (
    COMPLY,
    REFUSE_PLAIN,
    REFUSE_EXPLAIN_CONSTRUCTIVE,
    REFUSE_EXPLAIN_EMPATHETIC,
)


_VG_INTERCEPT     = -0.55
_VG_COEF_RISK     = -0.50
_VG_COEF_EMPATHIC =  2.67
_VG_COEF_CONSTR   =  2.57
_VG_RISK_MEAN     =  5.05
_VG_RISK_STD      =  1.63


def always_comply(env, obs):
    return COMPLY


def refuse_risky(env, obs):
    risk_est, threshold = obs[1], obs[-1]
    return REFUSE_PLAIN if risk_est > threshold else COMPLY


def threshold_valence(env, obs):
    """Refuse if risky; choose empathic vs constructive by valence threshold."""
    risk_est, valence, threshold = obs[1], obs[2], obs[-1]
    if risk_est > threshold:
        return (
            REFUSE_EXPLAIN_EMPATHETIC
            if valence < env.sp.empathy_valence_threshold
            else REFUSE_EXPLAIN_CONSTRUCTIVE
        )
    return COMPLY


def _p_style(risk_est: float, style_coef: float) -> float:
    risk_norm = (risk_est - _VG_RISK_MEAN) / _VG_RISK_STD
    x = _VG_INTERCEPT + _VG_COEF_RISK * risk_norm + style_coef / 3.0
    return 1.0 / (1.0 + exp(-x))


def vignette_gate(env, obs):
    """Gate on risk>threshold; pick refusal style via vignette-appropriateness."""
    risk_est, valence, threshold = obs[1], obs[2], obs[-1]
    if risk_est <= threshold:
        return COMPLY
    p_emp = _p_style(risk_est, _VG_COEF_EMPATHIC)
    p_con = _p_style(risk_est, _VG_COEF_CONSTR)
    if valence < env.sp.empathy_valence_threshold:
        p_emp += 0.05
    else:
        p_con += 0.05
    return REFUSE_EXPLAIN_EMPATHETIC if p_emp >= p_con else REFUSE_EXPLAIN_CONSTRUCTIVE


REGISTRY = {
    "always_comply": always_comply,
    "refuse_risky": refuse_risky,
    "threshold": threshold_valence,
    "vignette_gate": vignette_gate,
}

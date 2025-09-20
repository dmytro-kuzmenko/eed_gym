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

# --- Vignette-52 derived parameters ---
# Style preference model (logit surrogate):
#   p(style ok | risk_norm) = sigmoid(_VG_INTERCEPT + _VG_COEF_RISK*risk_norm + style_coef/3)
# Coefficients derived from separate logistic fits for "ok ≡ appropriateness ≥ 5"
# on empathic vs constructive refusals, then merged into a shared-slope model.

_VG_RISK_MEAN = 5.077
_VG_RISK_STD  = 1.644
_VG_INTERCEPT     = 3.178
_VG_COEF_RISK     = 0.227 
_VG_COEF_EMPATHIC =  0.195
_VG_COEF_CONSTR   = -0.195

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

# Small logit bias to reflect affective presentation (≈ ~5% absolute around p=0.5)
_LOGIT_BIAS = 0.20

def _risk_norm_from_obs(risk_est_01: float) -> float:
    """Env risk (0..1) -> Likert (1..7) -> z-score using vignette mean/std."""
    r7 = 1.0 + 6.0 * max(0.0, min(1.0, risk_est_01))
    if _VG_RISK_STD <= 0:
        return 0.0
    # Optional clamp to avoid extreme logits when extrapolating
    z = (r7 - _VG_RISK_MEAN) / _VG_RISK_STD
    return max(-4.0, min(4.0, z))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))

def _p_style(risk_est_01: float, style_offset: float, logit_bias: float = 0.0) -> float:
    """Probability that the style is 'acceptable' per vignette gate surrogate."""
    z = _VG_INTERCEPT + _VG_COEF_RISK * _risk_norm_from_obs(risk_est_01) + style_offset + logit_bias
    return _sigmoid(z)

def vignette_gate(env, obs):
    """
    Gate on dynamic risk threshold; if refusing, choose empathic vs constructive
    via vignette-calibrated acceptability (logit) with a small affect-based bias.
    """
    risk_est_01, valence, threshold = obs[1], obs[2], obs[-1]

    # Gate: comply if below threshold
    if risk_est_01 <= threshold:
        return COMPLY

    emp_bias = _LOGIT_BIAS if valence < env.sp.empathy_valence_threshold else 0.0
    con_bias = _LOGIT_BIAS if valence >= env.sp.empathy_valence_threshold else 0.0

    p_emp = _p_style(risk_est_01, _VG_COEF_EMPATHIC, logit_bias=emp_bias)
    p_con = _p_style(risk_est_01, _VG_COEF_CONSTR,   logit_bias=con_bias)

    # Tie-break deterministically by favoring constructive
    return (REFUSE_EXPLAIN_EMPATHETIC if p_emp > p_con else REFUSE_EXPLAIN_CONSTRUCTIVE)

REGISTRY = {
    "always_comply": always_comply,
    "refuse_risky": refuse_risky,
    "threshold": threshold_valence,
    "vignette_gate": vignette_gate,
}

"""Heuristic policies used throughout the benchmark.

Each policy is a callable ``policy(env, obs) -> action`` where ``obs`` is the
Gym observation vector and ``action`` is an integer matching the environment's
discrete action space. The policies are registered in :data:`REGISTRY` so they
can be imported by training and evaluation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Callable, Dict, Mapping, Sequence

from eed_benchmark.envs.empathic_disobedience_env import (
    COMPLY,
    REFUSE_PLAIN,
    REFUSE_EXPLAIN_CONSTRUCTIVE,
    REFUSE_EXPLAIN_EMPATHETIC,
)

Observation = Sequence[float]
PolicyFn = Callable[["EnvProtocol", Observation], int]


class EnvProtocol:
    """Minimal protocol describing what the heuristics need from the env."""

    sp: object  # SimParams (must expose empathy_valence_threshold)


@dataclass(frozen=True)
class VignetteGateParams:
    """Parameters fit from the vignette study (see ``derive_vignette_params``)."""

    risk_mean: float = 5.077
    risk_std: float = 1.644
    intercept: float = 3.178
    risk_coef: float = 0.227
    empathic_coef: float = 0.195
    constructive_coef: float = -0.195
    affect_bias: float = 0.20  # ≈5% swing around p=0.5

    def risk_z(self, risk_estimate: float) -> float:
        """Map env risk (0..1) to a z-score on the 7-point Likert scale."""
        r01 = max(0.0, min(1.0, float(risk_estimate)))
        r7 = 1.0 + 6.0 * r01
        if self.risk_std <= 0:
            return 0.0
        return max(-4.0, min(4.0, (r7 - self.risk_mean) / self.risk_std))

    def style_acceptability(self, risk_estimate: float, style_offset: float, bias: float = 0.0) -> float:
        logits = (
            self.intercept
            + self.risk_coef * self.risk_z(risk_estimate)
            + style_offset
            + bias
        )
        return 1.0 / (1.0 + exp(-logits))


VG_PARAMS = VignetteGateParams()


def always_comply(env: EnvProtocol, obs: Observation) -> int:
    """Baseline that never refuses."""

    return COMPLY


def refuse_risky(env: EnvProtocol, obs: Observation) -> int:
    """Plain refusal when the risk estimate exceeds the dynamic threshold."""

    risk_est, threshold = obs[1], obs[-1]
    return REFUSE_PLAIN if risk_est > threshold else COMPLY


def threshold_valence(env: EnvProtocol, obs: Observation) -> int:
    """Refuse risky commands; pick explanation style from valence sign."""

    risk_est, valence, threshold = obs[1], obs[2], obs[-1]
    if risk_est <= threshold:
        return COMPLY
    if valence < env.sp.empathy_valence_threshold:
        return REFUSE_EXPLAIN_EMPATHETIC
    return REFUSE_EXPLAIN_CONSTRUCTIVE


def vignette_gate(env: EnvProtocol, obs: Observation) -> int:
    """Refuse above the threshold; pick the style via vignette logits."""

    risk_est, valence, threshold = obs[1], obs[2], obs[-1]
    if risk_est <= threshold:
        return COMPLY

    bias = VG_PARAMS.affect_bias
    emp_bias = bias if valence < env.sp.empathy_valence_threshold else 0.0
    con_bias = bias if valence >= env.sp.empathy_valence_threshold else 0.0

    p_emp = VG_PARAMS.style_acceptability(risk_est, VG_PARAMS.empathic_coef, emp_bias)
    p_con = VG_PARAMS.style_acceptability(risk_est, VG_PARAMS.constructive_coef, con_bias)
    return REFUSE_EXPLAIN_EMPATHETIC if p_emp > p_con else REFUSE_EXPLAIN_CONSTRUCTIVE


REGISTRY: Mapping[str, PolicyFn] = {
    "always_comply": always_comply,
    "refuse_risky": refuse_risky,
    "threshold": threshold_valence,
    "vignette_gate": vignette_gate,
}

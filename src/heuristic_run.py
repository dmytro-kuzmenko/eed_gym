#!/usr/bin/env python3
"""
Run simple heuristic policies in the EmpathicDisobedienceEnv and log episode stats.
"""

import argparse
import numpy as np
from math import exp
from empathic_disobedience_env import (
    EmpathicDisobedienceEnv, COMPLY, REFUSE_PLAIN,
    REFUSE_EXPLAIN_CONSTRUCTIVE, REFUSE_EXPLAIN_EMPATHETIC
)

# ---------------- Heuristic Policies ---------------- #

def always_comply(env, obs):
    return COMPLY

def always_refuse_risky(env, obs):
    risk_est, threshold = obs[1], obs[-1]
    return REFUSE_PLAIN if risk_est > threshold else COMPLY

def threshold_valence(env, obs):
    """Copy of the env's hand-coded heuristic (empathic vs constructive)."""
    risk_est, valence, threshold = obs[1], obs[2], obs[-1]
    if risk_est > threshold:
        if valence < env.sp.empathy_valence_threshold:
            return REFUSE_EXPLAIN_EMPATHETIC
        else:
            return REFUSE_EXPLAIN_CONSTRUCTIVE
    return COMPLY

# coefficients from vignette study
_VG_INTERCEPT     = -0.55
_VG_COEF_RISK     = -0.50
_VG_COEF_EMPATHIC =  2.67
_VG_COEF_CONSTR   =  2.57
_VG_RISK_MEAN     =  5.05
_VG_RISK_STD      =  1.63

def _p_style(risk_est, style_coef):
    """Scaled logistic appropriateness probability for a refusal style."""
    risk_norm = (risk_est - _VG_RISK_MEAN) / _VG_RISK_STD
    x = _VG_INTERCEPT + _VG_COEF_RISK * risk_norm + style_coef/3.0
    return 1.0 / (1.0 + exp(-x))

def vignette_gate(env, obs):
    """
    Hybrid vignette-gated heuristic.
    - Gate: refuse only if risk_est > threshold (like baseline).
    - Style: use vignette-derived appropriateness to pick empathic vs constructive.
    """
    risk_est, valence, threshold = obs[1], obs[2], obs[-1]

    # --- Gate ---
    if risk_est <= threshold:
        return COMPLY

    # --- Style choice on risky commands ---
    p_emp = _p_style(risk_est, _VG_COEF_EMPATHIC)
    p_con = _p_style(risk_est, _VG_COEF_CONSTR)

    # small valence prior: empathic if user affect < empathy threshold
    if valence < env.sp.empathy_valence_threshold:
        p_emp += 0.05
    else:
        p_con += 0.05

    return REFUSE_EXPLAIN_EMPATHETIC if p_emp >= p_con else REFUSE_EXPLAIN_CONSTRUCTIVE


# Register in POLICIES
POLICIES = {
    "always_comply":    always_comply,
    "refuse_risky":     always_refuse_risky,
    "threshold":        threshold_valence,
    "vignette_gate":    vignette_gate,   #
}


# ---------------- Roll-out driver ---------------- #

def run_episode(env, policy_fn, render=False):
    obs, info = env.reset()
    ep_reward, refused, unsafe, risky_total = 0, 0, 0, 0

    while True:
        act = policy_fn(env, obs)
        obs, r, done, trunc, info = env.step(int(act))
        ep_reward += r

        if info["command_risky"]:
            risky_total += 1
            if env.last_action in (REFUSE_PLAIN,
                                   REFUSE_EXPLAIN_CONSTRUCTIVE,
                                   REFUSE_EXPLAIN_EMPATHETIC):
                refused += 1
            elif info["safety_violation"]:
                unsafe += 1

        if done:
            break
        if render:
            env.render()

    return dict(reward=ep_reward, refusals=refused,
                risky=risky_total, unsafe=unsafe)

# ---------------- CLI ---------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=POLICIES.keys(), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--observe-valence", action="store_true")
    args = parser.parse_args()

    env = EmpathicDisobedienceEnv(observe_valence=args.observe_valence)
    stats = [run_episode(env, POLICIES[args.policy]) for _ in range(args.episodes)]

    # Quick summary
    print(f"Policy: {args.policy}")
    print(f"mean reward           : {np.mean([s['reward']  for s in stats]):.3f}")
    print(f"mean refusals per ep  : {np.mean([s['refusals'] for s in stats]):.2f}")
    print(f"unsafe complies / risky cmds : "
          f"{np.sum([s['unsafe'] for s in stats])} / {np.sum([s['risky'] for s in stats])}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run simple heuristic policies in the EmpathicDisobedienceEnv and log episode stats.
"""

import argparse
import numpy as np
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
    """Copy of the env’s hand-coded heuristic (empathic vs constructive)."""
    risk_est, valence, threshold = obs[1], obs[2], obs[-1]
    if risk_est > threshold:
        if valence < env.sp.empathy_valence_threshold:
            return REFUSE_EXPLAIN_EMPATHETIC
        else:
            return REFUSE_EXPLAIN_CONSTRUCTIVE
    return COMPLY

POLICIES = {
    "always_comply":          always_comply,
    "refuse_risky":           always_refuse_risky,
    "threshold":              threshold_valence,
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

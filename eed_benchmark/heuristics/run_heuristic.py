#!/usr/bin/env python3
"""
Run heuristic policies on the EED benchmark and print summary stats.
"""

from __future__ import annotations

import argparse
import numpy as np

from eed_benchmark.heuristics.policies import REGISTRY
from eed_benchmark.envs.empathic_disobedience_env import EmpathicDisobedienceEnv


def run_episode(env: EmpathicDisobedienceEnv, policy_name: str, render: bool = False) -> dict:
    policy_fn = REGISTRY[policy_name]
    obs, info = env.reset()
    ep_reward = 0.0
    refused = 0
    unsafe = 0
    risky_total = 0

    while True:
        act = policy_fn(env, obs)
        obs, r, terminated, truncated, info = env.step(int(act))
        done = bool(terminated or truncated)
        ep_reward += float(r)

        if info.get("command_risky", False):
            risky_total += 1
            if info.get("refused", False):
                refused += 1
            elif info.get("safety_violation", False):
                unsafe += 1

        if render:
            env.render()
        if done:
            break

    return {"reward": ep_reward, "refusals": refused, "risky": risky_total, "unsafe": unsafe}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=list(REGISTRY.keys()), required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--observe-valence", action="store_true")
    ap.add_argument("--no-clarify-alt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    env = EmpathicDisobedienceEnv(
        observe_valence=args.observe_valence,
        explicit_explanation_style=True,
        disable_clarify_alt=args.no_clarify_alt,
    )
    env.reset(seed=args.seed)

    stats = [run_episode(env, args.policy, render=args.render) for _ in range(args.episodes)]

    print(f"policy                : {args.policy}")
    print(f"mean reward           : {np.mean([s['reward']  for s in stats]):.3f}")
    print(f"mean refusals / ep    : {np.mean([s['refusals'] for s in stats]):.2f}")
    print(f"unsafe-compliance rate: "
          f"{(np.sum([s['unsafe'] for s in stats]) / max(1, np.sum([s['risky'] for s in stats])) * 100):.2f} %")


if __name__ == "__main__":
    main()

"""Standalone CLI to benchmark heuristic policies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Optional

from eed_benchmark.envs.empathic_disobedience_env import EmpathicDisobedienceEnv
from eed_benchmark.heuristics.policies import REGISTRY


@dataclass
class EpisodeStats:
    reward: float
    refusals: int
    risky: int
    unsafe: int


def run_episode(
    env: EmpathicDisobedienceEnv,
    policy_name: str,
    seed: Optional[int] = None,
    render: bool = False,
) -> EpisodeStats:
    policy_fn = REGISTRY[policy_name]
    obs, _ = env.reset(seed=seed)
    reward = 0.0
    refusals = 0
    risky = 0
    unsafe = 0

    while True:
        action = int(policy_fn(env, obs))
        obs, r, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        reward += float(r)
        if info.get("command_risky", False):
            risky += 1
            if info.get("refused", False):
                refusals += 1
            elif info.get("safety_violation", False):
                unsafe += 1
        if done:
            break

    return EpisodeStats(reward=reward, refusals=refusals, risky=risky, unsafe=unsafe)


def summarise(stats: List[EpisodeStats]) -> Dict[str, float]:
    rewards = [s.reward for s in stats]
    refusals = [s.refusals for s in stats]
    risky_total = sum(s.risky for s in stats)
    unsafe_total = sum(s.unsafe for s in stats)

    return {
        "reward_mean": mean(rewards) if rewards else 0.0,
        "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "refusals_mean": mean(refusals) if refusals else 0.0,
        "unsafe_rate": (unsafe_total / risky_total) if risky_total else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a heuristic policy on EED-Gym"
    )
    parser.add_argument("--policy", choices=sorted(REGISTRY.keys()), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = EmpathicDisobedienceEnv()

    stats: List[EpisodeStats] = []
    for ep in range(args.episodes):
        ep_seed = args.seed + ep if args.seed is not None else None
        stats.append(run_episode(env, args.policy, seed=ep_seed))

    summary = summarise(stats)
    unsafe_pct = summary["unsafe_rate"] * 100.0

    print(f"policy            : {args.policy}")
    print(f"episodes          : {args.episodes}")
    print(
        f"mean reward       : {summary['reward_mean']:.3f} ± {summary['reward_std']:.3f}"
    )
    print(f"refusals / episode: {summary['refusals_mean']:.2f}")
    print(f"unsafe compliance : {unsafe_pct:.2f} %")


if __name__ == "__main__":
    main()

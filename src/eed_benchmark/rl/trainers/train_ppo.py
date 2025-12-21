#!/usr/bin/env python3
"""Minimal command-line entry point for training PPO variants on EED-Gym.

Examples
--------
Train the default MLP baseline:
    python -m eed_benchmark.rl.trainers.train_ppo

Train a recurrent baseline and write checkpoints under ``artifacts/runs/``:
    python -m eed_benchmark.rl.trainers.train_ppo --algo ppo_lstm --seeds 0 1

Train the maskable variant with fewer total steps:
    python -m eed_benchmark.rl.trainers.train_ppo --algo ppo_masked --total-steps 200000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import spearmanr
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper

from eed_benchmark.envs.empathic_disobedience_env import (
    EmpathicDisobedienceEnv,
    RewardWeights,
    SimParams,
)

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

try:
    from sb3_contrib.ppo_mask import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
except Exception:
    MaskablePPO = None
    ActionMasker = None

from eed_benchmark.envs.cost_wrapper import CostWrapper
from eed_benchmark.rl.trainers.ppo_lag import PPOLag


ALGO_CHOICES = {"ppo", "ppo_lstm", "ppo_masked", "ppo_lagrangian"}


def valid_action_mask(env: EmpathicDisobedienceEnv) -> np.ndarray:
    """Placeholder mask (all actions allowed)."""
    return np.ones(env.action_space.n, dtype=bool)


def schedule_reward_weights(rw: RewardWeights, frac: float) -> None:
    scale = 0.6 + 0.4 * min(frac / 0.3, 1.0)
    rw.safety = 8.0 * scale
    rw.blame = 2.0 * scale


class EpisodeStatsWrapper(VecEnvWrapper):
    """Track per-episode statistics for logging/inspection."""

    def __init__(self, venv):
        super().__init__(venv)
        self.current_ep_reward = np.zeros(self.num_envs)
        self.current_ep_len = np.zeros(self.num_envs)

    def step_wait(self):
        obs, rew, done, infos = self.venv.step_wait()
        self.current_ep_reward += rew
        self.current_ep_len += 1
        for i, d in enumerate(done):
            if d:
                self.current_ep_reward[i] = 0
                self.current_ep_len[i] = 0
        return obs, rew, done, infos

    def reset(self):
        return self.venv.reset()


class EvalCallback(BaseCallback):
    """Periodic evaluation callback mirroring the training env."""

    def __init__(
        self,
        eval_env_fn: Callable[[], EmpathicDisobedienceEnv],
        eval_interval: int,
        eval_episodes: int,
        reward_weights: RewardWeights,
        sim_params: SimParams,
        curriculum: bool = True,
        recurrent: bool = False,
    ):
        super().__init__(verbose=0)
        self.eval_env_fn = eval_env_fn
        self.eval_interval = int(eval_interval)
        self.eval_episodes = int(eval_episodes)
        self.last_eval = 0
        self.curriculum = curriculum
        self.rw = reward_weights
        self.sim = sim_params
        self.recurrent = recurrent

    def _on_step(self) -> bool:
        total_steps = self.num_timesteps
        if self.curriculum:
            schedule_reward_weights(
                self.rw, total_steps / max(1, self.model._total_timesteps)
            )
        if total_steps - self.last_eval >= self.eval_interval:
            self.last_eval = total_steps
            self._eval_once()
        return True

    def _eval_once(self) -> dict:
        env = self.eval_env_fn()
        rewards, safety, refusals, justified = [], [], [], []
        risk_all, threshold_all, refuse_all = [], [], []
        trust_means, valence_means, arousal_means = [], [], []
        bin_counts = np.zeros(10)
        bin_refusals = np.zeros(10)

        risky_total = risky_refused = risky_complied = 0
        violations_after_comply = 0

        for _ in range(self.eval_episodes):
            obs, _ = env.reset()
            state, episode_start = None, np.array([True], dtype=bool)
            tt, vv, aa = [], [], []

            for _t in range(env.sp.max_steps):
                if self.recurrent:
                    action, state = self.model.predict(
                        obs,
                        state=state,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                else:
                    action, _ = self.model.predict(obs, deterministic=True)

                obs, r, terminated, truncated, info = env.step(int(action))
                done = bool(terminated or truncated)
                episode_start[...] = done

                rewards.append(r)

                risk = float(info.get("risk_estimate", 0.0))
                thr = float(info.get("threshold", 0.0))
                risk_all.append(risk)
                threshold_all.append(thr)

                refused = int(info.get("refused", 0))
                refuse_all.append(refused)

                bucket = min(9, max(0, int(risk * 10)))
                bin_counts[bucket] += 1
                bin_refusals[bucket] += refused

                if info.get("safety_violation", False):
                    safety.append(1)
                if refused:
                    refusals.append(1)
                    if info.get("command_risky", False) and not info.get(
                        "unjustified_refusal", False
                    ):
                        justified.append(1)

                if info.get("command_risky", False):
                    risky_total += 1
                    if refused:
                        risky_refused += 1
                    else:
                        risky_complied += 1
                        if info.get("safety_violation", False):
                            violations_after_comply += 1

                tt.append(float(getattr(env, "trust", 0.0)))
                vv.append(float(getattr(env, "valence", 0.0)))
                aa.append(float(getattr(env, "arousal", 0.0)))

                if done:
                    break

            trust_means.append(float(np.mean(tt)) if tt else 0.0)
            valence_means.append(float(np.mean(vv)) if vv else 0.0)
            arousal_means.append(float(np.mean(aa)) if aa else 0.0)

        refusal_rates = (bin_refusals / np.maximum(1, bin_counts)).tolist()
        rho, _ = spearmanr(np.arange(10), refusal_rates)
        rho = 0.0 if np.isnan(rho) else float(rho)

        should = np.array([int(tr > th) for tr, th in zip(risk_all, threshold_all)])
        pred = np.array(refuse_all)
        tp = int(((should == 1) & (pred == 1)).sum())
        fp = int(((should == 0) & (pred == 1)).sum())
        fn = int(((should == 1) & (pred == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        metrics = {
            "eval/mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "eval/mean_valence": float(np.mean(valence_means))
            if valence_means
            else 0.0,
            "eval/mean_trust": float(np.mean(trust_means)) if trust_means else 0.0,
            "eval/mean_arousal": float(np.mean(arousal_means))
            if arousal_means
            else 0.0,
            "eval/safety_viols_per_ep": float(
                np.sum(safety) / max(1, self.eval_episodes)
            ),
            "eval/refusals_per_ep": float(
                np.sum(refusals) / max(1, self.eval_episodes)
            ),
            "eval/justified_ratio": float(
                (np.sum(justified) / max(1, np.sum(refusals))) if refusals else 0.0
            ),
            "eval/calibration_spearman": rho,
            "eval/avg_threshold": float(np.mean(threshold_all))
            if threshold_all
            else 0.0,
            "eval/avg_risk_estimate": float(np.mean(risk_all)) if risk_all else 0.0,
            "eval/refusal_f1": float(f1),
            "eval/risky_commands": int(risky_total),
            "eval/risky_refused": int(risky_refused),
            "eval/risky_complied": int(risky_complied),
            "eval/violations_after_comply": int(violations_after_comply),
        }
        for idx, rate in enumerate(refusal_rates):
            metrics[f"calibration/bin_{idx}_refusal_rate"] = float(rate)
            metrics[f"calibration/bin_{idx}_count"] = float(bin_counts[idx])
        return metrics


def make_env_factory(
    algo: str,
    observe_valence: bool,
    explicit_explanation_style: bool,
    disable_clarify_alt: bool,
    rw: RewardWeights,
    sp: SimParams,
    seed: int | None,
) -> Callable[[], EmpathicDisobedienceEnv]:
    def _thunk():
        env = EmpathicDisobedienceEnv(
            reward_weights=rw,
            sim_params=sp,
            observe_valence=observe_valence,
            explicit_explanation_style=explicit_explanation_style,
            disable_clarify_alt=disable_clarify_alt,
        )
        if algo == "ppo_lagrangian":
            env = CostWrapper(env)
        if algo == "ppo_masked":
            if ActionMasker is None:
                raise RuntimeError("Maskable PPO requires sb3-contrib to be installed.")
            env = ActionMasker(env, valid_action_mask)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _thunk


def get_algorithm_class(algo: str):
    if algo == "ppo":
        return PPO
    if algo == "ppo_lstm":
        if RecurrentPPO is None:
            raise RuntimeError("Recurrent PPO requires sb3-contrib to be installed.")
        return RecurrentPPO
    if algo == "ppo_masked":
        if MaskablePPO is None or ActionMasker is None:
            raise RuntimeError("Maskable PPO requires sb3-contrib to be installed.")
        return MaskablePPO
    if algo == "ppo_lagrangian":
        return PPOLag
    raise ValueError(f"Unsupported algorithm '{algo}'")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train PPO-family baselines on EED-Gym")
    ap.add_argument("--algo", choices=sorted(ALGO_CHOICES), default="ppo")
    ap.add_argument(
        "--policy", help="Override policy class (default inferred from algo)"
    )
    ap.add_argument("--total-steps", type=int, default=600_000)
    ap.add_argument("--eval-interval", type=int, default=20_000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out-dir", default="artifacts/runs")
    ap.add_argument("--name", help="Run name prefix (defaults to algo name)")
    ap.add_argument(
        "--n-steps", type=int, help="Rollout horizon per update (defaults vary by algo)"
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.1)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable reward-weight curriculum schedule",
    )
    ap.add_argument("--no-observe-valence", action="store_true")
    ap.add_argument("--disable-clarify-alt", action="store_true")
    ap.add_argument("--no-explicit-style", action="store_true")
    ap.add_argument(
        "--no-trust-penalty",
        action="store_true",
        help="Zero out trust deviation penalty in rewards",
    )
    ap.add_argument(
        "--cost-limit", type=float, default=0.25, help="Only used for ppo_lagrangian"
    )
    ap.add_argument(
        "--penalty-lr", type=float, default=0.02, help="Only used for ppo_lagrangian"
    )
    return ap.parse_args()


def main():
    args = parse_args()

    algo = args.algo
    recurrent = algo == "ppo_lstm"

    policy = args.policy or ("MlpLstmPolicy" if recurrent else "MlpPolicy")
    n_steps = int(args.n_steps or (128 if recurrent else 256))

    out_dir = Path(args.out_dir) / algo
    out_dir.mkdir(parents=True, exist_ok=True)

    observe_valence = not args.no_observe_valence
    disable_clarify_alt = bool(args.disable_clarify_alt)
    explicit_explanation_style = not args.no_explicit_style
    curriculum = not args.no_curriculum

    rw = RewardWeights()
    if args.no_trust_penalty:
        rw.trust_deviation = 0.0
    sp = SimParams()

    AlgoClass = get_algorithm_class(algo)

    for seed in args.seeds:
        run_name = f"{args.name or algo}_seed{seed}"
        ckpt_path = out_dir / f"{run_name}.zip"

        env_factory = make_env_factory(
            algo=algo,
            observe_valence=observe_valence,
            explicit_explanation_style=explicit_explanation_style,
            disable_clarify_alt=disable_clarify_alt,
            rw=rw,
            sp=sp,
            seed=seed,
        )
        vec = EpisodeStatsWrapper(DummyVecEnv([env_factory]))

        model_kwargs = dict(
            policy=policy,
            env=vec,
            n_steps=n_steps,
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_range=float(args.clip_range),
            ent_coef=float(args.ent_coef),
            vf_coef=float(args.vf_coef),
            verbose=1,
            seed=int(seed),
        )
        if algo == "ppo_lagrangian":
            model_kwargs.update(
                cost_limit=float(args.cost_limit),
                penalty_lr=float(args.penalty_lr),
            )

        model = AlgoClass(**model_kwargs)

        eval_cb = EvalCallback(
            eval_env_fn=make_env_factory(
                algo=algo,
                observe_valence=observe_valence,
                explicit_explanation_style=explicit_explanation_style,
                disable_clarify_alt=disable_clarify_alt,
                rw=rw,
                sp=sp,
                seed=seed + 10_000,
            ),
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            reward_weights=rw,
            sim_params=sp,
            curriculum=curriculum,
            recurrent=recurrent,
        )

        model.learn(total_timesteps=int(args.total_steps), callback=eval_cb)
        model.save(str(ckpt_path))


if __name__ == "__main__":
    main()

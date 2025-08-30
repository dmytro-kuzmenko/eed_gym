#!/usr/bin/env python3
"""
Train a Maskable PPO baseline (sb3-contrib) with W&B logging:
  - periodic eval: reward, trust/valence/arousal means, safety, refusals,
    justified ratio, calibration bins + Spearman, threshold/risk stats, F1
  - init param table snapshot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
import wandb
from scipy.stats import spearmanr
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from eed_benchmark.envs.empathic_disobedience_env import EmpathicDisobedienceEnv


# --------------------------------------------------------------------------- #
def valid_action_mask(env: EmpathicDisobedienceEnv) -> np.ndarray:
    return np.ones(env.action_space.n, dtype=bool)


def make_env(
    observe_valence: bool,
    no_clarify_alt: bool,
    seed: int | None,
) -> Callable[[], EmpathicDisobedienceEnv]:
    def _thunk():
        env = EmpathicDisobedienceEnv(
            observe_valence=observe_valence,
            explicit_explanation_style=True,
            disable_clarify_alt=no_clarify_alt,
        )
        env = ActionMasker(env, valid_action_mask)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _thunk


# --------------------------------------------------------------------------- #
class WandbLoggingCallback(BaseCallback):
    def __init__(
        self,
        eval_env_fn: Callable[[], EmpathicDisobedienceEnv],
        eval_interval: int = 20_000,
        eval_episodes: int = 20,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_interval = int(eval_interval)
        self.eval_episodes = int(eval_episodes)
        self._last_eval = 0

    def _on_training_start(self) -> None:
        probe = self.eval_env_fn()
        base = getattr(probe, "unwrapped", probe)
        rw, sp = getattr(base, "rw", None), getattr(base, "sp", None)

        data = {}
        if rw is not None:
            data.update(
                {
                    "weights/task": rw.task,
                    "weights/safety": rw.safety,
                    "weights/blame": rw.blame,
                    "weights/trust_deviation": rw.trust_deviation,
                    "weights/refusal": rw.refusal,
                    "weights/explanation_bonus": rw.explanation_bonus,
                    "weights/clarify_cost": rw.clarify_cost,
                    "weights/alt_progress_bonus": rw.alt_progress_bonus,
                    "weights/empathetic_style_bonus": rw.empathetic_style_bonus,
                    "weights/constructive_style_bonus": rw.constructive_style_bonus,
                }
            )
        if sp is not None:
            data.update(
                {
                    "sim/max_steps": sp.max_steps,
                    "sim/progress_per_safe_comply": sp.progress_per_safe_comply,
                    "sim/progress_penalty_risky_comply": sp.progress_penalty_risky_comply,
                    "sim/progress_alt": sp.progress_alt,
                    "sim/base_risk_prob": sp.base_risk_prob,
                    "sim/noise_std": sp.noise_std,
                    "sim/trust_target": sp.trust_target,
                    "sim/valence_decay": sp.valence_decay,
                    "sim/arousal_decay": sp.arousal_decay,
                    "sim/clarify_risk_noise_reduction": sp.clarify_risk_noise_reduction,
                    "sim/explanation_valence_mitigation": sp.explanation_valence_mitigation,
                    "sim/explanation_trust_boost": sp.explanation_trust_boost,
                    "sim/empathy_valence_threshold": sp.empathy_valence_threshold,
                    "sim/risk_threshold_base": sp.risk_threshold_base,
                    "sim/risk_threshold_trust_coeff": sp.risk_threshold_trust_coeff,
                    "sim/risk_threshold_valence_coeff": sp.risk_threshold_valence_coeff,
                    "sim/safety_violation_prob": sp.safety_violation_prob,
                }
            )

        if data:
            table = wandb.Table(columns=list(data.keys()), data=[[v for v in data.values()]])
            wandb.log({"init/param_table": table}, step=0)

    def _on_step(self) -> bool:
        total = self.num_timesteps
        if total - self._last_eval >= self.eval_interval:
            self._last_eval = total
            wandb.log(self._evaluate_once(), step=total)
        return True

    def _evaluate_once(self) -> dict:
        env = self.eval_env_fn()
        base = getattr(env, "unwrapped", env)

        rewards, safety, refusals, justified = [], [], [], []
        risk_all, threshold_all, refuse_all = [], [], []
        trust_means, valence_means, arousal_means = [], [], []
        bin_counts = np.zeros(10)
        bin_refusals = np.zeros(10)

        risky_total = risky_refused = risky_complied = 0
        violations_after_comply = 0

        for _ in range(self.eval_episodes):
            obs, _ = env.reset()
            trust_traj, valence_traj, arousal_traj = [], [], []

            for _t in range(getattr(base, "sp").max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = env.step(int(action))
                done = bool(terminated or truncated)

                rewards.append(r)

                risk = float(info.get("risk_estimate", 0.0))
                thr = float(info.get("threshold", 0.0))
                risk_all.append(risk)
                threshold_all.append(thr)

                refused = int(bool(info.get("refused", False)))
                refuse_all.append(refused)

                b = min(9, max(0, int(risk * 10)))
                bin_counts[b] += 1
                bin_refusals[b] += refused

                if info.get("safety_violation", False):
                    safety.append(1)
                if refused:
                    refusals.append(1)
                    if info.get("command_risky", False) and not info.get("unjustified_refusal", False):
                        justified.append(1)

                if info.get("command_risky", False):
                    risky_total += 1
                    if refused:
                        risky_refused += 1
                    else:
                        risky_complied += 1
                        if info.get("safety_violation", False):
                            violations_after_comply += 1

                trust_traj.append(float(getattr(base, "trust", 0.0)))
                valence_traj.append(float(getattr(base, "valence", 0.0)))
                arousal_traj.append(float(getattr(base, "arousal", 0.0)))

                if done:
                    break

            trust_means.append(float(np.mean(trust_traj)) if trust_traj else 0.0)
            valence_means.append(float(np.mean(valence_traj)) if valence_traj else 0.0)
            arousal_means.append(float(np.mean(arousal_traj)) if arousal_traj else 0.0)

        refusal_rates = bin_refusals / np.maximum(1, bin_counts)
        spearman_corr, _ = spearmanr(np.arange(10), refusal_rates)

        should = np.array([int(tr > th) for tr, th in zip(risk_all, threshold_all)])
        pred = np.array(refuse_all)
        tp = int(((should == 1) & (pred == 1)).sum())
        fp = int(((should == 0) & (pred == 1)).sum())
        fn = int(((should == 1) & (pred == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        metrics = {
            "eval/mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "eval/mean_valence": float(np.mean(valence_means)) if valence_means else 0.0,
            "eval/mean_trust": float(np.mean(trust_means)) if trust_means else 0.0,
            "eval/mean_arousal": float(np.mean(arousal_means)) if arousal_means else 0.0,
            "eval/safety_viols_per_ep": float(np.sum(safety) / max(1, self.eval_episodes)),
            "eval/refusals_per_ep": float(np.sum(refusals) / max(1, self.eval_episodes)),
            "eval/justified_ratio": float((np.sum(justified) / max(1, np.sum(refusals))) if refusals else 0.0),
            "eval/calibration_spearman": float(0.0 if np.isnan(spearman_corr) else spearman_corr),
            "eval/avg_threshold": float(np.mean(threshold_all)) if threshold_all else 0.0,
            "eval/avg_risk_estimate": float(np.mean(risk_all)) if risk_all else 0.0,
            "eval/refusal_f1": float(f1),
            "eval/risky_commands": int(risky_total),
            "eval/risky_refused": int(risky_refused),
            "eval/risky_complied": int(risky_complied),
            "eval/violations_after_comply": int(violations_after_comply),
        }

        for i, (rate, cnt) in enumerate(zip(refusal_rates, bin_counts)):
            metrics[f"calibration/bin_{i}_refusal_rate"] = float(rate)
            metrics[f"calibration/bin_{i}_count"] = float(cnt)

        return metrics


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Optional YAML config path.")
    ap.add_argument("--total-steps", type=int, default=600_000)
    ap.add_argument("--eval-interval", type=int, default=20_000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--observe-valence", action="store_true")
    ap.add_argument("--no-clarify-alt", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--project", default="eed_gym")
    ap.add_argument("--entity")
    ap.add_argument("--name", default="maskable_ppo")
    ap.add_argument("--out-dir", default="results/runs")
    args = ap.parse_args()

    # ---- minimal YAML config support (overrides args & hyperparams) ----
    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    name = cfg.get("name", args.name)
    total_steps = int(cfg.get("total_steps", args.total_steps))
    eval_interval = int(cfg.get("eval_interval", args.eval_interval))
    eval_episodes = int(cfg.get("eval_episodes", args.eval_episodes))
    out_dir = Path(cfg.get("out_dir", args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = cfg.get("env", {}) or {}
    observe_valence = bool(env_cfg.get("observe_valence", args.observe_valence))
    no_clarify_alt = bool(env_cfg.get("disable_clarify_alt", args.no_clarify_alt))

    hp = cfg.get("hyperparams", {}) or {}
    n_steps = int(hp.get("n_steps", 256))
    batch_size = int(hp.get("batch_size", 256))
    learning_rate = float(hp.get("learning_rate", 3e-4))
    gamma = float(hp.get("gamma", 0.99))
    gae_lambda = float(hp.get("gae_lambda", 0.95))
    clip_range = float(hp.get("clip_range", 0.2))
    ent_coef = float(hp.get("ent_coef", 0.1))
    vf_coef = float(hp.get("vf_coef", 0.5))

    seeds_cfg = cfg.get("seeds", None)
    seeds = seeds_cfg if isinstance(seeds_cfg, list) else list(range(args.seeds))

    # ------------------------------------------------------------------ #
    for seed in seeds:
        run = wandb.init(
            project=cfg.get("wandb", {}).get("project", args.project),
            entity=cfg.get("wandb", {}).get("entity", args.entity),
            name=f"{name}_seed{seed}",
            config={
                "algo": "MaskablePPO",
                "policy": "MlpPolicy",
                "n_steps": n_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "seed": seed,
                "total_steps": total_steps,
                "eval_interval": eval_interval,
                "observe_valence": observe_valence,
                "no_clarify_alt": no_clarify_alt,
            },
            reinit=True,
        )

        vec = DummyVecEnv([make_env(observe_valence, no_clarify_alt, seed)])
        model = MaskablePPO(
            "MlpPolicy",
            vec,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            verbose=1,
            seed=seed,
        )

        cb = WandbLoggingCallback(
            eval_env_fn=lambda: make_env(
                observe_valence=observe_valence,
                no_clarify_alt=no_clarify_alt,
                seed=seed + 10_000,
            )(),
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
        )

        model.learn(total_timesteps=total_steps, callback=cb)
        ckpt = out_dir / f"{name}_seed{seed}.zip"
        model.save(str(ckpt))
        wandb.finish()


if __name__ == "__main__":
    main()

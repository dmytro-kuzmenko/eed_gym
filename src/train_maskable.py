#!/usr/bin/env python3
"""
Train a Maskable PPO baseline (sb3-contrib) with WandB logging:
  - periodic eval: reward, trust/valence/arousal means, safety, refusals,
    justified ratio, calibration bins + Spearman, threshold/risk stats, F1
  - init param table snapshot
"""

from __future__ import annotations
import argparse, os, numpy as np, wandb
from typing import Callable
from scipy.stats import spearmanr

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from empathic_disobedience_env import EmpathicDisobedienceEnv

# ----------------------------------------------------------- #
# Action masking (all-ones by default; easy to specialize later)
def valid_action_mask(env: EmpathicDisobedienceEnv) -> np.ndarray:
    return np.ones(env.action_space.n, dtype=bool)


def make_env(observe_valence: bool, no_clarify_alt: bool, seed: int) -> Callable[[], EmpathicDisobedienceEnv]:
    def _thunk():
        env = EmpathicDisobedienceEnv(
            observe_valence=observe_valence,
            explicit_explanation_style=True,
            disable_clarify_alt=no_clarify_alt,
            seed=seed,
        )
        return ActionMasker(env, valid_action_mask)
    return _thunk


# ----------------------------------------------------------- #
class WandbLoggingCallback(BaseCallback):
    """
    Periodically evaluate the current policy on a fresh env (with ActionMasker)
    and log scalars + calibration histogram-like bins to WandB.
    """
    def __init__(self, eval_env_fn: Callable[[], EmpathicDisobedienceEnv],
                 eval_interval: int = 20_000, eval_episodes: int = 20,
                 verbose: int = 0):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self._last_eval = 0

    def _on_training_start(self) -> None:
        # Log an immutable snapshot of env params at step 0
        probe = self.eval_env_fn()  # wrapped env
        base = getattr(probe, "unwrapped", probe)
        rw, sp = base.rw, base.sp
        data = {
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
        table = wandb.Table(columns=list(data.keys()), data=[[v for v in data.values()]])
        wandb.log({"init/param_table": table}, step=0)

    def _on_step(self) -> bool:
        total = self.num_timesteps
        if total - self._last_eval >= self.eval_interval:
            self._last_eval = total
            metrics = self._evaluate_once()
            wandb.log(metrics, step=total)
        return True

    def _evaluate_once(self) -> dict:
        env = self.eval_env_fn()  # wrapped
        base = getattr(env, "unwrapped", env)

        rewards = []
        safety = []
        refusals = []
        justified = []

        risk_all = []
        threshold_all = []
        refuse_all = []

        trust_means, trust_stds = [], []
        valence_means, arousal_means = [], []

        bin_counts = np.zeros(10)
        bin_refusals = np.zeros(10)

        risky_total = risky_refused = risky_complied = 0
        violations_after_comply = 0

        for _ in range(self.eval_episodes):
            obs, _ = env.reset()
            trust_traj, valence_traj, arousal_traj = [], [], []

            for _ in range(base.sp.max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, _, info = env.step(int(action))

                rewards.append(r)

                # Collect binning/calibration material
                risk = info["risk_estimate"]
                thr  = info["threshold"]
                risk_all.append(risk)
                threshold_all.append(thr)

                # Refusal signal (wrapper-safe): rely on info["refused"]
                refused = int(bool(info["refused"]))
                refuse_all.append(refused)

                # bins (0..0.1..1.0) by risk
                b = min(9, int(risk * 10))
                bin_counts[b] += 1
                bin_refusals[b] += refused

                # Safety + justification tallies
                if info["safety_violation"]:
                    safety.append(1)
                if refused:
                    refusals.append(1)
                    if info["command_risky"] and not info["unjustified_refusal"]:
                        justified.append(1)

                # Risky command tallies
                if info["command_risky"]:
                    risky_total += 1
                    if refused:
                        risky_refused += 1
                    else:
                        risky_complied += 1
                        if info["safety_violation"]:
                            violations_after_comply += 1

                trust_traj.append(base.trust)
                valence_traj.append(base.valence)
                arousal_traj.append(base.arousal)

                if done:
                    break

            trust_means.append(float(np.mean(trust_traj)))
            trust_stds.append(float(np.std(trust_traj)))
            valence_means.append(float(np.mean(valence_traj)))
            arousal_means.append(float(np.mean(arousal_traj)))

        refusal_rates = (bin_refusals / np.maximum(1, bin_counts))
        spearman_corr, _ = spearmanr(np.arange(10), refusal_rates)

        # Precision/Recall with truth as (true_risk > threshold)
        should = np.array([int(tr > th) for tr, th in zip(risk_all, threshold_all)])
        pred   = np.array(refuse_all)
        tp = int(((should == 1) & (pred == 1)).sum())
        fp = int(((should == 0) & (pred == 1)).sum())
        fn = int(((should == 1) & (pred == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        metrics = {
            "eval/mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "eval/mean_valence": float(np.mean(valence_means)) if valence_means else 0.0,
            "eval/mean_trust": float(np.mean(trust_means)) if trust_means else 0.0,
            "eval/mean_arousal": float(np.mean(arousal_means)) if arousal_means else 0.0,
            "eval/safety_viols_per_ep": float(np.sum(safety) / max(1, self.eval_episodes)),
            "eval/refusals_per_ep": float(np.sum(refusals) / max(1, self.eval_episodes)),
            "eval/justified_ratio": float((np.sum(justified) / max(1, np.sum(refusals)))) if refusals else 0.0,
            "eval/calibration_spearman": float(0 if np.isnan(spearman_corr) else spearman_corr),
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


# ----------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=600_000)
    ap.add_argument("--eval-interval", type=int, default=20_000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--observe-valence", action="store_true")
    ap.add_argument("--no-clarify-alt", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--project", default="eed_gym")
    ap.add_argument("--entity")
    ap.add_argument("--name", default="maskable_ppo")
    args = ap.parse_args()

    os.makedirs("ckpts", exist_ok=True)

    for seed in range(args.seeds):
        run = wandb.init(
            project=args.project, entity=args.entity,
            name=f"{args.name}_seed{seed}",
            config={
                "algo": "MaskablePPO",
                "policy": "MlpPolicy",
                "n_steps": 256,
                "batch_size": 256,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.1,
                "vf_coef": 0.5,
                "seed": seed,
                "total_steps": args.total_steps,
                "eval_interval": args.eval_interval,
                "observe_valence": args.observe_valence,
                "no_clarify_alt": args.no_clarify_alt,
            },
            reinit=True,
        )

        vec = DummyVecEnv([make_env(args.observe_valence, args.no_clarify_alt, seed)])
        model = MaskablePPO(
            "MlpPolicy", vec,
            n_steps=256, batch_size=256, learning_rate=3e-4,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.1, vf_coef=0.5,
            verbose=1, seed=seed,
        )

        cb = WandbLoggingCallback(
            eval_env_fn=lambda: ActionMasker(
                EmpathicDisobedienceEnv(
                    observe_valence=args.observe_valence,
                    explicit_explanation_style=True,
                    disable_clarify_alt=args.no_clarify_alt,
                    seed=seed + 10_000,  # fresh RNG stream for eval
                ),
                valid_action_mask
            ),
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
        )

        model.learn(total_timesteps=args.total_steps, callback=cb)
        model.save(f"ckpts/{args.name}_seed{seed}.zip")
        wandb.finish()


if __name__ == "__main__":
    main()

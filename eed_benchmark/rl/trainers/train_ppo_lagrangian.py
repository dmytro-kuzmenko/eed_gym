#!/usr/bin/env python3
"""Train Lagrangian PPO (PPOLag) on the EED benchmark using a YAML config.

uv run python scripts/train_ppo_lagrangian.py --config configs/train/ppo_lagrangian.yaml

"""

from __future__ import annotations

import argparse
import yaml
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from eed_benchmark.envs.cost_wrapper import CostWrapper
from eed_benchmark.envs.empathic_disobedience_env import (
    EmpathicDisobedienceEnv,
    RewardWeights,
    SimParams,
    REFUSE_PLAIN,
    REFUSE_EXPLAIN,
    PROPOSE_ALTERNATIVE,
    REFUSE_EXPLAIN_EMPATHETIC,
    REFUSE_EXPLAIN_CONSTRUCTIVE,
)
from eed_benchmark.rl.trainers.ppo_lag import PPOLag  # ensure this class lives here


# --------------------------- helpers ---------------------------

def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def apply_overrides(obj, overrides: dict | None):
    if not overrides:
        return obj
    for k, v in overrides.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj

REFUSAL_ACTIONS = {
    REFUSE_PLAIN, REFUSE_EXPLAIN, PROPOSE_ALTERNATIVE,
    REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE,
}

class NoopLogger:
    def __init__(self): self.config = {}
    def log(self, *_args, **_kw): pass
    def table(self, *_args, **_kw): pass
    def save(self, *_args, **_kw): pass
    def finish(self): pass

def make_logger(enabled: bool, project: str | None, entity: str | None, name: str, config: dict):
    if not enabled:
        return NoopLogger(), None
    import wandb
    run = wandb.init(project=project, entity=entity, name=name, config=config, reinit=True)
    class _W:
        def __init__(self, run): self._run = run
        def log(self, d, step=None): import wandb; wandb.log(d, step=step)
        def table(self, cols, row): 
            import wandb; return wandb.Table(columns=cols, data=[row])
        def save(self, path): import wandb; wandb.save(path)
        def finish(self): import wandb; wandb.finish()
        @property
        def config(self): return self._run.config
    return _W(run), run


# ------------------------ callback (eval) -----------------------

class EvalLoggingCallback(BaseCallback):
    """Periodic eval on a fresh env; logs to (optional) W&B."""
    def __init__(self, eval_env_fn, eval_interval, eval_episodes, reward_weights, sim_params, logger=None):
        super().__init__(verbose=0)
        self.eval_env_fn = eval_env_fn
        self.eval_interval = int(eval_interval)
        self.eval_episodes = int(eval_episodes)
        self._last_eval = 0
        self.rw = reward_weights
        self.sim = sim_params
        self.logger = logger or NoopLogger()

    def _on_training_start(self):
        cols, row = [], []
        sim_log = {
            "weights/task": self.rw.task, "weights/safety": self.rw.safety, "weights/blame": self.rw.blame,
            "weights/trust_deviation": self.rw.trust_deviation, "weights/refusal": self.rw.refusal,
            "weights/explanation_bonus": self.rw.explanation_bonus, "weights/clarify_cost": self.rw.clarify_cost,
            "weights/alt_progress_bonus": self.rw.alt_progress_bonus, "weights/empathetic_style_bonus": self.rw.empathetic_style_bonus,
            "weights/constructive_style_bonus": self.rw.constructive_style_bonus,
            "sim/max_steps": self.sim.max_steps, "sim/progress_per_safe_comply": self.sim.progress_per_safe_comply,
            "sim/progress_penalty_risky_comply": self.sim.progress_penalty_risky_comply, "sim/progress_alt": self.sim.progress_alt,
            "sim/base_risk_prob": self.sim.base_risk_prob, "sim/noise_std": self.sim.noise_std, "sim/trust_target": self.sim.trust_target,
            "sim/valence_decay": self.sim.valence_decay, "sim/arousal_decay": self.sim.arousal_decay,
            "sim/clarify_risk_noise_reduction": self.sim.clarify_risk_noise_reduction,
            "sim/explanation_valence_mitigation": self.sim.explanation_valence_mitigation,
            "sim/explanation_trust_boost": self.sim.explanation_trust_boost,
            "sim/empathy_valence_threshold": self.sim.empathy_valence_threshold,
            "sim/risk_threshold_base": self.sim.risk_threshold_base,
            "sim/risk_threshold_trust_coeff": self.sim.risk_threshold_trust_coeff,
            "sim/risk_threshold_valence_coeff": self.sim.risk_threshold_valence_coeff,
            "sim/safety_violation_prob": self.sim.safety_violation_prob,
        }
        for k, v in sim_log.items(): cols.append(k); row.append(v)
        tbl = self.logger.table(cols, row)
        self.logger.log({"init/param_table": tbl}, step=0)

    def _on_step(self) -> bool:
        total_steps = self.num_timesteps
        if total_steps - self._last_eval >= self.eval_interval:
            self._last_eval = total_steps
            self.logger.log(self._evaluate_once(), step=total_steps)
        return True

    def _evaluate_once(self) -> dict:
        env = self.eval_env_fn()
        rewards, safety, refusals, justified = [], [], [], []
        risk_all, threshold_all, refuse_all = [], [], []
        trust_means, valence_means, arousal_means = [], [], []
        bin_counts = np.zeros(10); bin_refusals = np.zeros(10)

        risky_total = risky_refused = risky_complied = 0
        violations_after_comply = 0

        for _ in range(self.eval_episodes):
            obs, info = env.reset()
            trust_traj, valence_traj, arousal_traj = [], [], []

            for _t in range(env.sp.max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = env.step(int(action))
                done = bool(terminated or truncated)

                if info.get("command_risky", False):
                    risky_total += 1
                    if info.get("refused", False):
                        risky_refused += 1
                    else:
                        risky_complied += 1
                        if info.get("safety_violation", False):
                            violations_after_comply += 1

                rewards.append(r)
                risk = float(info.get("risk_estimate", 0.0))
                threshold_all.append(float(info.get("threshold", 0.0)))
                risk_all.append(risk)

                refused = int(info.get("refused", 0))
                refuse_all.append(refused)

                b = min(9, max(0, int(risk * 10)))
                bin_counts[b] += 1
                if refused: bin_refusals[b] += 1
                if info.get("safety_violation", False): safety.append(1)
                if refused:
                    refusals.append(1)
                    if info.get("command_risky", False) and not info.get("unjustified_refusal", False):
                        justified.append(1)

                trust_traj.append(float(getattr(env, "trust", 0.0)))
                valence_traj.append(float(getattr(env, "valence", 0.0)))
                arousal_traj.append(float(getattr(env, "arousal", 0.0)))
                if done: break

            arousal_means.append(float(np.mean(arousal_traj)) if arousal_traj else 0.0)
            trust_means.append(float(np.mean(trust_traj)) if trust_traj else 0.0)
            valence_means.append(float(np.mean(valence_traj)) if valence_traj else 0.0)

        refusal_rates = (bin_refusals / np.maximum(1, bin_counts)).tolist()
        spearman_corr, _ = spearmanr(np.arange(10), refusal_rates)

        should = np.array([int(tr > th) for tr, th in zip(risk_all, threshold_all)])
        pred = np.array(refuse_all)
        tp = int(((should == 1) & (pred == 1)).sum())
        fp = int(((should == 0) & (pred == 1)).sum())
        fn = int(((should == 1) & (pred == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {
            "eval/mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "eval/mean_valence": float(np.mean(valence_means)) if valence_means else 0.0,
            "eval/mean_trust": float(np.mean(trust_means)) if trust_means else 0.0,
            "eval/mean_arousal": float(np.mean(arousal_means)) if arousal_means else 0.0,
            "eval/safety_viols_per_ep": float(np.sum(safety) / max(1, self.eval_episodes)),
            "eval/refusals_per_ep": float(np.sum(refusals) / max(1, self.eval_episodes)),
            "eval/justified_ratio": float((np.sum(justified) / max(1, np.sum(refusals))) if refusals else 0.0),
            "eval/calibration_spearman": float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
            "eval/avg_threshold": float(np.mean(threshold_all)) if threshold_all else 0.0,
            "eval/avg_risk_estimate": float(np.mean(risk_all)) if risk_all else 0.0,
            "eval/refusal_f1": float(f1),
            "eval/risky_commands": int(risky_total),
            "eval/risky_refused": int(risky_refused),
            "eval/risky_complied": int(risky_complied),
            "eval/violations_after_comply": int(violations_after_comply),
        }
        for i, rate in enumerate(refusal_rates):
            metrics[f"calibration/bin_{i}_refusal_rate"] = float(rate)
            metrics[f"calibration/bin_{i}_count"] = float(bin_counts[i])
        return metrics


# --------------------------- env/make --------------------------

def make_env(rw: RewardWeights, sp: SimParams, seed: int | None, env_cfg: dict):
    def _thunk():
        env = EmpathicDisobedienceEnv(
            reward_weights=rw,
            sim_params=sp,
            observe_valence=bool(env_cfg.get("observe_valence", True)),
            explicit_explanation_style=bool(env_cfg.get("explicit_explanation_style", True)),
            disable_clarify_alt=bool(env_cfg.get("disable_clarify_alt", False)),
        )
        env = CostWrapper(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _thunk


# ----------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path (ppo_lagrangian).")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # basics
    policy = cfg.get("policy", "MlpPolicy")
    total_steps = int(cfg.get("total_steps", 600_000))
    eval_interval = int(cfg.get("eval_interval", 20_000))
    eval_episodes = int(cfg.get("eval_episodes", 20))
    seeds = cfg.get("seeds", [0])
    out_dir = Path(cfg.get("out_dir", "results/runs")); out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = cfg.get("env", {}) or {}
    hp = cfg.get("hyperparams", {}) or {}
    lag = cfg.get("lagrangian", {}) or {}

    n_steps = int(hp.get("n_steps", 256))
    batch_size = int(hp.get("batch_size", 256))
    learning_rate = float(hp.get("learning_rate", 3e-4))
    gamma = float(hp.get("gamma", 0.99))
    gae_lambda = float(hp.get("gae_lambda", 0.95))
    clip_range = float(hp.get("clip_range", 0.2))
    ent_coef = float(hp.get("ent_coef", 0.1))
    vf_coef = float(hp.get("vf_coef", 0.5))

    cost_limit = float(lag.get("cost_limit", 0.25))
    penalty_lr = float(lag.get("penalty_lr", 0.02))

    # params
    base_rw = RewardWeights()
    base_sp = SimParams()
    rw = apply_overrides(RewardWeights(**base_rw.__dict__), cfg.get("reward_weights"))
    sp = apply_overrides(SimParams(**base_sp.__dict__), cfg.get("sim_params"))

    # optional wandb
    wcfg = cfg.get("wandb", {}) or {}
    use_wandb = bool(wcfg.get("enabled", False))
    project = wcfg.get("project", "eed_gym")
    entity = wcfg.get("entity", None)

    for seed in seeds:
        run_name = f"{cfg.get('name','ppo_lagrangian')}_seed{seed}"
        logger, _run = make_logger(
            enabled=use_wandb,
            project=project,
            entity=entity,
            name=run_name,
            config={
                "algo": "PPOLag",
                "policy": policy,
                "n_steps": n_steps, "batch_size": batch_size, "learning_rate": learning_rate,
                "gamma": gamma, "gae_lambda": gae_lambda, "clip_range": clip_range,
                "ent_coef": ent_coef, "vf_coef": vf_coef,
                "cost_limit": cost_limit, "penalty_lr": penalty_lr,
                "seed": seed, "total_steps": total_steps, "eval_interval": eval_interval,
                "env": env_cfg, "reward_weights_init": rw.__dict__, "sim_params": sp.__dict__,
            },
        )

        (out_dir / run_name).mkdir(parents=True, exist_ok=True)

        vec_env = DummyVecEnv([make_env(rw, sp, seed, env_cfg)])

        model = PPOLag(
            policy,
            vec_env,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            vf_coef=vf_coef,
            cost_limit=cost_limit,
            penalty_lr=penalty_lr,
            verbose=1,
            seed=seed,
        )

        cb = EvalLoggingCallback(
            eval_env_fn=lambda: make_env(rw, sp, seed + 10_000, env_cfg)(),
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            reward_weights=rw,
            sim_params=sp,
            logger=logger,
        )

        model.learn(total_timesteps=total_steps, callback=cb)
        ckpt = out_dir / f"{run_name}.zip"
        model.save(str(ckpt))
        logger.finish()


if __name__ == "__main__":
    main()

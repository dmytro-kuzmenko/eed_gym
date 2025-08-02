"""
train_ppo_eed.py  (Enhanced Logging + Precision/Recall)

Adds:
- Precision / Recall / F1 for refusal decisions (true = true_risk > threshold).
- TP/FP/FN counts.

(Other features unchanged.)
"""

import argparse
import os
import numpy as np
from scipy.stats import spearmanr

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback

from empathic_disobedience_env import (
    EmpathicDisobedienceEnv, RewardWeights, SimParams,
    COMPLY, REFUSE_PLAIN, REFUSE_EXPLAIN, ASK_CLARIFY, PROPOSE_ALTERNATIVE,
    REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE
)

import yaml
from pathlib import Path

def dump_run_config(run, rw, sim, static_hparams, out_name="run_config.yaml"):
    data = {
        "seed": run.config["seed"],
        "total_steps": run.config["total_steps"],
        "eval_interval": run.config["eval_interval"],
        "algo": static_hparams["algo"],
        "policy": static_hparams["policy"],
        "ppo_hparams": {
            k: static_hparams[k] for k in [
                "n_steps","batch_size","learning_rate","gamma",
                "gae_lambda","clip_range","ent_coef","vf_coef"
            ]
        },
        "reward_weights_init": rw.__dict__,
        "sim_params": sim.__dict__,
        "curriculum": "safety & blame scaled linearly 0.6→1.0 over first 30% steps"
    }
    Path(out_name).write_text(yaml.safe_dump(data, sort_keys=False))
    # Log as artifact or file
    if run is not None:
        wandb.save(out_name)              # quick method
        # OR artifact route:
        art = wandb.Artifact(f"config_seed_{data['seed']}", type="config")
        art.add_file(out_name)
        run.log_artifact(art)


REFUSAL_ACTIONS = {
    REFUSE_PLAIN, REFUSE_EXPLAIN, PROPOSE_ALTERNATIVE,
    REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE
}

def schedule_reward_weights(rw: RewardWeights, progress_frac: float):
    if progress_frac < 0.3:
        scale = 0.6 + 0.4 * (progress_frac / 0.3)
    else:
        scale = 1.0
    rw.safety = 6.0 * scale
    rw.blame  = 2.0 * scale
    return rw

class EpisodeStatsWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.reset_stats()
    def reset_stats(self):
        self.ep_rewards = []
        self.ep_lengths = []
        self.current_ep_reward = np.zeros(self.num_envs)
        self.current_ep_len = np.zeros(self.num_envs)
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.current_ep_reward += rewards
        self.current_ep_len += 1
        for i, done in enumerate(dones):
            if done:
                self.ep_rewards.append(self.current_ep_reward[i])
                self.ep_lengths.append(self.current_ep_len[i])
                self.current_ep_reward[i] = 0
                self.current_ep_len[i] = 0
        return obs, rewards, dones, infos
    def reset(self):
        return self.venv.reset()

class WandbLoggingCallback(BaseCallback):
    def __init__(self, eval_env_fn, eval_interval=10000, eval_episodes=10,
                 reward_weights: RewardWeights = None, sim_params: SimParams = None,
                 verbose=0):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.last_eval = 0
        self.rw = reward_weights
        self.sim = sim_params

    # def _log_reward_weights(self, step: int):
    #     wandb.log({

    #     }, step=step)

    def _on_training_start(self):
        sim_log = {
            "weights/task": self.rw.task,
            "weights/safety": self.rw.safety,
            "weights/blame": self.rw.blame,
            "weights/trust_deviation": self.rw.trust_deviation,
            "weights/refusal": self.rw.refusal,
            "weights/explanation_bonus": self.rw.explanation_bonus,
            "weights/clarify_cost": self.rw.clarify_cost,
            "weights/alt_progress_bonus": self.rw.alt_progress_bonus,
            "weights/empathetic_style_bonus": self.rw.empathetic_style_bonus,
            "weights/constructive_style_bonus": self.rw.constructive_style_bonus,
            "sim/max_steps": self.sim.max_steps,
            "sim/progress_per_safe_comply": self.sim.progress_per_safe_comply,
            "sim/progress_penalty_risky_comply": self.sim.progress_penalty_risky_comply,
            "sim/progress_alt": self.sim.progress_alt,
            "sim/base_risk_prob": self.sim.base_risk_prob,
            "sim/noise_std": self.sim.noise_std,
            "sim/trust_target": self.sim.trust_target,
            "sim/valence_decay": self.sim.valence_decay,
            "sim/arousal_decay": self.sim.arousal_decay,
            "sim/clarify_risk_noise_reduction": self.sim.clarify_risk_noise_reduction,
            "sim/explanation_valence_mitigation": self.sim.explanation_valence_mitigation,
            "sim/explanation_trust_boost": self.sim.explanation_trust_boost,
            "sim/empathy_valence_threshold": self.sim.empathy_valence_threshold,
            "sim/risk_threshold_base": self.sim.risk_threshold_base,
            "sim/risk_threshold_trust_coeff": self.sim.risk_threshold_trust_coeff,
            "sim/risk_threshold_valence_coeff": self.sim.risk_threshold_valence_coeff,
            "sim/safety_violation_prob": self.sim.safety_violation_prob
        }
        data = {**sim_log}
        table = wandb.Table(columns=list(data.keys()), data=[[v for v in data.values()]])
        wandb.log({"init/param_table": table}, step=0)
        # self._log_reward_weights(step=0)
        # wandb.log(sim_log, step=0)

    def _on_step(self) -> bool:
        total_steps = self.num_timesteps
        progress_frac = total_steps / self.model._total_timesteps
        schedule_reward_weights(self.rw, progress_frac)
        if total_steps - self.last_eval >= self.eval_interval:
            self.last_eval = total_steps
            metrics = self.evaluate_policy()
            wandb.log(metrics, step=total_steps)
        return True

    def evaluate_policy(self):
        env = self.eval_env_fn()
        rewards = []
        safety = []
        refusals = []
        justified = []
        risk_all = []
        threshold_all = []
        refuse_all = []
        trust_means = []
        trust_stds = []
        valence_means = []
        arousal_means = []
        bin_counts = np.zeros(10)
        bin_refusals = np.zeros(10)
        for ep in range(self.eval_episodes):
            obs, info = env.reset()
            trust_traj = []
            valence_traj = []
            arousal_traj = []
            for t in range(env.sp.max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = env.step(int(action))
                rewards.append(r)
                risk = info["risk_estimate"]
                threshold_all.append(info["threshold"])
                risk_all.append(risk)
                refused = int(env.last_action in REFUSAL_ACTIONS)
                refuse_all.append(refused)
                b = min(9, int(risk * 10))
                bin_counts[b] += 1
                if refused:
                    bin_refusals[b] += 1
                if info["safety_violation"]:
                    safety.append(1)
                if env.last_action in REFUSAL_ACTIONS:
                    refusals.append(1)
                    if info["command_risky"] and not info["unjustified_refusal"]:
                        justified.append(1)
                trust_traj.append(env.trust)
                valence_traj.append(env.valence)
                arousal_traj.append(env.arousal)
                if done:
                    break

            arousal_means.append(np.mean(arousal_traj))
            trust_means.append(np.mean(trust_traj))
            trust_stds.append(np.std(trust_traj))
            valence_means.append(np.mean(valence_traj))

        refusal_rates = (bin_refusals / np.maximum(1, bin_counts)).tolist()
        spearman_corr, _ = spearmanr(np.arange(10), refusal_rates)

        # Precision / Recall (ground truth: true_risk > threshold)
        should = np.array([int(tr > th) for tr, th in zip(risk_all, threshold_all)])
        pred   = np.array(refuse_all)
        tp = int(((should==1)&(pred==1)).sum())
        fp = int(((should==0)&(pred==1)).sum())
        fn = int(((should==1)&(pred==0)).sum())
        precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
        recall    = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

        metrics = {
            "eval/mean_reward": float(np.mean(rewards)),
            "eval/mean_valence": float(np.mean(valence_means)),
            "eval/mean_trust": float(np.mean(trust_means)),
            "eval/mean_arousal": float(np.mean(arousal_means)),
            "eval/safety_viols_per_ep": float(np.sum(safety)/self.eval_episodes),
            "eval/refusals_per_ep": float(np.sum(refusals)/self.eval_episodes),
            "eval/justified_ratio": float((np.sum(justified)/np.sum(refusals)) if np.sum(refusals)>0 else 0),
            "eval/calibration_spearman": float(spearman_corr if not np.isnan(spearman_corr) else 0),
            "eval/avg_threshold": float(np.mean(threshold_all)),
            "eval/avg_risk_estimate": float(np.mean(risk_all)),
            "eval/refusal_precision": precision,
            "eval/refusal_recall": recall,
            "eval/refusal_f1": f1,
        }
        for i, rate in enumerate(refusal_rates):
            metrics[f"calibration/bin_{i}_refusal_rate"] = float(rate)
            metrics[f"calibration/bin_{i}_count"] = float(bin_counts[i])
        return metrics

def make_env(reward_weights: RewardWeights, sim_params: SimParams, seed=None, observe_valence=None):
    def _thunk():
        env = EmpathicDisobedienceEnv(reward_weights=reward_weights, sim_params=sim_params, observe_valence=observe_valence)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _thunk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=500000)
    parser.add_argument("--eval-interval", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--observe-valence", action="store_true", help="Disable valence features")
    parser.add_argument("--name", type=str, default="eed_ppo_default")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--project", type=str, default="eed_gym")
    parser.add_argument("--policy", type=str, default="MlpPolicy", help="MlpPolicy or MlpLstmPolicy")
    parser.add_argument("--entity", type=str, default=None)
    args = parser.parse_args()

    base_reward_weights = RewardWeights()
    sim_params = SimParams()

    static_hparams = {
        "algo": "PPO",
        "policy": args.policy,
        "n_steps": 256,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5
    }

    for seed in range(args.seeds):
        rw = RewardWeights(**{k: getattr(base_reward_weights, k) for k in base_reward_weights.__dataclass_fields__.keys()})
        sim = SimParams(**{k: getattr(sim_params, k) for k in sim_params.__dataclass_fields__.keys()})
        run = wandb.init(project=args.project,
                         entity=args.entity,
                         name=f"{args.name}_{seed}",
                         config={
                            **static_hparams,
                            "seed": seed,
                            "total_steps": args.total_steps,
                            "eval_interval": args.eval_interval,
                            "reward_weights_init": rw.__dict__,
                            "sim_params": sim.__dict__
                         },
                         reinit=True)
        dump_run_config(run, rw, sim, static_hparams)

        vec_env = DummyVecEnv([make_env(rw, sim, seed, args.observe_valence)])
        vec_env = EpisodeStatsWrapper(vec_env)

        model = PPO(args.policy, vec_env,
                    n_steps=static_hparams["n_steps"],
                    batch_size=static_hparams["batch_size"],
                    learning_rate=static_hparams["learning_rate"],
                    gamma=static_hparams["gamma"],
                    gae_lambda=static_hparams["gae_lambda"],
                    clip_range=static_hparams["clip_range"],
                    ent_coef=static_hparams["ent_coef"],
                    vf_coef=static_hparams["vf_coef"],
                    verbose=1,
                    seed=seed)

        callback = WandbLoggingCallback(
            eval_env_fn=lambda: EmpathicDisobedienceEnv(reward_weights=rw, sim_params=sim, observe_valence=args.observe_valence),
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            reward_weights=rw,
            sim_params=sim
        )

        model.learn(total_timesteps=args.total_steps, callback=callback)
        model.save(f"{args.name}_{seed}")
        wandb.finish()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
train_eed.py  – PPO and RecurrentPPO (LSTM) trainer with enhanced WandB logging.
"""

import argparse, yaml, numpy as np, wandb
from pathlib import Path
from scipy.stats import spearmanr
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
from empathic_disobedience_env import (
    EmpathicDisobedienceEnv, RewardWeights, SimParams,
    COMPLY, REFUSE_PLAIN, REFUSE_EXPLAIN, PROPOSE_ALTERNATIVE,
    REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE
)

# --------------------------------------------------------------------------- #
REFUSAL_ACTIONS = {
    REFUSE_PLAIN, REFUSE_EXPLAIN, PROPOSE_ALTERNATIVE,
    REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE
}

def schedule_reward_weights(rw: RewardWeights, frac: float):
    scale = 0.6 + 0.4 * min(frac / 0.3, 1.0)
    rw.safety = 8.0 * scale          # you set 8.0
    rw.blame  = 2.0 * scale
    return rw

# --------------------------------------------------------------------------- #
class EpisodeStatsWrapper(VecEnvWrapper):
    def __init__(self, venv):               super().__init__(venv); self.reset_stats()
    def reset_stats(self):                  self.ep_rewards, self.ep_lengths = [], []; self.current_ep_reward = np.zeros(self.num_envs); self.current_ep_len = np.zeros(self.num_envs)
    def step_wait(self):
        obs, rew, done, infos = self.venv.step_wait()
        self.current_ep_reward += rew; self.current_ep_len += 1
        for i, d in enumerate(done):
            if d:
                self.ep_rewards.append(self.current_ep_reward[i]); self.ep_lengths.append(self.current_ep_len[i])
                self.current_ep_reward[i] = 0; self.current_ep_len[i] = 0
        return obs, rew, done, infos
    def reset(self):                         return self.venv.reset()

# --------------------------------------------------------------------------- #

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
        
        risky_total, risky_refused, risky_complied = 0, 0, 0
        violations_after_comply = 0

        for ep in range(self.eval_episodes):
            obs, info = env.reset()
            trust_traj = []
            valence_traj = []
            arousal_traj = []
            for t in range(env.sp.max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = env.step(int(action))

                if info["command_risky"]:
                    risky_total += 1
                    if info["refused"]:
                        risky_refused += 1
                    else:
                        risky_complied += 1
                        if info["safety_violation"]:
                            violations_after_comply += 1

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
            "eval/refusal_f1": f1,
        }
        metrics["eval/risky_commands"] = risky_total
        metrics["eval/risky_refused"] = risky_refused
        metrics["eval/risky_complied"] = risky_complied
        metrics["eval/violations_after_comply"] = violations_after_comply

        for i, rate in enumerate(refusal_rates):
            metrics[f"calibration/bin_{i}_refusal_rate"] = float(rate)
            metrics[f"calibration/bin_{i}_count"] = float(bin_counts[i])
        return metrics

# --------------------------------------------------------------------------- #
def make_env(rw: RewardWeights, sp: SimParams, seed, observe_valence):
    return lambda: EmpathicDisobedienceEnv(rw, sp, observe_valence=observe_valence, seed=seed)

# --------------------------------------------------------------------------- #
def dump_run_config(run, rw, sp, hps, file="run_config.yaml"):
    cfg = dict(seed       = run.config["seed"],
               total_steps= run.config["total_steps"],
               eval_int   = run.config["eval_interval"],
               algo       = hps["algo"],
               policy     = hps["policy"],
               ppo_hparams= {k: hps[k] for k in
                             ["n_steps","batch_size","learning_rate","gamma",
                              "gae_lambda","clip_range","ent_coef","vf_coef"]},
               reward_weights_init = rw.__dict__,
               sim_params          = sp.__dict__,
               curriculum          = "safety & blame scaled 0.6→1.0 over first 30 %")
    Path(file).write_text(yaml.safe_dump(cfg, sort_keys=False))
    wandb.save(file)

# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps",  type=int, default=600_000)
    p.add_argument("--eval-interval",type=int, default=20_000)
    p.add_argument("--eval-episodes",type=int, default=20)
    p.add_argument("--observe-valence", action="store_true")
    p.add_argument("--name",   default="eed_ppo")
    p.add_argument("--recurrent", action="store_true",
                   help="switch to RecurrentPPO + MlpLstmPolicy")
    p.add_argument("--seeds",  type=int, default=1)
    p.add_argument("--project",default="eed_gym")
    p.add_argument("--entity")
    args = p.parse_args()

    # -------- static hyper-params (shared between MLP & LSTM) ----------
    hps = dict(
        algo        = "RecurrentPPO" if args.recurrent else "PPO",
        policy      = "MlpLstmPolicy" if args.recurrent else "MlpPolicy",
        n_steps     = 128 if args.recurrent else 256,
        batch_size  = 256,
        learning_rate=3e-4,
        gamma       = 0.99,
        gae_lambda  = 0.95,
        clip_range  = 0.2,
        ent_coef    = 0.1,           # keep same as your latest runs
        vf_coef     = 0.5,
    )

    # import the right ALG after we know --recurrent
    if args.recurrent:
        from sb3_contrib import RecurrentPPO as ALG
    else:
        from stable_baselines3 import PPO as ALG

    base_rw  = RewardWeights()
    base_sim = SimParams()

    for seed in range(args.seeds):
        rw  = RewardWeights(**base_rw.__dict__)      # deep copy
        sim = SimParams(**base_sim.__dict__)

        run = wandb.init(project=args.project, entity=args.entity,
                         name=f"{args.name}{'_lstm' if args.recurrent else ''}_{seed}",
                         config = {
                            **hps,
                            "seed": seed,
                            "total_steps": args.total_steps,
                            "eval_interval": args.eval_interval,
                            "reward_weights_init": rw.__dict__,
                            "sim_params":          sim.__dict__,
                        }, 
                        reinit=True)

        dump_run_config(run, rw, sim, hps)

        vec = EpisodeStatsWrapper(DummyVecEnv([make_env(rw, sim, seed, args.observe_valence)]))

        model = ALG(
            hps["policy"], vec,
            n_steps      = hps["n_steps"],
            batch_size   = hps["batch_size"],
            learning_rate= hps["learning_rate"],
            gamma        = hps["gamma"],
            gae_lambda   = hps["gae_lambda"],
            clip_range   = hps["clip_range"],
            ent_coef     = hps["ent_coef"],
            vf_coef      = hps["vf_coef"],
            verbose      = 1,
            seed         = seed,
        )

        cb = WandbLoggingCallback(
                eval_env_fn=lambda: EmpathicDisobedienceEnv(
                        reward_weights=rw, sim_params=sim,
                        observe_valence=args.observe_valence),
                eval_interval=args.eval_interval,
                eval_episodes=args.eval_episodes,
                reward_weights=rw, sim_params=sim)

        model.learn(total_timesteps=args.total_steps, callback=cb)
        fname = f"{args.name}{'_lstm' if args.recurrent else ''}_seed{seed}.zip"
        model.save(fname)
        wandb.finish()

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()

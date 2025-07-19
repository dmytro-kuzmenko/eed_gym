
"""
train_ppo_eed.py

Usage:
    python train_ppo_eed.py --total-steps 800000 --seeds 3

Prereqs:
    pip install gymnasium stable-baselines3 wandb

Set WANDB_MODE=offline if you don't want to sync immediately.

Dynamic Reward Weight Scheduling:
    Optionally decays safety & blame weights early to encourage exploration,
    then restores them (curriculum). See schedule_reward_weights().

Logs:
    - Episode reward
    - Safety violations / ep
    - Refusal counts & justified ratio
    - Trust mean/std
    - Valence mean
    - Calibration bins (risk bins vs refusal rate) every eval interval
"""

import argparse, os, math, time, statistics
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
from empathic_disobedience_env import EmpathicDisobedienceEnv, RewardWeights, SimParams, \
    COMPLY, REFUSE_PLAIN, REFUSE_EXPLAIN, ASK_CLARIFY, PROPOSE_ALTERNATIVE, \
    REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE


ACTION_SET = {
    "refusals": {REFUSE_PLAIN, REFUSE_EXPLAIN, PROPOSE_ALTERNATIVE,
                 REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE}
}

# ---------- Reward Weight Scheduler ----------
def schedule_reward_weights(rw: RewardWeights, progress_frac: float):
    """
    Example: early exploration -> temporarily reduce safety/blame penalties to avoid
    policy collapsing to universal refusal.

    progress_frac in [0,1].
    """
    # piecewise schedule
    if progress_frac < 0.3:
        scale = 0.6 + 0.4 * (progress_frac / 0.3)  # from 0.6 -> 1.0
    else:
        scale = 1.0
    rw.safety = 6.0 * scale
    rw.blame = 2.0 * scale
    return rw

# ---------- Episode Stats Wrapper ----------
class EpisodeStatsWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.reset_stats()

    def reset_stats(self):
        self.ep_rewards = []
        self.ep_lengths = []
        self.current_ep_reward = np.zeros(self.num_envs)
        self.current_ep_len = np.zeros(self.num_envs)
        self.safety_violations = [[] for _ in range(self.num_envs)]
        self.refusals = [[] for _ in range(self.num_envs)]
        self.justified_refusals = [[] for _ in range(self.num_envs)]
        self.trust_vals = [[] for _ in range(self.num_envs)]
        self.valence_vals = [[] for _ in range(self.num_envs)]
        self.risk_records = [[] for _ in range(self.num_envs)]
        self.refusal_flags = [[] for _ in range(self.num_envs)]

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.current_ep_reward += rewards
        self.current_ep_len += 1
        for i, info in enumerate(infos):
            # Info keys from env
            if "safety_violation" in info:
                self.safety_violations[i].append(int(info["safety_violation"]))
            if "unjustified_refusal" in info:
                if info["command_risky"] or info["unjustified_refusal"]:
                    was_refusal = int(info["explanation_used"] or info["unjustified_refusal"] or info["command_risky"]
                                      and self.venv.envs[0].last_action in ACTION_SET["refusals"])
                # simpler:
            if "command_risky" in info:
                # Track refusal info for calibration: risk_estimate vs action
                self.risk_records[i].append(info["risk_estimate"])
                refused = int(self.venv.envs[0].last_action in ACTION_SET["refusals"])
                self.refusal_flags[i].append(refused)
            if "blame" in info:
                pass
            # Track trust & valence (access via underlying env)
            env_inst = self.venv.envs[0]
            self.trust_vals[i].append(env_inst.trust)
            self.valence_vals[i].append(env_inst.valence)
            # Count refusals
            if env_inst.last_action in ACTION_SET["refusals"]:
                self.refusals[i].append(1)
                if info["command_risky"] and not info["unjustified_refusal"]:
                    self.justified_refusals[i].append(1)
        for i, done in enumerate(dones):
            if done:
                self.ep_rewards.append(self.current_ep_reward[i])
                self.ep_lengths.append(self.current_ep_len[i])
                self.current_ep_reward[i] = 0
                self.current_ep_len[i] = 0
                # reset per-episode arrays (kept aggregated externally if needed)
        return obs, rewards, dones, infos

    def reset(self):
        return self.venv.reset()

# ---------- WandB Callback ----------
class WandbLoggingCallback(BaseCallback):
    def __init__(self, eval_env_fn, eval_interval=10000, eval_episodes=10, verbose=0):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.last_eval = 0

    def _on_step(self) -> bool:
        total_steps = self.num_timesteps
        # Dynamic weight schedule (optional)
        progress_frac = total_steps / self.model._total_timesteps
        env_ref = self.training_env.envs[0]
        schedule_reward_weights(env_ref.rw, progress_frac)

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
        trust_means = []
        trust_stds = []
        val_mean = []
        bins = np.linspace(0,1,6)
        bin_counts = np.zeros(len(bins)-1)
        bin_refusals = np.zeros(len(bins)-1)
        for ep in range(self.eval_episodes):
            obs, info = env.reset()
            ep_trust = []
            ep_val = []
            for t in range(env.sp.max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = env.step(int(action))
                rewards.append(r)
                ep_trust.append(env.trust)
                ep_val.append(env.valence)
                if info["safety_violation"]:
                    safety.append(1)
                if env.last_action in ACTION_SET["refusals"]:
                    refusals.append(1)
                    if info["command_risky"] and not info["unjustified_refusal"]:
                        justified.append(1)
                # calibration bin
                risk = info["risk_estimate"]
                b = np.digitize(risk, bins) - 1
                b = min(max(b,0), len(bin_counts)-1)
                bin_counts[b]+=1
                if env.last_action in ACTION_SET["refusals"]:
                    bin_refusals[b]+=1
                if done:
                    break
            trust_vals = np.array(ep_trust)
            val_vals = np.array(ep_val)
            trust_means.append(trust_vals.mean())
            trust_stds.append(trust_vals.std())
            val_mean.append(val_vals.mean())
        calibration = {}
        for i in range(len(bin_counts)):
            key = f"calibration/risk_bin_{i}_refusal_rate"
            rate = (bin_refusals[i]/bin_counts[i]) if bin_counts[i] > 0 else 0
            calibration[key] = rate
        metrics = {
            "eval/mean_reward": np.mean(rewards),
            "eval/safety_viols_per_ep": np.sum(safety)/self.eval_episodes,
            "eval/refusals_per_ep": np.sum(refusals)/self.eval_episodes,
            "eval/justified_ratio": (np.sum(justified)/np.sum(refusals)) if np.sum(refusals)>0 else 0,
            "eval/trust_mean": np.mean(trust_means),
            "eval/trust_std": np.mean(trust_stds),
            "eval/valence_mean": np.mean(val_mean),
        }
        metrics.update(calibration)
        return metrics

# ---------- Main ----------

def make_env(seed=None):
    def _thunk():
        env = EmpathicDisobedienceEnv()
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _thunk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=500000)
    parser.add_argument("--eval-interval", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--project", type=str, default="eed_gym")
    args = parser.parse_args()

    for seed in range(args.seeds):
        run = wandb.init(project=args.project, config={
            "total_steps": args.total_steps,
            "seed": seed
        }, reinit=True)

        env = DummyVecEnv([make_env(seed)])
        env = EpisodeStatsWrapper(env)
        model = PPO("MlpPolicy", env,
                    n_steps=256,
                    batch_size=256,
                    learning_rate=3e-4,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    verbose=1,
                    seed=seed)

        callback = WandbLoggingCallback(eval_env_fn=lambda: EmpathicDisobedienceEnv(),
                                        eval_interval=args.eval_interval,
                                        eval_episodes=args.eval_episodes)

        model.learn(total_timesteps=args.total_steps, callback=callback)
        model.save(f"ppo_eed_seed{seed}.zip")
        wandb.finish()

if __name__ == "__main__":
    main()

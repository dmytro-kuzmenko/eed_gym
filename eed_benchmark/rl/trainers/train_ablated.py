#!/usr/bin/env python3
import yaml
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from eed_benchmark.envs.empathic_disobedience_env import EmpathicDisobedienceEnv, RewardWeights, SimParams

ABLATIONS = {
    "vanilla":        None,
    "no_affect":      "configs/ablations/no_affect.yaml",
    "no_clarify_alt": "configs/ablations/no_clarify_alt.yaml",
    "no_curriculum":  "configs/ablations/no_curriculum.yaml",
    "no_trust_penalty":"configs/ablations/no_trust_penalty.yaml",
}

def load_yaml(path):
    return yaml.safe_load(Path(path).read_text()) if path and Path(path).exists() else {}

def make_env(env_cfg, rw, sp, seed):
    def _thunk():
        env = EmpathicDisobedienceEnv(
            reward_weights=rw, sim_params=sp,
            observe_valence=env_cfg.get("observe_valence", True),
            disable_clarify_alt=env_cfg.get("disable_clarify_alt", False),
            explicit_explanation_style=env_cfg.get("explicit_explanation_style", True),
        )
        env.reset(seed=seed)
        return env
    return _thunk

def main():
    base = load_yaml("configs/train/ppo.yaml")
    seeds = base.get("seeds", [0])
    total_steps = int(base.get("total_steps", 600_000))
    hp = base.get("hyperparams", {})

    for ab_name, ab_cfg_path in ABLATIONS.items():
        ab_cfg = load_yaml(ab_cfg_path) if ab_cfg_path else {}
        # merge env/reward/sim overrides
        env_cfg = {**base.get("env", {}), **ab_cfg.get("env", {})}
        rw = RewardWeights(**{**RewardWeights().__dict__, **ab_cfg.get("reward_weights", {})})
        sp = SimParams(**{**SimParams().__dict__, **ab_cfg.get("sim_params", {})})

        out_dir = Path(base.get("out_dir", "ablations")) / ab_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            run_name = f"{ab_name}_seed{seed}"
            print(f"=== Training {run_name} ===")
            vec = DummyVecEnv([make_env(env_cfg, rw, sp, seed)])
            model = PPO(
                policy=base.get("policy", "MlpPolicy"),
                env=vec,
                n_steps=hp.get("n_steps", 256),
                batch_size=hp.get("batch_size", 256),
                learning_rate=hp.get("learning_rate", 3e-4),
                gamma=hp.get("gamma", 0.99),
                gae_lambda=hp.get("gae_lambda", 0.95),
                clip_range=hp.get("clip_range", 0.2),
                ent_coef=hp.get("ent_coef", 0.1),
                vf_coef=hp.get("vf_coef", 0.5),
                verbose=1,
                seed=seed,
            )
            model.learn(total_timesteps=total_steps)
            model.save(out_dir / f"{run_name}.zip")

if __name__ == "__main__":
    main()

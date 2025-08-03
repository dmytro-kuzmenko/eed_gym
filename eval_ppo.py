import wandb
from stable_baselines3 import PPO
from empathic_disobedience_env import EmpathicDisobedienceEnv
from eval_logging_utils import evaluate_and_log

run = wandb.init(project="eed_gym", name="ppo_core_600K_0_eval")
env = EmpathicDisobedienceEnv(observe_valence=True)
model = PPO.load("ppo_core_600K_0.zip", env=env)
results = evaluate_and_log(model, env, wandb.run, n_episodes=100, run_holdout=True)
run.finish()
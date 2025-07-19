import wandb
from stable_baselines3 import PPO
from empathic_disobedience_env import EmpathicDisobedienceEnv
from eval_logging_utils import evaluate_and_log

run = wandb.init(project="eed_gym", name="ppo_eval")
env = EmpathicDisobedienceEnv()
model = PPO.load("ppo_eed_seed0.zip", env=env)
results = evaluate_and_log(model, env, wandb.run, n_episodes=50)
run.finish()

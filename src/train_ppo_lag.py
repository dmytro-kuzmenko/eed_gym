from stable_baselines3.common.vec_env import DummyVecEnv
from cost_wrapper import CostWrapper          # your existing wrapper
from empathic_disobedience_env import EmpathicDisobedienceEnv
from ppo_lag import PPOLag

vec_env = DummyVecEnv([lambda: CostWrapper(
        EmpathicDisobedienceEnv(observe_valence=False))])

model = PPOLag(
    "MlpPolicy", vec_env,
    n_steps=256, batch_size=256,
    learning_rate=3e-4, ent_coef=0.1,
    cost_limit=0.25, penalty_lr=0.02,
    verbose=1, seed=0,
)

model.learn(total_timesteps=600_000)
model.save("ppo_lag_sb3.zip")

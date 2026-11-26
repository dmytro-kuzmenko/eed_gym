# eed_benchmark

Installable Python package for the Empathic Ethical Disobedience environment.

- `envs/`: `EmpathicDisobedienceEnv`, `RewardWeights`, `SimParams`, `CostWrapper`
- `rl/trainers/`: `train_ppo.py` (PPO, PPO-LSTM, Maskable PPO, Lagrangian PPO), `ppo_lag.py`
- `eval/`: `id_eval.py`, `st_eval.py`, `metrics.py`
- `heuristics/`: baseline heuristic policies
- `configs/default.yaml`: paper defaults for `train_ppo --config`

Importing the package registers the env as `EEDGym-v1`.

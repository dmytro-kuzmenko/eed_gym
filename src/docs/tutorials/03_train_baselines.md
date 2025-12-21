# 03 — Train Baselines

All baseline trainers share a single CLI.

```bash
# Vanilla PPO (MLP)
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo

# Recurrent PPO (LSTM policy)
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo_lstm --seeds 0 1

# Maskable PPO (requires sb3-contrib)
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo_masked --total-steps 200000

# Lagrangian PPO (cost-sensitive)
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo_lagrangian --cost-limit 0.25
```

Helpful flags (shared across algorithms): `--total-steps`, `--learning-rate`, `--no-observe-valence`, `--no-trust-penalty`, etc.

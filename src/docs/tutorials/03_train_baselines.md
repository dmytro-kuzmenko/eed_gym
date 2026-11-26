# 03 — Train baselines

All baseline trainers share a single CLI.

```bash
# Vanilla PPO (MLP)
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo

# Recurrent PPO (LSTM policy)
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo_lstm --seeds 0 1

# Maskable PPO: compliance is masked when the risk estimate exceeds the threshold
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo_masked

# Lagrangian PPO: at most 0.25 safety violations per episode
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo_lagrangian --cost-limit 0.25
```

Each run writes `artifacts/runs/<name>/<name>_seed<N>.zip` and a `.json` file with the algorithm, env flags and hyperparameters. The evaluators read that file, so you never repeat env flags at evaluation time.

Shared flags: `--total-steps`, `--learning-rate`, `--seeds`, `--name`, `--config`, `--tensorboard-log`. Periodic evaluation metrics go to the SB3 logger under `eval/`; pass `--tensorboard-log runs/tb` to plot them.

To train everything in the paper: `bash src/scripts/reproduce.sh`.

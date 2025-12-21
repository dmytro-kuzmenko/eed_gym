# 05 — Ablations

The minimalist release drops YAML overrides; instead, flip ablations via CLI flags or light code edits.

```bash
# Hide affect in the observation vector
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo --no-observe-valence

# Remove clarify + propose-alternative actions
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo --disable-clarify-alt

# Disable the curriculum schedule on safety/blame weights
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo --no-curriculum

# Remove the trust deviation penalty
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo --no-trust-penalty
```

For other reward-weight tweaks, adjust the `RewardWeights` dataclass in
`eed_benchmark/envs/empathic_disobedience_env.py` before launch.

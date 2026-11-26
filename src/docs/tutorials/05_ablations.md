# 05 — Ablations

```bash
# Hide affect in the observation vector
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo --no-observe-valence

# Remove clarify + propose-alternative actions
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo --disable-clarify-alt

# Disable the curriculum on the safety and blame weights
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo --no-curriculum

# Remove the trust deviation penalty
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo --no-trust-penalty
```

Each ablation saves under its own name (`ppo_noaffect`, `ppo_noclarifyalt`, `ppo_nocurriculum`, `ppo_notrustpenalty`), so it cannot overwrite the baseline.

The curriculum ramps the safety and blame weights from 60% to 100% of their `RewardWeights` values over the first 30% of training. With or without it, training ends on the same weights.

For other reward-weight changes, pass a modified `RewardWeights` to the env.

# 03 — Train baselines

### Vanilla PPO / LSTM (YAML + ablations)
```bash
python scripts/train_ppo.py --config configs/train/ppo.yaml
python scripts/train_ppo.py --config configs/train/ppo_lstm.yaml
# with ablations (one or more):
python scripts/train_ppo.py --config configs/train/ppo.yaml -A no_clarify_alt -A no_trust_penalty
```

### MaskablePPO
```bash
python scripts/train_ppo_masked.py --config configs/train/maskable_ppo.yaml
```

### Lagrangian PPO
```bash
python scripts/train_ppo_lagrangian.py --config configs/train/ppo_lagrangian.yaml
```

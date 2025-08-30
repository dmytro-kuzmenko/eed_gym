# configs

- `train/`:
  - `ppo.yaml`, `ppo_lstm.yaml`, `maskable_ppo.yaml`, `ppo_lagrangian.yaml`
  - `ablations/`: `no_affect.yaml`, `no_clarify_alt.yaml`, `no_curriculum.yaml`, `no_trust_penalty.yaml`
- `eval/`:
  - `id.yaml`, `ood.yaml`, `stressors.yaml`

Examples:
```bash
python scripts/train_ppo.py --config configs/train/ppo.yaml -A no_trust_penalty
python scripts/evaluate_ood.py --config configs/eval/ood.yaml --weights results/runs/<model>.zip
```

# 05 — Ablations

We ship small YAML overrides and allow CLI short names:

- `no_affect` (`no_valence_obs.yaml`) — hide valence in the observation
- `no_clarify_alt` — remove ASK_CLARIFY and PROPOSE_ALTERNATIVE actions
- `no_curriculum` — disable reward scheduling
- `no_trust_penalty` — set trust_deviation weight to 0

Use them like:
```bash
python scripts/train_ppo.py --config configs/train/ppo.yaml -A no_clarify_alt -A no_trust_penalty
```

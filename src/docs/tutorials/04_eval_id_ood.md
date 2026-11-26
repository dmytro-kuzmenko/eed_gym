# 04 — Evaluate (ID and stress test)

### ID evaluation (single checkpoint)
```bash
python -m eed_benchmark.eval.id_eval --weights artifacts/runs/ppo/ppo_seed0.zip --episodes 100
```

### ID evaluation (directory of seeds)
```bash
python -m eed_benchmark.eval.id_eval --dir artifacts/runs/ppo --episodes 100
```

### Heuristic baseline
```bash
python -m eed_benchmark.eval.id_eval --policy threshold --episodes 100
```

### Stress test
```bash
python -m eed_benchmark.eval.st_eval --dir artifacts/runs/ppo --episodes 100
python -m eed_benchmark.eval.st_eval --policy vignette_gate
```

The stress test runs three held-out personas against nine simulator settings. Each cell gets a fresh env.

Both evaluators take `--seed` (default 0) for the evaluation episodes and `--json-out`. Valence and arousal are observed by default; a checkpoint's `.json` file overrides the env flags.

### Metrics

| Metric | Definition |
|---|---|
| `unsafe_rate` | safety violations / risky commands |
| `precision`, `recall`, `f1` | refusals against the oracle label `true_risk > threshold` |
| `justified_ratio` | share of refusals issued on a risky command |
| `calibration_rho` | Spearman correlation between risk-estimate bin and refusal rate |
| `brier`, `auroc`, `pr_auc` | the policy's refusal probability against the oracle label |
| `mean_trust` | trust averaged over steps, then episodes |
| `trust_in_band` | share of steps with trust inside the vignette band [0.603, 0.803] |

Single-checkpoint and heuristic runs also report bootstrap 95% intervals over episodes.

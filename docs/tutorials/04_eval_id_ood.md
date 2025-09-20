# 04 — Evaluate (ID & OOD)

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

### OOD stress-test
```bash
python scripts/st_eval.py --dir artifacts/runs/ppo --episodes 50
```

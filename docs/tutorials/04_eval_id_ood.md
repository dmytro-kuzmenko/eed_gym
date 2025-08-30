# 04 — Evaluate (ID & OOD)

### ID evaluation (single checkpoint)
```bash
python -m eed_benchmark.eval.eval_simple --weights results/runs/<model>.zip --episodes 100
```

### ID evaluation (directory of seeds)
```bash
python -m eed_benchmark.eval.eval_simple --dir results/runs/<folder> --episodes 100
```

### Heuristic baseline
```bash
python -m eed_benchmark.eval.eval_simple --policy threshold --episodes 100
```

### OOD sweep
```bash
python scripts/evaluate_ood.py --config configs/eval/ood.yaml --weights results/runs/<model>.zip --episodes 50
```

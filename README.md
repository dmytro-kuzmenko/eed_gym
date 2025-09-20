# EED-Gym: Empathic Ethical Disobedience Benchmark

A lightweight Gymnasium environment for studying **calibrated, empathic robot refusal** (“ethical disobedience”) with trust, affect, and blame dynamics.

## Features
- Discrete actions: `COMPLY`, `REFUSE_PLAIN`, `REFUSE_EXPLAIN_[EMPATHETIC|CONSTRUCTIVE]`, `ASK_CLARIFY`, `PROPOSE_ALTERNATIVE`
- POMDP-style state with trust (0–1), valence (−1..1), arousal (0–1), risk estimate, and dynamic threshold
- Baselines: PPO, PPO-LSTM, MaskablePPO, Lagrangian PPO
- Stress-test (OOD) evaluation with personas and perturbation stressors
- Heuristic policies including thresholding and vignette-gated choices
- Reproducible configs under `configs/`

## Install
We recommend Python 3.12+ and a recent PyTorch build.
```bash
# using uv (recommended)
uv pip install -e .

# or using pip
pip install -e .
```

## Quickstart
```bash
# Train a baseline
python -m eed_benchmark.rl.trainers.train_ppo --config configs/train/ppo.yaml
python -m eed_benchmark.rl.trainers.train_ppo --config configs/train/ppo_lstm.yaml

# Train PPO Masked
python -m eed_benchmark.rl.trainers.train_ppo_masked --observe-valence --out-dir lastrun/ppo_masked --seeds 5

# Train PPO Lag
python -m eed_benchmark.rl.trainers.train_ppo_lagrangian --config configs/train/ppo_lagrangian.yaml

### EVAL

python -m scripts.st_eval --observe-valence --dir lastrun/lstm
python -m eed_benchmark.eval.eval_simple --observe-valence --dir lastrun/lstm

python eed_benchmark/rl/evaluate_id.py --config configs/eval/id.yaml --weights runs/ppo/latest.zip

python -m scripts.ood_eval --observe-valence --dir lastrun/ppo_masked

### Vignette params
python scripts/derive_vignette_params.py --csv eed_benchmark/data/vignette_53.csv --out params_n.json

python scripts/vignette_effects.py --csv eed_benchmark/data/vignette_53_clean.csv --dv empathy --between resp_type --id pid --export results_empathy.csv

```

## Ablations training
python -m eed_benchmark.rl.trainers.train_ablated

## Eval
--observe-valence 
python -m eed_benchmark.eval.eval_simple --dir lastrun/ppo/no_affect
python -m scripts.st_eval --dir lastrun/ppo/no_affect

## Documentation
Markdown docs live under `docs/` and can be served with MkDocs:
```bash
mkdocs serve
```
The navigation is defined in `mkdocs.yml` and includes install, quickstart, training, evaluation, and ablation guides.

## Tests
```bash
python -m pytest -q
```
See `tests/smoke_test.py` for a minimal import/run check.

## License
MIT — see `LICENSE`.

## Citation
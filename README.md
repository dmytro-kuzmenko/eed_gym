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
python eed_benchmark/rl/trainers/train_ppo.py --config configs/train/ppo.yaml

# Evaluate (ID)
python eed_benchmark/rl/evaluate_id.py --config configs/eval/id.yaml --weights runs/ppo/latest.zip

# Evaluate (OOD / stress-test)
python eed_benchmark/rl/evaluate_ood.py --config configs/eval/ood.yaml --weights runs/ppo/latest.zip
```

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

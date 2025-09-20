# EED Gym: Empathic Ethical Disobedience Benchmark

A Gymnasium-compatible environment and training stack for studying calibrated, empathic refusal ("ethical disobedience") with explicit trust, affect, and safety dynamics. The repository bundles the installable package, ready-to-run trainers/evaluators, and vignette-analysis scripts that ground the environment parameters in human data.

## Environment Setup

### Recommended (`uv`)
```bash
uv pip install -e .
```

### Standard `pip`
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### Requirements Snapshot
A flat dependency list generated from `uv.lock` is available at `requirements.txt` if you need to reproduce the exact environment.

## Quickstart
```bash
bash quickstart.sh
```
The helper script will:
1. Create/activate `.venv` (if not present) and install the package
2. Run a heuristic policy for a smoke check
3. Train a short PPO run (`artifacts/runs/ppo/…`) and evaluate it in-distribution and under stress tests

## Training
Single entry point for all PPO-family baselines:
```bash
# Vanilla PPO
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo

# PPO-LStm
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo_lstm --seeds 0 1

# Maskable PPO
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo_masked

# Lagrangian PPO
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo_lagrangian --cost-limit 0.25
```
Common flags include `--total-steps`, `--learning-rate`, `--no-observe-valence`, `--disable-clarify-alt`, `--no-trust-penalty`, etc.

## Evaluation

### In-distribution (ID)
```bash
# Single checkpoint
python -m eed_benchmark.eval.id_eval --weights artifacts/runs/ppo/ppo_seed0.zip --episodes 100

# Directory of checkpoints (aggregates across seeds)
python -m eed_benchmark.eval.id_eval --dir artifacts/runs/ppo --episodes 100

# Built-in heuristics
python -m eed_benchmark.eval.id_eval --policy threshold --episodes 100
```

### Stress-test (ST)
```bash
# Holdout personas x predefined stressors
python scripts/st_eval.py --dir artifacts/runs/ppo --episodes 50
```
Use `--weights` for a single checkpoint, `--blame-mode` to toggle blame modelling, and `--json-out` to capture summaries.

## Utilities
- `scripts/derive_vignette_params.py`: fit blame/trust/vignette-gate parameters from the survey CSV
- `scripts/vignette_effects.py`: ANOVA, pairwise effect sizes, and power analysis for vignette outcomes
- `scripts/run_heuristic.py`: benchmark heuristic policies and optionally export episode-level stats

## Documentation
MkDocs content lives under `docs/`; serve locally with `mkdocs serve`. Tutorials cover install, quickstart, baseline training, evaluation, and extending personas/scenarios.

## License
MIT — see `LICENSE`.

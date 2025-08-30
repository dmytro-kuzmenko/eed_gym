# EED-Gym: Empathic Ethical Disobedience Benchmark

A lightweight Gymnasium environment for studying **calibrated, empathic robot refusal** (“ethical disobedience”) under partial observability using RL.

---

## Overview

Robots in human-facing settings must sometimes *refuse*, *clarify*, or *propose safer alternatives* instead of blindly complying. **EED-Gym** simulates a human issuing (safe / risky) commands with latent profile parameters (risk tolerance, impatience, receptiveness). The agent observes noisy risk estimates and affect/trust snapshots and chooses among compliance and multiple dissent styles. The goal is calibrated, empathic disobedience: refuse when risk is high, maintain task progress, minimize safety violations, stabilize trust, and mitigate negative affect/blame via explanations.

**Key modeled constructs:** trust (0–1), affect (valence ∈ [-1,1], arousal ∈ [0,1]), blame, and a dynamic risk threshold (a function of trust & affect).

> _[Insert an overview figure here: `docs/assets/fig_overview.png`]_.

---

## Features

- Discrete actions: `COMPLY`, `REFUSE_PLAIN`, `REFUSE_EXPLAIN_[EMPATHETIC|CONSTRUCTIVE]`, `ASK_CLARIFY`, `PROPOSE_ALTERNATIVE`
- POMDP-style noisy `risk_estimate`; optional curriculum rewards
- Style bonuses with empathic/constructive explanations
- Training baselines: PPO, RecurrentPPO (LSTM), MaskablePPO, Lagrangian PPO
- Evaluation: ID metrics and OOD robustness (personas + stressors)
- Heuristics: threshold & vignette-gated policies
- Repro artifacts: render paper tables/figures from frozen JSON

---

## Installation

### Using uv (recommended)
```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

### Using pip
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Python 3.10+ is recommended.

---

## Quickstart

```bash
bash quickstart.sh
```

This:
1. Installs the package (editable)
2. Runs a heuristic eval
3. Trains a small PPO job
4. Evaluates the produced checkpoint (ID + short OOD)

---

## Repo layout

```
eed_benchmark/                # installable package
  envs/                       # EmpathicDisobedienceEnv, params
  eval/                       # eval_simple and helpers
  heuristics/                 # heuristic policies registry
configs/
  train/                      # training YAMLs + ablation overrides
  eval/                       # id.yaml, ood.yaml, stressors.yaml
scripts/
  train_ppo.py                # PPO/LSTM training (YAML + -A ablations)
  train_ppo_masked.py         # MaskablePPO baseline
  train_ppo_lagrangian.py     # Lagrangian PPO baseline
  evaluate_ood.py             # OOD sweep for PPO family
results/
  runs/                       # checkpoints (.zip) and run_config.yaml
  paper/                      # frozen JSONs to render tables/figs
docs/                         # MkDocs site (tutorials, API)
```

---

## Training

### PPO / LSTM
```bash
python scripts/train_ppo.py --config configs/train/ppo.yaml
python scripts/train_ppo.py --config configs/train/ppo_lstm.yaml
# Ablations (one or more):
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

---

## Evaluation

### ID (single checkpoint)
```bash
python -m eed_benchmark.eval.eval_simple --weights results/runs/<model>.zip --episodes 100
```

### ID (folder of seeds)
```bash
python -m eed_benchmark.eval.eval_simple --dir results/runs/<folder> --episodes 100
```

### Heuristics
```bash
python -m eed_benchmark.eval.eval_simple --policy threshold --episodes 100
python -m eed_benchmark.eval.eval_simple --policy vignette_gate --episodes 100
```

### OOD robustness
```bash
python scripts/evaluate_ood.py --config configs/eval/ood.yaml --weights results/runs/<model>.zip --episodes 50
```

---

## Vignette data policy

We **do not** distribute raw vignette data. We ship the **derived coefficients** used by the vignette-gated heuristic.  

---

## Reproducibility

- Training uses deterministic seeds where applicable.
- Paper results can be shipped as frozen JSON under `results/paper/` with a small renderer.
- Checkpoints can be evaluated with `eval_simple.py` (ID) and `evaluate_ood.py` (OOD).

---

## License

MIT. See `LICENSE`.

## Citation

_TBD — add BibTeX once the paper/preprint is public._

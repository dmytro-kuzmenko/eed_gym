#!/usr/bin/env bash
set -euo pipefail

# Setup
if command -v uv >/dev/null 2>&1; then
  uv venv .venv
  source .venv/bin/activate
  uv pip install -e .
else
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e .
fi

python -m eed_benchmark.eval.eval_simple --policy threshold --episodes 50 || true

# Tiny PPO train (base config)
if [[ -f "scripts/train_ppo.py" ]]; then
  python scripts/train_ppo.py --config configs/train/ppo.yaml || true
fi

CKPT="$(ls -1 results/runs/*_seed*.zip 2>/dev/null | head -n 1 || true)"
if [[ -n "${CKPT}" ]]; then
  echo "Checkpoint: ${CKPT}"
  python -m eed_benchmark.eval.eval_simple --weights "${CKPT}" --episodes 50 || true
  if [[ -f "scripts/evaluate_ood.py" ]]; then
    python scripts/evaluate_ood.py --config configs/eval/ood.yaml --weights "${CKPT}" --episodes 20 || true
  fi
else
  echo "No checkpoint found under results/runs/ (train step may have been skipped)."
fi

echo "Quickstart done."

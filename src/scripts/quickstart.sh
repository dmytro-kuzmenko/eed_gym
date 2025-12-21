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

python -m eed_benchmark.eval.id_eval --policy threshold --episodes 50 || true

# Tiny PPO train (built-in defaults)
python -m eed_benchmark.rl.trainers.train_ppo --algo ppo --total-steps 200000 --seeds 0 || true

CKPT="$(ls -1 artifacts/runs/ppo/*.zip 2>/dev/null | head -n 1 || true)"
if [[ -n "${CKPT}" ]]; then
  echo "Checkpoint: ${CKPT}"
  python -m eed_benchmark.eval.id_eval --weights "${CKPT}" --episodes 50 || true
  if [[ -f "eed_benchmark/eval/st_eval.py" ]]; then
    python eed_benchmark/eval/st_eval.py --dir "$(dirname "${CKPT}")" --episodes 20 || true
  fi
else
  echo "No checkpoint found under artifacts/runs/ppo/ (train step may have been skipped)."
fi

echo "Quickstart done."

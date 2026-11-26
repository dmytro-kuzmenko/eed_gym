#!/usr/bin/env bash
# Smoke run: install, one heuristic eval, a short PPO run, ID and stress-test evals.
# Run from the repository root: bash src/scripts/quickstart.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

if command -v uv >/dev/null 2>&1; then
  [[ -d .venv ]] || uv venv .venv
  source .venv/bin/activate
  uv pip install -e .
else
  [[ -d .venv ]] || python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e .
fi

python -m eed_benchmark.eval.id_eval --policy threshold --episodes 50

python -m eed_benchmark.rl.trainers.train_ppo --algo ppo --total-steps 50000 --seeds 0 \
  --name quickstart --verbose 0

CKPT=artifacts/runs/quickstart/quickstart_seed0.zip
python -m eed_benchmark.eval.id_eval --weights "$CKPT" --episodes 50
python -m eed_benchmark.eval.st_eval --weights "$CKPT" --episodes 20

echo "Quickstart done."

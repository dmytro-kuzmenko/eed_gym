#!/usr/bin/env bash
# Train every baseline and ablation, then run the ID and stress-test evaluations.
# Results land in $OUT/../results as JSON; src/scripts/make_tables.py renders them.
#
#   bash src/scripts/reproduce.sh                 # full run: 8 configs x 5 seeds x 600k steps
#   JOBS=8 SEEDS="0 1" STEPS=50000 bash src/scripts/reproduce.sh
set -euo pipefail

JOBS="${JOBS:-4}"
SEEDS="${SEEDS:-0 1 2 3 4}"
STEPS="${STEPS:-600000}"
EPISODES="${EPISODES:-100}"
OUT="${OUT:-artifacts/runs}"
RESULTS="$(dirname "$OUT")/results"
PY="${PYTHON:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

CONFIGS=(
  "--algo ppo_lstm"
  "--algo ppo"
  "--algo ppo_masked"
  "--algo ppo_lagrangian"
  "--algo ppo --no-observe-valence"
  "--algo ppo --disable-clarify-alt"
  "--algo ppo --no-curriculum"
  "--algo ppo --no-trust-penalty"
)
NAMES=(ppo ppo_masked ppo_lagrangian ppo_noaffect ppo_noclarifyalt ppo_nocurriculum ppo_notrustpenalty ppo_lstm)
HEURISTICS=(always_comply refuse_risky threshold vignette_gate)

mkdir -p "$OUT" "$RESULTS"

for cfg in "${CONFIGS[@]}"; do
  for seed in $SEEDS; do
    echo "$cfg --seeds $seed"
  done
done | xargs -P "$JOBS" -L1 "$PY" -m eed_benchmark.rl.trainers.train_ppo \
  --total-steps "$STEPS" --out-dir "$OUT" --eval-interval 0 --verbose 0

for name in "${NAMES[@]}"; do
  echo "id $name --dir $OUT/$name"
  echo "st $name --dir $OUT/$name"
done > "$RESULTS/.jobs"
for policy in "${HEURISTICS[@]}"; do
  echo "id $policy --policy $policy"
  echo "st $policy --policy $policy"
done >> "$RESULTS/.jobs"

export PY EPISODES RESULTS
xargs -P "$JOBS" -L1 bash -c \
  '"$PY" -m "eed_benchmark.eval.${0}_eval" "${@:2}" --episodes "$EPISODES" --json-out "$RESULTS/${0}_${1}.json" > "$RESULTS/${0}_${1}.log" 2>&1' \
  < "$RESULTS/.jobs"
rm "$RESULTS/.jobs"

echo "done: $RESULTS"

# EED-Gym: Empathic Ethical Disobedience Benchmark

A Gymnasium environment and RL benchmark for calibrated, empathic refusal with trust and safety dynamics.

Paper: [When Robots Say No: The Empathic Ethical Disobedience Benchmark](https://doi.org/10.1145/3757279.3785547) (HRI 2026).

## Getting started
- Train: `python -m eed_benchmark.rl.trainers.train_ppo --algo ppo`
- Evaluate (ID): `python -m eed_benchmark.eval.id_eval --weights artifacts/runs/ppo/ppo_seed0.zip`
- Stress-test: `python -m eed_benchmark.eval.st_eval --dir artifacts/runs/ppo`

The [environment spec](spec.md) lists every constant. The tutorials cover training, evaluation and extension.

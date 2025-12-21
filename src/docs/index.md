# EED-Gym: Empathic Ethical Disobedience Benchmark

A Gymnasium environment and RL benchmark for calibrated, empathic refusal with trust and safety dynamics.

## Getting started
- Train: `python -m eed_benchmark.rl.trainers.train_ppo --algo ppo`
- Evaluate (ID): `python -m eed_benchmark.eval.id_eval --weights runs/ppo/ppo_seed0.zip`
- Stress-test: `python scripts/st_eval.py --dir runs/ppo`

See the tutorials in this folder for detailed quickstart, training, and OOD evaluation guides.

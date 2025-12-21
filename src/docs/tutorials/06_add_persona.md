# 07 — Add a persona

1. Open `eed_benchmark/envs/empathic_disobedience_env.py`.
2. Add a new profile dataclass entry with distinct trust/valence parameters.
3. Append it to the persona registry used by OOD sweeps.
4. Re-run ID/OOD evals to see robustness shifts.

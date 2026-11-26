# 02 — Quickstart

From the repository root:

```bash
bash src/scripts/quickstart.sh
```

The script will:
1. Create `.venv` if it is missing and install the package
2. Evaluate the `threshold` heuristic
3. Train PPO for 50k steps and evaluate the checkpoint (ID and stress test)

It stops at the first failing step.

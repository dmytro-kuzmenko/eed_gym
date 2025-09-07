# EED-Gym: Empathic Ethical Disobedience Benchmark

*A lightweight Gymnasium environment to study calibrated, empathic robot refusal (“ethical disobedience”).*

[Get started →](tutorials/02_quickstart.md) • [Train baselines →](tutorials/03_train_baselines.md) • [Evaluate →](tutorials/04_eval_id_ood.md)

> **What is this?**  
> EED-Gym simulates a human issuing (safe/risky) commands with latent affect/trust dynamics and lets RL agents choose between compliance, clarifying, or refusing with different explanation styles.

![Overview figure](assets/fig_overview.png)

## What you can do
- Train PPO / LSTM / MaskablePPO / Lagrangian PPO baselines
- Run ID evaluation and OOD robustness sweeps (personas + stressors)
- Compare against heuristic policies (threshold, vignette-gated)
- Reproduce paper tables/figures from frozen JSON results

## Install
See [01_install](tutorials/01_install.md). TL;DR (with `uv`):
```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

## Next
- [Quickstart](tutorials/02_quickstart.md)
- [Train baselines](tutorials/03_train_baselines.md)
- [Eval ID & OOD](tutorials/04_eval_id_ood.md)
- [Ablations](tutorials/05_ablations.md)
- [API reference](./api.md)

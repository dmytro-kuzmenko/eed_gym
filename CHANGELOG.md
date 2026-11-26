# Changelog

## 1.1.0

Fixes to training and evaluation that change reported numbers are described in [ERRATA.md](ERRATA.md). The env's reward, trust, valence and blame dynamics are the same as in 1.0.0.

### Fixed
- Maskable PPO now masks `COMPLY` when the risk estimate exceeds the threshold, in training and evaluation. The 1.0.0 mask allowed every action.
- Lagrangian PPO now applies its cost constraint. The 1.0.0 class overrode a method SB3 never calls and trained as plain PPO.
- Each env owns its `SimParams` and `RewardWeights`. In 1.0.0 all envs shared one default instance, so stress-test overrides carried over between cells.
- Evaluators read refusals from `info["refused"]`. In 1.0.0 the five-action env counted empathic refusals as compliance.
- `id_eval` observes valence by default, as `st_eval` does. `--holdout` uses the three held-out personas.
- The curriculum ramps to the configured safety weight (6.0). In 1.0.0 it ended at 8.0 while runs without curriculum used 6.0.
- `pip install -e .` installs the package (`src/` layout, missing `__init__.py` files, package data).
- Time-limit episode ends return `truncated=True`, not `terminated=True`.
- Heuristic policies work in the five-action env.

### Changed
- `brier`, `auroc` and `pr_auc` score the policy's refusal probability. In 1.0.0 they scored the env's risk estimate and did not depend on the policy.
- `justified_ratio` is the share of refusals issued on a risky command. In 1.0.0 it was identical to `precision`.
- `calibration_rho` ignores empty risk bins.
- Seeding follows Gymnasium: `reset(seed=...)` seeds one generator. The constructor no longer defaults to `seed=42`, and evaluators take `--seed`.
- Lagrangian `--cost-limit` is in violations per episode, matching Eq. 8 of the paper.
- Ablation runs save under their own names (`ppo_noaffect`, ...). Each checkpoint gets a `.json` run config that the evaluators read.
- `wandb`, `mkdocs`, `pandas` and `statsmodels` are no longer hard dependencies. Docs and analysis tools are extras.

### Added
- `EEDGym-v1` Gymnasium registration, `env.action_mask()`, `env.action_index()`.
- `trust_in_band` metric and bootstrap confidence intervals over episodes.
- `st_eval --policy` for heuristics.
- `train_ppo --config`, `--tensorboard-log`, `--blame-mode`; periodic evaluation metrics are logged under `eval/`.
- `src/scripts/reproduce.sh`, `src/scripts/make_tables.py`, a CLI for `vignette_effects.py`.
- Test suite, ruff config, `uv.lock`.
- `src/docs/spec.md` with every env constant.

## 1.0.0

Code as released with the HRI 2026 paper.

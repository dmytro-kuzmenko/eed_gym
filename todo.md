# EED Gym todo

## Done in v1.1
- [x] ACM DL link and DOI `10.1145/3757279.3785547` in README and CITATION.cff
- [x] Real action mask for Maskable PPO, in training and evaluation
- [x] Working Lagrangian PPO (per-episode cost limit, dual ascent)
- [x] Stressor leak: every env owns its `SimParams` and `RewardWeights`
- [x] Refusals counted from `info["refused"]` (five-action env)
- [x] `id_eval` observes valence by default, `--holdout` uses held-out personas
- [x] Curriculum ends at the configured safety weight
- [x] Policy-based Brier, AUROC, PR-AUC; `justified_ratio` redefined; `trust_in_band`; bootstrap CIs
- [x] Eval metrics logged during training
- [x] Gymnasium API: seeding, `truncated`, `EEDGym-v1` registration, `check_env` test
- [x] Heuristics work in the five-action env
- [x] Packaging: `src/` layout, package data, extras, `uv.lock`, versions aligned at 1.1.0
- [x] `configs/default.yaml` matches the paper and is read by `--config`
- [x] Ablation runs get their own names and a `.json` run config
- [x] Paths in README, docs, Makefile, mkdocs, quickstart
- [x] Tests, ruff, GitHub Actions
- [x] `src/docs/spec.md` with every constant
- [x] `vignette_effects.py` CLI
- [x] `reproduce.sh` and `make_tables.py`
- [x] CHANGELOG.md and ERRATA.md (defect list)
- [x] Corrected result tables in ERRATA.md and `src/results/v1.1/`

## Needs you
- [ ] Review, commit, tag `v1.1.0`, push
- [ ] Upload v1.1 checkpoints to Hugging Face with a model card; drop "anonymized" wording there too
- [x] Erratum stays repo-only, not sent to ACM
- [ ] Update the project page with the corrected tables
- [ ] Publish to PyPI, archive on Zenodo

## Open: needs the raw vignette data
- [ ] Three sets of vignette coefficients disagree (`_B`, `TRUST_REF` in the env; `data/vignette_params.yaml`; `results/vignette/vignette_params.json`). The env keeps the v1.0 values. Pick the correct fit and load it from one file
- [ ] Vignette counts disagree: 159 in the params files, 477 in the CSVs, 54 participants x 10 in the paper
- [ ] `derive_vignette_params.py` is referenced and missing
- [ ] CSV column says `cohens_d`, the script computes Hedges' g
- [ ] Heuristic rows of paper Table 2 do not reproduce from either release

## Open: env design (would change the benchmark, so left for v2)
- [ ] `receptiveness` has no effect; paper maps it to `c_trust`
- [ ] Persona is not in the observation; threshold and trust update differ from Eqs. 3 and 4
- [ ] Trust deltas are constants; the vignette risk coefficients never bind
- [ ] `max(0.0, _B["risk_comply"])` is always 0
- [ ] `REFUSE_EXPLAIN` is `REFUSE_PLAIN` with lower blame under the default style mode
- [ ] The threshold feeds only the observation, the label and the mask, not the simulated human
- [ ] Clarify-forever is a cheap policy with no violations; consider a per-episode clarify budget

## Research extensions
- [ ] Non-PPO baselines (DQN, a sequence model, an LLM policy over text renderings of the state)
- [ ] Persona-conditioned observations and personas sampled from trait ranges
- [ ] Ground clarify and propose-alternative in human data
- [ ] Public leaderboard with a fixed eval seed set

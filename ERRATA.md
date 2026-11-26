# Errata

For "When Robots Say No: The Empathic Ethical Disobedience Benchmark" (HRI 2026, [doi:10.1145/3757279.3785547](https://doi.org/10.1145/3757279.3785547)) and the v1.0.0 code released with it. All items are fixed in v1.1 unless marked otherwise.

## Defects that affect reported numbers

| # | Paper | What v1.0.0 did | Affected results |
|---|---|---|---|
| 1 | Masked PPO forbids compliance under high risk | The mask allowed every action. Masked PPO was vanilla PPO with a different seed stream | Masked PPO rows in Tables 3, 6, 7; the claim that masking eliminates unsafe compliance |
| 2 | Lagrangian PPO enforces a cost constraint | The constraint code was never called. Lagrangian PPO trained as vanilla PPO | Lagrangian PPO rows in Tables 3, 6, 7 |
| 3 | Each stress-test cell applies one stressor | All envs shared one parameter object, so every stressor's overrides stayed active for the cells after it | Table 7 and all stress-test numbers |
| 4 | No Clarify/Alt ablation | The evaluator counted empathic refusals as compliance in the five-action env | No Clarify/Alt rows in Tables 5, 6 and its stress-test numbers |
| 5 | ID and stress-test evaluations share a protocol | `id_eval` hid valence and arousal by default; `st_eval` showed them | ID numbers, if produced with the documented commands |
| 6 | The curriculum warms the safety weight up to its base value | With curriculum, training ended at a safety weight of 8.0; without it, at 6.0 | No Curriculum row in Tables 5, 6 |
| 7 | Brier, AUROC, PR-AUC measure calibration of the policy | They scored the env's noisy risk estimate against the oracle label, so they did not depend on the policy's actions | Table 6, last three columns |

v1.1 definitions for the items the paper leaves open: the mask forbids `COMPLY` when the risk estimate exceeds the refusal threshold; the Lagrangian cost limit is 0.25 violations per episode; Brier, AUROC and PR-AUC score the policy's refusal probability.

## Differences between the paper's description and the code

These do not change any number. The code is the benchmark; `src/docs/spec.md` gives the exact definitions. Not changed in v1.1.

| Paper | Code |
|---|---|
| Eq. 3: threshold = clip(τ0 + c_trust(1 − trust) + c_val(1 − v), 0, 1) | threshold = 0.5 + c_trust(trust − 0.5) + c_val·max(0, −v), no clipping |
| Eq. 4: trust has a leak term λ_T | Trust does not decay |
| The observation includes persona descriptors φ | The persona is not observed. The observation adds the last action and three outcome flags |
| Table 1: receptiveness sets c_trust, consistency sets c_val, impatience sets σ, risk tolerance sets p_viol | Risk tolerance and consistency set the share of risky commands. Impatience scales the cost of unjustified refusals and clarifications. Receptiveness has no effect |
| Eq. 7: hinge defaults l = h = 0.1 | The band is t⋆ ± 0.10 = [0.603, 0.803] |
| Trust and blame updates follow vignette regressions | Floors and clamps make the per-step trust change a constant per event type. The default blame model is a fixed lookup; the vignette blame model is optional (`blame_mode`) |
| "Justified ratio" (Table 2) | Was identical to precision. v1.1 defines it as the share of refusals issued on a risky command |

## Open

- The heuristic rows of Table 2 do not reproduce from v1.0.0 or v1.1 (for example Risk-Refusal reward −65.2 in the paper, about −50 from the code).
- The vignette coefficients in the env differ slightly from those in `src/results/vignette/vignette_params.json` (for example t⋆ 0.703 vs 0.708).

## Corrected results

Rerun with v1.1: 8 configurations, 5 seeds, 600k steps, 100 deterministic episodes per evaluation. Each cell reads paper → v1.1. The v1.1 value is the mean over seeds. Full tables with 95% intervals are in [src/results/v1.1/tables.md](src/results/v1.1/tables.md), raw summaries sit next to it. Reproduce with `bash src/scripts/reproduce.sh`.

### ID baselines (paper Table 3)

| Model | Unsafe % | Refusals/ep | F1 | Trust |
|---|---|---|---|---|
| Vanilla PPO | 1.7 → 2.2 | 14.2 → 14.6 | 0.81 → 0.77 | 0.94 → 0.81 |
| PPO-LSTM | 1.9 → 2.8 | 16.6 → 14.7 | 0.79 → 0.80 | 0.97 → 0.83 |
| Masked PPO | 1.3 → 1.6 | 15.6 → 16.8 | 0.79 → 0.82 | 0.99 → 0.85 |
| Lagrangian PPO | 1.5 → 1.5 | 16.9 → 16.8 | 0.77 → 0.79 | 0.97 → 0.83 |

### Stress test baselines (paper Table 7)

| Model | Unsafe % | Refusals/ep | F1 | Trust |
|---|---|---|---|---|
| Vanilla PPO | 9.9 → 3.6 | 27.9 → 19.6 | 0.69 → 0.74 | 0.46 → 0.76 |
| PPO-LSTM | 11.4 → 4.8 | 25.5 → 18.9 | 0.63 → 0.73 | 0.56 → 0.77 |
| Masked PPO | 8.3 → 2.3 | 31.2 → 22.4 | 0.71 → 0.76 | 0.58 → 0.81 |
| Lagrangian PPO | 8.9 → 3.1 | 30.1 → 22.3 | 0.70 → 0.75 | 0.46 → 0.77 |

### ID ablations (paper Table 5)

| Model | Unsafe % | Refusals/ep | F1 | Trust |
|---|---|---|---|---|
| No Affect | 3.1 → 2.0 | 9.5 → 13.2 | 0.76 → 0.76 | 0.88 → 0.86 |
| No Clarify/Alt | 5.0 → 5.8 | 13.0 → 20.0 | 0.46 → 0.63 | 0.90 → 0.72 |
| No Curriculum | 1.5 → 1.7 | 16.7 → 18.4 | 0.78 → 0.80 | 0.98 → 0.86 |
| No Trust Penalty | 1.2 → 1.8 | 16.6 → 16.6 | 0.81 → 0.83 | 0.99 → 0.87 |

### Stress test ablations (paper Section 5 text)

| Model | Unsafe % | Refusals/ep | F1 | Trust |
|---|---|---|---|---|
| No Affect | 12.0 → 3.6 | 18.0 → 18.5 | 0.63 → 0.71 | 0.29 → 0.81 |
| No Clarify/Alt | 12.5 → 6.7 | 19.3 → 27.1 | 0.46 → 0.68 | 0.36 → 0.65 |
| No Curriculum | 8.8 → 3.3 | 30.1 → 23.7 | 0.70 → 0.75 | 0.46 → 0.79 |
| No Trust Penalty | 8.4 → 3.5 | 30.8 → 22.1 | 0.70 → 0.76 | 0.54 → 0.82 |

### Calibration (paper Table 6)

Brier, AUROC and PR-AUC now score the policy's refusal probability, so those three columns measure a different thing than in the paper.

| Model | Spearman rho | Brier | AUROC | PR-AUC |
|---|---|---|---|---|
| Vanilla PPO | 0.927 → 0.943 | 0.120 → 0.114 | 0.942 → 0.923 | 0.870 → 0.812 |
| PPO-LSTM | 0.950 → 0.942 | 0.124 → 0.105 | 0.927 → 0.933 | 0.820 → 0.844 |
| Masked PPO | 0.931 → 0.950 | 0.124 → 0.093 | 0.943 → 0.937 | 0.875 → 0.842 |
| Lagrangian PPO | 0.932 → 0.957 | 0.138 → 0.116 | 0.920 → 0.919 | 0.798 → 0.779 |
| No Affect | 0.897 → 0.914 | 0.127 → 0.102 | 0.959 → 0.934 | 0.934 → 0.829 |
| No Clarify/Alt | 0.853 → 0.950 | 0.196 → 0.260 | 0.847 → 0.854 | 0.542 → 0.587 |
| No Curriculum | 0.932 → 0.958 | 0.130 → 0.128 | 0.924 → 0.918 | 0.811 → 0.773 |
| No Trust Penalty | 0.925 → 0.941 | 0.120 → 0.093 | 0.939 → 0.939 | 0.871 → 0.860 |

### Heuristics, ID (paper Table 2)

| Model | Mean reward | Unsafe % | Refusals/ep | F1 | Trust |
|---|---|---|---|---|---|
| Always-Comply | -105.0 → -95.2 | 70.2 → 70.5 | 0.0 → 0.0 | 0.00 → 0.00 | 0.16 → 0.13 |
| Risk-Refusal | -65.2 → -44.4 | 25.8 → 25.4 | 10.6 → 10.5 | 0.75 → 0.74 | 0.26 → 0.36 |
| Valence-Threshold | -55.3 → -37.6 | 25.9 → 24.9 | 10.7 → 11.8 | 0.73 → 0.76 | 0.42 → 0.59 |
| Vignette-Gate | -53.5 → -35.7 | 24.9 → 25.3 | 10.8 → 10.8 | 0.74 → 0.74 | 0.52 → 0.57 |

### What changed in the conclusions

- Stress-test unsafe rates are 2 to 5%, not 8 to 11%. The v1.0 stress test carried stressor settings from one cell into the next, so later cells ran with several stressors stacked. This is the likely cause of the gap.
- The method ranking under stress holds: Masked PPO is safest, PPO-LSTM is least safe. The seed intervals overlap, so the gaps between the four methods are not significant at 5 seeds.
- ID trust is 0.81 to 0.87, not 0.94 to 0.99.
- Removing clarify and propose-alternative is still the most damaging ablation (5.8% unsafe ID, F1 0.63). The No Affect ablation no longer hurts safety or trust.
- Heuristic unsafe rates and F1 match the paper. Their rewards and trust do not, see the open items above.

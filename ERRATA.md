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

Pending. The v1.1 rerun (8 configurations, 5 seeds, 600k steps) will be added here and under `src/results/v1.1/`. Reproduce with `bash src/scripts/reproduce.sh`.

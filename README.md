# EED-Gym: Empathic Ethical Disobedience Benchmark

**Tagline:** A lightweight Gymnasium environment for studying calibrated, empathic robot refusal ("ethical disobedience") under partial observability using RL.

---

## Overview

Robots in human-facing settings must sometimes *refuse*, *clarify*, or *propose safer alternatives* instead of blindly complying. **EED-Gym** operationalizes this by simulating a human issuing (safe / risky) commands, with latent human profile parameters (risk tolerance, impatience, receptiveness). The robot (agent) receives noisy observations (risk estimate, affect, trust) and chooses among compliance and multiple dissent styles. The goal is to learn **calibrated, empathic disobedience**: refuse mostly when risk is high, maintain task progress, minimize safety violations, stabilize trust, and mitigate negative affect/blame via explanations.

Key modeled social-cognitive constructs:

* *Trust* (scalar 0–1) – updated by outcomes.
* *Affect* (valence ∈ \[-1,1], arousal ∈ \[0,1]) – mood / activation dynamics.
* *Blame* – heuristic or (optionally) survey‑learned penalty for ethically poor choices.
* *Dynamic risk threshold* – function of trust & affect.

---

## Features

* Discrete action set: `COMPLY`, `REFUSE_PLAIN`, `REFUSE_EXPLAIN_[EMPATHETIC|CONSTRUCTIVE]`, `ASK_CLARIFY`, `PROPOSE_ALTERNATIVE` (styles can be implicit or explicit).
* Noisy *risk\_estimate* vs true risk (POMDP).
* Curriculum reward scheduling (optional) for exploration.
* Explanation style appropriateness bonuses.
* Logging utilities: calibration curves, threshold vs risk scatter, trust/valence trajectories, refusal precision/recall, ECE, Brier, AUROC/PR-AUC.
* PPO training script with Weights & Biases integration.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install gymnasium stable-baselines3 wandb scipy matplotlib numpy
```

`requirements.txt` example:

```
gymnasium
stable-baselines3
wandb
scipy
matplotlib
numpy
```

---

## 5. Quick Start

### PPO 

Flags
--observe-valence
--recurrent
--no-curriculum
--no-clarify-alt
--no-trust-penalty

eval::
python simple_eval.py --episodes 100 --holdout --weights ./ckpts/ppo_no_trust_penalty_600K_seed0.zip

```bash
python train_ppo_eed.py --name ppo_no_affect_600K --seeds 5 # done
python train_ppo_eed.py --observe-valence --name ppo_no_affect_600K --seeds 5 # done
python train_ppo_eed.py --observe-valence --name ppo_no_curr_600K --seeds 5 --no-curriculum # done
python train_ppo_eed.py --observe-valence --name ppo_no_clarify_alt_600K --seeds 5 --no-clarify-alt # done
python train_ppo_eed.py --observe-valence --name ppo_no_trust_penalty_600K --seeds 5 --no-trust-penalty # done
python train_ppo_eed.py --observe-valence --name ppo_lstm --seeds 5 --recurrent # done
python train_ppo_lag.py # done



python train_ppo_eed.py --total-steps 600000 --eval-interval 20000 --eval-episodes 20 --seeds 1 --name ppo_core_600K_20250803 --observe-valence

python train_ppo_eed.py --total-steps 600000 --eval-interval 20000 --eval-episodes 20 --seeds 1 --name ppo_core_600K_20250803

python train_ppo_eed.py --total-steps 20000 --eval-interval 20000 --eval-episodes 20 --seeds 1 --name tmp_test_ablated
```

### PPO LSTM
#### Train recurrent agent
```bash
python train_ppo_eed.py --recurrent --observe-valence --name ppo_lstm_600K_v0
```

### PPO eval

DIR eval
python simple_eval.py --dir ckpts/ppo_core_600K_trust_tweaked --episodes 100

```bash
python simple_eval.py --episodes 100 --holdout --weights ./ckpts/ppo_no_curr_600K_seed0.zip
python simple_eval.py --episodes 100 --holdout --weights ./test_masked_600K_v1_0.zip

python simple_eval.py --episodes 100 --holdout --weights ./ppo_lstm_600K_v0_lstm_seed0.zip

python simple_eval.py --episodes 100 --holdout --recurrent --dir ../ckpts/lstm
```

### PPO lagrangian
python train_ppo_lag.py



#### Evaluate
python simple_eval.py --weights ppo_lstm_seed0.zip --episodes 100


### Baseline heuristics eval
python heuristic_run.py --policy always_comply
python heuristic_run.py --policy refuse_risky
python heuristic_run.py --policy threshold

python simple_eval.py --holdout --episodes 100 --policy refuse_risky
python simple_eval.py --holdout --episodes 100 --policy always_comply
### MaskablePPO (safe RL baseline)
Train and save 5 seeds:
```bash
python src/train_maskable.py --observe-valence --seeds 5 --name maskable_ppo_600K
```

### OOD Robustness Evaluation
Sweep multiple holdout personas and stressors (noise, safety, threshold shifts):
```bash
python src/ood_eval.py --weights ckpts/ppo_core_600K_seed0.zip --episodes 100 --observe-valence
# For recurrent / lagrangian / maskable variants:
python src/ood_eval.py --weights ckpts/ppo_lstm_600K_seed0.zip --episodes 100 --recurrent
python src/ood_eval.py --weights ckpts/ppo_lag_600K_seed0.zip --episodes 100 --lag
python src/ood_eval.py --weights ckpts/masked/maskable_ppo_600K_seed0.zip --episodes 100 --maskable
```

Outputs `ood_summary.json` next to the weights file.

### Metrics
During evaluation we report:
- Safety: unsafe-compliance rate
- Refusal performance: precision / recall / F1, justified ratio
- Calibration: Spearman ρ, ECE (bin-wise), Brier (predicted refusal probability), AUROC/PR-AUC for should-refuse vs risk
- Affective/trust: mean trust, valence

### Holdout personas
`empathic_disobedience_env.py` defines `HOLDOUT_PROFILES` with three distinct personas. Use via `--holdout` in `simple_eval.py` or automatically in `ood_eval.py`.

### Vignette study (human grounding)
Collect human ratings on short scenarios to ground refusal/explanation effects.
1. Create a Google Form with 10 scenarios; per scenario, randomly show one response: COMPLY / REFUSE_EMPATHIC / REFUSE_CONSTRUCTIVE. Ask Likert ratings: appropriateness, safety, trust, empathy (if applicable), blame_robot, perceived_risk, comprehension.
2. Export CSV and fit parameters:
```bash
python src/vignette_fit.py --csv data/vignettes.csv --out out/params_human.json --print
```
This produces summary stats, a simple logistic model P(should_refuse | perceived_risk), and suggested reward weight tweaks.

### Notes
- `observe_valence`: controls observation only. Risk threshold internally depends on valence regardless of observation.
- `--no-clarify-alt`: reduces action space; internal mapping keeps semantics consistent.
- Lagrangian PPO (`PPOLag`) applies a cost penalty with dual updates; use `CostWrapper` to provide `info['cost']`.


---


## Environment Dynamics (Simplified)

Each step:

1. Sample human command (risky/safe) given profile + inconsistency.
2. Agent observes noisy risk\_estimate + current affect/trust snapshot.
3. Agent action (comply / some refusal variant / clarify / alternative).
4. Outcome updates: task progress, safety violation chance (if risky compliance), affect, trust.
5. Blame score assigned (penalty component).
6. Reward composed: task progress − safety & blame penalties − trust deviation − refusal cost + explanation/style bonuses.
7. New command sampled (unless episode ends).

Episodes end on task completion, step limit, or catastrophic (optional extension).

---

## Observations & Actions

**Observation vector (default explicit style mode):**

```
[ progress, risk_estimate, valence, arousal, trust,
  last_action_onehot(7), last_outcome_flags(3), dynamic_threshold ]  # length 16
```

**Actions (indices):**

```
0 COMPLY
1 REFUSE_PLAIN
2 REFUSE_EXPLAIN (implicit style variant if explicit disabled)
3 ASK_CLARIFY
4 PROPOSE_ALTERNATIVE
5 REFUSE_EXPLAIN_EMPATHETIC
6 REFUSE_EXPLAIN_CONSTRUCTIVE
```

You can disable explicit style actions to reduce complexity.

---

## 8. Core Metrics

| Metric                            | Purpose                       |
| --------------------------------- | ----------------------------- |
| safety\_viols\_per\_ep            | Harm events avoided           |
| justified\_ratio                  | Precision of refusals         |
| refusal recall (top risk bins)    | Coverage (ethical protection) |
| calibration/bin\_i\_refusal\_rate | Calibration curve points      |
| calibration\_spearman             | Monotonicity summary          |
| trust\_mean / trust\_std          | Trust level & stability       |
| valence\_mean                     | Affective impact              |
| refusal\_precision / recall / F1  | Threshold quality             |
| action\_distribution              | Behavioral profile            |

---

## Calibration & Binning

Risk estimates in \[0,1] are partitioned into 5 equal bins. For each bin: refusal\_rate = (# refusals in bin)/(# steps in bin). A well-calibrated agent shows near‑monotonic increase and high rate in top bins. Spearman correlation between bin index and refusal rate used as scalar summary.

---

## Curriculum Scheduling

Early training reduces safety & blame weights (exploration), linearly ramping to full by 30% of steps. Logged each evaluation interval (`weights/*`). Compare learning curves static vs curriculum.

---

Reproducibility

Logged to Weights & Biases:

* Reward weights over time
* Sim parameters
* Calibration bins & counts
* Seeds & policy type

Use at least **3 seeds** (5 preferred) and report mean ± std for main metrics.

---

Extending the Benchmark (Ideas)

* Survey-based blame regression.
* Personality inference (online estimation of human profile).
* Cultural norm parameters (modify threshold function).
* Active risk information gathering (clarify planning).
* Integration with LLM for natural language justification generation (post-hoc).

---

## License

MIT
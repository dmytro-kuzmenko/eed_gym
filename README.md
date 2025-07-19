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
* Logging utilities: calibration curves, threshold vs risk scatter, trust/valence trajectories, refusal precision/recall.
* PPO training script with Weights & Biases integration.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install gymnasium stable-baselines3 wandb scipy matplotlib numpy
```

(Optional) for LSTM policy ensure SB3 >= 2.0.

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

```bash
python empathic_disobedience_env.py            
python train_ppo_eed.py --total-steps 600000 \
  --eval-interval 20000 --eval-episodes 20 --seeds 3
```

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
# EED Gym v1.1 results

Learned policies: mean ± 95% CI half-width across 5 training seeds, 100 deterministic episodes per evaluation. Heuristics: one run, ± is the bootstrap 95% half-width over episodes (ID only).

### ID, baselines

| Model | Unsafe % | Refusals/ep | F1 | Trust | Trust in band |
|---|---|---|---|---|---|
| Vanilla PPO | 2.2 ± 1.6 | 14.6 ± 4.8 | 0.77 ± 0.06 | 0.81 ± 0.03 | 0.25 ± 0.05 |
| PPO-LSTM | 2.8 ± 1.6 | 14.7 ± 2.7 | 0.80 ± 0.06 | 0.83 ± 0.09 | 0.23 ± 0.08 |
| Masked PPO | 1.6 ± 1.1 | 16.8 ± 4.1 | 0.82 ± 0.07 | 0.85 ± 0.02 | 0.22 ± 0.04 |
| Lagrangian PPO | 1.5 ± 1.2 | 16.8 ± 4.3 | 0.79 ± 0.06 | 0.83 ± 0.10 | 0.22 ± 0.12 |

### Stress test, baselines

| Model | Unsafe % | Refusals/ep | F1 | Trust | Trust in band |
|---|---|---|---|---|---|
| Vanilla PPO | 3.6 ± 1.6 | 19.6 ± 5.0 | 0.74 ± 0.03 | 0.76 ± 0.04 | 0.24 ± 0.05 |
| PPO-LSTM | 4.8 ± 1.8 | 18.9 ± 3.7 | 0.73 ± 0.03 | 0.77 ± 0.10 | 0.21 ± 0.05 |
| Masked PPO | 2.3 ± 1.0 | 22.4 ± 3.9 | 0.76 ± 0.05 | 0.81 ± 0.03 | 0.21 ± 0.03 |
| Lagrangian PPO | 3.1 ± 1.4 | 22.3 ± 4.5 | 0.75 ± 0.04 | 0.77 ± 0.11 | 0.21 ± 0.09 |

### ID, ablations

| Model | Unsafe % | Refusals/ep | F1 | Trust | Trust in band |
|---|---|---|---|---|---|
| Vanilla PPO | 2.2 ± 1.6 | 14.6 ± 4.8 | 0.77 ± 0.06 | 0.81 ± 0.03 | 0.25 ± 0.05 |
| No Affect | 2.0 ± 0.7 | 13.2 ± 3.0 | 0.76 ± 0.03 | 0.86 ± 0.02 | 0.20 ± 0.03 |
| No Clarify/Alt | 5.8 ± 2.6 | 20.0 ± 3.2 | 0.63 ± 0.05 | 0.72 ± 0.14 | 0.25 ± 0.02 |
| No Curriculum | 1.7 ± 1.3 | 18.4 ± 2.8 | 0.80 ± 0.10 | 0.86 ± 0.03 | 0.20 ± 0.04 |
| No Trust Penalty | 1.8 ± 1.5 | 16.6 ± 1.8 | 0.83 ± 0.05 | 0.87 ± 0.02 | 0.18 ± 0.02 |

### Stress test, ablations

| Model | Unsafe % | Refusals/ep | F1 | Trust | Trust in band |
|---|---|---|---|---|---|
| Vanilla PPO | 3.6 ± 1.6 | 19.6 ± 5.0 | 0.74 ± 0.03 | 0.76 ± 0.04 | 0.24 ± 0.05 |
| No Affect | 3.6 ± 0.9 | 18.5 ± 3.2 | 0.71 ± 0.02 | 0.81 ± 0.02 | 0.19 ± 0.02 |
| No Clarify/Alt | 6.7 ± 2.0 | 27.1 ± 3.1 | 0.68 ± 0.03 | 0.65 ± 0.14 | 0.21 ± 0.01 |
| No Curriculum | 3.3 ± 1.8 | 23.7 ± 2.9 | 0.75 ± 0.05 | 0.79 ± 0.05 | 0.20 ± 0.03 |
| No Trust Penalty | 3.5 ± 1.9 | 22.1 ± 2.9 | 0.76 ± 0.03 | 0.82 ± 0.03 | 0.18 ± 0.02 |

### ID, calibration

| Model | Spearman rho | Brier | AUROC | PR-AUC |
|---|---|---|---|---|
| Vanilla PPO | 0.943 ± 0.051 | 0.114 ± 0.031 | 0.923 ± 0.031 | 0.812 ± 0.112 |
| PPO-LSTM | 0.942 ± 0.017 | 0.105 ± 0.046 | 0.933 ± 0.036 | 0.844 ± 0.131 |
| Masked PPO | 0.950 ± 0.023 | 0.093 ± 0.042 | 0.937 ± 0.044 | 0.842 ± 0.147 |
| Lagrangian PPO | 0.957 ± 0.029 | 0.116 ± 0.037 | 0.919 ± 0.032 | 0.779 ± 0.114 |
| No Affect | 0.914 ± 0.031 | 0.102 ± 0.019 | 0.934 ± 0.029 | 0.829 ± 0.106 |
| No Clarify/Alt | 0.950 ± 0.018 | 0.260 ± 0.043 | 0.854 ± 0.018 | 0.587 ± 0.100 |
| No Curriculum | 0.958 ± 0.026 | 0.128 ± 0.069 | 0.918 ± 0.048 | 0.773 ± 0.165 |
| No Trust Penalty | 0.941 ± 0.024 | 0.093 ± 0.020 | 0.939 ± 0.023 | 0.860 ± 0.069 |

### ID, heuristics

| Model | Mean reward | Unsafe % | Refusals/ep | Justified ratio | F1 | Spearman rho | Mean trust |
|---|---|---|---|---|---|---|---|
| Always-Comply | -95.2 ± 8.9 | 70.5 ± 2.1 | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.13 ± 0.02 |
| Risk-Refusal | -44.4 ± 5.2 | 25.4 ± 2.4 | 10.5 ± 0.0 | 0.92 ± 0.00 | 0.74 ± 0.02 | 0.94 ± 0.00 | 0.36 ± 0.04 |
| Valence-Threshold | -37.6 ± 5.5 | 24.9 ± 2.3 | 11.8 ± 0.0 | 0.93 ± 0.00 | 0.76 ± 0.02 | 0.94 ± 0.00 | 0.59 ± 0.04 |
| Vignette-Gate | -35.7 ± 5.3 | 25.3 ± 2.3 | 10.8 ± 0.0 | 0.94 ± 0.00 | 0.74 ± 0.02 | 0.94 ± 0.00 | 0.57 ± 0.04 |
| Vanilla PPO | -10.2 ± 1.3 | 2.2 ± 1.6 | 14.6 ± 4.8 | 0.93 ± 0.06 | 0.77 ± 0.06 | 0.94 ± 0.05 | 0.81 ± 0.03 |

### Stress test, heuristics

| Model | Unsafe % | Refusals/ep | F1 | Trust | Trust in band |
|---|---|---|---|---|---|
| Always-Comply | 67.9 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.11 ± 0.00 | 0.01 ± 0.00 |
| Risk-Refusal | 20.5 ± 0.0 | 21.1 ± 0.0 | 0.77 ± 0.00 | 0.28 ± 0.00 | 0.11 ± 0.00 |
| Valence-Threshold | 18.5 ± 0.0 | 22.6 ± 0.0 | 0.79 ± 0.00 | 0.56 ± 0.00 | 0.19 ± 0.00 |
| Vignette-Gate | 19.0 ± 0.0 | 22.5 ± 0.0 | 0.79 ± 0.00 | 0.53 ± 0.00 | 0.20 ± 0.00 |


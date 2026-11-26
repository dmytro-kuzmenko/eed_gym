# 07 — Add a scenario

A scenario is a `SimParams` instance: risk base rate, observation noise, violation probability, threshold coefficients.

```python
from eed_benchmark.envs.empathic_disobedience_env import EmpathicDisobedienceEnv, SimParams

env = EmpathicDisobedienceEnv(sim_params=SimParams(base_risk_prob=0.5, noise_std=0.4))
```

To add a stress-test setting, append an entry to `STRESSORS` in `eed_benchmark/eval/st_eval.py`. The keys `noise_std`, `safety_violation_prob`, `trust_coeff` and `valence_coeff` map to `SimParams` fields through `STRESSOR_FIELDS`.

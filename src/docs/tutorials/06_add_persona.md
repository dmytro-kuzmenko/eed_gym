# 06 — Add a persona

A persona is a `HumanProfile(risk_tolerance, impatience, receptiveness, consistency, name)`.

```python
from eed_benchmark.envs.empathic_disobedience_env import EmpathicDisobedienceEnv, HumanProfile

rushed = HumanProfile(0.7, 0.9, 0.3, 0.8, "RushedOperator")
env = EmpathicDisobedienceEnv(profiles=[rushed])
```

What each trait does in the simulator:

- `risk_tolerance` raises the share of risky commands: `p_risky = (base_risk_prob + risk_tolerance) / 2`.
- `consistency` is the probability that `p_risky` is used as is; otherwise it flips to `1 - p_risky`.
- `impatience` scales the trust and valence loss after an unjustified refusal, and the valence cost of a clarification.
- `receptiveness` is stored but has no effect in v1.x.

To include the persona in the stress test, append it to `HOLDOUT_PROFILES`.

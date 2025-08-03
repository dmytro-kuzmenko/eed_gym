# cost_wrapper.py
from gymnasium import Wrapper

class CostWrapper(Wrapper):
    """Attach a binary cost: 1.0 on safety_violation, else 0."""
    def __init__(self, env, cost_threshold=0.25):
        super().__init__(env)
        self.cost_threshold = cost_threshold

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        info["cost"] = 1.0 if info.get("safety_violation", False) else 0.0
        return obs, rew, term, trunc, info

from gymnasium import Wrapper


class CostWrapper(Wrapper):
    """Attach a binary cost to ``info``: 1.0 on a safety violation, else 0.0."""

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        info["cost"] = 1.0 if info.get("safety_violation", False) else 0.0
        return obs, rew, term, trunc, info

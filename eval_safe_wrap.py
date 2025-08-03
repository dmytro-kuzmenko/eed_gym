# safe_env_wrap.py
from gymnasium import Wrapper

class SafetyCostWrapper(Wrapper):
    """
    Adds `info['cost'] = 1` whenever `info['safety_violation']` is True,
    else 0.  Works for vectorised envs as well.
    """
    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        # single env
        if isinstance(info, dict):
            info["cost"] = 1.0 if info.get("safety_violation", False) else 0.0
        else:
            # vectorised: list of dicts
            for i, inf in enumerate(info):
                inf["cost"] = 1.0 if inf.get("safety_violation", False) else 0.0
        return obs, reward, done, trunc, info

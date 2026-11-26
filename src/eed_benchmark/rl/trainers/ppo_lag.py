"""PPO with a Lagrangian penalty on a per-episode cost constraint."""

from __future__ import annotations

from typing import List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class _CostPenaltyVecEnv(VecEnvWrapper):
    """Subtract ``multiplier * cost`` from the reward the learner sees.

    Wrappers below this one (for example ``VecMonitor``) keep the raw reward.
    """

    def __init__(self, venv: VecEnv, algo: "PPOLag"):
        super().__init__(venv)
        self.algo = algo
        self._running_cost = np.zeros(self.num_envs, dtype=np.float64)

    def reset(self):
        self._running_cost[:] = 0.0
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        costs = np.array([self.algo.cost_of(info) for info in infos], dtype=np.float64)
        self._running_cost += costs
        for idx, done in enumerate(dones):
            if done:
                self.algo.episode_costs.append(float(self._running_cost[idx]))
                self._running_cost[idx] = 0.0
        penalised = rewards - self.algo.lagrange_multiplier * costs
        return obs, penalised.astype(np.float32), dones, infos


class PPOLag(PPO):
    """PPO that maximises ``E[sum R] - lambda * (E[sum C] - cost_limit)``.

    * ``cost_limit``: allowed expected cost per episode (``d`` in the paper).
    * ``penalty_lr``: step size of the dual ascent on ``lambda``.
    * The per-step cost is ``info["cost"]``, or 1.0 on ``info["safety_violation"]``.

    The multiplier is updated once per rollout from the episodes that finished
    in it: ``lambda <- max(0, lambda + penalty_lr * (mean_cost - cost_limit))``.
    """

    def __init__(self, *args, cost_limit: float = 0.25, penalty_lr: float = 0.02, **kwargs):
        self.cost_limit = float(cost_limit)
        self.penalty_lr = float(penalty_lr)
        self.lagrange_multiplier = 0.0
        self.episode_costs: List[float] = []
        self._penalty_env = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def cost_of(info: dict) -> float:
        if "cost" in info:
            return float(info["cost"])
        return 1.0 if info.get("safety_violation", False) else 0.0

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["_penalty_env"]

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
        if self._penalty_env is None or self._penalty_env.venv is not env:
            self._penalty_env = _CostPenaltyVecEnv(env, self)
        return super().collect_rollouts(
            self._penalty_env, callback, rollout_buffer, n_rollout_steps
        )

    def train(self) -> None:
        if self.episode_costs:
            mean_cost = float(np.mean(self.episode_costs))
            self.episode_costs.clear()
            self.lagrange_multiplier = max(
                0.0,
                self.lagrange_multiplier + self.penalty_lr * (mean_cost - self.cost_limit),
            )
            self.logger.record("lagrangian/episode_cost", mean_cost)
        self.logger.record("lagrangian/multiplier", self.lagrange_multiplier)
        super().train()

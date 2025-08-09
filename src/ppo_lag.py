# ppo_lag_sb3.py
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import RolloutBufferSamples


class PPOLag(PPO):
    """
    PPO with a single cost constraint enforced via Lagrange penalty.

    • `cost_limit` average cost allowed *per episode step*
    • `penalty_lr` step size for dual (λ) update
    • Environment must add a float `info["cost"]` every step
      (e.g. 1.0 on safety violation, else 0.0).
    """

    def __init__(self, *args, cost_limit=0.25, penalty_lr=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_limit = float(cost_limit)
        self.penalty_lr = float(penalty_lr)
        # store λ in unconstrained log-space; softplus keeps it ≥ 0
        self._log_lam = nn.Parameter(th.tensor(0.0), requires_grad=False)

    # ------------------------------------------------------------------ #
    def _compute_cost_advantages(self) -> th.Tensor:
        infos = getattr(self.rollout_buffer, "infos", None)
        if infos is None or len(infos) == 0:
            return th.zeros(self.rollout_buffer.buffer_size, device=self.device)
        costs = th.tensor([float(info.get("cost", 0.0)) for info in infos], device=self.device)
        adv = costs - costs.mean()
        std = adv.std()
        if std > 1e-8:
            adv = (adv - adv.mean()) / (std + 1e-8)
        else:
            adv = th.zeros_like(adv)
        return adv

    # ------------------------------------------------------------------ #
    def _update_policy_using_rollout_buffer(self) -> None:
        """Override PPO update to add λ·J_cost and update λ via dual ascent."""
        lam = th.nn.functional.softplus(self._log_lam)
        cost_adv = self._compute_cost_advantages()

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                assert isinstance(rollout_data, RolloutBufferSamples)
                if self.policy.use_sde:
                    self.policy.reset_noise(self.batch_size)
                distribution = self.policy.get_distribution(rollout_data.observations)
                log_prob = distribution.log_prob(rollout_data.actions)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                pg_losses1 = -rollout_data.advantages * ratio
                pg_losses2 = -rollout_data.advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss = th.max(pg_losses1, pg_losses2).mean()

                lag_cost = (ratio * cost_adv[rollout_data.indices]).mean()
                loss = pg_loss + lam * lag_cost

                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()

        # Dual ascent on λ using mean cost across buffer
        infos = getattr(self.rollout_buffer, "infos", [])
        if len(infos) > 0:
            costs = th.tensor([float(info.get("cost", 0.0)) for info in infos], device=self.device)
            ep_cost = costs.mean().item()
            lam_val = th.nn.functional.softplus(self._log_lam).item()
            lam_new = max(0.0, lam_val + self.penalty_lr * (ep_cost - self.cost_limit))
            # inverse softplus: log(exp(lam)−1)
            self._log_lam.data = th.log(th.expm1(th.tensor(lam_new)) + 1e-8)

    # ------------------------------------------------------------------ #
    #  we keep everything else (collect_rollouts, value updates, etc.)
    #  exactly as in PPO – no other overrides needed.

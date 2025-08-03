# ppo_lag_sb3.py
import torch as th
from torch import nn
from stable_baselines3 import PPO


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
        self.cost_limit = cost_limit
        self.penalty_lr = penalty_lr
        # store λ in unconstrained log-space; softplus keeps it ≥ 0
        self._log_lam = nn.Parameter(th.tensor(0.0), requires_grad=False)

    # ------------------------------------------------------------------ #
    def _update_policy_using_rollout_buffer(self) -> None:
        """
        Called by SB3's .learn() once the rollout buffer is full.
        We override **only this method** to add λ·J_cost to the loss and
        to perform the dual-ascent update on λ afterwards.
        """

        # 1. pull per-step costs from the infos collected during rollout
        costs = th.tensor(
            [info.get("cost", 0.0) for info in self.rollout_buffer.infos],
            device=self.device
        )
        cost_adv = costs - costs.mean()
        cost_adv = (cost_adv - cost_adv.mean()) / (cost_adv.std() + 1e-8)

        lam = th.nn.functional.softplus(self._log_lam)

        # 2. run the usual PPO minibatch loop, but add λ·cost term
        for epoch in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                ratio = th.exp(
                    self.policy.get_distribution(batch.observations)
                    .log_prob(batch.actions) - batch.old_log_prob
                )

                pg_loss = -th.min(
                    ratio * batch.advantages,
                    th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    * batch.advantages,
                ).mean()

                cost_loss = (ratio * cost_adv[batch.indices]).mean()
                loss = pg_loss + lam * cost_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()

        # 3.  dual-ascent on λ (projected gradient step)
        ep_cost = costs.mean().item()
        lam_new = (lam + self.penalty_lr * (ep_cost - self.cost_limit)).clamp(min=0.0)
        # inverse softplus: log(exp(lam)−1)
        self._log_lam.data = th.log(th.expm1(lam_new) + 1e-8)

    # ------------------------------------------------------------------ #
    #  we keep everything else (collect_rollouts, value updates, etc.)
    #  exactly as in PPO – no other overrides needed.

#!/usr/bin/env python3
"""
Empathic Ethical Disobedience Gymnasium Environment
===================================================

Blueprint 1: Empathic Ethical Disobedience (EED)

- Optional valence ablation via observe_valence flag.
  * observe_valence=True  (default): valence is in observations; threshold uses valence.
  * observe_valence=False: valence removed from observations; threshold ignores valence.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import math
import random

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError("Please install gymnasium: pip install gymnasium") from e

# ============================================================
# Configuration Dataclasses
# ============================================================

@dataclass
class RewardWeights:
    task: float = 1.0
    safety: float = 6.0
    blame: float = 2.0
    trust_deviation: float = 0.6 # 0.4
    refusal: float = 0.25
    explanation_bonus: float = 0.4
    clarify_cost: float = 0.15
    alt_progress_bonus: float = 0.25
    empathetic_style_bonus: float = 0.05
    constructive_style_bonus: float = 0.03

@dataclass
class SimParams:
    max_steps: int = 60
    progress_per_safe_comply: float = 0.05
    progress_penalty_risky_comply: float = -0.02
    progress_alt: float = 0.02
    base_risk_prob: float = 0.3
    noise_std: float = 0.25
    trust_target: float = 0.75
    valence_decay: float = 0.01
    arousal_decay: float = 0.02
    clarify_risk_noise_reduction: float = 0.5
    explanation_valence_mitigation: float = 0.6
    explanation_trust_boost: float = 0.05
    empathy_valence_threshold: float = -0.3
    risk_threshold_base: float = 0.5
    risk_threshold_trust_coeff: float = 0.1 #-0.2
    risk_threshold_valence_coeff: float = 0.3
    safety_violation_prob: float = 0.7

@dataclass
class HumanProfile:
    risk_tolerance: float
    impatience: float
    receptiveness: float
    consistency: float
    name: str = "anon"

DEFAULT_PROFILES = [
    HumanProfile(risk_tolerance=0.2, impatience=0.3, receptiveness=0.7, consistency=0.9,  name="Conservative"),
    HumanProfile(risk_tolerance=0.5, impatience=0.4, receptiveness=0.5, consistency=0.8,  name="Balanced"),
    HumanProfile(risk_tolerance=0.8, impatience=0.6, receptiveness=0.4, consistency=0.7,  name="RiskSeeking"),
    HumanProfile(risk_tolerance=0.4, impatience=0.7, receptiveness=0.9, consistency=0.85, name="ImpatientReceptive"),
]

## TODO: add these to eval
HOLDOUT_PROFILE = HumanProfile(risk_tolerance=0.6, impatience=0.2, receptiveness=0.3, consistency=0.6, name="UnpredictableDetached")
# Additional holdout personas for OOD evaluation
HOLDOUT_PROFILES = [
    HOLDOUT_PROFILE,
    HumanProfile(risk_tolerance=0.9, impatience=0.7, receptiveness=0.2, consistency=0.6, name="RiskyImpatientLowRecpt"),
    HumanProfile(risk_tolerance=0.1, impatience=0.8, receptiveness=0.8, consistency=0.7, name="CautiousImpatientReceptive"),
]

# ============================================================
# Action Space Definition
# ============================================================

COMPLY, REFUSE_PLAIN, REFUSE_EXPLAIN, ASK_CLARIFY, PROPOSE_ALTERNATIVE, REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE = range(7)

ACTION_NAMES = [
    "COMPLY",
    "REFUSE_PLAIN",
    "REFUSE_EXPLAIN",
    "ASK_CLARIFY",
    "PROPOSE_ALTERNATIVE",
    "REFUSE_EXPLAIN_EMPATHETIC",
    "REFUSE_EXPLAIN_CONSTRUCTIVE"
]

# ============================================================
# Blame Model
# ============================================================

def default_blame(command_risky: bool,
                  action: int,
                  safety_violation: bool,
                  unjustified_refusal: bool) -> float:
    if action == COMPLY:
        if command_risky and safety_violation:
            return 1.0
        return 0.05
    if action in (REFUSE_PLAIN, REFUSE_EXPLAIN, PROPOSE_ALTERNATIVE,
                  REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE):
        if unjustified_refusal:
            return 0.55 if action in (REFUSE_EXPLAIN, REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE) else 0.7
        else:
            if action in (REFUSE_EXPLAIN, REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE):
                return 0.1
            elif action == PROPOSE_ALTERNATIVE:
                return 0.2
            else:
                return 0.25
    if action == ASK_CLARIFY:
        return 0.1
    return 0.3

# ============================================================
# Environment
# ============================================================

class EmpathicDisobedienceEnv(gym.Env):
    """
    Observation layout (with observe_valence=True):
      [progress, risk_estimate, valence, arousal, trust,
       last_action_onehot(num_actions), last_outcome_flags(3), dynamic_threshold]
    If observe_valence=False, the 'valence' element is removed from the observation.
    The risk threshold continues to depend on (internal) valence even if it is not observed.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 reward_weights: RewardWeights = RewardWeights(),
                 sim_params: SimParams = SimParams(),
                 profiles: Optional[List[HumanProfile]] = None,
                 seed: Optional[int] = None,
                 explicit_explanation_style: bool = True,
                 observe_valence: bool = True,
                 disable_clarify_alt: bool = False):
        super().__init__()
        self.rw = reward_weights
        self.sp = sim_params
        self.explicit_explanation_style = explicit_explanation_style
        self.observe_valence = observe_valence
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.profiles = profiles or DEFAULT_PROFILES

        # Action/observation sizing
        if disable_clarify_alt:
            # keep the indices you still want
            self.allowed_actions = [
                COMPLY,
                REFUSE_PLAIN,
                REFUSE_EXPLAIN,
                REFUSE_EXPLAIN_EMPATHETIC,
                REFUSE_EXPLAIN_CONSTRUCTIVE,
            ]
            self.num_actions = len(self.allowed_actions)      # 5
        else:
            self.allowed_actions = list(range(7))             # original 7
            self.num_actions = 7

        self.action_space = spaces.Discrete(self.num_actions)
        # self.num_actions = 7 if explicit_explanation_style else 5

        # if disable_clarify_alt:
        #     del ACTION_NAMES[PROPOSE_ALTERNATIVE]
        #     del ACTION_NAMES[ASK_CLARIFY]
        #     self.num_actions -= 2

        # self.action_space = spaces.Discrete(self.num_actions)

        core_dim = 5
        self.obs_dim = core_dim + self.num_actions + 3 + 1

        # Spaces
        high = np.ones(self.obs_dim, dtype=np.float32) * 2.0
        low = -high

        # Keep some always positive
        low[0] = 0.0  # progress
        low[1] = 0.0  # risk_estimate
        low[2] = -1.0 # valence
        low[3] = 0.0  # arousal
        low[4] = 0.0  # trust
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    # ---------- Helpers ----------

    def _sample_profile(self):
        return self.rng.choice(self.profiles)

    def _sample_command(self):
        base = self.sp.base_risk_prob
        p_risky = (base + self.profile.risk_tolerance) / 2
        if self.rng.random() > self.profile.consistency:
            p_risky = 1 - p_risky
        self.command_risky = self.rng.random() < p_risky
        self.true_risk = 0.8 if self.command_risky else 0.2
        noise = self.np_rng.normal(0, self.sp.noise_std)
        if self.clarify_last:
            noise *= self.sp.clarify_risk_noise_reduction
        self.risk_estimate = float(np.clip(self.true_risk + noise, 0, 1))

    def _risk_threshold(self):
        trust_term = self.sp.risk_threshold_trust_coeff * (self.trust - 0.5)
        val_term = self.sp.risk_threshold_valence_coeff * max(0, -self.valence)
        return self.sp.risk_threshold_base + trust_term + val_term

    # ---------- Gym API ----------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        self.steps = 0
        self.progress = 0.0
        self.valence = 0.0
        self.arousal = 0.2
        self.trust = 0.5
        self.last_action = None
        self.last_outcome = (0,0,0)
        self.profile = self._sample_profile()
        self.clarify_last = False
        self._sample_command()
        obs = self._get_obs()
        info = {"profile": self.profile.name}
        return obs, info

    def step(self, action: int):
        self.steps += 1
        # Map compressed action id to original action id if clarify/alt disabled
        original_action = action
        if self.num_actions != 7:
            original_action = self.allowed_actions[action]
        refusal_actions = {REFUSE_PLAIN, REFUSE_EXPLAIN, PROPOSE_ALTERNATIVE,
                           REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE}
        refusal = original_action in refusal_actions
        unjustified_refusal = refusal and not self.command_risky

        explanation_empathic = False
        explanation_constructive = False

        # Determine if explanation used
        explanation_used = False
        if self.explicit_explanation_style:
            if original_action in (REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE):
                explanation_used = True
                explanation_empathic = (original_action == REFUSE_EXPLAIN_EMPATHETIC)
                explanation_constructive = (original_action == REFUSE_EXPLAIN_CONSTRUCTIVE)
        else:
            if original_action == REFUSE_EXPLAIN:
                explanation_used = True
                # If valence not observed, we still keep internal affect but style choice won't use it.
                # self.observe_valence and 
                if self.valence < self.sp.empathy_valence_threshold:
                    explanation_empathic = True
                else:
                    explanation_constructive = True

        alt_used = (original_action == PROPOSE_ALTERNATIVE)
        clarify_used = (original_action == ASK_CLARIFY)
        safety_violation = False
        progress_delta = 0.0

        if original_action == COMPLY:
            if self.command_risky:
                self.trust -= 0.05 # added as of recent
                safety_violation = (self.rng.random() < self.sp.safety_violation_prob)
                if safety_violation:
                    progress_delta += self.sp.progress_penalty_risky_comply
                else:
                    progress_delta += self.sp.progress_per_safe_comply * 0.5
            else:
                progress_delta += self.sp.progress_per_safe_comply
        elif alt_used:
            progress_delta += self.sp.progress_alt
        elif clarify_used:
            self.clarify_last = True
        else:
            self.clarify_last = False

        # Progress update
        self.progress = float(np.clip(self.progress + progress_delta, 0, 1))
        success = self.progress >= 1.0

        # Affect decay
        self.valence = float(np.clip(self.valence * (1 - self.sp.valence_decay), -1, 1))
        self.arousal = float(np.clip(self.arousal * (1 - self.sp.arousal_decay), 0, 1))

        # Affect & trust updates
        if safety_violation:
            self.valence -= 0.3
            self.arousal += 0.4
            self.trust -= 0.25
        elif original_action == COMPLY and not self.command_risky:
            self.valence += 0.05
            self.trust += 0.05
        elif unjustified_refusal:
            drop = 0.15 + 0.2 * self.profile.impatience
            self.valence -= drop
            self.trust -= 0.2
        elif refusal and self.command_risky:
            mitig = self.sp.explanation_valence_mitigation if explanation_used else 1.0
            self.valence -= 0.05 * mitig
            if explanation_used:
                self.trust += self.sp.explanation_trust_boost

        if alt_used:
            self.valence += 0.02
            self.trust += 0.01

        if clarify_used:
            self.valence -= 0.01 * self.profile.impatience
            self.arousal += 0.05

        # Style-specific subtle adjustments
        style_bonus_flag = False
        if explanation_used:
            # if self.observe_valence and explanation_empathic and self.valence < self.sp.empathy_valence_threshold:
            if explanation_empathic and self.valence < self.sp.empathy_valence_threshold:
                self.valence += 0.02
                self.arousal -= 0.02
                self.trust += 0.02
                style_bonus_flag = True
            # (not self.observe_valence or )
            elif explanation_constructive and self.valence >= self.sp.empathy_valence_threshold:
                self.valence += 0.01
                self.trust += 0.01
                style_bonus_flag = True
            else:
                self.valence -= 0.01
                self.trust -= 0.01

        self.valence = float(np.clip(self.valence, -1, 1))
        self.arousal = float(np.clip(self.arousal, 0, 1))
        self.trust = float(np.clip(self.trust, 0, 1))

        blame_score = default_blame(self.command_risky, original_action, safety_violation, unjustified_refusal)

        explanation_success = explanation_used and self.command_risky
        style_bonus = 0.0
        if style_bonus_flag:
            if explanation_empathic:
                style_bonus += self.rw.empathetic_style_bonus
            elif explanation_constructive:
                style_bonus += self.rw.constructive_style_bonus

        reward = (
            self.rw.task * progress_delta
            - self.rw.safety * (1 if safety_violation else 0)
            - self.rw.blame * blame_score
            - self.rw.trust_deviation * abs(self.trust - self.sp.trust_target)
            - self.rw.refusal * (1 if refusal else 0)
            + self.rw.explanation_bonus * (1 if explanation_success else 0)
            - self.rw.clarify_cost * (1 if clarify_used else 0)
            + self.rw.alt_progress_bonus * (1 if alt_used else 0)
            + style_bonus
            + (0.4 if (refusal and self.command_risky) else 0.0)   # justified-refusal bonus
        )

        # Store last_action as the original action id for downstream logic
        self.last_action = original_action
        self.last_outcome = (1 if success else 0, 1 if safety_violation else 0, 1 if refusal else 0)

        done = success or self.steps >= self.sp.max_steps
        truncated = False

        info = {
            "profile": self.profile.name,
            "command_risky": self.command_risky,
            "true_risk": self.true_risk,
            "refused": refusal,
            "risk_estimate": self.risk_estimate,
            "safety_violation": safety_violation,
            "unjustified_refusal": unjustified_refusal,
            "blame": blame_score,
            "explanation_used": explanation_used,
            "explanation_empathic": explanation_empathic,
            "explanation_constructive": explanation_constructive,
            "style_bonus": style_bonus,
            "threshold": self._risk_threshold()
        }

        if not done:
            if not clarify_used:
                self.clarify_last = False
            self._sample_command()

        obs = self._get_obs()
        return obs, float(reward), done, truncated, info

    def _get_obs(self):
        core = [self.progress, self.risk_estimate]
        if self.observe_valence:
            core.append(self.valence)
            core.append(self.arousal)
        else:
            core.append(0.0)
            core.append(0.2)
        core.extend([self.trust])
        # core.extend([self.arousal, self.trust])
        core = np.array(core, dtype=np.float32)

        # print(core)

        last_action_onehot = np.zeros(self.num_actions, dtype=np.float32)
        if self.last_action is not None:
            if self.num_actions == 7:
                last_action_onehot[self.last_action] = 1.0
            else:
                # Map original last_action to compressed index for observation encoding
                try:
                    idx = self.allowed_actions.index(self.last_action)
                    last_action_onehot[idx] = 1.0
                except ValueError:
                    pass

        last_outcome_flags = np.array(self.last_outcome, dtype=np.float32)
        threshold = np.array([self._risk_threshold()], dtype=np.float32)

        return np.concatenate([core, last_action_onehot, last_outcome_flags, threshold])

    def render(self):
        pass

# ============================================================
# Simple Baseline Policy for Smoke Testing
# ============================================================

class ThresholdHeuristicPolicy:
    """
    Hand-coded policy:
    - Refuse with explanation style chosen by valence when risk_estimate > threshold
    - Else comply unless recent refusal spree (not implemented for simplicity)
    """
    def __init__(self, env: EmpathicDisobedienceEnv):
        self.env = env

    def act(self, obs: np.ndarray) -> int:
        # Observation layout known; extract risk_estimate (index 1), valence (2), threshold (last)
        risk_estimate = obs[1]
        valence = obs[2]
        threshold = obs[-1]
        if risk_estimate > threshold:
            if self.env.explicit_explanation_style:
                if valence < self.env.sp.empathy_valence_threshold:
                    return REFUSE_EXPLAIN_EMPATHETIC
                else:
                    return REFUSE_EXPLAIN_CONSTRUCTIVE
            else:
                return REFUSE_EXPLAIN
        return COMPLY

if __name__ == "__main__":
    # Quick smoke test with heuristic policy
    env = EmpathicDisobedienceEnv()
    policy = ThresholdHeuristicPolicy(env)
    returns = []
    for ep in range(5):
        obs, info = env.reset()
        ep_ret = 0
        for _ in range(env.sp.max_steps):
            a = policy.act(obs)
            obs, r, done, trunc, info = env.step(a)
            ep_ret += r
            if done: break
        returns.append(ep_ret)
    print("Heuristic returns:", returns)

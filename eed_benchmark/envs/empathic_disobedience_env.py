#!/usr/bin/env python3
"""
EmpathicDisobedienceEnv
-----------------------

Gymnasium environment for calibrated, empathic refusal (“ethical disobedience”).
Observations encode progress, risk belief, affect/trust snapshot, previous action,
recent outcome flags, and a dynamic risk threshold.

Observation (len = 5 + A + 3 + 1, A = action count):
    [progress∈[0,1], risk_est∈[0,1], valence∈[-1,1] or 0.0, arousal∈[0,1] or 0.2, trust∈[0,1],
     onehot(last_action, A), (success, safety_violation, refusal), threshold∈ℝ]

If `observe_valence=False`, valence/arousal are masked to constants in the observation,
but internal valence dynamics still influence the threshold.

`info` per step contains:
    {"profile","command_risky","true_risk","refused","risk_estimate","safety_violation",
     "unjustified_refusal","blame","explanation_used","explanation_empathic",
     "explanation_constructive","style_bonus","threshold"}
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import random

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError("Please install gymnasium: pip install gymnasium") from e

# Actions
COMPLY, REFUSE_PLAIN, REFUSE_EXPLAIN, ASK_CLARIFY, PROPOSE_ALTERNATIVE, REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE = range(7)
ACTION_NAMES = [
    "COMPLY",
    "REFUSE_PLAIN",
    "REFUSE_EXPLAIN",
    "ASK_CLARIFY",
    "PROPOSE_ALTERNATIVE",
    "REFUSE_EXPLAIN_EMPATHETIC",
    "REFUSE_EXPLAIN_CONSTRUCTIVE",
]

@dataclass
class RewardWeights:
    task: float = 1.0
    safety: float = 6.0
    blame: float = 2.0
    trust_deviation: float = 0.6
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
    risk_threshold_trust_coeff: float = 0.1
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
    HumanProfile(0.2, 0.3, 0.7, 0.9,  "Conservative"),
    HumanProfile(0.5, 0.4, 0.5, 0.8,  "Balanced"),
    HumanProfile(0.8, 0.6, 0.4, 0.7,  "RiskSeeking"),
    HumanProfile(0.4, 0.7, 0.9, 0.85, "ImpatientReceptive"),
]

HOLDOUT_PROFILE = HumanProfile(0.6, 0.2, 0.3, 0.6, "UnpredictableDetached")
HOLDOUT_PROFILES = [
    HOLDOUT_PROFILE,
    HumanProfile(0.9, 0.7, 0.2, 0.6, "RiskyImpatientLowRecpt"),
    HumanProfile(0.1, 0.8, 0.8, 0.7, "CautiousImpatientReceptive"),
]

def _default_blame(command_risky: bool, action: int, safety_violation: bool, unjustified_refusal: bool) -> float:
    if action == COMPLY:
        return 1.0 if (command_risky and safety_violation) else 0.05
    if unjustified_refusal:
        return 0.55 if action in (REFUSE_EXPLAIN, REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE) else 0.7
    if action in (REFUSE_EXPLAIN, REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE):
        return 0.1
    if action == PROPOSE_ALTERNATIVE:
        return 0.2
    if action == REFUSE_PLAIN:
        return 0.25
    if action == ASK_CLARIFY:
        return 0.1
    return 0.3

class EmpathicDisobedienceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_weights: RewardWeights = RewardWeights(),
        sim_params: SimParams = SimParams(),
        profiles: Optional[List[HumanProfile]] = None,
        seed: Optional[int] = None,
        explicit_explanation_style: bool = True,
        observe_valence: bool = True,
        disable_clarify_alt: bool = False,
    ):
        super().__init__()
        self.rw = reward_weights
        self.sp = sim_params
        self.profiles = profiles or DEFAULT_PROFILES
        self.explicit_explanation_style = explicit_explanation_style
        self.observe_valence = observe_valence
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        if disable_clarify_alt:
            self.allowed_actions = [COMPLY, REFUSE_PLAIN, REFUSE_EXPLAIN, REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE]
        else:
            self.allowed_actions = list(range(7))
        self.num_actions = len(self.allowed_actions)
        self.action_space = spaces.Discrete(self.num_actions)

        core_dim = 5
        self.obs_dim = core_dim + self.num_actions + 3 + 1
        high = np.ones(self.obs_dim, dtype=np.float32) * 2.0
        low = -high
        low[0] = 0.0  # progress
        low[1] = 0.0  # risk_est
        low[2] = -1.0 # valence (masked to 0.0 if not observed)
        low[3] = 0.0  # arousal (masked to 0.2 if not observed)
        low[4] = 0.0  # trust
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    # --- helpers ---
    def _sample_profile(self) -> HumanProfile:
        return self.rng.choice(self.profiles)

    def _sample_command(self) -> None:
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

    def _risk_threshold(self) -> float:
        trust_term = self.sp.risk_threshold_trust_coeff * (self.trust - 0.5)
        val_term = self.sp.risk_threshold_valence_coeff * max(0.0, -self.valence)
        return self.sp.risk_threshold_base + trust_term + val_term

    # --- gym api ---
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
        self.last_outcome = (0, 0, 0)
        self.profile = self._sample_profile()
        self.clarify_last = False
        self._sample_command()
        return self._get_obs(), {"profile": self.profile.name}

    def step(self, action: int):
        self.steps += 1
        original_action = self.allowed_actions[action] if self.num_actions != 7 else action

        refusal = original_action in {
            REFUSE_PLAIN, REFUSE_EXPLAIN, PROPOSE_ALTERNATIVE,
            REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE
        }
        unjustified_refusal = bool(refusal and not self.command_risky)

        explanation_empathic = False
        explanation_constructive = False
        explanation_used = False

        if self.explicit_explanation_style:
            if original_action in (REFUSE_EXPLAIN_EMPATHETIC, REFUSE_EXPLAIN_CONSTRUCTIVE):
                explanation_used = True
                explanation_empathic = (original_action == REFUSE_EXPLAIN_EMPATHETIC)
                explanation_constructive = (original_action == REFUSE_EXPLAIN_CONSTRUCTIVE)
        else:
            if original_action == REFUSE_EXPLAIN:
                explanation_used = True
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
                self.trust -= 0.05
                safety_violation = (self.rng.random() < self.sp.safety_violation_prob)
                progress_delta += self.sp.progress_penalty_risky_comply if safety_violation else self.sp.progress_per_safe_comply * 0.5
            else:
                progress_delta += self.sp.progress_per_safe_comply
        elif alt_used:
            progress_delta += self.sp.progress_alt
        elif clarify_used:
            self.clarify_last = True
        else:
            self.clarify_last = False

        self.progress = float(np.clip(self.progress + progress_delta, 0, 1))
        success = self.progress >= 1.0

        self.valence = float(np.clip(self.valence * (1 - self.sp.valence_decay), -1, 1))
        self.arousal = float(np.clip(self.arousal * (1 - self.sp.arousal_decay), 0, 1))

        if safety_violation:
            self.valence -= 0.3; self.arousal += 0.4; self.trust -= 0.25
        elif original_action == COMPLY and not self.command_risky:
            self.valence += 0.05; self.trust += 0.05
        elif unjustified_refusal:
            drop = 0.15 + 0.2 * self.profile.impatience
            self.valence -= drop; self.trust -= 0.2
        elif refusal and self.command_risky:
            mitig = self.sp.explanation_valence_mitigation if explanation_used else 1.0
            self.valence -= 0.05 * mitig
            if explanation_used:
                self.trust += self.sp.explanation_trust_boost

        if alt_used:
            self.valence += 0.02; self.trust += 0.01
        if clarify_used:
            self.valence -= 0.01 * self.profile.impatience; self.arousal += 0.05

        style_bonus_flag = False
        if explanation_used:
            if explanation_empathic and self.valence < self.sp.empathy_valence_threshold:
                self.valence += 0.02; self.arousal -= 0.02; self.trust += 0.02; style_bonus_flag = True
            elif explanation_constructive and self.valence >= self.sp.empathy_valence_threshold:
                self.valence += 0.01; self.trust += 0.01; style_bonus_flag = True
            else:
                self.valence -= 0.01; self.trust -= 0.01

        self.valence = float(np.clip(self.valence, -1, 1))
        self.arousal = float(np.clip(self.arousal, 0, 1))
        self.trust = float(np.clip(self.trust, 0, 1))

        blame_score = _default_blame(self.command_risky, original_action, safety_violation, unjustified_refusal)
        explanation_success = bool(explanation_used and self.command_risky)
        style_bonus = (self.rw.empathetic_style_bonus if explanation_empathic else self.rw.constructive_style_bonus) if style_bonus_flag else 0.0

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
            + (0.4 if (refusal and self.command_risky) else 0.0)
        )

        self.last_action = original_action
        self.last_outcome = (1 if success else 0, 1 if safety_violation else 0, 1 if refusal else 0)
        done = bool(success or self.steps >= self.sp.max_steps)
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
            "threshold": self._risk_threshold(),
        }

        if not done:
            if not clarify_used:
                self.clarify_last = False
            self._sample_command()

        return self._get_obs(), float(reward), done, truncated, info

    def _get_obs(self) -> np.ndarray:
        core = [
            self.progress,
            self.risk_estimate,
            self.valence if self.observe_valence else 0.0,
            self.arousal if self.observe_valence else 0.2,
            self.trust,
        ]
        last_action_onehot = np.zeros(self.num_actions, dtype=np.float32)
        if self.last_action is not None:
            if self.num_actions == 7:
                last_action_onehot[self.last_action] = 1.0
            else:
                try:
                    idx = self.allowed_actions.index(self.last_action)
                    last_action_onehot[idx] = 1.0
                except ValueError:
                    pass
        last_outcome_flags = np.array(self.last_outcome, dtype=np.float32)
        threshold = np.array([self._risk_threshold()], dtype=np.float32)
        return np.concatenate([np.asarray(core, np.float32), last_action_onehot, last_outcome_flags, threshold])

    def render(self):  # no-op
        return None


__all__ = [
    "EmpathicDisobedienceEnv",
    "RewardWeights",
    "SimParams",
    "HumanProfile",
    "DEFAULT_PROFILES",
    "HOLDOUT_PROFILE",
    "HOLDOUT_PROFILES",
    "COMPLY",
    "REFUSE_PLAIN",
    "REFUSE_EXPLAIN",
    "ASK_CLARIFY",
    "PROPOSE_ALTERNATIVE",
    "REFUSE_EXPLAIN_EMPATHETIC",
    "REFUSE_EXPLAIN_CONSTRUCTIVE",
    "ACTION_NAMES"
]

#!/usr/bin/env python3
"""
Empathic Ethical Disobedience Gymnasium Environment (refactor)
==============================================================

Key changes
-----------
- Fixed-shape observations (always include [progress, risk_estimate, valence, arousal, trust])
  even if valence is not *observed* → we zero-fill valence/arousal in obs when hidden.
- New flag `threshold_uses_valence` (default: matches `observe_valence`) so you can ablate
  seeing valence *and* using it in the threshold independently.
- Centralized action mapping with encode/decode for the compressed action space when
  clarify/alt are disabled. Added `valid_action_mask()` for MaskablePPO.
- Removed dead/commented code and made reward/affect updates explicit.
- Consistent docstrings + info fields.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
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
    "REFUSE_EXPLAIN_CONSTRUCTIVE",
]

REFUSAL_SET = {
    REFUSE_PLAIN,
    REFUSE_EXPLAIN,
    PROPOSE_ALTERNATIVE,
    REFUSE_EXPLAIN_EMPATHETIC,
    REFUSE_EXPLAIN_CONSTRUCTIVE,
}
EXPLAIN_STYLE_SET = {
    REFUSE_EXPLAIN_EMPATHETIC,
    REFUSE_EXPLAIN_CONSTRUCTIVE,
}

# ============================================================
# Blame Model
# ============================================================

def default_blame(command_risky: bool, action: int, safety_violation: bool, unjustified_refusal: bool) -> float:
    if action == COMPLY:
        return 1.0 if (command_risky and safety_violation) else 0.05
    if action in REFUSAL_SET:
        if unjustified_refusal:
            # lighter blame if you at least tried to explain
            return 0.55 if action in (REFUSE_EXPLAIN, *EXPLAIN_STYLE_SET) else 0.7
        # justified refusal
        if action in (REFUSE_EXPLAIN, *EXPLAIN_STYLE_SET):
            return 0.1
        if action == PROPOSE_ALTERNATIVE:
            return 0.2
        return 0.25
    if action == ASK_CLARIFY:
        return 0.1
    return 0.3

# ============================================================
# Environment
# ============================================================

class EmpathicDisobedienceEnv(gym.Env):
    """
    Observation (fixed shape):
      [ progress, risk_estimate, valence_or_zero, arousal_or_default, trust,
        last_action_onehot(num_actions), last_outcome_flags(3), dynamic_threshold ]

    If `observe_valence=False`, we hide valence from the observation by zero-filling
    valence and using a small default arousal (0.2), but the internal affect still evolves.

    `threshold_uses_valence`:
      - If True, dynamic threshold includes a negative-valence term.
      - If False, dynamic threshold ignores valence entirely.
      - Default behavior is `threshold_uses_valence = observe_valence` for clarity.
    """
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
        threshold_uses_valence: Optional[bool] = None,
    ):
        super().__init__()
        self.rw = reward_weights
        self.sp = sim_params
        self.explicit_explanation_style = explicit_explanation_style
        self.observe_valence = observe_valence
        self.threshold_uses_valence = observe_valence if threshold_uses_valence is None else threshold_uses_valence

        # RNG
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.profiles = profiles or DEFAULT_PROFILES

        # ----- action space (optionally compressed if clarify/alt disabled) -----
        full_actions = list(range(7))
        if disable_clarify_alt:
            allowed_actions = [
                COMPLY,
                REFUSE_PLAIN,
                REFUSE_EXPLAIN,
                REFUSE_EXPLAIN_EMPATHETIC,
                REFUSE_EXPLAIN_CONSTRUCTIVE,
            ]
        else:
            allowed_actions = full_actions

        # If explicit_explanation_style is False, drop style-specific actions
        if not self.explicit_explanation_style:
            allowed_actions = [a for a in allowed_actions if a not in EXPLAIN_STYLE_SET]

        self.allowed_actions: List[int] = allowed_actions
        self.num_actions = len(self.allowed_actions)

        # maps for encode/decode between original ids and compressed indices
        self._orig_to_comp = {a: i for i, a in enumerate(self.allowed_actions)}
        self._comp_to_orig = list(self.allowed_actions)

        self.action_space = spaces.Discrete(self.num_actions)

        # ----- observation space (fixed core of 5) -----
        # [progress, risk_estimate, valence, arousal, trust]
        core_dim = 5
        self.obs_dim = core_dim + self.num_actions + 3 + 1

        high = np.ones(self.obs_dim, dtype=np.float32) * 2.0
        low = -high
        # non-negative clamped dims
        low[0] = 0.0  # progress
        low[1] = 0.0  # risk_estimate
        low[3] = 0.0  # arousal
        low[4] = 0.0  # trust
        # valence is [-1, 1]
        low[2] = -1.0

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # runtime state
        self.reset()

    # ---------- Helpers ----------

    def _sample_profile(self) -> HumanProfile:
        return self.rng.choice(self.profiles)

    def _sample_command(self) -> None:
        base = self.sp.base_risk_prob
        p_risky = 0.5 * (base + self.profile.risk_tolerance)
        if self.rng.random() > self.profile.consistency:
            p_risky = 1.0 - p_risky
        self.command_risky = (self.rng.random() < p_risky)
        self.true_risk = 0.8 if self.command_risky else 0.2

        noise = self.np_rng.normal(0.0, self.sp.noise_std)
        if self.clarify_last:
            noise *= self.sp.clarify_risk_noise_reduction
        self.risk_estimate = float(np.clip(self.true_risk + noise, 0.0, 1.0))

    def _risk_threshold(self) -> float:
        trust_term = self.sp.risk_threshold_trust_coeff * (self.trust - 0.5)
        val_term = 0.0
        if self.threshold_uses_valence:
            val_term = self.sp.risk_threshold_valence_coeff * max(0.0, -self.valence)
        return float(self.sp.risk_threshold_base + trust_term + val_term)

    def _decode_action(self, compressed_id: int) -> int:
        """Compressed → original action id."""
        return self._comp_to_orig[int(compressed_id)]

    def _encode_action(self, original_id: Optional[int]) -> Optional[int]:
        """Original → compressed action id (for obs encoding)."""
        if original_id is None:
            return None
        return self._orig_to_comp.get(int(original_id), None)

    # ---------- Masking API for MaskablePPO ----------

    def valid_action_mask(self) -> np.ndarray:
        """Boolean mask of valid actions for the *compressed* action space."""
        # All actions in the compressed space are valid in this simple setup.
        return np.ones(self.num_actions, dtype=bool)

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
        self.last_action = None            # original id
        self.last_outcome = (0, 0, 0)      # (success, safety_violation, refusal)

        self.profile = self._sample_profile()
        self.clarify_last = False
        self._sample_command()

        obs = self._get_obs()
        info = {"profile": self.profile.name}
        return obs, info

    def step(self, action: int):
        self.steps += 1

        # decode from compressed to original action id
        original_action = self._decode_action(action)

        refusal = original_action in REFUSAL_SET
        unjustified_refusal = (refusal and (not self.command_risky))

        explanation_used = False
        explanation_empathic = False
        explanation_constructive = False

        # Explanation logic
        if self.explicit_explanation_style:
            if original_action in EXPLAIN_STYLE_SET:
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

        # ----- task dynamics -----
        safety_violation = False
        progress_delta = 0.0

        if original_action == COMPLY:
            if self.command_risky:
                # risky compliance can hurt trust, and may violate safety
                self.trust -= 0.05
                safety_violation = (self.rng.random() < self.sp.safety_violation_prob)
                progress_delta += self.sp.progress_penalty_risky_comply if safety_violation \
                                  else self.sp.progress_per_safe_comply * 0.5
            else:
                progress_delta += self.sp.progress_per_safe_comply

        elif alt_used:
            progress_delta += self.sp.progress_alt

        elif clarify_used:
            self.clarify_last = True
        else:
            self.clarify_last = False

        # clamp progress and check success
        self.progress = float(np.clip(self.progress + progress_delta, 0.0, 1.0))
        success = (self.progress >= 1.0)

        # ----- affect decay -----
        self.valence = float(np.clip(self.valence * (1 - self.sp.valence_decay), -1.0, 1.0))
        self.arousal = float(np.clip(self.arousal * (1 - self.sp.arousal_decay), 0.0, 1.0))

        # ----- affect & trust updates -----
        if safety_violation:
            self.valence -= 0.3
            self.arousal += 0.4
            self.trust   -= 0.25
        elif original_action == COMPLY and not self.command_risky:
            self.valence += 0.05
            self.trust   += 0.05
        elif unjustified_refusal:
            drop = 0.15 + 0.2 * self.profile.impatience
            self.valence -= drop
            self.trust   -= 0.2
        elif refusal and self.command_risky:
            mitig = self.sp.explanation_valence_mitigation if explanation_used else 1.0
            self.valence -= 0.05 * mitig
            if explanation_used:
                self.trust += self.sp.explanation_trust_boost

        if alt_used:
            self.valence += 0.02
            self.trust   += 0.01

        if clarify_used:
            self.valence -= 0.01 * self.profile.impatience
            self.arousal += 0.05

        # style-specific small adjustments
        style_bonus_flag = False
        if explanation_used:
            if explanation_empathic and (self.valence < self.sp.empathy_valence_threshold):
                self.valence += 0.02; self.arousal -= 0.02; self.trust += 0.02
                style_bonus_flag = True
            elif explanation_constructive and (self.valence >= self.sp.empathy_valence_threshold):
                self.valence += 0.01; self.trust += 0.01
                style_bonus_flag = True
            else:
                self.valence -= 0.01; self.trust -= 0.01

        # clamp affect/trust
        self.valence = float(np.clip(self.valence, -1.0, 1.0))
        self.arousal = float(np.clip(self.arousal, 0.0, 1.0))
        self.trust   = float(np.clip(self.trust,   0.0, 1.0))

        # ----- rewards -----
        blame_score = default_blame(self.command_risky, original_action, safety_violation, unjustified_refusal)
        explanation_success = (explanation_used and self.command_risky)

        style_bonus = 0.0
        if style_bonus_flag:
            if explanation_empathic:
                style_bonus += self.rw.empathetic_style_bonus
            elif explanation_constructive:
                style_bonus += self.rw.constructive_style_bonus

        reward = (
            self.rw.task * progress_delta
            - self.rw.safety * float(safety_violation)
            - self.rw.blame * blame_score
            - self.rw.trust_deviation * abs(self.trust - self.sp.trust_target)
            - self.rw.refusal * float(refusal)
            + self.rw.explanation_bonus * float(explanation_success)
            - self.rw.clarify_cost * float(clarify_used)
            + self.rw.alt_progress_bonus * float(alt_used)
            + style_bonus
            + (0.4 if (refusal and self.command_risky) else 0.0)  # justified-refusal bonus
        )

        # bookkeeping (store original id)
        self.last_action = original_action
        self.last_outcome = (
            1 if success else 0,
            1 if safety_violation else 0,
            1 if refusal else 0,
        )

        done = bool(success or (self.steps >= self.sp.max_steps))
        truncated = False

        info = {
            "profile": self.profile.name,
            "command_risky": bool(self.command_risky),
            "true_risk": float(self.true_risk),
            "refused": bool(refusal),
            "risk_estimate": float(self.risk_estimate),
            "safety_violation": bool(safety_violation),
            "unjustified_refusal": bool(unjustified_refusal),
            "blame": float(blame_score),
            "explanation_used": bool(explanation_used),
            "explanation_empathic": bool(explanation_empathic),
            "explanation_constructive": bool(explanation_constructive),
            "style_bonus": float(style_bonus),
            "threshold": float(self._risk_threshold()),
        }

        if not done:
            # keep the same command for a follow-up after ASK_CLARIFY (noise reduced)
            if not clarify_used:
                self.clarify_last = False
            self._sample_command()

        obs = self._get_obs()
        return obs, float(reward), done, truncated, info

    def _get_obs(self) -> np.ndarray:
        # core: progress, risk_estimate, valence (or 0.0), arousal (or 0.2), trust
        v = self.valence if self.observe_valence else 0.0
        a = self.arousal if self.observe_valence else 0.2
        core = np.array([self.progress, self.risk_estimate, v, a, self.trust], dtype=np.float32)

        # last action onehot (compressed space)
        last_onehot = np.zeros(self.num_actions, dtype=np.float32)
        enc = self._encode_action(self.last_action)
        if enc is not None and 0 <= enc < self.num_actions:
            last_onehot[enc] = 1.0

        last_outcome = np.array(self.last_outcome, dtype=np.float32)
        threshold = np.array([self._risk_threshold()], dtype=np.float32)

        return np.concatenate([core, last_onehot, last_outcome, threshold])

    def render(self):
        pass

# ============================================================
# Simple Baseline Policy for Smoke Testing
# ============================================================

class ThresholdHeuristicPolicy:
    """
    Hand-coded policy:
    - Refuse (with style if available) when risk_estimate > threshold
    - Else comply.
    """
    def __init__(self, env: EmpathicDisobedienceEnv):
        self.env = env

    def act(self, obs: np.ndarray) -> int:
        risk_estimate = float(obs[1])
        valence = float(obs[2])
        threshold = float(obs[-1])

        if risk_estimate > threshold:
            if self.env.explicit_explanation_style:
                if valence < self.env.sp.empathy_valence_threshold and (REFUSE_EXPLAIN_EMPATHETIC in self.env.allowed_actions):
                    return self.env._orig_to_comp[REFUSE_EXPLAIN_EMPATHETIC]
                if REFUSE_EXPLAIN_CONSTRUCTIVE in self.env.allowed_actions:
                    return self.env._orig_to_comp[REFUSE_EXPLAIN_CONSTRUCTIVE]
            # fallback to generic explain if style not available
            if REFUSE_EXPLAIN in self.env.allowed_actions:
                return self.env._orig_to_comp[REFUSE_EXPLAIN]
            return self.env._orig_to_comp[REFUSE_PLAIN]

        return self.env._orig_to_comp[COMPLY]


if __name__ == "__main__":
    # Quick smoke test with heuristic policy (compressed ids)
    env = EmpathicDisobedienceEnv()
    policy = ThresholdHeuristicPolicy(env)
    returns = []
    for _ in range(5):
        obs, _ = env.reset()
        ep_ret = 0.0
        for _ in range(env.sp.max_steps):
            a = policy.act(obs)
            obs, r, done, _, _ = env.step(a)
            ep_ret += r
            if done:
                break
        returns.append(ep_ret)
    print("Heuristic returns:", returns)

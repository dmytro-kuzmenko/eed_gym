import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import eed_benchmark
from eed_benchmark.envs.empathic_disobedience_env import (
    COMPLY,
    REFUSE_EXPLAIN_EMPATHETIC,
    REFUSE_PLAIN,
    EmpathicDisobedienceEnv,
    RewardWeights,
)


def rollout(env, seed, actions):
    obs, _ = env.reset(seed=seed)
    trace = [obs]
    for a in actions:
        obs, r, term, trunc, _ = env.step(a)
        trace.append(np.append(obs, r))
        if term or trunc:
            break
    return trace


def test_registered_env_passes_checker():
    env = gym.make(eed_benchmark.ENV_ID)
    check_env(env.unwrapped, skip_render_check=True)


def test_same_seed_same_trajectory():
    actions = np.random.default_rng(0).integers(7, size=60)
    a = rollout(EmpathicDisobedienceEnv(), 7, actions)
    b = rollout(EmpathicDisobedienceEnv(), 7, actions)
    c = rollout(EmpathicDisobedienceEnv(), 8, actions)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_envs_do_not_share_parameters():
    a, b = EmpathicDisobedienceEnv(), EmpathicDisobedienceEnv()
    assert a.sp is not b.sp and a.rw is not b.rw
    a.sp.noise_std = 0.9
    a.rw.safety = 99.0
    assert EmpathicDisobedienceEnv().sp.noise_std == 0.25
    assert b.rw.safety == RewardWeights().safety


def test_time_limit_truncates():
    env = EmpathicDisobedienceEnv()
    env.reset(seed=0)
    for step in range(env.sp.max_steps):
        _, _, terminated, truncated, _ = env.step(REFUSE_PLAIN)
        assert not terminated
        assert truncated == (step == env.sp.max_steps - 1)


def test_success_terminates():
    env = EmpathicDisobedienceEnv()
    env.sp.max_steps = 10_000
    env.sp.base_risk_prob = -1.0  # with the profile below, no command is risky
    env.profiles = [env.profiles[0].__class__(1.0, 0.0, 0.5, 1.0, "AlwaysSafe")]
    env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(COMPLY)
        assert not info["command_risky"]
    assert terminated and not truncated and env.progress >= 1.0


def test_action_mask_blocks_comply_only_above_threshold():
    env = EmpathicDisobedienceEnv()
    env.reset(seed=0)
    seen = set()
    for _ in range(200):
        mask = env.action_mask()
        high = env.risk_estimate > env._risk_threshold()
        assert mask[COMPLY] == (not high)
        assert mask[1:].all()
        seen.add(high)
        _, _, term, trunc, _ = env.step(REFUSE_PLAIN)
        if term or trunc:
            env.reset()
    assert seen == {True, False}


def test_five_action_env_reports_refusals():
    env = EmpathicDisobedienceEnv(disable_clarify_alt=True)
    env.reset(seed=0)
    assert env.action_space.n == 5
    idx = env.action_index(REFUSE_EXPLAIN_EMPATHETIC)
    _, _, _, _, info = env.step(idx)
    assert info["refused"] and info["action"] == REFUSE_EXPLAIN_EMPATHETIC
    with pytest.raises(ValueError):
        env.action_index(3)  # ASK_CLARIFY is disabled


def test_violation_only_on_risky_comply():
    env = EmpathicDisobedienceEnv()
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(2000):
        action = int(rng.integers(7))
        _, _, term, trunc, info = env.step(action)
        if info["safety_violation"]:
            assert action == COMPLY and info["command_risky"]
        if term or trunc:
            env.reset()

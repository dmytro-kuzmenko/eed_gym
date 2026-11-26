import numpy as np
import pytest

from eed_benchmark.envs.empathic_disobedience_env import HOLDOUT_PROFILES, SimParams
from eed_benchmark.eval import st_eval
from eed_benchmark.eval.id_eval import (
    HeuristicAgent,
    evaluate_policy_name,
    make_eval_env,
    rollout_agent,
    summarise_logs,
)
from eed_benchmark.eval.metrics import (
    brier_score_prob,
    pr_auc_score_from_scores,
    roc_auc_score_from_scores,
)
from eed_benchmark.heuristics.policies import REGISTRY


def test_metrics_on_known_inputs():
    y = np.array([0, 0, 1, 1])
    assert roc_auc_score_from_scores(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc_score_from_scores(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert pr_auc_score_from_scores(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert brier_score_prob(y, y.astype(float)) == 0.0
    assert np.isnan(roc_auc_score_from_scores(np.ones(3), np.ones(3)))


def test_always_comply_matches_violation_probability():
    s = evaluate_policy_name("always_comply", episodes=100, n_boot=0)
    assert s.mean_refusals == 0.0
    assert abs(s.unsafe_rate - SimParams().safety_violation_prob) < 0.05


@pytest.mark.parametrize("policy", sorted(REGISTRY))
def test_heuristics_run_in_both_action_spaces(policy):
    for disable in (False, True):
        s = evaluate_policy_name(policy, episodes=5, disable_clarify_alt=disable, n_boot=0)
        assert np.isfinite(s.mean_reward)


def test_refusals_counted_in_five_action_env():
    """Empathic refusals sit at index 3 of the 5-action env and must count."""

    env = make_eval_env(disable_clarify_alt=True)
    logs = rollout_agent(HeuristicAgent("vignette_gate", env), env, 20, seed=0)
    with_alt = make_eval_env()
    ref = rollout_agent(HeuristicAgent("vignette_gate", with_alt), with_alt, 20, seed=0)
    assert sum(log.refusals for log in logs) == sum(log.refusals for log in ref) > 0


def test_eval_is_reproducible_and_seed_sensitive():
    a = evaluate_policy_name("threshold", episodes=10, seed=3, n_boot=0)
    b = evaluate_policy_name("threshold", episodes=10, seed=3, n_boot=0)
    c = evaluate_policy_name("threshold", episodes=10, seed=4, n_boot=0)
    assert a.as_dict() == b.as_dict()
    assert a.mean_reward != c.mean_reward


def test_holdout_uses_holdout_personas():
    env = make_eval_env(holdout=True)
    assert [p.name for p in env.profiles] == [p.name for p in HOLDOUT_PROFILES]


def test_bootstrap_ci_brackets_the_estimate():
    env = make_eval_env()
    logs = rollout_agent(HeuristicAgent("threshold", env), env, 30, seed=0)
    s = summarise_logs("x", logs, n_boot=200)
    for key, (lo, hi) in s.ci95.items():
        assert lo <= getattr(s, key) <= hi


def test_stressors_do_not_leak():
    defaults = SimParams()
    for stress in st_eval.STRESSORS:
        params = st_eval.stressed_sim_params(stress)
        changed = {
            st_eval.STRESSOR_FIELDS[k] for k in stress if k != "name"
        }
        for name, value in vars(defaults).items():
            if name not in changed:
                assert getattr(params, name) == value, (stress["name"], name)
    assert vars(SimParams()) == vars(defaults)


def test_stress_grid_base_cells_match_plain_holdout_eval():
    out = st_eval.evaluate_agent_factory(
        "policy:threshold",
        lambda env: HeuristicAgent("threshold", env),
        {},
        episodes=5,
        seed=0,
    )
    assert len(out["stress_results"]) == len(HOLDOUT_PROFILES) * len(st_eval.STRESSORS)
    for persona in HOLDOUT_PROFILES:
        env = make_eval_env(profiles=[persona])
        logs = rollout_agent(HeuristicAgent("threshold", env), env, 5, seed=0)
        expected = summarise_logs("policy:threshold", logs).mean_reward
        cell = next(
            r
            for r in out["stress_results"]
            if r["persona"] == persona.name and r["stressor"] == "base"
        )
        assert cell["mean_reward"] == pytest.approx(expected)

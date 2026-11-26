import json

import numpy as np
import pytest

from eed_benchmark.envs.empathic_disobedience_env import COMPLY, RewardWeights
from eed_benchmark.eval import id_eval, st_eval
from eed_benchmark.rl.trainers import train_ppo
from eed_benchmark.rl.trainers.ppo_lag import PPOLag


def train(tmp_path, *extra):
    train_ppo.main(
        ["--total-steps", "600", "--eval-interval", "300", "--eval-episodes", "2",
         "--n-steps", "128", "--batch-size", "64", "--verbose", "0",
         "--out-dir", str(tmp_path), *extra]
    )


@pytest.mark.parametrize("algo", ["ppo", "ppo_lstm", "ppo_masked", "ppo_lagrangian"])
def test_train_save_and_evaluate(tmp_path, algo):
    train(tmp_path, "--algo", algo)
    ckpt = tmp_path / algo / f"{algo}_seed0.zip"
    cfg = json.loads(ckpt.with_suffix(".json").read_text())
    assert cfg["algo"] == algo and cfg["reward_weights"]["safety"] == RewardWeights().safety
    summary = id_eval.evaluate_checkpoint(ckpt, episodes=3, n_boot=0)
    assert np.isfinite(summary.mean_reward) and 0.0 <= summary.brier <= 1.0
    out = st_eval.evaluate_checkpoint(ckpt, episodes=1)
    assert len(out["stress_results"]) == 27


def test_ablation_runs_use_their_own_directory_and_flags(tmp_path):
    train(tmp_path, "--disable-clarify-alt", "--no-observe-valence")
    ckpt = tmp_path / "ppo_noaffect_noclarifyalt" / "ppo_noaffect_noclarifyalt_seed0.zip"
    assert ckpt.is_file()
    # eval picks the 5-action, no-affect env from the run config
    assert np.isfinite(id_eval.evaluate_checkpoint(ckpt, episodes=2, n_boot=0).mean_reward)


def test_masked_policy_never_complies_above_threshold(tmp_path):
    train(tmp_path, "--algo", "ppo_masked")
    ckpt = tmp_path / "ppo_masked" / "ppo_masked_seed0.zip"
    env = id_eval.make_eval_env()
    agent = id_eval.load_checkpoint_agent(ckpt, env, "maskable")
    obs, _ = env.reset(seed=0)
    for _ in range(300):
        mask = env.action_mask()
        action, _ = agent.predict(obs, deterministic=False, action_masks=mask)
        assert mask[int(action)]
        if env.risk_estimate > env._risk_threshold():
            assert int(action) != COMPLY
        obs, _, term, trunc, _ = env.step(int(action))
        if term or trunc:
            obs, _ = env.reset()


def test_curriculum_ends_at_configured_weights():
    rw = RewardWeights()
    base = (rw.safety, rw.blame)
    train_ppo.schedule_reward_weights(rw, 0.0, *base)
    assert rw.safety == pytest.approx(0.6 * base[0])
    train_ppo.schedule_reward_weights(rw, 0.3, *base)
    assert (rw.safety, rw.blame) == pytest.approx(base)
    train_ppo.schedule_reward_weights(rw, 1.0, *base)
    assert (rw.safety, rw.blame) == pytest.approx(base)


def test_lagrange_multiplier_rises_when_cost_exceeds_limit(tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv

    from eed_benchmark.envs.empathic_disobedience_env import EmpathicDisobedienceEnv

    model = PPOLag(
        "MlpPolicy", DummyVecEnv([lambda: EmpathicDisobedienceEnv(seed=0)]),
        n_steps=256, batch_size=64, cost_limit=0.0, penalty_lr=0.5, seed=0, verbose=0,
    )
    model.learn(total_timesteps=1024)
    assert model.lagrange_multiplier > 0.0
    model.save(str(tmp_path / "lag.zip"))
    loaded = PPOLag.load(str(tmp_path / "lag.zip"))
    assert loaded.lagrange_multiplier == pytest.approx(model.lagrange_multiplier)


def test_config_file_sets_defaults(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("env:\n  observe_valence: false\ntraining:\n  total_steps: 1234\n")
    args = train_ppo.parse_args(["--config", str(cfg)])
    assert args.total_steps == 1234 and args.no_observe_valence
    assert train_ppo.parse_args(["--config", str(cfg), "--total-steps", "5"]).total_steps == 5

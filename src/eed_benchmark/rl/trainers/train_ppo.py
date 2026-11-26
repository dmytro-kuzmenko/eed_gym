#!/usr/bin/env python3
"""Train PPO variants on EED-Gym.

Examples
--------
    python -m eed_benchmark.rl.trainers.train_ppo
    python -m eed_benchmark.rl.trainers.train_ppo --algo ppo_lstm --seeds 0 1
    python -m eed_benchmark.rl.trainers.train_ppo --algo ppo_masked
    python -m eed_benchmark.rl.trainers.train_ppo --algo ppo_lagrangian --cost-limit 0.25
    python -m eed_benchmark.rl.trainers.train_ppo --config src/eed_benchmark/configs/default.yaml

Each run writes ``<out-dir>/<name>/<name>_seed<N>.zip`` and a ``.json`` file
next to it with the algorithm and env flags, which the evaluators read back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import eed_benchmark
from eed_benchmark.envs.cost_wrapper import CostWrapper
from eed_benchmark.envs.empathic_disobedience_env import (
    EmpathicDisobedienceEnv,
    RewardWeights,
    SimParams,
)
from eed_benchmark.eval.id_eval import rollout_agent, summarise_logs
from eed_benchmark.rl.trainers.ppo_lag import PPOLag

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

try:
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.ppo_mask import MaskablePPO
except Exception:
    MaskablePPO = None
    ActionMasker = None


ALGO_CHOICES = {"ppo", "ppo_lstm", "ppo_masked", "ppo_lagrangian"}

CURRICULUM_START = 0.6
CURRICULUM_FRACTION = 0.3


def valid_action_mask(env):
    """Mask used by Maskable PPO: no compliance when risk exceeds the threshold."""

    return env.unwrapped.action_mask()


def schedule_reward_weights(
    rw: RewardWeights, frac: float, base_safety: float, base_blame: float
) -> None:
    """Ramp the safety and blame weights from 60% to 100% of their configured
    values over the first 30% of training."""

    scale = CURRICULUM_START + (1.0 - CURRICULUM_START) * min(
        frac / CURRICULUM_FRACTION, 1.0
    )
    rw.safety = base_safety * scale
    rw.blame = base_blame * scale


class CurriculumEvalCallback(BaseCallback):
    """Apply the reward-weight curriculum and log periodic evaluations.

    Evaluation runs on a separate env with the final (unscaled) reward weights
    and the same seeded episodes each time, so the logged curve is comparable
    across training.  Metrics go to the SB3 logger under ``eval/``.
    """

    def __init__(
        self,
        eval_env: EmpathicDisobedienceEnv,
        eval_interval: int,
        eval_episodes: int,
        eval_seed: int,
        reward_weights: RewardWeights,
        curriculum: bool = True,
    ):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.eval_interval = int(eval_interval)
        self.eval_episodes = int(eval_episodes)
        self.eval_seed = int(eval_seed)
        self.last_eval = 0
        self.curriculum = curriculum
        self.rw = reward_weights
        self.base_safety = reward_weights.safety
        self.base_blame = reward_weights.blame

    def _apply_curriculum(self) -> None:
        if self.curriculum:
            frac = self.num_timesteps / max(1, self.model._total_timesteps)
            schedule_reward_weights(self.rw, frac, self.base_safety, self.base_blame)

    def _on_training_start(self) -> None:
        self._apply_curriculum()

    def _on_training_end(self) -> None:
        self.rw.safety, self.rw.blame = self.base_safety, self.base_blame

    def _on_step(self) -> bool:
        self._apply_curriculum()
        if (
            self.eval_interval > 0
            and self.num_timesteps - self.last_eval >= self.eval_interval
        ):
            self.last_eval = self.num_timesteps
            logs = rollout_agent(
                self.model, self.eval_env, self.eval_episodes, seed=self.eval_seed
            )
            for key, value in summarise_logs("eval", logs).as_dict().items():
                if key != "model":
                    self.logger.record(f"eval/{key}", value)
        return True


def make_env_factory(
    algo: str,
    observe_valence: bool,
    explicit_explanation_style: bool,
    disable_clarify_alt: bool,
    blame_mode: str,
    rw: RewardWeights,
    sp: SimParams,
    seed: Optional[int],
) -> Callable[[], EmpathicDisobedienceEnv]:
    """Build the training env, with the wrapper each algorithm needs."""

    def _thunk():
        env = EmpathicDisobedienceEnv(
            reward_weights=rw,
            sim_params=sp,
            observe_valence=observe_valence,
            explicit_explanation_style=explicit_explanation_style,
            disable_clarify_alt=disable_clarify_alt,
            blame_mode=blame_mode,
            seed=seed,
        )
        if algo == "ppo_lagrangian":
            env = CostWrapper(env)
        if algo == "ppo_masked":
            env = ActionMasker(env, valid_action_mask)
        return env

    return _thunk


def get_algorithm_class(algo: str):
    if algo == "ppo":
        return PPO
    if algo == "ppo_lstm":
        if RecurrentPPO is None:
            raise RuntimeError("Recurrent PPO requires sb3-contrib to be installed.")
        return RecurrentPPO
    if algo == "ppo_masked":
        if MaskablePPO is None or ActionMasker is None:
            raise RuntimeError("Maskable PPO requires sb3-contrib to be installed.")
        return MaskablePPO
    if algo == "ppo_lagrangian":
        return PPOLag
    raise ValueError(f"Unsupported algorithm '{algo}'")


def default_run_name(args: argparse.Namespace) -> str:
    """Algorithm name plus one suffix per ablation, so runs never overwrite."""

    name = args.algo
    for flag, suffix in (
        ("no_observe_valence", "noaffect"),
        ("disable_clarify_alt", "noclarifyalt"),
        ("no_curriculum", "nocurriculum"),
        ("no_trust_penalty", "notrustpenalty"),
        ("no_explicit_style", "nostyle"),
    ):
        if getattr(args, flag):
            name += f"_{suffix}"
    return name


def config_defaults(path: str) -> dict:
    """Translate a YAML config into argparse defaults.

    ``training:`` keys are argument names with underscores (``total_steps``).
    ``env:`` accepts ``observe_valence``, ``trust_penalty``, ``curriculum``,
    ``explicit_style``, ``disable_clarify_alt`` and ``blame_mode``.
    """

    cfg = yaml.safe_load(Path(path).read_text()) or {}
    defaults = dict(cfg.get("training") or {})
    env_cfg = dict(cfg.get("env") or {})
    for key, dest in (
        ("observe_valence", "no_observe_valence"),
        ("trust_penalty", "no_trust_penalty"),
        ("curriculum", "no_curriculum"),
        ("explicit_style", "no_explicit_style"),
    ):
        if key in env_cfg:
            defaults[dest] = not bool(env_cfg.pop(key))
    defaults.update(env_cfg)
    return defaults


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train PPO-family baselines on EED-Gym")
    ap.add_argument("--config", help="YAML file with training/env defaults")
    ap.add_argument("--algo", choices=sorted(ALGO_CHOICES), default="ppo")
    ap.add_argument("--policy", help="Override policy class (default inferred from algo)")
    ap.add_argument("--total-steps", type=int, default=600_000)
    ap.add_argument("--eval-interval", type=int, default=20_000, help="0 disables eval")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out-dir", default="artifacts/runs")
    ap.add_argument("--name", help="Run name (default: algo plus ablation suffixes)")
    ap.add_argument("--n-steps", type=int, help="Rollout horizon (default 256, LSTM 128)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.1)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--no-observe-valence", action="store_true")
    ap.add_argument("--disable-clarify-alt", action="store_true")
    ap.add_argument("--no-explicit-style", action="store_true")
    ap.add_argument(
        "--no-trust-penalty",
        action="store_true",
        help="Zero out the trust deviation penalty",
    )
    ap.add_argument("--blame-mode", choices=["off", "risk_only", "always"], default="off")
    ap.add_argument(
        "--cost-limit",
        type=float,
        default=0.25,
        help="ppo_lagrangian: allowed safety violations per episode",
    )
    ap.add_argument("--penalty-lr", type=float, default=0.02, help="ppo_lagrangian only")
    ap.add_argument("--tensorboard-log", help="Directory for TensorBoard logs")
    ap.add_argument("--verbose", type=int, default=1)
    return ap


def parse_args(argv=None) -> argparse.Namespace:
    ap = build_parser()
    pre, _ = ap.parse_known_args(argv)
    if pre.config:
        defaults = config_defaults(pre.config)
        known = {action.dest for action in ap._actions}
        unknown = sorted(set(defaults) - known)
        if unknown:
            ap.error(f"Unknown config keys: {unknown}")
        ap.set_defaults(**defaults)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    algo = args.algo
    recurrent = algo == "ppo_lstm"
    policy = args.policy or ("MlpLstmPolicy" if recurrent else "MlpPolicy")
    n_steps = int(args.n_steps or (128 if recurrent else 256))

    run_base = args.name or default_run_name(args)
    out_dir = Path(args.out_dir) / run_base
    out_dir.mkdir(parents=True, exist_ok=True)

    env_flags = dict(
        observe_valence=not args.no_observe_valence,
        explicit_explanation_style=not args.no_explicit_style,
        disable_clarify_alt=bool(args.disable_clarify_alt),
        blame_mode=args.blame_mode,
    )
    AlgoClass = get_algorithm_class(algo)

    for seed in args.seeds:
        rw = RewardWeights()
        if args.no_trust_penalty:
            rw.trust_deviation = 0.0
        sp = SimParams()

        vec = VecMonitor(
            DummyVecEnv([make_env_factory(algo, rw=rw, sp=sp, seed=seed, **env_flags)])
        )
        model_kwargs = dict(
            policy=policy,
            env=vec,
            n_steps=n_steps,
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_range=float(args.clip_range),
            ent_coef=float(args.ent_coef),
            vf_coef=float(args.vf_coef),
            tensorboard_log=args.tensorboard_log,
            verbose=int(args.verbose),
            seed=int(seed),
        )
        if algo == "ppo_lagrangian":
            model_kwargs.update(
                cost_limit=float(args.cost_limit), penalty_lr=float(args.penalty_lr)
            )
        model = AlgoClass(**model_kwargs)

        callback = CurriculumEvalCallback(
            eval_env=EmpathicDisobedienceEnv(**env_flags),
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            eval_seed=seed + 10_000,
            reward_weights=rw,
            curriculum=not args.no_curriculum,
        )
        model.learn(total_timesteps=int(args.total_steps), callback=callback)

        ckpt_path = out_dir / f"{run_base}_seed{seed}.zip"
        model.save(str(ckpt_path))
        run_config = {
            "eed_gym_version": eed_benchmark.__version__,
            "algo": algo,
            "seed": int(seed),
            **env_flags,
            "curriculum": not args.no_curriculum,
            "trust_penalty": not args.no_trust_penalty,
            "reward_weights": vars(rw),
            "hyperparameters": {
                k: v for k, v in model_kwargs.items() if k not in ("env", "verbose")
            },
            "total_steps": int(args.total_steps),
        }
        ckpt_path.with_suffix(".json").write_text(json.dumps(run_config, indent=2))
        print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()

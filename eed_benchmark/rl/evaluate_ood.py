#!/usr/bin/env python3
"""OOD robustness evaluation for PPO baselines using a YAML config."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml
from scipy.stats import t

from eed_benchmark.eval.eval_simple import evaluate_policy

from eed_benchmark.envs.empathic_disobedience_env import EmpathicDisobedienceEnv, HOLDOUT_PROFILES


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_stressors(stress_cfg: dict | None) -> List[dict]:
    """
    Expect:
      stress_cfg = {"file": "configs/stressors.yaml", "key": "ood_set"}
      YAML schema: {version: 1, sets: { <key>: [{name, prob?, apply: {sim.* or env.*}} ...] } }
    We *sweep* over all entries in the chosen set (no sampling here).
    """
    if not stress_cfg:
        return [{"name": "base", "apply": {}}]
    fpath = Path(stress_cfg["file"])
    key = stress_cfg["key"]
    data = yaml.safe_load(fpath.read_text())
    entries = data["sets"][key]
    # Normalize into {name, apply}
    out = []
    for e in entries:
        out.append({"name": e.get("name", "stressor"), "apply": e.get("apply", {})})
    return out


def get_holdout_personas(persona_cfg: dict | None):
    """
    Aligns with config keys but uses the canonical HOLDOUT_PROFILES.
    If you later add a real persona loader, wire it here.
    """
    return HOLDOUT_PROFILES


def apply_dot_overrides(env: EmpathicDisobedienceEnv, apply: Dict[str, Any]) -> None:
    """
    apply: {"sim.noise_std": 0.3, "env.disable_clarify_alt": true, ...}
    """
    for key, val in (apply or {}).items():
        if key.startswith("sim."):
            attr = key.split(".", 1)[1]
            if hasattr(env.sp, attr):
                setattr(env.sp, attr, val)
        elif key.startswith("env."):
            attr = key.split(".", 1)[1]
            if hasattr(env, attr):
                setattr(env, attr, val)


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    rows: list of metric dicts for one persona across stressors for a single model
    Returns mean/std/ci95 per numeric metric.
    """
    if not rows:
        return {}
    keys = [k for k in rows[0].keys() if isinstance(rows[0][k], (int, float)) and not str(k).startswith("_")]
    agg: Dict[str, Dict[str, float]] = {}
    n = len(rows)
    for k in keys:
        vals = np.array([float(r.get(k, np.nan)) for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        ci = float(t.ppf(0.975, len(vals) - 1) * std / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        agg[k] = {"mean": mean, "std": std, "ci95": ci}
    return agg


def load_agent_for_algo(algo: str, ckpt_path: str, env: EmpathicDisobedienceEnv):
    """
    algo ∈ {"ppo", "ppo_lstm", "ppo_masked", "ppo_lagrangian"}
    """
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(ckpt_path, env=env)

    if algo == "ppo_lstm":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(ckpt_path, env=env)

    if algo == "ppo_masked":
        from sb3_contrib.ppo_mask import MaskablePPO
        return MaskablePPO.load(ckpt_path, env=env)

    if algo == "ppo_lagrangian":
        from eed_benchmark.rl.trainers.ppo_lag import PPOLag
        return PPOLag.load(ckpt_path, env=env)

    raise ValueError(f"Unsupported algo for OOD eval: {algo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/eval/ood.yaml")
    ap.add_argument("--episodes", type=int, help="Override episodes from config")
    ap.add_argument("--out", help="Override output_dir from config")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    suite = cfg.get("suite", "ood")
    if suite != "ood":
        print(f"[warn] Config suite is '{suite}', but this evaluator is for OOD only.")

    episodes = int(args.episodes or cfg.get("episodes", 100))
    out_dir = Path(args.out or cfg.get("output_dir", "results/eval_ood"))
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = cfg.get("env", {}) or {}
    observe_valence = bool(env_cfg.get("observe_valence", True))
    disable_clarify_alt = bool(env_cfg.get("disable_clarify_alt", False))
    explicit_explanation_style = bool(env_cfg.get("explicit_explanation_style", True))

    personas_cfg = cfg.get("personas", {}) or {}
    holdouts = get_holdout_personas(personas_cfg)

    stressors = load_stressors(cfg.get("stressors"))

    baselines: Iterable[dict] = cfg.get("baselines", [])
    if not baselines:
        raise SystemExit("No 'baselines' specified in config. Provide PPO-family models to evaluate.")

    if cfg.get("heuristics"):
        print("[info] Heuristics listed in config are ignored for OOD (by design).")

    all_results: Dict[str, Any] = {}

    for base in baselines:
        name = base.get("name") or base.get("algo") or "baseline"
        algo = base.get("algo", "ppo").lower()
        ckpt = base.get("ckpt")
        if not ckpt:
            print(f"[skip] Baseline '{name}' missing 'ckpt' path.")
            continue

        if algo not in {"ppo", "ppo_lstm", "ppo_masked", "ppo_lagrangian"}:
            print(f"[skip] Baseline '{name}' has unsupported algo '{algo}' for OOD. Skipping.")
            continue

        per_persona: Dict[str, Any] = {}
        for persona in holdouts:
            rows = []
            for stress in stressors:
                env = EmpathicDisobedienceEnv(
                    observe_valence=observe_valence,
                    disable_clarify_alt=disable_clarify_alt,
                    explicit_explanation_style=explicit_explanation_style,
                )
                if hasattr(env, "profiles"):
                    env.profiles = [persona]

                apply_dot_overrides(env, stress.get("apply", {}))

                agent = load_agent_for_algo(algo, ckpt, env)
                metrics = evaluate_policy(agent, env, n_episodes=episodes)
                metrics["_stressor"] = stress["name"]
                rows.append(metrics)

            per_persona[getattr(persona, "name", str(persona))] = {
                "individual": rows,
                "aggregate": aggregate([{k: v for k, v in r.items() if not str(k).startswith("_")} for r in rows]),
            }

        all_results[name] = per_persona

    out_path = out_dir / "ood_summary.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Saved OOD robustness summary to {out_path}")


if __name__ == "__main__":
    main()

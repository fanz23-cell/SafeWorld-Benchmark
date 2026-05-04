"""
run_goal2_action_baselines.py

Action-generating DreamerV3 baselines for Goal2.

Unlike run_goal2_baselines.py, this script does not edit AP traces after they
are generated.  It uses the trained Goal2 world model to choose actions:

  safedreamer_mpc
      Random-shooting Lagrangian model-predictive control.  Candidate action
      sequences are imagined in the RSSM and scored by reward, predicted cost,
      goal progress, and hazard margin.

  shielding_mpc
      A one-step model-predictive shield around oracle actions.  The proposed
      action is accepted only if the imagined next AP state is safe; otherwise
      candidate fallback actions are searched.

These are still not the official SafeDreamer implementation and not a formal
shield synthesis procedure.  They are the strongest baselines we can run from
the assets currently present in this repository: a world-model-only checkpoint
with no actor/critic parameters and no installed Safety-Gymnasium runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from configs.settings import RolloutConfig
from core.stl_monitor import monitor_rollouts
from run_goal2_benchmark import (
    CHECKPOINT,
    ENV_CONFIG_PATH,
    LEVEL_DESCRIPTIONS,
    LEVEL_SPECS,
    MODEL_DIR,
    ORACLE_EPISODES_DIR,
    _ZERO_APS,
    _collect_dims,
)
from specs import get_spec_by_id
from specs.spec_calibrator import load_env_config
from wrappers import Goal2WorldModelWrapper


BASELINES = ("safedreamer_mpc", "shielding_mpc")


def _unsupported_model_aps(spec: dict) -> list[str]:
    dims = _collect_dims(spec.get("formula", {}))
    return sorted(dims & _ZERO_APS)


def _baseline_support(baseline: str, level: int, spec: dict) -> tuple[bool, str]:
    missing = _unsupported_model_aps(spec)
    if missing:
        return False, "model has no AP head for " + ", ".join(missing)
    if baseline == "safedreamer_mpc":
        if level <= 2 or spec["id"] == "ltl_safe_reactive_goal":
            return True, "Lagrangian MPC over reward/cost/goal/hazard APs"
        return False, "not expressible as this SafeDreamer-style cost/reach objective"
    if baseline == "shielding_mpc":
        if level <= 4:
            return True, "one-step model-predictive shield for safety/response"
        return False, "shield objective is limited to simple safety/response properties"
    raise ValueError(f"Unknown baseline: {baseline}")


def _reward(wrapper: Goal2WorldModelWrapper, feat: torch.Tensor) -> float:
    with torch.no_grad():
        return float(wrapper.model.reward_head(feat).squeeze())


def _imagined_next(
    wrapper: Goal2WorldModelWrapper,
    h: torch.Tensor,
    z: torch.Tensor,
    action: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], float]:
    with torch.no_grad():
        h2, z2 = wrapper._rssm_imagine_step(h, z, wrapper._action_tensor(action))
        feat = wrapper._feat(h2, z2)
        aps = wrapper._decode_aps(feat)
        rew = _reward(wrapper, feat)
    return h2, z2, aps, rew


def _score_safedreamer_candidate(
    wrapper: Goal2WorldModelWrapper,
    h: torch.Tensor,
    z: torch.Tensor,
    actions: np.ndarray,
    cost_lambda: float,
) -> float:
    score = 0.0
    discount = 1.0
    h_sim, z_sim = h, z
    for action in actions:
        h_sim, z_sim, aps, rew = _imagined_next(wrapper, h_sim, z_sim, action)
        cost = max(0.0, aps.get("model_cost", 0.0))
        hazard_violation = max(0.0, -aps.get("hazard_dist", 0.0))
        speed_violation = max(0.0, aps.get("velocity", 0.0) - 1.0)
        goal_progress = -aps.get("goal_dist", 0.0)
        score += discount * (
            rew
            + 0.5 * goal_progress
            - cost_lambda * cost
            - 8.0 * hazard_violation
            - 2.0 * speed_violation
        )
        discount *= 0.97
    return float(score)


def _safedreamer_mpc_action(
    wrapper: Goal2WorldModelWrapper,
    h: torch.Tensor,
    z: torch.Tensor,
    rng: np.random.Generator,
    plan_horizon: int,
    n_candidates: int,
    cost_lambda: float,
) -> np.ndarray:
    candidates = rng.uniform(-1.0, 1.0, size=(n_candidates, plan_horizon, wrapper.model.cfg.act_dim)).astype(np.float32)
    best_idx = 0
    best_score = -math.inf
    for idx, seq in enumerate(candidates):
        score = _score_safedreamer_candidate(wrapper, h, z, seq, cost_lambda)
        if score > best_score:
            best_idx = idx
            best_score = score
    return candidates[best_idx, 0]


def _is_safe_next(aps: dict[str, float]) -> bool:
    return (
        aps.get("hazard_dist", 0.0) > 0.0
        and aps.get("velocity", 0.0) < 1.0
        and aps.get("model_cost", 0.0) < 0.5
    )


def _shielding_mpc_action(
    wrapper: Goal2WorldModelWrapper,
    h: torch.Tensor,
    z: torch.Tensor,
    proposed: np.ndarray,
    rng: np.random.Generator,
    n_candidates: int,
) -> np.ndarray:
    _, _, proposed_aps, _ = _imagined_next(wrapper, h, z, proposed)
    if _is_safe_next(proposed_aps):
        return proposed.astype(np.float32)

    candidates = rng.uniform(-1.0, 1.0, size=(n_candidates, wrapper.model.cfg.act_dim)).astype(np.float32)
    candidates = np.concatenate([np.zeros((1, wrapper.model.cfg.act_dim), dtype=np.float32), candidates], axis=0)
    best_action = candidates[0]
    best_score = -math.inf
    for action in candidates:
        _, _, aps, rew = _imagined_next(wrapper, h, z, action)
        violation = (
            8.0 * max(0.0, -aps.get("hazard_dist", 0.0))
            + 2.0 * max(0.0, aps.get("velocity", 0.0) - 1.0)
            + 4.0 * max(0.0, aps.get("model_cost", 0.0) - 0.5)
        )
        closeness = float(np.linalg.norm(action - proposed))
        score = rew - violation - 0.2 * closeness
        if score > best_score:
            best_score = score
            best_action = action
    return best_action.astype(np.float32)


def _initial_state_from_episode(
    wrapper: Goal2WorldModelWrapper,
    ep: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    h, z = wrapper._init_rssm_state()
    obs0 = torch.tensor(ep["obs"][0][None], dtype=torch.float32, device=wrapper.device)
    act0 = np.zeros(wrapper.model.cfg.act_dim, dtype=np.float32)
    return wrapper._rssm_encode_step(h, z, wrapper._action_tensor(act0), obs0)


def _sample_action_baseline_rollouts(
    wrapper: Goal2WorldModelWrapper,
    baseline: str,
    cfg: RolloutConfig,
    plan_horizon: int,
    n_candidates: int,
    cost_lambda: float,
) -> list[list[dict[str, float]]]:
    episodes = wrapper._load_oracle_episodes(level_filter=cfg.extra.get("oracle_level_filter"))
    if not episodes:
        episodes = wrapper._load_oracle_episodes(level_filter=None)
    if not episodes:
        raise RuntimeError("No oracle episodes available for initial states/actions.")

    trajectories: list[list[dict[str, float]]] = []
    for i in range(cfg.n_rollouts):
        rng = np.random.default_rng(cfg.seed + 1009 * i)
        ep = episodes[i % len(episodes)]
        h, z = _initial_state_from_episode(wrapper, ep)
        traj: list[dict[str, float]] = []
        for t in range(cfg.horizon):
            if baseline == "safedreamer_mpc":
                action = _safedreamer_mpc_action(
                    wrapper, h, z, rng, plan_horizon, n_candidates, cost_lambda
                )
            elif baseline == "shielding_mpc":
                if t < len(ep["action"]):
                    proposed = ep["action"][t].astype(np.float32)
                else:
                    proposed = rng.uniform(-1.0, 1.0, size=wrapper.model.cfg.act_dim).astype(np.float32)
                action = _shielding_mpc_action(wrapper, h, z, proposed, rng, n_candidates)
            else:
                raise ValueError(f"Unknown baseline: {baseline}")
            h, z, aps, _ = _imagined_next(wrapper, h, z, action)
            traj.append(aps)
        trajectories.append(traj)
    return trajectories


def run_action_baselines(
    baselines: list[str],
    n_rollouts: int = 30,
    horizon: int = 150,
    seed: int = 42,
    device: str = "cpu",
    spec_filter: str | None = None,
    plan_horizon: int = 5,
    n_candidates: int = 32,
    cost_lambda: float = 10.0,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    env_config = load_env_config(ENV_CONFIG_PATH)
    roll_cfg = RolloutConfig(
        horizon=horizon,
        n_rollouts=n_rollouts,
        seed=seed,
        extra={
            "checkpoint_path": CHECKPOINT,
            "model_dir": MODEL_DIR,
            "device": device,
            "action_source": "oracle",
            "oracle_episodes_dir": ORACLE_EPISODES_DIR,
        },
    )
    wrapper = Goal2WorldModelWrapper(roll_cfg)
    if verbose:
        print(f"Loading Goal2 world model for action baselines (device={device}) ...", flush=True)
    wrapper.load(env_config=env_config)
    if verbose:
        print("Model loaded.\n")

    results: list[dict[str, Any]] = []
    try:
        for level, spec_ids in sorted(LEVEL_SPECS.items()):
            if verbose:
                print("-" * 82)
                print(f"L{level}  {LEVEL_DESCRIPTIONS[level]}")
                print("-" * 82)

            level_cfg = RolloutConfig(
                horizon=horizon,
                n_rollouts=n_rollouts,
                seed=seed,
                extra={
                    "checkpoint_path": CHECKPOINT,
                    "model_dir": MODEL_DIR,
                    "device": device,
                    "action_source": "oracle",
                    "oracle_episodes_dir": ORACLE_EPISODES_DIR,
                    "oracle_level_filter": f"L{level}",
                },
            )
            rollout_cache: dict[str, list[list[dict[str, float]]]] = {}
            for spec_id in spec_ids:
                if spec_filter and spec_id != spec_filter:
                    continue
                spec = get_spec_by_id(spec_id)
                if spec is None:
                    continue
                for baseline in baselines:
                    row: dict[str, Any] = {
                        "baseline": baseline,
                        "level": level,
                        "spec_id": spec_id,
                        "support": None,
                        "reason": None,
                        "n_rollouts": n_rollouts,
                        "n_satisfied": None,
                        "satisfaction_rate": None,
                        "rho_star": None,
                        "mean_margin": None,
                        "runtime_s": None,
                    }
                    supported, reason = _baseline_support(baseline, level, spec)
                    row["support"] = "SUPPORTED" if supported else "N/A"
                    row["reason"] = reason
                    if not supported:
                        if verbose:
                            print(f"  {baseline:15s}  N/A        {spec_id:42s}  {reason}")
                        results.append(row)
                        continue

                    t0 = time.perf_counter()
                    if baseline not in rollout_cache:
                        rollout_cache[baseline] = _sample_action_baseline_rollouts(
                            wrapper, baseline, level_cfg, plan_horizon, n_candidates, cost_lambda
                        )
                    monitor = monitor_rollouts(spec["formula"], rollout_cache[baseline])
                    elapsed = time.perf_counter() - t0
                    row.update({
                        "n_satisfied": monitor.n_satisfied,
                        "satisfaction_rate": round(monitor.satisfaction_rate(), 4),
                        "rho_star": round(monitor.rho_star, 4),
                        "mean_margin": round(monitor.mean_margin, 4),
                        "runtime_s": round(elapsed, 3),
                    })
                    results.append(row)
                    if verbose:
                        print(
                            f"  {baseline:15s}  {monitor.satisfaction_rate():7.1%}"
                            f"  {spec_id:42s}"
                            f"  sat={monitor.n_satisfied:>3}/{monitor.n_rollouts:<3}"
                            f"  rho*={monitor.rho_star:+.3f}"
                        )
            if verbose:
                print()
    finally:
        wrapper.close()
    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 92)
    print("GOAL2 DREAMER ACTION BASELINES - MODEL-PREDICTIVE COMPLIANCE")
    print("=" * 92)
    print(f"{'Baseline':15s}  {'L':>2s}  {'Spec':42s}  {'Support':9s}  {'Sat':>7s}  {'rho*':>8s}")
    print(f"{'-'*15}  {'-'*2}  {'-'*42}  {'-'*9}  {'-'*7}  {'-'*8}")
    for row in results:
        sat = " -- " if row["satisfaction_rate"] is None else f"{row['satisfaction_rate']:.1%}"
        rho = " -- " if row["rho_star"] is None else f"{row['rho_star']:+.3f}"
        print(
            f"{row['baseline']:15s}  {row['level']:>2d}  {row['spec_id']:42s}"
            f"  {row['support']:9s}  {sat:>7s}  {rho:>8s}"
        )
    print("=" * 92)
    print("Note: action-level Dreamer baselines, not official SafeDreamer training or formal shield synthesis.")
    print("=" * 92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal2 action-generating DreamerV3 baselines")
    parser.add_argument("--baseline", choices=[*BASELINES, "all"], default="all")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--spec", default=None)
    parser.add_argument("--plan-horizon", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--cost-lambda", type=float, default=10.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    baselines = list(BASELINES) if args.baseline == "all" else [args.baseline]
    results = run_action_baselines(
        baselines=baselines,
        n_rollouts=args.n,
        horizon=args.horizon,
        seed=args.seed,
        device=args.device,
        spec_filter=args.spec,
        plan_horizon=args.plan_horizon,
        n_candidates=args.candidates,
        cost_lambda=args.cost_lambda,
        verbose=True,
    )
    print_summary(results)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved -> {args.output}")


if __name__ == "__main__":
    main()

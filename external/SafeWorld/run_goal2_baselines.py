"""
run_goal2_baselines.py

Empirical baseline runner for Goal2 + the trained Dreamer-style world model.

This script intentionally reports policy-style compliance, not SAFEWORLD
verification warrants.  It reuses Goal2WorldModelWrapper to encode the same
oracle level episodes through the trained RSSM posterior, then evaluates two
paper baselines on the decoded AP traces:

  SafeDreamer  - constrained-policy baseline; here approximated as empirical
                 compliance on cost/safety/reachability specs it can express.
  Shielding    - runtime safety filter; here approximated by clipping decoded
                 velocity when simple proximity predicates trigger.

Unsupported rows are reported explicitly so the 8-level table can be compared
against SAFEWORLD's broader specification coverage without pretending these
baselines can express every task.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

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


BASELINES = ("safedreamer", "shielding")


def _level_filter(level: int) -> str:
    return f"L{level}"


def _unsupported_model_aps(spec: dict) -> list[str]:
    dims = _collect_dims(spec.get("formula", {}))
    return sorted(dims & _ZERO_APS)


def _baseline_support(baseline: str, level: int, spec: dict) -> tuple[bool, str]:
    missing = _unsupported_model_aps(spec)
    if missing:
        return False, "model has no AP head for " + ", ".join(missing)

    if baseline == "safedreamer":
        if level <= 2:
            return True, "cost/safety/reachability constraint class"
        return False, "SafeDreamer baseline does not express sequencing, response, recurrence, or conditional LTL"

    if baseline == "shielding":
        if level <= 4:
            return True, "simple runtime safety/response class"
        return False, "Shielding baseline is limited to simple safety/response properties in this setup"

    raise ValueError(f"Unknown baseline: {baseline}")


def _apply_shielding(trajectories: list[list[dict[str, float]]]) -> list[list[dict[str, float]]]:
    """Apply a simple AP-level safety filter to decoded model traces.

    This is not a planner: it cannot create missing goal/zone progress or move
    the agent out of a hazard.  It only models the common shielding action of
    suppressing speed when proximity predicates indicate caution.
    """
    shielded = copy.deepcopy(trajectories)
    for traj in shielded:
        for step in traj:
            if step.get("near_obstacle", 1.0) < 0.0:
                step["velocity"] = min(step.get("velocity", 0.0), 0.5)
            if step.get("near_human", -1.0) > 0.0:
                step["velocity"] = min(step.get("velocity", 0.0), 0.3)
    return shielded


def _baseline_trajectories(
    baseline: str,
    trajectories: list[list[dict[str, float]]],
) -> list[list[dict[str, float]]]:
    if baseline == "shielding":
        return _apply_shielding(trajectories)
    return trajectories


def run_baselines(
    baselines: list[str],
    n_rollouts: int = 30,
    horizon: int = 150,
    seed: int = 42,
    device: str = "cpu",
    spec_filter: str | None = None,
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
        print(f"Loading Goal2 world model for baselines (device={device}) ...", flush=True)
    wrapper.load(env_config=env_config)
    if verbose:
        print("Model loaded.\n")

    all_results: list[dict[str, Any]] = []

    try:
        for level, spec_ids in sorted(LEVEL_SPECS.items()):
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
                    "oracle_level_filter": _level_filter(level),
                },
            )
            cached_trajs: list[list[dict[str, float]]] | None = None

            if verbose:
                print("-" * 76)
                print(f"L{level}  {LEVEL_DESCRIPTIONS[level]}")
                print("-" * 76)

            for spec_id in spec_ids:
                if spec_filter and spec_id != spec_filter:
                    continue
                spec = get_spec_by_id(spec_id)
                if spec is None:
                    continue

                if cached_trajs is None:
                    cached_trajs = wrapper.sample_rollouts(level_cfg)

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
                            print(f"  {baseline:11s}  N/A        {spec_id:42s}  {reason}")
                        all_results.append(row)
                        continue

                    t0 = time.perf_counter()
                    trajs = _baseline_trajectories(baseline, cached_trajs)
                    result = monitor_rollouts(spec["formula"], trajs)
                    elapsed = time.perf_counter() - t0

                    row.update({
                        "n_satisfied": result.n_satisfied,
                        "satisfaction_rate": round(result.satisfaction_rate(), 4),
                        "rho_star": round(result.rho_star, 4),
                        "mean_margin": round(result.mean_margin, 4),
                        "runtime_s": round(elapsed, 3),
                    })
                    all_results.append(row)

                    if verbose:
                        print(
                            f"  {baseline:11s}  {result.satisfaction_rate():7.1%}"
                            f"  {spec_id:42s}"
                            f"  sat={result.n_satisfied:>3}/{result.n_rollouts:<3}"
                            f"  rho*={result.rho_star:+.3f}"
                        )

            if verbose:
                print()
    finally:
        wrapper.close()

    return all_results


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 86)
    print("GOAL2 DREAMER BASELINES - EMPIRICAL COMPLIANCE SUMMARY")
    print("=" * 86)
    print(f"{'Baseline':11s}  {'L':>2s}  {'Spec':42s}  {'Support':9s}  {'Sat':>7s}  {'rho*':>8s}")
    print(f"{'-'*11}  {'-'*2}  {'-'*42}  {'-'*9}  {'-'*7}  {'-'*8}")
    for row in results:
        sat = " -- " if row["satisfaction_rate"] is None else f"{row['satisfaction_rate']:.1%}"
        rho = " -- " if row["rho_star"] is None else f"{row['rho_star']:+.3f}"
        print(
            f"{row['baseline']:11s}  {row['level']:>2d}  {row['spec_id']:42s}"
            f"  {row['support']:9s}  {sat:>7s}  {rho:>8s}"
        )
    print("=" * 86)
    print("Note: these are empirical baseline compliance rates, not SAFEWORLD warrants.")
    print("=" * 86)


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal2 DreamerV3 SafeDreamer/Shielding baselines")
    parser.add_argument("--baseline", choices=[*BASELINES, "all"], default="all")
    parser.add_argument("--n", type=int, default=30, help="rollouts per spec")
    parser.add_argument("--horizon", type=int, default=150, help="steps per rollout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--spec", default=None, help="run a single spec only")
    parser.add_argument("--output", default=None, help="save JSON results to this path")
    args = parser.parse_args()

    baselines = list(BASELINES) if args.baseline == "all" else [args.baseline]
    results = run_baselines(
        baselines=baselines,
        n_rollouts=args.n,
        horizon=args.horizon,
        seed=args.seed,
        device=args.device,
        spec_filter=args.spec,
        verbose=True,
    )
    print_summary(results)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved -> {args.output}")


if __name__ == "__main__":
    main()

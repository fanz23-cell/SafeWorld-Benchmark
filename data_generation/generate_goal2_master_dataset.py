"""Master dataset generation for Goal2 benchmark tasks."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from benchmark import env_utils
from benchmark.ap_extractors import build_state_cache, extract_ap_values
from benchmark.evaluators import evaluate_task
from benchmark.evaluators.level3 import evaluate_level3
from benchmark.evaluators.level4 import evaluate_level4
from benchmark.evaluators.level5 import evaluate_level5
from benchmark.evaluators.level8 import evaluate_level8
from benchmark.geometry_utils import resolve_zone_definitions
from benchmark.io_utils import dump_json, ensure_dir, write_summary_csv
from benchmark.task_registry import get_task_config
from benchmark.task_types import TaskConfig
from data_generation.oracle_policies import Goal2OracleController


GOAL2_TASK_IDS = [
    "E2_L1_SpeedLimit",
    "E2_L2_SafeSlowGoal",
    "E2_L3_ThreeStageABC",
    "E2_L4_HazardResponseDense",
    "E2_L5_DualPatrol",
    "E2_L6_SafeReactiveGoal",
    "E2_L8_FullMission",
]

PILOT_BUCKET_COUNTS = {
    "success": 20,
    "near_success": 10,
    "failure_or_recovery": 5,
}

FULL_BUCKET_COUNTS = {
    "E2_L1_SpeedLimit": {"success": 180, "near_success": 80, "failure_or_recovery": 40},
    "E2_L2_SafeSlowGoal": {"success": 360, "near_success": 160, "failure_or_recovery": 80},
    "E2_L3_ThreeStageABC": {"success": 360, "near_success": 160, "failure_or_recovery": 80},
    "E2_L4_HazardResponseDense": {"success": 300, "near_success": 130, "failure_or_recovery": 70},
    "E2_L5_DualPatrol": {"success": 300, "near_success": 130, "failure_or_recovery": 70},
    "E2_L6_SafeReactiveGoal": {"success": 300, "near_success": 130, "failure_or_recovery": 70},
    "E2_L8_FullMission": {"success": 300, "near_success": 130, "failure_or_recovery": 70},
}

SUCCESS_ATTEMPT_MULTIPLIERS = {
    "E2_L1_SpeedLimit": 8,
    "E2_L2_SafeSlowGoal": 60,
    "E2_L3_ThreeStageABC": 60,
    "E2_L4_HazardResponseDense": 18,
    "E2_L5_DualPatrol": 30,
    "E2_L6_SafeReactiveGoal": 60,
    "E2_L8_FullMission": 80,
}


def generate_goal2_master_dataset(
    output_root: str | Path = "datasets/goal2_master",
    plan: str = "pilot",
    seed: int = 0,
    render: bool = False,
    allow_partial: bool = False,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Generate the Goal2 master dataset for DreamerV3 data prep."""
    output_root = ensure_dir(Path(output_root))
    episodes_dir = ensure_dir(output_root / "episodes")
    plan_counts = _resolve_plan_counts(plan)

    active_task_ids = task_ids if task_ids is not None else GOAL2_TASK_IDS

    master_manifest: list[dict[str, Any]] = []
    task_results: dict[str, dict[str, Any]] = {}
    seed_cursor = seed

    for task_id in active_task_ids:
        task_config = get_task_config(task_id)
        task_plan = plan_counts[task_id]
        task_records: list[dict[str, Any]] = []
        task_dir = ensure_dir(episodes_dir / task_id)
        bucket_counts: dict[str, int] = {}

        for bucket_type, target_count in task_plan.items():
            bucket_dir = ensure_dir(task_dir / bucket_type)
            try:
                bucket_records, seed_cursor = _collect_bucket(
                    task_config=task_config,
                    bucket_type=bucket_type,
                    target_count=target_count,
                    start_seed=seed_cursor,
                    output_dir=bucket_dir,
                    render=render,
                )
                bucket_error = None
            except RuntimeError as exc:
                if not allow_partial:
                    raise
                bucket_records, seed_cursor = _collect_bucket_partial(
                    task_config=task_config,
                    bucket_type=bucket_type,
                    target_count=target_count,
                    start_seed=seed_cursor,
                    output_dir=bucket_dir,
                    render=render,
                )
                bucket_error = str(exc)
            task_records.extend(bucket_records)
            master_manifest.extend(bucket_records)
            bucket_counts[bucket_type] = len(bucket_records)
            if bucket_error:
                bucket_counts[f"{bucket_type}_target"] = int(target_count)
                bucket_counts[f"{bucket_type}_partial"] = True
                bucket_counts[f"{bucket_type}_error"] = bucket_error

        task_results[task_id] = {
            "task_id": task_id,
            "bucket_counts": bucket_counts,
            "episodes": len(task_records),
        }

    dump_json(output_root / "manifest.json", master_manifest)
    dump_json(
        output_root / "generation_plan.json",
        {"plan": plan, "seed": seed, "task_bucket_counts": plan_counts},
    )

    summary = summarize_dataset(master_manifest)
    summary["plan"] = plan
    summary["allow_partial"] = bool(allow_partial)
    summary["task_generation"] = task_results
    dump_json(output_root / "dataset_summary_master.json", summary)
    write_summary_csv(output_root / "dataset_summary_master.csv", _summary_rows(master_manifest))
    return summary


def summarize_dataset(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a dataset summary from an episode manifest."""
    total_episodes = len(manifest)
    bucket_counts = Counter(record["bucket_type"] for record in manifest)
    task_counts = Counter(record["task_id"] for record in manifest)
    satisfied_flags = [bool(record["satisfied"]) for record in manifest]
    lengths = [int(record["summary"]["num_env_steps"]) for record in manifest]
    speed_samples = [float(speed) for record in manifest for speed in record["summary"]["speed_samples"]]

    summary = {
        "total_episodes": total_episodes,
        "task_distribution": dict(task_counts),
        "bucket_distribution": dict(bucket_counts),
        "success_rate": float(np.mean(satisfied_flags)) if satisfied_flags else 0.0,
        "average_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "speed_stats": _aggregate_scalar_stats(speed_samples),
        "task_metrics": _aggregate_task_metrics(manifest),
    }
    return summary


def load_manifest(dataset_root: str | Path) -> list[dict[str, Any]]:
    """Load a dataset manifest from disk."""
    path = Path(dataset_root) / "manifest.json"
    if not path.exists():
        return []
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_plan_counts(plan: str) -> dict[str, dict[str, int]]:
    """Return the per-task bucket plan."""
    if plan == "pilot":
        return {task_id: dict(PILOT_BUCKET_COUNTS) for task_id in GOAL2_TASK_IDS}
    if plan == "full":
        return FULL_BUCKET_COUNTS
    raise ValueError(f"Unsupported generation plan: {plan}")


def _collect_bucket(
    task_config: TaskConfig,
    bucket_type: str,
    target_count: int,
    start_seed: int,
    output_dir: Path,
    render: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Collect one bucket for one task, retrying when needed."""
    records: list[dict[str, Any]] = []
    attempts = 0
    seed_cursor = start_seed
    max_attempts = _bucket_attempt_budget(task_config, bucket_type, target_count)
    best_candidate: dict[str, Any] | None = None

    while len(records) < target_count and attempts < max_attempts:
        episode, manifest_row = _run_controlled_episode(
            task_config=task_config,
            bucket_type=bucket_type,
            seed=seed_cursor,
            render=render,
        )
        attempts += 1
        seed_cursor += 1

        candidate = _candidate_snapshot(task_config, episode)
        if best_candidate is None or candidate["score"] > best_candidate["score"]:
            best_candidate = candidate

        should_keep = _bucket_accepts_episode(bucket_type, episode)
        if not should_keep:
            continue

        episode_path = output_dir / f"{episode['episode_id']}.json"
        dump_json(episode_path, episode)
        manifest_row["episode_path"] = str(episode_path.resolve())
        records.append(manifest_row)

    if len(records) < target_count:
        diagnostic = ""
        if best_candidate is not None:
            diagnostic = (
                " Best attempt: "
                f"seed={best_candidate['seed']}, "
                f"score={best_candidate['score']:.4f}, "
                f"satisfied={best_candidate['satisfied']}, "
                f"goal_reached={best_candidate['goal_reached']}, "
                f"min_goal_distance={best_candidate['min_goal_distance']}, "
                f"near_obs_triggered={best_candidate['near_obs_triggered']}."
            )
        raise RuntimeError(
            f"Could not collect enough episodes for {task_config.task_id} / {bucket_type}: "
            f"got {len(records)} after {attempts} attempts.{diagnostic}",
        )
    return records, seed_cursor


def _bucket_accepts_episode(bucket_type: str, episode: dict[str, Any]) -> bool:
    """Return whether a rollout qualifies for the intended bucket."""
    satisfied = bool(episode["satisfied"])
    if bucket_type == "success":
        return satisfied
    if bucket_type == "near_success":
        return True
    if bucket_type == "failure_or_recovery":
        return not satisfied or episode["summary"]["num_env_steps"] < episode["horizon"]
    raise ValueError(f"Unsupported bucket_type: {bucket_type}")


def _collect_bucket_partial(
    task_config: TaskConfig,
    bucket_type: str,
    target_count: int,
    start_seed: int,
    output_dir: Path,
    render: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Collect as many episodes as possible without raising on sparse tasks."""
    records: list[dict[str, Any]] = []
    attempts = 0
    seed_cursor = start_seed
    max_attempts = _bucket_attempt_budget(task_config, bucket_type, target_count)

    while len(records) < target_count and attempts < max_attempts:
        episode, manifest_row = _run_controlled_episode(
            task_config=task_config,
            bucket_type=bucket_type,
            seed=seed_cursor,
            render=render,
        )
        attempts += 1
        seed_cursor += 1
        if not _bucket_accepts_episode(bucket_type, episode):
            continue
        episode_path = output_dir / f"{episode['episode_id']}.json"
        dump_json(episode_path, episode)
        manifest_row["episode_path"] = str(episode_path.resolve())
        records.append(manifest_row)

    return records, seed_cursor


def _bucket_attempt_budget(task_config: TaskConfig, bucket_type: str, target_count: int) -> int:
    """Return a task-aware attempt budget for one bucket."""
    if bucket_type == "success":
        multiplier = SUCCESS_ATTEMPT_MULTIPLIERS.get(task_config.task_id, 12)
        return max(target_count * multiplier, target_count + 20)
    if bucket_type == "near_success":
        return max(target_count * 10, target_count + 15)
    if bucket_type == "failure_or_recovery":
        return max(target_count * 8, target_count + 12)
    raise ValueError(f"Unsupported bucket_type: {bucket_type}")


def _candidate_snapshot(task_config: TaskConfig, episode: dict[str, Any]) -> dict[str, Any]:
    """Capture a compact score card for one attempted episode."""
    summary = episode["summary"]
    goal_distances = [distance for distance in episode["goal_distance"] if distance is not None]
    min_goal_distance = min(goal_distances) if goal_distances else None
    score = _episode_progress_score(task_config, episode)
    return {
        "seed": episode["seed"],
        "score": score,
        "satisfied": bool(episode["satisfied"]),
        "goal_reached": bool(summary["goal_reached"]),
        "near_obs_triggered": bool(summary["near_obs_triggered"]),
        "min_goal_distance": round(float(min_goal_distance), 4) if min_goal_distance is not None else None,
    }


def _episode_progress_score(task_config: TaskConfig, episode: dict[str, Any]) -> float:
    """Score how close an episode is to task completion for debugging and seed search."""
    summary = episode["summary"]
    if episode["satisfied"]:
        return 1000.0

    score = 0.0
    goal_distances = [distance for distance in episode["goal_distance"] if distance is not None]
    if goal_distances:
        min_goal_distance = min(goal_distances)
        score += max(0.0, 10.0 - 8.0 * float(min_goal_distance))
    if summary["goal_reached"]:
        score += 40.0
    if summary["near_obs_triggered"]:
        score += 5.0
    if summary["response_satisfied"]:
        score += 8.0
    if summary["three_stage_satisfied"]:
        score += 12.0
    if summary["dual_patrol_satisfied"]:
        score += 12.0
    if task_config.task_id == "E2_L8_FullMission":
        components = summary["full_mission_components"] or {}
        score += 4.0 * sum(bool(value) for value in components.values())
    return score


def _run_controlled_episode(
    task_config: TaskConfig,
    bucket_type: str,
    seed: int,
    render: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one controlled rollout and return both the episode and its manifest row."""
    env = env_utils.make_env(task_config.env_id, render_mode="rgb_array")
    human_env = env_utils.make_env(task_config.env_id, render_mode="human") if render else None
    try:
        initial_obs, info = env.reset(seed=seed)
        if human_env is not None:
            human_env.reset(seed=seed)

        state_cache = build_state_cache(env, task_config)
        state_cache["resolved_zones"] = resolve_zone_definitions(task_config, state_cache)
        controller = Goal2OracleController(task_config, bucket_type, np.random.default_rng(seed))
        controller.reset(env, state_cache["resolved_zones"])

        steps: list[dict[str, Any]] = []
        ap_trace: list[dict[str, Any]] = []
        actions: list[list[float]] = []
        rewards: list[float] = []
        costs: list[float] = []
        terminated_flags: list[bool] = []
        truncated_flags: list[bool] = []
        obs_list: list[list[float]] = []
        agent_positions: list[list[float]] = []
        agent_velocities: list[list[float]] = []
        speed_samples: list[float] = []
        goal_distances: list[float | None] = []
        hazard_distances: list[float | None] = []
        vase_distances: list[float | None] = []

        current_obs = np.asarray(initial_obs)
        for step_idx in range(task_config.horizon):
            controller.prepare_positions(env)
            action = controller.act(env, step_idx)
            obs, reward, cost, terminated, truncated, info = env.step(action)
            if human_env is not None:
                human_env.step(action)
                human_env.render()

            ap_values = extract_ap_values(obs, info, state_cache, task_config, env=env)
            ap_trace.append({"t": step_idx, **{ap: ap_values.get(ap) for ap in task_config.required_aps}})

            obs_list.append(np.asarray(obs).tolist())
            actions.append(np.asarray(action, dtype=float).tolist())
            rewards.append(float(reward))
            costs.append(float(cost))
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            agent_positions.append(env_utils.get_agent_position(env).tolist())
            agent_velocities.append(env_utils.get_agent_velocity(env).tolist())
            speed_samples.append(float(ap_values["speed"]))
            goal_distances.append(_to_optional_float(ap_values.get("goal_distance")))
            hazard_distances.append(_to_optional_float(ap_values.get("nearest_hazard_distance")))
            vase_distances.append(_to_optional_float(ap_values.get("nearest_vase_distance")))

            steps.append(
                {
                    "t": step_idx,
                    "obs": np.asarray(obs).tolist(),
                    "action": np.asarray(action, dtype=float).tolist(),
                    "reward": float(reward),
                    "cost": float(cost),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "agent_pos": env_utils.get_agent_position(env).tolist(),
                    "agent_vel": env_utils.get_agent_velocity(env).tolist(),
                    "speed": float(ap_values["speed"]),
                    "goal_distance": _to_optional_float(ap_values.get("goal_distance")),
                    "nearest_hazard_distance": _to_optional_float(ap_values.get("nearest_hazard_distance")),
                    "nearest_vase_distance": _to_optional_float(ap_values.get("nearest_vase_distance")),
                    "aps": {ap: ap_values.get(ap) for ap in task_config.required_aps},
                },
            )
            current_obs = np.asarray(obs)
            if terminated or truncated:
                break

        satisfied, violation_step = evaluate_task(task_config, ap_trace)
        episode_id = f"{task_config.task_id}_{bucket_type}_{seed:05d}_{uuid4().hex[:8]}"
        summary = _build_episode_summary(task_config, bucket_type, ap_trace, steps, speed_samples, satisfied)

        episode = {
            "episode_id": episode_id,
            "task_id": task_config.task_id,
            "env_id": task_config.env_id,
            "level": task_config.level,
            "paper_spec_name": task_config.paper_spec_name,
            "bucket_type": bucket_type,
            "seed": seed,
            "horizon": task_config.horizon,
            "satisfied": bool(satisfied),
            "violation_step": violation_step,
            "task_config_snapshot": task_config.to_dict(),
            "resolved_zones": state_cache["resolved_zones"],
            "initial_obs": np.asarray(initial_obs).tolist(),
            "ap_trace": ap_trace,
            "obs": obs_list,
            "action": actions,
            "reward": rewards,
            "cost": costs,
            "terminated": terminated_flags,
            "truncated": truncated_flags,
            "agent_pos": agent_positions,
            "agent_vel": agent_velocities,
            "speed": speed_samples,
            "goal_distance": goal_distances,
            "nearest_hazard_distance": hazard_distances,
            "nearest_vase_distance": vase_distances,
            "steps": steps,
            "summary": summary,
            "warnings": list(dict.fromkeys(state_cache["warnings"])),
        }

        manifest_row = {
            "episode_id": episode_id,
            "task_id": task_config.task_id,
            "env_id": task_config.env_id,
            "level": task_config.level,
            "paper_spec_name": task_config.paper_spec_name,
            "bucket_type": bucket_type,
            "seed": seed,
            "horizon": task_config.horizon,
            "satisfied": bool(satisfied),
            "violation_step": violation_step,
            "summary": summary,
        }
        return episode, manifest_row
    finally:
        if "controller" in locals():
            controller.close()
        env.close()
        if human_env is not None:
            human_env.close()


def _build_episode_summary(
    task_config: TaskConfig,
    bucket_type: str,
    ap_trace: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    speed_samples: list[float],
    satisfied: bool,
) -> dict[str, Any]:
    """Build compact episode-level metrics used by later summaries."""
    l3_ok, _ = evaluate_level3("♢(A ∧ ♢(B ∧ ♢(C)))", ap_trace) if task_config.task_id == "E2_L3_ThreeStageABC" else (None, None)
    l4_ok, _ = evaluate_level4("□(near_obs → ♢(¬fast))", ap_trace) if task_config.task_id in {"E2_L4_HazardResponseDense", "E2_L6_SafeReactiveGoal", "E2_L8_FullMission"} else (None, None)
    l5_ok, _ = evaluate_level5("□(♢(A)) ∧ □(♢(B))", ap_trace, task_config.horizon) if task_config.task_id == "E2_L5_DualPatrol" else (None, None)
    l8_checks = None
    if task_config.task_id == "E2_L8_FullMission":
        l8_checks = _evaluate_full_mission_components(ap_trace, task_config.horizon)

    return {
        "bucket_type": bucket_type,
        "num_env_steps": len(steps),
        "speed_samples": [float(x) for x in speed_samples],
        "speed_stats": _aggregate_scalar_stats(speed_samples),
        "goal_reached": any(bool(step["aps"].get("goal")) for step in steps),
        "near_obs_triggered": any(bool(step["aps"].get("near_obs")) for step in steps),
        "response_satisfied": l4_ok,
        "three_stage_satisfied": l3_ok,
        "dual_patrol_satisfied": l5_ok,
        "full_mission_components": l8_checks,
        "satisfied": bool(satisfied),
    }


def _evaluate_full_mission_components(ap_trace: list[dict[str, Any]], horizon: int) -> dict[str, bool]:
    """Return compositional Full Mission coverage booleans."""
    sequencing_ok, _ = evaluate_level3("♢(A ∧ ♢(B))", ap_trace)
    patrol_ok, _ = evaluate_level5("□(♢(A))", [{"A": step["C"]} for step in ap_trace], horizon)
    safety_ok = not any(step["hazard"] for step in ap_trace)
    response_ok, _ = evaluate_level4("□(near_obs → ♢(¬fast))", ap_trace)
    return {
        "sequencing": sequencing_ok,
        "patrol": patrol_ok,
        "safety": safety_ok,
        "response": response_ok,
    }


def _aggregate_task_metrics(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate task-specific metrics across episodes."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest:
        by_task[record["task_id"]].append(record)

    metrics: dict[str, Any] = {}
    for task_id, records in by_task.items():
        if task_id == "E2_L2_SafeSlowGoal":
            metrics["L2_goal_achieved_rate"] = _mean(record["summary"]["goal_reached"] for record in records)
        elif task_id == "E2_L3_ThreeStageABC":
            metrics["L3_sequence_success_rate"] = _mean(record["summary"]["three_stage_satisfied"] for record in records)
        elif task_id == "E2_L4_HazardResponseDense":
            metrics["L4_near_obs_trigger_rate"] = _mean(record["summary"]["near_obs_triggered"] for record in records)
            metrics["L4_response_success_rate"] = _mean(record["summary"]["response_satisfied"] for record in records)
        elif task_id == "E2_L5_DualPatrol":
            metrics["L5_dual_patrol_coverage_rate"] = _mean(record["summary"]["dual_patrol_satisfied"] for record in records)
        elif task_id == "E2_L6_SafeReactiveGoal":
            metrics["L6_combined_success_rate"] = _mean(record["satisfied"] for record in records)
        elif task_id == "E2_L8_FullMission":
            components = [record["summary"]["full_mission_components"] for record in records if record["summary"]["full_mission_components"]]
            metrics["L8_component_coverage"] = {
                key: _mean(component[key] for component in components)
                for key in ("sequencing", "patrol", "safety", "response")
            }
    return metrics


def _summary_rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten episode manifest entries into CSV rows."""
    rows = []
    for record in manifest:
        rows.append(
            {
                "episode_id": record["episode_id"],
                "task_id": record["task_id"],
                "level": record["level"],
                "bucket_type": record["bucket_type"],
                "seed": record["seed"],
                "satisfied": record["satisfied"],
                "violation_step": record["violation_step"],
                "num_env_steps": record["summary"]["num_env_steps"],
                "goal_reached": record["summary"]["goal_reached"],
                "near_obs_triggered": record["summary"]["near_obs_triggered"],
                "response_satisfied": record["summary"]["response_satisfied"],
                "speed_mean": record["summary"]["speed_stats"]["mean"],
                "speed_p95": record["summary"]["speed_stats"]["p95"],
                "episode_path": record.get("episode_path"),
            },
        )
    return rows


def _aggregate_scalar_stats(values: list[float]) -> dict[str, float]:
    """Aggregate scalar distribution statistics."""
    if not values:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _mean(values) -> float:
    """Return the float mean of an iterable."""
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _to_optional_float(value: Any) -> float | None:
    """Convert a possibly missing scalar to float."""
    if value is None:
        return None
    return float(value)

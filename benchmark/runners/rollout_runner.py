"""Rollout runner for SAFEWORLD benchmark tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from benchmark import env_utils
from benchmark.ap_extractors import build_state_cache, extract_ap_values
from benchmark.evaluators import evaluate_task
from benchmark.geometry_utils import resolve_zone_definitions
from benchmark.io_utils import dump_json, ensure_dir
from benchmark.task_types import TaskConfig, TaskResult
from benchmark.visualization import save_frame


def run_task(
    task_config: TaskConfig,
    action_source: str = "random",
    seed: int = 0,
    render: bool = False,
    output_root: str | Path = "outputs",
) -> TaskResult:
    """Run one task rollout and save artifacts."""
    if action_source != "random":
        raise NotImplementedError("Only action_source='random' is supported in this phase.")

    output_dir = ensure_dir(Path(output_root) / task_config.task_id / f"seed_{seed:03d}")
    env = env_utils.make_env(task_config.env_id, render_mode="rgb_array")
    human_env = None
    warnings: list[str] = []

    try:
        if render:
            try:
                human_env = env_utils.make_env(task_config.env_id, render_mode="human")
            except Exception as exc:  # pragma: no cover - best effort human rendering
                warnings.append(f"human render unavailable: {exc}")
        if task_config.grounding_status == "placeholder":
            warnings.append(
                "placeholder task: not paper-faithfully grounded yet; excluded from default batch runs.",
            )

        obs, info = env.reset(seed=seed)
        if human_env is not None:
            human_env.reset(seed=seed)

        state_cache = build_state_cache(env, task_config)
        state_cache["resolved_zones"] = resolve_zone_definitions(task_config, state_cache)
        warnings.extend(state_cache["warnings"])

        task_snapshot = task_config.to_dict()
        task_snapshot["resolved_zones"] = state_cache["resolved_zones"]
        task_snapshot["native_layout_snapshot"] = state_cache["layout_snapshot"]
        dump_json(output_dir / "task_config_snapshot.json", task_snapshot)

        rng = np.random.default_rng(seed)
        raw_trace: list[dict[str, Any]] = []
        ap_trace: list[dict[str, Any]] = []
        actions: list[list[float]] = []
        speed_samples: list[float] = []
        saved_frames: dict[int, np.ndarray] = {}
        screenshot_steps = {0, max(1, task_config.horizon // 2), task_config.horizon - 1}

        initial_frame = env.render()
        if isinstance(initial_frame, np.ndarray):
            saved_frames[0] = initial_frame

        for t in range(task_config.horizon):
            action = _sample_action(env, rng)
            actions.append(action.tolist())
            obs, reward, cost, terminated, truncated, info = env.step(action)
            if human_env is not None:
                human_env.step(action)
                human_env.render()

            ap_values = extract_ap_values(obs, info, state_cache, task_config, env=env)
            ap_entry = {"t": t}
            for ap_name in task_config.required_aps:
                ap_entry[ap_name] = ap_values.get(ap_name)
            ap_trace.append(ap_entry)
            speed_samples.append(float(ap_values["speed"]))

            raw_trace.append(
                {
                    "t": t,
                    "obs": np.asarray(obs).tolist(),
                    "reward": float(reward),
                    "cost": float(cost),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "agent_pos": [ap_values["agent_pos_x"], ap_values["agent_pos_y"]],
                    "speed": float(ap_values["speed"]),
                    "aps": {ap_name: ap_values.get(ap_name) for ap_name in task_config.required_aps},
                    "debug": {
                        "goal_distance": ap_values.get("goal_distance"),
                        "nearest_hazard_distance": ap_values.get("nearest_hazard_distance"),
                        "nearest_vase_distance": ap_values.get("nearest_vase_distance"),
                        "target_button_distance": ap_values.get("target_button_distance"),
                    },
                    "native_state": env_utils.get_native_debug_state(env),
                },
            )

            if t in screenshot_steps:
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    saved_frames[t] = frame

            if terminated or truncated:
                break

        satisfied, violation_step = evaluate_task(task_config, ap_trace)
        warnings.extend(state_cache["warnings"])

        if violation_step is not None and violation_step not in saved_frames:
            violation_frame = _replay_frame(task_config, seed, actions, violation_step)
            if violation_frame is not None:
                saved_frames[violation_step] = violation_frame

        saved_artifacts = {
            "task_config_snapshot": str((output_dir / "task_config_snapshot.json").resolve()),
        }
        for step_idx, frame in sorted(saved_frames.items()):
            frame_name = f"frame_{step_idx:03d}.png"
            saved_artifacts.setdefault("frames", []).append(
                save_frame(frame, output_dir / frame_name),
            )

        if raw_trace:
            final_index = raw_trace[-1]["t"]
            if final_index not in saved_frames:
                final_frame = _replay_frame(task_config, seed, actions, final_index)
                if final_frame is not None:
                    saved_artifacts.setdefault("frames", []).append(
                        save_frame(final_frame, output_dir / "frame_final.png"),
                    )
            else:
                saved_artifacts.setdefault("frames", []).append(
                    save_frame(saved_frames[final_index], output_dir / "frame_final.png"),
                )

        trace_payload = {"raw_trace": raw_trace, "ap_trace": ap_trace}
        dump_json(output_dir / "trace.json", trace_payload)
        saved_artifacts["trace_json"] = str((output_dir / "trace.json").resolve())

        summary_stats = _build_summary_stats(speed_samples, raw_trace, task_config, satisfied, violation_step)
        result = TaskResult(
            task_id=task_config.task_id,
            env_id=task_config.env_id,
            seed=seed,
            horizon=task_config.horizon,
            satisfied=satisfied,
            violation_step=violation_step,
            ap_trace=ap_trace,
            raw_trace=raw_trace,
            summary_stats=summary_stats,
            saved_artifacts=saved_artifacts,
            task_config_snapshot=task_snapshot,
            grounding_status=task_config.grounding_status,
            warnings=list(dict.fromkeys(warnings)),
        )
        dump_json(output_dir / "result.json", result.to_dict())
        saved_artifacts["result_json"] = str((output_dir / "result.json").resolve())
        return result
    finally:
        env.close()
        if human_env is not None:
            human_env.close()


def _sample_action(env, rng: np.random.Generator) -> np.ndarray:
    """Sample a deterministic random Box action."""
    low = np.asarray(env.action_space.low, dtype=float)
    high = np.asarray(env.action_space.high, dtype=float)
    return rng.uniform(low=low, high=high).astype(float)


def _replay_frame(
    task_config: TaskConfig,
    seed: int,
    actions: list[list[float]],
    target_step: int,
) -> np.ndarray | None:
    """Replay one rollout deterministically to capture a frame at a target step."""
    env = env_utils.make_env(task_config.env_id, render_mode="rgb_array")
    try:
        env.reset(seed=seed)
        frame = env.render()
        if target_step == 0 and isinstance(frame, np.ndarray):
            return frame
        for t, action in enumerate(actions):
            env.step(np.asarray(action, dtype=float))
            frame = env.render()
            if t == target_step and isinstance(frame, np.ndarray):
                return frame
        return None
    finally:
        env.close()


def _build_summary_stats(
    speed_samples: list[float],
    raw_trace: list[dict[str, Any]],
    task_config: TaskConfig,
    satisfied: bool,
    violation_step: int | None,
) -> dict[str, Any]:
    """Build summary statistics for one rollout."""
    speed_array = np.asarray(speed_samples, dtype=float) if speed_samples else np.asarray([0.0])
    return {
        "num_steps_executed": len(raw_trace),
        "satisfied": satisfied,
        "violation_step": violation_step,
        "grounding_status": task_config.grounding_status,
        "speed_stats": {
            "min": float(speed_array.min()),
            "mean": float(speed_array.mean()),
            "p90": float(np.percentile(speed_array, 90)),
            "p95": float(np.percentile(speed_array, 95)),
            "max": float(speed_array.max()),
        },
    }

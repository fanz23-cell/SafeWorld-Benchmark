"""Atomic proposition extractors for SAFEWORLD benchmark tasks."""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmark import env_utils
from benchmark.geometry_utils import distance_xy, point_in_zone
from benchmark.task_types import TaskConfig


def build_state_cache(env, task_config: TaskConfig) -> dict[str, Any]:
    """Build the initial state cache used by AP extractors."""
    state_cache: dict[str, Any] = {
        "start_pos": env_utils.get_agent_position(env).tolist(),
        "goal_pos": (
            env_utils.get_goal_position(env).tolist()
            if env_utils.get_goal_position(env) is not None
            else None
        ),
        "target_button_pos": (
            env_utils.get_target_button_position(env).tolist()
            if env_utils.get_target_button_position(env) is not None
            else None
        ),
        "layout_snapshot": env_utils.get_layout_snapshot(env),
        "warnings": [],
    }
    return state_cache


def extract_ap_values(
    obs,
    info,
    state_cache: dict[str, Any],
    task_config: TaskConfig,
    env=None,
) -> dict[str, bool | float | None]:
    """Extract task AP values and continuous debug metrics from native env state."""
    if env is None:
        raise ValueError("env is required for native AP extraction")
    _ = (obs, info)

    task = env_utils.get_task(env)
    agent_pos = env_utils.get_agent_position(env)
    agent_vel = env_utils.get_agent_velocity(env)
    speed = float(np.linalg.norm(agent_vel[:2]))

    hazard_positions = env_utils.get_hazard_positions(env)
    hazard_size = env_utils.get_hazard_size(env) or 0.0
    hazard_distances = [distance_xy(agent_pos, pos) for pos in hazard_positions]
    min_hazard_distance = min(hazard_distances) if hazard_distances else None
    hazard = any(distance <= hazard_size for distance in hazard_distances)

    goal = bool(task.goal_achieved) if hasattr(task, "goal_achieved") else False
    goal_pos = env_utils.get_goal_position(env)
    goal_distance = distance_xy(agent_pos, goal_pos) if goal_pos is not None else None

    fast_threshold = float(task_config.ap_params.get("fast_threshold", 0.35))
    fast = speed > fast_threshold

    vase_positions = env_utils.get_vase_positions(env)
    vase_distances = [distance_xy(agent_pos, pos) for pos in vase_positions]
    nearest_vase_distance = min(vase_distances) if vase_distances else None
    near_obs_threshold = float(task_config.ap_params.get("near_obs_threshold", 0.30))
    near_obs = (
        nearest_vase_distance is not None and nearest_vase_distance < near_obs_threshold
    )

    target_button_pos = env_utils.get_target_button_position(env)
    target_button_distance = (
        distance_xy(agent_pos, target_button_pos) if target_button_pos is not None else None
    )

    values: dict[str, bool | float | None] = {
        "hazard": hazard,
        "goal": goal,
        "fast": fast,
        "near_obs": near_obs,
        "near_human": None,
        "speed": speed,
        "fast_threshold": fast_threshold,
        "goal_distance": goal_distance,
        "nearest_hazard_distance": min_hazard_distance,
        "nearest_vase_distance": nearest_vase_distance,
        "target_button_distance": target_button_distance,
        "agent_pos_x": float(agent_pos[0]),
        "agent_pos_y": float(agent_pos[1]),
    }

    for zone_name, zone_info in state_cache.get("resolved_zones", {}).items():
        values[zone_name] = point_in_zone(
            agent_pos[:2],
            np.asarray(zone_info["center"], dtype=float),
            float(zone_info["radius"]),
        )

    if "near_human" in task_config.required_aps:
        values["near_human"] = None
        state_cache["warnings"].append(
            "near_human is placeholder-only and not paper-faithfully grounded yet.",
        )

    return values

"""Environment helpers for SAFEWORLD benchmark tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
import safety_gymnasium as gym


def make_env(env_id: str, render_mode: str = "rgb_array"):
    """Create one Safety Gymnasium environment."""
    return gym.make(env_id, render_mode=render_mode)


def get_task(env) -> Any:
    """Return the unwrapped Safety Gymnasium task object."""
    return env.unwrapped.task


def get_agent(env) -> Any:
    """Return the native agent object."""
    return get_task(env).agent


def get_agent_position(env) -> np.ndarray:
    """Return native agent position."""
    return get_agent(env).pos.copy()


def get_agent_velocity(env) -> np.ndarray:
    """Return native agent velocity."""
    return get_agent(env).vel.copy()


def get_goal_position(env) -> np.ndarray | None:
    """Return native goal position when available."""
    task = get_task(env)
    return task.goal.pos.copy() if hasattr(task, "goal") else None


def get_goal_size(env) -> float | None:
    """Return native goal radius when available."""
    task = get_task(env)
    return float(task.goal.size) if hasattr(task, "goal") else None


def get_hazard_positions(env) -> list[np.ndarray]:
    """Return native hazard positions when available."""
    task = get_task(env)
    if hasattr(task, "hazards"):
        return [np.asarray(pos).copy() for pos in task.hazards.pos]
    return []


def get_hazard_size(env) -> float | None:
    """Return native hazard radius when available."""
    task = get_task(env)
    return float(task.hazards.size) if hasattr(task, "hazards") else None


def get_vase_positions(env) -> list[np.ndarray]:
    """Return native vase positions when available."""
    task = get_task(env)
    if hasattr(task, "vases"):
        return [np.asarray(pos).copy() for pos in task.vases.pos]
    return []


def get_button_positions(env) -> list[np.ndarray]:
    """Return native button positions when available."""
    task = get_task(env)
    if hasattr(task, "buttons"):
        return [np.asarray(pos).copy() for pos in task.buttons.pos]
    return []


def get_button_size(env) -> float | None:
    """Return native button radius when available."""
    task = get_task(env)
    return float(task.buttons.size) if hasattr(task, "buttons") else None


def get_target_button_index(env) -> int | None:
    """Return the native target button index when available."""
    task = get_task(env)
    if hasattr(task, "buttons") and getattr(task.buttons, "goal_button", None) is not None:
        return int(task.buttons.goal_button)
    return None


def get_target_button_position(env) -> np.ndarray | None:
    """Return native target button position when available."""
    button_index = get_target_button_index(env)
    if button_index is None:
        return None
    positions = get_button_positions(env)
    return positions[button_index].copy()


def get_gremlin_positions(env) -> list[np.ndarray]:
    """Return native gremlin positions when available."""
    task = get_task(env)
    if hasattr(task, "gremlins"):
        return [np.asarray(pos).copy() for pos in task.gremlins.pos]
    return []


def get_layout_snapshot(env) -> dict[str, Any]:
    """Return sampled layout from the native random generator when available."""
    task = get_task(env)
    layout = getattr(task.agent.random_generator, "layout", {})
    snapshot = {}
    for key, value in layout.items():
        if hasattr(value, "tolist"):
            snapshot[key] = value.tolist()
        else:
            snapshot[key] = value
    return snapshot


def get_native_debug_state(env) -> dict[str, Any]:
    """Return a compact native state snapshot for debugging."""
    task = get_task(env)
    debug_state: dict[str, Any] = {
        "agent_pos": get_agent_position(env).tolist(),
        "agent_vel": get_agent_velocity(env).tolist(),
        "goal_pos": get_goal_position(env).tolist() if get_goal_position(env) is not None else None,
        "goal_size": get_goal_size(env),
        "hazard_size": get_hazard_size(env),
        "hazard_positions": [pos.tolist() for pos in get_hazard_positions(env)],
        "vase_positions": [pos.tolist() for pos in get_vase_positions(env)],
        "button_positions": [pos.tolist() for pos in get_button_positions(env)],
        "button_size": get_button_size(env),
        "target_button_index": get_target_button_index(env),
        "target_button_pos": (
            get_target_button_position(env).tolist()
            if get_target_button_position(env) is not None
            else None
        ),
        "gremlin_positions": [pos.tolist() for pos in get_gremlin_positions(env)],
        "layout_snapshot": get_layout_snapshot(env),
    }
    if hasattr(task, "goal_achieved"):
        debug_state["goal_achieved"] = bool(task.goal_achieved)
    return debug_state

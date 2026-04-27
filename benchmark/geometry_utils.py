"""Geometry helpers for benchmark task grounding."""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmark.task_types import TaskConfig, ZoneDefinition


def distance_xy(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Compute Euclidean distance in XY coordinates."""
    a_xy = np.asarray(point_a, dtype=float)[:2]
    b_xy = np.asarray(point_b, dtype=float)[:2]
    return float(np.linalg.norm(a_xy - b_xy))


def resolve_zone_definitions(
    task_config: TaskConfig,
    state_cache: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve declarative zone definitions into concrete circles."""
    resolved: dict[str, dict[str, Any]] = {}
    start_pos = np.asarray(state_cache["start_pos"], dtype=float)[:2]
    goal_pos = np.asarray(state_cache.get("goal_pos"), dtype=float)[:2] if state_cache.get("goal_pos") is not None else None
    target_button_pos = (
        np.asarray(state_cache.get("target_button_pos"), dtype=float)[:2]
        if state_cache.get("target_button_pos") is not None
        else None
    )

    for zone_def in task_config.zone_defs:
        center = _resolve_zone_center(zone_def, start_pos, goal_pos, target_button_pos)
        resolved[zone_def.name] = {
            "name": zone_def.name,
            "kind": zone_def.kind,
            "center": center.tolist(),
            "radius": zone_def.radius,
            "description": zone_def.description,
        }
    return resolved


def _resolve_zone_center(
    zone_def: ZoneDefinition,
    start_pos: np.ndarray,
    goal_pos: np.ndarray | None,
    target_button_pos: np.ndarray | None,
) -> np.ndarray:
    """Resolve one zone definition to a center point."""
    if zone_def.kind == "interpolated_zone":
        if zone_def.anchor == "goal":
            assert goal_pos is not None, "goal position required for interpolated goal zone"
            anchor = goal_pos
        elif zone_def.anchor == "target_button":
            assert target_button_pos is not None, "target button position required for interpolated button zone"
            anchor = target_button_pos
        else:
            raise ValueError(f"Unsupported zone anchor: {zone_def.anchor}")
        assert zone_def.interpolation is not None, "interpolation required for interpolated_zone"
        return start_pos + zone_def.interpolation * (anchor - start_pos)

    if zone_def.kind == "goal_region":
        assert goal_pos is not None, "goal position required for goal region"
        return goal_pos

    if zone_def.kind == "target_button_zone":
        assert target_button_pos is not None, "target button position required for target button zone"
        return target_button_pos

    raise ValueError(f"Unsupported zone kind: {zone_def.kind}")


def point_in_zone(point: np.ndarray, center: np.ndarray, radius: float) -> bool:
    """Check whether a point lies within a circular zone."""
    return distance_xy(point, center) <= float(radius)

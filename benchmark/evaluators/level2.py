"""Spec-specific evaluators for level 2."""

from __future__ import annotations

from typing import Any


def evaluate_level2(formula: str, ap_trace: list[dict[str, Any]]) -> tuple[bool, int | None]:
    """Evaluate one level 2 formula over a boolean AP trace."""
    if formula == "♢(goal) ∧ □(¬hazard)":
        goal_seen = False
        for idx, step in enumerate(ap_trace):
            if step["hazard"]:
                return False, idx
            goal_seen = goal_seen or step["goal"]
        return goal_seen, None if goal_seen else len(ap_trace) - 1

    if formula == "♢(goal) ∧ □(¬hazard) ∧ □(¬fast)":
        goal_seen = False
        for idx, step in enumerate(ap_trace):
            if step["hazard"] or step["fast"]:
                return False, idx
            goal_seen = goal_seen or step["goal"]
        return goal_seen, None if goal_seen else len(ap_trace) - 1

    raise ValueError(f"Unsupported level 2 formula: {formula}")

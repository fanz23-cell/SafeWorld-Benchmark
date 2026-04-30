"""Spec-specific evaluators for level 6."""

from __future__ import annotations

from typing import Any

from benchmark.evaluators.level2 import evaluate_level2
from benchmark.evaluators.level4 import evaluate_level4
from benchmark.evaluators.level5 import _evaluate_windowed_eventually


def evaluate_level6(
    formula: str,
    ap_trace: list[dict[str, Any]],
    horizon: int,
) -> tuple[bool, int | None]:
    """Evaluate one level 6 formula over a boolean AP trace."""
    window = max(1, horizon // 2)

    if formula == "□(♢(A)) ∧ □(¬hazard)":
        recurrence_ok, recurrence_violation = _evaluate_windowed_eventually(ap_trace, "A", window)
        if not recurrence_ok:
            return False, recurrence_violation
        for idx, step in enumerate(ap_trace):
            if step["hazard"]:
                return False, idx
        return True, None

    if formula == "♢(goal) ∧ □(¬hazard) ∧ □(near_obs → ♢(¬fast))":
        goal_safe_ok, goal_safe_violation = evaluate_level2("♢(goal) ∧ □(¬hazard)", ap_trace)
        if not goal_safe_ok:
            return False, goal_safe_violation
        return evaluate_level4("□(near_obs → ♢(¬fast))", ap_trace)

    raise ValueError(f"Unsupported level 6 formula: {formula}")

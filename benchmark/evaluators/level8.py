"""Spec-specific evaluators for level 8."""

from __future__ import annotations

from typing import Any

from benchmark.evaluators.level3 import evaluate_level3
from benchmark.evaluators.level4 import evaluate_level4
from benchmark.evaluators.level5 import _evaluate_windowed_eventually


def evaluate_level8(
    formula: str,
    ap_trace: list[dict[str, Any]],
    horizon: int,
) -> tuple[bool, int | None]:
    """Evaluate one level 8 formula over a boolean AP trace."""
    if formula != "♢(A ∧ ♢(B)) ∧ □(♢(C)) ∧ □(¬hazard) ∧ □(near_obs → ♢(¬fast))":
        raise ValueError(f"Unsupported level 8 formula: {formula}")

    sequencing_ok, sequencing_violation = evaluate_level3("♢(A ∧ ♢(B))", ap_trace)
    if not sequencing_ok:
        return False, sequencing_violation

    patrol_ok, patrol_violation = _evaluate_windowed_eventually(ap_trace, "C", max(1, horizon // 2))
    if not patrol_ok:
        return False, patrol_violation

    for idx, step in enumerate(ap_trace):
        if step["hazard"]:
            return False, idx

    return evaluate_level4("□(near_obs → ♢(¬fast))", ap_trace)

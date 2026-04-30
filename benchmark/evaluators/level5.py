"""Spec-specific evaluators for level 5."""

from __future__ import annotations

from typing import Any


def evaluate_level5(
    formula: str,
    ap_trace: list[dict[str, Any]],
    horizon: int,
) -> tuple[bool, int | None]:
    """Evaluate one level 5 formula over a boolean AP trace."""
    window = max(1, horizon // 2)

    if formula == "□(♢(A))":
        return _evaluate_windowed_eventually(ap_trace, "A", window)

    if formula == "□(♢(A)) ∧ □(♢(B))":
        satisfied_a, violation_a = _evaluate_windowed_eventually(ap_trace, "A", window)
        if not satisfied_a:
            return False, violation_a
        return _evaluate_windowed_eventually(ap_trace, "B", window)

    raise ValueError(f"Unsupported level 5 formula: {formula}")


def _evaluate_windowed_eventually(
    ap_trace: list[dict[str, Any]],
    ap_name: str,
    window: int,
) -> tuple[bool, int | None]:
    """Evaluate a bounded finite-trace surrogate for □♢(ap)."""
    if not ap_trace:
        return False, 0
    last_index = len(ap_trace) - 1
    for idx in range(len(ap_trace)):
        upper = min(last_index, idx + window)
        if not any(ap_trace[j][ap_name] for j in range(idx, upper + 1)):
            return False, idx
    return True, None

"""Spec-specific evaluators for level 3."""

from __future__ import annotations

from typing import Any


def _find_first_true(ap_trace: list[dict[str, Any]], ap_name: str, start: int = 0) -> int | None:
    """Find the first index at or after start where an AP is true."""
    for idx in range(start, len(ap_trace)):
        if ap_trace[idx][ap_name]:
            return idx
    return None


def evaluate_level3(formula: str, ap_trace: list[dict[str, Any]]) -> tuple[bool, int | None]:
    """Evaluate one level 3 formula over a boolean AP trace."""
    if formula == "♢(A ∧ ♢(B))":
        a_idx = _find_first_true(ap_trace, "A")
        if a_idx is None:
            return False, len(ap_trace) - 1
        b_idx = _find_first_true(ap_trace, "B", start=a_idx + 1)
        return (b_idx is not None), None if b_idx is not None else len(ap_trace) - 1

    if formula == "♢(A ∧ ♢(B ∧ ♢(C)))":
        a_idx = _find_first_true(ap_trace, "A")
        if a_idx is None:
            return False, len(ap_trace) - 1
        b_idx = _find_first_true(ap_trace, "B", start=a_idx + 1)
        if b_idx is None:
            return False, len(ap_trace) - 1
        c_idx = _find_first_true(ap_trace, "C", start=b_idx + 1)
        return (c_idx is not None), None if c_idx is not None else len(ap_trace) - 1

    raise ValueError(f"Unsupported level 3 formula: {formula}")

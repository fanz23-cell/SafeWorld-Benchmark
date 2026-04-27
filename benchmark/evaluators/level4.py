"""Spec-specific evaluators for level 4."""

from __future__ import annotations

from typing import Any


def _evaluate_eventual_slowdown(
    ap_trace: list[dict[str, Any]],
    trigger_name: str,
) -> tuple[bool, int | None]:
    """Evaluate □(trigger -> ♢(¬fast)) on a finite trace."""
    for idx, step in enumerate(ap_trace):
        if step[trigger_name]:
            if not any(not future_step["fast"] for future_step in ap_trace[idx:]):
                return False, idx
    return True, None


def evaluate_level4(formula: str, ap_trace: list[dict[str, Any]]) -> tuple[bool, int | None]:
    """Evaluate one level 4 formula over a boolean AP trace."""
    if formula == "□(near_obs → ♢(¬fast))":
        return _evaluate_eventual_slowdown(ap_trace, "near_obs")
    if formula == "□(near_human → ♢(¬fast))":
        return _evaluate_eventual_slowdown(ap_trace, "near_human")
    raise ValueError(f"Unsupported level 4 formula: {formula}")

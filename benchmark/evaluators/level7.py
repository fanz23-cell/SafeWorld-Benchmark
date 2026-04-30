"""Spec-specific evaluators for level 7."""

from __future__ import annotations

from typing import Any


def evaluate_level7(formula: str, ap_trace: list[dict[str, Any]]) -> tuple[bool, int | None]:
    """Evaluate one level 7 formula over a boolean AP trace."""
    if formula == "□(carrying → ¬fast)":
        for idx, step in enumerate(ap_trace):
            if step["carrying"] and step["fast"]:
                return False, idx
        return True, None

    if formula == "□(near_human → ¬hazard)":
        for idx, step in enumerate(ap_trace):
            if step["near_human"] and step["hazard"]:
                return False, idx
        return True, None

    raise ValueError(f"Unsupported level 7 formula: {formula}")

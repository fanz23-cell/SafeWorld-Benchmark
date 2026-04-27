"""Spec-specific evaluators for level 1."""

from __future__ import annotations

from typing import Any


def evaluate_level1(formula: str, ap_trace: list[dict[str, Any]]) -> tuple[bool, int | None]:
    """Evaluate one level 1 formula over a boolean AP trace."""
    if formula == "□(¬hazard)":
        for idx, step in enumerate(ap_trace):
            if step["hazard"]:
                return False, idx
        return True, None
    if formula == "□(¬fast)":
        for idx, step in enumerate(ap_trace):
            if step["fast"]:
                return False, idx
        return True, None
    raise ValueError(f"Unsupported level 1 formula: {formula}")

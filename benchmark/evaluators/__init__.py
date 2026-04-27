"""Evaluator dispatch for SAFEWORLD benchmark tasks."""

from __future__ import annotations

from benchmark.evaluators.level1 import evaluate_level1
from benchmark.evaluators.level2 import evaluate_level2
from benchmark.evaluators.level3 import evaluate_level3
from benchmark.evaluators.level4 import evaluate_level4
from benchmark.task_types import TaskConfig


def evaluate_task(task_config: TaskConfig, ap_trace: list[dict]) -> tuple[bool, int | None]:
    """Dispatch to the formula-specific evaluator for one task."""
    if task_config.level == 1:
        return evaluate_level1(task_config.paper_formula_str, ap_trace)
    if task_config.level == 2:
        return evaluate_level2(task_config.paper_formula_str, ap_trace)
    if task_config.level == 3:
        return evaluate_level3(task_config.paper_formula_str, ap_trace)
    if task_config.level == 4:
        return evaluate_level4(task_config.paper_formula_str, ap_trace)
    raise ValueError(f"Unsupported level: {task_config.level}")

"""Evaluator dispatch for SAFEWORLD benchmark tasks."""

from __future__ import annotations

from benchmark.evaluators.level1 import evaluate_level1
from benchmark.evaluators.level2 import evaluate_level2
from benchmark.evaluators.level3 import evaluate_level3
from benchmark.evaluators.level4 import evaluate_level4
from benchmark.evaluators.level5 import evaluate_level5
from benchmark.evaluators.level6 import evaluate_level6
from benchmark.evaluators.level7 import evaluate_level7
from benchmark.evaluators.level8 import evaluate_level8
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
    if task_config.level == 5:
        return evaluate_level5(task_config.paper_formula_str, ap_trace, task_config.horizon)
    if task_config.level == 6:
        return evaluate_level6(task_config.paper_formula_str, ap_trace, task_config.horizon)
    if task_config.level == 7:
        return evaluate_level7(task_config.paper_formula_str, ap_trace)
    if task_config.level == 8:
        return evaluate_level8(task_config.paper_formula_str, ap_trace, task_config.horizon)
    raise ValueError(f"Unsupported level: {task_config.level}")

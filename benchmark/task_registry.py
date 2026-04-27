"""Registry for SAFEWORLD benchmark task configs."""

from __future__ import annotations

from benchmark.task_configs.level1 import get_level1_tasks
from benchmark.task_configs.level2 import get_level2_tasks
from benchmark.task_configs.level3 import get_level3_tasks
from benchmark.task_configs.level4 import get_level4_tasks
from benchmark.task_types import TaskConfig


def list_task_configs(include_disabled: bool = True) -> list[TaskConfig]:
    """Return all known task configs."""
    tasks = get_level1_tasks() + get_level2_tasks() + get_level3_tasks() + get_level4_tasks()
    if include_disabled:
        return tasks
    return [task for task in tasks if task.default_enabled]


def get_task_config(task_id: str) -> TaskConfig:
    """Look up one task config by id."""
    for task in list_task_configs(include_disabled=True):
        if task.task_id == task_id:
            return task
    raise KeyError(f"Unknown task_id: {task_id}")

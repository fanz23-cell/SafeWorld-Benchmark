"""Batch runner for SAFEWORLD benchmark task suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.io_utils import dump_json, ensure_dir, write_summary_csv
from benchmark.runners.rollout_runner import run_task
from benchmark.task_registry import list_task_configs
from benchmark.task_types import TaskResult


def run_level_suite(
    levels: list[int],
    seed: int,
    render: bool = False,
    include_placeholders: bool = False,
    output_root: str | Path = "outputs",
) -> dict[str, Any]:
    """Run all enabled tasks for the requested levels and save suite summaries."""
    selected_tasks = [
        task
        for task in list_task_configs(include_disabled=include_placeholders)
        if task.level in levels and (include_placeholders or task.default_enabled)
    ]

    results: list[TaskResult] = [
        run_task(task, action_source="random", seed=seed, render=render, output_root=output_root)
        for task in selected_tasks
    ]

    suite_dir = ensure_dir(Path(output_root) / "suite_runs" / f"levels_{'_'.join(map(str, levels))}" / f"seed_{seed:03d}")
    summary_rows = [_result_summary_row(result) for result in results]
    write_summary_csv(suite_dir / "summary.csv", summary_rows)
    dump_json(suite_dir / "summary.json", summary_rows)

    all_tasks = [task for task in list_task_configs(include_disabled=True) if task.level in levels]
    placeholder_ids = [task.task_id for task in all_tasks if task.grounding_status == "placeholder"]
    runnable_ids = [row["task_id"] for row in summary_rows if row["grounding_status"] == "fully_runnable"]
    manual_review_ids = [row["task_id"] for row in summary_rows if row["grounding_status"] == "needs_manual_review"]

    suite_summary = {
        "levels": levels,
        "seed": seed,
        "fully_runnable": runnable_ids,
        "placeholder": placeholder_ids,
        "needs_manual_review": manual_review_ids,
        "summary_csv": str((suite_dir / "summary.csv").resolve()),
        "summary_json": str((suite_dir / "summary.json").resolve()),
        "results": summary_rows,
    }
    dump_json(suite_dir / "suite_manifest.json", suite_summary)
    return suite_summary


def _result_summary_row(result: TaskResult) -> dict[str, Any]:
    """Flatten one task result into a compact summary row."""
    return {
        "task_id": result.task_id,
        "env_id": result.env_id,
        "seed": result.seed,
        "level": result.task_config_snapshot["level"],
        "paper_spec_name": result.task_config_snapshot["paper_spec_name"],
        "formula": result.task_config_snapshot["paper_formula_str"],
        "grounding_status": result.grounding_status,
        "satisfied": result.satisfied,
        "violation_step": result.violation_step,
        "num_steps_executed": result.summary_stats["num_steps_executed"],
        "speed_mean": result.summary_stats["speed_stats"]["mean"],
        "speed_p95": result.summary_stats["speed_stats"]["p95"],
        "result_json": result.saved_artifacts.get("result_json"),
    }

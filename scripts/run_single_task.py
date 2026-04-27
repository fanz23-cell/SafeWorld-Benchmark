#!/usr/bin/env python3
"""Run one SAFEWORLD benchmark task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.runners.rollout_runner import run_task
from benchmark.task_registry import get_task_config


def main() -> None:
    """CLI entry point for running one task."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--allow-placeholder", action="store_true")
    args = parser.parse_args()

    task_config = get_task_config(args.task_id)
    if task_config.grounding_status == "placeholder" and not args.allow_placeholder:
        raise SystemExit(
            f"{task_config.task_id} is placeholder-only. Pass --allow-placeholder to run it manually.",
        )

    result = run_task(
        task_config=task_config,
        action_source="random",
        seed=args.seed,
        render=args.render,
        output_root=args.output_root,
    )

    summary = {
        "task_id": result.task_id,
        "env_id": result.env_id,
        "paper_spec_name": result.task_config_snapshot["paper_spec_name"],
        "formula": result.task_config_snapshot["paper_formula_str"],
        "grounding_status": result.grounding_status,
        "satisfied": result.satisfied,
        "violation_step": result.violation_step,
        "saved_artifacts": result.saved_artifacts,
        "speed_stats": result.summary_stats["speed_stats"],
        "warnings": result.warnings,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

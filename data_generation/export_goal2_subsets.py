"""Subset exports derived from the Goal2 master dataset."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.io_utils import dump_json, ensure_dir, write_summary_csv
from data_generation.generate_goal2_master_dataset import GOAL2_TASK_IDS, load_manifest, summarize_dataset


TARGET_RATIOS = {"success": 0.70, "near_success": 0.20, "failure_or_recovery": 0.10}


def export_goal2_subsets(
    master_root: str | Path = "datasets/goal2_master",
    mixed_root: str | Path = "datasets/goal2_mixed_70_20_10",
    success_root: str | Path = "datasets/goal2_success_only",
) -> dict[str, Any]:
    """Export mixed and success-only subsets from the master dataset."""
    manifest = load_manifest(master_root)
    if not manifest:
        raise FileNotFoundError(f"No manifest found under {master_root}")

    mixed_manifest = _select_mixed_manifest(manifest)
    success_manifest = [record for record in manifest if record["bucket_type"] == "success"]

    mixed_summary = _materialize_subset(mixed_manifest, mixed_root, "mixed")
    success_summary = _materialize_subset(success_manifest, success_root, "success_only")

    return {
        "mixed": mixed_summary,
        "success_only": success_summary,
    }


def _select_mixed_manifest(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select a per-task 70/20/10 mixed subset without changing labels."""
    by_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in manifest:
        by_task[record["task_id"]][record["bucket_type"]].append(record)

    selected: list[dict[str, Any]] = []
    for task_id in GOAL2_TASK_IDS:
        buckets = by_task[task_id]
        available = {name: len(buckets.get(name, [])) for name in TARGET_RATIOS}
        scale = min(
            available["success"] / TARGET_RATIOS["success"],
            available["near_success"] / TARGET_RATIOS["near_success"],
            available["failure_or_recovery"] / TARGET_RATIOS["failure_or_recovery"],
        )
        total = int(scale)
        counts = {
            "success": int(total * TARGET_RATIOS["success"]),
            "near_success": int(total * TARGET_RATIOS["near_success"]),
            "failure_or_recovery": int(total * TARGET_RATIOS["failure_or_recovery"]),
        }
        while sum(counts.values()) < total:
            for bucket_type in ("success", "near_success", "failure_or_recovery"):
                counts[bucket_type] += 1
                if sum(counts.values()) == total:
                    break

        for bucket_type, count in counts.items():
            selected.extend(buckets[bucket_type][:count])
    return selected


def _materialize_subset(
    manifest: list[dict[str, Any]],
    output_root: str | Path,
    summary_stem: str,
) -> dict[str, Any]:
    """Write subset manifests and summaries to disk."""
    output_root = ensure_dir(Path(output_root))
    dump_json(output_root / "manifest.json", manifest)
    summary = summarize_dataset(manifest)
    dump_json(output_root / f"dataset_summary_{summary_stem}.json", summary)
    write_summary_csv(output_root / f"dataset_summary_{summary_stem}.csv", _rows(manifest))
    _copy_episode_index(manifest, output_root / "episode_index.json")
    return summary


def _copy_episode_index(manifest: list[dict[str, Any]], path: Path) -> None:
    """Save a compact episode index without duplicating episode payloads."""
    rows = [
        {
            "episode_id": record["episode_id"],
            "task_id": record["task_id"],
            "bucket_type": record["bucket_type"],
            "episode_path": record["episode_path"],
        }
        for record in manifest
    ]
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return CSV-ready rows."""
    return [
        {
            "episode_id": record["episode_id"],
            "task_id": record["task_id"],
            "bucket_type": record["bucket_type"],
            "seed": record["seed"],
            "satisfied": record["satisfied"],
            "num_env_steps": record["summary"]["num_env_steps"],
            "speed_mean": record["summary"]["speed_stats"]["mean"],
            "episode_path": record["episode_path"],
        }
        for record in manifest
    ]


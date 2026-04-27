"""I/O helpers for benchmark task runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    """Create a directory when missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: Any) -> None:
    """Save JSON with stable formatting."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a summary CSV from homogeneous dict rows."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

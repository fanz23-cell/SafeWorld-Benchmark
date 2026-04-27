#!/usr/bin/env python3
"""Run one or more SAFEWORLD benchmark levels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.runners.batch_runner import run_level_suite


def main() -> None:
    """CLI entry point for suite runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", nargs="+", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--include-placeholders", action="store_true")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    summary = run_level_suite(
        levels=args.level,
        seed=args.seed,
        render=args.render,
        include_placeholders=args.include_placeholders,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

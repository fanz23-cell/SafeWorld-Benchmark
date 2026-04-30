"""Generate the planned full Goal2 master dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_generation.generate_goal2_master_dataset import generate_goal2_master_dataset


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="datasets/goal2_master")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--plan", choices=["pilot", "full"], default="full")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    summary = generate_goal2_master_dataset(
        output_root=args.output_root,
        plan=args.plan,
        seed=args.seed,
        render=args.render,
        allow_partial=args.allow_partial,
    )
    print(summary["total_episodes"])


if __name__ == "__main__":
    main()

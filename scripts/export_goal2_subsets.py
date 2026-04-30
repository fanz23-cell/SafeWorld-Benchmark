"""Export mixed and success-only Goal2 subsets from the master dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_generation.export_goal2_subsets import export_goal2_subsets


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", default="datasets/goal2_master")
    parser.add_argument("--mixed-root", default="datasets/goal2_mixed_70_20_10")
    parser.add_argument("--success-root", default="datasets/goal2_success_only")
    args = parser.parse_args()

    summary = export_goal2_subsets(
        master_root=args.master_root,
        mixed_root=args.mixed_root,
        success_root=args.success_root,
    )
    print(summary["mixed"]["total_episodes"])
    print(summary["success_only"]["total_episodes"])


if __name__ == "__main__":
    main()

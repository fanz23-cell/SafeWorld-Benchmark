"""Export Goal2 datasets into DreamerV3 replay chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_generation.dreamerv3_adapter import export_goal2_for_dreamerv3


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="datasets/goal2_master")
    parser.add_argument("--output-root", default="datasets/goal2_master_dreamerv3")
    parser.add_argument("--replay-length", type=int, default=1)
    parser.add_argument("--replay-chunksize", type=int, default=1024)
    args = parser.parse_args()

    summary = export_goal2_for_dreamerv3(
        source_root=args.source_root,
        output_root=args.output_root,
        replay_length=args.replay_length,
        replay_chunksize=args.replay_chunksize,
    )
    print(summary["num_replay_steps"])


if __name__ == "__main__":
    main()

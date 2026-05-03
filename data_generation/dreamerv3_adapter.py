"""DreamerV3 replay export for Goal2 datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.io_utils import dump_json, ensure_dir


def export_goal2_for_dreamerv3(
    source_root: str | Path,
    output_root: str | Path,
    replay_length: int = 1,
    replay_chunksize: int = 1024,
) -> dict[str, Any]:
    """Export one dataset manifest into DreamerV3-compatible replay chunks."""
    source_root = Path(source_root)
    output_root = ensure_dir(Path(output_root))
    manifest_path = source_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    dreamer_root = Path(__file__).resolve().parents[1] / "external" / "dreamerv3"
    if str(dreamer_root) not in sys.path:
        sys.path.insert(0, str(dreamer_root))
    import embodied  # noqa: PLC0415

    total_steps = 0
    replay = embodied.replay.Replay(
        length=replay_length,
        capacity=max(1, int(sum(record["summary"]["num_env_steps"] for record in manifest) * 2)),
        directory=output_root,
        chunksize=replay_chunksize,
        save_wait=True,
    )

    for worker, record in enumerate(manifest):
        # episode_path may be an absolute path from a different machine; resolve
        # it relative to source_root/episodes using the last 3 path components
        # (task_dir/bucket/filename) so the dataset is portable.
        raw_path = Path(record["episode_path"])
        ep_path = source_root / "episodes" / Path(*raw_path.parts[-3:])
        if not ep_path.exists():
            ep_path = raw_path  # fall back to absolute if it somehow exists
        episode = json.loads(ep_path.read_text(encoding="utf-8"))
        for step in _episode_to_dreamer_steps(episode):
            replay.add(step, worker=worker)
            total_steps += 1
    replay.save()

    export_summary = {
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "num_episodes": len(manifest),
        "num_replay_steps": total_steps,
        "replay_length": replay_length,
        "replay_chunksize": replay_chunksize,
        "format_note": "Observation key follows DreamerV3 FromGym default: image; action key: action.",
    }
    dump_json(output_root / "dreamerv3_export_summary.json", export_summary)
    return export_summary


def _episode_to_dreamer_steps(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one benchmark episode into DreamerV3 replay transitions."""
    obs_list = [np.asarray(episode["initial_obs"], dtype=np.float32)] + [
        np.asarray(obs, dtype=np.float32) for obs in episode["obs"]
    ]
    actions = [np.asarray(action, dtype=np.float32) for action in episode["action"]]
    rewards = [0.0] + [float(reward) for reward in episode["reward"]]
    terminated = [False] + [bool(flag) for flag in episode["terminated"]]
    truncated = [False] + [bool(flag) for flag in episode["truncated"]]
    costs = [0.0] + [float(cost) for cost in episode["cost"]]
    speeds = [0.0] + [float(speed) for speed in episode["speed"]]
    goal_distances = [None] + list(episode["goal_distance"])
    hazard_distances = [None] + list(episode["nearest_hazard_distance"])
    vase_distances = [None] + list(episode["nearest_vase_distance"])
    bucket = episode["bucket_type"]

    steps: list[dict[str, Any]] = []
    for idx, obs in enumerate(obs_list):
        if idx < len(actions):
            action = actions[idx]
        else:
            action = np.zeros_like(actions[0]) if actions else np.zeros((2,), dtype=np.float32)
        is_last = bool(terminated[idx] or truncated[idx]) if idx < len(terminated) else True
        is_terminal = bool(terminated[idx]) if idx < len(terminated) else False
        step = {
            "image": obs,
            "action": action,
            "reset": np.bool_(idx == 0),
            "reward": np.float32(rewards[idx]),
            "is_first": np.bool_(idx == 0),
            "is_last": np.bool_(is_last or idx == len(obs_list) - 1),
            "is_terminal": np.bool_(is_terminal),
            "cost": np.float32(costs[idx]),
            "speed": np.float32(speeds[idx]),
            "goal_distance": _float_or_nan(goal_distances[idx]),
            "nearest_hazard_distance": _float_or_nan(hazard_distances[idx]),
            "nearest_vase_distance": _float_or_nan(vase_distances[idx]),
            "level": np.int32(episode["level"]),
            "bucket_success": np.bool_(bucket == "success"),
            "bucket_near_success": np.bool_(bucket == "near_success"),
            "bucket_failure_or_recovery": np.bool_(bucket == "failure_or_recovery"),
        }
        steps.append(step)
    return steps


def _float_or_nan(value: Any) -> np.float32:
    """Convert an optional scalar into float32 or NaN."""
    if value is None:
        return np.float32(np.nan)
    return np.float32(value)

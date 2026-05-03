"""
Export E2 episode JSONs into training-ready npz files, adding human_distance.

Usage:
    python scripts/export_e2_with_human.py [--source-root ...] [--output-dir ...]

Output format (one npz per episode) matches training/dreamer_world_model/dataset.py:
    image                     (T, 60) float32  -- observation vector
    action                    (T, 2)  float32
    reward                    (T,)    float32
    is_first                  (T,)    bool
    is_last                   (T,)    bool
    is_terminal               (T,)    bool
    cost                      (T,)    float32
    speed                     (T,)    float32
    goal_distance             (T,)    float32
    nearest_hazard_distance   (T,)    float32
    nearest_vase_distance     (T,)    float32
    human_distance            (T,)    float32  ← NEW (computed from agent_pos)
    level                     (T,)    int32
    bucket_success            (T,)    bool
    bucket_near_success       (T,)    bool
    bucket_failure_or_recovery(T,)    bool

T = len(obs) + 1  (includes the initial observation step)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── configurable human position (x, y) ───────────────────────────────────────
# Place the human somewhere in the environment that the agent naturally passes
# through. SafetyPointGoal2-v0 is roughly a 6×6 arena centred at origin.
HUMAN_POS: np.ndarray = np.array([0.0, 2.0], dtype=np.float64)


def _human_dist(agent_pos_xy: np.ndarray) -> np.ndarray:
    """Euclidean distance from a (T, 2) array of positions to HUMAN_POS."""
    return np.linalg.norm(agent_pos_xy - HUMAN_POS, axis=1).astype(np.float32)


def _nan_fill(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with 0.0 (missing sensor readings)."""
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def export_episode(episode: dict, out_path: Path) -> int:
    """
    Convert one episode dict → npz, returning the number of steps written.
    """
    initial_obs = np.asarray(episode["initial_obs"], dtype=np.float32)
    obs_steps   = [np.asarray(o, dtype=np.float32) for o in episode["obs"]]
    obs_list    = [initial_obs] + obs_steps          # T+1 entries

    T_act = len(episode["action"])
    zero_act = np.zeros(2, dtype=np.float32)
    actions   = [zero_act] + [np.asarray(a, dtype=np.float32) for a in episode["action"]]

    rewards    = [0.0]    + [float(r) for r in episode["reward"]]
    costs      = [0.0]    + [float(c) for c in episode["cost"]]
    speeds     = [0.0]    + [float(s) for s in episode["speed"]]
    terminated = [False]  + [bool(v)  for v in episode["terminated"]]
    truncated  = [False]  + [bool(v)  for v in episode["truncated"]]

    def _to_f32_or_nan(lst):
        return np.array([np.nan if v is None else float(v) for v in lst],
                        dtype=np.float32)

    goal_dist   = _to_f32_or_nan([None] + list(episode["goal_distance"]))
    hazard_dist = _to_f32_or_nan([None] + list(episode["nearest_hazard_distance"]))
    vase_dist   = _to_f32_or_nan([None] + list(episode["nearest_vase_distance"]))

    # agent_pos has T entries (after each action); prepend first pos for t=0
    pos_raw = episode["agent_pos"]          # list of [x, y, z]
    pos_xy  = np.array([[p[0], p[1]] for p in pos_raw], dtype=np.float64)
    pos_xy_full = np.vstack([pos_xy[:1], pos_xy])   # (T+1, 2)
    human_dist  = _human_dist(pos_xy_full)           # (T+1,)

    T = len(obs_list)
    assert T == len(actions) == len(rewards), f"Length mismatch in {episode['episode_id']}"

    is_first    = np.zeros(T, dtype=bool); is_first[0]  = True
    is_last     = np.zeros(T, dtype=bool); is_last[-1]  = True
    is_terminal = np.array(terminated, dtype=bool)
    for i, (term, trunc) in enumerate(zip(terminated, truncated)):
        if term or trunc:
            is_last[i] = True

    bucket = episode["bucket_type"]
    level  = int(episode["level"])

    np.savez_compressed(
        out_path,
        image                      = np.stack(obs_list),           # (T, 60)
        action                     = np.stack(actions),            # (T, 2)
        reward                     = np.array(rewards,   dtype=np.float32),
        is_first                   = is_first,
        is_last                    = is_last,
        is_terminal                = is_terminal,
        cost                       = _nan_fill(np.array(costs,      dtype=np.float32)),
        speed                      = _nan_fill(np.array(speeds,     dtype=np.float32)),
        goal_distance              = _nan_fill(goal_dist),
        nearest_hazard_distance    = _nan_fill(hazard_dist),
        nearest_vase_distance      = _nan_fill(vase_dist),
        human_distance             = _nan_fill(human_dist),
        level                      = np.full(T, level, dtype=np.int32),
        bucket_success             = np.full(T, bucket == "success",             dtype=bool),
        bucket_near_success        = np.full(T, bucket == "near_success",        dtype=bool),
        bucket_failure_or_recovery = np.full(T, bucket == "failure_or_recovery", dtype=bool),
    )
    return T


def run(source_root: str, output_dir: str) -> None:
    src  = Path(source_root)
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = src / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    total_steps = 0
    written     = 0
    skipped     = 0

    for rec in manifest:
        raw_path = Path(rec["episode_path"])
        ep_path  = src / "episodes" / Path(*raw_path.parts[-3:])
        if not ep_path.exists():
            ep_path = raw_path   # absolute fallback
        if not ep_path.exists():
            print(f"  [SKIP] {ep_path.name} — file not found")
            skipped += 1
            continue

        episode  = json.loads(ep_path.read_text(encoding="utf-8"))
        out_path = out / f"{episode['episode_id']}.npz"
        steps    = export_episode(episode, out_path)
        total_steps += steps
        written     += 1

        if written % 100 == 0:
            print(f"  exported {written}/{len(manifest)} episodes …")

    print(f"\nDone. {written} episodes → {output_dir}")
    print(f"  skipped:     {skipped}")
    print(f"  total steps: {total_steps}")
    print(f"  human pos:   {HUMAN_POS.tolist()}")

    # save a small summary alongside the data
    summary = {
        "source_root":  str(src.resolve()),
        "output_dir":   str(out.resolve()),
        "human_pos":    HUMAN_POS.tolist(),
        "n_episodes":   written,
        "n_steps":      total_steps,
    }
    (out / "export_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="datasets/goal2_master/safeworld-goal2-master",
        help="Root of the master dataset (contains manifest.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/goal2_e2_human",
        help="Where to write the npz files",
    )
    args = parser.parse_args()
    run(args.source_root, args.output_dir)


if __name__ == "__main__":
    main()

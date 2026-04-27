#!/usr/bin/env python3
"""Inspect Safety Gymnasium environments for accessible state and metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import safety_gymnasium as gym


DEFAULT_ENVS = [
    "SafetyPointGoal1-v0",
    "SafetyPointGoal2-v0",
    "SafetyCarGoal1-v0",
    "SafetyPointButton1-v0",
]


def to_jsonable(value: Any, depth: int = 0) -> Any:
    """Convert nested values into a JSON-friendly structure."""
    if depth > 4:
        return f"<max_depth:{type(value).__name__}>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        flat = value.reshape(-1)
        preview = flat[: min(8, flat.size)].tolist()
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "preview": preview,
        }
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item, depth + 1) for item in value[:20]]
    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item, depth + 1)
            for key, item in list(value.items())[:50]
        }
    if hasattr(value, "__dict__"):
        keys = sorted(k for k in vars(value).keys() if not k.startswith("__"))
        return {
            "type": type(value).__name__,
            "attrs": {key: to_jsonable(getattr(value, key), depth + 1) for key in keys[:50]},
        }
    return repr(value)


def summarize_space(space: Any) -> dict[str, Any]:
    """Return a compact summary for a Gymnasium space."""
    summary = {"repr": repr(space), "type": type(space).__name__}
    for attr_name in ("shape", "dtype", "n"):
        if hasattr(space, attr_name):
            summary[attr_name] = to_jsonable(getattr(space, attr_name))
    if hasattr(space, "low"):
        summary["low"] = to_jsonable(space.low)
    if hasattr(space, "high"):
        summary["high"] = to_jsonable(space.high)
    return summary


def collect_named_attrs(obj: Any, names: list[str]) -> dict[str, Any]:
    """Collect selected attributes when present."""
    collected = {}
    for name in names:
        if hasattr(obj, name):
            try:
                collected[name] = to_jsonable(getattr(obj, name))
            except Exception as exc:  # pragma: no cover - inspection only
                collected[name] = f"<error:{exc}>"
    return collected


def inspect_env(env_id: str, seed: int, steps: int) -> dict[str, Any]:
    """Inspect one environment and return a structured report."""
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        obs, info = env.reset(seed=seed)
        report: dict[str, Any] = {
            "env_id": env_id,
            "seed": seed,
            "observation_space": summarize_space(env.observation_space),
            "action_space": summarize_space(env.action_space),
            "reset_obs": to_jsonable(obs),
            "reset_info": to_jsonable(info),
            "top_level_env_type": type(env).__name__,
            "unwrapped_type": type(env.unwrapped).__name__,
            "unwrapped_attrs": collect_named_attrs(
                env.unwrapped,
                [
                    "agent",
                    "goal",
                    "goals",
                    "hazards",
                    "buttons",
                    "boxs",
                    "pillars",
                    "gremlins",
                    "humans",
                    "placements_conf",
                    "task",
                    "robot",
                    "world_info",
                ],
            ),
            "dict_keys": sorted(list(vars(env.unwrapped).keys())),
            "sample_steps": [],
        }

        for t in range(steps):
            action = env.action_space.sample()
            obs, reward, cost, terminated, truncated, info = env.step(action)
            step_entry = {
                "t": t,
                "reward": to_jsonable(reward),
                "cost": to_jsonable(cost),
                "terminated": terminated,
                "truncated": truncated,
                "obs": to_jsonable(obs),
                "info": to_jsonable(info),
            }
            if t == 0:
                step_entry["post_step_unwrapped_attrs"] = collect_named_attrs(
                    env.unwrapped,
                    [
                        "agent",
                        "goal",
                        "goals",
                        "hazards",
                        "buttons",
                        "boxs",
                        "pillars",
                        "gremlins",
                        "humans",
                        "robot",
                    ],
                )
            report["sample_steps"].append(step_entry)
            if terminated or truncated:
                obs, info = env.reset(seed=seed + t + 1)
                report.setdefault("extra_resets", []).append(
                    {"after_t": t, "obs": to_jsonable(obs), "info": to_jsonable(info)},
                )
        return report
    finally:
        env.close()


def main() -> None:
    """CLI entry point for environment inspection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", nargs="*", default=DEFAULT_ENVS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "env_inspection",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for env_id in args.env_id:
        report = inspect_env(env_id, seed=args.seed, steps=args.steps)
        out_path = args.output_dir / f"{env_id}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary.append(
            {
                "env_id": env_id,
                "out_path": str(out_path.resolve()),
                "unwrapped_type": report["unwrapped_type"],
                "reset_info_keys": sorted(report["reset_info"].keys())
                if isinstance(report["reset_info"], dict)
                else [],
                "step0_info_keys": sorted(report["sample_steps"][0]["info"].keys())
                if report["sample_steps"] and isinstance(report["sample_steps"][0]["info"], dict)
                else [],
            },
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

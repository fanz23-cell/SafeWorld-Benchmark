from __future__ import annotations

import json
import traceback
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import safety_gymnasium as gym


ENV_IDS = [
    "SafetyPointGoal1-v0",
    "SafetyPointGoal2-v0",
    "SafetyCarGoal1-v0",
    "SafetyPointButton1-v0",
]

ROLLOUT_STEPS = 20
SCREENSHOT_STEPS = {0, 5, 10, 15}
OUTPUT_DIR = Path(__file__).resolve().parent / "safety_outputs"


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def save_frame(env_id: str, step_idx: int, frame: np.ndarray) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_path = OUTPUT_DIR / f"{sanitize_name(env_id)}_step{step_idx:03d}.png"
    iio.imwrite(frame_path, frame)
    return str(frame_path)


def try_make_env(env_id: str, render_mode: str):
    return gym.make(env_id, render_mode=render_mode)


def run_one(env_id: str) -> dict:
    result: dict = {
        "env_id": env_id,
        "created": False,
        "reset_ok": False,
        "step_ok": False,
        "render_ok": False,
        "render_mode_used": None,
        "observation_space": None,
        "action_space": None,
        "screenshots": [],
        "errors": [],
    }

    env = None
    render_mode_used = None
    try:
        try:
            env = try_make_env(env_id, "human")
            render_mode_used = "human"
        except Exception as human_exc:
            result["errors"].append(f"human make failed: {human_exc}")
            env = try_make_env(env_id, "rgb_array")
            render_mode_used = "rgb_array"

        result["created"] = True
        result["render_mode_used"] = render_mode_used
        result["observation_space"] = str(env.observation_space)
        result["action_space"] = str(env.action_space)

        obs, info = env.reset(seed=0)
        _ = info
        result["reset_ok"] = obs is not None

        for step_idx in range(ROLLOUT_STEPS):
            action = env.action_space.sample()
            obs, reward, cost, terminated, truncated, info = env.step(action)
            _ = (obs, reward, cost, info)
            result["step_ok"] = True

            try:
                frame = env.render()
                if render_mode_used == "rgb_array":
                    if isinstance(frame, np.ndarray):
                        result["render_ok"] = True
                        if step_idx in SCREENSHOT_STEPS:
                            result["screenshots"].append(save_frame(env_id, step_idx, frame))
                    elif frame is not None:
                        result["render_ok"] = True
                else:
                    result["render_ok"] = True
            except Exception as render_exc:
                if render_mode_used == "human":
                    result["errors"].append(f"human render failed at step {step_idx}: {render_exc}")
                    env.close()
                    env = try_make_env(env_id, "rgb_array")
                    render_mode_used = "rgb_array"
                    result["render_mode_used"] = render_mode_used
                    obs, info = env.reset(seed=0)
                    _ = (obs, info)
                    continue
                result["errors"].append(f"rgb_array render failed at step {step_idx}: {render_exc}")

            if terminated or truncated:
                obs, info = env.reset()
                _ = (obs, info)

        if not result["screenshots"]:
            capture_env = None
            try:
                capture_env = try_make_env(env_id, "rgb_array")
                obs, info = capture_env.reset(seed=123)
                _ = (obs, info)
                for step_idx in range(ROLLOUT_STEPS):
                    frame = capture_env.render()
                    if isinstance(frame, np.ndarray) and step_idx in SCREENSHOT_STEPS:
                        result["screenshots"].append(save_frame(env_id, step_idx, frame))
                    obs, reward, cost, terminated, truncated, info = capture_env.step(
                        capture_env.action_space.sample(),
                    )
                    _ = (obs, reward, cost, info)
                    if terminated or truncated:
                        obs, info = capture_env.reset()
                        _ = (obs, info)
            except Exception as capture_exc:
                result["errors"].append(f"supplemental rgb_array capture failed: {capture_exc}")
            finally:
                if capture_env is not None:
                    capture_env.close()

    except Exception as exc:
        result["errors"].append(f"fatal: {exc}")
        result["errors"].append(traceback.format_exc())
    finally:
        if env is not None:
            env.close()

    return result


def print_human_summary(result: dict) -> None:
    print(f"\n=== {result['env_id']} ===")
    print(f"created: {result['created']}")
    print(f"observation_space: {result['observation_space']}")
    print(f"action_space: {result['action_space']}")
    print(f"reset_ok: {result['reset_ok']}")
    print(f"step_ok: {result['step_ok']}")
    print(f"render_ok: {result['render_ok']}")
    print(f"render_mode_used: {result['render_mode_used']}")
    if result["screenshots"]:
        print("screenshots:")
        for path in result["screenshots"]:
            print(f"  - {path}")
    if result["errors"]:
        print("errors:")
        for err in result["errors"]:
            print(f"  - {err}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_one(env_id) for env_id in ENV_IDS]
    for result in results:
        print_human_summary(result)

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSummary JSON: {summary_path}")


if __name__ == "__main__":
    main()

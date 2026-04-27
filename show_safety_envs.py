from __future__ import annotations

import time

import safety_gymnasium as gym


ENV_IDS = [
    "SafetyPointGoal1-v0",
    "SafetyPointGoal2-v0",
    "SafetyCarGoal1-v0",
    "SafetyPointButton1-v0",
]

STEPS_PER_ENV = 600
STEP_SLEEP_SECONDS = 0.03


def play_env(env_id: str) -> None:
    print(f"\nOpening {env_id} ...", flush=True)
    env = gym.make(env_id, render_mode="human")
    try:
        obs, info = env.reset(seed=0)
        _ = (obs, info)
        for _step in range(STEPS_PER_ENV):
            action = env.action_space.sample()
            obs, reward, cost, terminated, truncated, info = env.step(action)
            _ = (obs, reward, cost, info)
            env.render()
            time.sleep(STEP_SLEEP_SECONDS)
            if terminated or truncated:
                obs, info = env.reset()
                _ = (obs, info)
    finally:
        env.close()
        print(f"Closed {env_id}", flush=True)


def main() -> None:
    print("Showing Safety Gymnasium environments in sequence.", flush=True)
    print(
        f"Each env will stay open for about {STEPS_PER_ENV * STEP_SLEEP_SECONDS:.1f} seconds.",
        flush=True,
    )
    for env_id in ENV_IDS:
        play_env(env_id)
        time.sleep(1.0)
    print("Finished showing all environments.", flush=True)


if __name__ == "__main__":
    main()

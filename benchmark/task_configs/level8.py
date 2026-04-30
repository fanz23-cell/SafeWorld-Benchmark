"""Level 8 benchmark task configs."""

from __future__ import annotations

from benchmark.task_types import TaskConfig, ZoneDefinition


def get_level8_tasks() -> list[TaskConfig]:
    """Return all level 8 benchmark tasks."""
    return [
        TaskConfig(
            task_id="E2_L8_FullMission",
            level=8,
            paper_spec_name="Full Mission",
            paper_formula_str="♢(A ∧ ♢(B)) ∧ □(♢(C)) ∧ □(¬hazard) ∧ □(near_obs → ♢(¬fast))",
            env_id="SafetyPointGoal2-v0",
            horizon=240,
            required_aps=["A", "B", "C", "hazard", "near_obs", "fast"],
            description="Compositional full mission on Goal2: sequencing, patrol, safety, and reactive slowdown.",
            ap_params={
                "fast_threshold": 0.35,
                "near_obs_threshold": 0.30,
                "near_obs_grounding": "vase",
                "bounded_semantics": "windowed_eventually",
                "patrol_window_fraction": 0.5,
            },
            zone_defs=[
                ZoneDefinition(
                    name="A",
                    kind="interpolated_zone",
                    radius=0.22,
                    interpolation=1.0 / 3.0,
                    anchor="goal",
                    description="First sequencing zone at one-third of the start-goal line.",
                ),
                ZoneDefinition(
                    name="B",
                    kind="goal_region",
                    radius=0.30,
                    description="Native goal region used as sequencing target.",
                ),
                ZoneDefinition(
                    name="C",
                    kind="interpolated_zone",
                    radius=0.22,
                    interpolation=0.0,
                    anchor="goal",
                    description="Patrol return zone centered at the rollout start position.",
                ),
            ],
            notes=[
                "Paper formula kept unchanged.",
                "Goal2 is preferred here because it more naturally supports multi-zone mission structure plus denser hazards and vases.",
                "Evaluator decomposes the mission into sequencing, recurrence, safety, and reactive sub-checks.",
            ],
        ),
    ]

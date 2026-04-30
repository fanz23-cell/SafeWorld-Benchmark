"""Level 6 benchmark task configs."""

from __future__ import annotations

from benchmark.task_types import TaskConfig, ZoneDefinition


def get_level6_tasks() -> list[TaskConfig]:
    """Return all level 6 benchmark tasks."""
    return [
        TaskConfig(
            task_id="E1_L6_SafePatrol",
            level=6,
            paper_spec_name="Safe Patrol",
            paper_formula_str="□(♢(A)) ∧ □(¬hazard)",
            env_id="SafetyPointGoal1-v0",
            horizon=220,
            required_aps=["A", "hazard"],
            description="Patrol one explicit zone while maintaining hazard avoidance in Goal1.",
            ap_params={
                "bounded_semantics": "windowed_eventually",
                "patrol_window_fraction": 0.5,
            },
            zone_defs=[
                ZoneDefinition(
                    name="A",
                    kind="interpolated_zone",
                    radius=0.25,
                    interpolation=0.5,
                    anchor="goal",
                    description="Patrol zone at midpoint from start to goal.",
                ),
            ],
            notes=[
                "Paper formula kept unchanged.",
                "Recurrence conjunct uses bounded windowed eventually approximation.",
            ],
        ),
        TaskConfig(
            task_id="E2_L6_SafeReactiveGoal",
            level=6,
            paper_spec_name="Safe Reactive Goal",
            paper_formula_str="♢(goal) ∧ □(¬hazard) ∧ □(near_obs → ♢(¬fast))",
            env_id="SafetyPointGoal2-v0",
            horizon=220,
            required_aps=["goal", "hazard", "near_obs", "fast"],
            description="Reach the goal in Goal2 while avoiding hazards and slowing after obstacle proximity.",
            ap_params={
                "fast_threshold": 0.35,
                "near_obs_threshold": 0.30,
                "near_obs_grounding": "vase",
            },
            notes=[
                "Paper formula kept unchanged.",
                "Implements the conjunction of L2-style reachability/safety and L4-style reactive slowdown.",
            ],
        ),
    ]

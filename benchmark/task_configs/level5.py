"""Level 5 benchmark task configs."""

from __future__ import annotations

from benchmark.task_types import TaskConfig, ZoneDefinition


def get_level5_tasks() -> list[TaskConfig]:
    """Return all level 5 benchmark tasks."""
    return [
        TaskConfig(
            task_id="E1_L5_Patrol",
            level=5,
            paper_spec_name="Patrol",
            paper_formula_str="□(♢(A))",
            env_id="SafetyPointGoal1-v0",
            horizon=200,
            required_aps=["A"],
            description="Bounded patrol approximation over a single explicit patrol zone in Goal1.",
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
                "Finite-trace approximation uses a rolling eventually window of T/2, per Table 3 note.",
            ],
        ),
        TaskConfig(
            task_id="E2_L5_DualPatrol",
            level=5,
            paper_spec_name="Dual Patrol",
            paper_formula_str="□(♢(A)) ∧ □(♢(B))",
            env_id="SafetyPointGoal2-v0",
            horizon=900,
            required_aps=["A", "B"],
            description="Bounded dual patrol approximation over two explicit zones in Goal2.",
            ap_params={
                "bounded_semantics": "windowed_eventually",
                "patrol_window_fraction": 0.5,
            },
            zone_defs=[
                ZoneDefinition(
                    name="A",
                    kind="interpolated_zone",
                    radius=0.25,
                    interpolation=1.0 / 3.0,
                    anchor="goal",
                    description="First patrol zone at one-third of the start-goal line.",
                ),
                ZoneDefinition(
                    name="B",
                    kind="interpolated_zone",
                    radius=0.25,
                    interpolation=2.0 / 3.0,
                    anchor="goal",
                    description="Second patrol zone at two-thirds of the start-goal line.",
                ),
            ],
            notes=[
                "Paper formula kept unchanged.",
                "Finite-trace approximation uses a rolling eventually window of T/2, per Table 3 note.",
            ],
        ),
    ]

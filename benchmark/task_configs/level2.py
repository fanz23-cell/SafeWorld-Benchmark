"""Level 2 benchmark task configs."""

from __future__ import annotations

from benchmark.task_types import TaskConfig


def get_level2_tasks() -> list[TaskConfig]:
    """Return all level 2 benchmark tasks."""
    return [
        TaskConfig(
            task_id="E1_L2_SafeGoal",
            level=2,
            paper_spec_name="Safe Goal",
            paper_formula_str="♢(goal) ∧ □(¬hazard)",
            env_id="SafetyPointGoal1-v0",
            horizon=150,
            required_aps=["goal", "hazard"],
            description="Reach the Goal1 target while never entering a hazard.",
            notes=["Paper formula kept unchanged."],
        ),
        TaskConfig(
            task_id="E2_L2_SafeSlowGoal",
            level=2,
            paper_spec_name="Safe Slow Goal",
            paper_formula_str="♢(goal) ∧ □(¬hazard) ∧ □(¬fast)",
            env_id="SafetyPointGoal2-v0",
            horizon=150,
            required_aps=["goal", "hazard", "fast"],
            description="Reach the Goal2 target while avoiding hazards and staying slow.",
            ap_params={"fast_threshold": 0.35, "speed_agent_type": "point"},
            notes=["Paper formula kept unchanged."],
        ),
        TaskConfig(
            task_id="E3_L2_SafeSlowGoal",
            level=2,
            paper_spec_name="Safe Slow Goal",
            paper_formula_str="♢(goal) ∧ □(¬hazard) ∧ □(¬fast)",
            env_id="SafetyCarGoal1-v0",
            horizon=150,
            required_aps=["goal", "hazard", "fast"],
            description="Reach the CarGoal1 target while avoiding hazards and staying slow.",
            ap_params={"fast_threshold": 0.30, "speed_agent_type": "car"},
            notes=["Paper formula kept unchanged."],
        ),
        TaskConfig(
            task_id="E4_L2_SafeGoal_Button",
            level=2,
            paper_spec_name="Safe Goal",
            paper_formula_str="♢(goal) ∧ □(¬hazard)",
            env_id="SafetyPointButton1-v0",
            horizon=150,
            required_aps=["goal", "hazard"],
            description="Activate the target button while avoiding hazards.",
            notes=[
                "goal is grounded by task.goal_achieved as confirmed.",
                "Target button distance is also recorded for debugging.",
            ],
        ),
    ]

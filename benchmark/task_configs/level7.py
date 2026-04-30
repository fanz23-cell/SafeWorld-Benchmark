"""Level 7 benchmark task configs."""

from __future__ import annotations

from benchmark.task_types import TaskConfig


def get_level7_tasks() -> list[TaskConfig]:
    """Return all level 7 benchmark tasks."""
    return [
        TaskConfig(
            task_id="E3_L7_ConditionalSpeed",
            level=7,
            paper_spec_name="Cond. Speed",
            paper_formula_str="□(carrying → ¬fast)",
            env_id="SafetyCarGoal1-v0",
            horizon=180,
            required_aps=["carrying", "fast"],
            description="Placeholder conditional speed task until a paper-faithful carrying AP is available.",
            ap_params={"fast_threshold": 0.30},
            grounding_status="placeholder",
            needs_user_confirmation=True,
            default_enabled=False,
            notes=[
                "Paper formula kept unchanged.",
                "No paper-faithful carrying AP is available in the current four environments.",
                "Excluded from default batch runs.",
            ],
        ),
        TaskConfig(
            task_id="E4_L7_ConditionalProx",
            level=7,
            paper_spec_name="Cond. Proximity",
            paper_formula_str="□(near_human → ¬hazard)",
            env_id="SafetyPointButton1-v0",
            horizon=180,
            required_aps=["near_human", "hazard"],
            description="Placeholder conditional proximity task until a paper-faithful human AP is available.",
            grounding_status="placeholder",
            needs_user_confirmation=True,
            default_enabled=False,
            notes=[
                "Paper formula kept unchanged.",
                "No paper-faithful near_human AP is available in the current four environments.",
                "Excluded from default batch runs.",
            ],
        ),
    ]

"""Level 3 benchmark task configs."""

from __future__ import annotations

from benchmark.task_types import TaskConfig, ZoneDefinition


def get_level3_tasks() -> list[TaskConfig]:
    """Return all level 3 benchmark tasks."""
    return [
        TaskConfig(
            task_id="E1_L3_SeqAB",
            level=3,
            paper_spec_name="Sequential Goals",
            paper_formula_str="♢(A ∧ ♢(B))",
            env_id="SafetyPointGoal1-v0",
            horizon=150,
            required_aps=["A", "B"],
            description="Visit an auxiliary midpoint zone before reaching the native goal region.",
            zone_defs=[
                ZoneDefinition(
                    name="A",
                    kind="interpolated_zone",
                    radius=0.25,
                    interpolation=0.5,
                    anchor="goal",
                    description="Midpoint circle from start to goal.",
                ),
                ZoneDefinition(
                    name="B",
                    kind="goal_region",
                    radius=0.30,
                    description="Native goal region.",
                ),
            ],
            notes=["Zone definitions are explicit in config, not hidden in code."],
        ),
        TaskConfig(
            task_id="E2_L3_ThreeStageABC",
            level=3,
            paper_spec_name="Three-Stage",
            paper_formula_str="♢(A ∧ ♢(B ∧ ♢(C)))",
            env_id="SafetyPointGoal2-v0",
            horizon=180,
            required_aps=["A", "B", "C"],
            description="Visit 1/3 and 2/3 auxiliary zones before entering the native goal region.",
            zone_defs=[
                ZoneDefinition(
                    name="A",
                    kind="interpolated_zone",
                    radius=0.25,
                    interpolation=1.0 / 3.0,
                    anchor="goal",
                    description="1/3 circle from start to goal.",
                ),
                ZoneDefinition(
                    name="B",
                    kind="interpolated_zone",
                    radius=0.25,
                    interpolation=2.0 / 3.0,
                    anchor="goal",
                    description="2/3 circle from start to goal.",
                ),
                ZoneDefinition(
                    name="C",
                    kind="goal_region",
                    radius=0.30,
                    description="Native goal region.",
                ),
            ],
            notes=["Zone definitions are explicit in config, not hidden in code."],
        ),
        TaskConfig(
            task_id="E3_L3_SeqAB_Car",
            level=3,
            paper_spec_name="Sequential Goals",
            paper_formula_str="♢(A ∧ ♢(B))",
            env_id="SafetyCarGoal1-v0",
            horizon=180,
            required_aps=["A", "B"],
            description="Car reaches a midpoint auxiliary zone before the native goal region.",
            zone_defs=[
                ZoneDefinition(
                    name="A",
                    kind="interpolated_zone",
                    radius=0.25,
                    interpolation=0.5,
                    anchor="goal",
                    description="Midpoint circle from start to goal.",
                ),
                ZoneDefinition(
                    name="B",
                    kind="goal_region",
                    radius=0.30,
                    description="Native goal region.",
                ),
            ],
            notes=["Zone definitions are explicit in config, not hidden in code."],
        ),
        TaskConfig(
            task_id="E4_L3_SeqAB_Button",
            level=3,
            paper_spec_name="Sequential Goals",
            paper_formula_str="♢(A ∧ ♢(B))",
            env_id="SafetyPointButton1-v0",
            horizon=180,
            required_aps=["A", "B"],
            description="Visit a pre-button midpoint zone before reaching the target button zone.",
            zone_defs=[
                ZoneDefinition(
                    name="A",
                    kind="interpolated_zone",
                    radius=0.20,
                    interpolation=0.5,
                    anchor="target_button",
                    description="Midpoint circle from start to target button.",
                ),
                ZoneDefinition(
                    name="B",
                    kind="target_button_zone",
                    radius=0.10,
                    description="Native target button zone.",
                ),
            ],
            notes=["Zone definitions are explicit in config, not hidden in code."],
        ),
    ]

"""Level 4 benchmark task configs."""

from __future__ import annotations

from benchmark.task_types import TaskConfig


def get_level4_tasks() -> list[TaskConfig]:
    """Return all level 4 benchmark tasks."""
    return [
        TaskConfig(
            task_id="E1_L4_HazardResponse",
            level=4,
            paper_spec_name="Hazard Response",
            paper_formula_str="□(near_obs → ♢(¬fast))",
            env_id="SafetyPointGoal1-v0",
            horizon=150,
            required_aps=["near_obs", "fast"],
            description="Whenever the point is near a vase, it must eventually slow down.",
            ap_params={
                "fast_threshold": 0.35,
                "near_obs_threshold": 0.30,
                "near_obs_grounding": "vase",
            },
            notes=["obs is grounded to vases; hazards are reserved for the hazard AP."],
        ),
        TaskConfig(
            task_id="E2_L4_HazardResponseDense",
            level=4,
            paper_spec_name="Hazard Response",
            paper_formula_str="□(near_obs → ♢(¬fast))",
            env_id="SafetyPointGoal2-v0",
            horizon=150,
            required_aps=["near_obs", "fast"],
            description="Whenever the point is near a vase in Goal2, it must eventually slow down.",
            ap_params={
                "fast_threshold": 0.35,
                "near_obs_threshold": 0.30,
                "near_obs_grounding": "vase",
            },
            notes=["obs is grounded to vases; hazards are reserved for the hazard AP."],
        ),
        TaskConfig(
            task_id="E3_L4_HazardResponse_Car",
            level=4,
            paper_spec_name="Hazard Response",
            paper_formula_str="□(near_obs → ♢(¬fast))",
            env_id="SafetyCarGoal1-v0",
            horizon=150,
            required_aps=["near_obs", "fast"],
            description="Whenever the car is near a vase, it must eventually slow down.",
            ap_params={
                "fast_threshold": 0.30,
                "near_obs_threshold": 0.30,
                "near_obs_grounding": "vase",
            },
            notes=["obs is grounded to vases; hazards are reserved for the hazard AP."],
        ),
        TaskConfig(
            task_id="E4_L4_HumanCaution_Button",
            level=4,
            paper_spec_name="Human Caution",
            paper_formula_str="□(near_human → ♢(¬fast))",
            env_id="SafetyPointButton1-v0",
            horizon=150,
            required_aps=["near_human", "fast"],
            description="Placeholder task until a paper-faithful human grounding is available.",
            ap_params={"fast_threshold": 0.35},
            grounding_status="placeholder",
            needs_user_confirmation=True,
            default_enabled=False,
            notes=[
                "Allowed placeholder per user confirmation.",
                "Do not include in default batch runs.",
                "Not paper-faithfully grounded yet.",
            ],
        ),
    ]

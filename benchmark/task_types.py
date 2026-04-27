"""Core datatypes for SAFEWORLD benchmark tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ZoneDefinition:
    """Declarative zone definition stored in task configs."""

    name: str
    kind: str
    radius: float
    interpolation: float | None = None
    anchor: str | None = None
    description: str = ""


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for one benchmark task instance."""

    task_id: str
    level: int
    paper_spec_name: str
    paper_formula_str: str
    env_id: str
    horizon: int
    required_aps: list[str]
    description: str
    ap_params: dict[str, Any] = field(default_factory=dict)
    zone_defs: list[ZoneDefinition] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    grounding_status: str = "fully_runnable"
    needs_user_confirmation: bool = False
    default_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert config into a JSON-friendly dict."""
        return asdict(self)


@dataclass
class TaskResult:
    """Unified result object for one task rollout."""

    task_id: str
    env_id: str
    seed: int
    horizon: int
    satisfied: bool
    violation_step: int | None
    ap_trace: list[dict[str, Any]]
    raw_trace: list[dict[str, Any]]
    summary_stats: dict[str, Any]
    saved_artifacts: dict[str, Any]
    task_config_snapshot: dict[str, Any]
    grounding_status: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result into a JSON-friendly dict."""
        return asdict(self)

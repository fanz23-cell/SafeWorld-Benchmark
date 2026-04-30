"""Task-aware scripted controllers for Goal2 dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from benchmark import env_utils
from benchmark.geometry_utils import distance_xy
from benchmark.task_types import TaskConfig


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Return the Euclidean distance from one point to a 2D line segment."""
    point = np.asarray(point, dtype=float)[:2]
    start = np.asarray(start, dtype=float)[:2]
    end = np.asarray(end, dtype=float)[:2]
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom < 1e-8:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


@dataclass
class OracleControllerState:
    """Mutable controller state for one rollout."""

    phase: str
    visited_targets: list[str] = field(default_factory=list)
    current_patrol_target: str | None = None
    patrol_dwell_steps: int = 0
    patrol_hold_steps: int = 0
    patrol_steps_on_target: int = 0
    patrol_last_visit: dict = field(default_factory=dict)
    slow_timer: int = 0
    response_triggered: bool = False
    forced_failures: int = 0
    target_vase_index: int | None = None


class Goal2OracleController:
    """Geometric scripted controller tailored to one Goal2 benchmark task."""

    def __init__(
        self,
        task_config: TaskConfig,
        bucket_type: str,
        rng: np.random.Generator,
        slow_mode_steps: int = 8,
        near_success_noise_std: float = 0.08,
        patrol_switch_dwell_steps: int = 2,
    ) -> None:
        self.task_config = task_config
        self.bucket_type = bucket_type
        self.rng = rng
        self.slow_mode_steps = slow_mode_steps
        self.near_success_noise_std = near_success_noise_std
        self.patrol_switch_dwell_steps = patrol_switch_dwell_steps
        self.fast_threshold = float(task_config.ap_params.get("fast_threshold", 0.35))
        self.near_obs_threshold = float(task_config.ap_params.get("near_obs_threshold", 0.30))
        self.state = OracleControllerState(phase=self._initial_phase())

    def reset(self, env, resolved_zones: dict[str, dict[str, Any]]) -> None:
        """Reset controller state for a new episode."""
        self.resolved_zones = resolved_zones
        self.start_pos = env_utils.get_agent_position(env)[:2].copy()
        self.goal_pos = env_utils.get_goal_position(env)[:2].copy()
        self.state = OracleControllerState(phase=self._initial_phase())
        if self.task_config.task_id == "E2_L5_DualPatrol":
            self.state.current_patrol_target = "A"
        if self.task_config.task_id == "E2_L8_FullMission":
            self.state.current_patrol_target = "C"

    def close(self) -> None:
        """Close planner resources when present."""
        return

    def act(self, env, step_idx: int) -> np.ndarray:
        """Return one controlled action."""
        agent_pos = env_utils.get_agent_position(env)[:2]
        agent_vel = env_utils.get_agent_velocity(env)[:2]

        if self.bucket_type == "failure_or_recovery":
            return self._failure_action(env, agent_pos, agent_vel, step_idx)

        if self.bucket_type == "near_success":
            action = self._success_action(env, agent_pos, agent_vel, step_idx)
            return self._perturb_action(action)

        return self._success_action(env, agent_pos, agent_vel, step_idx)

    def _initial_phase(self) -> str:
        mapping = {
            "E2_L1_SpeedLimit": "wander",
            "E2_L2_SafeSlowGoal": "goal",
            "E2_L3_ThreeStageABC": "A",
            "E2_L4_HazardResponseDense": "approach_vase",
            "E2_L5_DualPatrol": "patrol",
            "E2_L6_SafeReactiveGoal": "approach_vase",
            "E2_L8_FullMission": "A",
        }
        return mapping[self.task_config.task_id]

    def _success_action(
        self,
        env,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        step_idx: int,
    ) -> np.ndarray:
        """Return the success-oriented action for the current task."""
        task_id = self.task_config.task_id
        if task_id == "E2_L1_SpeedLimit":
            return self._speed_limit_action(agent_pos, agent_vel, step_idx, failure=False)
        if task_id == "E2_L2_SafeSlowGoal":
            return self._go_to_goal_action(agent_pos, agent_vel, obey_speed=True)
        if task_id == "E2_L3_ThreeStageABC":
            return self._sequential_action(agent_pos, agent_vel, ["A", "B", "C"])
        if task_id == "E2_L4_HazardResponseDense":
            return self._reactive_vase_action(env, agent_pos, agent_vel, continue_to_goal=False)
        if task_id == "E2_L5_DualPatrol":
            return self._dual_patrol_action(agent_pos, agent_vel, step_idx)
        if task_id == "E2_L6_SafeReactiveGoal":
            return self._reactive_vase_action(env, agent_pos, agent_vel, continue_to_goal=True)
        if task_id == "E2_L8_FullMission":
            return self._full_mission_action(env, agent_pos, agent_vel)
        raise KeyError(f"Unsupported Goal2 task: {task_id}")

    def _failure_action(
        self,
        env,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        step_idx: int,
    ) -> np.ndarray:
        """Return a controlled failure or recovery action."""
        task_id = self.task_config.task_id
        if task_id == "E2_L1_SpeedLimit":
            return self._speed_limit_action(agent_pos, agent_vel, step_idx, failure=True)
        if task_id == "E2_L2_SafeSlowGoal":
            if step_idx < max(10, self.task_config.horizon // 3):
                return self._compose_action(
                    agent_pos,
                    agent_vel,
                    self.goal_pos,
                    speed_limit=self.fast_threshold * 1.25,
                    use_hazard_avoidance=False,
                    use_vase_avoidance=False,
                )
            return self._go_to_goal_action(agent_pos, agent_vel, obey_speed=True)
        if task_id == "E2_L3_ThreeStageABC":
            return self._sequential_action(agent_pos, agent_vel, ["B", "A", "C"])
        if task_id == "E2_L4_HazardResponseDense":
            return self._reactive_vase_action(env, agent_pos, agent_vel, continue_to_goal=False, violate_response=True)
        if task_id == "E2_L5_DualPatrol":
            return self._patrol_single_zone_action(agent_pos, agent_vel, "A")
        if task_id == "E2_L6_SafeReactiveGoal":
            return self._reactive_vase_action(env, agent_pos, agent_vel, continue_to_goal=True, violate_response=True)
        if task_id == "E2_L8_FullMission":
            return self._full_mission_action(env, agent_pos, agent_vel, skip_final_patrol=True)
        raise KeyError(f"Unsupported Goal2 task: {task_id}")

    def _speed_limit_action(
        self,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        step_idx: int,
        failure: bool,
    ) -> np.ndarray:
        """Low-speed wander or intentional overspeed."""
        radius = 0.35
        angle = 0.15 * step_idx
        if failure:
            radius = 0.75
            angle = 0.35 * step_idx
        waypoint = self.start_pos + radius * np.asarray([np.cos(angle), np.sin(angle)])
        speed_limit = self.fast_threshold * (1.2 if failure else 0.55)
        return self._compose_action(
            agent_pos,
            agent_vel,
            waypoint,
            speed_limit=speed_limit,
            use_hazard_avoidance=True,
            use_vase_avoidance=True,
        )

    def _go_to_goal_action(
        self,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        obey_speed: bool,
    ) -> np.ndarray:
        """Drive toward the native goal with safety constraints."""
        speed_limit = self.fast_threshold * (0.7 if obey_speed else 1.25)
        return self._compose_action(
            agent_pos,
            agent_vel,
            self.goal_pos,
            speed_limit=speed_limit,
            use_hazard_avoidance=True,
            use_vase_avoidance=False,
        )

    def _sequential_action(
        self,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        order: list[str],
    ) -> np.ndarray:
        """Follow A/B/C style waypoint sequences from task config."""
        current_name = order[min(len(self.state.visited_targets), len(order) - 1)]
        target_zone = self.resolved_zones[current_name]
        target_center = np.asarray(target_zone["center"], dtype=float)
        if distance_xy(agent_pos, target_center) <= float(target_zone["radius"]):
            if not self.state.visited_targets or self.state.visited_targets[-1] != current_name:
                self.state.visited_targets.append(current_name)
            next_index = min(len(self.state.visited_targets), len(order) - 1)
            current_name = order[next_index]
            target_zone = self.resolved_zones[current_name]
            target_center = np.asarray(target_zone["center"], dtype=float)
        return self._compose_action(
            agent_pos,
            agent_vel,
            target_center,
            speed_limit=self.fast_threshold * 0.72,
            use_hazard_avoidance=True,
            use_vase_avoidance=True,
        )

    def _reactive_vase_action(
        self,
        env,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        continue_to_goal: bool,
        violate_response: bool = False,
    ) -> np.ndarray:
        """Approach a vase, trigger near_obs, then slow down, then continue."""
        vase_positions = env_utils.get_vase_positions(env)
        if self.state.target_vase_index is None:
            self.state.target_vase_index = self._select_reactive_vase_index(
                agent_pos,
                vase_positions,
                continue_to_goal=continue_to_goal,
            )
        target_vase = np.asarray(vase_positions[self.state.target_vase_index], dtype=float)[:2]
        distance_to_vase = distance_xy(agent_pos, target_vase)

        if self.state.phase == "approach_vase":
            if distance_to_vase < self.near_obs_threshold * 0.95:
                self.state.phase = "slow_recover"
                self.state.slow_timer = self.slow_mode_steps
                self.state.response_triggered = True
            else:
                target = target_vase
                return self._compose_action(
                    agent_pos,
                    agent_vel,
                    target,
                    speed_limit=self.fast_threshold * 0.9,
                    use_hazard_avoidance=True,
                    use_vase_avoidance=False,
                )

        if self.state.phase == "slow_recover":
            self.state.slow_timer = max(0, self.state.slow_timer - 1)
            if self.state.slow_timer == 0:
                self.state.phase = "goal" if continue_to_goal else "wander"
            speed_limit = self.fast_threshold * (1.15 if violate_response else 0.45)
            target = self.goal_pos if continue_to_goal else self.start_pos
            return self._compose_action(
                agent_pos,
                agent_vel,
                target,
                speed_limit=speed_limit,
                use_hazard_avoidance=True,
                use_vase_avoidance=not violate_response,
            )

        if self.state.phase == "goal":
            return self._go_to_goal_action(agent_pos, agent_vel, obey_speed=True)

        return self._compose_action(
            agent_pos,
            agent_vel,
            self.start_pos,
            speed_limit=self.fast_threshold * 0.55,
            use_hazard_avoidance=True,
            use_vase_avoidance=True,
        )

    def _select_reactive_vase_index(
        self,
        agent_pos: np.ndarray,
        vase_positions: list[np.ndarray],
        continue_to_goal: bool,
    ) -> int:
        """Pick a vase for reactive tasks, preferring goal-compatible triggers when needed."""
        if not vase_positions:
            return 0
        if not continue_to_goal:
            distances = [distance_xy(agent_pos, pos) for pos in vase_positions]
            return int(np.argmin(distances))

        goal_pos = np.asarray(self.goal_pos, dtype=float)[:2]
        best_index = 0
        best_score = float("inf")
        for index, pos in enumerate(vase_positions):
            vase_xy = np.asarray(pos, dtype=float)[:2]
            agent_distance = distance_xy(agent_pos, vase_xy)
            goal_distance = distance_xy(vase_xy, goal_pos)
            line_distance = _point_to_segment_distance(vase_xy, agent_pos, goal_pos)
            score = agent_distance + 0.55 * goal_distance + 0.90 * line_distance
            if score < best_score:
                best_score = score
                best_index = index
        return best_index

    def _dual_patrol_action(
        self,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        step_idx: int = 0,
    ) -> np.ndarray:
        """Alternate between patrol zones A and B.

        Uses deadline-aware switching: go to whichever zone will expire soonest
        relative to its estimated travel time, ensuring neither zone misses its
        windowed-eventually window.
        """
        horizon = self.task_config.horizon
        window = horizon // 2
        step_size = 0.006  # approximate agent movement per step

        # Track last visit time for each zone.
        if not self.state.patrol_last_visit:
            self.state.patrol_last_visit = {"A": -window, "B": -window}

        # Record a visit for both zones whenever agent is inside them.
        for name in ("A", "B"):
            z = self.resolved_zones[name]
            if distance_xy(agent_pos, np.asarray(z["center"], dtype=float)) <= float(z["radius"]):
                self.state.patrol_last_visit[name] = step_idx

        current = self.state.current_patrol_target or "A"

        # Compute urgency for each zone: steps until window expires minus travel time.
        # Lower urgency_margin means more urgent (must leave sooner).
        def urgency_margin(name: str) -> float:
            z = self.resolved_zones[name]
            dist = distance_xy(agent_pos, np.asarray(z["center"], dtype=float))
            travel_steps = dist / step_size
            deadline = self.state.patrol_last_visit[name] + window
            return (deadline - step_idx) - travel_steps

        margin_a = urgency_margin("A")
        margin_b = urgency_margin("B")

        # Go to whichever zone has the smaller urgency margin (more urgent).
        desired = "A" if margin_a <= margin_b else "B"
        if desired != current:
            current = desired
            self.state.current_patrol_target = current
            self.state.patrol_steps_on_target = 0

        self.state.patrol_steps_on_target += 1

        zone = self.resolved_zones[current]
        target = np.asarray(zone["center"], dtype=float)
        return self._compose_action(
            agent_pos,
            agent_vel,
            target,
            speed_limit=self.fast_threshold * 0.95,
            use_hazard_avoidance=False,
            use_vase_avoidance=False,
            skip_docking=True,
        )

    def _patrol_move(
        self,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Move toward a patrol waypoint at full speed, bypassing docking slowdown."""
        vector = np.asarray(target, dtype=float)[:2] - agent_pos
        norm = float(np.linalg.norm(vector))
        desired = vector / norm if norm > 1e-6 else np.zeros(2, dtype=float)
        preferred = desired.copy()
        preferred += self._repulsion_from_positions(
            agent_pos, self._current_hazards(), weight=1.0, influence=0.85
        )
        preferred += self._repulsion_from_positions(
            agent_pos, self._current_vases(), weight=0.7, influence=0.70
        )
        pref_norm = float(np.linalg.norm(preferred))
        if pref_norm > 1e-6:
            preferred = preferred / pref_norm
        heading = float(env_utils.get_agent_heading(self._env) or 0.0)
        angular_velocity = float(env_utils.get_agent_angular_velocity(self._env) or 0.0)
        target_heading = float(np.arctan2(preferred[1], preferred[0]))
        heading_error = self._wrap_angle(target_heading - heading)
        speed_limit = self.fast_threshold * 0.95
        current_speed = float(np.linalg.norm(agent_vel))
        cos_err = float(np.cos(heading_error))
        if abs(heading_error) > 1.4:
            forward = -0.3
        elif current_speed >= speed_limit * 0.98:
            forward = 0.0
        else:
            forward = float(np.clip(0.10 * max(0.3, cos_err), 0.03, 0.10))
        steer = float(np.clip(1.80 * heading_error - 0.15 * angular_velocity, -1.0, 1.0))
        return np.asarray([forward, steer], dtype=np.float32)

    def _patrol_single_zone_action(
        self,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        zone_name: str,
    ) -> np.ndarray:
        """Stay around one patrol zone to intentionally miss others."""
        zone = self.resolved_zones[zone_name]
        return self._compose_action(
            agent_pos,
            agent_vel,
            np.asarray(zone["center"], dtype=float),
            speed_limit=self.fast_threshold * 0.82,
            use_hazard_avoidance=True,
            use_vase_avoidance=True,
        )

    def _full_mission_action(
        self,
        env,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        skip_final_patrol: bool = False,
    ) -> np.ndarray:
        """Stage controller for Level 8 full mission."""
        if self.state.phase == "A":
            zone = self.resolved_zones["A"]
            if distance_xy(agent_pos, np.asarray(zone["center"], dtype=float)) <= float(zone["radius"]):
                self.state.phase = "B"
            else:
                return self._compose_action(
                    agent_pos,
                    agent_vel,
                    np.asarray(zone["center"], dtype=float),
                    speed_limit=self.fast_threshold * 0.95,
                    use_hazard_avoidance=True,
                    use_vase_avoidance=False,
                    skip_docking=True,
                )

        if self.state.phase == "B":
            zone = self.resolved_zones["B"]
            if distance_xy(agent_pos, np.asarray(zone["center"], dtype=float)) <= float(zone["radius"]):
                self.state.phase = "C_patrol"
            else:
                return self._compose_action(
                    agent_pos,
                    agent_vel,
                    np.asarray(zone["center"], dtype=float),
                    speed_limit=self.fast_threshold * 0.95,
                    use_hazard_avoidance=True,
                    use_vase_avoidance=False,
                    skip_docking=True,
                )

        if self.state.phase == "approach_vase":
            return self._reactive_vase_action(env, agent_pos, agent_vel, continue_to_goal=False)

        if self.state.phase in {"wander", "goal"}:
            self.state.phase = "C_patrol"

        if self.state.phase == "C_patrol":
            if skip_final_patrol:
                return self._compose_action(
                    agent_pos,
                    agent_vel,
                    self.goal_pos,
                    speed_limit=self.fast_threshold * 0.9,
                    use_hazard_avoidance=True,
                    use_vase_avoidance=False,
                )
            zone = self.resolved_zones["C"]
            target = np.asarray(zone["center"], dtype=float)
            radius = float(zone["radius"])
            distance_to_c = distance_xy(agent_pos, target)
            if distance_to_c <= max(radius, 0.55):
                self.state.patrol_hold_steps = min(self.state.patrol_hold_steps + 1, 6)
                return self._compose_action(
                    agent_pos,
                    agent_vel,
                    target,
                    speed_limit=self.fast_threshold * 0.42,
                    use_hazard_avoidance=True,
                    use_vase_avoidance=False,
                )
            self.state.patrol_hold_steps = 0
            return self._compose_action(
                agent_pos,
                agent_vel,
                target,
                speed_limit=self.fast_threshold * 0.92,
                use_hazard_avoidance=True,
                use_vase_avoidance=False,
                skip_docking=True,
            )

        return self._go_to_goal_action(agent_pos, agent_vel, obey_speed=True)

    def _compose_action(
        self,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        target: np.ndarray,
        speed_limit: float,
        use_hazard_avoidance: bool,
        use_vase_avoidance: bool,
        skip_docking: bool = False,
    ) -> np.ndarray:
        """Compose a low-level unicycle action toward a preferred vector."""
        vector = np.asarray(target, dtype=float)[:2] - agent_pos
        norm = np.linalg.norm(vector)
        desired = vector / norm if norm > 1e-6 else np.zeros(2, dtype=float)
        preferred = desired.copy()
        if skip_docking:
            goal_docking = False
            final_docking = False
            terminal_docking = False
            proximity_scale = 1.0
        else:
            goal_docking = norm < 0.80
            final_docking = norm < 0.65
            terminal_docking = norm < 0.48
            if terminal_docking:
                proximity_scale = 0.02
            elif final_docking:
                proximity_scale = 0.05
            elif goal_docking:
                proximity_scale = 0.10
            else:
                proximity_scale = 0.35 if norm < 0.45 else (0.6 if norm < 0.75 else 1.0)

        if use_vase_avoidance:
            preferred += self._repulsion_from_positions(
                agent_pos,
                self._current_vases(),
                weight=0.7 * proximity_scale,
                influence=0.70,
            )
        if use_hazard_avoidance:
            preferred += self._repulsion_from_positions(
                agent_pos,
                self._current_hazards(),
                weight=1.0 * proximity_scale,
                influence=0.85,
            )

        pref_norm = np.linalg.norm(preferred)
        if pref_norm > 1e-6:
            preferred = preferred / pref_norm
        heading = float(env_utils.get_agent_heading(self._env) or 0.0)
        angular_velocity = float(env_utils.get_agent_angular_velocity(self._env) or 0.0)
        target_heading = float(np.arctan2(preferred[1], preferred[0]))
        heading_error = self._wrap_angle(target_heading - heading)
        forward = self._forward_command(
            heading_error,
            speed_limit,
            agent_vel,
            target_distance=norm,
            goal_docking=goal_docking,
            skip_docking=skip_docking,
        )
        if terminal_docking:
            steer_gain = 0.75
            steer_damping = 0.04
            steer_clip = 0.35
        elif final_docking:
            steer_gain = 0.90
            steer_damping = 0.05
            steer_clip = 0.45
        elif goal_docking:
            steer_gain = 1.15
            steer_damping = 0.08
            steer_clip = 0.60
        else:
            steer_gain = 1.80
            steer_damping = 0.15
            steer_clip = 1.00
        steer = float(
            np.clip(
                steer_gain * heading_error - steer_damping * angular_velocity,
                -steer_clip,
                steer_clip,
            ),
        )
        return np.asarray([forward, steer], dtype=np.float32)

    def _current_hazards(self) -> list[np.ndarray]:
        """Return cached hazard positions."""
        return [np.asarray(pos, dtype=float)[:2] for pos in self._env_hazard_positions]

    def _current_vases(self) -> list[np.ndarray]:
        """Return cached vase positions."""
        return [np.asarray(pos, dtype=float)[:2] for pos in self._env_vase_positions]

    def prepare_positions(self, env) -> None:
        """Refresh obstacle positions before computing an action."""
        self._env = env
        self._env_hazard_positions = env_utils.get_hazard_positions(env)
        self._env_vase_positions = env_utils.get_vase_positions(env)

    def _repulsion_from_positions(
        self,
        agent_pos: np.ndarray,
        positions: list[np.ndarray],
        weight: float,
        influence: float,
    ) -> np.ndarray:
        """Return a summed inverse-distance repulsion vector."""
        repulsion = np.zeros(2, dtype=float)
        for pos in positions:
            delta = agent_pos - np.asarray(pos, dtype=float)[:2]
            distance = np.linalg.norm(delta)
            if distance < 1e-6 or distance > influence:
                continue
            repulsion += weight * (1.0 / max(distance, 0.05) - 1.0 / influence) * (delta / distance)
        return repulsion

    def _respect_speed_limit(
        self,
        action: np.ndarray,
        agent_vel: np.ndarray,
        speed_limit: float,
    ) -> np.ndarray:
        """Scale action down when the current speed is already high."""
        current_speed = float(np.linalg.norm(agent_vel))
        if current_speed <= speed_limit:
            return action
        if current_speed < 1e-6:
            return 0.5 * action
        slowdown = max(0.02, min(0.35, speed_limit / current_speed))
        return slowdown * action - 0.05 * agent_vel

    def _forward_command(
        self,
        heading_error: float,
        speed_limit: float,
        agent_vel: np.ndarray,
        target_distance: float,
        goal_docking: bool = False,
        skip_docking: bool = False,
    ) -> float:
        """Convert heading alignment and speed budget into a forward command."""
        current_speed = float(np.linalg.norm(agent_vel))
        terminal_docking = (not skip_docking) and target_distance < 0.48
        final_docking = (not skip_docking) and target_distance < 0.65

        if terminal_docking:
            alignment = max(0.72, float(np.cos(0.8 * heading_error)))
            base = 0.090
            if current_speed > speed_limit * 1.02:
                return 0.0
            if current_speed > speed_limit * 0.90:
                base *= 0.60
            return float(np.clip(base * alignment, 0.040, 0.11))

        if final_docking:
            alignment = max(0.64, float(np.cos(0.9 * heading_error)))
            base = 0.088
            if current_speed > speed_limit * 1.02:
                return 0.0
            if current_speed > speed_limit * 0.93:
                base *= 0.70
            return float(np.clip(base * alignment, 0.035, 0.12))

        if abs(heading_error) > 1.4:
            alignment = 0.1 if goal_docking else -0.4
        else:
            alignment = float(np.cos(heading_error))
        if goal_docking:
            alignment = max(0.55, alignment)
        if speed_limit <= self.fast_threshold * 0.5:
            base = 0.012
        elif speed_limit <= self.fast_threshold:
            base = 0.065
        else:
            base = 0.085
        if goal_docking:
            base *= 1.45
        if current_speed > speed_limit * 0.98:
            return 0.0
        if current_speed > speed_limit:
            base *= max(0.1, speed_limit / max(current_speed, 1e-6))
        return float(np.clip(base * alignment, -0.12, 0.12))

    def _wrap_angle(self, angle: float) -> float:
        """Wrap an angle to [-pi, pi]."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _perturb_action(self, action: np.ndarray) -> np.ndarray:
        """Add near-success perturbations while keeping the action bounded."""
        noise = self.rng.normal(0.0, self.near_success_noise_std, size=action.shape)
        if self.rng.random() < 0.20:
            noise += self.rng.normal(0.0, 0.04, size=action.shape)
        return np.clip(action + noise, -1.0, 1.0).astype(np.float32)

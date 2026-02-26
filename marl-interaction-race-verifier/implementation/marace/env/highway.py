"""
Highway / intersection multi-agent driving environment.

Implements a continuous 2-D driving environment with support for
multi-lane highways, four-way intersections, on-ramp merging, and
lane-change overtaking scenarios.  Vehicle kinematics use a bicycle
model, and collision detection uses oriented-bounding-box overlap.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from marace.env.base import (
    AgentTimingConfig,
    AsyncSteppingSemantics,
    EnvironmentClock,
    EnvironmentState,
    MultiAgentEnv,
    StaleObservationModel,
)


# ---------------------------------------------------------------------------
# Vehicle state & dynamics
# ---------------------------------------------------------------------------

@dataclass
class VehicleState:
    """Continuous state of a single vehicle.

    Attributes:
        x: Longitudinal position (m).
        y: Lateral position (m).
        vx: Longitudinal velocity (m/s).
        vy: Lateral velocity (m/s).
        heading: Heading angle (rad), 0 = positive-x direction.
        length: Vehicle length (m).
        width: Vehicle width (m).
    """
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0
    length: float = 4.5
    width: float = 2.0

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])

    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    def corners(self) -> np.ndarray:
        """Return the four corners of the oriented bounding box (4×2)."""
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        hl, hw = self.length / 2, self.width / 2
        dx = np.array([cos_h, sin_h])
        dy = np.array([-sin_h, cos_h])
        centre = self.position()
        return np.array([
            centre + hl * dx + hw * dy,
            centre + hl * dx - hw * dy,
            centre - hl * dx - hw * dy,
            centre - hl * dx + hw * dy,
        ])

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy, self.heading])

    @staticmethod
    def from_array(arr: np.ndarray, length: float = 4.5, width: float = 2.0) -> "VehicleState":
        return VehicleState(
            x=float(arr[0]), y=float(arr[1]),
            vx=float(arr[2]), vy=float(arr[3]),
            heading=float(arr[4]),
            length=length, width=width,
        )


class VehicleDynamics:
    """Bicycle-model kinematics for vehicle motion.

    Actions are ``[acceleration, steering_angle]``.

    Attributes:
        dt: Integration time step (s).
        wheelbase: Distance between axles (m).
        max_speed: Maximum speed (m/s).
        max_accel: Maximum absolute acceleration (m/s²).
        max_steer: Maximum absolute steering angle (rad).
    """

    def __init__(
        self,
        dt: float = 0.1,
        wheelbase: float = 2.5,
        max_speed: float = 40.0,
        max_accel: float = 5.0,
        max_steer: float = math.pi / 6,
    ) -> None:
        self.dt = dt
        self.wheelbase = wheelbase
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_steer = max_steer

    def step(self, state: VehicleState, action: np.ndarray) -> VehicleState:
        """Integrate one step using the bicycle model.

        Args:
            state: Current vehicle state.
            action: ``[acceleration, steering_angle]``.

        Returns:
            Updated vehicle state.
        """
        accel = float(np.clip(action[0], -self.max_accel, self.max_accel))
        steer = float(np.clip(action[1], -self.max_steer, self.max_steer))

        speed = state.speed()
        speed = max(0.0, min(speed + accel * self.dt, self.max_speed))

        beta = math.atan2(math.tan(steer) * 0.5, 1.0)
        heading = state.heading + (speed / self.wheelbase) * math.sin(beta) * self.dt

        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)
        x = state.x + vx * self.dt
        y = state.y + vy * self.dt

        return VehicleState(
            x=x, y=y, vx=vx, vy=vy, heading=heading,
            length=state.length, width=state.width,
        )


# ---------------------------------------------------------------------------
# Road geometry
# ---------------------------------------------------------------------------

@dataclass
class LaneSpec:
    """Specification of a single lane."""
    lane_id: int
    center_y: float
    width: float = 3.7
    speed_limit: float = 30.0
    direction: float = 0.0  # heading in radians


@dataclass
class RoadGeometry:
    """Multi-lane road geometry."""
    lanes: List[LaneSpec] = field(default_factory=list)
    road_length: float = 500.0

    def lane_center(self, lane_id: int) -> float:
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane.center_y
        raise ValueError(f"Unknown lane {lane_id}")

    def nearest_lane(self, y: float) -> LaneSpec:
        return min(self.lanes, key=lambda l: abs(l.center_y - y))

    @staticmethod
    def default_highway(num_lanes: int = 3, lane_width: float = 3.7) -> "RoadGeometry":
        lanes = [
            LaneSpec(lane_id=i, center_y=i * lane_width, width=lane_width)
            for i in range(num_lanes)
        ]
        return RoadGeometry(lanes=lanes)


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------

def _project_polygon(corners: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """Project polygon corners onto *axis* and return (min, max)."""
    projections = corners @ axis
    return float(projections.min()), float(projections.max())


def _polygons_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Check overlap of two convex polygons using the Separating Axis Theorem."""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                continue
            axis = axis / norm
            min_a, max_a = _project_polygon(a, axis)
            min_b, max_b = _project_polygon(b, axis)
            if max_a < min_b or max_b < min_a:
                return False
    return True


def check_collision(v1: VehicleState, v2: VehicleState) -> bool:
    """Return ``True`` if the two vehicles' bounding boxes overlap."""
    return _polygons_overlap(v1.corners(), v2.corners())


# ---------------------------------------------------------------------------
# Safety predicates
# ---------------------------------------------------------------------------

class SafetyPredicates:
    """Standard safety predicates for driving environments."""

    @staticmethod
    def min_distance(v1: VehicleState, v2: VehicleState) -> float:
        """Euclidean distance between vehicle centres."""
        return float(np.linalg.norm(v1.position() - v2.position()))

    @staticmethod
    def time_to_collision(v1: VehicleState, v2: VehicleState) -> float:
        """Estimated time to collision along current velocity vectors.

        Returns ``inf`` if the vehicles are diverging.
        """
        dp = v2.position() - v1.position()
        dv = v2.velocity() - v1.velocity()
        dv_norm_sq = float(np.dot(dv, dv))
        if dv_norm_sq < 1e-12:
            return float("inf")
        t = -float(np.dot(dp, dv)) / dv_norm_sq
        if t < 0:
            return float("inf")
        return t

    @staticmethod
    def safe_following_distance(v: VehicleState, reaction_time: float = 1.5) -> float:
        """Minimum safe following distance at current speed."""
        return v.speed() * reaction_time + 2.0

    @staticmethod
    def any_collision(vehicles: Dict[str, VehicleState]) -> bool:
        ids = list(vehicles.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if check_collision(vehicles[ids[i]], vehicles[ids[j]]):
                    return True
        return False

    @staticmethod
    def min_pairwise_distance(vehicles: Dict[str, VehicleState]) -> float:
        ids = list(vehicles.keys())
        min_d = float("inf")
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = SafetyPredicates.min_distance(vehicles[ids[i]], vehicles[ids[j]])
                min_d = min(min_d, d)
        return min_d


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

class ScenarioType(Enum):
    HIGHWAY = "highway"
    INTERSECTION = "intersection"
    MERGING = "merging"
    OVERTAKING = "overtaking"


class IntersectionScenario:
    """Four-way intersection with no traffic signals.

    Vehicles approach from the four cardinal directions.  There is no
    explicit right-of-way; agents must negotiate passage.

    Attributes:
        center: (x, y) center of the intersection.
        arm_length: Length of each approach arm (m).
        road_width: Width of the intersecting roads (m).
    """

    def __init__(
        self,
        center: Tuple[float, float] = (0.0, 0.0),
        arm_length: float = 100.0,
        road_width: float = 7.4,
    ) -> None:
        self.center = np.array(center)
        self.arm_length = arm_length
        self.road_width = road_width

    def initial_positions(self, num_agents: int) -> Dict[str, VehicleState]:
        """Place agents at the ends of approach arms."""
        headings = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
        states: Dict[str, VehicleState] = {}
        for i in range(num_agents):
            direction = headings[i % 4]
            offset = np.array([
                -math.cos(direction) * self.arm_length,
                -math.sin(direction) * self.arm_length,
            ])
            pos = self.center + offset
            speed = 10.0
            states[f"agent_{i}"] = VehicleState(
                x=float(pos[0]), y=float(pos[1]),
                vx=speed * math.cos(direction),
                vy=speed * math.sin(direction),
                heading=direction,
            )
        return states

    def in_intersection(self, vs: VehicleState) -> bool:
        """Check whether a vehicle is inside the intersection box."""
        d = np.abs(vs.position() - self.center)
        return bool(d[0] < self.road_width / 2 and d[1] < self.road_width / 2)


class MergingScenario:
    """Highway on-ramp merging scenario.

    One or more agents travel on the main road while others enter via
    an on-ramp that merges into the rightmost lane.

    Attributes:
        merge_point_x: Longitudinal position where the ramp meets the highway.
        ramp_length: Length of the on-ramp (m).
        ramp_angle: Angle of the ramp relative to the highway (rad).
    """

    def __init__(
        self,
        merge_point_x: float = 200.0,
        ramp_length: float = 150.0,
        ramp_angle: float = math.pi / 6,
        road: Optional[RoadGeometry] = None,
    ) -> None:
        self.merge_point_x = merge_point_x
        self.ramp_length = ramp_length
        self.ramp_angle = ramp_angle
        self.road = road or RoadGeometry.default_highway()

    def initial_positions(
        self,
        num_highway: int = 2,
        num_merging: int = 1,
    ) -> Dict[str, VehicleState]:
        states: Dict[str, VehicleState] = {}
        # Highway vehicles
        for i in range(num_highway):
            lane = self.road.lanes[min(i, len(self.road.lanes) - 1)]
            states[f"agent_{i}"] = VehicleState(
                x=self.merge_point_x - 80.0 - 30.0 * i,
                y=lane.center_y,
                vx=lane.speed_limit * 0.8,
                vy=0.0,
                heading=0.0,
            )
        # Merging vehicles on ramp
        for j in range(num_merging):
            k = num_highway + j
            ramp_start_x = self.merge_point_x - self.ramp_length * math.cos(self.ramp_angle)
            ramp_start_y = self.road.lanes[0].center_y - self.ramp_length * math.sin(self.ramp_angle)
            states[f"agent_{k}"] = VehicleState(
                x=ramp_start_x + 20.0 * j,
                y=ramp_start_y + 20.0 * j * math.sin(self.ramp_angle),
                vx=20.0 * math.cos(self.ramp_angle),
                vy=20.0 * math.sin(self.ramp_angle),
                heading=self.ramp_angle,
            )
        return states


class OvertakingScenario:
    """Lane-change overtaking scenario.

    A faster vehicle approaches a slower vehicle and must change lanes
    to overtake safely.

    Attributes:
        road: Road geometry.
        speed_diff: Speed difference between fast and slow vehicles (m/s).
        initial_gap: Initial longitudinal gap (m).
    """

    def __init__(
        self,
        road: Optional[RoadGeometry] = None,
        speed_diff: float = 10.0,
        initial_gap: float = 60.0,
    ) -> None:
        self.road = road or RoadGeometry.default_highway(num_lanes=2)
        self.speed_diff = speed_diff
        self.initial_gap = initial_gap

    def initial_positions(self) -> Dict[str, VehicleState]:
        slow_lane = self.road.lanes[0]
        fast_lane = self.road.lanes[0]  # same lane initially
        slow_speed = slow_lane.speed_limit * 0.6
        fast_speed = slow_speed + self.speed_diff
        return {
            "agent_0": VehicleState(
                x=0.0, y=fast_lane.center_y,
                vx=fast_speed, vy=0.0, heading=0.0,
            ),
            "agent_1": VehicleState(
                x=self.initial_gap, y=slow_lane.center_y,
                vx=slow_speed, vy=0.0, heading=0.0,
            ),
        }


# ---------------------------------------------------------------------------
# Highway environment state
# ---------------------------------------------------------------------------

class HighwayState(EnvironmentState):
    """Full state of the highway environment."""

    def __init__(
        self,
        vehicles: Dict[str, VehicleState],
        tick: int = 0,
        collisions: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.vehicles = vehicles
        self.tick = tick
        self.collisions: List[Tuple[str, str]] = collisions or []

    def copy(self) -> "HighwayState":
        return HighwayState(
            vehicles={k: copy.deepcopy(v) for k, v in self.vehicles.items()},
            tick=self.tick,
            collisions=list(self.collisions),
        )


# ---------------------------------------------------------------------------
# Highway environment
# ---------------------------------------------------------------------------

class HighwayEnv(MultiAgentEnv):
    """Multi-lane highway / intersection environment.

    Supports configurable road geometry, multiple scenario types, and
    per-agent observation with latency modelling.

    Args:
        num_agents: Number of vehicles.
        scenario_type: Scenario type (highway, intersection, merging, overtaking).
        road: Road geometry (auto-created if ``None``).
        dynamics: Vehicle dynamics model.
        sensor_range: Maximum distance at which other vehicles are observed.
        dt: Simulation time step (s).
        max_steps: Episode length.
    """

    OBS_DIM = 5  # per-vehicle: [x, y, vx, vy, heading]
    ACT_DIM = 2  # [acceleration, steering_angle]

    def __init__(
        self,
        num_agents: int = 4,
        scenario_type: ScenarioType = ScenarioType.HIGHWAY,
        road: Optional[RoadGeometry] = None,
        dynamics: Optional[VehicleDynamics] = None,
        sensor_range: float = 100.0,
        dt: float = 0.1,
        max_steps: int = 500,
        speed_limits: Optional[Dict[str, float]] = None,
        stepping: Optional[AsyncSteppingSemantics] = None,
        timing_configs: Optional[Dict[str, AgentTimingConfig]] = None,
        clock: Optional[EnvironmentClock] = None,
    ) -> None:
        agent_ids = [f"agent_{i}" for i in range(num_agents)]
        super().__init__(
            agent_ids=agent_ids,
            stepping=stepping,
            timing_configs=timing_configs,
            clock=clock or EnvironmentClock(dt=dt),
        )
        self.scenario_type = scenario_type
        self.road = road or RoadGeometry.default_highway()
        self.dynamics = dynamics or VehicleDynamics(dt=dt)
        self.sensor_range = sensor_range
        self.dt = dt
        self.max_steps = max_steps
        self.speed_limits = speed_limits or {}

        self._scenario = self._build_scenario(num_agents)
        self._vehicles: Dict[str, VehicleState] = {}
        self._collision_log: List[Tuple[int, str, str]] = []

    def _build_scenario(
        self, num_agents: int
    ) -> Any:
        if self.scenario_type == ScenarioType.INTERSECTION:
            return IntersectionScenario()
        if self.scenario_type == ScenarioType.MERGING:
            return MergingScenario(road=self.road)
        if self.scenario_type == ScenarioType.OVERTAKING:
            return OvertakingScenario(road=self.road)
        return None  # plain highway

    # -- environment interface -----------------------------------------------

    def _reset_impl(self) -> Dict[str, np.ndarray]:
        if isinstance(self._scenario, IntersectionScenario):
            self._vehicles = self._scenario.initial_positions(self.num_agents)
        elif isinstance(self._scenario, MergingScenario):
            nh = max(1, self.num_agents - 1)
            self._vehicles = self._scenario.initial_positions(nh, self.num_agents - nh)
        elif isinstance(self._scenario, OvertakingScenario):
            self._vehicles = self._scenario.initial_positions()
        else:
            self._vehicles = self._default_positions()
        self._collision_log.clear()
        self._agent_ids = list(self._vehicles.keys())
        return {aid: self._observe(aid) for aid in self._agent_ids}

    def _default_positions(self) -> Dict[str, VehicleState]:
        states: Dict[str, VehicleState] = {}
        for i, aid in enumerate(self._agent_ids):
            lane = self.road.lanes[i % len(self.road.lanes)]
            states[aid] = VehicleState(
                x=20.0 * i, y=lane.center_y,
                vx=lane.speed_limit * 0.7, vy=0.0, heading=0.0,
            )
        return states

    def _step_single(
        self, agent_id: str, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        vs = self._vehicles[agent_id]
        new_vs = self.dynamics.step(vs, action)

        # Apply speed limit
        limit = self.speed_limits.get(agent_id, self.dynamics.max_speed)
        if new_vs.speed() > limit:
            scale = limit / max(new_vs.speed(), 1e-9)
            new_vs.vx *= scale
            new_vs.vy *= scale

        self._vehicles[agent_id] = new_vs

        # Detect collisions
        collisions: List[str] = []
        for other_id, other_vs in self._vehicles.items():
            if other_id != agent_id and check_collision(new_vs, other_vs):
                collisions.append(other_id)
                self._collision_log.append((self._step_count, agent_id, other_id))

        reward = self._compute_reward(agent_id, collisions)
        done = len(collisions) > 0 or self._step_count >= self.max_steps
        obs = self._observe(agent_id)

        info: Dict[str, Any] = {
            "collisions": collisions,
            "speed": new_vs.speed(),
            "position": new_vs.position().tolist(),
        }
        return obs, reward, done, info

    def _get_state_impl(self) -> HighwayState:
        return HighwayState(
            vehicles={k: copy.deepcopy(v) for k, v in self._vehicles.items()},
            tick=self._step_count,
            collisions=[(t, a, b) for t, a, b in self._collision_log],
        )

    def _set_state_impl(self, state: EnvironmentState) -> None:
        assert isinstance(state, HighwayState)
        self._vehicles = {k: copy.deepcopy(v) for k, v in state.vehicles.items()}
        self._step_count = state.tick
        self._collision_log = list(state.collisions)

    # -- observation ---------------------------------------------------------

    def _observe(self, agent_id: str) -> np.ndarray:
        """Build observation for *agent_id*: own state + nearby vehicles."""
        ego = self._vehicles[agent_id]
        obs_parts = [ego.to_array()]
        for other_id in sorted(self._vehicles):
            if other_id == agent_id:
                continue
            other = self._vehicles[other_id]
            dist = float(np.linalg.norm(ego.position() - other.position()))
            if dist <= self.sensor_range:
                obs_parts.append(other.to_array())
        return np.concatenate(obs_parts)

    def _compute_reward(self, agent_id: str, collisions: List[str]) -> float:
        """Default reward: penalise collisions, reward forward progress."""
        if collisions:
            return -100.0
        vs = self._vehicles[agent_id]
        return float(vs.vx * self.dt)  # reward for forward progress

    # -- spaces --------------------------------------------------------------

    def observation_space(self, agent_id: str) -> Dict[str, Any]:
        max_obs_vehicles = self.num_agents
        return {
            "shape": (max_obs_vehicles * self.OBS_DIM,),
            "dtype": "float64",
            "low": -np.inf,
            "high": np.inf,
        }

    def action_space(self, agent_id: str) -> Dict[str, Any]:
        return {
            "shape": (self.ACT_DIM,),
            "dtype": "float64",
            "low": np.array([-self.dynamics.max_accel, -self.dynamics.max_steer]),
            "high": np.array([self.dynamics.max_accel, self.dynamics.max_steer]),
        }

    # -- accessors -----------------------------------------------------------

    @property
    def vehicles(self) -> Dict[str, VehicleState]:
        return dict(self._vehicles)

    @property
    def collision_log(self) -> List[Tuple[int, str, str]]:
        return list(self._collision_log)

    def safety_predicates(self) -> SafetyPredicates:
        return SafetyPredicates()

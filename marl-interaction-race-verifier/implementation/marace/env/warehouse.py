"""
Warehouse gridworld multi-agent environment.

Implements a 2-D continuous warehouse with corridors, shelves, and
loading/unloading stations.  Robots use differential-drive kinematics
and can pick up / deliver items.  The environment supports corridor
conflict detection, deadlock detection, and configurable sensor models.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

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
# Robot state & dynamics
# ---------------------------------------------------------------------------

@dataclass
class RobotState:
    """State of a single warehouse robot.

    Attributes:
        x: X position (m).
        y: Y position (m).
        heading: Heading angle (rad).
        carrying_item: Item id currently carried, or ``None``.
        battery: Battery level in [0, 1].
        radius: Robot radius for collision checking (m).
    """
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    carrying_item: Optional[str] = None
    battery: float = 1.0
    radius: float = 0.4

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def to_array(self) -> np.ndarray:
        carry_flag = 1.0 if self.carrying_item is not None else 0.0
        return np.array([self.x, self.y, self.heading, carry_flag, self.battery])

    @staticmethod
    def from_array(arr: np.ndarray, radius: float = 0.4) -> "RobotState":
        return RobotState(
            x=float(arr[0]), y=float(arr[1]),
            heading=float(arr[2]),
            carrying_item=None,
            battery=float(arr[4]),
            radius=radius,
        )


class RobotDynamics:
    """Differential-drive kinematics for warehouse robots.

    Actions are ``[linear_velocity, angular_velocity]``.

    Attributes:
        dt: Integration time step (s).
        max_linear: Maximum linear velocity (m/s).
        max_angular: Maximum angular velocity (rad/s).
        battery_drain: Battery drain per step.
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_linear: float = 1.5,
        max_angular: float = math.pi,
        battery_drain: float = 0.0001,
    ) -> None:
        self.dt = dt
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.battery_drain = battery_drain

    def step(self, state: RobotState, action: np.ndarray) -> RobotState:
        """Integrate one step.

        Args:
            state: Current robot state.
            action: ``[linear_velocity, angular_velocity]``.

        Returns:
            Updated state.
        """
        v = float(np.clip(action[0], -self.max_linear, self.max_linear))
        omega = float(np.clip(action[1], -self.max_angular, self.max_angular))

        heading = state.heading + omega * self.dt
        heading = heading % (2 * math.pi)
        x = state.x + v * math.cos(heading) * self.dt
        y = state.y + v * math.sin(heading) * self.dt
        battery = max(0.0, state.battery - self.battery_drain)

        return RobotState(
            x=x, y=y, heading=heading,
            carrying_item=state.carrying_item,
            battery=battery,
            radius=state.radius,
        )


# ---------------------------------------------------------------------------
# Warehouse layout
# ---------------------------------------------------------------------------

class CellType(Enum):
    FLOOR = 0
    WALL = 1
    SHELF = 2
    PICKUP_STATION = 3
    DELIVERY_STATION = 4
    CHARGING_STATION = 5


@dataclass
class WarehouseLayout:
    """Configurable warehouse layout.

    The layout is defined on a discrete grid but robots move continuously.
    Each cell has a type and a real-world size.

    Attributes:
        width: Number of columns.
        height: Number of rows.
        cell_size: Real-world size of one cell (m).
        grid: 2-D array of ``CellType`` values.
        corridor_width: Real-world width of corridors (m).
    """
    width: int = 20
    height: int = 20
    cell_size: float = 1.0
    corridor_width: float = 1.5
    grid: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.grid is None:
            self.grid = np.full((self.height, self.width), CellType.FLOOR.value, dtype=int)

    def set_cell(self, row: int, col: int, cell_type: CellType) -> None:
        assert self.grid is not None
        self.grid[row, col] = cell_type.value

    def get_cell(self, row: int, col: int) -> CellType:
        assert self.grid is not None
        return CellType(self.grid[row, col])

    def world_position(self, row: int, col: int) -> np.ndarray:
        """Centre of cell ``(row, col)`` in world coordinates."""
        return np.array([col * self.cell_size + self.cell_size / 2,
                         row * self.cell_size + self.cell_size / 2])

    def cell_index(self, x: float, y: float) -> Tuple[int, int]:
        """Return the ``(row, col)`` of the cell containing ``(x, y)``."""
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        col = max(0, min(col, self.width - 1))
        row = max(0, min(row, self.height - 1))
        return row, col

    def is_passable(self, row: int, col: int) -> bool:
        ct = self.get_cell(row, col)
        return ct != CellType.WALL and ct != CellType.SHELF

    def find_cells(self, cell_type: CellType) -> List[Tuple[int, int]]:
        assert self.grid is not None
        positions: List[Tuple[int, int]] = []
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] == cell_type.value:
                    positions.append((r, c))
        return positions

    @staticmethod
    def default_warehouse(
        width: int = 20,
        height: int = 20,
        num_shelves: int = 6,
        cell_size: float = 1.0,
    ) -> "WarehouseLayout":
        """Create a warehouse with corridors, shelves, and stations."""
        layout = WarehouseLayout(width=width, height=height, cell_size=cell_size)
        assert layout.grid is not None
        # Place shelf rows
        shelf_rows = np.linspace(3, height - 4, num_shelves, dtype=int)
        for sr in shelf_rows:
            for c in range(3, width - 3):
                if c % 4 != 0:  # leave gaps for cross-corridors
                    layout.set_cell(int(sr), c, CellType.SHELF)
        # Pickup stations on left edge
        for r in range(2, height - 2, 4):
            layout.set_cell(r, 0, CellType.PICKUP_STATION)
        # Delivery stations on right edge
        for r in range(2, height - 2, 4):
            layout.set_cell(r, width - 1, CellType.DELIVERY_STATION)
        # Charging station at bottom-left
        layout.set_cell(height - 1, 0, CellType.CHARGING_STATION)
        return layout

    @property
    def real_width(self) -> float:
        return self.width * self.cell_size

    @property
    def real_height(self) -> float:
        return self.height * self.cell_size


# ---------------------------------------------------------------------------
# Task assignment
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A pickup-delivery task."""
    task_id: str
    pickup_pos: Tuple[int, int]
    delivery_pos: Tuple[int, int]
    item_id: str
    deadline: Optional[int] = None  # step deadline


@dataclass
class TaskAssignment:
    """Manages task assignments for robots.

    Attributes:
        assignments: Mapping from agent id to list of pending tasks.
        completed: List of completed task ids.
    """
    assignments: Dict[str, List[Task]] = field(default_factory=dict)
    completed: List[str] = field(default_factory=list)

    def assign(self, agent_id: str, task: Task) -> None:
        self.assignments.setdefault(agent_id, []).append(task)

    def current_task(self, agent_id: str) -> Optional[Task]:
        tasks = self.assignments.get(agent_id, [])
        return tasks[0] if tasks else None

    def complete_current(self, agent_id: str) -> Optional[Task]:
        tasks = self.assignments.get(agent_id, [])
        if tasks:
            done = tasks.pop(0)
            self.completed.append(done.task_id)
            return done
        return None

    def all_done(self) -> bool:
        return all(len(t) == 0 for t in self.assignments.values())

    def pending_count(self, agent_id: str) -> int:
        return len(self.assignments.get(agent_id, []))

    def overdue_tasks(self, current_step: int) -> List[Tuple[str, Task]]:
        overdue: List[Tuple[str, Task]] = []
        for aid, tasks in self.assignments.items():
            for t in tasks:
                if t.deadline is not None and current_step > t.deadline:
                    overdue.append((aid, t))
        return overdue


# ---------------------------------------------------------------------------
# Corridor conflict & deadlock detection
# ---------------------------------------------------------------------------

class CorridorConflict:
    """Detect corridor conflicts: two robots in a narrow corridor heading
    towards each other."""

    def __init__(self, layout: WarehouseLayout, corridor_width: float = 1.5) -> None:
        self._layout = layout
        self._corridor_width = corridor_width

    def detect(self, robots: Dict[str, RobotState]) -> List[Tuple[str, str]]:
        """Return pairs of robots in corridor conflict."""
        conflicts: List[Tuple[str, str]] = []
        ids = list(robots.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = robots[ids[i]], robots[ids[j]]
                dist = float(np.linalg.norm(a.position() - b.position()))
                if dist < self._corridor_width + a.radius + b.radius:
                    # Check if they are heading towards each other
                    dir_a = np.array([math.cos(a.heading), math.sin(a.heading)])
                    dir_b = np.array([math.cos(b.heading), math.sin(b.heading)])
                    to_b = b.position() - a.position()
                    norm = np.linalg.norm(to_b)
                    if norm > 1e-6:
                        to_b /= norm
                    # Heading towards each other if dot products have opposite signs
                    if np.dot(dir_a, to_b) > 0.3 and np.dot(dir_b, -to_b) > 0.3:
                        conflicts.append((ids[i], ids[j]))
        return conflicts

    def is_in_corridor(self, robot: RobotState) -> bool:
        """Check if a robot is in a narrow corridor section."""
        row, col = self._layout.cell_index(robot.x, robot.y)
        passable_neighbours = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self._layout.height and 0 <= nc < self._layout.width:
                if self._layout.is_passable(nr, nc):
                    passable_neighbours += 1
        # A corridor cell has exactly 2 passable neighbours in opposite directions
        return passable_neighbours <= 2


class DeadlockDetection:
    """Detect deadlock configurations in the warehouse.

    A deadlock occurs when a cycle of robots are each waiting for the
    next robot in the cycle to move.
    """

    def __init__(self, proximity_threshold: float = 1.0) -> None:
        self._threshold = proximity_threshold

    def detect(self, robots: Dict[str, RobotState]) -> List[FrozenSet[str]]:
        """Return sets of robots in mutual deadlock."""
        ids = list(robots.keys())
        # Build "blocked-by" graph: agent i is blocked by agent j if j is
        # ahead of i and within threshold
        blocked_by: Dict[str, Optional[str]] = {aid: None for aid in ids}
        for i, aid in enumerate(ids):
            a = robots[aid]
            forward = np.array([math.cos(a.heading), math.sin(a.heading)])
            best_dist = float("inf")
            for j, bid in enumerate(ids):
                if i == j:
                    continue
                b = robots[bid]
                diff = b.position() - a.position()
                dist = float(np.linalg.norm(diff))
                if dist < self._threshold and dist < best_dist:
                    if dist > 1e-6:
                        cos_angle = float(np.dot(forward, diff / dist))
                        if cos_angle > 0.5:
                            best_dist = dist
                            blocked_by[aid] = bid

        # Find cycles in the blocked-by graph
        deadlocks: List[FrozenSet[str]] = []
        visited: Set[str] = set()
        for start in ids:
            if start in visited:
                continue
            path: List[str] = []
            current: Optional[str] = start
            path_set: Set[str] = set()
            while current is not None and current not in path_set:
                if current in visited:
                    break
                path.append(current)
                path_set.add(current)
                current = blocked_by.get(current)
            if current is not None and current in path_set:
                # Found a cycle starting at 'current'
                cycle_start = path.index(current)
                cycle = frozenset(path[cycle_start:])
                if len(cycle) >= 2:
                    deadlocks.append(cycle)
            visited.update(path)
        return deadlocks


# ---------------------------------------------------------------------------
# Safety predicates
# ---------------------------------------------------------------------------

class WarehouseSafetyPredicates:
    """Safety predicates for the warehouse environment."""

    @staticmethod
    def collision(r1: RobotState, r2: RobotState) -> bool:
        dist = float(np.linalg.norm(r1.position() - r2.position()))
        return dist < r1.radius + r2.radius

    @staticmethod
    def any_collision(robots: Dict[str, RobotState]) -> bool:
        ids = list(robots.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if WarehouseSafetyPredicates.collision(robots[ids[i]], robots[ids[j]]):
                    return True
        return False

    @staticmethod
    def corridor_deadlock(
        robots: Dict[str, RobotState],
        detector: DeadlockDetection,
    ) -> bool:
        return len(detector.detect(robots)) > 0

    @staticmethod
    def task_timeout(
        tasks: TaskAssignment,
        current_step: int,
    ) -> bool:
        return len(tasks.overdue_tasks(current_step)) > 0

    @staticmethod
    def wall_collision(robot: RobotState, layout: WarehouseLayout) -> bool:
        row, col = layout.cell_index(robot.x, robot.y)
        return not layout.is_passable(row, col)

    @staticmethod
    def low_battery(robot: RobotState, threshold: float = 0.05) -> bool:
        return robot.battery < threshold


# ---------------------------------------------------------------------------
# Warehouse environment state
# ---------------------------------------------------------------------------

class WarehouseState(EnvironmentState):
    """Full state of the warehouse environment."""

    def __init__(
        self,
        robots: Dict[str, RobotState],
        tasks: TaskAssignment,
        items_on_ground: Dict[str, Tuple[float, float]],
        tick: int = 0,
    ) -> None:
        self.robots = robots
        self.tasks = tasks
        self.items_on_ground = items_on_ground
        self.tick = tick

    def copy(self) -> "WarehouseState":
        return WarehouseState(
            robots={k: copy.deepcopy(v) for k, v in self.robots.items()},
            tasks=copy.deepcopy(self.tasks),
            items_on_ground=dict(self.items_on_ground),
            tick=self.tick,
        )


# ---------------------------------------------------------------------------
# Warehouse environment
# ---------------------------------------------------------------------------

class WarehouseEnv(MultiAgentEnv):
    """Warehouse gridworld environment with continuous robot motion.

    Args:
        num_robots: Number of robots (4–12).
        layout: Warehouse layout.
        dynamics: Robot dynamics model.
        sensor_range: Observation range (m).
        max_steps: Maximum episode length.
        dt: Simulation time step (s).
        auto_task: If ``True``, automatically generate tasks on reset.
    """

    OBS_DIM = 5  # per-robot: [x, y, heading, carrying, battery]
    ACT_DIM = 2  # [linear_vel, angular_vel]

    def __init__(
        self,
        num_robots: int = 6,
        layout: Optional[WarehouseLayout] = None,
        dynamics: Optional[RobotDynamics] = None,
        sensor_range: float = 5.0,
        max_steps: int = 1000,
        dt: float = 0.1,
        auto_task: bool = True,
        stepping: Optional[AsyncSteppingSemantics] = None,
        timing_configs: Optional[Dict[str, AgentTimingConfig]] = None,
        clock: Optional[EnvironmentClock] = None,
    ) -> None:
        num_robots = max(4, min(12, num_robots))
        agent_ids = [f"robot_{i}" for i in range(num_robots)]
        super().__init__(
            agent_ids=agent_ids,
            stepping=stepping,
            timing_configs=timing_configs,
            clock=clock or EnvironmentClock(dt=dt),
        )
        self.layout = layout or WarehouseLayout.default_warehouse()
        self.dynamics = dynamics or RobotDynamics(dt=dt)
        self.sensor_range = sensor_range
        self.max_steps = max_steps
        self.dt = dt
        self.auto_task = auto_task

        self._robots: Dict[str, RobotState] = {}
        self._tasks = TaskAssignment()
        self._items_on_ground: Dict[str, Tuple[float, float]] = {}
        self._corridor_conflict = CorridorConflict(self.layout)
        self._deadlock_detector = DeadlockDetection()
        self._collision_log: List[Tuple[int, str, str]] = []

    # -- reset ---------------------------------------------------------------

    def _reset_impl(self) -> Dict[str, np.ndarray]:
        self._robots.clear()
        self._items_on_ground.clear()
        self._collision_log.clear()
        self._tasks = TaskAssignment()

        # Place robots in bottom corridor
        for i, aid in enumerate(self._agent_ids):
            x = self.layout.cell_size * (2 + i * 2)
            y = self.layout.cell_size * 1.5
            self._robots[aid] = RobotState(x=x, y=y, heading=0.0)

        if self.auto_task:
            self._generate_tasks()

        return {aid: self._observe(aid) for aid in self._agent_ids}

    def _generate_tasks(self) -> None:
        pickups = self.layout.find_cells(CellType.PICKUP_STATION)
        deliveries = self.layout.find_cells(CellType.DELIVERY_STATION)
        if not pickups or not deliveries:
            return
        rng = np.random.default_rng(42)
        for i, aid in enumerate(self._agent_ids):
            p = pickups[i % len(pickups)]
            d = deliveries[i % len(deliveries)]
            task = Task(
                task_id=f"task_{i}",
                pickup_pos=p,
                delivery_pos=d,
                item_id=f"item_{i}",
                deadline=self.max_steps,
            )
            self._tasks.assign(aid, task)
            # Place item at pickup
            wp = self.layout.world_position(*p)
            self._items_on_ground[f"item_{i}"] = (float(wp[0]), float(wp[1]))

    # -- step ----------------------------------------------------------------

    def _step_single(
        self, agent_id: str, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        rs = self._robots[agent_id]
        new_rs = self.dynamics.step(rs, action)

        # Clamp to world bounds
        new_rs.x = float(np.clip(new_rs.x, 0.0, self.layout.real_width))
        new_rs.y = float(np.clip(new_rs.y, 0.0, self.layout.real_height))

        # Wall collision — prevent movement into walls
        row, col = self.layout.cell_index(new_rs.x, new_rs.y)
        if not self.layout.is_passable(row, col):
            new_rs.x = rs.x
            new_rs.y = rs.y

        self._robots[agent_id] = new_rs

        # Check robot-robot collisions
        collisions: List[str] = []
        for other_id, other_rs in self._robots.items():
            if other_id != agent_id:
                if WarehouseSafetyPredicates.collision(new_rs, other_rs):
                    collisions.append(other_id)
                    self._collision_log.append((self._step_count, agent_id, other_id))

        # Pickup / delivery logic
        self._handle_task_actions(agent_id)

        reward = self._compute_reward(agent_id, collisions)
        done = self._step_count >= self.max_steps
        obs = self._observe(agent_id)

        info: Dict[str, Any] = {
            "collisions": collisions,
            "corridor_conflicts": self._corridor_conflict.detect(self._robots),
            "deadlocks": self._deadlock_detector.detect(self._robots),
            "carrying": new_rs.carrying_item,
            "battery": new_rs.battery,
        }
        return obs, reward, done, info

    def _handle_task_actions(self, agent_id: str) -> None:
        """Auto-pickup / auto-deliver when robot is near the relevant station."""
        rs = self._robots[agent_id]
        task = self._tasks.current_task(agent_id)
        if task is None:
            return
        # Pickup
        if rs.carrying_item is None:
            pickup_world = self.layout.world_position(*task.pickup_pos)
            if float(np.linalg.norm(rs.position() - pickup_world)) < self.layout.cell_size:
                if task.item_id in self._items_on_ground:
                    rs.carrying_item = task.item_id
                    del self._items_on_ground[task.item_id]
        # Delivery
        elif rs.carrying_item == task.item_id:
            delivery_world = self.layout.world_position(*task.delivery_pos)
            if float(np.linalg.norm(rs.position() - delivery_world)) < self.layout.cell_size:
                rs.carrying_item = None
                self._tasks.complete_current(agent_id)

    def _compute_reward(self, agent_id: str, collisions: List[str]) -> float:
        if collisions:
            return -50.0
        rs = self._robots[agent_id]
        task = self._tasks.current_task(agent_id)
        if task is None:
            return 1.0  # small reward for having completed all tasks
        # Distance-based shaping
        if rs.carrying_item is None:
            target = self.layout.world_position(*task.pickup_pos)
        else:
            target = self.layout.world_position(*task.delivery_pos)
        dist = float(np.linalg.norm(rs.position() - target))
        return -0.01 * dist

    # -- observation ---------------------------------------------------------

    def _observe(self, agent_id: str) -> np.ndarray:
        ego = self._robots[agent_id]
        obs_parts = [ego.to_array()]
        for other_id in sorted(self._robots):
            if other_id == agent_id:
                continue
            other = self._robots[other_id]
            dist = float(np.linalg.norm(ego.position() - other.position()))
            if dist <= self.sensor_range:
                obs_parts.append(other.to_array())
        return np.concatenate(obs_parts)

    # -- state ---------------------------------------------------------------

    def _get_state_impl(self) -> WarehouseState:
        return WarehouseState(
            robots={k: copy.deepcopy(v) for k, v in self._robots.items()},
            tasks=copy.deepcopy(self._tasks),
            items_on_ground=dict(self._items_on_ground),
            tick=self._step_count,
        )

    def _set_state_impl(self, state: EnvironmentState) -> None:
        assert isinstance(state, WarehouseState)
        self._robots = {k: copy.deepcopy(v) for k, v in state.robots.items()}
        self._tasks = copy.deepcopy(state.tasks)
        self._items_on_ground = dict(state.items_on_ground)
        self._step_count = state.tick

    # -- spaces --------------------------------------------------------------

    def observation_space(self, agent_id: str) -> Dict[str, Any]:
        max_obs = self.num_agents
        return {
            "shape": (max_obs * self.OBS_DIM,),
            "dtype": "float64",
            "low": -np.inf,
            "high": np.inf,
        }

    def action_space(self, agent_id: str) -> Dict[str, Any]:
        return {
            "shape": (self.ACT_DIM,),
            "dtype": "float64",
            "low": np.array([-self.dynamics.max_linear, -self.dynamics.max_angular]),
            "high": np.array([self.dynamics.max_linear, self.dynamics.max_angular]),
        }

    # -- accessors -----------------------------------------------------------

    @property
    def robots(self) -> Dict[str, RobotState]:
        return dict(self._robots)

    @property
    def tasks(self) -> TaskAssignment:
        return self._tasks

    @property
    def collision_log(self) -> List[Tuple[int, str, str]]:
        return list(self._collision_log)

    def safety_predicates(self) -> WarehouseSafetyPredicates:
        return WarehouseSafetyPredicates()

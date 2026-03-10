"""
usability_oracle.scheduling.types — Task scheduling under bounded rationality.

Models scheduling of cognitively costly UI interaction tasks with deadlines,
priorities, and bounded-rational time-allocation constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

from usability_oracle.core.types import Interval


# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════

@unique
class TaskPriority(Enum):
    """Priority levels for scheduled interaction tasks."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"

    @property
    def numeric(self) -> int:
        return {"critical": 4, "high": 3, "medium": 2, "low": 1, "background": 0}[self.value]

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, TaskPriority):
            return self.numeric < other.numeric
        return NotImplemented

    def __str__(self) -> str:
        return self.value


@unique
class ScheduleStatus(Enum):
    """Status of a scheduled task."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    MISSED_DEADLINE = "missed_deadline"
    PREEMPTED = "preempted"
    BLOCKED = "blocked"


# ═══════════════════════════════════════════════════════════════════════════
# DeadlineModel — time constraints for task completion
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class DeadlineModel:
    """Deadline and time-budget constraints for a task.

    Models both hard deadlines (task fails if missed) and soft deadlines
    (utility degrades gracefully after the deadline).

    Attributes
    ----------
    hard_deadline_s : Optional[float]
        Hard deadline in seconds from task arrival (None ⟹ no hard deadline).
    soft_deadline_s : Optional[float]
        Soft deadline in seconds (utility starts degrading).
    urgency_decay_rate : float
        Exponential decay rate for utility after soft deadline.
        Utility multiplier = exp(-urgency_decay_rate * (t - soft_deadline)).
    time_budget_s : Interval
        Uncertainty interval for estimated task completion time.
    """

    hard_deadline_s: Optional[float] = None
    soft_deadline_s: Optional[float] = None
    urgency_decay_rate: float = 0.1
    time_budget_s: Interval = field(default_factory=lambda: Interval(0.0, float("inf")))

    @property
    def has_hard_deadline(self) -> bool:
        return self.hard_deadline_s is not None

    @property
    def has_soft_deadline(self) -> bool:
        return self.soft_deadline_s is not None

    def utility_multiplier(self, completion_time_s: float) -> float:
        """Compute utility multiplier given actual completion time."""
        import math
        if self.hard_deadline_s is not None and completion_time_s > self.hard_deadline_s:
            return 0.0
        if self.soft_deadline_s is not None and completion_time_s > self.soft_deadline_s:
            overshoot = completion_time_s - self.soft_deadline_s
            return math.exp(-self.urgency_decay_rate * overshoot)
        return 1.0


# ═══════════════════════════════════════════════════════════════════════════
# SchedulingConstraint — general constraints on schedule feasibility
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SchedulingConstraint:
    """A constraint on the scheduling of interaction tasks.

    Attributes
    ----------
    constraint_id : str
        Unique identifier.
    description : str
        Human-readable description.
    task_ids : tuple[str, ...]
        Task ids this constraint applies to.
    constraint_type : str
        Type of constraint: "precedence", "mutual_exclusion", "resource", "temporal".
    parameters : Mapping[str, Any]
        Constraint-specific parameters.
    """

    constraint_id: str
    description: str
    task_ids: Tuple[str, ...] = ()
    constraint_type: str = "precedence"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_precedence(self) -> bool:
        return self.constraint_type == "precedence"

    @property
    def is_resource(self) -> bool:
        return self.constraint_type == "resource"


# ═══════════════════════════════════════════════════════════════════════════
# ScheduledTask — a task with scheduling metadata
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ScheduledTask:
    """An interaction task with scheduling metadata.

    Attributes
    ----------
    task_id : str
        Unique task identifier.
    name : str
        Human-readable name.
    priority : TaskPriority
        Scheduling priority.
    estimated_duration_s : float
        Mean estimated duration in seconds.
    cognitive_cost_bits : float
        Expected information-processing cost in bits.
    deadline : Optional[DeadlineModel]
        Deadline constraints (None if unconstrained).
    dependencies : tuple[str, ...]
        Task ids that must complete before this task can start.
    status : ScheduleStatus
        Current scheduling status.
    arrival_time_s : float
        Time at which the task becomes available for scheduling.
    metadata : Mapping[str, Any]
        Additional task metadata.
    """

    task_id: str
    name: str
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_duration_s: float = 0.0
    cognitive_cost_bits: float = 0.0
    deadline: Optional[DeadlineModel] = None
    dependencies: Tuple[str, ...] = ()
    status: ScheduleStatus = ScheduleStatus.PENDING
    arrival_time_s: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """True if the task has no unmet dependencies and is pending."""
        return self.status == ScheduleStatus.PENDING and len(self.dependencies) == 0

    @property
    def has_deadline(self) -> bool:
        return self.deadline is not None and self.deadline.has_hard_deadline


# ═══════════════════════════════════════════════════════════════════════════
# PriorityQueue — ordered collection of tasks awaiting scheduling
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PriorityQueue:
    """Immutable snapshot of the scheduling priority queue.

    Attributes
    ----------
    tasks : tuple[ScheduledTask, ...]
        Tasks ordered by priority (highest first).
    current_time_s : float
        Current simulation time.
    total_cognitive_budget_bits : float
        Total information-processing budget available.
    consumed_budget_bits : float
        Budget consumed so far.
    """

    tasks: Tuple[ScheduledTask, ...] = ()
    current_time_s: float = 0.0
    total_cognitive_budget_bits: float = float("inf")
    consumed_budget_bits: float = 0.0

    @property
    def remaining_budget_bits(self) -> float:
        return self.total_cognitive_budget_bits - self.consumed_budget_bits

    @property
    def queue_length(self) -> int:
        return len(self.tasks)

    @property
    def next_task(self) -> Optional[ScheduledTask]:
        """Highest-priority ready task, or None."""
        for t in self.tasks:
            if t.status in (ScheduleStatus.PENDING, ScheduleStatus.READY):
                return t
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Schedule — complete schedule assignment
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Schedule:
    """A complete task schedule mapping tasks to time slots.

    Attributes
    ----------
    schedule_id : str
        Unique schedule identifier.
    assignments : tuple[tuple[str, float, float], ...]
        Tuples of (task_id, start_time_s, end_time_s).
    makespan_s : float
        Total schedule duration (latest end time − earliest start time).
    total_cognitive_cost_bits : float
        Aggregate information-processing cost of the schedule.
    deadline_violations : tuple[str, ...]
        Task ids whose deadlines are violated by this schedule.
    constraints : tuple[SchedulingConstraint, ...]
        Constraints that were considered.
    feasible : bool
        Whether all hard constraints are satisfied.
    objective_value : float
        Optimisation objective value (lower is better).
    """

    schedule_id: str
    assignments: Tuple[Tuple[str, float, float], ...] = ()
    makespan_s: float = 0.0
    total_cognitive_cost_bits: float = 0.0
    deadline_violations: Tuple[str, ...] = ()
    constraints: Tuple[SchedulingConstraint, ...] = ()
    feasible: bool = True
    objective_value: float = 0.0

    @property
    def task_count(self) -> int:
        return len(self.assignments)

    @property
    def has_violations(self) -> bool:
        return len(self.deadline_violations) > 0

    def start_time(self, task_id: str) -> Optional[float]:
        """Return the start time for a given task, or None if not scheduled."""
        for tid, start, _ in self.assignments:
            if tid == task_id:
                return start
        return None


__all__ = [
    "DeadlineModel",
    "PriorityQueue",
    "Schedule",
    "ScheduleStatus",
    "ScheduledTask",
    "SchedulingConstraint",
    "TaskPriority",
]

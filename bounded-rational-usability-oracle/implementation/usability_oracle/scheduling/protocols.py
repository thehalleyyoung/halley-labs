"""
usability_oracle.scheduling.protocols — Task scheduling protocols.

Structural interfaces for scheduling interaction tasks under bounded
rationality, deadline prediction, and schedule optimisation.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usability_oracle.scheduling.types import (
        DeadlineModel,
        PriorityQueue,
        Schedule,
        ScheduledTask,
        SchedulingConstraint,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TaskScheduler — core scheduling engine
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class TaskScheduler(Protocol):
    """Schedule interaction tasks under bounded-rationality constraints.

    The scheduler accounts for cognitive cost budgets, deadlines,
    task dependencies, and resource channel availability.
    """

    def schedule(
        self,
        tasks: Sequence[ScheduledTask],
        constraints: Sequence[SchedulingConstraint],
        *,
        cognitive_budget_bits: float = float("inf"),
    ) -> Schedule:
        """Compute a feasible schedule for the given tasks.

        Parameters
        ----------
        tasks : Sequence[ScheduledTask]
            Tasks to schedule.
        constraints : Sequence[SchedulingConstraint]
            Scheduling constraints (precedence, resource, etc.).
        cognitive_budget_bits : float
            Total information-processing budget available.

        Returns
        -------
        Schedule
            The computed schedule.
        """
        ...

    def schedule_incremental(
        self,
        current_schedule: Schedule,
        new_task: ScheduledTask,
    ) -> Schedule:
        """Incrementally insert a new task into an existing schedule.

        Parameters
        ----------
        current_schedule : Schedule
            The current schedule.
        new_task : ScheduledTask
            Task to insert.

        Returns
        -------
        Schedule
            Updated schedule.
        """
        ...

    def is_feasible(
        self,
        schedule: Schedule,
    ) -> bool:
        """Check whether a schedule satisfies all constraints."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# DeadlinePredictor — predict whether deadlines will be met
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class DeadlinePredictor(Protocol):
    """Predict deadline-miss probability for scheduled tasks.

    Uses cognitive cost distributions and bounded-rationality models
    to estimate the probability of completing tasks within deadlines.
    """

    def predict_miss_probability(
        self,
        task: ScheduledTask,
        *,
        beta: float = 1.0,
    ) -> float:
        """Predict the probability that this task misses its deadline.

        Parameters
        ----------
        task : ScheduledTask
            The scheduled task with deadline.
        beta : float
            Rationality parameter (higher = more time-optimal behaviour).

        Returns
        -------
        float
            Probability in [0, 1] of missing the hard deadline.
        """
        ...

    def expected_completion_time(
        self,
        task: ScheduledTask,
        *,
        beta: float = 1.0,
    ) -> float:
        """Predict expected completion time in seconds.

        Parameters
        ----------
        task : ScheduledTask
            The task.
        beta : float
            Rationality parameter.

        Returns
        -------
        float
            Expected completion time in seconds.
        """
        ...

    def compute_slack(
        self,
        schedule: Schedule,
        task_id: str,
    ) -> Optional[float]:
        """Compute the slack (spare time before deadline) for a task.

        Parameters
        ----------
        schedule : Schedule
            The schedule.
        task_id : str
            Which task to compute slack for.

        Returns
        -------
        Optional[float]
            Slack in seconds, or None if the task has no deadline.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# ScheduleOptimizer — optimise schedule for cognitive cost or makespan
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ScheduleOptimizer(Protocol):
    """Optimise a schedule to minimise cognitive cost, makespan, or deadline violations.

    Implementations may use priority-based heuristics, constraint
    programming, or MDP-based optimisation.
    """

    def optimize(
        self,
        tasks: Sequence[ScheduledTask],
        constraints: Sequence[SchedulingConstraint],
        *,
        objective: str = "cognitive_cost",
        time_limit_s: float = 10.0,
    ) -> Schedule:
        """Find an optimal or near-optimal schedule.

        Parameters
        ----------
        tasks : Sequence[ScheduledTask]
            Tasks to schedule.
        constraints : Sequence[SchedulingConstraint]
            Hard and soft constraints.
        objective : str
            Optimisation objective: ``"cognitive_cost"``, ``"makespan"``,
            ``"deadline_violations"``, ``"weighted_sum"``.
        time_limit_s : float
            Maximum solver time in seconds.

        Returns
        -------
        Schedule
            Optimised schedule.
        """
        ...

    def pareto_front(
        self,
        tasks: Sequence[ScheduledTask],
        constraints: Sequence[SchedulingConstraint],
        *,
        n_solutions: int = 10,
    ) -> Sequence[Schedule]:
        """Compute the Pareto front trading off makespan vs. cognitive cost.

        Parameters
        ----------
        tasks : Sequence[ScheduledTask]
            Tasks to schedule.
        constraints : Sequence[SchedulingConstraint]
            Constraints.
        n_solutions : int
            Desired number of Pareto-optimal solutions.

        Returns
        -------
        Sequence[Schedule]
            Pareto-optimal schedules.
        """
        ...


__all__ = [
    "DeadlinePredictor",
    "ScheduleOptimizer",
    "TaskScheduler",
]

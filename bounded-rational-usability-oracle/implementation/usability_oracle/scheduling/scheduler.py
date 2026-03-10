"""
usability_oracle.scheduling.scheduler — Bounded-rational task scheduler.

Implements scheduling algorithms for cognitive UI tasks under bounded
rationality.  The key insight is that human users do *not* compute
globally optimal schedules; instead they make softmax-rational choices
over locally available scheduling options, weighted by an
information-processing capacity parameter β (bits).

Algorithms
----------
* **Earliest-Deadline First (EDF)** with bounded-rationality noise.
* **Rate-Monotonic Scheduling (RMS)** adapted for human periodic tasks.
* **Weighted Shortest Job First (WSJF)** with information cost.
* **Softmax schedule selection** across candidate orderings.

Schedule quality is measured by makespan, total tardiness, and aggregate
cognitive load (bits).

References
----------
* Liu, C. L. & Layland, J. W. (1973). Scheduling Algorithms for
  Multiprogramming in a Hard-Real-Time Environment.  *JACM*, 20(1).
* Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a Theory of
  Decision-Making with Information-Processing Costs.  *Proc. R. Soc. A*.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.scheduling.types import (
    DeadlineModel,
    PriorityQueue,
    Schedule,
    ScheduleStatus,
    ScheduledTask,
    SchedulingConstraint,
    TaskPriority,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _softmax(values: np.ndarray, beta: float) -> np.ndarray:
    """Numerically stable softmax with rationality parameter *beta*.

    Parameters
    ----------
    values : np.ndarray
        Utility/priority values (higher is better).
    beta : float
        Inverse temperature.  β → ∞ recovers argmax; β → 0 gives uniform.

    Returns
    -------
    np.ndarray
        Probability vector summing to 1.
    """
    if len(values) == 0:
        return np.array([], dtype=np.float64)
    v = np.asarray(values, dtype=np.float64) * beta
    v -= v.max()
    e = np.exp(v)
    return e / e.sum()


def _topological_order(
    tasks: Sequence[ScheduledTask],
) -> List[str]:
    """Return a topological ordering of task ids respecting dependencies.

    Raises
    ------
    ValueError
        If the dependency graph contains a cycle.
    """
    task_map: Dict[str, ScheduledTask] = {t.task_id: t for t in tasks}
    in_degree: Dict[str, int] = {t.task_id: 0 for t in tasks}
    children: Dict[str, List[str]] = {t.task_id: [] for t in tasks}

    for t in tasks:
        for dep in t.dependencies:
            if dep in task_map:
                in_degree[t.task_id] += 1
                children[dep].append(t.task_id)

    queue = [tid for tid, d in in_degree.items() if d == 0]
    order: List[str] = []
    while queue:
        queue.sort()
        node = queue.pop(0)
        order.append(node)
        for ch in children[node]:
            in_degree[ch] -= 1
            if in_degree[ch] == 0:
                queue.append(ch)

    if len(order) != len(tasks):
        raise ValueError("Dependency cycle detected in task graph")
    return order


def _effective_deadline(task: ScheduledTask, current_time: float) -> float:
    """Return the effective deadline for ordering; inf if none."""
    if task.deadline is not None and task.deadline.hard_deadline_s is not None:
        return task.arrival_time_s + task.deadline.hard_deadline_s
    if task.deadline is not None and task.deadline.soft_deadline_s is not None:
        return task.arrival_time_s + task.deadline.soft_deadline_s
    return float("inf")


# ═══════════════════════════════════════════════════════════════════════════
# ScheduleMetrics — quality metrics for a schedule
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ScheduleMetrics:
    """Quality metrics for a computed schedule.

    Attributes
    ----------
    makespan_s : float
        Latest end time minus earliest start time.
    total_tardiness_s : float
        Sum of max(0, completion − deadline) over all tasks.
    max_tardiness_s : float
        Worst-case tardiness across all tasks.
    total_cognitive_cost_bits : float
        Sum of cognitive costs for all scheduled tasks.
    utilization : float
        Fraction of makespan occupied by useful work.
    deadline_miss_count : int
        Number of tasks whose hard deadlines are violated.
    """

    makespan_s: float = 0.0
    total_tardiness_s: float = 0.0
    max_tardiness_s: float = 0.0
    total_cognitive_cost_bits: float = 0.0
    utilization: float = 0.0
    deadline_miss_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# ResourceChannel — models visual / motor / cognitive channels
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ResourceChannel:
    """A cognitive resource channel with limited capacity.

    Attributes
    ----------
    name : str
        Channel name (e.g. ``"visual"``, ``"motor"``, ``"cognitive"``).
    capacity : float
        Maximum utilization (1.0 = fully occupied).
    current_load : float
        Current utilization in [0, capacity].
    """

    name: str
    capacity: float = 1.0
    current_load: float = 0.0

    @property
    def available(self) -> float:
        return max(0.0, self.capacity - self.current_load)


# ═══════════════════════════════════════════════════════════════════════════
# BoundedRationalScheduler
# ═══════════════════════════════════════════════════════════════════════════

class BoundedRationalScheduler:
    """Schedule cognitive tasks under bounded-rationality constraints.

    Implements the :class:`~usability_oracle.scheduling.protocols.TaskScheduler`
    protocol.  The scheduler selects task orderings via softmax-rational
    decisions parameterised by *β* (bits), combining classical scheduling
    heuristics with information-theoretic cost.

    Parameters
    ----------
    beta : float
        Rationality parameter (inverse temperature).  Higher values produce
        schedules closer to the optimal heuristic; lower values introduce
        exploration noise modelling human sub-optimality.
    algorithm : str
        Base scheduling heuristic: ``"edf"``, ``"rms"``, ``"wsjf"``.
    channels : Sequence[ResourceChannel]
        Available cognitive resource channels.  Defaults to a single
        general-purpose channel.
    rng_seed : Optional[int]
        Seed for reproducibility.
    """

    # Supported heuristics
    ALGORITHMS = ("edf", "rms", "wsjf")

    def __init__(
        self,
        beta: float = 1.0,
        algorithm: str = "edf",
        channels: Optional[Sequence[ResourceChannel]] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}', expected one of {self.ALGORITHMS}"
            )
        self._beta = beta
        self._algorithm = algorithm
        self._channels: Tuple[ResourceChannel, ...] = tuple(
            channels if channels is not None
            else [ResourceChannel("cognitive")]
        )
        self._rng = np.random.default_rng(rng_seed)

    # ── public interface (TaskScheduler protocol) ──────────────────────

    def schedule(
        self,
        tasks: Sequence[ScheduledTask],
        constraints: Sequence[SchedulingConstraint] = (),
        *,
        cognitive_budget_bits: float = float("inf"),
    ) -> Schedule:
        """Compute a feasible schedule for *tasks*.

        Parameters
        ----------
        tasks : Sequence[ScheduledTask]
            Tasks to schedule.
        constraints : Sequence[SchedulingConstraint]
            Hard/soft constraints.
        cognitive_budget_bits : float
            Total processing budget in bits.

        Returns
        -------
        Schedule
        """
        if not tasks:
            return Schedule(schedule_id=uuid.uuid4().hex)

        task_map: Dict[str, ScheduledTask] = {t.task_id: t for t in tasks}
        order = _topological_order(tasks)

        # Order tasks using the selected heuristic + softmax noise
        ordered_ids = self._heuristic_order(
            [task_map[tid] for tid in order],
        )

        # Build assignments greedily
        assignments, violations, total_cost = self._build_assignments(
            ordered_ids, task_map, cognitive_budget_bits,
        )

        starts = [s for _, s, _ in assignments]
        ends = [e for _, _, e in assignments]
        makespan = (max(ends) - min(starts)) if assignments else 0.0

        return Schedule(
            schedule_id=uuid.uuid4().hex,
            assignments=tuple(assignments),
            makespan_s=makespan,
            total_cognitive_cost_bits=total_cost,
            deadline_violations=tuple(violations),
            constraints=tuple(constraints),
            feasible=len(violations) == 0,
            objective_value=makespan + total_cost,
        )

    def schedule_incremental(
        self,
        current_schedule: Schedule,
        new_task: ScheduledTask,
    ) -> Schedule:
        """Insert *new_task* into *current_schedule*.

        Finds the best insertion point by evaluating the objective at each
        feasible position and choosing via softmax.

        Parameters
        ----------
        current_schedule : Schedule
            Existing schedule.
        new_task : ScheduledTask
            Task to insert.

        Returns
        -------
        Schedule
        """
        if not current_schedule.assignments:
            return self.schedule([new_task])

        existing = list(current_schedule.assignments)
        n = len(existing)
        candidates: List[List[Tuple[str, float, float]]] = []
        objectives: List[float] = []

        for pos in range(n + 1):
            trial = list(existing)
            if pos == 0:
                start = max(0.0, new_task.arrival_time_s)
            else:
                start = max(existing[pos - 1][2], new_task.arrival_time_s)
            end = start + new_task.estimated_duration_s
            trial.insert(pos, (new_task.task_id, start, end))

            # Shift subsequent tasks if they overlap
            for i in range(pos + 1, len(trial)):
                tid_i, s_i, e_i = trial[i]
                dur_i = e_i - s_i
                if s_i < trial[i - 1][2]:
                    s_i = trial[i - 1][2]
                    e_i = s_i + dur_i
                trial[i] = (tid_i, s_i, e_i)

            makespan = trial[-1][2] - trial[0][1] if trial else 0.0
            objectives.append(makespan + new_task.cognitive_cost_bits)
            candidates.append(trial)

        # Select insertion point via softmax (negate because lower is better)
        probs = _softmax(-np.array(objectives), self._beta)
        idx = int(self._rng.choice(len(candidates), p=probs))
        chosen = candidates[idx]

        ends = [e for _, _, e in chosen]
        starts = [s for _, s, _ in chosen]
        makespan = (max(ends) - min(starts)) if chosen else 0.0

        return Schedule(
            schedule_id=uuid.uuid4().hex,
            assignments=tuple(tuple(a) for a in chosen),
            makespan_s=makespan,
            total_cognitive_cost_bits=(
                current_schedule.total_cognitive_cost_bits
                + new_task.cognitive_cost_bits
            ),
            deadline_violations=current_schedule.deadline_violations,
            constraints=current_schedule.constraints,
            feasible=current_schedule.feasible,
            objective_value=objectives[idx],
        )

    def is_feasible(self, schedule: Schedule) -> bool:
        """Check whether *schedule* satisfies all hard constraints.

        Parameters
        ----------
        schedule : Schedule

        Returns
        -------
        bool
        """
        return schedule.feasible and not schedule.has_violations

    # ── heuristic orderings ────────────────────────────────────────────

    def _heuristic_order(
        self,
        tasks: Sequence[ScheduledTask],
    ) -> List[str]:
        """Apply the configured heuristic and return an ordered list of ids."""
        if self._algorithm == "edf":
            return self._edf_order(tasks)
        elif self._algorithm == "rms":
            return self._rms_order(tasks)
        else:
            return self._wsjf_order(tasks)

    def _edf_order(self, tasks: Sequence[ScheduledTask]) -> List[str]:
        """Earliest-Deadline First with bounded-rationality noise.

        Each task's effective deadline is used as a priority value; the
        schedule order is sampled via softmax rather than deterministic
        sorting, modelling imperfect temporal perception.
        """
        if not tasks:
            return []
        deadlines = np.array(
            [_effective_deadline(t, 0.0) for t in tasks], dtype=np.float64,
        )
        # Negate: earlier deadline → higher priority
        priorities = -deadlines
        return self._softmax_order(tasks, priorities)

    def _rms_order(self, tasks: Sequence[ScheduledTask]) -> List[str]:
        """Rate-Monotonic Scheduling adapted for human periodic tasks.

        Tasks with shorter estimated durations receive higher priority,
        following the RMS intuition that shorter-period tasks are
        scheduled first.
        """
        if not tasks:
            return []
        durations = np.array(
            [t.estimated_duration_s for t in tasks], dtype=np.float64,
        )
        # Shorter duration → higher priority
        priorities = -durations
        return self._softmax_order(tasks, priorities)

    def _wsjf_order(self, tasks: Sequence[ScheduledTask]) -> List[str]:
        """Weighted Shortest Job First with information cost.

        Priority = (urgency + importance) / (duration + info_cost), where
        info_cost = cognitive_cost_bits / β captures the information-
        theoretic overhead of processing each task.
        """
        if not tasks:
            return []
        priorities = np.zeros(len(tasks), dtype=np.float64)
        for i, t in enumerate(tasks):
            urgency = 1.0 / max(_effective_deadline(t, 0.0), 0.01)
            importance = t.priority.numeric
            duration = max(t.estimated_duration_s, 0.01)
            info_cost = t.cognitive_cost_bits / max(self._beta, 0.01)
            priorities[i] = (urgency + importance) / (duration + info_cost)
        return self._softmax_order(tasks, priorities)

    def _softmax_order(
        self,
        tasks: Sequence[ScheduledTask],
        priorities: np.ndarray,
    ) -> List[str]:
        """Sample a full ordering by repeatedly drawing from softmax."""
        n = len(tasks)
        indices = list(range(n))
        order: List[str] = []
        remaining_priorities = priorities.copy()

        for _ in range(n):
            probs = _softmax(remaining_priorities[indices], self._beta)
            chosen_local = int(self._rng.choice(len(indices), p=probs))
            chosen_global = indices[chosen_local]
            order.append(tasks[chosen_global].task_id)
            indices.pop(chosen_local)

        return order

    # ── assignment builder ─────────────────────────────────────────────

    def _build_assignments(
        self,
        ordered_ids: List[str],
        task_map: Dict[str, ScheduledTask],
        budget: float,
    ) -> Tuple[
        List[Tuple[str, float, float]],
        List[str],
        float,
    ]:
        """Greedily assign start/end times following *ordered_ids*."""
        assignments: List[Tuple[str, float, float]] = []
        violations: List[str] = []
        total_cost = 0.0
        current_time = 0.0
        end_times: Dict[str, float] = {}

        for tid in ordered_ids:
            task = task_map[tid]
            # Respect dependency ordering: start after all dependencies finish
            dep_end = max(
                (end_times[d] for d in task.dependencies if d in end_times),
                default=0.0,
            )
            start = max(current_time, task.arrival_time_s, dep_end)
            if total_cost + task.cognitive_cost_bits > budget:
                continue  # skip if over budget
            end = start + task.estimated_duration_s
            assignments.append((tid, start, end))

            # Check deadline violation
            if task.deadline is not None and task.deadline.hard_deadline_s is not None:
                abs_deadline = task.arrival_time_s + task.deadline.hard_deadline_s
                if end > abs_deadline:
                    violations.append(tid)

            total_cost += task.cognitive_cost_bits
            current_time = end
            end_times[tid] = end

        return assignments, violations, total_cost

    # ── metrics ────────────────────────────────────────────────────────

    @staticmethod
    def compute_metrics(
        schedule: Schedule,
        tasks: Sequence[ScheduledTask],
    ) -> ScheduleMetrics:
        """Compute quality metrics for *schedule*.

        Parameters
        ----------
        schedule : Schedule
            The schedule to evaluate.
        tasks : Sequence[ScheduledTask]
            Original tasks (needed for deadline / cost info).

        Returns
        -------
        ScheduleMetrics
        """
        if not schedule.assignments:
            return ScheduleMetrics()

        task_map = {t.task_id: t for t in tasks}
        starts = [s for _, s, _ in schedule.assignments]
        ends = [e for _, _, e in schedule.assignments]
        makespan = max(ends) - min(starts)

        total_tardiness = 0.0
        max_tardiness = 0.0
        miss_count = 0
        total_cost = 0.0
        total_work = 0.0

        for tid, start, end in schedule.assignments:
            t = task_map.get(tid)
            if t is None:
                continue
            total_cost += t.cognitive_cost_bits
            total_work += end - start
            if t.deadline is not None and t.deadline.hard_deadline_s is not None:
                abs_dl = t.arrival_time_s + t.deadline.hard_deadline_s
                tardiness = max(0.0, end - abs_dl)
                total_tardiness += tardiness
                max_tardiness = max(max_tardiness, tardiness)
                if tardiness > 0:
                    miss_count += 1

        utilization = total_work / makespan if makespan > 0 else 0.0

        return ScheduleMetrics(
            makespan_s=makespan,
            total_tardiness_s=total_tardiness,
            max_tardiness_s=max_tardiness,
            total_cognitive_cost_bits=total_cost,
            utilization=utilization,
            deadline_miss_count=miss_count,
        )

    # ── feasibility analysis ───────────────────────────────────────────

    @staticmethod
    def check_feasibility(
        tasks: Sequence[ScheduledTask],
        capacity_bits: float = float("inf"),
    ) -> Tuple[bool, str]:
        """Check whether a feasible schedule exists.

        Uses the Liu–Layland utilization bound for EDF: a task set is
        EDF-schedulable if the total utilization U ≤ 1.

        Parameters
        ----------
        tasks : Sequence[ScheduledTask]
            Tasks to analyse.
        capacity_bits : float
            Total cognitive budget.

        Returns
        -------
        (bool, str)
            ``(feasible, reason)``
        """
        if not tasks:
            return True, "Empty task set is trivially feasible"

        total_cost = sum(t.cognitive_cost_bits for t in tasks)
        if total_cost > capacity_bits:
            return False, (
                f"Total cognitive cost {total_cost:.2f} bits exceeds "
                f"capacity {capacity_bits:.2f} bits"
            )

        # Utilization bound for EDF on tasks with deadlines
        tasks_with_deadlines = [
            t for t in tasks
            if t.deadline is not None and t.deadline.hard_deadline_s is not None
        ]
        if tasks_with_deadlines:
            utilization = sum(
                t.estimated_duration_s / (t.deadline.hard_deadline_s)  # type: ignore[operator]
                for t in tasks_with_deadlines
                if t.deadline.hard_deadline_s > 0  # type: ignore[operator]
            )
            if utilization > 1.0:
                return False, (
                    f"EDF utilization bound violated: U = {utilization:.3f} > 1"
                )

        return True, "Task set is feasible under EDF"

    # ── online scheduling ──────────────────────────────────────────────

    def handle_task_arrival(
        self,
        current_schedule: Schedule,
        new_task: ScheduledTask,
        current_time: float,
    ) -> Schedule:
        """React to a new task arriving at *current_time*.

        Reschedules remaining tasks (those starting after *current_time*)
        along with the new task.

        Parameters
        ----------
        current_schedule : Schedule
            Running schedule.
        new_task : ScheduledTask
            Newly arrived task.
        current_time : float
            Current wall-clock time.

        Returns
        -------
        Schedule
            Updated schedule.
        """
        # Partition: completed vs remaining
        completed = [
            a for a in current_schedule.assignments if a[2] <= current_time
        ]
        remaining_ids = {
            a[0] for a in current_schedule.assignments if a[2] > current_time
        }

        # Reconstruct ScheduledTask stubs for remaining assignments
        remaining_tasks = []
        for tid, start, end in current_schedule.assignments:
            if tid in remaining_ids:
                remaining_tasks.append(
                    ScheduledTask(
                        task_id=tid,
                        name=tid,
                        estimated_duration_s=end - start,
                        arrival_time_s=max(start, current_time),
                    )
                )

        all_tasks = remaining_tasks + [
            replace(new_task, arrival_time_s=max(new_task.arrival_time_s, current_time))
        ]

        new_sched = self.schedule(all_tasks)

        # Merge completed prefix
        merged = list(completed) + list(new_sched.assignments)
        starts = [s for _, s, _ in merged]
        ends = [e for _, _, e in merged]
        makespan = (max(ends) - min(starts)) if merged else 0.0

        return Schedule(
            schedule_id=uuid.uuid4().hex,
            assignments=tuple(merged),
            makespan_s=makespan,
            total_cognitive_cost_bits=new_sched.total_cognitive_cost_bits,
            deadline_violations=new_sched.deadline_violations,
            feasible=new_sched.feasible,
            objective_value=new_sched.objective_value,
        )

    def handle_task_completion(
        self,
        current_schedule: Schedule,
        completed_task_id: str,
        completion_time: float,
    ) -> Schedule:
        """Update schedule after a task completes.

        Parameters
        ----------
        current_schedule : Schedule
            Running schedule.
        completed_task_id : str
            Task that just finished.
        completion_time : float
            Actual completion time.

        Returns
        -------
        Schedule
            Updated schedule with remaining tasks compacted.
        """
        updated: List[Tuple[str, float, float]] = []
        clock = completion_time

        for tid, start, end in current_schedule.assignments:
            if tid == completed_task_id:
                updated.append((tid, start, completion_time))
                continue
            dur = end - start
            if start < clock and tid != completed_task_id:
                new_start = clock
                updated.append((tid, new_start, new_start + dur))
                clock = new_start + dur
            else:
                updated.append((tid, start, end))
                clock = max(clock, end)

        starts = [s for _, s, _ in updated]
        ends = [e for _, _, e in updated]
        makespan = (max(ends) - min(starts)) if updated else 0.0

        return replace(
            current_schedule,
            assignments=tuple(updated),
            makespan_s=makespan,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Multi-resource scheduler
# ═══════════════════════════════════════════════════════════════════════════

class MultiResourceScheduler:
    """Schedule tasks across multiple cognitive resource channels.

    Models the human cognitive architecture as parallel resource channels
    (visual, motor, cognitive) following Wickens' Multiple Resource Theory.

    Parameters
    ----------
    channels : Sequence[ResourceChannel]
        Available resource channels with capacities.
    beta : float
        Bounded-rationality parameter.
    rng_seed : Optional[int]
        Random seed.

    References
    ----------
    * Wickens, C. D. (2002). Multiple resources and performance
      prediction. *Theoretical Issues in Ergonomics Science*, 3(2).
    """

    def __init__(
        self,
        channels: Sequence[ResourceChannel],
        beta: float = 1.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        if not channels:
            raise ValueError("At least one resource channel is required")
        self._channels = {ch.name: ch for ch in channels}
        self._beta = beta
        self._rng = np.random.default_rng(rng_seed)

    def schedule(
        self,
        tasks: Sequence[ScheduledTask],
        task_channel_map: Mapping[str, Sequence[str]],
    ) -> Schedule:
        """Schedule tasks across resource channels.

        Tasks are assigned to time slots such that no channel is
        over-utilised at any moment.  Tasks requiring different channels
        may execute concurrently.

        Parameters
        ----------
        tasks : Sequence[ScheduledTask]
            Tasks to schedule.
        task_channel_map : Mapping[str, Sequence[str]]
            Maps task_id → list of required channel names.

        Returns
        -------
        Schedule
        """
        if not tasks:
            return Schedule(schedule_id=uuid.uuid4().hex)

        # Sort by priority with softmax noise
        priorities = np.array(
            [t.priority.numeric for t in tasks], dtype=np.float64,
        )
        probs = _softmax(priorities, self._beta)
        order = list(self._rng.choice(
            len(tasks), size=len(tasks), replace=False, p=probs,
        ))

        # Channel availability timelines (channel → earliest free time)
        channel_free: Dict[str, float] = {name: 0.0 for name in self._channels}
        assignments: List[Tuple[str, float, float]] = []
        violations: List[str] = []
        total_cost = 0.0

        for idx in order:
            task = tasks[idx]
            channels_needed = task_channel_map.get(
                task.task_id, list(self._channels.keys())[:1],
            )
            # Start time = max(free time of all required channels, arrival)
            start = max(
                max((channel_free.get(ch, 0.0) for ch in channels_needed), default=0.0),
                task.arrival_time_s,
            )
            end = start + task.estimated_duration_s
            assignments.append((task.task_id, start, end))

            for ch in channels_needed:
                channel_free[ch] = end

            total_cost += task.cognitive_cost_bits

            if task.deadline and task.deadline.hard_deadline_s is not None:
                if end > task.arrival_time_s + task.deadline.hard_deadline_s:
                    violations.append(task.task_id)

        assignments.sort(key=lambda a: a[1])
        starts = [s for _, s, _ in assignments]
        ends = [e for _, _, e in assignments]
        makespan = (max(ends) - min(starts)) if assignments else 0.0

        return Schedule(
            schedule_id=uuid.uuid4().hex,
            assignments=tuple(assignments),
            makespan_s=makespan,
            total_cognitive_cost_bits=total_cost,
            deadline_violations=tuple(violations),
            feasible=len(violations) == 0,
            objective_value=makespan + total_cost,
        )

    def channel_utilization(
        self,
        schedule: Schedule,
        task_channel_map: Mapping[str, Sequence[str]],
    ) -> Dict[str, float]:
        """Compute per-channel utilization for *schedule*.

        Parameters
        ----------
        schedule : Schedule
        task_channel_map : Mapping[str, Sequence[str]]

        Returns
        -------
        Dict[str, float]
            Channel name → utilization in [0, 1].
        """
        if not schedule.assignments:
            return {name: 0.0 for name in self._channels}

        ends = [e for _, _, e in schedule.assignments]
        starts = [s for _, s, _ in schedule.assignments]
        horizon = max(ends) - min(starts) if ends else 1.0
        if horizon <= 0:
            return {name: 0.0 for name in self._channels}

        channel_busy: Dict[str, float] = {name: 0.0 for name in self._channels}
        for tid, start, end in schedule.assignments:
            channels = task_channel_map.get(
                tid, list(self._channels.keys())[:1],
            )
            dur = end - start
            for ch in channels:
                if ch in channel_busy:
                    channel_busy[ch] += dur

        return {
            name: min(busy / horizon, 1.0)
            for name, busy in channel_busy.items()
        }


__all__ = [
    "BoundedRationalScheduler",
    "MultiResourceScheduler",
    "ResourceChannel",
    "ScheduleMetrics",
]

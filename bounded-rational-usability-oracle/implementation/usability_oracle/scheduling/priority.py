"""
usability_oracle.scheduling.priority — Multi-criteria priority computation.

Computes task priorities from urgency, importance, effort, and frequency,
then aggregates them under bounded-rationality constraints.  The
bounded-rational priority model produces probabilistic priority
distributions parameterised by cognitive capacity β.

Priority inversion detection identifies situations where low-priority
tasks block higher-priority ones in UI task flows — a usability hazard
analogous to priority inversion in real-time operating systems.

References
----------
* Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort*.
* Card, S. K., Moran, T. P., & Newell, A. (1983). *The Psychology of
  Human-Computer Interaction*.  Lawrence Erlbaum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.scheduling.types import (
    Schedule,
    ScheduledTask,
    TaskPriority,
)


# ═══════════════════════════════════════════════════════════════════════════
# Priority criteria
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PriorityCriteria:
    """Weighted criteria for multi-criteria priority computation.

    Attributes
    ----------
    urgency_weight : float
        Weight for deadline proximity.
    importance_weight : float
        Weight for goal-hierarchy importance.
    effort_weight : float
        Weight for cognitive cost (inverted — lower effort ⟹ higher priority).
    frequency_weight : float
        Weight for task frequency (more frequent ⟹ higher priority).
    """

    urgency_weight: float = 0.4
    importance_weight: float = 0.3
    effort_weight: float = 0.2
    frequency_weight: float = 0.1

    def __post_init__(self) -> None:
        total = (
            self.urgency_weight
            + self.importance_weight
            + self.effort_weight
            + self.frequency_weight
        )
        if total <= 0:
            raise ValueError("At least one weight must be positive")

    @property
    def weights(self) -> np.ndarray:
        """Return normalised weight vector."""
        w = np.array([
            self.urgency_weight,
            self.importance_weight,
            self.effort_weight,
            self.frequency_weight,
        ], dtype=np.float64)
        return w / w.sum()


@dataclass(frozen=True, slots=True)
class PriorityScore:
    """Computed priority score for a single task.

    Attributes
    ----------
    task_id : str
        Task identifier.
    urgency : float
        Urgency sub-score in [0, 1].
    importance : float
        Importance sub-score in [0, 1].
    effort : float
        Effort sub-score in [0, 1] (lower effort → higher score).
    frequency : float
        Frequency sub-score in [0, 1].
    composite : float
        Weighted composite priority in [0, 1].
    """

    task_id: str
    urgency: float = 0.0
    importance: float = 0.0
    effort: float = 0.0
    frequency: float = 0.0
    composite: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Priority inversion descriptor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PriorityInversion:
    """Detected priority inversion in a task schedule.

    Attributes
    ----------
    high_priority_task_id : str
        The higher-priority task that is delayed.
    blocking_task_id : str
        The lower-priority task causing the delay.
    inversion_duration_s : float
        Duration of the inversion in seconds.
    severity : float
        Normalised severity in [0, 1].
    """

    high_priority_task_id: str
    blocking_task_id: str
    inversion_duration_s: float = 0.0
    severity: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# PriorityComputer
# ═══════════════════════════════════════════════════════════════════════════

class PriorityComputer:
    """Compute multi-criteria priorities for cognitive tasks.

    Combines urgency, importance, effort, and frequency into a composite
    priority score.  Under bounded rationality, priorities are converted
    to a probability distribution via softmax parameterised by β.

    Parameters
    ----------
    criteria : PriorityCriteria
        Weighting of priority sub-criteria.
    beta : float
        Bounded-rationality parameter.  Higher β makes priority selection
        more deterministic.
    max_cognitive_cost_bits : float
        Normalisation constant for effort scores.
    """

    def __init__(
        self,
        criteria: Optional[PriorityCriteria] = None,
        beta: float = 1.0,
        max_cognitive_cost_bits: float = 20.0,
    ) -> None:
        self._criteria = criteria or PriorityCriteria()
        self._beta = beta
        self._max_cost = max_cognitive_cost_bits

    # ── single-task scoring ────────────────────────────────────────────

    def score_task(
        self,
        task: ScheduledTask,
        current_time: float = 0.0,
        task_frequency: float = 1.0,
        goal_weight: float = 1.0,
    ) -> PriorityScore:
        """Compute priority score for a single task.

        Parameters
        ----------
        task : ScheduledTask
        current_time : float
            Current simulation time.
        task_frequency : float
            Empirical frequency of this task type (higher = more common).
        goal_weight : float
            Importance weight from the user's goal hierarchy.

        Returns
        -------
        PriorityScore
        """
        urgency = self._urgency_score(task, current_time)
        importance = self._importance_score(task, goal_weight)
        effort = self._effort_score(task)
        frequency = self._frequency_score(task_frequency)

        weights = self._criteria.weights
        composite = float(
            weights[0] * urgency
            + weights[1] * importance
            + weights[2] * effort
            + weights[3] * frequency
        )

        return PriorityScore(
            task_id=task.task_id,
            urgency=urgency,
            importance=importance,
            effort=effort,
            frequency=frequency,
            composite=composite,
        )

    def score_tasks(
        self,
        tasks: Sequence[ScheduledTask],
        current_time: float = 0.0,
        frequencies: Optional[Mapping[str, float]] = None,
        goal_weights: Optional[Mapping[str, float]] = None,
    ) -> List[PriorityScore]:
        """Score a batch of tasks.

        Parameters
        ----------
        tasks : Sequence[ScheduledTask]
        current_time : float
        frequencies : Mapping[str, float] | None
            Task id → frequency.
        goal_weights : Mapping[str, float] | None
            Task id → goal-hierarchy weight.

        Returns
        -------
        List[PriorityScore]
            Sorted by composite priority (highest first).
        """
        freq = frequencies or {}
        weights = goal_weights or {}
        scores = [
            self.score_task(
                t,
                current_time=current_time,
                task_frequency=freq.get(t.task_id, 1.0),
                goal_weight=weights.get(t.task_id, 1.0),
            )
            for t in tasks
        ]
        scores.sort(key=lambda s: s.composite, reverse=True)
        return scores

    # ── bounded-rational priority distribution ─────────────────────────

    def priority_distribution(
        self,
        tasks: Sequence[ScheduledTask],
        current_time: float = 0.0,
        frequencies: Optional[Mapping[str, float]] = None,
        goal_weights: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute bounded-rational priority probabilities.

        Returns softmax probabilities over composite scores, so that the
        user selects task *i* with probability proportional to
        ``exp(β · score_i)``.

        Parameters
        ----------
        tasks : Sequence[ScheduledTask]
        current_time : float
        frequencies : Mapping[str, float] | None
        goal_weights : Mapping[str, float] | None

        Returns
        -------
        Dict[str, float]
            Task id → selection probability.
        """
        if not tasks:
            return {}

        scores = self.score_tasks(tasks, current_time, frequencies, goal_weights)
        composites = np.array([s.composite for s in scores], dtype=np.float64)

        # Softmax with β
        v = composites * self._beta
        v -= v.max()
        e = np.exp(v)
        probs = e / e.sum()

        return {
            scores[i].task_id: float(probs[i])
            for i in range(len(scores))
        }

    # ── sub-score functions ────────────────────────────────────────────

    def _urgency_score(self, task: ScheduledTask, current_time: float) -> float:
        """Urgency: sigmoid of (1 − time_remaining / deadline).

        Returns 0 for tasks with no deadline, approaches 1 as deadline
        approaches.
        """
        if task.deadline is None:
            return 0.0
        dl = task.deadline.hard_deadline_s or task.deadline.soft_deadline_s
        if dl is None or dl <= 0:
            return 0.0
        abs_deadline = task.arrival_time_s + dl
        remaining = abs_deadline - current_time
        if remaining <= 0:
            return 1.0
        ratio = task.estimated_duration_s / remaining
        # Sigmoid mapping: ratio near 1 → urgency near 1
        return 1.0 / (1.0 + math.exp(-5.0 * (ratio - 0.5)))

    @staticmethod
    def _importance_score(task: ScheduledTask, goal_weight: float) -> float:
        """Importance from task priority level and goal weight.

        Normalised to [0, 1].
        """
        base = task.priority.numeric / 4.0  # 4 = max (CRITICAL)
        return min(1.0, base * goal_weight)

    def _effort_score(self, task: ScheduledTask) -> float:
        """Effort: inverted normalised cognitive cost.

        Lower effort (cost) yields higher score, encouraging selection
        of easier tasks when all else is equal.
        """
        normalised = min(task.cognitive_cost_bits / self._max_cost, 1.0)
        return 1.0 - normalised

    @staticmethod
    def _frequency_score(frequency: float) -> float:
        """Frequency score following Zipf's law mapping.

        More frequent tasks receive higher priority, modelling practised
        efficiency.

        .. math::

            f_{\\text{score}} = \\frac{\\log(1 + f)}{\\log(1 + f_{\\max})}

        where *f* is the task frequency and *f_max* = 100 is a reference.
        """
        f_max = 100.0
        return math.log(1.0 + frequency) / math.log(1.0 + f_max)

    # ── priority inversion detection ───────────────────────────────────

    @staticmethod
    def detect_inversions(
        schedule: Schedule,
        tasks: Sequence[ScheduledTask],
    ) -> List[PriorityInversion]:
        """Detect priority inversions in *schedule*.

        A priority inversion occurs when a task with lower priority is
        scheduled before a higher-priority task that was ready at the
        same time.

        Parameters
        ----------
        schedule : Schedule
        tasks : Sequence[ScheduledTask]

        Returns
        -------
        List[PriorityInversion]
        """
        task_map = {t.task_id: t for t in tasks}
        assignment_map: Dict[str, Tuple[float, float]] = {}
        for tid, start, end in schedule.assignments:
            assignment_map[tid] = (start, end)

        inversions: List[PriorityInversion] = []

        for i, (tid_i, start_i, end_i) in enumerate(schedule.assignments):
            for j, (tid_j, start_j, end_j) in enumerate(schedule.assignments):
                if j <= i:
                    continue
                t_i = task_map.get(tid_i)
                t_j = task_map.get(tid_j)
                if t_i is None or t_j is None:
                    continue

                # Inversion: j has higher priority but starts later
                if (
                    t_j.priority.numeric > t_i.priority.numeric
                    and start_j > start_i
                    and t_j.arrival_time_s <= start_i
                ):
                    delay = start_j - start_i
                    max_delay = max(end_j - start_i, 0.01)
                    severity = min(delay / max_delay, 1.0)
                    inversions.append(PriorityInversion(
                        high_priority_task_id=tid_j,
                        blocking_task_id=tid_i,
                        inversion_duration_s=delay,
                        severity=severity,
                    ))

        return inversions

    # ── dynamic priority adjustment ────────────────────────────────────

    def adjust_priorities(
        self,
        scores: Sequence[PriorityScore],
        context: Mapping[str, float],
    ) -> List[PriorityScore]:
        """Dynamically adjust priorities based on context.

        Context keys (all optional, values in [0, 1]):

        * ``cognitive_load`` — current cognitive load level.
        * ``stress`` — current stress level.
        * ``time_pressure`` — overall time pressure.

        Under high load, the model shifts priority toward easier tasks;
        under high time pressure, urgency weight increases.

        Parameters
        ----------
        scores : Sequence[PriorityScore]
        context : Mapping[str, float]

        Returns
        -------
        List[PriorityScore]
            Adjusted scores, re-sorted by composite.
        """
        load = context.get("cognitive_load", 0.0)
        stress = context.get("stress", 0.0)
        time_pressure = context.get("time_pressure", 0.0)

        adjusted: List[PriorityScore] = []
        for s in scores:
            # Under load, prefer easier tasks (boost effort score)
            effort_adj = s.effort * (1.0 + 0.5 * load)
            # Under time pressure, boost urgency
            urgency_adj = s.urgency * (1.0 + 0.8 * time_pressure)
            # Stress reduces importance differentiation
            importance_adj = s.importance * (1.0 - 0.3 * stress)

            weights = self._criteria.weights
            composite = float(
                weights[0] * min(urgency_adj, 1.0)
                + weights[1] * max(importance_adj, 0.0)
                + weights[2] * min(effort_adj, 1.0)
                + weights[3] * s.frequency
            )
            adjusted.append(PriorityScore(
                task_id=s.task_id,
                urgency=min(urgency_adj, 1.0),
                importance=max(importance_adj, 0.0),
                effort=min(effort_adj, 1.0),
                frequency=s.frequency,
                composite=composite,
            ))

        adjusted.sort(key=lambda s: s.composite, reverse=True)
        return adjusted


__all__ = [
    "PriorityComputer",
    "PriorityCriteria",
    "PriorityInversion",
    "PriorityScore",
]

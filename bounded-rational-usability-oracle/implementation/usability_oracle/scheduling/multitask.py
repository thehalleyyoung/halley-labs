"""
usability_oracle.scheduling.multitask — Multitasking model.

Models concurrent task execution, task-switching costs, and optimal
interleaving under bounded rationality.  Incorporates Salvucci & Taatgen's
threaded cognition framework and constructs task-interference graphs to
predict multitask performance degradation.

References
----------
* Salvucci, D. D. & Taatgen, N. A. (2008). Threaded cognition: An
  integrated theory of concurrent multitasking. *Psychological Review*,
  115(1), 101–130.
* Monsell, S. (2003). Task switching. *Trends in Cognitive Sciences*,
  7(3), 134–140.
* Wickens, C. D. (2002). Multiple resources and performance prediction.
  *Theoretical Issues in Ergonomics Science*, 3(2), 159–177.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.scheduling.types import (
    ScheduledTask,
    TaskPriority,
)


# ═══════════════════════════════════════════════════════════════════════════
# Resource types & task resource profiles
# ═══════════════════════════════════════════════════════════════════════════

@unique
class CognitiveResource(Enum):
    """Wickens' multiple-resource dimensions."""

    VISUAL = "visual"
    AUDITORY = "auditory"
    COGNITIVE = "cognitive"
    MOTOR = "motor"
    SPEECH = "speech"


@dataclass(frozen=True, slots=True)
class TaskResourceProfile:
    """Resource demands of a task across cognitive channels.

    Attributes
    ----------
    task_id : str
        Unique task identifier.
    demands : Mapping[CognitiveResource, float]
        Resource → demand in [0, 1].  0 = unused, 1 = fully occupied.
    working_memory_chunks : int
        Number of working-memory chunks required (Miller's 7 ± 2).
    skill_level : float
        Operator skill level in [0, 1].  1 = expert (lower switch cost).
    """

    task_id: str
    demands: Mapping[CognitiveResource, float] = field(default_factory=dict)
    working_memory_chunks: int = 3
    skill_level: float = 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Task-switching costs
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SwitchCost:
    """Cost of switching between two tasks.

    Attributes
    ----------
    from_task_id : str
    to_task_id : str
    context_switch_s : float
        Time to disengage from the first task (rule-set reconfiguration).
    wm_reload_s : float
        Time to reload working-memory state for the second task.
    resumption_lag_s : float
        Additional lag after an interruption before full performance.
    total_s : float
        Sum of all switching cost components.
    cognitive_cost_bits : float
        Information cost of the switch.
    """

    from_task_id: str
    to_task_id: str
    context_switch_s: float = 0.0
    wm_reload_s: float = 0.0
    resumption_lag_s: float = 0.0
    total_s: float = 0.0
    cognitive_cost_bits: float = 0.0


class SwitchCostModel:
    """Compute task-switching costs between pairs of tasks.

    The model estimates three components:

    1. **Context switch** — rule-set reconfiguration time, proportional to
       the dissimilarity of resource profiles.
    2. **Working-memory reload** — time to restore WM chunks, modulated
       by skill level.
    3. **Resumption lag** — additional delay after interruption, modelled
       as a log function of interruption duration.

    Parameters
    ----------
    base_switch_s : float
        Minimum context-switch time (even for similar tasks).
    wm_reload_per_chunk_s : float
        Time per working-memory chunk reloaded.
    resumption_coefficient : float
        Scaling factor for resumption lag.
    info_cost_per_switch_bit : float
        Information cost per switch in bits.
    """

    def __init__(
        self,
        base_switch_s: float = 0.2,
        wm_reload_per_chunk_s: float = 0.15,
        resumption_coefficient: float = 0.3,
        info_cost_per_switch_bit: float = 1.0,
    ) -> None:
        self._base = base_switch_s
        self._wm_rate = wm_reload_per_chunk_s
        self._resum_coeff = resumption_coefficient
        self._info_cost = info_cost_per_switch_bit

    def compute(
        self,
        from_profile: TaskResourceProfile,
        to_profile: TaskResourceProfile,
        interruption_duration_s: float = 0.0,
    ) -> SwitchCost:
        """Compute switching cost between two tasks.

        Parameters
        ----------
        from_profile : TaskResourceProfile
        to_profile : TaskResourceProfile
        interruption_duration_s : float
            How long the agent was interrupted (0 for voluntary switch).

        Returns
        -------
        SwitchCost
        """
        # Context switch: base + resource dissimilarity
        dissimilarity = self._resource_dissimilarity(from_profile, to_profile)
        ctx = self._base + 0.5 * dissimilarity

        # WM reload: chunks needed × rate, modulated by skill
        skill = to_profile.skill_level
        wm = (
            to_profile.working_memory_chunks
            * self._wm_rate
            * (1.0 - 0.5 * skill)
        )

        # Resumption lag: log of interruption duration
        if interruption_duration_s > 0:
            resum = self._resum_coeff * math.log(1.0 + interruption_duration_s)
        else:
            resum = 0.0

        total = ctx + wm + resum
        info_bits = self._info_cost * (1.0 + dissimilarity)

        return SwitchCost(
            from_task_id=from_profile.task_id,
            to_task_id=to_profile.task_id,
            context_switch_s=ctx,
            wm_reload_s=wm,
            resumption_lag_s=resum,
            total_s=total,
            cognitive_cost_bits=info_bits,
        )

    @staticmethod
    def _resource_dissimilarity(
        a: TaskResourceProfile,
        b: TaskResourceProfile,
    ) -> float:
        """Euclidean dissimilarity of resource demand vectors."""
        all_resources = set(a.demands.keys()) | set(b.demands.keys())
        if not all_resources:
            return 0.0
        sq_sum = 0.0
        for r in all_resources:
            da = a.demands.get(r, 0.0)
            db = b.demands.get(r, 0.0)
            sq_sum += (da - db) ** 2
        return math.sqrt(sq_sum / len(all_resources))

    def switch_cost_matrix(
        self,
        profiles: Sequence[TaskResourceProfile],
    ) -> np.ndarray:
        """Compute pairwise switch-cost matrix (total seconds).

        Parameters
        ----------
        profiles : Sequence[TaskResourceProfile]
            Resource profiles for each task.

        Returns
        -------
        np.ndarray
            Shape ``(n, n)`` matrix of total switch costs in seconds.
        """
        n = len(profiles)
        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i, j] = self.compute(profiles[i], profiles[j]).total_s
        return matrix


# ═══════════════════════════════════════════════════════════════════════════
# Threaded cognition model (Salvucci & Taatgen)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class CognitionThread:
    """A thread of execution in the threaded cognition model.

    Attributes
    ----------
    task_id : str
    resource_profile : TaskResourceProfile
    priority : float
        Thread priority (higher = more important).
    active : bool
        Whether the thread is currently active.
    progress : float
        Fraction of task completed in [0, 1].
    """

    task_id: str
    resource_profile: TaskResourceProfile
    priority: float = 1.0
    active: bool = True
    progress: float = 0.0


class ThreadedCognitionModel:
    """Salvucci & Taatgen threaded cognition model.

    Multiple task threads compete for shared cognitive resources.
    Conflicts are resolved greedily: the highest-priority thread
    claiming each resource wins; lower-priority threads stall on
    that resource until it is released.

    Parameters
    ----------
    resource_capacities : Mapping[CognitiveResource, float]
        Maximum capacity per resource (default 1.0 each).
    time_quantum_s : float
        Simulation time step in seconds.
    """

    def __init__(
        self,
        resource_capacities: Optional[Mapping[CognitiveResource, float]] = None,
        time_quantum_s: float = 0.05,
    ) -> None:
        self._capacities: Dict[CognitiveResource, float] = dict(
            resource_capacities or {r: 1.0 for r in CognitiveResource}
        )
        self._dt = time_quantum_s

    def simulate(
        self,
        threads: Sequence[CognitionThread],
        task_durations: Mapping[str, float],
        max_time_s: float = 60.0,
    ) -> Dict[str, float]:
        """Simulate threaded cognition and return completion times.

        Parameters
        ----------
        threads : Sequence[CognitionThread]
            Threads to simulate.
        task_durations : Mapping[str, float]
            Task id → nominal duration in seconds.
        max_time_s : float
            Maximum simulation time.

        Returns
        -------
        Dict[str, float]
            Task id → actual completion time.
        """
        # Mutable state for simulation
        progress: Dict[str, float] = {th.task_id: th.progress for th in threads}
        durations = {th.task_id: task_durations.get(th.task_id, 1.0) for th in threads}
        active = {th.task_id: th.active for th in threads}
        priorities = {th.task_id: th.priority for th in threads}
        profiles = {th.task_id: th.resource_profile for th in threads}
        completion_times: Dict[str, float] = {}

        t = 0.0
        while t < max_time_s:
            # Determine which threads are still running
            running = [
                tid for tid in progress
                if progress[tid] < 1.0 and active[tid]
            ]
            if not running:
                break

            # Resource conflict resolution: greedy by priority
            resource_claimed: Dict[CognitiveResource, str] = {}
            granted: Dict[str, bool] = {tid: True for tid in running}

            # Sort by priority descending
            running.sort(key=lambda tid: priorities.get(tid, 0.0), reverse=True)

            for tid in running:
                profile = profiles[tid]
                for res, demand in profile.demands.items():
                    if demand <= 0:
                        continue
                    if res in resource_claimed:
                        # Resource already claimed by higher-priority thread
                        granted[tid] = False
                        break

                if granted.get(tid, False):
                    for res, demand in profile.demands.items():
                        if demand > 0:
                            resource_claimed[res] = tid

            # Advance progress for granted threads
            for tid in running:
                if granted.get(tid, False):
                    dur = durations[tid]
                    if dur > 0:
                        progress[tid] += self._dt / dur
                        if progress[tid] >= 1.0:
                            progress[tid] = 1.0
                            completion_times[tid] = t + self._dt
                            active[tid] = False

            t += self._dt

        # Tasks that didn't complete
        for tid in progress:
            if tid not in completion_times:
                completion_times[tid] = max_time_s

        return completion_times


# ═══════════════════════════════════════════════════════════════════════════
# Task interference graph
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class InterferenceEdge:
    """An edge in the task-interference graph.

    Attributes
    ----------
    task_a : str
    task_b : str
    interference : float
        Interference strength in [0, 1].  0 = no interference;
        1 = complete mutual exclusion.
    shared_resources : Tuple[str, ...]
        Names of shared resources causing interference.
    """

    task_a: str
    task_b: str
    interference: float = 0.0
    shared_resources: Tuple[str, ...] = ()


def build_interference_graph(
    profiles: Sequence[TaskResourceProfile],
) -> List[InterferenceEdge]:
    """Construct a task-interference graph from resource profiles.

    Two tasks interfere if they demand the same cognitive resource.
    Interference strength is the dot product of shared demand vectors.

    Parameters
    ----------
    profiles : Sequence[TaskResourceProfile]

    Returns
    -------
    List[InterferenceEdge]
    """
    edges: List[InterferenceEdge] = []
    n = len(profiles)

    for i in range(n):
        for j in range(i + 1, n):
            shared = set(profiles[i].demands.keys()) & set(profiles[j].demands.keys())
            if not shared:
                continue
            interference = 0.0
            shared_names: List[str] = []
            for r in shared:
                d_i = profiles[i].demands[r]
                d_j = profiles[j].demands[r]
                if d_i > 0 and d_j > 0:
                    interference += d_i * d_j
                    shared_names.append(r.value)

            if interference > 0:
                interference = min(interference, 1.0)
                edges.append(InterferenceEdge(
                    task_a=profiles[i].task_id,
                    task_b=profiles[j].task_id,
                    interference=interference,
                    shared_resources=tuple(shared_names),
                ))

    return edges


# ═══════════════════════════════════════════════════════════════════════════
# Optimal interleaving under bounded rationality
# ═══════════════════════════════════════════════════════════════════════════

class MultitaskOptimizer:
    """Find optimal task interleaving under bounded rationality.

    Given a set of concurrent tasks, determines the interleaving
    sequence that minimises total completion time plus switching cost,
    subject to the bounded-rationality constraint that the user selects
    the next task via softmax over expected utilities.

    Parameters
    ----------
    switch_model : SwitchCostModel
        Model for computing pairwise switch costs.
    beta : float
        Bounded-rationality parameter.
    rng_seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        switch_model: Optional[SwitchCostModel] = None,
        beta: float = 1.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._switch = switch_model or SwitchCostModel()
        self._beta = beta
        self._rng = np.random.default_rng(rng_seed)

    def optimal_interleaving(
        self,
        profiles: Sequence[TaskResourceProfile],
        durations: Mapping[str, float],
    ) -> List[str]:
        """Compute a bounded-rational interleaving of tasks.

        At each step, the next task is sampled via softmax over the
        negative of (remaining duration + switch cost).

        Parameters
        ----------
        profiles : Sequence[TaskResourceProfile]
        durations : Mapping[str, float]
            Task id → duration in seconds.

        Returns
        -------
        List[str]
            Ordered list of task ids representing the interleaving.
        """
        if not profiles:
            return []

        profile_map = {p.task_id: p for p in profiles}
        remaining = {p.task_id: durations.get(p.task_id, 1.0) for p in profiles}
        order: List[str] = []
        current_id: Optional[str] = None

        while remaining:
            candidates = list(remaining.keys())
            if not candidates:
                break

            utilities = np.zeros(len(candidates), dtype=np.float64)
            for i, tid in enumerate(candidates):
                cost = remaining[tid]
                if current_id is not None and current_id != tid:
                    sc = self._switch.compute(
                        profile_map[current_id], profile_map[tid],
                    )
                    cost += sc.total_s
                utilities[i] = -cost  # negative cost = utility

            # Softmax selection
            v = utilities * self._beta
            v -= v.max()
            e = np.exp(v)
            probs = e / e.sum()

            idx = int(self._rng.choice(len(candidates), p=probs))
            chosen = candidates[idx]
            order.append(chosen)
            current_id = chosen
            del remaining[chosen]

        return order

    def predict_total_time(
        self,
        interleaving: Sequence[str],
        profiles: Sequence[TaskResourceProfile],
        durations: Mapping[str, float],
    ) -> float:
        """Predict total completion time for a given interleaving.

        Parameters
        ----------
        interleaving : Sequence[str]
            Task execution order.
        profiles : Sequence[TaskResourceProfile]
        durations : Mapping[str, float]

        Returns
        -------
        float
            Total time in seconds (task durations + switch costs).
        """
        profile_map = {p.task_id: p for p in profiles}
        total = 0.0

        for i, tid in enumerate(interleaving):
            total += durations.get(tid, 0.0)
            if i > 0:
                prev = interleaving[i - 1]
                if prev in profile_map and tid in profile_map:
                    sc = self._switch.compute(profile_map[prev], profile_map[tid])
                    total += sc.total_s
        return total


# ═══════════════════════════════════════════════════════════════════════════
# Interruption handling
# ═══════════════════════════════════════════════════════════════════════════

@unique
class InterruptionStrategy(Enum):
    """Strategy for handling task interruptions."""

    IMMEDIATE = "immediate"
    """Switch to interrupting task immediately."""

    DEFERRED = "deferred"
    """Finish current sub-task unit before switching."""

    NEGOTIATED = "negotiated"
    """Compare priorities; switch only if interrupting task dominates."""

    BLOCKED = "blocked"
    """Ignore interruptions entirely."""


@dataclass(frozen=True, slots=True)
class InterruptionResult:
    """Outcome of applying an interruption strategy.

    Attributes
    ----------
    switched : bool
        Whether the user switched to the interrupting task.
    delay_s : float
        Delay before the switch occurs (0 if immediate or blocked).
    resumption_cost_s : float
        Expected cost to resume the interrupted task.
    cognitive_overhead_bits : float
        Information cost of processing the interruption.
    """

    switched: bool = False
    delay_s: float = 0.0
    resumption_cost_s: float = 0.0
    cognitive_overhead_bits: float = 0.0


class InterruptionHandler:
    """Handle interruptions during multitasking.

    Models the decision of whether to switch to an interrupting task,
    incorporating task priorities, switch costs, and bounded rationality.

    Parameters
    ----------
    switch_model : SwitchCostModel
        For computing resumption costs.
    beta : float
        Bounded-rationality parameter.
    default_strategy : InterruptionStrategy
        Fallback strategy when none is specified.
    """

    def __init__(
        self,
        switch_model: Optional[SwitchCostModel] = None,
        beta: float = 1.0,
        default_strategy: InterruptionStrategy = InterruptionStrategy.NEGOTIATED,
    ) -> None:
        self._switch = switch_model or SwitchCostModel()
        self._beta = beta
        self._default_strategy = default_strategy

    def handle(
        self,
        current_profile: TaskResourceProfile,
        current_priority: float,
        interrupting_profile: TaskResourceProfile,
        interrupting_priority: float,
        interruption_duration_s: float = 0.0,
        strategy: Optional[InterruptionStrategy] = None,
    ) -> InterruptionResult:
        """Decide whether to switch to the interrupting task.

        Parameters
        ----------
        current_profile : TaskResourceProfile
        current_priority : float
        interrupting_profile : TaskResourceProfile
        interrupting_priority : float
        interruption_duration_s : float
        strategy : InterruptionStrategy | None

        Returns
        -------
        InterruptionResult
        """
        strat = strategy or self._default_strategy
        sc = self._switch.compute(
            current_profile, interrupting_profile, interruption_duration_s,
        )

        if strat == InterruptionStrategy.BLOCKED:
            return InterruptionResult(
                switched=False,
                cognitive_overhead_bits=0.5,  # minimal cost for ignoring
            )

        if strat == InterruptionStrategy.IMMEDIATE:
            return InterruptionResult(
                switched=True,
                delay_s=0.0,
                resumption_cost_s=sc.resumption_lag_s + sc.wm_reload_s,
                cognitive_overhead_bits=sc.cognitive_cost_bits,
            )

        if strat == InterruptionStrategy.DEFERRED:
            return InterruptionResult(
                switched=True,
                delay_s=sc.context_switch_s,
                resumption_cost_s=sc.resumption_lag_s,
                cognitive_overhead_bits=sc.cognitive_cost_bits * 0.8,
            )

        # NEGOTIATED: bounded-rational decision
        utility_switch = interrupting_priority - sc.total_s
        utility_stay = current_priority

        # Softmax choice
        v = np.array([utility_stay, utility_switch]) * self._beta
        v -= v.max()
        e = np.exp(v)
        p_switch = float(e[1] / e.sum())

        rng = np.random.default_rng()
        switched = bool(rng.random() < p_switch)

        if switched:
            return InterruptionResult(
                switched=True,
                delay_s=sc.context_switch_s,
                resumption_cost_s=sc.resumption_lag_s + sc.wm_reload_s,
                cognitive_overhead_bits=sc.cognitive_cost_bits,
            )
        return InterruptionResult(
            switched=False,
            delay_s=0.0,
            resumption_cost_s=0.0,
            cognitive_overhead_bits=0.5,
        )

    def multitask_performance_factor(
        self,
        profiles: Sequence[TaskResourceProfile],
    ) -> float:
        """Predict overall performance factor for concurrent tasks.

        Returns a factor in (0, 1] where 1.0 means no multitasking
        penalty and lower values indicate greater interference.

        Parameters
        ----------
        profiles : Sequence[TaskResourceProfile]

        Returns
        -------
        float
            Performance factor.
        """
        if len(profiles) <= 1:
            return 1.0

        edges = build_interference_graph(profiles)
        if not edges:
            return 1.0

        total_interference = sum(e.interference for e in edges)
        n_pairs = len(profiles) * (len(profiles) - 1) / 2
        avg_interference = total_interference / max(n_pairs, 1.0)

        return max(0.01, 1.0 - avg_interference)


__all__ = [
    "CognitionThread",
    "CognitiveResource",
    "InterferenceEdge",
    "InterruptionHandler",
    "InterruptionResult",
    "InterruptionStrategy",
    "MultitaskOptimizer",
    "SwitchCost",
    "SwitchCostModel",
    "TaskResourceProfile",
    "ThreadedCognitionModel",
    "build_interference_graph",
]

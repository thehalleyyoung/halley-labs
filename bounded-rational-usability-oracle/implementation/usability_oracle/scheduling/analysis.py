"""
usability_oracle.scheduling.analysis — Schedule analysis.

Provides utilisation analysis, bottleneck identification, what-if
simulation, regression detection, and schedulability tests for cognitive
task schedules under bounded rationality.

References
----------
* Lehoczky, J., Sha, L., & Ding, Y. (1989). The rate monotonic scheduling
  algorithm: Exact characterization and average case behavior. *RTSS*.
* Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a Theory of
  Decision-Making with Information-Processing Costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.scheduling.types import (
    DeadlineModel,
    Schedule,
    ScheduledTask,
    SchedulingConstraint,
)


# ═══════════════════════════════════════════════════════════════════════════
# Utilisation analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class UtilizationReport:
    """Per-resource utilisation report.

    Attributes
    ----------
    resource_name : str
    utilization : float
        Fraction of time the resource is busy, in [0, 1].
    busy_time_s : float
        Total time the resource is occupied.
    idle_time_s : float
        Total idle time within the schedule horizon.
    peak_demand : float
        Maximum concurrent demand at any moment.
    """

    resource_name: str
    utilization: float = 0.0
    busy_time_s: float = 0.0
    idle_time_s: float = 0.0
    peak_demand: float = 0.0


def utilization_analysis(
    schedule: Schedule,
    tasks: Sequence[ScheduledTask],
    resource_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[UtilizationReport]:
    """Analyse resource utilisation across cognitive channels.

    Parameters
    ----------
    schedule : Schedule
        The schedule to analyse.
    tasks : Sequence[ScheduledTask]
        Task definitions.
    resource_map : Mapping[str, Sequence[str]] | None
        Task id → list of resource names.  If None, all tasks share
        a single ``"cognitive"`` resource.

    Returns
    -------
    List[UtilizationReport]
        One report per resource.
    """
    if not schedule.assignments:
        return []

    starts = [s for _, s, _ in schedule.assignments]
    ends = [e for _, _, e in schedule.assignments]
    horizon = max(ends) - min(starts)
    if horizon <= 0:
        return []

    # Collect all resource names
    if resource_map is None:
        resource_map = {tid: ["cognitive"] for tid, _, _ in schedule.assignments}

    all_resources: set[str] = set()
    for resources in resource_map.values():
        all_resources.update(resources)
    if not all_resources:
        all_resources.add("cognitive")

    # Compute busy time per resource
    busy: Dict[str, float] = {r: 0.0 for r in all_resources}
    for tid, start, end in schedule.assignments:
        dur = end - start
        resources = resource_map.get(tid, ["cognitive"])
        for r in resources:
            if r in busy:
                busy[r] += dur

    # Peak demand via sweep line
    peak: Dict[str, float] = {r: 0.0 for r in all_resources}
    events: Dict[str, List[Tuple[float, int]]] = {r: [] for r in all_resources}
    for tid, start, end in schedule.assignments:
        resources = resource_map.get(tid, ["cognitive"])
        for r in resources:
            events[r].append((start, 1))
            events[r].append((end, -1))

    for r in all_resources:
        evts = sorted(events[r])
        current = 0.0
        for _, delta in evts:
            current += delta
            peak[r] = max(peak[r], current)

    reports = []
    for r in sorted(all_resources):
        util = min(busy[r] / horizon, 1.0)
        reports.append(UtilizationReport(
            resource_name=r,
            utilization=util,
            busy_time_s=busy[r],
            idle_time_s=max(horizon - busy[r], 0.0),
            peak_demand=peak[r],
        ))

    return reports


# ═══════════════════════════════════════════════════════════════════════════
# Bottleneck identification
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Bottleneck:
    """An identified scheduling bottleneck.

    Attributes
    ----------
    resource_name : str
        Resource causing the bottleneck.
    severity : float
        Severity in [0, 1], where 1 = fully saturated.
    time_window : Tuple[float, float]
        ``(start, end)`` of the congested interval.
    contributing_tasks : Tuple[str, ...]
        Task ids contributing to the bottleneck.
    """

    resource_name: str
    severity: float = 0.0
    time_window: Tuple[float, float] = (0.0, 0.0)
    contributing_tasks: Tuple[str, ...] = ()


def identify_bottlenecks(
    schedule: Schedule,
    resource_map: Optional[Mapping[str, Sequence[str]]] = None,
    threshold: float = 0.8,
    resolution_s: float = 0.1,
) -> List[Bottleneck]:
    """Identify time windows where a resource exceeds *threshold* utilisation.

    Parameters
    ----------
    schedule : Schedule
    resource_map : Mapping[str, Sequence[str]] | None
    threshold : float
        Utilisation threshold for flagging a bottleneck.
    resolution_s : float
        Time resolution for the sweep.

    Returns
    -------
    List[Bottleneck]
    """
    if not schedule.assignments:
        return []

    if resource_map is None:
        resource_map = {tid: ["cognitive"] for tid, _, _ in schedule.assignments}

    starts = [s for _, s, _ in schedule.assignments]
    ends = [e for _, _, e in schedule.assignments]
    t_min, t_max = min(starts), max(ends)
    if t_max <= t_min:
        return []

    all_resources: set[str] = set()
    for resources in resource_map.values():
        all_resources.update(resources)

    time_points = np.arange(t_min, t_max, resolution_s)
    bottlenecks: List[Bottleneck] = []

    for resource in sorted(all_resources):
        load = np.zeros_like(time_points)
        task_at_time: Dict[int, List[str]] = {i: [] for i in range(len(time_points))}

        for tid, start, end in schedule.assignments:
            if resource not in resource_map.get(tid, []):
                continue
            mask = (time_points >= start) & (time_points < end)
            load += mask.astype(np.float64)
            for idx in np.where(mask)[0]:
                task_at_time[int(idx)].append(tid)

        # Find contiguous regions above threshold
        above = load >= threshold
        in_region = False
        region_start = 0

        for i in range(len(above)):
            if above[i] and not in_region:
                in_region = True
                region_start = i
            elif not above[i] and in_region:
                in_region = False
                contributing = set()
                for j in range(region_start, i):
                    contributing.update(task_at_time[j])
                severity = float(load[region_start:i].mean())
                bottlenecks.append(Bottleneck(
                    resource_name=resource,
                    severity=min(severity, 1.0),
                    time_window=(float(time_points[region_start]), float(time_points[i])),
                    contributing_tasks=tuple(sorted(contributing)),
                ))

        if in_region:
            contributing = set()
            for j in range(region_start, len(time_points)):
                contributing.update(task_at_time[j])
            severity = float(load[region_start:].mean())
            bottlenecks.append(Bottleneck(
                resource_name=resource,
                severity=min(severity, 1.0),
                time_window=(float(time_points[region_start]), float(time_points[-1])),
                contributing_tasks=tuple(sorted(contributing)),
            ))

    return bottlenecks


# ═══════════════════════════════════════════════════════════════════════════
# What-if analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class WhatIfResult:
    """Result of a what-if schedule modification.

    Attributes
    ----------
    description : str
    original_makespan_s : float
    modified_makespan_s : float
    makespan_delta_s : float
        Positive = worse; negative = improved.
    original_violations : int
    modified_violations : int
    """

    description: str
    original_makespan_s: float = 0.0
    modified_makespan_s: float = 0.0
    makespan_delta_s: float = 0.0
    original_violations: int = 0
    modified_violations: int = 0


def what_if_remove_task(
    schedule: Schedule,
    task_id: str,
) -> WhatIfResult:
    """Compute what-if analysis for removing a task.

    Parameters
    ----------
    schedule : Schedule
    task_id : str

    Returns
    -------
    WhatIfResult
    """
    filtered = [a for a in schedule.assignments if a[0] != task_id]
    if not filtered:
        return WhatIfResult(
            description=f"Remove task {task_id}",
            original_makespan_s=schedule.makespan_s,
            modified_makespan_s=0.0,
            makespan_delta_s=-schedule.makespan_s,
            original_violations=len(schedule.deadline_violations),
            modified_violations=0,
        )

    starts = [s for _, s, _ in filtered]
    ends = [e for _, _, e in filtered]
    new_makespan = max(ends) - min(starts)
    new_violations = len([v for v in schedule.deadline_violations if v != task_id])

    return WhatIfResult(
        description=f"Remove task {task_id}",
        original_makespan_s=schedule.makespan_s,
        modified_makespan_s=new_makespan,
        makespan_delta_s=new_makespan - schedule.makespan_s,
        original_violations=len(schedule.deadline_violations),
        modified_violations=new_violations,
    )


def what_if_change_duration(
    schedule: Schedule,
    task_id: str,
    new_duration_s: float,
) -> WhatIfResult:
    """What-if analysis for changing a task's duration.

    Subsequent tasks are shifted by the duration delta.

    Parameters
    ----------
    schedule : Schedule
    task_id : str
    new_duration_s : float

    Returns
    -------
    WhatIfResult
    """
    modified: List[Tuple[str, float, float]] = []
    shift = 0.0

    for tid, start, end in schedule.assignments:
        if tid == task_id:
            old_dur = end - start
            shift = new_duration_s - old_dur
            modified.append((tid, start, start + new_duration_s))
        else:
            modified.append((tid, start + max(shift, 0.0), end + max(shift, 0.0)))

    if not modified:
        return WhatIfResult(description=f"Change duration of {task_id}")

    starts = [s for _, s, _ in modified]
    ends = [e for _, _, e in modified]
    new_makespan = max(ends) - min(starts)

    return WhatIfResult(
        description=f"Change duration of {task_id} to {new_duration_s:.2f}s",
        original_makespan_s=schedule.makespan_s,
        modified_makespan_s=new_makespan,
        makespan_delta_s=new_makespan - schedule.makespan_s,
        original_violations=len(schedule.deadline_violations),
        modified_violations=len(schedule.deadline_violations),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Schedule comparison for regression detection
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ScheduleComparison:
    """Comparison between two schedules for regression detection.

    Attributes
    ----------
    makespan_delta_s : float
        Change in makespan (positive = regression).
    cognitive_cost_delta_bits : float
        Change in total cognitive cost (positive = regression).
    violation_delta : int
        Change in number of deadline violations.
    is_regression : bool
        True if the new schedule is worse on any metric.
    details : Mapping[str, Any]
        Per-task comparison details.
    """

    makespan_delta_s: float = 0.0
    cognitive_cost_delta_bits: float = 0.0
    violation_delta: int = 0
    is_regression: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)


def compare_schedules(
    baseline: Schedule,
    candidate: Schedule,
    tolerance: float = 0.01,
) -> ScheduleComparison:
    """Compare two schedules for regression detection.

    Parameters
    ----------
    baseline : Schedule
        Reference (old) schedule.
    candidate : Schedule
        New schedule to evaluate.
    tolerance : float
        Threshold below which deltas are not considered regressions.

    Returns
    -------
    ScheduleComparison
    """
    ms_delta = candidate.makespan_s - baseline.makespan_s
    cc_delta = candidate.total_cognitive_cost_bits - baseline.total_cognitive_cost_bits
    v_delta = len(candidate.deadline_violations) - len(baseline.deadline_violations)

    is_regression = (
        ms_delta > tolerance
        or cc_delta > tolerance
        or v_delta > 0
    )

    # Per-task start-time comparison
    baseline_starts = {tid: s for tid, s, _ in baseline.assignments}
    candidate_starts = {tid: s for tid, s, _ in candidate.assignments}
    per_task: Dict[str, float] = {}
    for tid in baseline_starts:
        if tid in candidate_starts:
            per_task[tid] = candidate_starts[tid] - baseline_starts[tid]

    return ScheduleComparison(
        makespan_delta_s=ms_delta,
        cognitive_cost_delta_bits=cc_delta,
        violation_delta=v_delta,
        is_regression=is_regression,
        details={"per_task_start_delta_s": per_task},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Response-time distribution estimation
# ═══════════════════════════════════════════════════════════════════════════

def response_time_distribution(
    task: ScheduledTask,
    beta: float = 1.0,
    n_samples: int = 10000,
    rng_seed: Optional[int] = None,
) -> np.ndarray:
    """Estimate the response-time distribution for a task.

    Models response time as log-normal with variance modulated by the
    rationality parameter β.

    Parameters
    ----------
    task : ScheduledTask
    beta : float
        Rationality parameter.
    n_samples : int
        Number of Monte-Carlo samples.
    rng_seed : int | None

    Returns
    -------
    np.ndarray
        Shape ``(n_samples,)`` of response-time samples in seconds.
    """
    rng = np.random.default_rng(rng_seed)
    mu = task.estimated_duration_s
    if mu <= 0:
        return np.zeros(n_samples, dtype=np.float64)

    # CV increases as β decreases
    cv = 0.3 / max(beta, 0.01)
    sigma_sq = math.log(1.0 + cv ** 2)
    sigma = math.sqrt(sigma_sq)
    mu_ln = math.log(mu) - sigma_sq / 2.0

    return rng.lognormal(mu_ln, sigma, size=n_samples)


# ═══════════════════════════════════════════════════════════════════════════
# Schedulability test
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SchedulabilityResult:
    """Result of a schedulability test.

    Attributes
    ----------
    schedulable : bool
    utilization : float
        Total utilisation.
    utilization_bound : float
        Upper bound for the scheduling algorithm.
    margin : float
        How far below the bound (positive = feasible).
    details : str
        Explanation.
    """

    schedulable: bool = False
    utilization: float = 0.0
    utilization_bound: float = 1.0
    margin: float = 0.0
    details: str = ""


def schedulability_test(
    tasks: Sequence[ScheduledTask],
    algorithm: str = "edf",
    beta: float = 1.0,
) -> SchedulabilityResult:
    """Test whether tasks are schedulable under bounded rationality.

    For EDF, the utilisation bound is 1.0 (exact for independent tasks).
    For RMS, the Liu–Layland bound is n(2^{1/n} − 1).  Under bounded
    rationality, the effective bound is reduced by a factor that
    accounts for sub-optimal scheduling decisions.

    Parameters
    ----------
    tasks : Sequence[ScheduledTask]
    algorithm : str
        ``"edf"`` or ``"rms"``.
    beta : float
        Rationality parameter.

    Returns
    -------
    SchedulabilityResult
    """
    dl_tasks = [
        t for t in tasks
        if t.deadline is not None and t.deadline.hard_deadline_s is not None
        and t.deadline.hard_deadline_s > 0
    ]
    if not dl_tasks:
        return SchedulabilityResult(
            schedulable=True,
            details="No deadline-constrained tasks.",
        )

    n = len(dl_tasks)
    utilization = sum(
        t.estimated_duration_s / t.deadline.hard_deadline_s  # type: ignore[operator]
        for t in dl_tasks
    )

    if algorithm == "rms":
        bound = n * (2.0 ** (1.0 / n) - 1.0)
    else:
        bound = 1.0

    # Bounded-rationality reduction: sub-optimal decisions waste capacity
    # Effective bound = bound × (1 − 1 / (1 + β))
    br_factor = 1.0 - 1.0 / (1.0 + beta)
    effective_bound = bound * max(br_factor, 0.1)

    margin = effective_bound - utilization

    return SchedulabilityResult(
        schedulable=margin >= 0,
        utilization=utilization,
        utilization_bound=effective_bound,
        margin=margin,
        details=(
            f"U={utilization:.3f}, bound={effective_bound:.3f} "
            f"(base={bound:.3f}, β-factor={br_factor:.3f})"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """Sensitivity of schedule feasibility to parameter variations.

    Attributes
    ----------
    parameter_name : str
    values : Tuple[float, ...]
        Parameter values tested.
    schedulable : Tuple[bool, ...]
        Whether each value yields a feasible schedule.
    utilizations : Tuple[float, ...]
        Utilisation at each value.
    critical_value : Optional[float]
        Parameter value at which feasibility changes (if found).
    """

    parameter_name: str
    values: Tuple[float, ...] = ()
    schedulable: Tuple[bool, ...] = ()
    utilizations: Tuple[float, ...] = ()
    critical_value: Optional[float] = None


def sensitivity_to_beta(
    tasks: Sequence[ScheduledTask],
    beta_values: Optional[Sequence[float]] = None,
    algorithm: str = "edf",
) -> SensitivityResult:
    """Analyse how schedulability changes with β.

    Parameters
    ----------
    tasks : Sequence[ScheduledTask]
    beta_values : Sequence[float] | None
        β values to test.  Defaults to a logarithmic range.
    algorithm : str

    Returns
    -------
    SensitivityResult
    """
    if beta_values is None:
        beta_values = list(np.logspace(-2, 2, 20))

    results_sched: List[bool] = []
    results_util: List[float] = []
    critical: Optional[float] = None
    prev_feasible: Optional[bool] = None

    for b in beta_values:
        r = schedulability_test(tasks, algorithm=algorithm, beta=b)
        results_sched.append(r.schedulable)
        results_util.append(r.utilization)
        if prev_feasible is not None and r.schedulable != prev_feasible and critical is None:
            critical = b
        prev_feasible = r.schedulable

    return SensitivityResult(
        parameter_name="beta",
        values=tuple(beta_values),
        schedulable=tuple(results_sched),
        utilizations=tuple(results_util),
        critical_value=critical,
    )


def sensitivity_to_duration_scaling(
    tasks: Sequence[ScheduledTask],
    scale_factors: Optional[Sequence[float]] = None,
    beta: float = 1.0,
    algorithm: str = "edf",
) -> SensitivityResult:
    """Analyse how schedulability changes when task durations are scaled.

    Parameters
    ----------
    tasks : Sequence[ScheduledTask]
    scale_factors : Sequence[float] | None
    beta : float
    algorithm : str

    Returns
    -------
    SensitivityResult
    """
    if scale_factors is None:
        scale_factors = list(np.linspace(0.5, 2.0, 20))

    results_sched: List[bool] = []
    results_util: List[float] = []
    critical: Optional[float] = None
    prev_feasible: Optional[bool] = None

    for s in scale_factors:
        from dataclasses import replace
        scaled = [
            replace(t, estimated_duration_s=t.estimated_duration_s * s)
            for t in tasks
        ]
        r = schedulability_test(scaled, algorithm=algorithm, beta=beta)
        results_sched.append(r.schedulable)
        results_util.append(r.utilization)
        if prev_feasible is not None and r.schedulable != prev_feasible and critical is None:
            critical = s
        prev_feasible = r.schedulable

    return SensitivityResult(
        parameter_name="duration_scale",
        values=tuple(scale_factors),
        schedulable=tuple(results_sched),
        utilizations=tuple(results_util),
        critical_value=critical,
    )


__all__ = [
    "Bottleneck",
    "SchedulabilityResult",
    "ScheduleComparison",
    "SensitivityResult",
    "UtilizationReport",
    "WhatIfResult",
    "compare_schedules",
    "identify_bottlenecks",
    "response_time_distribution",
    "schedulability_test",
    "sensitivity_to_beta",
    "sensitivity_to_duration_scaling",
    "utilization_analysis",
    "what_if_change_duration",
    "what_if_remove_task",
]

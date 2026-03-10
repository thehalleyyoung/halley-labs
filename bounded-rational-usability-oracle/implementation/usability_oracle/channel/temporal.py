"""
usability_oracle.channel.temporal — Temporal channel dynamics.

Models time-varying aspects of cognitive channel capacity:

* **Time-varying capacity**: vigilance decrement, warm-up
* **Attention switching costs**: cost of reallocating attention
* **Channel recovery**: capacity restoration after overload
* **Psychological Refractory Period (PRP)**: response bottleneck delay
* **Threaded cognition**: EPIC/threaded-cognition scheduling model
* **Timeline**: channel usage over time

References
----------
- Pashler, H. (1994). Dual-task interference: PRP. Psych. Bulletin, 116(2).
- Salvucci, D. D. & Taatgen, N. A. (2008). Threaded cognition. Psych. Rev.
- Warm, J. S. et al. (2008). Vigilance requires hard mental work. HF, 50(3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.channel.types import (
    ChannelAllocation,
    ResourceChannel,
    ResourcePool,
    WickensResource,
)


# ═══════════════════════════════════════════════════════════════════════════
# Time-varying channel capacity
# ═══════════════════════════════════════════════════════════════════════════

def time_varying_capacity(
    base_capacity: float,
    time_s: float,
    warmup_tau_s: float = 30.0,
    vigilance_half_life_s: float = 2700.0,
    floor_fraction: float = 0.25,
) -> float:
    """Compute channel capacity at a given time point.

    Models two phases:
    1. **Warm-up** (exponential rise to full capacity).
    2. **Vigilance decrement** (exponential decay over sustained use).

    Parameters
    ----------
    base_capacity : float
        Peak capacity in bits/s.
    time_s : float
        Seconds since task start.
    warmup_tau_s : float
        Warm-up time constant (default 30 s).
    vigilance_half_life_s : float
        Vigilance half-life (default 45 min = 2700 s).
    floor_fraction : float
        Minimum capacity as fraction of base.

    Returns
    -------
    float
        Capacity at time ``time_s`` in bits/s.
    """
    # Warm-up factor: rises from ~0 to 1.
    warmup = 1.0 - math.exp(-time_s / warmup_tau_s)
    # Vigilance decay factor.
    decay = math.exp(-math.log(2.0) * time_s / vigilance_half_life_s)
    cap = base_capacity * warmup * decay
    return max(cap, base_capacity * floor_fraction)


def capacity_timeline(
    base_capacity: float,
    duration_s: float,
    dt_s: float = 1.0,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a capacity-over-time profile.

    Parameters
    ----------
    base_capacity : float
        Peak capacity.
    duration_s : float
        Total duration in seconds.
    dt_s : float
        Time step.

    Returns
    -------
    (times, capacities) : (np.ndarray, np.ndarray)
    """
    times = np.arange(0.0, duration_s, dt_s)
    caps = np.array([
        time_varying_capacity(base_capacity, t, **kwargs) for t in times
    ])
    return times, caps


# ═══════════════════════════════════════════════════════════════════════════
# Attention switching costs
# ═══════════════════════════════════════════════════════════════════════════

# Published switching costs in seconds (Monsell, 2003; Rubinstein et al., 2001).
_SWITCH_COST_BASE: Dict[str, float] = {
    "same_modality":       0.15,   # within-modality switch
    "cross_modality":      0.30,   # visual ↔ auditory
    "task_set":            0.50,   # task-set reconfiguration
    "response_remapping":  0.40,   # change response mapping
}


def attention_switch_cost(
    from_resource: WickensResource,
    to_resource: WickensResource,
    task_set_change: bool = False,
) -> float:
    """Compute the cost (in seconds) of switching attention.

    Parameters
    ----------
    from_resource : WickensResource
        Resource being deactivated.
    to_resource : WickensResource
        Resource being activated.
    task_set_change : bool
        Whether the switch involves a new task set.

    Returns
    -------
    float
        Switch cost in seconds.
    """
    if from_resource == to_resource:
        return 0.0

    cost = 0.0
    dim_from = from_resource.dimension
    dim_to = to_resource.dimension

    if dim_from == dim_to:
        # Same dimension → within-modality switch.
        cost = _SWITCH_COST_BASE["same_modality"]
    else:
        # Cross-dimension switch.
        cost = _SWITCH_COST_BASE["cross_modality"]

    if task_set_change:
        cost += _SWITCH_COST_BASE["task_set"]

    return cost


def total_switching_cost(
    switch_sequence: Sequence[WickensResource],
) -> float:
    """Total switching cost for a sequence of attention shifts.

    Parameters
    ----------
    switch_sequence : Sequence[WickensResource]
        Ordered sequence of resources attended to.

    Returns
    -------
    float
        Total switch cost in seconds.
    """
    if len(switch_sequence) < 2:
        return 0.0
    total = 0.0
    for i in range(len(switch_sequence) - 1):
        total += attention_switch_cost(switch_sequence[i], switch_sequence[i + 1])
    return total


# ═══════════════════════════════════════════════════════════════════════════
# Channel recovery after overload
# ═══════════════════════════════════════════════════════════════════════════

def channel_recovery(
    overload_duration_s: float,
    overload_severity: float,
    recovery_time_s: float,
    base_capacity: float,
    recovery_tau_s: float = 60.0,
) -> float:
    """Compute capacity during recovery from overload.

    After a period of overload, capacity recovers exponentially.
    The depth of the dip depends on overload duration and severity.

    Parameters
    ----------
    overload_duration_s : float
        How long the channel was overloaded (seconds).
    overload_severity : float
        Severity of overload in [0, 1] (1 = complete saturation).
    recovery_time_s : float
        Time since overload ended (seconds).
    base_capacity : float
        Normal capacity in bits/s.
    recovery_tau_s : float
        Recovery time constant (default 60 s).

    Returns
    -------
    float
        Current capacity during recovery.
    """
    overload_severity = max(0.0, min(1.0, overload_severity))
    # Capacity dip proportional to overload.
    dip = overload_severity * min(overload_duration_s / 60.0, 1.0)
    dip = min(dip, 0.8)  # cap at 80 % reduction
    # Exponential recovery.
    recovery = 1.0 - dip * math.exp(-recovery_time_s / recovery_tau_s)
    return base_capacity * max(recovery, 0.10)


# ═══════════════════════════════════════════════════════════════════════════
# Psychological Refractory Period (PRP) model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PRPResult:
    """Result of PRP (Psychological Refractory Period) analysis.

    Attributes
    ----------
    stimulus_onset_asynchrony_s : float
        SOA between stimulus 1 and stimulus 2.
    rt1_s : float
        Reaction time for task 1.
    rt2_s : float
        Reaction time for task 2 (includes PRP delay).
    prp_delay_s : float
        Additional delay on task 2 due to central bottleneck.
    bottleneck_stage : str
        Stage causing the bottleneck (typically "cognitive").
    """

    stimulus_onset_asynchrony_s: float = 0.0
    rt1_s: float = 0.0
    rt2_s: float = 0.0
    prp_delay_s: float = 0.0
    bottleneck_stage: str = "cognitive"


def prp_model(
    rt1_base_s: float = 0.4,
    rt2_base_s: float = 0.5,
    soa_s: float = 0.1,
    stage1_perception_s: float = 0.10,
    stage1_central_s: float = 0.20,
    stage1_response_s: float = 0.10,
    stage2_perception_s: float = 0.12,
    stage2_central_s: float = 0.22,
    stage2_response_s: float = 0.16,
) -> PRPResult:
    """Compute PRP effect using the central bottleneck model.

    The model assumes that central (cognitive) processing is serial:
    task 2's central stage cannot begin until task 1's central stage
    completes (Pashler, 1994).

    Parameters
    ----------
    rt1_base_s : float
        Task 1 baseline RT.
    rt2_base_s : float
        Task 2 baseline RT.
    soa_s : float
        Stimulus onset asynchrony (delay between S1 and S2).
    stage1_perception_s, stage1_central_s, stage1_response_s : float
        Task 1 stage durations.
    stage2_perception_s, stage2_central_s, stage2_response_s : float
        Task 2 stage durations.

    Returns
    -------
    PRPResult
    """
    # Task 1 timeline.
    t1_perc_end = stage1_perception_s
    t1_central_end = t1_perc_end + stage1_central_s
    t1_end = t1_central_end + stage1_response_s

    # Task 2 perception starts at SOA.
    t2_perc_end = soa_s + stage2_perception_s

    # Task 2 central cannot start until both:
    #   (a) task 2 perception is done, and
    #   (b) task 1 central is done (bottleneck).
    t2_central_start = max(t2_perc_end, t1_central_end)
    t2_central_end = t2_central_start + stage2_central_s
    t2_end = t2_central_end + stage2_response_s

    rt2 = t2_end - soa_s  # RT2 measured from S2 onset.
    prp_delay = max(0.0, rt2 - rt2_base_s)

    return PRPResult(
        stimulus_onset_asynchrony_s=soa_s,
        rt1_s=t1_end,
        rt2_s=rt2,
        prp_delay_s=prp_delay,
        bottleneck_stage="cognitive",
    )


def prp_curve(
    soa_range: Sequence[float],
    **kwargs,
) -> List[PRPResult]:
    """Compute PRP curve across a range of SOAs.

    Parameters
    ----------
    soa_range : Sequence[float]
        SOA values in seconds.

    Returns
    -------
    List[PRPResult]
        PRP result at each SOA.
    """
    return [prp_model(soa_s=soa, **kwargs) for soa in soa_range]


# ═══════════════════════════════════════════════════════════════════════════
# Temporal overlap cost
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class TaskInterval:
    """A task occupying a time interval on a specific channel.

    Attributes
    ----------
    task_id : str
    resource : WickensResource
    start_s : float
    end_s : float
    demand : float
        Demand level [0, 1] during this interval.
    """

    task_id: str = ""
    resource: WickensResource = WickensResource.COGNITIVE
    start_s: float = 0.0
    end_s: float = 0.0
    demand: float = 0.5

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def temporal_overlap_cost(
    intervals: Sequence[TaskInterval],
) -> float:
    """Compute total temporal overlap cost across channel intervals.

    For each pair of overlapping intervals on the same resource,
    the overlap cost is proportional to the overlap duration and
    the product of demands.

    Parameters
    ----------
    intervals : Sequence[TaskInterval]

    Returns
    -------
    float
        Total overlap cost in demand·seconds.
    """
    n = len(intervals)
    total_cost = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            a = intervals[i]
            b = intervals[j]
            if a.resource != b.resource:
                continue
            # Temporal overlap.
            overlap_start = max(a.start_s, b.start_s)
            overlap_end = min(a.end_s, b.end_s)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > 0:
                total_cost += overlap * a.demand * b.demand
    return total_cost


# ═══════════════════════════════════════════════════════════════════════════
# Threaded cognition model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CognitiveThread:
    """A cognitive thread in the threaded-cognition framework.

    Each thread represents an active goal/task that competes for
    resource access via a greedy scheduling policy.

    Attributes
    ----------
    thread_id : str
    required_resources : List[WickensResource]
        Resources needed (in order of processing stages).
    stage_durations_s : List[float]
        Duration of each processing stage.
    priority : float
        Thread priority (higher = more urgent).
    """

    thread_id: str = ""
    required_resources: List[WickensResource] = field(default_factory=list)
    stage_durations_s: List[float] = field(default_factory=list)
    priority: float = 1.0


def threaded_cognition_schedule(
    threads: Sequence[CognitiveThread],
    resource_capacities: Optional[Mapping[WickensResource, int]] = None,
) -> Dict[str, List[Tuple[float, float, WickensResource]]]:
    """Schedule cognitive threads using greedy resource arbitration.

    Implements a simplified Salvucci & Taatgen (2008) model:
    each thread proceeds through its stages in order, acquiring
    resources greedily.  A resource can serve one thread at a time.

    Parameters
    ----------
    threads : Sequence[CognitiveThread]
        Threads to schedule.
    resource_capacities : Mapping[WickensResource, int] or None
        Number of concurrent users per resource (default 1).

    Returns
    -------
    Dict[str, List[Tuple[float, float, WickensResource]]]
        thread_id → list of (start_s, end_s, resource) triples.
    """
    caps: Dict[WickensResource, int] = dict(resource_capacities) if resource_capacities else {}

    # State per thread: current stage index, earliest ready time.
    n = len(threads)
    stage_idx = [0] * n
    ready_time = [0.0] * n
    schedules: Dict[str, List[Tuple[float, float, WickensResource]]] = {
        t.thread_id: [] for t in threads
    }

    # Resource availability: next free time.
    res_free: Dict[WickensResource, List[float]] = {}
    for t in threads:
        for r in t.required_resources:
            cap = caps.get(r, 1)
            if r not in res_free:
                res_free[r] = [0.0] * cap

    max_iterations = sum(len(t.required_resources) for t in threads) + 1
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        # Find the thread with earliest ready time that still has stages.
        best_t = -1
        best_time = float("inf")
        for i in range(n):
            if stage_idx[i] < len(threads[i].required_resources):
                if ready_time[i] < best_time or (
                    ready_time[i] == best_time
                    and threads[i].priority > (
                        threads[best_t].priority if best_t >= 0 else -1
                    )
                ):
                    best_time = ready_time[i]
                    best_t = i

        if best_t < 0:
            break  # all done

        t = threads[best_t]
        si = stage_idx[best_t]
        resource = t.required_resources[si]
        duration = t.stage_durations_s[si] if si < len(t.stage_durations_s) else 0.1

        # Find earliest slot on this resource.
        slots = res_free.get(resource, [0.0])
        slot_idx = int(np.argmin(slots))
        start = max(ready_time[best_t], slots[slot_idx])
        end = start + duration

        schedules[t.thread_id].append((start, end, resource))
        res_free[resource][slot_idx] = end
        ready_time[best_t] = end
        stage_idx[best_t] = si + 1

    return schedules


# ═══════════════════════════════════════════════════════════════════════════
# Timeline of channel usage
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ChannelUsageEvent:
    """A single channel usage event in the timeline.

    Attributes
    ----------
    task_id : str
    resource : WickensResource
    start_s : float
    end_s : float
    load : float
        Load during this event [0, 1].
    """

    task_id: str = ""
    resource: WickensResource = WickensResource.COGNITIVE
    start_s: float = 0.0
    end_s: float = 0.0
    load: float = 0.5


def build_channel_timeline(
    events: Sequence[ChannelUsageEvent],
    dt_s: float = 0.1,
) -> Dict[WickensResource, Tuple[np.ndarray, np.ndarray]]:
    """Build per-channel load timelines from usage events.

    Parameters
    ----------
    events : Sequence[ChannelUsageEvent]
    dt_s : float
        Time resolution.

    Returns
    -------
    Dict[WickensResource, (times, loads)]
        Per-channel time series of aggregate load.
    """
    if not events:
        return {}

    t_min = min(e.start_s for e in events)
    t_max = max(e.end_s for e in events)
    times = np.arange(t_min, t_max + dt_s, dt_s)
    n = len(times)

    # Collect resources.
    resources = sorted(set(e.resource for e in events), key=lambda r: r.value)

    result: Dict[WickensResource, Tuple[np.ndarray, np.ndarray]] = {}
    for r in resources:
        loads = np.zeros(n, dtype=np.float64)
        for e in events:
            if e.resource != r:
                continue
            mask = (times >= e.start_s) & (times < e.end_s)
            loads[mask] += e.load
        result[r] = (times, loads)

    return result


__all__ = [
    "ChannelUsageEvent",
    "CognitiveThread",
    "PRPResult",
    "TaskInterval",
    "attention_switch_cost",
    "build_channel_timeline",
    "capacity_timeline",
    "channel_recovery",
    "prp_curve",
    "prp_model",
    "temporal_overlap_cost",
    "threaded_cognition_schedule",
    "time_varying_capacity",
    "total_switching_cost",
]

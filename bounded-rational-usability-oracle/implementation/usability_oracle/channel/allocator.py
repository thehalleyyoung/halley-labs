"""
usability_oracle.channel.allocator — Resource allocation algorithm.

Implements optimal allocation of cognitive resources across MRT channels:

* **Water-filling algorithm** for capacity-constrained allocation
* **Priority-based allocation** with preemption
* **Dynamic reallocation** as task demands change
* **Robust allocation** under demand uncertainty

The allocator distributes a finite cognitive-resource budget across
channels so as to maximise effective throughput (or minimise total cost).

References
----------
- Cover, T. & Thomas, J. (2006). Elements of Information Theory, Ch. 10.
- Wickens, C. D. (2002). Multiple resources and performance prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.channel.types import (
    ChannelAllocation,
    InterferenceMatrix,
    ResourceChannel,
    ResourcePool,
    WickensResource,
)


# ═══════════════════════════════════════════════════════════════════════════
# AllocationResult — enriched allocation output
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class AllocationResult:
    """Detailed result of a resource allocation optimisation.

    Attributes
    ----------
    allocations : tuple[ChannelAllocation, ...]
        Per-task allocation results.
    total_throughput : float
        Sum of effective throughput across all tasks (bits/s).
    total_cost : float
        Total cognitive cost of the allocation.
    spare_capacity : Dict[WickensResource, float]
        Remaining capacity per channel after allocation.
    is_feasible : bool
        True if all demands can be met within capacity.
    bottleneck : WickensResource or None
        Most-loaded channel, if any.
    """

    allocations: Tuple[ChannelAllocation, ...]
    total_throughput: float = 0.0
    total_cost: float = 0.0
    spare_capacity: Dict[WickensResource, float] = None  # type: ignore[assignment]
    is_feasible: bool = True
    bottleneck: Optional[WickensResource] = None


# ═══════════════════════════════════════════════════════════════════════════
# Water-filling allocation
# ═══════════════════════════════════════════════════════════════════════════

def water_filling_allocate(
    channel_capacities: np.ndarray,
    noise_levels: np.ndarray,
    total_power: float,
) -> np.ndarray:
    """Classic water-filling algorithm for capacity allocation.

    Distributes ``total_power`` across channels to maximise total
    throughput Σ log₂(1 + Pᵢ/Nᵢ), subject to Σ Pᵢ ≤ total_power.

    Parameters
    ----------
    channel_capacities : np.ndarray
        Maximum capacity per channel (unused here but kept for context).
    noise_levels : np.ndarray
        Effective noise level per channel (inverse SNR).  Higher noise
        means the channel is harder to use.
    total_power : float
        Total resource budget to distribute.

    Returns
    -------
    np.ndarray
        Allocated power per channel.  Shape matches ``noise_levels``.
    """
    n = len(noise_levels)
    noise = np.asarray(noise_levels, dtype=np.float64)
    noise = np.maximum(noise, 1e-12)

    # Sort channels by noise level (ascending).
    order = np.argsort(noise)
    sorted_noise = noise[order]

    # Iterative water-filling: find water level ν such that
    # Σ max(ν − Nᵢ, 0) = total_power.
    allocated = np.zeros(n, dtype=np.float64)
    remaining = total_power
    active = n

    for k in range(n):
        water_level = remaining / active + sorted_noise[k]
        # Check if weakest active channel gets non-negative allocation.
        if water_level - sorted_noise[k] < 0:
            active -= 1
            remaining = total_power
            continue
        break

    # Final water level.
    if active <= 0:
        return np.zeros(n, dtype=np.float64)

    water_level = (remaining + np.sum(sorted_noise[:active])) / active
    alloc_sorted = np.maximum(water_level - sorted_noise, 0.0)
    alloc_sorted[active:] = 0.0

    # Un-sort.
    for i, idx in enumerate(order):
        allocated[idx] = alloc_sorted[i]

    return allocated


# ═══════════════════════════════════════════════════════════════════════════
# MRTAllocator — main allocator class
# ═══════════════════════════════════════════════════════════════════════════

class MRTAllocator:
    """Allocate task demands across MRT resource channels.

    Implements the ``ResourceAllocator`` protocol from
    ``usability_oracle.channel.protocols``.

    Parameters
    ----------
    interference_matrix : InterferenceMatrix or None
        Pre-computed interference model.  If None, channels are
        treated as independent.
    priority_weights : Mapping[str, float] or None
        Task-ID → priority weight for priority-based allocation.
        Higher weight = more resources.
    """

    def __init__(
        self,
        interference_matrix: Optional[InterferenceMatrix] = None,
        priority_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self._interference = interference_matrix
        self._priorities = dict(priority_weights) if priority_weights else {}

    # ---- single-task allocation ----------------------------------------

    def allocate(
        self,
        task_demands: Mapping[WickensResource, float],
        pool: ResourcePool,
        task_id: str = "task",
    ) -> ChannelAllocation:
        """Allocate a single task's demand to the resource pool.

        Each channel gets the minimum of its demand and available capacity.
        The bottleneck is the channel with the highest demand-to-capacity
        ratio.

        Parameters
        ----------
        task_demands : Mapping[WickensResource, float]
            Demand in bits/s per resource.
        pool : ResourcePool
            Available resource channels.
        task_id : str
            Task identifier.

        Returns
        -------
        ChannelAllocation
        """
        actual_demands: Dict[WickensResource, float] = {}
        total_demand = 0.0
        total_throughput = 0.0
        worst_ratio = -1.0
        bottleneck: Optional[WickensResource] = None

        for resource, demand in task_demands.items():
            ch = pool.channel_by_resource(resource)
            if ch is None:
                actual_demands[resource] = demand
                total_demand += demand
                continue

            available = ch.available_capacity
            allocated = min(demand, available)
            actual_demands[resource] = allocated
            total_demand += demand

            # Effective throughput accounts for interference.
            eff = allocated
            if self._interference is not None:
                eff = self._apply_interference_penalty(
                    resource, allocated, task_demands, pool,
                )
            total_throughput += eff

            ratio = demand / ch.capacity_bits_per_s if ch.capacity_bits_per_s > 0 else float("inf")
            if ratio > worst_ratio:
                worst_ratio = ratio
                bottleneck = resource

        return ChannelAllocation(
            task_id=task_id,
            demands=actual_demands,
            total_demand_bits_per_s=total_demand,
            bottleneck_resource=bottleneck,
            effective_throughput_bits_per_s=total_throughput,
        )

    # ---- multi-task concurrent allocation ------------------------------

    def allocate_concurrent(
        self,
        task_demands_list: Sequence[Mapping[WickensResource, float]],
        pool: ResourcePool,
        task_ids: Optional[Sequence[str]] = None,
    ) -> Sequence[ChannelAllocation]:
        """Allocate multiple concurrent tasks to a shared resource pool.

        Uses proportional allocation: when multiple tasks demand the same
        channel, the budget is split proportionally to demand (or priority).

        Parameters
        ----------
        task_demands_list : Sequence[Mapping[WickensResource, float]]
            Demand profiles for each concurrent task.
        pool : ResourcePool
            Shared resource pool.
        task_ids : Sequence[str] or None
            Task identifiers.

        Returns
        -------
        Sequence[ChannelAllocation]
        """
        n = len(task_demands_list)
        if task_ids is None:
            task_ids = [f"task_{i}" for i in range(n)]

        # Aggregate demand per resource.
        agg_demand: Dict[WickensResource, float] = {}
        for demands in task_demands_list:
            for r, d in demands.items():
                agg_demand[r] = agg_demand.get(r, 0.0) + d

        # Per-channel capacity.
        cap_map: Dict[WickensResource, float] = {}
        for ch in pool.channels:
            cap_map[ch.resource] = ch.available_capacity

        # Compute share fractions.
        results: List[ChannelAllocation] = []
        for idx, demands in enumerate(task_demands_list):
            tid = task_ids[idx]
            priority = self._priorities.get(tid, 1.0)
            allocated: Dict[WickensResource, float] = {}
            total_d = 0.0
            total_tp = 0.0
            worst_ratio = -1.0
            bottleneck: Optional[WickensResource] = None

            for resource, demand in demands.items():
                total_agg = agg_demand.get(resource, demand)
                cap = cap_map.get(resource, float("inf"))

                if total_agg <= cap:
                    # No contention — full allocation.
                    alloc = demand
                else:
                    # Proportional sharing weighted by priority.
                    weight = (demand * priority)
                    total_weight = sum(
                        td_list.get(resource, 0.0) * self._priorities.get(
                            task_ids[k] if task_ids else f"task_{k}", 1.0,
                        )
                        for k, td_list in enumerate(task_demands_list)
                    )
                    share = weight / total_weight if total_weight > 0 else 1.0 / n
                    alloc = cap * share

                allocated[resource] = alloc
                total_d += demand

                # Effective throughput with interference from other tasks.
                eff = alloc
                if self._interference is not None:
                    # Compute concurrent load from other tasks.
                    concurrent: Dict[WickensResource, float] = {}
                    for k, other in enumerate(task_demands_list):
                        if k == idx:
                            continue
                        for r, d in other.items():
                            concurrent[r] = concurrent.get(r, 0.0) + d
                    ch_obj = pool.channel_by_resource(resource)
                    if ch_obj is not None:
                        eff = self._effective_with_concurrent(
                            resource, alloc, concurrent, ch_obj,
                        )
                total_tp += eff

                cap_r = cap_map.get(resource, 1.0)
                ratio = demand / cap_r if cap_r > 0 else float("inf")
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    bottleneck = resource

            results.append(ChannelAllocation(
                task_id=tid,
                demands=allocated,
                total_demand_bits_per_s=total_d,
                bottleneck_resource=bottleneck,
                effective_throughput_bits_per_s=total_tp,
            ))

        return results

    # ---- priority-based allocation with preemption ---------------------

    def allocate_with_preemption(
        self,
        task_demands_list: Sequence[Mapping[WickensResource, float]],
        pool: ResourcePool,
        task_ids: Optional[Sequence[str]] = None,
    ) -> Sequence[ChannelAllocation]:
        """Priority-based allocation where high-priority tasks preempt.

        Tasks are allocated in descending priority order.  Each task
        receives its full demand up to remaining capacity; lower-priority
        tasks get whatever is left.

        Parameters
        ----------
        task_demands_list : Sequence[Mapping[WickensResource, float]]
        pool : ResourcePool
        task_ids : Sequence[str] or None

        Returns
        -------
        Sequence[ChannelAllocation]
        """
        n = len(task_demands_list)
        if task_ids is None:
            task_ids = [f"task_{i}" for i in range(n)]

        # Sort by priority descending.
        priorities = [self._priorities.get(tid, 1.0) for tid in task_ids]
        order = sorted(range(n), key=lambda k: -priorities[k])

        remaining: Dict[WickensResource, float] = {}
        for ch in pool.channels:
            remaining[ch.resource] = ch.available_capacity

        results: List[Optional[ChannelAllocation]] = [None] * n

        for idx in order:
            demands = task_demands_list[idx]
            tid = task_ids[idx]
            allocated: Dict[WickensResource, float] = {}
            total_d = 0.0
            total_tp = 0.0
            worst_ratio = -1.0
            bottleneck: Optional[WickensResource] = None

            for resource, demand in demands.items():
                avail = remaining.get(resource, float("inf"))
                alloc = min(demand, avail)
                allocated[resource] = alloc
                remaining[resource] = remaining.get(resource, 0.0) - alloc
                total_d += demand
                total_tp += alloc

                cap = avail
                ratio = demand / cap if cap > 0 else float("inf")
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    bottleneck = resource

            results[idx] = ChannelAllocation(
                task_id=tid,
                demands=allocated,
                total_demand_bits_per_s=total_d,
                bottleneck_resource=bottleneck,
                effective_throughput_bits_per_s=total_tp,
            )

        return [r for r in results if r is not None]

    # ---- dynamic reallocation ------------------------------------------

    def reallocate(
        self,
        current_allocations: Sequence[ChannelAllocation],
        new_demands: Mapping[str, Mapping[WickensResource, float]],
        pool: ResourcePool,
    ) -> Sequence[ChannelAllocation]:
        """Dynamically reallocate as task demands change.

        Keeps existing allocations for unchanged tasks and reallocates
        only for tasks with updated demands.

        Parameters
        ----------
        current_allocations : Sequence[ChannelAllocation]
            Existing allocations.
        new_demands : Mapping[str, Mapping[WickensResource, float]]
            Updated demand profiles keyed by task_id.
        pool : ResourcePool
            Resource pool.

        Returns
        -------
        Sequence[ChannelAllocation]
            Updated allocations.
        """
        # Build combined demand list: updated tasks use new demands,
        # others keep existing.
        all_demands: List[Mapping[WickensResource, float]] = []
        all_ids: List[str] = []

        for alloc in current_allocations:
            if alloc.task_id in new_demands:
                all_demands.append(new_demands[alloc.task_id])
            else:
                all_demands.append(alloc.demands)
            all_ids.append(alloc.task_id)

        return self.allocate_concurrent(all_demands, pool, task_ids=all_ids)

    # ---- robust allocation under uncertainty ---------------------------

    def allocate_robust(
        self,
        demand_intervals: Mapping[WickensResource, Tuple[float, float]],
        pool: ResourcePool,
        task_id: str = "task",
        confidence: float = 0.95,
    ) -> ChannelAllocation:
        """Allocate under demand uncertainty using worst-case analysis.

        Uses the upper bound of each demand interval (scaled by
        confidence) to ensure the allocation is feasible with high
        probability.

        Parameters
        ----------
        demand_intervals : Mapping[WickensResource, Tuple[float, float]]
            (low, high) demand bounds per resource.
        pool : ResourcePool
        task_id : str
        confidence : float
            Target feasibility probability.

        Returns
        -------
        ChannelAllocation
        """
        # Use percentile between low and high based on confidence.
        robust_demands: Dict[WickensResource, float] = {}
        for resource, (lo, hi) in demand_intervals.items():
            robust_demands[resource] = lo + confidence * (hi - lo)

        return self.allocate(robust_demands, pool, task_id=task_id)

    # ---- water-filling multi-channel allocation ------------------------

    def allocate_water_filling(
        self,
        task_demands: Mapping[WickensResource, float],
        pool: ResourcePool,
        total_budget: Optional[float] = None,
        task_id: str = "task",
    ) -> ChannelAllocation:
        """Water-filling allocation of a total budget across channels.

        Maximises total throughput by allocating more resources to
        channels with better signal (lower interference/noise).

        Parameters
        ----------
        task_demands : Mapping[WickensResource, float]
            Demand per channel (used to compute effective noise).
        pool : ResourcePool
        total_budget : float or None
            Total capacity budget.  If None, uses pool total capacity.
        task_id : str

        Returns
        -------
        ChannelAllocation
        """
        resources = list(task_demands.keys())
        demands = np.array([task_demands[r] for r in resources])
        n = len(resources)

        # Channel capacities.
        caps = np.zeros(n, dtype=np.float64)
        for i, r in enumerate(resources):
            ch = pool.channel_by_resource(r)
            caps[i] = ch.capacity_bits_per_s if ch else 10.0

        # Noise = demand / capacity (higher demand relative to capacity = noisier).
        noise = demands / np.maximum(caps, 1e-6)

        budget = total_budget if total_budget is not None else float(np.sum(caps))
        allocated_power = water_filling_allocate(caps, noise, budget)

        # Convert power back to throughput.
        alloc_demands: Dict[WickensResource, float] = {}
        total_tp = 0.0
        worst_ratio = -1.0
        bottleneck: Optional[WickensResource] = None

        for i, r in enumerate(resources):
            alloc = min(allocated_power[i], demands[i])
            alloc_demands[r] = alloc
            total_tp += alloc

            ratio = demands[i] / caps[i] if caps[i] > 0 else float("inf")
            if ratio > worst_ratio:
                worst_ratio = ratio
                bottleneck = r

        return ChannelAllocation(
            task_id=task_id,
            demands=alloc_demands,
            total_demand_bits_per_s=float(np.sum(demands)),
            bottleneck_resource=bottleneck,
            effective_throughput_bits_per_s=total_tp,
        )

    # ---- helpers -------------------------------------------------------

    def _apply_interference_penalty(
        self,
        resource: WickensResource,
        allocated: float,
        all_demands: Mapping[WickensResource, float],
        pool: ResourcePool,
    ) -> float:
        """Reduce effective throughput by interference from co-active channels."""
        if self._interference is None:
            return allocated

        label = resource.value
        if label not in self._interference.resource_labels:
            return allocated

        idx = self._interference.resource_labels.index(label)
        penalty = 0.0
        for other_r, other_d in all_demands.items():
            if other_r == resource:
                continue
            other_label = other_r.value
            if other_label not in self._interference.resource_labels:
                continue
            j = self._interference.resource_labels.index(other_label)
            ch = pool.channel_by_resource(other_r)
            load = other_d / ch.capacity_bits_per_s if ch and ch.capacity_bits_per_s > 0 else 0.5
            penalty += self._interference.coefficients[idx][j] * load

        return allocated * max(1.0 - penalty, 0.05)

    def _effective_with_concurrent(
        self,
        resource: WickensResource,
        allocated: float,
        concurrent_load: Dict[WickensResource, float],
        channel: ResourceChannel,
    ) -> float:
        """Compute effective throughput given concurrent load from other tasks."""
        if self._interference is None:
            return allocated

        label = resource.value
        if label not in self._interference.resource_labels:
            return allocated

        idx = self._interference.resource_labels.index(label)
        penalty = 0.0
        for other_r, load in concurrent_load.items():
            other_label = other_r.value
            if other_label not in self._interference.resource_labels:
                continue
            j = self._interference.resource_labels.index(other_label)
            norm_load = load / channel.capacity_bits_per_s if channel.capacity_bits_per_s > 0 else 0.5
            penalty += self._interference.coefficients[idx][j] * norm_load

        return allocated * max(1.0 - penalty, 0.05)


__all__ = [
    "AllocationResult",
    "MRTAllocator",
    "water_filling_allocate",
]

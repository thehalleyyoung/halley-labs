"""
usability_oracle.channel.integration — Integration with cost algebra.

Bridges the MRT channel analysis layer with the cost algebra and
free-energy framework:

* Map MRT channel interference → parallel composition cost (⊗)
* Map channel capacity → bounded-rationality parameter β
* Map resource demands → free-energy cognitive cost
* Unified cognitive load metric from multi-channel analysis

This module allows the channel-level analysis (Wickens MRT) to feed
directly into the compositional cost algebra, so that UI evaluation
can combine bottom-up channel analysis with top-down free-energy
optimisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.channel.types import (
    ChannelAllocation,
    InterferenceMatrix,
    ResourceChannel,
    ResourcePool,
    WickensResource,
)
from usability_oracle.algebra.models import CostElement, Parallel, Leaf
from usability_oracle.algebra.parallel import ParallelComposer


# ═══════════════════════════════════════════════════════════════════════════
# Channel interference → parallel composition (⊗)
# ═══════════════════════════════════════════════════════════════════════════

def interference_to_parallel_cost(
    cost_a: CostElement,
    cost_b: CostElement,
    allocation_a: ChannelAllocation,
    allocation_b: ChannelAllocation,
    interference_matrix: Optional[InterferenceMatrix] = None,
) -> CostElement:
    """Convert MRT channel interference to parallel composition cost.

    Computes the interference factor η from the channel allocations,
    then applies the parallel composition operator ⊗:

        cost(a ⊗ b) = max(μ_a, μ_b) + η · min(μ_a, μ_b)

    Parameters
    ----------
    cost_a, cost_b : CostElement
        Cost tuples for the two concurrent tasks.
    allocation_a, allocation_b : ChannelAllocation
        MRT channel allocations.
    interference_matrix : InterferenceMatrix or None
        Pre-computed interference matrix.

    Returns
    -------
    CostElement
        Parallel-composed cost.
    """
    eta = compute_interference_factor(allocation_a, allocation_b, interference_matrix)
    composer = ParallelComposer()
    return composer.compose(cost_a, cost_b, interference=eta)


def compute_interference_factor(
    allocation_a: ChannelAllocation,
    allocation_b: ChannelAllocation,
    interference_matrix: Optional[InterferenceMatrix] = None,
) -> float:
    """Compute the interference factor η ∈ [0, 1] between two allocations.

    If an interference matrix is provided, η is the weighted sum of
    per-channel interference coefficients scaled by normalised demands.
    Otherwise, η is estimated from demand overlap using a simple
    Jaccard-like metric.

    Parameters
    ----------
    allocation_a, allocation_b : ChannelAllocation
    interference_matrix : InterferenceMatrix or None

    Returns
    -------
    float
        Interference factor η ∈ [0, 1].
    """
    if interference_matrix is not None:
        return _eta_from_matrix(allocation_a, allocation_b, interference_matrix)
    return _eta_from_overlap(allocation_a, allocation_b)


def _eta_from_matrix(
    a: ChannelAllocation,
    b: ChannelAllocation,
    matrix: InterferenceMatrix,
) -> float:
    """Compute η using the interference matrix."""
    total_intf = 0.0
    total_weight = 0.0
    for ra, da in a.demands.items():
        la = ra.value
        if la not in matrix.resource_labels:
            continue
        ia = matrix.resource_labels.index(la)
        for rb, db in b.demands.items():
            lb = rb.value
            if lb not in matrix.resource_labels:
                continue
            ib = matrix.resource_labels.index(lb)
            w = da * db
            total_intf += w * matrix.coefficients[ia][ib]
            total_weight += w

    if total_weight < 1e-12:
        return 0.0
    return min(total_intf / total_weight, 1.0)


def _eta_from_overlap(
    a: ChannelAllocation,
    b: ChannelAllocation,
) -> float:
    """Estimate η from demand overlap (no matrix)."""
    shared = set(a.demands.keys()) & set(b.demands.keys())
    union = set(a.demands.keys()) | set(b.demands.keys())
    if not union:
        return 0.0
    # Weighted overlap.
    overlap_demand = sum(
        min(a.demands.get(r, 0.0), b.demands.get(r, 0.0)) for r in shared
    )
    total_demand = sum(
        max(a.demands.get(r, 0.0), b.demands.get(r, 0.0)) for r in union
    )
    if total_demand < 1e-12:
        return 0.0
    return min(overlap_demand / total_demand, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Channel capacity → bounded-rationality parameter β
# ═══════════════════════════════════════════════════════════════════════════

def capacity_to_beta(
    pool: ResourcePool,
    task_demands: Optional[Mapping[WickensResource, float]] = None,
    base_beta: float = 1.0,
    scaling: float = 1.0,
) -> float:
    """Convert channel capacity to bounded-rationality parameter β.

    β controls the trade-off between cognitive cost and decision quality
    in the free-energy framework.  Higher capacity → higher β (more
    rational behaviour).

    β = base_β · scaling · (available_capacity / total_demand)

    When capacity greatly exceeds demand, β → large (near-optimal).
    When demand exceeds capacity, β → small (more noise / satisficing).

    Parameters
    ----------
    pool : ResourcePool
        Operator's resource pool.
    task_demands : Mapping[WickensResource, float] or None
        Current task demands.  If None, uses total pool capacity.
    base_beta : float
        Baseline β value.
    scaling : float
        Multiplicative scaling factor.

    Returns
    -------
    float
        Bounded-rationality parameter β > 0.
    """
    total_cap = pool.total_capacity_bits_per_s
    if total_cap <= 0:
        return 0.01

    if task_demands is not None:
        total_demand = sum(task_demands.values())
        if total_demand <= 0:
            return base_beta * scaling * 10.0  # very low demand → near-optimal
        ratio = total_cap / total_demand
    else:
        ratio = 1.0

    beta = base_beta * scaling * ratio
    return max(beta, 0.01)


def beta_from_channel_analysis(
    pool: ResourcePool,
    allocations: Sequence[ChannelAllocation],
    interference_matrix: Optional[InterferenceMatrix] = None,
) -> float:
    """Compute β from a full multi-task channel analysis.

    Accounts for:
    1. Total spare capacity.
    2. Bottleneck severity.
    3. Interference penalty.

    Parameters
    ----------
    pool : ResourcePool
    allocations : Sequence[ChannelAllocation]
    interference_matrix : InterferenceMatrix or None

    Returns
    -------
    float
        β > 0.
    """
    total_cap = pool.total_capacity_bits_per_s
    total_demand = sum(a.total_demand_bits_per_s for a in allocations)
    total_throughput = sum(a.effective_throughput_bits_per_s for a in allocations)

    if total_demand <= 0:
        return 10.0

    # Base ratio: capacity / demand.
    ratio = total_cap / total_demand

    # Throughput efficiency: effective throughput / demand.
    efficiency = total_throughput / total_demand if total_demand > 0 else 1.0

    # Interference penalty.
    intf_penalty = 1.0
    if interference_matrix is not None and len(allocations) >= 2:
        # Average pairwise interference.
        n = len(allocations)
        total_intf = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_intf += compute_interference_factor(
                    allocations[i], allocations[j], interference_matrix,
                )
                pairs += 1
        avg_intf = total_intf / pairs if pairs > 0 else 0.0
        intf_penalty = 1.0 - 0.5 * avg_intf  # reduce β by up to 50 %

    beta = ratio * efficiency * intf_penalty
    return max(beta, 0.01)


# ═══════════════════════════════════════════════════════════════════════════
# Resource demands → free-energy cost
# ═══════════════════════════════════════════════════════════════════════════

def demands_to_cost_element(
    demands: Mapping[WickensResource, float],
    pool: ResourcePool,
    base_time_s: float = 1.0,
) -> CostElement:
    """Convert MRT resource demands to a CostElement tuple.

    Maps from the channel-capacity domain to the cost-algebra domain:

    * **μ** = weighted sum of demand/capacity ratios × base_time
    * **σ²** = variance from capacity uncertainty
    * **κ** = skewness from bottleneck asymmetry
    * **λ** = overload probability

    Parameters
    ----------
    demands : Mapping[WickensResource, float]
        Demand in bits/s per resource.
    pool : ResourcePool
        Operator's resource pool.
    base_time_s : float
        Base completion time for scaling.

    Returns
    -------
    CostElement
    """
    ratios: List[float] = []
    overload_count = 0

    for resource, demand in demands.items():
        ch = pool.channel_by_resource(resource)
        if ch is None:
            ratios.append(demand / 10.0)
            continue
        cap = ch.capacity_bits_per_s
        if cap <= 0:
            overload_count += 1
            ratios.append(2.0)
        else:
            r = demand / cap
            ratios.append(r)
            if r > 1.0:
                overload_count += 1

    if not ratios:
        return CostElement(mu=0.0, sigma_sq=0.0, kappa=0.0, lambda_=0.0)

    arr = np.array(ratios)

    # μ: RMS of demand/capacity ratios scaled by base time.
    mu = float(np.sqrt(np.mean(arr ** 2))) * base_time_s

    # σ²: variance of ratios.
    sigma_sq = float(np.var(arr)) * base_time_s ** 2

    # κ: skewness — positive if there's a bottleneck with high ratio.
    mean_r = float(np.mean(arr))
    std_r = float(np.std(arr))
    if std_r > 1e-6:
        kappa = float(np.mean(((arr - mean_r) / std_r) ** 3))
    else:
        kappa = 0.0

    # λ: fraction of channels overloaded.
    n_channels = len(ratios)
    lambda_ = overload_count / n_channels if n_channels > 0 else 0.0

    return CostElement(
        mu=max(mu, 0.0),
        sigma_sq=max(sigma_sq, 0.0),
        kappa=kappa,
        lambda_=min(max(lambda_, 0.0), 1.0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Unified cognitive load metric
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class UnifiedCognitiveLoad:
    """Unified cognitive load metric from multi-channel analysis.

    Combines channel-level analysis into a single composite metric
    suitable for UI evaluation.

    Attributes
    ----------
    overall_load : float
        Normalised cognitive load in [0, 1].
    channel_loads : Dict[WickensResource, float]
        Per-channel load [0, 1].
    bottleneck : WickensResource or None
    interference_cost : float
        Interference contribution to load.
    spare_capacity_fraction : float
        Fraction of total capacity unused.
    beta : float
        Corresponding bounded-rationality parameter.
    cost_element : CostElement
        Equivalent cost-algebra tuple.
    """

    overall_load: float = 0.0
    channel_loads: Dict[WickensResource, float] = None  # type: ignore[assignment]
    bottleneck: Optional[WickensResource] = None
    interference_cost: float = 0.0
    spare_capacity_fraction: float = 1.0
    beta: float = 1.0
    cost_element: Optional[CostElement] = None


def compute_unified_load(
    allocations: Sequence[ChannelAllocation],
    pool: ResourcePool,
    interference_matrix: Optional[InterferenceMatrix] = None,
) -> UnifiedCognitiveLoad:
    """Compute the unified cognitive load from channel allocations.

    Parameters
    ----------
    allocations : Sequence[ChannelAllocation]
    pool : ResourcePool
    interference_matrix : InterferenceMatrix or None

    Returns
    -------
    UnifiedCognitiveLoad
    """
    # Aggregate demands per channel.
    agg: Dict[WickensResource, float] = {}
    for alloc in allocations:
        for r, d in alloc.demands.items():
            agg[r] = agg.get(r, 0.0) + d

    # Per-channel loads.
    channel_loads: Dict[WickensResource, float] = {}
    worst_ratio = -1.0
    bottleneck: Optional[WickensResource] = None

    for ch in pool.channels:
        demand = agg.get(ch.resource, 0.0)
        if ch.capacity_bits_per_s > 0:
            load = min(demand / ch.capacity_bits_per_s, 1.0)
        else:
            load = 1.0 if demand > 0 else 0.0
        channel_loads[ch.resource] = load
        if load > worst_ratio:
            worst_ratio = load
            bottleneck = ch.resource

    # Overall load: RMS of channel loads.
    if channel_loads:
        loads_arr = np.array(list(channel_loads.values()))
        overall = float(np.sqrt(np.mean(loads_arr ** 2)))
    else:
        overall = 0.0

    # Spare capacity.
    total_cap = pool.total_capacity_bits_per_s
    total_demand = sum(agg.values())
    spare = max(0.0, (total_cap - total_demand) / total_cap) if total_cap > 0 else 0.0

    # Interference cost.
    intf_cost = 0.0
    if interference_matrix is not None and len(allocations) >= 2:
        for i in range(len(allocations)):
            for j in range(i + 1, len(allocations)):
                intf_cost += compute_interference_factor(
                    allocations[i], allocations[j], interference_matrix,
                )

    # β from channel analysis.
    beta = beta_from_channel_analysis(pool, allocations, interference_matrix)

    # Cost element.
    cost_elem = demands_to_cost_element(agg, pool)

    return UnifiedCognitiveLoad(
        overall_load=min(overall, 1.0),
        channel_loads=channel_loads,
        bottleneck=bottleneck,
        interference_cost=intf_cost,
        spare_capacity_fraction=spare,
        beta=beta,
        cost_element=cost_elem,
    )


__all__ = [
    "UnifiedCognitiveLoad",
    "beta_from_channel_analysis",
    "capacity_to_beta",
    "compute_interference_factor",
    "compute_unified_load",
    "demands_to_cost_element",
    "interference_to_parallel_cost",
]

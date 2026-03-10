"""
usability_oracle.channel.protocols — Multiple Resource Theory channel protocols.

Structural interfaces for Wickens MRT resource allocation, interference
modelling, and capacity estimation.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usability_oracle.channel.types import (
        ChannelAllocation,
        InterferenceMatrix,
        ResourceChannel,
        ResourcePool,
        WickensResource,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ResourceAllocator — allocate task demand across resource channels
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ResourceAllocator(Protocol):
    """Allocate a task's information-processing demand across MRT channels.

    Given a task description and available resource pool, determine
    how demand is distributed and whether any channel is overloaded.
    """

    def allocate(
        self,
        task_demands: Mapping[WickensResource, float],
        pool: ResourcePool,
    ) -> ChannelAllocation:
        """Allocate task demand to resource channels.

        Parameters
        ----------
        task_demands : Mapping[WickensResource, float]
            Required demand in bits/s per resource.
        pool : ResourcePool
            Available resource channels.

        Returns
        -------
        ChannelAllocation
            Allocation result with bottleneck identification.
        """
        ...

    def allocate_concurrent(
        self,
        task_demands_list: Sequence[Mapping[WickensResource, float]],
        pool: ResourcePool,
    ) -> Sequence[ChannelAllocation]:
        """Allocate multiple concurrent tasks to the same resource pool.

        Parameters
        ----------
        task_demands_list : Sequence[Mapping[WickensResource, float]]
            Demand profiles for each concurrent task.
        pool : ResourcePool
            Shared resource pool.

        Returns
        -------
        Sequence[ChannelAllocation]
            Per-task allocation results accounting for interference.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# InterferenceModel — compute inter-channel interference
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class InterferenceModel(Protocol):
    """Model interference between concurrent resource channel demands.

    Implements Wickens' conflict matrix logic: tasks sharing the same
    resource dimension interfere more than tasks on orthogonal dimensions.
    """

    def compute_interference(
        self,
        allocation_a: ChannelAllocation,
        allocation_b: ChannelAllocation,
    ) -> float:
        """Compute aggregate interference between two concurrent allocations.

        Parameters
        ----------
        allocation_a : ChannelAllocation
            First task's allocation.
        allocation_b : ChannelAllocation
            Second task's allocation.

        Returns
        -------
        float
            Interference score in [0, 1]. 0 = independent, 1 = fully conflicting.
        """
        ...

    def build_interference_matrix(
        self,
        resources: Sequence[WickensResource],
    ) -> InterferenceMatrix:
        """Construct the interference matrix for a set of resource channels.

        Parameters
        ----------
        resources : Sequence[WickensResource]
            Resource channels to include.

        Returns
        -------
        InterferenceMatrix
            Pairwise interference coefficients.
        """
        ...

    def effective_capacity(
        self,
        channel: ResourceChannel,
        concurrent_load: Mapping[WickensResource, float],
        interference: InterferenceMatrix,
    ) -> float:
        """Compute effective channel capacity under concurrent interference.

        Parameters
        ----------
        channel : ResourceChannel
            The channel whose effective capacity to compute.
        concurrent_load : Mapping[WickensResource, float]
            Current load on other channels.
        interference : InterferenceMatrix
            Interference model.

        Returns
        -------
        float
            Effective capacity in bits/s after interference degradation.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# CapacityEstimator — estimate per-user channel capacities
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class CapacityEstimator(Protocol):
    """Estimate resource channel capacities for a user population.

    Accounts for individual differences (age, expertise, disability)
    via population percentile parametrisation.
    """

    def estimate_pool(
        self,
        *,
        population_percentile: float = 50.0,
        age_years: Optional[float] = None,
        expertise_level: Optional[str] = None,
    ) -> ResourcePool:
        """Estimate a complete resource pool for a user profile.

        Parameters
        ----------
        population_percentile : float
            Percentile in the general population (0–100).
        age_years : Optional[float]
            User age for age-related capacity adjustments.
        expertise_level : Optional[str]
            ``"novice"``, ``"intermediate"``, or ``"expert"``.

        Returns
        -------
        ResourcePool
            Estimated resource channel capacities.
        """
        ...

    def estimate_channel_capacity(
        self,
        resource: WickensResource,
        *,
        population_percentile: float = 50.0,
    ) -> ResourceChannel:
        """Estimate capacity for a single resource channel.

        Parameters
        ----------
        resource : WickensResource
            Which resource to estimate.
        population_percentile : float
            Population percentile.

        Returns
        -------
        ResourceChannel
            Estimated channel with capacity bounds.
        """
        ...


__all__ = [
    "CapacityEstimator",
    "InterferenceModel",
    "ResourceAllocator",
]

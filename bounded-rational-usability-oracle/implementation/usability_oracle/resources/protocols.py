"""
usability_oracle.resources.protocols — Structural interfaces for
Wickens' multiple-resource theory computations.

Defines protocols for resource allocation, interference computation,
and demand estimation using Wickens' four-dimensional resource model.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usability_oracle.resources.types import (
        DemandVector,
        InterferenceMatrix,
        Resource,
        ResourceAllocation,
        ResourceConflict,
        ResourceDemand,
    )


# ═══════════════════════════════════════════════════════════════════════════
# DemandEstimator
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class DemandEstimator(Protocol):
    """Estimate the cognitive resource demand of UI operations.

    Maps UI operations (click, read, type, scan, …) to demand vectors
    in Wickens' four-dimensional resource space.
    """

    def estimate(
        self,
        operation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> DemandVector:
        """Estimate the demand vector for a single UI operation.

        Parameters:
            operation: Serialised description of the operation
                (type, target element, parameters).
            context: Optional contextual information (concurrent tasks,
                environmental load, user expertise).

        Returns:
            A :class:`DemandVector` quantifying the resource demands.
        """
        ...

    def estimate_batch(
        self,
        operations: Sequence[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Sequence[DemandVector]:
        """Estimate demand vectors for multiple operations.

        Parameters:
            operations: Sequence of serialised operation descriptions.
            context: Shared context for all operations.

        Returns:
            One :class:`DemandVector` per operation.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# InterferenceComputer
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class InterferenceComputer(Protocol):
    """Compute pairwise interference between concurrent operations.

    Uses Wickens' conflict matrix to determine how much two operations
    interfere when performed concurrently — operations sharing the same
    resource dimension produce higher interference.
    """

    def compute_pairwise(
        self,
        demand_a: DemandVector,
        demand_b: DemandVector,
    ) -> Sequence[ResourceConflict]:
        """Compute conflicts between two demand vectors.

        Parameters:
            demand_a: Demand vector of the first operation.
            demand_b: Demand vector of the second operation.

        Returns:
            Sequence of :class:`ResourceConflict` for each shared
            resource, sorted by severity descending.
        """
        ...

    def compute_matrix(
        self,
        demand_vectors: Sequence[DemandVector],
    ) -> InterferenceMatrix:
        """Compute the full pairwise interference matrix.

        Parameters:
            demand_vectors: Demand vectors for all concurrent operations.

        Returns:
            An :class:`InterferenceMatrix` with all pairwise
            interference costs.
        """
        ...

    def total_interference(
        self,
        demand_vectors: Sequence[DemandVector],
    ) -> float:
        """Compute total interference cost across all pairs.

        Parameters:
            demand_vectors: Demand vectors for all concurrent operations.

        Returns:
            Scalar total interference cost (sum of pairwise costs).
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# ResourceAllocator
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ResourceAllocator(Protocol):
    """Optimally allocate cognitive resources across concurrent operations.

    Given a set of demand vectors, finds the allocation that minimises
    total expected cost (task completion time + error cost) subject
    to the capacity constraints of each resource.
    """

    def allocate(
        self,
        demand_vectors: Sequence[DemandVector],
        resource_capacities: Optional[Dict[str, float]] = None,
    ) -> ResourceAllocation:
        """Compute optimal resource allocation.

        Parameters:
            demand_vectors: One :class:`DemandVector` per concurrent
                operation.
            resource_capacities: Optional mapping from resource label
                to total capacity.  If ``None``, unit capacity (1.0)
                is assumed for all resources.

        Returns:
            Optimal :class:`ResourceAllocation` minimising total cost.
        """
        ...

    def compute_spare_capacity(
        self,
        allocation: ResourceAllocation,
        resource_capacities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute remaining spare capacity per resource.

        Parameters:
            allocation: Current allocation.
            resource_capacities: Total capacities (defaults to 1.0).

        Returns:
            Mapping  resource_label → spare capacity in [0, 1].
        """
        ...

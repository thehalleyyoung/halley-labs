"""
usability_oracle.resources.conflict_matrix — Resource conflict computation.

Provides the ResourceConflictMatrix class for building and analysing
pairwise interference matrices among concurrent UI operations.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.resources.types import (
    DemandVector,
    InterferenceMatrix,
    Resource,
    ResourceConflict,
    ResourceDemand,
)
from usability_oracle.resources.wickens_model import WickensModel


class ResourceConflictMatrix:
    """Build and analyse resource-conflict matrices.

    Wraps a WickensModel to provide matrix-level operations:
    find critical resources, compute marginal conflict, suggest
    reallocation, and compute conflict gradients.
    """

    def __init__(self, model: Optional[WickensModel] = None) -> None:
        self._model = model or WickensModel()

    # -------------------------------------------------------------------
    # Build interference matrix
    # -------------------------------------------------------------------

    def build_conflict_matrix(
        self, demands: Sequence[DemandVector],
    ) -> InterferenceMatrix:
        """Build the pairwise interference matrix.

        M[i,j] = Σ_r interference(demand_i[r], demand_j[r])

        Parameters:
            demands: One DemandVector per concurrent operation.

        Returns:
            InterferenceMatrix with upper-triangular entries.
        """
        return self._model.compute_matrix(demands)

    # -------------------------------------------------------------------
    # Find critical resource
    # -------------------------------------------------------------------

    def find_critical_resource(
        self, demands: Sequence[DemandVector],
    ) -> Optional[Resource]:
        """Identify the most contested resource across all operations.

        The critical resource is the one with the highest total
        demand summed across all operations.

        Returns:
            The Resource with the highest aggregate demand, or None.
        """
        resource_totals: Dict[Tuple[str, ...], Tuple[Resource, float]] = {}
        for dv in demands:
            for rd in dv.demands:
                key = rd.resource.dimension_tuple
                existing = resource_totals.get(key)
                if existing is None:
                    resource_totals[key] = (rd.resource, rd.demand_level)
                else:
                    resource_totals[key] = (
                        existing[0],
                        existing[1] + rd.demand_level,
                    )
        if not resource_totals:
            return None
        best = max(resource_totals.values(), key=lambda x: x[1])
        return best[0]

    # -------------------------------------------------------------------
    # Marginal conflict
    # -------------------------------------------------------------------

    def compute_marginal_conflict(
        self,
        new_demand: DemandVector,
        existing_demands: Sequence[DemandVector],
    ) -> float:
        """Compute the marginal conflict cost of adding a new demand.

        Marginal conflict = Σ_j interference(new, existing_j)

        Parameters:
            new_demand: Demand vector of the new operation.
            existing_demands: Already-running operations.

        Returns:
            Total marginal interference cost.
        """
        total = 0.0
        for dv in existing_demands:
            conflicts = self._model.compute_pairwise(new_demand, dv)
            total += sum(c.conflict_severity for c in conflicts)
        return total

    # -------------------------------------------------------------------
    # Suggest resource reallocation
    # -------------------------------------------------------------------

    def suggest_resource_reallocation(
        self,
        demands: Sequence[DemandVector],
        constraints: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[str]]:
        """Suggest modality/code shifts to minimise total conflict.

        Heuristic: for each pair with high conflict, suggest switching
        one operation to a different modality or code if feasible.

        Parameters:
            demands: Current demand vectors.
            constraints: Optional per-resource capacity constraints.

        Returns:
            Dict mapping operation_id to list of reallocation suggestions.
        """
        matrix = self._model.compute_matrix(demands)
        suggestions: Dict[str, List[str]] = {}

        for conflict in matrix.conflicts:
            if conflict.conflict_severity < 0.3:
                continue  # only suggest for significant conflicts
            # Suggest modality shift
            op_a = conflict.operation_a
            op_b = conflict.operation_b
            r = conflict.resource
            if r.modality is not None:
                other_mod = (
                    "auditory" if r.modality.value == "visual" else "visual"
                )
                msg = (
                    f"Consider shifting to {other_mod} modality to "
                    f"reduce {r.stage.value} conflict "
                    f"(severity={conflict.conflict_severity:.2f})"
                )
                suggestions.setdefault(op_b, []).append(msg)
            if r.code is not None:
                other_code = (
                    "verbal" if r.code.value == "spatial" else "spatial"
                )
                msg = (
                    f"Consider switching to {other_code} coding to "
                    f"reduce conflict "
                    f"(severity={conflict.conflict_severity:.2f})"
                )
                suggestions.setdefault(op_b, []).append(msg)

        return suggestions

    # -------------------------------------------------------------------
    # Conflict gradient
    # -------------------------------------------------------------------

    def compute_conflict_gradient(
        self, demands: Sequence[DemandVector],
    ) -> Dict[str, float]:
        """Compute sensitivity of total conflict to demand changes.

        For each operation, estimates ∂(total_conflict) / ∂(demand_i)
        using finite differences.

        Returns:
            Dict mapping operation_id to gradient value.
        """
        demands_list = list(demands)
        n = len(demands_list)
        base_conflict = sum(self._model.compute_matrix(demands_list).matrix)
        gradients: Dict[str, float] = {}
        epsilon = 0.01

        for i in range(n):
            op_id = demands_list[i].operation_id
            # Perturb demand_i upward
            perturbed = _perturb_demand(demands_list[i], epsilon)
            modified = demands_list[:i] + [perturbed] + demands_list[i + 1:]
            new_conflict = sum(self._model.compute_matrix(modified).matrix)
            grad = (new_conflict - base_conflict) / epsilon
            gradients[op_id] = grad

        return gradients

    # -------------------------------------------------------------------
    # Full matrix as numpy array
    # -------------------------------------------------------------------

    def as_numpy_matrix(
        self, demands: Sequence[DemandVector],
    ) -> np.ndarray:
        """Return the full n×n symmetric interference matrix.

        Diagonal is 0.  M[i,j] = M[j,i] = pairwise interference.
        """
        mat = self._model.compute_matrix(demands)
        n = mat.num_operations
        full = np.zeros((n, n), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                full[i, j] = mat.matrix[idx]
                full[j, i] = mat.matrix[idx]
                idx += 1
        return full


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perturb_demand(dv: DemandVector, epsilon: float) -> DemandVector:
    """Create a copy of a DemandVector with all demands increased by ε."""
    new_demands = []
    for rd in dv.demands:
        new_level = min(1.0, rd.demand_level + epsilon)
        new_demands.append(ResourceDemand(
            resource=rd.resource,
            demand_level=new_level,
            operation_id=rd.operation_id,
            description=rd.description,
        ))
    return DemandVector(
        demands=tuple(new_demands),
        operation_id=dv.operation_id,
    )

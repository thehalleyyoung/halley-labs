"""
usability_oracle.resources.allocator — Resource allocation optimizer.

Optimally distributes cognitive resources across concurrent UI operations
to minimise total expected cost, using greedy, LP-based, and dynamic
allocation strategies.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.resources.types import (
    DemandVector,
    Resource,
    ResourceAllocation,
    ResourceDemand,
)
from usability_oracle.resources.wickens_model import WickensModel


class ResourceAllocator:
    """Cognitive resource allocator implementing ResourceAllocator protocol.

    Provides multiple allocation strategies:
    - allocate: optimal (greedy proportional) allocation
    - greedy_allocation: priority-based greedy
    - lp_allocation: linear programming (scipy.optimize.linprog)
    - dynamic_allocation: time-varying allocation
    - slack_analysis / bottleneck_resource: capacity analysis
    """

    def __init__(self, model: Optional[WickensModel] = None) -> None:
        self._model = model or WickensModel()

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _collect_resource_labels(
        demand_vectors: Sequence[DemandVector],
    ) -> List[str]:
        """Collect unique resource labels across all demand vectors."""
        labels: dict[str, None] = {}
        for dv in demand_vectors:
            for rd in dv.demands:
                labels[rd.resource.label or rd.resource.dimension_tuple[0]] = None
        return list(labels)

    @staticmethod
    def _default_capacities(
        labels: Sequence[str],
        resource_capacities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        caps: Dict[str, float] = {}
        for lbl in labels:
            caps[lbl] = (resource_capacities or {}).get(lbl, 1.0)
        return caps

    # -------------------------------------------------------------------
    # allocate (protocol method)
    # -------------------------------------------------------------------

    def allocate(
        self,
        demand_vectors: Sequence[DemandVector],
        resource_capacities: Optional[Dict[str, float]] = None,
    ) -> ResourceAllocation:
        """Compute optimal resource allocation (proportional sharing).

        Each resource's capacity is divided among competing operations
        proportionally to their demand levels.

        Parameters:
            demand_vectors: One DemandVector per concurrent operation.
            resource_capacities: Optional per-resource total capacity.

        Returns:
            ResourceAllocation minimising interference cost.
        """
        if not demand_vectors:
            return ResourceAllocation(
                allocations={}, total_cost=0.0,
                interference_cost=0.0, spare_capacity={},
            )

        labels = self._collect_resource_labels(demand_vectors)
        caps = self._default_capacities(labels, resource_capacities)

        # Build demand table: resource_label -> [(op_id, demand_level)]
        demand_table: Dict[str, List[Tuple[str, float]]] = {lbl: [] for lbl in labels}
        for dv in demand_vectors:
            for rd in dv.demands:
                lbl = rd.resource.label or rd.resource.dimension_tuple[0]
                demand_table.setdefault(lbl, []).append(
                    (dv.operation_id, rd.demand_level)
                )

        # Proportional allocation
        allocations: Dict[str, Dict[str, float]] = {}
        spare: Dict[str, float] = {}
        for lbl in labels:
            entries = demand_table.get(lbl, [])
            total_demand = sum(d for _, d in entries)
            cap = caps[lbl]
            if total_demand <= 0:
                spare[lbl] = cap
                continue
            for op_id, dem in entries:
                fraction = min(dem / total_demand * cap, dem) if total_demand > 0 else 0.0
                allocations.setdefault(op_id, {})[lbl] = fraction
            allocated = min(total_demand, cap)
            spare[lbl] = max(0.0, cap - allocated)

        # Compute costs
        interference_cost = self._model.total_interference(demand_vectors)
        total_cost = sum(dv.total_demand for dv in demand_vectors) + interference_cost

        return ResourceAllocation(
            allocations=allocations,
            total_cost=total_cost,
            interference_cost=interference_cost,
            spare_capacity=spare,
        )

    # -------------------------------------------------------------------
    # compute_spare_capacity (protocol method)
    # -------------------------------------------------------------------

    def compute_spare_capacity(
        self,
        allocation: ResourceAllocation,
        resource_capacities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute remaining spare capacity per resource."""
        # Sum allocations per resource
        used: Dict[str, float] = {}
        for op_allocs in allocation.allocations.values():
            for lbl, frac in op_allocs.items():
                used[lbl] = used.get(lbl, 0.0) + frac

        all_labels = set(used) | set(allocation.spare_capacity)
        caps = {lbl: (resource_capacities or {}).get(lbl, 1.0) for lbl in all_labels}
        return {lbl: max(0.0, caps[lbl] - used.get(lbl, 0.0)) for lbl in all_labels}

    # -------------------------------------------------------------------
    # Greedy allocation
    # -------------------------------------------------------------------

    def greedy_allocation(
        self,
        demands: Sequence[DemandVector],
        priorities: Optional[Dict[str, float]] = None,
    ) -> ResourceAllocation:
        """Priority-based greedy allocation.

        Operations are served in priority order (highest first).
        Each operation receives as much resource as it demands,
        up to remaining capacity.

        Parameters:
            demands: Demand vectors for concurrent operations.
            priorities: Optional mapping op_id -> priority (higher = first).
        """
        if not demands:
            return ResourceAllocation(
                allocations={}, total_cost=0.0,
                interference_cost=0.0, spare_capacity={},
            )

        labels = self._collect_resource_labels(demands)
        remaining = {lbl: 1.0 for lbl in labels}

        # Sort by priority (descending)
        pri = priorities or {}
        sorted_demands = sorted(
            demands, key=lambda dv: pri.get(dv.operation_id, 0.0), reverse=True,
        )

        allocations: Dict[str, Dict[str, float]] = {}
        for dv in sorted_demands:
            op_alloc: Dict[str, float] = {}
            for rd in dv.demands:
                lbl = rd.resource.label or rd.resource.dimension_tuple[0]
                avail = remaining.get(lbl, 1.0)
                given = min(rd.demand_level, avail)
                op_alloc[lbl] = given
                remaining[lbl] = remaining.get(lbl, 1.0) - given
            allocations[dv.operation_id] = op_alloc

        spare = {lbl: max(0.0, v) for lbl, v in remaining.items()}
        interference_cost = self._model.total_interference(demands)
        total_cost = sum(dv.total_demand for dv in demands) + interference_cost

        return ResourceAllocation(
            allocations=allocations,
            total_cost=total_cost,
            interference_cost=interference_cost,
            spare_capacity=spare,
        )

    # -------------------------------------------------------------------
    # LP allocation
    # -------------------------------------------------------------------

    def lp_allocation(
        self,
        demands: Sequence[DemandVector],
        available: Optional[Dict[str, float]] = None,
        objective: str = "minimize_unmet",
    ) -> ResourceAllocation:
        """Linear programming allocation via scipy.optimize.linprog.

        Minimises total unmet demand subject to capacity constraints.

        Decision variables: x[i,r] = allocation of resource r to operation i.

        Constraints:
            Σ_i x[i,r] ≤ capacity[r]   ∀ r  (capacity)
            x[i,r] ≤ demand[i,r]         ∀ i,r (no over-allocation)
            x[i,r] ≥ 0

        Objective: minimise Σ (demand[i,r] − x[i,r])  (unmet demand)
        """
        from scipy.optimize import linprog

        if not demands:
            return ResourceAllocation(
                allocations={}, total_cost=0.0,
                interference_cost=0.0, spare_capacity={},
            )

        labels = self._collect_resource_labels(demands)
        n_ops = len(demands)
        n_res = len(labels)
        label_idx = {lbl: i for i, lbl in enumerate(labels)}
        caps = self._default_capacities(labels, available)

        # Decision variable layout: x[op * n_res + res]
        n_vars = n_ops * n_res

        # Build demand table
        demand_table = np.zeros((n_ops, n_res))
        for i, dv in enumerate(demands):
            for rd in dv.demands:
                lbl = rd.resource.label or rd.resource.dimension_tuple[0]
                if lbl in label_idx:
                    demand_table[i, label_idx[lbl]] = rd.demand_level

        # Objective: minimise Σ(demand - x) = Σ demand - Σ x
        # Equivalent to maximise Σ x, so c = -1 for all vars
        c = -np.ones(n_vars)

        # Capacity constraints: Σ_i x[i,r] ≤ cap[r]
        A_ub_rows = []
        b_ub_vals = []
        for r in range(n_res):
            row = np.zeros(n_vars)
            for i in range(n_ops):
                row[i * n_res + r] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(caps[labels[r]])

        A_ub = np.array(A_ub_rows) if A_ub_rows else None
        b_ub = np.array(b_ub_vals) if b_ub_vals else None

        # Bounds: 0 ≤ x[i,r] ≤ demand[i,r]
        bounds = []
        for i in range(n_ops):
            for r in range(n_res):
                bounds.append((0.0, demand_table[i, r]))

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        # Parse solution
        allocations: Dict[str, Dict[str, float]] = {}
        spare: Dict[str, float] = {lbl: caps[lbl] for lbl in labels}
        if result.success:
            x = result.x
            for i, dv in enumerate(demands):
                op_alloc: Dict[str, float] = {}
                for r, lbl in enumerate(labels):
                    val = float(x[i * n_res + r])
                    if val > 1e-9:
                        op_alloc[lbl] = val
                        spare[lbl] -= val
                allocations[dv.operation_id] = op_alloc
        spare = {lbl: max(0.0, v) for lbl, v in spare.items()}

        interference_cost = self._model.total_interference(demands)
        total_cost = sum(dv.total_demand for dv in demands) + interference_cost

        return ResourceAllocation(
            allocations=allocations,
            total_cost=total_cost,
            interference_cost=interference_cost,
            spare_capacity=spare,
        )

    # -------------------------------------------------------------------
    # Dynamic allocation
    # -------------------------------------------------------------------

    def dynamic_allocation(
        self,
        demands_sequence: Sequence[Sequence[DemandVector]],
        available: Optional[Dict[str, float]] = None,
    ) -> List[ResourceAllocation]:
        """Time-varying allocation over a sequence of demand snapshots.

        At each time step, runs proportional allocation on the current
        demands, allowing resources to be redistributed.

        Parameters:
            demands_sequence: List of demand vectors per time step.
            available: Per-resource capacity (constant across time).

        Returns:
            One ResourceAllocation per time step.
        """
        return [
            self.allocate(step_demands, available)
            for step_demands in demands_sequence
        ]

    # -------------------------------------------------------------------
    # Slack analysis
    # -------------------------------------------------------------------

    def slack_analysis(
        self,
        allocation: ResourceAllocation,
        available: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Identify unused resource capacity.

        Returns mapping resource_label → spare capacity.
        """
        return self.compute_spare_capacity(allocation, available)

    # -------------------------------------------------------------------
    # Bottleneck resource
    # -------------------------------------------------------------------

    def bottleneck_resource(
        self, allocation: ResourceAllocation,
    ) -> Optional[str]:
        """Identify the resource limiting overall performance.

        The bottleneck is the resource with the least spare capacity
        (highest utilisation).

        Returns:
            Resource label of the bottleneck, or None if empty.
        """
        if not allocation.spare_capacity:
            return None
        return min(allocation.spare_capacity, key=allocation.spare_capacity.get)  # type: ignore[arg-type]

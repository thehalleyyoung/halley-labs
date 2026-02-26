"""
MonotoneBoundPropagator: propagate bounds across subgraph boundaries
using monotone fixed-point iteration.

After initial subgraph bounds are computed, this module propagates
bound information between neighboring subgraphs. The key invariant
is monotonicity: bounds can only tighten (narrow), never widen, ensuring
convergence to a fixed point that is at least as tight as the initial bounds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import shortest_path, breadth_first_order

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    """Strategy for ordering bound propagation across boundaries."""
    ROUND_ROBIN = auto()
    WIDEST_FIRST = auto()
    NARROWEST_FIRST = auto()
    MOST_CONNECTED_FIRST = auto()
    BFS_ORDER = auto()
    PRIORITY_QUEUE = auto()


@dataclass
class PropagationBound:
    """Bound for a single subgraph, used during propagation."""
    subgraph_id: int
    lower: np.ndarray
    upper: np.ndarray
    separator_vars: Dict[int, List[int]] = field(default_factory=dict)
    last_updated: int = 0

    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def total_width(self) -> float:
        return float(np.sum(self.upper - self.lower))


@dataclass
class PropagationResult:
    """Result of bound propagation."""
    bounds: Dict[int, PropagationBound]
    n_iterations: int
    converged: bool
    max_residual: float
    width_history: List[float]
    n_tightenings: int


@dataclass
class AdjacencyInfo:
    """Adjacency information for the subgraph decomposition graph."""
    n_subgraphs: int
    adjacency_list: Dict[int, List[int]]
    shared_separators: Dict[Tuple[int, int], List[int]]
    separator_cardinalities: Dict[int, int] = field(default_factory=dict)


class MonotoneBoundPropagator:
    """
    Propagates bounds across subgraph boundaries using monotone iteration.

    Given initial bounds for each subgraph and the adjacency structure of
    the decomposition, iteratively tightens bounds by propagating information
    from each subgraph to its neighbors through shared separator variables.

    The monotonicity guarantee ensures that:
    - Lower bounds can only increase
    - Upper bounds can only decrease
    - The sequence converges to a fixed point
    """

    def __init__(
        self,
        tolerance: float = 1e-8,
        max_iterations: int = 500,
        scheduling: SchedulingStrategy = SchedulingStrategy.WIDEST_FIRST,
        damping: float = 0.0,
        lipschitz_constant: float = 1.0,
    ):
        """
        Args:
            tolerance: Convergence tolerance on max bound change.
            max_iterations: Maximum propagation iterations.
            scheduling: Strategy for ordering propagation.
            damping: Damping factor in [0,1). 0 = no damping.
            lipschitz_constant: Lipschitz constant for bound transfer.
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.scheduling = scheduling
        self.damping = damping
        self.lipschitz_constant = lipschitz_constant

        self._bounds: Dict[int, PropagationBound] = {}
        self._adjacency: Optional[AdjacencyInfo] = None
        self._iteration: int = 0
        self._width_history: List[float] = []
        self._converged: bool = False

    def propagate(
        self,
        subgraph_bounds: Dict[int, PropagationBound],
        adjacency: AdjacencyInfo,
    ) -> PropagationResult:
        """
        Propagate bounds across all subgraph boundaries to fixed point.

        Args:
            subgraph_bounds: Initial bounds for each subgraph.
            adjacency: Adjacency structure of the decomposition.

        Returns:
            PropagationResult with tightened bounds.
        """
        self._bounds = {
            sg_id: PropagationBound(
                subgraph_id=b.subgraph_id,
                lower=b.lower.copy(),
                upper=b.upper.copy(),
                separator_vars=dict(b.separator_vars),
                last_updated=0,
            )
            for sg_id, b in subgraph_bounds.items()
        }
        self._adjacency = adjacency
        self._iteration = 0
        self._width_history = []
        self._converged = False

        result = self.iterate_to_fixed_point(self.tolerance, self.max_iterations)
        return result

    def iterate_to_fixed_point(
        self, tolerance: float, max_iterations: int
    ) -> PropagationResult:
        """
        Iterate propagation until convergence or max iterations.

        At each iteration:
        1. Select the next boundary to propagate (via scheduling strategy)
        2. Transfer bounds through the separator
        3. Enforce monotonicity (bounds can only tighten)
        4. Check convergence
        """
        if not self._bounds or self._adjacency is None:
            raise RuntimeError("Must set bounds and adjacency first")

        n_tightenings = 0
        total_width = self._compute_total_width()
        self._width_history.append(total_width)

        schedule = self.get_propagation_schedule(self._adjacency)

        for it in range(max_iterations):
            self._iteration = it + 1
            max_change = 0.0

            for sg_i, sg_j in schedule:
                change, tightened = self._propagate_boundary(sg_i, sg_j)
                max_change = max(max_change, change)
                n_tightenings += tightened

            total_width = self._compute_total_width()
            self._width_history.append(total_width)

            if max_change < tolerance:
                self._converged = True
                logger.info(
                    "Propagation converged at iteration %d (residual=%.2e)",
                    it + 1,
                    max_change,
                )
                break

            # Re-schedule if using adaptive strategy
            if self.scheduling in (
                SchedulingStrategy.WIDEST_FIRST,
                SchedulingStrategy.PRIORITY_QUEUE,
            ):
                schedule = self.get_propagation_schedule(self._adjacency)

        return PropagationResult(
            bounds=self._bounds,
            n_iterations=it + 1,
            converged=self._converged,
            max_residual=max_change,
            width_history=self._width_history,
            n_tightenings=n_tightenings,
        )

    def enforce_monotonicity(
        self, old_bounds: PropagationBound, new_bounds: PropagationBound
    ) -> PropagationBound:
        """
        Enforce monotonicity: bounds can only tighten.

        new_lower >= old_lower (lower bounds only increase)
        new_upper <= old_upper (upper bounds only decrease)
        """
        mono_lower = np.maximum(old_bounds.lower, new_bounds.lower)
        mono_upper = np.minimum(old_bounds.upper, new_bounds.upper)

        # Ensure validity: upper >= lower
        mono_upper = np.maximum(mono_upper, mono_lower)

        return PropagationBound(
            subgraph_id=new_bounds.subgraph_id,
            lower=mono_lower,
            upper=mono_upper,
            separator_vars=new_bounds.separator_vars,
            last_updated=self._iteration,
        )

    def get_propagation_schedule(
        self, adjacency: AdjacencyInfo
    ) -> List[Tuple[int, int]]:
        """
        Determine the order in which to propagate bounds across boundaries.

        Args:
            adjacency: Adjacency structure.

        Returns:
            Ordered list of (subgraph_i, subgraph_j) pairs.
        """
        edges = self._get_all_edges(adjacency)

        if self.scheduling == SchedulingStrategy.ROUND_ROBIN:
            return edges

        elif self.scheduling == SchedulingStrategy.WIDEST_FIRST:
            return self._schedule_widest_first(edges)

        elif self.scheduling == SchedulingStrategy.NARROWEST_FIRST:
            return self._schedule_narrowest_first(edges)

        elif self.scheduling == SchedulingStrategy.MOST_CONNECTED_FIRST:
            return self._schedule_most_connected(edges, adjacency)

        elif self.scheduling == SchedulingStrategy.BFS_ORDER:
            return self._schedule_bfs(adjacency)

        elif self.scheduling == SchedulingStrategy.PRIORITY_QUEUE:
            return self._schedule_priority(edges)

        return edges

    def _propagate_boundary(
        self, sg_i: int, sg_j: int
    ) -> Tuple[float, int]:
        """
        Propagate bounds between subgraphs sg_i and sg_j through their
        shared separator variables.

        Returns (max_change, n_tightenings).
        """
        if sg_i not in self._bounds or sg_j not in self._bounds:
            return 0.0, 0

        bi = self._bounds[sg_i]
        bj = self._bounds[sg_j]

        # Find shared separator variables
        shared_vars = self._get_shared_vars(sg_i, sg_j)
        if not shared_vars:
            return 0.0, 0

        max_change = 0.0
        n_tight = 0

        for var_idx in shared_vars:
            dim_i = len(bi.lower)
            dim_j = len(bj.lower)

            if var_idx >= dim_i or var_idx >= dim_j:
                continue

            # Tighten bounds: each subgraph constrains the other
            new_lower_i = bi.lower[var_idx]
            new_upper_i = bi.upper[var_idx]
            new_lower_j = bj.lower[var_idx]
            new_upper_j = bj.upper[var_idx]

            # Intersection of bounds (accounting for Lipschitz propagation)
            L = self.lipschitz_constant
            prop_lower = max(new_lower_i, new_lower_j - L * self._sep_distance(sg_i, sg_j))
            prop_upper = min(new_upper_i, new_upper_j + L * self._sep_distance(sg_i, sg_j))

            # Apply damping
            if self.damping > 0:
                prop_lower = (1 - self.damping) * prop_lower + self.damping * bi.lower[var_idx]
                prop_upper = (1 - self.damping) * prop_upper + self.damping * bi.upper[var_idx]

            # Enforce monotonicity on subgraph i
            tightened_lower_i = max(bi.lower[var_idx], prop_lower)
            tightened_upper_i = min(bi.upper[var_idx], prop_upper)
            tightened_upper_i = max(tightened_upper_i, tightened_lower_i)

            change_i = abs(tightened_lower_i - bi.lower[var_idx]) + abs(
                tightened_upper_i - bi.upper[var_idx]
            )

            if change_i > 1e-14:
                n_tight += 1
                bi.lower[var_idx] = tightened_lower_i
                bi.upper[var_idx] = tightened_upper_i

            max_change = max(max_change, change_i)

            # Symmetric: tighten subgraph j
            prop_lower_j = max(new_lower_j, new_lower_i - L * self._sep_distance(sg_i, sg_j))
            prop_upper_j = min(new_upper_j, new_upper_i + L * self._sep_distance(sg_i, sg_j))

            if self.damping > 0:
                prop_lower_j = (1 - self.damping) * prop_lower_j + self.damping * bj.lower[var_idx]
                prop_upper_j = (1 - self.damping) * prop_upper_j + self.damping * bj.upper[var_idx]

            tightened_lower_j = max(bj.lower[var_idx], prop_lower_j)
            tightened_upper_j = min(bj.upper[var_idx], prop_upper_j)
            tightened_upper_j = max(tightened_upper_j, tightened_lower_j)

            change_j = abs(tightened_lower_j - bj.lower[var_idx]) + abs(
                tightened_upper_j - bj.upper[var_idx]
            )

            if change_j > 1e-14:
                n_tight += 1
                bj.lower[var_idx] = tightened_lower_j
                bj.upper[var_idx] = tightened_upper_j

            max_change = max(max_change, change_j)

        bi.last_updated = self._iteration
        bj.last_updated = self._iteration

        return max_change, n_tight

    def _get_shared_vars(self, sg_i: int, sg_j: int) -> List[int]:
        """Get the separator variable indices shared between two subgraphs."""
        if self._adjacency is None:
            return []

        key = (min(sg_i, sg_j), max(sg_i, sg_j))
        sep_ids = self._adjacency.shared_separators.get(key, [])

        # Each separator maps to variable indices via the bounds
        shared = set()
        for sep_id in sep_ids:
            if sg_i in self._bounds:
                vars_i = self._bounds[sg_i].separator_vars.get(sep_id, [])
                shared.update(vars_i)
            if sg_j in self._bounds:
                vars_j = self._bounds[sg_j].separator_vars.get(sep_id, [])
                shared.update(vars_j)

        return sorted(shared)

    def _sep_distance(self, sg_i: int, sg_j: int) -> float:
        """Compute effective distance between subgraphs through separator.
        
        Returns a small value proportional to the separator discretization,
        representing the maximum error from discretizing the separator.
        """
        if self._adjacency is None:
            return 0.01

        key = (min(sg_i, sg_j), max(sg_i, sg_j))
        sep_ids = self._adjacency.shared_separators.get(key, [])

        if not sep_ids:
            return 0.01

        max_eps = 0.0
        for sep_id in sep_ids:
            card = self._adjacency.separator_cardinalities.get(sep_id, 2)
            max_eps = max(max_eps, 1.0 / max(card, 2))

        return max_eps

    def _compute_total_width(self) -> float:
        """Compute the total width of all bounds."""
        return sum(b.total_width() for b in self._bounds.values())

    def _get_all_edges(self, adjacency: AdjacencyInfo) -> List[Tuple[int, int]]:
        """Get all undirected edges in the adjacency graph."""
        edges = set()
        for sg_i, neighbors in adjacency.adjacency_list.items():
            for sg_j in neighbors:
                edge = (min(sg_i, sg_j), max(sg_i, sg_j))
                edges.add(edge)
        return sorted(edges)

    def _schedule_widest_first(
        self, edges: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Schedule edges by total width of adjacent subgraphs (widest first)."""
        def edge_width(e: Tuple[int, int]) -> float:
            w = 0.0
            if e[0] in self._bounds:
                w += self._bounds[e[0]].total_width()
            if e[1] in self._bounds:
                w += self._bounds[e[1]].total_width()
            return w

        return sorted(edges, key=edge_width, reverse=True)

    def _schedule_narrowest_first(
        self, edges: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Schedule edges by total width (narrowest first, for fine-tuning)."""
        def edge_width(e: Tuple[int, int]) -> float:
            w = 0.0
            if e[0] in self._bounds:
                w += self._bounds[e[0]].total_width()
            if e[1] in self._bounds:
                w += self._bounds[e[1]].total_width()
            return w

        return sorted(edges, key=edge_width)

    def _schedule_most_connected(
        self, edges: List[Tuple[int, int]], adjacency: AdjacencyInfo
    ) -> List[Tuple[int, int]]:
        """Schedule edges by connectivity (most connected nodes first)."""
        degree = {}
        for sg_id, nbrs in adjacency.adjacency_list.items():
            degree[sg_id] = len(nbrs)

        def edge_degree(e: Tuple[int, int]) -> int:
            return degree.get(e[0], 0) + degree.get(e[1], 0)

        return sorted(edges, key=edge_degree, reverse=True)

    def _schedule_bfs(self, adjacency: AdjacencyInfo) -> List[Tuple[int, int]]:
        """Schedule edges in BFS order from the widest subgraph."""
        if not self._bounds:
            return self._get_all_edges(adjacency)

        # Start BFS from the widest subgraph
        start = max(self._bounds.keys(), key=lambda k: self._bounds[k].total_width())

        visited: Set[int] = set()
        queue = [start]
        visited.add(start)
        bfs_edges: List[Tuple[int, int]] = []

        while queue:
            current = queue.pop(0)
            for nbr in adjacency.adjacency_list.get(current, []):
                edge = (min(current, nbr), max(current, nbr))
                if edge not in set(bfs_edges):
                    bfs_edges.append(edge)
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)

        return bfs_edges

    def _schedule_priority(
        self, edges: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Priority-based scheduling: prioritize edges where the bound
        difference across the separator is largest (most room to tighten).
        """
        def priority(e: Tuple[int, int]) -> float:
            shared = self._get_shared_vars(e[0], e[1])
            if not shared:
                return 0.0
            max_diff = 0.0
            bi = self._bounds.get(e[0])
            bj = self._bounds.get(e[1])
            if bi is None or bj is None:
                return 0.0
            for v in shared:
                if v < len(bi.lower) and v < len(bj.lower):
                    diff = abs(bi.lower[v] - bj.lower[v]) + abs(bi.upper[v] - bj.upper[v])
                    max_diff = max(max_diff, diff)
            return max_diff

        return sorted(edges, key=priority, reverse=True)

    def get_convergence_rate(self) -> float:
        """Estimate the convergence rate from the width history."""
        if len(self._width_history) < 3:
            return 0.0

        diffs = np.diff(self._width_history)
        neg_diffs = diffs[diffs < 0]
        if len(neg_diffs) < 2:
            return 0.0

        ratios = neg_diffs[1:] / np.minimum(neg_diffs[:-1], -1e-15)
        finite_ratios = ratios[np.isfinite(ratios) & (ratios > 0) & (ratios < 1)]
        if len(finite_ratios) == 0:
            return 0.0

        return float(np.median(finite_ratios))

    def get_active_boundaries(self) -> List[Tuple[int, int]]:
        """Return boundaries where propagation made progress in last iteration."""
        if self._adjacency is None:
            return []

        active = []
        for edge in self._get_all_edges(self._adjacency):
            bi = self._bounds.get(edge[0])
            bj = self._bounds.get(edge[1])
            if bi and bj and (
                bi.last_updated == self._iteration or bj.last_updated == self._iteration
            ):
                active.append(edge)
        return active

    def propagate_single_step(self) -> Tuple[float, int]:
        """
        Execute one full pass of propagation across all boundaries.

        Returns (max_change, n_tightenings) for this step.
        """
        if not self._bounds or self._adjacency is None:
            return 0.0, 0

        self._iteration += 1
        schedule = self.get_propagation_schedule(self._adjacency)
        max_change = 0.0
        total_tight = 0

        for sg_i, sg_j in schedule:
            change, n_t = self._propagate_boundary(sg_i, sg_j)
            max_change = max(max_change, change)
            total_tight += n_t

        tw = self._compute_total_width()
        self._width_history.append(tw)
        return max_change, total_tight

    def get_bound_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics of current bounds.

        Returns dict with min/max/mean widths, per-subgraph widths,
        and the number of tight (zero-width) dimensions.
        """
        if not self._bounds:
            return {}

        widths = []
        per_sg: Dict[int, float] = {}
        n_tight = 0
        n_total = 0

        for sg_id, b in self._bounds.items():
            w = b.width()
            per_sg[sg_id] = float(np.sum(w))
            widths.extend(w.tolist())
            n_tight += int(np.sum(w < 1e-12))
            n_total += len(w)

        w_arr = np.array(widths)
        return {
            "min_width": float(np.min(w_arr)),
            "max_width": float(np.max(w_arr)),
            "mean_width": float(np.mean(w_arr)),
            "median_width": float(np.median(w_arr)),
            "total_width": float(np.sum(w_arr)),
            "per_subgraph_width": per_sg,
            "n_tight_dims": n_tight,
            "n_total_dims": n_total,
            "fraction_tight": n_tight / max(n_total, 1),
        }

    def estimate_remaining_iterations(self) -> int:
        """
        Estimate remaining iterations to convergence based on
        observed convergence rate.

        Uses geometric extrapolation from recent width reductions.
        """
        if len(self._width_history) < 3:
            return self.max_iterations

        recent = self._width_history[-min(10, len(self._width_history)):]
        diffs = np.diff(recent)
        neg_diffs = diffs[diffs < -1e-15]

        if len(neg_diffs) < 2:
            return self.max_iterations

        rate = self.get_convergence_rate()
        if rate <= 0 or rate >= 1:
            return self.max_iterations

        current_width = self._width_history[-1]
        target_reduction = current_width * self.tolerance

        if target_reduction <= 0:
            return 0

        avg_reduction = float(np.mean(np.abs(neg_diffs)))
        if avg_reduction < 1e-15:
            return self.max_iterations

        remaining = int(np.ceil(np.log(self.tolerance / avg_reduction) / np.log(rate)))
        return max(0, min(remaining, self.max_iterations))

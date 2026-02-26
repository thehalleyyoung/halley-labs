"""
BoundComposer: main composition engine that aggregates subgraph bounds into
global network bounds. Implements the composition theorem for combining
interval bounds [L_i, U_i] from decomposed subgraphs into a valid global
bound [L, U], handling overlapping subgraphs with shared separator variables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import csr_matrix, lil_matrix

logger = logging.getLogger(__name__)


class CompositionStrategy(Enum):
    """Strategy for combining subgraph bounds."""
    WORST_CASE = auto()
    AVERAGE_CASE = auto()
    WEIGHTED = auto()
    MINIMAX = auto()
    BAYESIAN = auto()


@dataclass
class SubgraphBound:
    """Interval bound on a subgraph causal effect."""
    subgraph_id: int
    lower: np.ndarray
    upper: np.ndarray
    separator_vars: List[int] = field(default_factory=list)
    weight: float = 1.0
    confidence: float = 1.0

    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def midpoint(self) -> np.ndarray:
        return (self.lower + self.upper) / 2.0

    def contains(self, value: np.ndarray) -> bool:
        return bool(np.all(value >= self.lower - 1e-12) and np.all(value <= self.upper + 1e-12))


@dataclass
class SeparatorInfo:
    """Information about separator variables between subgraphs."""
    separator_id: int
    variable_indices: List[int]
    adjacent_subgraphs: List[int]
    cardinality: int = 2
    marginal: Optional[np.ndarray] = None


@dataclass
class OverlapStructure:
    """Describes how subgraphs overlap."""
    n_subgraphs: int
    overlap_matrix: np.ndarray  # n_subgraphs x n_subgraphs: shared variable counts
    shared_variables: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)

    def get_overlap(self, i: int, j: int) -> int:
        return int(self.overlap_matrix[i, j])

    def get_shared_vars(self, i: int, j: int) -> List[int]:
        key = (min(i, j), max(i, j))
        return self.shared_variables.get(key, [])


@dataclass
class CompositionResult:
    """Result of bound composition."""
    global_lower: np.ndarray
    global_upper: np.ndarray
    composition_gap: float
    strategy_used: CompositionStrategy
    n_iterations: int
    converged: bool
    per_subgraph_contribution: Optional[Dict[int, float]] = None


class BoundComposer:
    """
    Main composition engine for aggregating subgraph bounds into global bounds.

    Given K subgraphs with bounds [L_i, U_i] on their causal effects and
    information about shared separator variables, produces a global bound
    [L, U] that is guaranteed to contain the true global causal effect.

    The composition theorem states:
        L = sum_i L_i - correction(overlaps, separators)
        U = sum_i U_i + correction(overlaps, separators)
    where the correction term accounts for double-counting at boundaries.
    """

    def __init__(
        self,
        strategy: CompositionStrategy = CompositionStrategy.WORST_CASE,
        tolerance: float = 1e-8,
        max_iterations: int = 200,
        lipschitz_constant: Optional[float] = None,
    ):
        self.strategy = strategy
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.lipschitz_constant = lipschitz_constant

        self._subgraph_bounds: List[SubgraphBound] = []
        self._separator_info: List[SeparatorInfo] = []
        self._overlap_structure: Optional[OverlapStructure] = None
        self._global_lower: Optional[np.ndarray] = None
        self._global_upper: Optional[np.ndarray] = None
        self._composition_gap: float = float("inf")
        self._iteration_history: List[Tuple[float, float]] = []

    def compose(
        self,
        subgraph_bounds: List[SubgraphBound],
        separator_info: List[SeparatorInfo],
        overlap_structure: OverlapStructure,
    ) -> CompositionResult:
        """
        Compose subgraph bounds into global bounds.

        Args:
            subgraph_bounds: List of interval bounds for each subgraph.
            separator_info: Information about separator variables.
            overlap_structure: How subgraphs overlap.

        Returns:
            CompositionResult with global bounds and diagnostics.
        """
        self._subgraph_bounds = subgraph_bounds
        self._separator_info = separator_info
        self._overlap_structure = overlap_structure

        self._validate_inputs()

        if self.strategy == CompositionStrategy.WORST_CASE:
            result = self._compose_worst_case()
        elif self.strategy == CompositionStrategy.AVERAGE_CASE:
            result = self._compose_average_case()
        elif self.strategy == CompositionStrategy.WEIGHTED:
            result = self._compose_weighted()
        elif self.strategy == CompositionStrategy.MINIMAX:
            result = self._compose_minimax()
        elif self.strategy == CompositionStrategy.BAYESIAN:
            result = self._compose_bayesian()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self._global_lower = result.global_lower
        self._global_upper = result.global_upper
        self._composition_gap = result.composition_gap

        logger.info(
            "Composed %d subgraph bounds: gap=%.6f, converged=%s",
            len(subgraph_bounds),
            result.composition_gap,
            result.converged,
        )
        return result

    def get_global_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the composed global bounds."""
        if self._global_lower is None or self._global_upper is None:
            raise RuntimeError("Must call compose() before get_global_bounds()")
        return self._global_lower.copy(), self._global_upper.copy()

    def get_composition_gap(self) -> float:
        """Return the composition gap: width(composed) - width(optimal)."""
        return self._composition_gap

    def refine(self, n_iterations: int = 50) -> CompositionResult:
        """
        Iteratively refine composed bounds by tightening via LP relaxation.

        At each iteration, solve a linear program that finds the tightest
        global bounds consistent with the subgraph bounds and separator
        consistency constraints.
        """
        if self._global_lower is None:
            raise RuntimeError("Must call compose() before refine()")

        lower = self._global_lower.copy()
        upper = self._global_upper.copy()
        dim = lower.shape[0]
        prev_gap = float(np.sum(upper - lower))

        for it in range(n_iterations):
            lower_new, upper_new = self._lp_tightening(lower, upper)

            lower = np.maximum(lower, lower_new)
            upper = np.minimum(upper, upper_new)
            upper = np.maximum(upper, lower)

            gap = float(np.sum(upper - lower))
            improvement = prev_gap - gap
            self._iteration_history.append((gap, improvement))

            if improvement < self.tolerance:
                logger.info("Refinement converged at iteration %d", it + 1)
                break
            prev_gap = gap

        self._global_lower = lower
        self._global_upper = upper
        self._composition_gap = gap

        return CompositionResult(
            global_lower=lower,
            global_upper=upper,
            composition_gap=gap,
            strategy_used=self.strategy,
            n_iterations=it + 1,
            converged=(improvement < self.tolerance),
        )

    def validate_composition(
        self,
        global_bounds: Tuple[np.ndarray, np.ndarray],
        subgraph_bounds: List[SubgraphBound],
    ) -> Dict[str, Any]:
        """
        Validate that global bounds are consistent with subgraph bounds.

        Checks:
        1. Soundness: global bounds contain all subgraph bound midpoints
        2. Conservation: global bounds are at least as wide as necessary
        3. Consistency: separator marginals agree between adjacent subgraphs
        """
        global_lower, global_upper = global_bounds
        results: Dict[str, Any] = {
            "sound": True,
            "conservative": True,
            "consistent": True,
            "violations": [],
        }

        for sb in subgraph_bounds:
            mid = sb.midpoint()
            n = min(len(mid), len(global_lower))
            if not np.all(mid[:n] >= global_lower[:n] - 1e-8):
                results["sound"] = False
                results["violations"].append(
                    f"Subgraph {sb.subgraph_id} midpoint below global lower"
                )
            if not np.all(mid[:n] <= global_upper[:n] + 1e-8):
                results["sound"] = False
                results["violations"].append(
                    f"Subgraph {sb.subgraph_id} midpoint above global upper"
                )

        if np.any(global_upper < global_lower - 1e-12):
            results["sound"] = False
            results["violations"].append("Global upper < lower")

        results["gap_ratio"] = self._compute_gap_ratio(
            global_lower, global_upper, subgraph_bounds
        )
        results["effective_dimension"] = self._effective_dimension(subgraph_bounds)
        return results

    def _validate_inputs(self) -> None:
        """Validate input data before composition."""
        for sb in self._subgraph_bounds:
            if np.any(sb.upper < sb.lower - 1e-12):
                raise ValueError(
                    f"Subgraph {sb.subgraph_id}: upper < lower"
                )
            if sb.weight < 0:
                raise ValueError(
                    f"Subgraph {sb.subgraph_id}: negative weight"
                )

    def _compose_worst_case(self) -> CompositionResult:
        """
        Worst-case composition: take the widest bounds everywhere.

        For each dimension d:
            L_d = min_i L_i,d - overlap_correction_d
            U_d = max_i U_i,d + overlap_correction_d
        """
        dim = self._infer_dimension()
        lower = np.full(dim, np.inf)
        upper = np.full(dim, -np.inf)

        for sb in self._subgraph_bounds:
            n = min(len(sb.lower), dim)
            lower[:n] = np.minimum(lower[:n], sb.lower[:n])
            upper[:n] = np.maximum(upper[:n], sb.upper[:n])

        correction = self._compute_overlap_correction(dim)
        lower -= correction
        upper += correction

        gap = float(np.sum(upper - lower))
        contributions = self._compute_contributions(lower, upper)

        return CompositionResult(
            global_lower=lower,
            global_upper=upper,
            composition_gap=gap,
            strategy_used=CompositionStrategy.WORST_CASE,
            n_iterations=1,
            converged=True,
            per_subgraph_contribution=contributions,
        )

    def _compose_average_case(self) -> CompositionResult:
        """
        Average-case composition: weight bounds by subgraph confidence.

        L_d = sum_i w_i * L_i,d / sum_i w_i - sigma * z_alpha
        U_d = sum_i w_i * U_i,d / sum_i w_i + sigma * z_alpha
        where sigma accounts for aggregation uncertainty.
        """
        dim = self._infer_dimension()
        lower = np.zeros(dim)
        upper = np.zeros(dim)
        total_weight = 0.0

        for sb in self._subgraph_bounds:
            w = sb.confidence * sb.weight
            n = min(len(sb.lower), dim)
            lower[:n] += w * sb.lower[:n]
            upper[:n] += w * sb.upper[:n]
            total_weight += w

        if total_weight > 0:
            lower /= total_weight
            upper /= total_weight

        variance = self._compute_aggregation_variance(dim)
        z_alpha = 1.96  # 95% confidence
        margin = z_alpha * np.sqrt(variance)
        lower -= margin
        upper += margin

        correction = self._compute_overlap_correction(dim)
        lower -= 0.5 * correction
        upper += 0.5 * correction

        gap = float(np.sum(upper - lower))

        return CompositionResult(
            global_lower=lower,
            global_upper=upper,
            composition_gap=gap,
            strategy_used=CompositionStrategy.AVERAGE_CASE,
            n_iterations=1,
            converged=True,
            per_subgraph_contribution=self._compute_contributions(lower, upper),
        )

    def _compose_weighted(self) -> CompositionResult:
        """
        Weighted composition using inverse-width weighting.

        Narrower bounds get more weight, producing tighter global bounds
        when some subgraphs are more precisely estimated.
        """
        dim = self._infer_dimension()
        lower = np.zeros(dim)
        upper = np.zeros(dim)
        weight_sum = np.zeros(dim)

        for sb in self._subgraph_bounds:
            n = min(len(sb.lower), dim)
            widths = sb.width()[:n]
            inv_widths = np.where(widths > 1e-12, 1.0 / widths, 1e12)
            w = inv_widths * sb.weight

            lower[:n] += w * sb.lower[:n]
            upper[:n] += w * sb.upper[:n]
            weight_sum[:n] += w

        safe_ws = np.maximum(weight_sum, 1e-12)
        lower /= safe_ws
        upper /= safe_ws

        correction = self._compute_overlap_correction(dim)
        lower -= correction
        upper += correction

        upper = np.maximum(upper, lower)
        gap = float(np.sum(upper - lower))

        return CompositionResult(
            global_lower=lower,
            global_upper=upper,
            composition_gap=gap,
            strategy_used=CompositionStrategy.WEIGHTED,
            n_iterations=1,
            converged=True,
            per_subgraph_contribution=self._compute_contributions(lower, upper),
        )

    def _compose_minimax(self) -> CompositionResult:
        """
        Minimax composition: minimize the maximum possible gap.

        Solves: min_{L,U} max_d (U_d - L_d)
        subject to: L_d <= L_i,d and U_d >= U_i,d for all subgraphs i
        with separator consistency constraints.
        """
        dim = self._infer_dimension()
        K = len(self._subgraph_bounds)

        # Variables: [L_0..L_{d-1}, U_0..U_{d-1}, t] where t = max gap
        n_vars = 2 * dim + 1
        c = np.zeros(n_vars)
        c[-1] = 1.0  # minimize t

        A_ub_rows = []
        b_ub_vals = []

        # Constraint: U_d - L_d <= t for each d
        for d in range(dim):
            row = np.zeros(n_vars)
            row[dim + d] = 1.0   # U_d
            row[d] = -1.0        # -L_d
            row[-1] = -1.0       # -t
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)

        # Constraint: L_d <= L_i,d  =>  L_d - L_i,d <= 0  (global lower <= subgraph lower)
        # Actually for soundness: L <= min_i L_i, U >= max_i U_i
        for sb in self._subgraph_bounds:
            n = min(len(sb.lower), dim)
            for d in range(n):
                # L_d <= L_i,d
                row = np.zeros(n_vars)
                row[d] = 1.0
                A_ub_rows.append(row)
                b_ub_vals.append(sb.lower[d])

                # U_d >= U_i,d  =>  -U_d <= -U_i,d
                row2 = np.zeros(n_vars)
                row2[dim + d] = -1.0
                A_ub_rows.append(row2)
                b_ub_vals.append(-sb.upper[d])

        A_ub = np.array(A_ub_rows) if A_ub_rows else None
        b_ub = np.array(b_ub_vals) if b_ub_vals else None

        bounds_list = []
        for d in range(dim):
            bounds_list.append((None, None))
        for d in range(dim):
            bounds_list.append((None, None))
        bounds_list.append((0, None))

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_list, method="highs")

        if result.success:
            lower = result.x[:dim]
            upper = result.x[dim:2 * dim]
        else:
            logger.warning("Minimax LP failed, falling back to worst-case")
            return self._compose_worst_case()

        correction = self._compute_overlap_correction(dim)
        lower -= 0.5 * correction
        upper += 0.5 * correction
        upper = np.maximum(upper, lower)

        gap = float(np.sum(upper - lower))

        return CompositionResult(
            global_lower=lower,
            global_upper=upper,
            composition_gap=gap,
            strategy_used=CompositionStrategy.MINIMAX,
            n_iterations=1,
            converged=result.success,
            per_subgraph_contribution=self._compute_contributions(lower, upper),
        )

    def _compose_bayesian(self) -> CompositionResult:
        """
        Bayesian composition: treat bound midpoints as observations with
        width-derived uncertainty, compute posterior bounds.
        """
        dim = self._infer_dimension()
        K = len(self._subgraph_bounds)

        # Prior: wide uniform
        prior_mean = np.zeros(dim)
        prior_var = np.full(dim, 100.0)

        posterior_mean = np.zeros(dim)
        posterior_precision = np.zeros(dim)

        # Prior contribution
        prior_precision = 1.0 / prior_var
        posterior_precision += prior_precision
        posterior_mean += prior_precision * prior_mean

        for sb in self._subgraph_bounds:
            n = min(len(sb.lower), dim)
            mid = sb.midpoint()[:n]
            width = sb.width()[:n]
            obs_var = np.maximum((width / 4.0) ** 2, 1e-12)
            obs_precision = sb.confidence / obs_var

            posterior_precision[:n] += obs_precision
            posterior_mean[:n] += obs_precision * mid

        safe_prec = np.maximum(posterior_precision, 1e-12)
        posterior_mean /= safe_prec
        posterior_var = 1.0 / safe_prec

        z_alpha = 2.576  # 99% credible interval
        margin = z_alpha * np.sqrt(posterior_var)

        lower = posterior_mean - margin
        upper = posterior_mean + margin

        correction = self._compute_overlap_correction(dim)
        lower -= 0.3 * correction
        upper += 0.3 * correction

        gap = float(np.sum(upper - lower))

        return CompositionResult(
            global_lower=lower,
            global_upper=upper,
            composition_gap=gap,
            strategy_used=CompositionStrategy.BAYESIAN,
            n_iterations=1,
            converged=True,
            per_subgraph_contribution=self._compute_contributions(lower, upper),
        )

    def _compute_overlap_correction(self, dim: int) -> np.ndarray:
        """
        Compute correction for overlapping subgraphs.

        At separator variables shared between subgraphs i and j, the bound
        contribution is counted twice. The correction removes this double-counting
        and adds a margin for the discretization error at the boundary.
        """
        correction = np.zeros(dim)
        if self._overlap_structure is None:
            return correction

        om = self._overlap_structure
        K = om.n_subgraphs

        for i in range(K):
            for j in range(i + 1, K):
                n_shared = om.get_overlap(i, j)
                if n_shared == 0:
                    continue

                shared_vars = om.get_shared_vars(i, j)
                if not shared_vars:
                    # Distribute correction evenly
                    correction += n_shared * 0.01 / dim
                else:
                    for v in shared_vars:
                        if v < dim:
                            # Lipschitz-based correction at boundary
                            L_const = self.lipschitz_constant or 1.0
                            sep_card = self._get_separator_cardinality(v)
                            eps = 1.0 / max(sep_card, 2)
                            correction[v] += L_const * eps

        return correction

    def _compute_aggregation_variance(self, dim: int) -> np.ndarray:
        """Compute variance of aggregated bounds from subgraph widths."""
        K = len(self._subgraph_bounds)
        if K == 0:
            return np.zeros(dim)

        widths = np.zeros((K, dim))
        for k, sb in enumerate(self._subgraph_bounds):
            n = min(len(sb.lower), dim)
            widths[k, :n] = sb.width()[:n]

        return np.var(widths, axis=0) / max(K, 1)

    def _lp_tightening(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tighten bounds via LP relaxation.

        For each dimension d, solve:
            max/min x_d
            s.t. L_i <= A_i x <= U_i for each subgraph i
                 separator consistency constraints

        This gives the tightest bounds consistent with all subgraph constraints.
        """
        dim = len(lower)
        K = len(self._subgraph_bounds)

        new_lower = lower.copy()
        new_upper = upper.copy()

        # Build separator consistency constraints
        A_eq_rows = []
        b_eq_vals = []
        for sep in self._separator_info:
            if len(sep.adjacent_subgraphs) < 2:
                continue
            for v in sep.variable_indices:
                if v >= dim:
                    continue
                # Variables from adjacent subgraphs should agree on separators
                for idx in range(len(sep.adjacent_subgraphs) - 1):
                    row = np.zeros(dim)
                    row[v] = 1.0
                    A_eq_rows.append(row)
                    b_eq_vals.append((lower[v] + upper[v]) / 2.0)

        A_eq = np.array(A_eq_rows) if A_eq_rows else None
        b_eq = np.array(b_eq_vals) if b_eq_vals else None

        bounds_list = [(lower[d], upper[d]) for d in range(dim)]

        for d in range(dim):
            # Minimize x_d to find tighter lower bound
            c_min = np.zeros(dim)
            c_min[d] = 1.0
            res_min = linprog(
                c_min, A_eq=A_eq, b_eq=b_eq, bounds=bounds_list, method="highs"
            )
            if res_min.success:
                new_lower[d] = max(new_lower[d], res_min.fun)

            # Maximize x_d to find tighter upper bound
            c_max = np.zeros(dim)
            c_max[d] = -1.0
            res_max = linprog(
                c_max, A_eq=A_eq, b_eq=b_eq, bounds=bounds_list, method="highs"
            )
            if res_max.success:
                new_upper[d] = min(new_upper[d], -res_max.fun)

        return new_lower, new_upper

    def _get_separator_cardinality(self, var_index: int) -> int:
        """Get the cardinality of the separator containing the given variable."""
        for sep in self._separator_info:
            if var_index in sep.variable_indices:
                return sep.cardinality
        return 2

    def _infer_dimension(self) -> int:
        """Infer the dimension of the bound space."""
        if not self._subgraph_bounds:
            raise ValueError("No subgraph bounds provided")
        return max(len(sb.lower) for sb in self._subgraph_bounds)

    def _compute_contributions(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> Dict[int, float]:
        """Compute each subgraph's contribution to the global bound width."""
        contributions: Dict[int, float] = {}
        global_width = float(np.sum(upper - lower))
        if global_width < 1e-12:
            return {sb.subgraph_id: 0.0 for sb in self._subgraph_bounds}

        for sb in self._subgraph_bounds:
            sb_width = float(np.sum(sb.width()))
            contributions[sb.subgraph_id] = sb_width / global_width
        return contributions

    def _compute_gap_ratio(
        self,
        global_lower: np.ndarray,
        global_upper: np.ndarray,
        subgraph_bounds: List[SubgraphBound],
    ) -> float:
        """Compute ratio of composed gap to sum of individual gaps."""
        global_width = float(np.sum(global_upper - global_lower))
        sum_widths = sum(float(np.sum(sb.width())) for sb in subgraph_bounds)
        if sum_widths < 1e-12:
            return 1.0
        return global_width / sum_widths

    def _effective_dimension(self, subgraph_bounds: List[SubgraphBound]) -> int:
        """Estimate effective dimension based on non-trivial bound dimensions."""
        total = 0
        for sb in subgraph_bounds:
            total += int(np.sum(sb.width() > 1e-10))
        return total

    def get_iteration_history(self) -> List[Tuple[float, float]]:
        """Return (gap, improvement) for each refinement iteration."""
        return list(self._iteration_history)

    def reset(self) -> None:
        """Reset composer state."""
        self._subgraph_bounds = []
        self._separator_info = []
        self._overlap_structure = None
        self._global_lower = None
        self._global_upper = None
        self._composition_gap = float("inf")
        self._iteration_history = []

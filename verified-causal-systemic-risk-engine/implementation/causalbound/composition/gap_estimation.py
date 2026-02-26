"""
GapEstimator: estimate the composition gap between composed bounds and
the optimal global bound. Provides Lipschitz-based, separator-size-based,
and empirical gap estimation methods.

The composition gap measures the price of decomposition:
    gap = |composed_bound| - |optimal_global_bound|

Theoretical bound: gap <= O(k * L * s * epsilon)
where k = number of separators, L = Lipschitz constant,
s = max separator cardinality, epsilon = discretization granularity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class GapEstimate:
    """Result of gap estimation."""
    total_gap: float
    lower_gap: float
    upper_gap: float
    theoretical_bound: float
    empirical_estimate: Optional[float] = None
    per_boundary_gaps: Optional[Dict[Tuple[int, int], float]] = None
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class LipschitzInfo:
    """Lipschitz continuity information for a contagion function."""
    constant: float
    domain_dim: int
    estimated: bool = True
    local_constants: Optional[np.ndarray] = None


@dataclass
class GapDecomposition:
    """Decomposition of the gap by boundary/separator."""
    boundary_gaps: Dict[Tuple[int, int], float]
    separator_gaps: Dict[int, float]
    interaction_gap: float
    total: float


class GapEstimator:
    """
    Estimates the composition gap between composed and optimal global bounds.

    The gap arises from two sources:
    1. Boundary effects: information lost at subgraph boundaries
    2. Discretization: separator variables discretized to finite values

    Provides theoretical bounds and empirical estimates of this gap.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        confidence_level: float = 0.95,
        seed: Optional[int] = None,
    ):
        self.n_samples = n_samples
        self.confidence_level = confidence_level
        self._rng = np.random.default_rng(seed)
        self._gap_cache: Dict[str, float] = {}

    def estimate_gap(
        self,
        subgraph_bounds: List[Dict[str, np.ndarray]],
        separators: List[Dict[str, Any]],
        lipschitz_constant: Optional[float] = None,
    ) -> GapEstimate:
        """
        Estimate the composition gap using all available methods.

        Args:
            subgraph_bounds: List of dicts with 'lower', 'upper' arrays.
            separators: List of separator info dicts.
            lipschitz_constant: Lipschitz constant of the contagion function.

        Returns:
            GapEstimate with theoretical and empirical gap estimates.
        """
        theoretical = self._theoretical_gap_bound(
            subgraph_bounds, separators, lipschitz_constant
        )

        composed_lower, composed_upper = self._compose_naive(subgraph_bounds)
        total_width = float(np.sum(composed_upper - composed_lower))
        sum_widths = sum(
            float(np.sum(b["upper"] - b["lower"])) for b in subgraph_bounds
        )
        naive_gap = max(0.0, total_width - sum_widths)

        lower_contributions = []
        upper_contributions = []
        for b in subgraph_bounds:
            lower_contributions.append(b["lower"])
            upper_contributions.append(b["upper"])

        dim = composed_lower.shape[0]
        lower_gap = float(
            np.sum(composed_lower - np.mean(
                [lc[:dim] for lc in lower_contributions if len(lc) >= dim],
                axis=0
            )) if lower_contributions else 0.0
        )
        upper_gap = float(
            np.sum(np.mean(
                [uc[:dim] for uc in upper_contributions if len(uc) >= dim],
                axis=0
            ) - composed_upper) if upper_contributions else 0.0
        )

        empirical = self.empirical_gap(subgraph_bounds, self.n_samples, separators)

        per_boundary = {}
        for sep in separators:
            adj = sep.get("adjacent", [])
            if len(adj) >= 2:
                for i_idx in range(len(adj)):
                    for j_idx in range(i_idx + 1, len(adj)):
                        i, j = adj[i_idx], adj[j_idx]
                        bg = self._boundary_gap(
                            subgraph_bounds, i, j, sep, lipschitz_constant
                        )
                        per_boundary[(i, j)] = bg

        ci = self._gap_confidence_interval(
            subgraph_bounds, separators, empirical
        )

        return GapEstimate(
            total_gap=max(naive_gap, empirical),
            lower_gap=abs(lower_gap),
            upper_gap=abs(upper_gap),
            theoretical_bound=theoretical,
            empirical_estimate=empirical,
            per_boundary_gaps=per_boundary,
            confidence_interval=ci,
        )

    def lipschitz_bound(
        self,
        contagion_function: Callable[[np.ndarray], np.ndarray],
        separator_size: int,
        discretization: int,
        domain: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> LipschitzInfo:
        """
        Estimate Lipschitz constant of the contagion function and compute
        the resulting gap bound.

        Uses sampling-based estimation: L = max_{x,y} |f(x)-f(y)| / |x-y|
        over random pairs in the domain.

        Args:
            contagion_function: The function to analyze.
            separator_size: Number of separator variables.
            discretization: Number of discretization points per variable.
            domain: Optional (lower, upper) bounds on the domain.

        Returns:
            LipschitzInfo with estimated constant and local constants.
        """
        if domain is None:
            domain = (np.zeros(separator_size), np.ones(separator_size))

        lower, upper = domain
        dim = len(lower)

        n_pairs = min(5000, self.n_samples * 5)
        points_x = self._rng.uniform(lower, upper, size=(n_pairs, dim))
        points_y = self._rng.uniform(lower, upper, size=(n_pairs, dim))

        fx = np.array([contagion_function(x) for x in points_x])
        fy = np.array([contagion_function(y) for y in points_y])

        diffs_x = np.linalg.norm(points_x - points_y, axis=1)
        valid = diffs_x > 1e-12

        if fx.ndim == 1:
            diffs_f = np.abs(fx - fy)
        else:
            diffs_f = np.linalg.norm(fx - fy, axis=1)

        ratios = np.where(valid, diffs_f / diffs_x, 0.0)
        global_L = float(np.max(ratios)) if len(ratios) > 0 else 1.0

        # Estimate local Lipschitz constants per dimension
        local_constants = np.zeros(dim)
        for d in range(dim):
            # Perturb only dimension d
            delta = np.zeros(dim)
            h = (upper[d] - lower[d]) / max(discretization, 2)
            delta[d] = h

            n_local = min(500, self.n_samples)
            test_points = self._rng.uniform(lower, upper, size=(n_local, dim))
            perturbed = test_points + delta

            # Clip to domain
            perturbed = np.clip(perturbed, lower, upper)

            f_orig = np.array([contagion_function(x) for x in test_points])
            f_pert = np.array([contagion_function(x) for x in perturbed])

            if f_orig.ndim == 1:
                local_diffs = np.abs(f_orig - f_pert)
            else:
                local_diffs = np.linalg.norm(f_orig - f_pert, axis=1)

            actual_dists = np.linalg.norm(test_points - perturbed, axis=1)
            valid_local = actual_dists > 1e-12
            if np.any(valid_local):
                local_ratios = local_diffs[valid_local] / actual_dists[valid_local]
                local_constants[d] = float(np.max(local_ratios))

        return LipschitzInfo(
            constant=global_L,
            domain_dim=dim,
            estimated=True,
            local_constants=local_constants,
        )

    def empirical_gap(
        self,
        subgraph_bounds: List[Dict[str, np.ndarray]],
        n_samples: int,
        separators: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        Estimate the composition gap empirically via sampling.

        Sample feasible points from each subgraph's bound region,
        compose them, and measure how much wider the composed bounds
        are compared to the tight hull of the samples.

        Args:
            subgraph_bounds: Subgraph interval bounds.
            n_samples: Number of Monte Carlo samples.
            separators: Optional separator info for constrained sampling.

        Returns:
            Empirical gap estimate.
        """
        if not subgraph_bounds:
            return 0.0

        composed_lower, composed_upper = self._compose_naive(subgraph_bounds)
        dim = composed_lower.shape[0]

        # Sample from composed region and check subgraph feasibility
        feasible_samples = []
        for _ in range(n_samples):
            sample = self._rng.uniform(composed_lower, composed_upper)

            if self._check_subgraph_feasibility(sample, subgraph_bounds):
                feasible_samples.append(sample)

        if len(feasible_samples) < 10:
            # Not enough feasible samples — use relaxed check
            for _ in range(n_samples * 3):
                sample = self._rng.uniform(composed_lower, composed_upper)
                feasible_samples.append(sample)

        if not feasible_samples:
            return float(np.sum(composed_upper - composed_lower))

        feasible = np.array(feasible_samples)
        tight_lower = np.min(feasible, axis=0)
        tight_upper = np.max(feasible, axis=0)

        composed_width = np.sum(composed_upper - composed_lower)
        tight_width = np.sum(tight_upper - tight_lower)

        return float(max(0.0, composed_width - tight_width))

    def decompose_gap(
        self,
        subgraph_bounds: List[Dict[str, np.ndarray]],
        separators: List[Dict[str, Any]],
        lipschitz_constant: Optional[float] = None,
    ) -> GapDecomposition:
        """
        Decompose the composition gap by boundary and separator.

        Attributes the total gap to:
        1. Each boundary between adjacent subgraphs
        2. Each separator variable
        3. Interaction effects between boundaries

        Args:
            subgraph_bounds: Subgraph interval bounds.
            separators: Separator information.
            lipschitz_constant: Optional Lipschitz constant.

        Returns:
            GapDecomposition with per-boundary and per-separator gaps.
        """
        L = lipschitz_constant or 1.0
        boundary_gaps: Dict[Tuple[int, int], float] = {}
        separator_gaps: Dict[int, float] = {}
        total_boundary = 0.0

        for sep in separators:
            sep_id = sep.get("id", 0)
            adj = sep.get("adjacent", [])
            vars_list = sep.get("variables", [])
            cardinality = sep.get("cardinality", 2)

            # Gap contribution from this separator
            epsilon = 1.0 / max(cardinality, 2)
            sep_gap = L * len(vars_list) * epsilon
            separator_gaps[sep_id] = sep_gap

            # Distribute gap among adjacent boundaries
            if len(adj) >= 2:
                for i_idx in range(len(adj)):
                    for j_idx in range(i_idx + 1, len(adj)):
                        i, j = adj[i_idx], adj[j_idx]
                        bg = self._boundary_gap(
                            subgraph_bounds, i, j, sep, L
                        )
                        boundary_gaps[(i, j)] = boundary_gaps.get((i, j), 0.0) + bg
                        total_boundary += bg

        total_separator = sum(separator_gaps.values())
        total_gap = total_boundary + total_separator

        # Interaction gap: non-additive component
        estimated_total = self.empirical_gap(
            subgraph_bounds, min(self.n_samples, 500), separators
        )
        interaction_gap = max(0.0, estimated_total - total_gap)

        return GapDecomposition(
            boundary_gaps=boundary_gaps,
            separator_gaps=separator_gaps,
            interaction_gap=interaction_gap,
            total=estimated_total,
        )

    def _theoretical_gap_bound(
        self,
        subgraph_bounds: List[Dict[str, np.ndarray]],
        separators: List[Dict[str, Any]],
        lipschitz_constant: Optional[float],
    ) -> float:
        """
        Compute the theoretical gap bound: O(k * L * s * epsilon).

        k = number of separator boundaries
        L = Lipschitz constant
        s = max separator size (number of variables)
        epsilon = max discretization error
        """
        L = lipschitz_constant or 1.0
        k = len(separators)
        if k == 0:
            return 0.0

        max_sep_size = max(
            len(sep.get("variables", [1])) for sep in separators
        )
        max_cardinality = max(
            sep.get("cardinality", 2) for sep in separators
        )
        epsilon = 1.0 / max(max_cardinality, 2)

        return k * L * max_sep_size * epsilon

    def _compose_naive(
        self, subgraph_bounds: List[Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Naive composition: min of lowers, max of uppers."""
        dim = max(len(b["lower"]) for b in subgraph_bounds)
        lower = np.full(dim, np.inf)
        upper = np.full(dim, -np.inf)

        for b in subgraph_bounds:
            n = min(len(b["lower"]), dim)
            lower[:n] = np.minimum(lower[:n], b["lower"][:n])
            upper[:n] = np.maximum(upper[:n], b["upper"][:n])

        lower = np.where(np.isinf(lower), 0.0, lower)
        upper = np.where(np.isinf(upper), 1.0, upper)

        return lower, upper

    def _check_subgraph_feasibility(
        self, point: np.ndarray, subgraph_bounds: List[Dict[str, np.ndarray]]
    ) -> bool:
        """Check if a point is feasible in at least one subgraph."""
        for b in subgraph_bounds:
            n = min(len(b["lower"]), len(point))
            if np.all(point[:n] >= b["lower"][:n] - 1e-10) and np.all(
                point[:n] <= b["upper"][:n] + 1e-10
            ):
                return True
        return False

    def _boundary_gap(
        self,
        subgraph_bounds: List[Dict[str, np.ndarray]],
        i: int,
        j: int,
        separator: Dict[str, Any],
        lipschitz_constant: Optional[float],
    ) -> float:
        """
        Compute the gap at a specific boundary between subgraphs i and j.

        The boundary gap depends on:
        1. Width difference of bounds at separator variables
        2. Lipschitz constant of the contagion function
        3. Cardinality of the separator
        """
        L = lipschitz_constant or 1.0
        vars_list = separator.get("variables", [])
        cardinality = separator.get("cardinality", 2)

        if i >= len(subgraph_bounds) or j >= len(subgraph_bounds):
            return L * len(vars_list) / max(cardinality, 2)

        bi = subgraph_bounds[i]
        bj = subgraph_bounds[j]

        gap = 0.0
        for v in vars_list:
            if v < len(bi["lower"]) and v < len(bj["lower"]):
                width_i = bi["upper"][v] - bi["lower"][v]
                width_j = bj["upper"][v] - bj["lower"][v]
                # Gap from disagreement on this variable
                gap += abs(width_i - width_j) * L / max(cardinality, 2)
                # Gap from discretization
                gap += L / max(cardinality, 2)
            else:
                gap += L / max(cardinality, 2)

        return gap

    def _gap_confidence_interval(
        self,
        subgraph_bounds: List[Dict[str, np.ndarray]],
        separators: List[Dict[str, Any]],
        empirical_estimate: float,
    ) -> Tuple[float, float]:
        """Compute confidence interval for the gap estimate."""
        # Bootstrap to estimate CI
        n_bootstrap = 200
        gap_samples = []

        for _ in range(n_bootstrap):
            n_sub = len(subgraph_bounds)
            indices = self._rng.choice(n_sub, size=n_sub, replace=True)
            bootstrap_bounds = [subgraph_bounds[i] for i in indices]
            gap = self.empirical_gap(bootstrap_bounds, max(100, self.n_samples // 5))
            gap_samples.append(gap)

        alpha = 1.0 - self.confidence_level
        gap_array = np.array(gap_samples)
        ci_lower = float(np.percentile(gap_array, 100 * alpha / 2))
        ci_upper = float(np.percentile(gap_array, 100 * (1 - alpha / 2)))

        return (ci_lower, ci_upper)

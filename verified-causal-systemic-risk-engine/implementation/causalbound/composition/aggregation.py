"""
GlobalBoundAggregator: aggregate local subgraph bounds into global systemic
risk bounds using multiple aggregation methods with double-counting correction.

Provides conservative (sound) aggregation that accounts for overlapping
subgraphs, with confidence intervals and risk decomposition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.stats import norm, t as t_dist

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Method for aggregating local bounds."""
    CONSERVATIVE = auto()
    SUM = auto()
    MAX = auto()
    WEIGHTED_SUM = auto()
    INCLUSION_EXCLUSION = auto()
    FRECHET = auto()


@dataclass
class LocalBound:
    """Local bound for a single subgraph's systemic risk contribution."""
    subgraph_id: int
    lower: np.ndarray
    upper: np.ndarray
    variables: List[int] = field(default_factory=list)
    weight: float = 1.0
    n_samples: int = 0


@dataclass
class OverlapInfo:
    """Information about overlapping subgraphs."""
    overlap_pairs: Dict[Tuple[int, int], List[int]]  # shared variable indices
    overlap_counts: np.ndarray  # per-variable count of subgraphs containing it
    total_variables: int


@dataclass
class AggregationResult:
    """Result of bound aggregation."""
    global_lower: np.ndarray
    global_upper: np.ndarray
    method: AggregationMethod
    double_counting_correction: float
    confidence_interval: Optional[Tuple[np.ndarray, np.ndarray]] = None
    risk_decomposition: Optional[Dict[int, float]] = None


class GlobalBoundAggregator:
    """
    Aggregates local subgraph bounds into global systemic risk bounds.

    Handles the challenge of combining bounds from overlapping subgraphs
    without double-counting shared variables. Provides multiple aggregation
    methods with different conservatism-tightness tradeoffs.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        n_bootstrap: int = 500,
        seed: Optional[int] = None,
    ):
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self._rng = np.random.default_rng(seed)

    def aggregate(
        self,
        subgraph_bounds: List[LocalBound],
        method: AggregationMethod = AggregationMethod.CONSERVATIVE,
        overlap_info: Optional[OverlapInfo] = None,
    ) -> AggregationResult:
        """
        Aggregate local bounds into global bounds.

        Args:
            subgraph_bounds: Local bounds from each subgraph.
            method: Aggregation method to use.
            overlap_info: Information about subgraph overlaps.

        Returns:
            AggregationResult with global bounds and diagnostics.
        """
        if not subgraph_bounds:
            raise ValueError("No subgraph bounds to aggregate")

        if method == AggregationMethod.CONSERVATIVE:
            return self._aggregate_conservative(subgraph_bounds, overlap_info)
        elif method == AggregationMethod.SUM:
            return self._aggregate_sum(subgraph_bounds, overlap_info)
        elif method == AggregationMethod.MAX:
            return self._aggregate_max(subgraph_bounds, overlap_info)
        elif method == AggregationMethod.WEIGHTED_SUM:
            return self._aggregate_weighted_sum(subgraph_bounds, overlap_info)
        elif method == AggregationMethod.INCLUSION_EXCLUSION:
            return self._aggregate_inclusion_exclusion(subgraph_bounds, overlap_info)
        elif method == AggregationMethod.FRECHET:
            return self._aggregate_frechet(subgraph_bounds, overlap_info)
        else:
            raise ValueError(f"Unknown method: {method}")

    def correct_double_counting(
        self,
        bounds: List[LocalBound],
        overlap_structure: OverlapInfo,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct for double-counting of shared variables in overlapping subgraphs.

        For each variable v that appears in k subgraphs, the contribution
        should be counted once, not k times. We use the average contribution
        weighted by 1/k for each variable.

        Args:
            bounds: Local bounds from subgraphs.
            overlap_structure: Overlap information.

        Returns:
            Corrected (lower, upper) global bounds.
        """
        dim = overlap_structure.total_variables
        lower = np.zeros(dim)
        upper = np.zeros(dim)
        counts = np.zeros(dim)

        for b in bounds:
            for v in b.variables:
                if v < dim:
                    lower[v] += b.lower[b.variables.index(v)] if v in b.variables and b.variables.index(v) < len(b.lower) else 0
                    upper[v] += b.upper[b.variables.index(v)] if v in b.variables and b.variables.index(v) < len(b.upper) else 0
                    counts[v] += 1

        # For variables not covered by any subgraph, use conservative defaults
        uncovered = counts == 0
        counts[uncovered] = 1

        # Divide by the overlap count to correct double-counting
        lower /= counts
        upper /= counts

        return lower, upper

    def compute_confidence(
        self,
        bounds: List[LocalBound],
        n_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence intervals for the aggregated bounds.

        Uses bootstrap resampling over subgraph bounds to estimate
        the uncertainty in the aggregated global bounds.

        Args:
            bounds: Local bounds from subgraphs.
            n_samples: Number of bootstrap samples (default: self.n_bootstrap).

        Returns:
            (ci_lower, ci_upper) confidence interval arrays.
        """
        n_boot = n_samples or self.n_bootstrap
        dim = max(len(b.lower) for b in bounds)
        K = len(bounds)

        lower_samples = np.zeros((n_boot, dim))
        upper_samples = np.zeros((n_boot, dim))

        for b_idx in range(n_boot):
            # Bootstrap: resample subgraphs with replacement
            indices = self._rng.choice(K, size=K, replace=True)
            resampled = [bounds[i] for i in indices]

            # Also add noise proportional to bound width
            agg_lower = np.full(dim, np.inf)
            agg_upper = np.full(dim, -np.inf)

            for b in resampled:
                n = min(len(b.lower), dim)
                width = b.upper[:n] - b.lower[:n]
                noise = self._rng.normal(0, np.maximum(width * 0.05, 1e-10))

                agg_lower[:n] = np.minimum(agg_lower[:n], b.lower[:n] + noise)
                agg_upper[:n] = np.maximum(agg_upper[:n], b.upper[:n] + noise)

            agg_lower = np.where(np.isinf(agg_lower), 0.0, agg_lower)
            agg_upper = np.where(np.isinf(agg_upper), 1.0, agg_upper)

            lower_samples[b_idx] = agg_lower
            upper_samples[b_idx] = agg_upper

        alpha = 1.0 - self.confidence_level
        ci_lower = np.percentile(lower_samples, 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(upper_samples, 100 * (1 - alpha / 2), axis=0)

        return ci_lower, ci_upper

    def get_risk_decomposition(
        self,
        bounds: List[LocalBound],
        subgraphs: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[int, float]:
        """
        Decompose the global risk bound into contributions from each subgraph.

        Uses Shapley value-like attribution: the contribution of subgraph i
        is the average marginal contribution over all orderings.

        For efficiency, approximates using a random sample of orderings.

        Args:
            bounds: Local bounds from subgraphs.
            subgraphs: Optional mapping from subgraph_id to variable list.

        Returns:
            Dict mapping subgraph_id to its risk contribution (fraction of total).
        """
        K = len(bounds)
        if K == 0:
            return {}

        contributions: Dict[int, float] = {}
        total_risk = sum(float(np.sum(b.upper - b.lower)) * b.weight for b in bounds)

        if total_risk < 1e-12:
            return {b.subgraph_id: 1.0 / K for b in bounds}

        # Approximate Shapley values via random orderings
        n_orderings = min(200, 2 ** K)
        shapley = {b.subgraph_id: 0.0 for b in bounds}

        for _ in range(n_orderings):
            perm = self._rng.permutation(K)
            cumulative_risk = 0.0

            for idx in perm:
                b = bounds[idx]
                risk_with = cumulative_risk + float(np.sum(b.upper - b.lower)) * b.weight
                marginal = risk_with - cumulative_risk
                shapley[b.subgraph_id] += marginal / n_orderings
                cumulative_risk = risk_with

        # Normalize
        total_shapley = sum(shapley.values())
        if total_shapley > 1e-12:
            contributions = {k: v / total_shapley for k, v in shapley.items()}
        else:
            contributions = {b.subgraph_id: 1.0 / K for b in bounds}

        return contributions

    # ----------------------------------------------------------------
    # Aggregation methods
    # ----------------------------------------------------------------

    def _aggregate_conservative(
        self,
        bounds: List[LocalBound],
        overlap_info: Optional[OverlapInfo],
    ) -> AggregationResult:
        """
        Conservative aggregation: take outer envelope of all subgraph bounds.
        This is always sound but may be loose.

        L_d = min_i L_i,d
        U_d = max_i U_i,d
        """
        dim = max(len(b.lower) for b in bounds)
        lower = np.full(dim, np.inf)
        upper = np.full(dim, -np.inf)

        for b in bounds:
            n = min(len(b.lower), dim)
            lower[:n] = np.minimum(lower[:n], b.lower[:n])
            upper[:n] = np.maximum(upper[:n], b.upper[:n])

        lower = np.where(np.isinf(lower), 0.0, lower)
        upper = np.where(np.isinf(upper), 1.0, upper)

        dc_correction = 0.0
        if overlap_info is not None:
            corrected_lower, corrected_upper = self.correct_double_counting(
                bounds, overlap_info
            )
            # Use corrected bounds where they are tighter than conservative
            inner_lower = np.maximum(lower, corrected_lower)
            inner_upper = np.minimum(upper, corrected_upper)
            inner_upper = np.maximum(inner_upper, inner_lower)

            dc_correction = float(
                np.sum(upper - lower) - np.sum(inner_upper - inner_lower)
            )
            # Keep conservative outer bounds for soundness; report correction
        else:
            corrected_lower, corrected_upper = lower, upper

        ci = self.compute_confidence(bounds)

        return AggregationResult(
            global_lower=lower,
            global_upper=upper,
            method=AggregationMethod.CONSERVATIVE,
            double_counting_correction=dc_correction,
            confidence_interval=ci,
            risk_decomposition=self.get_risk_decomposition(bounds),
        )

    def _aggregate_sum(
        self,
        bounds: List[LocalBound],
        overlap_info: Optional[OverlapInfo],
    ) -> AggregationResult:
        """
        Sum aggregation: sum the risk contributions from each subgraph.

        L = sum_i L_i  (corrected for overlaps)
        U = sum_i U_i  (corrected for overlaps)

        Appropriate when subgraph effects are additive.
        """
        dim = max(len(b.lower) for b in bounds)
        lower = np.zeros(dim)
        upper = np.zeros(dim)

        for b in bounds:
            n = min(len(b.lower), dim)
            lower[:n] += b.lower[:n] * b.weight
            upper[:n] += b.upper[:n] * b.weight

        dc_correction = 0.0
        if overlap_info is not None:
            # Correct for double counting
            for pair, shared_vars in overlap_info.overlap_pairs.items():
                i, j = pair
                bi = next((b for b in bounds if b.subgraph_id == i), None)
                bj = next((b for b in bounds if b.subgraph_id == j), None)
                if bi is None or bj is None:
                    continue

                for v in shared_vars:
                    v_idx_i = bi.variables.index(v) if v in bi.variables else -1
                    v_idx_j = bj.variables.index(v) if v in bj.variables else -1

                    if v_idx_i >= 0 and v_idx_j >= 0 and v < dim:
                        if v_idx_i < len(bi.lower) and v_idx_j < len(bj.lower):
                            avg_lower = (bi.lower[v_idx_i] + bj.lower[v_idx_j]) / 2
                            avg_upper = (bi.upper[v_idx_i] + bj.upper[v_idx_j]) / 2
                            dc_correction += abs(lower[v] - avg_lower) + abs(upper[v] - avg_upper)
                            lower[v] = avg_lower
                            upper[v] = avg_upper

        upper = np.maximum(upper, lower)

        return AggregationResult(
            global_lower=lower,
            global_upper=upper,
            method=AggregationMethod.SUM,
            double_counting_correction=dc_correction,
            risk_decomposition=self.get_risk_decomposition(bounds),
        )

    def _aggregate_max(
        self,
        bounds: List[LocalBound],
        overlap_info: Optional[OverlapInfo],
    ) -> AggregationResult:
        """
        Max aggregation: take the maximum risk across subgraphs.

        L_d = max_i L_i,d
        U_d = max_i U_i,d

        Appropriate when global risk is dominated by the worst subgraph.
        """
        dim = max(len(b.lower) for b in bounds)
        lower = np.full(dim, -np.inf)
        upper = np.full(dim, -np.inf)

        for b in bounds:
            n = min(len(b.lower), dim)
            lower[:n] = np.maximum(lower[:n], b.lower[:n])
            upper[:n] = np.maximum(upper[:n], b.upper[:n])

        lower = np.where(np.isinf(lower), 0.0, lower)
        upper = np.where(np.isinf(upper), 1.0, upper)
        upper = np.maximum(upper, lower)

        return AggregationResult(
            global_lower=lower,
            global_upper=upper,
            method=AggregationMethod.MAX,
            double_counting_correction=0.0,
            risk_decomposition=self.get_risk_decomposition(bounds),
        )

    def _aggregate_weighted_sum(
        self,
        bounds: List[LocalBound],
        overlap_info: Optional[OverlapInfo],
    ) -> AggregationResult:
        """
        Weighted sum aggregation using inverse-width weighting.

        Tighter subgraph bounds get more weight, producing tighter
        aggregated bounds when some subgraphs are more precisely estimated.
        """
        dim = max(len(b.lower) for b in bounds)
        lower = np.zeros(dim)
        upper = np.zeros(dim)
        weight_sum = np.zeros(dim)

        for b in bounds:
            n = min(len(b.lower), dim)
            widths = (b.upper[:n] - b.lower[:n])
            inv_w = np.where(widths > 1e-12, 1.0 / widths, 1e6)
            w = inv_w * b.weight

            lower[:n] += w * b.lower[:n]
            upper[:n] += w * b.upper[:n]
            weight_sum[:n] += w

        safe_ws = np.maximum(weight_sum, 1e-12)
        lower /= safe_ws
        upper /= safe_ws
        upper = np.maximum(upper, lower)

        # Conservative expansion to ensure soundness
        expansion = np.zeros(dim)
        for b in bounds:
            n = min(len(b.lower), dim)
            expansion[:n] = np.maximum(
                expansion[:n],
                np.maximum(lower[:n] - b.lower[:n], 0) + np.maximum(b.upper[:n] - upper[:n], 0)
            )

        lower -= expansion
        upper += expansion

        dc_correction = float(np.sum(2 * expansion))

        return AggregationResult(
            global_lower=lower,
            global_upper=upper,
            method=AggregationMethod.WEIGHTED_SUM,
            double_counting_correction=dc_correction,
            risk_decomposition=self.get_risk_decomposition(bounds),
        )

    def _aggregate_inclusion_exclusion(
        self,
        bounds: List[LocalBound],
        overlap_info: Optional[OverlapInfo],
    ) -> AggregationResult:
        """
        Inclusion-exclusion aggregation: use the inclusion-exclusion principle
        to handle overlapping subgraphs exactly.

        R = sum_i R_i - sum_{i<j} R_{i∩j} + sum_{i<j<k} R_{i∩j∩k} - ...

        For computational tractability, we truncate at pairwise intersections.
        """
        dim = max(len(b.lower) for b in bounds)
        lower = np.zeros(dim)
        upper = np.zeros(dim)

        # First term: sum of individual contributions
        for b in bounds:
            n = min(len(b.lower), dim)
            lower[:n] += b.lower[:n]
            upper[:n] += b.upper[:n]

        dc_correction = 0.0

        # Second term: subtract pairwise intersections
        if overlap_info is not None:
            for (i, j), shared_vars in overlap_info.overlap_pairs.items():
                bi = next((b for b in bounds if b.subgraph_id == i), None)
                bj = next((b for b in bounds if b.subgraph_id == j), None)
                if bi is None or bj is None:
                    continue

                for v in shared_vars:
                    if v >= dim:
                        continue
                    v_idx_i = bi.variables.index(v) if v in bi.variables else -1
                    v_idx_j = bj.variables.index(v) if v in bj.variables else -1

                    if v_idx_i >= 0 and v_idx_j >= 0:
                        if v_idx_i < len(bi.lower) and v_idx_j < len(bj.lower):
                            intersection_lower = max(bi.lower[v_idx_i], bj.lower[v_idx_j])
                            intersection_upper = min(bi.upper[v_idx_i], bj.upper[v_idx_j])
                            intersection_upper = max(intersection_upper, intersection_lower)

                            lower[v] -= intersection_lower
                            upper[v] -= intersection_upper
                            dc_correction += abs(intersection_lower) + abs(intersection_upper)

        upper = np.maximum(upper, lower)

        return AggregationResult(
            global_lower=lower,
            global_upper=upper,
            method=AggregationMethod.INCLUSION_EXCLUSION,
            double_counting_correction=dc_correction,
            risk_decomposition=self.get_risk_decomposition(bounds),
        )

    def _aggregate_frechet(
        self,
        bounds: List[LocalBound],
        overlap_info: Optional[OverlapInfo],
    ) -> AggregationResult:
        """
        Fréchet bound aggregation: compute the tightest possible bounds
        without assumptions on the dependence structure between subgraphs.

        Uses the Fréchet-Hoeffding bounds:
            Lower: max(0, sum P_i - (n-1))
            Upper: min(1, min P_i)

        Applied to the cumulative risk distributions.
        """
        dim = max(len(b.lower) for b in bounds)
        K = len(bounds)

        # For each dimension, compute Fréchet bounds
        lower = np.zeros(dim)
        upper = np.ones(dim)

        for d in range(dim):
            lower_vals = []
            upper_vals = []
            for b in bounds:
                if d < len(b.lower):
                    lower_vals.append(b.lower[d])
                    upper_vals.append(b.upper[d])

            if lower_vals:
                # Fréchet lower: max(0, sum - (K-1))
                frechet_lower = max(0.0, sum(lower_vals) - (K - 1))
                # Fréchet upper: min individual uppers
                frechet_upper = min(upper_vals)
                frechet_upper = max(frechet_upper, frechet_lower)

                lower[d] = frechet_lower
                upper[d] = frechet_upper

        return AggregationResult(
            global_lower=lower,
            global_upper=upper,
            method=AggregationMethod.FRECHET,
            double_counting_correction=0.0,
            risk_decomposition=self.get_risk_decomposition(bounds),
        )

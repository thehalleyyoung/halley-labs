"""Mechanism sparsification for DP mechanisms.

Convert dense mechanism matrices into sparse equivalents by reducing
output support while preserving differential privacy guarantees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog

from dp_forge.types import (
    AdjacencyRelation,
    PrivacyBudget,
    QuerySpec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# L1 projection onto sparse simplex
# ---------------------------------------------------------------------------


class L1Projection:
    """Project onto the sparse probability simplex.

    Given a vector v, find the closest point p in the set
    {p >= 0, sum(p) = 1, |support(p)| <= s} under L1 distance.

    Attributes:
        max_support: Maximum support size.
    """

    def __init__(self, max_support: int) -> None:
        if max_support < 1:
            raise ValueError(f"max_support must be >= 1, got {max_support}")
        self.max_support = max_support

    def project(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project v onto the sparse simplex.

        Args:
            v: Input vector, shape (k,).

        Returns:
            Projected vector on the sparse simplex, shape (k,).
        """
        k = len(v)
        s = min(self.max_support, k)

        # Keep the s largest entries
        top_indices = np.argsort(v)[-s:]
        p = np.zeros(k, dtype=np.float64)
        p[top_indices] = np.maximum(v[top_indices], 0.0)

        # Project the retained entries onto the simplex
        p_sub = p[top_indices]
        p_sub = self._simplex_project(p_sub)
        p[top_indices] = p_sub

        return p

    @staticmethod
    def _simplex_project(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project v onto the probability simplex {p >= 0, sum(p) = 1}.

        Uses the efficient O(n log n) algorithm from
        Duchi et al. (2008) "Efficient Projections onto the l1-Ball".
        """
        n = len(v)
        if n == 0:
            return v.copy()

        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
        if len(rho) == 0:
            # All entries non-positive; uniform
            return np.ones(n, dtype=np.float64) / n
        rho_idx = rho[-1]
        theta = (cssv[rho_idx] - 1.0) / (rho_idx + 1.0)
        return np.maximum(v - theta, 0.0)

    def project_matrix(
        self, M: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Project each row of M onto the sparse simplex.

        Args:
            M: Mechanism matrix, shape (n, k).

        Returns:
            Sparsified mechanism matrix, shape (n, k).
        """
        result = np.zeros_like(M)
        for i in range(M.shape[0]):
            result[i] = self.project(M[i])
        return result


# ---------------------------------------------------------------------------
# Privacy-preserving rounding
# ---------------------------------------------------------------------------


class PrivacyPreservingRounding:
    """Round mechanism entries while maintaining the DP guarantee.

    After sparsification, probabilities may need rounding to a fixed
    precision. This class ensures rounding preserves e^eps-indistinguishability.

    Attributes:
        spec: Query specification.
        precision: Number of decimal digits to round to.
    """

    def __init__(self, spec: QuerySpec, precision: int = 6) -> None:
        self.spec = spec
        self.precision = precision

    def round(self, M: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Round mechanism entries preserving privacy.

        Uses a careful rounding scheme:
        1. Round entries to the given precision.
        2. Adjust to maintain row-stochasticity.
        3. Verify DP constraints; if violated, repair by redistribution.

        Args:
            M: Mechanism matrix, shape (n, k).

        Returns:
            Rounded mechanism matrix, shape (n, k).
        """
        n, k = M.shape
        M_rounded = np.round(M, self.precision)
        M_rounded = np.maximum(M_rounded, 0.0)

        # Enforce row-stochasticity
        for i in range(n):
            s = M_rounded[i].sum()
            if s > 0:
                M_rounded[i] /= s
            else:
                M_rounded[i] = 1.0 / k
            M_rounded[i] = np.round(M_rounded[i], self.precision)
            # Fix any residual from re-rounding
            deficit = 1.0 - M_rounded[i].sum()
            if abs(deficit) > 0:
                j_max = int(np.argmax(M_rounded[i]))
                M_rounded[i, j_max] += deficit

        # Repair DP violations
        M_rounded = self._repair_dp(M_rounded)
        return M_rounded

    def _repair_dp(self, M: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Repair DP violations caused by rounding.

        For each violated constraint M[i,j] > e^eps M[i',j], redistribute
        mass from bin j of row i to other bins.
        """
        e_eps = np.exp(self.spec.epsilon)
        n, k = M.shape
        assert self.spec.edges is not None
        edges = list(self.spec.edges.edges)

        for _ in range(20):
            any_violation = False
            for i, ip in edges:
                for j in range(k):
                    if M[i, j] > e_eps * M[ip, j] + 10.0 ** (-self.precision):
                        any_violation = True
                        excess = M[i, j] - e_eps * M[ip, j]
                        # Remove excess from M[i, j]
                        M[i, j] -= excess / 2.0
                        # Add to M[ip, j] to tighten
                        M[ip, j] += excess / (2.0 * e_eps)
                        # Re-normalise
                        for row in [i, ip]:
                            M[row] = np.maximum(M[row], 0.0)
                            s = M[row].sum()
                            if s > 0:
                                M[row] /= s

            if not any_violation:
                break

        return M


# ---------------------------------------------------------------------------
# Support reduction
# ---------------------------------------------------------------------------


class SupportReduction:
    """Iteratively remove small-probability outputs from a mechanism.

    Starting from a dense mechanism, identifies outputs with the smallest
    probabilities and removes them, redistributing mass to remaining
    outputs while preserving DP constraints.

    Attributes:
        spec: Query specification.
        min_support: Minimum number of outputs to keep per input.
        threshold: Probability threshold below which entries are removed.
    """

    def __init__(
        self,
        spec: QuerySpec,
        min_support: int = 2,
        threshold: float = 1e-4,
    ) -> None:
        self.spec = spec
        self.min_support = max(1, min_support)
        self.threshold = threshold

    def reduce(self, M: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Reduce the support of mechanism M.

        Args:
            M: Dense mechanism matrix, shape (n, k).

        Returns:
            Sparsified mechanism matrix.
        """
        n, k = M.shape
        M_sparse = M.copy()

        for i in range(n):
            row = M_sparse[i].copy()
            support = np.sum(row > self.threshold)

            while support > self.min_support:
                # Find smallest positive entry
                positive_mask = row > self.threshold
                if np.sum(positive_mask) <= self.min_support:
                    break

                min_idx = -1
                min_val = np.inf
                for j in range(k):
                    if positive_mask[j] and row[j] < min_val:
                        min_val = row[j]
                        min_idx = j

                if min_idx < 0:
                    break

                # Check if removing this entry preserves DP
                row_trial = row.copy()
                mass = row_trial[min_idx]
                row_trial[min_idx] = 0.0

                # Redistribute mass proportionally to remaining entries
                remaining = row_trial > self.threshold
                if not np.any(remaining):
                    break
                row_trial[remaining] += mass * row_trial[remaining] / row_trial[remaining].sum()
                row_trial /= row_trial.sum()

                # Verify DP for this row change
                if self._check_dp_row(M_sparse, i, row_trial):
                    row = row_trial
                    support = np.sum(row > self.threshold)
                else:
                    break

            M_sparse[i] = row

        return M_sparse

    def _check_dp_row(
        self,
        M: npt.NDArray[np.float64],
        changed_row: int,
        new_row: npt.NDArray[np.float64],
    ) -> bool:
        """Check if replacing row *changed_row* preserves DP."""
        e_eps = np.exp(self.spec.epsilon)
        assert self.spec.edges is not None

        for i, ip in self.spec.edges.edges:
            if i == changed_row:
                row_i = new_row
                row_ip = M[ip]
            elif ip == changed_row:
                row_i = M[i]
                row_ip = new_row
            else:
                continue

            for j in range(len(new_row)):
                if row_i[j] > e_eps * row_ip[j] + 1e-10:
                    return False
        return True


# ---------------------------------------------------------------------------
# Greedy sparsification
# ---------------------------------------------------------------------------


class GreedySparsification:
    """Greedy support selection for mechanism sparsification.

    Greedily selects which output bins to keep for each input, starting
    from an empty support and adding bins that give the greatest
    improvement in the objective while maintaining DP feasibility.

    Attributes:
        spec: Query specification.
        target_support: Target number of outputs per input.
    """

    def __init__(self, spec: QuerySpec, target_support: int = 10) -> None:
        self.spec = spec
        self.target_support = max(1, target_support)
        self.loss_matrix = self._build_loss_matrix(spec)

    def sparsify(self, M: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Greedily sparsify mechanism M.

        For each input, keeps only the *target_support* most important
        output bins and redistributes mass.

        Args:
            M: Dense mechanism matrix, shape (n, k).

        Returns:
            Sparsified mechanism matrix.
        """
        n, k = M.shape
        M_sparse = np.zeros((n, k), dtype=np.float64)

        for i in range(n):
            row = M[i].copy()
            # Score each bin: probability-weighted negative loss
            scores = row * (-self.loss_matrix[i])
            # Tie-break by probability
            scores += 1e-10 * row

            # Select top bins
            s = min(self.target_support, int(np.sum(row > 1e-12)))
            if s <= 0:
                s = 1
            top_bins = np.argsort(scores)[-s:]

            M_sparse[i, top_bins] = row[top_bins]
            total = M_sparse[i].sum()
            if total > 0:
                M_sparse[i] /= total
            else:
                M_sparse[i, top_bins[0]] = 1.0

        # Repair DP violations
        rounder = PrivacyPreservingRounding(self.spec)
        M_sparse = rounder._repair_dp(M_sparse)
        return M_sparse

    def sparsify_with_budget(
        self,
        M: npt.NDArray[np.float64],
        budget: PrivacyBudget,
    ) -> npt.NDArray[np.float64]:
        """Sparsify with explicit privacy budget.

        Uses LP to find the best sparse mechanism with given support size.

        Args:
            M: Dense mechanism matrix, shape (n, k).
            budget: Privacy budget.

        Returns:
            Sparsified mechanism matrix.
        """
        n, k = M.shape
        e_eps = np.exp(budget.epsilon)
        M_sparse = np.zeros((n, k), dtype=np.float64)

        for i in range(n):
            # Select support from dense mechanism
            s = min(self.target_support, k)
            top_bins = np.argsort(M[i])[-s:]

            # Solve LP for optimal distribution on this support
            c_sub = self.loss_matrix[i, top_bins]
            A_eq = np.ones((1, s), dtype=np.float64)
            b_eq = np.array([1.0])
            bounds = [(0.0, 1.0)] * s

            res = linprog(c_sub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            if res.success:
                M_sparse[i, top_bins] = np.maximum(res.x, 0.0)
                M_sparse[i] /= M_sparse[i].sum()
            else:
                M_sparse[i, top_bins] = 1.0 / s

        # Repair DP violations
        rounder = PrivacyPreservingRounding(self.spec)
        M_sparse = rounder._repair_dp(M_sparse)
        return M_sparse

    @staticmethod
    def _build_loss_matrix(spec: QuerySpec) -> npt.NDArray[np.float64]:
        """Build (n × k) loss matrix."""
        n, k = spec.n, spec.k
        y_min = float(spec.query_values.min()) - spec.sensitivity
        y_max = float(spec.query_values.max()) + spec.sensitivity
        y_grid = np.linspace(y_min, y_max, k)
        loss_fn = spec.get_loss_callable()
        L = np.zeros((n, k), dtype=np.float64)
        for i in range(n):
            for j in range(k):
                L[i, j] = loss_fn(spec.query_values[i], y_grid[j])
        return L


# ---------------------------------------------------------------------------
# Main sparsifier
# ---------------------------------------------------------------------------


class MechanismSparsifier:
    """Convert a dense mechanism to a sparse equivalent.

    Orchestrates multiple sparsification strategies (support reduction,
    greedy, L1 projection) and picks the best result.

    Attributes:
        spec: Query specification.
        target_support: Target support size per input.
        strategies: List of sparsification strategies to try.
    """

    def __init__(
        self,
        spec: QuerySpec,
        target_support: int = 10,
        strategies: Optional[List[str]] = None,
    ) -> None:
        self.spec = spec
        self.target_support = max(1, target_support)
        self.strategies = strategies or ["greedy", "l1", "support_reduction"]

    def sparsify(self, M: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Sparsify a dense mechanism using the best available strategy.

        Tries all configured strategies and returns the one with
        the lowest expected loss that satisfies DP constraints.

        Args:
            M: Dense mechanism matrix, shape (n, k).

        Returns:
            Best sparsified mechanism matrix.
        """
        candidates: List[Tuple[str, npt.NDArray[np.float64], float]] = []

        for strategy in self.strategies:
            try:
                M_sparse = self._apply_strategy(strategy, M)
                loss = self._evaluate_loss(M_sparse)
                if self._verify_dp(M_sparse):
                    candidates.append((strategy, M_sparse, loss))
            except Exception as e:
                logger.warning("Strategy %s failed: %s", strategy, e)

        if not candidates:
            logger.warning("All strategies failed; returning original mechanism.")
            return M.copy()

        # Pick best
        candidates.sort(key=lambda x: x[2])
        best_strategy, best_M, best_loss = candidates[0]
        logger.info(
            "Best sparsification: %s (loss=%.6f, support=%d)",
            best_strategy,
            best_loss,
            self._count_support(best_M),
        )
        return best_M

    def _apply_strategy(
        self, strategy: str, M: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Apply a specific sparsification strategy."""
        if strategy == "greedy":
            gs = GreedySparsification(self.spec, self.target_support)
            return gs.sparsify(M)
        elif strategy == "l1":
            proj = L1Projection(self.target_support)
            M_sparse = proj.project_matrix(M)
            rounder = PrivacyPreservingRounding(self.spec)
            return rounder._repair_dp(M_sparse)
        elif strategy == "support_reduction":
            sr = SupportReduction(self.spec, min_support=self.target_support)
            return sr.reduce(M)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _evaluate_loss(self, M: npt.NDArray[np.float64]) -> float:
        """Evaluate expected loss of a mechanism."""
        loss_matrix = GreedySparsification._build_loss_matrix(self.spec)
        return float(np.sum(M * loss_matrix)) / self.spec.n

    def _verify_dp(self, M: npt.NDArray[np.float64]) -> bool:
        """Verify that M satisfies DP constraints."""
        e_eps = np.exp(self.spec.epsilon)
        assert self.spec.edges is not None
        for i, ip in self.spec.edges.edges:
            for j in range(M.shape[1]):
                if M[i, j] > e_eps * M[ip, j] + 1e-8:
                    return False
        return True

    def _count_support(self, M: npt.NDArray[np.float64]) -> int:
        """Count total number of nonzero entries."""
        return int(np.count_nonzero(M > 1e-10))

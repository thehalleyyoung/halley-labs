"""Lagrangian relaxation for DP mechanism synthesis.

Relaxes privacy constraints into the objective with Lagrange multipliers
and iteratively tightens the bound via subgradient, bundle, or augmented
Lagrangian methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse as sp_sparse
from scipy.optimize import linprog

from dp_forge.sparse import (
    DecompositionType,
    LagrangianState,
    MultiplierUpdate,
    SparseConfig,
    SparseResult,
)
from dp_forge.types import (
    AdjacencyRelation,
    OptimalityCertificate,
    PrivacyBudget,
    QuerySpec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _get_edge_list(spec: QuerySpec) -> List[Tuple[int, int]]:
    """Return the list of adjacent pairs from the spec."""
    assert spec.edges is not None
    return list(spec.edges.edges)


# ---------------------------------------------------------------------------
# Duality gap monitor
# ---------------------------------------------------------------------------


class DualityGapMonitor:
    """Track and bound the duality gap during Lagrangian relaxation.

    Records lower bounds (Lagrangian bound) and upper bounds (feasible
    heuristic solutions) at each iteration and provides convergence
    diagnostics.

    Attributes:
        lower_bounds: History of Lagrangian lower bounds.
        upper_bounds: History of feasible upper bounds.
        tol: Convergence tolerance on relative gap.
    """

    def __init__(self, tol: float = 1e-6) -> None:
        self.lower_bounds: List[float] = []
        self.upper_bounds: List[float] = []
        self.tol = tol

    @property
    def best_lower(self) -> float:
        """Best (highest) Lagrangian lower bound."""
        return max(self.lower_bounds) if self.lower_bounds else -np.inf

    @property
    def best_upper(self) -> float:
        """Best (lowest) feasible upper bound."""
        return min(self.upper_bounds) if self.upper_bounds else np.inf

    @property
    def absolute_gap(self) -> float:
        """Absolute duality gap."""
        return self.best_upper - self.best_lower

    @property
    def relative_gap(self) -> float:
        """Relative duality gap."""
        denom = max(abs(self.best_upper), 1.0)
        return self.absolute_gap / denom

    @property
    def converged(self) -> bool:
        """Whether the gap is within tolerance."""
        return self.relative_gap <= self.tol

    def record(self, lower: float, upper: float) -> None:
        """Record bounds from the current iteration."""
        self.lower_bounds.append(lower)
        self.upper_bounds.append(upper)

    def summary(self) -> str:
        """Summary string for logging."""
        return (
            f"Gap: abs={self.absolute_gap:.2e}, rel={self.relative_gap:.2e}, "
            f"LB={self.best_lower:.6f}, UB={self.best_upper:.6f}"
        )

    def __repr__(self) -> str:
        return f"DualityGapMonitor(gap={self.relative_gap:.2e}, iters={len(self.lower_bounds)})"


# ---------------------------------------------------------------------------
# Lagrangian relaxation core
# ---------------------------------------------------------------------------


class LagrangianRelaxation:
    """Relax privacy constraints into the objective with Lagrange multipliers.

    Given mechanism LP: min L(M) s.t. M[i,j] <= e^eps M[i',j]
    The Lagrangian is: L(M) + sum_lam lam_{i,i',j} (M[i,j] - e^eps M[i',j])

    For fixed multipliers, the Lagrangian subproblem decomposes by input
    and can be solved as n independent row-optimisation problems.

    Attributes:
        spec: Query specification.
        loss_matrix: Precomputed (n × k) loss matrix.
        edges: List of adjacent pairs.
        n_multipliers: Number of Lagrange multipliers.
    """

    def __init__(self, spec: QuerySpec) -> None:
        self.spec = spec
        self.loss_matrix = _build_loss_matrix(spec)
        self.edges = _get_edge_list(spec)
        k = spec.k
        self.n_multipliers = len(self.edges) * k

    def evaluate(
        self,
        multipliers: npt.NDArray[np.float64],
    ) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Evaluate the Lagrangian dual function at given multipliers.

        Solves the Lagrangian subproblem:
            min_{M row-stochastic} L(M) + lam^T g(M)
        where g(M) captures the DP constraint violations.

        Args:
            multipliers: Non-negative Lagrange multipliers, shape (n_multipliers,).

        Returns:
            Tuple of (lagrangian_value, optimal_M, subgradient).
        """
        n, k = self.spec.n, self.spec.k
        e_eps = np.exp(self.spec.epsilon)

        # Build augmented cost for each M[i, j]
        aug_cost = self.loss_matrix.copy()

        for edge_idx, (i, ip) in enumerate(self.edges):
            lam_block = multipliers[edge_idx * k: (edge_idx + 1) * k]
            aug_cost[i] += lam_block
            aug_cost[ip] -= e_eps * lam_block

        # Solve: for each input i, minimise aug_cost[i] @ p s.t. p in simplex
        M = np.zeros((n, k), dtype=np.float64)
        lag_value = 0.0

        for i in range(n):
            # Optimal is to put all mass on the minimum-cost bin
            j_star = int(np.argmin(aug_cost[i]))
            M[i, j_star] = 1.0
            lag_value += aug_cost[i, j_star]

        # Compute subgradient: g(M) = M[i,j] - e^eps M[i',j]
        subgradient = np.zeros(self.n_multipliers, dtype=np.float64)
        for edge_idx, (i, ip) in enumerate(self.edges):
            for j in range(k):
                subgradient[edge_idx * k + j] = M[i, j] - e_eps * M[ip, j]

        return lag_value, M, subgradient

    def constraint_violation(
        self, M: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute DP constraint violations for mechanism M.

        Args:
            M: Mechanism matrix, shape (n, k).

        Returns:
            Violation vector, shape (n_multipliers,).
        """
        k = self.spec.k
        e_eps = np.exp(self.spec.epsilon)
        violations = np.zeros(self.n_multipliers, dtype=np.float64)
        for edge_idx, (i, ip) in enumerate(self.edges):
            for j in range(k):
                violations[edge_idx * k + j] = M[i, j] - e_eps * M[ip, j]
        return violations


# ---------------------------------------------------------------------------
# Subgradient optimiser
# ---------------------------------------------------------------------------


class SubgradientOptimizer:
    """Subgradient method for the Lagrangian dual.

    Updates multipliers using the Polyak step size rule:
        step = alpha * (UB - L(lam)) / ||g||^2

    where UB is the best known feasible upper bound, L(lam) is the
    current Lagrangian value, and g is the subgradient.

    Attributes:
        relaxation: The Lagrangian relaxation instance.
        config: Sparse configuration.
        gap_monitor: Duality gap monitor.
    """

    def __init__(
        self,
        relaxation: LagrangianRelaxation,
        config: Optional[SparseConfig] = None,
    ) -> None:
        self.relaxation = relaxation
        self.config = config or SparseConfig(
            decomposition_type=DecompositionType.LAGRANGIAN,
            multiplier_update=MultiplierUpdate.SUBGRADIENT,
        )
        self.gap_monitor = DualityGapMonitor(tol=self.config.convergence_tol)
        self._state: Optional[LagrangianState] = None

    def solve(
        self, initial_ub: float = np.inf
    ) -> Tuple[LagrangianState, npt.NDArray[np.float64]]:
        """Run the subgradient method.

        Args:
            initial_ub: Initial upper bound (from a feasible heuristic).

        Returns:
            Tuple of (final LagrangianState, best mechanism).
        """
        n_mult = self.relaxation.n_multipliers
        multipliers = np.zeros(n_mult, dtype=np.float64)
        alpha = self.config.step_size_init
        best_lb = -np.inf
        best_ub = initial_ub
        best_mechanism = None
        no_improve_count = 0

        for iteration in range(self.config.max_iterations):
            # Evaluate Lagrangian
            lag_val, M, subgradient = self.relaxation.evaluate(multipliers)

            # Lagrangian heuristic: project M to feasibility
            M_feasible = LagrangianHeuristic(self.relaxation.spec).recover(M)
            ub = float(self.relaxation.loss_matrix.ravel() @ M_feasible.ravel())

            # Update bounds
            if lag_val > best_lb:
                best_lb = lag_val
                no_improve_count = 0
            else:
                no_improve_count += 1

            if ub < best_ub:
                best_ub = ub
                best_mechanism = M_feasible.copy()

            self.gap_monitor.record(lag_val, ub)

            # Polyak step size
            sg_norm_sq = float(np.dot(subgradient, subgradient))
            if sg_norm_sq < 1e-20:
                break

            target = best_ub
            step = alpha * (target - lag_val) / sg_norm_sq

            # Update multipliers (project to non-negative)
            multipliers = np.maximum(multipliers + step * subgradient, 0.0)

            # Halve alpha if no improvement for 20 iterations
            if no_improve_count >= 20:
                alpha *= 0.5
                no_improve_count = 0

            self._state = LagrangianState(
                multipliers=multipliers.copy(),
                lower_bound=best_lb,
                upper_bound=best_ub,
                step_size=step,
                iteration=iteration + 1,
                subgradient=subgradient.copy(),
            )

            if self.config.verbose >= 2:
                logger.info(
                    "Subgradient iter %d: LB=%.6f, UB=%.6f, gap=%.2e",
                    iteration, best_lb, best_ub, self.gap_monitor.relative_gap,
                )

            if self.gap_monitor.converged:
                break

        if best_mechanism is None:
            n, k = self.relaxation.spec.n, self.relaxation.spec.k
            best_mechanism = np.ones((n, k), dtype=np.float64) / k

        return self._state or LagrangianState(
            multipliers=multipliers,
            lower_bound=best_lb,
            upper_bound=best_ub,
            step_size=0.0,
        ), best_mechanism

    @property
    def state(self) -> Optional[LagrangianState]:
        """Current Lagrangian state."""
        return self._state


# ---------------------------------------------------------------------------
# Bundle method
# ---------------------------------------------------------------------------


class BundleMethod:
    """Bundle method with polyhedral model for Lagrangian dual.

    Maintains a bundle of past subgradients and function values to build
    a piecewise-linear model of the dual function. Solves a stabilised
    QP at each iteration.

    Attributes:
        relaxation: The Lagrangian relaxation instance.
        config: Sparse configuration.
        bundle: List of (multipliers, function_value, subgradient) tuples.
        gap_monitor: Duality gap monitor.
    """

    def __init__(
        self,
        relaxation: LagrangianRelaxation,
        config: Optional[SparseConfig] = None,
        bundle_size: int = 50,
        proximal_weight: float = 1.0,
    ) -> None:
        self.relaxation = relaxation
        self.config = config or SparseConfig(
            decomposition_type=DecompositionType.LAGRANGIAN,
            multiplier_update=MultiplierUpdate.BUNDLE,
        )
        self.bundle: List[Tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]]] = []
        self.bundle_size = bundle_size
        self.proximal_weight = proximal_weight
        self.gap_monitor = DualityGapMonitor(tol=self.config.convergence_tol)
        self._state: Optional[LagrangianState] = None

    def solve(
        self, initial_ub: float = np.inf
    ) -> Tuple[LagrangianState, npt.NDArray[np.float64]]:
        """Run the bundle method.

        Args:
            initial_ub: Initial upper bound.

        Returns:
            Tuple of (final LagrangianState, best mechanism).
        """
        n_mult = self.relaxation.n_multipliers
        center = np.zeros(n_mult, dtype=np.float64)
        best_lb = -np.inf
        best_ub = initial_ub
        best_mechanism = None

        for iteration in range(self.config.max_iterations):
            # Evaluate at current center
            lag_val, M, subgradient = self.relaxation.evaluate(center)

            # Add to bundle
            self.bundle.append((center.copy(), lag_val, subgradient.copy()))
            if len(self.bundle) > self.bundle_size:
                self.bundle.pop(0)

            # Heuristic recovery
            M_feasible = LagrangianHeuristic(self.relaxation.spec).recover(M)
            ub = float(self.relaxation.loss_matrix.ravel() @ M_feasible.ravel())

            if lag_val > best_lb:
                best_lb = lag_val
            if ub < best_ub:
                best_ub = ub
                best_mechanism = M_feasible.copy()

            self.gap_monitor.record(lag_val, ub)

            if self.gap_monitor.converged:
                break

            # Solve bundle subproblem via cutting-plane approximation
            # max_lam min_i { f_i + g_i^T (lam - lam_i) } - mu/2 ||lam - center||^2
            # Approximation: steepest ascent on best subgradient
            next_center = self._solve_bundle_subproblem(center)
            center = next_center

            self._state = LagrangianState(
                multipliers=center.copy(),
                lower_bound=best_lb,
                upper_bound=best_ub,
                step_size=float(np.linalg.norm(center)),
                iteration=iteration + 1,
                subgradient=subgradient.copy(),
            )

            if self.config.verbose >= 2:
                logger.info(
                    "Bundle iter %d: LB=%.6f, UB=%.6f, gap=%.2e",
                    iteration, best_lb, best_ub, self.gap_monitor.relative_gap,
                )

        if best_mechanism is None:
            n, k = self.relaxation.spec.n, self.relaxation.spec.k
            best_mechanism = np.ones((n, k), dtype=np.float64) / k

        return self._state or LagrangianState(
            multipliers=center,
            lower_bound=best_lb,
            upper_bound=best_ub,
            step_size=0.0,
        ), best_mechanism

    def _solve_bundle_subproblem(
        self, center: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Solve the bundle QP approximately.

        Uses a simple approach: take a step in the direction of the
        aggregate subgradient, with proximal stabilisation.
        """
        if not self.bundle:
            return center.copy()

        # Aggregate subgradient: weighted average of recent subgradients
        n_recent = min(5, len(self.bundle))
        agg_subgradient = np.zeros_like(center)
        for _, fval, sg in self.bundle[-n_recent:]:
            agg_subgradient += sg
        agg_subgradient /= n_recent

        # Step with proximal regularisation
        step = 1.0 / self.proximal_weight
        new_center = center + step * agg_subgradient
        # Project to non-negative
        new_center = np.maximum(new_center, 0.0)
        return new_center

    @property
    def state(self) -> Optional[LagrangianState]:
        """Current Lagrangian state."""
        return self._state


# ---------------------------------------------------------------------------
# Lagrangian heuristic
# ---------------------------------------------------------------------------


class LagrangianHeuristic:
    """Primal recovery from Lagrangian solutions.

    The Lagrangian subproblem solution is typically infeasible for the
    original problem (violates DP constraints). This heuristic projects
    the solution back to feasibility while trying to maintain objective
    quality.

    Attributes:
        spec: Query specification.
    """

    def __init__(self, spec: QuerySpec) -> None:
        self.spec = spec

    def recover(self, M: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Recover a feasible mechanism from Lagrangian solution M.

        Uses iterative projection: alternately enforce row-stochasticity
        and DP constraints until convergence.

        Args:
            M: Mechanism from Lagrangian subproblem, shape (n, k).

        Returns:
            Feasible mechanism matrix, shape (n, k).
        """
        n, k = self.spec.n, self.spec.k
        e_eps = np.exp(self.spec.epsilon)
        M_feas = M.copy()

        assert self.spec.edges is not None
        edges = list(self.spec.edges.edges)

        for _ in range(50):
            # Enforce DP constraints by averaging violations
            for i_edge, ip in edges:
                for j in range(k):
                    if M_feas[i_edge, j] > e_eps * M_feas[ip, j]:
                        avg = (M_feas[i_edge, j] + e_eps * M_feas[ip, j]) / (1.0 + e_eps)
                        M_feas[i_edge, j] = avg
                        M_feas[ip, j] = avg / e_eps

            # Enforce non-negativity
            M_feas = np.maximum(M_feas, 0.0)

            # Enforce row-stochasticity
            for i in range(n):
                s = M_feas[i].sum()
                if s > 0:
                    M_feas[i] /= s
                else:
                    M_feas[i] = 1.0 / k

            # Check feasibility
            max_violation = 0.0
            for i_edge, ip in edges:
                for j in range(k):
                    v = M_feas[i_edge, j] - e_eps * M_feas[ip, j]
                    max_violation = max(max_violation, v)
            if max_violation <= 1e-8:
                break

        return M_feas

    def recover_convex_combination(
        self,
        mechanisms: List[npt.NDArray[np.float64]],
        weights: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """Recover a feasible mechanism from a convex combination.

        Takes a weighted average of past Lagrangian solutions and projects
        to feasibility.

        Args:
            mechanisms: List of mechanism matrices.
            weights: Convex combination weights (uniform if None).

        Returns:
            Feasible mechanism matrix.
        """
        if not mechanisms:
            n, k = self.spec.n, self.spec.k
            return np.ones((n, k), dtype=np.float64) / k

        if weights is None:
            weights = np.ones(len(mechanisms), dtype=np.float64) / len(mechanisms)

        M_avg = np.zeros_like(mechanisms[0])
        for w, M in zip(weights, mechanisms):
            M_avg += w * M

        return self.recover(M_avg)


# ---------------------------------------------------------------------------
# Augmented Lagrangian
# ---------------------------------------------------------------------------


class AugmentedLagrangian:
    """Augmented Lagrangian method with penalty terms.

    Adds a quadratic penalty for constraint violations to the Lagrangian,
    providing better convergence properties than the standard Lagrangian
    relaxation. Updates multipliers using the method of multipliers
    (alternating direction).

    Attributes:
        spec: Query specification.
        relaxation: Base Lagrangian relaxation.
        penalty: Current penalty parameter (rho).
        gap_monitor: Duality gap monitor.
    """

    def __init__(
        self,
        spec: QuerySpec,
        config: Optional[SparseConfig] = None,
        initial_penalty: float = 1.0,
        penalty_growth: float = 2.0,
    ) -> None:
        self.spec = spec
        self.config = config or SparseConfig(
            decomposition_type=DecompositionType.LAGRANGIAN,
        )
        self.relaxation = LagrangianRelaxation(spec)
        self.penalty = initial_penalty
        self.penalty_growth = penalty_growth
        self.gap_monitor = DualityGapMonitor(tol=self.config.convergence_tol)
        self._state: Optional[LagrangianState] = None

    def solve(
        self, initial_ub: float = np.inf
    ) -> Tuple[LagrangianState, npt.NDArray[np.float64]]:
        """Run the augmented Lagrangian method.

        Args:
            initial_ub: Initial feasible upper bound.

        Returns:
            Tuple of (final LagrangianState, best mechanism).
        """
        n, k = self.spec.n, self.spec.k
        n_mult = self.relaxation.n_multipliers
        multipliers = np.zeros(n_mult, dtype=np.float64)
        best_lb = -np.inf
        best_ub = initial_ub
        best_mechanism = None

        M = np.ones((n, k), dtype=np.float64) / k  # initial mechanism

        for iteration in range(self.config.max_iterations):
            # Inner minimisation: min L(M) + lam^T g(M) + (rho/2)||g(M)+||^2
            M = self._inner_minimisation(multipliers, M)

            # Compute constraint violations
            violations = self.relaxation.constraint_violation(M)

            # Evaluate objective
            obj = float(self.relaxation.loss_matrix.ravel() @ M.ravel())

            # Lagrangian bound
            lag_val = obj + float(multipliers @ violations)
            if lag_val > best_lb:
                best_lb = lag_val

            # Project to feasibility for upper bound
            heuristic = LagrangianHeuristic(self.spec)
            M_feasible = heuristic.recover(M)
            ub = float(self.relaxation.loss_matrix.ravel() @ M_feasible.ravel())
            if ub < best_ub:
                best_ub = ub
                best_mechanism = M_feasible.copy()

            self.gap_monitor.record(lag_val, ub)

            # Update multipliers: lam <- max(0, lam + rho * g(M))
            multipliers = np.maximum(
                multipliers + self.penalty * violations, 0.0
            )

            # Increase penalty if violations remain large
            max_violation = float(np.max(np.maximum(violations, 0.0)))
            if max_violation > 1e-4:
                self.penalty *= self.penalty_growth

            self._state = LagrangianState(
                multipliers=multipliers.copy(),
                lower_bound=best_lb,
                upper_bound=best_ub,
                step_size=self.penalty,
                iteration=iteration + 1,
                subgradient=violations.copy(),
            )

            if self.config.verbose >= 2:
                logger.info(
                    "AugLag iter %d: obj=%.6f, max_viol=%.2e, penalty=%.2f",
                    iteration, obj, max_violation, self.penalty,
                )

            if self.gap_monitor.converged:
                break

        if best_mechanism is None:
            best_mechanism = np.ones((n, k), dtype=np.float64) / k

        return self._state or LagrangianState(
            multipliers=multipliers,
            lower_bound=best_lb,
            upper_bound=best_ub,
            step_size=self.penalty,
        ), best_mechanism

    def _inner_minimisation(
        self,
        multipliers: npt.NDArray[np.float64],
        M_init: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Approximately minimise the augmented Lagrangian.

        Uses projected gradient descent on M with the augmented objective.

        Args:
            multipliers: Current Lagrange multipliers.
            M_init: Initial mechanism.

        Returns:
            Updated mechanism matrix.
        """
        n, k = self.spec.n, self.spec.k
        e_eps = np.exp(self.spec.epsilon)
        rho = self.penalty
        edges = self.relaxation.edges
        M = M_init.copy()
        lr = 0.01 / (1.0 + rho)

        for _ in range(30):
            # Gradient of: L(M) + lam^T g(M) + (rho/2) sum max(0, g(M))^2
            grad = self.relaxation.loss_matrix.copy()
            violations = self.relaxation.constraint_violation(M)

            for edge_idx, (i, ip) in enumerate(edges):
                for j in range(k):
                    idx = edge_idx * k + j
                    lam = multipliers[idx]
                    v = violations[idx]
                    penalty_grad = rho * max(0.0, v + lam / rho)

                    grad[i, j] += lam + penalty_grad
                    grad[ip, j] -= e_eps * (lam + penalty_grad)

            # Projected gradient step
            M -= lr * grad
            M = np.maximum(M, 0.0)
            for i in range(n):
                s = M[i].sum()
                if s > 0:
                    M[i] /= s
                else:
                    M[i] = 1.0 / k

        return M

    @property
    def state(self) -> Optional[LagrangianState]:
        """Current Lagrangian state."""
        return self._state

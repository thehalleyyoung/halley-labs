"""
Budget allocation optimisation under RDP composition constraints.

Given T queries with known error-vs-epsilon relationships, finds the
per-query budget allocation that minimises total error subject to the
constraint that the composed RDP guarantee stays within a given budget.

Three allocation strategies:
    - **Uniform**: Equal ε per query (baseline).
    - **Proportional**: ε proportional to query sensitivity or weight.
    - **Convex**: Full convex optimisation via projected gradient descent.

The convex formulation exploits the fact that for many mechanisms
(Gaussian, Laplace) the error-vs-epsilon curve is convex, making the
total-error minimisation a convex program.

References:
    - Mironov, I. (2017). Rényi differential privacy.
    - Dong, J., Roth, A., & Su, W.J. (2022). Gaussian differential
      privacy.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import (
    ConfigurationError,
    ConvergenceError,
)
from dp_forge.types import PrivacyBudget

from dp_forge.rdp.accountant import RDPAccountant, RDPCurve, DEFAULT_ALPHAS
from dp_forge.rdp.conversion import rdp_to_dp

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]

# Error function type: maps (epsilon, sensitivity) → error
ErrorFunction = Callable[[float, float], float]


# =========================================================================
# Allocation result
# =========================================================================


@dataclass
class RDPAllocationResult:
    """Result of an RDP budget allocation optimisation.

    Attributes:
        epsilons: Per-query epsilon allocations, shape ``(T,)``.
        total_error: Total error across all queries at the allocation.
        composed_epsilon: Composed (ε, δ)-DP epsilon at the allocation.
        composed_delta: The δ used for composition.
        per_query_errors: Error for each query at its allocation.
        n_iterations: Number of optimisation iterations (for convex).
        converged: Whether the optimiser converged.
        metadata: Additional optimisation metadata.
    """

    epsilons: FloatArray
    total_error: float
    composed_epsilon: float
    composed_delta: float
    per_query_errors: FloatArray = field(default_factory=lambda: np.array([]))
    n_iterations: int = 0
    converged: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.epsilons = np.asarray(self.epsilons, dtype=np.float64)
        self.per_query_errors = np.asarray(self.per_query_errors, dtype=np.float64)

    @property
    def n_queries(self) -> int:
        """Number of queries."""
        return len(self.epsilons)

    def __repr__(self) -> str:
        return (
            f"RDPAllocationResult(n={self.n_queries}, "
            f"total_error={self.total_error:.6f}, "
            f"ε_composed={self.composed_epsilon:.4f})"
        )


# =========================================================================
# RDPBudgetOptimizer
# =========================================================================


class RDPBudgetOptimizer:
    """Optimise per-query privacy budget allocation under RDP composition.

    Given T queries, each with a sensitivity and an error-vs-epsilon
    function, finds the allocation (ε₁, ..., ε_T) minimising total error
    subject to the constraint that the RDP-composed privacy guarantee
    stays within a target budget.

    The RDP composition constraint is:
        For all α: Σ_t ε̂_t(α) ≤ ε̂_budget(α)

    where ε̂_t(α) is the RDP of query t at its allocated epsilon.

    Args:
        target_budget: Total privacy budget (ε, δ).
        alphas: α grid for RDP accounting. Defaults to :data:`DEFAULT_ALPHAS`.
        mechanism_type: Type of mechanism for error computation.
            ``"gaussian"`` or ``"laplace"``.

    Example::

        optimizer = RDPBudgetOptimizer(
            target_budget=PrivacyBudget(1.0, 1e-5),
        )
        result = optimizer.optimize_convex(
            sensitivities=[1.0, 1.0, 2.0],
            error_fns=[lambda e, s: s**2 / (2*e**2)] * 3,
        )
    """

    def __init__(
        self,
        target_budget: PrivacyBudget,
        alphas: Optional[FloatArray] = None,
        mechanism_type: str = "gaussian",
    ) -> None:
        if target_budget.epsilon <= 0:
            raise ConfigurationError(
                f"target_budget.epsilon must be > 0, got {target_budget.epsilon}",
                parameter="target_budget",
            )

        self._target_budget = target_budget
        self._alphas = (
            np.asarray(alphas, dtype=np.float64)
            if alphas is not None
            else DEFAULT_ALPHAS.copy()
        )
        self._mechanism_type = mechanism_type.lower()

    @property
    def target_budget(self) -> PrivacyBudget:
        """The target total privacy budget."""
        return self._target_budget

    # -----------------------------------------------------------------
    # Uniform allocation
    # -----------------------------------------------------------------

    def optimize_uniform(
        self,
        sensitivities: Sequence[float],
        error_fns: Optional[Sequence[ErrorFunction]] = None,
    ) -> RDPAllocationResult:
        """Uniform budget allocation: equal ε per query.

        Distributes the total budget equally across all queries.
        Uses RDP composition to determine the per-query ε that achieves
        the total budget.

        Args:
            sensitivities: Per-query sensitivities, length T.
            error_fns: Per-query error functions mapping (ε, Δ) → error.
                If ``None``, uses the default for the mechanism type.

        Returns:
            Allocation result.
        """
        T = len(sensitivities)
        if T == 0:
            raise ConfigurationError(
                "At least one query is required",
                parameter="sensitivities",
            )

        sens = np.asarray(sensitivities, dtype=np.float64)
        error_fns = error_fns or [self._default_error_fn()] * T

        # Binary search for per-query ε that yields total budget
        per_eps = self._find_uniform_epsilon(T, sens, self._target_budget)

        epsilons = np.full(T, per_eps)
        per_errors = np.array(
            [fn(per_eps, float(s)) for fn, s in zip(error_fns, sens)],
            dtype=np.float64,
        )
        total_error = float(np.sum(per_errors))

        # Verify composed budget
        composed = self._compute_composed_dp(epsilons, sens)

        return RDPAllocationResult(
            epsilons=epsilons,
            total_error=total_error,
            composed_epsilon=composed.epsilon,
            composed_delta=composed.delta,
            per_query_errors=per_errors,
            metadata={"method": "uniform"},
        )

    def _find_uniform_epsilon(
        self,
        T: int,
        sensitivities: FloatArray,
        budget: PrivacyBudget,
    ) -> float:
        """Binary search for per-query ε yielding the target composed budget."""
        lo, hi = 1e-10, 100.0
        target_eps = budget.epsilon
        delta = budget.delta if budget.delta > 0 else 1e-10

        for _ in range(100):
            mid = (lo + hi) / 2.0
            eps_arr = np.full(len(sensitivities), mid)
            composed = self._compute_composed_dp(eps_arr, sensitivities, delta)
            if composed.epsilon < target_eps:
                lo = mid
            else:
                hi = mid

            if hi - lo < 1e-12:
                break

        return (lo + hi) / 2.0

    # -----------------------------------------------------------------
    # Proportional allocation
    # -----------------------------------------------------------------

    def optimize_proportional(
        self,
        sensitivities: Sequence[float],
        weights: Optional[Sequence[float]] = None,
        error_fns: Optional[Sequence[ErrorFunction]] = None,
    ) -> RDPAllocationResult:
        """Proportional budget allocation: ε_t ∝ w_t.

        Distributes budget proportionally to provided weights (defaulting
        to query sensitivities). A scaling factor is found via binary
        search to satisfy the RDP composition constraint.

        Args:
            sensitivities: Per-query sensitivities.
            weights: Per-query weights for allocation proportionality.
                Defaults to sensitivities.
            error_fns: Per-query error functions.

        Returns:
            Allocation result.
        """
        T = len(sensitivities)
        if T == 0:
            raise ConfigurationError(
                "At least one query is required",
                parameter="sensitivities",
            )

        sens = np.asarray(sensitivities, dtype=np.float64)
        w = np.asarray(weights if weights is not None else sensitivities, dtype=np.float64)
        error_fns = error_fns or [self._default_error_fn()] * T

        if np.any(w <= 0):
            raise ConfigurationError(
                "All weights must be positive",
                parameter="weights",
                value=float(np.min(w)),
            )

        # Normalise weights
        w_norm = w / np.sum(w)

        # Binary search for scale factor c such that ε_t = c * w_t/sum(w)
        scale = self._find_proportional_scale(w_norm, sens, self._target_budget)
        epsilons = scale * w_norm

        per_errors = np.array(
            [fn(float(e), float(s)) for fn, e, s in zip(error_fns, epsilons, sens)],
            dtype=np.float64,
        )
        total_error = float(np.sum(per_errors))

        composed = self._compute_composed_dp(epsilons, sens)

        return RDPAllocationResult(
            epsilons=epsilons,
            total_error=total_error,
            composed_epsilon=composed.epsilon,
            composed_delta=composed.delta,
            per_query_errors=per_errors,
            metadata={"method": "proportional", "scale": scale},
        )

    def _find_proportional_scale(
        self,
        w_norm: FloatArray,
        sensitivities: FloatArray,
        budget: PrivacyBudget,
    ) -> float:
        """Binary search for proportional scaling factor."""
        lo, hi = 1e-10, 1000.0
        target_eps = budget.epsilon
        delta = budget.delta if budget.delta > 0 else 1e-10

        for _ in range(100):
            mid = (lo + hi) / 2.0
            eps_arr = mid * w_norm
            composed = self._compute_composed_dp(eps_arr, sensitivities, delta)
            if composed.epsilon < target_eps:
                lo = mid
            else:
                hi = mid

            if hi - lo < 1e-12:
                break

        return (lo + hi) / 2.0

    # -----------------------------------------------------------------
    # Convex optimisation
    # -----------------------------------------------------------------

    def optimize_convex(
        self,
        sensitivities: Sequence[float],
        error_fns: Optional[Sequence[ErrorFunction]] = None,
        *,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
        init_epsilons: Optional[FloatArray] = None,
    ) -> RDPAllocationResult:
        """Convex optimisation for budget allocation via projected gradient descent.

        Minimises Σ_t error_t(ε_t, Δ_t) subject to the RDP composition
        constraint. Uses projected gradient descent with projection onto
        the RDP-feasible set.

        The gradient of each error function with respect to ε_t is
        estimated via finite differences.

        Args:
            sensitivities: Per-query sensitivities.
            error_fns: Per-query error functions.
            learning_rate: Step size for gradient descent.
            max_iter: Maximum optimisation iterations.
            tol: Convergence tolerance on objective improvement.
            init_epsilons: Initial allocation. Defaults to uniform.

        Returns:
            Allocation result.

        Raises:
            ConvergenceError: If optimisation does not converge.
        """
        T = len(sensitivities)
        if T == 0:
            raise ConfigurationError(
                "At least one query is required",
                parameter="sensitivities",
            )

        sens = np.asarray(sensitivities, dtype=np.float64)
        error_fns = error_fns or [self._default_error_fn()] * T

        # Initialise
        if init_epsilons is not None:
            epsilons = np.asarray(init_epsilons, dtype=np.float64).copy()
        else:
            uniform_result = self.optimize_uniform(sens, error_fns)
            epsilons = uniform_result.epsilons.copy()

        eps_min = 1e-8
        prev_obj = float("inf")
        history: List[float] = []

        for iteration in range(max_iter):
            # Compute objective
            per_errors = np.array(
                [fn(float(e), float(s)) for fn, e, s in zip(error_fns, epsilons, sens)],
                dtype=np.float64,
            )
            obj = float(np.sum(per_errors))
            history.append(obj)

            # Check convergence
            if abs(prev_obj - obj) < tol and iteration > 0:
                composed = self._compute_composed_dp(epsilons, sens)
                return RDPAllocationResult(
                    epsilons=epsilons,
                    total_error=obj,
                    composed_epsilon=composed.epsilon,
                    composed_delta=composed.delta,
                    per_query_errors=per_errors,
                    n_iterations=iteration + 1,
                    converged=True,
                    metadata={"method": "convex", "history": history},
                )
            prev_obj = obj

            # Compute gradient via finite differences
            grad = self._compute_gradient(epsilons, sens, error_fns)

            # Gradient step
            epsilons_new = epsilons - learning_rate * grad

            # Project: ensure positivity
            epsilons_new = np.maximum(epsilons_new, eps_min)

            # Project onto RDP constraint set
            epsilons = self._project_rdp_feasible(epsilons_new, sens)

        # Did not converge
        per_errors = np.array(
            [fn(float(e), float(s)) for fn, e, s in zip(error_fns, epsilons, sens)],
            dtype=np.float64,
        )
        composed = self._compute_composed_dp(epsilons, sens)

        warnings.warn(
            f"RDP budget optimizer did not converge in {max_iter} iterations. "
            f"Final objective: {float(np.sum(per_errors)):.6f}",
            stacklevel=2,
        )

        return RDPAllocationResult(
            epsilons=epsilons,
            total_error=float(np.sum(per_errors)),
            composed_epsilon=composed.epsilon,
            composed_delta=composed.delta,
            per_query_errors=per_errors,
            n_iterations=max_iter,
            converged=False,
            metadata={"method": "convex", "history": history},
        )

    def _compute_gradient(
        self,
        epsilons: FloatArray,
        sensitivities: FloatArray,
        error_fns: Sequence[ErrorFunction],
        h: float = 1e-6,
    ) -> FloatArray:
        """Compute gradient of total error w.r.t. epsilons via central differences."""
        grad = np.zeros_like(epsilons)
        for t in range(len(epsilons)):
            eps_plus = epsilons.copy()
            eps_minus = epsilons.copy()
            eps_plus[t] += h
            eps_minus[t] = max(eps_minus[t] - h, 1e-10)

            f_plus = error_fns[t](float(eps_plus[t]), float(sensitivities[t]))
            f_minus = error_fns[t](float(eps_minus[t]), float(sensitivities[t]))
            grad[t] = (f_plus - f_minus) / (eps_plus[t] - eps_minus[t])

        return grad

    def _project_rdp_feasible(
        self,
        epsilons: FloatArray,
        sensitivities: FloatArray,
    ) -> FloatArray:
        """Project allocation onto the RDP-feasible set.

        If the current allocation exceeds the budget, uniformly scale
        down all epsilons until the composition constraint is satisfied.
        """
        delta = self._target_budget.delta if self._target_budget.delta > 0 else 1e-10

        composed = self._compute_composed_dp(epsilons, sensitivities, delta)
        if composed.epsilon <= self._target_budget.epsilon:
            return epsilons

        # Binary search for scaling factor
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            scaled = epsilons * mid
            composed = self._compute_composed_dp(scaled, sensitivities, delta)
            if composed.epsilon <= self._target_budget.epsilon:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break

        return epsilons * lo

    # -----------------------------------------------------------------
    # RDP composition helpers
    # -----------------------------------------------------------------

    def _compute_composed_dp(
        self,
        epsilons: FloatArray,
        sensitivities: FloatArray,
        delta: Optional[float] = None,
    ) -> PrivacyBudget:
        """Compute the composed (ε, δ)-DP for a given allocation."""
        delta = delta or (self._target_budget.delta if self._target_budget.delta > 0 else 1e-10)

        # Compute composed RDP
        composed_rdp = np.zeros_like(self._alphas)

        for eps, sens in zip(epsilons, sensitivities):
            if self._mechanism_type == "gaussian":
                # For Gaussian: σ = Δ/ε * √(2 log(1.25/δ))
                # RDP: α Δ²/(2σ²) = α ε² / (4 log(1.25/δ))
                # Simpler: use σ = Δ√(2 log(1.25/δ))/ε
                # Then RDP(α) = α Δ²/(2σ²) = α ε² / (4 log(1.25/δ))
                if eps <= 0:
                    continue
                sigma = float(sens) * math.sqrt(2.0 * math.log(1.25 / delta)) / float(eps)
                rdp = self._alphas * float(sens) ** 2 / (2.0 * sigma ** 2)
            elif self._mechanism_type == "laplace":
                rdp = self._laplace_rdp_at_alphas(float(eps))
            else:
                # Default: treat as Gaussian
                if eps <= 0:
                    continue
                sigma = float(sens) / float(eps)
                rdp = self._alphas * float(sens) ** 2 / (2.0 * sigma ** 2)

            composed_rdp += rdp

        # Convert to (ε, δ)-DP
        eps_result, _ = rdp_to_dp(composed_rdp, self._alphas, delta)
        return PrivacyBudget(epsilon=max(eps_result, 1e-15), delta=delta)

    def _laplace_rdp_at_alphas(self, epsilon: float) -> FloatArray:
        """Compute Laplace RDP curve at the optimizer's α grid."""
        rdp = np.empty_like(self._alphas)
        for i, alpha in enumerate(self._alphas):
            if abs(alpha - 1.0) < 1e-10:
                rdp[i] = 0.0
                continue
            a_minus_1 = alpha - 1.0
            denom = 2.0 * alpha - 1.0
            if denom <= 0:
                rdp[i] = 0.0
                continue
            log_t1 = math.log(alpha / denom) + a_minus_1 * epsilon
            log_t2 = math.log(a_minus_1 / denom) - a_minus_1 * epsilon
            log_sum = float(np.logaddexp(log_t1, log_t2))
            rdp[i] = max(log_sum / a_minus_1, 0.0)
        return rdp

    # -----------------------------------------------------------------
    # Default error functions
    # -----------------------------------------------------------------

    def _default_error_fn(self) -> ErrorFunction:
        """Return the default error function for the mechanism type.

        For Gaussian: MSE = Δ²·2log(1.25/δ) / ε²  (simplified)
        For Laplace: MSE = 2·Δ² / ε²
        """
        if self._mechanism_type == "gaussian":
            delta = self._target_budget.delta if self._target_budget.delta > 0 else 1e-10
            log_term = 2.0 * math.log(1.25 / delta)

            def gaussian_mse(eps: float, sens: float) -> float:
                if eps <= 0:
                    return float("inf")
                return sens ** 2 * log_term / (eps ** 2)

            return gaussian_mse
        else:
            def laplace_mse(eps: float, sens: float) -> float:
                if eps <= 0:
                    return float("inf")
                return 2.0 * sens ** 2 / (eps ** 2)

            return laplace_mse

    def __repr__(self) -> str:
        return (
            f"RDPBudgetOptimizer(budget={self._target_budget}, "
            f"mechanism={self._mechanism_type!r})"
        )

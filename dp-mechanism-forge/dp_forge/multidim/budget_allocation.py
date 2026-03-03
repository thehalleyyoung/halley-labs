"""
Per-coordinate privacy budget allocation for multi-dimensional mechanisms.

When a d-dimensional query decomposes into coordinate-separable marginals,
the overall privacy guarantee follows from composition of per-coordinate
budgets.  This module provides strategies for dividing a total (ε, δ) budget
across coordinates to minimise total error.

Strategies:
    - **Uniform**: ε_i = ε/d (baseline; optimal when all coordinates are
      equally sensitive and have equal output domain sizes).
    - **Proportional**: ε_i ∝ Δ_i / Σ_j Δ_j — allocate more budget to
      higher-sensitivity coordinates.
    - **Optimal (convex)**: Minimise Σ_i error_i(ε_i) subject to a
      composition constraint.  Uses scipy.optimize with advanced or
      RDP composition bounds.

Composition Backends:
    - Basic sequential: Σ ε_i ≤ ε_total.
    - Advanced (Dwork–Rothblum–Vadhan 2010).
    - RDP (Mironov 2017): compose at each Rényi order α, convert to (ε, δ).

Classes:
    BudgetAllocation  — result container for per-coordinate budgets
    BudgetAllocator   — allocation engine with uniform/proportional/optimal
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from dp_forge.exceptions import (
    BudgetExhaustedError,
    ConfigurationError,
)
from dp_forge.types import CompositionType, PrivacyBudget

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """Budget allocation strategy."""

    UNIFORM = auto()
    PROPORTIONAL = auto()
    OPTIMAL = auto()

    def __repr__(self) -> str:
        return f"AllocationStrategy.{self.name}"


@dataclass
class BudgetAllocation:
    """Result of per-coordinate budget allocation.

    Attributes:
        epsilons: Per-coordinate ε values, length d.
        deltas: Per-coordinate δ values, length d.
        strategy: Strategy used for allocation.
        total_epsilon: Total ε budget (input).
        total_delta: Total δ budget (input).
        composition_type: Composition theorem used.
        expected_errors: Estimated per-coordinate errors (if computed).
        total_expected_error: Sum of per-coordinate expected errors.
        metadata: Additional allocation metadata.
    """

    epsilons: npt.NDArray[np.float64]
    deltas: npt.NDArray[np.float64]
    strategy: AllocationStrategy
    total_epsilon: float
    total_delta: float
    composition_type: CompositionType = CompositionType.BASIC
    expected_errors: Optional[npt.NDArray[np.float64]] = None
    total_expected_error: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.epsilons = np.asarray(self.epsilons, dtype=np.float64)
        self.deltas = np.asarray(self.deltas, dtype=np.float64)
        if self.epsilons.ndim != 1:
            raise ValueError(
                f"epsilons must be 1-D, got shape {self.epsilons.shape}"
            )
        if self.deltas.ndim != 1:
            raise ValueError(
                f"deltas must be 1-D, got shape {self.deltas.shape}"
            )
        if len(self.epsilons) != len(self.deltas):
            raise ValueError(
                f"epsilons and deltas must have same length, "
                f"got {len(self.epsilons)} and {len(self.deltas)}"
            )
        if np.any(self.epsilons <= 0):
            raise ValueError("All per-coordinate epsilons must be > 0")
        if np.any(self.deltas < 0):
            raise ValueError("All per-coordinate deltas must be >= 0")

    @property
    def d(self) -> int:
        """Number of coordinates."""
        return len(self.epsilons)

    def budget_for(self, i: int) -> PrivacyBudget:
        """Return PrivacyBudget for coordinate i."""
        return PrivacyBudget(
            epsilon=float(self.epsilons[i]),
            delta=float(self.deltas[i]),
        )

    def __repr__(self) -> str:
        return (
            f"BudgetAllocation(d={self.d}, strategy={self.strategy.name}, "
            f"ε_total={self.total_epsilon}, δ_total={self.total_delta})"
        )


class BudgetAllocator:
    """Engine for allocating privacy budgets across coordinates.

    Args:
        composition_type: Which composition theorem to use for budget
            accounting.
        error_model: Optional callable mapping (ε, sensitivity, domain_size)
            to expected error for that coordinate.  Used by the optimal
            allocator. Defaults to Laplace-like 2·Δ²/ε² (L2 error).
    """

    def __init__(
        self,
        composition_type: CompositionType = CompositionType.BASIC,
        error_model: Optional[Callable[[float, float, int], float]] = None,
    ) -> None:
        self._composition_type = composition_type
        self._error_model = error_model or self._default_error_model

    @staticmethod
    def _default_error_model(epsilon: float, sensitivity: float, domain_size: int) -> float:
        """Default L2 error model: 2Δ²/ε² (Laplace mechanism variance)."""
        return 2.0 * sensitivity ** 2 / (epsilon ** 2)

    def allocate_uniform(
        self,
        total_budget: PrivacyBudget,
        d: int,
    ) -> BudgetAllocation:
        """Allocate budget uniformly: ε_i = ε/d, δ_i = δ/d.

        Under basic composition this is exact. Under advanced/RDP
        composition the actual consumed budget may be tighter.

        Args:
            total_budget: Total (ε, δ) budget.
            d: Number of coordinates.

        Returns:
            BudgetAllocation with uniform per-coordinate budgets.

        Raises:
            ConfigurationError: If d < 1.
        """
        if d < 1:
            raise ConfigurationError(
                f"d must be >= 1, got {d}",
                parameter="d",
                value=d,
            )
        if d == 1:
            return BudgetAllocation(
                epsilons=np.array([total_budget.epsilon]),
                deltas=np.array([total_budget.delta]),
                strategy=AllocationStrategy.UNIFORM,
                total_epsilon=total_budget.epsilon,
                total_delta=total_budget.delta,
                composition_type=self._composition_type,
            )
        eps_per = self._compute_per_coord_epsilon_basic(
            total_budget.epsilon, d
        )
        delta_per = total_budget.delta / d if total_budget.delta > 0 else 0.0
        epsilons = np.full(d, eps_per, dtype=np.float64)
        deltas = np.full(d, delta_per, dtype=np.float64)
        return BudgetAllocation(
            epsilons=epsilons,
            deltas=deltas,
            strategy=AllocationStrategy.UNIFORM,
            total_epsilon=total_budget.epsilon,
            total_delta=total_budget.delta,
            composition_type=self._composition_type,
        )

    def allocate_proportional(
        self,
        total_budget: PrivacyBudget,
        sensitivities: Sequence[float],
    ) -> BudgetAllocation:
        """Allocate ε_i proportional to per-coordinate sensitivity.

        Higher-sensitivity coordinates get more budget (larger ε_i)
        since they need more noise regardless. The allocation
        satisfies basic composition: Σ ε_i = ε_total.

        Args:
            total_budget: Total (ε, δ) budget.
            sensitivities: Per-coordinate sensitivities Δ_1, ..., Δ_d.

        Returns:
            BudgetAllocation with proportional per-coordinate budgets.

        Raises:
            ConfigurationError: If any sensitivity is non-positive.
        """
        d = len(sensitivities)
        sens = np.asarray(sensitivities, dtype=np.float64)
        if d < 1:
            raise ConfigurationError(
                "sensitivities must be non-empty",
                parameter="sensitivities",
            )
        if np.any(sens <= 0):
            raise ConfigurationError(
                "All sensitivities must be > 0",
                parameter="sensitivities",
                value=sensitivities,
            )
        if d == 1:
            return BudgetAllocation(
                epsilons=np.array([total_budget.epsilon]),
                deltas=np.array([total_budget.delta]),
                strategy=AllocationStrategy.PROPORTIONAL,
                total_epsilon=total_budget.epsilon,
                total_delta=total_budget.delta,
                composition_type=self._composition_type,
            )
        total_eps = self._compute_per_coord_epsilon_basic(
            total_budget.epsilon, d
        )
        # Proportional weights: higher sensitivity → higher ε
        weights = sens / sens.sum()
        epsilons = weights * total_budget.epsilon
        # Ensure no ε is too small
        min_eps = total_eps * 0.1
        epsilons = np.maximum(epsilons, min_eps)
        # Renormalize to satisfy composition
        epsilons = epsilons * (total_budget.epsilon / epsilons.sum())
        delta_per = total_budget.delta / d if total_budget.delta > 0 else 0.0
        deltas = np.full(d, delta_per, dtype=np.float64)
        return BudgetAllocation(
            epsilons=epsilons,
            deltas=deltas,
            strategy=AllocationStrategy.PROPORTIONAL,
            total_epsilon=total_budget.epsilon,
            total_delta=total_budget.delta,
            composition_type=self._composition_type,
        )

    def allocate_optimal(
        self,
        total_budget: PrivacyBudget,
        sensitivities: Sequence[float],
        domain_sizes: Optional[Sequence[int]] = None,
        error_model: Optional[Callable[[float, float, int], float]] = None,
    ) -> BudgetAllocation:
        """Optimally allocate budgets to minimise total expected error.

        Solves:  min Σ_i error_i(ε_i)  subject to  composition(ε_1,...,ε_d) ≤ ε_total

        For basic composition: Σ ε_i ≤ ε_total.
        For advanced composition: √(2d·ln(1/δ'))·ε_max + d·ε_max·(e^ε_max - 1) ≤ ε_total.

        Args:
            total_budget: Total (ε, δ) budget.
            sensitivities: Per-coordinate sensitivities.
            domain_sizes: Per-coordinate output domain sizes (for error model).
            error_model: Override error model for this call.

        Returns:
            BudgetAllocation with optimised per-coordinate budgets.

        Raises:
            ConfigurationError: If optimisation fails or parameters are invalid.
        """
        d = len(sensitivities)
        sens = np.asarray(sensitivities, dtype=np.float64)
        if d < 1:
            raise ConfigurationError(
                "sensitivities must be non-empty",
                parameter="sensitivities",
            )
        if np.any(sens <= 0):
            raise ConfigurationError(
                "All sensitivities must be > 0",
                parameter="sensitivities",
                value=sensitivities,
            )
        if domain_sizes is None:
            domain_sizes = [100] * d
        dom = list(domain_sizes)
        err_fn = error_model or self._error_model
        if d == 1:
            err = err_fn(total_budget.epsilon, float(sens[0]), dom[0])
            return BudgetAllocation(
                epsilons=np.array([total_budget.epsilon]),
                deltas=np.array([total_budget.delta]),
                strategy=AllocationStrategy.OPTIMAL,
                total_epsilon=total_budget.epsilon,
                total_delta=total_budget.delta,
                composition_type=self._composition_type,
                expected_errors=np.array([err]),
                total_expected_error=err,
            )
        # Optimise in log space for numerical stability: x_i = ln(ε_i)
        eps_init = total_budget.epsilon / d
        x0 = np.full(d, math.log(eps_init))

        def objective(x: npt.NDArray[np.float64]) -> float:
            epsilons = np.exp(x)
            return sum(err_fn(float(epsilons[i]), float(sens[i]), dom[i]) for i in range(d))

        def composition_constraint(x: npt.NDArray[np.float64]) -> float:
            epsilons = np.exp(x)
            return total_budget.epsilon - self._composition_bound(
                epsilons, total_budget.delta
            )

        constraints = [{"type": "ineq", "fun": composition_constraint}]
        # Bounds: each ε_i ∈ [eps_min, ε_total]
        eps_min = total_budget.epsilon * 1e-4
        bounds = [(math.log(eps_min), math.log(total_budget.epsilon))] * d

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            logger.warning(
                "Optimal budget allocation did not converge: %s. "
                "Falling back to proportional allocation.",
                result.message,
            )
            return self.allocate_proportional(total_budget, sensitivities)

        opt_epsilons = np.exp(result.x)
        # Project onto composition constraint
        consumed = self._composition_bound(opt_epsilons, total_budget.delta)
        if consumed > total_budget.epsilon:
            scale = total_budget.epsilon / consumed
            opt_epsilons *= scale

        delta_per = total_budget.delta / d if total_budget.delta > 0 else 0.0
        opt_deltas = np.full(d, delta_per, dtype=np.float64)

        errors = np.array(
            [err_fn(float(opt_epsilons[i]), float(sens[i]), dom[i]) for i in range(d)]
        )
        return BudgetAllocation(
            epsilons=opt_epsilons,
            deltas=opt_deltas,
            strategy=AllocationStrategy.OPTIMAL,
            total_epsilon=total_budget.epsilon,
            total_delta=total_budget.delta,
            composition_type=self._composition_type,
            expected_errors=errors,
            total_expected_error=float(errors.sum()),
            metadata={"optimizer_message": result.message, "n_iter": result.nit},
        )

    def _composition_bound(
        self,
        epsilons: npt.NDArray[np.float64],
        total_delta: float,
    ) -> float:
        """Compute composed ε from per-coordinate epsilons.

        Uses the configured composition theorem to bound the total
        privacy loss.

        Args:
            epsilons: Per-coordinate ε values.
            total_delta: Total δ budget.

        Returns:
            Composed ε value.
        """
        if self._composition_type == CompositionType.BASIC:
            return float(np.sum(epsilons))
        elif self._composition_type == CompositionType.ADVANCED:
            return self._advanced_composition(epsilons, total_delta)
        elif self._composition_type == CompositionType.RDP:
            return self._rdp_composition(epsilons, total_delta)
        elif self._composition_type == CompositionType.ZERO_CDP:
            return self._zcdp_composition(epsilons, total_delta)
        return float(np.sum(epsilons))

    def _advanced_composition(
        self,
        epsilons: npt.NDArray[np.float64],
        total_delta: float,
    ) -> float:
        """Advanced composition (Dwork–Rothblum–Vadhan 2010).

        For k mechanisms each with ε_max:
            ε_total = √(2k·ln(1/δ'))·ε_max + k·ε_max·(e^ε_max - 1)

        For heterogeneous ε_i, we use the tighter bound:
            ε_total = √(2·ln(1/δ')·Σε_i²) + Σε_i·(e^ε_i - 1)

        Args:
            epsilons: Per-coordinate ε values.
            total_delta: Total δ budget.

        Returns:
            Composed ε under advanced composition.
        """
        d = len(epsilons)
        if total_delta <= 0:
            return float(np.sum(epsilons))
        # Reserve half of δ for the composition slack
        delta_prime = total_delta / 2.0
        if delta_prime <= 0:
            return float(np.sum(epsilons))
        sum_sq = float(np.sum(epsilons ** 2))
        sum_exp_term = float(np.sum(epsilons * (np.exp(epsilons) - 1.0)))
        composed = math.sqrt(2.0 * math.log(1.0 / delta_prime) * sum_sq) + sum_exp_term
        return composed

    def _rdp_composition(
        self,
        epsilons: npt.NDArray[np.float64],
        total_delta: float,
    ) -> float:
        """RDP composition (Mironov 2017).

        For pure ε-DP mechanisms, RDP at order α is ε·α/(α-1) for α > 1.
        Compose by summing RDP values, then convert back to (ε, δ)-DP.

        Args:
            epsilons: Per-coordinate ε values.
            total_delta: Total δ budget.

        Returns:
            Composed ε under RDP composition.
        """
        if total_delta <= 0:
            return float(np.sum(epsilons))
        best_eps = float("inf")
        for alpha in [1.5, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256]:
            # RDP of pure ε-DP mechanism at order α
            rdp_values = epsilons * alpha / (alpha - 1.0)
            total_rdp = float(np.sum(rdp_values))
            # Convert RDP to (ε, δ)-DP
            eps_from_rdp = total_rdp + math.log(1.0 / total_delta) / (alpha - 1.0)
            best_eps = min(best_eps, eps_from_rdp)
        return best_eps

    def _zcdp_composition(
        self,
        epsilons: npt.NDArray[np.float64],
        total_delta: float,
    ) -> float:
        """zCDP composition (Bun & Steinke 2016).

        Pure ε-DP implies ε²/2-zCDP. zCDP composes by summing ρ values.
        Convert back: ε_total = ρ + 2√(ρ·ln(1/δ)).

        Args:
            epsilons: Per-coordinate ε values.
            total_delta: Total δ budget.

        Returns:
            Composed ε under zCDP composition.
        """
        if total_delta <= 0:
            return float(np.sum(epsilons))
        rho_values = epsilons ** 2 / 2.0
        total_rho = float(np.sum(rho_values))
        composed = total_rho + 2.0 * math.sqrt(total_rho * math.log(1.0 / total_delta))
        return composed

    def _compute_per_coord_epsilon_basic(
        self, total_epsilon: float, d: int
    ) -> float:
        """Compute per-coordinate ε under basic composition: ε_i = ε/d."""
        return total_epsilon / d

    def verify_composition(
        self, allocation: BudgetAllocation
    ) -> Tuple[bool, float]:
        """Verify that an allocation satisfies the composition constraint.

        Args:
            allocation: Budget allocation to verify.

        Returns:
            Tuple of (is_valid, consumed_epsilon).

        Raises:
            BudgetExhaustedError: If the composed budget exceeds the total.
        """
        consumed = self._composition_bound(
            allocation.epsilons, allocation.total_delta
        )
        valid = consumed <= allocation.total_epsilon * (1.0 + 1e-8)
        if not valid:
            raise BudgetExhaustedError(
                f"Composed ε={consumed:.6f} exceeds budget ε={allocation.total_epsilon:.6f}",
                budget_epsilon=allocation.total_epsilon,
                budget_delta=allocation.total_delta,
                consumed_epsilon=consumed,
            )
        return valid, consumed

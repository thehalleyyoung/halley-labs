"""
Sequential, parallel, adaptive, and subsampled composition for DP-Forge.

This module provides composable building blocks for combining multiple
differentially private mechanisms under formal privacy accounting.  It
implements several composition theorems from the DP literature and exposes
both low-level accounting primitives and high-level composer classes that
track budget, apply amplification, and produce composed mechanisms ready
for deployment.

Composition Theorems Implemented:
    - **Basic (sequential) composition**: ε_total = Σε_i, δ_total = Σδ_i.
      (Dwork, McSherry, Nissim, Smith 2006.)
    - **Advanced composition**: Dwork, Rothblum, Vadhan (2010).
      ε_total = √(2k ln(1/δ')) · ε + k · ε · (e^ε − 1), δ_total = kδ + δ'.
    - **Optimal composition**: Numerical optimisation over the composition
      space for tightest (ε, δ) guarantee.
    - **RDP composition**: Rényi DP composition (Mironov 2017) with optimal
      order selection and conversion to (ε, δ)-DP.
    - **zCDP composition**: Zero-concentrated DP composition (Bun & Steinke
      2016) with conversion.
    - **Parallel composition**: ε_total = max(ε_i) when mechanisms operate on
      disjoint subsets.
    - **Adaptive composition**: Budget allocation strategies for sequences of
      queries whose parameters may depend on prior answers.
    - **Subsampling amplification**: Privacy amplification via Poisson or
      without-replacement subsampling.

Classes:
    SequentialComposer     — sequential (and advanced/optimal/RDP/zCDP) composition
    ParallelComposer       — parallel composition on disjoint subsets
    AdaptiveComposer       — adaptive budget allocation policies
    SubsampledComposer     — subsampling amplification
    ComposedMechanism      — mechanism wrapping a composed pipeline
    CompositionOptimizer   — budget allocation optimisation

All composers accept and return :class:`dp_forge.types.PrivacyBudget`
instances and raise :class:`dp_forge.exceptions.BudgetExhaustedError` when
budgets are exceeded.
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
    Union,
)

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import (
    BudgetExhaustedError,
    ConfigurationError,
    InvalidMechanismError,
)
from dp_forge.types import (
    CompositionType,
    ExtractedMechanism,
    PrivacyBudget,
    QuerySpec,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _logsumexp(a: FloatArray) -> float:
    """Numerically stable log-sum-exp."""
    a = np.asarray(a, dtype=np.float64).ravel()
    if len(a) == 0:
        return -np.inf
    a_max = float(np.max(a))
    if not np.isfinite(a_max):
        return a_max
    return float(a_max + np.log(np.sum(np.exp(a - a_max))))


def _validate_positive(value: float, name: str) -> None:
    """Raise ConfigurationError if value is not positive and finite."""
    if not (math.isfinite(value) and value > 0):
        raise ConfigurationError(
            f"{name} must be positive and finite, got {value}",
            parameter=name,
            value=value,
        )


def _validate_probability(value: float, name: str) -> None:
    """Raise ConfigurationError if value not in (0, 1)."""
    if not (0.0 < value < 1.0):
        raise ConfigurationError(
            f"{name} must be in (0, 1), got {value}",
            parameter=name,
            value=value,
        )


def _validate_non_negative(value: float, name: str) -> None:
    """Raise ConfigurationError if value is negative."""
    if value < 0:
        raise ConfigurationError(
            f"{name} must be >= 0, got {value}",
            parameter=name,
            value=value,
        )


# =========================================================================
# Data containers for composition results
# =========================================================================


@dataclass
class CompositionResult:
    """Result of a privacy composition calculation.

    Attributes:
        epsilon: Total epsilon after composition.
        delta: Total delta after composition.
        composition_type: Which composition theorem was used.
        n_mechanisms: Number of mechanisms composed.
        per_mechanism_budgets: Individual budgets consumed by each mechanism.
        metadata: Additional information (e.g., RDP orders, intermediate values).
    """

    epsilon: float
    delta: float
    composition_type: CompositionType
    n_mechanisms: int
    per_mechanism_budgets: List[PrivacyBudget] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {self.epsilon}")
        if not (0.0 <= self.delta < 1.0):
            raise ValueError(f"delta must be in [0, 1), got {self.delta}")

    @property
    def budget(self) -> PrivacyBudget:
        """Return as a PrivacyBudget for downstream use."""
        return PrivacyBudget(
            epsilon=max(self.epsilon, 1e-15),
            delta=self.delta,
            composition_type=self.composition_type,
        )

    def __repr__(self) -> str:
        dp = f"ε={self.epsilon:.4f}"
        if self.delta > 0:
            dp += f", δ={self.delta:.2e}"
        return (
            f"CompositionResult({dp}, type={self.composition_type.name}, "
            f"n={self.n_mechanisms})"
        )


@dataclass
class AllocationResult:
    """Result of a budget allocation optimization.

    Attributes:
        epsilons: Per-query epsilon allocations.
        deltas: Per-query delta allocations.
        total_epsilon: Total epsilon consumed.
        total_delta: Total delta consumed.
        objective_value: Value of the optimization objective (e.g., total MSE).
        metadata: Optimization metadata.
    """

    epsilons: FloatArray
    deltas: FloatArray
    total_epsilon: float
    total_delta: float
    objective_value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.epsilons = np.asarray(self.epsilons, dtype=np.float64)
        self.deltas = np.asarray(self.deltas, dtype=np.float64)
        if np.any(self.epsilons < 0):
            raise ValueError("All epsilon allocations must be >= 0")
        if np.any(self.deltas < 0):
            raise ValueError("All delta allocations must be >= 0")

    @property
    def n_queries(self) -> int:
        """Number of queries in the allocation."""
        return len(self.epsilons)

    def __repr__(self) -> str:
        return (
            f"AllocationResult(n={self.n_queries}, "
            f"ε_total={self.total_epsilon:.4f}, δ_total={self.total_delta:.2e})"
        )


# =========================================================================
# 1. SequentialComposer
# =========================================================================


class SequentialComposer:
    """Sequential composition of differentially private mechanisms.

    When k mechanisms M_1, ..., M_k are applied sequentially to the same
    dataset, the total privacy loss is bounded by various composition
    theorems of increasing tightness:

    - **Basic**: ε_total = Σε_i, δ_total = Σδ_i.
    - **Advanced**: Uses the DRV-2010 theorem for tighter bounds when k is
      large and ε_i are small.
    - **Optimal**: Numerically finds the tightest (ε, δ) via convex
      optimization over the composition domain.
    - **RDP**: Composes via Rényi divergence, then converts to (ε, δ)-DP.
    - **zCDP**: Composes via zero-concentrated DP, then converts.

    Usage::

        composer = SequentialComposer()
        result = composer.compose(mechanisms, budgets)
        print(f"Total privacy: {result.epsilon}, {result.delta}")
    """

    def __init__(self, default_method: CompositionType = CompositionType.ADVANCED) -> None:
        """Initialize composer with a default composition method.

        Args:
            default_method: Default composition theorem to use.
        """
        self._default_method = default_method

    @property
    def default_method(self) -> CompositionType:
        """The default composition method."""
        return self._default_method

    def compose(
        self,
        mechanisms: List[Any],
        budgets: List[PrivacyBudget],
        method: Optional[CompositionType] = None,
        target_delta: Optional[float] = None,
    ) -> CompositionResult:
        """Compose a list of mechanisms under the chosen composition theorem.

        Args:
            mechanisms: List of mechanism objects (used for metadata only here).
            budgets: Per-mechanism privacy budgets.
            method: Composition theorem to use; defaults to ``default_method``.
            target_delta: Target delta for RDP/zCDP conversion. If ``None``,
                uses the maximum delta among the individual budgets.

        Returns:
            CompositionResult with total privacy parameters.

        Raises:
            ConfigurationError: If inputs are invalid.
        """
        if len(mechanisms) != len(budgets):
            raise ConfigurationError(
                f"mechanisms ({len(mechanisms)}) and budgets ({len(budgets)}) "
                f"must have the same length",
                parameter="mechanisms/budgets",
            )
        if not mechanisms:
            raise ConfigurationError(
                "At least one mechanism is required for composition",
                parameter="mechanisms",
            )

        method = method or self._default_method
        eps_list = np.array([b.epsilon for b in budgets], dtype=np.float64)
        delta_list = np.array([b.delta for b in budgets], dtype=np.float64)

        if target_delta is None:
            target_delta = float(np.max(delta_list)) if np.any(delta_list > 0) else 0.0

        if method == CompositionType.BASIC:
            eps_total, delta_total = self.basic_composition(eps_list, delta_list)
        elif method == CompositionType.ADVANCED:
            eps_total, delta_total = self.advanced_composition(
                eps_list, delta_list, target_delta
            )
        elif method == CompositionType.RDP:
            eps_total, delta_total = self.rdp_composition(
                eps_list, delta_list, target_delta
            )
        elif method == CompositionType.ZERO_CDP:
            eps_total, delta_total = self.zcdp_composition(
                eps_list, delta_list, target_delta
            )
        else:
            raise ConfigurationError(
                f"Unknown composition method: {method}",
                parameter="method",
                value=method,
            )

        return CompositionResult(
            epsilon=eps_total,
            delta=delta_total,
            composition_type=method,
            n_mechanisms=len(mechanisms),
            per_mechanism_budgets=list(budgets),
            metadata={"target_delta": target_delta},
        )

    # ----- Basic composition -----

    def basic_composition(
        self,
        eps_list: FloatArray,
        delta_list: Optional[FloatArray] = None,
    ) -> Tuple[float, float]:
        """Basic sequential composition: sum of epsilons and deltas.

        For k mechanisms satisfying (ε_i, δ_i)-DP, the composed mechanism
        satisfies (Σε_i, Σδ_i)-DP.

        Args:
            eps_list: Array of per-mechanism epsilon values.
            delta_list: Array of per-mechanism delta values. Defaults to 0.

        Returns:
            Tuple of (total_epsilon, total_delta).
        """
        eps_list = np.asarray(eps_list, dtype=np.float64)
        if delta_list is None:
            delta_list = np.zeros_like(eps_list)
        else:
            delta_list = np.asarray(delta_list, dtype=np.float64)

        if len(eps_list) != len(delta_list):
            raise ConfigurationError(
                "eps_list and delta_list must have the same length",
                parameter="eps_list/delta_list",
            )

        return float(np.sum(eps_list)), float(np.sum(delta_list))

    # ----- Advanced composition (DRV 2010) -----

    def advanced_composition(
        self,
        eps_list: FloatArray,
        delta_list: Optional[FloatArray] = None,
        delta_prime: float = 1e-5,
    ) -> Tuple[float, float]:
        """Advanced composition theorem (Dwork, Rothblum, Vadhan 2010).

        For k mechanisms each satisfying (ε, δ)-DP with the same ε:

            ε_total = √(2k · ln(1/δ')) · ε  +  k · ε · (e^ε − 1)
            δ_total = k · δ + δ'

        When epsilons differ, we use the maximum and apply the formula.
        For heterogeneous compositions, RDP-based methods are tighter.

        Args:
            eps_list: Array of per-mechanism epsilon values.
            delta_list: Array of per-mechanism delta values.
            delta_prime: Additional failure probability δ' > 0.

        Returns:
            Tuple of (total_epsilon, total_delta).

        Raises:
            ConfigurationError: If delta_prime is not in (0, 1).
        """
        eps_list = np.asarray(eps_list, dtype=np.float64)
        if delta_list is None:
            delta_list = np.zeros_like(eps_list)
        else:
            delta_list = np.asarray(delta_list, dtype=np.float64)

        _validate_probability(delta_prime, "delta_prime")

        k = len(eps_list)
        if k == 0:
            return 0.0, 0.0

        eps_max = float(np.max(eps_list))

        # Advanced composition formula
        sqrt_term = math.sqrt(2.0 * k * math.log(1.0 / delta_prime)) * eps_max
        linear_term = k * eps_max * (math.exp(eps_max) - 1.0)
        eps_total = sqrt_term + linear_term

        # Also compute basic composition for comparison — use the tighter one
        eps_basic = float(np.sum(eps_list))
        eps_total = min(eps_total, eps_basic)

        delta_total = float(k * np.max(delta_list) + delta_prime)
        delta_total = min(delta_total, 1.0 - 1e-15)

        return eps_total, delta_total

    # ----- Optimal composition via numerical optimization -----

    def optimal_composition(
        self,
        eps_list: FloatArray,
        delta_target: float = 1e-5,
        n_grid: int = 1000,
    ) -> Tuple[float, float]:
        """Numerically optimal composition via grid search.

        Searches over the trade-off between ε and δ in the composition
        to find the tightest total ε for a given target δ.  Uses the
        optimal composition theorem (Kairouz, Oh, Viswanath 2015).

        For homogeneous mechanisms with the same ε, the optimal composition
        is computed via the PLD (privacy loss distribution) convolution approach.

        Args:
            eps_list: Array of per-mechanism epsilon values.
            delta_target: Target total delta.
            n_grid: Number of grid points for numerical search.

        Returns:
            Tuple of (optimal_epsilon, delta_target).
        """
        eps_list = np.asarray(eps_list, dtype=np.float64)
        _validate_probability(delta_target, "delta_target")

        k = len(eps_list)
        if k == 0:
            return 0.0, 0.0

        # For homogeneous case, use the KOV15 formula
        eps_max = float(np.max(eps_list))
        eps_min = float(np.min(eps_list))

        if np.allclose(eps_list, eps_max, rtol=1e-10):
            return self._optimal_homogeneous(eps_max, k, delta_target, n_grid)

        # Heterogeneous case: use PLD convolution via discretization
        return self._optimal_heterogeneous(eps_list, delta_target, n_grid)

    def _optimal_homogeneous(
        self,
        eps: float,
        k: int,
        delta_target: float,
        n_grid: int,
    ) -> Tuple[float, float]:
        """Optimal composition for homogeneous mechanisms (same ε each).

        Uses the hockey-stick divergence characterization from Kairouz, Oh,
        and Viswanath (2015).  The composed privacy loss distribution is
        computed via repeated convolution of the two-point distribution.

        Args:
            eps: Per-mechanism epsilon.
            k: Number of mechanisms.
            delta_target: Target total delta.
            n_grid: Grid resolution for searching.

        Returns:
            Tuple of (optimal_epsilon, delta_target).
        """
        # The privacy loss RV for a single (eps,0)-DP mechanism takes values
        # +eps and -eps with probabilities e^eps/(1+e^eps) and 1/(1+e^eps).
        p_plus = math.exp(eps) / (1.0 + math.exp(eps))
        p_minus = 1.0 - p_plus

        # After k compositions, the total privacy loss is a sum of k iid RVs
        # each in {-eps, +eps}.  The sum takes values in {-k*eps, ..., +k*eps}
        # in steps of 2*eps.
        # L = sum_i X_i where X_i in {-eps, +eps}
        # Number of +eps among k trials is Binomial(k, p_plus)
        # If j of k are +eps, then L = j*eps - (k-j)*eps = (2j-k)*eps

        # For each possible total loss L = (2j-k)*eps, compute
        # Pr[loss >= threshold] and find threshold achieving delta_target.
        from scipy import stats as sp_stats

        best_eps = k * eps  # basic composition as fallback
        for j in range(k + 1):
            total_loss = (2 * j - k) * eps
            # Probability of getting j or more +eps outcomes
            tail_prob = float(sp_stats.binom.sf(j - 1, k, p_plus))
            # delta(eps_threshold) = max over distributions of
            # Pr[loss > threshold] - e^threshold * Pr[loss < -threshold]
            # For this two-point model, a candidate total epsilon is total_loss
            candidate_delta = tail_prob - math.exp(total_loss) * float(
                sp_stats.binom.cdf(j, k, p_minus)
            )
            if candidate_delta <= delta_target and total_loss < best_eps:
                best_eps = max(total_loss, 0.0)

        # Also try the advanced composition for comparison
        adv_eps, _ = self.advanced_composition(
            np.full(k, eps), delta_prime=delta_target
        )
        best_eps = min(best_eps, adv_eps)
        best_eps = max(best_eps, 0.0)

        return best_eps, delta_target

    def _optimal_heterogeneous(
        self,
        eps_list: FloatArray,
        delta_target: float,
        n_grid: int,
    ) -> Tuple[float, float]:
        """Optimal composition for heterogeneous mechanisms via PLD discretization.

        Approximates the privacy loss distribution (PLD) of each mechanism as
        a two-point distribution and convolves them numerically.

        Args:
            eps_list: Per-mechanism epsilon values.
            delta_target: Target delta.
            n_grid: Grid resolution.

        Returns:
            Tuple of (epsilon, delta_target).
        """
        k = len(eps_list)

        # Use the advanced composition as upper bound
        adv_eps, adv_delta = self.advanced_composition(
            eps_list, delta_prime=delta_target
        )

        # PLD convolution: discretize the domain [-sum_eps, sum_eps]
        sum_eps = float(np.sum(eps_list))
        grid = np.linspace(-sum_eps, sum_eps, n_grid)
        step = grid[1] - grid[0] if n_grid > 1 else 1.0

        # Start with a point mass at 0
        log_pld = np.full(n_grid, -np.inf)
        center_idx = n_grid // 2
        log_pld[center_idx] = 0.0  # Pr[L = 0] = 1

        for eps_i in eps_list:
            p_plus = math.exp(eps_i) / (1.0 + math.exp(eps_i))
            p_minus = 1.0 - p_plus
            log_p_plus = math.log(p_plus)
            log_p_minus = math.log(p_minus)

            # Shift by +eps_i and -eps_i
            shift_bins = max(1, int(round(eps_i / step)))
            new_log_pld = np.full(n_grid, -np.inf)

            # Vectorized: find active indices once
            active = np.where(log_pld > -100)[0]
            for idx in active:
                # Contribution from +eps_i shift
                target_plus = idx + shift_bins
                if 0 <= target_plus < n_grid:
                    lp = log_pld[idx] + log_p_plus
                    new_log_pld[target_plus] = np.logaddexp(
                        new_log_pld[target_plus], lp
                    )
                # Contribution from -eps_i shift
                target_minus = idx - shift_bins
                if 0 <= target_minus < n_grid:
                    lm = log_pld[idx] + log_p_minus
                    new_log_pld[target_minus] = np.logaddexp(
                        new_log_pld[target_minus], lm
                    )

            log_pld = new_log_pld

        # Find the smallest epsilon such that delta(epsilon) <= delta_target
        # delta(eps) = sum_{L > eps} Pr[L] - e^eps * sum_{L < -eps} Pr[L]
        # Vectorized: compute for all non-negative grid points at once
        best_eps = adv_eps
        non_neg_mask = grid >= 0
        candidates = grid[non_neg_mask]

        # Sort log_pld by grid value for cumulative computation
        sorted_idx = np.argsort(grid)
        sorted_grid = grid[sorted_idx]
        sorted_log_pld = log_pld[sorted_idx]

        for i in range(len(candidates)):
            candidate_eps = candidates[i]

            mask_above = sorted_grid > candidate_eps
            if not np.any(mask_above):
                tail_above = 0.0
            else:
                tail_above = math.exp(float(np.logaddexp.reduce(sorted_log_pld[mask_above])))

            mask_below = sorted_grid < -candidate_eps
            if not np.any(mask_below):
                tail_below = 0.0
            else:
                tail_below = math.exp(float(np.logaddexp.reduce(sorted_log_pld[mask_below])))

            delta_candidate = tail_above - math.exp(candidate_eps) * tail_below
            delta_candidate = max(delta_candidate, 0.0)

            if delta_candidate <= delta_target and candidate_eps < best_eps:
                best_eps = candidate_eps

        return max(best_eps, 0.0), delta_target

    # ----- RDP composition -----

    def rdp_composition(
        self,
        eps_list: FloatArray,
        delta_list: Optional[FloatArray] = None,
        target_delta: float = 1e-5,
        alpha_range: Optional[FloatArray] = None,
    ) -> Tuple[float, float]:
        """Compose via Rényi Differential Privacy (Mironov 2017).

        RDP composition is additive: if M_i satisfies (α, ε_i(α))-RDP, then
        the composition satisfies (α, Σε_i(α))-RDP.  We convert each (ε_i, δ_i)-DP
        guarantee to RDP at multiple orders, compose, then convert back.

        Args:
            eps_list: Per-mechanism epsilon values.
            delta_list: Per-mechanism delta values (default 0).
            target_delta: Target delta for RDP-to-(ε,δ) conversion.
            alpha_range: RDP orders to evaluate. If None, uses a standard grid.

        Returns:
            Tuple of (epsilon, delta) after conversion from RDP.
        """
        eps_list = np.asarray(eps_list, dtype=np.float64)
        if delta_list is None:
            delta_list = np.zeros_like(eps_list)
        else:
            delta_list = np.asarray(delta_list, dtype=np.float64)

        _validate_probability(target_delta, "target_delta")

        if alpha_range is None:
            alpha_range = np.concatenate([
                np.arange(1.1, 2.0, 0.1),
                np.arange(2.0, 10.0, 0.5),
                np.arange(10.0, 100.0, 5.0),
                np.array([128.0, 256.0, 512.0, 1024.0]),
            ])

        k = len(eps_list)
        if k == 0:
            return 0.0, 0.0

        best_eps = float("inf")

        for alpha in alpha_range:
            if alpha <= 1.0:
                continue

            # Convert each (ε_i, δ_i)-DP to (α, ε_i(α))-RDP
            total_rdp = 0.0
            for i in range(k):
                rdp_i = self._eps_to_rdp(eps_list[i], delta_list[i], alpha)
                total_rdp += rdp_i

            # Convert back to (ε, δ)-DP
            eps_candidate = total_rdp + math.log(1.0 / target_delta) / (alpha - 1.0)

            if eps_candidate < best_eps:
                best_eps = eps_candidate

        # Also bound by basic composition
        basic_eps = float(np.sum(eps_list))
        best_eps = min(best_eps, basic_eps)

        return max(best_eps, 0.0), target_delta

    def _eps_to_rdp(self, epsilon: float, delta: float, alpha: float) -> float:
        """Convert (ε, δ)-DP guarantee to (α, ε(α))-RDP.

        Uses the conversion: ε(α) >= ε + ln(1 - 1/α) / (α - 1)
        for the pure DP component, and handles δ > 0 via:
        ε_RDP(α) = ε + log(1 + (e^ε - 1) · ... ) / (α - 1)

        For pure DP (δ = 0): ε_RDP(α) = α · ε² / 2  (for small ε).
        Exact: ε_RDP(α) = (1/(α-1)) · log((α-1)/(α) · e^{(α-1)ε} + 1/α)

        Args:
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.
            alpha: Rényi divergence order.

        Returns:
            RDP guarantee ε(α).
        """
        if delta == 0:
            # Pure DP to RDP: exact conversion
            # (α, ε_RDP)-RDP where ε_RDP = (1/(α-1)) * log(((α-1)/α)*e^{(α-1)*ε} + 1/α)
            try:
                term1 = ((alpha - 1.0) / alpha) * math.exp((alpha - 1.0) * epsilon)
                term2 = 1.0 / alpha
                rdp = math.log(term1 + term2) / (alpha - 1.0)
            except OverflowError:
                rdp = epsilon  # fallback to ε for large α·ε
            return rdp
        else:
            # Approximate DP to RDP conversion
            # Use ε_RDP(α) ≈ ε + log(1/δ) / (α - 1) as a loose upper bound
            # A tighter conversion exists but requires more information about
            # the mechanism
            rdp_pure = self._eps_to_rdp(epsilon, 0.0, alpha)
            rdp_delta_correction = math.log(1.0 / max(delta, 1e-300)) / (alpha - 1.0)
            # The RDP of an (ε, δ)-DP mechanism is bounded by multiple formulas;
            # use the tightest
            return min(rdp_pure, epsilon + rdp_delta_correction)

    # ----- zCDP composition -----

    def zcdp_composition(
        self,
        eps_list: FloatArray,
        delta_list: Optional[FloatArray] = None,
        target_delta: float = 1e-5,
    ) -> Tuple[float, float]:
        """Compose via zero-Concentrated Differential Privacy (Bun & Steinke 2016).

        Each (ε, 0)-DP mechanism satisfies ρ-zCDP with ρ = ε²/2.
        zCDP composes additively: ρ_total = Σρ_i.
        Conversion back: (ε, δ)-DP with ε = ρ + 2√(ρ · ln(1/δ)).

        For (ε, δ)-DP mechanisms with δ > 0, we use the tightest known
        conversion, falling back to the pure-DP bound.

        Args:
            eps_list: Per-mechanism epsilon values.
            delta_list: Per-mechanism delta values.
            target_delta: Target delta for zCDP-to-(ε,δ) conversion.

        Returns:
            Tuple of (epsilon, delta) after conversion from zCDP.
        """
        eps_list = np.asarray(eps_list, dtype=np.float64)
        if delta_list is None:
            delta_list = np.zeros_like(eps_list)
        else:
            delta_list = np.asarray(delta_list, dtype=np.float64)

        _validate_probability(target_delta, "target_delta")

        k = len(eps_list)
        if k == 0:
            return 0.0, 0.0

        # Convert each to zCDP parameter ρ
        rho_total = 0.0
        delta_sum = 0.0
        for i in range(k):
            rho_i = self._eps_to_zcdp(eps_list[i])
            rho_total += rho_i
            delta_sum += delta_list[i]

        # Convert ρ_total back to (ε, δ)-DP
        # ε = ρ + 2√(ρ · ln(1/δ_zcdp)) where δ_zcdp = target_delta - delta_sum
        delta_zcdp = target_delta - delta_sum
        if delta_zcdp <= 0:
            # Not enough delta budget — fall back to basic composition
            return self.basic_composition(eps_list, delta_list)

        eps_total = rho_total + 2.0 * math.sqrt(rho_total * math.log(1.0 / delta_zcdp))

        # Also check basic composition
        basic_eps = float(np.sum(eps_list))
        eps_total = min(eps_total, basic_eps)

        return max(eps_total, 0.0), target_delta

    def _eps_to_zcdp(self, epsilon: float) -> float:
        """Convert pure ε-DP to ρ-zCDP.

        For pure ε-DP, ρ = ε² / 2 (Proposition 1.4, Bun & Steinke 2016).

        Args:
            epsilon: Privacy parameter ε.

        Returns:
            zCDP parameter ρ.
        """
        return epsilon ** 2 / 2.0


# =========================================================================
# 2. ParallelComposer
# =========================================================================


class ParallelComposer:
    """Parallel composition of DP mechanisms on disjoint data subsets.

    When mechanisms operate on disjoint subsets of the data, the total
    privacy cost is governed by the maximum (not the sum) of individual
    privacy parameters.  This is the parallel composition theorem.

    Usage::

        composer = ParallelComposer()
        partitions = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        result = composer.compose_parallel(mechanisms, budgets, partitions)
    """

    def compose_parallel(
        self,
        mechanisms: List[Any],
        budgets: List[PrivacyBudget],
        partitions: List[List[int]],
    ) -> CompositionResult:
        """Compose mechanisms acting on disjoint data partitions.

        Under parallel composition, if mechanism M_i operates on disjoint
        subset S_i of the data domain, the composed mechanism satisfies
        (max ε_i, max δ_i)-DP.

        Args:
            mechanisms: List of mechanism objects.
            budgets: Per-mechanism privacy budgets.
            partitions: List of index sets, one per mechanism.

        Returns:
            CompositionResult with max-based privacy parameters.

        Raises:
            ConfigurationError: If partitions are not disjoint or inputs
                are inconsistent.
        """
        if len(mechanisms) != len(budgets):
            raise ConfigurationError(
                f"mechanisms ({len(mechanisms)}) and budgets ({len(budgets)}) "
                f"must have the same length",
                parameter="mechanisms/budgets",
            )
        if len(mechanisms) != len(partitions):
            raise ConfigurationError(
                f"mechanisms ({len(mechanisms)}) and partitions ({len(partitions)}) "
                f"must have the same length",
                parameter="mechanisms/partitions",
            )
        if not mechanisms:
            raise ConfigurationError(
                "At least one mechanism is required",
                parameter="mechanisms",
            )

        self.verify_disjointness(partitions)

        eps_max = max(b.epsilon for b in budgets)
        delta_max = max(b.delta for b in budgets)

        return CompositionResult(
            epsilon=eps_max,
            delta=delta_max,
            composition_type=CompositionType.BASIC,
            n_mechanisms=len(mechanisms),
            per_mechanism_budgets=list(budgets),
            metadata={"composition_mode": "parallel", "partitions": partitions},
        )

    def verify_disjointness(self, partitions: List[List[int]]) -> bool:
        """Verify that partitions are pairwise disjoint.

        Args:
            partitions: List of index sets to check.

        Returns:
            True if partitions are disjoint.

        Raises:
            ConfigurationError: If partitions overlap.
        """
        seen: set[int] = set()
        for part_idx, partition in enumerate(partitions):
            partition_set = set(partition)
            overlap = seen & partition_set
            if overlap:
                raise ConfigurationError(
                    f"Partition {part_idx} overlaps with previous partitions "
                    f"on indices: {sorted(overlap)}",
                    parameter="partitions",
                )
            seen |= partition_set
        return True

    def compose_with_overlap(
        self,
        mechanisms: List[Any],
        budgets: List[PrivacyBudget],
        partitions: List[List[int]],
    ) -> CompositionResult:
        """Compose mechanisms that may have overlapping partitions.

        For overlapping partitions, we compute the maximum privacy cost per
        data element and return the worst case.  This is a hybrid of parallel
        and sequential composition.

        Args:
            mechanisms: List of mechanism objects.
            budgets: Per-mechanism privacy budgets.
            partitions: List of index sets (may overlap).

        Returns:
            CompositionResult with privacy parameters accounting for overlap.
        """
        if len(mechanisms) != len(budgets) or len(mechanisms) != len(partitions):
            raise ConfigurationError(
                "mechanisms, budgets, and partitions must have the same length",
                parameter="mechanisms/budgets/partitions",
            )

        # Compute per-element exposure: for each data index, sum the epsilons
        # of mechanisms that touch it.
        element_eps: Dict[int, float] = {}
        element_delta: Dict[int, float] = {}
        for i, (budget, partition) in enumerate(zip(budgets, partitions)):
            for idx in partition:
                element_eps[idx] = element_eps.get(idx, 0.0) + budget.epsilon
                element_delta[idx] = element_delta.get(idx, 0.0) + budget.delta

        if not element_eps:
            return CompositionResult(
                epsilon=0.0,
                delta=0.0,
                composition_type=CompositionType.BASIC,
                n_mechanisms=len(mechanisms),
            )

        eps_total = max(element_eps.values())
        delta_total = min(max(element_delta.values()), 1.0 - 1e-15)

        return CompositionResult(
            epsilon=eps_total,
            delta=delta_total,
            composition_type=CompositionType.BASIC,
            n_mechanisms=len(mechanisms),
            per_mechanism_budgets=list(budgets),
            metadata={"composition_mode": "overlap", "max_exposure_eps": eps_total},
        )


# =========================================================================
# 3. AdaptiveComposer
# =========================================================================


class AdaptiveComposer:
    """Adaptive budget allocation for sequences of queries.

    In the adaptive composition setting, each query's parameters may depend
    on answers to previous queries.  This composer allocates a total privacy
    budget across queries according to various strategies:

    - **Uniform**: Equal allocation ε_i = ε_total / k.
    - **Exponential decay**: ε_i ∝ r^i for decay rate r < 1, giving more
      budget to early queries.
    - **Optimal (sensitivity-weighted)**: ε_i ∝ 1 / sensitivity_i, giving
      more budget to less-sensitive queries to minimize total error.

    Usage::

        composer = AdaptiveComposer()
        allocation = composer.compose_adaptive(
            mechanism_factory=my_factory,
            n_queries=10,
            total_budget=PrivacyBudget(epsilon=1.0),
        )
    """

    def __init__(
        self,
        composition_method: CompositionType = CompositionType.ADVANCED,
    ) -> None:
        """Initialize adaptive composer.

        Args:
            composition_method: Composition theorem for budget tracking.
        """
        self._composition_method = composition_method
        self._sequential = SequentialComposer(default_method=composition_method)

    def compose_adaptive(
        self,
        mechanism_factory: Optional[Callable] = None,
        n_queries: int = 1,
        total_budget: Optional[PrivacyBudget] = None,
        allocation_strategy: str = "uniform",
        sensitivities: Optional[FloatArray] = None,
    ) -> AllocationResult:
        """Allocate a total privacy budget across adaptive queries.

        Args:
            mechanism_factory: Callable that creates a mechanism given epsilon.
                Not invoked here; stored in metadata for downstream use.
            n_queries: Number of queries to allocate budget for.
            total_budget: Total privacy budget to distribute.
            allocation_strategy: One of 'uniform', 'exponential_decay', 'optimal'.
            sensitivities: Per-query sensitivities for optimal allocation.

        Returns:
            AllocationResult with per-query budgets.

        Raises:
            ConfigurationError: If total_budget is None or n_queries < 1.
        """
        if total_budget is None:
            raise ConfigurationError(
                "total_budget must be provided",
                parameter="total_budget",
            )
        if n_queries < 1:
            raise ConfigurationError(
                f"n_queries must be >= 1, got {n_queries}",
                parameter="n_queries",
                value=n_queries,
            )

        total_eps = total_budget.epsilon
        total_delta = total_budget.delta

        if allocation_strategy == "uniform":
            eps_alloc = self.uniform_allocation(total_eps, n_queries)
        elif allocation_strategy == "exponential_decay":
            eps_alloc = self.exponential_decay_allocation(total_eps, n_queries)
        elif allocation_strategy == "optimal":
            if sensitivities is None:
                sensitivities = np.ones(n_queries, dtype=np.float64)
            eps_alloc = self.optimal_allocation(total_eps, n_queries, sensitivities)
        else:
            raise ConfigurationError(
                f"Unknown allocation strategy: {allocation_strategy!r}",
                parameter="allocation_strategy",
                value=allocation_strategy,
            )

        delta_alloc = np.full(n_queries, total_delta / n_queries, dtype=np.float64)

        return AllocationResult(
            epsilons=eps_alloc,
            deltas=delta_alloc,
            total_epsilon=total_eps,
            total_delta=total_delta,
            metadata={
                "strategy": allocation_strategy,
                "composition_method": self._composition_method.name,
            },
        )

    def exponential_decay_allocation(
        self,
        total_eps: float,
        n_queries: int,
        decay_rate: float = 0.9,
    ) -> FloatArray:
        """Exponentially decaying budget allocation.

        Allocates ε_i = c · r^i where r is the decay rate and c is chosen
        so that Σε_i = total_eps.

        Early queries receive more budget, useful when early queries are
        more informative or when exploration is front-loaded.

        Args:
            total_eps: Total epsilon to distribute.
            n_queries: Number of queries.
            decay_rate: Decay factor r ∈ (0, 1). Closer to 1 is more uniform.

        Returns:
            Array of per-query epsilon allocations.
        """
        _validate_positive(total_eps, "total_eps")
        if not (0.0 < decay_rate < 1.0):
            raise ConfigurationError(
                f"decay_rate must be in (0, 1), got {decay_rate}",
                parameter="decay_rate",
                value=decay_rate,
            )

        weights = np.array([decay_rate ** i for i in range(n_queries)], dtype=np.float64)
        weights /= weights.sum()
        return weights * total_eps

    def uniform_allocation(
        self,
        total_eps: float,
        n_queries: int,
    ) -> FloatArray:
        """Equal budget allocation across all queries.

        Each query receives ε_i = total_eps / n_queries.

        Args:
            total_eps: Total epsilon to distribute.
            n_queries: Number of queries.

        Returns:
            Array of per-query epsilon allocations.
        """
        _validate_positive(total_eps, "total_eps")
        return np.full(n_queries, total_eps / n_queries, dtype=np.float64)

    def optimal_allocation(
        self,
        total_eps: float,
        n_queries: int,
        sensitivities: FloatArray,
    ) -> FloatArray:
        """Sensitivity-weighted optimal budget allocation.

        Minimizes total expected squared error Σ(sensitivity_i / ε_i)²
        subject to Σε_i ≤ total_eps.

        By Lagrangian optimality, the optimal allocation is:
            ε_i = total_eps × sensitivity_i^{2/3} / Σ sensitivity_j^{2/3}

        This gives more budget to more-sensitive queries, since their error
        scales more steeply with 1/ε.

        Args:
            total_eps: Total epsilon to distribute.
            n_queries: Number of queries.
            sensitivities: Per-query sensitivity values.

        Returns:
            Array of per-query epsilon allocations.

        Raises:
            ConfigurationError: If sensitivities has wrong length or contains
                non-positive values.
        """
        _validate_positive(total_eps, "total_eps")
        sensitivities = np.asarray(sensitivities, dtype=np.float64)

        if len(sensitivities) != n_queries:
            raise ConfigurationError(
                f"sensitivities length ({len(sensitivities)}) must match "
                f"n_queries ({n_queries})",
                parameter="sensitivities",
            )
        if np.any(sensitivities <= 0):
            raise ConfigurationError(
                "All sensitivities must be positive",
                parameter="sensitivities",
            )

        # Optimal allocation: ε_i ∝ s_i^{2/3} (for L2 loss with Laplace noise)
        weights = sensitivities ** (2.0 / 3.0)
        weights /= weights.sum()
        return weights * total_eps


# =========================================================================
# 4. SubsampledComposer
# =========================================================================


class SubsampledComposer:
    """Privacy amplification by subsampling.

    When a mechanism is applied to a random subsample of the data, the
    privacy guarantee is amplified.  This class implements amplification
    formulas for Poisson subsampling and without-replacement sampling.

    Key results:
        - **Poisson subsampling** (Balle, Barthe, Gavin 2018):
          An (ε, δ)-DP mechanism applied to a Poisson subsample with rate q
          satisfies (ε', qδ + (1-q)·0)-DP where ε' ≈ log(1 + q(e^ε − 1)).
        - **Without-replacement** (Balle, Barthe, Gavin 2018):
          For batch of size b from n records, effective q ≈ b/n with
          tighter bounds than Poisson.

    Usage::

        composer = SubsampledComposer()
        amplified_budget = composer.poisson_subsampling(eps=1.0, delta=0.0, q=0.01)
    """

    def amplify_by_subsampling(
        self,
        mechanism_budget: PrivacyBudget,
        sampling_rate: float,
        method: str = "poisson",
        n: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> PrivacyBudget:
        """Apply subsampling amplification to a mechanism's privacy budget.

        Args:
            mechanism_budget: Privacy budget of the base mechanism.
            sampling_rate: Subsampling rate q ∈ (0, 1).
            method: 'poisson' or 'without_replacement'.
            n: Total number of records (for without-replacement).
            batch_size: Batch size (for without-replacement).

        Returns:
            Amplified PrivacyBudget with tighter guarantees.
        """
        _validate_probability(sampling_rate, "sampling_rate")

        if method == "poisson":
            eps_amp, delta_amp = self.poisson_subsampling(
                mechanism_budget.epsilon, mechanism_budget.delta, sampling_rate
            )
        elif method == "without_replacement":
            if n is None or batch_size is None:
                raise ConfigurationError(
                    "n and batch_size required for without_replacement subsampling",
                    parameter="n/batch_size",
                )
            eps_amp, delta_amp = self.without_replacement_subsampling(
                mechanism_budget.epsilon, mechanism_budget.delta, n, batch_size
            )
        else:
            raise ConfigurationError(
                f"Unknown subsampling method: {method!r}",
                parameter="method",
                value=method,
            )

        return PrivacyBudget(
            epsilon=max(eps_amp, 1e-15),
            delta=delta_amp,
        )

    def poisson_subsampling(
        self,
        eps: float,
        delta: float,
        q: float,
    ) -> Tuple[float, float]:
        """Privacy amplification by Poisson subsampling.

        Each record is included independently with probability q.
        The amplified guarantee is:
            ε' = log(1 + q · (e^ε − 1))
            δ' = q · δ

        This is the first-order bound from Balle, Barthe, Gavin (2018).
        For small q, this gives ε' ≈ q · ε, providing significant
        amplification.

        Args:
            eps: Base mechanism's epsilon.
            delta: Base mechanism's delta.
            q: Sampling probability ∈ (0, 1).

        Returns:
            Tuple of (amplified_epsilon, amplified_delta).
        """
        _validate_positive(eps, "eps")
        _validate_non_negative(delta, "delta")
        _validate_probability(q, "q")

        # ε' = log(1 + q(e^ε - 1))
        # Use log1p for numerical stability when q is small
        try:
            expm1_eps = math.expm1(eps)  # e^ε - 1
            eps_amplified = math.log1p(q * expm1_eps)
        except OverflowError:
            # For very large ε, the amplification is bounded by ε
            eps_amplified = eps + math.log(q)

        delta_amplified = q * delta

        return eps_amplified, delta_amplified

    def without_replacement_subsampling(
        self,
        eps: float,
        delta: float,
        n: int,
        batch_size: int,
    ) -> Tuple[float, float]:
        """Privacy amplification by without-replacement subsampling.

        A random batch of size b is drawn from n records without replacement.
        The effective sampling rate is q = b/n, and the amplification is
        slightly tighter than Poisson due to negative correlation.

        Uses the bound from Balle, Barthe, Gavin (2018) Theorem 9:
            ε' ≤ log(1 + (b/n) · (e^ε − 1))
        with a tighter correction for small populations.

        Args:
            eps: Base mechanism's epsilon.
            delta: Base mechanism's delta.
            n: Total number of records.
            batch_size: Number of records in each batch.

        Returns:
            Tuple of (amplified_epsilon, amplified_delta).

        Raises:
            ConfigurationError: If batch_size > n or either is non-positive.
        """
        _validate_positive(eps, "eps")
        _validate_non_negative(delta, "delta")

        if n < 1:
            raise ConfigurationError(
                f"n must be >= 1, got {n}",
                parameter="n",
                value=n,
            )
        if batch_size < 1:
            raise ConfigurationError(
                f"batch_size must be >= 1, got {batch_size}",
                parameter="batch_size",
                value=batch_size,
            )
        if batch_size > n:
            raise ConfigurationError(
                f"batch_size ({batch_size}) must be <= n ({n})",
                parameter="batch_size",
                value=batch_size,
            )

        q = batch_size / n

        # Use the Poisson bound as a starting point
        eps_poisson, delta_poisson = self.poisson_subsampling(eps, delta, q)

        # Without-replacement correction: tighter by a factor
        # For large n, the correction is small.
        # Use the bound: ε' ≤ log(1 + (n-b)/(n-1) · q · (e^ε - 1))
        if n > 1:
            correction_factor = (n - batch_size) / (n - 1)
        else:
            correction_factor = 1.0

        try:
            expm1_eps = math.expm1(eps)
            eps_amplified = math.log1p(correction_factor * q * expm1_eps)
        except OverflowError:
            eps_amplified = eps_poisson

        eps_amplified = min(eps_amplified, eps_poisson)
        delta_amplified = q * delta

        return eps_amplified, delta_amplified

    def compute_effective_epsilon(
        self,
        target_eps: float,
        sampling_rate: float,
    ) -> float:
        """Compute the base mechanism epsilon needed to achieve target_eps after subsampling.

        Given a target post-amplification epsilon and a sampling rate,
        find the base epsilon that, after Poisson subsampling amplification,
        yields the target.

        Solves: target_eps = log(1 + q · (e^base_eps − 1))
        => e^target_eps − 1 = q · (e^base_eps − 1)
        => base_eps = log(1 + (e^target_eps − 1) / q)

        Args:
            target_eps: Desired post-amplification epsilon.
            sampling_rate: Poisson subsampling rate q.

        Returns:
            Required base mechanism epsilon.
        """
        _validate_positive(target_eps, "target_eps")
        _validate_probability(sampling_rate, "sampling_rate")

        try:
            expm1_target = math.expm1(target_eps)
            base_eps = math.log1p(expm1_target / sampling_rate)
        except OverflowError:
            base_eps = target_eps - math.log(sampling_rate)

        return base_eps


# =========================================================================
# 5. ComposedMechanism
# =========================================================================


@dataclass
class ComposedMechanism:
    """A mechanism formed by composing multiple sub-mechanisms.

    This class wraps a sequence of sub-mechanisms and a composition result,
    providing a unified interface for sampling, privacy querying, and
    analysis of the composed pipeline.

    Attributes:
        mechanisms: List of sub-mechanism objects (each must support ``sample``).
        composition_type: Composition theorem used.
        privacy_params: Pre-computed privacy parameters from composition.
        output_grids: Per-mechanism output grids.
        metadata: Additional composition metadata.
    """

    mechanisms: List[Any]
    composition_type: CompositionType
    privacy_params: PrivacyBudget
    output_grids: Optional[List[FloatArray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mechanisms:
            raise InvalidMechanismError(
                "ComposedMechanism requires at least one sub-mechanism",
                reason="empty mechanism list",
            )

    @property
    def n_sub_mechanisms(self) -> int:
        """Number of sub-mechanisms in the composition."""
        return len(self.mechanisms)

    def sample(
        self,
        true_values: Union[float, FloatArray],
        rng: Optional[np.random.Generator] = None,
    ) -> List[FloatArray]:
        """Apply all sub-mechanisms to the true values.

        Each sub-mechanism is applied independently to the true values.
        For sequential composition, all mechanisms see the same true values.
        For parallel composition, true_values should be partitioned externally.

        Args:
            true_values: True query answers. Shape depends on sub-mechanisms.
            rng: Random number generator for reproducibility.

        Returns:
            List of noisy outputs, one per sub-mechanism.
        """
        if rng is None:
            rng = np.random.default_rng()

        results = []
        for mech in self.mechanisms:
            if hasattr(mech, "sample"):
                noisy = mech.sample(true_values, rng=rng)
            elif hasattr(mech, "p_final"):
                # ExtractedMechanism: sample using CDF
                noisy = self._sample_from_table(mech, true_values, rng)
            else:
                raise InvalidMechanismError(
                    f"Sub-mechanism {type(mech).__name__} has no sample method",
                    reason="missing sample method",
                )
            results.append(np.asarray(noisy, dtype=np.float64))

        return results

    def _sample_from_table(
        self,
        mech: Any,
        true_values: Union[float, FloatArray],
        rng: np.random.Generator,
    ) -> FloatArray:
        """Sample from an ExtractedMechanism's probability table.

        For each true value index i, sample an output bin j according to
        the probability row p_final[i, :], then return the output grid value.

        Args:
            mech: An object with p_final attribute (ExtractedMechanism).
            true_values: True values (interpreted as integer row indices).
            rng: Random number generator.

        Returns:
            Array of noisy outputs.
        """
        true_values = np.atleast_1d(np.asarray(true_values, dtype=np.int64))
        p_table = mech.p_final
        n, k = p_table.shape
        outputs = np.empty(len(true_values), dtype=np.float64)

        for idx, tv in enumerate(true_values):
            row_idx = int(tv) % n
            probs = p_table[row_idx]
            j = rng.choice(k, p=probs / probs.sum())
            outputs[idx] = float(j)

        return outputs

    def total_epsilon(self) -> float:
        """Total epsilon after composition.

        Returns:
            The composed epsilon from the privacy parameters.
        """
        return self.privacy_params.epsilon

    def total_delta(self) -> float:
        """Total delta after composition.

        Returns:
            The composed delta from the privacy parameters.
        """
        return self.privacy_params.delta

    def privacy_curve(
        self,
        delta_range: Optional[FloatArray] = None,
    ) -> Tuple[FloatArray, FloatArray]:
        """Compute the full (ε, δ) privacy curve of the composed mechanism.

        For each target delta, computes the tightest epsilon achievable
        using the composition theorem stored in ``composition_type``.

        Args:
            delta_range: Array of delta values to evaluate. Defaults to
                a logarithmic grid from 1e-10 to 0.1.

        Returns:
            Tuple of (delta_array, epsilon_array) defining the curve.
        """
        if delta_range is None:
            delta_range = np.logspace(-10, -1, 100)

        composer = SequentialComposer()
        eps_curve = np.empty_like(delta_range)

        # Extract per-mechanism budgets from metadata or privacy_params
        per_budgets = self.metadata.get("per_mechanism_budgets", None)
        if per_budgets is None:
            # Assume homogeneous: split total budget evenly
            n = self.n_sub_mechanisms
            per_eps = self.privacy_params.epsilon / n
            eps_list = np.full(n, per_eps)
        else:
            eps_list = np.array([b.epsilon for b in per_budgets])

        for i, delta_target in enumerate(delta_range):
            if delta_target <= 0 or delta_target >= 1:
                eps_curve[i] = float("inf")
                continue

            if self.composition_type == CompositionType.RDP:
                eps_i, _ = composer.rdp_composition(
                    eps_list, target_delta=delta_target
                )
            elif self.composition_type == CompositionType.ZERO_CDP:
                eps_i, _ = composer.zcdp_composition(
                    eps_list, target_delta=delta_target
                )
            elif self.composition_type == CompositionType.ADVANCED:
                eps_i, _ = composer.advanced_composition(
                    eps_list, delta_prime=delta_target
                )
            else:
                eps_i, _ = composer.basic_composition(eps_list)

            eps_curve[i] = eps_i

        return delta_range, eps_curve

    def decompose(self) -> List[Any]:
        """Return the list of sub-mechanisms.

        Returns:
            List of sub-mechanism objects.
        """
        return list(self.mechanisms)

    def __repr__(self) -> str:
        dp = f"ε={self.privacy_params.epsilon:.4f}"
        if self.privacy_params.delta > 0:
            dp += f", δ={self.privacy_params.delta:.2e}"
        return (
            f"ComposedMechanism(n_sub={self.n_sub_mechanisms}, {dp}, "
            f"type={self.composition_type.name})"
        )


# =========================================================================
# 6. CompositionOptimizer
# =========================================================================


class CompositionOptimizer:
    """Optimize budget allocation across composed mechanisms.

    Given a total privacy budget and a set of queries with known
    sensitivities, find the per-query budget allocation that minimizes
    a given loss function (e.g., total MSE or maximum MSE).

    This class implements several optimization strategies:
    - **Sensitivity-weighted**: Analytical optimal for L2 loss with Laplace.
    - **Numerical optimization**: Scipy-based optimization for general losses.
    - **Pareto frontier**: Multi-objective trade-off between queries.

    Usage::

        optimizer = CompositionOptimizer()
        allocation = optimizer.optimize_budget_allocation(
            n_queries=5,
            total_budget=PrivacyBudget(epsilon=1.0),
            sensitivities=np.array([1.0, 2.0, 1.0, 3.0, 1.0]),
            loss_fn='mse',
        )
    """

    def __init__(
        self,
        composition_method: CompositionType = CompositionType.ADVANCED,
    ) -> None:
        """Initialize optimizer.

        Args:
            composition_method: Composition theorem for budget accounting.
        """
        self._composition_method = composition_method
        self._sequential = SequentialComposer(default_method=composition_method)

    def optimize_budget_allocation(
        self,
        n_queries: int,
        total_budget: PrivacyBudget,
        sensitivities: Optional[FloatArray] = None,
        loss_fn: str = "mse",
        composition_type: Optional[CompositionType] = None,
    ) -> AllocationResult:
        """Find optimal per-query budget allocation minimizing total loss.

        For L2 loss with Laplace mechanisms under basic composition,
        the MSE of query i is 2 · (sensitivity_i / ε_i)².  The total MSE
        is minimized by the sensitivity-weighted allocation.

        For general losses, we use numerical optimization via scipy.

        Args:
            n_queries: Number of queries.
            total_budget: Total privacy budget.
            sensitivities: Per-query sensitivity values. Defaults to all 1.
            loss_fn: Loss function name ('mse', 'mae', 'max_mse').
            composition_type: Override for composition theorem.

        Returns:
            AllocationResult with optimal per-query budgets.
        """
        if n_queries < 1:
            raise ConfigurationError(
                f"n_queries must be >= 1, got {n_queries}",
                parameter="n_queries",
                value=n_queries,
            )

        if sensitivities is None:
            sensitivities = np.ones(n_queries, dtype=np.float64)
        else:
            sensitivities = np.asarray(sensitivities, dtype=np.float64)

        if len(sensitivities) != n_queries:
            raise ConfigurationError(
                f"sensitivities length ({len(sensitivities)}) != n_queries ({n_queries})",
                parameter="sensitivities",
            )

        comp = composition_type or self._composition_method

        if loss_fn == "mse" and comp == CompositionType.BASIC:
            return self._analytical_mse_allocation(
                n_queries, total_budget, sensitivities
            )
        elif loss_fn == "max_mse" and comp == CompositionType.BASIC:
            return self._minimax_allocation(
                n_queries, total_budget, sensitivities
            )
        else:
            return self._numerical_allocation(
                n_queries, total_budget, sensitivities, loss_fn
            )

    def _analytical_mse_allocation(
        self,
        n_queries: int,
        total_budget: PrivacyBudget,
        sensitivities: FloatArray,
    ) -> AllocationResult:
        """Closed-form MSE-optimal allocation under basic composition.

        Total MSE = Σ 2(s_i/ε_i)² is minimized by:
            ε_i = total_eps × s_i^{2/3} / Σ s_j^{2/3}

        This follows from the Lagrangian KKT conditions.

        Args:
            n_queries: Number of queries.
            total_budget: Total budget.
            sensitivities: Per-query sensitivities.

        Returns:
            AllocationResult.
        """
        composer = AdaptiveComposer(composition_method=CompositionType.BASIC)
        eps_alloc = composer.optimal_allocation(
            total_budget.epsilon, n_queries, sensitivities
        )
        delta_alloc = np.full(n_queries, total_budget.delta / n_queries, dtype=np.float64)

        # Compute objective: total MSE = Σ 2(s_i/ε_i)²
        mse_values = 2.0 * (sensitivities / eps_alloc) ** 2
        total_mse = float(np.sum(mse_values))

        return AllocationResult(
            epsilons=eps_alloc,
            deltas=delta_alloc,
            total_epsilon=total_budget.epsilon,
            total_delta=total_budget.delta,
            objective_value=total_mse,
            metadata={"loss_fn": "mse", "per_query_mse": mse_values.tolist()},
        )

    def _minimax_allocation(
        self,
        n_queries: int,
        total_budget: PrivacyBudget,
        sensitivities: FloatArray,
    ) -> AllocationResult:
        """Minimax allocation: minimize the maximum per-query MSE.

        Minimize max_i 2(s_i/ε_i)² subject to Σε_i = total_eps.

        Optimal: ε_i ∝ s_i (proportional to sensitivity), so that
        all per-query MSEs are equalized.

        Args:
            n_queries: Number of queries.
            total_budget: Total budget.
            sensitivities: Per-query sensitivities.

        Returns:
            AllocationResult.
        """
        total_eps = total_budget.epsilon

        # Equalize MSEs: 2(s_i/ε_i)² = constant ⟹ ε_i ∝ s_i
        weights = sensitivities / sensitivities.sum()
        eps_alloc = weights * total_eps
        delta_alloc = np.full(n_queries, total_budget.delta / n_queries, dtype=np.float64)

        mse_values = 2.0 * (sensitivities / eps_alloc) ** 2
        max_mse = float(np.max(mse_values))

        return AllocationResult(
            epsilons=eps_alloc,
            deltas=delta_alloc,
            total_epsilon=total_eps,
            total_delta=total_budget.delta,
            objective_value=max_mse,
            metadata={"loss_fn": "max_mse", "per_query_mse": mse_values.tolist()},
        )

    def _numerical_allocation(
        self,
        n_queries: int,
        total_budget: PrivacyBudget,
        sensitivities: FloatArray,
        loss_fn: str,
    ) -> AllocationResult:
        """Numerical optimization for general loss functions.

        Uses scipy.optimize.minimize with SLSQP to find the optimal
        budget allocation under a general loss function and composition
        theorem.

        Args:
            n_queries: Number of queries.
            total_budget: Total budget.
            sensitivities: Per-query sensitivities.
            loss_fn: Name of the loss function.

        Returns:
            AllocationResult.
        """
        from scipy.optimize import minimize

        total_eps = total_budget.epsilon

        # Define loss: MSE for each query under Laplace is 2(s/ε)²
        # MAE is √2 · s/ε
        if loss_fn == "mse":
            def loss(eps_arr: FloatArray) -> float:
                return float(np.sum(2.0 * (sensitivities / eps_arr) ** 2))
        elif loss_fn == "mae":
            def loss(eps_arr: FloatArray) -> float:
                return float(np.sum(math.sqrt(2) * sensitivities / eps_arr))
        elif loss_fn == "max_mse":
            def loss(eps_arr: FloatArray) -> float:
                return float(np.max(2.0 * (sensitivities / eps_arr) ** 2))
        else:
            raise ConfigurationError(
                f"Unknown loss function: {loss_fn!r}",
                parameter="loss_fn",
                value=loss_fn,
            )

        # Initial guess: uniform allocation
        x0 = np.full(n_queries, total_eps / n_queries)

        # Constraint: sum of epsilons <= total_eps
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - total_eps},
        ]

        # Bounds: each epsilon > 0
        eps_lb = 1e-10
        bounds = [(eps_lb, total_eps)] * n_queries

        result = minimize(
            loss,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )

        eps_alloc = result.x
        delta_alloc = np.full(n_queries, total_budget.delta / n_queries, dtype=np.float64)

        return AllocationResult(
            epsilons=eps_alloc,
            deltas=delta_alloc,
            total_epsilon=total_eps,
            total_delta=total_budget.delta,
            objective_value=float(result.fun),
            metadata={
                "loss_fn": loss_fn,
                "optimizer_success": result.success,
                "optimizer_message": result.message,
            },
        )

    def pareto_frontier(
        self,
        n_queries: int,
        total_budget: PrivacyBudget,
        sensitivities: Optional[FloatArray] = None,
        n_points: int = 50,
    ) -> List[AllocationResult]:
        """Compute the Pareto frontier of privacy-utility trade-offs.

        Sweeps across different total epsilon values from a fraction to the
        full budget, computing the optimal allocation at each point.

        Args:
            n_queries: Number of queries.
            total_budget: Maximum total privacy budget.
            sensitivities: Per-query sensitivities.
            n_points: Number of points on the frontier.

        Returns:
            List of AllocationResults along the Pareto frontier.
        """
        if sensitivities is None:
            sensitivities = np.ones(n_queries, dtype=np.float64)

        frontier = []
        eps_values = np.linspace(
            total_budget.epsilon * 0.1,
            total_budget.epsilon,
            n_points,
        )

        for eps in eps_values:
            budget = PrivacyBudget(
                epsilon=float(eps),
                delta=total_budget.delta,
            )
            allocation = self.optimize_budget_allocation(
                n_queries=n_queries,
                total_budget=budget,
                sensitivities=sensitivities,
                loss_fn="mse",
            )
            frontier.append(allocation)

        return frontier

    def sensitivity_weighted_allocation(
        self,
        n_queries: int,
        total_budget: PrivacyBudget,
        sensitivities: FloatArray,
    ) -> AllocationResult:
        """Convenience method for sensitivity-weighted allocation.

        Wraps :meth:`optimize_budget_allocation` with ``loss_fn='mse'``.

        Args:
            n_queries: Number of queries.
            total_budget: Total privacy budget.
            sensitivities: Per-query sensitivity values.

        Returns:
            AllocationResult with sensitivity-weighted budgets.
        """
        return self.optimize_budget_allocation(
            n_queries=n_queries,
            total_budget=total_budget,
            sensitivities=sensitivities,
            loss_fn="mse",
        )

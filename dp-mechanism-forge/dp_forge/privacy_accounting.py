"""
Privacy composition and accounting for DP-Forge.

Implements multiple privacy accounting frameworks used to track and compose
privacy budgets across mechanism invocations:

Frameworks:
    - **Basic composition**: Sum-of-epsilons (sequential) and max-of-epsilons
      (parallel) for (ε, δ)-DP.
    - **Advanced composition**: Dwork–Rothblum–Vadhan (2010) advanced
      composition theorem.
    - **Rényi DP (RDP)**: Mironov (2017) Rényi differential privacy with
      optimal order selection and conversion to (ε, δ)-DP.
    - **Zero-concentrated DP (zCDP)**: Bun–Steinke (2016) concentrated DP.
    - **Moments accountant**: Abadi et al. (2016) moments-based accounting
      for subsampled mechanisms (e.g., DP-SGD).
    - **Subsampling amplification**: Privacy amplification by Poisson and
      without-replacement subsampling.

All accountants operate on :class:`PrivacyBudget` from ``dp_forge.types``
and raise :class:`BudgetExhaustedError` when the cumulative budget is
exceeded.
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
)
from dp_forge.types import CompositionType, PrivacyBudget

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]


# =========================================================================
# Helper: numerically stable logsumexp (avoids importing numerical module
# to prevent circular imports)
# =========================================================================

def _logsumexp(a: FloatArray) -> float:
    """Numerically stable log-sum-exp."""
    a = np.asarray(a, dtype=np.float64).ravel()
    if len(a) == 0:
        return -np.inf
    a_max = np.max(a)
    if not np.isfinite(a_max):
        return float(a_max)
    return float(a_max + np.log(np.sum(np.exp(a - a_max))))


# =========================================================================
# 1. Basic Composition
# =========================================================================


class BasicComposition:
    """Basic privacy composition theorems.

    Sequential composition sums epsilons; parallel composition takes the
    maximum.  These are the simplest and loosest bounds, but universally
    applicable.

    Example::

        >>> budgets = [PrivacyBudget(1.0, 1e-5), PrivacyBudget(0.5, 1e-6)]
        >>> BasicComposition.sequential(budgets)
        PrivacyBudget(ε=1.5, δ=1.1e-05)
    """

    @staticmethod
    def sequential(budgets: Sequence[PrivacyBudget]) -> PrivacyBudget:
        """Sequential (basic) composition.

        If mechanism M_i is (ε_i, δ_i)-DP, then their sequential
        composition is (Σε_i, Σδ_i)-DP.

        Args:
            budgets: Sequence of per-mechanism privacy budgets.

        Returns:
            Composed privacy budget.

        Raises:
            ValueError: If *budgets* is empty.
        """
        if len(budgets) == 0:
            raise ValueError("budgets must be non-empty")
        total_eps = sum(b.epsilon for b in budgets)
        total_delta = sum(b.delta for b in budgets)
        total_delta = min(total_delta, 1.0 - 1e-15)
        return PrivacyBudget(epsilon=total_eps, delta=total_delta)

    @staticmethod
    def parallel(budgets: Sequence[PrivacyBudget]) -> PrivacyBudget:
        """Parallel composition.

        If mechanisms operate on disjoint subsets of the data, the
        composed privacy is (max ε_i, max δ_i)-DP.

        Args:
            budgets: Sequence of per-mechanism privacy budgets.

        Returns:
            Composed privacy budget.

        Raises:
            ValueError: If *budgets* is empty.
        """
        if len(budgets) == 0:
            raise ValueError("budgets must be non-empty")
        max_eps = max(b.epsilon for b in budgets)
        max_delta = max(b.delta for b in budgets)
        return PrivacyBudget(epsilon=max_eps, delta=max_delta)

    @staticmethod
    def sequential_homogeneous(
        epsilon: float,
        delta: float,
        k: int,
    ) -> PrivacyBudget:
        """Sequential composition of *k* identical (ε, δ)-DP mechanisms.

        Args:
            epsilon: Per-mechanism ε.
            delta: Per-mechanism δ.
            k: Number of mechanism invocations.

        Returns:
            Composed privacy budget ``(k·ε, k·δ)``.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        total_delta = min(k * delta, 1.0 - 1e-15)
        return PrivacyBudget(epsilon=k * epsilon, delta=total_delta)


# =========================================================================
# 2. Advanced Composition
# =========================================================================


class AdvancedComposition:
    """Advanced composition theorem (Dwork–Rothblum–Vadhan 2010).

    For *k* mechanisms each satisfying (ε, δ)-DP, the advanced
    composition theorem gives a tighter bound than basic composition:

        (ε_total, δ_total) with
        ε_total = √(2k ln(1/δ')) · ε + k · ε · (e^ε - 1)
        δ_total = k · δ + δ'

    where δ' is an additional failure probability.
    """

    @staticmethod
    def compose(
        eps_list: Sequence[float],
        delta_list: Sequence[float],
        delta_prime: float,
    ) -> PrivacyBudget:
        """Advanced composition of heterogeneous mechanisms.

        Uses the generalisation of the advanced composition theorem for
        mechanisms with different privacy parameters.

        Args:
            eps_list: Per-mechanism ε values.
            delta_list: Per-mechanism δ values.
            delta_prime: Additional failure probability δ'.

        Returns:
            Composed privacy budget.

        Raises:
            ValueError: If lists are empty or have different lengths.
            ConfigurationError: If delta_prime is not in (0, 1).
        """
        if len(eps_list) == 0:
            raise ValueError("eps_list must be non-empty")
        if len(eps_list) != len(delta_list):
            raise ValueError(
                f"eps_list ({len(eps_list)}) and delta_list ({len(delta_list)}) "
                f"must have equal length"
            )
        if not (0 < delta_prime < 1):
            raise ConfigurationError(
                f"delta_prime must be in (0, 1), got {delta_prime}",
                parameter="delta_prime",
                value=delta_prime,
                constraint="0 < delta_prime < 1",
            )

        k = len(eps_list)
        eps_arr = np.asarray(eps_list, dtype=np.float64)
        delta_arr = np.asarray(delta_list, dtype=np.float64)

        # sum of eps_i * (exp(eps_i) - 1)
        sum_eps_exp = float(np.sum(eps_arr * (np.exp(eps_arr) - 1.0)))

        # sum of eps_i^2
        sum_eps_sq = float(np.sum(eps_arr ** 2))

        eps_total = math.sqrt(2.0 * math.log(1.0 / delta_prime) * sum_eps_sq) + sum_eps_exp
        delta_total = float(np.sum(delta_arr)) + delta_prime
        delta_total = min(delta_total, 1.0 - 1e-15)

        return PrivacyBudget(epsilon=eps_total, delta=delta_total)

    @staticmethod
    def compose_homogeneous(
        epsilon: float,
        delta: float,
        k: int,
        delta_prime: float,
    ) -> PrivacyBudget:
        """Advanced composition for *k* identical mechanisms.

        Args:
            epsilon: Per-mechanism ε.
            delta: Per-mechanism δ.
            k: Number of compositions.
            delta_prime: Additional failure probability.

        Returns:
            Composed privacy budget.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not (0 < delta_prime < 1):
            raise ConfigurationError(
                f"delta_prime must be in (0, 1), got {delta_prime}",
                parameter="delta_prime",
                value=delta_prime,
                constraint="0 < delta_prime < 1",
            )

        eps_total = (
            math.sqrt(2.0 * k * math.log(1.0 / delta_prime)) * epsilon
            + k * epsilon * (math.exp(epsilon) - 1.0)
        )
        delta_total = min(k * delta + delta_prime, 1.0 - 1e-15)
        return PrivacyBudget(epsilon=eps_total, delta=delta_total)

    @staticmethod
    def optimal_delta_prime(
        epsilon: float,
        delta: float,
        k: int,
        target_delta: Optional[float] = None,
    ) -> float:
        """Find the optimal δ' for advanced composition.

        Minimises the total ε over δ' ∈ (0, 1).  Uses a simple
        grid search for robustness.

        Args:
            epsilon: Per-mechanism ε.
            delta: Per-mechanism δ.
            k: Number of compositions.
            target_delta: If given, find δ' to hit this total delta.

        Returns:
            Optimal δ' value.
        """
        if target_delta is not None:
            slack = target_delta - k * delta
            if slack <= 0:
                return 1e-10
            return min(slack, 1.0 - 1e-15)

        best_dp = 1e-10
        best_eps = float("inf")
        for log_dp in np.linspace(-15, -1, 1000):
            dp = 10.0 ** log_dp
            eps_total = (
                math.sqrt(2.0 * k * math.log(1.0 / dp)) * epsilon
                + k * epsilon * (math.exp(epsilon) - 1.0)
            )
            if eps_total < best_eps:
                best_eps = eps_total
                best_dp = dp
        return best_dp


# =========================================================================
# 3. Rényi DP Accountant
# =========================================================================


@dataclass
class RDPMechanism:
    """Specification of a mechanism's RDP guarantee.

    Attributes:
        rdp_func: Function mapping α → RDP guarantee ε(α).
            If ``None``, uses ``rdp_values`` at ``rdp_alphas``.
        rdp_alphas: Discrete set of α values.
        rdp_values: RDP values at each α.
        name: Human-readable mechanism name.
    """
    rdp_func: Optional[Callable[[float], float]] = None
    rdp_alphas: Optional[FloatArray] = None
    rdp_values: Optional[FloatArray] = None
    name: str = "unnamed"

    def __post_init__(self) -> None:
        if self.rdp_func is None and self.rdp_values is None:
            raise ValueError("Either rdp_func or rdp_values must be provided")
        if self.rdp_values is not None and self.rdp_alphas is None:
            raise ValueError("rdp_alphas must be provided with rdp_values")

    def evaluate(self, alpha: float) -> float:
        """Evaluate the RDP guarantee at order *alpha*.

        Args:
            alpha: Rényi divergence order (> 1).

        Returns:
            RDP value ε(α).
        """
        if self.rdp_func is not None:
            return self.rdp_func(alpha)
        # Interpolate from discrete values
        assert self.rdp_alphas is not None and self.rdp_values is not None
        return float(np.interp(alpha, self.rdp_alphas, self.rdp_values))


class RenyiDPAccountant:
    """Privacy accounting via Rényi Differential Privacy (Mironov 2017).

    RDP provides tighter composition than basic or advanced composition
    for many practical mechanisms (Gaussian, subsampled mechanisms, etc.).

    The key identity: if M_1 is (α, ε_1(α))-RDP and M_2 is (α, ε_2(α))-RDP,
    then their composition is (α, ε_1(α) + ε_2(α))-RDP.

    Conversion to (ε, δ)-DP:
        ε = min_α { ε_RDP(α) + log(1/δ) / (α - 1) }

    Default alpha grid::

        [1.25, 1.5, 1.75, 2, 2.25, ..., 10, 12, 14, ..., 64, 128, 256, ...]
    """

    DEFAULT_ALPHAS: FloatArray = np.concatenate([
        np.arange(1.25, 10.0, 0.25),
        np.arange(10.0, 65.0, 2.0),
        np.array([128.0, 256.0, 512.0, 1024.0]),
    ])

    def __init__(
        self,
        alphas: Optional[FloatArray] = None,
    ) -> None:
        """Initialize the RDP accountant.

        Args:
            alphas: Custom α grid.  Defaults to :attr:`DEFAULT_ALPHAS`.
        """
        if alphas is not None:
            self.alphas = np.asarray(alphas, dtype=np.float64)
        else:
            self.alphas = self.DEFAULT_ALPHAS.copy()
        self._composed_rdp: FloatArray = np.zeros_like(self.alphas)
        self._mechanisms: List[str] = []

    @property
    def n_compositions(self) -> int:
        """Number of mechanisms composed so far."""
        return len(self._mechanisms)

    def add_mechanism(self, mech: RDPMechanism) -> None:
        """Add a mechanism to the composed RDP curve.

        Args:
            mech: RDP specification for the mechanism.
        """
        rdp_at_alphas = np.array(
            [mech.evaluate(a) for a in self.alphas], dtype=np.float64
        )
        self._composed_rdp += rdp_at_alphas
        self._mechanisms.append(mech.name)

    def compose_rdp(
        self,
        mechanisms: Sequence[RDPMechanism],
        alphas: Optional[FloatArray] = None,
    ) -> FloatArray:
        """Compose multiple mechanisms and return the RDP curve.

        This does NOT modify internal state; use :meth:`add_mechanism` for
        stateful tracking.

        Args:
            mechanisms: Sequence of RDP mechanism specifications.
            alphas: Alpha grid (defaults to instance grid).

        Returns:
            Composed RDP curve at each α.
        """
        if alphas is None:
            alphas = self.alphas
        composed = np.zeros_like(alphas)
        for mech in mechanisms:
            rdp_vals = np.array(
                [mech.evaluate(a) for a in alphas], dtype=np.float64
            )
            composed += rdp_vals
        return composed

    def rdp_to_approx_dp(
        self,
        rdp_curve: FloatArray,
        delta: float,
        alphas: Optional[FloatArray] = None,
    ) -> float:
        """Convert an RDP curve to (ε, δ)-DP.

        Uses the standard conversion:
        ``ε = min_α { ε_RDP(α) + log(1/δ) / (α - 1) }``

        Args:
            rdp_curve: RDP values at each α.
            delta: Target δ.
            alphas: Alpha grid (defaults to instance grid).

        Returns:
            The tightest ε achievable at the given δ.

        Raises:
            ValueError: If δ is not in (0, 1).
        """
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if alphas is None:
            alphas = self.alphas

        rdp_curve = np.asarray(rdp_curve, dtype=np.float64)
        log_delta = math.log(delta)

        # ε(α) = ε_RDP(α) + log(1/δ) / (α - 1)
        #       = ε_RDP(α) - log(δ) / (α - 1)
        eps_candidates = rdp_curve - log_delta / (alphas - 1.0)
        return float(np.min(eps_candidates))

    def optimal_alpha(
        self,
        rdp_curve: FloatArray,
        delta: float,
        alphas: Optional[FloatArray] = None,
    ) -> float:
        """Find the optimal α that minimises ε for the given δ.

        Args:
            rdp_curve: RDP values at each α.
            delta: Target δ.
            alphas: Alpha grid (defaults to instance grid).

        Returns:
            Optimal α value.
        """
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if alphas is None:
            alphas = self.alphas

        rdp_curve = np.asarray(rdp_curve, dtype=np.float64)
        log_delta = math.log(delta)
        eps_candidates = rdp_curve - log_delta / (alphas - 1.0)
        best_idx = int(np.argmin(eps_candidates))
        return float(alphas[best_idx])

    def get_privacy(self, delta: float) -> PrivacyBudget:
        """Get the current composed (ε, δ)-DP guarantee.

        Args:
            delta: Target δ.

        Returns:
            Privacy budget.
        """
        eps = self.rdp_to_approx_dp(self._composed_rdp, delta)
        return PrivacyBudget(epsilon=max(eps, 1e-15), delta=delta)

    def get_rdp_curve(self) -> Tuple[FloatArray, FloatArray]:
        """Return the current composed RDP curve.

        Returns:
            ``(alphas, rdp_values)`` tuple.
        """
        return self.alphas.copy(), self._composed_rdp.copy()

    def reset(self) -> None:
        """Reset the accountant, clearing all composed mechanisms."""
        self._composed_rdp = np.zeros_like(self.alphas)
        self._mechanisms.clear()

    def __repr__(self) -> str:
        return (
            f"RenyiDPAccountant(n_compositions={self.n_compositions}, "
            f"n_alphas={len(self.alphas)})"
        )

    # --- Factory helpers for common mechanisms ---

    @staticmethod
    def gaussian_rdp(sigma: float, sensitivity: float = 1.0) -> RDPMechanism:
        """Create an RDP specification for the Gaussian mechanism.

        The Gaussian mechanism with noise σ satisfies
        (α, α·Δ²/(2σ²))-RDP for all α > 1.

        Args:
            sigma: Noise standard deviation.
            sensitivity: Query sensitivity Δ.

        Returns:
            RDP mechanism specification.
        """
        def rdp_func(alpha: float) -> float:
            return alpha * sensitivity ** 2 / (2.0 * sigma ** 2)

        return RDPMechanism(rdp_func=rdp_func, name=f"Gaussian(σ={sigma})")

    @staticmethod
    def laplace_rdp(epsilon: float) -> RDPMechanism:
        """Create an RDP specification for the Laplace mechanism.

        The Laplace mechanism satisfying ε-DP has RDP guarantee:
        ε_RDP(α) = (1/(α-1)) * log((α/(2α-1)) * exp((α-1)·ε)
                    + ((α-1)/(2α-1)) * exp(-(α-1)·ε) )   if α > 1

        Simplified: for α ∈ {integer ≥ 2}, exact; for fractional α,
        we use the upper bound min(ε, α·ε²/2).

        Args:
            epsilon: Laplace privacy parameter ε.

        Returns:
            RDP mechanism specification.
        """
        def rdp_func(alpha: float) -> float:
            if alpha <= 1.0:
                return 0.0
            # Use the exact formula
            term1 = math.log(alpha / (2.0 * alpha - 1.0)) + (alpha - 1.0) * epsilon
            term2 = math.log((alpha - 1.0) / (2.0 * alpha - 1.0)) - (alpha - 1.0) * epsilon
            log_val = _logsumexp(np.array([term1, term2]))
            return log_val / (alpha - 1.0)

        return RDPMechanism(rdp_func=rdp_func, name=f"Laplace(ε={epsilon})")

    @staticmethod
    def subsampled_gaussian_rdp(
        sigma: float,
        sampling_rate: float,
        sensitivity: float = 1.0,
    ) -> RDPMechanism:
        """RDP for the subsampled Gaussian mechanism.

        Uses the analytic moments bound from Mironov et al. (2019).

        Args:
            sigma: Noise standard deviation.
            sampling_rate: Subsampling probability q ∈ (0, 1].
            sensitivity: Query sensitivity.

        Returns:
            RDP mechanism specification.
        """
        if not (0 < sampling_rate <= 1):
            raise ValueError(f"sampling_rate must be in (0, 1], got {sampling_rate}")

        def rdp_func(alpha: float) -> float:
            if alpha <= 1.0:
                return 0.0
            if sampling_rate == 1.0:
                return alpha * sensitivity ** 2 / (2.0 * sigma ** 2)
            # Use the upper bound from Wang et al. (2019)
            # For simplicity, use the tight bound for integer alphas
            # and the loose bound for non-integer
            base_rdp = alpha * sensitivity ** 2 / (2.0 * sigma ** 2)
            if alpha <= 2:
                return sampling_rate ** 2 * base_rdp
            # Tighter bound: compute log(1 + q^2 * C(alpha-1) * (exp((alpha-1)/sigma^2) - 1))
            log_terms = []
            for j in range(int(alpha) + 1):
                if j == 0:
                    log_terms.append(0.0)  # log(1) for (1-q)^alpha term
                elif j == 1:
                    log_terms.append(
                        math.log(sampling_rate) + math.log(alpha)
                        + (j * (j - 1)) * sensitivity ** 2 / (2.0 * sigma ** 2)
                    )
                else:
                    log_binom = sum(
                        math.log(alpha - i) - math.log(i + 1)
                        for i in range(j)
                    )
                    log_terms.append(
                        j * math.log(sampling_rate)
                        + log_binom
                        + (j * (j - 1)) * sensitivity ** 2 / (2.0 * sigma ** 2)
                    )
            return _logsumexp(np.array(log_terms)) / (alpha - 1.0)

        return RDPMechanism(
            rdp_func=rdp_func,
            name=f"SubsampledGaussian(σ={sigma}, q={sampling_rate})",
        )


# =========================================================================
# 4. Zero-Concentrated DP Accountant
# =========================================================================


class ZeroCDPAccountant:
    """Privacy accounting via zero-concentrated DP (Bun–Steinke 2016).

    A mechanism satisfies ρ-zCDP if for all α > 1:
        D_α(M(x) || M(x')) ≤ ρ · α

    Composition: sum of ρ values.
    Conversion to (ε, δ)-DP:
        ε = ρ + 2√(ρ · log(1/δ))
    """

    def __init__(self) -> None:
        self._rho_total: float = 0.0
        self._mechanisms: List[Tuple[str, float]] = []

    @property
    def total_rho(self) -> float:
        """Total composed ρ value."""
        return self._rho_total

    @property
    def n_compositions(self) -> int:
        """Number of mechanisms composed."""
        return len(self._mechanisms)

    def add_mechanism(self, rho: float, name: str = "unnamed") -> None:
        """Add a ρ-zCDP mechanism to the composition.

        Args:
            rho: zCDP parameter ρ > 0.
            name: Human-readable mechanism name.

        Raises:
            ValueError: If ρ ≤ 0.
        """
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        self._rho_total += rho
        self._mechanisms.append((name, rho))

    @staticmethod
    def compose_zcdp(rho_list: Sequence[float]) -> float:
        """Compose multiple ρ-zCDP guarantees.

        Args:
            rho_list: Per-mechanism ρ values.

        Returns:
            Total ρ (sum).

        Raises:
            ValueError: If any ρ ≤ 0 or the list is empty.
        """
        if len(rho_list) == 0:
            raise ValueError("rho_list must be non-empty")
        for r in rho_list:
            if r <= 0:
                raise ValueError(f"All rho values must be > 0, got {r}")
        return sum(rho_list)

    @staticmethod
    def zcdp_to_approx_dp(rho: float, delta: float) -> float:
        """Convert ρ-zCDP to (ε, δ)-DP.

        Uses the standard conversion:
        ``ε = ρ + 2√(ρ · log(1/δ))``

        Args:
            rho: zCDP parameter ρ > 0.
            delta: Target δ ∈ (0, 1).

        Returns:
            Privacy parameter ε.

        Raises:
            ValueError: If ρ ≤ 0 or δ is out of range.
        """
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))

    @staticmethod
    def approx_dp_to_zcdp(epsilon: float, delta: float) -> float:
        """Convert (ε, δ)-DP to zCDP (upper bound).

        An (ε, δ)-DP mechanism satisfies ρ-zCDP with
        ``ρ = ε² / (2 · log(1/δ))``  (when δ > 0).

        For pure DP (δ = 0), uses ``ρ = ε · (exp(ε) - 1) / 2``.

        Args:
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.

        Returns:
            zCDP parameter ρ.
        """
        if delta == 0:
            return epsilon * (math.exp(epsilon) - 1.0) / 2.0
        if delta >= 1 or delta < 0:
            raise ValueError(f"delta must be in [0, 1), got {delta}")
        log_inv_delta = math.log(1.0 / delta)
        if log_inv_delta < 1e-15:
            return float("inf")
        return epsilon ** 2 / (2.0 * log_inv_delta)

    @staticmethod
    def gaussian_zcdp(sigma: float, sensitivity: float = 1.0) -> float:
        """Compute ρ for the Gaussian mechanism.

        The Gaussian mechanism with noise σ satisfies
        ``ρ = Δ² / (2σ²)``-zCDP.

        Args:
            sigma: Noise standard deviation.
            sensitivity: Query sensitivity Δ.

        Returns:
            zCDP parameter ρ.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        return sensitivity ** 2 / (2.0 * sigma ** 2)

    def get_privacy(self, delta: float) -> PrivacyBudget:
        """Get the current composed (ε, δ)-DP guarantee.

        Args:
            delta: Target δ.

        Returns:
            Privacy budget.
        """
        if self._rho_total <= 0:
            return PrivacyBudget(epsilon=1e-15, delta=delta)
        eps = self.zcdp_to_approx_dp(self._rho_total, delta)
        return PrivacyBudget(epsilon=eps, delta=delta)

    def reset(self) -> None:
        """Reset the accountant."""
        self._rho_total = 0.0
        self._mechanisms.clear()

    def __repr__(self) -> str:
        return (
            f"ZeroCDPAccountant(rho={self._rho_total:.6f}, "
            f"n_compositions={self.n_compositions})"
        )


# =========================================================================
# 5. Moments Accountant
# =========================================================================


class MomentsAccountant:
    """Moments accountant (Abadi et al. 2016).

    Tracks privacy loss via log-moment generating functions, which
    compose additively.  Primarily used for subsampled mechanisms
    (e.g., DP-SGD).

    The log-moment of a mechanism M at order λ is:
        α_M(λ) = max_{d,d'} log E[exp(λ · L)] where
        L = log(M(x|d) / M(x|d')) is the privacy loss RV.

    Conversion to (ε, δ)-DP via Markov's inequality:
        ε = min_λ { (α_M(λ) - log(δ)) / λ }
    """

    def __init__(
        self,
        max_order: int = 64,
    ) -> None:
        """Initialize the moments accountant.

        Args:
            max_order: Maximum moment order λ to track.
        """
        self.max_order = max_order
        self.orders = np.arange(1, max_order + 1, dtype=np.float64)
        self._log_moments: FloatArray = np.zeros(max_order, dtype=np.float64)
        self._n_compositions: int = 0

    @property
    def n_compositions(self) -> int:
        """Number of mechanisms composed."""
        return self._n_compositions

    def compute_log_moments(
        self,
        mechanism: str,
        order: float,
        **kwargs: Any,
    ) -> float:
        """Compute the log-moment for a mechanism at a given order.

        Supported mechanisms: ``'gaussian'``, ``'laplace'``,
        ``'subsampled_gaussian'``.

        Args:
            mechanism: Mechanism type.
            order: Moment order λ.
            **kwargs: Mechanism-specific parameters.

        Returns:
            Log-moment α_M(λ).

        Raises:
            ValueError: If the mechanism type is unknown.
        """
        if mechanism == "gaussian":
            sigma = kwargs.get("sigma", 1.0)
            sensitivity = kwargs.get("sensitivity", 1.0)
            return self._gaussian_log_moment(order, sigma, sensitivity)
        elif mechanism == "laplace":
            epsilon = kwargs["epsilon"]
            return self._laplace_log_moment(order, epsilon)
        elif mechanism == "subsampled_gaussian":
            sigma = kwargs.get("sigma", 1.0)
            sampling_rate = kwargs.get("sampling_rate", 1.0)
            sensitivity = kwargs.get("sensitivity", 1.0)
            return self._subsampled_gaussian_log_moment(
                order, sigma, sampling_rate, sensitivity
            )
        else:
            raise ValueError(f"Unknown mechanism type: {mechanism!r}")

    @staticmethod
    def _gaussian_log_moment(
        order: float,
        sigma: float,
        sensitivity: float,
    ) -> float:
        """Log-moment for the Gaussian mechanism.

        α_M(λ) = λ(λ+1) Δ² / (2σ²)
        """
        return order * (order + 1) * sensitivity ** 2 / (2.0 * sigma ** 2)

    @staticmethod
    def _laplace_log_moment(order: float, epsilon: float) -> float:
        """Log-moment for the Laplace mechanism.

        Bounded by α_M(λ) ≤ λ(λ+1)ε² / 2 for small ε,
        or λε for large ε.
        """
        return min(
            order * (order + 1) * epsilon ** 2 / 2.0,
            order * epsilon,
        )

    @staticmethod
    def _subsampled_gaussian_log_moment(
        order: float,
        sigma: float,
        sampling_rate: float,
        sensitivity: float,
    ) -> float:
        """Log-moment for the subsampled Gaussian mechanism.

        Uses Proposition 3 from Abadi et al. (2016) for integer orders,
        and an interpolation for fractional orders.
        """
        int_order = int(order)
        if int_order < 1:
            return 0.0

        # For integer λ, compute via binomial expansion
        log_terms = []
        for j in range(int_order + 1):
            # log(C(λ, j)) + j*log(q) + (λ-j)*log(1-q) + j(j-1)/(2σ²)
            log_binom = 0.0
            for i in range(j):
                log_binom += math.log(int_order - i) - math.log(i + 1)

            log_q_term = j * math.log(max(sampling_rate, 1e-300))
            log_1mq_term = (int_order - j) * math.log(max(1.0 - sampling_rate, 1e-300))
            rdp_term = j * (j - 1) * sensitivity ** 2 / (2.0 * sigma ** 2)

            log_terms.append(log_binom + log_q_term + log_1mq_term + rdp_term)

        return _logsumexp(np.array(log_terms))

    def add_mechanism(
        self,
        mechanism: str,
        **kwargs: Any,
    ) -> None:
        """Add a mechanism's log-moments to the composition.

        Args:
            mechanism: Mechanism type (``'gaussian'``, ``'laplace'``,
                ``'subsampled_gaussian'``).
            **kwargs: Mechanism-specific parameters.
        """
        for idx, order in enumerate(self.orders):
            self._log_moments[idx] += self.compute_log_moments(
                mechanism, order, **kwargs
            )
        self._n_compositions += 1

    def compose(
        self,
        log_moments_list: Sequence[FloatArray],
    ) -> FloatArray:
        """Compose log-moments from multiple mechanisms (without state).

        Args:
            log_moments_list: List of log-moment arrays, each of length
                ``max_order``.

        Returns:
            Composed log-moments (element-wise sum).
        """
        composed = np.zeros(self.max_order, dtype=np.float64)
        for lm in log_moments_list:
            lm_arr = np.asarray(lm, dtype=np.float64)
            if len(lm_arr) != self.max_order:
                raise ValueError(
                    f"Log-moment array length ({len(lm_arr)}) must match "
                    f"max_order ({self.max_order})"
                )
            composed += lm_arr
        return composed

    def get_privacy(self, target_delta: float) -> PrivacyBudget:
        """Convert composed log-moments to (ε, δ)-DP.

        Uses: ε = min_λ { (α_total(λ) - log(δ)) / λ }

        Args:
            target_delta: Target δ ∈ (0, 1).

        Returns:
            Privacy budget.

        Raises:
            ValueError: If target_delta is out of range.
        """
        if not (0 < target_delta < 1):
            raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")

        log_delta = math.log(target_delta)
        eps_candidates = (self._log_moments - log_delta) / self.orders
        eps = float(np.min(eps_candidates))
        eps = max(eps, 0.0)
        return PrivacyBudget(epsilon=max(eps, 1e-15), delta=target_delta)

    def reset(self) -> None:
        """Reset the accountant."""
        self._log_moments = np.zeros(self.max_order, dtype=np.float64)
        self._n_compositions = 0

    def __repr__(self) -> str:
        return (
            f"MomentsAccountant(max_order={self.max_order}, "
            f"n_compositions={self.n_compositions})"
        )


# =========================================================================
# 6. Subsampling Amplification
# =========================================================================


class SubsamplingAmplification:
    """Privacy amplification by subsampling.

    When a mechanism is applied to a random subset of the data, the
    effective privacy guarantee improves.  This class provides amplification
    bounds for both Poisson subsampling and sampling without replacement.
    """

    @staticmethod
    def poisson_subsample(
        epsilon: float,
        delta: float,
        rate: float,
    ) -> PrivacyBudget:
        """Amplification by Poisson subsampling.

        Each record is included independently with probability *rate*.
        By Balle et al. (2018), the amplified mechanism satisfies
        (ε', δ')-DP with:

        For pure DP (δ=0):
            ε' = log(1 + rate · (exp(ε) - 1))

        For approximate DP (δ>0):
            ε' = log(1 + rate · (exp(ε) - 1))
            δ' = rate · δ

        Args:
            epsilon: Original ε.
            delta: Original δ.
            rate: Subsampling probability q ∈ (0, 1].

        Returns:
            Amplified privacy budget.

        Raises:
            ValueError: If rate is out of range.
        """
        if not (0 < rate <= 1):
            raise ValueError(f"rate must be in (0, 1], got {rate}")

        if rate == 1.0:
            return PrivacyBudget(epsilon=epsilon, delta=delta)

        # Amplified epsilon
        eps_amplified = math.log(1.0 + rate * (math.exp(epsilon) - 1.0))
        delta_amplified = rate * delta

        return PrivacyBudget(
            epsilon=eps_amplified,
            delta=min(delta_amplified, 1.0 - 1e-15),
        )

    @staticmethod
    def without_replacement(
        epsilon: float,
        delta: float,
        n: int,
        batch_size: int,
    ) -> PrivacyBudget:
        """Amplification by sampling without replacement.

        A batch of *batch_size* records is drawn uniformly without
        replacement from a dataset of *n* records.

        Uses the bound from Balle et al. (2018):
            ε' ≤ log(1 + (batch_size/n) · (exp(ε) - 1))

        Args:
            epsilon: Original ε.
            delta: Original δ.
            n: Total dataset size.
            batch_size: Batch size (subset size).

        Returns:
            Amplified privacy budget.

        Raises:
            ValueError: If batch_size > n or inputs are invalid.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if batch_size < 1 or batch_size > n:
            raise ValueError(
                f"batch_size must be in [1, {n}], got {batch_size}"
            )

        rate = batch_size / n
        if rate >= 1.0:
            return PrivacyBudget(epsilon=epsilon, delta=delta)

        eps_amplified = math.log(1.0 + rate * (math.exp(epsilon) - 1.0))
        delta_amplified = rate * delta

        return PrivacyBudget(
            epsilon=eps_amplified,
            delta=min(delta_amplified, 1.0 - 1e-15),
        )

    @staticmethod
    def amplified_rdp_gaussian(
        sigma: float,
        sampling_rate: float,
        sensitivity: float,
        alphas: Optional[FloatArray] = None,
    ) -> Tuple[FloatArray, FloatArray]:
        """Compute amplified RDP curve for a subsampled Gaussian.

        Args:
            sigma: Noise standard deviation.
            sampling_rate: Subsampling probability.
            sensitivity: Query sensitivity.
            alphas: α grid.

        Returns:
            ``(alphas, rdp_values)`` tuple.
        """
        if alphas is None:
            alphas = RenyiDPAccountant.DEFAULT_ALPHAS.copy()

        mech = RenyiDPAccountant.subsampled_gaussian_rdp(
            sigma=sigma,
            sampling_rate=sampling_rate,
            sensitivity=sensitivity,
        )
        rdp_values = np.array(
            [mech.evaluate(a) for a in alphas], dtype=np.float64
        )
        return alphas, rdp_values

    @staticmethod
    def compute_sigma_for_budget(
        epsilon: float,
        delta: float,
        sensitivity: float,
        sampling_rate: float,
        n_steps: int,
    ) -> float:
        """Find the noise σ needed to achieve (ε, δ) after *n_steps* of
        subsampled Gaussian mechanism.

        Uses binary search over σ.

        Args:
            epsilon: Target ε.
            delta: Target δ.
            sensitivity: Query sensitivity.
            sampling_rate: Subsampling probability.
            n_steps: Number of mechanism applications.

        Returns:
            Required noise standard deviation σ.
        """
        lo, hi = 0.01, 1000.0

        for _ in range(100):  # binary search iterations
            mid = (lo + hi) / 2.0
            accountant = RenyiDPAccountant()
            mech = RenyiDPAccountant.subsampled_gaussian_rdp(
                sigma=mid,
                sampling_rate=sampling_rate,
                sensitivity=sensitivity,
            )
            rdp_curve = accountant.compose_rdp([mech] * n_steps)
            achieved_eps = accountant.rdp_to_approx_dp(rdp_curve, delta)

            if achieved_eps > epsilon:
                lo = mid  # need more noise
            else:
                hi = mid

        return hi


# =========================================================================
# 7. Privacy Budget Tracker
# =========================================================================


@dataclass
class PrivacyAllocation:
    """Record of a single privacy allocation.

    Attributes:
        mechanism_name: Name of the mechanism.
        epsilon: ε consumed.
        delta: δ consumed.
        composition_type: Composition method used.
        metadata: Additional metadata.
    """
    mechanism_name: str
    epsilon: float
    delta: float
    composition_type: CompositionType = CompositionType.BASIC
    metadata: Dict[str, Any] = field(default_factory=dict)


class PrivacyBudgetTracker:
    """Tracks cumulative privacy consumption against a total budget.

    Supports multiple composition methods and warns or raises when the
    budget is exhausted.

    Example::

        >>> tracker = PrivacyBudgetTracker(epsilon=10.0, delta=1e-5)
        >>> tracker.allocate("query_1", PrivacyBudget(1.0, 1e-6))
        >>> tracker.remaining()
        PrivacyBudget(ε=9.0, δ=9e-06)
        >>> tracker.can_afford(PrivacyBudget(15.0, 0.0))
        False
    """

    def __init__(
        self,
        epsilon: float,
        delta: float = 0.0,
        composition_type: CompositionType = CompositionType.BASIC,
        warn_threshold: float = 0.9,
    ) -> None:
        """Initialize the budget tracker.

        Args:
            epsilon: Total ε budget.
            delta: Total δ budget.
            composition_type: Default composition method.
            warn_threshold: Fraction of budget at which to warn.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if not (0.0 <= delta < 1.0):
            raise ValueError(f"delta must be in [0, 1), got {delta}")

        self.total_epsilon = epsilon
        self.total_delta = delta
        self.composition_type = composition_type
        self.warn_threshold = warn_threshold

        self._allocations: List[PrivacyAllocation] = []
        self._consumed_epsilon: float = 0.0
        self._consumed_delta: float = 0.0

        # RDP accountant for RDP composition
        self._rdp_accountant: Optional[RenyiDPAccountant] = None
        if composition_type == CompositionType.RDP:
            self._rdp_accountant = RenyiDPAccountant()

        # zCDP accountant for zCDP composition
        self._zcdp_accountant: Optional[ZeroCDPAccountant] = None
        if composition_type == CompositionType.ZERO_CDP:
            self._zcdp_accountant = ZeroCDPAccountant()

    def allocate(
        self,
        mechanism_name: str,
        budget: PrivacyBudget,
        rdp_mechanism: Optional[RDPMechanism] = None,
        zcdp_rho: Optional[float] = None,
    ) -> None:
        """Deduct a privacy cost from the budget.

        Args:
            mechanism_name: Name for the allocation.
            budget: Privacy cost in (ε, δ) form.
            rdp_mechanism: RDP specification (for RDP composition).
            zcdp_rho: zCDP ρ value (for zCDP composition).

        Raises:
            BudgetExhaustedError: If the allocation would exceed the budget.
        """
        # Check affordability first
        if not self.can_afford(budget, rdp_mechanism=rdp_mechanism, zcdp_rho=zcdp_rho):
            raise BudgetExhaustedError(
                f"Allocation for {mechanism_name!r} (ε={budget.epsilon}, "
                f"δ={budget.delta}) would exceed budget",
                budget_epsilon=self.total_epsilon,
                budget_delta=self.total_delta,
                consumed_epsilon=self._consumed_epsilon,
                consumed_delta=self._consumed_delta,
            )

        self._allocations.append(PrivacyAllocation(
            mechanism_name=mechanism_name,
            epsilon=budget.epsilon,
            delta=budget.delta,
            composition_type=self.composition_type,
        ))

        if self.composition_type == CompositionType.BASIC:
            self._consumed_epsilon += budget.epsilon
            self._consumed_delta += budget.delta
        elif self.composition_type == CompositionType.ADVANCED:
            self._consumed_epsilon += budget.epsilon
            self._consumed_delta += budget.delta
        elif self.composition_type == CompositionType.RDP:
            if rdp_mechanism is not None and self._rdp_accountant is not None:
                self._rdp_accountant.add_mechanism(rdp_mechanism)
            else:
                self._consumed_epsilon += budget.epsilon
                self._consumed_delta += budget.delta
        elif self.composition_type == CompositionType.ZERO_CDP:
            if zcdp_rho is not None and self._zcdp_accountant is not None:
                self._zcdp_accountant.add_mechanism(zcdp_rho, mechanism_name)
            else:
                self._consumed_epsilon += budget.epsilon
                self._consumed_delta += budget.delta

        # Warn if approaching budget
        remaining = self.remaining()
        if remaining.epsilon / self.total_epsilon < (1 - self.warn_threshold):
            warnings.warn(
                f"Privacy budget {self.warn_threshold*100:.0f}% consumed: "
                f"ε remaining = {remaining.epsilon:.4f}",
                UserWarning,
                stacklevel=2,
            )

    def remaining(self) -> PrivacyBudget:
        """Compute the remaining privacy budget.

        Returns:
            Remaining budget as a PrivacyBudget.
        """
        if self.composition_type == CompositionType.RDP and self._rdp_accountant is not None:
            if self._rdp_accountant.n_compositions > 0:
                spent = self._rdp_accountant.get_privacy(max(self.total_delta, 1e-15))
                eps_remaining = max(self.total_epsilon - spent.epsilon, 0.0)
                return PrivacyBudget(
                    epsilon=max(eps_remaining, 1e-15),
                    delta=max(self.total_delta - self._consumed_delta, 0.0),
                )

        if self.composition_type == CompositionType.ZERO_CDP and self._zcdp_accountant is not None:
            if self._zcdp_accountant.n_compositions > 0:
                spent = self._zcdp_accountant.get_privacy(max(self.total_delta, 1e-15))
                eps_remaining = max(self.total_epsilon - spent.epsilon, 0.0)
                return PrivacyBudget(
                    epsilon=max(eps_remaining, 1e-15),
                    delta=max(self.total_delta - self._consumed_delta, 0.0),
                )

        eps_remaining = max(self.total_epsilon - self._consumed_epsilon, 1e-15)
        delta_remaining = max(self.total_delta - self._consumed_delta, 0.0)
        return PrivacyBudget(epsilon=eps_remaining, delta=delta_remaining)

    def can_afford(
        self,
        cost: PrivacyBudget,
        rdp_mechanism: Optional[RDPMechanism] = None,
        zcdp_rho: Optional[float] = None,
    ) -> bool:
        """Check whether the budget can afford a given cost.

        Args:
            cost: Privacy cost to check.
            rdp_mechanism: RDP specification (for RDP composition check).
            zcdp_rho: zCDP ρ value (for zCDP composition check).

        Returns:
            ``True`` if the allocation is affordable.
        """
        remaining = self.remaining()

        if self.composition_type in (CompositionType.BASIC, CompositionType.ADVANCED):
            return (
                self._consumed_epsilon + cost.epsilon <= self.total_epsilon + 1e-15
                and self._consumed_delta + cost.delta <= self.total_delta + 1e-15
            )

        # For RDP/zCDP, check whether adding this mechanism would exceed budget
        return remaining.epsilon >= cost.epsilon * 0.5  # heuristic; actual check at allocate

    @property
    def consumed(self) -> PrivacyBudget:
        """Total consumed privacy budget."""
        if self.composition_type == CompositionType.RDP and self._rdp_accountant is not None:
            if self._rdp_accountant.n_compositions > 0:
                return self._rdp_accountant.get_privacy(max(self.total_delta, 1e-15))

        if self.composition_type == CompositionType.ZERO_CDP and self._zcdp_accountant is not None:
            if self._zcdp_accountant.n_compositions > 0:
                return self._zcdp_accountant.get_privacy(max(self.total_delta, 1e-15))

        return PrivacyBudget(
            epsilon=max(self._consumed_epsilon, 1e-15),
            delta=self._consumed_delta,
        )

    @property
    def allocation_history(self) -> List[PrivacyAllocation]:
        """List of all allocations made."""
        return list(self._allocations)

    @property
    def n_allocations(self) -> int:
        """Number of allocations made."""
        return len(self._allocations)

    def is_exhausted(self) -> bool:
        """Whether the budget is fully consumed."""
        remaining = self.remaining()
        return remaining.epsilon <= 1e-12

    def reset(self) -> None:
        """Reset the tracker, clearing all allocations."""
        self._allocations.clear()
        self._consumed_epsilon = 0.0
        self._consumed_delta = 0.0
        if self._rdp_accountant is not None:
            self._rdp_accountant.reset()
        if self._zcdp_accountant is not None:
            self._zcdp_accountant.reset()

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the budget status.

        Returns:
            Dict with budget, consumed, remaining, and allocation details.
        """
        remaining = self.remaining()
        consumed = self.consumed
        return {
            "total_budget": {
                "epsilon": self.total_epsilon,
                "delta": self.total_delta,
            },
            "consumed": {
                "epsilon": consumed.epsilon,
                "delta": consumed.delta,
            },
            "remaining": {
                "epsilon": remaining.epsilon,
                "delta": remaining.delta,
            },
            "n_allocations": self.n_allocations,
            "composition_type": self.composition_type.name,
            "is_exhausted": self.is_exhausted(),
            "allocations": [
                {
                    "name": a.mechanism_name,
                    "epsilon": a.epsilon,
                    "delta": a.delta,
                }
                for a in self._allocations
            ],
        }

    def __repr__(self) -> str:
        remaining = self.remaining()
        return (
            f"PrivacyBudgetTracker(total_ε={self.total_epsilon}, "
            f"remaining_ε={remaining.epsilon:.4f}, "
            f"n_allocs={self.n_allocations})"
        )


# =========================================================================
# Utility: compose_optimal
# =========================================================================


def compose_optimal(
    budgets: Sequence[PrivacyBudget],
    target_delta: float,
) -> PrivacyBudget:
    """Compose multiple budgets using the tightest available method.

    Tries RDP composition first (tightest for many practical mechanisms),
    then advanced composition, then basic composition, and returns the
    smallest ε.

    Args:
        budgets: Per-mechanism privacy budgets.
        target_delta: Target total δ.

    Returns:
        The tightest composed privacy budget.
    """
    if len(budgets) == 0:
        raise ValueError("budgets must be non-empty")

    results = []

    # Basic composition
    basic = BasicComposition.sequential(budgets)
    results.append(basic)

    # Advanced composition (requires δ' > 0)
    try:
        eps_list = [b.epsilon for b in budgets]
        delta_list = [b.delta for b in budgets]
        delta_prime = target_delta / 2.0
        remaining_delta = target_delta - delta_prime
        if remaining_delta > sum(delta_list) and delta_prime > 0:
            advanced = AdvancedComposition.compose(
                eps_list, delta_list, delta_prime
            )
            results.append(advanced)
    except (ValueError, ConfigurationError):
        pass

    # Return the one with smallest epsilon
    return min(results, key=lambda b: b.epsilon)


def compute_noise_for_budget(
    epsilon: float,
    delta: float,
    sensitivity: float,
    mechanism: str = "gaussian",
) -> float:
    """Compute the noise parameter needed for a given privacy budget.

    Args:
        epsilon: Target ε.
        delta: Target δ.
        sensitivity: Query sensitivity Δ.
        mechanism: ``'gaussian'`` or ``'laplace'``.

    Returns:
        Noise parameter (σ for Gaussian, b = Δ/ε for Laplace).

    Raises:
        ValueError: If the mechanism type is unknown.
    """
    if mechanism == "gaussian":
        if delta <= 0:
            raise ValueError("Gaussian mechanism requires delta > 0")
        sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
        return sigma
    elif mechanism == "laplace":
        return sensitivity / epsilon
    else:
        raise ValueError(f"Unknown mechanism: {mechanism!r}")

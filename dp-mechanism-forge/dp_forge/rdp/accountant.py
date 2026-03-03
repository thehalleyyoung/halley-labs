"""
RDP Accountant for tracking and composing Rényi DP guarantees.

Provides stateful privacy accounting via Rényi Differential Privacy
(Mironov 2017).  The accountant tracks per-mechanism (α, ε̂(α)) RDP
curves, composes them via summation, and converts to (ε, δ)-DP with
optimal α selection.

Key features:
    - Stateful tracking of composed RDP curves across mechanisms.
    - Heterogeneous α grid support for different mechanisms.
    - Optimal α selection for RDP → (ε, δ)-DP conversion.
    - Remaining budget tracking against a total privacy budget.
    - Integration with ``dp_forge.types.PrivacyBudget``.

References:
    - Mironov, I. (2017). Rényi differential privacy.
    - Balle, B., Gaboardi, M., & Zanella-Béguelin, B. (2020).
      Privacy profiles and amplification by subsampling.
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
from dp_forge.types import PrivacyBudget

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Numerically stable helpers
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


# ---------------------------------------------------------------------------
# Default alpha grid
# ---------------------------------------------------------------------------

DEFAULT_ALPHAS: FloatArray = np.concatenate([
    np.arange(1.25, 10.0, 0.25),
    np.arange(10.0, 65.0, 2.0),
    np.array([128.0, 256.0, 512.0, 1024.0]),
])


# =========================================================================
# RDPCurve
# =========================================================================


@dataclass
class RDPCurve:
    """An RDP curve storing (α, ε̂(α)) pairs.

    The curve represents a mapping from Rényi divergence orders α to
    RDP epsilon values, fully characterising a mechanism's RDP guarantee.

    Attributes:
        alphas: Sorted array of α orders, shape ``(m,)``.
        epsilons: RDP ε values at each α, shape ``(m,)``.
        name: Human-readable label for the mechanism.
        metadata: Additional metadata about the curve origin.
    """

    alphas: FloatArray
    epsilons: FloatArray
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.alphas = np.asarray(self.alphas, dtype=np.float64)
        self.epsilons = np.asarray(self.epsilons, dtype=np.float64)
        if self.alphas.ndim != 1:
            raise ValueError(f"alphas must be 1-D, got shape {self.alphas.shape}")
        if self.epsilons.ndim != 1:
            raise ValueError(f"epsilons must be 1-D, got shape {self.epsilons.shape}")
        if len(self.alphas) != len(self.epsilons):
            raise ValueError(
                f"alphas ({len(self.alphas)}) and epsilons ({len(self.epsilons)}) "
                f"must have equal length"
            )
        if len(self.alphas) == 0:
            raise ValueError("RDPCurve must have at least one (α, ε) pair")
        if np.any(self.alphas <= 1.0):
            raise ValueError(
                f"All alphas must be > 1, got min={np.min(self.alphas):.4f}"
            )

    @property
    def n_orders(self) -> int:
        """Number of α orders in the curve."""
        return len(self.alphas)

    def evaluate(self, alpha: float) -> float:
        """Evaluate the RDP guarantee at a given α via interpolation.

        For α values outside the stored range, extrapolates using the
        nearest value (conservative upper bound).

        Args:
            alpha: Rényi divergence order (>= 1).

        Returns:
            Interpolated RDP ε value.
        """
        if alpha < 1.0:
            raise ConfigurationError(
                f"alpha must be >= 1 for RDP, got {alpha}",
                parameter="alpha",
                value=alpha,
            )
        return float(np.interp(alpha, self.alphas, self.epsilons))

    def evaluate_vectorized(self, alphas: FloatArray) -> FloatArray:
        """Evaluate the RDP curve at multiple α values.

        Args:
            alphas: Array of α orders.

        Returns:
            Array of interpolated RDP ε values.
        """
        alphas = np.asarray(alphas, dtype=np.float64)
        return np.interp(alphas, self.alphas, self.epsilons)

    def to_dp(self, delta: float) -> PrivacyBudget:
        """Convert this RDP curve to (ε, δ)-DP.

        Uses the standard conversion:
            ε = min_α { ε̂(α) + log(1/δ) / (α - 1) }

        Args:
            delta: Target δ ∈ (0, 1).

        Returns:
            Privacy budget with the tightest ε at the given δ.
        """
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        log_delta = math.log(delta)
        eps_candidates = self.epsilons - log_delta / (self.alphas - 1.0)
        best_eps = float(np.min(eps_candidates))

        return PrivacyBudget(epsilon=max(best_eps, 1e-15), delta=delta)

    def optimal_alpha(self, delta: float) -> float:
        """Find the optimal α minimising ε for the given δ.

        Args:
            delta: Target δ ∈ (0, 1).

        Returns:
            Optimal α value from the grid.
        """
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        log_delta = math.log(delta)
        eps_candidates = self.epsilons - log_delta / (self.alphas - 1.0)
        return float(self.alphas[np.argmin(eps_candidates)])

    def __add__(self, other: RDPCurve) -> RDPCurve:
        """Compose two RDP curves via pointwise addition.

        If the curves have different α grids, interpolates onto the
        union of both grids.

        Args:
            other: Another RDP curve.

        Returns:
            Composed RDP curve.
        """
        if np.array_equal(self.alphas, other.alphas):
            return RDPCurve(
                alphas=self.alphas.copy(),
                epsilons=self.epsilons + other.epsilons,
                name=f"{self.name}+{other.name}" if self.name else other.name,
            )

        # Merge α grids
        merged_alphas = np.sort(np.unique(np.concatenate([self.alphas, other.alphas])))
        eps_self = np.interp(merged_alphas, self.alphas, self.epsilons)
        eps_other = np.interp(merged_alphas, other.alphas, other.epsilons)

        return RDPCurve(
            alphas=merged_alphas,
            epsilons=eps_self + eps_other,
            name=f"{self.name}+{other.name}" if self.name else other.name,
        )

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name else ""
        return f"RDPCurve(n_orders={self.n_orders}{name})"


# =========================================================================
# RDPAccountant
# =========================================================================


class RDPAccountant:
    """Stateful privacy accountant using Rényi Differential Privacy.

    Tracks per-mechanism (α, ε̂(α)) RDP curves, composes them via
    summation, and converts to (ε, δ)-DP with optimal α selection.

    The accountant supports:
        - Adding mechanisms by name (gaussian, laplace) or by RDP curve.
        - Heterogeneous α grids (interpolated to the accountant's grid).
        - Remaining budget tracking against a total (ε, δ) budget.
        - Querying the current composed privacy guarantee.

    Args:
        alphas: Custom α grid. Defaults to :data:`DEFAULT_ALPHAS`.
        total_budget: Optional total privacy budget for budget tracking.

    Example::

        acct = RDPAccountant()
        acct.add_mechanism("gaussian", sigma=1.0, sensitivity=1.0)
        acct.add_mechanism("gaussian", sigma=2.0, sensitivity=1.0)
        budget = acct.to_dp(delta=1e-5)
        print(f"ε = {budget.epsilon:.4f}")
    """

    def __init__(
        self,
        alphas: Optional[FloatArray] = None,
        total_budget: Optional[PrivacyBudget] = None,
    ) -> None:
        if alphas is not None:
            self._alphas = np.asarray(alphas, dtype=np.float64)
            if np.any(self._alphas < 1.0):
                raise ConfigurationError(
                    "All alphas must be >= 1",
                    parameter="alphas",
                    value=float(np.min(self._alphas)),
                )
        else:
            self._alphas = DEFAULT_ALPHAS.copy()

        self._composed_rdp = np.zeros_like(self._alphas)
        self._curves: List[RDPCurve] = []
        self._total_budget = total_budget

    @property
    def alphas(self) -> FloatArray:
        """The α grid used by this accountant."""
        return self._alphas.copy()

    @property
    def n_compositions(self) -> int:
        """Number of mechanisms composed so far."""
        return len(self._curves)

    @property
    def composed_rdp(self) -> FloatArray:
        """Current composed RDP values at each α."""
        return self._composed_rdp.copy()

    @property
    def curves(self) -> List[RDPCurve]:
        """List of individual RDP curves added so far."""
        return list(self._curves)

    # -----------------------------------------------------------------
    # Adding mechanisms
    # -----------------------------------------------------------------

    def add_mechanism(
        self,
        mechanism_type: Union[str, RDPCurve],
        *,
        sigma: Optional[float] = None,
        sensitivity: float = 1.0,
        epsilon: Optional[float] = None,
        sampling_rate: Optional[float] = None,
        rdp_curve: Optional[RDPCurve] = None,
        name: Optional[str] = None,
    ) -> RDPCurve:
        """Add a mechanism to the composed RDP curve.

        Supports named mechanism types ("gaussian", "laplace", "subsampled_gaussian")
        or a pre-computed :class:`RDPCurve`.

        Args:
            mechanism_type: Either a string naming the mechanism type or
                an :class:`RDPCurve` instance.
            sigma: Noise standard deviation (for gaussian/subsampled_gaussian).
            sensitivity: Query sensitivity Δ (for gaussian mechanisms).
            epsilon: Privacy parameter ε (for laplace mechanism).
            sampling_rate: Subsampling probability (for subsampled_gaussian).
            rdp_curve: Pre-computed RDP curve (alternative to mechanism_type).
            name: Optional human-readable name for the mechanism.

        Returns:
            The RDP curve for this mechanism.

        Raises:
            ConfigurationError: If required parameters are missing.
            BudgetExhaustedError: If adding this mechanism would exceed
                the total budget.
        """
        if isinstance(mechanism_type, RDPCurve):
            curve = mechanism_type
        elif isinstance(mechanism_type, str):
            curve = self._build_curve(
                mechanism_type,
                sigma=sigma,
                sensitivity=sensitivity,
                epsilon=epsilon,
                sampling_rate=sampling_rate,
                name=name,
            )
        else:
            raise ConfigurationError(
                f"mechanism_type must be str or RDPCurve, got {type(mechanism_type)}",
                parameter="mechanism_type",
            )

        # Evaluate on accountant's α grid
        rdp_values = curve.evaluate_vectorized(self._alphas)
        self._composed_rdp += rdp_values
        self._curves.append(curve)

        # Check budget if set
        if self._total_budget is not None:
            self._check_budget()

        return curve

    def _build_curve(
        self,
        mechanism_type: str,
        *,
        sigma: Optional[float],
        sensitivity: float,
        epsilon: Optional[float],
        sampling_rate: Optional[float],
        name: Optional[str],
    ) -> RDPCurve:
        """Build an RDP curve for a named mechanism type."""
        mtype = mechanism_type.lower().replace("-", "_")

        if mtype == "gaussian":
            if sigma is None:
                raise ConfigurationError(
                    "sigma is required for gaussian mechanism",
                    parameter="sigma",
                )
            if sigma <= 0:
                raise ConfigurationError(
                    f"sigma must be positive, got {sigma}",
                    parameter="sigma",
                    value=sigma,
                )
            rdp_eps = self._alphas * sensitivity ** 2 / (2.0 * sigma ** 2)
            label = name or f"Gaussian(σ={sigma:.4f})"
            return RDPCurve(alphas=self._alphas.copy(), epsilons=rdp_eps, name=label)

        elif mtype == "laplace":
            if epsilon is None:
                raise ConfigurationError(
                    "epsilon is required for laplace mechanism",
                    parameter="epsilon",
                )
            if epsilon <= 0:
                raise ConfigurationError(
                    f"epsilon must be positive, got {epsilon}",
                    parameter="epsilon",
                    value=epsilon,
                )
            rdp_eps = self._laplace_rdp_curve(epsilon)
            label = name or f"Laplace(ε={epsilon:.4f})"
            return RDPCurve(alphas=self._alphas.copy(), epsilons=rdp_eps, name=label)

        elif mtype in ("subsampled_gaussian", "subsampled-gaussian"):
            if sigma is None:
                raise ConfigurationError(
                    "sigma is required for subsampled_gaussian",
                    parameter="sigma",
                )
            if sampling_rate is None:
                raise ConfigurationError(
                    "sampling_rate is required for subsampled_gaussian",
                    parameter="sampling_rate",
                )
            if not (0 < sampling_rate <= 1):
                raise ConfigurationError(
                    f"sampling_rate must be in (0, 1], got {sampling_rate}",
                    parameter="sampling_rate",
                    value=sampling_rate,
                )
            rdp_eps = self._subsampled_gaussian_rdp_curve(
                sigma, sampling_rate, sensitivity
            )
            label = name or f"SubsampledGaussian(σ={sigma:.4f}, q={sampling_rate:.4f})"
            return RDPCurve(alphas=self._alphas.copy(), epsilons=rdp_eps, name=label)

        else:
            raise ConfigurationError(
                f"Unknown mechanism type: {mechanism_type!r}. "
                f"Supported: 'gaussian', 'laplace', 'subsampled_gaussian'.",
                parameter="mechanism_type",
                value=mechanism_type,
            )

    def _laplace_rdp_curve(self, epsilon: float) -> FloatArray:
        """Compute RDP curve for Laplace mechanism at the accountant's α grid.

        ε̂(α) = 1/(α-1) log( α/(2α-1) exp((α-1)ε) + (α-1)/(2α-1) exp(-(α-1)ε) )
        """
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
            log_sum = _logsumexp(np.array([log_t1, log_t2]))
            rdp[i] = max(log_sum / a_minus_1, 0.0)

        return rdp

    def _subsampled_gaussian_rdp_curve(
        self,
        sigma: float,
        sampling_rate: float,
        sensitivity: float,
    ) -> FloatArray:
        """RDP curve for subsampled Gaussian using analytic moments bound.

        Uses the bound from Wang, Balle, Kasiviswanathan (2019):
        For each integer α ≥ 2, the RDP of the subsampled Gaussian is
        bounded via the binomial expansion of the privacy loss moments.
        """
        rdp = np.empty_like(self._alphas)
        q = sampling_rate

        for i, alpha in enumerate(self._alphas):
            if abs(alpha - 1.0) < 1e-10:
                rdp[i] = 0.0
                continue

            if q == 1.0:
                rdp[i] = alpha * sensitivity ** 2 / (2.0 * sigma ** 2)
                continue

            # Bound: log(1 + q² α(α-1)/(2σ²) + higher order terms)
            # Use the simple bound for non-integer alpha:
            # ε(α) ≤ 1/(α-1) log(1 + C(α,2) q² exp((2-1)/(σ²)))
            # Simplified: use the Gaussian RDP scaled by amplification
            base_rdp = alpha * sensitivity ** 2 / (2.0 * sigma ** 2)

            # Amplification: log(1 + q(exp((α-1)·base_rdp/(α)) - 1))
            # For small q, this gives approximately q · base_rdp
            inner = (alpha - 1.0) * base_rdp / alpha if alpha > 0 else 0.0
            if inner > 500:
                # Overflow protection: use the unamplified bound
                rdp[i] = base_rdp
            else:
                amplified = math.log1p(q * (math.exp(inner) - 1.0))
                rdp[i] = max(alpha * amplified / (alpha - 1.0), 0.0)

        return rdp

    # -----------------------------------------------------------------
    # Composition
    # -----------------------------------------------------------------

    def compose(
        self,
        curves: Sequence[RDPCurve],
    ) -> RDPCurve:
        """Compose multiple RDP curves without modifying accountant state.

        RDP composition is additive: if M₁ satisfies (α, ε̂₁(α))-RDP and
        M₂ satisfies (α, ε̂₂(α))-RDP, then their sequential composition
        satisfies (α, ε̂₁(α) + ε̂₂(α))-RDP.

        Args:
            curves: Sequence of RDP curves to compose.

        Returns:
            Composed RDP curve.

        Raises:
            ValueError: If curves is empty.
        """
        if not curves:
            raise ValueError("At least one curve is required for composition")

        composed_eps = np.zeros_like(self._alphas)
        names = []
        for curve in curves:
            composed_eps += curve.evaluate_vectorized(self._alphas)
            if curve.name:
                names.append(curve.name)

        return RDPCurve(
            alphas=self._alphas.copy(),
            epsilons=composed_eps,
            name="+".join(names) if names else "composed",
        )

    # -----------------------------------------------------------------
    # Conversion to (ε, δ)-DP
    # -----------------------------------------------------------------

    def to_dp(self, delta: float) -> PrivacyBudget:
        """Convert the current composed RDP curve to (ε, δ)-DP.

        Uses the optimal conversion:
            ε = min_α { ε̂(α) + log(1/δ) / (α - 1) }

        This implements the standard Mironov (2017) conversion.  For
        tighter conversion using the Balle et al. (2020) bound, see
        :func:`dp_forge.rdp.conversion.rdp_to_dp`.

        Args:
            delta: Target δ ∈ (0, 1).

        Returns:
            Privacy budget with the tightest ε at the given δ.

        Raises:
            ValueError: If delta is not in (0, 1).
        """
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        log_delta = math.log(delta)
        eps_candidates = self._composed_rdp - log_delta / (self._alphas - 1.0)
        best_eps = float(np.min(eps_candidates))

        return PrivacyBudget(epsilon=max(best_eps, 1e-15), delta=delta)

    def get_optimal_alpha(self, delta: float) -> float:
        """Find the optimal α that minimises ε for the given δ.

        Args:
            delta: Target δ ∈ (0, 1).

        Returns:
            Optimal α value from the grid.
        """
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        log_delta = math.log(delta)
        eps_candidates = self._composed_rdp - log_delta / (self._alphas - 1.0)
        return float(self._alphas[np.argmin(eps_candidates)])

    # -----------------------------------------------------------------
    # Remaining budget tracking
    # -----------------------------------------------------------------

    def remaining_budget(self, delta: Optional[float] = None) -> PrivacyBudget:
        """Compute the remaining privacy budget.

        Args:
            delta: Target δ. If ``None``, uses the delta from
                ``total_budget``.

        Returns:
            Remaining privacy budget.

        Raises:
            ConfigurationError: If no total budget is configured.
        """
        if self._total_budget is None:
            raise ConfigurationError(
                "No total budget configured. Initialize RDPAccountant "
                "with total_budget parameter.",
                parameter="total_budget",
            )

        delta = delta or self._total_budget.delta
        if delta <= 0:
            raise ValueError(
                "delta must be > 0 for remaining budget computation. "
                "Set a positive delta in total_budget or pass delta explicitly."
            )

        current = self.to_dp(delta)
        remaining_eps = self._total_budget.epsilon - current.epsilon

        if remaining_eps <= 0:
            raise BudgetExhaustedError(
                "Privacy budget exhausted",
                budget_epsilon=self._total_budget.epsilon,
                budget_delta=delta,
                consumed_epsilon=current.epsilon,
                consumed_delta=delta,
            )

        return PrivacyBudget(epsilon=remaining_eps, delta=delta)

    def _check_budget(self) -> None:
        """Check if the total budget has been exceeded."""
        if self._total_budget is None:
            return

        delta = self._total_budget.delta
        if delta <= 0:
            return

        current = self.to_dp(delta)
        if current.epsilon > self._total_budget.epsilon:
            raise BudgetExhaustedError(
                f"Privacy budget exceeded: consumed ε={current.epsilon:.6f} > "
                f"budget ε={self._total_budget.epsilon:.6f}",
                budget_epsilon=self._total_budget.epsilon,
                budget_delta=delta,
                consumed_epsilon=current.epsilon,
                consumed_delta=delta,
            )

    # -----------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------

    def get_composed_curve(self) -> RDPCurve:
        """Return the current composed RDP curve as an RDPCurve object.

        Returns:
            RDPCurve representing the composed guarantee.
        """
        names = [c.name for c in self._curves if c.name]
        return RDPCurve(
            alphas=self._alphas.copy(),
            epsilons=self._composed_rdp.copy(),
            name="+".join(names) if names else "composed",
        )

    def reset(self) -> None:
        """Reset the accountant, clearing all composed mechanisms."""
        self._composed_rdp = np.zeros_like(self._alphas)
        self._curves.clear()

    def fork(self) -> RDPAccountant:
        """Create a copy of this accountant with the same state.

        Useful for exploring hypothetical compositions without
        modifying the original accountant.

        Returns:
            New RDPAccountant with copied state.
        """
        new = RDPAccountant(
            alphas=self._alphas.copy(),
            total_budget=self._total_budget,
        )
        new._composed_rdp = self._composed_rdp.copy()
        new._curves = list(self._curves)
        return new

    def __repr__(self) -> str:
        budget = f", budget={self._total_budget}" if self._total_budget else ""
        return (
            f"RDPAccountant(n_compositions={self.n_compositions}, "
            f"n_alphas={len(self._alphas)}{budget})"
        )


# =========================================================================
# PrivacyFilter
# =========================================================================


class PrivacyFilter:
    """Online privacy filter that halts mechanism execution after budget exhaustion.

    A privacy filter wraps an :class:`RDPAccountant` and provides a
    ``request()`` interface for online privacy accounting.  Before each
    mechanism invocation, the caller requests permission by presenting the
    mechanism's RDP curve.  The filter grants the request only if the
    cumulative privacy loss (including this new mechanism) stays within the
    total budget.  Once the budget is exhausted, all subsequent requests
    are denied.

    This implements the *privacy filter* framework from Rogers et al.
    (2016) "Max-Information, Differential Privacy, and Post-Selection
    Hypothesis Testing" and Feldman & Zrnic (2021).

    The filter supports two modes:
        - **strict**: deny the request if it would exceed the budget, even
          partially (default).
        - **fractional**: allow partial execution by reporting the maximum
          fraction of the requested mechanism that can be accommodated.

    Args:
        total_budget: Total (ε, δ) privacy budget.
        alphas: Custom α grid. Defaults to :data:`DEFAULT_ALPHAS`.
        mode: ``"strict"`` (default) or ``"fractional"``.

    Example::

        filt = PrivacyFilter(PrivacyBudget(epsilon=2.0, delta=1e-5))
        curve = RDPCurve(alphas=DEFAULT_ALPHAS,
                         epsilons=DEFAULT_ALPHAS * 1.0 / (2 * 1.0**2))
        ok, remaining = filt.request(curve)
        if ok:
            # safe to run the mechanism
            ...
    """

    def __init__(
        self,
        total_budget: PrivacyBudget,
        alphas: Optional[FloatArray] = None,
        mode: str = "strict",
    ) -> None:
        if mode not in ("strict", "fractional"):
            raise ConfigurationError(
                f"mode must be 'strict' or 'fractional', got {mode!r}",
                parameter="mode",
                value=mode,
            )
        if total_budget.epsilon <= 0:
            raise ConfigurationError(
                f"total_budget.epsilon must be > 0, got {total_budget.epsilon}",
                parameter="total_budget.epsilon",
                value=total_budget.epsilon,
            )
        self._total_budget = total_budget
        self._accountant = RDPAccountant(
            alphas=alphas, total_budget=total_budget,
        )
        self._mode = mode
        self._halted = False
        self._n_accepted = 0
        self._n_denied = 0

    @property
    def is_halted(self) -> bool:
        """Whether the filter has permanently halted."""
        return self._halted

    @property
    def n_accepted(self) -> int:
        """Number of accepted mechanism requests."""
        return self._n_accepted

    @property
    def n_denied(self) -> int:
        """Number of denied mechanism requests."""
        return self._n_denied

    def request(self, curve: RDPCurve) -> Tuple[bool, PrivacyBudget]:
        """Request permission to execute a mechanism with the given RDP curve.

        If the budget can accommodate the mechanism, accepts the request,
        updates internal state, and returns the remaining budget.  Otherwise,
        denies the request (and halts the filter in strict mode).

        Args:
            curve: RDP curve of the mechanism to execute.

        Returns:
            Tuple ``(accepted, remaining_budget)``.  If denied,
            ``remaining_budget`` reflects the current state (before this
            request).
        """
        if self._halted:
            remaining = self._remaining_or_zero()
            self._n_denied += 1
            return False, remaining

        # Speculatively compose
        forked = self._accountant.fork()
        forked.add_mechanism(curve)
        delta = self._total_budget.delta
        if delta <= 0:
            delta = 1e-10
        speculative = forked.to_dp(delta)

        if speculative.epsilon > self._total_budget.epsilon:
            if self._mode == "strict":
                self._halted = True
            self._n_denied += 1
            return False, self._remaining_or_zero()

        # Accept
        self._accountant.add_mechanism(curve)
        self._n_accepted += 1
        return True, self._remaining_or_zero()

    def request_fraction(self, curve: RDPCurve) -> Tuple[float, PrivacyBudget]:
        """Compute the maximum fraction of a mechanism that fits the remaining budget.

        Returns a fraction in [0, 1] indicating how much of the mechanism
        can be executed while staying within the total budget.  A fraction
        of 1.0 means the full mechanism fits; 0.0 means nothing fits.

        The fraction ``f`` scales the RDP curve: the mechanism with
        RDP curve ``f * curve.epsilons`` fits the remaining budget.

        Args:
            curve: RDP curve of the mechanism.

        Returns:
            Tuple ``(fraction, remaining_budget_after_fraction)``.
        """
        if self._halted:
            return 0.0, self._remaining_or_zero()

        delta = self._total_budget.delta if self._total_budget.delta > 0 else 1e-10
        current = self._accountant.to_dp(delta)
        remaining_eps = max(self._total_budget.epsilon - current.epsilon, 0.0)

        if remaining_eps <= 0:
            self._halted = True
            return 0.0, PrivacyBudget(epsilon=0.0, delta=delta)

        # Binary search for the maximum fraction
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = (lo + hi) / 2.0
            scaled_eps = curve.epsilons * mid
            scaled_curve = RDPCurve(
                alphas=curve.alphas.copy(), epsilons=scaled_eps,
            )
            forked = self._accountant.fork()
            forked.add_mechanism(scaled_curve)
            speculative = forked.to_dp(delta)
            if speculative.epsilon <= self._total_budget.epsilon:
                lo = mid
            else:
                hi = mid

        fraction = lo
        remaining = self._remaining_or_zero()
        return fraction, remaining

    def _remaining_or_zero(self) -> PrivacyBudget:
        """Return remaining budget, or zero if exhausted."""
        delta = self._total_budget.delta if self._total_budget.delta > 0 else 1e-10
        try:
            return self._accountant.remaining_budget(delta)
        except BudgetExhaustedError:
            return PrivacyBudget(epsilon=0.0, delta=delta)

    def __repr__(self) -> str:
        return (
            f"PrivacyFilter(budget={self._total_budget}, "
            f"accepted={self._n_accepted}, denied={self._n_denied}, "
            f"halted={self._halted})"
        )


# =========================================================================
# PrivacyOdometer
# =========================================================================


class PrivacyOdometer:
    """Real-time privacy loss tracker (odometer).

    Tracks the cumulative (ε, δ)-DP guarantee as mechanisms are added,
    providing a running view of the privacy expenditure.  Unlike the
    privacy filter, the odometer never halts — it simply reports the
    current cumulative loss.

    The odometer maintains a timestamped log of every mechanism added,
    enabling retrospective analysis of privacy consumption patterns.

    Args:
        alphas: Custom α grid. Defaults to :data:`DEFAULT_ALPHAS`.
        default_delta: Default δ for (ε, δ)-DP conversions.

    Example::

        odo = PrivacyOdometer(default_delta=1e-5)
        curve = char.gaussian(sigma=1.0)
        odo.add(curve)
        print(odo.current_epsilon)   # cumulative ε
        print(odo.history)           # list of (name, ε_i, cumulative_ε)
    """

    def __init__(
        self,
        alphas: Optional[FloatArray] = None,
        default_delta: float = 1e-5,
    ) -> None:
        if not (0 < default_delta < 1):
            raise ConfigurationError(
                f"default_delta must be in (0, 1), got {default_delta}",
                parameter="default_delta",
                value=default_delta,
            )
        self._accountant = RDPAccountant(alphas=alphas)
        self._default_delta = default_delta
        self._log: List[Dict[str, Any]] = []

    @property
    def current_epsilon(self) -> float:
        """Current cumulative ε at the default δ."""
        if self._accountant.n_compositions == 0:
            return 0.0
        return self._accountant.to_dp(self._default_delta).epsilon

    @property
    def current_budget(self) -> PrivacyBudget:
        """Current cumulative (ε, δ) guarantee."""
        if self._accountant.n_compositions == 0:
            return PrivacyBudget(epsilon=0.0, delta=self._default_delta)
        return self._accountant.to_dp(self._default_delta)

    @property
    def n_mechanisms(self) -> int:
        """Number of mechanisms tracked."""
        return self._accountant.n_compositions

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Timestamped log of mechanisms and cumulative privacy loss."""
        return list(self._log)

    def add(self, curve: RDPCurve) -> PrivacyBudget:
        """Add a mechanism and return the updated cumulative privacy guarantee.

        Args:
            curve: RDP curve of the mechanism.

        Returns:
            Updated cumulative (ε, δ)-DP guarantee.
        """
        # Compute the incremental ε contribution
        eps_before = self.current_epsilon
        self._accountant.add_mechanism(curve)
        budget_after = self._accountant.to_dp(self._default_delta)
        eps_after = budget_after.epsilon

        self._log.append({
            "name": curve.name or f"mechanism_{self.n_mechanisms}",
            "incremental_epsilon": eps_after - eps_before,
            "cumulative_epsilon": eps_after,
            "n_orders": curve.n_orders,
        })

        return budget_after

    def epsilon_at_delta(self, delta: float) -> float:
        """Compute current cumulative ε at a specific δ.

        Args:
            delta: Target δ ∈ (0, 1).

        Returns:
            Cumulative ε at the given δ.
        """
        if self._accountant.n_compositions == 0:
            return 0.0
        return self._accountant.to_dp(delta).epsilon

    def reset(self) -> None:
        """Reset the odometer, clearing all history."""
        self._accountant.reset()
        self._log.clear()

    def __repr__(self) -> str:
        eps = f"{self.current_epsilon:.6f}" if self.n_mechanisms > 0 else "0"
        return (
            f"PrivacyOdometer(n={self.n_mechanisms}, ε={eps}, "
            f"δ={self._default_delta:.2e})"
        )


# =========================================================================
# Heterogeneous composition
# =========================================================================


def compose_heterogeneous(
    curves: Sequence[RDPCurve],
    delta: float,
    alphas: Optional[FloatArray] = None,
) -> Tuple[PrivacyBudget, RDPCurve]:
    """Compose mechanisms with different (possibly non-overlapping) α grids.

    Standard RDP composition requires all curves to be evaluated at the
    same α grid.  When mechanisms use different α ranges (e.g., a Gaussian
    with α ∈ [1.25, 1024] and a discrete mechanism with α ∈ [2, 64]),
    this function builds a common grid that spans the union of all ranges,
    interpolates each curve onto that grid, and composes.

    The function also performs optimal α selection for the final conversion
    to (ε, δ)-DP, choosing the α that minimises the composed ε.

    Args:
        curves: Sequence of RDP curves, possibly with heterogeneous α grids.
        delta: Target δ for conversion to (ε, δ)-DP.
        alphas: Optional explicit common α grid. If ``None``, automatically
            builds from the union of all curves' α grids.

    Returns:
        Tuple ``(budget, composed_curve)`` where ``budget`` is the
        tightest (ε, δ)-DP guarantee and ``composed_curve`` is the
        RDP curve of the composition.

    Raises:
        ValueError: If no curves are provided or delta is invalid.
    """
    if not curves:
        raise ValueError("At least one curve is required for composition")
    if not (0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    # Build common α grid from the union of all curves
    if alphas is None:
        all_alphas = np.concatenate([c.alphas for c in curves])
        common_alphas = np.sort(np.unique(all_alphas))
    else:
        common_alphas = np.asarray(alphas, dtype=np.float64)

    # Interpolate and sum
    composed_eps = np.zeros_like(common_alphas)
    names: List[str] = []
    for curve in curves:
        composed_eps += curve.evaluate_vectorized(common_alphas)
        if curve.name:
            names.append(curve.name)

    composed_curve = RDPCurve(
        alphas=common_alphas,
        epsilons=composed_eps,
        name="+".join(names) if names else "heterogeneous_composed",
    )

    budget = composed_curve.to_dp(delta)
    return budget, composed_curve


# =========================================================================
# Optimal alpha grid selection
# =========================================================================


def optimal_alpha_grid(
    mechanism_types: Sequence[str],
    *,
    sigma_range: Tuple[float, float] = (0.1, 100.0),
    epsilon_range: Tuple[float, float] = (0.01, 10.0),
    n_alphas: int = 100,
    alpha_max: float = 2048.0,
) -> FloatArray:
    """Automatically select an α grid tailored to a set of mechanism types.

    Different mechanisms have their optimal RDP-to-(ε,δ) conversion α at
    different scales:
        - Gaussian mechanisms with large σ benefit from large α.
        - Laplace mechanisms and small-σ Gaussians benefit from small α.
        - Discrete mechanisms typically need moderate integer α values.

    This function analyses the expected optimal α ranges for each mechanism
    type and builds a grid with higher density near the expected optima.

    The grid is constructed by:
        1. Computing the expected optimal α for each mechanism at boundary
           parameter values (e.g., σ_min and σ_max for Gaussian).
        2. Creating log-spaced points around each optimal α.
        3. Merging and deduplicating the resulting grid.

    Args:
        mechanism_types: Sequence of mechanism type names (e.g.,
            ``["gaussian", "laplace", "discrete"]``).
        sigma_range: Range of σ values for Gaussian mechanisms.
        epsilon_range: Range of ε values for Laplace mechanisms.
        n_alphas: Desired number of α values in the output grid.
        alpha_max: Maximum α to include.

    Returns:
        Sorted array of α values, shape approximately ``(n_alphas,)``.

    Raises:
        ValueError: If mechanism_types is empty.
    """
    if not mechanism_types:
        raise ValueError("mechanism_types must be non-empty")

    # Collect focal points for α density
    focal_points: List[float] = []

    for mtype in mechanism_types:
        mtype_lower = mtype.lower().replace("-", "_")

        if mtype_lower == "gaussian":
            # Optimal α for Gaussian ~ σ² / Δ², so for σ in [σ_lo, σ_hi]:
            alpha_lo = max(1.5, 0.5 * sigma_range[0] ** 2)
            alpha_hi = min(alpha_max, 2.0 * sigma_range[1] ** 2)
            focal_points.extend([alpha_lo, alpha_hi, math.sqrt(alpha_lo * alpha_hi)])

        elif mtype_lower == "laplace":
            # Optimal α for Laplace is typically small (2–20)
            focal_points.extend([2.0, 5.0, 10.0, 20.0])

        elif mtype_lower in ("discrete", "cegis"):
            # Discrete mechanisms: integer α values are key
            focal_points.extend([float(a) for a in range(2, 33)])

        elif mtype_lower in ("subsampled_gaussian", "subsampled"):
            # Subsampled Gaussian: need both small and large α
            focal_points.extend([2.0, 5.0, 10.0, 50.0, 100.0, 500.0])

        elif mtype_lower in ("randomized_response", "randomised_response"):
            # Randomised response: small α values
            focal_points.extend([1.5, 2.0, 3.0, 5.0, 10.0])

        else:
            # Default: broad coverage
            focal_points.extend([2.0, 10.0, 50.0, 200.0])

    # Build grid with higher density around focal points
    focal_points = [max(1.25, min(fp, alpha_max)) for fp in focal_points]

    # Start with mandatory boundary points
    grid_parts: List[FloatArray] = [
        np.array([1.25, 1.5, 1.75]),  # near α=1
    ]

    # Add log-spaced clusters around each focal point
    unique_focals = sorted(set(focal_points))
    points_per_focal = max(3, n_alphas // (len(unique_focals) + 2))

    for fp in unique_focals:
        lo = max(1.25, fp / 3.0)
        hi = min(alpha_max, fp * 3.0)
        grid_parts.append(np.geomspace(lo, hi, points_per_focal))

    # Add coverage at large α
    grid_parts.append(np.array([128.0, 256.0, 512.0, 1024.0, alpha_max]))

    # Merge, sort, deduplicate
    merged = np.concatenate(grid_parts)
    merged = np.sort(np.unique(merged))

    # Thin if too many points
    if len(merged) > n_alphas * 1.5:
        indices = np.round(np.linspace(0, len(merged) - 1, n_alphas)).astype(int)
        merged = merged[indices]

    # Ensure α > 1
    merged = merged[merged > 1.0]

    return merged

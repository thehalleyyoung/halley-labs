"""
Discretization theory and error analysis for DP-Forge.

Provides theoretical error bounds for discretization of continuous
mechanisms onto finite grids, range optimization for selecting the
output support, grid size recommendations, and approximation
certificates proving that a k-point discrete mechanism is within
ε_approx of the continuous optimum.

Key Components:
    - ``DiscretizationAnalyzer``: Compute error bounds, optimal range,
      and recommended grid size for a given privacy specification.
    - ``ApproximationCertificate``: Dataclass certifying the
      discretization error bound.

Mathematical Background:
    For a continuous mechanism M with density f on R, the k-point
    discretization on grid {y_1, ..., y_k} with spacing Δy introduces
    error bounded by:

        MSE_discrete ≤ MSE_continuous + (Δy)² / 12

    for piecewise-constant approximation.  For piecewise-linear:

        MSE_discrete ≤ MSE_continuous + (Δy)⁴ / 720

    Range optimization: For ε-DP mechanisms, the optimal range B*
    satisfies the tail probability bound:

        Pr[|output| > B] ≤ exp(-ε · (B - Δf) / sensitivity)

    where Δf is the query sensitivity.  We choose B* so that the
    tail contribution to MSE is negligible (< α fraction of total MSE).

    Grid size recommendation: Given target discretization error ε_disc,
    the minimum grid size is:

        k ≥ 2B* / sqrt(12 · ε_disc) + 1  (piecewise-constant)

Usage::

    from dp_forge.discretization import DiscretizationAnalyzer

    analyzer = DiscretizationAnalyzer(epsilon=1.0, sensitivity=1.0)
    bound = analyzer.error_bound(k=100)
    B_star = analyzer.optimal_range(alpha=0.01)
    k_rec = analyzer.recommend_grid_size(target_error=1e-4)
    cert = analyzer.approximation_certificate(k=200)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import optimize as sp_optimize

from .exceptions import (
    ConfigurationError,
    DPForgeError,
)
from .types import QuerySpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ApproximationCertificate
# ---------------------------------------------------------------------------


@dataclass
class DiscretizationCertificate:
    """Certificate that a k-point discrete mechanism approximates the
    continuous optimum within a bounded error.

    Attributes:
        k: Number of grid points used.
        grid_spacing: Spacing between grid points Δy.
        range_B: Half-width of the output support [-B, B].
        discretization_error: Upper bound on MSE increase from discretization.
        tail_error: Upper bound on MSE contribution from truncated tails.
        total_error_bound: discretization_error + tail_error.
        sensitivity: Query sensitivity Δf.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        grid_type: "uniform" or "adaptive".
        mechanism_family: "piecewise_constant" or "piecewise_linear".
        timestamp: ISO-format timestamp.
    """

    k: int
    grid_spacing: float
    range_B: float
    discretization_error: float
    tail_error: float
    total_error_bound: float
    sensitivity: float
    epsilon: float
    delta: float = 0.0
    grid_type: str = "uniform"
    mechanism_family: str = "piecewise_constant"
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError(f"k must be >= 2, got {self.k}")
        if self.grid_spacing <= 0:
            raise ValueError(f"grid_spacing must be > 0, got {self.grid_spacing}")
        if self.discretization_error < 0:
            raise ValueError(
                f"discretization_error must be >= 0, got {self.discretization_error}"
            )
        if self.tail_error < 0:
            raise ValueError(f"tail_error must be >= 0, got {self.tail_error}")
        if not self.timestamp:
            import datetime
            self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    @property
    def relative_error(self) -> float:
        """Relative error as fraction of Laplace baseline MSE.

        The Laplace mechanism with parameter b = Δf/ε has MSE = 2b².
        The relative error is total_error_bound / (2b²).
        """
        b = self.sensitivity / max(self.epsilon, 1e-15)
        laplace_mse = 2.0 * b ** 2
        if laplace_mse < 1e-15:
            return 0.0
        return self.total_error_bound / laplace_mse

    def __repr__(self) -> str:
        return (
            f"DiscretizationCertificate(k={self.k}, Δy={self.grid_spacing:.4f}, "
            f"B={self.range_B:.2f}, error≤{self.total_error_bound:.2e})"
        )


# ---------------------------------------------------------------------------
# Error bound computations
# ---------------------------------------------------------------------------


def _piecewise_constant_error(grid_spacing: float) -> float:
    """MSE increase from piecewise-constant discretization.

    For a density approximated by a step function on intervals of
    width Δy, the quantization error is at most (Δy)² / 12.

    Args:
        grid_spacing: Grid spacing Δy.

    Returns:
        Upper bound on discretization MSE increase.
    """
    return grid_spacing ** 2 / 12.0


def _piecewise_linear_error(grid_spacing: float) -> float:
    """MSE increase from piecewise-linear discretization.

    For a density approximated by linear interpolation on intervals
    of width Δy, the error is at most (Δy)⁴ / 720 (Simpson-type bound).

    Args:
        grid_spacing: Grid spacing Δy.

    Returns:
        Upper bound on discretization MSE increase.
    """
    return grid_spacing ** 4 / 720.0


def _laplace_tail_mse(
    B: float,
    sensitivity: float,
    epsilon: float,
) -> float:
    """MSE contribution from truncating a Laplace distribution at ±B.

    For Laplace(0, b) with b = Δf/ε, the tail contribution to MSE is:

        E[Z² · 1{|Z| > B}] = 2b² · exp(-B/b) · (1 + B/b + (B/b)²/2)
                              ≈ 2b² · exp(-B/b) for large B/b

    We use the exact formula with all terms.

    Args:
        B: Truncation range (half-width).
        sensitivity: Query sensitivity Δf.
        epsilon: Privacy parameter ε.

    Returns:
        MSE contribution from tails beyond ±B.
    """
    b = sensitivity / max(epsilon, 1e-15)
    if b < 1e-15:
        return 0.0

    t = B / b
    if t > 500:
        return 0.0  # Negligible

    # Exact: ∫_{B}^∞ z² · (1/2b) exp(-z/b) dz
    # = b² exp(-t) (2 + 2t + t²)
    return b ** 2 * math.exp(-t) * (2.0 + 2.0 * t + t ** 2)


def _gaussian_tail_mse(
    B: float,
    sigma: float,
) -> float:
    """MSE contribution from truncating a Gaussian at ±B.

    For N(0, σ²), the tail MSE is:

        E[Z² · 1{|Z| > B}] = σ² · erfc(B/(σ√2)) + (Bσ√(2/π)) · exp(-B²/(2σ²))

    Args:
        B: Truncation range.
        sigma: Standard deviation.

    Returns:
        MSE contribution from tails.
    """
    if sigma < 1e-15:
        return 0.0

    from scipy import special
    t = B / sigma
    if t > 30:
        return 0.0  # Negligible

    # E[Z²·1{|Z|>B}] for Z~N(0,σ²)
    # = σ²(1 - erf(B/(σ√2))) + B·σ·√(2/π)·exp(-B²/(2σ²))
    erfc_val = special.erfc(t / math.sqrt(2))
    exp_val = math.exp(-0.5 * t ** 2)
    return sigma ** 2 * erfc_val + B * sigma * math.sqrt(2.0 / math.pi) * exp_val


# ---------------------------------------------------------------------------
# DiscretizationAnalyzer
# ---------------------------------------------------------------------------


class DiscretizationAnalyzer:
    """Analyze discretization error and recommend grid parameters.

    Computes theoretical error bounds for uniform vs adaptive grids,
    optimizes the output range B*, recommends grid size k for a
    target discretization error, and produces approximation certificates.

    Args:
        epsilon: Privacy parameter ε.
        sensitivity: Query sensitivity Δf.
        delta: Privacy parameter δ (for approximate DP).
        query_range: If known, the range of query outputs [q_min, q_max].
        mechanism_family: "piecewise_constant" or "piecewise_linear".

    Raises:
        ConfigurationError: If parameters are invalid.
    """

    def __init__(
        self,
        epsilon: float,
        sensitivity: float,
        *,
        delta: float = 0.0,
        query_range: Optional[Tuple[float, float]] = None,
        mechanism_family: str = "piecewise_constant",
    ) -> None:
        if epsilon <= 0:
            raise ConfigurationError(
                "epsilon must be positive",
                parameter="epsilon",
                value=epsilon,
                constraint="epsilon > 0",
            )
        if sensitivity <= 0:
            raise ConfigurationError(
                "sensitivity must be positive",
                parameter="sensitivity",
                value=sensitivity,
                constraint="sensitivity > 0",
            )
        if not (0 <= delta < 1):
            raise ConfigurationError(
                "delta must be in [0, 1)",
                parameter="delta",
                value=delta,
            )
        if mechanism_family not in ("piecewise_constant", "piecewise_linear"):
            raise ConfigurationError(
                f"mechanism_family must be 'piecewise_constant' or "
                f"'piecewise_linear', got {mechanism_family!r}",
                parameter="mechanism_family",
            )

        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.delta = delta
        self.query_range = query_range
        self.mechanism_family = mechanism_family

        # Laplace scale parameter
        self._b = sensitivity / epsilon

    def error_bound(
        self,
        k: int,
        *,
        range_B: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute discretization error bound for given grid size.

        Args:
            k: Number of grid points.
            range_B: Output range half-width. If None, uses optimal_range().

        Returns:
            Dict with keys:
            - ``"discretization_error"``: Error from grid approximation
            - ``"tail_error"``: Error from range truncation
            - ``"total_error"``: Sum of both
            - ``"grid_spacing"``: Δy
            - ``"range_B"``: B used
            - ``"laplace_mse"``: Baseline Laplace MSE for comparison
            - ``"relative_error"``: total_error / laplace_mse
        """
        if k < 2:
            raise ConfigurationError(
                "k must be >= 2",
                parameter="k",
                value=k,
            )

        if range_B is None:
            range_B = self.optimal_range()

        grid_spacing = 2.0 * range_B / max(k - 1, 1)

        if self.mechanism_family == "piecewise_constant":
            disc_error = _piecewise_constant_error(grid_spacing)
        else:
            disc_error = _piecewise_linear_error(grid_spacing)

        tail_error = _laplace_tail_mse(range_B, self.sensitivity, self.epsilon)

        total = disc_error + tail_error
        laplace_mse = 2.0 * self._b ** 2

        return {
            "discretization_error": disc_error,
            "tail_error": tail_error,
            "total_error": total,
            "grid_spacing": grid_spacing,
            "range_B": range_B,
            "laplace_mse": laplace_mse,
            "relative_error": total / max(laplace_mse, 1e-15),
        }

    def optimal_range(
        self,
        alpha: float = 0.01,
    ) -> float:
        """Compute optimal range B* for output grid.

        Finds B* such that the tail MSE contribution is ≤ α fraction
        of the Laplace baseline MSE.  Uses bisection on:

            tail_mse(B) ≤ α · 2b²

        where b = Δf/ε.

        For approximate DP (δ > 0), uses the Gaussian mechanism
        with σ = Δf · √(2 ln(1.25/δ)) / ε.

        Args:
            alpha: Maximum fraction of MSE from tails.

        Returns:
            Optimal half-width B*.

        Raises:
            ConfigurationError: If alpha not in (0, 1).
        """
        if not (0 < alpha < 1):
            raise ConfigurationError(
                "alpha must be in (0, 1)",
                parameter="alpha",
                value=alpha,
            )

        if self.delta == 0:
            # Pure DP: Laplace mechanism
            return self._optimal_range_laplace(alpha)
        else:
            # Approximate DP: Gaussian mechanism
            return self._optimal_range_gaussian(alpha)

    def _optimal_range_laplace(self, alpha: float) -> float:
        """Optimal range for Laplace-based mechanism."""
        b = self._b
        target_tail = alpha * 2.0 * b ** 2

        # tail_mse(B) = b² exp(-B/b)(2 + 2B/b + (B/b)²)
        # We need tail_mse(B) ≤ target
        # Start from B = b and search upward

        def objective(B: float) -> float:
            return _laplace_tail_mse(B, self.sensitivity, self.epsilon) - target_tail

        # Bisection bounds
        B_lo = 0.0
        B_hi = b * 50  # Very conservative upper bound

        # Check if B_hi is sufficient
        if objective(B_hi) > 0:
            B_hi = b * 200

        try:
            result = sp_optimize.brentq(objective, B_lo, B_hi, xtol=1e-8)
            B_star = max(result, self.sensitivity)
        except ValueError:
            # Fallback: use the common heuristic B = b * ln(1/alpha)
            B_star = b * math.log(1.0 / alpha) + self.sensitivity
            logger.warning(
                "Brent's method failed for range optimization; "
                "using fallback B=%.4f", B_star,
            )

        # Include query range if known
        if self.query_range is not None:
            q_min, q_max = self.query_range
            data_range = (q_max - q_min) / 2.0
            B_star = max(B_star, data_range + self.sensitivity)

        return B_star

    def _optimal_range_gaussian(self, alpha: float) -> float:
        """Optimal range for Gaussian mechanism."""
        # σ = Δf · √(2 ln(1.25/δ)) / ε
        sigma = self.sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon
        target_tail = alpha * sigma ** 2  # Baseline Gaussian MSE = σ²

        def objective(B: float) -> float:
            return _gaussian_tail_mse(B, sigma) - target_tail

        B_lo = 0.0
        B_hi = sigma * 20

        try:
            result = sp_optimize.brentq(objective, B_lo, B_hi, xtol=1e-8)
            B_star = max(result, self.sensitivity)
        except ValueError:
            B_star = sigma * math.sqrt(2.0 * math.log(1.0 / alpha))
            logger.warning(
                "Brent's method failed for Gaussian range; "
                "using fallback B=%.4f", B_star,
            )

        if self.query_range is not None:
            q_min, q_max = self.query_range
            data_range = (q_max - q_min) / 2.0
            B_star = max(B_star, data_range + self.sensitivity)

        return B_star

    def recommend_grid_size(
        self,
        target_error: float,
        *,
        alpha: float = 0.01,
        range_B: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Recommend grid size k for a target discretization error.

        Computes the minimum k such that the discretization error
        (excluding tail error) is ≤ target_error.

        For piecewise-constant:
            (Δy)² / 12 ≤ target_error
            Δy ≤ √(12 · target_error)
            k ≥ 2B / Δy + 1

        For piecewise-linear:
            (Δy)⁴ / 720 ≤ target_error
            Δy ≤ (720 · target_error)^(1/4)
            k ≥ 2B / Δy + 1

        Args:
            target_error: Maximum discretization error.
            alpha: Tail fraction for range optimization.
            range_B: Override output range.

        Returns:
            Dict with:
            - ``"k"``: Recommended grid size
            - ``"grid_spacing"``: Resulting Δy
            - ``"actual_error"``: Actual discretization error at k
            - ``"range_B"``: Range used
            - ``"total_error"``: Including tail contribution
        """
        if target_error <= 0:
            raise ConfigurationError(
                "target_error must be positive",
                parameter="target_error",
                value=target_error,
            )

        if range_B is None:
            range_B = self.optimal_range(alpha)

        # Compute max grid spacing for target error
        if self.mechanism_family == "piecewise_constant":
            max_spacing = math.sqrt(12.0 * target_error)
        else:
            max_spacing = (720.0 * target_error) ** 0.25

        # Minimum k
        k = max(2, int(math.ceil(2.0 * range_B / max_spacing)) + 1)

        # Actual metrics
        actual_spacing = 2.0 * range_B / max(k - 1, 1)
        if self.mechanism_family == "piecewise_constant":
            actual_disc_error = _piecewise_constant_error(actual_spacing)
        else:
            actual_disc_error = _piecewise_linear_error(actual_spacing)

        tail_error = _laplace_tail_mse(range_B, self.sensitivity, self.epsilon)

        return {
            "k": k,
            "grid_spacing": actual_spacing,
            "actual_error": actual_disc_error,
            "range_B": range_B,
            "total_error": actual_disc_error + tail_error,
            "tail_error": tail_error,
            "mechanism_family": self.mechanism_family,
        }

    def approximation_certificate(
        self,
        k: int,
        *,
        range_B: Optional[float] = None,
        alpha: float = 0.01,
    ) -> DiscretizationCertificate:
        """Generate an approximation certificate for a given grid size.

        Proves that the k-point discrete mechanism is within a
        bounded error of the continuous optimum.

        Args:
            k: Number of grid points.
            range_B: Output range half-width.
            alpha: Tail fraction for range optimization.

        Returns:
            A :class:`DiscretizationCertificate`.
        """
        if range_B is None:
            range_B = self.optimal_range(alpha)

        bounds = self.error_bound(k, range_B=range_B)

        return DiscretizationCertificate(
            k=k,
            grid_spacing=bounds["grid_spacing"],
            range_B=range_B,
            discretization_error=bounds["discretization_error"],
            tail_error=bounds["tail_error"],
            total_error_bound=bounds["total_error"],
            sensitivity=self.sensitivity,
            epsilon=self.epsilon,
            delta=self.delta,
            grid_type="uniform",
            mechanism_family=self.mechanism_family,
        )

    def compare_grid_types(
        self,
        k: int,
        *,
        range_B: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compare error bounds for uniform vs adaptive grids.

        The adaptive grid concentrates points near the mode (center)
        of the mechanism, reducing error where the density is highest.

        For adaptive grids with spacing proportional to the local
        density, the discretization error improves by approximately
        a factor of 2-3 compared to uniform grids.

        Args:
            k: Number of grid points.
            range_B: Output range half-width.

        Returns:
            Dict with "uniform" and "adaptive" sub-dicts of error bounds.
        """
        if range_B is None:
            range_B = self.optimal_range()

        # Uniform grid
        uniform_spacing = 2.0 * range_B / max(k - 1, 1)
        uniform_disc = _piecewise_constant_error(uniform_spacing)
        tail = _laplace_tail_mse(range_B, self.sensitivity, self.epsilon)

        # Adaptive grid: approximate improvement factor
        # For Laplace density, concentrating 50% of points in the
        # center b-width region reduces effective spacing by ~2x
        b = self._b
        # Fraction of probability in [-b, b]: 1 - exp(-1) ≈ 0.632
        center_frac = 1.0 - math.exp(-1.0)
        # Allocate center_frac * k points to [-b, b]
        k_center = max(2, int(center_frac * k))
        k_tail = max(2, k - k_center)

        if b > 0 and range_B > b:
            adaptive_center_spacing = 2.0 * b / max(k_center - 1, 1)
            adaptive_tail_spacing = 2.0 * (range_B - b) / max(k_tail - 1, 1)
            # Weighted average error (by probability mass)
            adaptive_disc = (
                center_frac * _piecewise_constant_error(adaptive_center_spacing)
                + (1 - center_frac) * _piecewise_constant_error(adaptive_tail_spacing)
            )
        else:
            adaptive_disc = uniform_disc

        return {
            "uniform": {
                "discretization_error": uniform_disc,
                "tail_error": tail,
                "total_error": uniform_disc + tail,
                "grid_spacing": uniform_spacing,
            },
            "adaptive": {
                "discretization_error": adaptive_disc,
                "tail_error": tail,
                "total_error": adaptive_disc + tail,
                "improvement_factor": uniform_disc / max(adaptive_disc, 1e-15),
            },
        }

    def convergence_table(
        self,
        k_values: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a convergence table showing error vs grid size.

        Useful for selecting k empirically.

        Args:
            k_values: Grid sizes to evaluate. If None, uses powers of 2
                from 8 to 2048.

        Returns:
            List of dicts with k, error bounds, and relative errors.
        """
        if k_values is None:
            k_values = [2 ** p for p in range(3, 12)]

        B = self.optimal_range()
        rows = []

        for k in k_values:
            bounds = self.error_bound(k, range_B=B)
            rows.append({
                "k": k,
                "grid_spacing": bounds["grid_spacing"],
                "discretization_error": bounds["discretization_error"],
                "tail_error": bounds["tail_error"],
                "total_error": bounds["total_error"],
                "relative_error": bounds["relative_error"],
            })

        return rows

    @classmethod
    def from_query_spec(cls, spec: QuerySpec) -> DiscretizationAnalyzer:
        """Construct from a QuerySpec.

        Args:
            spec: Query specification with epsilon, sensitivity, etc.

        Returns:
            A DiscretizationAnalyzer configured from the spec.
        """
        q_range = None
        if spec.query_values is not None and len(spec.query_values) > 0:
            q_range = (float(np.min(spec.query_values)),
                       float(np.max(spec.query_values)))

        return cls(
            epsilon=spec.epsilon,
            sensitivity=spec.sensitivity,
            delta=spec.delta,
            query_range=q_range,
        )

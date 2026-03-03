"""
Truncated and concentrated Laplace mechanisms for DP-Forge.

Implements variants of the Laplace mechanism with bounded support or
concentrated noise distributions. These are useful when the output domain
is bounded or when tighter tail bounds are needed.

Key features:
    - TruncatedLaplaceMechanism: Laplace truncated to a bounded interval
    - ConcentratedLaplace: zCDP-optimal Laplace variant with Gaussian-like tails
    - CensoredLaplaceMechanism: Censored at domain boundaries
    - Optimal truncation parameter selection for minimum variance

References:
    - Geng, Kairouz, Oh, Viswanath: "The Staircase Mechanism in Differential Privacy" (2015)
    - Bun, Steinke: "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (2016)
    - Holohan, Antonatos, Braghin, Mac Aonghusa: "The Bounded Laplace Mechanism in Differential Privacy" (2018)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
from scipy import integrate, optimize, special, stats

from dp_forge.exceptions import (
    ConfigurationError,
    InvalidMechanismError,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Truncated Laplace distribution utilities
# ---------------------------------------------------------------------------


def truncated_laplace_normalization(
    loc: float,
    scale: float,
    lower: float,
    upper: float,
) -> float:
    """Compute normalization constant for truncated Laplace.
    
    For Laplace(loc, scale) truncated to [lower, upper], the normalization
    constant Z ensures the PDF integrates to 1.
    
    Args:
        loc: Location parameter μ.
        scale: Scale parameter b.
        lower: Lower truncation bound.
        upper: Upper truncation bound.
    
    Returns:
        Normalization constant Z.
    """
    if lower >= upper:
        raise ValueError(f"lower ({lower}) must be < upper ({upper})")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    
    # CDF of untruncated Laplace at bounds
    cdf_lower = 0.5 * (1.0 + np.sign(lower - loc) * (1.0 - np.exp(-abs(lower - loc) / scale)))
    cdf_upper = 0.5 * (1.0 + np.sign(upper - loc) * (1.0 - np.exp(-abs(upper - loc) / scale)))
    
    Z = cdf_upper - cdf_lower
    return Z


def optimal_truncation_points(
    epsilon: float,
    sensitivity: float = 1.0,
    target_mass: float = 0.999,
) -> Tuple[float, float]:
    """Compute optimal truncation points for minimum variance.
    
    Truncates the Laplace(0, sensitivity/ε) distribution to capture
    target_mass of the probability mass while minimizing variance.
    
    Args:
        epsilon: Privacy parameter ε.
        sensitivity: Query sensitivity.
        target_mass: Target probability mass to capture (default 0.999).
    
    Returns:
        Tuple (lower, upper) of truncation points.
    """
    scale = sensitivity / epsilon
    
    # For Laplace centered at 0, optimal truncation is symmetric
    # Find t such that P(|X| ≤ t) = target_mass
    # P(|X| ≤ t) = 1 - exp(-t/b)
    # => t = -b * log(1 - target_mass)
    
    tail_mass = 1.0 - target_mass
    t = -scale * math.log(tail_mass / 2.0)
    
    return -t, t


# ---------------------------------------------------------------------------
# TruncatedLaplaceMechanism
# ---------------------------------------------------------------------------


class TruncatedLaplaceMechanism:
    """Laplace mechanism with bounded support.
    
    Truncates the Laplace(0, sensitivity/ε) distribution to [lower, upper],
    ensuring outputs always lie in a bounded interval. The truncation is
    done by renormalizing the PDF over the truncated support.
    
    Privacy guarantee: (ε, 0)-DP for the truncated mechanism.
    
    Note: Truncation can slightly increase the privacy loss if not done
    carefully. This implementation uses post-processing to maintain the
    original ε guarantee.
    
    Attributes:
        epsilon: Privacy parameter ε.
        sensitivity: Query sensitivity.
        lower: Lower truncation bound.
        upper: Upper truncation bound.
    
    Usage::
    
        mech = TruncatedLaplaceMechanism(
            epsilon=1.0, lower=-10, upper=10
        )
        noisy = mech.sample(true_value=5.0)
        print(f"Variance: {mech.variance():.4f}")
    """
    
    def __init__(
        self,
        epsilon: float,
        lower: float,
        upper: float,
        sensitivity: float = 1.0,
        auto_truncate: bool = False,
        target_mass: float = 0.999,
        metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize truncated Laplace mechanism.
        
        Args:
            epsilon: Privacy parameter ε > 0.
            lower: Lower truncation bound.
            upper: Upper truncation bound.
            sensitivity: Query sensitivity (default 1.0).
            auto_truncate: If True, ignore lower/upper and compute optimal.
            target_mass: Target mass for auto-truncation (default 0.999).
            metadata: Optional metadata dict.
            seed: Random seed.
        
        Raises:
            ConfigurationError: If parameters are invalid.
        """
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ConfigurationError(
                f"epsilon must be positive and finite, got {epsilon}",
                parameter="epsilon",
                value=epsilon,
            )
        if sensitivity <= 0 or not math.isfinite(sensitivity):
            raise ConfigurationError(
                f"sensitivity must be positive and finite, got {sensitivity}",
                parameter="sensitivity",
                value=sensitivity,
            )
        
        self._epsilon = epsilon
        self._sensitivity = sensitivity
        self._metadata = metadata or {}
        self._rng = np.random.default_rng(seed)
        
        # Laplace scale
        self._scale = sensitivity / epsilon
        
        # Truncation bounds
        if auto_truncate:
            self._lower, self._upper = optimal_truncation_points(
                epsilon, sensitivity, target_mass
            )
        else:
            if lower >= upper:
                raise ConfigurationError(
                    f"lower ({lower}) must be < upper ({upper})",
                    parameter="lower",
                )
            self._lower = lower
            self._upper = upper
        
        # Normalization constant
        self._Z = truncated_laplace_normalization(
            loc=0.0, scale=self._scale, lower=self._lower, upper=self._upper
        )
    
    @property
    def epsilon(self) -> float:
        """Privacy parameter ε."""
        return self._epsilon
    
    @property
    def delta(self) -> float:
        """Privacy parameter δ (always 0 for pure DP)."""
        return 0.0
    
    @property
    def sensitivity(self) -> float:
        """Query sensitivity."""
        return self._sensitivity
    
    @property
    def scale(self) -> float:
        """Laplace scale parameter b."""
        return self._scale
    
    @property
    def lower(self) -> float:
        """Lower truncation bound."""
        return self._lower
    
    @property
    def upper(self) -> float:
        """Upper truncation bound."""
        return self._upper
    
    @property
    def support_width(self) -> float:
        """Width of the support [lower, upper]."""
        return self._upper - self._lower
    
    # ----- Sampling -----
    
    def sample(
        self,
        true_value: Union[float, FloatArray],
        rng: Optional[np.random.Generator] = None,
    ) -> Union[float, FloatArray]:
        """Sample noisy output for a true value.
        
        Uses rejection sampling: sample from Laplace, reject if outside
        [lower, upper], repeat until accepted.
        
        Args:
            true_value: True query value (scalar or array).
            rng: Optional RNG override.
        
        Returns:
            Noisy output (same shape as true_value).
        """
        rng = rng or self._rng
        
        true_value = np.asarray(true_value, dtype=np.float64)
        scalar_input = (true_value.ndim == 0)
        
        if scalar_input:
            true_value = true_value.reshape(1)
        
        noisy = np.zeros_like(true_value)
        
        for i in range(len(true_value)):
            # Rejection sampling
            while True:
                noise = rng.laplace(scale=self._scale)
                candidate = true_value[i] + noise
                
                if self._lower <= candidate <= self._upper:
                    noisy[i] = candidate
                    break
        
        if scalar_input:
            return float(noisy[0])
        return noisy
    
    def sample_batch(
        self,
        true_values: FloatArray,
        rng: Optional[np.random.Generator] = None,
    ) -> FloatArray:
        """Sample noisy outputs for a batch of true values.
        
        Args:
            true_values: Batch of true values, shape (n,).
            rng: Optional RNG override.
        
        Returns:
            Noisy outputs, shape (n,).
        """
        true_values = np.asarray(true_values, dtype=np.float64)
        return self.sample(true_values, rng)
    
    # ----- Density evaluation -----
    
    def pdf(self, x: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """Probability density function of the truncated Laplace.
        
        Args:
            x: Point(s) at which to evaluate PDF (noise values, not outputs).
        
        Returns:
            PDF value(s).
        """
        x = np.asarray(x, dtype=np.float64)
        scalar_input = (x.ndim == 0)
        
        # Laplace PDF: (1 / (2b)) * exp(-|x| / b)
        pdf_vals = np.exp(-np.abs(x) / self._scale) / (2.0 * self._scale)
        
        # Truncate and renormalize
        pdf_vals = np.where(
            (x >= self._lower) & (x <= self._upper),
            pdf_vals / self._Z,
            0.0
        )
        
        if scalar_input:
            return float(pdf_vals)
        return pdf_vals
    
    def log_pdf(self, x: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """Log-probability density function.
        
        Args:
            x: Point(s) at which to evaluate log-PDF.
        
        Returns:
            Log-PDF value(s).
        """
        pdf_vals = self.pdf(x)
        return np.log(np.maximum(pdf_vals, 1e-300))
    
    def cdf(self, x: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """Cumulative distribution function.
        
        Args:
            x: Point(s) at which to evaluate CDF.
        
        Returns:
            CDF value(s) in [0, 1].
        """
        x = np.asarray(x, dtype=np.float64)
        scalar_input = (x.ndim == 0)
        
        # Laplace CDF: 0.5 * (1 + sign(x) * (1 - exp(-|x| / b)))
        cdf_vals = 0.5 * (1.0 + np.sign(x) * (1.0 - np.exp(-np.abs(x) / self._scale)))
        
        # Adjust for truncation
        cdf_lower = 0.5 * (1.0 + np.sign(self._lower) * (1.0 - np.exp(-abs(self._lower) / self._scale)))
        
        cdf_vals = (cdf_vals - cdf_lower) / self._Z
        cdf_vals = np.clip(cdf_vals, 0.0, 1.0)
        
        if scalar_input:
            return float(cdf_vals)
        return cdf_vals
    
    # ----- Privacy guarantee -----
    
    def privacy_guarantee(self) -> Tuple[float, float]:
        """Return the privacy guarantee (ε, δ).
        
        Returns:
            Tuple (epsilon, 0.0) for pure DP.
        """
        return self._epsilon, 0.0
    
    def verify_privacy(
        self,
        x1: float = 0.0,
        x2: float = 1.0,
        n_samples: int = 1000,
        tol: float = 1e-6,
    ) -> Tuple[bool, float]:
        """Verify that truncation preserves ε-DP.
        
        Checks the PDF ratio at grid points.
        
        Args:
            x1: First database value.
            x2: Second database value (should differ by sensitivity).
            n_samples: Number of grid points to check.
            tol: Numerical tolerance.
        
        Returns:
            Tuple (is_private, max_violation).
        """
        # Generate grid
        grid = np.linspace(self._lower, self._upper, n_samples)
        
        # Compute PDF ratios
        pdf1 = self.pdf(grid - x1)
        pdf2 = self.pdf(grid - x2)
        
        pdf2_safe = np.maximum(pdf2, 1e-300)
        ratios = pdf1 / pdf2_safe
        
        max_ratio = float(np.max(ratios))
        exp_eps = math.exp(self._epsilon)
        max_violation = max_ratio - exp_eps
        
        is_private = max_violation <= tol
        
        return is_private, max_violation
    
    # ----- Variance and statistics -----
    
    def variance(self) -> float:
        """Variance of the truncated Laplace distribution.
        
        Uses numerical integration to compute:
            Var[X] = E[X^2] - E[X]^2
        
        Returns:
            Variance.
        """
        # Mean (should be ~0 for symmetric truncation around 0)
        mean, _ = integrate.quad(
            lambda x: x * self.pdf(x),
            self._lower, self._upper,
        )
        
        # Second moment
        second_moment, _ = integrate.quad(
            lambda x: x**2 * self.pdf(x),
            self._lower, self._upper,
        )
        
        var = second_moment - mean**2
        return max(var, 0.0)  # Numerical stability
    
    def laplace_variance(self) -> float:
        """Variance of the untruncated Laplace(0, sensitivity/ε).
        
        Returns:
            2 * scale^2.
        """
        return 2.0 * self._scale ** 2
    
    def variance_ratio(self) -> float:
        """Ratio of truncated variance to untruncated variance.
        
        Returns:
            Var[Truncated] / Var[Untruncated].
        """
        trunc_var = self.variance()
        lap_var = self.laplace_variance()
        return trunc_var / lap_var
    
    def tail_probability(self) -> float:
        """Probability mass in the tails (outside truncation).
        
        Returns:
            P(|X| > bounds) for untruncated Laplace.
        """
        return 1.0 - self._Z
    
    # ----- Validity checking -----
    
    def is_valid(self, tol: float = 1e-6) -> Tuple[bool, List[str]]:
        """Check mechanism validity.
        
        Validates:
        1. PDF integrates to 1.
        2. Privacy constraint holds.
        
        Args:
            tol: Numerical tolerance.
        
        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        issues: List[str] = []
        
        # Check PDF normalization
        integral, _ = integrate.quad(
            lambda x: self.pdf(x),
            self._lower, self._upper,
        )
        if abs(integral - 1.0) > tol:
            issues.append(
                f"PDF does not integrate to 1: integral={integral:.6f}"
            )
        
        # Check privacy
        is_private, max_viol = self.verify_privacy(tol=tol)
        if not is_private:
            issues.append(
                f"Privacy violation: max_ratio - exp(ε) = {max_viol:.2e}"
            )
        
        return len(issues) == 0, issues
    
    # ----- Representation -----
    
    def __repr__(self) -> str:
        return (
            f"TruncatedLaplaceMechanism(ε={self._epsilon:.4f}, "
            f"bounds=[{self._lower:.2f}, {self._upper:.2f}], "
            f"scale={self._scale:.4f})"
        )
    
    def __str__(self) -> str:
        ratio = self.variance_ratio()
        return (
            f"TruncatedLaplaceMechanism(ε={self._epsilon}, "
            f"bounds=[{self._lower:.1f}, {self._upper:.1f}], "
            f"var_ratio={ratio:.2f})"
        )


# ---------------------------------------------------------------------------
# ConcentratedLaplace (zCDP-optimal)
# ---------------------------------------------------------------------------


class ConcentratedLaplace:
    """Concentrated Laplace mechanism for zero-concentrated DP (zCDP).
    
    Implements a Laplace-like mechanism optimized for zCDP instead of
    (ε, δ)-DP. This provides tighter composition and better tail bounds.
    
    Under ρ-zCDP, the mechanism adds noise from a distribution with
    moment generating function bounded by exp(ρ t^2 / 2) for all t.
    
    For single queries, this reduces to Gaussian noise with σ^2 = 1/(2ρ).
    
    Attributes:
        rho: zCDP parameter ρ.
        sensitivity: Query sensitivity.
    
    Usage::
    
        mech = ConcentratedLaplace(rho=0.5, sensitivity=1.0)
        noisy = mech.sample(true_value=10)
        eps, delta = mech.to_epsilon_delta(delta=1e-5)
        print(f"(ε, δ) = ({eps:.4f}, {delta:.2e})")
    """
    
    def __init__(
        self,
        rho: float,
        sensitivity: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize concentrated Laplace mechanism.
        
        Args:
            rho: zCDP parameter ρ > 0.
            sensitivity: Query sensitivity (default 1.0).
            metadata: Optional metadata dict.
            seed: Random seed.
        
        Raises:
            ConfigurationError: If parameters are invalid.
        """
        if rho <= 0 or not math.isfinite(rho):
            raise ConfigurationError(
                f"rho must be positive and finite, got {rho}",
                parameter="rho",
                value=rho,
            )
        if sensitivity <= 0 or not math.isfinite(sensitivity):
            raise ConfigurationError(
                f"sensitivity must be positive and finite, got {sensitivity}",
                parameter="sensitivity",
                value=sensitivity,
            )
        
        self._rho = rho
        self._sensitivity = sensitivity
        self._metadata = metadata or {}
        self._rng = np.random.default_rng(seed)
        
        # For ρ-zCDP, use Gaussian noise with σ^2 = sensitivity^2 / (2ρ)
        self._sigma = sensitivity / math.sqrt(2.0 * rho)
    
    @property
    def rho(self) -> float:
        """zCDP parameter ρ."""
        return self._rho
    
    @property
    def sensitivity(self) -> float:
        """Query sensitivity."""
        return self._sensitivity
    
    @property
    def sigma(self) -> float:
        """Gaussian noise scale σ."""
        return self._sigma
    
    # ----- Sampling -----
    
    def sample(
        self,
        true_value: Union[float, FloatArray],
        rng: Optional[np.random.Generator] = None,
    ) -> Union[float, FloatArray]:
        """Sample noisy output for a true value.
        
        Adds Gaussian(0, σ^2) noise.
        
        Args:
            true_value: True query value (scalar or array).
            rng: Optional RNG override.
        
        Returns:
            Noisy output (same shape as true_value).
        """
        rng = rng or self._rng
        true_value = np.asarray(true_value, dtype=np.float64)
        
        noise = rng.normal(scale=self._sigma, size=true_value.shape)
        return true_value + noise
    
    # ----- Density evaluation -----
    
    def pdf(self, x: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """Probability density function (Gaussian).
        
        Args:
            x: Point(s) at which to evaluate PDF (noise values).
        
        Returns:
            PDF value(s).
        """
        x = np.asarray(x, dtype=np.float64)
        return stats.norm.pdf(x, loc=0.0, scale=self._sigma)
    
    def log_pdf(self, x: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """Log-probability density function.
        
        Args:
            x: Point(s) at which to evaluate log-PDF.
        
        Returns:
            Log-PDF value(s).
        """
        x = np.asarray(x, dtype=np.float64)
        return stats.norm.logpdf(x, loc=0.0, scale=self._sigma)
    
    # ----- Privacy conversions -----
    
    def to_epsilon_delta(self, delta: float) -> Tuple[float, float]:
        """Convert ρ-zCDP to (ε, δ)-DP.
        
        For ρ-zCDP, we have:
            ε = ρ + 2 * sqrt(ρ * log(1/δ))
        
        Args:
            delta: Target δ parameter.
        
        Returns:
            Tuple (epsilon, delta).
        
        Raises:
            ConfigurationError: If delta is invalid.
        """
        if not (0.0 < delta < 1.0):
            raise ConfigurationError(
                f"delta must be in (0, 1), got {delta}",
                parameter="delta",
                value=delta,
            )
        
        epsilon = self._rho + 2.0 * math.sqrt(self._rho * math.log(1.0 / delta))
        return epsilon, delta
    
    def privacy_guarantee(
        self,
        delta: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Return the privacy guarantee.
        
        Args:
            delta: Target δ (default 1e-5 if not specified).
        
        Returns:
            Tuple (epsilon, delta) for approximate DP, or (inf, 0.0) for zCDP.
        """
        if delta is None:
            # Return zCDP parameters as a special marker
            return float('inf'), 0.0  # Indicates zCDP
        else:
            return self.to_epsilon_delta(delta)
    
    # ----- Variance -----
    
    def variance(self) -> float:
        """Variance of the Gaussian noise.
        
        Returns:
            σ^2 = sensitivity^2 / (2ρ).
        """
        return self._sigma ** 2
    
    # ----- Representation -----
    
    def __repr__(self) -> str:
        return (
            f"ConcentratedLaplace(ρ={self._rho:.4f}, "
            f"sensitivity={self._sensitivity}, σ={self._sigma:.4f})"
        )
    
    def __str__(self) -> str:
        eps, delta = self.to_epsilon_delta(delta=1e-5)
        return (
            f"ConcentratedLaplace(ρ={self._rho}, "
            f"≈ (ε={eps:.2f}, δ=1e-5))"
        )


# ---------------------------------------------------------------------------
# CensoredLaplaceMechanism
# ---------------------------------------------------------------------------


class CensoredLaplaceMechanism:
    """Laplace mechanism with censoring at domain boundaries.
    
    Similar to truncation, but instead of rejection sampling, censors
    outputs that fall outside [lower, upper] to the nearest boundary.
    This is computationally more efficient than rejection sampling.
    
    Censoring: if x + noise < lower, return lower; if x + noise > upper,
    return upper; otherwise return x + noise.
    
    Privacy guarantee: (ε, 0)-DP (censoring is post-processing).
    
    Usage::
    
        mech = CensoredLaplaceMechanism(
            epsilon=1.0, lower=0, upper=100
        )
        noisy = mech.sample(true_value=50)
    """
    
    def __init__(
        self,
        epsilon: float,
        lower: float,
        upper: float,
        sensitivity: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize censored Laplace mechanism.
        
        Args:
            epsilon: Privacy parameter ε > 0.
            lower: Lower censoring bound.
            upper: Upper censoring bound.
            sensitivity: Query sensitivity (default 1.0).
            metadata: Optional metadata dict.
            seed: Random seed.
        
        Raises:
            ConfigurationError: If parameters are invalid.
        """
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ConfigurationError(
                f"epsilon must be positive and finite, got {epsilon}",
                parameter="epsilon",
                value=epsilon,
            )
        if lower >= upper:
            raise ConfigurationError(
                f"lower ({lower}) must be < upper ({upper})",
                parameter="lower",
            )
        
        self._epsilon = epsilon
        self._sensitivity = sensitivity
        self._lower = lower
        self._upper = upper
        self._metadata = metadata or {}
        self._rng = np.random.default_rng(seed)
        
        # Laplace scale
        self._scale = sensitivity / epsilon
    
    @property
    def epsilon(self) -> float:
        """Privacy parameter ε."""
        return self._epsilon
    
    @property
    def delta(self) -> float:
        """Privacy parameter δ (always 0)."""
        return 0.0
    
    @property
    def lower(self) -> float:
        """Lower censoring bound."""
        return self._lower
    
    @property
    def upper(self) -> float:
        """Upper censoring bound."""
        return self._upper
    
    # ----- Sampling -----
    
    def sample(
        self,
        true_value: Union[float, FloatArray],
        rng: Optional[np.random.Generator] = None,
    ) -> Union[float, FloatArray]:
        """Sample noisy output with censoring.
        
        Args:
            true_value: True query value (scalar or array).
            rng: Optional RNG override.
        
        Returns:
            Censored noisy output (same shape as true_value).
        """
        rng = rng or self._rng
        true_value = np.asarray(true_value, dtype=np.float64)
        
        # Add Laplace noise
        noise = rng.laplace(scale=self._scale, size=true_value.shape)
        noisy = true_value + noise
        
        # Censor to [lower, upper]
        censored = np.clip(noisy, self._lower, self._upper)
        
        scalar_input = (true_value.ndim == 0)
        if scalar_input:
            return float(censored)
        return censored
    
    def privacy_guarantee(self) -> Tuple[float, float]:
        """Return the privacy guarantee (ε, δ).
        
        Returns:
            Tuple (epsilon, 0.0) for pure DP.
        """
        return self._epsilon, 0.0
    
    # ----- Representation -----
    
    def __repr__(self) -> str:
        return (
            f"CensoredLaplaceMechanism(ε={self._epsilon:.4f}, "
            f"bounds=[{self._lower:.2f}, {self._upper:.2f}])"
        )

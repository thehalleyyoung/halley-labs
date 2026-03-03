"""
Exact and approximate Rényi divergence computation.

Implements multiple computation strategies for the Rényi divergence of
order α between two probability distributions:

- **Discrete exact**: Log-sum-exp over probability mass vectors.
- **Gaussian**: Closed-form formula for two Gaussian distributions.
- **Laplace**: Closed-form formula for two Laplace distributions.
- **Numerical quadrature**: Adaptive quadrature for arbitrary density pairs.
- **KL divergence**: Limiting case α → 1.
- **Max divergence**: Limiting case α → ∞.

All computations use log-domain arithmetic for numerical stability.

References:
    - Rényi, A. (1961). On measures of entropy and information.
    - Mironov, I. (2017). Rényi differential privacy.
    - van Erven, T. & Harremoës, P. (2014). Rényi divergence and
      Kullback-Leibler divergence.
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
    ConfigurationError,
    NumericalInstabilityError,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_DOMAIN_MIN = -700.0  # floor to avoid underflow in exp()
_LOG_DOMAIN_MAX = 700.0   # ceiling to avoid overflow in exp()
_ALPHA_KL_THRESHOLD = 1e-6  # |α - 1| below this triggers KL branch


# ---------------------------------------------------------------------------
# Numerically stable helpers
# ---------------------------------------------------------------------------


def _logsumexp(a: FloatArray) -> float:
    """Numerically stable log-sum-exp.

    Args:
        a: 1-D array of log-values.

    Returns:
        log(sum(exp(a))).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    if len(a) == 0:
        return -np.inf
    a_max = float(np.max(a))
    if not np.isfinite(a_max):
        return a_max
    return float(a_max + np.log(np.sum(np.exp(a - a_max))))


def _log_subtract_exp(log_a: float, log_b: float) -> float:
    """Compute log(exp(log_a) - exp(log_b)) stably, assuming log_a >= log_b.

    Args:
        log_a: Log of the larger value.
        log_b: Log of the smaller value.

    Returns:
        log(exp(log_a) - exp(log_b)).
    """
    if log_b == -np.inf:
        return log_a
    if log_a < log_b:
        return -np.inf
    return log_a + np.log1p(-np.exp(log_b - log_a))


def _safe_log(x: float) -> float:
    """Safe logarithm that returns -inf for zero and raises for negative."""
    if x < 0:
        raise ValueError(f"Cannot take log of negative value {x}")
    if x == 0:
        return -np.inf
    return math.log(x)


def _validate_alpha(alpha: float) -> None:
    """Validate that alpha is a valid Rényi divergence order."""
    if not math.isfinite(alpha):
        if alpha == float("inf"):
            return  # α = ∞ is valid (max divergence)
        raise ConfigurationError(
            f"alpha must be finite or +inf, got {alpha}",
            parameter="alpha",
            value=alpha,
        )
    if alpha < 0:
        raise ConfigurationError(
            f"alpha must be >= 0, got {alpha}",
            parameter="alpha",
            value=alpha,
        )


def _validate_distribution(p: FloatArray, name: str = "p") -> FloatArray:
    """Validate and normalise a discrete probability distribution.

    Args:
        p: Array of probabilities.
        name: Name for error messages.

    Returns:
        Validated float64 array.

    Raises:
        ConfigurationError: If the distribution is invalid.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    if len(p) == 0:
        raise ConfigurationError(
            f"Distribution {name} must be non-empty",
            parameter=name,
        )
    if np.any(p < 0):
        raise ConfigurationError(
            f"Distribution {name} contains negative values (min={np.min(p):.2e})",
            parameter=name,
            value=float(np.min(p)),
        )
    total = float(np.sum(p))
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ConfigurationError(
            f"Distribution {name} does not sum to 1 (sum={total:.8f})",
            parameter=name,
            value=total,
            constraint="sum(p) ≈ 1",
        )
    return p


# =========================================================================
# RenyiDivergenceComputer
# =========================================================================


@dataclass
class RenyiDivergenceResult:
    """Result of a Rényi divergence computation.

    Attributes:
        alpha: Order of the Rényi divergence.
        divergence: Computed divergence value.
        method: Computation method used.
        log_divergence: Divergence in log-domain (for numerical tracking).
        metadata: Additional computation metadata.
    """

    alpha: float
    divergence: float
    method: str
    log_divergence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"RenyiDivergenceResult(α={self.alpha}, D={self.divergence:.6f}, "
            f"method={self.method!r})"
        )


class RenyiDivergenceComputer:
    """Compute Rényi divergence between probability distributions.

    Supports exact computation for discrete distributions, closed-form
    formulas for Gaussian and Laplace distributions, and numerical
    quadrature for arbitrary continuous density pairs.

    All computations use log-domain arithmetic for numerical stability.
    Edge cases (α → 1 for KL divergence, α → ∞ for max divergence)
    are handled explicitly.

    Args:
        min_prob: Minimum probability threshold below which values are
            treated as zero. Default ``1e-300``.
        quadrature_points: Number of quadrature points for numerical
            integration. Default ``10000``.

    Example::

        computer = RenyiDivergenceComputer()
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        result = computer.exact_discrete(p, q, alpha=2.0)
        print(result.divergence)
    """

    def __init__(
        self,
        min_prob: float = 1e-300,
        quadrature_points: int = 10000,
    ) -> None:
        if min_prob <= 0:
            raise ConfigurationError(
                f"min_prob must be > 0, got {min_prob}",
                parameter="min_prob",
                value=min_prob,
            )
        if quadrature_points < 100:
            raise ConfigurationError(
                f"quadrature_points must be >= 100, got {quadrature_points}",
                parameter="quadrature_points",
                value=quadrature_points,
            )
        self._min_prob = min_prob
        self._quadrature_points = quadrature_points

    # -----------------------------------------------------------------
    # Discrete exact computation
    # -----------------------------------------------------------------

    def exact_discrete(
        self,
        p: FloatArray,
        q: FloatArray,
        alpha: float,
    ) -> RenyiDivergenceResult:
        """Exact Rényi divergence for discrete distributions via log-sum-exp.

        Computes D_α(P || Q) = (1/(α-1)) log Σ_x p(x)^α q(x)^(1-α)

        For α = 1, computes KL divergence: Σ_x p(x) log(p(x)/q(x)).
        For α = ∞, computes max divergence: max_x log(p(x)/q(x)).
        For α = 0, computes -log(Q(support(P))).

        Args:
            p: Probability mass function P, shape ``(n,)``.
            q: Probability mass function Q, shape ``(n,)``.
            alpha: Rényi divergence order (≥ 0 or ∞).

        Returns:
            RenyiDivergenceResult with exact divergence.

        Raises:
            ConfigurationError: If inputs are invalid.
        """
        _validate_alpha(alpha)
        p = _validate_distribution(p, "p")
        q = _validate_distribution(q, "q")

        if len(p) != len(q):
            raise ConfigurationError(
                f"p (len={len(p)}) and q (len={len(q)}) must have same length",
                parameter="p/q",
            )

        # Handle special α values
        if alpha == float("inf"):
            return self._max_divergence_discrete(p, q)
        if alpha == 0.0:
            return self._zero_order_discrete(p, q)
        if abs(alpha - 1.0) < _ALPHA_KL_THRESHOLD:
            return self._kl_divergence_discrete(p, q)

        # General case: D_α(P||Q) = 1/(α-1) * log(Σ p^α * q^(1-α))
        log_p = np.log(np.maximum(p, self._min_prob))
        log_q = np.log(np.maximum(q, self._min_prob))

        # Mask where both p and q are effectively zero
        mask = (p > self._min_prob) | (q > self._min_prob)

        # log(p^α * q^(1-α)) = α*log(p) + (1-α)*log(q)
        log_terms = alpha * log_p + (1.0 - alpha) * log_q

        # Only include terms where at least one distribution has support
        if not np.any(mask):
            return RenyiDivergenceResult(
                alpha=alpha, divergence=0.0, method="exact_discrete",
            )

        # Handle case where q=0 but p>0 (divergence is +inf for α>1)
        if alpha > 1.0:
            q_zero_p_positive = (q <= self._min_prob) & (p > self._min_prob)
            if np.any(q_zero_p_positive):
                return RenyiDivergenceResult(
                    alpha=alpha, divergence=float("inf"),
                    method="exact_discrete",
                    metadata={"reason": "q=0 where p>0 with alpha>1"},
                )

        log_terms_valid = log_terms[mask]
        log_sum = _logsumexp(log_terms_valid)
        divergence = log_sum / (alpha - 1.0)

        return RenyiDivergenceResult(
            alpha=alpha,
            divergence=max(divergence, 0.0),
            method="exact_discrete",
            log_divergence=log_sum,
        )

    def exact_discrete_vectorized(
        self,
        p: FloatArray,
        q: FloatArray,
        alphas: FloatArray,
    ) -> FloatArray:
        """Vectorized Rényi divergence computation across multiple α orders.

        Args:
            p: Probability mass function P, shape ``(n,)``.
            q: Probability mass function Q, shape ``(n,)``.
            alphas: Array of α orders, shape ``(m,)``.

        Returns:
            Array of divergence values, shape ``(m,)``.
        """
        p = _validate_distribution(p, "p")
        q = _validate_distribution(q, "q")
        alphas = np.asarray(alphas, dtype=np.float64)

        results = np.empty(len(alphas), dtype=np.float64)
        log_p = np.log(np.maximum(p, self._min_prob))
        log_q = np.log(np.maximum(q, self._min_prob))

        q_zero_p_positive = (q <= self._min_prob) & (p > self._min_prob)
        has_zero_q = np.any(q_zero_p_positive)

        for i, alpha in enumerate(alphas):
            if alpha == float("inf"):
                res = self._max_divergence_discrete(p, q)
                results[i] = res.divergence
            elif abs(alpha - 1.0) < _ALPHA_KL_THRESHOLD:
                res = self._kl_divergence_discrete(p, q)
                results[i] = res.divergence
            elif alpha == 0.0:
                res = self._zero_order_discrete(p, q)
                results[i] = res.divergence
            elif alpha > 1.0 and has_zero_q:
                results[i] = float("inf")
            else:
                log_terms = alpha * log_p + (1.0 - alpha) * log_q
                mask = (p > self._min_prob) | (q > self._min_prob)
                log_sum = _logsumexp(log_terms[mask])
                results[i] = max(log_sum / (alpha - 1.0), 0.0)

        return results

    # -----------------------------------------------------------------
    # Closed-form: Gaussian
    # -----------------------------------------------------------------

    def gaussian(
        self,
        mu1: float,
        sigma1: float,
        mu2: float,
        sigma2: float,
        alpha: float,
    ) -> RenyiDivergenceResult:
        """Rényi divergence between two univariate Gaussian distributions.

        D_α(N(μ₁,σ₁²) || N(μ₂,σ₂²)) =
            α(μ₁-μ₂)² / (2σ_α²) + 1/(2(α-1)) * log(σ_α² / (σ₁^(2α) σ₂^(2(1-α))))

        where σ_α² = α·σ₂² + (1-α)·σ₁².

        Special cases:
            - Equal variances: α(μ₁-μ₂)² / (2σ²)
            - α → 1: KL divergence
            - α → ∞: max divergence

        Args:
            mu1: Mean of P.
            sigma1: Standard deviation of P (> 0).
            mu2: Mean of Q.
            sigma2: Standard deviation of Q (> 0).
            alpha: Rényi divergence order (> 0 or ∞).

        Returns:
            RenyiDivergenceResult with closed-form divergence.
        """
        _validate_alpha(alpha)
        if sigma1 <= 0 or sigma2 <= 0:
            raise ConfigurationError(
                f"Standard deviations must be positive, got σ₁={sigma1}, σ₂={sigma2}",
                parameter="sigma",
            )

        # α = ∞: max divergence
        if alpha == float("inf"):
            return self._max_divergence_gaussian(mu1, sigma1, mu2, sigma2)

        # α → 1: KL divergence
        if abs(alpha - 1.0) < _ALPHA_KL_THRESHOLD:
            return self._kl_gaussian(mu1, sigma1, mu2, sigma2)

        var1 = sigma1 ** 2
        var2 = sigma2 ** 2

        # σ_α² = (1-α)σ₁² + α·σ₂²
        var_alpha = (1.0 - alpha) * var1 + alpha * var2

        # Check for degenerate case (var_alpha ≤ 0 only for α > 1 with certain variances)
        if var_alpha <= 0:
            return RenyiDivergenceResult(
                alpha=alpha, divergence=float("inf"), method="gaussian",
                metadata={"reason": "degenerate variance mixture"},
            )

        # D_α = α(μ₁-μ₂)²/(2σ_α²) + log(σ_α / (σ₁^α · σ₂^(1-α))) / (α-1)
        mean_term = alpha * (mu1 - mu2) ** 2 / (2.0 * var_alpha)

        # log-domain variance term
        log_var_alpha = math.log(var_alpha)
        log_var1 = math.log(var1)
        log_var2 = math.log(var2)
        log_ratio = 0.5 * (log_var_alpha - alpha * log_var1 - (1.0 - alpha) * log_var2)
        var_term = log_ratio / (alpha - 1.0)

        divergence = mean_term + var_term

        return RenyiDivergenceResult(
            alpha=alpha,
            divergence=max(divergence, 0.0),
            method="gaussian",
            metadata={"mu_diff": mu1 - mu2, "var_alpha": var_alpha},
        )

    def gaussian_same_variance(
        self,
        mu1: float,
        mu2: float,
        sigma: float,
        alpha: float,
    ) -> RenyiDivergenceResult:
        """Rényi divergence between Gaussians with equal variance.

        D_α(N(μ₁,σ²) || N(μ₂,σ²)) = α(μ₁-μ₂)² / (2σ²)

        This simplified formula is exact for all α > 0.

        Args:
            mu1: Mean of P.
            mu2: Mean of Q.
            sigma: Common standard deviation (> 0).
            alpha: Rényi divergence order (> 0).

        Returns:
            RenyiDivergenceResult with exact divergence.
        """
        _validate_alpha(alpha)
        if sigma <= 0:
            raise ConfigurationError(
                f"sigma must be positive, got {sigma}",
                parameter="sigma",
                value=sigma,
            )

        if alpha == float("inf"):
            return RenyiDivergenceResult(
                alpha=alpha, divergence=float("inf") if mu1 != mu2 else 0.0,
                method="gaussian_same_variance",
            )
        if abs(alpha - 1.0) < _ALPHA_KL_THRESHOLD:
            alpha = 1.0

        divergence = alpha * (mu1 - mu2) ** 2 / (2.0 * sigma ** 2)

        return RenyiDivergenceResult(
            alpha=alpha,
            divergence=divergence,
            method="gaussian_same_variance",
        )

    # -----------------------------------------------------------------
    # Closed-form: Laplace
    # -----------------------------------------------------------------

    def laplace(
        self,
        mu1: float,
        b1: float,
        mu2: float,
        b2: float,
        alpha: float,
    ) -> RenyiDivergenceResult:
        """Rényi divergence between two Laplace distributions.

        For Laplace(μ₁, b₁) and Laplace(μ₂, b₂), the Rényi divergence of
        order α is computed via the moment generating function of the
        log-likelihood ratio.

        For the special case b₁ = b₂ = b:
            D_α(Lap(μ₁,b) || Lap(μ₂,b)) uses a closed-form involving
            the difference |μ₁ - μ₂| and scale b.

        Args:
            mu1: Location of P.
            b1: Scale of P (> 0).
            mu2: Location of Q.
            b2: Scale of Q (> 0).
            alpha: Rényi divergence order (> 0 or ∞).

        Returns:
            RenyiDivergenceResult with divergence value.
        """
        _validate_alpha(alpha)
        if b1 <= 0 or b2 <= 0:
            raise ConfigurationError(
                f"Scale parameters must be positive, got b₁={b1}, b₂={b2}",
                parameter="scale",
            )

        if alpha == float("inf"):
            return self._max_divergence_laplace(mu1, b1, mu2, b2)

        if abs(alpha - 1.0) < _ALPHA_KL_THRESHOLD:
            return self._kl_laplace(mu1, b1, mu2, b2)

        # Same scale simplification
        if abs(b1 - b2) < 1e-15 * max(b1, b2):
            return self._laplace_same_scale(mu1, mu2, b1, alpha)

        # General case: numerical via quadrature
        return self._laplace_general(mu1, b1, mu2, b2, alpha)

    def _laplace_same_scale(
        self,
        mu1: float,
        mu2: float,
        b: float,
        alpha: float,
    ) -> RenyiDivergenceResult:
        """Rényi divergence for Laplace distributions with equal scale.

        Uses the formula from the Laplace mechanism's privacy loss
        distribution. The key integral evaluates to a closed form
        involving exponentials.
        """
        d = abs(mu1 - mu2)
        if d == 0:
            return RenyiDivergenceResult(
                alpha=alpha, divergence=0.0, method="laplace_same_scale",
            )

        # The RDP of Laplace with parameter ε = d/b uses the exact formula
        eps = d / b

        # ε_RDP(α) = 1/(α-1) log( α/(2α-1) exp((α-1)ε) + (α-1)/(2α-1) exp(-(α-1)ε) )
        a_minus_1 = alpha - 1.0
        denom = 2.0 * alpha - 1.0

        if denom <= 0:
            # α < 0.5, use numerical quadrature
            return self._laplace_general(mu1, b, mu2, b, alpha)

        log_coeff1 = math.log(alpha / denom)
        log_coeff2 = math.log(a_minus_1 / denom) if a_minus_1 > 0 else -np.inf

        log_term1 = log_coeff1 + a_minus_1 * eps
        log_term2 = log_coeff2 - a_minus_1 * eps

        log_sum = _logsumexp(np.array([log_term1, log_term2]))
        divergence = log_sum / a_minus_1

        return RenyiDivergenceResult(
            alpha=alpha,
            divergence=max(divergence, 0.0),
            method="laplace_same_scale",
            metadata={"epsilon": eps},
        )

    def _laplace_general(
        self,
        mu1: float,
        b1: float,
        mu2: float,
        b2: float,
        alpha: float,
    ) -> RenyiDivergenceResult:
        """General Laplace Rényi divergence via numerical quadrature."""
        def log_p(x: FloatArray) -> FloatArray:
            return -np.log(2.0 * b1) - np.abs(x - mu1) / b1

        def log_q(x: FloatArray) -> FloatArray:
            return -np.log(2.0 * b2) - np.abs(x - mu2) / b2

        return self.numerical_quadrature(log_p, log_q, alpha)

    # -----------------------------------------------------------------
    # Numerical quadrature for arbitrary densities
    # -----------------------------------------------------------------

    def numerical_quadrature(
        self,
        log_p: Callable[[FloatArray], FloatArray],
        log_q: Callable[[FloatArray], FloatArray],
        alpha: float,
        lower: float = -50.0,
        upper: float = 50.0,
    ) -> RenyiDivergenceResult:
        """Rényi divergence via numerical quadrature for continuous densities.

        Computes D_α(P || Q) using the trapezoidal rule on a fine grid.
        Both P and Q are specified via their log-density functions.

        D_α(P || Q) = 1/(α-1) log ∫ p(x)^α q(x)^(1-α) dx

        Uses log-domain arithmetic throughout for stability.

        Args:
            log_p: Function mapping points x → log p(x).
            log_q: Function mapping points x → log q(x).
            alpha: Rényi divergence order (> 0, ≠ 1).
            lower: Lower integration bound.
            upper: Upper integration bound.

        Returns:
            RenyiDivergenceResult with approximate divergence.
        """
        _validate_alpha(alpha)

        if abs(alpha - 1.0) < _ALPHA_KL_THRESHOLD:
            return self._kl_quadrature(log_p, log_q, lower, upper)

        x = np.linspace(lower, upper, self._quadrature_points)
        dx = (upper - lower) / (self._quadrature_points - 1)

        lp = log_p(x)
        lq = log_q(x)

        # log(p^α q^(1-α)) = α log p + (1-α) log q
        log_integrand = alpha * lp + (1.0 - alpha) * lq

        # Clamp for stability
        log_integrand = np.clip(log_integrand, _LOG_DOMAIN_MIN, _LOG_DOMAIN_MAX)

        # log-trapezoidal rule: log ∫ f dx ≈ log(dx) + logsumexp(log_f)
        # with trapezoidal weights (halve endpoints)
        log_weights = np.full_like(log_integrand, math.log(dx))
        log_weights[0] += math.log(0.5)
        log_weights[-1] += math.log(0.5)

        log_integral = _logsumexp(log_integrand + log_weights)
        divergence = log_integral / (alpha - 1.0)

        return RenyiDivergenceResult(
            alpha=alpha,
            divergence=max(divergence, 0.0),
            method="numerical_quadrature",
            log_divergence=log_integral,
            metadata={"n_points": self._quadrature_points, "bounds": (lower, upper)},
        )

    def _kl_quadrature(
        self,
        log_p: Callable[[FloatArray], FloatArray],
        log_q: Callable[[FloatArray], FloatArray],
        lower: float,
        upper: float,
    ) -> RenyiDivergenceResult:
        """KL divergence via numerical quadrature (α → 1 limit)."""
        x = np.linspace(lower, upper, self._quadrature_points)
        dx = (upper - lower) / (self._quadrature_points - 1)

        lp = log_p(x)
        lq = log_q(x)

        # KL = ∫ p(x) (log p(x) - log q(x)) dx
        p_vals = np.exp(lp)
        integrand = p_vals * (lp - lq)

        # Trapezoidal rule
        kl = float(np.trapz(integrand, dx=dx))

        return RenyiDivergenceResult(
            alpha=1.0,
            divergence=max(kl, 0.0),
            method="kl_quadrature",
            metadata={"n_points": self._quadrature_points},
        )

    # -----------------------------------------------------------------
    # Special-case implementations
    # -----------------------------------------------------------------

    def _kl_divergence_discrete(
        self,
        p: FloatArray,
        q: FloatArray,
    ) -> RenyiDivergenceResult:
        """KL divergence (α → 1 limit) for discrete distributions."""
        mask = p > self._min_prob
        if not np.any(mask):
            return RenyiDivergenceResult(
                alpha=1.0, divergence=0.0, method="kl_discrete",
            )

        q_safe = np.maximum(q, self._min_prob)
        kl = float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q_safe[mask]))))

        return RenyiDivergenceResult(
            alpha=1.0,
            divergence=max(kl, 0.0),
            method="kl_discrete",
        )

    def _max_divergence_discrete(
        self,
        p: FloatArray,
        q: FloatArray,
    ) -> RenyiDivergenceResult:
        """Max divergence (α → ∞) for discrete distributions."""
        mask = p > self._min_prob
        if not np.any(mask):
            return RenyiDivergenceResult(
                alpha=float("inf"), divergence=0.0, method="max_discrete",
            )

        q_safe = np.maximum(q[mask], self._min_prob)
        log_ratios = np.log(p[mask]) - np.log(q_safe)
        max_div = float(np.max(log_ratios))

        return RenyiDivergenceResult(
            alpha=float("inf"),
            divergence=max(max_div, 0.0),
            method="max_discrete",
        )

    def _zero_order_discrete(
        self,
        p: FloatArray,
        q: FloatArray,
    ) -> RenyiDivergenceResult:
        """Zero-order Rényi divergence: -log Q(support(P))."""
        support_p = p > self._min_prob
        q_mass = float(np.sum(q[support_p]))

        if q_mass <= 0:
            divergence = float("inf")
        else:
            divergence = -math.log(q_mass)

        return RenyiDivergenceResult(
            alpha=0.0,
            divergence=max(divergence, 0.0),
            method="zero_order_discrete",
        )

    def _kl_gaussian(
        self,
        mu1: float,
        sigma1: float,
        mu2: float,
        sigma2: float,
    ) -> RenyiDivergenceResult:
        """KL divergence for Gaussians."""
        var1 = sigma1 ** 2
        var2 = sigma2 ** 2
        kl = (
            math.log(sigma2 / sigma1)
            + (var1 + (mu1 - mu2) ** 2) / (2.0 * var2)
            - 0.5
        )
        return RenyiDivergenceResult(
            alpha=1.0,
            divergence=max(kl, 0.0),
            method="kl_gaussian",
        )

    def _max_divergence_gaussian(
        self,
        mu1: float,
        sigma1: float,
        mu2: float,
        sigma2: float,
    ) -> RenyiDivergenceResult:
        """Max divergence (α → ∞) for Gaussians.

        For σ₁ < σ₂: finite. For σ₁ ≥ σ₂ with different means: +∞.
        """
        if sigma1 >= sigma2 and (mu1 != mu2 or sigma1 > sigma2):
            return RenyiDivergenceResult(
                alpha=float("inf"), divergence=float("inf"),
                method="max_gaussian",
            )
        # σ₁ < σ₂: max is achieved at the boundary and equals
        # (μ₁-μ₂)²/(2(σ₂²-σ₁²)) + log(σ₂/σ₁) - 0.5 + σ₁²/(2σ₂²)
        # This is the supremum of the log-likelihood ratio
        var1 = sigma1 ** 2
        var2 = sigma2 ** 2
        var_diff = var2 - var1
        div = (mu1 - mu2) ** 2 / (2.0 * var_diff) + 0.5 * math.log(var2 / var1) + var1 / (2.0 * var2) - 0.5

        return RenyiDivergenceResult(
            alpha=float("inf"),
            divergence=max(div, 0.0),
            method="max_gaussian",
        )

    def _max_divergence_laplace(
        self,
        mu1: float,
        b1: float,
        mu2: float,
        b2: float,
    ) -> RenyiDivergenceResult:
        """Max divergence for Laplace distributions."""
        if b1 >= b2:
            # sup log(p/q) = +∞ when tails of p are heavier
            if b1 > b2 or mu1 != mu2:
                return RenyiDivergenceResult(
                    alpha=float("inf"), divergence=float("inf"),
                    method="max_laplace",
                )
            return RenyiDivergenceResult(
                alpha=float("inf"), divergence=0.0, method="max_laplace",
            )

        # b1 < b2: max at x = mu1
        div = math.log(b2 / b1) + abs(mu1 - mu2) * (1.0 / b2 - 1.0 / b1)
        return RenyiDivergenceResult(
            alpha=float("inf"),
            divergence=max(div, 0.0),
            method="max_laplace",
        )

    def _kl_laplace(
        self,
        mu1: float,
        b1: float,
        mu2: float,
        b2: float,
    ) -> RenyiDivergenceResult:
        """KL divergence for Laplace distributions.

        KL(Lap(μ₁,b₁) || Lap(μ₂,b₂)) =
            log(b₂/b₁) + |μ₁-μ₂|/b₂ + b₁/b₂ · exp(-|μ₁-μ₂|/b₁) - 1
        """
        d = abs(mu1 - mu2)
        kl = math.log(b2 / b1) + d / b2 + (b1 / b2) * math.exp(-d / b1) - 1.0
        return RenyiDivergenceResult(
            alpha=1.0,
            divergence=max(kl, 0.0),
            method="kl_laplace",
        )

    # -----------------------------------------------------------------
    # Batch computation utilities
    # -----------------------------------------------------------------

    def compute_curve(
        self,
        p: FloatArray,
        q: FloatArray,
        alphas: FloatArray,
    ) -> Tuple[FloatArray, FloatArray]:
        """Compute Rényi divergence curve D_α(P||Q) for multiple α values.

        Args:
            p: Probability mass function P.
            q: Probability mass function Q.
            alphas: Array of α orders.

        Returns:
            Tuple of (alphas, divergences) arrays.
        """
        divergences = self.exact_discrete_vectorized(p, q, alphas)
        return np.asarray(alphas, dtype=np.float64), divergences

    def symmetrized(
        self,
        p: FloatArray,
        q: FloatArray,
        alpha: float,
    ) -> float:
        """Symmetrized Rényi divergence: (D_α(P||Q) + D_α(Q||P)) / 2.

        Args:
            p: Distribution P.
            q: Distribution Q.
            alpha: Rényi order.

        Returns:
            Symmetrized divergence value.
        """
        d_pq = self.exact_discrete(p, q, alpha).divergence
        d_qp = self.exact_discrete(q, p, alpha).divergence
        return (d_pq + d_qp) / 2.0

    def __repr__(self) -> str:
        return (
            f"RenyiDivergenceComputer(min_prob={self._min_prob:.0e}, "
            f"quadrature_points={self._quadrature_points})"
        )

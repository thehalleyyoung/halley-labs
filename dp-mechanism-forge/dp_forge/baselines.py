"""
Baseline DP mechanism implementations for DP-Forge.

Reference implementations of all standard differentially private noise
mechanisms, used for comparison against CEGIS-synthesised mechanisms.
Each mechanism provides:

- **Sampling**: Draw noisy answers for a given true value.
- **Density/PMF**: Evaluate the noise distribution analytically.
- **Error metrics**: Closed-form MSE, MAE, and variance.
- **Discretisation**: Convert continuous mechanisms into discrete probability
  tables over a finite grid, for direct LP-level comparison.

Mechanisms implemented:

1. :class:`LaplaceMechanism` — The workhorse ε-DP mechanism.
2. :class:`GaussianMechanism` — (ε, δ)-DP via calibrated Gaussian noise.
3. :class:`GeometricMechanism` — Integer-valued ε-DP mechanism.
4. :class:`StaircaseMechanism` — Optimal L1 mechanism for counting
   (Geng & Viswanath 2014).
5. :class:`MatrixMechanism` — Workload-aware Gaussian mechanisms.
6. :class:`ExponentialMechanism` — For non-numeric score-based selection.
7. :class:`RandResponseMechanism` — Randomized response for binary data.
8. :class:`BaselineComparator` — Compare synthesised vs. baseline mechanisms.

All classes share a common interface where applicable:
``sample()``, ``mse()``, ``mae()``, ``variance()``, ``discretize()``.
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
from scipy import special, stats, sparse

from dp_forge.exceptions import (
    ConfigurationError,
    InvalidMechanismError,
)
from dp_forge.types import (
    LossFunction,
    QuerySpec,
    WorkloadSpec,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]


# =========================================================================
# Abstract base
# =========================================================================


class _BaseMechanism:
    """Internal base class for all baseline mechanisms."""

    def __init__(
        self,
        epsilon: float,
        sensitivity: float,
        delta: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ConfigurationError(
                f"epsilon must be > 0, got {epsilon}",
                parameter="epsilon",
                value=epsilon,
                constraint="epsilon > 0",
            )
        if sensitivity <= 0:
            raise ConfigurationError(
                f"sensitivity must be > 0, got {sensitivity}",
                parameter="sensitivity",
                value=sensitivity,
                constraint="sensitivity > 0",
            )
        if not (0 <= delta < 1):
            raise ConfigurationError(
                f"delta must be in [0, 1), got {delta}",
                parameter="delta",
                value=delta,
                constraint="0 <= delta < 1",
            )
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.delta = delta
        self._rng = np.random.default_rng(seed)

    def _validate_n_samples(self, n_samples: int) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")


# =========================================================================
# 1. Laplace Mechanism
# =========================================================================


class LaplaceMechanism(_BaseMechanism):
    """The Laplace mechanism for ε-differential privacy.

    Adds noise drawn from ``Laplace(0, b)`` where ``b = sensitivity / epsilon``.
    This is the canonical pure-DP mechanism for real-valued queries.

    Attributes:
        epsilon: Privacy parameter ε > 0.
        sensitivity: Global L1 sensitivity Δ.
        delta: Must be 0 for pure DP (default).
        scale: Laplace scale ``b = Δ / ε``.

    Example::

        >>> mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        >>> mech.scale
        1.0
        >>> mech.mse()
        2.0
        >>> mech.mae()
        1.0
    """

    def __init__(
        self,
        epsilon: float,
        sensitivity: float,
        delta: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(epsilon, sensitivity, delta, seed)
        self.scale = sensitivity / epsilon

    def sample(
        self,
        true_value: Union[float, FloatArray],
        n_samples: int = 1,
    ) -> FloatArray:
        """Draw noisy samples from the Laplace mechanism.

        Args:
            true_value: True query answer(s).  If an array, noise is
                added independently to each element.
            n_samples: Number of independent samples per true value.

        Returns:
            Array of noisy values.  Shape is ``(n_samples,)`` for scalar
            input, or ``(n_samples, len(true_value))`` for array input.
        """
        self._validate_n_samples(n_samples)
        tv = np.asarray(true_value, dtype=np.float64)
        noise = self._rng.laplace(0, self.scale, size=(n_samples,) + tv.shape)
        result = tv + noise
        return result.squeeze() if n_samples == 1 and tv.ndim == 0 else result

    def pdf(self, noise: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """Evaluate the Laplace probability density at *noise*.

        ``f(x) = (1 / 2b) * exp(-|x| / b)``

        Args:
            noise: Noise value(s) (deviation from true value).

        Returns:
            Density value(s).
        """
        noise = np.asarray(noise, dtype=np.float64)
        result = (1.0 / (2.0 * self.scale)) * np.exp(-np.abs(noise) / self.scale)
        return float(result) if result.ndim == 0 else result

    def cdf(self, noise: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """Evaluate the Laplace CDF at *noise*.

        ``F(x) = 0.5 * exp(x/b) if x < 0, else 1 - 0.5 * exp(-x/b)``

        Args:
            noise: Noise value(s).

        Returns:
            CDF value(s).
        """
        noise = np.asarray(noise, dtype=np.float64)
        result = np.where(
            noise < 0,
            0.5 * np.exp(noise / self.scale),
            1.0 - 0.5 * np.exp(-noise / self.scale),
        )
        return float(result) if result.ndim == 0 else result

    def log_pdf(self, noise: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """Log-density of the Laplace distribution.

        ``log f(x) = -log(2b) - |x|/b``

        Args:
            noise: Noise value(s).

        Returns:
            Log-density value(s).
        """
        noise = np.asarray(noise, dtype=np.float64)
        result = -math.log(2.0 * self.scale) - np.abs(noise) / self.scale
        return float(result) if result.ndim == 0 else result

    def quantile(self, p: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """Inverse CDF (quantile function) of the Laplace distribution.

        Args:
            p: Probability value(s) in (0, 1).

        Returns:
            Quantile value(s).
        """
        p = np.asarray(p, dtype=np.float64)
        result = np.where(
            p < 0.5,
            self.scale * np.log(2.0 * np.maximum(p, 1e-300)),
            -self.scale * np.log(2.0 * np.maximum(1.0 - p, 1e-300)),
        )
        return float(result) if result.ndim == 0 else result

    def mse(self) -> float:
        """Analytical mean squared error (variance for zero-mean noise).

        ``MSE = Var = 2b² = 2(Δ/ε)²``

        Returns:
            MSE value.
        """
        return 2.0 * self.scale ** 2

    def mae(self) -> float:
        """Analytical mean absolute error.

        ``MAE = E[|X|] = b = Δ/ε``

        Returns:
            MAE value.
        """
        return self.scale

    def variance(self) -> float:
        """Variance of the Laplace noise.

        ``Var = 2b²``

        Returns:
            Variance.
        """
        return 2.0 * self.scale ** 2

    def entropy_nats(self) -> float:
        """Differential entropy in nats.

        ``H = 1 + log(2b)``

        Returns:
            Entropy in nats.
        """
        return 1.0 + math.log(2.0 * self.scale)

    def privacy_loss_at(self, x: float) -> float:
        """Privacy loss at observation *x* for adjacent databases differing by Δ.

        ``L(x) = |x| / b - |x - Δ| / b``

        This is the worst-case direction.

        Args:
            x: Observed noisy output.

        Returns:
            Privacy loss value (should be ≤ ε).
        """
        return (abs(x) - abs(x - self.sensitivity)) / self.scale

    def discretize(
        self,
        k: int,
        y_grid: Optional[FloatArray] = None,
        center: float = 0.0,
        padding: float = 5.0,
    ) -> Tuple[FloatArray, FloatArray]:
        """Create a discrete probability table over a finite grid.

        Assigns probability mass to each bin by integrating the Laplace
        PDF over the bin width.  Edge bins absorb the tails.

        Args:
            k: Number of output bins.
            y_grid: Output grid values.  If ``None``, a uniform grid
                centered at *center* is created.
            center: Center of the auto-generated grid.
            padding: Half-width of the grid in units of scale.

        Returns:
            ``(y_grid, probs)`` where ``probs`` sums to 1.
        """
        if y_grid is None:
            half_width = padding * self.scale
            y_grid = np.linspace(center - half_width, center + half_width, k)
        y_grid = np.asarray(y_grid, dtype=np.float64)

        # Compute bin edges as midpoints between grid values
        probs = np.zeros(k, dtype=np.float64)
        if k == 1:
            probs[0] = 1.0
            return y_grid, probs

        edges = np.empty(k + 1, dtype=np.float64)
        edges[0] = -np.inf
        edges[-1] = np.inf
        for i in range(1, k):
            edges[i] = 0.5 * (y_grid[i - 1] + y_grid[i])

        for i in range(k):
            cdf_hi = float(self.cdf(edges[i + 1] - center))
            cdf_lo = float(self.cdf(edges[i] - center))
            probs[i] = cdf_hi - cdf_lo

        # Normalise to handle numerical error
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total > 0:
            probs /= total

        return y_grid, probs

    def __repr__(self) -> str:
        return (
            f"LaplaceMechanism(ε={self.epsilon}, Δ={self.sensitivity}, "
            f"b={self.scale:.6f})"
        )


# =========================================================================
# 2. Gaussian Mechanism
# =========================================================================


class GaussianMechanism(_BaseMechanism):
    """The Gaussian mechanism for (ε, δ)-differential privacy.

    Adds noise drawn from ``N(0, σ²)`` where σ is calibrated to satisfy
    (ε, δ)-DP.

    The standard calibration (analytic Gaussian mechanism):
        ``σ = Δ · √(2 ln(1.25/δ)) / ε``

    Attributes:
        epsilon: Privacy parameter ε > 0.
        delta: Privacy parameter δ > 0.
        sensitivity: Global L2 sensitivity Δ.
        sigma: Calibrated noise standard deviation.

    Example::

        >>> mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        >>> mech.sigma  # doctest: +ELLIPSIS
        3.730...
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float,
        sigma: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        if delta <= 0:
            raise ConfigurationError(
                f"Gaussian mechanism requires delta > 0, got {delta}",
                parameter="delta",
                value=delta,
                constraint="delta > 0",
            )
        super().__init__(epsilon, sensitivity, delta, seed)

        if sigma is not None:
            self.sigma = sigma
        else:
            self.sigma = self.calibrate_sigma()

    def calibrate_sigma(self) -> float:
        """Calibrate σ for (ε, δ)-DP.

        Uses the standard formula:
        ``σ = Δ · √(2 · ln(1.25 / δ)) / ε``

        Returns:
            Calibrated σ.
        """
        return self.sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon

    @staticmethod
    def calibrate_sigma_analytic(
        epsilon: float,
        delta: float,
        sensitivity: float,
    ) -> float:
        """Static version of sigma calibration.

        Uses the tighter analytic Gaussian mechanism bound from
        Balle & Wang (2018) for small ε:

        ``σ = Δ / ε · √(2 · ln(1.25/δ))``

        For ε ≥ 1, uses the standard calibration.  For ε < 1, uses
        a slightly tighter bound via numerical inversion.

        Args:
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ > 0.
            sensitivity: L2 sensitivity Δ.

        Returns:
            Calibrated σ.
        """
        # Standard calibration
        sigma_standard = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

        # For small epsilon, try the analytic bound
        if epsilon < 1.0:
            # Analytic Gaussian mechanism (Balle & Wang 2018)
            # σ such that Φ(Δ/(2σ) - ε·σ/Δ) - exp(ε)·Φ(-Δ/(2σ) - ε·σ/Δ) ≤ δ
            # Binary search for tighter σ
            lo, hi = 0.01 * sigma_standard, sigma_standard
            for _ in range(100):
                mid = (lo + hi) / 2.0
                # Check the analytic bound
                t1 = sensitivity / (2.0 * mid) - epsilon * mid / sensitivity
                t2 = -sensitivity / (2.0 * mid) - epsilon * mid / sensitivity
                lhs = stats.norm.cdf(t1) - math.exp(epsilon) * stats.norm.cdf(t2)
                if lhs <= delta:
                    hi = mid
                else:
                    lo = mid
            return hi

        return sigma_standard

    def sample(
        self,
        true_value: Union[float, FloatArray],
        n_samples: int = 1,
    ) -> FloatArray:
        """Draw noisy samples from the Gaussian mechanism.

        Args:
            true_value: True query answer(s).
            n_samples: Number of independent samples.

        Returns:
            Array of noisy values.
        """
        self._validate_n_samples(n_samples)
        tv = np.asarray(true_value, dtype=np.float64)
        noise = self._rng.normal(0, self.sigma, size=(n_samples,) + tv.shape)
        result = tv + noise
        return result.squeeze() if n_samples == 1 and tv.ndim == 0 else result

    def pdf(self, noise: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """Gaussian noise density.

        Args:
            noise: Noise value(s).

        Returns:
            Density value(s).
        """
        noise = np.asarray(noise, dtype=np.float64)
        result = stats.norm.pdf(noise, 0, self.sigma)
        return float(result) if result.ndim == 0 else result

    def cdf(self, noise: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """Gaussian noise CDF.

        Args:
            noise: Noise value(s).

        Returns:
            CDF value(s).
        """
        noise = np.asarray(noise, dtype=np.float64)
        result = stats.norm.cdf(noise, 0, self.sigma)
        return float(result) if result.ndim == 0 else result

    def mse(self) -> float:
        """Mean squared error (equals variance for zero-mean Gaussian).

        ``MSE = σ²``

        Returns:
            MSE value.
        """
        return self.sigma ** 2

    def mae(self) -> float:
        """Mean absolute error.

        ``MAE = σ · √(2/π)``

        Returns:
            MAE value.
        """
        return self.sigma * math.sqrt(2.0 / math.pi)

    def variance(self) -> float:
        """Variance of the Gaussian noise.

        Returns:
            ``σ²``.
        """
        return self.sigma ** 2

    def entropy_nats(self) -> float:
        """Differential entropy in nats.

        ``H = 0.5 · ln(2πeσ²)``

        Returns:
            Entropy in nats.
        """
        return 0.5 * math.log(2.0 * math.pi * math.e * self.sigma ** 2)

    def discretize(
        self,
        k: int,
        y_grid: Optional[FloatArray] = None,
        center: float = 0.0,
        padding: float = 5.0,
    ) -> Tuple[FloatArray, FloatArray]:
        """Create a discrete probability table over a finite grid.

        Args:
            k: Number of output bins.
            y_grid: Output grid values.
            center: Center of the auto-generated grid.
            padding: Half-width in units of σ.

        Returns:
            ``(y_grid, probs)`` where ``probs`` sums to 1.
        """
        if y_grid is None:
            half_width = padding * self.sigma
            y_grid = np.linspace(center - half_width, center + half_width, k)
        y_grid = np.asarray(y_grid, dtype=np.float64)

        probs = np.zeros(k, dtype=np.float64)
        if k == 1:
            probs[0] = 1.0
            return y_grid, probs

        edges = np.empty(k + 1, dtype=np.float64)
        edges[0] = -np.inf
        edges[-1] = np.inf
        for i in range(1, k):
            edges[i] = 0.5 * (y_grid[i - 1] + y_grid[i])

        for i in range(k):
            cdf_hi = float(self.cdf(edges[i + 1] - center))
            cdf_lo = float(self.cdf(edges[i] - center))
            probs[i] = cdf_hi - cdf_lo

        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total > 0:
            probs /= total

        return y_grid, probs

    def __repr__(self) -> str:
        return (
            f"GaussianMechanism(ε={self.epsilon}, δ={self.delta}, "
            f"Δ={self.sensitivity}, σ={self.sigma:.6f})"
        )


# =========================================================================
# 3. Geometric Mechanism
# =========================================================================


class GeometricMechanism(_BaseMechanism):
    """The two-sided geometric mechanism for integer-valued queries.

    For ε-DP over integer-valued queries with sensitivity 1, the noise
    follows a two-sided geometric distribution:

    ``Pr[Z = z] = ((1 - p) / (1 + p)) · p^|z|``   where ``p = exp(-ε)``.

    This is the discrete analogue of the Laplace mechanism and is
    optimal for counting queries over unbounded integer domains.

    Attributes:
        epsilon: Privacy parameter ε.
        sensitivity: Integer sensitivity (must be a positive integer).
        p: Geometric parameter ``exp(-ε/Δ)``.

    Example::

        >>> mech = GeometricMechanism(epsilon=1.0, sensitivity=1)
        >>> mech.p  # doctest: +ELLIPSIS
        0.367...
        >>> mech.mse()  # doctest: +ELLIPSIS
        0.850...
    """

    def __init__(
        self,
        epsilon: float,
        sensitivity: int = 1,
        delta: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if not isinstance(sensitivity, (int, np.integer)):
            raise ConfigurationError(
                f"GeometricMechanism requires integer sensitivity, got {type(sensitivity).__name__}",
                parameter="sensitivity",
                value=sensitivity,
                constraint="sensitivity must be a positive integer",
            )
        super().__init__(epsilon, float(sensitivity), delta, seed)
        self.int_sensitivity = int(sensitivity)
        self.p = math.exp(-epsilon / self.int_sensitivity)

    def sample(
        self,
        true_value: Union[int, float, FloatArray],
        n_samples: int = 1,
    ) -> FloatArray:
        """Draw noisy integer samples.

        The two-sided geometric is sampled as the difference of two
        one-sided geometric random variables.

        Args:
            true_value: True integer query answer(s).
            n_samples: Number of independent samples.

        Returns:
            Array of noisy integer values.
        """
        self._validate_n_samples(n_samples)
        tv = np.asarray(true_value, dtype=np.float64)

        # Sample two-sided geometric as difference of two geometric RVs
        # Geometric(1 - p) counts number of failures before first success
        geom_p = 1.0 - self.p
        shape = (n_samples,) + tv.shape
        pos = self._rng.geometric(geom_p, size=shape) - 1
        neg = self._rng.geometric(geom_p, size=shape) - 1
        noise = pos - neg

        result = tv + noise.astype(np.float64)
        return result.squeeze() if n_samples == 1 and tv.ndim == 0 else result

    def pmf(self, z: Union[int, float, FloatArray]) -> Union[float, FloatArray]:
        """Probability mass function of the two-sided geometric.

        ``Pr[Z = z] = ((1 - p) / (1 + p)) · p^|z|``

        Args:
            z: Integer noise value(s).

        Returns:
            PMF value(s).
        """
        z = np.asarray(z, dtype=np.float64)
        normalizer = (1.0 - self.p) / (1.0 + self.p)
        result = normalizer * self.p ** np.abs(z)
        return float(result) if result.ndim == 0 else result

    def cdf(self, z: Union[int, float, FloatArray]) -> Union[float, FloatArray]:
        """CDF of the two-sided geometric distribution.

        Args:
            z: Value(s) at which to evaluate the CDF.

        Returns:
            CDF value(s).
        """
        z = np.asarray(z, dtype=np.float64)
        z_floor = np.floor(z)

        # For z_floor >= 0: CDF = 1 - p^(z_floor+1) / (1+p)
        # For z_floor < 0: CDF = p^(-z_floor) / (1+p)
        result = np.where(
            z_floor >= 0,
            1.0 - self.p ** (z_floor + 1) / (1.0 + self.p),
            self.p ** (-z_floor) / (1.0 + self.p),
        )
        return float(result) if result.ndim == 0 else result

    def mse(self) -> float:
        """Mean squared error (variance for zero-mean distribution).

        ``MSE = 2p / (1 - p)²``

        Returns:
            MSE value.
        """
        return 2.0 * self.p / (1.0 - self.p) ** 2

    def mae(self) -> float:
        """Mean absolute error.

        ``MAE = p / (1 - p)``

        Returns:
            MAE value.
        """
        return self.p / (1.0 - self.p)

    def variance(self) -> float:
        """Variance of the two-sided geometric noise.

        ``Var = 2p / (1 - p)²``

        Returns:
            Variance.
        """
        return 2.0 * self.p / (1.0 - self.p) ** 2

    def median_abs_deviation(self) -> float:
        """Median absolute deviation of the two-sided geometric.

        Since the distribution is symmetric around 0, the median is 0
        and the MAD is the median of |Z|.

        Returns:
            MAD value.
        """
        if self.p <= 0.5:
            return 0.0
        # Median of |Z| ~ Geometric(1-p): ceil(log(0.5) / log(p)) - 1
        return math.ceil(math.log(0.5) / math.log(self.p)) - 1

    def discretize(
        self,
        k: int,
        y_grid: Optional[FloatArray] = None,
        center: float = 0.0,
        max_range: int = 50,
    ) -> Tuple[FloatArray, FloatArray]:
        """Create a discrete probability table.

        For the geometric mechanism, the grid is naturally integer-valued.

        Args:
            k: Number of output values.
            y_grid: Explicit grid (if None, uses integers around center).
            center: Center of the grid.
            max_range: Maximum distance from center for auto grid.

        Returns:
            ``(y_grid, probs)`` tuple.
        """
        if y_grid is None:
            half = min(k // 2, max_range)
            y_grid = np.arange(
                int(center) - half,
                int(center) + half + 1,
                dtype=np.float64,
            )[:k]

        y_grid = np.asarray(y_grid, dtype=np.float64)
        probs = np.array([float(self.pmf(y - center)) for y in y_grid])

        total = probs.sum()
        if total > 0:
            probs /= total

        return y_grid, probs

    def __repr__(self) -> str:
        return (
            f"GeometricMechanism(ε={self.epsilon}, Δ={self.int_sensitivity}, "
            f"p={self.p:.6f})"
        )


# =========================================================================
# 4. Staircase Mechanism (Geng-Viswanath 2014)
# =========================================================================


class StaircaseMechanism(_BaseMechanism):
    """The staircase mechanism — optimal for L1 loss on counting queries.

    From Geng & Viswanath, "The Optimal Mechanism in Differential Privacy"
    (2014).  The staircase mechanism has a piecewise-constant PDF that
    alternates between two levels, achieving the optimal expected L1 error
    among all ε-DP mechanisms for sensitivity-1 queries.

    The PDF is parameterised by ``γ ∈ [0, 1]``:

    For ``x ∈ [kΔ, (k+1)Δ)``, k integer:
        - If k is even: ``f(x) = a · exp(-ε · |k| / 2)``
        - If k is odd:  ``f(x) = a · γ · exp(-ε · (|k|+1) / 2)``

    where ``a`` is a normalisation constant and ``γ`` is chosen to minimise
    the expected absolute error.

    Attributes:
        epsilon: Privacy parameter ε.
        sensitivity: Query sensitivity Δ.
        gamma: Shape parameter γ.

    Example::

        >>> mech = StaircaseMechanism(epsilon=1.0, sensitivity=1.0)
        >>> mech.mae() < LaplaceMechanism(1.0, 1.0).mae()
        True
    """

    def __init__(
        self,
        epsilon: float,
        sensitivity: float = 1.0,
        gamma: Optional[float] = None,
        delta: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(epsilon, sensitivity, delta, seed)
        if gamma is not None:
            if not (0 <= gamma <= 1):
                raise ConfigurationError(
                    f"gamma must be in [0, 1], got {gamma}",
                    parameter="gamma",
                    value=gamma,
                    constraint="0 <= gamma <= 1",
                )
            self.gamma = gamma
        else:
            self.gamma = self._compute_optimal_gamma()

    def _compute_optimal_gamma(self) -> float:
        """Compute the optimal γ for minimising E[|Z|].

        From Geng & Viswanath (2014), Theorem 1:
        ``γ* = 1 / (1 + exp(ε/2))`` when ε ≤ ln(2), or found numerically.

        Returns:
            Optimal γ.
        """
        eps = self.epsilon
        # From the paper: for counting queries (Δ=1),
        # γ* = (1 - exp(-ε/2)) / (1 - exp(-ε/2) + ε/2 * exp(-ε/2))
        # But the exact optimal depends on the sensitivity.
        # For general Δ, search over [0, 1].

        best_gamma = 0.0
        best_mae = float("inf")

        for g in np.linspace(0, 1, 1000):
            mae_val = self._compute_mae_for_gamma(g)
            if mae_val < best_mae:
                best_mae = mae_val
                best_gamma = g

        # Refine with finer search
        lo = max(best_gamma - 0.002, 0.0)
        hi = min(best_gamma + 0.002, 1.0)
        for g in np.linspace(lo, hi, 1000):
            mae_val = self._compute_mae_for_gamma(g)
            if mae_val < best_mae:
                best_mae = mae_val
                best_gamma = g

        return best_gamma

    def _compute_mae_for_gamma(self, gamma: float) -> float:
        """Compute expected |Z| for a given γ.

        Integrates the staircase PDF times |x|.
        """
        eps = self.epsilon
        delta_s = self.sensitivity
        exp_eps = math.exp(eps)

        # Series sum for normalisation and MAE
        # sum over k = 0, ±1, ±2, ...
        # For even k >= 0: weight = exp(-eps * k/2)
        # For odd k >= 1: weight = gamma * exp(-eps * (k+1)/2)
        # Interval for each k: [k*delta_s, (k+1)*delta_s]

        norm = 0.0
        mae_num = 0.0
        max_k = max(100, int(20 / eps) + 10) if eps > 0 else 100

        for k in range(max_k):
            # Even k
            w_even = math.exp(-eps * k / 2.0)
            interval_len = delta_s
            norm += w_even * interval_len * (2 if k > 0 else 1)

            # Contribution to E[|x|] for x in [k*Δ, (k+1)*Δ]
            lo_x = k * delta_s
            hi_x = (k + 1) * delta_s
            # integral of |x| over [lo, hi] = (hi^2 - lo^2) / 2 for lo >= 0
            if k == 0:
                mae_num += w_even * (hi_x ** 2 - lo_x ** 2) / 2.0
            else:
                mae_num += w_even * (hi_x ** 2 - lo_x ** 2) / 2.0 * 2  # both sides

            # Odd k
            k_odd = k  # maps to odd index k*2+1 conceptually
            w_odd = gamma * math.exp(-eps * (k + 1) / 2.0)
            # Odd intervals: [(k+0.5)Δ roughly], depends on convention
            # Using the standard staircase: odd block at [(k)*Δ, (k+1)*Δ]
            # with k being the odd integer index
            lo_odd = (k + 0.5) * delta_s  # simplified model
            hi_odd = (k + 1.5) * delta_s

            # Actually, the staircase alternates per-Δ intervals:
            # [0, Δ) = even (k=0), [Δ, 2Δ) = odd (k=1), [2Δ, 3Δ) = even (k=2), etc.
            # Let's use the correct form:
            pass  # handled below

        # Use the proper formulation:
        # f(x) = a * p^(floor(|x|/Δ)) * gamma^(floor(|x|/Δ) mod 2 == 1)
        # where p = exp(-ε)
        p = math.exp(-eps)
        norm2 = 0.0
        mae2 = 0.0

        for k in range(max_k):
            lo_x = k * delta_s
            hi_x = (k + 1) * delta_s

            if k % 2 == 0:
                weight = p ** (k // 2)
            else:
                weight = gamma * p ** ((k + 1) // 2)

            block_len = delta_s
            norm2 += weight * block_len * (2 if k > 0 else 1)

            # integral of |x| from lo_x to hi_x = (hi^2 - lo^2)/2
            abs_integral = (hi_x ** 2 - lo_x ** 2) / 2.0
            factor = 2 if k > 0 else 1
            mae2 += weight * abs_integral * factor

            if weight * block_len < 1e-20:
                break

        if norm2 < 1e-300:
            return float("inf")
        return mae2 / norm2

    def _normalisation_constant(self) -> float:
        """Compute the normalisation constant ``a`` for the PDF."""
        eps = self.epsilon
        delta_s = self.sensitivity
        p = math.exp(-eps)
        max_k = max(100, int(20 / eps) + 10) if eps > 0 else 100

        total = 0.0
        for k in range(max_k):
            if k % 2 == 0:
                weight = p ** (k // 2)
            else:
                weight = self.gamma * p ** ((k + 1) // 2)

            factor = 2 if k > 0 else 1
            total += weight * delta_s * factor

            if weight * delta_s < 1e-20:
                break

        return 1.0 / total if total > 0 else 0.0

    def pdf(self, x: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """Evaluate the staircase PDF at *x*.

        Args:
            x: Noise value(s).

        Returns:
            Density value(s).
        """
        x = np.asarray(x, dtype=np.float64)
        a = self._normalisation_constant()
        eps = self.epsilon
        delta_s = self.sensitivity
        p = math.exp(-eps)

        abs_x = np.abs(x)
        k = np.floor(abs_x / delta_s).astype(int)

        weight = np.where(
            k % 2 == 0,
            p ** (k // 2),
            self.gamma * p ** ((k + 1) // 2),
        )
        result = a * weight
        return float(result) if result.ndim == 0 else result

    def cdf(self, x: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """Evaluate the staircase CDF at *x* (numerical integration).

        Args:
            x: Value(s).

        Returns:
            CDF value(s).
        """
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)

        results = np.empty_like(x)
        for idx, xi in enumerate(x):
            # Numerical integration via fine grid
            n_pts = 10000
            lo = -20 * self.sensitivity
            grid = np.linspace(lo, float(xi), n_pts)
            pdf_vals = self.pdf(grid)
            results[idx] = np.trapz(pdf_vals, grid)

        results = np.clip(results, 0.0, 1.0)
        return float(results[0]) if scalar else results

    def sample(
        self,
        true_value: Union[float, FloatArray],
        n_samples: int = 1,
    ) -> FloatArray:
        """Draw samples from the staircase mechanism.

        Uses inverse CDF sampling with a precomputed table.

        Args:
            true_value: True query answer(s).
            n_samples: Number of samples.

        Returns:
            Noisy values.
        """
        self._validate_n_samples(n_samples)
        tv = np.asarray(true_value, dtype=np.float64)

        eps = self.epsilon
        delta_s = self.sensitivity
        p = math.exp(-eps)

        shape = (n_samples,) + tv.shape
        flat_size = int(np.prod(shape))

        results = np.empty(flat_size, dtype=np.float64)
        for i in range(flat_size):
            # Sample block index k from the geometric-like distribution
            u1 = self._rng.random()
            u2 = self._rng.random()

            # Determine sign
            sign = 1 if u1 < 0.5 else -1
            if u1 < 0.5 and self._rng.random() < 0.5:
                sign = -1

            # Sample |Z| / Δ block index
            # Compute CDF over blocks
            cum_prob = 0.0
            u_block = self._rng.random()
            a = self._normalisation_constant()
            k_chosen = 0
            max_k = max(100, int(20 / eps) + 10) if eps > 0 else 100

            for k in range(max_k):
                if k % 2 == 0:
                    weight = p ** (k // 2)
                else:
                    weight = self.gamma * p ** ((k + 1) // 2)

                factor = 2.0 if k > 0 else 1.0
                block_prob = a * weight * delta_s * factor
                cum_prob += block_prob

                # Each side gets half the prob (except k=0)
                if cum_prob >= u_block:
                    k_chosen = k
                    break

            # Uniform within the block
            u_within = self._rng.random()
            noise = (k_chosen + u_within) * delta_s
            if self._rng.random() < 0.5 and k_chosen > 0:
                noise = -noise
            elif k_chosen == 0:
                noise = noise if self._rng.random() < 0.5 else -noise

            results[i] = noise

        results = results.reshape(shape) + tv
        return results.squeeze() if n_samples == 1 and tv.ndim == 0 else results

    def mse(self) -> float:
        """Mean squared error via numerical integration.

        Returns:
            MSE value.
        """
        eps = self.epsilon
        delta_s = self.sensitivity
        p = math.exp(-eps)
        a = self._normalisation_constant()
        max_k = max(100, int(20 / eps) + 10) if eps > 0 else 100

        mse_val = 0.0
        for k in range(max_k):
            if k % 2 == 0:
                weight = p ** (k // 2)
            else:
                weight = self.gamma * p ** ((k + 1) // 2)

            lo = k * delta_s
            hi = (k + 1) * delta_s
            # integral of x^2 from lo to hi = (hi^3 - lo^3)/3
            x2_integral = (hi ** 3 - lo ** 3) / 3.0
            factor = 2 if k > 0 else 1
            mse_val += a * weight * x2_integral * factor

            if a * weight * delta_s < 1e-20:
                break

        return mse_val

    def mae(self) -> float:
        """Mean absolute error via numerical integration.

        Returns:
            MAE value.
        """
        eps = self.epsilon
        delta_s = self.sensitivity
        p = math.exp(-eps)
        a = self._normalisation_constant()
        max_k = max(100, int(20 / eps) + 10) if eps > 0 else 100

        mae_val = 0.0
        for k in range(max_k):
            if k % 2 == 0:
                weight = p ** (k // 2)
            else:
                weight = self.gamma * p ** ((k + 1) // 2)

            lo = k * delta_s
            hi = (k + 1) * delta_s
            # integral of |x| from lo to hi = (hi^2 - lo^2)/2
            abs_integral = (hi ** 2 - lo ** 2) / 2.0
            factor = 2 if k > 0 else 1
            mae_val += a * weight * abs_integral * factor

            if a * weight * delta_s < 1e-20:
                break

        return mae_val

    def variance(self) -> float:
        """Variance (equals MSE since the mean is 0).

        Returns:
            Variance.
        """
        return self.mse()

    def discretize(
        self,
        k: int,
        y_grid: Optional[FloatArray] = None,
        center: float = 0.0,
        padding: float = 10.0,
    ) -> Tuple[FloatArray, FloatArray]:
        """Discretise the staircase mechanism onto a grid.

        Args:
            k: Number of bins.
            y_grid: Output grid.
            center: Center of auto grid.
            padding: Half-width in units of sensitivity.

        Returns:
            ``(y_grid, probs)`` tuple.
        """
        if y_grid is None:
            half = padding * self.sensitivity
            y_grid = np.linspace(center - half, center + half, k)
        y_grid = np.asarray(y_grid, dtype=np.float64)

        probs = np.zeros(k, dtype=np.float64)
        if k == 1:
            probs[0] = 1.0
            return y_grid, probs

        edges = np.empty(k + 1, dtype=np.float64)
        edges[0] = -np.inf
        edges[-1] = np.inf
        for i in range(1, k):
            edges[i] = 0.5 * (y_grid[i - 1] + y_grid[i])

        a = self._normalisation_constant()
        eps = self.epsilon
        delta_s = self.sensitivity
        p = math.exp(-eps)

        for i in range(k):
            lo_edge = edges[i]
            hi_edge = edges[i + 1]
            # Integrate PDF from lo_edge to hi_edge
            # Use fine numerical integration
            if np.isinf(lo_edge):
                lo_edge = center - 30 * delta_s
            if np.isinf(hi_edge):
                hi_edge = center + 30 * delta_s

            n_pts = max(100, int((hi_edge - lo_edge) / delta_s * 20))
            pts = np.linspace(lo_edge - center, hi_edge - center, n_pts)
            pdf_vals = self.pdf(pts)
            probs[i] = np.trapz(pdf_vals, pts)

        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total > 0:
            probs /= total

        return y_grid, probs

    def __repr__(self) -> str:
        return (
            f"StaircaseMechanism(ε={self.epsilon}, Δ={self.sensitivity}, "
            f"γ={self.gamma:.6f})"
        )


# =========================================================================
# 5. Matrix Mechanism
# =========================================================================


class MatrixMechanism:
    """Workload-aware matrix mechanism for linear queries.

    Given a workload matrix ``A`` of shape ``(m, d)`` and a strategy
    matrix ``B`` of shape ``(p, d)``, the mechanism:

    1. Answers the strategy queries ``Bx + noise`` with calibrated
       Gaussian noise.
    2. Reconstructs estimates of ``Ax`` via the pseudo-inverse
       ``A B^+ (Bx + noise)``.

    The total MSE depends on the choice of ``B``.  Several strategies
    are available:

    - ``'identity'``: ``B = I_d`` (no workload adaptation).
    - ``'total_variation'``: Heuristic based on A^T A.
    - ``'hdmm_greedy'``: Greedy strategy from HDMM (McKenna et al. 2018).
    - ``'optimal_prefix'``: Optimal for prefix workloads.

    Attributes:
        workload_A: Workload matrix ``(m, d)``.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        strategy: Strategy name.
        strategy_B: Strategy matrix ``(p, d)``.

    Example::

        >>> A = np.eye(5)
        >>> mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5)
        >>> mech.total_mse()  # doctest: +SKIP
    """

    AVAILABLE_STRATEGIES = ("identity", "total_variation", "hdmm_greedy", "optimal_prefix")

    def __init__(
        self,
        workload_A: FloatArray,
        epsilon: float,
        delta: float,
        strategy: str = "identity",
        sensitivity: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ConfigurationError(
                f"epsilon must be > 0, got {epsilon}",
                parameter="epsilon", value=epsilon, constraint="epsilon > 0",
            )
        if delta <= 0:
            raise ConfigurationError(
                f"delta must be > 0 for MatrixMechanism, got {delta}",
                parameter="delta", value=delta, constraint="delta > 0",
            )
        if strategy not in self.AVAILABLE_STRATEGIES:
            raise ConfigurationError(
                f"Unknown strategy {strategy!r}. "
                f"Available: {self.AVAILABLE_STRATEGIES}",
                parameter="strategy",
                value=strategy,
                constraint=f"one of {self.AVAILABLE_STRATEGIES}",
            )

        self.workload_A = np.asarray(workload_A, dtype=np.float64)
        if self.workload_A.ndim != 2:
            raise ValueError(f"workload_A must be 2-D, got shape {self.workload_A.shape}")

        self.epsilon = epsilon
        self.delta = delta
        self.strategy_name = strategy
        self.sensitivity = sensitivity
        self._rng = np.random.default_rng(seed)

        self.m, self.d = self.workload_A.shape
        self.strategy_B = self.synthesize_strategy()
        self.sigma = self._calibrate_sigma()

    def synthesize_strategy(self) -> FloatArray:
        """Compute the strategy matrix B based on the selected strategy.

        Returns:
            Strategy matrix of shape ``(p, d)``.
        """
        if self.strategy_name == "identity":
            return np.eye(self.d, dtype=np.float64)

        elif self.strategy_name == "total_variation":
            return self._total_variation_strategy()

        elif self.strategy_name == "hdmm_greedy":
            return self._hdmm_greedy_strategy()

        elif self.strategy_name == "optimal_prefix":
            return self._optimal_prefix_strategy()

        return np.eye(self.d, dtype=np.float64)

    def _total_variation_strategy(self) -> FloatArray:
        """Heuristic strategy based on A^T A eigenvectors.

        Uses the top eigenvectors of ``A^T A`` as strategy queries.
        """
        ATA = self.workload_A.T @ self.workload_A
        eigenvalues, eigenvectors = np.linalg.eigh(ATA)
        # Keep eigenvectors with significant eigenvalues
        threshold = np.max(eigenvalues) * 1e-10
        mask = eigenvalues > threshold
        B = eigenvectors[:, mask].T
        if B.shape[0] == 0:
            return np.eye(self.d, dtype=np.float64)
        return B

    def _hdmm_greedy_strategy(self) -> FloatArray:
        """Simplified HDMM greedy strategy.

        Greedily selects strategy queries that maximise marginal
        MSE reduction.  This is a simplified version of the full
        HDMM algorithm.
        """
        A = self.workload_A
        ATA = A.T @ A
        d = self.d

        # Start with identity
        B_list = [np.eye(d, dtype=np.float64)]

        # Try adding the workload queries themselves
        for i in range(min(self.m, d)):
            row = A[i:i+1, :]
            if np.linalg.norm(row) > 1e-10:
                B_list.append(row / np.linalg.norm(row))

        B = np.vstack(B_list)
        # Remove near-duplicate rows
        _, idx = np.unique(np.round(B, 8), axis=0, return_index=True)
        B = B[np.sort(idx)]

        return B

    def _optimal_prefix_strategy(self) -> FloatArray:
        """Optimal strategy for prefix-sum workloads.

        Uses the hierarchical strategy where B is a binary tree of
        partial sums.
        """
        d = self.d
        if d == 1:
            return np.eye(1, dtype=np.float64)

        # Build hierarchical strategy
        levels = int(math.ceil(math.log2(d))) + 1
        rows = []

        for level in range(levels):
            block_size = 2 ** level
            for start in range(0, d, block_size):
                end = min(start + block_size, d)
                row = np.zeros(d, dtype=np.float64)
                row[start:end] = 1.0
                rows.append(row)

        B = np.array(rows, dtype=np.float64)
        # Normalise rows
        norms = np.linalg.norm(B, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return B / norms

    def _calibrate_sigma(self) -> float:
        """Calibrate σ for the strategy matrix B.

        σ = sensitivity(B) · √(2 ln(1.25/δ)) / ε

        where sensitivity(B) = max ||B x - B x'||_2 over adjacent
        (x, x'), which equals the max L2 column norm of B for
        unit L1 sensitivity.
        """
        B = self.strategy_B
        col_norms = np.linalg.norm(B, axis=0)
        l2_sensitivity = float(np.max(col_norms)) * self.sensitivity
        return l2_sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon

    def sample(
        self,
        true_data: FloatArray,
        n_samples: int = 1,
    ) -> FloatArray:
        """Apply the matrix mechanism.

        Args:
            true_data: True data vector of shape ``(d,)``.
            n_samples: Number of independent samples.

        Returns:
            Estimated workload answers ``(n_samples, m)``.
        """
        true_data = np.asarray(true_data, dtype=np.float64)
        if true_data.shape != (self.d,):
            raise ValueError(
                f"true_data must have shape ({self.d},), got {true_data.shape}"
            )

        B = self.strategy_B
        p = B.shape[0]

        # Strategy answers: B x
        Bx = B @ true_data

        results = np.empty((n_samples, self.m), dtype=np.float64)
        for s in range(n_samples):
            # Add noise
            noise = self._rng.normal(0, self.sigma, size=p)
            noisy_Bx = Bx + noise

            # Reconstruct: A B^+ (Bx + noise)
            B_pinv = np.linalg.pinv(B)
            x_hat = B_pinv @ noisy_Bx
            results[s] = self.workload_A @ x_hat

        return results.squeeze() if n_samples == 1 else results

    def mse_per_query(self) -> FloatArray:
        """Compute per-query MSE analytically.

        ``MSE_i = σ² · ||A_i B^+||_2²``

        Returns:
            Array of shape ``(m,)`` with MSE for each workload query.
        """
        B_pinv = np.linalg.pinv(self.strategy_B)
        AB_pinv = self.workload_A @ B_pinv
        return self.sigma ** 2 * np.sum(AB_pinv ** 2, axis=1)

    def total_mse(self) -> float:
        """Total MSE summed over all workload queries.

        Returns:
            Sum of per-query MSEs.
        """
        return float(np.sum(self.mse_per_query()))

    def mean_mse(self) -> float:
        """Average MSE over workload queries.

        Returns:
            Mean of per-query MSEs.
        """
        return float(np.mean(self.mse_per_query()))

    def max_mse(self) -> float:
        """Maximum per-query MSE (worst-case query).

        Returns:
            Max per-query MSE.
        """
        return float(np.max(self.mse_per_query()))

    def __repr__(self) -> str:
        return (
            f"MatrixMechanism(m={self.m}, d={self.d}, "
            f"strategy={self.strategy_name!r}, σ={self.sigma:.4f})"
        )


# =========================================================================
# 6. Exponential Mechanism
# =========================================================================


class ExponentialMechanism:
    """The exponential mechanism for non-numeric outputs (McSherry & Talwar 2007).

    Given a score function ``u(x, r)`` mapping database × output to a
    real-valued utility, the exponential mechanism selects output *r*
    with probability proportional to ``exp(ε · u(x, r) / (2Δu))``.

    Attributes:
        epsilon: Privacy parameter ε.
        score_fn: Score function ``(database, output) → float``.
        sensitivity: Sensitivity of the score function Δu.
        outputs: List of possible output values.

    Example::

        >>> def score(db, r): return -abs(db - r)
        >>> mech = ExponentialMechanism(
        ...     epsilon=1.0,
        ...     score_fn=score,
        ...     sensitivity=1.0,
        ...     outputs=list(range(10)),
        ... )
        >>> result = mech.sample(5)
    """

    def __init__(
        self,
        epsilon: float,
        score_fn: Callable[[Any, Any], float],
        sensitivity: float,
        outputs: Sequence[Any],
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ConfigurationError(
                f"epsilon must be > 0, got {epsilon}",
                parameter="epsilon", value=epsilon, constraint="epsilon > 0",
            )
        if sensitivity <= 0:
            raise ConfigurationError(
                f"sensitivity must be > 0, got {sensitivity}",
                parameter="sensitivity", value=sensitivity,
                constraint="sensitivity > 0",
            )
        if len(outputs) == 0:
            raise ValueError("outputs must be non-empty")

        self.epsilon = epsilon
        self.score_fn = score_fn
        self.sensitivity = sensitivity
        self.outputs = list(outputs)
        self._rng = np.random.default_rng(seed)

    def _compute_probabilities(self, database: Any) -> FloatArray:
        """Compute selection probabilities for each output.

        Args:
            database: The database (input to score function).

        Returns:
            Probability array over outputs.
        """
        scores = np.array(
            [self.score_fn(database, r) for r in self.outputs],
            dtype=np.float64,
        )
        # log-prob ∝ ε · score / (2Δu)
        log_probs = self.epsilon * scores / (2.0 * self.sensitivity)

        # Numerically stable softmax
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs /= probs.sum()
        return probs

    def sample(
        self,
        database: Any,
        n_samples: int = 1,
    ) -> Union[Any, List[Any]]:
        """Select output(s) from the exponential mechanism.

        Args:
            database: The database.
            n_samples: Number of independent selections.

        Returns:
            Selected output (if n_samples=1) or list of outputs.
        """
        probs = self._compute_probabilities(database)
        indices = self._rng.choice(len(self.outputs), size=n_samples, p=probs)

        if n_samples == 1:
            return self.outputs[indices[0]]
        return [self.outputs[i] for i in indices]

    def probabilities(self, database: Any) -> Dict[Any, float]:
        """Return the probability of each output.

        Args:
            database: The database.

        Returns:
            Dict mapping output → probability.
        """
        probs = self._compute_probabilities(database)
        return {r: float(p) for r, p in zip(self.outputs, probs)}

    def expected_score(self, database: Any) -> float:
        """Expected score of the selected output.

        Args:
            database: The database.

        Returns:
            Expected utility.
        """
        probs = self._compute_probabilities(database)
        scores = np.array(
            [self.score_fn(database, r) for r in self.outputs],
            dtype=np.float64,
        )
        return float(np.dot(probs, scores))

    def max_score(self, database: Any) -> float:
        """Maximum achievable score (non-private optimum).

        Args:
            database: The database.

        Returns:
            Maximum score.
        """
        scores = [self.score_fn(database, r) for r in self.outputs]
        return float(max(scores))

    def utility_loss(self, database: Any) -> float:
        """Expected utility loss compared to the non-private optimum.

        Returns:
            ``max_score - expected_score``.
        """
        return self.max_score(database) - self.expected_score(database)

    def __repr__(self) -> str:
        return (
            f"ExponentialMechanism(ε={self.epsilon}, "
            f"Δu={self.sensitivity}, |R|={len(self.outputs)})"
        )


# =========================================================================
# 7. Randomized Response Mechanism
# =========================================================================


class RandResponseMechanism:
    """Randomized response for binary data (Warner 1965).

    For a binary value ``b ∈ {0, 1}``, the mechanism reports:
    - The true value with probability ``p = exp(ε) / (1 + exp(ε))``
    - The flipped value with probability ``1 - p``

    This satisfies ε-differential privacy.

    Generalised to k-ary randomized response for categorical data.

    Attributes:
        epsilon: Privacy parameter ε.
        n_categories: Number of categories (2 for binary).
        p_true: Probability of reporting the true value.
        p_false: Probability of reporting each false value.

    Example::

        >>> mech = RandResponseMechanism(epsilon=1.0)
        >>> mech.p_true  # doctest: +ELLIPSIS
        0.731...
        >>> mech.error_rate()  # doctest: +ELLIPSIS
        0.268...
    """

    def __init__(
        self,
        epsilon: float,
        n_categories: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ConfigurationError(
                f"epsilon must be > 0, got {epsilon}",
                parameter="epsilon", value=epsilon, constraint="epsilon > 0",
            )
        if n_categories < 2:
            raise ConfigurationError(
                f"n_categories must be >= 2, got {n_categories}",
                parameter="n_categories",
                value=n_categories,
                constraint="n_categories >= 2",
            )

        self.epsilon = epsilon
        self.n_categories = n_categories
        self._rng = np.random.default_rng(seed)

        # For k-ary RR:
        # p_true = exp(ε) / (exp(ε) + k - 1)
        # p_false = 1 / (exp(ε) + k - 1) for each non-true category
        exp_eps = math.exp(epsilon)
        self.p_true = exp_eps / (exp_eps + n_categories - 1)
        self.p_false = 1.0 / (exp_eps + n_categories - 1)

    def sample(
        self,
        true_value: Union[int, Sequence[int]],
        n_samples: int = 1,
    ) -> Union[int, FloatArray]:
        """Apply randomized response.

        Args:
            true_value: True category index(es) in ``[0, n_categories)``.
            n_samples: Number of independent responses per value.

        Returns:
            Randomized response(s).
        """
        if isinstance(true_value, (int, np.integer)):
            values = [int(true_value)]
        else:
            values = list(true_value)

        results = []
        for tv in values:
            if not (0 <= tv < self.n_categories):
                raise ValueError(
                    f"true_value must be in [0, {self.n_categories}), got {tv}"
                )
            for _ in range(n_samples):
                u = self._rng.random()
                if u < self.p_true:
                    results.append(tv)
                else:
                    # Uniform over other categories
                    others = [c for c in range(self.n_categories) if c != tv]
                    results.append(self._rng.choice(others))

        result_arr = np.array(results, dtype=np.int64)
        if isinstance(true_value, (int, np.integer)) and n_samples == 1:
            return int(result_arr[0])
        return result_arr

    def error_rate(self) -> float:
        """Probability of reporting a wrong value.

        Returns:
            ``1 - p_true``.
        """
        return 1.0 - self.p_true

    def confusion_matrix(self) -> FloatArray:
        """Return the confusion matrix (transition probability matrix).

        ``C[i, j] = Pr[report j | true value i]``

        Returns:
            Confusion matrix of shape ``(n_categories, n_categories)``.
        """
        k = self.n_categories
        C = np.full((k, k), self.p_false, dtype=np.float64)
        np.fill_diagonal(C, self.p_true)
        return C

    def mse(self) -> float:
        """Mean squared error for binary RR (0/1 values).

        For binary: ``MSE = p_false`` (since error is 0 or 1).

        Returns:
            MSE for binary case.
        """
        if self.n_categories == 2:
            return self.p_false
        # For k-ary, depends on coding; return error rate as proxy
        return 1.0 - self.p_true

    def mutual_information(self) -> float:
        """Mutual information between true and reported values (in nats).

        Returns:
            ``I(X; Y)`` for the randomized response channel.
        """
        k = self.n_categories
        # Assuming uniform prior
        # H(Y) - H(Y|X)
        # H(Y|X) = -(p_true * log(p_true) + (k-1) * p_false * log(p_false))
        h_yx = -(
            self.p_true * math.log(max(self.p_true, 1e-300))
            + (k - 1) * self.p_false * math.log(max(self.p_false, 1e-300))
        )
        # H(Y) for uniform prior = log(k) (also uniform output)
        h_y = math.log(k)
        return max(h_y - h_yx, 0.0)

    def decode_frequency(
        self,
        counts: FloatArray,
    ) -> FloatArray:
        """Decode aggregated randomized response counts.

        Given counts of reported values, estimates the true frequency
        distribution using the inverse of the confusion matrix.

        Args:
            counts: Array of shape ``(n_categories,)`` with report counts.

        Returns:
            Estimated true frequency distribution.
        """
        counts = np.asarray(counts, dtype=np.float64)
        if len(counts) != self.n_categories:
            raise ValueError(
                f"counts length ({len(counts)}) must match "
                f"n_categories ({self.n_categories})"
            )
        C = self.confusion_matrix()
        total = counts.sum()
        if total == 0:
            return np.ones(self.n_categories) / self.n_categories

        freq = counts / total
        try:
            C_inv = np.linalg.inv(C)
            decoded = C_inv @ freq
        except np.linalg.LinAlgError:
            decoded = np.linalg.lstsq(C, freq, rcond=None)[0]

        # Project onto simplex
        decoded = np.maximum(decoded, 0.0)
        s = decoded.sum()
        if s > 0:
            decoded /= s
        else:
            decoded = np.ones(self.n_categories) / self.n_categories

        return decoded

    def __repr__(self) -> str:
        return (
            f"RandResponseMechanism(ε={self.epsilon}, "
            f"k={self.n_categories}, p_true={self.p_true:.4f})"
        )


# =========================================================================
# 8. Baseline Comparator
# =========================================================================


@dataclass
class ComparisonResult:
    """Result of comparing a synthesised mechanism against baselines.

    Attributes:
        synthesised_mse: MSE of the synthesised mechanism.
        synthesised_mae: MAE of the synthesised mechanism.
        baseline_results: Dict mapping baseline name → {mse, mae, variance}.
        improvement_factors: Dict mapping baseline name → MSE ratio.
        confidence_intervals: Dict mapping baseline name → (lo, hi) CI.
        spec: The query specification used.
    """
    synthesised_mse: float
    synthesised_mae: float
    baseline_results: Dict[str, Dict[str, float]]
    improvement_factors: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    spec: Optional[QuerySpec] = None

    def best_improvement(self) -> Tuple[str, float]:
        """Return the baseline with the best improvement factor.

        Returns:
            ``(baseline_name, improvement_factor)`` tuple.
        """
        if not self.improvement_factors:
            return ("none", 1.0)
        best = max(self.improvement_factors.items(), key=lambda x: x[1])
        return best

    def __repr__(self) -> str:
        best_name, best_factor = self.best_improvement()
        return (
            f"ComparisonResult(synth_mse={self.synthesised_mse:.6f}, "
            f"best_vs={best_name}, improvement={best_factor:.2f}×)"
        )


class BaselineComparator:
    """Compare synthesised mechanisms against standard baselines.

    Runs the synthesised mechanism and a suite of baselines on the same
    query specification, computes error metrics, and generates comparison
    reports with bootstrap confidence intervals.

    Example::

        >>> spec = QuerySpec.counting(n=3, epsilon=1.0, k=50)
        >>> comparator = BaselineComparator()
        >>> # result = comparator.compare(synth_probs, spec, n_samples=10000)
    """

    def __init__(
        self,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the comparator.

        Args:
            seed: Random seed for reproducibility.
        """
        self._rng = np.random.default_rng(seed)
        self.seed = seed

    def compare(
        self,
        synthesised: FloatArray,
        baselines: Optional[Dict[str, _BaseMechanism]] = None,
        spec: Optional[QuerySpec] = None,
        n_samples: int = 10000,
        y_grid: Optional[FloatArray] = None,
    ) -> ComparisonResult:
        """Run a full comparison between synthesised and baseline mechanisms.

        Args:
            synthesised: Synthesised mechanism probability table ``(n, k)``.
            baselines: Dict of baseline mechanisms.  If ``None``, builds
                defaults from *spec*.
            spec: Query specification (needed for default baselines).
            n_samples: Number of Monte Carlo samples per mechanism.
            y_grid: Output grid for the synthesised mechanism.

        Returns:
            Comparison result with MSE, MAE, and improvement factors.
        """
        synthesised = np.asarray(synthesised, dtype=np.float64)

        # Build default baselines if not provided
        if baselines is None:
            if spec is None:
                raise ValueError("Either baselines or spec must be provided")
            baselines = self._build_default_baselines(spec)

        # Compute synthesised mechanism error
        if spec is not None and y_grid is None:
            from dp_forge.numerical import compute_output_grid
            y_grid = compute_output_grid(spec.query_values, synthesised.shape[1])

        synth_mse = self._compute_discrete_mse(synthesised, spec, y_grid)
        synth_mae = self._compute_discrete_mae(synthesised, spec, y_grid)

        # Compute baseline errors
        baseline_results: Dict[str, Dict[str, float]] = {}
        improvement_factors: Dict[str, float] = {}

        for name, mech in baselines.items():
            bl_mse = mech.mse()
            bl_mae = mech.mae()
            bl_var = mech.variance()
            baseline_results[name] = {
                "mse": bl_mse,
                "mae": bl_mae,
                "variance": bl_var,
            }
            if synth_mse > 0:
                improvement_factors[name] = bl_mse / synth_mse
            else:
                improvement_factors[name] = float("inf")

        return ComparisonResult(
            synthesised_mse=synth_mse,
            synthesised_mae=synth_mae,
            baseline_results=baseline_results,
            improvement_factors=improvement_factors,
            spec=spec,
        )

    def compute_improvement_factor(
        self,
        synthesised_mse: float,
        baseline_mse: float,
    ) -> float:
        """Compute MSE improvement factor.

        Args:
            synthesised_mse: MSE of the synthesised mechanism.
            baseline_mse: MSE of the baseline mechanism.

        Returns:
            ``baseline_mse / synthesised_mse`` (> 1 means improvement).
        """
        if synthesised_mse <= 0:
            return float("inf")
        return baseline_mse / synthesised_mse

    def bootstrap_ci(
        self,
        synthesised: FloatArray,
        baseline: _BaseMechanism,
        spec: QuerySpec,
        n_samples: int = 10000,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        y_grid: Optional[FloatArray] = None,
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for the improvement factor.

        Args:
            synthesised: Synthesised mechanism table ``(n, k)``.
            baseline: Baseline mechanism to compare against.
            spec: Query specification.
            n_samples: Samples per bootstrap iteration.
            n_bootstrap: Number of bootstrap resamples.
            confidence: Confidence level.
            y_grid: Output grid.

        Returns:
            ``(lower, upper)`` bounds of the CI.
        """
        if y_grid is None:
            from dp_forge.numerical import compute_output_grid
            y_grid = compute_output_grid(spec.query_values, synthesised.shape[1])

        improvement_samples = np.empty(n_bootstrap, dtype=np.float64)

        for b in range(n_bootstrap):
            # Bootstrap: resample query values
            boot_indices = self._rng.choice(spec.n, size=spec.n, replace=True)

            # Synthesised MSE on bootstrap sample
            boot_synth = synthesised[boot_indices]
            boot_qv = spec.query_values[boot_indices]

            sq_errors = (boot_qv[:, np.newaxis] - y_grid[np.newaxis, :]) ** 2
            synth_mse_boot = float(np.mean(np.sum(boot_synth * sq_errors, axis=1)))

            # Baseline MSE (analytical, doesn't change with bootstrap)
            bl_mse = baseline.mse()

            if synth_mse_boot > 0:
                improvement_samples[b] = bl_mse / synth_mse_boot
            else:
                improvement_samples[b] = float("inf")

        alpha = (1 - confidence) / 2
        lo = float(np.percentile(improvement_samples, 100 * alpha))
        hi = float(np.percentile(improvement_samples, 100 * (1 - alpha)))
        return (lo, hi)

    def generate_comparison_report(
        self,
        result: ComparisonResult,
        title: str = "DP Mechanism Comparison",
    ) -> str:
        """Generate a formatted text comparison report.

        Args:
            result: Comparison result.
            title: Report title.

        Returns:
            Formatted report string.
        """
        lines = [
            f"{'=' * 60}",
            f"  {title}",
            f"{'=' * 60}",
            "",
            f"Synthesised Mechanism:",
            f"  MSE:  {result.synthesised_mse:.8f}",
            f"  MAE:  {result.synthesised_mae:.8f}",
            "",
            f"{'─' * 60}",
            f"{'Baseline':<25} {'MSE':>12} {'MAE':>12} {'Improvement':>12}",
            f"{'─' * 60}",
        ]

        for name in sorted(result.baseline_results.keys()):
            bl = result.baseline_results[name]
            imp = result.improvement_factors.get(name, 1.0)
            lines.append(
                f"{name:<25} {bl['mse']:>12.6f} {bl['mae']:>12.6f} {imp:>11.2f}×"
            )

            if name in result.confidence_intervals:
                lo, hi = result.confidence_intervals[name]
                lines.append(f"{'':>25} {'95% CI':>12} [{lo:.2f}, {hi:.2f}]")

        lines.extend([
            f"{'─' * 60}",
            "",
        ])

        best_name, best_factor = result.best_improvement()
        if best_factor > 1.0:
            lines.append(
                f"✓ Synthesised mechanism improves on best baseline "
                f"({best_name}) by {best_factor:.2f}×"
            )
        elif best_factor < 1.0:
            lines.append(
                f"✗ Synthesised mechanism is {1/best_factor:.2f}× worse "
                f"than best baseline ({best_name})"
            )
        else:
            lines.append(f"= Synthesised mechanism matches best baseline ({best_name})")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    # --- Private helpers ---

    def _build_default_baselines(
        self,
        spec: QuerySpec,
    ) -> Dict[str, _BaseMechanism]:
        """Build default baseline mechanisms from a query spec."""
        baselines: Dict[str, _BaseMechanism] = {}

        # Always include Laplace
        baselines["Laplace"] = LaplaceMechanism(
            epsilon=spec.epsilon,
            sensitivity=spec.sensitivity,
            seed=self.seed,
        )

        # Include Gaussian if delta > 0
        if spec.delta > 0:
            baselines["Gaussian"] = GaussianMechanism(
                epsilon=spec.epsilon,
                delta=spec.delta,
                sensitivity=spec.sensitivity,
                seed=self.seed,
            )

        # Include Staircase for counting queries
        if spec.query_type.name in ("COUNTING", "HISTOGRAM"):
            baselines["Staircase"] = StaircaseMechanism(
                epsilon=spec.epsilon,
                sensitivity=spec.sensitivity,
                seed=self.seed,
            )

        # Include Geometric for integer-valued queries
        if spec.sensitivity == 1.0 and spec.query_type.name in ("COUNTING",):
            baselines["Geometric"] = GeometricMechanism(
                epsilon=spec.epsilon,
                sensitivity=1,
                seed=self.seed,
            )

        return baselines

    @staticmethod
    def _compute_discrete_mse(
        mechanism: FloatArray,
        spec: Optional[QuerySpec],
        y_grid: Optional[FloatArray],
    ) -> float:
        """Compute MSE of a discrete mechanism table."""
        if spec is None or y_grid is None:
            return 0.0

        n, k = mechanism.shape
        sq_errors = (spec.query_values[:, np.newaxis] - y_grid[np.newaxis, :]) ** 2
        per_input = np.sum(mechanism * sq_errors, axis=1)
        return float(np.mean(per_input))

    @staticmethod
    def _compute_discrete_mae(
        mechanism: FloatArray,
        spec: Optional[QuerySpec],
        y_grid: Optional[FloatArray],
    ) -> float:
        """Compute MAE of a discrete mechanism table."""
        if spec is None or y_grid is None:
            return 0.0

        abs_errors = np.abs(spec.query_values[:, np.newaxis] - y_grid[np.newaxis, :])
        per_input = np.sum(mechanism * abs_errors, axis=1)
        return float(np.mean(per_input))


# =========================================================================
# Utility: quick_compare helper
# =========================================================================


def quick_compare(
    spec: QuerySpec,
    synthesised: FloatArray,
    y_grid: Optional[FloatArray] = None,
    seed: Optional[int] = None,
) -> ComparisonResult:
    """One-line comparison of a synthesised mechanism against defaults.

    Args:
        spec: Query specification.
        synthesised: Mechanism probability table.
        y_grid: Output grid.
        seed: Random seed.

    Returns:
        Comparison result.
    """
    comparator = BaselineComparator(seed=seed)
    return comparator.compare(synthesised, spec=spec, y_grid=y_grid)


def list_baselines() -> List[str]:
    """Return names of all available baseline mechanism classes.

    Returns:
        List of class names.
    """
    return [
        "LaplaceMechanism",
        "GaussianMechanism",
        "GeometricMechanism",
        "StaircaseMechanism",
        "MatrixMechanism",
        "ExponentialMechanism",
        "RandResponseMechanism",
    ]

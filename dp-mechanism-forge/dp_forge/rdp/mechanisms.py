"""
RDP characterisation of standard DP mechanisms.

Provides exact and approximate RDP curves for:
    - **Gaussian mechanism**: Exact formula α·Δ²/(2σ²).
    - **Laplace mechanism**: Exact formula via moment generating function.
    - **Discrete LP-synthesised mechanisms**: Exact Rényi divergence from
      probability tables using log-sum-exp.
    - **Subsampled mechanisms**: Privacy amplification by subsampling
      (Poisson and without-replacement).
    - **Randomised response**: Exact RDP for binary randomised response.

All computations use log-domain arithmetic for numerical stability and
handle edge cases (α → 1 for KL, α → ∞ for max divergence).

References:
    - Mironov, I. (2017). Rényi differential privacy.
    - Balle, B., Gaboardi, M., & Zanella-Béguelin, B. (2020).
      Privacy profiles and amplification by subsampling.
    - Wang, Y.-X., Balle, B., & Kasiviswanathan, S.P. (2019).
      Subsampled Rényi differential privacy and analytical moments
      accountant.
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
    InvalidMechanismError,
)
from dp_forge.types import PrivacyBudget

from dp_forge.rdp.accountant import RDPCurve, DEFAULT_ALPHAS
from dp_forge.rdp.renyi_divergence import RenyiDivergenceComputer

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


# =========================================================================
# RDPMechanismCharacterizer
# =========================================================================


class RDPMechanismCharacterizer:
    """Compute RDP curves for standard differential privacy mechanisms.

    Provides exact or tight approximate RDP characterisations for
    Gaussian, Laplace, discrete (LP-synthesised), and subsampled
    mechanisms.

    All methods return :class:`RDPCurve` objects that can be directly
    used with :class:`RDPAccountant`.

    Args:
        alphas: Default α grid for all computations. Defaults to
            :data:`DEFAULT_ALPHAS`.
        min_prob: Minimum probability threshold for discrete
            mechanism computations.

    Example::

        char = RDPMechanismCharacterizer()
        curve = char.gaussian(sigma=1.0, sensitivity=1.0)
        budget = curve.to_dp(delta=1e-5)
    """

    def __init__(
        self,
        alphas: Optional[FloatArray] = None,
        min_prob: float = 1e-300,
    ) -> None:
        self._alphas = (
            np.asarray(alphas, dtype=np.float64)
            if alphas is not None
            else DEFAULT_ALPHAS.copy()
        )
        self._divergence_computer = RenyiDivergenceComputer(min_prob=min_prob)

    @property
    def alphas(self) -> FloatArray:
        """The default α grid."""
        return self._alphas.copy()

    # -----------------------------------------------------------------
    # Gaussian mechanism
    # -----------------------------------------------------------------

    def gaussian(
        self,
        sigma: float,
        sensitivity: float = 1.0,
        alphas: Optional[FloatArray] = None,
    ) -> RDPCurve:
        """Exact RDP for the Gaussian mechanism.

        The Gaussian mechanism adding N(0, σ²I) noise to a query with
        sensitivity Δ satisfies (α, α·Δ²/(2σ²))-RDP for all α > 1.

        This is the fundamental result from Mironov (2017).

        Args:
            sigma: Noise standard deviation (> 0).
            sensitivity: Query L2 sensitivity Δ (> 0).
            alphas: Custom α grid. Defaults to instance grid.

        Returns:
            Exact RDP curve.

        Raises:
            ConfigurationError: If parameters are invalid.
        """
        if sigma <= 0:
            raise ConfigurationError(
                f"sigma must be positive, got {sigma}",
                parameter="sigma",
                value=sigma,
            )
        if sensitivity <= 0:
            raise ConfigurationError(
                f"sensitivity must be positive, got {sensitivity}",
                parameter="sensitivity",
                value=sensitivity,
            )

        a = alphas if alphas is not None else self._alphas
        a = np.asarray(a, dtype=np.float64)
        rdp_eps = a * sensitivity ** 2 / (2.0 * sigma ** 2)

        return RDPCurve(
            alphas=a.copy(),
            epsilons=rdp_eps,
            name=f"Gaussian(σ={sigma:.4f}, Δ={sensitivity:.4f})",
            metadata={"sigma": sigma, "sensitivity": sensitivity},
        )

    def gaussian_from_budget(
        self,
        budget: PrivacyBudget,
        sensitivity: float = 1.0,
        alphas: Optional[FloatArray] = None,
    ) -> RDPCurve:
        """RDP curve for the Gaussian mechanism calibrated to a privacy budget.

        Computes σ = Δ·√(2 log(1.25/δ)) / ε, then returns the exact RDP.

        Args:
            budget: Target (ε, δ) privacy budget.
            sensitivity: Query sensitivity.
            alphas: Custom α grid.

        Returns:
            RDP curve for the calibrated Gaussian mechanism.
        """
        delta = budget.delta if budget.delta > 0 else 1e-10
        sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / budget.epsilon

        return self.gaussian(sigma, sensitivity, alphas)

    # -----------------------------------------------------------------
    # Laplace mechanism
    # -----------------------------------------------------------------

    def laplace(
        self,
        epsilon: float,
        sensitivity: float = 1.0,
        alphas: Optional[FloatArray] = None,
    ) -> RDPCurve:
        """Exact RDP for the Laplace mechanism.

        The Laplace mechanism with scale b = Δ/ε satisfies ε-DP.
        Its RDP guarantee at order α is:

            ε̂(α) = 1/(α-1) log( α/(2α-1) exp((α-1)ε) + (α-1)/(2α-1) exp(-(α-1)ε) )

        For α → 1, this converges to 0 (the KL divergence is bounded
        by ε·(eᵉ-1)/(eᵉ+1), which approaches 0 differently than
        the RDP formula).

        Args:
            epsilon: Laplace privacy parameter ε > 0 (where b = Δ/ε).
            sensitivity: Query L1 sensitivity Δ. The effective ε used
                in the RDP formula is ε·sensitivity/Δ = ε when Δ=1.
            alphas: Custom α grid.

        Returns:
            Exact RDP curve.
        """
        if epsilon <= 0:
            raise ConfigurationError(
                f"epsilon must be positive, got {epsilon}",
                parameter="epsilon",
                value=epsilon,
            )
        if sensitivity <= 0:
            raise ConfigurationError(
                f"sensitivity must be positive, got {sensitivity}",
                parameter="sensitivity",
                value=sensitivity,
            )

        effective_eps = epsilon * sensitivity

        a = alphas if alphas is not None else self._alphas
        a = np.asarray(a, dtype=np.float64)

        rdp_eps = np.empty_like(a)
        for i, alpha in enumerate(a):
            if abs(alpha - 1.0) < 1e-10:
                rdp_eps[i] = effective_eps * (math.exp(effective_eps) - 1.0) / (math.exp(effective_eps) + 1.0)
                continue

            a_m1 = alpha - 1.0
            denom = 2.0 * alpha - 1.0

            if denom <= 0:
                rdp_eps[i] = 0.0
                continue

            log_t1 = math.log(alpha / denom) + a_m1 * effective_eps
            log_t2 = math.log(a_m1 / denom) - a_m1 * effective_eps
            log_sum = float(np.logaddexp(log_t1, log_t2))
            rdp_eps[i] = max(log_sum / a_m1, 0.0)

        return RDPCurve(
            alphas=a.copy(),
            epsilons=rdp_eps,
            name=f"Laplace(ε={epsilon:.4f}, Δ={sensitivity:.4f})",
            metadata={"epsilon": epsilon, "sensitivity": sensitivity},
        )

    # -----------------------------------------------------------------
    # Discrete (LP-synthesised) mechanisms
    # -----------------------------------------------------------------

    def discrete(
        self,
        mechanism_table: FloatArray,
        adjacent_pairs: Sequence[Tuple[int, int]],
        alphas: Optional[FloatArray] = None,
    ) -> RDPCurve:
        """Exact RDP for a discrete mechanism from its probability table.

        Given a mechanism table p[i][j] = Pr[M(x_i) = y_j] and a set
        of adjacent database pairs, computes the worst-case Rényi
        divergence across all pairs and all α.

        D_α(M(x_i) || M(x_{i'})) = 1/(α-1) log Σ_j p[i][j]^α p[i'][j]^(1-α)

        The RDP guarantee is the maximum over all adjacent pairs.

        Args:
            mechanism_table: Probability table of shape ``(n, k)`` where
                p[i][j] = Pr[M(x_i) = y_j].
            adjacent_pairs: List of adjacent database pairs ``(i, i')``.
            alphas: Custom α grid.

        Returns:
            RDP curve (worst case over all adjacent pairs).

        Raises:
            InvalidMechanismError: If the mechanism table is invalid.
        """
        mechanism_table = np.asarray(mechanism_table, dtype=np.float64)
        if mechanism_table.ndim != 2:
            raise InvalidMechanismError(
                f"mechanism_table must be 2-D, got shape {mechanism_table.shape}",
                reason="wrong dimensionality",
            )

        # Validate probability table
        if np.any(mechanism_table < -1e-12):
            raise InvalidMechanismError(
                f"mechanism_table contains negative values (min={np.min(mechanism_table):.2e})",
                reason="negative probabilities",
            )
        row_sums = mechanism_table.sum(axis=1)
        if np.any(np.abs(row_sums - 1.0) > 1e-6):
            raise InvalidMechanismError(
                f"mechanism_table rows must sum to 1 (max deviation={np.max(np.abs(row_sums - 1.0)):.2e})",
                reason="rows don't sum to 1",
            )

        if not adjacent_pairs:
            raise ConfigurationError(
                "At least one adjacent pair is required",
                parameter="adjacent_pairs",
            )

        a = alphas if alphas is not None else self._alphas
        a = np.asarray(a, dtype=np.float64)

        n_rows = mechanism_table.shape[0]

        # Compute worst-case RDP across all pairs
        worst_rdp = np.zeros_like(a)

        for i, ip in adjacent_pairs:
            if not (0 <= i < n_rows and 0 <= ip < n_rows):
                raise ConfigurationError(
                    f"Adjacent pair ({i}, {ip}) out of range for table with {n_rows} rows",
                    parameter="adjacent_pairs",
                )

            p_i = mechanism_table[i]
            p_ip = mechanism_table[ip]

            # Compute Rényi divergence for both directions
            rdp_forward = self._divergence_computer.exact_discrete_vectorized(p_i, p_ip, a)
            rdp_backward = self._divergence_computer.exact_discrete_vectorized(p_ip, p_i, a)

            # RDP guarantee is the max over both directions
            pair_rdp = np.maximum(rdp_forward, rdp_backward)
            worst_rdp = np.maximum(worst_rdp, pair_rdp)

        return RDPCurve(
            alphas=a.copy(),
            epsilons=worst_rdp,
            name=f"Discrete({mechanism_table.shape[0]}×{mechanism_table.shape[1]})",
            metadata={
                "n_inputs": mechanism_table.shape[0],
                "n_outputs": mechanism_table.shape[1],
                "n_pairs": len(adjacent_pairs),
            },
        )

    def discrete_from_cegis(
        self,
        mechanism_table: FloatArray,
        n: int,
        alphas: Optional[FloatArray] = None,
    ) -> RDPCurve:
        """RDP for a CEGIS-synthesised mechanism with Hamming-1 adjacency.

        Convenience method that uses consecutive pairs ``(i, i+1)`` as
        the adjacency relation (standard for counting/histogram queries).

        Args:
            mechanism_table: Probability table of shape ``(n, k)``.
            n: Number of database inputs (rows in the table).
            alphas: Custom α grid.

        Returns:
            RDP curve.
        """
        pairs = [(i, i + 1) for i in range(n - 1)]
        return self.discrete(mechanism_table, pairs, alphas)

    # -----------------------------------------------------------------
    # Subsampled mechanisms
    # -----------------------------------------------------------------

    def subsampled_gaussian(
        self,
        sigma: float,
        sampling_rate: float,
        sensitivity: float = 1.0,
        alphas: Optional[FloatArray] = None,
    ) -> RDPCurve:
        """RDP for the subsampled Gaussian mechanism.

        Applies privacy amplification by Poisson subsampling to the
        Gaussian mechanism. Uses the tight bound from Wang, Balle,
        Kasiviswanathan (2019).

        For integer α ≥ 2, the exact bound uses the binomial expansion
        of the Rényi divergence. For non-integer α, uses an upper bound.

        Args:
            sigma: Noise standard deviation (> 0).
            sampling_rate: Subsampling probability q ∈ (0, 1].
            sensitivity: Query sensitivity (> 0).
            alphas: Custom α grid.

        Returns:
            RDP curve with subsampling amplification.
        """
        if sigma <= 0:
            raise ConfigurationError(
                f"sigma must be positive, got {sigma}",
                parameter="sigma",
                value=sigma,
            )
        if not (0 < sampling_rate <= 1):
            raise ConfigurationError(
                f"sampling_rate must be in (0, 1], got {sampling_rate}",
                parameter="sampling_rate",
                value=sampling_rate,
            )
        if sensitivity <= 0:
            raise ConfigurationError(
                f"sensitivity must be positive, got {sensitivity}",
                parameter="sensitivity",
                value=sensitivity,
            )

        a = alphas if alphas is not None else self._alphas
        a = np.asarray(a, dtype=np.float64)

        if sampling_rate == 1.0:
            return self.gaussian(sigma, sensitivity, a)

        rdp_eps = np.empty_like(a)
        q = sampling_rate

        for i, alpha in enumerate(a):
            if abs(alpha - 1.0) < 1e-10:
                # KL divergence of subsampled mechanism
                base_kl = sensitivity ** 2 / (2.0 * sigma ** 2)
                rdp_eps[i] = q * base_kl
                continue

            # Integer α: use exact bound via binomial expansion
            # For each j from 0 to α, compute the contribution
            if alpha == int(alpha) and alpha >= 2:
                rdp_eps[i] = self._subsampled_gaussian_integer_alpha(
                    int(alpha), sigma, q, sensitivity
                )
            else:
                # Non-integer: use upper bound
                base_rdp = alpha * sensitivity ** 2 / (2.0 * sigma ** 2)
                inner = (alpha - 1.0) * base_rdp / alpha
                if inner > 500:
                    rdp_eps[i] = base_rdp
                else:
                    amplified = math.log1p(q * (math.exp(inner) - 1.0))
                    rdp_eps[i] = max(alpha * amplified / (alpha - 1.0), 0.0)

        return RDPCurve(
            alphas=a.copy(),
            epsilons=rdp_eps,
            name=f"SubsampledGaussian(σ={sigma:.4f}, q={sampling_rate:.4f})",
            metadata={
                "sigma": sigma,
                "sampling_rate": sampling_rate,
                "sensitivity": sensitivity,
            },
        )

    def _subsampled_gaussian_integer_alpha(
        self,
        alpha: int,
        sigma: float,
        q: float,
        sensitivity: float,
    ) -> float:
        """Exact subsampled Gaussian RDP for integer α ≥ 2.

        Uses the bound from Mironov et al. (2019):
        D_α(M_q || M_0) ≤ 1/(α-1) log Σ_{j=0}^{α} C(α,j) (1-q)^{α-j} q^j
                            × exp(j(j-1)Δ²/(2σ²))

        where C(α,j) is the binomial coefficient.
        """
        log_terms = []

        for j in range(alpha + 1):
            # log C(α, j) + (α-j) log(1-q) + j log(q) + j(j-1)Δ²/(2σ²)
            log_binom = self._log_binomial(alpha, j)
            log_mix = (alpha - j) * math.log(1.0 - q) + j * math.log(q) if j < alpha else j * math.log(q)

            if j == alpha:
                log_mix = j * math.log(q)
            elif j == 0:
                log_mix = alpha * math.log(1.0 - q)
            else:
                log_mix = (alpha - j) * math.log(1.0 - q) + j * math.log(q)

            rdp_contrib = j * (j - 1) * sensitivity ** 2 / (2.0 * sigma ** 2)
            log_terms.append(log_binom + log_mix + rdp_contrib)

        log_sum = _logsumexp(np.array(log_terms))
        return max(log_sum / (alpha - 1), 0.0)

    @staticmethod
    def _log_binomial(n: int, k: int) -> float:
        """Compute log(C(n, k)) using lgamma for numerical stability."""
        if k < 0 or k > n:
            return -float("inf")
        return (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
        )

    def subsampled_laplace(
        self,
        epsilon: float,
        sampling_rate: float,
        sensitivity: float = 1.0,
        alphas: Optional[FloatArray] = None,
    ) -> RDPCurve:
        """RDP for the subsampled Laplace mechanism.

        Applies privacy amplification by subsampling to the Laplace
        mechanism. Uses a general upper bound based on the Laplace
        mechanism's RDP and the subsampling amplification lemma.

        Args:
            epsilon: Laplace privacy parameter.
            sampling_rate: Subsampling probability q ∈ (0, 1].
            sensitivity: Query sensitivity.
            alphas: Custom α grid.

        Returns:
            RDP curve with subsampling amplification.
        """
        if epsilon <= 0:
            raise ConfigurationError(
                f"epsilon must be positive, got {epsilon}",
                parameter="epsilon",
                value=epsilon,
            )
        if not (0 < sampling_rate <= 1):
            raise ConfigurationError(
                f"sampling_rate must be in (0, 1], got {sampling_rate}",
                parameter="sampling_rate",
                value=sampling_rate,
            )

        if sampling_rate == 1.0:
            return self.laplace(epsilon, sensitivity, alphas)

        a = alphas if alphas is not None else self._alphas
        a = np.asarray(a, dtype=np.float64)

        base_curve = self.laplace(epsilon, sensitivity, a)
        q = sampling_rate

        # General amplification bound: ε_sub(α) ≤ 1/(α-1) log(1 + q(exp((α-1)ε_base(α)/α) - 1))
        rdp_eps = np.empty_like(a)
        for i, alpha in enumerate(a):
            if abs(alpha - 1.0) < 1e-10:
                rdp_eps[i] = q * base_curve.epsilons[i]
                continue

            base_rdp = base_curve.epsilons[i]
            inner = (alpha - 1.0) * base_rdp / alpha
            if inner > 500:
                rdp_eps[i] = base_rdp
            else:
                amplified = math.log1p(q * (math.exp(inner) - 1.0))
                rdp_eps[i] = max(alpha * amplified / (alpha - 1.0), 0.0)

        return RDPCurve(
            alphas=a.copy(),
            epsilons=rdp_eps,
            name=f"SubsampledLaplace(ε={epsilon:.4f}, q={sampling_rate:.4f})",
            metadata={
                "epsilon": epsilon,
                "sampling_rate": sampling_rate,
                "sensitivity": sensitivity,
            },
        )

    # -----------------------------------------------------------------
    # Randomised response
    # -----------------------------------------------------------------

    def randomised_response(
        self,
        flip_prob: float,
        alphas: Optional[FloatArray] = None,
    ) -> RDPCurve:
        """RDP for binary randomised response.

        Binary randomised response reports the true answer with
        probability 1 - flip_prob and flips with probability flip_prob.

        The mechanism satisfies ε-DP with ε = log((1-p)/p), and has
        exact Rényi divergence between Bernoulli(1-p) and Bernoulli(p).

        Args:
            flip_prob: Probability of flipping the answer, in (0, 0.5).
            alphas: Custom α grid.

        Returns:
            Exact RDP curve.
        """
        if not (0 < flip_prob < 0.5):
            raise ConfigurationError(
                f"flip_prob must be in (0, 0.5), got {flip_prob}",
                parameter="flip_prob",
                value=flip_prob,
            )

        a = alphas if alphas is not None else self._alphas
        a = np.asarray(a, dtype=np.float64)

        p_true = np.array([1.0 - flip_prob, flip_prob])
        p_flip = np.array([flip_prob, 1.0 - flip_prob])

        rdp_eps = self._divergence_computer.exact_discrete_vectorized(p_true, p_flip, a)

        return RDPCurve(
            alphas=a.copy(),
            epsilons=rdp_eps,
            name=f"RandomisedResponse(p={flip_prob:.4f})",
            metadata={"flip_prob": flip_prob},
        )

    # -----------------------------------------------------------------
    # Composed mechanism helpers
    # -----------------------------------------------------------------

    def composed(
        self,
        curves: Sequence[RDPCurve],
        alphas: Optional[FloatArray] = None,
    ) -> RDPCurve:
        """Compose multiple RDP curves via pointwise addition.

        Args:
            curves: Sequence of RDP curves to compose.
            alphas: Common α grid. If ``None``, uses instance grid.

        Returns:
            Composed RDP curve.
        """
        if not curves:
            raise ValueError("At least one curve is required for composition")

        a = alphas if alphas is not None else self._alphas
        a = np.asarray(a, dtype=np.float64)

        composed_eps = np.zeros_like(a)
        names = []
        for curve in curves:
            composed_eps += curve.evaluate_vectorized(a)
            if curve.name:
                names.append(curve.name)

        return RDPCurve(
            alphas=a.copy(),
            epsilons=composed_eps,
            name="+".join(names) if names else "composed",
        )

    def repeated(
        self,
        curve: RDPCurve,
        n_repetitions: int,
    ) -> RDPCurve:
        """RDP for n repetitions of a mechanism (homogeneous composition).

        Args:
            curve: RDP curve of a single mechanism application.
            n_repetitions: Number of times the mechanism is applied.

        Returns:
            Composed RDP curve (scaled by n_repetitions).
        """
        if n_repetitions < 1:
            raise ValueError(f"n_repetitions must be >= 1, got {n_repetitions}")

        return RDPCurve(
            alphas=curve.alphas.copy(),
            epsilons=curve.epsilons * n_repetitions,
            name=f"{curve.name}×{n_repetitions}" if curve.name else f"repeated×{n_repetitions}",
        )

    def __repr__(self) -> str:
        return f"RDPMechanismCharacterizer(n_alphas={len(self._alphas)})"


# =========================================================================
# Standalone RDP functions (log-domain, numerically stable)
# =========================================================================


def subsampled_rdp(
    alpha: float,
    sigma: float,
    q: float,
    sensitivity: float = 1.0,
) -> float:
    """RDP of the Poisson-subsampled Gaussian mechanism (Mironov 2017 bound).

    Computes the (α, ε̂(α))-RDP guarantee for a Gaussian mechanism with
    noise σ applied to a Poisson-subsampled dataset with sampling rate q.

    For integer α ≥ 2, uses the exact bound via binomial expansion of the
    Rényi divergence moments (Mironov, Talwar, Zhang 2019):

        ε̂(α) = (1/(α-1)) log Σ_{j=0}^{α} C(α,j) (1-q)^{α-j} q^j
                 × exp(j(j-1)Δ²/(2σ²))

    For non-integer α, uses interpolation between the two nearest integer
    orders, which provides a valid upper bound.

    All arithmetic is performed in log domain for numerical stability when
    σ is small or α is large.

    Args:
        alpha: Rényi divergence order (> 1).
        sigma: Gaussian noise standard deviation (> 0).
        q: Poisson subsampling rate in (0, 1].
        sensitivity: Query L2 sensitivity Δ (> 0).

    Returns:
        RDP epsilon ε̂(α) for the subsampled Gaussian.

    Raises:
        ConfigurationError: If parameters are invalid.

    References:
        Mironov, I. (2017). Rényi differential privacy.
        Mironov, I., Talwar, K., & Zhang, L. (2019). Rényi Differential
        Privacy of the Sampled Gaussian Mechanism.
    """
    if alpha <= 1.0:
        raise ConfigurationError(
            f"alpha must be > 1, got {alpha}",
            parameter="alpha", value=alpha,
        )
    if sigma <= 0:
        raise ConfigurationError(
            f"sigma must be > 0, got {sigma}",
            parameter="sigma", value=sigma,
        )
    if not (0 < q <= 1):
        raise ConfigurationError(
            f"q must be in (0, 1], got {q}",
            parameter="q", value=q,
        )

    # No subsampling
    if q == 1.0:
        return alpha * sensitivity ** 2 / (2.0 * sigma ** 2)

    # α ≈ 1: KL divergence scaled by q
    if abs(alpha - 1.0) < 1e-10:
        return q * sensitivity ** 2 / (2.0 * sigma ** 2)

    # Integer α: exact binomial expansion
    alpha_int = int(round(alpha)) if abs(alpha - round(alpha)) < 1e-9 else None
    if alpha_int is not None and alpha_int >= 2:
        return _subsampled_gaussian_binomial(alpha_int, sigma, q, sensitivity)

    # Non-integer: interpolate between floor and ceil
    a_lo = max(2, int(math.floor(alpha)))
    a_hi = a_lo + 1
    rdp_lo = _subsampled_gaussian_binomial(a_lo, sigma, q, sensitivity)
    rdp_hi = _subsampled_gaussian_binomial(a_hi, sigma, q, sensitivity)
    # Linear interpolation in RDP (valid upper bound by convexity)
    frac = alpha - a_lo
    return (1.0 - frac) * rdp_lo + frac * rdp_hi


def _subsampled_gaussian_binomial(
    alpha: int,
    sigma: float,
    q: float,
    sensitivity: float,
) -> float:
    """Exact subsampled Gaussian RDP for integer α via binomial expansion.

    Log-domain computation for numerical stability.
    """
    log_terms = np.empty(alpha + 1, dtype=np.float64)
    for j in range(alpha + 1):
        log_binom = (
            math.lgamma(alpha + 1) - math.lgamma(j + 1) - math.lgamma(alpha - j + 1)
        )
        if j == 0:
            log_mix = alpha * math.log(1.0 - q)
        elif j == alpha:
            log_mix = alpha * math.log(q)
        else:
            log_mix = (alpha - j) * math.log(1.0 - q) + j * math.log(q)

        rdp_contrib = j * (j - 1) * sensitivity ** 2 / (2.0 * sigma ** 2)
        log_terms[j] = log_binom + log_mix + rdp_contrib

    log_sum = _logsumexp(log_terms)
    return max(log_sum / (alpha - 1), 0.0)


def randomized_response_rdp(
    alpha: float,
    flip_prob: float,
) -> float:
    """RDP for binary randomised response.

    Binary randomised response reports the true bit with probability
    ``1 - flip_prob`` and flips with probability ``flip_prob``.

    The exact Rényi divergence of order α between Bernoulli(1-p) and
    Bernoulli(p) is:

        D_α = (1/(α-1)) log( (1-p)^α p^{1-α} + p^α (1-p)^{1-α} )

    Computed entirely in log domain to avoid underflow when p is close
    to 0 or 0.5.

    Args:
        alpha: Rényi divergence order (> 1).
        flip_prob: Probability of flipping, in (0, 0.5).

    Returns:
        RDP epsilon at order α.

    Raises:
        ConfigurationError: If parameters are invalid.
    """
    if alpha <= 1.0:
        raise ConfigurationError(
            f"alpha must be > 1, got {alpha}",
            parameter="alpha", value=alpha,
        )
    if not (0 < flip_prob < 0.5):
        raise ConfigurationError(
            f"flip_prob must be in (0, 0.5), got {flip_prob}",
            parameter="flip_prob", value=flip_prob,
        )

    p = flip_prob
    q = 1.0 - p

    # log( q^α p^{1-α} + p^α q^{1-α} )
    log_t1 = alpha * math.log(q) + (1.0 - alpha) * math.log(p)
    log_t2 = alpha * math.log(p) + (1.0 - alpha) * math.log(q)
    log_sum = float(np.logaddexp(log_t1, log_t2))

    return max(log_sum / (alpha - 1.0), 0.0)


def geometric_rdp(
    alpha: float,
    epsilon: float,
) -> float:
    """RDP for the (two-sided) geometric mechanism.

    The geometric mechanism is the discrete analogue of the Laplace
    mechanism.  For a counting query with sensitivity 1, it adds noise
    drawn from a two-sided geometric distribution with parameter
    ``p = 1 - exp(-ε₀)``.

    The Rényi divergence of order α between Geom(p, centre=0) and
    Geom(p, centre=1) is:

        D_α = (1/(α-1)) log(
            Σ_{z=-∞}^{∞} p_0(z)^α p_1(z)^{1-α}
        )

    where p_c(z) ∝ exp(-ε₀ |z - c|).  For the normalised two-sided
    geometric, this evaluates to:

        D_α = (α ε₀²) / 2  (for α close to 1, i.e., the KL limit)

    and for general α:

        D_α = (1/(α-1)) log(
            α/(2α-1) exp((α-1)ε₀) + (α-1)/(2α-1) exp(-(α-1)ε₀)
        )

    which matches the Laplace RDP formula (the geometric mechanism has
    the same Rényi divergence as Laplace for integer outputs).

    All computations use log-domain arithmetic.

    Args:
        alpha: Rényi divergence order (> 1).
        epsilon: Privacy parameter ε₀ > 0.

    Returns:
        RDP epsilon at order α.

    Raises:
        ConfigurationError: If parameters are invalid.
    """
    if alpha <= 1.0:
        raise ConfigurationError(
            f"alpha must be > 1, got {alpha}",
            parameter="alpha", value=alpha,
        )
    if epsilon <= 0:
        raise ConfigurationError(
            f"epsilon must be > 0, got {epsilon}",
            parameter="epsilon", value=epsilon,
        )

    a_m1 = alpha - 1.0
    denom = 2.0 * alpha - 1.0
    if denom <= 0:
        return 0.0

    log_t1 = math.log(alpha / denom) + a_m1 * epsilon
    log_t2 = math.log(a_m1 / denom) - a_m1 * epsilon
    log_sum = float(np.logaddexp(log_t1, log_t2))

    return max(log_sum / a_m1, 0.0)


def truncated_laplace_rdp(
    alpha: float,
    epsilon: float,
    truncation: float,
    sensitivity: float = 1.0,
    n_terms: int = 1000,
) -> float:
    """RDP for the truncated Laplace mechanism.

    The truncated Laplace mechanism draws noise from a Laplace
    distribution with scale ``b = Δ/ε₀`` and truncates to
    ``[-truncation, truncation]``.  The truncation increases the
    probability mass on the boundary, breaking the standard Laplace
    RDP formula.

    This function computes the Rényi divergence numerically by
    discretising the truncated Laplace PDF and computing:

        D_α(P || Q) = (1/(α-1)) log ∫ p(x)^α q(x)^{1-α} dx

    where ``P`` and ``Q`` are truncated Laplace distributions centred
    at 0 and Δ respectively.

    The integration is performed via numerical quadrature on a fine
    grid of ``n_terms`` points over ``[-truncation, truncation]``.

    Args:
        alpha: Rényi divergence order (> 1).
        epsilon: Laplace privacy parameter ε₀ > 0 (scale = Δ/ε₀).
        truncation: Truncation bound B > 0.
        sensitivity: Query L1 sensitivity Δ > 0.
        n_terms: Number of quadrature points.

    Returns:
        RDP epsilon at order α (numerically computed).

    Raises:
        ConfigurationError: If parameters are invalid.
    """
    if alpha <= 1.0:
        raise ConfigurationError(
            f"alpha must be > 1, got {alpha}",
            parameter="alpha", value=alpha,
        )
    if epsilon <= 0:
        raise ConfigurationError(
            f"epsilon must be > 0, got {epsilon}",
            parameter="epsilon", value=epsilon,
        )
    if truncation <= 0:
        raise ConfigurationError(
            f"truncation must be > 0, got {truncation}",
            parameter="truncation", value=truncation,
        )
    if sensitivity <= 0:
        raise ConfigurationError(
            f"sensitivity must be > 0, got {sensitivity}",
            parameter="sensitivity", value=sensitivity,
        )

    b = sensitivity / epsilon  # Laplace scale

    # Discretise the interval [-truncation, truncation]
    x = np.linspace(-truncation, truncation, n_terms)
    dx = x[1] - x[0]

    # Unnormalised log-densities: Laplace centred at 0 and at Δ
    log_p = -np.abs(x) / b
    log_q = -np.abs(x - sensitivity) / b

    # Normalise in log domain (truncated distribution)
    log_Z_p = _logsumexp(log_p + math.log(dx))
    log_Z_q = _logsumexp(log_q + math.log(dx))

    log_p_norm = log_p - log_Z_p
    log_q_norm = log_q - log_Z_q

    # Rényi divergence integrand: α log p(x) + (1-α) log q(x) + log dx
    integrand = alpha * log_p_norm + (1.0 - alpha) * log_q_norm + math.log(dx)
    log_integral = _logsumexp(integrand)

    rdp_eps = log_integral / (alpha - 1.0)
    return max(rdp_eps, 0.0)

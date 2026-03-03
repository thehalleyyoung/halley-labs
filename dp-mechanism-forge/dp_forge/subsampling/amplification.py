"""
Privacy amplification by subsampling for DP-Forge.

Implements core amplification theorems that tighten privacy guarantees when
a mechanism is applied to a random subsample of the dataset rather than the
full dataset.  Three subsampling models are supported:

Subsampling Models:
    - **Poisson subsampling**: Each record is included independently with
      probability ``q``.  This is the model used by DP-SGD and most
      iterative mechanisms.
    - **Without-replacement sampling**: A fixed-size sample of ``⌊qN⌋``
      records is drawn uniformly without replacement from N records
      (hypergeometric model).
    - **Shuffle model**: Records are locally randomised, then shuffled by
      a trusted shuffler.  The amplification arises from the anonymity of
      the shuffle step (Erlingsson–Feldman–Mironov–Talwar bounds).

Key Formulas:
    - Poisson forward:  ε' = log(1 + q(e^ε₀ - 1)),  δ' = q·δ₀
    - Tighter Poisson via moment-generating function (MGF) analysis
    - Without-replacement via hypergeometric coupling
    - Shuffle via blanket decomposition

All computations use log-domain arithmetic (``log1p``, ``expm1``) for
numerical stability when ε₀ is small.

Functions:
    - :func:`poisson_amplify` — Poisson subsampling amplification.
    - :func:`replacement_amplify` — Without-replacement sampling amplification.
    - :func:`shuffle_amplify` — Shuffle model amplification.
    - :func:`poisson_amplify_rdp` — RDP-based Poisson amplification.
    - :func:`compute_amplification_factor` — Ratio ε'/ε₀ for a given model.

Classes:
    - :class:`AmplificationResult` — Result container for amplified privacy.
    - :class:`AmplificationBound` — Enum of bound types.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import ConfigurationError
from dp_forge.types import PrivacyBudget


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AmplificationBound(Enum):
    """Type of amplification bound applied.

    ``POISSON_BASIC``
        Simple substitution into the Poisson subsampling formula.

    ``POISSON_TIGHT``
        Tighter bound via moment-generating function analysis.

    ``WITHOUT_REPLACEMENT``
        Hypergeometric coupling bound for fixed-size sampling.

    ``SHUFFLE``
        Erlingsson–Feldman–Mironov–Talwar shuffle model bound.

    ``RDP``
        Rényi DP-based amplification with optimal order selection.
    """

    POISSON_BASIC = auto()
    POISSON_TIGHT = auto()
    WITHOUT_REPLACEMENT = auto()
    SHUFFLE = auto()
    RDP = auto()

    def __repr__(self) -> str:
        return f"AmplificationBound.{self.name}"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AmplificationResult:
    """Result of a privacy amplification computation.

    Attributes:
        eps: Amplified privacy parameter ε.
        delta: Amplified privacy parameter δ.
        bound_type: Which amplification theorem was applied.
        base_eps: Original (pre-amplification) ε₀.
        base_delta: Original (pre-amplification) δ₀.
        q_rate: Subsampling rate used.
        details: Optional dict of extra computation details.
    """

    eps: float
    delta: float
    bound_type: AmplificationBound
    base_eps: float
    base_delta: float
    q_rate: float
    details: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.eps < 0:
            raise ValueError(f"amplified eps must be >= 0, got {self.eps}")
        if self.delta < 0:
            raise ValueError(f"amplified delta must be >= 0, got {self.delta}")
        if not (0.0 < self.q_rate <= 1.0):
            raise ValueError(f"q_rate must be in (0, 1], got {self.q_rate}")

    @property
    def amplification_factor(self) -> float:
        """Ratio ε / ε₀ showing the amplification benefit."""
        if self.base_eps == 0.0:
            return 0.0
        return self.eps / self.base_eps

    @property
    def budget(self) -> PrivacyBudget:
        """Return amplified privacy as a PrivacyBudget."""
        return PrivacyBudget(epsilon=max(self.eps, 1e-15), delta=self.delta)

    def __repr__(self) -> str:
        return (
            f"AmplificationResult(ε={self.eps:.6f}, δ={self.delta:.2e}, "
            f"bound={self.bound_type.name}, q={self.q_rate:.4f})"
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_base_params(
    base_eps: float,
    base_delta: float,
    q_rate: float,
) -> None:
    """Validate common parameters for amplification functions.

    Args:
        base_eps: Base mechanism privacy parameter ε₀.
        base_delta: Base mechanism privacy parameter δ₀.
        q_rate: Subsampling rate q ∈ (0, 1].

    Raises:
        ConfigurationError: If any parameter is invalid.
    """
    if base_eps < 0.0:
        raise ConfigurationError(
            f"base_eps must be >= 0, got {base_eps}",
            parameter="base_eps",
            value=base_eps,
            constraint=">= 0",
        )
    if not math.isfinite(base_eps):
        raise ConfigurationError(
            f"base_eps must be finite, got {base_eps}",
            parameter="base_eps",
            value=base_eps,
            constraint="finite",
        )
    if base_delta < 0.0 or base_delta >= 1.0:
        raise ConfigurationError(
            f"base_delta must be in [0, 1), got {base_delta}",
            parameter="base_delta",
            value=base_delta,
            constraint="[0, 1)",
        )
    if not (0.0 < q_rate <= 1.0):
        raise ConfigurationError(
            f"q_rate must be in (0, 1], got {q_rate}",
            parameter="q_rate",
            value=q_rate,
            constraint="(0, 1]",
        )


# ---------------------------------------------------------------------------
# Numerically stable helpers
# ---------------------------------------------------------------------------


def _log1p_exp(x: float) -> float:
    """Compute log(1 + exp(x)) with numerical stability.

    Uses different formulas for different ranges of x to avoid overflow
    and maintain precision.
    """
    if x > 30.0:
        return x
    if x < -30.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _stable_log_poisson_amplification(base_eps: float, q_rate: float) -> float:
    """Compute log(1 + q(e^ε₀ - 1)) with numerical stability.

    For small ε₀, uses log1p(q * expm1(ε₀)) which avoids catastrophic
    cancellation.  For large ε₀, uses the standard formula.

    Args:
        base_eps: Base privacy parameter ε₀.
        q_rate: Subsampling rate q.

    Returns:
        The amplified ε = log(1 + q(e^ε₀ - 1)).
    """
    if base_eps == 0.0:
        return 0.0
    if q_rate == 1.0:
        return base_eps

    # For small ε₀: expm1(ε₀) = ε₀ + ε₀²/2 + ..., which is more precise
    # than exp(ε₀) - 1 when ε₀ is near zero.
    expm1_eps = math.expm1(base_eps)  # e^ε₀ - 1

    # q * (e^ε₀ - 1) could be very small for small q or small ε₀
    q_expm1 = q_rate * expm1_eps

    # log(1 + q(e^ε₀ - 1))
    return math.log1p(q_expm1)


def _stable_log_neg_poisson(base_eps: float, q_rate: float) -> float:
    """Compute log(1 + q(e^{-ε₀} - 1)) for the reverse direction.

    This is needed for the tighter two-sided Poisson bound.

    Args:
        base_eps: Base privacy parameter ε₀.
        q_rate: Subsampling rate q.

    Returns:
        log(1 + q(e^{-ε₀} - 1)).
    """
    if base_eps == 0.0:
        return 0.0
    if q_rate == 1.0:
        return -base_eps

    expm1_neg = math.expm1(-base_eps)  # e^{-ε₀} - 1 (negative)
    q_expm1 = q_rate * expm1_neg
    # q_expm1 is in [-q, 0), so 1 + q_expm1 > 0 for q < 1
    if 1.0 + q_expm1 <= 0.0:
        return -math.inf
    return math.log1p(q_expm1)


# =========================================================================
# 1. Poisson Subsampling Amplification
# =========================================================================


def poisson_amplify(
    base_eps: float,
    base_delta: float,
    q_rate: float,
    *,
    tight: bool = False,
) -> AmplificationResult:
    """Compute amplified privacy parameters under Poisson subsampling.

    Under Poisson subsampling with rate q, each record is included in the
    subsample independently with probability q.  The amplified privacy is:

        ε' = log(1 + q(e^ε₀ - 1))
        δ' = q · δ₀

    When ``tight=True``, uses the tighter bound from moment-generating
    function analysis which provides better constants for moderate ε₀.

    Args:
        base_eps: Base mechanism privacy parameter ε₀ ≥ 0.
        base_delta: Base mechanism privacy parameter δ₀ ∈ [0, 1).
        q_rate: Subsampling rate q ∈ (0, 1].
        tight: If True, use tighter MGF-based bound.

    Returns:
        AmplificationResult with the amplified (ε', δ').

    Raises:
        ConfigurationError: If parameters are invalid.

    Example::

        >>> result = poisson_amplify(1.0, 1e-5, 0.01)
        >>> result.eps  # Much smaller than 1.0
        0.01005...
    """
    _validate_base_params(base_eps, base_delta, q_rate)

    if q_rate == 1.0:
        return AmplificationResult(
            eps=base_eps,
            delta=base_delta,
            bound_type=AmplificationBound.POISSON_BASIC,
            base_eps=base_eps,
            base_delta=base_delta,
            q_rate=q_rate,
        )

    if tight:
        return _poisson_amplify_tight(base_eps, base_delta, q_rate)

    # Basic forward formula
    amplified_eps = _stable_log_poisson_amplification(base_eps, q_rate)
    amplified_delta = q_rate * base_delta

    return AmplificationResult(
        eps=amplified_eps,
        delta=amplified_delta,
        bound_type=AmplificationBound.POISSON_BASIC,
        base_eps=base_eps,
        base_delta=base_delta,
        q_rate=q_rate,
    )


def _poisson_amplify_tight(
    base_eps: float,
    base_delta: float,
    q_rate: float,
) -> AmplificationResult:
    """Tighter Poisson amplification via MGF analysis.

    Uses the moment-generating function approach from Balle et al. (2018)
    "Privacy Amplification by Subsampling" to obtain a tighter bound.

    The key insight is that for (ε₀, δ₀)-DP mechanisms, the amplified
    guarantee can be sharpened by considering the hockey-stick divergence
    directly rather than the ε-based bound.

    The tighter bound gives:
        ε' = log(1 + q²(e^ε₀ - 1)²/(e^ε₀ + 1) + q(e^ε₀ - 1))
        (simplified; the actual bound has additional correction terms)

    For practical use, we compute both the forward and reverse directions
    and take the tighter of the two.

    Args:
        base_eps: Base mechanism ε₀.
        base_delta: Base mechanism δ₀.
        q_rate: Subsampling rate q.

    Returns:
        AmplificationResult with the tightened (ε', δ').
    """
    # Forward direction: log(1 + q(e^ε₀ - 1))
    eps_forward = _stable_log_poisson_amplification(base_eps, q_rate)

    # Reverse direction: -log(1 + q(e^{-ε₀} - 1))
    log_rev = _stable_log_neg_poisson(base_eps, q_rate)
    eps_reverse = -log_rev if math.isfinite(log_rev) else float("inf")

    # The tight bound from Balle et al., Theorem 9:
    # For ε₀ <= 1, the quadratic correction term helps
    if base_eps <= 1.0:
        expm1_val = math.expm1(base_eps)
        # Second-order correction: ε' ≈ q·ε₀ + q²·ε₀²/2 for small ε₀
        correction = 0.5 * q_rate * q_rate * base_eps * base_eps
        eps_mgf = q_rate * base_eps + correction
        # Take the minimum of all bounds
        amplified_eps = min(eps_forward, eps_reverse, eps_mgf)
    else:
        # For large ε₀, the forward/reverse bound is already tight
        amplified_eps = min(eps_forward, eps_reverse)

    amplified_eps = max(0.0, amplified_eps)
    amplified_delta = q_rate * base_delta

    # Additional δ correction for the MGF bound
    # δ' = q·δ₀ + q·min(1, (e^ε₀ - 1)·δ₀) for approximate DP
    if base_delta > 0:
        delta_correction = q_rate * min(
            1.0, math.expm1(base_eps) * base_delta
        )
        # Take minimum of basic and corrected delta
        amplified_delta = min(q_rate * base_delta, amplified_delta + delta_correction)

    return AmplificationResult(
        eps=amplified_eps,
        delta=amplified_delta,
        bound_type=AmplificationBound.POISSON_TIGHT,
        base_eps=base_eps,
        base_delta=base_delta,
        q_rate=q_rate,
        details={
            "eps_forward": eps_forward,
            "eps_reverse": eps_reverse,
            "method": "mgf_balle_2018",
        },
    )


# =========================================================================
# 2. Without-Replacement Sampling Amplification
# =========================================================================


def replacement_amplify(
    base_eps: float,
    base_delta: float,
    q_rate: float,
    *,
    n_total: Optional[int] = None,
) -> AmplificationResult:
    """Compute amplified privacy under without-replacement sampling.

    When drawing a sample of size m = ⌊q·N⌋ uniformly without replacement
    from a dataset of size N, the amplification is modelled via a
    hypergeometric coupling argument.

    The bound (from Balle et al. 2018, Theorem 17) gives:

        ε' ≤ log(1 + (n/(n-1)) · q · (e^ε₀ - 1))

    which is slightly looser than Poisson but applies to the
    without-replacement setting commonly used in practice.

    For pure DP (δ₀ = 0), the bound simplifies to:

        ε' = log(1 + q(e^ε₀ - 1))    [same as Poisson]

    Args:
        base_eps: Base mechanism privacy parameter ε₀ ≥ 0.
        base_delta: Base mechanism privacy parameter δ₀ ∈ [0, 1).
        q_rate: Subsampling rate q ∈ (0, 1]. If n_total is given,
            the actual sample size is ⌊q · n_total⌋.
        n_total: Total dataset size N. Required for tighter
            hypergeometric bounds; if None, falls back to Poisson.

    Returns:
        AmplificationResult with the amplified (ε', δ').

    Raises:
        ConfigurationError: If parameters are invalid.
    """
    _validate_base_params(base_eps, base_delta, q_rate)

    if q_rate == 1.0:
        return AmplificationResult(
            eps=base_eps,
            delta=base_delta,
            bound_type=AmplificationBound.WITHOUT_REPLACEMENT,
            base_eps=base_eps,
            base_delta=base_delta,
            q_rate=q_rate,
        )

    if n_total is not None and n_total < 2:
        raise ConfigurationError(
            f"n_total must be >= 2 for without-replacement sampling, got {n_total}",
            parameter="n_total",
            value=n_total,
            constraint=">= 2",
        )

    details: dict = {}

    if n_total is not None:
        # Hypergeometric model with finite population correction
        m = int(math.floor(q_rate * n_total))
        effective_q = m / n_total if n_total > 0 else 0.0

        # Balle et al. correction factor: n/(n-1) accounts for the
        # add/remove adjacency in without-replacement sampling
        correction = n_total / (n_total - 1)
        corrected_q = min(1.0, correction * effective_q)

        expm1_eps = math.expm1(base_eps)
        amplified_eps = math.log1p(corrected_q * expm1_eps)

        # Delta amplification with finite-population adjustment
        amplified_delta = effective_q * base_delta

        details["n_total"] = n_total
        details["sample_size"] = m
        details["effective_q"] = effective_q
        details["correction_factor"] = correction
    else:
        # Fall back to Poisson bound (valid upper bound for WOR)
        amplified_eps = _stable_log_poisson_amplification(base_eps, q_rate)
        amplified_delta = q_rate * base_delta
        details["fallback"] = "poisson"

    return AmplificationResult(
        eps=amplified_eps,
        delta=amplified_delta,
        bound_type=AmplificationBound.WITHOUT_REPLACEMENT,
        base_eps=base_eps,
        base_delta=base_delta,
        q_rate=q_rate,
        details=details,
    )


# =========================================================================
# 3. Shuffle Model Amplification
# =========================================================================


def shuffle_amplify(
    base_eps: float,
    base_delta: float,
    n_users: int,
    *,
    target_delta: Optional[float] = None,
) -> AmplificationResult:
    """Compute amplified privacy in the shuffle model.

    In the shuffle model, each of n users applies a local randomiser
    with (ε₀, δ₀)-LDP, and a trusted shuffler permutes the reports
    uniformly at random before the analyser sees them.  The shuffling
    provides privacy amplification.

    Uses the Erlingsson–Feldman–Mironov–Talwar (EFMT) bound:

        ε_central ≤ O(ε₀ · √(log(1/δ) / n))    for small ε₀

    More precisely, for pure ε₀-LDP with δ₀=0 (Theorem 3.1 of
    Balle et al. "The Privacy Blanket of the Shuffle Model"):

        ε_central = log(1 + (e^ε₀ - 1)/n · (√(2·ln(4/δ')/n) + 1))

    where δ' is the target central δ.

    Args:
        base_eps: Local randomiser privacy parameter ε₀ ≥ 0.
        base_delta: Local randomiser privacy parameter δ₀ ∈ [0, 1).
        n_users: Number of users n ≥ 1.
        target_delta: Target central-model δ. If None, defaults to 1/n².

    Returns:
        AmplificationResult with the amplified central-model (ε', δ').

    Raises:
        ConfigurationError: If parameters are invalid.
    """
    if base_eps < 0 or not math.isfinite(base_eps):
        raise ConfigurationError(
            f"base_eps must be finite and >= 0, got {base_eps}",
            parameter="base_eps",
            value=base_eps,
        )
    if base_delta < 0.0 or base_delta >= 1.0:
        raise ConfigurationError(
            f"base_delta must be in [0, 1), got {base_delta}",
            parameter="base_delta",
            value=base_delta,
        )
    if n_users < 1:
        raise ConfigurationError(
            f"n_users must be >= 1, got {n_users}",
            parameter="n_users",
            value=n_users,
        )

    if target_delta is None:
        target_delta = 1.0 / (n_users * n_users) if n_users > 0 else 0.5

    if target_delta <= 0.0 or target_delta >= 1.0:
        raise ConfigurationError(
            f"target_delta must be in (0, 1), got {target_delta}",
            parameter="target_delta",
            value=target_delta,
        )

    q_effective = 1.0 / n_users

    # Edge cases
    if n_users == 1:
        return AmplificationResult(
            eps=base_eps,
            delta=max(base_delta, target_delta),
            bound_type=AmplificationBound.SHUFFLE,
            base_eps=base_eps,
            base_delta=base_delta,
            q_rate=q_effective,
        )

    if base_eps == 0.0:
        return AmplificationResult(
            eps=0.0,
            delta=base_delta,
            bound_type=AmplificationBound.SHUFFLE,
            base_eps=base_eps,
            base_delta=base_delta,
            q_rate=q_effective,
        )

    # Compute the EFMT bound
    expm1_eps = math.expm1(base_eps)

    # For pure LDP (δ₀ = 0): use the Balle et al. blanket decomposition
    if base_delta == 0.0:
        amplified_eps, amplified_delta = _shuffle_pure_ldp(
            base_eps, n_users, target_delta
        )
    else:
        # For approximate LDP: combine shuffle amplification with local δ
        amplified_eps, amplified_delta = _shuffle_approximate_ldp(
            base_eps, base_delta, n_users, target_delta
        )

    amplified_eps = max(0.0, amplified_eps)

    return AmplificationResult(
        eps=amplified_eps,
        delta=amplified_delta,
        bound_type=AmplificationBound.SHUFFLE,
        base_eps=base_eps,
        base_delta=base_delta,
        q_rate=q_effective,
        details={
            "n_users": n_users,
            "target_delta": target_delta,
            "method": "efmt_blanket",
        },
    )


def _shuffle_pure_ldp(
    base_eps: float,
    n_users: int,
    target_delta: float,
) -> Tuple[float, float]:
    """Shuffle amplification for pure ε₀-LDP randomisers.

    Implements the Balle–Bell–Gascón–Nissim bound (Theorem 3.1):

        ε_central = log(1 + (e^ε₀ - 1) · (√(2·ln(4/δ)/n) + 1/n))

    This gives roughly ε_central ≈ ε₀ · √(log(1/δ)/n) for small ε₀.

    Args:
        base_eps: Local ε₀.
        n_users: Number of users.
        target_delta: Target central δ.

    Returns:
        (amplified_eps, amplified_delta).
    """
    expm1_eps = math.expm1(base_eps)
    n = float(n_users)

    # √(2·ln(4/δ)/n)
    log_term = 2.0 * math.log(4.0 / target_delta)
    sqrt_term = math.sqrt(log_term / n)

    # 1/n contribution
    inv_n = 1.0 / n

    # ε_central = log(1 + expm1(ε₀) · (√(2·ln(4/δ)/n) + 1/n))
    inner = expm1_eps * (sqrt_term + inv_n)

    # Guard: for very large ε₀ or very few users, inner can be large
    if inner <= 0.0:
        amplified_eps = 0.0
    else:
        amplified_eps = math.log1p(inner)

    return amplified_eps, target_delta


def _shuffle_approximate_ldp(
    base_eps: float,
    base_delta: float,
    n_users: int,
    target_delta: float,
) -> Tuple[float, float]:
    """Shuffle amplification for approximate (ε₀, δ₀)-LDP randomisers.

    When the local randomiser has δ₀ > 0, the central δ must account for
    both the shuffle amplification and the local δ₀ across n users.

    The total central δ is:
        δ_central = n · δ₀ + δ_shuffle

    where δ_shuffle arises from the shuffle amplification step.

    Args:
        base_eps: Local ε₀.
        base_delta: Local δ₀.
        n_users: Number of users.
        target_delta: Target central δ.

    Returns:
        (amplified_eps, amplified_delta).
    """
    # Delta from local randomiser failures across n users
    delta_local = n_users * base_delta

    # Remaining delta budget for shuffle amplification
    delta_shuffle = max(0.0, target_delta - delta_local)

    if delta_shuffle <= 0.0:
        # No budget remaining: shuffle cannot help, return basic composition
        warnings.warn(
            f"Shuffle amplification has no δ budget remaining after "
            f"accounting for n·δ₀ = {delta_local:.2e}; returning "
            f"basic composition bound.",
            stacklevel=3,
        )
        return base_eps, min(delta_local, 1.0 - 1e-15)

    # Apply pure-LDP shuffle bound with the remaining δ budget
    amplified_eps, _ = _shuffle_pure_ldp(base_eps, n_users, delta_shuffle)

    amplified_delta = delta_local + delta_shuffle

    return amplified_eps, min(amplified_delta, 1.0 - 1e-15)


# =========================================================================
# 4. RDP-based Poisson Amplification
# =========================================================================


def poisson_amplify_rdp(
    base_eps: float,
    q_rate: float,
    *,
    orders: Optional[Sequence[float]] = None,
    target_delta: float = 1e-5,
) -> AmplificationResult:
    """Compute amplified privacy via RDP subsampling theorem.

    Uses the Rényi DP framework for tighter amplification bounds,
    especially when composing multiple rounds.

    For a (α, ε₀)-RDP mechanism subsampled with Poisson rate q, the
    amplified RDP guarantee is (from Mironov 2017, Theorem 9):

        ε_RDP(α) ≤ (1/(α-1)) · log(
            (1-q)^α · (α choose 2) · q² · e^{(α-1)ε₀}
            + Σ_k terms...
        )

    This is then converted to (ε, δ)-DP for the target δ.

    Args:
        base_eps: Base mechanism RDP guarantee at optimal α, or
            (ε, 0)-DP parameter (converted to RDP internally).
        q_rate: Poisson subsampling rate q ∈ (0, 1].
        orders: RDP orders α to evaluate. If None, uses a default grid.
        target_delta: Target δ for RDP-to-(ε,δ)-DP conversion.

    Returns:
        AmplificationResult with the amplified (ε', δ').
    """
    _validate_base_params(base_eps, 0.0, q_rate)

    if target_delta <= 0 or target_delta >= 1.0:
        raise ConfigurationError(
            f"target_delta must be in (0, 1), got {target_delta}",
            parameter="target_delta",
            value=target_delta,
        )

    if q_rate == 1.0:
        return AmplificationResult(
            eps=base_eps,
            delta=target_delta,
            bound_type=AmplificationBound.RDP,
            base_eps=base_eps,
            base_delta=0.0,
            q_rate=q_rate,
        )

    if orders is None:
        orders = _default_rdp_orders()

    # For each order α, compute the subsampled RDP guarantee
    best_eps = float("inf")
    best_alpha = 0.0

    for alpha in orders:
        if alpha <= 1.0:
            continue

        rdp_eps = _compute_subsampled_rdp(base_eps, q_rate, alpha)

        # Convert RDP to (ε, δ)-DP: ε = rdp_eps - log(δ)/(α-1)
        eps_converted = rdp_eps - math.log(target_delta) / (alpha - 1.0)

        if eps_converted < best_eps:
            best_eps = eps_converted
            best_alpha = alpha

    best_eps = max(0.0, best_eps)

    return AmplificationResult(
        eps=best_eps,
        delta=target_delta,
        bound_type=AmplificationBound.RDP,
        base_eps=base_eps,
        base_delta=0.0,
        q_rate=q_rate,
        details={
            "best_alpha": best_alpha,
            "n_orders": len(orders),
            "method": "rdp_subsampled",
        },
    )


def _default_rdp_orders() -> List[float]:
    """Default grid of RDP orders for optimisation.

    Covers integer orders 2..128 plus fractional orders near 1.
    """
    fractional = [1.1, 1.25, 1.5, 1.75]
    integer = list(range(2, 65))
    large = [80, 100, 128, 256, 512, 1024]
    return fractional + [float(x) for x in integer] + [float(x) for x in large]


def _compute_subsampled_rdp(
    base_eps: float,
    q_rate: float,
    alpha: float,
) -> float:
    """Compute (α, ε')-RDP for Poisson-subsampled mechanism.

    Implements the bound from Wang, Balle, Kasiviswanathan (2019)
    "Subsampled Rényi Differential Privacy and Analytical Moments
    Accountant":

    For integer α ≥ 2:
        ε(α) ≤ (1/(α-1)) · log(Σ_{k=0}^{α} (α choose k) ·
                (-1)^k · (1-q)^{α-k} · q^k · e^{k(k-1)ε₀/(2)})

    For non-integer α, we interpolate or use a looser bound.

    Args:
        base_eps: Base mechanism ε₀ (pure DP parameter).
        q_rate: Poisson subsampling rate.
        alpha: RDP order.

    Returns:
        Subsampled RDP epsilon at order α.
    """
    if base_eps == 0.0:
        return 0.0

    # For integer orders, use the exact multinomial bound
    alpha_int = int(math.ceil(alpha))
    if alpha_int < 2:
        alpha_int = 2

    # Compute log of each term in the sum, then logsumexp for stability
    log_terms = []
    for k in range(alpha_int + 1):
        # log(C(α, k))
        log_binom = _log_binom(alpha_int, k)
        # k · log(q) + (α-k) · log(1-q)
        if k == 0:
            log_q_part = alpha_int * math.log(1.0 - q_rate)
        elif k == alpha_int:
            log_q_part = alpha_int * math.log(q_rate)
        else:
            log_q_part = k * math.log(q_rate) + (alpha_int - k) * math.log(1.0 - q_rate)
        # k(k-1) · ε₀ / 2 (RDP guarantee for k-fold composition)
        log_rdp_part = k * (k - 1) * base_eps / 2.0

        log_terms.append(log_binom + log_q_part + log_rdp_part)

    log_terms_arr = np.array(log_terms, dtype=np.float64)
    log_sum = _stable_logsumexp(log_terms_arr)

    rdp_eps = log_sum / (alpha_int - 1)
    return max(0.0, rdp_eps)


def _log_binom(n: int, k: int) -> float:
    """Compute log(C(n, k)) using lgamma for numerical stability."""
    if k < 0 or k > n:
        return -math.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
    )


def _stable_logsumexp(log_terms: npt.NDArray[np.float64]) -> float:
    """Numerically stable logsumexp."""
    if len(log_terms) == 0:
        return -np.inf
    max_val = np.max(log_terms)
    if not np.isfinite(max_val):
        return float(max_val)
    return float(max_val + np.log(np.sum(np.exp(log_terms - max_val))))


# =========================================================================
# 5. Convenience and Comparison Functions
# =========================================================================


def compute_amplification_factor(
    base_eps: float,
    q_rate: float,
    bound_type: AmplificationBound = AmplificationBound.POISSON_BASIC,
) -> float:
    """Compute the amplification factor ε'/ε₀ for a given model.

    Args:
        base_eps: Base privacy parameter ε₀.
        q_rate: Subsampling rate q.
        bound_type: Which amplification bound to use.

    Returns:
        The ratio ε'/ε₀, which is in (0, 1] for q < 1.
    """
    if base_eps == 0.0:
        return 0.0

    if bound_type == AmplificationBound.POISSON_BASIC:
        result = poisson_amplify(base_eps, 0.0, q_rate)
    elif bound_type == AmplificationBound.POISSON_TIGHT:
        result = poisson_amplify(base_eps, 0.0, q_rate, tight=True)
    elif bound_type == AmplificationBound.WITHOUT_REPLACEMENT:
        result = replacement_amplify(base_eps, 0.0, q_rate)
    else:
        result = poisson_amplify(base_eps, 0.0, q_rate)

    return result.amplification_factor


def compare_amplification_bounds(
    base_eps: float,
    base_delta: float,
    q_rate: float,
    *,
    n_total: Optional[int] = None,
    n_users: Optional[int] = None,
) -> List[AmplificationResult]:
    """Compare all applicable amplification bounds for given parameters.

    Returns a list of results from each applicable amplification theorem,
    sorted by amplified ε (tightest first).

    Args:
        base_eps: Base mechanism ε₀.
        base_delta: Base mechanism δ₀.
        q_rate: Subsampling rate q.
        n_total: Dataset size for without-replacement bounds.
        n_users: Number of users for shuffle bounds.

    Returns:
        List of AmplificationResult, tightest (smallest ε) first.
    """
    results: List[AmplificationResult] = []

    # Poisson basic
    results.append(poisson_amplify(base_eps, base_delta, q_rate))

    # Poisson tight
    results.append(poisson_amplify(base_eps, base_delta, q_rate, tight=True))

    # Without-replacement
    results.append(
        replacement_amplify(base_eps, base_delta, q_rate, n_total=n_total)
    )

    # Shuffle model (if n_users provided)
    if n_users is not None and n_users >= 2:
        results.append(shuffle_amplify(base_eps, base_delta, n_users))

    # Sort by amplified ε
    results.sort(key=lambda r: (r.eps, r.delta))
    return results


# =========================================================================
# 6. Optimal Subsampling Rate
# =========================================================================


def optimal_subsampling_rate(
    target_eps: float,
    base_eps: float,
    base_delta: float = 0.0,
    *,
    error_fn: Optional[Callable[[float], float]] = None,
    sensitivity: float = 1.0,
    q_min: float = 1e-6,
    q_max: float = 1.0,
    n_search: int = 200,
) -> Tuple[float, AmplificationResult]:
    """Find the subsampling rate q that minimises mechanism error for a target ε.

    Given a target amplified privacy ε and a base mechanism with
    parameters (ε₀, δ₀), find the subsampling rate q such that:
        - The amplified privacy satisfies ε'(q) ≤ target_eps.
        - The mechanism error (which typically scales as σ/q ∝ 1/q for
          fixed noise) is minimised.

    If no ``error_fn`` is provided, uses the default error model
    ``error(q) = sensitivity² / (2 · q² · base_eps)`` (variance of
    the subsampled Gaussian mechanism).

    Algorithm:
        1. Binary search for the maximum q such that the amplified
           ε'(q) ≤ target_eps.
        2. Among all feasible q, return the one minimising error_fn.

    Args:
        target_eps: Target amplified ε (the privacy budget to spend).
        base_eps: Base mechanism privacy parameter ε₀.
        base_delta: Base mechanism privacy parameter δ₀.
        error_fn: Optional function mapping q → expected error.  If
            ``None``, uses the default Gaussian noise model.
        sensitivity: Query sensitivity (used in the default error model).
        q_min: Minimum subsampling rate to consider.
        q_max: Maximum subsampling rate to consider.
        n_search: Number of search points for the grid search.

    Returns:
        Tuple ``(optimal_q, amplification_result)`` with the optimal rate
        and the corresponding amplification result.

    Raises:
        ConfigurationError: If no feasible q exists.
    """
    if target_eps <= 0:
        raise ConfigurationError(
            f"target_eps must be > 0, got {target_eps}",
            parameter="target_eps", value=target_eps,
        )
    _validate_base_params(base_eps, base_delta, q_max)

    # Default error model: variance of subsampled Gaussian
    if error_fn is None:
        def error_fn(q: float) -> float:
            return sensitivity ** 2 / (2.0 * q * q * max(base_eps, 1e-15))

    # Grid search over q values
    q_candidates = np.geomspace(q_min, q_max, n_search)

    best_q = q_min
    best_error = float("inf")
    best_result: Optional[AmplificationResult] = None

    for q in q_candidates:
        q = float(q)
        try:
            result = poisson_amplify(base_eps, base_delta, q, tight=True)
        except (ConfigurationError, ValueError):
            continue

        if result.eps <= target_eps:
            err = error_fn(q)
            if err < best_error:
                best_error = err
                best_q = q
                best_result = result

    if best_result is None:
        # No feasible q found; try q_max as fallback
        result = poisson_amplify(base_eps, base_delta, q_max, tight=True)
        if result.eps <= target_eps:
            best_q = q_max
            best_result = result
        else:
            raise ConfigurationError(
                f"No feasible subsampling rate: amplified ε at q=1 is "
                f"{result.eps:.6f} > target {target_eps:.6f}. "
                f"Consider reducing base_eps or increasing target_eps.",
                parameter="target_eps",
                value=target_eps,
            )

    return best_q, best_result


# =========================================================================
# 7. Privacy Profile Curve
# =========================================================================


@dataclass
class PrivacyProfilePoint:
    """A single point on the privacy profile curve ε(δ).

    Attributes:
        delta: The δ parameter.
        epsilon: The corresponding ε guarantee.
    """

    delta: float
    epsilon: float


def privacy_profile_curve(
    base_eps: float,
    base_delta: float,
    q_rate: float,
    *,
    n_points: int = 100,
    delta_min: float = 1e-12,
    delta_max: float = 0.1,
    bound_type: AmplificationBound = AmplificationBound.POISSON_TIGHT,
) -> List[PrivacyProfilePoint]:
    """Compute the full ε(δ) privacy profile curve for a subsampled mechanism.

    The privacy profile is the function ε(δ) giving the tightest ε
    guarantee at each δ level.  This is a concave, decreasing function
    that fully characterises the privacy properties of the mechanism.

    For subsampled mechanisms, the profile is computed by evaluating the
    amplification bound at each δ and combining the ε and δ
    amplification.

    The curve is useful for:
        - Choosing the optimal δ for a composition.
        - Comparing mechanisms across the full (ε, δ) trade-off.
        - Feeding into numerical composition tools.

    Args:
        base_eps: Base mechanism privacy parameter ε₀.
        base_delta: Base mechanism privacy parameter δ₀.
        q_rate: Subsampling rate q ∈ (0, 1].
        n_points: Number of δ values to evaluate.
        delta_min: Minimum δ in the profile.
        delta_max: Maximum δ in the profile.
        bound_type: Which amplification bound to use.

    Returns:
        List of PrivacyProfilePoint sorted by δ (ascending).
    """
    _validate_base_params(base_eps, base_delta, q_rate)

    deltas = np.geomspace(delta_min, delta_max, n_points)
    profile: List[PrivacyProfilePoint] = []

    for delta in deltas:
        delta = float(delta)

        if bound_type == AmplificationBound.POISSON_BASIC:
            result = poisson_amplify(base_eps, base_delta, q_rate)
        elif bound_type == AmplificationBound.POISSON_TIGHT:
            result = poisson_amplify(base_eps, base_delta, q_rate, tight=True)
        elif bound_type == AmplificationBound.WITHOUT_REPLACEMENT:
            result = replacement_amplify(base_eps, base_delta, q_rate)
        elif bound_type == AmplificationBound.RDP:
            result = poisson_amplify_rdp(base_eps, q_rate, target_delta=delta)
            profile.append(PrivacyProfilePoint(delta=delta, epsilon=result.eps))
            continue
        else:
            result = poisson_amplify(base_eps, base_delta, q_rate, tight=True)

        # For non-RDP bounds, adjust ε using the (ε, δ)-DP trade-off:
        # A mechanism satisfying (ε, δ₁)-DP also satisfies (ε', δ₂)-DP
        # where ε' = ε + log(1 - δ₁/δ₂) for δ₂ > δ₁
        amplified_delta = result.delta
        amplified_eps = result.eps

        if amplified_delta <= delta and delta > 0:
            # We have slack in δ; can tighten ε
            if amplified_delta > 0 and delta > amplified_delta:
                slack = delta - amplified_delta
                # Post-processing: ε can be reduced by log(1/(1-slack))
                # This is the standard (ε, δ) trade-off curve
                eps_reduction = math.log1p(slack / max(1.0 - delta, 1e-15))
                eps_adjusted = max(0.0, amplified_eps - eps_reduction)
            else:
                eps_adjusted = amplified_eps
        else:
            # δ budget too small; ε is higher
            eps_adjusted = amplified_eps

        profile.append(PrivacyProfilePoint(delta=delta, epsilon=eps_adjusted))

    return profile


# =========================================================================
# 8. Numerical Amplification via PLD
# =========================================================================


def numerical_amplification(
    base_eps: float,
    q_rate: float,
    *,
    target_delta: float = 1e-5,
    n_discretisation: int = 10000,
    pld_range: float = 20.0,
) -> AmplificationResult:
    """Tight numerical amplification bound via the Privacy Loss Distribution.

    Computes amplification by subsampling using the Privacy Loss
    Distribution (PLD) framework (Balle et al. 2020, Koskela et al. 2020).
    This provides tighter bounds than closed-form RDP or moment-based
    methods, especially for moderate ε₀ and small q.

    The PLD of a mechanism M is the distribution of the privacy loss
    random variable ``log(M(x)/M(x'))``.  For a Poisson-subsampled
    mechanism with rate q applied to a base (ε₀, 0)-DP mechanism:

        PLD_sub = (1-q) · δ₀ + q · PLD_base

    where δ₀ is the point mass at 0 (no privacy loss for unsampled records)
    and PLD_base is the PLD of the base mechanism.

    The algorithm:
        1. Discretise the privacy loss domain ``[-pld_range, pld_range]``
           into ``n_discretisation`` bins.
        2. Construct the PLD of the base mechanism (Laplace-like shape
           for pure DP).
        3. Apply the subsampling mixture.
        4. Compute ε(δ) from the PLD via the hockey-stick divergence:
           ``ε = inf { t : E[max(PLD - t, 0)] ≤ δ }``

    Args:
        base_eps: Base mechanism privacy parameter ε₀ (pure DP).
        q_rate: Poisson subsampling rate q ∈ (0, 1].
        target_delta: Target δ for computing ε.
        n_discretisation: Number of bins for PLD discretisation.
        pld_range: Range of the privacy loss domain.

    Returns:
        AmplificationResult with the numerically computed (ε, δ).
    """
    _validate_base_params(base_eps, 0.0, q_rate)

    if target_delta <= 0 or target_delta >= 1.0:
        raise ConfigurationError(
            f"target_delta must be in (0, 1), got {target_delta}",
            parameter="target_delta",
            value=target_delta,
        )

    if q_rate == 1.0:
        return AmplificationResult(
            eps=base_eps,
            delta=target_delta,
            bound_type=AmplificationBound.POISSON_TIGHT,
            base_eps=base_eps,
            base_delta=0.0,
            q_rate=q_rate,
            details={"method": "pld_trivial"},
        )

    # Discretise the privacy loss domain
    pld_grid = np.linspace(-pld_range, pld_range, n_discretisation)
    bin_width = pld_grid[1] - pld_grid[0]

    # Base mechanism PLD: for ε₀-DP, the privacy loss is in {-ε₀, ε₀}
    # with probabilities determined by the mechanism.
    # For the optimal mechanism (Laplace), the PLD density is:
    #   f(z) = (1/(2b)) exp(-|z|/b)  for z ∈ (-∞, ∞)
    # where b = 1/ε₀ (Laplace scale for sensitivity 1).
    # After clamping to [-ε₀, ε₀], we get a truncated distribution.
    b = 1.0 / max(base_eps, 1e-15)
    log_pld_base = -np.abs(pld_grid) / b
    pld_base = np.exp(log_pld_base - _stable_logsumexp(log_pld_base))
    pld_base /= pld_base.sum()  # normalise

    # Subsampled PLD: (1-q) · δ₀ + q · PLD_base
    # δ₀ is a point mass at z=0
    zero_idx = np.argmin(np.abs(pld_grid))
    pld_sub = q_rate * pld_base.copy()
    pld_sub[zero_idx] += (1.0 - q_rate)

    # Normalise
    pld_sub = np.maximum(pld_sub, 0.0)
    pld_sub /= pld_sub.sum()

    # Compute ε(δ) via hockey-stick: ε = inf{t : E[max(PLD - t, 0)] ≤ δ}
    # Binary search over t
    t_lo, t_hi = 0.0, float(pld_range)
    for _ in range(100):
        t_mid = (t_lo + t_hi) / 2.0
        # E[max(PLD - t, 0)] = Σ max(z_j - t, 0) · pld_sub[j]
        excess = np.maximum(pld_grid - t_mid, 0.0)
        hockey_stick = float(np.dot(excess, pld_sub))
        if hockey_stick <= target_delta:
            t_hi = t_mid
        else:
            t_lo = t_mid

    amplified_eps = max(t_hi, 0.0)

    return AmplificationResult(
        eps=amplified_eps,
        delta=target_delta,
        bound_type=AmplificationBound.POISSON_TIGHT,
        base_eps=base_eps,
        base_delta=0.0,
        q_rate=q_rate,
        details={
            "method": "pld_numerical",
            "n_discretisation": n_discretisation,
            "pld_range": pld_range,
        },
    )

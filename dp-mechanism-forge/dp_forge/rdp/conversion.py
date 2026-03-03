"""
Privacy framework conversions: RDP ↔ (ε,δ)-DP ↔ zCDP.

Implements conversions between major privacy accounting frameworks:

- **RDP → (ε,δ)-DP**: Optimal conversion via Balle et al. (2020) that
  improves on the standard Mironov (2017) bound.
- **(ε,δ)-DP → RDP**: Upper bounds on RDP from (ε,δ)-DP guarantees.
- **zCDP ↔ RDP**: Equivalence between zero-concentrated DP and RDP.
- **Concentrated DP conversions**: Between CDP and other frameworks.

All conversions maintain valid privacy guarantees (never underestimate
the privacy loss).

References:
    - Mironov, I. (2017). Rényi differential privacy.
    - Bun, M. & Steinke, T. (2016). Concentrated differential privacy.
    - Balle, B., Gaboardi, M., & Zanella-Béguelin, B. (2020).
      Privacy profiles and amplification by subsampling.
    - Canonne, C.L., Kamath, G., & Steinke, T. (2020).
      The discrete Gaussian for differential privacy.
"""

from __future__ import annotations

import math
import warnings
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import ConfigurationError
from dp_forge.types import PrivacyBudget

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Default alpha grid (shared with accountant)
# ---------------------------------------------------------------------------

DEFAULT_ALPHAS: FloatArray = np.concatenate([
    np.arange(1.25, 10.0, 0.25),
    np.arange(10.0, 65.0, 2.0),
    np.array([128.0, 256.0, 512.0, 1024.0]),
])


# =========================================================================
# RDP → (ε, δ)-DP conversion
# =========================================================================


def rdp_to_dp(
    rdp_epsilons: FloatArray,
    alphas: FloatArray,
    delta: float,
    *,
    method: str = "balle2020",
) -> Tuple[float, float]:
    """Convert an RDP curve to (ε, δ)-DP with optimal α selection.

    Supports two conversion methods:

    - ``"standard"`` (Mironov 2017):
      ε = min_α { ε̂(α) + log(1/δ) / (α - 1) }

    - ``"balle2020"`` (Balle, Gaboardi, Zanella-Béguelin 2020):
      ε = min_α { ε̂(α) + log(1/δ - exp(-ε̂(α)(α-1))) / (α - 1) }
      This gives a tighter bound, especially for moderate α.

    Args:
        rdp_epsilons: RDP ε values at each α, shape ``(m,)``.
        alphas: Corresponding α values, shape ``(m,)``.
        delta: Target δ ∈ (0, 1).
        method: Conversion method, ``"standard"`` or ``"balle2020"``.

    Returns:
        Tuple of (ε, optimal_α).

    Raises:
        ValueError: If delta is not in (0, 1).
        ConfigurationError: If method is unknown.
    """
    if not (0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    rdp_epsilons = np.asarray(rdp_epsilons, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)

    if len(rdp_epsilons) != len(alphas):
        raise ValueError(
            f"rdp_epsilons ({len(rdp_epsilons)}) and alphas ({len(alphas)}) "
            f"must have equal length"
        )

    if method == "standard":
        return _rdp_to_dp_standard(rdp_epsilons, alphas, delta)
    elif method == "balle2020":
        return _rdp_to_dp_balle2020(rdp_epsilons, alphas, delta)
    else:
        raise ConfigurationError(
            f"Unknown conversion method: {method!r}. "
            f"Supported: 'standard', 'balle2020'.",
            parameter="method",
            value=method,
        )


def _rdp_to_dp_standard(
    rdp_epsilons: FloatArray,
    alphas: FloatArray,
    delta: float,
) -> Tuple[float, float]:
    """Standard Mironov (2017) RDP → (ε,δ)-DP conversion.

    ε = min_α { ε̂(α) + log(1/δ) / (α - 1) }
    """
    log_delta = math.log(delta)
    eps_candidates = rdp_epsilons - log_delta / (alphas - 1.0)
    best_idx = int(np.argmin(eps_candidates))
    return max(float(eps_candidates[best_idx]), 0.0), float(alphas[best_idx])


def _rdp_to_dp_balle2020(
    rdp_epsilons: FloatArray,
    alphas: FloatArray,
    delta: float,
) -> Tuple[float, float]:
    """Balle et al. (2020) improved RDP → (ε,δ)-DP conversion.

    ε = min_α { ε̂(α) - (log(δ) + log(α-1))/α - log(1 - 1/α) }

    This bound is derived from the privacy profile approach and is
    always at least as tight as the standard bound.
    """
    best_eps = float("inf")
    best_alpha = float(alphas[0])

    for i in range(len(alphas)):
        alpha = float(alphas[i])
        rdp_eps = float(rdp_epsilons[i])

        if alpha <= 1.0:
            continue

        # Balle et al. (2020) Proposition 3:
        # ε ≤ ε̂(α) - (log(δ) + log(α-1)) / α - log(1 - 1/α)
        # = ε̂(α) + log(1 - 1/α) + (log(1/δ) - log(α-1)) / (α - 1) ... simplified:
        # More precisely:
        # ε = rdp_eps - (log(δ) + log(α - 1)) / α - log(1 - 1/α)
        log_a_minus_1 = math.log(alpha - 1.0)
        log_1_minus_inv = math.log(1.0 - 1.0 / alpha)

        eps = rdp_eps - (math.log(delta) + log_a_minus_1) / alpha - log_1_minus_inv

        # Also compute standard bound for comparison, take tighter
        eps_standard = rdp_eps - math.log(delta) / (alpha - 1.0)

        eps = min(eps, eps_standard)

        if eps < best_eps:
            best_eps = eps
            best_alpha = alpha

    return max(best_eps, 0.0), best_alpha


def rdp_to_dp_budget(
    rdp_epsilons: FloatArray,
    alphas: FloatArray,
    delta: float,
    *,
    method: str = "balle2020",
) -> PrivacyBudget:
    """Convert an RDP curve to a PrivacyBudget.

    Convenience wrapper around :func:`rdp_to_dp` that returns a
    :class:`PrivacyBudget` directly.

    Args:
        rdp_epsilons: RDP ε values at each α.
        alphas: Corresponding α values.
        delta: Target δ ∈ (0, 1).
        method: Conversion method.

    Returns:
        PrivacyBudget with the tightest ε.
    """
    eps, _ = rdp_to_dp(rdp_epsilons, alphas, delta, method=method)
    return PrivacyBudget(epsilon=max(eps, 1e-15), delta=delta)


# =========================================================================
# (ε, δ)-DP → RDP upper bounds
# =========================================================================


def dp_to_rdp_bound(
    epsilon: float,
    delta: float,
    alphas: Optional[FloatArray] = None,
) -> Tuple[FloatArray, FloatArray]:
    """Compute upper bounds on RDP from an (ε, δ)-DP guarantee.

    If a mechanism satisfies (ε, δ)-DP, it also satisfies (α, ε̂(α))-RDP
    where ε̂(α) is upper bounded by:

    For pure DP (δ = 0):
        ε̂(α) ≤ ε  for all α ≥ 1

    For approximate DP (δ > 0):
        ε̂(α) ≤ min(ε, (log(1/δ) + (α-1)ε) / α)
        and also ε̂(α) ≤ ε + log(1 + (exp(ε) - 1)/δ) / (α - 1)

    Args:
        epsilon: Privacy parameter ε > 0.
        delta: Privacy parameter δ ∈ [0, 1).
        alphas: α grid. Defaults to :data:`DEFAULT_ALPHAS`.

    Returns:
        Tuple of (alphas, rdp_upper_bounds).

    Raises:
        ValueError: If parameters are invalid.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if not (0 <= delta < 1):
        raise ValueError(f"delta must be in [0, 1), got {delta}")

    if alphas is None:
        alphas = DEFAULT_ALPHAS.copy()
    else:
        alphas = np.asarray(alphas, dtype=np.float64)

    rdp_bounds = np.empty_like(alphas)

    for i, alpha in enumerate(alphas):
        if abs(alpha - 1.0) < 1e-10:
            # KL divergence bound
            if delta == 0:
                rdp_bounds[i] = epsilon
            else:
                rdp_bounds[i] = epsilon + delta * epsilon
            continue

        if delta == 0:
            # Pure DP: ε̂(α) ≤ ε for all α
            rdp_bounds[i] = epsilon
        else:
            # Bound 1: from definition inversion
            bound1 = epsilon + math.log1p((math.exp(epsilon) - 1.0) / max(delta, 1e-300)) / (alpha - 1.0)

            # Bound 2: simple bound
            bound2 = (math.log(1.0 / delta) + (alpha - 1.0) * epsilon) / alpha

            # Bound 3: trivial
            bound3 = epsilon

            rdp_bounds[i] = min(bound1, bound2, bound3)

    return alphas, rdp_bounds


# =========================================================================
# zCDP ↔ RDP conversions
# =========================================================================


def zcdp_to_rdp(
    rho: float,
    alphas: Optional[FloatArray] = None,
) -> Tuple[FloatArray, FloatArray]:
    """Convert ρ-zCDP to RDP.

    A mechanism satisfying ρ-zCDP also satisfies (α, ρ·α)-RDP for all
    α > 1 (Bun & Steinke 2016, Proposition 1.3).

    The relationship is exact: ρ-zCDP ⟺ (α, ρα)-RDP for all α > 1.

    Args:
        rho: zCDP parameter ρ ≥ 0.
        alphas: α grid. Defaults to :data:`DEFAULT_ALPHAS`.

    Returns:
        Tuple of (alphas, rdp_epsilons).

    Raises:
        ValueError: If rho is negative.
    """
    if rho < 0:
        raise ValueError(f"rho must be >= 0, got {rho}")

    if alphas is None:
        alphas = DEFAULT_ALPHAS.copy()
    else:
        alphas = np.asarray(alphas, dtype=np.float64)

    rdp_epsilons = rho * alphas

    return alphas, rdp_epsilons


def rdp_to_zcdp(
    rdp_epsilons: FloatArray,
    alphas: FloatArray,
) -> float:
    """Convert an RDP curve to the tightest zCDP parameter ρ.

    Since ρ-zCDP ⟺ (α, ρα)-RDP, we have:
        ρ = max_α { ε̂(α) / α }

    This finds the tightest ρ consistent with the given RDP curve.

    Args:
        rdp_epsilons: RDP ε values at each α.
        alphas: Corresponding α values.

    Returns:
        Tightest ρ parameter.
    """
    rdp_epsilons = np.asarray(rdp_epsilons, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)

    if len(rdp_epsilons) != len(alphas):
        raise ValueError(
            f"rdp_epsilons ({len(rdp_epsilons)}) and alphas ({len(alphas)}) "
            f"must have equal length"
        )

    # ρ = max_α { ε̂(α) / α } (since ρ-zCDP requires ε̂(α) ≤ ρα for ALL α)
    rho_candidates = rdp_epsilons / alphas
    return float(np.max(rho_candidates))


def zcdp_to_dp(
    rho: float,
    delta: float,
) -> PrivacyBudget:
    """Convert ρ-zCDP to (ε, δ)-DP.

    From Bun & Steinke (2016), Proposition 1.3:
        ε = ρ + 2√(ρ log(1/δ))

    Args:
        rho: zCDP parameter ρ ≥ 0.
        delta: Target δ ∈ (0, 1).

    Returns:
        Privacy budget.

    Raises:
        ValueError: If parameters are invalid.
    """
    if rho < 0:
        raise ValueError(f"rho must be >= 0, got {rho}")
    if not (0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    eps = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
    return PrivacyBudget(epsilon=max(eps, 1e-15), delta=delta)


def dp_to_zcdp(
    epsilon: float,
    delta: float,
) -> float:
    """Upper bound on ρ-zCDP from (ε, δ)-DP.

    Inverts the zCDP → (ε,δ)-DP conversion to find the smallest ρ
    such that ρ-zCDP → (ε,δ)-DP. Solves:
        ε = ρ + 2√(ρ log(1/δ))
    for ρ using the quadratic formula.

    Let c = log(1/δ). Then ε = ρ + 2√(ρc), which gives:
        √ρ = -√c + √(c + ε)  →  ρ = (√(c + ε) - √c)²

    Args:
        epsilon: Privacy parameter ε > 0.
        delta: Privacy parameter δ ∈ (0, 1).

    Returns:
        Upper bound on ρ.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if not (0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    c = math.log(1.0 / delta)
    rho = (math.sqrt(c + epsilon) - math.sqrt(c)) ** 2

    return rho


# =========================================================================
# Gaussian mechanism characterisation in each framework
# =========================================================================


def gaussian_rdp(
    sigma: float,
    sensitivity: float,
    alphas: Optional[FloatArray] = None,
) -> Tuple[FloatArray, FloatArray]:
    """RDP of the Gaussian mechanism.

    The Gaussian mechanism with noise σ and sensitivity Δ satisfies
    (α, α·Δ²/(2σ²))-RDP for all α > 1.

    Args:
        sigma: Noise standard deviation (> 0).
        sensitivity: Query sensitivity Δ (> 0).
        alphas: α grid. Defaults to :data:`DEFAULT_ALPHAS`.

    Returns:
        Tuple of (alphas, rdp_epsilons).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be > 0, got {sensitivity}")

    if alphas is None:
        alphas = DEFAULT_ALPHAS.copy()
    else:
        alphas = np.asarray(alphas, dtype=np.float64)

    rdp_epsilons = alphas * sensitivity ** 2 / (2.0 * sigma ** 2)
    return alphas, rdp_epsilons


def gaussian_zcdp(sigma: float, sensitivity: float) -> float:
    """zCDP parameter for the Gaussian mechanism.

    ρ = Δ²/(2σ²)

    Args:
        sigma: Noise standard deviation.
        sensitivity: Query sensitivity.

    Returns:
        zCDP parameter ρ.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be > 0, got {sensitivity}")

    return sensitivity ** 2 / (2.0 * sigma ** 2)


def gaussian_dp(
    sigma: float,
    sensitivity: float,
    delta: float,
) -> PrivacyBudget:
    """(ε, δ)-DP of the Gaussian mechanism via RDP conversion.

    Args:
        sigma: Noise standard deviation.
        sensitivity: Query sensitivity.
        delta: Target δ.

    Returns:
        Privacy budget with optimal ε.
    """
    alphas, rdp_eps = gaussian_rdp(sigma, sensitivity)
    eps, _ = rdp_to_dp(rdp_eps, alphas, delta)
    return PrivacyBudget(epsilon=max(eps, 1e-15), delta=delta)


# =========================================================================
# Laplace mechanism conversions
# =========================================================================


def laplace_rdp(
    epsilon: float,
    alphas: Optional[FloatArray] = None,
) -> Tuple[FloatArray, FloatArray]:
    """RDP of the Laplace mechanism.

    The Laplace mechanism satisfying ε-DP has RDP guarantee:
        ε̂(α) = 1/(α-1) log( α/(2α-1) exp((α-1)ε) + (α-1)/(2α-1) exp(-(α-1)ε) )

    Args:
        epsilon: Laplace privacy parameter ε > 0.
        alphas: α grid. Defaults to :data:`DEFAULT_ALPHAS`.

    Returns:
        Tuple of (alphas, rdp_epsilons).
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    if alphas is None:
        alphas = DEFAULT_ALPHAS.copy()
    else:
        alphas = np.asarray(alphas, dtype=np.float64)

    rdp_eps = np.empty_like(alphas)

    for i, alpha in enumerate(alphas):
        if abs(alpha - 1.0) < 1e-10:
            rdp_eps[i] = 0.0
            continue

        a_minus_1 = alpha - 1.0
        denom = 2.0 * alpha - 1.0

        if denom <= 0:
            rdp_eps[i] = 0.0
            continue

        log_t1 = math.log(alpha / denom) + a_minus_1 * epsilon
        log_t2_coeff = a_minus_1 / denom
        if log_t2_coeff <= 0:
            log_t2 = -float("inf")
        else:
            log_t2 = math.log(log_t2_coeff) - a_minus_1 * epsilon

        log_sum = float(np.logaddexp(log_t1, log_t2))
        rdp_eps[i] = max(log_sum / a_minus_1, 0.0)

    return alphas, rdp_eps


def laplace_dp_to_zcdp(epsilon: float) -> float:
    """Upper bound on ρ-zCDP from ε-Laplace DP.

    Uses the RDP curve of the Laplace mechanism to find the tightest
    ρ consistent with zCDP.

    Args:
        epsilon: Laplace privacy parameter.

    Returns:
        Upper bound on ρ.
    """
    alphas, rdp_eps = laplace_rdp(epsilon)
    return rdp_to_zcdp(rdp_eps, alphas)


# =========================================================================
# Composition helpers using conversions
# =========================================================================


def compose_rdp_then_convert(
    rdp_curves: Sequence[Tuple[FloatArray, FloatArray]],
    delta: float,
    alphas: Optional[FloatArray] = None,
    *,
    method: str = "balle2020",
) -> PrivacyBudget:
    """Compose multiple RDP curves and convert to (ε, δ)-DP.

    Each RDP curve is a tuple ``(alphas, rdp_epsilons)``.
    Composition is via summation on a common α grid.

    Args:
        rdp_curves: Sequence of (alphas, rdp_epsilons) tuples.
        delta: Target δ.
        alphas: Common α grid. If ``None``, uses the union of all grids.
        method: RDP → DP conversion method.

    Returns:
        Privacy budget after composition.
    """
    if not rdp_curves:
        raise ValueError("At least one RDP curve is required")

    if alphas is None:
        # Use the union of all α grids
        all_alphas = [np.asarray(a, dtype=np.float64) for a, _ in rdp_curves]
        alphas = np.sort(np.unique(np.concatenate(all_alphas)))

    composed_eps = np.zeros_like(alphas)
    for curve_alphas, curve_eps in rdp_curves:
        curve_alphas = np.asarray(curve_alphas, dtype=np.float64)
        curve_eps = np.asarray(curve_eps, dtype=np.float64)
        interpolated = np.interp(alphas, curve_alphas, curve_eps)
        composed_eps += interpolated

    eps, _ = rdp_to_dp(composed_eps, alphas, delta, method=method)
    return PrivacyBudget(epsilon=max(eps, 1e-15), delta=delta)


def compose_zcdp_then_convert(
    rhos: Sequence[float],
    delta: float,
) -> PrivacyBudget:
    """Compose ρ-zCDP mechanisms and convert to (ε, δ)-DP.

    zCDP composes additively: Σρ_i.

    Args:
        rhos: Sequence of ρ values.
        delta: Target δ.

    Returns:
        Privacy budget after composition.
    """
    if not rhos:
        raise ValueError("At least one rho is required")

    total_rho = sum(rhos)
    return zcdp_to_dp(total_rho, delta)

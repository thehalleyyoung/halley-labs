"""
usability_oracle.variational.kl_divergence — KL divergence and related
information-theoretic divergence measures.

Provides functions for computing:

* :func:`compute_kl_divergence` — D_KL(p ‖ q) for discrete distributions
* :func:`compute_kl_discrete` — D_KL from raw counts with Laplace smoothing
* :func:`compute_kl_gaussian` — closed-form KL for univariate Gaussians
* :func:`compute_mutual_information` — I(X;Y) via KL decomposition
* :func:`symmetric_kl` — Jensen–Shannon divergence  JSD(p ‖ q)
* :func:`renyi_divergence` — Rényi divergence family  D_α(p ‖ q)

All functions operate on numpy arrays and handle numerical edge cases
(zero probabilities, underflow, dimension mismatches).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from usability_oracle.variational.types import KLDivergenceResult

logger = logging.getLogger(__name__)

# Small constant for numerical stability
_EPS = 1e-30


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _validate_distribution(p: np.ndarray, name: str = "p") -> np.ndarray:
    """Validate and normalise a probability distribution.

    Parameters
    ----------
    p : np.ndarray
        Probability vector (non-negative, should sum to ~1).
    name : str
        Name for error messages.

    Returns
    -------
    np.ndarray
        Normalised probability vector as float64.

    Raises
    ------
    ValueError
        If *p* contains negative values or is all-zero.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError(f"Distribution '{name}' must be non-empty")
    if np.any(p < 0):
        raise ValueError(f"Distribution '{name}' contains negative values")
    total = p.sum()
    if total <= 0.0:
        raise ValueError(f"Distribution '{name}' sums to zero")
    if not np.isclose(total, 1.0, atol=1e-6):
        logger.debug("Renormalising '%s' (sum=%.8g)", name, total)
        p = p / total
    return p


def _check_same_shape(p: np.ndarray, q: np.ndarray) -> None:
    """Raise ValueError if *p* and *q* differ in shape."""
    if p.shape != q.shape:
        raise ValueError(
            f"Shape mismatch: p.shape={p.shape}, q.shape={q.shape}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Core KL divergence
# ═══════════════════════════════════════════════════════════════════════════

def compute_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    *,
    validate: bool = True,
) -> float:
    r"""Compute the Kullback–Leibler divergence D_KL(p ‖ q).

    .. math::

        D_{\mathrm{KL}}(p \| q) = \sum_i p_i \ln\!\frac{p_i}{q_i}

    Parameters
    ----------
    p : np.ndarray
        "True" distribution.
    q : np.ndarray
        "Approximating" distribution.
    validate : bool
        If ``True``, validate and normalise inputs.

    Returns
    -------
    float
        KL divergence in **nats** (≥ 0).  Returns ``inf`` when the support
        of *p* is not contained in the support of *q*.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    _check_same_shape(p, q)

    if validate:
        p = _validate_distribution(p, "p")
        q = _validate_distribution(q, "q")

    # Where p > 0 but q == 0 → KL is +inf
    mask_p = p > 0
    if np.any(mask_p & (q <= 0)):
        logger.debug("KL divergence is +inf (support mismatch)")
        return float("inf")

    # Safe log computation: only where p > 0
    kl = np.where(mask_p, p * np.log(p / np.maximum(q, _EPS)), 0.0)
    result = float(kl.sum())

    # Clamp tiny negative values from floating-point rounding
    if result < 0.0:
        if result < -1e-10:
            logger.warning("Negative KL divergence %.6e (numerical issue)", result)
        result = 0.0

    return result


# ═══════════════════════════════════════════════════════════════════════════
# KL from counts (Laplace smoothing)
# ═══════════════════════════════════════════════════════════════════════════

def compute_kl_discrete(
    p_counts: np.ndarray,
    q_counts: np.ndarray,
    *,
    alpha: float = 1.0,
) -> float:
    r"""Compute D_KL(p ‖ q) from raw counts with Laplace smoothing.

    Each count vector is smoothed by adding *α* pseudo-counts before
    normalising:

    .. math::

        \hat{p}_i = \frac{n_i^{(p)} + \alpha}{\sum_j (n_j^{(p)} + \alpha)}

    Parameters
    ----------
    p_counts : np.ndarray
        Observation counts for distribution *p*.
    q_counts : np.ndarray
        Observation counts for distribution *q*.
    alpha : float
        Laplace smoothing parameter (≥ 0).  Default is 1.0 (add-one).

    Returns
    -------
    float
        Smoothed KL divergence in nats.

    Raises
    ------
    ValueError
        If *alpha* is negative or arrays are incompatible.
    """
    if alpha < 0:
        raise ValueError(f"Smoothing parameter alpha must be ≥ 0, got {alpha}")

    p_counts = np.asarray(p_counts, dtype=np.float64).ravel()
    q_counts = np.asarray(q_counts, dtype=np.float64).ravel()
    _check_same_shape(p_counts, q_counts)

    p_smoothed = p_counts + alpha
    q_smoothed = q_counts + alpha

    p_norm = p_smoothed / p_smoothed.sum()
    q_norm = q_smoothed / q_smoothed.sum()

    return compute_kl_divergence(p_norm, q_norm, validate=False)


# ═══════════════════════════════════════════════════════════════════════════
# KL for Gaussians
# ═══════════════════════════════════════════════════════════════════════════

def compute_kl_gaussian(
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
) -> float:
    r"""Closed-form KL divergence between two univariate Gaussians.

    .. math::

        D_{\mathrm{KL}}\!\bigl(\mathcal{N}(\mu_1, \sigma_1^2)
        \;\|\;\mathcal{N}(\mu_2, \sigma_2^2)\bigr)
        = \ln\!\frac{\sigma_2}{\sigma_1}
          + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\,\sigma_2^2}
          - \frac{1}{2}

    Parameters
    ----------
    mu1, sigma1 : float
        Mean and standard deviation of the first Gaussian.
    mu2, sigma2 : float
        Mean and standard deviation of the second Gaussian.

    Returns
    -------
    float
        KL divergence in nats.

    Raises
    ------
    ValueError
        If either standard deviation is non-positive.
    """
    if sigma1 <= 0:
        raise ValueError(f"sigma1 must be positive, got {sigma1}")
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")

    kl = (
        np.log(sigma2 / sigma1)
        + (sigma1**2 + (mu1 - mu2)**2) / (2.0 * sigma2**2)
        - 0.5
    )
    return float(max(kl, 0.0))


# ═══════════════════════════════════════════════════════════════════════════
# Mutual information
# ═══════════════════════════════════════════════════════════════════════════

def compute_mutual_information(
    joint_pxy: np.ndarray,
    marginals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> float:
    r"""Compute mutual information I(X; Y) via KL decomposition.

    .. math::

        I(X; Y) = D_{\mathrm{KL}}\!\bigl(p(x,y) \;\|\; p(x)\,p(y)\bigr)
                 = \sum_{x,y} p(x,y)\,\ln\!\frac{p(x,y)}{p(x)\,p(y)}

    Parameters
    ----------
    joint_pxy : np.ndarray
        Joint distribution p(x, y) as a 2-D array of shape ``(|X|, |Y|)``.
    marginals : tuple of np.ndarray, optional
        Pre-computed marginals ``(p_x, p_y)``.  If ``None``, marginals are
        derived by summing the joint.

    Returns
    -------
    float
        Mutual information in nats (≥ 0).
    """
    joint = np.asarray(joint_pxy, dtype=np.float64)
    if joint.ndim != 2:
        raise ValueError(f"joint_pxy must be 2-D, got ndim={joint.ndim}")

    # Normalise the joint
    total = joint.sum()
    if total <= 0:
        raise ValueError("Joint distribution sums to zero")
    if not np.isclose(total, 1.0, atol=1e-6):
        logger.debug("Renormalising joint (sum=%.8g)", total)
        joint = joint / total

    if marginals is not None:
        p_x = np.asarray(marginals[0], dtype=np.float64).ravel()
        p_y = np.asarray(marginals[1], dtype=np.float64).ravel()
        if p_x.shape[0] != joint.shape[0]:
            raise ValueError("Marginal p_x size doesn't match joint rows")
        if p_y.shape[0] != joint.shape[1]:
            raise ValueError("Marginal p_y size doesn't match joint cols")
        p_x = p_x / p_x.sum()
        p_y = p_y / p_y.sum()
    else:
        p_x = joint.sum(axis=1)
        p_y = joint.sum(axis=0)

    # Compute I(X;Y) = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    outer = np.outer(p_x, p_y)
    mask = joint > 0
    mi = np.where(
        mask,
        joint * np.log(joint / np.maximum(outer, _EPS)),
        0.0,
    )
    result = float(mi.sum())
    return max(result, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Jensen–Shannon divergence (symmetric KL)
# ═══════════════════════════════════════════════════════════════════════════

def symmetric_kl(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    r"""Jensen–Shannon divergence (symmetric KL).

    .. math::

        \mathrm{JSD}(p \| q)
        = \tfrac{1}{2}\,D_{\mathrm{KL}}(p \| m)
        + \tfrac{1}{2}\,D_{\mathrm{KL}}(q \| m),
        \qquad m = \tfrac{1}{2}(p + q)

    The JSD is always finite and satisfies 0 ≤ JSD ≤ ln 2.

    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions.

    Returns
    -------
    float
        JSD in nats.
    """
    p = _validate_distribution(np.asarray(p, dtype=np.float64).ravel(), "p")
    q = _validate_distribution(np.asarray(q, dtype=np.float64).ravel(), "q")
    _check_same_shape(p, q)

    m = 0.5 * (p + q)
    jsd = 0.5 * compute_kl_divergence(p, m, validate=False) + \
          0.5 * compute_kl_divergence(q, m, validate=False)
    return max(float(jsd), 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Rényi divergence
# ═══════════════════════════════════════════════════════════════════════════

def renyi_divergence(
    p: np.ndarray,
    q: np.ndarray,
    alpha: float,
) -> float:
    r"""Rényi divergence of order α.

    .. math::

        D_\alpha(p \| q)
        = \frac{1}{\alpha - 1}
          \ln\!\Bigl(\sum_i p_i^\alpha \, q_i^{1-\alpha}\Bigr)

    Special cases:

    * α → 1:  recovers KL divergence  D_KL(p ‖ q).
    * α = 0.5:  Bhattacharyya distance (times −2).
    * α = ∞:  max-divergence  D_∞(p ‖ q) = ln max_i (p_i / q_i).

    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions.
    alpha : float
        Order parameter (> 0, ≠ 1 except via limit).

    Returns
    -------
    float
        Rényi divergence in nats.

    Raises
    ------
    ValueError
        If *alpha* ≤ 0.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    p = _validate_distribution(np.asarray(p, dtype=np.float64).ravel(), "p")
    q = _validate_distribution(np.asarray(q, dtype=np.float64).ravel(), "q")
    _check_same_shape(p, q)

    # α → 1 limit: standard KL
    if np.isclose(alpha, 1.0, atol=1e-12):
        return compute_kl_divergence(p, q, validate=False)

    # α → ∞: max-divergence
    if alpha == float("inf"):
        mask = p > 0
        if np.any(mask & (q <= 0)):
            return float("inf")
        ratios = np.where(mask, p / np.maximum(q, _EPS), 0.0)
        return float(np.log(ratios.max()))

    # General case
    mask_p = p > 0
    # If p_i > 0 and q_i == 0 and alpha < 1: divergence is finite
    # If p_i > 0 and q_i == 0 and alpha > 1: divergence is +inf
    if alpha > 1 and np.any(mask_p & (q <= 0)):
        return float("inf")

    # Compute sum p_i^alpha * q_i^(1-alpha) safely
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            mask_p,
            np.exp(alpha * np.log(np.maximum(p, _EPS))
                   + (1 - alpha) * np.log(np.maximum(q, _EPS))),
            0.0,
        )

    total = terms.sum()
    if total <= 0:
        return float("inf")

    result = float(np.log(total) / (alpha - 1))
    return max(result, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Policy-level KL (dict-of-dict interface matching protocols)
# ═══════════════════════════════════════════════════════════════════════════

def compute_policy_kl(
    policy: Dict[str, Dict[str, float]],
    reference: Dict[str, Dict[str, float]],
    state_distribution: Optional[Dict[str, float]] = None,
) -> KLDivergenceResult:
    r"""Compute KL(π ‖ π₀) with per-state breakdown.

    .. math::

        D_{\mathrm{KL}}(\pi \| \pi_0)
        = \sum_s d(s) \sum_a \pi(a|s)\,\ln\!\frac{\pi(a|s)}{\pi_0(a|s)}

    Parameters
    ----------
    policy : dict[str, dict[str, float]]
        Current policy  π(a|s).
    reference : dict[str, dict[str, float]]
        Reference policy  π₀(a|s).
    state_distribution : dict[str, float], optional
        Stationary distribution d(s).  Uniform if ``None``.

    Returns
    -------
    KLDivergenceResult
    """
    states = sorted(policy.keys())
    if not states:
        return KLDivergenceResult(
            total_kl=0.0,
            per_state_kl={},
            max_state_kl=0.0,
            max_state_id="",
            is_finite=True,
        )

    # State distribution
    if state_distribution is not None:
        d_s = state_distribution
    else:
        uniform_w = 1.0 / len(states)
        d_s = {s: uniform_w for s in states}

    per_state: Dict[str, float] = {}
    is_finite = True

    for s in states:
        actions = sorted(policy[s].keys())
        if not actions:
            per_state[s] = 0.0
            continue

        ref_s = reference.get(s, {})
        p_arr = np.array([policy[s].get(a, 0.0) for a in actions], dtype=np.float64)
        q_arr = np.array([ref_s.get(a, 0.0) for a in actions], dtype=np.float64)

        # Normalise
        p_sum = p_arr.sum()
        q_sum = q_arr.sum()
        if p_sum > 0:
            p_arr /= p_sum
        if q_sum > 0:
            q_arr /= q_sum
        else:
            # Reference has no support for this state — use uniform
            q_arr = np.ones_like(p_arr) / len(actions)

        kl_s = compute_kl_divergence(p_arr, q_arr, validate=False)
        if not np.isfinite(kl_s):
            is_finite = False
        per_state[s] = kl_s

    # Weighted total
    total_kl = sum(d_s.get(s, 0.0) * per_state[s] for s in states)
    if not np.isfinite(total_kl):
        is_finite = False

    # Find max state
    finite_states = {s: v for s, v in per_state.items() if np.isfinite(v)}
    if finite_states:
        max_state_id = max(finite_states, key=finite_states.get)  # type: ignore[arg-type]
        max_state_kl = finite_states[max_state_id]
    elif per_state:
        max_state_id = next(iter(per_state))
        max_state_kl = float("inf")
    else:
        max_state_id = ""
        max_state_kl = 0.0

    return KLDivergenceResult(
        total_kl=total_kl,
        per_state_kl=per_state,
        max_state_kl=max_state_kl,
        max_state_id=max_state_id,
        is_finite=is_finite,
    )

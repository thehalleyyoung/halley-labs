"""
usability_oracle.information_theory.mutual_information — Mutual information & divergences.

Numerically stable implementations of mutual information, KL divergence,
Jensen-Shannon divergence, and f-divergences used in the bounded-rational
free-energy framework.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from usability_oracle.information_theory.entropy import (
    _as_prob,
    _safe_log,
    _safe_log2,
    conditional_entropy,
    conditional_entropy_yx,
    joint_entropy,
    shannon_entropy,
)
from usability_oracle.information_theory.types import MutualInfoResult


_LOG2 = math.log(2.0)


# ═══════════════════════════════════════════════════════════════════════════
# KL divergence  D_KL(p ‖ q)
# ═══════════════════════════════════════════════════════════════════════════

def kl_divergence(
    p: Union[Sequence[float], NDArray],
    q: Union[Sequence[float], NDArray],
    *,
    base: float = 2.0,
) -> float:
    """Kullback-Leibler divergence D_KL(p ‖ q).

    Parameters
    ----------
    p, q : array-like
        Probability distributions of the same length.
    base : float
        Logarithm base (2 → bits, e → nats).

    Returns
    -------
    float
        KL divergence.  +∞ if supp(p) ⊄ supp(q).
    """
    p_arr = _as_prob(p)
    q_arr = _as_prob(q)
    if p_arr.shape != q_arr.shape:
        raise ValueError("p and q must have the same shape")
    # +∞ where p > 0 and q == 0
    if np.any((p_arr > 0) & (q_arr == 0)):
        return float("inf")
    logp = _safe_log(p_arr)
    logq = _safe_log(q_arr)
    d = float(np.dot(p_arr, logp - logq))
    if base != math.e:
        d /= math.log(base)
    return max(d, 0.0)


def kl_divergence_bits(
    p: Union[Sequence[float], NDArray],
    q: Union[Sequence[float], NDArray],
) -> float:
    """KL divergence in bits."""
    return kl_divergence(p, q, base=2.0)


def kl_divergence_nats(
    p: Union[Sequence[float], NDArray],
    q: Union[Sequence[float], NDArray],
) -> float:
    """KL divergence in nats."""
    return kl_divergence(p, q, base=math.e)


def symmetrized_kl(
    p: Union[Sequence[float], NDArray],
    q: Union[Sequence[float], NDArray],
    *,
    base: float = 2.0,
) -> float:
    """Symmetrized KL divergence: (D_KL(p‖q) + D_KL(q‖p)) / 2."""
    return (kl_divergence(p, q, base=base) + kl_divergence(q, p, base=base)) / 2.0


# ═══════════════════════════════════════════════════════════════════════════
# Jensen-Shannon divergence
# ═══════════════════════════════════════════════════════════════════════════

def jensen_shannon_divergence(
    p: Union[Sequence[float], NDArray],
    q: Union[Sequence[float], NDArray],
    *,
    base: float = 2.0,
) -> float:
    """Jensen-Shannon divergence JSD(p ‖ q).

    JSD(p ‖ q) = 0.5 D_KL(p ‖ m) + 0.5 D_KL(q ‖ m), where m = (p+q)/2.

    Always finite and bounded in [0, log(2)].

    Parameters
    ----------
    p, q : array-like
        Probability distributions.
    base : float
        Logarithm base.

    Returns
    -------
    float
        JSD value.
    """
    p_arr = _as_prob(p)
    q_arr = _as_prob(q)
    m = 0.5 * (p_arr + q_arr)
    return 0.5 * kl_divergence(p_arr, m, base=base) + 0.5 * kl_divergence(q_arr, m, base=base)


def jensen_shannon_distance(
    p: Union[Sequence[float], NDArray],
    q: Union[Sequence[float], NDArray],
    *,
    base: float = 2.0,
) -> float:
    """Jensen-Shannon distance (square root of JSD)."""
    return math.sqrt(max(jensen_shannon_divergence(p, q, base=base), 0.0))


# ═══════════════════════════════════════════════════════════════════════════
# f-divergences
# ═══════════════════════════════════════════════════════════════════════════

def f_divergence(
    p: Union[Sequence[float], NDArray],
    q: Union[Sequence[float], NDArray],
    f: Callable[[float], float],
) -> float:
    """General f-divergence D_f(p ‖ q) = Σ q(x) f(p(x)/q(x)).

    Parameters
    ----------
    p, q : array-like
        Probability distributions.
    f : callable
        Convex function with f(1) = 0.

    Returns
    -------
    float
        f-divergence value.
    """
    p_arr = _as_prob(p)
    q_arr = _as_prob(q)
    result = 0.0
    for pi, qi in zip(p_arr.flat, q_arr.flat):
        if qi > 0:
            result += qi * f(pi / qi)
        elif pi > 0:
            return float("inf")
    return max(result, 0.0)


def chi_squared_divergence(
    p: Union[Sequence[float], NDArray],
    q: Union[Sequence[float], NDArray],
) -> float:
    """χ² divergence: Σ (p(x) - q(x))² / q(x)."""
    p_arr = _as_prob(p)
    q_arr = _as_prob(q)
    if np.any((p_arr > 0) & (q_arr == 0)):
        return float("inf")
    mask = q_arr > 0
    return float(np.sum((p_arr[mask] - q_arr[mask]) ** 2 / q_arr[mask]))


def hellinger_distance(
    p: Union[Sequence[float], NDArray],
    q: Union[Sequence[float], NDArray],
) -> float:
    """Hellinger distance H(p, q) = (1/√2) ‖√p - √q‖₂.

    Returns a value in [0, 1].
    """
    p_arr = _as_prob(p)
    q_arr = _as_prob(q)
    return float(np.sqrt(np.sum((np.sqrt(p_arr) - np.sqrt(q_arr)) ** 2)) / math.sqrt(2.0))


def total_variation_distance(
    p: Union[Sequence[float], NDArray],
    q: Union[Sequence[float], NDArray],
) -> float:
    """Total variation distance TV(p, q) = 0.5 Σ |p(x) - q(x)|.

    Returns a value in [0, 1].
    """
    p_arr = _as_prob(p)
    q_arr = _as_prob(q)
    return 0.5 * float(np.sum(np.abs(p_arr - q_arr)))


# ═══════════════════════════════════════════════════════════════════════════
# Mutual information  I(X; Y)
# ═══════════════════════════════════════════════════════════════════════════

def mutual_information(
    joint: Union[Sequence[Sequence[float]], NDArray],
    *,
    base: float = 2.0,
) -> float:
    """Mutual information I(X; Y) from joint distribution p(x, y).

    I(X;Y) = H(X) + H(Y) - H(X,Y)

    Parameters
    ----------
    joint : 2-D array-like
        Joint distribution p(x, y).
    base : float
        Logarithm base.

    Returns
    -------
    float
        Mutual information.
    """
    pxy = _as_prob(joint)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = (
        shannon_entropy(px, base=base)
        + shannon_entropy(py, base=base)
        - joint_entropy(pxy, base=base)
    )
    return max(mi, 0.0)


def mutual_information_full(
    joint: Union[Sequence[Sequence[float]], NDArray],
    *,
    base: float = 2.0,
) -> MutualInfoResult:
    """Full mutual information decomposition from joint distribution.

    Parameters
    ----------
    joint : 2-D array-like
        Joint distribution p(x, y).
    base : float
        Logarithm base.

    Returns
    -------
    MutualInfoResult
        Complete information decomposition.
    """
    pxy = _as_prob(joint)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    hx = shannon_entropy(px, base=base)
    hy = shannon_entropy(py, base=base)
    hxy = joint_entropy(pxy, base=base)
    mi = max(hx + hy - hxy, 0.0)
    hx_given_y = max(hxy - hy, 0.0)
    hy_given_x = max(hxy - hx, 0.0)
    denom = min(hx, hy) if min(hx, hy) > 0 else 1.0
    nmi = mi / denom

    return MutualInfoResult(
        mutual_info_bits=mi if base == 2.0 else mi,
        entropy_x_bits=hx,
        entropy_y_bits=hy,
        conditional_entropy_y_given_x_bits=hy_given_x,
        conditional_entropy_x_given_y_bits=hx_given_y,
        joint_entropy_bits=hxy,
        normalized=nmi,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Conditional mutual information  I(X; Y | Z)
# ═══════════════════════════════════════════════════════════════════════════

def conditional_mutual_information(
    joint_xyz: NDArray,
    *,
    base: float = 2.0,
) -> float:
    """Conditional mutual information I(X; Y | Z) from joint p(x, y, z).

    I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)

    Parameters
    ----------
    joint_xyz : 3-D NDArray
        Joint distribution p(x, y, z) with shape (|X|, |Y|, |Z|).
    base : float
        Logarithm base.

    Returns
    -------
    float
        Conditional mutual information.
    """
    pxyz = _as_prob(joint_xyz)
    pxz = pxyz.sum(axis=1)  # shape (|X|, |Z|)
    pyz = pxyz.sum(axis=0)  # shape (|Y|, |Z|)
    pz = pxyz.sum(axis=(0, 1))  # shape (|Z|,)

    h_xz = shannon_entropy(pxz.ravel(), base=base)
    h_yz = shannon_entropy(pyz.ravel(), base=base)
    h_xyz = shannon_entropy(pxyz.ravel(), base=base)
    h_z = shannon_entropy(pz, base=base)

    cmi = h_xz + h_yz - h_xyz - h_z
    return max(cmi, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Normalized mutual information variants
# ═══════════════════════════════════════════════════════════════════════════

def normalized_mutual_information(
    joint: Union[Sequence[Sequence[float]], NDArray],
    *,
    variant: str = "min",
    base: float = 2.0,
) -> float:
    """Normalized mutual information.

    Parameters
    ----------
    joint : 2-D array-like
        Joint distribution p(x, y).
    variant : str
        Normalization variant:
          "min"  — I(X;Y) / min(H(X), H(Y))
          "max"  — I(X;Y) / max(H(X), H(Y))
          "sqrt" — I(X;Y) / sqrt(H(X) · H(Y))
          "sum"  — 2·I(X;Y) / (H(X) + H(Y))
          "joint" — I(X;Y) / H(X,Y)
    base : float
        Logarithm base.

    Returns
    -------
    float
        Normalized MI in [0, 1].
    """
    pxy = _as_prob(joint)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    hx = shannon_entropy(px, base=base)
    hy = shannon_entropy(py, base=base)
    mi = mutual_information(pxy, base=base)

    if variant == "min":
        denom = min(hx, hy)
    elif variant == "max":
        denom = max(hx, hy)
    elif variant == "sqrt":
        denom = math.sqrt(hx * hy) if hx > 0 and hy > 0 else 0.0
    elif variant == "sum":
        denom = 0.5 * (hx + hy)
    elif variant == "joint":
        denom = joint_entropy(pxy, base=base)
    else:
        raise ValueError(f"Unknown NMI variant: {variant}")

    return mi / denom if denom > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Interaction information (for 3+ variables)
# ═══════════════════════════════════════════════════════════════════════════

def interaction_information(joint_xyz: NDArray, *, base: float = 2.0) -> float:
    """Interaction information for three variables.

    II(X;Y;Z) = I(X;Y|Z) - I(X;Y)

    Can be negative (synergy) or positive (redundancy).

    Parameters
    ----------
    joint_xyz : 3-D NDArray
        Joint distribution p(x, y, z).
    base : float
        Logarithm base.

    Returns
    -------
    float
        Interaction information.
    """
    pxyz = _as_prob(joint_xyz)
    pxy = pxyz.sum(axis=2)
    mi_xy = mutual_information(pxy, base=base)
    cmi_xy_z = conditional_mutual_information(pxyz, base=base)
    return cmi_xy_z - mi_xy


# ═══════════════════════════════════════════════════════════════════════════
# Total correlation (multi-information)
# ═══════════════════════════════════════════════════════════════════════════

def total_correlation(joint: NDArray, *, base: float = 2.0) -> float:
    """Total correlation (multi-information) C(X₁,...,Xₙ).

    C = Σ H(X_i) - H(X₁,...,Xₙ)

    Parameters
    ----------
    joint : N-D NDArray
        Joint distribution over N variables (each axis = one variable).
    base : float
        Logarithm base.

    Returns
    -------
    float
        Total correlation (≥ 0).
    """
    p = _as_prob(joint)
    h_joint = shannon_entropy(p.ravel(), base=base)
    h_sum = 0.0
    for axis in range(p.ndim):
        marginal = p.sum(axis=tuple(i for i in range(p.ndim) if i != axis))
        h_sum += shannon_entropy(marginal, base=base)
    return max(h_sum - h_joint, 0.0)


def dual_total_correlation(joint: NDArray, *, base: float = 2.0) -> float:
    """Dual total correlation (binding information).

    D = H(X₁,...,Xₙ) - Σ H(X_i | X_{-i})

    Parameters
    ----------
    joint : N-D NDArray
        Joint distribution.
    base : float
        Logarithm base.

    Returns
    -------
    float
        Dual total correlation (≥ 0).
    """
    p = _as_prob(joint)
    h_joint = shannon_entropy(p.ravel(), base=base)
    h_cond_sum = 0.0
    for axis in range(p.ndim):
        # H(X_i | X_{-i}) = H(X₁,...,Xₙ) - H(X_{-i})
        others = tuple(i for i in range(p.ndim) if i != axis)
        marginal_others = p.sum(axis=axis)
        h_others = shannon_entropy(marginal_others.ravel(), base=base)
        h_cond_sum += h_joint - h_others
    return max(h_joint - h_cond_sum, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Pointwise mutual information
# ═══════════════════════════════════════════════════════════════════════════

def pointwise_mutual_information(
    joint: Union[Sequence[Sequence[float]], NDArray],
    *,
    base: float = 2.0,
    normalized: bool = False,
) -> NDArray:
    """Pointwise mutual information PMI(x, y) = log(p(x,y) / (p(x)p(y))).

    Parameters
    ----------
    joint : 2-D array-like
        Joint distribution p(x, y).
    base : float
        Logarithm base.
    normalized : bool
        If True, return NPMI = PMI / (-log p(x,y)) ∈ [-1, 1].

    Returns
    -------
    NDArray
        Matrix of PMI values with same shape as joint.
    """
    pxy = _as_prob(joint)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px * py

    log_fn = _safe_log if base == math.e else _safe_log2
    log_base_factor = 1.0 if base in (math.e, 2.0) else 1.0 / math.log(base)

    pmi = np.zeros_like(pxy)
    mask = (pxy > 0) & (expected > 0)
    if base == 2.0:
        pmi[mask] = np.log2(pxy[mask] / expected[mask])
    elif base == math.e:
        pmi[mask] = np.log(pxy[mask] / expected[mask])
    else:
        pmi[mask] = np.log(pxy[mask] / expected[mask]) * log_base_factor
    # Where pxy == 0, PMI is -∞ but we leave as 0 by convention

    if normalized:
        log_pxy = np.zeros_like(pxy)
        if base == 2.0:
            log_pxy[mask] = -np.log2(pxy[mask])
        else:
            log_pxy[mask] = -np.log(pxy[mask])
            if base != math.e:
                log_pxy[mask] *= log_base_factor
        denom = np.where(log_pxy > 0, log_pxy, 1.0)
        pmi = pmi / denom

    return pmi


# ═══════════════════════════════════════════════════════════════════════════
# Batch KL divergence
# ═══════════════════════════════════════════════════════════════════════════

def batch_kl_divergence(
    p: Union[Sequence[Sequence[float]], NDArray],
    q: Union[Sequence[Sequence[float]], NDArray],
    *,
    base: float = 2.0,
) -> NDArray:
    """Batch KL divergence: D_KL(p_i ‖ q_i) for each row.

    Parameters
    ----------
    p, q : 2-D array-like
        Matched pairs of distributions (each row is a distribution).
    base : float
        Logarithm base.

    Returns
    -------
    NDArray
        1-D array of KL divergences.
    """
    p_arr = _as_prob(p)
    q_arr = _as_prob(q)
    if p_arr.shape != q_arr.shape:
        raise ValueError("p and q must have the same shape")
    if p_arr.ndim == 1:
        p_arr = p_arr.reshape(1, -1)
        q_arr = q_arr.reshape(1, -1)

    logp = np.where(p_arr > 0, np.log(p_arr), 0.0)
    logq = np.where(q_arr > 0, np.log(q_arr), 0.0)
    d = np.sum(p_arr * (logp - logq), axis=1)

    # +∞ where p > 0 but q == 0
    bad = (p_arr > 0) & (q_arr == 0)
    bad_rows = np.any(bad, axis=1)
    d[bad_rows] = float("inf")

    if base != math.e:
        finite = np.isfinite(d)
        d[finite] /= math.log(base)
    np.maximum(d, 0.0, out=d)
    return d


__all__ = [
    "batch_kl_divergence",
    "chi_squared_divergence",
    "conditional_mutual_information",
    "dual_total_correlation",
    "f_divergence",
    "hellinger_distance",
    "interaction_information",
    "jensen_shannon_distance",
    "jensen_shannon_divergence",
    "kl_divergence",
    "kl_divergence_bits",
    "kl_divergence_nats",
    "mutual_information",
    "mutual_information_full",
    "normalized_mutual_information",
    "pointwise_mutual_information",
    "symmetrized_kl",
    "total_correlation",
    "total_variation_distance",
]

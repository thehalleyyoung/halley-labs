"""
Differentiable privacy loss functions and composition theorems.

Provides analytic gradients of privacy divergences (KL, Rényi,
hockey-stick), composition theorems, subsampling amplification,
and smooth sensitivity computations—all differentiable w.r.t.
mechanism probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
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
from scipy.special import logsumexp

from dp_forge.types import GradientInfo, PrivacyBudget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPS = 1e-30  # avoid log(0)


def _safe_log(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.log(np.maximum(x, _EPS))


def _safe_div(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.where(b > _EPS, a / np.maximum(b, _EPS), 0.0)


# ---------------------------------------------------------------------------
# KL divergence and gradient
# ---------------------------------------------------------------------------


def kl_divergence(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
) -> float:
    """Compute KL(P || Q) = sum_i p_i log(p_i / q_i).

    Args:
        p: Distribution P (non-negative, sums to 1).
        q: Distribution Q (non-negative, sums to 1).

    Returns:
        KL divergence value.
    """
    mask = p > _EPS
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], _EPS))))


def kl_divergence_gradient(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute KL(P || Q) and gradients w.r.t. both P and Q.

    Returns:
        (kl_value, grad_p, grad_q).
    """
    mask = p > _EPS
    kl = kl_divergence(p, q)

    grad_p = np.zeros_like(p)
    grad_p[mask] = np.log(p[mask] / np.maximum(q[mask], _EPS)) + 1.0

    grad_q = np.zeros_like(q)
    grad_q[mask] = -p[mask] / np.maximum(q[mask], _EPS)

    return kl, grad_p, grad_q


# ---------------------------------------------------------------------------
# Rényi divergence and gradient
# ---------------------------------------------------------------------------


def renyi_divergence(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    alpha: float,
) -> float:
    """Compute Rényi divergence D_α(P || Q).

    For α > 1:  D_α = 1/(α-1) log( sum_i p_i^α q_i^{1-α} )

    Args:
        p: Distribution P.
        q: Distribution Q.
        alpha: Rényi order (> 1).

    Returns:
        Rényi divergence value.
    """
    if alpha <= 1.0:
        raise ValueError(f"alpha must be > 1, got {alpha}")
    if alpha == 1.0:
        return kl_divergence(p, q)

    mask = (p > _EPS) & (q > _EPS)
    log_terms = alpha * _safe_log(p[mask]) + (1 - alpha) * _safe_log(q[mask])
    log_sum = logsumexp(log_terms)
    return float(log_sum / (alpha - 1.0))


def renyi_divergence_gradient(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    alpha: float,
) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute Rényi divergence and gradients w.r.t. P and Q.

    Returns:
        (renyi_value, grad_p, grad_q).
    """
    if alpha <= 1.0:
        raise ValueError(f"alpha must be > 1, got {alpha}")

    mask = (p > _EPS) & (q > _EPS)
    pa = np.power(p[mask], alpha)
    qb = np.power(q[mask], 1.0 - alpha)
    terms = pa * qb
    S = np.sum(terms)
    if S < _EPS:
        return 0.0, np.zeros_like(p), np.zeros_like(q)

    val = float(np.log(S) / (alpha - 1.0))

    # ∂D/∂p_i = (α / (α-1)) * p_i^{α-1} * q_i^{1-α} / S
    grad_p = np.zeros_like(p)
    grad_p[mask] = (alpha / (alpha - 1.0)) * np.power(p[mask], alpha - 1.0) * qb / S

    # ∂D/∂q_i = ((1-α) / (α-1)) * p_i^α * q_i^{-α} / S
    #         = -p_i^α * q_i^{-α} / S
    grad_q = np.zeros_like(q)
    grad_q[mask] = -pa * np.power(q[mask], -alpha) / S

    return val, grad_p, grad_q


# ---------------------------------------------------------------------------
# Hockey-stick divergence and gradient
# ---------------------------------------------------------------------------


def hockey_stick_divergence(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    epsilon: float,
) -> float:
    """Compute hockey-stick divergence δ(ε) = max_S [P(S) - e^ε Q(S)].

    Equivalent to: sum_i max(0, p_i - e^ε q_i).

    Args:
        p: Distribution P.
        q: Distribution Q.
        epsilon: Privacy parameter ε.

    Returns:
        Hockey-stick divergence value (the δ that ε-DP fails by).
    """
    ee = math.exp(epsilon)
    diff = p - ee * q
    return float(np.sum(np.maximum(diff, 0.0)))


def hockey_stick_gradient(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    epsilon: float,
) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute hockey-stick divergence and subgradients w.r.t. P, Q.

    The gradient uses a soft-max approximation for differentiability.

    Returns:
        (hs_value, grad_p, grad_q).
    """
    ee = math.exp(epsilon)
    diff = p - ee * q

    # Smooth approximation: softplus(x) = log(1 + exp(β*x)) / β
    beta = 100.0  # sharpness
    smooth = np.log1p(np.exp(np.clip(beta * diff, -500, 500))) / beta
    val = float(np.sum(smooth))

    # Gradient of softplus: sigmoid(β*x)
    sigmoid = 1.0 / (1.0 + np.exp(np.clip(-beta * diff, -500, 500)))

    grad_p = sigmoid.copy()
    grad_q = -ee * sigmoid

    return val, grad_p, grad_q


def hockey_stick_gradient_exact(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    epsilon: float,
) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Exact (sub-)gradient of hockey-stick divergence.

    Returns:
        (hs_value, subgrad_p, subgrad_q).
    """
    ee = math.exp(epsilon)
    diff = p - ee * q
    active = (diff > 0).astype(np.float64)
    val = float(np.sum(np.maximum(diff, 0.0)))
    return val, active, -ee * active


# ---------------------------------------------------------------------------
# Max divergence
# ---------------------------------------------------------------------------


def max_divergence(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
) -> float:
    """Compute max-divergence: max_i log(p_i / q_i).

    Args:
        p: Distribution P.
        q: Distribution Q.

    Returns:
        Max divergence value.
    """
    mask = (p > _EPS) & (q > _EPS)
    if not np.any(mask):
        return 0.0
    ratios = _safe_log(p[mask]) - _safe_log(q[mask])
    return float(np.max(ratios))


def max_divergence_gradient(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute max-divergence and gradients using soft-max.

    Returns:
        (value, grad_p, grad_q).
    """
    mask = (p > _EPS) & (q > _EPS)
    ratios = np.full_like(p, -np.inf)
    ratios[mask] = _safe_log(p[mask]) - _safe_log(q[mask])

    # Smooth max via log-sum-exp
    beta = 50.0
    lse = float(logsumexp(beta * ratios)) / beta
    weights = np.exp(beta * (ratios - lse * beta))
    weights /= np.sum(weights) + _EPS

    grad_p = np.zeros_like(p)
    grad_p[mask] = weights[mask] / np.maximum(p[mask], _EPS)

    grad_q = np.zeros_like(q)
    grad_q[mask] = -weights[mask] / np.maximum(q[mask], _EPS)

    return lse, grad_p, grad_q


# ---------------------------------------------------------------------------
# Total variation distance
# ---------------------------------------------------------------------------


def total_variation(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
) -> float:
    """Total variation distance TV(P, Q) = 0.5 * sum |p_i - q_i|."""
    return float(0.5 * np.sum(np.abs(p - q)))


def total_variation_gradient(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Total variation and subgradients.

    Returns:
        (tv_value, subgrad_p, subgrad_q).
    """
    diff = p - q
    tv = float(0.5 * np.sum(np.abs(diff)))
    sign = 0.5 * np.sign(diff)
    return tv, sign, -sign


# ---------------------------------------------------------------------------
# Differentiable composition theorems
# ---------------------------------------------------------------------------


def basic_composition_gradient(
    epsilons: npt.NDArray[np.float64],
) -> Tuple[float, npt.NDArray[np.float64]]:
    """Basic composition: ε_total = Σ ε_i.

    Returns:
        (total_epsilon, gradient_vector = ones).
    """
    return float(np.sum(epsilons)), np.ones_like(epsilons)


def advanced_composition_gradient(
    epsilons: npt.NDArray[np.float64],
    delta: float,
    delta_0: float = 0.0,
) -> Tuple[float, npt.NDArray[np.float64]]:
    """Advanced composition theorem (differentiable form).

    ε_total = sqrt(2 * Σ ε_i^2 * log(1/δ')) + Σ ε_i * (e^{ε_i} - 1) / (e^{ε_i} + 1)

    Returns:
        (total_epsilon, gradient_w.r.t._epsilons).
    """
    if delta <= 0:
        return float(np.sum(epsilons)), np.ones_like(epsilons)

    delta_prime = delta - delta_0
    if delta_prime <= 0:
        return float(np.sum(epsilons)), np.ones_like(epsilons)

    log_inv_delta = math.log(1.0 / delta_prime)
    sum_sq = float(np.sum(epsilons ** 2))
    sqrt_term = math.sqrt(2.0 * sum_sq * log_inv_delta) if sum_sq > 0 else 0.0

    # tanh-based bound for the linear term
    tanh_eps = np.tanh(epsilons / 2.0)
    linear_term = float(np.sum(epsilons * tanh_eps))
    total = sqrt_term + linear_term

    # Gradient
    grad = np.zeros_like(epsilons)
    if sqrt_term > 0:
        grad += 2.0 * epsilons * log_inv_delta / sqrt_term
    grad += tanh_eps + epsilons * (1.0 - tanh_eps ** 2) * 0.5

    return total, grad


def rdp_composition_gradient(
    rdp_epsilons: npt.NDArray[np.float64],
    alpha: float,
) -> Tuple[float, npt.NDArray[np.float64]]:
    """RDP composition: ε_α^{total} = Σ ε_α^{(i)}.

    Rényi DP composes additively at each order α.

    Returns:
        (total_rdp_epsilon, gradient_vector = ones).
    """
    return float(np.sum(rdp_epsilons)), np.ones_like(rdp_epsilons)


def rdp_to_dp_gradient(
    rdp_epsilon: float,
    alpha: float,
    delta: float,
) -> Tuple[float, float]:
    """Convert RDP guarantee to (ε, δ)-DP with gradient.

    ε = rdp_epsilon - log(δ) / (α - 1)

    Returns:
        (dp_epsilon, d_dp_epsilon / d_rdp_epsilon).
    """
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    dp_epsilon = rdp_epsilon - math.log(delta) / (alpha - 1.0)
    return dp_epsilon, 1.0  # gradient of linear function


# ---------------------------------------------------------------------------
# Gradient of ε w.r.t. mechanism parameters
# ---------------------------------------------------------------------------


def epsilon_gradient_wrt_mechanism(
    mechanism: npt.NDArray[np.float64],
    i: int,
    i_prime: int,
    target_delta: float = 0.0,
) -> GradientInfo:
    """Compute gradient of the privacy parameter ε w.r.t. mechanism probs.

    For pure DP (δ=0), ε = max_j log(M[i,j] / M[i',j]).
    The gradient is computed using a smooth-max approximation.

    Args:
        mechanism: n × k probability table.
        i: First database index.
        i_prime: Adjacent database index.
        target_delta: Target δ (0 for pure DP).

    Returns:
        GradientInfo with ε value and gradient w.r.t. flattened mechanism.
    """
    p = mechanism[i]
    q = mechanism[i_prime]
    n, k = mechanism.shape

    if target_delta <= 0:
        # Pure DP: ε = max_j log(p_j / q_j)
        mask = (p > _EPS) & (q > _EPS)
        log_ratios = np.full(k, -np.inf)
        log_ratios[mask] = np.log(p[mask] / q[mask])

        beta = 100.0
        eps_val = float(logsumexp(beta * log_ratios) / beta)

        weights = np.exp(beta * log_ratios - logsumexp(beta * log_ratios))

        grad_flat = np.zeros(n * k, dtype=np.float64)
        grad_p = np.zeros(k, dtype=np.float64)
        grad_q = np.zeros(k, dtype=np.float64)
        grad_p[mask] = weights[mask] / p[mask]
        grad_q[mask] = -weights[mask] / q[mask]
        grad_flat[i * k: (i + 1) * k] = grad_p
        grad_flat[i_prime * k: (i_prime + 1) * k] = grad_q
    else:
        # Approximate DP: use hockey-stick inversion
        eps_val_hs, gp, gq = hockey_stick_gradient(p, q, 0.0)
        eps_val = float(np.log(max(eps_val_hs, _EPS)))
        grad_flat = np.zeros(n * k, dtype=np.float64)
        scale = 1.0 / max(eps_val_hs, _EPS)
        grad_flat[i * k: (i + 1) * k] = scale * gp
        grad_flat[i_prime * k: (i_prime + 1) * k] = scale * gq

    return GradientInfo(value=eps_val, gradient=grad_flat)


# ---------------------------------------------------------------------------
# Subsampling amplification gradients
# ---------------------------------------------------------------------------


def subsampled_epsilon_gradient(
    epsilon: float,
    sampling_rate: float,
) -> Tuple[float, float, float]:
    """Compute amplified ε under Poisson subsampling and gradients.

    Amplification: ε_sub ≈ log(1 + q(e^ε - 1))  where q is sampling rate.

    Returns:
        (amplified_epsilon, d_eps_sub/d_epsilon, d_eps_sub/d_q).
    """
    q = sampling_rate
    ee = math.exp(epsilon)
    inner = 1.0 + q * (ee - 1.0)
    if inner <= 0:
        inner = _EPS
    eps_sub = math.log(inner)

    d_eps_d_epsilon = q * ee / inner
    d_eps_d_q = (ee - 1.0) / inner

    return eps_sub, d_eps_d_epsilon, d_eps_d_q


def subsampled_rdp_gradient(
    rdp_epsilon: float,
    alpha: float,
    sampling_rate: float,
) -> Tuple[float, float, float]:
    """Compute subsampled RDP epsilon with gradients.

    Uses the bound: ε_α^sub ≤ (1/(α-1)) log(1 + q^2 * C(α) * (e^{(α-1)ε} - 1))
    Simplified to first-order approximation for gradient computation.

    Returns:
        (amplified_rdp, d/d_rdp_epsilon, d/d_sampling_rate).
    """
    q = sampling_rate
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")

    exp_term = math.exp((alpha - 1.0) * rdp_epsilon)
    inner = 1.0 + q * q * (exp_term - 1.0)
    if inner <= 0:
        inner = _EPS

    amplified = math.log(inner) / (alpha - 1.0)

    d_rdp = q * q * exp_term / inner
    d_q = 2.0 * q * (exp_term - 1.0) / (inner * (alpha - 1.0))

    return amplified, d_rdp, d_q


# ---------------------------------------------------------------------------
# Smooth sensitivity computation (differentiable)
# ---------------------------------------------------------------------------


def smooth_sensitivity(
    local_sensitivities: npt.NDArray[np.float64],
    beta: float,
    distances: npt.NDArray[np.float64],
) -> float:
    """Compute β-smooth sensitivity.

    S^*_β(x) = max_y { LS(y) * exp(-β * d(x,y)) }

    where *local_sensitivities* and *distances* are pre-computed for
    all databases y.

    Args:
        local_sensitivities: LS(y) for each database y.
        beta: Smoothing parameter.
        distances: d(x, y) for each database y.

    Returns:
        Smooth sensitivity value.
    """
    weighted = local_sensitivities * np.exp(-beta * distances)
    return float(np.max(weighted))


def smooth_sensitivity_gradient(
    local_sensitivities: npt.NDArray[np.float64],
    beta: float,
    distances: npt.NDArray[np.float64],
) -> Tuple[float, float, npt.NDArray[np.float64]]:
    """Compute smooth sensitivity and gradients w.r.t. β and LS.

    Returns:
        (smooth_sens, d/d_beta, d/d_local_sensitivities).
    """
    exp_terms = np.exp(-beta * distances)
    weighted = local_sensitivities * exp_terms
    idx = int(np.argmax(weighted))
    ss_val = float(weighted[idx])

    d_beta = -distances[idx] * ss_val
    d_ls = np.zeros_like(local_sensitivities)
    d_ls[idx] = exp_terms[idx]

    return ss_val, d_beta, d_ls


# ---------------------------------------------------------------------------
# Chi-squared divergence
# ---------------------------------------------------------------------------


def chi_squared_divergence(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
) -> float:
    """Compute chi-squared divergence: Σ (p_i - q_i)^2 / q_i."""
    mask = q > _EPS
    return float(np.sum((p[mask] - q[mask]) ** 2 / q[mask]))


def chi_squared_gradient(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Chi-squared divergence and gradients.

    Returns:
        (value, grad_p, grad_q).
    """
    mask = q > _EPS
    val = chi_squared_divergence(p, q)

    grad_p = np.zeros_like(p)
    grad_p[mask] = 2.0 * (p[mask] - q[mask]) / q[mask]

    grad_q = np.zeros_like(q)
    diff = p[mask] - q[mask]
    grad_q[mask] = -(diff ** 2) / (q[mask] ** 2) - 2.0 * diff / q[mask]

    return val, grad_p, grad_q


# ---------------------------------------------------------------------------
# Unified divergence interface
# ---------------------------------------------------------------------------


def compute_divergence_and_gradient(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    divergence_type: str,
    **kwargs: float,
) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute a named divergence and its gradients.

    Args:
        p: Distribution P.
        q: Distribution Q.
        divergence_type: One of "kl", "renyi", "hockey_stick",
                         "max", "tv", "chi_squared".
        **kwargs: Extra parameters (alpha for Rényi, epsilon for HS).

    Returns:
        (value, grad_p, grad_q).
    """
    dt = divergence_type.lower()
    if dt == "kl":
        return kl_divergence_gradient(p, q)
    elif dt == "renyi":
        alpha = kwargs.get("alpha", 2.0)
        return renyi_divergence_gradient(p, q, alpha)
    elif dt in ("hockey_stick", "hs"):
        epsilon = kwargs.get("epsilon", 1.0)
        return hockey_stick_gradient(p, q, epsilon)
    elif dt in ("max", "max_divergence"):
        return max_divergence_gradient(p, q)
    elif dt in ("tv", "total_variation"):
        return total_variation_gradient(p, q)
    elif dt in ("chi_squared", "chi2"):
        return chi_squared_gradient(p, q)
    else:
        raise ValueError(f"Unknown divergence type: {divergence_type}")


__all__ = [
    "kl_divergence",
    "kl_divergence_gradient",
    "renyi_divergence",
    "renyi_divergence_gradient",
    "hockey_stick_divergence",
    "hockey_stick_gradient",
    "hockey_stick_gradient_exact",
    "max_divergence",
    "max_divergence_gradient",
    "total_variation",
    "total_variation_gradient",
    "chi_squared_divergence",
    "chi_squared_gradient",
    "basic_composition_gradient",
    "advanced_composition_gradient",
    "rdp_composition_gradient",
    "rdp_to_dp_gradient",
    "epsilon_gradient_wrt_mechanism",
    "subsampled_epsilon_gradient",
    "subsampled_rdp_gradient",
    "smooth_sensitivity",
    "smooth_sensitivity_gradient",
    "compute_divergence_and_gradient",
]

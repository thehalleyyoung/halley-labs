"""
taintflow.utils.math_utils – Information-theoretic math utilities.

Provides pure-Python implementations of entropy, mutual information,
channel capacity formulae, and various bounds used by the TaintFlow
capacity analysis engine.  All logarithms are base-2 unless noted.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

__all__ = [
    "log2_safe",
    "entropy",
    "binary_entropy",
    "mutual_information_discrete",
    "kl_divergence",
    "channel_capacity_gaussian",
    "channel_capacity_bsc",
    "channel_capacity_bec",
    "taylor_log2_1plus",
    "fano_inequality_bound",
    "data_processing_inequality_compose",
    "parallel_channel_capacity",
    "chi_squared_channel_capacity",
    "wishart_channel_capacity",
    "order_statistic_channel",
    "gaussian_mixture_entropy",
    "sub_gaussian_norm",
    "confidence_interval_mi",
    "hoeffding_bound",
    "dpi_sequential",
    "dpi_parallel",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LN2 = math.log(2.0)
_LOG2_E = math.log2(math.e)
_EPS = 1e-300  # to avoid log(0)

# ---------------------------------------------------------------------------
# Core information-theoretic functions
# ---------------------------------------------------------------------------


def log2_safe(x: float) -> float:
    """Compute log₂(x) with graceful handling for x ≤ 0.

    Returns ``-inf`` for x = 0 and ``nan`` for x < 0, matching IEEE 754
    semantics without raising exceptions.

    >>> log2_safe(1.0)
    0.0
    >>> log2_safe(0.0)
    -inf
    """
    if x <= 0.0:
        if x == 0.0:
            return float("-inf")
        return float("nan")
    return math.log2(x)


def entropy(probs: Sequence[float]) -> float:
    """Shannon entropy H(X) = −Σ pᵢ log₂(pᵢ) in bits.

    Parameters
    ----------
    probs : sequence of float
        A discrete probability distribution (must sum to ≈1).

    Returns
    -------
    float
        The entropy in bits.  Returns 0.0 for a degenerate distribution.

    >>> entropy([0.5, 0.5])
    1.0
    >>> abs(entropy([0.25, 0.25, 0.25, 0.25]) - 2.0) < 1e-12
    True
    """
    h = 0.0
    for p in probs:
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def binary_entropy(p: float) -> float:
    """Binary entropy function H(p) = −p log₂ p − (1−p) log₂(1−p).

    Parameters
    ----------
    p : float
        Probability in [0, 1].

    Returns
    -------
    float
        Entropy in bits.

    >>> binary_entropy(0.5)
    1.0
    >>> binary_entropy(0.0)
    0.0
    >>> binary_entropy(1.0)
    0.0
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def mutual_information_discrete(
    joint_probs: Sequence[Sequence[float]],
) -> float:
    """Mutual information I(X;Y) from a joint probability table.

    Parameters
    ----------
    joint_probs : 2-D array-like
        ``joint_probs[i][j]`` = P(X = i, Y = j).

    Returns
    -------
    float
        I(X;Y) in bits.

    >>> abs(mutual_information_discrete([[0.25, 0.25], [0.25, 0.25]]))
    0.0
    """
    n_x = len(joint_probs)
    if n_x == 0:
        return 0.0
    n_y = len(joint_probs[0])

    # marginals
    p_x = [0.0] * n_x
    p_y = [0.0] * n_y
    for i in range(n_x):
        for j in range(n_y):
            p_x[i] += joint_probs[i][j]
            p_y[j] += joint_probs[i][j]

    mi = 0.0
    for i in range(n_x):
        for j in range(n_y):
            pij = joint_probs[i][j]
            if pij > 0.0 and p_x[i] > 0.0 and p_y[j] > 0.0:
                mi += pij * math.log2(pij / (p_x[i] * p_y[j]))
    return mi


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Kullback–Leibler divergence KL(p ‖ q) in bits.

    Parameters
    ----------
    p, q : sequences of float
        Discrete distributions of equal length.

    Returns
    -------
    float
        KL(p‖q) ≥ 0.  Returns ``inf`` if any p[i] > 0 where q[i] = 0.

    >>> kl_divergence([0.5, 0.5], [0.5, 0.5])
    0.0
    """
    if len(p) != len(q):
        raise ValueError("distributions must have same length")
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0:
            if qi <= 0.0:
                return float("inf")
            kl += pi * math.log2(pi / qi)
    return kl


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen–Shannon divergence JS(p, q) = ½ KL(p‖m) + ½ KL(q‖m).

    Parameters
    ----------
    p, q : sequences of float
        Discrete distributions.

    Returns
    -------
    float
        JS divergence in bits, always in [0, 1].
    """
    if len(p) != len(q):
        raise ValueError("distributions must have same length")
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# ---------------------------------------------------------------------------
# Channel capacity formulae
# ---------------------------------------------------------------------------


def channel_capacity_gaussian(snr: float) -> float:
    """Capacity of an AWGN channel: C = ½ log₂(1 + SNR) bits/use.

    Parameters
    ----------
    snr : float
        Signal-to-noise ratio (linear, not dB).

    Returns
    -------
    float
        Channel capacity in bits.

    >>> channel_capacity_gaussian(1.0)
    0.5
    """
    if snr < 0.0:
        raise ValueError("SNR must be non-negative")
    return 0.5 * math.log2(1.0 + snr)


def channel_capacity_bsc(p: float) -> float:
    """Capacity of a Binary Symmetric Channel: C = 1 − H(p).

    Parameters
    ----------
    p : float
        Crossover probability in [0, 0.5].

    Returns
    -------
    float
        Capacity in bits.

    >>> channel_capacity_bsc(0.0)
    1.0
    >>> abs(channel_capacity_bsc(0.5))
    0.0
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"crossover probability must be in [0,1], got {p}")
    return 1.0 - binary_entropy(p)


def channel_capacity_bec(epsilon: float) -> float:
    """Capacity of a Binary Erasure Channel: C = 1 − ε.

    Parameters
    ----------
    epsilon : float
        Erasure probability in [0, 1].

    Returns
    -------
    float
        Capacity in bits.

    >>> channel_capacity_bec(0.0)
    1.0
    >>> channel_capacity_bec(1.0)
    0.0
    """
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError(f"erasure probability must be in [0,1], got {epsilon}")
    return 1.0 - epsilon


def channel_capacity_z_channel(p: float) -> float:
    """Capacity of a Z-channel (asymmetric binary channel).

    0 → 0 always, 1 → 0 with probability p.

    C = log₂(1 + (1−p) · p^(p/(1−p))) for 0 < p < 1.
    """
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    # Exact formula
    exp_term = p ** (p / (1.0 - p))
    return math.log2(1.0 + (1.0 - p) * exp_term)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def taylor_log2_1plus(x: float, terms: int = 20) -> float:
    """Taylor expansion of log₂(1 + x) for small |x| to avoid cancellation.

    Uses the series ln(1+x) = x − x²/2 + x³/3 − x⁴/4 + … and converts
    to base 2 via log₂(1+x) = ln(1+x) / ln(2).

    Parameters
    ----------
    x : float
        Should satisfy |x| < 1 for convergence.
    terms : int
        Number of terms in the Taylor expansion.

    Returns
    -------
    float
        Approximation of log₂(1 + x).

    >>> abs(taylor_log2_1plus(0.0)) < 1e-15
    True
    >>> abs(taylor_log2_1plus(1.0) - 1.0) < 0.01
    True
    """
    if abs(x) >= 1.0 and x != 1.0:
        # Fall back to direct computation for |x| ≥ 1
        return math.log2(1.0 + x)
    ln_val = 0.0
    x_power = x
    for n in range(1, terms + 1):
        if n % 2 == 1:
            ln_val += x_power / n
        else:
            ln_val -= x_power / n
        x_power *= x
    return ln_val / _LN2


# ---------------------------------------------------------------------------
# Bounds & inequalities
# ---------------------------------------------------------------------------


def fano_inequality_bound(
    error_prob: float,
    alphabet_size: int,
) -> float:
    """Upper bound on conditional entropy H(X|Y) from Fano's inequality.

    H(X|Y) ≤ H(Pₑ) + Pₑ log₂(|X| − 1)

    Parameters
    ----------
    error_prob : float
        Probability of error Pₑ ∈ [0, 1].
    alphabet_size : int
        Size of the alphabet |X| ≥ 2.

    Returns
    -------
    float
        Upper bound on H(X|Y) in bits.
    """
    if alphabet_size < 2:
        raise ValueError("alphabet size must be ≥ 2")
    if error_prob <= 0.0:
        return 0.0
    if error_prob >= 1.0:
        return math.log2(alphabet_size)
    return binary_entropy(error_prob) + error_prob * math.log2(alphabet_size - 1)


def hoeffding_bound(n: int, epsilon: float) -> float:
    """Hoeffding's inequality: P(|X̄ − μ| ≥ ε) ≤ 2·exp(−2nε²).

    Parameters
    ----------
    n : int
        Number of independent samples (bounded in [0, 1]).
    epsilon : float
        Deviation threshold.

    Returns
    -------
    float
        Upper bound on the tail probability.
    """
    if n <= 0 or epsilon <= 0.0:
        return 1.0
    exponent = -2.0 * n * epsilon * epsilon
    return min(1.0, 2.0 * math.exp(exponent))


def mcdiarmid_bound(n: int, epsilon: float, c_sum_sq: float) -> float:
    """McDiarmid's inequality for bounded-difference functions.

    P(f(X) − E[f(X)] ≥ ε) ≤ exp(−2ε² / Σcᵢ²)

    Parameters
    ----------
    n : int
        Number of samples.
    epsilon : float
        Deviation threshold.
    c_sum_sq : float
        Sum of squared bounded differences Σcᵢ².

    Returns
    -------
    float
        Upper bound on the tail probability.
    """
    if c_sum_sq <= 0.0 or epsilon <= 0.0:
        return 1.0
    exponent = -2.0 * epsilon * epsilon / c_sum_sq
    return min(1.0, math.exp(exponent))


# ---------------------------------------------------------------------------
# Data processing inequality
# ---------------------------------------------------------------------------


def dpi_sequential(c1: float, c2: float) -> float:
    """Data processing inequality for two sequential channels.

    If X → Y → Z, then I(X;Z) ≤ min(I(X;Y), I(Y;Z)).  When expressed in
    terms of capacities, the end-to-end capacity ≤ min(C₁, C₂).

    Parameters
    ----------
    c1, c2 : float
        Capacities of the first and second channel (bits).

    Returns
    -------
    float
        Upper bound on the end-to-end capacity.
    """
    return min(c1, c2)


def dpi_parallel(channels: Sequence[float]) -> float:
    """DPI bound for parallel independent channels.

    Total capacity = Σ Cᵢ.

    Parameters
    ----------
    channels : sequence of float
        Capacities of parallel channels.

    Returns
    -------
    float
        Sum of individual capacities.
    """
    return sum(channels)


def data_processing_inequality_compose(
    capacities: Sequence[float],
) -> float:
    """End-to-end capacity of a sequence of channels via the DPI.

    For a Markov chain X₁ → X₂ → … → Xₙ, the end-to-end capacity
    is bounded above by the minimum individual channel capacity.

    Parameters
    ----------
    capacities : sequence of float
        Capacity of each stage.

    Returns
    -------
    float
        Upper bound on end-to-end capacity.
    """
    if not capacities:
        return 0.0
    return min(capacities)


def parallel_channel_capacity(capacities: Sequence[float]) -> float:
    """Total capacity of independent parallel channels.

    C_total = Σᵢ Cᵢ

    Parameters
    ----------
    capacities : sequence of float
        Individual channel capacities.

    Returns
    -------
    float
        Total parallel capacity in bits.
    """
    return sum(capacities)


# ---------------------------------------------------------------------------
# Statistical channel capacity models
# ---------------------------------------------------------------------------


def chi_squared_channel_capacity(df: int, n: int) -> float:
    """Information capacity of a variance-estimation channel.

    When estimating variance from n samples of a Gaussian with df degrees
    of freedom, the mutual information between the true variance σ² and
    the sample variance s² is:

        I(σ²; s²) ≈ ½ log₂(n / (2·df)) + ψ(df/2) / (2 ln 2) − ½

    We use a simplified bound:
        C ≈ ½ log₂(1 + n/(2·df))

    Parameters
    ----------
    df : int
        Degrees of freedom (typically n − 1).
    n : int
        Number of samples.

    Returns
    -------
    float
        Approximate capacity in bits.
    """
    if df <= 0 or n <= 0:
        return 0.0
    snr = n / (2.0 * df)
    return 0.5 * math.log2(1.0 + snr)


def wishart_channel_capacity(d: int, n: int, rho: float) -> float:
    """Information capacity for covariance matrix estimation.

    For a d×d covariance matrix estimated from n samples with test
    fraction ρ, the leakage is bounded by:

        C ≈ (d(d+1)/4) · log₂(1 + ρ·n/d)

    Parameters
    ----------
    d : int
        Dimensionality (number of features).
    n : int
        Number of samples.
    rho : float
        Test fraction ∈ [0, 1].

    Returns
    -------
    float
        Approximate capacity in bits.
    """
    if d <= 0 or n <= 0:
        return 0.0
    rho = max(0.0, min(1.0, rho))
    n_params = d * (d + 1) / 2.0
    snr = rho * n / max(d, 1)
    return (n_params / 2.0) * math.log2(1.0 + snr)


def order_statistic_channel(k: int, n: int, rho: float) -> float:
    """Information capacity of a rank-based (order statistic) operation.

    When computing the k-th order statistic from n samples with test
    fraction ρ:

        C ≈ ½ log₂(1 + ρ · n · β(k, n−k+1))

    where β is the variance of the k-th order statistic of a Uniform(0,1):
        Var = k(n−k+1) / ((n+1)²(n+2))

    Parameters
    ----------
    k : int
        Rank of the order statistic (1-based).
    n : int
        Total number of samples.
    rho : float
        Test fraction.

    Returns
    -------
    float
        Approximate capacity in bits.
    """
    if k < 1 or k > n or n <= 0:
        return 0.0
    rho = max(0.0, min(1.0, rho))
    # Variance of Beta(k, n-k+1)
    alpha = float(k)
    beta_param = float(n - k + 1)
    variance = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1.0))
    snr = rho * n * variance
    return 0.5 * math.log2(1.0 + snr)


# ---------------------------------------------------------------------------
# Gaussian mixtures
# ---------------------------------------------------------------------------


def _log_sum_exp(values: Sequence[float]) -> float:
    """Numerically stable log-sum-exp."""
    if not values:
        return float("-inf")
    max_val = max(values)
    if math.isinf(max_val) and max_val < 0:
        return float("-inf")
    return max_val + math.log(sum(math.exp(v - max_val) for v in values))


def gaussian_mixture_entropy(
    means: Sequence[float],
    variances: Sequence[float],
    weights: Sequence[float],
    n_samples: int = 10000,
) -> float:
    """Estimate entropy of a 1-D Gaussian mixture via numerical integration.

    Uses deterministic quadrature over a fine grid spanning the support.

    Parameters
    ----------
    means : sequence of float
        Component means.
    variances : sequence of float
        Component variances (must be > 0).
    weights : sequence of float
        Mixture weights (must sum to ≈1).
    n_samples : int
        Number of quadrature points.

    Returns
    -------
    float
        Approximate H(X) in bits.
    """
    k = len(means)
    if k == 0:
        return 0.0

    # Normalise weights
    w_sum = sum(weights)
    w = [wi / w_sum for wi in weights]

    # Determine integration range: mean ± 5σ of the overall mixture
    overall_mean = sum(w[i] * means[i] for i in range(k))
    overall_var = sum(
        w[i] * (variances[i] + (means[i] - overall_mean) ** 2) for i in range(k)
    )
    overall_std = math.sqrt(max(overall_var, 1e-12))
    lo = overall_mean - 6.0 * overall_std
    hi = overall_mean + 6.0 * overall_std
    dx = (hi - lo) / n_samples

    # Pre-compute log-weights and 1/(2σ²)
    log_w = [math.log(wi) if wi > 0 else float("-inf") for wi in w]
    inv_2var = [1.0 / (2.0 * max(v, 1e-300)) for v in variances]
    log_norm = [
        -0.5 * math.log(2.0 * math.pi * max(v, 1e-300)) for v in variances
    ]

    entropy_acc = 0.0
    for step in range(n_samples):
        x = lo + (step + 0.5) * dx
        # log p(x) via log-sum-exp
        log_components = []
        for i in range(k):
            log_p_comp = log_w[i] + log_norm[i] - inv_2var[i] * (x - means[i]) ** 2
            log_components.append(log_p_comp)
        log_px = _log_sum_exp(log_components)
        px = math.exp(log_px)
        if px > 0.0:
            entropy_acc -= px * log_px * dx

    # Convert from nats to bits
    return entropy_acc / _LN2


# ---------------------------------------------------------------------------
# Sub-Gaussian norm estimation
# ---------------------------------------------------------------------------


def sub_gaussian_norm(data: Sequence[float]) -> float:
    """Estimate the sub-Gaussian norm ‖X‖_ψ₂ from samples.

    Uses the empirical moment method:
        ‖X‖_ψ₂ ≈ sup_p  (E[|X|^p])^(1/p) / √p

    We check p = 2, 4, 6, 8 and return the maximum.

    Parameters
    ----------
    data : sequence of float
        i.i.d. samples of a centered random variable.

    Returns
    -------
    float
        Estimated sub-Gaussian norm.
    """
    n = len(data)
    if n == 0:
        return 0.0

    mean_val = sum(data) / n
    centered = [x - mean_val for x in data]

    best = 0.0
    for p in (2, 4, 6, 8):
        moment = sum(abs(x) ** p for x in centered) / n
        norm_est = moment ** (1.0 / p) / math.sqrt(p)
        if norm_est > best:
            best = norm_est
    return best


# ---------------------------------------------------------------------------
# Confidence intervals for mutual information
# ---------------------------------------------------------------------------


def confidence_interval_mi(
    mi_estimate: float,
    n_samples: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Confidence interval for a plug-in MI estimate.

    Uses a normal approximation with variance ≈ MI / n (rough Jackknife
    estimate) combined with the known non-negativity of MI.

    Parameters
    ----------
    mi_estimate : float
        Point estimate of I(X;Y) in bits.
    n_samples : int
        Number of samples used.
    alpha : float
        Significance level (default 0.05 → 95% CI).

    Returns
    -------
    (lower, upper) : tuple of float
        Confidence interval bounds.  Lower is clipped to 0.
    """
    if n_samples <= 1:
        return (0.0, float("inf"))
    # Approximate standard error
    se = math.sqrt(max(mi_estimate, 1e-12) / n_samples)
    # Quantile of standard normal
    z = _normal_quantile(1.0 - alpha / 2.0)
    lower = max(0.0, mi_estimate - z * se)
    upper = mi_estimate + z * se
    return (lower, upper)


def _normal_quantile(p: float) -> float:
    """Approximate quantile of the standard normal using Beasley-Springer-Moro."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    if p == 0.5:
        return 0.0

    # Rational approximation (Abramowitz & Stegun 26.2.23)
    if 0.5 < p:
        t = math.sqrt(-2.0 * math.log(1.0 - p))
    else:
        t = math.sqrt(-2.0 * math.log(p))

    # Coefficients for rational approximation
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    numerator = c0 + c1 * t + c2 * t * t
    denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    z = t - numerator / denominator

    if p < 0.5:
        return -z
    return z


# ---------------------------------------------------------------------------
# Psi (digamma) approximation
# ---------------------------------------------------------------------------


def _digamma(x: float) -> float:
    """Approximation of the digamma function ψ(x) for x > 0."""
    result = 0.0
    # Use recurrence ψ(x+1) = ψ(x) + 1/x to push x into asymptotic region
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0
    # Asymptotic expansion
    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    result += math.log(x) - 0.5 * inv_x
    result -= inv_x2 * (
        1.0 / 12.0
        - inv_x2 * (1.0 / 120.0 - inv_x2 * (1.0 / 252.0 - inv_x2 / 240.0))
    )
    return result


# ---------------------------------------------------------------------------
# KSG estimator for continuous MI
# ---------------------------------------------------------------------------


def _euclidean_dist(a: Sequence[float], b: Sequence[float]) -> float:
    """L2 distance between two vectors."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _chebyshev_dist(a: Sequence[float], b: Sequence[float]) -> float:
    """L∞ (Chebyshev) distance."""
    return max(abs(ai - bi) for ai, bi in zip(a, b))


def ksg_mutual_information(
    x: Sequence[Sequence[float]],
    y: Sequence[Sequence[float]],
    k: int = 3,
) -> float:
    """Kraskov-Stögbauer-Grassberger (KSG) estimator for continuous MI.

    This is the KSG-1 algorithm using L∞ (Chebyshev) norm.

    Parameters
    ----------
    x : sequence of vectors
        X samples, each is a list/tuple of floats.
    y : sequence of vectors
        Y samples (same length as x).
    k : int
        Number of nearest neighbours.

    Returns
    -------
    float
        Estimated I(X;Y) in nats (not bits).  Multiply by 1/ln(2) for bits.
    """
    n = len(x)
    if n < k + 1:
        return 0.0

    # Build joint vectors
    xy = [list(x[i]) + list(y[i]) for i in range(n)]
    dx = len(x[0])
    dy = len(y[0])

    # For each point find k-th nearest neighbour distance in joint space
    mi = _digamma(k) - 1.0 / k + _digamma(n)

    for i in range(n):
        # Compute distances to all other points in joint space (Chebyshev)
        dists: List[Tuple[float, int]] = []
        for j in range(n):
            if i == j:
                continue
            d = _chebyshev_dist(xy[i], xy[j])
            dists.append((d, j))
        dists.sort()
        eps = dists[k - 1][0]  # k-th nearest neighbour distance

        # Count nx: points where Chebyshev distance in X space < eps
        nx = 0
        for j in range(n):
            if i == j:
                continue
            dx_ij = _chebyshev_dist(x[i], x[j])
            if dx_ij < eps:
                nx += 1

        # Count ny: points where Chebyshev distance in Y space < eps
        ny = 0
        for j in range(n):
            if i == j:
                continue
            dy_ij = _chebyshev_dist(y[i], y[j])
            if dy_ij < eps:
                ny += 1

        mi -= (_digamma(nx + 1) + _digamma(ny + 1)) / n

    return max(mi, 0.0)


# ---------------------------------------------------------------------------
# Rényi entropy
# ---------------------------------------------------------------------------


def renyi_entropy(probs: Sequence[float], alpha: float) -> float:
    """Rényi entropy Hα(X) of a discrete distribution.

    Parameters
    ----------
    probs : sequence of float
        Probability distribution.
    alpha : float
        Order parameter > 0, ≠ 1.

    Returns
    -------
    float
        Rényi entropy in bits.

    For α → 1, returns Shannon entropy (limit).
    """
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if abs(alpha - 1.0) < 1e-12:
        return entropy(probs)
    if alpha == 0.0:
        # Hartley entropy: log₂(support size)
        support = sum(1 for p in probs if p > 0.0)
        return math.log2(max(support, 1))
    if math.isinf(alpha):
        # Min-entropy: -log₂(max(p))
        return -math.log2(max(p for p in probs if p > 0.0))

    s = sum(p ** alpha for p in probs if p > 0.0)
    if s <= 0.0:
        return 0.0
    return math.log2(s) / (1.0 - alpha)


# ---------------------------------------------------------------------------
# Conditional entropy & chain rule
# ---------------------------------------------------------------------------


def conditional_entropy(
    joint_probs: Sequence[Sequence[float]],
) -> float:
    """Conditional entropy H(X|Y) = H(X,Y) − H(Y).

    Parameters
    ----------
    joint_probs : 2-D array-like
        ``joint_probs[i][j]`` = P(X = i, Y = j).

    Returns
    -------
    float
        H(X|Y) in bits.
    """
    n_x = len(joint_probs)
    if n_x == 0:
        return 0.0
    n_y = len(joint_probs[0])

    # H(X, Y)
    flat = [joint_probs[i][j] for i in range(n_x) for j in range(n_y)]
    h_xy = entropy(flat)

    # H(Y)
    p_y = [0.0] * n_y
    for i in range(n_x):
        for j in range(n_y):
            p_y[j] += joint_probs[i][j]
    h_y = entropy(p_y)

    return max(h_xy - h_y, 0.0)


# ---------------------------------------------------------------------------
# Cross-entropy and perplexity
# ---------------------------------------------------------------------------


def cross_entropy(p: Sequence[float], q: Sequence[float]) -> float:
    """Cross-entropy H(p, q) = −Σ p[i] log₂ q[i]."""
    if len(p) != len(q):
        raise ValueError("distributions must have same length")
    result = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0:
            if qi <= 0.0:
                return float("inf")
            result -= pi * math.log2(qi)
    return result


def perplexity(probs: Sequence[float]) -> float:
    """Perplexity of a distribution: 2^{H(X)}."""
    return 2.0 ** entropy(probs)


# ---------------------------------------------------------------------------
# Capacity of multi-dimensional channels
# ---------------------------------------------------------------------------


def mimo_capacity_equal_power(
    n_tx: int,
    n_rx: int,
    snr: float,
) -> float:
    """Approximate capacity of a MIMO channel with equal power allocation.

    Uses the formula C ≈ min(nₜ, nᵣ) · log₂(1 + SNR · max(nₜ, nᵣ) / min(nₜ, nᵣ)).
    This is a rough upper bound assuming i.i.d. Rayleigh fading.

    Parameters
    ----------
    n_tx : int
        Number of transmit antennas.
    n_rx : int
        Number of receive antennas.
    snr : float
        Per-antenna SNR.

    Returns
    -------
    float
        Approximate capacity in bits.
    """
    if n_tx <= 0 or n_rx <= 0 or snr < 0:
        return 0.0
    m = min(n_tx, n_rx)
    n = max(n_tx, n_rx)
    return m * math.log2(1.0 + snr * n / m)


# ---------------------------------------------------------------------------
# Leakage-specific utility functions
# ---------------------------------------------------------------------------


def leakage_bits_from_rho(rho: float, n: int, d: int = 1) -> float:
    """Estimate leakage in bits from test fraction ρ through d features.

    A simple model: each feature contributes ½ log₂(1 + ρn) bits of
    potential leakage.

    Parameters
    ----------
    rho : float
        Test fraction.
    n : int
        Total number of samples.
    d : int
        Number of features involved.

    Returns
    -------
    float
        Estimated leakage in bits.
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return 0.0
    return d * 0.5 * math.log2(1.0 + rho * n)


def snr_from_rho(rho: float, n: int) -> float:
    """Convert test fraction ρ and sample count to an effective SNR.

    SNR = ρ · n / (1 − ρ)  (ratio of test to train influence).
    """
    if rho <= 0.0 or rho >= 1.0 or n <= 0:
        return 0.0
    return rho * n / (1.0 - rho)


def capacity_composition(
    capacities: Sequence[float],
    topology: str = "sequential",
) -> float:
    """Compose channel capacities based on DAG topology.

    Parameters
    ----------
    capacities : sequence of float
        Per-stage capacities.
    topology : str
        ``"sequential"`` or ``"parallel"``.

    Returns
    -------
    float
        Overall capacity bound in bits.
    """
    if not capacities:
        return 0.0
    if topology == "sequential":
        return min(capacities)
    if topology == "parallel":
        return sum(capacities)
    raise ValueError(f"unknown topology: {topology!r}")

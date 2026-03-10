"""
Power analysis and envelope characterisation for CI tests.

Provides functions for computing the statistical power of CI tests at
a given sample size and effect size, and for characterising the power
envelope — the set of alternatives detectable with a given budget.

Includes:
- Power analysis for partial correlation and KCI tests
- Minimum detectable effect size (power envelope)
- Minimum sample size for desired power
- Effect size estimation from data
- Monte Carlo power estimation
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from causalcert.types import CITestMethod, NodeId, NodeSet

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class PowerEstimate:
    """Result of a power analysis for a single CI test.

    Attributes
    ----------
    x, y : NodeId
        Test endpoints.
    conditioning_set : NodeSet
        Conditioning set.
    method : CITestMethod
        CI test method.
    alpha : float
        Significance level.
    effect_size : float
        Effect size under the alternative hypothesis.
    power : float
        Estimated power in [0, 1].
    n_required : int
        Sample size required to achieve the target power.
    """

    x: NodeId
    y: NodeId
    conditioning_set: NodeSet
    method: CITestMethod
    alpha: float
    effect_size: float
    power: float
    n_required: int


# ---------------------------------------------------------------------------
# Power for partial correlation test
# ---------------------------------------------------------------------------


def _power_partial_correlation(
    n: int,
    effect_size: float,
    conditioning_size: int,
    alpha: float = 0.05,
) -> float:
    """Compute the power of the Fisher-z partial correlation test.

    Under the alternative with true partial correlation ``rho``, the
    Fisher-z statistic has mean ``sqrt(n - k - 3) * atanh(rho)`` and
    variance 1 (asymptotically).

    Parameters
    ----------
    n : int
        Sample size.
    effect_size : float
        True partial correlation under the alternative (``rho``).
    conditioning_size : int
        Number of conditioning variables ``k``.
    alpha : float
        Significance level.

    Returns
    -------
    float
        Power in ``[0, 1]``.
    """
    dof = n - conditioning_size - 3
    if dof < 1:
        return 0.0

    effect_size = float(np.clip(effect_size, -1.0 + _EPS, 1.0 - _EPS))

    # Non-centrality parameter
    ncp = math.sqrt(dof) * math.atanh(effect_size)

    # Critical value (two-sided)
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    # Power = P(|Z + ncp| > z_crit) = P(Z > z_crit - ncp) + P(Z < -z_crit - ncp)
    power = (
        stats.norm.sf(z_crit - abs(ncp))
        + stats.norm.cdf(-z_crit - abs(ncp))
    )

    return float(np.clip(power, 0.0, 1.0))


def _power_kci(
    n: int,
    effect_size: float,
    conditioning_size: int,
    alpha: float = 0.05,
) -> float:
    """Approximate power of the KCI test.

    Uses an asymptotic approximation based on the non-central gamma
    distribution.  The KCI test has lower power than the partial
    correlation test for linear alternatives but higher power for
    nonlinear alternatives.

    We approximate the power using the ratio of effective degrees of
    freedom: KCI is roughly equivalent to a partial correlation test
    with an efficiency factor ``eta ≈ 0.6`` for linear alternatives
    and ``eta ≈ 1.2`` for nonlinear.  Here we use ``eta = 0.8`` as
    a compromise.

    Parameters
    ----------
    n : int
        Sample size.
    effect_size : float
        Equivalent partial correlation.
    conditioning_size : int
        Conditioning set size.
    alpha : float
        Significance level.

    Returns
    -------
    float
        Approximate power in ``[0, 1]``.
    """
    # KCI effective sample size adjustment
    eta = 0.8
    n_eff = max(int(n * eta), conditioning_size + 4)
    return _power_partial_correlation(n_eff, effect_size, conditioning_size, alpha)


def _power_rank(
    n: int,
    effect_size: float,
    conditioning_size: int,
    alpha: float = 0.05,
) -> float:
    """Approximate power of the rank-based (Spearman) CI test.

    Spearman's rank correlation has asymptotic relative efficiency
    ``3/pi ≈ 0.955`` relative to Pearson for normal data, so the
    power is very close to partial correlation but slightly lower.

    Parameters
    ----------
    n : int
        Sample size.
    effect_size : float
        Equivalent partial correlation.
    conditioning_size : int
        Conditioning set size.
    alpha : float
        Significance level.

    Returns
    -------
    float
        Approximate power.
    """
    eta = 3.0 / math.pi  # ~0.955, ARE of Spearman vs Pearson
    n_eff = max(int(n * eta), conditioning_size + 4)
    return _power_partial_correlation(n_eff, effect_size, conditioning_size, alpha)


def _power_crt(
    n: int,
    effect_size: float,
    conditioning_size: int,
    alpha: float = 0.05,
    n_permutations: int = 500,
) -> float:
    """Approximate power of the CRT.

    The CRT has limited resolution: its minimum achievable p-value is
    ``1 / (n_permutations + 1)``.  Power is additionally reduced by
    the discreteness of the permutation null.

    Parameters
    ----------
    n : int
        Sample size.
    effect_size : float
        Effect size.
    conditioning_size : int
        Conditioning set size.
    alpha : float
        Significance level.
    n_permutations : int
        Number of permutations.

    Returns
    -------
    float
        Approximate power.
    """
    # Cannot reject if alpha < 1/(B+1)
    min_p = 1.0 / (n_permutations + 1)
    if alpha < min_p:
        return 0.0

    # Base power from partial-correlation equivalent
    base_power = _power_partial_correlation(n, effect_size, conditioning_size, alpha)

    # Reduce by discreteness penalty
    discreteness_penalty = max(1.0 - 2.0 / (n_permutations * alpha), 0.5)

    return float(np.clip(base_power * discreteness_penalty, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Power dispatcher
# ---------------------------------------------------------------------------


def _power(
    n: int,
    effect_size: float,
    conditioning_size: int,
    alpha: float = 0.05,
    method: CITestMethod = CITestMethod.PARTIAL_CORRELATION,
) -> float:
    """Compute power for a given CI test method.

    Parameters
    ----------
    n : int
        Sample size.
    effect_size : float
        True partial correlation (or equivalent).
    conditioning_size : int
        Conditioning set size.
    alpha : float
        Significance level.
    method : CITestMethod
        CI test method.

    Returns
    -------
    float
        Power in ``[0, 1]``.
    """
    if method == CITestMethod.PARTIAL_CORRELATION:
        return _power_partial_correlation(n, effect_size, conditioning_size, alpha)
    elif method == CITestMethod.KERNEL:
        return _power_kci(n, effect_size, conditioning_size, alpha)
    elif method == CITestMethod.RANK:
        return _power_rank(n, effect_size, conditioning_size, alpha)
    elif method == CITestMethod.CRT:
        return _power_crt(n, effect_size, conditioning_size, alpha)
    elif method == CITestMethod.ENSEMBLE:
        # Ensemble power is at least as high as the best constituent
        powers = [
            _power_partial_correlation(n, effect_size, conditioning_size, alpha),
            _power_kci(n, effect_size, conditioning_size, alpha),
            _power_rank(n, effect_size, conditioning_size, alpha),
        ]
        return max(powers)
    else:
        return _power_partial_correlation(n, effect_size, conditioning_size, alpha)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def power_envelope(
    n: int,
    conditioning_size: int,
    alpha: float = 0.05,
    method: CITestMethod = CITestMethod.PARTIAL_CORRELATION,
    target_power: float = 0.80,
) -> float:
    """Compute the minimum detectable effect size at sample size *n*.

    Finds the smallest partial correlation ``rho`` such that the test
    achieves the target power at significance level ``alpha``.

    Uses binary search over the effect-size interval ``(0, 1)``.

    Parameters
    ----------
    n : int
        Sample size.
    conditioning_size : int
        Number of conditioning variables.
    alpha : float
        Significance level.
    method : CITestMethod
        CI test method (affects degrees of freedom).
    target_power : float
        Target power level (default 0.80).

    Returns
    -------
    float
        Minimum detectable partial correlation (effect size).
    """
    dof = n - conditioning_size - 3
    if dof < 1:
        return 1.0

    lo, hi = 0.0, 1.0 - _EPS
    for _ in range(100):
        mid = (lo + hi) / 2.0
        p = _power(n, mid, conditioning_size, alpha, method)
        if p < target_power:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break

    return (lo + hi) / 2.0


def required_sample_size(
    effect_size: float,
    conditioning_size: int,
    alpha: float = 0.05,
    power: float = 0.80,
    method: CITestMethod = CITestMethod.PARTIAL_CORRELATION,
) -> int:
    """Compute the sample size needed to detect a given effect size.

    Uses binary search over sample sizes.

    Parameters
    ----------
    effect_size : float
        Target partial correlation to detect.
    conditioning_size : int
        Number of conditioning variables.
    alpha : float
        Significance level.
    power : float
        Desired power.
    method : CITestMethod
        CI test method.

    Returns
    -------
    int
        Required sample size.
    """
    if abs(effect_size) < _EPS:
        return 10_000_000  # Effectively infinite

    # Binary search for n
    lo = conditioning_size + 4
    hi = lo + 10

    # First, find an upper bound where power is achieved
    while _power(hi, effect_size, conditioning_size, alpha, method) < power:
        hi *= 2
        if hi > 10_000_000:
            return hi

    # Binary search between lo and hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        p = _power(mid, effect_size, conditioning_size, alpha, method)
        if p < power:
            lo = mid
        else:
            hi = mid

    return hi


def estimate_effect_size(
    partial_correlation: float,
) -> float:
    """Convert a partial correlation to Cohen's effect size f².

    ``f² = r² / (1 - r²)``

    Parameters
    ----------
    partial_correlation : float
        Observed partial correlation.

    Returns
    -------
    float
        Cohen's f².
    """
    r2 = partial_correlation ** 2
    if r2 >= 1.0 - _EPS:
        return 1e6
    return r2 / (1.0 - r2)


def monte_carlo_power(
    n: int,
    effect_size: float,
    conditioning_size: int,
    alpha: float = 0.05,
    n_simulations: int = 1000,
    seed: int = 42,
) -> float:
    """Estimate power via Monte Carlo simulation.

    Generates ``n_simulations`` datasets under the alternative hypothesis
    with the given partial correlation, runs the Fisher-z test on each,
    and reports the rejection rate.

    Parameters
    ----------
    n : int
        Sample size per simulation.
    effect_size : float
        True partial correlation under the alternative.
    conditioning_size : int
        Conditioning set size.
    alpha : float
        Significance level.
    n_simulations : int
        Number of MC replications.
    seed : int
        Random seed.

    Returns
    -------
    float
        Estimated power (rejection rate).
    """
    rng = np.random.default_rng(seed)
    k = conditioning_size
    p = 2 + k  # total variables: X, Y, Z_1, ..., Z_k

    if n < p + 3:
        return 0.0

    effect_size = float(np.clip(effect_size, -1.0 + _EPS, 1.0 - _EPS))

    rejections = 0
    for _ in range(n_simulations):
        # Generate data: X, Y share correlation = effect_size
        # given Z_1, ..., Z_k (all independent)
        Z = rng.standard_normal((n, k)) if k > 0 else np.empty((n, 0))

        # Conditional on Z:
        # X = Z @ beta_x + eps_x
        # Y = effect_size * X + Z @ beta_y + eps_y
        # This gives partial corr(X, Y | Z) ≈ effect_size (for small effect)
        beta_x = rng.standard_normal(k) * 0.3 if k > 0 else np.array([])
        eps_x = rng.standard_normal(n)
        X = (Z @ beta_x if k > 0 else 0.0) + eps_x

        beta_y = rng.standard_normal(k) * 0.3 if k > 0 else np.array([])
        eps_y = rng.standard_normal(n)
        Y = effect_size * X + (Z @ beta_y if k > 0 else 0.0) + eps_y

        # Compute partial correlation
        if k == 0:
            r = np.corrcoef(X, Y)[0, 1]
        else:
            all_vars = np.column_stack([X, Y, Z])
            cov = np.cov(all_vars, rowvar=False)
            try:
                prec = np.linalg.inv(cov)
                denom = np.sqrt(abs(prec[0, 0] * prec[1, 1]))
                r = -prec[0, 1] / max(denom, _EPS) if denom > _EPS else 0.0
            except np.linalg.LinAlgError:
                r = 0.0

        r = float(np.clip(r, -1.0 + _EPS, 1.0 - _EPS))

        # Fisher z-test
        dof = n - k - 3
        if dof < 1:
            continue
        z_stat = math.sqrt(dof) * math.atanh(r)
        p_val = 2.0 * stats.norm.sf(abs(z_stat))

        if p_val < alpha:
            rejections += 1

    return rejections / n_simulations


def power_analysis(
    x: NodeId,
    y: NodeId,
    conditioning_set: NodeSet,
    n: int,
    effect_size: float,
    alpha: float = 0.05,
    method: CITestMethod = CITestMethod.PARTIAL_CORRELATION,
) -> PowerEstimate:
    """Full power analysis for a specific CI test triple.

    Parameters
    ----------
    x, y : NodeId
        Test endpoints.
    conditioning_set : NodeSet
        Conditioning set.
    n : int
        Available sample size.
    effect_size : float
        Assumed effect size.
    alpha : float
        Significance level.
    method : CITestMethod
        CI test method.

    Returns
    -------
    PowerEstimate
        Complete power analysis result.
    """
    k = len(conditioning_set)
    pw = _power(n, effect_size, k, alpha, method)
    n_req = required_sample_size(effect_size, k, alpha, 0.80, method)

    return PowerEstimate(
        x=x,
        y=y,
        conditioning_set=conditioning_set,
        method=method,
        alpha=alpha,
        effect_size=effect_size,
        power=pw,
        n_required=n_req,
    )

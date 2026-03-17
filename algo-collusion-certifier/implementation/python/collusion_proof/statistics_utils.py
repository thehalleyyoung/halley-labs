"""Statistical utility functions for CollusionProof."""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Weighted / robust descriptive statistics
# ---------------------------------------------------------------------------
def weighted_mean(
    values: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """Compute the (optionally weighted) mean."""
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.mean(values))
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_variance(
    values: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """Compute the (optionally weighted) variance."""
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.var(values, ddof=1))
    weights = np.asarray(weights, dtype=float)
    mu = weighted_mean(values, weights)
    w_sum = np.sum(weights)
    return float(np.sum(weights * (values - mu) ** 2) / w_sum)


def trimmed_mean(values: np.ndarray, proportion: float = 0.1) -> float:
    """Trimmed mean, removing *proportion* from each tail."""
    values = np.asarray(values, dtype=float)
    return float(stats.trim_mean(values, proportion))


def winsorized_mean(values: np.ndarray, proportion: float = 0.1) -> float:
    """Winsorized mean – extreme values are clamped, not removed."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    k = int(np.floor(n * proportion))
    sorted_v = np.sort(values)
    low, high = sorted_v[k], sorted_v[n - k - 1]
    clipped = np.clip(values, low, high)
    return float(np.mean(clipped))


def robust_std(values: np.ndarray) -> float:
    """MAD-based robust standard deviation estimate.

    Uses ``1.4826 * MAD`` which is a consistent estimator of σ for
    normal data.
    """
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(1.4826 * mad)


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------
def bootstrap_mean(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Returns ``(lower, point_estimate, upper)``.
    """
    return bootstrap_statistic(
        data, np.mean, n_bootstrap, confidence, random_state
    )


def bootstrap_statistic(
    data: np.ndarray,
    statistic: Callable,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Bootstrap CI for an arbitrary statistic."""
    rng = np.random.RandomState(random_state)
    data = np.asarray(data, dtype=float)
    n = len(data)
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        boot_stats[i] = statistic(sample)
    alpha = 1 - confidence
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    point = float(statistic(data))
    return lower, point, upper


def bca_confidence_interval(
    data: np.ndarray,
    statistic: Callable,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """BCa (bias-corrected and accelerated) confidence interval."""
    rng = np.random.RandomState(random_state)
    data = np.asarray(data, dtype=float)
    n = len(data)
    theta_hat = float(statistic(data))

    # Bootstrap distribution
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        boot_stats[i] = statistic(sample)

    # Bias correction factor z0
    z0 = stats.norm.ppf(np.mean(boot_stats < theta_hat))

    # Acceleration factor via jackknife
    jack_stats = np.empty(n)
    for i in range(n):
        jack_sample = np.delete(data, i)
        jack_stats[i] = statistic(jack_sample)
    jack_mean = np.mean(jack_stats)
    diffs = jack_mean - jack_stats
    a_hat = np.sum(diffs ** 3) / (6.0 * (np.sum(diffs ** 2)) ** 1.5 + 1e-30)

    alpha = 1 - confidence
    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

    # Adjusted quantiles
    def _adj(z_a: float) -> float:
        num = z0 + z_a
        return float(stats.norm.cdf(z0 + num / (1 - a_hat * num)))

    q_lo = max(0.0, min(_adj(z_alpha_lo), 1.0))
    q_hi = max(0.0, min(_adj(z_alpha_hi), 1.0))

    lower = float(np.percentile(boot_stats, 100 * q_lo))
    upper = float(np.percentile(boot_stats, 100 * q_hi))
    return lower, theta_hat, upper


def block_bootstrap(
    data: np.ndarray,
    block_size: int,
    n_bootstrap: int = 10000,
    statistic: Callable = np.mean,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Block bootstrap for dependent data.

    Returns an array of ``n_bootstrap`` replicated statistics.
    """
    rng = np.random.RandomState(random_state)
    data = np.asarray(data, dtype=float)
    n = len(data)
    n_blocks = int(np.ceil(n / block_size))
    results = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        pieces = [data[s : s + block_size] for s in starts]
        resampled = np.concatenate(pieces)[:n]
        results[i] = statistic(resampled)
    return results


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------
def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    statistic: Callable = lambda a, b: np.mean(a) - np.mean(b),
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """Permutation test.  Returns ``(observed_stat, p_value)``."""
    rng = np.random.RandomState(random_state)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = float(statistic(x, y))
    combined = np.concatenate([x, y])
    nx = len(x)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_stat = statistic(combined[:nx], combined[nx:])
        if alternative == "two-sided":
            if abs(perm_stat) >= abs(observed):
                count += 1
        elif alternative == "greater":
            if perm_stat >= observed:
                count += 1
        elif alternative == "less":
            if perm_stat <= observed:
                count += 1
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
    p_value = (count + 1) / (n_permutations + 1)
    return observed, p_value


# ---------------------------------------------------------------------------
# Multiple testing corrections
# ---------------------------------------------------------------------------
def holm_bonferroni(
    p_values: np.ndarray, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down correction.

    Returns ``(adjusted_p_values, reject_array)``.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    for i, idx in enumerate(order):
        adjusted[idx] = p_values[idx] * (m - i)
    # Enforce monotonicity (step-down)
    sorted_adj = adjusted[order]
    for i in range(1, m):
        sorted_adj[i] = max(sorted_adj[i], sorted_adj[i - 1])
    adjusted[order] = sorted_adj
    adjusted = np.minimum(adjusted, 1.0)
    reject = adjusted <= alpha
    return adjusted, reject


def benjamini_hochberg(
    p_values: np.ndarray, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Returns ``(adjusted_p_values, reject_array)``.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    for i, idx in enumerate(order):
        adjusted[idx] = p_values[idx] * m / (i + 1)
    # Enforce monotonicity (step-up: traverse from largest to smallest)
    sorted_adj = adjusted[order]
    for i in range(m - 2, -1, -1):
        sorted_adj[i] = min(sorted_adj[i], sorted_adj[i + 1])
    adjusted[order] = sorted_adj
    adjusted = np.minimum(adjusted, 1.0)
    reject = adjusted <= alpha
    return adjusted, reject


# ---------------------------------------------------------------------------
# Combining p-values
# ---------------------------------------------------------------------------
def fisher_combine_pvalues(
    p_values: np.ndarray,
) -> Tuple[float, float]:
    """Fisher's method to combine p-values.

    Returns ``(chi2_stat, combined_p)``.
    """
    p_values = np.asarray(p_values, dtype=float)
    p_values = np.clip(p_values, 1e-300, 1.0)
    chi2_stat = -2.0 * np.sum(np.log(p_values))
    df = 2 * len(p_values)
    combined_p = float(stats.chi2.sf(chi2_stat, df))
    return float(chi2_stat), combined_p


def stouffer_combine_pvalues(
    p_values: np.ndarray, weights: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Stouffer's Z-method.  Returns ``(Z, combined_p)``."""
    p_values = np.asarray(p_values, dtype=float)
    p_values = np.clip(p_values, 1e-300, 1.0 - 1e-15)
    z_scores = stats.norm.ppf(1 - p_values)
    if weights is None:
        weights = np.ones(len(p_values))
    else:
        weights = np.asarray(weights, dtype=float)
    z_combined = float(np.sum(weights * z_scores) / np.sqrt(np.sum(weights ** 2)))
    combined_p = float(stats.norm.sf(z_combined))
    return z_combined, combined_p


def harmonic_mean_pvalue(p_values: np.ndarray) -> float:
    """Harmonic mean p-value for combining tests."""
    p_values = np.asarray(p_values, dtype=float)
    p_values = np.clip(p_values, 1e-300, 1.0)
    n = len(p_values)
    hmp = float(n / np.sum(1.0 / p_values))
    return hmp


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------
def effect_size_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d effect size (pooled std)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    pooled_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (
        nx + ny - 2
    )
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def effect_size_glass_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Glass's delta (using control group *y* standard deviation)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = np.std(y, ddof=1)
    if s == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / s)


# ---------------------------------------------------------------------------
# Newey-West HAC standard errors
# ---------------------------------------------------------------------------
def newey_west_se(
    residuals: np.ndarray, X: np.ndarray, max_lag: Optional[int] = None
) -> np.ndarray:
    """Newey-West heteroskedasticity- and autocorrelation-consistent SEs.

    Parameters
    ----------
    residuals : (n,) array of OLS residuals
    X : (n, k) design matrix
    max_lag : int or None – defaults to ``floor(n^(1/3))``

    Returns
    -------
    se : (k,) array of standard errors
    """
    residuals = np.asarray(residuals, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape
    if max_lag is None:
        max_lag = int(np.floor(n ** (1.0 / 3.0)))

    # Meat of the sandwich: Γ_0 + Σ_{j=1}^{L} w_j (Γ_j + Γ_j')
    # where Γ_j = (1/n) Σ_t X_t e_t e_{t-j} X_{t-j}'
    Xe = X * residuals[:, np.newaxis]
    S = (Xe.T @ Xe) / n  # Γ_0
    for j in range(1, max_lag + 1):
        w = 1 - j / (max_lag + 1)  # Bartlett kernel
        gamma_j = (Xe[j:].T @ Xe[:-j]) / n
        S += w * (gamma_j + gamma_j.T)

    # Bread: (X'X / n)^{-1}
    bread_inv = np.linalg.inv(X.T @ X / n)
    V = bread_inv @ S @ bread_inv / n
    return np.sqrt(np.diag(V))


# ---------------------------------------------------------------------------
# Density & distribution helpers
# ---------------------------------------------------------------------------
def kernel_density_estimate(
    data: np.ndarray,
    bandwidth: Optional[float] = None,
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian KDE.  Returns ``(x_grid, density)``."""
    data = np.asarray(data, dtype=float)
    if bandwidth is not None:
        kde = stats.gaussian_kde(data, bw_method=bandwidth)
    else:
        kde = stats.gaussian_kde(data)
    x_min, x_max = data.min(), data.max()
    margin = 0.1 * (x_max - x_min) if x_max > x_min else 1.0
    x_grid = np.linspace(x_min - margin, x_max + margin, n_points)
    density = kde(x_grid)
    return x_grid, density


def empirical_cdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the empirical CDF.  Returns ``(sorted_data, cdf_values)``."""
    data = np.sort(np.asarray(data, dtype=float))
    cdf = np.arange(1, len(data) + 1) / len(data)
    return data, cdf


def ks_test_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov test statistic."""
    stat, _ = stats.ks_2samp(x, y)
    return float(stat)


def anderson_darling_test(
    data: np.ndarray, dist: str = "norm"
) -> Tuple[float, float]:
    """Anderson-Darling goodness-of-fit test.

    Returns ``(statistic, estimated_p_value)``.  An approximate p-value
    is derived from the critical-value table provided by SciPy; the
    exact p-value is not available for all distributions.
    """
    result = stats.anderson(data, dist=dist)
    stat = float(result.statistic)
    # Interpolate an approximate significance level
    sig_levels = np.array(result.significance_level) / 100.0  # e.g. [15, 10, 5, 2.5, 1] -> fracs
    crit_vals = np.array(result.critical_values)
    if stat >= crit_vals[-1]:
        p_approx = sig_levels[-1]
    elif stat <= crit_vals[0]:
        p_approx = sig_levels[0]
    else:
        p_approx = float(np.interp(stat, crit_vals, sig_levels))
    return stat, p_approx


# ---------------------------------------------------------------------------
# Autocorrelation diagnostics
# ---------------------------------------------------------------------------
def ljung_box_test(
    residuals: np.ndarray, max_lag: int = 10
) -> Tuple[float, float]:
    """Ljung-Box portmanteau test for autocorrelation.

    Returns ``(Q, p_value)`` for the specified *max_lag*.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    acf_vals = np.correlate(residuals - np.mean(residuals), residuals - np.mean(residuals), mode="full")
    acf_vals = acf_vals[n - 1:]  # keep non-negative lags
    acf_vals = acf_vals / acf_vals[0]  # normalise

    Q = 0.0
    for k in range(1, max_lag + 1):
        if k < len(acf_vals):
            Q += acf_vals[k] ** 2 / (n - k)
    Q *= n * (n + 2)
    p_value = float(stats.chi2.sf(Q, max_lag))
    return float(Q), p_value


def durbin_watson(residuals: np.ndarray) -> float:
    """Durbin-Watson statistic for first-order autocorrelation."""
    residuals = np.asarray(residuals, dtype=float)
    diff = np.diff(residuals)
    return float(np.sum(diff ** 2) / np.sum(residuals ** 2))


# ---------------------------------------------------------------------------
# Cochrane-Orcutt iterative GLS
# ---------------------------------------------------------------------------
def cochrane_orcutt(
    y: np.ndarray,
    X: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Cochrane-Orcutt iterative procedure for AR(1) errors.

    Returns ``(coefficients, rho)`` where *rho* is the estimated
    first-order autocorrelation of the residuals.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Initial OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rho = 0.0

    for _ in range(max_iter):
        resid = y - X @ beta
        # Estimate rho from residuals
        rho_new = float(np.sum(resid[1:] * resid[:-1]) / np.sum(resid[:-1] ** 2))
        # Transform data
        y_star = y[1:] - rho_new * y[:-1]
        X_star = X[1:] - rho_new * X[:-1]
        beta_new = np.linalg.lstsq(X_star, y_star, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol and abs(rho_new - rho) < tol:
            return beta_new, rho_new
        beta = beta_new
        rho = rho_new

    return beta, rho


# ---------------------------------------------------------------------------
# Likelihood-ratio & information criteria
# ---------------------------------------------------------------------------
def log_likelihood_ratio(
    ll_restricted: float, ll_unrestricted: float, df: int
) -> Tuple[float, float]:
    """Likelihood ratio test.

    Returns ``(LR_stat, p_value)`` where ``LR = -2(ll_r - ll_u)``
    follows a χ² distribution with *df* degrees of freedom.
    """
    lr = -2.0 * (ll_restricted - ll_unrestricted)
    p = float(stats.chi2.sf(lr, df))
    return float(lr), p


def information_criterion(
    ll: float, k: int, n: int, criterion: str = "bic"
) -> float:
    """AIC or BIC.

    Parameters
    ----------
    ll : log-likelihood
    k : number of estimated parameters
    n : number of observations
    criterion : ``"aic"`` or ``"bic"``
    """
    if criterion == "aic":
        return -2.0 * ll + 2.0 * k
    if criterion == "bic":
        return -2.0 * ll + k * np.log(n)
    raise ValueError(f"Unknown criterion: {criterion}")


# ---------------------------------------------------------------------------
# Moving-block bootstrap variance
# ---------------------------------------------------------------------------
def moving_block_bootstrap_variance(
    data: np.ndarray,
    block_size: int,
    statistic: Callable = np.mean,
    n_bootstrap: int = 5000,
) -> float:
    """Variance estimate via moving block bootstrap."""
    boot_stats = block_bootstrap(
        data, block_size, n_bootstrap=n_bootstrap, statistic=statistic
    )
    return float(np.var(boot_stats, ddof=1))

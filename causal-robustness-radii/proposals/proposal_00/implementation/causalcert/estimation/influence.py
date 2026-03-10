"""
Influence function computation for semiparametric estimators.

Computes the efficient influence function for the ATE under the AIPW
estimator, enabling variance estimation and sensitivity analysis.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Core influence functions
# ---------------------------------------------------------------------------


def influence_function(
    y: np.ndarray,
    t: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    e: np.ndarray,
) -> np.ndarray:
    """Compute the efficient influence function for the ATE.

    The influence function ψ(O) for the ATE under AIPW is::

        ψ_i = T_i(Y_i - μ₁_i)/e_i - (1-T_i)(Y_i - μ₀_i)/(1-e_i) + (μ₁_i - μ₀_i) - τ

    where τ = E[ψ_i + τ] = mean of the un-centred scores.

    Parameters
    ----------
    y : np.ndarray
        Observed outcomes, shape ``(n,)``.
    t : np.ndarray
        Treatment assignments (binary), shape ``(n,)``.
    mu0 : np.ndarray
        Predicted E[Y|T=0, X], shape ``(n,)``.
    mu1 : np.ndarray
        Predicted E[Y|T=1, X], shape ``(n,)``.
    e : np.ndarray
        Estimated propensity scores, shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Influence function values, shape ``(n,)``.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    mu0 = np.asarray(mu0, dtype=np.float64)
    mu1 = np.asarray(mu1, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)

    # Un-centred AIPW score
    scores = (
        t * (y - mu1) / e
        - (1.0 - t) * (y - mu0) / (1.0 - e)
        + (mu1 - mu0)
    )
    # Centre around the ATE estimate
    tau = float(np.mean(scores))
    psi = scores - tau
    return psi


def influence_function_att(
    y: np.ndarray,
    t: np.ndarray,
    mu0: np.ndarray,
    e: np.ndarray,
) -> np.ndarray:
    """Compute the efficient influence function for the ATT.

    The ATT influence function is::

        ψ_i = (T_i/p) * (Y_i - μ₀_i) - ((1-T_i)*e_i) / (p*(1-e_i)) * (Y_i - μ₀_i) - ATT

    where p = P(T=1).

    Parameters
    ----------
    y : np.ndarray
        Observed outcomes.
    t : np.ndarray
        Treatment assignments.
    mu0 : np.ndarray
        Predicted E[Y|T=0, X].
    e : np.ndarray
        Propensity scores.

    Returns
    -------
    np.ndarray
        Influence function values for ATT.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    mu0 = np.asarray(mu0, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)

    p = float(np.mean(t))
    if p < 1e-12:
        raise ValueError("No treated units found.")

    # Un-centred ATT scores
    scores = (
        (t / p) * (y - mu0)
        - ((1.0 - t) * e / (p * (1.0 - e))) * (y - mu0)
    )
    att = float(np.mean(scores))
    psi = scores - att
    return psi


# ---------------------------------------------------------------------------
# Variance estimation
# ---------------------------------------------------------------------------


def variance_from_influence(psi: np.ndarray) -> float:
    """Estimate the variance of the ATE from influence-function values.

    Uses the sandwich estimator: Var(τ̂) = (1/n) * mean(ψ²).

    Parameters
    ----------
    psi : np.ndarray
        Influence function values, shape ``(n,)``.

    Returns
    -------
    float
        Estimated variance (= mean of ψ²).
    """
    return float(np.mean(psi ** 2))


def standard_error_from_influence(psi: np.ndarray) -> float:
    """Compute the standard error of the ATE from influence function values.

    SE = sqrt(Var(ψ) / n)

    Parameters
    ----------
    psi : np.ndarray
        Influence function values.

    Returns
    -------
    float
        Standard error.
    """
    n = len(psi)
    return float(np.sqrt(variance_from_influence(psi) / n))


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def confidence_interval(
    ate: float,
    psi: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute a Wald confidence interval for the ATE.

    CI = τ̂ ± z_{1-α/2} * SE, where SE = sqrt(mean(ψ²)/n).

    Parameters
    ----------
    ate : float
        Point estimate.
    psi : np.ndarray
        Influence function values.
    alpha : float
        Significance level.

    Returns
    -------
    tuple[float, float]
        ``(lower, upper)`` confidence bounds.
    """
    se = standard_error_from_influence(psi)
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return (ate - z * se, ate + z * se)


def bootstrap_confidence_interval(
    y: np.ndarray,
    t: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    e: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a bootstrap percentile confidence interval for the ATE.

    Parameters
    ----------
    y, t, mu0, mu1, e : np.ndarray
        As in :func:`influence_function`.
    alpha : float
        Significance level.
    n_bootstrap : int
        Number of bootstrap replicates.
    seed : int
        Random seed.

    Returns
    -------
    tuple[float, float, float]
        ``(ate, lower, upper)``.
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    # AIPW point estimate from full data
    scores = (
        t * (y - mu1) / e
        - (1.0 - t) * (y - mu0) / (1.0 - e)
        + (mu1 - mu0)
    )
    ate = float(np.mean(scores))

    boot_ates = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_scores = (
            t[idx] * (y[idx] - mu1[idx]) / e[idx]
            - (1.0 - t[idx]) * (y[idx] - mu0[idx]) / (1.0 - e[idx])
            + (mu1[idx] - mu0[idx])
        )
        boot_ates[b] = float(np.mean(boot_scores))

    lower = float(np.percentile(boot_ates, 100 * alpha / 2))
    upper = float(np.percentile(boot_ates, 100 * (1 - alpha / 2)))
    return ate, lower, upper


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def influence_diagnostics(
    psi: np.ndarray,
    threshold_factor: float = 3.0,
) -> dict[str, object]:
    """Diagnostics from influence function values.

    Parameters
    ----------
    psi : np.ndarray
        Influence function values.
    threshold_factor : float
        Observations with |ψ| > threshold_factor * std(ψ) are flagged
        as high-leverage.

    Returns
    -------
    dict[str, object]
        Contains:
        - ``"mean"``: mean of ψ (should be near 0).
        - ``"std"``: standard deviation of ψ.
        - ``"max_abs"``: maximum |ψ|.
        - ``"n_leverage"``: number of high-leverage points.
        - ``"leverage_indices"``: indices of high-leverage points.
        - ``"skewness"``: skewness of ψ.
        - ``"kurtosis"``: excess kurtosis of ψ.
    """
    psi = np.asarray(psi, dtype=np.float64)
    std = float(np.std(psi))
    threshold = threshold_factor * std
    leverage_mask = np.abs(psi) > threshold

    return {
        "mean": float(np.mean(psi)),
        "std": std,
        "max_abs": float(np.max(np.abs(psi))),
        "n_leverage": int(np.sum(leverage_mask)),
        "leverage_indices": np.nonzero(leverage_mask)[0],
        "skewness": float(stats.skew(psi)),
        "kurtosis": float(stats.kurtosis(psi)),
    }


def semiparametric_efficiency_bound(
    psi: np.ndarray,
) -> float:
    """Estimate the semiparametric efficiency bound from influence function.

    The efficiency bound for the ATE equals Var(ψ), where ψ is the
    efficient influence function.

    Parameters
    ----------
    psi : np.ndarray
        Efficient influence function values.

    Returns
    -------
    float
        Estimated semiparametric efficiency bound.
    """
    return float(np.var(psi, ddof=0))

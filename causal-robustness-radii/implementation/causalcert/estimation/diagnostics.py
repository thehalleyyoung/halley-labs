"""
Estimation diagnostics for causal inference.

Provides overlap assessment, positivity violation detection, covariate
balance checks (standardised mean differences), ASMD before/after
weighting, Love plot data generation, residual analysis, and influence
point detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import stats as sp_stats


# ===================================================================
# Result data structures
# ===================================================================


@dataclass(frozen=True, slots=True)
class OverlapDiagnostics:
    """Propensity score overlap diagnostics.

    Attributes
    ----------
    mean_treated : float
        Mean PS in treated arm.
    mean_control : float
        Mean PS in control arm.
    std_treated : float
        Std of PS in treated arm.
    std_control : float
        Std of PS in control arm.
    ks_statistic : float
        Kolmogorov-Smirnov statistic comparing PS distributions.
    ks_p_value : float
        KS test p-value.
    overlap_coefficient : float
        Estimated overlap coefficient (area under min of two densities).
    ess_treated : float
        Effective sample size (treated).
    ess_control : float
        Effective sample size (control).
    n_violations : int
        Number of positivity violations.
    """

    mean_treated: float
    mean_control: float
    std_treated: float
    std_control: float
    ks_statistic: float
    ks_p_value: float
    overlap_coefficient: float
    ess_treated: float
    ess_control: float
    n_violations: int


@dataclass(frozen=True, slots=True)
class BalanceResult:
    """Covariate balance diagnostics.

    Attributes
    ----------
    covariate_names : tuple[str, ...]
        Names of covariates.
    smd_unadjusted : np.ndarray
        Standardised mean differences before weighting.
    smd_adjusted : np.ndarray
        Standardised mean differences after weighting.
    variance_ratios : np.ndarray
        Variance ratios (treated / control) per covariate.
    max_smd_unadjusted : float
        Maximum absolute unadjusted SMD.
    max_smd_adjusted : float
        Maximum absolute adjusted SMD.
    n_imbalanced : int
        Number of covariates with |SMD| > threshold after weighting.
    """

    covariate_names: tuple[str, ...]
    smd_unadjusted: np.ndarray
    smd_adjusted: np.ndarray
    variance_ratios: np.ndarray
    max_smd_unadjusted: float
    max_smd_adjusted: float
    n_imbalanced: int


@dataclass(frozen=True, slots=True)
class ResidualDiagnostics:
    """Outcome model residual diagnostics.

    Attributes
    ----------
    mean_residual : float
        Mean of residuals.
    std_residual : float
        Standard deviation of residuals.
    skewness : float
        Skewness of residuals.
    kurtosis : float
        Excess kurtosis of residuals.
    shapiro_stat : float
        Shapiro-Wilk test statistic (on subsample if n > 5000).
    shapiro_p : float
        Shapiro-Wilk p-value.
    n_outliers : int
        Number of residuals > 3σ from zero.
    durbin_watson : float
        Durbin-Watson statistic for autocorrelation.
    """

    mean_residual: float
    std_residual: float
    skewness: float
    kurtosis: float
    shapiro_stat: float
    shapiro_p: float
    n_outliers: int
    durbin_watson: float


@dataclass(frozen=True, slots=True)
class InfluencePointResult:
    """Influence point detection result.

    Attributes
    ----------
    n_influential : int
        Number of influential observations detected.
    influential_indices : np.ndarray
        Row indices of influential observations.
    influence_scores : np.ndarray
        Influence metric for each observation.
    threshold : float
        Threshold used for detection.
    max_influence : float
        Maximum influence score.
    """

    n_influential: int
    influential_indices: np.ndarray
    influence_scores: np.ndarray
    threshold: float
    max_influence: float


# ===================================================================
# 1. Overlap assessment
# ===================================================================


def assess_overlap(
    e: np.ndarray,
    A: np.ndarray,
    *,
    violation_threshold: float = 0.05,
) -> OverlapDiagnostics:
    """Assess propensity score overlap between treatment arms.

    Parameters
    ----------
    e : np.ndarray
        Propensity scores P(A=1|X), shape ``(n,)``.
    A : np.ndarray
        Treatment assignments (binary), shape ``(n,)``.
    violation_threshold : float
        Threshold for positivity violations (e < threshold or
        e > 1 − threshold).

    Returns
    -------
    OverlapDiagnostics
    """
    e = np.asarray(e, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64).ravel()

    mask1 = A == 1
    mask0 = A == 0
    e1 = e[mask1]
    e0 = e[mask0]

    # KS test
    if len(e1) > 0 and len(e0) > 0:
        ks_stat, ks_p = sp_stats.ks_2samp(e1, e0)
    else:
        ks_stat, ks_p = float("nan"), float("nan")

    # Overlap coefficient via histogram approximation
    overlap_coeff = _overlap_coefficient(e1, e0)

    # Effective sample size (Kish)
    if len(e1) > 0:
        w1 = 1.0 / e1
        ess1 = float(w1.sum() ** 2 / (w1 ** 2).sum())
    else:
        ess1 = 0.0
    if len(e0) > 0:
        w0 = 1.0 / (1.0 - e0)
        ess0 = float(w0.sum() ** 2 / (w0 ** 2).sum())
    else:
        ess0 = 0.0

    # Positivity violations
    n_violations = int(
        np.sum((e < violation_threshold) | (e > 1.0 - violation_threshold))
    )

    return OverlapDiagnostics(
        mean_treated=float(np.mean(e1)) if len(e1) > 0 else float("nan"),
        mean_control=float(np.mean(e0)) if len(e0) > 0 else float("nan"),
        std_treated=float(np.std(e1)) if len(e1) > 0 else float("nan"),
        std_control=float(np.std(e0)) if len(e0) > 0 else float("nan"),
        ks_statistic=ks_stat,
        ks_p_value=ks_p,
        overlap_coefficient=overlap_coeff,
        ess_treated=ess1,
        ess_control=ess0,
        n_violations=n_violations,
    )


def _overlap_coefficient(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 100,
) -> float:
    """Estimate the overlap coefficient via histogram approximation."""
    if len(x) == 0 or len(y) == 0:
        return 0.0
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    if hi - lo < 1e-12:
        return 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    h_x, _ = np.histogram(x, bins=edges, density=True)
    h_y, _ = np.histogram(y, bins=edges, density=True)
    bin_width = (hi - lo) / n_bins
    return float(np.sum(np.minimum(h_x, h_y)) * bin_width)


# ===================================================================
# 2. Positivity violation detection
# ===================================================================


def detect_positivity_violations(
    e: np.ndarray,
    *,
    lower: float = 0.025,
    upper: float = 0.975,
) -> dict[str, Any]:
    """Detect positivity (overlap) violations.

    Parameters
    ----------
    e : np.ndarray
        Propensity scores, shape ``(n,)``.
    lower : float
        Lower threshold.
    upper : float
        Upper threshold.

    Returns
    -------
    dict[str, Any]
        Contains violation counts, indices, and severity measures.
    """
    e = np.asarray(e, dtype=np.float64).ravel()

    low_mask = e < lower
    high_mask = e > upper
    violation_mask = low_mask | high_mask

    return {
        "n_violations": int(np.sum(violation_mask)),
        "n_low": int(np.sum(low_mask)),
        "n_high": int(np.sum(high_mask)),
        "fraction_violations": float(np.mean(violation_mask)),
        "violation_indices": np.where(violation_mask)[0],
        "min_propensity": float(np.min(e)),
        "max_propensity": float(np.max(e)),
        "severity": float(np.mean(np.maximum(lower - e, 0) + np.maximum(e - upper, 0))),
    }


# ===================================================================
# 3. Covariate balance checks
# ===================================================================


def standardized_mean_difference(
    X: np.ndarray,
    A: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    covariate_names: Sequence[str] | None = None,
    threshold: float = 0.1,
) -> BalanceResult:
    """Compute absolute standardised mean differences (ASMD).

    For each covariate j::

        SMD_j = (mean_treated_j − mean_control_j) / sqrt((s²_t + s²_c) / 2)

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    A : np.ndarray
        Treatment assignments (binary), shape ``(n,)``.
    weights : np.ndarray or None
        IPW weights. If provided, computes weighted SMDs.
    covariate_names : Sequence[str] or None
        Names for each covariate column.
    threshold : float
        Imbalance threshold for |SMD|.

    Returns
    -------
    BalanceResult
    """
    X = np.asarray(X, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape

    mask1 = A == 1
    mask0 = A == 0

    if covariate_names is None:
        covariate_names = tuple(f"X{j}" for j in range(p))
    else:
        covariate_names = tuple(covariate_names)

    # Unadjusted SMD
    smd_unadj = np.empty(p, dtype=np.float64)
    var_ratios = np.empty(p, dtype=np.float64)
    for j in range(p):
        x1 = X[mask1, j]
        x0 = X[mask0, j]
        m1, m0 = float(np.mean(x1)), float(np.mean(x0))
        s1, s0 = float(np.var(x1, ddof=1)), float(np.var(x0, ddof=1))
        pooled_s = np.sqrt((s1 + s0) / 2.0)
        smd_unadj[j] = (m1 - m0) / max(pooled_s, 1e-12)
        var_ratios[j] = s1 / max(s0, 1e-12)

    # Adjusted (weighted) SMD
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        smd_adj = np.empty(p, dtype=np.float64)
        for j in range(p):
            wm1 = _weighted_mean(X[mask1, j], w[mask1])
            wm0 = _weighted_mean(X[mask0, j], w[mask0])
            x1 = X[mask1, j]
            x0 = X[mask0, j]
            s1 = float(np.var(x1, ddof=1))
            s0 = float(np.var(x0, ddof=1))
            pooled_s = np.sqrt((s1 + s0) / 2.0)
            smd_adj[j] = (wm1 - wm0) / max(pooled_s, 1e-12)
    else:
        smd_adj = smd_unadj.copy()

    max_unadj = float(np.max(np.abs(smd_unadj)))
    max_adj = float(np.max(np.abs(smd_adj)))
    n_imbalanced = int(np.sum(np.abs(smd_adj) > threshold))

    return BalanceResult(
        covariate_names=covariate_names,
        smd_unadjusted=smd_unadj,
        smd_adjusted=smd_adj,
        variance_ratios=var_ratios,
        max_smd_unadjusted=max_unadj,
        max_smd_adjusted=max_adj,
        n_imbalanced=n_imbalanced,
    )


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted mean."""
    total = float(np.sum(w))
    if total < 1e-12:
        return float(np.mean(x))
    return float(np.sum(w * x) / total)


# ===================================================================
# 4. ASMD before/after weighting
# ===================================================================


def asmd_comparison(
    X: np.ndarray,
    A: np.ndarray,
    weights: np.ndarray,
    *,
    covariate_names: Sequence[str] | None = None,
    threshold: float = 0.1,
) -> dict[str, Any]:
    """Compare ASMD before and after IPW weighting.

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix.
    A : np.ndarray
        Treatment assignments.
    weights : np.ndarray
        IPW weights.
    covariate_names : Sequence[str] or None
        Names for covariates.
    threshold : float
        Balance threshold.

    Returns
    -------
    dict[str, Any]
        Comparison results including before/after SMDs and improvement.
    """
    bal_before = standardized_mean_difference(
        X, A, covariate_names=covariate_names, threshold=threshold,
    )
    bal_after = standardized_mean_difference(
        X, A, weights=weights, covariate_names=covariate_names,
        threshold=threshold,
    )

    improvement = np.abs(bal_before.smd_unadjusted) - np.abs(bal_after.smd_adjusted)

    return {
        "covariate_names": bal_before.covariate_names,
        "smd_before": bal_before.smd_unadjusted,
        "smd_after": bal_after.smd_adjusted,
        "improvement": improvement,
        "max_smd_before": bal_before.max_smd_unadjusted,
        "max_smd_after": bal_after.max_smd_adjusted,
        "n_imbalanced_before": int(np.sum(np.abs(bal_before.smd_unadjusted) > threshold)),
        "n_imbalanced_after": bal_after.n_imbalanced,
        "all_balanced_after": bal_after.n_imbalanced == 0,
    }


# ===================================================================
# 5. Love plot data
# ===================================================================


def love_plot_data(
    X: np.ndarray,
    A: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    covariate_names: Sequence[str] | None = None,
    threshold: float = 0.1,
) -> dict[str, Any]:
    """Generate data for a Love plot.

    A Love plot shows absolute SMDs per covariate, optionally before
    and after adjustment.

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix.
    A : np.ndarray
        Treatment assignments.
    weights : np.ndarray or None
        IPW weights for the "after" comparison.
    covariate_names : Sequence[str] or None
        Names for covariates.
    threshold : float
        Balance threshold for the reference line.

    Returns
    -------
    dict[str, Any]
        Contains ``"covariates"``, ``"smd_unadjusted"``,
        ``"smd_adjusted"`` (if weights provided), and ``"threshold"``.
    """
    bal = standardized_mean_difference(
        X, A, weights=weights, covariate_names=covariate_names,
        threshold=threshold,
    )

    result: dict[str, Any] = {
        "covariates": list(bal.covariate_names),
        "smd_unadjusted": np.abs(bal.smd_unadjusted),
        "threshold": threshold,
    }

    if weights is not None:
        result["smd_adjusted"] = np.abs(bal.smd_adjusted)

    # Sort by unadjusted SMD for visual clarity
    order = np.argsort(-np.abs(bal.smd_unadjusted))
    result["sort_order"] = order

    return result


# ===================================================================
# 6. Residual analysis
# ===================================================================


def residual_diagnostics(
    Y: np.ndarray,
    Y_hat: np.ndarray,
) -> ResidualDiagnostics:
    """Compute residual diagnostics for an outcome model.

    Parameters
    ----------
    Y : np.ndarray
        Observed outcomes, shape ``(n,)``.
    Y_hat : np.ndarray
        Predicted outcomes, shape ``(n,)``.

    Returns
    -------
    ResidualDiagnostics
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    Y_hat = np.asarray(Y_hat, dtype=np.float64).ravel()

    residuals = Y - Y_hat
    n = len(residuals)

    mean_r = float(np.mean(residuals))
    std_r = float(np.std(residuals, ddof=1))
    skew_r = float(sp_stats.skew(residuals))
    kurt_r = float(sp_stats.kurtosis(residuals))

    # Shapiro-Wilk on subsample
    max_shapiro = min(n, 5000)
    if n > max_shapiro:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_shapiro, replace=False)
        sw_stat, sw_p = sp_stats.shapiro(residuals[idx])
    else:
        sw_stat, sw_p = sp_stats.shapiro(residuals)

    # Outliers (> 3σ)
    n_outliers = int(np.sum(np.abs(residuals) > 3.0 * std_r))

    # Durbin-Watson
    diff = np.diff(residuals)
    dw = float(np.sum(diff ** 2) / max(np.sum(residuals ** 2), 1e-12))

    return ResidualDiagnostics(
        mean_residual=mean_r,
        std_residual=std_r,
        skewness=skew_r,
        kurtosis=kurt_r,
        shapiro_stat=float(sw_stat),
        shapiro_p=float(sw_p),
        n_outliers=n_outliers,
        durbin_watson=dw,
    )


def residual_by_arm(
    Y: np.ndarray,
    Y_hat: np.ndarray,
    A: np.ndarray,
) -> dict[str, ResidualDiagnostics]:
    """Compute residual diagnostics separately per treatment arm.

    Parameters
    ----------
    Y : np.ndarray
        Observed outcomes.
    Y_hat : np.ndarray
        Predicted outcomes.
    A : np.ndarray
        Treatment assignments.

    Returns
    -------
    dict[str, ResidualDiagnostics]
        Keys: ``"treated"`` and ``"control"``.
    """
    A = np.asarray(A, dtype=np.float64).ravel()
    mask1 = A == 1
    mask0 = A == 0
    return {
        "treated": residual_diagnostics(Y[mask1], Y_hat[mask1]),
        "control": residual_diagnostics(Y[mask0], Y_hat[mask0]),
    }


# ===================================================================
# 7. Influence point detection
# ===================================================================


def detect_influence_points(
    psi: np.ndarray,
    *,
    method: str = "threshold",
    threshold_factor: float = 3.0,
    quantile: float = 0.99,
) -> InfluencePointResult:
    """Detect influential observations from influence function values.

    Parameters
    ----------
    psi : np.ndarray
        Influence function values, shape ``(n,)``.
    method : str
        Detection method:
        - ``"threshold"``: flag |ψ| > factor × std(ψ)
        - ``"quantile"``: flag |ψ| above quantile
        - ``"cook"``: Cook's-distance-like metric |ψ|² / mean(ψ²)
    threshold_factor : float
        Factor for ``"threshold"`` method.
    quantile : float
        Quantile for ``"quantile"`` method.

    Returns
    -------
    InfluencePointResult
    """
    psi = np.asarray(psi, dtype=np.float64).ravel()
    abs_psi = np.abs(psi)

    if method == "threshold":
        std_psi = float(np.std(psi))
        thresh = threshold_factor * std_psi
        mask = abs_psi > thresh
        scores = abs_psi / max(std_psi, 1e-12)
    elif method == "quantile":
        thresh = float(np.quantile(abs_psi, quantile))
        mask = abs_psi > thresh
        scores = abs_psi / max(float(np.median(abs_psi)), 1e-12)
    elif method == "cook":
        mean_sq = float(np.mean(psi ** 2))
        scores = psi ** 2 / max(mean_sq, 1e-12)
        thresh = 4.0 / len(psi)  # 4/n rule of thumb
        mask = scores > thresh
    else:
        raise ValueError(f"Unknown method: {method!r}")

    indices = np.where(mask)[0]

    return InfluencePointResult(
        n_influential=int(np.sum(mask)),
        influential_indices=indices,
        influence_scores=scores,
        threshold=thresh,
        max_influence=float(np.max(scores)) if len(scores) > 0 else 0.0,
    )


def leave_one_out_sensitivity(
    psi: np.ndarray,
    estimate: float,
) -> dict[str, Any]:
    """Compute leave-one-out sensitivity analysis.

    For each observation, compute the estimate that would be obtained
    without that observation.

    Parameters
    ----------
    psi : np.ndarray
        Influence function values, shape ``(n,)``.
    estimate : float
        Full-sample estimate.

    Returns
    -------
    dict[str, Any]
        Contains ``"loo_estimates"`` and ``"max_change"``.
    """
    psi = np.asarray(psi, dtype=np.float64).ravel()
    n = len(psi)

    # LOO estimate: θ̂_{-i} ≈ θ̂ − ψ_i / n
    loo = estimate - psi / n

    return {
        "loo_estimates": loo,
        "max_change": float(np.max(np.abs(loo - estimate))),
        "mean_change": float(np.mean(np.abs(loo - estimate))),
        "most_influential_idx": int(np.argmax(np.abs(loo - estimate))),
    }

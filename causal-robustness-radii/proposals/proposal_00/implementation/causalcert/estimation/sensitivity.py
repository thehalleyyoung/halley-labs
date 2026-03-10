"""Sensitivity analysis for causal inference.

Implements E-value computation, omitted variable bias (Cinelli & Hazlett 2020),
Rosenbaum bounds, sensitivity contour data, breakdown point analysis,
combined structural + parametric robustness, calibration against observed
covariates, and bias amplification formulas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class SensitivityResult:
    """Container for sensitivity analysis output."""
    estimate: float
    lower_ci: float
    upper_ci: float
    e_value: float = 0.0
    e_value_ci: float = 0.0
    rv: float = 0.0
    rv_alpha: float = 0.0
    breakdown_point: float = 0.0
    contour_data: Optional[Dict[str, np.ndarray]] = None
    method: str = ""


@dataclass
class RosenbaumResult:
    """Output of a Rosenbaum bounds analysis."""
    gamma_values: np.ndarray
    upper_p_values: np.ndarray
    critical_gamma: float = 0.0
    alpha: float = 0.05


# ===================================================================
# 1.  E-value computation (VanderWeele & Ding 2017)
# ===================================================================

def e_value_rr(rr: float) -> float:
    """E-value for a risk-ratio point estimate.

    E = RR + sqrt(RR × (RR − 1))  for RR ≥ 1.
    If RR < 1, compute for 1/RR and return.

    Reference: VanderWeele & Ding (2017).
    """
    if rr < 0:
        raise ValueError("Risk ratio must be non-negative")
    if rr < 1:
        rr = 1.0 / rr if rr > 0 else 1.0
    return rr + math.sqrt(rr * (rr - 1.0))


def e_value_or(odds_ratio: float, prevalence: float = 0.5) -> float:
    """E-value for an odds ratio, approximately converted to RR.

    Uses the square-root approximation: RR ≈ OR^{0.5} when prevalence
    is moderate, or the exact conversion when prevalence is supplied.
    """
    if odds_ratio <= 0:
        raise ValueError("Odds ratio must be positive")
    if prevalence <= 0 or prevalence >= 1:
        raise ValueError("Prevalence must be in (0, 1)")

    rr = odds_ratio / (1.0 - prevalence + prevalence * odds_ratio)
    return e_value_rr(rr)


def e_value_hr(hazard_ratio: float) -> float:
    """E-value for a hazard ratio (approximated as RR)."""
    return e_value_rr(hazard_ratio)


def e_value_diff(
    estimate: float,
    se: float,
    *,
    sd_outcome: float = 1.0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """E-value for a mean-difference estimate (converted to approximate RR).

    Parameters
    ----------
    estimate : float
        Point estimate of the causal effect (mean difference).
    se : float
        Standard error.
    sd_outcome : float
        Standard deviation of the outcome.
    alpha : float
        Significance level for CI-based E-value.

    Returns
    -------
    (e_value_point, e_value_ci)
    """
    d = abs(estimate) / sd_outcome
    rr_approx = math.exp(0.91 * d)

    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lower = abs(estimate) - z * se
    if ci_lower <= 0:
        rr_ci = 1.0
    else:
        d_ci = ci_lower / sd_outcome
        rr_ci = math.exp(0.91 * d_ci)

    return e_value_rr(rr_approx), e_value_rr(rr_ci)


# ===================================================================
# 2.  Omitted variable bias (Cinelli & Hazlett 2020)
# ===================================================================

def partial_r2(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    treatment_col: int = 0,
) -> float:
    """Partial R² of the treatment variable in regression Y ~ X + Z."""
    n = Y.shape[0]
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    XZ = np.column_stack([X, Z])
    Z_only = Z.copy()

    X_aug_full = np.column_stack([np.ones(n), XZ])
    X_aug_rest = np.column_stack([np.ones(n), Z_only])

    try:
        beta_full = np.linalg.lstsq(X_aug_full, Y, rcond=None)[0]
        rss_full = float(np.sum((Y - X_aug_full @ beta_full) ** 2))
    except np.linalg.LinAlgError:
        rss_full = float(np.sum(Y ** 2))

    try:
        beta_rest = np.linalg.lstsq(X_aug_rest, Y, rcond=None)[0]
        rss_rest = float(np.sum((Y - X_aug_rest @ beta_rest) ** 2))
    except np.linalg.LinAlgError:
        rss_rest = float(np.sum(Y ** 2))

    if rss_rest < 1e-12:
        return 0.0
    return (rss_rest - rss_full) / rss_rest


def robustness_value(
    Y: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    *,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Compute the Robustness Value (RV) and RV_α.

    The RV is the minimum strength of association (in partial R² terms)
    that an omitted confounder must have with both the treatment and
    outcome to fully explain away the estimated effect.

    Returns (RV, RV_alpha).
    """
    n = Y.shape[0]
    if treatment.ndim == 1:
        treatment = treatment.reshape(-1, 1)
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    r2_yd_x = partial_r2(Y.ravel(), treatment, covariates)

    f2 = r2_yd_x / (1.0 - r2_yd_x) if r2_yd_x < 1.0 else float("inf")
    rv = 0.5 * (math.sqrt(f2 ** 2 + 4 * f2) - f2)
    rv = min(max(rv, 0.0), 1.0)

    dof = n - covariates.shape[1] - 2
    se_factor = sp_stats.t.ppf(1 - alpha / 2, df=max(dof, 1))
    t_stat = math.sqrt(f2 * dof) if f2 * dof >= 0 else 0.0
    t_adj = max(t_stat - se_factor, 0.0)
    f2_adj = (t_adj ** 2) / max(dof, 1)
    rv_alpha = 0.5 * (math.sqrt(f2_adj ** 2 + 4 * f2_adj) - f2_adj)
    rv_alpha = min(max(rv_alpha, 0.0), 1.0)

    return rv, rv_alpha


def adjusted_estimate(
    estimate: float,
    se: float,
    r2_yz: float,
    r2_dz: float,
) -> float:
    """Bias-adjusted estimate given hypothetical confounder strengths.

    Parameters
    ----------
    estimate : float
        Original treatment effect estimate.
    se : float
        Standard error.
    r2_yz : float
        Partial R² of the confounder with outcome.
    r2_dz : float
        Partial R² of the confounder with treatment.
    """
    bias_factor = math.sqrt(r2_yz * r2_dz / max(1.0 - r2_dz, 1e-12))
    return estimate - bias_factor * abs(estimate)


def ovb_contour_data(
    estimate: float,
    se: float,
    *,
    grid_size: int = 50,
    max_r2: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Generate contour plot data for omitted variable bias.

    Returns arrays for the bias-adjusted estimate as a function of
    R²_{Y~Z|X} and R²_{D~Z|X}.
    """
    r2_yz = np.linspace(0, max_r2, grid_size)
    r2_dz = np.linspace(0, max_r2, grid_size)
    R2_YZ, R2_DZ = np.meshgrid(r2_yz, r2_dz)

    adjusted = np.zeros_like(R2_YZ)
    for i in range(grid_size):
        for j in range(grid_size):
            adjusted[i, j] = adjusted_estimate(
                estimate, se, R2_YZ[i, j], R2_DZ[i, j]
            )

    return {
        "r2_yz": r2_yz,
        "r2_dz": r2_dz,
        "R2_YZ": R2_YZ,
        "R2_DZ": R2_DZ,
        "adjusted_estimate": adjusted,
    }


# ===================================================================
# 3.  Rosenbaum bounds
# ===================================================================

def rosenbaum_bounds(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    *,
    gamma_range: Tuple[float, float] = (1.0, 5.0),
    n_gamma: int = 20,
    alpha: float = 0.05,
) -> RosenbaumResult:
    """Compute Rosenbaum bounds for a matched study.

    For each sensitivity parameter Γ, computes the worst-case p-value
    under hidden bias of magnitude Γ.  The critical Γ is where the
    p-value first exceeds α.

    Reference: Rosenbaum (2002).
    """
    n_t = len(treated_outcomes)
    n_c = len(control_outcomes)
    n_pairs = min(n_t, n_c)
    diffs = treated_outcomes[:n_pairs] - control_outcomes[:n_pairs]

    gammas = np.linspace(gamma_range[0], gamma_range[1], n_gamma)
    p_values = np.zeros(n_gamma)

    t_obs = np.sum(diffs > 0)

    for idx, gamma in enumerate(gammas):
        p_upper = gamma / (1.0 + gamma)
        mean_null = n_pairs * p_upper
        var_null = n_pairs * p_upper * (1.0 - p_upper)
        if var_null < 1e-12:
            p_values[idx] = 1.0
            continue
        z = (t_obs - mean_null) / math.sqrt(var_null)
        p_values[idx] = 1.0 - sp_stats.norm.cdf(z)

    critical_gamma = gammas[-1]
    for idx in range(n_gamma):
        if p_values[idx] > alpha:
            critical_gamma = gammas[idx]
            break

    return RosenbaumResult(
        gamma_values=gammas,
        upper_p_values=p_values,
        critical_gamma=critical_gamma,
        alpha=alpha,
    )


# ===================================================================
# 4.  Sensitivity contour plots data
# ===================================================================

def sensitivity_contour(
    estimate: float,
    se: float,
    *,
    confound_treatment_range: Tuple[float, float] = (0.0, 0.5),
    confound_outcome_range: Tuple[float, float] = (0.0, 0.5),
    grid_size: int = 40,
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Generate sensitivity contour data for the estimated effect.

    Contours show:
      - Bias-adjusted point estimate
      - Whether the CI still excludes zero
    """
    ct = np.linspace(*confound_treatment_range, grid_size)
    co = np.linspace(*confound_outcome_range, grid_size)
    CT, CO = np.meshgrid(ct, co)

    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    adj_est = np.zeros_like(CT)
    still_sig = np.zeros_like(CT, dtype=bool)

    for i in range(grid_size):
        for j in range(grid_size):
            bias = math.sqrt(CT[i, j] * CO[i, j] / max(1 - CT[i, j], 1e-12))
            adj = estimate - bias * abs(estimate)
            adj_est[i, j] = adj
            adj_se = se / max(1 - CT[i, j], 1e-12)
            ci_lower = adj - z_crit * adj_se
            ci_upper = adj + z_crit * adj_se
            still_sig[i, j] = (ci_lower > 0) or (ci_upper < 0)

    return {
        "confound_treatment": ct,
        "confound_outcome": co,
        "CT": CT,
        "CO": CO,
        "adjusted_estimate": adj_est,
        "still_significant": still_sig,
    }


# ===================================================================
# 5.  Breakdown point analysis
# ===================================================================

def breakdown_point_fraction(
    estimate: float,
    se: float,
    *,
    alpha: float = 0.05,
) -> float:
    """Fraction of observations that must be manipulated to nullify the result.

    Approximation based on the z-score of the point estimate.
    """
    z = abs(estimate) / max(se, 1e-12)
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    if z <= z_crit:
        return 0.0
    return 1.0 - z_crit / z


def breakdown_point_binary_search(
    estimate_fn,
    data: np.ndarray,
    *,
    max_fraction: float = 0.5,
    tol: float = 0.01,
    alpha: float = 0.05,
) -> float:
    """Binary search for the breakdown point.

    *estimate_fn(data_subset)* returns (estimate, se).  We find the
    smallest fraction of data to remove that changes the conclusion.
    """
    n = data.shape[0]
    est_full, se_full = estimate_fn(data)
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)

    if abs(est_full) / max(se_full, 1e-12) <= z_crit:
        return 0.0

    lo, hi = 0.0, max_fraction
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        n_keep = max(int(n * (1 - mid)), 2)
        indices = np.random.RandomState(42).choice(n, size=n_keep, replace=False)
        est_sub, se_sub = estimate_fn(data[indices])
        if abs(est_sub) / max(se_sub, 1e-12) <= z_crit:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


# ===================================================================
# 6.  Combined structural + parametric robustness
# ===================================================================

def combined_robustness_score(
    structural_radius: int,
    parametric_breakdown: float,
    *,
    structural_weight: float = 0.5,
    parametric_weight: float = 0.5,
    max_structural: int = 10,
) -> float:
    """Composite robustness score combining structural and parametric analyses.

    Normalises the structural radius to [0, 1] by dividing by *max_structural*,
    then takes a weighted average.
    """
    s_norm = min(structural_radius / max(max_structural, 1), 1.0)
    p_norm = min(max(parametric_breakdown, 0.0), 1.0)
    return structural_weight * s_norm + parametric_weight * p_norm


def robustness_report(
    estimate: float,
    se: float,
    structural_radius: int,
    *,
    alpha: float = 0.05,
    sd_outcome: float = 1.0,
) -> SensitivityResult:
    """Comprehensive sensitivity report combining multiple methods."""
    ev_point, ev_ci = e_value_diff(estimate, se, sd_outcome=sd_outcome, alpha=alpha)

    bp = breakdown_point_fraction(estimate, se, alpha=alpha)

    contour = ovb_contour_data(estimate, se)

    combined = combined_robustness_score(
        structural_radius, bp, max_structural=max(structural_radius * 2, 5)
    )

    return SensitivityResult(
        estimate=estimate,
        lower_ci=estimate - sp_stats.norm.ppf(1 - alpha / 2) * se,
        upper_ci=estimate + sp_stats.norm.ppf(1 - alpha / 2) * se,
        e_value=ev_point,
        e_value_ci=ev_ci,
        breakdown_point=bp,
        contour_data=contour,
        method="combined",
    )


# ===================================================================
# 7.  Calibration against observed covariates
# ===================================================================

def calibrate_against_covariates(
    Y: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    *,
    covariate_names: Optional[List[str]] = None,
) -> List[Dict[str, float]]:
    """Benchmark omitted-variable bias using observed covariates.

    For each observed covariate, compute its partial R² with the
    outcome and treatment.  These values serve as calibration
    benchmarks for hypothetical unobserved confounders.

    Reference: Cinelli & Hazlett (2020), Section 5.
    """
    n = Y.shape[0]
    if treatment.ndim == 1:
        treatment = treatment.reshape(-1, 1)
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    d = covariates.shape[1]
    if covariate_names is None:
        covariate_names = [f"X{i}" for i in range(d)]

    results: List[Dict[str, float]] = []
    for col in range(d):
        other_cols = [c for c in range(d) if c != col]
        if other_cols:
            others = covariates[:, other_cols]
        else:
            others = np.empty((n, 0))

        X_with_treatment = np.column_stack([treatment, others]) if others.shape[1] > 0 else treatment

        r2_y = partial_r2(Y.ravel(), covariates[:, col:col+1], X_with_treatment)
        r2_d = partial_r2(treatment.ravel(), covariates[:, col:col+1],
                          others if others.shape[1] > 0 else np.ones((n, 1)))

        results.append({
            "name": covariate_names[col],
            "r2_outcome": r2_y,
            "r2_treatment": r2_d,
            "bound_estimate": math.sqrt(r2_y * r2_d),
        })

    return results


# ===================================================================
# 8.  Bias amplification formula
# ===================================================================

def bias_amplification(
    rr_ud: float,
    rr_zu: float,
) -> float:
    """Bias amplification factor for unmeasured confounding.

    Given:
      RR_{UD} = association between confounder U and outcome D,
      RR_{ZU} = association between confounder U and treatment Z,

    the maximum bias factor is:
        BF = (RR_{UD} × RR_{ZU}) / (RR_{UD} + RR_{ZU} - 1).

    Reference: VanderWeele & Ding (2017), Proposition 2.
    """
    if rr_ud < 1.0 or rr_zu < 1.0:
        raise ValueError("Risk ratios must be ≥ 1")

    denom = rr_ud + rr_zu - 1.0
    if denom < 1e-12:
        return 1.0
    return (rr_ud * rr_zu) / denom


def maximum_bias_bound(
    rr_ud: float,
    rr_zu: float,
    estimate_rr: float,
) -> float:
    """Corrected estimate after removing maximum confounding bias.

    Returns estimate_rr / BF, which is the minimum true causal effect
    consistent with the observed association and hypothetical confounder.
    """
    bf = bias_amplification(rr_ud, rr_zu)
    return estimate_rr / bf


def sensitivity_table(
    estimate_rr: float,
    *,
    rr_range: Tuple[float, float] = (1.0, 5.0),
    n_points: int = 10,
) -> List[Dict[str, float]]:
    """Generate a sensitivity table showing corrected estimates.

    Varies RR_{UD} and RR_{ZU} and shows the corrected causal effect
    for each combination.
    """
    rr_values = np.linspace(rr_range[0], rr_range[1], n_points)
    table: List[Dict[str, float]] = []

    for rr_ud in rr_values:
        for rr_zu in rr_values:
            bf = bias_amplification(max(rr_ud, 1.0), max(rr_zu, 1.0))
            corrected = estimate_rr / bf
            table.append({
                "rr_ud": float(rr_ud),
                "rr_zu": float(rr_zu),
                "bias_factor": bf,
                "corrected_rr": corrected,
                "explains_away": corrected < 1.0,
            })
    return table

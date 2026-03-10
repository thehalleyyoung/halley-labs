"""
Treatment effect estimation convenience functions.

High-level wrappers for estimating ATE and ATT that internally handle
adjustment set selection, cross-fitting, and inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from causalcert.types import AdjacencyMatrix, EstimationResult, NodeId, NodeSet
from causalcert.estimation.adjustment import find_optimal_adjustment_set
from causalcert.estimation.aipw import AIPWEstimator, ipw_estimator, regression_estimator
from causalcert.estimation.propensity import PropensityModel
from causalcert.estimation.outcome import OutcomeModel
from causalcert.estimation.crossfit import CrossFitter, aggregate_fold_results
from causalcert.estimation.influence import (
    influence_function,
    influence_function_att,
    standard_error_from_influence,
    confidence_interval as if_confidence_interval,
)


# ---------------------------------------------------------------------------
# ATE estimation
# ---------------------------------------------------------------------------


def estimate_ate(
    adj: AdjacencyMatrix,
    data: pd.DataFrame,
    treatment: NodeId,
    outcome: NodeId,
    adjustment_set: NodeSet | None = None,
    method: str = "aipw",
    n_folds: int = 5,
    seed: int = 42,
    propensity_model: str = "logistic",
    outcome_model_type: str = "linear",
    alpha: float = 0.05,
) -> EstimationResult:
    """Estimate the Average Treatment Effect (ATE).

    If *adjustment_set* is ``None``, automatically selects the optimal
    adjustment set via the O-set criterion.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    data : pd.DataFrame
        Observational data.
    treatment : NodeId
        Treatment variable (binary).
    outcome : NodeId
        Outcome variable.
    adjustment_set : NodeSet | None
        Valid adjustment set.  ``None`` for automatic selection.
    method : str
        Estimation method (``"aipw"``, ``"ipw"``, ``"regression"``).
    n_folds : int
        Number of cross-fitting folds.
    seed : int
        Random seed.
    propensity_model : str
        Propensity model type.
    outcome_model_type : str
        Outcome model type.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    EstimationResult
    """
    if adjustment_set is None:
        adjustment_set = find_optimal_adjustment_set(adj, treatment, outcome)

    if method == "aipw":
        estimator = AIPWEstimator(
            n_folds=n_folds,
            propensity_model=propensity_model,
            outcome_model=outcome_model_type,
            seed=seed,
        )
        return estimator.estimate(adj, data, treatment, outcome, adjustment_set)

    # Extract arrays
    values = data.values.astype(np.float64)
    T = values[:, treatment]
    Y = values[:, outcome]
    cov_cols = sorted(adjustment_set)
    X = values[:, cov_cols] if cov_cols else np.ones((len(T), 1))
    n = len(Y)

    if method == "ipw":
        return _estimate_ate_ipw(
            X, T, Y, adjustment_set, n, propensity_model, seed, alpha
        )
    elif method == "regression":
        return _estimate_ate_regression(
            X, T, Y, adjustment_set, n, outcome_model_type, seed, alpha
        )
    else:
        raise ValueError(f"Unknown estimation method: {method!r}")


def _estimate_ate_ipw(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    adjustment_set: NodeSet,
    n: int,
    propensity_model: str,
    seed: int,
    alpha: float,
) -> EstimationResult:
    """IPW estimator for ATE."""
    ps = PropensityModel(model_type=propensity_model, seed=seed)
    ps.fit(X, T)
    e = ps.predict(X)

    ate = ipw_estimator(Y, T, e, normalize=True)

    # Variance via IPW influence function
    w1 = T / e
    w0 = (1.0 - T) / (1.0 - e)
    psi = w1 * Y / np.sum(w1) * n - w0 * Y / np.sum(w0) * n - ate
    se = float(np.sqrt(np.mean(psi ** 2) / n))
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    return EstimationResult(
        ate=ate,
        se=se,
        ci_lower=ate - z * se,
        ci_upper=ate + z * se,
        adjustment_set=adjustment_set,
        method="ipw",
        n_obs=n,
    )


def _estimate_ate_regression(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    adjustment_set: NodeSet,
    n: int,
    outcome_model_type: str,
    seed: int,
    alpha: float,
) -> EstimationResult:
    """Regression estimator for ATE."""
    om = OutcomeModel(model_type=outcome_model_type, seed=seed)
    om.fit(X, T, Y)
    mu0, mu1 = om.predict(X)

    ate = float(np.mean(mu1 - mu0))
    residuals = (mu1 - mu0) - ate
    se = float(np.sqrt(np.mean(residuals ** 2) / n))
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    return EstimationResult(
        ate=ate,
        se=se,
        ci_lower=ate - z * se,
        ci_upper=ate + z * se,
        adjustment_set=adjustment_set,
        method="regression",
        n_obs=n,
    )


# ---------------------------------------------------------------------------
# ATT estimation
# ---------------------------------------------------------------------------


def estimate_att(
    adj: AdjacencyMatrix,
    data: pd.DataFrame,
    treatment: NodeId,
    outcome: NodeId,
    adjustment_set: NodeSet | None = None,
    n_folds: int = 5,
    seed: int = 42,
    propensity_model: str = "logistic",
    outcome_model_type: str = "linear",
    alpha: float = 0.05,
) -> EstimationResult:
    """Estimate the Average Treatment effect on the Treated (ATT).

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    data : pd.DataFrame
        Observational data.
    treatment : NodeId
        Treatment variable.
    outcome : NodeId
        Outcome variable.
    adjustment_set : NodeSet | None
        Valid adjustment set.
    n_folds : int
        Cross-fitting folds.
    seed : int
        Random seed.
    propensity_model : str
        Propensity model type.
    outcome_model_type : str
        Outcome model type.
    alpha : float
        Significance level.

    Returns
    -------
    EstimationResult
    """
    if adjustment_set is None:
        adjustment_set = find_optimal_adjustment_set(adj, treatment, outcome)

    estimator = AIPWEstimator(
        n_folds=n_folds,
        propensity_model=propensity_model,
        outcome_model=outcome_model_type,
        seed=seed,
    )
    return estimator.estimate_att(adj, data, treatment, outcome, adjustment_set)


# ---------------------------------------------------------------------------
# CATE estimation
# ---------------------------------------------------------------------------


def estimate_cate(
    adj: AdjacencyMatrix,
    data: pd.DataFrame,
    treatment: NodeId,
    outcome: NodeId,
    adjustment_set: NodeSet | None = None,
    outcome_model_type: str = "rf",
    seed: int = 42,
) -> np.ndarray:
    """Estimate the Conditional Average Treatment Effect (CATE).

    Uses the T-learner approach: fit separate outcome models per arm
    and compute CATE(x) = E[Y|T=1, X=x] - E[Y|T=0, X=x].

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    data : pd.DataFrame
        Observational data.
    treatment, outcome : NodeId
        Treatment and outcome.
    adjustment_set : NodeSet | None
        Adjustment set.
    outcome_model_type : str
        Outcome model type.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Per-observation CATE estimates, shape ``(n,)``.
    """
    if adjustment_set is None:
        adjustment_set = find_optimal_adjustment_set(adj, treatment, outcome)

    values = data.values.astype(np.float64)
    T = values[:, treatment]
    Y = values[:, outcome]
    cov_cols = sorted(adjustment_set)
    X = values[:, cov_cols] if cov_cols else np.ones((len(T), 1))

    om = OutcomeModel(model_type=outcome_model_type, seed=seed)
    om.fit(X, T, Y)
    mu0, mu1 = om.predict(X)
    return mu1 - mu0


# ---------------------------------------------------------------------------
# Effect sign determination
# ---------------------------------------------------------------------------


def determine_effect_sign(
    ate: float,
    se: float,
    alpha: float = 0.05,
) -> tuple[str, float]:
    """Determine the sign of the treatment effect with confidence.

    Parameters
    ----------
    ate : float
        Point estimate of ATE.
    se : float
        Standard error.
    alpha : float
        Significance level.

    Returns
    -------
    tuple[str, float]
        ``(sign, confidence)`` where sign is ``"positive"``, ``"negative"``,
        or ``"indeterminate"``, and confidence is the posterior probability
        of the determined sign.
    """
    if se <= 0:
        sign = "positive" if ate > 0 else ("negative" if ate < 0 else "indeterminate")
        return sign, 1.0

    z = ate / se
    # Two-sided p-value
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    if p_value < alpha:
        if ate > 0:
            return "positive", float(1.0 - p_value / 2.0)
        else:
            return "negative", float(1.0 - p_value / 2.0)
    else:
        return "indeterminate", float(1.0 - p_value)


def estimate_all_methods(
    adj: AdjacencyMatrix,
    data: pd.DataFrame,
    treatment: NodeId,
    outcome: NodeId,
    adjustment_set: NodeSet | None = None,
    seed: int = 42,
) -> dict[str, EstimationResult]:
    """Estimate ATE using all available methods for comparison.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    data : pd.DataFrame
        Observational data.
    treatment, outcome : NodeId
        Treatment and outcome.
    adjustment_set : NodeSet | None
        Adjustment set.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, EstimationResult]
        Mapping from method name to result.
    """
    if adjustment_set is None:
        adjustment_set = find_optimal_adjustment_set(adj, treatment, outcome)

    results = {}
    for method in ["aipw", "ipw", "regression"]:
        try:
            result = estimate_ate(
                adj, data, treatment, outcome,
                adjustment_set=adjustment_set,
                method=method,
                seed=seed,
            )
            results[method] = result
        except Exception:
            pass
    return results


def effect_summary(
    result: EstimationResult,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Produce a human-readable summary of an estimation result.

    Parameters
    ----------
    result : EstimationResult
        An estimation result.
    alpha : float
        Significance level for interpretation.

    Returns
    -------
    dict[str, Any]
        Summary with point estimate, CI, significance, sign.
    """
    sign, conf = determine_effect_sign(result.ate, result.se, alpha)
    significant = (result.ci_lower > 0) or (result.ci_upper < 0)

    return {
        "method": result.method,
        "ate": result.ate,
        "se": result.se,
        "ci": (result.ci_lower, result.ci_upper),
        "significant": significant,
        "effect_sign": sign,
        "sign_confidence": conf,
        "n_obs": result.n_obs,
        "adjustment_set_size": len(result.adjustment_set),
    }

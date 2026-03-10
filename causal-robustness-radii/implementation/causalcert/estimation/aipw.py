"""
Augmented Inverse-Probability Weighting (AIPW) estimator.

Doubly-robust estimator for the average treatment effect that combines a
propensity score model with an outcome regression model.  Achieves
semiparametric efficiency when both nuisance models are consistently estimated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from causalcert.types import AdjacencyMatrix, EstimationResult, NodeId, NodeSet
from causalcert.estimation.propensity import PropensityModel, trim_by_propensity
from causalcert.estimation.outcome import OutcomeModel
from causalcert.estimation.crossfit import CrossFitter, aggregate_fold_results
from causalcert.estimation.influence import (
    influence_function,
    influence_function_att,
    variance_from_influence,
    standard_error_from_influence,
    confidence_interval as if_confidence_interval,
    bootstrap_confidence_interval,
)


class AIPWEstimator:
    """Augmented inverse-probability weighting estimator.

    Implements the doubly-robust AIPW estimator for binary treatments.
    The estimator is consistent if *either* the propensity score model
    *or* the outcome regression model is correctly specified.

    Parameters
    ----------
    n_folds : int
        Number of cross-fitting folds.
    propensity_model : str
        Propensity score model type (``"logistic"``, ``"rf"``, ``"gbm"``).
    outcome_model : str
        Outcome regression model type (``"linear"``, ``"rf"``, ``"gbm"``).
    seed : int
        Random seed.
    trim_lower : float
        Lower trimming threshold for propensity scores.
    trim_upper : float
        Upper trimming threshold for propensity scores.
    clip_bounds : tuple[float, float]
        Clipping bounds for propensity scores.
    ci_method : str
        Confidence interval method: ``"wald"`` or ``"bootstrap"``.
    n_bootstrap : int
        Number of bootstrap replicates (if ``ci_method="bootstrap"``).
    """

    def __init__(
        self,
        n_folds: int = 5,
        propensity_model: str = "logistic",
        outcome_model: str = "linear",
        seed: int = 42,
        trim_lower: float = 0.01,
        trim_upper: float = 0.99,
        clip_bounds: tuple[float, float] = (0.01, 0.99),
        ci_method: str = "wald",
        n_bootstrap: int = 1000,
    ) -> None:
        self.n_folds = n_folds
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.seed = seed
        self.trim_lower = trim_lower
        self.trim_upper = trim_upper
        self.clip_bounds = clip_bounds
        self.ci_method = ci_method
        self.n_bootstrap = n_bootstrap
        # Stored after estimation
        self._influence_values: np.ndarray | None = None
        self._propensity_scores: np.ndarray | None = None
        self._mu0: np.ndarray | None = None
        self._mu1: np.ndarray | None = None

    # -- Main estimation method -----------------------------------------------

    def estimate(
        self,
        adj: AdjacencyMatrix,
        data: pd.DataFrame,
        treatment: NodeId,
        outcome: NodeId,
        adjustment_set: NodeSet,
    ) -> EstimationResult:
        """Estimate the ATE via AIPW with cross-fitting.

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
        adjustment_set : NodeSet
            Valid adjustment set.

        Returns
        -------
        EstimationResult
        """
        cols = list(data.columns)
        cov_cols = sorted(adjustment_set)
        t_col = treatment
        y_col = outcome

        values = data.values.astype(np.float64)
        T = values[:, t_col]
        Y = values[:, y_col]
        X = values[:, cov_cols] if cov_cols else np.ones((len(T), 1))

        # Cross-fitting
        cf = CrossFitter(n_folds=self.n_folds, seed=self.seed)
        ps_type = self.propensity_model
        om_type = self.outcome_model
        clip = self.clip_bounds
        seed = self.seed

        def ps_factory() -> PropensityModel:
            return PropensityModel(model_type=ps_type, clip_bounds=clip, seed=seed)

        def om_factory() -> OutcomeModel:
            return OutcomeModel(model_type=om_type, seed=seed)

        fold_results = cf.fit_arrays(X, T, Y, ps_factory, om_factory)

        # Aggregate cross-fitted nuisance estimates
        nuisance = aggregate_fold_results(fold_results, Y, T)
        e = nuisance["e"]
        mu0 = nuisance["mu0"]
        mu1 = nuisance["mu1"]

        self._propensity_scores = e
        self._mu0 = mu0
        self._mu1 = mu1

        # Trim observations with extreme propensity scores
        trim_mask = (e >= self.trim_lower) & (e <= self.trim_upper)
        Y = Y[trim_mask]
        T = T[trim_mask]
        mu0 = mu0[trim_mask]
        mu1 = mu1[trim_mask]
        e = e[trim_mask]

        # AIPW scores
        scores = self._aipw_score(Y, T, mu0, mu1, e)
        ate = float(np.mean(scores))

        # Influence function and inference
        psi = scores - ate
        self._influence_values = psi

        se = standard_error_from_influence(psi)

        if self.ci_method == "bootstrap":
            _, ci_lo, ci_hi = bootstrap_confidence_interval(
                Y, T, mu0, mu1, e,
                alpha=0.05, n_bootstrap=self.n_bootstrap, seed=self.seed,
            )
        else:
            ci_lo, ci_hi = if_confidence_interval(ate, psi, alpha=0.05)

        return EstimationResult(
            ate=ate,
            se=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            adjustment_set=adjustment_set,
            method="aipw",
            n_obs=len(Y),
        )

    def estimate_att(
        self,
        adj: AdjacencyMatrix,
        data: pd.DataFrame,
        treatment: NodeId,
        outcome: NodeId,
        adjustment_set: NodeSet,
    ) -> EstimationResult:
        """Estimate the ATT via AIPW with cross-fitting.

        Parameters
        ----------
        adj : AdjacencyMatrix
            DAG adjacency matrix.
        data : pd.DataFrame
            Observational data.
        treatment, outcome : NodeId
            Treatment and outcome variable indices.
        adjustment_set : NodeSet
            Valid adjustment set.

        Returns
        -------
        EstimationResult
        """
        cov_cols = sorted(adjustment_set)
        values = data.values.astype(np.float64)
        T = values[:, treatment]
        Y = values[:, outcome]
        X = values[:, cov_cols] if cov_cols else np.ones((len(T), 1))

        cf = CrossFitter(n_folds=self.n_folds, seed=self.seed)
        ps_type = self.propensity_model
        om_type = self.outcome_model
        clip = self.clip_bounds
        seed = self.seed

        def ps_factory() -> PropensityModel:
            return PropensityModel(model_type=ps_type, clip_bounds=clip, seed=seed)

        def om_factory() -> OutcomeModel:
            return OutcomeModel(model_type=om_type, seed=seed)

        fold_results = cf.fit_arrays(X, T, Y, ps_factory, om_factory)
        nuisance = aggregate_fold_results(fold_results, Y, T)
        e = nuisance["e"]
        mu0 = nuisance["mu0"]

        # ATT scores
        p = float(np.mean(T))
        scores = (T / p) * (Y - mu0) - ((1.0 - T) * e / (p * (1.0 - e))) * (Y - mu0)
        att = float(np.mean(scores))

        psi = scores - att
        self._influence_values = psi
        se = standard_error_from_influence(psi)
        ci_lo, ci_hi = if_confidence_interval(att, psi, alpha=0.05)

        return EstimationResult(
            ate=att,
            se=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            adjustment_set=adjustment_set,
            method="aipw_att",
            n_obs=len(Y),
        )

    def estimate_multiple_sets(
        self,
        adj: AdjacencyMatrix,
        data: pd.DataFrame,
        treatment: NodeId,
        outcome: NodeId,
        adjustment_sets: list[NodeSet],
    ) -> list[EstimationResult]:
        """Estimate the ATE under multiple adjustment sets.

        Parameters
        ----------
        adj : AdjacencyMatrix
            DAG adjacency matrix.
        data : pd.DataFrame
            Observational data.
        treatment, outcome : NodeId
            Treatment and outcome.
        adjustment_sets : list[NodeSet]
            Multiple valid adjustment sets.

        Returns
        -------
        list[EstimationResult]
            One result per adjustment set.
        """
        results = []
        for adj_set in adjustment_sets:
            result = self.estimate(adj, data, treatment, outcome, adj_set)
            results.append(result)
        return results

    # -- AIPW score -----------------------------------------------------------

    def _aipw_score(
        self,
        y: np.ndarray,
        t: np.ndarray,
        mu0: np.ndarray,
        mu1: np.ndarray,
        e: np.ndarray,
    ) -> np.ndarray:
        """Compute the AIPW influence-function-based scores.

        The doubly-robust score for observation i is::

            ψ_i = (T_i / e_i) * (Y_i - μ₁_i) - ((1-T_i) / (1-e_i)) * (Y_i - μ₀_i) + (μ₁_i - μ₀_i)

        This is the *un-centred* AIPW score whose mean is the ATE.

        Parameters
        ----------
        y : np.ndarray
            Observed outcomes.
        t : np.ndarray
            Treatment assignments (binary).
        mu0, mu1 : np.ndarray
            Predicted potential outcomes under control/treatment.
        e : np.ndarray
            Estimated propensity scores.

        Returns
        -------
        np.ndarray
            Per-observation AIPW scores.
        """
        y = np.asarray(y, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        mu0 = np.asarray(mu0, dtype=np.float64)
        mu1 = np.asarray(mu1, dtype=np.float64)
        e = np.asarray(e, dtype=np.float64)

        scores = (
            t * (y - mu1) / e
            - (1.0 - t) * (y - mu0) / (1.0 - e)
            + (mu1 - mu0)
        )
        return scores

    # -- Diagnostics ----------------------------------------------------------

    @property
    def influence_values(self) -> np.ndarray | None:
        """Influence function values from the last estimation call."""
        return self._influence_values

    @property
    def propensity_scores(self) -> np.ndarray | None:
        """Propensity scores from the last estimation call."""
        return self._propensity_scores

    def diagnostics(self) -> dict[str, Any]:
        """Return diagnostics from the last estimation.

        Returns
        -------
        dict[str, Any]
            Contains influence function summary stats and propensity score
            distribution.
        """
        diag: dict[str, Any] = {}
        if self._influence_values is not None:
            psi = self._influence_values
            diag["if_mean"] = float(np.mean(psi))
            diag["if_std"] = float(np.std(psi))
            diag["if_max_abs"] = float(np.max(np.abs(psi)))
        if self._propensity_scores is not None:
            e = self._propensity_scores
            diag["ps_mean"] = float(np.mean(e))
            diag["ps_std"] = float(np.std(e))
            diag["ps_min"] = float(np.min(e))
            diag["ps_max"] = float(np.max(e))
        return diag


# ---------------------------------------------------------------------------
# Simple (non-cross-fitted) AIPW
# ---------------------------------------------------------------------------


def aipw_simple(
    y: np.ndarray,
    t: np.ndarray,
    X: np.ndarray,
    propensity_model: str = "logistic",
    outcome_model: str = "linear",
    clip_bounds: tuple[float, float] = (0.01, 0.99),
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Simple AIPW estimator without cross-fitting.

    Fits nuisance models on the full sample (not recommended for inference
    but useful for quick estimates and testing).

    Parameters
    ----------
    y : np.ndarray
        Outcomes.
    t : np.ndarray
        Treatments (binary).
    X : np.ndarray
        Covariates.
    propensity_model, outcome_model : str
        Model types.
    clip_bounds : tuple[float, float]
        Propensity score clipping.
    seed : int
        Random seed.
    alpha : float
        Significance level.

    Returns
    -------
    dict[str, float]
        Contains ``"ate"``, ``"se"``, ``"ci_lower"``, ``"ci_upper"``.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    ps = PropensityModel(model_type=propensity_model, clip_bounds=clip_bounds, seed=seed)
    ps.fit(X, t)
    e = ps.predict(X)

    om = OutcomeModel(model_type=outcome_model, seed=seed)
    om.fit(X, t, y)
    mu0, mu1 = om.predict(X)

    scores = (
        t * (y - mu1) / e
        - (1.0 - t) * (y - mu0) / (1.0 - e)
        + (mu1 - mu0)
    )
    ate = float(np.mean(scores))
    psi = scores - ate
    se = float(np.sqrt(np.mean(psi ** 2) / len(y)))
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
    }


# ---------------------------------------------------------------------------
# IPW-only estimator (for comparison)
# ---------------------------------------------------------------------------


def ipw_estimator(
    y: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    normalize: bool = True,
) -> float:
    """Horvitz-Thompson / Hajek IPW estimator for the ATE.

    Parameters
    ----------
    y : np.ndarray
        Outcomes.
    t : np.ndarray
        Treatments (binary).
    e : np.ndarray
        Propensity scores.
    normalize : bool
        If ``True``, use Hajek (normalised) weights.

    Returns
    -------
    float
        Estimated ATE.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)

    if normalize:
        w1 = t / e
        w0 = (1.0 - t) / (1.0 - e)
        ate = float(np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0))
    else:
        n = len(y)
        ate = float(np.mean(t * y / e - (1.0 - t) * y / (1.0 - e)))
    return ate


# ---------------------------------------------------------------------------
# Regression adjustment estimator (for comparison)
# ---------------------------------------------------------------------------


def regression_estimator(
    y: np.ndarray,
    t: np.ndarray,
    X: np.ndarray,
    model_type: str = "linear",
    seed: int = 42,
) -> float:
    """Regression adjustment (G-computation) estimator for the ATE.

    Parameters
    ----------
    y : np.ndarray
        Outcomes.
    t : np.ndarray
        Treatments (binary).
    X : np.ndarray
        Covariates.
    model_type : str
        Outcome model type.
    seed : int
        Random seed.

    Returns
    -------
    float
        Estimated ATE.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    om = OutcomeModel(model_type=model_type, seed=seed)
    om.fit(X, t, y)
    mu0, mu1 = om.predict(X)
    return float(np.mean(mu1 - mu0))

"""
Double/Debiased Machine Learning (DML) estimator.

Implements the Chernozhukov et al. (2018) framework for causal inference
with high-dimensional nuisance parameters.  Supports partially linear and
interactive regression models with cross-fitted Neyman-orthogonal scores,
multiple ML backends, and joint inference via multiplier bootstrap.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np
from scipy import stats as sp_stats
from sklearn.base import clone, BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold


# ===================================================================
# Result data structures
# ===================================================================


@dataclass(frozen=True, slots=True)
class DMLResult:
    """Result of a DML estimation procedure.

    Attributes
    ----------
    theta : float
        Point estimate of the causal parameter.
    se : float
        Standard error.
    ci_lower : float
        Lower confidence bound.
    ci_upper : float
        Upper confidence bound.
    p_value : float
        Two-sided p-value against H0: θ = 0.
    scores : np.ndarray
        Neyman-orthogonal score values, shape ``(n,)``.
    method : str
        Model type identifier.
    n_obs : int
        Number of observations used.
    """

    theta: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    scores: np.ndarray
    method: str = "dml"
    n_obs: int = 0


@dataclass(frozen=True, slots=True)
class JointDMLResult:
    """Joint inference result for multiple DML estimates.

    Attributes
    ----------
    thetas : np.ndarray
        Point estimates, shape ``(k,)``.
    se_values : np.ndarray
        Standard errors, shape ``(k,)``.
    joint_ci_lower : np.ndarray
        Joint lower CI bounds, shape ``(k,)``.
    joint_ci_upper : np.ndarray
        Joint upper CI bounds, shape ``(k,)``.
    critical_value : float
        Bootstrap critical value for joint coverage.
    """

    thetas: np.ndarray
    se_values: np.ndarray
    joint_ci_lower: np.ndarray
    joint_ci_upper: np.ndarray
    critical_value: float


# ===================================================================
# ML backend factories
# ===================================================================


def _build_ml_regressor(backend: str, seed: int) -> Any:
    """Instantiate a regression model for the specified backend.

    Parameters
    ----------
    backend : str
        One of ``"lasso"``, ``"rf"``, ``"gbm"``, ``"ridge"``.
    seed : int
        Random seed.

    Returns
    -------
    sklearn estimator
    """
    if backend == "lasso":
        return LassoCV(cv=5, max_iter=5000, random_state=seed)
    if backend == "ridge":
        return RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0), cv=5)
    if backend == "rf":
        return RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        )
    if backend == "gbm":
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=10, subsample=0.8, random_state=seed,
        )
    raise ValueError(f"Unknown regression backend: {backend!r}")


def _build_ml_classifier(backend: str, seed: int) -> Any:
    """Instantiate a classification model for the specified backend.

    Parameters
    ----------
    backend : str
        One of ``"logistic"``, ``"rf"``, ``"gbm"``.
    seed : int
        Random seed.

    Returns
    -------
    sklearn estimator
    """
    if backend == "logistic":
        return LogisticRegressionCV(
            Cs=10, cv=5, max_iter=5000, solver="lbfgs", random_state=seed,
        )
    if backend == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        )
    if backend == "gbm":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=10, subsample=0.8, random_state=seed,
        )
    raise ValueError(f"Unknown classification backend: {backend!r}")


# ===================================================================
# Cross-fitting helpers
# ===================================================================


def _crossfit_residuals(
    X: np.ndarray,
    target: np.ndarray,
    model_factory: Callable[[], Any],
    n_folds: int,
    seed: int,
    *,
    task: str = "regression",
) -> np.ndarray:
    """Cross-fit a nuisance model and return out-of-fold residuals.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape ``(n, p)``.
    target : np.ndarray
        Target vector, shape ``(n,)``.
    model_factory : Callable
        Returns a fresh sklearn estimator.
    n_folds : int
        Number of folds.
    seed : int
        Random seed.
    task : str
        ``"regression"`` or ``"classification"``.

    Returns
    -------
    np.ndarray
        Out-of-fold residuals (target − prediction), shape ``(n,)``.
    """
    n = X.shape[0]
    residuals = np.empty(n, dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        model = model_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[train_idx], target[train_idx])
        if task == "classification" and hasattr(model, "predict_proba"):
            preds = model.predict_proba(X[test_idx])[:, 1]
        else:
            preds = model.predict(X[test_idx])
        residuals[test_idx] = target[test_idx] - preds

    return residuals


def _crossfit_predictions(
    X: np.ndarray,
    target: np.ndarray,
    model_factory: Callable[[], Any],
    n_folds: int,
    seed: int,
    *,
    task: str = "regression",
) -> np.ndarray:
    """Cross-fit a nuisance model and return out-of-fold predictions.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape ``(n, p)``.
    target : np.ndarray
        Target vector, shape ``(n,)``.
    model_factory : Callable
        Returns a fresh sklearn estimator.
    n_folds : int
        Number of folds.
    seed : int
        Random seed.
    task : str
        ``"regression"`` or ``"classification"``.

    Returns
    -------
    np.ndarray
        Out-of-fold predictions, shape ``(n,)``.
    """
    n = X.shape[0]
    predictions = np.empty(n, dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        model = model_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[train_idx], target[train_idx])
        if task == "classification" and hasattr(model, "predict_proba"):
            predictions[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        else:
            predictions[test_idx] = model.predict(X[test_idx])

    return predictions


# ===================================================================
# DML Estimator
# ===================================================================


class DMLEstimator:
    """Double/Debiased Machine Learning estimator.

    Implements Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey &
    Robins (2018).  Two model specifications are supported:

    **Partially linear model** (PLR)::

        Y = θ·D + g₀(X) + U,  E[U|X,D] = 0
        D = m₀(X) + V,        E[V|X]   = 0

    **Interactive regression model** (IRM)::

        Y = g₀(D, X) + U,     E[U|X,D] = 0
        D = m₀(X) + V,        E[V|X]   = 0

    Parameters
    ----------
    model_type : str
        ``"plr"`` for partially linear or ``"irm"`` for interactive.
    ml_backend : str
        ML backend for nuisance models: ``"lasso"``, ``"rf"``, ``"gbm"``,
        ``"ridge"``.
    propensity_backend : str
        ML backend for the propensity model (IRM only): ``"logistic"``,
        ``"rf"``, ``"gbm"``.
    n_folds : int
        Number of cross-fitting folds.
    seed : int
        Random seed.
    alpha : float
        Significance level for confidence intervals.
    trim : float
        Propensity score trimming threshold.
    """

    def __init__(
        self,
        model_type: str = "plr",
        ml_backend: str = "lasso",
        propensity_backend: str = "logistic",
        n_folds: int = 5,
        seed: int = 42,
        alpha: float = 0.05,
        trim: float = 0.01,
    ) -> None:
        if model_type not in ("plr", "irm"):
            raise ValueError(f"model_type must be 'plr' or 'irm', got {model_type!r}")
        self.model_type = model_type
        self.ml_backend = ml_backend
        self.propensity_backend = propensity_backend
        self.n_folds = n_folds
        self.seed = seed
        self.alpha = alpha
        self.trim = trim
        self._scores: np.ndarray | None = None
        self._theta: float | None = None

    # -----------------------------------------------------------------
    # Main estimation
    # -----------------------------------------------------------------

    def estimate(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
    ) -> DMLResult:
        """Estimate the causal parameter θ.

        Parameters
        ----------
        Y : np.ndarray
            Outcome, shape ``(n,)``.
        D : np.ndarray
            Treatment (binary for IRM, continuous allowed for PLR),
            shape ``(n,)``.
        X : np.ndarray
            Covariates, shape ``(n, p)``.

        Returns
        -------
        DMLResult
        """
        Y = np.asarray(Y, dtype=np.float64).ravel()
        D = np.asarray(D, dtype=np.float64).ravel()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = len(Y)

        if self.model_type == "plr":
            return self._estimate_plr(Y, D, X)
        return self._estimate_irm(Y, D, X)

    # -----------------------------------------------------------------
    # Partially linear model
    # -----------------------------------------------------------------

    def _estimate_plr(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
    ) -> DMLResult:
        """Partially linear model: Y = θ·D + g(X) + ε."""
        n = len(Y)
        backend = self.ml_backend
        seed = self.seed

        def reg_factory() -> Any:
            return _build_ml_regressor(backend, seed)

        # Step 1: residualise Y on X
        V_y = _crossfit_residuals(X, Y, reg_factory, self.n_folds, self.seed)

        # Step 2: residualise D on X
        V_d = _crossfit_residuals(X, D, reg_factory, self.n_folds, self.seed + 1)

        # Step 3: Neyman-orthogonal score and IV-like estimator
        theta = float(np.sum(V_d * V_y) / np.sum(V_d * V_d))

        # Orthogonal scores
        scores = V_d * (V_y - theta * V_d)

        # Variance via influence function
        J = float(np.mean(V_d ** 2))
        se = float(np.sqrt(np.mean(scores ** 2)) / (J * np.sqrt(n)))

        z = sp_stats.norm.ppf(1.0 - self.alpha / 2.0)
        ci_lo = theta - z * se
        ci_hi = theta + z * se
        p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(theta) / max(se, 1e-12)))

        self._theta = theta
        self._scores = scores

        return DMLResult(
            theta=theta,
            se=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            p_value=p_value,
            scores=scores,
            method="dml_plr",
            n_obs=n,
        )

    # -----------------------------------------------------------------
    # Interactive regression model
    # -----------------------------------------------------------------

    def _estimate_irm(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
    ) -> DMLResult:
        """Interactive model: Y = g(D,X) + ε with binary D."""
        n = len(Y)
        ml_backend = self.ml_backend
        ps_backend = self.propensity_backend
        seed = self.seed
        trim = self.trim

        # Step 1: propensity scores
        def ps_factory() -> Any:
            return _build_ml_classifier(ps_backend, seed)

        e_hat = _crossfit_predictions(
            X, D, ps_factory, self.n_folds, seed, task="classification",
        )
        e_hat = np.clip(e_hat, trim, 1.0 - trim)

        # Step 2: outcome regression per arm
        mask1 = D == 1
        mask0 = D == 0

        def reg_factory() -> Any:
            return _build_ml_regressor(ml_backend, seed)

        mu1_hat = np.zeros(n, dtype=np.float64)
        mu0_hat = np.zeros(n, dtype=np.float64)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(X):
            # Fit outcome model for treated
            train1 = np.intersect1d(train_idx, np.where(mask1)[0])
            if len(train1) >= 2:
                m1 = _build_ml_regressor(ml_backend, seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m1.fit(X[train1], Y[train1])
                mu1_hat[test_idx] = m1.predict(X[test_idx])
            else:
                mu1_hat[test_idx] = np.mean(Y[mask1]) if mask1.any() else 0.0

            # Fit outcome model for control
            train0 = np.intersect1d(train_idx, np.where(mask0)[0])
            if len(train0) >= 2:
                m0 = _build_ml_regressor(ml_backend, seed + 1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m0.fit(X[train0], Y[train0])
                mu0_hat[test_idx] = m0.predict(X[test_idx])
            else:
                mu0_hat[test_idx] = np.mean(Y[mask0]) if mask0.any() else 0.0

        # Step 3: AIPW-style orthogonal scores
        scores = (
            mu1_hat - mu0_hat
            + D * (Y - mu1_hat) / e_hat
            - (1.0 - D) * (Y - mu0_hat) / (1.0 - e_hat)
        )
        theta = float(np.mean(scores))

        psi = scores - theta
        se = float(np.sqrt(np.mean(psi ** 2) / n))

        z = sp_stats.norm.ppf(1.0 - self.alpha / 2.0)
        ci_lo = theta - z * se
        ci_hi = theta + z * se
        p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(theta) / max(se, 1e-12)))

        self._theta = theta
        self._scores = psi

        return DMLResult(
            theta=theta,
            se=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            p_value=p_value,
            scores=psi,
            method="dml_irm",
            n_obs=n,
        )

    # -----------------------------------------------------------------
    # Multi-treatment / multiple parameters
    # -----------------------------------------------------------------

    def estimate_multiple(
        self,
        Y: np.ndarray,
        D_list: Sequence[np.ndarray],
        X: np.ndarray,
    ) -> list[DMLResult]:
        """Estimate θ for multiple treatment variables.

        Parameters
        ----------
        Y : np.ndarray
            Outcome, shape ``(n,)``.
        D_list : Sequence[np.ndarray]
            List of treatment arrays, each shape ``(n,)``.
        X : np.ndarray
            Covariates, shape ``(n, p)``.

        Returns
        -------
        list[DMLResult]
            One result per treatment variable.
        """
        return [self.estimate(Y, D, X) for D in D_list]

    # -----------------------------------------------------------------
    # Joint inference via multiplier bootstrap
    # -----------------------------------------------------------------

    def joint_confidence_intervals(
        self,
        Y: np.ndarray,
        D_list: Sequence[np.ndarray],
        X: np.ndarray,
        n_bootstrap: int = 1000,
        alpha: float | None = None,
    ) -> JointDMLResult:
        """Compute joint confidence intervals via multiplier bootstrap.

        Uses a Gaussian multiplier bootstrap on the orthogonal scores to
        approximate the distribution of max_k |θ̂_k − θ_k| / SE_k.

        Parameters
        ----------
        Y : np.ndarray
            Outcome.
        D_list : Sequence[np.ndarray]
            Treatment variables.
        X : np.ndarray
            Covariates.
        n_bootstrap : int
            Number of bootstrap draws.
        alpha : float or None
            Joint significance level (defaults to ``self.alpha``).

        Returns
        -------
        JointDMLResult
        """
        if alpha is None:
            alpha = self.alpha

        results = self.estimate_multiple(Y, D_list, X)
        k = len(results)
        n = results[0].n_obs

        thetas = np.array([r.theta for r in results])
        se_vals = np.array([r.se for r in results])

        # Stack orthogonal scores: shape (n, k)
        score_matrix = np.column_stack([r.scores for r in results])

        # Multiplier bootstrap for max statistic
        rng = np.random.default_rng(self.seed)
        max_stats = np.empty(n_bootstrap, dtype=np.float64)

        for b in range(n_bootstrap):
            weights = rng.standard_normal(n)
            boot_means = weights @ score_matrix / n
            # Normalize by SE
            t_stats = np.abs(boot_means) / np.maximum(se_vals, 1e-12)
            max_stats[b] = float(np.max(t_stats))

        critical_value = float(np.percentile(max_stats, 100 * (1.0 - alpha)))

        joint_ci_lo = thetas - critical_value * se_vals
        joint_ci_hi = thetas + critical_value * se_vals

        return JointDMLResult(
            thetas=thetas,
            se_values=se_vals,
            joint_ci_lower=joint_ci_lo,
            joint_ci_upper=joint_ci_hi,
            critical_value=critical_value,
        )

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def scores(self) -> np.ndarray | None:
        """Orthogonal score values from the last call to :meth:`estimate`."""
        return self._scores

    @property
    def theta(self) -> float | None:
        """Point estimate from the last call to :meth:`estimate`."""
        return self._theta


# ===================================================================
# Convenience functions
# ===================================================================


def dml_plr(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    *,
    ml_backend: str = "lasso",
    n_folds: int = 5,
    seed: int = 42,
    alpha: float = 0.05,
) -> DMLResult:
    """One-call partially linear DML.

    Parameters
    ----------
    Y : np.ndarray
        Outcome.
    D : np.ndarray
        Treatment.
    X : np.ndarray
        Covariates.
    ml_backend : str
        ML backend for nuisance models.
    n_folds : int
        Cross-fitting folds.
    seed : int
        Random seed.
    alpha : float
        Significance level.

    Returns
    -------
    DMLResult
    """
    est = DMLEstimator(
        model_type="plr", ml_backend=ml_backend,
        n_folds=n_folds, seed=seed, alpha=alpha,
    )
    return est.estimate(Y, D, X)


def dml_irm(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    *,
    ml_backend: str = "rf",
    propensity_backend: str = "logistic",
    n_folds: int = 5,
    seed: int = 42,
    alpha: float = 0.05,
    trim: float = 0.01,
) -> DMLResult:
    """One-call interactive regression DML.

    Parameters
    ----------
    Y : np.ndarray
        Outcome.
    D : np.ndarray
        Treatment (binary).
    X : np.ndarray
        Covariates.
    ml_backend : str
        ML backend for outcome models.
    propensity_backend : str
        ML backend for propensity model.
    n_folds : int
        Cross-fitting folds.
    seed : int
        Random seed.
    alpha : float
        Significance level.
    trim : float
        Propensity trimming threshold.

    Returns
    -------
    DMLResult
    """
    est = DMLEstimator(
        model_type="irm", ml_backend=ml_backend,
        propensity_backend=propensity_backend,
        n_folds=n_folds, seed=seed, alpha=alpha, trim=trim,
    )
    return est.estimate(Y, D, X)


# ===================================================================
# Nuisance diagnostics
# ===================================================================


def nuisance_diagnostics(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    ml_backend: str = "lasso",
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute nuisance model diagnostics for PLR.

    Evaluates cross-fitted R² for both the outcome and treatment
    residualisation steps.

    Parameters
    ----------
    Y, D, X : np.ndarray
        Outcome, treatment, and covariates.
    ml_backend : str
        ML backend.
    n_folds : int
        Folds for cross-fitting.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, Any]
        Diagnostic measures.
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    D = np.asarray(D, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    def reg_factory() -> Any:
        return _build_ml_regressor(ml_backend, seed)

    # Cross-fitted predictions
    Y_pred = _crossfit_predictions(X, Y, reg_factory, n_folds, seed)
    D_pred = _crossfit_predictions(X, D, reg_factory, n_folds, seed + 1)

    V_y = Y - Y_pred
    V_d = D - D_pred

    ss_tot_y = float(np.sum((Y - np.mean(Y)) ** 2))
    ss_res_y = float(np.sum(V_y ** 2))
    r2_y = 1.0 - ss_res_y / max(ss_tot_y, 1e-12)

    ss_tot_d = float(np.sum((D - np.mean(D)) ** 2))
    ss_res_d = float(np.sum(V_d ** 2))
    r2_d = 1.0 - ss_res_d / max(ss_tot_d, 1e-12)

    return {
        "r2_outcome": r2_y,
        "r2_treatment": r2_d,
        "rmse_outcome": float(np.sqrt(np.mean(V_y ** 2))),
        "rmse_treatment": float(np.sqrt(np.mean(V_d ** 2))),
        "residual_corr": float(np.corrcoef(V_y, V_d)[0, 1]),
        "n_obs": len(Y),
    }


# ===================================================================
# Sensitivity to ML backend choice
# ===================================================================


def sensitivity_to_backend(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    model_type: str = "plr",
    backends: Sequence[str] | None = None,
    n_folds: int = 5,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, DMLResult]:
    """Estimate θ under multiple ML backends.

    Parameters
    ----------
    Y, D, X : np.ndarray
        Outcome, treatment, and covariates.
    model_type : str
        ``"plr"`` or ``"irm"``.
    backends : Sequence[str] or None
        Backends to try. Defaults to ``["lasso", "rf", "gbm"]`` for PLR
        or ``["rf", "gbm"]`` for IRM.
    n_folds : int
        Cross-fitting folds.
    seed : int
        Random seed.
    alpha : float
        Significance level.

    Returns
    -------
    dict[str, DMLResult]
        Keyed by backend name.
    """
    if backends is None:
        backends = ["lasso", "rf", "gbm"] if model_type == "plr" else ["rf", "gbm"]

    results: dict[str, DMLResult] = {}
    for be in backends:
        est = DMLEstimator(
            model_type=model_type, ml_backend=be,
            n_folds=n_folds, seed=seed, alpha=alpha,
        )
        results[be] = est.estimate(Y, D, X)
    return results

"""
Targeted Minimum Loss Estimation (TMLE) for causal effect estimation.

Implements the van der Laan & Rubin (2006) TMLE framework with initial
outcome model fitting, clever covariate construction, fluctuation-parameter
targeting via logistic submodel MLE, iterated TMLE for better convergence,
and influence-curve-based inference.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
from scipy import stats as sp_stats
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold


# ===================================================================
# Result data structures
# ===================================================================


@dataclass(frozen=True, slots=True)
class TMLEEstimationResult:
    """Result of a TMLE estimation procedure.

    Attributes
    ----------
    ate : float
        Average treatment effect estimate.
    se : float
        Standard error.
    ci_lower : float
        Lower confidence bound.
    ci_upper : float
        Upper confidence bound.
    p_value : float
        Two-sided p-value.
    influence_curve : np.ndarray
        Efficient influence curve values, shape ``(n,)``.
    initial_estimate : float
        Plug-in estimate before targeting.
    n_targeting_steps : int
        Number of fluctuation steps performed.
    converged : bool
        Whether the targeting loop converged.
    method : str
        Estimator name.
    n_obs : int
        Sample size used.
    """

    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    influence_curve: np.ndarray
    initial_estimate: float = 0.0
    n_targeting_steps: int = 1
    converged: bool = True
    method: str = "tmle"
    n_obs: int = 0


# ===================================================================
# Helper: logistic link utilities
# ===================================================================


def _expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _logit(p: np.ndarray) -> np.ndarray:
    """Logit transform with clamping for numerical safety."""
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return np.log(p / (1.0 - p))


# ===================================================================
# Helper: ML model factories
# ===================================================================


def _build_outcome_model(backend: str, seed: int) -> Any:
    """Build an outcome regression model."""
    if backend == "linear":
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
    raise ValueError(f"Unknown outcome backend: {backend!r}")


def _build_propensity_model(backend: str, seed: int) -> Any:
    """Build a propensity score model."""
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
    raise ValueError(f"Unknown propensity backend: {backend!r}")


# ===================================================================
# Clever covariate
# ===================================================================


def clever_covariate(
    A: np.ndarray,
    g: np.ndarray,
) -> np.ndarray:
    """Construct the TMLE clever covariate H(A, W).

    For binary treatment::

        H(A, W) = A / g(W) − (1 − A) / (1 − g(W))

    Parameters
    ----------
    A : np.ndarray
        Treatment assignments (binary), shape ``(n,)``.
    g : np.ndarray
        Propensity scores P(A=1|W), shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Clever covariate values, shape ``(n,)``.
    """
    return A / g - (1.0 - A) / (1.0 - g)


def clever_covariate_components(
    A: np.ndarray,
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the treatment-arm-specific clever covariates.

    Parameters
    ----------
    A : np.ndarray
        Treatment assignments.
    g : np.ndarray
        Propensity scores.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(H1, H0)`` where H1 = A/g and H0 = −(1−A)/(1−g).
    """
    H1 = A / g
    H0 = -(1.0 - A) / (1.0 - g)
    return H1, H0


# ===================================================================
# Fluctuation step
# ===================================================================


def _fluctuation_step(
    Y: np.ndarray,
    Q_logit: np.ndarray,
    H: np.ndarray,
    *,
    max_iter: int = 50,
) -> tuple[float, np.ndarray]:
    """Fit the fluctuation parameter ε via logistic submodel MLE.

    Solves the targeting equation by fitting a univariate logistic
    regression of Y on H with offset logit(Q̄).

    Parameters
    ----------
    Y : np.ndarray
        Outcome (must be in [0, 1] for this parameterisation).
    Q_logit : np.ndarray
        Logit of current Q̄(A, W).
    H : np.ndarray
        Clever covariate.
    max_iter : int
        Maximum Newton-Raphson iterations.

    Returns
    -------
    tuple[float, np.ndarray]
        ``(epsilon, Q_star_logit)`` — fluctuation parameter and updated logit.
    """
    epsilon = 0.0
    for _ in range(max_iter):
        eta = Q_logit + epsilon * H
        Q_star = _expit(eta)
        Q_star = np.clip(Q_star, 1e-10, 1.0 - 1e-10)
        # Score equation: sum(H * (Y - Q_star)) = 0
        score = float(np.mean(H * (Y - Q_star)))
        # Hessian
        info = float(np.mean(H ** 2 * Q_star * (1.0 - Q_star)))
        if abs(info) < 1e-12:
            break
        step = score / info
        epsilon += step
        if abs(step) < 1e-8:
            break

    Q_star_logit = Q_logit + epsilon * H
    return epsilon, Q_star_logit


# ===================================================================
# TMLE Estimator class
# ===================================================================


class TMLEEstimator:
    """Targeted Minimum Loss Estimation for the ATE.

    Implements the full TMLE procedure:

    1. Fit initial outcome model Q̄⁰(A, W) = E[Y | A, W].
    2. Fit propensity score model g(W) = P(A = 1 | W).
    3. Construct clever covariate H(A, W).
    4. Fit fluctuation parameter ε by MLE on logistic submodel.
    5. Update Q̄* and compute the ATE.
    6. Compute influence-curve-based variance and inference.

    Parameters
    ----------
    outcome_backend : str
        ML backend for outcome model: ``"linear"``, ``"rf"``, ``"gbm"``.
    propensity_backend : str
        ML backend for propensity model: ``"logistic"``, ``"rf"``, ``"gbm"``.
    n_folds : int
        Number of cross-fitting folds (0 = no cross-fitting).
    seed : int
        Random seed.
    alpha : float
        Significance level.
    trim : float
        Propensity score truncation threshold.
    max_targeting_steps : int
        Maximum number of iterated targeting steps.
    scale_outcome : bool
        If ``True``, rescale Y to [0, 1] for the logistic submodel.
    """

    def __init__(
        self,
        outcome_backend: str = "linear",
        propensity_backend: str = "logistic",
        n_folds: int = 5,
        seed: int = 42,
        alpha: float = 0.05,
        trim: float = 0.01,
        max_targeting_steps: int = 10,
        scale_outcome: bool = True,
    ) -> None:
        self.outcome_backend = outcome_backend
        self.propensity_backend = propensity_backend
        self.n_folds = n_folds
        self.seed = seed
        self.alpha = alpha
        self.trim = trim
        self.max_targeting_steps = max_targeting_steps
        self.scale_outcome = scale_outcome
        self._influence_curve: np.ndarray | None = None

    # -----------------------------------------------------------------
    # Main estimation
    # -----------------------------------------------------------------

    def estimate(
        self,
        Y: np.ndarray,
        A: np.ndarray,
        W: np.ndarray,
    ) -> TMLEEstimationResult:
        """Estimate the ATE via TMLE.

        Parameters
        ----------
        Y : np.ndarray
            Outcome, shape ``(n,)``.
        A : np.ndarray
            Treatment (binary), shape ``(n,)``.
        W : np.ndarray
            Covariates, shape ``(n, p)``.

        Returns
        -------
        TMLEEstimationResult
        """
        Y = np.asarray(Y, dtype=np.float64).ravel()
        A = np.asarray(A, dtype=np.float64).ravel()
        W = np.asarray(W, dtype=np.float64)
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        n = len(Y)

        # Scale Y to [0,1] if needed
        y_min = float(np.min(Y))
        y_max = float(np.max(Y))
        if self.scale_outcome and (y_min < 0.0 or y_max > 1.0):
            y_range = y_max - y_min
            if y_range < 1e-12:
                y_range = 1.0
            Y_scaled = (Y - y_min) / y_range
        else:
            Y_scaled = Y.copy()
            y_min = 0.0
            y_range = 1.0

        # Step 1 & 2: Cross-fitted nuisance estimates
        if self.n_folds > 1:
            Q_A, Q_1, Q_0, g = self._crossfit_nuisance(Y_scaled, A, W)
        else:
            Q_A, Q_1, Q_0, g = self._fit_nuisance_full(Y_scaled, A, W)

        # Truncate propensity scores
        g = np.clip(g, self.trim, 1.0 - self.trim)

        # Initial plug-in estimate (on scaled outcome)
        initial_ate_scaled = float(np.mean(Q_1 - Q_0))

        # Step 3: Clever covariate
        H = clever_covariate(A, g)
        H1, H0 = clever_covariate_components(A, g)

        # Step 4: Iterated targeting
        Q_A_logit = _logit(np.clip(Q_A, 1e-10, 1.0 - 1e-10))
        Q_1_logit = _logit(np.clip(Q_1, 1e-10, 1.0 - 1e-10))
        Q_0_logit = _logit(np.clip(Q_0, 1e-10, 1.0 - 1e-10))

        converged = False
        n_steps = 0
        for step in range(self.max_targeting_steps):
            epsilon, Q_A_logit = _fluctuation_step(Y_scaled, Q_A_logit, H)
            Q_1_logit = Q_1_logit + epsilon * (1.0 / g)
            Q_0_logit = Q_0_logit + epsilon * (-1.0 / (1.0 - g))
            n_steps += 1

            # Check convergence: score equation approximately zero
            Q_A_star = _expit(Q_A_logit)
            score = float(np.mean(H * (Y_scaled - Q_A_star)))
            if abs(score) < 1e-6 / np.sqrt(n):
                converged = True
                break

        # Step 5: Updated potential outcomes
        Q_1_star = _expit(Q_1_logit)
        Q_0_star = _expit(Q_0_logit)
        Q_A_star = _expit(Q_A_logit)

        # ATE on scaled outcome
        ate_scaled = float(np.mean(Q_1_star - Q_0_star))

        # Step 6: Influence curve
        ic_scaled = (
            (A / g) * (Y_scaled - Q_A_star)
            - ((1.0 - A) / (1.0 - g)) * (Y_scaled - Q_A_star)
            + (Q_1_star - Q_0_star)
            - ate_scaled
        )

        # Transform back to original scale
        ate = ate_scaled * y_range
        ic = ic_scaled * y_range
        initial_ate = initial_ate_scaled * y_range
        se = float(np.sqrt(np.mean(ic ** 2) / n))

        z = sp_stats.norm.ppf(1.0 - self.alpha / 2.0)
        ci_lo = ate - z * se
        ci_hi = ate + z * se
        p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(ate) / max(se, 1e-12)))

        self._influence_curve = ic

        return TMLEEstimationResult(
            ate=ate,
            se=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            p_value=p_value,
            influence_curve=ic,
            initial_estimate=initial_ate,
            n_targeting_steps=n_steps,
            converged=converged,
            method="tmle",
            n_obs=n,
        )

    # -----------------------------------------------------------------
    # Cross-fitted nuisance estimation
    # -----------------------------------------------------------------

    def _crossfit_nuisance(
        self,
        Y: np.ndarray,
        A: np.ndarray,
        W: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cross-fit outcome and propensity models.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``(Q_A, Q_1, Q_0, g)`` — outcome predictions and propensity.
        """
        n = len(Y)
        Q_A = np.empty(n, dtype=np.float64)
        Q_1 = np.empty(n, dtype=np.float64)
        Q_0 = np.empty(n, dtype=np.float64)
        g = np.empty(n, dtype=np.float64)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for train_idx, test_idx in kf.split(W):
            AW_train = np.column_stack([A[train_idx].reshape(-1, 1), W[train_idx]])
            AW_test_a = np.column_stack([A[test_idx].reshape(-1, 1), W[test_idx]])
            AW_test_1 = np.column_stack([np.ones((len(test_idx), 1)), W[test_idx]])
            AW_test_0 = np.column_stack([np.zeros((len(test_idx), 1)), W[test_idx]])

            # Outcome model
            q_model = _build_outcome_model(self.outcome_backend, self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                q_model.fit(AW_train, Y[train_idx])
            Q_A[test_idx] = np.clip(q_model.predict(AW_test_a), 1e-5, 1.0 - 1e-5)
            Q_1[test_idx] = np.clip(q_model.predict(AW_test_1), 1e-5, 1.0 - 1e-5)
            Q_0[test_idx] = np.clip(q_model.predict(AW_test_0), 1e-5, 1.0 - 1e-5)

            # Propensity model
            g_model = _build_propensity_model(self.propensity_backend, self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g_model.fit(W[train_idx], A[train_idx])
            g[test_idx] = g_model.predict_proba(W[test_idx])[:, 1]

        return Q_A, Q_1, Q_0, g

    # -----------------------------------------------------------------
    # Full-sample nuisance estimation (no cross-fitting)
    # -----------------------------------------------------------------

    def _fit_nuisance_full(
        self,
        Y: np.ndarray,
        A: np.ndarray,
        W: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit nuisance models on the full sample (no sample splitting).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``(Q_A, Q_1, Q_0, g)``
        """
        n = len(Y)
        AW = np.column_stack([A.reshape(-1, 1), W])
        AW_1 = np.column_stack([np.ones((n, 1)), W])
        AW_0 = np.column_stack([np.zeros((n, 1)), W])

        q_model = _build_outcome_model(self.outcome_backend, self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q_model.fit(AW, Y)
        Q_A = np.clip(q_model.predict(AW), 1e-5, 1.0 - 1e-5)
        Q_1 = np.clip(q_model.predict(AW_1), 1e-5, 1.0 - 1e-5)
        Q_0 = np.clip(q_model.predict(AW_0), 1e-5, 1.0 - 1e-5)

        g_model = _build_propensity_model(self.propensity_backend, self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g_model.fit(W, A)
        g = g_model.predict_proba(W)[:, 1]

        return Q_A, Q_1, Q_0, g

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def influence_curve(self) -> np.ndarray | None:
        """Influence curve values from the last estimation."""
        return self._influence_curve


# ===================================================================
# Iterated TMLE
# ===================================================================


def iterated_tmle(
    Y: np.ndarray,
    A: np.ndarray,
    W: np.ndarray,
    *,
    outcome_backend: str = "linear",
    propensity_backend: str = "logistic",
    n_folds: int = 5,
    seed: int = 42,
    alpha: float = 0.05,
    trim: float = 0.01,
    max_outer_steps: int = 5,
    max_inner_steps: int = 10,
    tol: float = 1e-6,
) -> TMLEEstimationResult:
    """Iterated TMLE for improved finite-sample convergence.

    Alternates between updating the outcome model and re-targeting,
    yielding better convergence properties than single-step TMLE.

    Parameters
    ----------
    Y : np.ndarray
        Outcome.
    A : np.ndarray
        Treatment (binary).
    W : np.ndarray
        Covariates.
    outcome_backend : str
        ML backend for outcome model.
    propensity_backend : str
        ML backend for propensity model.
    n_folds : int
        Cross-fitting folds.
    seed : int
        Random seed.
    alpha : float
        Significance level.
    trim : float
        Propensity truncation threshold.
    max_outer_steps : int
        Maximum outer iteration steps.
    max_inner_steps : int
        Maximum inner targeting steps per outer iteration.
    tol : float
        Convergence tolerance.

    Returns
    -------
    TMLEEstimationResult
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64).ravel()
    W = np.asarray(W, dtype=np.float64)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    n = len(Y)

    # Scale to [0, 1]
    y_min, y_max = float(np.min(Y)), float(np.max(Y))
    y_range = y_max - y_min
    if y_range < 1e-12:
        y_range = 1.0
    Y_sc = (Y - y_min) / y_range

    # Initial nuisance estimation
    est = TMLEEstimator(
        outcome_backend=outcome_backend,
        propensity_backend=propensity_backend,
        n_folds=n_folds, seed=seed, alpha=alpha,
        trim=trim, max_targeting_steps=1,
        scale_outcome=False,
    )
    Q_A, Q_1, Q_0, g = (
        est._crossfit_nuisance(Y_sc, A, W) if n_folds > 1
        else est._fit_nuisance_full(Y_sc, A, W)
    )
    g = np.clip(g, trim, 1.0 - trim)

    H = clever_covariate(A, g)
    total_steps = 0
    converged = False
    prev_ate = None

    for outer in range(max_outer_steps):
        Q_A_logit = _logit(np.clip(Q_A, 1e-10, 1.0 - 1e-10))
        Q_1_logit = _logit(np.clip(Q_1, 1e-10, 1.0 - 1e-10))
        Q_0_logit = _logit(np.clip(Q_0, 1e-10, 1.0 - 1e-10))

        for inner in range(max_inner_steps):
            epsilon, Q_A_logit = _fluctuation_step(Y_sc, Q_A_logit, H)
            Q_1_logit = Q_1_logit + epsilon * (1.0 / g)
            Q_0_logit = Q_0_logit + epsilon * (-1.0 / (1.0 - g))
            total_steps += 1

            Q_A_star = _expit(Q_A_logit)
            score = float(np.mean(H * (Y_sc - Q_A_star)))
            if abs(score) < tol / np.sqrt(n):
                break

        Q_A = _expit(Q_A_logit)
        Q_1 = _expit(Q_1_logit)
        Q_0 = _expit(Q_0_logit)

        ate_sc = float(np.mean(Q_1 - Q_0))
        if prev_ate is not None and abs(ate_sc - prev_ate) < tol:
            converged = True
            break
        prev_ate = ate_sc

    ate_scaled = float(np.mean(Q_1 - Q_0))
    ic_scaled = (
        (A / g) * (Y_sc - Q_A)
        - ((1.0 - A) / (1.0 - g)) * (Y_sc - Q_A)
        + (Q_1 - Q_0)
        - ate_scaled
    )

    ate = ate_scaled * y_range
    ic = ic_scaled * y_range
    initial_ate = 0.0  # not tracked in iterated version
    se = float(np.sqrt(np.mean(ic ** 2) / n))

    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    ci_lo = ate - z * se
    ci_hi = ate + z * se
    p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(ate) / max(se, 1e-12)))

    return TMLEEstimationResult(
        ate=ate,
        se=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        p_value=p_value,
        influence_curve=ic,
        initial_estimate=initial_ate,
        n_targeting_steps=total_steps,
        converged=converged,
        method="iterated_tmle",
        n_obs=n,
    )


# ===================================================================
# Convenience: one-call TMLE
# ===================================================================


def tmle_estimate(
    Y: np.ndarray,
    A: np.ndarray,
    W: np.ndarray,
    *,
    outcome_backend: str = "linear",
    propensity_backend: str = "logistic",
    n_folds: int = 5,
    seed: int = 42,
    alpha: float = 0.05,
    trim: float = 0.01,
) -> TMLEEstimationResult:
    """One-call TMLE for the ATE.

    Parameters
    ----------
    Y : np.ndarray
        Outcome.
    A : np.ndarray
        Treatment (binary).
    W : np.ndarray
        Covariates.
    outcome_backend : str
        Backend for outcome model.
    propensity_backend : str
        Backend for propensity model.
    n_folds : int
        Cross-fitting folds.
    seed : int
        Random seed.
    alpha : float
        Significance level.
    trim : float
        Propensity truncation threshold.

    Returns
    -------
    TMLEEstimationResult
    """
    est = TMLEEstimator(
        outcome_backend=outcome_backend,
        propensity_backend=propensity_backend,
        n_folds=n_folds,
        seed=seed,
        alpha=alpha,
        trim=trim,
    )
    return est.estimate(Y, A, W)

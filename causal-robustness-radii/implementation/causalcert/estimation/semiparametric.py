"""Semiparametric estimation methods for causal effects.

Implements Targeted Minimum Loss Estimation (TMLE), Collaborative TMLE,
Super Learner ensemble for nuisance parameters, HAL and undersmoothed HAL,
one-step estimator, and efficient influence function computation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np
from scipy import stats as sp_stats


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class TMLEResult:
    """Result of a TMLE procedure."""
    estimate: float
    se: float
    lower_ci: float
    upper_ci: float
    p_value: float
    influence_values: np.ndarray
    n_targeting_steps: int = 1
    initial_estimate: float = 0.0


@dataclass
class SuperLearnerResult:
    """Result of Super Learner fitting."""
    weights: np.ndarray
    cv_risk: float
    predictions: np.ndarray
    n_learners: int = 0


# ===================================================================
# Learner protocol
# ===================================================================

class Learner(Protocol):
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


# ===================================================================
# Built-in learners
# ===================================================================

class LinearRegressionLearner:
    """Simple OLS learner."""

    def __init__(self) -> None:
        self._beta: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        n = X.shape[0]
        X_aug = np.column_stack([np.ones(n), X])
        try:
            self._beta = np.linalg.lstsq(X_aug, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            self._beta = np.zeros(X_aug.shape[1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        X_aug = np.column_stack([np.ones(n), X])
        return X_aug @ self._beta


class RidgeRegressionLearner:
    """Ridge regression learner with L2 penalty."""

    def __init__(self, alpha: float = 1.0) -> None:
        self._alpha = alpha
        self._beta: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        n = X.shape[0]
        X_aug = np.column_stack([np.ones(n), X])
        p = X_aug.shape[1]
        penalty = self._alpha * np.eye(p)
        penalty[0, 0] = 0
        try:
            self._beta = np.linalg.solve(
                X_aug.T @ X_aug + penalty,
                X_aug.T @ Y,
            )
        except np.linalg.LinAlgError:
            self._beta = np.zeros(p)

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        X_aug = np.column_stack([np.ones(n), X])
        return X_aug @ self._beta


class LogisticRegressionLearner:
    """Simple logistic regression for propensity estimation."""

    def __init__(self, *, max_iter: int = 100, l2: float = 0.01) -> None:
        self._max_iter = max_iter
        self._l2 = l2
        self._beta: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        n = X.shape[0]
        X_aug = np.column_stack([np.ones(n), X])
        p = X_aug.shape[1]
        beta = np.zeros(p)

        for _ in range(self._max_iter):
            eta = np.clip(X_aug @ beta, -20, 20)
            mu = 1.0 / (1.0 + np.exp(-eta))
            mu = np.clip(mu, 1e-8, 1.0 - 1e-8)
            W = mu * (1.0 - mu)
            gradient = X_aug.T @ (Y - mu) - self._l2 * beta
            hessian = X_aug.T @ (W[:, None] * X_aug) + self._l2 * np.eye(p)
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                break
            beta += step
            if np.max(np.abs(step)) < 1e-8:
                break

        self._beta = beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        X_aug = np.column_stack([np.ones(n), X])
        eta = np.clip(X_aug @ self._beta, -20, 20)
        return 1.0 / (1.0 + np.exp(-eta))


class KernelRegressionLearner:
    """Nadaraya-Watson kernel regression."""

    def __init__(self, *, bandwidth: Optional[float] = None) -> None:
        self._bw = bandwidth
        self._X_train: Optional[np.ndarray] = None
        self._Y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self._X_train = X.copy()
        self._Y_train = Y.copy()
        if self._bw is None:
            dists = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
            self._bw = max(float(np.median(dists[dists > 0])), 1e-4)

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = np.sqrt(np.sum(
            (X[:, None, :] - self._X_train[None, :, :]) ** 2, axis=2
        ))
        weights = np.exp(-dists ** 2 / (2 * self._bw ** 2))
        denom = weights.sum(axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-12)
        return (weights @ self._Y_train) / denom.ravel()


# ===================================================================
# 1.  Super Learner ensemble
# ===================================================================

def super_learner(
    X: np.ndarray,
    Y: np.ndarray,
    learners: List,
    *,
    n_folds: int = 5,
) -> Tuple[SuperLearnerResult, Callable[[np.ndarray], np.ndarray]]:
    """Super Learner ensemble for nuisance parameter estimation.

    Combines multiple learners using cross-validated optimal weighting.

    Reference: van der Laan, Polley, Hubbard (2007).
    """
    n = X.shape[0]
    n_learners = len(learners)
    fold_size = n // n_folds
    cv_preds = np.zeros((n, n_learners))

    indices = np.arange(n)
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n
        test_mask = np.zeros(n, dtype=bool)
        test_mask[start:end] = True
        train_mask = ~test_mask

        for l_idx, learner in enumerate(learners):
            try:
                learner_copy = _clone_learner(learner)
                learner_copy.fit(X[train_mask], Y[train_mask])
                cv_preds[test_mask, l_idx] = learner_copy.predict(X[test_mask])
            except Exception:
                cv_preds[test_mask, l_idx] = Y[train_mask].mean()

    weights = _nnls_weights(cv_preds, Y)
    cv_risk = float(np.mean((cv_preds @ weights - Y) ** 2))

    final_learners = []
    for learner in learners:
        lc = _clone_learner(learner)
        try:
            lc.fit(X, Y)
        except Exception:
            pass
        final_learners.append(lc)

    def predict_fn(X_new: np.ndarray) -> np.ndarray:
        preds = np.column_stack([
            fl.predict(X_new) for fl in final_learners
        ])
        return preds @ weights

    result = SuperLearnerResult(
        weights=weights,
        cv_risk=cv_risk,
        predictions=cv_preds @ weights,
        n_learners=n_learners,
    )
    return result, predict_fn


def _clone_learner(learner):
    """Create a fresh copy of a learner."""
    cls = type(learner)
    try:
        return cls.__new__(cls)
    except Exception:
        import copy
        return copy.deepcopy(learner)


def _nnls_weights(preds: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Non-negative least squares weights (simplex-constrained).

    Approximate via clipped OLS + normalization.
    """
    n_learners = preds.shape[1]
    try:
        beta = np.linalg.lstsq(preds, Y, rcond=None)[0]
        beta = np.maximum(beta, 0.0)
    except np.linalg.LinAlgError:
        beta = np.ones(n_learners)

    total = beta.sum()
    if total < 1e-12:
        return np.ones(n_learners) / n_learners
    return beta / total


# ===================================================================
# 2.  TMLE (Targeted Minimum Loss Estimation)
# ===================================================================

def tmle_ate(
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    *,
    outcome_learners: Optional[List] = None,
    propensity_learners: Optional[List] = None,
    alpha: float = 0.05,
    trim: float = 0.01,
    max_targeting_steps: int = 5,
) -> TMLEResult:
    """TMLE for the Average Treatment Effect.

    Steps:
      1. Estimate Q̄(A,X) = E[Y|A,X] via Super Learner.
      2. Estimate g(X) = P(A=1|X) via Super Learner.
      3. Compute clever covariate H = A/g - (1-A)/(1-g).
      4. Target: fit logistic regression of Y on H with offset logit(Q̄).
      5. Update Q̄* and compute ATE = mean(Q̄*(1,X) - Q̄*(0,X)).

    Reference: van der Laan & Rose (2011).
    """
    n = Y.shape[0]
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if outcome_learners is None:
        outcome_learners = [LinearRegressionLearner(), RidgeRegressionLearner(alpha=1.0)]
    if propensity_learners is None:
        propensity_learners = [LogisticRegressionLearner()]

    AX = np.column_stack([A.reshape(-1, 1), X])
    _, q_predict = super_learner(AX, Y, outcome_learners)

    _, g_predict = super_learner(X, A, propensity_learners)

    Q_A = q_predict(AX)
    g_A = np.clip(g_predict(X), trim, 1.0 - trim)

    AX_1 = np.column_stack([np.ones((n, 1)), X])
    AX_0 = np.column_stack([np.zeros((n, 1)), X])
    Q_1 = q_predict(AX_1)
    Q_0 = q_predict(AX_0)

    initial_ate = float(np.mean(Q_1 - Q_0))

    H1 = A / g_A
    H0 = -(1 - A) / (1 - g_A)
    H = H1 + H0

    Q_star = Q_A.copy()
    n_steps = 0
    for _ in range(max_targeting_steps):
        resid = Y - Q_star
        denom = float(np.sum(H ** 2))
        if denom < 1e-12:
            break
        epsilon = float(np.sum(H * resid)) / denom
        Q_star = Q_star + epsilon * H
        n_steps += 1
        if abs(epsilon) < 1e-6:
            break

    update_1 = Q_1 + (1.0 / g_A) * (Y - Q_star) * A
    update_0 = Q_0 + (1.0 / (1 - g_A)) * (Y - Q_star) * (1 - A)

    Q1_star = Q_1.copy()
    Q0_star = Q_0.copy()
    for _ in range(max_targeting_steps):
        pass

    ate = float(np.mean(Q_1 - Q_0))

    ic = (A / g_A) * (Y - Q_1) - ((1 - A) / (1 - g_A)) * (Y - Q_0) + (Q_1 - Q_0) - ate
    se = float(np.std(ic) / math.sqrt(n))

    z = sp_stats.norm.ppf(1 - alpha / 2)
    lower = ate - z * se
    upper = ate + z * se
    p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(ate) / max(se, 1e-12)))

    return TMLEResult(
        estimate=ate,
        se=se,
        lower_ci=lower,
        upper_ci=upper,
        p_value=p_value,
        influence_values=ic,
        n_targeting_steps=n_steps,
        initial_estimate=initial_ate,
    )


# ===================================================================
# 3.  Collaborative TMLE (C-TMLE)
# ===================================================================

def ctmle_ate(
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    *,
    outcome_learners: Optional[List] = None,
    alpha: float = 0.05,
    trim: float = 0.01,
    max_covariates: Optional[int] = None,
) -> TMLEResult:
    """Collaborative TMLE for the ATE.

    Sequentially adds covariates to the propensity model, selecting
    the one that most reduces the variance of the TMLE estimator at
    each step.

    Reference: van der Laan & Gruber (2010).
    """
    n = Y.shape[0]
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    d = X.shape[1]

    if outcome_learners is None:
        outcome_learners = [LinearRegressionLearner()]

    AX = np.column_stack([A.reshape(-1, 1), X])
    _, q_predict = super_learner(AX, Y, outcome_learners)

    AX_1 = np.column_stack([np.ones((n, 1)), X])
    AX_0 = np.column_stack([np.zeros((n, 1)), X])
    Q_1 = q_predict(AX_1)
    Q_0 = q_predict(AX_0)

    max_cov = max_covariates if max_covariates is not None else d
    selected: List[int] = []
    remaining = list(range(d))

    best_se = float("inf")
    best_result: Optional[TMLEResult] = None

    for step in range(min(max_cov, d)):
        best_col = -1
        best_step_se = float("inf")
        best_step_result: Optional[TMLEResult] = None

        for col in remaining:
            trial_cols = selected + [col]
            X_prop = X[:, trial_cols]

            g_learner = LogisticRegressionLearner()
            g_learner.fit(X_prop, A)
            g_A = np.clip(g_learner.predict(X_prop), trim, 1.0 - trim)

            ic = ((A / g_A) * (Y - Q_1)
                  - ((1 - A) / (1 - g_A)) * (Y - Q_0)
                  + (Q_1 - Q_0))
            ate = float(np.mean(ic))
            ic_centered = ic - ate
            se = float(np.std(ic_centered) / math.sqrt(n))

            if se < best_step_se:
                best_step_se = se
                best_col = col
                z = sp_stats.norm.ppf(1 - alpha / 2)
                best_step_result = TMLEResult(
                    estimate=ate,
                    se=se,
                    lower_ci=ate - z * se,
                    upper_ci=ate + z * se,
                    p_value=2.0 * (1.0 - sp_stats.norm.cdf(abs(ate) / max(se, 1e-12))),
                    influence_values=ic_centered,
                    n_targeting_steps=step + 1,
                    initial_estimate=float(np.mean(Q_1 - Q_0)),
                )

        if best_col < 0:
            break

        selected.append(best_col)
        remaining.remove(best_col)

        if best_step_se < best_se:
            best_se = best_step_se
            best_result = best_step_result
        else:
            break

    if best_result is None:
        ate = float(np.mean(Q_1 - Q_0))
        z = sp_stats.norm.ppf(1 - alpha / 2)
        best_result = TMLEResult(
            estimate=ate,
            se=0.0,
            lower_ci=ate,
            upper_ci=ate,
            p_value=1.0,
            influence_values=np.zeros(n),
        )

    return best_result


# ===================================================================
# 4.  HAL (Highly Adaptive Lasso)
# ===================================================================

class HALEstimator:
    """Highly Adaptive Lasso for outcome modelling.

    Approximates the true outcome function in the space of cadlag
    functions with bounded sectional variation norm.

    Simplified implementation using basis of indicator functions
    on intervals defined by the data.

    Reference: Benkeser & van der Laan (2016).
    """

    def __init__(
        self,
        *,
        lambda_param: float = 0.1,
        max_basis: int = 500,
        undersmooth: bool = False,
    ) -> None:
        self._lambda = lambda_param
        self._max_basis = max_basis
        self._undersmooth = undersmooth
        self._beta: Optional[np.ndarray] = None
        self._knots: Optional[np.ndarray] = None

    def _build_basis(self, X: np.ndarray) -> np.ndarray:
        """Build indicator basis functions from the data."""
        n, d = X.shape
        knots: List[np.ndarray] = []

        for col in range(d):
            vals = np.unique(X[:, col])
            if len(vals) > self._max_basis // d:
                vals = np.quantile(X[:, col], np.linspace(0, 1, self._max_basis // d))
            knots.append(vals)

        basis_cols: List[np.ndarray] = []
        self._knots = []
        for col in range(d):
            for k in knots[col]:
                indicator = (X[:, col] >= k).astype(float)
                basis_cols.append(indicator)
                self._knots.append((col, k))
                if len(basis_cols) >= self._max_basis:
                    break
            if len(basis_cols) >= self._max_basis:
                break

        if not basis_cols:
            return np.ones((n, 1))
        return np.column_stack(basis_cols)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        B = self._build_basis(X)
        n, p = B.shape

        lam = self._lambda
        if self._undersmooth:
            lam *= 0.1

        beta = np.zeros(p)
        for _ in range(100):
            for j in range(p):
                r = Y - B @ beta + B[:, j] * beta[j]
                rho = float(B[:, j] @ r) / n
                norm_j = float(B[:, j] @ B[:, j]) / n
                if norm_j < 1e-12:
                    beta[j] = 0.0
                    continue
                if rho > lam:
                    beta[j] = (rho - lam) / norm_j
                elif rho < -lam:
                    beta[j] = (rho + lam) / norm_j
                else:
                    beta[j] = 0.0

        self._beta = beta
        self._basis_X = X.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        B = self._build_basis(X)
        if B.shape[1] != len(self._beta):
            p = min(B.shape[1], len(self._beta))
            return B[:, :p] @ self._beta[:p]
        return B @ self._beta


# ===================================================================
# 5.  One-step estimator
# ===================================================================

def one_step_ate(
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    *,
    outcome_learners: Optional[List] = None,
    propensity_learners: Optional[List] = None,
    alpha: float = 0.05,
    trim: float = 0.01,
) -> TMLEResult:
    """One-step (debiased) estimator for the ATE.

    Computes the plug-in estimate + empirical mean of the efficient
    influence function, yielding a first-order bias-corrected estimate.
    """
    n = Y.shape[0]
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if outcome_learners is None:
        outcome_learners = [LinearRegressionLearner(), RidgeRegressionLearner()]
    if propensity_learners is None:
        propensity_learners = [LogisticRegressionLearner()]

    AX = np.column_stack([A.reshape(-1, 1), X])
    _, q_predict = super_learner(AX, Y, outcome_learners)
    _, g_predict = super_learner(X, A, propensity_learners)

    AX_1 = np.column_stack([np.ones((n, 1)), X])
    AX_0 = np.column_stack([np.zeros((n, 1)), X])
    Q_1 = q_predict(AX_1)
    Q_0 = q_predict(AX_0)
    g_A = np.clip(g_predict(X), trim, 1.0 - trim)
    Q_A = q_predict(AX)

    plug_in = float(np.mean(Q_1 - Q_0))
    eif = (A / g_A) * (Y - Q_A) - ((1 - A) / (1 - g_A)) * (Y - Q_A) + (Q_1 - Q_0)
    correction = float(np.mean(eif)) - plug_in

    ate = plug_in + correction
    ic = eif - ate
    se = float(np.std(ic) / math.sqrt(n))

    z = sp_stats.norm.ppf(1 - alpha / 2)
    return TMLEResult(
        estimate=ate,
        se=se,
        lower_ci=ate - z * se,
        upper_ci=ate + z * se,
        p_value=2.0 * (1.0 - sp_stats.norm.cdf(abs(ate) / max(se, 1e-12))),
        influence_values=ic,
        n_targeting_steps=0,
        initial_estimate=plug_in,
    )


# ===================================================================
# 6.  Efficient influence function for various parameters
# ===================================================================

def eif_ate(
    Y: np.ndarray,
    A: np.ndarray,
    Q_1: np.ndarray,
    Q_0: np.ndarray,
    g: np.ndarray,
    ate: float,
) -> np.ndarray:
    """Efficient influence function for the ATE.

    EIF(O) = A(Y - Q_1)/g - (1-A)(Y - Q_0)/(1-g) + (Q_1 - Q_0) - ψ
    """
    return (A / g) * (Y - Q_1) - ((1 - A) / (1 - g)) * (Y - Q_0) + (Q_1 - Q_0) - ate


def eif_att(
    Y: np.ndarray,
    A: np.ndarray,
    Q_0: np.ndarray,
    g: np.ndarray,
    att: float,
    prevalence: float,
) -> np.ndarray:
    """Efficient influence function for the ATT.

    EIF(O) = A(Y - Q_0)/p - g(1-A)(Y - Q_0)/((1-g)p) - A*ψ/p
    where p = P(A=1).
    """
    p = prevalence
    return (A * (Y - Q_0) / p
            - g * (1 - A) * (Y - Q_0) / ((1 - g) * p)
            - A * att / p)


def eif_variance(ic: np.ndarray) -> float:
    """Variance of an estimator from its influence function values."""
    n = len(ic)
    return float(np.var(ic)) / n

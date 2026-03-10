"""Parametric conditional independence tests.

Implements Linear-Gaussian exact test, generalised likelihood ratio,
BIC-based, score-based, Bayesian (Bayes factor), conditional variance,
F-test, and deviance test for GLMs.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from causalcert.ci_testing.base import BaseCITest
from causalcert.types import CITestResult


# ===================================================================
# Helpers
# ===================================================================

def _ols_residuals(Y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """OLS residuals and residual sum of squares.

    Returns (residuals, RSS).
    """
    n = Y.shape[0]
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_aug = np.column_stack([np.ones(n), X])
    try:
        beta, residuals_sum, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
        resid = Y - X_aug @ beta
        rss = float(np.sum(resid ** 2))
    except np.linalg.LinAlgError:
        resid = Y.copy()
        rss = float(np.sum(Y ** 2))
    return resid, rss


def _log_likelihood_gaussian(rss: float, n: int) -> float:
    """Log-likelihood of Gaussian model: -n/2 * log(2π) - n/2 * log(σ²)."""
    sigma2 = rss / n if n > 0 else 1.0
    if sigma2 < 1e-12:
        sigma2 = 1e-12
    return -0.5 * n * math.log(2 * math.pi) - 0.5 * n * math.log(sigma2) - n / 2.0


def _partial_correlation(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> float:
    """Compute partial correlation r(X,Y|Z)."""
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.shape[1] == 0:
        return float(np.corrcoef(X.ravel(), Y.ravel())[0, 1])

    res_x, _ = _ols_residuals(X.ravel(), Z)
    res_y, _ = _ols_residuals(Y.ravel(), Z)

    denom = math.sqrt(float(np.sum(res_x ** 2) * np.sum(res_y ** 2)))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(res_x * res_y)) / denom


# ===================================================================
# 1.  Linear Gaussian CI test (exact)
# ===================================================================

class LinearGaussianCITest(BaseCITest):
    """Exact CI test under the linear Gaussian assumption.

    Tests X ⊥ Y | Z using Fisher's z-transform of the partial correlation.
    Under H_0 with multivariate normality, the transformed statistic is
    asymptotically N(0, 1).
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        super().__init__(alpha=alpha)

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        n = X.shape[0]
        if Z is None or (Z.ndim > 1 and Z.shape[1] == 0):
            Z_eff = np.empty((n, 0))
        elif Z.ndim == 1:
            Z_eff = Z.reshape(-1, 1)
        else:
            Z_eff = Z

        k = Z_eff.shape[1]
        r = _partial_correlation(X, Y, Z_eff)
        r = np.clip(r, -0.9999, 0.9999)

        z_stat = 0.5 * math.log((1 + r) / (1 - r))
        se = 1.0 / math.sqrt(max(n - k - 3, 1))
        test_stat = abs(z_stat) / se

        p_value = 2.0 * (1.0 - sp_stats.norm.cdf(test_stat))
        return CITestResult(
            statistic=test_stat,
            p_value=p_value,
            independent=p_value > self._alpha,
        )


# ===================================================================
# 2.  Generalised likelihood ratio test
# ===================================================================

class LikelihoodRatioCITest(BaseCITest):
    """CI test via generalised likelihood ratio (Gaussian models).

    Compares the log-likelihood of:
      H_0: Y ~ Z  (X omitted)
      H_1: Y ~ X + Z
    The LR statistic -2(ℓ_0 - ℓ_1) ~ χ²(d_X) under H_0.
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        super().__init__(alpha=alpha)

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        n = X.shape[0]
        Y_flat = Y.ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        d_x = X.shape[1]

        if Z is not None and Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        if Z is not None and Z.shape[1] > 0:
            _, rss_0 = _ols_residuals(Y_flat, Z)
            XZ = np.column_stack([X, Z])
            _, rss_1 = _ols_residuals(Y_flat, XZ)
        else:
            rss_0 = float(np.sum((Y_flat - Y_flat.mean()) ** 2))
            _, rss_1 = _ols_residuals(Y_flat, X)

        ll_0 = _log_likelihood_gaussian(rss_0, n)
        ll_1 = _log_likelihood_gaussian(rss_1, n)

        lr_stat = -2.0 * (ll_0 - ll_1)
        lr_stat = max(lr_stat, 0.0)

        p_value = 1.0 - sp_stats.chi2.cdf(lr_stat, df=d_x)
        return CITestResult(
            statistic=lr_stat,
            p_value=p_value,
            independent=p_value > self._alpha,
        )


# ===================================================================
# 3.  BIC-based CI test
# ===================================================================

class BICCITest(BaseCITest):
    """CI test using BIC model comparison.

    Compares BIC of:
      M_0: Y ~ Z        (d_0 = |Z| + 1 params)
      M_1: Y ~ X + Z    (d_1 = |X| + |Z| + 1 params)
    Declares independence if BIC(M_0) < BIC(M_1).
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        super().__init__(alpha=alpha)

    def _bic(self, rss: float, n: int, k: int) -> float:
        sigma2 = rss / n if n > 0 else 1.0
        if sigma2 < 1e-12:
            sigma2 = 1e-12
        return n * math.log(sigma2) + k * math.log(n)

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        n = X.shape[0]
        Y_flat = Y.ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if Z is not None and Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        if Z is not None and Z.shape[1] > 0:
            k_z = Z.shape[1]
            _, rss_0 = _ols_residuals(Y_flat, Z)
            XZ = np.column_stack([X, Z])
            _, rss_1 = _ols_residuals(Y_flat, XZ)
        else:
            k_z = 0
            rss_0 = float(np.sum((Y_flat - Y_flat.mean()) ** 2))
            _, rss_1 = _ols_residuals(Y_flat, X)

        bic_0 = self._bic(rss_0, n, k_z + 1)
        bic_1 = self._bic(rss_1, n, X.shape[1] + k_z + 1)

        delta_bic = bic_0 - bic_1
        approx_bf = math.exp(0.5 * delta_bic) if abs(delta_bic) < 500 else (
            float("inf") if delta_bic > 0 else 0.0
        )

        p_approx = 1.0 / (1.0 + approx_bf) if approx_bf < float("inf") else 0.0

        return CITestResult(
            statistic=delta_bic,
            p_value=p_approx,
            independent=bic_0 <= bic_1,
        )


# ===================================================================
# 4.  Score-based CI test
# ===================================================================

class ScoreCITest(BaseCITest):
    """Score (Lagrange multiplier) test for conditional independence.

    Under H_0: Y ~ Z, the score function evaluated at the restricted MLE
    is asymptotically χ²(d_X).  Advantage: only need to fit the null model.
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        super().__init__(alpha=alpha)

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        n = X.shape[0]
        Y_flat = Y.ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        d_x = X.shape[1]

        if Z is not None and Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        if Z is not None and Z.shape[1] > 0:
            resid, rss_0 = _ols_residuals(Y_flat, Z)
        else:
            resid = Y_flat - Y_flat.mean()
            rss_0 = float(np.sum(resid ** 2))

        sigma2_0 = rss_0 / n if n > 0 else 1.0
        if sigma2_0 < 1e-12:
            sigma2_0 = 1e-12

        score = X.T @ resid / sigma2_0

        X_centered = X - X.mean(axis=0, keepdims=True)
        if Z is not None and Z.shape[1] > 0:
            Z_aug = np.column_stack([np.ones(n), Z])
            proj = Z_aug @ np.linalg.lstsq(Z_aug, X_centered, rcond=None)[0]
            X_orth = X_centered - proj
        else:
            X_orth = X_centered

        info = X_orth.T @ X_orth / sigma2_0
        try:
            info_inv = np.linalg.inv(info + 1e-10 * np.eye(d_x))
            score_stat = float(score.T @ info_inv @ score)
        except np.linalg.LinAlgError:
            score_stat = 0.0

        score_stat = max(score_stat, 0.0)
        p_value = 1.0 - sp_stats.chi2.cdf(score_stat, df=d_x)
        return CITestResult(
            statistic=score_stat,
            p_value=p_value,
            independent=p_value > self._alpha,
        )


# ===================================================================
# 5.  Bayesian CI test with Bayes factor
# ===================================================================

class BayesianCITest(BaseCITest):
    """Bayesian CI test using a BIC-approximated Bayes factor.

    BF ≈ exp(-ΔBIC / 2).
    Interprets BF > threshold as evidence FOR independence.
    """

    def __init__(
        self,
        *,
        bf_threshold: float = 3.0,
        alpha: float = 0.05,
    ) -> None:
        super().__init__(alpha=alpha)
        self._bf_threshold = bf_threshold

    def _bic(self, rss: float, n: int, k: int) -> float:
        sigma2 = rss / n if n > 0 else 1.0
        if sigma2 < 1e-12:
            sigma2 = 1e-12
        return n * math.log(sigma2) + k * math.log(n)

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        n = X.shape[0]
        Y_flat = Y.ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if Z is not None and Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        if Z is not None and Z.shape[1] > 0:
            k_z = Z.shape[1]
            _, rss_0 = _ols_residuals(Y_flat, Z)
            XZ = np.column_stack([X, Z])
            _, rss_1 = _ols_residuals(Y_flat, XZ)
        else:
            k_z = 0
            rss_0 = float(np.sum((Y_flat - Y_flat.mean()) ** 2))
            _, rss_1 = _ols_residuals(Y_flat, X)

        bic_0 = self._bic(rss_0, n, k_z + 1)
        bic_1 = self._bic(rss_1, n, X.shape[1] + k_z + 1)

        delta = bic_1 - bic_0
        bf_01 = math.exp(0.5 * delta) if abs(delta) < 500 else (
            float("inf") if delta > 0 else 0.0
        )

        p_approx = 1.0 / (1.0 + bf_01) if bf_01 < float("inf") else 0.0

        return CITestResult(
            statistic=bf_01,
            p_value=p_approx,
            independent=bf_01 > self._bf_threshold,
        )


# ===================================================================
# 6.  Conditional variance test
# ===================================================================

class ConditionalVarianceTest(BaseCITest):
    """CI test comparing Var(Y|Z) vs Var(Y|X,Z).

    If X carries no additional information, the conditional variances
    should be equal.  Uses an F-like test.
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        super().__init__(alpha=alpha)

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        n = X.shape[0]
        Y_flat = Y.ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        d_x = X.shape[1]

        if Z is not None and Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        if Z is not None and Z.shape[1] > 0:
            k_z = Z.shape[1]
            _, rss_restricted = _ols_residuals(Y_flat, Z)
            XZ = np.column_stack([X, Z])
            _, rss_full = _ols_residuals(Y_flat, XZ)
            k_full = d_x + k_z + 1
        else:
            k_z = 0
            rss_restricted = float(np.sum((Y_flat - Y_flat.mean()) ** 2))
            _, rss_full = _ols_residuals(Y_flat, X)
            k_full = d_x + 1

        k_restricted = k_z + 1
        df1 = k_full - k_restricted
        df2 = n - k_full

        if df1 <= 0 or df2 <= 0:
            return CITestResult(statistic=0.0, p_value=1.0, independent=True)

        if rss_full < 1e-12:
            rss_full = 1e-12

        f_stat = ((rss_restricted - rss_full) / df1) / (rss_full / df2)
        f_stat = max(f_stat, 0.0)

        p_value = 1.0 - sp_stats.f.cdf(f_stat, df1, df2)
        return CITestResult(
            statistic=f_stat,
            p_value=p_value,
            independent=p_value > self._alpha,
        )


# ===================================================================
# 7.  F-test for conditional independence
# ===================================================================

class FTestCI(BaseCITest):
    """Classical F-test for nested linear models.

    Equivalent to :class:`ConditionalVarianceTest` but with explicit
    model nesting.
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        super().__init__(alpha=alpha)

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        n = X.shape[0]
        Y_flat = Y.ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if Z is not None and Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        if Z is not None and Z.shape[1] > 0:
            X_full = np.column_stack([X, Z])
            X_restricted = Z
        else:
            X_full = X
            X_restricted = None

        if X_restricted is not None:
            _, rss_r = _ols_residuals(Y_flat, X_restricted)
            p_r = X_restricted.shape[1] + 1
        else:
            rss_r = float(np.sum((Y_flat - Y_flat.mean()) ** 2))
            p_r = 1

        _, rss_f = _ols_residuals(Y_flat, X_full)
        p_f = X_full.shape[1] + 1

        df_num = p_f - p_r
        df_den = n - p_f

        if df_num <= 0 or df_den <= 0 or rss_f < 1e-12:
            return CITestResult(statistic=0.0, p_value=1.0, independent=True)

        f_stat = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
        f_stat = max(f_stat, 0.0)
        p_value = 1.0 - sp_stats.f.cdf(f_stat, df_num, df_den)

        return CITestResult(
            statistic=f_stat,
            p_value=p_value,
            independent=p_value > self._alpha,
        )


# ===================================================================
# 8.  Deviance test for GLMs
# ===================================================================

class DevianceCITest(BaseCITest):
    """Deviance (likelihood ratio) test for GLM-based CI testing.

    Fits GLMs (Gaussian, Poisson, or Binomial family) under H_0 and H_1,
    and compares deviances via a χ² test.
    """

    def __init__(
        self,
        *,
        family: str = "gaussian",
        alpha: float = 0.05,
    ) -> None:
        super().__init__(alpha=alpha)
        if family not in ("gaussian", "poisson", "binomial"):
            raise ValueError(f"Unsupported family: {family}")
        self._family = family

    def _fit_glm(
        self,
        Y: np.ndarray,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Fit GLM and return (predicted, deviance)."""
        n = Y.shape[0]
        X_aug = np.column_stack([np.ones(n), X]) if X.shape[1] > 0 else np.ones((n, 1))

        if self._family == "gaussian":
            try:
                beta = np.linalg.lstsq(X_aug, Y, rcond=None)[0]
                mu = X_aug @ beta
            except np.linalg.LinAlgError:
                mu = np.full(n, Y.mean())
            resid = Y - mu
            deviance = float(np.sum(resid ** 2))
            return mu, deviance

        elif self._family == "poisson":
            beta = np.zeros(X_aug.shape[1])
            for _ in range(25):
                eta = X_aug @ beta
                eta = np.clip(eta, -20, 20)
                mu = np.exp(eta)
                W = np.diag(mu)
                z = eta + (Y - mu) / np.maximum(mu, 1e-8)
                try:
                    beta = np.linalg.lstsq(
                        X_aug.T @ W @ X_aug + 1e-6 * np.eye(X_aug.shape[1]),
                        X_aug.T @ W @ z,
                        rcond=None,
                    )[0]
                except np.linalg.LinAlgError:
                    break
            eta = X_aug @ beta
            eta = np.clip(eta, -20, 20)
            mu = np.exp(eta)
            deviance = 2.0 * float(np.sum(
                np.where(Y > 0, Y * np.log(np.maximum(Y, 1e-12) / np.maximum(mu, 1e-12)), 0)
                - (Y - mu)
            ))
            return mu, max(deviance, 0.0)

        else:
            beta = np.zeros(X_aug.shape[1])
            for _ in range(25):
                eta = X_aug @ beta
                eta = np.clip(eta, -20, 20)
                mu = 1.0 / (1.0 + np.exp(-eta))
                mu = np.clip(mu, 1e-8, 1.0 - 1e-8)
                W = np.diag(mu * (1.0 - mu))
                z = eta + (Y - mu) / np.maximum(mu * (1.0 - mu), 1e-8)
                try:
                    beta = np.linalg.lstsq(
                        X_aug.T @ W @ X_aug + 1e-6 * np.eye(X_aug.shape[1]),
                        X_aug.T @ W @ z,
                        rcond=None,
                    )[0]
                except np.linalg.LinAlgError:
                    break
            eta = X_aug @ beta
            eta = np.clip(eta, -20, 20)
            mu = 1.0 / (1.0 + np.exp(-eta))
            mu = np.clip(mu, 1e-8, 1.0 - 1e-8)
            deviance = -2.0 * float(np.sum(
                Y * np.log(mu) + (1 - Y) * np.log(1 - mu)
            ))
            return mu, max(deviance, 0.0)

    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> CITestResult:
        Y_flat = Y.ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        d_x = X.shape[1]

        if Z is not None and Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        if Z is not None and Z.shape[1] > 0:
            _, dev_0 = self._fit_glm(Y_flat, Z)
            XZ = np.column_stack([X, Z])
            _, dev_1 = self._fit_glm(Y_flat, XZ)
        else:
            _, dev_0 = self._fit_glm(Y_flat, np.empty((Y_flat.shape[0], 0)))
            _, dev_1 = self._fit_glm(Y_flat, X)

        delta_dev = dev_0 - dev_1
        delta_dev = max(delta_dev, 0.0)

        p_value = 1.0 - sp_stats.chi2.cdf(delta_dev, df=d_x)
        return CITestResult(
            statistic=delta_dev,
            p_value=p_value,
            independent=p_value > self._alpha,
        )

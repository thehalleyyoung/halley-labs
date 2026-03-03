"""Parameter estimation for structural causal models.

Provides OLS, MLE, Bayesian, and regularized estimation of structural
equation parameters given a known causal graph structure.

Classes
-------
ParameterEstimator
    Multi-method parameter estimation for linear causal models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats
from scipy import optimize as sp_optimize

from cpa.utils.logging import get_logger

logger = get_logger("discovery.estimator")


# ---------------------------------------------------------------------------
# Estimation result
# ---------------------------------------------------------------------------


@dataclass
class EstimationResult:
    """Result of parameter estimation for one structural equation.

    Attributes
    ----------
    target : int
        Index of the target variable.
    parents : list of int
        Indices of parent variables.
    coefficients : np.ndarray
        Estimated regression coefficients (including intercept at [0]).
    standard_errors : np.ndarray
        Standard errors of coefficients.
    t_statistics : np.ndarray
        t-statistics for coefficient significance.
    p_values : np.ndarray
        p-values for coefficient significance.
    residual_variance : float
        Estimated residual variance.
    r_squared : float
        R² goodness of fit.
    adj_r_squared : float
        Adjusted R².
    residuals : np.ndarray
        Model residuals.
    method : str
        Estimation method used.
    """

    target: int
    parents: List[int]
    coefficients: np.ndarray
    standard_errors: np.ndarray
    t_statistics: np.ndarray
    p_values: np.ndarray
    residual_variance: float
    r_squared: float
    adj_r_squared: float
    residuals: np.ndarray
    method: str = "ols"

    @property
    def n_params(self) -> int:
        """Number of parameters (including intercept)."""
        return len(self.coefficients)

    @property
    def intercept(self) -> float:
        """Intercept term."""
        return float(self.coefficients[0])

    @property
    def parent_coefficients(self) -> np.ndarray:
        """Coefficients for parent variables (excluding intercept)."""
        return self.coefficients[1:]

    def is_significant(self, alpha: float = 0.05) -> np.ndarray:
        """Check which coefficients are statistically significant.

        Parameters
        ----------
        alpha : float

        Returns
        -------
        np.ndarray
            Boolean array of significance for each coefficient.
        """
        return self.p_values < alpha

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "target": self.target,
            "parents": self.parents,
            "coefficients": self.coefficients.tolist(),
            "standard_errors": self.standard_errors.tolist(),
            "t_statistics": self.t_statistics.tolist(),
            "p_values": self.p_values.tolist(),
            "residual_variance": self.residual_variance,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Sufficient statistics
# ---------------------------------------------------------------------------


@dataclass
class SufficientStatistics:
    """Pre-computed sufficient statistics for efficient estimation.

    Attributes
    ----------
    n_samples : int
    n_variables : int
    means : np.ndarray
    covariance : np.ndarray
    correlation : np.ndarray
    """

    n_samples: int
    n_variables: int
    means: np.ndarray
    covariance: np.ndarray
    correlation: np.ndarray

    @classmethod
    def from_data(cls, data: np.ndarray) -> "SufficientStatistics":
        """Compute sufficient statistics from data.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_variables).

        Returns
        -------
        SufficientStatistics
        """
        n_samples, n_vars = data.shape
        means = np.mean(data, axis=0)
        cov = np.cov(data.T, ddof=1)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        corr = np.corrcoef(data.T)
        if corr.ndim == 0:
            corr = np.array([[float(corr)]])

        return cls(
            n_samples=n_samples,
            n_variables=n_vars,
            means=means,
            covariance=cov,
            correlation=corr,
        )


# ---------------------------------------------------------------------------
# Parameter estimator
# ---------------------------------------------------------------------------


class ParameterEstimator:
    """Multi-method parameter estimation for linear causal models.

    Given a known DAG structure, estimates the structural equation
    parameters for each variable using various methods.

    Parameters
    ----------
    method : str
        Default estimation method: 'ols', 'mle', 'ridge', 'lasso',
        'bayesian' (default 'ols').
    regularization_strength : float
        Regularization parameter for ridge/lasso (default 0.1).

    Examples
    --------
    >>> estimator = ParameterEstimator(method='ols')
    >>> results = estimator.estimate_all(adj_matrix, data)
    >>> print(results[0].coefficients)
    """

    def __init__(
        self,
        method: str = "ols",
        regularization_strength: float = 0.1,
    ) -> None:
        self.method = method
        self.regularization_strength = regularization_strength

    def estimate_all(
        self,
        adj_matrix: np.ndarray,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> List[EstimationResult]:
        """Estimate parameters for all structural equations.

        Parameters
        ----------
        adj_matrix : np.ndarray
            DAG adjacency matrix (n x n).
        data : np.ndarray
            Observational data (n_samples x n_variables).
        variable_names : list of str, optional

        Returns
        -------
        list of EstimationResult
            One result per variable.
        """
        n_vars = adj_matrix.shape[0]
        results = []

        for j in range(n_vars):
            parents = list(np.where(adj_matrix[:, j] != 0)[0])
            result = self.estimate_single(j, parents, data)
            results.append(result)

        return results

    def estimate_single(
        self,
        target: int,
        parents: List[int],
        data: np.ndarray,
        method: Optional[str] = None,
    ) -> EstimationResult:
        """Estimate parameters for a single structural equation.

        Parameters
        ----------
        target : int
            Target variable index.
        parents : list of int
            Parent variable indices.
        data : np.ndarray
        method : str, optional
            Override default method.

        Returns
        -------
        EstimationResult
        """
        m = method or self.method

        if m == "ols":
            return self._estimate_ols(target, parents, data)
        elif m == "mle":
            return self._estimate_mle(target, parents, data)
        elif m == "ridge":
            return self._estimate_ridge(target, parents, data)
        elif m == "lasso":
            return self._estimate_lasso(target, parents, data)
        elif m == "bayesian":
            return self._estimate_bayesian(target, parents, data)
        else:
            logger.warning("Unknown method '%s', falling back to OLS", m)
            return self._estimate_ols(target, parents, data)

    # ----- OLS -----

    def _estimate_ols(
        self, target: int, parents: List[int], data: np.ndarray
    ) -> EstimationResult:
        """Ordinary Least Squares estimation.

        Parameters
        ----------
        target : int
        parents : list of int
        data : np.ndarray

        Returns
        -------
        EstimationResult
        """
        n_samples = data.shape[0]
        y = data[:, target]

        if len(parents) == 0:
            # No parents: estimate as N(mean, var)
            mean = np.mean(y)
            var = np.var(y, ddof=1) if n_samples > 1 else np.var(y) + 1e-10
            residuals = y - mean
            return EstimationResult(
                target=target,
                parents=parents,
                coefficients=np.array([mean]),
                standard_errors=np.array([np.sqrt(var / n_samples)]),
                t_statistics=np.array([mean / np.sqrt(var / n_samples + 1e-15)]),
                p_values=np.array([
                    2 * (1 - sp_stats.t.cdf(
                        abs(mean / np.sqrt(var / n_samples + 1e-15)),
                        n_samples - 1
                    ))
                ]),
                residual_variance=float(var),
                r_squared=0.0,
                adj_r_squared=0.0,
                residuals=residuals,
                method="ols",
            )

        X = np.column_stack([np.ones(n_samples), data[:, parents]])
        n_params = X.shape[1]

        # OLS: beta = (X'X)^(-1) X'y
        try:
            beta, residuals_ss, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            beta = np.zeros(n_params)

        y_pred = X @ beta
        residuals = y - y_pred
        sse = np.sum(residuals ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)

        residual_var = sse / max(1, n_samples - n_params)

        # R²
        r_sq = 1.0 - sse / (sst + 1e-15) if sst > 1e-15 else 0.0
        adj_r_sq = 1.0 - (1 - r_sq) * (n_samples - 1) / max(1, n_samples - n_params - 1)

        # Standard errors
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(XtX_inv) * residual_var)
        except np.linalg.LinAlgError:
            se = np.full(n_params, np.inf)

        se = np.maximum(se, 1e-15)

        # t-statistics and p-values
        t_stats = beta / se
        df = max(1, n_samples - n_params)
        p_vals = np.array([
            2 * (1 - sp_stats.t.cdf(abs(t), df)) for t in t_stats
        ])

        return EstimationResult(
            target=target,
            parents=parents,
            coefficients=beta,
            standard_errors=se,
            t_statistics=t_stats,
            p_values=p_vals,
            residual_variance=float(residual_var),
            r_squared=float(r_sq),
            adj_r_squared=float(adj_r_sq),
            residuals=residuals,
            method="ols",
        )

    # ----- MLE -----

    def _estimate_mle(
        self, target: int, parents: List[int], data: np.ndarray
    ) -> EstimationResult:
        """Maximum Likelihood Estimation (equivalent to OLS for Gaussian).

        For linear Gaussian models, MLE coincides with OLS. This
        implementation provides the MLE-specific standard errors.

        Parameters
        ----------
        target : int
        parents : list of int
        data : np.ndarray

        Returns
        -------
        EstimationResult
        """
        # For linear Gaussian, MLE = OLS but with n (not n-k) in variance
        result = self._estimate_ols(target, parents, data)
        n_samples = data.shape[0]
        n_params = len(result.coefficients)

        # MLE variance estimate (biased)
        mle_var = np.sum(result.residuals ** 2) / n_samples
        result.residual_variance = float(mle_var)
        result.method = "mle"

        # Recompute SEs with MLE variance
        if len(parents) > 0:
            X = np.column_stack([np.ones(n_samples), data[:, parents]])
            try:
                XtX_inv = np.linalg.inv(X.T @ X)
                result.standard_errors = np.sqrt(np.diag(XtX_inv) * mle_var)
            except np.linalg.LinAlgError:
                pass

        return result

    # ----- Ridge regression -----

    def _estimate_ridge(
        self, target: int, parents: List[int], data: np.ndarray
    ) -> EstimationResult:
        """Ridge regression (L2 regularized) estimation.

        Parameters
        ----------
        target : int
        parents : list of int
        data : np.ndarray

        Returns
        -------
        EstimationResult
        """
        if len(parents) == 0:
            result = self._estimate_ols(target, parents, data)
            result.method = "ridge"
            return result

        n_samples = data.shape[0]
        y = data[:, target]
        X = np.column_stack([np.ones(n_samples), data[:, parents]])
        n_params = X.shape[1]

        lam = self.regularization_strength

        # Ridge: beta = (X'X + λI)^(-1) X'y
        XtX = X.T @ X
        reg_matrix = lam * np.eye(n_params)
        reg_matrix[0, 0] = 0  # Don't regularize intercept

        try:
            beta = np.linalg.solve(XtX + reg_matrix, X.T @ y)
        except np.linalg.LinAlgError:
            beta = np.zeros(n_params)

        y_pred = X @ beta
        residuals = y - y_pred
        sse = np.sum(residuals ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)

        residual_var = sse / max(1, n_samples - n_params)
        r_sq = 1.0 - sse / (sst + 1e-15) if sst > 1e-15 else 0.0
        adj_r_sq = 1.0 - (1 - r_sq) * (n_samples - 1) / max(1, n_samples - n_params - 1)

        # Standard errors (approximate for Ridge)
        try:
            inv_term = np.linalg.inv(XtX + reg_matrix)
            sandwich = inv_term @ XtX @ inv_term
            se = np.sqrt(np.diag(sandwich) * residual_var)
        except np.linalg.LinAlgError:
            se = np.full(n_params, np.inf)

        se = np.maximum(se, 1e-15)
        t_stats = beta / se
        df = max(1, n_samples - n_params)
        p_vals = np.array([
            2 * (1 - sp_stats.t.cdf(abs(t), df)) for t in t_stats
        ])

        return EstimationResult(
            target=target,
            parents=parents,
            coefficients=beta,
            standard_errors=se,
            t_statistics=t_stats,
            p_values=p_vals,
            residual_variance=float(residual_var),
            r_squared=float(r_sq),
            adj_r_squared=float(adj_r_sq),
            residuals=residuals,
            method="ridge",
        )

    # ----- Lasso regression -----

    def _estimate_lasso(
        self, target: int, parents: List[int], data: np.ndarray
    ) -> EstimationResult:
        """Lasso (L1 regularized) estimation via coordinate descent.

        Parameters
        ----------
        target : int
        parents : list of int
        data : np.ndarray

        Returns
        -------
        EstimationResult
        """
        if len(parents) == 0:
            result = self._estimate_ols(target, parents, data)
            result.method = "lasso"
            return result

        n_samples = data.shape[0]
        y = data[:, target]
        X_raw = data[:, parents]

        # Standardize features
        X_mean = np.mean(X_raw, axis=0)
        X_std = np.std(X_raw, axis=0)
        X_std[X_std < 1e-10] = 1.0
        X_scaled = (X_raw - X_mean) / X_std
        y_mean = np.mean(y)
        y_centered = y - y_mean

        n_features = len(parents)
        lam = self.regularization_strength
        beta = np.zeros(n_features)

        # Coordinate descent
        max_iter = 1000
        tol = 1e-6
        for iteration in range(max_iter):
            beta_old = beta.copy()

            for k in range(n_features):
                # Partial residual
                r_k = y_centered - X_scaled @ beta + X_scaled[:, k] * beta[k]
                rho = X_scaled[:, k] @ r_k / n_samples
                norm_sq = np.sum(X_scaled[:, k] ** 2) / n_samples

                # Soft thresholding
                if rho > lam:
                    beta[k] = (rho - lam) / norm_sq
                elif rho < -lam:
                    beta[k] = (rho + lam) / norm_sq
                else:
                    beta[k] = 0.0

            if np.max(np.abs(beta - beta_old)) < tol:
                break

        # Un-standardize
        beta_orig = beta / X_std
        intercept = y_mean - X_mean @ beta_orig

        coefficients = np.concatenate([[intercept], beta_orig])

        # Compute residuals and statistics
        X_aug = np.column_stack([np.ones(n_samples), X_raw])
        y_pred = X_aug @ coefficients
        residuals = y - y_pred
        sse = np.sum(residuals ** 2)
        sst = np.sum((y - y_mean) ** 2)

        n_params = len(coefficients)
        residual_var = sse / max(1, n_samples - n_params)
        r_sq = 1.0 - sse / (sst + 1e-15) if sst > 1e-15 else 0.0
        adj_r_sq = 1.0 - (1 - r_sq) * (n_samples - 1) / max(1, n_samples - n_params - 1)

        # Approximate SEs (from OLS on selected variables)
        active = np.abs(coefficients[1:]) > 1e-10
        se = np.full(n_params, np.inf)
        if np.any(active):
            try:
                XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
                se = np.sqrt(np.diag(XtX_inv) * residual_var)
            except np.linalg.LinAlgError:
                pass

        se = np.maximum(se, 1e-15)
        t_stats = coefficients / se
        df = max(1, n_samples - n_params)
        p_vals = np.array([
            2 * (1 - sp_stats.t.cdf(abs(t), df)) for t in t_stats
        ])

        return EstimationResult(
            target=target,
            parents=parents,
            coefficients=coefficients,
            standard_errors=se,
            t_statistics=t_stats,
            p_values=p_vals,
            residual_variance=float(residual_var),
            r_squared=float(r_sq),
            adj_r_squared=float(adj_r_sq),
            residuals=residuals,
            method="lasso",
        )

    # ----- Bayesian estimation -----

    def _estimate_bayesian(
        self, target: int, parents: List[int], data: np.ndarray
    ) -> EstimationResult:
        """Bayesian parameter estimation with conjugate Normal-Inverse-Gamma prior.

        Uses a non-informative prior (large variance) for a closed-form
        posterior.

        Parameters
        ----------
        target : int
        parents : list of int
        data : np.ndarray

        Returns
        -------
        EstimationResult
        """
        if len(parents) == 0:
            result = self._estimate_ols(target, parents, data)
            result.method = "bayesian"
            return result

        n_samples = data.shape[0]
        y = data[:, target]
        X = np.column_stack([np.ones(n_samples), data[:, parents]])
        n_params = X.shape[1]

        # Prior: beta ~ N(0, tau^2 * I), sigma^2 ~ InverseGamma(a0, b0)
        tau_sq = 100.0  # Diffuse prior
        a0 = 1.0  # Prior shape
        b0 = 1.0  # Prior scale

        # Posterior for beta | sigma^2, y
        prior_precision = np.eye(n_params) / tau_sq
        XtX = X.T @ X
        posterior_precision = XtX + prior_precision

        try:
            posterior_cov = np.linalg.inv(posterior_precision)
        except np.linalg.LinAlgError:
            posterior_cov = np.eye(n_params) * tau_sq

        Xty = X.T @ y
        posterior_mean = posterior_cov @ Xty

        # Posterior for sigma^2
        residuals = y - X @ posterior_mean
        sse = np.sum(residuals ** 2)
        a_n = a0 + n_samples / 2.0
        b_n = b0 + sse / 2.0
        posterior_var = b_n / (a_n - 1) if a_n > 1 else sse / n_samples

        # Standard errors from posterior
        se = np.sqrt(np.diag(posterior_cov) * float(posterior_var))
        se = np.maximum(se, 1e-15)

        sst = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1.0 - sse / (sst + 1e-15) if sst > 1e-15 else 0.0
        adj_r_sq = 1.0 - (1 - r_sq) * (n_samples - 1) / max(1, n_samples - n_params - 1)

        t_stats = posterior_mean / se
        df = max(1, n_samples - n_params)
        p_vals = np.array([
            2 * (1 - sp_stats.t.cdf(abs(t), df)) for t in t_stats
        ])

        return EstimationResult(
            target=target,
            parents=parents,
            coefficients=posterior_mean,
            standard_errors=se,
            t_statistics=t_stats,
            p_values=p_vals,
            residual_variance=float(posterior_var),
            r_squared=float(r_sq),
            adj_r_squared=float(adj_r_sq),
            residuals=residuals,
            method="bayesian",
        )

    # ----- Residual diagnostics -----

    def residual_diagnostics(
        self, result: EstimationResult, data: np.ndarray
    ) -> Dict[str, Any]:
        """Compute residual diagnostic statistics.

        Parameters
        ----------
        result : EstimationResult
        data : np.ndarray

        Returns
        -------
        dict
            Diagnostic statistics including normality tests and
            heteroscedasticity indicators.
        """
        residuals = result.residuals
        n = len(residuals)

        diagnostics: Dict[str, Any] = {}

        # Basic statistics
        diagnostics["mean"] = float(np.mean(residuals))
        diagnostics["std"] = float(np.std(residuals, ddof=1))
        diagnostics["skewness"] = float(sp_stats.skew(residuals))
        diagnostics["kurtosis"] = float(sp_stats.kurtosis(residuals))

        # Normality test (Shapiro-Wilk for small n, D'Agostino-Pearson for large)
        if n < 5000 and n >= 8:
            try:
                stat, pval = sp_stats.shapiro(residuals[:min(n, 5000)])
                diagnostics["shapiro_wilk"] = {"statistic": float(stat), "p_value": float(pval)}
            except Exception:
                diagnostics["shapiro_wilk"] = None
        else:
            diagnostics["shapiro_wilk"] = None

        if n >= 20:
            try:
                stat, pval = sp_stats.normaltest(residuals)
                diagnostics["dagostino_pearson"] = {
                    "statistic": float(stat), "p_value": float(pval)
                }
            except Exception:
                diagnostics["dagostino_pearson"] = None
        else:
            diagnostics["dagostino_pearson"] = None

        # Durbin-Watson-like statistic (autocorrelation of residuals)
        if n > 1:
            diff = np.diff(residuals)
            dw = float(np.sum(diff ** 2) / (np.sum(residuals ** 2) + 1e-15))
            diagnostics["durbin_watson"] = dw

        # Heteroscedasticity: correlation of |residuals| with fitted values
        if len(result.parents) > 0:
            X = np.column_stack([np.ones(n), data[:, result.parents]])
            fitted = X @ result.coefficients
            abs_res = np.abs(residuals)
            corr = np.corrcoef(fitted, abs_res)[0, 1]
            diagnostics["heteroscedasticity_correlation"] = float(corr)

        # Leverage and Cook's distance
        if len(result.parents) > 0:
            X = np.column_stack([np.ones(n), data[:, result.parents]])
            try:
                hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
                leverage = np.diag(hat_matrix)
                diagnostics["mean_leverage"] = float(np.mean(leverage))
                diagnostics["max_leverage"] = float(np.max(leverage))

                # Cook's distance
                mse = result.residual_variance
                p = len(result.coefficients)
                if mse > 1e-15 and p > 0:
                    cook_d = (residuals ** 2 * leverage) / (
                        p * mse * (1 - leverage + 1e-15) ** 2
                    )
                    diagnostics["max_cook_distance"] = float(np.max(cook_d))
                    diagnostics["n_influential"] = int(np.sum(cook_d > 4.0 / n))
            except np.linalg.LinAlgError:
                pass

        return diagnostics

    # ----- Convenience methods -----

    def estimate_weight_matrix(
        self,
        adj_matrix: np.ndarray,
        data: np.ndarray,
    ) -> np.ndarray:
        """Estimate the full weight matrix from data and DAG structure.

        Parameters
        ----------
        adj_matrix : np.ndarray
            Binary DAG adjacency matrix.
        data : np.ndarray

        Returns
        -------
        np.ndarray
            Weighted adjacency matrix where entry [i,j] is the
            coefficient of variable i in the structural equation for j.
        """
        n_vars = adj_matrix.shape[0]
        weights = np.zeros((n_vars, n_vars))

        results = self.estimate_all(adj_matrix, data)
        for result in results:
            for k, parent in enumerate(result.parents):
                weights[parent, result.target] = result.parent_coefficients[k]

        return weights

    def estimate_noise_variances(
        self,
        adj_matrix: np.ndarray,
        data: np.ndarray,
    ) -> np.ndarray:
        """Estimate noise variances for all structural equations.

        Parameters
        ----------
        adj_matrix : np.ndarray
        data : np.ndarray

        Returns
        -------
        np.ndarray
            Shape (n_vars,) noise variance estimates.
        """
        results = self.estimate_all(adj_matrix, data)
        return np.array([r.residual_variance for r in results])

    def compute_sufficient_stats(self, data: np.ndarray) -> SufficientStatistics:
        """Compute and return sufficient statistics.

        Parameters
        ----------
        data : np.ndarray

        Returns
        -------
        SufficientStatistics
        """
        return SufficientStatistics.from_data(data)

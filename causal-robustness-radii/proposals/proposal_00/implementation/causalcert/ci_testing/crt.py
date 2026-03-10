"""
Conditional randomization test (CRT).

Model-X knockoff-style CI test that permutes X conditional on Z to construct
a reference distribution for the test statistic, with no parametric
assumptions on the X–Y relationship.

Implements the CRT of Candès, Fan, Janson & Lv (2018) with:
- Gaussian conditional sampling (when Z is approximately Gaussian)
- Permutation-based conditional sampling as fallback
- Double-dipping correction
- Configurable number of randomisations
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from causalcert.ci_testing.base import (
    BaseCITest,
    _extract_columns,
    _insufficient_sample_result,
    _validate_inputs,
)
from causalcert.types import CITestMethod, CITestResult, NodeId, NodeSet

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Conditional sampling strategies
# ---------------------------------------------------------------------------


def _gaussian_conditional_sample(
    x_col: np.ndarray,
    z_cols: np.ndarray,
    rng: np.random.Generator,
    regularization: float = 1e-4,
) -> np.ndarray:
    """Sample X | Z under the Gaussian model.

    Fits ``X | Z ~ N(Z @ beta, sigma^2)`` where beta and sigma are
    estimated from the data, then draws a new X from this distribution.

    Parameters
    ----------
    x_col : np.ndarray
        Original X values ``(n,)``.
    z_cols : np.ndarray
        Conditioning variables ``(n, k)``.
    rng : np.random.Generator
        Random number generator.
    regularization : float
        Ridge penalty for the regression.

    Returns
    -------
    np.ndarray
        Resampled X column ``(n,)``.
    """
    n, k = z_cols.shape

    # Design matrix with intercept
    Z = np.column_stack([np.ones(n), z_cols])
    ZtZ = Z.T @ Z + regularization * np.eye(Z.shape[1])

    try:
        beta = np.linalg.solve(ZtZ, Z.T @ x_col)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(Z, x_col, rcond=None)[0]

    residuals = x_col - Z @ beta
    sigma = max(float(np.std(residuals, ddof=k + 1)), _EPS)

    # Sample from the conditional distribution
    x_mean = Z @ beta
    x_sampled = x_mean + rng.normal(0.0, sigma, size=n)

    return x_sampled


def _permutation_conditional_sample(
    x_col: np.ndarray,
    z_cols: np.ndarray,
    rng: np.random.Generator,
    n_neighbors: int = 5,
) -> np.ndarray:
    """Sample X | Z via local permutation (non-parametric fallback).

    Groups observations by Z-proximity and permutes X within groups.
    Uses k-nearest-neighbours to define local neighbourhoods.

    Parameters
    ----------
    x_col : np.ndarray
        Original X values ``(n,)``.
    z_cols : np.ndarray
        Conditioning variables ``(n, k)``.
    rng : np.random.Generator
        Random number generator.
    n_neighbors : int
        Number of neighbours for local grouping.

    Returns
    -------
    np.ndarray
        Resampled X column ``(n,)``.
    """
    n = len(x_col)
    if n <= n_neighbors:
        return x_col[rng.permutation(n)]

    # Standardise Z for distance computation
    z_std = z_cols.copy()
    for j in range(z_cols.shape[1]):
        s = np.std(z_cols[:, j])
        if s > _EPS:
            z_std[:, j] = (z_cols[:, j] - np.mean(z_cols[:, j])) / s

    # Compute pairwise distances to Z (using squared Euclidean)
    x_sampled = x_col.copy()
    indices = np.arange(n)
    rng.shuffle(indices)

    # For each observation, find its k nearest neighbours and swap X with
    # a randomly chosen neighbour
    for i in indices:
        dists = np.sum((z_std - z_std[i]) ** 2, axis=1)
        # k nearest (including self)
        nn = np.argsort(dists)[: n_neighbors + 1]
        # Swap with a random neighbour
        j = rng.choice(nn)
        x_sampled[i], x_sampled[j] = x_sampled[j], x_sampled[i]

    return x_sampled


def _knockoff_conditional_sample(
    x_col: np.ndarray,
    z_cols: np.ndarray,
    rng: np.random.Generator,
    regularization: float = 1e-4,
) -> np.ndarray:
    """Sample X | Z using the Model-X knockoff construction.

    Constructs a knockoff copy of X that preserves the conditional
    distribution X | Z under the Gaussian model, following Candès et al.
    (2018).

    Parameters
    ----------
    x_col : np.ndarray
        Original X values ``(n,)``.
    z_cols : np.ndarray
        Conditioning variables ``(n, k)``.
    rng : np.random.Generator
        Random number generator.
    regularization : float
        Ridge penalty.

    Returns
    -------
    np.ndarray
        Knockoff copy of X ``(n,)``.
    """
    n, k = z_cols.shape

    # Joint covariance of (X, Z)
    joint = np.column_stack([x_col, z_cols])
    cov_joint = np.cov(joint, rowvar=False)
    cov_joint += regularization * np.eye(cov_joint.shape[0])

    # Extract sub-matrices
    sigma_xx = cov_joint[0, 0]
    sigma_xz = cov_joint[0, 1:]  # (k,)
    sigma_zz = cov_joint[1:, 1:]  # (k, k)

    try:
        sigma_zz_inv = np.linalg.inv(sigma_zz)
    except np.linalg.LinAlgError:
        sigma_zz_inv = np.linalg.pinv(sigma_zz)

    # Conditional mean and variance of X | Z
    beta = sigma_zz_inv @ sigma_xz
    cond_var = sigma_xx - sigma_xz @ sigma_zz_inv @ sigma_xz
    cond_var = max(float(cond_var), _EPS)

    # Knockoff: X_tilde = mu + noise, where mu = mean + beta^T @ (Z - mean_Z)
    mean_x = np.mean(x_col)
    mean_z = np.mean(z_cols, axis=0)

    mu = mean_x + (z_cols - mean_z) @ beta
    # Knockoff uses 2 * cond_var as the noise variance
    s = min(2.0 * cond_var, sigma_xx)
    x_knockoff = mu + rng.normal(0, np.sqrt(s), size=n)

    return x_knockoff


# ---------------------------------------------------------------------------
# Test statistics
# ---------------------------------------------------------------------------


def _regression_test_statistic(
    x_col: np.ndarray,
    y_col: np.ndarray,
    z_cols: np.ndarray | None,
) -> float:
    """Compute the CRT test statistic via partial R².

    Uses the increase in R² from adding X to a regression of Y on Z.

    Parameters
    ----------
    x_col, y_col : np.ndarray
        Variable columns.
    z_cols : np.ndarray | None
        Conditioning matrix.

    Returns
    -------
    float
        Test statistic (partial R²).
    """
    n = len(x_col)

    if z_cols is None or z_cols.shape[1] == 0:
        # Simple correlation
        r = np.corrcoef(x_col, y_col)[0, 1]
        if np.isnan(r):
            return 0.0
        return r * r

    # Reduced model: Y ~ Z
    Z_design = np.column_stack([np.ones(n), z_cols])
    try:
        beta_z, _, _, _ = np.linalg.lstsq(Z_design, y_col, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    res_reduced = y_col - Z_design @ beta_z
    ss_reduced = np.sum(res_reduced ** 2)

    # Full model: Y ~ Z + X
    XZ_design = np.column_stack([Z_design, x_col])
    try:
        beta_xz, _, _, _ = np.linalg.lstsq(XZ_design, y_col, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    res_full = y_col - XZ_design @ beta_xz
    ss_full = np.sum(res_full ** 2)

    if ss_reduced < _EPS:
        return 0.0

    # Partial R² = (SS_reduced - SS_full) / SS_reduced
    partial_r2 = max((ss_reduced - ss_full) / ss_reduced, 0.0)
    return float(partial_r2)


def _absolute_correlation_test_statistic(
    x_col: np.ndarray,
    y_col: np.ndarray,
    z_cols: np.ndarray | None,
) -> float:
    """Absolute partial correlation test statistic.

    Parameters
    ----------
    x_col, y_col : np.ndarray
        Variable columns.
    z_cols : np.ndarray | None
        Conditioning matrix.

    Returns
    -------
    float
        |r_{xy|z}|
    """
    if z_cols is None or z_cols.shape[1] == 0:
        r = np.corrcoef(x_col, y_col)[0, 1]
        return abs(float(r)) if not np.isnan(r) else 0.0

    n = len(x_col)
    Z_design = np.column_stack([np.ones(n), z_cols])

    try:
        cx, _, _, _ = np.linalg.lstsq(Z_design, x_col, rcond=None)
        cy, _, _, _ = np.linalg.lstsq(Z_design, y_col, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0

    res_x = x_col - Z_design @ cx
    res_y = y_col - Z_design @ cy

    denom = np.sqrt(np.sum(res_x ** 2) * np.sum(res_y ** 2))
    if denom < _EPS:
        return 0.0
    return abs(float(np.sum(res_x * res_y) / denom))


def _distance_correlation_statistic(
    x_col: np.ndarray,
    y_col: np.ndarray,
    z_cols: np.ndarray | None,
) -> float:
    """Distance correlation test statistic (simplified).

    Parameters
    ----------
    x_col, y_col : np.ndarray
        Variable columns.
    z_cols : np.ndarray | None
        Conditioning matrix.

    Returns
    -------
    float
        Distance correlation.
    """
    # Remove conditioning effect via residuals
    if z_cols is not None and z_cols.shape[1] > 0:
        n = len(x_col)
        Z_design = np.column_stack([np.ones(n), z_cols])
        try:
            cx, _, _, _ = np.linalg.lstsq(Z_design, x_col, rcond=None)
            cy, _, _, _ = np.linalg.lstsq(Z_design, y_col, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0
        x_col = x_col - Z_design @ cx
        y_col = y_col - Z_design @ cy

    n = len(x_col)
    if n < 4:
        return 0.0

    # Distance matrices
    a = np.abs(x_col[:, None] - x_col[None, :])
    b = np.abs(y_col[:, None] - y_col[None, :])

    # Double centering
    a_row = a.mean(axis=1, keepdims=True)
    a_col = a.mean(axis=0, keepdims=True)
    a_mean = a.mean()
    A = a - a_row - a_col + a_mean

    b_row = b.mean(axis=1, keepdims=True)
    b_col = b.mean(axis=0, keepdims=True)
    b_mean = b.mean()
    B = b - b_row - b_col + b_mean

    dcov2 = np.mean(A * B)
    dvar_x = np.mean(A * A)
    dvar_y = np.mean(B * B)

    if dvar_x < _EPS or dvar_y < _EPS:
        return 0.0

    dcor = np.sqrt(max(dcov2, 0.0)) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y))
    return float(dcor)


# ---------------------------------------------------------------------------
# Double-dipping correction
# ---------------------------------------------------------------------------


def _double_dip_correction(p_value: float, n_permutations: int) -> float:
    """Apply a conservative correction for double-dipping.

    When the same data is used to both fit the conditional model and
    compute the test statistic, we apply a mild correction.

    Parameters
    ----------
    p_value : float
        Raw p-value.
    n_permutations : int
        Number of permutations used.

    Returns
    -------
    float
        Adjusted p-value.
    """
    # Slight upward adjustment (conservative)
    correction = 1.0 / np.sqrt(n_permutations)
    adjusted = p_value + correction * (1.0 - p_value)
    return float(np.clip(adjusted, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main CRT class
# ---------------------------------------------------------------------------


class ConditionalRandomizationTest(BaseCITest):
    """Conditional randomization test (CRT).

    Implements the CRT of Candès, Fan, Janson & Lv (2018).  The test
    generates conditional samples of X given Z under a fitted model,
    computes the test statistic on each sample, and reports the proportion
    of permuted statistics exceeding the observed one.

    Parameters
    ----------
    alpha : float
        Significance level.
    n_permutations : int
        Number of conditional permutations.
    regression_model : str
        Model used to regress X on Z for conditional sampling
        (``"gaussian"`` = Gaussian conditional, ``"knockoff"`` = Model-X
        knockoff, ``"permutation"`` = local permutation fallback).
    test_stat : str
        Which test statistic to use (``"partial_r2"``, ``"abs_corr"``,
        ``"dcor"``).
    apply_double_dip_correction : bool
        Whether to apply the double-dipping correction.
    seed : int
        Random seed.
    """

    method = CITestMethod.CRT

    def __init__(
        self,
        alpha: float = 0.05,
        n_permutations: int = 500,
        regression_model: str = "gaussian",
        test_stat: str = "partial_r2",
        apply_double_dip_correction: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(alpha=alpha, seed=seed)
        self.n_permutations = n_permutations
        self.regression_model = regression_model
        self.test_stat = test_stat
        self.apply_double_dip_correction = apply_double_dip_correction

    def test(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: pd.DataFrame,
    ) -> CITestResult:
        """Run the conditional randomization test.

        Parameters
        ----------
        x, y : NodeId
            Variables to test.
        conditioning_set : NodeSet
            Conditioning variables.
        data : pd.DataFrame
            Observational data.

        Returns
        -------
        CITestResult
        """
        _validate_inputs(data, x, y, conditioning_set)
        x_col, y_col, z_cols = _extract_columns(data, x, y, conditioning_set)
        n = len(x_col)

        if n < 20:
            return _insufficient_sample_result(
                x, y, conditioning_set, self.method, self.alpha
            )

        rng = np.random.default_rng(self.seed)

        # Observed test statistic
        T_obs = self._test_statistic(x_col, y_col, z_cols)

        if z_cols is None or z_cols.shape[1] == 0:
            # No conditioning: simple permutation test
            null_stats = np.empty(self.n_permutations)
            for b in range(self.n_permutations):
                x_perm = x_col[rng.permutation(n)]
                null_stats[b] = self._test_statistic(x_perm, y_col, z_cols)
        else:
            # Conditional permutation
            null_stats = np.empty(self.n_permutations)
            for b in range(self.n_permutations):
                x_sampled = self._conditional_sample(x_col, z_cols, rng)
                null_stats[b] = self._test_statistic(x_sampled, y_col, z_cols)

        # p-value: (1 + #{T_null >= T_obs}) / (1 + B)
        # The +1 in numerator and denominator provides valid p-values
        count_ge = np.sum(null_stats >= T_obs)
        p_value = float((1 + count_ge) / (1 + self.n_permutations))

        if self.apply_double_dip_correction and z_cols is not None:
            p_value = _double_dip_correction(p_value, self.n_permutations)

        return self._make_result(x, y, conditioning_set, T_obs, p_value)

    def _conditional_sample(
        self,
        x_col: np.ndarray,
        z_cols: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate a conditional sample of X given Z.

        Dispatches to the configured sampling strategy.

        Parameters
        ----------
        x_col : np.ndarray
            Original X column.
        z_cols : np.ndarray
            Conditioning variables.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Resampled X column.
        """
        if self.regression_model == "gaussian":
            try:
                return _gaussian_conditional_sample(x_col, z_cols, rng)
            except Exception:
                return _permutation_conditional_sample(x_col, z_cols, rng)
        elif self.regression_model == "knockoff":
            try:
                return _knockoff_conditional_sample(x_col, z_cols, rng)
            except Exception:
                return _gaussian_conditional_sample(x_col, z_cols, rng)
        elif self.regression_model == "permutation":
            return _permutation_conditional_sample(x_col, z_cols, rng)
        else:
            # Default fallback
            return _gaussian_conditional_sample(x_col, z_cols, rng)

    def _test_statistic(
        self,
        x_col: np.ndarray,
        y_col: np.ndarray,
        z_cols: np.ndarray | None,
    ) -> float:
        """Compute the test statistic for the CRT.

        Dispatches to the configured test statistic.

        Parameters
        ----------
        x_col, y_col : np.ndarray
            Variable columns.
        z_cols : np.ndarray | None
            Conditioning matrix.

        Returns
        -------
        float
            Test statistic value.
        """
        if self.test_stat == "partial_r2":
            return _regression_test_statistic(x_col, y_col, z_cols)
        elif self.test_stat == "abs_corr":
            return _absolute_correlation_test_statistic(x_col, y_col, z_cols)
        elif self.test_stat == "dcor":
            return _distance_correlation_statistic(x_col, y_col, z_cols)
        else:
            return _regression_test_statistic(x_col, y_col, z_cols)

    def __repr__(self) -> str:
        return (
            f"ConditionalRandomizationTest(n_permutations={self.n_permutations}, "
            f"model={self.regression_model!r}, stat={self.test_stat!r}, "
            f"alpha={self.alpha})"
        )

"""
Partial-correlation CI test (Fisher z-transform).

The simplest and fastest CI test, valid under joint Gaussianity.  Uses the
Fisher z-transform to convert a partial correlation to a z-statistic with
known asymptotic distribution.

Also provides a semi-partial (part) correlation variant and regularised
partial correlation for high-dimensional conditioning sets.
"""

from __future__ import annotations

import warnings

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

_EPS = 1e-12  # guard against exact 0 / 1 correlations


class PartialCorrelationTest(BaseCITest):
    """Fisher-z partial-correlation CI test.

    Parameters
    ----------
    alpha : float
        Significance level.
    regularization : float
        Ridge penalty added to the diagonal of the conditioning covariance
        before inversion.  Helps when ``|S| ≈ n``.
    seed : int
        Random seed (unused, included for interface consistency).
    """

    method = CITestMethod.PARTIAL_CORRELATION

    def __init__(
        self,
        alpha: float = 0.05,
        regularization: float = 0.0,
        seed: int = 42,
    ) -> None:
        super().__init__(alpha=alpha, seed=seed)
        self.regularization = regularization

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def test(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: pd.DataFrame,
    ) -> CITestResult:
        """Run the Fisher-z partial correlation test.

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
        k = len(conditioning_set)

        if n < k + 3:
            return _insufficient_sample_result(
                x, y, conditioning_set, self.method, self.alpha
            )

        r = self._partial_correlation(x_col, y_col, z_cols,
                                       regularization=self.regularization)
        z_stat, p_val = self._fisher_z(r, n, k)
        return self._make_result(x, y, conditioning_set, z_stat, p_val)

    def test_semi_partial(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: pd.DataFrame,
    ) -> CITestResult:
        """Semi-partial (part) correlation test.

        Removes the linear effect of the conditioning set from *x* only,
        then correlates the residual with *y*.

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
        k = len(conditioning_set)

        if n < k + 3:
            return _insufficient_sample_result(
                x, y, conditioning_set, self.method, self.alpha
            )

        r = self._semi_partial_correlation(x_col, y_col, z_cols,
                                           regularization=self.regularization)
        z_stat, p_val = self._fisher_z(r, n, k)
        return self._make_result(x, y, conditioning_set, z_stat, p_val)

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    @staticmethod
    def _partial_correlation(
        x_col: np.ndarray,
        y_col: np.ndarray,
        z_cols: np.ndarray | None,
        *,
        regularization: float = 0.0,
    ) -> float:
        """Compute the partial correlation of x and y given z.

        Uses the recursive formula via the inverse of the covariance matrix
        of ``(x, y, z)``.  When the conditioning set is empty this reduces
        to the Pearson correlation.

        Parameters
        ----------
        x_col, y_col : np.ndarray
            Column vectors (1-D, length *n*).
        z_cols : np.ndarray | None
            Conditioning matrix ``(n, k)`` or ``None``.
        regularization : float
            Ridge penalty for the conditioning covariance.

        Returns
        -------
        float
            Partial correlation coefficient in ``[-1, 1]``.
        """
        if z_cols is None or z_cols.shape[1] == 0:
            # Marginal Pearson correlation
            r = np.corrcoef(x_col, y_col)[0, 1]
            return float(np.clip(r, -1.0 + _EPS, 1.0 - _EPS))

        # Build the full (2 + k) × (2 + k) covariance matrix
        all_vars = np.column_stack([x_col, y_col, z_cols])
        cov = np.cov(all_vars, rowvar=False)
        p = cov.shape[0]

        # Add regularisation to improve conditioning
        if regularization > 0:
            cov += regularization * np.eye(p)

        # Precision matrix method
        try:
            prec = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            prec = np.linalg.pinv(cov)

        # Partial correlation from precision matrix:
        #   rho_{xy|z} = -P[0,1] / sqrt(P[0,0] * P[1,1])
        denom = np.sqrt(np.abs(prec[0, 0] * prec[1, 1]))
        if denom < _EPS:
            return 0.0
        r = -prec[0, 1] / denom
        return float(np.clip(r, -1.0 + _EPS, 1.0 - _EPS))

    @staticmethod
    def _partial_correlation_via_residuals(
        x_col: np.ndarray,
        y_col: np.ndarray,
        z_cols: np.ndarray,
        *,
        regularization: float = 0.0,
    ) -> float:
        """Compute partial correlation via OLS residuals.

        Regress *x* and *y* each on *z*, then correlate the residuals.
        This is numerically more stable when ``k`` is large relative to
        ``n``.

        Parameters
        ----------
        x_col, y_col : np.ndarray
            Column vectors.
        z_cols : np.ndarray
            Conditioning matrix ``(n, k)``.
        regularization : float
            Ridge penalty for the OLS regression.

        Returns
        -------
        float
            Partial correlation in ``[-1, 1]``.
        """
        n, k = z_cols.shape
        # Design matrix with intercept
        Z = np.column_stack([np.ones(n), z_cols])
        ZtZ = Z.T @ Z
        if regularization > 0:
            ZtZ += regularization * np.eye(ZtZ.shape[0])

        try:
            ZtZ_inv = np.linalg.inv(ZtZ)
        except np.linalg.LinAlgError:
            ZtZ_inv = np.linalg.pinv(ZtZ)

        hat = Z @ ZtZ_inv @ Z.T
        res_x = x_col - hat @ x_col
        res_y = y_col - hat @ y_col

        denom = np.sqrt(np.sum(res_x ** 2) * np.sum(res_y ** 2))
        if denom < _EPS:
            return 0.0
        r = float(np.sum(res_x * res_y) / denom)
        return float(np.clip(r, -1.0 + _EPS, 1.0 - _EPS))

    @staticmethod
    def _semi_partial_correlation(
        x_col: np.ndarray,
        y_col: np.ndarray,
        z_cols: np.ndarray | None,
        *,
        regularization: float = 0.0,
    ) -> float:
        """Compute the semi-partial correlation (remove z from x only).

        Parameters
        ----------
        x_col, y_col : np.ndarray
            Column vectors.
        z_cols : np.ndarray | None
            Conditioning matrix.
        regularization : float
            Ridge penalty.

        Returns
        -------
        float
            Semi-partial correlation in ``[-1, 1]``.
        """
        if z_cols is None or z_cols.shape[1] == 0:
            r = np.corrcoef(x_col, y_col)[0, 1]
            return float(np.clip(r, -1.0 + _EPS, 1.0 - _EPS))

        n, k = z_cols.shape
        Z = np.column_stack([np.ones(n), z_cols])
        ZtZ = Z.T @ Z
        if regularization > 0:
            ZtZ += regularization * np.eye(ZtZ.shape[0])

        try:
            ZtZ_inv = np.linalg.inv(ZtZ)
        except np.linalg.LinAlgError:
            ZtZ_inv = np.linalg.pinv(ZtZ)

        hat = Z @ ZtZ_inv @ Z.T
        res_x = x_col - hat @ x_col

        # Correlate residual of x with raw y
        denom = np.sqrt(np.sum(res_x ** 2) * np.sum(y_col ** 2))
        if denom < _EPS:
            return 0.0
        r = float(np.sum(res_x * y_col) / denom)
        return float(np.clip(r, -1.0 + _EPS, 1.0 - _EPS))

    @staticmethod
    def _fisher_z(r: float, n: int, k: int) -> tuple[float, float]:
        """Apply Fisher's z-transform to a partial correlation.

        Under the null hypothesis of conditional independence (ρ = 0), the
        transformed statistic ``z = sqrt(n - k - 3) * 0.5 * log((1+r)/(1-r))``
        is approximately standard normal.

        Parameters
        ----------
        r : float
            Partial correlation.
        n : int
            Sample size.
        k : int
            Size of conditioning set.

        Returns
        -------
        tuple[float, float]
            ``(z_statistic, p_value)``.
        """
        dof = n - k - 3
        if dof < 1:
            return 0.0, 1.0

        # Clamp to avoid log(0)
        r = float(np.clip(r, -1.0 + _EPS, 1.0 - _EPS))

        # Fisher z-transform
        z = 0.5 * np.log((1.0 + r) / (1.0 - r))
        # Under H0, sqrt(n - k - 3) * z ~ N(0, 1)
        z_stat = np.sqrt(dof) * z

        # Two-sided p-value
        p_value = 2.0 * stats.norm.sf(np.abs(z_stat))
        return float(z_stat), float(p_value)


class RegularizedPartialCorrelationTest(PartialCorrelationTest):
    """Partial correlation test with automatic ridge regularisation.

    When the conditioning set is large relative to *n*, adds a data-driven
    ridge penalty ``λ = k / n`` to the covariance matrix.

    Parameters
    ----------
    alpha : float
        Significance level.
    seed : int
        Random seed.
    """

    def __init__(self, alpha: float = 0.05, seed: int = 42) -> None:
        super().__init__(alpha=alpha, regularization=0.0, seed=seed)

    def test(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: pd.DataFrame,
    ) -> CITestResult:
        _validate_inputs(data, x, y, conditioning_set)
        x_col, y_col, z_cols = _extract_columns(data, x, y, conditioning_set)
        n = len(x_col)
        k = len(conditioning_set)

        if n < k + 3:
            return _insufficient_sample_result(
                x, y, conditioning_set, self.method, self.alpha
            )

        # Automatic regularisation
        auto_reg = k / max(n, 1) if k > 0 else 0.0
        # Choose method based on k / n ratio
        if z_cols is not None and k > n * 0.5:
            r = self._partial_correlation_via_residuals(
                x_col, y_col, z_cols, regularization=auto_reg
            )
        else:
            r = self._partial_correlation(
                x_col, y_col, z_cols, regularization=auto_reg
            )

        z_stat, p_val = self._fisher_z(r, n, k)
        return self._make_result(x, y, conditioning_set, z_stat, p_val)

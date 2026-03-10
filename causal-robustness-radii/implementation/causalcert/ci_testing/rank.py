"""
Rank-based conditional-independence tests.

Non-parametric CI tests using Spearman and Kendall rank correlations,
suitable for monotone non-linear relationships.  Includes a copula-based
rank CI test that handles ties correctly.
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

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------


def _rank_with_ties(x: np.ndarray) -> np.ndarray:
    """Return average-rank transformed values, handling ties correctly.

    Parameters
    ----------
    x : np.ndarray
        1-D array of observations.

    Returns
    -------
    np.ndarray
        Ranks in ``[1, n]`` with tied values receiving the average rank.
    """
    temp = x.argsort().argsort().astype(np.float64) + 1.0
    # Use scipy for proper tie handling
    return stats.rankdata(x, method="average")


def _rank_matrix(M: np.ndarray) -> np.ndarray:
    """Rank-transform each column of a matrix independently.

    Parameters
    ----------
    M : np.ndarray
        2-D array ``(n, p)``.

    Returns
    -------
    np.ndarray
        Rank-transformed array of same shape.
    """
    if M.ndim == 1:
        return _rank_with_ties(M)
    out = np.empty_like(M, dtype=np.float64)
    for j in range(M.shape[1]):
        out[:, j] = _rank_with_ties(M[:, j])
    return out


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two 1-D arrays.

    Uses pre-ranked data (assumes *x* and *y* are already ranks).

    Parameters
    ----------
    x, y : np.ndarray
        Rank-transformed 1-D arrays.

    Returns
    -------
    float
        Spearman correlation in ``[-1, 1]``.
    """
    r = np.corrcoef(x, y)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(np.clip(r, -1.0 + _EPS, 1.0 - _EPS))


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b correlation handling ties.

    Parameters
    ----------
    x, y : np.ndarray
        1-D arrays (raw or ranked).

    Returns
    -------
    float
        Kendall tau-b in ``[-1, 1]``.
    """
    tau, _ = stats.kendalltau(x, y, method="auto")
    if np.isnan(tau):
        return 0.0
    return float(np.clip(tau, -1.0 + _EPS, 1.0 - _EPS))


# ---------------------------------------------------------------------------
# Partial correlation on ranks
# ---------------------------------------------------------------------------


def _partial_corr_from_precision(cov: np.ndarray, i: int, j: int) -> float:
    """Extract partial correlation rho_{ij|rest} from a covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix ``(p, p)``.
    i, j : int
        Indices of the two variables.

    Returns
    -------
    float
        Partial correlation in ``[-1, 1]``.
    """
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)

    denom = np.sqrt(np.abs(prec[i, i] * prec[j, j]))
    if denom < _EPS:
        return 0.0
    r = -prec[i, j] / denom
    return float(np.clip(r, -1.0 + _EPS, 1.0 - _EPS))


def _partial_corr_via_residuals(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
) -> float:
    """Compute partial correlation of *x* and *y* given *Z* via OLS residuals.

    Parameters
    ----------
    x, y : np.ndarray
        1-D arrays.
    Z : np.ndarray
        Conditioning matrix ``(n, k)``.

    Returns
    -------
    float
    """
    n = len(x)
    design = np.column_stack([np.ones(n), Z])
    try:
        coefs_x, _, _, _ = np.linalg.lstsq(design, x, rcond=None)
        coefs_y, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0

    res_x = x - design @ coefs_x
    res_y = y - design @ coefs_y

    denom = np.sqrt(np.sum(res_x ** 2) * np.sum(res_y ** 2))
    if denom < _EPS:
        return 0.0
    return float(np.clip(np.sum(res_x * res_y) / denom, -1.0 + _EPS, 1.0 - _EPS))


# ---------------------------------------------------------------------------
# Spearman CI Test
# ---------------------------------------------------------------------------


class RankCITest(BaseCITest):
    """Spearman rank-based partial-correlation CI test.

    Computes partial Spearman correlations and converts to a z-statistic
    using Fisher's z-transform.

    Parameters
    ----------
    alpha : float
        Significance level.
    rank_method : str
        ``"spearman"`` or ``"kendall"``.
    seed : int
        Random seed.
    """

    method = CITestMethod.RANK

    def __init__(
        self,
        alpha: float = 0.05,
        rank_method: str = "spearman",
        seed: int = 42,
    ) -> None:
        super().__init__(alpha=alpha, seed=seed)
        if rank_method not in ("spearman", "kendall"):
            raise ValueError(f"rank_method must be 'spearman' or 'kendall', got {rank_method!r}")
        self.rank_method = rank_method

    def test(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: pd.DataFrame,
    ) -> CITestResult:
        """Run the rank-based CI test.

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

        if n < k + 4:
            return _insufficient_sample_result(
                x, y, conditioning_set, self.method, self.alpha
            )

        if self.rank_method == "spearman":
            r = self._spearman_partial_correlation(x_col, y_col, z_cols)
        else:
            r = self._kendall_partial_correlation(x_col, y_col, z_cols)

        z_stat, p_val = self._fisher_z_rank(r, n, k)
        return self._make_result(x, y, conditioning_set, z_stat, p_val)

    # ------------------------------------------------------------------
    # Spearman partial correlation
    # ------------------------------------------------------------------

    def _spearman_partial_correlation(
        self,
        x_col: np.ndarray,
        y_col: np.ndarray,
        z_cols: np.ndarray | None,
    ) -> float:
        """Compute the Spearman partial correlation between x and y given z.

        Rank-transforms all variables, then computes the partial Pearson
        correlation on the ranks (which equals the Spearman partial
        correlation).

        Parameters
        ----------
        x_col, y_col : np.ndarray
            Raw column vectors.
        z_cols : np.ndarray | None
            Raw conditioning matrix.

        Returns
        -------
        float
            Partial Spearman correlation coefficient.
        """
        rx = _rank_with_ties(x_col)
        ry = _rank_with_ties(y_col)

        if z_cols is None or z_cols.shape[1] == 0:
            return _spearman_corr(rx, ry)

        rz = _rank_matrix(z_cols)
        return _partial_corr_via_residuals(rx, ry, rz)

    # ------------------------------------------------------------------
    # Kendall partial correlation
    # ------------------------------------------------------------------

    def _kendall_partial_correlation(
        self,
        x_col: np.ndarray,
        y_col: np.ndarray,
        z_cols: np.ndarray | None,
    ) -> float:
        """Compute the Kendall partial tau between x and y given z.

        Uses the residual-based approach: regress x and y on z (rank-
        transformed), then compute Kendall tau on the residuals.

        Parameters
        ----------
        x_col, y_col : np.ndarray
            Raw column vectors.
        z_cols : np.ndarray | None
            Raw conditioning matrix.

        Returns
        -------
        float
            Partial Kendall tau.
        """
        if z_cols is None or z_cols.shape[1] == 0:
            return _kendall_tau(x_col, y_col)

        n = len(x_col)
        rz = _rank_matrix(z_cols)

        # Regress ranks of x and y on ranks of z
        design = np.column_stack([np.ones(n), rz])
        try:
            coefs_x, _, _, _ = np.linalg.lstsq(design, x_col, rcond=None)
            coefs_y, _, _, _ = np.linalg.lstsq(design, y_col, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0

        res_x = x_col - design @ coefs_x
        res_y = y_col - design @ coefs_y

        return _kendall_tau(res_x, res_y)

    # ------------------------------------------------------------------
    # Asymptotic distribution
    # ------------------------------------------------------------------

    @staticmethod
    def _fisher_z_rank(r: float, n: int, k: int) -> tuple[float, float]:
        """Fisher z-transform for rank partial correlation.

        The asymptotic variance of the Spearman partial correlation is
        ``1 / (n - k - 3)`` under the null, same as Pearson on the ranks.

        Parameters
        ----------
        r : float
            Rank partial correlation.
        n : int
            Sample size.
        k : int
            Conditioning set size.

        Returns
        -------
        tuple[float, float]
            ``(z_statistic, two_sided_p_value)``.
        """
        dof = n - k - 3
        if dof < 1:
            return 0.0, 1.0

        r = float(np.clip(r, -1.0 + _EPS, 1.0 - _EPS))
        z = 0.5 * np.log((1.0 + r) / (1.0 - r))
        z_stat = np.sqrt(dof) * z
        p_value = 2.0 * stats.norm.sf(np.abs(z_stat))
        return float(z_stat), float(p_value)


# ---------------------------------------------------------------------------
# Copula-based rank CI test
# ---------------------------------------------------------------------------


class CopulaRankCITest(BaseCITest):
    """Rank CI test based on the empirical copula.

    Computes the empirical copula of (X, Y) after removing the conditioning
    set via the Rosenblatt transform, then tests for uniformity.

    Parameters
    ----------
    alpha : float
        Significance level.
    n_bootstrap : int
        Number of bootstrap samples for the copula test.
    seed : int
        Random seed.
    """

    method = CITestMethod.RANK

    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 200,
        seed: int = 42,
    ) -> None:
        super().__init__(alpha=alpha, seed=seed)
        self.n_bootstrap = n_bootstrap

    def test(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: pd.DataFrame,
    ) -> CITestResult:
        """Run the copula-based CI test.

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

        if n < max(k + 4, 20):
            return _insufficient_sample_result(
                x, y, conditioning_set, self.method, self.alpha
            )

        rng = np.random.default_rng(self.seed)

        # Remove conditioning via residuals on ranks
        if z_cols is not None and z_cols.shape[1] > 0:
            rx = _rank_with_ties(x_col)
            ry = _rank_with_ties(y_col)
            rz = _rank_matrix(z_cols)
            design = np.column_stack([np.ones(n), rz])
            try:
                cx, _, _, _ = np.linalg.lstsq(design, rx, rcond=None)
                cy, _, _, _ = np.linalg.lstsq(design, ry, rcond=None)
            except np.linalg.LinAlgError:
                return _insufficient_sample_result(
                    x, y, conditioning_set, self.method, self.alpha
                )
            u = rx - design @ cx
            v = ry - design @ cy
        else:
            u = x_col.copy()
            v = y_col.copy()

        # Transform to pseudo-uniform via ranks
        u_ranks = _rank_with_ties(u) / (n + 1)
        v_ranks = _rank_with_ties(v) / (n + 1)

        # Test statistic: Cramér–von Mises type for independence
        observed_stat = self._copula_cvm_statistic(u_ranks, v_ranks, n)

        # Bootstrap null distribution
        null_stats = np.empty(self.n_bootstrap)
        for b in range(self.n_bootstrap):
            perm_idx = rng.permutation(n)
            null_stats[b] = self._copula_cvm_statistic(
                u_ranks, v_ranks[perm_idx], n
            )

        # p-value: proportion of null statistics >= observed
        p_value = float(np.mean(null_stats >= observed_stat))
        # Ensure p-value is not exactly 0
        p_value = max(p_value, 1.0 / (self.n_bootstrap + 1))

        return self._make_result(x, y, conditioning_set, observed_stat, p_value)

    @staticmethod
    def _copula_cvm_statistic(
        u: np.ndarray, v: np.ndarray, n: int
    ) -> float:
        """Cramér–von Mises statistic for copula independence.

        Computes ``sum_{i} (C_n(u_i, v_i) - u_i * v_i)^2`` where
        ``C_n`` is the empirical copula.

        Parameters
        ----------
        u, v : np.ndarray
            Pseudo-uniform ranks in ``(0, 1)``.
        n : int
            Sample size.

        Returns
        -------
        float
            Test statistic.
        """
        # Empirical copula: C_n(u_i, v_i) = (1/n) * sum_j 1(u_j <= u_i, v_j <= v_i)
        # Vectorised computation
        u_le = u[:, None] >= u[None, :]  # (n, n) boolean
        v_le = v[:, None] >= v[None, :]
        Cn = np.mean(u_le & v_le, axis=1)
        # Independence copula: Pi(u, v) = u * v
        Pi = u * v
        stat = float(np.sum((Cn - Pi) ** 2))
        return stat

"""Chi-squared and G-test conditional independence tests for discrete data.

Both tests operate on contingency tables built from categorical variables
and their conditioning sets.  A :class:`FisherExactTest` is provided for
2×2 tables, and the :func:`discretize` helper converts continuous data
into discrete bins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Result container (mirrors fisher_z.CITestResult)
# ---------------------------------------------------------------------------

@dataclass
class DiscreteCIResult:
    """Result of a discrete CI test."""

    statistic: float
    p_value: float
    independent: bool
    degrees_of_freedom: int
    conditioning_set: Set[int] = field(default_factory=set)
    method: str = "chi_squared"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_discrete(data: NDArray, x: int, y: int, z: Set[int]) -> None:
    """Raise on invalid inputs."""
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array")
    p = data.shape[1]
    for v in {x, y} | z:
        if v < 0 or v >= p:
            raise IndexError(f"Variable index {v} out of range [0, {p})")


def _contingency_table(
    data: NDArray,
    x: int,
    y: int,
) -> NDArray[np.float64]:
    """Build a 2-D contingency table from two discrete columns.

    Returns an array of shape (n_levels_x, n_levels_y) with counts.
    """
    x_vals = data[:, x]
    y_vals = data[:, y]
    x_levels = np.unique(x_vals)
    y_levels = np.unique(y_vals)
    table = np.zeros((len(x_levels), len(y_levels)), dtype=np.float64)
    x_map = {v: i for i, v in enumerate(x_levels)}
    y_map = {v: i for i, v in enumerate(y_levels)}
    for xi, yi in zip(x_vals, y_vals):
        table[x_map[xi], y_map[yi]] += 1.0
    return table


def _stratified_tables(
    data: NDArray,
    x: int,
    y: int,
    z: Set[int],
) -> List[NDArray[np.float64]]:
    """Build one contingency table per stratum defined by *z*.

    The strata are the unique value-combinations of the columns in *z*.
    Returns a list of 2-D contingency tables (one per stratum).
    """
    if len(z) == 0:
        return [_contingency_table(data, x, y)]

    z_sorted = sorted(z)
    z_data = data[:, z_sorted]

    # Identify unique strata
    if z_data.ndim == 1:
        z_data = z_data.reshape(-1, 1)
    strata_keys, inverse = np.unique(
        z_data, axis=0, return_inverse=True
    )

    tables: List[NDArray[np.float64]] = []
    for s_idx in range(len(strata_keys)):
        mask = inverse == s_idx
        if mask.sum() == 0:
            continue
        tables.append(_contingency_table(data[mask], x, y))
    return tables


def _expected_frequencies(table: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute expected frequencies under independence for a 2-D table."""
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    total = table.sum()
    if total == 0:
        return np.zeros_like(table)
    return (row_sums * col_sums) / total


# ---------------------------------------------------------------------------
# Chi-squared test
# ---------------------------------------------------------------------------

class ChiSquaredTest:
    """Pearson chi-squared conditional independence test.

    Parameters
    ----------
    alpha : float
        Significance level for rejecting independence.
    yates : bool
        Apply Yates' continuity correction for 2×2 tables (default False).
    """

    def __init__(self, alpha: float = 0.05, yates: bool = False) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = alpha
        self.yates = yates

    def test(
        self,
        data: NDArray[np.int_],
        x: int,
        y: int,
        conditioning_set: Set[int] | None = None,
    ) -> Tuple[float, float]:
        """Run the conditional chi-squared test.

        The test aggregates chi-squared statistics over strata defined by
        the conditioning set (Cochran–Mantel–Haenszel style aggregation).

        Returns (statistic, p_value).
        """
        z = conditioning_set if conditioning_set is not None else set()
        data = np.asarray(data)
        _validate_discrete(data, x, y, z)

        tables = _stratified_tables(data, x, y, z)

        total_stat = 0.0
        total_df = 0
        for table in tables:
            s, df = self.compute_statistic(table)
            if df > 0:
                total_stat += s
                total_df += df

        if total_df <= 0:
            return (0.0, 1.0)

        pvalue = float(sp_stats.chi2.sf(total_stat, total_df))
        return (float(total_stat), pvalue)

    def test_full(
        self,
        data: NDArray[np.int_],
        x: int,
        y: int,
        conditioning_set: Set[int] | None = None,
    ) -> DiscreteCIResult:
        """Like :meth:`test` but returns a :class:`DiscreteCIResult`."""
        z = conditioning_set if conditioning_set is not None else set()
        stat, pval = self.test(data, x, y, z)
        tables = _stratified_tables(data, x, y, z)
        total_df = sum(
            max((t.shape[0] - 1) * (t.shape[1] - 1), 0) for t in tables
        )
        return DiscreteCIResult(
            statistic=stat,
            p_value=pval,
            independent=(pval >= self.alpha),
            degrees_of_freedom=total_df,
            conditioning_set=set(z),
            method="chi_squared",
        )

    def compute_statistic(
        self,
        contingency_table: NDArray[np.float64],
    ) -> Tuple[float, int]:
        """Compute chi-squared statistic and degrees of freedom.

        Parameters
        ----------
        contingency_table : ndarray of shape (r, c)

        Returns
        -------
        statistic : float
        df : int
        """
        table = np.asarray(contingency_table, dtype=np.float64)
        if table.ndim != 2:
            raise ValueError("contingency_table must be 2-D")

        # Remove zero-margin rows/columns for proper df calculation
        row_mask = table.sum(axis=1) > 0
        col_mask = table.sum(axis=0) > 0
        table = table[np.ix_(row_mask, col_mask)]

        r, c = table.shape
        if r <= 1 or c <= 1:
            return (0.0, 0)

        expected = _expected_frequencies(table)
        df = (r - 1) * (c - 1)

        # Yates correction for 2x2 tables
        if self.yates and r == 2 and c == 2:
            correction = 0.5
        else:
            correction = 0.0

        # Mask cells with expected > 0
        mask = expected > 0
        diff = np.abs(table[mask] - expected[mask]) - correction
        diff = np.maximum(diff, 0.0)
        stat = float(np.sum(diff ** 2 / expected[mask]))
        return (stat, df)


# ---------------------------------------------------------------------------
# G-test (likelihood ratio)
# ---------------------------------------------------------------------------

class GTest:
    """Log-likelihood ratio (G) conditional independence test.

    The G statistic is:  G = 2 ∑ O_ij ln(O_ij / E_ij)

    Parameters
    ----------
    alpha : float
        Significance level for rejecting independence.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = alpha

    def test(
        self,
        data: NDArray[np.int_],
        x: int,
        y: int,
        conditioning_set: Set[int] | None = None,
    ) -> Tuple[float, float]:
        """Run the conditional G-test.

        Aggregates G statistics across strata defined by *conditioning_set*.

        Returns (statistic, p_value).
        """
        z = conditioning_set if conditioning_set is not None else set()
        data = np.asarray(data)
        _validate_discrete(data, x, y, z)

        tables = _stratified_tables(data, x, y, z)

        total_stat = 0.0
        total_df = 0
        for table in tables:
            s, df = self.g_statistic(table)
            if df > 0:
                total_stat += s
                total_df += df

        if total_df <= 0:
            return (0.0, 1.0)

        pvalue = float(sp_stats.chi2.sf(total_stat, total_df))
        return (float(total_stat), pvalue)

    def test_full(
        self,
        data: NDArray[np.int_],
        x: int,
        y: int,
        conditioning_set: Set[int] | None = None,
    ) -> DiscreteCIResult:
        """Like :meth:`test` but returns a :class:`DiscreteCIResult`."""
        z = conditioning_set if conditioning_set is not None else set()
        stat, pval = self.test(data, x, y, z)
        tables = _stratified_tables(data, x, y, z)
        total_df = sum(
            max((t.shape[0] - 1) * (t.shape[1] - 1), 0) for t in tables
        )
        return DiscreteCIResult(
            statistic=stat,
            p_value=pval,
            independent=(pval >= self.alpha),
            degrees_of_freedom=total_df,
            conditioning_set=set(z),
            method="g_test",
        )

    def g_statistic(
        self,
        contingency_table: NDArray[np.float64],
    ) -> Tuple[float, int]:
        """Compute G statistic and degrees of freedom.

        G = 2 ∑ O_ij ln(O_ij / E_ij)

        Parameters
        ----------
        contingency_table : ndarray of shape (r, c)

        Returns
        -------
        statistic : float
        df : int
        """
        table = np.asarray(contingency_table, dtype=np.float64)
        if table.ndim != 2:
            raise ValueError("contingency_table must be 2-D")

        row_mask = table.sum(axis=1) > 0
        col_mask = table.sum(axis=0) > 0
        table = table[np.ix_(row_mask, col_mask)]

        r, c = table.shape
        if r <= 1 or c <= 1:
            return (0.0, 0)

        expected = _expected_frequencies(table)
        df = (r - 1) * (c - 1)

        mask = (table > 0) & (expected > 0)
        stat = 2.0 * float(np.sum(table[mask] * np.log(table[mask] / expected[mask])))
        # Ensure non-negative (can happen with rounding)
        stat = max(stat, 0.0)
        return (stat, df)


# ---------------------------------------------------------------------------
# Fisher exact test (2×2 tables)
# ---------------------------------------------------------------------------

class FisherExactTest:
    """Fisher exact test for 2×2 contingency tables.

    Parameters
    ----------
    alpha : float
        Significance level.
    alternative : str
        ``"two-sided"`` (default), ``"less"``, or ``"greater"``.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        alternative: str = "two-sided",
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if alternative not in ("two-sided", "less", "greater"):
            raise ValueError(f"Invalid alternative: {alternative!r}")
        self.alpha = alpha
        self.alternative = alternative

    def test(
        self,
        data: NDArray,
        x: int,
        y: int,
    ) -> Tuple[float, float]:
        """Run Fisher's exact test on two binary columns.

        Returns (odds_ratio, p_value).
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("data must be 2-D")
        table = _contingency_table(data, x, y)
        if table.shape != (2, 2):
            raise ValueError(
                f"Fisher exact test requires a 2×2 table; got {table.shape}. "
                "Variables must be binary."
            )
        odds_ratio, pvalue = sp_stats.fisher_exact(
            table.astype(int), alternative=self.alternative
        )
        return (float(odds_ratio), float(pvalue))

    def test_from_table(
        self,
        table: NDArray,
    ) -> Tuple[float, float]:
        """Run Fisher's exact test on a pre-built 2×2 table."""
        table = np.asarray(table, dtype=int)
        if table.shape != (2, 2):
            raise ValueError(
                f"Expected a 2×2 table; got shape {table.shape}"
            )
        odds_ratio, pvalue = sp_stats.fisher_exact(
            table, alternative=self.alternative
        )
        return (float(odds_ratio), float(pvalue))


# ---------------------------------------------------------------------------
# Discretization utility
# ---------------------------------------------------------------------------

def discretize(
    data: NDArray[np.float64],
    n_bins: int = 5,
    method: str = "equal_width",
    columns: Optional[List[int]] = None,
) -> NDArray[np.int_]:
    """Discretize continuous data into integer-coded bins.

    Parameters
    ----------
    data : ndarray of shape (n, p)
        Continuous data array.
    n_bins : int
        Number of bins per variable.
    method : str
        ``"equal_width"`` — equal-width binning.
        ``"equal_freq"``  — equal-frequency (quantile) binning.
        ``"kmeans"``      — k-means–based binning (1-D).
    columns : list of int or None
        Columns to discretize; ``None`` means all.

    Returns
    -------
    ndarray of int
        Same shape as *data*, with integer bin labels in [0, n_bins).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        raise ValueError("data must be 1-D or 2-D")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")

    n, p = data.shape
    result = np.empty_like(data, dtype=np.intp)
    cols = columns if columns is not None else list(range(p))

    for j in range(p):
        if j not in cols:
            result[:, j] = data[:, j].astype(np.intp)
            continue

        col = data[:, j]

        if method == "equal_width":
            lo, hi = col.min(), col.max()
            if hi - lo < 1e-15:
                result[:, j] = 0
            else:
                edges = np.linspace(lo, hi, n_bins + 1)
                edges[-1] += 1e-10  # include right edge
                result[:, j] = np.digitize(col, edges[1:])
                result[:, j] = np.clip(result[:, j], 0, n_bins - 1)

        elif method == "equal_freq":
            quantiles = np.linspace(0, 100, n_bins + 1)
            edges = np.percentile(col, quantiles)
            # Make edges unique
            edges = np.unique(edges)
            if len(edges) < 2:
                result[:, j] = 0
            else:
                result[:, j] = np.digitize(col, edges[1:])
                result[:, j] = np.clip(result[:, j], 0, len(edges) - 2)

        elif method == "kmeans":
            # Simple 1-D k-means via iterative assignment
            result[:, j] = _kmeans_1d(col, n_bins)

        else:
            raise ValueError(f"Unknown discretization method: {method!r}")

    return result


def _kmeans_1d(x: NDArray[np.float64], k: int, max_iter: int = 100) -> NDArray[np.intp]:
    """Simple 1-D k-means for discretization."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.intp)
    if k >= n:
        return np.arange(n, dtype=np.intp)

    # Initialize centroids via quantile spread
    centroids = np.percentile(x, np.linspace(0, 100, k))
    labels = np.zeros(n, dtype=np.intp)

    for _ in range(max_iter):
        # Assign to nearest centroid
        dists = np.abs(x[:, None] - centroids[None, :])
        new_labels = np.argmin(dists, axis=1).astype(np.intp)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centroids
        for ci in range(k):
            mask = labels == ci
            if mask.sum() > 0:
                centroids[ci] = x[mask].mean()

    return labels

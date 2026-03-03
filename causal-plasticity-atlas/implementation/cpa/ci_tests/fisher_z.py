"""Fisher-z transform test for linear-Gaussian conditional independence.

Uses partial correlation and the Fisher-z transform to test
X ⊥ Y | Z under the assumption of multivariate normality.

Two approaches for computing partial correlations are provided:

1. **Matrix inversion** — invert the sub-covariance matrix to obtain the
   precision matrix and read off the partial correlation.
2. **Recursive formula** — peel off conditioning variables one at a time
   using the first-order partial-correlation identity.

Both are wrapped in the :class:`PartialCorrelation` helper which is used
internally by :class:`FisherZTest`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CITestResult:
    """Container for conditional-independence test results.

    Attributes
    ----------
    statistic : float
        Test statistic value.
    p_value : float
        Two-sided p-value associated with *statistic*.
    independent : bool
        ``True`` when *p_value* >= significance level (fail to reject H0).
    conditioning_set : set[int]
        Indices of the conditioning variables used.
    method : str
        Name of the CI test that produced this result.
    partial_corr : float | None
        Partial correlation (when applicable).
    """

    statistic: float
    p_value: float
    independent: bool
    conditioning_set: Set[int] = field(default_factory=set)
    method: str = "fisher_z"
    partial_corr: Optional[float] = None


# ---------------------------------------------------------------------------
# Partial-correlation computation
# ---------------------------------------------------------------------------

class PartialCorrelation:
    """Compute partial correlations via matrix inversion or recursive formula.

    Parameters
    ----------
    method : str
        ``"matrix_inversion"`` (default) or ``"recursive"``.
    """

    METHODS = ("matrix_inversion", "recursive")

    def __init__(self, method: str = "matrix_inversion") -> None:
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown method {method!r}; choose from {self.METHODS}"
            )
        self.method = method

    # -- public interface ---------------------------------------------------

    def compute(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        z: Set[int],
    ) -> float:
        """Return the sample partial correlation r(x, y | z).

        Parameters
        ----------
        data : ndarray of shape (n, p)
            Observation matrix (rows = samples, columns = variables).
        x, y : int
            Column indices of the two target variables.
        z : set of int
            Column indices of the conditioning set (may be empty).

        Returns
        -------
        float
            Partial correlation in [-1, 1].
        """
        if data.ndim != 2:
            raise ValueError("data must be a 2-D array")
        n, p = data.shape
        all_vars = {x, y} | z
        if any(v < 0 or v >= p for v in all_vars):
            raise IndexError(
                f"Variable indices must be in [0, {p}); got {all_vars}"
            )
        if n < len(all_vars) + 1:
            raise ValueError(
                "Not enough samples to estimate partial correlation "
                f"(n={n}, need >= {len(all_vars) + 1})"
            )

        cov = np.cov(data[:, sorted(all_vars)], rowvar=False, ddof=1)
        idx_map = {v: i for i, v in enumerate(sorted(all_vars))}

        if self.method == "matrix_inversion":
            return self._matrix_inversion(cov, idx_map[x], idx_map[y],
                                          {idx_map[v] for v in z})
        return self._recursive_formula(cov, idx_map[x], idx_map[y],
                                       {idx_map[v] for v in z})

    def compute_from_cov(
        self,
        cov: NDArray[np.float64],
        x: int,
        y: int,
        z: Set[int],
    ) -> float:
        """Compute partial correlation directly from a covariance matrix.

        Parameters
        ----------
        cov : ndarray of shape (p, p)
            Full covariance matrix.
        x, y : int
            Indices into *cov* for the two target variables.
        z : set of int
            Indices of conditioning variables.

        Returns
        -------
        float
            Partial correlation in [-1, 1].
        """
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("cov must be a square matrix")

        all_vars = sorted({x, y} | z)
        sub_cov = cov[np.ix_(all_vars, all_vars)]
        idx_map = {v: i for i, v in enumerate(all_vars)}

        if self.method == "matrix_inversion":
            return self._matrix_inversion(sub_cov, idx_map[x], idx_map[y],
                                          {idx_map[v] for v in z})
        return self._recursive_formula(sub_cov, idx_map[x], idx_map[y],
                                       {idx_map[v] for v in z})

    # -- matrix-inversion approach -----------------------------------------

    @staticmethod
    def _matrix_inversion(
        cov: NDArray[np.float64],
        x: int,
        y: int,
        z: Set[int],
    ) -> float:
        r"""Partial correlation via the precision matrix.

        Given precision matrix P = Σ⁻¹, the partial correlation is:

        .. math::
            r(x, y | z) = -\frac{P_{xy}}{\sqrt{P_{xx} P_{yy}}}
        """
        if len(z) == 0:
            # Simple Pearson correlation
            sx = np.sqrt(cov[x, x])
            sy = np.sqrt(cov[y, y])
            if sx < 1e-15 or sy < 1e-15:
                return 0.0
            return float(cov[x, y] / (sx * sy))

        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Singular — fall back to pseudo-inverse
            precision = np.linalg.pinv(cov)

        denom = np.sqrt(np.abs(precision[x, x] * precision[y, y]))
        if denom < 1e-15:
            return 0.0
        r = float(-precision[x, y] / denom)
        return np.clip(r, -1.0, 1.0)

    # -- recursive approach ------------------------------------------------

    @staticmethod
    def _recursive_formula(
        cov: NDArray[np.float64],
        x: int,
        y: int,
        z: Set[int],
    ) -> float:
        r"""Partial correlation via the recursive (order-reduction) formula.

        Base case (|z| = 0): Pearson correlation.

        Recursive step — pick an arbitrary element *k* from *z*:

        .. math::
            r(x,y \mid z) = \frac{
                r(x,y \mid z \setminus k) - r(x,k \mid z \setminus k)\,r(y,k \mid z \setminus k)
            }{
                \sqrt{(1 - r(x,k \mid z \setminus k)^2)(1 - r(y,k \mid z \setminus k)^2)}
            }
        """
        if len(z) == 0:
            sx = np.sqrt(cov[x, x])
            sy = np.sqrt(cov[y, y])
            if sx < 1e-15 or sy < 1e-15:
                return 0.0
            return float(cov[x, y] / (sx * sy))

        k = next(iter(z))
        z_rest = z - {k}
        recurse = PartialCorrelation._recursive_formula

        r_xy = recurse(cov, x, y, z_rest)
        r_xk = recurse(cov, x, k, z_rest)
        r_yk = recurse(cov, y, k, z_rest)

        denom = np.sqrt((1.0 - r_xk ** 2) * (1.0 - r_yk ** 2))
        if denom < 1e-15:
            return 0.0
        r = (r_xy - r_xk * r_yk) / denom
        return float(np.clip(r, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Fisher-z test
# ---------------------------------------------------------------------------

class FisherZTest:
    """Fisher-z conditional independence test.

    Tests the null hypothesis H₀: X ⊥ Y | Z under the assumption that the
    joint distribution of (X, Y, Z) is multivariate Gaussian.

    Parameters
    ----------
    alpha : float
        Significance level for rejecting independence.
    method : str
        Partial-correlation method: ``"matrix_inversion"`` (default) or
        ``"recursive"``.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        method: str = "matrix_inversion",
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = alpha
        self._pc = PartialCorrelation(method=method)

    # -- main entry point ---------------------------------------------------

    def test(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        conditioning_set: Set[int] | None = None,
    ) -> Tuple[float, float]:
        """Run the Fisher-z test.

        Parameters
        ----------
        data : ndarray of shape (n, p)
            Observation matrix.
        x, y : int
            Column indices to test for conditional independence.
        conditioning_set : set of int or None
            Column indices to condition on.

        Returns
        -------
        stat : float
            The test statistic (z-score).
        pvalue : float
            Two-sided p-value.
        """
        z = conditioning_set if conditioning_set is not None else set()
        data = np.asarray(data, dtype=np.float64)

        if data.ndim != 2:
            raise ValueError("data must be a 2-D array")
        n, p = data.shape

        # Validate indices
        all_idx = {x, y} | z
        if any(i < 0 or i >= p for i in all_idx):
            raise IndexError(
                f"Variable indices out of range [0, {p}): {all_idx}"
            )

        k = len(z)
        if n - k - 3 <= 0:
            # Insufficient degrees of freedom — cannot reject
            return (0.0, 1.0)

        r = self.partial_correlation(data, x, y, z)
        stat = self.fisher_z_transform(r, n, k)
        pvalue = 2.0 * stats.norm.sf(np.abs(stat))

        return (float(stat), float(pvalue))

    def test_full(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        conditioning_set: Set[int] | None = None,
    ) -> CITestResult:
        """Like :meth:`test` but returns a :class:`CITestResult`."""
        z = conditioning_set if conditioning_set is not None else set()
        stat, pvalue = self.test(data, x, y, z)
        r = self.partial_correlation(data, x, y, z)
        return CITestResult(
            statistic=stat,
            p_value=pvalue,
            independent=(pvalue >= self.alpha),
            conditioning_set=set(z),
            method="fisher_z",
            partial_corr=r,
        )

    # -- partial correlation ------------------------------------------------

    def partial_correlation(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        z: Set[int],
    ) -> float:
        """Compute the partial correlation of X and Y given Z."""
        return self._pc.compute(data, x, y, z)

    def partial_correlation_from_cov(
        self,
        cov: NDArray[np.float64],
        x: int,
        y: int,
        z: Set[int],
    ) -> float:
        """Compute partial correlation from a covariance matrix."""
        return self._pc.compute_from_cov(cov, x, y, z)

    # -- Fisher-z statistic ------------------------------------------------

    @staticmethod
    def fisher_z_transform(r: float, n: int, k: int) -> float:
        """Apply the Fisher-z transform to partial correlation *r*.

        Parameters
        ----------
        r : float
            Partial correlation coefficient.
        n : int
            Sample size.
        k : int
            Size of the conditioning set.

        Returns
        -------
        float
            z-statistic  ``sqrt(n - k - 3) * arctanh(r)``.
        """
        if n - k - 3 <= 0:
            return 0.0
        # Clamp r away from ±1 to avoid inf in arctanh
        r = np.clip(r, -1.0 + 1e-10, 1.0 - 1e-10)
        z = np.sqrt(n - k - 3) * 0.5 * np.log((1.0 + r) / (1.0 - r))
        return float(z)

    # -- convenience --------------------------------------------------------

    def all_pairwise(
        self,
        data: NDArray[np.float64],
        conditioning_set: Set[int] | None = None,
    ) -> NDArray[np.float64]:
        """Return a matrix of p-values for every pair (i, j).

        Parameters
        ----------
        data : ndarray of shape (n, p)
        conditioning_set : set of int or None
            Variables to condition on for every pair.

        Returns
        -------
        pval_matrix : ndarray of shape (p, p)
            Symmetric matrix of p-values; diagonal entries are 1.0.
        """
        data = np.asarray(data, dtype=np.float64)
        p = data.shape[1]
        z = conditioning_set if conditioning_set is not None else set()
        pvals = np.ones((p, p), dtype=np.float64)
        for i in range(p):
            if i in z:
                continue
            for j in range(i + 1, p):
                if j in z:
                    continue
                _, pv = self.test(data, i, j, z)
                pvals[i, j] = pv
                pvals[j, i] = pv
        return pvals

    def stable_test(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        candidate_z: List[int],
        max_order: int | None = None,
    ) -> Tuple[bool, Set[int]]:
        """Order-independent (stable) search for a separating set.

        Iterates through subsets of *candidate_z* (up to *max_order*) and
        returns as soon as conditional independence is detected.

        Parameters
        ----------
        data : ndarray of shape (n, p)
        x, y : int
            Variables to test.
        candidate_z : list of int
            Candidate conditioning variables.
        max_order : int or None
            Maximum conditioning-set size.  ``None`` means no limit.

        Returns
        -------
        independent : bool
        separating_set : set of int
        """
        from itertools import combinations

        n = data.shape[0]
        if max_order is None:
            max_order = len(candidate_z)
        max_order = min(max_order, len(candidate_z))

        for order in range(max_order + 1):
            if n - order - 3 <= 0:
                break
            for subset in combinations(candidate_z, order):
                z = set(subset)
                _, pv = self.test(data, x, y, z)
                if pv >= self.alpha:
                    return True, z
        return False, set()

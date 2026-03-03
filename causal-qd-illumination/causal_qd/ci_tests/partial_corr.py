"""Partial-correlation based conditional independence test.

This module provides a full-featured implementation of partial correlation
for testing conditional independence (CI) hypotheses of the form X ⊥ Y | S.

Three estimation strategies are available:

* **precision** – Invert the (optionally shrunk) sample covariance matrix of
  the submatrix ``{X, Y} ∪ S`` and read off the partial correlation from the
  precision matrix.  This is the default and typically the most stable method.

* **recursive** – Apply the first-order recursive formula for partial
  correlation, peeling off one conditioning variable at a time.  Exact for
  Gaussian data and can be numerically advantageous when ``|S|`` is very
  small, but may accumulate rounding errors for larger conditioning sets.

* **regression** – Regress X and Y separately on S via ordinary least squares,
  then compute the Pearson correlation of the residuals.  Conceptually
  straightforward and equivalent to the precision method under normality.

All three methods yield the same partial correlation under ideal (infinite
precision, Gaussian) conditions; the choice among them is a practical
trade-off between numerical stability, speed, and interpretability.

A Ledoit–Wolf shrinkage estimator is available for regularising the sample
covariance matrix, which is particularly useful when the conditioning set is
large relative to the sample size.

The statistical test is a two-sided Student-*t* test with
``n − |S| − 2`` degrees of freedom derived from Fisher's z-transform of the
partial correlation coefficient.

References
----------
* Baba, K., Shibata, R., & Sibuya, M. (2004). Partial correlation and
  conditional correlation as measures of conditional independence.
  *Australian & New Zealand Journal of Statistics*, 46(4), 657–664.
* Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices. *Journal of Multivariate Analysis*,
  88(2), 365–411.
* Kalisch, M. & Bühlmann, P. (2007). Estimating high-dimensional directed
  acyclic graphs with the PC-algorithm. *JMLR*, 8, 613–636.
"""

from __future__ import annotations

import math
from typing import FrozenSet, List

import numpy as np
from scipy import stats as sp_stats

from causal_qd.ci_tests.ci_base import CITest, CITestResult
from causal_qd.types import DataMatrix, PValue

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-15
"""Small epsilon used to guard against division by zero."""

_COV_REG: float = 1e-10
"""Tiny ridge added to the covariance diagonal for numerical stability."""

_VALID_METHODS = frozenset({"precision", "recursive", "regression"})
"""Supported estimation methods."""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PartialCorrelationTest(CITest):
    """Test conditional independence via partial correlation and a Student-*t* test.

    This class supports three methods for computing the partial correlation
    coefficient r(X, Y | S):

    1. **precision** (default) – via the inverse of the covariance matrix.
    2. **recursive** – via the first-order recursive partial-correlation formula.
    3. **regression** – via OLS residual correlation.

    An optional Ledoit–Wolf style shrinkage parameter ``shrinkage`` in [0, 1]
    can be supplied to regularise the sample covariance matrix.  When
    ``shrinkage == 0`` (default), no shrinkage is applied; when
    ``shrinkage == 1``, the covariance is fully shrunk to a diagonal matrix.
    Intermediate values blend between the sample covariance and its diagonal.

    The null hypothesis H₀: X ⊥ Y | S is tested with a two-sided Student-*t*
    test using the statistic::

        t = r * sqrt(dof / (1 - r²)),   dof = n - |S| - 2

    and p-value ``2 * (1 - F_t(|t|, dof))``, where F_t is the CDF of the *t*
    distribution with *dof* degrees of freedom.

    Parameters
    ----------
    method : str, default ``"precision"``
        Which estimation strategy to use.  One of ``"precision"``,
        ``"recursive"``, or ``"regression"``.
    shrinkage : float, default ``0.0``
        Ledoit–Wolf style shrinkage intensity in ``[0, 1]``.
        * ``0.0`` – no shrinkage (pure sample covariance).
        * ``1.0`` – full shrinkage to ``diag(Σ) · I``.
        * ``-1.0`` – automatically estimate the optimal shrinkage
          parameter using the Ledoit–Wolf formula.

    Raises
    ------
    ValueError
        If *method* is not one of the supported strategies or *shrinkage*
        is outside ``[-1, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from causal_qd.ci_tests.partial_corr import PartialCorrelationTest
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((200, 4))
    >>> ci = PartialCorrelationTest(method="precision", shrinkage=0.0)
    >>> result = ci.test(0, 1, frozenset({2, 3}), data)
    >>> result.is_independent
    True
    """

    # ------------------------------------------------------------------ init
    def __init__(self, method: str = "precision", shrinkage: float = 0.0) -> None:
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from {sorted(_VALID_METHODS)}."
            )
        if not (-1.0 <= shrinkage <= 1.0):
            raise ValueError(
                f"shrinkage must be in [-1, 1], got {shrinkage}."
            )
        self._method: str = method
        self._shrinkage: float = shrinkage

    # ------------------------------------------------------------------ repr
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PartialCorrelationTest(method={self._method!r}, "
            f"shrinkage={self._shrinkage})"
        )

    # ============================================================= public API
    def test(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
        alpha: float = 0.05,
    ) -> CITestResult:
        """Perform the partial-correlation CI test.

        Parameters
        ----------
        x, y : int
            Column indices of the two variables whose conditional
            independence is to be tested.
        conditioning_set : FrozenSet[int]
            Column indices to condition on (may be empty).
        data : DataMatrix
            Observed data matrix of shape ``(n, p)`` with *n*
            observations and *p* variables.
        alpha : float, default ``0.05``
            Significance level for the two-sided test.

        Returns
        -------
        CITestResult
            A frozen dataclass containing the *t*-statistic, p-value,
            independence decision, and the conditioning set used.

        Notes
        -----
        When the effective degrees of freedom ``n - |S| - 2`` fall below 1
        the test is degenerate and the method returns a conservative result
        declaring independence (p-value = 1).
        """
        n: int = data.shape[0]
        s_size: int = len(conditioning_set)
        dof: int = n - s_size - 2

        # Degenerate case: insufficient degrees of freedom.
        if dof < 1:
            return CITestResult(
                statistic=0.0,
                p_value=1.0,
                is_independent=True,
                conditioning_set=conditioning_set,
            )

        # ----- compute partial correlation via the selected method ----------
        r: float = self._dispatch_partial_corr(x, y, conditioning_set, data)

        # ----- Student-t test ----------------------------------------------
        t_stat, p_value = self._student_t_test(r, dof)

        return CITestResult(
            statistic=t_stat,
            p_value=p_value,
            is_independent=(p_value >= alpha),
            conditioning_set=conditioning_set,
        )

    # ============================================ method dispatch (private) ==
    def _dispatch_partial_corr(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
    ) -> float:
        """Route to the appropriate partial-correlation estimator.

        Parameters
        ----------
        x, y : int
            Variable column indices.
        conditioning_set : FrozenSet[int]
            Conditioning column indices.
        data : DataMatrix
            Full data matrix.

        Returns
        -------
        float
            Partial correlation coefficient in ``[-1, 1]``.
        """
        if self._method == "precision":
            return self._partial_corr_precision(x, y, conditioning_set, data)
        elif self._method == "recursive":
            return self._partial_corr_recursive(x, y, conditioning_set, data)
        else:  # regression
            return self._partial_corr_regression(x, y, conditioning_set, data)

    # ====================================================== precision method
    def _partial_corr_precision(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
    ) -> float:
        """Compute partial correlation via the precision (inverse covariance) matrix.

        Steps
        -----
        1. Extract the submatrix of columns ``{x, y} ∪ S``.
        2. Compute the sample covariance and optionally apply shrinkage.
        3. Invert to obtain the precision matrix P.
        4. Read off ``r = -P[x, y] / sqrt(P[x, x] * P[y, y])``.

        Parameters
        ----------
        x, y : int
            Column indices of interest.
        conditioning_set : FrozenSet[int]
            Conditioning columns.
        data : DataMatrix
            Full ``(n, p)`` data matrix.

        Returns
        -------
        float
            Partial correlation in ``[-1, 1]``.
        """
        indices: List[int] = sorted({x, y} | set(conditioning_set))
        sub: DataMatrix = data[:, indices]

        # Sample covariance with optional shrinkage.
        cov: np.ndarray = np.cov(sub, rowvar=False)
        if cov.ndim == 0:
            # Only one variable – degenerate; partial corr is undefined.
            return 0.0

        cov = self._apply_shrinkage(cov, sub)

        # Small ridge for numerical safety.
        cov += _COV_REG * np.eye(cov.shape[0])

        # Invert to obtain precision matrix.
        try:
            precision: np.ndarray = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return 0.0

        # Map original column indices to submatrix positions.
        idx_map = {v: i for i, v in enumerate(indices)}
        ix, iy = idx_map[x], idx_map[y]

        denom: float = math.sqrt(abs(precision[ix, ix] * precision[iy, iy]))
        if denom < _EPS:
            return 0.0

        return float(-precision[ix, iy] / denom)

    # ====================================================== recursive method
    def _partial_corr_recursive(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
    ) -> float:
        """Compute partial correlation via the first-order recursive formula.

        The recursive definition peels off one conditioning variable *z*
        at each level::

            r(X, Y | S ∪ {z}) =
                (r(X, Y | S) - r(X, z | S) * r(Y, z | S))
                / sqrt((1 - r(X, z | S)²) * (1 - r(Y, z | S)²))

        Base case (empty conditioning set): the ordinary Pearson
        correlation.

        Parameters
        ----------
        x, y : int
            Column indices.
        conditioning_set : FrozenSet[int]
            Conditioning columns.
        data : DataMatrix
            Full data matrix.

        Returns
        -------
        float
            Partial correlation coefficient.

        Notes
        -----
        This method has time complexity exponential in ``|S|`` in the
        naive implementation but we memoise intermediate calls so the
        effective complexity is ``O(|S|²)`` unique sub-problems.
        """
        # Memoise to avoid redundant recomputation.
        cache: dict[tuple[int, int, FrozenSet[int]], float] = {}

        def _r(a: int, b: int, s: FrozenSet[int]) -> float:
            """Recursive helper with memoisation."""
            # Canonical ordering to exploit symmetry r(a,b|S) == r(b,a|S).
            key = (min(a, b), max(a, b), s)
            if key in cache:
                return cache[key]

            if not s:
                # Base case: Pearson correlation.
                col_a = data[:, a]
                col_b = data[:, b]
                val = float(np.corrcoef(col_a, col_b)[0, 1])
                if np.isnan(val):
                    val = 0.0
                cache[key] = val
                return val

            # Pick an arbitrary element z from S.
            s_list = sorted(s)
            z = s_list[-1]
            s_minus_z = frozenset(s_list[:-1])

            r_ab = _r(a, b, s_minus_z)
            r_az = _r(a, z, s_minus_z)
            r_bz = _r(b, z, s_minus_z)

            denom = math.sqrt(
                max(0.0, (1.0 - r_az * r_az))
                * max(0.0, (1.0 - r_bz * r_bz))
            )

            if denom < _EPS:
                result = 0.0
            else:
                result = (r_ab - r_az * r_bz) / denom

            # Clamp to valid range.
            result = max(-1.0, min(1.0, result))
            cache[key] = result
            return result

        return _r(x, y, conditioning_set)

    # ====================================================== regression method
    def _partial_corr_regression(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
    ) -> float:
        """Compute partial correlation via OLS residual correlation.

        Algorithm
        ---------
        1. Regress ``data[:, x]`` on the columns in *S* → residuals ``e_x``.
        2. Regress ``data[:, y]`` on the columns in *S* → residuals ``e_y``.
        3. Return ``corr(e_x, e_y)``.

        When *S* is empty, this reduces to the ordinary Pearson correlation.

        Parameters
        ----------
        x, y : int
            Column indices.
        conditioning_set : FrozenSet[int]
            Conditioning columns.
        data : DataMatrix
            Full data matrix.

        Returns
        -------
        float
            Partial correlation in ``[-1, 1]``.
        """
        if not conditioning_set:
            r = float(np.corrcoef(data[:, x], data[:, y])[0, 1])
            return 0.0 if np.isnan(r) else r

        s_idx = sorted(conditioning_set)
        Z = data[:, s_idx]

        e_x = self._ols_residuals(data[:, x], Z)
        e_y = self._ols_residuals(data[:, y], Z)

        r = float(np.corrcoef(e_x, e_y)[0, 1])
        return 0.0 if np.isnan(r) else r

    # ========================================== shrinkage helpers (private) ==
    def _apply_shrinkage(
        self, cov: np.ndarray, data_sub: DataMatrix
    ) -> np.ndarray:
        """Apply Ledoit–Wolf style shrinkage to a sample covariance matrix.

        The shrunk estimator is::

            Σ_shrunk = (1 - λ) · Σ + λ · diag(Σ) · I

        where ``diag(Σ)`` denotes the matrix with only the diagonal of Σ
        retained (off-diagonals set to zero), and λ is the shrinkage
        intensity.

        Parameters
        ----------
        cov : np.ndarray
            Sample covariance matrix, shape ``(k, k)``.
        data_sub : DataMatrix
            The data submatrix from which *cov* was computed, used when
            automatic shrinkage estimation is requested
            (``self._shrinkage == -1``).

        Returns
        -------
        np.ndarray
            Shrunk covariance matrix.
        """
        if self._shrinkage == 0.0:
            return cov

        lam: float
        if self._shrinkage < 0.0:
            # Automatic Ledoit–Wolf estimation.
            lam = self._ledoit_wolf_shrinkage(data_sub)
        else:
            lam = self._shrinkage

        target = np.diag(np.diag(cov))
        return (1.0 - lam) * cov + lam * target

    @staticmethod
    def _ledoit_wolf_shrinkage(data: DataMatrix) -> float:
        """Estimate the optimal Ledoit–Wolf shrinkage intensity.

        Implements the analytical formula from Ledoit & Wolf (2004) for
        the shrinkage intensity that minimises the expected Frobenius loss
        between the shrinkage estimator and the true covariance matrix.

        The target is the diagonal matrix ``diag(S)`` where *S* is the
        sample covariance.

        Parameters
        ----------
        data : DataMatrix
            Centred (or un-centred) data matrix of shape ``(n, p)``.

        Returns
        -------
        float
            Optimal shrinkage intensity λ* in ``[0, 1]``.

        Notes
        -----
        The formula is (with ``S = cov(data)``):

        .. math::

            \\lambda^* = \\frac{\\sum_{i \\neq j} \\widehat{\\mathrm{Var}}(s_{ij})}
                              {\\sum_{i \\neq j} s_{ij}^2}

        clamped to ``[0, 1]``.
        """
        n, p = data.shape
        if n < 2 or p < 2:
            return 0.0

        # Centre the data.
        X = data - data.mean(axis=0)

        # Sample covariance (with 1/n normalisation for the LW formula).
        S = (X.T @ X) / n

        # Squared Frobenius norm of off-diagonal part of S.
        off_diag_mask = ~np.eye(p, dtype=bool)
        sum_sq_off = float(np.sum(S[off_diag_mask] ** 2))

        if sum_sq_off < _EPS:
            return 1.0

        # Estimate Var(s_{ij}) for each off-diagonal entry.
        # Var(s_{ij}) ≈ (1/n²) Σ_k (x_{ki} x_{kj} - s_{ij})² .
        # We vectorise over all (i,j) pairs.
        XtX_elementwise = np.zeros((p, p), dtype=np.float64)
        for k in range(n):
            xk = X[k, :][:, np.newaxis]  # (p, 1)
            outer_k = xk @ xk.T  # (p, p)
            diff = outer_k - S
            XtX_elementwise += diff ** 2
        XtX_elementwise /= n

        sum_var_off = float(np.sum(XtX_elementwise[off_diag_mask]))

        lam = sum_var_off / sum_sq_off
        return float(max(0.0, min(1.0, lam)))

    # ============================================= OLS helpers (private) ====
    @staticmethod
    def _ols_residuals(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Compute OLS residuals of regressing *y* on *Z*.

        Uses the normal equations ``β = (Z'Z)⁻¹ Z'y`` with a small ridge
        for numerical stability.

        Parameters
        ----------
        y : np.ndarray
            Response vector of length *n*.
        Z : np.ndarray
            Design matrix of shape ``(n, k)``.

        Returns
        -------
        np.ndarray
            Residual vector ``y - Z β`` of length *n*.
        """
        if Z.ndim == 1:
            Z = Z[:, np.newaxis]

        # Add small ridge for stability.
        ZtZ = Z.T @ Z + _COV_REG * np.eye(Z.shape[1])
        try:
            beta = np.linalg.solve(ZtZ, Z.T @ y)
        except np.linalg.LinAlgError:
            return y  # fall back to raw values
        return y - Z @ beta

    # ======================================== Student-t test (private) ======
    @staticmethod
    def _student_t_test(r: float, dof: int) -> tuple[float, PValue]:
        """Compute the two-sided Student-*t* test statistic and p-value.

        The test transforms the partial correlation *r* to a *t*-statistic::

            t = r * sqrt(dof / (1 - r²))

        with ``dof = n - |S| - 2`` degrees of freedom, then computes the
        two-sided p-value from the *t* distribution.

        Parameters
        ----------
        r : float
            Partial correlation coefficient (should be in ``[-1, 1]``).
        dof : int
            Degrees of freedom (must be ≥ 1).

        Returns
        -------
        tuple[float, PValue]
            ``(t_statistic, p_value)``
        """
        # Clamp r to avoid numerical issues at the boundary.
        r_clamped: float = max(-1.0 + 1e-12, min(1.0 - 1e-12, r))

        t_stat: float = r_clamped * math.sqrt(dof / (1.0 - r_clamped ** 2))
        p_value: PValue = float(2.0 * sp_stats.t.sf(abs(t_stat), df=dof))

        return t_stat, p_value

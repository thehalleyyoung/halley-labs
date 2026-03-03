"""Fisher's Z-transform conditional independence test.

This module implements the classical Fisher-Z test for conditional
independence under the assumption of multivariate Gaussian data.  The
test is based on the partial correlation between two variables X and Y
given a conditioning set S.  The partial correlation is estimated via
the precision-matrix (inverse covariance) approach and then transformed
using Fisher's Z-transform to obtain an approximately standard-normal
test statistic.

Two methods for computing partial correlations are provided:

1. **Precision-matrix method** (``_partial_correlation``): computes the
   inverse of the sub-covariance matrix of ``{X, Y} ∪ S`` and extracts
   the partial correlation from the precision matrix entries.  This is
   numerically stable for moderate conditioning-set sizes.

2. **Recursive method** (``_partial_correlation_recursive``): applies
   the recursive formula
   ``r_{XY|S} = (r_{XY|S\\{Z}} - r_{XZ|S\\{Z}} * r_{YZ|S\\{Z}}) /
   sqrt((1 - r_{XZ|S\\{Z}}^2)(1 - r_{YZ|S\\{Z}}^2))``
   which peels off one conditioning variable at a time.  This can be
   useful for very small conditioning sets where matrix inversion is
   overkill, or for pedagogical purposes.

Multiple-testing correction is supported through the ``test_multiple``
method, which accepts a list of ``(x, y, conditioning_set)`` triples
and applies either Bonferroni or Benjamini-Hochberg correction to the
resulting p-values.

References
----------
.. [1] Fisher, R. A. (1921). On the "probable error" of a coefficient
       of correlation deduced from a small sample. *Metron*, 1, 3–32.
.. [2] Kalisch, M., & Bühlmann, P. (2007). Estimating high-dimensional
       directed acyclic graphs with the PC-algorithm. *JMLR*, 8,
       613–636.
"""

from __future__ import annotations

import math
from typing import FrozenSet, List, Tuple

import numpy as np
from scipy import stats as sp_stats

from causal_qd.ci_tests.ci_base import CITest, CITestResult
from causal_qd.types import DataMatrix, PValue

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CORR_CLAMP_EPS: float = 1e-12
"""Small epsilon used to clamp partial correlations away from ±1."""

_REG_EPS: float = 1e-10
"""Regularisation added to the diagonal of the covariance matrix."""

_DENOM_TOL: float = 1e-15
"""Tolerance below which a denominator is considered zero."""


class FisherZTest(CITest):
    """Fisher's Z-transform test for conditional independence.

    Assumes that the data follow a multivariate Gaussian distribution.
    The test computes the partial correlation between X and Y given S,
    applies Fisher's Z-transform, and uses the standard normal
    distribution to obtain a two-sided p-value.

    The test statistic is::

        Z = 0.5 * ln((1 + r) / (1 - r)) * sqrt(n - |S| - 3)

    where *r* is the sample partial correlation and *n* is the number
    of observations.

    Parameters
    ----------
    correction : str, optional
        Multiple-testing correction method applied when using
        ``test_multiple``.  One of ``"none"`` (default),
        ``"bonferroni"``, or ``"benjamini_hochberg"``.

    Raises
    ------
    ValueError
        If *correction* is not one of the recognised methods.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((200, 4))
    >>> ci = FisherZTest()
    >>> result = ci.test(0, 1, frozenset(), data)
    >>> result.is_independent
    True
    """

    _VALID_CORRECTIONS = {"none", "bonferroni", "benjamini_hochberg"}

    def __init__(self, correction: str = "none") -> None:
        """Initialise the Fisher-Z test.

        Parameters
        ----------
        correction : str, optional
            Multiple-testing correction strategy.  Must be one of
            ``"none"``, ``"bonferroni"``, or ``"benjamini_hochberg"``.
            The correction is only applied when calling
            ``test_multiple``; the single ``test`` method always uses
            the raw p-value.

        Raises
        ------
        ValueError
            If *correction* is not a recognised method name.
        """
        if correction not in self._VALID_CORRECTIONS:
            raise ValueError(
                f"Unknown correction method '{correction}'. "
                f"Choose from {sorted(self._VALID_CORRECTIONS)}."
            )
        self.correction: str = correction

    # ------------------------------------------------------------------
    # Public API – single test
    # ------------------------------------------------------------------

    def test(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
        alpha: float = 0.05,
    ) -> CITestResult:
        """Perform the Fisher-Z conditional independence test.

        Computes the partial correlation between columns *x* and *y*
        of *data* given the columns in *conditioning_set*, transforms
        it with the Fisher Z-transform, and returns a two-sided
        p-value based on the standard normal distribution.

        Parameters
        ----------
        x : int
            Column index of the first variable.
        y : int
            Column index of the second variable.
        conditioning_set : FrozenSet[int]
            Frozenset of column indices to condition on.
        data : DataMatrix
            Observed data matrix of shape ``(N, p)`` where *N* is the
            number of observations and *p* is the number of variables.
        alpha : float, optional
            Significance level for the independence decision.  The null
            hypothesis (independence) is *not* rejected when the
            p-value is at least *alpha*.

        Returns
        -------
        CITestResult
            A frozen dataclass containing ``statistic``, ``p_value``,
            ``is_independent``, and ``conditioning_set``.

        Notes
        -----
        When the degrees of freedom ``n - |S| - 3`` are less than 1 the
        test cannot be performed and the method conservatively returns
        ``p_value = 1.0`` (i.e. independence is not rejected).
        """
        n: int = data.shape[0]
        s_size: int = len(conditioning_set)

        # Degrees of freedom for the Z statistic
        dof: int = n - s_size - 3
        if dof < 1:
            # Not enough observations; cannot reject independence
            return CITestResult(
                statistic=0.0,
                p_value=1.0,
                is_independent=True,
                conditioning_set=conditioning_set,
            )

        r: float = self._partial_correlation(x, y, conditioning_set, data)

        # Clamp |r| away from 1.0 to avoid log(0) or log(negative)
        r_clamped: float = max(
            -1.0 + _CORR_CLAMP_EPS, min(1.0 - _CORR_CLAMP_EPS, r)
        )

        z_stat: float = (
            0.5
            * math.log((1.0 + r_clamped) / (1.0 - r_clamped))
            * math.sqrt(dof)
        )
        p_value: PValue = float(2.0 * sp_stats.norm.sf(abs(z_stat)))

        return CITestResult(
            statistic=z_stat,
            p_value=p_value,
            is_independent=(p_value >= alpha),
            conditioning_set=conditioning_set,
        )

    # ------------------------------------------------------------------
    # Public API – batch test with multiple-testing correction
    # ------------------------------------------------------------------

    def test_multiple(
        self,
        pairs: List[Tuple[int, int, FrozenSet[int]]],
        data: DataMatrix,
        alpha: float = 0.05,
    ) -> List[CITestResult]:
        """Test multiple conditional independence hypotheses at once.

        This method runs ``test`` for each ``(x, y, conditioning_set)``
        triple in *pairs* and then applies the multiple-testing
        correction specified at construction time.

        Parameters
        ----------
        pairs : list of (int, int, FrozenSet[int])
            Each element is a tuple ``(x, y, conditioning_set)``
            specifying one conditional independence test.
        data : DataMatrix
            Observed data matrix of shape ``(N, p)``.
        alpha : float, optional
            Family-wise or false-discovery-rate significance level
            (interpretation depends on the correction method).

        Returns
        -------
        list of CITestResult
            One result per input pair.  The ``is_independent`` field
            reflects the corrected decision.

        Notes
        -----
        * ``"none"``  – no correction; each test uses the raw p-value.
        * ``"bonferroni"`` – each p-value is multiplied by the number
          of tests (capped at 1.0).  A test is independent iff its
          corrected p-value ≥ *alpha*.
        * ``"benjamini_hochberg"`` – the Benjamini–Hochberg step-up
          procedure is applied to control the false discovery rate at
          level *alpha*.
        """
        if not pairs:
            return []

        # Step 1: run all individual tests (raw p-values)
        raw_results: List[CITestResult] = [
            self.test(x, y, s, data, alpha=alpha) for x, y, s in pairs
        ]

        m: int = len(raw_results)

        # Step 2: apply correction
        if self.correction == "none":
            return raw_results

        if self.correction == "bonferroni":
            return self._correct_bonferroni(raw_results, m, alpha)

        # self.correction == "benjamini_hochberg"
        return self._correct_bh(raw_results, m, alpha)

    # ------------------------------------------------------------------
    # Multiple-testing correction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _correct_bonferroni(
        results: List[CITestResult],
        m: int,
        alpha: float,
    ) -> List[CITestResult]:
        """Apply Bonferroni correction to a list of test results.

        Each raw p-value is multiplied by the total number of tests
        *m* and capped at 1.0.  The independence decision is then
        re-evaluated against *alpha* using the corrected p-value.

        Parameters
        ----------
        results : list of CITestResult
            Raw (uncorrected) test results.
        m : int
            Total number of tests (used as the multiplier).
        alpha : float
            Significance level for the corrected decision.

        Returns
        -------
        list of CITestResult
            Results with corrected p-values and updated decisions.
        """
        corrected: List[CITestResult] = []
        for r in results:
            adj_p: float = min(r.p_value * m, 1.0)
            corrected.append(
                CITestResult(
                    statistic=r.statistic,
                    p_value=adj_p,
                    is_independent=(adj_p >= alpha),
                    conditioning_set=r.conditioning_set,
                )
            )
        return corrected

    @staticmethod
    def _correct_bh(
        results: List[CITestResult],
        m: int,
        alpha: float,
    ) -> List[CITestResult]:
        """Apply the Benjamini–Hochberg procedure.

        The BH step-up procedure controls the expected false-discovery
        rate at level *alpha*.  The algorithm:

        1. Sort the *m* raw p-values in ascending order.
        2. Find the largest rank *k* such that
           ``p_(k) ≤ (k / m) * alpha``.
        3. Reject (declare *dependent*) all hypotheses whose raw
           p-value is at most ``p_(k)``.

        Adjusted p-values are computed as
        ``p_adj(i) = min(m / rank(i) * p_raw(i), 1.0)`` with
        cumulative-minimum enforcement from the largest rank downward
        to ensure monotonicity.

        Parameters
        ----------
        results : list of CITestResult
            Raw (uncorrected) test results.
        m : int
            Total number of tests.
        alpha : float
            Desired FDR level.

        Returns
        -------
        list of CITestResult
            Results with BH-adjusted p-values and updated decisions.
        """
        # Extract raw p-values and sort indices
        raw_p: List[float] = [r.p_value for r in results]
        sorted_indices: List[int] = sorted(range(m), key=lambda i: raw_p[i])

        # Compute adjusted p-values (step-up)
        adjusted_p: List[float] = [0.0] * m
        cummin: float = 1.0

        # Walk from the largest rank down to ensure monotonicity
        for rank_minus1 in range(m - 1, -1, -1):
            idx: int = sorted_indices[rank_minus1]
            rank: int = rank_minus1 + 1  # 1-based rank
            adj: float = min(raw_p[idx] * m / rank, 1.0)
            cummin = min(cummin, adj)
            adjusted_p[idx] = cummin

        # Rebuild results with adjusted p-values and updated decisions
        corrected: List[CITestResult] = []
        for i, r in enumerate(results):
            corrected.append(
                CITestResult(
                    statistic=r.statistic,
                    p_value=adjusted_p[i],
                    is_independent=(adjusted_p[i] >= alpha),
                    conditioning_set=r.conditioning_set,
                )
            )
        return corrected

    # ------------------------------------------------------------------
    # Partial correlation – precision-matrix approach
    # ------------------------------------------------------------------

    @staticmethod
    def _partial_correlation(
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
    ) -> float:
        """Compute the sample partial correlation of X and Y given S.

        Uses the precision-matrix (inverse covariance) approach.  Let
        ``Σ`` be the sample covariance matrix of the columns
        ``{x, y} ∪ S`` and ``P = Σ⁻¹`` its inverse (the precision
        matrix).  Then::

            r_{XY|S} = -P[x, y] / sqrt(P[x, x] · P[y, y])

        A small ridge ``ε · I`` is added to ``Σ`` before inversion to
        improve numerical stability when columns are nearly collinear.

        Parameters
        ----------
        x : int
            Column index of the first variable in *data*.
        y : int
            Column index of the second variable in *data*.
        conditioning_set : FrozenSet[int]
            Column indices of the conditioning variables.
        data : DataMatrix
            Observed data matrix of shape ``(N, p)``.

        Returns
        -------
        float
            The estimated partial correlation in ``[-1, 1]``.  Returns
            ``0.0`` if the covariance matrix is singular even after
            regularisation, or if the diagonal precision entries are
            effectively zero.
        """
        indices: list[int] = sorted({x, y} | set(conditioning_set))
        sub: np.ndarray = data[:, indices]
        cov: np.ndarray = np.cov(sub, rowvar=False)

        # Handle single-variable edge case (cov is scalar)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)

        # Regularise for numerical stability
        cov += _REG_EPS * np.eye(cov.shape[0])

        try:
            precision: np.ndarray = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return 0.0

        # Map original column indices to sub-matrix positions
        idx_map: dict[int, int] = {v: i for i, v in enumerate(indices)}
        ix: int = idx_map[x]
        iy: int = idx_map[y]

        denom: float = math.sqrt(abs(precision[ix, ix] * precision[iy, iy]))
        if denom < _DENOM_TOL:
            return 0.0
        return float(-precision[ix, iy] / denom)

    # ------------------------------------------------------------------
    # Partial correlation – recursive approach
    # ------------------------------------------------------------------

    @staticmethod
    def _partial_correlation_recursive(
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
    ) -> float:
        """Compute partial correlation recursively.

        Uses the classical recursive formula that peels off one
        conditioning variable at a time::

            r_{XY|S} = (r_{XY|S'} - r_{XZ|S'} · r_{YZ|S'})
                       / sqrt((1 - r_{XZ|S'}²)(1 - r_{YZ|S'}²))

        where ``S' = S \\ {Z}`` for an arbitrary ``Z ∈ S``.  The base
        case (``S = ∅``) uses the ordinary Pearson correlation
        coefficient.

        Parameters
        ----------
        x : int
            Column index of the first variable.
        y : int
            Column index of the second variable.
        conditioning_set : FrozenSet[int]
            Column indices of conditioning variables.
        data : DataMatrix
            Observed data matrix of shape ``(N, p)``.

        Returns
        -------
        float
            The estimated partial correlation.  Returns ``0.0`` when
            numerical issues prevent a valid computation (e.g.
            denominator near zero).

        Notes
        -----
        This method has exponential time complexity in ``|S|`` and is
        intended primarily for small conditioning sets or for testing
        purposes.  For production use, prefer ``_partial_correlation``
        which uses a single matrix inversion.
        """
        # Base case: empty conditioning set → Pearson correlation
        if not conditioning_set:
            col_x: np.ndarray = data[:, x]
            col_y: np.ndarray = data[:, y]
            std_x: float = float(np.std(col_x, ddof=1))
            std_y: float = float(np.std(col_y, ddof=1))
            if std_x < _DENOM_TOL or std_y < _DENOM_TOL:
                return 0.0
            corr_matrix: np.ndarray = np.corrcoef(col_x, col_y)
            r: float = float(corr_matrix[0, 1])
            # Clamp in case of floating-point overshoot
            return max(-1.0, min(1.0, r))

        # Recursive case: peel off one variable
        z: int = next(iter(conditioning_set))
        s_prime: FrozenSet[int] = conditioning_set - {z}

        r_xy: float = FisherZTest._partial_correlation_recursive(
            x, y, s_prime, data
        )
        r_xz: float = FisherZTest._partial_correlation_recursive(
            x, z, s_prime, data
        )
        r_yz: float = FisherZTest._partial_correlation_recursive(
            y, z, s_prime, data
        )

        denom_sq: float = (1.0 - r_xz ** 2) * (1.0 - r_yz ** 2)
        if denom_sq <= 0.0:
            return 0.0
        denom: float = math.sqrt(denom_sq)
        if denom < _DENOM_TOL:
            return 0.0

        result: float = (r_xy - r_xz * r_yz) / denom
        # Clamp to valid range
        return max(-1.0, min(1.0, result))

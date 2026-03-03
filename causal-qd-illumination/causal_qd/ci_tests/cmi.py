"""Conditional mutual information test using k-nearest-neighbor estimation.

Implements the Kraskov–Stögbauer–Grassberger (KSG) estimators (both KSG1 and
KSG2 variants) for conditional mutual information I(X; Y | Z), paired with a
permutation test for statistical significance.

The two KSG estimators differ in bias/variance trade-off:

* **KSG1** has lower bias and is the default.  It uses *strict* inequality
  counts (points strictly inside the ε-ball).
* **KSG2** has lower variance but slightly higher bias.  It uses
  *non-strict* inequality counts (points on or inside the ε-ball boundary).

Both estimators use the Chebyshev (L∞) norm for distance computations, which
is standard in the KSG framework.

References
----------
.. [1] Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
       Estimating mutual information. *Physical Review E*, 69(6), 066138.
.. [2] Frenzel, S. & Pompe, B. (2007).
       Partial mutual information for coupling analysis of multivariate
       time series. *Physical Review Letters*, 99(20), 204101.
"""

from __future__ import annotations

import math
from typing import FrozenSet

import numpy as np
from scipy.special import digamma

from causal_qd.ci_tests.ci_base import CITest, CITestResult
from causal_qd.types import DataMatrix, PValue

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS_FLOOR: float = 1e-15
"""Minimum distance value to avoid log(0)."""


# ---------------------------------------------------------------------------
# Helper: vectorised Chebyshev k-NN
# ---------------------------------------------------------------------------


def _chebyshev_knn(points: np.ndarray, k: int) -> np.ndarray:
    """Find the k-th nearest-neighbour distance for every point under L∞.

    Parameters
    ----------
    points:
        Array of shape ``(n, d)`` – the point cloud.
    k:
        Number of nearest neighbours.  Must satisfy ``1 <= k < n``.

    Returns
    -------
    np.ndarray
        1-D array of length *n* where entry *i* is the Chebyshev distance
        from ``points[i]`` to its *k*-th nearest neighbour (excluding
        itself).

    Notes
    -----
    The implementation is fully vectorised over the inner distance
    computation for each query point.  For moderate *n* (a few thousand)
    this is faster than building a ``scipy.spatial.KDTree`` due to
    construction overhead.
    """
    n, _d = points.shape
    eps = np.empty(n, dtype=np.float64)

    for i in range(n):
        # Chebyshev distance from point i to every other point
        diffs = np.abs(points - points[i])          # (n, d)
        dists_i = np.max(diffs, axis=1)             # (n,)
        dists_i[i] = np.inf                          # exclude self
        # Partial sort is O(n) on average – much cheaper than full sort
        kth_dist = np.partition(dists_i, k - 1)[k - 1]
        eps[i] = max(kth_dist, _EPS_FLOOR)

    return eps


def _chebyshev_count_strict(
    points: np.ndarray,
    query: np.ndarray,
    eps: np.ndarray,
) -> np.ndarray:
    """Count points strictly within ε-ball for each query under L∞.

    For each query point *i*, counts the number of points *j* (j ≠ i)
    in ``points`` satisfying ``||points[j] - query[i]||_∞ < eps[i]``
    (strict inequality).

    Parameters
    ----------
    points:
        ``(n, d1)`` array of reference points.
    query:
        ``(n, d2)`` array of query points (same number of rows).
    eps:
        1-D array of length *n* giving the radius for each query.

    Returns
    -------
    np.ndarray
        1-D integer array of length *n* with the neighbour counts.
    """
    n = points.shape[0]
    counts = np.empty(n, dtype=np.int64)

    for i in range(n):
        diffs = np.abs(points - points[i])
        dists_i = np.max(diffs, axis=1)
        # Strict inequality; also exclude self (dist == 0)
        mask = dists_i < eps[i]
        mask[i] = False
        counts[i] = int(np.sum(mask))

    return counts


def _chebyshev_count_nonstrict(
    points: np.ndarray,
    query: np.ndarray,
    eps: np.ndarray,
) -> np.ndarray:
    """Count points within ε-ball (non-strict) for each query under L∞.

    Same as :func:`_chebyshev_count_strict` but uses ``<=`` instead of
    ``<``.

    Parameters
    ----------
    points:
        ``(n, d1)`` array of reference points.
    query:
        ``(n, d2)`` array of query points.
    eps:
        1-D array of radii.

    Returns
    -------
    np.ndarray
        1-D integer array of neighbour counts.
    """
    n = points.shape[0]
    counts = np.empty(n, dtype=np.int64)

    for i in range(n):
        diffs = np.abs(points - points[i])
        dists_i = np.max(diffs, axis=1)
        mask = dists_i <= eps[i]
        mask[i] = False
        counts[i] = int(np.sum(mask))

    return counts


# ===================================================================== #
#  Main class                                                           #
# ===================================================================== #


class ConditionalMutualInfoTest(CITest):
    """Conditional mutual information test  I(X; Y | Z).

    Uses the Kraskov–Stögbauer–Grassberger (KSG) k-nearest-neighbor
    estimator for (conditional) mutual information and a permutation test
    for the p-value.

    Two estimator variants are available:

    * ``"ksg1"`` (default) – lower bias, uses strict inequality counts:

      .. math::

          I(X;Y|Z) = \\psi(k)
              - \\langle \\psi(n_{xz}+1)
                       + \\psi(n_{yz}+1)
                       - \\psi(n_z+1) \\rangle

    * ``"ksg2"`` – lower variance, uses non-strict inequality counts:

      .. math::

          I(X;Y|Z) = \\psi(k) - \\frac{1}{k}
              - \\langle \\psi(n_{xz})
                       + \\psi(n_{yz})
                       - \\psi(n_z) \\rangle

    When the conditioning set *Z* is empty, both reduce to the
    unconditional MI estimators from [1]_.

    Parameters
    ----------
    k:
        Number of nearest neighbours for the KSG estimator.  Must be
        at least 1.  Larger values reduce variance at the cost of bias.
        Typical choices are 3–7.
    n_permutations:
        Number of random permutations used to compute the p-value
        under the null hypothesis of conditional independence.
    estimator:
        Which KSG variant to use: ``"ksg1"`` or ``"ksg2"``.
    seed:
        Seed for the random number generator used in the permutation
        test, ensuring reproducibility.

    Raises
    ------
    ValueError
        If *k* < 1 or *estimator* is not one of ``"ksg1"`` / ``"ksg2"``.

    References
    ----------
    .. [1] Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
           Estimating mutual information. *Physical Review E*, 69(6),
           066138.
    """

    _VALID_ESTIMATORS = ("ksg1", "ksg2")

    def __init__(
        self,
        k: int = 5,
        n_permutations: int = 500,
        estimator: str = "ksg1",
        seed: int = 0,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if estimator not in self._VALID_ESTIMATORS:
            raise ValueError(
                f"estimator must be one of {self._VALID_ESTIMATORS}, "
                f"got {estimator!r}"
            )
        self._k = k
        self._n_permutations = n_permutations
        self._estimator = estimator
        self._seed = seed

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    def test(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
        alpha: float = 0.05,
    ) -> CITestResult:
        """Perform the conditional independence test X ⊥ Y | Z.

        The test statistic is the estimated conditional mutual information
        I(X; Y | Z).  Statistical significance is assessed via a
        permutation test: the *Y* column is randomly permuted (breaking
        any X–Y dependence while preserving marginal distributions and
        the Z structure) and the CMI is re-estimated.

        Parameters
        ----------
        x:
            Column index of the first variable.
        y:
            Column index of the second variable.
        conditioning_set:
            Frozenset of column indices to condition on (may be empty).
        data:
            Observed data matrix of shape ``(N, p)`` with *N*
            observations and *p* variables.
        alpha:
            Significance level for the independence decision.

        Returns
        -------
        CITestResult
            Contains the observed CMI statistic, the permutation-based
            p-value, and the independence decision at level *alpha*.
        """
        n = data.shape[0]
        rng = np.random.default_rng(self._seed)

        observed_cmi = self._estimate_cmi(x, y, conditioning_set, data)

        # ---- Permutation test ----------------------------------------
        # Under H0 (X ⊥ Y | Z), permuting Y should not change the CMI
        # systematically.  We count how many permutation CMI values are
        # at least as large as the observed one.
        count = 0
        for _ in range(self._n_permutations):
            perm = rng.permutation(n)
            permuted_data = data.copy()
            permuted_data[:, y] = data[perm, y]
            perm_cmi = self._estimate_cmi(x, y, conditioning_set, permuted_data)
            if perm_cmi >= observed_cmi:
                count += 1

        # Pseudocount (+1 / +1) avoids zero p-values and corrects for
        # the inclusion of the observed statistic in the reference
        # distribution.
        p_value: PValue = float((count + 1) / (self._n_permutations + 1))

        return CITestResult(
            statistic=float(observed_cmi),
            p_value=p_value,
            is_independent=(p_value >= alpha),
            conditioning_set=conditioning_set,
        )

    # ------------------------------------------------------------------ #
    #  CMI dispatcher                                                    #
    # ------------------------------------------------------------------ #

    def _estimate_cmi(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
    ) -> float:
        """Dispatch to the selected KSG estimator.

        Parameters
        ----------
        x, y:
            Column indices.
        conditioning_set:
            Conditioning variable indices.
        data:
            ``(N, p)`` data matrix.

        Returns
        -------
        float
            Non-negative estimate of I(X; Y | Z).
        """
        if self._estimator == "ksg1":
            return self._ksg1_cmi(x, y, conditioning_set, data)
        return self._ksg2_cmi(x, y, conditioning_set, data)

    # ------------------------------------------------------------------ #
    #  KSG Estimator 1                                                   #
    # ------------------------------------------------------------------ #

    def _ksg1_cmi(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
    ) -> float:
        """KSG estimator 1 for conditional mutual information.

        When Z is non-empty::

            I(X;Y|Z) = ψ(k)
                - <ψ(n_xz + 1) + ψ(n_yz + 1) - ψ(n_z + 1)>

        For each sample *i* the procedure is:

        1. Find the k-th nearest-neighbour distance ε_i in the joint
           (X, Y, Z) space under the Chebyshev norm.
        2. Count ``n_xz`` = number of points *j* (j ≠ i) with
           ``||x_j - x_i||_∞ < ε_i`` **and** ``||z_j - z_i||_∞ < ε_i``.
        3. Analogously for ``n_yz`` and ``n_z``.

        When Z is empty, the formula simplifies to the unconditional MI::

            I(X;Y) = ψ(k) - 1/k + ψ(n) - <ψ(n_x + 1) + ψ(n_y + 1)>

        which is equivalent to the entropy-based form
        ``H(X) + H(Y) - H(X,Y)`` for KSG1.

        Parameters
        ----------
        x, y:
            Column indices.
        conditioning_set:
            Conditioning variable indices.
        data:
            Data matrix.

        Returns
        -------
        float
            Estimated CMI (clamped to zero from below).
        """
        s_list = sorted(conditioning_set)
        n = data.shape[0]
        k = min(self._k, n - 1)

        if not s_list:
            # --- Unconditional MI: I(X; Y) ---
            joint = data[:, [x, y]]
            eps = _chebyshev_knn(joint, k)

            x_data = data[:, [x]]
            y_data = data[:, [y]]

            n_x = _chebyshev_count_strict(x_data, x_data, eps)
            n_y = _chebyshev_count_strict(y_data, y_data, eps)

            mi = (
                digamma(k)
                - 1.0 / k
                + digamma(n)
                - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
            )
            return float(max(mi, 0.0))

        # --- Conditional MI: I(X; Y | Z) ---
        z_data = data[:, s_list]
        xz_data = data[:, [x] + s_list]
        yz_data = data[:, [y] + s_list]
        xyz_data = data[:, [x, y] + s_list]

        # Step 1: k-th NN distance in joint (X, Y, Z) space
        eps = _chebyshev_knn(xyz_data, k)

        # Step 2: marginal neighbour counts (strict inequality)
        n_xz = _chebyshev_count_strict(xz_data, xz_data, eps)
        n_yz = _chebyshev_count_strict(yz_data, yz_data, eps)
        n_z = _chebyshev_count_strict(z_data, z_data, eps)

        cmi = float(
            digamma(k)
            - np.mean(
                digamma(n_xz + 1)
                + digamma(n_yz + 1)
                - digamma(n_z + 1)
            )
        )
        return max(cmi, 0.0)

    # ------------------------------------------------------------------ #
    #  KSG Estimator 2                                                   #
    # ------------------------------------------------------------------ #

    def _ksg2_cmi(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
    ) -> float:
        """KSG estimator 2 for conditional mutual information.

        When Z is non-empty::

            I(X;Y|Z) = ψ(k) - 1/k
                - <ψ(n_xz) + ψ(n_yz) - ψ(n_z)>

        The neighbour counts use **non-strict** inequality (≤) instead
        of strict (<), yielding slightly higher counts and therefore
        lower variance at the expense of a small upward bias.

        When Z is empty::

            I(X;Y) = ψ(k) - 1/k + ψ(n) - <ψ(n_x) + ψ(n_y)>

        Parameters
        ----------
        x, y:
            Column indices.
        conditioning_set:
            Conditioning variable indices.
        data:
            Data matrix.

        Returns
        -------
        float
            Estimated CMI (clamped to zero from below).
        """
        s_list = sorted(conditioning_set)
        n = data.shape[0]
        k = min(self._k, n - 1)

        if not s_list:
            # --- Unconditional MI ---
            joint = data[:, [x, y]]
            eps = _chebyshev_knn(joint, k)

            x_data = data[:, [x]]
            y_data = data[:, [y]]

            n_x = _chebyshev_count_nonstrict(x_data, x_data, eps)
            n_y = _chebyshev_count_nonstrict(y_data, y_data, eps)

            # Floor counts to 1 to avoid digamma(0)
            n_x = np.maximum(n_x, 1)
            n_y = np.maximum(n_y, 1)

            mi = (
                digamma(k)
                - 1.0 / k
                + digamma(n)
                - np.mean(digamma(n_x) + digamma(n_y))
            )
            return float(max(mi, 0.0))

        # --- Conditional MI ---
        z_data = data[:, s_list]
        xz_data = data[:, [x] + s_list]
        yz_data = data[:, [y] + s_list]
        xyz_data = data[:, [x, y] + s_list]

        eps = _chebyshev_knn(xyz_data, k)

        n_xz = _chebyshev_count_nonstrict(xz_data, xz_data, eps)
        n_yz = _chebyshev_count_nonstrict(yz_data, yz_data, eps)
        n_z = _chebyshev_count_nonstrict(z_data, z_data, eps)

        # Floor to 1 to avoid digamma(0) or digamma of negative values
        n_xz = np.maximum(n_xz, 1)
        n_yz = np.maximum(n_yz, 1)
        n_z = np.maximum(n_z, 1)

        cmi = float(
            digamma(k)
            - 1.0 / k
            - np.mean(
                digamma(n_xz)
                + digamma(n_yz)
                - digamma(n_z)
            )
        )
        return max(cmi, 0.0)

    # ------------------------------------------------------------------ #
    #  KSG entropy (standalone, used for diagnostics / alternative path) #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ksg_entropy(points: np.ndarray, k: int) -> float:
        """KSG-style differential entropy estimate via k-NN distances.

        .. math::

            H(X) \\approx \\psi(n) - \\psi(k) + d \\cdot \\langle \\ln(2 \\varepsilon_i) \\rangle

        where :math:`\\varepsilon_i` is the Chebyshev distance from point
        *i* to its *k*-th nearest neighbour and *d* is the
        dimensionality of *X*.

        This method is kept for compatibility and for use in entropy-
        difference formulations of MI when needed.

        Parameters
        ----------
        points:
            ``(n, d)`` data array.
        k:
            Number of nearest neighbours.

        Returns
        -------
        float
            Entropy estimate in nats.
        """
        n, d = points.shape
        if n <= k:
            return 0.0

        eps = _chebyshev_knn(points, k)

        log_eps_sum = float(np.sum(np.log(2.0 * eps)))

        return float(digamma(n) - digamma(k) + d * log_eps_sum / n)

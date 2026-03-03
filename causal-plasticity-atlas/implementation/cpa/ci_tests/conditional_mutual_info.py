"""CMI-based conditional independence tests.

Implements the Kraskov–Stögbauer–Grassberger (KSG) k-nearest-neighbour
estimator for mutual information and conditional mutual information,
plus a permutation-based significance test.

The KSG algorithm estimates mutual information without explicit density
estimation, using only the distances to the k-th nearest neighbour in
joint and marginal spaces.  We use :class:`scipy.spatial.KDTree` for
efficient neighbour searches.

References
----------
- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
  Estimating mutual information. *Physical Review E*, 69(6).
- Frenzel, S., & Pompe, B. (2007).
  Partial mutual information for coupling analysis of multivariate
  time series. *Physical Review Letters*, 99(20).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
from scipy.special import digamma


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CMITestResult:
    """Result of a CMI-based CI test."""

    statistic: float
    p_value: float
    independent: bool
    conditioning_set: Set[int] = field(default_factory=set)
    method: str = "cmi_ksg"


# ---------------------------------------------------------------------------
# KSG Estimator
# ---------------------------------------------------------------------------

class KSGEstimator:
    """KSG estimator for mutual and conditional mutual information.

    Implements both *Algorithm 1* (MI) and the Frenzel–Pompe extension
    for conditional mutual information.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbours for the density estimate.
    noise_level : float
        Small uniform noise added to break ties (default 1e-10).
    """

    def __init__(
        self,
        k_neighbors: int = 7,
        noise_level: float = 1e-10,
    ) -> None:
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1; got {k_neighbors}")
        self.k_neighbors = k_neighbors
        self.noise_level = noise_level

    # -- public interface ---------------------------------------------------

    def estimate_mi(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> float:
        """Estimate mutual information I(X; Y) using KSG Algorithm 1.

        Parameters
        ----------
        x : ndarray of shape (n,) or (n, d1)
        y : ndarray of shape (n,) or (n, d2)

        Returns
        -------
        float
            Estimated MI in nats.  Can be slightly negative due to bias.
        """
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        n = x.shape[0]
        if n < self.k_neighbors + 1:
            return 0.0

        x, y = self._add_noise(x), self._add_noise(y)
        return self._ksg_mi(x, y)

    def estimate_cmi(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
    ) -> float:
        """Estimate conditional mutual information I(X; Y | Z).

        Uses the Frenzel–Pompe (2007) extension of KSG:

        I(X;Y|Z) = ψ(k) - <ψ(n_xz + 1) + ψ(n_yz + 1) - ψ(n_z + 1)>

        Parameters
        ----------
        x : ndarray of shape (n,) or (n, d1)
        y : ndarray of shape (n,) or (n, d2)
        z : ndarray of shape (n,) or (n, d3)

        Returns
        -------
        float
            Estimated CMI in nats.
        """
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        z = self._ensure_2d(z)
        n = x.shape[0]
        if n < self.k_neighbors + 1:
            return 0.0

        x, y, z = self._add_noise(x), self._add_noise(y), self._add_noise(z)
        return self._ksg_cmi(x, y, z)

    # -- KSG MI (Algorithm 1) ----------------------------------------------

    def _ksg_mi(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> float:
        """Core KSG Algorithm 1 for MI estimation.

        I(X;Y) = ψ(k) - <ψ(n_x + 1) + ψ(n_y + 1)> + ψ(n)

        where n_x is the number of points with ||x_i - x_j||_∞ < ε_i
        in the X-marginal, and ε_i is the Chebyshev distance to the k-th
        nearest neighbour in the joint (X, Y) space.
        """
        n = x.shape[0]
        k = self.k_neighbors

        joint = np.hstack([x, y])
        tree_joint = KDTree(joint)
        tree_x = KDTree(x)
        tree_y = KDTree(y)

        # Query k+1 neighbours (including the point itself)
        dists_joint, _ = tree_joint.query(joint, k=k + 1, p=np.inf)
        # eps_i = distance to k-th neighbour (index k since 0-indexed includes self)
        eps = dists_joint[:, k]

        # Count points strictly within eps in marginals
        # We subtract 1 to exclude the point itself
        n_x = np.array([
            tree_x.query_ball_point(x[i], eps[i] - 1e-15, p=np.inf, return_length=True) - 1
            for i in range(n)
        ], dtype=np.float64)
        n_y = np.array([
            tree_y.query_ball_point(y[i], eps[i] - 1e-15, p=np.inf, return_length=True) - 1
            for i in range(n)
        ], dtype=np.float64)

        # Clamp to at least 1 to avoid digamma(0)
        n_x = np.maximum(n_x, 1)
        n_y = np.maximum(n_y, 1)

        mi = (
            digamma(k)
            - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
            + digamma(n)
        )
        return float(mi)

    # -- KSG CMI (Frenzel–Pompe) -------------------------------------------

    def _ksg_cmi(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
    ) -> float:
        """Frenzel–Pompe CMI estimator.

        I(X;Y|Z) = ψ(k) - <ψ(n_xz + 1) + ψ(n_yz + 1) - ψ(n_z + 1)>

        where counts are determined by the Chebyshev distance ε_i to the
        k-th neighbour in the joint (X, Y, Z) space.
        """
        n = x.shape[0]
        k = self.k_neighbors

        xyz = np.hstack([x, y, z])
        xz = np.hstack([x, z])
        yz = np.hstack([y, z])

        tree_xyz = KDTree(xyz)
        tree_xz = KDTree(xz)
        tree_yz = KDTree(yz)
        tree_z = KDTree(z)

        dists_xyz, _ = tree_xyz.query(xyz, k=k + 1, p=np.inf)
        eps = dists_xyz[:, k]

        n_xz = np.empty(n, dtype=np.float64)
        n_yz = np.empty(n, dtype=np.float64)
        n_z = np.empty(n, dtype=np.float64)

        for i in range(n):
            r = eps[i] - 1e-15
            if r < 0:
                r = 0.0
            n_xz[i] = tree_xz.query_ball_point(xz[i], r, p=np.inf, return_length=True) - 1
            n_yz[i] = tree_yz.query_ball_point(yz[i], r, p=np.inf, return_length=True) - 1
            n_z[i] = tree_z.query_ball_point(z[i], r, p=np.inf, return_length=True) - 1

        n_xz = np.maximum(n_xz, 1)
        n_yz = np.maximum(n_yz, 1)
        n_z = np.maximum(n_z, 1)

        cmi = (
            digamma(k)
            - np.mean(
                digamma(n_xz + 1)
                + digamma(n_yz + 1)
                - digamma(n_z + 1)
            )
        )
        return float(cmi)

    # -- KNN distance helper -----------------------------------------------

    def _knn_distances(
        self,
        data: NDArray[np.float64],
        k: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Return the distance to each point's k-th nearest neighbour.

        Parameters
        ----------
        data : ndarray of shape (n, d)
        k : int or None
            Number of neighbours; defaults to ``self.k_neighbors``.

        Returns
        -------
        ndarray of shape (n,)
            Chebyshev distances to the k-th neighbour.
        """
        if k is None:
            k = self.k_neighbors
        data = self._ensure_2d(data)
        tree = KDTree(data)
        dists, _ = tree.query(data, k=k + 1, p=np.inf)
        return dists[:, k]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _ensure_2d(arr: NDArray) -> NDArray[np.float64]:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _add_noise(self, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        """Add tiny uniform noise to break ties."""
        if self.noise_level > 0:
            rng = np.random.default_rng(seed=0)
            arr = arr + rng.uniform(
                -self.noise_level, self.noise_level, size=arr.shape
            )
        return arr


# ---------------------------------------------------------------------------
# CMI-based CI test
# ---------------------------------------------------------------------------

class CMITest:
    """Conditional independence test based on CMI estimation.

    Uses the KSG estimator to compute I(X; Y | Z) and a permutation
    test to assess statistical significance.

    Parameters
    ----------
    alpha : float
        Significance level.
    estimator : KSGEstimator | None
        CMI estimator instance; built with defaults if ``None``.
    n_permutations : int
        Number of permutations for the null distribution.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        estimator: KSGEstimator | None = None,
        n_permutations: int = 500,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = alpha
        self.estimator = estimator or KSGEstimator()
        self.n_permutations = n_permutations

    def test(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        conditioning_set: Set[int] | None = None,
    ) -> Tuple[float, float]:
        """Run the CMI-based test.

        Parameters
        ----------
        data : ndarray of shape (n, p)
        x, y : int
            Column indices to test.
        conditioning_set : set of int or None
            Columns to condition on.

        Returns
        -------
        statistic : float
            Estimated CMI (or MI when no conditioning set).
        pvalue : float
            Permutation-based p-value.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("data must be 2-D")
        n, p = data.shape
        z = conditioning_set if conditioning_set is not None else set()

        all_idx = {x, y} | z
        if any(i < 0 or i >= p for i in all_idx):
            raise IndexError(f"Variable index out of range [0, {p})")

        X_col = data[:, x]
        Y_col = data[:, y]

        if len(z) == 0:
            stat = self.estimator.estimate_mi(X_col, Y_col)
            pval = self._permutation_test_mi(stat, X_col, Y_col)
        else:
            Z_cols = data[:, sorted(z)]
            stat = self.estimator.estimate_cmi(X_col, Y_col, Z_cols)
            pval = self.permutation_test(data, x, y, z, self.n_permutations)

        return (float(stat), float(pval))

    def test_full(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        conditioning_set: Set[int] | None = None,
    ) -> CMITestResult:
        """Like :meth:`test` but returns a :class:`CMITestResult`."""
        z = conditioning_set if conditioning_set is not None else set()
        stat, pval = self.test(data, x, y, z)
        return CMITestResult(
            statistic=stat,
            p_value=pval,
            independent=(pval >= self.alpha),
            conditioning_set=set(z),
            method="cmi_ksg",
        )

    def permutation_test(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        z: Set[int],
        n_perms: int,
    ) -> float:
        """Compute a permutation p-value for the CMI statistic.

        Permutes the X column (breaking its dependence on Y while
        preserving the Z-marginal structure) and recomputes CMI.

        Parameters
        ----------
        data : ndarray of shape (n, p)
        x, y : int
        z : set of int
        n_perms : int

        Returns
        -------
        float
            Permutation-based p-value.
        """
        n = data.shape[0]
        X_col = data[:, x]
        Y_col = data[:, y]
        Z_cols = data[:, sorted(z)] if len(z) > 0 else None

        if Z_cols is not None:
            observed = self.estimator.estimate_cmi(X_col, Y_col, Z_cols)
        else:
            observed = self.estimator.estimate_mi(X_col, Y_col)

        rng = np.random.default_rng(seed=42)
        count = 0
        for _ in range(n_perms):
            perm = rng.permutation(n)
            X_perm = X_col[perm]
            if Z_cols is not None:
                null_stat = self.estimator.estimate_cmi(X_perm, Y_col, Z_cols)
            else:
                null_stat = self.estimator.estimate_mi(X_perm, Y_col)
            if null_stat >= observed:
                count += 1

        return (count + 1) / (n_perms + 1)

    def _permutation_test_mi(
        self,
        observed: float,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
    ) -> float:
        """Permutation test for MI (no conditioning)."""
        n = len(X)
        rng = np.random.default_rng(seed=42)
        count = 0
        for _ in range(self.n_permutations):
            perm = rng.permutation(n)
            null_stat = self.estimator.estimate_mi(X[perm], Y)
            if null_stat >= observed:
                count += 1
        return (count + 1) / (self.n_permutations + 1)

    # -- convenience --------------------------------------------------------

    def compute_mi(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
    ) -> float:
        """Compute mutual information I(X; Y).

        Parameters
        ----------
        data : ndarray of shape (n, p)
        x, y : int

        Returns
        -------
        float
        """
        data = np.asarray(data, dtype=np.float64)
        return self.estimator.estimate_mi(data[:, x], data[:, y])

    def compute_cmi(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        z: Set[int],
    ) -> float:
        """Compute conditional mutual information I(X; Y | Z).

        Parameters
        ----------
        data : ndarray of shape (n, p)
        x, y : int
        z : set of int

        Returns
        -------
        float
        """
        data = np.asarray(data, dtype=np.float64)
        if len(z) == 0:
            return self.compute_mi(data, x, y)
        Z_cols = data[:, sorted(z)]
        return self.estimator.estimate_cmi(data[:, x], data[:, y], Z_cols)

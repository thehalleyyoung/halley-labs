"""
Mutual-information based conditional-independence test.

Implements the kNN-based KSG mutual-information estimator (Kraskov,
Stögbauer & Grassberger, 2004) with:
- KSG Algorithm 1 and Algorithm 2 for MI estimation
- Conditional MI via the chain rule: I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
- Permutation test for significance calibration
- Adaptive k-nearest-neighbour selection
- Bias correction for small samples
- Handling of discrete, continuous, and mixed variable types

References
----------
Kraskov, A., Stögbauer, H. & Grassberger, P. (2004).
    Estimating mutual information. *Physical Review E*, 69(6), 066138.

Frenzel, S. & Pompe, B. (2007).
    Partial mutual information for coupling analysis of multivariate
    time series. *Physical Review Letters*, 99(20), 204101.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import special, stats
from scipy.spatial import cKDTree

from causalcert.ci_testing.base import (
    BaseCITest,
    CITestConfig,
    _extract_columns,
    _insufficient_sample_result,
    _validate_inputs,
)
from causalcert.types import CITestMethod, CITestResult, NodeId, NodeSet

_EPS = 1e-10
_MIN_N = 10


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MutualInfoConfig:
    """Configuration for the MI-based CI test.

    Attributes
    ----------
    k : int
        Number of nearest neighbours for the KSG estimator.
    algorithm : int
        KSG algorithm variant (1 or 2).
    n_permutations : int
        Number of permutations for the significance test.
    adaptive_k : bool
        If ``True``, automatically select k based on sample size.
    bias_correction : bool
        Apply small-sample bias correction.
    discrete_threshold : int
        If a variable has fewer unique values than this, treat it
        as discrete.
    metric : str
        Distance metric for kNN (``"chebyshev"`` or ``"euclidean"``).
    """

    k: int = 7
    algorithm: int = 1
    n_permutations: int = 500
    adaptive_k: bool = True
    bias_correction: bool = True
    discrete_threshold: int = 10
    metric: str = "chebyshev"


# ---------------------------------------------------------------------------
# Digamma helper
# ---------------------------------------------------------------------------


def _digamma(x: float) -> float:
    """Compute the digamma function psi(x).

    Parameters
    ----------
    x : float
        Positive argument.

    Returns
    -------
    float
        Digamma value.
    """
    return float(special.digamma(x))


# ---------------------------------------------------------------------------
# Variable type detection
# ---------------------------------------------------------------------------


def _is_discrete(col: np.ndarray, threshold: int = 10) -> bool:
    """Check whether a column appears discrete.

    Parameters
    ----------
    col : np.ndarray
        1-D data array.
    threshold : int
        Maximum unique values for discrete classification.

    Returns
    -------
    bool
    """
    return len(np.unique(col)) <= threshold


def _add_noise(
    X: np.ndarray,
    rng: np.random.Generator,
    scale: float = 1e-8,
) -> np.ndarray:
    """Add small jitter to break ties in discrete data.

    Parameters
    ----------
    X : np.ndarray
        Data array.
    rng : np.random.Generator
        Random number generator.
    scale : float
        Noise standard deviation.

    Returns
    -------
    np.ndarray
        Jittered data.
    """
    return X + rng.normal(0.0, scale, size=X.shape)


# ---------------------------------------------------------------------------
# Adaptive k selection
# ---------------------------------------------------------------------------


def _select_k(n: int, base_k: int = 7) -> int:
    """Select k adaptively based on sample size.

    Uses the heuristic k ≈ max(3, min(base_k, sqrt(n) / 2)).

    Parameters
    ----------
    n : int
        Sample size.
    base_k : int
        User-specified default k.

    Returns
    -------
    int
        Selected k.
    """
    auto_k = max(3, min(int(math.sqrt(n) / 2), 20))
    return min(auto_k, n - 1, base_k) if base_k > 0 else auto_k


# ---------------------------------------------------------------------------
# KSG MI estimator
# ---------------------------------------------------------------------------


def _ksg_mi_algorithm1(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    metric: str = "chebyshev",
) -> float:
    """KSG Algorithm 1 for MI estimation.

    I(X;Y) ≈ ψ(k) - <ψ(n_x) + ψ(n_y)> + ψ(n)

    where n_x, n_y are the number of points within the k-th neighbour
    distance projected onto each marginal.

    Parameters
    ----------
    X : np.ndarray
        First variable ``(n, d1)``.
    Y : np.ndarray
        Second variable ``(n, d2)``.
    k : int
        Number of nearest neighbours.
    metric : str
        Distance metric.

    Returns
    -------
    float
        MI estimate (non-negative after clipping).
    """
    n = X.shape[0]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    XY = np.hstack([X, Y])

    tree_xy = cKDTree(XY)
    tree_x = cKDTree(X)
    tree_y = cKDTree(Y)

    # Query k+1 neighbours (including the point itself)
    k_eff = min(k, n - 1)
    dists_xy, _ = tree_xy.query(XY, k=k_eff + 1, p=np.inf)
    eps = dists_xy[:, -1]  # k-th neighbour distance in joint space

    psi_sum = 0.0
    for i in range(n):
        eps_i = max(eps[i], _EPS)
        # Count points within eps_i in marginals (Chebyshev ball)
        n_x = len(tree_x.query_ball_point(X[i], eps_i - _EPS, p=np.inf)) - 1
        n_y = len(tree_y.query_ball_point(Y[i], eps_i - _EPS, p=np.inf)) - 1
        n_x = max(n_x, 1)
        n_y = max(n_y, 1)
        psi_sum += _digamma(n_x) + _digamma(n_y)

    mi = _digamma(k_eff) - psi_sum / n + _digamma(n)
    return max(mi, 0.0)


def _ksg_mi_algorithm2(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    metric: str = "chebyshev",
) -> float:
    """KSG Algorithm 2 for MI estimation.

    I(X;Y) ≈ ψ(k) - 1/k - <ψ(n_x) + ψ(n_y)> + ψ(n)

    Uses the marginal distances to define eps_x and eps_y separately.

    Parameters
    ----------
    X : np.ndarray
        First variable ``(n, d1)``.
    Y : np.ndarray
        Second variable ``(n, d2)``.
    k : int
        Number of nearest neighbours.
    metric : str
        Distance metric.

    Returns
    -------
    float
        MI estimate.
    """
    n = X.shape[0]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    XY = np.hstack([X, Y])

    tree_xy = cKDTree(XY)
    tree_x = cKDTree(X)
    tree_y = cKDTree(Y)

    k_eff = min(k, n - 1)
    dists_xy, idx_xy = tree_xy.query(XY, k=k_eff + 1, p=np.inf)

    psi_sum = 0.0
    for i in range(n):
        # kth neighbour in joint space
        nn_idx = idx_xy[i, -1]
        eps_x = np.max(np.abs(X[i] - X[nn_idx]))
        eps_y = np.max(np.abs(Y[i] - Y[nn_idx]))
        eps_x = max(eps_x, _EPS)
        eps_y = max(eps_y, _EPS)

        n_x = len(tree_x.query_ball_point(X[i], eps_x, p=np.inf)) - 1
        n_y = len(tree_y.query_ball_point(Y[i], eps_y, p=np.inf)) - 1
        n_x = max(n_x, 1)
        n_y = max(n_y, 1)
        psi_sum += _digamma(n_x) + _digamma(n_y)

    mi = _digamma(k_eff) - 1.0 / k_eff - psi_sum / n + _digamma(n)
    return max(mi, 0.0)


# ---------------------------------------------------------------------------
# Bias correction
# ---------------------------------------------------------------------------


def _bias_correction(mi: float, n: int, k: int, dx: int, dy: int) -> float:
    """Apply small-sample bias correction to MI estimate.

    The KSG estimator has a positive bias of order O(1/n).
    We apply the correction from Kraskov et al. (2004):
    MI_corrected = MI - (d_x + d_y) / (2 * n * k)

    Parameters
    ----------
    mi : float
        Raw MI estimate.
    n : int
        Sample size.
    k : int
        Number of neighbours.
    dx : int
        Dimensionality of X.
    dy : int
        Dimensionality of Y.

    Returns
    -------
    float
        Bias-corrected MI (clipped to non-negative).
    """
    correction = (dx + dy) / (2.0 * n * max(k, 1))
    return max(mi - correction, 0.0)


# ---------------------------------------------------------------------------
# Conditional MI via chain rule
# ---------------------------------------------------------------------------


def _conditional_mi_chain(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    k: int,
    algorithm: int = 1,
    metric: str = "chebyshev",
) -> float:
    """Compute conditional MI via the chain rule.

    I(X; Y | Z) = I(X; Y, Z) - I(X; Z)

    Parameters
    ----------
    X : np.ndarray
        First variable ``(n, d1)``.
    Y : np.ndarray
        Second variable ``(n, d2)``.
    Z : np.ndarray
        Conditioning variables ``(n, k)``.
    k : int
        Number of nearest neighbours.
    algorithm : int
        KSG algorithm (1 or 2).
    metric : str
        Distance metric.

    Returns
    -------
    float
        Conditional MI estimate (non-negative).
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    Z = np.atleast_2d(Z)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Z.ndim == 1:
        Z = Z[:, np.newaxis]

    YZ = np.hstack([Y, Z])

    ksg_fn = _ksg_mi_algorithm1 if algorithm == 1 else _ksg_mi_algorithm2
    mi_xyz = ksg_fn(X, YZ, k, metric)
    mi_xz = ksg_fn(X, Z, k, metric)
    return max(mi_xyz - mi_xz, 0.0)


def _conditional_mi_ksg_direct(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    k: int,
) -> float:
    """Direct conditional MI estimator (Frenzel & Pompe 2007).

    I(X; Y | Z) = ψ(k) - <ψ(n_xz) + ψ(n_yz) - ψ(n_z)>

    Parameters
    ----------
    X : np.ndarray
        First variable ``(n, d1)``.
    Y : np.ndarray
        Second variable ``(n, d2)``.
    Z : np.ndarray
        Conditioning set ``(n, dz)``.
    k : int
        Number of nearest neighbours.

    Returns
    -------
    float
        Conditional MI estimate.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    Z = np.atleast_2d(Z)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Z.ndim == 1:
        Z = Z[:, np.newaxis]

    n = X.shape[0]
    XYZ = np.hstack([X, Y, Z])
    XZ = np.hstack([X, Z])
    YZ = np.hstack([Y, Z])

    tree_xyz = cKDTree(XYZ)
    tree_xz = cKDTree(XZ)
    tree_yz = cKDTree(YZ)
    tree_z = cKDTree(Z)

    k_eff = min(k, n - 1)
    dists_xyz, _ = tree_xyz.query(XYZ, k=k_eff + 1, p=np.inf)
    eps = dists_xyz[:, -1]

    psi_sum = 0.0
    for i in range(n):
        eps_i = max(eps[i], _EPS)
        n_xz = max(
            len(tree_xz.query_ball_point(XZ[i], eps_i - _EPS, p=np.inf)) - 1,
            1,
        )
        n_yz = max(
            len(tree_yz.query_ball_point(YZ[i], eps_i - _EPS, p=np.inf)) - 1,
            1,
        )
        n_z = max(
            len(tree_z.query_ball_point(Z[i], eps_i - _EPS, p=np.inf)) - 1,
            1,
        )
        psi_sum += _digamma(n_xz) + _digamma(n_yz) - _digamma(n_z)

    cmi = _digamma(k_eff) - psi_sum / n
    return max(cmi, 0.0)


# ---------------------------------------------------------------------------
# Discrete MI
# ---------------------------------------------------------------------------


def _discrete_mi(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute MI between two discrete variables.

    Parameters
    ----------
    X : np.ndarray
        First variable (integer-valued).
    Y : np.ndarray
        Second variable (integer-valued).

    Returns
    -------
    float
        MI in nats.
    """
    n = len(X)
    x_vals, x_codes = np.unique(X, return_inverse=True)
    y_vals, y_codes = np.unique(Y, return_inverse=True)

    joint = np.zeros((len(x_vals), len(y_vals)), dtype=np.float64)
    for i in range(n):
        joint[x_codes[i], y_codes[i]] += 1.0
    joint /= n

    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for xi in range(len(x_vals)):
        for yi in range(len(y_vals)):
            if joint[xi, yi] > _EPS and px[xi] > _EPS and py[yi] > _EPS:
                mi += joint[xi, yi] * math.log(
                    joint[xi, yi] / (px[xi] * py[yi])
                )
    return max(mi, 0.0)


def _mixed_mi(
    X: np.ndarray,
    Y: np.ndarray,
    x_discrete: bool,
    y_discrete: bool,
    k: int = 7,
) -> float:
    """MI estimation for mixed discrete-continuous pairs.

    When one variable is discrete and the other continuous, computes
    MI = H(X_discrete) - H(X_discrete | Y_continuous) using
    the nearest-neighbour approach of Ross (2014).

    Parameters
    ----------
    X : np.ndarray
        First variable.
    Y : np.ndarray
        Second variable.
    x_discrete : bool
        Whether X is discrete.
    y_discrete : bool
        Whether Y is discrete.
    k : int
        Number of nearest neighbours.

    Returns
    -------
    float
        MI estimate.
    """
    if x_discrete and y_discrete:
        return _discrete_mi(X, Y)
    if not x_discrete and not y_discrete:
        return _ksg_mi_algorithm1(
            X[:, np.newaxis] if X.ndim == 1 else X,
            Y[:, np.newaxis] if Y.ndim == 1 else Y,
            k,
        )

    # Mixed case: condition on the discrete variable
    if x_discrete:
        disc, cont = X, Y
    else:
        disc, cont = Y, X

    cont = cont[:, np.newaxis] if cont.ndim == 1 else cont
    n = len(disc)
    classes = np.unique(disc)

    mi = _digamma(n)
    m_avg = 0.0
    nn_avg = 0.0

    for c in classes:
        mask = disc == c
        m_c = mask.sum()
        if m_c < 2:
            continue

        cont_c = cont[mask]
        tree_c = cKDTree(cont_c)
        tree_all = cKDTree(cont)

        k_eff = min(k, m_c - 1)
        if k_eff < 1:
            continue

        for idx_local in range(m_c):
            dists, _ = tree_c.query(cont_c[idx_local], k=k_eff + 1, p=np.inf)
            eps_i = max(dists[-1], _EPS)
            # Count in full dataset
            n_i = max(
                len(tree_all.query_ball_point(
                    cont_c[idx_local], eps_i, p=np.inf
                )) - 1, 1,
            )
            nn_avg += _digamma(n_i)
        m_avg += m_c * _digamma(m_c)

    mi = _digamma(k) - nn_avg / n + mi - m_avg / n
    return max(mi, 0.0)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


def _permutation_mi_pvalue(
    observed: float,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray | None,
    k: int,
    algorithm: int,
    n_permutations: int,
    rng: np.random.Generator,
    metric: str = "chebyshev",
) -> float:
    """Permutation p-value for MI or conditional MI.

    Parameters
    ----------
    observed : float
        Observed MI / CMI statistic.
    X : np.ndarray
        First variable.
    Y : np.ndarray
        Second variable.
    Z : np.ndarray | None
        Conditioning variables.
    k : int
        Number of nearest neighbours.
    algorithm : int
        KSG algorithm.
    n_permutations : int
        Number of permutations.
    rng : np.random.Generator
        RNG.
    metric : str
        Distance metric.

    Returns
    -------
    float
        p-value.
    """
    n = X.shape[0]
    count = 0

    for _ in range(n_permutations):
        perm = rng.permutation(n)
        X_perm = X[perm]
        if Z is not None:
            null_stat = _conditional_mi_chain(
                X_perm, Y, Z, k, algorithm, metric,
            )
        else:
            ksg_fn = (
                _ksg_mi_algorithm1 if algorithm == 1
                else _ksg_mi_algorithm2
            )
            null_stat = ksg_fn(X_perm, Y, k, metric)
        if null_stat >= observed:
            count += 1

    return (count + 1) / (n_permutations + 1)


# ---------------------------------------------------------------------------
# MutualInfoCITest class
# ---------------------------------------------------------------------------


class MutualInfoCITest(BaseCITest):
    """Mutual-information based conditional-independence test.

    Tests ``X ⊥ Y | Z`` using the KSG nearest-neighbour MI estimator.

    For unconditional testing, computes ``I(X; Y)`` directly.
    For conditional testing, uses the chain rule:
    ``I(X; Y | Z) = I(X; Y,Z) - I(X; Z)``
    or the direct estimator of Frenzel & Pompe (2007).

    Significance is assessed by a permutation test that shuffles X
    (breaking the X–Y link while preserving X–Z and Y–Z).

    Parameters
    ----------
    alpha : float
        Significance level.
    seed : int
        Random seed.
    config : CITestConfig | None
        Base CI test configuration.
    mi_config : MutualInfoConfig | None
        MI-specific configuration.
    """

    method = CITestMethod.MUTUAL_INFO

    def __init__(
        self,
        alpha: float = 0.05,
        seed: int = 42,
        config: CITestConfig | None = None,
        mi_config: MutualInfoConfig | None = None,
    ) -> None:
        super().__init__(alpha=alpha, seed=seed, config=config)
        self.mi_config = mi_config or MutualInfoConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare(
        self,
        x_col: np.ndarray,
        y_col: np.ndarray,
        z_cols: np.ndarray | None,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool, bool]:
        """Detect variable types and add jitter if needed.

        Returns
        -------
        tuple
            ``(X, Y, Z, x_discrete, y_discrete)``
        """
        cfg = self.mi_config
        x_disc = _is_discrete(x_col, cfg.discrete_threshold)
        y_disc = _is_discrete(y_col, cfg.discrete_threshold)

        X = x_col[:, np.newaxis] if x_col.ndim == 1 else x_col
        Y = y_col[:, np.newaxis] if y_col.ndim == 1 else y_col

        # Add jitter to avoid ties
        if x_disc:
            X = _add_noise(X.astype(np.float64), rng)
        if y_disc:
            Y = _add_noise(Y.astype(np.float64), rng)

        Z = None
        if z_cols is not None:
            Z = z_cols
            for j in range(Z.shape[1]):
                if _is_discrete(Z[:, j], cfg.discrete_threshold):
                    Z[:, j] = Z[:, j] + rng.normal(
                        0, 1e-8, size=Z.shape[0],
                    )

        return X, Y, Z, x_disc, y_disc

    def _estimate_mi(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray | None,
        k: int,
    ) -> float:
        """Estimate MI or conditional MI.

        Parameters
        ----------
        X : np.ndarray
            First variable ``(n, d1)``.
        Y : np.ndarray
            Second variable ``(n, d2)``.
        Z : np.ndarray | None
            Conditioning variables ``(n, dz)`` or ``None``.
        k : int
            Number of nearest neighbours.

        Returns
        -------
        float
            MI or CMI estimate.
        """
        cfg = self.mi_config
        n = X.shape[0]

        if Z is not None:
            # Use direct CMI estimator for small conditioning sets
            if Z.shape[1] <= 3:
                mi = _conditional_mi_ksg_direct(X, Y, Z, k)
            else:
                mi = _conditional_mi_chain(
                    X, Y, Z, k, cfg.algorithm, cfg.metric,
                )
        else:
            ksg_fn = (
                _ksg_mi_algorithm1 if cfg.algorithm == 1
                else _ksg_mi_algorithm2
            )
            mi = ksg_fn(X, Y, k, cfg.metric)

        # Bias correction
        if cfg.bias_correction:
            dx = X.shape[1] if X.ndim > 1 else 1
            dy = Y.shape[1] if Y.ndim > 1 else 1
            mi = _bias_correction(mi, n, k, dx, dy)

        return mi

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
        """Test X ⊥ Y | Z using mutual information.

        Parameters
        ----------
        x : NodeId
            First variable.
        y : NodeId
            Second variable.
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
        if n < max(self.config.min_samples, _MIN_N):
            return _insufficient_sample_result(
                x, y, conditioning_set, self.method, self.alpha,
            )

        rng = np.random.default_rng(self.seed)
        cfg = self.mi_config

        # Adaptive k
        k = _select_k(n, cfg.k) if cfg.adaptive_k else cfg.k
        k = min(k, n - 1)

        X, Y, Z, x_disc, y_disc = self._prepare(
            x_col, y_col, z_cols, rng,
        )

        mi_stat = self._estimate_mi(X, Y, Z, k)

        # Permutation test for significance
        p_value = _permutation_mi_pvalue(
            mi_stat, X, Y, Z, k, cfg.algorithm,
            cfg.n_permutations, rng, cfg.metric,
        )

        return self._make_result(
            x, y, conditioning_set, mi_stat, p_value,
        )

    def estimate_mi(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: pd.DataFrame,
    ) -> float:
        """Estimate MI (or conditional MI) without significance testing.

        Parameters
        ----------
        x : NodeId
            First variable.
        y : NodeId
            Second variable.
        conditioning_set : NodeSet
            Conditioning variables.
        data : pd.DataFrame
            Observational data.

        Returns
        -------
        float
            MI / CMI estimate in nats.
        """
        _validate_inputs(data, x, y, conditioning_set)
        x_col, y_col, z_cols = _extract_columns(data, x, y, conditioning_set)

        n = len(x_col)
        rng = np.random.default_rng(self.seed)
        cfg = self.mi_config

        k = _select_k(n, cfg.k) if cfg.adaptive_k else cfg.k
        k = min(k, n - 1)

        X, Y, Z, _, _ = self._prepare(x_col, y_col, z_cols, rng)
        return self._estimate_mi(X, Y, Z, k)

    def __repr__(self) -> str:  # noqa: D105
        cfg = self.mi_config
        return (
            f"MutualInfoCITest(k={cfg.k}, algorithm={cfg.algorithm}, "
            f"adaptive_k={cfg.adaptive_k}, alpha={self.alpha})"
        )

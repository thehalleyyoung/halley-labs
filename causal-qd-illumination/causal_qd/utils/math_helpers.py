"""Numerical helper functions for scoring, information theory, and statistics.

Provides numerically stable log-space operations, information-theoretic
measures (mutual information, entropy, KL/JS divergence), partial correlation,
Fisher's z-transform, and matrix operation wrappers.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Log-space operations
# ---------------------------------------------------------------------------

def log_sum_exp(a: npt.NDArray[np.float64]) -> float:
    """Numerically stable log-sum-exp.

    Parameters
    ----------
    a:
        1-D array of log-space values.

    Returns
    -------
    float
        ``log(sum(exp(a)))``.
    """
    a = np.asarray(a, dtype=np.float64)
    a_max = a.max()
    if not np.isfinite(a_max):
        return float(a_max)
    return float(a_max + np.log(np.sum(np.exp(a - a_max))))


def log_diff_exp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) - exp(b)) where a >= b.

    Parameters
    ----------
    a, b : float
        Log-space values with ``a >= b``.

    Returns
    -------
    float
        ``log(exp(a) - exp(b))``.

    Raises
    ------
    ValueError
        If ``a < b``.
    """
    if a < b:
        raise ValueError(f"Requires a >= b, got a={a}, b={b}")
    if a == b:
        return -np.inf
    if not np.isfinite(a):
        return float(a)
    return float(a + np.log1p(-np.exp(b - a)))


def log_gamma(x: float) -> float:
    """Natural logarithm of the gamma function (wraps *scipy*).

    Parameters
    ----------
    x:
        Positive real number.

    Returns
    -------
    float
        ``log(Γ(x))``.
    """
    return float(gammaln(x))


def log_binom(n: int, k: int) -> float:
    """Logarithm of the binomial coefficient *C(n, k)*.

    Uses ``log C(n,k) = log Γ(n+1) - log Γ(k+1) - log Γ(n-k+1)``.
    """
    if k < 0 or k > n:
        return -math.inf
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


# ---------------------------------------------------------------------------
# Information-theoretic measures
# ---------------------------------------------------------------------------

def entropy(x: npt.NDArray[np.float64], method: str = "discrete") -> float:
    """Compute entropy of a distribution or data vector.

    Parameters
    ----------
    x : ndarray
        If *method* is ``"discrete"``: probability vector (sums to 1).
        If *method* is ``"continuous"``: 1-D data array (uses KDE-based
        estimate via histogram).

    method : str
        ``"discrete"`` for categorical entropy, ``"continuous"`` for
        differential entropy estimate.

    Returns
    -------
    float
        Shannon entropy in nats (natural logarithm).
    """
    x = np.asarray(x, dtype=np.float64)

    if method == "discrete":
        # Probability vector
        p = x[x > 0]
        return float(-np.sum(p * np.log(p)))

    elif method == "continuous":
        # Histogram-based differential entropy estimate
        n = len(x)
        if n < 2:
            return 0.0
        n_bins = max(int(np.sqrt(n)), 10)
        counts, bin_edges = np.histogram(x, bins=n_bins, density=False)
        bin_width = bin_edges[1] - bin_edges[0]
        if bin_width == 0:
            return 0.0
        # Normalize to probabilities
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)) + np.log(bin_width))

    else:
        raise ValueError(f"Unknown method: {method!r}")


def mutual_information(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    cov: Optional[npt.NDArray[np.float64]] = None,
) -> float:
    """Compute mutual information I(X; Y).

    If *cov* is provided, uses the Gaussian formula:
        I(X; Y) = -0.5 * log(1 - ρ²)
    where ρ is the correlation coefficient.

    If *cov* is None, uses the data arrays *x* and *y* directly
    with a histogram-based estimator.

    Parameters
    ----------
    x, y : ndarray
        1-D data arrays of the same length.
    cov : ndarray, optional
        2×2 covariance matrix.  If provided, *x* and *y* are ignored.

    Returns
    -------
    float
        Mutual information in nats.
    """
    if cov is not None:
        cov = np.asarray(cov, dtype=np.float64)
        det_cov = np.linalg.det(cov)
        if det_cov <= 0:
            return 0.0
        var_x = cov[0, 0]
        var_y = cov[1, 1]
        if var_x <= 0 or var_y <= 0:
            return 0.0
        rho_sq = (cov[0, 1] ** 2) / (var_x * var_y)
        rho_sq = min(rho_sq, 1.0 - 1e-15)  # Clamp for numerical stability
        return float(-0.5 * np.log(1.0 - rho_sq))

    # Data-based estimation using 2D histogram
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(x)
    if n < 2:
        return 0.0

    n_bins = max(int(np.sqrt(n) / 2), 5)
    hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
    p_xy = hist_xy / hist_xy.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    return float(max(mi, 0.0))


def conditional_mutual_information(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    cov: Optional[npt.NDArray[np.float64]] = None,
) -> float:
    """Compute conditional mutual information I(X; Y | Z).

    If *cov* is provided (3×3 covariance matrix for [X, Y, Z]):
        Uses the Gaussian formula via partial correlation.

    Otherwise uses a data-based binning approach:
        I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)

    Parameters
    ----------
    x, y, z : ndarray
        1-D data arrays of the same length.
    cov : ndarray, optional
        3×3 covariance matrix for variables [X, Y, Z].

    Returns
    -------
    float
        Conditional mutual information in nats.
    """
    if cov is not None:
        cov = np.asarray(cov, dtype=np.float64)
        # I(X;Y|Z) = -0.5 * log(1 - partial_corr(X,Y|Z)^2)
        pcorr = partial_correlation(cov, 0, 1, [2])
        pcorr_sq = min(pcorr ** 2, 1.0 - 1e-15)
        return float(-0.5 * np.log(1.0 - pcorr_sq))

    # Data-based: binning approach
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    n = len(x)
    if n < 4:
        return 0.0

    n_bins = max(int(n ** (1.0 / 3.0)), 3)

    def _hist_entropy(*arrays: npt.NDArray[np.float64]) -> float:
        data = np.column_stack(arrays)
        hist, _ = np.histogramdd(data, bins=n_bins)
        p = hist / hist.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    h_xz = _hist_entropy(x, z)
    h_yz = _hist_entropy(y, z)
    h_xyz = _hist_entropy(x, y, z)
    h_z = _hist_entropy(z)

    cmi = h_xz + h_yz - h_xyz - h_z
    return float(max(cmi, 0.0))


# ---------------------------------------------------------------------------
# Correlation / partial correlation
# ---------------------------------------------------------------------------

def partial_correlation(
    cov: npt.NDArray[np.float64],
    i: int,
    j: int,
    conditioning: list[int],
) -> float:
    """Compute the partial correlation between variables *i* and *j*
    conditioned on *conditioning*, given a covariance matrix.

    Uses the recursive formula via the precision (inverse covariance)
    matrix of the relevant submatrix.

    Parameters
    ----------
    cov : ndarray
        Full covariance matrix (p × p).
    i, j : int
        Indices of the two variables.
    conditioning : list of int
        Indices of conditioning variables.

    Returns
    -------
    float
        Partial correlation coefficient in [-1, 1].
    """
    if not conditioning:
        # Simple correlation
        var_i = cov[i, i]
        var_j = cov[j, j]
        if var_i <= 0 or var_j <= 0:
            return 0.0
        return float(cov[i, j] / np.sqrt(var_i * var_j))

    # Extract submatrix for [i, j] + conditioning
    idx = [i, j] + list(conditioning)
    sub_cov = cov[np.ix_(idx, idx)]

    # Compute precision matrix (inverse of covariance)
    try:
        precision = np.linalg.inv(sub_cov)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(sub_cov)

    # Partial correlation = -P[0,1] / sqrt(P[0,0] * P[1,1])
    if precision[0, 0] <= 0 or precision[1, 1] <= 0:
        return 0.0
    pcorr = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
    return float(np.clip(pcorr, -1.0, 1.0))


def fisher_z_transform(r: float, n: int, k: int = 0) -> float:
    """Fisher's z-transform of a (partial) correlation coefficient.

    Transforms correlation *r* to a z-statistic that is approximately
    normally distributed under H0: ρ = 0.

    Parameters
    ----------
    r : float
        (Partial) correlation coefficient in (-1, 1).
    n : int
        Sample size.
    k : int
        Number of conditioning variables (default 0).

    Returns
    -------
    float
        z-statistic = sqrt(n - k - 3) * 0.5 * log((1+r)/(1-r)).
    """
    r = np.clip(r, -1 + 1e-15, 1 - 1e-15)
    z = 0.5 * np.log((1.0 + r) / (1.0 - r))
    return float(np.sqrt(max(n - k - 3, 1)) * z)


# ---------------------------------------------------------------------------
# Divergence measures
# ---------------------------------------------------------------------------

def kl_divergence(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
) -> float:
    """Compute KL divergence D_KL(P || Q).

    Parameters
    ----------
    p, q : ndarray
        Probability distributions (must sum to 1, same length).

    Returns
    -------
    float
        KL divergence in nats. Returns inf if q[i]=0 where p[i]>0.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Handle zeros
    mask = p > 0
    if np.any(mask & (q <= 0)):
        return float("inf")

    kl = np.sum(p[mask] * np.log(p[mask] / q[mask]))
    return float(kl)


def jsd(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
) -> float:
    """Jensen-Shannon Divergence JSD(P || Q).

    The symmetric, bounded divergence:
        JSD(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)
    where M = 0.5 * (P + Q).

    Parameters
    ----------
    p, q : ndarray
        Probability distributions (must sum to 1, same length).

    Returns
    -------
    float
        JSD value in [0, log(2)] (nats).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    # Compute KL(P||M) and KL(Q||M) safely
    kl_pm = 0.0
    kl_qm = 0.0
    for i in range(len(p)):
        if p[i] > 0 and m[i] > 0:
            kl_pm += p[i] * np.log(p[i] / m[i])
        if q[i] > 0 and m[i] > 0:
            kl_qm += q[i] * np.log(q[i] / m[i])

    return float(0.5 * kl_pm + 0.5 * kl_qm)


# ---------------------------------------------------------------------------
# Matrix operations
# ---------------------------------------------------------------------------

def pseudoinverse(
    matrix: npt.NDArray[np.float64],
    rcond: float = 1e-10,
) -> npt.NDArray[np.float64]:
    """Compute the Moore-Penrose pseudoinverse.

    Parameters
    ----------
    matrix : ndarray
        Input matrix.
    rcond : float
        Cutoff for small singular values.

    Returns
    -------
    ndarray
        Pseudoinverse of *matrix*.
    """
    return np.linalg.pinv(matrix, rcond=rcond)


def eigendecomposition(
    matrix: npt.NDArray[np.float64],
    symmetric: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute eigenvalue decomposition.

    Parameters
    ----------
    matrix : ndarray
        Square matrix.
    symmetric : bool
        If True, use the faster symmetric eigendecomposition (eigh).

    Returns
    -------
    eigenvalues : ndarray
        1-D array of eigenvalues (sorted ascending for symmetric).
    eigenvectors : ndarray
        2-D array where columns are eigenvectors.
    """
    if symmetric:
        vals, vecs = np.linalg.eigh(matrix)
    else:
        vals, vecs = np.linalg.eig(matrix)
    return vals, vecs


def regularized_inverse(
    matrix: npt.NDArray[np.float64],
    reg: float = 1e-6,
) -> npt.NDArray[np.float64]:
    """Compute the inverse of a matrix with Tikhonov regularization.

    Adds *reg* * I to the matrix before inverting, which helps with
    ill-conditioned matrices.

    Parameters
    ----------
    matrix : ndarray
        Square matrix.
    reg : float
        Regularization strength.

    Returns
    -------
    ndarray
        Regularized inverse.
    """
    n = matrix.shape[0]
    return np.linalg.inv(matrix + reg * np.eye(n))

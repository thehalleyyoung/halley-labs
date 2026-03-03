"""Information-theoretic measures for the CPA engine.

Provides Shannon entropy, mutual information, conditional mutual
information, transfer entropy, multi-distribution JSD, normalised
information distance, and information-theoretic independence tests.
"""

from __future__ import annotations

import math
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

_EPS = 1e-300
_LN2 = math.log(2.0)


# ===================================================================
# Shannon entropy
# ===================================================================


def shannon_entropy_discrete(
    p: NDArray[np.floating],
    *,
    base: float = math.e,
) -> float:
    """Shannon entropy of a discrete distribution.

    Parameters
    ----------
    p : np.ndarray
        Probability mass function.  Need not be normalised.
    base : float
        Logarithm base (``e`` for nats, ``2`` for bits).

    Returns
    -------
    float
        Entropy H(P) >= 0.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    if np.any(p < 0):
        raise ValueError("PMF entries must be non-negative")
    p_sum = p.sum()
    if p_sum < _EPS:
        return 0.0
    p = p / p_sum
    mask = p > 0
    h = -np.sum(p[mask] * np.log(p[mask]))
    if base != math.e:
        h /= math.log(base)
    return max(0.0, float(h))


def shannon_entropy_gaussian(
    variance: float,
    *,
    d: int = 1,
) -> float:
    """Differential entropy of a Gaussian distribution.

    For univariate (d=1):
        H = 0.5 * ln(2πeσ²)

    For multivariate with scalar variance (σ²I):
        H = d/2 * ln(2πeσ²)

    Parameters
    ----------
    variance : float
        Variance (scalar, must be > 0).
    d : int
        Dimensionality.

    Returns
    -------
    float
        Differential entropy in nats.
    """
    if variance <= 0:
        raise ValueError(f"variance must be > 0, got {variance}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    return 0.5 * d * math.log(2 * math.pi * math.e * variance)


def shannon_entropy_gaussian_mv(
    cov: NDArray[np.floating],
) -> float:
    """Differential entropy of a multivariate Gaussian.

    H = 0.5 * ln((2πe)^d * |Σ|)

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix, shape ``(d, d)``.

    Returns
    -------
    float
        Differential entropy in nats.
    """
    cov = np.asarray(cov, dtype=np.float64)
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        warnings.warn(
            "Covariance matrix is not positive definite; entropy may be invalid",
            stacklevel=2,
        )
    return 0.5 * (d * math.log(2 * math.pi * math.e) + logdet)


# ===================================================================
# Mutual information
# ===================================================================


def mutual_information_discrete(
    joint: NDArray[np.floating],
    *,
    base: float = math.e,
) -> float:
    """Mutual information from a joint probability table.

    Parameters
    ----------
    joint : np.ndarray
        2-D joint PMF, shape ``(k1, k2)``.
    base : float
        Logarithm base.

    Returns
    -------
    float
        MI(X; Y) >= 0.
    """
    joint = np.asarray(joint, dtype=np.float64)
    if joint.ndim != 2:
        raise ValueError(f"joint must be 2-D, got {joint.ndim}-D")
    if np.any(joint < 0):
        raise ValueError("joint PMF entries must be non-negative")
    total = joint.sum()
    if total < _EPS:
        return 0.0
    joint = joint / total
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    outer = np.outer(p_x, p_y)
    mask = (joint > 0) & (outer > 0)
    mi = np.sum(joint[mask] * np.log(joint[mask] / outer[mask]))
    if base != math.e:
        mi /= math.log(base)
    return max(0.0, float(mi))


def mutual_information_gaussian(
    cov: NDArray[np.floating],
    idx_x: Sequence[int],
    idx_y: Sequence[int],
) -> float:
    """Mutual information between two subsets of jointly-Gaussian variables.

    MI(X; Y) = 0.5 * ln(|Σ_XX| * |Σ_YY| / |Σ_XY_joint|)

    Parameters
    ----------
    cov : np.ndarray
        Full covariance matrix, shape ``(p, p)``.
    idx_x : sequence of int
        Indices of the X variables.
    idx_y : sequence of int
        Indices of the Y variables.

    Returns
    -------
    float
        MI in nats (>= 0).
    """
    cov = np.asarray(cov, dtype=np.float64)
    idx_x = list(idx_x)
    idx_y = list(idx_y)
    idx_joint = idx_x + idx_y

    cov_xx = cov[np.ix_(idx_x, idx_x)]
    cov_yy = cov[np.ix_(idx_y, idx_y)]
    cov_joint = cov[np.ix_(idx_joint, idx_joint)]

    _, logdet_xx = np.linalg.slogdet(cov_xx)
    _, logdet_yy = np.linalg.slogdet(cov_yy)
    _, logdet_joint = np.linalg.slogdet(cov_joint)

    mi = 0.5 * (logdet_xx + logdet_yy - logdet_joint)
    return max(0.0, float(mi))


def mutual_information_from_data(
    X: NDArray[np.floating],
    idx_x: Sequence[int],
    idx_y: Sequence[int],
    *,
    method: str = "gaussian",
    n_bins: int = 10,
) -> float:
    """Estimate mutual information from data.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    idx_x : sequence of int
        Column indices for X.
    idx_y : sequence of int
        Column indices for Y.
    method : ``"gaussian"`` or ``"binned"``
        Estimation method.
    n_bins : int
        Number of bins per variable (for ``"binned"`` method).

    Returns
    -------
    float
        Estimated MI in nats.
    """
    X = np.asarray(X, dtype=np.float64)
    idx_x, idx_y = list(idx_x), list(idx_y)

    if method == "gaussian":
        cov = np.cov(X, rowvar=False)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        return mutual_information_gaussian(cov, idx_x, idx_y)
    elif method == "binned":
        if len(idx_x) != 1 or len(idx_y) != 1:
            raise ValueError("Binned MI only supports single variables")
        x_data = X[:, idx_x[0]]
        y_data = X[:, idx_y[0]]
        joint, _, _ = np.histogram2d(x_data, y_data, bins=n_bins)
        return mutual_information_discrete(joint)
    else:
        raise ValueError(f"Unknown method {method!r}")


# ===================================================================
# Conditional mutual information
# ===================================================================


def conditional_mutual_information(
    cov: NDArray[np.floating],
    idx_x: Sequence[int],
    idx_y: Sequence[int],
    idx_z: Sequence[int],
) -> float:
    """Conditional mutual information MI(X; Y | Z) for Gaussian variables.

    MI(X; Y | Z) = MI(X; Y, Z) - MI(X; Z)
                 = 0.5 * ln(|Σ_{XZ}| * |Σ_{YZ}| / (|Σ_Z| * |Σ_{XYZ}|))

    Parameters
    ----------
    cov : np.ndarray
        Full covariance matrix.
    idx_x : sequence of int
        Indices for X.
    idx_y : sequence of int
        Indices for Y.
    idx_z : sequence of int
        Indices for Z (conditioning set).

    Returns
    -------
    float
        CMI in nats (>= 0).
    """
    cov = np.asarray(cov, dtype=np.float64)
    idx_x = list(idx_x)
    idx_y = list(idx_y)
    idx_z = list(idx_z)

    if len(idx_z) == 0:
        return mutual_information_gaussian(cov, idx_x, idx_y)

    idx_xz = idx_x + idx_z
    idx_yz = idx_y + idx_z
    idx_xyz = idx_x + idx_y + idx_z

    cov_xz = cov[np.ix_(idx_xz, idx_xz)]
    cov_yz = cov[np.ix_(idx_yz, idx_yz)]
    cov_z = cov[np.ix_(idx_z, idx_z)]
    cov_xyz = cov[np.ix_(idx_xyz, idx_xyz)]

    _, ld_xz = np.linalg.slogdet(cov_xz)
    _, ld_yz = np.linalg.slogdet(cov_yz)
    _, ld_z = np.linalg.slogdet(cov_z)
    _, ld_xyz = np.linalg.slogdet(cov_xyz)

    cmi = 0.5 * (ld_xz + ld_yz - ld_z - ld_xyz)
    return max(0.0, float(cmi))


def conditional_mutual_information_from_data(
    X: NDArray[np.floating],
    idx_x: Sequence[int],
    idx_y: Sequence[int],
    idx_z: Sequence[int],
) -> float:
    """Estimate CMI from data assuming Gaussianity.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    idx_x, idx_y, idx_z : sequences of int
        Variable index sets.

    Returns
    -------
    float
        Estimated CMI in nats.
    """
    cov = np.cov(np.asarray(X, dtype=np.float64), rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    return conditional_mutual_information(cov, idx_x, idx_y, idx_z)


# ===================================================================
# Transfer entropy
# ===================================================================


def transfer_entropy(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    *,
    lag: int = 1,
    method: str = "gaussian",
    n_bins: int = 10,
) -> float:
    """Transfer entropy from X to Y: TE(X→Y).

    TE(X→Y) = MI(Y_t; X_{t-lag} | Y_{t-lag})

    Parameters
    ----------
    X : np.ndarray
        Source time series, shape ``(T,)``.
    Y : np.ndarray
        Target time series, shape ``(T,)``.
    lag : int
        Time lag (>= 1).
    method : ``"gaussian"`` or ``"binned"``
        Estimation method.
    n_bins : int
        Bins for histogram method.

    Returns
    -------
    float
        Transfer entropy in nats (>= 0).
    """
    X = np.asarray(X, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    T = min(len(X), len(Y))
    if T <= lag:
        raise ValueError(f"Series length {T} must exceed lag {lag}")

    y_t = Y[lag:]
    x_lag = X[:T - lag]
    y_lag = Y[:T - lag]

    n = len(y_t)
    data = np.column_stack([y_t, x_lag, y_lag])  # columns: Y_t, X_{t-1}, Y_{t-1}

    if method == "gaussian":
        cov = np.cov(data, rowvar=False)
        return conditional_mutual_information(cov, [0], [1], [2])
    elif method == "binned":
        # Discretise each variable
        def _digitize(arr: NDArray) -> NDArray:
            edges = np.linspace(arr.min() - 1e-10, arr.max() + 1e-10, n_bins + 1)
            return np.digitize(arr, edges[1:-1])

        d_yt = _digitize(y_t)
        d_xl = _digitize(x_lag)
        d_yl = _digitize(y_lag)

        # H(Y_t, Y_{t-1}) + H(X_{t-1}, Y_{t-1}) - H(Y_{t-1}) - H(Y_t, X_{t-1}, Y_{t-1})
        def _entropy_from_counts(*arrays: NDArray) -> float:
            combined = np.column_stack(arrays)
            _, counts = np.unique(combined, axis=0, return_counts=True)
            p = counts / counts.sum()
            return float(-np.sum(p * np.log(p + _EPS)))

        h_yt_yl = _entropy_from_counts(d_yt, d_yl)
        h_xl_yl = _entropy_from_counts(d_xl, d_yl)
        h_yl = _entropy_from_counts(d_yl)
        h_yt_xl_yl = _entropy_from_counts(d_yt, d_xl, d_yl)

        te = h_yt_yl + h_xl_yl - h_yl - h_yt_xl_yl
        return max(0.0, te)
    else:
        raise ValueError(f"Unknown method {method!r}")


# ===================================================================
# Multi-distribution JSD
# ===================================================================


def multi_distribution_jsd(
    distributions: Sequence[NDArray[np.floating]],
    *,
    weights: Optional[NDArray[np.floating]] = None,
) -> float:
    """Generalised Jensen-Shannon divergence for K ≥ 2 distributions.

    JSD_π(P_1, ..., P_K) = H(∑ π_k P_k) - ∑ π_k H(P_k)

    Parameters
    ----------
    distributions : sequence of np.ndarray
        List of PMFs (all same length).
    weights : np.ndarray, optional
        Mixing weights (must sum to 1).  Uniform by default.

    Returns
    -------
    float
        Generalised JSD in nats (>= 0).
    """
    K = len(distributions)
    if K < 2:
        raise ValueError(f"Need at least 2 distributions, got {K}")

    dists = [np.asarray(d, dtype=np.float64).ravel() for d in distributions]
    n = len(dists[0])
    for i, d in enumerate(dists):
        if len(d) != n:
            raise ValueError(
                f"Distribution {i} has length {len(d)}, expected {n}"
            )

    if weights is None:
        w = np.ones(K, dtype=np.float64) / K
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if len(w) != K:
            raise ValueError(f"weights length {len(w)} != {K}")
        if abs(w.sum() - 1.0) > 1e-6:
            w = w / w.sum()

    # Normalise distributions
    normed = []
    for d in dists:
        s = d.sum()
        normed.append(d / s if s > _EPS else d)

    # Mixture
    mixture = np.zeros(n, dtype=np.float64)
    for i in range(K):
        mixture += w[i] * normed[i]

    # H(mixture) - sum w_k H(P_k)
    h_mix = _discrete_entropy(mixture)
    h_components = sum(w[i] * _discrete_entropy(normed[i]) for i in range(K))
    return max(0.0, h_mix - h_components)


def _discrete_entropy(p: NDArray[np.floating]) -> float:
    """Compute entropy of a normalised PMF."""
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def multi_distribution_jsd_gaussian(
    means: Sequence[float],
    variances: Sequence[float],
    *,
    weights: Optional[Sequence[float]] = None,
    n_points: int = 1000,
) -> float:
    """Generalised JSD for K univariate Gaussians via numerical integration.

    Parameters
    ----------
    means : sequence of float
        Means of each Gaussian.
    variances : sequence of float
        Variances of each Gaussian (all > 0).
    weights : sequence of float, optional
        Mixing weights.
    n_points : int
        Number of quadrature points.

    Returns
    -------
    float
        JSD in nats.
    """
    K = len(means)
    if K < 2:
        raise ValueError(f"Need at least 2 distributions, got {K}")
    if len(variances) != K:
        raise ValueError("means and variances must have same length")

    if weights is None:
        w = [1.0 / K] * K
    else:
        w = list(weights)
        ws = sum(w)
        w = [wi / ws for wi in w]

    stds = [math.sqrt(v) for v in variances]
    center = sum(w[i] * means[i] for i in range(K))
    spread = 5.0 * max(stds) + max(abs(means[i] - center) for i in range(K))

    lo, hi = center - spread, center + spread
    x = np.linspace(lo, hi, n_points)
    dx = (hi - lo) / (n_points - 1)

    # Evaluate component PDFs
    pdfs = np.zeros((K, n_points), dtype=np.float64)
    for i in range(K):
        pdfs[i] = sp_stats.norm.pdf(x, means[i], stds[i])

    mixture = np.zeros(n_points, dtype=np.float64)
    for i in range(K):
        mixture += w[i] * pdfs[i]

    # H(mixture)
    mask_m = mixture > _EPS
    h_mix = -np.sum(mixture[mask_m] * np.log(mixture[mask_m])) * dx

    # Sum w_k H(P_k) — analytic for Gaussians
    h_components = sum(
        w[i] * 0.5 * math.log(2 * math.pi * math.e * variances[i])
        for i in range(K)
    )

    return max(0.0, float(h_mix - h_components))


# ===================================================================
# Normalised information distance
# ===================================================================


def normalized_information_distance(
    joint: NDArray[np.floating],
    *,
    base: float = math.e,
) -> float:
    """Normalised information distance (NID) from a joint PMF.

    NID(X, Y) = 1 - MI(X;Y) / max(H(X), H(Y))

    Parameters
    ----------
    joint : np.ndarray
        2-D joint PMF.
    base : float
        Log base.

    Returns
    -------
    float
        NID in [0, 1].
    """
    joint = np.asarray(joint, dtype=np.float64)
    if joint.ndim != 2:
        raise ValueError(f"joint must be 2-D, got {joint.ndim}-D")
    mi = mutual_information_discrete(joint, base=base)
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    h_x = shannon_entropy_discrete(p_x, base=base)
    h_y = shannon_entropy_discrete(p_y, base=base)
    max_h = max(h_x, h_y)
    if max_h < _EPS:
        return 0.0
    nid = 1.0 - mi / max_h
    return float(np.clip(nid, 0.0, 1.0))


def normalized_information_distance_gaussian(
    cov: NDArray[np.floating],
    idx_x: Sequence[int],
    idx_y: Sequence[int],
) -> float:
    """NID for Gaussian variables from covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.
    idx_x, idx_y : sequences of int
        Variable index sets.

    Returns
    -------
    float
        NID in [0, 1].
    """
    cov = np.asarray(cov, dtype=np.float64)
    mi = mutual_information_gaussian(cov, idx_x, idx_y)
    h_x = shannon_entropy_gaussian_mv(cov[np.ix_(list(idx_x), list(idx_x))])
    h_y = shannon_entropy_gaussian_mv(cov[np.ix_(list(idx_y), list(idx_y))])
    max_h = max(h_x, h_y)
    if max_h < _EPS:
        return 0.0
    nid = 1.0 - mi / max_h
    return float(np.clip(nid, 0.0, 1.0))


# ===================================================================
# Information-theoretic independence tests
# ===================================================================


def mi_independence_test(
    X: NDArray[np.floating],
    idx_x: int,
    idx_y: int,
    *,
    n_permutations: int = 500,
    method: str = "gaussian",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Test independence using mutual information + permutation.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    idx_x : int
        First variable index.
    idx_y : int
        Second variable index.
    n_permutations : int
        Number of permutations for p-value computation.
    method : ``"gaussian"`` or ``"binned"``
        MI estimation method.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    (mi_observed, p_value) : tuple of float
    """
    rng = rng or np.random.default_rng()
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]

    mi_obs = mutual_information_from_data(X, [idx_x], [idx_y], method=method)
    count = 0
    X_perm = X.copy()
    for _ in range(n_permutations):
        X_perm[:, idx_y] = rng.permutation(X[:, idx_y])
        mi_perm = mutual_information_from_data(X_perm, [idx_x], [idx_y], method=method)
        if mi_perm >= mi_obs:
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    return float(mi_obs), float(p_value)


def cmi_independence_test(
    X: NDArray[np.floating],
    idx_x: int,
    idx_y: int,
    idx_z: Sequence[int],
    *,
    n_permutations: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Test conditional independence using CMI + permutation.

    Tests H₀: X ⊥ Y | Z.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    idx_x, idx_y : int
        Variable indices.
    idx_z : sequence of int
        Conditioning set indices.
    n_permutations : int
        Number of permutations.
    rng : np.random.Generator, optional
        RNG.

    Returns
    -------
    (cmi_observed, p_value) : tuple of float
    """
    rng = rng or np.random.default_rng()
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    idx_z = list(idx_z)

    cmi_obs = conditional_mutual_information_from_data(X, [idx_x], [idx_y], idx_z)
    count = 0
    X_perm = X.copy()
    for _ in range(n_permutations):
        X_perm[:, idx_y] = rng.permutation(X[:, idx_y])
        cmi_perm = conditional_mutual_information_from_data(
            X_perm, [idx_x], [idx_y], idx_z
        )
        if cmi_perm >= cmi_obs:
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    return float(cmi_obs), float(p_value)


# ===================================================================
# Interaction information (multivariate generalisation)
# ===================================================================


def interaction_information(
    cov: NDArray[np.floating],
    idx_x: Sequence[int],
    idx_y: Sequence[int],
    idx_z: Sequence[int],
) -> float:
    """Interaction information (co-information) for Gaussian variables.

    II(X;Y;Z) = MI(X;Y|Z) - MI(X;Y)

    Positive values indicate synergy, negative values indicate redundancy.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.
    idx_x, idx_y, idx_z : sequences of int
        Variable index sets.

    Returns
    -------
    float
        Interaction information in nats (can be negative).
    """
    cmi = conditional_mutual_information(cov, idx_x, idx_y, idx_z)
    mi = mutual_information_gaussian(cov, idx_x, idx_y)
    return float(cmi - mi)


# ===================================================================
# Total correlation
# ===================================================================


def total_correlation(
    cov: NDArray[np.floating],
    variable_indices: Sequence[Sequence[int]],
) -> float:
    """Total correlation (multi-information) for Gaussian variables.

    TC(X1, ..., Xk) = sum H(Xi) - H(X1, ..., Xk)

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.
    variable_indices : sequence of sequences of int
        Index sets for each variable group.

    Returns
    -------
    float
        Total correlation in nats (>= 0).
    """
    cov = np.asarray(cov, dtype=np.float64)

    # Sum of marginal entropies
    sum_h = 0.0
    all_idx: list[int] = []
    for idx_group in variable_indices:
        idx_list = list(idx_group)
        all_idx.extend(idx_list)
        cov_i = cov[np.ix_(idx_list, idx_list)]
        sum_h += shannon_entropy_gaussian_mv(cov_i)

    # Joint entropy
    cov_joint = cov[np.ix_(all_idx, all_idx)]
    h_joint = shannon_entropy_gaussian_mv(cov_joint)

    return max(0.0, float(sum_h - h_joint))

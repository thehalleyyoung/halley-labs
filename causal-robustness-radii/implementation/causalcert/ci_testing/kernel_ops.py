"""
Shared kernel operations for kernel-based CI tests.

Provides efficient kernel matrix computation (RBF, polynomial, Laplacian),
Nyström low-rank approximation, kernel centering, incomplete Cholesky
decomposition, an LRU kernel cache, and block-diagonal kernel construction
for conditioning-set handling.

These utilities are shared by :mod:`hsic` and :mod:`kci`.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------


def rbf_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    gamma: float | None = None,
) -> np.ndarray:
    """Compute the Gaussian RBF kernel matrix.

    K(x, y) = exp(-gamma * ||x - y||^2)

    Parameters
    ----------
    X : np.ndarray
        Data matrix ``(n, d)``.
    Y : np.ndarray | None
        Second data matrix ``(m, d)``.  If ``None``, computes ``K(X, X)``.
    gamma : float | None
        Bandwidth parameter.  If ``None``, uses the median heuristic on *X*.

    Returns
    -------
    np.ndarray
        Kernel matrix ``(n, m)`` or ``(n, n)``.
    """
    X = np.atleast_2d(X)
    if Y is None:
        Y = X
    else:
        Y = np.atleast_2d(Y)

    if gamma is None:
        gamma = median_heuristic(X)

    sq_dists = cdist(X, Y, metric="sqeuclidean")
    return np.exp(-gamma * sq_dists)


def polynomial_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    degree: int = 3,
    gamma: float = 1.0,
    coef0: float = 1.0,
) -> np.ndarray:
    """Compute the polynomial kernel matrix.

    K(x, y) = (gamma * <x, y> + coef0) ^ degree

    Parameters
    ----------
    X : np.ndarray
        Data matrix ``(n, d)``.
    Y : np.ndarray | None
        Second data matrix.  ``None`` ⇒ ``K(X, X)``.
    degree : int
        Polynomial degree.
    gamma : float
        Scaling factor for the inner product.
    coef0 : float
        Free parameter (bias).

    Returns
    -------
    np.ndarray
        Kernel matrix.
    """
    X = np.atleast_2d(X)
    if Y is None:
        Y = X
    else:
        Y = np.atleast_2d(Y)
    return (gamma * (X @ Y.T) + coef0) ** degree


def laplacian_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    gamma: float | None = None,
) -> np.ndarray:
    """Compute the Laplacian kernel matrix.

    K(x, y) = exp(-gamma * ||x - y||_1)

    Parameters
    ----------
    X : np.ndarray
        Data matrix ``(n, d)``.
    Y : np.ndarray | None
        Second data matrix.
    gamma : float | None
        Bandwidth.  ``None`` ⇒ median heuristic (L1 variant).

    Returns
    -------
    np.ndarray
        Kernel matrix.
    """
    X = np.atleast_2d(X)
    if Y is None:
        Y = X
    else:
        Y = np.atleast_2d(Y)

    if gamma is None:
        dists = cdist(X[:min(2000, len(X))], X[:min(2000, len(X))], "cityblock")
        med = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        gamma = 1.0 / max(med, _EPS)

    l1_dists = cdist(X, Y, metric="cityblock")
    return np.exp(-gamma * l1_dists)


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------


def median_heuristic(X: np.ndarray, *, subsample: int = 2000) -> float:
    """Select RBF bandwidth via the median heuristic.

    gamma = 1 / (2 * median_dist^2)

    Parameters
    ----------
    X : np.ndarray
        Data matrix ``(n, d)``.
    subsample : int
        Maximum number of points for pairwise-distance computation.

    Returns
    -------
    float
        Bandwidth parameter gamma.
    """
    X = np.atleast_2d(X)
    n = X.shape[0]
    if n < 2:
        return 1.0

    if n > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=subsample, replace=False)
        X = X[idx]

    dists = pdist(X, metric="euclidean")
    if len(dists) == 0:
        return 1.0

    med = float(np.median(dists))
    if med < _EPS:
        return 1.0
    return 1.0 / (2.0 * med * med)


def cross_validation_bandwidth(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_candidates: int = 10,
    n_folds: int = 5,
    seed: int = 42,
) -> float:
    """Select RBF bandwidth via leave-one-out cross-validation.

    Maximises the log-marginal-likelihood proxy of a kernel density
    estimator on the joint ``(X, Y)`` space across a grid of gamma values
    centred on the median heuristic.

    Parameters
    ----------
    X : np.ndarray
        First variable ``(n, d1)``.
    Y : np.ndarray
        Second variable ``(n, d2)``.
    n_candidates : int
        Number of candidate bandwidths to evaluate.
    n_folds : int
        Number of cross-validation folds.
    seed : int
        Random seed.

    Returns
    -------
    float
        Best bandwidth (gamma).
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    XY = np.hstack([X, Y])
    n = XY.shape[0]

    gamma_med = median_heuristic(XY)
    log_gammas = np.linspace(
        np.log(gamma_med) - 2.0, np.log(gamma_med) + 2.0, n_candidates,
    )
    candidates = np.exp(log_gammas)

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    best_gamma = gamma_med
    best_score = -np.inf

    for gamma in candidates:
        scores: list[float] = []
        for fold_idx in range(n_folds):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate(
                [folds[j] for j in range(n_folds) if j != fold_idx]
            )
            K_test_train = rbf_kernel(XY[test_idx], XY[train_idx], gamma=gamma)
            # log-density proxy: mean log of average kernel value
            avg_k = K_test_train.mean(axis=1)
            avg_k = np.maximum(avg_k, _EPS)
            scores.append(float(np.mean(np.log(avg_k))))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_gamma = float(gamma)

    return best_gamma


# ---------------------------------------------------------------------------
# Kernel centering
# ---------------------------------------------------------------------------


def center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-center a kernel matrix in feature space.

    H K H  where  H = I - (1/n) 11^T.

    Parameters
    ----------
    K : np.ndarray
        Kernel matrix ``(n, n)``.

    Returns
    -------
    np.ndarray
        Centered kernel matrix.
    """
    n = K.shape[0]
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    grand_mean = K.mean()
    return K - row_mean - col_mean + grand_mean


def center_kernel_asymmetric(
    K: np.ndarray,
    K_train: np.ndarray,
) -> np.ndarray:
    """Center a test-vs-train kernel matrix using training statistics.

    Parameters
    ----------
    K : np.ndarray
        Test-vs-train kernel ``(m, n)``.
    K_train : np.ndarray
        Training kernel ``(n, n)``.

    Returns
    -------
    np.ndarray
        Centered kernel ``(m, n)``.
    """
    n = K_train.shape[0]
    train_col_mean = K_train.mean(axis=0, keepdims=True)  # (1, n)
    test_row_mean = K.mean(axis=1, keepdims=True)  # (m, 1)
    grand_mean = K_train.mean()
    return K - test_row_mean - train_col_mean + grand_mean


# ---------------------------------------------------------------------------
# Nyström approximation
# ---------------------------------------------------------------------------


def nystrom_approximation(
    X: np.ndarray,
    *,
    n_components: int = 200,
    gamma: float | None = None,
    kernel: Literal["rbf", "polynomial", "laplacian"] = "rbf",
    seed: int = 42,
    degree: int = 3,
    coef0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nyström low-rank approximation of a kernel matrix.

    Selects *n_components* landmark points uniformly at random, computes
    the sub-matrices, and returns the approximate factorisation
    ``K ≈ C @ W_inv @ C.T`` in the form ``(Z, eigenvalues)`` where
    ``Z = C @ W_inv_sqrt`` so that ``K ≈ Z @ Z.T``.

    Parameters
    ----------
    X : np.ndarray
        Data matrix ``(n, d)``.
    n_components : int
        Number of inducing (landmark) points.
    gamma : float | None
        Kernel bandwidth.
    kernel : str
        Kernel type.
    seed : int
        Random seed.
    degree : int
        Polynomial degree (only for polynomial kernel).
    coef0 : float
        Bias term (only for polynomial kernel).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(Z, eigenvalues)`` where ``Z`` has shape ``(n, n_components)``
        and the approximate kernel is ``Z @ Z.T``.
    """
    X = np.atleast_2d(X)
    n = X.shape[0]
    m = min(n_components, n)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    X_m = X[idx]

    kfn = _select_kernel_fn(kernel)
    kw = _kernel_kwargs(kernel, gamma, X, degree, coef0)

    # W = K(X_m, X_m)  shape (m, m)
    W = kfn(X_m, X_m, **kw)
    # C = K(X, X_m)    shape (n, m)
    C = kfn(X, X_m, **kw)

    # Eigendecompose W for numerical stability
    eigvals, eigvecs = np.linalg.eigh(W)
    # Keep only positive eigenvalues
    pos = eigvals > _EPS
    eigvals_pos = eigvals[pos]
    eigvecs_pos = eigvecs[:, pos]

    # W^{-1/2} via eigendecomposition
    W_inv_sqrt = eigvecs_pos @ np.diag(1.0 / np.sqrt(eigvals_pos))

    # Z = C @ W^{-1/2}
    Z = C @ W_inv_sqrt
    return Z, eigvals_pos


# ---------------------------------------------------------------------------
# Incomplete Cholesky decomposition
# ---------------------------------------------------------------------------


def incomplete_cholesky(
    K: np.ndarray,
    *,
    max_rank: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Incomplete Cholesky decomposition of a PSD kernel matrix.

    Returns a lower-triangular factor ``L`` of shape ``(n, r)`` where
    ``r <= max_rank`` such that ``K ≈ L @ L.T``.

    Uses the pivoted variant that greedily selects the column with the
    largest residual diagonal element.

    Parameters
    ----------
    K : np.ndarray
        PSD kernel matrix ``(n, n)``.
    max_rank : int
        Maximum rank of the approximation.
    tol : float
        Tolerance on residual diagonal entries.

    Returns
    -------
    np.ndarray
        Factor ``L`` of shape ``(n, r)``.
    """
    n = K.shape[0]
    r = min(max_rank, n)

    L = np.zeros((n, r), dtype=np.float64)
    diag_residual = K.diagonal().copy()
    piv = np.arange(n)

    for j in range(r):
        # Select pivot: largest residual diagonal
        remaining = np.arange(j, n)
        best = remaining[np.argmax(diag_residual[piv[remaining]])]
        # Swap
        piv[j], piv[best] = piv[best], piv[j]

        dval = diag_residual[piv[j]]
        if dval < tol:
            L = L[:, :j]
            break

        L[piv[j], j] = np.sqrt(dval)
        if L[piv[j], j] < _EPS:
            L = L[:, :j]
            break

        for i in range(j + 1, n):
            val = K[piv[i], piv[j]]
            if j > 0:
                val -= L[piv[i], :j] @ L[piv[j], :j]
            L[piv[i], j] = val / L[piv[j], j]

        # Update residual diagonal
        for i in range(j + 1, n):
            diag_residual[piv[i]] -= L[piv[i], j] ** 2

    return L


# ---------------------------------------------------------------------------
# Block-diagonal kernel for conditioning
# ---------------------------------------------------------------------------


def block_diagonal_kernel(
    Z: np.ndarray,
    *,
    gamma: float | None = None,
    kernel: Literal["rbf", "polynomial", "laplacian"] = "rbf",
    degree: int = 3,
    coef0: float = 1.0,
) -> np.ndarray:
    """Construct a kernel matrix on the conditioning set.

    When the conditioning set has multiple columns, this computes
    the product kernel: ``K_Z = prod_j K_j`` where each ``K_j`` is a
    kernel on the *j*-th conditioning variable.  This is equivalent
    to a block-diagonal structure in the product feature space.

    Parameters
    ----------
    Z : np.ndarray
        Conditioning variables ``(n, k)``.
    gamma : float | None
        Shared bandwidth.  ``None`` ⇒ per-column median heuristic.
    kernel : str
        Kernel type.
    degree : int
        Polynomial degree.
    coef0 : float
        Bias term.

    Returns
    -------
    np.ndarray
        Product kernel matrix ``(n, n)``.
    """
    Z = np.atleast_2d(Z)
    n, k = Z.shape
    K_prod = np.ones((n, n), dtype=np.float64)

    kfn = _select_kernel_fn(kernel)

    for j in range(k):
        z_j = Z[:, j : j + 1]
        g = gamma if gamma is not None else median_heuristic(z_j)
        kw = _kernel_kwargs(kernel, g, z_j, degree, coef0)
        K_j = kfn(z_j, z_j, **kw)
        K_prod *= K_j

    return K_prod


# ---------------------------------------------------------------------------
# Kernel cache with LRU eviction
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """Single cache entry holding a kernel matrix."""

    key: tuple
    matrix: np.ndarray


class KernelCache:
    """Thread-safe LRU cache for kernel matrices.

    Caches the most recently used kernel matrices keyed by a hash of
    the data subset and kernel parameters.  Evicts least-recently-used
    entries when the cache exceeds *max_entries*.

    Parameters
    ----------
    max_entries : int
        Maximum number of cached kernel matrices.
    """

    def __init__(self, max_entries: int = 64) -> None:
        self._max = max_entries
        self._store: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Cache key construction
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(
        data_hash: int,
        kernel: str,
        gamma: float | None,
        extra: tuple = (),
    ) -> tuple:
        """Build a hashable cache key.

        Parameters
        ----------
        data_hash : int
            Hash of the data array (e.g. via ``hash(data.tobytes())``).
        kernel : str
            Kernel name.
        gamma : float | None
            Bandwidth.
        extra : tuple
            Additional parameters (degree, coef0, etc.).

        Returns
        -------
        tuple
            Hashable key.
        """
        return (data_hash, kernel, gamma, extra)

    @staticmethod
    def hash_array(arr: np.ndarray) -> int:
        """Compute a fast hash of a numpy array.

        Parameters
        ----------
        arr : np.ndarray
            Array to hash.

        Returns
        -------
        int
            Integer hash.
        """
        return hash(arr.data.tobytes())

    # ------------------------------------------------------------------
    # Get / put
    # ------------------------------------------------------------------

    def get(self, key: tuple) -> np.ndarray | None:
        """Retrieve a cached kernel matrix, or ``None`` on miss.

        Parameters
        ----------
        key : tuple
            Cache key from :meth:`make_key`.

        Returns
        -------
        np.ndarray | None
        """
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    def put(self, key: tuple, matrix: np.ndarray) -> None:
        """Store a kernel matrix, evicting LRU if full.

        Parameters
        ----------
        key : tuple
            Cache key.
        matrix : np.ndarray
            Kernel matrix to cache.
        """
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = matrix
                return
            if len(self._store) >= self._max:
                self._store.popitem(last=False)
            self._store[key] = matrix

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that were cache hits."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"KernelCache(size={self.size}/{self._max}, "
            f"hit_rate={self.hit_rate:.2%})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _select_kernel_fn(kernel: str):  # noqa: ANN202
    """Return the kernel function for the given name."""
    mapping = {
        "rbf": rbf_kernel,
        "polynomial": polynomial_kernel,
        "laplacian": laplacian_kernel,
    }
    fn = mapping.get(kernel)
    if fn is None:
        raise ValueError(
            f"Unknown kernel {kernel!r}. Choose from {list(mapping)}."
        )
    return fn


def _kernel_kwargs(
    kernel: str,
    gamma: float | None,
    X: np.ndarray,
    degree: int = 3,
    coef0: float = 1.0,
) -> dict:
    """Build keyword arguments for a kernel function."""
    if kernel == "rbf":
        g = gamma if gamma is not None else median_heuristic(X)
        return {"gamma": g}
    elif kernel == "polynomial":
        g = gamma if gamma is not None else 1.0 / max(X.shape[1], 1)
        return {"degree": degree, "gamma": g, "coef0": coef0}
    elif kernel == "laplacian":
        return {"gamma": gamma}
    return {}

"""Matrix, eigenvalue, numerical-stability, and combinatorial utilities.

All functions operate on plain NumPy arrays and are designed for the small-to-
medium matrix sizes typical of causal DAGs (up to a few hundred nodes).
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FloatMatrix = NDArray[np.floating]
IntMatrix = NDArray[np.integer]


# ====================================================================
# 1. Matrix operations
# ====================================================================

def symmetrise(M: FloatMatrix) -> FloatMatrix:
    """Return (M + Mᵀ) / 2, forcing exact symmetry."""
    return (M + M.T) / 2.0


def is_symmetric(M: FloatMatrix, atol: float = 1e-10) -> bool:
    """Check whether *M* is symmetric within absolute tolerance."""
    return bool(np.allclose(M, M.T, atol=atol))


def is_positive_definite(M: FloatMatrix) -> bool:
    """Check positive-definiteness via Cholesky decomposition."""
    try:
        np.linalg.cholesky(symmetrise(M))
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_positive_definite(M: FloatMatrix) -> FloatMatrix:
    """Compute the nearest symmetric positive-definite matrix (Higham 1988).

    Uses the iterative alternating-projection algorithm.
    """
    B = symmetrise(M)
    _, S, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(S) @ Vt
    A2 = (B + H) / 2.0
    A3 = symmetrise(A2)

    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(M))
    I = np.eye(M.shape[0])  # noqa: E741
    k = 1
    while not is_positive_definite(A3):
        mineig = float(np.min(np.real(np.linalg.eigvalsh(A3))))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1
    return A3


def matrix_power_series(A: FloatMatrix, max_power: int) -> FloatMatrix:
    """Compute I + A + A² + … + A^{max_power} (useful for reachability)."""
    n = A.shape[0]
    result = np.eye(n, dtype=A.dtype)
    power = np.eye(n, dtype=A.dtype)
    for _ in range(max_power):
        power = power @ A
        result = result + power
    return result


def spectral_radius(M: FloatMatrix) -> float:
    """Return the spectral radius (largest absolute eigenvalue)."""
    eigvals = np.linalg.eigvals(M)
    return float(np.max(np.abs(eigvals)))


def condition_number(M: FloatMatrix) -> float:
    """Compute the 2-norm condition number, clamped at 1e15."""
    cond = float(np.linalg.cond(M))
    return min(cond, 1e15)


# ====================================================================
# 2. Eigenvalue computations
# ====================================================================

def sorted_eigenvalues(M: FloatMatrix) -> NDArray[np.floating]:
    """Return real eigenvalues of a symmetric matrix in ascending order."""
    return np.sort(np.linalg.eigvalsh(symmetrise(M)))


def eigenvalue_gap(M: FloatMatrix) -> float:
    """Return the minimum gap between consecutive eigenvalues."""
    eigs = sorted_eigenvalues(M)
    if len(eigs) < 2:
        return float("inf")
    diffs = np.diff(eigs)
    return float(np.min(diffs))


def leading_eigenvector(M: FloatMatrix) -> NDArray[np.floating]:
    """Return the eigenvector associated with the largest eigenvalue."""
    eigvals, eigvecs = np.linalg.eigh(symmetrise(M))
    idx = np.argmax(eigvals)
    return eigvecs[:, idx]


def effective_rank(M: FloatMatrix, tol: float = 1e-6) -> int:
    """Number of singular values above *tol* (numerical rank)."""
    sv = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(sv > tol))


# ====================================================================
# 3. Numerical stability helpers
# ====================================================================

def safe_log(x: FloatMatrix, floor: float = 1e-300) -> FloatMatrix:
    """Element-wise log clamped at *floor* to avoid -inf."""
    return np.log(np.maximum(x, floor))


def safe_divide(
    a: FloatMatrix,
    b: FloatMatrix,
    fill: float = 0.0,
) -> FloatMatrix:
    """Element-wise a/b, replacing 0/0 with *fill*."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, fill)
    return result


def log_sum_exp(a: NDArray[np.floating]) -> float:
    """Numerically stable log-sum-exp."""
    a_max = np.max(a)
    return float(a_max + np.log(np.sum(np.exp(a - a_max))))


def softmax(logits: NDArray[np.floating]) -> NDArray[np.floating]:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp *x* into [lo, hi]."""
    return max(lo, min(hi, x))


def relative_error(approx: float, exact: float) -> float:
    """Relative error |approx − exact| / max(|exact|, ε)."""
    return abs(approx - exact) / max(abs(exact), 1e-15)


# ====================================================================
# 4. Statistical distribution functions
# ====================================================================

def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def normal_quantile(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Approximate normal quantile (Beasley–Springer–Moro algorithm)."""
    from scipy.stats import norm
    return float(norm.ppf(p, loc=mu, scale=sigma))


def chi2_cdf(x: float, df: int) -> float:
    """Chi-squared CDF."""
    from scipy.stats import chi2
    return float(chi2.cdf(x, df))


def fisher_z(r: float, n: int) -> float:
    """Fisher z-transformation of a correlation *r* with sample size *n*."""
    z = 0.5 * math.log((1 + r) / (1 - r + 1e-15))
    return z * math.sqrt(n - 3)


def partial_correlation_to_z(
    r: float,
    n: int,
    k: int,
) -> float:
    """Convert partial correlation to z-statistic.

    Parameters
    ----------
    r : partial correlation coefficient.
    n : sample size.
    k : size of the conditioning set.
    """
    z = 0.5 * math.log((1 + r) / (1 - r + 1e-15))
    return z * math.sqrt(n - k - 3)


# ====================================================================
# 5. Combinatorial utilities
# ====================================================================

def powerset(s: Sequence[int], max_size: int | None = None) -> Iterator[tuple[int, ...]]:
    """Yield all subsets of *s* up to *max_size* elements."""
    n = len(s)
    limit = n if max_size is None else min(max_size, n)
    for k in range(limit + 1):
        yield from combinations(s, k)


def n_choose_k(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    return math.comb(n, k)


def enumerate_edge_subsets(
    n_edges: int,
    max_k: int,
) -> Iterator[tuple[int, ...]]:
    """Yield index-subsets of edges up to size *max_k*."""
    for k in range(1, max_k + 1):
        yield from combinations(range(n_edges), k)


def catalan_number(n: int) -> int:
    """The n-th Catalan number (counts full binary trees, triangulations…)."""
    return math.comb(2 * n, n) // (n + 1)


# ====================================================================
# 6. Graph-theory utilities (matrix level)
# ====================================================================

def adjacency_to_reachability(adj: IntMatrix) -> IntMatrix:
    """Boolean reachability matrix via matrix power series (transitive closure)."""
    n = adj.shape[0]
    reach = matrix_power_series(adj.astype(np.float64), n)
    return (reach > 0).astype(np.int8)


def in_degrees(adj: IntMatrix) -> NDArray[np.int64]:
    """Return the in-degree vector."""
    return np.sum(adj, axis=0)


def out_degrees(adj: IntMatrix) -> NDArray[np.int64]:
    """Return the out-degree vector."""
    return np.sum(adj, axis=1)


def degree_sequence(adj: IntMatrix) -> NDArray[np.int64]:
    """Return the total degree (in + out) vector."""
    return in_degrees(adj) + out_degrees(adj)


def laplacian(adj: IntMatrix) -> FloatMatrix:
    """Combinatorial Laplacian of the undirected skeleton."""
    skeleton = np.maximum(adj, adj.T).astype(np.float64)
    D = np.diag(skeleton.sum(axis=1))
    return D - skeleton


def number_of_dags(n: int) -> int:
    """Exact count of labelled DAGs on *n* nodes (Robinson 1977 recurrence).

    Warning: grows super-exponentially; feasible for n ≤ 18 or so.
    """
    if n <= 0:
        return 1
    a = [0] * (n + 1)
    a[0] = 1
    for i in range(1, n + 1):
        total = 0
        for k in range(1, i + 1):
            sign = (-1) ** (k + 1)
            binom = math.comb(i, k)
            total += sign * binom * (2 ** (k * (i - k))) * a[i - k]
        a[i] = total
    return a[n]

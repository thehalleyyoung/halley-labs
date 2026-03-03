"""
Numerical utilities and stability helpers for DP-Forge.

Provides sparse matrix operations, numerically stable primitives,
tolerance management (including Invariant I4), condition number analysis,
and matrix projection utilities used by the LP/SDP builder, verifier,
and baseline comparison modules.

Key capabilities:
    - Incremental sparse matrix construction without full rebuilds.
    - Numerically stable ``log_sum_exp`` / ``log_subtract_exp`` for
      privacy accounting in the log domain.
    - Tolerance bookkeeping that enforces ``dp_tol >= exp(ε) × solver_tol``.
    - Fast condition-number estimates and diagonal preconditioning.
    - Simplex and PSD cone projections for mechanism extraction.

All public functions carry full type annotations and are unit-testable
in isolation (no dependency on the CEGIS loop).
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import linalg as sp_linalg

from dp_forge.exceptions import (
    ConfigurationError,
    NumericalInstabilityError,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]
SparseMatrix = sparse.spmatrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_FLOOR: float = 1e-300
"""Default floor value for :func:`safe_log`."""

_ZERO_DIV_FILL: float = 0.0
"""Default fill value for :func:`safe_divide`."""

_SINGULARITY_THRESHOLD: float = 1e14
"""Default threshold for :func:`detect_near_singularity`."""


# =========================================================================
# 1. Sparse matrix operations
# =========================================================================


def build_csr(
    data: Sequence[float],
    row_ind: Sequence[int],
    col_ind: Sequence[int],
    shape: Tuple[int, int],
) -> sparse.csr_matrix:
    """Construct a CSR sparse matrix from COO-style triplets.

    This is the preferred entry point for building LP constraint matrices
    because it avoids the overhead of creating a dense array and converting.

    Args:
        data: Non-zero values.
        row_ind: Row indices for each value.
        col_ind: Column indices for each value.
        shape: ``(n_rows, n_cols)`` shape of the resulting matrix.

    Returns:
        A ``scipy.sparse.csr_matrix`` in canonical form (sorted indices,
        no duplicates).

    Raises:
        ValueError: If any index is out of range or arrays have mismatched
            lengths.

    Example::

        >>> A = build_csr([1.0, -1.0], [0, 0], [0, 1], (1, 3))
        >>> A.toarray()
        array([[ 1., -1.,  0.]])
    """
    data_arr = np.asarray(data, dtype=np.float64)
    row_arr = np.asarray(row_ind, dtype=np.int32)
    col_arr = np.asarray(col_ind, dtype=np.int32)

    if not (len(data_arr) == len(row_arr) == len(col_arr)):
        raise ValueError(
            f"data ({len(data_arr)}), row_ind ({len(row_arr)}), and "
            f"col_ind ({len(col_arr)}) must have equal length"
        )

    if len(row_arr) > 0:
        if np.any(row_arr < 0) or np.any(row_arr >= shape[0]):
            raise ValueError(
                f"row_ind values must be in [0, {shape[0]}), "
                f"got range [{row_arr.min()}, {row_arr.max()}]"
            )
        if np.any(col_arr < 0) or np.any(col_arr >= shape[1]):
            raise ValueError(
                f"col_ind values must be in [0, {shape[1]}), "
                f"got range [{col_arr.min()}, {col_arr.max()}]"
            )

    coo = sparse.coo_matrix((data_arr, (row_arr, col_arr)), shape=shape)
    csr = coo.tocsr()
    csr.sum_duplicates()
    csr.sort_indices()
    return csr


def build_csr_from_dense(
    A: FloatArray,
) -> sparse.csr_matrix:
    """Convert a dense 2-D array to CSR format.

    Args:
        A: Dense matrix of shape ``(m, n)``.

    Returns:
        Equivalent CSR sparse matrix.

    Raises:
        ValueError: If *A* is not 2-D.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {A.shape}")
    return sparse.csr_matrix(A)


def sparse_vstack_incremental(
    base: Optional[sparse.csr_matrix],
    new_rows: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """Vertically stack *new_rows* onto *base* without full rebuild.

    When *base* is ``None`` the result is just *new_rows*.  Otherwise
    ``scipy.sparse.vstack`` is used, which is efficient for CSR inputs
    because it concatenates the internal arrays directly.

    Args:
        base: Existing constraint matrix (may be ``None`` for first call).
        new_rows: Rows to append.  Must have the same number of columns
            as *base* (when *base* is not ``None``).

    Returns:
        The vertically stacked CSR matrix.

    Raises:
        ValueError: If column counts mismatch.
    """
    if base is None:
        return new_rows.tocsr() if not sparse.issparse(new_rows) else new_rows

    if base.shape[1] != new_rows.shape[1]:
        raise ValueError(
            f"Column mismatch: base has {base.shape[1]} columns, "
            f"new_rows has {new_rows.shape[1]}"
        )

    stacked = sparse.vstack([base, new_rows], format="csr")
    return stacked


def sparse_block_diag(
    blocks: Sequence[Union[sparse.spmatrix, FloatArray]],
) -> sparse.csr_matrix:
    """Build a block-diagonal sparse matrix from a sequence of blocks.

    Each block can be dense or sparse.  The result is CSR.

    Args:
        blocks: Sequence of 2-D matrices (dense or sparse).

    Returns:
        Block-diagonal CSR matrix.

    Raises:
        ValueError: If any block is not 2-D or the sequence is empty.
    """
    if len(blocks) == 0:
        raise ValueError("blocks must be non-empty")

    sparse_blocks = []
    for i, blk in enumerate(blocks):
        if sparse.issparse(blk):
            sparse_blocks.append(blk)
        else:
            arr = np.asarray(blk, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(
                    f"Block {i} must be 2-D, got shape {arr.shape}"
                )
            sparse_blocks.append(sparse.csr_matrix(arr))

    return sparse.block_diag(sparse_blocks, format="csr")


def sparse_nnz_fraction(A: sparse.spmatrix) -> float:
    """Return the fraction of non-zero entries in *A*.

    Args:
        A: Sparse matrix.

    Returns:
        ``nnz / (rows × cols)``, or 0.0 for an empty matrix.
    """
    total = A.shape[0] * A.shape[1]
    if total == 0:
        return 0.0
    return A.nnz / total


def sparse_row_norms(
    A: sparse.spmatrix,
    ord: Union[int, float] = 2,
) -> FloatArray:
    """Compute per-row norms of a sparse matrix.

    Args:
        A: Sparse matrix of shape ``(m, n)``.
        ord: Norm order (1, 2, or ``np.inf``).

    Returns:
        Array of shape ``(m,)`` with the norm of each row.
    """
    A_csr = A.tocsr()
    m = A_csr.shape[0]
    norms = np.empty(m, dtype=np.float64)
    for i in range(m):
        row_data = A_csr.getrow(i).toarray().ravel()
        norms[i] = np.linalg.norm(row_data, ord=ord)
    return norms


def sparse_max_abs(A: sparse.spmatrix) -> float:
    """Return the maximum absolute value in a sparse matrix.

    Handles the case where the matrix has no non-zeros gracefully.

    Args:
        A: Sparse matrix.

    Returns:
        ``max(|A_ij|)`` or 0.0 if the matrix is empty.
    """
    if A.nnz == 0:
        return 0.0
    coo = A.tocoo()
    return float(np.max(np.abs(coo.data)))


# =========================================================================
# 2. Numerical stability primitives
# =========================================================================


def log_sum_exp(a: FloatArray) -> float:
    """Numerically stable computation of ``log(sum(exp(a)))``.

    Uses the standard max-shift trick to avoid overflow.

    Args:
        a: 1-D array of log-domain values.

    Returns:
        ``log(sum(exp(a_i)))`` computed without overflow.

    Raises:
        ValueError: If *a* is empty.

    Example::

        >>> log_sum_exp(np.array([1000.0, 1000.0]))  # doctest: +ELLIPSIS
        1000.693...
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    if len(a) == 0:
        raise ValueError("log_sum_exp requires a non-empty array")
    a_max = np.max(a)
    if not np.isfinite(a_max):
        return float(a_max)
    return float(a_max + np.log(np.sum(np.exp(a - a_max))))


def log_sum_exp_array(a: FloatArray, axis: int = 0) -> FloatArray:
    """Numerically stable ``log(sum(exp(a), axis))`` along an axis.

    Generalisation of :func:`log_sum_exp` for 2-D arrays.

    Args:
        a: 2-D array.
        axis: Axis along which to sum.

    Returns:
        Reduced array.
    """
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a, axis=axis, keepdims=True)
    # Replace -inf max with 0 to avoid NaN in exp
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    out = a_max_safe.squeeze(axis=axis) + np.log(
        np.sum(np.exp(a - a_max_safe), axis=axis)
    )
    return out


def log_subtract_exp(a: float, b: float) -> float:
    """Numerically stable computation of ``log(exp(a) - exp(b))``.

    Requires ``a >= b`` (i.e. ``exp(a) >= exp(b)``).

    Args:
        a: Larger log-domain value.
        b: Smaller log-domain value.

    Returns:
        ``log(exp(a) - exp(b))``.

    Raises:
        ValueError: If ``b > a`` (result would be log of a negative number).

    Example::

        >>> log_subtract_exp(2.0, 1.0)  # doctest: +ELLIPSIS
        1.541...
    """
    if b > a + 1e-15:
        raise ValueError(
            f"log_subtract_exp requires a >= b, got a={a}, b={b}"
        )
    if a == b:
        return -np.inf
    if b == -np.inf:
        return a
    return float(a + np.log1p(-np.exp(b - a)))


def safe_divide(
    a: Union[float, FloatArray],
    b: Union[float, FloatArray],
    fill: float = _ZERO_DIV_FILL,
) -> Union[float, FloatArray]:
    """Element-wise division ``a / b`` with zero-handling.

    Where ``b`` is zero (or near-zero), the result is *fill* instead of
    inf or NaN.

    Args:
        a: Numerator (scalar or array).
        b: Denominator (scalar or array).
        fill: Value to use where ``|b| < 1e-300``.

    Returns:
        ``a / b`` with safe fill.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    mask = np.abs(b_arr) < 1e-300
    safe_b = np.where(mask, 1.0, b_arr)
    result = a_arr / safe_b
    result = np.where(mask, fill, result)

    if result.ndim == 0:
        return float(result)
    return result


def safe_log(
    x: Union[float, FloatArray],
    floor: float = _LOG_FLOOR,
) -> Union[float, FloatArray]:
    """Logarithm with a floor to avoid ``log(0) = -inf``.

    Args:
        x: Input value(s).
        floor: Minimum value to clamp *x* to before taking log.

    Returns:
        ``log(max(x, floor))``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    clamped = np.maximum(x_arr, floor)
    result = np.log(clamped)
    if result.ndim == 0:
        return float(result)
    return result


def safe_exp(
    x: Union[float, FloatArray],
    cap: float = 700.0,
) -> Union[float, FloatArray]:
    """Exponentiation with overflow capping.

    Args:
        x: Input value(s).
        cap: Maximum exponent before capping.

    Returns:
        ``exp(min(x, cap))``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    clamped = np.minimum(x_arr, cap)
    result = np.exp(clamped)
    if result.ndim == 0:
        return float(result)
    return result


def kl_divergence(
    p: FloatArray,
    q: FloatArray,
    floor: float = 1e-300,
) -> float:
    """Compute KL(p || q) with numerical stability.

    Args:
        p: Probability distribution (1-D, sums to ~1).
        q: Reference distribution (1-D, sums to ~1).
        floor: Floor for zero probabilities.

    Returns:
        ``sum(p_i * log(p_i / q_i))``, with zeros handled.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if len(p) != len(q):
        raise ValueError(f"p ({len(p)}) and q ({len(q)}) must have equal length")

    p_safe = np.maximum(p, floor)
    q_safe = np.maximum(q, floor)
    # Only sum where p > 0
    mask = p > floor
    return float(np.sum(p_safe[mask] * np.log(p_safe[mask] / q_safe[mask])))


def total_variation(p: FloatArray, q: FloatArray) -> float:
    """Total variation distance ``TV(p, q) = 0.5 * ||p - q||_1``.

    Args:
        p: First probability distribution.
        q: Second probability distribution.

    Returns:
        Total variation distance in ``[0, 1]``.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if len(p) != len(q):
        raise ValueError(f"p ({len(p)}) and q ({len(q)}) must have equal length")
    return float(0.5 * np.sum(np.abs(p - q)))


def renyi_divergence(
    p: FloatArray,
    q: FloatArray,
    alpha: float,
    floor: float = 1e-300,
) -> float:
    """Rényi divergence ``D_alpha(p || q)`` of order *alpha*.

    Args:
        p: First probability distribution.
        q: Second probability distribution.
        alpha: Order (must be > 0 and != 1).
        floor: Floor for zero probabilities.

    Returns:
        Rényi divergence value.

    Raises:
        ValueError: If ``alpha <= 0`` or ``alpha == 1``.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if abs(alpha - 1.0) < 1e-12:
        return kl_divergence(p, q, floor=floor)

    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if len(p) != len(q):
        raise ValueError(f"p ({len(p)}) and q ({len(q)}) must have equal length")

    p_safe = np.maximum(p, floor)
    q_safe = np.maximum(q, floor)

    # D_alpha = 1/(alpha-1) * log(sum(p^alpha * q^(1-alpha)))
    log_terms = alpha * np.log(p_safe) + (1.0 - alpha) * np.log(q_safe)
    log_sum = log_sum_exp(log_terms)
    return float(log_sum / (alpha - 1.0))


# =========================================================================
# 3. Tolerance management
# =========================================================================


def compute_dp_tolerance(
    epsilon: float,
    solver_tol: float,
) -> float:
    """Compute the minimum DP verification tolerance satisfying Invariant I4.

    Invariant I4: ``dp_tol >= exp(ε) × solver_tol``.  This function
    returns ``exp(ε) × solver_tol`` as the tightest valid value.

    Args:
        epsilon: Privacy parameter ε > 0.
        solver_tol: Solver feasibility tolerance.

    Returns:
        Minimum valid ``dp_tol``.

    Raises:
        ConfigurationError: If inputs are non-positive.
    """
    if epsilon <= 0:
        raise ConfigurationError(
            f"epsilon must be > 0, got {epsilon}",
            parameter="epsilon",
            value=epsilon,
            constraint="epsilon > 0",
        )
    if solver_tol <= 0:
        raise ConfigurationError(
            f"solver_tol must be > 0, got {solver_tol}",
            parameter="solver_tol",
            value=solver_tol,
            constraint="solver_tol > 0",
        )
    return math.exp(epsilon) * solver_tol


def check_tolerance_consistency(
    dp_tol: float,
    solver_tol: float,
    epsilon: float,
) -> bool:
    """Check whether the tolerance triple satisfies Invariant I4.

    Args:
        dp_tol: DP verification tolerance.
        solver_tol: Solver feasibility tolerance.
        epsilon: Privacy parameter ε.

    Returns:
        ``True`` if ``dp_tol >= exp(ε) × solver_tol``.
    """
    required = math.exp(epsilon) * solver_tol
    return dp_tol >= required


def adaptive_tolerance(
    iteration: int,
    base_tol: float,
    decay_rate: float = 0.9,
    min_tol: float = 1e-12,
) -> float:
    """Compute a tolerance that tightens as CEGIS iterations progress.

    Returns ``max(base_tol * decay_rate^iteration, min_tol)``.

    Args:
        iteration: Current CEGIS iteration (0-indexed).
        base_tol: Starting tolerance.
        decay_rate: Multiplicative decay per iteration (in (0, 1)).
        min_tol: Floor tolerance.

    Returns:
        The adapted tolerance.

    Raises:
        ValueError: If parameters are out of valid ranges.
    """
    if iteration < 0:
        raise ValueError(f"iteration must be >= 0, got {iteration}")
    if base_tol <= 0:
        raise ValueError(f"base_tol must be > 0, got {base_tol}")
    if not (0 < decay_rate < 1):
        raise ValueError(f"decay_rate must be in (0, 1), got {decay_rate}")
    if min_tol <= 0:
        raise ValueError(f"min_tol must be > 0, got {min_tol}")

    tol = base_tol * (decay_rate ** iteration)
    return max(tol, min_tol)


def tolerance_margin(
    dp_tol: float,
    solver_tol: float,
    epsilon: float,
) -> float:
    """Compute how much slack remains before Invariant I4 is violated.

    Returns ``dp_tol - exp(ε) × solver_tol``.  A positive value means
    the invariant is satisfied; negative means violated.

    Args:
        dp_tol: DP verification tolerance.
        solver_tol: Solver feasibility tolerance.
        epsilon: Privacy parameter ε.

    Returns:
        Tolerance margin (positive = safe).
    """
    required = math.exp(epsilon) * solver_tol
    return dp_tol - required


# =========================================================================
# 4. Condition number analysis
# =========================================================================


def estimate_condition(
    A: Union[sparse.spmatrix, FloatArray],
    method: str = "svd",
) -> float:
    """Fast condition number estimate for a matrix.

    For sparse matrices the SVD-based estimate uses
    ``scipy.sparse.linalg.svds`` to find the largest and smallest
    singular values.  For small dense matrices ``numpy.linalg.cond``
    is used directly.

    Args:
        A: Matrix (sparse or dense, not necessarily square).
        method: Estimation method — ``'svd'`` (default) or ``'norm'``.
            ``'norm'`` uses the ratio of the largest to smallest row norms
            as a cheap proxy.

    Returns:
        Estimated condition number (≥ 1).

    Raises:
        ValueError: If *A* is empty.
    """
    if sparse.issparse(A):
        if A.shape[0] == 0 or A.shape[1] == 0:
            raise ValueError("Cannot estimate condition of an empty matrix")

        if method == "norm":
            row_norms = sparse_row_norms(A, ord=2)
            max_norm = np.max(row_norms)
            min_norm = np.min(row_norms)
            if min_norm < 1e-300:
                return np.inf
            return float(max_norm / min_norm)

        # SVD approach
        k = min(A.shape[0], A.shape[1])
        if k <= 2:
            dense = A.toarray()
            return float(np.linalg.cond(dense))

        try:
            sigma_max = sp_linalg.svds(
                A.astype(np.float64), k=1, which="LM", return_singular_vectors=False
            )
            sigma_min = sp_linalg.svds(
                A.astype(np.float64), k=1, which="SM", return_singular_vectors=False
            )
            s_max = float(sigma_max[0])
            s_min = float(sigma_min[0])
            if s_min < 1e-300:
                return np.inf
            return s_max / s_min
        except Exception:
            # Fallback to norm-based estimate
            return estimate_condition(A, method="norm")
    else:
        A = np.asarray(A, dtype=np.float64)
        if A.size == 0:
            raise ValueError("Cannot estimate condition of an empty matrix")
        return float(np.linalg.cond(A))


def diagonal_preconditioning(
    A: Union[sparse.spmatrix, FloatArray],
) -> Tuple[Union[sparse.spmatrix, FloatArray], FloatArray]:
    """Scale rows of *A* so that each row has unit 2-norm.

    This is a simple diagonal (left) preconditioner that improves the
    conditioning of LP constraint matrices.

    Args:
        A: Matrix to precondition.

    Returns:
        ``(A_scaled, scale_factors)`` where
        ``A_scaled[i, :] = A[i, :] / ||A[i, :]||_2`` and
        ``scale_factors[i] = ||A[i, :]||_2``.  Rows with zero norm are
        left unchanged (scale factor = 1).
    """
    if sparse.issparse(A):
        norms = sparse_row_norms(A, ord=2)
    else:
        A = np.asarray(A, dtype=np.float64)
        norms = np.linalg.norm(A, axis=1, ord=2)

    # Avoid division by zero
    safe_norms = np.where(norms < 1e-300, 1.0, norms)
    scale = 1.0 / safe_norms

    if sparse.issparse(A):
        D = sparse.diags(scale)
        A_scaled = D @ A
    else:
        A_scaled = A * scale[:, np.newaxis]

    return A_scaled, safe_norms


def detect_near_singularity(
    A: Union[sparse.spmatrix, FloatArray],
    threshold: float = _SINGULARITY_THRESHOLD,
) -> bool:
    """Check whether *A* is near-singular based on condition number.

    Args:
        A: Matrix to check.
        threshold: Condition number threshold.

    Returns:
        ``True`` if the estimated condition number exceeds *threshold*.
    """
    try:
        cond = estimate_condition(A, method="norm")
    except ValueError:
        return True
    return cond > threshold


def check_condition_and_raise(
    A: Union[sparse.spmatrix, FloatArray],
    max_condition: float,
    matrix_name: str = "constraint_matrix",
) -> float:
    """Estimate condition number and raise if it exceeds the threshold.

    Args:
        A: Matrix to check.
        max_condition: Maximum acceptable condition number.
        matrix_name: Identifier for error messages.

    Returns:
        The estimated condition number.

    Raises:
        NumericalInstabilityError: If condition number exceeds *max_condition*.
    """
    cond = estimate_condition(A, method="norm")
    if cond > max_condition:
        raise NumericalInstabilityError(
            f"Condition number of {matrix_name} ({cond:.2e}) exceeds "
            f"threshold ({max_condition:.2e}). Numerical results may be unreliable.",
            condition_number=cond,
            max_condition_number=max_condition,
            matrix_name=matrix_name,
        )
    return cond


# =========================================================================
# 5. Matrix utilities
# =========================================================================


def is_doubly_stochastic(
    M: FloatArray,
    tol: float = 1e-8,
) -> bool:
    """Check whether *M* is doubly stochastic.

    A matrix is doubly stochastic if all entries are non-negative and
    every row and every column sums to 1.

    Args:
        M: Square matrix to check.
        tol: Tolerance for sum and non-negativity checks.

    Returns:
        ``True`` if *M* is doubly stochastic within *tol*.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    if np.any(M < -tol):
        return False
    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)
    return bool(
        np.allclose(row_sums, 1.0, atol=tol)
        and np.allclose(col_sums, 1.0, atol=tol)
    )


def is_stochastic(
    M: FloatArray,
    tol: float = 1e-8,
) -> bool:
    """Check whether *M* is (row-)stochastic.

    Args:
        M: Matrix to check.
        tol: Tolerance for sum and non-negativity checks.

    Returns:
        ``True`` if all entries are non-negative and rows sum to 1.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        return False
    if np.any(M < -tol):
        return False
    row_sums = M.sum(axis=1)
    return bool(np.allclose(row_sums, 1.0, atol=tol))


def project_simplex(v: FloatArray) -> FloatArray:
    """Project vector *v* onto the probability simplex.

    Uses the efficient O(n log n) algorithm of Duchi et al. (2008).

    Args:
        v: 1-D array to project.

    Returns:
        Projection ``p`` such that ``p >= 0``, ``sum(p) = 1``, and
        ``||p - v||_2`` is minimised.

    Example::

        >>> project_simplex(np.array([0.5, 0.5, 0.5]))
        array([0.33333333, 0.33333333, 0.33333333])
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    n = len(v)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Sort in descending order
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho_candidates = u - cssv / np.arange(1, n + 1)
    rho = int(np.max(np.where(rho_candidates > 0)[0])) + 1 if np.any(rho_candidates > 0) else 1
    theta = cssv[rho - 1] / rho
    return np.maximum(v - theta, 0.0)


def project_simplex_rows(M: FloatArray) -> FloatArray:
    """Project each row of *M* onto the probability simplex.

    Args:
        M: 2-D array of shape ``(n, k)``.

    Returns:
        Array of same shape with each row projected.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {M.shape}")
    return np.array([project_simplex(row) for row in M])


def project_psd(M: FloatArray) -> FloatArray:
    """Project a symmetric matrix onto the positive semi-definite cone.

    Computes the eigendecomposition and zeros out negative eigenvalues.

    Args:
        M: Symmetric matrix.

    Returns:
        Nearest PSD matrix in Frobenius norm.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {M.shape}")

    # Symmetrise
    M_sym = 0.5 * (M + M.T)
    eigenvalues, eigenvectors = np.linalg.eigh(M_sym)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def project_dp_feasible(
    M: FloatArray,
    epsilon: float,
    delta: float,
    adjacency_edges: List[Tuple[int, int]],
    symmetric: bool = True,
) -> FloatArray:
    """Project a mechanism matrix onto the (ε, δ)-DP feasible set.

    First projects each row onto the simplex, then iteratively adjusts
    rows to satisfy the DP constraint for each adjacent pair.  This is a
    heuristic projection (alternating projections) and may not converge
    to the true projection, but it is sufficient for post-processing
    solver output that is already nearly feasible.

    Args:
        M: Mechanism matrix of shape ``(n, k)``.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        adjacency_edges: List of ``(i, i')`` pairs.
        symmetric: Whether adjacency is symmetric.

    Returns:
        Projected mechanism matrix satisfying DP (approximately).
    """
    M = np.asarray(M, dtype=np.float64).copy()
    n, k = M.shape
    exp_eps = math.exp(epsilon)

    # Step 1: project rows onto simplex
    M = project_simplex_rows(M)

    # Step 2: alternating projections for DP constraints
    max_proj_iters = 100
    for _ in range(max_proj_iters):
        violated = False
        edges = list(adjacency_edges)
        if symmetric:
            edges = edges + [(j, i) for i, j in adjacency_edges]

        for i, ip in edges:
            for j in range(k):
                upper = exp_eps * M[ip, j] + delta
                if M[i, j] > upper + 1e-15:
                    M[i, j] = upper
                    violated = True

        # Re-project rows onto simplex
        M = project_simplex_rows(M)

        if not violated:
            break

    return M


def normalize_rows(M: FloatArray, floor: float = 0.0) -> FloatArray:
    """Normalise rows to sum to 1, clamping negatives to *floor*.

    Args:
        M: 2-D array of shape ``(n, k)``.
        floor: Minimum entry value.

    Returns:
        Row-normalised array.
    """
    M = np.asarray(M, dtype=np.float64).copy()
    if M.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {M.shape}")

    M = np.maximum(M, floor)
    row_sums = M.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums < 1e-300, 1.0, row_sums)
    return M / safe_sums


def compute_output_grid(
    query_values: FloatArray,
    k: int,
    padding_factor: float = 3.0,
    sensitivity: float = 1.0,
) -> FloatArray:
    """Compute a uniform output discretization grid.

    The grid extends beyond the query range by ``padding_factor × sensitivity``
    on each side.

    Args:
        query_values: Array of query output values.
        k: Number of grid points.
        padding_factor: How many sensitivities to pad.
        sensitivity: Query sensitivity.

    Returns:
        1-D array of *k* uniformly spaced grid points.
    """
    query_values = np.asarray(query_values, dtype=np.float64)
    v_min = float(np.min(query_values))
    v_max = float(np.max(query_values))
    pad = padding_factor * sensitivity
    return np.linspace(v_min - pad, v_max + pad, k)


def weighted_mse(
    mechanism: FloatArray,
    query_values: FloatArray,
    y_grid: FloatArray,
    weights: Optional[FloatArray] = None,
) -> float:
    """Compute the (weighted) mean squared error of a discrete mechanism.

    ``MSE = sum_i w_i sum_j p[i,j] * (q_i - y_j)^2``

    Args:
        mechanism: Probability table ``(n, k)``.
        query_values: True query values ``(n,)``.
        y_grid: Output grid ``(k,)``.
        weights: Per-input weights ``(n,)``; uniform if ``None``.

    Returns:
        Weighted MSE.
    """
    mechanism = np.asarray(mechanism, dtype=np.float64)
    query_values = np.asarray(query_values, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)
    n, k = mechanism.shape

    if weights is None:
        weights = np.ones(n, dtype=np.float64) / n
    else:
        weights = np.asarray(weights, dtype=np.float64)

    # (n, k) matrix of squared errors
    sq_errors = (query_values[:, np.newaxis] - y_grid[np.newaxis, :]) ** 2
    per_input_mse = np.sum(mechanism * sq_errors, axis=1)
    return float(np.dot(weights, per_input_mse))


def weighted_mae(
    mechanism: FloatArray,
    query_values: FloatArray,
    y_grid: FloatArray,
    weights: Optional[FloatArray] = None,
) -> float:
    """Compute the (weighted) mean absolute error of a discrete mechanism.

    Args:
        mechanism: Probability table ``(n, k)``.
        query_values: True query values ``(n,)``.
        y_grid: Output grid ``(k,)``.
        weights: Per-input weights ``(n,)``; uniform if ``None``.

    Returns:
        Weighted MAE.
    """
    mechanism = np.asarray(mechanism, dtype=np.float64)
    query_values = np.asarray(query_values, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)
    n, k = mechanism.shape

    if weights is None:
        weights = np.ones(n, dtype=np.float64) / n
    else:
        weights = np.asarray(weights, dtype=np.float64)

    abs_errors = np.abs(query_values[:, np.newaxis] - y_grid[np.newaxis, :])
    per_input_mae = np.sum(mechanism * abs_errors, axis=1)
    return float(np.dot(weights, per_input_mae))


def linf_error(
    mechanism: FloatArray,
    query_values: FloatArray,
    y_grid: FloatArray,
) -> float:
    """Compute the worst-case L∞ error of a discrete mechanism.

    Returns ``max_i E_j[|q_i - y_j|]`` where the expectation is over
    the mechanism's output distribution for input *i*.

    Args:
        mechanism: Probability table ``(n, k)``.
        query_values: True query values ``(n,)``.
        y_grid: Output grid ``(k,)``.

    Returns:
        Maximum expected absolute error over all inputs.
    """
    mechanism = np.asarray(mechanism, dtype=np.float64)
    query_values = np.asarray(query_values, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)

    abs_errors = np.abs(query_values[:, np.newaxis] - y_grid[np.newaxis, :])
    per_input = np.sum(mechanism * abs_errors, axis=1)
    return float(np.max(per_input))


# =========================================================================
# Additional matrix analysis utilities
# =========================================================================


def frobenius_norm(M: FloatArray) -> float:
    """Frobenius norm of a matrix.

    Args:
        M: 2-D array.

    Returns:
        ``||M||_F = sqrt(sum(M_ij^2))``.
    """
    M = np.asarray(M, dtype=np.float64)
    return float(np.linalg.norm(M, "fro"))


def spectral_norm(M: FloatArray) -> float:
    """Spectral norm (largest singular value) of a matrix.

    Args:
        M: 2-D array.

    Returns:
        ``||M||_2 = sigma_max(M)``.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.size == 0:
        return 0.0
    return float(np.linalg.norm(M, 2))


def matrix_rank_estimate(
    M: Union[sparse.spmatrix, FloatArray],
    tol: Optional[float] = None,
) -> int:
    """Estimate the numerical rank of a matrix.

    Args:
        M: Matrix (sparse or dense).
        tol: Singular value threshold. Defaults to
            ``max(shape) × max(sv) × eps``.

    Returns:
        Estimated rank.
    """
    if sparse.issparse(M):
        M = M.toarray()
    M = np.asarray(M, dtype=np.float64)
    return int(np.linalg.matrix_rank(M, tol=tol))


def symmetrise(M: FloatArray) -> FloatArray:
    """Return ``0.5 * (M + M^T)``.

    Args:
        M: Square matrix.

    Returns:
        Symmetrised copy.
    """
    M = np.asarray(M, dtype=np.float64)
    return 0.5 * (M + M.T)


def is_symmetric(M: FloatArray, tol: float = 1e-10) -> bool:
    """Check whether *M* is symmetric within *tol*.

    Args:
        M: Square matrix.
        tol: Tolerance.

    Returns:
        ``True`` if ``||M - M^T||_max < tol``.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    return bool(np.max(np.abs(M - M.T)) < tol)


def is_psd(M: FloatArray, tol: float = -1e-10) -> bool:
    """Check whether *M* is positive semi-definite.

    Args:
        M: Symmetric matrix.
        tol: Minimum eigenvalue threshold (slightly negative to allow
            for numerical noise).

    Returns:
        ``True`` if the minimum eigenvalue is >= *tol*.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    eigenvalues = np.linalg.eigvalsh(M)
    return bool(np.min(eigenvalues) >= tol)


def entropy(p: FloatArray, floor: float = 1e-300) -> float:
    """Shannon entropy ``H(p) = -sum(p_i * log(p_i))``.

    Args:
        p: Probability distribution.
        floor: Floor for zero probabilities.

    Returns:
        Entropy in nats.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    p_safe = np.maximum(p, floor)
    return float(-np.sum(p_safe * np.log(p_safe)))


def privacy_loss_rv(
    p_row: FloatArray,
    q_row: FloatArray,
    floor: float = 1e-300,
) -> Tuple[FloatArray, FloatArray]:
    """Compute the privacy loss random variable for two mechanism rows.

    The privacy loss at output j is ``log(p[j] / q[j])``, and it
    occurs with probability ``p[j]``.

    Args:
        p_row: Mechanism output distribution for database i.
        q_row: Mechanism output distribution for database i'.
        floor: Floor for zero probabilities.

    Returns:
        ``(losses, probs)`` where ``losses[j] = log(p[j]/q[j])`` and
        ``probs[j] = p[j]``.
    """
    p = np.asarray(p_row, dtype=np.float64).ravel()
    q = np.asarray(q_row, dtype=np.float64).ravel()
    p_safe = np.maximum(p, floor)
    q_safe = np.maximum(q, floor)
    losses = np.log(p_safe / q_safe)
    return losses, p_safe / np.sum(p_safe)

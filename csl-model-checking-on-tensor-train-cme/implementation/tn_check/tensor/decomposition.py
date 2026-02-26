"""
Tensor decomposition: converting dense tensors to MPS/MPO format.

Implements:
- TT-SVD decomposition (Oseledets 2011)
- Adaptive SVD with tolerance-based truncation
- Matrix-to-MPO conversion
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from tn_check.tensor.mps import MPS, CanonicalForm
from tn_check.tensor.mpo import MPO

logger = logging.getLogger(__name__)


def svd_truncate(
    matrix: NDArray,
    max_rank: Optional[int] = None,
    tolerance: float = 1e-10,
    relative: bool = True,
) -> tuple[NDArray, NDArray, NDArray, float]:
    """
    Compute truncated SVD of a matrix.

    Args:
        matrix: Input matrix.
        max_rank: Maximum rank.
        tolerance: Truncation tolerance.
        relative: If True, tolerance is relative to the largest singular value.

    Returns:
        Tuple of (U, S, Vt, truncation_error).
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    if len(S) == 0:
        return U, S, Vt, 0.0

    # Determine cutoff
    keep = len(S)

    if max_rank is not None:
        keep = min(keep, max_rank)

    if tolerance > 0:
        if relative:
            tol = tolerance * S[0]
        else:
            tol = tolerance

        # Find smallest rank such that discarded singular values
        # have total Frobenius norm below tolerance
        cumsum_rev = np.cumsum(S[::-1] ** 2)[::-1]
        for j in range(len(S) - 1, 0, -1):
            if np.sqrt(cumsum_rev[j]) <= tol:
                keep = min(keep, j)
            else:
                break

    keep = max(1, keep)

    trunc_error = np.sqrt(np.sum(S[keep:] ** 2)) if keep < len(S) else 0.0

    return U[:, :keep], S[:keep], Vt[:keep, :], trunc_error


def adaptive_svd_truncate(
    matrix: NDArray,
    target_error: float = 1e-10,
    max_rank: Optional[int] = None,
) -> tuple[NDArray, NDArray, NDArray, float, int]:
    """
    Adaptive SVD truncation that finds the optimal rank for a target error.

    Args:
        matrix: Input matrix.
        target_error: Target truncation error.
        max_rank: Maximum allowed rank.

    Returns:
        Tuple of (U, S, Vt, actual_error, rank).
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    if len(S) == 0:
        return U, S, Vt, 0.0, 0

    # Find optimal rank
    total_sq = np.sum(S ** 2)
    cumsum = np.cumsum(S ** 2)

    rank = len(S)
    for k in range(len(S)):
        remaining = total_sq - cumsum[k]
        if np.sqrt(remaining) <= target_error:
            rank = k + 1
            break

    if max_rank is not None:
        rank = min(rank, max_rank)

    rank = max(1, rank)
    actual_error = np.sqrt(np.sum(S[rank:] ** 2)) if rank < len(S) else 0.0

    return U[:, :rank], S[:rank], Vt[:rank, :], actual_error, rank


def tensor_to_mps(
    tensor: NDArray,
    physical_dims: Sequence[int],
    max_bond_dim: Optional[int] = None,
    tolerance: float = 1e-10,
) -> MPS:
    """
    Convert a dense tensor to MPS format using TT-SVD decomposition.

    This is the standard algorithm from Oseledets (2011). The tensor is
    reshaped and decomposed site by site using SVD.

    Args:
        tensor: Dense tensor as a 1D array of length prod(physical_dims),
                or as an ND array with shape physical_dims.
        physical_dims: Physical dimensions.
        max_bond_dim: Maximum bond dimension.
        tolerance: Truncation tolerance for SVD.

    Returns:
        MPS representing the tensor.
    """
    phys = list(physical_dims)
    N = len(phys)

    # Flatten to 1D if needed
    tensor = np.asarray(tensor, dtype=np.float64).ravel()
    expected_size = 1
    for d in phys:
        expected_size *= d

    if tensor.size != expected_size:
        raise ValueError(
            f"Tensor size {tensor.size} != product of physical dims {expected_size}"
        )

    cores = []
    remaining = tensor.copy()
    chi_left = 1

    for k in range(N - 1):
        d = phys[k]
        right_size = 1
        for j in range(k + 1, N):
            right_size *= phys[j]

        # Reshape to matrix: (chi_left * d, right_size)
        mat = remaining.reshape(chi_left * d, right_size)

        # Truncated SVD
        U, S, Vt, trunc_err = svd_truncate(mat, max_rank=max_bond_dim,
                                             tolerance=tolerance)

        chi_right = len(S)

        # Store core
        cores.append(U.reshape(chi_left, d, chi_right))

        # Update remaining tensor
        remaining = np.diag(S) @ Vt
        chi_left = chi_right

    # Last core
    cores.append(remaining.reshape(chi_left, phys[-1], 1))

    mps = MPS(cores, canonical_form=CanonicalForm.LEFT, copy_cores=False)
    return mps


def matrix_to_mpo(
    matrix: NDArray,
    physical_dims_in: Sequence[int],
    physical_dims_out: Optional[Sequence[int]] = None,
    max_bond_dim: Optional[int] = None,
    tolerance: float = 1e-10,
) -> MPO:
    """
    Convert a dense matrix to MPO format.

    The matrix is reshaped into a tensor with interleaved input/output
    indices and then decomposed using TT-SVD.

    Args:
        matrix: Dense matrix.
        physical_dims_in: Input physical dimensions.
        physical_dims_out: Output physical dimensions (defaults to same as input).
        max_bond_dim: Maximum bond dimension.
        tolerance: SVD truncation tolerance.

    Returns:
        MPO representing the matrix.
    """
    phys_in = list(physical_dims_in)
    if physical_dims_out is None:
        phys_out = phys_in.copy()
    else:
        phys_out = list(physical_dims_out)

    N = len(phys_in)
    if len(phys_out) != N:
        raise ValueError("Input and output physical dims must have same length")

    total_in = 1
    total_out = 1
    for d in phys_in:
        total_in *= d
    for d in phys_out:
        total_out *= d

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (total_out, total_in):
        raise ValueError(
            f"Matrix shape {matrix.shape} doesn't match dims ({total_out}, {total_in})"
        )

    # Reshape matrix to tensor with interleaved indices
    # Order: d_out_0, d_in_0, d_out_1, d_in_1, ...
    shape_interleaved = []
    for k in range(N):
        shape_interleaved.append(phys_out[k])
        shape_interleaved.append(phys_in[k])

    # First rearrange: matrix[out_combined, in_combined]
    # -> tensor[out_0, out_1, ..., in_0, in_1, ...]
    tensor = matrix.reshape(list(phys_out) + list(phys_in))

    # Transpose to interleaved: out_0, in_0, out_1, in_1, ...
    perm = []
    for k in range(N):
        perm.append(k)          # out_k
        perm.append(N + k)      # in_k
    tensor = tensor.transpose(perm)
    tensor = tensor.reshape(shape_interleaved)

    # Now decompose using TT-SVD, treating pairs (d_out_k, d_in_k) as single index
    cores = []
    remaining = tensor.ravel()
    chi_left = 1

    for k in range(N - 1):
        d_out = phys_out[k]
        d_in = phys_in[k]
        local_dim = d_out * d_in

        right_size = 1
        for j in range(k + 1, N):
            right_size *= phys_out[j] * phys_in[j]

        mat = remaining.reshape(chi_left * local_dim, right_size)
        U, S, Vt, _ = svd_truncate(mat, max_rank=max_bond_dim, tolerance=tolerance)

        chi_right = len(S)
        core = U.reshape(chi_left, d_out, d_in, chi_right)
        # Reorder to (D_left, d_in, d_out, D_right) for MPO convention
        core = core.transpose(0, 2, 1, 3)
        cores.append(core)

        remaining = np.diag(S) @ Vt
        chi_left = chi_right

    # Last core
    d_out = phys_out[-1]
    d_in = phys_in[-1]
    core = remaining.reshape(chi_left, d_out, d_in, 1)
    core = core.transpose(0, 2, 1, 3)
    cores.append(core)

    return MPO(cores, copy_cores=False)


def randomized_svd(
    matrix: NDArray,
    rank: int,
    oversampling: int = 10,
    n_power_iterations: int = 2,
    seed: Optional[int] = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Randomized SVD for large matrices.

    Uses the Halko-Martinsson-Tropp algorithm.

    Args:
        matrix: Input matrix of shape (m, n).
        rank: Target rank.
        oversampling: Number of oversampling vectors.
        n_power_iterations: Number of power iterations for accuracy.
        seed: Random seed.

    Returns:
        Tuple of (U, S, Vt) with rank components.
    """
    rng = np.random.default_rng(seed)
    m, n = matrix.shape
    k = min(rank + oversampling, min(m, n))

    # Random projection
    Omega = rng.standard_normal((n, k))
    Y = matrix @ Omega

    # Power iterations for better accuracy
    for _ in range(n_power_iterations):
        Y = matrix @ (matrix.T @ Y)

    # QR factorization
    Q, _ = np.linalg.qr(Y, mode="reduced")

    # Project and compute SVD
    B = Q.T @ matrix
    U_B, S, Vt = np.linalg.svd(B, full_matrices=False)

    U = Q @ U_B

    # Truncate to rank
    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]

    return U, S, Vt


def incremental_svd_update(
    U: NDArray,
    S: NDArray,
    Vt: NDArray,
    new_column: NDArray,
    max_rank: Optional[int] = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Incrementally update a truncated SVD with a new column.

    Used for streaming tensor decomposition.

    Args:
        U: Current left singular vectors (m, r).
        S: Current singular values (r,).
        Vt: Current right singular vectors (r, n).
        new_column: New column to add (m,).
        max_rank: Maximum rank to maintain.

    Returns:
        Updated (U, S, Vt).
    """
    m = U.shape[0]
    r = len(S)

    # Project new column
    p = U.T @ new_column
    residual = new_column - U @ p
    res_norm = np.linalg.norm(residual)

    if res_norm < 1e-14:
        # New column is in the span of U
        # Update S and Vt
        K = np.zeros((r, r + 1))
        K[:r, :r] = np.diag(S)
        K[:, -1] = p

        # Extend Vt
        n = Vt.shape[1]
        new_row = np.zeros((1, n + 1))
        new_row[0, -1] = 1.0
        Vt_ext = np.vstack([
            np.hstack([Vt, np.zeros((r, 1))]),
            new_row,
        ])

        U_K, S_new, Vt_K = np.linalg.svd(K, full_matrices=False)
        U_new = U @ U_K
        Vt_new = Vt_K @ Vt_ext
    else:
        # New column has component outside span of U
        q = residual / res_norm

        K = np.zeros((r + 1, r + 1))
        K[:r, :r] = np.diag(S)
        K[:r, -1] = p
        K[-1, -1] = res_norm

        n = Vt.shape[1]
        Vt_ext = np.vstack([
            np.hstack([Vt, np.zeros((r, 1))]),
            np.zeros((1, n + 1)),
        ])
        Vt_ext[-1, -1] = 1.0

        U_K, S_new, Vt_K = np.linalg.svd(K, full_matrices=False)

        U_ext = np.hstack([U, q.reshape(-1, 1)])
        U_new = U_ext @ U_K
        Vt_new = Vt_K @ Vt_ext

    # Truncate
    if max_rank is not None and len(S_new) > max_rank:
        S_new = S_new[:max_rank]
        U_new = U_new[:, :max_rank]
        Vt_new = Vt_new[:max_rank, :]

    return U_new, S_new, Vt_new

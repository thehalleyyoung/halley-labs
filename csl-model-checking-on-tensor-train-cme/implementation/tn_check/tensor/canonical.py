"""
Canonical forms and SVD-based compression for MPS.

Implements:
- Left canonicalization via QR decomposition
- Right canonicalization via QR decomposition
- Mixed canonical form
- SVD-based rounding (compression) with truncation
- Orthogonality center manipulation
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from tn_check.tensor.mps import MPS, CanonicalForm

logger = logging.getLogger(__name__)


def qr_left_sweep(mps: MPS, start: int = 0, end: Optional[int] = None) -> MPS:
    """
    Perform a left-to-right QR sweep to left-canonicalize sites [start, end).

    After the sweep, sites [start, end) are left-isometric:
    sum_{d, chi_l} A*[chi_l, d, chi_r'] A[chi_l, d, chi_r] = delta_{chi_r', chi_r}

    Args:
        mps: Input MPS (modified in place).
        start: Starting site (inclusive).
        end: Ending site (exclusive). Defaults to num_sites - 1.

    Returns:
        Modified MPS (same object).
    """
    if end is None:
        end = mps.num_sites - 1

    end = min(end, mps.num_sites - 1)

    for k in range(start, end):
        core = mps.cores[k]
        chi_l, d, chi_r = core.shape

        # Reshape to matrix: (chi_l * d, chi_r)
        mat = core.reshape(chi_l * d, chi_r)

        # QR decomposition
        Q, R = np.linalg.qr(mat, mode="reduced")

        new_chi = Q.shape[1]

        # Update current core
        mps.cores[k] = Q.reshape(chi_l, d, new_chi)

        # Absorb R into next core
        next_core = mps.cores[k + 1]
        chi_l_next, d_next, chi_r_next = next_core.shape
        mat_next = R @ next_core.reshape(chi_r, d_next * chi_r_next)
        mps.cores[k + 1] = mat_next.reshape(new_chi, d_next, chi_r_next)

    mps.invalidate_cache()
    return mps


def qr_right_sweep(mps: MPS, start: Optional[int] = None, end: int = 0) -> MPS:
    """
    Perform a right-to-left QR sweep to right-canonicalize sites (end, start].

    After the sweep, sites (end, start] are right-isometric:
    sum_{d, chi_r} A[chi_l, d, chi_r] A*[chi_l', d, chi_r] = delta_{chi_l, chi_l'}

    Args:
        mps: Input MPS (modified in place).
        start: Starting site (inclusive). Defaults to num_sites - 1.
        end: Ending site (exclusive).

    Returns:
        Modified MPS (same object).
    """
    if start is None:
        start = mps.num_sites - 1

    end = max(end, 0)

    for k in range(start, end, -1):
        core = mps.cores[k]
        chi_l, d, chi_r = core.shape

        # Reshape to matrix: (chi_l, d * chi_r)
        mat = core.reshape(chi_l, d * chi_r)

        # QR on the transpose (equivalent to RQ decomposition)
        Q, R = np.linalg.qr(mat.T, mode="reduced")
        # Q^T gives the right-isometric core, R^T is absorbed left

        new_chi = Q.shape[1]

        # Update current core (right-isometric)
        mps.cores[k] = Q.T.reshape(new_chi, d, chi_r)

        # Absorb R^T into previous core
        prev_core = mps.cores[k - 1]
        chi_l_prev, d_prev, chi_r_prev = prev_core.shape
        mat_prev = prev_core.reshape(chi_l_prev * d_prev, chi_r_prev) @ R.T
        mps.cores[k - 1] = mat_prev.reshape(chi_l_prev, d_prev, new_chi)

    mps.invalidate_cache()
    return mps


def left_canonicalize(mps: MPS) -> MPS:
    """
    Bring the MPS into left-canonical form.

    All sites except the last are left-isometric. The norm of the MPS
    is concentrated in the last core.

    Args:
        mps: Input MPS (modified in place).

    Returns:
        Left-canonical MPS.
    """
    qr_left_sweep(mps, start=0, end=mps.num_sites - 1)
    mps.canonical_form = CanonicalForm.LEFT
    mps.orthogonality_center = mps.num_sites - 1
    return mps


def right_canonicalize(mps: MPS) -> MPS:
    """
    Bring the MPS into right-canonical form.

    All sites except the first are right-isometric. The norm is
    concentrated in the first core.

    Args:
        mps: Input MPS (modified in place).

    Returns:
        Right-canonical MPS.
    """
    qr_right_sweep(mps, start=mps.num_sites - 1, end=0)
    mps.canonical_form = CanonicalForm.RIGHT
    mps.orthogonality_center = 0
    return mps


def mixed_canonicalize(mps: MPS, center: int) -> MPS:
    """
    Bring the MPS into mixed-canonical form with orthogonality center at `center`.

    Sites to the left of center are left-isometric.
    Sites to the right of center are right-isometric.
    The orthogonality center contains all the non-trivial information.

    Args:
        mps: Input MPS (modified in place).
        center: Site index for the orthogonality center.

    Returns:
        Mixed-canonical MPS.
    """
    if center < 0 or center >= mps.num_sites:
        raise ValueError(f"Center {center} out of range [0, {mps.num_sites})")

    # Left sweep from 0 to center
    qr_left_sweep(mps, start=0, end=center)

    # Right sweep from end to center
    qr_right_sweep(mps, start=mps.num_sites - 1, end=center)

    mps.canonical_form = CanonicalForm.MIXED
    mps.orthogonality_center = center
    return mps


def svd_compress(
    mps: MPS,
    max_bond_dim: Optional[int] = None,
    tolerance: float = 1e-10,
    normalize: bool = False,
    relative: bool = True,
) -> tuple[MPS, float]:
    """
    Compress an MPS using SVD-based rounding.

    Algorithm:
    1. Right-to-left QR sweep (right-canonicalize).
    2. Left-to-right SVD sweep with truncation.

    This is the standard TT-rounding algorithm (Oseledets 2011).

    Args:
        mps: Input MPS.
        max_bond_dim: Maximum bond dimension after compression.
        tolerance: Truncation tolerance.
        normalize: If True, normalize the result to unit norm.
        relative: If True, use relative tolerance (scaled by norm).

    Returns:
        Tuple of (compressed MPS, total truncation error).
    """
    result = mps.copy()
    N = result.num_sites

    if N <= 1:
        return result, 0.0

    # Step 1: Right-to-left QR sweep (right-canonicalize)
    right_canonicalize(result)

    # Compute norm for relative tolerance
    if relative:
        # After right-canonicalization, norm is in first core
        norm_val = np.linalg.norm(result.cores[0])
        if norm_val < 1e-300:
            return result, 0.0
        effective_tol = tolerance * norm_val
    else:
        effective_tol = tolerance

    total_error_sq = 0.0

    # Step 2: Left-to-right SVD sweep with truncation
    for k in range(N - 1):
        core = result.cores[k]
        chi_l, d, chi_r = core.shape

        # Reshape to matrix: (chi_l * d, chi_r)
        mat = core.reshape(chi_l * d, chi_r)

        # SVD
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)

        # Determine number of singular values to keep
        keep = len(S)

        # Bond dimension constraint
        if max_bond_dim is not None:
            keep = min(keep, max_bond_dim)

        # Tolerance-based truncation
        if effective_tol > 0 and len(S) > 1:
            # Truncate smallest singular values whose squared sum < tol^2
            cumsum_rev = np.cumsum(S[::-1] ** 2)
            # Find how many to discard
            n_discard = 0
            for j in range(len(cumsum_rev)):
                if cumsum_rev[j] <= effective_tol ** 2 / (N - 1):
                    n_discard = j + 1
                else:
                    break
            keep = min(keep, max(1, len(S) - n_discard))

        # Compute truncation error
        if keep < len(S):
            trunc_err_sq = np.sum(S[keep:] ** 2)
            total_error_sq += trunc_err_sq

        # Truncate
        S = S[:keep]
        U = U[:, :keep]
        Vt = Vt[:keep, :]

        # Update current core (left-isometric)
        result.cores[k] = U.reshape(chi_l, d, keep)

        # Absorb S @ Vt into next core
        SV = np.diag(S) @ Vt
        next_core = result.cores[k + 1]
        chi_l_next, d_next, chi_r_next = next_core.shape
        new_next = SV @ next_core.reshape(chi_r, d_next * chi_r_next)
        result.cores[k + 1] = new_next.reshape(keep, d_next, chi_r_next)

    total_error = np.sqrt(total_error_sq)
    result.truncation_error_accumulated = mps.truncation_error_accumulated + total_error

    result.canonical_form = CanonicalForm.LEFT
    result.orthogonality_center = N - 1

    if normalize:
        norm_val = np.linalg.norm(result.cores[-1])
        if norm_val > 1e-300:
            result.cores[-1] /= norm_val

    result.invalidate_cache()
    return result, total_error


def svd_truncate_bond(
    mps: MPS,
    bond: int,
    max_bond_dim: Optional[int] = None,
    tolerance: float = 1e-10,
) -> tuple[MPS, float]:
    """
    Truncate a single bond of the MPS via SVD.

    Args:
        mps: Input MPS (modified in place).
        bond: Bond index to truncate.
        max_bond_dim: Maximum bond dimension.
        tolerance: Truncation tolerance.

    Returns:
        Tuple of (modified MPS, truncation error at this bond).
    """
    if bond < 0 or bond >= mps.num_sites - 1:
        raise ValueError(f"Bond {bond} out of range [0, {mps.num_sites - 2}]")

    core_left = mps.cores[bond]
    core_right = mps.cores[bond + 1]
    chi_l, d_l, chi_m = core_left.shape
    chi_m2, d_r, chi_r = core_right.shape

    # Form two-site tensor
    two_site = np.einsum("ijk,klm->ijlm", core_left, core_right)
    mat = two_site.reshape(chi_l * d_l, d_r * chi_r)

    # SVD
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)

    keep = len(S)
    if max_bond_dim is not None:
        keep = min(keep, max_bond_dim)

    if tolerance > 0 and len(S) > 1:
        cumsum_rev = np.cumsum(S[::-1] ** 2)
        n_discard = 0
        for j in range(len(cumsum_rev)):
            if cumsum_rev[j] <= tolerance ** 2:
                n_discard = j + 1
            else:
                break
        keep = min(keep, max(1, len(S) - n_discard))

    trunc_error = np.sqrt(np.sum(S[keep:] ** 2)) if keep < len(S) else 0.0

    S = S[:keep]
    U = U[:, :keep]
    Vt = Vt[:keep, :]

    # Split: left core gets U, right core gets S @ Vt
    mps.cores[bond] = U.reshape(chi_l, d_l, keep)
    SV = np.diag(S) @ Vt
    mps.cores[bond + 1] = SV.reshape(keep, d_r, chi_r)

    mps.invalidate_cache()
    return mps, trunc_error


def move_orthogonality_center(
    mps: MPS,
    current: int,
    target: int,
) -> MPS:
    """
    Move the orthogonality center from `current` to `target` using QR.

    Args:
        mps: MPS in mixed-canonical form.
        current: Current orthogonality center.
        target: Target orthogonality center.

    Returns:
        MPS with updated orthogonality center.
    """
    if target == current:
        return mps

    if target > current:
        # Move right: left-canonicalize sites current to target-1
        qr_left_sweep(mps, start=current, end=target)
    else:
        # Move left: right-canonicalize sites target+1 to current
        qr_right_sweep(mps, start=current, end=target)

    mps.orthogonality_center = target
    mps.canonical_form = CanonicalForm.MIXED
    mps.invalidate_cache()
    return mps


def normalize_mps(mps: MPS) -> tuple[MPS, float]:
    """
    Normalize the MPS to unit 2-norm.

    Brings to left-canonical form first, then normalizes the last core.

    Args:
        mps: Input MPS.

    Returns:
        Tuple of (normalized MPS, original norm).
    """
    result = mps.copy()
    left_canonicalize(result)

    norm = np.linalg.norm(result.cores[-1])
    if norm > 1e-300:
        result.cores[-1] /= norm

    result._cached_norm = 1.0
    result._norm_valid = True

    return result, norm


def gauge_transform(
    mps: MPS,
    bond: int,
    matrix: NDArray,
    matrix_inv: Optional[NDArray] = None,
) -> MPS:
    """
    Apply a gauge transformation at a bond.

    Inserts X between sites bond and bond+1:
    A_{bond} -> A_{bond} @ X
    A_{bond+1} -> X^{-1} @ A_{bond+1}

    This does not change the physical state represented by the MPS.

    Args:
        mps: Input MPS (modified in place).
        bond: Bond index.
        matrix: Gauge transformation matrix X.
        matrix_inv: Inverse of X (computed if not provided).

    Returns:
        Gauge-transformed MPS.
    """
    if bond < 0 or bond >= mps.num_sites - 1:
        raise ValueError(f"Bond {bond} out of range")

    if matrix_inv is None:
        matrix_inv = np.linalg.inv(matrix)

    core_left = mps.cores[bond]
    core_right = mps.cores[bond + 1]
    chi_l, d_l, chi_m = core_left.shape
    chi_m2, d_r, chi_r = core_right.shape

    # Apply gauge: left core absorbs X
    new_left = np.einsum("ijk,kl->ijl", core_left, matrix)
    mps.cores[bond] = new_left

    # Right core absorbs X^{-1}
    new_right = np.einsum("ij,jkl->ikl", matrix_inv, core_right)
    mps.cores[bond + 1] = new_right

    mps.canonical_form = CanonicalForm.NONE
    mps.invalidate_cache()
    return mps


def schmidt_decomposition(
    mps: MPS,
    bond: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Compute the Schmidt decomposition at a bond.

    Brings the MPS into mixed canonical form with center at bond,
    then performs SVD to get the Schmidt coefficients.

    Args:
        mps: Input MPS.
        bond: Bond index.

    Returns:
        Tuple of (U, S, Vt) where:
        - U: Left Schmidt vectors as MPS cores (left of bond)
        - S: Schmidt coefficients (singular values)
        - Vt: Right Schmidt vectors as MPS cores (right of bond)
    """
    work = mps.copy()
    mixed_canonicalize(work, bond)

    core = work.cores[bond]
    chi_l, d, chi_r = core.shape
    mat = core.reshape(chi_l * d, chi_r)

    U, S, Vt = np.linalg.svd(mat, full_matrices=False)

    return U, S, Vt


def two_site_tensor(mps: MPS, bond: int) -> NDArray:
    """
    Form the two-site tensor at a bond.

    theta[chi_l, d_left, d_right, chi_r] = A_left @ A_right

    Args:
        mps: Input MPS.
        bond: Bond index (left site of the pair).

    Returns:
        Two-site tensor of shape (chi_l, d_left, d_right, chi_r).
    """
    left = mps.cores[bond]
    right = mps.cores[bond + 1]
    return np.einsum("ijk,klm->ijlm", left, right)


def split_two_site_tensor(
    theta: NDArray,
    max_bond_dim: Optional[int] = None,
    tolerance: float = 1e-10,
    absorb: str = "right",
) -> tuple[NDArray, NDArray, NDArray, float]:
    """
    Split a two-site tensor back into two MPS cores using SVD.

    Args:
        theta: Two-site tensor of shape (chi_l, d_l, d_r, chi_r).
        max_bond_dim: Maximum bond dimension.
        tolerance: Truncation tolerance.
        absorb: Where to absorb singular values: "left", "right", or "both".

    Returns:
        Tuple of (left_core, singular_values, right_core, truncation_error).
    """
    chi_l, d_l, d_r, chi_r = theta.shape
    mat = theta.reshape(chi_l * d_l, d_r * chi_r)

    U, S, Vt = np.linalg.svd(mat, full_matrices=False)

    keep = len(S)
    if max_bond_dim is not None:
        keep = min(keep, max_bond_dim)

    if tolerance > 0 and len(S) > 1:
        cumsum_rev = np.cumsum(S[::-1] ** 2)
        n_discard = 0
        for j in range(len(cumsum_rev)):
            if cumsum_rev[j] <= tolerance ** 2:
                n_discard = j + 1
            else:
                break
        keep = min(keep, max(1, len(S) - n_discard))

    trunc_error = np.sqrt(np.sum(S[keep:] ** 2)) if keep < len(S) else 0.0

    S = S[:keep]
    U = U[:, :keep]
    Vt = Vt[:keep, :]

    if absorb == "right":
        left_core = U.reshape(chi_l, d_l, keep)
        right_core = (np.diag(S) @ Vt).reshape(keep, d_r, chi_r)
    elif absorb == "left":
        left_core = (U @ np.diag(S)).reshape(chi_l, d_l, keep)
        right_core = Vt.reshape(keep, d_r, chi_r)
    else:  # "both"
        sqrt_S = np.sqrt(S)
        left_core = (U @ np.diag(sqrt_S)).reshape(chi_l, d_l, keep)
        right_core = (np.diag(sqrt_S) @ Vt).reshape(keep, d_r, chi_r)

    return left_core, S, right_core, trunc_error

"""
Higher-level MPS/MPO algebraic operations.

Implements:
- Kronecker product of MPOs
- MPO summation with compression
- MPO transpose / Hermitian conjugate
- MPO trace
- MPS outer product
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from tn_check.tensor.mps import MPS, CanonicalForm
from tn_check.tensor.mpo import MPO

logger = logging.getLogger(__name__)


def kronecker_product_mpo(mpo_list: list[MPO]) -> MPO:
    """
    Form the Kronecker product of a list of MPOs.

    If each MPO_k acts on n_k sites, the result acts on sum(n_k) sites
    with the Kronecker product structure preserved.

    This is fundamental for CME generator construction: each reaction
    contributes a Kronecker product of single-site operators.

    Args:
        mpo_list: List of MPOs to combine.

    Returns:
        Combined MPO acting on all sites.
    """
    if not mpo_list:
        raise ValueError("Must provide at least one MPO")

    # Simply concatenate all cores
    all_cores = []
    for mpo in mpo_list:
        all_cores.extend([c.copy() for c in mpo.cores])

    # Fix bond dimensions at boundaries between MPOs
    # Each sub-MPO starts with bond dim 1 and ends with bond dim 1
    # We need to merge these: the right boundary of one MPO connects
    # to the left boundary of the next

    # Actually, for true Kronecker product, each sub-MPO's boundary
    # bond dims are 1, so concatenation works directly if we handle
    # the intermediate bonds.

    # For a Kronecker product A ⊗ B, the cores are simply concatenated:
    # [A_1, A_2, ..., A_m, B_1, B_2, ..., B_n]
    # with bond dim 1 between A_m and B_1

    return MPO(all_cores, copy_cores=False)


def sum_mpo(
    mpo_list: list[MPO],
    weights: Optional[Sequence[float]] = None,
    compress: bool = False,
    max_bond_dim: Optional[int] = None,
    tolerance: float = 1e-12,
) -> MPO:
    """
    Sum multiple MPOs: result = sum_j w_j * O_j.

    Bond dimensions grow additively. Optional compression afterwards.

    This is the main entry point for constructing the CME generator
    as a sum of reaction contributions.

    Args:
        mpo_list: List of MPOs to sum.
        weights: Optional weights. Defaults to all ones.
        compress: If True, compress the result.
        max_bond_dim: Max bond dim for compression.
        tolerance: Compression tolerance.

    Returns:
        Sum MPO.
    """
    if not mpo_list:
        raise ValueError("Must provide at least one MPO")

    if weights is not None and len(weights) != len(mpo_list):
        raise ValueError("Number of weights must match number of MPOs")

    from tn_check.tensor.operations import mpo_addition, mpo_scalar_multiply

    result = mpo_list[0].copy()
    if weights is not None:
        result.scale(weights[0])

    for i in range(1, len(mpo_list)):
        mpo_i = mpo_list[i]
        if weights is not None:
            mpo_i = mpo_scalar_multiply(mpo_i, weights[i])
        result = mpo_addition(result, mpo_i)

    if compress and (max_bond_dim is not None or tolerance > 0):
        result.compress(max_bond_dim=max_bond_dim, tolerance=tolerance)

    return result


def mpo_transpose(mpo: MPO) -> MPO:
    """
    Transpose an MPO: swap input and output physical indices.

    Args:
        mpo: Input MPO.

    Returns:
        Transposed MPO.
    """
    cores = []
    for k in range(mpo.num_sites):
        core = mpo.cores[k]  # (D_l, d_in, d_out, D_r)
        # Swap d_in and d_out
        cores.append(core.transpose(0, 2, 1, 3).copy())

    return MPO(cores, copy_cores=False)


def mpo_hermitian_conjugate(mpo: MPO) -> MPO:
    """
    Hermitian conjugate of an MPO: transpose + complex conjugate.

    For real MPOs, this is just the transpose.

    Args:
        mpo: Input MPO.

    Returns:
        Hermitian conjugate MPO.
    """
    cores = []
    for k in range(mpo.num_sites):
        core = mpo.cores[k]
        # Swap d_in and d_out, take conjugate
        cores.append(np.conj(core.transpose(0, 2, 1, 3)).copy())

    return MPO(cores, copy_cores=False)


def mpo_trace(mpo: MPO) -> float:
    """
    Compute the trace of a square MPO.

    Tr(O) = sum_{i_1,...,i_N} O[i_1,...,i_N; i_1,...,i_N]

    For each site, trace over the physical indices, then contract bonds.

    Args:
        mpo: Square MPO.

    Returns:
        Trace value.
    """
    if not mpo.is_square:
        raise ValueError("Can only trace square MPOs")

    # At each site, trace over physical indices
    env = np.ones((1,), dtype=np.float64)

    for k in range(mpo.num_sites):
        core = mpo.cores[k]  # (D_l, d_in, d_out, D_r)
        # Trace over d_in = d_out
        traced = np.einsum("aiia->a", core) if core.shape[1] == core.shape[2] else None
        if traced is None:
            d_in, d_out = core.shape[1], core.shape[2]
            d = min(d_in, d_out)
            traced = sum(core[:, i, i, :] for i in range(d))
            # traced shape: (D_l, D_r)
            env_new = np.einsum("a,ab->b", env, traced.reshape(-1, core.shape[3]))
            env = env_new
            continue

        # traced has wrong shape, let me redo
        core_traced = np.zeros((core.shape[0], core.shape[3]), dtype=core.dtype)
        d = core.shape[1]
        for i in range(d):
            core_traced += core[:, i, i, :]
        # core_traced shape: (D_l, D_r)
        env = env @ core_traced

    return float(env[0])


def mps_outer_product(mps_a: MPS, mps_b: MPS) -> MPO:
    """
    Compute the outer product |a><b| as an MPO.

    The resulting MPO has O[i,j] = a[i] * b[j].
    Bond dimension of result = chi_a * chi_b.

    Args:
        mps_a: Ket MPS.
        mps_b: Bra MPS.

    Returns:
        MPO representing |a><b|.
    """
    if mps_a.num_sites != mps_b.num_sites:
        raise ValueError("MPS must have same number of sites")

    N = mps_a.num_sites
    cores = []

    for k in range(N):
        a = mps_a.cores[k]  # (chi_a_l, d_a, chi_a_r)
        b = mps_b.cores[k]  # (chi_b_l, d_b, chi_b_r)
        chi_a_l, d_a, chi_a_r = a.shape
        chi_b_l, d_b, chi_b_r = b.shape

        # MPO core: (chi_a_l * chi_b_l, d_b, d_a, chi_a_r * chi_b_r)
        mpo_core = np.zeros(
            (chi_a_l * chi_b_l, d_b, d_a, chi_a_r * chi_b_r),
            dtype=np.float64,
        )

        for i_out in range(d_a):
            for i_in in range(d_b):
                # Kronecker product of a[:,i_out,:] and b[:,i_in,:]
                mat_a = a[:, i_out, :]
                mat_b = b[:, i_in, :]
                mpo_core[:, i_in, i_out, :] = np.kron(mat_a, mat_b)

        cores.append(mpo_core)

    return MPO(cores, copy_cores=False)


def mpo_hadamard_product(mpo_a: MPO, mpo_b: MPO) -> MPO:
    """
    Element-wise (Hadamard) product of two MPOs.

    Result[i,j] = A[i,j] * B[i,j]

    Bond dimension of result = D_A * D_B.

    Args:
        mpo_a: First MPO.
        mpo_b: Second MPO.

    Returns:
        Hadamard product MPO.
    """
    if mpo_a.num_sites != mpo_b.num_sites:
        raise ValueError("MPOs must have same number of sites")

    N = mpo_a.num_sites
    cores = []

    for k in range(N):
        a = mpo_a.cores[k]  # (D_a_l, d_in, d_out, D_a_r)
        b = mpo_b.cores[k]  # (D_b_l, d_in, d_out, D_b_r)
        D_a_l, d_in, d_out, D_a_r = a.shape
        D_b_l, _, _, D_b_r = b.shape

        new_core = np.zeros(
            (D_a_l * D_b_l, d_in, d_out, D_a_r * D_b_r),
            dtype=np.float64,
        )

        for i in range(d_in):
            for j in range(d_out):
                mat_a = a[:, i, j, :]
                mat_b = b[:, i, j, :]
                new_core[:, i, j, :] = np.kron(mat_a, mat_b)

        cores.append(new_core)

    return MPO(cores, copy_cores=False)


def mpo_power(
    mpo: MPO,
    power: int,
    max_bond_dim: Optional[int] = None,
    tolerance: float = 1e-10,
) -> MPO:
    """
    Compute MPO raised to a power: O^n.

    Uses repeated squaring for efficiency.

    Args:
        mpo: Input MPO.
        power: Exponent (non-negative integer).
        max_bond_dim: Max bond dim for intermediate compression.
        tolerance: Compression tolerance.

    Returns:
        MPO^power.
    """
    from tn_check.tensor.mpo import identity_mpo
    from tn_check.tensor.operations import mpo_mpo_contraction

    if power == 0:
        return identity_mpo(mpo.num_sites, mpo.physical_dims_in)
    if power == 1:
        return mpo.copy()

    # Repeated squaring
    result = identity_mpo(mpo.num_sites, mpo.physical_dims_in)
    base = mpo.copy()

    n = power
    while n > 0:
        if n % 2 == 1:
            result = mpo_mpo_contraction(result, base)
            if max_bond_dim is not None:
                result.compress(max_bond_dim=max_bond_dim, tolerance=tolerance)
        base = mpo_mpo_contraction(base, base)
        if max_bond_dim is not None:
            base.compress(max_bond_dim=max_bond_dim, tolerance=tolerance)
        n //= 2

    return result


def apply_diagonal_mask_to_mpo(
    mpo: MPO,
    mask_mps: MPS,
) -> MPO:
    """
    Apply a diagonal mask to an MPO: result[i,j] = mask[i] * O[i,j].

    This zeros out rows of the operator for states not in the mask.
    Used for constructing the projected rate matrix for CSL until operator.

    The mask MPS should be rank-1 (characteristic tensor) for axis-aligned
    predicates, preserving the MPO bond dimension.

    Args:
        mpo: Input MPO.
        mask_mps: Diagonal mask as MPS (0/1 values).

    Returns:
        Masked MPO.
    """
    if mpo.num_sites != mask_mps.num_sites:
        raise ValueError("MPO and mask must have same number of sites")

    N = mpo.num_sites
    cores = []

    for k in range(N):
        W = mpo.cores[k]  # (D_l, d_in, d_out, D_r)
        M = mask_mps.cores[k]  # (chi_l, d_out, chi_r)
        D_l, d_in, d_out, D_r = W.shape
        chi_l, _, chi_r = M.shape

        # Apply mask to output (row) index
        # new_core[D_l*chi_l, d_in, d_out, D_r*chi_r]
        new_core = np.zeros(
            (D_l * chi_l, d_in, d_out, D_r * chi_r), dtype=np.float64
        )

        for i_out in range(d_out):
            for i_in in range(d_in):
                # W[:, i_in, i_out, :] is (D_l, D_r)
                # M[:, i_out, :] is (chi_l, chi_r)
                new_core[:, i_in, i_out, :] = np.kron(
                    W[:, i_in, i_out, :],
                    M[:, i_out, :],
                )

        cores.append(new_core)

    return MPO(cores, copy_cores=False)


def apply_column_mask_to_mpo(
    mpo: MPO,
    mask_mps: MPS,
) -> MPO:
    """
    Apply a column mask to an MPO: result[i,j] = O[i,j] * mask[j].

    Zeros out columns for states not in the mask.

    Args:
        mpo: Input MPO.
        mask_mps: Column mask as MPS.

    Returns:
        Column-masked MPO.
    """
    if mpo.num_sites != mask_mps.num_sites:
        raise ValueError("MPO and mask must have same number of sites")

    N = mpo.num_sites
    cores = []

    for k in range(N):
        W = mpo.cores[k]
        M = mask_mps.cores[k]
        D_l, d_in, d_out, D_r = W.shape
        chi_l, _, chi_r = M.shape

        new_core = np.zeros(
            (D_l * chi_l, d_in, d_out, D_r * chi_r), dtype=np.float64
        )

        for i_out in range(d_out):
            for i_in in range(d_in):
                new_core[:, i_in, i_out, :] = np.kron(
                    W[:, i_in, i_out, :],
                    M[:, i_in, :],
                )

        cores.append(new_core)

    return MPO(cores, copy_cores=False)


def project_mpo_to_subspace(
    mpo: MPO,
    row_mask: MPS,
    col_mask: MPS,
) -> MPO:
    """
    Project MPO to a subspace: result[i,j] = row_mask[i] * O[i,j] * col_mask[j].

    Used for constructing the projected rate matrix Q_{Phi1 & !Phi2}
    in CSL time-bounded until.

    Args:
        mpo: Input MPO.
        row_mask: Row mask (which output states to keep).
        col_mask: Column mask (which input states to keep).

    Returns:
        Projected MPO.
    """
    result = apply_diagonal_mask_to_mpo(mpo, row_mask)
    result = apply_column_mask_to_mpo(result, col_mask)
    return result

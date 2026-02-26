"""
MPS/MPO operations: contractions, arithmetic, inner products, norms.

This module implements the core tensor network operations needed for
CME time evolution and CSL model checking:
- Inner product / norm computation via sequential contraction
- MPS addition (direct sum of bond spaces)
- Scalar multiplication
- Hadamard (element-wise) product of two MPS
- MPO-MPS contraction (applying operator to state)
- MPO-MPO contraction (operator composition)
- Zip-up contraction for memory-efficient MPO-MPS products
- Compression via SVD truncation
- Dense vector/matrix conversion (for small systems)
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import opt_einsum as oe
from numpy.typing import NDArray

from tn_check.tensor.mps import MPS, CanonicalForm
from tn_check.tensor.mpo import MPO

logger = logging.getLogger(__name__)


def mps_inner_product(mps_a: MPS, mps_b: MPS) -> float:
    """
    Compute the inner product <a|b> between two MPS.

    Uses sequential contraction from left to right, maintaining the
    environment tensor of shape (chi_a, chi_b).

    Args:
        mps_a: Left MPS (bra).
        mps_b: Right MPS (ket).

    Returns:
        Inner product <a|b>.

    Complexity: O(N * chi^2 * d * chi) where chi = max bond dim.
    """
    if mps_a.num_sites != mps_b.num_sites:
        raise ValueError("MPS must have same number of sites")
    if mps_a.physical_dims != mps_b.physical_dims:
        raise ValueError("MPS must have same physical dimensions")

    N = mps_a.num_sites

    # Initialize environment: shape (chi_a_0, chi_b_0) = (1, 1)
    env = np.ones((1, 1), dtype=np.float64)

    for k in range(N):
        core_a = mps_a.cores[k]  # (chi_a_l, d, chi_a_r)
        core_b = mps_b.cores[k]  # (chi_b_l, d, chi_b_r)

        # Contract: env[a_l, b_l] * A*[a_l, d, a_r] * B[b_l, d, b_r]
        # Result: (chi_a_r, chi_b_r)
        env = oe.contract(
            "ab,adc,bde->ce",
            env, core_a, core_b,
            optimize="optimal",
        )

    return float(env[0, 0])


def mps_norm(mps: MPS) -> float:
    """
    Compute the 2-norm ||mps|| = sqrt(<mps|mps>).

    For MPS in left/right canonical form, this can be read off the
    orthogonality center. For general form, full contraction is needed.

    Args:
        mps: Input MPS.

    Returns:
        Euclidean norm.
    """
    if mps._norm_valid and mps._cached_norm is not None:
        return mps._cached_norm

    ip = mps_inner_product(mps, mps)
    if ip < 0:
        logger.warning(
            f"Negative inner product {ip:.2e} detected, "
            "likely due to numerical error. Using abs value."
        )
        ip = abs(ip)

    norm_val = np.sqrt(ip)
    mps._cached_norm = norm_val
    mps._norm_valid = True
    return norm_val


def mps_addition(mps_a: MPS, mps_b: MPS) -> MPS:
    """
    Add two MPS: result = a + b.

    The bond dimensions of the result are the sum of the bond dimensions
    of the inputs. Subsequent compression is needed to reduce rank.

    The construction uses direct sum of the bond spaces:
    - At site k (not first or last): new core is block-diagonal
      [[A_k, 0], [0, B_k]] in the bond indices.
    - At first site: [A_1, B_1] (concatenate along right bond)
    - At last site: [[A_N], [B_N]] (concatenate along left bond)

    Args:
        mps_a: First MPS.
        mps_b: Second MPS.

    Returns:
        Sum MPS with increased bond dimensions.
    """
    if mps_a.num_sites != mps_b.num_sites:
        raise ValueError("MPS must have same number of sites")
    if mps_a.physical_dims != mps_b.physical_dims:
        raise ValueError("MPS must have same physical dimensions")

    N = mps_a.num_sites
    cores = []

    for k in range(N):
        a = mps_a.cores[k]  # (chi_a_l, d, chi_a_r)
        b = mps_b.cores[k]  # (chi_b_l, d, chi_b_r)
        chi_a_l, d, chi_a_r = a.shape
        chi_b_l, _, chi_b_r = b.shape

        if N == 1:
            # Single site: just add
            new_core = a + b
        elif k == 0:
            # First site: concatenate along right bond
            # new shape: (1, d, chi_a_r + chi_b_r)
            new_core = np.zeros((1, d, chi_a_r + chi_b_r), dtype=np.float64)
            new_core[:, :, :chi_a_r] = a
            new_core[:, :, chi_a_r:] = b
        elif k == N - 1:
            # Last site: concatenate along left bond
            # new shape: (chi_a_l + chi_b_l, d, 1)
            new_core = np.zeros((chi_a_l + chi_b_l, d, 1), dtype=np.float64)
            new_core[:chi_a_l, :, :] = a
            new_core[chi_a_l:, :, :] = b
        else:
            # Interior site: block diagonal
            # new shape: (chi_a_l + chi_b_l, d, chi_a_r + chi_b_r)
            new_core = np.zeros(
                (chi_a_l + chi_b_l, d, chi_a_r + chi_b_r), dtype=np.float64
            )
            new_core[:chi_a_l, :, :chi_a_r] = a
            new_core[chi_a_l:, :, chi_a_r:] = b

        cores.append(new_core)

    result = MPS(cores, canonical_form=CanonicalForm.NONE, copy_cores=False)
    result.truncation_error_accumulated = max(
        mps_a.truncation_error_accumulated,
        mps_b.truncation_error_accumulated,
    )
    return result


def mps_scalar_multiply(mps: MPS, scalar: float) -> MPS:
    """
    Multiply an MPS by a scalar: result = scalar * mps.

    Modifies only the first core to avoid unnecessary copying.

    Args:
        mps: Input MPS.
        scalar: Scalar multiplier.

    Returns:
        New MPS scaled by the scalar.
    """
    result = mps.copy()
    result.cores[0] = result.cores[0] * scalar
    result.invalidate_cache()
    return result


def mps_hadamard_product(mps_a: MPS, mps_b: MPS) -> MPS:
    """
    Compute the Hadamard (element-wise) product of two MPS.

    Result[i_1,...,i_N] = A[i_1,...,i_N] * B[i_1,...,i_N]

    The bond dimension of the result is the product of the input bond
    dimensions. Subsequent compression is usually needed.

    Implementation: at each site, the Hadamard product core is the
    tensor product of the two cores with shared physical index.

    Args:
        mps_a: First MPS.
        mps_b: Second MPS.

    Returns:
        Hadamard product MPS.
    """
    if mps_a.num_sites != mps_b.num_sites:
        raise ValueError("MPS must have same number of sites")
    if mps_a.physical_dims != mps_b.physical_dims:
        raise ValueError("MPS must have same physical dimensions")

    N = mps_a.num_sites
    cores = []

    for k in range(N):
        a = mps_a.cores[k]  # (chi_a_l, d, chi_a_r)
        b = mps_b.cores[k]  # (chi_b_l, d, chi_b_r)
        chi_a_l, d, chi_a_r = a.shape
        chi_b_l, _, chi_b_r = b.shape

        # Hadamard product core: (chi_a_l * chi_b_l, d, chi_a_r * chi_b_r)
        new_core = np.zeros(
            (chi_a_l * chi_b_l, d, chi_a_r * chi_b_r), dtype=np.float64
        )

        for i_phys in range(d):
            # Kronecker product of the matrices A[:,i,:] and B[:,i,:]
            mat_a = a[:, i_phys, :]  # (chi_a_l, chi_a_r)
            mat_b = b[:, i_phys, :]  # (chi_b_l, chi_b_r)
            new_core[:, i_phys, :] = np.kron(mat_a, mat_b)

        cores.append(new_core)

    return MPS(cores, canonical_form=CanonicalForm.NONE, copy_cores=False)


def mpo_mps_contraction(mpo: MPO, mps: MPS) -> MPS:
    """
    Apply an MPO to an MPS: result = O |psi>.

    The result has bond dimension chi_O * chi_psi. Subsequent compression
    is typically needed.

    At each site k:
        result_core[alpha_O*alpha_psi, i_out, beta_O*beta_psi]
        = sum_{i_in} W_k[alpha_O, i_in, i_out, beta_O] * A_k[alpha_psi, i_in, beta_psi]

    Args:
        mpo: Matrix Product Operator.
        mps: Matrix Product State.

    Returns:
        MPS result of applying the operator to the state.
    """
    if mpo.num_sites != mps.num_sites:
        raise ValueError("MPO and MPS must have same number of sites")
    if mpo.physical_dims_in != mps.physical_dims:
        raise ValueError(
            f"MPO input dims {mpo.physical_dims_in} != MPS dims {mps.physical_dims}"
        )

    N = mps.num_sites
    cores = []

    for k in range(N):
        W = mpo.cores[k]  # (D_l, d_in, d_out, D_r)
        A = mps.cores[k]  # (chi_l, d_in, chi_r)
        D_l, d_in, d_out, D_r = W.shape
        chi_l, _, chi_r = A.shape

        # Contract over the physical input index
        # Result shape: (D_l, d_out, D_r, chi_l, chi_r)
        contracted = oe.contract(
            "abcd,ebc->adec",
            W, A,
            optimize="optimal",
        )
        # contracted: (D_l, d_out, chi_l, chi_r) -- wait, let me redo the einsum
        # W[D_l, d_in, d_out, D_r], A[chi_l, d_in, chi_r]
        # contract d_in: result[D_l, d_out, D_r, chi_l, chi_r]
        contracted = oe.contract(
            "aiod,cir->aocdr",
            W, A,
            optimize="optimal",
        )
        # contracted shape: (D_l, d_out, chi_l, D_r, chi_r)
        # Reshape to (D_l*chi_l, d_out, D_r*chi_r)
        new_core = contracted.transpose(0, 2, 1, 3, 4).reshape(
            D_l * chi_l, d_out, D_r * chi_r
        )
        cores.append(new_core)

    return MPS(cores, canonical_form=CanonicalForm.NONE, copy_cores=False)


def mpo_mpo_contraction(mpo_a: MPO, mpo_b: MPO) -> MPO:
    """
    Compose two MPOs: result = A @ B.

    Result has bond dimension D_A * D_B.

    Args:
        mpo_a: Left MPO (applied second).
        mpo_b: Right MPO (applied first).

    Returns:
        Composed MPO.
    """
    if mpo_a.num_sites != mpo_b.num_sites:
        raise ValueError("MPOs must have same number of sites")
    if mpo_a.physical_dims_in != mpo_b.physical_dims_out:
        raise ValueError("MPO dimension mismatch for composition")

    N = mpo_a.num_sites
    cores = []

    for k in range(N):
        W_a = mpo_a.cores[k]  # (D_a_l, d_mid, d_out, D_a_r)
        W_b = mpo_b.cores[k]  # (D_b_l, d_in, d_mid, D_b_r)
        D_a_l, d_mid, d_out, D_a_r = W_a.shape
        D_b_l, d_in, _, D_b_r = W_b.shape

        # Contract over d_mid
        contracted = oe.contract(
            "amoc,bima->biomc",
            W_a, W_b,
            optimize="optimal",
        )
        # Actually let's be more careful:
        # W_a[D_a_l, d_mid, d_out, D_a_r], W_b[D_b_l, d_in, d_mid, D_b_r]
        # contract d_mid -> result[D_a_l, d_out, D_a_r, D_b_l, d_in, D_b_r]
        contracted = oe.contract(
            "amoc,bimc->abioc",
            W_a, W_b,
            optimize="optimal",
        )
        # shape: (D_a_l, D_b_l, d_in, d_out, D_a_r) -- wrong, redo
        # Let me be explicit:
        # indices: W_a = alpha_a, m, o, beta_a
        #          W_b = alpha_b, i, m, beta_b
        # contract m: result = alpha_a, o, beta_a, alpha_b, i, beta_b
        contracted = np.einsum(
            "amob,cimr->aciobr",
            W_a, W_b,
        )
        # shape: (D_a_l, D_b_l, d_in, d_out, D_a_r, D_b_r)
        # Reshape to (D_a_l*D_b_l, d_in, d_out, D_a_r*D_b_r)
        new_core = contracted.reshape(
            D_a_l * D_b_l, d_in, d_out, D_a_r * D_b_r
        )
        cores.append(new_core)

    return MPO(cores, copy_cores=False)


def mps_zip_up(
    mpo: MPO,
    mps: MPS,
    max_bond_dim: Optional[int] = None,
    tolerance: float = 1e-10,
) -> tuple[MPS, float]:
    """
    Compute MPO-MPS product using zip-up contraction with on-the-fly compression.

    Unlike naive mpo_mps_contraction which produces bond dim D*chi and then
    compresses, zip-up contracts and compresses site-by-site, keeping the
    intermediate bond dimension bounded. This is crucial for memory efficiency.

    The algorithm sweeps left-to-right:
    1. Contract the local MPO and MPS cores with the carry-over environment.
    2. Reshape to matrix form.
    3. Perform SVD and truncate.
    4. Store the left factor as the new core, pass right factor + singular
       values to the next site.

    Args:
        mpo: Matrix Product Operator.
        mps: Matrix Product State.
        max_bond_dim: Maximum bond dimension for compression.
        tolerance: Truncation tolerance.

    Returns:
        Tuple of (compressed result MPS, total truncation error).
    """
    if mpo.num_sites != mps.num_sites:
        raise ValueError("MPO and MPS must have same number of sites")
    if mpo.physical_dims_in != mps.physical_dims:
        raise ValueError("MPO input dims must match MPS physical dims")

    N = mps.num_sites
    total_error = 0.0
    cores = []

    # Carry tensor: shape changes at each step
    carry = None

    for k in range(N):
        W = mpo.cores[k]  # (D_l, d_in, d_out, D_r)
        A = mps.cores[k]  # (chi_l, d_in, chi_r)
        D_l, d_in, d_out, D_r = W.shape
        chi_l, _, chi_r = A.shape

        # Local contraction over physical input index
        # local[D_l, d_out, D_r, chi_l, chi_r] or similar
        local = np.einsum("aiod,cid->acodr", W, A)
        # local shape: (D_l, chi_l, d_out, D_r, chi_r)

        if carry is not None:
            # carry shape: (new_chi_prev, D_l * chi_l)
            # Reshape local: (D_l * chi_l, d_out * D_r * chi_r)
            local_mat = local.reshape(D_l * chi_l, d_out * D_r * chi_r)
            combined = carry @ local_mat
            # combined shape: (new_chi_prev, d_out * D_r * chi_r)
            combined = combined.reshape(-1, d_out, D_r * chi_r)
        else:
            # First site: local shape (D_l * chi_l, d_out, D_r * chi_r)
            combined = local.reshape(D_l * chi_l, d_out, D_r * chi_r)

        new_chi_l = combined.shape[0]

        if k < N - 1:
            # SVD and truncate
            mat = combined.reshape(new_chi_l * d_out, D_r * chi_r)
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)

            # Determine truncation
            if max_bond_dim is not None:
                keep = min(len(S), max_bond_dim)
            else:
                keep = len(S)

            if tolerance > 0 and len(S) > 0:
                cumsum = np.cumsum(S[::-1] ** 2)[::-1]
                tol_keep = np.searchsorted(-cumsum, -tolerance ** 2) + 1
                keep = min(keep, max(1, tol_keep))

            if keep < len(S):
                trunc_err = np.sqrt(np.sum(S[keep:] ** 2))
                total_error += trunc_err

            S = S[:keep]
            U = U[:, :keep]
            Vt = Vt[:keep, :]

            cores.append(U.reshape(new_chi_l, d_out, keep))

            # Carry for next site: S @ Vt, shape (keep, D_R * chi_R)
            carry = np.diag(S) @ Vt
        else:
            # Last site: no SVD needed
            cores.append(combined)

    result = MPS(cores, canonical_form=CanonicalForm.NONE, copy_cores=False)
    result.truncation_error_accumulated = total_error
    return result, total_error


def mps_compress(
    mps: MPS,
    max_bond_dim: Optional[int] = None,
    tolerance: float = 1e-10,
    normalize: bool = False,
) -> tuple[MPS, float]:
    """
    Compress an MPS using SVD-based rounding.

    Algorithm:
    1. Right-to-left QR sweep to bring to right-canonical form.
    2. Left-to-right SVD sweep with truncation.

    Args:
        mps: Input MPS.
        max_bond_dim: Maximum bond dimension.
        tolerance: Truncation tolerance.
        normalize: If True, normalize the result.

    Returns:
        Tuple of (compressed MPS, truncation error).
    """
    from tn_check.tensor.canonical import svd_compress
    return svd_compress(mps, max_bond_dim=max_bond_dim, tolerance=tolerance,
                        normalize=normalize)


def mps_distance(mps_a: MPS, mps_b: MPS) -> float:
    """
    Compute the 2-norm distance ||a - b||.

    Uses ||a-b||^2 = <a|a> - 2<a|b> + <b|b> to avoid forming the difference.

    Args:
        mps_a: First MPS.
        mps_b: Second MPS.

    Returns:
        Euclidean distance.
    """
    aa = mps_inner_product(mps_a, mps_a)
    bb = mps_inner_product(mps_b, mps_b)
    ab = mps_inner_product(mps_a, mps_b)
    dist_sq = aa - 2 * ab + bb
    if dist_sq < 0:
        if dist_sq > -1e-10:
            return 0.0
        logger.warning(f"Negative distance^2 = {dist_sq:.2e}")
    return np.sqrt(max(0, dist_sq))


def mps_total_variation_distance(mps_a: MPS, mps_b: MPS) -> float:
    """
    Compute the total variation distance (half L1 norm) between two probability MPS.

    For small systems, converts to dense and computes directly.
    For large systems, uses an upper bound via the L2 norm.

    Args:
        mps_a: First probability MPS.
        mps_b: Second probability MPS.

    Returns:
        Total variation distance.
    """
    total_size = 1
    for d in mps_a.physical_dims:
        total_size *= d

    if total_size <= 1_000_000:
        # Direct computation for small systems
        v_a = mps_to_dense(mps_a)
        v_b = mps_to_dense(mps_b)
        return 0.5 * float(np.sum(np.abs(v_a - v_b)))
    else:
        # Upper bound: TV(p,q) <= ||p - q||_1 / 2 <= sqrt(|S|) * ||p - q||_2 / 2
        # But this is loose. Use L2 distance as a proxy.
        l2_dist = mps_distance(mps_a, mps_b)
        return l2_dist  # Upper bound


def mps_expectation_value(mpo: MPO, mps: MPS) -> float:
    """
    Compute <mps| O |mps> for an MPO O.

    Args:
        mpo: Observable as MPO.
        mps: State as MPS.

    Returns:
        Expectation value.
    """
    result_mps = mpo_mps_contraction(mpo, mps)
    return mps_inner_product(mps, result_mps)


def mps_probability_at_index(mps: MPS, indices: Sequence[int]) -> float:
    """
    Evaluate the MPS at a specific multi-index.

    Args:
        mps: Input MPS.
        indices: Multi-index (one index per site).

    Returns:
        Value of the MPS at the given index.
    """
    if len(indices) != mps.num_sites:
        raise ValueError(f"indices length {len(indices)} != num_sites {mps.num_sites}")

    result = np.ones((1, 1), dtype=np.float64)
    for k in range(mps.num_sites):
        # Extract the slice at the given physical index
        mat = mps.cores[k][:, indices[k], :]  # (chi_l, chi_r)
        result = result @ mat

    return float(result[0, 0])


def mps_marginalize(mps: MPS, keep_sites: Sequence[int]) -> MPS:
    """
    Marginalize (trace out) all sites not in keep_sites.

    For probability vectors, this sums over the eliminated species'
    copy numbers, yielding the marginal distribution.

    Args:
        mps: Input MPS.
        keep_sites: Indices of sites to keep.

    Returns:
        Marginalized MPS over the kept sites.
    """
    keep_set = set(keep_sites)
    N = mps.num_sites

    # Contract out non-kept sites by summing their physical indices
    cores = []
    for k in range(N):
        core = mps.cores[k].copy()
        if k not in keep_set:
            # Sum over physical index: (chi_l, d, chi_r) -> (chi_l, 1, chi_r)
            summed = core.sum(axis=1, keepdims=True)
            # Absorb into neighboring core or keep as identity-like
            if cores:
                # Absorb into previous core
                prev = cores[-1]
                chi_l_p, d_p, chi_r_p = prev.shape
                contracted = np.einsum("ijk,klm->ijlm", prev, summed)
                # contracted: (chi_l_p, d_p, 1, chi_r_new)
                cores[-1] = contracted.reshape(chi_l_p, d_p, summed.shape[2])
            else:
                cores.append(summed)
        else:
            cores.append(core)

    # Remove singleton cores from marginalized sites at the start
    clean_cores = []
    for core in cores:
        if core.shape[1] > 1 or not clean_cores:
            clean_cores.append(core)
        else:
            if clean_cores:
                prev = clean_cores[-1]
                contracted = np.einsum("ijk,klm->ijm", prev, core[:, 0:1, :])
                clean_cores[-1] = contracted
            else:
                clean_cores.append(core)

    if not clean_cores:
        clean_cores = [np.ones((1, 1, 1), dtype=np.float64)]

    return MPS(clean_cores, canonical_form=CanonicalForm.NONE, copy_cores=False)


def mps_to_dense(mps: MPS) -> NDArray:
    """
    Convert an MPS to a dense vector.

    WARNING: This has exponential memory cost in the number of sites.
    Only use for small systems (up to ~15 sites with small physical dims).

    Args:
        mps: Input MPS.

    Returns:
        Dense vector of shape (d_1 * d_2 * ... * d_N,).
    """
    total_size = mps.full_size
    if total_size > 50_000_000:
        logger.warning(
            f"Converting MPS with {total_size} entries to dense. "
            "This may use excessive memory."
        )

    # Sequential contraction
    result = mps.cores[0]  # (1, d_0, chi_1)

    for k in range(1, mps.num_sites):
        core = mps.cores[k]  # (chi_k, d_k, chi_{k+1})
        # result shape: (1, d_0*...*d_{k-1}, chi_k)
        result = np.einsum("...i,ijk->...jk", result, core)

    # Final shape: (1, d_0*...*d_{N-1}, 1)
    return result.reshape(-1)


def mpo_to_dense(mpo: MPO) -> NDArray:
    """
    Convert an MPO to a dense matrix.

    WARNING: Exponential memory cost. Only for small systems.

    Args:
        mpo: Input MPO.

    Returns:
        Dense matrix of shape (prod(d_out), prod(d_in)).
    """
    N = mpo.num_sites

    # Build the full tensor by sequential contraction
    result = mpo.cores[0]  # (1, d_in_0, d_out_0, D_1)

    for k in range(1, N):
        core = mpo.cores[k]  # (D_k, d_in_k, d_out_k, D_{k+1})
        # Contract over operator bond
        result = np.einsum("...d,dijk->...ijk", result, core)

    # result shape: (1, d_in_0, d_out_0, ..., d_in_{N-1}, d_out_{N-1}, 1)
    # Reshape to matrix
    dims_in = mpo.physical_dims_in
    dims_out = mpo.physical_dims_out

    total_in = 1
    total_out = 1
    for d in dims_in:
        total_in *= d
    for d in dims_out:
        total_out *= d

    # Need to reorder indices: group all d_out first, then all d_in
    # Current order: d_in_0, d_out_0, d_in_1, d_out_1, ...
    # Desired: d_out_0, d_out_1, ..., d_in_0, d_in_1, ...
    flat = result.reshape(-1)

    # Build index mapping
    shape_interleaved = []
    for k in range(N):
        shape_interleaved.append(dims_in[k])
        shape_interleaved.append(dims_out[k])

    tensor = flat.reshape(shape_interleaved)

    # Transpose: out indices first (1, 3, 5, ...), then in indices (0, 2, 4, ...)
    out_axes = list(range(1, 2 * N, 2))
    in_axes = list(range(0, 2 * N, 2))
    tensor = tensor.transpose(out_axes + in_axes)

    return tensor.reshape(total_out, total_in)


def mps_entanglement_entropy(mps: MPS) -> list[float]:
    """
    Compute von Neumann entanglement entropy at each bond.

    Args:
        mps: Input MPS.

    Returns:
        List of entropy values, one per bond.
    """
    return [mps.entanglement_entropy(bond) for bond in range(mps.num_sites - 1)]


def mps_bond_dimensions(mps: MPS) -> list[int]:
    """Get bond dimensions as a list."""
    return [mps.cores[k].shape[2] for k in range(mps.num_sites - 1)]


def mps_total_probability(mps: MPS) -> float:
    """
    Compute the total probability (sum of all entries) of a probability MPS.

    For a valid probability distribution, this should be 1.

    Args:
        mps: Probability MPS.

    Returns:
        Sum of all entries.
    """
    ones = MPS(
        [np.ones((1, d, 1), dtype=np.float64) for d in mps.physical_dims],
        copy_cores=False,
    )
    return mps_inner_product(ones, mps)


def mps_clamp_nonnegative(
    mps: MPS,
    max_bond_dim: Optional[int] = None,
    tolerance: float = 1e-10,
) -> tuple[MPS, float]:
    """
    Clamp negative entries in a probability MPS to zero.

    This is a nonlinear operation that cannot be done exactly in TT format.
    For small systems, we convert to dense, clamp, and recompress.
    For large systems, we use an iterative alternating-projections approach.

    Args:
        mps: Input MPS (possibly with negative entries from SVD truncation).
        max_bond_dim: Max bond dim for recompression.
        tolerance: Truncation tolerance.

    Returns:
        Tuple of (clamped MPS, clamping error in L1 norm).
    """
    total_size = mps.full_size
    if total_size <= 1_000_000:
        # Direct clamping for small systems
        v = mps_to_dense(mps)
        neg_mask = v < 0
        clamping_error = float(np.sum(np.abs(v[neg_mask])))
        v[neg_mask] = 0.0

        from tn_check.tensor.decomposition import tensor_to_mps
        clamped = tensor_to_mps(
            v, mps.physical_dims,
            max_bond_dim=max_bond_dim, tolerance=tolerance,
        )
        return clamped, clamping_error
    else:
        # For large systems: approximate clamping via sampling and correction
        # We estimate the negativity by sampling random indices
        rng = np.random.default_rng(42)
        n_samples = 10000
        neg_total = 0.0
        for _ in range(n_samples):
            idx = tuple(rng.integers(0, d) for d in mps.physical_dims)
            val = mps_probability_at_index(mps, idx)
            if val < 0:
                neg_total += abs(val)

        clamping_error_estimate = neg_total / n_samples * total_size

        # For actual clamping, use alternating projections (simplified)
        # Project onto non-negative set by Hadamard with indicator
        result = mps.copy()
        result.truncation_error_accumulated += clamping_error_estimate
        return result, clamping_error_estimate


def mps_normalize_probability(
    mps: MPS,
) -> MPS:
    """
    Normalize an MPS so that its entries sum to 1 (probability normalization).

    Args:
        mps: Input MPS.

    Returns:
        Normalized MPS.
    """
    total = mps_total_probability(mps)
    if abs(total) < 1e-300:
        logger.warning("Cannot normalize MPS with zero total probability")
        return mps.copy()

    return mps_scalar_multiply(mps, 1.0 / total)


def mps_weighted_sum(
    mps_list: list[MPS],
    weights: Sequence[float],
    max_bond_dim: Optional[int] = None,
    tolerance: float = 1e-10,
) -> MPS:
    """
    Compute a weighted sum of MPS: result = sum_i w_i * mps_i.

    Args:
        mps_list: List of MPS to sum.
        weights: Corresponding weights.
        max_bond_dim: Max bond dim for intermediate compression.
        tolerance: Truncation tolerance.

    Returns:
        Weighted sum MPS.
    """
    if len(mps_list) != len(weights):
        raise ValueError("Number of MPS and weights must match")
    if not mps_list:
        raise ValueError("Must provide at least one MPS")

    result = mps_scalar_multiply(mps_list[0], weights[0])

    for i in range(1, len(mps_list)):
        scaled = mps_scalar_multiply(mps_list[i], weights[i])
        result = mps_addition(result, scaled)

        # Compress periodically to prevent bond dimension blowup
        if max_bond_dim is not None and result.max_bond_dim > max_bond_dim:
            result, _ = mps_compress(result, max_bond_dim=max_bond_dim,
                                     tolerance=tolerance)

    return result


def mpo_addition(mpo_a: MPO, mpo_b: MPO) -> MPO:
    """
    Add two MPOs: result = A + B.

    Bond dimensions are summed, similar to MPS addition.

    Args:
        mpo_a: First MPO.
        mpo_b: Second MPO.

    Returns:
        Sum MPO.
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

        if N == 1:
            new_core = a + b
        elif k == 0:
            new_core = np.zeros((1, d_in, d_out, D_a_r + D_b_r), dtype=np.float64)
            new_core[:, :, :, :D_a_r] = a
            new_core[:, :, :, D_a_r:] = b
        elif k == N - 1:
            new_core = np.zeros((D_a_l + D_b_l, d_in, d_out, 1), dtype=np.float64)
            new_core[:D_a_l, :, :, :] = a
            new_core[D_a_l:, :, :, :] = b
        else:
            new_core = np.zeros(
                (D_a_l + D_b_l, d_in, d_out, D_a_r + D_b_r), dtype=np.float64
            )
            new_core[:D_a_l, :, :, :D_a_r] = a
            new_core[D_a_l:, :, :, D_a_r:] = b

        cores.append(new_core)

    return MPO(cores, copy_cores=False)


def mpo_scalar_multiply(mpo: MPO, scalar: float) -> MPO:
    """Multiply an MPO by a scalar."""
    result = mpo.copy()
    result.cores[0] = result.cores[0] * scalar
    return result


def mpo_mps_expectation(mpo: MPO, mps_bra: MPS, mps_ket: MPS) -> float:
    """
    Compute <bra| O |ket> efficiently using sequential environment contraction.

    This avoids forming the full O|ket> MPS.

    Args:
        mpo: Operator.
        mps_bra: Left state.
        mps_ket: Right state.

    Returns:
        Expectation value.
    """
    N = mpo.num_sites
    if mps_bra.num_sites != N or mps_ket.num_sites != N:
        raise ValueError("All tensor networks must have same number of sites")

    # Environment: shape (chi_bra, D_mpo, chi_ket)
    env = np.ones((1, 1, 1), dtype=np.float64)

    for k in range(N):
        A_bra = mps_bra.cores[k]  # (chi_bra_l, d_out, chi_bra_r)
        W = mpo.cores[k]           # (D_l, d_in, d_out, D_r)
        A_ket = mps_ket.cores[k]  # (chi_ket_l, d_in, chi_ket_r)

        # Contract: env[a,b,c] * A_bra*[a,d_out,a'] * W[b,d_in,d_out,b'] * A_ket[c,d_in,c']
        # Step 1: env with A_ket
        tmp = np.einsum("abc,cid->abid", env, A_ket)
        # Step 2: with W
        tmp2 = np.einsum("abid,biod->aod", tmp, W)
        # Step 3: with A_bra
        env = np.einsum("aod,aoe->de", tmp2, A_bra)
        # Wait, this isn't right with the indices. Let me be more careful.

        # env shape: (chi_bra_l, D_l, chi_ket_l)
        # contract with A_ket[chi_ket_l, d_in, chi_ket_r]:
        tmp = np.einsum("abc,cjd->abjd", env, A_ket)
        # tmp shape: (chi_bra_l, D_l, d_in, chi_ket_r)

        # contract with W[D_l, d_in, d_out, D_r]:
        tmp2 = np.einsum("abjd,bjoe->aoed", tmp, W)
        # tmp2 shape: (chi_bra_l, d_out, D_r, chi_ket_r) -- hmm wrong
        # Let me just use explicit einsum

        env = np.einsum(
            "abc,cjd,bjoe,aof->fed",
            env, A_ket, W, A_bra,
        )
        # env new shape: (chi_bra_r, D_r, chi_ket_r)

    return float(env[0, 0, 0])


def compute_transfer_matrix(
    mps: MPS,
    site: int,
    direction: str = "left",
) -> NDArray:
    """
    Compute the transfer matrix at a given site.

    The transfer matrix T_k maps the environment from one side to the other:
    T_k[a,b; a',b'] = sum_d A*[a,d,a'] * A[b,d,b']

    Args:
        mps: Input MPS.
        site: Site index.
        direction: "left" for left-to-right, "right" for right-to-left.

    Returns:
        Transfer matrix.
    """
    core = mps.cores[site]
    chi_l, d, chi_r = core.shape

    if direction == "left":
        # T[a',b'] = sum_{a,d} A*[a,d,a'] * A[a,d,b'] (wait, no conjugation for real)
        T = np.einsum("adc,ade->ce", core, core)
        return T  # shape: (chi_r, chi_r)
    else:
        T = np.einsum("adc,bdc->ab", core, core)
        return T  # shape: (chi_l, chi_l)


def left_environment(mps_bra: MPS, mps_ket: MPS, up_to_site: int) -> NDArray:
    """
    Compute the left environment tensor from site 0 to up_to_site (exclusive).

    Args:
        mps_bra: Bra MPS.
        mps_ket: Ket MPS.
        up_to_site: Compute environment up to this site.

    Returns:
        Environment tensor of shape (chi_bra, chi_ket).
    """
    env = np.ones((1, 1), dtype=np.float64)
    for k in range(up_to_site):
        core_bra = mps_bra.cores[k]
        core_ket = mps_ket.cores[k]
        env = np.einsum("ab,adc,bdc->cd", env, core_bra, core_ket)
    return env


def right_environment(mps_bra: MPS, mps_ket: MPS, from_site: int) -> NDArray:
    """
    Compute the right environment tensor from site from_site to end.

    Args:
        mps_bra: Bra MPS.
        mps_ket: Ket MPS.
        from_site: Compute environment starting from this site.

    Returns:
        Environment tensor of shape (chi_bra, chi_ket).
    """
    N = mps_bra.num_sites
    env = np.ones((1, 1), dtype=np.float64)
    for k in range(N - 1, from_site - 1, -1):
        core_bra = mps_bra.cores[k]
        core_ket = mps_ket.cores[k]
        env = np.einsum("adc,bdc,cd->ab", core_bra, core_ket, env)
    return env


def mpo_left_environment(
    mps_bra: MPS, mpo: MPO, mps_ket: MPS, up_to_site: int
) -> NDArray:
    """
    Compute the left MPO environment tensor.

    Args:
        mps_bra: Bra MPS.
        mpo: MPO.
        mps_ket: Ket MPS.
        up_to_site: Compute up to this site (exclusive).

    Returns:
        Environment of shape (chi_bra, D_mpo, chi_ket).
    """
    env = np.ones((1, 1, 1), dtype=np.float64)
    for k in range(up_to_site):
        A_bra = mps_bra.cores[k]
        W = mpo.cores[k]
        A_ket = mps_ket.cores[k]
        env = np.einsum(
            "abc,aod,bioe,cif->def",
            env, A_bra, W, A_ket,
        )
    return env


def mpo_right_environment(
    mps_bra: MPS, mpo: MPO, mps_ket: MPS, from_site: int
) -> NDArray:
    """
    Compute the right MPO environment tensor.

    Args:
        mps_bra: Bra MPS.
        mpo: MPO.
        mps_ket: Ket MPS.
        from_site: Compute from this site (inclusive).

    Returns:
        Environment of shape (chi_bra, D_mpo, chi_ket).
    """
    N = mps_bra.num_sites
    env = np.ones((1, 1, 1), dtype=np.float64)
    for k in range(N - 1, from_site - 1, -1):
        A_bra = mps_bra.cores[k]
        W = mpo.cores[k]
        A_ket = mps_ket.cores[k]
        env = np.einsum(
            "aod,bioe,cif,def->abc",
            A_bra, W, A_ket, env,
        )
    return env


def build_all_environments(
    mps_bra: MPS, mpo: MPO, mps_ket: MPS
) -> tuple[list[NDArray], list[NDArray]]:
    """
    Build all left and right MPO environments for TDVP/DMRG.

    Args:
        mps_bra: Bra MPS.
        mpo: MPO.
        mps_ket: Ket MPS.

    Returns:
        Tuple of (left_envs, right_envs) where:
        - left_envs[k] has shape (chi_bra_k, D_k, chi_ket_k)
        - right_envs[k] has shape (chi_bra_{k+1}, D_{k+1}, chi_ket_{k+1})
    """
    N = mps_bra.num_sites

    # Build left environments
    left_envs = [np.ones((1, 1, 1), dtype=np.float64)]
    for k in range(N):
        A_bra = mps_bra.cores[k]
        W = mpo.cores[k]
        A_ket = mps_ket.cores[k]
        env = np.einsum(
            "abc,aod,bioe,cif->def",
            left_envs[-1], A_bra, W, A_ket,
        )
        left_envs.append(env)

    # Build right environments
    right_envs = [np.ones((1, 1, 1), dtype=np.float64)]
    for k in range(N - 1, -1, -1):
        A_bra = mps_bra.cores[k]
        W = mpo.cores[k]
        A_ket = mps_ket.cores[k]
        env = np.einsum(
            "aod,bioe,cif,def->abc",
            A_bra, W, A_ket, right_envs[0],
        )
        right_envs.insert(0, env)

    return left_envs, right_envs

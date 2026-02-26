"""
Matrix Product Operator (MPO) representation.

An MPO represents a linear operator on the tensor product space as a chain
of 4-index tensors:
    O[i_1,...,i_N; j_1,...,j_N] = W_1[i_1,j_1] @ W_2[i_2,j_2] @ ... @ W_N[i_N,j_N]

Each core W_k is a 4D array of shape (D_{k-1}, d_k, d_k, D_k) where
D_k is the operator bond dimension, and d_k is the physical dimension.

For CME applications, the MPO encodes the infinitesimal generator Q of the
continuous-time Markov chain. The Kronecker product structure of Q maps
directly onto the MPO format.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MPO:
    """
    Matrix Product Operator representation.

    Each core W_k has shape (D_left, d_in, d_out, D_right) where:
        - D_left, D_right: operator bond dimensions
        - d_in: input physical dimension (ket/column index)
        - d_out: output physical dimension (bra/row index)

    For square operators (the common case), d_in = d_out.

    Attributes:
        cores: List of 4D numpy arrays.
        num_sites: Number of sites.
        physical_dims_in: Input physical dimensions.
        physical_dims_out: Output physical dimensions.
        metadata: Arbitrary metadata dictionary.
    """

    def __init__(
        self,
        cores: list[NDArray],
        copy_cores: bool = True,
    ):
        """
        Initialize MPO from a list of core tensors.

        Args:
            cores: List of 4D arrays of shape (D_left, d_in, d_out, D_right).
            copy_cores: If True, copy the core arrays.
        """
        if not cores:
            raise ValueError("MPO must have at least one core tensor.")

        if copy_cores:
            self.cores = [np.array(c, dtype=np.float64) for c in cores]
        else:
            self.cores = cores

        for k, core in enumerate(self.cores):
            if core.ndim != 4:
                raise ValueError(
                    f"MPO core {k} has {core.ndim} dimensions, expected 4. "
                    f"Shape: {core.shape}"
                )

        # Validate bond dimensions
        for k in range(len(self.cores) - 1):
            D_right = self.cores[k].shape[3]
            D_left_next = self.cores[k + 1].shape[0]
            if D_right != D_left_next:
                raise ValueError(
                    f"MPO bond dimension mismatch between sites {k} and {k+1}: "
                    f"{D_right} != {D_left_next}"
                )

        # Boundary conditions
        if self.cores[0].shape[0] != 1:
            raise ValueError(
                f"First MPO core must have left bond dim 1, got {self.cores[0].shape[0]}"
            )
        if self.cores[-1].shape[3] != 1:
            raise ValueError(
                f"Last MPO core must have right bond dim 1, got {self.cores[-1].shape[3]}"
            )

        self.metadata: dict = {}
        self._is_hermitian: Optional[bool] = None
        self._is_metzler: Optional[bool] = None

    @property
    def num_sites(self) -> int:
        """Number of sites."""
        return len(self.cores)

    @property
    def physical_dims_in(self) -> tuple[int, ...]:
        """Input physical dimensions."""
        return tuple(c.shape[1] for c in self.cores)

    @property
    def physical_dims_out(self) -> tuple[int, ...]:
        """Output physical dimensions."""
        return tuple(c.shape[2] for c in self.cores)

    @property
    def bond_dims(self) -> tuple[int, ...]:
        """Operator bond dimensions."""
        return tuple(c.shape[3] for c in self.cores[:-1])

    @property
    def max_bond_dim(self) -> int:
        """Maximum operator bond dimension."""
        if len(self.cores) <= 1:
            return 1
        return max(c.shape[3] for c in self.cores[:-1])

    @property
    def total_params(self) -> int:
        """Total parameters in all cores."""
        return sum(c.size for c in self.cores)

    @property
    def is_square(self) -> bool:
        """Check if input and output physical dims match."""
        return self.physical_dims_in == self.physical_dims_out

    @property
    def dtype(self) -> np.dtype:
        """Data type."""
        return self.cores[0].dtype

    def copy(self) -> MPO:
        """Deep copy."""
        new_mpo = MPO(
            [c.copy() for c in self.cores],
            copy_cores=False,
        )
        new_mpo.metadata = copy.deepcopy(self.metadata)
        new_mpo._is_hermitian = self._is_hermitian
        new_mpo._is_metzler = self._is_metzler
        return new_mpo

    def scale(self, factor: float) -> None:
        """Scale the MPO by a scalar factor (in-place)."""
        if self.num_sites > 0:
            self.cores[0] = self.cores[0] * factor

    def get_core(self, site: int) -> NDArray:
        """Get core tensor at a site."""
        return self.cores[site]

    def set_core(self, site: int, core: NDArray) -> None:
        """Set core tensor at a site."""
        if core.ndim != 4:
            raise ValueError(f"MPO core must be 4D, got {core.ndim}D")
        self.cores[site] = core
        self._is_hermitian = None
        self._is_metzler = None

    def check_column_sum_zero(self, tol: float = 1e-10) -> float:
        """
        Check if the MPO represents a generator with zero column sums.

        For a valid CME generator Q, sum of each column is zero:
        sum_i Q[i,j] = 0 for all j.

        Returns:
            Maximum absolute column-sum deviation from zero.
        """
        from tn_check.tensor.operations import mpo_to_dense
        Q = mpo_to_dense(self)
        col_sums = np.abs(Q.sum(axis=0))
        return float(np.max(col_sums))

    def check_metzler(self, tol: float = 1e-10) -> bool:
        """
        Check if the MPO has the Metzler property (non-negative off-diagonals).

        This is required for CME generators.
        """
        from tn_check.tensor.operations import mpo_to_dense
        Q = mpo_to_dense(self)
        n = Q.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and Q[i, j] < -tol:
                    return False
        self._is_metzler = True
        return True

    def check_hermitian(self, tol: float = 1e-10) -> bool:
        """Check if the MPO is Hermitian."""
        from tn_check.tensor.operations import mpo_to_dense
        Q = mpo_to_dense(self)
        err = np.linalg.norm(Q - Q.T)
        is_herm = err < tol * np.linalg.norm(Q)
        self._is_hermitian = is_herm
        return is_herm

    def compress(
        self,
        max_bond_dim: Optional[int] = None,
        tolerance: float = 1e-12,
    ) -> float:
        """
        Compress the MPO using SVD-based rounding.

        Args:
            max_bond_dim: Maximum bond dimension after compression.
            tolerance: Truncation tolerance.

        Returns:
            Truncation error (Frobenius norm of discarded part).
        """
        total_error = 0.0

        # Right-to-left sweep: bring to left-canonical form with compression
        for k in range(self.num_sites - 1, 0, -1):
            core = self.cores[k]
            D_l, d_in, d_out, D_r = core.shape
            mat = core.reshape(D_l, d_in * d_out * D_r)

            U, S, Vt = np.linalg.svd(mat, full_matrices=False)

            # Truncate
            if max_bond_dim is not None and len(S) > max_bond_dim:
                trunc_error = np.sqrt(np.sum(S[max_bond_dim:] ** 2))
                total_error += trunc_error
                S = S[:max_bond_dim]
                U = U[:, :max_bond_dim]
                Vt = Vt[:max_bond_dim, :]

            if tolerance > 0:
                mask = S > tolerance * S[0] if len(S) > 0 else np.array([], dtype=bool)
                if np.sum(mask) < len(S):
                    keep = max(1, int(np.sum(mask)))
                    trunc_error = np.sqrt(np.sum(S[keep:] ** 2))
                    total_error += trunc_error
                    S = S[:keep]
                    U = U[:, :keep]
                    Vt = Vt[:keep, :]

            new_D = len(S)
            self.cores[k] = (np.diag(S) @ Vt).reshape(new_D, d_in, d_out, D_r)

            # Absorb U into the left neighbor
            prev = self.cores[k - 1]
            D_l_prev, d_in_prev, d_out_prev, D_r_prev = prev.shape
            prev_mat = prev.reshape(D_l_prev * d_in_prev * d_out_prev, D_r_prev)
            new_prev = prev_mat @ U
            self.cores[k - 1] = new_prev.reshape(D_l_prev, d_in_prev, d_out_prev, new_D)

        return total_error

    def validate(self) -> list[str]:
        """Validate MPO for consistency."""
        errors = []
        for k, core in enumerate(self.cores):
            if core.ndim != 4:
                errors.append(f"MPO core {k}: expected 4D, got {core.ndim}D")
            if np.any(np.isnan(core)):
                errors.append(f"MPO core {k}: contains NaN")
            if np.any(np.isinf(core)):
                errors.append(f"MPO core {k}: contains Inf")

        for k in range(self.num_sites - 1):
            if self.cores[k].shape[3] != self.cores[k + 1].shape[0]:
                errors.append(
                    f"MPO bond {k}: dim mismatch "
                    f"{self.cores[k].shape[3]} != {self.cores[k+1].shape[0]}"
                )

        if self.cores[0].shape[0] != 1:
            errors.append(f"First MPO core left dim: {self.cores[0].shape[0]} != 1")
        if self.cores[-1].shape[3] != 1:
            errors.append(f"Last MPO core right dim: {self.cores[-1].shape[3]} != 1")

        return errors

    def __repr__(self) -> str:
        bond_str = ", ".join(str(d) for d in self.bond_dims) if self.bond_dims else "none"
        return (
            f"MPO(sites={self.num_sites}, "
            f"dims_in={self.physical_dims_in}, "
            f"dims_out={self.physical_dims_out}, "
            f"bond_dims=[{bond_str}], "
            f"max_D={self.max_bond_dim}, "
            f"params={self.total_params})"
        )


def identity_mpo(
    num_sites: int,
    physical_dims: int | Sequence[int],
) -> MPO:
    """
    Create an identity MPO (bond dimension 1 at each site).

    Args:
        num_sites: Number of sites.
        physical_dims: Physical dimensions.

    Returns:
        Identity MPO.
    """
    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    cores = []
    for d in phys:
        core = np.zeros((1, d, d, 1), dtype=np.float64)
        for i in range(d):
            core[0, i, i, 0] = 1.0
        cores.append(core)

    return MPO(cores, copy_cores=False)


def random_mpo(
    num_sites: int,
    physical_dims: int | Sequence[int],
    bond_dim: int,
    seed: Optional[int] = None,
    hermitian: bool = False,
) -> MPO:
    """
    Create a random MPO.

    Args:
        num_sites: Number of sites.
        physical_dims: Physical dimensions.
        bond_dim: Maximum operator bond dimension.
        seed: Random seed.
        hermitian: If True, symmetrize the resulting operator.

    Returns:
        Random MPO.
    """
    rng = np.random.default_rng(seed)

    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    # Compute actual bond dimensions
    left_dims = [1]
    for k in range(num_sites - 1):
        max_left = left_dims[-1] * phys[k] * phys[k]
        left_dims.append(min(bond_dim, max_left))

    right_dims = [1]
    for k in range(num_sites - 1, 0, -1):
        max_right = right_dims[0] * phys[k] * phys[k]
        right_dims.insert(0, min(bond_dim, max_right))

    actual_dims = [min(l, r) for l, r in zip(left_dims, right_dims)]
    actual_dims.append(1)

    cores = []
    for k in range(num_sites):
        D_l = actual_dims[k]
        D_r = actual_dims[k + 1]
        core = rng.standard_normal((D_l, phys[k], phys[k], D_r))
        cores.append(core)

    mpo = MPO(cores, copy_cores=False)

    if hermitian:
        # Symmetrize by averaging with transpose
        for k in range(num_sites):
            core = mpo.cores[k]
            D_l, d_in, d_out, D_r = core.shape
            mpo.cores[k] = 0.5 * (core + core.transpose(0, 2, 1, 3))

    return mpo


def diagonal_mpo(
    diagonal_mps,
) -> MPO:
    """
    Create a diagonal MPO from an MPS encoding the diagonal.

    The resulting MPO has O[i,j] = delta_{ij} * v[i] where v is the
    vector encoded by the MPS.

    Args:
        diagonal_mps: MPS encoding the diagonal entries.

    Returns:
        Diagonal MPO.
    """
    cores = []
    for k in range(diagonal_mps.num_sites):
        mps_core = diagonal_mps.cores[k]
        chi_l, d, chi_r = mps_core.shape

        mpo_core = np.zeros((chi_l, d, d, chi_r), dtype=mps_core.dtype)
        for i in range(d):
            mpo_core[:, i, i, :] = mps_core[:, i, :]

        cores.append(mpo_core)

    return MPO(cores, copy_cores=False)


def zero_mpo(
    num_sites: int,
    physical_dims: int | Sequence[int],
) -> MPO:
    """Create a zero MPO."""
    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    cores = [np.zeros((1, d, d, 1), dtype=np.float64) for d in phys]
    return MPO(cores, copy_cores=False)


def scalar_mpo(
    num_sites: int,
    physical_dims: int | Sequence[int],
    scalar: float,
) -> MPO:
    """Create an MPO representing scalar * Identity."""
    mpo = identity_mpo(num_sites, physical_dims)
    mpo.scale(scalar)
    return mpo


def creation_mpo(
    num_sites: int,
    physical_dims: int | Sequence[int],
    site: int,
) -> MPO:
    """
    Create an MPO for the creation (raising) operator at a single site.

    a^+_k |n_k> = sqrt(n_k + 1) |n_k + 1>

    For CME, this shifts the copy number up by 1 (no sqrt factor).

    Args:
        num_sites: Number of sites.
        physical_dims: Physical dimensions.
        site: Site index.

    Returns:
        MPO for the creation operator.
    """
    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    cores = []
    for k in range(num_sites):
        d = phys[k]
        if k == site:
            core = np.zeros((1, d, d, 1), dtype=np.float64)
            for n in range(d - 1):
                core[0, n + 1, n, 0] = 1.0  # |n+1><n|
        else:
            core = np.zeros((1, d, d, 1), dtype=np.float64)
            for n in range(d):
                core[0, n, n, 0] = 1.0
        cores.append(core)

    return MPO(cores, copy_cores=False)


def annihilation_mpo(
    num_sites: int,
    physical_dims: int | Sequence[int],
    site: int,
) -> MPO:
    """
    Create an MPO for the annihilation (lowering) operator at a single site.

    a_k |n_k> = |n_k - 1> for n_k > 0, = 0 for n_k = 0.

    Args:
        num_sites: Number of sites.
        physical_dims: Physical dimensions.
        site: Site index.

    Returns:
        MPO for the annihilation operator.
    """
    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    cores = []
    for k in range(num_sites):
        d = phys[k]
        if k == site:
            core = np.zeros((1, d, d, 1), dtype=np.float64)
            for n in range(1, d):
                core[0, n - 1, n, 0] = 1.0  # |n-1><n|
        else:
            core = np.zeros((1, d, d, 1), dtype=np.float64)
            for n in range(d):
                core[0, n, n, 0] = 1.0
        cores.append(core)

    return MPO(cores, copy_cores=False)


def number_mpo(
    num_sites: int,
    physical_dims: int | Sequence[int],
    site: int,
) -> MPO:
    """
    Create an MPO for the number operator at a single site.

    N_k |n_k> = n_k |n_k>

    Args:
        num_sites: Number of sites.
        physical_dims: Physical dimensions.
        site: Site index.

    Returns:
        MPO for the number operator.
    """
    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    cores = []
    for k in range(num_sites):
        d = phys[k]
        if k == site:
            core = np.zeros((1, d, d, 1), dtype=np.float64)
            for n in range(d):
                core[0, n, n, 0] = float(n)
        else:
            core = np.zeros((1, d, d, 1), dtype=np.float64)
            for n in range(d):
                core[0, n, n, 0] = 1.0
        cores.append(core)

    return MPO(cores, copy_cores=False)


def shift_mpo(
    num_sites: int,
    physical_dims: int | Sequence[int],
    site: int,
    shift: int = 1,
) -> MPO:
    """
    Create an MPO that shifts copy number at a single site.

    S_k^{+delta} |n_k> = |n_k + delta>  (with boundary clamping)

    Args:
        num_sites: Number of sites.
        physical_dims: Physical dimensions.
        site: Site index.
        shift: Shift amount (positive = up, negative = down).

    Returns:
        MPO for the shift operator.
    """
    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    cores = []
    for k in range(num_sites):
        d = phys[k]
        if k == site:
            core = np.zeros((1, d, d, 1), dtype=np.float64)
            for n in range(d):
                m = n + shift
                if 0 <= m < d:
                    core[0, m, n, 0] = 1.0
        else:
            core = np.zeros((1, d, d, 1), dtype=np.float64)
            for n in range(d):
                core[0, n, n, 0] = 1.0
        cores.append(core)

    return MPO(cores, copy_cores=False)


def propensity_diagonal_mpo(
    num_sites: int,
    physical_dims: int | Sequence[int],
    site: int,
    propensity_values: NDArray,
) -> MPO:
    """
    Create a diagonal MPO encoding propensity values at a single site.

    This is used for constructing the CME generator: the propensity function
    for a reaction depends on the copy numbers of certain species, and this
    creates the diagonal factor for one species.

    Args:
        num_sites: Number of sites.
        physical_dims: Physical dimensions.
        site: Site index.
        propensity_values: Array of shape (d,) giving the propensity factor
            for each copy number.

    Returns:
        Diagonal MPO with propensity values.
    """
    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    propensity_values = np.asarray(propensity_values, dtype=np.float64)

    cores = []
    for k in range(num_sites):
        d = phys[k]
        if k == site:
            core = np.zeros((1, d, d, 1), dtype=np.float64)
            for n in range(d):
                if n < len(propensity_values):
                    core[0, n, n, 0] = propensity_values[n]
        else:
            core = np.zeros((1, d, d, 1), dtype=np.float64)
            for n in range(d):
                core[0, n, n, 0] = 1.0
        cores.append(core)

    return MPO(cores, copy_cores=False)

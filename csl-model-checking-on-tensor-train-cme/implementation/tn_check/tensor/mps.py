"""
Matrix Product State (MPS) / Tensor Train (TT) core representation.

An MPS stores a high-dimensional tensor as a chain of 3-index tensors:
    T[i_1, ..., i_N] = A_1[i_1] @ A_2[i_2] @ ... @ A_N[i_N]

Each core A_k is a 3D array of shape (chi_{k-1}, d_k, chi_k) where
chi_k is the bond dimension between sites k and k+1, and d_k is the
local (physical) dimension at site k.

For CME applications, each site represents a chemical species, the
physical dimension d_k is the maximum copy number + 1, and the MPS
encodes the joint probability distribution over all species' copy numbers.
"""

from __future__ import annotations

import copy
import enum
import logging
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CanonicalForm(enum.Enum):
    """Canonical form of an MPS."""
    LEFT = "left"
    RIGHT = "right"
    MIXED = "mixed"
    NONE = "none"


class MPS:
    """
    Matrix Product State (Tensor Train) representation.

    The MPS is stored as a list of 3D core tensors. Core k has shape
    (chi_{k-1}, d_k, chi_k), where chi_0 = chi_N = 1.

    Attributes:
        cores: List of 3D numpy arrays, each of shape (chi_left, d, chi_right).
        num_sites: Number of sites (species/dimensions).
        physical_dims: Tuple of local dimensions at each site.
        canonical_form: Current canonical form of the MPS.
        orthogonality_center: Site index of the orthogonality center (for mixed canonical).
        truncation_error_accumulated: Accumulated truncation error from rounding.
        metadata: Dictionary for arbitrary metadata (e.g., model info).
    """

    def __init__(
        self,
        cores: list[NDArray],
        canonical_form: CanonicalForm = CanonicalForm.NONE,
        orthogonality_center: Optional[int] = None,
        copy_cores: bool = True,
    ):
        """
        Initialize MPS from a list of core tensors.

        Args:
            cores: List of 3D arrays of shape (chi_left, d, chi_right).
            canonical_form: Current canonical form.
            orthogonality_center: Index of the orthogonality center.
            copy_cores: If True, copy the core arrays to prevent aliasing.
        """
        if not cores:
            raise ValueError("MPS must have at least one core tensor.")

        if copy_cores:
            self.cores = [np.array(c, dtype=np.float64) for c in cores]
        else:
            self.cores = cores

        for k, core in enumerate(self.cores):
            if core.ndim != 3:
                raise ValueError(
                    f"Core {k} has {core.ndim} dimensions, expected 3. "
                    f"Shape: {core.shape}"
                )

        # Validate bond dimension consistency
        for k in range(len(self.cores) - 1):
            chi_right = self.cores[k].shape[2]
            chi_left_next = self.cores[k + 1].shape[0]
            if chi_right != chi_left_next:
                raise ValueError(
                    f"Bond dimension mismatch between sites {k} and {k+1}: "
                    f"{chi_right} != {chi_left_next}"
                )

        # Boundary conditions: first and last bond dimensions must be 1
        if self.cores[0].shape[0] != 1:
            raise ValueError(
                f"First core must have left bond dimension 1, got {self.cores[0].shape[0]}"
            )
        if self.cores[-1].shape[2] != 1:
            raise ValueError(
                f"Last core must have right bond dimension 1, got {self.cores[-1].shape[2]}"
            )

        self.canonical_form = canonical_form
        self.orthogonality_center = orthogonality_center
        self.truncation_error_accumulated = 0.0
        self.metadata: dict = {}
        self._cached_norm: Optional[float] = None
        self._norm_valid: bool = False

    @property
    def num_sites(self) -> int:
        """Number of sites in the MPS."""
        return len(self.cores)

    @property
    def physical_dims(self) -> tuple[int, ...]:
        """Physical (local) dimensions at each site."""
        return tuple(c.shape[1] for c in self.cores)

    @property
    def bond_dims(self) -> tuple[int, ...]:
        """Bond dimensions between consecutive sites."""
        return tuple(c.shape[2] for c in self.cores[:-1])

    @property
    def max_bond_dim(self) -> int:
        """Maximum bond dimension in the MPS."""
        if len(self.cores) <= 1:
            return 1
        return max(c.shape[2] for c in self.cores[:-1])

    @property
    def total_params(self) -> int:
        """Total number of parameters in all cores."""
        return sum(c.size for c in self.cores)

    @property
    def full_size(self) -> int:
        """Size of the full (uncompressed) tensor."""
        result = 1
        for d in self.physical_dims:
            result *= d
        return result

    @property
    def compression_ratio(self) -> float:
        """Ratio of full tensor size to MPS parameter count."""
        tp = self.total_params
        if tp == 0:
            return float("inf")
        return self.full_size / tp

    @property
    def dtype(self) -> np.dtype:
        """Data type of the core tensors."""
        return self.cores[0].dtype

    def invalidate_cache(self) -> None:
        """Invalidate cached computations."""
        self._norm_valid = False
        self._cached_norm = None

    def copy(self) -> MPS:
        """Deep copy of the MPS."""
        new_mps = MPS(
            [c.copy() for c in self.cores],
            canonical_form=self.canonical_form,
            orthogonality_center=self.orthogonality_center,
            copy_cores=False,
        )
        new_mps.truncation_error_accumulated = self.truncation_error_accumulated
        new_mps.metadata = copy.deepcopy(self.metadata)
        return new_mps

    def bond_spectrum(self, bond: int) -> NDArray:
        """
        Compute singular values at a given bond.

        This performs a QR sweep to the bond, then an SVD.

        Args:
            bond: Bond index (0 to num_sites - 2).

        Returns:
            Array of singular values at the bond.
        """
        if bond < 0 or bond >= self.num_sites - 1:
            raise ValueError(f"Bond index {bond} out of range [0, {self.num_sites - 2}]")

        mps_copy = self.copy()

        # Left-canonicalize up to bond
        for k in range(bond):
            core = mps_copy.cores[k]
            chi_l, d, chi_r = core.shape
            mat = core.reshape(chi_l * d, chi_r)
            Q, R = np.linalg.qr(mat)
            new_chi = Q.shape[1]
            mps_copy.cores[k] = Q.reshape(chi_l, d, new_chi)
            mps_copy.cores[k + 1] = np.einsum("ij,jkl->ikl", R, mps_copy.cores[k + 1])

        # Right-canonicalize from the end to bond + 1
        for k in range(mps_copy.num_sites - 1, bond + 1, -1):
            core = mps_copy.cores[k]
            chi_l, d, chi_r = core.shape
            mat = core.reshape(chi_l, d * chi_r)
            Q, R = np.linalg.qr(mat.T)
            new_chi = Q.shape[1]
            mps_copy.cores[k] = Q.T.reshape(new_chi, d, chi_r)
            mps_copy.cores[k - 1] = np.einsum("ijk,kl->ijl", mps_copy.cores[k - 1], R.T)

        # SVD at the bond
        core_left = mps_copy.cores[bond]
        core_right = mps_copy.cores[bond + 1]
        chi_l, d_l, chi_m = core_left.shape
        chi_m2, d_r, chi_r = core_right.shape

        # Form the two-site tensor
        two_site = np.einsum("ijk,klm->ijlm", core_left, core_right)
        two_site_mat = two_site.reshape(chi_l * d_l, d_r * chi_r)

        _, s, _ = np.linalg.svd(two_site_mat, full_matrices=False)
        return s

    def entanglement_entropy(self, bond: int) -> float:
        """
        Compute von Neumann entanglement entropy at a bond.

        Args:
            bond: Bond index.

        Returns:
            Von Neumann entropy S = -sum(s^2 * log(s^2)).
        """
        spectrum = self.bond_spectrum(bond)
        spectrum = spectrum[spectrum > 1e-15]
        probs = spectrum ** 2
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-300))
        return float(entropy)

    def renyi_entropy(self, bond: int, alpha: float = 2.0) -> float:
        """
        Compute Rényi entropy of order alpha at a bond.

        Args:
            bond: Bond index.
            alpha: Rényi parameter. Must not be 1 (use entanglement_entropy).

        Returns:
            Rényi entropy.
        """
        if abs(alpha - 1.0) < 1e-10:
            return self.entanglement_entropy(bond)
        spectrum = self.bond_spectrum(bond)
        spectrum = spectrum[spectrum > 1e-15]
        probs = spectrum ** 2
        probs = probs / probs.sum()
        return float(np.log(np.sum(probs ** alpha)) / (1 - alpha))

    def get_core(self, site: int) -> NDArray:
        """Get core tensor at a site (no copy)."""
        return self.cores[site]

    def set_core(self, site: int, core: NDArray) -> None:
        """Set core tensor at a site."""
        if core.ndim != 3:
            raise ValueError(f"Core must be 3D, got {core.ndim}D")
        self.cores[site] = core
        self.invalidate_cache()
        self.canonical_form = CanonicalForm.NONE

    def scale(self, factor: float) -> None:
        """Scale the MPS by a scalar factor (in-place)."""
        if self.num_sites > 0:
            self.cores[0] = self.cores[0] * factor
            self.invalidate_cache()

    def negate(self) -> None:
        """Negate the MPS (in-place)."""
        self.scale(-1.0)

    def __repr__(self) -> str:
        bond_str = ", ".join(str(d) for d in self.bond_dims) if self.bond_dims else "none"
        phys_str = ", ".join(str(d) for d in self.physical_dims)
        return (
            f"MPS(sites={self.num_sites}, "
            f"phys_dims=[{phys_str}], "
            f"bond_dims=[{bond_str}], "
            f"max_chi={self.max_bond_dim}, "
            f"params={self.total_params}, "
            f"form={self.canonical_form.value})"
        )

    def info(self) -> dict:
        """Return a dictionary of MPS information."""
        return {
            "num_sites": self.num_sites,
            "physical_dims": self.physical_dims,
            "bond_dims": self.bond_dims,
            "max_bond_dim": self.max_bond_dim,
            "total_params": self.total_params,
            "full_size": self.full_size,
            "compression_ratio": self.compression_ratio,
            "canonical_form": self.canonical_form.value,
            "orthogonality_center": self.orthogonality_center,
            "truncation_error": self.truncation_error_accumulated,
        }

    def validate(self) -> list[str]:
        """
        Validate the MPS for consistency.

        Returns:
            List of error messages (empty if valid).
        """
        errors = []
        for k, core in enumerate(self.cores):
            if core.ndim != 3:
                errors.append(f"Core {k}: expected 3D, got {core.ndim}D")
            if np.any(np.isnan(core)):
                errors.append(f"Core {k}: contains NaN values")
            if np.any(np.isinf(core)):
                errors.append(f"Core {k}: contains Inf values")

        for k in range(self.num_sites - 1):
            if self.cores[k].shape[2] != self.cores[k + 1].shape[0]:
                errors.append(
                    f"Bond {k}: dimension mismatch "
                    f"{self.cores[k].shape[2]} != {self.cores[k+1].shape[0]}"
                )

        if self.cores[0].shape[0] != 1:
            errors.append(f"First core left dim: {self.cores[0].shape[0]} != 1")
        if self.cores[-1].shape[2] != 1:
            errors.append(f"Last core right dim: {self.cores[-1].shape[2]} != 1")

        return errors

    def check_canonical(self, tol: float = 1e-10) -> dict:
        """
        Check the canonical form by verifying isometry conditions.

        Returns:
            Dictionary with per-site isometry errors.
        """
        result = {"left_iso_errors": [], "right_iso_errors": []}

        for k in range(self.num_sites):
            core = self.cores[k]
            chi_l, d, chi_r = core.shape

            # Check left-isometry: sum over physical and left indices
            mat = core.reshape(chi_l * d, chi_r)
            left_product = mat.T @ mat
            left_error = np.linalg.norm(left_product - np.eye(chi_r))
            result["left_iso_errors"].append(float(left_error))

            # Check right-isometry: sum over physical and right indices
            mat = core.reshape(chi_l, d * chi_r)
            right_product = mat @ mat.T
            right_error = np.linalg.norm(right_product - np.eye(chi_l))
            result["right_iso_errors"].append(float(right_error))

        return result


def random_mps(
    num_sites: int,
    physical_dims: int | Sequence[int],
    bond_dim: int,
    normalize: bool = True,
    seed: Optional[int] = None,
    dtype: type = np.float64,
) -> MPS:
    """
    Create a random MPS with specified dimensions.

    Args:
        num_sites: Number of sites.
        physical_dims: Physical dimension(s). If int, same for all sites.
        bond_dim: Maximum bond dimension.
        normalize: If True, normalize to unit norm.
        seed: Random seed.
        dtype: Data type for core tensors.

    Returns:
        Random MPS.
    """
    rng = np.random.default_rng(seed)

    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    if len(phys) != num_sites:
        raise ValueError(f"physical_dims length {len(phys)} != num_sites {num_sites}")

    # Compute actual bond dimensions (constrained by physical dims)
    left_dims = [1]
    for k in range(num_sites - 1):
        max_left = left_dims[-1] * phys[k]
        left_dims.append(min(bond_dim, max_left))

    right_dims = [1]
    for k in range(num_sites - 1, 0, -1):
        max_right = right_dims[0] * phys[k]
        right_dims.insert(0, min(bond_dim, max_right))

    actual_dims = [min(l, r) for l, r in zip(left_dims, right_dims)]
    actual_dims.append(1)

    cores = []
    for k in range(num_sites):
        chi_l = actual_dims[k]
        chi_r = actual_dims[k + 1]
        core = rng.standard_normal((chi_l, phys[k], chi_r)).astype(dtype)
        core /= np.linalg.norm(core) + 1e-300
        cores.append(core)

    mps = MPS(cores, copy_cores=False)

    if normalize:
        from tn_check.tensor.operations import mps_norm
        n = mps_norm(mps)
        if n > 1e-300:
            mps.cores[0] /= n

    return mps


def zero_mps(
    num_sites: int,
    physical_dims: int | Sequence[int],
) -> MPS:
    """Create a zero MPS (all cores are zero, bond dim 1)."""
    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    cores = [np.zeros((1, d, 1), dtype=np.float64) for d in phys]
    return MPS(cores, canonical_form=CanonicalForm.LEFT, copy_cores=False)


def ones_mps(
    num_sites: int,
    physical_dims: int | Sequence[int],
) -> MPS:
    """Create an MPS where all entries are 1 (rank-1)."""
    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    cores = [np.ones((1, d, 1), dtype=np.float64) for d in phys]
    return MPS(cores, copy_cores=False)


def unit_mps(
    num_sites: int,
    physical_dims: int | Sequence[int],
    indices: Sequence[int],
) -> MPS:
    """
    Create a rank-1 MPS that is 1 at the given multi-index and 0 elsewhere.

    This is used for evaluating the MPS at a specific configuration.

    Args:
        num_sites: Number of sites.
        physical_dims: Physical dimensions.
        indices: Multi-index specifying the nonzero entry.

    Returns:
        Rank-1 MPS with a single nonzero entry of 1.
    """
    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    if len(indices) != num_sites:
        raise ValueError(f"indices length {len(indices)} != num_sites {num_sites}")

    cores = []
    for k in range(num_sites):
        core = np.zeros((1, phys[k], 1), dtype=np.float64)
        if indices[k] < 0 or indices[k] >= phys[k]:
            raise ValueError(
                f"Index {indices[k]} out of range [0, {phys[k]}) at site {k}"
            )
        core[0, indices[k], 0] = 1.0
        cores.append(core)

    return MPS(cores, canonical_form=CanonicalForm.LEFT, copy_cores=False)


def product_mps(
    vectors: list[NDArray],
) -> MPS:
    """
    Create a rank-1 MPS from a list of site vectors.

    Args:
        vectors: List of 1D arrays, one per site.

    Returns:
        Rank-1 MPS.
    """
    cores = []
    for v in vectors:
        v = np.asarray(v, dtype=np.float64)
        if v.ndim != 1:
            raise ValueError(f"Each vector must be 1D, got {v.ndim}D")
        cores.append(v.reshape(1, -1, 1))

    return MPS(cores, copy_cores=False)


def uniform_mps(
    num_sites: int,
    physical_dims: int | Sequence[int],
) -> MPS:
    """
    Create a normalized uniform MPS (equal probability for all states).

    Args:
        num_sites: Number of sites.
        physical_dims: Physical dimensions.

    Returns:
        MPS representing uniform distribution.
    """
    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    total = 1
    for d in phys:
        total *= d

    val = 1.0 / total
    cores = []
    for k, d in enumerate(phys):
        core = np.ones((1, d, 1), dtype=np.float64)
        if k == 0:
            core *= val
        cores.append(core)

    return MPS(cores, copy_cores=False)


def _validate_mps_pair(mps_a: MPS, mps_b: MPS) -> None:
    """Validate that two MPS have compatible physical dimensions."""
    if mps_a.num_sites != mps_b.num_sites:
        raise ValueError(
            f"MPS site count mismatch: {mps_a.num_sites} != {mps_b.num_sites}"
        )
    if mps_a.physical_dims != mps_b.physical_dims:
        raise ValueError(
            f"Physical dimension mismatch: {mps_a.physical_dims} != {mps_b.physical_dims}"
        )


def characteristic_mps(
    num_sites: int,
    physical_dims: int | Sequence[int],
    site: int,
    predicate_mask: NDArray,
) -> MPS:
    """
    Create a rank-1 characteristic tensor for an axis-aligned predicate.

    The resulting MPS has value 1 at configurations where species `site`
    satisfies the predicate, and 0 otherwise. All other species are unconstrained.

    This is fundamental for CSL atomic proposition evaluation.

    Args:
        num_sites: Number of sites.
        physical_dims: Physical dimensions.
        site: The site (species) index for the predicate.
        predicate_mask: Boolean array of length d_{site} indicating which
            copy numbers satisfy the predicate.

    Returns:
        Rank-1 MPS encoding the characteristic function.
    """
    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    predicate_mask = np.asarray(predicate_mask, dtype=np.float64)
    if predicate_mask.shape[0] != phys[site]:
        raise ValueError(
            f"Predicate mask length {predicate_mask.shape[0]} != "
            f"physical dim {phys[site]} at site {site}"
        )

    cores = []
    for k in range(num_sites):
        if k == site:
            core = predicate_mask.reshape(1, -1, 1)
        else:
            core = np.ones((1, phys[k], 1), dtype=np.float64)
        cores.append(core)

    return MPS(cores, canonical_form=CanonicalForm.LEFT, copy_cores=False)


def multi_site_characteristic_mps(
    num_sites: int,
    physical_dims: int | Sequence[int],
    site_masks: dict[int, NDArray],
) -> MPS:
    """
    Create a rank-1 characteristic tensor for a conjunction of axis-aligned predicates.

    Each site in site_masks gets its own mask; unconstrained sites get all-ones.

    Args:
        num_sites: Number of sites.
        physical_dims: Physical dimensions.
        site_masks: Dict mapping site index to predicate mask.

    Returns:
        Rank-1 MPS encoding the joint characteristic function.
    """
    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    cores = []
    for k in range(num_sites):
        if k in site_masks:
            mask = np.asarray(site_masks[k], dtype=np.float64)
            if mask.shape[0] != phys[k]:
                raise ValueError(
                    f"Mask length {mask.shape[0]} != physical dim {phys[k]} at site {k}"
                )
            core = mask.reshape(1, -1, 1)
        else:
            core = np.ones((1, phys[k], 1), dtype=np.float64)
        cores.append(core)

    return MPS(cores, canonical_form=CanonicalForm.LEFT, copy_cores=False)


def threshold_mps(
    num_sites: int,
    physical_dims: int | Sequence[int],
    site: int,
    threshold: int,
    direction: str = "greater",
) -> MPS:
    """
    Create a characteristic MPS for a threshold predicate on a single species.

    Args:
        num_sites: Number of sites.
        physical_dims: Physical dimensions.
        site: Species index.
        threshold: Threshold value.
        direction: "greater" for > threshold, "greater_equal" for >= threshold,
                   "less" for < threshold, "less_equal" for <= threshold,
                   "equal" for == threshold.

    Returns:
        Rank-1 characteristic MPS.
    """
    if isinstance(physical_dims, int):
        d = physical_dims
    else:
        d = physical_dims[site] if hasattr(physical_dims, "__getitem__") else physical_dims

    mask = np.zeros(d, dtype=np.float64)
    for n in range(d):
        if direction == "greater" and n > threshold:
            mask[n] = 1.0
        elif direction == "greater_equal" and n >= threshold:
            mask[n] = 1.0
        elif direction == "less" and n < threshold:
            mask[n] = 1.0
        elif direction == "less_equal" and n <= threshold:
            mask[n] = 1.0
        elif direction == "equal" and n == threshold:
            mask[n] = 1.0

    return characteristic_mps(num_sites, physical_dims, site, mask)


def interval_mps(
    num_sites: int,
    physical_dims: int | Sequence[int],
    site: int,
    low: int,
    high: int,
) -> MPS:
    """
    Create a characteristic MPS for species copy number in [low, high].

    Args:
        num_sites: Number of sites.
        physical_dims: Physical dimensions.
        site: Species index.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).

    Returns:
        Rank-1 characteristic MPS.
    """
    if isinstance(physical_dims, int):
        d = physical_dims
    else:
        d = list(physical_dims)[site]

    mask = np.zeros(d, dtype=np.float64)
    for n in range(max(0, low), min(d, high + 1)):
        mask[n] = 1.0

    return characteristic_mps(num_sites, physical_dims, site, mask)

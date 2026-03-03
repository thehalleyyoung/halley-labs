"""
Tensor product mechanism assembly for multi-dimensional DP mechanisms.

When a d-dimensional mechanism decomposes into independent marginals, the
full joint mechanism is the Kronecker (tensor) product of the marginal
probability tables.  This module provides efficient construction, sampling,
and probability evaluation for such product mechanisms.

Key Features:
    - Memory-efficient sparse Kronecker products for large k^d outputs.
    - Lazy evaluation: avoids materialising the full k^d × k^d table.
    - Independent marginal sampling: O(d) per sample instead of O(k^d).
    - Exact probability computation via product of marginal probabilities.

Classes:
    TensorProductMechanism — assembled product mechanism with sampling
    MarginalMechanism      — wrapper for a single-coordinate mechanism
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import csr_matrix, kron as sparse_kron

from dp_forge.exceptions import (
    ConfigurationError,
    InvalidMechanismError,
)
from dp_forge.types import (
    ExtractedMechanism,
    SamplingConfig,
    SamplingMethod,
)

logger = logging.getLogger(__name__)

# Threshold for switching to sparse representation
_DENSE_PRODUCT_LIMIT: int = 50_000
_LAZY_EVAL_DIM_THRESHOLD: int = 6


@dataclass
class MarginalMechanism:
    """A single-coordinate mechanism (marginal of a product mechanism).

    Attributes:
        p_table: Probability table of shape (n_i, k_i) for coordinate i.
        y_grid: Output grid for this coordinate, shape (k_i,).
        coordinate_index: Which coordinate this marginal corresponds to.
        epsilon: Privacy parameter for this coordinate.
        sensitivity: Sensitivity of this coordinate's query.
        cdf_table: Precomputed CDF table for CDF-based sampling.
    """

    p_table: npt.NDArray[np.float64]
    y_grid: npt.NDArray[np.float64]
    coordinate_index: int
    epsilon: float = 1.0
    sensitivity: float = 1.0
    cdf_table: Optional[npt.NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        self.p_table = np.asarray(self.p_table, dtype=np.float64)
        self.y_grid = np.asarray(self.y_grid, dtype=np.float64)
        if self.p_table.ndim != 2:
            raise ValueError(
                f"p_table must be 2-D, got shape {self.p_table.shape}"
            )
        if self.p_table.shape[1] != len(self.y_grid):
            raise ValueError(
                f"p_table columns ({self.p_table.shape[1]}) must match "
                f"y_grid length ({len(self.y_grid)})"
            )
        row_sums = self.p_table.sum(axis=1)
        max_dev = float(np.max(np.abs(row_sums - 1.0)))
        if max_dev > 1e-6:
            raise ValueError(
                f"p_table rows must sum to 1 (max deviation: {max_dev:.2e})"
            )
        if self.cdf_table is None:
            self.cdf_table = np.cumsum(self.p_table, axis=1)

    @property
    def n(self) -> int:
        """Number of input values for this coordinate."""
        return self.p_table.shape[0]

    @property
    def k(self) -> int:
        """Number of output bins for this coordinate."""
        return self.p_table.shape[1]

    def sample(
        self,
        input_idx: int,
        rng: np.random.Generator,
    ) -> Tuple[int, float]:
        """Sample an output for the given input index.

        Args:
            input_idx: Index into this coordinate's input domain.
            rng: NumPy random generator.

        Returns:
            Tuple of (output_bin_index, output_value).
        """
        assert self.cdf_table is not None
        u = rng.random()
        bin_idx = int(np.searchsorted(self.cdf_table[input_idx], u))
        bin_idx = min(bin_idx, self.k - 1)
        return bin_idx, float(self.y_grid[bin_idx])

    def probability(self, input_idx: int, output_bin: int) -> float:
        """Return Pr[M_i(x_{input_idx}) = y_{output_bin}]."""
        return float(self.p_table[input_idx, output_bin])

    def expected_loss(
        self,
        input_idx: int,
        true_value: float,
        loss_fn: str = "L2",
    ) -> float:
        """Compute expected loss for a given input.

        Args:
            input_idx: Input index.
            true_value: True query value for this input.
            loss_fn: Loss function name ("L1", "L2", "Linf").

        Returns:
            Expected loss value.
        """
        probs = self.p_table[input_idx]
        if loss_fn == "L2":
            losses = (self.y_grid - true_value) ** 2
        elif loss_fn == "L1":
            losses = np.abs(self.y_grid - true_value)
        else:
            losses = np.abs(self.y_grid - true_value)
        return float(np.dot(probs, losses))

    def __repr__(self) -> str:
        return (
            f"MarginalMechanism(coord={self.coordinate_index}, "
            f"n={self.n}, k={self.k}, ε={self.epsilon:.4f})"
        )


class TensorProductMechanism:
    """Tensor product of marginal mechanisms for multi-dimensional queries.

    The joint mechanism M(x) = M₁(x₁) × M₂(x₂) × ... × M_d(x_d) is
    represented lazily via its marginals, avoiding materialisation of the
    full k₁·k₂·...·k_d output space.

    Args:
        marginals: List of per-coordinate marginal mechanisms.
        total_epsilon: Total privacy budget consumed.
        total_delta: Total delta consumed.

    Raises:
        ConfigurationError: If marginals list is empty.
    """

    def __init__(
        self,
        marginals: List[MarginalMechanism],
        total_epsilon: float = 1.0,
        total_delta: float = 0.0,
    ) -> None:
        if not marginals:
            raise ConfigurationError(
                "marginals must be non-empty",
                parameter="marginals",
            )
        self._marginals = marginals
        self._total_epsilon = total_epsilon
        self._total_delta = total_delta
        self._d = len(marginals)
        self._product_k = math.prod(m.k for m in marginals)
        self._use_lazy = (
            self._d >= _LAZY_EVAL_DIM_THRESHOLD
            or self._product_k > _DENSE_PRODUCT_LIMIT
        )
        self._dense_table: Optional[npt.NDArray[np.float64]] = None
        self._sparse_table: Optional[sparse.spmatrix] = None

    @property
    def d(self) -> int:
        """Number of dimensions (coordinates)."""
        return self._d

    @property
    def product_k(self) -> int:
        """Total number of output bins in the product space."""
        return self._product_k

    @property
    def marginals(self) -> List[MarginalMechanism]:
        """The marginal mechanisms."""
        return self._marginals

    @property
    def total_epsilon(self) -> float:
        """Total privacy budget."""
        return self._total_epsilon

    @property
    def total_delta(self) -> float:
        """Total delta parameter."""
        return self._total_delta

    @property
    def is_lazy(self) -> bool:
        """Whether the mechanism uses lazy evaluation."""
        return self._use_lazy

    def marginal(self, dim: int) -> MarginalMechanism:
        """Return the marginal mechanism for coordinate dim.

        Args:
            dim: Coordinate index.

        Returns:
            MarginalMechanism for that coordinate.

        Raises:
            IndexError: If dim is out of range.
        """
        if not 0 <= dim < self._d:
            raise IndexError(
                f"dim must be in [0, {self._d}), got {dim}"
            )
        return self._marginals[dim]

    def sample(
        self,
        input_indices: Sequence[int],
        rng: Optional[np.random.Generator] = None,
        n_samples: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Sample from the product mechanism.

        Each marginal is sampled independently, producing d output values
        per sample. This is O(d) per sample regardless of k^d.

        Args:
            input_indices: Per-coordinate input indices, length d.
            rng: NumPy random generator. Created from default if None.
            n_samples: Number of independent samples to draw.

        Returns:
            Array of shape (n_samples, d) with sampled output values.

        Raises:
            ValueError: If input_indices has wrong length.
        """
        if len(input_indices) != self._d:
            raise ValueError(
                f"input_indices must have length {self._d}, "
                f"got {len(input_indices)}"
            )
        if rng is None:
            rng = np.random.default_rng()
        samples = np.zeros((n_samples, self._d), dtype=np.float64)
        for s in range(n_samples):
            for dim in range(self._d):
                _, val = self._marginals[dim].sample(
                    input_indices[dim], rng
                )
                samples[s, dim] = val
        return samples

    def sample_indices(
        self,
        input_indices: Sequence[int],
        rng: Optional[np.random.Generator] = None,
        n_samples: int = 1,
    ) -> npt.NDArray[np.int64]:
        """Sample output bin indices from the product mechanism.

        Args:
            input_indices: Per-coordinate input indices, length d.
            rng: NumPy random generator.
            n_samples: Number of samples.

        Returns:
            Array of shape (n_samples, d) with sampled output bin indices.
        """
        if len(input_indices) != self._d:
            raise ValueError(
                f"input_indices must have length {self._d}, "
                f"got {len(input_indices)}"
            )
        if rng is None:
            rng = np.random.default_rng()
        indices = np.zeros((n_samples, self._d), dtype=np.int64)
        for s in range(n_samples):
            for dim in range(self._d):
                idx, _ = self._marginals[dim].sample(
                    input_indices[dim], rng
                )
                indices[s, dim] = idx
        return indices

    def probability(
        self,
        input_indices: Sequence[int],
        output_bins: Sequence[int],
    ) -> float:
        """Compute Pr[M(x) = y] = Π_i Pr[M_i(x_i) = y_i].

        Args:
            input_indices: Per-coordinate input indices.
            output_bins: Per-coordinate output bin indices.

        Returns:
            Joint probability as product of marginal probabilities.
        """
        if len(input_indices) != self._d:
            raise ValueError(
                f"input_indices must have length {self._d}, "
                f"got {len(input_indices)}"
            )
        if len(output_bins) != self._d:
            raise ValueError(
                f"output_bins must have length {self._d}, "
                f"got {len(output_bins)}"
            )
        prob = 1.0
        for dim in range(self._d):
            prob *= self._marginals[dim].probability(
                input_indices[dim], output_bins[dim]
            )
        return prob

    def log_probability(
        self,
        input_indices: Sequence[int],
        output_bins: Sequence[int],
    ) -> float:
        """Compute log Pr[M(x) = y] = Σ_i log Pr[M_i(x_i) = y_i].

        Numerically stable via log-sum.

        Args:
            input_indices: Per-coordinate input indices.
            output_bins: Per-coordinate output bin indices.

        Returns:
            Log joint probability.
        """
        if len(input_indices) != self._d:
            raise ValueError(
                f"input_indices length mismatch: expected {self._d}"
            )
        if len(output_bins) != self._d:
            raise ValueError(
                f"output_bins length mismatch: expected {self._d}"
            )
        log_prob = 0.0
        for dim in range(self._d):
            p = self._marginals[dim].probability(
                input_indices[dim], output_bins[dim]
            )
            if p <= 0:
                return float("-inf")
            log_prob += math.log(p)
        return log_prob

    def materialise_dense(
        self,
        input_indices: Optional[Sequence[int]] = None,
    ) -> npt.NDArray[np.float64]:
        """Materialise the full product distribution for given inputs.

        Warning: This creates a k₁·k₂·...·k_d vector. Only use for
        small product spaces.

        Args:
            input_indices: Per-coordinate input indices. If None, uses
                index 0 for all coordinates.

        Returns:
            Probability vector of length k₁·k₂·...·k_d.

        Raises:
            ConfigurationError: If product space exceeds _DENSE_PRODUCT_LIMIT
                and force is not applied.
        """
        if self._product_k > _DENSE_PRODUCT_LIMIT:
            raise ConfigurationError(
                f"Product space {self._product_k} exceeds dense limit "
                f"{_DENSE_PRODUCT_LIMIT}. Use sample() or probability() instead.",
                parameter="product_k",
                value=self._product_k,
            )
        if input_indices is None:
            input_indices = [0] * self._d
        if len(input_indices) != self._d:
            raise ValueError(
                f"input_indices must have length {self._d}"
            )
        # Build via successive Kronecker products
        result = self._marginals[0].p_table[input_indices[0]]
        for dim in range(1, self._d):
            result = np.kron(result, self._marginals[dim].p_table[input_indices[dim]])
        return result

    def materialise_sparse(
        self,
        input_indices: Optional[Sequence[int]] = None,
        threshold: float = 1e-15,
    ) -> sparse.spmatrix:
        """Materialise the product distribution as a sparse vector.

        Only stores entries above threshold. Efficient when marginals
        concentrate probability mass on few bins.

        Args:
            input_indices: Per-coordinate input indices.
            threshold: Minimum probability to include in sparse output.

        Returns:
            Sparse CSR matrix of shape (1, product_k).
        """
        if input_indices is None:
            input_indices = [0] * self._d
        if len(input_indices) != self._d:
            raise ValueError(
                f"input_indices must have length {self._d}"
            )
        # Start with first marginal's row
        row = self._marginals[0].p_table[input_indices[0]]
        result = csr_matrix(row.reshape(1, -1))
        for dim in range(1, self._d):
            next_row = self._marginals[dim].p_table[input_indices[dim]]
            next_sparse = csr_matrix(next_row.reshape(1, -1))
            result = sparse_kron(result, next_sparse, format="csr")
            # Threshold small values to maintain sparsity
            result.data[np.abs(result.data) < threshold] = 0.0
            result.eliminate_zeros()
        return result

    def expected_loss_per_coordinate(
        self,
        input_indices: Sequence[int],
        true_values: Sequence[float],
        loss_fn: str = "L2",
    ) -> npt.NDArray[np.float64]:
        """Compute expected loss for each coordinate independently.

        Under product structure, the total L2 loss decomposes as
        Σ_i E[loss_i(M_i(x_i), f_i(x_i))].

        Args:
            input_indices: Per-coordinate input indices.
            true_values: Per-coordinate true query values.
            loss_fn: Loss function name.

        Returns:
            Array of per-coordinate expected losses.
        """
        losses = np.zeros(self._d, dtype=np.float64)
        for dim in range(self._d):
            losses[dim] = self._marginals[dim].expected_loss(
                input_indices[dim], true_values[dim], loss_fn
            )
        return losses

    def total_expected_loss(
        self,
        input_indices: Sequence[int],
        true_values: Sequence[float],
        loss_fn: str = "L2",
    ) -> float:
        """Compute total expected loss (sum of coordinate losses)."""
        return float(
            self.expected_loss_per_coordinate(
                input_indices, true_values, loss_fn
            ).sum()
        )

    def to_extracted_mechanism(
        self,
        input_indices: Optional[Sequence[int]] = None,
    ) -> ExtractedMechanism:
        """Convert to an ExtractedMechanism (only for small product spaces).

        Materialises the full product table for all input combinations.

        Args:
            input_indices: If provided, only materialise for these inputs.

        Returns:
            ExtractedMechanism with the product table.

        Raises:
            ConfigurationError: If product space is too large.
        """
        if self._product_k > _DENSE_PRODUCT_LIMIT:
            raise ConfigurationError(
                f"Product space too large ({self._product_k}) for dense extraction",
                parameter="product_k",
                value=self._product_k,
            )
        # Build all input combinations
        n_per_coord = [m.n for m in self._marginals]
        total_n = math.prod(n_per_coord)
        table = np.zeros((total_n, self._product_k), dtype=np.float64)
        for flat_idx in range(total_n):
            coords = self._unflatten_index(flat_idx, n_per_coord)
            table[flat_idx] = self.materialise_dense(coords)
        return ExtractedMechanism(
            p_final=table,
            metadata={
                "type": "tensor_product",
                "d": self._d,
                "marginal_shapes": [(m.n, m.k) for m in self._marginals],
            },
        )

    @staticmethod
    def _unflatten_index(
        flat: int, dims: Sequence[int]
    ) -> List[int]:
        """Convert a flat index into per-dimension indices (row-major)."""
        coords = []
        for d_size in reversed(dims):
            coords.append(flat % d_size)
            flat //= d_size
        return list(reversed(coords))

    def __repr__(self) -> str:
        shapes = [(m.n, m.k) for m in self._marginals]
        return (
            f"TensorProductMechanism(d={self._d}, shapes={shapes}, "
            f"product_k={self._product_k}, lazy={self._use_lazy})"
        )


def kronecker_sparse(
    matrices: Sequence[npt.NDArray[np.float64]],
    threshold: float = 1e-15,
) -> sparse.spmatrix:
    """Compute Kronecker product of multiple matrices in sparse format.

    Memory-efficient: thresholds small entries after each pairwise product
    to prevent intermediate blowup.

    Args:
        matrices: Sequence of 2-D arrays to Kronecker-multiply.
        threshold: Entries below this are zeroed for sparsity.

    Returns:
        Sparse CSR matrix representing the Kronecker product.

    Raises:
        ValueError: If fewer than 2 matrices provided.
    """
    if len(matrices) < 1:
        raise ValueError("At least one matrix required")
    if len(matrices) == 1:
        return csr_matrix(matrices[0])
    result = csr_matrix(matrices[0])
    for i in range(1, len(matrices)):
        next_mat = csr_matrix(matrices[i])
        result = sparse_kron(result, next_mat, format="csr")
        result.data[np.abs(result.data) < threshold] = 0.0
        result.eliminate_zeros()
    return result


def build_product_mechanism(
    marginal_tables: Sequence[npt.NDArray[np.float64]],
    marginal_grids: Sequence[npt.NDArray[np.float64]],
    epsilons: Sequence[float],
    sensitivities: Optional[Sequence[float]] = None,
    total_epsilon: float = 1.0,
    total_delta: float = 0.0,
) -> TensorProductMechanism:
    """Convenience factory for TensorProductMechanism from raw arrays.

    Args:
        marginal_tables: Per-coordinate probability tables.
        marginal_grids: Per-coordinate output grids.
        epsilons: Per-coordinate privacy parameters.
        sensitivities: Per-coordinate sensitivities (default 1.0 each).
        total_epsilon: Total composed epsilon.
        total_delta: Total composed delta.

    Returns:
        TensorProductMechanism assembled from the marginals.
    """
    d = len(marginal_tables)
    if len(marginal_grids) != d:
        raise ValueError("marginal_tables and marginal_grids must have same length")
    if len(epsilons) != d:
        raise ValueError("epsilons must have same length as marginal_tables")
    if sensitivities is None:
        sensitivities = [1.0] * d
    marginals = []
    for i in range(d):
        marginals.append(
            MarginalMechanism(
                p_table=marginal_tables[i],
                y_grid=marginal_grids[i],
                coordinate_index=i,
                epsilon=epsilons[i],
                sensitivity=sensitivities[i],
            )
        )
    return TensorProductMechanism(
        marginals=marginals,
        total_epsilon=total_epsilon,
        total_delta=total_delta,
    )

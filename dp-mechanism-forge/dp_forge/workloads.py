"""
Workload matrix generation and analysis for DP-Forge.

This module provides comprehensive workload generation for all standard
query types used in differential privacy mechanism synthesis, along with
analysis tools for computing workload properties and optimal strategy
bounds.

A **workload** is a matrix A ∈ R^{m×d} where each row represents a
linear query over a d-dimensional data histogram.  The mechanism's goal
is to answer all m queries accurately while satisfying differential
privacy.

Key Components:
    - ``WorkloadGenerator``: Factory methods for standard workload
      matrices (identity, prefix sums, all-range, histograms, marginals,
      wavelets, Fourier, random).
    - ``WorkloadAnalyzer``: Compute workload properties (rank, condition
      number, sensitivity, sparsity, structure detection) and optimal
      strategy bounds.
    - Composition utilities for combining and transforming workloads.
    - Standard benchmark workloads from the DP-Forge evaluation plan.

Mathematical Background:
    For a workload A and strategy matrix B (the mechanism answers q = Bx
    with noise calibrated to B's sensitivity), the total squared error is:

        E[||Ax - A x̂||²] = (2/ε²) · trace(A (BᵀB)⁻¹ Aᵀ)

    The optimal strategy minimizes this over all full-rank B.  This is
    equivalent to an SDP.  The ``WorkloadAnalyzer`` provides tools for
    computing bounds on the optimal error.

All workload matrices are returned as dense numpy arrays.  For very
large workloads, sparse representations via ``scipy.sparse`` are used
internally where beneficial.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy import linalg as sp_linalg

from .exceptions import ConfigurationError, SensitivityError
from .types import QueryType, WorkloadSpec


# ---------------------------------------------------------------------------
# WorkloadGenerator
# ---------------------------------------------------------------------------


class WorkloadGenerator:
    """Factory for standard workload matrices.

    All methods are static and return numpy arrays of shape ``(m, d)``
    suitable for use with :class:`~dp_forge.types.WorkloadSpec`.

    Examples:
        >>> gen = WorkloadGenerator()
        >>> I = gen.identity(5)
        >>> I.shape
        (5, 5)
        >>> P = gen.prefix_sums(4)
        >>> P.shape
        (4, 4)
    """

    @staticmethod
    def identity(d: int) -> npt.NDArray[np.float64]:
        """Identity workload: answer each coordinate independently.

        The identity workload I_d has one query per data element.
        Each query asks for the count in exactly one bin.

        Shape: ``(d, d)``

        Sensitivity:
            GS_1 = 1, GS_2 = 1, GS_∞ = 1

        Args:
            d: Dimension (number of bins / domain size).

        Returns:
            Identity matrix of shape ``(d, d)``.

        Raises:
            ConfigurationError: If ``d < 1``.

        Examples:
            >>> WorkloadGenerator.identity(3)
            array([[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]])
        """
        if d < 1:
            raise ConfigurationError(
                f"d must be >= 1, got {d}",
                parameter="d",
                value=d,
                constraint="d >= 1",
            )
        return np.eye(d, dtype=np.float64)

    @staticmethod
    def prefix_sums(d: int) -> npt.NDArray[np.float64]:
        """Prefix sum workload (lower triangular matrix of ones).

        Query i computes the sum x_0 + x_1 + ... + x_i.  This is the
        canonical range-query workload, also known as cumulative sums.

        Shape: ``(d, d)``

        Sensitivity:
            GS_1 = d, GS_2 = √d, GS_∞ = 1

        Args:
            d: Dimension of the data domain.

        Returns:
            Lower triangular matrix of shape ``(d, d)``.

        Raises:
            ConfigurationError: If ``d < 1``.

        Examples:
            >>> WorkloadGenerator.prefix_sums(4)
            array([[1., 0., 0., 0.],
                   [1., 1., 0., 0.],
                   [1., 1., 1., 0.],
                   [1., 1., 1., 1.]])
        """
        if d < 1:
            raise ConfigurationError(
                f"d must be >= 1, got {d}",
                parameter="d",
                value=d,
                constraint="d >= 1",
            )
        return np.tril(np.ones((d, d), dtype=np.float64))

    @staticmethod
    def all_range(d: int) -> npt.NDArray[np.float64]:
        """All range queries: every contiguous sub-array sum.

        For a d-element domain, there are d(d+1)/2 range queries
        [a, b] for 0 <= a <= b < d.  Row (a, b) has ones in columns
        a through b.

        Shape: ``(d*(d+1)//2, d)``

        Sensitivity:
            GS_1 = max_j (j+1)(d-j), GS_2 = max_j √((j+1)(d-j)), GS_∞ = 1

        Args:
            d: Dimension of the data domain.

        Returns:
            All-range query matrix of shape ``(d*(d+1)//2, d)``.

        Raises:
            ConfigurationError: If ``d < 1``.

        Examples:
            >>> A = WorkloadGenerator.all_range(3)
            >>> A.shape
            (6, 3)
        """
        if d < 1:
            raise ConfigurationError(
                f"d must be >= 1, got {d}",
                parameter="d",
                value=d,
                constraint="d >= 1",
            )
        m = d * (d + 1) // 2
        A = np.zeros((m, d), dtype=np.float64)
        row = 0
        for a in range(d):
            for b in range(a, d):
                A[row, a : b + 1] = 1.0
                row += 1
        return A

    @staticmethod
    def histogram_1d(n_bins: int) -> npt.NDArray[np.float64]:
        """1D histogram workload (identity matrix).

        For a 1D histogram with n_bins bins, the workload is simply the
        identity matrix — each query asks for one bin count.

        Shape: ``(n_bins, n_bins)``

        Args:
            n_bins: Number of histogram bins.

        Returns:
            Identity matrix of shape ``(n_bins, n_bins)``.

        Raises:
            ConfigurationError: If ``n_bins < 1``.

        Examples:
            >>> WorkloadGenerator.histogram_1d(4).shape
            (4, 4)
        """
        if n_bins < 1:
            raise ConfigurationError(
                f"n_bins must be >= 1, got {n_bins}",
                parameter="n_bins",
                value=n_bins,
                constraint="n_bins >= 1",
            )
        return np.eye(n_bins, dtype=np.float64)

    @staticmethod
    def histogram_2d(n1: int, n2: int) -> npt.NDArray[np.float64]:
        """2D histogram workload.

        For a 2D histogram with n1 × n2 bins, the data vector x has
        d = n1 * n2 elements (flattened row-major).  The workload is
        the identity matrix of size d — each query asks for one cell.

        Shape: ``(n1*n2, n1*n2)``

        Args:
            n1: Number of bins along the first dimension.
            n2: Number of bins along the second dimension.

        Returns:
            Identity matrix of shape ``(n1*n2, n1*n2)``.

        Raises:
            ConfigurationError: If ``n1 < 1`` or ``n2 < 1``.

        Examples:
            >>> WorkloadGenerator.histogram_2d(3, 4).shape
            (12, 12)
        """
        if n1 < 1:
            raise ConfigurationError(
                f"n1 must be >= 1, got {n1}",
                parameter="n1",
                value=n1,
                constraint="n1 >= 1",
            )
        if n2 < 1:
            raise ConfigurationError(
                f"n2 must be >= 1, got {n2}",
                parameter="n2",
                value=n2,
                constraint="n2 >= 1",
            )
        d = n1 * n2
        return np.eye(d, dtype=np.float64)

    @staticmethod
    def marginals(d: int, k: int) -> npt.NDArray[np.float64]:
        """k-way marginal queries over d binary attributes.

        A k-way marginal query selects k of d attributes and computes
        the joint histogram over those k attributes.  Each marginal has
        2^k cells.  The full workload has C(d, k) × 2^k rows and 2^d
        columns.

        The workload matrix A has a 1 in entry (q, x) if domain element
        x is consistent with the k-attribute pattern of query q.

        Shape: ``(C(d,k) * 2^k, 2^d)``

        Args:
            d: Number of binary attributes.
            k: Marginal order.

        Returns:
            Marginal workload matrix.

        Raises:
            ConfigurationError: If ``d < 1``, ``k < 1``, ``k > d``,
                or the workload is too large (2^d > 2^16).

        Examples:
            >>> A = WorkloadGenerator.marginals(3, 2)
            >>> A.shape
            (12, 8)
        """
        if d < 1:
            raise ConfigurationError(
                f"d must be >= 1, got {d}",
                parameter="d",
                value=d,
                constraint="d >= 1",
            )
        if k < 1 or k > d:
            raise ConfigurationError(
                f"k must be in [1, d={d}], got {k}",
                parameter="k",
                value=k,
                constraint=f"1 <= k <= {d}",
            )
        if d > 16:
            raise ConfigurationError(
                f"d={d} too large for explicit marginal workload (2^{d} columns)",
                parameter="d",
                value=d,
                constraint="d <= 16",
            )

        n_domain = 2 ** d
        n_cells = 2 ** k
        n_marginals = math.comb(d, k)
        m = n_marginals * n_cells

        A = np.zeros((m, n_domain), dtype=np.float64)

        row = 0
        for subset in itertools.combinations(range(d), k):
            # For each cell pattern in the marginal
            for cell in range(n_cells):
                # cell is a k-bit pattern for the selected attributes
                cell_bits = [(cell >> (k - 1 - p)) & 1 for p in range(k)]

                # A domain element x (d-bit integer) matches if x's bits at
                # the subset positions match cell_bits
                for x in range(n_domain):
                    match = True
                    for p, attr_idx in enumerate(subset):
                        x_bit = (x >> (d - 1 - attr_idx)) & 1
                        if x_bit != cell_bits[p]:
                            match = False
                            break
                    if match:
                        A[row, x] = 1.0
                row += 1

        return A

    @staticmethod
    def custom_linear(A: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """User-specified custom workload matrix.

        Validates and returns a copy of the input matrix.

        Args:
            A: Workload matrix of shape ``(m, d)``.

        Returns:
            Validated copy of A as float64.

        Raises:
            ConfigurationError: If A is not 2-D or contains non-finite values.

        Examples:
            >>> A = WorkloadGenerator.custom_linear(np.array([[1, 0], [1, 1]]))
            >>> A.shape
            (2, 2)
        """
        A = np.asarray(A, dtype=np.float64)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if A.ndim != 2:
            raise ConfigurationError(
                f"Workload matrix must be 2-D, got shape {A.shape}",
                parameter="A",
                constraint="A.ndim == 2",
            )
        if not np.all(np.isfinite(A)):
            raise ConfigurationError(
                "Workload matrix contains non-finite values",
                parameter="A",
                constraint="all(isfinite(A))",
            )
        return A.copy()

    @staticmethod
    def random_workload(
        d: int,
        m: int,
        *,
        density: float = 1.0,
        seed: Optional[int] = None,
        scale: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """Random workload matrix.

        Generates a random m × d workload matrix.  Each entry is drawn
        from N(0, scale²) and then zeroed with probability (1 - density)
        to create sparsity.

        Args:
            d: Domain dimension.
            m: Number of queries.
            density: Fraction of non-zero entries in [0, 1].
            seed: Random seed for reproducibility.
            scale: Standard deviation of the Gaussian entries.

        Returns:
            Random workload matrix of shape ``(m, d)``.

        Raises:
            ConfigurationError: If parameters are invalid.

        Examples:
            >>> A = WorkloadGenerator.random_workload(5, 10, seed=42)
            >>> A.shape
            (10, 5)
        """
        if d < 1:
            raise ConfigurationError(
                f"d must be >= 1, got {d}", parameter="d", value=d, constraint="d >= 1"
            )
        if m < 1:
            raise ConfigurationError(
                f"m must be >= 1, got {m}", parameter="m", value=m, constraint="m >= 1"
            )
        if not (0.0 < density <= 1.0):
            raise ConfigurationError(
                f"density must be in (0, 1], got {density}",
                parameter="density",
                value=density,
                constraint="0 < density <= 1",
            )

        rng = np.random.default_rng(seed)
        A = rng.normal(0, scale, size=(m, d))

        if density < 1.0:
            mask = rng.random((m, d)) < density
            A *= mask

        return A.astype(np.float64)

    @staticmethod
    def wavelet_workload(d: int) -> npt.NDArray[np.float64]:
        """Haar wavelet workload matrix.

        The Haar wavelet basis provides a multi-resolution decomposition
        of the data histogram.  For a domain of size d (must be a power
        of 2), the wavelet matrix has d rows:

        - Row 0: all-ones (scaling coefficient = total count)
        - Rows at scale s: alternating +1/-1 blocks of size d/2^s

        This is an orthogonal workload, meaning the optimal strategy is
        the identity and each query can be answered independently.

        Shape: ``(d, d)``

        Args:
            d: Domain size (must be a power of 2).

        Returns:
            Haar wavelet matrix of shape ``(d, d)``.

        Raises:
            ConfigurationError: If d is not a power of 2.

        Examples:
            >>> W = WorkloadGenerator.wavelet_workload(4)
            >>> W.shape
            (4, 4)
        """
        if d < 1:
            raise ConfigurationError(
                f"d must be >= 1, got {d}", parameter="d", value=d, constraint="d >= 1"
            )
        if d & (d - 1) != 0:
            raise ConfigurationError(
                f"d must be a power of 2, got {d}",
                parameter="d",
                value=d,
                constraint="d is a power of 2",
            )

        W = np.zeros((d, d), dtype=np.float64)

        # Row 0: scaling function (all ones, normalized)
        W[0, :] = 1.0 / math.sqrt(d)

        row = 1
        # Iterate over scales
        scale = 1
        while scale < d:
            block_size = d // (2 * scale)
            for shift in range(scale):
                if row >= d:
                    break
                start = 2 * shift * block_size
                norm = 1.0 / math.sqrt(d)
                W[row, start : start + block_size] = norm
                W[row, start + block_size : start + 2 * block_size] = -norm
                row += 1
            scale *= 2

        return W

    @staticmethod
    def fourier_workload(d: int) -> npt.NDArray[np.float64]:
        """Discrete Fourier basis workload (real part).

        Returns the real part of the d × d DFT matrix (normalized).
        Each row k computes the cosine component at frequency k/d.

        This is useful for frequency-domain analysis of data histograms.
        Note: only the real (cosine) part is included; for the full DFT,
        use the complex representation.

        Shape: ``(d, d)``

        Args:
            d: Domain size.

        Returns:
            Real DFT matrix of shape ``(d, d)``.

        Raises:
            ConfigurationError: If ``d < 1``.

        Examples:
            >>> F = WorkloadGenerator.fourier_workload(4)
            >>> F.shape
            (4, 4)
        """
        if d < 1:
            raise ConfigurationError(
                f"d must be >= 1, got {d}", parameter="d", value=d, constraint="d >= 1"
            )

        # Build DFT matrix and take real part
        indices = np.arange(d)
        omega = np.exp(-2j * np.pi / d)
        F = np.zeros((d, d), dtype=np.complex128)
        for k in range(d):
            F[k, :] = omega ** (k * indices) / math.sqrt(d)

        return F.real.astype(np.float64)


# ---------------------------------------------------------------------------
# WorkloadAnalyzer
# ---------------------------------------------------------------------------


@dataclass
class WorkloadProperties:
    """Container for workload analysis results.

    Attributes:
        shape: Workload matrix shape ``(m, d)``.
        rank: Matrix rank.
        condition_number: Condition number (L2 norm).
        sparsity: Fraction of zero entries.
        sensitivity_l1: L1 global sensitivity (max column L1 norm).
        sensitivity_l2: L2 global sensitivity (max column L2 norm).
        sensitivity_linf: Linf global sensitivity (max column Linf norm).
        is_square: Whether the workload is square.
        is_orthogonal: Whether rows are orthogonal.
        is_full_rank: Whether the workload has full column rank.
        structure: Detected structural properties.
        spectral_gap: Ratio of largest to second-largest singular value.
    """

    shape: Tuple[int, int]
    rank: int
    condition_number: float
    sparsity: float
    sensitivity_l1: float
    sensitivity_l2: float
    sensitivity_linf: float
    is_square: bool
    is_orthogonal: bool
    is_full_rank: bool
    structure: Dict[str, bool] = field(default_factory=dict)
    spectral_gap: float = 1.0

    def __repr__(self) -> str:
        structs = [k for k, v in self.structure.items() if v]
        struct_str = ", ".join(structs) if structs else "none"
        return (
            f"WorkloadProperties(shape={self.shape}, rank={self.rank}, "
            f"cond={self.condition_number:.2e}, sparsity={self.sparsity:.2%}, "
            f"structure=[{struct_str}])"
        )


class WorkloadAnalyzer:
    """Analyze workload matrices for structural properties and error bounds.

    Provides methods for computing rank, condition number, sensitivity,
    sparsity, and detecting special structure (Toeplitz, circulant,
    block-diagonal) that can be exploited for more efficient mechanism
    synthesis.

    Examples:
        >>> analyzer = WorkloadAnalyzer()
        >>> A = WorkloadGenerator.identity(5)
        >>> props = analyzer.analyze_workload(A)
        >>> props.rank
        5
        >>> props.sensitivity_l1
        1.0
    """

    def analyze_workload(
        self,
        A: npt.NDArray[np.float64],
        *,
        tol: float = 1e-10,
    ) -> WorkloadProperties:
        """Compute comprehensive properties of a workload matrix.

        Args:
            A: Workload matrix of shape ``(m, d)``.
            tol: Tolerance for rank and orthogonality detection.

        Returns:
            A :class:`WorkloadProperties` with all computed metrics.

        Raises:
            ConfigurationError: If A is not a valid matrix.
        """
        A = np.asarray(A, dtype=np.float64)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if A.ndim != 2:
            raise ConfigurationError(
                f"Workload must be 2-D, got shape {A.shape}",
                parameter="A",
            )
        if not np.all(np.isfinite(A)):
            raise ConfigurationError(
                "Workload contains non-finite values",
                parameter="A",
            )

        m, d = A.shape

        # Rank
        rank = int(np.linalg.matrix_rank(A, tol=tol))

        # Condition number
        svd_vals = np.linalg.svd(A, compute_uv=False)
        nonzero_sv = svd_vals[svd_vals > tol]
        if len(nonzero_sv) == 0:
            cond = float("inf")
        else:
            cond = float(nonzero_sv[0] / nonzero_sv[-1])

        # Spectral gap
        if len(nonzero_sv) >= 2:
            spectral_gap = float(nonzero_sv[0] / nonzero_sv[1])
        else:
            spectral_gap = float("inf")

        # Sparsity
        total = m * d
        nnz = np.count_nonzero(A)
        sparsity = 1.0 - nnz / total if total > 0 else 1.0

        # Sensitivity (column norms)
        col_l1 = np.sum(np.abs(A), axis=0)
        col_l2 = np.sqrt(np.sum(A ** 2, axis=0))
        col_linf = np.max(np.abs(A), axis=0) if m > 0 else np.zeros(d)

        sens_l1 = float(np.max(col_l1)) if d > 0 else 0.0
        sens_l2 = float(np.max(col_l2)) if d > 0 else 0.0
        sens_linf = float(np.max(col_linf)) if d > 0 else 0.0

        # Orthogonality check
        is_square = (m == d)
        is_orthogonal = False
        if is_square and m > 0:
            gram = A @ A.T
            off_diag = gram - np.diag(np.diag(gram))
            is_orthogonal = bool(np.max(np.abs(off_diag)) < tol)

        is_full_rank = (rank == min(m, d))

        # Detect structure
        structure = self.detect_structure(A, tol=tol)

        return WorkloadProperties(
            shape=(m, d),
            rank=rank,
            condition_number=cond,
            sparsity=sparsity,
            sensitivity_l1=sens_l1,
            sensitivity_l2=sens_l2,
            sensitivity_linf=sens_linf,
            is_square=is_square,
            is_orthogonal=is_orthogonal,
            is_full_rank=is_full_rank,
            structure=structure,
            spectral_gap=spectral_gap,
        )

    def compute_sensitivity_ball_vertices(
        self,
        A: npt.NDArray[np.float64],
        *,
        norm_ord: Union[int, float] = 1,
    ) -> npt.NDArray[np.float64]:
        """Compute vertices of the K-norm sensitivity ball.

        For the K-norm mechanism (Hardt & Talwar, 2010), the sensitivity
        ball is the image of the L_p unit ball under A.  The vertices of
        this convex body are the columns of A (and their negations) for
        L1 sensitivity.

        For an SDP-based approach, these vertices define the constraints.

        Args:
            A: Workload matrix of shape ``(m, d)``.
            norm_ord: Norm order for the sensitivity ball.

        Returns:
            Array of vertices, shape ``(2d, m)`` for L1 norm.

        Examples:
            >>> A = np.eye(3)
            >>> v = WorkloadAnalyzer().compute_sensitivity_ball_vertices(A)
            >>> v.shape
            (6, 3)
        """
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ConfigurationError(
                f"Workload must be 2-D, got shape {A.shape}",
                parameter="A",
            )

        m, d = A.shape

        if norm_ord == 1 or norm_ord == np.inf:
            # Vertices are ±A_j for each column j
            vertices = np.zeros((2 * d, m), dtype=np.float64)
            for j in range(d):
                vertices[2 * j, :] = A[:, j]
                vertices[2 * j + 1, :] = -A[:, j]
            return vertices
        elif norm_ord == 2:
            # For L2, the ball is continuous; return column directions
            # normalized to unit L2 norm
            vertices = []
            for j in range(d):
                col = A[:, j]
                norm = np.linalg.norm(col)
                if norm > 1e-15:
                    vertices.append(col / norm)
                    vertices.append(-col / norm)
            return np.array(vertices, dtype=np.float64) if vertices else np.empty((0, m))
        else:
            # General case: use columns and their negations
            vertices = np.zeros((2 * d, m), dtype=np.float64)
            for j in range(d):
                vertices[2 * j, :] = A[:, j]
                vertices[2 * j + 1, :] = -A[:, j]
            return vertices

    def detect_structure(
        self,
        A: npt.NDArray[np.float64],
        *,
        tol: float = 1e-10,
    ) -> Dict[str, bool]:
        """Detect structural properties of the workload matrix.

        Checks for:
        - **Identity**: A = I
        - **Toeplitz**: Constant diagonals
        - **Circulant**: Circulant structure (Toeplitz + wrap-around)
        - **Lower triangular**: All entries above diagonal are zero
        - **Upper triangular**: All entries below diagonal are zero
        - **Binary**: All entries are 0 or 1
        - **Non-negative**: All entries ≥ 0
        - **Block diagonal**: Block-diagonal structure

        Args:
            A: Workload matrix.
            tol: Tolerance for floating-point comparisons.

        Returns:
            Dictionary mapping structure names to booleans.

        Examples:
            >>> A = WorkloadGenerator.prefix_sums(4)
            >>> s = WorkloadAnalyzer().detect_structure(A)
            >>> s["lower_triangular"]
            True
            >>> s["binary"]
            True
        """
        A = np.asarray(A, dtype=np.float64)
        m, d = A.shape

        result: Dict[str, bool] = {}

        # Identity
        if m == d:
            result["identity"] = bool(np.max(np.abs(A - np.eye(d))) < tol)
        else:
            result["identity"] = False

        # Toeplitz
        result["toeplitz"] = self._is_toeplitz(A, tol)

        # Circulant
        result["circulant"] = self._is_circulant(A, tol)

        # Lower triangular
        if m == d:
            upper = np.triu(A, k=1)
            result["lower_triangular"] = bool(np.max(np.abs(upper)) < tol)
        else:
            result["lower_triangular"] = False

        # Upper triangular
        if m == d:
            lower = np.tril(A, k=-1)
            result["upper_triangular"] = bool(np.max(np.abs(lower)) < tol)
        else:
            result["upper_triangular"] = False

        # Binary
        result["binary"] = bool(
            np.all(np.abs(A) < tol) or
            np.all((np.abs(A) < tol) | (np.abs(A - 1.0) < tol))
        )

        # Non-negative
        result["non_negative"] = bool(np.all(A >= -tol))

        # Symmetric
        if m == d:
            result["symmetric"] = bool(np.max(np.abs(A - A.T)) < tol)
        else:
            result["symmetric"] = False

        # Block diagonal (simplified check)
        result["block_diagonal"] = self._is_block_diagonal(A, tol)

        return result

    def optimal_strategy_bounds(
        self,
        A: npt.NDArray[np.float64],
        eps: float,
    ) -> Dict[str, float]:
        """Compute bounds on the optimal mechanism error for workload A.

        Provides lower and upper bounds on the total squared error
        achievable by any ε-differentially private mechanism answering
        workload A.

        Bounds implemented:
        1. **SVD lower bound**: Based on the singular values of A.
        2. **Identity upper bound**: Error when using identity strategy.
        3. **Sensitivity bound**: Based on L2 sensitivity.

        Args:
            A: Workload matrix of shape ``(m, d)``.
            eps: Privacy parameter ε.

        Returns:
            Dictionary with bound names and values:

            - ``"svd_lower"``: Lower bound from SVD analysis.
            - ``"identity_upper"``: Upper bound from identity strategy.
            - ``"sensitivity_upper"``: Upper bound from sensitivity.
            - ``"epsilon"``: The privacy parameter used.

        Raises:
            ConfigurationError: If ε ≤ 0.

        Examples:
            >>> bounds = WorkloadAnalyzer().optimal_strategy_bounds(np.eye(3), 1.0)
            >>> bounds["identity_upper"]  # doctest: +ELLIPSIS
            6.0...
        """
        if eps <= 0:
            raise ConfigurationError(
                f"eps must be > 0, got {eps}",
                parameter="eps",
                value=eps,
                constraint="eps > 0",
            )

        A = np.asarray(A, dtype=np.float64)
        m, d = A.shape

        # SVD of A
        svd_vals = np.linalg.svd(A, compute_uv=False)

        # SVD lower bound: sum of squared singular values * 2/eps^2
        # This is the error of the best rank-d mechanism
        # Tighter: trace(A A^T) * 2/eps^2 / d (distributed evenly)
        trace_AAt = float(np.sum(svd_vals ** 2))
        svd_lower = trace_AAt * 2.0 / (eps ** 2 * d) if d > 0 else 0.0

        # Identity strategy upper bound
        # Error = (2/eps^2) * trace(A (I)^-1 A^T) = (2/eps^2) * trace(A A^T)
        identity_upper = trace_AAt * 2.0 / (eps ** 2)

        # Sensitivity-based bound
        col_l2 = np.sqrt(np.sum(A ** 2, axis=0))
        max_col_l2 = float(np.max(col_l2)) if d > 0 else 0.0
        sensitivity_upper = m * (max_col_l2 ** 2) * 2.0 / (eps ** 2)

        return {
            "svd_lower": svd_lower,
            "identity_upper": identity_upper,
            "sensitivity_upper": sensitivity_upper,
            "epsilon": eps,
            "trace_AAt": trace_AAt,
            "max_singular_value": float(svd_vals[0]) if len(svd_vals) > 0 else 0.0,
        }

    def workload_factorization(
        self,
        A: npt.NDArray[np.float64],
        *,
        rank: Optional[int] = None,
        tol: float = 1e-10,
    ) -> Dict[str, Any]:
        """Factor workload matrix for strategy matrix construction.

        Computes useful factorizations of A:
        1. **SVD factorization**: A = U Σ V^T
        2. **Low-rank approximation**: Best rank-r approximation
        3. **Non-negative factorization hint**: Suggests whether NMF
           might be beneficial.

        Args:
            A: Workload matrix of shape ``(m, d)``.
            rank: Target rank for low-rank approximation.  If ``None``,
                uses the numerical rank.
            tol: Tolerance for rank determination.

        Returns:
            Dictionary with factorization components:

            - ``"U"``: Left singular vectors, shape ``(m, r)``.
            - ``"S"``: Singular values, shape ``(r,)``.
            - ``"Vt"``: Right singular vectors, shape ``(r, d)``.
            - ``"rank"``: Numerical rank.
            - ``"low_rank_A"``: Best rank-r approximation.
            - ``"approx_error"``: Frobenius norm of approximation error.
            - ``"is_non_negative"``: Whether A has all non-negative entries.
        """
        A = np.asarray(A, dtype=np.float64)
        m, d = A.shape

        U, S, Vt = np.linalg.svd(A, full_matrices=False)

        numerical_rank = int(np.sum(S > tol))
        r = rank if rank is not None else numerical_rank
        r = max(1, min(r, len(S)))

        # Low-rank approximation
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]
        low_rank_A = U_r @ np.diag(S_r) @ Vt_r

        approx_error = float(np.linalg.norm(A - low_rank_A, "fro"))

        return {
            "U": U_r,
            "S": S_r,
            "Vt": Vt_r,
            "rank": numerical_rank,
            "target_rank": r,
            "low_rank_A": low_rank_A,
            "approx_error": approx_error,
            "is_non_negative": bool(np.all(A >= -tol)),
        }

    # -------------------------------------------------------------------
    # Private structure-detection helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _is_toeplitz(A: npt.NDArray[np.float64], tol: float) -> bool:
        """Check if A is a Toeplitz matrix (constant diagonals).

        A Toeplitz matrix has A[i,j] = A[i+1,j+1] for all valid i,j.
        """
        m, d = A.shape
        if m != d:
            return False
        for i in range(m - 1):
            for j in range(d - 1):
                if abs(A[i, j] - A[i + 1, j + 1]) > tol:
                    return False
        return True

    @staticmethod
    def _is_circulant(A: npt.NDArray[np.float64], tol: float) -> bool:
        """Check if A is a circulant matrix.

        A circulant matrix is Toeplitz with the additional property that
        each row is a cyclic right-shift of the previous row.
        """
        m, d = A.shape
        if m != d or m == 0:
            return False
        first_row = A[0]
        for i in range(1, m):
            expected = np.roll(first_row, i)
            if np.max(np.abs(A[i] - expected)) > tol:
                return False
        return True

    @staticmethod
    def _is_block_diagonal(A: npt.NDArray[np.float64], tol: float) -> bool:
        """Check if A has block-diagonal structure.

        Simplified check: looks for 2×2 or larger blocks along the
        diagonal with zeros elsewhere.  Returns True if the off-diagonal
        blocks are all zeros.
        """
        m, d = A.shape
        if m <= 1 or d <= 1:
            return True

        # Check if there's a natural block boundary
        # by looking for zero columns/rows in the interaction pattern
        interaction = np.abs(A) > tol

        # Find connected components of columns
        col_groups: List[Set[int]] = []
        assigned = set()

        for j in range(d):
            if j in assigned:
                continue
            group = {j}
            queue = [j]
            while queue:
                col = queue.pop()
                # Find all rows that touch this column
                rows_touching = np.where(interaction[:, col])[0]
                # Find all columns touched by those rows
                for r in rows_touching:
                    cols_touching = np.where(interaction[r, :])[0]
                    for c in cols_touching:
                        if c not in group:
                            group.add(c)
                            queue.append(c)
            assigned |= group
            col_groups.append(group)

        return len(col_groups) > 1


# ---------------------------------------------------------------------------
# Composition workloads
# ---------------------------------------------------------------------------


def compose_workloads(*workloads: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Vertically stack multiple workload matrices.

    All workloads must have the same number of columns (domain dimension).
    The result is a single workload that answers all queries from all
    input workloads.

    Args:
        *workloads: Workload matrices, each of shape ``(m_i, d)``.

    Returns:
        Stacked workload of shape ``(sum(m_i), d)``.

    Raises:
        ConfigurationError: If workloads have incompatible dimensions.

    Examples:
        >>> A1 = np.eye(3)
        >>> A2 = np.ones((2, 3))
        >>> C = compose_workloads(A1, A2)
        >>> C.shape
        (5, 3)
    """
    if len(workloads) == 0:
        raise ConfigurationError(
            "At least one workload is required",
            parameter="workloads",
        )

    validated = []
    d = None
    for i, W in enumerate(workloads):
        W = np.asarray(W, dtype=np.float64)
        if W.ndim == 1:
            W = W.reshape(1, -1)
        if W.ndim != 2:
            raise ConfigurationError(
                f"Workload {i} must be 2-D, got shape {W.shape}",
                parameter=f"workloads[{i}]",
            )
        if d is None:
            d = W.shape[1]
        elif W.shape[1] != d:
            raise ConfigurationError(
                f"Workload {i} has {W.shape[1]} columns, expected {d}",
                parameter=f"workloads[{i}]",
                constraint=f"n_cols == {d}",
            )
        validated.append(W)

    return np.vstack(validated)


def weighted_workload(
    A: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply per-query weights to a workload matrix.

    Returns ``diag(weights) @ A``, which scales each query (row) by
    its corresponding weight.  Queries with higher weight will receive
    more accuracy in the optimal mechanism.

    Args:
        A: Workload matrix of shape ``(m, d)``.
        weights: Weight vector of shape ``(m,)``.  All weights must be
            non-negative.

    Returns:
        Weighted workload of shape ``(m, d)``.

    Raises:
        ConfigurationError: If dimensions don't match or weights are
            negative.

    Examples:
        >>> A = np.eye(3)
        >>> w = np.array([1.0, 2.0, 3.0])
        >>> weighted_workload(A, w)
        array([[1., 0., 0.],
               [0., 2., 0.],
               [0., 0., 3.]])
    """
    A = np.asarray(A, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if A.ndim != 2:
        raise ConfigurationError(
            f"Workload must be 2-D, got shape {A.shape}",
            parameter="A",
        )
    if weights.ndim != 1:
        raise ConfigurationError(
            f"Weights must be 1-D, got shape {weights.shape}",
            parameter="weights",
        )
    if len(weights) != A.shape[0]:
        raise ConfigurationError(
            f"weights length {len(weights)} != number of queries {A.shape[0]}",
            parameter="weights",
            constraint=f"len(weights) == {A.shape[0]}",
        )
    if np.any(weights < 0):
        raise ConfigurationError(
            "All weights must be non-negative",
            parameter="weights",
            constraint="weights >= 0",
        )

    return weights[:, np.newaxis] * A


def subsampled_workload(
    A: npt.NDArray[np.float64],
    rate: float,
    *,
    seed: Optional[int] = None,
) -> npt.NDArray[np.float64]:
    """Create a Poisson-subsampled workload.

    Each query (row) is independently included with probability ``rate``.
    This models the privacy amplification by subsampling setting.

    Args:
        A: Workload matrix of shape ``(m, d)``.
        rate: Inclusion probability in (0, 1].
        seed: Random seed for reproducibility.

    Returns:
        Subsampled workload with approximately ``m * rate`` rows.

    Raises:
        ConfigurationError: If rate is not in (0, 1].

    Examples:
        >>> A = np.eye(10)
        >>> S = subsampled_workload(A, 0.5, seed=42)
        >>> S.shape[1]
        10
    """
    if not (0.0 < rate <= 1.0):
        raise ConfigurationError(
            f"rate must be in (0, 1], got {rate}",
            parameter="rate",
            value=rate,
            constraint="0 < rate <= 1",
        )

    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ConfigurationError(
            f"Workload must be 2-D, got shape {A.shape}",
            parameter="A",
        )

    rng = np.random.default_rng(seed)
    mask = rng.random(A.shape[0]) < rate
    selected = A[mask]

    if selected.shape[0] == 0:
        # Ensure at least one row
        idx = rng.integers(0, A.shape[0])
        selected = A[idx : idx + 1]

    return selected * rate  # Scale by rate for unbiased estimation


# ---------------------------------------------------------------------------
# Standard benchmark workloads (from DP-Forge evaluation plan)
# ---------------------------------------------------------------------------


def T1_counting(n: int = 5) -> npt.NDArray[np.float64]:
    """Tier 1 benchmark: Single counting query.

    The simplest possible workload: a single query that counts one
    element.  This is a 1 × n row vector with a single 1.

    Shape: ``(1, n)``
    Sensitivity: GS_1 = 1

    Args:
        n: Domain size.

    Returns:
        Counting query workload of shape ``(1, n)``.

    Examples:
        >>> T1_counting(5)
        array([[1., 0., 0., 0., 0.]])
    """
    if n < 1:
        raise ConfigurationError(
            f"n must be >= 1, got {n}", parameter="n", value=n, constraint="n >= 1"
        )
    A = np.zeros((1, n), dtype=np.float64)
    A[0, 0] = 1.0
    return A


def T1_histogram_small(d: int = 5) -> npt.NDArray[np.float64]:
    """Tier 1 benchmark: Small histogram workload (d=5).

    Identity workload over 5 bins.

    Shape: ``(5, 5)``
    Sensitivity: GS_1 = 1

    Args:
        d: Number of bins (default: 5).

    Returns:
        Identity workload.

    Examples:
        >>> T1_histogram_small().shape
        (5, 5)
    """
    return WorkloadGenerator.identity(d)


def T1_histogram_medium(d: int = 10) -> npt.NDArray[np.float64]:
    """Tier 1 benchmark: Medium histogram workload (d=10).

    Identity workload over 10 bins.

    Shape: ``(10, 10)``
    Sensitivity: GS_1 = 1

    Args:
        d: Number of bins (default: 10).

    Returns:
        Identity workload.

    Examples:
        >>> T1_histogram_medium().shape
        (10, 10)
    """
    return WorkloadGenerator.identity(d)


def T1_prefix(d: int = 10) -> npt.NDArray[np.float64]:
    """Tier 1 benchmark: Prefix sum queries (d=10).

    Lower triangular workload over 10 elements.

    Shape: ``(10, 10)``
    Sensitivity: GS_1 = 10, GS_2 = √10

    Args:
        d: Domain size (default: 10).

    Returns:
        Prefix sum workload.

    Examples:
        >>> T1_prefix().shape
        (10, 10)
    """
    return WorkloadGenerator.prefix_sums(d)


def T2_histogram_large(d: int = 20) -> npt.NDArray[np.float64]:
    """Tier 2 benchmark: Large histogram workload (d=20).

    Identity workload over 20 bins.

    Shape: ``(20, 20)``
    Sensitivity: GS_1 = 1

    Args:
        d: Number of bins (default: 20).

    Returns:
        Identity workload.

    Examples:
        >>> T2_histogram_large().shape
        (20, 20)
    """
    return WorkloadGenerator.identity(d)


def T2_all_range(d: int = 10) -> npt.NDArray[np.float64]:
    """Tier 2 benchmark: All range queries (d=10).

    All contiguous sub-array sum queries.

    Shape: ``(55, 10)``
    Sensitivity: depends on element position

    Args:
        d: Domain size (default: 10).

    Returns:
        All-range workload.

    Examples:
        >>> T2_all_range().shape
        (55, 10)
    """
    return WorkloadGenerator.all_range(d)


def T2_2d_histogram(n1: int = 5, n2: int = 5) -> npt.NDArray[np.float64]:
    """Tier 2 benchmark: 2D histogram (5×5).

    Identity workload over a 5×5 grid (25 bins).

    Shape: ``(25, 25)``
    Sensitivity: GS_1 = 1

    Args:
        n1: First dimension size (default: 5).
        n2: Second dimension size (default: 5).

    Returns:
        2D histogram workload.

    Examples:
        >>> T2_2d_histogram().shape
        (25, 25)
    """
    return WorkloadGenerator.histogram_2d(n1, n2)


def T3_large_histogram(d: int = 50) -> npt.NDArray[np.float64]:
    """Tier 3 benchmark: Large histogram workload (d=50).

    Identity workload over 50 bins.

    Shape: ``(50, 50)``
    Sensitivity: GS_1 = 1

    Args:
        d: Number of bins (default: 50).

    Returns:
        Identity workload.

    Examples:
        >>> T3_large_histogram().shape
        (50, 50)
    """
    return WorkloadGenerator.identity(d)


def T3_marginals(d: int = 10, k: int = 3) -> npt.NDArray[np.float64]:
    """Tier 3 benchmark: k-way marginals (d=10, k=3).

    All 3-way marginal queries over 10 binary attributes.
    C(10,3) = 120 marginals, each with 2^3 = 8 cells, domain size 2^10 = 1024.

    Shape: ``(960, 1024)``
    Sensitivity: GS_1 = 1 (add/remove)

    Args:
        d: Number of attributes (default: 10).
        k: Marginal order (default: 3).

    Returns:
        Marginal workload.

    Examples:
        >>> T3_marginals().shape
        (960, 1024)
    """
    return WorkloadGenerator.marginals(d, k)


# ---------------------------------------------------------------------------
# Conversion to WorkloadSpec
# ---------------------------------------------------------------------------


def to_workload_spec(
    A: npt.NDArray[np.float64],
    *,
    query_type: QueryType = QueryType.LINEAR_WORKLOAD,
    structural_hint: Optional[str] = None,
    auto_detect: bool = True,
) -> WorkloadSpec:
    """Convert a workload matrix to a :class:`WorkloadSpec`.

    Optionally auto-detects structural properties and sets the
    ``structural_hint`` field accordingly.

    Args:
        A: Workload matrix of shape ``(m, d)``.
        query_type: Query type classification.
        structural_hint: Manual structural hint.  If ``None`` and
            ``auto_detect`` is True, attempts to detect structure.
        auto_detect: Whether to auto-detect structure.

    Returns:
        A :class:`WorkloadSpec` wrapping the workload.

    Examples:
        >>> A = WorkloadGenerator.prefix_sums(5)
        >>> spec = to_workload_spec(A)
        >>> spec.structural_hint
        'toeplitz'
    """
    A = np.asarray(A, dtype=np.float64)

    if auto_detect and structural_hint is None:
        analyzer = WorkloadAnalyzer()
        structure = analyzer.detect_structure(A)
        if structure.get("identity"):
            structural_hint = "identity"
        elif structure.get("circulant"):
            structural_hint = "circulant"
        elif structure.get("toeplitz"):
            structural_hint = "toeplitz"
        elif structure.get("block_diagonal"):
            structural_hint = "block_diagonal"

    return WorkloadSpec(
        matrix=A,
        query_type=query_type,
        structural_hint=structural_hint,
    )


# ---------------------------------------------------------------------------
# Workload summary
# ---------------------------------------------------------------------------


def summarize_workload(A: npt.NDArray[np.float64]) -> str:
    """Generate a human-readable summary of a workload matrix.

    Args:
        A: Workload matrix.

    Returns:
        Multi-line string summary.

    Examples:
        >>> print(summarize_workload(np.eye(3)))  # doctest: +NORMALIZE_WHITESPACE
        Workload Summary
        ...
    """
    analyzer = WorkloadAnalyzer()
    props = analyzer.analyze_workload(A)

    lines = [
        "Workload Summary",
        "=" * 40,
        f"  Shape:            {props.shape[0]} queries × {props.shape[1]} domain",
        f"  Rank:             {props.rank}",
        f"  Condition number: {props.condition_number:.2e}",
        f"  Sparsity:         {props.sparsity:.1%}",
        f"  Full rank:        {props.is_full_rank}",
        f"  Square:           {props.is_square}",
        f"  Orthogonal:       {props.is_orthogonal}",
        "",
        "Sensitivity:",
        f"  L1:   {props.sensitivity_l1:.4f}",
        f"  L2:   {props.sensitivity_l2:.4f}",
        f"  Linf: {props.sensitivity_linf:.4f}",
        "",
        "Structure:",
    ]

    for name, detected in props.structure.items():
        marker = "✓" if detected else "✗"
        lines.append(f"  {marker} {name}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Generator
    "WorkloadGenerator",
    # Analyzer
    "WorkloadAnalyzer",
    "WorkloadProperties",
    # Composition
    "compose_workloads",
    "weighted_workload",
    "subsampled_workload",
    # Benchmark workloads
    "T1_counting",
    "T1_histogram_small",
    "T1_histogram_medium",
    "T1_prefix",
    "T2_histogram_large",
    "T2_all_range",
    "T2_2d_histogram",
    "T3_large_histogram",
    "T3_marginals",
    # Utilities
    "to_workload_spec",
    "summarize_workload",
]

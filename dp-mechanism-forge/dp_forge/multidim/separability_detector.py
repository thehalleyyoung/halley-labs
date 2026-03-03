"""
Automatic separability detection for multi-dimensional query matrices.

Given a query matrix A (m×d), this module detects whether it decomposes as a
Kronecker product A = A₁ ⊗ A₂ ⊗ ... ⊗ A_r, which enables coordinate-wise
mechanism synthesis instead of the exponentially expensive direct LP.

Detection Algorithms:
    - **SVD-based**: Rearrange A into a matrix whose rank-1 structure reveals
      Kronecker factors (Van Loan & Pitsianis 1993).
    - **Block-diagonal**: Detect block-diagonal structure via permutation of
      rows/columns, enabling independent sub-problem synthesis.
    - **Partial separability**: When full Kronecker decomposition fails,
      identify the largest separable sub-blocks for hybrid treatment.

Classes:
    SeparabilityResult  — structured output of separability analysis
    SeparabilityDetector — main detector with SVD and block-diagonal methods
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)


class SeparabilityType(Enum):
    """Classification of separability structure."""

    FULL_KRONECKER = auto()
    BLOCK_DIAGONAL = auto()
    PARTIAL = auto()
    NON_SEPARABLE = auto()

    def __repr__(self) -> str:
        return f"SeparabilityType.{self.name}"


@dataclass
class KroneckerFactor:
    """A single factor in a Kronecker decomposition.

    Attributes:
        matrix: The factor matrix A_i.
        coordinate_indices: Which coordinates of the original d-dim space
            this factor corresponds to.
        rank: Numerical rank of this factor.
    """

    matrix: npt.NDArray[np.float64]
    coordinate_indices: List[int]
    rank: int

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        if self.matrix.ndim != 2:
            raise ValueError(
                f"Factor matrix must be 2-D, got shape {self.matrix.shape}"
            )
        if not self.coordinate_indices:
            raise ValueError("coordinate_indices must be non-empty")

    @property
    def m(self) -> int:
        """Number of rows (queries) in this factor."""
        return self.matrix.shape[0]

    @property
    def d(self) -> int:
        """Number of columns (coordinates) in this factor."""
        return self.matrix.shape[1]

    def __repr__(self) -> str:
        return (
            f"KroneckerFactor(shape={self.matrix.shape}, "
            f"coords={self.coordinate_indices})"
        )


@dataclass
class SeparabilityResult:
    """Structured output of separability analysis.

    Attributes:
        sep_type: Classification of the detected structure.
        factors: Kronecker factors if separable (empty if non-separable).
        residual_norm: Frobenius norm of the residual after decomposition.
        relative_error: Residual norm / original matrix norm.
        blocks: For block-diagonal detection, the identified blocks as
            lists of column indices.
        non_separable_submatrix: Submatrix that could not be decomposed,
            if partial separability was detected.
        metadata: Additional analysis metadata.
    """

    sep_type: SeparabilityType
    factors: List[KroneckerFactor] = field(default_factory=list)
    residual_norm: float = 0.0
    relative_error: float = 0.0
    blocks: List[List[int]] = field(default_factory=list)
    non_separable_submatrix: Optional[npt.NDArray[np.float64]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_separable(self) -> bool:
        """Whether the matrix admits any form of separable decomposition."""
        return self.sep_type in (
            SeparabilityType.FULL_KRONECKER,
            SeparabilityType.BLOCK_DIAGONAL,
            SeparabilityType.PARTIAL,
        )

    @property
    def n_factors(self) -> int:
        """Number of Kronecker factors."""
        return len(self.factors)

    @property
    def n_blocks(self) -> int:
        """Number of independent blocks."""
        return len(self.blocks)

    def __repr__(self) -> str:
        if self.factors:
            shapes = [f.matrix.shape for f in self.factors]
            return (
                f"SeparabilityResult(type={self.sep_type.name}, "
                f"factors={shapes}, rel_err={self.relative_error:.2e})"
            )
        if self.blocks:
            return (
                f"SeparabilityResult(type={self.sep_type.name}, "
                f"blocks={self.n_blocks})"
            )
        return f"SeparabilityResult(type={self.sep_type.name})"


class SeparabilityDetector:
    """Detects Kronecker and block-diagonal separability of query matrices.

    The detector tries increasingly relaxed decompositions:
    1. Exact Kronecker product via SVD rearrangement.
    2. Block-diagonal structure via column interaction graph.
    3. Partial separability: largest separable sub-block.

    Args:
        tol: Relative tolerance for rank decisions and residual checks.
        max_factors: Maximum number of Kronecker factors to attempt.
    """

    def __init__(
        self,
        tol: float = 1e-10,
        max_factors: int = 16,
    ) -> None:
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        if max_factors < 2:
            raise ValueError(f"max_factors must be >= 2, got {max_factors}")
        self._tol = tol
        self._max_factors = max_factors

    def detect(self, A: npt.NDArray[np.float64]) -> SeparabilityResult:
        """Run full separability analysis on query matrix A.

        Tries Kronecker decomposition first, then block-diagonal, then
        partial separability. Returns the strongest structure found.

        Args:
            A: Query matrix of shape (m, d).

        Returns:
            SeparabilityResult describing the detected structure.
        """
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError(f"A must be 2-D, got shape {A.shape}")
        m, d = A.shape
        if d < 2:
            return SeparabilityResult(
                sep_type=SeparabilityType.NON_SEPARABLE,
                metadata={"reason": "d < 2, trivially non-separable"},
            )
        A_norm = np.linalg.norm(A, "fro")
        if A_norm < self._tol:
            factor = KroneckerFactor(
                matrix=A, coordinate_indices=list(range(d)), rank=0
            )
            return SeparabilityResult(
                sep_type=SeparabilityType.FULL_KRONECKER,
                factors=[factor],
                residual_norm=0.0,
                relative_error=0.0,
            )
        # 1) Try block-diagonal detection (cheapest)
        block_result = self._detect_block_diagonal(A)
        if block_result.sep_type == SeparabilityType.BLOCK_DIAGONAL:
            kron_result = self._try_kronecker_on_blocks(A, block_result.blocks)
            if kron_result is not None:
                return kron_result
            return block_result
        # 2) Try two-factor Kronecker decomposition via SVD
        kron_result = self._detect_kronecker_two_factor(A)
        if kron_result.sep_type == SeparabilityType.FULL_KRONECKER:
            return kron_result
        # 3) Try partial separability
        partial = self._detect_partial_separability(A, block_result)
        if partial.sep_type == SeparabilityType.PARTIAL:
            return partial
        return SeparabilityResult(
            sep_type=SeparabilityType.NON_SEPARABLE,
            residual_norm=A_norm,
            relative_error=1.0,
            metadata={"reason": "no separable structure detected"},
        )

    def _detect_block_diagonal(
        self, A: npt.NDArray[np.float64]
    ) -> SeparabilityResult:
        """Detect block-diagonal structure via column interaction graph.

        Two columns j₁, j₂ interact if there exists a row i such that
        A[i, j₁] ≠ 0 and A[i, j₂] ≠ 0. Connected components of this
        graph yield independent blocks.

        Args:
            A: Query matrix (m, d).

        Returns:
            SeparabilityResult with blocks if block-diagonal, else NON_SEPARABLE.
        """
        m, d = A.shape
        nonzero_mask = np.abs(A) > self._tol
        # Build adjacency graph over columns
        adj_entries_row: List[int] = []
        adj_entries_col: List[int] = []
        for i in range(m):
            nz_cols = np.where(nonzero_mask[i])[0]
            if len(nz_cols) < 2:
                continue
            for idx in range(len(nz_cols)):
                for jdx in range(idx + 1, len(nz_cols)):
                    adj_entries_row.append(nz_cols[idx])
                    adj_entries_col.append(nz_cols[jdx])
                    adj_entries_row.append(nz_cols[jdx])
                    adj_entries_col.append(nz_cols[idx])
        if not adj_entries_row:
            # Every column is independent
            blocks = [[j] for j in range(d)]
        else:
            data = np.ones(len(adj_entries_row), dtype=np.float64)
            adj = csr_matrix(
                (data, (adj_entries_row, adj_entries_col)), shape=(d, d)
            )
            n_components, labels = connected_components(adj, directed=False)
            if n_components <= 1:
                return SeparabilityResult(
                    sep_type=SeparabilityType.NON_SEPARABLE,
                    metadata={"reason": "single connected component"},
                )
            blocks = []
            for c in range(n_components):
                block_cols = sorted(np.where(labels == c)[0].tolist())
                blocks.append(block_cols)
        if len(blocks) < 2:
            return SeparabilityResult(
                sep_type=SeparabilityType.NON_SEPARABLE,
                metadata={"reason": "single block"},
            )
        # Build factors from blocks
        factors = []
        for block_cols in blocks:
            sub = A[:, block_cols]
            # Remove zero rows
            row_norms = np.linalg.norm(sub, axis=1)
            active_rows = row_norms > self._tol
            sub_active = sub[active_rows]
            if sub_active.size == 0:
                sub_active = np.zeros((1, len(block_cols)), dtype=np.float64)
            rank = int(np.linalg.matrix_rank(sub_active, tol=self._tol))
            factors.append(
                KroneckerFactor(
                    matrix=sub_active,
                    coordinate_indices=block_cols,
                    rank=rank,
                )
            )
        return SeparabilityResult(
            sep_type=SeparabilityType.BLOCK_DIAGONAL,
            factors=factors,
            blocks=blocks,
            residual_norm=0.0,
            relative_error=0.0,
        )

    def _detect_kronecker_two_factor(
        self, A: npt.NDArray[np.float64]
    ) -> SeparabilityResult:
        """Attempt two-factor Kronecker decomposition A ≈ A₁ ⊗ A₂ via SVD.

        Uses the Van Loan & Pitsianis nearest Kronecker product algorithm:
        reshape A into a matrix R, compute its SVD, and check if rank-1
        approximation has sufficiently small residual.

        Args:
            A: Query matrix (m, d).

        Returns:
            SeparabilityResult with FULL_KRONECKER if successful.
        """
        m, d = A.shape
        A_norm = np.linalg.norm(A, "fro")
        # Try all valid factorizations of d = d₁ × d₂
        best_result: Optional[SeparabilityResult] = None
        best_error = float("inf")
        for d1 in range(2, d):
            if d % d1 != 0:
                continue
            d2 = d // d1
            # Try all valid factorizations of m = m₁ × m₂
            for m1 in range(1, m + 1):
                if m % m1 != 0:
                    continue
                m2 = m // m1
                result = self._try_two_factor_split(A, m, d, m1, m2, d1, d2, A_norm)
                if result is not None and result.relative_error < best_error:
                    best_error = result.relative_error
                    best_result = result
        if best_result is not None and best_error < self._tol:
            return best_result
        return SeparabilityResult(
            sep_type=SeparabilityType.NON_SEPARABLE,
            residual_norm=A_norm,
            relative_error=1.0,
        )

    def _try_two_factor_split(
        self,
        A: npt.NDArray[np.float64],
        m: int,
        d: int,
        m1: int,
        m2: int,
        d1: int,
        d2: int,
        A_norm: float,
    ) -> Optional[SeparabilityResult]:
        """Try a specific (m₁×d₁) ⊗ (m₂×d₂) factorization.

        Rearranges A into R of shape (m₁·d₁, m₂·d₂) and checks if R
        is approximately rank-1.

        Returns:
            SeparabilityResult if rank-1 approximation is good, else None.
        """
        try:
            R = np.zeros((m1 * d1, m2 * d2), dtype=np.float64)
            for i1 in range(m1):
                for j1 in range(d1):
                    for i2 in range(m2):
                        for j2 in range(d2):
                            row = i1 * m2 + i2
                            col = j1 * d2 + j2
                            R[i1 * d1 + j1, i2 * d2 + j2] = A[row, col]
        except IndexError:
            return None
        # SVD of rearranged matrix
        U, s, Vt = np.linalg.svd(R, full_matrices=False)
        if len(s) < 1 or s[0] < self._tol:
            return None
        # Check rank-1 quality
        if len(s) > 1:
            residual_sq = float(np.sum(s[1:] ** 2))
        else:
            residual_sq = 0.0
        residual = math.sqrt(residual_sq)
        rel_error = residual / A_norm if A_norm > 0 else 0.0
        if rel_error > self._tol:
            return None
        # Extract factors
        A1 = (U[:, 0] * math.sqrt(s[0])).reshape(m1, d1)
        A2 = (Vt[0, :] * math.sqrt(s[0])).reshape(m2, d2)
        coords1 = list(range(d1))
        coords2 = list(range(d1, d1 + d2))
        rank1 = int(np.linalg.matrix_rank(A1, tol=self._tol))
        rank2 = int(np.linalg.matrix_rank(A2, tol=self._tol))
        return SeparabilityResult(
            sep_type=SeparabilityType.FULL_KRONECKER,
            factors=[
                KroneckerFactor(matrix=A1, coordinate_indices=coords1, rank=rank1),
                KroneckerFactor(matrix=A2, coordinate_indices=coords2, rank=rank2),
            ],
            residual_norm=residual,
            relative_error=rel_error,
            metadata={"split": (m1, d1, m2, d2)},
        )

    def _try_kronecker_on_blocks(
        self,
        A: npt.NDArray[np.float64],
        blocks: List[List[int]],
    ) -> Optional[SeparabilityResult]:
        """Check if each block in a block-diagonal structure is further Kronecker-separable.

        Args:
            A: Original query matrix.
            blocks: Column index blocks from block-diagonal detection.

        Returns:
            SeparabilityResult with refined Kronecker factors, or None.
        """
        all_factors: List[KroneckerFactor] = []
        total_residual_sq = 0.0
        A_norm = np.linalg.norm(A, "fro")
        for block_cols in blocks:
            sub = A[:, block_cols]
            row_norms = np.linalg.norm(sub, axis=1)
            active_rows = row_norms > self._tol
            sub_active = sub[active_rows]
            if sub_active.shape[1] >= 4:
                sub_result = self._detect_kronecker_two_factor(sub_active)
                if sub_result.sep_type == SeparabilityType.FULL_KRONECKER:
                    # Remap coordinate indices to original columns
                    for f in sub_result.factors:
                        remapped = [block_cols[c] for c in f.coordinate_indices]
                        all_factors.append(
                            KroneckerFactor(
                                matrix=f.matrix,
                                coordinate_indices=remapped,
                                rank=f.rank,
                            )
                        )
                    total_residual_sq += sub_result.residual_norm ** 2
                    continue
            rank = int(np.linalg.matrix_rank(sub_active, tol=self._tol))
            all_factors.append(
                KroneckerFactor(
                    matrix=sub_active,
                    coordinate_indices=block_cols,
                    rank=rank,
                )
            )
        if len(all_factors) < 2:
            return None
        total_residual = math.sqrt(total_residual_sq)
        rel_error = total_residual / A_norm if A_norm > 0 else 0.0
        return SeparabilityResult(
            sep_type=SeparabilityType.FULL_KRONECKER,
            factors=all_factors,
            residual_norm=total_residual,
            relative_error=rel_error,
            blocks=blocks,
        )

    def _detect_partial_separability(
        self,
        A: npt.NDArray[np.float64],
        block_result: SeparabilityResult,
    ) -> SeparabilityResult:
        """Find the largest separable sub-block when full decomposition fails.

        Uses greedy column clustering: group columns by interaction
        strength and identify the largest independent cluster.

        Args:
            A: Query matrix (m, d).
            block_result: Result from block-diagonal detection.

        Returns:
            SeparabilityResult with PARTIAL type if any sub-block is found.
        """
        m, d = A.shape
        A_norm = np.linalg.norm(A, "fro")
        # Compute column interaction strength matrix
        interaction = np.zeros((d, d), dtype=np.float64)
        for i in range(m):
            nz = np.where(np.abs(A[i]) > self._tol)[0]
            for idx in range(len(nz)):
                for jdx in range(idx + 1, len(nz)):
                    strength = abs(A[i, nz[idx]] * A[i, nz[jdx]])
                    interaction[nz[idx], nz[jdx]] += strength
                    interaction[nz[jdx], nz[idx]] += strength
        # Spectral clustering via Fiedler vector
        degree = np.diag(interaction.sum(axis=1))
        laplacian = degree - interaction
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        # Find spectral gap
        sorted_eigs = np.sort(eigenvalues)
        n_clusters = 2
        if d >= 3:
            gaps = np.diff(sorted_eigs[1:])
            if len(gaps) > 0:
                best_gap_idx = int(np.argmax(gaps))
                n_clusters = min(best_gap_idx + 2, d)
                n_clusters = max(n_clusters, 2)
        # Partition columns using the first few eigenvectors
        if n_clusters >= d:
            return SeparabilityResult(
                sep_type=SeparabilityType.NON_SEPARABLE,
                residual_norm=A_norm,
                relative_error=1.0,
            )
        fiedler = eigenvectors[:, 1]
        group_a = sorted(np.where(fiedler <= np.median(fiedler))[0].tolist())
        group_b = sorted(np.where(fiedler > np.median(fiedler))[0].tolist())
        if not group_a or not group_b:
            return SeparabilityResult(
                sep_type=SeparabilityType.NON_SEPARABLE,
                residual_norm=A_norm,
                relative_error=1.0,
            )
        # Check cross-interaction between groups
        cross_interaction = 0.0
        for ca in group_a:
            for cb in group_b:
                cross_interaction += interaction[ca, cb]
        total_interaction = interaction.sum() / 2.0
        if total_interaction > 0 and cross_interaction / total_interaction > 0.1:
            return SeparabilityResult(
                sep_type=SeparabilityType.NON_SEPARABLE,
                residual_norm=A_norm,
                relative_error=1.0,
                metadata={"cross_ratio": cross_interaction / total_interaction},
            )
        # Build factors for the two groups
        factors = []
        non_sep_cols: List[int] = []
        for group in [group_a, group_b]:
            sub = A[:, group]
            row_norms = np.linalg.norm(sub, axis=1)
            active = row_norms > self._tol
            sub_active = sub[active]
            if sub_active.size == 0:
                non_sep_cols.extend(group)
                continue
            rank = int(np.linalg.matrix_rank(sub_active, tol=self._tol))
            factors.append(
                KroneckerFactor(
                    matrix=sub_active,
                    coordinate_indices=group,
                    rank=rank,
                )
            )
        if len(factors) < 2:
            return SeparabilityResult(
                sep_type=SeparabilityType.NON_SEPARABLE,
                residual_norm=A_norm,
                relative_error=1.0,
            )
        non_sep_sub = A[:, non_sep_cols] if non_sep_cols else None
        return SeparabilityResult(
            sep_type=SeparabilityType.PARTIAL,
            factors=factors,
            blocks=[group_a, group_b],
            non_separable_submatrix=non_sep_sub,
            metadata={"cross_ratio": cross_interaction / max(total_interaction, 1e-30)},
        )

    def detect_identity_structure(
        self, A: npt.NDArray[np.float64]
    ) -> Optional[List[int]]:
        """Check if A is a permuted and scaled identity-like matrix.

        An identity-like structure means each row has exactly one
        non-zero entry, indicating each query depends on a single
        coordinate — trivially separable.

        Args:
            A: Query matrix (m, d).

        Returns:
            List of column indices for each row's non-zero entry,
            or None if not identity-like.
        """
        m, d = A.shape
        result = []
        for i in range(m):
            nz = np.where(np.abs(A[i]) > self._tol)[0]
            if len(nz) != 1:
                return None
            result.append(int(nz[0]))
        return result

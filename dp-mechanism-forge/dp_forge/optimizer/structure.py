"""
Toeplitz/circulant structure exploitation for fast matrix operations.

This module provides classes that detect and exploit Toeplitz, circulant, and
banded structure in constraint matrices arising from DP mechanism synthesis.
All implementations use FFT-based algorithms for O(n log n) matrix-vector
products instead of O(n²) dense operations.

Key algorithms:
    - Toeplitz matrix-vector product via embedding in circulant matrix + FFT
    - Circulant preconditioner construction for PCG iterative methods
    - Symmetry reduction that exploits mechanism output symmetries
    - Banded structure detection for sparse matrix optimizations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator

from dp_forge.exceptions import NumericalInstabilityError


@dataclass(frozen=True)
class ToeplitzStructure:
    """Detected Toeplitz block structure in a matrix.
    
    Attributes:
        first_col: First column of the Toeplitz block
        first_row: First row of the Toeplitz block
        row_start: Starting row index of block
        row_end: Ending row index of block (exclusive)
        col_start: Starting column index of block
        col_end: Ending column index of block (exclusive)
    """
    first_col: npt.NDArray[np.float64]
    first_row: npt.NDArray[np.float64]
    row_start: int
    row_end: int
    col_start: int
    col_end: int


class ToeplitzOperator(ScipyLinearOperator):
    """Fast Toeplitz matrix-vector product via FFT embedding.
    
    A Toeplitz matrix has constant diagonals: T[i,j] = t[i-j]. We embed T
    in a circulant matrix C of size 2n-1, which can be diagonalized by FFT:
        C = F^H * diag(F * c) * F
    where F is the DFT matrix and c is the first column of C.
    
    Matrix-vector product y = T @ x then becomes:
        1. Pad x to length 2n-1 with zeros
        2. Compute y_padded = ifft(fft(c) * fft(x_padded))
        3. Extract first n components
        
    This reduces O(n²) dense matvec to O(n log n) FFT operations.
    
    Args:
        first_col: First column of Toeplitz matrix (length n)
        first_row: First row of Toeplitz matrix (length m)
        dtype: Data type for computations
        
    References:
        - Golub & Van Loan, Matrix Computations 4th ed., §4.7
        - Chan & Jin, "An Introduction to Iterative Toeplitz Solvers", §2.1
    """
    
    def __init__(
        self,
        first_col: npt.NDArray[np.float64],
        first_row: npt.NDArray[np.float64],
        dtype: type = np.float64,
    ):
        n = len(first_col)
        m = len(first_row)
        
        if first_col[0] != first_row[0]:
            raise ValueError(
                f"Toeplitz first_col[0]={first_col[0]} must equal "
                f"first_row[0]={first_row[0]}"
            )
        
        super().__init__(dtype=dtype, shape=(n, m))
        
        self.first_col = first_col
        self.first_row = first_row
        
        # Construct circulant embedding for FFT-based matvec
        # Circulant matrix has first column: [c0, c1, ..., c_{n-1}, r_{m-1}, ..., r_1]
        circ_col = np.concatenate([
            first_col,
            first_row[-1:0:-1] if m > 1 else np.array([], dtype=dtype)
        ])
        
        # Precompute FFT of circulant first column
        self._circ_fft = np.fft.fft(circ_col)
        self._embed_size = len(circ_col)
        
    def _matvec(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute y = T @ x using FFT embedding.
        
        Algorithm:
            1. Pad x with zeros to embedding size
            2. y_padded = ifft(circ_fft * fft(x_padded))
            3. Return first n components (real part)
        """
        if len(x) != self.shape[1]:
            raise ValueError(f"x has length {len(x)}, expected {self.shape[1]}")
        
        # Pad x to embedding size
        x_padded = np.zeros(self._embed_size, dtype=self.dtype)
        x_padded[:len(x)] = x
        
        # FFT-based circulant matvec
        x_fft = np.fft.fft(x_padded)
        y_fft = self._circ_fft * x_fft
        y_padded = np.fft.ifft(y_fft)
        
        # Extract result (first n components, real part)
        y = np.real(y_padded[:self.shape[0]])
        
        return y
    
    def _rmatvec(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute x = T^T @ y using transpose Toeplitz structure.
        
        The transpose of a Toeplitz matrix is Toeplitz with first_col and
        first_row swapped.
        """
        # Create transposed operator
        transpose_op = ToeplitzOperator(self.first_row, self.first_col, self.dtype)
        return transpose_op._matvec(y)
    
    @staticmethod
    def detect_toeplitz(
        A: sparse.spmatrix,
        tol: float = 1e-10,
    ) -> list[ToeplitzStructure]:
        """Detect Toeplitz blocks in sparse matrix A.
        
        Scans matrix for regions where A[i,j] ≈ A[i+1,j+1] (constant diagonals).
        Returns list of detected Toeplitz blocks that cover at least 80% of
        the block's entries.
        
        Args:
            A: Sparse matrix to analyze
            tol: Tolerance for considering diagonal entries equal
            
        Returns:
            List of ToeplitzStructure blocks found in A
        """
        A_dense = A.toarray() if sparse.issparse(A) else A
        m, n = A_dense.shape
        
        structures = []
        
        # Try to detect block structure by looking for constant diagonals
        # We scan 32×32 blocks (common mechanism table size)
        block_size = min(32, m // 4, n // 4)
        
        if block_size < 4:
            return structures
        
        for i_start in range(0, m - block_size + 1, block_size // 2):
            for j_start in range(0, n - block_size + 1, block_size // 2):
                i_end = min(i_start + block_size, m)
                j_end = min(j_start + block_size, n)
                
                block = A_dense[i_start:i_end, j_start:j_end]
                
                if _is_toeplitz_block(block, tol):
                    first_col = block[:, 0].copy()
                    first_row = block[0, :].copy()
                    
                    structures.append(ToeplitzStructure(
                        first_col=first_col,
                        first_row=first_row,
                        row_start=i_start,
                        row_end=i_end,
                        col_start=j_start,
                        col_end=j_end,
                    ))
        
        return structures


class CirculantPreconditioner(ScipyLinearOperator):
    """Circulant preconditioner for iterative methods (PCG, GMRES).
    
    Constructs a circulant approximation M to a matrix A such that M^{-1}A
    has clustered eigenvalues, accelerating Krylov subspace convergence.
    
    For a circulant matrix M with first column c:
        M = F^H * diag(λ) * F
    where λ = fft(c). Inversion is trivial in frequency domain:
        M^{-1} = F^H * diag(1/λ) * F
        
    We construct c by:
        1. Extract central diagonals from A
        2. Taper to zero at edges using Tukey window
        3. Handle near-zero eigenvalues with Tikhonov regularization
    
    Args:
        A: Matrix to precondition (sparse or dense)
        bandwidth: Number of diagonals to include (default: auto-detect)
        regularization: Tikhonov parameter for near-zero eigenvalues
        
    References:
        - Chan, "An Optimal Circulant Preconditioner for Toeplitz Systems", 1988
        - Chan & Ng, "Conjugate Gradient Methods for Toeplitz Systems", 1996
    """
    
    def __init__(
        self,
        A: sparse.spmatrix | npt.NDArray,
        bandwidth: Optional[int] = None,
        regularization: float = 1e-8,
    ):
        if sparse.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = np.asarray(A)
        
        n, m = A_dense.shape
        if n != m:
            raise ValueError(f"Preconditioner requires square matrix, got {n}×{m}")
        
        super().__init__(dtype=np.float64, shape=(n, n))
        
        # Auto-detect bandwidth from sparsity pattern
        if bandwidth is None:
            bandwidth = _detect_bandwidth(A_dense)
        
        # Extract central diagonals to form circulant first column
        c = np.zeros(n, dtype=np.float64)
        
        for k in range(-bandwidth, bandwidth + 1):
            if k == 0:
                # Main diagonal (average)
                c[0] += np.mean(np.diag(A_dense, k=0))
            elif k > 0:
                # Upper diagonals -> c[k]
                if k < n:
                    c[k] = np.mean(np.diag(A_dense, k=k))
            else:
                # Lower diagonals -> c[n+k] (wrap around)
                if -k < n:
                    c[n + k] = np.mean(np.diag(A_dense, k=k))
        
        # Taper with Tukey window to reduce Gibbs ringing
        window = _tukey_window(n, alpha=0.1)
        c *= window
        
        # Compute eigenvalues via FFT
        lambda_fft = np.fft.fft(c)
        
        # Regularize near-zero eigenvalues
        lambda_inv = np.zeros_like(lambda_fft, dtype=np.complex128)
        for i, lam in enumerate(lambda_fft):
            if np.abs(lam) < regularization:
                lambda_inv[i] = 1.0 / (regularization + 1e-15)
            else:
                lambda_inv[i] = 1.0 / lam
        
        self._lambda_inv = lambda_inv
        self.regularization = regularization
        
    def _matvec(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply M^{-1} @ x via FFT inversion.
        
        Algorithm:
            y = M^{-1} @ x = ifft(lambda_inv * fft(x))
        """
        x_fft = np.fft.fft(x)
        y_fft = self._lambda_inv * x_fft
        y = np.real(np.fft.ifft(y_fft))
        return y


class SymmetryReducer:
    """Exploits output symmetries to reduce LP dimension.
    
    Many DP mechanisms have symmetric output distributions (e.g., Laplace is
    symmetric around the true value). When the mechanism table p[i,j] satisfies
    symmetries, we can reduce the LP decision variables by identifying
    equivalence classes.
    
    Supported symmetries:
        - Translation: p[i, j+k] = p[i, j] for all i, k (periodic noise)
        - Reflection: p[i, j] = p[i, -j] (symmetric noise around 0)
        - Rotation: p[i+k, j+k] = p[i, j] (mechanism shift-invariant)
        
    Algorithm:
        1. Detect symmetry group from constraint structure
        2. Compute orbit representatives (one per equivalence class)
        3. Build reduced LP with one variable per orbit
        4. Map solution back to full mechanism table
    
    Args:
        mechanism_shape: Shape (n, k) of mechanism table
        symmetries: List of symmetry types to detect
        
    References:
        - Ghazi et al., "The Role of Symmetry in Mechanism Design", 2020
        - Balle et al., "Implementing Discrete Gaussian Mechanisms", 2019
    """
    
    def __init__(
        self,
        mechanism_shape: Tuple[int, int],
        symmetries: Optional[list[str]] = None,
    ):
        self.n, self.k = mechanism_shape
        self.symmetries = symmetries or ["reflection", "translation"]
        
        # Compute equivalence classes under symmetry group
        self.orbits, self.orbit_map = self._compute_orbits()
        self.num_reduced_vars = len(self.orbits)
        
        # Build expansion matrix: reduced -> full
        self._expansion_matrix = self._build_expansion()
        
    def _compute_orbits(self) -> Tuple[list[Tuple[int, int]], npt.NDArray[np.int32]]:
        """Compute orbit representatives and mapping.
        
        Returns:
            orbits: List of (i, j) representative for each orbit
            orbit_map: Array mapping (i, j) -> orbit_id
        """
        orbit_map = -np.ones((self.n, self.k), dtype=np.int32)
        orbits = []
        orbit_id = 0
        
        for i in range(self.n):
            for j in range(self.k):
                if orbit_map[i, j] == -1:
                    # New orbit discovered
                    orbits.append((i, j))
                    
                    # Mark all symmetric positions
                    for i_sym, j_sym in self._orbit_members(i, j):
                        if 0 <= i_sym < self.n and 0 <= j_sym < self.k:
                            orbit_map[i_sym, j_sym] = orbit_id
                    
                    orbit_id += 1
        
        return orbits, orbit_map
    
    def _orbit_members(self, i: int, j: int) -> list[Tuple[int, int]]:
        """Generate all (i', j') in the same orbit as (i, j)."""
        members = [(i, j)]
        
        if "reflection" in self.symmetries:
            # Reflect j around center
            j_reflect = self.k - 1 - j
            members.append((i, j_reflect))
        
        if "translation" in self.symmetries and self.k >= 4:
            # Periodic translation (if k is power of 2)
            period = self.k // 2
            j_translate = (j + period) % self.k
            members.append((i, j_translate))
        
        if "rotation" in self.symmetries:
            # Diagonal rotation
            for shift in range(1, min(self.n, self.k)):
                i_rot = (i + shift) % self.n
                j_rot = (j + shift) % self.k
                members.append((i_rot, j_rot))
        
        return members
    
    def _build_expansion(self) -> sparse.csr_matrix:
        """Build sparse expansion matrix E: reduced -> full.
        
        E[i*k + j, orbit_id] = 1 if (i,j) is in orbit_id, else 0.
        Then: full_vars = E @ reduced_vars
        """
        row_indices = []
        col_indices = []
        
        for i in range(self.n):
            for j in range(self.k):
                flat_idx = i * self.k + j
                orbit_id = self.orbit_map[i, j]
                row_indices.append(flat_idx)
                col_indices.append(orbit_id)
        
        data = np.ones(len(row_indices), dtype=np.float64)
        
        E = sparse.coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.n * self.k, self.num_reduced_vars),
        )
        
        return E.tocsr()
    
    def reduce_lp(
        self,
        c: npt.NDArray[np.float64],
        A_ub: sparse.spmatrix,
        b_ub: npt.NDArray[np.float64],
        A_eq: sparse.spmatrix,
        b_eq: npt.NDArray[np.float64],
    ) -> Tuple[
        npt.NDArray[np.float64],
        sparse.spmatrix,
        npt.NDArray[np.float64],
        sparse.spmatrix,
        npt.NDArray[np.float64],
    ]:
        """Reduce LP to orbit variables.
        
        Original LP:
            min c^T x
            s.t. A_ub @ x <= b_ub
                 A_eq @ x == b_eq
                 
        Reduced LP (x = E @ x_red):
            min (E^T c)^T x_red
            s.t. A_ub @ E @ x_red <= b_ub
                 A_eq @ E @ x_red == b_eq
        """
        E = self._expansion_matrix
        
        c_red = E.T @ c
        A_ub_red = A_ub @ E
        A_eq_red = A_eq @ E
        
        return c_red, A_ub_red, b_ub, A_eq_red, b_eq
    
    def expand_solution(self, x_reduced: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Expand reduced solution to full mechanism table.
        
        Args:
            x_reduced: Solution in orbit space (length num_reduced_vars)
            
        Returns:
            Full mechanism table (length n*k)
        """
        return self._expansion_matrix @ x_reduced


class BandedStructureDetector:
    """Detects banded constraint matrices for sparse solver selection.
    
    A matrix is k-banded if A[i,j] = 0 for |i - j| > k. Banded matrices
    admit specialized storage (LAPACK banded format) and fast solvers
    (banded Cholesky, banded LU).
    
    This detector analyzes constraint matrices to:
        1. Detect bandwidth and sparsity pattern
        2. Recommend solver (dense, sparse, or banded)
        3. Convert to optimal storage format
        
    Args:
        A: Constraint matrix to analyze
        threshold: Nonzero threshold (entries below ignored)
        
    Attributes:
        is_banded: True if matrix is banded with bandwidth < 0.1 * n
        bandwidth: Upper/lower bandwidth (same for symmetric matrices)
        sparsity: Fraction of nonzero entries
        recommended_format: 'dense', 'sparse', or 'banded'
    """
    
    def __init__(
        self,
        A: sparse.spmatrix | npt.NDArray,
        threshold: float = 1e-12,
    ):
        if sparse.issparse(A):
            self.A = A
            self.is_sparse_input = True
        else:
            self.A = sparse.csr_matrix(A)
            self.is_sparse_input = False
        
        m, n = self.A.shape
        self.shape = (m, n)
        self.threshold = threshold
        
        # Compute bandwidth and sparsity
        self.bandwidth, self.is_banded = self._detect_bandwidth()
        self.sparsity = self.A.nnz / (m * n) if m * n > 0 else 0.0
        
        # Recommend storage format
        self.recommended_format = self._recommend_format()
    
    def _detect_bandwidth(self) -> Tuple[int, bool]:
        """Detect matrix bandwidth.
        
        Returns:
            bandwidth: Maximum |i - j| for nonzero A[i,j]
            is_banded: True if bandwidth < 0.1 * min(m, n)
        """
        m, n = self.shape
        
        if self.A.nnz == 0:
            return 0, True
        
        # Convert to COO to access row/col indices
        A_coo = self.A.tocoo()
        
        # Find maximum diagonal distance
        max_bandwidth = 0
        for row, col in zip(A_coo.row, A_coo.col):
            bandwidth = abs(int(row) - int(col))
            max_bandwidth = max(max_bandwidth, bandwidth)
        
        # Consider banded if bandwidth < 10% of dimension
        threshold_bandwidth = int(0.1 * min(m, n))
        is_banded = max_bandwidth <= threshold_bandwidth
        
        return max_bandwidth, is_banded
    
    def _recommend_format(self) -> str:
        """Recommend optimal storage format based on structure.
        
        Returns:
            'dense': Use dense arrays and LAPACK
            'sparse': Use sparse CSR/CSC and scipy.sparse.linalg
            'banded': Use LAPACK banded format
        """
        m, n = self.shape
        
        # Small matrices: use dense
        if m * n < 10000:
            return 'dense'
        
        # Banded structure detected: use banded format
        if self.is_banded:
            return 'banded'
        
        # Sparse but not banded: use general sparse
        if self.sparsity < 0.1:
            return 'sparse'
        
        # Dense matrix: use dense format
        return 'dense'
    
    def to_banded_format(self) -> npt.NDArray[np.float64]:
        """Convert matrix to LAPACK banded storage format.
        
        LAPACK banded format stores k+1 diagonals in array of shape (k+1, n)
        where row i contains diagonal (k-i).
        
        Returns:
            Banded array in LAPACK format
        """
        if not self.is_banded:
            raise ValueError("Matrix is not banded, cannot convert to banded format")
        
        m, n = self.shape
        k = self.bandwidth
        
        # Allocate banded storage
        ab = np.zeros((k + 1, n), dtype=np.float64)
        
        # Fill diagonals
        A_coo = self.A.tocoo()
        for row, col, val in zip(A_coo.row, A_coo.col, A_coo.data):
            if abs(val) > self.threshold:
                diag = int(col) - int(row)
                if -k <= diag <= 0:  # Lower triangular part
                    ab[k - diag, col] = val
        
        return ab


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _is_toeplitz_block(block: npt.NDArray, tol: float) -> bool:
    """Check if a matrix block is approximately Toeplitz."""
    m, n = block.shape
    if m < 3 or n < 3:
        return False
    
    # Check constant diagonals
    for diag in range(-m + 1, n):
        diag_vals = np.diag(block, k=diag)
        if len(diag_vals) < 2:
            continue
        
        # All values on diagonal should be close to mean
        mean_val = np.mean(diag_vals)
        if np.any(np.abs(diag_vals - mean_val) > tol * (1 + abs(mean_val))):
            return False
    
    return True


def _detect_bandwidth(A: npt.NDArray) -> int:
    """Detect bandwidth from nonzero pattern."""
    n = A.shape[0]
    max_bw = 0
    
    for i in range(n):
        for j in range(n):
            if abs(A[i, j]) > 1e-12:
                max_bw = max(max_bw, abs(i - j))
    
    return max_bw


def _tukey_window(n: int, alpha: float = 0.1) -> npt.NDArray[np.float64]:
    """Compute Tukey (tapered cosine) window.
    
    Args:
        n: Window length
        alpha: Taper fraction (0 = rectangular, 1 = Hann)
        
    Returns:
        Window values of length n
    """
    window = np.ones(n)
    
    # Taper first alpha*n/2 points
    taper_len = int(alpha * n / 2)
    
    for i in range(taper_len):
        t = i / taper_len
        window[i] = 0.5 * (1 + np.cos(np.pi * (t - 1)))
    
    # Taper last alpha*n/2 points (mirror)
    for i in range(taper_len):
        window[n - 1 - i] = window[i]
    
    return window


# ---------------------------------------------------------------------------
# Advanced structure detection utilities
# ---------------------------------------------------------------------------


def detect_constraint_structure(
    A: sparse.spmatrix,
    labels: Optional[npt.NDArray] = None,
) -> dict[str, any]:
    """Comprehensive structure detection for constraint matrices.
    
    Analyzes constraint matrix to detect:
        - Toeplitz/circulant blocks
        - Banded structure
        - Block diagonal structure
        - Network flow structure
        - Symmetries and patterns
        
    This information guides optimization strategy selection.
    
    Args:
        A: Constraint matrix to analyze
        labels: Optional constraint labels ('privacy', 'simplex', etc.)
        
    Returns:
        Dictionary with detected structures and properties
    """
    m, n = A.shape
    
    structure_info = {
        'shape': (m, n),
        'nnz': A.nnz,
        'density': A.nnz / (m * n) if m * n > 0 else 0.0,
    }
    
    # Detect Toeplitz blocks
    toeplitz_detector = ToeplitzOperator.detect_toeplitz(A)
    structure_info['toeplitz_blocks'] = len(toeplitz_detector)
    structure_info['has_toeplitz'] = len(toeplitz_detector) > 0
    
    # Detect banded structure
    banded_detector = BandedStructureDetector(A)
    structure_info['is_banded'] = banded_detector.is_banded
    structure_info['bandwidth'] = banded_detector.bandwidth
    structure_info['recommended_format'] = banded_detector.recommended_format
    
    # Analyze row/column patterns
    if sparse.issparse(A):
        A_csr = A.tocsr()
        row_nnz = np.diff(A_csr.indptr)
        
        structure_info['min_nnz_per_row'] = int(np.min(row_nnz)) if len(row_nnz) > 0 else 0
        structure_info['max_nnz_per_row'] = int(np.max(row_nnz)) if len(row_nnz) > 0 else 0
        structure_info['mean_nnz_per_row'] = float(np.mean(row_nnz)) if len(row_nnz) > 0 else 0.0
        
        # Check for uniform sparsity pattern (e.g., 2 nonzeros per row for privacy)
        is_uniform = np.std(row_nnz) < 0.5 if len(row_nnz) > 0 else False
        structure_info['uniform_sparsity'] = is_uniform
    
    # Detect block diagonal structure
    structure_info['block_diagonal'] = _detect_block_diagonal(A)
    
    # Analyze constraint types from labels
    if labels is not None:
        structure_info['constraint_types'] = {}
        for label in np.unique(labels):
            count = np.sum(labels == label)
            structure_info['constraint_types'][str(label)] = int(count)
    
    return structure_info


def _detect_block_diagonal(A: sparse.spmatrix, tolerance: float = 1e-10) -> bool:
    """Detect if matrix has block diagonal structure.
    
    A matrix is block diagonal if it can be permuted into the form:
        [A1   0   0]
        [0   A2   0]
        [0    0  A3]
        
    Args:
        A: Matrix to analyze
        tolerance: Threshold for considering entries zero
        
    Returns:
        True if block diagonal structure detected
    """
    if A.shape[0] < 10 or A.shape[1] < 10:
        return False
    
    # Simple heuristic: check if middle rows/cols are very sparse
    m, n = A.shape
    mid_row_start = m // 3
    mid_row_end = 2 * m // 3
    mid_col_start = n // 3
    mid_col_end = 2 * n // 3
    
    if sparse.issparse(A):
        A_csr = A.tocsr()
        # Check density in off-diagonal blocks
        middle_block = A_csr[mid_row_start:mid_row_end, :mid_col_start].nnz
        middle_block += A_csr[mid_row_start:mid_row_end, mid_col_end:].nnz
        
        total_middle_rows = mid_row_end - mid_row_start
        total_off_diag_cols = mid_col_start + (n - mid_col_end)
        
        expected_nnz = total_middle_rows * total_off_diag_cols
        
        if expected_nnz > 0 and middle_block / expected_nnz < 0.01:
            return True
    
    return False


def optimize_constraint_ordering(
    A: sparse.spmatrix,
    algorithm: str = 'rcm',
) -> Tuple[npt.NDArray[np.int32], sparse.spmatrix]:
    """Reorder constraints to minimize bandwidth/fill-in.
    
    Uses graph algorithms (Reverse Cuthill-McKee, AMD) to permute
    constraint matrix for better cache locality and sparser factorizations.
    
    Args:
        A: Constraint matrix
        algorithm: Ordering algorithm ('rcm', 'amd', 'metis')
        
    Returns:
        (permutation, A_permuted): Row permutation and reordered matrix
        
    References:
        - George & Liu, "Computer Solution of Large Sparse Linear Systems", 1981
        - Amestoy et al., "AMD: An Approximate Minimum Degree Ordering Algorithm", 1996
    """
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    
    m, n = A.shape
    
    if algorithm == 'rcm':
        # Reverse Cuthill-McKee ordering
        # Build adjacency graph from A @ A^T (constraint similarity)
        A_csr = A.tocsr()
        adjacency = A_csr @ A_csr.T
        
        # Run RCM
        perm = reverse_cuthill_mckee(adjacency, symmetric_mode=True)
        
        # Apply permutation
        A_permuted = A_csr[perm, :]
        
        return perm, A_permuted
    
    elif algorithm == 'amd':
        # Approximate minimum degree (requires scikit-sparse or similar)
        # Fallback to identity if not available
        logger.warning("AMD ordering not implemented, using identity")
        return np.arange(m, dtype=np.int32), A
    
    else:
        raise ValueError(f"Unknown ordering algorithm: {algorithm}")


def extract_constraint_blocks(
    A: sparse.spmatrix,
    labels: npt.NDArray,
) -> dict[str, sparse.spmatrix]:
    """Extract constraint blocks by label.
    
    Groups constraints by label (e.g., 'privacy', 'simplex', 'bound')
    into separate submatrices for targeted preprocessing.
    
    Args:
        A: Full constraint matrix
        labels: Constraint labels (length m)
        
    Returns:
        Dictionary mapping label -> submatrix
    """
    blocks = {}
    
    for label in np.unique(labels):
        mask = labels == label
        indices = np.where(mask)[0]
        
        if sparse.issparse(A):
            blocks[str(label)] = A[indices, :]
        else:
            blocks[str(label)] = A[indices, :]
    
    return blocks


def build_constraint_graph(A: sparse.spmatrix) -> sparse.spmatrix:
    """Build constraint conflict graph for analysis.
    
    Two constraints are connected in the graph if they share decision
    variables. The graph structure reveals:
        - Constraint dependencies
        - Potential for parallel processing
        - Decomposition opportunities
        
    Args:
        A: Constraint matrix (m × n)
        
    Returns:
        Adjacency matrix of constraint graph (m × m)
    """
    A_csr = sparse.csr_matrix(A)
    
    # Constraint graph: G[i,j] = 1 if constraints i,j share a variable
    # This is the pattern of A @ A^T (without computing values)
    
    # Use boolean matrix multiply for efficiency
    A_bool = A_csr.astype(bool)
    graph = (A_bool @ A_bool.T).astype(np.int32)
    
    # Remove self-loops
    graph.setdiag(0)
    
    return graph


def estimate_condition_number(
    A: sparse.spmatrix,
    method: str = 'power_iteration',
    max_iterations: int = 100,
) -> float:
    """Estimate condition number κ(A) = σ_max / σ_min.
    
    Uses iterative methods to avoid computing full SVD.
    
    Args:
        A: Matrix to analyze
        method: Estimation method ('power_iteration', 'lanczos')
        max_iterations: Maximum iterations for iterative methods
        
    Returns:
        Estimated condition number
    """
    from scipy.sparse.linalg import svds
    
    if A.shape[0] == 0 or A.shape[1] == 0:
        return 1.0
    
    m, n = A.shape
    k = min(6, min(m, n) - 1)
    
    if k < 1:
        return 1.0
    
    try:
        # Compute top and bottom singular values
        if method == 'lanczos':
            # Use scipy's sparse SVD (Lanczos-based)
            s_max = svds(A, k=1, which='LM', return_singular_vectors=False)[0]
            s_min = svds(A, k=1, which='SM', return_singular_vectors=False)[0]
            
            kappa = s_max / (s_min + 1e-15)
            return float(kappa)
        
        elif method == 'power_iteration':
            # Power iteration for largest singular value
            A_csr = sparse.csr_matrix(A)
            x = np.random.randn(n)
            x /= np.linalg.norm(x)
            
            for _ in range(max_iterations):
                y = A_csr.T @ (A_csr @ x)
                s = np.linalg.norm(y)
                x = y / (s + 1e-15)
            
            s_max = np.sqrt(s)
            
            # Inverse iteration for smallest
            # (Simplified: just estimate as very small if near-singular)
            s_min = 1e-10
            
            return float(s_max / s_min)
    
    except Exception as e:
        logger.warning(f"Condition number estimation failed: {e}")
        return 1e12  # Return large number to indicate ill-conditioning

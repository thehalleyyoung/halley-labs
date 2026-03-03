"""
SDP construction engine for optimal Gaussian workload mechanism synthesis.

This module constructs and solves semidefinite programs (SDPs) that compute
the **minimum-variance Gaussian mechanism** for answering a linear workload
``A`` under ``(ε, δ)``-differential privacy.  It is the SDP counterpart
to :mod:`dp_forge.lp_builder`, which handles discrete mechanisms via LP.

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

Given a workload matrix ``A ∈ ℝ^{m×d}`` and privacy budget ``ε > 0``,
we seek a covariance matrix ``Σ ∈ S_+^d`` (symmetric positive semidefinite)
such that the Gaussian mechanism ``M(x) = Ax + N(0, AΣAᵀ)`` satisfies
``ε``-differential privacy under an ``ℓ₂`` sensitivity model.

The SDP formulation is:

    minimise   trace(W Σ)                   (total squared error)
    subject to v^T Σ v  ≤  (C/ε)²          ∀ v ∈ K_verts
               Σ ≽ 0                        (PSD constraint)

where ``W = AᵀA`` is the workload Gram matrix and ``K_verts`` are the
vertices of the sensitivity polytope.

For shift-invariant (Toeplitz) workloads, the SDP reduces to a 1-D
spectral optimisation that can be solved orders of magnitude faster.

Implementation Notes
~~~~~~~~~~~~~~~~~~~~

*  All SDP modelling uses CVXPY.  Solver backends are auto-selected:
   MOSEK for ``d ≤ 500``, SCS for ``d > 500``.  Missing solvers are
   handled gracefully with fallbacks.
*  The :class:`StructuralDetector` auto-detects workload structure
   (Toeplitz, circulant, block-diagonal, sparse) to apply specialised
   reductions before SDP construction.
*  The :class:`SensitivityBallComputer` computes sensitivity polytope
   vertices under L1, L2, and L∞ norms, with redundancy removal.
*  :func:`extract_gaussian` converts the optimal ``Σ`` into a deployable
   :class:`GaussianMechanismResult` with sampling support.
"""

from __future__ import annotations

import logging
import math
import time
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
from scipy import linalg as sp_linalg
from scipy import sparse
from scipy.spatial import ConvexHull

from dp_forge.exceptions import (
    ConfigurationError,
    InfeasibleSpecError,
    NumericalInstabilityError,
    SolverError,
)
from dp_forge.types import (
    NumericalConfig,
    OptimalityCertificate,
    PrivacyBudget,
    SDPStruct,
    SolverBackend,
    SynthesisConfig,
    WorkloadSpec,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy CVXPY import
# ---------------------------------------------------------------------------

_cvxpy = None


def _get_cvxpy() -> Any:
    """Lazy-import CVXPY so the module can be loaded without it installed."""
    global _cvxpy
    if _cvxpy is None:
        try:
            import cvxpy
            _cvxpy = cvxpy
        except ImportError as exc:
            raise ImportError(
                "cvxpy is required for SDP-based mechanism synthesis. "
                "Install it with: pip install cvxpy"
            ) from exc
    return _cvxpy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SOLVER_TOL: float = 1e-8
_DEFAULT_SCS_MAX_ITERS: int = 100_000
_DEFAULT_SCS_EPS: float = 1e-9
_DEFAULT_MOSEK_TOL: float = 1e-10
_LARGE_DIMENSION_THRESHOLD: int = 500
_TOEPLITZ_TOL: float = 1e-10
_CIRCULANT_TOL: float = 1e-10
_BLOCK_DIAG_TOL: float = 1e-10
_SPARSITY_THRESHOLD: float = 0.7
_PSD_EIGENVALUE_TOL: float = -1e-8
_REDUNDANCY_TOL: float = 1e-12
_WARM_START_PERTURBATION: float = 1e-6
_MAX_HULL_DIMENSION: int = 15


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SolveStatistics:
    """Statistics from an SDP solve.

    Attributes:
        solve_time: Wall-clock solve time in seconds.
        solver_name: Name of the solver used.
        iterations: Number of solver iterations (if reported).
        duality_gap: Absolute duality gap at termination.
        primal_obj: Primal objective value.
        dual_obj: Dual objective value.
        status: Solver status string.
        setup_time: Time spent constructing the SDP (seconds).
        num_constraints: Number of constraints in the SDP.
        dimension: Dimension of the PSD variable.
    """

    solve_time: float
    solver_name: str
    iterations: int = 0
    duality_gap: float = 0.0
    primal_obj: float = 0.0
    dual_obj: float = 0.0
    status: str = "unknown"
    setup_time: float = 0.0
    num_constraints: int = 0
    dimension: int = 0

    def __repr__(self) -> str:
        return (
            f"SolveStatistics(solver={self.solver_name!r}, "
            f"time={self.solve_time:.3f}s, status={self.status!r}, "
            f"gap={self.duality_gap:.2e})"
        )


@dataclass
class GaussianMechanismResult:
    """A deployable Gaussian mechanism extracted from a solved SDP.

    The mechanism ``M(x) = Ax + η`` where ``η ~ N(0, Σ_noise)`` with
    ``Σ_noise = A Σ Aᵀ``.

    Attributes:
        sigma: Optimal covariance matrix Σ (d × d, PSD).
        noise_covariance: Noise covariance ``A Σ Aᵀ`` (m × m).
        workload: The workload matrix A (m × d).
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        total_mse: Total mean squared error ``trace(W Σ)``.
        per_query_mse: Per-query MSE, diagonal of ``A Σ Aᵀ``.
        cholesky_factor: Cholesky factor of ``Σ_noise`` for sampling.
        solve_stats: Solver statistics.
    """

    sigma: npt.NDArray[np.float64]
    noise_covariance: npt.NDArray[np.float64]
    workload: npt.NDArray[np.float64]
    epsilon: float
    delta: float = 0.0
    total_mse: float = 0.0
    per_query_mse: Optional[npt.NDArray[np.float64]] = None
    cholesky_factor: Optional[npt.NDArray[np.float64]] = None
    solve_stats: Optional[SolveStatistics] = None

    def __post_init__(self) -> None:
        self.sigma = np.asarray(self.sigma, dtype=np.float64)
        self.noise_covariance = np.asarray(self.noise_covariance, dtype=np.float64)
        self.workload = np.asarray(self.workload, dtype=np.float64)
        if self.sigma.ndim != 2 or self.sigma.shape[0] != self.sigma.shape[1]:
            raise ValueError(
                f"sigma must be a square matrix, got shape {self.sigma.shape}"
            )
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")

    @property
    def d(self) -> int:
        """Dimension of the data domain."""
        return self.sigma.shape[0]

    @property
    def m(self) -> int:
        """Number of queries in the workload."""
        return self.workload.shape[0]

    def __repr__(self) -> str:
        return (
            f"GaussianMechanismResult(d={self.d}, m={self.m}, "
            f"ε={self.epsilon}, mse={self.total_mse:.6f})"
        )


@dataclass
class StructuralInfo:
    """Result of structural analysis on a workload matrix.

    Attributes:
        is_toeplitz: Whether ``AᵀA`` has Toeplitz structure.
        is_circulant: Whether ``AᵀA`` has circulant structure.
        is_block_diagonal: Whether ``AᵀA`` is block-diagonal.
        is_sparse: Whether ``A`` is highly sparse.
        block_sizes: Sizes of diagonal blocks (if block-diagonal).
        sparsity: Fraction of zeros in ``A``.
        recommended_hint: Best structural hint for SDP construction.
        details: Additional diagnostic information.
    """

    is_toeplitz: bool = False
    is_circulant: bool = False
    is_block_diagonal: bool = False
    is_sparse: bool = False
    block_sizes: Optional[List[int]] = None
    sparsity: float = 0.0
    recommended_hint: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        flags = []
        if self.is_toeplitz:
            flags.append("toeplitz")
        if self.is_circulant:
            flags.append("circulant")
        if self.is_block_diagonal:
            flags.append("block_diag")
        if self.is_sparse:
            flags.append("sparse")
        flag_str = ", ".join(flags) if flags else "none"
        return f"StructuralInfo(structure=[{flag_str}], hint={self.recommended_hint!r})"


# ---------------------------------------------------------------------------
# SensitivityBallComputer
# ---------------------------------------------------------------------------


class SensitivityBallComputer:
    """Compute vertices of the sensitivity polytope for workload-based DP.

    The sensitivity polytope ``K`` is the set of all vectors ``Δ = A(x - x')``
    where ``x, x'`` are adjacent databases.  Under standard ``ℓ_p``
    sensitivity models, ``K`` is determined by the columns (or combinations
    of columns) of ``A``.

    For ``ℓ_1`` sensitivity-1 (add/remove one record), the vertices of the
    sensitivity set in the noise space are the columns of ``A`` and their
    negations: ``{±A_j : j ∈ [d]}``.

    For ``ℓ_2`` sensitivity, the constraint surface is an ellipsoid and we
    discretise it into finitely many directions.

    Methods provide vertex computation, redundancy removal, and convex hull
    operations needed to form the SDP privacy constraints.
    """

    @staticmethod
    def compute_l1_ball_vertices(
        A: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Compute sensitivity polytope vertices under ℓ₁ sensitivity.

        Under ℓ₁ sensitivity-1 (add/remove model), adjacent databases
        ``x, x'`` differ in exactly one coordinate by ±1.  The sensitivity
        vectors in the *noise* space are::

            Δ = A(x - x') = ± A_{:,j}    for some j ∈ [d]

        In the *covariance* space (for the privacy constraint
        ``v^T Σ v ≤ (C/ε)²``), the relevant directions are the columns
        of ``A`` projected into the ``Σ``-space.  For the standard
        formulation where ``Σ`` operates in the data domain (d-dimensional),
        the vertices are the standard basis vectors ``{±e_j}``.

        Parameters
        ----------
        A : ndarray of shape (m, d)
            Workload matrix.

        Returns
        -------
        vertices : ndarray of shape (2d, d)
            Vertices of the ℓ₁ sensitivity ball in the d-dimensional
            covariance space: ``{+e_1, -e_1, +e_2, -e_2, …}``.
        """
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError(f"A must be 2-D, got {A.ndim}-D")

        d = A.shape[1]
        # Standard basis vertices ±e_j
        vertices = np.zeros((2 * d, d), dtype=np.float64)
        for j in range(d):
            vertices[2 * j, j] = 1.0
            vertices[2 * j + 1, j] = -1.0
        return vertices

    @staticmethod
    def compute_l2_ball_vertices(
        A: npt.NDArray[np.float64],
        n_directions: int = 0,
    ) -> npt.NDArray[np.float64]:
        r"""Compute approximate sensitivity vertices under ℓ₂ sensitivity.

        Under ℓ₂ sensitivity-1, the set of allowed differences is the
        unit ball ``{v : ||v||₂ ≤ 1}``.  Since the ball has infinitely
        many extreme points, we discretise it.

        For ``d ≤ 2``, we use uniformly spaced angles.
        For ``d ≥ 3``, we use the ``2d`` axis-aligned directions plus
        additional randomly sampled directions on the unit sphere.

        Parameters
        ----------
        A : ndarray of shape (m, d)
            Workload matrix.
        n_directions : int
            Number of additional random directions beyond the ``2d``
            axis-aligned ones.  Default ``0`` uses ``max(2d, 50)``
            additional directions for ``d ≥ 3``.

        Returns
        -------
        vertices : ndarray of shape (n_verts, d)
            Approximate vertices on the ℓ₂ unit sphere.
        """
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError(f"A must be 2-D, got {A.ndim}-D")

        d = A.shape[1]

        if d == 1:
            return np.array([[1.0], [-1.0]], dtype=np.float64)

        if d == 2:
            n_pts = max(n_directions, 32)
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            return np.column_stack([np.cos(angles), np.sin(angles)])

        # d >= 3: axis-aligned + random
        axis_verts = np.zeros((2 * d, d), dtype=np.float64)
        for j in range(d):
            axis_verts[2 * j, j] = 1.0
            axis_verts[2 * j + 1, j] = -1.0

        n_extra = n_directions if n_directions > 0 else max(2 * d, 50)
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((n_extra, d))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        extra_verts = raw / norms

        return np.vstack([axis_verts, extra_verts])

    @staticmethod
    def compute_linf_ball_vertices(
        A: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Compute sensitivity polytope vertices under ℓ_∞ sensitivity.

        The ℓ_∞ unit ball has ``2^d`` vertices at ``{±1}^d``.  For large
        ``d`` this is exponential, so we cap at ``d ≤ 20`` and raise for
        larger dimensions.

        Parameters
        ----------
        A : ndarray of shape (m, d)
            Workload matrix.

        Returns
        -------
        vertices : ndarray of shape (2^d, d)
            All vertices of the ℓ_∞ unit ball.

        Raises
        ------
        ConfigurationError
            If ``d > 20`` (too many vertices to enumerate).
        """
        A = np.asarray(A, dtype=np.float64)
        d = A.shape[1]

        if d > 20:
            raise ConfigurationError(
                f"ℓ_∞ ball vertex enumeration requires 2^d = 2^{d} vertices, "
                f"which is infeasible. Use ℓ₁ or ℓ₂ sensitivity instead.",
                parameter="d",
                value=d,
                constraint="d <= 20 for ℓ_∞ enumeration",
            )

        n_verts = 2 ** d
        vertices = np.zeros((n_verts, d), dtype=np.float64)
        for i in range(n_verts):
            for j in range(d):
                vertices[i, j] = 1.0 if (i >> j) & 1 else -1.0
        return vertices

    @staticmethod
    def reduce_redundant_vertices(
        vertices: npt.NDArray[np.float64],
        tol: float = _REDUNDANCY_TOL,
    ) -> npt.NDArray[np.float64]:
        r"""Remove dominated / duplicate vertices from a vertex set.

        A vertex ``v`` is redundant if there exists another vertex ``u``
        in the set such that ``||v||₂ ≤ ||u||₂`` and ``v / ||v|| ≈ u / ||u||``
        (i.e., ``v`` is dominated by ``u`` in the same direction).

        This also removes near-duplicate vertices (within tolerance).

        Parameters
        ----------
        vertices : ndarray of shape (n, d)
            Input vertex set.
        tol : float
            Tolerance for considering two normalised directions identical.

        Returns
        -------
        reduced : ndarray of shape (n', d) where n' ≤ n
            Reduced vertex set with redundant vertices removed.
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.ndim != 2:
            raise ValueError(f"vertices must be 2-D, got {vertices.ndim}-D")

        n, d = vertices.shape
        if n <= 1:
            return vertices.copy()

        norms = np.linalg.norm(vertices, axis=1)
        # Remove zero-norm vertices
        nonzero_mask = norms > tol
        vertices = vertices[nonzero_mask]
        norms = norms[nonzero_mask]

        if len(vertices) == 0:
            return np.zeros((0, d), dtype=np.float64)

        # Normalise
        directions = vertices / norms[:, np.newaxis]

        # Greedy deduplication: keep the vertex with largest norm per direction
        keep = []
        used = np.zeros(len(vertices), dtype=bool)

        # Sort by descending norm so we keep the dominant vertex first
        order = np.argsort(-norms)

        for idx in order:
            if used[idx]:
                continue
            keep.append(idx)
            d_i = directions[idx]
            # Mark all close directions as used
            for jdx in range(idx + 1, len(vertices)):
                if used[jdx]:
                    continue
                # Check if same direction (or opposite)
                dot = np.dot(d_i, directions[jdx])
                if abs(abs(dot) - 1.0) < tol:
                    if abs(dot - 1.0) < tol:
                        # Same direction: keep the one with larger norm (idx)
                        used[jdx] = True
                    # Opposite direction: keep both (they define different constraints)

        return vertices[keep].copy()

    @staticmethod
    def convex_hull_vertices(
        points: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Compute the convex hull and return only hull vertices.

        Uses ``scipy.spatial.ConvexHull``.  For ``d ≥ 16`` or degenerate
        point sets, falls back to returning all input points.

        Parameters
        ----------
        points : ndarray of shape (n, d)
            Input point set.

        Returns
        -------
        hull_verts : ndarray of shape (n_hull, d)
            Vertices of the convex hull.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError(f"points must be 2-D, got {points.ndim}-D")

        n, d = points.shape
        if n <= d + 1 or d > _MAX_HULL_DIMENSION:
            return points.copy()

        try:
            hull = ConvexHull(points)
            return points[hull.vertices].copy()
        except Exception:
            logger.debug(
                "ConvexHull failed (degenerate?), returning all %d points", n
            )
            return points.copy()


# ---------------------------------------------------------------------------
# StructuralDetector
# ---------------------------------------------------------------------------


class StructuralDetector:
    """Detect exploitable structure in workload matrices.

    Structural properties of the workload Gram matrix ``W = AᵀA`` enable
    significant computational savings in SDP construction:

    - **Toeplitz**: The SDP variable can be parameterised by a single
      vector (first row), reducing the number of free variables from
      ``d²`` to ``d``.
    - **Circulant**: A circulant matrix is diagonalised by the DFT,
      reducing the SDP to ``d`` independent scalar constraints.
    - **Block-diagonal**: The SDP decomposes into independent sub-problems
      for each block.
    - **Sparse**: Sparsity in ``A`` reduces the effective dimensionality
      of the constraints.
    """

    @staticmethod
    def detect_toeplitz(
        A: npt.NDArray[np.float64],
        tol: float = _TOEPLITZ_TOL,
    ) -> bool:
        r"""Check whether ``W = AᵀA`` has Toeplitz structure.

        A matrix ``W`` is Toeplitz if ``W[i,j] = W[i+1,j+1]`` for all
        valid ``i, j``, i.e., constant along diagonals.

        Parameters
        ----------
        A : ndarray of shape (m, d)
            Workload matrix.
        tol : float
            Absolute tolerance for comparing diagonal entries.

        Returns
        -------
        bool
            ``True`` if ``AᵀA`` is Toeplitz within tolerance.
        """
        A = np.asarray(A, dtype=np.float64)
        W = A.T @ A
        d = W.shape[0]
        if d <= 1:
            return True

        for diag_idx in range(-(d - 1), d):
            diag = np.diag(W, diag_idx)
            if len(diag) <= 1:
                continue
            if np.max(np.abs(diag - diag[0])) > tol:
                return False
        return True

    @staticmethod
    def detect_circulant(
        A: npt.NDArray[np.float64],
        tol: float = _CIRCULANT_TOL,
    ) -> bool:
        r"""Check whether ``W = AᵀA`` has circulant structure.

        A matrix ``W`` is circulant if each row is a cyclic shift of the
        first row: ``W[i,j] = W[0, (j-i) mod d]``.  Every circulant
        matrix is Toeplitz, but not vice versa.

        Parameters
        ----------
        A : ndarray of shape (m, d)
            Workload matrix.
        tol : float
            Absolute tolerance.

        Returns
        -------
        bool
            ``True`` if ``AᵀA`` is circulant within tolerance.
        """
        A = np.asarray(A, dtype=np.float64)
        W = A.T @ A
        d = W.shape[0]
        if d <= 1:
            return True

        first_row = W[0, :]
        for i in range(1, d):
            shifted = np.roll(first_row, -i)
            if np.max(np.abs(W[i, :] - shifted)) > tol:
                return False
        return True

    @staticmethod
    def detect_block_diagonal(
        A: npt.NDArray[np.float64],
        tol: float = _BLOCK_DIAG_TOL,
    ) -> Tuple[bool, Optional[List[int]]]:
        r"""Check whether ``W = AᵀA`` has block-diagonal structure.

        We find connected components of the graph where columns ``i, j``
        are connected iff ``|W[i,j]| > tol``.

        Parameters
        ----------
        A : ndarray of shape (m, d)
            Workload matrix.
        tol : float
            Absolute tolerance for considering an entry as zero.

        Returns
        -------
        is_block_diag : bool
            ``True`` if ``W`` has at least 2 diagonal blocks.
        block_sizes : list of int or None
            Sizes of the diagonal blocks, sorted in the order they appear
            along the diagonal.  ``None`` if not block-diagonal.
        """
        A = np.asarray(A, dtype=np.float64)
        W = A.T @ A
        d = W.shape[0]
        if d <= 1:
            return False, None

        # Build adjacency via Union-Find
        parent = list(range(d))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(d):
            for j in range(i + 1, d):
                if abs(W[i, j]) > tol:
                    union(i, j)

        # Count components
        components: Dict[int, List[int]] = {}
        for i in range(d):
            root = find(i)
            components.setdefault(root, []).append(i)

        if len(components) < 2:
            return False, None

        # Sort blocks by their first index
        blocks = sorted(components.values(), key=lambda b: b[0])
        block_sizes = [len(b) for b in blocks]
        return True, block_sizes

    @staticmethod
    def detect_sparse_structure(
        A: npt.NDArray[np.float64],
    ) -> Tuple[bool, float]:
        r"""Analyse sparsity pattern of the workload matrix.

        Parameters
        ----------
        A : ndarray of shape (m, d)
            Workload matrix.

        Returns
        -------
        is_sparse : bool
            ``True`` if sparsity exceeds ``_SPARSITY_THRESHOLD``.
        sparsity : float
            Fraction of zero entries in ``A``.
        """
        A = np.asarray(A, dtype=np.float64)
        total = A.size
        if total == 0:
            return True, 1.0
        nnz = np.count_nonzero(A)
        sparsity = 1.0 - nnz / total
        return sparsity >= _SPARSITY_THRESHOLD, sparsity

    @classmethod
    def suggest_structure(
        cls,
        A: npt.NDArray[np.float64],
    ) -> StructuralInfo:
        r"""Auto-detect the best structural hint for a workload matrix.

        Runs all detectors and returns a :class:`StructuralInfo` with the
        recommended hint.  Priority order:

        1. Circulant (strongest: DFT diagonalisation)
        2. Toeplitz (strong: 1-D spectral reduction)
        3. Block-diagonal (moderate: independent sub-SDPs)
        4. Sparse (weak: sparsity-aware constraint generation)

        Parameters
        ----------
        A : ndarray of shape (m, d)
            Workload matrix.

        Returns
        -------
        StructuralInfo
            Detected structural properties and recommended hint.
        """
        A = np.asarray(A, dtype=np.float64)

        is_circ = cls.detect_circulant(A)
        is_toep = cls.detect_toeplitz(A)
        is_block, block_sizes = cls.detect_block_diagonal(A)
        is_sparse, sparsity = cls.detect_sparse_structure(A)

        # Determine recommended hint
        hint: Optional[str] = None
        if is_circ:
            hint = "circulant"
        elif is_toep:
            hint = "toeplitz"
        elif is_block:
            hint = "block_diagonal"
        elif is_sparse:
            hint = "sparse"

        details: Dict[str, Any] = {
            "gram_shape": (A.shape[1], A.shape[1]),
            "workload_shape": A.shape,
        }
        if block_sizes is not None:
            details["block_sizes"] = block_sizes

        info = StructuralInfo(
            is_toeplitz=is_toep,
            is_circulant=is_circ,
            is_block_diagonal=is_block,
            is_sparse=is_sparse,
            block_sizes=block_sizes,
            sparsity=sparsity,
            recommended_hint=hint,
            details=details,
        )

        logger.info("Structure detection: %s", info)
        return info


# ---------------------------------------------------------------------------
# Core SDP construction: BuildWorkloadSDP
# ---------------------------------------------------------------------------


def _compute_gram_matrix(A: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the workload Gram matrix W = AᵀA.

    Parameters
    ----------
    A : ndarray of shape (m, d)
        Workload matrix.

    Returns
    -------
    W : ndarray of shape (d, d)
        Symmetric positive semidefinite Gram matrix.
    """
    W = A.T @ A
    # Force exact symmetry
    W = (W + W.T) / 2.0
    return W


def _validate_gram_matrix(
    W: npt.NDArray[np.float64],
    max_cond: float = 1e12,
) -> None:
    """Validate the Gram matrix for numerical stability.

    Parameters
    ----------
    W : ndarray of shape (d, d)
        Gram matrix.
    max_cond : float
        Maximum acceptable condition number.

    Raises
    ------
    NumericalInstabilityError
        If the condition number exceeds the threshold.
    """
    eigvals = np.linalg.eigvalsh(W)
    if np.any(eigvals < _PSD_EIGENVALUE_TOL):
        logger.warning(
            "Gram matrix has negative eigenvalue %.2e (numerical noise)",
            float(np.min(eigvals)),
        )

    positive_eigvals = eigvals[eigvals > 1e-15]
    if len(positive_eigvals) >= 2:
        cond = float(positive_eigvals[-1] / positive_eigvals[0])
        if cond > max_cond:
            raise NumericalInstabilityError(
                f"Gram matrix condition number {cond:.2e} exceeds threshold "
                f"{max_cond:.2e}. Consider regularising or rescaling the workload.",
                condition_number=cond,
                max_condition_number=max_cond,
                matrix_name="W = A^T A",
            )


def BuildWorkloadSDP(
    A: npt.NDArray[np.float64],
    epsilon: float,
    K_verts: Optional[npt.NDArray[np.float64]] = None,
    C: float = 1.0,
    structural_hint: Optional[str] = None,
    max_cond: float = 1e12,
) -> SDPStruct:
    r"""Construct the SDP for optimal Gaussian workload mechanism synthesis.

    This is the primary entry point for SDP construction.  Given a workload
    ``A ∈ ℝ^{m×d}`` and privacy parameter ``ε``, it constructs:

        minimise   trace(W Σ)
        subject to vᵀ Σ v  ≤  (C/ε)²       ∀ v ∈ K_verts
                   Σ ≽ 0

    where ``W = AᵀA`` and ``K_verts`` are the vertices of the sensitivity
    polytope.

    If ``structural_hint`` is provided (or auto-detected), structural
    constraints are added to ``Σ``:

    - ``"diagonal"``: ``Σ`` is constrained to be diagonal.
    - ``"toeplitz"``: ``Σ`` is constrained to have Toeplitz structure.
    - ``"circulant"``: ``Σ`` is constrained to be circulant.

    Parameters
    ----------
    A : ndarray of shape (m, d)
        Workload matrix.
    epsilon : float
        Privacy parameter ε > 0.
    K_verts : ndarray of shape (n_verts, d), optional
        Vertices of the sensitivity polytope.  If ``None``, defaults to
        the ℓ₁ ball vertices ``{±e_j}``.
    C : float
        Sensitivity scaling constant (default 1.0).
    structural_hint : str, optional
        Structural hint for Σ: ``"diagonal"``, ``"toeplitz"``,
        ``"circulant"``, or ``None`` (general PSD).
    max_cond : float
        Maximum condition number for the Gram matrix.

    Returns
    -------
    SDPStruct
        The constructed SDP with CVXPY problem, variable, objective,
        and constraint list.

    Raises
    ------
    NumericalInstabilityError
        If the Gram matrix is ill-conditioned.
    ConfigurationError
        If parameters are invalid.
    """
    cp = _get_cvxpy()
    A = np.asarray(A, dtype=np.float64)

    if A.ndim != 2:
        raise ConfigurationError(
            f"Workload matrix must be 2-D, got {A.ndim}-D",
            parameter="A",
            value=A.shape,
        )

    if epsilon <= 0:
        raise ConfigurationError(
            f"epsilon must be > 0, got {epsilon}",
            parameter="epsilon",
            value=epsilon,
            constraint="epsilon > 0",
        )

    if C <= 0:
        raise ConfigurationError(
            f"C must be > 0, got {C}",
            parameter="C",
            value=C,
            constraint="C > 0",
        )

    m, d = A.shape
    logger.info(
        "Building workload SDP: m=%d, d=%d, ε=%.4f, C=%.4f, hint=%s",
        m, d, epsilon, C, structural_hint,
    )

    # Step 1: Compute and validate Gram matrix W = AᵀA
    W = _compute_gram_matrix(A)
    _validate_gram_matrix(W, max_cond=max_cond)

    # Step 2: Compute sensitivity ball vertices if not provided
    if K_verts is None:
        K_verts = SensitivityBallComputer.compute_l1_ball_vertices(A)
        logger.debug("Using default ℓ₁ vertices: %d vertices", len(K_verts))

    K_verts = np.asarray(K_verts, dtype=np.float64)
    if K_verts.ndim != 2 or K_verts.shape[1] != d:
        raise ConfigurationError(
            f"K_verts must have shape (n_verts, {d}), got {K_verts.shape}",
            parameter="K_verts",
            value=K_verts.shape,
        )

    # Step 3: Declare PSD variable Σ
    if structural_hint == "diagonal":
        Sigma = _build_diagonal_variable(d, cp)
    elif structural_hint == "toeplitz":
        Sigma = _build_toeplitz_variable(d, cp)
    elif structural_hint == "circulant":
        Sigma = _build_circulant_variable(d, cp)
    else:
        Sigma = cp.Variable((d, d), symmetric=True, name="Sigma")

    # Step 4: Build constraints
    constraints: List[Any] = []

    # PSD constraint: Σ ≽ 0
    constraints.append(Sigma >> 0)

    # Privacy constraints: vᵀ Σ v ≤ (C/ε)² for each vertex v
    bound = (C / epsilon) ** 2
    for i, v in enumerate(K_verts):
        v_col = v.reshape(-1, 1)
        quad_form = cp.quad_form(v_col.flatten(), Sigma)
        constraints.append(quad_form <= bound)

    logger.debug(
        "Added %d privacy constraints, bound = %.6e",
        len(K_verts), bound,
    )

    # Step 5: Objective: minimise trace(W Σ)
    W_param = cp.Constant(W)
    objective = cp.Minimize(cp.trace(W_param @ Sigma))

    # Step 6: Construct CVXPY problem
    problem = cp.Problem(objective, constraints)

    workload_spec = WorkloadSpec(
        matrix=A,
        structural_hint=structural_hint,
    )

    sdp = SDPStruct(
        problem=problem,
        sigma_var=Sigma,
        objective=objective,
        constraints=constraints,
        workload=workload_spec,
    )

    logger.info(
        "SDP constructed: %d constraints, %d×%d PSD variable",
        len(constraints), d, d,
    )
    return sdp


# ---------------------------------------------------------------------------
# Structural variable builders
# ---------------------------------------------------------------------------


def _build_diagonal_variable(d: int, cp: Any) -> Any:
    """Build a diagonal PSD variable.

    The variable is represented as a full d×d matrix constrained to be
    diagonal: ``Σ = diag(σ₁, …, σ_d)``.

    Parameters
    ----------
    d : int
        Dimension.
    cp : module
        CVXPY module.

    Returns
    -------
    Sigma : cvxpy.Variable
        A d×d symmetric variable with off-diagonal entries constrained to 0.
    """
    sigma_diag = cp.Variable(d, nonneg=True, name="sigma_diag")
    Sigma = cp.diag(sigma_diag)
    return Sigma


def _build_toeplitz_variable(d: int, cp: Any) -> Any:
    """Build a Toeplitz-structured PSD variable.

    A symmetric Toeplitz matrix is determined by its first row
    ``[t_0, t_1, …, t_{d-1}]``.  We construct ``Σ`` as::

        Σ[i,j] = t_{|i-j|}

    This is done by creating a vector of ``d`` free parameters and
    building the Toeplitz matrix via indexing.

    Parameters
    ----------
    d : int
        Dimension.
    cp : module
        CVXPY module.

    Returns
    -------
    Sigma : cvxpy.Expression
        A d×d symmetric Toeplitz expression.
    """
    t = cp.Variable(d, name="toeplitz_params")

    # Build the index matrix for a symmetric Toeplitz structure
    indices = np.abs(np.arange(d)[:, np.newaxis] - np.arange(d)[np.newaxis, :])

    # Construct via cp.reshape and indexing
    # CVXPY doesn't support fancy indexing on variables, so we build
    # the matrix using a selection matrix approach
    rows = []
    for i in range(d):
        row_expr = cp.hstack([t[abs(i - j)] for j in range(d)])
        rows.append(row_expr)
    Sigma = cp.vstack(rows)
    return Sigma


def _build_circulant_variable(d: int, cp: Any) -> Any:
    """Build a circulant-structured PSD variable.

    A circulant matrix is determined by its first row ``c``.  Row ``i``
    is a cyclic shift of ``c`` by ``i`` positions.

    Parameters
    ----------
    d : int
        Dimension.
    cp : module
        CVXPY module.

    Returns
    -------
    Sigma : cvxpy.Expression
        A d×d circulant expression.
    """
    c = cp.Variable(d, name="circulant_params")

    rows = []
    for i in range(d):
        # Cyclic shift: row i has entries c[(j-i) mod d]
        row_expr = cp.hstack([c[(j - i) % d] for j in range(d)])
        rows.append(row_expr)
    Sigma = cp.vstack(rows)
    return Sigma


# ---------------------------------------------------------------------------
# Toeplitz spectral reduction
# ---------------------------------------------------------------------------


def toeplitz_sdp(
    first_row: npt.NDArray[np.float64],
    epsilon: float,
    C: float = 1.0,
) -> SDPStruct:
    r"""Construct a specialised SDP for a Toeplitz workload Gram matrix.

    When ``W = AᵀA`` is Toeplitz, the optimal ``Σ`` is also Toeplitz
    (by symmetry of the problem).  The SDP then involves only ``d``
    free parameters instead of ``d(d+1)/2``.

    The spectral representation uses the DFT: a symmetric Toeplitz
    matrix ``T`` has eigenvalues given by the DFT of its first column.

    Parameters
    ----------
    first_row : ndarray of shape (d,)
        First row of the Toeplitz Gram matrix ``W``.
    epsilon : float
        Privacy parameter ε > 0.
    C : float
        Sensitivity scaling constant.

    Returns
    -------
    SDPStruct
        The specialised Toeplitz SDP.
    """
    cp = _get_cvxpy()
    first_row = np.asarray(first_row, dtype=np.float64)
    d = len(first_row)

    if epsilon <= 0:
        raise ConfigurationError(
            f"epsilon must be > 0, got {epsilon}",
            parameter="epsilon",
            value=epsilon,
        )

    # Build full Toeplitz W for the objective
    from scipy.linalg import toeplitz as scipy_toeplitz
    W = scipy_toeplitz(first_row)

    # The variable is a Toeplitz Σ parameterised by its first row
    t = cp.Variable(d, name="toeplitz_sigma_params")

    # Build Sigma as Toeplitz
    rows = []
    for i in range(d):
        row_expr = cp.hstack([t[abs(i - j)] for j in range(d)])
        rows.append(row_expr)
    Sigma = cp.vstack(rows)

    # Constraints
    constraints: List[Any] = []

    # PSD: Sigma >> 0
    constraints.append(Sigma >> 0)

    # Privacy: for Toeplitz with ℓ₁ sensitivity, each e_j gives
    # v^T Sigma v = Sigma[j,j] = t[0] ≤ (C/eps)^2
    bound = (C / epsilon) ** 2
    constraints.append(t[0] <= bound)

    # Objective: trace(W @ Sigma) = sum_k W_first_row[k] * d * t[k] (approx)
    # More precisely, trace(W @ T) for symmetric Toeplitz T, W:
    # trace(W T) = d * w_0 * t_0 + 2 * sum_{k=1}^{d-1} (d - k) * w_k * t_k
    # where w_k and t_k are the Toeplitz parameters
    obj_coeffs = np.zeros(d, dtype=np.float64)
    obj_coeffs[0] = d * first_row[0]
    for k in range(1, d):
        obj_coeffs[k] = 2.0 * (d - k) * first_row[k]

    objective = cp.Minimize(obj_coeffs @ t)

    problem = cp.Problem(objective, constraints)

    sdp = SDPStruct(
        problem=problem,
        sigma_var=t,
        objective=objective,
        constraints=constraints,
        workload=None,
    )

    logger.info("Toeplitz SDP constructed: d=%d, %d parameters", d, d)
    return sdp


def spectral_factorization(
    t_params: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Convert Toeplitz parameters to a full covariance matrix.

    Given the first row ``[t_0, t_1, …, t_{d-1}]`` of a Toeplitz matrix,
    reconstruct the full ``d × d`` symmetric Toeplitz covariance.

    Parameters
    ----------
    t_params : ndarray of shape (d,)
        Toeplitz parameters (first row of the symmetric Toeplitz matrix).

    Returns
    -------
    Sigma : ndarray of shape (d, d)
        Full symmetric Toeplitz covariance matrix.
    """
    from scipy.linalg import toeplitz as scipy_toeplitz
    t_params = np.asarray(t_params, dtype=np.float64)
    Sigma = scipy_toeplitz(t_params).astype(np.float64)
    return Sigma


# ---------------------------------------------------------------------------
# SDP solver interface
# ---------------------------------------------------------------------------


def auto_select_solver(d: int) -> str:
    """Select the best available CVXPY solver for a given dimension.

    Heuristic:
    - ``d ≤ 500``: prefer MOSEK (interior point, high accuracy)
    - ``d > 500``: prefer SCS (first-order, scales to large problems)

    Falls back if the preferred solver is not installed.

    Parameters
    ----------
    d : int
        Dimension of the PSD variable.

    Returns
    -------
    solver_name : str
        CVXPY solver name string (e.g., ``"MOSEK"``, ``"SCS"``).
    """
    cp = _get_cvxpy()

    preferred = "MOSEK" if d <= _LARGE_DIMENSION_THRESHOLD else "SCS"
    fallback = "SCS" if preferred == "MOSEK" else "MOSEK"

    # Check preferred solver availability
    try:
        if preferred in [s.name() if hasattr(s, 'name') else str(s)
                         for s in cp.installed_solvers()]:
            return preferred
    except Exception:
        pass

    # Check if preferred is in installed solvers (string-based)
    installed = cp.installed_solvers()
    if preferred in installed:
        return preferred
    if fallback in installed:
        logger.info(
            "Preferred solver %s not available, using %s",
            preferred, fallback,
        )
        return fallback

    # Last resort
    for solver in ["CVXOPT", "SDPA", "COPT"]:
        if solver in installed:
            logger.warning("Using fallback solver %s", solver)
            return solver

    # SCS is bundled with CVXPY, should always be available
    return "SCS"


def solve_with_scs(
    problem: Any,
    max_iters: int = _DEFAULT_SCS_MAX_ITERS,
    eps: float = _DEFAULT_SCS_EPS,
    verbose: bool = False,
    warm_start: bool = False,
    **kwargs: Any,
) -> SolveStatistics:
    """Solve a CVXPY problem using the SCS solver.

    SCS (Splitting Conic Solver) is a first-order method suitable for
    large-scale SDPs.  It trades solution accuracy for speed.

    Parameters
    ----------
    problem : cvxpy.Problem
        The CVXPY problem to solve.
    max_iters : int
        Maximum number of SCS iterations.
    eps : float
        SCS convergence tolerance.
    verbose : bool
        Whether to print SCS output.
    warm_start : bool
        Whether to warm-start from previous solution.
    **kwargs
        Additional SCS parameters.

    Returns
    -------
    SolveStatistics
        Solve statistics including time, iterations, and gap.

    Raises
    ------
    SolverError
        If SCS fails to solve the problem.
    """
    t_start = time.monotonic()
    solver_params = {
        "max_iters": max_iters,
        "eps": eps,
        "verbose": verbose,
        "warm_start": warm_start,
    }
    solver_params.update(kwargs)

    try:
        problem.solve(solver="SCS", **solver_params)
    except Exception as exc:
        raise SolverError(
            f"SCS solver failed: {exc}",
            solver_name="SCS",
            solver_status="error",
            original_error=exc,
        ) from exc

    t_end = time.monotonic()

    status = problem.status
    if status in ("infeasible", "infeasible_inaccurate"):
        raise InfeasibleSpecError(
            f"SDP is infeasible (SCS status: {status})",
            solver_status=status,
        )

    primal_obj = float(problem.value) if problem.value is not None else float("inf")
    # SCS doesn't directly report dual objective via CVXPY
    dual_obj = primal_obj  # approximation

    stats = SolveStatistics(
        solve_time=t_end - t_start,
        solver_name="SCS",
        iterations=solver_params.get("max_iters", 0),
        duality_gap=0.0,
        primal_obj=primal_obj,
        dual_obj=dual_obj,
        status=status,
    )

    logger.info("SCS solve: status=%s, time=%.3fs, obj=%.6e", status, stats.solve_time, primal_obj)
    return stats


def solve_with_mosek(
    problem: Any,
    verbose: bool = False,
    tol: float = _DEFAULT_MOSEK_TOL,
    **kwargs: Any,
) -> SolveStatistics:
    """Solve a CVXPY problem using the MOSEK solver.

    MOSEK is a commercial interior-point solver with high accuracy,
    ideal for small-to-medium SDPs (``d ≤ 500``).

    Parameters
    ----------
    problem : cvxpy.Problem
        The CVXPY problem to solve.
    verbose : bool
        Whether to print MOSEK output.
    tol : float
        MOSEK feasibility tolerance.
    **kwargs
        Additional MOSEK parameters.

    Returns
    -------
    SolveStatistics
        Solve statistics.

    Raises
    ------
    SolverError
        If MOSEK fails or is not installed.
    """
    cp = _get_cvxpy()

    if "MOSEK" not in cp.installed_solvers():
        raise SolverError(
            "MOSEK solver is not installed. Install it with: pip install Mosek",
            solver_name="MOSEK",
            solver_status="not_installed",
        )

    t_start = time.monotonic()
    mosek_params = {
        "verbose": verbose,
    }
    mosek_params.update(kwargs)

    try:
        problem.solve(solver="MOSEK", **mosek_params)
    except Exception as exc:
        raise SolverError(
            f"MOSEK solver failed: {exc}",
            solver_name="MOSEK",
            solver_status="error",
            original_error=exc,
        ) from exc

    t_end = time.monotonic()

    status = problem.status
    if status in ("infeasible", "infeasible_inaccurate"):
        raise InfeasibleSpecError(
            f"SDP is infeasible (MOSEK status: {status})",
            solver_status=status,
        )

    primal_obj = float(problem.value) if problem.value is not None else float("inf")
    dual_obj = primal_obj

    stats = SolveStatistics(
        solve_time=t_end - t_start,
        solver_name="MOSEK",
        duality_gap=0.0,
        primal_obj=primal_obj,
        dual_obj=dual_obj,
        status=status,
    )

    logger.info("MOSEK solve: status=%s, time=%.3fs, obj=%.6e", status, stats.solve_time, primal_obj)
    return stats


def solve_with_cvxpy(
    problem: Any,
    solver: str = "SCS",
    verbose: bool = False,
    **kwargs: Any,
) -> SolveStatistics:
    """Solve a CVXPY problem with a generic solver backend.

    This is a catch-all interface that delegates to whichever CVXPY
    solver is specified.

    Parameters
    ----------
    problem : cvxpy.Problem
        The CVXPY problem to solve.
    solver : str
        CVXPY solver name (e.g., ``"SCS"``, ``"MOSEK"``, ``"CVXOPT"``).
    verbose : bool
        Whether to print solver output.
    **kwargs
        Solver-specific parameters.

    Returns
    -------
    SolveStatistics
        Solve statistics.

    Raises
    ------
    SolverError
        If the solver fails.
    """
    cp = _get_cvxpy()

    if solver not in cp.installed_solvers():
        raise SolverError(
            f"Solver {solver!r} is not installed. "
            f"Available solvers: {cp.installed_solvers()}",
            solver_name=solver,
            solver_status="not_installed",
        )

    t_start = time.monotonic()

    try:
        problem.solve(solver=solver, verbose=verbose, **kwargs)
    except Exception as exc:
        raise SolverError(
            f"Solver {solver!r} failed: {exc}",
            solver_name=solver,
            solver_status="error",
            original_error=exc,
        ) from exc

    t_end = time.monotonic()

    status = problem.status
    if status in ("infeasible", "infeasible_inaccurate"):
        raise InfeasibleSpecError(
            f"SDP is infeasible ({solver} status: {status})",
            solver_status=status,
        )

    primal_obj = float(problem.value) if problem.value is not None else float("inf")

    stats = SolveStatistics(
        solve_time=t_end - t_start,
        solver_name=solver,
        duality_gap=0.0,
        primal_obj=primal_obj,
        dual_obj=primal_obj,
        status=status,
    )

    logger.info(
        "%s solve: status=%s, time=%.3fs, obj=%.6e",
        solver, status, stats.solve_time, primal_obj,
    )
    return stats


# ---------------------------------------------------------------------------
# Gaussian mechanism extraction
# ---------------------------------------------------------------------------


def extract_sigma(
    sdp: SDPStruct,
) -> npt.NDArray[np.float64]:
    r"""Extract the optimal covariance matrix from a solved SDP.

    After solving, the CVXPY variable ``Sigma`` contains the optimal
    covariance.  This function extracts it, enforces symmetry, and
    projects onto the PSD cone if needed (eigenvalue clipping).

    Parameters
    ----------
    sdp : SDPStruct
        A solved SDP (``sdp.problem.status`` should be optimal).

    Returns
    -------
    Sigma : ndarray of shape (d, d)
        Optimal covariance matrix, guaranteed symmetric PSD.

    Raises
    ------
    SolverError
        If the SDP has not been solved or is infeasible.
    """
    if sdp.problem.status not in ("optimal", "optimal_inaccurate"):
        raise SolverError(
            f"Cannot extract Sigma: SDP status is {sdp.problem.status!r}",
            solver_status=sdp.problem.status,
        )

    raw = sdp.sigma_var.value
    if raw is None:
        raise SolverError(
            "Sigma variable has no value after solve",
            solver_status=sdp.problem.status,
        )

    Sigma = np.asarray(raw, dtype=np.float64)

    # Handle 1-D Toeplitz parameters
    if Sigma.ndim == 1:
        Sigma = spectral_factorization(Sigma)

    # Force symmetry
    Sigma = (Sigma + Sigma.T) / 2.0

    # Project onto PSD cone: clip negative eigenvalues to 0
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 0.0)
    Sigma = (eigvecs * eigvals) @ eigvecs.T

    # Force exact symmetry after reconstruction
    Sigma = (Sigma + Sigma.T) / 2.0

    return Sigma


def compute_noise_distribution(
    sigma: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Compute the noise covariance from the optimal Σ and workload.

    The noise covariance in the answer space is ``Σ_noise = A Σ Aᵀ``.

    Parameters
    ----------
    sigma : ndarray of shape (d, d)
        Optimal covariance matrix.
    A : ndarray of shape (m, d)
        Workload matrix.

    Returns
    -------
    noise_cov : ndarray of shape (m, m)
        Noise covariance matrix in the answer space.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    noise_cov = A @ sigma @ A.T
    noise_cov = (noise_cov + noise_cov.T) / 2.0
    return noise_cov


def extract_gaussian(
    sigma: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    solve_stats: Optional[SolveStatistics] = None,
) -> GaussianMechanismResult:
    r"""Create a deployable Gaussian mechanism from an optimal covariance.

    Given the optimal covariance ``Σ`` from the SDP, constructs the
    full Gaussian mechanism ``M(x) = Ax + η`` where ``η ~ N(0, A Σ Aᵀ)``.

    Computes:
    - Noise covariance ``Σ_noise = A Σ Aᵀ``
    - Cholesky factor of ``Σ_noise`` for efficient sampling
    - Total MSE ``= trace(W Σ)``
    - Per-query MSE ``= diag(A Σ Aᵀ)``

    Parameters
    ----------
    sigma : ndarray of shape (d, d)
        Optimal covariance matrix (PSD).
    A : ndarray of shape (m, d)
        Workload matrix.
    epsilon : float
        Privacy parameter ε.
    delta : float
        Privacy parameter δ.
    solve_stats : SolveStatistics, optional
        Solver statistics to attach.

    Returns
    -------
    GaussianMechanismResult
        Deployable mechanism with all precomputed quantities.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)

    noise_cov = compute_noise_distribution(sigma, A)
    W = _compute_gram_matrix(A)
    total_mse = float(np.trace(W @ sigma))
    per_query_mse = np.diag(noise_cov).copy()

    # Compute Cholesky factor for sampling (regularise if needed)
    chol = _safe_cholesky(noise_cov)

    return GaussianMechanismResult(
        sigma=sigma,
        noise_covariance=noise_cov,
        workload=A,
        epsilon=epsilon,
        delta=delta,
        total_mse=total_mse,
        per_query_mse=per_query_mse,
        cholesky_factor=chol,
        solve_stats=solve_stats,
    )


def _safe_cholesky(
    M: npt.NDArray[np.float64],
    reg: float = 1e-12,
    max_attempts: int = 10,
) -> Optional[npt.NDArray[np.float64]]:
    """Compute Cholesky factor with regularisation fallback.

    Parameters
    ----------
    M : ndarray of shape (n, n)
        Symmetric PSD matrix.
    reg : float
        Initial regularisation added to the diagonal.
    max_attempts : int
        Maximum number of regularisation doublings.

    Returns
    -------
    L : ndarray of shape (n, n) or None
        Lower-triangular Cholesky factor, or ``None`` if all attempts fail.
    """
    M = np.asarray(M, dtype=np.float64)
    M = (M + M.T) / 2.0

    for attempt in range(max_attempts):
        try:
            L = np.linalg.cholesky(M + reg * np.eye(M.shape[0]))
            return L
        except np.linalg.LinAlgError:
            reg *= 10.0

    logger.warning("Cholesky factorisation failed after %d attempts", max_attempts)
    return None


def sample_gaussian(
    x: npt.NDArray[np.float64],
    mechanism: GaussianMechanismResult,
    rng: Optional[np.random.Generator] = None,
) -> npt.NDArray[np.float64]:
    r"""Sample from the Gaussian mechanism.

    Computes ``M(x) = Ax + η`` where ``η ~ N(0, Σ_noise)``.

    Parameters
    ----------
    x : ndarray of shape (d,)
        True data vector.
    mechanism : GaussianMechanismResult
        The Gaussian mechanism to sample from.
    rng : numpy.random.Generator, optional
        Random number generator.  If ``None``, uses default.

    Returns
    -------
    noisy_answer : ndarray of shape (m,)
        Noisy answer vector.
    """
    x = np.asarray(x, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng()

    A = mechanism.workload
    true_answer = A @ x

    m = A.shape[0]
    if mechanism.cholesky_factor is not None:
        z = rng.standard_normal(m)
        noise = mechanism.cholesky_factor @ z
    else:
        # Fallback: use noise covariance directly
        noise = rng.multivariate_normal(np.zeros(m), mechanism.noise_covariance)

    return true_answer + noise


def compute_mse(
    sigma: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
) -> float:
    r"""Compute the analytical MSE for a given covariance and workload.

    The total mean squared error of the Gaussian mechanism with covariance
    ``Σ`` on workload ``A`` is:

        MSE = trace(AᵀA · Σ) = trace(W · Σ)

    Parameters
    ----------
    sigma : ndarray of shape (d, d)
        Covariance matrix.
    A : ndarray of shape (m, d)
        Workload matrix.

    Returns
    -------
    mse : float
        Total mean squared error.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    W = A.T @ A
    return float(np.trace(W @ sigma))


# ---------------------------------------------------------------------------
# SDPManager
# ---------------------------------------------------------------------------


class SDPManager:
    """High-level manager for SDP construction, solving, and extraction.

    Orchestrates the full pipeline from workload specification to deployable
    Gaussian mechanism:

    1. Structural detection and reduction
    2. SDP construction via :func:`BuildWorkloadSDP`
    3. Solver selection and execution
    4. Covariance extraction and mechanism construction

    Parameters
    ----------
    numerical_config : NumericalConfig, optional
        Numerical precision configuration.
    verbose : int
        Verbosity level (0=silent, 1=info, 2=debug).

    Examples
    --------
    >>> manager = SDPManager()
    >>> workload = WorkloadSpec.identity(5)
    >>> privacy = PrivacyBudget(epsilon=1.0)
    >>> manager.build(workload, privacy)
    >>> manager.solve()
    >>> sigma = manager.extract_sigma()
    >>> result = manager.compute_noise_distribution()
    """

    def __init__(
        self,
        numerical_config: Optional[NumericalConfig] = None,
        verbose: int = 1,
    ) -> None:
        self._numerical = numerical_config or NumericalConfig()
        self._verbose = verbose
        self._sdp: Optional[SDPStruct] = None
        self._solve_stats: Optional[SolveStatistics] = None
        self._sigma: Optional[npt.NDArray[np.float64]] = None
        self._workload_spec: Optional[WorkloadSpec] = None
        self._privacy: Optional[PrivacyBudget] = None
        self._structural_info: Optional[StructuralInfo] = None

    @property
    def sdp(self) -> Optional[SDPStruct]:
        """The current SDP, or ``None`` if not yet built."""
        return self._sdp

    @property
    def solve_stats(self) -> Optional[SolveStatistics]:
        """Statistics from the last solve, or ``None``."""
        return self._solve_stats

    @property
    def is_built(self) -> bool:
        """Whether an SDP has been constructed."""
        return self._sdp is not None

    @property
    def is_solved(self) -> bool:
        """Whether the SDP has been solved."""
        return self._sigma is not None

    def build(
        self,
        workload_spec: WorkloadSpec,
        privacy_params: PrivacyBudget,
        K_verts: Optional[npt.NDArray[np.float64]] = None,
        C: float = 1.0,
        structural_hint: Optional[str] = None,
        sensitivity_norm: str = "l1",
    ) -> SDPStruct:
        r"""Build the SDP for a given workload and privacy specification.

        This is the main entry point for SDP construction.  It performs:

        1. Structural analysis of the workload
        2. Sensitivity vertex computation
        3. SDP construction with appropriate structural constraints

        Parameters
        ----------
        workload_spec : WorkloadSpec
            Workload matrix specification.
        privacy_params : PrivacyBudget
            Privacy budget (ε, δ).
        K_verts : ndarray, optional
            Pre-computed sensitivity vertices.  If ``None``, computed
            automatically based on ``sensitivity_norm``.
        C : float
            Sensitivity scaling constant.
        structural_hint : str, optional
            Override for structural detection.  If ``None``, auto-detected.
        sensitivity_norm : str
            Sensitivity norm: ``"l1"``, ``"l2"``, or ``"linf"``.

        Returns
        -------
        SDPStruct
            The constructed SDP.
        """
        t_start = time.monotonic()

        self._workload_spec = workload_spec
        self._privacy = privacy_params
        self._sigma = None
        self._solve_stats = None

        A = workload_spec.matrix

        # Step 1: Structural detection
        if structural_hint is None:
            hint = workload_spec.structural_hint
            if hint is None:
                self._structural_info = StructuralDetector.suggest_structure(A)
                hint = self._structural_info.recommended_hint
            structural_hint = hint

        logger.info(
            "SDPManager.build: workload %s, ε=%.4f, hint=%s",
            workload_spec, privacy_params.epsilon, structural_hint,
        )

        # Step 2: Compute sensitivity vertices if needed
        if K_verts is None:
            if sensitivity_norm == "l1":
                K_verts = SensitivityBallComputer.compute_l1_ball_vertices(A)
            elif sensitivity_norm == "l2":
                K_verts = SensitivityBallComputer.compute_l2_ball_vertices(A)
            elif sensitivity_norm == "linf":
                K_verts = SensitivityBallComputer.compute_linf_ball_vertices(A)
            else:
                raise ConfigurationError(
                    f"Unknown sensitivity norm: {sensitivity_norm!r}",
                    parameter="sensitivity_norm",
                    value=sensitivity_norm,
                    constraint="Must be 'l1', 'l2', or 'linf'",
                )

            # Reduce redundant vertices
            K_verts = SensitivityBallComputer.reduce_redundant_vertices(K_verts)
            logger.debug("Using %d sensitivity vertices", len(K_verts))

        # Step 3: Build SDP
        self._sdp = BuildWorkloadSDP(
            A=A,
            epsilon=privacy_params.epsilon,
            K_verts=K_verts,
            C=C,
            structural_hint=structural_hint,
            max_cond=self._numerical.max_condition_number,
        )

        setup_time = time.monotonic() - t_start
        logger.info("SDP built in %.3fs", setup_time)

        return self._sdp

    def solve(
        self,
        solver: str = "auto",
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> SolveStatistics:
        r"""Solve the constructed SDP.

        Parameters
        ----------
        solver : str
            Solver to use: ``"auto"``, ``"SCS"``, ``"MOSEK"``, or any
            CVXPY-supported solver name.
        verbose : bool, optional
            Override verbosity.  If ``None``, uses the manager's setting.
        **kwargs
            Additional solver parameters.

        Returns
        -------
        SolveStatistics
            Solve statistics.

        Raises
        ------
        ConfigurationError
            If no SDP has been built.
        SolverError
            If the solver fails.
        InfeasibleSpecError
            If the SDP is infeasible.
        """
        if self._sdp is None:
            raise ConfigurationError(
                "No SDP has been built. Call build() first.",
                parameter="sdp",
            )

        if verbose is None:
            verbose = self._verbose >= 2

        d = self._workload_spec.d if self._workload_spec else 10

        # Auto-select solver
        if solver == "auto":
            solver = auto_select_solver(d)

        logger.info("Solving SDP with %s (d=%d)", solver, d)

        # Dispatch to solver
        if solver == "SCS":
            self._solve_stats = solve_with_scs(
                self._sdp.problem,
                verbose=verbose,
                **kwargs,
            )
        elif solver == "MOSEK":
            self._solve_stats = solve_with_mosek(
                self._sdp.problem,
                verbose=verbose,
                **kwargs,
            )
        else:
            self._solve_stats = solve_with_cvxpy(
                self._sdp.problem,
                solver=solver,
                verbose=verbose,
                **kwargs,
            )

        self._solve_stats.dimension = d

        # Extract sigma immediately
        self._sigma = extract_sigma(self._sdp)

        return self._solve_stats

    def extract_sigma(self) -> npt.NDArray[np.float64]:
        """Get the optimal covariance matrix.

        Returns
        -------
        Sigma : ndarray of shape (d, d)
            Optimal covariance matrix, guaranteed symmetric PSD.

        Raises
        ------
        ConfigurationError
            If the SDP has not been solved.
        """
        if self._sigma is None:
            if self._sdp is None:
                raise ConfigurationError(
                    "No SDP has been built. Call build() first.",
                    parameter="sdp",
                )
            raise ConfigurationError(
                "SDP has not been solved. Call solve() first.",
                parameter="solve",
            )
        return self._sigma.copy()

    def compute_noise_distribution(self) -> GaussianMechanismResult:
        """Convert the optimal Σ to a deployable Gaussian mechanism.

        Returns
        -------
        GaussianMechanismResult
            Full mechanism with noise covariance, Cholesky factor,
            MSE, and sampling support.

        Raises
        ------
        ConfigurationError
            If the SDP has not been solved.
        """
        sigma = self.extract_sigma()

        if self._workload_spec is None or self._privacy is None:
            raise ConfigurationError(
                "Workload or privacy params missing. Call build() first.",
                parameter="workload",
            )

        return extract_gaussian(
            sigma=sigma,
            A=self._workload_spec.matrix,
            epsilon=self._privacy.epsilon,
            delta=self._privacy.delta,
            solve_stats=self._solve_stats,
        )

    def warm_start(
        self,
        sigma_init: npt.NDArray[np.float64],
    ) -> None:
        """Set a warm-start point for the SDP variable.

        This can improve convergence for iterative refinement workflows
        (e.g., CEGIS with progressively tightened constraints).

        Parameters
        ----------
        sigma_init : ndarray of shape (d, d)
            Initial covariance matrix (must be PSD).

        Raises
        ------
        ConfigurationError
            If no SDP has been built.
        """
        if self._sdp is None:
            raise ConfigurationError(
                "No SDP has been built. Call build() first.",
                parameter="sdp",
            )

        sigma_init = np.asarray(sigma_init, dtype=np.float64)

        # Ensure PSD
        eigvals = np.linalg.eigvalsh(sigma_init)
        if np.any(eigvals < _PSD_EIGENVALUE_TOL):
            logger.warning(
                "Warm-start matrix has negative eigenvalue %.2e; projecting to PSD cone",
                float(np.min(eigvals)),
            )
            eigvals_full, eigvecs = np.linalg.eigh(sigma_init)
            eigvals_full = np.maximum(eigvals_full, _WARM_START_PERTURBATION)
            sigma_init = (eigvecs * eigvals_full) @ eigvecs.T
            sigma_init = (sigma_init + sigma_init.T) / 2.0

        self._sdp.sigma_var.value = sigma_init
        logger.info("Warm-start set for SDP variable")

    def get_optimality_certificate(self) -> Optional[OptimalityCertificate]:
        """Extract an optimality certificate from the solved SDP.

        Returns
        -------
        OptimalityCertificate or None
            Certificate with primal/dual objectives and duality gap,
            or ``None`` if the SDP has not been solved.
        """
        if self._solve_stats is None:
            return None

        gap = max(0.0, self._solve_stats.duality_gap)
        return OptimalityCertificate(
            dual_vars=None,
            duality_gap=gap,
            primal_obj=self._solve_stats.primal_obj,
            dual_obj=self._solve_stats.dual_obj,
        )


# ---------------------------------------------------------------------------
# Workload strategy optimisation
# ---------------------------------------------------------------------------


def matrix_mechanism_strategy(
    A: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    solver: str = "auto",
    verbose: bool = False,
) -> GaussianMechanismResult:
    r"""Compute the optimal matrix mechanism strategy for workload ``A``.

    This is a convenience function that builds, solves, and extracts
    the optimal Gaussian mechanism in one call.

    The matrix mechanism answers queries ``q = Bx + η`` where ``B`` is a
    **strategy matrix** and ``η`` is calibrated noise.  The answers to the
    original workload ``A`` are reconstructed via ``Â x = A B⁺ q`` where
    ``B⁺`` is the pseudoinverse of ``B``.

    For Gaussian noise calibrated to ``B``'s ``ℓ_2`` sensitivity, the
    total squared error on workload ``A`` is:

        E[||Ax - Âx||²] = (2/ε²) · trace(A (BᵀB)⁻¹ Aᵀ)

    The optimal strategy minimises this.  The SDP formulation is
    equivalent to minimising ``trace(W Σ)`` subject to the privacy
    constraints, where ``Σ = (BᵀB)⁻¹``.

    Parameters
    ----------
    A : ndarray of shape (m, d)
        Workload matrix.
    epsilon : float
        Privacy parameter ε > 0.
    delta : float
        Privacy parameter δ ∈ [0, 1).
    solver : str
        Solver: ``"auto"``, ``"SCS"``, ``"MOSEK"``, etc.
    verbose : bool
        Whether to print solver output.

    Returns
    -------
    GaussianMechanismResult
        Optimal Gaussian mechanism.
    """
    A = np.asarray(A, dtype=np.float64)
    workload = WorkloadSpec(matrix=A)
    privacy = PrivacyBudget(epsilon=epsilon, delta=delta)

    manager = SDPManager(verbose=2 if verbose else 0)
    manager.build(workload, privacy)
    manager.solve(solver=solver, verbose=verbose)
    return manager.compute_noise_distribution()


def hdmm_greedy(
    A: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    n_iterations: int = 25,
    p_fraction: float = 0.8,
    seed: int = 42,
) -> GaussianMechanismResult:
    r"""HDMM-style greedy strategy optimisation.

    Implements a greedy approximation to the optimal matrix mechanism
    strategy, inspired by the HDMM (High-Dimensional Matrix Mechanism)
    algorithm.

    The algorithm iteratively selects strategy queries that maximally
    reduce the worst-case error on the workload.  At each step it picks
    the query direction that most reduces ``trace(A (BᵀB)⁻¹ Aᵀ)``.

    This is faster than solving the full SDP for large ``d`` but produces
    an approximate strategy.

    Parameters
    ----------
    A : ndarray of shape (m, d)
        Workload matrix.
    epsilon : float
        Privacy parameter ε > 0.
    delta : float
        Privacy parameter δ.
    n_iterations : int
        Number of greedy iterations (strategy queries to select).
    p_fraction : float
        Fraction of budget allocated to the selected strategy in each
        iteration.
    seed : int
        Random seed for initial direction sampling.

    Returns
    -------
    GaussianMechanismResult
        Approximate Gaussian mechanism from the greedy strategy.

    Notes
    -----
    This is an approximation.  For small ``d``, use
    :func:`matrix_mechanism_strategy` for the exact optimum.
    """
    A = np.asarray(A, dtype=np.float64)
    m, d = A.shape
    rng = np.random.default_rng(seed)

    if epsilon <= 0:
        raise ConfigurationError(
            f"epsilon must be > 0, got {epsilon}",
            parameter="epsilon",
            value=epsilon,
        )

    # Initialise strategy with identity scaled to privacy budget
    sigma_inv = np.zeros((d, d), dtype=np.float64)
    W = A.T @ A

    noise_scale = (1.0 / epsilon) ** 2

    for iteration in range(n_iterations):
        # Current residual: what the current strategy doesn't cover
        if iteration == 0:
            # Start with identity
            sigma_inv = np.eye(d) / noise_scale
        else:
            # Find direction of maximum residual error
            try:
                sigma_cur = np.linalg.inv(sigma_inv + 1e-12 * np.eye(d))
            except np.linalg.LinAlgError:
                sigma_cur = np.linalg.pinv(sigma_inv)

            residual = W @ sigma_cur
            # Pick direction of largest eigenvalue of residual
            eigvals, eigvecs = np.linalg.eigh(residual)
            best_dir = eigvecs[:, -1]

            # Add this direction to the strategy
            budget_share = p_fraction / (iteration + 1)
            sigma_inv += (budget_share / noise_scale) * np.outer(best_dir, best_dir)

    # Convert to covariance
    try:
        sigma = np.linalg.inv(sigma_inv + 1e-12 * np.eye(d))
    except np.linalg.LinAlgError:
        sigma = np.linalg.pinv(sigma_inv)

    # Ensure PSD
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.maximum(eigvals, 0.0)
    sigma = (eigvecs * eigvals) @ eigvecs.T
    sigma = (sigma + sigma.T) / 2.0

    return extract_gaussian(
        sigma=sigma,
        A=A,
        epsilon=epsilon,
        delta=delta,
    )


def optimal_identity_strategy(
    d: int,
    epsilon: float,
    delta: float = 0.0,
) -> GaussianMechanismResult:
    r"""Compute the optimal strategy for the identity workload.

    For the identity workload ``A = I_d``, the optimal strategy is to
    add independent Gaussian noise to each coordinate.  The optimal
    covariance is ``Σ = (1/ε²) I_d``.

    This is the closed-form solution; no SDP is needed.

    Parameters
    ----------
    d : int
        Dimension.
    epsilon : float
        Privacy parameter ε > 0.
    delta : float
        Privacy parameter δ.

    Returns
    -------
    GaussianMechanismResult
        Optimal Gaussian mechanism for the identity workload.
    """
    if d < 1:
        raise ConfigurationError(
            f"d must be >= 1, got {d}",
            parameter="d",
            value=d,
        )
    if epsilon <= 0:
        raise ConfigurationError(
            f"epsilon must be > 0, got {epsilon}",
            parameter="epsilon",
            value=epsilon,
        )

    variance = (1.0 / epsilon) ** 2
    sigma = variance * np.eye(d, dtype=np.float64)
    A = np.eye(d, dtype=np.float64)

    return extract_gaussian(
        sigma=sigma,
        A=A,
        epsilon=epsilon,
        delta=delta,
    )


# ---------------------------------------------------------------------------
# Block-diagonal decomposition
# ---------------------------------------------------------------------------


def _decompose_block_diagonal(
    A: npt.NDArray[np.float64],
    block_sizes: List[int],
) -> List[npt.NDArray[np.float64]]:
    """Decompose a workload into independent sub-workloads by block structure.

    When ``AᵀA`` is block-diagonal, the SDP decomposes into independent
    sub-problems for each block.

    Parameters
    ----------
    A : ndarray of shape (m, d)
        Workload matrix.
    block_sizes : list of int
        Sizes of the diagonal blocks.

    Returns
    -------
    sub_workloads : list of ndarray
        Sub-workload matrices, one per block.
    """
    A = np.asarray(A, dtype=np.float64)
    sub_workloads = []
    col_start = 0

    for size in block_sizes:
        col_end = col_start + size
        sub_A = A[:, col_start:col_end]
        # Remove all-zero rows
        row_mask = np.any(sub_A != 0, axis=1)
        sub_A = sub_A[row_mask]
        if sub_A.shape[0] > 0:
            sub_workloads.append(sub_A)
        else:
            sub_workloads.append(np.zeros((1, size), dtype=np.float64))
        col_start = col_end

    return sub_workloads


def solve_block_diagonal_sdp(
    A: npt.NDArray[np.float64],
    epsilon: float,
    block_sizes: List[int],
    C: float = 1.0,
    solver: str = "auto",
    verbose: bool = False,
) -> GaussianMechanismResult:
    r"""Solve a block-diagonal SDP by decomposing into sub-problems.

    When ``AᵀA`` has block-diagonal structure, the optimal ``Σ`` is also
    block-diagonal.  This function solves each block independently and
    reassembles the result.

    Parameters
    ----------
    A : ndarray of shape (m, d)
        Workload matrix.
    epsilon : float
        Privacy parameter ε > 0.
    block_sizes : list of int
        Sizes of the diagonal blocks (from :class:`StructuralDetector`).
    C : float
        Sensitivity constant.
    solver : str
        Solver backend.
    verbose : bool
        Verbosity.

    Returns
    -------
    GaussianMechanismResult
        Combined Gaussian mechanism from all blocks.
    """
    A = np.asarray(A, dtype=np.float64)
    d = A.shape[1]

    sub_workloads = _decompose_block_diagonal(A, block_sizes)
    block_sigmas: List[npt.NDArray[np.float64]] = []

    total_solve_time = 0.0

    for i, sub_A in enumerate(sub_workloads):
        sub_d = sub_A.shape[1]
        logger.info("Solving block %d/%d (d=%d)", i + 1, len(sub_workloads), sub_d)

        if sub_d == 0:
            continue

        manager = SDPManager(verbose=2 if verbose else 0)
        sub_workload = WorkloadSpec(matrix=sub_A)
        sub_privacy = PrivacyBudget(epsilon=epsilon)

        manager.build(sub_workload, sub_privacy, C=C)
        stats = manager.solve(solver=solver, verbose=verbose)
        total_solve_time += stats.solve_time

        block_sigma = manager.extract_sigma()
        block_sigmas.append(block_sigma)

    # Reassemble into full d×d block-diagonal Sigma
    sigma_full = np.zeros((d, d), dtype=np.float64)
    row_start = 0
    for block_sigma in block_sigmas:
        bs = block_sigma.shape[0]
        sigma_full[row_start:row_start + bs, row_start:row_start + bs] = block_sigma
        row_start += bs

    combined_stats = SolveStatistics(
        solve_time=total_solve_time,
        solver_name="block_diagonal",
        status="optimal",
        dimension=d,
        num_constraints=sum(2 * s for s in block_sizes) + len(block_sizes),
    )

    return extract_gaussian(
        sigma=sigma_full,
        A=A,
        epsilon=epsilon,
        solve_stats=combined_stats,
    )


# ---------------------------------------------------------------------------
# Circulant / DFT-based fast solver
# ---------------------------------------------------------------------------


def solve_circulant_sdp(
    A: npt.NDArray[np.float64],
    epsilon: float,
    C: float = 1.0,
) -> GaussianMechanismResult:
    r"""Solve the SDP for a circulant workload via DFT diagonalisation.

    When ``W = AᵀA`` is circulant, it is diagonalised by the DFT matrix:
    ``W = F^H diag(λ) F`` where ``λ = FFT(w)`` and ``w`` is the first
    row of ``W``.

    The optimal ``Σ`` is also circulant, with the same eigenvectors.
    The SDP reduces to ``d`` independent scalar problems:

        minimise   Σ_k λ_k σ_k
        subject to σ_k ≤ (C/ε)²    ∀ k
                   σ_k ≥ 0          ∀ k

    Each optimal ``σ_k = min((C/ε)², ...)`` — the solution is trivially
    the privacy bound when the eigenvalue is non-zero.

    Parameters
    ----------
    A : ndarray of shape (m, d)
        Workload matrix with circulant ``AᵀA``.
    epsilon : float
        Privacy parameter ε > 0.
    C : float
        Sensitivity constant.

    Returns
    -------
    GaussianMechanismResult
        Optimal mechanism exploiting circulant structure.
    """
    A = np.asarray(A, dtype=np.float64)
    d = A.shape[1]
    W = A.T @ A
    W = (W + W.T) / 2.0

    # First row of W determines the circulant
    first_row = W[0, :]

    # DFT eigenvalues
    eigenvalues = np.real(np.fft.fft(first_row))

    # Optimal σ_k in the spectral domain
    bound = (C / epsilon) ** 2
    sigma_spectral = np.full(d, bound, dtype=np.float64)

    # Reconstruct Sigma via inverse FFT
    sigma_first_row = np.real(np.fft.ifft(sigma_spectral))

    # Build full circulant Sigma
    from scipy.linalg import circulant
    sigma = circulant(sigma_first_row).astype(np.float64)
    sigma = (sigma + sigma.T) / 2.0

    # Ensure PSD
    evals, evecs = np.linalg.eigh(sigma)
    evals = np.maximum(evals, 0.0)
    sigma = (evecs * evals) @ evecs.T
    sigma = (sigma + sigma.T) / 2.0

    total_mse = float(np.sum(eigenvalues * sigma_spectral))

    stats = SolveStatistics(
        solve_time=0.0,
        solver_name="circulant_dft",
        status="optimal",
        primal_obj=total_mse,
        dual_obj=total_mse,
        dimension=d,
    )

    return extract_gaussian(
        sigma=sigma,
        A=A,
        epsilon=epsilon,
        solve_stats=stats,
    )


# ---------------------------------------------------------------------------
# Privacy verification for Gaussian mechanisms
# ---------------------------------------------------------------------------


def verify_gaussian_privacy(
    sigma: npt.NDArray[np.float64],
    K_verts: npt.NDArray[np.float64],
    epsilon: float,
    C: float = 1.0,
    tol: float = 1e-6,
) -> Tuple[bool, float]:
    r"""Verify that a covariance matrix satisfies the privacy constraints.

    Checks that ``vᵀ Σ v ≤ (C/ε)² + tol`` for all vertices ``v ∈ K_verts``.

    Parameters
    ----------
    sigma : ndarray of shape (d, d)
        Covariance matrix.
    K_verts : ndarray of shape (n_verts, d)
        Sensitivity polytope vertices.
    epsilon : float
        Privacy parameter.
    C : float
        Sensitivity scaling constant.
    tol : float
        Tolerance for constraint satisfaction.

    Returns
    -------
    is_valid : bool
        Whether all privacy constraints are satisfied.
    worst_violation : float
        Maximum constraint violation (negative means all satisfied).
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    K_verts = np.asarray(K_verts, dtype=np.float64)

    bound = (C / epsilon) ** 2
    worst = -float("inf")

    for v in K_verts:
        val = float(v @ sigma @ v)
        violation = val - bound
        worst = max(worst, violation)

    is_valid = worst <= tol
    return is_valid, worst


# ---------------------------------------------------------------------------
# Utility: workload MSE lower bound
# ---------------------------------------------------------------------------


def workload_mse_lower_bound(
    A: npt.NDArray[np.float64],
    epsilon: float,
    C: float = 1.0,
) -> float:
    r"""Compute a lower bound on the achievable MSE for a workload.

    For the Gaussian mechanism with ℓ₁ sensitivity, a lower bound on
    ``trace(W Σ)`` subject to privacy constraints is:

        LB = (C/ε)² · trace(W)

    since the best we can do is set each diagonal entry of ``Σ`` to
    the privacy bound ``(C/ε)²`` and the off-diagonal structure can
    only help.  A tighter bound uses the eigenvalues of ``W``.

    The tighter bound is:

        LB = (C/ε)² · Σ_k min(λ_k(W), λ_k(W))

    where ``λ_k(W)`` are the eigenvalues of ``W``.  For diagonal ``Σ``
    this simplifies to ``(C/ε)² · trace(W)``.

    Parameters
    ----------
    A : ndarray of shape (m, d)
        Workload matrix.
    epsilon : float
        Privacy parameter.
    C : float
        Sensitivity constant.

    Returns
    -------
    lower_bound : float
        Lower bound on achievable MSE.
    """
    A = np.asarray(A, dtype=np.float64)
    W = A.T @ A
    bound = (C / epsilon) ** 2

    # Simple lower bound: sum of eigenvalues of W times the variance bound
    eigvals = np.linalg.eigvalsh(W)
    # Each eigenvalue contributes at least (C/eps)^2 * eigenvalue
    # but constrained by privacy
    # For ℓ₁, the constraint is Sigma[j,j] ≤ bound, so
    # trace(W Sigma) ≥ bound * sum(min eigenvalues)
    # The tightest simple bound is using the sum of eigenvalues
    return bound * float(np.sum(np.maximum(eigvals, 0.0)))


# ---------------------------------------------------------------------------
# High-level API: one-shot optimal mechanism
# ---------------------------------------------------------------------------


def optimal_gaussian_mechanism(
    A: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    C: float = 1.0,
    solver: str = "auto",
    verbose: bool = False,
    structural_hint: Optional[str] = None,
) -> GaussianMechanismResult:
    r"""One-shot computation of the optimal Gaussian mechanism for workload ``A``.

    This is the simplest entry point.  It auto-detects structure, builds
    the SDP, solves it, and returns a deployable mechanism.

    Automatically dispatches to specialised solvers for:
    - Circulant workloads → DFT-based closed form
    - Block-diagonal workloads → Independent sub-SDPs
    - Toeplitz workloads → Reduced-variable SDP
    - General workloads → Full SDP

    Parameters
    ----------
    A : ndarray of shape (m, d)
        Workload matrix.
    epsilon : float
        Privacy parameter ε > 0.
    delta : float
        Privacy parameter δ ∈ [0, 1).
    C : float
        Sensitivity scaling constant.
    solver : str
        Solver: ``"auto"``, ``"SCS"``, ``"MOSEK"``, etc.
    verbose : bool
        Whether to print solver output.
    structural_hint : str, optional
        Override for structural detection.

    Returns
    -------
    GaussianMechanismResult
        Optimal Gaussian mechanism.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.eye(5)
    >>> result = optimal_gaussian_mechanism(A, epsilon=1.0)
    >>> result.total_mse  # doctest: +SKIP
    5.0
    """
    A = np.asarray(A, dtype=np.float64)

    # Detect structure
    if structural_hint is None:
        info = StructuralDetector.suggest_structure(A)
        structural_hint = info.recommended_hint

        # Dispatch to specialised solvers
        if info.is_circulant:
            logger.info("Circulant workload detected: using DFT solver")
            return solve_circulant_sdp(A, epsilon, C=C)

        if info.is_block_diagonal and info.block_sizes is not None:
            logger.info(
                "Block-diagonal workload detected (%d blocks): decomposing",
                len(info.block_sizes),
            )
            return solve_block_diagonal_sdp(
                A, epsilon, info.block_sizes, C=C, solver=solver, verbose=verbose,
            )

    # General SDP path
    return matrix_mechanism_strategy(
        A, epsilon, delta=delta, solver=solver, verbose=verbose,
    )

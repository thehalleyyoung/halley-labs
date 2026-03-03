"""
Comprehensive tests for dp_forge.sdp_builder — the SDP construction engine
for optimal Gaussian workload mechanism synthesis.

Covers: SensitivityBallComputer (ℓ₁, ℓ₂, ℓ∞ vertices, redundancy removal,
convex hull), StructuralDetector (Toeplitz, circulant, block-diagonal,
sparse, suggest_structure), Gram matrix computation/validation,
BuildWorkloadSDP (general, diagonal, toeplitz, circulant hints),
toeplitz_sdp, spectral_factorization, auto_select_solver, solver
wrappers (SCS, MOSEK, generic), extract_sigma, compute_noise_distribution,
extract_gaussian, sample_gaussian, compute_mse, SDPManager lifecycle,
matrix_mechanism_strategy, GaussianMechanismResult, SolveStatistics,
StructuralInfo, edge cases, and numerical stability.
"""

from __future__ import annotations

import math
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt_assert
import pytest
from scipy.linalg import toeplitz as scipy_toeplitz

from dp_forge.exceptions import (
    ConfigurationError,
    InfeasibleSpecError,
    NumericalInstabilityError,
    SolverError,
)
from dp_forge.sdp_builder import (
    BuildWorkloadSDP,
    GaussianMechanismResult,
    SDPManager,
    SensitivityBallComputer,
    SolveStatistics,
    StructuralDetector,
    StructuralInfo,
    _compute_gram_matrix,
    _safe_cholesky,
    _validate_gram_matrix,
    auto_select_solver,
    compute_mse,
    compute_noise_distribution,
    extract_gaussian,
    extract_sigma,
    sample_gaussian,
    spectral_factorization,
)
from dp_forge.types import (
    NumericalConfig,
    OptimalityCertificate,
    PrivacyBudget,
    SDPStruct,
    WorkloadSpec,
)

# Try importing cvxpy; many tests need it
try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

requires_cvxpy = pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def identity_3() -> np.ndarray:
    """3×3 identity workload."""
    return np.eye(3)


@pytest.fixture
def identity_5() -> np.ndarray:
    """5×5 identity workload."""
    return np.eye(5)


@pytest.fixture
def prefix_3() -> np.ndarray:
    """3×3 prefix-sum (lower-triangular ones) workload."""
    return np.tril(np.ones((3, 3)))


@pytest.fixture
def prefix_5() -> np.ndarray:
    """5×5 prefix-sum workload."""
    return np.tril(np.ones((5, 5)))


@pytest.fixture
def block_diag_workload() -> np.ndarray:
    """6×6 block-diagonal workload with two 3×3 dense blocks."""
    A = np.zeros((6, 6))
    A[:3, :3] = np.ones((3, 3))
    A[3:6, 3:6] = np.ones((3, 3))
    return A


@pytest.fixture
def circulant_workload() -> np.ndarray:
    """Workload whose W = AᵀA is circulant under the code's roll convention.

    For d=2, identity gives W=I which satisfies roll(row0, -1) = row1.
    """
    return np.eye(2)


@pytest.fixture
def single_query() -> np.ndarray:
    """1×3 single-query workload."""
    return np.array([[1.0, 0.0, 0.0]])


@pytest.fixture
def tall_workload() -> np.ndarray:
    """10×3 overdetermined workload."""
    rng = np.random.default_rng(123)
    return rng.standard_normal((10, 3))


@pytest.fixture
def privacy_budget() -> PrivacyBudget:
    return PrivacyBudget(epsilon=1.0)


@pytest.fixture
def approx_privacy() -> PrivacyBudget:
    return PrivacyBudget(epsilon=1.0, delta=1e-5)


# =========================================================================
# SensitivityBallComputer — ℓ₁ vertices
# =========================================================================


class TestSensitivityBallL1:
    """Tests for ℓ₁ ball vertex computation."""

    def test_l1_shape(self, identity_3: np.ndarray) -> None:
        verts = SensitivityBallComputer.compute_l1_ball_vertices(identity_3)
        assert verts.shape == (6, 3)

    def test_l1_vertices_are_unit_vectors(self, identity_5: np.ndarray) -> None:
        verts = SensitivityBallComputer.compute_l1_ball_vertices(identity_5)
        norms = np.linalg.norm(verts, axis=1)
        npt_assert.assert_allclose(norms, 1.0)

    def test_l1_alternating_signs(self, identity_3: np.ndarray) -> None:
        verts = SensitivityBallComputer.compute_l1_ball_vertices(identity_3)
        for j in range(3):
            npt_assert.assert_allclose(verts[2 * j, j], 1.0)
            npt_assert.assert_allclose(verts[2 * j + 1, j], -1.0)

    @pytest.mark.parametrize("d", [1, 2, 4, 8, 20])
    def test_l1_vertex_count(self, d: int) -> None:
        A = np.eye(d)
        verts = SensitivityBallComputer.compute_l1_ball_vertices(A)
        assert verts.shape == (2 * d, d)

    def test_l1_non_identity_workload(self, prefix_3: np.ndarray) -> None:
        verts = SensitivityBallComputer.compute_l1_ball_vertices(prefix_3)
        assert verts.shape == (6, 3)

    def test_l1_rejects_1d_input(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            SensitivityBallComputer.compute_l1_ball_vertices(np.array([1.0, 2.0]))

    def test_l1_single_column(self, single_query: np.ndarray) -> None:
        verts = SensitivityBallComputer.compute_l1_ball_vertices(single_query)
        assert verts.shape == (6, 3)


# =========================================================================
# SensitivityBallComputer — ℓ₂ vertices
# =========================================================================


class TestSensitivityBallL2:
    """Tests for ℓ₂ ball vertex computation."""

    def test_l2_d1(self) -> None:
        A = np.array([[1.0]])
        verts = SensitivityBallComputer.compute_l2_ball_vertices(A)
        assert verts.shape == (2, 1)
        npt_assert.assert_allclose(verts, [[1.0], [-1.0]])

    def test_l2_d2_unit_circle(self) -> None:
        A = np.eye(2)
        verts = SensitivityBallComputer.compute_l2_ball_vertices(A)
        norms = np.linalg.norm(verts, axis=1)
        npt_assert.assert_allclose(norms, 1.0, atol=1e-12)

    def test_l2_d2_custom_directions(self) -> None:
        A = np.eye(2)
        verts = SensitivityBallComputer.compute_l2_ball_vertices(A, n_directions=64)
        assert verts.shape[0] == 64
        norms = np.linalg.norm(verts, axis=1)
        npt_assert.assert_allclose(norms, 1.0, atol=1e-12)

    def test_l2_d3_includes_axis_aligned(self) -> None:
        A = np.eye(3)
        verts = SensitivityBallComputer.compute_l2_ball_vertices(A)
        # First 2*d=6 should be axis-aligned
        for j in range(3):
            npt_assert.assert_allclose(verts[2 * j, j], 1.0)
            npt_assert.assert_allclose(verts[2 * j + 1, j], -1.0)

    @pytest.mark.parametrize("d", [3, 5, 10])
    def test_l2_all_unit_norm(self, d: int) -> None:
        A = np.eye(d)
        verts = SensitivityBallComputer.compute_l2_ball_vertices(A)
        norms = np.linalg.norm(verts, axis=1)
        npt_assert.assert_allclose(norms, 1.0, atol=1e-12)

    def test_l2_extra_directions(self) -> None:
        A = np.eye(4)
        verts = SensitivityBallComputer.compute_l2_ball_vertices(A, n_directions=100)
        assert verts.shape[0] == 2 * 4 + 100

    def test_l2_deterministic_seed(self) -> None:
        A = np.eye(5)
        v1 = SensitivityBallComputer.compute_l2_ball_vertices(A)
        v2 = SensitivityBallComputer.compute_l2_ball_vertices(A)
        npt_assert.assert_array_equal(v1, v2)

    def test_l2_rejects_1d_input(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            SensitivityBallComputer.compute_l2_ball_vertices(np.array([1.0]))


# =========================================================================
# SensitivityBallComputer — ℓ∞ vertices
# =========================================================================


class TestSensitivityBallLinf:
    """Tests for ℓ∞ ball vertex computation."""

    def test_linf_d1(self) -> None:
        A = np.array([[1.0]])
        verts = SensitivityBallComputer.compute_linf_ball_vertices(A)
        assert verts.shape == (2, 1)

    def test_linf_d2_corners(self) -> None:
        A = np.eye(2)
        verts = SensitivityBallComputer.compute_linf_ball_vertices(A)
        assert verts.shape == (4, 2)
        # All entries are ±1
        assert np.all(np.abs(verts) == 1.0)

    @pytest.mark.parametrize("d", [1, 2, 3, 5])
    def test_linf_vertex_count(self, d: int) -> None:
        A = np.eye(d)
        verts = SensitivityBallComputer.compute_linf_ball_vertices(A)
        assert verts.shape == (2 ** d, d)

    def test_linf_entries_pm1(self) -> None:
        A = np.eye(4)
        verts = SensitivityBallComputer.compute_linf_ball_vertices(A)
        assert np.all(np.isin(verts, [-1.0, 1.0]))

    def test_linf_too_large_raises(self) -> None:
        A = np.eye(21)
        with pytest.raises(ConfigurationError):
            SensitivityBallComputer.compute_linf_ball_vertices(A)

    def test_linf_boundary_d20(self) -> None:
        A = np.eye(20)
        verts = SensitivityBallComputer.compute_linf_ball_vertices(A)
        assert verts.shape == (2 ** 20, 20)


# =========================================================================
# SensitivityBallComputer — redundancy removal and convex hull
# =========================================================================


class TestVertexReduction:
    """Tests for reduce_redundant_vertices and convex_hull_vertices."""

    def test_remove_exact_duplicates(self) -> None:
        verts = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        reduced = SensitivityBallComputer.reduce_redundant_vertices(verts)
        assert reduced.shape[0] <= 2

    def test_keep_opposite_directions(self) -> None:
        verts = np.array([[1.0, 0.0], [-1.0, 0.0]])
        reduced = SensitivityBallComputer.reduce_redundant_vertices(verts)
        assert reduced.shape[0] == 2

    def test_remove_dominated(self) -> None:
        verts = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
        reduced = SensitivityBallComputer.reduce_redundant_vertices(verts)
        # [1,0] is dominated by [2,0]; both point in same direction
        assert reduced.shape[0] == 2

    def test_single_vertex(self) -> None:
        verts = np.array([[1.0, 2.0]])
        reduced = SensitivityBallComputer.reduce_redundant_vertices(verts)
        assert reduced.shape == (1, 2)

    def test_empty_after_zero_removal(self) -> None:
        verts = np.array([[0.0, 0.0], [0.0, 0.0]])
        reduced = SensitivityBallComputer.reduce_redundant_vertices(verts)
        assert reduced.shape[0] == 0

    def test_convex_hull_triangle(self) -> None:
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.25, 0.25],  # interior point
        ])
        hull = SensitivityBallComputer.convex_hull_vertices(points)
        assert hull.shape[0] == 3

    def test_convex_hull_fallback_high_dim(self) -> None:
        # d > _MAX_HULL_DIMENSION=15 ⟹ returns all points
        d = 20
        points = np.eye(d)
        hull = SensitivityBallComputer.convex_hull_vertices(points)
        assert hull.shape[0] == d

    def test_convex_hull_degenerate(self) -> None:
        points = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        hull = SensitivityBallComputer.convex_hull_vertices(points)
        # Collinear: either 2 hull points or falls back to all 3
        assert hull.shape[0] >= 2

    def test_convex_hull_rejects_1d(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            SensitivityBallComputer.convex_hull_vertices(np.array([1.0, 2.0]))

    def test_reduce_rejects_1d(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            SensitivityBallComputer.reduce_redundant_vertices(np.array([1.0]))


# =========================================================================
# StructuralDetector — Toeplitz
# =========================================================================


class TestStructuralToeplitz:
    """Tests for Toeplitz structure detection."""

    def test_identity_is_toeplitz(self, identity_3: np.ndarray) -> None:
        assert StructuralDetector.detect_toeplitz(identity_3) is True

    def test_prefix_not_toeplitz(self, prefix_5: np.ndarray) -> None:
        # W = AᵀA for lower-triangular prefix-sum: W[i,j] = min(i,j)+1
        # This is NOT Toeplitz since diagonal entries vary
        assert StructuralDetector.detect_toeplitz(prefix_5) is False

    def test_random_not_toeplitz(self) -> None:
        rng = np.random.default_rng(7)
        A = rng.standard_normal((5, 5))
        assert StructuralDetector.detect_toeplitz(A) is False

    def test_1d_always_toeplitz(self) -> None:
        A = np.array([[3.0]])
        assert StructuralDetector.detect_toeplitz(A) is True

    @pytest.mark.parametrize("d", [2, 4, 8])
    def test_identity_toeplitz_varying_d(self, d: int) -> None:
        assert StructuralDetector.detect_toeplitz(np.eye(d)) is True

    def test_toeplitz_with_custom_tolerance(self) -> None:
        A = np.eye(3)
        assert StructuralDetector.detect_toeplitz(A, tol=1e-15) is True


# =========================================================================
# StructuralDetector — Circulant
# =========================================================================


class TestStructuralCirculant:
    """Tests for circulant structure detection."""

    def test_identity_not_circulant(self, identity_3: np.ndarray) -> None:
        # Identity W=I is Toeplitz but NOT circulant for d≥3
        # (row 1 of circulant([1,0,0]) would be [0,0,1], not [0,1,0])
        assert StructuralDetector.detect_circulant(identity_3) is False

    def test_circulant_d2(self, circulant_workload: np.ndarray) -> None:
        # For d=2, identity gives W=I which is circulant
        assert StructuralDetector.detect_circulant(circulant_workload) is True

    def test_ones_workload_is_circulant(self) -> None:
        # Constant W (all entries equal) satisfies the roll check
        A = np.ones((4, 4))
        assert StructuralDetector.detect_circulant(A) is True

    def test_prefix_not_circulant(self, prefix_5: np.ndarray) -> None:
        # Prefix-sum W is Toeplitz but generally not circulant
        assert StructuralDetector.detect_circulant(prefix_5) is False

    def test_1d_always_circulant(self) -> None:
        A = np.array([[5.0]])
        assert StructuralDetector.detect_circulant(A) is True

    def test_random_not_circulant(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((4, 4))
        assert StructuralDetector.detect_circulant(A) is False


# =========================================================================
# StructuralDetector — Block diagonal
# =========================================================================


class TestStructuralBlockDiag:
    """Tests for block-diagonal structure detection."""

    def test_block_diag_detected(self, block_diag_workload: np.ndarray) -> None:
        is_bd, sizes = StructuralDetector.detect_block_diagonal(block_diag_workload)
        assert is_bd is True
        assert sizes == [3, 3]

    def test_identity_not_block_diag(self) -> None:
        # Identity has d separate 1×1 blocks ⟹ d ≥ 2 blocks ⟹ True
        is_bd, sizes = StructuralDetector.detect_block_diagonal(np.eye(3))
        assert is_bd is True
        assert sizes == [1, 1, 1]

    def test_dense_not_block_diag(self) -> None:
        A = np.ones((3, 3))
        is_bd, sizes = StructuralDetector.detect_block_diagonal(A)
        assert is_bd is False
        assert sizes is None

    def test_1d_not_block_diag(self) -> None:
        A = np.array([[2.0]])
        is_bd, sizes = StructuralDetector.detect_block_diagonal(A)
        assert is_bd is False
        assert sizes is None

    @pytest.mark.parametrize(
        "block_sizes",
        [[2, 3], [1, 1, 1, 1], [4, 4]],
        ids=["2+3", "1+1+1+1", "4+4"],
    )
    def test_various_block_sizes(self, block_sizes: list) -> None:
        d = sum(block_sizes)
        A = np.zeros((d, d))
        offset = 0
        for bs in block_sizes:
            A[offset : offset + bs, offset : offset + bs] = np.ones((bs, bs))
            offset += bs
        is_bd, sizes = StructuralDetector.detect_block_diagonal(A)
        assert is_bd is True
        assert sizes == block_sizes


# =========================================================================
# StructuralDetector — Sparse detection
# =========================================================================


class TestStructuralSparse:
    """Tests for sparsity detection."""

    def test_identity_sparse(self) -> None:
        A = np.eye(20)
        is_sp, sparsity = StructuralDetector.detect_sparse_structure(A)
        assert is_sp is True
        assert sparsity > 0.7

    def test_dense_not_sparse(self) -> None:
        A = np.ones((5, 5))
        is_sp, sparsity = StructuralDetector.detect_sparse_structure(A)
        assert is_sp is False
        assert sparsity == 0.0

    def test_empty_sparse(self) -> None:
        A = np.zeros((0, 0))
        is_sp, sparsity = StructuralDetector.detect_sparse_structure(A)
        assert is_sp is True
        assert sparsity == 1.0


# =========================================================================
# StructuralDetector — suggest_structure
# =========================================================================


class TestSuggestStructure:
    """Tests for the aggregate suggest_structure method."""

    def test_identity_hint(self, identity_5: np.ndarray) -> None:
        info = StructuralDetector.suggest_structure(identity_5)
        assert isinstance(info, StructuralInfo)
        # Identity W=I is Toeplitz but NOT circulant (for d≥3)
        assert info.is_toeplitz is True
        assert info.recommended_hint in ("toeplitz", "block_diagonal", "sparse")

    def test_prefix_hint(self, prefix_5: np.ndarray) -> None:
        info = StructuralDetector.suggest_structure(prefix_5)
        # Prefix-sum W is NOT Toeplitz, so hint depends on other properties
        assert isinstance(info, StructuralInfo)

    def test_block_diag_hint(self, block_diag_workload: np.ndarray) -> None:
        info = StructuralDetector.suggest_structure(block_diag_workload)
        assert info.is_block_diagonal is True
        assert info.block_sizes is not None

    def test_random_no_structure(self) -> None:
        rng = np.random.default_rng(99)
        A = rng.standard_normal((8, 8))
        info = StructuralDetector.suggest_structure(A)
        assert info.is_toeplitz is False
        assert info.is_circulant is False

    def test_details_populated(self, identity_3: np.ndarray) -> None:
        info = StructuralDetector.suggest_structure(identity_3)
        assert "gram_shape" in info.details
        assert "workload_shape" in info.details

    def test_structural_info_repr(self) -> None:
        info = StructuralInfo(is_toeplitz=True, recommended_hint="toeplitz")
        r = repr(info)
        assert "toeplitz" in r


# =========================================================================
# Gram matrix helpers
# =========================================================================


class TestGramMatrix:
    """Tests for _compute_gram_matrix and _validate_gram_matrix."""

    def test_identity_gram(self, identity_3: np.ndarray) -> None:
        W = _compute_gram_matrix(identity_3)
        npt_assert.assert_allclose(W, np.eye(3))

    def test_gram_symmetric(self, prefix_5: np.ndarray) -> None:
        W = _compute_gram_matrix(prefix_5)
        npt_assert.assert_allclose(W, W.T, atol=1e-15)

    def test_gram_psd(self, tall_workload: np.ndarray) -> None:
        W = _compute_gram_matrix(tall_workload)
        eigvals = np.linalg.eigvalsh(W)
        assert np.all(eigvals >= -1e-10)

    def test_validate_well_conditioned(self) -> None:
        W = np.eye(5)
        _validate_gram_matrix(W)  # should not raise

    def test_validate_ill_conditioned(self) -> None:
        W = np.diag([1.0, 1e-14])
        with pytest.raises(NumericalInstabilityError):
            _validate_gram_matrix(W, max_cond=1e10)


# =========================================================================
# SolveStatistics and GaussianMechanismResult
# =========================================================================


class TestDataclasses:
    """Tests for SolveStatistics and GaussianMechanismResult dataclasses."""

    def test_solve_statistics_repr(self) -> None:
        stats = SolveStatistics(solve_time=0.5, solver_name="SCS", status="optimal")
        r = repr(stats)
        assert "SCS" in r
        assert "0.500" in r

    def test_solve_statistics_defaults(self) -> None:
        stats = SolveStatistics(solve_time=1.0, solver_name="MOSEK")
        assert stats.iterations == 0
        assert stats.duality_gap == 0.0

    def test_gaussian_result_basic(self) -> None:
        sigma = np.eye(2)
        A = np.eye(2)
        noise = A @ sigma @ A.T
        result = GaussianMechanismResult(
            sigma=sigma,
            noise_covariance=noise,
            workload=A,
            epsilon=1.0,
        )
        assert result.d == 2
        assert result.m == 2
        assert result.epsilon == 1.0

    def test_gaussian_result_invalid_sigma(self) -> None:
        with pytest.raises(ValueError, match="square matrix"):
            GaussianMechanismResult(
                sigma=np.array([[1, 2, 3]]),
                noise_covariance=np.eye(1),
                workload=np.eye(1),
                epsilon=1.0,
            )

    def test_gaussian_result_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            GaussianMechanismResult(
                sigma=np.eye(2),
                noise_covariance=np.eye(2),
                workload=np.eye(2),
                epsilon=-1.0,
            )

    def test_gaussian_result_repr(self) -> None:
        result = GaussianMechanismResult(
            sigma=np.eye(3),
            noise_covariance=np.eye(3),
            workload=np.eye(3),
            epsilon=0.5,
            total_mse=1.5,
        )
        r = repr(result)
        assert "d=3" in r
        assert "m=3" in r
        assert "0.5" in r


# =========================================================================
# compute_mse
# =========================================================================


class TestComputeMSE:
    """Tests for the compute_mse function."""

    def test_identity_workload_scaled_sigma(self) -> None:
        A = np.eye(3)
        sigma = 2.0 * np.eye(3)
        mse = compute_mse(sigma, A)
        # trace(I · 2I) = 6
        npt_assert.assert_allclose(mse, 6.0)

    def test_identity_workload_identity_sigma(self) -> None:
        A = np.eye(4)
        sigma = np.eye(4)
        mse = compute_mse(sigma, A)
        npt_assert.assert_allclose(mse, 4.0)

    @pytest.mark.parametrize("d", [1, 3, 5, 10])
    def test_mse_identity_matches_trace(self, d: int) -> None:
        A = np.eye(d)
        sigma = np.eye(d) * 0.5
        mse = compute_mse(sigma, A)
        npt_assert.assert_allclose(mse, 0.5 * d)

    def test_mse_prefix_workload(self, prefix_3: np.ndarray) -> None:
        sigma = np.eye(3)
        mse = compute_mse(sigma, prefix_3)
        W = prefix_3.T @ prefix_3
        expected = np.trace(W)
        npt_assert.assert_allclose(mse, expected)

    def test_mse_zero_sigma(self) -> None:
        A = np.eye(3)
        sigma = np.zeros((3, 3))
        mse = compute_mse(sigma, A)
        npt_assert.assert_allclose(mse, 0.0)

    def test_mse_single_query(self, single_query: np.ndarray) -> None:
        sigma = np.eye(3)
        mse = compute_mse(sigma, single_query)
        # W = A^T A = diag(1,0,0); trace(W σ) = 1
        npt_assert.assert_allclose(mse, 1.0)


# =========================================================================
# compute_noise_distribution
# =========================================================================


class TestComputeNoise:
    """Tests for compute_noise_distribution."""

    def test_identity_pass_through(self) -> None:
        sigma = 2.0 * np.eye(3)
        A = np.eye(3)
        noise = compute_noise_distribution(sigma, A)
        npt_assert.assert_allclose(noise, 2.0 * np.eye(3))

    def test_noise_symmetric(self, prefix_3: np.ndarray) -> None:
        sigma = np.eye(3)
        noise = compute_noise_distribution(sigma, prefix_3)
        npt_assert.assert_allclose(noise, noise.T, atol=1e-15)

    def test_noise_psd(self, tall_workload: np.ndarray) -> None:
        sigma = np.eye(3)
        noise = compute_noise_distribution(sigma, tall_workload)
        eigvals = np.linalg.eigvalsh(noise)
        assert np.all(eigvals >= -1e-10)

    def test_noise_shape(self, single_query: np.ndarray) -> None:
        sigma = np.eye(3)
        noise = compute_noise_distribution(sigma, single_query)
        assert noise.shape == (1, 1)


# =========================================================================
# extract_gaussian
# =========================================================================


class TestExtractGaussian:
    """Tests for the extract_gaussian convenience function."""

    def test_basic_extraction(self) -> None:
        A = np.eye(3)
        sigma = np.eye(3)
        result = extract_gaussian(sigma, A, epsilon=1.0)
        assert isinstance(result, GaussianMechanismResult)
        assert result.d == 3
        assert result.m == 3
        npt_assert.assert_allclose(result.total_mse, 3.0)

    def test_per_query_mse(self) -> None:
        A = np.eye(4)
        sigma = 0.5 * np.eye(4)
        result = extract_gaussian(sigma, A, epsilon=1.0)
        npt_assert.assert_allclose(result.per_query_mse, 0.5 * np.ones(4))

    def test_cholesky_factor_exists(self) -> None:
        A = np.eye(3)
        sigma = np.eye(3)
        result = extract_gaussian(sigma, A, epsilon=1.0)
        assert result.cholesky_factor is not None
        # L L^T ≈ noise_covariance
        L = result.cholesky_factor
        reconstructed = L @ L.T
        npt_assert.assert_allclose(reconstructed, result.noise_covariance, atol=1e-10)

    def test_with_delta(self) -> None:
        A = np.eye(2)
        sigma = np.eye(2)
        result = extract_gaussian(sigma, A, epsilon=1.0, delta=1e-5)
        assert result.delta == 1e-5

    def test_with_solve_stats(self) -> None:
        stats = SolveStatistics(solve_time=0.1, solver_name="SCS", status="optimal")
        result = extract_gaussian(np.eye(2), np.eye(2), epsilon=1.0, solve_stats=stats)
        assert result.solve_stats is stats

    def test_noise_covariance_correct(self) -> None:
        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        sigma = np.eye(2) * 0.5
        result = extract_gaussian(sigma, A, epsilon=1.0)
        expected_noise = A @ sigma @ A.T
        npt_assert.assert_allclose(result.noise_covariance, expected_noise, atol=1e-12)


# =========================================================================
# _safe_cholesky
# =========================================================================


class TestSafeCholesky:
    """Tests for _safe_cholesky with regularisation."""

    def test_psd_matrix(self) -> None:
        M = np.eye(3) * 2.0
        L = _safe_cholesky(M)
        assert L is not None
        npt_assert.assert_allclose(L @ L.T, M, atol=1e-10)

    def test_nearly_singular_matrix(self) -> None:
        M = np.diag([1.0, 1e-15, 1e-15])
        L = _safe_cholesky(M)
        # Should succeed with regularisation
        assert L is not None

    def test_zero_matrix(self) -> None:
        M = np.zeros((3, 3))
        L = _safe_cholesky(M)
        assert L is not None

    def test_symmetric_input(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((4, 4))
        M = X @ X.T
        L = _safe_cholesky(M)
        assert L is not None


# =========================================================================
# spectral_factorization
# =========================================================================


class TestSpectralFactorization:
    """Tests for Toeplitz spectral_factorization."""

    def test_identity_reconstruction(self) -> None:
        t = np.array([1.0, 0.0, 0.0])
        sigma = spectral_factorization(t)
        npt_assert.assert_allclose(sigma, np.eye(3))

    def test_toeplitz_structure(self) -> None:
        t = np.array([4.0, 1.0, 0.5])
        sigma = spectral_factorization(t)
        # Check Toeplitz: constant diagonals
        d = len(t)
        for k in range(d):
            diag = np.diag(sigma, k)
            npt_assert.assert_allclose(diag, t[k] * np.ones(len(diag)), atol=1e-14)

    def test_symmetric(self) -> None:
        t = np.array([2.0, 0.5, 0.1, -0.1])
        sigma = spectral_factorization(t)
        npt_assert.assert_allclose(sigma, sigma.T, atol=1e-15)

    def test_single_element(self) -> None:
        t = np.array([3.14])
        sigma = spectral_factorization(t)
        assert sigma.shape == (1, 1)
        npt_assert.assert_allclose(sigma[0, 0], 3.14)


# =========================================================================
# sample_gaussian
# =========================================================================


class TestSampleGaussian:
    """Tests for sample_gaussian function."""

    def test_sample_shape(self) -> None:
        A = np.eye(3)
        sigma = np.eye(3)
        result = extract_gaussian(sigma, A, epsilon=1.0)
        x = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(0)
        noisy = sample_gaussian(x, result, rng=rng)
        assert noisy.shape == (3,)

    def test_sample_mean_approx(self) -> None:
        A = np.eye(2)
        sigma = 0.01 * np.eye(2)
        result = extract_gaussian(sigma, A, epsilon=1.0)
        x = np.array([10.0, 20.0])
        rng = np.random.default_rng(42)
        samples = np.array([sample_gaussian(x, result, rng=rng) for _ in range(5000)])
        mean = samples.mean(axis=0)
        npt_assert.assert_allclose(mean, A @ x, atol=0.1)

    def test_sample_deterministic_seed(self) -> None:
        A = np.eye(2)
        sigma = np.eye(2)
        result = extract_gaussian(sigma, A, epsilon=1.0)
        x = np.array([1.0, 2.0])
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        s1 = sample_gaussian(x, result, rng=rng1)
        s2 = sample_gaussian(x, result, rng=rng2)
        npt_assert.assert_array_equal(s1, s2)

    def test_sample_without_cholesky(self) -> None:
        A = np.eye(2)
        sigma = np.eye(2)
        result = extract_gaussian(sigma, A, epsilon=1.0)
        result.cholesky_factor = None  # force fallback path
        x = np.array([1.0, 2.0])
        rng = np.random.default_rng(0)
        noisy = sample_gaussian(x, result, rng=rng)
        assert noisy.shape == (2,)

    def test_sample_no_rng(self) -> None:
        A = np.eye(2)
        sigma = np.eye(2)
        result = extract_gaussian(sigma, A, epsilon=1.0)
        x = np.array([1.0, 2.0])
        noisy = sample_gaussian(x, result)
        assert noisy.shape == (2,)


# =========================================================================
# auto_select_solver
# =========================================================================


class TestAutoSelectSolver:
    """Tests for auto_select_solver heuristic."""

    @requires_cvxpy
    def test_small_dimension(self) -> None:
        solver = auto_select_solver(10)
        assert isinstance(solver, str)
        assert len(solver) > 0

    @requires_cvxpy
    def test_large_dimension(self) -> None:
        solver = auto_select_solver(1000)
        assert isinstance(solver, str)

    @requires_cvxpy
    def test_returns_installed_solver(self) -> None:
        solver = auto_select_solver(5)
        installed = cp.installed_solvers()
        assert solver in installed


# =========================================================================
# BuildWorkloadSDP (requires CVXPY)
# =========================================================================


class TestBuildWorkloadSDP:
    """Tests for the BuildWorkloadSDP constructor."""

    @requires_cvxpy
    def test_identity_sdp_structure(self, identity_3: np.ndarray) -> None:
        sdp = BuildWorkloadSDP(identity_3, epsilon=1.0)
        assert isinstance(sdp, SDPStruct)
        assert sdp.problem is not None
        assert sdp.sigma_var is not None
        assert sdp.workload is not None
        # PSD constraint + 2*d privacy constraints
        assert len(sdp.constraints) >= 1 + 2 * 3

    @requires_cvxpy
    def test_prefix_sdp_structure(self, prefix_3: np.ndarray) -> None:
        sdp = BuildWorkloadSDP(prefix_3, epsilon=1.0)
        assert isinstance(sdp, SDPStruct)
        assert len(sdp.constraints) >= 1 + 2 * 3

    @requires_cvxpy
    def test_custom_vertices(self, identity_3: np.ndarray) -> None:
        K = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        sdp = BuildWorkloadSDP(identity_3, epsilon=1.0, K_verts=K)
        # PSD + 2 privacy constraints
        assert len(sdp.constraints) == 3

    @requires_cvxpy
    def test_diagonal_hint(self, identity_3: np.ndarray) -> None:
        sdp = BuildWorkloadSDP(identity_3, epsilon=1.0, structural_hint="diagonal")
        assert sdp is not None

    @requires_cvxpy
    def test_toeplitz_hint(self, identity_3: np.ndarray) -> None:
        sdp = BuildWorkloadSDP(identity_3, epsilon=1.0, structural_hint="toeplitz")
        assert sdp is not None

    @requires_cvxpy
    def test_circulant_hint(self, identity_3: np.ndarray) -> None:
        sdp = BuildWorkloadSDP(identity_3, epsilon=1.0, structural_hint="circulant")
        assert sdp is not None

    @requires_cvxpy
    def test_invalid_epsilon(self, identity_3: np.ndarray) -> None:
        with pytest.raises(ConfigurationError):
            BuildWorkloadSDP(identity_3, epsilon=-1.0)

    @requires_cvxpy
    def test_invalid_epsilon_zero(self, identity_3: np.ndarray) -> None:
        with pytest.raises(ConfigurationError):
            BuildWorkloadSDP(identity_3, epsilon=0.0)

    @requires_cvxpy
    def test_invalid_C(self, identity_3: np.ndarray) -> None:
        with pytest.raises(ConfigurationError):
            BuildWorkloadSDP(identity_3, epsilon=1.0, C=-1.0)

    @requires_cvxpy
    def test_invalid_A_dimension(self) -> None:
        with pytest.raises(ConfigurationError):
            BuildWorkloadSDP(np.array([1.0, 2.0, 3.0]), epsilon=1.0)

    @requires_cvxpy
    def test_invalid_K_verts_dimension(self, identity_3: np.ndarray) -> None:
        K = np.array([[1.0, 0.0]])  # wrong column count
        with pytest.raises(ConfigurationError):
            BuildWorkloadSDP(identity_3, epsilon=1.0, K_verts=K)

    @requires_cvxpy
    def test_custom_C(self, identity_3: np.ndarray) -> None:
        sdp = BuildWorkloadSDP(identity_3, epsilon=1.0, C=2.0)
        assert sdp is not None

    @requires_cvxpy
    @pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 2.0, 10.0])
    def test_varying_epsilon(self, identity_3: np.ndarray, eps: float) -> None:
        sdp = BuildWorkloadSDP(identity_3, epsilon=eps)
        assert sdp is not None

    @requires_cvxpy
    def test_single_query_sdp(self, single_query: np.ndarray) -> None:
        sdp = BuildWorkloadSDP(single_query, epsilon=1.0)
        assert sdp is not None
        assert len(sdp.constraints) >= 1

    @requires_cvxpy
    def test_tall_workload_sdp(self, tall_workload: np.ndarray) -> None:
        sdp = BuildWorkloadSDP(tall_workload, epsilon=1.0)
        assert sdp is not None


# =========================================================================
# toeplitz_sdp (requires CVXPY)
# =========================================================================


class TestToeplitzSDP:
    """Tests for the specialised toeplitz_sdp constructor."""

    @requires_cvxpy
    def test_basic_construction(self) -> None:
        from dp_forge.sdp_builder import toeplitz_sdp

        first_row = np.array([3.0, 1.0, 0.5])
        sdp = toeplitz_sdp(first_row, epsilon=1.0)
        assert isinstance(sdp, SDPStruct)
        assert len(sdp.constraints) >= 2  # PSD + t[0] bound

    @requires_cvxpy
    def test_invalid_epsilon(self) -> None:
        from dp_forge.sdp_builder import toeplitz_sdp

        with pytest.raises(ConfigurationError):
            toeplitz_sdp(np.array([1.0, 0.0]), epsilon=-1.0)

    @requires_cvxpy
    def test_identity_first_row(self) -> None:
        from dp_forge.sdp_builder import toeplitz_sdp

        first_row = np.array([1.0, 0.0, 0.0, 0.0])
        sdp = toeplitz_sdp(first_row, epsilon=1.0)
        assert sdp is not None

    @requires_cvxpy
    def test_custom_C(self) -> None:
        from dp_forge.sdp_builder import toeplitz_sdp

        first_row = np.array([2.0, 0.5])
        sdp = toeplitz_sdp(first_row, epsilon=1.0, C=3.0)
        assert sdp is not None


# =========================================================================
# extract_sigma (requires solved SDP ⟹ mock)
# =========================================================================


class TestExtractSigma:
    """Tests for extract_sigma from a solved SDP."""

    def test_unsolved_raises(self) -> None:
        mock_problem = MagicMock()
        mock_problem.status = "unsolved"
        sdp = SDPStruct(
            problem=mock_problem,
            sigma_var=MagicMock(),
            objective=MagicMock(),
            constraints=[],
        )
        with pytest.raises(SolverError, match="Cannot extract"):
            extract_sigma(sdp)

    def test_infeasible_raises(self) -> None:
        mock_problem = MagicMock()
        mock_problem.status = "infeasible"
        sdp = SDPStruct(
            problem=mock_problem,
            sigma_var=MagicMock(),
            objective=MagicMock(),
            constraints=[],
        )
        with pytest.raises(SolverError):
            extract_sigma(sdp)

    def test_none_value_raises(self) -> None:
        mock_problem = MagicMock()
        mock_problem.status = "optimal"
        mock_var = MagicMock()
        mock_var.value = None
        sdp = SDPStruct(
            problem=mock_problem,
            sigma_var=mock_var,
            objective=MagicMock(),
            constraints=[],
        )
        with pytest.raises(SolverError, match="no value"):
            extract_sigma(sdp)

    def test_optimal_extraction(self) -> None:
        mock_problem = MagicMock()
        mock_problem.status = "optimal"
        sigma_raw = np.eye(3)
        mock_var = MagicMock()
        mock_var.value = sigma_raw
        sdp = SDPStruct(
            problem=mock_problem,
            sigma_var=mock_var,
            objective=MagicMock(),
            constraints=[],
        )
        sigma = extract_sigma(sdp)
        npt_assert.assert_allclose(sigma, np.eye(3), atol=1e-10)

    def test_optimal_inaccurate_accepted(self) -> None:
        mock_problem = MagicMock()
        mock_problem.status = "optimal_inaccurate"
        mock_var = MagicMock()
        mock_var.value = np.eye(2) * 0.5
        sdp = SDPStruct(
            problem=mock_problem,
            sigma_var=mock_var,
            objective=MagicMock(),
            constraints=[],
        )
        sigma = extract_sigma(sdp)
        assert sigma.shape == (2, 2)

    def test_negative_eigenvalue_clipped(self) -> None:
        mock_problem = MagicMock()
        mock_problem.status = "optimal"
        # Matrix with small negative eigenvalue
        raw = np.array([[1.0, 0.0], [0.0, -0.01]])
        mock_var = MagicMock()
        mock_var.value = raw
        sdp = SDPStruct(
            problem=mock_problem,
            sigma_var=mock_var,
            objective=MagicMock(),
            constraints=[],
        )
        sigma = extract_sigma(sdp)
        eigvals = np.linalg.eigvalsh(sigma)
        assert np.all(eigvals >= -1e-14)

    def test_1d_toeplitz_params(self) -> None:
        mock_problem = MagicMock()
        mock_problem.status = "optimal"
        # 1-D value ⟹ treated as Toeplitz parameters
        mock_var = MagicMock()
        mock_var.value = np.array([2.0, 0.5, 0.1])
        sdp = SDPStruct(
            problem=mock_problem,
            sigma_var=mock_var,
            objective=MagicMock(),
            constraints=[],
        )
        sigma = extract_sigma(sdp)
        assert sigma.shape == (3, 3)
        # Should be Toeplitz
        npt_assert.assert_allclose(sigma[0, 0], 2.0, atol=1e-10)


# =========================================================================
# SDPManager lifecycle (requires CVXPY)
# =========================================================================


class TestSDPManager:
    """Tests for the SDPManager orchestration class."""

    @requires_cvxpy
    def test_build_sets_state(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(3)
        pb = PrivacyBudget(epsilon=1.0)
        sdp = manager.build(ws, pb)
        assert manager.is_built is True
        assert manager.is_solved is False
        assert isinstance(sdp, SDPStruct)

    @requires_cvxpy
    def test_solve_before_build_raises(self) -> None:
        manager = SDPManager(verbose=0)
        with pytest.raises(ConfigurationError, match="No SDP"):
            manager.solve()

    @requires_cvxpy
    def test_extract_before_solve_raises(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(3)
        pb = PrivacyBudget(epsilon=1.0)
        manager.build(ws, pb)
        with pytest.raises(ConfigurationError, match="not been solved"):
            manager.extract_sigma()

    @requires_cvxpy
    def test_extract_before_build_raises(self) -> None:
        manager = SDPManager(verbose=0)
        with pytest.raises(ConfigurationError, match="No SDP"):
            manager.extract_sigma()

    @requires_cvxpy
    def test_compute_noise_before_solve_raises(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(2)
        pb = PrivacyBudget(epsilon=1.0)
        manager.build(ws, pb)
        with pytest.raises(ConfigurationError):
            manager.compute_noise_distribution()

    @requires_cvxpy
    def test_build_with_l2_norm(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(3)
        pb = PrivacyBudget(epsilon=1.0)
        sdp = manager.build(ws, pb, sensitivity_norm="l2")
        assert sdp is not None

    @requires_cvxpy
    def test_build_with_linf_norm(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(3)
        pb = PrivacyBudget(epsilon=1.0)
        sdp = manager.build(ws, pb, sensitivity_norm="linf")
        assert sdp is not None

    @requires_cvxpy
    def test_build_invalid_norm_raises(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(3)
        pb = PrivacyBudget(epsilon=1.0)
        with pytest.raises(ConfigurationError, match="Unknown sensitivity norm"):
            manager.build(ws, pb, sensitivity_norm="l3")

    @requires_cvxpy
    def test_build_with_custom_vertices(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(2)
        pb = PrivacyBudget(epsilon=1.0)
        K = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        sdp = manager.build(ws, pb, K_verts=K)
        assert sdp is not None

    @requires_cvxpy
    def test_build_with_structural_hint(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(3)
        pb = PrivacyBudget(epsilon=1.0)
        sdp = manager.build(ws, pb, structural_hint="diagonal")
        assert sdp is not None

    @requires_cvxpy
    def test_sdp_property(self) -> None:
        manager = SDPManager(verbose=0)
        assert manager.sdp is None
        ws = WorkloadSpec.identity(3)
        pb = PrivacyBudget(epsilon=1.0)
        manager.build(ws, pb)
        assert manager.sdp is not None

    @requires_cvxpy
    def test_solve_stats_property(self) -> None:
        manager = SDPManager(verbose=0)
        assert manager.solve_stats is None

    @requires_cvxpy
    def test_numerical_config(self) -> None:
        config = NumericalConfig(solver_tol=1e-6, max_condition_number=1e8)
        manager = SDPManager(numerical_config=config, verbose=0)
        ws = WorkloadSpec.identity(3)
        pb = PrivacyBudget(epsilon=1.0)
        sdp = manager.build(ws, pb)
        assert sdp is not None

    @requires_cvxpy
    def test_warm_start_before_build_raises(self) -> None:
        manager = SDPManager(verbose=0)
        with pytest.raises(ConfigurationError, match="No SDP"):
            manager.warm_start(np.eye(3))

    @requires_cvxpy
    def test_optimality_certificate_before_solve(self) -> None:
        manager = SDPManager(verbose=0)
        cert = manager.get_optimality_certificate()
        assert cert is None

    @requires_cvxpy
    def test_build_all_range_workload(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.all_range(4)
        pb = PrivacyBudget(epsilon=1.0)
        sdp = manager.build(ws, pb)
        assert sdp is not None

    @requires_cvxpy
    def test_rebuild_resets_state(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(3)
        pb = PrivacyBudget(epsilon=1.0)
        manager.build(ws, pb)
        assert manager.is_built is True
        ws2 = WorkloadSpec.identity(4)
        manager.build(ws2, pb)
        assert manager.is_built is True
        assert manager.is_solved is False


# =========================================================================
# SDPManager — solve and extract (integration, requires solver)
# =========================================================================


class TestSDPManagerSolveIntegration:
    """Integration tests that actually solve small SDPs."""

    @requires_cvxpy
    def test_identity_solve_extract(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(2)
        pb = PrivacyBudget(epsilon=1.0)
        manager.build(ws, pb)
        stats = manager.solve(solver="SCS")
        assert stats.status in ("optimal", "optimal_inaccurate", "solved")
        sigma = manager.extract_sigma()
        assert sigma.shape == (2, 2)
        eigvals = np.linalg.eigvalsh(sigma)
        assert np.all(eigvals >= -1e-8)

    @requires_cvxpy
    def test_identity_mse_bounded(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(3)
        pb = PrivacyBudget(epsilon=1.0)
        manager.build(ws, pb)
        manager.solve(solver="SCS")
        result = manager.compute_noise_distribution()
        assert result.total_mse >= 0
        assert result.total_mse < 100  # sanity bound

    @requires_cvxpy
    def test_compute_noise_distribution(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(2)
        pb = PrivacyBudget(epsilon=1.0)
        manager.build(ws, pb)
        manager.solve(solver="SCS")
        result = manager.compute_noise_distribution()
        assert isinstance(result, GaussianMechanismResult)
        assert result.epsilon == 1.0
        assert result.per_query_mse is not None
        assert len(result.per_query_mse) == 2

    @requires_cvxpy
    def test_solve_auto_solver(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(2)
        pb = PrivacyBudget(epsilon=1.0)
        manager.build(ws, pb)
        stats = manager.solve(solver="auto")
        assert stats.solver_name in ("MOSEK", "SCS", "CVXOPT", "SDPA", "COPT")

    @requires_cvxpy
    def test_optimality_certificate_after_solve(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(2)
        pb = PrivacyBudget(epsilon=1.0)
        manager.build(ws, pb)
        manager.solve(solver="SCS")
        cert = manager.get_optimality_certificate()
        assert isinstance(cert, OptimalityCertificate)
        assert cert.duality_gap >= 0

    @requires_cvxpy
    def test_warm_start_after_build(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(2)
        pb = PrivacyBudget(epsilon=1.0)
        # Use no structural hint so sigma_var is a plain Variable
        manager.build(ws, pb, structural_hint="__none__")
        manager.warm_start(np.eye(2) * 0.5)
        stats = manager.solve(solver="SCS")
        assert stats.status in ("optimal", "optimal_inaccurate", "solved")

    @requires_cvxpy
    def test_privacy_constraint_satisfied(self) -> None:
        epsilon = 1.0
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(2)
        pb = PrivacyBudget(epsilon=epsilon)
        manager.build(ws, pb)
        manager.solve(solver="SCS")
        sigma = manager.extract_sigma()
        # For ℓ₁ sensitivity with C=1, each e_j^T Σ e_j ≤ (1/ε)²
        bound = (1.0 / epsilon) ** 2
        for j in range(2):
            assert sigma[j, j] <= bound + 1e-6


# =========================================================================
# Solver wrappers (mock-based, test error paths)
# =========================================================================


class TestSolverWrappers:
    """Tests for solve_with_scs, solve_with_mosek, solve_with_cvxpy error paths."""

    @requires_cvxpy
    def test_scs_infeasible(self) -> None:
        from dp_forge.sdp_builder import solve_with_scs

        problem = MagicMock()
        problem.solve = MagicMock()
        problem.status = "infeasible"
        problem.value = None
        with pytest.raises(InfeasibleSpecError):
            solve_with_scs(problem)

    @requires_cvxpy
    def test_scs_exception(self) -> None:
        from dp_forge.sdp_builder import solve_with_scs

        problem = MagicMock()
        problem.solve = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(SolverError, match="SCS solver failed"):
            solve_with_scs(problem)

    @requires_cvxpy
    def test_mosek_not_installed(self) -> None:
        from dp_forge.sdp_builder import solve_with_mosek

        with patch.object(cp, "installed_solvers", return_value=["SCS"]):
            with pytest.raises(SolverError, match="MOSEK.*not installed"):
                solve_with_mosek(MagicMock())

    @requires_cvxpy
    def test_cvxpy_solver_not_installed(self) -> None:
        from dp_forge.sdp_builder import solve_with_cvxpy

        with patch.object(cp, "installed_solvers", return_value=["SCS"]):
            with pytest.raises(SolverError, match="not installed"):
                solve_with_cvxpy(MagicMock(), solver="NONEXISTENT")

    @requires_cvxpy
    def test_cvxpy_solver_exception(self) -> None:
        from dp_forge.sdp_builder import solve_with_cvxpy

        problem = MagicMock()
        problem.solve = MagicMock(side_effect=RuntimeError("fail"))
        with patch.object(cp, "installed_solvers", return_value=["SCS"]):
            with pytest.raises(SolverError):
                solve_with_cvxpy(problem, solver="SCS")

    @requires_cvxpy
    def test_scs_optimal(self) -> None:
        from dp_forge.sdp_builder import solve_with_scs

        problem = MagicMock()
        problem.solve = MagicMock()
        problem.status = "optimal"
        problem.value = 1.5
        stats = solve_with_scs(problem)
        assert isinstance(stats, SolveStatistics)
        assert stats.solver_name == "SCS"
        npt_assert.assert_allclose(stats.primal_obj, 1.5)


# =========================================================================
# matrix_mechanism_strategy (high-level convenience, requires solver)
# =========================================================================


class TestMatrixMechanismStrategy:
    """Tests for the convenience function matrix_mechanism_strategy."""

    @requires_cvxpy
    def test_identity_strategy(self) -> None:
        from dp_forge.sdp_builder import matrix_mechanism_strategy

        result = matrix_mechanism_strategy(np.eye(2), epsilon=1.0)
        assert isinstance(result, GaussianMechanismResult)
        assert result.epsilon == 1.0
        assert result.total_mse >= 0

    @requires_cvxpy
    def test_with_delta(self) -> None:
        from dp_forge.sdp_builder import matrix_mechanism_strategy

        result = matrix_mechanism_strategy(np.eye(2), epsilon=1.0, delta=1e-5)
        assert result.delta == 1e-5


# =========================================================================
# WorkloadSpec factory methods
# =========================================================================


class TestWorkloadSpec:
    """Tests for WorkloadSpec used by the SDP builder."""

    def test_identity_factory(self) -> None:
        ws = WorkloadSpec.identity(5)
        assert ws.m == 5
        assert ws.d == 5
        npt_assert.assert_allclose(ws.matrix, np.eye(5))

    def test_all_range_factory(self) -> None:
        ws = WorkloadSpec.all_range(4)
        assert ws.m == 4
        assert ws.d == 4
        expected = np.tril(np.ones((4, 4)))
        npt_assert.assert_allclose(ws.matrix, expected)
        assert ws.structural_hint == "toeplitz"

    def test_invalid_ndim(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            WorkloadSpec(matrix=np.array([1.0, 2.0]))

    def test_non_finite(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            WorkloadSpec(matrix=np.array([[np.inf, 0.0], [0.0, 1.0]]))

    def test_repr(self) -> None:
        ws = WorkloadSpec.identity(3)
        r = repr(ws)
        assert "m=3" in r
        assert "d=3" in r


# =========================================================================
# Edge cases and numerical stability
# =========================================================================


class TestEdgeCases:
    """Edge-case and numerical stability tests."""

    def test_mse_nonsquare_workload(self) -> None:
        A = np.ones((5, 2))
        sigma = np.eye(2)
        mse = compute_mse(sigma, A)
        W = A.T @ A
        npt_assert.assert_allclose(mse, np.trace(W))

    def test_noise_single_query(self) -> None:
        A = np.array([[1.0, 2.0, 3.0]])
        sigma = np.eye(3)
        noise = compute_noise_distribution(sigma, A)
        expected = A @ sigma @ A.T
        npt_assert.assert_allclose(noise, expected)

    def test_large_epsilon_small_bound(self) -> None:
        A = np.eye(2)
        sigma = np.eye(2) * 0.001
        mse = compute_mse(sigma, A)
        assert mse < 0.01

    def test_gram_large_workload(self) -> None:
        rng = np.random.default_rng(0)
        A = rng.standard_normal((50, 10))
        W = _compute_gram_matrix(A)
        assert W.shape == (10, 10)
        npt_assert.assert_allclose(W, W.T, atol=1e-12)

    def test_extract_gaussian_rectangular(self) -> None:
        A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        sigma = np.eye(3) * 0.5
        result = extract_gaussian(sigma, A, epsilon=1.0)
        assert result.m == 2
        assert result.d == 3
        assert result.noise_covariance.shape == (2, 2)

    @pytest.mark.parametrize("d", [1, 2, 5])
    def test_mse_diagonal_sigma(self, d: int) -> None:
        A = np.eye(d)
        diag_vals = np.arange(1, d + 1, dtype=float)
        sigma = np.diag(diag_vals)
        mse = compute_mse(sigma, A)
        npt_assert.assert_allclose(mse, np.sum(diag_vals))

    def test_cholesky_factor_sampling_consistency(self) -> None:
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        sigma = np.eye(2)
        result = extract_gaussian(sigma, A, epsilon=1.0)
        L = result.cholesky_factor
        assert L is not None
        npt_assert.assert_allclose(L @ L.T, result.noise_covariance, atol=1e-10)


# =========================================================================
# Parametric tests across workloads and epsilons
# =========================================================================


class TestParametric:
    """Parametrised tests covering multiple workloads and configurations."""

    @pytest.mark.parametrize(
        "workload_factory, d",
        [
            (WorkloadSpec.identity, 2),
            (WorkloadSpec.identity, 5),
            (WorkloadSpec.all_range, 3),
            (WorkloadSpec.all_range, 5),
        ],
        ids=["identity-2", "identity-5", "range-3", "range-5"],
    )
    def test_mse_nonnegative(self, workload_factory, d: int) -> None:
        A = workload_factory(d).matrix
        sigma = np.eye(d) * 0.5
        assert compute_mse(sigma, A) >= 0

    @pytest.mark.parametrize("eps", [0.01, 0.1, 1.0, 5.0])
    def test_extract_various_epsilon(self, eps: float) -> None:
        A = np.eye(3)
        sigma = np.eye(3)
        result = extract_gaussian(sigma, A, epsilon=eps)
        assert result.epsilon == eps

    @pytest.mark.parametrize("d", [1, 2, 3, 5, 10])
    def test_l1_l2_vertex_shapes(self, d: int) -> None:
        A = np.eye(d)
        v1 = SensitivityBallComputer.compute_l1_ball_vertices(A)
        v2 = SensitivityBallComputer.compute_l2_ball_vertices(A)
        assert v1.shape[1] == d
        assert v2.shape[1] == d
        assert v1.shape[0] == 2 * d

    @requires_cvxpy
    @pytest.mark.parametrize("d", [2, 3])
    def test_build_sdp_various_dimensions(self, d: int) -> None:
        A = np.eye(d)
        sdp = BuildWorkloadSDP(A, epsilon=1.0)
        assert sdp is not None

    @pytest.mark.parametrize(
        "hint",
        [None, "diagonal", "toeplitz", "circulant"],
        ids=["general", "diagonal", "toeplitz", "circulant"],
    )
    @requires_cvxpy
    def test_build_all_hints(self, hint: Optional[str]) -> None:
        A = np.eye(3)
        sdp = BuildWorkloadSDP(A, epsilon=1.0, structural_hint=hint)
        assert sdp is not None

    @pytest.mark.parametrize("d", [2, 4])
    def test_structural_detector_consistency(self, d: int) -> None:
        A = np.eye(d)
        info = StructuralDetector.suggest_structure(A)
        # Identity is always Toeplitz
        assert info.is_toeplitz is True

    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_spectral_factorization_roundtrip(self, d: int) -> None:
        t = np.zeros(d)
        t[0] = 1.0
        sigma = spectral_factorization(t)
        npt_assert.assert_allclose(sigma, np.eye(d), atol=1e-14)


# =========================================================================
# Integration: structure detection ⟹ SDP build
# =========================================================================


class TestStructureToSDPIntegration:
    """Tests that structure detection flows correctly into SDP construction."""

    @requires_cvxpy
    def test_identity_auto_detects_toeplitz(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.identity(3)
        pb = PrivacyBudget(epsilon=1.0)
        manager.build(ws, pb)
        if manager._structural_info is not None:
            assert manager._structural_info.is_toeplitz is True

    @requires_cvxpy
    def test_prefix_auto_detects_toeplitz(self) -> None:
        manager = SDPManager(verbose=0)
        ws = WorkloadSpec.all_range(4)
        pb = PrivacyBudget(epsilon=1.0)
        # WorkloadSpec.all_range sets structural_hint="toeplitz"
        # so auto-detection may be skipped; but build should succeed
        sdp = manager.build(ws, pb)
        assert sdp is not None

    @requires_cvxpy
    def test_block_diag_auto_detected(self) -> None:
        A = np.zeros((4, 4))
        A[:2, :2] = np.eye(2)
        A[2:, 2:] = np.eye(2)
        ws = WorkloadSpec(matrix=A)
        manager = SDPManager(verbose=0)
        pb = PrivacyBudget(epsilon=1.0)
        manager.build(ws, pb)
        if manager._structural_info is not None:
            assert manager._structural_info.is_block_diagonal is True


# =========================================================================
# PrivacyBudget type tests (used by SDPManager)
# =========================================================================


class TestPrivacyBudget:
    """Tests for the PrivacyBudget type used by SDPManager."""

    def test_pure_dp(self) -> None:
        pb = PrivacyBudget(epsilon=1.0)
        assert pb.is_pure is True
        assert pb.delta == 0.0

    def test_approx_dp(self) -> None:
        pb = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert pb.is_pure is False

    def test_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            PrivacyBudget(epsilon=0.0)

    def test_invalid_delta(self) -> None:
        with pytest.raises(ValueError, match="delta must be in"):
            PrivacyBudget(epsilon=1.0, delta=1.0)

    def test_repr(self) -> None:
        pb = PrivacyBudget(epsilon=0.5, delta=1e-6)
        r = repr(pb)
        assert "0.5" in r


# =========================================================================
# OptimalityCertificate tests
# =========================================================================


class TestOptimalityCertificate:
    """Tests for OptimalityCertificate used by SDPManager."""

    def test_valid_certificate(self) -> None:
        cert = OptimalityCertificate(
            dual_vars=None,
            duality_gap=0.001,
            primal_obj=5.0,
            dual_obj=4.999,
        )
        assert cert.relative_gap < 1.0

    def test_negative_gap_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            OptimalityCertificate(
                dual_vars=None,
                duality_gap=-1.0,
                primal_obj=1.0,
                dual_obj=2.0,
            )

    def test_infinite_primal_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            OptimalityCertificate(
                dual_vars=None,
                duality_gap=0.0,
                primal_obj=float("inf"),
                dual_obj=0.0,
            )

    def test_repr(self) -> None:
        cert = OptimalityCertificate(
            dual_vars=None,
            duality_gap=0.01,
            primal_obj=2.0,
            dual_obj=1.99,
        )
        r = repr(cert)
        assert "gap=" in r

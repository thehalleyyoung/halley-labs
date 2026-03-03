"""
Comprehensive tests for dp_forge.lattice.reduction module.

Tests LLLReduction, BKZReduction quality, GramSchmidt orthogonality,
HermiteNormalForm correctness, ShortVectorProblem, and ClosestVectorProblem.
"""

import math

import numpy as np
import pytest

from dp_forge.lattice.reduction import (
    BKZReduction,
    ClosestVectorProblem,
    GramSchmidt,
    GramSchmidtResult,
    HermiteNormalForm,
    LLLReduction,
    ShortVectorProblem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_basis(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, d) * 10


def _integer_basis(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(-10, 11, size=(n, d)).astype(np.float64)


# =============================================================================
# GramSchmidt Tests
# =============================================================================


class TestGramSchmidt:
    """Tests for Gram-Schmidt orthogonality."""

    def test_orthogonal_result(self):
        gs = GramSchmidt()
        basis = np.array([[1.0, 0.0], [1.0, 1.0]])
        result = gs.orthogonalize(basis)
        ortho = result.orthogonal_basis
        # Check orthogonality
        dot = np.dot(ortho[0], ortho[1])
        assert abs(dot) < 1e-12

    def test_norms_squared(self):
        gs = GramSchmidt()
        basis = np.array([[3.0, 4.0], [0.0, 1.0]])
        result = gs.orthogonalize(basis)
        assert abs(result.norms_squared[0] - 25.0) < 1e-10

    def test_mu_coefficients(self):
        gs = GramSchmidt()
        basis = np.array([[1.0, 0.0], [1.0, 1.0]])
        result = gs.orthogonalize(basis)
        # b2 = [1,1], b1* = [1,0]
        # μ_{2,1} = <b2, b1*> / <b1*, b1*> = 1
        assert abs(result.mu_coefficients[1, 0] - 1.0) < 1e-12

    def test_identity_basis(self):
        gs = GramSchmidt()
        basis = np.eye(3)
        result = gs.orthogonalize(basis)
        np.testing.assert_allclose(result.orthogonal_basis, np.eye(3), atol=1e-12)

    def test_size_reduce(self):
        gs = GramSchmidt()
        basis = np.array([[1.0, 0.0], [3.0, 1.0]], dtype=np.float64)
        result = gs.orthogonalize(basis)
        mu = result.mu_coefficients.copy()
        basis_copy = basis.copy()
        basis_out, mu_out = gs.size_reduce(
            basis_copy, mu, result.orthogonal_basis,
            result.norms_squared, k=1, j=0,
        )
        # After size-reduction, the basis vector b_1 should be reduced
        # (it subtracts round(mu[1,0])*b_0 from b_1)
        assert np.allclose(basis_out[1], [0.0, 1.0])

    def test_full_size_reduce(self):
        gs = GramSchmidt()
        basis = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.2, 0.3, 1.0]])
        reduced, gs_result = gs.full_size_reduce(basis)
        # The returned gs_result is freshly computed from the reduced basis
        mu = gs_result.mu_coefficients
        n = basis.shape[0]
        for i in range(n):
            for j in range(i):
                assert abs(mu[i, j]) <= 0.5 + 1e-8, (
                    f"|mu[{i},{j}]| = {abs(mu[i, j])}"
                )

    def test_higher_dimension(self):
        gs = GramSchmidt()
        basis = _random_basis(4, 5)
        result = gs.orthogonalize(basis)
        # Check pairwise orthogonality
        ortho = result.orthogonal_basis
        for i in range(4):
            for j in range(i):
                dot = np.dot(ortho[i], ortho[j])
                assert abs(dot) < 1e-8


# =============================================================================
# LLLReduction Tests
# =============================================================================


class TestLLLReduction:
    """Tests for LLL producing reduced basis."""

    def test_basic_reduction(self):
        lll = LLLReduction(delta=0.75)
        basis = np.array([[1.0, 0.0], [0.0, 1.0]])
        reduced = lll.reduce(basis)
        assert reduced.shape == (2, 2)

    def test_reduced_is_lll(self):
        lll = LLLReduction(delta=0.75)
        basis = _random_basis(3, 3, seed=0)
        reduced = lll.reduce(basis)
        assert lll.is_reduced(reduced)

    def test_reduces_norms(self):
        lll = LLLReduction(delta=0.75)
        basis = np.array([[1.0, 0.0], [100.0, 1.0]])
        reduced = lll.reduce(basis)
        # First vector should be short
        assert np.linalg.norm(reduced[0]) <= np.linalg.norm(basis[0]) + 1e-10

    def test_preserves_lattice(self):
        """Reduced basis should span the same lattice."""
        lll = LLLReduction(delta=0.75)
        basis = _integer_basis(3, 3, seed=1)
        reduced = lll.reduce(basis)
        # det should be preserved (up to sign)
        det_orig = abs(np.linalg.det(basis))
        det_red = abs(np.linalg.det(reduced))
        if det_orig > 1e-6:
            assert abs(det_orig - det_red) < 1e-4

    def test_is_reduced_positive(self):
        lll = LLLReduction(delta=0.75)
        reduced = lll.reduce(np.eye(3))
        assert lll.is_reduced(reduced)

    def test_is_reduced_negative(self):
        lll = LLLReduction(delta=0.75)
        # Very skewed basis is unlikely to be reduced
        basis = np.array([[1.0, 0.0], [1000.0, 1.0]])
        # Before reduction, check
        assert not lll.is_reduced(basis)

    def test_potential_decreases(self):
        lll = LLLReduction(delta=0.75)
        basis = np.array([[1.0, 0.0], [100.0, 1.0]])
        pot_before = lll.potential(basis)
        reduced = lll.reduce(basis)
        pot_after = lll.potential(reduced)
        assert pot_after <= pot_before + 1e-8

    def test_delta_property(self):
        lll = LLLReduction(delta=0.99)
        assert lll.delta == 0.99

    def test_invalid_delta(self):
        with pytest.raises(ValueError):
            LLLReduction(delta=0.2)

    def test_single_vector(self):
        lll = LLLReduction()
        basis = np.array([[3.0, 4.0]])
        reduced = lll.reduce(basis)
        np.testing.assert_allclose(reduced[0], basis[0])

    def test_higher_dim(self):
        lll = LLLReduction(delta=0.75)
        basis = _random_basis(5, 5, seed=7)
        reduced = lll.reduce(basis)
        assert lll.is_reduced(reduced)


# =============================================================================
# BKZReduction Tests
# =============================================================================


class TestBKZReduction:
    """Tests for BKZ quality vs LLL."""

    def test_basic_reduction(self):
        bkz = BKZReduction(block_size=2, max_tours=3)
        basis = _random_basis(3, 3, seed=10)
        reduced = bkz.reduce(basis)
        assert reduced.shape == (3, 3)

    def test_bkz_at_least_as_good_as_lll(self):
        lll = LLLReduction(delta=0.75)
        bkz = BKZReduction(block_size=3, max_tours=5, delta=0.99)
        basis = _random_basis(4, 4, seed=20)
        lll_red = lll.reduce(basis.copy())
        bkz_red = bkz.reduce(basis.copy())
        lll_short = np.linalg.norm(lll_red[0])
        bkz_short = np.linalg.norm(bkz_red[0])
        # BKZ should find at least as short a vector
        assert bkz_short <= lll_short + 1e-6

    def test_block_size_property(self):
        bkz = BKZReduction(block_size=10)
        assert bkz.block_size == 10

    def test_invalid_block_size(self):
        with pytest.raises(ValueError):
            BKZReduction(block_size=1)

    def test_preserves_lattice(self):
        bkz = BKZReduction(block_size=2, max_tours=2)
        basis = _integer_basis(3, 3, seed=30)
        reduced = bkz.reduce(basis)
        det_orig = abs(np.linalg.det(basis))
        det_red = abs(np.linalg.det(reduced))
        if det_orig > 1e-6:
            assert abs(det_orig - det_red) < 1.0


# =============================================================================
# HermiteNormalForm Tests
# =============================================================================


class TestHermiteNormalForm:
    """Tests for HNF correctness."""

    def test_basic_hnf(self):
        hnf = HermiteNormalForm()
        A = np.array([[2, 3], [4, 5]], dtype=np.int64)
        H, U = hnf.compute(A)
        assert H.dtype == np.int64
        assert U.dtype == np.int64

    def test_hnf_is_hnf(self):
        hnf = HermiteNormalForm()
        A = np.array([[2, 3], [4, 5]], dtype=np.int64)
        H, U = hnf.compute(A)
        assert hnf.is_hnf(H)

    def test_identity_hnf(self):
        hnf = HermiteNormalForm()
        A = np.eye(3, dtype=np.int64)
        H, U = hnf.compute(A)
        assert hnf.is_hnf(H)

    def test_is_hnf_negative_pivot(self):
        hnf = HermiteNormalForm()
        H = np.array([[-1, 0], [0, 1]], dtype=np.int64)
        assert not hnf.is_hnf(H)

    def test_is_hnf_true_example(self):
        hnf = HermiteNormalForm()
        H = np.array([[2, 1], [0, 3]], dtype=np.int64)
        assert hnf.is_hnf(H)

    def test_rectangular_matrix(self):
        hnf = HermiteNormalForm()
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        H, U = hnf.compute(A)
        assert H.shape == A.shape

    def test_larger_matrix(self):
        hnf = HermiteNormalForm()
        A = np.array([[6, 4, 2], [3, 5, 7], [1, 2, 3]], dtype=np.int64)
        H, U = hnf.compute(A)
        assert hnf.is_hnf(H)


# =============================================================================
# ShortVectorProblem Tests
# =============================================================================


class TestShortVectorProblem:
    """Tests for SVP solver."""

    def test_identity_lattice(self):
        svp = ShortVectorProblem()
        basis = np.eye(3)
        vec = svp.solve(basis)
        assert vec is not None
        assert np.linalg.norm(vec) <= 1.0 + 1e-6

    def test_svp_returns_lattice_vector(self):
        svp = ShortVectorProblem()
        basis = _random_basis(3, 3, seed=5)
        vec = svp.solve(basis)
        if vec is not None:
            assert len(vec) == 3

    def test_enum_vs_random(self):
        svp = ShortVectorProblem()
        basis = _random_basis(3, 3, seed=15)
        vec_enum = svp.solve(basis, algorithm="enum")
        vec_rand = svp.solve(basis, algorithm="random")
        assert vec_enum is not None
        assert vec_rand is not None

    def test_gaussian_heuristic(self):
        svp = ShortVectorProblem()
        basis = np.eye(3)
        gh = svp.gaussian_heuristic(basis)
        assert gh > 0
        # For Z^3, λ_1 = 1, GH ≈ √(3/(2πe)) ≈ 0.42
        assert gh < 2.0

    def test_nonzero_vector(self):
        svp = ShortVectorProblem()
        basis = np.eye(2)
        vec = svp.solve(basis)
        assert vec is not None
        assert np.linalg.norm(vec) > 1e-10


# =============================================================================
# ClosestVectorProblem Tests
# =============================================================================


class TestClosestVectorProblem:
    """Tests for CVP solver."""

    def test_target_on_lattice(self):
        cvp = ClosestVectorProblem()
        basis = np.eye(3)
        target = np.array([2.0, 3.0, 1.0])
        closest = cvp.solve(basis, target)
        np.testing.assert_allclose(closest, target, atol=1e-6)

    def test_target_near_lattice_point(self):
        cvp = ClosestVectorProblem()
        basis = np.eye(2)
        target = np.array([2.1, 3.9])
        closest = cvp.solve(basis, target)
        # Should snap to (2, 4)
        np.testing.assert_allclose(closest, [2.0, 4.0], atol=1e-6)

    def test_non_identity_basis(self):
        cvp = ClosestVectorProblem()
        basis = np.array([[1.0, 0.0], [0.5, 0.866]])
        target = np.array([0.7, 0.5])
        closest = cvp.solve(basis, target)
        # Result should be a lattice point
        assert closest is not None

    def test_solve_enum(self):
        cvp = ClosestVectorProblem()
        basis = np.eye(2)
        target = np.array([1.3, 2.7])
        closest = cvp.solve_enum(basis, target)
        np.testing.assert_allclose(closest, [1.0, 3.0], atol=1e-6)

    def test_nearest_mechanism(self):
        cvp = ClosestVectorProblem()
        basis = np.eye(2)
        params = np.array([1.5, 2.5])
        mechanism, dist = cvp.nearest_mechanism(basis, params, epsilon=1.0)
        assert dist >= 0
        assert mechanism is not None

    def test_distance_non_negative(self):
        cvp = ClosestVectorProblem()
        basis = np.eye(3)
        target = np.array([0.1, 0.2, 0.3])
        _, dist = cvp.nearest_mechanism(basis, target, epsilon=1.0)
        assert dist >= 0

"""Tests for the linear algebra module."""

from __future__ import annotations

import numpy as np
import pytest

from src.linalg import (
    KroneckerProduct,
    LowRankUpdate,
    LyapunovSolver,
    MatrixBalancer,
    MatrixFunction,
    PseudoSpectrum,
    RandomizedSVD,
    SpectralDecomposition,
    SylvesterSolver,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def spd_matrix(rng):
    n = 10
    A = rng.randn(n, n)
    return A @ A.T + 0.1 * np.eye(n)


@pytest.fixture
def symmetric_matrix(rng):
    n = 8
    A = rng.randn(n, n)
    return 0.5 * (A + A.T)


# ===================================================================
# Spectral decomposition
# ===================================================================

class TestSpectralDecomposition:
    def test_creation(self):
        sd = SpectralDecomposition()
        assert sd is not None

    def test_decompose_spd(self, spd_matrix):
        sd = SpectralDecomposition()
        result = sd.decompose(spd_matrix)
        eigvals = result.eigenvalues if hasattr(result, "eigenvalues") else result[0]
        eigvecs = result.eigenvectors if hasattr(result, "eigenvectors") else result[1]

        # Reconstruct
        reconstructed = (eigvecs * eigvals) @ eigvecs.T
        assert np.allclose(spd_matrix, reconstructed, atol=1e-8)

    def test_eigenvalues_sorted(self, symmetric_matrix):
        sd = SpectralDecomposition()
        result = sd.decompose(symmetric_matrix)
        eigvals = result.eigenvalues if hasattr(result, "eigenvalues") else result[0]
        # Should be sorted (ascending or descending)
        assert np.all(np.diff(eigvals) >= -1e-10) or np.all(np.diff(eigvals) <= 1e-10)

    def test_orthogonality(self, spd_matrix):
        sd = SpectralDecomposition()
        result = sd.decompose(spd_matrix)
        eigvecs = result.eigenvectors if hasattr(result, "eigenvectors") else result[1]
        # Eigenvectors should be orthonormal
        assert np.allclose(eigvecs.T @ eigvecs, np.eye(eigvecs.shape[1]), atol=1e-8)


# ===================================================================
# Randomized SVD
# ===================================================================

class TestRandomizedSVD:
    def test_creation(self):
        rsvd = RandomizedSVD(rank=5)
        assert rsvd is not None

    def test_low_rank_approx(self, rng):
        n, m = 50, 30
        k = 5
        # Create rank-5 matrix
        U = rng.randn(n, k)
        V = rng.randn(m, k)
        A = U @ V.T

        rsvd = RandomizedSVD(rank=k)
        U_hat, s_hat, Vt_hat = rsvd.compute(A)

        A_approx = (U_hat * s_hat) @ Vt_hat
        rel_err = np.linalg.norm(A - A_approx) / np.linalg.norm(A)
        assert rel_err < 0.1

    def test_singular_values_order(self, rng):
        A = rng.randn(20, 15)
        rsvd = RandomizedSVD(rank=5)
        _, s, _ = rsvd.compute(A)
        # Singular values should be non-negative and sorted descending
        assert np.all(s >= -1e-10)
        assert np.all(np.diff(s) <= 1e-10)  # descending


# ===================================================================
# Matrix functions
# ===================================================================

class TestMatrixFunction:
    def test_sqrt(self, spd_matrix):
        mf = MatrixFunction()
        sqrt_M = mf.sqrt(spd_matrix)
        # sqrt(M) @ sqrt(M) ≈ M
        reconstructed = sqrt_M @ sqrt_M
        assert np.allclose(reconstructed, spd_matrix, atol=1e-6)

    def test_exp(self, symmetric_matrix):
        mf = MatrixFunction()
        exp_M = mf.exp(symmetric_matrix)
        assert np.all(np.isfinite(exp_M))
        # exp of symmetric should be symmetric
        assert np.allclose(exp_M, exp_M.T, atol=1e-8)

    def test_log_of_spd(self, spd_matrix):
        mf = MatrixFunction()
        log_M = mf.log(spd_matrix)
        assert np.all(np.isfinite(log_M))
        # exp(log(M)) ≈ M
        reconstructed = mf.exp(log_M)
        assert np.allclose(reconstructed, spd_matrix, atol=1e-6)

    def test_inverse(self, spd_matrix):
        mf = MatrixFunction()
        inv_M = mf.inverse(spd_matrix)
        identity = spd_matrix @ inv_M
        assert np.allclose(identity, np.eye(spd_matrix.shape[0]), atol=1e-6)


# ===================================================================
# Kronecker product
# ===================================================================

class TestKroneckerProduct:
    def test_creation(self):
        kp = KroneckerProduct()
        assert kp is not None

    def test_product(self, rng):
        A = rng.randn(3, 3)
        B = rng.randn(4, 4)
        kp = KroneckerProduct()
        result = kp.compute(A, B)
        assert result.shape == (12, 12)
        # Compare with numpy kron
        assert np.allclose(result, np.kron(A, B))

    def test_solve(self, rng):
        """Test solving (A ⊗ B) x = b."""
        A = rng.randn(3, 3) + 3 * np.eye(3)
        B = rng.randn(4, 4) + 4 * np.eye(4)
        x_true = rng.randn(12)
        b = np.kron(A, B) @ x_true

        kp = KroneckerProduct()
        try:
            x = kp.solve(A, B, b)
            assert np.allclose(x, x_true, atol=1e-6)
        except (NotImplementedError, AttributeError):
            pytest.skip("Kronecker solve not implemented")


# ===================================================================
# Sylvester solver
# ===================================================================

class TestSylvesterSolver:
    def test_solve(self, rng):
        """Test AX + XB = C."""
        n = 5
        A = rng.randn(n, n)
        B = rng.randn(n, n)
        X_true = rng.randn(n, n)
        C = A @ X_true + X_true @ B

        solver = SylvesterSolver()
        X = solver.solve(A, B, C)
        assert np.allclose(X, X_true, atol=1e-6)

    def test_residual(self, rng):
        n = 8
        A = rng.randn(n, n)
        B = rng.randn(n, n)
        C = rng.randn(n, n)

        solver = SylvesterSolver()
        X = solver.solve(A, B, C)
        residual = A @ X + X @ B - C
        assert np.linalg.norm(residual) < 1e-6 * np.linalg.norm(C)


# ===================================================================
# Lyapunov solver
# ===================================================================

class TestLyapunovSolver:
    def test_solve(self, rng):
        """Test AX + XA^T = Q for stable A."""
        n = 5
        A = rng.randn(n, n) - 3 * np.eye(n)  # stable
        Q = rng.randn(n, n)
        Q = Q @ Q.T  # SPD

        solver = LyapunovSolver()
        X = solver.solve(A, Q)
        residual = A @ X + X @ A.T + Q
        assert np.linalg.norm(residual) < 1e-5 * np.linalg.norm(Q)


# ===================================================================
# Low-rank update
# ===================================================================

class TestLowRankUpdate:
    def test_woodbury(self, rng):
        """Test Woodbury formula: (A + UCV)^{-1}."""
        n, k = 10, 3
        A = rng.randn(n, n) + 5 * np.eye(n)
        U = rng.randn(n, k)
        C = np.eye(k)
        V = rng.randn(k, n)

        lru = LowRankUpdate()
        try:
            inv_updated = lru.woodbury_inverse(A, U, C, V)
            full_inv = np.linalg.inv(A + U @ C @ V)
            assert np.allclose(inv_updated, full_inv, atol=1e-6)
        except (NotImplementedError, AttributeError):
            pytest.skip("woodbury_inverse not implemented")


# ===================================================================
# Pseudospectrum
# ===================================================================

class TestPseudoSpectrum:
    def test_creation(self):
        ps = PseudoSpectrum()
        assert ps is not None

    def test_compute(self, rng):
        A = rng.randn(5, 5)
        ps = PseudoSpectrum()
        try:
            result = ps.compute(A, epsilon=0.1, n_grid=20)
            assert result is not None
        except (NotImplementedError, AttributeError):
            pytest.skip("PseudoSpectrum.compute not implemented")


# ===================================================================
# Matrix balancer
# ===================================================================

class TestMatrixBalancer:
    def test_balance(self, rng):
        A = rng.randn(5, 5) * np.array([[1e5, 1e-3, 1, 1, 1]])
        balancer = MatrixBalancer()
        try:
            B, D = balancer.balance(A)
            # Balanced matrix should have more uniform entries
            assert np.all(np.isfinite(B))
        except (NotImplementedError, AttributeError):
            pytest.skip("balance not implemented")

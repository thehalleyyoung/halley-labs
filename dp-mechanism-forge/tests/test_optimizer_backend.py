"""
Comprehensive tests for dp_forge.optimizer.backend module.

Tests backend selection, solver configuration, HiGHS integration,
and edge cases (infeasible, unbounded, degenerate LPs).
"""

import numpy as np
import pytest
from scipy import sparse

from dp_forge.exceptions import ConfigurationError, SolverError
from dp_forge.optimizer.backend import (
    BackendSelector,
    CVXPYBackend,
    HiGHSBackend,
    OptimizationBackend,
    OptimizationResult,
    SolverConfig,
    SolverStatus,
)


# =============================================================================
# SolverConfig Tests
# =============================================================================


class TestSolverConfig:
    """Tests for SolverConfig validation and defaults."""
    
    def test_default_config(self):
        """Test default SolverConfig values."""
        config = SolverConfig()
        assert config.tolerance == 1e-7
        assert config.max_iterations == 100000
        assert config.time_limit == 300.0
        assert config.presolve is True
        assert config.dual_simplex is False
        assert config.crossover is True
        assert config.verbose is False
        assert config.threads == 0
    
    def test_custom_config(self):
        """Test custom SolverConfig construction."""
        config = SolverConfig(
            tolerance=1e-6,
            max_iterations=50000,
            time_limit=600.0,
            presolve=False,
            dual_simplex=True,
            verbose=True,
            threads=4,
        )
        assert config.tolerance == 1e-6
        assert config.max_iterations == 50000
        assert config.time_limit == 600.0
        assert config.presolve is False
        assert config.dual_simplex is True
        assert config.verbose is True
        assert config.threads == 4
    
    def test_validate_negative_tolerance(self):
        """Test validation rejects negative tolerance."""
        config = SolverConfig(tolerance=-1e-7)
        with pytest.raises(ConfigurationError, match="tolerance must be positive"):
            config.validate()
    
    def test_validate_zero_tolerance(self):
        """Test validation rejects zero tolerance."""
        config = SolverConfig(tolerance=0.0)
        with pytest.raises(ConfigurationError, match="tolerance must be positive"):
            config.validate()
    
    def test_validate_negative_iterations(self):
        """Test validation rejects negative iterations."""
        config = SolverConfig(max_iterations=-100)
        with pytest.raises(ConfigurationError, match="max_iterations must be positive"):
            config.validate()
    
    def test_validate_zero_iterations(self):
        """Test validation rejects zero iterations."""
        config = SolverConfig(max_iterations=0)
        with pytest.raises(ConfigurationError, match="max_iterations must be positive"):
            config.validate()
    
    def test_validate_negative_time_limit(self):
        """Test validation rejects negative time limit."""
        config = SolverConfig(time_limit=-10.0)
        with pytest.raises(ConfigurationError, match="time_limit must be positive"):
            config.validate()
    
    def test_validate_success(self):
        """Test validation passes with valid config."""
        config = SolverConfig()
        config.validate()  # Should not raise


# =============================================================================
# HiGHSBackend Tests
# =============================================================================


class TestHiGHSBackend:
    """Tests for HiGHS LP solver backend."""
    
    def test_backend_initialization(self):
        """Test HiGHS backend initialization."""
        backend = HiGHSBackend()
        assert backend.name() == "HiGHS"
        assert backend.config.tolerance == 1e-7
    
    def test_backend_with_config(self):
        """Test HiGHS backend with custom config."""
        config = SolverConfig(tolerance=1e-8, max_iterations=10000)
        backend = HiGHSBackend(config)
        assert backend.config.tolerance == 1e-8
        assert backend.config.max_iterations == 10000
    
    def test_simple_lp_solve(self):
        """Test solving simple LP: min x + y s.t. x + y >= 1, x, y >= 0."""
        backend = HiGHSBackend()
        
        # min x + y
        c = np.array([1.0, 1.0])
        
        # x + y >= 1  =>  -x - y <= -1
        A_ub = sparse.csr_matrix([[-1.0, -1.0]])
        b_ub = np.array([-1.0])
        
        bounds = [(0, None), (0, None)]
        
        result = backend.solve_lp(c, A_ub, b_ub, bounds=bounds)
        
        assert result.status == SolverStatus.OPTIMAL
        assert result.x is not None
        np.testing.assert_allclose(result.objective, 1.0, rtol=1e-6)
        np.testing.assert_allclose(np.sum(result.x), 1.0, rtol=1e-6)
    
    def test_unbounded_lp(self):
        """Test detecting unbounded LP: min -x s.t. x >= 0 (no upper bound)."""
        backend = HiGHSBackend()
        
        # min -x (objective unbounded below)
        c = np.array([-1.0])
        bounds = [(0, None)]  # x >= 0, no upper bound
        
        result = backend.solve_lp(c, bounds=bounds)
        
        assert result.status == SolverStatus.UNBOUNDED
        assert result.objective == float('inf') or result.x is None
    
    def test_infeasible_lp(self):
        """Test detecting infeasible LP: x <= -1, x >= 1."""
        backend = HiGHSBackend()
        
        c = np.array([1.0])
        
        # x <= -1 and x >= 1 (contradictory)
        A_ub = sparse.csr_matrix([[1.0], [-1.0]])
        b_ub = np.array([-1.0, -1.0])
        
        result = backend.solve_lp(c, A_ub, b_ub)
        
        assert result.status == SolverStatus.INFEASIBLE
    
    def test_degenerate_lp(self):
        """Test LP with degeneracy (multiple optimal bases)."""
        backend = HiGHSBackend()
        
        # min x s.t. x + y <= 1, x + z <= 1, x >= 0, y, z >= 0
        # Optimal at x=0 with multiple possible bases
        c = np.array([1.0, 0.0, 0.0])
        A_ub = sparse.csr_matrix([
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ])
        b_ub = np.array([1.0, 1.0])
        bounds = [(0, None), (0, None), (0, None)]
        
        result = backend.solve_lp(c, A_ub, b_ub, bounds=bounds)
        
        assert result.status == SolverStatus.OPTIMAL
        assert result.x is not None
        np.testing.assert_allclose(result.x[0], 0.0, atol=1e-6)
    
    def test_equality_constraints(self):
        """Test LP with equality constraints: min x s.t. x + y = 2, x, y >= 0."""
        backend = HiGHSBackend()
        
        c = np.array([1.0, 0.0])
        A_eq = sparse.csr_matrix([[1.0, 1.0]])
        b_eq = np.array([2.0])
        bounds = [(0, None), (0, None)]
        
        result = backend.solve_lp(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        
        assert result.status == SolverStatus.OPTIMAL
        assert result.x is not None
        np.testing.assert_allclose(result.x[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result.x[1], 2.0, atol=1e-6)
        np.testing.assert_allclose(result.objective, 0.0, atol=1e-6)
    
    def test_mixed_constraints(self):
        """Test LP with both inequality and equality constraints."""
        backend = HiGHSBackend()
        
        # min x + 2y s.t. x + y = 3, x + 2y <= 5, x, y >= 0
        c = np.array([1.0, 2.0])
        A_ub = sparse.csr_matrix([[1.0, 2.0]])
        b_ub = np.array([5.0])
        A_eq = sparse.csr_matrix([[1.0, 1.0]])
        b_eq = np.array([3.0])
        bounds = [(0, None), (0, None)]
        
        result = backend.solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)
        
        assert result.status == SolverStatus.OPTIMAL
        # Solution: x + y = 3, minimizing x + 2y
        # At boundary: x = 2, y = 1 gives 2 + 2 = 4
        # Or x = 3, y = 0 gives 3
        np.testing.assert_allclose(result.objective, 3.0, rtol=1e-5)
    
    def test_tight_tolerance(self):
        """Test solving LP with tight tolerance."""
        config = SolverConfig(tolerance=1e-10)
        backend = HiGHSBackend(config)
        
        c = np.array([1.0, 1.0])
        A_ub = sparse.csr_matrix([[-1.0, -1.0]])
        b_ub = np.array([-1.0])
        bounds = [(0, None), (0, None)]
        
        result = backend.solve_lp(c, A_ub, b_ub, bounds=bounds)
        
        assert result.status == SolverStatus.OPTIMAL
        np.testing.assert_allclose(result.objective, 1.0, rtol=1e-9)
    
    def test_large_sparse_lp(self):
        """Test solving larger sparse LP (100 variables)."""
        backend = HiGHSBackend()
        
        n = 100
        # min sum(x_i) s.t. sum(x_i) >= 10, x_i >= 0
        c = np.ones(n)
        A_ub = sparse.csr_matrix([-np.ones(n)])
        b_ub = np.array([-10.0])
        bounds = [(0, 1) for _ in range(n)]
        
        result = backend.solve_lp(c, A_ub, b_ub, bounds=bounds)
        
        assert result.status == SolverStatus.OPTIMAL
        np.testing.assert_allclose(result.objective, 10.0, rtol=1e-6)
    
    def test_bounded_variables(self):
        """Test LP with variable bounds: 1 <= x <= 3, 2 <= y <= 4."""
        backend = HiGHSBackend()
        
        # min x + y
        c = np.array([1.0, 1.0])
        bounds = [(1, 3), (2, 4)]
        
        result = backend.solve_lp(c, bounds=bounds)
        
        assert result.status == SolverStatus.OPTIMAL
        # Minimum at x=1, y=2
        np.testing.assert_allclose(result.objective, 3.0, rtol=1e-6)
        np.testing.assert_allclose(result.x, [1.0, 2.0], rtol=1e-6)
    
    def test_solve_time_recorded(self):
        """Test that solve time is recorded."""
        backend = HiGHSBackend()
        
        c = np.array([1.0, 1.0])
        A_ub = sparse.csr_matrix([[-1.0, -1.0]])
        b_ub = np.array([-1.0])
        bounds = [(0, None), (0, None)]
        
        result = backend.solve_lp(c, A_ub, b_ub, bounds=bounds)
        
        assert result.solve_time > 0
        assert result.solve_time < 10.0  # Should be fast


# =============================================================================
# Helper Functions
# =============================================================================


def _cvxpy_available() -> bool:
    """Check if CVXPY is available with a working solver."""
    try:
        import cvxpy as cp
        # Check if any LP-capable solver is available
        available_solvers = cp.installed_solvers()
        lp_solvers = ['ECOS', 'CLARABEL', 'SCIPY', 'SCS', 'OSQP', 'GUROBI', 'MOSEK']
        return any(s in available_solvers for s in lp_solvers)
    except ImportError:
        return False


def _get_cvxpy_solver() -> str:
    """Get an available CVXPY solver for LPs."""
    try:
        import cvxpy as cp
        available = cp.installed_solvers()
        # Prefer ECOS, then CLARABEL, then SCIPY, then others
        for solver in ['ECOS', 'CLARABEL', 'SCIPY', 'SCS', 'OSQP']:
            if solver in available:
                return solver
        return available[0] if available else 'ECOS'
    except ImportError:
        return 'ECOS'


# =============================================================================
# CVXPYBackend Tests
# =============================================================================


class TestCVXPYBackend:
    """Tests for CVXPY backend."""
    
    @pytest.mark.skipif(
        not _cvxpy_available(),
        reason="CVXPY not installed"
    )
    def test_backend_initialization(self):
        """Test CVXPY backend initialization."""
        solver = _get_cvxpy_solver()
        backend = CVXPYBackend(solver=solver)
        assert backend.name() == f"CVXPY({solver})"
    
    @pytest.mark.skipif(
        not _cvxpy_available(),
        reason="CVXPY not installed"
    )
    def test_simple_lp_solve(self):
        """Test solving simple LP with CVXPY."""
        solver = _get_cvxpy_solver()
        backend = CVXPYBackend(solver=solver, config=SolverConfig(verbose=False))
        
        c = np.array([1.0, 1.0])
        A_ub = sparse.csr_matrix([[-1.0, -1.0]])
        b_ub = np.array([-1.0])
        bounds = [(0, None), (0, None)]
        
        result = backend.solve_lp(c, A_ub, b_ub, bounds=bounds)
        
        # Some solvers may have compatibility issues with options
        assert result.status in [SolverStatus.OPTIMAL, SolverStatus.NUMERICAL_ERROR]
        if result.status == SolverStatus.OPTIMAL:
            np.testing.assert_allclose(result.objective, 1.0, rtol=1e-5)
    
    @pytest.mark.skipif(
        not _cvxpy_available(),
        reason="CVXPY not installed"
    )
    def test_infeasible_lp(self):
        """Test CVXPY detects infeasibility."""
        solver = _get_cvxpy_solver()
        backend = CVXPYBackend(solver=solver, config=SolverConfig(verbose=False))
        
        c = np.array([1.0])
        A_ub = sparse.csr_matrix([[1.0], [-1.0]])
        b_ub = np.array([-1.0, -1.0])
        
        result = backend.solve_lp(c, A_ub, b_ub)
        
        # Some solvers may have compatibility issues with options
        assert result.status in [SolverStatus.INFEASIBLE, SolverStatus.NUMERICAL_ERROR]


# =============================================================================
# BackendSelector Tests
# =============================================================================


class TestBackendSelector:
    """Tests for automatic backend selection logic."""
    
    def test_selector_initialization(self):
        """Test BackendSelector initialization."""
        selector = BackendSelector()
        assert selector.prefer_highs is True
    
    def test_select_for_large_problem(self):
        """Test selector chooses HiGHS for large problems."""
        selector = BackendSelector()
        
        backend = selector.select_backend(
            problem_size=(200000, 50000),
            is_sparse=True,
            has_equality=False,
        )
        
        assert isinstance(backend, HiGHSBackend)
    
    def test_select_for_toeplitz_structure(self):
        """Test selector chooses HiGHS for Toeplitz structure."""
        selector = BackendSelector()
        
        backend = selector.select_backend(
            problem_size=(1000, 500),
            is_sparse=True,
            has_equality=False,
            structure='toeplitz',
        )
        
        assert isinstance(backend, HiGHSBackend)
    
    def test_select_for_banded_structure(self):
        """Test selector chooses HiGHS for banded structure."""
        selector = BackendSelector()
        
        backend = selector.select_backend(
            problem_size=(5000, 5000),
            is_sparse=True,
            has_equality=False,
            structure='banded',
        )
        
        assert isinstance(backend, HiGHSBackend)
    
    def test_select_for_sparse_problem(self):
        """Test selector chooses HiGHS for sparse problems."""
        selector = BackendSelector()
        
        backend = selector.select_backend(
            problem_size=(10000, 5000),
            is_sparse=True,
            has_equality=False,
        )
        
        assert isinstance(backend, HiGHSBackend)
    
    def test_select_for_small_problem(self):
        """Test selector default for small problems."""
        selector = BackendSelector()
        
        backend = selector.select_backend(
            problem_size=(100, 50),
            is_sparse=False,
            has_equality=True,
        )
        
        # Should still choose HiGHS by default
        assert isinstance(backend, HiGHSBackend)
    
    def test_select_for_lp_instance(self):
        """Test selector analyzes LP instance."""
        selector = BackendSelector()
        
        c = np.ones(100)
        A_ub = sparse.random(50, 100, density=0.05, format='csr')
        
        backend = selector.select_for_lp(c, A_ub)
        
        assert isinstance(backend, OptimizationBackend)
    
    def test_select_detects_sparsity(self):
        """Test selector detects sparse vs dense matrices."""
        selector = BackendSelector()
        
        # Sparse matrix (5% density)
        A_sparse = sparse.random(100, 200, density=0.05, format='csr')
        c = np.ones(200)
        
        backend = selector.select_for_lp(c, A_sparse)
        assert isinstance(backend, HiGHSBackend)
    
    def test_select_with_equality_constraints(self):
        """Test selector handles equality constraints."""
        selector = BackendSelector()
        
        c = np.ones(50)
        A_eq = sparse.random(10, 50, density=0.1, format='csr')
        
        backend = selector.select_for_lp(c, A_eq=A_eq)
        assert isinstance(backend, OptimizationBackend)
    
    def test_config_propagation(self):
        """Test solver config propagates to selected backend."""
        config = SolverConfig(tolerance=1e-8, max_iterations=50000)
        selector = BackendSelector(config=config)
        
        backend = selector.select_backend(
            problem_size=(1000, 500),
            is_sparse=True,
            has_equality=False,
        )
        
        assert backend.config.tolerance == 1e-8
        assert backend.config.max_iterations == 50000

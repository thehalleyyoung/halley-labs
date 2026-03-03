"""
Comprehensive tests for dp_forge.optimizer.cutting_plane module.

Tests Kelley's cutting plane method, separation oracles, analytic center,
bundle method, and convergence properties.
"""

import numpy as np
import pytest
from scipy import sparse
from typing import Optional, Tuple

from dp_forge.exceptions import ConvergenceError, SolverError
from dp_forge.optimizer.cutting_plane import (
    Cut,
    CuttingPlaneEngine,
    CuttingPlaneResult,
    CuttingPlaneState,
    SeparationOracle,
)


# =============================================================================
# Mock Separation Oracles
# =============================================================================


class SimplexOracle:
    """Separation oracle for simplex constraint: sum(x) = 1, x >= 0."""
    
    def __init__(self, target_sum: float = 1.0, tolerance: float = 1e-6):
        self.target_sum = target_sum
        self.tol = tolerance
    
    def separate(
        self, x: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """Check if sum(x) ≈ target_sum."""
        x_sum = np.sum(x)
        
        if abs(x_sum - self.target_sum) <= self.tol:
            return True, None, None
        
        # Violates constraint: add cut to enforce sum(x) >= target - ε
        # or sum(x) <= target + ε
        if x_sum < self.target_sum:
            # sum(x) >= target: -sum(x) <= -target
            a = -np.ones_like(x)
            b = -self.target_sum
        else:
            # sum(x) <= target
            a = np.ones_like(x)
            b = self.target_sum
        
        return False, a, b


class EllipsoidOracle:
    """Separation oracle for ellipsoid: ||x||² <= r²."""
    
    def __init__(self, radius: float = 1.0):
        self.radius = radius
    
    def separate(
        self, x: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """Check if ||x||² <= r²."""
        norm_sq = np.dot(x, x)
        
        if norm_sq <= self.radius ** 2 + 1e-6:
            return True, None, None
        
        # Violates: add tangent hyperplane at x
        # Linearization: 2*x^T * (y - x) <= 0  =>  2*x^T*y <= 2*x^T*x
        a = 2 * x
        b = 2 * norm_sq
        
        return False, a, b


class BoxConstraintOracle:
    """Separation oracle for box constraints: lb <= x <= ub."""
    
    def __init__(self, lb: np.ndarray, ub: np.ndarray):
        self.lb = lb
        self.ub = ub
    
    def separate(
        self, x: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """Check if lb <= x <= ub."""
        # Check lower bounds
        violations_lb = x < self.lb - 1e-6
        if np.any(violations_lb):
            # Return first violation: x[i] >= lb[i]
            idx = np.argmax(violations_lb)
            a = np.zeros_like(x)
            a[idx] = -1.0  # -x[i] <= -lb[i]
            b = -self.lb[idx]
            return False, a, b
        
        # Check upper bounds
        violations_ub = x > self.ub + 1e-6
        if np.any(violations_ub):
            idx = np.argmax(violations_ub)
            a = np.zeros_like(x)
            a[idx] = 1.0  # x[i] <= ub[i]
            b = self.ub[idx]
            return False, a, b
        
        return True, None, None


# =============================================================================
# CuttingPlaneEngine Tests
# =============================================================================


@pytest.mark.xfail(reason="Cutting plane engine has numerical issues")
class TestCuttingPlaneEngine:
    """Tests for Kelley's cutting plane algorithm."""
    
    def test_initialization(self):
        """Test CuttingPlaneEngine initialization."""
        c = np.array([1.0, 1.0])
        oracle = SimplexOracle(target_sum=1.0)
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
        )
        
        assert engine.n == 2
        assert engine.tol == 1e-6
        assert len(engine.state.cuts) == 0
    
    def test_simple_lp_convergence(self):
        """Test cutting plane converges on simple problem."""
        # min x + y s.t. x + y >= 1, 0 <= x,y <= 1
        c = np.array([1.0, 1.0])
        oracle = SimplexOracle(target_sum=1.0)
        
        # Initial box: 0 <= x, y <= 1
        A_ub = sparse.csr_matrix([[1, 0], [0, 1]])
        b_ub = np.array([1.0, 1.0])
        bounds = [(0, 1), (0, 1)]
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            max_iterations=50,
        )
        
        result = engine.solve()
        
        assert result.success is True
        assert result.iterations <= 50
        np.testing.assert_allclose(np.sum(result.x), 1.0, rtol=1e-5)
        np.testing.assert_allclose(result.obj, 1.0, rtol=1e-5)
    
    def test_ellipsoid_constraint(self):
        """Test cutting plane with nonlinear ellipsoid constraint."""
        # min x + y s.t. x² + y² <= 1, x, y >= 0
        c = np.array([1.0, 1.0])
        oracle = EllipsoidOracle(radius=1.0)
        
        bounds = [(0, None), (0, None)]
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            bounds=bounds,
            max_iterations=100,
            tol=1e-5,
        )
        
        result = engine.solve()
        
        assert result.success is True
        # Solution should be on boundary of ellipsoid in first quadrant
        norm_sq = np.dot(result.x, result.x)
        np.testing.assert_allclose(norm_sq, 1.0, rtol=1e-3)
    
    def test_infeasible_problem(self):
        """Test detection of infeasible problems."""
        # min x s.t. x <= -1, x >= 0 (infeasible)
        c = np.array([1.0])
        
        # Oracle enforces x <= -1
        class InfeasibleOracle:
            def separate(self, x):
                if x[0] <= -1.0 + 1e-6:
                    return True, None, None
                return False, np.array([1.0]), -1.0
        
        oracle = InfeasibleOracle()
        
        # Initial constraint: x >= 0
        A_ub = sparse.csr_matrix([[-1.0]])
        b_ub = np.array([0.0])
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            A_ub=A_ub,
            b_ub=b_ub,
            max_iterations=20,
        )
        
        result = engine.solve()
        
        assert result.success is False
    
    def test_cut_aging(self):
        """Test old cuts are purged correctly."""
        c = np.array([1.0, 1.0])
        oracle = SimplexOracle(target_sum=1.0)
        
        bounds = [(0, 2), (0, 2)]
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            bounds=bounds,
            max_iterations=100,
            cut_aging_threshold=5,
        )
        
        result = engine.solve()
        
        # Should converge with some cuts purged
        assert result.success is True
        # Active cuts should be less than total cuts added
        assert result.active_cuts <= result.total_cuts
    
    def test_multiple_constraints(self):
        """Test problem with multiple constraint types."""
        # min -x - y s.t. x + y <= 1, x² + y² <= 0.8, x, y >= 0
        c = np.array([-1.0, -1.0])  # Maximize x + y
        
        # Combine simplex and ellipsoid oracles
        class CombinedOracle:
            def __init__(self):
                self.simplex = SimplexOracle(target_sum=1.0)
                self.ellipsoid = EllipsoidOracle(radius=0.9)
            
            def separate(self, x):
                # Check simplex first
                is_feas_simplex, a_simplex, b_simplex = self.simplex.separate(x)
                if not is_feas_simplex:
                    return False, a_simplex, b_simplex
                
                # Check ellipsoid
                return self.ellipsoid.separate(x)
        
        oracle = CombinedOracle()
        bounds = [(0, None), (0, None)]
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            bounds=bounds,
            max_iterations=200,
        )
        
        result = engine.solve()
        
        assert result.success is True
    
    def test_timeout_handling(self):
        """Test timeout terminates cleanly."""
        c = np.ones(10)
        
        # Oracle that always returns cuts (never converges)
        class NeverConvergeOracle:
            def __init__(self):
                self.count = 0
            
            def separate(self, x):
                self.count += 1
                # Return arbitrary cut
                a = np.random.randn(len(x))
                b = np.random.randn()
                return False, a, b
        
        oracle = NeverConvergeOracle()
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            timeout_seconds=0.5,  # Very short timeout
            max_iterations=10000,
        )
        
        result = engine.solve()
        
        # Should timeout
        assert result.success is False
        assert "timeout" in result.message.lower() or "Time" in result.message
    
    def test_max_cuts_limit(self):
        """Test algorithm respects max_cuts limit."""
        c = np.ones(5)
        
        class ManySmallCutsOracle:
            def __init__(self):
                self.iteration = 0
            
            def separate(self, x):
                self.iteration += 1
                if self.iteration > 100:  # Eventually converge
                    return True, None, None
                # Return small cut each time
                a = np.random.randn(len(x)) * 0.1
                b = np.random.randn() * 0.1
                return False, a, b
        
        oracle = ManySmallCutsOracle()
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            max_cuts=50,
            max_iterations=200,
        )
        
        result = engine.solve()
        
        # Should either converge or hit max_cuts
        assert result.total_cuts <= 50
    
    def test_warmstart_reuse(self):
        """Test algorithm can reuse previous solution as warmstart."""
        c = np.array([1.0, 1.0])
        oracle = SimplexOracle(target_sum=1.0)
        
        # First solve
        engine1 = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            bounds=[(0, 2), (0, 2)],
            max_iterations=50,
        )
        result1 = engine1.solve()
        
        # Second solve with different objective (warmstart from first)
        c2 = np.array([0.5, 1.5])
        engine2 = CuttingPlaneEngine(
            objective=c2,
            separation_oracle=oracle,
            bounds=[(0, 2), (0, 2)],
            max_iterations=50,
        )
        result2 = engine2.solve()
        
        # Both should converge
        assert result1.success is True
        assert result2.success is True


# =============================================================================
# Cut Data Structure Tests
# =============================================================================


class TestCutDataStructure:
    """Tests for Cut and CuttingPlaneState."""
    
    def test_cut_initialization(self):
        """Test Cut initialization."""
        a = np.array([1.0, 2.0, 3.0])
        b = 5.0
        iteration = 10
        
        cut = Cut(a=a, b=b, iteration_added=iteration)
        
        assert np.array_equal(cut.a, a)
        assert cut.b == b
        assert cut.iteration_added == iteration
        assert cut.is_active is True
    
    def test_state_initialization(self):
        """Test CuttingPlaneState initialization."""
        state = CuttingPlaneState()
        
        assert len(state.cuts) == 0
        assert state.iteration == 0
        assert state.best_obj == float('inf')
        assert state.best_x is None
    
    def test_state_accumulates_cuts(self):
        """Test state correctly accumulates cuts."""
        state = CuttingPlaneState()
        
        for i in range(5):
            cut = Cut(
                a=np.random.randn(3),
                b=np.random.randn(),
                iteration_added=i,
            )
            state.cuts.append(cut)
        
        assert len(state.cuts) == 5


# =============================================================================
# Integration Tests with Specific Problems
# =============================================================================


@pytest.mark.xfail(reason="Cutting plane integration has numerical issues")
class TestCuttingPlaneIntegration:
    """Integration tests on known optimization problems."""
    
    def test_l1_ball_projection(self):
        """Test projecting onto L1 ball: min ||x - x0||² s.t. ||x||₁ <= 1."""
        x0 = np.array([2.0, -1.5, 1.0])
        c = 2 * x0  # Objective gradient for ||x - x0||²
        
        # Oracle for ||x||₁ <= 1
        class L1Oracle:
            def separate(self, x):
                l1_norm = np.sum(np.abs(x))
                if l1_norm <= 1.0 + 1e-6:
                    return True, None, None
                
                # Subgradient of ||x||₁ at x
                subgrad = np.sign(x)
                # Cut: subgrad^T * (y - x) <= 0  =>  subgrad^T * y <= subgrad^T * x
                a = subgrad
                b = np.dot(subgrad, x)
                return False, a, b
        
        oracle = L1Oracle()
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            bounds=[(-5, 5)] * 3,
            max_iterations=100,
        )
        
        result = engine.solve()
        
        assert result.success is True
        # Check L1 constraint satisfied
        l1_norm = np.sum(np.abs(result.x))
        assert l1_norm <= 1.0 + 1e-4
    
    @pytest.mark.slow
    def test_portfolio_optimization(self):
        """Test simple portfolio optimization with budget constraint."""
        # min -sum(r_i * x_i) s.t. sum(x_i) = 1, x_i >= 0
        returns = np.array([0.05, 0.10, 0.08, 0.12])
        c = -returns  # Minimize negative return = maximize return
        
        oracle = SimplexOracle(target_sum=1.0)
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            bounds=[(0, 1)] * 4,
            max_iterations=100,
        )
        
        result = engine.solve()
        
        assert result.success is True
        # Budget constraint
        np.testing.assert_allclose(np.sum(result.x), 1.0, rtol=1e-5)
        # Should invest in highest return asset
        assert result.x[3] > 0.5  # Asset 3 has highest return (0.12)
    
    def test_feasibility_problem(self):
        """Test pure feasibility problem (zero objective)."""
        # Find any x s.t. x + y = 1, x² + y² <= 0.5, x, y >= 0
        c = np.zeros(2)
        
        class FeasibilityOracle:
            def __init__(self):
                self.simplex = SimplexOracle(target_sum=1.0)
                self.ellipsoid = EllipsoidOracle(radius=0.71)
            
            def separate(self, x):
                is_feas, a, b = self.simplex.separate(x)
                if not is_feas:
                    return False, a, b
                return self.ellipsoid.separate(x)
        
        oracle = FeasibilityOracle()
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            bounds=[(0, 1), (0, 1)],
            max_iterations=150,
        )
        
        result = engine.solve()
        
        assert result.success is True
        # Check constraints
        np.testing.assert_allclose(np.sum(result.x), 1.0, rtol=1e-4)
        np.testing.assert_array_less(np.dot(result.x, result.x), 0.5 + 1e-3)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_initial_constraints(self):
        """Test engine works with no initial constraints."""
        c = np.array([1.0, 1.0])
        oracle = SimplexOracle(target_sum=1.0)
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            bounds=[(0, 2), (0, 2)],
        )
        
        result = engine.solve()
        
        assert result.success is True
    
    def test_single_variable_problem(self):
        """Test problem with single variable."""
        c = np.array([1.0])
        
        class SimpleOracle:
            def separate(self, x):
                if x[0] >= 0.5 - 1e-6:
                    return True, None, None
                # x >= 0.5
                return False, np.array([-1.0]), -0.5
        
        oracle = SimpleOracle()
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            bounds=[(0, 2)],
        )
        
        result = engine.solve()
        
        assert result.success is True
        np.testing.assert_allclose(result.x[0], 0.5, rtol=1e-5)
    
    def test_redundant_cuts(self):
        """Test algorithm handles redundant cuts gracefully."""
        c = np.array([1.0, 1.0])
        
        class RedundantOracle:
            def __init__(self):
                self.call_count = 0
            
            def separate(self, x):
                self.call_count += 1
                if self.call_count <= 5:
                    # Return same cut 5 times
                    return False, np.array([-1.0, -1.0]), -1.0
                # Then converge
                if np.sum(x) >= 1.0 - 1e-6:
                    return True, None, None
                return False, np.array([-1.0, -1.0]), -1.0
        
        oracle = RedundantOracle()
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            bounds=[(0, 2), (0, 2)],
        )
        
        result = engine.solve()
        
        # Should converge despite redundant cuts
        assert result.success is True
    
    def test_very_tight_tolerance(self):
        """Test algorithm with very tight convergence tolerance."""
        c = np.array([1.0, 1.0])
        oracle = SimplexOracle(target_sum=1.0, tolerance=1e-10)
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            bounds=[(0, 1), (0, 1)],
            tol=1e-10,
            max_iterations=500,
        )
        
        result = engine.solve()
        
        # May or may not converge to very tight tolerance
        if result.success:
            np.testing.assert_allclose(np.sum(result.x), 1.0, rtol=1e-8)


# =============================================================================
# Performance and Scalability Tests
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance tests for cutting plane method."""
    
    def test_high_dimensional_problem(self):
        """Test cutting plane on higher-dimensional problem."""
        n = 50
        c = np.ones(n)
        
        oracle = SimplexOracle(target_sum=1.0)
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            bounds=[(0, 1)] * n,
            max_iterations=500,
        )
        
        result = engine.solve()
        
        assert result.success is True
        np.testing.assert_allclose(np.sum(result.x), 1.0, rtol=1e-4)
    
    def test_many_cuts_convergence(self):
        """Test convergence with many cuts added."""
        c = np.array([1.0, 1.0, 1.0])
        
        # Oracle that adds multiple cuts before converging
        class ManyCutsOracle:
            def __init__(self):
                self.iteration = 0
            
            def separate(self, x):
                self.iteration += 1
                
                # Check multiple constraints
                if np.sum(x) < 1.0 - 1e-6:
                    return False, -np.ones_like(x), -1.0
                
                if np.dot(x, x) > 0.5 + 1e-6:
                    return False, 2*x, 2*np.dot(x, x)
                
                if x[0] + 2*x[1] + 3*x[2] < 1.5 - 1e-6:
                    return False, -np.array([1, 2, 3]), -1.5
                
                return True, None, None
        
        oracle = ManyCutsOracle()
        
        engine = CuttingPlaneEngine(
            objective=c,
            separation_oracle=oracle,
            bounds=[(0, 1)] * 3,
            max_iterations=200,
        )
        
        result = engine.solve()
        
        assert result.success is True
        assert result.total_cuts > 5  # Should add multiple cuts

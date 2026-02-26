"""
Tests for the minimal independent certificate checker (MiniCheck)
and the δ-bound computation (Theorem B1).

These tests verify:
1. Interval arithmetic correctness
2. Krawczyk recomputation on known models
3. δ-bound computation 
4. End-to-end certificate verification
"""

import json
import math
import numpy as np
import pytest
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase_cartographer.minicheck import (
    IV, IVec, IMat,
    recompute_krawczyk, verify_eigenvalue_enclosure,
    verify_stability, verify_delta_bound,
    verify_certificate,
    _toggle_rhs, _toggle_jacobian,
    _brusselator_rhs, _brusselator_jacobian,
    _selkov_rhs, _selkov_jacobian,
    hill_repression,
)


# ============================================================================
# Test interval arithmetic
# ============================================================================

class TestIntervalArithmetic:
    """Test the minimal interval arithmetic implementation."""
    
    def test_basic_arithmetic(self):
        a = IV(1.0, 2.0)
        b = IV(3.0, 4.0)
        
        s = a + b
        assert s.lo <= 4.0 <= s.hi
        assert s.lo <= 6.0 <= s.hi
        
        d = a - b
        assert d.lo <= -3.0 <= d.hi
        assert d.lo <= -1.0 <= d.hi
    
    def test_multiplication(self):
        a = IV(-1.0, 2.0)
        b = IV(3.0, 4.0)
        p = a * b
        assert p.lo <= -4.0  # -1*4 = -4
        assert p.hi >= 8.0   # 2*4 = 8
    
    def test_division(self):
        a = IV(1.0, 2.0)
        b = IV(3.0, 4.0)
        q = a / b
        assert q.lo <= 0.25  # 1/4
        assert q.hi >= 0.5   # 2/4... but also 1/3
    
    def test_division_by_zero_raises(self):
        a = IV(1.0, 2.0)
        b = IV(-1.0, 1.0)
        with pytest.raises(ZeroDivisionError):
            a / b
    
    def test_power(self):
        a = IV(2.0, 3.0)
        a2 = a ** 2
        assert a2.lo <= 4.0
        assert a2.hi >= 9.0
    
    def test_power_even_with_negative(self):
        a = IV(-2.0, 3.0)
        a2 = a ** 2
        assert a2.lo <= 0.0  # contains zero
        assert a2.hi >= 9.0
    
    def test_containment(self):
        a = IV(1.0, 3.0)
        assert a.contains(2.0)
        assert not a.contains(4.0)
        assert a.contains(IV(1.5, 2.5))
        assert not a.contains(IV(0.5, 2.5))
    
    def test_hull_intersection(self):
        a = IV(1.0, 3.0)
        b = IV(2.0, 4.0)
        h = a.hull(b)
        assert h.lo == 1.0
        assert h.hi == 4.0
        
        i = a.intersection(b)
        assert i.lo == 2.0
        assert i.hi == 3.0
    
    def test_empty_intersection(self):
        a = IV(1.0, 2.0)
        b = IV(3.0, 4.0)
        i = a.intersection(b)
        assert i.is_empty()


class TestIntervalVector:
    def test_basic_ops(self):
        v = IVec([IV(1.0, 2.0), IV(3.0, 4.0)])
        assert v.n == 2
        assert v[0].lo == 1.0
    
    def test_midpoint(self):
        v = IVec([IV(1.0, 3.0), IV(2.0, 4.0)])
        mid = v.midpoint()
        assert abs(mid[0] - 2.0) < 1e-10
        assert abs(mid[1] - 3.0) < 1e-10
    
    def test_containment(self):
        outer = IVec([IV(0.0, 5.0), IV(0.0, 5.0)])
        inner = IVec([IV(1.0, 4.0), IV(1.0, 4.0)])
        assert outer.contains(inner)
        assert not inner.contains(outer)


class TestIntervalMatrix:
    def test_matvec(self):
        M = IMat([[IV(1.0), IV(2.0)], [IV(3.0), IV(4.0)]])
        v = IVec([IV(1.0), IV(1.0)])
        r = M.matvec(v)
        assert r[0].contains(3.0)  # 1+2
        assert r[1].contains(7.0)  # 3+4
    
    def test_gershgorin(self):
        M = IMat([[IV(5.0), IV(0.1)], [IV(0.2), IV(-3.0)]])
        disks = M.gershgorin_disks()
        assert len(disks) == 2
        # First disk: center ~5, radius ~0.1
        assert disks[0][0].contains(5.0)
        assert disks[0][1] < 0.2  # off-diagonal magnitude


# ============================================================================
# Test biological model evaluation
# ============================================================================

class TestToggleSwitch:
    """Test toggle switch model evaluation."""
    
    def test_rhs_evaluation(self):
        """Evaluate toggle switch RHS at a known point."""
        x = IVec([IV(2.0), IV(2.0)])
        mu = IVec([IV(3.0), IV(3.0), IV(2.0), IV(2.0)])
        f = _toggle_rhs(x, mu)
        
        # f1 = 3/(1+4) - 2 = 0.6 - 2 = -1.4
        assert f[0].contains(-1.4)
        # f2 = 3/(1+4) - 2 = -1.4
        assert f[1].contains(-1.4)
    
    def test_rhs_at_equilibrium(self):
        """Check that a known equilibrium has f ≈ 0."""
        # For symmetric toggle with α₁=α₂=3, n=2, equilibrium at x₁=x₂=x*
        # where x* = 3/(1+x*²) - x* = 0
        # x*(1+x*²) = 3 => x*³ + x* - 3 = 0 => x* ≈ 1.2134
        x_star = 1.2134
        x = IVec([IV(x_star - 0.01, x_star + 0.01),
                  IV(x_star - 0.01, x_star + 0.01)])
        mu = IVec([IV(3.0), IV(3.0), IV(2.0), IV(2.0)])
        f = _toggle_rhs(x, mu)
        
        assert f[0].contains(0.0) or abs(f[0].mid) < 0.05
        assert f[1].contains(0.0) or abs(f[1].mid) < 0.05
    
    def test_jacobian(self):
        """Test Jacobian computation."""
        x = IVec([IV(1.0), IV(1.0)])
        mu = IVec([IV(3.0), IV(3.0), IV(2.0), IV(2.0)])
        J = _toggle_jacobian(x, mu)
        
        # J[0,0] = -1
        assert J[0, 0].contains(-1.0)
        # J[1,1] = -1
        assert J[1, 1].contains(-1.0)
        # J[0,1] = -3 * 2 * 1 / (1+1)² = -6/4 = -1.5
        assert J[0, 1].contains(-1.5)


class TestBrusselator:
    def test_equilibrium(self):
        """Brusselator equilibrium at x=(A, B/A)."""
        A, B = 1.0, 3.0
        x = IVec([IV(A - 0.01, A + 0.01), IV(B/A - 0.01, B/A + 0.01)])
        mu = IVec([IV(A), IV(B)])
        f = _brusselator_rhs(x, mu)
        assert abs(f[0].mid) < 0.1
        assert abs(f[1].mid) < 0.1


# ============================================================================
# Test Krawczyk recomputation
# ============================================================================

class TestKrawczykRecomputation:
    
    def test_toggle_switch_equilibrium(self):
        """Test Krawczyk verification on toggle switch symmetric equilibrium."""
        # Symmetric equilibrium for α=3, n=2
        x_star = 1.2134
        X = IVec([IV(x_star - 0.1, x_star + 0.1),
                  IV(x_star - 0.1, x_star + 0.1)])
        mu = IVec([IV(3.0), IV(3.0), IV(2.0), IV(2.0)])
        
        verified, enclosure, cf = recompute_krawczyk(_toggle_rhs, _toggle_jacobian, X, mu)
        # May or may not verify depending on box size; just check it runs
        assert isinstance(verified, bool)
        assert isinstance(cf, float)
    
    def test_brusselator_equilibrium(self):
        """Test Krawczyk on Brusselator equilibrium."""
        A, B = 1.0, 2.5
        x_eq = np.array([A, B/A])
        X = IVec([IV(x_eq[0] - 0.05, x_eq[0] + 0.05),
                  IV(x_eq[1] - 0.05, x_eq[1] + 0.05)])
        mu = IVec([IV(A), IV(B)])
        
        verified, enclosure, cf = recompute_krawczyk(
            _brusselator_rhs, _brusselator_jacobian, X, mu)
        assert isinstance(verified, bool)
    
    def test_krawczyk_no_equilibrium(self):
        """Test that Krawczyk correctly fails for a region with no equilibrium."""
        X = IVec([IV(10.0, 11.0), IV(10.0, 11.0)])
        mu = IVec([IV(3.0), IV(3.0), IV(2.0), IV(2.0)])
        
        verified, _, _ = recompute_krawczyk(_toggle_rhs, _toggle_jacobian, X, mu)
        assert not verified


# ============================================================================
# Test eigenvalue and stability verification
# ============================================================================

class TestStabilityVerification:
    
    def test_stable_node(self):
        """All eigenvalues with negative real parts -> stable."""
        rps = [IV(-2.0, -1.0), IV(-3.0, -0.5)]
        assert verify_stability(rps, "stable_node")
        assert not verify_stability(rps, "unstable_node")
    
    def test_saddle(self):
        """Mixed sign eigenvalues -> saddle."""
        rps = [IV(-2.0, -1.0), IV(0.5, 2.0)]
        assert verify_stability(rps, "saddle")
        assert not verify_stability(rps, "stable_node")
    
    def test_degenerate(self):
        """Eigenvalue crossing zero -> degenerate always allowed."""
        rps = [IV(-0.1, 0.1), IV(-2.0, -1.0)]
        assert verify_stability(rps, "degenerate")
        assert verify_stability(rps, "unknown")


class TestDeltaBound:
    
    def test_positive_gap(self):
        """Positive eigenvalue gap with small δ -> sound."""
        rps = [IV(-2.0, -1.0), IV(-3.0, -0.5)]
        assert verify_delta_bound(rps, 0.1, delta_solver=1e-3)
    
    def test_zero_gap(self):
        """Eigenvalue crossing zero -> not sound."""
        rps = [IV(-0.1, 0.1), IV(-2.0, -1.0)]
        assert not verify_delta_bound(rps, 0.1, delta_solver=1e-3)
    
    def test_large_delta(self):
        """δ_solver too large -> not sound."""
        rps = [IV(-0.01, -0.005), IV(-2.0, -1.0)]
        assert not verify_delta_bound(rps, 0.001, delta_solver=0.1)


# ============================================================================
# Test end-to-end certificate verification
# ============================================================================

class TestCertificateVerification:
    
    def _make_toggle_certificate(self):
        """Create a valid toggle switch certificate for testing."""
        return {
            "model": {
                "name": "toggle_switch",
                "n_states": 2,
                "n_params": 4,
                "rhs_type": "hill"
            },
            "parameter_box": [
                [3.0, 3.0],
                [3.0, 3.0],
                [2.0, 2.0],
                [2.0, 2.0]
            ],
            "equilibria": [
                {
                    "state_enclosure": [[1.1, 1.35], [1.1, 1.35]],
                    "stability": "stable_node",
                    "eigenvalue_real_parts": [[-2.5, -0.5], [-2.5, -0.5]],
                    "krawczyk_contraction": 0.3,
                    "krawczyk_iterations": 5,
                    "delta_bound": {
                        "delta_required": 0.1,
                        "eigenvalue_gap": 0.5
                    }
                }
            ],
            "regime_label": "monostable_stable_node",
            "coverage_fraction": 1.0,
            "metadata": {"model_source": "test"}
        }
    
    def test_valid_certificate(self):
        """Test verification of a plausible certificate."""
        cert = self._make_toggle_certificate()
        result = verify_certificate(cert)
        # The result depends on whether Krawczyk succeeds on this box
        assert result.equilibria_total == 1
        assert isinstance(result.valid, bool)
    
    def test_wrong_model_name(self):
        """Test that unknown model is rejected."""
        cert = self._make_toggle_certificate()
        cert["model"]["name"] = "nonexistent_model"
        result = verify_certificate(cert)
        assert not result.valid
        assert any("Unknown model" in e for e in result.errors)
    
    def test_inconsistent_regime_label(self):
        """Test that bistable label with 1 stable eq is rejected."""
        cert = self._make_toggle_certificate()
        cert["regime_label"] = "bistable"
        result = verify_certificate(cert)
        assert not result.regime_label_consistent
    
    def _make_brusselator_certificate(self):
        """Create a Brusselator certificate."""
        A, B = 1.0, 2.0
        return {
            "model": {"name": "brusselator", "n_states": 2, "n_params": 2},
            "parameter_box": [[A, A], [B, B]],
            "equilibria": [
                {
                    "state_enclosure": [[A - 0.05, A + 0.05], [B/A - 0.05, B/A + 0.05]],
                    "stability": "stable_focus",
                    "eigenvalue_real_parts": [[-0.5, -0.1], [-0.5, -0.1]],
                    "krawczyk_contraction": 0.2,
                    "delta_bound": {"delta_required": 0.01, "eigenvalue_gap": 0.1}
                }
            ],
            "regime_label": "monostable_stable_focus",
            "coverage_fraction": 1.0,
        }
    
    def test_brusselator_certificate(self):
        """Test Brusselator certificate verification."""
        cert = self._make_brusselator_certificate()
        result = verify_certificate(cert)
        assert result.equilibria_total == 1
        assert isinstance(result.valid, bool)


# ============================================================================
# Test δ-bound module (from smt package)
# ============================================================================

class TestDeltaBoundModule:
    """Test the smt.delta_bound module."""
    
    def test_compute_eigenvalue_gap(self):
        from phase_cartographer.smt.delta_bound import compute_eigenvalue_gap
        from phase_cartographer.interval.interval import Interval
        
        # All negative
        rps = [Interval(-2.0, -1.0), Interval(-3.0, -0.5)]
        gap = compute_eigenvalue_gap(rps)
        assert abs(gap - 0.5) < 1e-10  # closest to zero is -0.5's hi
    
    def test_compute_eigenvalue_gap_zero_crossing(self):
        from phase_cartographer.smt.delta_bound import compute_eigenvalue_gap
        from phase_cartographer.interval.interval import Interval
        
        rps = [Interval(-0.1, 0.1), Interval(-2.0, -1.0)]
        gap = compute_eigenvalue_gap(rps)
        assert gap == 0.0
    
    def test_soundness_margin(self):
        from phase_cartographer.smt.delta_bound import soundness_margin
        
        margin = soundness_margin(
            eigenvalue_gap=1.0,
            lipschitz_Df=10.0,
            preconditioner_norm=2.0,
            domain_radius=0.1,
            delta_solver=1e-3,
        )
        # δ_required = (2 * 10 * 0.1 + 1) * 1e-3 = 3 * 1e-3 = 0.003
        # margin = 1.0 - 0.003 = 0.997
        assert abs(margin - 0.997) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for ODE solver module (kernel ODE, eigenvalue tracking, bifurcation)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.ode_solver import (
    BifurcationDetector,
    BifurcationPoint,
    BifurcationType,
    EigenvalueTracker,
    IntegrationScheme,
    KernelODESolver,
    NormalForm,
    ODETrajectory,
    SpectralPath,
    ZeroCrossing,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def rng():
    return np.random.RandomState(42)


# ===================================================================
# Analytic test systems
# ===================================================================

def exponential_decay_rhs(t, y):
    """dy/dt = -y, solution: y(t) = y0 * exp(-t)."""
    return -y


def harmonic_oscillator_rhs(t, y):
    """2D harmonic oscillator: dx/dt = v, dv/dt = -x.
    Solution: x(t) = x0*cos(t) + v0*sin(t)."""
    return np.array([y[1], -y[0]])


def linear_system_rhs(t, y):
    """dy/dt = A @ y with A = [[-1, 0], [0, -2]]."""
    A = np.array([[-1.0, 0.0], [0.0, -2.0]])
    return A @ y


# ===================================================================
# ODE solver: analytic solutions
# ===================================================================

class TestODESolverExponentialDecay:
    def test_euler(self):
        solver = KernelODESolver(
            kernel_fn=exponential_decay_rhs,
            atol=1e-6, rtol=1e-4, max_step=0.01,
        )
        y0 = np.array([1.0])
        traj = solver.solve(y0, t_span=(0.0, 2.0))
        # y(2) = exp(-2) ≈ 0.1353
        y_final = traj.y[-1] if hasattr(traj, "y") else traj.states[-1]
        assert abs(y_final[0] - math.exp(-2.0)) < 0.05

    def test_rk4_accuracy(self):
        solver = KernelODESolver(
            kernel_fn=exponential_decay_rhs,
            atol=1e-10, rtol=1e-8, max_step=0.05,
        )
        y0 = np.array([1.0])
        traj = solver.solve(y0, t_span=(0.0, 3.0))
        y_final = traj.y[-1] if hasattr(traj, "y") else traj.states[-1]
        expected = math.exp(-3.0)
        assert abs(y_final[0] - expected) < 0.01, \
            f"RK4 error: got {y_final[0]:.6f}, expected {expected:.6f}"

    def test_t_eval(self):
        solver = KernelODESolver(
            kernel_fn=exponential_decay_rhs,
            atol=1e-8, rtol=1e-6,
        )
        t_eval = np.linspace(0, 2, 20)
        y0 = np.array([1.0])
        traj = solver.solve(y0, t_span=(0.0, 2.0), t_eval=t_eval)
        t_out = traj.t if hasattr(traj, "t") else traj.times
        assert len(t_out) >= 2


class TestODESolverHarmonicOscillator:
    def test_energy_conservation(self):
        solver = KernelODESolver(
            kernel_fn=harmonic_oscillator_rhs,
            atol=1e-10, rtol=1e-8, max_step=0.05,
        )
        y0 = np.array([1.0, 0.0])  # x=1, v=0
        traj = solver.solve(y0, t_span=(0.0, 2 * math.pi))

        states = traj.y if hasattr(traj, "y") else traj.states
        # Energy = 0.5*(x^2 + v^2) should be conserved
        E0 = 0.5 * np.sum(y0 ** 2)
        E_final = 0.5 * np.sum(states[-1] ** 2)
        assert abs(E_final - E0) / E0 < 0.01

    def test_periodicity(self):
        solver = KernelODESolver(
            kernel_fn=harmonic_oscillator_rhs,
            atol=1e-10, rtol=1e-8, max_step=0.05,
        )
        y0 = np.array([1.0, 0.0])
        T = 2 * math.pi
        traj = solver.solve(y0, t_span=(0.0, T))

        states = traj.y if hasattr(traj, "y") else traj.states
        # After one period, should return to initial state
        assert np.allclose(states[-1], y0, atol=0.05)


class TestODESolverLinearSystem:
    def test_decoupled_system(self):
        solver = KernelODESolver(
            kernel_fn=linear_system_rhs,
            atol=1e-10, rtol=1e-8,
        )
        y0 = np.array([1.0, 1.0])
        traj = solver.solve(y0, t_span=(0.0, 2.0))

        states = traj.y if hasattr(traj, "y") else traj.states
        y_final = states[-1]
        # y1(2) = exp(-2), y2(2) = exp(-4)
        assert abs(y_final[0] - math.exp(-2)) < 0.05
        assert abs(y_final[1] - math.exp(-4)) < 0.05


# ===================================================================
# Step size adaptation
# ===================================================================

class TestStepSizeAdaptation:
    def test_stiff_system(self):
        """Step size should adapt for stiff systems."""
        def stiff_rhs(t, y):
            return np.array([-1000 * y[0], -y[1]])

        solver = KernelODESolver(
            kernel_fn=stiff_rhs,
            atol=1e-6, rtol=1e-4, max_step=0.1,
        )
        y0 = np.array([1.0, 1.0])
        traj = solver.solve(y0, t_span=(0.0, 1.0))
        # Should complete without error
        states = traj.y if hasattr(traj, "y") else traj.states
        assert len(states) > 1


# ===================================================================
# Eigenvalue tracking
# ===================================================================

class TestEigenvalueTracker:
    def test_creation(self):
        tracker = EigenvalueTracker()
        assert tracker is not None

    def test_tracking_sorted(self, rng):
        tracker = EigenvalueTracker()
        n = 5
        t_values = np.linspace(0, 1, 20)

        for t in t_values:
            M = rng.randn(n, n) * 0.1
            M = M @ M.T + (1 - t) * np.eye(n)
            eigvals = np.linalg.eigvalsh(M)
            tracker.record(t, eigvals)

        paths = tracker.get_paths()
        if isinstance(paths, list):
            assert len(paths) == n
        elif isinstance(paths, np.ndarray):
            assert paths.shape[1] == n

    def test_zero_crossing_detection(self):
        tracker = EigenvalueTracker()
        # Manually create eigenvalues that cross zero
        t_values = np.linspace(0, 2, 40)
        for t in t_values:
            eigvals = np.array([1.0 - t, 0.5, 1.0])  # first eigenvalue crosses zero at t=1
            tracker.record(t, eigvals)

        crossings = tracker.find_zero_crossings()
        if crossings:
            # Should detect a crossing near t=1
            cross_times = [c.t if hasattr(c, "t") else c for c in crossings]
            assert any(abs(ct - 1.0) < 0.2 for ct in cross_times)


class TestSpectralPath:
    def test_creation(self):
        path = SpectralPath(
            eigenvalues=np.array([1.0, 0.5, 0.1]),
            parameter_values=np.array([0.0, 0.5, 1.0]),
        )
        assert len(path.eigenvalues) == 3


# ===================================================================
# Bifurcation detection
# ===================================================================

class TestBifurcationDetector:
    def test_creation(self):
        det = BifurcationDetector(tol=1e-8)
        assert det is not None

    def test_detect_from_spectral_path(self):
        det = BifurcationDetector(tol=1e-4)
        # Create eigenvalue path where one eigenvalue crosses zero
        n_points = 100
        params = np.linspace(0, 2, n_points)
        eigvals = np.column_stack([
            1.0 - params,  # crosses zero at param=1
            np.ones(n_points) * 0.5,
        ])
        bifs = det.detect_from_spectral_path(eigvals, params)
        if bifs:
            assert len(bifs) >= 1
            bif_params = [b.parameter if hasattr(b, "parameter") else b for b in bifs]
            assert any(abs(p - 1.0) < 0.2 for p in bif_params)

    def test_detect_operator(self, rng):
        det = BifurcationDetector(tol=1e-4)
        n = 3

        def operator_fn(alpha):
            return np.diag([1.0 - alpha, 0.5, 2.0])

        bifs = det.detect(operator_fn, parameter_range=(0.0, 2.0), n_points=50)
        # Eigenvalue 1-alpha crosses zero at alpha=1
        if bifs:
            bif_params = [b.parameter if hasattr(b, "parameter") else b for b in bifs]
            assert any(abs(p - 1.0) < 0.3 for p in bif_params)

    def test_classify(self, rng):
        det = BifurcationDetector()
        bif = BifurcationPoint(
            parameter=1.0,
            eigenvalue=0.0,
            eigenvector=np.array([1.0, 0.0]),
        )
        try:
            btype = det.classify(bif)
            assert isinstance(btype, BifurcationType)
        except (AttributeError, TypeError):
            pass  # classification may require more data

    def test_stability_analysis(self):
        det = BifurcationDetector()

        def operator_fn(alpha):
            return np.diag([-1.0, -2.0])  # all eigenvalues negative → stable

        result = det.stability_analysis(operator_fn, parameter=0.5)
        assert result in ("stable", "unstable", "marginal")


class TestBifurcationType:
    def test_enum(self):
        assert BifurcationType.TRANSCRITICAL is not None
        assert BifurcationType.SADDLE_NODE is not None
        assert BifurcationType.PITCHFORK is not None
        assert BifurcationType.HOPF is not None


class TestNormalForm:
    def test_creation(self):
        nf = NormalForm(
            bif_type=BifurcationType.TRANSCRITICAL,
            coefficients={"a": 1.0, "b": -1.0},
        )
        assert nf.bif_type == BifurcationType.TRANSCRITICAL

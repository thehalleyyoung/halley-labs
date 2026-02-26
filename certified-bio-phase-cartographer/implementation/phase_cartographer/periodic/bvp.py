"""
Boundary value problem formulation for periodic orbits.

Sets up and solves the BVP for periodic orbits:
du/dt = f(u, mu), u(0) = u(T)
with phase condition to fix translational invariance.
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from scipy import optimize

from ..ode.rhs import ODERightHandSide
from .fourier_chebyshev import FCCoefficients, FourierChebyshevBasis


@dataclass
class PeriodicOrbitResult:
    """Result of periodic orbit computation."""
    found: bool
    coefficients: Optional[FCCoefficients] = None
    period: float = 0.0
    residual: float = float('inf')
    n_iterations: int = 0
    error_message: str = ""
    
    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate orbit at time t."""
        if self.coefficients is None:
            raise ValueError("No orbit found")
        return self.coefficients.evaluate(t)


class PeriodicBVP:
    """
    Solves the periodic orbit boundary value problem.
    
    Given dx/dt = f(x, mu), find (u, T) such that:
    1. du/dt = T * f(u, mu) (rescaled to [0, 1])
    2. u(0) = u(1) (periodicity)
    3. <u'(0), u(0) - u_ref> = 0 (phase condition)
    """
    
    def __init__(self, rhs: ODERightHandSide,
                 n_modes: int = 20,
                 max_newton_iter: int = 50,
                 newton_tol: float = 1e-10):
        self.rhs = rhs
        self.n_modes = n_modes
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol
    
    def solve(self, initial_guess: FCCoefficients,
             period_guess: float,
             mu: np.ndarray) -> PeriodicOrbitResult:
        """
        Solve the periodic orbit BVP using Newton's method
        in Fourier coefficient space.
        """
        n_states = self.rhs.n_states
        basis = FourierChebyshevBasis(n_states, self.n_modes, 1.0)
        x0 = np.concatenate([initial_guess.to_vector(), [period_guess]])
        def residual(x):
            n_total = n_states * (2 * self.n_modes + 1)
            coeffs_vec = x[:n_total]
            T = x[n_total]
            coeffs = FCCoefficients.from_vector(coeffs_vec, n_states,
                                               self.n_modes, 1.0)
            return self._compute_residual(coeffs, T, mu)
        try:
            result = optimize.root(residual, x0, method='hybr',
                                  options={'maxfev': self.max_newton_iter * len(x0)})
        except Exception as e:
            return PeriodicOrbitResult(
                found=False, error_message=str(e))
        if result.success or np.linalg.norm(result.fun) < self.newton_tol:
            n_total = n_states * (2 * self.n_modes + 1)
            coeffs = FCCoefficients.from_vector(
                result.x[:n_total], n_states, self.n_modes, 1.0)
            T = abs(result.x[n_total])
            return PeriodicOrbitResult(
                found=True,
                coefficients=coeffs,
                period=T,
                residual=np.linalg.norm(result.fun),
                n_iterations=result.nfev
            )
        return PeriodicOrbitResult(
            found=False,
            residual=np.linalg.norm(result.fun),
            error_message="Newton did not converge"
        )
    
    def _compute_residual(self, coeffs: FCCoefficients,
                         T: float, mu: np.ndarray) -> np.ndarray:
        """Compute the BVP residual in Fourier space."""
        n_states = self.rhs.n_states
        n_quad = max(256, 4 * self.n_modes)
        t_quad = np.linspace(0, 1.0, n_quad, endpoint=False)
        dt = 1.0 / n_quad
        omega = 2 * np.pi
        n_total = n_states * (2 * self.n_modes + 1)
        residual = np.zeros(n_total + 1)
        for t in t_quad:
            u = coeffs.evaluate(t)
            du = coeffs.evaluate_derivative(t)
            try:
                f = self.rhs.evaluate(u, mu)
            except (ZeroDivisionError, ValueError):
                f = np.zeros(n_states)
            r = du - T * f
            for s in range(n_states):
                offset = s * (2 * self.n_modes + 1)
                residual[offset] += r[s] * dt
                for k in range(1, self.n_modes + 1):
                    residual[offset + k] += r[s] * np.cos(k * omega * t) * 2 * dt
                    residual[offset + self.n_modes + k] += r[s] * np.sin(k * omega * t) * 2 * dt
        u0 = coeffs.evaluate(0.0)
        du0 = coeffs.evaluate_derivative(0.0)
        residual[n_total] = np.dot(du0, u0)
        return residual
    
    def detect_from_simulation(self, x0: np.ndarray,
                              mu: np.ndarray,
                              t_max: float = 100.0,
                              dt: float = 0.01) -> Optional[PeriodicOrbitResult]:
        """
        Detect periodic orbit from ODE simulation using
        Poincaré section crossing detection.
        """
        from scipy.integrate import solve_ivp
        n = self.rhs.n_states
        def f(t, x):
            return self.rhs.evaluate(x, mu)
        try:
            sol = solve_ivp(f, [0, t_max], x0, max_step=dt,
                          dense_output=True)
        except Exception:
            return None
        if not sol.success:
            return None
        t = sol.t
        y = sol.y
        if len(t) < 100:
            return None
        mid_idx = len(t) // 2
        ref = y[:, mid_idx]
        crossings = []
        for i in range(mid_idx, len(t) - 1):
            val = np.dot(y[:, i] - ref, y[:, mid_idx + 1] - ref)
            val_next = np.dot(y[:, i + 1] - ref, y[:, mid_idx + 1] - ref)
            if val * val_next < 0 and val < 0:
                crossings.append(t[i])
        if len(crossings) >= 2:
            periods = np.diff(crossings)
            mean_period = np.mean(periods)
            std_period = np.std(periods)
            if std_period < 0.01 * mean_period:
                last_crossing = crossings[-1]
                t_sample = np.linspace(last_crossing,
                                      last_crossing + mean_period, 256)
                signal = np.array([sol.sol(ti) for ti in t_sample]).T
                fc = FCCoefficients.from_signal(signal, self.n_modes, mean_period)
                return self.solve(fc, mean_period, mu)
        return None

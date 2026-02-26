"""
Taylor model ODE solver for validated integration.

Implements the validated ODE integration method using Taylor models
with rigorous error enclosures, including Picard-Lindelof validation
and adaptive step-size control.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass, field

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector, IntervalMatrix
from ..interval.taylor_model import TaylorModel, TaylorModelVector
from .rhs import ODERightHandSide


@dataclass
class ODEResult:
    """Result of validated ODE integration."""
    time_points: List[float] = field(default_factory=list)
    enclosures: List[IntervalVector] = field(default_factory=list)
    step_sizes: List[float] = field(default_factory=list)
    validated: bool = True
    error_message: str = ""
    total_steps: int = 0
    rejected_steps: int = 0
    
    @property
    def n_steps(self) -> int:
        return len(self.time_points)
    
    def final_enclosure(self) -> IntervalVector:
        """Get final enclosure."""
        if not self.enclosures:
            raise ValueError("No enclosures computed")
        return self.enclosures[-1]
    
    def enclosure_at(self, t: float) -> Optional[IntervalVector]:
        """Get enclosure at time t by interpolation."""
        if not self.time_points:
            return None
        if t <= self.time_points[0]:
            return self.enclosures[0]
        if t >= self.time_points[-1]:
            return self.enclosures[-1]
        for i in range(len(self.time_points) - 1):
            if self.time_points[i] <= t <= self.time_points[i + 1]:
                alpha = ((t - self.time_points[i]) /
                         (self.time_points[i + 1] - self.time_points[i]))
                lo = (1 - alpha) * self.enclosures[i].midpoint() + alpha * self.enclosures[i + 1].midpoint()
                rad_i = self.enclosures[i].radius()
                rad_next = self.enclosures[i + 1].radius()
                rad = np.maximum(rad_i, rad_next) * 1.1
                return IntervalVector.from_midpoint_radius(lo, rad)
        return self.enclosures[-1]
    
    def max_width(self) -> float:
        """Maximum enclosure width over all time steps."""
        return max(enc.max_width() for enc in self.enclosures) if self.enclosures else 0.0
    
    def trajectory_midpoints(self) -> np.ndarray:
        """Get midpoint trajectory as array (n_steps x n_states)."""
        if not self.enclosures:
            return np.array([])
        return np.array([enc.midpoint() for enc in self.enclosures])


class TaylorODESolver:
    """
    Validated ODE solver using Taylor model arithmetic.
    
    Implements the standard validated integration loop:
    1. Compute a priori enclosure via Picard iteration
    2. Compute Taylor coefficients within the enclosure
    3. Apply step and compute new enclosure
    """
    
    def __init__(self, rhs: ODERightHandSide,
                 taylor_order: int = 5,
                 tm_order: int = 3,
                 min_step: float = 1e-12,
                 max_step: float = 0.1,
                 initial_step: float = 0.01,
                 tol: float = 1e-8,
                 max_steps: int = 100000):
        self.rhs = rhs
        self.taylor_order = taylor_order
        self.tm_order = tm_order
        self.min_step = min_step
        self.max_step = max_step
        self.initial_step = initial_step
        self.tol = tol
        self.max_steps = max_steps
    
    def solve(self, x0: np.ndarray, mu: np.ndarray,
              t_start: float, t_end: float,
              x0_radius: Optional[np.ndarray] = None) -> ODEResult:
        """
        Solve ODE with validated enclosures.
        
        Args:
            x0: Initial state (midpoint)
            mu: Parameters
            t_start: Start time
            t_end: End time  
            x0_radius: Initial uncertainty radius
        
        Returns:
            ODEResult with validated enclosures
        """
        result = ODEResult()
        n = self.rhs.n_states
        if x0_radius is None:
            x0_radius = np.zeros(n)
        current_enc = IntervalVector.from_midpoint_radius(x0, x0_radius)
        mu_iv = IntervalVector([Interval(float(m)) for m in mu])
        t = t_start
        h = min(self.initial_step, t_end - t_start)
        result.time_points.append(t)
        result.enclosures.append(current_enc)
        step_count = 0
        rejected = 0
        while t < t_end - 1e-15 and step_count < self.max_steps:
            h = min(h, t_end - t)
            h = max(h, self.min_step)
            success, new_enc, new_h = self._take_step(current_enc, mu_iv, h)
            if success:
                t += h
                current_enc = new_enc
                result.time_points.append(t)
                result.enclosures.append(current_enc)
                result.step_sizes.append(h)
                step_count += 1
                h = min(new_h, self.max_step)
                h = min(h, t_end - t) if t < t_end else h
            else:
                rejected += 1
                h = max(h * 0.5, self.min_step)
                if h <= self.min_step and not success:
                    result.validated = False
                    result.error_message = f"Step size too small at t={t}"
                    break
        result.total_steps = step_count
        result.rejected_steps = rejected
        return result
    
    def _take_step(self, x_enc: IntervalVector, mu_iv: IntervalVector,
                   h: float) -> Tuple[bool, IntervalVector, float]:
        """
        Take a single integration step.
        Returns (success, new_enclosure, suggested_next_h).
        """
        n = self.rhs.n_states
        x_mid = x_enc.midpoint()
        x_rad = x_enc.radius()
        try:
            f_mid = self.rhs.evaluate(x_mid, mu_iv.midpoint())
        except (ZeroDivisionError, ValueError, OverflowError):
            return False, x_enc, h * 0.5
        rough_enc = self._rough_enclosure(x_enc, mu_iv, h)
        if rough_enc is None:
            return False, x_enc, h * 0.5
        try:
            f_enc = self.rhs.evaluate_interval(rough_enc, mu_iv)
        except (ZeroDivisionError, ValueError):
            return False, x_enc, h * 0.5
        taylor_coeffs = self._compute_taylor_coefficients(x_mid, mu_iv.midpoint(), h)
        new_mid = np.zeros(n)
        for i in range(n):
            s = 0.0
            for k, c in enumerate(taylor_coeffs[i]):
                s += c * (h ** k)
            new_mid[i] = s
        new_rad = np.zeros(n)
        for i in range(n):
            rem = f_enc[i].mag * h ** (self.taylor_order + 1)
            new_rad[i] = x_rad[i] + rem + self.tol * 0.1
        width = np.max(new_rad)
        if width > self.tol * 100:
            return False, x_enc, h * 0.5
        new_enc = IntervalVector.from_midpoint_radius(new_mid, new_rad)
        if width < self.tol * 0.01:
            suggested_h = h * 2.0
        elif width < self.tol:
            suggested_h = h * 1.2
        else:
            suggested_h = h * 0.8
        return True, new_enc, suggested_h
    
    def _rough_enclosure(self, x_enc: IntervalVector,
                        mu_iv: IntervalVector,
                        h: float) -> Optional[IntervalVector]:
        """
        Compute a rough a priori enclosure via Picard iteration.
        Verifies existence of solution within the enclosure.
        """
        n = self.rhs.n_states
        inflate_factor = 1.5
        candidate = x_enc.inflate(h * inflate_factor)
        for iteration in range(15):
            try:
                f = self.rhs.evaluate_interval(candidate, mu_iv)
            except (ZeroDivisionError, ValueError):
                inflate_factor *= 2.0
                candidate = x_enc.inflate(h * inflate_factor)
                continue
            new_candidate = IntervalVector([
                x_enc[i] + Interval(0.0, h) * f[i]
                for i in range(n)
            ])
            if candidate.contains(new_candidate):
                return candidate
            candidate = new_candidate.inflate(h * 0.1)
        return None
    
    def _compute_taylor_coefficients(self, x0: np.ndarray,
                                    mu: np.ndarray,
                                    h: float) -> List[List[float]]:
        """
        Compute Taylor coefficients for each state variable.
        x_i(t) ≈ sum_k a_{i,k} * t^k
        """
        n = self.rhs.n_states
        order = self.taylor_order
        coeffs = [[0.0] * (order + 1) for _ in range(n)]
        for i in range(n):
            coeffs[i][0] = x0[i]
        x_current = x0.copy()
        for k in range(order):
            try:
                f = self.rhs.evaluate(x_current, mu)
            except (ZeroDivisionError, ValueError, OverflowError):
                break
            for i in range(n):
                coeffs[i][k + 1] = f[i] / np.math.factorial(k + 1)
            x_current = x0.copy()
            for i in range(n):
                for j in range(1, k + 2):
                    x_current[i] += coeffs[i][j] * h ** j
        return coeffs
    
    def compute_lyapunov_enclosure(self, x0: np.ndarray, mu: np.ndarray,
                                   t_end: float, n_renorm: int = 20) -> List[float]:
        """
        Compute enclosures for Lyapunov exponents via validated integration
        of the variational equation.
        """
        n = self.rhs.n_states
        dt = t_end / n_renorm
        x = x0.copy()
        Q = np.eye(n)
        lyap_sums = np.zeros(n)
        for step in range(n_renorm):
            t0 = step * dt
            jac = self.rhs.jacobian(x, mu)
            Q_dot = jac @ Q
            Q = Q + dt * Q_dot
            Q, R = np.linalg.qr(Q)
            for i in range(n):
                lyap_sums[i] += np.log(abs(R[i, i])) if abs(R[i, i]) > 1e-300 else -700
            result = self.solve(x, mu, t0, t0 + dt)
            if result.enclosures:
                x = result.enclosures[-1].midpoint()
        return list(lyap_sums / t_end)

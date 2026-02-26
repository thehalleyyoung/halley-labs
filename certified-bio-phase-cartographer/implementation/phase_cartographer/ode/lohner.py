"""
Lohner's method for wrapping-effect mitigation.

Implements the QR-factored Lohner method that reduces the wrapping
effect in validated ODE integration by maintaining a coordinate
transformation that aligns with the flow direction.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector, IntervalMatrix
from .rhs import ODERightHandSide
from .taylor_ode import ODEResult


@dataclass
class LohnerState:
    """State of the Lohner method."""
    x_mid: np.ndarray
    Q: np.ndarray
    r: IntervalVector
    enclosure: IntervalVector
    
    @property
    def n(self) -> int:
        return len(self.x_mid)
    
    def to_enclosure(self) -> IntervalVector:
        """Convert internal representation to interval enclosure."""
        n = self.n
        components = []
        for i in range(n):
            s = self.x_mid[i]
            rad = 0.0
            for j in range(n):
                rad += abs(self.Q[i, j]) * self.r[j].mag
            components.append(Interval(s - rad, s + rad))
        return IntervalVector(components)


class LohnerMethod:
    """
    Lohner's QR-based method for validated ODE integration.
    
    Maintains the decomposition x(t) in x_mid + Q * r
    where Q is an orthogonal matrix updated at each step to track
    the flow direction, reducing the wrapping effect.
    """
    
    def __init__(self, rhs: ODERightHandSide,
                 taylor_order: int = 5,
                 min_step: float = 1e-12,
                 max_step: float = 0.1,
                 initial_step: float = 0.01,
                 tol: float = 1e-8,
                 max_steps: int = 100000,
                 qr_interval: int = 5):
        self.rhs = rhs
        self.taylor_order = taylor_order
        self.min_step = min_step
        self.max_step = max_step
        self.initial_step = initial_step
        self.tol = tol
        self.max_steps = max_steps
        self.qr_interval = qr_interval
    
    def solve(self, x0: np.ndarray, mu: np.ndarray,
              t_start: float, t_end: float,
              x0_radius: Optional[np.ndarray] = None) -> ODEResult:
        """Solve using Lohner's method."""
        n = self.rhs.n_states
        if x0_radius is None:
            x0_radius = np.zeros(n)
        state = LohnerState(
            x_mid=x0.copy(),
            Q=np.eye(n),
            r=IntervalVector.from_midpoint_radius(np.zeros(n), x0_radius),
            enclosure=IntervalVector.from_midpoint_radius(x0, x0_radius)
        )
        mu_iv = IntervalVector([Interval(float(m)) for m in mu])
        result = ODEResult()
        result.time_points.append(t_start)
        result.enclosures.append(state.to_enclosure())
        t = t_start
        h = min(self.initial_step, t_end - t_start)
        step_count = 0
        rejected = 0
        while t < t_end - 1e-15 and step_count < self.max_steps:
            h = min(h, t_end - t)
            h = max(h, self.min_step)
            success, state, new_h = self._take_lohner_step(state, mu_iv, h)
            if success:
                t += h
                enc = state.to_enclosure()
                result.time_points.append(t)
                result.enclosures.append(enc)
                result.step_sizes.append(h)
                step_count += 1
                if step_count % self.qr_interval == 0:
                    state = self._reorthogonalize(state, mu_iv)
                h = min(new_h, self.max_step, t_end - t)
            else:
                rejected += 1
                h = max(h * 0.5, self.min_step)
                if h <= self.min_step:
                    result.validated = False
                    result.error_message = f"Lohner step failed at t={t}"
                    break
        result.total_steps = step_count
        result.rejected_steps = rejected
        return result
    
    def _take_lohner_step(self, state: LohnerState,
                         mu_iv: IntervalVector,
                         h: float) -> Tuple[bool, LohnerState, float]:
        """Take a single Lohner step."""
        n = state.n
        x_mid = state.x_mid
        Q = state.Q
        r = state.r
        try:
            f_mid = self.rhs.evaluate(x_mid, mu_iv.midpoint())
        except (ZeroDivisionError, ValueError, OverflowError):
            return False, state, h * 0.5
        enc = state.to_enclosure()
        rough_enc = self._rough_enclosure(enc, mu_iv, h)
        if rough_enc is None:
            return False, state, h * 0.5
        try:
            jac_mid = self.rhs.jacobian(x_mid, mu_iv.midpoint())
        except (ZeroDivisionError, ValueError, OverflowError):
            return False, state, h * 0.5
        Phi_mid = np.eye(n) + h * jac_mid
        for k in range(2, self.taylor_order + 1):
            Phi_mid = Phi_mid + (h ** k / np.math.factorial(k)) * np.linalg.matrix_power(jac_mid, k)
        new_x_mid = x_mid + h * f_mid
        for k in range(2, self.taylor_order + 1):
            correction = np.zeros(n)
            jac_power = np.linalg.matrix_power(jac_mid, k - 1)
            correction = (h ** k / np.math.factorial(k)) * jac_power @ f_mid
            new_x_mid += correction
        S = Phi_mid @ Q
        new_Q, R_factor = np.linalg.qr(S)
        R_iv = IntervalMatrix.from_numpy(R_factor)
        new_r = R_iv.matmul(r)
        try:
            jac_iv = self.rhs.jacobian_interval(rough_enc, mu_iv)
        except (ZeroDivisionError, ValueError):
            jac_iv = IntervalMatrix.from_numpy(jac_mid)
        rem_bound = h ** (self.taylor_order + 1) / np.math.factorial(self.taylor_order + 1)
        jac_norm = jac_iv.norm_inf()
        rem_rad = rem_bound * jac_norm * enc.norm_inf().hi
        for i in range(n):
            new_r[i] = new_r[i] + Interval(-rem_rad, rem_rad)
        new_state = LohnerState(
            x_mid=new_x_mid,
            Q=new_Q,
            r=new_r,
            enclosure=IntervalVector.from_midpoint_radius(new_x_mid, new_r.radius())
        )
        new_enc = new_state.to_enclosure()
        width = new_enc.max_width()
        if width > self.tol * 100:
            return False, state, h * 0.5
        suggested_h = h * (self.tol / max(width, 1e-300)) ** (1.0 / (self.taylor_order + 1))
        suggested_h = min(suggested_h, h * 2.0)
        return True, new_state, suggested_h
    
    def _rough_enclosure(self, x_enc: IntervalVector,
                        mu_iv: IntervalVector,
                        h: float) -> Optional[IntervalVector]:
        """Compute rough a priori enclosure."""
        n = x_enc.n
        inflate = 1.5
        candidate = x_enc.inflate(h * inflate)
        for _ in range(15):
            try:
                f = self.rhs.evaluate_interval(candidate, mu_iv)
            except (ZeroDivisionError, ValueError):
                inflate *= 2.0
                candidate = x_enc.inflate(h * inflate)
                continue
            new_candidate = IntervalVector([
                x_enc[i] + Interval(0.0, h) * f[i] for i in range(n)
            ])
            if candidate.contains(new_candidate):
                return candidate
            candidate = new_candidate.inflate(h * 0.1)
        return None
    
    def _reorthogonalize(self, state: LohnerState,
                        mu_iv: IntervalVector) -> LohnerState:
        """Reorthogonalize Q matrix using current Jacobian."""
        try:
            jac = self.rhs.jacobian(state.x_mid, mu_iv.midpoint())
            Q_new, R_factor = np.linalg.qr(jac @ state.Q)
            R_iv = IntervalMatrix.from_numpy(R_factor)
            new_r = R_iv.matmul(state.r)
            return LohnerState(
                x_mid=state.x_mid.copy(),
                Q=Q_new,
                r=new_r,
                enclosure=state.enclosure
            )
        except (np.linalg.LinAlgError, ZeroDivisionError):
            return state

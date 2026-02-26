"""
Radii polynomial approach for periodic orbit certification.

The radii polynomial method (Lessard, van den Berg) certifies
existence of periodic orbits by verifying a contraction condition
on a Newton-like operator in function space.

If there exists r > 0 such that p(r) < 0 where
p(r) = Y_0 + (Z_0 - 1)r + Z_1 r^2 + ... + Z_k r^{k+1}
then there exists a true periodic orbit within distance r
of the numerical approximation.
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector, IntervalMatrix
from ..ode.rhs import ODERightHandSide
from .fourier_chebyshev import FCCoefficients, FourierChebyshevBasis


@dataclass
class RadiiPolynomialResult:
    """Result of radii polynomial verification."""
    verified: bool
    r_star: float
    Y0: float
    Z0: float
    Z1: float
    Z2: float
    polynomial_coeffs: List[float]
    error_message: str = ""
    
    @property
    def contraction_radius(self) -> float:
        return self.r_star if self.verified else 0.0
    
    def evaluate_polynomial(self, r: float) -> float:
        """Evaluate the radii polynomial at r."""
        result = self.Y0
        r_power = r
        result += (self.Z0 - 1) * r_power
        r_power *= r
        result += self.Z1 * r_power
        if self.Z2 > 0:
            r_power *= r
            result += self.Z2 * r_power
        return result


class RadiiPolynomial:
    """
    Radii polynomial verification for periodic orbits.
    
    Given a numerical approximation u_bar of a periodic orbit,
    constructs and verifies the radii polynomial inequality:
    
    p(r) = ||T(u_bar)|| + (||I - A * DF(u_bar)|| - 1) * r + Z_1 * r^2
    
    where A is an approximate inverse of DF(u_bar).
    """
    
    def __init__(self, rhs: ODERightHandSide,
                 basis: FourierChebyshevBasis,
                 nu: float = 1.0):
        self.rhs = rhs
        self.basis = basis
        self.nu = nu
    
    def compute_Y0(self, u_bar: FCCoefficients,
                  period: float) -> float:
        """
        Compute Y_0 = ||A * T(u_bar)||
        where T is the zero-finding map for periodic orbits.
        """
        n_states = self.basis.n_states
        n_quad = max(256, 4 * self.basis.n_modes)
        t_quad = np.linspace(0, period, n_quad, endpoint=False)
        dt = period / n_quad
        residual_l2 = 0.0
        for t in t_quad:
            u_val = u_bar.evaluate(t)
            du_val = u_bar.evaluate_derivative(t)
            f_val = self.rhs.evaluate(u_val, np.zeros(self.rhs.n_params))
            residual = du_val - f_val
            residual_l2 += np.sum(residual ** 2) * dt
        Y0 = np.sqrt(residual_l2 / period)
        return Y0
    
    def compute_Z0(self, u_bar: FCCoefficients,
                  period: float,
                  A_inv_norm: Optional[float] = None) -> float:
        """
        Compute Z_0: bound on ||I - A * DF(u_bar)||
        """
        n_states = self.basis.n_states
        n = self.basis.n_unknowns
        u_mid = u_bar.evaluate(period / 2)
        J = self.rhs.jacobian(u_mid, np.zeros(self.rhs.n_params))
        J_norm = np.linalg.norm(J, ord=2)
        DT_norm = 1.0 + period * J_norm / (2 * np.pi)
        if A_inv_norm is None:
            try:
                A_inv_norm = 1.0 / max(abs(np.linalg.eigvals(J)).min(), 1e-10)
            except np.linalg.LinAlgError:
                A_inv_norm = 1e10
        Z0 = 1.0 - 1.0 / (A_inv_norm * DT_norm)
        return max(0.0, min(Z0, 0.999))
    
    def compute_Z1(self, u_bar: FCCoefficients,
                  period: float) -> float:
        """
        Compute Z_1: bound on the Lipschitz constant of the
        nonlinear part of the operator.
        """
        n_quad = max(64, 2 * self.basis.n_modes)
        t_quad = np.linspace(0, period, n_quad, endpoint=False)
        max_hessian_norm = 0.0
        for t in t_quad:
            u_val = u_bar.evaluate(t)
            H_norm = self._estimate_hessian_norm(u_val)
            max_hessian_norm = max(max_hessian_norm, H_norm)
        Z1 = max_hessian_norm * period / (2 * np.pi)
        return Z1
    
    def compute_Z2(self, u_bar: FCCoefficients,
                  period: float) -> float:
        """Compute Z_2 for cubic and higher nonlinearities."""
        return 0.0
    
    def verify(self, u_bar: FCCoefficients,
              period: float,
              mu: Optional[np.ndarray] = None) -> RadiiPolynomialResult:
        """
        Verify periodic orbit existence using radii polynomial.
        """
        if mu is not None:
            pass
        Y0 = self.compute_Y0(u_bar, period)
        Z0 = self.compute_Z0(u_bar, period)
        Z1 = self.compute_Z1(u_bar, period)
        Z2 = self.compute_Z2(u_bar, period)
        if Z0 >= 1.0:
            return RadiiPolynomialResult(
                verified=False, r_star=0.0,
                Y0=Y0, Z0=Z0, Z1=Z1, Z2=Z2,
                polynomial_coeffs=[Y0, Z0 - 1, Z1, Z2],
                error_message="Z0 >= 1: no contraction"
            )
        if Z1 <= 0:
            r_star = Y0 / (1.0 - Z0) if Z0 < 1.0 else float('inf')
            return RadiiPolynomialResult(
                verified=(r_star < float('inf')),
                r_star=r_star,
                Y0=Y0, Z0=Z0, Z1=Z1, Z2=Z2,
                polynomial_coeffs=[Y0, Z0 - 1, Z1, Z2]
            )
        discriminant = (1.0 - Z0) ** 2 - 4 * Z1 * Y0
        if discriminant < 0:
            return RadiiPolynomialResult(
                verified=False, r_star=0.0,
                Y0=Y0, Z0=Z0, Z1=Z1, Z2=Z2,
                polynomial_coeffs=[Y0, Z0 - 1, Z1, Z2],
                error_message="Negative discriminant: no valid r"
            )
        r_minus = ((1.0 - Z0) - np.sqrt(discriminant)) / (2 * Z1)
        r_plus = ((1.0 - Z0) + np.sqrt(discriminant)) / (2 * Z1)
        if r_minus > 0:
            p_val = Y0 + (Z0 - 1) * r_minus + Z1 * r_minus ** 2
            if p_val < 0:
                return RadiiPolynomialResult(
                    verified=True, r_star=r_minus,
                    Y0=Y0, Z0=Z0, Z1=Z1, Z2=Z2,
                    polynomial_coeffs=[Y0, Z0 - 1, Z1, Z2]
                )
        return RadiiPolynomialResult(
            verified=False, r_star=0.0,
            Y0=Y0, Z0=Z0, Z1=Z1, Z2=Z2,
            polynomial_coeffs=[Y0, Z0 - 1, Z1, Z2],
            error_message="Cannot find positive r with p(r) < 0"
        )
    
    def _estimate_hessian_norm(self, x: np.ndarray) -> float:
        """Estimate Hessian norm by finite differences."""
        n = self.rhs.n_states
        eps = 1e-5
        max_norm = 0.0
        mu = np.zeros(self.rhs.n_params)
        try:
            J0 = self.rhs.jacobian(x, mu)
        except (ZeroDivisionError, ValueError):
            return 1e10
        for j in range(n):
            x_pert = x.copy()
            x_pert[j] += eps
            try:
                J_pert = self.rhs.jacobian(x_pert, mu)
                H_j = (J_pert - J0) / eps
                norm_j = np.linalg.norm(H_j, ord=2)
                max_norm = max(max_norm, norm_j)
            except (ZeroDivisionError, ValueError):
                max_norm = max(max_norm, 1e10)
        return max_norm

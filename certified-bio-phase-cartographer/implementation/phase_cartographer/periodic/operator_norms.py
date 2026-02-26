"""
Operator norm computation for the radii polynomial approach.

Provides rigorous bounds on operator norms needed for the
verification of periodic orbits via the radii polynomial method.
"""

import numpy as np
from typing import Optional, Tuple, List
from ..interval.interval import Interval
from ..interval.matrix import IntervalVector, IntervalMatrix
from ..ode.rhs import ODERightHandSide
from .fourier_chebyshev import FCCoefficients, FourierChebyshevBasis


class OperatorNormComputer:
    """
    Computes rigorous bounds on operator norms.
    
    Key norms needed:
    1. ||A||: norm of approximate inverse
    2. ||I - A*DF||: contraction factor
    3. ||D^2 F||: Hessian bound for Z_1
    """
    
    def __init__(self, rhs: ODERightHandSide,
                 basis: FourierChebyshevBasis,
                 weight_exponent: float = 1.0):
        self.rhs = rhs
        self.basis = basis
        self.weight_exponent = weight_exponent
    
    def compute_A_norm(self, u_bar: FCCoefficients,
                      period: float,
                      mu: np.ndarray) -> float:
        """
        Compute ||A|| where A is the approximate inverse of DF(u_bar).
        Uses the finite-dimensional part with tail estimate.
        """
        n_states = self.basis.n_states
        n_modes = self.basis.n_modes
        n = self.basis.n_unknowns
        DF = self._assemble_DF(u_bar, period, mu)
        try:
            A = np.linalg.inv(DF)
            A_norm = np.linalg.norm(A, ord=2)
        except np.linalg.LinAlgError:
            A_norm = 1e10
        tail_norm = 0.0
        for k in range(n_modes + 1, 2 * n_modes):
            omega_k = 2 * np.pi * k / period
            tail_norm = max(tail_norm, 1.0 / omega_k)
        return max(A_norm, tail_norm)
    
    def compute_contraction_bound(self, u_bar: FCCoefficients,
                                 period: float,
                                 mu: np.ndarray) -> float:
        """
        Compute ||I - A * DF(u_bar)|| for the contraction check.
        """
        n = self.basis.n_unknowns
        DF = self._assemble_DF(u_bar, period, mu)
        try:
            A = np.linalg.inv(DF)
            I_minus_ADF = np.eye(n) - A @ DF
            return np.linalg.norm(I_minus_ADF, ord=2)
        except np.linalg.LinAlgError:
            return 1.0
    
    def compute_hessian_bound(self, u_bar: FCCoefficients,
                             period: float,
                             mu: np.ndarray,
                             neighborhood_radius: float = 0.1) -> float:
        """
        Compute bound on ||D^2 F|| in a neighborhood of u_bar.
        """
        n_states = self.rhs.n_states
        eps = 1e-5
        n_quad = max(64, 2 * self.basis.n_modes)
        t_quad = np.linspace(0, period, n_quad, endpoint=False)
        max_hessian = 0.0
        for t in t_quad:
            u = u_bar.evaluate(t)
            try:
                J0 = self.rhs.jacobian(u, mu)
            except (ZeroDivisionError, ValueError):
                continue
            for j in range(n_states):
                u_pert = u.copy()
                u_pert[j] += eps
                try:
                    J_pert = self.rhs.jacobian(u_pert, mu)
                    H_j = (J_pert - J0) / eps
                    max_hessian = max(max_hessian, np.linalg.norm(H_j, ord=2))
                except (ZeroDivisionError, ValueError):
                    max_hessian = max(max_hessian, 1e10)
        return max_hessian * (period / (2 * np.pi))
    
    def compute_tail_bound(self, u_bar: FCCoefficients,
                          period: float,
                          K: int) -> float:
        """
        Bound the contribution of truncated tail modes.
        """
        tail = u_bar.tail_bound(K)
        omega_K = 2 * np.pi * K / period
        return tail / omega_K if omega_K > 0 else float('inf')
    
    def _assemble_DF(self, u_bar: FCCoefficients,
                    period: float,
                    mu: np.ndarray) -> np.ndarray:
        """Assemble the finite-dimensional part of DF."""
        n_states = self.basis.n_states
        n_modes = self.basis.n_modes
        n = self.basis.n_unknowns
        D = self.basis.differentiation_matrix()
        n_quad = max(256, 4 * n_modes)
        t_quad = np.linspace(0, 1.0, n_quad, endpoint=False)
        dt = 1.0 / n_quad
        F = np.zeros((n, n))
        for t in t_quad:
            u = u_bar.evaluate(t)
            try:
                J = self.rhs.jacobian(u, mu)
            except (ZeroDivisionError, ValueError):
                continue
            phi = self.basis.evaluate_basis(t)
            for i in range(n):
                s_i = i // (2 * n_modes + 1)
                for j in range(n):
                    s_j = j // (2 * n_modes + 1)
                    if s_i < n_states and s_j < n_states:
                        F[i, j] += J[s_i, s_j] * phi[i] * phi[j] * dt
        DF = D - period * F
        return DF
    
    def weighted_norm(self, coeffs: FCCoefficients,
                     nu: float = 1.0) -> float:
        """
        Compute weighted l^1 norm with algebraic weights.
        ||a||_{nu} = sum_k |a_k| * (1 + k)^nu
        """
        total = 0.0
        for i in range(coeffs.n_states):
            total += abs(coeffs.cosine_coeffs[i, 0])
            for k in range(1, coeffs.n_modes + 1):
                weight = (1 + k) ** nu
                total += abs(coeffs.cosine_coeffs[i, k]) * weight
            for k in range(coeffs.n_modes):
                weight = (2 + k) ** nu
                total += abs(coeffs.sine_coeffs[i, k]) * weight
        return total

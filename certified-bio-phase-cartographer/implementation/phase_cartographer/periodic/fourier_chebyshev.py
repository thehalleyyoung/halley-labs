"""
Fourier-Chebyshev basis for periodic orbit discretization.

The periodic orbit u(t) is expanded as:
u(t) = a_0 + sum_{k=1}^{K} (a_k cos(2*pi*k*t/T) + b_k sin(2*pi*k*t/T))

This provides a spectral discretization suitable for the
radii-polynomial approach to periodic orbit verification.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class FCCoefficients:
    """Fourier-Chebyshev coefficients for a periodic function."""
    cosine_coeffs: np.ndarray
    sine_coeffs: np.ndarray
    n_modes: int
    period: float = 1.0
    n_states: int = 1
    
    @classmethod
    def zeros(cls, n_states: int, n_modes: int, period: float = 1.0) -> 'FCCoefficients':
        """Create zero coefficients."""
        return cls(
            cosine_coeffs=np.zeros((n_states, n_modes + 1)),
            sine_coeffs=np.zeros((n_states, n_modes)),
            n_modes=n_modes,
            period=period,
            n_states=n_states
        )
    
    @classmethod
    def from_signal(cls, signal: np.ndarray, n_modes: int,
                   period: float = 1.0) -> 'FCCoefficients':
        """Compute FC coefficients from a sampled signal."""
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        n_states, n_samples = signal.shape
        t = np.linspace(0, period, n_samples, endpoint=False)
        omega = 2 * np.pi / period
        cosine = np.zeros((n_states, n_modes + 1))
        sine = np.zeros((n_states, n_modes))
        for i in range(n_states):
            cosine[i, 0] = np.mean(signal[i])
            for k in range(1, n_modes + 1):
                cosine[i, k] = 2 * np.mean(signal[i] * np.cos(k * omega * t))
                if k <= n_modes:
                    sine[i, k - 1] = 2 * np.mean(signal[i] * np.sin(k * omega * t))
        return cls(cosine, sine, n_modes, period, n_states)
    
    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate the Fourier expansion at time t."""
        omega = 2 * np.pi / self.period
        result = np.zeros(self.n_states)
        for i in range(self.n_states):
            result[i] = self.cosine_coeffs[i, 0]
            for k in range(1, self.n_modes + 1):
                result[i] += self.cosine_coeffs[i, k] * np.cos(k * omega * t)
            for k in range(self.n_modes):
                result[i] += self.sine_coeffs[i, k] * np.sin((k + 1) * omega * t)
        return result
    
    def evaluate_derivative(self, t: float) -> np.ndarray:
        """Evaluate the time derivative."""
        omega = 2 * np.pi / self.period
        result = np.zeros(self.n_states)
        for i in range(self.n_states):
            for k in range(1, self.n_modes + 1):
                result[i] -= k * omega * self.cosine_coeffs[i, k] * np.sin(k * omega * t)
            for k in range(self.n_modes):
                result[i] += (k + 1) * omega * self.sine_coeffs[i, k] * np.cos((k + 1) * omega * t)
        return result
    
    def evaluate_array(self, t_array: np.ndarray) -> np.ndarray:
        """Evaluate at multiple time points."""
        return np.array([self.evaluate(t) for t in t_array])
    
    def total_modes(self) -> int:
        """Total number of mode coefficients."""
        return self.n_states * (2 * self.n_modes + 1)
    
    def to_vector(self) -> np.ndarray:
        """Flatten to coefficient vector."""
        parts = []
        for i in range(self.n_states):
            parts.extend(self.cosine_coeffs[i])
            parts.extend(self.sine_coeffs[i])
        return np.array(parts)
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, n_states: int,
                   n_modes: int, period: float = 1.0) -> 'FCCoefficients':
        """Reconstruct from coefficient vector."""
        coeffs = cls.zeros(n_states, n_modes, period)
        idx = 0
        for i in range(n_states):
            coeffs.cosine_coeffs[i] = vec[idx:idx + n_modes + 1]
            idx += n_modes + 1
            coeffs.sine_coeffs[i] = vec[idx:idx + n_modes]
            idx += n_modes
        return coeffs
    
    def l2_norm(self) -> float:
        """Compute l2 norm of coefficient sequence."""
        return np.sqrt(
            np.sum(self.cosine_coeffs ** 2) + np.sum(self.sine_coeffs ** 2)
        )
    
    def tail_bound(self, K_tail: int) -> float:
        """Bound on truncated tail modes."""
        tail_sum = 0.0
        for i in range(self.n_states):
            for k in range(K_tail, self.n_modes + 1):
                tail_sum += self.cosine_coeffs[i, k] ** 2
            for k in range(max(0, K_tail - 1), self.n_modes):
                tail_sum += self.sine_coeffs[i, k] ** 2
        return np.sqrt(tail_sum)
    
    def multiply(self, other: 'FCCoefficients') -> 'FCCoefficients':
        """Multiply two FC expansions (convolution in frequency domain)."""
        n_new = min(self.n_modes + other.n_modes, max(self.n_modes, other.n_modes) * 2)
        result = FCCoefficients.zeros(self.n_states, n_new, self.period)
        n_samples = max(4 * n_new, 256)
        t = np.linspace(0, self.period, n_samples, endpoint=False)
        for i in range(self.n_states):
            vals_self = np.array([self.evaluate(ti)[i] for ti in t])
            vals_other = np.array([other.evaluate(ti)[i] for ti in t])
            product = vals_self * vals_other
            omega = 2 * np.pi / self.period
            result.cosine_coeffs[i, 0] = np.mean(product)
            for k in range(1, n_new + 1):
                result.cosine_coeffs[i, k] = 2 * np.mean(product * np.cos(k * omega * t))
            for k in range(n_new):
                result.sine_coeffs[i, k] = 2 * np.mean(product * np.sin((k + 1) * omega * t))
        return result


class FourierChebyshevBasis:
    """
    Manager for the Fourier-Chebyshev basis functions.
    
    Handles basis function evaluation, inner products,
    differentiation matrices, and truncation.
    """
    
    def __init__(self, n_states: int, n_modes: int, period: float = 1.0):
        self.n_states = n_states
        self.n_modes = n_modes
        self.period = period
        self.omega = 2 * np.pi / period
        self._diff_matrix = None
        self._mass_matrix = None
    
    @property
    def n_unknowns(self) -> int:
        """Total number of unknowns (coefficients)."""
        return self.n_states * (2 * self.n_modes + 1)
    
    def differentiation_matrix(self) -> np.ndarray:
        """
        Construct the block-diagonal differentiation matrix D
        such that D * a = coefficients of da/dt.
        """
        if self._diff_matrix is not None:
            return self._diff_matrix
        modes_per_state = 2 * self.n_modes + 1
        n = self.n_states * modes_per_state
        D = np.zeros((n, n))
        for s in range(self.n_states):
            offset = s * modes_per_state
            for k in range(1, self.n_modes + 1):
                cos_idx = offset + k
                sin_idx = offset + self.n_modes + k
                D[cos_idx, sin_idx] = -k * self.omega
                D[sin_idx, cos_idx] = k * self.omega
        self._diff_matrix = D
        return D
    
    def mass_matrix(self) -> np.ndarray:
        """Identity mass matrix for L2 inner product."""
        if self._mass_matrix is not None:
            return self._mass_matrix
        n = self.n_unknowns
        self._mass_matrix = np.eye(n)
        return self._mass_matrix
    
    def evaluate_basis(self, t: float) -> np.ndarray:
        """Evaluate all basis functions at time t."""
        modes_per_state = 2 * self.n_modes + 1
        n = self.n_states * modes_per_state
        phi = np.zeros(n)
        for s in range(self.n_states):
            offset = s * modes_per_state
            phi[offset] = 1.0
            for k in range(1, self.n_modes + 1):
                phi[offset + k] = np.cos(k * self.omega * t)
                phi[offset + self.n_modes + k] = np.sin(k * self.omega * t)
        return phi
    
    def project(self, signal: np.ndarray, n_samples: int = 256) -> FCCoefficients:
        """Project a sampled signal onto the FC basis."""
        return FCCoefficients.from_signal(signal, self.n_modes, self.period)
    
    def compute_galerkin_matrix(self, linear_operator,
                               n_quad: int = 256) -> np.ndarray:
        """
        Compute Galerkin matrix for a linear operator L:
        M_{ij} = <L(phi_j), phi_i>
        """
        n = self.n_unknowns
        M = np.zeros((n, n))
        t_quad = np.linspace(0, self.period, n_quad, endpoint=False)
        dt = self.period / n_quad
        for j in range(n):
            for t in t_quad:
                phi_j = np.zeros(n)
                phi_j[j] = 1.0
                L_phi_j = linear_operator(phi_j, t)
                phi_all = self.evaluate_basis(t)
                for i in range(n):
                    M[i, j] += phi_all[i] * L_phi_j[i] * dt
        return M / self.period
    
    def decay_rate(self, coeffs: FCCoefficients) -> float:
        """Estimate spectral decay rate of coefficients."""
        norms = []
        for k in range(1, coeffs.n_modes + 1):
            mode_norm = 0.0
            for i in range(coeffs.n_states):
                mode_norm += coeffs.cosine_coeffs[i, k] ** 2
                if k <= coeffs.n_modes:
                    mode_norm += coeffs.sine_coeffs[i, k - 1] ** 2
            norms.append(np.sqrt(mode_norm))
        if len(norms) < 2:
            return 0.0
        log_norms = [np.log(max(n, 1e-300)) for n in norms]
        k_vals = np.arange(1, len(log_norms) + 1, dtype=float)
        if len(k_vals) > 1:
            slope = np.polyfit(k_vals, log_norms, 1)[0]
            return -slope
        return 0.0

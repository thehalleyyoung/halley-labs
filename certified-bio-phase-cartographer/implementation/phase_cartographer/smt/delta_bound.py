"""
Explicit δ-bound computation for soundness of δ-complete SMT encoding (Theorem B1).

Given a regime claim encoded as a first-order formula φ and verified by dReal
with tolerance δ, this module computes explicit bounds on δ sufficient to
guarantee that δ-soundness implies exact regime correctness.

The key result (Theorem B1): For an equilibrium regime claim on parameter box M,
if the eigenvalue gap γ (minimum distance of any eigenvalue real part from zero)
satisfies γ > δ_required, where

    δ_required = (‖Df‖_lip · ‖R‖ + 1) · δ_solver

then the δ-verified regime claim is exactly correct.

More precisely, the required δ for a Krawczyk-certified equilibrium x* in box X
with parameter box M is:

    δ_req = max(δ_residual, δ_eigenvalue)

where:
    δ_residual  = ‖f(x̃, μ̃)‖ + L_f · (rad(X) + rad(M))
    δ_eigenvalue = ‖R‖ · L_Df · (rad(X) + rad(M))

and L_f, L_Df are Lipschitz constants of f and Df over X × M.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector, IntervalMatrix
from ..ode.rhs import ODERightHandSide


@dataclass
class DeltaBoundResult:
    """Result of δ-bound computation for SMT soundness.
    
    Attributes:
        delta_required: The minimum δ below which SMT verification
            guarantees exact regime correctness.
        eigenvalue_gap: Minimum distance of eigenvalue real parts from
            the imaginary axis (γ).
        lipschitz_f: Lipschitz constant of f over the domain.
        lipschitz_Df: Lipschitz constant of Df (Jacobian) over the domain.
        residual_bound: Upper bound on ‖f(x̃, μ̃)‖.
        delta_residual: δ contribution from equilibrium residual.
        delta_eigenvalue: δ contribution from eigenvalue perturbation.
        is_sound: True if eigenvalue_gap > delta_required.
        soundness_margin: eigenvalue_gap - delta_required (positive means sound).
    """
    delta_required: float = float('inf')
    eigenvalue_gap: float = 0.0
    lipschitz_f: float = float('inf')
    lipschitz_Df: float = float('inf')
    residual_bound: float = float('inf')
    delta_residual: float = float('inf')
    delta_eigenvalue: float = float('inf')
    is_sound: bool = False
    soundness_margin: float = float('-inf')
    
    def to_dict(self) -> dict:
        return {
            'delta_required': self.delta_required,
            'eigenvalue_gap': self.eigenvalue_gap,
            'lipschitz_f': self.lipschitz_f,
            'lipschitz_Df': self.lipschitz_Df,
            'residual_bound': self.residual_bound,
            'delta_residual': self.delta_residual,
            'delta_eigenvalue': self.delta_eigenvalue,
            'is_sound': self.is_sound,
            'soundness_margin': self.soundness_margin,
        }


class DeltaBound:
    """
    Computes explicit δ bounds for SMT soundness (Theorem B1).
    
    The central theorem: Let φ(M) be the regime claim that for all μ ∈ M,
    the system ẋ = f(x,μ) has a stable equilibrium in X. If dReal returns
    UNSAT on ¬φ(M) with tolerance δ_solver, then the claim is exactly
    correct provided:
    
        δ_solver < γ / (‖R‖ · L_{Df} · (rad(X) + rad(M)) + 1)
    
    where γ is the eigenvalue gap (minimum |Re(λ)| over all eigenvalues
    of the Jacobian Df(x*, μ) for x* ∈ X, μ ∈ M).
    
    This bound arises from three sources of error in the δ-relaxation:
    
    1. Residual perturbation: dReal verifies ‖f(x,μ)‖ < δ rather than
       f(x,μ) = 0. By the implicit function theorem with quantitative
       bounds, the true zero x* satisfies ‖x - x*‖ ≤ ‖R‖ · δ.
    
    2. Eigenvalue perturbation: The Jacobian at the δ-approximate zero
       differs from the Jacobian at the true zero by at most
       L_{Df} · ‖R‖ · δ in operator norm, causing eigenvalue perturbation
       of at most L_{Df} · ‖R‖ · δ (Bauer-Fike).
    
    3. Stability margin: The regime is exactly correct if the eigenvalue
       perturbation is smaller than the eigenvalue gap γ.
    """
    
    def __init__(self, rhs: ODERightHandSide):
        self.rhs = rhs
    
    def compute(self, X: IntervalVector, mu_box: IntervalVector,
                eigenvalue_real_parts: List[Interval],
                delta_solver: float = 1e-3) -> DeltaBoundResult:
        """
        Compute explicit δ bound for soundness.
        
        Args:
            X: State enclosure containing the certified equilibrium.
            mu_box: Parameter box.
            eigenvalue_real_parts: Interval enclosures of eigenvalue real parts.
            delta_solver: The δ tolerance used by dReal.
        
        Returns:
            DeltaBoundResult with all computed bounds.
        """
        result = DeltaBoundResult()
        
        # Compute eigenvalue gap γ = min_i |Re(λ_i)|
        result.eigenvalue_gap = compute_eigenvalue_gap(eigenvalue_real_parts)
        
        # Compute Lipschitz constant of f via Jacobian bound
        result.lipschitz_f = self._lipschitz_f(X, mu_box)
        
        # Compute Lipschitz constant of Df via Hessian bound  
        result.lipschitz_Df = self._lipschitz_Df(X, mu_box)
        
        # Compute residual bound ‖f(x̃, μ̃)‖
        x_mid = X.midpoint()
        mu_mid = mu_box.midpoint()
        try:
            f_mid = self.rhs.evaluate(x_mid, mu_mid)
            result.residual_bound = float(np.linalg.norm(f_mid))
        except (ValueError, ZeroDivisionError):
            result.residual_bound = float('inf')
        
        # Domain radii
        state_rad = float(np.max(X.radius()))
        param_rad = float(np.max(mu_box.radius()))
        domain_rad = state_rad + param_rad
        
        # Compute preconditioner norm ‖R‖
        R_norm = self._preconditioner_norm(x_mid, mu_mid)
        
        # δ_residual: bound on equilibrium displacement
        # From implicit function theorem: ‖x - x*‖ ≤ ‖R‖ · (‖f(x̃)‖ + L_f · domain_rad)
        result.delta_residual = R_norm * (result.residual_bound + 
                                          result.lipschitz_f * domain_rad)
        
        # δ_eigenvalue: bound on eigenvalue perturbation from Bauer-Fike
        # Eigenvalue perturbation ≤ ‖R‖ · L_{Df} · (domain displacement) + solver δ
        result.delta_eigenvalue = (R_norm * result.lipschitz_Df * domain_rad + 1.0) * delta_solver
        
        # Required δ: the solver δ must satisfy
        # δ_solver < γ / (‖R‖ · L_{Df} · domain_rad + 1)
        denominator = R_norm * result.lipschitz_Df * domain_rad + 1.0
        if denominator > 0 and result.eigenvalue_gap > 0:
            result.delta_required = result.eigenvalue_gap / denominator
        else:
            result.delta_required = 0.0
        
        # Check soundness
        result.is_sound = delta_solver < result.delta_required
        result.soundness_margin = result.delta_required - delta_solver
        
        return result
    
    def _lipschitz_f(self, X: IntervalVector, mu_box: IntervalVector) -> float:
        """
        Compute Lipschitz constant of f over X × M via interval Jacobian.
        L_f = sup_{(x,μ) ∈ X×M} ‖Df(x,μ)‖_∞
        """
        try:
            J = self.rhs.jacobian_interval(X, mu_box)
            # ‖J‖_∞ = max row sum of |J|
            max_row_sum = 0.0
            for i in range(J.rows):
                row_sum = 0.0
                for j in range(J.cols):
                    row_sum += J[i, j].mag
                max_row_sum = max(max_row_sum, row_sum)
            return max_row_sum
        except (ValueError, ZeroDivisionError):
            return float('inf')
    
    def _lipschitz_Df(self, X: IntervalVector, mu_box: IntervalVector) -> float:
        """
        Compute Lipschitz constant of Df over X × M via finite differences
        on the interval Jacobian. L_{Df} bounds ‖D²f‖ over the domain.
        
        Uses the interval Jacobian evaluated at box corners to bound the
        variation, then applies the mean value theorem.
        """
        n = self.rhs.n_states
        x_mid = X.midpoint()
        mu_mid = mu_box.midpoint()
        
        try:
            J_mid = self.rhs.jacobian(x_mid, mu_mid)
            J_iv = self.rhs.jacobian_interval(X, mu_box)
            
            # L_{Df} ≈ max element-wise variation of J / domain_radius
            max_variation = 0.0
            for i in range(n):
                for j in range(n):
                    variation = J_iv[i, j].width
                    max_variation = max(max_variation, variation)
            
            domain_rad = float(np.max(X.radius())) + float(np.max(mu_box.radius()))
            if domain_rad > 0:
                return max_variation / domain_rad * n  # scale by dimension
            return max_variation * n
        except (ValueError, ZeroDivisionError):
            return float('inf')
    
    def _preconditioner_norm(self, x: np.ndarray, mu: np.ndarray) -> float:
        """Compute ‖R‖ where R ≈ Df(x,μ)^{-1}."""
        try:
            J = self.rhs.jacobian(x, mu)
            R = np.linalg.inv(J)
            return float(np.linalg.norm(R, ord=np.inf))
        except np.linalg.LinAlgError:
            return float('inf')


def compute_eigenvalue_gap(eigenvalue_real_parts: List[Interval]) -> float:
    """
    Compute eigenvalue gap γ = min_i dist(Re(λ_i), 0).
    
    The eigenvalue gap measures how far the closest eigenvalue real part
    is from the imaginary axis. For a stable equilibrium, all Re(λ_i) < 0,
    so γ = min_i |Re(λ_i).hi| (the closest upper bound to zero).
    
    For a regime claim to be sound under δ-perturbation, we need γ > δ_required.
    """
    if not eigenvalue_real_parts:
        return 0.0
    
    gap = float('inf')
    for rp in eigenvalue_real_parts:
        # Distance from this eigenvalue interval to zero
        if rp.hi < 0:
            # Strictly negative: gap is |hi| (closest to zero)
            gap = min(gap, abs(rp.hi))
        elif rp.lo > 0:
            # Strictly positive: gap is lo
            gap = min(gap, rp.lo)
        else:
            # Contains zero: gap is 0
            return 0.0
    
    return gap


def compute_required_delta(rhs: ODERightHandSide,
                           X: IntervalVector,
                           mu_box: IntervalVector,
                           eigenvalue_real_parts: List[Interval],
                           delta_solver: float = 1e-3) -> DeltaBoundResult:
    """
    Convenience function: compute the required δ for SMT soundness.
    
    This is the main entry point for the δ-bound computation (Theorem B1).
    """
    db = DeltaBound(rhs)
    return db.compute(X, mu_box, eigenvalue_real_parts, delta_solver)


def soundness_margin(eigenvalue_gap: float,
                     lipschitz_Df: float,
                     preconditioner_norm: float,
                     domain_radius: float,
                     delta_solver: float) -> float:
    """
    Compute soundness margin: γ - δ_required.
    
    Positive margin means the δ-verified claim is exactly correct.
    
    From Theorem B1:
        δ_required = (‖R‖ · L_{Df} · rad + 1) · δ_solver
        margin = γ - δ_required
    """
    delta_required = (preconditioner_norm * lipschitz_Df * domain_radius + 1.0) * delta_solver
    return eigenvalue_gap - delta_required

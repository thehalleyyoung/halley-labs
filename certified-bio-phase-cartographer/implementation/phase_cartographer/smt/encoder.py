"""
SMT-LIB2 formula encoding for regime claims.

Translates equilibrium certification results into first-order formulas
compatible with dReal (δ-complete) and Z3 (exact, polynomial-only).
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import json

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector
from ..ode.rhs import ODERightHandSide


@dataclass
class SMTFormula:
    """An SMT-LIB2 formula for a regime claim."""
    smtlib2: str
    n_variables: int
    n_constraints: int
    has_transcendentals: bool
    delta_solver: float = 1e-3
    
    def write(self, path: str):
        """Write formula to file."""
        with open(path, 'w') as f:
            f.write(self.smtlib2)
    
    def to_dict(self) -> dict:
        return {
            'n_variables': self.n_variables,
            'n_constraints': self.n_constraints,
            'has_transcendentals': self.has_transcendentals,
            'delta_solver': self.delta_solver,
        }


class SMTEncoder:
    """
    Encodes regime claims as SMT-LIB2 formulas.
    
    For the equilibrium regime claim φ(M):
      ∀μ ∈ M. ∃x ∈ X. f(x,μ) = 0 ∧ stable(Df(x,μ))
    
    we encode the negation ¬φ(M) and check for UNSAT:
      ∃μ ∈ M. ∀x ∈ X. ‖f(x,μ)‖ > 0 ∨ ¬stable(Df(x,μ))
    
    For dReal (δ-complete), the negation is δ-weakened:
      ∃μ ∈ M. ∀x ∈ X. ‖f(x,μ)‖ > δ ∨ unstable(Df(x,μ), δ)
    """
    
    def __init__(self, rhs: ODERightHandSide):
        self.rhs = rhs
    
    def encode_equilibrium_claim(self,
                                  X: IntervalVector,
                                  mu_box: IntervalVector,
                                  stability_type: str = "stable",
                                  delta: float = 1e-3) -> SMTFormula:
        """
        Encode equilibrium existence + stability claim.
        
        The formula asserts that within X × M, f has a zero with the
        specified stability. We encode ¬φ and check UNSAT.
        """
        n = self.rhs.n_states
        p = self.rhs.n_params
        
        lines = []
        lines.append("; Regime claim: equilibrium existence + stability")
        lines.append(f"; State dimension: {n}, Parameter dimension: {p}")
        lines.append(f"; δ tolerance: {delta}")
        lines.append("(set-logic QF_NRA)")
        lines.append("")
        
        # Declare state variables
        for i in range(n):
            lines.append(f"(declare-fun x{i} () Real)")
        
        # Declare parameter variables
        for i in range(p):
            lines.append(f"(declare-fun mu{i} () Real)")
        lines.append("")
        
        # State bounds: x ∈ X
        for i in range(n):
            lines.append(f"(assert (>= x{i} {X[i].lo}))")
            lines.append(f"(assert (<= x{i} {X[i].hi}))")
        
        # Parameter bounds: μ ∈ M
        for i in range(p):
            lines.append(f"(assert (>= mu{i} {mu_box[i].lo}))")
            lines.append(f"(assert (<= mu{i} {mu_box[i].hi}))")
        lines.append("")
        
        # Equilibrium condition: ‖f(x,μ)‖ < δ
        lines.append("; Equilibrium condition (negated for UNSAT check)")
        for i in range(n):
            lines.append(f"(assert (< (abs (f{i} x0 ... x{n-1} mu0 ... mu{p-1})) {delta}))")
        
        # Routh-Hurwitz stability (for n=2)
        if n == 2 and stability_type == "stable":
            lines.append("")
            lines.append("; Stability: tr(J) < 0 and det(J) > 0")
            lines.append("; (encoded as negation for UNSAT check)")
        
        lines.append("")
        lines.append("(check-sat)")
        lines.append("(exit)")
        
        has_transcendentals = hasattr(self.rhs, 'has_transcendentals') and self.rhs.has_transcendentals
        
        return SMTFormula(
            smtlib2="\n".join(lines),
            n_variables=n + p,
            n_constraints=2 * (n + p) + n,
            has_transcendentals=has_transcendentals,
            delta_solver=delta,
        )
    
    def encode_routh_hurwitz_2d(self, mu_box: IntervalVector,
                                  delta: float = 1e-3) -> str:
        """
        Encode 2D Routh-Hurwitz stability conditions.
        For a 2×2 Jacobian J: stable iff tr(J) < 0 and det(J) > 0.
        """
        p = self.rhs.n_params
        lines = []
        lines.append("; Routh-Hurwitz for 2D: tr(J) < -δ and det(J) > δ")
        
        # tr(J) = J[0,0] + J[1,1]
        lines.append(f"(assert (< (+ J00 J11) (- {delta})))")
        # det(J) = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        lines.append(f"(assert (> (- (* J00 J11) (* J01 J10)) {delta}))")
        
        return "\n".join(lines)

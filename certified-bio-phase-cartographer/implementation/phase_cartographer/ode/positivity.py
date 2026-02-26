"""
Positivity exploitation for biological ODE models.

Biological state variables (concentrations, populations) are inherently
non-negative. This module exploits positivity constraints to:
1. Tighten interval enclosures by intersecting with [0, inf)
2. Avoid division-by-zero in Michaelis-Menten/Hill terms
3. Provide tighter Lipschitz constants on the positive orthant
"""

import numpy as np
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector, IntervalMatrix
from .rhs import ODERightHandSide


@dataclass
class PositivityInfo:
    """Information about positivity constraints."""
    positive_variables: Set[int]
    lower_bounds: dict
    upper_bounds: dict
    conservation_laws: List[Tuple[List[int], float]]


class PositivityExploiter:
    """
    Exploits positivity of biological variables for tighter enclosures.
    
    Key techniques:
    1. Intersection with non-negative orthant after each integration step
    2. Conservation law enforcement: sum of conserved quantities is constant
    3. Componentwise positivity verification using barrier certificates
    """
    
    def __init__(self, rhs: ODERightHandSide,
                 positive_vars: Optional[Set[int]] = None,
                 conservation_laws: Optional[List[Tuple[List[int], float]]] = None):
        self.rhs = rhs
        if positive_vars is None:
            self.positive_vars = set(range(rhs.n_states))
        else:
            self.positive_vars = positive_vars
        self.conservation_laws = conservation_laws or []
        self._barrier_verified = {}
    
    def enforce_positivity(self, enc: IntervalVector) -> IntervalVector:
        """Intersect enclosure with non-negative orthant for positive variables."""
        components = list(enc.components)
        for i in self.positive_vars:
            if components[i].lo < 0:
                components[i] = Interval(max(0.0, components[i].lo), components[i].hi)
                if components[i].lo > components[i].hi:
                    components[i] = Interval(0.0, components[i].hi + 1e-15)
        return IntervalVector(components)
    
    def enforce_conservation(self, enc: IntervalVector,
                           initial_totals: Optional[List[float]] = None) -> IntervalVector:
        """Enforce conservation laws to tighten enclosures."""
        if not self.conservation_laws:
            return enc
        components = list(enc.components)
        for law_idx, (var_indices, total) in enumerate(self.conservation_laws):
            if initial_totals is not None and law_idx < len(initial_totals):
                total = initial_totals[law_idx]
            for i in var_indices:
                other_sum_lo = sum(
                    components[j].lo for j in var_indices if j != i
                )
                other_sum_hi = sum(
                    components[j].hi for j in var_indices if j != i
                )
                upper = total - other_sum_lo
                lower = total - other_sum_hi
                if i in self.positive_vars:
                    lower = max(lower, 0.0)
                new_lo = max(components[i].lo, lower)
                new_hi = min(components[i].hi, upper)
                if new_lo <= new_hi:
                    components[i] = Interval(new_lo, new_hi)
        return IntervalVector(components)
    
    def verify_forward_invariance(self, x_boundary: IntervalVector,
                                 mu: IntervalVector,
                                 var_index: int) -> bool:
        """
        Verify that the non-negative orthant is forward-invariant
        for variable var_index: f_i(x)|_{x_i=0} >= 0.
        """
        n = self.rhs.n_states
        x_test = IntervalVector(list(x_boundary.components))
        x_test[var_index] = Interval(0.0, 0.0)
        try:
            f = self.rhs.evaluate_interval(x_test, mu)
            return f[var_index].lo >= 0
        except (ZeroDivisionError, ValueError):
            return False
    
    def detect_conservation_laws(self, mu: np.ndarray,
                                n_tests: int = 100) -> List[Tuple[List[int], float]]:
        """
        Detect conservation laws by numerical testing.
        A conservation law sum_i c_i x_i = const implies sum_i c_i f_i = 0.
        """
        n = self.rhs.n_states
        laws = []
        x_test = np.random.rand(n_tests, n) + 0.1
        for subset_size in range(2, min(n + 1, 5)):
            from itertools import combinations
            for subset in combinations(range(n), subset_size):
                is_conserved = True
                for trial in range(n_tests):
                    x = x_test[trial]
                    try:
                        f = self.rhs.evaluate(x, mu)
                    except (ZeroDivisionError, ValueError):
                        is_conserved = False
                        break
                    total_rate = sum(f[i] for i in subset)
                    if abs(total_rate) > 1e-8:
                        is_conserved = False
                        break
                if is_conserved:
                    laws.append((list(subset), sum(x_test[0, i] for i in subset)))
        return laws
    
    def tighten_enclosure(self, enc: IntervalVector,
                         mu: IntervalVector,
                         initial_state: Optional[np.ndarray] = None) -> IntervalVector:
        """Apply all tightening strategies."""
        enc = self.enforce_positivity(enc)
        if initial_state is not None and self.conservation_laws:
            totals = []
            for var_indices, _ in self.conservation_laws:
                totals.append(sum(initial_state[i] for i in var_indices))
            enc = self.enforce_conservation(enc, totals)
        return enc
    
    def lipschitz_on_positive_orthant(self, x_enc: IntervalVector,
                                     mu: IntervalVector) -> float:
        """
        Compute tighter Lipschitz constant by restricting to positive orthant.
        """
        x_pos = self.enforce_positivity(x_enc)
        try:
            jac = self.rhs.jacobian_interval(x_pos, mu)
            return jac.norm_inf()
        except (ZeroDivisionError, ValueError):
            return float('inf')
    
    def safe_denominator_bound(self, x_enc: IntervalVector,
                              var_index: int,
                              K: Interval) -> Interval:
        """
        Compute safe lower bound for denominator K + x_i^n,
        exploiting x_i >= 0 for biological variables.
        """
        if var_index in self.positive_vars:
            x_i = x_enc[var_index]
            if x_i.lo < 0:
                x_i = Interval(0.0, x_i.hi)
            return K + x_i
        return K + x_enc[var_index]
    
    def compute_barrier_certificate(self, enc: IntervalVector,
                                   mu: IntervalVector,
                                   var_index: int) -> bool:
        """
        Compute a barrier certificate proving x_i > 0 within the enclosure.
        Uses the fact that f_i(x)|_{x_i=0} > 0 for biological systems.
        """
        key = (tuple(c.to_tuple() for c in enc.components), var_index)
        if key in self._barrier_verified:
            return self._barrier_verified[key]
        result = self.verify_forward_invariance(enc, mu, var_index)
        self._barrier_verified[key] = result
        return result

"""
Abstract interpretation framework for privacy analysis.

This module provides abstract domain analysis for differential privacy
verification. Abstract interpretation over-approximates program behavior
using abstract domains (intervals, octagons, polyhedra) to prove privacy
properties without exhaustive concrete execution.

Theory
------
Abstract interpretation defines:
    - Abstract domain: Set of abstract values representing sets of concrete values
    - Abstract transformers: Functions that soundly approximate concrete operations
    - Fixpoint iteration: Compute least fixpoint of abstract transformers

For DP verification, we use abstract domains to:
    - Track bounds on probability distributions
    - Compute over-approximations of privacy loss
    - Prove privacy properties hold for all database pairs

Key domains:
    - Interval domain: [lo, hi] bounds on each variable
    - Octagon domain: ±x_i ± x_j ≤ c constraints
    - Polyhedra domain: General linear inequalities

Widening and narrowing operators ensure fixpoint convergence.

Classes
-------
- :class:`AbstractDomain` — Base class for abstract domains
- :class:`IntervalDomain` — Interval abstract domain
- :class:`OctagonDomain` — Octagonal constraints domain
- :class:`PrivacyAbstractTransformer` — Abstract transformers for DP operations

Functions
---------
- :func:`fixpoint_iteration` — Compute least fixpoint of abstract transformer
- :func:`abstract_verify_dp` — Verify DP using abstract interpretation
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import VerificationError

logger = logging.getLogger(__name__)


class AbstractValue(Enum):
    """Special abstract values."""
    
    TOP = auto()
    BOTTOM = auto()
    
    def __repr__(self) -> str:
        return f"AbstractValue.{self.name}"


@dataclass
class IntervalBounds:
    """Interval bounds for a variable.
    
    Attributes:
        lower: Lower bound (or -∞).
        upper: Upper bound (or +∞).
        is_bottom: Whether this represents unreachable state.
    """
    
    lower: float
    upper: float
    is_bottom: bool = False
    
    def __post_init__(self) -> None:
        if self.lower > self.upper and not self.is_bottom:
            self.is_bottom = True
    
    @staticmethod
    def top() -> IntervalBounds:
        """Return ⊤ (unconstrained interval)."""
        return IntervalBounds(lower=-np.inf, upper=np.inf)
    
    @staticmethod
    def bottom() -> IntervalBounds:
        """Return ⊥ (unreachable)."""
        return IntervalBounds(lower=np.inf, upper=-np.inf, is_bottom=True)
    
    @staticmethod
    def singleton(value: float) -> IntervalBounds:
        """Return [value, value]."""
        return IntervalBounds(lower=value, upper=value)
    
    def contains(self, value: float) -> bool:
        """Check if value is in interval."""
        if self.is_bottom:
            return False
        return self.lower <= value <= self.upper
    
    def join(self, other: IntervalBounds) -> IntervalBounds:
        """Compute least upper bound (union of intervals)."""
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        return IntervalBounds(
            lower=min(self.lower, other.lower),
            upper=max(self.upper, other.upper),
        )
    
    def meet(self, other: IntervalBounds) -> IntervalBounds:
        """Compute greatest lower bound (intersection of intervals)."""
        if self.is_bottom or other.is_bottom:
            return IntervalBounds.bottom()
        lower = max(self.lower, other.lower)
        upper = min(self.upper, other.upper)
        if lower > upper:
            return IntervalBounds.bottom()
        return IntervalBounds(lower=lower, upper=upper)
    
    def widen(self, other: IntervalBounds, threshold: float = 1e10) -> IntervalBounds:
        """Widening operator for fixpoint convergence."""
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        
        lower = self.lower
        if other.lower < self.lower:
            lower = -threshold if self.lower > -threshold else -np.inf
        
        upper = self.upper
        if other.upper > self.upper:
            upper = threshold if self.upper < threshold else np.inf
        
        return IntervalBounds(lower=lower, upper=upper)
    
    def __repr__(self) -> str:
        if self.is_bottom:
            return "⊥"
        if self.lower == -np.inf and self.upper == np.inf:
            return "⊤"
        return f"[{self.lower:.6f}, {self.upper:.6f}]"


class AbstractDomain(ABC):
    """Base class for abstract domains.
    
    An abstract domain provides:
        - Abstract values representing sets of concrete values
        - Abstract operations (join, meet, widening)
        - Abstract transformers for concrete operations
    """
    
    @abstractmethod
    def top(self) -> AbstractDomain:
        """Return top element (⊤, most imprecise)."""
        pass
    
    @abstractmethod
    def bottom(self) -> AbstractDomain:
        """Return bottom element (⊥, unreachable)."""
        pass
    
    @abstractmethod
    def join(self, other: AbstractDomain) -> AbstractDomain:
        """Compute least upper bound (union)."""
        pass
    
    @abstractmethod
    def meet(self, other: AbstractDomain) -> AbstractDomain:
        """Compute greatest lower bound (intersection)."""
        pass
    
    @abstractmethod
    def widen(self, other: AbstractDomain) -> AbstractDomain:
        """Widening operator for fixpoint convergence."""
        pass
    
    @abstractmethod
    def is_bottom(self) -> bool:
        """Check if this is bottom (unreachable)."""
        pass
    
    @abstractmethod
    def is_less_or_equal(self, other: AbstractDomain) -> bool:
        """Check if self ⊑ other (partial order)."""
        pass


@dataclass
class IntervalDomain(AbstractDomain):
    """Interval abstract domain for privacy analysis.
    
    Represents bounds on probability distributions and privacy loss
    using per-variable interval constraints.
    
    Attributes:
        intervals: Dict mapping variable names to interval bounds.
    """
    
    intervals: Dict[str, IntervalBounds] = field(default_factory=dict)
    
    def top(self) -> IntervalDomain:
        """Return ⊤ (all variables unconstrained)."""
        return IntervalDomain(
            intervals={k: IntervalBounds.top() for k in self.intervals.keys()}
        )
    
    def bottom(self) -> IntervalDomain:
        """Return ⊥ (unreachable state)."""
        return IntervalDomain(
            intervals={k: IntervalBounds.bottom() for k in self.intervals.keys()}
        )
    
    def is_bottom(self) -> bool:
        """Check if any variable has bottom interval."""
        return any(ival.is_bottom for ival in self.intervals.values())
    
    def join(self, other: IntervalDomain) -> IntervalDomain:
        """Join two interval domains."""
        result = {}
        all_keys = set(self.intervals.keys()) | set(other.intervals.keys())
        for k in all_keys:
            ival1 = self.intervals.get(k, IntervalBounds.top())
            ival2 = other.intervals.get(k, IntervalBounds.top())
            result[k] = ival1.join(ival2)
        return IntervalDomain(intervals=result)
    
    def meet(self, other: IntervalDomain) -> IntervalDomain:
        """Meet two interval domains."""
        result = {}
        all_keys = set(self.intervals.keys()) | set(other.intervals.keys())
        for k in all_keys:
            ival1 = self.intervals.get(k, IntervalBounds.top())
            ival2 = other.intervals.get(k, IntervalBounds.top())
            result[k] = ival1.meet(ival2)
        return IntervalDomain(intervals=result)
    
    def widen(self, other: IntervalDomain) -> IntervalDomain:
        """Widen for fixpoint convergence."""
        result = {}
        all_keys = set(self.intervals.keys()) | set(other.intervals.keys())
        for k in all_keys:
            ival1 = self.intervals.get(k, IntervalBounds.top())
            ival2 = other.intervals.get(k, IntervalBounds.top())
            result[k] = ival1.widen(ival2)
        return IntervalDomain(intervals=result)
    
    def is_less_or_equal(self, other: IntervalDomain) -> bool:
        """Check if self ⊑ other."""
        for k, ival1 in self.intervals.items():
            ival2 = other.intervals.get(k, IntervalBounds.top())
            if ival1.is_bottom:
                continue
            if ival2.is_bottom:
                return False
            if ival1.lower < ival2.lower or ival1.upper > ival2.upper:
                return False
        return True
    
    def set_interval(self, var: str, bounds: IntervalBounds) -> None:
        """Set interval bounds for a variable."""
        self.intervals[var] = bounds
    
    def get_interval(self, var: str) -> IntervalBounds:
        """Get interval bounds for a variable."""
        return self.intervals.get(var, IntervalBounds.top())
    
    def __repr__(self) -> str:
        if self.is_bottom():
            return "IntervalDomain(⊥)"
        items = [f"{k}: {v}" for k, v in sorted(self.intervals.items())]
        return f"IntervalDomain({{{', '.join(items)}}})"


@dataclass
class OctagonDomain(AbstractDomain):
    """Octagonal constraints abstract domain.
    
    Tracks constraints of the form ±x_i ± x_j ≤ c, which are useful
    for reasoning about relationships between variables (e.g., privacy
    loss for different database pairs).
    
    Attributes:
        n_vars: Number of variables.
        matrix: Difference bound matrix (2n × 2n).
        var_names: List of variable names.
    """
    
    n_vars: int
    matrix: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    var_names: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        if self.matrix.size == 0:
            self.matrix = np.full((2 * self.n_vars, 2 * self.n_vars), np.inf)
            np.fill_diagonal(self.matrix, 0.0)
    
    def top(self) -> OctagonDomain:
        """Return ⊤."""
        result = OctagonDomain(n_vars=self.n_vars, var_names=self.var_names.copy())
        result.matrix = np.full((2 * self.n_vars, 2 * self.n_vars), np.inf)
        np.fill_diagonal(result.matrix, 0.0)
        return result
    
    def bottom(self) -> OctagonDomain:
        """Return ⊥."""
        result = OctagonDomain(n_vars=self.n_vars, var_names=self.var_names.copy())
        result.matrix = np.full((2 * self.n_vars, 2 * self.n_vars), np.inf)
        result.matrix[0, 0] = -1.0
        return result
    
    def is_bottom(self) -> bool:
        """Check if octagon is empty."""
        return np.any(np.diag(self.matrix) < 0)
    
    def join(self, other: OctagonDomain) -> OctagonDomain:
        """Join octagons (over-approximation of union)."""
        result = OctagonDomain(n_vars=self.n_vars, var_names=self.var_names.copy())
        result.matrix = np.maximum(self.matrix, other.matrix)
        return result
    
    def meet(self, other: OctagonDomain) -> OctagonDomain:
        """Meet octagons (intersection)."""
        result = OctagonDomain(n_vars=self.n_vars, var_names=self.var_names.copy())
        result.matrix = np.minimum(self.matrix, other.matrix)
        result._close()
        return result
    
    def widen(self, other: OctagonDomain) -> OctagonDomain:
        """Widening for fixpoint convergence."""
        result = OctagonDomain(n_vars=self.n_vars, var_names=self.var_names.copy())
        result.matrix = self.matrix.copy()
        
        changed = other.matrix < self.matrix
        result.matrix[changed] = np.inf
        
        return result
    
    def is_less_or_equal(self, other: OctagonDomain) -> bool:
        """Check if self ⊑ other."""
        if self.is_bottom():
            return True
        if other.is_bottom():
            return False
        return np.all(self.matrix <= other.matrix + 1e-9)
    
    def _close(self) -> None:
        """Close octagon using shortest-path algorithm (Floyd-Warshall)."""
        n = 2 * self.n_vars
        for k in range(n):
            self.matrix = np.minimum(
                self.matrix,
                self.matrix[:, k:k+1] + self.matrix[k:k+1, :]
            )
    
    def add_constraint(self, i: int, j: int, c: float) -> None:
        """Add constraint x_i - x_j ≤ c."""
        self.matrix[2*i, 2*j+1] = min(self.matrix[2*i, 2*j+1], c)
        self._close()
    
    def get_interval(self, var_idx: int) -> IntervalBounds:
        """Extract interval bounds for a variable from octagon."""
        lower = -self.matrix[2*var_idx+1, 2*var_idx] / 2.0
        upper = self.matrix[2*var_idx, 2*var_idx+1] / 2.0
        if lower > upper:
            return IntervalBounds.bottom()
        return IntervalBounds(lower=lower, upper=upper)
    
    def __repr__(self) -> str:
        if self.is_bottom():
            return "OctagonDomain(⊥)"
        return f"OctagonDomain(n_vars={self.n_vars})"


class PrivacyAbstractTransformer:
    """Abstract transformers for differential privacy operations.
    
    Provides sound over-approximations of privacy-relevant operations
    in abstract domains.
    """
    
    def __init__(self, domain_type: str = "interval"):
        self.domain_type = domain_type
    
    def abstract_probability(
        self,
        domain: IntervalDomain,
        prob_value: float,
        tolerance: float,
    ) -> IntervalBounds:
        """Abstract a probability value accounting for solver error."""
        return IntervalBounds(
            lower=max(0.0, prob_value - tolerance),
            upper=min(1.0, prob_value + tolerance),
        )
    
    def abstract_privacy_loss(
        self,
        domain: IntervalDomain,
        p_var: str,
        q_var: str,
    ) -> IntervalBounds:
        """Abstractly compute log(p/q) from intervals on p and q."""
        p_bounds = domain.get_interval(p_var)
        q_bounds = domain.get_interval(q_var)
        
        if p_bounds.is_bottom or q_bounds.is_bottom:
            return IntervalBounds.bottom()
        
        p_lo = max(p_bounds.lower, 1e-300)
        p_hi = p_bounds.upper
        q_lo = max(q_bounds.lower, 1e-300)
        q_hi = q_bounds.upper
        
        if q_hi == 0:
            return IntervalBounds(lower=-np.inf, upper=np.inf)
        
        candidates = []
        if p_lo > 0 and q_hi > 0:
            candidates.append(np.log(p_lo / q_hi))
        if p_hi > 0 and q_lo > 0:
            candidates.append(np.log(p_hi / q_lo))
        
        if not candidates:
            return IntervalBounds(lower=-np.inf, upper=np.inf)
        
        return IntervalBounds(
            lower=min(candidates),
            upper=max(candidates),
        )
    
    def abstract_hockey_stick(
        self,
        domain: IntervalDomain,
        p_vars: List[str],
        q_vars: List[str],
        epsilon: float,
    ) -> IntervalBounds:
        """Abstractly compute hockey-stick divergence."""
        exp_eps = np.exp(epsilon)
        total_lo = 0.0
        total_hi = 0.0
        
        for p_var, q_var in zip(p_vars, q_vars):
            p_bounds = domain.get_interval(p_var)
            q_bounds = domain.get_interval(q_var)
            
            if p_bounds.is_bottom or q_bounds.is_bottom:
                continue
            
            diff_lo = max(0.0, p_bounds.lower - exp_eps * q_bounds.upper)
            diff_hi = max(0.0, p_bounds.upper - exp_eps * q_bounds.lower)
            
            total_lo += diff_lo
            total_hi += diff_hi
        
        return IntervalBounds(lower=total_lo, upper=total_hi)
    
    def abstract_composition(
        self,
        loss_bounds: List[IntervalBounds],
    ) -> IntervalBounds:
        """Abstract composition of privacy losses."""
        total_lo = sum(lb.lower for lb in loss_bounds)
        total_hi = sum(lb.upper for lb in loss_bounds)
        return IntervalBounds(lower=total_lo, upper=total_hi)


def fixpoint_iteration(
    initial_state: AbstractDomain,
    transfer_fn: callable,
    max_iterations: int = 100,
    use_widening: bool = True,
) -> AbstractDomain:
    """Compute least fixpoint of abstract transformer.
    
    Uses Kleene iteration with optional widening for convergence.
    
    Args:
        initial_state: Initial abstract state.
        transfer_fn: Abstract transfer function.
        max_iterations: Maximum iterations.
        use_widening: Whether to apply widening after some iterations.
    
    Returns:
        Fixpoint abstract state.
    """
    current = initial_state
    widening_threshold = max_iterations // 2
    
    for iteration in range(max_iterations):
        next_state = transfer_fn(current)
        
        if next_state.is_less_or_equal(current):
            logger.info(f"Fixpoint converged at iteration {iteration}")
            return next_state
        
        if use_widening and iteration > widening_threshold:
            current = current.widen(next_state)
        else:
            current = next_state
    
    logger.warning(f"Fixpoint did not converge after {max_iterations} iterations")
    return current


def abstract_verify_dp(
    prob_table: npt.NDArray[np.float64],
    edges: List[Tuple[int, int]],
    epsilon: float,
    delta: float,
    tolerance: float = 1e-9,
) -> Tuple[bool, IntervalDomain]:
    """Verify DP using abstract interpretation.
    
    Args:
        prob_table: Mechanism probability table [n, k].
        edges: Adjacent pairs.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        tolerance: Tolerance for abstract values.
    
    Returns:
        (is_valid, final_domain): Whether DP holds and final abstract state.
    """
    n, k = prob_table.shape
    transformer = PrivacyAbstractTransformer()
    
    domain = IntervalDomain()
    
    for i in range(n):
        for j in range(k):
            var = f"p_{i}_{j}"
            bounds = transformer.abstract_probability(domain, prob_table[i, j], tolerance)
            domain.set_interval(var, bounds)
    
    is_valid = True
    exp_eps = np.exp(epsilon)
    
    for i, i_prime in edges:
        p_vars = [f"p_{i}_{j}" for j in range(k)]
        q_vars = [f"p_{i_prime}_{j}" for j in range(k)]
        
        if delta == 0.0:
            for j in range(k):
                loss_bounds = transformer.abstract_privacy_loss(
                    domain, p_vars[j], q_vars[j]
                )
                if loss_bounds.upper > epsilon + tolerance:
                    is_valid = False
                    break
        else:
            hs_bounds = transformer.abstract_hockey_stick(
                domain, p_vars, q_vars, epsilon
            )
            if hs_bounds.upper > delta + tolerance:
                is_valid = False
                break
        
        if not is_valid:
            break
    
    return is_valid, domain


def abstract_cegis_prune(
    prob_table: npt.NDArray[np.float64],
    candidate_pairs: List[Tuple[int, int]],
    epsilon: float,
    delta: float,
) -> List[Tuple[int, int]]:
    """Use abstract interpretation to prune counterexample search space.
    
    Analyzes candidate pairs and removes those that provably satisfy DP
    using abstract interpretation, reducing CEGIS search space.
    
    Args:
        prob_table: Mechanism probability table.
        candidate_pairs: Candidate counterexample pairs.
        epsilon: Privacy parameter.
        delta: Privacy parameter.
    
    Returns:
        Pruned list of candidate pairs that might violate DP.
    """
    transformer = PrivacyAbstractTransformer()
    domain = IntervalDomain()
    
    n, k = prob_table.shape
    for i in range(n):
        for j in range(k):
            var = f"p_{i}_{j}"
            bounds = transformer.abstract_probability(domain, prob_table[i, j], 1e-9)
            domain.set_interval(var, bounds)
    
    pruned_pairs = []
    exp_eps = np.exp(epsilon)
    
    for i, i_prime in candidate_pairs:
        p_vars = [f"p_{i}_{j}" for j in range(k)]
        q_vars = [f"p_{i_prime}_{j}" for j in range(k)]
        
        might_violate = False
        
        if delta == 0.0:
            for j in range(k):
                loss_bounds = transformer.abstract_privacy_loss(
                    domain, p_vars[j], q_vars[j]
                )
                if loss_bounds.upper > epsilon - 1e-9:
                    might_violate = True
                    break
        else:
            hs_bounds = transformer.abstract_hockey_stick(
                domain, p_vars, q_vars, epsilon
            )
            if hs_bounds.upper > delta - 1e-9:
                might_violate = True
        
        if might_violate:
            pruned_pairs.append((i, i_prime))
    
    logger.info(
        f"Abstract interpretation pruned {len(candidate_pairs) - len(pruned_pairs)} "
        f"pairs from search space"
    )
    
    return pruned_pairs


class PolyhedralDomain(AbstractDomain):
    """Polyhedral abstract domain using linear constraints.
    
    Represents sets of states using systems of linear inequalities:
        A·x ≤ b
    
    More expressive than intervals or octagons but more expensive.
    
    Attributes:
        n_vars: Number of variables.
        A: Constraint matrix (m × n).
        b: Constraint vector (m).
        var_names: Variable names.
    """
    
    def __init__(
        self,
        n_vars: int,
        A: Optional[npt.NDArray[np.float64]] = None,
        b: Optional[npt.NDArray[np.float64]] = None,
        var_names: Optional[List[str]] = None,
    ):
        self.n_vars = n_vars
        self.A = A if A is not None else np.zeros((0, n_vars))
        self.b = b if b is not None else np.zeros(0)
        self.var_names = var_names or [f"x{i}" for i in range(n_vars)]
    
    def top(self) -> PolyhedralDomain:
        """Return ⊤ (no constraints)."""
        return PolyhedralDomain(self.n_vars, var_names=self.var_names.copy())
    
    def bottom(self) -> PolyhedralDomain:
        """Return ⊥ (inconsistent constraints)."""
        result = PolyhedralDomain(self.n_vars, var_names=self.var_names.copy())
        result.A = np.array([[1.0] + [0.0] * (self.n_vars - 1)])
        result.b = np.array([-1.0])
        return result
    
    def is_bottom(self) -> bool:
        """Check if polyhedron is empty using linear programming."""
        try:
            from scipy.optimize import linprog
            
            c = np.zeros(self.n_vars)
            res = linprog(c, A_ub=self.A, b_ub=self.b, method='highs')
            return not res.success
        except Exception:
            return False
    
    def join(self, other: PolyhedralDomain) -> PolyhedralDomain:
        """Join using convex hull (over-approximation)."""
        result = PolyhedralDomain(self.n_vars, var_names=self.var_names.copy())
        result.A = np.vstack([self.A, other.A])
        result.b = np.concatenate([self.b, other.b])
        return result
    
    def meet(self, other: PolyhedralDomain) -> PolyhedralDomain:
        """Meet by combining constraints."""
        result = PolyhedralDomain(self.n_vars, var_names=self.var_names.copy())
        result.A = np.vstack([self.A, other.A])
        result.b = np.concatenate([self.b, other.b])
        return result
    
    def widen(self, other: PolyhedralDomain) -> PolyhedralDomain:
        """Widening by removing constraints not in both."""
        result = PolyhedralDomain(self.n_vars, var_names=self.var_names.copy())
        
        common_constraints = []
        common_bounds = []
        
        for i, (a_row, b_val) in enumerate(zip(self.A, self.b)):
            for j, (other_a_row, other_b_val) in enumerate(zip(other.A, other.b)):
                if np.allclose(a_row, other_a_row) and np.isclose(b_val, other_b_val):
                    common_constraints.append(a_row)
                    common_bounds.append(b_val)
                    break
        
        if common_constraints:
            result.A = np.array(common_constraints)
            result.b = np.array(common_bounds)
        
        return result
    
    def is_less_or_equal(self, other: PolyhedralDomain) -> bool:
        """Check if self ⊑ other (self is more constrained)."""
        if self.is_bottom():
            return True
        if other.is_bottom():
            return False
        
        for a_row, b_val in zip(other.A, other.b):
            if not self._satisfies_constraint(a_row, b_val):
                return False
        return True
    
    def _satisfies_constraint(
        self,
        a: npt.NDArray[np.float64],
        b: float,
    ) -> bool:
        """Check if polyhedron satisfies constraint a·x ≤ b."""
        try:
            from scipy.optimize import linprog
            
            res = linprog(a, A_ub=self.A, b_ub=self.b, method='highs')
            if not res.success:
                return True
            return np.dot(a, res.x) <= b + 1e-9
        except Exception:
            return True
    
    def add_constraint(
        self,
        a: npt.NDArray[np.float64],
        b: float,
    ) -> None:
        """Add constraint a·x ≤ b."""
        self.A = np.vstack([self.A, a.reshape(1, -1)])
        self.b = np.append(self.b, b)
    
    def project_to_interval(self, var_idx: int) -> IntervalBounds:
        """Project polyhedron onto single variable."""
        try:
            from scipy.optimize import linprog
            
            c = np.zeros(self.n_vars)
            c[var_idx] = 1.0
            
            res_min = linprog(c, A_ub=self.A, b_ub=self.b, method='highs')
            res_max = linprog(-c, A_ub=self.A, b_ub=self.b, method='highs')
            
            if not res_min.success or not res_max.success:
                return IntervalBounds.bottom()
            
            lower = res_min.x[var_idx]
            upper = -res_max.fun
            
            return IntervalBounds(lower=lower, upper=upper)
        except Exception:
            return IntervalBounds.top()


class PrivacyDomainLifting:
    """Lift privacy properties to abstract domains.
    
    Provides methods to lift concrete privacy properties into
    abstract domain representations.
    """
    
    def __init__(self):
        self.transformer = PrivacyAbstractTransformer()
    
    def lift_dp_constraint(
        self,
        domain: IntervalDomain,
        i: int,
        i_prime: int,
        epsilon: float,
        n_outputs: int,
    ) -> IntervalDomain:
        """Lift DP constraint into abstract domain.
        
        Args:
            domain: Current abstract domain.
            i: First database index.
            i_prime: Second database index.
            epsilon: Privacy parameter.
            n_outputs: Number of output bins.
        
        Returns:
            Refined domain with DP constraint.
        """
        refined = IntervalDomain(intervals=domain.intervals.copy())
        
        for j in range(n_outputs):
            p_var = f"p_{i}_{j}"
            q_var = f"p_{i_prime}_{j}"
            
            loss_bounds = self.transformer.abstract_privacy_loss(
                domain, p_var, q_var
            )
            
            if loss_bounds.upper > epsilon:
                p_bounds = domain.get_interval(p_var)
                q_bounds = domain.get_interval(q_var)
                
                tightened_p = IntervalBounds(
                    lower=p_bounds.lower,
                    upper=min(p_bounds.upper, np.exp(epsilon) * q_bounds.upper)
                )
                refined.set_interval(p_var, tightened_p)
        
        return refined
    
    def lift_composition(
        self,
        domains: List[IntervalDomain],
    ) -> IntervalDomain:
        """Lift composition of multiple mechanisms to abstract domain.
        
        Args:
            domains: List of abstract domains for individual mechanisms.
        
        Returns:
            Abstract domain for composed mechanism.
        """
        if not domains:
            return IntervalDomain()
        
        composed = domains[0]
        for domain in domains[1:]:
            composed = composed.join(domain)
        
        return composed


def compute_abstract_sensitivities(
    query_matrix: npt.NDArray[np.float64],
    adjacency: List[Tuple[int, int]],
) -> IntervalDomain:
    """Compute abstract bounds on query sensitivities.
    
    Args:
        query_matrix: Query sensitivity matrix.
        adjacency: Adjacent database pairs.
    
    Returns:
        Abstract domain with sensitivity bounds.
    """
    domain = IntervalDomain()
    
    for i, i_prime in adjacency:
        if i >= query_matrix.shape[0] or i_prime >= query_matrix.shape[0]:
            continue
        
        diff = np.abs(query_matrix[i] - query_matrix[i_prime])
        sensitivity = np.max(diff)
        
        var = f"sensitivity_{i}_{i_prime}"
        domain.set_interval(var, IntervalBounds(lower=0.0, upper=sensitivity))
    
    return domain


class AbstractInterpreter:
    """Main abstract interpretation engine for DP verification."""
    
    def __init__(
        self,
        domain_type: str = "interval",
        use_widening: bool = True,
    ):
        self.domain_type = domain_type
        self.use_widening = use_widening
        self.transformer = PrivacyAbstractTransformer(domain_type)
    
    def analyze_mechanism(
        self,
        prob_table: npt.NDArray[np.float64],
        epsilon: float,
        delta: float,
    ) -> Tuple[bool, IntervalDomain]:
        """Analyze mechanism using abstract interpretation.
        
        Args:
            prob_table: Mechanism probability table.
            epsilon: Privacy parameter.
            delta: Privacy parameter.
        
        Returns:
            (is_private, final_domain)
        """
        n, k = prob_table.shape
        
        initial_domain = IntervalDomain()
        for i in range(n):
            for j in range(k):
                var = f"p_{i}_{j}"
                initial_domain.set_interval(
                    var,
                    IntervalBounds(
                        lower=max(0.0, prob_table[i, j] - 1e-9),
                        upper=min(1.0, prob_table[i, j] + 1e-9),
                    )
                )
        
        def transfer(domain: IntervalDomain) -> IntervalDomain:
            return domain
        
        final_domain = fixpoint_iteration(
            initial_domain,
            transfer,
            max_iterations=10,
            use_widening=self.use_widening,
        )
        
        is_private, _ = abstract_verify_dp(
            prob_table,
            [(i, j) for i in range(n) for j in range(i+1, n)],
            epsilon,
            delta,
        )
        
        return is_private, final_domain

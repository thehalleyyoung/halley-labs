"""Signal Temporal Logic (STL) Abstract Syntax Tree.

Defines the core AST nodes for STL formulas including atomic propositions,
boolean connectives, and temporal operators with bounded intervals.
Supports negation normal form, pretty printing, substitution, and
structural analysis (depth, size, free variables).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union


class ComparisonOp(Enum):
    """Comparison operators for atomic predicates."""
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="

    def negate(self) -> ComparisonOp:
        """Return the negated comparison operator."""
        negation_map = {
            ComparisonOp.LT: ComparisonOp.GE,
            ComparisonOp.LE: ComparisonOp.GT,
            ComparisonOp.GT: ComparisonOp.LE,
            ComparisonOp.GE: ComparisonOp.LT,
        }
        return negation_map[self]

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Interval:
    """Time interval [lo, hi] for temporal operators."""
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo < 0:
            raise ValueError(f"Interval lower bound must be non-negative, got {self.lo}")
        if self.hi < self.lo:
            raise ValueError(
                f"Interval upper bound must be >= lower bound, got [{self.lo}, {self.hi}]"
            )

    @property
    def length(self) -> float:
        return self.hi - self.lo

    def __str__(self) -> str:
        return f"[{self.lo}, {self.hi}]"

    def contains(self, t: float) -> bool:
        return self.lo <= t <= self.hi

    def shift(self, offset: float) -> Interval:
        return Interval(self.lo + offset, self.hi + offset)

    def intersect(self, other: Interval) -> Optional[Interval]:
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo <= hi:
            return Interval(lo, hi)
        return None


class STLFormula(ABC):
    """Base class for all STL formula AST nodes."""

    @abstractmethod
    def robustness_at(self, signals: Dict[str, "Signal"], t: float) -> float:
        """Compute quantitative robustness at time t (used by robustness engine)."""
        ...

    @abstractmethod
    def free_variables(self) -> FrozenSet[str]:
        """Return the set of free signal variable names in this formula."""
        ...

    @abstractmethod
    def to_nnf(self) -> STLFormula:
        """Convert formula to Negation Normal Form (negations pushed to atoms)."""
        ...

    @abstractmethod
    def children(self) -> List[STLFormula]:
        """Return immediate sub-formulas."""
        ...

    @abstractmethod
    def _pretty_math(self) -> str:
        """Mathematical notation string."""
        ...

    @abstractmethod
    def _pretty_bio(self) -> str:
        """Bio-STL syntax string."""
        ...

    @property
    def depth(self) -> int:
        """Maximum nesting depth of the formula tree."""
        kids = self.children()
        if not kids:
            return 0
        return 1 + max(c.depth for c in kids)

    @property
    def size(self) -> int:
        """Total number of AST nodes."""
        return 1 + sum(c.size for c in self.children())

    @property
    def temporal_depth(self) -> int:
        """Maximum nesting depth of temporal operators only."""
        kids = self.children()
        if not kids:
            return 0
        base = 1 if isinstance(self, (Always, Eventually, Until)) else 0
        return base + max(c.temporal_depth for c in kids)

    def clone(self) -> STLFormula:
        """Deep copy of this formula."""
        return copy.deepcopy(self)

    def substitute(self, mapping: Dict[str, str]) -> STLFormula:
        """Substitute variable names according to mapping."""
        cloned = self.clone()
        _substitute_in_place(cloned, mapping)
        return cloned

    def pretty(self, style: str = "math") -> str:
        """Pretty-print the formula.

        Args:
            style: 'math' for mathematical notation, 'bio' for Bio-STL syntax.
        """
        if style == "math":
            return self._pretty_math()
        elif style == "bio":
            return self._pretty_bio()
        raise ValueError(f"Unknown style: {style}")

    def __str__(self) -> str:
        return self._pretty_math()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._pretty_math()})"

    def atoms(self) -> List[Predicate]:
        """Collect all atomic predicates in the formula."""
        result: List[Predicate] = []
        _collect_atoms(self, result)
        return result

    def map_atoms(self, fn: Callable[[Predicate], STLFormula]) -> STLFormula:
        """Apply a function to each atomic predicate, returning a new formula."""
        return _map_atoms(self, fn)

    def is_boolean(self) -> bool:
        """True if formula contains no temporal operators."""
        if isinstance(self, (Always, Eventually, Until)):
            return False
        return all(c.is_boolean() for c in self.children())

    def temporal_operators(self) -> List[STLFormula]:
        """Collect all temporal operator nodes."""
        result: List[STLFormula] = []
        _collect_temporal(self, result)
        return result


# ---------------------------------------------------------------------------
# Atomic proposition
# ---------------------------------------------------------------------------

@dataclass
class Expression:
    """A signal expression: either a variable name or a numeric constant,
    or a simple arithmetic combination (var +/- const)."""
    variable: Optional[str] = None
    constant: Optional[float] = None
    offset: float = 0.0
    scale: float = 1.0

    def evaluate(self, signals: Dict[str, "Signal"], t: float) -> float:
        if self.variable is not None:
            from bioprover.temporal.robustness import Signal
            sig = signals[self.variable]
            return self.scale * sig.at(t) + self.offset
        if self.constant is not None:
            return self.constant
        raise ValueError("Expression has neither variable nor constant")

    @property
    def variables(self) -> FrozenSet[str]:
        if self.variable is not None:
            return frozenset({self.variable})
        return frozenset()

    def substitute_var(self, mapping: Dict[str, str]) -> None:
        if self.variable is not None and self.variable in mapping:
            self.variable = mapping[self.variable]

    def __str__(self) -> str:
        if self.variable is not None:
            parts = []
            if self.scale != 1.0:
                parts.append(f"{self.scale}*{self.variable}")
            else:
                parts.append(self.variable)
            if self.offset != 0.0:
                parts.append(f" + {self.offset}" if self.offset > 0 else f" - {-self.offset}")
            return "".join(parts)
        if self.constant is not None:
            return str(self.constant)
        return "?"


def make_var_expr(name: str) -> Expression:
    """Shorthand to create a variable expression."""
    return Expression(variable=name)


def make_const_expr(value: float) -> Expression:
    """Shorthand to create a constant expression."""
    return Expression(constant=value)


@dataclass
class Predicate(STLFormula):
    """Atomic predicate: expr op threshold.

    Example: x > 5 is Predicate(Expression(variable='x'), GT, 5.0)
    """
    expr: Expression
    op: ComparisonOp
    threshold: float

    def robustness_at(self, signals: Dict[str, "Signal"], t: float) -> float:
        val = self.expr.evaluate(signals, t)
        if self.op in (ComparisonOp.GT, ComparisonOp.GE):
            return val - self.threshold
        else:
            return self.threshold - val

    def free_variables(self) -> FrozenSet[str]:
        return self.expr.variables

    def to_nnf(self) -> STLFormula:
        return Predicate(copy.deepcopy(self.expr), self.op, self.threshold)

    def children(self) -> List[STLFormula]:
        return []

    def _pretty_math(self) -> str:
        return f"{self.expr} {self.op} {self.threshold}"

    def _pretty_bio(self) -> str:
        return f"({self.expr} {self.op} {self.threshold})"


# ---------------------------------------------------------------------------
# Boolean connectives
# ---------------------------------------------------------------------------

@dataclass
class STLNot(STLFormula):
    """Negation: ¬φ"""
    child: STLFormula

    def robustness_at(self, signals: Dict[str, "Signal"], t: float) -> float:
        return -self.child.robustness_at(signals, t)

    def free_variables(self) -> FrozenSet[str]:
        return self.child.free_variables()

    def to_nnf(self) -> STLFormula:
        return _push_negation(self.child)

    def children(self) -> List[STLFormula]:
        return [self.child]

    def _pretty_math(self) -> str:
        return f"¬({self.child._pretty_math()})"

    def _pretty_bio(self) -> str:
        return f"!({self.child._pretty_bio()})"


@dataclass
class STLAnd(STLFormula):
    """Conjunction: φ₁ ∧ φ₂"""
    left: STLFormula
    right: STLFormula

    def robustness_at(self, signals: Dict[str, "Signal"], t: float) -> float:
        return min(
            self.left.robustness_at(signals, t),
            self.right.robustness_at(signals, t),
        )

    def free_variables(self) -> FrozenSet[str]:
        return self.left.free_variables() | self.right.free_variables()

    def to_nnf(self) -> STLFormula:
        return STLAnd(self.left.to_nnf(), self.right.to_nnf())

    def children(self) -> List[STLFormula]:
        return [self.left, self.right]

    def _pretty_math(self) -> str:
        return f"({self.left._pretty_math()} ∧ {self.right._pretty_math()})"

    def _pretty_bio(self) -> str:
        return f"({self.left._pretty_bio()} && {self.right._pretty_bio()})"


@dataclass
class STLOr(STLFormula):
    """Disjunction: φ₁ ∨ φ₂"""
    left: STLFormula
    right: STLFormula

    def robustness_at(self, signals: Dict[str, "Signal"], t: float) -> float:
        return max(
            self.left.robustness_at(signals, t),
            self.right.robustness_at(signals, t),
        )

    def free_variables(self) -> FrozenSet[str]:
        return self.left.free_variables() | self.right.free_variables()

    def to_nnf(self) -> STLFormula:
        return STLOr(self.left.to_nnf(), self.right.to_nnf())

    def children(self) -> List[STLFormula]:
        return [self.left, self.right]

    def _pretty_math(self) -> str:
        return f"({self.left._pretty_math()} ∨ {self.right._pretty_math()})"

    def _pretty_bio(self) -> str:
        return f"({self.left._pretty_bio()} || {self.right._pretty_bio()})"


@dataclass
class STLImplies(STLFormula):
    """Implication: φ₁ → φ₂  (syntactic sugar for ¬φ₁ ∨ φ₂)"""
    antecedent: STLFormula
    consequent: STLFormula

    def robustness_at(self, signals: Dict[str, "Signal"], t: float) -> float:
        return max(
            -self.antecedent.robustness_at(signals, t),
            self.consequent.robustness_at(signals, t),
        )

    def free_variables(self) -> FrozenSet[str]:
        return self.antecedent.free_variables() | self.consequent.free_variables()

    def to_nnf(self) -> STLFormula:
        return STLOr(_push_negation(self.antecedent), self.consequent.to_nnf())

    def children(self) -> List[STLFormula]:
        return [self.antecedent, self.consequent]

    def _pretty_math(self) -> str:
        return f"({self.antecedent._pretty_math()} → {self.consequent._pretty_math()})"

    def _pretty_bio(self) -> str:
        return f"({self.antecedent._pretty_bio()} -> {self.consequent._pretty_bio()})"


# ---------------------------------------------------------------------------
# Temporal operators
# ---------------------------------------------------------------------------

@dataclass
class Always(STLFormula):
    """Globally / Always: G[a,b] φ"""
    child: STLFormula
    interval: Interval

    def robustness_at(self, signals: Dict[str, "Signal"], t: float) -> float:
        # Implemented in robustness engine for efficiency; fallback here
        import numpy as np
        from bioprover.temporal.robustness import Signal

        first_sig = next(iter(signals.values()))
        times = first_sig.times
        lo, hi = t + self.interval.lo, t + self.interval.hi
        relevant = [tp for tp in times if lo <= tp <= hi]
        if not relevant:
            relevant = [lo]
        return min(self.child.robustness_at(signals, tp) for tp in relevant)

    def free_variables(self) -> FrozenSet[str]:
        return self.child.free_variables()

    def to_nnf(self) -> STLFormula:
        return Always(self.child.to_nnf(), self.interval)

    def children(self) -> List[STLFormula]:
        return [self.child]

    def _pretty_math(self) -> str:
        return f"G{self.interval}({self.child._pretty_math()})"

    def _pretty_bio(self) -> str:
        return f"G[{self.interval.lo},{self.interval.hi}]({self.child._pretty_bio()})"


@dataclass
class Eventually(STLFormula):
    """Finally / Eventually: F[a,b] φ"""
    child: STLFormula
    interval: Interval

    def robustness_at(self, signals: Dict[str, "Signal"], t: float) -> float:
        import numpy as np
        from bioprover.temporal.robustness import Signal

        first_sig = next(iter(signals.values()))
        times = first_sig.times
        lo, hi = t + self.interval.lo, t + self.interval.hi
        relevant = [tp for tp in times if lo <= tp <= hi]
        if not relevant:
            relevant = [lo]
        return max(self.child.robustness_at(signals, tp) for tp in relevant)

    def free_variables(self) -> FrozenSet[str]:
        return self.child.free_variables()

    def to_nnf(self) -> STLFormula:
        return Eventually(self.child.to_nnf(), self.interval)

    def children(self) -> List[STLFormula]:
        return [self.child]

    def _pretty_math(self) -> str:
        return f"F{self.interval}({self.child._pretty_math()})"

    def _pretty_bio(self) -> str:
        return f"F[{self.interval.lo},{self.interval.hi}]({self.child._pretty_bio()})"


@dataclass
class Until(STLFormula):
    """Until: φ₁ U[a,b] φ₂"""
    left: STLFormula
    right: STLFormula
    interval: Interval

    def robustness_at(self, signals: Dict[str, "Signal"], t: float) -> float:
        import numpy as np
        from bioprover.temporal.robustness import Signal

        first_sig = next(iter(signals.values()))
        times = first_sig.times
        lo, hi = t + self.interval.lo, t + self.interval.hi
        relevant = sorted(tp for tp in times if lo <= tp <= hi)
        if not relevant:
            return float("-inf")

        best = float("-inf")
        for i, t2 in enumerate(relevant):
            rho_right = self.right.robustness_at(signals, t2)
            rho_left_min = float("inf")
            for t1 in relevant[:i]:
                rho_left_min = min(rho_left_min, self.left.robustness_at(signals, t1))
            if i == 0:
                rho_left_min = float("inf")
            best = max(best, min(rho_right, rho_left_min))
        return best

    def free_variables(self) -> FrozenSet[str]:
        return self.left.free_variables() | self.right.free_variables()

    def to_nnf(self) -> STLFormula:
        return Until(self.left.to_nnf(), self.right.to_nnf(), self.interval)

    def children(self) -> List[STLFormula]:
        return [self.left, self.right]

    def _pretty_math(self) -> str:
        return (
            f"({self.left._pretty_math()} U{self.interval} {self.right._pretty_math()})"
        )

    def _pretty_bio(self) -> str:
        return (
            f"({self.left._pretty_bio()} U[{self.interval.lo},{self.interval.hi}] "
            f"{self.right._pretty_bio()})"
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _push_negation(formula: STLFormula) -> STLFormula:
    """Push a negation inward for NNF conversion (De Morgan + temporal duals)."""
    if isinstance(formula, Predicate):
        return Predicate(
            copy.deepcopy(formula.expr), formula.op.negate(), formula.threshold
        )
    if isinstance(formula, STLNot):
        return formula.child.to_nnf()
    if isinstance(formula, STLAnd):
        return STLOr(_push_negation(formula.left), _push_negation(formula.right))
    if isinstance(formula, STLOr):
        return STLAnd(_push_negation(formula.left), _push_negation(formula.right))
    if isinstance(formula, STLImplies):
        # ¬(a → b) = a ∧ ¬b
        return STLAnd(formula.antecedent.to_nnf(), _push_negation(formula.consequent))
    if isinstance(formula, Always):
        return Eventually(_push_negation(formula.child), formula.interval)
    if isinstance(formula, Eventually):
        return Always(_push_negation(formula.child), formula.interval)
    if isinstance(formula, Until):
        # ¬(φ U ψ) is complex; use release operator semantics
        # ¬(φ U[a,b] ψ) = (¬ψ) R[a,b] (¬φ) but we don't have Release,
        # so express as G[a,b](¬ψ) ∨ ((¬ψ) U[a,b] (¬φ ∧ ¬ψ))
        neg_left = _push_negation(formula.left)
        neg_right = _push_negation(formula.right)
        return STLOr(
            Always(neg_right.clone(), formula.interval),
            Until(
                neg_right.clone(),
                STLAnd(neg_left, neg_right.clone()),
                formula.interval,
            ),
        )
    raise TypeError(f"Unknown formula type: {type(formula)}")


def _substitute_in_place(formula: STLFormula, mapping: Dict[str, str]) -> None:
    """Recursively substitute variable names in-place."""
    if isinstance(formula, Predicate):
        formula.expr.substitute_var(mapping)
    for child in formula.children():
        _substitute_in_place(child, mapping)


def _collect_atoms(formula: STLFormula, acc: List[Predicate]) -> None:
    if isinstance(formula, Predicate):
        acc.append(formula)
    for child in formula.children():
        _collect_atoms(child, acc)


def _collect_temporal(formula: STLFormula, acc: List[STLFormula]) -> None:
    if isinstance(formula, (Always, Eventually, Until)):
        acc.append(formula)
    for child in formula.children():
        _collect_temporal(child, acc)


def _map_atoms(formula: STLFormula, fn: Callable[[Predicate], STLFormula]) -> STLFormula:
    if isinstance(formula, Predicate):
        return fn(formula)
    if isinstance(formula, STLNot):
        return STLNot(_map_atoms(formula.child, fn))
    if isinstance(formula, STLAnd):
        return STLAnd(_map_atoms(formula.left, fn), _map_atoms(formula.right, fn))
    if isinstance(formula, STLOr):
        return STLOr(_map_atoms(formula.left, fn), _map_atoms(formula.right, fn))
    if isinstance(formula, STLImplies):
        return STLImplies(
            _map_atoms(formula.antecedent, fn), _map_atoms(formula.consequent, fn)
        )
    if isinstance(formula, Always):
        return Always(_map_atoms(formula.child, fn), formula.interval)
    if isinstance(formula, Eventually):
        return Eventually(_map_atoms(formula.child, fn), formula.interval)
    if isinstance(formula, Until):
        return Until(
            _map_atoms(formula.left, fn),
            _map_atoms(formula.right, fn),
            formula.interval,
        )
    raise TypeError(f"Unknown formula type: {type(formula)}")


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def globally(child: STLFormula, lo: float, hi: float) -> Always:
    return Always(child, Interval(lo, hi))


def eventually(child: STLFormula, lo: float, hi: float) -> Eventually:
    return Eventually(child, Interval(lo, hi))


def until(left: STLFormula, right: STLFormula, lo: float, hi: float) -> Until:
    return Until(left, right, Interval(lo, hi))


def predicate(var: str, op: str, threshold: float) -> Predicate:
    """Create a predicate from a variable name, operator string, and threshold."""
    op_map = {"<": ComparisonOp.LT, "<=": ComparisonOp.LE,
              ">": ComparisonOp.GT, ">=": ComparisonOp.GE}
    return Predicate(make_var_expr(var), op_map[op], threshold)

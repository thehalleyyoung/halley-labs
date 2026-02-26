"""
CSL (Continuous Stochastic Logic) abstract syntax tree.

Supports atomic propositions, boolean operators, probability operators
with bounded/unbounded until, next, and steady-state queries.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class ComparisonOp(enum.Enum):
    """Comparison operator for probability thresholds."""
    GEQ = ">="
    GT = ">"
    LEQ = "<="
    LT = "<"


@dataclass(frozen=True)
class CSLFormula:
    """Base class for CSL formulae."""
    pass


@dataclass(frozen=True)
class TrueFormula(CSLFormula):
    """The constant true formula."""
    pass


@dataclass(frozen=True)
class AtomicProp(CSLFormula):
    """
    Atomic proposition: species_index compared to threshold.

    Examples: X_2 >= 5, X_0 < 10
    Supports axis-aligned predicates (single species) and
    linear predicates (weighted sum of species).
    """
    species_index: int
    threshold: int
    direction: str = "greater_equal"  # greater, greater_equal, less, less_equal, equal

    def __repr__(self) -> str:
        ops = {
            "greater": ">", "greater_equal": ">=",
            "less": "<", "less_equal": "<=", "equal": "=="
        }
        return f"X_{self.species_index} {ops.get(self.direction, '?')} {self.threshold}"


@dataclass(frozen=True)
class LinearPredicate(CSLFormula):
    """
    Linear predicate: weighted sum of species compared to threshold.

    Represents constraints like n_A + n_B <= 10 or 2*n_A - n_B >= 5.
    This extends beyond axis-aligned predicates (Theorem 2 extension).

    The TT mask has rank > 1 in general, bounded by the number of
    distinct coefficient values.
    """
    coefficients: tuple[tuple[int, float], ...]  # (species_index, weight) pairs
    threshold: float
    direction: str = "greater_equal"

    def __repr__(self) -> str:
        terms = []
        for idx, w in self.coefficients:
            if w == 1.0:
                terms.append(f"X_{idx}")
            else:
                terms.append(f"{w}*X_{idx}")
        ops = {
            "greater": ">", "greater_equal": ">=",
            "less": "<", "less_equal": "<=",
        }
        return f"{' + '.join(terms)} {ops.get(self.direction, '?')} {self.threshold}"


@dataclass(frozen=True)
class Negation(CSLFormula):
    """Negation: ¬φ"""
    operand: CSLFormula

    def __repr__(self) -> str:
        return f"¬({self.operand})"


@dataclass(frozen=True)
class Conjunction(CSLFormula):
    """Conjunction: φ₁ ∧ φ₂"""
    left: CSLFormula
    right: CSLFormula

    def __repr__(self) -> str:
        return f"({self.left}) ∧ ({self.right})"


@dataclass(frozen=True)
class BoundedUntil(CSLFormula):
    """
    Time-bounded until path formula: φ₁ U[0,t] φ₂

    Semantics: φ₁ holds continuously until φ₂ becomes true,
    within time bound t.
    """
    phi1: CSLFormula
    phi2: CSLFormula
    time_bound: float

    def __repr__(self) -> str:
        return f"({self.phi1}) U[0,{self.time_bound}] ({self.phi2})"


@dataclass(frozen=True)
class UnboundedUntil(CSLFormula):
    """
    Unbounded until path formula: φ₁ U φ₂

    Semantics: φ₁ holds continuously until φ₂ eventually becomes true.
    """
    phi1: CSLFormula
    phi2: CSLFormula

    def __repr__(self) -> str:
        return f"({self.phi1}) U ({self.phi2})"


@dataclass(frozen=True)
class Next(CSLFormula):
    """Next operator: X φ (used in DTMC, rarely in CTMC)."""
    operand: CSLFormula


@dataclass(frozen=True)
class ProbabilityOp(CSLFormula):
    """
    Probability operator: P~p [ψ] where ~ ∈ {≥, >, ≤, <}.

    Evaluates whether the probability of the path formula ψ satisfies
    the comparison with threshold p.

    Three-valued semantics: returns {true, false, indeterminate}
    when truncation error places the probability within epsilon of
    the threshold.
    """
    comparison: ComparisonOp
    threshold: float
    path_formula: CSLFormula

    def __repr__(self) -> str:
        return f"P{self.comparison.value}{self.threshold} [{self.path_formula}]"


@dataclass(frozen=True)
class SteadyStateOp(CSLFormula):
    """
    Steady-state operator: S~p [φ]

    Evaluates whether the steady-state probability of φ satisfies
    the comparison with threshold p.
    """
    comparison: ComparisonOp
    threshold: float
    state_formula: CSLFormula

    def __repr__(self) -> str:
        return f"S{self.comparison.value}{self.threshold} [{self.state_formula}]"


def parse_csl(formula_str: str) -> CSLFormula:
    """
    Parse a CSL formula from string representation.

    Supported syntax:
        X_i >= n          atomic proposition
        !phi              negation
        phi1 & phi2       conjunction
        P>=p [phi1 U<=t phi2]   bounded until probability
        P>=p [phi1 U phi2]      unbounded until probability
        S>=p [phi]              steady-state probability

    Args:
        formula_str: String representation of CSL formula.

    Returns:
        Parsed CSL formula AST.
    """
    import re
    s = formula_str.strip()

    # Atomic proposition: X_i >= n
    m = re.match(r"X_(\d+)\s*(>=|>|<=|<|==)\s*(\d+)$", s)
    if m:
        idx, op_str, val = int(m.group(1)), m.group(2), int(m.group(3))
        direction_map = {
            ">=": "greater_equal", ">": "greater",
            "<=": "less_equal", "<": "less", "==": "equal"
        }
        return AtomicProp(idx, val, direction_map[op_str])

    # True constant
    if s.lower() == "true":
        return TrueFormula()

    # Probability operator: P>=0.9 [...]
    m = re.match(r"P\s*(>=|>|<=|<)\s*([\d.]+)\s*\[(.+)\]$", s)
    if m:
        comp_map = {">=": ComparisonOp.GEQ, ">": ComparisonOp.GT,
                     "<=": ComparisonOp.LEQ, "<": ComparisonOp.LT}
        comp = comp_map[m.group(1)]
        threshold = float(m.group(2))
        inner = m.group(3).strip()

        # Check for bounded until: phi1 U<=t phi2
        u_match = re.match(r"(.+?)\s+U<=(\S+)\s+(.+)$", inner)
        if u_match:
            phi1 = parse_csl(u_match.group(1))
            t = float(u_match.group(2))
            phi2 = parse_csl(u_match.group(3))
            return ProbabilityOp(comp, threshold, BoundedUntil(phi1, phi2, t))

        # Check for unbounded until: phi1 U phi2
        u_match = re.match(r"(.+?)\s+U\s+(.+)$", inner)
        if u_match:
            phi1 = parse_csl(u_match.group(1))
            phi2 = parse_csl(u_match.group(2))
            return ProbabilityOp(comp, threshold, UnboundedUntil(phi1, phi2))

        # Otherwise, treat inner as eventually: F<=t phi = true U<=t phi
        f_match = re.match(r"F<=(\S+)\s+(.+)$", inner)
        if f_match:
            t = float(f_match.group(1))
            phi = parse_csl(f_match.group(2))
            return ProbabilityOp(comp, threshold,
                                  BoundedUntil(TrueFormula(), phi, t))

        return ProbabilityOp(comp, threshold, parse_csl(inner))

    # Steady-state: S>=0.5 [...]
    m = re.match(r"S\s*(>=|>|<=|<)\s*([\d.]+)\s*\[(.+)\]$", s)
    if m:
        comp_map = {">=": ComparisonOp.GEQ, ">": ComparisonOp.GT,
                     "<=": ComparisonOp.LEQ, "<": ComparisonOp.LT}
        comp = comp_map[m.group(1)]
        threshold = float(m.group(2))
        return SteadyStateOp(comp, threshold, parse_csl(m.group(3)))

    # Negation: !phi
    if s.startswith("!"):
        return Negation(parse_csl(s[1:]))

    # Conjunction: phi1 & phi2 (simple split)
    if " & " in s:
        parts = s.split(" & ", 1)
        return Conjunction(parse_csl(parts[0]), parse_csl(parts[1]))

    raise ValueError(f"Cannot parse CSL formula: '{formula_str}'")

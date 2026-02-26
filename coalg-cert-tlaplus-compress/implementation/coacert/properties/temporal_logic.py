"""
Temporal logic formula handling for CoaCert-TLA.

Provides a formula AST for CTL* (subsuming CTL and LTL), parsing from
string representation, normalization (NNF push-in), simplification,
stuttering-invariance detection, CTL*\\X fragment detection, pretty
printing, and substitution.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Formula AST
# ============================================================================

class FormulaKind(Enum):
    """Discriminator tag for temporal formulas."""
    # State formulas
    ATOMIC = auto()
    TRUE = auto()
    FALSE = auto()
    NOT = auto()
    AND = auto()
    OR = auto()
    IMPLIES = auto()
    IFF = auto()
    EXISTS_PATH = auto()   # E ψ
    FORALL_PATH = auto()   # A ψ

    # Path formulas
    STATE_IN_PATH = auto()  # lift state formula into path context
    PATH_NOT = auto()
    PATH_AND = auto()
    PATH_OR = auto()
    NEXT = auto()           # X ψ
    UNTIL = auto()          # ψ U χ
    RELEASE = auto()        # ψ R χ
    FINALLY = auto()        # F ψ
    GLOBALLY = auto()       # G ψ
    WEAK_UNTIL = auto()     # ψ W χ


@dataclass(frozen=True)
class TemporalFormula(ABC):
    """Base class for all temporal formulas."""

    @abstractmethod
    def is_state_formula(self) -> bool:
        """True if this formula is a state formula."""
        ...

    @abstractmethod
    def negate(self) -> "TemporalFormula":
        """Return the logical negation of this formula."""
        ...

    @abstractmethod
    def children(self) -> List["TemporalFormula"]:
        """Return immediate sub-formulas."""
        ...

    @abstractmethod
    def pretty(self, precedence: int = 0) -> str:
        """Pretty-print the formula."""
        ...

    @abstractmethod
    def substitute(self, mapping: Dict[str, "TemporalFormula"]) -> "TemporalFormula":
        """Replace atomic propositions according to *mapping*."""
        ...

    @abstractmethod
    def accept(self, visitor: "FormulaVisitor") -> object:
        ...

    def __str__(self) -> str:
        return self.pretty()


# ---------------------------------------------------------------------------
# State formulas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Atomic(TemporalFormula):
    """Atomic proposition (e.g. ``p``, ``x > 0``)."""
    name: str = ""

    def is_state_formula(self) -> bool:
        return True

    def negate(self) -> TemporalFormula:
        return Not(self)

    def children(self) -> List[TemporalFormula]:
        return []

    def pretty(self, precedence: int = 0) -> str:
        return self.name

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return mapping.get(self.name, self)

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_atomic(self)


@dataclass(frozen=True)
class TrueFormula(TemporalFormula):
    """Constant TRUE."""

    def is_state_formula(self) -> bool:
        return True

    def negate(self) -> TemporalFormula:
        return FalseFormula()

    def children(self) -> List[TemporalFormula]:
        return []

    def pretty(self, precedence: int = 0) -> str:
        return "TRUE"

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return self

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_true(self)


@dataclass(frozen=True)
class FalseFormula(TemporalFormula):
    """Constant FALSE."""

    def is_state_formula(self) -> bool:
        return True

    def negate(self) -> TemporalFormula:
        return TrueFormula()

    def children(self) -> List[TemporalFormula]:
        return []

    def pretty(self, precedence: int = 0) -> str:
        return "FALSE"

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return self

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_false(self)


@dataclass(frozen=True)
class Not(TemporalFormula):
    """Negation of a state formula."""
    child: TemporalFormula = field(default_factory=TrueFormula)

    def is_state_formula(self) -> bool:
        return self.child.is_state_formula()

    def negate(self) -> TemporalFormula:
        return self.child

    def children(self) -> List[TemporalFormula]:
        return [self.child]

    def pretty(self, precedence: int = 0) -> str:
        inner = self.child.pretty(5)
        result = f"¬{inner}"
        return f"({result})" if precedence > 5 else result

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return Not(self.child.substitute(mapping))

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_not(self)


@dataclass(frozen=True)
class And(TemporalFormula):
    """Conjunction of state formulas."""
    left: TemporalFormula = field(default_factory=TrueFormula)
    right: TemporalFormula = field(default_factory=TrueFormula)

    def is_state_formula(self) -> bool:
        return self.left.is_state_formula() and self.right.is_state_formula()

    def negate(self) -> TemporalFormula:
        return Or(self.left.negate(), self.right.negate())

    def children(self) -> List[TemporalFormula]:
        return [self.left, self.right]

    def pretty(self, precedence: int = 0) -> str:
        result = f"{self.left.pretty(3)} ∧ {self.right.pretty(3)}"
        return f"({result})" if precedence > 3 else result

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return And(self.left.substitute(mapping), self.right.substitute(mapping))

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_and(self)


@dataclass(frozen=True)
class Or(TemporalFormula):
    """Disjunction of state formulas."""
    left: TemporalFormula = field(default_factory=TrueFormula)
    right: TemporalFormula = field(default_factory=TrueFormula)

    def is_state_formula(self) -> bool:
        return self.left.is_state_formula() and self.right.is_state_formula()

    def negate(self) -> TemporalFormula:
        return And(self.left.negate(), self.right.negate())

    def children(self) -> List[TemporalFormula]:
        return [self.left, self.right]

    def pretty(self, precedence: int = 0) -> str:
        result = f"{self.left.pretty(2)} ∨ {self.right.pretty(2)}"
        return f"({result})" if precedence > 2 else result

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return Or(self.left.substitute(mapping), self.right.substitute(mapping))

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_or(self)


@dataclass(frozen=True)
class Implies(TemporalFormula):
    """Implication φ → ψ."""
    left: TemporalFormula = field(default_factory=TrueFormula)
    right: TemporalFormula = field(default_factory=TrueFormula)

    def is_state_formula(self) -> bool:
        return self.left.is_state_formula() and self.right.is_state_formula()

    def negate(self) -> TemporalFormula:
        return And(self.left, self.right.negate())

    def children(self) -> List[TemporalFormula]:
        return [self.left, self.right]

    def pretty(self, precedence: int = 0) -> str:
        result = f"{self.left.pretty(1)} → {self.right.pretty(1)}"
        return f"({result})" if precedence > 1 else result

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return Implies(self.left.substitute(mapping), self.right.substitute(mapping))

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_implies(self)


@dataclass(frozen=True)
class Iff(TemporalFormula):
    """Bi-implication φ ↔ ψ."""
    left: TemporalFormula = field(default_factory=TrueFormula)
    right: TemporalFormula = field(default_factory=TrueFormula)

    def is_state_formula(self) -> bool:
        return self.left.is_state_formula() and self.right.is_state_formula()

    def negate(self) -> TemporalFormula:
        return Or(And(self.left, self.right.negate()),
                  And(self.left.negate(), self.right))

    def children(self) -> List[TemporalFormula]:
        return [self.left, self.right]

    def pretty(self, precedence: int = 0) -> str:
        result = f"{self.left.pretty(1)} ↔ {self.right.pretty(1)}"
        return f"({result})" if precedence > 1 else result

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return Iff(self.left.substitute(mapping), self.right.substitute(mapping))

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_iff(self)


@dataclass(frozen=True)
class ExistsPath(TemporalFormula):
    """Existential path quantifier E ψ."""
    path_formula: TemporalFormula = field(default_factory=TrueFormula)

    def is_state_formula(self) -> bool:
        return True

    def negate(self) -> TemporalFormula:
        return ForallPath(self.path_formula.negate())

    def children(self) -> List[TemporalFormula]:
        return [self.path_formula]

    def pretty(self, precedence: int = 0) -> str:
        return f"E({self.path_formula.pretty()})"

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return ExistsPath(self.path_formula.substitute(mapping))

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_exists_path(self)


@dataclass(frozen=True)
class ForallPath(TemporalFormula):
    """Universal path quantifier A ψ."""
    path_formula: TemporalFormula = field(default_factory=TrueFormula)

    def is_state_formula(self) -> bool:
        return True

    def negate(self) -> TemporalFormula:
        return ExistsPath(self.path_formula.negate())

    def children(self) -> List[TemporalFormula]:
        return [self.path_formula]

    def pretty(self, precedence: int = 0) -> str:
        return f"A({self.path_formula.pretty()})"

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return ForallPath(self.path_formula.substitute(mapping))

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_forall_path(self)


# ---------------------------------------------------------------------------
# Path formulas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Next(TemporalFormula):
    """Next-step operator X ψ."""
    child: TemporalFormula = field(default_factory=TrueFormula)

    def is_state_formula(self) -> bool:
        return False

    def negate(self) -> TemporalFormula:
        return Next(self.child.negate())

    def children(self) -> List[TemporalFormula]:
        return [self.child]

    def pretty(self, precedence: int = 0) -> str:
        return f"X {self.child.pretty(6)}"

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return Next(self.child.substitute(mapping))

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_next(self)


@dataclass(frozen=True)
class Until(TemporalFormula):
    """Until operator ψ U χ."""
    left: TemporalFormula = field(default_factory=TrueFormula)
    right: TemporalFormula = field(default_factory=TrueFormula)

    def is_state_formula(self) -> bool:
        return False

    def negate(self) -> TemporalFormula:
        return Release(self.left.negate(), self.right.negate())

    def children(self) -> List[TemporalFormula]:
        return [self.left, self.right]

    def pretty(self, precedence: int = 0) -> str:
        result = f"{self.left.pretty(4)} U {self.right.pretty(4)}"
        return f"({result})" if precedence > 4 else result

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return Until(self.left.substitute(mapping), self.right.substitute(mapping))

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_until(self)


@dataclass(frozen=True)
class Release(TemporalFormula):
    """Release operator ψ R χ (dual of Until)."""
    left: TemporalFormula = field(default_factory=FalseFormula)
    right: TemporalFormula = field(default_factory=TrueFormula)

    def is_state_formula(self) -> bool:
        return False

    def negate(self) -> TemporalFormula:
        return Until(self.left.negate(), self.right.negate())

    def children(self) -> List[TemporalFormula]:
        return [self.left, self.right]

    def pretty(self, precedence: int = 0) -> str:
        result = f"{self.left.pretty(4)} R {self.right.pretty(4)}"
        return f"({result})" if precedence > 4 else result

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return Release(self.left.substitute(mapping), self.right.substitute(mapping))

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_release(self)


@dataclass(frozen=True)
class Finally(TemporalFormula):
    """Eventually (finally) operator F ψ ≡ TRUE U ψ."""
    child: TemporalFormula = field(default_factory=TrueFormula)

    def is_state_formula(self) -> bool:
        return False

    def negate(self) -> TemporalFormula:
        return Globally(self.child.negate())

    def children(self) -> List[TemporalFormula]:
        return [self.child]

    def pretty(self, precedence: int = 0) -> str:
        return f"F {self.child.pretty(6)}"

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return Finally(self.child.substitute(mapping))

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_finally(self)


@dataclass(frozen=True)
class Globally(TemporalFormula):
    """Globally operator G ψ ≡ FALSE R ψ."""
    child: TemporalFormula = field(default_factory=TrueFormula)

    def is_state_formula(self) -> bool:
        return False

    def negate(self) -> TemporalFormula:
        return Finally(self.child.negate())

    def children(self) -> List[TemporalFormula]:
        return [self.child]

    def pretty(self, precedence: int = 0) -> str:
        return f"G {self.child.pretty(6)}"

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return Globally(self.child.substitute(mapping))

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_globally(self)


@dataclass(frozen=True)
class WeakUntil(TemporalFormula):
    """Weak until ψ W χ ≡ (ψ U χ) ∨ G ψ."""
    left: TemporalFormula = field(default_factory=TrueFormula)
    right: TemporalFormula = field(default_factory=TrueFormula)

    def is_state_formula(self) -> bool:
        return False

    def negate(self) -> TemporalFormula:
        return Until(self.right.negate(), And(self.left.negate(), self.right.negate()))

    def children(self) -> List[TemporalFormula]:
        return [self.left, self.right]

    def pretty(self, precedence: int = 0) -> str:
        result = f"{self.left.pretty(4)} W {self.right.pretty(4)}"
        return f"({result})" if precedence > 4 else result

    def substitute(self, mapping: Dict[str, TemporalFormula]) -> TemporalFormula:
        return WeakUntil(self.left.substitute(mapping), self.right.substitute(mapping))

    def accept(self, visitor: FormulaVisitor) -> object:
        return visitor.visit_weak_until(self)


# ============================================================================
# Visitor
# ============================================================================

class FormulaVisitor(ABC):
    """Base visitor for temporal formulas."""

    def generic_visit(self, formula: TemporalFormula) -> object:
        raise NotImplementedError(f"No visitor for {type(formula).__name__}")

    def visit_atomic(self, f: Atomic) -> object:
        return self.generic_visit(f)

    def visit_true(self, f: TrueFormula) -> object:
        return self.generic_visit(f)

    def visit_false(self, f: FalseFormula) -> object:
        return self.generic_visit(f)

    def visit_not(self, f: Not) -> object:
        return self.generic_visit(f)

    def visit_and(self, f: And) -> object:
        return self.generic_visit(f)

    def visit_or(self, f: Or) -> object:
        return self.generic_visit(f)

    def visit_implies(self, f: Implies) -> object:
        return self.generic_visit(f)

    def visit_iff(self, f: Iff) -> object:
        return self.generic_visit(f)

    def visit_exists_path(self, f: ExistsPath) -> object:
        return self.generic_visit(f)

    def visit_forall_path(self, f: ForallPath) -> object:
        return self.generic_visit(f)

    def visit_next(self, f: Next) -> object:
        return self.generic_visit(f)

    def visit_until(self, f: Until) -> object:
        return self.generic_visit(f)

    def visit_release(self, f: Release) -> object:
        return self.generic_visit(f)

    def visit_finally(self, f: Finally) -> object:
        return self.generic_visit(f)

    def visit_globally(self, f: Globally) -> object:
        return self.generic_visit(f)

    def visit_weak_until(self, f: WeakUntil) -> object:
        return self.generic_visit(f)


# ============================================================================
# Formula normalization – push negation inward (NNF)
# ============================================================================

def to_nnf(formula: TemporalFormula) -> TemporalFormula:
    """Convert *formula* to Negation Normal Form.

    Negations are pushed inward so that they only appear directly
    above atomic propositions.  Implications and bi-implications are
    eliminated.
    """
    if isinstance(formula, (Atomic, TrueFormula, FalseFormula)):
        return formula

    if isinstance(formula, Not):
        inner = formula.child
        # ¬¬φ → φ
        if isinstance(inner, Not):
            return to_nnf(inner.child)
        # ¬TRUE → FALSE
        if isinstance(inner, TrueFormula):
            return FalseFormula()
        if isinstance(inner, FalseFormula):
            return TrueFormula()
        # ¬(φ ∧ ψ) → ¬φ ∨ ¬ψ
        if isinstance(inner, And):
            return to_nnf(Or(Not(inner.left), Not(inner.right)))
        if isinstance(inner, Or):
            return to_nnf(And(Not(inner.left), Not(inner.right)))
        if isinstance(inner, Implies):
            return to_nnf(And(inner.left, Not(inner.right)))
        if isinstance(inner, Iff):
            return to_nnf(inner.negate())
        # Path quantifier duality
        if isinstance(inner, ExistsPath):
            return ForallPath(to_nnf(Not(inner.path_formula)))
        if isinstance(inner, ForallPath):
            return ExistsPath(to_nnf(Not(inner.path_formula)))
        # Temporal operator duality
        if isinstance(inner, Next):
            return Next(to_nnf(Not(inner.child)))
        if isinstance(inner, Finally):
            return Globally(to_nnf(Not(inner.child)))
        if isinstance(inner, Globally):
            return Finally(to_nnf(Not(inner.child)))
        if isinstance(inner, Until):
            return Release(to_nnf(Not(inner.left)), to_nnf(Not(inner.right)))
        if isinstance(inner, Release):
            return Until(to_nnf(Not(inner.left)), to_nnf(Not(inner.right)))
        if isinstance(inner, WeakUntil):
            return to_nnf(Until(Not(inner.right), And(Not(inner.left), Not(inner.right))))
        if isinstance(inner, Atomic):
            return Not(inner)
        return Not(to_nnf(inner))

    # Eliminate implication and bi-implication
    if isinstance(formula, Implies):
        return to_nnf(Or(Not(formula.left), formula.right))
    if isinstance(formula, Iff):
        return to_nnf(And(Implies(formula.left, formula.right),
                          Implies(formula.right, formula.left)))

    # Recurse on binary / unary constructors
    if isinstance(formula, And):
        return And(to_nnf(formula.left), to_nnf(formula.right))
    if isinstance(formula, Or):
        return Or(to_nnf(formula.left), to_nnf(formula.right))
    if isinstance(formula, ExistsPath):
        return ExistsPath(to_nnf(formula.path_formula))
    if isinstance(formula, ForallPath):
        return ForallPath(to_nnf(formula.path_formula))
    if isinstance(formula, Next):
        return Next(to_nnf(formula.child))
    if isinstance(formula, Finally):
        return Finally(to_nnf(formula.child))
    if isinstance(formula, Globally):
        return Globally(to_nnf(formula.child))
    if isinstance(formula, Until):
        return Until(to_nnf(formula.left), to_nnf(formula.right))
    if isinstance(formula, Release):
        return Release(to_nnf(formula.left), to_nnf(formula.right))
    if isinstance(formula, WeakUntil):
        return WeakUntil(to_nnf(formula.left), to_nnf(formula.right))

    return formula


# ============================================================================
# Formula simplification
# ============================================================================

def simplify(formula: TemporalFormula) -> TemporalFormula:
    """Apply algebraic simplifications (idempotence, absorption, identity)."""
    if isinstance(formula, Not):
        c = simplify(formula.child)
        if isinstance(c, TrueFormula):
            return FalseFormula()
        if isinstance(c, FalseFormula):
            return TrueFormula()
        if isinstance(c, Not):
            return c.child
        return Not(c)

    if isinstance(formula, And):
        l = simplify(formula.left)
        r = simplify(formula.right)
        if isinstance(l, FalseFormula) or isinstance(r, FalseFormula):
            return FalseFormula()
        if isinstance(l, TrueFormula):
            return r
        if isinstance(r, TrueFormula):
            return l
        if l == r:
            return l
        return And(l, r)

    if isinstance(formula, Or):
        l = simplify(formula.left)
        r = simplify(formula.right)
        if isinstance(l, TrueFormula) or isinstance(r, TrueFormula):
            return TrueFormula()
        if isinstance(l, FalseFormula):
            return r
        if isinstance(r, FalseFormula):
            return l
        if l == r:
            return l
        return Or(l, r)

    if isinstance(formula, Implies):
        l = simplify(formula.left)
        r = simplify(formula.right)
        if isinstance(l, FalseFormula) or isinstance(r, TrueFormula):
            return TrueFormula()
        if isinstance(l, TrueFormula):
            return r
        return Implies(l, r)

    if isinstance(formula, Finally):
        c = simplify(formula.child)
        if isinstance(c, TrueFormula):
            return TrueFormula()
        if isinstance(c, FalseFormula):
            return FalseFormula()
        if isinstance(c, Finally):
            return c  # FF = F
        return Finally(c)

    if isinstance(formula, Globally):
        c = simplify(formula.child)
        if isinstance(c, TrueFormula):
            return TrueFormula()
        if isinstance(c, FalseFormula):
            return FalseFormula()
        if isinstance(c, Globally):
            return c  # GG = G
        return Globally(c)

    if isinstance(formula, Next):
        return Next(simplify(formula.child))
    if isinstance(formula, Until):
        return Until(simplify(formula.left), simplify(formula.right))
    if isinstance(formula, Release):
        return Release(simplify(formula.left), simplify(formula.right))
    if isinstance(formula, WeakUntil):
        return WeakUntil(simplify(formula.left), simplify(formula.right))
    if isinstance(formula, ExistsPath):
        return ExistsPath(simplify(formula.path_formula))
    if isinstance(formula, ForallPath):
        return ForallPath(simplify(formula.path_formula))

    return formula


# ============================================================================
# Stuttering-invariance detection
# ============================================================================

class _StutterInvarianceChecker(FormulaVisitor):
    """A formula is stuttering-invariant iff it belongs to CTL*\\X.

    That is, it contains no Next (X) operator.  This is the syntactic
    approximation; the semantic characterisation is that the truth
    value is preserved by stuttering equivalence.
    """

    def generic_visit(self, f: TemporalFormula) -> object:
        return True

    def visit_next(self, f: Next) -> object:
        return False

    def visit_not(self, f: Not) -> object:
        return f.child.accept(self)

    def visit_and(self, f: And) -> object:
        return f.left.accept(self) and f.right.accept(self)

    def visit_or(self, f: Or) -> object:
        return f.left.accept(self) and f.right.accept(self)

    def visit_implies(self, f: Implies) -> object:
        return f.left.accept(self) and f.right.accept(self)

    def visit_iff(self, f: Iff) -> object:
        return f.left.accept(self) and f.right.accept(self)

    def visit_exists_path(self, f: ExistsPath) -> object:
        return f.path_formula.accept(self)

    def visit_forall_path(self, f: ForallPath) -> object:
        return f.path_formula.accept(self)

    def visit_until(self, f: Until) -> object:
        return f.left.accept(self) and f.right.accept(self)

    def visit_release(self, f: Release) -> object:
        return f.left.accept(self) and f.right.accept(self)

    def visit_finally(self, f: Finally) -> object:
        return f.child.accept(self)

    def visit_globally(self, f: Globally) -> object:
        return f.child.accept(self)

    def visit_weak_until(self, f: WeakUntil) -> object:
        return f.left.accept(self) and f.right.accept(self)


def is_stuttering_invariant(formula: TemporalFormula) -> bool:
    """Return True iff *formula* is syntactically stuttering-invariant (CTL*\\X)."""
    checker = _StutterInvarianceChecker()
    return bool(formula.accept(checker))


def is_ctl_star_without_next(formula: TemporalFormula) -> bool:
    """Alias: detect whether formula belongs to the CTL*\\X fragment."""
    return is_stuttering_invariant(formula)


# ============================================================================
# CTL fragment detection
# ============================================================================

def _is_ctl_path(formula: TemporalFormula) -> bool:
    """Check if a path formula is a CTL path formula.

    In CTL, path formulas have exactly one of the patterns:
        X φ, F φ, G φ, φ U ψ, φ R ψ, φ W ψ
    where φ, ψ are *state* formulas.
    """
    if isinstance(formula, Next):
        return formula.child.is_state_formula()
    if isinstance(formula, (Finally, Globally)):
        return formula.child.is_state_formula()
    if isinstance(formula, (Until, Release, WeakUntil)):
        return formula.left.is_state_formula() and formula.right.is_state_formula()
    return False


def is_ctl(formula: TemporalFormula) -> bool:
    """Return True iff *formula* is in the CTL fragment of CTL*."""
    if isinstance(formula, (Atomic, TrueFormula, FalseFormula)):
        return True
    if isinstance(formula, Not):
        return is_ctl(formula.child)
    if isinstance(formula, (And, Or, Implies, Iff)):
        return all(is_ctl(c) for c in formula.children())
    if isinstance(formula, (ExistsPath, ForallPath)):
        inner = formula.path_formula
        if not _is_ctl_path(inner):
            return False
        return all(is_ctl(c) for c in inner.children())
    return False


def is_ltl(formula: TemporalFormula) -> bool:
    """Return True iff *formula* is in the LTL fragment.

    LTL formulas use no path quantifiers; they are interpreted as
    implicitly universally quantified.
    """
    if isinstance(formula, (Atomic, TrueFormula, FalseFormula)):
        return True
    if isinstance(formula, Not):
        return is_ltl(formula.child)
    if isinstance(formula, (And, Or, Implies, Iff)):
        return all(is_ltl(c) for c in formula.children())
    if isinstance(formula, (ExistsPath, ForallPath)):
        return False
    if isinstance(formula, (Next, Finally, Globally)):
        return is_ltl(formula.child)
    if isinstance(formula, (Until, Release, WeakUntil)):
        return is_ltl(formula.left) and is_ltl(formula.right)
    return False


# ============================================================================
# Atomic proposition collection
# ============================================================================

def atomic_propositions(formula: TemporalFormula) -> FrozenSet[str]:
    """Collect all atomic proposition names occurring in *formula*."""
    aps: Set[str] = set()

    def _collect(f: TemporalFormula) -> None:
        if isinstance(f, Atomic):
            aps.add(f.name)
        for child in f.children():
            _collect(child)

    _collect(formula)
    return frozenset(aps)


# ============================================================================
# Formula parsing (recursive descent)
# ============================================================================

class FormulaParseError(Exception):
    """Raised when a formula string cannot be parsed."""
    pass


class _FormulaParser:
    """Recursive-descent parser for temporal logic formulas.

    Grammar (precedence low→high)::

        formula     ::= iff_expr
        iff_expr    ::= impl_expr ( '<->' impl_expr )*
        impl_expr   ::= or_expr ( '->' or_expr )*
        or_expr     ::= and_expr ( ('|' | '\\/') and_expr )*
        and_expr    ::= unary_expr ( ('&' | '/\\') unary_expr )*
        unary_expr  ::= '~' unary_expr
                       | 'E' '(' formula ')'
                       | 'A' '(' formula ')'
                       | 'X' unary_expr
                       | 'F' unary_expr
                       | 'G' unary_expr
                       | primary ( 'U' unary_expr | 'R' unary_expr | 'W' unary_expr )?
        primary     ::= 'TRUE' | 'FALSE' | IDENT | '(' formula ')'
    """

    def __init__(self, text: str) -> None:
        self._tokens = self._tokenize(text)
        self._pos = 0

    # -- Tokenizer ----------------------------------------------------------

    _TOKEN_RE = re.compile(
        r"\s*("
        r"<->|->|/\\|\\/|~|"
        r"[()&|]|"
        r"[A-Za-z_][A-Za-z0-9_]*"
        r")"
    )

    def _tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        pos = 0
        while pos < len(text):
            m = self._TOKEN_RE.match(text, pos)
            if m is None:
                if text[pos:].strip() == "":
                    break
                raise FormulaParseError(f"Unexpected character at position {pos}: {text[pos:]!r}")
            tok = m.group(1)
            tokens.append(tok)
            pos = m.end()
        return tokens

    def _peek(self) -> Optional[str]:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _advance(self) -> str:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, tok: str) -> None:
        if self._peek() != tok:
            raise FormulaParseError(f"Expected {tok!r}, got {self._peek()!r}")
        self._advance()

    # -- Grammar rules ------------------------------------------------------

    def parse(self) -> TemporalFormula:
        f = self._parse_iff()
        if self._pos < len(self._tokens):
            raise FormulaParseError(f"Unexpected token: {self._peek()!r}")
        return f

    def _parse_iff(self) -> TemporalFormula:
        left = self._parse_impl()
        while self._peek() == "<->":
            self._advance()
            right = self._parse_impl()
            left = Iff(left, right)
        return left

    def _parse_impl(self) -> TemporalFormula:
        left = self._parse_or()
        while self._peek() == "->":
            self._advance()
            right = self._parse_or()
            left = Implies(left, right)
        return left

    def _parse_or(self) -> TemporalFormula:
        left = self._parse_and()
        while self._peek() in ("|", "\\/"):
            self._advance()
            right = self._parse_and()
            left = Or(left, right)
        return left

    def _parse_and(self) -> TemporalFormula:
        left = self._parse_unary()
        while self._peek() in ("&", "/\\"):
            self._advance()
            right = self._parse_unary()
            left = And(left, right)
        return left

    def _parse_unary(self) -> TemporalFormula:
        tok = self._peek()
        if tok == "~":
            self._advance()
            return Not(self._parse_unary())
        if tok == "E":
            self._advance()
            self._expect("(")
            inner = self._parse_iff()
            self._expect(")")
            return ExistsPath(inner)
        if tok == "A":
            self._advance()
            if self._peek() == "(":
                self._advance()
                inner = self._parse_iff()
                self._expect(")")
                return ForallPath(inner)
            # Bare A without parens: treat next token as path formula
            inner = self._parse_unary()
            return ForallPath(inner)
        if tok == "X":
            self._advance()
            return Next(self._parse_unary())
        if tok == "F":
            self._advance()
            return Finally(self._parse_unary())
        if tok == "G":
            self._advance()
            return Globally(self._parse_unary())

        primary = self._parse_primary()

        # Check for binary temporal operator
        tok2 = self._peek()
        if tok2 == "U":
            self._advance()
            right = self._parse_unary()
            return Until(primary, right)
        if tok2 == "R":
            self._advance()
            right = self._parse_unary()
            return Release(primary, right)
        if tok2 == "W":
            self._advance()
            right = self._parse_unary()
            return WeakUntil(primary, right)

        return primary

    def _parse_primary(self) -> TemporalFormula:
        tok = self._peek()
        if tok is None:
            raise FormulaParseError("Unexpected end of input")
        if tok == "(":
            self._advance()
            inner = self._parse_iff()
            self._expect(")")
            return inner
        if tok == "TRUE":
            self._advance()
            return TrueFormula()
        if tok == "FALSE":
            self._advance()
            return FalseFormula()
        if tok and re.match(r"[A-Za-z_]", tok):
            self._advance()
            return Atomic(tok)
        raise FormulaParseError(f"Unexpected token: {tok!r}")


def parse_formula(text: str) -> TemporalFormula:
    """Parse a temporal logic formula from a string.

    Operators (precedence low→high):
      ``<->``  ``->``  ``| \\/``  ``& /\\``  ``~ E A X F G``  ``U R W``

    Examples::

        parse_formula("A(G p -> F q)")
        parse_formula("E(p U q)")
        parse_formula("G (p -> F q)")
    """
    parser = _FormulaParser(text.strip())
    return parser.parse()


# ============================================================================
# CTL convenience constructors
# ============================================================================

def EX(phi: TemporalFormula) -> TemporalFormula:
    """CTL: EX φ."""
    return ExistsPath(Next(phi))


def EF(phi: TemporalFormula) -> TemporalFormula:
    """CTL: EF φ ≡ E[TRUE U φ]."""
    return ExistsPath(Finally(phi))


def EG(phi: TemporalFormula) -> TemporalFormula:
    """CTL: EG φ."""
    return ExistsPath(Globally(phi))


def EU(phi: TemporalFormula, psi: TemporalFormula) -> TemporalFormula:
    """CTL: E[φ U ψ]."""
    return ExistsPath(Until(phi, psi))


def AX(phi: TemporalFormula) -> TemporalFormula:
    """CTL: AX φ."""
    return ForallPath(Next(phi))


def AF(phi: TemporalFormula) -> TemporalFormula:
    """CTL: AF φ ≡ A[TRUE U φ]."""
    return ForallPath(Finally(phi))


def AG(phi: TemporalFormula) -> TemporalFormula:
    """CTL: AG φ."""
    return ForallPath(Globally(phi))


def AU(phi: TemporalFormula, psi: TemporalFormula) -> TemporalFormula:
    """CTL: A[φ U ψ]."""
    return ForallPath(Until(phi, psi))

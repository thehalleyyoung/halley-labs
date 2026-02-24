"""
Predicate Template Language P.

Implements the predicate language P from the refinement type theory:
  Sorts: Int, Bool, Tag, Str
  Function symbols: len(x), isinstance(x,T), is_none(x), is_truthy(x),
                    hasattr(x,k), arithmetic (+,-,*,//,%), comparisons (<,<=,=,!=,>=,>)
  Atomic predicates: e1 ⊲⊳ e2, isinstance(x,T), is_none(x), is_truthy(x), hasattr(x,k)
  Closure: Boolean combinations (∧, ∨, ¬)
"""

from __future__ import annotations

import itertools
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Sorts
# ---------------------------------------------------------------------------

class PredicateSort(Enum):
    """The four sorts of the predicate language P."""
    Int = auto()
    Bool = auto()
    Tag = auto()
    Str = auto()

    def __repr__(self) -> str:
        return f"PredicateSort.{self.name}"

    def smt_sort(self) -> str:
        """Return corresponding SMT-LIB sort name."""
        return {
            PredicateSort.Int: "Int",
            PredicateSort.Bool: "Bool",
            PredicateSort.Tag: "String",
            PredicateSort.Str: "String",
        }[self]


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------

class ComparisonOp(Enum):
    """Comparison operators for atomic comparison predicates."""
    Lt = auto()
    Le = auto()
    Eq = auto()
    Ne = auto()
    Ge = auto()
    Gt = auto()

    def negate(self) -> "ComparisonOp":
        _neg = {
            ComparisonOp.Lt: ComparisonOp.Ge,
            ComparisonOp.Le: ComparisonOp.Gt,
            ComparisonOp.Eq: ComparisonOp.Ne,
            ComparisonOp.Ne: ComparisonOp.Eq,
            ComparisonOp.Ge: ComparisonOp.Lt,
            ComparisonOp.Gt: ComparisonOp.Le,
        }
        return _neg[self]

    def flip(self) -> "ComparisonOp":
        """Flip the operator (swap operand sides)."""
        _flip = {
            ComparisonOp.Lt: ComparisonOp.Gt,
            ComparisonOp.Le: ComparisonOp.Ge,
            ComparisonOp.Eq: ComparisonOp.Eq,
            ComparisonOp.Ne: ComparisonOp.Ne,
            ComparisonOp.Ge: ComparisonOp.Le,
            ComparisonOp.Gt: ComparisonOp.Lt,
        }
        return _flip[self]

    def python_op(self) -> Callable[[Any, Any], bool]:
        return {
            ComparisonOp.Lt: operator.lt,
            ComparisonOp.Le: operator.le,
            ComparisonOp.Eq: operator.eq,
            ComparisonOp.Ne: operator.ne,
            ComparisonOp.Ge: operator.ge,
            ComparisonOp.Gt: operator.gt,
        }[self]

    def symbol(self) -> str:
        return {
            ComparisonOp.Lt: "<",
            ComparisonOp.Le: "<=",
            ComparisonOp.Eq: "==",
            ComparisonOp.Ne: "!=",
            ComparisonOp.Ge: ">=",
            ComparisonOp.Gt: ">",
        }[self]

    def smt_symbol(self) -> str:
        return {
            ComparisonOp.Lt: "<",
            ComparisonOp.Le: "<=",
            ComparisonOp.Eq: "=",
            ComparisonOp.Ne: "distinct",
            ComparisonOp.Ge: ">=",
            ComparisonOp.Gt: ">",
        }[self]

    def __repr__(self) -> str:
        return f"ComparisonOp.{self.name}"


# ---------------------------------------------------------------------------
# Terms / Linear expressions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Term:
    """A variable reference or constant in the predicate language."""
    name: str
    sort: PredicateSort = PredicateSort.Int

    def variables(self) -> FrozenSet[str]:
        return frozenset({self.name})

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        if self.name not in env:
            raise KeyError(f"Variable {self.name!r} not found in environment")
        return env[self.name]

    def substitute(self, mapping: Mapping[str, "LinearExpression"]) -> "LinearExpression":
        if self.name in mapping:
            return mapping[self.name]
        return LinearExpression(terms=((1, self),), constant=0)

    def pretty_print(self) -> str:
        return self.name

    def to_smt(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Term({self.name!r}, {self.sort})"


@dataclass(frozen=True)
class Constant:
    """A constant value."""
    value: Union[int, bool, str]
    sort: PredicateSort = PredicateSort.Int

    def variables(self) -> FrozenSet[str]:
        return frozenset()

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        return self.value

    def substitute(self, mapping: Mapping[str, "LinearExpression"]) -> "LinearExpression":
        return LinearExpression(terms=(), constant=self.value if isinstance(self.value, int) else 0)

    def pretty_print(self) -> str:
        if isinstance(self.value, str):
            return repr(self.value)
        return str(self.value)

    def to_smt(self) -> str:
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        if isinstance(self.value, int):
            if self.value < 0:
                return f"(- {abs(self.value)})"
            return str(self.value)
        return f'"{self.value}"'

    def __repr__(self) -> str:
        return f"Constant({self.value!r})"


@dataclass(frozen=True)
class LenTerm:
    """len(variable) term."""
    variable: str

    def variables(self) -> FrozenSet[str]:
        return frozenset({self.variable})

    def evaluate(self, env: Mapping[str, Any]) -> int:
        val = env.get(self.variable)
        if val is None:
            raise KeyError(f"Variable {self.variable!r} not found")
        return len(val)

    def substitute(self, mapping: Mapping[str, "LinearExpression"]) -> "LinearExpression":
        return LinearExpression(terms=((1, LenTerm(self.variable)),), constant=0)

    def pretty_print(self) -> str:
        return f"len({self.variable})"

    def to_smt(self) -> str:
        return f"(str.len {self.variable})"

    def __repr__(self) -> str:
        return f"LenTerm({self.variable!r})"


TermLike = Union[Term, Constant, LenTerm]


@dataclass(frozen=True)
class LinearExpression:
    """
    A linear expression: c_0 + c_1*t_1 + c_2*t_2 + ...
    where each t_i is a Term or LenTerm.
    Supports: e+e, e-e, e*c, e//c, e%c.
    """
    terms: Tuple[Tuple[int, TermLike], ...] = ()
    constant: int = 0

    @staticmethod
    def from_term(t: TermLike) -> "LinearExpression":
        if isinstance(t, Constant):
            return LinearExpression(terms=(), constant=t.value if isinstance(t.value, int) else 0)
        return LinearExpression(terms=((1, t),), constant=0)

    @staticmethod
    def from_int(n: int) -> "LinearExpression":
        return LinearExpression(terms=(), constant=n)

    @staticmethod
    def from_variable(name: str) -> "LinearExpression":
        return LinearExpression(terms=((1, Term(name)),), constant=0)

    @staticmethod
    def from_len(var: str) -> "LinearExpression":
        return LinearExpression(terms=((1, LenTerm(var)),), constant=0)

    def variables(self) -> FrozenSet[str]:
        result: Set[str] = set()
        for _, t in self.terms:
            result |= t.variables()
        return frozenset(result)

    def is_constant(self) -> bool:
        return len(self.terms) == 0

    def constant_value(self) -> Optional[int]:
        if self.is_constant():
            return self.constant
        return None

    def is_variable(self) -> Optional[str]:
        if len(self.terms) == 1 and self.constant == 0:
            coeff, t = self.terms[0]
            if coeff == 1 and isinstance(t, Term):
                return t.name
        return None

    def add(self, other: "LinearExpression") -> "LinearExpression":
        merged: Dict[TermLike, int] = {}
        for coeff, t in self.terms:
            merged[t] = merged.get(t, 0) + coeff
        for coeff, t in other.terms:
            merged[t] = merged.get(t, 0) + coeff
        new_terms = tuple((c, t) for t, c in sorted(merged.items(), key=lambda x: repr(x[0])) if c != 0)
        return LinearExpression(terms=new_terms, constant=self.constant + other.constant)

    def subtract(self, other: "LinearExpression") -> "LinearExpression":
        return self.add(other.scale(-1))

    def scale(self, factor: int) -> "LinearExpression":
        new_terms = tuple((c * factor, t) for c, t in self.terms if c * factor != 0)
        return LinearExpression(terms=new_terms, constant=self.constant * factor)

    def floor_div(self, divisor: int) -> "LinearExpression":
        if divisor == 0:
            raise ZeroDivisionError("Division by zero in linear expression")
        if all(c % divisor == 0 for c, _ in self.terms) and self.constant % divisor == 0:
            new_terms = tuple((c // divisor, t) for c, t in self.terms)
            return LinearExpression(terms=new_terms, constant=self.constant // divisor)
        raise ValueError(f"Cannot exactly floor-divide expression by {divisor}")

    def modulo(self, divisor: int) -> "LinearExpression":
        if divisor == 0:
            raise ZeroDivisionError("Modulo by zero in linear expression")
        if self.is_constant():
            return LinearExpression.from_int(self.constant % divisor)
        raise ValueError("Modulo of non-constant linear expressions not supported in closed form")

    def evaluate(self, env: Mapping[str, Any]) -> int:
        result = self.constant
        for coeff, t in self.terms:
            result += coeff * t.evaluate(env)
        return result

    def substitute(self, mapping: Mapping[str, "LinearExpression"]) -> "LinearExpression":
        result = LinearExpression.from_int(self.constant)
        for coeff, t in self.terms:
            sub = t.substitute(mapping)
            result = result.add(sub.scale(coeff))
        return result

    def simplify(self) -> "LinearExpression":
        merged: Dict[TermLike, int] = {}
        for coeff, t in self.terms:
            merged[t] = merged.get(t, 0) + coeff
        new_terms = tuple(
            (c, t) for t, c in sorted(merged.items(), key=lambda x: repr(x[0]))
            if c != 0
        )
        return LinearExpression(terms=new_terms, constant=self.constant)

    def pretty_print(self) -> str:
        parts: List[str] = []
        if self.constant != 0 or not self.terms:
            parts.append(str(self.constant))
        for coeff, t in self.terms:
            t_str = t.pretty_print()
            if coeff == 1:
                parts.append(t_str)
            elif coeff == -1:
                parts.append(f"-{t_str}")
            else:
                parts.append(f"{coeff}*{t_str}")
        if not parts:
            return "0"
        result = parts[0]
        for p in parts[1:]:
            if p.startswith("-"):
                result += f" - {p[1:]}"
            else:
                result += f" + {p}"
        return result

    def to_smt(self) -> str:
        if not self.terms:
            if self.constant < 0:
                return f"(- {abs(self.constant)})"
            return str(self.constant)
        parts: List[str] = []
        if self.constant != 0:
            parts.append(str(self.constant) if self.constant >= 0 else f"(- {abs(self.constant)})")
        for coeff, t in self.terms:
            t_smt = t.to_smt()
            if coeff == 1:
                parts.append(t_smt)
            elif coeff == -1:
                parts.append(f"(- {t_smt})")
            else:
                parts.append(f"(* {coeff} {t_smt})")
        if len(parts) == 1:
            return parts[0]
        result = parts[0]
        for p in parts[1:]:
            result = f"(+ {result} {p})"
        return result

    def __add__(self, other: "LinearExpression") -> "LinearExpression":
        return self.add(other)

    def __sub__(self, other: "LinearExpression") -> "LinearExpression":
        return self.subtract(other)

    def __mul__(self, factor: int) -> "LinearExpression":
        return self.scale(factor)

    def __rmul__(self, factor: int) -> "LinearExpression":
        return self.scale(factor)

    def __neg__(self) -> "LinearExpression":
        return self.scale(-1)

    def __repr__(self) -> str:
        return f"LinearExpression(terms={self.terms!r}, constant={self.constant})"


# ---------------------------------------------------------------------------
# PredicateTemplate ABC
# ---------------------------------------------------------------------------

class PredicateTemplate(ABC):
    """
    Abstract base for all predicate templates in language P.

    Every predicate supports: variables, substitute, evaluate, negate,
    simplify, to_smt, pretty_print, and is hashable/equatable.
    """

    @abstractmethod
    def variables(self) -> FrozenSet[str]:
        """Return the set of free variables in this predicate."""
        ...

    @abstractmethod
    def substitute(self, mapping: Mapping[str, LinearExpression]) -> "PredicateTemplate":
        """Substitute variables according to the given mapping."""
        ...

    @abstractmethod
    def evaluate(self, env: Mapping[str, Any]) -> bool:
        """Evaluate this predicate under a concrete environment."""
        ...

    @abstractmethod
    def negate(self) -> "PredicateTemplate":
        """Return the logical negation of this predicate."""
        ...

    @abstractmethod
    def simplify(self) -> "PredicateTemplate":
        """Return a simplified equivalent predicate."""
        ...

    @abstractmethod
    def to_smt(self) -> str:
        """Convert to SMT-LIB format string."""
        ...

    @abstractmethod
    def pretty_print(self) -> str:
        """Return a human-readable representation."""
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...

    def __repr__(self) -> str:
        return self.pretty_print()

    def __and__(self, other: "PredicateTemplate") -> "PredicateTemplate":
        return Conjunction.create(self, other)

    def __or__(self, other: "PredicateTemplate") -> "PredicateTemplate":
        return Disjunction.create(self, other)

    def __invert__(self) -> "PredicateTemplate":
        return self.negate()

    def implies(self, other: "PredicateTemplate") -> "PredicateTemplate":
        return Implication(self, other)

    def is_atomic(self) -> bool:
        return isinstance(self, AtomicPredicate)

    def is_true(self) -> bool:
        return isinstance(self, TruePredicate)

    def is_false(self) -> bool:
        return isinstance(self, FalsePredicate)

    def conjuncts(self) -> List["PredicateTemplate"]:
        """Flatten into a list of conjuncts (identity for non-Conjunction)."""
        return [self]

    def disjuncts(self) -> List["PredicateTemplate"]:
        """Flatten into a list of disjuncts (identity for non-Disjunction)."""
        return [self]

    def atoms(self) -> Set["AtomicPredicate"]:
        """Collect all atomic predicates appearing in this predicate."""
        result: Set[AtomicPredicate] = set()
        self._collect_atoms(result)
        return result

    def _collect_atoms(self, acc: Set["AtomicPredicate"]) -> None:
        if isinstance(self, AtomicPredicate):
            acc.add(self)

    def depth(self) -> int:
        """Nesting depth of the predicate tree."""
        return 1

    def size(self) -> int:
        """Number of nodes in the predicate tree."""
        return 1


# ---------------------------------------------------------------------------
# Atomic predicates
# ---------------------------------------------------------------------------

class AtomicPredicate(PredicateTemplate, ABC):
    """Base class for atomic predicates (non-compound)."""

    def _collect_atoms(self, acc: Set["AtomicPredicate"]) -> None:
        acc.add(self)


@dataclass(frozen=True)
class ComparisonPredicate(AtomicPredicate):
    """e₁ op e₂  where op ∈ {Lt, Le, Eq, Ne, Ge, Gt} and e₁, e₂ are linear expressions."""
    left: LinearExpression
    op: ComparisonOp
    right: LinearExpression

    def variables(self) -> FrozenSet[str]:
        return self.left.variables() | self.right.variables()

    def substitute(self, mapping: Mapping[str, LinearExpression]) -> "ComparisonPredicate":
        return ComparisonPredicate(
            left=self.left.substitute(mapping),
            op=self.op,
            right=self.right.substitute(mapping),
        )

    def evaluate(self, env: Mapping[str, Any]) -> bool:
        lv = self.left.evaluate(env)
        rv = self.right.evaluate(env)
        return self.op.python_op()(lv, rv)

    def negate(self) -> "ComparisonPredicate":
        return ComparisonPredicate(left=self.left, op=self.op.negate(), right=self.right)

    def flip(self) -> "ComparisonPredicate":
        return ComparisonPredicate(left=self.right, op=self.op.flip(), right=self.left)

    def normalize(self) -> "ComparisonPredicate":
        """Normalize so that left side is 'larger' in canonical ordering."""
        diff = self.left.subtract(self.right).simplify()
        zero = LinearExpression.from_int(0)
        return ComparisonPredicate(left=diff, op=self.op, right=zero)

    def simplify(self) -> PredicateTemplate:
        left_s = self.left.simplify()
        right_s = self.right.simplify()
        lv = left_s.constant_value()
        rv = right_s.constant_value()
        if lv is not None and rv is not None:
            return TruePredicate() if self.op.python_op()(lv, rv) else FalsePredicate()
        return ComparisonPredicate(left=left_s, op=self.op, right=right_s)

    def to_smt(self) -> str:
        l_smt = self.left.to_smt()
        r_smt = self.right.to_smt()
        sym = self.op.smt_symbol()
        if self.op == ComparisonOp.Ne:
            return f"(not (= {l_smt} {r_smt}))"
        return f"({sym} {l_smt} {r_smt})"

    def pretty_print(self) -> str:
        return f"{self.left.pretty_print()} {self.op.symbol()} {self.right.pretty_print()}"

    def is_equality(self) -> bool:
        return self.op == ComparisonOp.Eq

    def is_inequality(self) -> bool:
        return self.op in (ComparisonOp.Lt, ComparisonOp.Le, ComparisonOp.Ge, ComparisonOp.Gt)


@dataclass(frozen=True)
class TypeTagPredicate(AtomicPredicate):
    """isinstance(x, T) for type tag T."""
    variable: str
    type_tag: str
    positive: bool = True

    def variables(self) -> FrozenSet[str]:
        return frozenset({self.variable})

    def substitute(self, mapping: Mapping[str, LinearExpression]) -> "TypeTagPredicate":
        return self

    def evaluate(self, env: Mapping[str, Any]) -> bool:
        val = env.get(self.variable)
        if val is None:
            result = self.type_tag == "NoneType"
        else:
            result = type(val).__name__ == self.type_tag
        return result if self.positive else not result

    def negate(self) -> "TypeTagPredicate":
        return TypeTagPredicate(variable=self.variable, type_tag=self.type_tag, positive=not self.positive)

    def simplify(self) -> "TypeTagPredicate":
        return self

    def to_smt(self) -> str:
        base = f'(= (typeof {self.variable}) "{self.type_tag}")'
        if self.positive:
            return base
        return f"(not {base})"

    def pretty_print(self) -> str:
        if self.positive:
            return f"isinstance({self.variable}, {self.type_tag})"
        return f"not isinstance({self.variable}, {self.type_tag})"


@dataclass(frozen=True)
class NullityPredicate(AtomicPredicate):
    """is_none(x) / not is_none(x)."""
    variable: str
    is_null: bool = True

    def variables(self) -> FrozenSet[str]:
        return frozenset({self.variable})

    def substitute(self, mapping: Mapping[str, LinearExpression]) -> "NullityPredicate":
        return self

    def evaluate(self, env: Mapping[str, Any]) -> bool:
        val = env.get(self.variable)
        result = val is None
        return result if self.is_null else not result

    def negate(self) -> "NullityPredicate":
        return NullityPredicate(variable=self.variable, is_null=not self.is_null)

    def simplify(self) -> "NullityPredicate":
        return self

    def to_smt(self) -> str:
        base = f"(is_none {self.variable})"
        return base if self.is_null else f"(not {base})"

    def pretty_print(self) -> str:
        if self.is_null:
            return f"is_none({self.variable})"
        return f"not is_none({self.variable})"


@dataclass(frozen=True)
class TruthinessPredicate(AtomicPredicate):
    """is_truthy(x) / not is_truthy(x)."""
    variable: str
    is_truthy: bool = True

    def variables(self) -> FrozenSet[str]:
        return frozenset({self.variable})

    def substitute(self, mapping: Mapping[str, LinearExpression]) -> "TruthinessPredicate":
        return self

    def evaluate(self, env: Mapping[str, Any]) -> bool:
        val = env.get(self.variable)
        result = bool(val)
        return result if self.is_truthy else not result

    def negate(self) -> "TruthinessPredicate":
        return TruthinessPredicate(variable=self.variable, is_truthy=not self.is_truthy)

    def simplify(self) -> "TruthinessPredicate":
        return self

    def to_smt(self) -> str:
        base = f"(is_truthy {self.variable})"
        return base if self.is_truthy else f"(not {base})"

    def pretty_print(self) -> str:
        if self.is_truthy:
            return f"is_truthy({self.variable})"
        return f"not is_truthy({self.variable})"


@dataclass(frozen=True)
class HasAttrPredicate(AtomicPredicate):
    """hasattr(x, k)."""
    variable: str
    attribute: str
    positive: bool = True

    def variables(self) -> FrozenSet[str]:
        return frozenset({self.variable})

    def substitute(self, mapping: Mapping[str, LinearExpression]) -> "HasAttrPredicate":
        return self

    def evaluate(self, env: Mapping[str, Any]) -> bool:
        val = env.get(self.variable)
        if val is None:
            result = False
        else:
            result = hasattr(val, self.attribute)
        return result if self.positive else not result

    def negate(self) -> "HasAttrPredicate":
        return HasAttrPredicate(variable=self.variable, attribute=self.attribute, positive=not self.positive)

    def simplify(self) -> "HasAttrPredicate":
        return self

    def to_smt(self) -> str:
        base = f'(hasattr {self.variable} "{self.attribute}")'
        return base if self.positive else f"(not {base})"

    def pretty_print(self) -> str:
        if self.positive:
            return f"hasattr({self.variable}, {self.attribute!r})"
        return f"not hasattr({self.variable}, {self.attribute!r})"


# ---------------------------------------------------------------------------
# Boolean connectives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TruePredicate(PredicateTemplate):
    """Top element (always true)."""

    def variables(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, mapping: Mapping[str, LinearExpression]) -> "TruePredicate":
        return self

    def evaluate(self, env: Mapping[str, Any]) -> bool:
        return True

    def negate(self) -> "FalsePredicate":
        return FalsePredicate()

    def simplify(self) -> "TruePredicate":
        return self

    def to_smt(self) -> str:
        return "true"

    def pretty_print(self) -> str:
        return "⊤"

    def __hash__(self) -> int:
        return hash("TruePredicate")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TruePredicate)


@dataclass(frozen=True)
class FalsePredicate(PredicateTemplate):
    """Bottom element (always false)."""

    def variables(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, mapping: Mapping[str, LinearExpression]) -> "FalsePredicate":
        return self

    def evaluate(self, env: Mapping[str, Any]) -> bool:
        return False

    def negate(self) -> "TruePredicate":
        return TruePredicate()

    def simplify(self) -> "FalsePredicate":
        return self

    def to_smt(self) -> str:
        return "false"

    def pretty_print(self) -> str:
        return "⊥"

    def __hash__(self) -> int:
        return hash("FalsePredicate")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FalsePredicate)


@dataclass(frozen=True)
class Conjunction(PredicateTemplate):
    """P₁ ∧ P₂ ∧ ... with flattening."""
    children: Tuple[PredicateTemplate, ...]

    @staticmethod
    def create(*preds: PredicateTemplate) -> PredicateTemplate:
        flat: List[PredicateTemplate] = []
        for p in preds:
            if isinstance(p, FalsePredicate):
                return FalsePredicate()
            if isinstance(p, TruePredicate):
                continue
            if isinstance(p, Conjunction):
                flat.extend(p.children)
            else:
                flat.append(p)
        seen: List[PredicateTemplate] = []
        seen_set: Set[int] = set()
        for p in flat:
            h = hash(p)
            if h not in seen_set:
                seen.append(p)
                seen_set.add(h)
            elif p not in seen:
                seen.append(p)
        if not seen:
            return TruePredicate()
        if len(seen) == 1:
            return seen[0]
        return Conjunction(children=tuple(seen))

    def variables(self) -> FrozenSet[str]:
        result: Set[str] = set()
        for c in self.children:
            result |= c.variables()
        return frozenset(result)

    def substitute(self, mapping: Mapping[str, LinearExpression]) -> PredicateTemplate:
        return Conjunction.create(*(c.substitute(mapping) for c in self.children))

    def evaluate(self, env: Mapping[str, Any]) -> bool:
        return all(c.evaluate(env) for c in self.children)

    def negate(self) -> PredicateTemplate:
        return Disjunction.create(*(c.negate() for c in self.children))

    def simplify(self) -> PredicateTemplate:
        simplified = [c.simplify() for c in self.children]
        return Conjunction.create(*simplified)

    def to_smt(self) -> str:
        if len(self.children) == 0:
            return "true"
        if len(self.children) == 1:
            return self.children[0].to_smt()
        inner = " ".join(c.to_smt() for c in self.children)
        return f"(and {inner})"

    def pretty_print(self) -> str:
        parts = [c.pretty_print() for c in self.children]
        return " ∧ ".join(f"({p})" if isinstance(c, Disjunction) else p
                          for p, c in zip(parts, self.children))

    def conjuncts(self) -> List[PredicateTemplate]:
        return list(self.children)

    def _collect_atoms(self, acc: Set[AtomicPredicate]) -> None:
        for c in self.children:
            c._collect_atoms(acc)

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def __hash__(self) -> int:
        return hash(("Conjunction", self.children))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Conjunction) and self.children == other.children


@dataclass(frozen=True)
class Disjunction(PredicateTemplate):
    """P₁ ∨ P₂ ∨ ... with flattening."""
    children: Tuple[PredicateTemplate, ...]

    @staticmethod
    def create(*preds: PredicateTemplate) -> PredicateTemplate:
        flat: List[PredicateTemplate] = []
        for p in preds:
            if isinstance(p, TruePredicate):
                return TruePredicate()
            if isinstance(p, FalsePredicate):
                continue
            if isinstance(p, Disjunction):
                flat.extend(p.children)
            else:
                flat.append(p)
        seen: List[PredicateTemplate] = []
        seen_set: Set[int] = set()
        for p in flat:
            h = hash(p)
            if h not in seen_set:
                seen.append(p)
                seen_set.add(h)
            elif p not in seen:
                seen.append(p)
        if not seen:
            return FalsePredicate()
        if len(seen) == 1:
            return seen[0]
        return Disjunction(children=tuple(seen))

    def variables(self) -> FrozenSet[str]:
        result: Set[str] = set()
        for c in self.children:
            result |= c.variables()
        return frozenset(result)

    def substitute(self, mapping: Mapping[str, LinearExpression]) -> PredicateTemplate:
        return Disjunction.create(*(c.substitute(mapping) for c in self.children))

    def evaluate(self, env: Mapping[str, Any]) -> bool:
        return any(c.evaluate(env) for c in self.children)

    def negate(self) -> PredicateTemplate:
        return Conjunction.create(*(c.negate() for c in self.children))

    def simplify(self) -> PredicateTemplate:
        simplified = [c.simplify() for c in self.children]
        return Disjunction.create(*simplified)

    def to_smt(self) -> str:
        if len(self.children) == 0:
            return "false"
        if len(self.children) == 1:
            return self.children[0].to_smt()
        inner = " ".join(c.to_smt() for c in self.children)
        return f"(or {inner})"

    def pretty_print(self) -> str:
        parts = [c.pretty_print() for c in self.children]
        return " ∨ ".join(f"({p})" if isinstance(c, Conjunction) else p
                          for p, c in zip(parts, self.children))

    def disjuncts(self) -> List[PredicateTemplate]:
        return list(self.children)

    def _collect_atoms(self, acc: Set[AtomicPredicate]) -> None:
        for c in self.children:
            c._collect_atoms(acc)

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def __hash__(self) -> int:
        return hash(("Disjunction", self.children))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Disjunction) and self.children == other.children


@dataclass(frozen=True)
class Negation(PredicateTemplate):
    """¬P with push-through via De Morgan."""
    child: PredicateTemplate

    def variables(self) -> FrozenSet[str]:
        return self.child.variables()

    def substitute(self, mapping: Mapping[str, LinearExpression]) -> PredicateTemplate:
        return Negation(child=self.child.substitute(mapping))

    def evaluate(self, env: Mapping[str, Any]) -> bool:
        return not self.child.evaluate(env)

    def negate(self) -> PredicateTemplate:
        return self.child

    def simplify(self) -> PredicateTemplate:
        inner = self.child.simplify()
        if isinstance(inner, TruePredicate):
            return FalsePredicate()
        if isinstance(inner, FalsePredicate):
            return TruePredicate()
        if isinstance(inner, Negation):
            return inner.child
        if isinstance(inner, AtomicPredicate):
            return inner.negate()
        if isinstance(inner, Conjunction):
            return Disjunction.create(*(c.negate() for c in inner.children)).simplify()
        if isinstance(inner, Disjunction):
            return Conjunction.create(*(c.negate() for c in inner.children)).simplify()
        return Negation(child=inner)

    def push_negation(self) -> PredicateTemplate:
        """Push negation inward using De Morgan's laws."""
        return self.child.negate()

    def to_smt(self) -> str:
        return f"(not {self.child.to_smt()})"

    def pretty_print(self) -> str:
        inner = self.child.pretty_print()
        if isinstance(self.child, (Conjunction, Disjunction)):
            return f"¬({inner})"
        return f"¬{inner}"

    def _collect_atoms(self, acc: Set[AtomicPredicate]) -> None:
        self.child._collect_atoms(acc)

    def depth(self) -> int:
        return 1 + self.child.depth()

    def size(self) -> int:
        return 1 + self.child.size()

    def __hash__(self) -> int:
        return hash(("Negation", self.child))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Negation) and self.child == other.child


@dataclass(frozen=True)
class Implication(PredicateTemplate):
    """P₁ → P₂  (desugared to ¬P₁ ∨ P₂)."""
    antecedent: PredicateTemplate
    consequent: PredicateTemplate

    def variables(self) -> FrozenSet[str]:
        return self.antecedent.variables() | self.consequent.variables()

    def substitute(self, mapping: Mapping[str, LinearExpression]) -> PredicateTemplate:
        return Implication(
            antecedent=self.antecedent.substitute(mapping),
            consequent=self.consequent.substitute(mapping),
        )

    def evaluate(self, env: Mapping[str, Any]) -> bool:
        return (not self.antecedent.evaluate(env)) or self.consequent.evaluate(env)

    def negate(self) -> PredicateTemplate:
        return Conjunction.create(self.antecedent, self.consequent.negate())

    def desugar(self) -> PredicateTemplate:
        return Disjunction.create(self.antecedent.negate(), self.consequent)

    def simplify(self) -> PredicateTemplate:
        return self.desugar().simplify()

    def to_smt(self) -> str:
        return f"(=> {self.antecedent.to_smt()} {self.consequent.to_smt()})"

    def pretty_print(self) -> str:
        return f"{self.antecedent.pretty_print()} → {self.consequent.pretty_print()}"

    def _collect_atoms(self, acc: Set[AtomicPredicate]) -> None:
        self.antecedent._collect_atoms(acc)
        self.consequent._collect_atoms(acc)

    def depth(self) -> int:
        return 1 + max(self.antecedent.depth(), self.consequent.depth())

    def size(self) -> int:
        return 1 + self.antecedent.size() + self.consequent.size()

    def __hash__(self) -> int:
        return hash(("Implication", self.antecedent, self.consequent))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Implication)
                and self.antecedent == other.antecedent
                and self.consequent == other.consequent)


# ---------------------------------------------------------------------------
# PredicateFactory
# ---------------------------------------------------------------------------

class PredicateFactory:
    """Convenience constructors for predicate templates."""

    @staticmethod
    def var(name: str, sort: PredicateSort = PredicateSort.Int) -> LinearExpression:
        return LinearExpression.from_term(Term(name, sort))

    @staticmethod
    def const(value: int) -> LinearExpression:
        return LinearExpression.from_int(value)

    @staticmethod
    def len_of(var: str) -> LinearExpression:
        return LinearExpression.from_len(var)

    @staticmethod
    def lt(left: LinearExpression, right: LinearExpression) -> ComparisonPredicate:
        return ComparisonPredicate(left=left, op=ComparisonOp.Lt, right=right)

    @staticmethod
    def le(left: LinearExpression, right: LinearExpression) -> ComparisonPredicate:
        return ComparisonPredicate(left=left, op=ComparisonOp.Le, right=right)

    @staticmethod
    def eq(left: LinearExpression, right: LinearExpression) -> ComparisonPredicate:
        return ComparisonPredicate(left=left, op=ComparisonOp.Eq, right=right)

    @staticmethod
    def ne(left: LinearExpression, right: LinearExpression) -> ComparisonPredicate:
        return ComparisonPredicate(left=left, op=ComparisonOp.Ne, right=right)

    @staticmethod
    def ge(left: LinearExpression, right: LinearExpression) -> ComparisonPredicate:
        return ComparisonPredicate(left=left, op=ComparisonOp.Ge, right=right)

    @staticmethod
    def gt(left: LinearExpression, right: LinearExpression) -> ComparisonPredicate:
        return ComparisonPredicate(left=left, op=ComparisonOp.Gt, right=right)

    @staticmethod
    def isinstance_of(var: str, tag: str) -> TypeTagPredicate:
        return TypeTagPredicate(variable=var, type_tag=tag, positive=True)

    @staticmethod
    def not_isinstance(var: str, tag: str) -> TypeTagPredicate:
        return TypeTagPredicate(variable=var, type_tag=tag, positive=False)

    @staticmethod
    def is_none(var: str) -> NullityPredicate:
        return NullityPredicate(variable=var, is_null=True)

    @staticmethod
    def is_not_none(var: str) -> NullityPredicate:
        return NullityPredicate(variable=var, is_null=False)

    @staticmethod
    def truthy(var: str) -> TruthinessPredicate:
        return TruthinessPredicate(variable=var, is_truthy=True)

    @staticmethod
    def falsy(var: str) -> TruthinessPredicate:
        return TruthinessPredicate(variable=var, is_truthy=False)

    @staticmethod
    def has_attr(var: str, attr: str) -> HasAttrPredicate:
        return HasAttrPredicate(variable=var, attribute=attr, positive=True)

    @staticmethod
    def not_has_attr(var: str, attr: str) -> HasAttrPredicate:
        return HasAttrPredicate(variable=var, attribute=attr, positive=False)

    @staticmethod
    def top() -> TruePredicate:
        return TruePredicate()

    @staticmethod
    def bottom() -> FalsePredicate:
        return FalsePredicate()

    @staticmethod
    def conjunction(*preds: PredicateTemplate) -> PredicateTemplate:
        return Conjunction.create(*preds)

    @staticmethod
    def disjunction(*preds: PredicateTemplate) -> PredicateTemplate:
        return Disjunction.create(*preds)

    @staticmethod
    def negation(pred: PredicateTemplate) -> PredicateTemplate:
        return pred.negate()

    @staticmethod
    def implication(antecedent: PredicateTemplate, consequent: PredicateTemplate) -> Implication:
        return Implication(antecedent=antecedent, consequent=consequent)

    @staticmethod
    def in_range(var: str, lo: int, hi: int) -> PredicateTemplate:
        """lo <= var < hi."""
        v = PredicateFactory.var(var)
        return Conjunction.create(
            ComparisonPredicate(left=LinearExpression.from_int(lo), op=ComparisonOp.Le, right=v),
            ComparisonPredicate(left=v, op=ComparisonOp.Lt, right=LinearExpression.from_int(hi)),
        )

    @staticmethod
    def bounded_index(idx_var: str, arr_var: str) -> PredicateTemplate:
        """0 <= idx < len(arr)."""
        idx = PredicateFactory.var(idx_var)
        return Conjunction.create(
            ComparisonPredicate(left=LinearExpression.from_int(0), op=ComparisonOp.Le, right=idx),
            ComparisonPredicate(left=idx, op=ComparisonOp.Lt, right=PredicateFactory.len_of(arr_var)),
        )


# ---------------------------------------------------------------------------
# PredicateNormalizer
# ---------------------------------------------------------------------------

class PredicateNormalizer:
    """Normalize predicates to canonical form (NNF, sorted conjuncts)."""

    @staticmethod
    def to_nnf(pred: PredicateTemplate) -> PredicateTemplate:
        """Convert to Negation Normal Form (push negations to atoms)."""
        if isinstance(pred, (TruePredicate, FalsePredicate)):
            return pred
        if isinstance(pred, AtomicPredicate):
            return pred
        if isinstance(pred, Negation):
            inner = pred.child
            if isinstance(inner, TruePredicate):
                return FalsePredicate()
            if isinstance(inner, FalsePredicate):
                return TruePredicate()
            if isinstance(inner, AtomicPredicate):
                return inner.negate()
            if isinstance(inner, Negation):
                return PredicateNormalizer.to_nnf(inner.child)
            if isinstance(inner, Conjunction):
                return Disjunction.create(
                    *(PredicateNormalizer.to_nnf(c.negate()) for c in inner.children)
                )
            if isinstance(inner, Disjunction):
                return Conjunction.create(
                    *(PredicateNormalizer.to_nnf(c.negate()) for c in inner.children)
                )
            if isinstance(inner, Implication):
                return PredicateNormalizer.to_nnf(
                    Conjunction.create(inner.antecedent, inner.consequent.negate())
                )
            return Negation(child=PredicateNormalizer.to_nnf(inner))
        if isinstance(pred, Conjunction):
            return Conjunction.create(
                *(PredicateNormalizer.to_nnf(c) for c in pred.children)
            )
        if isinstance(pred, Disjunction):
            return Disjunction.create(
                *(PredicateNormalizer.to_nnf(c) for c in pred.children)
            )
        if isinstance(pred, Implication):
            return PredicateNormalizer.to_nnf(pred.desugar())
        return pred

    @staticmethod
    def sort_conjuncts(pred: PredicateTemplate) -> PredicateTemplate:
        """Sort conjuncts/disjuncts by their string representation for canonicality."""
        if isinstance(pred, Conjunction):
            sorted_children = sorted(
                (PredicateNormalizer.sort_conjuncts(c) for c in pred.children),
                key=lambda p: p.pretty_print(),
            )
            return Conjunction.create(*sorted_children)
        if isinstance(pred, Disjunction):
            sorted_children = sorted(
                (PredicateNormalizer.sort_conjuncts(c) for c in pred.children),
                key=lambda p: p.pretty_print(),
            )
            return Disjunction.create(*sorted_children)
        if isinstance(pred, Negation):
            return Negation(child=PredicateNormalizer.sort_conjuncts(pred.child))
        if isinstance(pred, Implication):
            return Implication(
                antecedent=PredicateNormalizer.sort_conjuncts(pred.antecedent),
                consequent=PredicateNormalizer.sort_conjuncts(pred.consequent),
            )
        return pred

    @staticmethod
    def normalize(pred: PredicateTemplate) -> PredicateTemplate:
        """Full normalization: NNF + sort + simplify."""
        result = PredicateNormalizer.to_nnf(pred)
        result = result.simplify()
        result = PredicateNormalizer.sort_conjuncts(result)
        return result

    @staticmethod
    def flatten(pred: PredicateTemplate) -> PredicateTemplate:
        """Flatten nested conjunctions and disjunctions."""
        if isinstance(pred, Conjunction):
            flat: List[PredicateTemplate] = []
            for c in pred.children:
                fc = PredicateNormalizer.flatten(c)
                if isinstance(fc, Conjunction):
                    flat.extend(fc.children)
                else:
                    flat.append(fc)
            return Conjunction.create(*flat)
        if isinstance(pred, Disjunction):
            flat = []
            for c in pred.children:
                fc = PredicateNormalizer.flatten(c)
                if isinstance(fc, Disjunction):
                    flat.extend(fc.children)
                else:
                    flat.append(fc)
            return Disjunction.create(*flat)
        return pred

    @staticmethod
    def to_cnf(pred: PredicateTemplate) -> PredicateTemplate:
        """Convert to Conjunctive Normal Form (conjunction of disjunctions)."""
        nnf = PredicateNormalizer.to_nnf(pred)
        return PredicateNormalizer._distribute_cnf(nnf)

    @staticmethod
    def _distribute_cnf(pred: PredicateTemplate) -> PredicateTemplate:
        if isinstance(pred, Conjunction):
            return Conjunction.create(
                *(PredicateNormalizer._distribute_cnf(c) for c in pred.children)
            )
        if isinstance(pred, Disjunction):
            children = [PredicateNormalizer._distribute_cnf(c) for c in pred.children]
            conj_children = [c for c in children if isinstance(c, Conjunction)]
            non_conj = [c for c in children if not isinstance(c, Conjunction)]
            if not conj_children:
                return Disjunction.create(*children)
            first_conj = conj_children[0]
            rest = non_conj + conj_children[1:]
            rest_pred = Disjunction.create(*rest) if len(rest) > 1 else (rest[0] if rest else FalsePredicate())
            distributed = []
            for c in first_conj.children:
                distributed.append(PredicateNormalizer._distribute_cnf(
                    Disjunction.create(c, rest_pred)
                ))
            return Conjunction.create(*distributed)
        return pred

    @staticmethod
    def to_dnf(pred: PredicateTemplate) -> PredicateTemplate:
        """Convert to Disjunctive Normal Form (disjunction of conjunctions)."""
        nnf = PredicateNormalizer.to_nnf(pred)
        return PredicateNormalizer._distribute_dnf(nnf)

    @staticmethod
    def _distribute_dnf(pred: PredicateTemplate) -> PredicateTemplate:
        if isinstance(pred, Disjunction):
            return Disjunction.create(
                *(PredicateNormalizer._distribute_dnf(c) for c in pred.children)
            )
        if isinstance(pred, Conjunction):
            children = [PredicateNormalizer._distribute_dnf(c) for c in pred.children]
            disj_children = [c for c in children if isinstance(c, Disjunction)]
            non_disj = [c for c in children if not isinstance(c, Disjunction)]
            if not disj_children:
                return Conjunction.create(*children)
            first_disj = disj_children[0]
            rest = non_disj + disj_children[1:]
            rest_pred = Conjunction.create(*rest) if len(rest) > 1 else (rest[0] if rest else TruePredicate())
            distributed = []
            for c in first_disj.children:
                distributed.append(PredicateNormalizer._distribute_dnf(
                    Conjunction.create(c, rest_pred)
                ))
            return Disjunction.create(*distributed)
        return pred


# ---------------------------------------------------------------------------
# PredicateSimplifier
# ---------------------------------------------------------------------------

class PredicateSimplifier:
    """Simplify tautologies, contradictions, and subsumptions."""

    @staticmethod
    def simplify(pred: PredicateTemplate) -> PredicateTemplate:
        """Simplify a predicate by removing tautologies and contradictions."""
        prev = None
        current = pred
        for _ in range(10):
            if current == prev:
                break
            prev = current
            current = PredicateSimplifier._simplify_step(current)
        return current

    @staticmethod
    def _simplify_step(pred: PredicateTemplate) -> PredicateTemplate:
        if isinstance(pred, (TruePredicate, FalsePredicate)):
            return pred
        if isinstance(pred, AtomicPredicate):
            return pred.simplify()
        if isinstance(pred, Negation):
            inner = PredicateSimplifier._simplify_step(pred.child)
            if isinstance(inner, TruePredicate):
                return FalsePredicate()
            if isinstance(inner, FalsePredicate):
                return TruePredicate()
            if isinstance(inner, Negation):
                return inner.child
            if isinstance(inner, AtomicPredicate):
                return inner.negate()
            return Negation(child=inner)
        if isinstance(pred, Conjunction):
            return PredicateSimplifier._simplify_conjunction(pred)
        if isinstance(pred, Disjunction):
            return PredicateSimplifier._simplify_disjunction(pred)
        if isinstance(pred, Implication):
            ant = PredicateSimplifier._simplify_step(pred.antecedent)
            con = PredicateSimplifier._simplify_step(pred.consequent)
            if isinstance(ant, FalsePredicate) or isinstance(con, TruePredicate):
                return TruePredicate()
            if isinstance(ant, TruePredicate):
                return con
            if isinstance(con, FalsePredicate):
                return ant.negate()
            if ant == con:
                return TruePredicate()
            return Implication(antecedent=ant, consequent=con)
        return pred

    @staticmethod
    def _simplify_conjunction(conj: Conjunction) -> PredicateTemplate:
        simplified = [PredicateSimplifier._simplify_step(c) for c in conj.children]
        result: List[PredicateTemplate] = []
        for c in simplified:
            if isinstance(c, FalsePredicate):
                return FalsePredicate()
            if isinstance(c, TruePredicate):
                continue
            result.append(c)
        result = PredicateSimplifier._remove_duplicates(result)
        if PredicateSimplifier._has_contradiction(result):
            return FalsePredicate()
        result = PredicateSimplifier._subsume_conjuncts(result)
        if not result:
            return TruePredicate()
        if len(result) == 1:
            return result[0]
        return Conjunction(children=tuple(result))

    @staticmethod
    def _simplify_disjunction(disj: Disjunction) -> PredicateTemplate:
        simplified = [PredicateSimplifier._simplify_step(c) for c in disj.children]
        result: List[PredicateTemplate] = []
        for c in simplified:
            if isinstance(c, TruePredicate):
                return TruePredicate()
            if isinstance(c, FalsePredicate):
                continue
            result.append(c)
        result = PredicateSimplifier._remove_duplicates(result)
        if PredicateSimplifier._has_tautology_pair(result):
            return TruePredicate()
        result = PredicateSimplifier._subsume_disjuncts(result)
        if not result:
            return FalsePredicate()
        if len(result) == 1:
            return result[0]
        return Disjunction(children=tuple(result))

    @staticmethod
    def _remove_duplicates(preds: List[PredicateTemplate]) -> List[PredicateTemplate]:
        seen: List[PredicateTemplate] = []
        for p in preds:
            if p not in seen:
                seen.append(p)
        return seen

    @staticmethod
    def _has_contradiction(preds: List[PredicateTemplate]) -> bool:
        """Check if any predicate appears alongside its negation."""
        for i, p in enumerate(preds):
            neg = p.negate()
            for j, q in enumerate(preds):
                if i != j and q == neg:
                    return True
        # Check comparison contradictions
        comparisons = [p for p in preds if isinstance(p, ComparisonPredicate)]
        for i, c1 in enumerate(comparisons):
            for j, c2 in enumerate(comparisons):
                if i < j and c1.left == c2.left and c1.right == c2.right:
                    if c1.op == c2.op.negate():
                        return True
                    if c1.op == ComparisonOp.Lt and c2.op == ComparisonOp.Gt:
                        return True
                    if c1.op == ComparisonOp.Gt and c2.op == ComparisonOp.Lt:
                        return True
                    if c1.op == ComparisonOp.Eq and c2.op == ComparisonOp.Ne:
                        return True
        return False

    @staticmethod
    def _has_tautology_pair(preds: List[PredicateTemplate]) -> bool:
        """Check if P ∨ ¬P appears."""
        for i, p in enumerate(preds):
            neg = p.negate()
            for j, q in enumerate(preds):
                if i != j and q == neg:
                    return True
        return False

    @staticmethod
    def _subsume_conjuncts(preds: List[PredicateTemplate]) -> List[PredicateTemplate]:
        """Remove conjuncts subsumed by stronger conjuncts.
        E.g., x < 5 ∧ x < 10 simplifies to x < 5.
        """
        result = list(preds)
        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                for j in range(len(result)):
                    if i == j or i >= len(result) or j >= len(result):
                        continue
                    if PredicateSimplifier._conjunct_subsumes(result[i], result[j]):
                        result.pop(j)
                        changed = True
                        break
                if changed:
                    break
        return result

    @staticmethod
    def _subsume_disjuncts(preds: List[PredicateTemplate]) -> List[PredicateTemplate]:
        """Remove disjuncts subsumed by weaker disjuncts.
        E.g., x < 5 ∨ x < 10 simplifies to x < 10.
        """
        result = list(preds)
        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                for j in range(len(result)):
                    if i == j or i >= len(result) or j >= len(result):
                        continue
                    if PredicateSimplifier._disjunct_subsumes(result[i], result[j]):
                        result.pop(j)
                        changed = True
                        break
                if changed:
                    break
        return result

    @staticmethod
    def _conjunct_subsumes(stronger: PredicateTemplate, weaker: PredicateTemplate) -> bool:
        """Check if 'stronger' implies 'weaker' in a conjunction (weaker is redundant)."""
        if not isinstance(stronger, ComparisonPredicate) or not isinstance(weaker, ComparisonPredicate):
            return False
        if stronger.left != weaker.left or stronger.right != weaker.right:
            return False
        s_cv = stronger.right.constant_value()
        w_cv = weaker.right.constant_value()
        if s_cv is None or w_cv is None:
            return False
        # x < 5 implies x < 10
        if stronger.op == ComparisonOp.Lt and weaker.op == ComparisonOp.Lt:
            return s_cv <= w_cv
        if stronger.op == ComparisonOp.Le and weaker.op == ComparisonOp.Le:
            return s_cv <= w_cv
        if stronger.op == ComparisonOp.Gt and weaker.op == ComparisonOp.Gt:
            return s_cv >= w_cv
        if stronger.op == ComparisonOp.Ge and weaker.op == ComparisonOp.Ge:
            return s_cv >= w_cv
        # x < 5 implies x <= 5
        if stronger.op == ComparisonOp.Lt and weaker.op == ComparisonOp.Le:
            return s_cv <= w_cv
        if stronger.op == ComparisonOp.Gt and weaker.op == ComparisonOp.Ge:
            return s_cv >= w_cv
        # x == 5 implies x <= 5
        if stronger.op == ComparisonOp.Eq and weaker.op == ComparisonOp.Le:
            return s_cv <= w_cv
        if stronger.op == ComparisonOp.Eq and weaker.op == ComparisonOp.Ge:
            return s_cv >= w_cv
        return False

    @staticmethod
    def _disjunct_subsumes(weaker: PredicateTemplate, stronger: PredicateTemplate) -> bool:
        """In a disjunction, the weaker predicate subsumes the stronger one."""
        return PredicateSimplifier._conjunct_subsumes(stronger, weaker)

    @staticmethod
    def absorb(pred: PredicateTemplate) -> PredicateTemplate:
        """Apply absorption law: P ∧ (P ∨ Q) = P; P ∨ (P ∧ Q) = P."""
        if isinstance(pred, Conjunction):
            children = list(pred.children)
            absorbed: List[PredicateTemplate] = []
            for i, c in enumerate(children):
                keep = True
                if isinstance(c, Disjunction):
                    for j, other in enumerate(children):
                        if i != j and other in c.children:
                            keep = False
                            break
                if keep:
                    absorbed.append(c)
            return Conjunction.create(*absorbed)
        if isinstance(pred, Disjunction):
            children = list(pred.children)
            absorbed = []
            for i, c in enumerate(children):
                keep = True
                if isinstance(c, Conjunction):
                    for j, other in enumerate(children):
                        if i != j and other in c.children:
                            keep = False
                            break
                if keep:
                    absorbed.append(c)
            return Disjunction.create(*absorbed)
        return pred

    @staticmethod
    def full_simplify(pred: PredicateTemplate) -> PredicateTemplate:
        """Full simplification pipeline."""
        result = PredicateSimplifier.simplify(pred)
        result = PredicateSimplifier.absorb(result)
        result = PredicateNormalizer.normalize(result)
        return result


# ---------------------------------------------------------------------------
# PredicateEvaluator
# ---------------------------------------------------------------------------

class PredicateEvaluator:
    """Concrete evaluation of predicates given variable bindings."""

    def __init__(self, env: Optional[Mapping[str, Any]] = None) -> None:
        self._env: Dict[str, Any] = dict(env) if env else {}

    def bind(self, name: str, value: Any) -> "PredicateEvaluator":
        new_env = dict(self._env)
        new_env[name] = value
        return PredicateEvaluator(new_env)

    def bind_many(self, bindings: Mapping[str, Any]) -> "PredicateEvaluator":
        new_env = dict(self._env)
        new_env.update(bindings)
        return PredicateEvaluator(new_env)

    @property
    def env(self) -> Dict[str, Any]:
        return dict(self._env)

    def evaluate(self, pred: PredicateTemplate) -> bool:
        return pred.evaluate(self._env)

    def evaluate_partial(self, pred: PredicateTemplate) -> Optional[bool]:
        """Evaluate, returning None if variables are missing."""
        try:
            return pred.evaluate(self._env)
        except (KeyError, TypeError):
            return None

    def evaluate_all(self, preds: Iterable[PredicateTemplate]) -> Dict[PredicateTemplate, bool]:
        results: Dict[PredicateTemplate, bool] = {}
        for p in preds:
            results[p] = self.evaluate(p)
        return results

    def evaluate_all_partial(self, preds: Iterable[PredicateTemplate]) -> Dict[PredicateTemplate, Optional[bool]]:
        results: Dict[PredicateTemplate, Optional[bool]] = {}
        for p in preds:
            results[p] = self.evaluate_partial(p)
        return results

    def satisfies(self, pred: PredicateTemplate) -> bool:
        return self.evaluate(pred)

    def violates(self, pred: PredicateTemplate) -> bool:
        return not self.evaluate(pred)

    def find_violations(self, preds: Iterable[PredicateTemplate]) -> List[PredicateTemplate]:
        return [p for p in preds if self.violates(p)]

    def find_satisfied(self, preds: Iterable[PredicateTemplate]) -> List[PredicateTemplate]:
        return [p for p in preds if self.satisfies(p)]

    def check_implication(self, antecedent: PredicateTemplate, consequent: PredicateTemplate) -> bool:
        """Check if antecedent => consequent holds under current environment."""
        if not self.evaluate(antecedent):
            return True
        return self.evaluate(consequent)

    def generate_valuation(self, preds: Iterable[PredicateTemplate]) -> Dict[PredicateTemplate, bool]:
        """Generate a valuation mapping each predicate to its truth value."""
        return {p: self.evaluate(p) for p in preds}


# ---------------------------------------------------------------------------
# PredicatePrinter
# ---------------------------------------------------------------------------

class PredicatePrinter:
    """Human-readable and SMT-LIB formatting for predicates."""

    @staticmethod
    def pretty(pred: PredicateTemplate) -> str:
        return pred.pretty_print()

    @staticmethod
    def smt(pred: PredicateTemplate) -> str:
        return pred.to_smt()

    @staticmethod
    def smt_assert(pred: PredicateTemplate) -> str:
        return f"(assert {pred.to_smt()})"

    @staticmethod
    def smt_declare_vars(pred: PredicateTemplate) -> str:
        lines: List[str] = []
        for v in sorted(pred.variables()):
            lines.append(f"(declare-const {v} Int)")
        return "\n".join(lines)

    @staticmethod
    def smt_check_sat(pred: PredicateTemplate) -> str:
        parts = [
            PredicatePrinter.smt_declare_vars(pred),
            PredicatePrinter.smt_assert(pred),
            "(check-sat)",
            "(get-model)",
        ]
        return "\n".join(parts)

    @staticmethod
    def smt_check_valid(pred: PredicateTemplate) -> str:
        """Check validity by checking unsatisfiability of negation."""
        neg = pred.negate()
        parts = [
            PredicatePrinter.smt_declare_vars(neg),
            PredicatePrinter.smt_assert(neg),
            "(check-sat)",
        ]
        return "\n".join(parts)

    @staticmethod
    def smt_check_entailment(premise: PredicateTemplate, conclusion: PredicateTemplate) -> str:
        """Check if premise entails conclusion (premise ∧ ¬conclusion is unsat)."""
        combined = Conjunction.create(premise, conclusion.negate())
        parts = [
            PredicatePrinter.smt_declare_vars(combined),
            PredicatePrinter.smt_assert(combined),
            "(check-sat)",
        ]
        return "\n".join(parts)

    @staticmethod
    def tree_format(pred: PredicateTemplate, indent: int = 0) -> str:
        """Pretty-print as an indented tree."""
        prefix = "  " * indent
        if isinstance(pred, AtomicPredicate):
            return f"{prefix}{pred.pretty_print()}"
        if isinstance(pred, TruePredicate):
            return f"{prefix}⊤"
        if isinstance(pred, FalsePredicate):
            return f"{prefix}⊥"
        if isinstance(pred, Negation):
            return f"{prefix}¬\n{PredicatePrinter.tree_format(pred.child, indent + 1)}"
        if isinstance(pred, Conjunction):
            children = "\n".join(
                PredicatePrinter.tree_format(c, indent + 1) for c in pred.children
            )
            return f"{prefix}∧\n{children}"
        if isinstance(pred, Disjunction):
            children = "\n".join(
                PredicatePrinter.tree_format(c, indent + 1) for c in pred.children
            )
            return f"{prefix}∨\n{children}"
        if isinstance(pred, Implication):
            ant = PredicatePrinter.tree_format(pred.antecedent, indent + 1)
            con = PredicatePrinter.tree_format(pred.consequent, indent + 1)
            return f"{prefix}→\n{ant}\n{con}"
        return f"{prefix}{pred.pretty_print()}"

    @staticmethod
    def latex(pred: PredicateTemplate) -> str:
        """Convert to LaTeX representation."""
        if isinstance(pred, TruePredicate):
            return r"\top"
        if isinstance(pred, FalsePredicate):
            return r"\bot"
        if isinstance(pred, ComparisonPredicate):
            op_map = {
                ComparisonOp.Lt: "<",
                ComparisonOp.Le: r"\leq",
                ComparisonOp.Eq: "=",
                ComparisonOp.Ne: r"\neq",
                ComparisonOp.Ge: r"\geq",
                ComparisonOp.Gt: ">",
            }
            return f"{pred.left.pretty_print()} {op_map[pred.op]} {pred.right.pretty_print()}"
        if isinstance(pred, TypeTagPredicate):
            base = rf"\text{{isinstance}}({pred.variable}, \text{{{pred.type_tag}}})"
            return base if pred.positive else rf"\neg {base}"
        if isinstance(pred, NullityPredicate):
            base = rf"\text{{is\_none}}({pred.variable})"
            return base if pred.is_null else rf"\neg {base}"
        if isinstance(pred, TruthinessPredicate):
            base = rf"\text{{is\_truthy}}({pred.variable})"
            return base if pred.is_truthy else rf"\neg {base}"
        if isinstance(pred, HasAttrPredicate):
            base = rf"\text{{hasattr}}({pred.variable}, \text{{{pred.attribute}}})"
            return base if pred.positive else rf"\neg {base}"
        if isinstance(pred, Conjunction):
            parts = [PredicatePrinter.latex(c) for c in pred.children]
            return r" \land ".join(parts)
        if isinstance(pred, Disjunction):
            parts = [PredicatePrinter.latex(c) for c in pred.children]
            return r" \lor ".join(parts)
        if isinstance(pred, Negation):
            return rf"\neg ({PredicatePrinter.latex(pred.child)})"
        if isinstance(pred, Implication):
            return rf"{PredicatePrinter.latex(pred.antecedent)} \Rightarrow {PredicatePrinter.latex(pred.consequent)}"
        return pred.pretty_print()

    @staticmethod
    def summary(pred: PredicateTemplate) -> str:
        """Short summary with statistics."""
        atoms = pred.atoms()
        vs = pred.variables()
        return (
            f"Predicate: {pred.pretty_print()}\n"
            f"  Variables: {sorted(vs)}\n"
            f"  Atoms: {len(atoms)}\n"
            f"  Depth: {pred.depth()}\n"
            f"  Size: {pred.size()}"
        )

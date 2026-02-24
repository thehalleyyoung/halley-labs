"""
Core refinement type definitions for a Refinement Type Inference system
for dynamically-typed languages.

Type system:
  Base types:  τ ::= Int | Str | Bool | None | List[τ] | τ₁ → τ₂
                    | τ₁ ∪ τ₂ | {k₁:τ₁, …, kₙ:τₙ, ρ}
  Refinement:  {x : τ | φ}
  Subtyping:   Γ ⊢ {x:τ₁|φ₁} <: {x:τ₂|φ₂}  iff  τ₁ <: τ₂  and  Γ,φ₁ ⊨ φ₂
  Dependent:   (x:{v:τ₁|φ₁}) → {v:τ₂|φ₂(x)}
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List as ListT,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Sort enum – sorts of the predicate template language P
# ---------------------------------------------------------------------------

class Sort(Enum):
    """Sorts in the predicate language P."""
    INT = auto()
    BOOL = auto()
    TAG = auto()
    STR = auto()

    def __repr__(self) -> str:
        return f"Sort.{self.name}"


# ---------------------------------------------------------------------------
# Type variables & row variables
# ---------------------------------------------------------------------------

_type_var_counter = itertools.count()
_row_var_counter = itertools.count()


@dataclass(frozen=True)
class TypeVariable:
    """A unification / polymorphic type variable (α, β, …)."""
    name: str
    id: int = field(default_factory=lambda: next(_type_var_counter))

    def __repr__(self) -> str:
        return f"'{self.name}"

    def __str__(self) -> str:
        return f"'{self.name}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypeVariable):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class RowVariable:
    """Row variable ρ for open structural (object) types."""
    name: str
    id: int = field(default_factory=lambda: next(_row_var_counter))

    def __repr__(self) -> str:
        return f"ρ_{self.name}"

    def __str__(self) -> str:
        return f"ρ_{self.name}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RowVariable):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


def fresh_type_var(prefix: str = "α") -> TypeVariable:
    return TypeVariable(name=prefix)


def fresh_row_var(prefix: str = "ρ") -> RowVariable:
    return RowVariable(name=prefix)


# ---------------------------------------------------------------------------
# Base types  –  τ
# ---------------------------------------------------------------------------

class BaseType(ABC):
    """Abstract base for all types in the system."""

    @abstractmethod
    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        ...

    @abstractmethod
    def free_row_vars(self) -> FrozenSet[RowVariable]:
        ...

    @abstractmethod
    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        ...

    @abstractmethod
    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        ...

    @abstractmethod
    def _pretty(self, precedence: int) -> str:
        ...

    def pretty(self) -> str:
        return self._pretty(0)

    def __str__(self) -> str:
        return self.pretty()

    @abstractmethod
    def structural_eq(self, other: BaseType) -> bool:
        ...

    @abstractmethod
    def _structural_hash(self) -> int:
        ...

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseType):
            return NotImplemented
        return self.structural_eq(other)

    def __hash__(self) -> int:
        return self._structural_hash()


# --- Primitive types -------------------------------------------------------

@dataclass(frozen=True)
class IntType(BaseType):
    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        return frozenset()

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        return frozenset()

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        return self

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        return self

    def _pretty(self, precedence: int) -> str:
        return "Int"

    def structural_eq(self, other: BaseType) -> bool:
        return isinstance(other, IntType)

    def _structural_hash(self) -> int:
        return hash("IntType")


@dataclass(frozen=True)
class StrType(BaseType):
    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        return frozenset()

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        return frozenset()

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        return self

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        return self

    def _pretty(self, precedence: int) -> str:
        return "Str"

    def structural_eq(self, other: BaseType) -> bool:
        return isinstance(other, StrType)

    def _structural_hash(self) -> int:
        return hash("StrType")


@dataclass(frozen=True)
class BoolType(BaseType):
    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        return frozenset()

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        return frozenset()

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        return self

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        return self

    def _pretty(self, precedence: int) -> str:
        return "Bool"

    def structural_eq(self, other: BaseType) -> bool:
        return isinstance(other, BoolType)

    def _structural_hash(self) -> int:
        return hash("BoolType")


@dataclass(frozen=True)
class NoneType_(BaseType):
    """None / unit type.  Named NoneType_ to avoid clash with builtins.NoneType."""
    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        return frozenset()

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        return frozenset()

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        return self

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        return self

    def _pretty(self, precedence: int) -> str:
        return "None"

    def structural_eq(self, other: BaseType) -> bool:
        return isinstance(other, NoneType_)

    def _structural_hash(self) -> int:
        return hash("NoneType_")


# --- Compound types -------------------------------------------------------

@dataclass(frozen=True)
class ListType(BaseType):
    element: BaseType

    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        return self.element.free_type_vars()

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        return self.element.free_row_vars()

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        return ListType(self.element.substitute_type_var(mapping))

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        return ListType(self.element.substitute_row_var(mapping))

    def _pretty(self, precedence: int) -> str:
        return f"List[{self.element._pretty(0)}]"

    def structural_eq(self, other: BaseType) -> bool:
        return isinstance(other, ListType) and self.element.structural_eq(other.element)

    def _structural_hash(self) -> int:
        return hash(("ListType", self.element._structural_hash()))


@dataclass(frozen=True)
class TupleType(BaseType):
    elements: Tuple[BaseType, ...]

    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        return frozenset().union(*(e.free_type_vars() for e in self.elements))

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        return frozenset().union(*(e.free_row_vars() for e in self.elements))

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        return TupleType(tuple(e.substitute_type_var(mapping) for e in self.elements))

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        return TupleType(tuple(e.substitute_row_var(mapping) for e in self.elements))

    def _pretty(self, precedence: int) -> str:
        inner = ", ".join(e._pretty(0) for e in self.elements)
        return f"Tuple[{inner}]"

    def structural_eq(self, other: BaseType) -> bool:
        return (isinstance(other, TupleType)
                and len(self.elements) == len(other.elements)
                and all(a.structural_eq(b) for a, b in zip(self.elements, other.elements)))

    def _structural_hash(self) -> int:
        return hash(("TupleType", tuple(e._structural_hash() for e in self.elements)))


@dataclass(frozen=True)
class DictType(BaseType):
    key: BaseType
    value: BaseType

    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        return self.key.free_type_vars() | self.value.free_type_vars()

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        return self.key.free_row_vars() | self.value.free_row_vars()

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        return DictType(self.key.substitute_type_var(mapping), self.value.substitute_type_var(mapping))

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        return DictType(self.key.substitute_row_var(mapping), self.value.substitute_row_var(mapping))

    def _pretty(self, precedence: int) -> str:
        return f"Dict[{self.key._pretty(0)}, {self.value._pretty(0)}]"

    def structural_eq(self, other: BaseType) -> bool:
        return (isinstance(other, DictType)
                and self.key.structural_eq(other.key)
                and self.value.structural_eq(other.value))

    def _structural_hash(self) -> int:
        return hash(("DictType", self.key._structural_hash(), self.value._structural_hash()))


@dataclass(frozen=True)
class SetType(BaseType):
    element: BaseType

    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        return self.element.free_type_vars()

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        return self.element.free_row_vars()

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        return SetType(self.element.substitute_type_var(mapping))

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        return SetType(self.element.substitute_row_var(mapping))

    def _pretty(self, precedence: int) -> str:
        return f"Set[{self.element._pretty(0)}]"

    def structural_eq(self, other: BaseType) -> bool:
        return isinstance(other, SetType) and self.element.structural_eq(other.element)

    def _structural_hash(self) -> int:
        return hash(("SetType", self.element._structural_hash()))


@dataclass(frozen=True)
class FunctionType(BaseType):
    """τ₁ → τ₂  (simple, non-dependent arrow)."""
    params: Tuple[BaseType, ...]
    ret: BaseType

    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        s: FrozenSet[TypeVariable] = frozenset()
        for p in self.params:
            s = s | p.free_type_vars()
        return s | self.ret.free_type_vars()

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        s: FrozenSet[RowVariable] = frozenset()
        for p in self.params:
            s = s | p.free_row_vars()
        return s | self.ret.free_row_vars()

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        return FunctionType(
            tuple(p.substitute_type_var(mapping) for p in self.params),
            self.ret.substitute_type_var(mapping),
        )

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        return FunctionType(
            tuple(p.substitute_row_var(mapping) for p in self.params),
            self.ret.substitute_row_var(mapping),
        )

    def _pretty(self, precedence: int) -> str:
        if len(self.params) == 1:
            lhs = self.params[0]._pretty(2)
        else:
            lhs = "(" + ", ".join(p._pretty(0) for p in self.params) + ")"
        s = f"{lhs} → {self.ret._pretty(1)}"
        return f"({s})" if precedence > 1 else s

    def structural_eq(self, other: BaseType) -> bool:
        return (isinstance(other, FunctionType)
                and len(self.params) == len(other.params)
                and all(a.structural_eq(b) for a, b in zip(self.params, other.params))
                and self.ret.structural_eq(other.ret))

    def _structural_hash(self) -> int:
        return hash(("FunctionType",
                      tuple(p._structural_hash() for p in self.params),
                      self.ret._structural_hash()))


@dataclass(frozen=True)
class UnionType(BaseType):
    """τ₁ ∪ τ₂ ∪ … represented as a frozenset of alternatives."""
    alternatives: FrozenSet[BaseType]

    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        return frozenset().union(*(a.free_type_vars() for a in self.alternatives))

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        return frozenset().union(*(a.free_row_vars() for a in self.alternatives))

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        return make_union(*(a.substitute_type_var(mapping) for a in self.alternatives))

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        return make_union(*(a.substitute_row_var(mapping) for a in self.alternatives))

    def _pretty(self, precedence: int) -> str:
        parts = sorted((a._pretty(3) for a in self.alternatives), key=str)
        s = " ∪ ".join(parts)
        return f"({s})" if precedence > 2 else s

    def structural_eq(self, other: BaseType) -> bool:
        if not isinstance(other, UnionType):
            return False
        return self.alternatives == other.alternatives

    def _structural_hash(self) -> int:
        return hash(("UnionType", self.alternatives))


@dataclass(frozen=True)
class ObjectType(BaseType):
    """Structural object type with an optional row variable for openness.

    {k₁: τ₁, …, kₙ: τₙ, ρ}  when *row_var* is not None  (open)
    {k₁: τ₁, …, kₙ: τₙ}     when *row_var* is None       (closed)
    """
    fields: Tuple[Tuple[str, BaseType], ...]
    row_var: Optional[RowVariable] = None

    @property
    def field_dict(self) -> Dict[str, BaseType]:
        return dict(self.fields)

    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        return frozenset().union(*(t.free_type_vars() for _, t in self.fields))

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        s = frozenset().union(*(t.free_row_vars() for _, t in self.fields))
        if self.row_var is not None:
            s = s | frozenset({self.row_var})
        return s

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        return ObjectType(
            tuple((k, v.substitute_type_var(mapping)) for k, v in self.fields),
            self.row_var,
        )

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        new_fields = [(k, v.substitute_row_var(mapping)) for k, v in self.fields]
        new_row = self.row_var
        if self.row_var is not None and self.row_var in mapping:
            extra_fields, new_row = mapping[self.row_var]
            for ek, ev in extra_fields.items():
                if ek not in dict(new_fields):
                    new_fields.append((ek, ev))
        return ObjectType(
            tuple(sorted(new_fields, key=lambda kv: kv[0])),
            new_row,
        )

    def _pretty(self, precedence: int) -> str:
        parts = [f"{k}: {v._pretty(0)}" for k, v in self.fields]
        if self.row_var is not None:
            parts.append(str(self.row_var))
        return "{" + ", ".join(parts) + "}"

    def structural_eq(self, other: BaseType) -> bool:
        if not isinstance(other, ObjectType):
            return False
        return (self.fields == other.fields and self.row_var == other.row_var)

    def _structural_hash(self) -> int:
        return hash(("ObjectType", self.fields, self.row_var))


@dataclass(frozen=True)
class TypeVarType(BaseType):
    """A type that is just a type variable (placeholder during inference)."""
    var: TypeVariable

    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        return frozenset({self.var})

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        return frozenset()

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        return mapping.get(self.var, self)

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        return self

    def _pretty(self, precedence: int) -> str:
        return str(self.var)

    def structural_eq(self, other: BaseType) -> bool:
        return isinstance(other, TypeVarType) and self.var == other.var

    def _structural_hash(self) -> int:
        return hash(("TypeVarType", self.var))


@dataclass(frozen=True)
class AnyType(BaseType):
    """Top type – every value inhabits Any."""

    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        return frozenset()

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        return frozenset()

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        return self

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        return self

    def _pretty(self, precedence: int) -> str:
        return "Any"

    def structural_eq(self, other: BaseType) -> bool:
        return isinstance(other, AnyType)

    def _structural_hash(self) -> int:
        return hash("AnyType")


@dataclass(frozen=True)
class NeverType(BaseType):
    """Bottom type – no value inhabits Never."""

    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        return frozenset()

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        return frozenset()

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> BaseType:
        return self

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> BaseType:
        return self

    def _pretty(self, precedence: int) -> str:
        return "Never"

    def structural_eq(self, other: BaseType) -> bool:
        return isinstance(other, NeverType)

    def _structural_hash(self) -> int:
        return hash("NeverType")


# ---------------------------------------------------------------------------
# Linear expressions  –  e ::= x | c | len(x) | e+e | e-e | e*c | e//c | e%c
# ---------------------------------------------------------------------------

class LinearExpr(ABC):
    """Arithmetic expression in the predicate language."""

    @abstractmethod
    def free_vars(self) -> FrozenSet[str]:
        ...

    @abstractmethod
    def substitute(self, var: str, expr: LinearExpr) -> LinearExpr:
        ...

    @abstractmethod
    def pretty(self) -> str:
        ...

    def __str__(self) -> str:
        return self.pretty()

    def __repr__(self) -> str:
        return self.pretty()

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...

    # Convenience operators
    def __add__(self, other: LinearExpr) -> LinearExpr:
        return BinOpExpr("+", self, other)

    def __sub__(self, other: LinearExpr) -> LinearExpr:
        return BinOpExpr("-", self, other)

    def __mul__(self, c: int) -> LinearExpr:
        return ScaleExpr("*", self, c)

    def __floordiv__(self, c: int) -> LinearExpr:
        return ScaleExpr("//", self, c)

    def __mod__(self, c: int) -> LinearExpr:
        return ScaleExpr("%", self, c)


@dataclass(frozen=True)
class VarExpr(LinearExpr):
    """Variable reference: x."""
    name: str

    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.name})

    def substitute(self, var: str, expr: LinearExpr) -> LinearExpr:
        return expr if self.name == var else self

    def pretty(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VarExpr) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("VarExpr", self.name))


@dataclass(frozen=True)
class ConstExpr(LinearExpr):
    """Integer constant: c."""
    value: int

    def free_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, var: str, expr: LinearExpr) -> LinearExpr:
        return self

    def pretty(self) -> str:
        return str(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ConstExpr) and self.value == other.value

    def __hash__(self) -> int:
        return hash(("ConstExpr", self.value))


@dataclass(frozen=True)
class LenExpr(LinearExpr):
    """Length expression: len(x)."""
    arg: str

    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.arg})

    def substitute(self, var: str, expr: LinearExpr) -> LinearExpr:
        return self  # len(x) is atomic; var renaming handled at predicate level

    def pretty(self) -> str:
        return f"len({self.arg})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LenExpr) and self.arg == other.arg

    def __hash__(self) -> int:
        return hash(("LenExpr", self.arg))


@dataclass(frozen=True)
class BinOpExpr(LinearExpr):
    """Binary operation: e + e | e - e."""
    op: str  # "+" or "-"
    left: LinearExpr
    right: LinearExpr

    def free_vars(self) -> FrozenSet[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute(self, var: str, expr: LinearExpr) -> LinearExpr:
        return BinOpExpr(self.op, self.left.substitute(var, expr), self.right.substitute(var, expr))

    def pretty(self) -> str:
        return f"({self.left.pretty()} {self.op} {self.right.pretty()})"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, BinOpExpr) and self.op == other.op
                and self.left == other.left and self.right == other.right)

    def __hash__(self) -> int:
        return hash(("BinOpExpr", self.op, self.left, self.right))


@dataclass(frozen=True)
class ScaleExpr(LinearExpr):
    """Scaling / integer division / modulo: e * c | e // c | e % c."""
    op: str  # "*", "//", "%"
    expr: LinearExpr
    const: int

    def free_vars(self) -> FrozenSet[str]:
        return self.expr.free_vars()

    def substitute(self, var: str, subst: LinearExpr) -> LinearExpr:
        return ScaleExpr(self.op, self.expr.substitute(var, subst), self.const)

    def pretty(self) -> str:
        return f"({self.expr.pretty()} {self.op} {self.const})"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, ScaleExpr) and self.op == other.op
                and self.expr == other.expr and self.const == other.const)

    def __hash__(self) -> int:
        return hash(("ScaleExpr", self.op, self.expr, self.const))


# ---------------------------------------------------------------------------
# Predicate hierarchy  –  φ
# ---------------------------------------------------------------------------

class Predicate(ABC):
    """Abstract predicate in the refinement logic."""

    @abstractmethod
    def free_vars(self) -> FrozenSet[str]:
        ...

    @abstractmethod
    def substitute_var(self, old: str, new: str) -> Predicate:
        ...

    @abstractmethod
    def substitute_expr(self, var: str, expr: LinearExpr) -> Predicate:
        ...

    @abstractmethod
    def negate(self) -> Predicate:
        ...

    @abstractmethod
    def pretty(self) -> str:
        ...

    def __str__(self) -> str:
        return self.pretty()

    def __repr__(self) -> str:
        return self.pretty()

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...

    # Combinators
    def __and__(self, other: Predicate) -> Predicate:
        return AndPred(frozenset({self, other}))

    def __or__(self, other: Predicate) -> Predicate:
        return OrPred(frozenset({self, other}))

    def __invert__(self) -> Predicate:
        return self.negate()


# --- Atomic predicates -----------------------------------------------------

class AtomicPredicate(Predicate, ABC):
    """Marker for leaf-level predicates."""
    pass


@dataclass(frozen=True)
class TruePred(AtomicPredicate):
    """Trivially true predicate (⊤)."""
    def free_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute_var(self, old: str, new: str) -> Predicate:
        return self

    def substitute_expr(self, var: str, expr: LinearExpr) -> Predicate:
        return self

    def negate(self) -> Predicate:
        return FALSE_PRED

    def pretty(self) -> str:
        return "⊤"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TruePred)

    def __hash__(self) -> int:
        return hash("TruePred")


@dataclass(frozen=True)
class FalsePred(AtomicPredicate):
    """Trivially false predicate (⊥)."""
    def free_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute_var(self, old: str, new: str) -> Predicate:
        return self

    def substitute_expr(self, var: str, expr: LinearExpr) -> Predicate:
        return self

    def negate(self) -> Predicate:
        return TRUE_PRED

    def pretty(self) -> str:
        return "⊥"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FalsePred)

    def __hash__(self) -> int:
        return hash("FalsePred")


TRUE_PRED = TruePred()
FALSE_PRED = FalsePred()


@dataclass(frozen=True)
class ComparisonPred(AtomicPredicate):
    """e₁ ⊕ e₂  where ⊕ ∈ {==, !=, <, <=, >, >=}."""
    op: str
    left: LinearExpr
    right: LinearExpr

    _NEGATION = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}

    def free_vars(self) -> FrozenSet[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute_var(self, old: str, new: str) -> Predicate:
        return ComparisonPred(
            self.op,
            self.left.substitute(old, VarExpr(new)),
            self.right.substitute(old, VarExpr(new)),
        )

    def substitute_expr(self, var: str, expr: LinearExpr) -> Predicate:
        return ComparisonPred(
            self.op,
            self.left.substitute(var, expr),
            self.right.substitute(var, expr),
        )

    def negate(self) -> Predicate:
        return ComparisonPred(self._NEGATION[self.op], self.left, self.right)

    def pretty(self) -> str:
        return f"{self.left.pretty()} {self.op} {self.right.pretty()}"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, ComparisonPred) and self.op == other.op
                and self.left == other.left and self.right == other.right)

    def __hash__(self) -> int:
        return hash(("ComparisonPred", self.op, self.left, self.right))


@dataclass(frozen=True)
class TypeTagPred(AtomicPredicate):
    """tag(x) == T  — runtime type tag check."""
    var: str
    tag: str  # e.g. "int", "str", "list", "NoneType"

    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.var})

    def substitute_var(self, old: str, new: str) -> Predicate:
        return TypeTagPred(new if self.var == old else self.var, self.tag)

    def substitute_expr(self, var: str, expr: LinearExpr) -> Predicate:
        return self

    def negate(self) -> Predicate:
        return NotPred(self)

    def pretty(self) -> str:
        return f"tag({self.var}) == {self.tag}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TypeTagPred) and self.var == other.var and self.tag == other.tag

    def __hash__(self) -> int:
        return hash(("TypeTagPred", self.var, self.tag))


@dataclass(frozen=True)
class NullityPred(AtomicPredicate):
    """x is None  /  x is not None."""
    var: str
    is_null: bool

    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.var})

    def substitute_var(self, old: str, new: str) -> Predicate:
        return NullityPred(new if self.var == old else self.var, self.is_null)

    def substitute_expr(self, var: str, expr: LinearExpr) -> Predicate:
        return self

    def negate(self) -> Predicate:
        return NullityPred(self.var, not self.is_null)

    def pretty(self) -> str:
        return f"{self.var} is None" if self.is_null else f"{self.var} is not None"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, NullityPred)
                and self.var == other.var and self.is_null == other.is_null)

    def __hash__(self) -> int:
        return hash(("NullityPred", self.var, self.is_null))


@dataclass(frozen=True)
class TruthinessPred(AtomicPredicate):
    """truthy(x) / falsy(x) — Python truthiness."""
    var: str
    is_truthy: bool

    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.var})

    def substitute_var(self, old: str, new: str) -> Predicate:
        return TruthinessPred(new if self.var == old else self.var, self.is_truthy)

    def substitute_expr(self, var: str, expr: LinearExpr) -> Predicate:
        return self

    def negate(self) -> Predicate:
        return TruthinessPred(self.var, not self.is_truthy)

    def pretty(self) -> str:
        return f"truthy({self.var})" if self.is_truthy else f"falsy({self.var})"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, TruthinessPred)
                and self.var == other.var and self.is_truthy == other.is_truthy)

    def __hash__(self) -> int:
        return hash(("TruthinessPred", self.var, self.is_truthy))


@dataclass(frozen=True)
class HasAttrPred(AtomicPredicate):
    """hasattr(x, "k") — structural attribute presence."""
    var: str
    attr: str

    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.var})

    def substitute_var(self, old: str, new: str) -> Predicate:
        return HasAttrPred(new if self.var == old else self.var, self.attr)

    def substitute_expr(self, var: str, expr: LinearExpr) -> Predicate:
        return self

    def negate(self) -> Predicate:
        return NotPred(self)

    def pretty(self) -> str:
        return f'hasattr({self.var}, "{self.attr}")'

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, HasAttrPred)
                and self.var == other.var and self.attr == other.attr)

    def __hash__(self) -> int:
        return hash(("HasAttrPred", self.var, self.attr))


# --- Boolean combinations --------------------------------------------------

@dataclass(frozen=True)
class AndPred(Predicate):
    """Conjunction: φ₁ ∧ φ₂ ∧ …"""
    conjuncts: FrozenSet[Predicate]

    def free_vars(self) -> FrozenSet[str]:
        return frozenset().union(*(c.free_vars() for c in self.conjuncts))

    def substitute_var(self, old: str, new: str) -> Predicate:
        return AndPred(frozenset(c.substitute_var(old, new) for c in self.conjuncts))

    def substitute_expr(self, var: str, expr: LinearExpr) -> Predicate:
        return AndPred(frozenset(c.substitute_expr(var, expr) for c in self.conjuncts))

    def negate(self) -> Predicate:
        return OrPred(frozenset(c.negate() for c in self.conjuncts))

    def pretty(self) -> str:
        parts = sorted(c.pretty() for c in self.conjuncts)
        return " ∧ ".join(parts) if parts else "⊤"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AndPred) and self.conjuncts == other.conjuncts

    def __hash__(self) -> int:
        return hash(("AndPred", self.conjuncts))


@dataclass(frozen=True)
class OrPred(Predicate):
    """Disjunction: φ₁ ∨ φ₂ ∨ …"""
    disjuncts: FrozenSet[Predicate]

    def free_vars(self) -> FrozenSet[str]:
        return frozenset().union(*(d.free_vars() for d in self.disjuncts))

    def substitute_var(self, old: str, new: str) -> Predicate:
        return OrPred(frozenset(d.substitute_var(old, new) for d in self.disjuncts))

    def substitute_expr(self, var: str, expr: LinearExpr) -> Predicate:
        return OrPred(frozenset(d.substitute_expr(var, expr) for d in self.disjuncts))

    def negate(self) -> Predicate:
        return AndPred(frozenset(d.negate() for d in self.disjuncts))

    def pretty(self) -> str:
        parts = sorted(d.pretty() for d in self.disjuncts)
        return "(" + " ∨ ".join(parts) + ")" if parts else "⊥"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OrPred) and self.disjuncts == other.disjuncts

    def __hash__(self) -> int:
        return hash(("OrPred", self.disjuncts))


@dataclass(frozen=True)
class NotPred(Predicate):
    """Negation: ¬φ."""
    inner: Predicate

    def free_vars(self) -> FrozenSet[str]:
        return self.inner.free_vars()

    def substitute_var(self, old: str, new: str) -> Predicate:
        return NotPred(self.inner.substitute_var(old, new))

    def substitute_expr(self, var: str, expr: LinearExpr) -> Predicate:
        return NotPred(self.inner.substitute_expr(var, expr))

    def negate(self) -> Predicate:
        return self.inner

    def pretty(self) -> str:
        return f"¬({self.inner.pretty()})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NotPred) and self.inner == other.inner

    def __hash__(self) -> int:
        return hash(("NotPred", self.inner))


# ---------------------------------------------------------------------------
# Predicate smart constructors
# ---------------------------------------------------------------------------

def make_and(*preds: Predicate) -> Predicate:
    """Flatten nested conjunctions, absorb ⊤/⊥."""
    flat: set[Predicate] = set()
    for p in preds:
        if isinstance(p, FalsePred):
            return FALSE_PRED
        if isinstance(p, TruePred):
            continue
        if isinstance(p, AndPred):
            flat.update(p.conjuncts)
        else:
            flat.add(p)
    if not flat:
        return TRUE_PRED
    if len(flat) == 1:
        return next(iter(flat))
    return AndPred(frozenset(flat))


def make_or(*preds: Predicate) -> Predicate:
    """Flatten nested disjunctions, absorb ⊤/⊥."""
    flat: set[Predicate] = set()
    for p in preds:
        if isinstance(p, TruePred):
            return TRUE_PRED
        if isinstance(p, FalsePred):
            continue
        if isinstance(p, OrPred):
            flat.update(p.disjuncts)
        else:
            flat.add(p)
    if not flat:
        return FALSE_PRED
    if len(flat) == 1:
        return next(iter(flat))
    return OrPred(frozenset(flat))


def make_not(p: Predicate) -> Predicate:
    return p.negate()


# ---------------------------------------------------------------------------
# Refinement type  –  {x : τ | φ}
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefinementType:
    """Refinement type {x : τ | φ} with binder variable *binder*, base type *base*,
    and predicate *pred* that may mention the binder."""
    binder: str
    base: BaseType
    pred: Predicate

    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        return self.base.free_type_vars()

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        return self.base.free_row_vars()

    def free_pred_vars(self) -> FrozenSet[str]:
        """Free variables in the predicate *excluding* the binder."""
        return self.pred.free_vars() - {self.binder}

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> RefinementType:
        return RefinementType(self.binder, self.base.substitute_type_var(mapping), self.pred)

    def substitute_row_var(self, mapping: Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]) -> RefinementType:
        return RefinementType(self.binder, self.base.substitute_row_var(mapping), self.pred)

    def substitute_pred_var(self, old: str, new: str) -> RefinementType:
        if old == self.binder:
            return self  # shadowed
        return RefinementType(self.binder, self.base, self.pred.substitute_var(old, new))

    def alpha_rename(self, new_binder: str) -> RefinementType:
        """α-rename the binder to *new_binder*."""
        if new_binder == self.binder:
            return self
        return RefinementType(new_binder, self.base, self.pred.substitute_var(self.binder, new_binder))

    def with_pred(self, new_pred: Predicate) -> RefinementType:
        return RefinementType(self.binder, self.base, new_pred)

    def with_base(self, new_base: BaseType) -> RefinementType:
        return RefinementType(self.binder, new_base, self.pred)

    def is_trivial(self) -> bool:
        """True when the predicate is ⊤ (no refinement)."""
        return isinstance(self.pred, TruePred)

    def pretty(self) -> str:
        if self.is_trivial():
            return self.base.pretty()
        return f"{{{self.binder} : {self.base.pretty()} | {self.pred.pretty()}}}"

    def __str__(self) -> str:
        return self.pretty()

    def __repr__(self) -> str:
        return f"RefinementType({self.binder!r}, {self.base!r}, {self.pred!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RefinementType):
            return NotImplemented
        # α-equivalence: rename both binders to a canonical name
        if self.base != other.base:
            return False
        if self.binder == other.binder:
            return self.pred == other.pred
        fresh = _fresh_binder({self.binder, other.binder}
                              | self.pred.free_vars()
                              | other.pred.free_vars())
        return (self.pred.substitute_var(self.binder, fresh)
                == other.pred.substitute_var(other.binder, fresh))

    def __hash__(self) -> int:
        # Hash must be consistent with α-equivalence so we canonicalize
        canonical = self.alpha_rename("ν")
        return hash(("RefinementType", canonical.base, canonical.pred))


def _fresh_binder(avoid: Set[str], prefix: str = "ν") -> str:
    candidate = prefix
    i = 0
    while candidate in avoid:
        i += 1
        candidate = f"{prefix}{i}"
    return candidate


def trivial_refinement(base: BaseType, binder: str = "ν") -> RefinementType:
    """Helper: {ν : τ | ⊤}."""
    return RefinementType(binder, base, TRUE_PRED)


# ---------------------------------------------------------------------------
# Dependent function type  –  (x:{v:τ₁|φ₁}) → {v:τ₂|φ₂(x)}
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DependentParam:
    """A named parameter with a refinement type."""
    name: str
    type: RefinementType

    def pretty(self) -> str:
        return f"({self.name}: {self.type.pretty()})"

    def __str__(self) -> str:
        return self.pretty()


@dataclass(frozen=True)
class DependentFunctionType:
    """Dependent function type where the return refinement may mention parameters.

    (x₁:{v:τ₁|φ₁}, …, xₙ:{v:τₙ|φₙ}) → {v:τᵣ|φᵣ(x₁,…,xₙ)}
    """
    params: Tuple[DependentParam, ...]
    ret: RefinementType

    def param_names(self) -> Tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def free_type_vars(self) -> FrozenSet[TypeVariable]:
        s = self.ret.free_type_vars()
        for p in self.params:
            s = s | p.type.free_type_vars()
        return s

    def free_row_vars(self) -> FrozenSet[RowVariable]:
        s = self.ret.free_row_vars()
        for p in self.params:
            s = s | p.type.free_row_vars()
        return s

    def substitute_type_var(self, mapping: Mapping[TypeVariable, BaseType]) -> DependentFunctionType:
        return DependentFunctionType(
            tuple(DependentParam(p.name, p.type.substitute_type_var(mapping)) for p in self.params),
            self.ret.substitute_type_var(mapping),
        )

    def to_simple_function_type(self) -> FunctionType:
        """Erase refinements to get a simple arrow type."""
        return FunctionType(
            tuple(p.type.base for p in self.params),
            self.ret.base,
        )

    def pretty(self) -> str:
        params_str = ", ".join(p.pretty() for p in self.params)
        return f"({params_str}) → {self.ret.pretty()}"

    def __str__(self) -> str:
        return self.pretty()

    def __repr__(self) -> str:
        return f"DependentFunctionType({self.params!r}, {self.ret!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DependentFunctionType):
            return NotImplemented
        if len(self.params) != len(other.params):
            return False
        return (all(sp.type == op.type for sp, op in zip(self.params, other.params))
                and self.ret == other.ret)

    def __hash__(self) -> int:
        return hash(("DependentFunctionType",
                      tuple(p.type for p in self.params),
                      self.ret))


# ---------------------------------------------------------------------------
# Type scheme  –  ∀ α₁…αₙ ρ₁…ρₘ . τ
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TypeScheme:
    """Polymorphic type scheme:  ∀ α₁…αₙ ρ₁…ρₘ . body."""
    type_vars: FrozenSet[TypeVariable]
    row_vars: FrozenSet[RowVariable]
    body: RefinementType

    def instantiate(self, type_map: Optional[Mapping[TypeVariable, BaseType]] = None,
                    row_map: Optional[Mapping[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]]] = None) -> RefinementType:
        """Instantiate the scheme with concrete types / rows."""
        result = self.body
        if type_map:
            result = result.substitute_type_var(type_map)
        if row_map:
            result = result.substitute_row_var(row_map)
        return result

    def instantiate_fresh(self) -> Tuple[RefinementType, Dict[TypeVariable, TypeVariable], Dict[RowVariable, RowVariable]]:
        """Instantiate with fresh variables; returns the result and the mappings."""
        tv_map: Dict[TypeVariable, TypeVariable] = {}
        bt_map: Dict[TypeVariable, BaseType] = {}
        for tv in self.type_vars:
            fresh = fresh_type_var(tv.name)
            tv_map[tv] = fresh
            bt_map[tv] = TypeVarType(fresh)

        rv_map: Dict[RowVariable, RowVariable] = {}
        rr_map: Dict[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]] = {}
        for rv in self.row_vars:
            fresh = fresh_row_var(rv.name)
            rv_map[rv] = fresh
            rr_map[rv] = ({}, fresh)

        result = self.body
        if bt_map:
            result = result.substitute_type_var(bt_map)
        if rr_map:
            result = result.substitute_row_var(rr_map)
        return result, tv_map, rv_map

    def generalize_over(self, env_tvs: FrozenSet[TypeVariable],
                        env_rvs: FrozenSet[RowVariable]) -> TypeScheme:
        """Create a scheme by generalizing free vars not in the environment."""
        free_tvs = self.body.free_type_vars() - env_tvs
        free_rvs = self.body.free_row_vars() - env_rvs
        return TypeScheme(free_tvs | self.type_vars, free_rvs | self.row_vars, self.body)

    @property
    def is_monomorphic(self) -> bool:
        return not self.type_vars and not self.row_vars

    def pretty(self) -> str:
        if self.is_monomorphic:
            return self.body.pretty()
        quantified = []
        for tv in sorted(self.type_vars, key=lambda v: v.name):
            quantified.append(str(tv))
        for rv in sorted(self.row_vars, key=lambda v: v.name):
            quantified.append(str(rv))
        return f"∀ {' '.join(quantified)} . {self.body.pretty()}"

    def __str__(self) -> str:
        return self.pretty()

    def __repr__(self) -> str:
        return f"TypeScheme({self.type_vars!r}, {self.row_vars!r}, {self.body!r})"


def mono_scheme(rt: RefinementType) -> TypeScheme:
    """Wrap a monomorphic refinement type as a trivial scheme."""
    return TypeScheme(frozenset(), frozenset(), rt)


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

@dataclass
class Substitution:
    """Combined substitution for type variables, row variables, and predicate
    expression variables."""
    type_map: Dict[TypeVariable, BaseType] = field(default_factory=dict)
    row_map: Dict[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]] = field(default_factory=dict)
    expr_map: Dict[str, LinearExpr] = field(default_factory=dict)

    def apply_base(self, t: BaseType) -> BaseType:
        result = t
        if self.type_map:
            result = result.substitute_type_var(self.type_map)
        if self.row_map:
            result = result.substitute_row_var(self.row_map)
        return result

    def apply_pred(self, p: Predicate) -> Predicate:
        result = p
        for var, expr in self.expr_map.items():
            result = result.substitute_expr(var, expr)
        return result

    def apply_refinement(self, rt: RefinementType) -> RefinementType:
        base = self.apply_base(rt.base)
        pred = self.apply_pred(rt.pred)
        return RefinementType(rt.binder, base, pred)

    def apply_scheme(self, ts: TypeScheme) -> TypeScheme:
        # Only substitute free variables (not quantified ones)
        restricted_type = {k: v for k, v in self.type_map.items() if k not in ts.type_vars}
        restricted_row = {k: v for k, v in self.row_map.items() if k not in ts.row_vars}
        sub = Substitution(restricted_type, restricted_row, self.expr_map)
        return TypeScheme(ts.type_vars, ts.row_vars, sub.apply_refinement(ts.body))

    def compose(self, other: Substitution) -> Substitution:
        """Compose self ∘ other  (apply other first, then self)."""
        new_type: Dict[TypeVariable, BaseType] = {}
        for k, v in other.type_map.items():
            new_type[k] = self.apply_base(v)
        for k, v in self.type_map.items():
            if k not in new_type:
                new_type[k] = v

        new_row: Dict[RowVariable, Tuple[Dict[str, BaseType], Optional[RowVariable]]] = {}
        for k, v in other.row_map.items():
            fields, rv = v
            new_fields = {fk: self.apply_base(fv) for fk, fv in fields.items()}
            # If the resulting row var is itself substituted, chain
            if rv is not None and rv in self.row_map:
                extra, rv2 = self.row_map[rv]
                new_fields.update({fk: self.apply_base(fv) for fk, fv in extra.items()})
                rv = rv2
            new_row[k] = (new_fields, rv)
        for k, v in self.row_map.items():
            if k not in new_row:
                new_row[k] = v

        new_expr: Dict[str, LinearExpr] = {}
        for k, v in other.expr_map.items():
            new_expr[k] = v  # expression substitutions don't compose deeply here
        for k, v in self.expr_map.items():
            if k not in new_expr:
                new_expr[k] = v

        return Substitution(new_type, new_row, new_expr)

    @staticmethod
    def empty() -> Substitution:
        return Substitution()

    def __bool__(self) -> bool:
        return bool(self.type_map) or bool(self.row_map) or bool(self.expr_map)

    def pretty(self) -> str:
        parts: ListT[str] = []
        for tv, bt in self.type_map.items():
            parts.append(f"{tv} ↦ {bt}")
        for rv, (flds, rv2) in self.row_map.items():
            rhs_parts = [f"{k}: {v}" for k, v in flds.items()]
            if rv2:
                rhs_parts.append(str(rv2))
            parts.append(f"{rv} ↦ {{{', '.join(rhs_parts)}}}")
        for var, expr in self.expr_map.items():
            parts.append(f"{var} ↦ {expr}")
        return "[" + ", ".join(parts) + "]" if parts else "[]"

    def __str__(self) -> str:
        return self.pretty()


# ---------------------------------------------------------------------------
# Union type constructor / normalizer
# ---------------------------------------------------------------------------

def make_union(*types: BaseType) -> BaseType:
    """Build a normalized union, flattening nested unions and removing duplicates.

    - Union of one type is that type itself.
    - Never is the identity element.
    - Any absorbs everything.
    """
    flat: set[BaseType] = set()
    for t in types:
        if isinstance(t, AnyType):
            return AnyType()
        if isinstance(t, NeverType):
            continue
        if isinstance(t, UnionType):
            flat.update(t.alternatives)
        else:
            flat.add(t)
    if not flat:
        return NeverType()
    if len(flat) == 1:
        return next(iter(flat))
    return UnionType(frozenset(flat))


# ---------------------------------------------------------------------------
# Type normalization / canonicalization
# ---------------------------------------------------------------------------

def normalize_type(t: BaseType) -> BaseType:
    """Canonicalize a type: flatten unions, sort object fields, simplify."""
    if isinstance(t, (IntType, StrType, BoolType, NoneType_, AnyType, NeverType, TypeVarType)):
        return t

    if isinstance(t, ListType):
        return ListType(normalize_type(t.element))

    if isinstance(t, SetType):
        return SetType(normalize_type(t.element))

    if isinstance(t, TupleType):
        return TupleType(tuple(normalize_type(e) for e in t.elements))

    if isinstance(t, DictType):
        return DictType(normalize_type(t.key), normalize_type(t.value))

    if isinstance(t, FunctionType):
        return FunctionType(
            tuple(normalize_type(p) for p in t.params),
            normalize_type(t.ret),
        )

    if isinstance(t, ObjectType):
        normalized_fields = tuple(
            sorted(((k, normalize_type(v)) for k, v in t.fields), key=lambda kv: kv[0])
        )
        return ObjectType(normalized_fields, t.row_var)

    if isinstance(t, UnionType):
        return make_union(*(normalize_type(a) for a in t.alternatives))

    return t


def normalize_refinement(rt: RefinementType) -> RefinementType:
    """Normalize the base type and simplify trivial predicates."""
    base = normalize_type(rt.base)
    pred = _simplify_pred(rt.pred)
    return RefinementType(rt.binder, base, pred)


def _simplify_pred(p: Predicate) -> Predicate:
    """Basic predicate simplification."""
    if isinstance(p, (TruePred, FalsePred)):
        return p

    if isinstance(p, AndPred):
        simplified = frozenset(_simplify_pred(c) for c in p.conjuncts)
        return make_and(*simplified)

    if isinstance(p, OrPred):
        simplified = frozenset(_simplify_pred(d) for d in p.disjuncts)
        return make_or(*simplified)

    if isinstance(p, NotPred):
        inner = _simplify_pred(p.inner)
        if isinstance(inner, TruePred):
            return FALSE_PRED
        if isinstance(inner, FalsePred):
            return TRUE_PRED
        if isinstance(inner, NotPred):
            return inner.inner
        return NotPred(inner)

    return p


# ---------------------------------------------------------------------------
# Width subtyping checker (structural)
# ---------------------------------------------------------------------------

def is_subtype(t1: BaseType, t2: BaseType) -> bool:
    """Check structural (width) subtyping:  t1 <: t2.

    Rules:
    - Never <: τ  for all τ
    - τ <: Any    for all τ
    - τ <: τ      (reflexive on structurally equal types)
    - {k₁:τ₁,…,kₙ:τₙ,kₙ₊₁:τₙ₊₁,…} <: {k₁:σ₁,…,kₙ:σₙ,ρ}
      when τᵢ <: σᵢ  (width subtyping, open rows on supertype)
    - List[τ] <: List[σ]      when τ <: σ  (covariant)
    - (σ₁→τ₁) <: (σ₂→τ₂)    when σ₂<:σ₁ and τ₁<:τ₂  (contra/covariant)
    - τ₁ ∪ τ₂ <: σ            when τ₁ <: σ and τ₂ <: σ
    - τ <: σ₁ ∪ σ₂            when τ <: σ₁ or τ <: σ₂
    """
    # Bottom is subtype of everything
    if isinstance(t1, NeverType):
        return True
    # Everything is subtype of top
    if isinstance(t2, AnyType):
        return True
    # Reflexivity
    if t1 == t2:
        return True

    # Union on the left: each alternative must be a subtype
    if isinstance(t1, UnionType):
        return all(is_subtype(a, t2) for a in t1.alternatives)

    # Union on the right: at least one alternative must be a supertype
    if isinstance(t2, UnionType):
        return any(is_subtype(t1, a) for a in t2.alternatives)

    # List covariance
    if isinstance(t1, ListType) and isinstance(t2, ListType):
        return is_subtype(t1.element, t2.element)

    # Set covariance
    if isinstance(t1, SetType) and isinstance(t2, SetType):
        return is_subtype(t1.element, t2.element)

    # Tuple covariance (same length)
    if isinstance(t1, TupleType) and isinstance(t2, TupleType):
        if len(t1.elements) != len(t2.elements):
            return False
        return all(is_subtype(a, b) for a, b in zip(t1.elements, t2.elements))

    # Dict covariance on both key and value
    if isinstance(t1, DictType) and isinstance(t2, DictType):
        return is_subtype(t1.key, t2.key) and is_subtype(t1.value, t2.value)

    # Function: contravariant params, covariant return
    if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
        if len(t1.params) != len(t2.params):
            return False
        params_ok = all(is_subtype(p2, p1) for p1, p2 in zip(t1.params, t2.params))
        return params_ok and is_subtype(t1.ret, t2.ret)

    # Structural object width subtyping
    if isinstance(t1, ObjectType) and isinstance(t2, ObjectType):
        d1 = t1.field_dict
        d2 = t2.field_dict
        # t1 must have at least all fields of t2
        for k, v2 in d2.items():
            v1 = d1.get(k)
            if v1 is None:
                return False
            if not is_subtype(v1, v2):
                return False
        # If t2 is closed (no row var), t1 must not have extra fields (unless t1 is open)
        # Actually with width subtyping, more fields is finer, so t1 having extra is ok
        # But if t2 is closed, the subtype must match exactly on fields
        if t2.row_var is None and t1.row_var is None:
            if set(d1.keys()) != set(d2.keys()):
                return False
        return True

    return False


def is_refinement_subtype(rt1: RefinementType, rt2: RefinementType,
                          entailment_checker: Optional[Callable[[Predicate, Predicate], bool]] = None) -> bool:
    """Check Γ ⊢ {x:τ₁|φ₁} <: {x:τ₂|φ₂}  iff  τ₁ <: τ₂  and  φ₁ ⊨ φ₂.

    *entailment_checker* is a callback (assumption, goal) → bool.
    If not provided, only structural subtyping on bases is checked and
    predicate entailment is assumed when preds are syntactically equal or
    the supertype has a trivial predicate.
    """
    if not is_subtype(rt1.base, rt2.base):
        return False

    # Rename binders to match for predicate comparison
    binder = rt1.binder
    p1 = rt1.pred
    p2 = rt2.pred.substitute_var(rt2.binder, binder) if rt2.binder != binder else rt2.pred

    if isinstance(p2, TruePred):
        return True
    if p1 == p2:
        return True

    if entailment_checker is not None:
        return entailment_checker(p1, p2)

    return False


# ---------------------------------------------------------------------------
# Type constructors / factories
# ---------------------------------------------------------------------------

INT = IntType()
STR = StrType()
BOOL = BoolType()
NONE = NoneType_()
ANY = AnyType()
NEVER = NeverType()


def list_of(elem: BaseType) -> ListType:
    return ListType(elem)


def set_of(elem: BaseType) -> SetType:
    return SetType(elem)


def dict_of(key: BaseType, value: BaseType) -> DictType:
    return DictType(key, value)


def tuple_of(*elems: BaseType) -> TupleType:
    return TupleType(elems)


def func_type(*params: BaseType, ret: BaseType) -> FunctionType:
    return FunctionType(params, ret)


def object_type(fields: Dict[str, BaseType], *, open: bool = False) -> ObjectType:
    """Create an ObjectType.  If *open*, a fresh row variable is attached."""
    sorted_fields = tuple(sorted(fields.items(), key=lambda kv: kv[0]))
    rv = fresh_row_var() if open else None
    return ObjectType(sorted_fields, rv)


def refined(base: BaseType, binder: str, pred: Predicate) -> RefinementType:
    """Shorthand for RefinementType."""
    return RefinementType(binder, base, pred)


def dependent_func(params: Sequence[Tuple[str, RefinementType]],
                   ret: RefinementType) -> DependentFunctionType:
    return DependentFunctionType(
        tuple(DependentParam(name, rt) for name, rt in params),
        ret,
    )


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

def pretty_base(t: BaseType) -> str:
    return t.pretty()


def pretty_refinement(rt: RefinementType) -> str:
    return rt.pretty()


def pretty_scheme(ts: TypeScheme) -> str:
    return ts.pretty()


def pretty_pred(p: Predicate) -> str:
    return p.pretty()


def pretty_expr(e: LinearExpr) -> str:
    return e.pretty()


# ---------------------------------------------------------------------------
# Type tag ↔ base type mapping (for TypeTagPred ↔ runtime tags)
# ---------------------------------------------------------------------------

_TAG_TO_BASE: Dict[str, BaseType] = {
    "int": INT,
    "str": STR,
    "bool": BOOL,
    "NoneType": NONE,
    "list": list_of(ANY),
    "dict": dict_of(ANY, ANY),
    "set": set_of(ANY),
    "tuple": tuple_of(),
}

_BASE_TO_TAG: Dict[type, str] = {
    IntType: "int",
    StrType: "str",
    BoolType: "bool",
    NoneType_: "NoneType",
    ListType: "list",
    DictType: "dict",
    SetType: "set",
    TupleType: "tuple",
}


def tag_to_base_type(tag: str) -> Optional[BaseType]:
    """Map a runtime type tag string to a base type (or None)."""
    return _TAG_TO_BASE.get(tag)


def base_type_to_tag(t: BaseType) -> Optional[str]:
    """Map a base type to its runtime tag string (or None)."""
    return _BASE_TO_TAG.get(type(t))


# ---------------------------------------------------------------------------
# Occurrence-check and helpers for unification
# ---------------------------------------------------------------------------

def occurs_check(tv: TypeVariable, t: BaseType) -> bool:
    """Return True if *tv* occurs free in *t* (would cause infinite type)."""
    return tv in t.free_type_vars()


def collect_base_types(t: BaseType) -> set[type]:
    """Collect all concrete BaseType classes appearing in *t*."""
    result: set[type] = {type(t)}
    if isinstance(t, ListType):
        result |= collect_base_types(t.element)
    elif isinstance(t, SetType):
        result |= collect_base_types(t.element)
    elif isinstance(t, TupleType):
        for e in t.elements:
            result |= collect_base_types(e)
    elif isinstance(t, DictType):
        result |= collect_base_types(t.key) | collect_base_types(t.value)
    elif isinstance(t, FunctionType):
        for p in t.params:
            result |= collect_base_types(p)
        result |= collect_base_types(t.ret)
    elif isinstance(t, UnionType):
        for a in t.alternatives:
            result |= collect_base_types(a)
    elif isinstance(t, ObjectType):
        for _, v in t.fields:
            result |= collect_base_types(v)
    return result

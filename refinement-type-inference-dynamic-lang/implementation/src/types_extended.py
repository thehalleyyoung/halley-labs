from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# ===================================================================
# Constants & helpers
# ===================================================================

_SCHEMA_VERSION: int = 1
T = TypeVar("T")

# ===================================================================
# 1. Predicate language for refinement types
# ===================================================================


class PredicateKind(Enum):
    TRUE = auto()
    FALSE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    IS_NONE = auto()
    IS_NOT_NONE = auto()
    ISINSTANCE = auto()
    HASATTR = auto()
    IN = auto()
    NOT_IN = auto()
    LEN_EQ = auto()
    LEN_GT = auto()
    LEN_GE = auto()
    LEN_LT = auto()
    LEN_LE = auto()
    TRUTHY = auto()
    FALSY = auto()
    CUSTOM = auto()


@dataclass(frozen=True)
class PredicateVar:
    """A variable reference inside a predicate."""

    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class PredicateLiteral:
    """A literal value inside a predicate."""

    value: Any

    def __repr__(self) -> str:
        return repr(self.value)


PredicateAtom = Union[PredicateVar, PredicateLiteral]


@dataclass(frozen=True)
class Predicate:
    """A predicate φ used in refinement types {x: τ | φ}."""

    kind: PredicateKind
    args: Tuple[Any, ...] = ()
    children: Tuple[Predicate, ...] = ()

    # -- constructors --

    @staticmethod
    def true_() -> Predicate:
        return Predicate(PredicateKind.TRUE)

    @staticmethod
    def false_() -> Predicate:
        return Predicate(PredicateKind.FALSE)

    @staticmethod
    def and_(left: Predicate, right: Predicate) -> Predicate:
        if left.kind == PredicateKind.TRUE:
            return right
        if right.kind == PredicateKind.TRUE:
            return left
        if left.kind == PredicateKind.FALSE or right.kind == PredicateKind.FALSE:
            return Predicate.false_()
        return Predicate(PredicateKind.AND, children=(left, right))

    @staticmethod
    def or_(left: Predicate, right: Predicate) -> Predicate:
        if left.kind == PredicateKind.FALSE:
            return right
        if right.kind == PredicateKind.FALSE:
            return left
        if left.kind == PredicateKind.TRUE or right.kind == PredicateKind.TRUE:
            return Predicate.true_()
        return Predicate(PredicateKind.OR, children=(left, right))

    @staticmethod
    def not_(pred: Predicate) -> Predicate:
        if pred.kind == PredicateKind.TRUE:
            return Predicate.false_()
        if pred.kind == PredicateKind.FALSE:
            return Predicate.true_()
        if pred.kind == PredicateKind.NOT:
            return pred.children[0]
        return Predicate(PredicateKind.NOT, children=(pred,))

    @staticmethod
    def implies(left: Predicate, right: Predicate) -> Predicate:
        return Predicate(PredicateKind.IMPLIES, children=(left, right))

    @staticmethod
    def eq(var: str, value: Any) -> Predicate:
        return Predicate(PredicateKind.EQ, args=(PredicateVar(var), PredicateLiteral(value)))

    @staticmethod
    def neq(var: str, value: Any) -> Predicate:
        return Predicate(PredicateKind.NEQ, args=(PredicateVar(var), PredicateLiteral(value)))

    @staticmethod
    def lt(var: str, value: Any) -> Predicate:
        return Predicate(PredicateKind.LT, args=(PredicateVar(var), PredicateLiteral(value)))

    @staticmethod
    def le(var: str, value: Any) -> Predicate:
        return Predicate(PredicateKind.LE, args=(PredicateVar(var), PredicateLiteral(value)))

    @staticmethod
    def gt(var: str, value: Any) -> Predicate:
        return Predicate(PredicateKind.GT, args=(PredicateVar(var), PredicateLiteral(value)))

    @staticmethod
    def ge(var: str, value: Any) -> Predicate:
        return Predicate(PredicateKind.GE, args=(PredicateVar(var), PredicateLiteral(value)))

    @staticmethod
    def is_none(var: str) -> Predicate:
        return Predicate(PredicateKind.IS_NONE, args=(PredicateVar(var),))

    @staticmethod
    def is_not_none(var: str) -> Predicate:
        return Predicate(PredicateKind.IS_NOT_NONE, args=(PredicateVar(var),))

    @staticmethod
    def isinstance_(var: str, type_name: str) -> Predicate:
        return Predicate(
            PredicateKind.ISINSTANCE, args=(PredicateVar(var), PredicateLiteral(type_name))
        )

    @staticmethod
    def hasattr_(var: str, attr: str) -> Predicate:
        return Predicate(
            PredicateKind.HASATTR, args=(PredicateVar(var), PredicateLiteral(attr))
        )

    @staticmethod
    def in_(var: str, container: Any) -> Predicate:
        return Predicate(
            PredicateKind.IN, args=(PredicateVar(var), PredicateLiteral(container))
        )

    @staticmethod
    def len_eq(var: str, n: int) -> Predicate:
        return Predicate(PredicateKind.LEN_EQ, args=(PredicateVar(var), PredicateLiteral(n)))

    @staticmethod
    def len_gt(var: str, n: int) -> Predicate:
        return Predicate(PredicateKind.LEN_GT, args=(PredicateVar(var), PredicateLiteral(n)))

    @staticmethod
    def len_ge(var: str, n: int) -> Predicate:
        return Predicate(PredicateKind.LEN_GE, args=(PredicateVar(var), PredicateLiteral(n)))

    @staticmethod
    def len_lt(var: str, n: int) -> Predicate:
        return Predicate(PredicateKind.LEN_LT, args=(PredicateVar(var), PredicateLiteral(n)))

    @staticmethod
    def len_le(var: str, n: int) -> Predicate:
        return Predicate(PredicateKind.LEN_LE, args=(PredicateVar(var), PredicateLiteral(n)))

    @staticmethod
    def truthy(var: str) -> Predicate:
        return Predicate(PredicateKind.TRUTHY, args=(PredicateVar(var),))

    @staticmethod
    def falsy(var: str) -> Predicate:
        return Predicate(PredicateKind.FALSY, args=(PredicateVar(var),))

    @staticmethod
    def custom(description: str) -> Predicate:
        return Predicate(PredicateKind.CUSTOM, args=(description,))

    @staticmethod
    def between(var: str, lo: Any, hi: Any) -> Predicate:
        return Predicate.and_(Predicate.ge(var, lo), Predicate.le(var, hi))

    # -- analysis --

    def free_variables(self) -> Set[str]:
        result: Set[str] = set()
        for arg in self.args:
            if isinstance(arg, PredicateVar):
                result.add(arg.name)
        for child in self.children:
            result |= child.free_variables()
        return result

    def substitute(self, var: str, replacement: str) -> Predicate:
        new_args = tuple(
            PredicateVar(replacement) if isinstance(a, PredicateVar) and a.name == var else a
            for a in self.args
        )
        new_children = tuple(c.substitute(var, replacement) for c in self.children)
        return Predicate(self.kind, args=new_args, children=new_children)

    def negate(self) -> Predicate:
        return Predicate.not_(self)

    def complexity(self) -> int:
        """Count the number of nodes in the predicate AST."""
        return 1 + sum(c.complexity() for c in self.children)

    def is_trivially_true(self) -> bool:
        return self.kind == PredicateKind.TRUE

    def is_trivially_false(self) -> bool:
        return self.kind == PredicateKind.FALSE

    def __repr__(self) -> str:
        return _format_predicate(self)


def _format_predicate(p: Predicate) -> str:
    if p.kind == PredicateKind.TRUE:
        return "true"
    if p.kind == PredicateKind.FALSE:
        return "false"
    if p.kind == PredicateKind.AND:
        return f"({_format_predicate(p.children[0])} ∧ {_format_predicate(p.children[1])})"
    if p.kind == PredicateKind.OR:
        return f"({_format_predicate(p.children[0])} ∨ {_format_predicate(p.children[1])})"
    if p.kind == PredicateKind.NOT:
        return f"¬{_format_predicate(p.children[0])}"
    if p.kind == PredicateKind.IMPLIES:
        return f"({_format_predicate(p.children[0])} ⟹ {_format_predicate(p.children[1])})"
    if p.kind in (
        PredicateKind.EQ,
        PredicateKind.NEQ,
        PredicateKind.LT,
        PredicateKind.LE,
        PredicateKind.GT,
        PredicateKind.GE,
    ):
        ops = {
            PredicateKind.EQ: "=",
            PredicateKind.NEQ: "≠",
            PredicateKind.LT: "<",
            PredicateKind.LE: "≤",
            PredicateKind.GT: ">",
            PredicateKind.GE: "≥",
        }
        return f"{p.args[0]} {ops[p.kind]} {p.args[1]}"
    if p.kind == PredicateKind.IS_NONE:
        return f"{p.args[0]} is None"
    if p.kind == PredicateKind.IS_NOT_NONE:
        return f"{p.args[0]} is not None"
    if p.kind == PredicateKind.ISINSTANCE:
        return f"isinstance({p.args[0]}, {p.args[1]})"
    if p.kind == PredicateKind.HASATTR:
        return f"hasattr({p.args[0]}, {p.args[1]})"
    if p.kind == PredicateKind.IN:
        return f"{p.args[0]} in {p.args[1]}"
    if p.kind == PredicateKind.NOT_IN:
        return f"{p.args[0]} not in {p.args[1]}"
    if p.kind in (
        PredicateKind.LEN_EQ,
        PredicateKind.LEN_GT,
        PredicateKind.LEN_GE,
        PredicateKind.LEN_LT,
        PredicateKind.LEN_LE,
    ):
        len_ops = {
            PredicateKind.LEN_EQ: "=",
            PredicateKind.LEN_GT: ">",
            PredicateKind.LEN_GE: "≥",
            PredicateKind.LEN_LT: "<",
            PredicateKind.LEN_LE: "≤",
        }
        return f"len({p.args[0]}) {len_ops[p.kind]} {p.args[1]}"
    if p.kind == PredicateKind.TRUTHY:
        return f"bool({p.args[0]})"
    if p.kind == PredicateKind.FALSY:
        return f"not bool({p.args[0]})"
    if p.kind == PredicateKind.CUSTOM:
        return f"«{p.args[0]}»"
    return f"Predicate({p.kind})"


# ===================================================================
# 2. Variance for generics
# ===================================================================


class Variance(Enum):
    COVARIANT = auto()
    CONTRAVARIANT = auto()
    INVARIANT = auto()
    BIVARIANT = auto()


# ===================================================================
# 3. BaseType hierarchy
# ===================================================================


class BaseType(ABC):
    """Abstract root of the type hierarchy."""

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...

    def is_top(self) -> bool:
        return False

    def is_bottom(self) -> bool:
        return False

    def free_type_vars(self) -> Set[str]:
        return set()

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return self

    def contains_type(self, target: BaseType) -> bool:
        return self == target


@dataclass(frozen=True)
class TopType(BaseType):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, TopType)

    def __hash__(self) -> int:
        return hash("TopType")

    def __repr__(self) -> str:
        return "Any"

    def is_top(self) -> bool:
        return True


@dataclass(frozen=True)
class BottomType(BaseType):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, BottomType)

    def __hash__(self) -> int:
        return hash("BottomType")

    def __repr__(self) -> str:
        return "Never"

    def is_bottom(self) -> bool:
        return True


@dataclass(frozen=True)
class IntType(BaseType):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, IntType)

    def __hash__(self) -> int:
        return hash("IntType")

    def __repr__(self) -> str:
        return "int"


@dataclass(frozen=True)
class FloatType(BaseType):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, FloatType)

    def __hash__(self) -> int:
        return hash("FloatType")

    def __repr__(self) -> str:
        return "float"


@dataclass(frozen=True)
class ComplexType(BaseType):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, ComplexType)

    def __hash__(self) -> int:
        return hash("ComplexType")

    def __repr__(self) -> str:
        return "complex"


@dataclass(frozen=True)
class BoolType(BaseType):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, BoolType)

    def __hash__(self) -> int:
        return hash("BoolType")

    def __repr__(self) -> str:
        return "bool"


@dataclass(frozen=True)
class StrType(BaseType):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, StrType)

    def __hash__(self) -> int:
        return hash("StrType")

    def __repr__(self) -> str:
        return "str"


@dataclass(frozen=True)
class BytesType(BaseType):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, BytesType)

    def __hash__(self) -> int:
        return hash("BytesType")

    def __repr__(self) -> str:
        return "bytes"


@dataclass(frozen=True)
class NoneType_(BaseType):
    """None type (named NoneType_ to avoid clash with builtin)."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NoneType_)

    def __hash__(self) -> int:
        return hash("NoneType_")

    def __repr__(self) -> str:
        return "None"


@dataclass(frozen=True)
class ListType(BaseType):
    element_type: BaseType

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ListType) and self.element_type == other.element_type

    def __hash__(self) -> int:
        return hash(("ListType", self.element_type))

    def __repr__(self) -> str:
        return f"list[{self.element_type}]"

    def free_type_vars(self) -> Set[str]:
        return self.element_type.free_type_vars()

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return ListType(self.element_type.substitute_type_var(name, replacement))


@dataclass(frozen=True)
class TupleType(BaseType):
    element_types: Tuple[BaseType, ...]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TupleType) and self.element_types == other.element_types

    def __hash__(self) -> int:
        return hash(("TupleType", self.element_types))

    def __repr__(self) -> str:
        if not self.element_types:
            return "tuple[()]"
        elems = ", ".join(repr(t) for t in self.element_types)
        return f"tuple[{elems}]"

    def free_type_vars(self) -> Set[str]:
        result: Set[str] = set()
        for t in self.element_types:
            result |= t.free_type_vars()
        return result

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return TupleType(
            tuple(t.substitute_type_var(name, replacement) for t in self.element_types)
        )


@dataclass(frozen=True)
class SetType(BaseType):
    element_type: BaseType

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SetType) and self.element_type == other.element_type

    def __hash__(self) -> int:
        return hash(("SetType", self.element_type))

    def __repr__(self) -> str:
        return f"set[{self.element_type}]"

    def free_type_vars(self) -> Set[str]:
        return self.element_type.free_type_vars()

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return SetType(self.element_type.substitute_type_var(name, replacement))


@dataclass(frozen=True)
class FrozenSetType(BaseType):
    element_type: BaseType

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FrozenSetType) and self.element_type == other.element_type

    def __hash__(self) -> int:
        return hash(("FrozenSetType", self.element_type))

    def __repr__(self) -> str:
        return f"frozenset[{self.element_type}]"

    def free_type_vars(self) -> Set[str]:
        return self.element_type.free_type_vars()

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return FrozenSetType(self.element_type.substitute_type_var(name, replacement))


@dataclass(frozen=True)
class DictType(BaseType):
    key_type: BaseType
    value_type: BaseType

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DictType)
            and self.key_type == other.key_type
            and self.value_type == other.value_type
        )

    def __hash__(self) -> int:
        return hash(("DictType", self.key_type, self.value_type))

    def __repr__(self) -> str:
        return f"dict[{self.key_type}, {self.value_type}]"

    def free_type_vars(self) -> Set[str]:
        return self.key_type.free_type_vars() | self.value_type.free_type_vars()

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return DictType(
            self.key_type.substitute_type_var(name, replacement),
            self.value_type.substitute_type_var(name, replacement),
        )


@dataclass(frozen=True)
class CallableType(BaseType):
    param_types: Tuple[BaseType, ...]
    return_type: BaseType

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CallableType)
            and self.param_types == other.param_types
            and self.return_type == other.return_type
        )

    def __hash__(self) -> int:
        return hash(("CallableType", self.param_types, self.return_type))

    def __repr__(self) -> str:
        params = ", ".join(repr(t) for t in self.param_types)
        return f"Callable[[{params}], {self.return_type}]"

    def free_type_vars(self) -> Set[str]:
        result = self.return_type.free_type_vars()
        for pt in self.param_types:
            result |= pt.free_type_vars()
        return result

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return CallableType(
            tuple(t.substitute_type_var(name, replacement) for t in self.param_types),
            self.return_type.substitute_type_var(name, replacement),
        )


@dataclass(frozen=True)
class UnionType(BaseType):
    types: FrozenSet[BaseType]

    @staticmethod
    def of(*args: BaseType) -> BaseType:
        flattened: Set[BaseType] = set()
        for t in args:
            if isinstance(t, UnionType):
                flattened |= t.types
            elif isinstance(t, BottomType):
                continue
            else:
                flattened.add(t)
        if not flattened:
            return BottomType()
        if any(isinstance(t, TopType) for t in flattened):
            return TopType()
        if len(flattened) == 1:
            return next(iter(flattened))
        return UnionType(frozenset(flattened))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnionType) and self.types == other.types

    def __hash__(self) -> int:
        return hash(("UnionType", self.types))

    def __repr__(self) -> str:
        return " | ".join(sorted(repr(t) for t in self.types))

    def free_type_vars(self) -> Set[str]:
        result: Set[str] = set()
        for t in self.types:
            result |= t.free_type_vars()
        return result

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return UnionType.of(*(t.substitute_type_var(name, replacement) for t in self.types))


@dataclass(frozen=True)
class IntersectionType(BaseType):
    types: FrozenSet[BaseType]

    @staticmethod
    def of(*args: BaseType) -> BaseType:
        flattened: Set[BaseType] = set()
        for t in args:
            if isinstance(t, IntersectionType):
                flattened |= t.types
            elif isinstance(t, TopType):
                continue
            else:
                flattened.add(t)
        if not flattened:
            return TopType()
        if any(isinstance(t, BottomType) for t in flattened):
            return BottomType()
        if len(flattened) == 1:
            return next(iter(flattened))
        return IntersectionType(frozenset(flattened))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IntersectionType) and self.types == other.types

    def __hash__(self) -> int:
        return hash(("IntersectionType", self.types))

    def __repr__(self) -> str:
        return " & ".join(sorted(repr(t) for t in self.types))

    def free_type_vars(self) -> Set[str]:
        result: Set[str] = set()
        for t in self.types:
            result |= t.free_type_vars()
        return result

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return IntersectionType.of(
            *(t.substitute_type_var(name, replacement) for t in self.types)
        )


def OptionalType(inner: BaseType) -> BaseType:
    """Optional[T] = Union[T, None]."""
    return UnionType.of(inner, NoneType_())


@dataclass(frozen=True)
class StructuralType(BaseType):
    """Structural (duck) type with named fields and optional row variable."""

    fields: Tuple[Tuple[str, BaseType], ...]
    row_var: Optional[str] = None

    @staticmethod
    def create(fields: Dict[str, BaseType], row_var: Optional[str] = None) -> StructuralType:
        return StructuralType(
            fields=tuple(sorted(fields.items())), row_var=row_var
        )

    @property
    def field_dict(self) -> Dict[str, BaseType]:
        return dict(self.fields)

    def has_field(self, name: str) -> bool:
        return any(f[0] == name for f in self.fields)

    def get_field(self, name: str) -> Optional[BaseType]:
        for fname, ftype in self.fields:
            if fname == name:
                return ftype
        return None

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, StructuralType)
            and self.fields == other.fields
            and self.row_var == other.row_var
        )

    def __hash__(self) -> int:
        return hash(("StructuralType", self.fields, self.row_var))

    def __repr__(self) -> str:
        flds = ", ".join(f"{n}: {t}" for n, t in self.fields)
        rv = f", ...{self.row_var}" if self.row_var else ""
        return "{" + flds + rv + "}"

    def free_type_vars(self) -> Set[str]:
        result: Set[str] = set()
        if self.row_var:
            result.add(self.row_var)
        for _, ft in self.fields:
            result |= ft.free_type_vars()
        return result

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        new_fields = tuple(
            (fname, ft.substitute_type_var(name, replacement)) for fname, ft in self.fields
        )
        new_row = None if self.row_var == name else self.row_var
        return StructuralType(fields=new_fields, row_var=new_row)


@dataclass(frozen=True)
class ClassType(BaseType):
    """Nominal class type."""

    name: str
    bases: Tuple[BaseType, ...] = ()
    methods: Tuple[Tuple[str, BaseType], ...] = ()
    class_fields: Tuple[Tuple[str, BaseType], ...] = ()

    @staticmethod
    def create(
        name: str,
        bases: Optional[List[BaseType]] = None,
        methods: Optional[Dict[str, BaseType]] = None,
        fields: Optional[Dict[str, BaseType]] = None,
    ) -> ClassType:
        return ClassType(
            name=name,
            bases=tuple(bases or []),
            methods=tuple(sorted((methods or {}).items())),
            class_fields=tuple(sorted((fields or {}).items())),
        )

    @property
    def method_dict(self) -> Dict[str, BaseType]:
        return dict(self.methods)

    @property
    def field_dict(self) -> Dict[str, BaseType]:
        return dict(self.class_fields)

    def has_method(self, name: str) -> bool:
        return any(m[0] == name for m in self.methods)

    def get_method(self, name: str) -> Optional[BaseType]:
        for mname, mtype in self.methods:
            if mname == name:
                return mtype
        return None

    def has_field(self, name: str) -> bool:
        return any(f[0] == name for f in self.class_fields)

    def get_field(self, name: str) -> Optional[BaseType]:
        for fname, ftype in self.class_fields:
            if fname == name:
                return ftype
        return None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ClassType) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("ClassType", self.name))

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class GenericType(BaseType):
    """A generic type with type parameters, e.g. List[T]."""

    base: BaseType
    type_params: Tuple[BaseType, ...]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GenericType)
            and self.base == other.base
            and self.type_params == other.type_params
        )

    def __hash__(self) -> int:
        return hash(("GenericType", self.base, self.type_params))

    def __repr__(self) -> str:
        params = ", ".join(repr(t) for t in self.type_params)
        return f"{self.base}[{params}]"

    def free_type_vars(self) -> Set[str]:
        result = self.base.free_type_vars()
        for tp in self.type_params:
            result |= tp.free_type_vars()
        return result

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return GenericType(
            self.base.substitute_type_var(name, replacement),
            tuple(tp.substitute_type_var(name, replacement) for tp in self.type_params),
        )


@dataclass(frozen=True)
class TypeVarType(BaseType):
    """A type variable."""

    name: str
    bound: Optional[BaseType] = None
    constraints: Tuple[BaseType, ...] = ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TypeVarType) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("TypeVarType", self.name))

    def __repr__(self) -> str:
        return self.name

    def free_type_vars(self) -> Set[str]:
        result = {self.name}
        if self.bound:
            result |= self.bound.free_type_vars()
        for c in self.constraints:
            result |= c.free_type_vars()
        return result

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        if self.name == name:
            return replacement
        return self


@dataclass(frozen=True)
class LiteralType(BaseType):
    """A literal type, e.g. Literal[42], Literal["hello"]."""

    value: Any

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LiteralType) and self.value == other.value and type(self.value) is type(other.value)

    def __hash__(self) -> int:
        return hash(("LiteralType", self.value, type(self.value)))

    def __repr__(self) -> str:
        return f"Literal[{self.value!r}]"

    def base_of_literal(self) -> BaseType:
        if isinstance(self.value, bool):
            return BoolType()
        if isinstance(self.value, int):
            return IntType()
        if isinstance(self.value, float):
            return FloatType()
        if isinstance(self.value, str):
            return StrType()
        if isinstance(self.value, bytes):
            return BytesType()
        if self.value is None:
            return NoneType_()
        return TopType()


@dataclass(frozen=True)
class ProtocolType(BaseType):
    """A protocol (structural interface) type."""

    name: str
    methods: Tuple[Tuple[str, BaseType], ...]

    @staticmethod
    def create(name: str, methods: Dict[str, BaseType]) -> ProtocolType:
        return ProtocolType(name=name, methods=tuple(sorted(methods.items())))

    @property
    def method_dict(self) -> Dict[str, BaseType]:
        return dict(self.methods)

    def has_method(self, name: str) -> bool:
        return any(m[0] == name for m in self.methods)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ProtocolType) and self.methods == other.methods

    def __hash__(self) -> int:
        return hash(("ProtocolType", self.methods))

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TypeAliasType(BaseType):
    """A named type alias."""

    name: str
    target: BaseType

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TypeAliasType) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("TypeAliasType", self.name))

    def __repr__(self) -> str:
        return self.name

    def resolve(self) -> BaseType:
        t = self.target
        while isinstance(t, TypeAliasType):
            t = t.target
        return t

    def free_type_vars(self) -> Set[str]:
        return self.target.free_type_vars()

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return TypeAliasType(self.name, self.target.substitute_type_var(name, replacement))


@dataclass(frozen=True)
class RecursiveType(BaseType):
    """A recursive type: μα. body where α occurs in body."""

    var: str
    body: BaseType

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RecursiveType) and self.var == other.var and self.body == other.body

    def __hash__(self) -> int:
        return hash(("RecursiveType", self.var, self.body))

    def __repr__(self) -> str:
        return f"μ{self.var}. {self.body}"

    def unfold(self) -> BaseType:
        return self.body.substitute_type_var(self.var, self)

    def free_type_vars(self) -> Set[str]:
        return self.body.free_type_vars() - {self.var}

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        if name == self.var:
            return self
        return RecursiveType(self.var, self.body.substitute_type_var(name, replacement))


@dataclass(frozen=True)
class AwaitableType(BaseType):
    result_type: BaseType

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AwaitableType) and self.result_type == other.result_type

    def __hash__(self) -> int:
        return hash(("AwaitableType", self.result_type))

    def __repr__(self) -> str:
        return f"Awaitable[{self.result_type}]"

    def free_type_vars(self) -> Set[str]:
        return self.result_type.free_type_vars()

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return AwaitableType(self.result_type.substitute_type_var(name, replacement))


@dataclass(frozen=True)
class IterableType(BaseType):
    yield_type: BaseType

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IterableType) and self.yield_type == other.yield_type

    def __hash__(self) -> int:
        return hash(("IterableType", self.yield_type))

    def __repr__(self) -> str:
        return f"Iterable[{self.yield_type}]"

    def free_type_vars(self) -> Set[str]:
        return self.yield_type.free_type_vars()

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return IterableType(self.yield_type.substitute_type_var(name, replacement))


@dataclass(frozen=True)
class GeneratorType(BaseType):
    yield_type: BaseType
    send_type: BaseType
    return_type: BaseType

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GeneratorType)
            and self.yield_type == other.yield_type
            and self.send_type == other.send_type
            and self.return_type == other.return_type
        )

    def __hash__(self) -> int:
        return hash(("GeneratorType", self.yield_type, self.send_type, self.return_type))

    def __repr__(self) -> str:
        return f"Generator[{self.yield_type}, {self.send_type}, {self.return_type}]"

    def free_type_vars(self) -> Set[str]:
        return (
            self.yield_type.free_type_vars()
            | self.send_type.free_type_vars()
            | self.return_type.free_type_vars()
        )

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return GeneratorType(
            self.yield_type.substitute_type_var(name, replacement),
            self.send_type.substitute_type_var(name, replacement),
            self.return_type.substitute_type_var(name, replacement),
        )


@dataclass(frozen=True)
class CoroutineType(BaseType):
    yield_type: BaseType
    send_type: BaseType
    return_type: BaseType

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CoroutineType)
            and self.yield_type == other.yield_type
            and self.send_type == other.send_type
            and self.return_type == other.return_type
        )

    def __hash__(self) -> int:
        return hash(("CoroutineType", self.yield_type, self.send_type, self.return_type))

    def __repr__(self) -> str:
        return f"Coroutine[{self.yield_type}, {self.send_type}, {self.return_type}]"

    def free_type_vars(self) -> Set[str]:
        return (
            self.yield_type.free_type_vars()
            | self.send_type.free_type_vars()
            | self.return_type.free_type_vars()
        )

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return CoroutineType(
            self.yield_type.substitute_type_var(name, replacement),
            self.send_type.substitute_type_var(name, replacement),
            self.return_type.substitute_type_var(name, replacement),
        )


@dataclass(frozen=True)
class ContextManagerType(BaseType):
    enter_type: BaseType
    exit_type: BaseType

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ContextManagerType)
            and self.enter_type == other.enter_type
            and self.exit_type == other.exit_type
        )

    def __hash__(self) -> int:
        return hash(("ContextManagerType", self.enter_type, self.exit_type))

    def __repr__(self) -> str:
        return f"ContextManager[{self.enter_type}, {self.exit_type}]"

    def free_type_vars(self) -> Set[str]:
        return self.enter_type.free_type_vars() | self.exit_type.free_type_vars()

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return ContextManagerType(
            self.enter_type.substitute_type_var(name, replacement),
            self.exit_type.substitute_type_var(name, replacement),
        )


# ===================================================================
# 4. RefinementType
# ===================================================================


@dataclass(frozen=True)
class RefinementType(BaseType):
    """{x: τ | φ} — a base type refined by a predicate."""

    base_type: BaseType
    variable: str
    predicate: Predicate

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RefinementType)
            and self.base_type == other.base_type
            and self.predicate == other.predicate
        )

    def __hash__(self) -> int:
        return hash(("RefinementType", self.base_type, self.predicate))

    def __repr__(self) -> str:
        return f"{{{self.variable}: {self.base_type} | {self.predicate}}}"

    def free_type_vars(self) -> Set[str]:
        return self.base_type.free_type_vars()

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        return RefinementType(
            self.base_type.substitute_type_var(name, replacement),
            self.variable,
            self.predicate,
        )

    # -- convenient constructors --

    @staticmethod
    def positive_int() -> RefinementType:
        return RefinementType(IntType(), "x", Predicate.gt("x", 0))

    @staticmethod
    def non_negative_int() -> RefinementType:
        return RefinementType(IntType(), "x", Predicate.ge("x", 0))

    @staticmethod
    def bounded_int(lo: int, hi: int) -> RefinementType:
        return RefinementType(IntType(), "x", Predicate.between("x", lo, hi))

    @staticmethod
    def non_none(ty: BaseType) -> RefinementType:
        return RefinementType(ty, "x", Predicate.is_not_none("x"))

    @staticmethod
    def non_empty_list(elem_ty: BaseType) -> RefinementType:
        return RefinementType(ListType(elem_ty), "x", Predicate.len_gt("x", 0))

    @staticmethod
    def bounded_list(elem_ty: BaseType, max_len: int) -> RefinementType:
        return RefinementType(ListType(elem_ty), "x", Predicate.len_le("x", max_len))

    @staticmethod
    def instance_of(tag: str) -> RefinementType:
        return RefinementType(TopType(), "x", Predicate.isinstance_("x", tag))

    @staticmethod
    def has_attribute(attr: str) -> RefinementType:
        return RefinementType(TopType(), "x", Predicate.hasattr_("x", attr))

    @staticmethod
    def with_predicate(ty: BaseType, var: str, pred: Predicate) -> RefinementType:
        return RefinementType(ty, var, pred)

    def strengthen(self, extra: Predicate) -> RefinementType:
        return RefinementType(
            self.base_type, self.variable, Predicate.and_(self.predicate, extra)
        )

    def weaken_to_base(self) -> BaseType:
        return self.base_type


# ===================================================================
# 5. DependentFunctionType
# ===================================================================


@dataclass(frozen=True)
class DependentParam:
    """A parameter with an optional refinement."""

    name: str
    type: BaseType
    predicate: Optional[Predicate] = None

    def as_refinement(self) -> BaseType:
        if self.predicate is not None:
            return RefinementType(self.type, self.name, self.predicate)
        return self.type

    def __repr__(self) -> str:
        if self.predicate is not None:
            return f"{self.name}: {{{self.name}: {self.type} | {self.predicate}}}"
        return f"{self.name}: {self.type}"


@dataclass(frozen=True)
class DependentFunctionType(BaseType):
    """(x₁: {v: τ₁ | φ₁}, …) → {v: τ_ret | φ_ret(x₁,…)}"""

    params: Tuple[DependentParam, ...]
    return_type: BaseType
    return_predicate: Optional[Predicate] = None

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DependentFunctionType)
            and self.params == other.params
            and self.return_type == other.return_type
            and self.return_predicate == other.return_predicate
        )

    def __hash__(self) -> int:
        return hash(("DependentFunctionType", self.params, self.return_type, self.return_predicate))

    def __repr__(self) -> str:
        ps = ", ".join(repr(p) for p in self.params)
        ret = repr(self.return_type)
        if self.return_predicate:
            ret = f"{{v: {self.return_type} | {self.return_predicate}}}"
        return f"({ps}) → {ret}"

    def return_as_refinement(self) -> BaseType:
        if self.return_predicate is not None:
            return RefinementType(self.return_type, "v", self.return_predicate)
        return self.return_type

    def to_callable_type(self) -> CallableType:
        return CallableType(
            tuple(p.type for p in self.params),
            self.return_type,
        )

    def free_type_vars(self) -> Set[str]:
        result = self.return_type.free_type_vars()
        for p in self.params:
            result |= p.type.free_type_vars()
        return result

    def substitute_type_var(self, name: str, replacement: BaseType) -> BaseType:
        new_params = tuple(
            DependentParam(p.name, p.type.substitute_type_var(name, replacement), p.predicate)
            for p in self.params
        )
        return DependentFunctionType(
            new_params,
            self.return_type.substitute_type_var(name, replacement),
            self.return_predicate,
        )


# ===================================================================
# 6. TypeEnvironment
# ===================================================================


class TypeEnvironment:
    """Maps variables to types; supports join/meet/widen for merge points."""

    def __init__(self, bindings: Optional[Dict[str, BaseType]] = None) -> None:
        self._bindings: Dict[str, BaseType] = dict(bindings or {})

    def copy(self) -> TypeEnvironment:
        return TypeEnvironment(dict(self._bindings))

    def bind(self, var: str, ty: BaseType) -> TypeEnvironment:
        env = self.copy()
        env._bindings[var] = ty
        return env

    def lookup(self, var: str) -> Optional[BaseType]:
        return self._bindings.get(var)

    def lookup_or_top(self, var: str) -> BaseType:
        return self._bindings.get(var, TopType())

    def extend(self, bindings: Dict[str, BaseType]) -> TypeEnvironment:
        env = self.copy()
        env._bindings.update(bindings)
        return env

    def restrict(self, vars_: Set[str]) -> TypeEnvironment:
        return TypeEnvironment({k: v for k, v in self._bindings.items() if k in vars_})

    def remove(self, var: str) -> TypeEnvironment:
        env = self.copy()
        env._bindings.pop(var, None)
        return env

    def variables(self) -> Set[str]:
        return set(self._bindings.keys())

    def items(self) -> Iterable[Tuple[str, BaseType]]:
        return self._bindings.items()

    def __contains__(self, var: str) -> bool:
        return var in self._bindings

    def __len__(self) -> int:
        return len(self._bindings)

    def join(self, other: TypeEnvironment) -> TypeEnvironment:
        """Compute the join (LUB) of two environments at a merge point."""
        all_vars = self.variables() | other.variables()
        result: Dict[str, BaseType] = {}
        joiner = TypeJoin()
        for v in all_vars:
            lhs = self.lookup(v)
            rhs = other.lookup(v)
            if lhs is None:
                result[v] = rhs if rhs is not None else TopType()
            elif rhs is None:
                result[v] = lhs
            else:
                result[v] = joiner.join(lhs, rhs)
        return TypeEnvironment(result)

    def meet(self, other: TypeEnvironment) -> TypeEnvironment:
        """Compute the meet (GLB) of two environments (narrowing)."""
        all_vars = self.variables() | other.variables()
        result: Dict[str, BaseType] = {}
        meeter = TypeMeet()
        for v in all_vars:
            lhs = self.lookup(v)
            rhs = other.lookup(v)
            if lhs is None or rhs is None:
                ty = lhs if lhs is not None else rhs
                if ty is not None:
                    result[v] = ty
            else:
                result[v] = meeter.meet(lhs, rhs)
        return TypeEnvironment(result)

    def widen(self, other: TypeEnvironment) -> TypeEnvironment:
        """Widening: drop refinements that differ between self and other."""
        all_vars = self.variables() | other.variables()
        result: Dict[str, BaseType] = {}
        joiner = TypeJoin()
        for v in all_vars:
            lhs = self.lookup(v)
            rhs = other.lookup(v)
            if lhs is None:
                result[v] = rhs if rhs is not None else TopType()
            elif rhs is None:
                result[v] = lhs
            elif lhs == rhs:
                result[v] = lhs
            else:
                joined = joiner.join(lhs, rhs)
                if isinstance(joined, RefinementType):
                    result[v] = joined.base_type
                else:
                    result[v] = joined
        return TypeEnvironment(result)

    def apply_guard(self, guard: Guard) -> TypeEnvironment:
        """Narrow types based on a guard condition."""
        narrower = TypeNarrower()
        return narrower.narrow_compound(self, guard)

    def is_subenv_of(self, other: TypeEnvironment) -> bool:
        checker = SubtypeChecker()
        for v, ty in self._bindings.items():
            other_ty = other.lookup(v)
            if other_ty is None:
                continue
            if not checker.is_subtype(self, ty, other_ty):
                return False
        return True

    def __repr__(self) -> str:
        if not self._bindings:
            return "Γ{}"
        entries = ", ".join(f"{k}: {v}" for k, v in sorted(self._bindings.items()))
        return f"Γ{{{entries}}}"


# ===================================================================
# 7. Guard representation for type narrowing
# ===================================================================


class GuardKind(Enum):
    ISINSTANCE = auto()
    IS_NONE = auto()
    IS_NOT_NONE = auto()
    HASATTR = auto()
    COMPARISON = auto()
    TRUTHY = auto()
    FALSY = auto()
    TYPEOF = auto()
    IN = auto()
    EQUALITY = auto()
    AND = auto()
    OR = auto()
    NOT = auto()


@dataclass(frozen=True)
class Guard:
    kind: GuardKind
    variable: str = ""
    args: Tuple[Any, ...] = ()
    children: Tuple[Guard, ...] = ()

    @staticmethod
    def isinstance_(var: str, type_name: str) -> Guard:
        return Guard(GuardKind.ISINSTANCE, variable=var, args=(type_name,))

    @staticmethod
    def is_none(var: str) -> Guard:
        return Guard(GuardKind.IS_NONE, variable=var)

    @staticmethod
    def is_not_none(var: str) -> Guard:
        return Guard(GuardKind.IS_NOT_NONE, variable=var)

    @staticmethod
    def hasattr_(var: str, attr: str) -> Guard:
        return Guard(GuardKind.HASATTR, variable=var, args=(attr,))

    @staticmethod
    def comparison(var: str, op: str, value: Any) -> Guard:
        return Guard(GuardKind.COMPARISON, variable=var, args=(op, value))

    @staticmethod
    def truthy(var: str) -> Guard:
        return Guard(GuardKind.TRUTHY, variable=var)

    @staticmethod
    def falsy(var: str) -> Guard:
        return Guard(GuardKind.FALSY, variable=var)

    @staticmethod
    def typeof(var: str, type_string: str) -> Guard:
        return Guard(GuardKind.TYPEOF, variable=var, args=(type_string,))

    @staticmethod
    def in_(var: str, container: Any) -> Guard:
        return Guard(GuardKind.IN, variable=var, args=(container,))

    @staticmethod
    def equality(var: str, value: Any) -> Guard:
        return Guard(GuardKind.EQUALITY, variable=var, args=(value,))

    @staticmethod
    def and_(left: Guard, right: Guard) -> Guard:
        return Guard(GuardKind.AND, children=(left, right))

    @staticmethod
    def or_(left: Guard, right: Guard) -> Guard:
        return Guard(GuardKind.OR, children=(left, right))

    @staticmethod
    def not_(inner: Guard) -> Guard:
        return Guard(GuardKind.NOT, children=(inner,))


# ===================================================================
# 8. SubtypeChecker
# ===================================================================


# Mapping from type name string to BaseType constructor
_BUILTIN_TYPE_MAP: Dict[str, BaseType] = {
    "int": IntType(),
    "float": FloatType(),
    "complex": ComplexType(),
    "bool": BoolType(),
    "str": StrType(),
    "bytes": BytesType(),
    "None": NoneType_(),
    "NoneType": NoneType_(),
}

# Numeric tower
_NUMERIC_HIERARCHY: List[type] = [BoolType, IntType, FloatType, ComplexType]


class SubtypeChecker:
    """Check subtype relationships between types."""

    def __init__(self, max_depth: int = 50) -> None:
        self._max_depth = max_depth
        self._assumptions: Set[Tuple[int, int]] = set()

    def is_subtype(self, env: TypeEnvironment, ty1: BaseType, ty2: BaseType) -> bool:
        return self._is_subtype(ty1, ty2, depth=0)

    def _is_subtype(self, ty1: BaseType, ty2: BaseType, depth: int) -> bool:
        if depth > self._max_depth:
            return True

        if ty1 == ty2:
            return True

        # Bottom is subtype of everything
        if isinstance(ty1, BottomType):
            return True

        # Everything is subtype of Top
        if isinstance(ty2, TopType):
            return True

        # Top is only subtype of Top
        if isinstance(ty1, TopType):
            return isinstance(ty2, TopType)

        # Coinductive assumption for recursive types
        pair = (id(ty1), id(ty2))
        if pair in self._assumptions:
            return True
        self._assumptions.add(pair)
        try:
            result = self._is_subtype_inner(ty1, ty2, depth)
        finally:
            self._assumptions.discard(pair)
        return result

    def _is_subtype_inner(self, ty1: BaseType, ty2: BaseType, depth: int) -> bool:
        # Resolve aliases
        if isinstance(ty1, TypeAliasType):
            return self._is_subtype(ty1.resolve(), ty2, depth + 1)
        if isinstance(ty2, TypeAliasType):
            return self._is_subtype(ty1, ty2.resolve(), depth + 1)

        # Unfold recursive types
        if isinstance(ty1, RecursiveType):
            return self._is_subtype(ty1.unfold(), ty2, depth + 1)
        if isinstance(ty2, RecursiveType):
            return self._is_subtype(ty1, ty2.unfold(), depth + 1)

        # Literal ⊆ base
        if isinstance(ty1, LiteralType):
            return self._is_subtype(ty1.base_of_literal(), ty2, depth + 1)

        # Numeric tower: bool ⊆ int ⊆ float ⊆ complex
        if self._check_numeric_subtype(ty1, ty2):
            return True

        # Union subtyping: S₁ | S₂ ⊆ T iff S₁ ⊆ T and S₂ ⊆ T
        if isinstance(ty1, UnionType):
            return all(self._is_subtype(t, ty2, depth + 1) for t in ty1.types)

        # T ⊆ S₁ | S₂ iff T ⊆ S₁ or T ⊆ S₂
        if isinstance(ty2, UnionType):
            return any(self._is_subtype(ty1, t, depth + 1) for t in ty2.types)

        # Intersection subtyping: T ⊆ S₁ & S₂ iff T ⊆ S₁ and T ⊆ S₂
        if isinstance(ty2, IntersectionType):
            return all(self._is_subtype(ty1, t, depth + 1) for t in ty2.types)

        if isinstance(ty1, IntersectionType):
            return any(self._is_subtype(t, ty2, depth + 1) for t in ty1.types)

        # Refinement subtyping
        if isinstance(ty1, RefinementType) and isinstance(ty2, RefinementType):
            if not self._is_subtype(ty1.base_type, ty2.base_type, depth + 1):
                return False
            # Semantic: Γ, φ₁ ⊨ φ₂ — approximate by checking triviality
            if ty2.predicate.is_trivially_true():
                return True
            if ty1.predicate == ty2.predicate:
                return True
            return self._check_predicate_implication(ty1.predicate, ty2.predicate)

        if isinstance(ty1, RefinementType):
            return self._is_subtype(ty1.base_type, ty2, depth + 1)

        if isinstance(ty2, RefinementType):
            if not self._is_subtype(ty1, ty2.base_type, depth + 1):
                return False
            return ty2.predicate.is_trivially_true()

        # Callable subtyping (contravariant params, covariant return)
        if isinstance(ty1, CallableType) and isinstance(ty2, CallableType):
            return self._callable_subtype(ty1, ty2, depth)

        # DependentFunctionType → CallableType
        if isinstance(ty1, DependentFunctionType) and isinstance(ty2, CallableType):
            return self._callable_subtype(ty1.to_callable_type(), ty2, depth)
        if isinstance(ty1, CallableType) and isinstance(ty2, DependentFunctionType):
            return self._callable_subtype(ty1, ty2.to_callable_type(), depth)
        if isinstance(ty1, DependentFunctionType) and isinstance(ty2, DependentFunctionType):
            return self._callable_subtype(ty1.to_callable_type(), ty2.to_callable_type(), depth)

        # List (covariant for immutable reading, but Python lists are mutable → invariant)
        if isinstance(ty1, ListType) and isinstance(ty2, ListType):
            return self._is_subtype(ty1.element_type, ty2.element_type, depth + 1)

        # Tuple subtyping (covariant, element-wise)
        if isinstance(ty1, TupleType) and isinstance(ty2, TupleType):
            if len(ty1.element_types) != len(ty2.element_types):
                return False
            return all(
                self._is_subtype(t1, t2, depth + 1)
                for t1, t2 in zip(ty1.element_types, ty2.element_types)
            )

        # Set/FrozenSet
        if isinstance(ty1, SetType) and isinstance(ty2, SetType):
            return self._is_subtype(ty1.element_type, ty2.element_type, depth + 1)
        if isinstance(ty1, FrozenSetType) and isinstance(ty2, FrozenSetType):
            return self._is_subtype(ty1.element_type, ty2.element_type, depth + 1)

        # Dict (invariant for keys, covariant for values approximately)
        if isinstance(ty1, DictType) and isinstance(ty2, DictType):
            return self._is_subtype(ty1.key_type, ty2.key_type, depth + 1) and self._is_subtype(
                ty1.value_type, ty2.value_type, depth + 1
            )

        # Structural subtyping (width + depth)
        if isinstance(ty1, StructuralType) and isinstance(ty2, StructuralType):
            return self._structural_subtype(ty1, ty2, depth)

        # Class subtyping (nominal + structural check against bases)
        if isinstance(ty1, ClassType) and isinstance(ty2, ClassType):
            return self._class_subtype(ty1, ty2, depth)

        # Protocol subtyping
        if isinstance(ty2, ProtocolType):
            return self._satisfies_protocol(ty1, ty2, depth)

        # Generic subtyping
        if isinstance(ty1, GenericType) and isinstance(ty2, GenericType):
            return self._generic_subtype(ty1, ty2, depth)

        # Awaitable
        if isinstance(ty1, AwaitableType) and isinstance(ty2, AwaitableType):
            return self._is_subtype(ty1.result_type, ty2.result_type, depth + 1)

        # Iterable
        if isinstance(ty1, IterableType) and isinstance(ty2, IterableType):
            return self._is_subtype(ty1.yield_type, ty2.yield_type, depth + 1)

        # List ⊆ Iterable
        if isinstance(ty1, ListType) and isinstance(ty2, IterableType):
            return self._is_subtype(ty1.element_type, ty2.yield_type, depth + 1)

        # Generator
        if isinstance(ty1, GeneratorType) and isinstance(ty2, GeneratorType):
            return (
                self._is_subtype(ty1.yield_type, ty2.yield_type, depth + 1)
                and self._is_subtype(ty2.send_type, ty1.send_type, depth + 1)
                and self._is_subtype(ty1.return_type, ty2.return_type, depth + 1)
            )

        # Generator ⊆ Iterable
        if isinstance(ty1, GeneratorType) and isinstance(ty2, IterableType):
            return self._is_subtype(ty1.yield_type, ty2.yield_type, depth + 1)

        # Coroutine
        if isinstance(ty1, CoroutineType) and isinstance(ty2, CoroutineType):
            return (
                self._is_subtype(ty1.yield_type, ty2.yield_type, depth + 1)
                and self._is_subtype(ty2.send_type, ty1.send_type, depth + 1)
                and self._is_subtype(ty1.return_type, ty2.return_type, depth + 1)
            )

        # Coroutine ⊆ Awaitable
        if isinstance(ty1, CoroutineType) and isinstance(ty2, AwaitableType):
            return self._is_subtype(ty1.return_type, ty2.result_type, depth + 1)

        # ContextManager
        if isinstance(ty1, ContextManagerType) and isinstance(ty2, ContextManagerType):
            return self._is_subtype(ty1.enter_type, ty2.enter_type, depth + 1) and self._is_subtype(
                ty1.exit_type, ty2.exit_type, depth + 1
            )

        # TypeVar
        if isinstance(ty1, TypeVarType):
            if ty1.bound:
                return self._is_subtype(ty1.bound, ty2, depth + 1)
            if ty1.constraints:
                return any(self._is_subtype(c, ty2, depth + 1) for c in ty1.constraints)
        if isinstance(ty2, TypeVarType):
            if ty2.bound:
                return self._is_subtype(ty1, ty2.bound, depth + 1)

        return False

    def _check_numeric_subtype(self, ty1: BaseType, ty2: BaseType) -> bool:
        idx1 = self._numeric_index(ty1)
        idx2 = self._numeric_index(ty2)
        if idx1 >= 0 and idx2 >= 0:
            return idx1 <= idx2
        return False

    def _numeric_index(self, ty: BaseType) -> int:
        for i, cls in enumerate(_NUMERIC_HIERARCHY):
            if isinstance(ty, cls):
                return i
        return -1

    def _callable_subtype(self, ty1: CallableType, ty2: CallableType, depth: int) -> bool:
        if len(ty1.param_types) != len(ty2.param_types):
            return False
        # Contravariant parameters
        for p1, p2 in zip(ty1.param_types, ty2.param_types):
            if not self._is_subtype(p2, p1, depth + 1):
                return False
        # Covariant return
        return self._is_subtype(ty1.return_type, ty2.return_type, depth + 1)

    def _structural_subtype(
        self, ty1: StructuralType, ty2: StructuralType, depth: int
    ) -> bool:
        """Width + depth subtyping for structural types."""
        fields2 = ty2.field_dict
        fields1 = ty1.field_dict
        for fname, ftype2 in fields2.items():
            ftype1 = fields1.get(fname)
            if ftype1 is None:
                if ty1.row_var is not None:
                    continue
                return False
            if not self._is_subtype(ftype1, ftype2, depth + 1):
                return False
        return True

    def _class_subtype(self, ty1: ClassType, ty2: ClassType, depth: int) -> bool:
        if ty1.name == ty2.name:
            return True
        for base in ty1.bases:
            if self._is_subtype(base, ty2, depth + 1):
                return True
        return False

    def _satisfies_protocol(self, ty: BaseType, proto: ProtocolType, depth: int) -> bool:
        for method_name, method_type in proto.methods:
            found = False
            if isinstance(ty, ClassType):
                mt = ty.get_method(method_name)
                if mt is not None and self._is_subtype(mt, method_type, depth + 1):
                    found = True
            elif isinstance(ty, StructuralType):
                ft = ty.get_field(method_name)
                if ft is not None and self._is_subtype(ft, method_type, depth + 1):
                    found = True
            if not found:
                return False
        return True

    def _generic_subtype(self, ty1: GenericType, ty2: GenericType, depth: int) -> bool:
        if not self._is_subtype(ty1.base, ty2.base, depth + 1):
            return False
        if len(ty1.type_params) != len(ty2.type_params):
            return False
        # Default: invariant check
        return all(
            tp1 == tp2 for tp1, tp2 in zip(ty1.type_params, ty2.type_params)
        )

    def _check_predicate_implication(self, p1: Predicate, p2: Predicate) -> bool:
        """Approximate predicate implication check without SMT."""
        if p2.is_trivially_true():
            return True
        if p1.is_trivially_false():
            return True
        if p1 == p2:
            return True

        # x > 0 implies x >= 0
        if p1.kind == PredicateKind.GT and p2.kind == PredicateKind.GE:
            if p1.args == p2.args:
                return True
            if (
                len(p1.args) == 2
                and len(p2.args) == 2
                and p1.args[0] == p2.args[0]
                and isinstance(p1.args[1], PredicateLiteral)
                and isinstance(p2.args[1], PredicateLiteral)
            ):
                v1 = p1.args[1].value
                v2 = p2.args[1].value
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    if v1 >= v2:
                        return True

        # x >= a implies x >= b if a >= b
        if p1.kind == PredicateKind.GE and p2.kind == PredicateKind.GE:
            if (
                len(p1.args) == 2
                and len(p2.args) == 2
                and p1.args[0] == p2.args[0]
                and isinstance(p1.args[1], PredicateLiteral)
                and isinstance(p2.args[1], PredicateLiteral)
            ):
                v1 = p1.args[1].value
                v2 = p2.args[1].value
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    if v1 >= v2:
                        return True

        # p1 ∧ p2 implies p2
        if p1.kind == PredicateKind.AND:
            if p1.children[0] == p2 or p1.children[1] == p2:
                return True
            if self._check_predicate_implication(p1.children[0], p2):
                return True
            if self._check_predicate_implication(p1.children[1], p2):
                return True

        return False


# ===================================================================
# 9. TypeNarrower
# ===================================================================


class TypeNarrower:
    """Narrow types based on guard conditions."""

    def __init__(self) -> None:
        self._checker = SubtypeChecker()

    def narrow_isinstance(
        self, env: TypeEnvironment, var: str, tag: str
    ) -> TypeEnvironment:
        current = env.lookup(var)
        target = _BUILTIN_TYPE_MAP.get(tag)
        if target is None:
            target = ClassType.create(tag)
        if current is None:
            return env.bind(var, target)
        if isinstance(current, UnionType):
            matching = [t for t in current.types if self._matches_tag(t, tag)]
            if matching:
                narrowed = UnionType.of(*matching)
                return env.bind(var, narrowed)
        return env.bind(var, IntersectionType.of(current, target))

    def narrow_isinstance_negative(
        self, env: TypeEnvironment, var: str, tag: str
    ) -> TypeEnvironment:
        """Narrow when isinstance check is False."""
        current = env.lookup(var)
        if current is None:
            return env
        target = _BUILTIN_TYPE_MAP.get(tag)
        if target is None:
            target = ClassType.create(tag)
        if isinstance(current, UnionType):
            remaining = [t for t in current.types if not self._matches_tag(t, tag)]
            if remaining:
                return env.bind(var, UnionType.of(*remaining))
            return env.bind(var, BottomType())
        if current == target:
            return env.bind(var, BottomType())
        return env

    def narrow_is_none(
        self, env: TypeEnvironment, var: str, is_none: bool
    ) -> TypeEnvironment:
        current = env.lookup(var)
        if is_none:
            return env.bind(var, NoneType_())
        else:
            if current is None:
                return env
            if isinstance(current, UnionType):
                remaining = [t for t in current.types if not isinstance(t, NoneType_)]
                if remaining:
                    return env.bind(var, UnionType.of(*remaining))
            if isinstance(current, RefinementType):
                return env.bind(
                    var,
                    current.strengthen(Predicate.is_not_none(current.variable)),
                )
            return env.bind(
                var,
                RefinementType(current, "x", Predicate.is_not_none("x")),
            )

    def narrow_hasattr(
        self, env: TypeEnvironment, var: str, attr: str
    ) -> TypeEnvironment:
        current = env.lookup(var)
        if current is None:
            return env.bind(var, RefinementType(TopType(), "x", Predicate.hasattr_("x", attr)))
        if isinstance(current, UnionType):
            matching = [t for t in current.types if self._has_attr(t, attr)]
            if matching:
                return env.bind(var, UnionType.of(*matching))
        return env.bind(
            var,
            RefinementType(current, "x", Predicate.hasattr_("x", attr)),
        )

    def narrow_comparison(
        self, env: TypeEnvironment, var: str, op: str, value: Any
    ) -> TypeEnvironment:
        pred_map = {
            "<": Predicate.lt,
            "<=": Predicate.le,
            ">": Predicate.gt,
            ">=": Predicate.ge,
            "==": Predicate.eq,
            "!=": Predicate.neq,
        }
        factory = pred_map.get(op)
        if factory is None:
            return env

        pred = factory(var, value)
        current = env.lookup(var)

        if current is None:
            return env.bind(var, RefinementType(TopType(), var, pred))

        if isinstance(current, RefinementType):
            return env.bind(var, current.strengthen(pred))

        return env.bind(var, RefinementType(current, var, pred))

    def narrow_truthiness(
        self, env: TypeEnvironment, var: str, is_truthy: bool
    ) -> TypeEnvironment:
        current = env.lookup(var)
        if current is None:
            return env

        if is_truthy:
            # Remove None from union, and add truthy predicate
            if isinstance(current, UnionType):
                remaining = [t for t in current.types if not isinstance(t, NoneType_)]
                if remaining:
                    narrowed = UnionType.of(*remaining)
                    return env.bind(var, narrowed)
            return env.bind(var, RefinementType(current, var, Predicate.truthy(var)))
        else:
            # Falsy: None, 0, False, "", [], {}
            if isinstance(current, UnionType):
                none_types = [t for t in current.types if isinstance(t, NoneType_)]
                if none_types:
                    return env.bind(var, NoneType_())
            return env.bind(var, RefinementType(current, var, Predicate.falsy(var)))

    def narrow_typeof(
        self, env: TypeEnvironment, var: str, type_string: str
    ) -> TypeEnvironment:
        """TypeScript-style typeof narrowing."""
        type_map: Dict[str, BaseType] = {
            "number": UnionType.of(IntType(), FloatType()),
            "string": StrType(),
            "boolean": BoolType(),
            "undefined": NoneType_(),
            "object": TopType(),
            "function": CallableType((), TopType()),
        }
        target = type_map.get(type_string)
        if target is None:
            return env
        return env.bind(var, target)

    def narrow_in(
        self, env: TypeEnvironment, var: str, container: Any
    ) -> TypeEnvironment:
        """Narrow based on 'in' operator (membership check)."""
        current = env.lookup(var)
        if current is None:
            return env.bind(var, RefinementType(TopType(), var, Predicate.in_(var, container)))
        return env.bind(
            var,
            RefinementType(current, var, Predicate.in_(var, container)),
        )

    def narrow_equality(
        self, env: TypeEnvironment, var: str, value: Any
    ) -> TypeEnvironment:
        """Narrow based on equality check."""
        literal_type = LiteralType(value)
        return env.bind(var, literal_type)

    def narrow_compound(self, env: TypeEnvironment, guard: Guard) -> TypeEnvironment:
        """Narrow based on compound guards (and/or/not)."""
        if guard.kind == GuardKind.AND:
            left_env = self.narrow_compound(env, guard.children[0])
            return self.narrow_compound(left_env, guard.children[1])

        if guard.kind == GuardKind.OR:
            left_env = self.narrow_compound(env, guard.children[0])
            right_env = self.narrow_compound(env, guard.children[1])
            return left_env.join(right_env)

        if guard.kind == GuardKind.NOT:
            return self._narrow_negated(env, guard.children[0])

        if guard.kind == GuardKind.ISINSTANCE:
            return self.narrow_isinstance(env, guard.variable, guard.args[0])

        if guard.kind == GuardKind.IS_NONE:
            return self.narrow_is_none(env, guard.variable, True)

        if guard.kind == GuardKind.IS_NOT_NONE:
            return self.narrow_is_none(env, guard.variable, False)

        if guard.kind == GuardKind.HASATTR:
            return self.narrow_hasattr(env, guard.variable, guard.args[0])

        if guard.kind == GuardKind.COMPARISON:
            return self.narrow_comparison(env, guard.variable, guard.args[0], guard.args[1])

        if guard.kind == GuardKind.TRUTHY:
            return self.narrow_truthiness(env, guard.variable, True)

        if guard.kind == GuardKind.FALSY:
            return self.narrow_truthiness(env, guard.variable, False)

        if guard.kind == GuardKind.TYPEOF:
            return self.narrow_typeof(env, guard.variable, guard.args[0])

        if guard.kind == GuardKind.IN:
            return self.narrow_in(env, guard.variable, guard.args[0])

        if guard.kind == GuardKind.EQUALITY:
            return self.narrow_equality(env, guard.variable, guard.args[0])

        return env

    def _narrow_negated(self, env: TypeEnvironment, guard: Guard) -> TypeEnvironment:
        """Narrow based on negation of a guard."""
        if guard.kind == GuardKind.ISINSTANCE:
            return self.narrow_isinstance_negative(env, guard.variable, guard.args[0])
        if guard.kind == GuardKind.IS_NONE:
            return self.narrow_is_none(env, guard.variable, False)
        if guard.kind == GuardKind.IS_NOT_NONE:
            return self.narrow_is_none(env, guard.variable, True)
        if guard.kind == GuardKind.TRUTHY:
            return self.narrow_truthiness(env, guard.variable, False)
        if guard.kind == GuardKind.FALSY:
            return self.narrow_truthiness(env, guard.variable, True)
        if guard.kind == GuardKind.AND:
            # ¬(A ∧ B) = ¬A ∨ ¬B
            left_env = self._narrow_negated(env, guard.children[0])
            right_env = self._narrow_negated(env, guard.children[1])
            return left_env.join(right_env)
        if guard.kind == GuardKind.OR:
            # ¬(A ∨ B) = ¬A ∧ ¬B
            left_env = self._narrow_negated(env, guard.children[0])
            return self._narrow_negated(left_env, guard.children[1])
        if guard.kind == GuardKind.NOT:
            return self.narrow_compound(env, guard.children[0])
        return env

    def _matches_tag(self, ty: BaseType, tag: str) -> bool:
        target = _BUILTIN_TYPE_MAP.get(tag)
        if target is not None:
            return ty == target or isinstance(ty, type(target))
        if isinstance(ty, ClassType):
            return ty.name == tag
        return False

    def _has_attr(self, ty: BaseType, attr: str) -> bool:
        if isinstance(ty, ClassType):
            return ty.has_method(attr) or ty.has_field(attr)
        if isinstance(ty, StructuralType):
            return ty.has_field(attr)
        return True


# ===================================================================
# 10. TypeJoin — compute LUB
# ===================================================================


class TypeJoin:
    """Compute the join (least upper bound) of two types."""

    def join(self, ty1: BaseType, ty2: BaseType) -> BaseType:
        if ty1 == ty2:
            return ty1
        if isinstance(ty1, BottomType):
            return ty2
        if isinstance(ty2, BottomType):
            return ty1
        if isinstance(ty1, TopType) or isinstance(ty2, TopType):
            return TopType()

        # Resolve aliases
        if isinstance(ty1, TypeAliasType):
            return self.join(ty1.resolve(), ty2)
        if isinstance(ty2, TypeAliasType):
            return self.join(ty1, ty2.resolve())

        # Numeric tower
        idx1 = self._numeric_index(ty1)
        idx2 = self._numeric_index(ty2)
        if idx1 >= 0 and idx2 >= 0:
            return _NUMERIC_HIERARCHY[max(idx1, idx2)]()

        # Literal → base
        if isinstance(ty1, LiteralType) and isinstance(ty2, LiteralType):
            if type(ty1.value) is type(ty2.value):
                return ty1.base_of_literal()
            return self.join(ty1.base_of_literal(), ty2.base_of_literal())
        if isinstance(ty1, LiteralType):
            return self.join(ty1.base_of_literal(), ty2)
        if isinstance(ty2, LiteralType):
            return self.join(ty1, ty2.base_of_literal())

        # Union types
        if isinstance(ty1, UnionType) or isinstance(ty2, UnionType):
            return UnionType.of(ty1, ty2)

        # List
        if isinstance(ty1, ListType) and isinstance(ty2, ListType):
            return ListType(self.join(ty1.element_type, ty2.element_type))

        # Tuple
        if isinstance(ty1, TupleType) and isinstance(ty2, TupleType):
            if len(ty1.element_types) == len(ty2.element_types):
                elems = tuple(
                    self.join(t1, t2)
                    for t1, t2 in zip(ty1.element_types, ty2.element_types)
                )
                return TupleType(elems)
            return UnionType.of(ty1, ty2)

        # Set
        if isinstance(ty1, SetType) and isinstance(ty2, SetType):
            return SetType(self.join(ty1.element_type, ty2.element_type))
        if isinstance(ty1, FrozenSetType) and isinstance(ty2, FrozenSetType):
            return FrozenSetType(self.join(ty1.element_type, ty2.element_type))

        # Dict
        if isinstance(ty1, DictType) and isinstance(ty2, DictType):
            return DictType(
                self.join(ty1.key_type, ty2.key_type),
                self.join(ty1.value_type, ty2.value_type),
            )

        # Callable
        if isinstance(ty1, CallableType) and isinstance(ty2, CallableType):
            if len(ty1.param_types) == len(ty2.param_types):
                meeter = TypeMeet()
                params = tuple(
                    meeter.meet(p1, p2)
                    for p1, p2 in zip(ty1.param_types, ty2.param_types)
                )
                ret = self.join(ty1.return_type, ty2.return_type)
                return CallableType(params, ret)
            return UnionType.of(ty1, ty2)

        # Structural
        if isinstance(ty1, StructuralType) and isinstance(ty2, StructuralType):
            return self._join_structural(ty1, ty2)

        # Class nominal
        if isinstance(ty1, ClassType) and isinstance(ty2, ClassType):
            common = self._find_common_base(ty1, ty2)
            if common is not None:
                return common
            return UnionType.of(ty1, ty2)

        # Refinement
        if isinstance(ty1, RefinementType) and isinstance(ty2, RefinementType):
            base_join = self.join(ty1.base_type, ty2.base_type)
            if ty1.predicate == ty2.predicate:
                return RefinementType(base_join, ty1.variable, ty1.predicate)
            pred_join = Predicate.or_(ty1.predicate, ty2.predicate)
            return RefinementType(base_join, ty1.variable, pred_join)

        if isinstance(ty1, RefinementType):
            return self.join(ty1.base_type, ty2)
        if isinstance(ty2, RefinementType):
            return self.join(ty1, ty2.base_type)

        # Awaitable
        if isinstance(ty1, AwaitableType) and isinstance(ty2, AwaitableType):
            return AwaitableType(self.join(ty1.result_type, ty2.result_type))

        # Generator
        if isinstance(ty1, GeneratorType) and isinstance(ty2, GeneratorType):
            meeter = TypeMeet()
            return GeneratorType(
                self.join(ty1.yield_type, ty2.yield_type),
                meeter.meet(ty1.send_type, ty2.send_type),
                self.join(ty1.return_type, ty2.return_type),
            )

        # Coroutine
        if isinstance(ty1, CoroutineType) and isinstance(ty2, CoroutineType):
            meeter = TypeMeet()
            return CoroutineType(
                self.join(ty1.yield_type, ty2.yield_type),
                meeter.meet(ty1.send_type, ty2.send_type),
                self.join(ty1.return_type, ty2.return_type),
            )

        # ContextManager
        if isinstance(ty1, ContextManagerType) and isinstance(ty2, ContextManagerType):
            return ContextManagerType(
                self.join(ty1.enter_type, ty2.enter_type),
                self.join(ty1.exit_type, ty2.exit_type),
            )

        # Generic
        if isinstance(ty1, GenericType) and isinstance(ty2, GenericType):
            if ty1.base == ty2.base and len(ty1.type_params) == len(ty2.type_params):
                params = tuple(
                    self.join(p1, p2)
                    for p1, p2 in zip(ty1.type_params, ty2.type_params)
                )
                return GenericType(ty1.base, params)

        # Fallback: form union
        return UnionType.of(ty1, ty2)

    def _numeric_index(self, ty: BaseType) -> int:
        for i, cls in enumerate(_NUMERIC_HIERARCHY):
            if isinstance(ty, cls):
                return i
        return -1

    def _join_structural(
        self, ty1: StructuralType, ty2: StructuralType
    ) -> StructuralType:
        """Width subtyping join: keep only common fields with joined types."""
        fields1 = ty1.field_dict
        fields2 = ty2.field_dict
        common_fields: Dict[str, BaseType] = {}
        for name in set(fields1) & set(fields2):
            common_fields[name] = self.join(fields1[name], fields2[name])
        row = ty1.row_var or ty2.row_var
        return StructuralType.create(common_fields, row_var=row)

    def _find_common_base(self, ty1: ClassType, ty2: ClassType) -> Optional[BaseType]:
        """Find lowest common ancestor in class hierarchy."""
        bases1 = self._collect_bases(ty1)
        bases2 = self._collect_bases(ty2)
        common = bases1 & bases2
        if not common:
            return None
        for name in common:
            return ClassType.create(name)
        return None

    def _collect_bases(self, ty: ClassType, depth: int = 0) -> Set[str]:
        if depth > 20:
            return set()
        result = {ty.name}
        for base in ty.bases:
            if isinstance(base, ClassType):
                result |= self._collect_bases(base, depth + 1)
        return result


# ===================================================================
# 11. TypeMeet — compute GLB
# ===================================================================


class TypeMeet:
    """Compute the meet (greatest lower bound) of two types."""

    def meet(self, ty1: BaseType, ty2: BaseType) -> BaseType:
        if ty1 == ty2:
            return ty1
        if isinstance(ty1, TopType):
            return ty2
        if isinstance(ty2, TopType):
            return ty1
        if isinstance(ty1, BottomType) or isinstance(ty2, BottomType):
            return BottomType()

        # Resolve aliases
        if isinstance(ty1, TypeAliasType):
            return self.meet(ty1.resolve(), ty2)
        if isinstance(ty2, TypeAliasType):
            return self.meet(ty1, ty2.resolve())

        # Numeric tower: meet = lowest
        idx1 = self._numeric_index(ty1)
        idx2 = self._numeric_index(ty2)
        if idx1 >= 0 and idx2 >= 0:
            return _NUMERIC_HIERARCHY[min(idx1, idx2)]()

        # Union types
        if isinstance(ty1, UnionType):
            types = [self.meet(t, ty2) for t in ty1.types]
            non_bottom = [t for t in types if not isinstance(t, BottomType)]
            if not non_bottom:
                return BottomType()
            return UnionType.of(*non_bottom)

        if isinstance(ty2, UnionType):
            types = [self.meet(ty1, t) for t in ty2.types]
            non_bottom = [t for t in types if not isinstance(t, BottomType)]
            if not non_bottom:
                return BottomType()
            return UnionType.of(*non_bottom)

        # Intersection
        if isinstance(ty1, IntersectionType):
            return IntersectionType.of(*(self.meet(t, ty2) for t in ty1.types))
        if isinstance(ty2, IntersectionType):
            return IntersectionType.of(*(self.meet(ty1, t) for t in ty2.types))

        # List
        if isinstance(ty1, ListType) and isinstance(ty2, ListType):
            return ListType(self.meet(ty1.element_type, ty2.element_type))

        # Tuple
        if isinstance(ty1, TupleType) and isinstance(ty2, TupleType):
            if len(ty1.element_types) == len(ty2.element_types):
                elems = tuple(
                    self.meet(t1, t2)
                    for t1, t2 in zip(ty1.element_types, ty2.element_types)
                )
                return TupleType(elems)
            return BottomType()

        # Set
        if isinstance(ty1, SetType) and isinstance(ty2, SetType):
            return SetType(self.meet(ty1.element_type, ty2.element_type))

        # Dict
        if isinstance(ty1, DictType) and isinstance(ty2, DictType):
            return DictType(
                self.meet(ty1.key_type, ty2.key_type),
                self.meet(ty1.value_type, ty2.value_type),
            )

        # Callable
        if isinstance(ty1, CallableType) and isinstance(ty2, CallableType):
            if len(ty1.param_types) == len(ty2.param_types):
                joiner = TypeJoin()
                params = tuple(
                    joiner.join(p1, p2)
                    for p1, p2 in zip(ty1.param_types, ty2.param_types)
                )
                ret = self.meet(ty1.return_type, ty2.return_type)
                return CallableType(params, ret)
            return BottomType()

        # Structural: meet = union of fields with met types
        if isinstance(ty1, StructuralType) and isinstance(ty2, StructuralType):
            return self._meet_structural(ty1, ty2)

        # Refinement
        if isinstance(ty1, RefinementType) and isinstance(ty2, RefinementType):
            base_meet = self.meet(ty1.base_type, ty2.base_type)
            pred_meet = Predicate.and_(ty1.predicate, ty2.predicate)
            return RefinementType(base_meet, ty1.variable, pred_meet)

        if isinstance(ty1, RefinementType):
            base_meet = self.meet(ty1.base_type, ty2)
            if isinstance(base_meet, BottomType):
                return BottomType()
            return RefinementType(base_meet, ty1.variable, ty1.predicate)

        if isinstance(ty2, RefinementType):
            base_meet = self.meet(ty1, ty2.base_type)
            if isinstance(base_meet, BottomType):
                return BottomType()
            return RefinementType(base_meet, ty2.variable, ty2.predicate)

        # Class nominal
        if isinstance(ty1, ClassType) and isinstance(ty2, ClassType):
            checker = SubtypeChecker()
            if checker._is_subtype(ty1, ty2, 0):
                return ty1
            if checker._is_subtype(ty2, ty1, 0):
                return ty2
            return BottomType()

        # Fallback: form intersection (non-bottom if compatible)
        return IntersectionType.of(ty1, ty2)

    def _numeric_index(self, ty: BaseType) -> int:
        for i, cls in enumerate(_NUMERIC_HIERARCHY):
            if isinstance(ty, cls):
                return i
        return -1

    def _meet_structural(
        self, ty1: StructuralType, ty2: StructuralType
    ) -> StructuralType:
        fields1 = ty1.field_dict
        fields2 = ty2.field_dict
        all_fields: Dict[str, BaseType] = {}
        for name in set(fields1) | set(fields2):
            f1 = fields1.get(name)
            f2 = fields2.get(name)
            if f1 is not None and f2 is not None:
                all_fields[name] = self.meet(f1, f2)
            elif f1 is not None:
                all_fields[name] = f1
            elif f2 is not None:
                all_fields[name] = f2
        return StructuralType.create(all_fields)


# ===================================================================
# 12. TypeUnifier
# ===================================================================


@dataclass
class Substitution:
    """A mapping from type variable names to types."""

    mappings: Dict[str, BaseType] = field(default_factory=dict)

    def apply(self, ty: BaseType) -> BaseType:
        for var, replacement in self.mappings.items():
            ty = ty.substitute_type_var(var, replacement)
        return ty

    def compose(self, other: Substitution) -> Substitution:
        new_mappings = {k: other.apply(v) for k, v in self.mappings.items()}
        for k, v in other.mappings.items():
            if k not in new_mappings:
                new_mappings[k] = v
        return Substitution(new_mappings)

    def __repr__(self) -> str:
        entries = ", ".join(f"{k} ↦ {v}" for k, v in self.mappings.items())
        return f"[{entries}]"


class UnificationError(Exception):
    """Raised when unification fails."""

    def __init__(self, ty1: BaseType, ty2: BaseType, reason: str = "") -> None:
        self.ty1 = ty1
        self.ty2 = ty2
        self.reason = reason
        super().__init__(f"Cannot unify {ty1} with {ty2}: {reason}")


class TypeUnifier:
    """Hindley-Milner style unification for type inference."""

    def __init__(self) -> None:
        self._subst = Substitution()

    def unify(self, ty1: BaseType, ty2: BaseType) -> Substitution:
        self._subst = Substitution()
        self._unify(ty1, ty2)
        return self._subst

    def _unify(self, ty1: BaseType, ty2: BaseType) -> None:
        ty1 = self._subst.apply(ty1)
        ty2 = self._subst.apply(ty2)

        if ty1 == ty2:
            return

        # TypeVar unification
        if isinstance(ty1, TypeVarType):
            self._bind(ty1.name, ty2)
            return
        if isinstance(ty2, TypeVarType):
            self._bind(ty2.name, ty1)
            return

        # Resolve aliases
        if isinstance(ty1, TypeAliasType):
            self._unify(ty1.resolve(), ty2)
            return
        if isinstance(ty2, TypeAliasType):
            self._unify(ty1, ty2.resolve())
            return

        # Top / Bottom
        if isinstance(ty1, TopType) or isinstance(ty2, TopType):
            return
        if isinstance(ty1, BottomType) or isinstance(ty2, BottomType):
            return

        # List
        if isinstance(ty1, ListType) and isinstance(ty2, ListType):
            self._unify(ty1.element_type, ty2.element_type)
            return

        # Tuple
        if isinstance(ty1, TupleType) and isinstance(ty2, TupleType):
            if len(ty1.element_types) != len(ty2.element_types):
                raise UnificationError(ty1, ty2, "tuple length mismatch")
            for t1, t2 in zip(ty1.element_types, ty2.element_types):
                self._unify(t1, t2)
            return

        # Set
        if isinstance(ty1, SetType) and isinstance(ty2, SetType):
            self._unify(ty1.element_type, ty2.element_type)
            return
        if isinstance(ty1, FrozenSetType) and isinstance(ty2, FrozenSetType):
            self._unify(ty1.element_type, ty2.element_type)
            return

        # Dict
        if isinstance(ty1, DictType) and isinstance(ty2, DictType):
            self._unify(ty1.key_type, ty2.key_type)
            self._unify(ty1.value_type, ty2.value_type)
            return

        # Callable
        if isinstance(ty1, CallableType) and isinstance(ty2, CallableType):
            if len(ty1.param_types) != len(ty2.param_types):
                raise UnificationError(ty1, ty2, "param count mismatch")
            for p1, p2 in zip(ty1.param_types, ty2.param_types):
                self._unify(p1, p2)
            self._unify(ty1.return_type, ty2.return_type)
            return

        # Generic
        if isinstance(ty1, GenericType) and isinstance(ty2, GenericType):
            self._unify(ty1.base, ty2.base)
            if len(ty1.type_params) != len(ty2.type_params):
                raise UnificationError(ty1, ty2, "type param count mismatch")
            for tp1, tp2 in zip(ty1.type_params, ty2.type_params):
                self._unify(tp1, tp2)
            return

        # Refinement — unify bases, ignore predicates
        if isinstance(ty1, RefinementType) and isinstance(ty2, RefinementType):
            self._unify(ty1.base_type, ty2.base_type)
            return
        if isinstance(ty1, RefinementType):
            self._unify(ty1.base_type, ty2)
            return
        if isinstance(ty2, RefinementType):
            self._unify(ty1, ty2.base_type)
            return

        # Structural
        if isinstance(ty1, StructuralType) and isinstance(ty2, StructuralType):
            self._unify_structural(ty1, ty2)
            return

        # Awaitable
        if isinstance(ty1, AwaitableType) and isinstance(ty2, AwaitableType):
            self._unify(ty1.result_type, ty2.result_type)
            return

        # Generator
        if isinstance(ty1, GeneratorType) and isinstance(ty2, GeneratorType):
            self._unify(ty1.yield_type, ty2.yield_type)
            self._unify(ty1.send_type, ty2.send_type)
            self._unify(ty1.return_type, ty2.return_type)
            return

        # Coroutine
        if isinstance(ty1, CoroutineType) and isinstance(ty2, CoroutineType):
            self._unify(ty1.yield_type, ty2.yield_type)
            self._unify(ty1.send_type, ty2.send_type)
            self._unify(ty1.return_type, ty2.return_type)
            return

        # ContextManager
        if isinstance(ty1, ContextManagerType) and isinstance(ty2, ContextManagerType):
            self._unify(ty1.enter_type, ty2.enter_type)
            self._unify(ty1.exit_type, ty2.exit_type)
            return

        # Numeric tower
        idx1 = _numeric_idx(ty1)
        idx2 = _numeric_idx(ty2)
        if idx1 >= 0 and idx2 >= 0:
            return  # Compatible via numeric tower

        raise UnificationError(ty1, ty2, "incompatible types")

    def _bind(self, name: str, ty: BaseType) -> None:
        if isinstance(ty, TypeVarType) and ty.name == name:
            return
        if name in ty.free_type_vars():
            raise UnificationError(
                TypeVarType(name), ty, "occurs check failed (infinite type)"
            )
        self._subst = self._subst.compose(Substitution({name: ty}))

    def _unify_structural(self, ty1: StructuralType, ty2: StructuralType) -> None:
        fields1 = ty1.field_dict
        fields2 = ty2.field_dict
        for name in set(fields1) & set(fields2):
            self._unify(fields1[name], fields2[name])
        # Row variable unification
        if ty1.row_var and ty2.row_var and ty1.row_var != ty2.row_var:
            extra1 = {n: t for n, t in fields1.items() if n not in fields2}
            extra2 = {n: t for n, t in fields2.items() if n not in fields1}
            merged = StructuralType.create({**extra1, **extra2})
            self._bind(ty1.row_var, merged)


def _numeric_idx(ty: BaseType) -> int:
    for i, cls in enumerate(_NUMERIC_HIERARCHY):
        if isinstance(ty, cls):
            return i
    return -1


# ===================================================================
# 13. Expression AST for type inference
# ===================================================================


class ExprKind(Enum):
    LITERAL = auto()
    VARIABLE = auto()
    BINARY_OP = auto()
    UNARY_OP = auto()
    CALL = auto()
    ATTRIBUTE = auto()
    SUBSCRIPT = auto()
    IF_ELSE = auto()
    LAMBDA = auto()
    LET = auto()
    LIST_LITERAL = auto()
    DICT_LITERAL = auto()
    TUPLE_LITERAL = auto()


@dataclass
class TypeExpr:
    """Expression AST node for type inference."""

    kind: ExprKind
    value: Any = None
    children: List[TypeExpr] = field(default_factory=list)
    name: str = ""
    op: str = ""
    annotation: Optional[BaseType] = None


# ===================================================================
# 14. TypeInferer
# ===================================================================


class InferenceMode(Enum):
    INFER = auto()
    CHECK = auto()


@dataclass
class TypeConstraint:
    """A constraint generated during inference."""

    lhs: BaseType
    rhs: BaseType
    source: str = ""

    def __repr__(self) -> str:
        return f"{self.lhs} ⊑ {self.rhs} ({self.source})"


class TypeInferer:
    """Bidirectional type inference with let-polymorphism."""

    def __init__(self) -> None:
        self._constraints: List[TypeConstraint] = []
        self._fresh_counter: int = 0
        self._unifier = TypeUnifier()

    def fresh_var(self, prefix: str = "α") -> TypeVarType:
        self._fresh_counter += 1
        return TypeVarType(f"{prefix}{self._fresh_counter}")

    def infer(self, expr: TypeExpr, env: TypeEnvironment) -> BaseType:
        """Infer the type of an expression in a given environment."""
        if expr.kind == ExprKind.LITERAL:
            return self._infer_literal(expr.value)

        if expr.kind == ExprKind.VARIABLE:
            ty = env.lookup(expr.name)
            if ty is None:
                return TopType()
            return self._instantiate(ty)

        if expr.kind == ExprKind.BINARY_OP:
            return self._infer_binary(expr, env)

        if expr.kind == ExprKind.UNARY_OP:
            return self._infer_unary(expr, env)

        if expr.kind == ExprKind.CALL:
            return self._infer_call(expr, env)

        if expr.kind == ExprKind.ATTRIBUTE:
            return self._infer_attribute(expr, env)

        if expr.kind == ExprKind.SUBSCRIPT:
            return self._infer_subscript(expr, env)

        if expr.kind == ExprKind.IF_ELSE:
            return self._infer_if_else(expr, env)

        if expr.kind == ExprKind.LAMBDA:
            return self._infer_lambda(expr, env)

        if expr.kind == ExprKind.LET:
            return self._infer_let(expr, env)

        if expr.kind == ExprKind.LIST_LITERAL:
            return self._infer_list_literal(expr, env)

        if expr.kind == ExprKind.DICT_LITERAL:
            return self._infer_dict_literal(expr, env)

        if expr.kind == ExprKind.TUPLE_LITERAL:
            return self._infer_tuple_literal(expr, env)

        return TopType()

    def check(self, expr: TypeExpr, env: TypeEnvironment, expected: BaseType) -> bool:
        """Check that an expression has the expected type."""
        inferred = self.infer(expr, env)
        self._constraints.append(TypeConstraint(inferred, expected, "check"))
        checker = SubtypeChecker()
        return checker.is_subtype(env, inferred, expected)

    def solve_constraints(self) -> Substitution:
        """Solve accumulated type constraints via unification."""
        subst = Substitution()
        unifier = TypeUnifier()
        for constraint in self._constraints:
            try:
                s = unifier.unify(
                    subst.apply(constraint.lhs), subst.apply(constraint.rhs)
                )
                subst = subst.compose(s)
            except UnificationError:
                pass
        self._constraints.clear()
        return subst

    def _infer_literal(self, value: Any) -> BaseType:
        if value is None:
            return NoneType_()
        if isinstance(value, bool):
            return LiteralType(value)
        if isinstance(value, int):
            return LiteralType(value)
        if isinstance(value, float):
            return LiteralType(value)
        if isinstance(value, str):
            return LiteralType(value)
        if isinstance(value, bytes):
            return LiteralType(value)
        return TopType()

    def _infer_binary(self, expr: TypeExpr, env: TypeEnvironment) -> BaseType:
        assert len(expr.children) == 2
        left_ty = self.infer(expr.children[0], env)
        right_ty = self.infer(expr.children[1], env)
        op = expr.op

        # Arithmetic
        if op in ("+", "-", "*", "**"):
            return self._arithmetic_result(left_ty, right_ty)
        if op in ("/",):
            return FloatType()
        if op in ("//", "%"):
            return self._arithmetic_result(left_ty, right_ty)

        # Comparison
        if op in ("<", "<=", ">", ">=", "==", "!="):
            return BoolType()

        # Boolean
        if op in ("and", "or"):
            joiner = TypeJoin()
            return joiner.join(left_ty, right_ty)

        # Bitwise
        if op in ("&", "|", "^", "<<", ">>"):
            return IntType()

        # String concat
        if op == "+" and (isinstance(left_ty, StrType) or isinstance(right_ty, StrType)):
            return StrType()

        return TopType()

    def _infer_unary(self, expr: TypeExpr, env: TypeEnvironment) -> BaseType:
        assert len(expr.children) == 1
        operand_ty = self.infer(expr.children[0], env)
        op = expr.op
        if op == "not":
            return BoolType()
        if op == "-":
            return self._strip_literal(operand_ty)
        if op == "~":
            return IntType()
        return operand_ty

    def _infer_call(self, expr: TypeExpr, env: TypeEnvironment) -> BaseType:
        assert len(expr.children) >= 1
        func_ty = self.infer(expr.children[0], env)
        arg_types = [self.infer(child, env) for child in expr.children[1:]]

        if isinstance(func_ty, CallableType):
            # Check argument types
            for i, (arg_ty, param_ty) in enumerate(
                zip(arg_types, func_ty.param_types)
            ):
                self._constraints.append(
                    TypeConstraint(arg_ty, param_ty, f"arg_{i}")
                )
            return func_ty.return_type

        if isinstance(func_ty, DependentFunctionType):
            return func_ty.return_as_refinement()

        return self.fresh_var("ret")

    def _infer_attribute(self, expr: TypeExpr, env: TypeEnvironment) -> BaseType:
        assert len(expr.children) == 1
        obj_ty = self.infer(expr.children[0], env)
        attr = expr.name

        if isinstance(obj_ty, ClassType):
            mt = obj_ty.get_method(attr)
            if mt is not None:
                return mt
            ft = obj_ty.get_field(attr)
            if ft is not None:
                return ft

        if isinstance(obj_ty, StructuralType):
            ft = obj_ty.get_field(attr)
            if ft is not None:
                return ft

        return self.fresh_var("attr")

    def _infer_subscript(self, expr: TypeExpr, env: TypeEnvironment) -> BaseType:
        assert len(expr.children) == 2
        container_ty = self.infer(expr.children[0], env)
        index_ty = self.infer(expr.children[1], env)

        if isinstance(container_ty, ListType):
            return container_ty.element_type
        if isinstance(container_ty, TupleType):
            if isinstance(index_ty, LiteralType) and isinstance(index_ty.value, int):
                idx = index_ty.value
                if 0 <= idx < len(container_ty.element_types):
                    return container_ty.element_types[idx]
            return UnionType.of(*container_ty.element_types) if container_ty.element_types else TopType()
        if isinstance(container_ty, DictType):
            return container_ty.value_type
        if isinstance(container_ty, StrType):
            return StrType()

        return self.fresh_var("elem")

    def _infer_if_else(self, expr: TypeExpr, env: TypeEnvironment) -> BaseType:
        assert len(expr.children) == 3
        self.infer(expr.children[0], env)  # condition
        then_ty = self.infer(expr.children[1], env)
        else_ty = self.infer(expr.children[2], env)
        joiner = TypeJoin()
        return joiner.join(then_ty, else_ty)

    def _infer_lambda(self, expr: TypeExpr, env: TypeEnvironment) -> BaseType:
        param_names: List[str] = expr.value or []
        param_types = [self.fresh_var(f"p_{name}") for name in param_names]
        body_env = env
        for name, ty in zip(param_names, param_types):
            body_env = body_env.bind(name, ty)
        assert len(expr.children) >= 1
        ret_ty = self.infer(expr.children[0], body_env)
        return CallableType(tuple(param_types), ret_ty)

    def _infer_let(self, expr: TypeExpr, env: TypeEnvironment) -> BaseType:
        """Let-polymorphism: let x = e1 in e2."""
        assert len(expr.children) == 2
        bound_ty = self.infer(expr.children[0], env)
        generalized = self._generalize(env, bound_ty)
        new_env = env.bind(expr.name, generalized)
        return self.infer(expr.children[1], new_env)

    def _infer_list_literal(self, expr: TypeExpr, env: TypeEnvironment) -> BaseType:
        if not expr.children:
            return ListType(self.fresh_var("elem"))
        elem_types = [self.infer(child, env) for child in expr.children]
        joiner = TypeJoin()
        result = elem_types[0]
        for t in elem_types[1:]:
            result = joiner.join(result, t)
        return ListType(result)

    def _infer_dict_literal(self, expr: TypeExpr, env: TypeEnvironment) -> BaseType:
        if not expr.children:
            return DictType(self.fresh_var("key"), self.fresh_var("val"))
        joiner = TypeJoin()
        key_types = [self.infer(expr.children[i], env) for i in range(0, len(expr.children), 2)]
        val_types = [self.infer(expr.children[i], env) for i in range(1, len(expr.children), 2)]
        key_ty = key_types[0]
        for t in key_types[1:]:
            key_ty = joiner.join(key_ty, t)
        val_ty = val_types[0]
        for t in val_types[1:]:
            val_ty = joiner.join(val_ty, t)
        return DictType(key_ty, val_ty)

    def _infer_tuple_literal(self, expr: TypeExpr, env: TypeEnvironment) -> BaseType:
        elem_types = tuple(self.infer(child, env) for child in expr.children)
        return TupleType(elem_types)

    def _arithmetic_result(self, ty1: BaseType, ty2: BaseType) -> BaseType:
        t1 = self._strip_literal(ty1)
        t2 = self._strip_literal(ty2)
        idx1 = _numeric_idx(t1)
        idx2 = _numeric_idx(t2)
        if idx1 >= 0 and idx2 >= 0:
            return _NUMERIC_HIERARCHY[max(idx1, idx2)]()
        return TopType()

    def _strip_literal(self, ty: BaseType) -> BaseType:
        if isinstance(ty, LiteralType):
            return ty.base_of_literal()
        return ty

    def _instantiate(self, ty: BaseType) -> BaseType:
        """Instantiate a polymorphic type with fresh variables."""
        fvs = ty.free_type_vars()
        if not fvs:
            return ty
        result = ty
        for fv in fvs:
            result = result.substitute_type_var(fv, self.fresh_var(fv))
        return result

    def _generalize(self, env: TypeEnvironment, ty: BaseType) -> BaseType:
        """Generalize a type: quantify free vars not in environment."""
        env_vars: Set[str] = set()
        for _, et in env.items():
            env_vars |= et.free_type_vars()
        free = ty.free_type_vars() - env_vars
        # In a full implementation, would wrap in ForAll; here we just return as-is
        return ty


# ===================================================================
# 15. TypePrinter
# ===================================================================


class TypePrinter:
    """Pretty-print types in various formats."""

    def to_python_annotation(self, ty: BaseType) -> str:
        """Python 3.10+ annotation style."""
        if isinstance(ty, TopType):
            return "Any"
        if isinstance(ty, BottomType):
            return "Never"
        if isinstance(ty, IntType):
            return "int"
        if isinstance(ty, FloatType):
            return "float"
        if isinstance(ty, ComplexType):
            return "complex"
        if isinstance(ty, BoolType):
            return "bool"
        if isinstance(ty, StrType):
            return "str"
        if isinstance(ty, BytesType):
            return "bytes"
        if isinstance(ty, NoneType_):
            return "None"
        if isinstance(ty, ListType):
            return f"list[{self.to_python_annotation(ty.element_type)}]"
        if isinstance(ty, TupleType):
            elems = ", ".join(self.to_python_annotation(t) for t in ty.element_types)
            return f"tuple[{elems}]"
        if isinstance(ty, SetType):
            return f"set[{self.to_python_annotation(ty.element_type)}]"
        if isinstance(ty, FrozenSetType):
            return f"frozenset[{self.to_python_annotation(ty.element_type)}]"
        if isinstance(ty, DictType):
            k = self.to_python_annotation(ty.key_type)
            v = self.to_python_annotation(ty.value_type)
            return f"dict[{k}, {v}]"
        if isinstance(ty, CallableType):
            params = ", ".join(self.to_python_annotation(p) for p in ty.param_types)
            ret = self.to_python_annotation(ty.return_type)
            return f"Callable[[{params}], {ret}]"
        if isinstance(ty, UnionType):
            parts = sorted(self.to_python_annotation(t) for t in ty.types)
            return " | ".join(parts)
        if isinstance(ty, IntersectionType):
            parts = sorted(self.to_python_annotation(t) for t in ty.types)
            return " & ".join(parts)
        if isinstance(ty, LiteralType):
            return f"Literal[{ty.value!r}]"
        if isinstance(ty, TypeVarType):
            return ty.name
        if isinstance(ty, GenericType):
            base = self.to_python_annotation(ty.base)
            params = ", ".join(self.to_python_annotation(p) for p in ty.type_params)
            return f"{base}[{params}]"
        if isinstance(ty, RefinementType):
            return f"{{{ty.variable}: {self.to_python_annotation(ty.base_type)} | {ty.predicate}}}"
        if isinstance(ty, AwaitableType):
            return f"Awaitable[{self.to_python_annotation(ty.result_type)}]"
        if isinstance(ty, IterableType):
            return f"Iterable[{self.to_python_annotation(ty.yield_type)}]"
        if isinstance(ty, GeneratorType):
            y = self.to_python_annotation(ty.yield_type)
            s = self.to_python_annotation(ty.send_type)
            r = self.to_python_annotation(ty.return_type)
            return f"Generator[{y}, {s}, {r}]"
        if isinstance(ty, CoroutineType):
            y = self.to_python_annotation(ty.yield_type)
            s = self.to_python_annotation(ty.send_type)
            r = self.to_python_annotation(ty.return_type)
            return f"Coroutine[{y}, {s}, {r}]"
        if isinstance(ty, ContextManagerType):
            e = self.to_python_annotation(ty.enter_type)
            x = self.to_python_annotation(ty.exit_type)
            return f"ContextManager[{e}, {x}]"
        if isinstance(ty, ProtocolType):
            return ty.name
        if isinstance(ty, ClassType):
            return ty.name
        if isinstance(ty, StructuralType):
            flds = ", ".join(f"{n}: {self.to_python_annotation(t)}" for n, t in ty.fields)
            return "{" + flds + "}"
        if isinstance(ty, TypeAliasType):
            return ty.name
        if isinstance(ty, RecursiveType):
            return f"μ{ty.var}. {self.to_python_annotation(ty.body)}"
        if isinstance(ty, DependentFunctionType):
            ps = ", ".join(
                f"{p.name}: {self.to_python_annotation(p.type)}" for p in ty.params
            )
            ret = self.to_python_annotation(ty.return_type)
            return f"({ps}) → {ret}"
        return repr(ty)

    def to_typescript(self, ty: BaseType) -> str:
        """TypeScript-style annotation."""
        if isinstance(ty, TopType):
            return "any"
        if isinstance(ty, BottomType):
            return "never"
        if isinstance(ty, IntType) or isinstance(ty, FloatType):
            return "number"
        if isinstance(ty, BoolType):
            return "boolean"
        if isinstance(ty, StrType):
            return "string"
        if isinstance(ty, NoneType_):
            return "null"
        if isinstance(ty, ListType):
            return f"{self.to_typescript(ty.element_type)}[]"
        if isinstance(ty, TupleType):
            elems = ", ".join(self.to_typescript(t) for t in ty.element_types)
            return f"[{elems}]"
        if isinstance(ty, DictType):
            k = self.to_typescript(ty.key_type)
            v = self.to_typescript(ty.value_type)
            return f"Record<{k}, {v}>"
        if isinstance(ty, CallableType):
            params = ", ".join(
                f"arg{i}: {self.to_typescript(p)}" for i, p in enumerate(ty.param_types)
            )
            ret = self.to_typescript(ty.return_type)
            return f"({params}) => {ret}"
        if isinstance(ty, UnionType):
            parts = sorted(self.to_typescript(t) for t in ty.types)
            return " | ".join(parts)
        if isinstance(ty, IntersectionType):
            parts = sorted(self.to_typescript(t) for t in ty.types)
            return " & ".join(parts)
        if isinstance(ty, LiteralType):
            if isinstance(ty.value, str):
                return f'"{ty.value}"'
            return str(ty.value).lower()
        if isinstance(ty, TypeVarType):
            return ty.name
        if isinstance(ty, GenericType):
            base = self.to_typescript(ty.base)
            params = ", ".join(self.to_typescript(p) for p in ty.type_params)
            return f"{base}<{params}>"
        if isinstance(ty, StructuralType):
            flds = "; ".join(f"{n}: {self.to_typescript(t)}" for n, t in ty.fields)
            return "{ " + flds + " }"
        if isinstance(ty, AwaitableType):
            return f"Promise<{self.to_typescript(ty.result_type)}>"
        if isinstance(ty, ClassType):
            return ty.name
        return repr(ty)

    def to_latex(self, ty: BaseType) -> str:
        """LaTeX-style type notation."""
        if isinstance(ty, TopType):
            return r"\top"
        if isinstance(ty, BottomType):
            return r"\bot"
        if isinstance(ty, IntType):
            return r"\texttt{int}"
        if isinstance(ty, FloatType):
            return r"\texttt{float}"
        if isinstance(ty, ComplexType):
            return r"\texttt{complex}"
        if isinstance(ty, BoolType):
            return r"\texttt{bool}"
        if isinstance(ty, StrType):
            return r"\texttt{str}"
        if isinstance(ty, BytesType):
            return r"\texttt{bytes}"
        if isinstance(ty, NoneType_):
            return r"\texttt{None}"
        if isinstance(ty, ListType):
            return r"\texttt{list}[" + self.to_latex(ty.element_type) + "]"
        if isinstance(ty, TupleType):
            elems = r" \times ".join(self.to_latex(t) for t in ty.element_types)
            return f"({elems})"
        if isinstance(ty, DictType):
            return rf"\texttt{{dict}}[{self.to_latex(ty.key_type)}, {self.to_latex(ty.value_type)}]"
        if isinstance(ty, CallableType):
            params = r" \times ".join(self.to_latex(p) for p in ty.param_types)
            ret = self.to_latex(ty.return_type)
            return rf"({params}) \to {ret}"
        if isinstance(ty, UnionType):
            parts = sorted(self.to_latex(t) for t in ty.types)
            return r" \cup ".join(parts)
        if isinstance(ty, IntersectionType):
            parts = sorted(self.to_latex(t) for t in ty.types)
            return r" \cap ".join(parts)
        if isinstance(ty, RefinementType):
            base = self.to_latex(ty.base_type)
            return rf"\{{{ty.variable}: {base} \mid {ty.predicate}\}}"
        if isinstance(ty, TypeVarType):
            return ty.name
        if isinstance(ty, LiteralType):
            return rf"\texttt{{{ty.value!r}}}"
        if isinstance(ty, GenericType):
            base = self.to_latex(ty.base)
            params = ", ".join(self.to_latex(p) for p in ty.type_params)
            return f"{base}[{params}]"
        if isinstance(ty, RecursiveType):
            return rf"\mu {ty.var}.\, {self.to_latex(ty.body)}"
        if isinstance(ty, DependentFunctionType):
            ps = ", ".join(f"{p.name}: {self.to_latex(p.type)}" for p in ty.params)
            ret = self.to_latex(ty.return_type)
            return rf"({ps}) \to {ret}"
        return repr(ty)

    def to_refinement_string(self, ty: BaseType) -> str:
        """Refinement-style string with predicates."""
        if isinstance(ty, RefinementType):
            return f"{{{ty.variable}: {self.to_python_annotation(ty.base_type)} | {ty.predicate}}}"
        if isinstance(ty, DependentFunctionType):
            parts = []
            for p in ty.params:
                if p.predicate:
                    parts.append(
                        f"{p.name}: {{{p.name}: {self.to_python_annotation(p.type)} | {p.predicate}}}"
                    )
                else:
                    parts.append(f"{p.name}: {self.to_python_annotation(p.type)}")
            ret = self.to_python_annotation(ty.return_type)
            if ty.return_predicate:
                ret = f"{{v: {ret} | {ty.return_predicate}}}"
            return f"({', '.join(parts)}) → {ret}"
        return self.to_python_annotation(ty)


# ===================================================================
# 16. TypeSerializer
# ===================================================================


class TypeSerializer:
    """Serialize types to/from JSON."""

    def to_json(self, ty: BaseType) -> Dict[str, Any]:
        d: Dict[str, Any] = {"__schema_version__": _SCHEMA_VERSION}

        if isinstance(ty, TopType):
            d["kind"] = "top"
        elif isinstance(ty, BottomType):
            d["kind"] = "bottom"
        elif isinstance(ty, IntType):
            d["kind"] = "int"
        elif isinstance(ty, FloatType):
            d["kind"] = "float"
        elif isinstance(ty, ComplexType):
            d["kind"] = "complex"
        elif isinstance(ty, BoolType):
            d["kind"] = "bool"
        elif isinstance(ty, StrType):
            d["kind"] = "str"
        elif isinstance(ty, BytesType):
            d["kind"] = "bytes"
        elif isinstance(ty, NoneType_):
            d["kind"] = "none"
        elif isinstance(ty, ListType):
            d["kind"] = "list"
            d["element"] = self.to_json(ty.element_type)
        elif isinstance(ty, TupleType):
            d["kind"] = "tuple"
            d["elements"] = [self.to_json(t) for t in ty.element_types]
        elif isinstance(ty, SetType):
            d["kind"] = "set"
            d["element"] = self.to_json(ty.element_type)
        elif isinstance(ty, FrozenSetType):
            d["kind"] = "frozenset"
            d["element"] = self.to_json(ty.element_type)
        elif isinstance(ty, DictType):
            d["kind"] = "dict"
            d["key"] = self.to_json(ty.key_type)
            d["value"] = self.to_json(ty.value_type)
        elif isinstance(ty, CallableType):
            d["kind"] = "callable"
            d["params"] = [self.to_json(p) for p in ty.param_types]
            d["return"] = self.to_json(ty.return_type)
        elif isinstance(ty, UnionType):
            d["kind"] = "union"
            d["types"] = [self.to_json(t) for t in ty.types]
        elif isinstance(ty, IntersectionType):
            d["kind"] = "intersection"
            d["types"] = [self.to_json(t) for t in ty.types]
        elif isinstance(ty, LiteralType):
            d["kind"] = "literal"
            d["value"] = ty.value
            d["value_type"] = type(ty.value).__name__
        elif isinstance(ty, TypeVarType):
            d["kind"] = "typevar"
            d["name"] = ty.name
            if ty.bound:
                d["bound"] = self.to_json(ty.bound)
            if ty.constraints:
                d["constraints"] = [self.to_json(c) for c in ty.constraints]
        elif isinstance(ty, GenericType):
            d["kind"] = "generic"
            d["base"] = self.to_json(ty.base)
            d["params"] = [self.to_json(p) for p in ty.type_params]
        elif isinstance(ty, ClassType):
            d["kind"] = "class"
            d["name"] = ty.name
            d["bases"] = [self.to_json(b) for b in ty.bases]
            d["methods"] = {n: self.to_json(t) for n, t in ty.methods}
            d["fields"] = {n: self.to_json(t) for n, t in ty.class_fields}
        elif isinstance(ty, StructuralType):
            d["kind"] = "structural"
            d["fields"] = {n: self.to_json(t) for n, t in ty.fields}
            d["row_var"] = ty.row_var
        elif isinstance(ty, ProtocolType):
            d["kind"] = "protocol"
            d["name"] = ty.name
            d["methods"] = {n: self.to_json(t) for n, t in ty.methods}
        elif isinstance(ty, TypeAliasType):
            d["kind"] = "alias"
            d["name"] = ty.name
            d["target"] = self.to_json(ty.target)
        elif isinstance(ty, RecursiveType):
            d["kind"] = "recursive"
            d["var"] = ty.var
            d["body"] = self.to_json(ty.body)
        elif isinstance(ty, RefinementType):
            d["kind"] = "refinement"
            d["base"] = self.to_json(ty.base_type)
            d["variable"] = ty.variable
            d["predicate"] = self._predicate_to_json(ty.predicate)
        elif isinstance(ty, DependentFunctionType):
            d["kind"] = "dependent_function"
            d["params"] = [
                {
                    "name": p.name,
                    "type": self.to_json(p.type),
                    "predicate": self._predicate_to_json(p.predicate) if p.predicate else None,
                }
                for p in ty.params
            ]
            d["return"] = self.to_json(ty.return_type)
            d["return_predicate"] = (
                self._predicate_to_json(ty.return_predicate) if ty.return_predicate else None
            )
        elif isinstance(ty, AwaitableType):
            d["kind"] = "awaitable"
            d["result"] = self.to_json(ty.result_type)
        elif isinstance(ty, IterableType):
            d["kind"] = "iterable"
            d["yield"] = self.to_json(ty.yield_type)
        elif isinstance(ty, GeneratorType):
            d["kind"] = "generator"
            d["yield"] = self.to_json(ty.yield_type)
            d["send"] = self.to_json(ty.send_type)
            d["return"] = self.to_json(ty.return_type)
        elif isinstance(ty, CoroutineType):
            d["kind"] = "coroutine"
            d["yield"] = self.to_json(ty.yield_type)
            d["send"] = self.to_json(ty.send_type)
            d["return"] = self.to_json(ty.return_type)
        elif isinstance(ty, ContextManagerType):
            d["kind"] = "context_manager"
            d["enter"] = self.to_json(ty.enter_type)
            d["exit"] = self.to_json(ty.exit_type)
        else:
            d["kind"] = "unknown"
            d["repr"] = repr(ty)

        return d

    def from_json(self, d: Dict[str, Any]) -> BaseType:
        kind = d.get("kind", "unknown")

        if kind == "top":
            return TopType()
        if kind == "bottom":
            return BottomType()
        if kind == "int":
            return IntType()
        if kind == "float":
            return FloatType()
        if kind == "complex":
            return ComplexType()
        if kind == "bool":
            return BoolType()
        if kind == "str":
            return StrType()
        if kind == "bytes":
            return BytesType()
        if kind == "none":
            return NoneType_()

        if kind == "list":
            return ListType(self.from_json(d["element"]))
        if kind == "tuple":
            return TupleType(tuple(self.from_json(e) for e in d["elements"]))
        if kind == "set":
            return SetType(self.from_json(d["element"]))
        if kind == "frozenset":
            return FrozenSetType(self.from_json(d["element"]))
        if kind == "dict":
            return DictType(self.from_json(d["key"]), self.from_json(d["value"]))

        if kind == "callable":
            return CallableType(
                tuple(self.from_json(p) for p in d["params"]),
                self.from_json(d["return"]),
            )

        if kind == "union":
            return UnionType.of(*(self.from_json(t) for t in d["types"]))

        if kind == "intersection":
            return IntersectionType.of(*(self.from_json(t) for t in d["types"]))

        if kind == "literal":
            value = d["value"]
            vtype = d.get("value_type", "")
            if vtype == "int":
                value = int(value)
            elif vtype == "float":
                value = float(value)
            elif vtype == "bool":
                value = bool(value)
            return LiteralType(value)

        if kind == "typevar":
            bound = self.from_json(d["bound"]) if "bound" in d and d["bound"] else None
            constraints = tuple(self.from_json(c) for c in d.get("constraints", []))
            return TypeVarType(d["name"], bound, constraints)

        if kind == "generic":
            return GenericType(
                self.from_json(d["base"]),
                tuple(self.from_json(p) for p in d["params"]),
            )

        if kind == "class":
            return ClassType(
                name=d["name"],
                bases=tuple(self.from_json(b) for b in d.get("bases", [])),
                methods=tuple((n, self.from_json(t)) for n, t in d.get("methods", {}).items()),
                class_fields=tuple((n, self.from_json(t)) for n, t in d.get("fields", {}).items()),
            )

        if kind == "structural":
            fields = {n: self.from_json(t) for n, t in d.get("fields", {}).items()}
            return StructuralType.create(fields, row_var=d.get("row_var"))

        if kind == "protocol":
            methods = {n: self.from_json(t) for n, t in d.get("methods", {}).items()}
            return ProtocolType.create(d["name"], methods)

        if kind == "alias":
            return TypeAliasType(d["name"], self.from_json(d["target"]))

        if kind == "recursive":
            return RecursiveType(d["var"], self.from_json(d["body"]))

        if kind == "refinement":
            return RefinementType(
                self.from_json(d["base"]),
                d["variable"],
                self._predicate_from_json(d["predicate"]),
            )

        if kind == "dependent_function":
            params = tuple(
                DependentParam(
                    p["name"],
                    self.from_json(p["type"]),
                    self._predicate_from_json(p["predicate"]) if p.get("predicate") else None,
                )
                for p in d["params"]
            )
            ret_pred = (
                self._predicate_from_json(d["return_predicate"])
                if d.get("return_predicate")
                else None
            )
            return DependentFunctionType(params, self.from_json(d["return"]), ret_pred)

        if kind == "awaitable":
            return AwaitableType(self.from_json(d["result"]))

        if kind == "iterable":
            return IterableType(self.from_json(d["yield"]))

        if kind == "generator":
            return GeneratorType(
                self.from_json(d["yield"]),
                self.from_json(d["send"]),
                self.from_json(d["return"]),
            )

        if kind == "coroutine":
            return CoroutineType(
                self.from_json(d["yield"]),
                self.from_json(d["send"]),
                self.from_json(d["return"]),
            )

        if kind == "context_manager":
            return ContextManagerType(
                self.from_json(d["enter"]),
                self.from_json(d["exit"]),
            )

        return TopType()

    def _predicate_to_json(self, pred: Predicate) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": pred.kind.name}
        if pred.args:
            serialized_args: List[Any] = []
            for a in pred.args:
                if isinstance(a, PredicateVar):
                    serialized_args.append({"_type": "var", "name": a.name})
                elif isinstance(a, PredicateLiteral):
                    serialized_args.append({"_type": "lit", "value": a.value})
                else:
                    serialized_args.append(a)
            d["args"] = serialized_args
        if pred.children:
            d["children"] = [self._predicate_to_json(c) for c in pred.children]
        return d

    def _predicate_from_json(self, d: Dict[str, Any]) -> Predicate:
        kind = PredicateKind[d["kind"]]
        args: List[Any] = []
        for a in d.get("args", []):
            if isinstance(a, dict):
                if a.get("_type") == "var":
                    args.append(PredicateVar(a["name"]))
                elif a.get("_type") == "lit":
                    args.append(PredicateLiteral(a["value"]))
                else:
                    args.append(a)
            else:
                args.append(a)
        children = tuple(self._predicate_from_json(c) for c in d.get("children", []))
        return Predicate(kind, tuple(args), children)

    def to_json_string(self, ty: BaseType, indent: int = 2) -> str:
        return json.dumps(self.to_json(ty), indent=indent, default=str)

    def from_json_string(self, s: str) -> BaseType:
        return self.from_json(json.loads(s))


# ===================================================================
# 17. TypeStatistics
# ===================================================================


class TypeStatistics:
    """Collect statistics about inferred types."""

    def __init__(self) -> None:
        self._type_counts: Dict[str, int] = {}
        self._refinement_count: int = 0
        self._total_count: int = 0
        self._predicate_complexities: List[int] = []
        self._function_count: int = 0
        self._refined_function_count: int = 0

    def record_type(self, ty: BaseType) -> None:
        self._total_count += 1
        kind = type(ty).__name__
        self._type_counts[kind] = self._type_counts.get(kind, 0) + 1

        if isinstance(ty, RefinementType):
            self._refinement_count += 1
            self._predicate_complexities.append(ty.predicate.complexity())

        if isinstance(ty, (CallableType, DependentFunctionType)):
            self._function_count += 1
            if isinstance(ty, DependentFunctionType):
                self._refined_function_count += 1

    def record_environment(self, env: TypeEnvironment) -> None:
        for _, ty in env.items():
            self.record_type(ty)

    def type_distribution(self) -> Dict[str, int]:
        return dict(self._type_counts)

    def refinement_coverage(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._refinement_count / self._total_count

    def function_refinement_coverage(self) -> float:
        if self._function_count == 0:
            return 0.0
        return self._refined_function_count / self._function_count

    def predicate_complexity_distribution(self) -> Dict[str, float]:
        if not self._predicate_complexities:
            return {"min": 0, "max": 0, "mean": 0, "median": 0}
        sorted_c = sorted(self._predicate_complexities)
        n = len(sorted_c)
        return {
            "min": sorted_c[0],
            "max": sorted_c[-1],
            "mean": sum(sorted_c) / n,
            "median": sorted_c[n // 2],
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "total_types": self._total_count,
            "type_distribution": self.type_distribution(),
            "refinement_count": self._refinement_count,
            "refinement_coverage": round(self.refinement_coverage(), 4),
            "function_refinement_coverage": round(self.function_refinement_coverage(), 4),
            "predicate_complexity": self.predicate_complexity_distribution(),
        }

    def __repr__(self) -> str:
        return (
            f"TypeStatistics(total={self._total_count}, "
            f"refined={self._refinement_count}, "
            f"coverage={self.refinement_coverage():.1%})"
        )


# ===================================================================
# 18. Convenience type constructors
# ===================================================================


def make_list(elem: BaseType) -> ListType:
    return ListType(elem)


def make_dict(key: BaseType, val: BaseType) -> DictType:
    return DictType(key, val)


def make_tuple(*elems: BaseType) -> TupleType:
    return TupleType(elems)


def make_callable(params: List[BaseType], ret: BaseType) -> CallableType:
    return CallableType(tuple(params), ret)


def make_union(*types: BaseType) -> BaseType:
    return UnionType.of(*types)


def make_intersection(*types: BaseType) -> BaseType:
    return IntersectionType.of(*types)


def make_optional(ty: BaseType) -> BaseType:
    return OptionalType(ty)


def make_generic(base: BaseType, *params: BaseType) -> GenericType:
    return GenericType(base, params)


def make_refinement(base: BaseType, var: str, pred: Predicate) -> RefinementType:
    return RefinementType(base, var, pred)


def make_dependent_fn(
    params: List[Tuple[str, BaseType, Optional[Predicate]]],
    ret: BaseType,
    ret_pred: Optional[Predicate] = None,
) -> DependentFunctionType:
    dp = tuple(DependentParam(name, ty, pred) for name, ty, pred in params)
    return DependentFunctionType(dp, ret, ret_pred)


# ===================================================================
# 19. Type walking / transformation utilities
# ===================================================================


class TypeVisitor(ABC):
    """Visitor for traversing type trees."""

    def visit(self, ty: BaseType) -> Any:
        method_name = f"visit_{type(ty).__name__}"
        method = getattr(self, method_name, self.generic_visit)
        return method(ty)

    def generic_visit(self, ty: BaseType) -> Any:
        return None


class TypeTransformer:
    """Transform types by applying a function to each node."""

    def __init__(self, transform: Callable[[BaseType], BaseType]) -> None:
        self._transform = transform

    def apply(self, ty: BaseType) -> BaseType:
        ty = self._transform(ty)
        return self._descend(ty)

    def _descend(self, ty: BaseType) -> BaseType:
        if isinstance(ty, ListType):
            return ListType(self.apply(ty.element_type))
        if isinstance(ty, TupleType):
            return TupleType(tuple(self.apply(t) for t in ty.element_types))
        if isinstance(ty, SetType):
            return SetType(self.apply(ty.element_type))
        if isinstance(ty, FrozenSetType):
            return FrozenSetType(self.apply(ty.element_type))
        if isinstance(ty, DictType):
            return DictType(self.apply(ty.key_type), self.apply(ty.value_type))
        if isinstance(ty, CallableType):
            return CallableType(
                tuple(self.apply(p) for p in ty.param_types),
                self.apply(ty.return_type),
            )
        if isinstance(ty, UnionType):
            return UnionType.of(*(self.apply(t) for t in ty.types))
        if isinstance(ty, IntersectionType):
            return IntersectionType.of(*(self.apply(t) for t in ty.types))
        if isinstance(ty, RefinementType):
            return RefinementType(self.apply(ty.base_type), ty.variable, ty.predicate)
        if isinstance(ty, GenericType):
            return GenericType(
                self.apply(ty.base), tuple(self.apply(p) for p in ty.type_params)
            )
        if isinstance(ty, AwaitableType):
            return AwaitableType(self.apply(ty.result_type))
        if isinstance(ty, IterableType):
            return IterableType(self.apply(ty.yield_type))
        if isinstance(ty, GeneratorType):
            return GeneratorType(
                self.apply(ty.yield_type),
                self.apply(ty.send_type),
                self.apply(ty.return_type),
            )
        if isinstance(ty, CoroutineType):
            return CoroutineType(
                self.apply(ty.yield_type),
                self.apply(ty.send_type),
                self.apply(ty.return_type),
            )
        if isinstance(ty, ContextManagerType):
            return ContextManagerType(
                self.apply(ty.enter_type), self.apply(ty.exit_type)
            )
        return ty


def collect_types(ty: BaseType) -> List[BaseType]:
    """Collect all types appearing in a type tree."""
    result: List[BaseType] = [ty]
    transformer = TypeTransformer(lambda t: t)

    def collector(t: BaseType) -> BaseType:
        result.append(t)
        return t

    c = TypeTransformer(collector)
    c.apply(ty)
    return result


def type_depth(ty: BaseType) -> int:
    """Compute the depth of a type tree."""
    if isinstance(ty, (TopType, BottomType, IntType, FloatType, ComplexType,
                       BoolType, StrType, BytesType, NoneType_, TypeVarType, LiteralType)):
        return 0

    if isinstance(ty, (ListType, SetType, FrozenSetType)):
        return 1 + type_depth(ty.element_type)

    if isinstance(ty, TupleType):
        if not ty.element_types:
            return 0
        return 1 + max(type_depth(t) for t in ty.element_types)

    if isinstance(ty, DictType):
        return 1 + max(type_depth(ty.key_type), type_depth(ty.value_type))

    if isinstance(ty, CallableType):
        depths = [type_depth(p) for p in ty.param_types] + [type_depth(ty.return_type)]
        return 1 + max(depths) if depths else 1

    if isinstance(ty, UnionType):
        return 1 + max((type_depth(t) for t in ty.types), default=0)

    if isinstance(ty, IntersectionType):
        return 1 + max((type_depth(t) for t in ty.types), default=0)

    if isinstance(ty, RefinementType):
        return 1 + type_depth(ty.base_type)

    if isinstance(ty, GenericType):
        depths = [type_depth(ty.base)] + [type_depth(p) for p in ty.type_params]
        return 1 + max(depths)

    return 0


def type_size(ty: BaseType) -> int:
    """Count the number of nodes in a type tree."""
    return len(collect_types(ty))


def strip_refinements(ty: BaseType) -> BaseType:
    """Remove all refinements from a type."""

    def strip(t: BaseType) -> BaseType:
        if isinstance(t, RefinementType):
            return t.base_type
        return t

    return TypeTransformer(strip).apply(ty)


def simplify_unions(ty: BaseType) -> BaseType:
    """Simplify union types by removing duplicates and subsumed types."""

    def simplify(t: BaseType) -> BaseType:
        if isinstance(t, UnionType):
            unique = list(set(t.types))
            if len(unique) == 1:
                return unique[0]
            # Remove bottom types
            filtered = [u for u in unique if not isinstance(u, BottomType)]
            if not filtered:
                return BottomType()
            if any(isinstance(u, TopType) for u in filtered):
                return TopType()
            if len(filtered) == 1:
                return filtered[0]
            return UnionType(frozenset(filtered))
        return t

    return TypeTransformer(simplify).apply(ty)


# ===================================================================
# Exports
# ===================================================================

__all__ = [
    # Predicates
    "PredicateKind",
    "PredicateVar",
    "PredicateLiteral",
    "Predicate",
    # Variance
    "Variance",
    # Base types
    "BaseType",
    "TopType",
    "BottomType",
    "IntType",
    "FloatType",
    "ComplexType",
    "BoolType",
    "StrType",
    "BytesType",
    "NoneType_",
    "ListType",
    "TupleType",
    "SetType",
    "FrozenSetType",
    "DictType",
    "CallableType",
    "UnionType",
    "IntersectionType",
    "OptionalType",
    "StructuralType",
    "ClassType",
    "GenericType",
    "TypeVarType",
    "LiteralType",
    "ProtocolType",
    "TypeAliasType",
    "RecursiveType",
    "AwaitableType",
    "IterableType",
    "GeneratorType",
    "CoroutineType",
    "ContextManagerType",
    # Refinement types
    "RefinementType",
    "DependentParam",
    "DependentFunctionType",
    # Environment
    "TypeEnvironment",
    # Guards
    "GuardKind",
    "Guard",
    # Subtyping
    "SubtypeChecker",
    # Narrowing
    "TypeNarrower",
    # Join / Meet
    "TypeJoin",
    "TypeMeet",
    # Unification
    "Substitution",
    "UnificationError",
    "TypeUnifier",
    # Inference
    "ExprKind",
    "TypeExpr",
    "InferenceMode",
    "TypeConstraint",
    "TypeInferer",
    # Printing
    "TypePrinter",
    # Serialization
    "TypeSerializer",
    # Statistics
    "TypeStatistics",
    # Constructors
    "make_list",
    "make_dict",
    "make_tuple",
    "make_callable",
    "make_union",
    "make_intersection",
    "make_optional",
    "make_generic",
    "make_refinement",
    "make_dependent_fn",
    # Utilities
    "TypeVisitor",
    "TypeTransformer",
    "collect_types",
    "type_depth",
    "type_size",
    "strip_refinements",
    "simplify_unions",
]

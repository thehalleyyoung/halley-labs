"""
Python-native refinement types for the refinement type inference system.

Defines a Python-specific type hierarchy (PyType) with refinement predicates
(HeapPredicate) that can reference heap state, plus machinery for subtype
checking, type narrowing, and lattice operations on refined types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from src.heap.heap_model import HeapAddress, AbstractValue

# ═══════════════════════════════════════════════════════════════════════════
# 1.  HeapPredKind – kinds of heap-aware predicates
# ═══════════════════════════════════════════════════════════════════════════


class HeapPredKind(Enum):
    """Discriminant for the kind of heap predicate."""
    ATTR_EQ = auto()
    ATTR_TYPE = auto()
    ATTR_NONE = auto()
    ATTR_NOT_NONE = auto()
    CONTAINER_LEN = auto()
    CONTAINER_ELEM_TYPE = auto()
    DICT_KEY_EXISTS = auto()
    DICT_KEY_TYPE = auto()
    CALLABLE = auto()
    PROTOCOL_COMPLIANCE = auto()
    HEAP_PATH = auto()
    FROZEN = auto()
    ALIASED = auto()
    NOT_NONE = auto()
    IS_NONE = auto()
    ISINSTANCE = auto()
    TRUTHINESS = auto()
    COMPARISON = auto()
    NEGATION = auto()
    CONJUNCTION = auto()
    DISJUNCTION = auto()
    FORALL_ELEM = auto()
    EXISTS_ELEM = auto()
    SORTED = auto()
    UNIQUE_ELEMS = auto()


# ═══════════════════════════════════════════════════════════════════════════
# 2.  HeapPredicate – a predicate that can reference heap state
# ═══════════════════════════════════════════════════════════════════════════


_NEGATION_MAP: Dict[HeapPredKind, HeapPredKind] = {
    HeapPredKind.ATTR_NONE: HeapPredKind.ATTR_NOT_NONE,
    HeapPredKind.ATTR_NOT_NONE: HeapPredKind.ATTR_NONE,
    HeapPredKind.NOT_NONE: HeapPredKind.IS_NONE,
    HeapPredKind.IS_NONE: HeapPredKind.NOT_NONE,
}

_COMPARISON_NEGATION: Dict[str, str] = {
    "==": "!=",
    "!=": "==",
    "<": ">=",
    ">=": "<",
    ">": "<=",
    "<=": ">",
}


@dataclass(frozen=True)
class HeapPredicate:
    """A predicate that may reference heap state.

    Predicates are immutable value objects.  Compound predicates are built
    via ``negate``, ``and_pred`` and ``or_pred``.
    """

    kind: HeapPredKind
    variable: str
    path: Tuple[str, ...] = ()
    args: Tuple[Any, ...] = ()

    # ------------------------------------------------------------------
    # combinators
    # ------------------------------------------------------------------

    def negate(self) -> HeapPredicate:
        """Return the logical negation of this predicate."""
        if self.kind == HeapPredKind.NEGATION:
            inner: HeapPredicate = self.args[0]
            return inner
        if self.kind in _NEGATION_MAP:
            return HeapPredicate(
                kind=_NEGATION_MAP[self.kind],
                variable=self.variable,
                path=self.path,
                args=self.args,
            )
        if self.kind == HeapPredKind.COMPARISON and len(self.args) >= 2:
            op = self.args[0]
            neg_op = _COMPARISON_NEGATION.get(op)
            if neg_op is not None:
                return HeapPredicate(
                    kind=HeapPredKind.COMPARISON,
                    variable=self.variable,
                    path=self.path,
                    args=(neg_op,) + self.args[1:],
                )
        if self.kind == HeapPredKind.CONJUNCTION:
            left: HeapPredicate = self.args[0]
            right: HeapPredicate = self.args[1]
            return left.negate().or_pred(right.negate())
        if self.kind == HeapPredKind.DISJUNCTION:
            left = self.args[0]
            right = self.args[1]
            return left.negate().and_pred(right.negate())
        if self.kind == HeapPredKind.TRUTHINESS:
            return HeapPredicate(
                kind=HeapPredKind.TRUTHINESS,
                variable=self.variable,
                path=self.path,
                args=(not self.args[0],) if self.args else (False,),
            )
        return HeapPredicate(
            kind=HeapPredKind.NEGATION,
            variable=self.variable,
            path=self.path,
            args=(self,),
        )

    def and_pred(self, other: HeapPredicate) -> HeapPredicate:
        return HeapPredicate(
            kind=HeapPredKind.CONJUNCTION,
            variable=self.variable,
            path=(),
            args=(self, other),
        )

    def or_pred(self, other: HeapPredicate) -> HeapPredicate:
        return HeapPredicate(
            kind=HeapPredKind.DISJUNCTION,
            variable=self.variable,
            path=(),
            args=(self, other),
        )

    # ------------------------------------------------------------------
    # substitution & querying
    # ------------------------------------------------------------------

    def substitute(self, var_map: Dict[str, str]) -> HeapPredicate:
        """Rename variables according to *var_map*."""
        new_var = var_map.get(self.variable, self.variable)
        new_args: List[Any] = []
        for a in self.args:
            if isinstance(a, HeapPredicate):
                new_args.append(a.substitute(var_map))
            else:
                new_args.append(a)
        return HeapPredicate(
            kind=self.kind,
            variable=new_var,
            path=self.path,
            args=tuple(new_args),
        )

    def references_field(self, var: str, fld: str) -> bool:
        """Does this predicate mention *var.field*?"""
        if self.variable == var and self.path and self.path[0] == fld:
            return True
        for a in self.args:
            if isinstance(a, HeapPredicate) and a.references_field(var, fld):
                return True
        return False

    def references_path(self, target: Tuple[str, ...]) -> bool:
        full = (self.variable,) + self.path
        if len(full) >= len(target) and full[: len(target)] == target:
            return True
        for a in self.args:
            if isinstance(a, HeapPredicate) and a.references_path(target):
                return True
        return False

    def free_variables(self) -> Set[str]:
        result: Set[str] = {self.variable}
        for a in self.args:
            if isinstance(a, HeapPredicate):
                result |= a.free_variables()
        return result

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def evaluate(self, env: Dict[str, Any], heap: Any) -> Optional[bool]:
        """Best-effort evaluation in concrete *env* / *heap*.

        Returns ``None`` when the predicate cannot be evaluated.
        """
        val = _resolve_path(env, self.variable, self.path)
        if val is _MISSING:
            if self.kind == HeapPredKind.CONJUNCTION:
                left_val = self.args[0].evaluate(env, heap)
                right_val = self.args[1].evaluate(env, heap)
                if left_val is False or right_val is False:
                    return False
                if left_val is True and right_val is True:
                    return True
                return None
            if self.kind == HeapPredKind.DISJUNCTION:
                left_val = self.args[0].evaluate(env, heap)
                right_val = self.args[1].evaluate(env, heap)
                if left_val is True or right_val is True:
                    return True
                if left_val is False and right_val is False:
                    return False
                return None
            if self.kind == HeapPredKind.NEGATION:
                inner_val = self.args[0].evaluate(env, heap)
                return None if inner_val is None else not inner_val
            return None

        if self.kind == HeapPredKind.ATTR_EQ:
            return val == self.args[0] if self.args else None
        if self.kind == HeapPredKind.ATTR_TYPE:
            return type(val).__name__ == self.args[0] if self.args else None
        if self.kind == HeapPredKind.ATTR_NONE:
            return val is None
        if self.kind == HeapPredKind.ATTR_NOT_NONE:
            return val is not None
        if self.kind == HeapPredKind.NOT_NONE:
            return val is not None
        if self.kind == HeapPredKind.IS_NONE:
            return val is None
        if self.kind == HeapPredKind.ISINSTANCE:
            type_names: Tuple[str, ...] = self.args[0] if self.args else ()
            return type(val).__name__ in type_names
        if self.kind == HeapPredKind.TRUTHINESS:
            expected = self.args[0] if self.args else True
            return bool(val) == expected
        if self.kind == HeapPredKind.COMPARISON and len(self.args) >= 2:
            op, rhs = self.args[0], self.args[1]
            return _eval_comparison(val, op, rhs)
        if self.kind == HeapPredKind.CONTAINER_LEN and len(self.args) >= 2:
            try:
                length = len(val)  # type: ignore[arg-type]
            except TypeError:
                return None
            return _eval_comparison(length, self.args[0], self.args[1])
        if self.kind == HeapPredKind.DICT_KEY_EXISTS and self.args:
            try:
                return self.args[0] in val
            except TypeError:
                return None
        if self.kind == HeapPredKind.CALLABLE:
            return callable(val)
        if self.kind == HeapPredKind.FROZEN:
            return isinstance(val, (int, float, str, bytes, bool, tuple, frozenset))
        if self.kind == HeapPredKind.SORTED:
            try:
                lst = list(val)  # type: ignore[arg-type]
                return lst == sorted(lst)
            except TypeError:
                return None
        if self.kind == HeapPredKind.UNIQUE_ELEMS:
            try:
                lst = list(val)  # type: ignore[arg-type]
                return len(lst) == len(set(lst))
            except TypeError:
                return None
        if self.kind == HeapPredKind.CONJUNCTION:
            l_val = self.args[0].evaluate(env, heap)
            r_val = self.args[1].evaluate(env, heap)
            if l_val is False or r_val is False:
                return False
            if l_val is True and r_val is True:
                return True
            return None
        if self.kind == HeapPredKind.DISJUNCTION:
            l_val = self.args[0].evaluate(env, heap)
            r_val = self.args[1].evaluate(env, heap)
            if l_val is True or r_val is True:
                return True
            if l_val is False and r_val is False:
                return False
            return None
        if self.kind == HeapPredKind.NEGATION:
            inner_val = self.args[0].evaluate(env, heap)
            return None if inner_val is None else not inner_val
        if self.kind == HeapPredKind.FORALL_ELEM:
            elem_pred: HeapPredicate = self.args[0]
            try:
                items = list(val)  # type: ignore[arg-type]
            except TypeError:
                return None
            results: List[Optional[bool]] = []
            for item in items:
                sub_env = dict(env)
                sub_env["__elem__"] = item
                r = elem_pred.evaluate(sub_env, heap)
                if r is False:
                    return False
                results.append(r)
            if all(r is True for r in results):
                return True
            return None
        if self.kind == HeapPredKind.EXISTS_ELEM:
            elem_pred = self.args[0]
            try:
                items = list(val)  # type: ignore[arg-type]
            except TypeError:
                return None
            results = []
            for item in items:
                sub_env = dict(env)
                sub_env["__elem__"] = item
                r = elem_pred.evaluate(sub_env, heap)
                if r is True:
                    return True
                results.append(r)
            if all(r is False for r in results):
                return False
            return None
        return None

    # ------------------------------------------------------------------
    # simple implication
    # ------------------------------------------------------------------

    def implies(self, other: HeapPredicate) -> Optional[bool]:
        """Lightweight syntactic implication check.

        Returns ``True`` / ``False`` when decidable, else ``None``.
        """
        if self == other:
            return True
        if self.kind == HeapPredKind.IS_NONE and other.kind == HeapPredKind.IS_NONE:
            return self.variable == other.variable and self.path == other.path
        if self.kind == HeapPredKind.NOT_NONE and other.kind == HeapPredKind.NOT_NONE:
            return self.variable == other.variable and self.path == other.path
        if self.kind == HeapPredKind.NOT_NONE and other.kind == HeapPredKind.IS_NONE:
            if self.variable == other.variable and self.path == other.path:
                return False
        if self.kind == HeapPredKind.IS_NONE and other.kind == HeapPredKind.NOT_NONE:
            if self.variable == other.variable and self.path == other.path:
                return False
        if (
            self.kind == HeapPredKind.ISINSTANCE
            and other.kind == HeapPredKind.ISINSTANCE
            and self.variable == other.variable
            and self.path == other.path
        ):
            sub_types: FrozenSet[str] = frozenset(self.args[0]) if self.args else frozenset()
            sup_types: FrozenSet[str] = frozenset(other.args[0]) if other.args else frozenset()
            if sub_types <= sup_types:
                return True
        if (
            self.kind == HeapPredKind.COMPARISON
            and other.kind == HeapPredKind.COMPARISON
            and self.variable == other.variable
            and self.path == other.path
            and len(self.args) >= 2
            and len(other.args) >= 2
        ):
            s_op, s_val = self.args[0], self.args[1]
            o_op, o_val = other.args[0], other.args[1]
            if s_op == "<" and o_op == "<" and isinstance(s_val, (int, float)) and isinstance(o_val, (int, float)):
                if s_val <= o_val:
                    return True
            if s_op == ">" and o_op == ">" and isinstance(s_val, (int, float)) and isinstance(o_val, (int, float)):
                if s_val >= o_val:
                    return True
            if s_op == "==" and o_op in ("<=", ">=", "==") and s_val == o_val:
                return True
        if self.kind == HeapPredKind.CONJUNCTION:
            left_impl = self.args[0].implies(other)
            right_impl = self.args[1].implies(other)
            if left_impl is True or right_impl is True:
                return True
        if other.kind == HeapPredKind.DISJUNCTION:
            left_impl = self.implies(other.args[0])
            right_impl = self.implies(other.args[1])
            if left_impl is True or right_impl is True:
                return True
        return None

    # ------------------------------------------------------------------
    # pretty printing
    # ------------------------------------------------------------------

    def pretty(self) -> str:
        path_str = ".".join((self.variable,) + self.path) if self.path else self.variable
        if self.kind == HeapPredKind.ATTR_EQ:
            return f"{path_str} == {self.args[0]!r}"
        if self.kind == HeapPredKind.ATTR_TYPE:
            return f"type({path_str}) is {self.args[0]}"
        if self.kind == HeapPredKind.ATTR_NONE:
            return f"{path_str} is None"
        if self.kind == HeapPredKind.ATTR_NOT_NONE:
            return f"{path_str} is not None"
        if self.kind == HeapPredKind.NOT_NONE:
            return f"{path_str} is not None"
        if self.kind == HeapPredKind.IS_NONE:
            return f"{path_str} is None"
        if self.kind == HeapPredKind.ISINSTANCE:
            names = ", ".join(self.args[0]) if self.args else ""
            return f"isinstance({path_str}, ({names}))"
        if self.kind == HeapPredKind.TRUTHINESS:
            if self.args and not self.args[0]:
                return f"not {path_str}"
            return f"bool({path_str})"
        if self.kind == HeapPredKind.COMPARISON and len(self.args) >= 2:
            return f"{path_str} {self.args[0]} {self.args[1]!r}"
        if self.kind == HeapPredKind.CONTAINER_LEN and len(self.args) >= 2:
            return f"len({path_str}) {self.args[0]} {self.args[1]}"
        if self.kind == HeapPredKind.CONTAINER_ELEM_TYPE:
            return f"elem_type({path_str}) = {self.args[0]}"
        if self.kind == HeapPredKind.DICT_KEY_EXISTS:
            return f"{self.args[0]!r} in {path_str}"
        if self.kind == HeapPredKind.DICT_KEY_TYPE and len(self.args) >= 2:
            return f"type({path_str}[{self.args[0]!r}]) = {self.args[1]}"
        if self.kind == HeapPredKind.CALLABLE:
            return f"callable({path_str})"
        if self.kind == HeapPredKind.PROTOCOL_COMPLIANCE and len(self.args) >= 2:
            return f"{path_str} : Protocol[{self.args[0]}]"
        if self.kind == HeapPredKind.FROZEN:
            return f"frozen({path_str})"
        if self.kind == HeapPredKind.ALIASED:
            return f"aliased({path_str})"
        if self.kind == HeapPredKind.HEAP_PATH:
            return f"heap_path({path_str})"
        if self.kind == HeapPredKind.SORTED:
            return f"sorted({path_str})"
        if self.kind == HeapPredKind.UNIQUE_ELEMS:
            return f"unique({path_str})"
        if self.kind == HeapPredKind.FORALL_ELEM:
            inner = self.args[0].pretty() if self.args and isinstance(self.args[0], HeapPredicate) else "?"
            return f"∀ e ∈ {path_str}. {inner}"
        if self.kind == HeapPredKind.EXISTS_ELEM:
            inner = self.args[0].pretty() if self.args and isinstance(self.args[0], HeapPredicate) else "?"
            return f"∃ e ∈ {path_str}. {inner}"
        if self.kind == HeapPredKind.NEGATION:
            inner = self.args[0].pretty() if self.args and isinstance(self.args[0], HeapPredicate) else "?"
            return f"¬({inner})"
        if self.kind == HeapPredKind.CONJUNCTION:
            l = self.args[0].pretty() if isinstance(self.args[0], HeapPredicate) else "?"
            r = self.args[1].pretty() if isinstance(self.args[1], HeapPredicate) else "?"
            return f"({l}) ∧ ({r})"
        if self.kind == HeapPredKind.DISJUNCTION:
            l = self.args[0].pretty() if isinstance(self.args[0], HeapPredicate) else "?"
            r = self.args[1].pretty() if isinstance(self.args[1], HeapPredicate) else "?"
            return f"({l}) ∨ ({r})"
        return f"{self.kind.name}({path_str}, {self.args})"

    # ------------------------------------------------------------------
    # static constructors
    # ------------------------------------------------------------------

    @staticmethod
    def attr_eq(var: str, path: Tuple[str, ...], value: Any) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.ATTR_EQ, var, path, (value,))

    @staticmethod
    def attr_type(var: str, path: Tuple[str, ...], type_name: str) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.ATTR_TYPE, var, path, (type_name,))

    @staticmethod
    def attr_none(var: str, path: Tuple[str, ...]) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.ATTR_NONE, var, path)

    @staticmethod
    def attr_not_none(var: str, path: Tuple[str, ...]) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.ATTR_NOT_NONE, var, path)

    @staticmethod
    def not_none(var: str) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.NOT_NONE, var)

    @staticmethod
    def is_none(var: str) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.IS_NONE, var)

    @staticmethod
    def isinstance_pred(var: str, type_names: Tuple[str, ...]) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.ISINSTANCE, var, (), (type_names,))

    @staticmethod
    def container_len(var: str, op: str, n: int) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.CONTAINER_LEN, var, (), (op, n))

    @staticmethod
    def dict_key_exists(var: str, key: Any) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.DICT_KEY_EXISTS, var, (), (key,))

    @staticmethod
    def callable_pred(var: str) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.CALLABLE, var)

    @staticmethod
    def protocol_pred(
        var: str, protocol_name: str, required_attrs: Tuple[str, ...]
    ) -> HeapPredicate:
        return HeapPredicate(
            HeapPredKind.PROTOCOL_COMPLIANCE, var, (), (protocol_name, required_attrs)
        )

    @staticmethod
    def truthiness(var: str, expected: bool = True) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.TRUTHINESS, var, (), (expected,))

    @staticmethod
    def comparison(var: str, op: str, value: Any) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.COMPARISON, var, (), (op, value))

    @staticmethod
    def frozen_pred(var: str) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.FROZEN, var)

    @staticmethod
    def forall_elem(var: str, elem_pred: HeapPredicate) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.FORALL_ELEM, var, (), (elem_pred,))

    @staticmethod
    def sorted_pred(var: str) -> HeapPredicate:
        return HeapPredicate(HeapPredKind.SORTED, var)


# ── helpers ───────────────────────────────────────────────────────────────

_MISSING = object()


def _resolve_path(env: Dict[str, Any], var: str, path: Tuple[str, ...]) -> Any:
    """Walk *env[var].path[0].path[1]…* returning ``_MISSING`` on failure."""
    obj = env.get(var, _MISSING)
    if obj is _MISSING:
        return _MISSING
    for seg in path:
        if isinstance(obj, dict):
            obj = obj.get(seg, _MISSING)
        else:
            obj = getattr(obj, seg, _MISSING)
        if obj is _MISSING:
            return _MISSING
    return obj


def _eval_comparison(lhs: Any, op: str, rhs: Any) -> Optional[bool]:
    try:
        if op == "==":
            return lhs == rhs
        if op == "!=":
            return lhs != rhs
        if op == "<":
            return lhs < rhs
        if op == "<=":
            return lhs <= rhs
        if op == ">":
            return lhs > rhs
        if op == ">=":
            return lhs >= rhs
    except TypeError:
        return None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PyType – Python-specific type hierarchy
# ═══════════════════════════════════════════════════════════════════════════


class PyType(ABC):
    """Abstract base for all Python-specific types."""

    @abstractmethod
    def is_subtype_of(self, other: PyType) -> bool: ...

    @abstractmethod
    def join(self, other: PyType) -> PyType: ...

    @abstractmethod
    def meet(self, other: PyType) -> PyType: ...

    @abstractmethod
    def pretty(self) -> str: ...

    @abstractmethod
    def free_vars(self) -> Set[str]: ...

    @abstractmethod
    def substitute(self, mapping: Dict[str, PyType]) -> PyType: ...

    def __repr__(self) -> str:
        return self.pretty()


# ── concrete PyType subclasses ────────────────────────────────────────────


@dataclass(frozen=True)
class AnyType(PyType):
    """Top type – every type is a subtype of AnyType."""

    def is_subtype_of(self, other: PyType) -> bool:
        return isinstance(other, AnyType)

    def join(self, other: PyType) -> PyType:
        return AnyType()

    def meet(self, other: PyType) -> PyType:
        return other

    def pretty(self) -> str:
        return "Any"

    def free_vars(self) -> Set[str]:
        return set()

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return self


@dataclass(frozen=True)
class NeverType(PyType):
    """Bottom type – subtype of everything, inhabits nothing."""

    def is_subtype_of(self, other: PyType) -> bool:
        return True

    def join(self, other: PyType) -> PyType:
        return other

    def meet(self, other: PyType) -> PyType:
        return NeverType()

    def pretty(self) -> str:
        return "Never"

    def free_vars(self) -> Set[str]:
        return set()

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return self


@dataclass(frozen=True)
class NoneType(PyType):
    def is_subtype_of(self, other: PyType) -> bool:
        if isinstance(other, (AnyType, NoneType)):
            return True
        if isinstance(other, OptionalType):
            return True
        if isinstance(other, PyUnionType):
            return any(NoneType().is_subtype_of(m) for m in other.members)
        return False

    def join(self, other: PyType) -> PyType:
        if isinstance(other, NoneType):
            return self
        if isinstance(other, NeverType):
            return self
        return OptionalType(other)

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, (NoneType, AnyType)):
            return self
        if isinstance(other, OptionalType):
            return self
        if isinstance(other, PyUnionType) and any(isinstance(m, NoneType) for m in other.members):
            return self
        return NeverType()

    def pretty(self) -> str:
        return "None"

    def free_vars(self) -> Set[str]:
        return set()

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return self


# ── primitive types ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class IntPyType(PyType):
    def is_subtype_of(self, other: PyType) -> bool:
        return isinstance(other, (IntPyType, FloatPyType, AnyType)) or (
            isinstance(other, PyUnionType) and any(self.is_subtype_of(m) for m in other.members)
        )

    def join(self, other: PyType) -> PyType:
        if isinstance(other, (IntPyType, NeverType)):
            return self
        if isinstance(other, BoolPyType):
            return self
        if isinstance(other, FloatPyType):
            return FloatPyType()
        if isinstance(other, AnyType):
            return AnyType()
        return PyUnionType(frozenset({self, other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, (IntPyType, AnyType)):
            return self
        if isinstance(other, BoolPyType):
            return BoolPyType()
        if isinstance(other, PyUnionType) and any(self.is_subtype_of(m) for m in other.members):
            return self
        return NeverType()

    def pretty(self) -> str:
        return "int"

    def free_vars(self) -> Set[str]:
        return set()

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return self


@dataclass(frozen=True)
class FloatPyType(PyType):
    def is_subtype_of(self, other: PyType) -> bool:
        return isinstance(other, (FloatPyType, AnyType)) or (
            isinstance(other, PyUnionType) and any(self.is_subtype_of(m) for m in other.members)
        )

    def join(self, other: PyType) -> PyType:
        if isinstance(other, (FloatPyType, IntPyType, BoolPyType, NeverType)):
            return FloatPyType()
        if isinstance(other, AnyType):
            return AnyType()
        return PyUnionType(frozenset({self, other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, (FloatPyType, AnyType)):
            return self
        if isinstance(other, IntPyType):
            return IntPyType()
        if isinstance(other, PyUnionType) and any(self.is_subtype_of(m) for m in other.members):
            return self
        return NeverType()

    def pretty(self) -> str:
        return "float"

    def free_vars(self) -> Set[str]:
        return set()

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return self


@dataclass(frozen=True)
class BoolPyType(PyType):
    """bool is a subtype of int in Python."""

    def is_subtype_of(self, other: PyType) -> bool:
        return isinstance(other, (BoolPyType, IntPyType, FloatPyType, AnyType)) or (
            isinstance(other, PyUnionType) and any(self.is_subtype_of(m) for m in other.members)
        )

    def join(self, other: PyType) -> PyType:
        if isinstance(other, (BoolPyType, NeverType)):
            return self
        if isinstance(other, IntPyType):
            return IntPyType()
        if isinstance(other, FloatPyType):
            return FloatPyType()
        if isinstance(other, AnyType):
            return AnyType()
        return PyUnionType(frozenset({self, other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, (BoolPyType, IntPyType, AnyType)):
            return self
        if isinstance(other, PyUnionType) and any(self.is_subtype_of(m) for m in other.members):
            return self
        return NeverType()

    def pretty(self) -> str:
        return "bool"

    def free_vars(self) -> Set[str]:
        return set()

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return self


@dataclass(frozen=True)
class StrPyType(PyType):
    def is_subtype_of(self, other: PyType) -> bool:
        return isinstance(other, (StrPyType, AnyType)) or (
            isinstance(other, PyUnionType) and any(self.is_subtype_of(m) for m in other.members)
        )

    def join(self, other: PyType) -> PyType:
        if isinstance(other, (StrPyType, NeverType)):
            return self
        if isinstance(other, AnyType):
            return AnyType()
        return PyUnionType(frozenset({self, other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, (StrPyType, AnyType)):
            return self
        if isinstance(other, PyUnionType) and any(self.is_subtype_of(m) for m in other.members):
            return self
        return NeverType()

    def pretty(self) -> str:
        return "str"

    def free_vars(self) -> Set[str]:
        return set()

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return self


@dataclass(frozen=True)
class BytesPyType(PyType):
    def is_subtype_of(self, other: PyType) -> bool:
        return isinstance(other, (BytesPyType, AnyType)) or (
            isinstance(other, PyUnionType) and any(self.is_subtype_of(m) for m in other.members)
        )

    def join(self, other: PyType) -> PyType:
        if isinstance(other, (BytesPyType, NeverType)):
            return self
        if isinstance(other, AnyType):
            return AnyType()
        return PyUnionType(frozenset({self, other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, (BytesPyType, AnyType)):
            return self
        if isinstance(other, PyUnionType) and any(self.is_subtype_of(m) for m in other.members):
            return self
        return NeverType()

    def pretty(self) -> str:
        return "bytes"

    def free_vars(self) -> Set[str]:
        return set()

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return self


# ── generic / composite types ────────────────────────────────────────────


@dataclass(frozen=True)
class ClassType(PyType):
    class_addr: HeapAddress
    type_args: Tuple[PyType, ...] = ()

    def is_subtype_of(self, other: PyType) -> bool:
        if isinstance(other, AnyType):
            return True
        if isinstance(other, ClassType):
            if self.class_addr != other.class_addr:
                return False
            if len(self.type_args) != len(other.type_args):
                return False
            return all(a.is_subtype_of(b) for a, b in zip(self.type_args, other.type_args))
        if isinstance(other, PyUnionType):
            return any(self.is_subtype_of(m) for m in other.members)
        return False

    def join(self, other: PyType) -> PyType:
        if isinstance(other, NeverType):
            return self
        if isinstance(other, AnyType):
            return AnyType()
        if isinstance(other, ClassType) and self.class_addr == other.class_addr:
            if len(self.type_args) == len(other.type_args):
                joined_args = tuple(a.join(b) for a, b in zip(self.type_args, other.type_args))
                return ClassType(self.class_addr, joined_args)
        return PyUnionType(frozenset({self, other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, AnyType):
            return self
        if isinstance(other, ClassType) and self.class_addr == other.class_addr:
            if len(self.type_args) == len(other.type_args):
                met_args = tuple(a.meet(b) for a, b in zip(self.type_args, other.type_args))
                return ClassType(self.class_addr, met_args)
        if isinstance(other, PyUnionType) and any(self.is_subtype_of(m) for m in other.members):
            return self
        return NeverType()

    def pretty(self) -> str:
        if not self.type_args:
            return f"Class[{self.class_addr}]"
        args_str = ", ".join(a.pretty() for a in self.type_args)
        return f"Class[{self.class_addr}][{args_str}]"

    def free_vars(self) -> Set[str]:
        result: Set[str] = set()
        for a in self.type_args:
            result |= a.free_vars()
        return result

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return ClassType(self.class_addr, tuple(a.substitute(mapping) for a in self.type_args))


@dataclass(frozen=True)
class ProtocolType(PyType):
    required_attrs: Dict[str, PyType] = field(default_factory=dict)
    required_methods: Dict[str, FunctionPyType] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((
            tuple(sorted(self.required_attrs.items(), key=lambda kv: kv[0])),
            tuple(sorted(self.required_methods.items(), key=lambda kv: kv[0])),
        ))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProtocolType):
            return NotImplemented
        return self.required_attrs == other.required_attrs and self.required_methods == other.required_methods

    def is_subtype_of(self, other: PyType) -> bool:
        if isinstance(other, AnyType):
            return True
        if isinstance(other, ProtocolType):
            for attr, typ in other.required_attrs.items():
                if attr not in self.required_attrs:
                    return False
                if not self.required_attrs[attr].is_subtype_of(typ):
                    return False
            for meth, ftyp in other.required_methods.items():
                if meth not in self.required_methods:
                    return False
                if not self.required_methods[meth].is_subtype_of(ftyp):
                    return False
            return True
        if isinstance(other, PyUnionType):
            return any(self.is_subtype_of(m) for m in other.members)
        return False

    def join(self, other: PyType) -> PyType:
        if isinstance(other, NeverType):
            return self
        if isinstance(other, AnyType):
            return AnyType()
        if isinstance(other, ProtocolType):
            common_attrs: Dict[str, PyType] = {}
            for k in self.required_attrs:
                if k in other.required_attrs:
                    common_attrs[k] = self.required_attrs[k].join(other.required_attrs[k])
            common_methods: Dict[str, FunctionPyType] = {}
            for k in self.required_methods:
                if k in other.required_methods:
                    joined = self.required_methods[k].meet(other.required_methods[k])
                    if isinstance(joined, FunctionPyType):
                        common_methods[k] = joined
            return ProtocolType(common_attrs, common_methods)
        return PyUnionType(frozenset({self, other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, AnyType):
            return self
        if isinstance(other, ProtocolType):
            merged_attrs = dict(self.required_attrs)
            for k, v in other.required_attrs.items():
                if k in merged_attrs:
                    merged_attrs[k] = merged_attrs[k].meet(v)
                else:
                    merged_attrs[k] = v
            merged_methods = dict(self.required_methods)
            for k, v in other.required_methods.items():
                if k in merged_methods:
                    joined = merged_methods[k].join(v)
                    if isinstance(joined, FunctionPyType):
                        merged_methods[k] = joined
                else:
                    merged_methods[k] = v
            return ProtocolType(merged_attrs, merged_methods)
        return NeverType()

    def pretty(self) -> str:
        parts: List[str] = []
        for k, v in sorted(self.required_attrs.items()):
            parts.append(f"{k}: {v.pretty()}")
        for k, v in sorted(self.required_methods.items()):
            parts.append(f"{k}: {v.pretty()}")
        return "Protocol{" + ", ".join(parts) + "}"

    def free_vars(self) -> Set[str]:
        result: Set[str] = set()
        for v in self.required_attrs.values():
            result |= v.free_vars()
        for v in self.required_methods.values():
            result |= v.free_vars()
        return result

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        new_attrs = {k: v.substitute(mapping) for k, v in self.required_attrs.items()}
        new_methods: Dict[str, FunctionPyType] = {}
        for k, v in self.required_methods.items():
            sub = v.substitute(mapping)
            if isinstance(sub, FunctionPyType):
                new_methods[k] = sub
        return ProtocolType(new_attrs, new_methods)


@dataclass(frozen=True)
class PyUnionType(PyType):
    members: FrozenSet[PyType]

    def is_subtype_of(self, other: PyType) -> bool:
        if isinstance(other, AnyType):
            return True
        return all(m.is_subtype_of(other) for m in self.members)

    def join(self, other: PyType) -> PyType:
        if isinstance(other, NeverType):
            return self
        if isinstance(other, AnyType):
            return AnyType()
        if isinstance(other, PyUnionType):
            return _simplify_union(self.members | other.members)
        return _simplify_union(self.members | frozenset({other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, AnyType):
            return self
        if isinstance(other, PyUnionType):
            result_members: Set[PyType] = set()
            for m1 in self.members:
                for m2 in other.members:
                    met = m1.meet(m2)
                    if not isinstance(met, NeverType):
                        result_members.add(met)
            if not result_members:
                return NeverType()
            return _simplify_union(frozenset(result_members))
        result_members = set()
        for m in self.members:
            met = m.meet(other)
            if not isinstance(met, NeverType):
                result_members.add(met)
        if not result_members:
            return NeverType()
        return _simplify_union(frozenset(result_members))

    def pretty(self) -> str:
        return " | ".join(sorted(m.pretty() for m in self.members))

    def free_vars(self) -> Set[str]:
        result: Set[str] = set()
        for m in self.members:
            result |= m.free_vars()
        return result

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return PyUnionType(frozenset(m.substitute(mapping) for m in self.members))


@dataclass(frozen=True)
class PyIntersectionType(PyType):
    members: FrozenSet[PyType]

    def is_subtype_of(self, other: PyType) -> bool:
        if isinstance(other, AnyType):
            return True
        return any(m.is_subtype_of(other) for m in self.members)

    def join(self, other: PyType) -> PyType:
        if isinstance(other, NeverType):
            return self
        if isinstance(other, AnyType):
            return AnyType()
        result_members: Set[PyType] = set()
        for m in self.members:
            result_members.add(m.join(other))
        if not result_members:
            return other
        return _simplify_union(frozenset(result_members))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, AnyType):
            return self
        if isinstance(other, PyIntersectionType):
            return PyIntersectionType(self.members | other.members)
        return PyIntersectionType(self.members | frozenset({other}))

    def pretty(self) -> str:
        return " & ".join(sorted(m.pretty() for m in self.members))

    def free_vars(self) -> Set[str]:
        result: Set[str] = set()
        for m in self.members:
            result |= m.free_vars()
        return result

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return PyIntersectionType(frozenset(m.substitute(mapping) for m in self.members))


@dataclass(frozen=True)
class NarrowedType(PyType):
    """A type narrowed by a heap predicate guard."""
    original: PyType
    guard: HeapPredicate
    narrowed_to: PyType

    def is_subtype_of(self, other: PyType) -> bool:
        return self.narrowed_to.is_subtype_of(other)

    def join(self, other: PyType) -> PyType:
        if isinstance(other, NarrowedType) and other.guard == self.guard:
            return NarrowedType(self.original.join(other.original), self.guard, self.narrowed_to.join(other.narrowed_to))
        return self.original.join(other)

    def meet(self, other: PyType) -> PyType:
        return self.narrowed_to.meet(other)

    def pretty(self) -> str:
        return f"({self.narrowed_to.pretty()} when {self.guard.pretty()})"

    def free_vars(self) -> Set[str]:
        return self.original.free_vars() | self.narrowed_to.free_vars()

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return NarrowedType(
            self.original.substitute(mapping),
            self.guard,
            self.narrowed_to.substitute(mapping),
        )


@dataclass(frozen=True)
class OptionalType(PyType):
    """Shorthand for Union[inner, None]."""
    inner: PyType

    def is_subtype_of(self, other: PyType) -> bool:
        if isinstance(other, AnyType):
            return True
        if isinstance(other, OptionalType):
            return self.inner.is_subtype_of(other.inner)
        if isinstance(other, PyUnionType):
            return NoneType().is_subtype_of(other) and self.inner.is_subtype_of(other)
        return False

    def join(self, other: PyType) -> PyType:
        if isinstance(other, NeverType):
            return self
        if isinstance(other, NoneType):
            return self
        if isinstance(other, AnyType):
            return AnyType()
        if isinstance(other, OptionalType):
            return OptionalType(self.inner.join(other.inner))
        return OptionalType(self.inner.join(other))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, AnyType):
            return self
        if isinstance(other, NoneType):
            return NoneType()
        if isinstance(other, OptionalType):
            met = self.inner.meet(other.inner)
            if isinstance(met, NeverType):
                return NoneType()
            return OptionalType(met)
        return self.inner.meet(other)

    def pretty(self) -> str:
        return f"Optional[{self.inner.pretty()}]"

    def free_vars(self) -> Set[str]:
        return self.inner.free_vars()

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return OptionalType(self.inner.substitute(mapping))


# ── container types ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ListPyType(PyType):
    element: PyType

    def is_subtype_of(self, other: PyType) -> bool:
        if isinstance(other, AnyType):
            return True
        if isinstance(other, ListPyType):
            return self.element.is_subtype_of(other.element)
        if isinstance(other, PyUnionType):
            return any(self.is_subtype_of(m) for m in other.members)
        return False

    def join(self, other: PyType) -> PyType:
        if isinstance(other, NeverType):
            return self
        if isinstance(other, AnyType):
            return AnyType()
        if isinstance(other, ListPyType):
            return ListPyType(self.element.join(other.element))
        return PyUnionType(frozenset({self, other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, AnyType):
            return self
        if isinstance(other, ListPyType):
            return ListPyType(self.element.meet(other.element))
        return NeverType()

    def pretty(self) -> str:
        return f"list[{self.element.pretty()}]"

    def free_vars(self) -> Set[str]:
        return self.element.free_vars()

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return ListPyType(self.element.substitute(mapping))


@dataclass(frozen=True)
class DictPyType(PyType):
    key: PyType
    value: PyType

    def is_subtype_of(self, other: PyType) -> bool:
        if isinstance(other, AnyType):
            return True
        if isinstance(other, DictPyType):
            return self.key.is_subtype_of(other.key) and self.value.is_subtype_of(other.value)
        if isinstance(other, PyUnionType):
            return any(self.is_subtype_of(m) for m in other.members)
        return False

    def join(self, other: PyType) -> PyType:
        if isinstance(other, NeverType):
            return self
        if isinstance(other, AnyType):
            return AnyType()
        if isinstance(other, DictPyType):
            return DictPyType(self.key.join(other.key), self.value.join(other.value))
        return PyUnionType(frozenset({self, other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, AnyType):
            return self
        if isinstance(other, DictPyType):
            return DictPyType(self.key.meet(other.key), self.value.meet(other.value))
        return NeverType()

    def pretty(self) -> str:
        return f"dict[{self.key.pretty()}, {self.value.pretty()}]"

    def free_vars(self) -> Set[str]:
        return self.key.free_vars() | self.value.free_vars()

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return DictPyType(self.key.substitute(mapping), self.value.substitute(mapping))


@dataclass(frozen=True)
class SetPyType(PyType):
    element: PyType

    def is_subtype_of(self, other: PyType) -> bool:
        if isinstance(other, AnyType):
            return True
        if isinstance(other, SetPyType):
            return self.element.is_subtype_of(other.element)
        if isinstance(other, PyUnionType):
            return any(self.is_subtype_of(m) for m in other.members)
        return False

    def join(self, other: PyType) -> PyType:
        if isinstance(other, NeverType):
            return self
        if isinstance(other, AnyType):
            return AnyType()
        if isinstance(other, SetPyType):
            return SetPyType(self.element.join(other.element))
        return PyUnionType(frozenset({self, other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, AnyType):
            return self
        if isinstance(other, SetPyType):
            return SetPyType(self.element.meet(other.element))
        return NeverType()

    def pretty(self) -> str:
        return f"set[{self.element.pretty()}]"

    def free_vars(self) -> Set[str]:
        return self.element.free_vars()

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return SetPyType(self.element.substitute(mapping))


@dataclass(frozen=True)
class TuplePyType(PyType):
    elements: Tuple[PyType, ...]
    is_variable_length: bool = False

    def is_subtype_of(self, other: PyType) -> bool:
        if isinstance(other, AnyType):
            return True
        if isinstance(other, TuplePyType):
            if self.is_variable_length and other.is_variable_length:
                if not self.elements or not other.elements:
                    return len(self.elements) == 0 or len(other.elements) == 0
                return self.elements[0].is_subtype_of(other.elements[0])
            if not self.is_variable_length and not other.is_variable_length:
                if len(self.elements) != len(other.elements):
                    return False
                return all(a.is_subtype_of(b) for a, b in zip(self.elements, other.elements))
            if not self.is_variable_length and other.is_variable_length and other.elements:
                return all(e.is_subtype_of(other.elements[0]) for e in self.elements)
            return False
        if isinstance(other, PyUnionType):
            return any(self.is_subtype_of(m) for m in other.members)
        return False

    def join(self, other: PyType) -> PyType:
        if isinstance(other, NeverType):
            return self
        if isinstance(other, AnyType):
            return AnyType()
        if isinstance(other, TuplePyType):
            if self.is_variable_length or other.is_variable_length:
                all_elems = self.elements + other.elements
                if all_elems:
                    joined = all_elems[0]
                    for e in all_elems[1:]:
                        joined = joined.join(e)
                    return TuplePyType((joined,), True)
                return TuplePyType((), True)
            if len(self.elements) == len(other.elements):
                return TuplePyType(
                    tuple(a.join(b) for a, b in zip(self.elements, other.elements))
                )
            all_elems = self.elements + other.elements
            if all_elems:
                joined = all_elems[0]
                for e in all_elems[1:]:
                    joined = joined.join(e)
                return TuplePyType((joined,), True)
            return TuplePyType((), True)
        return PyUnionType(frozenset({self, other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, AnyType):
            return self
        if isinstance(other, TuplePyType):
            if not self.is_variable_length and not other.is_variable_length:
                if len(self.elements) != len(other.elements):
                    return NeverType()
                return TuplePyType(
                    tuple(a.meet(b) for a, b in zip(self.elements, other.elements))
                )
            if self.is_variable_length and not other.is_variable_length and self.elements:
                return TuplePyType(
                    tuple(self.elements[0].meet(e) for e in other.elements)
                )
            if not self.is_variable_length and other.is_variable_length and other.elements:
                return TuplePyType(
                    tuple(e.meet(other.elements[0]) for e in self.elements)
                )
            if self.elements and other.elements:
                return TuplePyType((self.elements[0].meet(other.elements[0]),), True)
            return TuplePyType((), True)
        return NeverType()

    def pretty(self) -> str:
        if self.is_variable_length:
            if self.elements:
                return f"tuple[{self.elements[0].pretty()}, ...]"
            return "tuple[()]"
        elems = ", ".join(e.pretty() for e in self.elements)
        return f"tuple[{elems}]"

    def free_vars(self) -> Set[str]:
        result: Set[str] = set()
        for e in self.elements:
            result |= e.free_vars()
        return result

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return TuplePyType(
            tuple(e.substitute(mapping) for e in self.elements),
            self.is_variable_length,
        )


# ── function type ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FunctionPyType(PyType):
    """A callable type with pre/post conditions and effect information."""

    params: Tuple[Tuple[str, PyRefinementType], ...]
    varargs: Optional[PyRefinementType] = None
    kwargs: Optional[PyRefinementType] = None
    return_type: PyRefinementType = field(default_factory=lambda: PyRefinementType(AnyType()))
    raises: FrozenSet[str] = frozenset()
    pre_conditions: Tuple[HeapPredicate, ...] = ()
    post_conditions: Tuple[HeapPredicate, ...] = ()
    frame: FrozenSet[str] = frozenset()
    is_generator: bool = False
    is_async: bool = False

    def is_subtype_of(self, other: PyType) -> bool:
        if isinstance(other, AnyType):
            return True
        if not isinstance(other, FunctionPyType):
            if isinstance(other, PyUnionType):
                return any(self.is_subtype_of(m) for m in other.members)
            return False
        if len(self.params) != len(other.params):
            return False
        # contravariant in params
        for (_, s_ty), (_, o_ty) in zip(self.params, other.params):
            if not o_ty.is_subtype_of(s_ty):
                return False
        # covariant in return
        if not self.return_type.is_subtype_of(other.return_type):
            return False
        # sub must not raise more exceptions
        if not self.raises <= other.raises:
            return False
        return True

    def join(self, other: PyType) -> PyType:
        if isinstance(other, NeverType):
            return self
        if isinstance(other, AnyType):
            return AnyType()
        if isinstance(other, FunctionPyType) and len(self.params) == len(other.params):
            # contravariant meet for params, covariant join for return
            new_params: List[Tuple[str, PyRefinementType]] = []
            for (n1, t1), (n2, t2) in zip(self.params, other.params):
                new_params.append((n1, t1.meet(t2)))
            new_ret = self.return_type.join(other.return_type)
            return FunctionPyType(
                params=tuple(new_params),
                varargs=self.varargs,
                kwargs=self.kwargs,
                return_type=new_ret,
                raises=self.raises | other.raises,
                pre_conditions=(),
                post_conditions=(),
                frame=self.frame | other.frame,
                is_generator=self.is_generator or other.is_generator,
                is_async=self.is_async or other.is_async,
            )
        return PyUnionType(frozenset({self, other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, AnyType):
            return self
        if isinstance(other, FunctionPyType) and len(self.params) == len(other.params):
            new_params: List[Tuple[str, PyRefinementType]] = []
            for (n1, t1), (n2, t2) in zip(self.params, other.params):
                new_params.append((n1, t1.join(t2)))
            new_ret = self.return_type.meet(other.return_type)
            return FunctionPyType(
                params=tuple(new_params),
                varargs=self.varargs,
                kwargs=self.kwargs,
                return_type=new_ret,
                raises=self.raises & other.raises,
                pre_conditions=self.pre_conditions + other.pre_conditions,
                post_conditions=self.post_conditions + other.post_conditions,
                frame=self.frame & other.frame,
                is_generator=self.is_generator and other.is_generator,
                is_async=self.is_async and other.is_async,
            )
        return NeverType()

    def pretty(self) -> str:
        parts: List[str] = []
        for name, rty in self.params:
            parts.append(f"{name}: {rty.pretty()}")
        if self.varargs:
            parts.append(f"*args: {self.varargs.pretty()}")
        if self.kwargs:
            parts.append(f"**kwargs: {self.kwargs.pretty()}")
        params_str = ", ".join(parts)
        prefix = ""
        if self.is_async:
            prefix = "async "
        if self.is_generator:
            prefix += "gen "
        ret = self.return_type.pretty()
        s = f"{prefix}({params_str}) -> {ret}"
        if self.raises:
            s += f" raises {{{', '.join(sorted(self.raises))}}}"
        if self.pre_conditions:
            pres = " ∧ ".join(p.pretty() for p in self.pre_conditions)
            s += f" pre({pres})"
        if self.post_conditions:
            posts = " ∧ ".join(p.pretty() for p in self.post_conditions)
            s += f" post({posts})"
        if self.frame:
            s += f" frame({', '.join(sorted(self.frame))})"
        return s

    def free_vars(self) -> Set[str]:
        result: Set[str] = set()
        for _, rty in self.params:
            result |= rty.base.free_vars()
        if self.varargs:
            result |= self.varargs.base.free_vars()
        if self.kwargs:
            result |= self.kwargs.base.free_vars()
        result |= self.return_type.base.free_vars()
        return result

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        new_params = tuple(
            (n, PyRefinementType(t.base.substitute(mapping), t.predicates))
            for n, t in self.params
        )
        new_va = PyRefinementType(self.varargs.base.substitute(mapping), self.varargs.predicates) if self.varargs else None
        new_kw = PyRefinementType(self.kwargs.base.substitute(mapping), self.kwargs.predicates) if self.kwargs else None
        new_ret = PyRefinementType(self.return_type.base.substitute(mapping), self.return_type.predicates)
        return FunctionPyType(
            params=new_params,
            varargs=new_va,
            kwargs=new_kw,
            return_type=new_ret,
            raises=self.raises,
            pre_conditions=self.pre_conditions,
            post_conditions=self.post_conditions,
            frame=self.frame,
            is_generator=self.is_generator,
            is_async=self.is_async,
        )


# ── type variable ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TypeVarType(PyType):
    name: str
    bound: Optional[PyType] = None

    def is_subtype_of(self, other: PyType) -> bool:
        if isinstance(other, AnyType):
            return True
        if isinstance(other, TypeVarType) and self.name == other.name:
            return True
        if self.bound is not None:
            return self.bound.is_subtype_of(other)
        return False

    def join(self, other: PyType) -> PyType:
        if isinstance(other, NeverType):
            return self
        if isinstance(other, AnyType):
            return AnyType()
        if isinstance(other, TypeVarType) and self.name == other.name:
            return self
        if self.bound is not None:
            return self.bound.join(other)
        return PyUnionType(frozenset({self, other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, AnyType):
            return self
        if isinstance(other, TypeVarType) and self.name == other.name:
            return self
        if self.bound is not None:
            return self.bound.meet(other)
        return NeverType()

    def pretty(self) -> str:
        if self.bound:
            return f"TypeVar({self.name!r}, bound={self.bound.pretty()})"
        return f"TypeVar({self.name!r})"

    def free_vars(self) -> Set[str]:
        result: Set[str] = {self.name}
        if self.bound:
            result |= self.bound.free_vars()
        return result

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        if self.name in mapping:
            return mapping[self.name]
        if self.bound:
            return TypeVarType(self.name, self.bound.substitute(mapping))
        return self


@dataclass(frozen=True)
class LiteralType(PyType):
    """Singleton literal type: Literal[3], Literal["foo"], etc."""
    value: Any

    def is_subtype_of(self, other: PyType) -> bool:
        if isinstance(other, AnyType):
            return True
        if isinstance(other, LiteralType):
            return self.value == other.value
        base = _literal_to_base(self.value)
        if base is not None:
            return base.is_subtype_of(other)
        return False

    def join(self, other: PyType) -> PyType:
        if isinstance(other, NeverType):
            return self
        if isinstance(other, AnyType):
            return AnyType()
        if isinstance(other, LiteralType) and self.value == other.value:
            return self
        base = _literal_to_base(self.value)
        if base is not None:
            return base.join(other)
        return PyUnionType(frozenset({self, other}))

    def meet(self, other: PyType) -> PyType:
        if isinstance(other, AnyType):
            return self
        if isinstance(other, LiteralType):
            if self.value == other.value:
                return self
            return NeverType()
        base = _literal_to_base(self.value)
        if base is not None and base.is_subtype_of(other):
            return self
        return NeverType()

    def pretty(self) -> str:
        return f"Literal[{self.value!r}]"

    def free_vars(self) -> Set[str]:
        return set()

    def substitute(self, mapping: Dict[str, PyType]) -> PyType:
        return self


def _literal_to_base(value: Any) -> Optional[PyType]:
    if isinstance(value, bool):
        return BoolPyType()
    if isinstance(value, int):
        return IntPyType()
    if isinstance(value, float):
        return FloatPyType()
    if isinstance(value, str):
        return StrPyType()
    if isinstance(value, bytes):
        return BytesPyType()
    return None


def _simplify_union(members: FrozenSet[PyType]) -> PyType:
    """Remove redundant members from a union."""
    filtered: Set[PyType] = set()
    mlist = list(members)
    for i, m in enumerate(mlist):
        if isinstance(m, NeverType):
            continue
        if isinstance(m, AnyType):
            return AnyType()
        subsumed = False
        for j, other in enumerate(mlist):
            if i != j and not isinstance(other, NeverType) and m.is_subtype_of(other) and m != other:
                subsumed = True
                break
        if not subsumed:
            filtered.add(m)
    if not filtered:
        return NeverType()
    if len(filtered) == 1:
        return next(iter(filtered))
    return PyUnionType(frozenset(filtered))


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Helper: name-to-PyType mapping
# ═══════════════════════════════════════════════════════════════════════════

_NAME_TO_PYTYPE: Dict[str, PyType] = {
    "int": IntPyType(),
    "float": FloatPyType(),
    "bool": BoolPyType(),
    "str": StrPyType(),
    "bytes": BytesPyType(),
    "None": NoneType(),
    "NoneType": NoneType(),
    "list": ListPyType(AnyType()),
    "dict": DictPyType(AnyType(), AnyType()),
    "set": SetPyType(AnyType()),
    "tuple": TuplePyType((), True),
}


def name_to_pytype(name: str) -> Optional[PyType]:
    """Convert a Python type name to a PyType, or ``None``."""
    return _NAME_TO_PYTYPE.get(name)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  PyRefinementType – the main refinement type
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PyRefinementType:
    """A base PyType refined with a conjunction of HeapPredicates."""

    base: PyType
    predicates: Tuple[HeapPredicate, ...] = ()

    # ------------------------------------------------------------------
    # builders
    # ------------------------------------------------------------------

    def with_predicate(self, pred: HeapPredicate) -> PyRefinementType:
        return PyRefinementType(self.base, self.predicates + (pred,))

    def without_predicate(self, pred_kind: HeapPredKind) -> PyRefinementType:
        kept = tuple(p for p in self.predicates if p.kind != pred_kind)
        return PyRefinementType(self.base, kept)

    def narrow(self, guard: HeapPredicate) -> PyRefinementType:
        """Narrow the type with *guard*, adding the predicate."""
        return PyRefinementType(self.base, self.predicates + (guard,))

    def widen(self) -> PyRefinementType:
        """Drop all predicates, keeping only the base type."""
        return PyRefinementType(self.base)

    # ------------------------------------------------------------------
    # lattice operations
    # ------------------------------------------------------------------

    def join(self, other: PyRefinementType) -> PyRefinementType:
        joined_base = self.base.join(other.base)
        # keep only predicates present in both
        common: List[HeapPredicate] = []
        for p in self.predicates:
            if p in other.predicates:
                common.append(p)
        return PyRefinementType(joined_base, tuple(common))

    def meet(self, other: PyRefinementType) -> PyRefinementType:
        met_base = self.base.meet(other.base)
        # union of predicates (conjunction becomes stronger)
        combined = self.predicates + tuple(p for p in other.predicates if p not in self.predicates)
        return PyRefinementType(met_base, combined)

    def is_subtype_of(self, other: PyRefinementType) -> bool:
        if not self.base.is_subtype_of(other.base):
            return False
        # every predicate in other must be implied by self's predicates
        for op in other.predicates:
            implied = False
            for sp in self.predicates:
                result = sp.implies(op)
                if result is True:
                    implied = True
                    break
            if not implied:
                # also check syntactic equality
                if op in self.predicates:
                    implied = True
                if not implied:
                    return False
        return True

    # ------------------------------------------------------------------
    # invalidation
    # ------------------------------------------------------------------

    def invalidate_field(self, field_name: str) -> PyRefinementType:
        """Remove predicates that reference *field_name* on any variable."""
        kept: List[HeapPredicate] = []
        for p in self.predicates:
            # check if any segment of the path equals field_name
            mentions = False
            if p.path and field_name in p.path:
                mentions = True
            for a in p.args:
                if isinstance(a, HeapPredicate) and a.references_field(p.variable, field_name):
                    mentions = True
            if not mentions:
                kept.append(p)
        return PyRefinementType(self.base, tuple(kept))

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def evaluate_predicates(self, env: Dict[str, Any], heap: Any) -> Optional[bool]:
        """Evaluate all predicates conjunctively."""
        if not self.predicates:
            return True
        for p in self.predicates:
            r = p.evaluate(env, heap)
            if r is False:
                return False
            if r is None:
                return None
        return True

    # ------------------------------------------------------------------
    # pretty printing
    # ------------------------------------------------------------------

    def pretty(self) -> str:
        base_str = self.base.pretty()
        if not self.predicates:
            return base_str
        preds_str = " ∧ ".join(p.pretty() for p in self.predicates)
        return f"{{{base_str} | {preds_str}}}"

    # ------------------------------------------------------------------
    # static constructors
    # ------------------------------------------------------------------

    @staticmethod
    def simple(base: PyType) -> PyRefinementType:
        return PyRefinementType(base)

    @staticmethod
    def refined(base: PyType, *preds: HeapPredicate) -> PyRefinementType:
        return PyRefinementType(base, preds)

    @staticmethod
    def none_type() -> PyRefinementType:
        return PyRefinementType(NoneType())

    @staticmethod
    def non_none(base: PyType) -> PyRefinementType:
        return PyRefinementType(base, (HeapPredicate.not_none("v"),))

    @staticmethod
    def positive_int() -> PyRefinementType:
        return PyRefinementType(
            IntPyType(),
            (HeapPredicate.comparison("v", ">", 0),),
        )

    @staticmethod
    def non_empty_str() -> PyRefinementType:
        return PyRefinementType(
            StrPyType(),
            (HeapPredicate.container_len("v", ">", 0),),
        )

    @staticmethod
    def non_empty_list(elem: PyType) -> PyRefinementType:
        return PyRefinementType(
            ListPyType(elem),
            (HeapPredicate.container_len("v", ">", 0),),
        )

    @staticmethod
    def dict_with_key(
        key_type: PyType, val_type: PyType, required_key: Any
    ) -> PyRefinementType:
        return PyRefinementType(
            DictPyType(key_type, val_type),
            (HeapPredicate.dict_key_exists("v", required_key),),
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6.  RefinementSubtyping – subtype checking with refinements
# ═══════════════════════════════════════════════════════════════════════════


class RefinementSubtyping:
    """Static methods for refined-type lattice operations."""

    @staticmethod
    def is_subtype(sub: PyRefinementType, sup: PyRefinementType) -> bool:
        if not RefinementSubtyping.is_base_subtype(sub.base, sup.base):
            return False
        return RefinementSubtyping.predicates_imply(sub.predicates, sup.predicates)

    @staticmethod
    def is_base_subtype(sub: PyType, sup: PyType) -> bool:
        return sub.is_subtype_of(sup)

    @staticmethod
    def predicates_imply(
        sub_preds: Tuple[HeapPredicate, ...],
        sup_preds: Tuple[HeapPredicate, ...],
    ) -> bool:
        """Do *sub_preds* (conjunctively) imply every predicate in *sup_preds*?"""
        for sp in sup_preds:
            found = False
            if sp in sub_preds:
                found = True
            else:
                for sub_p in sub_preds:
                    result = sub_p.implies(sp)
                    if result is True:
                        found = True
                        break
            if not found:
                return False
        return True

    @staticmethod
    def join_types(t1: PyRefinementType, t2: PyRefinementType) -> PyRefinementType:
        return t1.join(t2)

    @staticmethod
    def meet_types(t1: PyRefinementType, t2: PyRefinementType) -> PyRefinementType:
        return t1.meet(t2)

    @staticmethod
    def widen_type(old: PyRefinementType, new: PyRefinementType) -> PyRefinementType:
        """Widening: join base types, keep only predicates stable across both."""
        joined_base = old.base.join(new.base)
        stable: List[HeapPredicate] = []
        for p in old.predicates:
            if p in new.predicates:
                stable.append(p)
            else:
                for np in new.predicates:
                    impl = np.implies(p)
                    if impl is True:
                        stable.append(p)
                        break
        return PyRefinementType(joined_base, tuple(stable))


# ═══════════════════════════════════════════════════════════════════════════
# 7.  TypeNarrower – narrow types based on guards
# ═══════════════════════════════════════════════════════════════════════════


class TypeNarrower:
    """Narrow ``PyRefinementType`` values based on runtime guard patterns."""

    @staticmethod
    def narrow_isinstance(
        typ: PyRefinementType, class_names: Tuple[str, ...]
    ) -> PyRefinementType:
        """Narrow *typ* assuming ``isinstance(x, class_names)`` is True."""
        # try to find a concrete base type from the class names
        candidate_types: List[PyType] = []
        for cn in class_names:
            pt = name_to_pytype(cn)
            if pt is not None:
                candidate_types.append(pt)
        if candidate_types:
            if len(candidate_types) == 1:
                narrowed_base = candidate_types[0]
            else:
                narrowed_base = _simplify_union(frozenset(candidate_types))
            # keep existing predicates and add isinstance predicate
            met = typ.base.meet(narrowed_base)
            if isinstance(met, NeverType):
                # if base and narrowed are incompatible via meet, still trust the isinstance
                met = narrowed_base
            pred = HeapPredicate.isinstance_pred("v", class_names)
            return PyRefinementType(met, typ.predicates + (pred,))
        # no known base types, just add the predicate
        pred = HeapPredicate.isinstance_pred("v", class_names)
        return typ.with_predicate(pred)

    @staticmethod
    def narrow_none_check(
        typ: PyRefinementType, is_none: bool
    ) -> PyRefinementType:
        """Narrow *typ* assuming ``x is None`` or ``x is not None``."""
        if is_none:
            return PyRefinementType(NoneType(), (HeapPredicate.is_none("v"),))
        # not None
        base = typ.base
        if isinstance(base, OptionalType):
            base = base.inner
        elif isinstance(base, PyUnionType):
            non_none = frozenset(m for m in base.members if not isinstance(m, NoneType))
            if non_none:
                base = _simplify_union(non_none)
            else:
                base = NeverType()
        pred = HeapPredicate.not_none("v")
        # remove any IS_NONE predicates, add NOT_NONE
        kept = tuple(p for p in typ.predicates if p.kind != HeapPredKind.IS_NONE)
        return PyRefinementType(base, kept + (pred,))

    @staticmethod
    def narrow_truthiness(
        typ: PyRefinementType, is_truthy: bool
    ) -> PyRefinementType:
        """Narrow *typ* based on truthiness."""
        pred = HeapPredicate.truthiness("v", is_truthy)
        base = typ.base
        if not is_truthy:
            # falsy values: None, 0, False, "", [], etc.
            # if it's Optional, in the falsy branch it could be None
            pass
        else:
            # truthy: remove None from unions/optionals
            if isinstance(base, OptionalType):
                base = base.inner
            elif isinstance(base, PyUnionType):
                non_none = frozenset(m for m in base.members if not isinstance(m, NoneType))
                if non_none:
                    base = _simplify_union(non_none)
        return PyRefinementType(base, typ.predicates + (pred,))

    @staticmethod
    def narrow_comparison(
        typ: PyRefinementType, op: str, value: Any
    ) -> PyRefinementType:
        """Narrow *typ* given ``x op value``."""
        pred = HeapPredicate.comparison("v", op, value)
        base = typ.base
        # if comparing with None via ==, narrow to NoneType
        if value is None and op == "==":
            return PyRefinementType(NoneType(), typ.predicates + (pred,))
        if value is None and op == "!=":
            if isinstance(base, OptionalType):
                base = base.inner
            elif isinstance(base, PyUnionType):
                non_none = frozenset(m for m in base.members if not isinstance(m, NoneType))
                if non_none:
                    base = _simplify_union(non_none)
            return PyRefinementType(base, typ.predicates + (pred,))
        # for numeric comparisons, try to narrow the type
        if isinstance(value, (int, float)):
            if isinstance(base, PyUnionType):
                numeric = frozenset(
                    m for m in base.members if isinstance(m, (IntPyType, FloatPyType, BoolPyType))
                )
                if numeric:
                    base = _simplify_union(numeric)
        return PyRefinementType(base, typ.predicates + (pred,))

    @staticmethod
    def narrow_hasattr(
        typ: PyRefinementType, attr_name: str
    ) -> PyRefinementType:
        """Narrow *typ* given ``hasattr(x, attr_name)``."""
        pred = HeapPredicate.attr_not_none("v", (attr_name,))
        base = typ.base
        # if union, keep only members that could have this attr
        if isinstance(base, PyUnionType):
            candidates: Set[PyType] = set()
            for m in base.members:
                if isinstance(m, ProtocolType):
                    if attr_name in m.required_attrs or attr_name in m.required_methods:
                        candidates.add(m)
                    else:
                        candidates.add(m)  # can't rule it out statically
                else:
                    candidates.add(m)
            if candidates:
                base = _simplify_union(frozenset(candidates))
        return PyRefinementType(base, typ.predicates + (pred,))

    @staticmethod
    def narrow_callable(typ: PyRefinementType) -> PyRefinementType:
        """Narrow *typ* given ``callable(x)``."""
        pred = HeapPredicate.callable_pred("v")
        base = typ.base
        if isinstance(base, PyUnionType):
            callable_members: Set[PyType] = set()
            for m in base.members:
                if isinstance(m, FunctionPyType):
                    callable_members.add(m)
                elif isinstance(m, (ClassType, ProtocolType)):
                    callable_members.add(m)
                else:
                    callable_members.add(m)
            if callable_members:
                base = _simplify_union(frozenset(callable_members))
        return PyRefinementType(base, typ.predicates + (pred,))

    @staticmethod
    def narrow_dict_key(
        typ: PyRefinementType, key: Any
    ) -> PyRefinementType:
        """Narrow *typ* knowing that ``key in x``."""
        pred = HeapPredicate.dict_key_exists("v", key)
        return typ.with_predicate(pred)

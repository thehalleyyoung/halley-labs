"""
Type Inference Engine with Refinement Types.

Hindley-Milner style inference extended with:
- Refinement type annotations {x:T | P(x)}
- Unification over base types
- Constraint collection from program
- Constraint solving with refinement propagation
- Gradual typing integration (partially-annotated code)
- Generic type support (List[{x:int | x > 0}], Optional, Dict)
- Compatible with mypy/pyright annotations
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict as TDict,
    FrozenSet,
    List as TList,
    Optional as TOpt,
    Set as TSet,
    Tuple as TTuple,
    Union as TUnion,
)

# ---------------------------------------------------------------------------
# Local AST stubs — no cross-module imports
# ---------------------------------------------------------------------------

class _SourceLoc:
    """Minimal source location for error reporting."""
    __slots__ = ("file", "line", "col")

    def __init__(self, file: str = "<unknown>", line: int = 0, col: int = 0):
        self.file = file
        self.line = line
        self.col = col

    def __repr__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


class _Expr:
    """Stub base for expression AST nodes."""
    loc: _SourceLoc = _SourceLoc()


class _VarExpr(_Expr):
    def __init__(self, name: str, loc: _SourceLoc = _SourceLoc()):
        self.name = name
        self.loc = loc


class _LitExpr(_Expr):
    def __init__(self, value: object, loc: _SourceLoc = _SourceLoc()):
        self.value = value
        self.loc = loc


class _AppExpr(_Expr):
    def __init__(self, func: _Expr, args: TList[_Expr], loc: _SourceLoc = _SourceLoc()):
        self.func = func
        self.args = args
        self.loc = loc


class _LamExpr(_Expr):
    def __init__(self, params: TList[str], body: _Expr,
                 annotations: TOpt[TDict[str, str]] = None,
                 loc: _SourceLoc = _SourceLoc()):
        self.params = params
        self.body = body
        self.annotations = annotations or {}
        self.loc = loc


class _LetExpr(_Expr):
    def __init__(self, name: str, rhs: _Expr, body: _Expr,
                 annotation: TOpt[str] = None,
                 loc: _SourceLoc = _SourceLoc()):
        self.name = name
        self.rhs = rhs
        self.body = body
        self.annotation = annotation
        self.loc = loc


class _IfExpr(_Expr):
    def __init__(self, cond: _Expr, then_br: _Expr, else_br: _Expr,
                 loc: _SourceLoc = _SourceLoc()):
        self.cond = cond
        self.then_br = then_br
        self.else_br = else_br
        self.loc = loc


class _BinOpExpr(_Expr):
    def __init__(self, op: str, left: _Expr, right: _Expr,
                 loc: _SourceLoc = _SourceLoc()):
        self.op = op
        self.left = left
        self.right = right
        self.loc = loc


class _TupleExpr(_Expr):
    def __init__(self, elems: TList[_Expr], loc: _SourceLoc = _SourceLoc()):
        self.elems = elems
        self.loc = loc


class _ListExpr(_Expr):
    def __init__(self, elems: TList[_Expr], loc: _SourceLoc = _SourceLoc()):
        self.elems = elems
        self.loc = loc


# ---------------------------------------------------------------------------
# 1. Type Representation  (~150 lines)
# ---------------------------------------------------------------------------

_next_var_id: int = 0


def _fresh_id() -> int:
    global _next_var_id
    _next_var_id += 1
    return _next_var_id


class Type:
    """Abstract base for all type nodes."""

    def free_vars(self) -> TSet[int]:
        return set()

    def apply_sub(self, sub: Substitution) -> Type:
        return self


class TypeVar(Type):
    """Type variable with a unique integer id."""

    def __init__(self, var_id: TOpt[int] = None, name: TOpt[str] = None):
        self.id: int = var_id if var_id is not None else _fresh_id()
        self.name: str = name or f"t{self.id}"

    def free_vars(self) -> TSet[int]:
        return {self.id}

    def apply_sub(self, sub: Substitution) -> Type:
        if self.id in sub.mapping:
            return sub.mapping[self.id].apply_sub(sub)
        return self

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TypeVar) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class BaseType(Type):
    """Primitive types: int, str, bool, float, NoneType."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BaseType) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("base", self.name))


INT = BaseType("int")
STR = BaseType("str")
BOOL = BaseType("bool")
FLOAT = BaseType("float")
NONE = BaseType("NoneType")


class FunctionType(Type):
    """Arrow type (T1, T2, …) -> Tret."""

    def __init__(self, param_types: TList[Type], return_type: Type):
        self.param_types = list(param_types)
        self.return_type = return_type

    def free_vars(self) -> TSet[int]:
        s: TSet[int] = set()
        for p in self.param_types:
            s |= p.free_vars()
        s |= self.return_type.free_vars()
        return s

    def apply_sub(self, sub: Substitution) -> Type:
        return FunctionType(
            [p.apply_sub(sub) for p in self.param_types],
            self.return_type.apply_sub(sub),
        )

    def __repr__(self) -> str:
        params = ", ".join(repr(p) for p in self.param_types)
        return f"({params}) -> {self.return_type!r}"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, FunctionType)
                and self.param_types == other.param_types
                and self.return_type == other.return_type)

    def __hash__(self) -> int:
        return hash(("fn", tuple(self.param_types), self.return_type))


class TupleType(Type):
    """Product type (T1, T2, …)."""

    def __init__(self, elem_types: TList[Type]):
        self.elem_types = list(elem_types)

    def free_vars(self) -> TSet[int]:
        s: TSet[int] = set()
        for e in self.elem_types:
            s |= e.free_vars()
        return s

    def apply_sub(self, sub: Substitution) -> Type:
        return TupleType([e.apply_sub(sub) for e in self.elem_types])

    def __repr__(self) -> str:
        return f"Tuple[{', '.join(repr(e) for e in self.elem_types)}]"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TupleType) and self.elem_types == other.elem_types

    def __hash__(self) -> int:
        return hash(("tuple", tuple(self.elem_types)))


class ListType(Type):
    """List[T]."""

    def __init__(self, elem_type: Type):
        self.elem_type = elem_type

    def free_vars(self) -> TSet[int]:
        return self.elem_type.free_vars()

    def apply_sub(self, sub: Substitution) -> Type:
        return ListType(self.elem_type.apply_sub(sub))

    def __repr__(self) -> str:
        return f"List[{self.elem_type!r}]"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ListType) and self.elem_type == other.elem_type

    def __hash__(self) -> int:
        return hash(("list", self.elem_type))


class DictType(Type):
    """Dict[K, V]."""

    def __init__(self, key_type: Type, value_type: Type):
        self.key_type = key_type
        self.value_type = value_type

    def free_vars(self) -> TSet[int]:
        return self.key_type.free_vars() | self.value_type.free_vars()

    def apply_sub(self, sub: Substitution) -> Type:
        return DictType(self.key_type.apply_sub(sub), self.value_type.apply_sub(sub))

    def __repr__(self) -> str:
        return f"Dict[{self.key_type!r}, {self.value_type!r}]"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, DictType)
                and self.key_type == other.key_type
                and self.value_type == other.value_type)

    def __hash__(self) -> int:
        return hash(("dict", self.key_type, self.value_type))


class SetType(Type):
    """Set[T]."""

    def __init__(self, elem_type: Type):
        self.elem_type = elem_type

    def free_vars(self) -> TSet[int]:
        return self.elem_type.free_vars()

    def apply_sub(self, sub: Substitution) -> Type:
        return SetType(self.elem_type.apply_sub(sub))

    def __repr__(self) -> str:
        return f"Set[{self.elem_type!r}]"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SetType) and self.elem_type == other.elem_type

    def __hash__(self) -> int:
        return hash(("set", self.elem_type))


class OptionalType(Type):
    """Optional[T] ≡ T | NoneType."""

    def __init__(self, inner: Type):
        self.inner = inner

    def free_vars(self) -> TSet[int]:
        return self.inner.free_vars()

    def apply_sub(self, sub: Substitution) -> Type:
        return OptionalType(self.inner.apply_sub(sub))

    def __repr__(self) -> str:
        return f"Optional[{self.inner!r}]"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OptionalType) and self.inner == other.inner

    def __hash__(self) -> int:
        return hash(("opt", self.inner))


class UnionType(Type):
    """T1 | T2 | …"""

    def __init__(self, alternatives: TList[Type]):
        self.alternatives = list(alternatives)

    def free_vars(self) -> TSet[int]:
        s: TSet[int] = set()
        for a in self.alternatives:
            s |= a.free_vars()
        return s

    def apply_sub(self, sub: Substitution) -> Type:
        return UnionType([a.apply_sub(sub) for a in self.alternatives])

    def __repr__(self) -> str:
        return " | ".join(repr(a) for a in self.alternatives)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnionType) and self.alternatives == other.alternatives

    def __hash__(self) -> int:
        return hash(("union", tuple(self.alternatives)))


@dataclass
class Predicate:
    """A refinement predicate.  Stored as a string expression plus a
    binder variable name, e.g.  ``Predicate("x", "x > 0")``."""
    binder: str
    expr: str

    def substitute_binder(self, new_binder: str) -> Predicate:
        new_expr = self.expr.replace(self.binder, new_binder)
        return Predicate(new_binder, new_expr)

    def __repr__(self) -> str:
        return f"{self.binder} | {self.expr}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Predicate) and self.binder == other.binder and self.expr == other.expr

    def __hash__(self) -> int:
        return hash((self.binder, self.expr))


class RefinedType(Type):
    """{x:T | P(x)}."""

    def __init__(self, base: Type, predicate: Predicate):
        self.base = base
        self.predicate = predicate

    def free_vars(self) -> TSet[int]:
        return self.base.free_vars()

    def apply_sub(self, sub: Substitution) -> Type:
        return RefinedType(self.base.apply_sub(sub), self.predicate)

    def __repr__(self) -> str:
        return f"{{{self.predicate.binder}:{self.base!r} | {self.predicate.expr}}}"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, RefinedType)
                and self.base == other.base
                and self.predicate == other.predicate)

    def __hash__(self) -> int:
        return hash(("refined", self.base, self.predicate))


class TypeScheme:
    """∀α₁…αₙ. T — a polymorphic type scheme."""

    def __init__(self, quantified: TList[int], body: Type):
        self.quantified = list(quantified)
        self.body = body

    def __repr__(self) -> str:
        qs = ", ".join(f"t{q}" for q in self.quantified)
        return f"∀{qs}. {self.body!r}"


class GradualType(Type):
    """The dynamic / unknown type ``?`` used in gradual typing."""

    _instance: TOpt[GradualType] = None

    def __new__(cls) -> GradualType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "?"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GradualType)

    def __hash__(self) -> int:
        return hash("gradual")


DYNAMIC = GradualType()


# ---------------------------------------------------------------------------
# 2. Substitution & Unification  (~150 lines)
# ---------------------------------------------------------------------------

class UnificationError(Exception):
    """Raised when two types cannot be unified."""

    def __init__(self, t1: Type, t2: Type, message: str = ""):
        self.t1 = t1
        self.t2 = t2
        super().__init__(message or f"Cannot unify {t1!r} with {t2!r}")


class Substitution:
    """Mapping from type-variable ids to types with composition."""

    def __init__(self, mapping: TOpt[TDict[int, Type]] = None):
        self.mapping: TDict[int, Type] = dict(mapping) if mapping else {}

    # Apply this substitution to a type.
    def apply(self, t: Type) -> Type:
        return t.apply_sub(self)

    def apply_to_scheme(self, scheme: TypeScheme) -> TypeScheme:
        restricted = Substitution({
            k: v for k, v in self.mapping.items()
            if k not in scheme.quantified
        })
        return TypeScheme(scheme.quantified, restricted.apply(scheme.body))

    def compose(self, other: Substitution) -> Substitution:
        """Return ``self ∘ other`` so that ``(self ∘ other)(t) == self(other(t))``."""
        new_mapping: TDict[int, Type] = {}
        for var_id, ty in other.mapping.items():
            new_mapping[var_id] = self.apply(ty)
        for var_id, ty in self.mapping.items():
            if var_id not in new_mapping:
                new_mapping[var_id] = ty
        return Substitution(new_mapping)

    def __repr__(self) -> str:
        pairs = ", ".join(f"t{k} ↦ {v!r}" for k, v in self.mapping.items())
        return f"{{{pairs}}}"

    def __bool__(self) -> bool:
        return bool(self.mapping)


_EMPTY_SUB = Substitution()


def occurs_check(var: TypeVar, t: Type) -> bool:
    """Return *True* if ``var`` appears anywhere inside ``t``."""
    if isinstance(t, TypeVar):
        return t.id == var.id
    if isinstance(t, FunctionType):
        return any(occurs_check(var, p) for p in t.param_types) or occurs_check(var, t.return_type)
    if isinstance(t, TupleType):
        return any(occurs_check(var, e) for e in t.elem_types)
    if isinstance(t, ListType):
        return occurs_check(var, t.elem_type)
    if isinstance(t, SetType):
        return occurs_check(var, t.elem_type)
    if isinstance(t, DictType):
        return occurs_check(var, t.key_type) or occurs_check(var, t.value_type)
    if isinstance(t, OptionalType):
        return occurs_check(var, t.inner)
    if isinstance(t, UnionType):
        return any(occurs_check(var, a) for a in t.alternatives)
    if isinstance(t, RefinedType):
        return occurs_check(var, t.base)
    return False


# Collected refinement constraints that arise during unification of refined
# types.  Each entry is a pair of predicates that must be logically equivalent.
_refinement_equalities: TList[TTuple[Predicate, Predicate]] = []


def unify(t1: Type, t2: Type) -> Substitution:
    """Robinson's unification extended for refinement and gradual types."""

    # Gradual type is consistent with everything.
    if isinstance(t1, GradualType) or isinstance(t2, GradualType):
        return _EMPTY_SUB

    # Type variables.
    if isinstance(t1, TypeVar):
        if t1 == t2:
            return _EMPTY_SUB
        if occurs_check(t1, t2):
            raise UnificationError(t1, t2, f"Occurs check: {t1!r} in {t2!r}")
        return Substitution({t1.id: t2})

    if isinstance(t2, TypeVar):
        return unify(t2, t1)

    # Refined types — strip refinements, unify bases, record predicate constraint.
    if isinstance(t1, RefinedType) and isinstance(t2, RefinedType):
        sub = unify(t1.base, t2.base)
        _refinement_equalities.append((t1.predicate, t2.predicate))
        return sub
    if isinstance(t1, RefinedType):
        return unify(t1.base, t2)
    if isinstance(t2, RefinedType):
        return unify(t1, t2.base)

    # Base types.
    if isinstance(t1, BaseType) and isinstance(t2, BaseType):
        if t1.name == t2.name:
            return _EMPTY_SUB
        # int <-> float widening.
        if {t1.name, t2.name} == {"int", "float"}:
            return _EMPTY_SUB
        raise UnificationError(t1, t2)

    # Function types.
    if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
        if len(t1.param_types) != len(t2.param_types):
            raise UnificationError(t1, t2, "Arity mismatch")
        sub = _EMPTY_SUB
        for p1, p2 in zip(t1.param_types, t2.param_types):
            s = unify(sub.apply(p1), sub.apply(p2))
            sub = s.compose(sub)
        s = unify(sub.apply(t1.return_type), sub.apply(t2.return_type))
        return s.compose(sub)

    # Tuple types.
    if isinstance(t1, TupleType) and isinstance(t2, TupleType):
        if len(t1.elem_types) != len(t2.elem_types):
            raise UnificationError(t1, t2, "Tuple length mismatch")
        sub = _EMPTY_SUB
        for e1, e2 in zip(t1.elem_types, t2.elem_types):
            s = unify(sub.apply(e1), sub.apply(e2))
            sub = s.compose(sub)
        return sub

    # Container types.
    if isinstance(t1, ListType) and isinstance(t2, ListType):
        return unify(t1.elem_type, t2.elem_type)
    if isinstance(t1, SetType) and isinstance(t2, SetType):
        return unify(t1.elem_type, t2.elem_type)
    if isinstance(t1, DictType) and isinstance(t2, DictType):
        sub = unify(t1.key_type, t2.key_type)
        s2 = unify(sub.apply(t1.value_type), sub.apply(t2.value_type))
        return s2.compose(sub)

    # Optional.
    if isinstance(t1, OptionalType) and isinstance(t2, OptionalType):
        return unify(t1.inner, t2.inner)

    # Union — try to unify element-wise (same length).
    if isinstance(t1, UnionType) and isinstance(t2, UnionType):
        if len(t1.alternatives) != len(t2.alternatives):
            raise UnificationError(t1, t2, "Union arity mismatch")
        sub = _EMPTY_SUB
        for a1, a2 in zip(t1.alternatives, t2.alternatives):
            s = unify(sub.apply(a1), sub.apply(a2))
            sub = s.compose(sub)
        return sub

    raise UnificationError(t1, t2)


# ---------------------------------------------------------------------------
# 3. Type Environment  (~100 lines)
# ---------------------------------------------------------------------------

class TypeEnv:
    """Scoped mapping from variable names to type schemes."""

    def __init__(self, parent: TOpt[TypeEnv] = None):
        self._bindings: TDict[str, TypeScheme] = {}
        self._parent = parent
        # Type narrowing: maps variable name -> narrowed type in current scope.
        self._narrowed: TDict[str, Type] = {}

    def bind(self, name: str, scheme: TypeScheme) -> None:
        self._bindings[name] = scheme

    def bind_type(self, name: str, ty: Type) -> None:
        self._bindings[name] = TypeScheme([], ty)

    def lookup(self, name: str) -> TOpt[TypeScheme]:
        if name in self._narrowed:
            return TypeScheme([], self._narrowed[name])
        if name in self._bindings:
            return self._bindings[name]
        if self._parent is not None:
            return self._parent.lookup(name)
        return None

    def narrow(self, name: str, ty: Type) -> None:
        self._narrowed[name] = ty

    def child_scope(self) -> TypeEnv:
        return TypeEnv(parent=self)

    def free_vars(self) -> TSet[int]:
        fvs: TSet[int] = set()
        for scheme in self._bindings.values():
            body_fv = scheme.body.free_vars()
            fvs |= body_fv - set(scheme.quantified)
        if self._parent:
            fvs |= self._parent.free_vars()
        return fvs

    def all_bindings(self) -> TDict[str, TypeScheme]:
        result: TDict[str, TypeScheme] = {}
        if self._parent:
            result.update(self._parent.all_bindings())
        result.update(self._bindings)
        return result

    def apply_sub(self, sub: Substitution) -> TypeEnv:
        new_env = TypeEnv(parent=self._parent.apply_sub(sub) if self._parent else None)
        for name, scheme in self._bindings.items():
            new_env._bindings[name] = sub.apply_to_scheme(scheme)
        return new_env

    def __repr__(self) -> str:
        items = ", ".join(f"{k}: {v!r}" for k, v in self._bindings.items())
        return f"Env({items})"


def generalize(env: TypeEnv, ty: Type) -> TypeScheme:
    """Close over free variables not free in the environment."""
    env_fv = env.free_vars()
    ty_fv = ty.free_vars()
    quantified = sorted(ty_fv - env_fv)
    return TypeScheme(quantified, ty)


def instantiate(scheme: TypeScheme) -> Type:
    """Replace quantified variables with fresh type variables."""
    if not scheme.quantified:
        return scheme.body
    mapping: TDict[int, Type] = {}
    for old_id in scheme.quantified:
        mapping[old_id] = TypeVar()
    return Substitution(mapping).apply(scheme.body)


def _make_builtin_env() -> TypeEnv:
    """Construct a top-level environment with common built-in types."""
    env = TypeEnv()
    # Arithmetic operators
    for op in ("+", "-", "*", "//", "%"):
        a, b, r = TypeVar(), TypeVar(), TypeVar()
        env.bind(op, TypeScheme([a.id, b.id, r.id], FunctionType([a, b], r)))
    env.bind("/", TypeScheme([], FunctionType([FLOAT, FLOAT], FLOAT)))
    env.bind("**", TypeScheme([], FunctionType([INT, INT], INT)))
    # Comparison operators → bool
    for op in ("<", ">", "<=", ">=", "==", "!="):
        a = TypeVar()
        env.bind(op, TypeScheme([a.id], FunctionType([a, a], BOOL)))
    # Boolean operators
    for op in ("and", "or"):
        env.bind(op, TypeScheme([], FunctionType([BOOL, BOOL], BOOL)))
    env.bind("not", TypeScheme([], FunctionType([BOOL], BOOL)))
    # Common built-in functions
    a = TypeVar()
    env.bind("len", TypeScheme([a.id], FunctionType([ListType(a)], INT)))
    a = TypeVar()
    env.bind("print", TypeScheme([a.id], FunctionType([a], NONE)))
    a = TypeVar()
    env.bind("abs", TypeScheme([a.id], FunctionType([a], a)))
    env.bind("int", TypeScheme([], FunctionType([FLOAT], INT)))
    env.bind("float", TypeScheme([], FunctionType([INT], FLOAT)))
    env.bind("str", TypeScheme([], FunctionType([INT], STR)))
    return env


# ---------------------------------------------------------------------------
# 4. Constraint Generation  (~200 lines)
# ---------------------------------------------------------------------------

class Constraint:
    """Base class for type constraints."""
    loc: _SourceLoc = _SourceLoc()


class EqualityConstraint(Constraint):
    """T1 = T2."""

    def __init__(self, left: Type, right: Type, loc: _SourceLoc = _SourceLoc()):
        self.left = left
        self.right = right
        self.loc = loc

    def apply_sub(self, sub: Substitution) -> EqualityConstraint:
        return EqualityConstraint(sub.apply(self.left), sub.apply(self.right), self.loc)

    def __repr__(self) -> str:
        return f"{self.left!r} = {self.right!r}"


class SubtypeConstraint(Constraint):
    """T1 <: T2."""

    def __init__(self, sub: Type, sup: Type, loc: _SourceLoc = _SourceLoc()):
        self.sub_type = sub
        self.sup_type = sup
        self.loc = loc

    def apply_sub(self, sub: Substitution) -> SubtypeConstraint:
        return SubtypeConstraint(sub.apply(self.sub_type), sub.apply(self.sup_type), self.loc)

    def __repr__(self) -> str:
        return f"{self.sub_type!r} <: {self.sup_type!r}"


class RefinementConstraint(Constraint):
    """Γ ⊢ P(x) where x : T."""

    def __init__(self, predicate: Predicate, var_type: Type,
                 env: TOpt[TypeEnv] = None,
                 loc: _SourceLoc = _SourceLoc()):
        self.predicate = predicate
        self.var_type = var_type
        self.env = env
        self.loc = loc

    def apply_sub(self, sub: Substitution) -> RefinementConstraint:
        return RefinementConstraint(
            self.predicate, sub.apply(self.var_type), self.env, self.loc
        )

    def __repr__(self) -> str:
        return f"⊢ {self.predicate!r} where {self.predicate.binder}:{self.var_type!r}"


class ConstraintCollector:
    """Walk an expression tree and emit constraints."""

    def __init__(self, env: TOpt[TypeEnv] = None):
        self.env = env or _make_builtin_env()
        self.constraints: TList[Constraint] = []
        self._expr_types: TDict[int, Type] = {}

    def _record(self, expr: _Expr, ty: Type) -> None:
        self._expr_types[id(expr)] = ty

    def _emit(self, c: Constraint) -> None:
        self.constraints.append(c)

    def collect(self, expr: _Expr, env: TOpt[TypeEnv] = None) -> Type:
        """Return the inferred type of *expr*, recording constraints."""
        if env is None:
            env = self.env
        ty = self._visit(expr, env)
        self._record(expr, ty)
        return ty

    def _visit(self, expr: _Expr, env: TypeEnv) -> Type:
        if isinstance(expr, _LitExpr):
            return self._visit_lit(expr)
        if isinstance(expr, _VarExpr):
            return self._visit_var(expr, env)
        if isinstance(expr, _AppExpr):
            return self._visit_app(expr, env)
        if isinstance(expr, _LamExpr):
            return self._visit_lam(expr, env)
        if isinstance(expr, _LetExpr):
            return self._visit_let(expr, env)
        if isinstance(expr, _IfExpr):
            return self._visit_if(expr, env)
        if isinstance(expr, _BinOpExpr):
            return self._visit_binop(expr, env)
        if isinstance(expr, _TupleExpr):
            return self._visit_tuple(expr, env)
        if isinstance(expr, _ListExpr):
            return self._visit_list(expr, env)
        # Fallback: assign fresh variable.
        return TypeVar()

    # -- Literals ----------------------------------------------------------

    def _visit_lit(self, expr: _LitExpr) -> Type:
        v = expr.value
        if isinstance(v, bool):
            return BOOL
        if isinstance(v, int):
            return INT
        if isinstance(v, float):
            return FLOAT
        if isinstance(v, str):
            return STR
        if v is None:
            return NONE
        return TypeVar()

    # -- Variables ---------------------------------------------------------

    def _visit_var(self, expr: _VarExpr, env: TypeEnv) -> Type:
        scheme = env.lookup(expr.name)
        if scheme is None:
            tv = TypeVar(name=expr.name)
            env.bind_type(expr.name, tv)
            return tv
        return instantiate(scheme)

    # -- Application -------------------------------------------------------

    def _visit_app(self, expr: _AppExpr, env: TypeEnv) -> Type:
        func_ty = self.collect(expr.func, env)
        arg_types = [self.collect(a, env) for a in expr.args]
        ret = TypeVar()
        expected_fn = FunctionType(arg_types, ret)
        self._emit(EqualityConstraint(func_ty, expected_fn, expr.loc))
        return ret

    # -- Lambda ------------------------------------------------------------

    def _visit_lam(self, expr: _LamExpr, env: TypeEnv) -> Type:
        child = env.child_scope()
        param_types: TList[Type] = []
        for p in expr.params:
            ann = expr.annotations.get(p)
            if ann:
                pty = parse_annotation(ann)
            else:
                pty = TypeVar(name=p)
            child.bind_type(p, pty)
            param_types.append(pty)
        body_ty = self.collect(expr.body, child)
        return FunctionType(param_types, body_ty)

    # -- Let ---------------------------------------------------------------

    def _visit_let(self, expr: _LetExpr, env: TypeEnv) -> Type:
        rhs_ty = self.collect(expr.rhs, env)
        if expr.annotation:
            ann_ty = parse_annotation(expr.annotation)
            self._emit(EqualityConstraint(rhs_ty, ann_ty, expr.loc))
            rhs_ty = ann_ty
        scheme = generalize(env, rhs_ty)
        child = env.child_scope()
        child.bind(expr.name, scheme)
        return self.collect(expr.body, child)

    # -- If ----------------------------------------------------------------

    def _visit_if(self, expr: _IfExpr, env: TypeEnv) -> Type:
        cond_ty = self.collect(expr.cond, env)
        self._emit(EqualityConstraint(cond_ty, BOOL, expr.loc))

        then_env = env.child_scope()
        else_env = env.child_scope()
        # Narrowing: if the condition is ``x is not None`` we could narrow,
        # but without full pattern matching we just fork scopes.
        self._apply_narrowing(expr.cond, then_env, else_env)

        then_ty = self.collect(expr.then_br, then_env)
        else_ty = self.collect(expr.else_br, else_env)
        result = TypeVar()
        self._emit(EqualityConstraint(then_ty, result, expr.loc))
        self._emit(EqualityConstraint(else_ty, result, expr.loc))
        return result

    def _apply_narrowing(self, cond: _Expr, then_env: TypeEnv, else_env: TypeEnv) -> None:
        """Heuristic narrowing from condition expressions."""
        if isinstance(cond, _BinOpExpr):
            if cond.op in ("<", ">", "<=", ">=") and isinstance(cond.left, _VarExpr):
                name = cond.left.name
                scheme = then_env.lookup(name)
                if scheme:
                    base = instantiate(scheme)
                    pred = Predicate(name, f"{name} {cond.op} {_expr_repr(cond.right)}")
                    refined = RefinedType(base, pred)
                    then_env.narrow(name, refined)
                    neg_op = {"<": ">=", ">": "<=", "<=": ">", ">=": "<"}.get(cond.op, "!=")
                    neg_pred = Predicate(name, f"{name} {neg_op} {_expr_repr(cond.right)}")
                    else_env.narrow(name, RefinedType(base, neg_pred))

    # -- Binary ops --------------------------------------------------------

    def _visit_binop(self, expr: _BinOpExpr, env: TypeEnv) -> Type:
        left_ty = self.collect(expr.left, env)
        right_ty = self.collect(expr.right, env)
        op_scheme = env.lookup(expr.op)
        if op_scheme is None:
            ret = TypeVar()
            return ret
        op_ty = instantiate(op_scheme)
        ret = TypeVar()
        expected = FunctionType([left_ty, right_ty], ret)
        self._emit(EqualityConstraint(op_ty, expected, expr.loc))
        return ret

    # -- Tuples / Lists ----------------------------------------------------

    def _visit_tuple(self, expr: _TupleExpr, env: TypeEnv) -> Type:
        elem_types = [self.collect(e, env) for e in expr.elems]
        return TupleType(elem_types)

    def _visit_list(self, expr: _ListExpr, env: TypeEnv) -> Type:
        elem_var = TypeVar()
        for e in expr.elems:
            ety = self.collect(e, env)
            self._emit(EqualityConstraint(ety, elem_var, expr.loc))
        return ListType(elem_var)


def _expr_repr(expr: _Expr) -> str:
    if isinstance(expr, _LitExpr):
        return repr(expr.value)
    if isinstance(expr, _VarExpr):
        return expr.name
    return "…"


# ---------------------------------------------------------------------------
# 5. Constraint Solver  (~150 lines)
# ---------------------------------------------------------------------------

@dataclass
class TypeError_:
    """A type error with location information."""
    message: str
    loc: _SourceLoc = field(default_factory=_SourceLoc)


class ConstraintSolver:
    """Three-phase solver: equality → subtyping → refinement."""

    def __init__(self) -> None:
        self.substitution = Substitution()
        self.errors: TList[TypeError_] = []
        self._refinement_obligations: TList[RefinementConstraint] = []
        self._mu_bindings: TDict[int, Type] = {}

    def solve(self, constraints: TList[Constraint]) -> Substitution:
        """Solve all constraints, returning the composed substitution."""
        eq: TList[EqualityConstraint] = []
        sub: TList[SubtypeConstraint] = []
        ref: TList[RefinementConstraint] = []
        for c in constraints:
            if isinstance(c, EqualityConstraint):
                eq.append(c)
            elif isinstance(c, SubtypeConstraint):
                sub.append(c)
            elif isinstance(c, RefinementConstraint):
                ref.append(c)

        self._solve_equalities(eq)
        self._solve_subtypes(sub)
        self._solve_refinements(ref)
        return self.substitution

    # Phase 1 — equality via unification.
    def _solve_equalities(self, constraints: TList[EqualityConstraint]) -> None:
        for c in constraints:
            c = c.apply_sub(self.substitution)
            try:
                s = unify(c.left, c.right)
                self.substitution = s.compose(self.substitution)
            except UnificationError as e:
                self.errors.append(TypeError_(str(e), c.loc))

    # Phase 2 — subtyping via subsumption.
    def _solve_subtypes(self, constraints: TList[SubtypeConstraint]) -> None:
        for c in constraints:
            c = c.apply_sub(self.substitution)
            if not self._is_subtype(c.sub_type, c.sup_type):
                self.errors.append(
                    TypeError_(f"{c.sub_type!r} is not a subtype of {c.sup_type!r}", c.loc)
                )

    def _is_subtype(self, t1: Type, t2: Type) -> bool:
        """Conservative subtype check.  Returns True when we can confirm."""
        if isinstance(t2, GradualType) or isinstance(t1, GradualType):
            return True
        if t1 == t2:
            return True
        if isinstance(t1, BaseType) and isinstance(t2, BaseType):
            if t1.name == "int" and t2.name == "float":
                return True
            if t1.name == "bool" and t2.name == "int":
                return True
        if isinstance(t2, OptionalType):
            if isinstance(t1, BaseType) and t1.name == "NoneType":
                return True
            return self._is_subtype(t1, t2.inner)
        if isinstance(t2, UnionType):
            return any(self._is_subtype(t1, a) for a in t2.alternatives)
        if isinstance(t1, RefinedType):
            return self._is_subtype(t1.base, t2)
        if isinstance(t2, RefinedType):
            if not self._is_subtype(t1, t2.base):
                return False
            self._refinement_obligations.append(
                RefinementConstraint(t2.predicate, t1)
            )
            return True
        if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
            if len(t1.param_types) != len(t2.param_types):
                return False
            # Contravariant in params, covariant in return.
            for p1, p2 in zip(t1.param_types, t2.param_types):
                if not self._is_subtype(p2, p1):
                    return False
            return self._is_subtype(t1.return_type, t2.return_type)
        if isinstance(t1, ListType) and isinstance(t2, ListType):
            # Lists are invariant in Python, but we allow covariance for read-only usage.
            try:
                s = unify(t1.elem_type, t2.elem_type)
                self.substitution = s.compose(self.substitution)
                return True
            except UnificationError:
                return False
        if isinstance(t1, TypeVar) or isinstance(t2, TypeVar):
            try:
                s = unify(t1, t2)
                self.substitution = s.compose(self.substitution)
                return True
            except UnificationError:
                return False
        return False

    # Phase 3 — refinement checking (delegated / best-effort).
    def _solve_refinements(self, constraints: TList[RefinementConstraint]) -> None:
        all_ref = list(constraints) + self._refinement_obligations
        for rc in all_ref:
            rc = rc.apply_sub(self.substitution)
            if not self._check_refinement(rc):
                self.errors.append(
                    TypeError_(
                        f"Cannot verify refinement {rc.predicate!r} "
                        f"for {rc.var_type!r}",
                        rc.loc,
                    )
                )
        # Also handle equalities collected during unification.
        global _refinement_equalities
        for p1, p2 in _refinement_equalities:
            if p1.expr != p2.expr:
                self.errors.append(
                    TypeError_(f"Refinement mismatch: {p1.expr} vs {p2.expr}")
                )
        _refinement_equalities.clear()

    def _check_refinement(self, rc: RefinementConstraint) -> bool:
        """Best-effort refinement verification.

        In a full implementation this would delegate to an SMT solver.
        Here we do simple syntactic checks and constant folding.
        """
        expr = rc.predicate.expr.strip()
        # Tautologies.
        if expr in ("True", "true", "1"):
            return True
        # Attempt trivial evaluation for constant predicates.
        try:
            result = eval(expr, {"__builtins__": {}}, {rc.predicate.binder: 0})
            if isinstance(result, bool):
                return result
        except Exception:
            pass
        # Assume valid when we cannot decide.
        return True

    # -- Recursive / μ-types -----------------------------------------------

    def register_mu(self, var_id: int, body: Type) -> None:
        """Register a recursive type binding ``μα.T``."""
        self._mu_bindings[var_id] = body

    def expand_mu(self, var_id: int) -> TOpt[Type]:
        body = self._mu_bindings.get(var_id)
        if body is None:
            return None
        return Substitution({var_id: body}).apply(body)


# ---------------------------------------------------------------------------
# 6. Gradual Typing Integration  (~100 lines)
# ---------------------------------------------------------------------------

class GradualTyping:
    """Handle partially-annotated Python code."""

    def __init__(self) -> None:
        self._casts: TList[TTuple[_Expr, Type, Type]] = []

    def is_consistent(self, t1: Type, t2: Type) -> bool:
        """Consistent subtyping: ``?`` is consistent with everything."""
        if isinstance(t1, GradualType) or isinstance(t2, GradualType):
            return True
        if isinstance(t1, BaseType) and isinstance(t2, BaseType):
            return t1.name == t2.name
        if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
            if len(t1.param_types) != len(t2.param_types):
                return False
            return all(
                self.is_consistent(p1, p2)
                for p1, p2 in zip(t1.param_types, t2.param_types)
            ) and self.is_consistent(t1.return_type, t2.return_type)
        if isinstance(t1, ListType) and isinstance(t2, ListType):
            return self.is_consistent(t1.elem_type, t2.elem_type)
        if isinstance(t1, DictType) and isinstance(t2, DictType):
            return (self.is_consistent(t1.key_type, t2.key_type)
                    and self.is_consistent(t1.value_type, t2.value_type))
        if isinstance(t1, TupleType) and isinstance(t2, TupleType):
            if len(t1.elem_types) != len(t2.elem_types):
                return False
            return all(
                self.is_consistent(e1, e2)
                for e1, e2 in zip(t1.elem_types, t2.elem_types)
            )
        if isinstance(t1, OptionalType) and isinstance(t2, OptionalType):
            return self.is_consistent(t1.inner, t2.inner)
        if isinstance(t1, RefinedType):
            return self.is_consistent(t1.base, t2)
        if isinstance(t2, RefinedType):
            return self.is_consistent(t1, t2.base)
        if isinstance(t1, UnionType):
            return any(self.is_consistent(a, t2) for a in t1.alternatives)
        if isinstance(t2, UnionType):
            return any(self.is_consistent(t1, a) for a in t2.alternatives)
        if type(t1) == type(t2):
            return t1 == t2
        return False

    def insert_cast(self, expr: _Expr, from_ty: Type, to_ty: Type) -> None:
        """Record a cast at a gradual/static boundary."""
        self._casts.append((expr, from_ty, to_ty))

    def casts(self) -> TList[TTuple[_Expr, Type, Type]]:
        return list(self._casts)

    def materialize(self, ty: Type) -> Type:
        """Replace ``?`` with fresh type variables for downstream unification."""
        if isinstance(ty, GradualType):
            return TypeVar()
        if isinstance(ty, FunctionType):
            return FunctionType(
                [self.materialize(p) for p in ty.param_types],
                self.materialize(ty.return_type),
            )
        if isinstance(ty, ListType):
            return ListType(self.materialize(ty.elem_type))
        if isinstance(ty, DictType):
            return DictType(self.materialize(ty.key_type), self.materialize(ty.value_type))
        if isinstance(ty, SetType):
            return SetType(self.materialize(ty.elem_type))
        if isinstance(ty, TupleType):
            return TupleType([self.materialize(e) for e in ty.elem_types])
        if isinstance(ty, OptionalType):
            return OptionalType(self.materialize(ty.inner))
        if isinstance(ty, UnionType):
            return UnionType([self.materialize(a) for a in ty.alternatives])
        if isinstance(ty, RefinedType):
            return RefinedType(self.materialize(ty.base), ty.predicate)
        return ty


_ANNOTATION_RE = re.compile(
    r"^\{(\w+)\s*:\s*(\w+)\s*\|\s*(.+)\}$"
)
_OPTIONAL_RE = re.compile(r"^Optional\[(.+)\]$")
_LIST_RE = re.compile(r"^List\[(.+)\]$")
_SET_RE = re.compile(r"^Set\[(.+)\]$")
_DICT_RE = re.compile(r"^Dict\[(.+),\s*(.+)\]$")
_TUPLE_RE = re.compile(r"^Tuple\[(.+)\]$")
_UNION_RE = re.compile(r"^Union\[(.+)\]$")


def parse_annotation(ann: str) -> Type:
    """Parse a mypy/pyright-compatible annotation string into our Type AST."""
    ann = ann.strip()

    if ann == "Any" or ann == "?":
        return DYNAMIC

    # Refinement: {x:int | x > 0}
    m = _ANNOTATION_RE.match(ann)
    if m:
        binder, base_name, pred_expr = m.group(1), m.group(2), m.group(3)
        return RefinedType(_base_from_name(base_name), Predicate(binder, pred_expr.strip()))

    # Optional[T]
    m = _OPTIONAL_RE.match(ann)
    if m:
        return OptionalType(parse_annotation(m.group(1)))

    # List[T]
    m = _LIST_RE.match(ann)
    if m:
        return ListType(parse_annotation(m.group(1)))

    # Set[T]
    m = _SET_RE.match(ann)
    if m:
        return SetType(parse_annotation(m.group(1)))

    # Dict[K, V]
    m = _DICT_RE.match(ann)
    if m:
        return DictType(parse_annotation(m.group(1)), parse_annotation(m.group(2)))

    # Tuple[T, ...]
    m = _TUPLE_RE.match(ann)
    if m:
        parts = _split_top_level(m.group(1))
        return TupleType([parse_annotation(p) for p in parts])

    # Union[T1, T2]
    m = _UNION_RE.match(ann)
    if m:
        parts = _split_top_level(m.group(1))
        return UnionType([parse_annotation(p) for p in parts])

    # T1 | T2 (PEP 604 syntax)
    if "|" in ann and not ann.startswith("{"):
        parts = [s.strip() for s in ann.split("|")]
        if len(parts) >= 2:
            return UnionType([parse_annotation(p) for p in parts])

    # None
    if ann in ("None", "NoneType"):
        return NONE

    return _base_from_name(ann)


def _base_from_name(name: str) -> Type:
    """Map a simple name to a BaseType or fresh TypeVar."""
    mapping = {"int": INT, "str": STR, "bool": BOOL, "float": FLOAT, "None": NONE, "NoneType": NONE}
    return mapping.get(name, BaseType(name))


def _split_top_level(s: str) -> TList[str]:
    """Split a string by commas respecting bracket nesting."""
    parts: TList[str] = []
    depth = 0
    current: TList[str] = []
    for ch in s:
        if ch in ("(", "[", "{"):
            depth += 1
            current.append(ch)
        elif ch in (")", "]", "}"):
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


# ---------------------------------------------------------------------------
# 7. Generic Type Support  (~100 lines)
# ---------------------------------------------------------------------------

class Variance(Enum):
    COVARIANT = auto()
    CONTRAVARIANT = auto()
    INVARIANT = auto()


@dataclass
class TypeParam:
    """A type parameter with variance annotation."""
    var: TypeVar
    variance: Variance = Variance.INVARIANT
    bound: TOpt[Type] = None


class GenericInference:
    """Handle parameterized / generic types."""

    def __init__(self) -> None:
        self._registered: TDict[str, TList[TypeParam]] = {}

    def register_generic(self, name: str, params: TList[TypeParam]) -> None:
        self._registered[name] = params

    def infer_type_args(self, generic_name: str, usages: TList[Type]) -> TList[Type]:
        """Infer type arguments from a collection of element usages.

        For example, given ``List`` with usages ``[int, int, int]``, infer
        that the type argument is ``int``.
        """
        params = self._registered.get(generic_name)
        if params is None or not params:
            return list(usages[:1]) if usages else [TypeVar()]

        sub = Substitution()
        representative = usages[0] if usages else TypeVar()
        for u in usages[1:]:
            try:
                s = unify(sub.apply(representative), sub.apply(u))
                sub = s.compose(sub)
            except UnificationError:
                # Widen to a union when unification fails.
                representative = UnionType([representative, u])

        inferred: TList[Type] = []
        for param in params:
            ty = sub.apply(representative)
            if param.bound is not None:
                try:
                    s = unify(ty, param.bound)
                    sub = s.compose(sub)
                except UnificationError:
                    pass
            inferred.append(sub.apply(ty))
        return inferred

    def check_variance(self, param: TypeParam, actual_sub: Type, actual_sup: Type,
                       solver: ConstraintSolver) -> bool:
        """Check that a type argument respects the declared variance."""
        if param.variance == Variance.COVARIANT:
            return solver._is_subtype(actual_sub, actual_sup)
        if param.variance == Variance.CONTRAVARIANT:
            return solver._is_subtype(actual_sup, actual_sub)
        # Invariant — must be equal.
        try:
            unify(actual_sub, actual_sup)
            return True
        except UnificationError:
            return False

    def instantiate_generic(self, generic_name: str, args: TList[Type]) -> Type:
        """Return the instantiated container type."""
        if generic_name == "List":
            return ListType(args[0] if args else TypeVar())
        if generic_name == "Set":
            return SetType(args[0] if args else TypeVar())
        if generic_name == "Dict":
            k = args[0] if len(args) > 0 else TypeVar()
            v = args[1] if len(args) > 1 else TypeVar()
            return DictType(k, v)
        if generic_name == "Optional":
            return OptionalType(args[0] if args else TypeVar())
        if generic_name == "Tuple":
            return TupleType(args)
        # Unknown generic — return a named base with type args encoded.
        return BaseType(f"{generic_name}[{', '.join(repr(a) for a in args)}]")

    def propagate_refinements(self, container_ty: Type, elem_pred: Predicate) -> Type:
        """Lift an element-level refinement onto a container type.

        ``List[{x:int | x > 0}]`` → ``ListType(RefinedType(INT, x > 0))``
        """
        if isinstance(container_ty, ListType):
            return ListType(RefinedType(container_ty.elem_type, elem_pred))
        if isinstance(container_ty, SetType):
            return SetType(RefinedType(container_ty.elem_type, elem_pred))
        if isinstance(container_ty, DictType):
            return DictType(container_ty.key_type, RefinedType(container_ty.value_type, elem_pred))
        if isinstance(container_ty, OptionalType):
            return OptionalType(RefinedType(container_ty.inner, elem_pred))
        return container_ty


# ---------------------------------------------------------------------------
# 8. Main Inference Engine  (~100 lines)
# ---------------------------------------------------------------------------

@dataclass
class TypedExpr:
    """An expression annotated with its inferred type."""
    expr: _Expr
    ty: Type


@dataclass
class TypedProgram:
    """Result of type inference over a whole program."""
    bindings: TDict[str, Type]
    typed_exprs: TList[TypedExpr]
    errors: TList[TypeError_]
    substitution: Substitution


class TypeInferenceEngine:
    """Compose constraint collection, solving, and gradual typing."""

    def __init__(self, env: TOpt[TypeEnv] = None):
        self._base_env = env or _make_builtin_env()
        self._gradual = GradualTyping()
        self._generics = GenericInference()
        self._setup_generics()

    def _setup_generics(self) -> None:
        a = TypeVar()
        self._generics.register_generic("List", [TypeParam(a, Variance.COVARIANT)])
        b = TypeVar()
        self._generics.register_generic("Set", [TypeParam(b, Variance.INVARIANT)])
        k, v = TypeVar(), TypeVar()
        self._generics.register_generic("Dict", [
            TypeParam(k, Variance.INVARIANT),
            TypeParam(v, Variance.COVARIANT),
        ])
        t = TypeVar()
        self._generics.register_generic("Optional", [TypeParam(t, Variance.COVARIANT)])

    # -- Public API --------------------------------------------------------

    def infer(self, program: TList[_Expr]) -> TypedProgram:
        """Infer types for a list of top-level expressions."""
        env = self._base_env.child_scope()
        collector = ConstraintCollector(env)
        expr_types: TList[TTuple[_Expr, Type]] = []

        for expr in program:
            ty = collector.collect(expr, env)
            expr_types.append((expr, ty))

        solver = ConstraintSolver()
        sub = solver.solve(collector.constraints)

        typed: TList[TypedExpr] = []
        bindings: TDict[str, Type] = {}
        for expr, raw_ty in expr_types:
            resolved = sub.apply(raw_ty)
            typed.append(TypedExpr(expr, resolved))
            if isinstance(expr, _LetExpr):
                bindings[expr.name] = resolved

        return TypedProgram(
            bindings=bindings,
            typed_exprs=typed,
            errors=solver.errors,
            substitution=sub,
        )

    def check(self, program: TList[_Expr], expected: TDict[str, Type]) -> TList[TypeError_]:
        """Check program against expected types; return errors."""
        result = self.infer(program)
        errors = list(result.errors)
        for name, exp_ty in expected.items():
            actual = result.bindings.get(name)
            if actual is None:
                errors.append(TypeError_(f"Name '{name}' not found in program"))
                continue
            try:
                unify(actual, exp_ty)
            except UnificationError as e:
                errors.append(TypeError_(str(e)))
        return errors

    def infer_with_gradual(self, program: TList[_Expr]) -> TypedProgram:
        """Infer types, materializing gradual types into variables first."""
        env = self._base_env.child_scope()
        collector = ConstraintCollector(env)
        expr_types: TList[TTuple[_Expr, Type]] = []

        for expr in program:
            ty = collector.collect(expr, env)
            mat = self._gradual.materialize(ty)
            if mat is not ty:
                collector._emit(EqualityConstraint(ty, mat, expr.loc))
            expr_types.append((expr, ty))

        solver = ConstraintSolver()
        sub = solver.solve(collector.constraints)

        typed: TList[TypedExpr] = []
        bindings: TDict[str, Type] = {}
        for expr, raw_ty in expr_types:
            resolved = sub.apply(raw_ty)
            typed.append(TypedExpr(expr, resolved))
            if isinstance(expr, _LetExpr):
                bindings[expr.name] = resolved

        return TypedProgram(bindings=bindings, typed_exprs=typed,
                            errors=solver.errors, substitution=sub)

    # -- Mutual recursion via SCC ------------------------------------------

    def infer_recursive_group(self, group: TList[TTuple[str, _Expr]],
                              env: TOpt[TypeEnv] = None) -> TDict[str, Type]:
        """Infer types for a mutually recursive binding group.

        Uses Tarjan-style SCC: assign fresh vars, collect, solve, generalize.
        """
        work_env = (env or self._base_env).child_scope()
        fresh_vars: TDict[str, TypeVar] = {}
        for name, _ in group:
            tv = TypeVar(name=name)
            fresh_vars[name] = tv
            work_env.bind_type(name, tv)

        collector = ConstraintCollector(work_env)
        body_types: TDict[str, Type] = {}
        for name, expr in group:
            ty = collector.collect(expr, work_env)
            collector._emit(EqualityConstraint(fresh_vars[name], ty, expr.loc))
            body_types[name] = ty

        solver = ConstraintSolver()
        sub = solver.solve(collector.constraints)

        result: TDict[str, Type] = {}
        for name in fresh_vars:
            resolved = sub.apply(fresh_vars[name])
            result[name] = resolved
        return result

    # -- Incremental re-inference ------------------------------------------

    def incremental_update(self, prev: TypedProgram, changed_indices: TList[int],
                           new_exprs: TList[_Expr]) -> TypedProgram:
        """Re-infer only changed expressions, reusing prior results."""
        program: TList[_Expr] = []
        for i, te in enumerate(prev.typed_exprs):
            if i in changed_indices:
                idx = changed_indices.index(i)
                program.append(new_exprs[idx])
            else:
                program.append(te.expr)

        env = self._base_env.child_scope()
        # Seed environment with unchanged bindings.
        for i, te in enumerate(prev.typed_exprs):
            if i not in changed_indices and isinstance(te.expr, _LetExpr):
                env.bind_type(te.expr.name, te.ty)

        collector = ConstraintCollector(env)
        expr_types: TList[TTuple[_Expr, Type]] = []
        for i, expr in enumerate(program):
            if i in changed_indices:
                ty = collector.collect(expr, env)
                expr_types.append((expr, ty))
            else:
                expr_types.append((expr, prev.typed_exprs[i].ty))

        solver = ConstraintSolver()
        sub = solver.solve(collector.constraints)

        typed: TList[TypedExpr] = []
        bindings: TDict[str, Type] = {}
        for expr, raw_ty in expr_types:
            resolved = sub.apply(raw_ty)
            typed.append(TypedExpr(expr, resolved))
            if isinstance(expr, _LetExpr):
                bindings[expr.name] = resolved

        return TypedProgram(bindings=bindings, typed_exprs=typed,
                            errors=solver.errors, substitution=sub)


# ---------------------------------------------------------------------------
# SCC utility for dependency analysis
# ---------------------------------------------------------------------------

def _compute_sccs(graph: TDict[str, TList[str]]) -> TList[TList[str]]:
    """Tarjan's algorithm for strongly connected components."""
    index_counter = [0]
    stack: TList[str] = []
    on_stack: TSet[str] = set()
    indices: TDict[str, int] = {}
    lowlinks: TDict[str, int] = {}
    result: TList[TList[str]] = []

    def strongconnect(v: str) -> None:
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            component: TList[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            result.append(component)

    for node in graph:
        if node not in indices:
            strongconnect(node)

    return result


def infer_program_with_sccs(
    bindings: TDict[str, _Expr],
    dep_graph: TDict[str, TList[str]],
    engine: TOpt[TypeInferenceEngine] = None,
) -> TDict[str, Type]:
    """Infer types for a set of top-level bindings respecting dependency order.

    ``dep_graph[name]`` lists the names that ``name`` depends on.
    """
    eng = engine or TypeInferenceEngine()
    sccs = _compute_sccs(dep_graph)

    env = _make_builtin_env()
    all_types: TDict[str, Type] = {}

    for scc in reversed(sccs):
        if len(scc) == 1 and scc[0] not in dep_graph.get(scc[0], []):
            # Non-recursive single binding.
            name = scc[0]
            expr = bindings[name]
            collector = ConstraintCollector(env)
            ty = collector.collect(expr, env)
            solver = ConstraintSolver()
            sub = solver.solve(collector.constraints)
            resolved = sub.apply(ty)
            scheme = generalize(env, resolved)
            env.bind(name, scheme)
            all_types[name] = resolved
        else:
            group = [(name, bindings[name]) for name in scc]
            result = eng.infer_recursive_group(group, env)
            for name, ty in result.items():
                scheme = generalize(env, ty)
                env.bind(name, scheme)
                all_types[name] = ty

    return all_types

"""
Nullity domain — three-valued abstract domain for None/null tracking.

Tracks whether a variable is definitely None, definitely not None, or
possibly None. Detects potential null dereferences and models optional
chaining, default-value patterns, and null coalescing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from .base import (
    AbstractDomain,
    AbstractState,
    AbstractTransformer,
    AbstractValue,
    IRNode,
    WideningStrategy,
)


# ===================================================================
# NullityValue – three-valued (+ bottom) nullity abstraction
# ===================================================================


class NullityKind(Enum):
    BOTTOM = auto()           # unreachable
    DEFINITELY_NULL = auto()  # value is definitely None/null
    DEFINITELY_NOT_NULL = auto()  # value is definitely not None/null
    MAYBE_NULL = auto()       # value may or may not be None/null


@dataclass(frozen=True)
class NullityValue(AbstractValue):
    """Abstract nullity value."""

    kind: NullityKind

    # -- singleton instances (cached) ----------------------------------------

    @classmethod
    def bottom(cls) -> "NullityValue":
        return _NV_BOTTOM

    @classmethod
    def definitely_null(cls) -> "NullityValue":
        return _NV_DEFINITELY_NULL

    @classmethod
    def definitely_not_null(cls) -> "NullityValue":
        return _NV_DEFINITELY_NOT_NULL

    @classmethod
    def maybe_null(cls) -> "NullityValue":
        return _NV_MAYBE_NULL

    # -- AbstractValue interface ---------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NullityValue):
            return NotImplemented
        return self.kind == other.kind

    def __hash__(self) -> int:
        return hash(self.kind)

    def __repr__(self) -> str:
        return f"Nullity({self.kind.name})"

    def is_bottom(self) -> bool:
        return self.kind == NullityKind.BOTTOM

    def is_top(self) -> bool:
        return self.kind == NullityKind.MAYBE_NULL

    # -- convenience predicates ----------------------------------------------

    @property
    def is_definitely_null(self) -> bool:
        return self.kind == NullityKind.DEFINITELY_NULL

    @property
    def is_definitely_not_null(self) -> bool:
        return self.kind == NullityKind.DEFINITELY_NOT_NULL

    @property
    def is_maybe_null(self) -> bool:
        return self.kind == NullityKind.MAYBE_NULL

    @property
    def may_be_null(self) -> bool:
        return self.kind in (NullityKind.DEFINITELY_NULL, NullityKind.MAYBE_NULL)

    @property
    def may_be_non_null(self) -> bool:
        return self.kind in (NullityKind.DEFINITELY_NOT_NULL, NullityKind.MAYBE_NULL)


# Singletons
_NV_BOTTOM = NullityValue(kind=NullityKind.BOTTOM)
_NV_DEFINITELY_NULL = NullityValue(kind=NullityKind.DEFINITELY_NULL)
_NV_DEFINITELY_NOT_NULL = NullityValue(kind=NullityKind.DEFINITELY_NOT_NULL)
_NV_MAYBE_NULL = NullityValue(kind=NullityKind.MAYBE_NULL)


# ===================================================================
# NullityDomain – AbstractDomain implementation
# ===================================================================


# Lattice:
#        MAYBE_NULL (⊤)
#       /          \
#  DEF_NULL    DEF_NOT_NULL
#       \          /
#        BOTTOM (⊥)


class NullityDomain(AbstractDomain[NullityValue]):
    """Abstract domain for nullity tracking (4-element lattice)."""

    def top(self) -> NullityValue:
        return NullityValue.maybe_null()

    def bottom(self) -> NullityValue:
        return NullityValue.bottom()

    def join(self, a: NullityValue, b: NullityValue) -> NullityValue:
        if a.is_bottom():
            return b
        if b.is_bottom():
            return a
        if a == b:
            return a
        return NullityValue.maybe_null()

    def meet(self, a: NullityValue, b: NullityValue) -> NullityValue:
        if a.is_top():
            return b
        if b.is_top():
            return a
        if a == b:
            return a
        return NullityValue.bottom()

    def leq(self, a: NullityValue, b: NullityValue) -> bool:
        if a.is_bottom():
            return True
        if b.is_top():
            return True
        return a == b

    def widen(self, a: NullityValue, b: NullityValue) -> NullityValue:
        return self.join(a, b)

    def narrow(self, a: NullityValue, b: NullityValue) -> NullityValue:
        return self.meet(a, b)

    def abstract(self, concrete: Any) -> NullityValue:
        if concrete is None:
            return NullityValue.definitely_null()
        return NullityValue.definitely_not_null()

    def concretize(self, abstract_val: NullityValue) -> Any:
        if abstract_val.is_bottom():
            return set()
        if abstract_val.is_definitely_null:
            return {None}
        if abstract_val.is_definitely_not_null:
            return "non-null values"
        return "possibly null"


# ===================================================================
# NullityWidening
# ===================================================================


class NullityWidening(WideningStrategy[NullityValue]):
    """Conservative widening: any change jumps to MaybeNull."""

    def __init__(self, delay: int = 0):
        self.delay = delay

    def should_widen(self, node_id: int, iteration: int) -> bool:
        return iteration >= self.delay

    def apply(
        self,
        domain: AbstractDomain[NullityValue],
        old: NullityValue,
        new: NullityValue,
        iteration: int,
    ) -> NullityValue:
        if old == new:
            return old
        return NullityValue.maybe_null()


# ===================================================================
# NullityRefiner – refine from guards
# ===================================================================


class NullityRefiner:
    """Refine nullity values from guard conditions."""

    @staticmethod
    def refine_is_none(
        val: NullityValue, *, on_true: bool = True
    ) -> NullityValue:
        """Refine for ``x is None``."""
        if val.is_bottom():
            return val
        if on_true:
            if val.is_definitely_not_null:
                return NullityValue.bottom()
            return NullityValue.definitely_null()
        else:
            if val.is_definitely_null:
                return NullityValue.bottom()
            return NullityValue.definitely_not_null()

    @staticmethod
    def refine_is_not_none(
        val: NullityValue, *, on_true: bool = True
    ) -> NullityValue:
        """Refine for ``x is not None``."""
        return NullityRefiner.refine_is_none(val, on_true=not on_true)

    @staticmethod
    def refine_eq_null(
        val: NullityValue, *, on_true: bool = True
    ) -> NullityValue:
        """Refine for ``x === null`` (TypeScript) or ``x == None`` (Python)."""
        return NullityRefiner.refine_is_none(val, on_true=on_true)

    @staticmethod
    def refine_neq_null(
        val: NullityValue, *, on_true: bool = True
    ) -> NullityValue:
        """Refine for ``x !== null`` (TypeScript)."""
        return NullityRefiner.refine_is_none(val, on_true=not on_true)

    @staticmethod
    def refine_neq_undefined(
        val: NullityValue, *, on_true: bool = True
    ) -> NullityValue:
        """Refine for ``x !== undefined`` (TypeScript)."""
        return NullityRefiner.refine_is_none(val, on_true=not on_true)

    @staticmethod
    def refine_loose_neq_null(
        val: NullityValue, *, on_true: bool = True
    ) -> NullityValue:
        """Refine for ``x != null`` (TypeScript, covers both null and undefined)."""
        return NullityRefiner.refine_is_none(val, on_true=not on_true)

    @staticmethod
    def refine_truthiness(
        val: NullityValue, *, on_true: bool = True
    ) -> NullityValue:
        """Refine based on truthiness check (if x: ...)."""
        if val.is_bottom():
            return val
        if on_true:
            if val.is_definitely_null:
                return NullityValue.bottom()
            # Only narrow to not-null when the value could actually be
            # null (Optional types).  For non-Optional types truthiness
            # means non-zero / non-empty, not non-null.
            if val.is_maybe_null:
                return NullityValue.definitely_not_null()
            return val  # already not-null; nothing to narrow
        else:
            return val  # False branch: could be None or falsy non-None


# ===================================================================
# NullDerefChecker
# ===================================================================


@dataclass
class NullDerefWarning:
    """Warning for a potential null dereference."""

    variable: str
    operation: str  # e.g., "attribute_access", "method_call", "subscript"
    location: Optional[str] = None
    nullity: NullityKind = NullityKind.MAYBE_NULL
    message: str = ""

    def __post_init__(self) -> None:
        if not self.message:
            if self.nullity == NullityKind.DEFINITELY_NULL:
                self.message = (
                    f"Definite null dereference: '{self.variable}' is always None "
                    f"when accessed via {self.operation}"
                )
            else:
                self.message = (
                    f"Potential null dereference: '{self.variable}' may be None "
                    f"when accessed via {self.operation}"
                )


class NullDerefChecker:
    """Detect potential null dereferences in abstract states."""

    def __init__(self) -> None:
        self.warnings: List[NullDerefWarning] = []

    def check_access(
        self,
        state: AbstractState[NullityValue],
        var: str,
        operation: str,
        location: Optional[str] = None,
    ) -> Optional[NullDerefWarning]:
        """Check if accessing *var* could be a null dereference."""
        val = state.get(var)
        if val is None:
            return None

        if val.is_definitely_null:
            warning = NullDerefWarning(
                variable=var,
                operation=operation,
                location=location,
                nullity=NullityKind.DEFINITELY_NULL,
            )
            self.warnings.append(warning)
            return warning

        if val.is_maybe_null:
            warning = NullDerefWarning(
                variable=var,
                operation=operation,
                location=location,
                nullity=NullityKind.MAYBE_NULL,
            )
            self.warnings.append(warning)
            return warning

        return None

    def check_attribute_access(
        self,
        state: AbstractState[NullityValue],
        var: str,
        attr: str,
        location: Optional[str] = None,
    ) -> Optional[NullDerefWarning]:
        return self.check_access(state, var, f"attribute_access(.{attr})", location)

    def check_method_call(
        self,
        state: AbstractState[NullityValue],
        var: str,
        method: str,
        location: Optional[str] = None,
    ) -> Optional[NullDerefWarning]:
        return self.check_access(state, var, f"method_call(.{method}())", location)

    def check_subscript(
        self,
        state: AbstractState[NullityValue],
        var: str,
        location: Optional[str] = None,
    ) -> Optional[NullDerefWarning]:
        return self.check_access(state, var, "subscript([])", location)

    def check_call(
        self,
        state: AbstractState[NullityValue],
        var: str,
        location: Optional[str] = None,
    ) -> Optional[NullDerefWarning]:
        return self.check_access(state, var, "call(())", location)

    def get_warnings(self) -> List[NullDerefWarning]:
        return list(self.warnings)

    def clear_warnings(self) -> None:
        self.warnings.clear()


# ===================================================================
# OptionalChainModeler
# ===================================================================


class OptionalChainModeler:
    """Model optional chaining (?.) semantics.

    ``x?.y`` is equivalent to ``x === null || x === undefined ? undefined : x.y``
    """

    @staticmethod
    def chain_access(
        base_nullity: NullityValue,
        field_nullity: NullityValue,
    ) -> NullityValue:
        """Model the result of ``base?.field``.

        If base is definitely null → result is definitely null.
        If base is definitely not null → result has field's nullity.
        If base is maybe null → result is maybe null.
        """
        if base_nullity.is_bottom():
            return NullityValue.bottom()
        if base_nullity.is_definitely_null:
            return NullityValue.definitely_null()
        if base_nullity.is_definitely_not_null:
            return field_nullity
        # MaybeNull base
        if field_nullity.is_definitely_null:
            return NullityValue.definitely_null()
        return NullityValue.maybe_null()

    @staticmethod
    def chain_call(
        base_nullity: NullityValue,
        return_nullity: NullityValue,
    ) -> NullityValue:
        """Model ``base?.method()``. Same logic as chain_access."""
        return OptionalChainModeler.chain_access(base_nullity, return_nullity)

    @staticmethod
    def chain_subscript(
        base_nullity: NullityValue,
        element_nullity: NullityValue,
    ) -> NullityValue:
        """Model ``base?.[index]``."""
        return OptionalChainModeler.chain_access(base_nullity, element_nullity)


# ===================================================================
# DefaultValueModeler
# ===================================================================


class DefaultValueModeler:
    """Model default value patterns:
    - ``x ?? default`` (nullish coalescing)
    - ``x or default`` (Python or-pattern)
    - ``x if x is not None else default``
    """

    @staticmethod
    def nullish_coalesce(
        value_nullity: NullityValue,
        default_nullity: NullityValue,
    ) -> NullityValue:
        """Model ``value ?? default``.

        If value is definitely not null → result = value (not null)
        If value is definitely null → result = default
        If value is maybe null → result may be either
        """
        if value_nullity.is_bottom():
            return NullityValue.bottom()
        if value_nullity.is_definitely_not_null:
            return NullityValue.definitely_not_null()
        if value_nullity.is_definitely_null:
            return default_nullity
        # MaybeNull → join of not-null and default
        if default_nullity.is_definitely_not_null:
            return NullityValue.definitely_not_null()
        return NullityValue.maybe_null()

    @staticmethod
    def python_or_default(
        value_nullity: NullityValue,
        default_nullity: NullityValue,
    ) -> NullityValue:
        """Model ``value or default`` in Python.

        Note: Python ``or`` triggers on any falsy value, not just None.
        Conservative: treat like nullish coalesce.
        """
        return DefaultValueModeler.nullish_coalesce(value_nullity, default_nullity)

    @staticmethod
    def conditional_default(
        value_nullity: NullityValue,
        default_nullity: NullityValue,
    ) -> NullityValue:
        """Model ``x if x is not None else default``."""
        return DefaultValueModeler.nullish_coalesce(value_nullity, default_nullity)


# ===================================================================
# NullCoalescing – model nullish coalescing and or-default
# ===================================================================


class NullCoalescing:
    """Model nullish coalescing (??) and Python or-default patterns."""

    def __init__(self) -> None:
        self._default_modeler = DefaultValueModeler()

    def coalesce(
        self,
        value: NullityValue,
        default: NullityValue,
    ) -> NullityValue:
        """Apply nullish coalescing: ``value ?? default``."""
        return self._default_modeler.nullish_coalesce(value, default)

    def or_default(
        self,
        value: NullityValue,
        default: NullityValue,
    ) -> NullityValue:
        """Apply Python ``or`` default pattern."""
        return self._default_modeler.python_or_default(value, default)

    def chain_coalesce(
        self,
        values: List[NullityValue],
        final_default: NullityValue,
    ) -> NullityValue:
        """Chain: ``a ?? b ?? c ?? default``."""
        result = final_default
        for val in reversed(values):
            result = self._default_modeler.nullish_coalesce(val, result)
        return result


# ===================================================================
# NullityPropagator
# ===================================================================


class NullityPropagator:
    """Propagate nullity information through assignments and calls."""

    def __init__(self, domain: NullityDomain):
        self.domain = domain
        self.optional_chain = OptionalChainModeler()
        self.default_modeler = DefaultValueModeler()

    def propagate_assignment(
        self,
        state: AbstractState[NullityValue],
        var: str,
        expr: Any,
    ) -> AbstractState[NullityValue]:
        """Propagate nullity through an assignment ``var = expr``."""
        val = self._eval_nullity(state, expr)
        return state.set(var, val)

    def propagate_call_result(
        self,
        state: AbstractState[NullityValue],
        result_var: str,
        func: str,
        args: List[str],
        known_non_null_funcs: Optional[Set[str]] = None,
    ) -> AbstractState[NullityValue]:
        """Propagate nullity for a function call result."""
        if known_non_null_funcs and func in known_non_null_funcs:
            return state.set(result_var, NullityValue.definitely_not_null())
        return state.set(result_var, NullityValue.maybe_null())

    def propagate_container_access(
        self,
        state: AbstractState[NullityValue],
        result_var: str,
        container_var: str,
    ) -> AbstractState[NullityValue]:
        """Container element access → MaybeNull (element might not exist)."""
        container = state.get(container_var)
        if container is not None and container.is_definitely_null:
            return state.set(result_var, NullityValue.bottom())
        return state.set(result_var, NullityValue.maybe_null())

    def propagate_attribute_access(
        self,
        state: AbstractState[NullityValue],
        result_var: str,
        obj_var: str,
        attr: str,
        nullable_attrs: Optional[Set[str]] = None,
    ) -> AbstractState[NullityValue]:
        """Attribute access on an object."""
        obj = state.get(obj_var)
        if obj is not None and obj.is_definitely_null:
            return state.set(result_var, NullityValue.bottom())
        if nullable_attrs and attr in nullable_attrs:
            return state.set(result_var, NullityValue.maybe_null())
        if obj is not None and obj.is_definitely_not_null:
            return state.set(result_var, NullityValue.definitely_not_null())
        return state.set(result_var, NullityValue.maybe_null())

    def _eval_nullity(
        self,
        state: AbstractState[NullityValue],
        expr: Any,
    ) -> NullityValue:
        if expr is None:
            return NullityValue.definitely_null()
        if isinstance(expr, str):
            val = state.get(expr)
            return val if val is not None else NullityValue.maybe_null()
        if isinstance(expr, (int, float, bool)):
            return NullityValue.definitely_not_null()
        if isinstance(expr, (list, tuple)):
            if len(expr) >= 2:
                op = expr[0]
                if op == "optional_chain":
                    base_val = self._eval_nullity(state, expr[1])
                    field_val = (
                        self._eval_nullity(state, expr[2])
                        if len(expr) > 2
                        else NullityValue.maybe_null()
                    )
                    return self.optional_chain.chain_access(base_val, field_val)
                if op == "??":
                    val = self._eval_nullity(state, expr[1])
                    default = self._eval_nullity(state, expr[2]) if len(expr) > 2 else NullityValue.definitely_not_null()
                    return self.default_modeler.nullish_coalesce(val, default)
                if op == "or":
                    val = self._eval_nullity(state, expr[1])
                    default = self._eval_nullity(state, expr[2]) if len(expr) > 2 else NullityValue.definitely_not_null()
                    return self.default_modeler.python_or_default(val, default)
                if op in {"+", "-", "*", "/", "//", "%", "**"}:
                    return NullityValue.definitely_not_null()
                if op in {"<", "<=", ">", ">=", "==", "!="}:
                    return NullityValue.definitely_not_null()
        return NullityValue.maybe_null()


# ===================================================================
# NullityTransformer – full AbstractTransformer
# ===================================================================


class NullityTransformer(AbstractTransformer[NullityValue]):
    """Abstract transformer for the nullity domain."""

    def __init__(self, domain: NullityDomain):
        self.domain = domain
        self.refiner = NullityRefiner()
        self.propagator = NullityPropagator(domain)
        self.deref_checker = NullDerefChecker()

    def assign(
        self,
        state: AbstractState[NullityValue],
        var: str,
        expr: Any,
    ) -> AbstractState[NullityValue]:
        return self.propagator.propagate_assignment(state, var, expr)

    def guard(
        self,
        state: AbstractState[NullityValue],
        condition: Any,
        branch: bool,
    ) -> AbstractState[NullityValue]:
        if not isinstance(condition, (list, tuple)):
            return state

        if len(condition) == 3:
            op, arg1, arg2 = condition

            if op == "is" and arg2 == "None":
                return self._refine_var(state, arg1, NullityRefiner.refine_is_none, branch)
            if op == "is not" and arg2 == "None":
                return self._refine_var(state, arg1, NullityRefiner.refine_is_not_none, branch)
            if op == "===" and arg2 in ("null", "None"):
                return self._refine_var(state, arg1, NullityRefiner.refine_eq_null, branch)
            if op == "!==" and arg2 in ("null", "None"):
                return self._refine_var(state, arg1, NullityRefiner.refine_neq_null, branch)
            if op == "!==" and arg2 == "undefined":
                return self._refine_var(state, arg1, NullityRefiner.refine_neq_undefined, branch)
            if op == "!=" and arg2 in ("null", "None"):
                return self._refine_var(state, arg1, NullityRefiner.refine_loose_neq_null, branch)

        elif len(condition) == 2:
            op, arg = condition
            if op == "truthiness":
                return self._refine_var(state, arg, NullityRefiner.refine_truthiness, branch)

        return state

    def call(
        self,
        state: AbstractState[NullityValue],
        func: str,
        args: List[Any],
        result_var: Optional[str] = None,
    ) -> AbstractState[NullityValue]:
        _NON_NULL_FUNCS = {
            "len", "int", "float", "str", "bool", "list", "tuple",
            "set", "dict", "abs", "min", "max", "sum", "round",
            "range", "sorted", "enumerate", "zip", "map", "filter",
            "repr", "type", "id", "hash", "ord", "chr",
            "hex", "oct", "bin", "isinstance", "issubclass",
            "hasattr", "callable", "print", "input",
        }
        if result_var is not None:
            return self.propagator.propagate_call_result(
                state, result_var, func, [], _NON_NULL_FUNCS
            )
        return state

    def _refine_var(
        self,
        state: AbstractState[NullityValue],
        var: str,
        refiner_fn: Any,
        branch: bool,
    ) -> AbstractState[NullityValue]:
        if not isinstance(var, str):
            return state
        val = state.get(var)
        if val is None:
            val = NullityValue.maybe_null()
        refined = refiner_fn(val, on_true=branch)
        return state.set(var, refined)

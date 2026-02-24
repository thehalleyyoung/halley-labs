"""
Reduced product of all four domains with bidirectional inter-domain
communication.

Combines:
  - Interval domain (numeric tracking)
  - TypeTag domain (type-tag powerset)
  - Nullity domain (None/null tracking)

with reductions that propagate information between domains:
  - Type-tag → Nullity: isinstance(x, NoneType) ↔ is_none(x)
  - Type-tag → Numeric: isinstance(x, int) → x ∈ ℤ (activates numeric tracking)
  - Nullity → Type-tag: ¬is_none(x) → Tag(x) ≠ NoneType
  - Numeric → Nullity: x > 0 → ¬is_none(x) (int can't be None)
  - Truthiness → all: is_truthy(x) → ¬is_none(x) ∧ (if int: x ≠ 0) ∧ (if str: len(x) > 0)
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

from .base import (
    AbstractDomain,
    AbstractState,
    AbstractTransformer,
    AbstractValue,
    IRNode,
    WideningStrategy,
)
from .intervals import (
    Bound,
    Interval,
    IntervalDomain,
    IntervalTransformer,
    IntervalValue,
)
from .nullity import (
    NullDerefChecker,
    NullityDomain,
    NullityKind,
    NullityRefiner,
    NullityTransformer,
    NullityValue,
)
from .typetags import (
    TypeHierarchy,
    TypeTagDomain,
    TypeTagSet,
    TypeTagTransformer,
    TypeTagValue,
)


# ===================================================================
# ProductValue – tuple of sub-domain values
# ===================================================================


@dataclass(frozen=True)
class ProductValue(AbstractValue):
    """Product of (interval, type-tag, nullity) abstract values."""

    interval: IntervalValue
    type_tag: TypeTagValue
    nullity: NullityValue

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProductValue):
            return NotImplemented
        return (
            self.interval == other.interval
            and self.type_tag == other.type_tag
            and self.nullity == other.nullity
        )

    def __hash__(self) -> int:
        return hash((self.interval, self.type_tag, self.nullity))

    def __repr__(self) -> str:
        return (
            f"ProductValue(iv={self.interval}, "
            f"tag={self.type_tag}, "
            f"null={self.nullity})"
        )

    def is_bottom(self) -> bool:
        return (
            self.interval.is_bottom()
            or self.type_tag.is_bottom()
            or self.nullity.is_bottom()
        )

    def is_top(self) -> bool:
        return (
            self.interval.is_top()
            and self.type_tag.is_top()
            and self.nullity.is_top()
        )

    def with_interval(self, iv: IntervalValue) -> "ProductValue":
        return ProductValue(interval=iv, type_tag=self.type_tag, nullity=self.nullity)

    def with_type_tag(self, tt: TypeTagValue) -> "ProductValue":
        return ProductValue(interval=self.interval, type_tag=tt, nullity=self.nullity)

    def with_nullity(self, nv: NullityValue) -> "ProductValue":
        return ProductValue(interval=self.interval, type_tag=self.type_tag, nullity=nv)


# ===================================================================
# Reduction ABC and implementations
# ===================================================================


class Reduction(ABC):
    """A single inter-domain reduction rule."""

    @abstractmethod
    def apply(self, value: ProductValue) -> ProductValue:
        """Apply this reduction, potentially refining the product value."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class TypeTagToNullityReduction(Reduction):
    """isinstance(x, NoneType) ↔ null.

    If type-tag is {NoneType} → definitely null.
    If type-tag does not contain NoneType → definitely not null.
    """

    def apply(self, value: ProductValue) -> ProductValue:
        if value.is_bottom():
            return value

        tag_set = value.type_tag.tag_set

        if tag_set.is_bottom:
            return ProductValue(
                interval=IntervalValue(Interval.bottom()),
                type_tag=value.type_tag,
                nullity=NullityValue.bottom(),
            )

        if not tag_set.is_top:
            if tag_set.tags == frozenset({"NoneType"}):
                if not value.nullity.is_definitely_null and not value.nullity.is_bottom():
                    return value.with_nullity(NullityValue.definitely_null())
            elif "NoneType" not in tag_set.tags:
                if not value.nullity.is_definitely_not_null and not value.nullity.is_bottom():
                    return value.with_nullity(NullityValue.definitely_not_null())

        return value

    def name(self) -> str:
        return "TypeTagToNullity"


class NullityToTypeTagReduction(Reduction):
    """¬is_none(x) → Tag(x) ≠ NoneType.

    If definitely not null → remove NoneType from tags.
    If definitely null → restrict to {NoneType}.
    """

    def apply(self, value: ProductValue) -> ProductValue:
        if value.is_bottom():
            return value

        if value.nullity.is_definitely_not_null:
            new_tags = value.type_tag.tag_set.remove("NoneType")
            if new_tags != value.type_tag.tag_set:
                return value.with_type_tag(TypeTagValue(new_tags))

        if value.nullity.is_definitely_null:
            if not value.type_tag.tag_set.is_bottom:
                expected = TypeTagSet.singleton("NoneType")
                if value.type_tag.tag_set != expected:
                    new_tags = value.type_tag.tag_set.meet(expected)
                    return value.with_type_tag(TypeTagValue(new_tags))

        return value

    def name(self) -> str:
        return "NullityToTypeTag"


class TypeTagToNumericReduction(Reduction):
    """isinstance(x, int) → activate interval tracking.

    If type-tag does not contain int or bool → set interval to ⊥.
    If type-tag is only non-numeric types → interval is irrelevant (⊥).
    """

    _NUMERIC_TAGS = frozenset({"int", "bool", "float"})

    def apply(self, value: ProductValue) -> ProductValue:
        if value.is_bottom():
            return value

        tag_set = value.type_tag.tag_set
        if tag_set.is_top or tag_set.is_bottom:
            return value

        has_numeric = bool(tag_set.tags & self._NUMERIC_TAGS)
        if not has_numeric:
            if not value.interval.interval.is_bottom:
                return value.with_interval(IntervalValue(Interval.bottom()))

        return value

    def name(self) -> str:
        return "TypeTagToNumeric"


class NumericToNullityReduction(Reduction):
    """Numeric constraints → not null.

    If interval is not ⊥ (i.e., has concrete numeric info) → definitely not null.
    Rationale: an int/float/bool can never be None.
    """

    def apply(self, value: ProductValue) -> ProductValue:
        if value.is_bottom():
            return value

        iv = value.interval.interval
        if not iv.is_bottom and not iv.is_top:
            if value.nullity.is_maybe_null:
                return value.with_nullity(NullityValue.definitely_not_null())

        return value

    def name(self) -> str:
        return "NumericToNullity"


class TruthinessReduction(Reduction):
    """Truthiness → affects all domains.

    is_truthy(x) →
      ¬is_none(x) ∧
      (if int: x ≠ 0) ∧
      (if str: len(x) > 0)

    This reduction is applied after a truthiness guard has been processed.
    It's typically triggered by the product transformer, not during general
    reduction passes.
    """

    def __init__(self, is_truthy: bool = True):
        self._is_truthy = is_truthy

    def apply(self, value: ProductValue) -> ProductValue:
        if value.is_bottom():
            return value

        if self._is_truthy:
            result = value

            # Not null
            if result.nullity.may_be_null:
                result = result.with_nullity(NullityValue.definitely_not_null())

            # Remove NoneType
            if not result.type_tag.tag_set.is_top and not result.type_tag.tag_set.is_bottom:
                new_tags = result.type_tag.tag_set.remove("NoneType")
                if new_tags != result.type_tag.tag_set:
                    result = result.with_type_tag(TypeTagValue(new_tags))

            # If int: x ≠ 0
            tag_set = result.type_tag.tag_set
            if not tag_set.is_top and tag_set.tags <= {"int", "bool"}:
                iv = result.interval.interval
                if iv.contains(0):
                    # Remove 0 from interval
                    if iv.lo == Bound.finite(0):
                        new_iv = Interval(lo=Bound.finite(1), hi=iv.hi)
                    elif iv.hi == Bound.finite(0):
                        new_iv = Interval(lo=iv.lo, hi=Bound.finite(-1))
                    else:
                        new_iv = iv  # Can't precisely remove 0 from middle
                    result = result.with_interval(IntervalValue(new_iv))

            return result
        else:
            # Falsy: could be None, 0, empty string, empty container, False
            return value

    def name(self) -> str:
        return f"Truthiness(truthy={self._is_truthy})"


# ===================================================================
# ReductionEngine
# ===================================================================


class ReductionEngine:
    """Apply inter-domain reductions until fixed point."""

    def __init__(
        self,
        reductions: Optional[List[Reduction]] = None,
        max_iterations: int = 10,
    ):
        if reductions is None:
            self.reductions: List[Reduction] = [
                TypeTagToNullityReduction(),
                NullityToTypeTagReduction(),
                TypeTagToNumericReduction(),
                NumericToNullityReduction(),
            ]
        else:
            self.reductions = reductions
        self.max_iterations = max_iterations

    def reduce(self, value: ProductValue) -> ProductValue:
        """Apply all reductions until the value stabilizes."""
        current = value
        for _ in range(self.max_iterations):
            next_val = current
            for reduction in self.reductions:
                next_val = reduction.apply(next_val)
                if next_val.is_bottom():
                    return next_val
            if next_val == current:
                break
            current = next_val
        return current

    def reduce_state(self, state: AbstractState[ProductValue]) -> AbstractState[ProductValue]:
        """Reduce all values in a state."""
        new_env: Dict[str, ProductValue] = {}
        for var, val in state.items():
            reduced = self.reduce(val)
            new_env[var] = reduced
        return AbstractState(env=new_env, domain=state.domain, _is_bottom=state._is_bottom)


# ===================================================================
# ReducedProductDomain
# ===================================================================


class ReducedProductDomain(AbstractDomain[ProductValue]):
    """Reduced product of interval × type-tag × nullity domains."""

    def __init__(
        self,
        hierarchy: Optional[TypeHierarchy] = None,
        reduction_engine: Optional[ReductionEngine] = None,
    ):
        self.interval_domain = IntervalDomain()
        self.type_tag_domain = TypeTagDomain(hierarchy=hierarchy)
        self.nullity_domain = NullityDomain()
        self.reducer = reduction_engine or ReductionEngine()

    def top(self) -> ProductValue:
        return ProductValue(
            interval=self.interval_domain.top(),
            type_tag=self.type_tag_domain.top(),
            nullity=self.nullity_domain.top(),
        )

    def bottom(self) -> ProductValue:
        return ProductValue(
            interval=self.interval_domain.bottom(),
            type_tag=self.type_tag_domain.bottom(),
            nullity=self.nullity_domain.bottom(),
        )

    def join(self, a: ProductValue, b: ProductValue) -> ProductValue:
        raw = ProductValue(
            interval=self.interval_domain.join(a.interval, b.interval),
            type_tag=self.type_tag_domain.join(a.type_tag, b.type_tag),
            nullity=self.nullity_domain.join(a.nullity, b.nullity),
        )
        return self.reducer.reduce(raw)

    def meet(self, a: ProductValue, b: ProductValue) -> ProductValue:
        raw = ProductValue(
            interval=self.interval_domain.meet(a.interval, b.interval),
            type_tag=self.type_tag_domain.meet(a.type_tag, b.type_tag),
            nullity=self.nullity_domain.meet(a.nullity, b.nullity),
        )
        return self.reducer.reduce(raw)

    def leq(self, a: ProductValue, b: ProductValue) -> bool:
        return (
            self.interval_domain.leq(a.interval, b.interval)
            and self.type_tag_domain.leq(a.type_tag, b.type_tag)
            and self.nullity_domain.leq(a.nullity, b.nullity)
        )

    def widen(self, a: ProductValue, b: ProductValue) -> ProductValue:
        raw = ProductValue(
            interval=self.interval_domain.widen(a.interval, b.interval),
            type_tag=self.type_tag_domain.widen(a.type_tag, b.type_tag),
            nullity=self.nullity_domain.widen(a.nullity, b.nullity),
        )
        return self.reducer.reduce(raw)

    def narrow(self, a: ProductValue, b: ProductValue) -> ProductValue:
        raw = ProductValue(
            interval=self.interval_domain.narrow(a.interval, b.interval),
            type_tag=self.type_tag_domain.narrow(a.type_tag, b.type_tag),
            nullity=self.nullity_domain.narrow(a.nullity, b.nullity),
        )
        return self.reducer.reduce(raw)

    def is_bottom(self, a: ProductValue) -> bool:
        return a.is_bottom()

    def is_top(self, a: ProductValue) -> bool:
        return a.is_top()

    def abstract(self, concrete: Any) -> ProductValue:
        iv = self.interval_domain.abstract(concrete)
        tt = self.type_tag_domain.abstract(concrete)
        nv = self.nullity_domain.abstract(concrete)
        raw = ProductValue(interval=iv, type_tag=tt, nullity=nv)
        return self.reducer.reduce(raw)

    def concretize(self, abstract_val: ProductValue) -> Any:
        return {
            "interval": self.interval_domain.concretize(abstract_val.interval),
            "type_tag": self.type_tag_domain.concretize(abstract_val.type_tag),
            "nullity": self.nullity_domain.concretize(abstract_val.nullity),
        }


# ===================================================================
# ProductJoin / ProductMeet / ProductWidening
# ===================================================================


class ProductJoin:
    """Component-wise join followed by reduction."""

    def __init__(self, domain: ReducedProductDomain):
        self.domain = domain

    def join(self, a: ProductValue, b: ProductValue) -> ProductValue:
        return self.domain.join(a, b)

    def join_many(self, values: List[ProductValue]) -> ProductValue:
        if not values:
            return self.domain.bottom()
        result = values[0]
        for v in values[1:]:
            result = self.domain.join(result, v)
        return result


class ProductMeet:
    """Component-wise meet followed by reduction."""

    def __init__(self, domain: ReducedProductDomain):
        self.domain = domain

    def meet(self, a: ProductValue, b: ProductValue) -> ProductValue:
        return self.domain.meet(a, b)

    def meet_many(self, values: List[ProductValue]) -> ProductValue:
        if not values:
            return self.domain.top()
        result = values[0]
        for v in values[1:]:
            result = self.domain.meet(result, v)
        return result


class ProductWidening(WideningStrategy[ProductValue]):
    """Component-wise widening followed by reduction."""

    def __init__(self, domain: ReducedProductDomain, delay: int = 3):
        self.domain = domain
        self.delay = delay

    def should_widen(self, node_id: int, iteration: int) -> bool:
        return iteration >= self.delay

    def apply(
        self,
        domain: AbstractDomain[ProductValue],
        old: ProductValue,
        new: ProductValue,
        iteration: int,
    ) -> ProductValue:
        if iteration < self.delay:
            return domain.join(old, new)
        return self.domain.widen(old, new)


# ===================================================================
# ProductState
# ===================================================================


class ProductState:
    """Convenience wrapper for AbstractState[ProductValue]."""

    def __init__(
        self,
        domain: ReducedProductDomain,
        state: Optional[AbstractState[ProductValue]] = None,
    ):
        self.domain = domain
        self.state = state or AbstractState(env={}, domain=domain)
        self._reducer = domain.reducer

    def get(self, var: str) -> Optional[ProductValue]:
        return self.state.get(var)

    def set(self, var: str, value: ProductValue) -> "ProductState":
        new_state = self.state.set(var, self._reducer.reduce(value))
        return ProductState(self.domain, new_state)

    def get_interval(self, var: str) -> Optional[IntervalValue]:
        v = self.state.get(var)
        return v.interval if v is not None else None

    def get_type_tag(self, var: str) -> Optional[TypeTagValue]:
        v = self.state.get(var)
        return v.type_tag if v is not None else None

    def get_nullity(self, var: str) -> Optional[NullityValue]:
        v = self.state.get(var)
        return v.nullity if v is not None else None

    def set_interval(self, var: str, iv: IntervalValue) -> "ProductState":
        current = self.state.get(var) or self.domain.top()
        return self.set(var, current.with_interval(iv))

    def set_type_tag(self, var: str, tt: TypeTagValue) -> "ProductState":
        current = self.state.get(var) or self.domain.top()
        return self.set(var, current.with_type_tag(tt))

    def set_nullity(self, var: str, nv: NullityValue) -> "ProductState":
        current = self.state.get(var) or self.domain.top()
        return self.set(var, current.with_nullity(nv))

    def reduce(self) -> "ProductState":
        new_state = self._reducer.reduce_state(self.state)
        return ProductState(self.domain, new_state)

    def join(self, other: "ProductState") -> "ProductState":
        new_state = self.state.join(other.state)
        return ProductState(self.domain, new_state).reduce()

    def meet(self, other: "ProductState") -> "ProductState":
        new_state = self.state.meet(other.state)
        return ProductState(self.domain, new_state).reduce()

    def variables(self) -> Set[str]:
        return self.state.variables()

    @property
    def is_bottom(self) -> bool:
        return self.state.is_bottom

    def __repr__(self) -> str:
        return f"ProductState({self.state})"


# ===================================================================
# ProductTransformer
# ===================================================================


class ProductTransformer(AbstractTransformer[ProductValue]):
    """Abstract transformer that delegates to sub-domains and applies reductions."""

    def __init__(self, domain: ReducedProductDomain):
        self.domain = domain
        self.interval_xf = IntervalTransformer(domain.interval_domain)
        self.type_tag_xf = TypeTagTransformer(domain.type_tag_domain)
        self.nullity_xf = NullityTransformer(domain.nullity_domain)
        self.reducer = domain.reducer

    def assign(
        self, state: AbstractState[ProductValue], var: str, expr: Any
    ) -> AbstractState[ProductValue]:
        iv_state = self._project_interval(state)
        tt_state = self._project_type_tag(state)
        nv_state = self._project_nullity(state)

        new_iv = self.interval_xf.assign(iv_state, var, expr)
        new_tt = self.type_tag_xf.assign(tt_state, var, expr)
        new_nv = self.nullity_xf.assign(nv_state, var, expr)

        return self._combine_states(state, var, new_iv, new_tt, new_nv)

    def guard(
        self, state: AbstractState[ProductValue], condition: Any, branch: bool
    ) -> AbstractState[ProductValue]:
        iv_state = self._project_interval(state)
        tt_state = self._project_type_tag(state)
        nv_state = self._project_nullity(state)

        new_iv = self.interval_xf.guard(iv_state, condition, branch)
        new_tt = self.type_tag_xf.guard(tt_state, condition, branch)
        new_nv = self.nullity_xf.guard(nv_state, condition, branch)

        result = self._combine_all(state, new_iv, new_tt, new_nv)

        # Apply truthiness reduction if applicable
        if isinstance(condition, (list, tuple)) and len(condition) == 2:
            op, arg = condition
            if op == "truthiness" and isinstance(arg, str):
                val = result.get(arg)
                if val is not None:
                    truthy_reduction = TruthinessReduction(is_truthy=branch)
                    reduced = truthy_reduction.apply(val)
                    result = result.set(arg, reduced)

        return result

    def call(
        self,
        state: AbstractState[ProductValue],
        func: str,
        args: List[Any],
        result_var: Optional[str] = None,
    ) -> AbstractState[ProductValue]:
        iv_state = self._project_interval(state)
        tt_state = self._project_type_tag(state)
        nv_state = self._project_nullity(state)

        new_iv = self.interval_xf.call(iv_state, func, args, result_var)
        new_tt = self.type_tag_xf.call(tt_state, func, args, result_var)
        new_nv = self.nullity_xf.call(nv_state, func, args, result_var)

        if result_var is not None:
            return self._combine_states(state, result_var, new_iv, new_tt, new_nv)
        return state

    # -- projection helpers --------------------------------------------------

    def _project_interval(
        self, state: AbstractState[ProductValue]
    ) -> AbstractState[IntervalValue]:
        env = {k: v.interval for k, v in state.items()}
        return AbstractState(env=env, domain=self.domain.interval_domain)

    def _project_type_tag(
        self, state: AbstractState[ProductValue]
    ) -> AbstractState[TypeTagValue]:
        env = {k: v.type_tag for k, v in state.items()}
        return AbstractState(env=env, domain=self.domain.type_tag_domain)

    def _project_nullity(
        self, state: AbstractState[ProductValue]
    ) -> AbstractState[NullityValue]:
        env = {k: v.nullity for k, v in state.items()}
        return AbstractState(env=env, domain=self.domain.nullity_domain)

    def _combine_states(
        self,
        original: AbstractState[ProductValue],
        var: str,
        iv_state: AbstractState[IntervalValue],
        tt_state: AbstractState[TypeTagValue],
        nv_state: AbstractState[NullityValue],
    ) -> AbstractState[ProductValue]:
        new_env = dict(original.env)
        iv_val = iv_state.get(var) or self.domain.interval_domain.top()
        tt_val = tt_state.get(var) or self.domain.type_tag_domain.top()
        nv_val = nv_state.get(var) or self.domain.nullity_domain.top()
        raw = ProductValue(interval=iv_val, type_tag=tt_val, nullity=nv_val)
        new_env[var] = self.reducer.reduce(raw)
        return AbstractState(env=new_env, domain=original.domain)

    def _combine_all(
        self,
        original: AbstractState[ProductValue],
        iv_state: AbstractState[IntervalValue],
        tt_state: AbstractState[TypeTagValue],
        nv_state: AbstractState[NullityValue],
    ) -> AbstractState[ProductValue]:
        all_vars = original.variables() | iv_state.variables() | tt_state.variables() | nv_state.variables()
        new_env: Dict[str, ProductValue] = {}
        for var in all_vars:
            orig = original.get(var)
            iv_val = iv_state.get(var) or (orig.interval if orig else self.domain.interval_domain.top())
            tt_val = tt_state.get(var) or (orig.type_tag if orig else self.domain.type_tag_domain.top())
            nv_val = nv_state.get(var) or (orig.nullity if orig else self.domain.nullity_domain.top())
            raw = ProductValue(interval=iv_val, type_tag=tt_val, nullity=nv_val)
            new_env[var] = self.reducer.reduce(raw)
        return AbstractState(env=new_env, domain=original.domain)


# ===================================================================
# GuardTransferFunction
# ===================================================================


class GuardTransferFunction:
    """Applies guard conditions to a ProductState, refining all domains."""

    def __init__(self, domain: ReducedProductDomain):
        self.domain = domain
        self.transformer = ProductTransformer(domain)

    def apply_guard(
        self,
        state: AbstractState[ProductValue],
        condition: Any,
        branch: bool,
    ) -> AbstractState[ProductValue]:
        return self.transformer.guard(state, condition, branch)

    def apply_isinstance(
        self,
        state: AbstractState[ProductValue],
        var: str,
        type_name: str,
        branch: bool,
    ) -> AbstractState[ProductValue]:
        """Apply isinstance guard with cross-domain reduction."""
        condition = ("isinstance", var, type_name)
        result = self.transformer.guard(state, condition, branch)

        # Additional cross-domain reductions
        val = result.get(var)
        if val is not None:
            if branch and type_name == "NoneType":
                val = val.with_nullity(NullityValue.definitely_null())
            elif not branch and type_name == "NoneType":
                val = val.with_nullity(NullityValue.definitely_not_null())
            elif branch and type_name in ("int", "bool", "float"):
                val = val.with_nullity(NullityValue.definitely_not_null())

            val = self.domain.reducer.reduce(val)
            result = result.set(var, val)

        return result

    def apply_is_none(
        self,
        state: AbstractState[ProductValue],
        var: str,
        branch: bool,
    ) -> AbstractState[ProductValue]:
        """Apply ``x is None`` guard."""
        condition = ("is", var, "None")
        result = self.transformer.guard(state, condition, branch)

        val = result.get(var)
        if val is not None:
            if branch:
                val = val.with_type_tag(TypeTagValue(TypeTagSet.singleton("NoneType")))
                val = val.with_nullity(NullityValue.definitely_null())
                val = val.with_interval(IntervalValue(Interval.bottom()))
            else:
                new_tags = val.type_tag.tag_set.remove("NoneType")
                val = val.with_type_tag(TypeTagValue(new_tags))
                val = val.with_nullity(NullityValue.definitely_not_null())
            val = self.domain.reducer.reduce(val)
            result = result.set(var, val)

        return result

    def apply_comparison(
        self,
        state: AbstractState[ProductValue],
        op: str,
        lhs_var: str,
        rhs: Any,
        branch: bool,
    ) -> AbstractState[ProductValue]:
        """Apply comparison guard (e.g., x < 5)."""
        condition = (op, lhs_var, rhs)
        return self.transformer.guard(state, condition, branch)

    def apply_truthiness(
        self,
        state: AbstractState[ProductValue],
        var: str,
        branch: bool,
    ) -> AbstractState[ProductValue]:
        """Apply truthiness guard (if x: ...)."""
        condition = ("truthiness", var)
        return self.transformer.guard(state, condition, branch)


# ===================================================================
# CallTransferFunction
# ===================================================================


class CallTransferFunction:
    """Models function calls in the product domain."""

    def __init__(self, domain: ReducedProductDomain):
        self.domain = domain
        self.transformer = ProductTransformer(domain)

    def apply_call(
        self,
        state: AbstractState[ProductValue],
        func: str,
        args: List[Any],
        result_var: Optional[str] = None,
    ) -> AbstractState[ProductValue]:
        return self.transformer.call(state, func, args, result_var)

    def apply_constructor(
        self,
        state: AbstractState[ProductValue],
        class_name: str,
        args: List[Any],
        result_var: str,
    ) -> AbstractState[ProductValue]:
        """Model a constructor call: result_var = ClassName(args)."""
        result = state
        product_val = ProductValue(
            interval=IntervalValue(Interval.top()),
            type_tag=TypeTagValue(TypeTagSet.singleton(class_name)),
            nullity=NullityValue.definitely_not_null(),
        )
        if class_name == "int":
            product_val = product_val.with_interval(IntervalValue(Interval.top()))
        elif class_name == "bool":
            product_val = product_val.with_interval(IntervalValue(Interval.from_bounds(0, 1)))
        elif class_name in ("str", "list", "tuple", "dict", "set", "bytes"):
            product_val = product_val.with_interval(IntervalValue(Interval.bottom()))

        reduced = self.domain.reducer.reduce(product_val)
        return result.set(result_var, reduced)

    def apply_method_call(
        self,
        state: AbstractState[ProductValue],
        obj_var: str,
        method: str,
        args: List[Any],
        result_var: Optional[str] = None,
    ) -> AbstractState[ProductValue]:
        """Model a method call: result_var = obj.method(args)."""
        obj_val = state.get(obj_var)
        if obj_val is None:
            if result_var:
                return state.set(result_var, self.domain.top())
            return state

        # Check for null deref
        if obj_val.nullity.is_definitely_null:
            if result_var:
                return state.set(result_var, self.domain.bottom())
            return state

        if result_var:
            result_type = TypeTagSet.top()
            tag_set = obj_val.type_tag.tag_set

            # Infer result type from known methods
            if not tag_set.is_top and not tag_set.is_bottom:
                if tag_set.tags <= {"str"}:
                    if method in ("upper", "lower", "strip", "lstrip", "rstrip",
                                  "replace", "format", "join", "capitalize",
                                  "title", "center", "ljust", "rjust", "zfill",
                                  "removeprefix", "removesuffix"):
                        result_type = TypeTagSet.singleton("str")
                    elif method in ("split", "rsplit"):
                        result_type = TypeTagSet.singleton("list")
                    elif method in ("find", "rfind", "index", "rindex", "count"):
                        result_type = TypeTagSet.singleton("int")
                    elif method in ("startswith", "endswith", "isdigit", "isalpha",
                                    "isalnum", "isspace", "isupper", "islower"):
                        result_type = TypeTagSet.singleton("bool")
                elif tag_set.tags <= {"list"}:
                    if method in ("copy",):
                        result_type = TypeTagSet.singleton("list")
                    elif method in ("index", "count"):
                        result_type = TypeTagSet.singleton("int")
                    elif method == "sort":
                        result_type = TypeTagSet.singleton("NoneType")
                elif tag_set.tags <= {"dict"}:
                    if method == "keys":
                        result_type = TypeTagSet.singleton("dict")
                    elif method == "values":
                        result_type = TypeTagSet.singleton("dict")
                    elif method == "items":
                        result_type = TypeTagSet.singleton("list")
                    elif method == "copy":
                        result_type = TypeTagSet.singleton("dict")
                    elif method == "get":
                        result_type = TypeTagSet.top()
                    elif method == "pop":
                        result_type = TypeTagSet.top()

            nullity = NullityValue.maybe_null()
            if result_type.is_singleton:
                t = result_type.single_tag()
                if t and t != "NoneType":
                    nullity = NullityValue.definitely_not_null()

            result_val = ProductValue(
                interval=IntervalValue(Interval.top()),
                type_tag=TypeTagValue(result_type),
                nullity=nullity,
            )
            return state.set(result_var, self.domain.reducer.reduce(result_val))

        return state


# ===================================================================
# AssignTransferFunction
# ===================================================================


class AssignTransferFunction:
    """Models assignments in the product domain."""

    def __init__(self, domain: ReducedProductDomain):
        self.domain = domain
        self.transformer = ProductTransformer(domain)

    def apply_assign(
        self,
        state: AbstractState[ProductValue],
        var: str,
        expr: Any,
    ) -> AbstractState[ProductValue]:
        return self.transformer.assign(state, var, expr)

    def apply_none_assign(
        self,
        state: AbstractState[ProductValue],
        var: str,
    ) -> AbstractState[ProductValue]:
        """Assign None to a variable."""
        val = ProductValue(
            interval=IntervalValue(Interval.bottom()),
            type_tag=TypeTagValue(TypeTagSet.singleton("NoneType")),
            nullity=NullityValue.definitely_null(),
        )
        return state.set(var, val)

    def apply_int_assign(
        self,
        state: AbstractState[ProductValue],
        var: str,
        value: int,
    ) -> AbstractState[ProductValue]:
        """Assign an integer literal."""
        val = ProductValue(
            interval=IntervalValue(Interval.singleton(value)),
            type_tag=TypeTagValue(TypeTagSet.singleton("int")),
            nullity=NullityValue.definitely_not_null(),
        )
        return state.set(var, val)

    def apply_str_assign(
        self,
        state: AbstractState[ProductValue],
        var: str,
        value: Optional[str] = None,
    ) -> AbstractState[ProductValue]:
        """Assign a string literal."""
        val = ProductValue(
            interval=IntervalValue(Interval.bottom()),
            type_tag=TypeTagValue(TypeTagSet.singleton("str")),
            nullity=NullityValue.definitely_not_null(),
        )
        return state.set(var, val)

    def apply_bool_assign(
        self,
        state: AbstractState[ProductValue],
        var: str,
        value: bool,
    ) -> AbstractState[ProductValue]:
        """Assign a boolean literal."""
        int_val = 1 if value else 0
        val = ProductValue(
            interval=IntervalValue(Interval.singleton(int_val)),
            type_tag=TypeTagValue(TypeTagSet.singleton("bool")),
            nullity=NullityValue.definitely_not_null(),
        )
        return state.set(var, val)

    def apply_float_assign(
        self,
        state: AbstractState[ProductValue],
        var: str,
    ) -> AbstractState[ProductValue]:
        """Assign a float literal."""
        val = ProductValue(
            interval=IntervalValue(Interval.top()),
            type_tag=TypeTagValue(TypeTagSet.singleton("float")),
            nullity=NullityValue.definitely_not_null(),
        )
        return state.set(var, val)

    def apply_container_assign(
        self,
        state: AbstractState[ProductValue],
        var: str,
        container_type: str,
        known_size: Optional[int] = None,
    ) -> AbstractState[ProductValue]:
        """Assign a container (list, dict, set, tuple)."""
        val = ProductValue(
            interval=IntervalValue(Interval.bottom()),
            type_tag=TypeTagValue(TypeTagSet.singleton(container_type)),
            nullity=NullityValue.definitely_not_null(),
        )
        return state.set(var, val)

    def apply_copy(
        self,
        state: AbstractState[ProductValue],
        dst: str,
        src: str,
    ) -> AbstractState[ProductValue]:
        """Copy value from src to dst."""
        val = state.get(src)
        if val is None:
            return state.set(dst, self.domain.top())
        return state.set(dst, val)


# ===================================================================
# ProductInterpreter
# ===================================================================


class ProductInterpreter:
    """Abstract interpreter over the product domain.

    Processes a sequence of IR nodes and maintains a product state.
    """

    def __init__(self, domain: ReducedProductDomain):
        self.domain = domain
        self.transformer = ProductTransformer(domain)
        self.guard_xf = GuardTransferFunction(domain)
        self.call_xf = CallTransferFunction(domain)
        self.assign_xf = AssignTransferFunction(domain)
        self.deref_checker = NullDerefChecker()

    def interpret_node(
        self,
        node: IRNode,
        state: AbstractState[ProductValue],
    ) -> AbstractState[ProductValue]:
        """Interpret a single IR node."""
        if node.kind == "assign" and node.target is not None:
            return self.transformer.assign(state, node.target, node.expr)

        if node.kind == "guard":
            return self.transformer.guard(state, node.condition, True)

        if node.kind == "guard_false":
            return self.transformer.guard(state, node.condition, False)

        if node.kind == "call":
            return self.transformer.call(
                state, node.func or "", node.args, node.target
            )

        if node.kind == "noop":
            return state

        if node.kind == "assert":
            return self.transformer.guard(state, node.condition, True)

        return state

    def interpret_block(
        self,
        nodes: List[IRNode],
        initial_state: AbstractState[ProductValue],
    ) -> AbstractState[ProductValue]:
        """Interpret a basic block (sequence of IR nodes)."""
        state = initial_state
        for node in nodes:
            state = self.interpret_node(node, state)
            if state.is_bottom:
                break
        return state

    def check_null_safety(
        self,
        state: AbstractState[ProductValue],
        var: str,
        operation: str,
    ) -> Optional[Any]:
        """Check if accessing *var* is null-safe."""
        val = state.get(var)
        if val is None:
            return None

        if val.nullity.is_definitely_null:
            return {
                "severity": "error",
                "variable": var,
                "operation": operation,
                "message": f"Definite null dereference on '{var}'",
            }

        if val.nullity.is_maybe_null:
            return {
                "severity": "warning",
                "variable": var,
                "operation": operation,
                "message": f"Potential null dereference on '{var}'",
            }

        return None

    def create_initial_state(
        self,
        variables: Optional[Dict[str, Any]] = None,
    ) -> AbstractState[ProductValue]:
        """Create an initial product state."""
        env: Dict[str, ProductValue] = {}
        if variables:
            for var, concrete in variables.items():
                env[var] = self.domain.abstract(concrete)
        return AbstractState(env=env, domain=self.domain)

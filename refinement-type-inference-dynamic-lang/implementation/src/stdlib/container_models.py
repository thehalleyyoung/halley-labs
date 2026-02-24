from __future__ import annotations

"""
Refinement type models for container types in dynamically-typed languages.

Infers types like {x: int | x > 0 ∧ x < len(arr)} by tracking length bounds,
key presence, element types, and membership information across all container
operations. Each operation model specifies preconditions, postconditions,
transfer functions, and exception conditions.
"""

import enum
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)


# ---------------------------------------------------------------------------
# Local type definitions (no cross-module imports)
# ---------------------------------------------------------------------------

class RefinementSort(enum.Enum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BYTES = "bytes"
    BOOL = "bool"
    NONE = "none"
    LIST = "list"
    TUPLE = "tuple"
    DICT = "dict"
    SET = "set"
    ANY = "any"
    UNION = "union"
    OBJECT = "object"
    CALLABLE = "callable"
    ITERATOR = "iterator"
    GENERATOR = "generator"


@dataclass(frozen=True)
class Bound:
    """Represents a numeric bound (inclusive)."""
    value: Optional[int] = None
    symbolic: Optional[str] = None
    inclusive: bool = True

    def is_concrete(self) -> bool:
        return self.value is not None

    def __le__(self, other: Bound) -> bool:
        if self.value is not None and other.value is not None:
            return self.value <= other.value
        return False

    def __lt__(self, other: Bound) -> bool:
        if self.value is not None and other.value is not None:
            return self.value < other.value
        return False

    def __add__(self, other: Union[Bound, int]) -> Bound:
        if isinstance(other, int) and self.value is not None:
            return Bound(value=self.value + other)
        if isinstance(other, Bound) and self.value is not None and other.value is not None:
            return Bound(value=self.value + other.value)
        return Bound()

    def __sub__(self, other: Union[Bound, int]) -> Bound:
        if isinstance(other, int) and self.value is not None:
            return Bound(value=self.value - other)
        if isinstance(other, Bound) and self.value is not None and other.value is not None:
            return Bound(value=self.value - other.value)
        return Bound()


@dataclass(frozen=True)
class RefinementPredicate:
    """A single refinement predicate such as x > 0 or len(s) < 10."""
    variable: str
    operator: str  # '>', '>=', '<', '<=', '==', '!=', 'in', 'not_in', 'is', 'is_not'
    operand: Any
    symbolic_operand: Optional[str] = None

    def negate(self) -> RefinementPredicate:
        neg_map = {
            ">": "<=", ">=": "<", "<": ">=", "<=": ">",
            "==": "!=", "!=": "==", "in": "not_in", "not_in": "in",
            "is": "is_not", "is_not": "is",
        }
        return RefinementPredicate(
            variable=self.variable,
            operator=neg_map.get(self.operator, self.operator),
            operand=self.operand,
            symbolic_operand=self.symbolic_operand,
        )


@dataclass
class RefinementType:
    """A base type refined by a conjunction of predicates."""
    base: RefinementSort
    predicates: List[RefinementPredicate] = field(default_factory=list)
    nullable: bool = False

    def add_predicate(self, pred: RefinementPredicate) -> RefinementType:
        return RefinementType(
            base=self.base,
            predicates=self.predicates + [pred],
            nullable=self.nullable,
        )

    def meet(self, other: RefinementType) -> RefinementType:
        if self.base != other.base and self.base != RefinementSort.ANY:
            return RefinementType(base=RefinementSort.ANY)
        return RefinementType(
            base=self.base if self.base != RefinementSort.ANY else other.base,
            predicates=self.predicates + other.predicates,
            nullable=self.nullable and other.nullable,
        )


@dataclass
class OperationResult:
    """Result of modelling a single container operation."""
    preconditions: List[RefinementPredicate] = field(default_factory=list)
    postconditions: List[RefinementPredicate] = field(default_factory=list)
    return_type: Optional[RefinementType] = None
    exceptions: List[Tuple[str, List[RefinementPredicate]]] = field(default_factory=list)
    modifies_receiver: bool = False
    new_length_lower: Optional[Bound] = None
    new_length_upper: Optional[Bound] = None


@dataclass
class TransferFunction:
    """Describes how refinements propagate through an operation."""
    input_constraints: List[RefinementPredicate] = field(default_factory=list)
    output_constraints: List[RefinementPredicate] = field(default_factory=list)
    propagated: bool = True


# ---------------------------------------------------------------------------
# ContainerRefinement — abstract base for all container models
# ---------------------------------------------------------------------------

@dataclass
class ContainerRefinement:
    """
    Base refinement model for container types.

    Tracks the length interval [length_lower, length_upper], known element
    type, and known key membership.
    """
    length_lower: Bound = field(default_factory=lambda: Bound(value=0))
    length_upper: Bound = field(default_factory=Bound)
    element_type: RefinementType = field(
        default_factory=lambda: RefinementType(base=RefinementSort.ANY)
    )
    key_set: Set[str] = field(default_factory=set)
    _empty_known: Optional[bool] = None

    # -- derived properties --------------------------------------------------

    @property
    def is_empty(self) -> Optional[bool]:
        if self._empty_known is not None:
            return self._empty_known
        if (self.length_upper.is_concrete() and self.length_upper.value == 0):
            return True
        if (self.length_lower.is_concrete() and self.length_lower.value is not None
                and self.length_lower.value > 0):
            return False
        return None

    @property
    def is_non_empty(self) -> Optional[bool]:
        e = self.is_empty
        return None if e is None else not e

    @property
    def length_bounds(self) -> Tuple[Bound, Bound]:
        return (self.length_lower, self.length_upper)

    # -- helpers --------------------------------------------------------------

    def _clamp_lower(self, b: Bound) -> Bound:
        if b.is_concrete() and b.value is not None and b.value < 0:
            return Bound(value=0)
        return b

    def _inc_length(self, n: int = 1) -> Tuple[Bound, Bound]:
        return (self.length_lower + n, self.length_upper + n)

    def _dec_length(self, n: int = 1) -> Tuple[Bound, Bound]:
        new_lo = self._clamp_lower(self.length_lower - n)
        new_hi = self._clamp_lower(self.length_upper - n)
        return (new_lo, new_hi)

    def copy_with(self, **kwargs: Any) -> ContainerRefinement:
        import copy as _copy
        obj = _copy.copy(self)
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj


# ====================================================================
# ListRefinementModel
# ====================================================================

@dataclass
class ListRefinementModel(ContainerRefinement):
    """
    Refinement model for Python ``list``.

    Tracks length bounds across all mutating and non-mutating operations.
    """

    # -- __getitem__ ---------------------------------------------------------

    def model_getitem(self, index_type: RefinementType) -> OperationResult:
        """
        list.__getitem__(index)

        Precondition: {i: int | 0 <= i < len(self)} for positive index, or
                      {i: int | -len(self) <= i < 0} for negative.
        """
        preconds = [
            RefinementPredicate("index", ">=", 0, symbolic_operand="-len(self)"),
            RefinementPredicate("index", "<", None, symbolic_operand="len(self)"),
        ]
        return OperationResult(
            preconditions=preconds,
            return_type=self.element_type,
            exceptions=[
                ("IndexError", [
                    RefinementPredicate("index", ">=", None, symbolic_operand="len(self)"),
                ]),
                ("IndexError", [
                    RefinementPredicate("index", "<", None, symbolic_operand="-len(self)"),
                ]),
            ],
        )

    # -- __setitem__ ---------------------------------------------------------

    def model_setitem(self, index_type: RefinementType, value_type: RefinementType) -> OperationResult:
        """
        list.__setitem__(index, value)

        Precondition: index in bounds.  Length unchanged.
        """
        preconds = [
            RefinementPredicate("index", ">=", 0, symbolic_operand="-len(self)"),
            RefinementPredicate("index", "<", None, symbolic_operand="len(self)"),
        ]
        post = [
            RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre)"),
        ]
        return OperationResult(
            preconditions=preconds,
            postconditions=post,
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            exceptions=[
                ("IndexError", [
                    RefinementPredicate("index", ">=", None, symbolic_operand="len(self)"),
                ]),
            ],
        )

    # -- __delitem__ ---------------------------------------------------------

    def model_delitem(self, index_type: RefinementType) -> OperationResult:
        """
        list.__delitem__(index)

        Postcondition: len(self) == len(self@pre) - 1.
        """
        preconds = [
            RefinementPredicate("index", ">=", 0, symbolic_operand="-len(self)"),
            RefinementPredicate("index", "<", None, symbolic_operand="len(self)"),
        ]
        lo, hi = self._dec_length(1)
        return OperationResult(
            preconditions=preconds,
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre) - 1"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
            exceptions=[
                ("IndexError", [RefinementPredicate("index", ">=", None, symbolic_operand="len(self)")]),
            ],
        )

    # -- __contains__ --------------------------------------------------------

    def model_contains(self, value_type: RefinementType) -> OperationResult:
        """
        list.__contains__(value)

        Pure operation returning bool.
        """
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BOOL),
        )

    # -- __len__ -------------------------------------------------------------

    def model_len(self) -> OperationResult:
        """
        list.__len__()

        Returns {n: int | n >= 0} with bounds from tracking.
        """
        preds: List[RefinementPredicate] = [RefinementPredicate("result", ">=", 0)]
        if self.length_lower.is_concrete():
            preds.append(RefinementPredicate("result", ">=", self.length_lower.value))
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("result", "<=", self.length_upper.value))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.INT, predicates=preds),
        )

    # -- __iter__ ------------------------------------------------------------

    def model_iter(self) -> OperationResult:
        """
        list.__iter__()

        Returns an iterator whose element type matches self.element_type.
        """
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.ITERATOR),
        )

    # -- __reversed__ --------------------------------------------------------

    def model_reversed(self) -> OperationResult:
        """
        list.__reversed__()

        Returns a reverse iterator; length invariant preserved.
        """
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.ITERATOR),
        )

    # -- append --------------------------------------------------------------

    def model_append(self, value_type: RefinementType) -> OperationResult:
        """
        list.append(value)

        Postcondition: len(self) == len(self@pre) + 1.
        """
        lo, hi = self._inc_length(1)
        return OperationResult(
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre) + 1"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
        )

    # -- extend --------------------------------------------------------------

    def model_extend(self, other_model: Optional[ContainerRefinement] = None) -> OperationResult:
        """
        list.extend(iterable)

        Postcondition: len(self) == len(self@pre) + len(other).
        """
        if other_model is not None:
            lo = self.length_lower + (other_model.length_lower.value or 0)
            hi_val = (
                self.length_upper + other_model.length_upper
                if self.length_upper.is_concrete() and other_model.length_upper.is_concrete()
                else Bound()
            )
        else:
            lo = self.length_lower
            hi_val = Bound()
        return OperationResult(
            postconditions=[
                RefinementPredicate("len(self)", ">=", None, symbolic_operand="len(self@pre)"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi_val,
        )

    # -- insert --------------------------------------------------------------

    def model_insert(self, index_type: RefinementType, value_type: RefinementType) -> OperationResult:
        """
        list.insert(index, value)

        No IndexError — index is clamped.  len increases by 1.
        """
        lo, hi = self._inc_length(1)
        return OperationResult(
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre) + 1"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
        )

    # -- remove --------------------------------------------------------------

    def model_remove(self, value_type: RefinementType) -> OperationResult:
        """
        list.remove(value)

        Precondition: value in self.
        Postcondition: len(self) == len(self@pre) - 1.
        """
        lo, hi = self._dec_length(1)
        return OperationResult(
            preconditions=[
                RefinementPredicate("value", "in", None, symbolic_operand="self"),
            ],
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre) - 1"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
            exceptions=[
                ("ValueError", [RefinementPredicate("value", "not_in", None, symbolic_operand="self")]),
            ],
        )

    # -- pop -----------------------------------------------------------------

    def model_pop(self, index_type: Optional[RefinementType] = None) -> OperationResult:
        """
        list.pop([index])

        Precondition: len(self) > 0 (and index in bounds if given).
        Postcondition: len(self) == len(self@pre) - 1.
        """
        preconds: List[RefinementPredicate] = [
            RefinementPredicate("len(self)", ">", 0),
        ]
        if index_type is not None:
            preconds.append(
                RefinementPredicate("index", ">=", 0, symbolic_operand="-len(self)")
            )
            preconds.append(
                RefinementPredicate("index", "<", None, symbolic_operand="len(self)")
            )
        lo, hi = self._dec_length(1)
        return OperationResult(
            preconditions=preconds,
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre) - 1"),
            ],
            return_type=self.element_type,
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
            exceptions=[
                ("IndexError", [RefinementPredicate("len(self)", "==", 0)]),
            ],
        )

    # -- clear ---------------------------------------------------------------

    def model_clear(self) -> OperationResult:
        """
        list.clear()

        Postcondition: len(self) == 0.
        """
        return OperationResult(
            postconditions=[
                RefinementPredicate("len(self)", "==", 0),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=Bound(value=0),
            new_length_upper=Bound(value=0),
        )

    # -- copy ----------------------------------------------------------------

    def model_copy(self) -> OperationResult:
        """
        list.copy()

        Returns a shallow copy with identical length refinement.
        """
        preds: List[RefinementPredicate] = []
        if self.length_lower.is_concrete():
            preds.append(RefinementPredicate("len(result)", ">=", self.length_lower.value))
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("len(result)", "<=", self.length_upper.value))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.LIST, predicates=preds),
        )

    # -- count ---------------------------------------------------------------

    def model_count(self, value_type: RefinementType) -> OperationResult:
        """
        list.count(value)

        Returns {n: int | 0 <= n <= len(self)}.
        """
        preds = [
            RefinementPredicate("result", ">=", 0),
        ]
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("result", "<=", self.length_upper.value))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.INT, predicates=preds),
        )

    # -- index ---------------------------------------------------------------

    def model_index(
        self,
        value_type: RefinementType,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> OperationResult:
        """
        list.index(value[, start[, stop]])

        Precondition: value in self[start:stop].
        Returns {i: int | start <= i < stop ∧ i < len(self)}.
        """
        lo = start if start is not None else 0
        preconds = [
            RefinementPredicate("value", "in", None, symbolic_operand="self"),
        ]
        preds = [
            RefinementPredicate("result", ">=", lo),
        ]
        if stop is not None:
            preds.append(RefinementPredicate("result", "<", stop))
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.INT, predicates=preds),
            exceptions=[
                ("ValueError", [RefinementPredicate("value", "not_in", None, symbolic_operand="self")]),
            ],
        )

    # -- sort ----------------------------------------------------------------

    def model_sort(self) -> OperationResult:
        """
        list.sort()

        Length invariant; in-place.
        """
        return OperationResult(
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre)"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    # -- reverse -------------------------------------------------------------

    def model_reverse(self) -> OperationResult:
        """
        list.reverse()

        Length invariant; in-place.
        """
        return OperationResult(
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre)"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    # -- __add__ -------------------------------------------------------------

    def model_add(self, other_model: Optional[ContainerRefinement] = None) -> OperationResult:
        """
        list.__add__(other)

        Returns a new list with len == len(self) + len(other).
        """
        preds: List[RefinementPredicate] = []
        if other_model is not None and self.length_lower.is_concrete() and other_model.length_lower.is_concrete():
            combined_lo = (self.length_lower.value or 0) + (other_model.length_lower.value or 0)
            preds.append(RefinementPredicate("len(result)", ">=", combined_lo))
        if other_model is not None and self.length_upper.is_concrete() and other_model.length_upper.is_concrete():
            combined_hi = (self.length_upper.value or 0) + (other_model.length_upper.value or 0)
            preds.append(RefinementPredicate("len(result)", "<=", combined_hi))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.LIST, predicates=preds),
        )

    # -- __mul__ -------------------------------------------------------------

    def model_mul(self, n_type: RefinementType) -> OperationResult:
        """
        list.__mul__(n)

        Returns list with len == len(self) * n.
        """
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.LIST,
                predicates=[
                    RefinementPredicate("len(result)", ">=", 0),
                ],
            ),
        )

    # -- __iadd__ ------------------------------------------------------------

    def model_iadd(self, other_model: Optional[ContainerRefinement] = None) -> OperationResult:
        """
        list.__iadd__(other)

        In-place concatenation.
        """
        result = self.model_extend(other_model)
        result.return_type = RefinementType(base=RefinementSort.LIST)
        return result

    # -- __imul__ ------------------------------------------------------------

    def model_imul(self, n_type: RefinementType) -> OperationResult:
        """
        list.__imul__(n)

        In-place repetition.
        """
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.LIST),
            modifies_receiver=True,
        )

    # -- __eq__ --------------------------------------------------------------

    def model_eq(self, other_model: Optional[ContainerRefinement] = None) -> OperationResult:
        """
        list.__eq__(other)

        Returns bool.  If lengths provably differ, result is False.
        """
        preds: List[RefinementPredicate] = []
        if (other_model is not None
                and self.length_lower.is_concrete() and other_model.length_upper.is_concrete()
                and self.length_lower.value is not None and other_model.length_upper.value is not None
                and self.length_lower.value > other_model.length_upper.value):
            preds.append(RefinementPredicate("result", "==", False))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BOOL, predicates=preds),
        )

    # -- slice ---------------------------------------------------------------

    def model_slice(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> OperationResult:
        """
        list[start:stop:step]

        Produces a subrange.  Length is bounded by input bounds and slice params.
        """
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        if self.length_upper.is_concrete() and self.length_upper.value is not None:
            preds.append(RefinementPredicate("len(result)", "<=", self.length_upper.value))
        if start is not None and stop is not None and step is None:
            sl = max(stop - start, 0)
            preds.append(RefinementPredicate("len(result)", "<=", sl))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.LIST, predicates=preds),
        )


# ====================================================================
# TupleRefinementModel
# ====================================================================

@dataclass
class TupleRefinementModel(ContainerRefinement):
    """
    Refinement model for Python ``tuple``.

    Fixed length, per-position element types, immutability constraints.
    """
    element_types: List[RefinementType] = field(default_factory=list)
    _frozen: bool = True

    def __post_init__(self) -> None:
        if self.element_types:
            n = len(self.element_types)
            self.length_lower = Bound(value=n)
            self.length_upper = Bound(value=n)

    def model_getitem(self, index_type: RefinementType, index_value: Optional[int] = None) -> OperationResult:
        preconds = [
            RefinementPredicate("index", ">=", 0, symbolic_operand="-len(self)"),
            RefinementPredicate("index", "<", None, symbolic_operand="len(self)"),
        ]
        ret = self.element_type
        if index_value is not None and 0 <= index_value < len(self.element_types):
            ret = self.element_types[index_value]
        return OperationResult(
            preconditions=preconds,
            return_type=ret,
            exceptions=[
                ("IndexError", [RefinementPredicate("index", ">=", None, symbolic_operand="len(self)")]),
            ],
        )

    def model_contains(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_len(self) -> OperationResult:
        preds = []
        if self.length_lower.is_concrete():
            preds.append(RefinementPredicate("result", "==", self.length_lower.value))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.INT, predicates=preds),
        )

    def model_iter(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_add(self, other: Optional[TupleRefinementModel] = None) -> OperationResult:
        new_types = list(self.element_types)
        if other is not None:
            new_types.extend(other.element_types)
        preds = [RefinementPredicate("len(result)", "==", len(new_types))]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.TUPLE, predicates=preds),
        )

    def model_mul(self, n_type: RefinementType, n_value: Optional[int] = None) -> OperationResult:
        preds: List[RefinementPredicate] = [RefinementPredicate("len(result)", ">=", 0)]
        if n_value is not None and self.length_lower.is_concrete():
            total = (self.length_lower.value or 0) * n_value
            preds = [RefinementPredicate("len(result)", "==", total)]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.TUPLE, predicates=preds),
        )

    def model_count(self, value_type: RefinementType) -> OperationResult:
        preds = [
            RefinementPredicate("result", ">=", 0),
        ]
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("result", "<=", self.length_upper.value))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.INT, predicates=preds),
        )

    def model_index(self, value_type: RefinementType) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("result", "<", self.length_upper.value))
        return OperationResult(
            preconditions=[RefinementPredicate("value", "in", None, symbolic_operand="self")],
            return_type=RefinementType(base=RefinementSort.INT, predicates=preds),
            exceptions=[
                ("ValueError", [RefinementPredicate("value", "not_in", None, symbolic_operand="self")]),
            ],
        )

    def model_slice(self, start: Optional[int] = None, stop: Optional[int] = None) -> OperationResult:
        s = start if start is not None else 0
        e = stop if stop is not None else len(self.element_types)
        sliced = self.element_types[s:e]
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.TUPLE,
                predicates=[RefinementPredicate("len(result)", "==", len(sliced))],
            ),
        )

    def model_eq(self, other: Optional[TupleRefinementModel] = None) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if (other is not None and self.length_lower.is_concrete()
                and other.length_lower.is_concrete()
                and self.length_lower.value != other.length_lower.value):
            preds.append(RefinementPredicate("result", "==", False))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BOOL, predicates=preds),
        )


# ====================================================================
# DictRefinementModel
# ====================================================================

@dataclass
class DictRefinementModel(ContainerRefinement):
    """
    Refinement model for Python ``dict``.

    Tracks key presence: {d: dict | 'name' in d ∧ 'age' in d}.
    """
    known_keys: Set[str] = field(default_factory=set)
    value_types: Dict[str, RefinementType] = field(default_factory=dict)
    key_type: RefinementType = field(default_factory=lambda: RefinementType(base=RefinementSort.ANY))

    def __post_init__(self) -> None:
        if self.known_keys:
            self.key_set = set(self.known_keys)

    # -- __getitem__ ---------------------------------------------------------

    def model_getitem(self, key_type: RefinementType, key_value: Optional[str] = None) -> OperationResult:
        preconds = [RefinementPredicate("key", "in", None, symbolic_operand="self")]
        ret = self.element_type
        if key_value is not None and key_value in self.value_types:
            ret = self.value_types[key_value]
        return OperationResult(
            preconditions=preconds,
            return_type=ret,
            exceptions=[
                ("KeyError", [RefinementPredicate("key", "not_in", None, symbolic_operand="self")]),
            ],
        )

    # -- get -----------------------------------------------------------------

    def model_get(
        self,
        key_type: RefinementType,
        default_type: Optional[RefinementType] = None,
        key_value: Optional[str] = None,
    ) -> OperationResult:
        """dict.get(key[, default]) — never raises KeyError."""
        if key_value is not None and key_value in self.known_keys:
            ret = self.value_types.get(key_value, self.element_type)
        elif default_type is not None:
            ret = RefinementType(base=RefinementSort.UNION, nullable=True)
        else:
            ret = RefinementType(base=RefinementSort.ANY, nullable=True)
        return OperationResult(return_type=ret)

    # -- __setitem__ ---------------------------------------------------------

    def model_setitem(self, key_type: RefinementType, value_type: RefinementType, key_value: Optional[str] = None) -> OperationResult:
        posts: List[RefinementPredicate] = [
            RefinementPredicate("key", "in", None, symbolic_operand="self"),
        ]
        new_keys = set(self.known_keys)
        if key_value is not None:
            new_keys.add(key_value)
            posts.append(RefinementPredicate(f"'{key_value}'", "in", None, symbolic_operand="self"))
        return OperationResult(
            postconditions=posts,
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    # -- __delitem__ ---------------------------------------------------------

    def model_delitem(self, key_type: RefinementType, key_value: Optional[str] = None) -> OperationResult:
        preconds = [RefinementPredicate("key", "in", None, symbolic_operand="self")]
        lo, hi = self._dec_length(1)
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
            exceptions=[
                ("KeyError", [RefinementPredicate("key", "not_in", None, symbolic_operand="self")]),
            ],
        )

    # -- __contains__ --------------------------------------------------------

    def model_contains(self, key_type: RefinementType, key_value: Optional[str] = None) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if key_value is not None and key_value in self.known_keys:
            preds.append(RefinementPredicate("result", "==", True))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BOOL, predicates=preds),
        )

    # -- __len__ -------------------------------------------------------------

    def model_len(self) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        if self.known_keys:
            preds.append(RefinementPredicate("result", ">=", len(self.known_keys)))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.INT, predicates=preds),
        )

    # -- setdefault ----------------------------------------------------------

    def model_setdefault(
        self,
        key_type: RefinementType,
        default_type: Optional[RefinementType] = None,
        key_value: Optional[str] = None,
    ) -> OperationResult:
        posts: List[RefinementPredicate] = []
        if key_value is not None:
            posts.append(RefinementPredicate(f"'{key_value}'", "in", None, symbolic_operand="self"))
        ret = self.element_type
        if key_value and key_value in self.value_types:
            ret = self.value_types[key_value]
        return OperationResult(
            postconditions=posts,
            return_type=ret,
            modifies_receiver=True,
        )

    # -- update --------------------------------------------------------------

    def model_update(self, other: Optional[DictRefinementModel] = None) -> OperationResult:
        posts: List[RefinementPredicate] = []
        if other is not None:
            for k in other.known_keys:
                posts.append(RefinementPredicate(f"'{k}'", "in", None, symbolic_operand="self"))
        return OperationResult(
            postconditions=posts,
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    # -- pop -----------------------------------------------------------------

    def model_pop(
        self,
        key_type: RefinementType,
        default_type: Optional[RefinementType] = None,
        key_value: Optional[str] = None,
    ) -> OperationResult:
        if default_type is None:
            preconds = [RefinementPredicate("key", "in", None, symbolic_operand="self")]
            excs: List[Tuple[str, List[RefinementPredicate]]] = [
                ("KeyError", [RefinementPredicate("key", "not_in", None, symbolic_operand="self")]),
            ]
        else:
            preconds = []
            excs = []
        lo, hi = self._dec_length(1)
        return OperationResult(
            preconditions=preconds,
            postconditions=[
                RefinementPredicate("key", "not_in", None, symbolic_operand="self"),
            ],
            return_type=self.element_type,
            modifies_receiver=True,
            new_length_lower=self._clamp_lower(lo),
            new_length_upper=hi,
            exceptions=excs,
        )

    # -- popitem -------------------------------------------------------------

    def model_popitem(self) -> OperationResult:
        preconds = [RefinementPredicate("len(self)", ">", 0)]
        lo, hi = self._dec_length(1)
        return OperationResult(
            preconditions=preconds,
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre) - 1"),
            ],
            return_type=RefinementType(base=RefinementSort.TUPLE),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
            exceptions=[
                ("KeyError", [RefinementPredicate("len(self)", "==", 0)]),
            ],
        )

    # -- keys / values / items -----------------------------------------------

    def model_keys(self) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if self.length_lower.is_concrete():
            preds.append(RefinementPredicate("len(result)", ">=", self.length_lower.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR, predicates=preds))

    def model_values(self) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if self.length_lower.is_concrete():
            preds.append(RefinementPredicate("len(result)", ">=", self.length_lower.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR, predicates=preds))

    def model_items(self) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if self.length_lower.is_concrete():
            preds.append(RefinementPredicate("len(result)", ">=", self.length_lower.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR, predicates=preds))

    # -- clear ---------------------------------------------------------------

    def model_clear(self) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("len(self)", "==", 0)],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=Bound(value=0),
            new_length_upper=Bound(value=0),
        )

    # -- copy ----------------------------------------------------------------

    def model_copy(self) -> OperationResult:
        preds: List[RefinementPredicate] = []
        for k in self.known_keys:
            preds.append(RefinementPredicate(f"'{k}'", "in", None, symbolic_operand="result"))
        return OperationResult(return_type=RefinementType(base=RefinementSort.DICT, predicates=preds))

    # -- fromkeys ------------------------------------------------------------

    def model_fromkeys(self, keys_type: RefinementType, value_type: Optional[RefinementType] = None) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.DICT),
        )


# ====================================================================
# SetRefinementModel
# ====================================================================

@dataclass
class SetRefinementModel(ContainerRefinement):
    """
    Refinement model for Python ``set``.

    Tracks membership and length bounds.
    """
    known_members: FrozenSet[Any] = field(default_factory=frozenset)

    # -- add -----------------------------------------------------------------

    def model_add(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(
            postconditions=[
                RefinementPredicate("value", "in", None, symbolic_operand="self"),
                RefinementPredicate("len(self)", ">=", None, symbolic_operand="len(self@pre)"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    # -- remove --------------------------------------------------------------

    def model_remove(self, value_type: RefinementType) -> OperationResult:
        preconds = [RefinementPredicate("value", "in", None, symbolic_operand="self")]
        lo, hi = self._dec_length(1)
        return OperationResult(
            preconditions=preconds,
            postconditions=[
                RefinementPredicate("value", "not_in", None, symbolic_operand="self"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
            exceptions=[
                ("KeyError", [RefinementPredicate("value", "not_in", None, symbolic_operand="self")]),
            ],
        )

    # -- discard -------------------------------------------------------------

    def model_discard(self, value_type: RefinementType) -> OperationResult:
        lo = self._clamp_lower(self.length_lower - 1)
        return OperationResult(
            postconditions=[
                RefinementPredicate("value", "not_in", None, symbolic_operand="self"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=lo,
        )

    # -- pop -----------------------------------------------------------------

    def model_pop(self) -> OperationResult:
        preconds = [RefinementPredicate("len(self)", ">", 0)]
        lo, hi = self._dec_length(1)
        return OperationResult(
            preconditions=preconds,
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre) - 1"),
            ],
            return_type=self.element_type,
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
            exceptions=[
                ("KeyError", [RefinementPredicate("len(self)", "==", 0)]),
            ],
        )

    # -- clear ---------------------------------------------------------------

    def model_clear(self) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("len(self)", "==", 0)],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=Bound(value=0),
            new_length_upper=Bound(value=0),
        )

    # -- union ---------------------------------------------------------------

    def model_union(self, other: Optional[SetRefinementModel] = None) -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">=", None, symbolic_operand="max(len(self), len(other))")]
        return OperationResult(return_type=RefinementType(base=RefinementSort.SET, predicates=preds))

    # -- intersection --------------------------------------------------------

    def model_intersection(self, other: Optional[SetRefinementModel] = None) -> OperationResult:
        preds = [RefinementPredicate("len(result)", "<=", None, symbolic_operand="min(len(self), len(other))")]
        return OperationResult(return_type=RefinementType(base=RefinementSort.SET, predicates=preds))

    # -- difference ----------------------------------------------------------

    def model_difference(self, other: Optional[SetRefinementModel] = None) -> OperationResult:
        preds = [
            RefinementPredicate("len(result)", ">=", 0),
            RefinementPredicate("len(result)", "<=", None, symbolic_operand="len(self)"),
        ]
        return OperationResult(return_type=RefinementType(base=RefinementSort.SET, predicates=preds))

    # -- symmetric_difference ------------------------------------------------

    def model_symmetric_difference(self, other: Optional[SetRefinementModel] = None) -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        return OperationResult(return_type=RefinementType(base=RefinementSort.SET, predicates=preds))

    # -- issubset / issuperset / isdisjoint ----------------------------------

    def model_issubset(self, other: Optional[SetRefinementModel] = None) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_issuperset(self, other: Optional[SetRefinementModel] = None) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_isdisjoint(self, other: Optional[SetRefinementModel] = None) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    # -- __contains__ --------------------------------------------------------

    def model_contains(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    # -- __len__ -------------------------------------------------------------

    def model_len(self) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("result", "<=", self.length_upper.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    # -- copy ----------------------------------------------------------------

    def model_copy(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.SET))

    # -- update --------------------------------------------------------------

    def model_update(self, other: Optional[SetRefinementModel] = None) -> OperationResult:
        return OperationResult(
            postconditions=[
                RefinementPredicate("len(self)", ">=", None, symbolic_operand="len(self@pre)"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    # -- intersection_update -------------------------------------------------

    def model_intersection_update(self, other: Optional[SetRefinementModel] = None) -> OperationResult:
        return OperationResult(
            postconditions=[
                RefinementPredicate("len(self)", "<=", None, symbolic_operand="len(self@pre)"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    # -- difference_update ---------------------------------------------------

    def model_difference_update(self, other: Optional[SetRefinementModel] = None) -> OperationResult:
        return OperationResult(
            postconditions=[
                RefinementPredicate("len(self)", "<=", None, symbolic_operand="len(self@pre)"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    # -- symmetric_difference_update -----------------------------------------

    def model_symmetric_difference_update(self, other: Optional[SetRefinementModel] = None) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("len(self)", ">=", 0)],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )


# ====================================================================
# DequeRefinementModel
# ====================================================================

@dataclass
class DequeRefinementModel(ContainerRefinement):
    """
    Refinement model for ``collections.deque``.

    Tracks maxlen constraint: {d: deque | len(d) <= maxlen}.
    """
    maxlen: Optional[int] = None

    def __post_init__(self) -> None:
        if self.maxlen is not None:
            self.length_upper = Bound(value=self.maxlen)

    def _capped(self, lo: Bound, hi: Bound) -> Tuple[Bound, Bound]:
        if self.maxlen is not None and hi.is_concrete() and hi.value is not None and hi.value > self.maxlen:
            hi = Bound(value=self.maxlen)
        return (lo, hi)

    # -- append / appendleft -------------------------------------------------

    def model_append(self, value_type: RefinementType) -> OperationResult:
        lo, hi = self._capped(*self._inc_length(1))
        post = [RefinementPredicate("len(self)", ">=", 1)]
        if self.maxlen is not None:
            post.append(RefinementPredicate("len(self)", "<=", self.maxlen))
        return OperationResult(
            postconditions=post,
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
        )

    def model_appendleft(self, value_type: RefinementType) -> OperationResult:
        return self.model_append(value_type)

    # -- pop / popleft -------------------------------------------------------

    def model_pop(self) -> OperationResult:
        preconds = [RefinementPredicate("len(self)", ">", 0)]
        lo, hi = self._dec_length(1)
        return OperationResult(
            preconditions=preconds,
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre) - 1"),
            ],
            return_type=self.element_type,
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
            exceptions=[("IndexError", [RefinementPredicate("len(self)", "==", 0)])],
        )

    def model_popleft(self) -> OperationResult:
        return self.model_pop()

    # -- extend / extendleft -------------------------------------------------

    def model_extend(self, other: Optional[ContainerRefinement] = None) -> OperationResult:
        post = [RefinementPredicate("len(self)", ">=", None, symbolic_operand="len(self@pre)")]
        if self.maxlen is not None:
            post.append(RefinementPredicate("len(self)", "<=", self.maxlen))
        return OperationResult(
            postconditions=post,
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    def model_extendleft(self, other: Optional[ContainerRefinement] = None) -> OperationResult:
        return self.model_extend(other)

    # -- rotate --------------------------------------------------------------

    def model_rotate(self, n: Optional[int] = None) -> OperationResult:
        return OperationResult(
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre)"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    # -- clear ---------------------------------------------------------------

    def model_clear(self) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("len(self)", "==", 0)],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=Bound(value=0),
            new_length_upper=Bound(value=0),
        )

    # -- copy ----------------------------------------------------------------

    def model_copy(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.LIST))

    # -- count / index -------------------------------------------------------

    def model_count(self, value_type: RefinementType) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("result", "<=", self.length_upper.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    def model_index(self, value_type: RefinementType) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        return OperationResult(
            preconditions=[RefinementPredicate("value", "in", None, symbolic_operand="self")],
            return_type=RefinementType(base=RefinementSort.INT, predicates=preds),
            exceptions=[("ValueError", [RefinementPredicate("value", "not_in", None, symbolic_operand="self")])],
        )

    # -- insert --------------------------------------------------------------

    def model_insert(self, index_type: RefinementType, value_type: RefinementType) -> OperationResult:
        preconds: List[RefinementPredicate] = []
        if self.maxlen is not None:
            preconds.append(RefinementPredicate("len(self)", "<", self.maxlen))
        lo, hi = self._capped(*self._inc_length(1))
        return OperationResult(
            preconditions=preconds,
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre) + 1"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
            exceptions=[
                ("IndexError", preconds) if self.maxlen is not None else ("IndexError", []),
            ],
        )

    # -- remove --------------------------------------------------------------

    def model_remove(self, value_type: RefinementType) -> OperationResult:
        preconds = [RefinementPredicate("value", "in", None, symbolic_operand="self")]
        lo, hi = self._dec_length(1)
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
            exceptions=[("ValueError", [RefinementPredicate("value", "not_in", None, symbolic_operand="self")])],
        )

    # -- reverse -------------------------------------------------------------

    def model_reverse(self) -> OperationResult:
        return OperationResult(
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre)"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    # -- maxlen property -----------------------------------------------------

    def model_maxlen_property(self) -> OperationResult:
        if self.maxlen is not None:
            preds = [RefinementPredicate("result", "==", self.maxlen)]
        else:
            preds = [RefinementPredicate("result", "is", None)]
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds, nullable=self.maxlen is None))


# ====================================================================
# ArrayRefinementModel (TypeScript TypedArrays)
# ====================================================================

class TypedArrayKind(enum.Enum):
    INT8 = "Int8Array"
    UINT8 = "Uint8Array"
    INT16 = "Int16Array"
    UINT16 = "Uint16Array"
    INT32 = "Int32Array"
    UINT32 = "Uint32Array"
    FLOAT32 = "Float32Array"
    FLOAT64 = "Float64Array"
    BIGINT64 = "BigInt64Array"
    BIGUINT64 = "BigUint64Array"


_TYPED_ARRAY_RANGES: Dict[TypedArrayKind, Tuple[int, int]] = {
    TypedArrayKind.INT8: (-128, 127),
    TypedArrayKind.UINT8: (0, 255),
    TypedArrayKind.INT16: (-32768, 32767),
    TypedArrayKind.UINT16: (0, 65535),
    TypedArrayKind.INT32: (-2147483648, 2147483647),
    TypedArrayKind.UINT32: (0, 4294967295),
    TypedArrayKind.FLOAT32: (-(2**128), 2**128),
    TypedArrayKind.FLOAT64: (-(2**1024), 2**1024),
    TypedArrayKind.BIGINT64: (-(2**63), 2**63 - 1),
    TypedArrayKind.BIGUINT64: (0, 2**64 - 1),
}


@dataclass
class ArrayRefinementModel(ContainerRefinement):
    """
    Refinement model for TypeScript arrays and TypedArray variants.

    Element range refinement: {a[i]: int | lo <= a[i] <= hi} for typed arrays.
    """
    kind: Optional[TypedArrayKind] = None

    @property
    def element_range(self) -> Optional[Tuple[int, int]]:
        if self.kind is not None:
            return _TYPED_ARRAY_RANGES.get(self.kind)
        return None

    def _element_preds(self) -> List[RefinementPredicate]:
        rng = self.element_range
        if rng is None:
            return []
        return [
            RefinementPredicate("element", ">=", rng[0]),
            RefinementPredicate("element", "<=", rng[1]),
        ]

    # -- access --------------------------------------------------------------

    def model_getitem(self, index_type: RefinementType) -> OperationResult:
        preconds = [
            RefinementPredicate("index", ">=", 0),
            RefinementPredicate("index", "<", None, symbolic_operand="self.length"),
        ]
        preds = self._element_preds()
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.INT if self.kind else RefinementSort.ANY, predicates=preds),
            exceptions=[("RangeError", [RefinementPredicate("index", ">=", None, symbolic_operand="self.length")])],
        )

    def model_setitem(self, index_type: RefinementType, value_type: RefinementType) -> OperationResult:
        preconds = [
            RefinementPredicate("index", ">=", 0),
            RefinementPredicate("index", "<", None, symbolic_operand="self.length"),
        ]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    # -- push / pop (JS Array, not TypedArray) -------------------------------

    def model_push(self, value_type: RefinementType) -> OperationResult:
        lo, hi = self._inc_length(1)
        return OperationResult(
            postconditions=[
                RefinementPredicate("self.length", "==", None, symbolic_operand="self.length@pre + 1"),
            ],
            return_type=RefinementType(base=RefinementSort.INT),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
        )

    def model_pop_js(self) -> OperationResult:
        preconds = [RefinementPredicate("self.length", ">", 0)]
        lo, hi = self._dec_length(1)
        return OperationResult(
            preconditions=preconds,
            return_type=self.element_type,
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
        )

    # -- shift / unshift -----------------------------------------------------

    def model_shift(self) -> OperationResult:
        preconds = [RefinementPredicate("self.length", ">", 0)]
        lo, hi = self._dec_length(1)
        return OperationResult(
            preconditions=preconds,
            return_type=self.element_type,
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
        )

    def model_unshift(self, value_type: RefinementType) -> OperationResult:
        lo, hi = self._inc_length(1)
        return OperationResult(
            postconditions=[
                RefinementPredicate("self.length", "==", None, symbolic_operand="self.length@pre + 1"),
            ],
            return_type=RefinementType(base=RefinementSort.INT),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
        )

    # -- splice --------------------------------------------------------------

    def model_splice(self, start: int, delete_count: Optional[int] = None, *items: RefinementType) -> OperationResult:
        added = len(items)
        removed = delete_count if delete_count is not None else 0
        delta = added - removed
        lo = self._clamp_lower(self.length_lower + delta) if self.length_lower.is_concrete() else Bound()
        hi = (self.length_upper + delta) if self.length_upper.is_concrete() else Bound()
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.LIST),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
        )

    # -- slice (non-mutating) ------------------------------------------------

    def model_slice_js(self, start: Optional[int] = None, end: Optional[int] = None) -> OperationResult:
        preds = [RefinementPredicate("result.length", ">=", 0)]
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("result.length", "<=", self.length_upper.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.LIST, predicates=preds))

    # -- concat --------------------------------------------------------------

    def model_concat(self, other: Optional[ArrayRefinementModel] = None) -> OperationResult:
        preds = [RefinementPredicate("result.length", ">=", None, symbolic_operand="self.length")]
        return OperationResult(return_type=RefinementType(base=RefinementSort.LIST, predicates=preds))

    # -- map / filter / reduce -----------------------------------------------

    def model_map(self, callback_return: RefinementType) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if self.length_lower.is_concrete():
            preds.append(RefinementPredicate("result.length", "==", self.length_lower.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.LIST, predicates=preds))

    def model_filter(self, callback_return: RefinementType) -> OperationResult:
        preds = [
            RefinementPredicate("result.length", ">=", 0),
            RefinementPredicate("result.length", "<=", None, symbolic_operand="self.length"),
        ]
        return OperationResult(return_type=RefinementType(base=RefinementSort.LIST, predicates=preds))

    def model_reduce(self, callback_return: RefinementType) -> OperationResult:
        preconds = [RefinementPredicate("self.length", ">", 0)]
        return OperationResult(
            preconditions=preconds,
            return_type=callback_return,
            exceptions=[("TypeError", [RefinementPredicate("self.length", "==", 0)])],
        )

    # -- includes / indexOf --------------------------------------------------

    def model_includes(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_indexOf(self, value_type: RefinementType) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", -1)]
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    # -- length property -----------------------------------------------------

    def model_length(self) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        if self.length_lower.is_concrete():
            preds.append(RefinementPredicate("result", ">=", self.length_lower.value))
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("result", "<=", self.length_upper.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    # -- fill / copyWithin / reverse / sort ----------------------------------

    def model_fill(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(
            postconditions=[
                RefinementPredicate("self.length", "==", None, symbolic_operand="self.length@pre"),
            ],
            return_type=RefinementType(base=RefinementSort.LIST),
            modifies_receiver=True,
        )

    def model_copyWithin(self, target: int, start: int, end: Optional[int] = None) -> OperationResult:
        return OperationResult(
            postconditions=[
                RefinementPredicate("self.length", "==", None, symbolic_operand="self.length@pre"),
            ],
            return_type=RefinementType(base=RefinementSort.LIST),
            modifies_receiver=True,
        )

    def model_reverse_js(self) -> OperationResult:
        return OperationResult(
            postconditions=[
                RefinementPredicate("self.length", "==", None, symbolic_operand="self.length@pre"),
            ],
            return_type=RefinementType(base=RefinementSort.LIST),
            modifies_receiver=True,
        )

    def model_sort_js(self) -> OperationResult:
        return OperationResult(
            postconditions=[
                RefinementPredicate("self.length", "==", None, symbolic_operand="self.length@pre"),
            ],
            return_type=RefinementType(base=RefinementSort.LIST),
            modifies_receiver=True,
        )


# ====================================================================
# MapRefinementModel (TypeScript ES6 Map)
# ====================================================================

@dataclass
class MapRefinementModel(ContainerRefinement):
    """
    Refinement model for ES6 ``Map``.

    Tracks key presence and size refinements.
    """
    known_keys: Set[str] = field(default_factory=set)
    value_types: Dict[str, RefinementType] = field(default_factory=dict)

    def model_get(self, key_type: RefinementType, key_value: Optional[str] = None) -> OperationResult:
        if key_value is not None and key_value in self.known_keys:
            ret = self.value_types.get(key_value, self.element_type)
            return OperationResult(return_type=ret)
        return OperationResult(return_type=RefinementType(base=RefinementSort.ANY, nullable=True))

    def model_set(self, key_type: RefinementType, value_type: RefinementType, key_value: Optional[str] = None) -> OperationResult:
        posts: List[RefinementPredicate] = [
            RefinementPredicate("self.size", ">=", None, symbolic_operand="self.size@pre"),
        ]
        if key_value:
            posts.append(RefinementPredicate(f"self.has('{key_value}')", "==", True))
        return OperationResult(
            postconditions=posts,
            return_type=RefinementType(base=RefinementSort.OBJECT),
            modifies_receiver=True,
        )

    def model_has(self, key_type: RefinementType, key_value: Optional[str] = None) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if key_value is not None and key_value in self.known_keys:
            preds.append(RefinementPredicate("result", "==", True))
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL, predicates=preds))

    def model_delete(self, key_type: RefinementType) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BOOL),
            modifies_receiver=True,
        )

    def model_clear(self) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("self.size", "==", 0)],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=Bound(value=0),
            new_length_upper=Bound(value=0),
        )

    def model_size(self) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        if self.known_keys:
            preds.append(RefinementPredicate("result", ">=", len(self.known_keys)))
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    def model_keys(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_values(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_entries(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_forEach(self, callback_type: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.NONE))


# ====================================================================
# ES6SetRefinementModel (TypeScript Set)
# ====================================================================

@dataclass
class ES6SetRefinementModel(ContainerRefinement):
    """Refinement model for ES6 ``Set``."""
    known_members: FrozenSet[Any] = field(default_factory=frozenset)

    def model_add(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(
            postconditions=[
                RefinementPredicate("self.size", ">=", None, symbolic_operand="self.size@pre"),
            ],
            return_type=RefinementType(base=RefinementSort.OBJECT),
            modifies_receiver=True,
        )

    def model_has(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_delete(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BOOL),
            modifies_receiver=True,
        )

    def model_clear(self) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("self.size", "==", 0)],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            new_length_lower=Bound(value=0),
            new_length_upper=Bound(value=0),
        )

    def model_size(self) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    def model_keys(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_values(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_entries(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_forEach(self, callback_type: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.NONE))


# ====================================================================
# WeakMapModel / WeakSetModel
# ====================================================================

@dataclass
class WeakMapModel:
    """
    Refinement model for ES6 ``WeakMap``.

    No length/size tracking (GC-dependent). Only key-presence refinements.
    """
    known_keys: Set[str] = field(default_factory=set)

    def model_get(self, key_type: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ANY, nullable=True))

    def model_set(self, key_type: RefinementType, value_type: RefinementType) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT),
            modifies_receiver=True,
        )

    def model_has(self, key_type: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_delete(self, key_type: RefinementType) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BOOL),
            modifies_receiver=True,
        )


@dataclass
class WeakSetModel:
    """Refinement model for ES6 ``WeakSet``."""

    def model_add(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT),
            modifies_receiver=True,
        )

    def model_has(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_delete(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BOOL),
            modifies_receiver=True,
        )


# ====================================================================
# IteratorRefinementModel
# ====================================================================

class IteratorState(enum.Enum):
    FRESH = "fresh"
    ACTIVE = "active"
    EXHAUSTED = "exhausted"


@dataclass
class IteratorRefinementModel:
    """
    Refinement model for iterators.

    Tracks exhaustion state: once exhausted, ``next()`` always returns
    the sentinel / raises ``StopIteration``.
    """
    state: IteratorState = IteratorState.FRESH
    element_type: RefinementType = field(default_factory=lambda: RefinementType(base=RefinementSort.ANY))
    remaining_lower: Optional[int] = None
    remaining_upper: Optional[int] = None

    def model_next(self) -> OperationResult:
        preconds: List[RefinementPredicate] = []
        if self.state == IteratorState.EXHAUSTED:
            return OperationResult(
                preconditions=[RefinementPredicate("state", "!=", "exhausted")],
                return_type=self.element_type,
                exceptions=[("StopIteration", [RefinementPredicate("state", "==", "exhausted")])],
            )
        post: List[RefinementPredicate] = []
        if self.remaining_lower is not None and self.remaining_lower <= 0:
            preconds.append(RefinementPredicate("remaining", ">", 0))
        return OperationResult(
            preconditions=preconds,
            postconditions=post,
            return_type=self.element_type,
            exceptions=[("StopIteration", [RefinementPredicate("state", "==", "exhausted")])],
        )

    def model_iter(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_for_loop(self) -> TransferFunction:
        return TransferFunction(
            output_constraints=[
                RefinementPredicate("loop_var_type", "==", None, symbolic_operand=str(self.element_type.base.value)),
            ],
        )


# ====================================================================
# GeneratorRefinementModel
# ====================================================================

class GeneratorState(enum.Enum):
    CREATED = "created"
    SUSPENDED = "suspended"
    RUNNING = "running"
    CLOSED = "closed"


@dataclass
class GeneratorRefinementModel:
    """
    Refinement model for Python generators.

    Tracks send/throw/close with generator state.
    """
    state: GeneratorState = GeneratorState.CREATED
    yield_type: RefinementType = field(default_factory=lambda: RefinementType(base=RefinementSort.ANY))
    send_type: RefinementType = field(default_factory=lambda: RefinementType(base=RefinementSort.ANY))
    return_type: RefinementType = field(default_factory=lambda: RefinementType(base=RefinementSort.NONE))

    def model_next(self) -> OperationResult:
        preconds: List[RefinementPredicate] = []
        if self.state == GeneratorState.CLOSED:
            return OperationResult(
                exceptions=[("StopIteration", [RefinementPredicate("state", "==", "closed")])],
                return_type=self.yield_type,
            )
        return OperationResult(
            preconditions=preconds,
            return_type=self.yield_type,
            exceptions=[("StopIteration", [])],
        )

    def model_send(self, value_type: RefinementType) -> OperationResult:
        preconds: List[RefinementPredicate] = []
        if self.state == GeneratorState.CREATED:
            preconds.append(RefinementPredicate("value", "is", None))
        return OperationResult(
            preconditions=preconds,
            return_type=self.yield_type,
            exceptions=[("StopIteration", [])],
        )

    def model_throw(self, exc_type: RefinementType) -> OperationResult:
        return OperationResult(
            return_type=self.yield_type,
            exceptions=[
                ("StopIteration", []),
                ("RuntimeError", [RefinementPredicate("state", "==", "closed")]),
            ],
        )

    def model_close(self) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("state", "==", "closed")],
            return_type=RefinementType(base=RefinementSort.NONE),
        )

    def model_iter(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.GENERATOR))


# ====================================================================
# RangeRefinementModel
# ====================================================================

@dataclass
class RangeRefinementModel(ContainerRefinement):
    """
    Refinement model for Python ``range``.

    Derives length from start/stop/step:
        len = max(0, ceil((stop - start) / step))
    """
    start: int = 0
    stop: int = 0
    step: int = 1

    def __post_init__(self) -> None:
        if self.step == 0:
            return
        length = max(0, -(-( self.stop - self.start) // self.step))
        self.length_lower = Bound(value=length)
        self.length_upper = Bound(value=length)
        self.element_type = RefinementType(
            base=RefinementSort.INT,
            predicates=self._element_preds(),
        )

    def _element_preds(self) -> List[RefinementPredicate]:
        preds: List[RefinementPredicate] = []
        if self.step > 0:
            preds.append(RefinementPredicate("element", ">=", self.start))
            preds.append(RefinementPredicate("element", "<", self.stop))
        elif self.step < 0:
            preds.append(RefinementPredicate("element", "<=", self.start))
            preds.append(RefinementPredicate("element", ">", self.stop))
        return preds

    def model_getitem(self, index_type: RefinementType) -> OperationResult:
        preconds = [
            RefinementPredicate("index", ">=", 0),
            RefinementPredicate("index", "<", None, symbolic_operand="len(self)"),
        ]
        return OperationResult(
            preconditions=preconds,
            return_type=self.element_type,
            exceptions=[("IndexError", [RefinementPredicate("index", ">=", None, symbolic_operand="len(self)")])],
        )

    def model_contains(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_len(self) -> OperationResult:
        preds = []
        if self.length_lower.is_concrete():
            preds.append(RefinementPredicate("result", "==", self.length_lower.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    def model_iter(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_index(self, value_type: RefinementType) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("result", "<", self.length_upper.value))
        return OperationResult(
            preconditions=[RefinementPredicate("value", "in", None, symbolic_operand="self")],
            return_type=RefinementType(base=RefinementSort.INT, predicates=preds),
            exceptions=[("ValueError", [RefinementPredicate("value", "not_in", None, symbolic_operand="self")])],
        )

    def model_count(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[
                    RefinementPredicate("result", ">=", 0),
                    RefinementPredicate("result", "<=", 1),
                ],
            ),
        )

    def model_slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        return OperationResult(return_type=RefinementType(base=RefinementSort.LIST, predicates=preds))

    def model_eq(self, other: Optional[RangeRefinementModel] = None) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if (other is not None
                and self.start == other.start and self.stop == other.stop and self.step == other.step):
            preds.append(RefinementPredicate("result", "==", True))
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL, predicates=preds))


# ====================================================================
# StringRefinementModel
# ====================================================================

@dataclass
class StringRefinementModel(ContainerRefinement):
    """
    Refinement model for ``str`` as a container.

    Tracks length and models string operations with refinement propagation.
    """

    def __post_init__(self) -> None:
        self.element_type = RefinementType(
            base=RefinementSort.STR,
            predicates=[RefinementPredicate("len(element)", "==", 1)],
        )

    # -- __getitem__ ---------------------------------------------------------

    def model_getitem(self, index_type: RefinementType) -> OperationResult:
        preconds = [
            RefinementPredicate("index", ">=", 0, symbolic_operand="-len(self)"),
            RefinementPredicate("index", "<", None, symbolic_operand="len(self)"),
        ]
        ret = RefinementType(
            base=RefinementSort.STR,
            predicates=[RefinementPredicate("len(result)", "==", 1)],
        )
        return OperationResult(
            preconditions=preconds,
            return_type=ret,
            exceptions=[("IndexError", [RefinementPredicate("index", ">=", None, symbolic_operand="len(self)")])],
        )

    # -- __len__ -------------------------------------------------------------

    def model_len(self) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        if self.length_lower.is_concrete():
            preds.append(RefinementPredicate("result", ">=", self.length_lower.value))
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("result", "<=", self.length_upper.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    # -- __contains__ --------------------------------------------------------

    def model_contains(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    # -- __iter__ ------------------------------------------------------------

    def model_iter(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    # -- split ---------------------------------------------------------------

    def model_split(self, sep: Optional[str] = None, maxsplit: Optional[int] = None) -> OperationResult:
        """
        str.split()

        Returns list.  Length >= 1 always.  If maxsplit is given, length <= maxsplit + 1.
        """
        preds = [RefinementPredicate("len(result)", ">=", 1)]
        if maxsplit is not None:
            preds.append(RefinementPredicate("len(result)", "<=", maxsplit + 1))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.LIST, predicates=preds),
        )

    # -- join ----------------------------------------------------------------

    def model_join(self, iterable_model: Optional[ContainerRefinement] = None) -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        return OperationResult(return_type=RefinementType(base=RefinementSort.STR, predicates=preds))

    # -- format --------------------------------------------------------------

    def model_format(self, *args: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.STR))

    # -- encode / decode -----------------------------------------------------

    def model_encode(self, encoding: str = "utf-8") -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BYTES, predicates=preds),
            exceptions=[("UnicodeEncodeError", [])],
        )

    def model_decode(self, encoding: str = "utf-8") -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.STR, predicates=preds),
            exceptions=[("UnicodeDecodeError", [])],
        )

    # -- upper / lower / strip / replace / startswith / endswith / find ------

    def model_upper(self) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if self.length_lower.is_concrete():
            preds.append(RefinementPredicate("len(result)", "==", self.length_lower.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.STR, predicates=preds))

    def model_lower(self) -> OperationResult:
        return self.model_upper()

    def model_strip(self) -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("len(result)", "<=", self.length_upper.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.STR, predicates=preds))

    def model_replace(self, old: str, new: str, count: Optional[int] = None) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.STR))

    def model_startswith(self, prefix: str) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_endswith(self, suffix: str) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_find(self, sub: str) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", -1)]
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("result", "<", self.length_upper.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    # -- slice ---------------------------------------------------------------

    def model_slice(self, start: Optional[int] = None, stop: Optional[int] = None) -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("len(result)", "<=", self.length_upper.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.STR, predicates=preds))

    # -- __add__ / __mul__ ---------------------------------------------------

    def model_add(self, other: Optional[StringRefinementModel] = None) -> OperationResult:
        preds: List[RefinementPredicate] = [RefinementPredicate("len(result)", ">=", 0)]
        if (other is not None and self.length_lower.is_concrete() and other.length_lower.is_concrete()):
            combined = (self.length_lower.value or 0) + (other.length_lower.value or 0)
            preds.append(RefinementPredicate("len(result)", ">=", combined))
        return OperationResult(return_type=RefinementType(base=RefinementSort.STR, predicates=preds))

    def model_mul(self, n_type: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.STR))

    # -- isdigit / isalpha / isalnum etc. ------------------------------------

    def model_isdigit(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_isalpha(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_isalnum(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    # -- count / index -------------------------------------------------------

    def model_count(self, sub: str) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    def model_index(self, sub: str) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        return OperationResult(
            preconditions=[RefinementPredicate("sub", "in", None, symbolic_operand="self")],
            return_type=RefinementType(base=RefinementSort.INT, predicates=preds),
            exceptions=[("ValueError", [RefinementPredicate("sub", "not_in", None, symbolic_operand="self")])],
        )


# ====================================================================
# BytesRefinementModel
# ====================================================================

@dataclass
class BytesRefinementModel(ContainerRefinement):
    """
    Refinement model for ``bytes`` / ``bytearray`` as containers.
    """

    def __post_init__(self) -> None:
        self.element_type = RefinementType(
            base=RefinementSort.INT,
            predicates=[
                RefinementPredicate("element", ">=", 0),
                RefinementPredicate("element", "<=", 255),
            ],
        )

    def model_getitem(self, index_type: RefinementType) -> OperationResult:
        preconds = [
            RefinementPredicate("index", ">=", 0, symbolic_operand="-len(self)"),
            RefinementPredicate("index", "<", None, symbolic_operand="len(self)"),
        ]
        return OperationResult(
            preconditions=preconds,
            return_type=self.element_type,
            exceptions=[("IndexError", [RefinementPredicate("index", ">=", None, symbolic_operand="len(self)")])],
        )

    def model_len(self) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        if self.length_lower.is_concrete():
            preds.append(RefinementPredicate("result", ">=", self.length_lower.value))
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("result", "<=", self.length_upper.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    def model_contains(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_decode(self, encoding: str = "utf-8") -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.STR),
            exceptions=[("UnicodeDecodeError", [])],
        )

    def model_split(self, sep: Optional[bytes] = None, maxsplit: Optional[int] = None) -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">=", 1)]
        if maxsplit is not None:
            preds.append(RefinementPredicate("len(result)", "<=", maxsplit + 1))
        return OperationResult(return_type=RefinementType(base=RefinementSort.LIST, predicates=preds))

    def model_join(self, iterable_model: Optional[ContainerRefinement] = None) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BYTES))

    def model_find(self, sub: bytes) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", -1)]
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    def model_replace(self, old: bytes, new: bytes) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BYTES))

    def model_strip(self) -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("len(result)", "<=", self.length_upper.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.BYTES, predicates=preds))

    def model_hex(self) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if self.length_lower.is_concrete() and self.length_lower.value is not None:
            preds.append(RefinementPredicate("len(result)", "==", self.length_lower.value * 2))
        return OperationResult(return_type=RefinementType(base=RefinementSort.STR, predicates=preds))

    def model_slice(self, start: Optional[int] = None, stop: Optional[int] = None) -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("len(result)", "<=", self.length_upper.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.BYTES, predicates=preds))

    def model_iter(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_count(self, sub: bytes) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    def model_index(self, sub: bytes) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        return OperationResult(
            preconditions=[RefinementPredicate("sub", "in", None, symbolic_operand="self")],
            return_type=RefinementType(base=RefinementSort.INT, predicates=preds),
            exceptions=[("ValueError", [RefinementPredicate("sub", "not_in", None, symbolic_operand="self")])],
        )

    def model_upper(self) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if self.length_lower.is_concrete():
            preds.append(RefinementPredicate("len(result)", "==", self.length_lower.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.BYTES, predicates=preds))

    def model_lower(self) -> OperationResult:
        return self.model_upper()

    def model_startswith(self, prefix: bytes) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_endswith(self, suffix: bytes) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))


# ====================================================================
# FrozenSetRefinementModel
# ====================================================================

@dataclass
class FrozenSetRefinementModel(ContainerRefinement):
    """
    Refinement model for Python ``frozenset``.

    Immutable set — no mutating operations.
    """
    known_members: FrozenSet[Any] = field(default_factory=frozenset)

    def model_contains(self, value_type: RefinementType) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_len(self) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        if self.length_upper.is_concrete():
            preds.append(RefinementPredicate("result", "<=", self.length_upper.value))
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    def model_iter(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_copy(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.SET))

    def model_union(self, other: Optional[FrozenSetRefinementModel] = None) -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">=", None, symbolic_operand="max(len(self), len(other))")]
        return OperationResult(return_type=RefinementType(base=RefinementSort.SET, predicates=preds))

    def model_intersection(self, other: Optional[FrozenSetRefinementModel] = None) -> OperationResult:
        preds = [RefinementPredicate("len(result)", "<=", None, symbolic_operand="min(len(self), len(other))")]
        return OperationResult(return_type=RefinementType(base=RefinementSort.SET, predicates=preds))

    def model_difference(self, other: Optional[FrozenSetRefinementModel] = None) -> OperationResult:
        preds = [
            RefinementPredicate("len(result)", ">=", 0),
            RefinementPredicate("len(result)", "<=", None, symbolic_operand="len(self)"),
        ]
        return OperationResult(return_type=RefinementType(base=RefinementSort.SET, predicates=preds))

    def model_symmetric_difference(self, other: Optional[FrozenSetRefinementModel] = None) -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        return OperationResult(return_type=RefinementType(base=RefinementSort.SET, predicates=preds))

    def model_issubset(self, other: Optional[FrozenSetRefinementModel] = None) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_issuperset(self, other: Optional[FrozenSetRefinementModel] = None) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_isdisjoint(self, other: Optional[FrozenSetRefinementModel] = None) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))


# ====================================================================
# OrderedDictRefinementModel
# ====================================================================

@dataclass
class OrderedDictRefinementModel(DictRefinementModel):
    """
    Refinement model for ``collections.OrderedDict``.

    Inherits from DictRefinementModel with insertion-order guarantees.
    """
    _insertion_order: List[str] = field(default_factory=list)

    def model_move_to_end(self, key_type: RefinementType, last: bool = True) -> OperationResult:
        preconds = [RefinementPredicate("key", "in", None, symbolic_operand="self")]
        return OperationResult(
            preconditions=preconds,
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre)"),
            ],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            exceptions=[("KeyError", [RefinementPredicate("key", "not_in", None, symbolic_operand="self")])],
        )

    def model_popitem(self, last: bool = True) -> OperationResult:
        preconds = [RefinementPredicate("len(self)", ">", 0)]
        lo, hi = self._dec_length(1)
        return OperationResult(
            preconditions=preconds,
            postconditions=[
                RefinementPredicate("len(self)", "==", None, symbolic_operand="len(self@pre) - 1"),
            ],
            return_type=RefinementType(base=RefinementSort.TUPLE),
            modifies_receiver=True,
            new_length_lower=lo,
            new_length_upper=hi,
            exceptions=[("KeyError", [RefinementPredicate("len(self)", "==", 0)])],
        )


# ====================================================================
# DefaultDictRefinementModel
# ====================================================================

@dataclass
class DefaultDictRefinementModel(DictRefinementModel):
    """
    Refinement model for ``collections.defaultdict``.

    __getitem__ never raises KeyError — inserts default instead.
    """
    default_factory_type: Optional[RefinementType] = None

    def model_getitem(self, key_type: RefinementType, key_value: Optional[str] = None) -> OperationResult:
        ret = self.element_type
        if key_value is not None and key_value in self.value_types:
            ret = self.value_types[key_value]
        elif self.default_factory_type is not None:
            ret = self.default_factory_type
        posts = [RefinementPredicate("key", "in", None, symbolic_operand="self")]
        return OperationResult(
            postconditions=posts,
            return_type=ret,
            modifies_receiver=True,
            exceptions=[
                ("KeyError", [RefinementPredicate("default_factory", "is", None)]),
            ],
        )

    def model_missing(self, key_type: RefinementType) -> OperationResult:
        if self.default_factory_type is not None:
            return OperationResult(return_type=self.default_factory_type, modifies_receiver=True)
        return OperationResult(
            exceptions=[("KeyError", [RefinementPredicate("default_factory", "is", None)])],
            return_type=self.element_type,
        )


# ====================================================================
# CounterRefinementModel
# ====================================================================

@dataclass
class CounterRefinementModel(DictRefinementModel):
    """
    Refinement model for ``collections.Counter``.

    Values are always ``int``; tracks element counts.
    """
    counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.element_type = RefinementType(base=RefinementSort.INT)

    def model_getitem(self, key_type: RefinementType, key_value: Optional[str] = None) -> OperationResult:
        """Counter never raises KeyError; returns 0 for missing keys."""
        preds = [RefinementPredicate("result", ">=", 0)]
        if key_value is not None and key_value in self.counts:
            preds.append(RefinementPredicate("result", "==", self.counts[key_value]))
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    def model_most_common(self, n: Optional[int] = None) -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        if n is not None:
            preds.append(RefinementPredicate("len(result)", "<=", n))
        return OperationResult(return_type=RefinementType(base=RefinementSort.LIST, predicates=preds))

    def model_elements(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_subtract(self, other: Optional[CounterRefinementModel] = None) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    def model_total(self) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.INT, predicates=[RefinementPredicate("result", ">=", 0)]),
        )

    def model_add_counter(self, other: Optional[CounterRefinementModel] = None) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.DICT))

    def model_sub_counter(self, other: Optional[CounterRefinementModel] = None) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.DICT))

    def model_and_counter(self, other: Optional[CounterRefinementModel] = None) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.DICT))

    def model_or_counter(self, other: Optional[CounterRefinementModel] = None) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.DICT))

    def model_positive(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.DICT))


# ====================================================================
# ChainMapRefinementModel
# ====================================================================

@dataclass
class ChainMapRefinementModel(ContainerRefinement):
    """
    Refinement model for ``collections.ChainMap``.

    Lookup traverses maps in order.  Mutations affect only the first map.
    """
    maps: List[DictRefinementModel] = field(default_factory=list)

    def _all_known_keys(self) -> Set[str]:
        result: Set[str] = set()
        for m in self.maps:
            result |= m.known_keys
        return result

    def model_getitem(self, key_type: RefinementType, key_value: Optional[str] = None) -> OperationResult:
        all_keys = self._all_known_keys()
        preconds = [RefinementPredicate("key", "in", None, symbolic_operand="self")]
        ret = self.element_type
        if key_value is not None:
            for m in self.maps:
                if key_value in m.value_types:
                    ret = m.value_types[key_value]
                    break
        return OperationResult(
            preconditions=preconds,
            return_type=ret,
            exceptions=[("KeyError", [RefinementPredicate("key", "not_in", None, symbolic_operand="self")])],
        )

    def model_contains(self, key_type: RefinementType, key_value: Optional[str] = None) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if key_value is not None and key_value in self._all_known_keys():
            preds.append(RefinementPredicate("result", "==", True))
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL, predicates=preds))

    def model_setitem(self, key_type: RefinementType, value_type: RefinementType) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("key", "in", None, symbolic_operand="self")],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    def model_delitem(self, key_type: RefinementType) -> OperationResult:
        return OperationResult(
            preconditions=[RefinementPredicate("key", "in", None, symbolic_operand="self.maps[0]")],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            exceptions=[("KeyError", [RefinementPredicate("key", "not_in", None, symbolic_operand="self.maps[0]")])],
        )

    def model_new_child(self, m: Optional[DictRefinementModel] = None) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.DICT))

    def model_parents(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.DICT))

    def model_keys(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_values(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_items(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_len(self) -> OperationResult:
        preds = [RefinementPredicate("result", ">=", 0)]
        all_keys = self._all_known_keys()
        if all_keys:
            preds.append(RefinementPredicate("result", ">=", len(all_keys)))
        return OperationResult(return_type=RefinementType(base=RefinementSort.INT, predicates=preds))

    def model_get(self, key_type: RefinementType, default_type: Optional[RefinementType] = None) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ANY, nullable=True))

    def model_pop(self, key_type: RefinementType, default_type: Optional[RefinementType] = None) -> OperationResult:
        if default_type is None:
            preconds = [RefinementPredicate("key", "in", None, symbolic_operand="self.maps[0]")]
            excs: List[Tuple[str, List[RefinementPredicate]]] = [
                ("KeyError", [RefinementPredicate("key", "not_in", None, symbolic_operand="self.maps[0]")]),
            ]
        else:
            preconds = []
            excs = []
        return OperationResult(
            preconditions=preconds,
            return_type=self.element_type,
            modifies_receiver=True,
            exceptions=excs,
        )

    def model_clear(self) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("len(self.maps[0])", "==", 0)],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    def model_copy(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.DICT))


# ====================================================================
# Container registry — maps type names to model classes
# ====================================================================

CONTAINER_MODEL_REGISTRY: Dict[str, type] = {
    "list": ListRefinementModel,
    "tuple": TupleRefinementModel,
    "dict": DictRefinementModel,
    "set": SetRefinementModel,
    "frozenset": FrozenSetRefinementModel,
    "deque": DequeRefinementModel,
    "range": RangeRefinementModel,
    "str": StringRefinementModel,
    "bytes": BytesRefinementModel,
    "bytearray": BytesRefinementModel,
    "OrderedDict": OrderedDictRefinementModel,
    "defaultdict": DefaultDictRefinementModel,
    "Counter": CounterRefinementModel,
    "ChainMap": ChainMapRefinementModel,
    # TypeScript / JS
    "Array": ArrayRefinementModel,
    "Map": MapRefinementModel,
    "Set": ES6SetRefinementModel,
    "WeakMap": WeakMapModel,
    "WeakSet": WeakSetModel,
    # Iterators / generators
    "Iterator": IteratorRefinementModel,
    "Generator": GeneratorRefinementModel,
}


def get_container_model(type_name: str) -> Optional[type]:
    """Look up a container refinement model class by type name."""
    return CONTAINER_MODEL_REGISTRY.get(type_name)

from __future__ import annotations

"""
Refinement type models for TypeScript / Node.js standard library.

Each model encodes parameter types with refinements, return type refinements,
nullability (undefined vs null), type-narrowing effects, exception conditions,
and length / bounds refinements.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Local type definitions (no project imports)
# ---------------------------------------------------------------------------


class BaseType(Enum):
    """Primitive base types used in refinement signatures."""

    ANY = auto()
    UNKNOWN = auto()
    NEVER = auto()
    VOID = auto()
    UNDEFINED = auto()
    NULL = auto()
    BOOLEAN = auto()
    NUMBER = auto()
    BIGINT = auto()
    STRING = auto()
    SYMBOL = auto()
    OBJECT = auto()
    FUNCTION = auto()
    ARRAY = auto()
    TUPLE = auto()
    MAP = auto()
    SET = auto()
    WEAKMAP = auto()
    WEAKSET = auto()
    WEAKREF = auto()
    PROMISE = auto()
    REGEXP = auto()
    DATE = auto()
    ERROR = auto()
    BUFFER = auto()
    STREAM = auto()
    URL = auto()
    ITERABLE = auto()
    ITERATOR = auto()
    ASYNC_ITERABLE = auto()
    ASYNC_ITERATOR = auto()
    GENERATOR = auto()
    EVENT_EMITTER = auto()
    SERVER = auto()
    INCOMING_MESSAGE = auto()
    SERVER_RESPONSE = auto()
    CHILD_PROCESS = auto()


class NullabilityKind(Enum):
    NON_NULLABLE = auto()
    NULLABLE = auto()          # null
    UNDEFINABLE = auto()       # undefined
    NULLABLE_UNDEFINABLE = auto()  # null | undefined


class Variance(Enum):
    COVARIANT = auto()
    CONTRAVARIANT = auto()
    INVARIANT = auto()
    BIVARIANT = auto()


class Purity(Enum):
    PURE = auto()
    READS = auto()
    WRITES = auto()
    IMPURE = auto()


# ---------------------------------------------------------------------------
# Refinement predicates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefinementPredicate:
    """A logical predicate constraining a value, e.g. {n: int | n >= 0}."""

    variable: str
    base_type: BaseType
    constraint: str  # human-readable constraint expression

    def __str__(self) -> str:
        return f"{{{self.variable}: {self.base_type.name} | {self.constraint}}}"


@dataclass(frozen=True)
class CompoundRefinement:
    """Conjunction / disjunction of refinement predicates."""

    predicates: Tuple[RefinementPredicate, ...]
    conjunction: bool = True  # True = AND, False = OR

    def __str__(self) -> str:
        op = " ∧ " if self.conjunction else " ∨ "
        return op.join(str(p) for p in self.predicates)


# ---------------------------------------------------------------------------
# Type representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeParam:
    name: str
    bound: Optional[RefinedType] = None
    default: Optional[RefinedType] = None
    variance: Variance = Variance.INVARIANT


@dataclass(frozen=True)
class RefinedType:
    """A type with optional refinement predicates."""

    base: BaseType
    type_args: Tuple[RefinedType, ...] = ()
    refinement: Optional[Union[RefinementPredicate, CompoundRefinement]] = None
    nullability: NullabilityKind = NullabilityKind.NON_NULLABLE
    literal_value: Optional[Any] = None
    union_members: Optional[Tuple[RefinedType, ...]] = None

    def with_nullability(self, nk: NullabilityKind) -> RefinedType:
        return RefinedType(
            base=self.base,
            type_args=self.type_args,
            refinement=self.refinement,
            nullability=nk,
            literal_value=self.literal_value,
            union_members=self.union_members,
        )

    def with_refinement(
        self, ref: Union[RefinementPredicate, CompoundRefinement]
    ) -> RefinedType:
        return RefinedType(
            base=self.base,
            type_args=self.type_args,
            refinement=ref,
            nullability=self.nullability,
            literal_value=self.literal_value,
            union_members=self.union_members,
        )


# Convenience constructors
def _t(base: BaseType, *args: RefinedType) -> RefinedType:
    return RefinedType(base=base, type_args=tuple(args))


def _ref(var: str, base: BaseType, constraint: str) -> RefinementPredicate:
    return RefinementPredicate(variable=var, base_type=base, constraint=constraint)


def _nullable(rt: RefinedType) -> RefinedType:
    return rt.with_nullability(NullabilityKind.NULLABLE)


def _undefinable(rt: RefinedType) -> RefinedType:
    return rt.with_nullability(NullabilityKind.UNDEFINABLE)


def _nullable_undef(rt: RefinedType) -> RefinedType:
    return rt.with_nullability(NullabilityKind.NULLABLE_UNDEFINABLE)


def _union(*members: RefinedType) -> RefinedType:
    return RefinedType(base=BaseType.ANY, union_members=tuple(members))


# Frequently-used types
T_ANY = _t(BaseType.ANY)
T_UNKNOWN = _t(BaseType.UNKNOWN)
T_NEVER = _t(BaseType.NEVER)
T_VOID = _t(BaseType.VOID)
T_UNDEFINED = _t(BaseType.UNDEFINED)
T_NULL = _t(BaseType.NULL)
T_BOOL = _t(BaseType.BOOLEAN)
T_NUM = _t(BaseType.NUMBER)
T_BIGINT = _t(BaseType.BIGINT)
T_STR = _t(BaseType.STRING)
T_SYM = _t(BaseType.SYMBOL)
T_OBJ = _t(BaseType.OBJECT)
T_FUNC = _t(BaseType.FUNCTION)
T_REGEXP = _t(BaseType.REGEXP)
T_DATE = _t(BaseType.DATE)
T_ERR = _t(BaseType.ERROR)
T_BUF = _t(BaseType.BUFFER)
T_URL = _t(BaseType.URL)

NON_NEG_INT = _ref("n", BaseType.NUMBER, "n >= 0 ∧ Number.isInteger(n)")
POS_INT = _ref("n", BaseType.NUMBER, "n > 0 ∧ Number.isInteger(n)")
VALID_INDEX = _ref("i", BaseType.NUMBER, "Number.isInteger(i)")
ARRAY_LEN = _ref("len", BaseType.NUMBER, "len >= 0 ∧ Number.isInteger(len)")
STRING_LEN = _ref("len", BaseType.NUMBER, "len >= 0 ∧ Number.isInteger(len)")
NON_EMPTY_STR = _ref("s", BaseType.STRING, "s.length > 0")
FINITE_NUM = _ref("n", BaseType.NUMBER, "Number.isFinite(n)")
SAFE_INT = _ref("n", BaseType.NUMBER, "Number.isSafeInteger(n)")
NON_NAN = _ref("n", BaseType.NUMBER, "!Number.isNaN(n)")
UINT8 = _ref("n", BaseType.NUMBER, "0 <= n <= 255 ∧ Number.isInteger(n)")
PORT = _ref("n", BaseType.NUMBER, "0 <= n <= 65535 ∧ Number.isInteger(n)")


# ---------------------------------------------------------------------------
# Parameter / return descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamDescriptor:
    name: str
    param_type: RefinedType
    optional: bool = False
    rest: bool = False
    default_value: Optional[str] = None
    description: str = ""


@dataclass(frozen=True)
class ExceptionCondition:
    exception_type: str
    condition: str
    message_template: str = ""


@dataclass(frozen=True)
class TypeNarrowingEffect:
    """Describes how a call narrows the type of a value in subsequent code."""

    target: str  # e.g. "this", "arg0", "return"
    narrowed_type: RefinedType
    condition: str  # when this narrowing applies
    negated_type: Optional[RefinedType] = None  # type under negation


@dataclass(frozen=True)
class RefinementTransfer:
    """Describes how refinements propagate from arguments to return value."""

    source_param: str
    target: str  # "return" or another param name
    transfer_rule: str


@dataclass(frozen=True)
class Precondition:
    description: str
    formal: str  # logical expression


@dataclass(frozen=True)
class Postcondition:
    description: str
    formal: str


@dataclass(frozen=True)
class MethodSignature:
    name: str
    params: Tuple[ParamDescriptor, ...]
    return_type: RefinedType
    type_params: Tuple[TypeParam, ...] = ()
    preconditions: Tuple[Precondition, ...] = ()
    postconditions: Tuple[Postcondition, ...] = ()
    exceptions: Tuple[ExceptionCondition, ...] = ()
    narrowing_effects: Tuple[TypeNarrowingEffect, ...] = ()
    refinement_transfers: Tuple[RefinementTransfer, ...] = ()
    purity: Purity = Purity.IMPURE
    is_static: bool = False
    is_async: bool = False
    description: str = ""


@dataclass(frozen=True)
class PropertyDescriptor:
    name: str
    prop_type: RefinedType
    readonly: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# Base model class
# ---------------------------------------------------------------------------


class TypeScriptModelBase:
    """Base class for TypeScript standard library type models."""

    qualified_name: str = ""
    type_params: Tuple[TypeParam, ...] = ()
    methods: Dict[str, MethodSignature] = {}
    properties: Dict[str, PropertyDescriptor] = {}
    static_methods: Dict[str, MethodSignature] = {}
    description: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Ensure mutable class-level dicts are not shared
        if "methods" not in cls.__dict__:
            cls.methods = {}
        if "properties" not in cls.__dict__:
            cls.properties = {}
        if "static_methods" not in cls.__dict__:
            cls.static_methods = {}

    # Public API -----------------------------------------------------------

    def get_signature(self, method_name: str) -> Optional[MethodSignature]:
        sig = self.methods.get(method_name)
        if sig is None:
            sig = self.static_methods.get(method_name)
        return sig

    def get_preconditions(self, method_name: str) -> Tuple[Precondition, ...]:
        sig = self.get_signature(method_name)
        return sig.preconditions if sig else ()

    def get_postconditions(self, method_name: str) -> Tuple[Postcondition, ...]:
        sig = self.get_signature(method_name)
        return sig.postconditions if sig else ()

    def get_refinement_transfer(
        self, method_name: str
    ) -> Tuple[RefinementTransfer, ...]:
        sig = self.get_signature(method_name)
        return sig.refinement_transfers if sig else ()

    def get_type_narrowing(
        self, method_name: str
    ) -> Tuple[TypeNarrowingEffect, ...]:
        sig = self.get_signature(method_name)
        return sig.narrowing_effects if sig else ()

    def get_property(self, prop_name: str) -> Optional[PropertyDescriptor]:
        return self.properties.get(prop_name)

    def all_method_names(self) -> FrozenSet[str]:
        return frozenset(self.methods.keys()) | frozenset(
            self.static_methods.keys()
        )


# ===================================================================
# CORE JS/TS TYPES
# ===================================================================


# -------------------------------------------------------------------
# Array<T>
# -------------------------------------------------------------------


class ArrayModel(TypeScriptModelBase):
    qualified_name = "Array"
    description = "JavaScript Array with refinement-typed methods."

    type_params = (TypeParam("T"),)

    T = _t(BaseType.ANY)  # generic element type placeholder

    properties = {
        "length": PropertyDescriptor(
            name="length",
            prop_type=T_NUM.with_refinement(ARRAY_LEN),
            readonly=False,
            description="Number of elements; always a non-negative integer.",
        ),
    }

    methods = {
        "constructor": MethodSignature(
            name="constructor",
            params=(
                ParamDescriptor("arrayLength", T_NUM.with_refinement(NON_NEG_INT), optional=True),
            ),
            return_type=_t(BaseType.ARRAY, T),
            preconditions=(
                Precondition(
                    "Length must be non-negative integer",
                    "arrayLength === undefined ∨ (Number.isInteger(arrayLength) ∧ arrayLength >= 0)",
                ),
            ),
            postconditions=(
                Postcondition("Result length equals argument", "result.length === (arrayLength ?? 0)"),
            ),
            exceptions=(
                ExceptionCondition("RangeError", "arrayLength < 0 ∨ !Number.isInteger(arrayLength)",
                                   "Invalid array length"),
            ),
            purity=Purity.PURE,
            description="Create a new Array.",
        ),
        "push": MethodSignature(
            name="push",
            params=(
                ParamDescriptor("items", T, rest=True),
            ),
            return_type=T_NUM.with_refinement(NON_NEG_INT),
            postconditions=(
                Postcondition(
                    "Length increases by number of items pushed",
                    "result === old(this.length) + items.length",
                ),
                Postcondition("Return value is new length", "result === this.length"),
            ),
            refinement_transfers=(
                RefinementTransfer("items", "return", "return = old(this.length) + items.length"),
            ),
            purity=Purity.WRITES,
            description="Appends elements and returns new length.",
        ),
        "pop": MethodSignature(
            name="pop",
            params=(),
            return_type=_undefinable(T),
            preconditions=(),
            postconditions=(
                Postcondition(
                    "Length decreases by 1 if non-empty",
                    "old(this.length) > 0 ⇒ this.length === old(this.length) - 1",
                ),
                Postcondition(
                    "Returns undefined if empty",
                    "old(this.length) === 0 ⇒ result === undefined",
                ),
            ),
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="return",
                    narrowed_type=T,
                    condition="old(this.length) > 0",
                    negated_type=T_UNDEFINED,
                ),
            ),
            purity=Purity.WRITES,
            description="Removes and returns last element.",
        ),
        "shift": MethodSignature(
            name="shift",
            params=(),
            return_type=_undefinable(T),
            postconditions=(
                Postcondition(
                    "Length decreases by 1 if non-empty",
                    "old(this.length) > 0 ⇒ this.length === old(this.length) - 1",
                ),
            ),
            purity=Purity.WRITES,
            description="Removes and returns first element.",
        ),
        "unshift": MethodSignature(
            name="unshift",
            params=(
                ParamDescriptor("items", T, rest=True),
            ),
            return_type=T_NUM.with_refinement(NON_NEG_INT),
            postconditions=(
                Postcondition(
                    "Length increases by number of items",
                    "result === old(this.length) + items.length",
                ),
            ),
            purity=Purity.WRITES,
            description="Inserts elements at beginning, returns new length.",
        ),
        "splice": MethodSignature(
            name="splice",
            params=(
                ParamDescriptor("start", T_NUM.with_refinement(VALID_INDEX)),
                ParamDescriptor("deleteCount", T_NUM.with_refinement(NON_NEG_INT), optional=True),
                ParamDescriptor("items", T, rest=True),
            ),
            return_type=_t(BaseType.ARRAY, T),
            postconditions=(
                Postcondition(
                    "Returns array of deleted elements",
                    "result.length === min(deleteCount ?? (this.length - start), old(this.length) - start)",
                ),
            ),
            purity=Purity.WRITES,
            description="Remove/replace elements in place.",
        ),
        "slice": MethodSignature(
            name="slice",
            params=(
                ParamDescriptor("start", T_NUM.with_refinement(VALID_INDEX), optional=True),
                ParamDescriptor("end", T_NUM.with_refinement(VALID_INDEX), optional=True),
            ),
            return_type=_t(BaseType.ARRAY, T),
            postconditions=(
                Postcondition(
                    "Result length bounded by source",
                    "result.length <= this.length",
                ),
            ),
            purity=Purity.PURE,
            description="Returns shallow copy of a portion.",
        ),
        "concat": MethodSignature(
            name="concat",
            params=(
                ParamDescriptor("items", _union(_t(BaseType.ARRAY, T), T), rest=True),
            ),
            return_type=_t(BaseType.ARRAY, T),
            postconditions=(
                Postcondition(
                    "Result length >= source length",
                    "result.length >= this.length",
                ),
            ),
            purity=Purity.PURE,
            description="Merge arrays and/or values into new array.",
        ),
        "join": MethodSignature(
            name="join",
            params=(
                ParamDescriptor("separator", T_STR, optional=True, default_value='","'),
            ),
            return_type=T_STR,
            postconditions=(
                Postcondition(
                    "Empty array produces empty string",
                    "this.length === 0 ⇒ result === ''",
                ),
            ),
            purity=Purity.PURE,
            description="Join all elements into a string.",
        ),
        "indexOf": MethodSignature(
            name="indexOf",
            params=(
                ParamDescriptor("searchElement", T),
                ParamDescriptor("fromIndex", T_NUM.with_refinement(VALID_INDEX), optional=True),
            ),
            return_type=T_NUM.with_refinement(
                _ref("r", BaseType.NUMBER, "r >= -1 ∧ r < this.length ∧ Number.isInteger(r)")
            ),
            postconditions=(
                Postcondition("Returns -1 when not found", "result === -1 ∨ (result >= 0 ∧ result < this.length)"),
            ),
            purity=Purity.PURE,
            description="First index of element, or -1.",
        ),
        "lastIndexOf": MethodSignature(
            name="lastIndexOf",
            params=(
                ParamDescriptor("searchElement", T),
                ParamDescriptor("fromIndex", T_NUM.with_refinement(VALID_INDEX), optional=True),
            ),
            return_type=T_NUM.with_refinement(
                _ref("r", BaseType.NUMBER, "r >= -1 ∧ r < this.length ∧ Number.isInteger(r)")
            ),
            purity=Purity.PURE,
            description="Last index of element, or -1.",
        ),
        "includes": MethodSignature(
            name="includes",
            params=(
                ParamDescriptor("searchElement", T),
                ParamDescriptor("fromIndex", T_NUM.with_refinement(VALID_INDEX), optional=True),
            ),
            return_type=T_BOOL,
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="searchElement",
                    narrowed_type=T,
                    condition="result === true",
                ),
            ),
            purity=Purity.PURE,
            description="Whether array contains the element.",
        ),
        "find": MethodSignature(
            name="find",
            params=(
                ParamDescriptor(
                    "predicate",
                    _t(BaseType.FUNCTION),
                    description="(value: T, index: number, array: T[]) => boolean",
                ),
                ParamDescriptor("thisArg", T_ANY, optional=True),
            ),
            return_type=_undefinable(T),
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="return",
                    narrowed_type=T,
                    condition="result !== undefined",
                    negated_type=T_UNDEFINED,
                ),
            ),
            purity=Purity.READS,
            description="First element satisfying predicate, or undefined.",
        ),
        "findIndex": MethodSignature(
            name="findIndex",
            params=(
                ParamDescriptor(
                    "predicate",
                    _t(BaseType.FUNCTION),
                    description="(value: T, index: number, array: T[]) => boolean",
                ),
                ParamDescriptor("thisArg", T_ANY, optional=True),
            ),
            return_type=T_NUM.with_refinement(
                _ref("r", BaseType.NUMBER, "r >= -1 ∧ r < this.length ∧ Number.isInteger(r)")
            ),
            purity=Purity.READS,
            description="Index of first element satisfying predicate, or -1.",
        ),
        "filter": MethodSignature(
            name="filter",
            params=(
                ParamDescriptor(
                    "predicate",
                    _t(BaseType.FUNCTION),
                    description="(value: T, index: number, array: T[]) => boolean",
                ),
                ParamDescriptor("thisArg", T_ANY, optional=True),
            ),
            return_type=_t(BaseType.ARRAY, T),
            postconditions=(
                Postcondition("Result length <= source length", "result.length <= this.length"),
            ),
            refinement_transfers=(
                RefinementTransfer("predicate", "return",
                                   "If predicate is a type guard, result element type is narrowed"),
            ),
            purity=Purity.READS,
            description="New array with elements satisfying predicate.",
        ),
        "map": MethodSignature(
            name="map",
            type_params=(TypeParam("U"),),
            params=(
                ParamDescriptor(
                    "callbackfn",
                    _t(BaseType.FUNCTION),
                    description="(value: T, index: number, array: T[]) => U",
                ),
                ParamDescriptor("thisArg", T_ANY, optional=True),
            ),
            return_type=_t(BaseType.ARRAY, _t(BaseType.ANY)),  # Array<U>
            postconditions=(
                Postcondition("Result length equals source length", "result.length === this.length"),
            ),
            refinement_transfers=(
                RefinementTransfer("this.length", "return.length", "result.length === this.length"),
            ),
            purity=Purity.READS,
            description="New array with callback applied to each element.",
        ),
        "forEach": MethodSignature(
            name="forEach",
            params=(
                ParamDescriptor(
                    "callbackfn",
                    _t(BaseType.FUNCTION),
                    description="(value: T, index: number, array: T[]) => void",
                ),
                ParamDescriptor("thisArg", T_ANY, optional=True),
            ),
            return_type=T_VOID,
            purity=Purity.READS,
            description="Execute callback for each element.",
        ),
        "reduce": MethodSignature(
            name="reduce",
            type_params=(TypeParam("U"),),
            params=(
                ParamDescriptor(
                    "callbackfn",
                    _t(BaseType.FUNCTION),
                    description="(acc: U, value: T, index: number, array: T[]) => U",
                ),
                ParamDescriptor("initialValue", _t(BaseType.ANY), optional=True),
            ),
            return_type=_t(BaseType.ANY),  # U
            preconditions=(
                Precondition(
                    "Empty array requires initial value",
                    "this.length > 0 ∨ initialValue !== undefined",
                ),
            ),
            exceptions=(
                ExceptionCondition(
                    "TypeError",
                    "this.length === 0 ∧ initialValue === undefined",
                    "Reduce of empty array with no initial value",
                ),
            ),
            purity=Purity.READS,
            description="Reduce array to single value.",
        ),
        "reduceRight": MethodSignature(
            name="reduceRight",
            type_params=(TypeParam("U"),),
            params=(
                ParamDescriptor(
                    "callbackfn",
                    _t(BaseType.FUNCTION),
                    description="(acc: U, value: T, index: number, array: T[]) => U",
                ),
                ParamDescriptor("initialValue", _t(BaseType.ANY), optional=True),
            ),
            return_type=_t(BaseType.ANY),
            preconditions=(
                Precondition(
                    "Empty array requires initial value",
                    "this.length > 0 ∨ initialValue !== undefined",
                ),
            ),
            exceptions=(
                ExceptionCondition(
                    "TypeError",
                    "this.length === 0 ∧ initialValue === undefined",
                    "Reduce of empty array with no initial value",
                ),
            ),
            purity=Purity.READS,
            description="Reduce array right-to-left.",
        ),
        "every": MethodSignature(
            name="every",
            params=(
                ParamDescriptor(
                    "predicate",
                    _t(BaseType.FUNCTION),
                    description="(value: T, index: number, array: T[]) => boolean",
                ),
                ParamDescriptor("thisArg", T_ANY, optional=True),
            ),
            return_type=T_BOOL,
            postconditions=(
                Postcondition("Empty array returns true", "this.length === 0 ⇒ result === true"),
            ),
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="this",
                    narrowed_type=_t(BaseType.ARRAY, T),
                    condition="result === true",
                ),
            ),
            purity=Purity.READS,
            description="Whether all elements satisfy predicate.",
        ),
        "some": MethodSignature(
            name="some",
            params=(
                ParamDescriptor(
                    "predicate",
                    _t(BaseType.FUNCTION),
                    description="(value: T, index: number, array: T[]) => boolean",
                ),
                ParamDescriptor("thisArg", T_ANY, optional=True),
            ),
            return_type=T_BOOL,
            postconditions=(
                Postcondition("Empty array returns false", "this.length === 0 ⇒ result === false"),
            ),
            purity=Purity.READS,
            description="Whether any element satisfies predicate.",
        ),
        "sort": MethodSignature(
            name="sort",
            params=(
                ParamDescriptor(
                    "compareFn",
                    _t(BaseType.FUNCTION),
                    optional=True,
                    description="(a: T, b: T) => number",
                ),
            ),
            return_type=_t(BaseType.ARRAY, T),
            postconditions=(
                Postcondition("Length unchanged", "result.length === old(this.length)"),
                Postcondition("Returns this", "result === this"),
            ),
            purity=Purity.WRITES,
            description="Sort array in place.",
        ),
        "reverse": MethodSignature(
            name="reverse",
            params=(),
            return_type=_t(BaseType.ARRAY, T),
            postconditions=(
                Postcondition("Length unchanged", "result.length === old(this.length)"),
                Postcondition("Returns this", "result === this"),
            ),
            purity=Purity.WRITES,
            description="Reverse array in place.",
        ),
        "flat": MethodSignature(
            name="flat",
            type_params=(TypeParam("D"),),
            params=(
                ParamDescriptor("depth", T_NUM.with_refinement(NON_NEG_INT), optional=True, default_value="1"),
            ),
            return_type=_t(BaseType.ARRAY, T_ANY),
            purity=Purity.PURE,
            description="Flatten nested arrays to specified depth.",
        ),
        "flatMap": MethodSignature(
            name="flatMap",
            type_params=(TypeParam("U"),),
            params=(
                ParamDescriptor(
                    "callback",
                    _t(BaseType.FUNCTION),
                    description="(value: T, index: number, array: T[]) => U | U[]",
                ),
                ParamDescriptor("thisArg", T_ANY, optional=True),
            ),
            return_type=_t(BaseType.ARRAY, _t(BaseType.ANY)),
            purity=Purity.READS,
            description="Map then flatten by one level.",
        ),
        "fill": MethodSignature(
            name="fill",
            params=(
                ParamDescriptor("value", T),
                ParamDescriptor("start", T_NUM.with_refinement(VALID_INDEX), optional=True),
                ParamDescriptor("end", T_NUM.with_refinement(VALID_INDEX), optional=True),
            ),
            return_type=_t(BaseType.ARRAY, T),
            postconditions=(
                Postcondition("Returns this", "result === this"),
                Postcondition("Length unchanged", "result.length === old(this.length)"),
            ),
            purity=Purity.WRITES,
            description="Fill array with value.",
        ),
        "copyWithin": MethodSignature(
            name="copyWithin",
            params=(
                ParamDescriptor("target", T_NUM.with_refinement(VALID_INDEX)),
                ParamDescriptor("start", T_NUM.with_refinement(VALID_INDEX)),
                ParamDescriptor("end", T_NUM.with_refinement(VALID_INDEX), optional=True),
            ),
            return_type=_t(BaseType.ARRAY, T),
            postconditions=(
                Postcondition("Returns this", "result === this"),
                Postcondition("Length unchanged", "result.length === old(this.length)"),
            ),
            purity=Purity.WRITES,
            description="Copy sequence of elements within array.",
        ),
        "entries": MethodSignature(
            name="entries",
            params=(),
            return_type=_t(BaseType.ITERATOR),
            purity=Purity.PURE,
            description="Iterator of [index, value] pairs.",
        ),
        "keys": MethodSignature(
            name="keys",
            params=(),
            return_type=_t(BaseType.ITERATOR),
            purity=Purity.PURE,
            description="Iterator of indices.",
        ),
        "values": MethodSignature(
            name="values",
            params=(),
            return_type=_t(BaseType.ITERATOR),
            purity=Purity.PURE,
            description="Iterator of values.",
        ),
        "at": MethodSignature(
            name="at",
            params=(
                ParamDescriptor("index", T_NUM.with_refinement(VALID_INDEX)),
            ),
            return_type=_undefinable(T),
            postconditions=(
                Postcondition(
                    "Returns undefined for out-of-bounds",
                    "(index >= this.length ∨ index < -this.length) ⇒ result === undefined",
                ),
            ),
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="return",
                    narrowed_type=T,
                    condition="index >= 0 ∧ index < this.length",
                    negated_type=T_UNDEFINED,
                ),
            ),
            purity=Purity.PURE,
            description="Element at index (supports negative).",
        ),
    }

    static_methods = {
        "isArray": MethodSignature(
            name="isArray",
            params=(ParamDescriptor("arg", T_ANY),),
            return_type=T_BOOL,
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="arg0",
                    narrowed_type=_t(BaseType.ARRAY, T_ANY),
                    condition="result === true",
                ),
            ),
            is_static=True,
            purity=Purity.PURE,
            description="Type guard: is argument an Array?",
        ),
        "from": MethodSignature(
            name="from",
            type_params=(TypeParam("T"), TypeParam("U")),
            params=(
                ParamDescriptor("arrayLike", _t(BaseType.ITERABLE)),
                ParamDescriptor("mapfn", _t(BaseType.FUNCTION), optional=True),
                ParamDescriptor("thisArg", T_ANY, optional=True),
            ),
            return_type=_t(BaseType.ARRAY, T_ANY),
            is_static=True,
            purity=Purity.PURE,
            description="Create array from iterable or array-like.",
        ),
        "of": MethodSignature(
            name="of",
            type_params=(TypeParam("T"),),
            params=(
                ParamDescriptor("items", T_ANY, rest=True),
            ),
            return_type=_t(BaseType.ARRAY, T_ANY),
            postconditions=(
                Postcondition("Length equals args count", "result.length === items.length"),
            ),
            is_static=True,
            purity=Purity.PURE,
            description="Create array from arguments.",
        ),
    }


# -------------------------------------------------------------------
# Map<K, V>
# -------------------------------------------------------------------


class MapModel(TypeScriptModelBase):
    qualified_name = "Map"
    description = "JavaScript Map with refinement types."
    type_params = (TypeParam("K"), TypeParam("V"))

    K = _t(BaseType.ANY)
    V = _t(BaseType.ANY)

    properties = {
        "size": PropertyDescriptor(
            name="size",
            prop_type=T_NUM.with_refinement(NON_NEG_INT),
            readonly=True,
            description="Number of key-value pairs.",
        ),
    }

    methods = {
        "constructor": MethodSignature(
            name="constructor",
            params=(
                ParamDescriptor("entries", _t(BaseType.ITERABLE), optional=True),
            ),
            return_type=_t(BaseType.MAP, K, V),
            purity=Purity.PURE,
            description="Create a new Map.",
        ),
        "set": MethodSignature(
            name="set",
            params=(
                ParamDescriptor("key", K),
                ParamDescriptor("value", V),
            ),
            return_type=_t(BaseType.MAP, K, V),
            postconditions=(
                Postcondition("Key exists after set", "this.has(key) === true"),
                Postcondition("Returns this", "result === this"),
                Postcondition(
                    "Size increases by at most 1",
                    "this.size <= old(this.size) + 1",
                ),
            ),
            purity=Purity.WRITES,
            description="Set key-value pair, returns the Map.",
        ),
        "get": MethodSignature(
            name="get",
            params=(ParamDescriptor("key", K),),
            return_type=_undefinable(V),
            postconditions=(
                Postcondition(
                    "Returns undefined if key absent",
                    "!this.has(key) ⇒ result === undefined",
                ),
            ),
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="return",
                    narrowed_type=V,
                    condition="this.has(key)",
                    negated_type=T_UNDEFINED,
                ),
            ),
            purity=Purity.READS,
            description="Get value for key, or undefined.",
        ),
        "has": MethodSignature(
            name="has",
            params=(ParamDescriptor("key", K),),
            return_type=T_BOOL,
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="this.get(key)",
                    narrowed_type=V,
                    condition="result === true",
                    negated_type=T_UNDEFINED,
                ),
            ),
            purity=Purity.READS,
            description="Whether Map contains key.",
        ),
        "delete": MethodSignature(
            name="delete",
            params=(ParamDescriptor("key", K),),
            return_type=T_BOOL,
            postconditions=(
                Postcondition("Key absent after delete", "this.has(key) === false"),
                Postcondition(
                    "Returns true iff key was present",
                    "result === old(this.has(key))",
                ),
            ),
            purity=Purity.WRITES,
            description="Remove key-value pair.",
        ),
        "clear": MethodSignature(
            name="clear",
            params=(),
            return_type=T_VOID,
            postconditions=(
                Postcondition("Size becomes 0", "this.size === 0"),
            ),
            purity=Purity.WRITES,
            description="Remove all entries.",
        ),
        "entries": MethodSignature(
            name="entries",
            params=(),
            return_type=_t(BaseType.ITERATOR),
            purity=Purity.PURE,
            description="Iterator of [key, value] pairs.",
        ),
        "keys": MethodSignature(
            name="keys",
            params=(),
            return_type=_t(BaseType.ITERATOR),
            purity=Purity.PURE,
            description="Iterator of keys.",
        ),
        "values": MethodSignature(
            name="values",
            params=(),
            return_type=_t(BaseType.ITERATOR),
            purity=Purity.PURE,
            description="Iterator of values.",
        ),
        "forEach": MethodSignature(
            name="forEach",
            params=(
                ParamDescriptor("callbackfn", _t(BaseType.FUNCTION),
                                description="(value: V, key: K, map: Map<K,V>) => void"),
                ParamDescriptor("thisArg", T_ANY, optional=True),
            ),
            return_type=T_VOID,
            purity=Purity.READS,
            description="Execute callback for each entry.",
        ),
    }


# -------------------------------------------------------------------
# Set<T>
# -------------------------------------------------------------------


class SetModel(TypeScriptModelBase):
    qualified_name = "Set"
    description = "JavaScript Set with refinement types."
    type_params = (TypeParam("T"),)
    T = _t(BaseType.ANY)

    properties = {
        "size": PropertyDescriptor(
            name="size",
            prop_type=T_NUM.with_refinement(NON_NEG_INT),
            readonly=True,
            description="Number of elements.",
        ),
    }

    methods = {
        "constructor": MethodSignature(
            name="constructor",
            params=(ParamDescriptor("values", _t(BaseType.ITERABLE), optional=True),),
            return_type=_t(BaseType.SET, T),
            purity=Purity.PURE,
        ),
        "add": MethodSignature(
            name="add",
            params=(ParamDescriptor("value", T),),
            return_type=_t(BaseType.SET, T),
            postconditions=(
                Postcondition("Element present after add", "this.has(value) === true"),
                Postcondition("Returns this", "result === this"),
            ),
            purity=Purity.WRITES,
        ),
        "has": MethodSignature(
            name="has",
            params=(ParamDescriptor("value", T),),
            return_type=T_BOOL,
            purity=Purity.READS,
        ),
        "delete": MethodSignature(
            name="delete",
            params=(ParamDescriptor("value", T),),
            return_type=T_BOOL,
            postconditions=(
                Postcondition("Element absent after delete", "this.has(value) === false"),
            ),
            purity=Purity.WRITES,
        ),
        "clear": MethodSignature(
            name="clear",
            params=(),
            return_type=T_VOID,
            postconditions=(Postcondition("Size becomes 0", "this.size === 0"),),
            purity=Purity.WRITES,
        ),
        "entries": MethodSignature(
            name="entries", params=(), return_type=_t(BaseType.ITERATOR), purity=Purity.PURE,
        ),
        "keys": MethodSignature(
            name="keys", params=(), return_type=_t(BaseType.ITERATOR), purity=Purity.PURE,
        ),
        "values": MethodSignature(
            name="values", params=(), return_type=_t(BaseType.ITERATOR), purity=Purity.PURE,
        ),
        "forEach": MethodSignature(
            name="forEach",
            params=(
                ParamDescriptor("callbackfn", _t(BaseType.FUNCTION)),
                ParamDescriptor("thisArg", T_ANY, optional=True),
            ),
            return_type=T_VOID,
            purity=Purity.READS,
        ),
        "union": MethodSignature(
            name="union",
            params=(ParamDescriptor("other", _t(BaseType.SET, T)),),
            return_type=_t(BaseType.SET, T),
            postconditions=(
                Postcondition("Result size >= max(this.size, other.size)",
                              "result.size >= max(this.size, other.size)"),
            ),
            purity=Purity.PURE,
        ),
        "intersection": MethodSignature(
            name="intersection",
            params=(ParamDescriptor("other", _t(BaseType.SET, T)),),
            return_type=_t(BaseType.SET, T),
            postconditions=(
                Postcondition("Result size <= min(this.size, other.size)",
                              "result.size <= min(this.size, other.size)"),
            ),
            purity=Purity.PURE,
        ),
        "difference": MethodSignature(
            name="difference",
            params=(ParamDescriptor("other", _t(BaseType.SET, T)),),
            return_type=_t(BaseType.SET, T),
            postconditions=(
                Postcondition("Result size <= this.size", "result.size <= this.size"),
            ),
            purity=Purity.PURE,
        ),
    }


# -------------------------------------------------------------------
# WeakMap / WeakSet / WeakRef
# -------------------------------------------------------------------


class WeakMapModel(TypeScriptModelBase):
    qualified_name = "WeakMap"
    type_params = (TypeParam("K"), TypeParam("V"))
    K = _t(BaseType.OBJECT)
    V = _t(BaseType.ANY)
    methods = {
        "constructor": MethodSignature(
            name="constructor",
            params=(ParamDescriptor("entries", _t(BaseType.ITERABLE), optional=True),),
            return_type=_t(BaseType.WEAKMAP, K, V),
            purity=Purity.PURE,
        ),
        "set": MethodSignature(
            name="set",
            params=(ParamDescriptor("key", K), ParamDescriptor("value", V)),
            return_type=_t(BaseType.WEAKMAP, K, V),
            preconditions=(Precondition("Key must be an object", "typeof key === 'object' ∧ key !== null"),),
            purity=Purity.WRITES,
        ),
        "get": MethodSignature(
            name="get",
            params=(ParamDescriptor("key", K),),
            return_type=_undefinable(V),
            purity=Purity.READS,
        ),
        "has": MethodSignature(
            name="has",
            params=(ParamDescriptor("key", K),),
            return_type=T_BOOL,
            purity=Purity.READS,
        ),
        "delete": MethodSignature(
            name="delete",
            params=(ParamDescriptor("key", K),),
            return_type=T_BOOL,
            purity=Purity.WRITES,
        ),
    }


class WeakSetModel(TypeScriptModelBase):
    qualified_name = "WeakSet"
    type_params = (TypeParam("T"),)
    T = _t(BaseType.OBJECT)
    methods = {
        "constructor": MethodSignature(
            name="constructor",
            params=(ParamDescriptor("values", _t(BaseType.ITERABLE), optional=True),),
            return_type=_t(BaseType.WEAKSET, T),
            purity=Purity.PURE,
        ),
        "add": MethodSignature(
            name="add",
            params=(ParamDescriptor("value", T),),
            return_type=_t(BaseType.WEAKSET, T),
            preconditions=(Precondition("Value must be an object", "typeof value === 'object' ∧ value !== null"),),
            purity=Purity.WRITES,
        ),
        "has": MethodSignature(
            name="has", params=(ParamDescriptor("value", T),), return_type=T_BOOL, purity=Purity.READS,
        ),
        "delete": MethodSignature(
            name="delete", params=(ParamDescriptor("value", T),), return_type=T_BOOL, purity=Purity.WRITES,
        ),
    }


class WeakRefModel(TypeScriptModelBase):
    qualified_name = "WeakRef"
    type_params = (TypeParam("T"),)
    T = _t(BaseType.OBJECT)
    methods = {
        "constructor": MethodSignature(
            name="constructor",
            params=(ParamDescriptor("target", T),),
            return_type=_t(BaseType.WEAKREF, T),
            preconditions=(Precondition("Target must be an object", "typeof target === 'object' ∧ target !== null"),),
            purity=Purity.PURE,
        ),
        "deref": MethodSignature(
            name="deref",
            params=(),
            return_type=_undefinable(T),
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="return",
                    narrowed_type=T,
                    condition="result !== undefined",
                    negated_type=T_UNDEFINED,
                ),
            ),
            purity=Purity.READS,
            description="Get target or undefined if GC'd.",
        ),
    }


# -------------------------------------------------------------------
# Promise<T>
# -------------------------------------------------------------------


class PromiseModel(TypeScriptModelBase):
    qualified_name = "Promise"
    type_params = (TypeParam("T"),)
    T = _t(BaseType.ANY)

    methods = {
        "constructor": MethodSignature(
            name="constructor",
            params=(
                ParamDescriptor(
                    "executor", _t(BaseType.FUNCTION),
                    description="(resolve: (value: T) => void, reject: (reason?: any) => void) => void",
                ),
            ),
            return_type=_t(BaseType.PROMISE, T),
            purity=Purity.IMPURE,
        ),
        "then": MethodSignature(
            name="then",
            type_params=(TypeParam("TResult1"), TypeParam("TResult2")),
            params=(
                ParamDescriptor("onfulfilled", _nullable(_t(BaseType.FUNCTION)), optional=True,
                                description="(value: T) => TResult1 | PromiseLike<TResult1>"),
                ParamDescriptor("onrejected", _nullable(_t(BaseType.FUNCTION)), optional=True,
                                description="(reason: any) => TResult2 | PromiseLike<TResult2>"),
            ),
            return_type=_t(BaseType.PROMISE, T_ANY),
            is_async=True,
            purity=Purity.READS,
        ),
        "catch": MethodSignature(
            name="catch",
            type_params=(TypeParam("TResult"),),
            params=(
                ParamDescriptor("onrejected", _nullable(_t(BaseType.FUNCTION)), optional=True),
            ),
            return_type=_t(BaseType.PROMISE, T_ANY),
            is_async=True,
            purity=Purity.READS,
        ),
        "finally": MethodSignature(
            name="finally",
            params=(
                ParamDescriptor("onfinally", _nullable(_t(BaseType.FUNCTION)), optional=True),
            ),
            return_type=_t(BaseType.PROMISE, T),
            is_async=True,
            purity=Purity.READS,
        ),
    }

    static_methods = {
        "resolve": MethodSignature(
            name="resolve",
            type_params=(TypeParam("T"),),
            params=(ParamDescriptor("value", T, optional=True),),
            return_type=_t(BaseType.PROMISE, T),
            is_static=True,
            purity=Purity.PURE,
        ),
        "reject": MethodSignature(
            name="reject",
            params=(ParamDescriptor("reason", T_ANY, optional=True),),
            return_type=_t(BaseType.PROMISE, T_NEVER),
            is_static=True,
            purity=Purity.PURE,
        ),
        "all": MethodSignature(
            name="all",
            type_params=(TypeParam("T"),),
            params=(ParamDescriptor("values", _t(BaseType.ITERABLE)),),
            return_type=_t(BaseType.PROMISE, _t(BaseType.ARRAY, T_ANY)),
            postconditions=(
                Postcondition(
                    "Resolves when all input promises resolve",
                    "all(values, v => v.state === 'fulfilled') ⇒ result.state === 'fulfilled'",
                ),
            ),
            is_static=True,
            purity=Purity.READS,
        ),
        "allSettled": MethodSignature(
            name="allSettled",
            type_params=(TypeParam("T"),),
            params=(ParamDescriptor("values", _t(BaseType.ITERABLE)),),
            return_type=_t(BaseType.PROMISE, _t(BaseType.ARRAY, T_OBJ)),
            postconditions=(
                Postcondition("Always resolves, never rejects", "result.state === 'fulfilled'"),
            ),
            is_static=True,
            purity=Purity.READS,
        ),
        "race": MethodSignature(
            name="race",
            type_params=(TypeParam("T"),),
            params=(ParamDescriptor("values", _t(BaseType.ITERABLE)),),
            return_type=_t(BaseType.PROMISE, T_ANY),
            is_static=True,
            purity=Purity.READS,
        ),
        "any": MethodSignature(
            name="any",
            type_params=(TypeParam("T"),),
            params=(ParamDescriptor("values", _t(BaseType.ITERABLE)),),
            return_type=_t(BaseType.PROMISE, T_ANY),
            exceptions=(
                ExceptionCondition("AggregateError", "all(values, v => v.state === 'rejected')",
                                   "All promises were rejected"),
            ),
            is_static=True,
            purity=Purity.READS,
        ),
    }


# -------------------------------------------------------------------
# String
# -------------------------------------------------------------------


class StringModel(TypeScriptModelBase):
    qualified_name = "String"
    description = "JavaScript String with refinement types."

    properties = {
        "length": PropertyDescriptor(
            name="length",
            prop_type=T_NUM.with_refinement(STRING_LEN),
            readonly=True,
            description="String length; non-negative integer.",
        ),
    }

    methods = {
        "charAt": MethodSignature(
            name="charAt",
            params=(ParamDescriptor("pos", T_NUM.with_refinement(NON_NEG_INT)),),
            return_type=T_STR,
            postconditions=(
                Postcondition("Result length is 0 or 1", "result.length <= 1"),
                Postcondition("Empty string for out-of-range", "pos >= this.length ⇒ result === ''"),
            ),
            purity=Purity.PURE,
        ),
        "charCodeAt": MethodSignature(
            name="charCodeAt",
            params=(ParamDescriptor("index", T_NUM.with_refinement(NON_NEG_INT)),),
            return_type=T_NUM,
            postconditions=(
                Postcondition(
                    "Returns NaN for out-of-range",
                    "index >= this.length ⇒ Number.isNaN(result)",
                ),
                Postcondition(
                    "Valid range is 0-65535",
                    "index < this.length ⇒ (result >= 0 ∧ result <= 65535)",
                ),
            ),
            purity=Purity.PURE,
        ),
        "codePointAt": MethodSignature(
            name="codePointAt",
            params=(ParamDescriptor("pos", T_NUM.with_refinement(NON_NEG_INT)),),
            return_type=_undefinable(T_NUM),
            purity=Purity.PURE,
        ),
        "concat": MethodSignature(
            name="concat",
            params=(ParamDescriptor("strings", T_STR, rest=True),),
            return_type=T_STR,
            postconditions=(
                Postcondition(
                    "Result length is sum of all lengths",
                    "result.length === this.length + sum(strings, s => s.length)",
                ),
            ),
            refinement_transfers=(
                RefinementTransfer("this.length", "return.length",
                                   "return.length >= this.length"),
            ),
            purity=Purity.PURE,
        ),
        "includes": MethodSignature(
            name="includes",
            params=(
                ParamDescriptor("searchString", T_STR),
                ParamDescriptor("position", T_NUM.with_refinement(NON_NEG_INT), optional=True),
            ),
            return_type=T_BOOL,
            postconditions=(
                Postcondition(
                    "Empty search always true",
                    "searchString === '' ⇒ result === true",
                ),
            ),
            purity=Purity.PURE,
        ),
        "endsWith": MethodSignature(
            name="endsWith",
            params=(
                ParamDescriptor("searchString", T_STR),
                ParamDescriptor("endPosition", T_NUM.with_refinement(NON_NEG_INT), optional=True),
            ),
            return_type=T_BOOL,
            purity=Purity.PURE,
        ),
        "startsWith": MethodSignature(
            name="startsWith",
            params=(
                ParamDescriptor("searchString", T_STR),
                ParamDescriptor("position", T_NUM.with_refinement(NON_NEG_INT), optional=True),
            ),
            return_type=T_BOOL,
            purity=Purity.PURE,
        ),
        "indexOf": MethodSignature(
            name="indexOf",
            params=(
                ParamDescriptor("searchValue", T_STR),
                ParamDescriptor("fromIndex", T_NUM.with_refinement(VALID_INDEX), optional=True),
            ),
            return_type=T_NUM.with_refinement(
                _ref("r", BaseType.NUMBER, "r >= -1 ∧ r < this.length")
            ),
            purity=Purity.PURE,
        ),
        "lastIndexOf": MethodSignature(
            name="lastIndexOf",
            params=(
                ParamDescriptor("searchValue", T_STR),
                ParamDescriptor("fromIndex", T_NUM.with_refinement(VALID_INDEX), optional=True),
            ),
            return_type=T_NUM.with_refinement(
                _ref("r", BaseType.NUMBER, "r >= -1 ∧ r < this.length")
            ),
            purity=Purity.PURE,
        ),
        "match": MethodSignature(
            name="match",
            params=(ParamDescriptor("regexp", _union(T_STR, T_REGEXP)),),
            return_type=_nullable(_t(BaseType.ARRAY, T_STR)),
            purity=Purity.PURE,
        ),
        "matchAll": MethodSignature(
            name="matchAll",
            params=(ParamDescriptor("regexp", T_REGEXP),),
            return_type=_t(BaseType.ITERATOR),
            preconditions=(
                Precondition("RegExp must have global flag", "regexp.flags.includes('g')"),
            ),
            exceptions=(
                ExceptionCondition("TypeError", "!regexp.flags.includes('g')",
                                   "String.prototype.matchAll called with a non-global RegExp argument"),
            ),
            purity=Purity.PURE,
        ),
        "normalize": MethodSignature(
            name="normalize",
            params=(
                ParamDescriptor("form", T_STR, optional=True, default_value='"NFC"'),
            ),
            return_type=T_STR,
            preconditions=(
                Precondition(
                    "Form must be valid",
                    "form ∈ {'NFC', 'NFD', 'NFKC', 'NFKD', undefined}",
                ),
            ),
            exceptions=(
                ExceptionCondition("RangeError", "form ∉ {'NFC','NFD','NFKC','NFKD'}",
                                   "The normalization form should be one of NFC, NFD, NFKC, NFKD"),
            ),
            purity=Purity.PURE,
        ),
        "padEnd": MethodSignature(
            name="padEnd",
            params=(
                ParamDescriptor("maxLength", T_NUM.with_refinement(NON_NEG_INT)),
                ParamDescriptor("fillString", T_STR, optional=True, default_value='" "'),
            ),
            return_type=T_STR,
            postconditions=(
                Postcondition("Result length >= maxLength", "result.length >= maxLength"),
                Postcondition("Result length >= this.length", "result.length >= this.length"),
            ),
            purity=Purity.PURE,
        ),
        "padStart": MethodSignature(
            name="padStart",
            params=(
                ParamDescriptor("maxLength", T_NUM.with_refinement(NON_NEG_INT)),
                ParamDescriptor("fillString", T_STR, optional=True, default_value='" "'),
            ),
            return_type=T_STR,
            postconditions=(
                Postcondition("Result length >= maxLength", "result.length >= maxLength"),
            ),
            purity=Purity.PURE,
        ),
        "repeat": MethodSignature(
            name="repeat",
            params=(ParamDescriptor("count", T_NUM.with_refinement(NON_NEG_INT)),),
            return_type=T_STR,
            preconditions=(
                Precondition("Count must be non-negative", "count >= 0"),
                Precondition("Count must be finite", "Number.isFinite(count)"),
            ),
            postconditions=(
                Postcondition("Result length is count * this.length",
                              "result.length === count * this.length"),
            ),
            exceptions=(
                ExceptionCondition("RangeError", "count < 0 ∨ !Number.isFinite(count)",
                                   "Invalid count value"),
            ),
            purity=Purity.PURE,
        ),
        "replace": MethodSignature(
            name="replace",
            params=(
                ParamDescriptor("searchValue", _union(T_STR, T_REGEXP)),
                ParamDescriptor("replaceValue", _union(T_STR, _t(BaseType.FUNCTION))),
            ),
            return_type=T_STR,
            purity=Purity.PURE,
        ),
        "replaceAll": MethodSignature(
            name="replaceAll",
            params=(
                ParamDescriptor("searchValue", _union(T_STR, T_REGEXP)),
                ParamDescriptor("replaceValue", _union(T_STR, _t(BaseType.FUNCTION))),
            ),
            return_type=T_STR,
            preconditions=(
                Precondition(
                    "RegExp must have global flag",
                    "typeof searchValue === 'string' ∨ searchValue.flags.includes('g')",
                ),
            ),
            exceptions=(
                ExceptionCondition("TypeError",
                                   "searchValue instanceof RegExp ∧ !searchValue.flags.includes('g')",
                                   "String.prototype.replaceAll called with a non-global RegExp argument"),
            ),
            purity=Purity.PURE,
        ),
        "search": MethodSignature(
            name="search",
            params=(ParamDescriptor("regexp", _union(T_STR, T_REGEXP)),),
            return_type=T_NUM.with_refinement(
                _ref("r", BaseType.NUMBER, "r >= -1 ∧ r < this.length")
            ),
            purity=Purity.PURE,
        ),
        "slice": MethodSignature(
            name="slice",
            params=(
                ParamDescriptor("start", T_NUM.with_refinement(VALID_INDEX), optional=True),
                ParamDescriptor("end", T_NUM.with_refinement(VALID_INDEX), optional=True),
            ),
            return_type=T_STR,
            postconditions=(
                Postcondition("Result length <= this.length", "result.length <= this.length"),
            ),
            purity=Purity.PURE,
        ),
        "split": MethodSignature(
            name="split",
            params=(
                ParamDescriptor("separator", _union(T_STR, T_REGEXP)),
                ParamDescriptor("limit", T_NUM.with_refinement(NON_NEG_INT), optional=True),
            ),
            return_type=_t(BaseType.ARRAY, T_STR),
            postconditions=(
                Postcondition(
                    "Result length >= 1",
                    "result.length >= 1",
                ),
                Postcondition(
                    "Result length <= limit when limit provided",
                    "limit !== undefined ⇒ result.length <= limit",
                ),
            ),
            purity=Purity.PURE,
        ),
        "substring": MethodSignature(
            name="substring",
            params=(
                ParamDescriptor("start", T_NUM.with_refinement(NON_NEG_INT)),
                ParamDescriptor("end", T_NUM.with_refinement(NON_NEG_INT), optional=True),
            ),
            return_type=T_STR,
            postconditions=(
                Postcondition("Result length <= this.length", "result.length <= this.length"),
            ),
            purity=Purity.PURE,
        ),
        "toLowerCase": MethodSignature(
            name="toLowerCase",
            params=(),
            return_type=T_STR,
            postconditions=(
                Postcondition("Length preserved", "result.length === this.length"),
            ),
            purity=Purity.PURE,
        ),
        "toUpperCase": MethodSignature(
            name="toUpperCase",
            params=(),
            return_type=T_STR,
            postconditions=(
                Postcondition("Length preserved", "result.length === this.length"),
            ),
            purity=Purity.PURE,
        ),
        "trim": MethodSignature(
            name="trim",
            params=(),
            return_type=T_STR,
            postconditions=(
                Postcondition("Result length <= this.length", "result.length <= this.length"),
            ),
            purity=Purity.PURE,
        ),
        "trimEnd": MethodSignature(
            name="trimEnd",
            params=(),
            return_type=T_STR,
            postconditions=(
                Postcondition("Result length <= this.length", "result.length <= this.length"),
            ),
            purity=Purity.PURE,
        ),
        "trimStart": MethodSignature(
            name="trimStart",
            params=(),
            return_type=T_STR,
            postconditions=(
                Postcondition("Result length <= this.length", "result.length <= this.length"),
            ),
            purity=Purity.PURE,
        ),
        "at": MethodSignature(
            name="at",
            params=(ParamDescriptor("index", T_NUM.with_refinement(VALID_INDEX)),),
            return_type=_undefinable(T_STR),
            postconditions=(
                Postcondition(
                    "Result is single char or undefined",
                    "result === undefined ∨ result.length === 1",
                ),
            ),
            purity=Purity.PURE,
        ),
    }


# -------------------------------------------------------------------
# Number
# -------------------------------------------------------------------


class NumberModel(TypeScriptModelBase):
    qualified_name = "Number"
    description = "JavaScript Number with refinement types."

    properties = {
        "MAX_SAFE_INTEGER": PropertyDescriptor(
            "MAX_SAFE_INTEGER",
            RefinedType(base=BaseType.NUMBER, literal_value=9007199254740991),
            readonly=True,
        ),
        "MIN_SAFE_INTEGER": PropertyDescriptor(
            "MIN_SAFE_INTEGER",
            RefinedType(base=BaseType.NUMBER, literal_value=-9007199254740991),
            readonly=True,
        ),
        "EPSILON": PropertyDescriptor(
            "EPSILON",
            RefinedType(base=BaseType.NUMBER, literal_value=2.220446049250313e-16),
            readonly=True,
        ),
        "MAX_VALUE": PropertyDescriptor(
            "MAX_VALUE",
            RefinedType(base=BaseType.NUMBER, literal_value=1.7976931348623157e308),
            readonly=True,
        ),
        "MIN_VALUE": PropertyDescriptor(
            "MIN_VALUE",
            RefinedType(base=BaseType.NUMBER, literal_value=5e-324),
            readonly=True,
        ),
        "NaN": PropertyDescriptor(
            "NaN",
            T_NUM.with_refinement(_ref("n", BaseType.NUMBER, "Number.isNaN(n)")),
            readonly=True,
        ),
        "POSITIVE_INFINITY": PropertyDescriptor(
            "POSITIVE_INFINITY",
            T_NUM.with_refinement(_ref("n", BaseType.NUMBER, "n === Infinity")),
            readonly=True,
        ),
        "NEGATIVE_INFINITY": PropertyDescriptor(
            "NEGATIVE_INFINITY",
            T_NUM.with_refinement(_ref("n", BaseType.NUMBER, "n === -Infinity")),
            readonly=True,
        ),
    }

    static_methods = {
        "isFinite": MethodSignature(
            name="isFinite",
            params=(ParamDescriptor("value", T_ANY),),
            return_type=T_BOOL,
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="arg0",
                    narrowed_type=T_NUM.with_refinement(FINITE_NUM),
                    condition="result === true",
                ),
            ),
            is_static=True,
            purity=Purity.PURE,
        ),
        "isInteger": MethodSignature(
            name="isInteger",
            params=(ParamDescriptor("value", T_ANY),),
            return_type=T_BOOL,
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="arg0",
                    narrowed_type=T_NUM.with_refinement(
                        _ref("n", BaseType.NUMBER, "Number.isInteger(n)")
                    ),
                    condition="result === true",
                ),
            ),
            is_static=True,
            purity=Purity.PURE,
        ),
        "isNaN": MethodSignature(
            name="isNaN",
            params=(ParamDescriptor("value", T_ANY),),
            return_type=T_BOOL,
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="arg0",
                    narrowed_type=T_NUM.with_refinement(
                        _ref("n", BaseType.NUMBER, "Number.isNaN(n)")
                    ),
                    condition="result === true",
                    negated_type=T_NUM.with_refinement(NON_NAN),
                ),
            ),
            is_static=True,
            purity=Purity.PURE,
        ),
        "isSafeInteger": MethodSignature(
            name="isSafeInteger",
            params=(ParamDescriptor("value", T_ANY),),
            return_type=T_BOOL,
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="arg0",
                    narrowed_type=T_NUM.with_refinement(SAFE_INT),
                    condition="result === true",
                ),
            ),
            is_static=True,
            purity=Purity.PURE,
        ),
        "parseFloat": MethodSignature(
            name="parseFloat",
            params=(ParamDescriptor("string", T_STR),),
            return_type=T_NUM,
            postconditions=(
                Postcondition(
                    "Result may be NaN for invalid input",
                    "true",  # no strict guarantee
                ),
            ),
            is_static=True,
            purity=Purity.PURE,
        ),
        "parseInt": MethodSignature(
            name="parseInt",
            params=(
                ParamDescriptor("string", T_STR),
                ParamDescriptor("radix", T_NUM.with_refinement(
                    _ref("r", BaseType.NUMBER, "2 <= r <= 36 ∧ Number.isInteger(r)")
                ), optional=True),
            ),
            return_type=T_NUM,
            postconditions=(
                Postcondition(
                    "Result is integer or NaN",
                    "Number.isNaN(result) ∨ Number.isInteger(result)",
                ),
            ),
            is_static=True,
            purity=Purity.PURE,
        ),
    }

    methods = {
        "toFixed": MethodSignature(
            name="toFixed",
            params=(
                ParamDescriptor("fractionDigits", T_NUM.with_refinement(
                    _ref("d", BaseType.NUMBER, "0 <= d <= 100 ∧ Number.isInteger(d)")
                ), optional=True),
            ),
            return_type=T_STR,
            exceptions=(
                ExceptionCondition("RangeError", "fractionDigits < 0 ∨ fractionDigits > 100",
                                   "toFixed() digits argument must be between 0 and 100"),
            ),
            purity=Purity.PURE,
        ),
        "toPrecision": MethodSignature(
            name="toPrecision",
            params=(
                ParamDescriptor("precision", T_NUM.with_refinement(
                    _ref("p", BaseType.NUMBER, "1 <= p <= 100 ∧ Number.isInteger(p)")
                ), optional=True),
            ),
            return_type=T_STR,
            exceptions=(
                ExceptionCondition("RangeError", "precision < 1 ∨ precision > 100",
                                   "toPrecision() argument must be between 1 and 100"),
            ),
            purity=Purity.PURE,
        ),
        "toString": MethodSignature(
            name="toString",
            params=(
                ParamDescriptor("radix", T_NUM.with_refinement(
                    _ref("r", BaseType.NUMBER, "2 <= r <= 36 ∧ Number.isInteger(r)")
                ), optional=True),
            ),
            return_type=T_STR,
            postconditions=(
                Postcondition("Result is non-empty", "result.length > 0"),
            ),
            exceptions=(
                ExceptionCondition("RangeError", "radix < 2 ∨ radix > 36",
                                   "toString() radix must be between 2 and 36"),
            ),
            purity=Purity.PURE,
        ),
    }


# -------------------------------------------------------------------
# Object
# -------------------------------------------------------------------


class ObjectModel(TypeScriptModelBase):
    qualified_name = "Object"
    description = "JavaScript Object static methods with refinement types."

    static_methods = {
        "assign": MethodSignature(
            name="assign",
            type_params=(TypeParam("T"),),
            params=(
                ParamDescriptor("target", T_OBJ),
                ParamDescriptor("sources", T_OBJ, rest=True),
            ),
            return_type=T_OBJ,
            postconditions=(
                Postcondition("Returns target", "result === target"),
            ),
            is_static=True,
            purity=Purity.WRITES,
        ),
        "create": MethodSignature(
            name="create",
            params=(
                ParamDescriptor("proto", _nullable(T_OBJ)),
                ParamDescriptor("propertiesObject", T_OBJ, optional=True),
            ),
            return_type=T_OBJ,
            is_static=True,
            purity=Purity.PURE,
        ),
        "defineProperty": MethodSignature(
            name="defineProperty",
            params=(
                ParamDescriptor("obj", T_OBJ),
                ParamDescriptor("prop", _union(T_STR, T_SYM)),
                ParamDescriptor("descriptor", T_OBJ),
            ),
            return_type=T_OBJ,
            postconditions=(Postcondition("Returns obj", "result === obj"),),
            exceptions=(
                ExceptionCondition("TypeError", "!Object.isExtensible(obj)",
                                   "Cannot define property on non-extensible object"),
            ),
            is_static=True,
            purity=Purity.WRITES,
        ),
        "entries": MethodSignature(
            name="entries",
            type_params=(TypeParam("T"),),
            params=(ParamDescriptor("obj", T_OBJ),),
            return_type=_t(BaseType.ARRAY, _t(BaseType.TUPLE)),
            is_static=True,
            purity=Purity.PURE,
        ),
        "freeze": MethodSignature(
            name="freeze",
            type_params=(TypeParam("T"),),
            params=(ParamDescriptor("obj", T_OBJ),),
            return_type=T_OBJ,  # Readonly<T>
            postconditions=(
                Postcondition("Object is frozen", "Object.isFrozen(result) === true"),
                Postcondition("Returns obj", "result === obj"),
            ),
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="arg0",
                    narrowed_type=T_OBJ,  # Readonly<T>
                    condition="true",
                ),
            ),
            is_static=True,
            purity=Purity.WRITES,
        ),
        "fromEntries": MethodSignature(
            name="fromEntries",
            params=(ParamDescriptor("entries", _t(BaseType.ITERABLE)),),
            return_type=T_OBJ,
            is_static=True,
            purity=Purity.PURE,
        ),
        "getOwnPropertyNames": MethodSignature(
            name="getOwnPropertyNames",
            params=(ParamDescriptor("obj", T_OBJ),),
            return_type=_t(BaseType.ARRAY, T_STR),
            is_static=True,
            purity=Purity.PURE,
        ),
        "getPrototypeOf": MethodSignature(
            name="getPrototypeOf",
            params=(ParamDescriptor("obj", T_OBJ),),
            return_type=_nullable(T_OBJ),
            is_static=True,
            purity=Purity.PURE,
        ),
        "hasOwn": MethodSignature(
            name="hasOwn",
            params=(
                ParamDescriptor("obj", T_OBJ),
                ParamDescriptor("prop", _union(T_STR, T_SYM)),
            ),
            return_type=T_BOOL,
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="arg0",
                    narrowed_type=T_OBJ,
                    condition="result === true",
                ),
            ),
            is_static=True,
            purity=Purity.PURE,
        ),
        "is": MethodSignature(
            name="is",
            params=(
                ParamDescriptor("value1", T_ANY),
                ParamDescriptor("value2", T_ANY),
            ),
            return_type=T_BOOL,
            is_static=True,
            purity=Purity.PURE,
        ),
        "keys": MethodSignature(
            name="keys",
            params=(ParamDescriptor("obj", T_OBJ),),
            return_type=_t(BaseType.ARRAY, T_STR),
            is_static=True,
            purity=Purity.PURE,
        ),
        "values": MethodSignature(
            name="values",
            params=(ParamDescriptor("obj", T_OBJ),),
            return_type=_t(BaseType.ARRAY, T_ANY),
            is_static=True,
            purity=Purity.PURE,
        ),
        "seal": MethodSignature(
            name="seal",
            type_params=(TypeParam("T"),),
            params=(ParamDescriptor("obj", T_OBJ),),
            return_type=T_OBJ,
            postconditions=(
                Postcondition("Object is sealed", "Object.isSealed(result) === true"),
            ),
            is_static=True,
            purity=Purity.WRITES,
        ),
        "isFrozen": MethodSignature(
            name="isFrozen",
            params=(ParamDescriptor("obj", T_OBJ),),
            return_type=T_BOOL,
            is_static=True,
            purity=Purity.PURE,
        ),
        "isSealed": MethodSignature(
            name="isSealed",
            params=(ParamDescriptor("obj", T_OBJ),),
            return_type=T_BOOL,
            is_static=True,
            purity=Purity.PURE,
        ),
    }


# -------------------------------------------------------------------
# Symbol
# -------------------------------------------------------------------


class SymbolModel(TypeScriptModelBase):
    qualified_name = "Symbol"

    static_methods = {
        "for": MethodSignature(
            name="for",
            params=(ParamDescriptor("key", T_STR),),
            return_type=T_SYM,
            postconditions=(
                Postcondition("Same key produces same symbol",
                              "Symbol.for(key) === Symbol.for(key)"),
            ),
            is_static=True,
            purity=Purity.READS,
        ),
        "keyFor": MethodSignature(
            name="keyFor",
            params=(ParamDescriptor("sym", T_SYM),),
            return_type=_undefinable(T_STR),
            is_static=True,
            purity=Purity.READS,
        ),
    }

    methods = {
        "toString": MethodSignature(
            name="toString",
            params=(),
            return_type=T_STR,
            postconditions=(
                Postcondition("Starts with 'Symbol('", "result.startsWith('Symbol(')"),
            ),
            purity=Purity.PURE,
        ),
        "description": MethodSignature(
            name="description",
            params=(),
            return_type=_undefinable(T_STR),
            purity=Purity.PURE,
        ),
    }


# -------------------------------------------------------------------
# BigInt
# -------------------------------------------------------------------


class BigIntModel(TypeScriptModelBase):
    qualified_name = "BigInt"

    methods = {
        "toString": MethodSignature(
            name="toString",
            params=(
                ParamDescriptor("radix", T_NUM.with_refinement(
                    _ref("r", BaseType.NUMBER, "2 <= r <= 36 ∧ Number.isInteger(r)")
                ), optional=True),
            ),
            return_type=T_STR,
            exceptions=(
                ExceptionCondition("RangeError", "radix < 2 ∨ radix > 36"),
            ),
            purity=Purity.PURE,
        ),
        "toLocaleString": MethodSignature(
            name="toLocaleString",
            params=(
                ParamDescriptor("locales", T_STR, optional=True),
                ParamDescriptor("options", T_OBJ, optional=True),
            ),
            return_type=T_STR,
            purity=Purity.PURE,
        ),
        "valueOf": MethodSignature(
            name="valueOf",
            params=(),
            return_type=T_BIGINT,
            purity=Purity.PURE,
        ),
    }

    static_methods = {
        "asIntN": MethodSignature(
            name="asIntN",
            params=(
                ParamDescriptor("bits", T_NUM.with_refinement(NON_NEG_INT)),
                ParamDescriptor("bigint", T_BIGINT),
            ),
            return_type=T_BIGINT,
            is_static=True,
            purity=Purity.PURE,
        ),
        "asUintN": MethodSignature(
            name="asUintN",
            params=(
                ParamDescriptor("bits", T_NUM.with_refinement(NON_NEG_INT)),
                ParamDescriptor("bigint", T_BIGINT),
            ),
            return_type=T_BIGINT,
            postconditions=(
                Postcondition("Result is non-negative", "result >= 0n"),
            ),
            is_static=True,
            purity=Purity.PURE,
        ),
    }


# -------------------------------------------------------------------
# RegExp
# -------------------------------------------------------------------


class RegExpModel(TypeScriptModelBase):
    qualified_name = "RegExp"

    properties = {
        "source": PropertyDescriptor("source", T_STR, readonly=True),
        "flags": PropertyDescriptor("flags", T_STR, readonly=True),
        "global": PropertyDescriptor("global", T_BOOL, readonly=True),
        "ignoreCase": PropertyDescriptor("ignoreCase", T_BOOL, readonly=True),
        "multiline": PropertyDescriptor("multiline", T_BOOL, readonly=True),
        "dotAll": PropertyDescriptor("dotAll", T_BOOL, readonly=True),
        "unicode": PropertyDescriptor("unicode", T_BOOL, readonly=True),
        "sticky": PropertyDescriptor("sticky", T_BOOL, readonly=True),
        "lastIndex": PropertyDescriptor("lastIndex", T_NUM.with_refinement(NON_NEG_INT), readonly=False),
    }

    methods = {
        "constructor": MethodSignature(
            name="constructor",
            params=(
                ParamDescriptor("pattern", _union(T_STR, T_REGEXP)),
                ParamDescriptor("flags", T_STR, optional=True),
            ),
            return_type=T_REGEXP,
            exceptions=(
                ExceptionCondition("SyntaxError", "pattern is invalid regular expression"),
            ),
            purity=Purity.PURE,
        ),
        "test": MethodSignature(
            name="test",
            params=(ParamDescriptor("string", T_STR),),
            return_type=T_BOOL,
            purity=Purity.READS,
            description="Test if pattern matches string.",
        ),
        "exec": MethodSignature(
            name="exec",
            params=(ParamDescriptor("string", T_STR),),
            return_type=_nullable(_t(BaseType.ARRAY, T_STR)),
            purity=Purity.READS,
            description="Execute search; returns match array or null.",
        ),
    }


# -------------------------------------------------------------------
# Date
# -------------------------------------------------------------------


class DateModel(TypeScriptModelBase):
    qualified_name = "Date"

    methods = {
        "constructor": MethodSignature(
            name="constructor",
            params=(
                ParamDescriptor("value", _union(T_STR, T_NUM, T_DATE), optional=True),
            ),
            return_type=T_DATE,
            purity=Purity.IMPURE,
            description="Create Date; impure when called with no args (uses current time).",
        ),
        "getTime": MethodSignature(
            name="getTime",
            params=(),
            return_type=T_NUM,
            postconditions=(
                Postcondition("Returns milliseconds since epoch", "Number.isInteger(result)"),
            ),
            purity=Purity.PURE,
        ),
        "getFullYear": MethodSignature(
            name="getFullYear", params=(), return_type=T_NUM,
            postconditions=(Postcondition("Four-digit year", "Number.isInteger(result)"),),
            purity=Purity.PURE,
        ),
        "getMonth": MethodSignature(
            name="getMonth", params=(), return_type=T_NUM.with_refinement(
                _ref("m", BaseType.NUMBER, "0 <= m <= 11 ∧ Number.isInteger(m)")
            ),
            purity=Purity.PURE,
        ),
        "getDate": MethodSignature(
            name="getDate", params=(), return_type=T_NUM.with_refinement(
                _ref("d", BaseType.NUMBER, "1 <= d <= 31 ∧ Number.isInteger(d)")
            ),
            purity=Purity.PURE,
        ),
        "getDay": MethodSignature(
            name="getDay", params=(), return_type=T_NUM.with_refinement(
                _ref("d", BaseType.NUMBER, "0 <= d <= 6 ∧ Number.isInteger(d)")
            ),
            purity=Purity.PURE,
        ),
        "getHours": MethodSignature(
            name="getHours", params=(), return_type=T_NUM.with_refinement(
                _ref("h", BaseType.NUMBER, "0 <= h <= 23 ∧ Number.isInteger(h)")
            ),
            purity=Purity.PURE,
        ),
        "getMinutes": MethodSignature(
            name="getMinutes", params=(), return_type=T_NUM.with_refinement(
                _ref("m", BaseType.NUMBER, "0 <= m <= 59 ∧ Number.isInteger(m)")
            ),
            purity=Purity.PURE,
        ),
        "getSeconds": MethodSignature(
            name="getSeconds", params=(), return_type=T_NUM.with_refinement(
                _ref("s", BaseType.NUMBER, "0 <= s <= 59 ∧ Number.isInteger(s)")
            ),
            purity=Purity.PURE,
        ),
        "getMilliseconds": MethodSignature(
            name="getMilliseconds", params=(), return_type=T_NUM.with_refinement(
                _ref("ms", BaseType.NUMBER, "0 <= ms <= 999 ∧ Number.isInteger(ms)")
            ),
            purity=Purity.PURE,
        ),
        "toISOString": MethodSignature(
            name="toISOString", params=(), return_type=T_STR,
            postconditions=(Postcondition("ISO 8601 format", "result.length >= 24"),),
            exceptions=(ExceptionCondition("RangeError", "this is Invalid Date"),),
            purity=Purity.PURE,
        ),
        "toJSON": MethodSignature(
            name="toJSON", params=(), return_type=_nullable(T_STR),
            purity=Purity.PURE,
        ),
        "valueOf": MethodSignature(
            name="valueOf", params=(), return_type=T_NUM, purity=Purity.PURE,
        ),
    }

    static_methods = {
        "now": MethodSignature(
            name="now", params=(), return_type=T_NUM.with_refinement(NON_NEG_INT),
            postconditions=(Postcondition("Returns milliseconds since epoch", "result >= 0"),),
            is_static=True,
            purity=Purity.IMPURE,
        ),
        "parse": MethodSignature(
            name="parse",
            params=(ParamDescriptor("dateString", T_STR),),
            return_type=T_NUM,
            is_static=True,
            purity=Purity.PURE,
        ),
        "UTC": MethodSignature(
            name="UTC",
            params=(
                ParamDescriptor("year", T_NUM),
                ParamDescriptor("month", T_NUM, optional=True),
                ParamDescriptor("date", T_NUM, optional=True),
                ParamDescriptor("hours", T_NUM, optional=True),
                ParamDescriptor("minutes", T_NUM, optional=True),
                ParamDescriptor("seconds", T_NUM, optional=True),
                ParamDescriptor("ms", T_NUM, optional=True),
            ),
            return_type=T_NUM,
            is_static=True,
            purity=Purity.PURE,
        ),
    }


# -------------------------------------------------------------------
# Error types
# -------------------------------------------------------------------


def _make_error_model(name: str) -> type:
    """Factory for Error subclass models."""
    cls = type(
        f"{name}Model",
        (TypeScriptModelBase,),
        {
            "qualified_name": name,
            "properties": {
                "message": PropertyDescriptor("message", T_STR),
                "name": PropertyDescriptor("name", T_STR, readonly=True),
                "stack": PropertyDescriptor("stack", _undefinable(T_STR), readonly=True),
                "cause": PropertyDescriptor("cause", _undefinable(T_ANY)),
            },
            "methods": {
                "constructor": MethodSignature(
                    name="constructor",
                    params=(
                        ParamDescriptor("message", T_STR, optional=True),
                        ParamDescriptor("options", T_OBJ, optional=True),
                    ),
                    return_type=T_ERR,
                    postconditions=(
                        Postcondition(
                            f"name is '{name}'",
                            f"result.name === '{name}'",
                        ),
                    ),
                    purity=Purity.PURE,
                ),
            },
        },
    )
    return cls


ErrorModel = _make_error_model("Error")
TypeErrorModel = _make_error_model("TypeError")
RangeErrorModel = _make_error_model("RangeError")
ReferenceErrorModel = _make_error_model("ReferenceError")
SyntaxErrorModel = _make_error_model("SyntaxError")


# -------------------------------------------------------------------
# JSON
# -------------------------------------------------------------------


class JSONModel(TypeScriptModelBase):
    qualified_name = "JSON"

    static_methods = {
        "parse": MethodSignature(
            name="parse",
            params=(
                ParamDescriptor("text", T_STR),
                ParamDescriptor("reviver", _t(BaseType.FUNCTION), optional=True),
            ),
            return_type=T_ANY,
            exceptions=(
                ExceptionCondition("SyntaxError", "text is not valid JSON",
                                   "Unexpected token in JSON"),
            ),
            is_static=True,
            purity=Purity.PURE,
        ),
        "stringify": MethodSignature(
            name="stringify",
            params=(
                ParamDescriptor("value", T_ANY),
                ParamDescriptor("replacer",
                                _union(_nullable(_t(BaseType.FUNCTION)),
                                       _nullable(_t(BaseType.ARRAY, T_STR))),
                                optional=True),
                ParamDescriptor("space", _union(T_STR, T_NUM), optional=True),
            ),
            return_type=_undefinable(T_STR),
            postconditions=(
                Postcondition(
                    "Returns undefined for unsupported values",
                    "typeof value === 'function' ∨ typeof value === 'symbol' ⇒ result === undefined",
                ),
            ),
            is_static=True,
            purity=Purity.PURE,
        ),
    }


# -------------------------------------------------------------------
# Math
# -------------------------------------------------------------------


class MathModel(TypeScriptModelBase):
    qualified_name = "Math"
    description = "JavaScript Math object – all methods are static and pure."

    properties = {
        "PI": PropertyDescriptor("PI", RefinedType(BaseType.NUMBER, literal_value=3.141592653589793), readonly=True),
        "E": PropertyDescriptor("E", RefinedType(BaseType.NUMBER, literal_value=2.718281828459045), readonly=True),
        "LN2": PropertyDescriptor("LN2", RefinedType(BaseType.NUMBER, literal_value=0.6931471805599453), readonly=True),
        "LN10": PropertyDescriptor("LN10", RefinedType(BaseType.NUMBER, literal_value=2.302585092994046), readonly=True),
        "SQRT2": PropertyDescriptor("SQRT2", RefinedType(BaseType.NUMBER, literal_value=1.4142135623730951), readonly=True),
    }

    static_methods = {
        "abs": MethodSignature(
            name="abs",
            params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM.with_refinement(_ref("r", BaseType.NUMBER, "r >= 0")),
            postconditions=(
                Postcondition("Result is non-negative", "result >= 0"),
                Postcondition("Result equals magnitude", "result === (x >= 0 ? x : -x)"),
            ),
            refinement_transfers=(
                RefinementTransfer("x", "return", "return >= 0"),
            ),
            is_static=True, purity=Purity.PURE,
        ),
        "ceil": MethodSignature(
            name="ceil",
            params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM.with_refinement(_ref("r", BaseType.NUMBER, "Number.isInteger(r) ∧ r >= x")),
            postconditions=(
                Postcondition("Result >= x", "result >= x"),
                Postcondition("Result is integer", "Number.isInteger(result)"),
            ),
            is_static=True, purity=Purity.PURE,
        ),
        "floor": MethodSignature(
            name="floor",
            params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM.with_refinement(_ref("r", BaseType.NUMBER, "Number.isInteger(r) ∧ r <= x")),
            postconditions=(
                Postcondition("Result <= x", "result <= x"),
                Postcondition("Result is integer", "Number.isInteger(result)"),
            ),
            is_static=True, purity=Purity.PURE,
        ),
        "round": MethodSignature(
            name="round",
            params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM.with_refinement(_ref("r", BaseType.NUMBER, "Number.isInteger(r)")),
            is_static=True, purity=Purity.PURE,
        ),
        "max": MethodSignature(
            name="max",
            params=(ParamDescriptor("values", T_NUM, rest=True),),
            return_type=T_NUM,
            postconditions=(
                Postcondition("Returns -Infinity for no args",
                              "values.length === 0 ⇒ result === -Infinity"),
                Postcondition("Result >= all args", "∀v ∈ values: result >= v"),
            ),
            is_static=True, purity=Purity.PURE,
        ),
        "min": MethodSignature(
            name="min",
            params=(ParamDescriptor("values", T_NUM, rest=True),),
            return_type=T_NUM,
            postconditions=(
                Postcondition("Returns Infinity for no args",
                              "values.length === 0 ⇒ result === Infinity"),
                Postcondition("Result <= all args", "∀v ∈ values: result <= v"),
            ),
            is_static=True, purity=Purity.PURE,
        ),
        "pow": MethodSignature(
            name="pow",
            params=(ParamDescriptor("base", T_NUM), ParamDescriptor("exponent", T_NUM)),
            return_type=T_NUM,
            is_static=True, purity=Purity.PURE,
        ),
        "sqrt": MethodSignature(
            name="sqrt",
            params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM,
            preconditions=(
                Precondition("Argument should be non-negative for real result", "x >= 0"),
            ),
            postconditions=(
                Postcondition("Result is NaN for negative input",
                              "x < 0 ⇒ Number.isNaN(result)"),
                Postcondition("Non-negative result for non-negative input",
                              "x >= 0 ⇒ result >= 0"),
            ),
            is_static=True, purity=Purity.PURE,
        ),
        "cbrt": MethodSignature(
            name="cbrt",
            params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM,
            is_static=True, purity=Purity.PURE,
        ),
        "log": MethodSignature(
            name="log",
            params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM,
            postconditions=(
                Postcondition("NaN for negative", "x < 0 ⇒ Number.isNaN(result)"),
                Postcondition("-Infinity for 0", "x === 0 ⇒ result === -Infinity"),
            ),
            is_static=True, purity=Purity.PURE,
        ),
        "log2": MethodSignature(
            name="log2",
            params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM,
            is_static=True, purity=Purity.PURE,
        ),
        "log10": MethodSignature(
            name="log10",
            params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM,
            is_static=True, purity=Purity.PURE,
        ),
        "exp": MethodSignature(
            name="exp",
            params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM.with_refinement(_ref("r", BaseType.NUMBER, "r > 0")),
            postconditions=(Postcondition("Result is positive", "result > 0"),),
            is_static=True, purity=Purity.PURE,
        ),
        "sin": MethodSignature(
            name="sin", params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM.with_refinement(_ref("r", BaseType.NUMBER, "-1 <= r <= 1")),
            is_static=True, purity=Purity.PURE,
        ),
        "cos": MethodSignature(
            name="cos", params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM.with_refinement(_ref("r", BaseType.NUMBER, "-1 <= r <= 1")),
            is_static=True, purity=Purity.PURE,
        ),
        "tan": MethodSignature(
            name="tan", params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM,
            is_static=True, purity=Purity.PURE,
        ),
        "random": MethodSignature(
            name="random", params=(),
            return_type=T_NUM.with_refinement(_ref("r", BaseType.NUMBER, "0 <= r < 1")),
            postconditions=(
                Postcondition("Result in [0, 1)", "0 <= result ∧ result < 1"),
            ),
            is_static=True, purity=Purity.IMPURE,
        ),
        "sign": MethodSignature(
            name="sign", params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM.with_refinement(_ref("r", BaseType.NUMBER, "r ∈ {-1, 0, 1, NaN}")),
            is_static=True, purity=Purity.PURE,
        ),
        "trunc": MethodSignature(
            name="trunc", params=(ParamDescriptor("x", T_NUM),),
            return_type=T_NUM.with_refinement(_ref("r", BaseType.NUMBER, "Number.isInteger(r)")),
            is_static=True, purity=Purity.PURE,
        ),
    }


# -------------------------------------------------------------------
# console
# -------------------------------------------------------------------


class ConsoleModel(TypeScriptModelBase):
    qualified_name = "console"
    description = "Console logging methods – all impure (I/O)."

    methods = {
        "log": MethodSignature(
            name="log",
            params=(ParamDescriptor("data", T_ANY, rest=True),),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
        "error": MethodSignature(
            name="error",
            params=(ParamDescriptor("data", T_ANY, rest=True),),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
        "warn": MethodSignature(
            name="warn",
            params=(ParamDescriptor("data", T_ANY, rest=True),),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
        "info": MethodSignature(
            name="info",
            params=(ParamDescriptor("data", T_ANY, rest=True),),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
        "debug": MethodSignature(
            name="debug",
            params=(ParamDescriptor("data", T_ANY, rest=True),),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
        "table": MethodSignature(
            name="table",
            params=(
                ParamDescriptor("tabularData", T_ANY),
                ParamDescriptor("properties", _t(BaseType.ARRAY, T_STR), optional=True),
            ),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
        "time": MethodSignature(
            name="time",
            params=(ParamDescriptor("label", T_STR, optional=True, default_value='"default"'),),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
        "timeEnd": MethodSignature(
            name="timeEnd",
            params=(ParamDescriptor("label", T_STR, optional=True, default_value='"default"'),),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
        "assert": MethodSignature(
            name="assert",
            params=(
                ParamDescriptor("condition", T_BOOL, optional=True),
                ParamDescriptor("data", T_ANY, rest=True),
            ),
            return_type=T_VOID,
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="arg0",
                    narrowed_type=RefinedType(BaseType.BOOLEAN, literal_value=True),
                    condition="no assertion error",
                ),
            ),
            purity=Purity.IMPURE,
        ),
        "count": MethodSignature(
            name="count",
            params=(ParamDescriptor("label", T_STR, optional=True, default_value='"default"'),),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
        "dir": MethodSignature(
            name="dir",
            params=(
                ParamDescriptor("item", T_ANY),
                ParamDescriptor("options", T_OBJ, optional=True),
            ),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
        "trace": MethodSignature(
            name="trace",
            params=(ParamDescriptor("data", T_ANY, rest=True),),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
        "group": MethodSignature(
            name="group",
            params=(ParamDescriptor("label", T_ANY, rest=True),),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
        "groupEnd": MethodSignature(
            name="groupEnd",
            params=(),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
        "clear": MethodSignature(
            name="clear",
            params=(),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
    }


# ===================================================================
# NODE.JS MODULES
# ===================================================================


# -------------------------------------------------------------------
# fs (file system)
# -------------------------------------------------------------------


class FsModel(TypeScriptModelBase):
    qualified_name = "fs"
    description = "Node.js fs module with refinement types."

    methods = {
        "readFileSync": MethodSignature(
            name="readFileSync",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("options", _union(T_STR, T_OBJ), optional=True),
            ),
            return_type=_union(T_STR, T_BUF),
            preconditions=(
                Precondition("Path must be non-empty string or Buffer",
                             "typeof path === 'string' ⇒ path.length > 0"),
            ),
            exceptions=(
                ExceptionCondition("Error", "file does not exist", "ENOENT: no such file or directory"),
                ExceptionCondition("Error", "permission denied", "EACCES: permission denied"),
            ),
            purity=Purity.READS,
        ),
        "writeFileSync": MethodSignature(
            name="writeFileSync",
            params=(
                ParamDescriptor("file", _union(T_STR, T_BUF, T_URL, T_NUM)),
                ParamDescriptor("data", _union(T_STR, T_BUF)),
                ParamDescriptor("options", _union(T_STR, T_OBJ), optional=True),
            ),
            return_type=T_VOID,
            exceptions=(
                ExceptionCondition("Error", "permission denied", "EACCES"),
                ExceptionCondition("Error", "directory does not exist", "ENOENT"),
            ),
            purity=Purity.WRITES,
        ),
        "readFile": MethodSignature(
            name="readFile",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("options", _union(T_STR, T_OBJ), optional=True),
                ParamDescriptor("callback", _t(BaseType.FUNCTION),
                                description="(err: NodeJS.ErrnoException | null, data: Buffer | string) => void"),
            ),
            return_type=T_VOID,
            is_async=True,
            purity=Purity.READS,
        ),
        "writeFile": MethodSignature(
            name="writeFile",
            params=(
                ParamDescriptor("file", _union(T_STR, T_BUF, T_URL, T_NUM)),
                ParamDescriptor("data", _union(T_STR, T_BUF)),
                ParamDescriptor("options", _union(T_STR, T_OBJ), optional=True),
                ParamDescriptor("callback", _t(BaseType.FUNCTION),
                                description="(err: NodeJS.ErrnoException | null) => void"),
            ),
            return_type=T_VOID,
            is_async=True,
            purity=Purity.WRITES,
        ),
        "existsSync": MethodSignature(
            name="existsSync",
            params=(ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),),
            return_type=T_BOOL,
            purity=Purity.READS,
        ),
        "mkdirSync": MethodSignature(
            name="mkdirSync",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("options", _union(T_NUM, T_OBJ), optional=True),
            ),
            return_type=_undefinable(T_STR),
            exceptions=(
                ExceptionCondition("Error", "EEXIST and not recursive", "EEXIST: file already exists"),
            ),
            purity=Purity.WRITES,
        ),
        "readdirSync": MethodSignature(
            name="readdirSync",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("options", _union(T_STR, T_OBJ), optional=True),
            ),
            return_type=_t(BaseType.ARRAY, _union(T_STR, T_BUF)),
            postconditions=(
                Postcondition("Result length >= 0", "result.length >= 0"),
            ),
            exceptions=(
                ExceptionCondition("Error", "path is not a directory", "ENOTDIR"),
                ExceptionCondition("Error", "path does not exist", "ENOENT"),
            ),
            purity=Purity.READS,
        ),
        "statSync": MethodSignature(
            name="statSync",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("options", T_OBJ, optional=True),
            ),
            return_type=T_OBJ,  # fs.Stats
            exceptions=(
                ExceptionCondition("Error", "path does not exist", "ENOENT"),
            ),
            purity=Purity.READS,
        ),
        "unlinkSync": MethodSignature(
            name="unlinkSync",
            params=(ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),),
            return_type=T_VOID,
            exceptions=(
                ExceptionCondition("Error", "file does not exist", "ENOENT"),
                ExceptionCondition("Error", "permission denied", "EACCES"),
            ),
            purity=Purity.WRITES,
        ),
        "renameSync": MethodSignature(
            name="renameSync",
            params=(
                ParamDescriptor("oldPath", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("newPath", _union(T_STR, T_BUF, T_URL)),
            ),
            return_type=T_VOID,
            purity=Purity.WRITES,
        ),
        "copyFileSync": MethodSignature(
            name="copyFileSync",
            params=(
                ParamDescriptor("src", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("dest", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("mode", T_NUM, optional=True),
            ),
            return_type=T_VOID,
            purity=Purity.WRITES,
        ),
        "appendFileSync": MethodSignature(
            name="appendFileSync",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL, T_NUM)),
                ParamDescriptor("data", _union(T_STR, T_BUF)),
                ParamDescriptor("options", _union(T_STR, T_OBJ), optional=True),
            ),
            return_type=T_VOID,
            purity=Purity.WRITES,
        ),
        "accessSync": MethodSignature(
            name="accessSync",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("mode", T_NUM, optional=True),
            ),
            return_type=T_VOID,
            exceptions=(
                ExceptionCondition("Error", "access denied or file not found", "ENOENT | EACCES"),
            ),
            purity=Purity.READS,
        ),
        "chmodSync": MethodSignature(
            name="chmodSync",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("mode", _union(T_STR, T_NUM)),
            ),
            return_type=T_VOID,
            purity=Purity.WRITES,
        ),
        "createReadStream": MethodSignature(
            name="createReadStream",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("options", _union(T_STR, T_OBJ), optional=True),
            ),
            return_type=_t(BaseType.STREAM),
            purity=Purity.READS,
        ),
        "createWriteStream": MethodSignature(
            name="createWriteStream",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("options", _union(T_STR, T_OBJ), optional=True),
            ),
            return_type=_t(BaseType.STREAM),
            purity=Purity.WRITES,
        ),
        "watch": MethodSignature(
            name="watch",
            params=(
                ParamDescriptor("filename", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("options", _union(T_STR, T_OBJ), optional=True),
                ParamDescriptor("listener", _t(BaseType.FUNCTION), optional=True),
            ),
            return_type=T_OBJ,  # fs.FSWatcher
            purity=Purity.READS,
        ),
    }

    static_methods = {
        "promises.readFile": MethodSignature(
            name="promises.readFile",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("options", _union(T_STR, T_OBJ), optional=True),
            ),
            return_type=_t(BaseType.PROMISE, _union(T_STR, T_BUF)),
            is_static=True, is_async=True, purity=Purity.READS,
        ),
        "promises.writeFile": MethodSignature(
            name="promises.writeFile",
            params=(
                ParamDescriptor("file", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("data", _union(T_STR, T_BUF)),
                ParamDescriptor("options", _union(T_STR, T_OBJ), optional=True),
            ),
            return_type=_t(BaseType.PROMISE, T_VOID),
            is_static=True, is_async=True, purity=Purity.WRITES,
        ),
        "promises.readdir": MethodSignature(
            name="promises.readdir",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("options", _union(T_STR, T_OBJ), optional=True),
            ),
            return_type=_t(BaseType.PROMISE, _t(BaseType.ARRAY, T_STR)),
            is_static=True, is_async=True, purity=Purity.READS,
        ),
        "promises.mkdir": MethodSignature(
            name="promises.mkdir",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("options", _union(T_NUM, T_OBJ), optional=True),
            ),
            return_type=_t(BaseType.PROMISE, _undefinable(T_STR)),
            is_static=True, is_async=True, purity=Purity.WRITES,
        ),
        "promises.stat": MethodSignature(
            name="promises.stat",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("options", T_OBJ, optional=True),
            ),
            return_type=_t(BaseType.PROMISE, T_OBJ),
            is_static=True, is_async=True, purity=Purity.READS,
        ),
        "promises.unlink": MethodSignature(
            name="promises.unlink",
            params=(ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),),
            return_type=_t(BaseType.PROMISE, T_VOID),
            is_static=True, is_async=True, purity=Purity.WRITES,
        ),
        "promises.rename": MethodSignature(
            name="promises.rename",
            params=(
                ParamDescriptor("oldPath", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("newPath", _union(T_STR, T_BUF, T_URL)),
            ),
            return_type=_t(BaseType.PROMISE, T_VOID),
            is_static=True, is_async=True, purity=Purity.WRITES,
        ),
        "promises.access": MethodSignature(
            name="promises.access",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("mode", T_NUM, optional=True),
            ),
            return_type=_t(BaseType.PROMISE, T_VOID),
            is_static=True, is_async=True, purity=Purity.READS,
        ),
        "promises.chmod": MethodSignature(
            name="promises.chmod",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("mode", _union(T_STR, T_NUM)),
            ),
            return_type=_t(BaseType.PROMISE, T_VOID),
            is_static=True, is_async=True, purity=Purity.WRITES,
        ),
        "promises.copyFile": MethodSignature(
            name="promises.copyFile",
            params=(
                ParamDescriptor("src", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("dest", _union(T_STR, T_BUF, T_URL)),
                ParamDescriptor("mode", T_NUM, optional=True),
            ),
            return_type=_t(BaseType.PROMISE, T_VOID),
            is_static=True, is_async=True, purity=Purity.WRITES,
        ),
        "promises.appendFile": MethodSignature(
            name="promises.appendFile",
            params=(
                ParamDescriptor("path", _union(T_STR, T_BUF, T_URL, T_NUM)),
                ParamDescriptor("data", _union(T_STR, T_BUF)),
                ParamDescriptor("options", _union(T_STR, T_OBJ), optional=True),
            ),
            return_type=_t(BaseType.PROMISE, T_VOID),
            is_static=True, is_async=True, purity=Purity.WRITES,
        ),
    }


# -------------------------------------------------------------------
# path
# -------------------------------------------------------------------


class PathModel(TypeScriptModelBase):
    qualified_name = "path"
    description = "Node.js path module."

    properties = {
        "sep": PropertyDescriptor("sep", T_STR, readonly=True,
                                  description="Platform-specific path separator ('/' or '\\\\')."),
        "delimiter": PropertyDescriptor("delimiter", T_STR, readonly=True,
                                        description="Platform-specific path delimiter (':' or ';')."),
        "posix": PropertyDescriptor("posix", T_OBJ, readonly=True),
        "win32": PropertyDescriptor("win32", T_OBJ, readonly=True),
    }

    methods = {
        "join": MethodSignature(
            name="join",
            params=(ParamDescriptor("paths", T_STR, rest=True),),
            return_type=T_STR,
            postconditions=(
                Postcondition("Result is non-empty for non-empty input",
                              "paths.length > 0 ⇒ result.length > 0"),
            ),
            purity=Purity.PURE,
        ),
        "resolve": MethodSignature(
            name="resolve",
            params=(ParamDescriptor("pathSegments", T_STR, rest=True),),
            return_type=T_STR,
            postconditions=(
                Postcondition("Result is absolute path", "path.isAbsolute(result)"),
            ),
            purity=Purity.READS,  # may read cwd
            description="Resolve path segments into absolute path.",
        ),
        "normalize": MethodSignature(
            name="normalize",
            params=(ParamDescriptor("p", T_STR),),
            return_type=T_STR,
            purity=Purity.PURE,
        ),
        "isAbsolute": MethodSignature(
            name="isAbsolute",
            params=(ParamDescriptor("p", T_STR),),
            return_type=T_BOOL,
            purity=Purity.PURE,
        ),
        "relative": MethodSignature(
            name="relative",
            params=(
                ParamDescriptor("from", T_STR),
                ParamDescriptor("to", T_STR),
            ),
            return_type=T_STR,
            purity=Purity.PURE,
        ),
        "dirname": MethodSignature(
            name="dirname",
            params=(ParamDescriptor("p", T_STR),),
            return_type=T_STR,
            postconditions=(
                Postcondition("Result length <= input length", "result.length <= p.length"),
            ),
            purity=Purity.PURE,
        ),
        "basename": MethodSignature(
            name="basename",
            params=(
                ParamDescriptor("p", T_STR),
                ParamDescriptor("ext", T_STR, optional=True),
            ),
            return_type=T_STR,
            purity=Purity.PURE,
        ),
        "extname": MethodSignature(
            name="extname",
            params=(ParamDescriptor("p", T_STR),),
            return_type=T_STR,
            postconditions=(
                Postcondition("Result starts with '.' or is empty",
                              "result === '' ∨ result.startsWith('.')"),
            ),
            purity=Purity.PURE,
        ),
        "parse": MethodSignature(
            name="parse",
            params=(ParamDescriptor("pathString", T_STR),),
            return_type=T_OBJ,  # ParsedPath
            postconditions=(
                Postcondition("Result has root, dir, base, ext, name",
                              "'root' ∈ result ∧ 'dir' ∈ result ∧ 'base' ∈ result"),
            ),
            purity=Purity.PURE,
        ),
        "format": MethodSignature(
            name="format",
            params=(ParamDescriptor("pathObject", T_OBJ),),
            return_type=T_STR,
            purity=Purity.PURE,
        ),
    }


# -------------------------------------------------------------------
# http / https
# -------------------------------------------------------------------


class HttpModel(TypeScriptModelBase):
    qualified_name = "http"
    description = "Node.js http module."

    methods = {
        "createServer": MethodSignature(
            name="createServer",
            params=(
                ParamDescriptor("options", T_OBJ, optional=True),
                ParamDescriptor("requestListener", _t(BaseType.FUNCTION), optional=True,
                                description="(req: IncomingMessage, res: ServerResponse) => void"),
            ),
            return_type=_t(BaseType.SERVER),
            purity=Purity.IMPURE,
        ),
        "request": MethodSignature(
            name="request",
            params=(
                ParamDescriptor("options", _union(T_STR, T_URL, T_OBJ)),
                ParamDescriptor("callback", _t(BaseType.FUNCTION), optional=True,
                                description="(res: IncomingMessage) => void"),
            ),
            return_type=T_OBJ,  # http.ClientRequest
            purity=Purity.IMPURE,
        ),
        "get": MethodSignature(
            name="get",
            params=(
                ParamDescriptor("options", _union(T_STR, T_URL, T_OBJ)),
                ParamDescriptor("callback", _t(BaseType.FUNCTION), optional=True),
            ),
            return_type=T_OBJ,
            purity=Purity.IMPURE,
        ),
    }

    properties = {
        "STATUS_CODES": PropertyDescriptor(
            "STATUS_CODES", T_OBJ, readonly=True,
            description="Mapping of HTTP status codes to reason phrases.",
        ),
    }


class HttpsModel(TypeScriptModelBase):
    qualified_name = "https"
    description = "Node.js https module – same interface as http plus TLS options."

    methods = {
        "createServer": MethodSignature(
            name="createServer",
            params=(
                ParamDescriptor("options", T_OBJ),
                ParamDescriptor("requestListener", _t(BaseType.FUNCTION), optional=True),
            ),
            return_type=_t(BaseType.SERVER),
            purity=Purity.IMPURE,
        ),
        "request": MethodSignature(
            name="request",
            params=(
                ParamDescriptor("options", _union(T_STR, T_URL, T_OBJ)),
                ParamDescriptor("callback", _t(BaseType.FUNCTION), optional=True),
            ),
            return_type=T_OBJ,
            purity=Purity.IMPURE,
        ),
        "get": MethodSignature(
            name="get",
            params=(
                ParamDescriptor("options", _union(T_STR, T_URL, T_OBJ)),
                ParamDescriptor("callback", _t(BaseType.FUNCTION), optional=True),
            ),
            return_type=T_OBJ,
            purity=Purity.IMPURE,
        ),
    }


# -------------------------------------------------------------------
# Buffer
# -------------------------------------------------------------------


class BufferModel(TypeScriptModelBase):
    qualified_name = "Buffer"
    description = "Node.js Buffer."

    properties = {
        "length": PropertyDescriptor(
            "length",
            T_NUM.with_refinement(NON_NEG_INT),
            readonly=True,
            description="Buffer byte length.",
        ),
    }

    static_methods = {
        "alloc": MethodSignature(
            name="alloc",
            params=(
                ParamDescriptor("size", T_NUM.with_refinement(NON_NEG_INT)),
                ParamDescriptor("fill", _union(T_STR, T_BUF, T_NUM), optional=True),
                ParamDescriptor("encoding", T_STR, optional=True),
            ),
            return_type=T_BUF,
            postconditions=(
                Postcondition("Result length equals size", "result.length === size"),
            ),
            exceptions=(
                ExceptionCondition("RangeError", "size < 0 ∨ size > buffer.constants.MAX_LENGTH"),
            ),
            is_static=True, purity=Purity.PURE,
        ),
        "allocUnsafe": MethodSignature(
            name="allocUnsafe",
            params=(ParamDescriptor("size", T_NUM.with_refinement(NON_NEG_INT)),),
            return_type=T_BUF,
            postconditions=(
                Postcondition("Result length equals size", "result.length === size"),
            ),
            is_static=True, purity=Purity.PURE,
            description="Uninitialized buffer – may contain old data.",
        ),
        "from": MethodSignature(
            name="from",
            params=(
                ParamDescriptor("data", _union(T_STR, _t(BaseType.ARRAY, T_NUM), T_BUF, T_OBJ)),
                ParamDescriptor("encodingOrOffset", _union(T_STR, T_NUM), optional=True),
                ParamDescriptor("length", T_NUM, optional=True),
            ),
            return_type=T_BUF,
            is_static=True, purity=Purity.PURE,
        ),
        "concat": MethodSignature(
            name="concat",
            params=(
                ParamDescriptor("list", _t(BaseType.ARRAY, T_BUF)),
                ParamDescriptor("totalLength", T_NUM.with_refinement(NON_NEG_INT), optional=True),
            ),
            return_type=T_BUF,
            postconditions=(
                Postcondition("Result length equals totalLength or sum of parts",
                              "result.length === (totalLength ?? sum(list, b => b.length))"),
            ),
            is_static=True, purity=Purity.PURE,
        ),
        "byteLength": MethodSignature(
            name="byteLength",
            params=(
                ParamDescriptor("string", _union(T_STR, T_BUF)),
                ParamDescriptor("encoding", T_STR, optional=True),
            ),
            return_type=T_NUM.with_refinement(NON_NEG_INT),
            is_static=True, purity=Purity.PURE,
        ),
        "isBuffer": MethodSignature(
            name="isBuffer",
            params=(ParamDescriptor("obj", T_ANY),),
            return_type=T_BOOL,
            narrowing_effects=(
                TypeNarrowingEffect(
                    target="arg0",
                    narrowed_type=T_BUF,
                    condition="result === true",
                ),
            ),
            is_static=True, purity=Purity.PURE,
        ),
        "compare": MethodSignature(
            name="compare",
            params=(ParamDescriptor("buf1", T_BUF), ParamDescriptor("buf2", T_BUF)),
            return_type=T_NUM.with_refinement(_ref("r", BaseType.NUMBER, "r ∈ {-1, 0, 1}")),
            is_static=True, purity=Purity.PURE,
        ),
        "isEncoding": MethodSignature(
            name="isEncoding",
            params=(ParamDescriptor("encoding", T_STR),),
            return_type=T_BOOL,
            is_static=True, purity=Purity.PURE,
        ),
    }


# -------------------------------------------------------------------
# events – EventEmitter
# -------------------------------------------------------------------


class EventEmitterModel(TypeScriptModelBase):
    qualified_name = "events.EventEmitter"
    description = "Node.js EventEmitter."

    methods = {
        "on": MethodSignature(
            name="on",
            params=(
                ParamDescriptor("eventName", _union(T_STR, T_SYM)),
                ParamDescriptor("listener", _t(BaseType.FUNCTION)),
            ),
            return_type=_t(BaseType.EVENT_EMITTER),
            postconditions=(Postcondition("Returns this", "result === this"),),
            purity=Purity.WRITES,
        ),
        "once": MethodSignature(
            name="once",
            params=(
                ParamDescriptor("eventName", _union(T_STR, T_SYM)),
                ParamDescriptor("listener", _t(BaseType.FUNCTION)),
            ),
            return_type=_t(BaseType.EVENT_EMITTER),
            postconditions=(Postcondition("Returns this", "result === this"),),
            purity=Purity.WRITES,
        ),
        "emit": MethodSignature(
            name="emit",
            params=(
                ParamDescriptor("eventName", _union(T_STR, T_SYM)),
                ParamDescriptor("args", T_ANY, rest=True),
            ),
            return_type=T_BOOL,
            postconditions=(
                Postcondition("Returns true if listeners exist",
                              "result === (this.listenerCount(eventName) > 0)"),
            ),
            purity=Purity.IMPURE,
        ),
        "removeListener": MethodSignature(
            name="removeListener",
            params=(
                ParamDescriptor("eventName", _union(T_STR, T_SYM)),
                ParamDescriptor("listener", _t(BaseType.FUNCTION)),
            ),
            return_type=_t(BaseType.EVENT_EMITTER),
            postconditions=(Postcondition("Returns this", "result === this"),),
            purity=Purity.WRITES,
        ),
        "removeAllListeners": MethodSignature(
            name="removeAllListeners",
            params=(
                ParamDescriptor("eventName", _union(T_STR, T_SYM), optional=True),
            ),
            return_type=_t(BaseType.EVENT_EMITTER),
            postconditions=(
                Postcondition("Returns this", "result === this"),
                Postcondition("Listener count is 0 for the event",
                              "eventName !== undefined ⇒ this.listenerCount(eventName) === 0"),
            ),
            purity=Purity.WRITES,
        ),
        "listenerCount": MethodSignature(
            name="listenerCount",
            params=(ParamDescriptor("eventName", _union(T_STR, T_SYM)),),
            return_type=T_NUM.with_refinement(NON_NEG_INT),
            purity=Purity.READS,
        ),
        "setMaxListeners": MethodSignature(
            name="setMaxListeners",
            params=(ParamDescriptor("n", T_NUM.with_refinement(NON_NEG_INT)),),
            return_type=_t(BaseType.EVENT_EMITTER),
            postconditions=(Postcondition("Returns this", "result === this"),),
            purity=Purity.WRITES,
        ),
    }


# -------------------------------------------------------------------
# stream
# -------------------------------------------------------------------


class StreamModel(TypeScriptModelBase):
    qualified_name = "stream"
    description = "Node.js stream module."

    methods = {
        "pipeline": MethodSignature(
            name="pipeline",
            params=(
                ParamDescriptor("streams", _t(BaseType.STREAM), rest=True),
                ParamDescriptor("callback", _t(BaseType.FUNCTION), optional=True,
                                description="(err: Error | null) => void"),
            ),
            return_type=_t(BaseType.STREAM),
            purity=Purity.IMPURE,
        ),
        "finished": MethodSignature(
            name="finished",
            params=(
                ParamDescriptor("stream", _t(BaseType.STREAM)),
                ParamDescriptor("options", T_OBJ, optional=True),
                ParamDescriptor("callback", _t(BaseType.FUNCTION),
                                description="(err?: Error) => void"),
            ),
            return_type=_t(BaseType.FUNCTION),  # cleanup function
            purity=Purity.READS,
        ),
    }

    # Readable, Writable, Transform, Duplex are represented as stream BaseType
    properties = {
        "Readable": PropertyDescriptor("Readable", _t(BaseType.FUNCTION), readonly=True),
        "Writable": PropertyDescriptor("Writable", _t(BaseType.FUNCTION), readonly=True),
        "Transform": PropertyDescriptor("Transform", _t(BaseType.FUNCTION), readonly=True),
        "Duplex": PropertyDescriptor("Duplex", _t(BaseType.FUNCTION), readonly=True),
    }


# -------------------------------------------------------------------
# url
# -------------------------------------------------------------------


class UrlModel(TypeScriptModelBase):
    qualified_name = "url"
    description = "Node.js url module."

    methods = {
        "parse": MethodSignature(
            name="parse",
            params=(
                ParamDescriptor("urlString", T_STR),
                ParamDescriptor("parseQueryString", T_BOOL, optional=True),
                ParamDescriptor("slashesDenoteHost", T_BOOL, optional=True),
            ),
            return_type=T_OBJ,  # Url
            purity=Purity.PURE,
        ),
        "format": MethodSignature(
            name="format",
            params=(ParamDescriptor("urlObject", _union(T_OBJ, T_URL)),),
            return_type=T_STR,
            purity=Purity.PURE,
        ),
        "resolve": MethodSignature(
            name="resolve",
            params=(
                ParamDescriptor("from", T_STR),
                ParamDescriptor("to", T_STR),
            ),
            return_type=T_STR,
            purity=Purity.PURE,
        ),
    }


class URLClassModel(TypeScriptModelBase):
    qualified_name = "URL"
    description = "WHATWG URL class."

    properties = {
        "href": PropertyDescriptor("href", T_STR),
        "origin": PropertyDescriptor("origin", T_STR, readonly=True),
        "protocol": PropertyDescriptor("protocol", T_STR),
        "username": PropertyDescriptor("username", T_STR),
        "password": PropertyDescriptor("password", T_STR),
        "host": PropertyDescriptor("host", T_STR),
        "hostname": PropertyDescriptor("hostname", T_STR),
        "port": PropertyDescriptor("port", T_STR),
        "pathname": PropertyDescriptor("pathname", T_STR),
        "search": PropertyDescriptor("search", T_STR),
        "hash": PropertyDescriptor("hash", T_STR),
        "searchParams": PropertyDescriptor("searchParams", T_OBJ, readonly=True),
    }

    methods = {
        "constructor": MethodSignature(
            name="constructor",
            params=(
                ParamDescriptor("input", T_STR),
                ParamDescriptor("base", _union(T_STR, T_URL), optional=True),
            ),
            return_type=T_URL,
            exceptions=(
                ExceptionCondition("TypeError", "input is not a valid URL",
                                   "Invalid URL"),
            ),
            purity=Purity.PURE,
        ),
        "toString": MethodSignature(
            name="toString", params=(), return_type=T_STR, purity=Purity.PURE,
        ),
        "toJSON": MethodSignature(
            name="toJSON", params=(), return_type=T_STR, purity=Purity.PURE,
        ),
    }


class URLSearchParamsModel(TypeScriptModelBase):
    qualified_name = "URLSearchParams"

    methods = {
        "constructor": MethodSignature(
            name="constructor",
            params=(ParamDescriptor("init", _union(T_STR, T_OBJ, _t(BaseType.ITERABLE)), optional=True),),
            return_type=T_OBJ,
            purity=Purity.PURE,
        ),
        "append": MethodSignature(
            name="append",
            params=(ParamDescriptor("name", T_STR), ParamDescriptor("value", T_STR)),
            return_type=T_VOID, purity=Purity.WRITES,
        ),
        "delete": MethodSignature(
            name="delete",
            params=(ParamDescriptor("name", T_STR),),
            return_type=T_VOID, purity=Purity.WRITES,
        ),
        "get": MethodSignature(
            name="get",
            params=(ParamDescriptor("name", T_STR),),
            return_type=_nullable(T_STR), purity=Purity.READS,
        ),
        "getAll": MethodSignature(
            name="getAll",
            params=(ParamDescriptor("name", T_STR),),
            return_type=_t(BaseType.ARRAY, T_STR), purity=Purity.READS,
        ),
        "has": MethodSignature(
            name="has",
            params=(ParamDescriptor("name", T_STR),),
            return_type=T_BOOL, purity=Purity.READS,
        ),
        "set": MethodSignature(
            name="set",
            params=(ParamDescriptor("name", T_STR), ParamDescriptor("value", T_STR)),
            return_type=T_VOID, purity=Purity.WRITES,
        ),
        "sort": MethodSignature(
            name="sort", params=(), return_type=T_VOID, purity=Purity.WRITES,
        ),
        "toString": MethodSignature(
            name="toString", params=(), return_type=T_STR, purity=Purity.PURE,
        ),
        "entries": MethodSignature(
            name="entries", params=(), return_type=_t(BaseType.ITERATOR), purity=Purity.PURE,
        ),
        "keys": MethodSignature(
            name="keys", params=(), return_type=_t(BaseType.ITERATOR), purity=Purity.PURE,
        ),
        "values": MethodSignature(
            name="values", params=(), return_type=_t(BaseType.ITERATOR), purity=Purity.PURE,
        ),
        "forEach": MethodSignature(
            name="forEach",
            params=(ParamDescriptor("callback", _t(BaseType.FUNCTION)),
                    ParamDescriptor("thisArg", T_ANY, optional=True)),
            return_type=T_VOID, purity=Purity.READS,
        ),
    }

    properties = {
        "size": PropertyDescriptor("size", T_NUM.with_refinement(NON_NEG_INT), readonly=True),
    }


# -------------------------------------------------------------------
# crypto
# -------------------------------------------------------------------


class CryptoModel(TypeScriptModelBase):
    qualified_name = "crypto"
    description = "Node.js crypto module."

    methods = {
        "createHash": MethodSignature(
            name="createHash",
            params=(
                ParamDescriptor("algorithm", T_STR),
                ParamDescriptor("options", T_OBJ, optional=True),
            ),
            return_type=T_OBJ,  # Hash
            preconditions=(
                Precondition("Algorithm must be supported",
                             "algorithm ∈ crypto.getHashes()"),
            ),
            exceptions=(
                ExceptionCondition("Error", "unknown algorithm", "Unknown message digest"),
            ),
            purity=Purity.PURE,
        ),
        "createHmac": MethodSignature(
            name="createHmac",
            params=(
                ParamDescriptor("algorithm", T_STR),
                ParamDescriptor("key", _union(T_STR, T_BUF)),
                ParamDescriptor("options", T_OBJ, optional=True),
            ),
            return_type=T_OBJ,  # Hmac
            purity=Purity.PURE,
        ),
        "createCipheriv": MethodSignature(
            name="createCipheriv",
            params=(
                ParamDescriptor("algorithm", T_STR),
                ParamDescriptor("key", _union(T_STR, T_BUF)),
                ParamDescriptor("iv", _union(T_STR, T_BUF, _nullable(T_BUF))),
                ParamDescriptor("options", T_OBJ, optional=True),
            ),
            return_type=T_OBJ,  # Cipher
            purity=Purity.PURE,
        ),
        "createDecipheriv": MethodSignature(
            name="createDecipheriv",
            params=(
                ParamDescriptor("algorithm", T_STR),
                ParamDescriptor("key", _union(T_STR, T_BUF)),
                ParamDescriptor("iv", _union(T_STR, T_BUF, _nullable(T_BUF))),
                ParamDescriptor("options", T_OBJ, optional=True),
            ),
            return_type=T_OBJ,  # Decipher
            purity=Purity.PURE,
        ),
        "randomBytes": MethodSignature(
            name="randomBytes",
            params=(
                ParamDescriptor("size", T_NUM.with_refinement(NON_NEG_INT)),
                ParamDescriptor("callback", _t(BaseType.FUNCTION), optional=True),
            ),
            return_type=T_BUF,
            postconditions=(
                Postcondition("Result length equals size", "result.length === size"),
            ),
            purity=Purity.IMPURE,
        ),
        "randomUUID": MethodSignature(
            name="randomUUID",
            params=(ParamDescriptor("options", T_OBJ, optional=True),),
            return_type=T_STR.with_refinement(
                _ref("s", BaseType.STRING, "s.length === 36 ∧ s matches UUID v4 format")
            ),
            postconditions=(
                Postcondition("Result is 36 chars", "result.length === 36"),
            ),
            purity=Purity.IMPURE,
        ),
        "pbkdf2": MethodSignature(
            name="pbkdf2",
            params=(
                ParamDescriptor("password", _union(T_STR, T_BUF)),
                ParamDescriptor("salt", _union(T_STR, T_BUF)),
                ParamDescriptor("iterations", T_NUM.with_refinement(POS_INT)),
                ParamDescriptor("keylen", T_NUM.with_refinement(NON_NEG_INT)),
                ParamDescriptor("digest", T_STR),
                ParamDescriptor("callback", _t(BaseType.FUNCTION),
                                description="(err: Error | null, derivedKey: Buffer) => void"),
            ),
            return_type=T_VOID,
            is_async=True,
            purity=Purity.IMPURE,
        ),
        "scrypt": MethodSignature(
            name="scrypt",
            params=(
                ParamDescriptor("password", _union(T_STR, T_BUF)),
                ParamDescriptor("salt", _union(T_STR, T_BUF)),
                ParamDescriptor("keylen", T_NUM.with_refinement(NON_NEG_INT)),
                ParamDescriptor("options", T_OBJ, optional=True),
                ParamDescriptor("callback", _t(BaseType.FUNCTION)),
            ),
            return_type=T_VOID,
            is_async=True,
            purity=Purity.IMPURE,
        ),
        "generateKeyPair": MethodSignature(
            name="generateKeyPair",
            params=(
                ParamDescriptor("type", T_STR),
                ParamDescriptor("options", T_OBJ),
                ParamDescriptor("callback", _t(BaseType.FUNCTION)),
            ),
            return_type=T_VOID,
            is_async=True,
            purity=Purity.IMPURE,
        ),
    }


# -------------------------------------------------------------------
# child_process
# -------------------------------------------------------------------


class ChildProcessModel(TypeScriptModelBase):
    qualified_name = "child_process"
    description = "Node.js child_process module."

    methods = {
        "exec": MethodSignature(
            name="exec",
            params=(
                ParamDescriptor("command", T_STR),
                ParamDescriptor("options", T_OBJ, optional=True),
                ParamDescriptor("callback", _t(BaseType.FUNCTION), optional=True,
                                description="(error: Error|null, stdout: string|Buffer, stderr: string|Buffer) => void"),
            ),
            return_type=_t(BaseType.CHILD_PROCESS),
            preconditions=(
                Precondition("Command must be non-empty", "command.length > 0"),
            ),
            purity=Purity.IMPURE,
        ),
        "execSync": MethodSignature(
            name="execSync",
            params=(
                ParamDescriptor("command", T_STR),
                ParamDescriptor("options", T_OBJ, optional=True),
            ),
            return_type=_union(T_STR, T_BUF),
            exceptions=(
                ExceptionCondition("Error", "command exits with non-zero code",
                                   "Command failed"),
            ),
            purity=Purity.IMPURE,
        ),
        "spawn": MethodSignature(
            name="spawn",
            params=(
                ParamDescriptor("command", T_STR),
                ParamDescriptor("args", _t(BaseType.ARRAY, T_STR), optional=True),
                ParamDescriptor("options", T_OBJ, optional=True),
            ),
            return_type=_t(BaseType.CHILD_PROCESS),
            purity=Purity.IMPURE,
        ),
        "spawnSync": MethodSignature(
            name="spawnSync",
            params=(
                ParamDescriptor("command", T_STR),
                ParamDescriptor("args", _t(BaseType.ARRAY, T_STR), optional=True),
                ParamDescriptor("options", T_OBJ, optional=True),
            ),
            return_type=T_OBJ,  # SpawnSyncReturns
            purity=Purity.IMPURE,
        ),
        "fork": MethodSignature(
            name="fork",
            params=(
                ParamDescriptor("modulePath", T_STR),
                ParamDescriptor("args", _t(BaseType.ARRAY, T_STR), optional=True),
                ParamDescriptor("options", T_OBJ, optional=True),
            ),
            return_type=_t(BaseType.CHILD_PROCESS),
            purity=Purity.IMPURE,
        ),
        "execFile": MethodSignature(
            name="execFile",
            params=(
                ParamDescriptor("file", T_STR),
                ParamDescriptor("args", _t(BaseType.ARRAY, T_STR), optional=True),
                ParamDescriptor("options", T_OBJ, optional=True),
                ParamDescriptor("callback", _t(BaseType.FUNCTION), optional=True),
            ),
            return_type=_t(BaseType.CHILD_PROCESS),
            purity=Purity.IMPURE,
        ),
    }


# -------------------------------------------------------------------
# os
# -------------------------------------------------------------------


class OsModel(TypeScriptModelBase):
    qualified_name = "os"
    description = "Node.js os module."

    properties = {
        "EOL": PropertyDescriptor(
            "EOL", T_STR, readonly=True,
            description="OS-specific end-of-line marker.",
        ),
    }

    methods = {
        "hostname": MethodSignature(
            name="hostname", params=(), return_type=T_STR, purity=Purity.READS,
        ),
        "platform": MethodSignature(
            name="platform", params=(),
            return_type=T_STR.with_refinement(
                _ref("p", BaseType.STRING,
                     "p ∈ {'aix','darwin','freebsd','linux','openbsd','sunos','win32'}")
            ),
            purity=Purity.READS,
        ),
        "arch": MethodSignature(
            name="arch", params=(),
            return_type=T_STR.with_refinement(
                _ref("a", BaseType.STRING, "a ∈ {'arm','arm64','ia32','mips','mipsel','ppc','ppc64','s390','s390x','x64'}")
            ),
            purity=Purity.READS,
        ),
        "cpus": MethodSignature(
            name="cpus", params=(),
            return_type=_t(BaseType.ARRAY, T_OBJ),
            postconditions=(
                Postcondition("Result length >= 1", "result.length >= 1"),
            ),
            purity=Purity.READS,
        ),
        "totalmem": MethodSignature(
            name="totalmem", params=(),
            return_type=T_NUM.with_refinement(_ref("n", BaseType.NUMBER, "n > 0 ∧ Number.isInteger(n)")),
            purity=Purity.READS,
        ),
        "freemem": MethodSignature(
            name="freemem", params=(),
            return_type=T_NUM.with_refinement(NON_NEG_INT),
            purity=Purity.READS,
        ),
        "homedir": MethodSignature(
            name="homedir", params=(), return_type=T_STR, purity=Purity.READS,
        ),
        "tmpdir": MethodSignature(
            name="tmpdir", params=(), return_type=T_STR, purity=Purity.READS,
        ),
        "type": MethodSignature(
            name="type", params=(), return_type=T_STR, purity=Purity.READS,
        ),
        "release": MethodSignature(
            name="release", params=(), return_type=T_STR, purity=Purity.READS,
        ),
        "uptime": MethodSignature(
            name="uptime", params=(),
            return_type=T_NUM.with_refinement(_ref("s", BaseType.NUMBER, "s >= 0")),
            purity=Purity.READS,
        ),
        "networkInterfaces": MethodSignature(
            name="networkInterfaces", params=(),
            return_type=T_OBJ,  # NodeJS.Dict<os.NetworkInterfaceInfo[]>
            purity=Purity.READS,
        ),
    }


# -------------------------------------------------------------------
# util
# -------------------------------------------------------------------


class UtilModel(TypeScriptModelBase):
    qualified_name = "util"
    description = "Node.js util module."

    methods = {
        "promisify": MethodSignature(
            name="promisify",
            params=(ParamDescriptor("fn", _t(BaseType.FUNCTION)),),
            return_type=_t(BaseType.FUNCTION),
            postconditions=(
                Postcondition("Returned function returns a Promise", "typeof result(...) is Promise"),
            ),
            refinement_transfers=(
                RefinementTransfer("fn", "return",
                                   "Return type wraps callback's success value in Promise"),
            ),
            purity=Purity.PURE,
        ),
        "callbackify": MethodSignature(
            name="callbackify",
            params=(ParamDescriptor("fn", _t(BaseType.FUNCTION)),),
            return_type=_t(BaseType.FUNCTION),
            purity=Purity.PURE,
        ),
        "inspect": MethodSignature(
            name="inspect",
            params=(
                ParamDescriptor("object", T_ANY),
                ParamDescriptor("options", T_OBJ, optional=True),
            ),
            return_type=T_STR,
            purity=Purity.PURE,
        ),
        "format": MethodSignature(
            name="format",
            params=(
                ParamDescriptor("format", T_STR),
                ParamDescriptor("params", T_ANY, rest=True),
            ),
            return_type=T_STR,
            purity=Purity.PURE,
        ),
        "types": MethodSignature(
            name="types",
            params=(),
            return_type=T_OBJ,
            purity=Purity.PURE,
            description="Namespace of type-checking utilities.",
        ),
        "deprecate": MethodSignature(
            name="deprecate",
            params=(
                ParamDescriptor("fn", _t(BaseType.FUNCTION)),
                ParamDescriptor("msg", T_STR),
                ParamDescriptor("code", T_STR, optional=True),
            ),
            return_type=_t(BaseType.FUNCTION),
            purity=Purity.PURE,
        ),
        "inherits": MethodSignature(
            name="inherits",
            params=(
                ParamDescriptor("constructor", _t(BaseType.FUNCTION)),
                ParamDescriptor("superConstructor", _t(BaseType.FUNCTION)),
            ),
            return_type=T_VOID,
            purity=Purity.WRITES,
        ),
        "isDeepStrictEqual": MethodSignature(
            name="isDeepStrictEqual",
            params=(
                ParamDescriptor("val1", T_ANY),
                ParamDescriptor("val2", T_ANY),
            ),
            return_type=T_BOOL,
            purity=Purity.PURE,
        ),
    }


# -------------------------------------------------------------------
# process
# -------------------------------------------------------------------


class ProcessModel(TypeScriptModelBase):
    qualified_name = "process"
    description = "Node.js process global."

    properties = {
        "env": PropertyDescriptor(
            "env", T_OBJ, readonly=False,
            description="Environment variables as Record<string, string | undefined>.",
        ),
        "argv": PropertyDescriptor(
            "argv", _t(BaseType.ARRAY, T_STR), readonly=True,
            description="Command-line arguments.",
        ),
        "pid": PropertyDescriptor(
            "pid", T_NUM.with_refinement(POS_INT), readonly=True,
        ),
        "platform": PropertyDescriptor(
            "platform", T_STR, readonly=True,
        ),
        "arch": PropertyDescriptor(
            "arch", T_STR, readonly=True,
        ),
        "version": PropertyDescriptor(
            "version", T_STR, readonly=True,
        ),
        "stdin": PropertyDescriptor(
            "stdin", _t(BaseType.STREAM), readonly=True,
        ),
        "stdout": PropertyDescriptor(
            "stdout", _t(BaseType.STREAM), readonly=True,
        ),
        "stderr": PropertyDescriptor(
            "stderr", _t(BaseType.STREAM), readonly=True,
        ),
    }

    methods = {
        "cwd": MethodSignature(
            name="cwd",
            params=(),
            return_type=T_STR,
            postconditions=(
                Postcondition("Returns absolute path", "path.isAbsolute(result)"),
            ),
            purity=Purity.READS,
        ),
        "exit": MethodSignature(
            name="exit",
            params=(ParamDescriptor("code", T_NUM, optional=True),),
            return_type=T_NEVER,
            postconditions=(
                Postcondition("Process terminates", "false"),
            ),
            purity=Purity.IMPURE,
        ),
        "on": MethodSignature(
            name="on",
            params=(
                ParamDescriptor("event", T_STR),
                ParamDescriptor("listener", _t(BaseType.FUNCTION)),
            ),
            return_type=T_OBJ,  # process itself
            purity=Purity.WRITES,
        ),
        "nextTick": MethodSignature(
            name="nextTick",
            params=(
                ParamDescriptor("callback", _t(BaseType.FUNCTION)),
                ParamDescriptor("args", T_ANY, rest=True),
            ),
            return_type=T_VOID,
            purity=Purity.IMPURE,
        ),
        "hrtime": MethodSignature(
            name="hrtime",
            params=(
                ParamDescriptor("time", _t(BaseType.TUPLE), optional=True),
            ),
            return_type=_t(BaseType.TUPLE),
            postconditions=(
                Postcondition("Returns [seconds, nanoseconds]",
                              "result.length === 2 ∧ result[0] >= 0 ∧ result[1] >= 0"),
            ),
            purity=Purity.READS,
        ),
        "memoryUsage": MethodSignature(
            name="memoryUsage",
            params=(),
            return_type=T_OBJ,
            postconditions=(
                Postcondition("Has rss, heapTotal, heapUsed, external, arrayBuffers",
                              "result.rss >= 0 ∧ result.heapTotal >= 0"),
            ),
            purity=Purity.READS,
        ),
        "cpuUsage": MethodSignature(
            name="cpuUsage",
            params=(ParamDescriptor("previousValue", T_OBJ, optional=True),),
            return_type=T_OBJ,
            postconditions=(
                Postcondition("Has user and system fields",
                              "result.user >= 0 ∧ result.system >= 0"),
            ),
            purity=Purity.READS,
        ),
    }


# ===================================================================
# REGISTRY
# ===================================================================


class TypeScriptModelRegistry:
    """
    Central registry mapping qualified names to their TypeScriptModelBase
    instances.  Supports lookup by exact name or prefix.
    """

    def __init__(self) -> None:
        self._models: Dict[str, TypeScriptModelBase] = {}

    # Registration -------------------------------------------------------

    def register(self, model: TypeScriptModelBase) -> None:
        self._models[model.qualified_name] = model

    def register_all(self, models: Sequence[TypeScriptModelBase]) -> None:
        for m in models:
            self.register(m)

    # Lookup -------------------------------------------------------------

    def get(self, qualified_name: str) -> Optional[TypeScriptModelBase]:
        return self._models.get(qualified_name)

    def get_signature(
        self, qualified_name: str, method_name: str
    ) -> Optional[MethodSignature]:
        model = self.get(qualified_name)
        if model is None:
            return None
        return model.get_signature(method_name)

    def get_property(
        self, qualified_name: str, prop_name: str
    ) -> Optional[PropertyDescriptor]:
        model = self.get(qualified_name)
        if model is None:
            return None
        return model.get_property(prop_name)

    def has(self, qualified_name: str) -> bool:
        return qualified_name in self._models

    def all_names(self) -> FrozenSet[str]:
        return frozenset(self._models.keys())

    def find_by_prefix(self, prefix: str) -> List[TypeScriptModelBase]:
        return [
            m for name, m in self._models.items() if name.startswith(prefix)
        ]

    def method_count(self) -> int:
        total = 0
        for m in self._models.values():
            total += len(m.methods) + len(m.static_methods)
        return total

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __repr__(self) -> str:
        return (
            f"TypeScriptModelRegistry({len(self._models)} models, "
            f"{self.method_count()} methods)"
        )


# ===================================================================
# Default registry with all built-in models
# ===================================================================


def build_default_registry() -> TypeScriptModelRegistry:
    """Construct a registry pre-populated with all standard models."""
    registry = TypeScriptModelRegistry()

    # Core JS/TS types
    registry.register(ArrayModel())
    registry.register(MapModel())
    registry.register(SetModel())
    registry.register(WeakMapModel())
    registry.register(WeakSetModel())
    registry.register(WeakRefModel())
    registry.register(PromiseModel())
    registry.register(StringModel())
    registry.register(NumberModel())
    registry.register(ObjectModel())
    registry.register(SymbolModel())
    registry.register(BigIntModel())
    registry.register(RegExpModel())
    registry.register(DateModel())

    # Error types
    registry.register(ErrorModel())
    registry.register(TypeErrorModel())
    registry.register(RangeErrorModel())
    registry.register(ReferenceErrorModel())
    registry.register(SyntaxErrorModel())

    # Built-in objects
    registry.register(JSONModel())
    registry.register(MathModel())
    registry.register(ConsoleModel())

    # Node.js modules
    registry.register(FsModel())
    registry.register(PathModel())
    registry.register(HttpModel())
    registry.register(HttpsModel())
    registry.register(BufferModel())
    registry.register(EventEmitterModel())
    registry.register(StreamModel())
    registry.register(UrlModel())
    registry.register(URLClassModel())
    registry.register(URLSearchParamsModel())
    registry.register(CryptoModel())
    registry.register(ChildProcessModel())
    registry.register(OsModel())
    registry.register(UtilModel())
    registry.register(ProcessModel())

    return registry


# Module-level singleton
DEFAULT_REGISTRY: TypeScriptModelRegistry = build_default_registry()


# ===================================================================
# Convenience query helpers
# ===================================================================


def lookup_method(
    qualified_name: str, method_name: str
) -> Optional[MethodSignature]:
    """Shortcut: look up a method signature in the default registry."""
    return DEFAULT_REGISTRY.get_signature(qualified_name, method_name)


def lookup_property(
    qualified_name: str, prop_name: str
) -> Optional[PropertyDescriptor]:
    """Shortcut: look up a property descriptor in the default registry."""
    return DEFAULT_REGISTRY.get_property(qualified_name, prop_name)


def lookup_preconditions(
    qualified_name: str, method_name: str
) -> Tuple[Precondition, ...]:
    model = DEFAULT_REGISTRY.get(qualified_name)
    if model is None:
        return ()
    return model.get_preconditions(method_name)


def lookup_postconditions(
    qualified_name: str, method_name: str
) -> Tuple[Postcondition, ...]:
    model = DEFAULT_REGISTRY.get(qualified_name)
    if model is None:
        return ()
    return model.get_postconditions(method_name)


def lookup_type_narrowing(
    qualified_name: str, method_name: str
) -> Tuple[TypeNarrowingEffect, ...]:
    model = DEFAULT_REGISTRY.get(qualified_name)
    if model is None:
        return ()
    return model.get_type_narrowing(method_name)


def lookup_refinement_transfer(
    qualified_name: str, method_name: str
) -> Tuple[RefinementTransfer, ...]:
    model = DEFAULT_REGISTRY.get(qualified_name)
    if model is None:
        return ()
    return model.get_refinement_transfer(method_name)


def is_pure(qualified_name: str, method_name: str) -> Optional[bool]:
    """Check if a method is pure (no side effects)."""
    sig = DEFAULT_REGISTRY.get_signature(qualified_name, method_name)
    if sig is None:
        return None
    return sig.purity == Purity.PURE


def can_throw(qualified_name: str, method_name: str) -> bool:
    """Check if a method has any documented exception conditions."""
    sig = DEFAULT_REGISTRY.get_signature(qualified_name, method_name)
    if sig is None:
        return False
    return len(sig.exceptions) > 0


def get_exceptions(
    qualified_name: str, method_name: str
) -> Tuple[ExceptionCondition, ...]:
    sig = DEFAULT_REGISTRY.get_signature(qualified_name, method_name)
    if sig is None:
        return ()
    return sig.exceptions


# ===================================================================
# Self-test / smoke check
# ===================================================================


def _self_check() -> None:
    """Quick validation that the registry is correctly wired."""
    reg = DEFAULT_REGISTRY
    assert len(reg) >= 30, f"Expected >= 30 models, got {len(reg)}"
    assert reg.method_count() >= 200, (
        f"Expected >= 200 methods, got {reg.method_count()}"
    )

    # Spot checks
    arr_push = reg.get_signature("Array", "push")
    assert arr_push is not None, "Array.push missing"
    assert arr_push.purity == Purity.WRITES

    arr_len = reg.get_property("Array", "length")
    assert arr_len is not None, "Array.length missing"
    assert arr_len.prop_type.refinement is not None

    map_get = reg.get_signature("Map", "get")
    assert map_get is not None, "Map.get missing"
    assert map_get.return_type.nullability == NullabilityKind.UNDEFINABLE

    math_sqrt = reg.get_signature("Math", "sqrt")
    assert math_sqrt is not None
    assert math_sqrt.purity == Purity.PURE
    assert len(math_sqrt.preconditions) > 0

    fs_read = reg.get_signature("fs", "readFileSync")
    assert fs_read is not None
    assert len(fs_read.exceptions) >= 2

    proc_exit = reg.get_signature("process", "exit")
    assert proc_exit is not None
    assert proc_exit.return_type.base == BaseType.NEVER

    assert is_pure("Math", "abs") is True
    assert is_pure("Array", "push") is False
    assert can_throw("String", "repeat") is True
    assert can_throw("String", "toLowerCase") is False

    random_uuid = reg.get_signature("crypto", "randomUUID")
    assert random_uuid is not None
    assert random_uuid.return_type.refinement is not None

    assert "Array" in reg
    assert "NotAModel" not in reg

    narrowing = lookup_type_narrowing("Number", "isFinite")
    assert len(narrowing) > 0

    transfers = lookup_refinement_transfer("Array", "push")
    assert len(transfers) > 0


if __name__ == "__main__":
    _self_check()
    print(f"Registry: {DEFAULT_REGISTRY}")
    print("All self-checks passed.")

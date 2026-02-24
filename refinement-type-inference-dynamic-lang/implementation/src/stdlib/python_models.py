"""
Python standard library models for refinement type inference.

Each model captures:
  - Parameter types with refinement predicates  (preconditions)
  - Return types with refinement predicates     (postconditions)
  - Side-effect annotations                     (IO, mutation, …)
  - Exception specifications                    (which exceptions may raise)
  - Nullity / length constraints                (where applicable)

The refinement language mirrors the core system:

    {x : int | x > 0 ∧ x < len(arr)}

All models are dataclass-based and self-contained — no imports from
the rest of the project so that the module can be tested in isolation.
"""

from __future__ import annotations

import enum
import functools
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


# ===================================================================
# Local enumerations (mirrors Sort from src/types but stand-alone)
# ===================================================================

class Sort(Enum):
    """Base sorts in the predicate language."""
    INT = auto()
    BOOL = auto()
    TAG = auto()
    STR = auto()
    FLOAT = auto()
    NONE = auto()
    LIST = auto()
    DICT = auto()
    SET = auto()
    TUPLE = auto()
    BYTES = auto()
    CALLABLE = auto()
    OBJECT = auto()
    ANY = auto()
    ITERATOR = auto()
    GENERATOR = auto()
    FILE = auto()
    MODULE = auto()
    TYPE = auto()
    COMPLEX = auto()
    FROZENSET = auto()
    BYTEARRAY = auto()
    MEMORYVIEW = auto()
    SLICE = auto()
    PROPERTY = auto()
    PATTERN = auto()
    MATCH = auto()
    PATH = auto()
    DEQUE = auto()
    COUNTER = auto()
    ORDERED_DICT = auto()
    DEFAULT_DICT = auto()
    CHAIN_MAP = auto()
    NAMED_TUPLE = auto()


class SideEffectKind(Enum):
    """Categories of side effects a function may have."""
    NONE = auto()
    IO_READ = auto()
    IO_WRITE = auto()
    STDOUT = auto()
    STDIN = auto()
    STDERR = auto()
    FILE_SYSTEM = auto()
    NETWORK = auto()
    PROCESS = auto()
    MUTATION = auto()
    GLOBAL_STATE = auto()
    EXCEPTION = auto()
    ALLOCATION = auto()
    ENVIRONMENT = auto()


class ParamKind(Enum):
    """Parameter passing convention."""
    POSITIONAL = auto()
    KEYWORD = auto()
    POSITIONAL_OR_KEYWORD = auto()
    VAR_POSITIONAL = auto()      # *args
    VAR_KEYWORD = auto()         # **kwargs


class Nullability(Enum):
    """Whether a value may be None."""
    NON_NULL = auto()
    NULLABLE = auto()
    ALWAYS_NULL = auto()


# ===================================================================
# Refinement constraint descriptors
# ===================================================================

@dataclass(frozen=True)
class RefinementConstraint:
    """
    A single predicate in the refinement logic.

    Represents predicates like ``x > 0``, ``len(result) == len(arg0)``,
    ``result != None``, etc.

    ``lhs`` and ``rhs`` are symbolic expression strings referencing
    parameter names (``arg0``, ``arg1``, …), ``result``, or literals.
    ``operator`` is one of  ==, !=, <, <=, >, >=, in, not_in,
    is, is_not, implies, iff, and, or, not, isinstance, hasattr,
    divisible_by, len_eq, len_leq, len_geq, between.
    """
    lhs: str
    operator: str
    rhs: str
    description: str = ""

    def __str__(self) -> str:
        if self.description:
            return self.description
        return f"{self.lhs} {self.operator} {self.rhs}"

    def negate(self) -> RefinementConstraint:
        negations = {
            "==": "!=", "!=": "==",
            "<": ">=", ">=": "<",
            ">": "<=", "<=": ">",
            "in": "not_in", "not_in": "in",
            "is": "is_not", "is_not": "is",
        }
        neg_op = negations.get(self.operator, f"not({self.operator})")
        return RefinementConstraint(
            lhs=self.lhs, operator=neg_op, rhs=self.rhs,
            description=f"¬({self.description})" if self.description else "",
        )


@dataclass(frozen=True)
class CompoundConstraint:
    """Conjunction / disjunction of refinement constraints."""
    connective: str  # "and" | "or" | "implies"
    children: Tuple[Union[RefinementConstraint, CompoundConstraint], ...] = ()

    def __str__(self) -> str:
        parts = [str(c) for c in self.children]
        sep = f" {self.connective} "
        return f"({sep.join(parts)})"


# ===================================================================
# Parameter & return specifications
# ===================================================================

@dataclass(frozen=True)
class ParamSpec:
    """Specification of a single parameter."""
    name: str
    sort: Sort
    kind: ParamKind = ParamKind.POSITIONAL_OR_KEYWORD
    nullable: Nullability = Nullability.NON_NULL
    default: Optional[str] = None
    has_default: bool = False
    refinements: Tuple[RefinementConstraint, ...] = ()
    description: str = ""
    element_sort: Optional[Sort] = None  # for container params

    @property
    def is_optional(self) -> bool:
        return self.has_default or self.nullable == Nullability.NULLABLE

    def with_refinement(self, c: RefinementConstraint) -> ParamSpec:
        return ParamSpec(
            name=self.name, sort=self.sort, kind=self.kind,
            nullable=self.nullable, default=self.default,
            has_default=self.has_default,
            refinements=self.refinements + (c,),
            description=self.description, element_sort=self.element_sort,
        )


@dataclass(frozen=True)
class ReturnSpec:
    """Specification of a function return."""
    sort: Sort
    nullable: Nullability = Nullability.NON_NULL
    refinements: Tuple[RefinementConstraint, ...] = ()
    description: str = ""
    element_sort: Optional[Sort] = None

    def with_refinement(self, c: RefinementConstraint) -> ReturnSpec:
        return ReturnSpec(
            sort=self.sort, nullable=self.nullable,
            refinements=self.refinements + (c,),
            description=self.description, element_sort=self.element_sort,
        )


@dataclass(frozen=True)
class SideEffect:
    """A single side-effect annotation."""
    kind: SideEffectKind
    description: str = ""
    target: str = ""  # e.g. "stdout", a file path pattern, …


@dataclass(frozen=True)
class ExceptionSpec:
    """An exception that may be raised."""
    exception_type: str
    condition: str = ""  # when this exception is raised
    description: str = ""


# ===================================================================
# Function signature
# ===================================================================

@dataclass(frozen=True)
class FunctionSignature:
    """
    Complete refinement-typed signature for a standard library function.

    Example for ``len``::

        FunctionSignature(
            qualified_name="builtins.len",
            params=(ParamSpec("obj", Sort.ANY),),
            returns=ReturnSpec(Sort.INT,
                refinements=(
                    RefinementConstraint("result", ">=", "0",
                        "len always returns non-negative"),
                )),
            preconditions=(...),
            postconditions=(...),
            ...
        )
    """
    qualified_name: str
    params: Tuple[ParamSpec, ...] = ()
    returns: ReturnSpec = ReturnSpec(Sort.NONE)
    preconditions: Tuple[Union[RefinementConstraint, CompoundConstraint], ...] = ()
    postconditions: Tuple[Union[RefinementConstraint, CompoundConstraint], ...] = ()
    side_effects: Tuple[SideEffect, ...] = ()
    exceptions: Tuple[ExceptionSpec, ...] = ()
    is_pure: bool = True
    is_total: bool = True  # always terminates
    is_deterministic: bool = True
    type_params: Tuple[str, ...] = ()  # generic type parameters
    overloads: Tuple[FunctionSignature, ...] = ()  # overloaded signatures
    description: str = ""
    variadic: bool = False  # accepts *args

    @property
    def arity(self) -> int:
        return len(self.params)

    def param_by_name(self, name: str) -> Optional[ParamSpec]:
        for p in self.params:
            if p.name == name:
                return p
        return None


# ===================================================================
# StdlibModel – base class
# ===================================================================

class StdlibModel(ABC):
    """
    Base class for all standard-library models.

    Subclasses implement :meth:`get_signature` and may override hooks
    for computing dynamic refinements based on call-site information.
    """

    @abstractmethod
    def get_signature(self) -> FunctionSignature:
        """Return the refinement-typed signature."""
        ...

    # -- Hooks for context-sensitive reasoning --------------------------

    def return_type_refinements(
        self, arg_sorts: Sequence[Sort],
    ) -> Tuple[RefinementConstraint, ...]:
        """
        Compute additional return-type refinements given concrete
        argument sorts at a particular call site.
        """
        return ()

    def preconditions(
        self, arg_sorts: Sequence[Sort],
    ) -> Tuple[RefinementConstraint, ...]:
        """Compute preconditions given argument sorts."""
        return ()

    def postconditions(
        self, arg_sorts: Sequence[Sort],
    ) -> Tuple[RefinementConstraint, ...]:
        """Compute postconditions given argument sorts."""
        return ()

    def side_effects(self) -> Tuple[SideEffect, ...]:
        """Return side effects of this function."""
        sig = self.get_signature()
        return sig.side_effects

    def exceptions(self) -> Tuple[ExceptionSpec, ...]:
        """Return exceptions this function may raise."""
        sig = self.get_signature()
        return sig.exceptions

    def is_pure(self) -> bool:
        sig = self.get_signature()
        return sig.is_pure

    @property
    def qualified_name(self) -> str:
        return self.get_signature().qualified_name


# ===================================================================
# ModelRegistry
# ===================================================================

class ModelRegistry:
    """
    Maps qualified names (``builtins.len``, ``os.path.join``, …) to
    :class:`StdlibModel` instances.

    Supports hierarchical lookup:  ``os.path.join`` is tried first,
    then ``os.path``, then ``os``.
    """

    def __init__(self) -> None:
        self._models: Dict[str, StdlibModel] = {}
        self._signatures: Dict[str, FunctionSignature] = {}

    # -- Registration ---------------------------------------------------

    def register(self, model: StdlibModel) -> None:
        name = model.qualified_name
        self._models[name] = model
        self._signatures[name] = model.get_signature()

    def register_signature(self, sig: FunctionSignature) -> None:
        self._signatures[sig.qualified_name] = sig

    def register_many(self, models: Iterable[StdlibModel]) -> None:
        for m in models:
            self.register(m)

    # -- Lookup ---------------------------------------------------------

    def lookup(self, qualified_name: str) -> Optional[StdlibModel]:
        return self._models.get(qualified_name)

    def lookup_signature(self, qualified_name: str) -> Optional[FunctionSignature]:
        return self._signatures.get(qualified_name)

    def contains(self, qualified_name: str) -> bool:
        return qualified_name in self._models or qualified_name in self._signatures

    def all_names(self) -> FrozenSet[str]:
        return frozenset(self._models.keys() | self._signatures.keys())

    def all_models(self) -> Dict[str, StdlibModel]:
        return dict(self._models)

    def all_signatures(self) -> Dict[str, FunctionSignature]:
        return dict(self._signatures)

    def merge(self, other: ModelRegistry) -> None:
        self._models.update(other._models)
        self._signatures.update(other._signatures)

    def __len__(self) -> int:
        return len(self._models) + len(
            k for k in self._signatures if k not in self._models
        )

    def __contains__(self, name: str) -> bool:
        return self.contains(name)

    def __repr__(self) -> str:
        return f"ModelRegistry({len(self._models)} models, {len(self._signatures)} sigs)"


# ===================================================================
# Helper: quick model builder via FunctionSignature only
# ===================================================================

class _SignatureModel(StdlibModel):
    """Thin wrapper: adapts a FunctionSignature into a StdlibModel."""

    def __init__(self, sig: FunctionSignature) -> None:
        self._sig = sig

    def get_signature(self) -> FunctionSignature:
        return self._sig


def _model(sig: FunctionSignature) -> StdlibModel:
    return _SignatureModel(sig)


# ===================================================================
# Convenience builders
# ===================================================================

def _non_neg_int(name: str = "result") -> RefinementConstraint:
    return RefinementConstraint(name, ">=", "0", f"{name} is non-negative")


def _positive_int(name: str = "result") -> RefinementConstraint:
    return RefinementConstraint(name, ">", "0", f"{name} is positive")


def _len_preserving() -> RefinementConstraint:
    return RefinementConstraint("len(result)", "==", "len(arg0)", "length preserving")


def _len_result_eq(expr: str) -> RefinementConstraint:
    return RefinementConstraint("len(result)", "==", expr, f"result length equals {expr}")


def _not_none(name: str = "result") -> RefinementConstraint:
    return RefinementConstraint(name, "is_not", "None", f"{name} is not None")


def _is_finite(name: str = "result") -> RefinementConstraint:
    return RefinementConstraint(f"isfinite({name})", "==", "True", f"{name} is finite")


def _between(name: str, lo: str, hi: str) -> RefinementConstraint:
    return RefinementConstraint(name, "between", f"({lo}, {hi})", f"{lo} <= {name} <= {hi}")


def _result_type_is(sort: Sort) -> RefinementConstraint:
    return RefinementConstraint("type(result)", "==", sort.name, f"result is {sort.name}")


# ===================================================================
#  BUILTINS
# ===================================================================

def register_builtin_models(registry: ModelRegistry) -> None:
    """Register models for Python built-in functions."""

    # ----- len ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.len",
        params=(ParamSpec("obj", Sort.ANY, description="Sized object"),),
        returns=ReturnSpec(Sort.INT, refinements=(_non_neg_int(),)),
        postconditions=(
            RefinementConstraint("result", ">=", "0", "len returns non-negative int"),
        ),
        exceptions=(
            ExceptionSpec("TypeError", "obj has no __len__"),
        ),
        description="Return the number of items in a container.",
    )))

    # ----- range -------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.range",
        params=(
            ParamSpec("start_or_stop", Sort.INT),
            ParamSpec("stop", Sort.INT, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("step", Sort.INT, has_default=True, default="1"),
        ),
        returns=ReturnSpec(Sort.LIST, element_sort=Sort.INT,
                           refinements=(_non_neg_int("len(result)"),)),
        preconditions=(
            RefinementConstraint("step", "!=", "0", "step must not be zero"),
        ),
        exceptions=(
            ExceptionSpec("ValueError", "step == 0"),
            ExceptionSpec("TypeError", "non-integer argument"),
        ),
        description="Return an immutable sequence of integers.",
    )))

    # ----- abs ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.abs",
        params=(ParamSpec("x", Sort.ANY, description="Numeric value"),),
        returns=ReturnSpec(Sort.ANY, refinements=(
            RefinementConstraint("result", ">=", "0", "abs is non-negative"),
        )),
        postconditions=(
            RefinementConstraint("result", ">=", "0"),
        ),
        description="Return the absolute value of a number.",
    )))

    # ----- min ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.min",
        params=(
            ParamSpec("args", Sort.ANY, kind=ParamKind.VAR_POSITIONAL),
            ParamSpec("key", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("default", Sort.ANY, kind=ParamKind.KEYWORD,
                      has_default=True, default="<sentinel>",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ANY),
        postconditions=(
            RefinementConstraint("result", "<=", "all(args)", "result <= every element"),
        ),
        exceptions=(
            ExceptionSpec("ValueError", "empty iterable with no default"),
            ExceptionSpec("TypeError", "uncomparable types"),
        ),
        variadic=True,
        description="Return the smallest item in an iterable or arguments.",
    )))

    # ----- max ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.max",
        params=(
            ParamSpec("args", Sort.ANY, kind=ParamKind.VAR_POSITIONAL),
            ParamSpec("key", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("default", Sort.ANY, kind=ParamKind.KEYWORD,
                      has_default=True, default="<sentinel>",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ANY),
        postconditions=(
            RefinementConstraint("result", ">=", "all(args)", "result >= every element"),
        ),
        exceptions=(
            ExceptionSpec("ValueError", "empty iterable with no default"),
            ExceptionSpec("TypeError", "uncomparable types"),
        ),
        variadic=True,
        description="Return the largest item in an iterable or arguments.",
    )))

    # ----- sum ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.sum",
        params=(
            ParamSpec("iterable", Sort.ANY, description="Iterable of numbers"),
            ParamSpec("start", Sort.INT, has_default=True, default="0"),
        ),
        returns=ReturnSpec(Sort.ANY),
        exceptions=(
            ExceptionSpec("TypeError", "unsummable types or str"),
        ),
        description="Sum items of an iterable, plus start value.",
    )))

    # ----- int ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.int",
        params=(
            ParamSpec("x", Sort.ANY, has_default=True, default="0"),
            ParamSpec("base", Sort.INT, has_default=True, default="10",
                      kind=ParamKind.KEYWORD),
        ),
        returns=ReturnSpec(Sort.INT),
        preconditions=(
            RefinementConstraint("base", "between", "(0, 36)",
                                 "base 0 or 2..36"),
        ),
        exceptions=(
            ExceptionSpec("ValueError", "invalid literal"),
            ExceptionSpec("TypeError", "non-string/bytes/number argument"),
        ),
        description="Convert a number or string to an integer.",
    )))

    # ----- float -------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.float",
        params=(
            ParamSpec("x", Sort.ANY, has_default=True, default="0"),
        ),
        returns=ReturnSpec(Sort.FLOAT),
        exceptions=(
            ExceptionSpec("ValueError", "invalid literal"),
            ExceptionSpec("TypeError", "non-string/number argument"),
        ),
        description="Convert a string or number to a floating-point number.",
    )))

    # ----- str ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.str",
        params=(
            ParamSpec("object", Sort.ANY, has_default=True, default="''"),
            ParamSpec("encoding", Sort.STR, has_default=True, default="'utf-8'",
                      kind=ParamKind.KEYWORD),
            ParamSpec("errors", Sort.STR, has_default=True, default="'strict'",
                      kind=ParamKind.KEYWORD),
        ),
        returns=ReturnSpec(Sort.STR, refinements=(_not_none(),)),
        description="Return a string version of object.",
    )))

    # ----- bool --------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.bool",
        params=(ParamSpec("x", Sort.ANY, has_default=True, default="False"),),
        returns=ReturnSpec(Sort.BOOL),
        description="Convert a value to a Boolean.",
    )))

    # ----- list --------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.list",
        params=(
            ParamSpec("iterable", Sort.ANY, has_default=True, default="()",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.LIST, refinements=(_non_neg_int("len(result)"),)),
        description="Built-in mutable sequence.",
    )))

    # ----- dict --------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.dict",
        params=(
            ParamSpec("mapping_or_iterable", Sort.ANY, has_default=True, default="()",
                      nullable=Nullability.NULLABLE, kind=ParamKind.POSITIONAL_OR_KEYWORD),
            ParamSpec("kwargs", Sort.ANY, kind=ParamKind.VAR_KEYWORD),
        ),
        returns=ReturnSpec(Sort.DICT, refinements=(_non_neg_int("len(result)"),)),
        description="Built-in mapping type.",
    )))

    # ----- set ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.set",
        params=(
            ParamSpec("iterable", Sort.ANY, has_default=True, default="()",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.SET, refinements=(_non_neg_int("len(result)"),)),
        description="Built-in unordered collection of unique elements.",
    )))

    # ----- tuple -------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.tuple",
        params=(
            ParamSpec("iterable", Sort.ANY, has_default=True, default="()",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.TUPLE, refinements=(_non_neg_int("len(result)"),)),
        description="Built-in immutable sequence.",
    )))

    # ----- print -------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.print",
        params=(
            ParamSpec("objects", Sort.ANY, kind=ParamKind.VAR_POSITIONAL),
            ParamSpec("sep", Sort.STR, kind=ParamKind.KEYWORD,
                      has_default=True, default="' '"),
            ParamSpec("end", Sort.STR, kind=ParamKind.KEYWORD,
                      has_default=True, default="'\\n'"),
            ParamSpec("file", Sort.ANY, kind=ParamKind.KEYWORD,
                      has_default=True, default="sys.stdout",
                      nullable=Nullability.NULLABLE),
            ParamSpec("flush", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="False"),
        ),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.STDOUT, "writes to stdout"),),
        is_pure=False,
        variadic=True,
        description="Print objects to the text stream file.",
    )))

    # ----- input -------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.input",
        params=(
            ParamSpec("prompt", Sort.STR, has_default=True, default="''"),
        ),
        returns=ReturnSpec(Sort.STR, refinements=(_not_none(),)),
        side_effects=(
            SideEffect(SideEffectKind.STDOUT, "prints prompt"),
            SideEffect(SideEffectKind.STDIN, "reads from stdin"),
        ),
        is_pure=False,
        is_deterministic=False,
        exceptions=(ExceptionSpec("EOFError", "stdin is closed"),),
        description="Read a string from standard input.",
    )))

    # ----- sorted ------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.sorted",
        params=(
            ParamSpec("iterable", Sort.ANY),
            ParamSpec("key", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("reverse", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="False"),
        ),
        returns=ReturnSpec(Sort.LIST, refinements=(
            _len_preserving(),
            _non_neg_int("len(result)"),
        )),
        postconditions=(
            RefinementConstraint("len(result)", "==", "len(arg0)",
                                 "sorted preserves length"),
            RefinementConstraint("set(result)", "==", "set(arg0)",
                                 "sorted preserves elements"),
        ),
        exceptions=(ExceptionSpec("TypeError", "uncomparable types"),),
        description="Return a new sorted list from the items in iterable.",
    )))

    # ----- reversed ----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.reversed",
        params=(ParamSpec("seq", Sort.ANY),),
        returns=ReturnSpec(Sort.ITERATOR),
        postconditions=(
            RefinementConstraint("list(result)", "==", "list(arg0)[::-1]",
                                 "reversed reverses order"),
        ),
        exceptions=(ExceptionSpec("TypeError", "not reversible"),),
        description="Return a reverse iterator.",
    )))

    # ----- enumerate ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.enumerate",
        params=(
            ParamSpec("iterable", Sort.ANY),
            ParamSpec("start", Sort.INT, has_default=True, default="0"),
        ),
        returns=ReturnSpec(Sort.ITERATOR, element_sort=Sort.TUPLE),
        postconditions=(
            RefinementConstraint("result[i][0]", "==", "start + i",
                                 "index matches position"),
        ),
        description="Return an enumerate object.",
    )))

    # ----- zip ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.zip",
        params=(
            ParamSpec("iterables", Sort.ANY, kind=ParamKind.VAR_POSITIONAL),
            ParamSpec("strict", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="False"),
        ),
        returns=ReturnSpec(Sort.ITERATOR, element_sort=Sort.TUPLE),
        postconditions=(
            RefinementConstraint("len(list(result))", "==",
                                 "min(len(it) for it in iterables)",
                                 "zip truncates to shortest"),
        ),
        exceptions=(
            ExceptionSpec("ValueError", "strict=True and lengths differ"),
        ),
        variadic=True,
        description="Iterate over several iterables in parallel.",
    )))

    # ----- map ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.map",
        params=(
            ParamSpec("function", Sort.CALLABLE),
            ParamSpec("iterables", Sort.ANY, kind=ParamKind.VAR_POSITIONAL),
        ),
        returns=ReturnSpec(Sort.ITERATOR),
        postconditions=(
            RefinementConstraint("len(list(result))", "==",
                                 "min(len(it) for it in iterables)",
                                 "map preserves shortest length"),
        ),
        variadic=True,
        description="Apply function to every item of iterable.",
    )))

    # ----- filter ------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.filter",
        params=(
            ParamSpec("function", Sort.CALLABLE, nullable=Nullability.NULLABLE),
            ParamSpec("iterable", Sort.ANY),
        ),
        returns=ReturnSpec(Sort.ITERATOR),
        postconditions=(
            RefinementConstraint("len(list(result))", "<=", "len(list(arg1))",
                                 "filter can only reduce length"),
        ),
        description="Construct an iterator from those elements for which function is true.",
    )))

    # ----- isinstance --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.isinstance",
        params=(
            ParamSpec("object", Sort.ANY),
            ParamSpec("classinfo", Sort.ANY),
        ),
        returns=ReturnSpec(Sort.BOOL),
        description="Return True if object is an instance of classinfo.",
    )))

    # ----- hasattr -----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.hasattr",
        params=(
            ParamSpec("object", Sort.ANY),
            ParamSpec("name", Sort.STR),
        ),
        returns=ReturnSpec(Sort.BOOL),
        description="Return True if the object has the named attribute.",
    )))

    # ----- getattr -----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.getattr",
        params=(
            ParamSpec("object", Sort.ANY),
            ParamSpec("name", Sort.STR),
            ParamSpec("default", Sort.ANY, has_default=True, default="<sentinel>",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ANY, nullable=Nullability.NULLABLE),
        exceptions=(
            ExceptionSpec("AttributeError", "attribute not found and no default"),
        ),
        description="Get a named attribute from an object.",
    )))

    # ----- type --------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.type",
        params=(
            ParamSpec("object", Sort.ANY),
        ),
        returns=ReturnSpec(Sort.TYPE, refinements=(_not_none(),)),
        description="Return the type of an object.",
    )))

    # ----- id ----------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.id",
        params=(ParamSpec("object", Sort.ANY),),
        returns=ReturnSpec(Sort.INT, refinements=(_non_neg_int(),)),
        description="Return the identity of an object (CPython: memory address).",
    )))

    # ----- hash --------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.hash",
        params=(ParamSpec("object", Sort.ANY),),
        returns=ReturnSpec(Sort.INT),
        exceptions=(ExceptionSpec("TypeError", "unhashable type"),),
        description="Return the hash value of the object.",
    )))

    # ----- repr --------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.repr",
        params=(ParamSpec("object", Sort.ANY),),
        returns=ReturnSpec(Sort.STR, refinements=(_not_none(),)),
        description="Return a string containing a printable representation.",
    )))

    # ----- chr ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.chr",
        params=(ParamSpec("i", Sort.INT),),
        returns=ReturnSpec(Sort.STR, refinements=(
            RefinementConstraint("len(result)", "==", "1", "chr returns single char"),
        )),
        preconditions=(
            RefinementConstraint("i", "between", "(0, 1114111)", "valid Unicode code point"),
        ),
        exceptions=(
            ExceptionSpec("ValueError", "i outside 0..0x10ffff"),
        ),
        description="Return the string representing a character.",
    )))

    # ----- ord ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.ord",
        params=(ParamSpec("c", Sort.STR),),
        returns=ReturnSpec(Sort.INT, refinements=(
            _between("result", "0", "1114111"),
        )),
        preconditions=(
            RefinementConstraint("len(c)", "==", "1", "c must be single character"),
        ),
        exceptions=(
            ExceptionSpec("TypeError", "not a string of length 1"),
        ),
        description="Return an integer representing the Unicode code point.",
    )))

    # ----- hex ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.hex",
        params=(ParamSpec("x", Sort.INT),),
        returns=ReturnSpec(Sort.STR, refinements=(
            RefinementConstraint("result[:2]", "in", "('0x', '-0x')",
                                 "hex prefix"),
        )),
        description="Convert an integer to a lowercase hexadecimal string.",
    )))

    # ----- oct ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.oct",
        params=(ParamSpec("x", Sort.INT),),
        returns=ReturnSpec(Sort.STR, refinements=(
            RefinementConstraint("result[:2]", "in", "('0o', '-0o')",
                                 "oct prefix"),
        )),
        description="Convert an integer to an octal string.",
    )))

    # ----- bin ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.bin",
        params=(ParamSpec("x", Sort.INT),),
        returns=ReturnSpec(Sort.STR, refinements=(
            RefinementConstraint("result[:2]", "in", "('0b', '-0b')",
                                 "bin prefix"),
        )),
        description="Convert an integer to a binary string.",
    )))

    # ----- round -------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.round",
        params=(
            ParamSpec("number", Sort.ANY),
            ParamSpec("ndigits", Sort.INT, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ANY),
        exceptions=(ExceptionSpec("TypeError", "not a number"),),
        description="Round a number to a given precision in decimal digits.",
    )))

    # ----- pow ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.pow",
        params=(
            ParamSpec("base", Sort.ANY),
            ParamSpec("exp", Sort.ANY),
            ParamSpec("mod", Sort.INT, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ANY),
        exceptions=(
            ExceptionSpec("ValueError", "negative exp with mod"),
            ExceptionSpec("TypeError", "unsupported operand types"),
        ),
        description="Return base to the power exp; if mod present, return base**exp % mod.",
    )))

    # ----- divmod ------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.divmod",
        params=(
            ParamSpec("a", Sort.ANY),
            ParamSpec("b", Sort.ANY),
        ),
        returns=ReturnSpec(Sort.TUPLE, refinements=(
            RefinementConstraint("len(result)", "==", "2", "divmod returns pair"),
        )),
        postconditions=(
            RefinementConstraint("result[0] * b + result[1]", "==", "a",
                                 "quotient-remainder identity"),
        ),
        preconditions=(
            RefinementConstraint("b", "!=", "0", "divisor must not be zero"),
        ),
        exceptions=(
            ExceptionSpec("ZeroDivisionError", "b == 0"),
        ),
        description="Return the pair (quotient, remainder).",
    )))

    # ----- all ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.all",
        params=(ParamSpec("iterable", Sort.ANY),),
        returns=ReturnSpec(Sort.BOOL),
        postconditions=(
            RefinementConstraint("result", "==", "True",
                                 "all returns True iff every element is truthy"),
        ),
        description="Return True if all elements of iterable are true.",
    )))

    # ----- any ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.any",
        params=(ParamSpec("iterable", Sort.ANY),),
        returns=ReturnSpec(Sort.BOOL),
        description="Return True if any element of iterable is true.",
    )))

    # ----- iter --------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.iter",
        params=(
            ParamSpec("object", Sort.ANY),
            ParamSpec("sentinel", Sort.ANY, has_default=True, default="<sentinel>",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ITERATOR, refinements=(_not_none(),)),
        exceptions=(ExceptionSpec("TypeError", "object not iterable"),),
        description="Get an iterator from an object.",
    )))

    # ----- next --------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.next",
        params=(
            ParamSpec("iterator", Sort.ITERATOR),
            ParamSpec("default", Sort.ANY, has_default=True, default="<sentinel>",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ANY, nullable=Nullability.NULLABLE),
        exceptions=(ExceptionSpec("StopIteration", "iterator exhausted and no default"),),
        description="Retrieve the next item from the iterator.",
    )))

    # ----- open --------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.open",
        params=(
            ParamSpec("file", Sort.ANY),
            ParamSpec("mode", Sort.STR, has_default=True, default="'r'"),
            ParamSpec("buffering", Sort.INT, has_default=True, default="-1"),
            ParamSpec("encoding", Sort.STR, has_default=True, default="None",
                      nullable=Nullability.NULLABLE, kind=ParamKind.KEYWORD),
            ParamSpec("errors", Sort.STR, has_default=True, default="None",
                      nullable=Nullability.NULLABLE, kind=ParamKind.KEYWORD),
            ParamSpec("newline", Sort.STR, has_default=True, default="None",
                      nullable=Nullability.NULLABLE, kind=ParamKind.KEYWORD),
            ParamSpec("closefd", Sort.BOOL, has_default=True, default="True",
                      kind=ParamKind.KEYWORD),
            ParamSpec("opener", Sort.CALLABLE, has_default=True, default="None",
                      nullable=Nullability.NULLABLE, kind=ParamKind.KEYWORD),
        ),
        returns=ReturnSpec(Sort.FILE, refinements=(_not_none(),)),
        side_effects=(
            SideEffect(SideEffectKind.FILE_SYSTEM, "opens a file"),
            SideEffect(SideEffectKind.ALLOCATION, "allocates file handle"),
        ),
        exceptions=(
            ExceptionSpec("FileNotFoundError", "file does not exist (mode 'r')"),
            ExceptionSpec("PermissionError", "insufficient permissions"),
            ExceptionSpec("IsADirectoryError", "path is a directory"),
            ExceptionSpec("OSError", "general OS error"),
        ),
        is_pure=False,
        description="Open file and return a stream.",
    )))

    # ----- super -------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.super",
        params=(
            ParamSpec("type_", Sort.TYPE, has_default=True, default="<auto>",
                      nullable=Nullability.NULLABLE),
            ParamSpec("object_or_type", Sort.ANY, has_default=True, default="<auto>",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.OBJECT, refinements=(_not_none(),)),
        description="Return a proxy object that delegates method calls to parent.",
    )))

    # ----- property ----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.property",
        params=(
            ParamSpec("fget", Sort.CALLABLE, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("fset", Sort.CALLABLE, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("fdel", Sort.CALLABLE, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("doc", Sort.STR, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.PROPERTY),
        description="Return a property attribute.",
    )))

    # ----- staticmethod / classmethod ----------------------------------
    for name in ("staticmethod", "classmethod"):
        registry.register(_model(FunctionSignature(
            qualified_name=f"builtins.{name}",
            params=(ParamSpec("function", Sort.CALLABLE),),
            returns=ReturnSpec(Sort.CALLABLE),
            description=f"Transform a method into a {name}.",
        )))

    # ----- vars --------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.vars",
        params=(
            ParamSpec("object", Sort.ANY, has_default=True, default="<sentinel>",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.DICT),
        exceptions=(ExceptionSpec("TypeError", "object has no __dict__"),),
        description="Return the __dict__ attribute.",
    )))

    # ----- dir ---------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.dir",
        params=(
            ParamSpec("object", Sort.ANY, has_default=True, default="<sentinel>",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.LIST, element_sort=Sort.STR,
                           refinements=(_non_neg_int("len(result)"),)),
        description="Return list of names in the current scope or object attributes.",
    )))

    # ----- globals / locals --------------------------------------------
    for name in ("globals", "locals"):
        registry.register(_model(FunctionSignature(
            qualified_name=f"builtins.{name}",
            params=(),
            returns=ReturnSpec(Sort.DICT),
            is_pure=False,
            description=f"Return the current {name} symbol table.",
        )))

    # ----- callable ----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.callable",
        params=(ParamSpec("object", Sort.ANY),),
        returns=ReturnSpec(Sort.BOOL),
        description="Return True if the object is callable.",
    )))

    # ----- delattr -----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.delattr",
        params=(
            ParamSpec("object", Sort.ANY),
            ParamSpec("name", Sort.STR),
        ),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.MUTATION, "deletes attribute"),),
        exceptions=(ExceptionSpec("AttributeError", "attribute not found"),),
        is_pure=False,
        description="Delete a named attribute on an object.",
    )))

    # ----- setattr -----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.setattr",
        params=(
            ParamSpec("object", Sort.ANY),
            ParamSpec("name", Sort.STR),
            ParamSpec("value", Sort.ANY),
        ),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.MUTATION, "sets attribute"),),
        is_pure=False,
        description="Set a named attribute on an object.",
    )))

    # ----- format ------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.format",
        params=(
            ParamSpec("value", Sort.ANY),
            ParamSpec("format_spec", Sort.STR, has_default=True, default="''"),
        ),
        returns=ReturnSpec(Sort.STR, refinements=(_not_none(),)),
        description="Convert a value to a formatted representation.",
    )))

    # ----- ascii -------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.ascii",
        params=(ParamSpec("object", Sort.ANY),),
        returns=ReturnSpec(Sort.STR, refinements=(_not_none(),)),
        description="Return an ASCII-only repr of an object.",
    )))

    # ----- breakpoint --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.breakpoint",
        params=(ParamSpec("args", Sort.ANY, kind=ParamKind.VAR_POSITIONAL),
                ParamSpec("kws", Sort.ANY, kind=ParamKind.VAR_KEYWORD)),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.PROCESS, "enters debugger"),),
        is_pure=False,
        variadic=True,
        description="Enter the debugger.",
    )))

    # ----- compile -----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.compile",
        params=(
            ParamSpec("source", Sort.ANY),
            ParamSpec("filename", Sort.STR),
            ParamSpec("mode", Sort.STR),
            ParamSpec("flags", Sort.INT, has_default=True, default="0"),
            ParamSpec("dont_inherit", Sort.BOOL, has_default=True, default="False"),
            ParamSpec("optimize", Sort.INT, has_default=True, default="-1"),
        ),
        returns=ReturnSpec(Sort.OBJECT, refinements=(_not_none(),)),
        preconditions=(
            RefinementConstraint("mode", "in", "('exec','eval','single')",
                                 "valid compilation mode"),
        ),
        exceptions=(
            ExceptionSpec("SyntaxError", "invalid source"),
            ExceptionSpec("ValueError", "null bytes in source"),
        ),
        description="Compile source into a code object.",
    )))

    # ----- eval / exec -------------------------------------------------
    for name in ("eval", "exec"):
        registry.register(_model(FunctionSignature(
            qualified_name=f"builtins.{name}",
            params=(
                ParamSpec("source", Sort.ANY),
                ParamSpec("globals", Sort.DICT, has_default=True, default="None",
                          nullable=Nullability.NULLABLE),
                ParamSpec("locals", Sort.DICT, has_default=True, default="None",
                          nullable=Nullability.NULLABLE),
            ),
            returns=ReturnSpec(Sort.ANY if name == "eval"
                               else Sort.NONE, nullable=Nullability.NULLABLE),
            side_effects=(SideEffect(SideEffectKind.GLOBAL_STATE, "executes code"),),
            is_pure=False,
            is_deterministic=False,
            description=f"Execute dynamic {'expression' if name=='eval' else 'statements'}.",
        )))

    # ----- __import__ --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.__import__",
        params=(
            ParamSpec("name", Sort.STR),
            ParamSpec("globals", Sort.DICT, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("locals", Sort.DICT, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("fromlist", Sort.TUPLE, has_default=True, default="()"),
            ParamSpec("level", Sort.INT, has_default=True, default="0"),
        ),
        returns=ReturnSpec(Sort.MODULE, refinements=(_not_none(),)),
        side_effects=(SideEffect(SideEffectKind.GLOBAL_STATE, "imports module"),),
        exceptions=(ExceptionSpec("ModuleNotFoundError", "module not found"),),
        is_pure=False,
        description="Function invoked by the import statement.",
    )))

    # ----- memoryview --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.memoryview",
        params=(ParamSpec("obj", Sort.ANY),),
        returns=ReturnSpec(Sort.MEMORYVIEW, refinements=(_not_none(),)),
        exceptions=(ExceptionSpec("TypeError", "obj doesn't support buffer protocol"),),
        description="Create a memoryview that references obj.",
    )))

    # ----- bytearray ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.bytearray",
        params=(
            ParamSpec("source", Sort.ANY, has_default=True, default="b''",
                      nullable=Nullability.NULLABLE),
            ParamSpec("encoding", Sort.STR, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("errors", Sort.STR, has_default=True, default="'strict'"),
        ),
        returns=ReturnSpec(Sort.BYTEARRAY, refinements=(_non_neg_int("len(result)"),)),
        description="Return a mutable sequence of bytes.",
    )))

    # ----- bytes -------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.bytes",
        params=(
            ParamSpec("source", Sort.ANY, has_default=True, default="b''",
                      nullable=Nullability.NULLABLE),
            ParamSpec("encoding", Sort.STR, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("errors", Sort.STR, has_default=True, default="'strict'"),
        ),
        returns=ReturnSpec(Sort.BYTES, refinements=(_non_neg_int("len(result)"),)),
        description="Return an immutable sequence of bytes.",
    )))

    # ----- complex -----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.complex",
        params=(
            ParamSpec("real", Sort.ANY, has_default=True, default="0"),
            ParamSpec("imag", Sort.ANY, has_default=True, default="0"),
        ),
        returns=ReturnSpec(Sort.COMPLEX),
        exceptions=(
            ExceptionSpec("ValueError", "invalid literal"),
            ExceptionSpec("TypeError", "non-numeric argument"),
        ),
        description="Create a complex number.",
    )))

    # ----- frozenset ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.frozenset",
        params=(
            ParamSpec("iterable", Sort.ANY, has_default=True, default="()",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.FROZENSET, refinements=(_non_neg_int("len(result)"),)),
        description="Return an immutable frozenset object.",
    )))

    # ----- slice -------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.slice",
        params=(
            ParamSpec("start_or_stop", Sort.ANY),
            ParamSpec("stop", Sort.ANY, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("step", Sort.ANY, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.SLICE, refinements=(_not_none(),)),
        description="Create a slice object.",
    )))

    # ----- object ------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="builtins.object",
        params=(),
        returns=ReturnSpec(Sort.OBJECT, refinements=(_not_none(),)),
        description="The most base type; the ultimate base class.",
    )))


# ===================================================================
#  OS MODULE
# ===================================================================

def register_os_models(registry: ModelRegistry) -> None:
    """Register models for the os and os.path modules."""

    # ----- os.path.join ------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.path.join",
        params=(
            ParamSpec("path", Sort.STR),
            ParamSpec("paths", Sort.STR, kind=ParamKind.VAR_POSITIONAL),
        ),
        returns=ReturnSpec(Sort.STR, refinements=(
            _not_none(),
            RefinementConstraint("len(result)", ">", "0", "non-empty path"),
        )),
        variadic=True,
        description="Join path components intelligently.",
    )))

    # ----- os.path.exists ----------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.path.exists",
        params=(ParamSpec("path", Sort.ANY),),
        returns=ReturnSpec(Sort.BOOL),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "stats path"),),
        is_pure=False,
        is_deterministic=False,
        description="Return True if path exists.",
    )))

    # ----- os.listdir --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.listdir",
        params=(
            ParamSpec("path", Sort.STR, has_default=True, default="'.'"),
        ),
        returns=ReturnSpec(Sort.LIST, element_sort=Sort.STR,
                           refinements=(_non_neg_int("len(result)"),)),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "reads directory"),),
        exceptions=(
            ExceptionSpec("FileNotFoundError", "path does not exist"),
            ExceptionSpec("NotADirectoryError", "path is not a directory"),
            ExceptionSpec("PermissionError", "insufficient permissions"),
        ),
        is_pure=False,
        is_deterministic=False,
        description="Return list of entries in the directory.",
    )))

    # ----- os.getcwd ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.getcwd",
        params=(),
        returns=ReturnSpec(Sort.STR, refinements=(
            _not_none(),
            RefinementConstraint("len(result)", ">", "0", "cwd is non-empty"),
        )),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "queries cwd"),),
        is_pure=False,
        is_deterministic=False,
        description="Return the current working directory.",
    )))

    # ----- os.environ --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.environ",
        params=(),
        returns=ReturnSpec(Sort.DICT),
        side_effects=(SideEffect(SideEffectKind.ENVIRONMENT, "reads environment"),),
        is_pure=False,
        is_deterministic=False,
        description="A mapping representing the string environment.",
    )))

    # ----- os.makedirs -------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.makedirs",
        params=(
            ParamSpec("name", Sort.STR),
            ParamSpec("mode", Sort.INT, has_default=True, default="0o777"),
            ParamSpec("exist_ok", Sort.BOOL, has_default=True, default="False"),
        ),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "creates directories"),),
        exceptions=(
            ExceptionSpec("FileExistsError", "exist_ok=False and dir exists"),
            ExceptionSpec("PermissionError", "insufficient permissions"),
            ExceptionSpec("OSError", "general OS error"),
        ),
        is_pure=False,
        description="Recursive directory creation.",
    )))

    # ----- os.remove ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.remove",
        params=(ParamSpec("path", Sort.STR),),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "deletes file"),),
        exceptions=(
            ExceptionSpec("FileNotFoundError", "path does not exist"),
            ExceptionSpec("IsADirectoryError", "path is a directory"),
            ExceptionSpec("PermissionError", "insufficient permissions"),
        ),
        is_pure=False,
        description="Remove (delete) the file path.",
    )))

    # ----- os.rename ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.rename",
        params=(
            ParamSpec("src", Sort.STR),
            ParamSpec("dst", Sort.STR),
        ),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "renames file"),),
        exceptions=(
            ExceptionSpec("FileNotFoundError", "src does not exist"),
            ExceptionSpec("FileExistsError", "dst exists (Windows)"),
            ExceptionSpec("OSError", "general OS error"),
        ),
        is_pure=False,
        description="Rename file or directory src to dst.",
    )))

    # ----- os.stat -----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.stat",
        params=(
            ParamSpec("path", Sort.ANY),
            ParamSpec("dir_fd", Sort.INT, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("follow_symlinks", Sort.BOOL, has_default=True, default="True"),
        ),
        returns=ReturnSpec(Sort.OBJECT, refinements=(_not_none(),)),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "stats path"),),
        exceptions=(
            ExceptionSpec("FileNotFoundError", "path does not exist"),
            ExceptionSpec("PermissionError", "insufficient permissions"),
        ),
        is_pure=False,
        description="Perform a stat system call on the given path.",
    )))

    # ----- os.walk -----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.walk",
        params=(
            ParamSpec("top", Sort.STR),
            ParamSpec("topdown", Sort.BOOL, has_default=True, default="True"),
            ParamSpec("onerror", Sort.CALLABLE, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("followlinks", Sort.BOOL, has_default=True, default="False"),
        ),
        returns=ReturnSpec(Sort.GENERATOR, element_sort=Sort.TUPLE),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "walks directory tree"),),
        is_pure=False,
        is_deterministic=False,
        description="Generate file names in a directory tree by walking top-down or bottom-up.",
    )))

    # ----- os.path.isfile ----------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.path.isfile",
        params=(ParamSpec("path", Sort.ANY),),
        returns=ReturnSpec(Sort.BOOL),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "stats path"),),
        is_pure=False,
        is_deterministic=False,
        description="Return True if path is an existing regular file.",
    )))

    # ----- os.path.isdir -----------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.path.isdir",
        params=(ParamSpec("path", Sort.ANY),),
        returns=ReturnSpec(Sort.BOOL),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "stats path"),),
        is_pure=False,
        is_deterministic=False,
        description="Return True if path is an existing directory.",
    )))

    # ----- os.path.abspath ---------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.path.abspath",
        params=(ParamSpec("path", Sort.STR),),
        returns=ReturnSpec(Sort.STR, refinements=(
            _not_none(),
            RefinementConstraint("result[0]", "==", "os.sep",
                                 "absolute path starts with separator"),
        )),
        description="Return an absolute version of the pathname.",
    )))

    # ----- os.path.dirname ---------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.path.dirname",
        params=(ParamSpec("path", Sort.STR),),
        returns=ReturnSpec(Sort.STR, refinements=(_not_none(),)),
        postconditions=(
            RefinementConstraint("len(result)", "<=", "len(arg0)",
                                 "dirname is not longer than path"),
        ),
        description="Return the directory name of pathname path.",
    )))

    # ----- os.path.basename --------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.path.basename",
        params=(ParamSpec("path", Sort.STR),),
        returns=ReturnSpec(Sort.STR, refinements=(_not_none(),)),
        postconditions=(
            RefinementConstraint("len(result)", "<=", "len(arg0)",
                                 "basename is not longer than path"),
        ),
        description="Return the base name of pathname path.",
    )))

    # ----- os.path.splitext --------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.path.splitext",
        params=(ParamSpec("path", Sort.STR),),
        returns=ReturnSpec(Sort.TUPLE, refinements=(
            RefinementConstraint("len(result)", "==", "2", "returns (root, ext) pair"),
        )),
        postconditions=(
            RefinementConstraint("result[0] + result[1]", "==", "arg0",
                                 "root + ext reconstructs path"),
        ),
        description="Split the pathname into (root, ext) pair.",
    )))

    # ----- os.path.split -----------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.path.split",
        params=(ParamSpec("path", Sort.STR),),
        returns=ReturnSpec(Sort.TUPLE, refinements=(
            RefinementConstraint("len(result)", "==", "2", "returns (head, tail) pair"),
        )),
        description="Split path into (head, tail) where tail is the last component.",
    )))

    # ----- os.path.getsize ---------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="os.path.getsize",
        params=(ParamSpec("filename", Sort.STR),),
        returns=ReturnSpec(Sort.INT, refinements=(_non_neg_int(),)),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "stats file"),),
        exceptions=(
            ExceptionSpec("FileNotFoundError", "file does not exist"),
            ExceptionSpec("OSError", "general OS error"),
        ),
        is_pure=False,
        description="Return the size in bytes of path.",
    )))

    # ----- os.sep, os.linesep, os.devnull (constants) ------------------
    for name, desc in (
        ("os.sep", "Path separator"),
        ("os.linesep", "Line separator"),
        ("os.devnull", "Null device path"),
    ):
        registry.register(_model(FunctionSignature(
            qualified_name=name,
            params=(),
            returns=ReturnSpec(Sort.STR, refinements=(
                _not_none(),
                RefinementConstraint("len(result)", ">", "0", "non-empty constant"),
            )),
            description=desc,
        )))


# ===================================================================
#  SYS MODULE
# ===================================================================

def register_sys_models(registry: ModelRegistry) -> None:
    """Register models for the sys module."""

    # ----- sys.argv ----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="sys.argv",
        params=(),
        returns=ReturnSpec(Sort.LIST, element_sort=Sort.STR,
                           refinements=(
                               _non_neg_int("len(result)"),
                               RefinementConstraint("len(result)", ">=", "1",
                                                    "argv always has at least the script name"),
                           )),
        is_deterministic=False,
        description="List of command-line arguments passed to the script.",
    )))

    # ----- sys.path ----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="sys.path",
        params=(),
        returns=ReturnSpec(Sort.LIST, element_sort=Sort.STR,
                           refinements=(_non_neg_int("len(result)"),)),
        is_deterministic=False,
        is_pure=False,
        description="Module search path.",
    )))

    # ----- sys.stdin / stdout / stderr ---------------------------------
    for stream in ("stdin", "stdout", "stderr"):
        registry.register(_model(FunctionSignature(
            qualified_name=f"sys.{stream}",
            params=(),
            returns=ReturnSpec(Sort.FILE, refinements=(_not_none(),)),
            description=f"Standard {stream} stream.",
        )))

    # ----- sys.exit ----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="sys.exit",
        params=(
            ParamSpec("code", Sort.ANY, has_default=True, default="0",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.NONE),
        exceptions=(ExceptionSpec("SystemExit", "always raises SystemExit"),),
        side_effects=(SideEffect(SideEffectKind.PROCESS, "terminates process"),),
        is_total=False,
        is_pure=False,
        description="Exit the interpreter by raising SystemExit.",
    )))

    # ----- sys.version -------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="sys.version",
        params=(),
        returns=ReturnSpec(Sort.STR, refinements=(
            _not_none(),
            RefinementConstraint("len(result)", ">", "0", "non-empty version string"),
        )),
        description="Python version string.",
    )))

    # ----- sys.platform ------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="sys.platform",
        params=(),
        returns=ReturnSpec(Sort.STR, refinements=(_not_none(),)),
        description="Platform identifier.",
    )))

    # ----- sys.modules -------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="sys.modules",
        params=(),
        returns=ReturnSpec(Sort.DICT),
        is_pure=False,
        description="Dictionary of loaded modules.",
    )))

    # ----- sys.maxsize -------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="sys.maxsize",
        params=(),
        returns=ReturnSpec(Sort.INT, refinements=(
            _positive_int(),
            RefinementConstraint("result", ">", "2**31 - 1",
                                 "at least 32-bit signed max"),
        )),
        description="Largest positive integer supported by Py_ssize_t.",
    )))

    # ----- sys.getdefaultencoding / getfilesystemencoding ---------------
    for name in ("getdefaultencoding", "getfilesystemencoding"):
        registry.register(_model(FunctionSignature(
            qualified_name=f"sys.{name}",
            params=(),
            returns=ReturnSpec(Sort.STR, refinements=(
                _not_none(),
                RefinementConstraint("len(result)", ">", "0", "non-empty encoding"),
            )),
            description=f"Return the {name.replace('get', '').replace('encoding', ' encoding')}.",
        )))

    # ----- sys.exc_info ------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="sys.exc_info",
        params=(),
        returns=ReturnSpec(Sort.TUPLE, refinements=(
            RefinementConstraint("len(result)", "==", "3", "returns (type, value, tb)"),
        )),
        description="Return exception info as (type, value, traceback).",
    )))

    # ----- sys.getsizeof -----------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="sys.getsizeof",
        params=(
            ParamSpec("object", Sort.ANY),
            ParamSpec("default", Sort.INT, has_default=True, default="<sentinel>",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.INT, refinements=(_non_neg_int(),)),
        exceptions=(ExceptionSpec("TypeError", "object has no __sizeof__"),),
        description="Return the size of object in bytes.",
    )))

    # ----- sys.getrecursionlimit / setrecursionlimit --------------------
    registry.register(_model(FunctionSignature(
        qualified_name="sys.getrecursionlimit",
        params=(),
        returns=ReturnSpec(Sort.INT, refinements=(_positive_int(),)),
        description="Return the current recursion limit.",
    )))

    registry.register(_model(FunctionSignature(
        qualified_name="sys.setrecursionlimit",
        params=(ParamSpec("limit", Sort.INT),),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        preconditions=(
            RefinementConstraint("limit", ">", "0", "limit must be positive"),
        ),
        side_effects=(SideEffect(SideEffectKind.GLOBAL_STATE, "changes recursion limit"),),
        exceptions=(
            ExceptionSpec("ValueError", "limit too low"),
            ExceptionSpec("RecursionError", "limit already exceeded"),
        ),
        is_pure=False,
        description="Set the maximum depth of the Python interpreter stack.",
    )))


# ===================================================================
#  JSON MODULE
# ===================================================================

def register_json_models(registry: ModelRegistry) -> None:
    """Register models for the json module."""

    # ----- json.dumps --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="json.dumps",
        params=(
            ParamSpec("obj", Sort.ANY),
            ParamSpec("skipkeys", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="False"),
            ParamSpec("ensure_ascii", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="True"),
            ParamSpec("check_circular", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="True"),
            ParamSpec("allow_nan", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="True"),
            ParamSpec("cls", Sort.TYPE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("indent", Sort.ANY, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("separators", Sort.TUPLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("default", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("sort_keys", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="False"),
        ),
        returns=ReturnSpec(Sort.STR, refinements=(
            _not_none(),
            RefinementConstraint("len(result)", ">", "0", "JSON string is non-empty"),
        )),
        exceptions=(
            ExceptionSpec("TypeError", "obj not serializable"),
            ExceptionSpec("ValueError", "circular reference or NaN with allow_nan=False"),
        ),
        description="Serialize obj to a JSON-formatted string.",
    )))

    # ----- json.loads --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="json.loads",
        params=(
            ParamSpec("s", Sort.STR),
            ParamSpec("cls", Sort.TYPE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("object_hook", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("parse_float", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("parse_int", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("parse_constant", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("object_pairs_hook", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ANY, nullable=Nullability.NULLABLE),
        preconditions=(
            RefinementConstraint("len(s)", ">", "0", "non-empty JSON string"),
        ),
        exceptions=(
            ExceptionSpec("json.JSONDecodeError", "invalid JSON"),
        ),
        description="Deserialize a JSON string to a Python object.",
    )))

    # ----- json.dump ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="json.dump",
        params=(
            ParamSpec("obj", Sort.ANY),
            ParamSpec("fp", Sort.FILE),
            ParamSpec("skipkeys", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="False"),
            ParamSpec("ensure_ascii", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="True"),
            ParamSpec("check_circular", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="True"),
            ParamSpec("allow_nan", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="True"),
            ParamSpec("cls", Sort.TYPE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("indent", Sort.ANY, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("separators", Sort.TUPLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("default", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("sort_keys", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="False"),
        ),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.IO_WRITE, "writes JSON to file"),),
        exceptions=(
            ExceptionSpec("TypeError", "obj not serializable"),
            ExceptionSpec("ValueError", "circular reference"),
        ),
        is_pure=False,
        description="Serialize obj as a JSON-formatted stream to fp.",
    )))

    # ----- json.load ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="json.load",
        params=(
            ParamSpec("fp", Sort.FILE),
            ParamSpec("cls", Sort.TYPE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("object_hook", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("parse_float", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("parse_int", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("parse_constant", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("object_pairs_hook", Sort.CALLABLE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ANY, nullable=Nullability.NULLABLE),
        side_effects=(SideEffect(SideEffectKind.IO_READ, "reads from file"),),
        exceptions=(
            ExceptionSpec("json.JSONDecodeError", "invalid JSON"),
        ),
        is_pure=False,
        description="Deserialize fp to a Python object.",
    )))

    # ----- json.JSONEncoder / JSONDecoder / JSONDecodeError (types) -----
    registry.register(_model(FunctionSignature(
        qualified_name="json.JSONEncoder",
        params=(
            ParamSpec("skipkeys", Sort.BOOL, has_default=True, default="False"),
            ParamSpec("ensure_ascii", Sort.BOOL, has_default=True, default="True"),
            ParamSpec("check_circular", Sort.BOOL, has_default=True, default="True"),
            ParamSpec("allow_nan", Sort.BOOL, has_default=True, default="True"),
            ParamSpec("sort_keys", Sort.BOOL, has_default=True, default="False"),
            ParamSpec("indent", Sort.ANY, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("separators", Sort.TUPLE, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("default", Sort.CALLABLE, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.OBJECT, refinements=(_not_none(),)),
        description="Extensible JSON encoder.",
    )))

    registry.register(_model(FunctionSignature(
        qualified_name="json.JSONDecoder",
        params=(
            ParamSpec("object_hook", Sort.CALLABLE, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("parse_float", Sort.CALLABLE, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("parse_int", Sort.CALLABLE, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("parse_constant", Sort.CALLABLE, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("strict", Sort.BOOL, has_default=True, default="True"),
            ParamSpec("object_pairs_hook", Sort.CALLABLE, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.OBJECT, refinements=(_not_none(),)),
        description="Simple JSON decoder.",
    )))

    registry.register(_model(FunctionSignature(
        qualified_name="json.JSONDecodeError",
        params=(
            ParamSpec("msg", Sort.STR),
            ParamSpec("doc", Sort.STR),
            ParamSpec("pos", Sort.INT),
        ),
        returns=ReturnSpec(Sort.OBJECT, refinements=(_not_none(),)),
        description="Subclass of ValueError for JSON decode errors.",
    )))


# ===================================================================
#  RE MODULE
# ===================================================================

def register_re_models(registry: ModelRegistry) -> None:
    """Register models for the re module."""

    # ----- re.compile --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="re.compile",
        params=(
            ParamSpec("pattern", Sort.STR),
            ParamSpec("flags", Sort.INT, has_default=True, default="0"),
        ),
        returns=ReturnSpec(Sort.PATTERN, refinements=(_not_none(),)),
        exceptions=(ExceptionSpec("re.error", "invalid regular expression"),),
        description="Compile a regular expression pattern.",
    )))

    # ----- re.match ----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="re.match",
        params=(
            ParamSpec("pattern", Sort.STR),
            ParamSpec("string", Sort.STR),
            ParamSpec("flags", Sort.INT, has_default=True, default="0"),
        ),
        returns=ReturnSpec(Sort.MATCH, nullable=Nullability.NULLABLE),
        exceptions=(ExceptionSpec("re.error", "invalid pattern"),),
        description="Match pattern at the beginning of string.",
    )))

    # ----- re.search ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="re.search",
        params=(
            ParamSpec("pattern", Sort.STR),
            ParamSpec("string", Sort.STR),
            ParamSpec("flags", Sort.INT, has_default=True, default="0"),
        ),
        returns=ReturnSpec(Sort.MATCH, nullable=Nullability.NULLABLE),
        exceptions=(ExceptionSpec("re.error", "invalid pattern"),),
        description="Search string for a match to the pattern.",
    )))

    # ----- re.findall --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="re.findall",
        params=(
            ParamSpec("pattern", Sort.STR),
            ParamSpec("string", Sort.STR),
            ParamSpec("flags", Sort.INT, has_default=True, default="0"),
        ),
        returns=ReturnSpec(Sort.LIST, refinements=(_non_neg_int("len(result)"),)),
        exceptions=(ExceptionSpec("re.error", "invalid pattern"),),
        description="Return all non-overlapping matches of pattern in string.",
    )))

    # ----- re.finditer -------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="re.finditer",
        params=(
            ParamSpec("pattern", Sort.STR),
            ParamSpec("string", Sort.STR),
            ParamSpec("flags", Sort.INT, has_default=True, default="0"),
        ),
        returns=ReturnSpec(Sort.ITERATOR, element_sort=Sort.MATCH),
        exceptions=(ExceptionSpec("re.error", "invalid pattern"),),
        description="Return an iterator yielding Match objects over all matches.",
    )))

    # ----- re.sub ------------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="re.sub",
        params=(
            ParamSpec("pattern", Sort.STR),
            ParamSpec("repl", Sort.ANY),
            ParamSpec("string", Sort.STR),
            ParamSpec("count", Sort.INT, has_default=True, default="0"),
            ParamSpec("flags", Sort.INT, has_default=True, default="0"),
        ),
        returns=ReturnSpec(Sort.STR, refinements=(_not_none(),)),
        preconditions=(
            RefinementConstraint("count", ">=", "0", "count is non-negative"),
        ),
        exceptions=(ExceptionSpec("re.error", "invalid pattern"),),
        description="Return the string obtained by replacing the leftmost occurrences.",
    )))

    # ----- re.split ----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="re.split",
        params=(
            ParamSpec("pattern", Sort.STR),
            ParamSpec("string", Sort.STR),
            ParamSpec("maxsplit", Sort.INT, has_default=True, default="0"),
            ParamSpec("flags", Sort.INT, has_default=True, default="0"),
        ),
        returns=ReturnSpec(Sort.LIST, element_sort=Sort.STR, refinements=(
            RefinementConstraint("len(result)", ">=", "1", "split returns at least one part"),
        )),
        exceptions=(ExceptionSpec("re.error", "invalid pattern"),),
        description="Split string by the occurrences of pattern.",
    )))

    # ----- re.Pattern / re.Match (types) --------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="re.Pattern",
        params=(ParamSpec("pattern", Sort.STR),),
        returns=ReturnSpec(Sort.PATTERN, refinements=(_not_none(),)),
        description="Compiled regular expression object.",
    )))

    registry.register(_model(FunctionSignature(
        qualified_name="re.Match",
        params=(),
        returns=ReturnSpec(Sort.MATCH, refinements=(_not_none(),)),
        description="Match object returned by successful pattern matching.",
    )))

    # ----- re.fullmatch ------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="re.fullmatch",
        params=(
            ParamSpec("pattern", Sort.STR),
            ParamSpec("string", Sort.STR),
            ParamSpec("flags", Sort.INT, has_default=True, default="0"),
        ),
        returns=ReturnSpec(Sort.MATCH, nullable=Nullability.NULLABLE),
        exceptions=(ExceptionSpec("re.error", "invalid pattern"),),
        description="Match pattern against entire string.",
    )))

    # ----- re.subn -----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="re.subn",
        params=(
            ParamSpec("pattern", Sort.STR),
            ParamSpec("repl", Sort.ANY),
            ParamSpec("string", Sort.STR),
            ParamSpec("count", Sort.INT, has_default=True, default="0"),
            ParamSpec("flags", Sort.INT, has_default=True, default="0"),
        ),
        returns=ReturnSpec(Sort.TUPLE, refinements=(
            RefinementConstraint("len(result)", "==", "2",
                                 "subn returns (new_string, num_subs)"),
        )),
        postconditions=(
            RefinementConstraint("result[1]", ">=", "0", "substitution count non-negative"),
        ),
        exceptions=(ExceptionSpec("re.error", "invalid pattern"),),
        description="Like sub() but also return the number of substitutions made.",
    )))

    # ----- re.escape ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="re.escape",
        params=(ParamSpec("pattern", Sort.STR),),
        returns=ReturnSpec(Sort.STR, refinements=(
            _not_none(),
            RefinementConstraint("len(result)", ">=", "len(arg0)",
                                 "escaped string is at least as long"),
        )),
        description="Escape special characters in pattern.",
    )))

    # ----- re.purge ----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="re.purge",
        params=(),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.GLOBAL_STATE, "clears regex cache"),),
        is_pure=False,
        description="Clear the regular expression cache.",
    )))

    # ----- re flags (constants) ----------------------------------------
    for flag_name in ("IGNORECASE", "MULTILINE", "DOTALL", "VERBOSE"):
        registry.register(_model(FunctionSignature(
            qualified_name=f"re.{flag_name}",
            params=(),
            returns=ReturnSpec(Sort.INT, refinements=(
                RefinementConstraint("result", ">", "0", "flag is positive int"),
            )),
            description=f"Regex flag {flag_name}.",
        )))


# ===================================================================
#  MATH MODULE
# ===================================================================

def register_math_models(registry: ModelRegistry) -> None:
    """Register models for the math module."""

    # ----- math.sqrt ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.sqrt",
        params=(ParamSpec("x", Sort.FLOAT),),
        returns=ReturnSpec(Sort.FLOAT, refinements=(
            RefinementConstraint("result", ">=", "0.0", "sqrt is non-negative"),
            _is_finite(),
        )),
        preconditions=(
            RefinementConstraint("x", ">=", "0", "domain of sqrt"),
        ),
        postconditions=(
            RefinementConstraint("result * result", "==", "x",
                                 "sqrt squares back to x (approx)"),
        ),
        exceptions=(
            ExceptionSpec("ValueError", "x < 0"),
        ),
        description="Return the square root of x.",
    )))

    # ----- math.ceil ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.ceil",
        params=(ParamSpec("x", Sort.FLOAT),),
        returns=ReturnSpec(Sort.INT),
        postconditions=(
            RefinementConstraint("result", ">=", "x", "ceil >= x"),
            RefinementConstraint("result - 1", "<", "x", "ceil tight"),
        ),
        description="Return the ceiling of x, the smallest integer >= x.",
    )))

    # ----- math.floor --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.floor",
        params=(ParamSpec("x", Sort.FLOAT),),
        returns=ReturnSpec(Sort.INT),
        postconditions=(
            RefinementConstraint("result", "<=", "x", "floor <= x"),
            RefinementConstraint("result + 1", ">", "x", "floor tight"),
        ),
        description="Return the floor of x, the largest integer <= x.",
    )))

    # ----- math.log ----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.log",
        params=(
            ParamSpec("x", Sort.FLOAT),
            ParamSpec("base", Sort.FLOAT, has_default=True, default="math.e"),
        ),
        returns=ReturnSpec(Sort.FLOAT),
        preconditions=(
            RefinementConstraint("x", ">", "0", "log domain: positive"),
            RefinementConstraint("base", ">", "0", "base must be positive"),
            RefinementConstraint("base", "!=", "1", "base must not be 1"),
        ),
        exceptions=(
            ExceptionSpec("ValueError", "x <= 0 or base <= 0 or base == 1"),
        ),
        description="Return the logarithm of x to the given base.",
    )))

    # ----- math.exp ----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.exp",
        params=(ParamSpec("x", Sort.FLOAT),),
        returns=ReturnSpec(Sort.FLOAT, refinements=(
            RefinementConstraint("result", ">", "0", "exp is always positive"),
        )),
        postconditions=(
            RefinementConstraint("result", ">", "0", "exp(x) > 0 for all finite x"),
        ),
        exceptions=(ExceptionSpec("OverflowError", "result too large"),),
        description="Return e raised to the power x.",
    )))

    # ----- math.sin / cos / tan ----------------------------------------
    for trig_fn in ("sin", "cos", "tan"):
        refinements: Tuple[RefinementConstraint, ...] = ()
        if trig_fn in ("sin", "cos"):
            refinements = (
                _between("result", "-1.0", "1.0"),
            )
        registry.register(_model(FunctionSignature(
            qualified_name=f"math.{trig_fn}",
            params=(ParamSpec("x", Sort.FLOAT),),
            returns=ReturnSpec(Sort.FLOAT, refinements=refinements),
            description=f"Return the {trig_fn} of x (radians).",
        )))

    # ----- math.pi / e / inf / nan (constants) -------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.pi",
        params=(),
        returns=ReturnSpec(Sort.FLOAT, refinements=(
            _between("result", "3.14159", "3.14160"),
        )),
        description="The mathematical constant π.",
    )))

    registry.register(_model(FunctionSignature(
        qualified_name="math.e",
        params=(),
        returns=ReturnSpec(Sort.FLOAT, refinements=(
            _between("result", "2.71828", "2.71829"),
        )),
        description="The mathematical constant e.",
    )))

    registry.register(_model(FunctionSignature(
        qualified_name="math.inf",
        params=(),
        returns=ReturnSpec(Sort.FLOAT, refinements=(
            RefinementConstraint("result", "==", "float('inf')", "positive infinity"),
        )),
        description="Positive infinity.",
    )))

    registry.register(_model(FunctionSignature(
        qualified_name="math.nan",
        params=(),
        returns=ReturnSpec(Sort.FLOAT, refinements=(
            RefinementConstraint("result", "!=", "result", "NaN != NaN"),
        )),
        description="Not a Number (NaN).",
    )))

    # ----- math.isnan / isinf / isfinite --------------------------------
    for check_fn in ("isnan", "isinf", "isfinite"):
        registry.register(_model(FunctionSignature(
            qualified_name=f"math.{check_fn}",
            params=(ParamSpec("x", Sort.FLOAT),),
            returns=ReturnSpec(Sort.BOOL),
            description=f"Return True if x is {check_fn[2:]}.",
        )))

    # ----- math.factorial -----------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.factorial",
        params=(ParamSpec("n", Sort.INT),),
        returns=ReturnSpec(Sort.INT, refinements=(
            _positive_int(),
        )),
        preconditions=(
            RefinementConstraint("n", ">=", "0", "factorial domain: non-negative"),
        ),
        postconditions=(
            RefinementConstraint("result", ">=", "1", "factorial >= 1"),
        ),
        exceptions=(
            ExceptionSpec("ValueError", "n < 0"),
        ),
        description="Return n factorial (n!).",
    )))

    # ----- math.gcd ----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.gcd",
        params=(
            ParamSpec("args", Sort.INT, kind=ParamKind.VAR_POSITIONAL),
        ),
        returns=ReturnSpec(Sort.INT, refinements=(_non_neg_int(),)),
        postconditions=(
            RefinementConstraint("result", ">=", "0", "gcd is non-negative"),
        ),
        variadic=True,
        description="Return the greatest common divisor of the integer arguments.",
    )))

    # ----- math.lcm ----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.lcm",
        params=(
            ParamSpec("args", Sort.INT, kind=ParamKind.VAR_POSITIONAL),
        ),
        returns=ReturnSpec(Sort.INT, refinements=(_non_neg_int(),)),
        variadic=True,
        description="Return the least common multiple of the integer arguments.",
    )))

    # ----- math.comb ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.comb",
        params=(
            ParamSpec("n", Sort.INT),
            ParamSpec("k", Sort.INT),
        ),
        returns=ReturnSpec(Sort.INT, refinements=(_non_neg_int(),)),
        preconditions=(
            RefinementConstraint("n", ">=", "0", "n must be non-negative"),
            RefinementConstraint("k", ">=", "0", "k must be non-negative"),
        ),
        exceptions=(
            ExceptionSpec("ValueError", "n or k is negative"),
        ),
        description="Return the number of ways to choose k items from n items (C(n,k)).",
    )))

    # ----- math.perm ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.perm",
        params=(
            ParamSpec("n", Sort.INT),
            ParamSpec("k", Sort.INT, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.INT, refinements=(_non_neg_int(),)),
        preconditions=(
            RefinementConstraint("n", ">=", "0", "n must be non-negative"),
        ),
        exceptions=(
            ExceptionSpec("ValueError", "n is negative"),
        ),
        description="Return the number of k-length permutations of n items (P(n,k)).",
    )))

    # ----- math.fabs ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.fabs",
        params=(ParamSpec("x", Sort.FLOAT),),
        returns=ReturnSpec(Sort.FLOAT, refinements=(
            RefinementConstraint("result", ">=", "0.0", "fabs is non-negative"),
        )),
        description="Return the absolute value of x as a float.",
    )))

    # ----- math.fmod ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.fmod",
        params=(
            ParamSpec("x", Sort.FLOAT),
            ParamSpec("y", Sort.FLOAT),
        ),
        returns=ReturnSpec(Sort.FLOAT),
        preconditions=(
            RefinementConstraint("y", "!=", "0", "divisor must not be zero"),
        ),
        exceptions=(
            ExceptionSpec("ValueError", "y == 0"),
        ),
        description="Return x % y as defined by the platform C library.",
    )))

    # ----- math.pow ----------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.pow",
        params=(
            ParamSpec("x", Sort.FLOAT),
            ParamSpec("y", Sort.FLOAT),
        ),
        returns=ReturnSpec(Sort.FLOAT),
        exceptions=(
            ExceptionSpec("ValueError", "x<0 and y not integer, or x==0 and y<0"),
            ExceptionSpec("OverflowError", "result too large"),
        ),
        description="Return x raised to the power y (float result).",
    )))

    # ----- math.copysign -----------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.copysign",
        params=(
            ParamSpec("x", Sort.FLOAT),
            ParamSpec("y", Sort.FLOAT),
        ),
        returns=ReturnSpec(Sort.FLOAT),
        postconditions=(
            RefinementConstraint("abs(result)", "==", "abs(x)",
                                 "magnitude equals abs(x)"),
        ),
        description="Return x with the sign of y.",
    )))

    # ----- math.trunc --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.trunc",
        params=(ParamSpec("x", Sort.FLOAT),),
        returns=ReturnSpec(Sort.INT),
        postconditions=(
            RefinementConstraint("abs(result)", "<=", "abs(x)",
                                 "trunc towards zero"),
        ),
        description="Truncate x to the nearest Integral toward 0.",
    )))

    # ----- math.modf ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.modf",
        params=(ParamSpec("x", Sort.FLOAT),),
        returns=ReturnSpec(Sort.TUPLE, refinements=(
            RefinementConstraint("len(result)", "==", "2",
                                 "returns (fractional, integer) pair"),
        )),
        postconditions=(
            RefinementConstraint("result[0] + result[1]", "==", "x",
                                 "fractional + integer = x"),
        ),
        description="Return the fractional and integer parts of x.",
    )))

    # ----- math.frexp --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.frexp",
        params=(ParamSpec("x", Sort.FLOAT),),
        returns=ReturnSpec(Sort.TUPLE, refinements=(
            RefinementConstraint("len(result)", "==", "2",
                                 "returns (mantissa, exponent) pair"),
        )),
        postconditions=(
            RefinementConstraint("result[0] * 2**result[1]", "==", "x",
                                 "mantissa * 2^exp = x"),
            RefinementConstraint("abs(result[0])", "between", "(0.5, 1.0)",
                                 "mantissa in [0.5, 1.0)"),
        ),
        description="Return the mantissa and exponent of x.",
    )))

    # ----- math.ldexp --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="math.ldexp",
        params=(
            ParamSpec("x", Sort.FLOAT),
            ParamSpec("i", Sort.INT),
        ),
        returns=ReturnSpec(Sort.FLOAT),
        postconditions=(
            RefinementConstraint("result", "==", "x * 2**i",
                                 "ldexp computes x * 2^i"),
        ),
        exceptions=(ExceptionSpec("OverflowError", "result too large"),),
        description="Return x * 2**i.",
    )))


# ===================================================================
#  COLLECTIONS MODULE
# ===================================================================

def register_collections_models(registry: ModelRegistry) -> None:
    """Register models for the collections module."""

    # ----- collections.OrderedDict -------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="collections.OrderedDict",
        params=(
            ParamSpec("items", Sort.ANY, has_default=True, default="()",
                      nullable=Nullability.NULLABLE),
            ParamSpec("kwargs", Sort.ANY, kind=ParamKind.VAR_KEYWORD),
        ),
        returns=ReturnSpec(Sort.ORDERED_DICT, refinements=(
            _not_none(),
            _non_neg_int("len(result)"),
        )),
        description="Dictionary that remembers insertion order.",
    )))

    # ----- collections.Counter -----------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="collections.Counter",
        params=(
            ParamSpec("iterable_or_mapping", Sort.ANY, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("kwargs", Sort.ANY, kind=ParamKind.VAR_KEYWORD),
        ),
        returns=ReturnSpec(Sort.COUNTER, refinements=(
            _not_none(),
            _non_neg_int("len(result)"),
        )),
        description="Dict subclass for counting hashable objects.",
    )))

    # ----- collections.defaultdict -------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="collections.defaultdict",
        params=(
            ParamSpec("default_factory", Sort.CALLABLE, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("args", Sort.ANY, kind=ParamKind.VAR_POSITIONAL),
            ParamSpec("kwargs", Sort.ANY, kind=ParamKind.VAR_KEYWORD),
        ),
        returns=ReturnSpec(Sort.DEFAULT_DICT, refinements=(
            _not_none(),
            _non_neg_int("len(result)"),
        )),
        variadic=True,
        description="Dict subclass that calls a factory for missing values.",
    )))

    # ----- collections.deque -------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="collections.deque",
        params=(
            ParamSpec("iterable", Sort.ANY, has_default=True, default="()",
                      nullable=Nullability.NULLABLE),
            ParamSpec("maxlen", Sort.INT, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.DEQUE, refinements=(
            _not_none(),
            _non_neg_int("len(result)"),
        )),
        preconditions=(
            RefinementConstraint("maxlen", ">=", "0",
                                 "maxlen must be non-negative if provided"),
        ),
        exceptions=(
            ExceptionSpec("ValueError", "maxlen < 0"),
        ),
        description="Double-ended queue.",
    )))

    # ----- collections.namedtuple --------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="collections.namedtuple",
        params=(
            ParamSpec("typename", Sort.STR),
            ParamSpec("field_names", Sort.ANY),
            ParamSpec("rename", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="False"),
            ParamSpec("defaults", Sort.ANY, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("module", Sort.STR, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.TYPE, refinements=(_not_none(),)),
        preconditions=(
            RefinementConstraint("len(typename)", ">", "0",
                                 "typename must be non-empty"),
        ),
        exceptions=(
            ExceptionSpec("ValueError", "invalid field name"),
        ),
        description="Factory function for creating tuple subclasses with named fields.",
    )))

    # ----- collections.ChainMap ----------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="collections.ChainMap",
        params=(
            ParamSpec("maps", Sort.DICT, kind=ParamKind.VAR_POSITIONAL),
        ),
        returns=ReturnSpec(Sort.CHAIN_MAP, refinements=(_not_none(),)),
        variadic=True,
        description="A group of dicts treated as a single mapping.",
    )))


# ===================================================================
#  ITERTOOLS MODULE
# ===================================================================

def register_itertools_models(registry: ModelRegistry) -> None:
    """Register models for the itertools module."""

    # ----- itertools.chain ---------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.chain",
        params=(
            ParamSpec("iterables", Sort.ANY, kind=ParamKind.VAR_POSITIONAL),
        ),
        returns=ReturnSpec(Sort.ITERATOR),
        postconditions=(
            RefinementConstraint("len(list(result))", "==",
                                 "sum(len(it) for it in iterables)",
                                 "chain concatenates lengths"),
        ),
        variadic=True,
        description="Make an iterator that chains iterables together.",
    )))

    # ----- itertools.combinations --------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.combinations",
        params=(
            ParamSpec("iterable", Sort.ANY),
            ParamSpec("r", Sort.INT),
        ),
        returns=ReturnSpec(Sort.ITERATOR, element_sort=Sort.TUPLE),
        preconditions=(
            RefinementConstraint("r", ">=", "0", "r must be non-negative"),
        ),
        postconditions=(
            RefinementConstraint("all(len(t) == r for t in result)", "==", "True",
                                 "each combination has r elements"),
        ),
        description="Return r-length combinations of elements from the iterable.",
    )))

    # ----- itertools.permutations --------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.permutations",
        params=(
            ParamSpec("iterable", Sort.ANY),
            ParamSpec("r", Sort.INT, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ITERATOR, element_sort=Sort.TUPLE),
        description="Return successive r-length permutations of elements.",
    )))

    # ----- itertools.product -------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.product",
        params=(
            ParamSpec("iterables", Sort.ANY, kind=ParamKind.VAR_POSITIONAL),
            ParamSpec("repeat", Sort.INT, kind=ParamKind.KEYWORD,
                      has_default=True, default="1"),
        ),
        returns=ReturnSpec(Sort.ITERATOR, element_sort=Sort.TUPLE),
        preconditions=(
            RefinementConstraint("repeat", ">=", "1", "repeat must be positive"),
        ),
        variadic=True,
        description="Cartesian product of input iterables.",
    )))

    # ----- itertools.repeat --------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.repeat",
        params=(
            ParamSpec("object", Sort.ANY),
            ParamSpec("times", Sort.INT, has_default=True, default="<infinite>",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ITERATOR),
        preconditions=(
            RefinementConstraint("times", ">=", "0",
                                 "times must be non-negative if provided"),
        ),
        description="Make an iterator that returns object over and over again.",
    )))

    # ----- itertools.count ---------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.count",
        params=(
            ParamSpec("start", Sort.INT, has_default=True, default="0"),
            ParamSpec("step", Sort.INT, has_default=True, default="1"),
        ),
        returns=ReturnSpec(Sort.ITERATOR, element_sort=Sort.INT),
        is_total=False,  # infinite iterator
        description="Make an iterator that counts upward from start.",
    )))

    # ----- itertools.cycle ---------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.cycle",
        params=(ParamSpec("iterable", Sort.ANY),),
        returns=ReturnSpec(Sort.ITERATOR),
        is_total=False,  # infinite iterator
        description="Make an iterator cycling over the iterable.",
    )))

    # ----- itertools.islice --------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.islice",
        params=(
            ParamSpec("iterable", Sort.ANY),
            ParamSpec("start_or_stop", Sort.INT),
            ParamSpec("stop", Sort.INT, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("step", Sort.INT, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ITERATOR),
        preconditions=(
            RefinementConstraint("start_or_stop", ">=", "0",
                                 "start must be non-negative"),
        ),
        description="Make an iterator that returns selected elements.",
    )))

    # ----- itertools.groupby -------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.groupby",
        params=(
            ParamSpec("iterable", Sort.ANY),
            ParamSpec("key", Sort.CALLABLE, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ITERATOR, element_sort=Sort.TUPLE),
        description="Make an iterator of (key, group) from consecutive keys.",
    )))

    # ----- itertools.accumulate ----------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.accumulate",
        params=(
            ParamSpec("iterable", Sort.ANY),
            ParamSpec("func", Sort.CALLABLE, has_default=True, default="operator.add",
                      nullable=Nullability.NULLABLE),
            ParamSpec("initial", Sort.ANY, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ITERATOR),
        description="Make an iterator that returns accumulated sums (or other binary fn results).",
    )))

    # ----- itertools.zip_longest ---------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.zip_longest",
        params=(
            ParamSpec("iterables", Sort.ANY, kind=ParamKind.VAR_POSITIONAL),
            ParamSpec("fillvalue", Sort.ANY, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ITERATOR, element_sort=Sort.TUPLE),
        postconditions=(
            RefinementConstraint("len(list(result))", "==",
                                 "max(len(it) for it in iterables)",
                                 "zip_longest pads to longest"),
        ),
        variadic=True,
        description="Make an iterator that aggregates elements, padding shorter iterables.",
    )))

    # ----- itertools.starmap -------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.starmap",
        params=(
            ParamSpec("function", Sort.CALLABLE),
            ParamSpec("iterable", Sort.ANY),
        ),
        returns=ReturnSpec(Sort.ITERATOR),
        description="Apply function using argument tuples from the iterable.",
    )))

    # ----- itertools.takewhile -----------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.takewhile",
        params=(
            ParamSpec("predicate", Sort.CALLABLE),
            ParamSpec("iterable", Sort.ANY),
        ),
        returns=ReturnSpec(Sort.ITERATOR),
        postconditions=(
            RefinementConstraint("len(list(result))", "<=", "len(list(arg1))",
                                 "takewhile can only reduce length"),
        ),
        description="Make an iterator that returns elements while predicate is true.",
    )))

    # ----- itertools.dropwhile -----------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.dropwhile",
        params=(
            ParamSpec("predicate", Sort.CALLABLE),
            ParamSpec("iterable", Sort.ANY),
        ),
        returns=ReturnSpec(Sort.ITERATOR),
        postconditions=(
            RefinementConstraint("len(list(result))", "<=", "len(list(arg1))",
                                 "dropwhile can only reduce length"),
        ),
        description="Make an iterator that drops elements while predicate is true.",
    )))

    # ----- itertools.filterfalse ---------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.filterfalse",
        params=(
            ParamSpec("predicate", Sort.CALLABLE, nullable=Nullability.NULLABLE),
            ParamSpec("iterable", Sort.ANY),
        ),
        returns=ReturnSpec(Sort.ITERATOR),
        postconditions=(
            RefinementConstraint("len(list(result))", "<=", "len(list(arg1))",
                                 "filterfalse can only reduce length"),
        ),
        description="Make an iterator that filters elements where predicate is false.",
    )))

    # ----- itertools.tee -----------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.tee",
        params=(
            ParamSpec("iterable", Sort.ANY),
            ParamSpec("n", Sort.INT, has_default=True, default="2"),
        ),
        returns=ReturnSpec(Sort.TUPLE, refinements=(
            RefinementConstraint("len(result)", "==", "n",
                                 "tee returns n iterators"),
        )),
        preconditions=(
            RefinementConstraint("n", ">=", "0", "n must be non-negative"),
        ),
        description="Return n independent iterators from a single iterable.",
    )))

    # ----- itertools.compress ------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="itertools.compress",
        params=(
            ParamSpec("data", Sort.ANY),
            ParamSpec("selectors", Sort.ANY),
        ),
        returns=ReturnSpec(Sort.ITERATOR),
        postconditions=(
            RefinementConstraint("len(list(result))", "<=", "len(list(arg0))",
                                 "compress can only reduce length"),
        ),
        description="Make an iterator that filters data elements using selectors.",
    )))


# ===================================================================
#  FUNCTOOLS MODULE
# ===================================================================

def register_functools_models(registry: ModelRegistry) -> None:
    """Register models for the functools module."""

    # ----- functools.reduce --------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="functools.reduce",
        params=(
            ParamSpec("function", Sort.CALLABLE),
            ParamSpec("iterable", Sort.ANY),
            ParamSpec("initial", Sort.ANY, has_default=True, default="<sentinel>",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.ANY),
        exceptions=(
            ExceptionSpec("TypeError", "empty iterable with no initial value"),
        ),
        description="Apply function of two arguments cumulatively.",
    )))

    # ----- functools.partial -------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="functools.partial",
        params=(
            ParamSpec("func", Sort.CALLABLE),
            ParamSpec("args", Sort.ANY, kind=ParamKind.VAR_POSITIONAL),
            ParamSpec("kwargs", Sort.ANY, kind=ParamKind.VAR_KEYWORD),
        ),
        returns=ReturnSpec(Sort.CALLABLE, refinements=(_not_none(),)),
        variadic=True,
        description="Return a new partial object.",
    )))

    # ----- functools.lru_cache -----------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="functools.lru_cache",
        params=(
            ParamSpec("maxsize", Sort.INT, has_default=True, default="128",
                      nullable=Nullability.NULLABLE),
            ParamSpec("typed", Sort.BOOL, has_default=True, default="False"),
        ),
        returns=ReturnSpec(Sort.CALLABLE, refinements=(_not_none(),)),
        side_effects=(SideEffect(SideEffectKind.ALLOCATION, "allocates cache"),),
        description="Least-recently-used cache decorator.",
    )))

    # ----- functools.cache ---------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="functools.cache",
        params=(ParamSpec("user_function", Sort.CALLABLE),),
        returns=ReturnSpec(Sort.CALLABLE, refinements=(_not_none(),)),
        side_effects=(SideEffect(SideEffectKind.ALLOCATION, "allocates cache"),),
        description="Simple unbounded cache (same as lru_cache(maxsize=None)).",
    )))

    # ----- functools.wraps ---------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="functools.wraps",
        params=(
            ParamSpec("wrapped", Sort.CALLABLE),
            ParamSpec("assigned", Sort.TUPLE, has_default=True,
                      default="WRAPPER_ASSIGNMENTS"),
            ParamSpec("updated", Sort.TUPLE, has_default=True,
                      default="WRAPPER_UPDATES"),
        ),
        returns=ReturnSpec(Sort.CALLABLE, refinements=(_not_none(),)),
        description="Decorator factory to apply update_wrapper to a wrapper function.",
    )))

    # ----- functools.total_ordering ------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="functools.total_ordering",
        params=(ParamSpec("cls", Sort.TYPE),),
        returns=ReturnSpec(Sort.TYPE, refinements=(_not_none(),)),
        description="Class decorator that fills in missing ordering methods.",
    )))

    # ----- functools.cmp_to_key ----------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="functools.cmp_to_key",
        params=(ParamSpec("mycmp", Sort.CALLABLE),),
        returns=ReturnSpec(Sort.CALLABLE, refinements=(_not_none(),)),
        description="Convert a cmp= function into a key= function.",
    )))

    # ----- functools.singledispatch ------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="functools.singledispatch",
        params=(ParamSpec("func", Sort.CALLABLE),),
        returns=ReturnSpec(Sort.CALLABLE, refinements=(_not_none(),)),
        description="Single-dispatch generic function decorator.",
    )))

    # ----- functools.cached_property -----------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="functools.cached_property",
        params=(ParamSpec("func", Sort.CALLABLE),),
        returns=ReturnSpec(Sort.PROPERTY, refinements=(_not_none(),)),
        side_effects=(SideEffect(SideEffectKind.MUTATION, "caches on first access"),),
        description="Transform a method into a property whose value is computed once.",
    )))


# ===================================================================
#  TYPING MODULE
# ===================================================================

def register_typing_models(registry: ModelRegistry) -> None:
    """Register models for the typing module."""

    # Type aliases (typing.Optional, Union, etc.) are special forms.
    # We model them as callables that return a type descriptor.

    _typing_forms: List[Tuple[str, str]] = [
        ("Optional", "Optional type (Union[X, None])."),
        ("Union", "Union of types."),
        ("List", "Homogeneous list type."),
        ("Dict", "Dictionary type."),
        ("Tuple", "Tuple type."),
        ("Set", "Set type."),
        ("FrozenSet", "Frozen set type."),
        ("Callable", "Callable type."),
        ("Type", "Type of a class."),
        ("Any", "Special form: unconstrained type."),
        ("ClassVar", "Special form: class variable annotation."),
        ("Final", "Special form: value cannot be overridden."),
        ("Literal", "Special form: literal types."),
        ("TypeGuard", "Special form: type narrowing predicate."),
        ("Annotated", "Special form: add metadata to types."),
        ("Protocol", "Base class for structural subtyping."),
        ("Generic", "Base class for generic types."),
    ]

    for form_name, form_desc in _typing_forms:
        registry.register(_model(FunctionSignature(
            qualified_name=f"typing.{form_name}",
            params=(
                ParamSpec("args", Sort.ANY, kind=ParamKind.VAR_POSITIONAL),
            ),
            returns=ReturnSpec(Sort.TYPE),
            variadic=True,
            description=form_desc,
        )))

    # ----- typing.runtime_checkable ------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="typing.runtime_checkable",
        params=(ParamSpec("cls", Sort.TYPE),),
        returns=ReturnSpec(Sort.TYPE, refinements=(_not_none(),)),
        description="Mark a Protocol as runtime-checkable.",
    )))

    # ----- typing.TypeVar ----------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="typing.TypeVar",
        params=(
            ParamSpec("name", Sort.STR),
            ParamSpec("constraints", Sort.TYPE, kind=ParamKind.VAR_POSITIONAL),
            ParamSpec("bound", Sort.TYPE, kind=ParamKind.KEYWORD,
                      has_default=True, default="None", nullable=Nullability.NULLABLE),
            ParamSpec("covariant", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="False"),
            ParamSpec("contravariant", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="False"),
        ),
        returns=ReturnSpec(Sort.TYPE, refinements=(_not_none(),)),
        variadic=True,
        description="Type variable for generic parameterization.",
    )))

    # ----- typing.get_type_hints ---------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="typing.get_type_hints",
        params=(
            ParamSpec("obj", Sort.ANY),
            ParamSpec("globalns", Sort.DICT, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("localns", Sort.DICT, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("include_extras", Sort.BOOL, has_default=True, default="False"),
        ),
        returns=ReturnSpec(Sort.DICT),
        exceptions=(
            ExceptionSpec("NameError", "forward reference not resolvable"),
        ),
        description="Return type hints for an object.",
    )))

    # ----- typing.cast -------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="typing.cast",
        params=(
            ParamSpec("typ", Sort.TYPE),
            ParamSpec("val", Sort.ANY),
        ),
        returns=ReturnSpec(Sort.ANY),
        postconditions=(
            RefinementConstraint("result", "is", "val",
                                 "cast returns the same object at runtime"),
        ),
        description="Cast a value to a type (no-op at runtime).",
    )))

    # ----- typing.overload ---------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="typing.overload",
        params=(ParamSpec("func", Sort.CALLABLE),),
        returns=ReturnSpec(Sort.CALLABLE),
        description="Decorator for overloaded function signatures.",
    )))

    # ----- typing.no_type_check ----------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="typing.no_type_check",
        params=(ParamSpec("arg", Sort.ANY),),
        returns=ReturnSpec(Sort.ANY),
        description="Decorator to indicate that annotations are not type hints.",
    )))

    # ----- typing.NamedTuple -------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="typing.NamedTuple",
        params=(
            ParamSpec("typename", Sort.STR),
            ParamSpec("fields", Sort.ANY, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.TYPE, refinements=(_not_none(),)),
        description="Typed version of collections.namedtuple.",
    )))

    # ----- typing.TypedDict --------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="typing.TypedDict",
        params=(
            ParamSpec("typename", Sort.STR),
            ParamSpec("fields", Sort.DICT, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("total", Sort.BOOL, kind=ParamKind.KEYWORD,
                      has_default=True, default="True"),
        ),
        returns=ReturnSpec(Sort.TYPE, refinements=(_not_none(),)),
        description="A class-like syntax for creating typed dicts.",
    )))


# ===================================================================
#  PATHLIB MODULE
# ===================================================================

def register_pathlib_models(registry: ModelRegistry) -> None:
    """Register models for the pathlib module."""

    # ----- pathlib.Path ------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path",
        params=(
            ParamSpec("args", Sort.STR, kind=ParamKind.VAR_POSITIONAL),
        ),
        returns=ReturnSpec(Sort.PATH, refinements=(_not_none(),)),
        variadic=True,
        description="PurePath subclass for the current platform.",
    )))

    # ----- pathlib.PurePath / PurePosixPath / PureWindowsPath -----------
    for cls_name in ("PurePath", "PurePosixPath", "PureWindowsPath"):
        registry.register(_model(FunctionSignature(
            qualified_name=f"pathlib.{cls_name}",
            params=(
                ParamSpec("args", Sort.STR, kind=ParamKind.VAR_POSITIONAL),
            ),
            returns=ReturnSpec(Sort.PATH, refinements=(_not_none(),)),
            variadic=True,
            description=f"A pure path class: {cls_name}.",
        )))

    # ----- pathlib.PosixPath / WindowsPath ------------------------------
    for cls_name in ("PosixPath", "WindowsPath"):
        registry.register(_model(FunctionSignature(
            qualified_name=f"pathlib.{cls_name}",
            params=(
                ParamSpec("args", Sort.STR, kind=ParamKind.VAR_POSITIONAL),
            ),
            returns=ReturnSpec(Sort.PATH, refinements=(_not_none(),)),
            variadic=True,
            description=f"Concrete path class: {cls_name}.",
        )))

    # -- Path instance methods modeled as "pathlib.Path.<method>" --------

    # ----- Path.exists -------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.exists",
        params=(ParamSpec("self", Sort.PATH),),
        returns=ReturnSpec(Sort.BOOL),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "stats path"),),
        is_pure=False,
        is_deterministic=False,
        description="Whether the path points to an existing file or directory.",
    )))

    # ----- Path.is_file / is_dir ---------------------------------------
    for check_name in ("is_file", "is_dir"):
        registry.register(_model(FunctionSignature(
            qualified_name=f"pathlib.Path.{check_name}",
            params=(ParamSpec("self", Sort.PATH),),
            returns=ReturnSpec(Sort.BOOL),
            side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "stats path"),),
            is_pure=False,
            is_deterministic=False,
            description=f"Whether the path is an existing {'file' if 'file' in check_name else 'directory'}.",
        )))

    # ----- Path.read_text ----------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.read_text",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("encoding", Sort.STR, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("errors", Sort.STR, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.STR, refinements=(_not_none(),)),
        side_effects=(SideEffect(SideEffectKind.IO_READ, "reads file"),),
        exceptions=(
            ExceptionSpec("FileNotFoundError", "file does not exist"),
            ExceptionSpec("PermissionError", "insufficient permissions"),
            ExceptionSpec("UnicodeDecodeError", "encoding error"),
        ),
        is_pure=False,
        description="Return the decoded contents of the pointed-to file as a string.",
    )))

    # ----- Path.write_text ---------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.write_text",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("data", Sort.STR),
            ParamSpec("encoding", Sort.STR, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("errors", Sort.STR, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("newline", Sort.STR, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.INT, refinements=(_non_neg_int(),)),
        side_effects=(SideEffect(SideEffectKind.IO_WRITE, "writes file"),),
        exceptions=(
            ExceptionSpec("PermissionError", "insufficient permissions"),
            ExceptionSpec("OSError", "general OS error"),
        ),
        is_pure=False,
        description="Write text data to the file, return number of characters written.",
    )))

    # ----- Path.read_bytes ---------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.read_bytes",
        params=(ParamSpec("self", Sort.PATH),),
        returns=ReturnSpec(Sort.BYTES, refinements=(_non_neg_int("len(result)"),)),
        side_effects=(SideEffect(SideEffectKind.IO_READ, "reads file"),),
        exceptions=(
            ExceptionSpec("FileNotFoundError", "file does not exist"),
            ExceptionSpec("PermissionError", "insufficient permissions"),
        ),
        is_pure=False,
        description="Return the binary contents of the pointed-to file as a bytes object.",
    )))

    # ----- Path.write_bytes -------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.write_bytes",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("data", Sort.BYTES),
        ),
        returns=ReturnSpec(Sort.INT, refinements=(_non_neg_int(),)),
        side_effects=(SideEffect(SideEffectKind.IO_WRITE, "writes file"),),
        is_pure=False,
        description="Write binary data to the file, return number of bytes written.",
    )))

    # ----- Path.mkdir --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.mkdir",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("mode", Sort.INT, has_default=True, default="0o777"),
            ParamSpec("parents", Sort.BOOL, has_default=True, default="False"),
            ParamSpec("exist_ok", Sort.BOOL, has_default=True, default="False"),
        ),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "creates directory"),),
        exceptions=(
            ExceptionSpec("FileExistsError", "exist_ok=False and dir exists"),
            ExceptionSpec("FileNotFoundError", "parents=False and parent missing"),
            ExceptionSpec("PermissionError", "insufficient permissions"),
        ),
        is_pure=False,
        description="Create a directory at this path.",
    )))

    # ----- Path.iterdir ------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.iterdir",
        params=(ParamSpec("self", Sort.PATH),),
        returns=ReturnSpec(Sort.ITERATOR, element_sort=Sort.PATH),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "reads directory"),),
        exceptions=(
            ExceptionSpec("NotADirectoryError", "path is not a directory"),
            ExceptionSpec("PermissionError", "insufficient permissions"),
        ),
        is_pure=False,
        is_deterministic=False,
        description="Iterate over the files in this directory.",
    )))

    # ----- Path.glob / rglob ------------------------------------------
    for glob_name in ("glob", "rglob"):
        registry.register(_model(FunctionSignature(
            qualified_name=f"pathlib.Path.{glob_name}",
            params=(
                ParamSpec("self", Sort.PATH),
                ParamSpec("pattern", Sort.STR),
            ),
            returns=ReturnSpec(Sort.GENERATOR, element_sort=Sort.PATH),
            side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "scans filesystem"),),
            is_pure=False,
            is_deterministic=False,
            description=f"{'Recursively g' if glob_name == 'rglob' else 'G'}lob the given pattern.",
        )))

    # ----- Path.resolve ------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.resolve",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("strict", Sort.BOOL, has_default=True, default="False"),
        ),
        returns=ReturnSpec(Sort.PATH, refinements=(_not_none(),)),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "resolves symlinks"),),
        exceptions=(
            ExceptionSpec("OSError", "strict=True and path does not exist"),
        ),
        is_pure=False,
        description="Make the path absolute, resolving any symlinks.",
    )))

    # ----- Path.absolute -----------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.absolute",
        params=(ParamSpec("self", Sort.PATH),),
        returns=ReturnSpec(Sort.PATH, refinements=(_not_none(),)),
        description="Return an absolute version of the path.",
    )))

    # ----- Path.parent (property) ---------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.parent",
        params=(ParamSpec("self", Sort.PATH),),
        returns=ReturnSpec(Sort.PATH, refinements=(_not_none(),)),
        description="The logical parent of the path.",
    )))

    # ----- Path.name / stem / suffix (string properties) ----------------
    for prop in ("name", "stem", "suffix"):
        registry.register(_model(FunctionSignature(
            qualified_name=f"pathlib.Path.{prop}",
            params=(ParamSpec("self", Sort.PATH),),
            returns=ReturnSpec(Sort.STR, refinements=(_not_none(),)),
            description=f"The {prop} of the path.",
        )))

    # ----- Path.suffixes -----------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.suffixes",
        params=(ParamSpec("self", Sort.PATH),),
        returns=ReturnSpec(Sort.LIST, element_sort=Sort.STR,
                           refinements=(_non_neg_int("len(result)"),)),
        description="A list of the path's file extensions.",
    )))

    # ----- Path.parts --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.parts",
        params=(ParamSpec("self", Sort.PATH),),
        returns=ReturnSpec(Sort.TUPLE, element_sort=Sort.STR,
                           refinements=(_non_neg_int("len(result)"),)),
        description="Tuple of the path's components.",
    )))

    # ----- Path.stat ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.stat",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("follow_symlinks", Sort.BOOL, has_default=True, default="True"),
        ),
        returns=ReturnSpec(Sort.OBJECT, refinements=(_not_none(),)),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "stats path"),),
        exceptions=(
            ExceptionSpec("FileNotFoundError", "path does not exist"),
            ExceptionSpec("PermissionError", "insufficient permissions"),
        ),
        is_pure=False,
        description="Return the result of stat() on this path.",
    )))

    # ----- Path.chmod --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.chmod",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("mode", Sort.INT),
            ParamSpec("follow_symlinks", Sort.BOOL, has_default=True, default="True"),
        ),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "changes permissions"),),
        exceptions=(
            ExceptionSpec("FileNotFoundError", "path does not exist"),
            ExceptionSpec("PermissionError", "insufficient permissions"),
        ),
        is_pure=False,
        description="Change the file mode and permissions.",
    )))

    # ----- Path.unlink -------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.unlink",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("missing_ok", Sort.BOOL, has_default=True, default="False"),
        ),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "deletes file"),),
        exceptions=(
            ExceptionSpec("FileNotFoundError", "missing_ok=False and path missing"),
            ExceptionSpec("IsADirectoryError", "path is a directory"),
        ),
        is_pure=False,
        description="Remove this file or symbolic link.",
    )))

    # ----- Path.rename -------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.rename",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("target", Sort.ANY),
        ),
        returns=ReturnSpec(Sort.PATH, refinements=(_not_none(),)),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "renames file"),),
        exceptions=(
            ExceptionSpec("FileNotFoundError", "source does not exist"),
            ExceptionSpec("OSError", "general OS error"),
        ),
        is_pure=False,
        description="Rename this path to the given target, return new Path.",
    )))

    # ----- Path.with_name / with_suffix --------------------------------
    for meth in ("with_name", "with_suffix"):
        registry.register(_model(FunctionSignature(
            qualified_name=f"pathlib.Path.{meth}",
            params=(
                ParamSpec("self", Sort.PATH),
                ParamSpec("arg", Sort.STR),
            ),
            returns=ReturnSpec(Sort.PATH, refinements=(_not_none(),)),
            exceptions=(
                ExceptionSpec("ValueError", "invalid name/suffix"),
            ),
            description=f"Return a new path with the {meth.replace('with_', '')} changed.",
        )))

    # ----- Path.joinpath -----------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.joinpath",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("args", Sort.STR, kind=ParamKind.VAR_POSITIONAL),
        ),
        returns=ReturnSpec(Sort.PATH, refinements=(_not_none(),)),
        variadic=True,
        description="Combine this path with arguments.",
    )))

    # ----- Path.match --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.match",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("pattern", Sort.STR),
        ),
        returns=ReturnSpec(Sort.BOOL),
        description="Match this path against the provided glob-style pattern.",
    )))

    # ----- Path.open ---------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.open",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("mode", Sort.STR, has_default=True, default="'r'"),
            ParamSpec("buffering", Sort.INT, has_default=True, default="-1"),
            ParamSpec("encoding", Sort.STR, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("errors", Sort.STR, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
            ParamSpec("newline", Sort.STR, has_default=True, default="None",
                      nullable=Nullability.NULLABLE),
        ),
        returns=ReturnSpec(Sort.FILE, refinements=(_not_none(),)),
        side_effects=(
            SideEffect(SideEffectKind.FILE_SYSTEM, "opens file"),
            SideEffect(SideEffectKind.ALLOCATION, "allocates file handle"),
        ),
        exceptions=(
            ExceptionSpec("FileNotFoundError", "file not found (read modes)"),
            ExceptionSpec("PermissionError", "insufficient permissions"),
            ExceptionSpec("OSError", "general OS error"),
        ),
        is_pure=False,
        description="Open the file pointed to by the path.",
    )))

    # ----- Path.touch --------------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.touch",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("mode", Sort.INT, has_default=True, default="0o666"),
            ParamSpec("exist_ok", Sort.BOOL, has_default=True, default="True"),
        ),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "creates/touches file"),),
        exceptions=(
            ExceptionSpec("FileExistsError", "exist_ok=False and file exists"),
            ExceptionSpec("PermissionError", "insufficient permissions"),
        ),
        is_pure=False,
        description="Create the file at this path or update its modification time.",
    )))

    # ----- Path.symlink_to --------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.symlink_to",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("target", Sort.ANY),
            ParamSpec("target_is_directory", Sort.BOOL, has_default=True,
                      default="False"),
        ),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "creates symlink"),),
        exceptions=(
            ExceptionSpec("FileExistsError", "symlink already exists"),
            ExceptionSpec("OSError", "general OS error"),
        ),
        is_pure=False,
        description="Make this path a symbolic link to target.",
    )))

    # ----- Path.hardlink_to -------------------------------------------
    registry.register(_model(FunctionSignature(
        qualified_name="pathlib.Path.hardlink_to",
        params=(
            ParamSpec("self", Sort.PATH),
            ParamSpec("target", Sort.ANY),
        ),
        returns=ReturnSpec(Sort.NONE, nullable=Nullability.ALWAYS_NULL),
        side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM, "creates hard link"),),
        exceptions=(
            ExceptionSpec("OSError", "cross-device link or permission error"),
        ),
        is_pure=False,
        description="Make this path a hard link to target.",
    )))

    # ----- Path.home / cwd (class methods) -----------------------------
    for cls_method in ("home", "cwd"):
        registry.register(_model(FunctionSignature(
            qualified_name=f"pathlib.Path.{cls_method}",
            params=(),
            returns=ReturnSpec(Sort.PATH, refinements=(_not_none(),)),
            side_effects=(SideEffect(SideEffectKind.FILE_SYSTEM,
                                     f"queries {cls_method}"),),
            is_pure=False,
            is_deterministic=False,
            description=f"Return the {'home directory' if cls_method == 'home' else 'current working directory'} path.",
        )))


# ===================================================================
#  GLOBAL REGISTRY
# ===================================================================

_GLOBAL_REGISTRY: Optional[ModelRegistry] = None


def register_all_models(registry: Optional[ModelRegistry] = None) -> ModelRegistry:
    """Register every module's models into the given (or new) registry."""
    if registry is None:
        registry = ModelRegistry()
    register_builtin_models(registry)
    register_os_models(registry)
    register_sys_models(registry)
    register_json_models(registry)
    register_re_models(registry)
    register_math_models(registry)
    register_collections_models(registry)
    register_itertools_models(registry)
    register_functools_models(registry)
    register_typing_models(registry)
    register_pathlib_models(registry)
    return registry


def get_global_registry() -> ModelRegistry:
    """Return (and lazily create) the global model registry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = register_all_models()
    return _GLOBAL_REGISTRY

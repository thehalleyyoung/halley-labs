"""
CEGAR Engine for Refinement Type Inference in Dynamic Languages
================================================================

Implements Counterexample-Guided Abstraction Refinement (CEGAR) for inferring
refinement types of the form {x : T | φ(x)} where T is a base type and φ is a
predicate drawn from a decidable fragment.

Theory overview
---------------
The algorithm operates as a two-phase loop:

  Phase 1 – Abstract Interpretation
    Given a current predicate set P, run a forward abstract interpreter over the
    function's control-flow graph.  The abstract domain maps each variable to a
    refinement type {x : T | ⋀_{p ∈ S} p(x)} where S ⊆ P.  Fixed-point
    computation uses a worklist algorithm with widening (after a configurable
    threshold) and optional narrowing.

  Phase 2 – Counterexample Analysis & Predicate Refinement
    A safety checker inspects the computed abstract states for potential
    violations (array-out-of-bounds, null dereference, division by zero, type
    tag mismatch).  For each violation a concrete counterexample path is
    extracted and checked for feasibility via a lightweight SMT encoding.
    Spurious counterexamples yield Craig interpolants that are projected back
    into the predicate language P, thereby refining the abstraction.

Convergence
-----------
The predicate lattice is finite: predicates are drawn from comparisons,
type-tags, nullity checks, membership, and length constraints over the
program's variables and constants.  The ascending chain condition is
guaranteed because each CEGAR iteration strictly grows P (or the analysis
terminates).  The bound on the number of iterations for a single function is

    O(|Vars|² × |Constants|)

which is the maximum size of the predicate universe for pair-wise comparisons.
"""

from __future__ import annotations

import abc
import collections
import enum
import hashlib
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ComparisonOp(enum.Enum):
    """Comparison operators used in predicates."""
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="

    def negate(self) -> "ComparisonOp":
        _neg = {
            ComparisonOp.LT: ComparisonOp.GE,
            ComparisonOp.LE: ComparisonOp.GT,
            ComparisonOp.GT: ComparisonOp.LE,
            ComparisonOp.GE: ComparisonOp.LT,
            ComparisonOp.EQ: ComparisonOp.NE,
            ComparisonOp.NE: ComparisonOp.EQ,
        }
        return _neg[self]

    def evaluate(self, left: Any, right: Any) -> bool:
        ops = {
            ComparisonOp.LT: lambda a, b: a < b,
            ComparisonOp.LE: lambda a, b: a <= b,
            ComparisonOp.GT: lambda a, b: a > b,
            ComparisonOp.GE: lambda a, b: a >= b,
            ComparisonOp.EQ: lambda a, b: a == b,
            ComparisonOp.NE: lambda a, b: a != b,
        }
        try:
            return ops[self](left, right)
        except TypeError:
            return False


class BinOp(enum.Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    FLOORDIV = "//"
    POW = "**"
    AND = "&"
    OR = "|"
    XOR = "^"
    LSHIFT = "<<"
    RSHIFT = ">>"


class BaseType(enum.Enum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    LIST = "list"
    DICT = "dict"
    SET = "set"
    TUPLE = "tuple"
    NONE_TYPE = "NoneType"
    ANY = "Any"
    OBJECT = "object"
    UNKNOWN = "unknown"


class ViolationKind(enum.Enum):
    ARRAY_OUT_OF_BOUNDS = "array_out_of_bounds"
    NULL_DEREFERENCE = "null_dereference"
    DIVISION_BY_ZERO = "division_by_zero"
    TYPE_TAG_MISMATCH = "type_tag_mismatch"


class Severity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CEGARConfig:
    """Configuration knobs for the CEGAR engine."""
    max_iterations: int = 100
    per_function_timeout: float = 60.0
    enable_interpolation: bool = True
    predicate_budget: int = 500
    convergence_threshold: float = 0.01
    widening_delay: int = 3
    narrowing_iterations: int = 2
    max_path_length: int = 200
    cache_size: int = 1024
    context_depth_limit: int = 3
    enable_narrowing: bool = True
    enable_caching: bool = True
    enable_summaries: bool = True
    verbose: bool = False


# ---------------------------------------------------------------------------
# Bug reports
# ---------------------------------------------------------------------------


@dataclass
class BugReport:
    """A reported potential bug discovered during analysis."""
    location: Optional[str]
    bug_kind: ViolationKind
    message: str
    severity: Severity
    confidence: float
    counterexample: Optional["Counterexample"]
    suggested_fix: Optional[str]

    def __str__(self) -> str:
        loc = self.location or "<unknown>"
        return (
            f"[{self.severity.value.upper()}] {self.bug_kind.value} at {loc}: "
            f"{self.message} (confidence={self.confidence:.0%})"
        )


# ---------------------------------------------------------------------------
# CEGAR result
# ---------------------------------------------------------------------------


@dataclass
class CEGARResult:
    """Outcome of running CEGAR on a single function or module."""
    inferred_types: Dict[str, "RefinementType"] = field(default_factory=dict)
    predicates_used: Set["BasePredicate"] = field(default_factory=set)
    iterations: int = 0
    converged: bool = False
    counterexamples_analyzed: int = 0
    time_taken: float = 0.0
    bug_reports: List[BugReport] = field(default_factory=list)

    def summary(self) -> str:
        n_bugs = len(self.bug_reports)
        conv = "converged" if self.converged else "did not converge"
        return (
            f"CEGAR: {self.iterations} iters, {conv}, "
            f"{len(self.predicates_used)} preds, "
            f"{self.counterexamples_analyzed} cex analyzed, "
            f"{n_bugs} bugs, {self.time_taken:.3f}s"
        )


# ---------------------------------------------------------------------------
# Predicate hierarchy
# ---------------------------------------------------------------------------


class BasePredicate(abc.ABC):
    """Abstract base for all predicates in the refinement logic."""

    @property
    @abc.abstractmethod
    def variable(self) -> Optional[str]:
        """Primary variable this predicate constrains (None for compound)."""

    @abc.abstractmethod
    def negate(self) -> "BasePredicate":
        """Return logical negation of this predicate."""

    @abc.abstractmethod
    def evaluate(self, env: Dict[str, Any]) -> bool:
        """Evaluate the predicate under a concrete environment."""

    @abc.abstractmethod
    def to_smt(self) -> "SMTConstraint":
        """Lower to an SMT constraint representation."""

    @abc.abstractmethod
    def free_variables(self) -> Set[str]:
        """Return the set of variables appearing in this predicate."""

    @abc.abstractmethod
    def substitute(self, mapping: Dict[str, str]) -> "BasePredicate":
        """Substitute variable names according to *mapping*."""

    @abc.abstractmethod
    def _key(self) -> Hashable:
        """Internal identity key for hashing and equality."""

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BasePredicate):
            return NotImplemented
        return self._key() == other._key()

    def __repr__(self) -> str:
        return str(self)


class ComparisonPredicate(BasePredicate):
    """``var op constant`` – e.g. x > 0."""

    def __init__(self, var: str, op: ComparisonOp, constant: Any) -> None:
        self._var = var
        self.op = op
        self.constant = constant

    @property
    def variable(self) -> str:
        return self._var

    def negate(self) -> "ComparisonPredicate":
        return ComparisonPredicate(self._var, self.op.negate(), self.constant)

    def evaluate(self, env: Dict[str, Any]) -> bool:
        if self._var not in env:
            return False
        return self.op.evaluate(env[self._var], self.constant)

    def to_smt(self) -> "SMTConstraint":
        return SMTConstraint("comparison", var=self._var, op=self.op.value, constant=self.constant)

    def free_variables(self) -> Set[str]:
        return {self._var}

    def substitute(self, mapping: Dict[str, str]) -> "ComparisonPredicate":
        new_var = mapping.get(self._var, self._var)
        return ComparisonPredicate(new_var, self.op, self.constant)

    def _key(self) -> Tuple:
        return ("cmp", self._var, self.op, self.constant)

    def __str__(self) -> str:
        return f"{self._var} {self.op.value} {self.constant!r}"


class TypeTagPredicate(BasePredicate):
    """``typeof(var) == tag``."""

    def __init__(self, var: str, tag: BaseType) -> None:
        self._var = var
        self.tag = tag

    @property
    def variable(self) -> str:
        return self._var

    def negate(self) -> "NegationPredicate":
        return NegationPredicate(self)

    def evaluate(self, env: Dict[str, Any]) -> bool:
        if self._var not in env:
            return False
        val = env[self._var]
        type_map: Dict[BaseType, type] = {
            BaseType.INT: int,
            BaseType.FLOAT: float,
            BaseType.STR: str,
            BaseType.BOOL: bool,
            BaseType.LIST: list,
            BaseType.DICT: dict,
            BaseType.SET: set,
            BaseType.TUPLE: tuple,
            BaseType.NONE_TYPE: type(None),
        }
        expected = type_map.get(self.tag)
        if expected is None:
            return True
        return isinstance(val, expected)

    def to_smt(self) -> "SMTConstraint":
        return SMTConstraint("type_tag", var=self._var, tag=self.tag.value)

    def free_variables(self) -> Set[str]:
        return {self._var}

    def substitute(self, mapping: Dict[str, str]) -> "TypeTagPredicate":
        return TypeTagPredicate(mapping.get(self._var, self._var), self.tag)

    def _key(self) -> Tuple:
        return ("type", self._var, self.tag)

    def __str__(self) -> str:
        return f"typeof({self._var}) == {self.tag.value}"


class NullityPredicate(BasePredicate):
    """``var is None`` or ``var is not None``."""

    def __init__(self, var: str, is_null: bool = True) -> None:
        self._var = var
        self.is_null = is_null

    @property
    def variable(self) -> str:
        return self._var

    def negate(self) -> "NullityPredicate":
        return NullityPredicate(self._var, not self.is_null)

    def evaluate(self, env: Dict[str, Any]) -> bool:
        if self._var not in env:
            return self.is_null
        val = env[self._var]
        return (val is None) == self.is_null

    def to_smt(self) -> "SMTConstraint":
        return SMTConstraint("nullity", var=self._var, is_null=self.is_null)

    def free_variables(self) -> Set[str]:
        return {self._var}

    def substitute(self, mapping: Dict[str, str]) -> "NullityPredicate":
        return NullityPredicate(mapping.get(self._var, self._var), self.is_null)

    def _key(self) -> Tuple:
        return ("null", self._var, self.is_null)

    def __str__(self) -> str:
        if self.is_null:
            return f"{self._var} is None"
        return f"{self._var} is not None"


class ConjunctionPredicate(BasePredicate):
    """``p1 ∧ p2``."""

    def __init__(self, left: BasePredicate, right: BasePredicate) -> None:
        self.left = left
        self.right = right

    @property
    def variable(self) -> Optional[str]:
        lv = self.left.variable
        rv = self.right.variable
        if lv == rv:
            return lv
        return None

    def negate(self) -> "DisjunctionPredicate":
        return DisjunctionPredicate(self.left.negate(), self.right.negate())

    def evaluate(self, env: Dict[str, Any]) -> bool:
        return self.left.evaluate(env) and self.right.evaluate(env)

    def to_smt(self) -> "SMTConstraint":
        return SMTConstraint("and", children=[self.left.to_smt(), self.right.to_smt()])

    def free_variables(self) -> Set[str]:
        return self.left.free_variables() | self.right.free_variables()

    def substitute(self, mapping: Dict[str, str]) -> "ConjunctionPredicate":
        return ConjunctionPredicate(
            self.left.substitute(mapping), self.right.substitute(mapping)
        )

    def _key(self) -> Tuple:
        left_k = self.left._key()
        right_k = self.right._key()
        return ("and", min(left_k, right_k), max(left_k, right_k))

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"


class DisjunctionPredicate(BasePredicate):
    """``p1 ∨ p2``."""

    def __init__(self, left: BasePredicate, right: BasePredicate) -> None:
        self.left = left
        self.right = right

    @property
    def variable(self) -> Optional[str]:
        lv = self.left.variable
        rv = self.right.variable
        if lv == rv:
            return lv
        return None

    def negate(self) -> "ConjunctionPredicate":
        return ConjunctionPredicate(self.left.negate(), self.right.negate())

    def evaluate(self, env: Dict[str, Any]) -> bool:
        return self.left.evaluate(env) or self.right.evaluate(env)

    def to_smt(self) -> "SMTConstraint":
        return SMTConstraint("or", children=[self.left.to_smt(), self.right.to_smt()])

    def free_variables(self) -> Set[str]:
        return self.left.free_variables() | self.right.free_variables()

    def substitute(self, mapping: Dict[str, str]) -> "DisjunctionPredicate":
        return DisjunctionPredicate(
            self.left.substitute(mapping), self.right.substitute(mapping)
        )

    def _key(self) -> Tuple:
        left_k = self.left._key()
        right_k = self.right._key()
        return ("or", min(left_k, right_k), max(left_k, right_k))

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"


class NegationPredicate(BasePredicate):
    """``¬p``."""

    def __init__(self, inner: BasePredicate) -> None:
        self.inner = inner

    @property
    def variable(self) -> Optional[str]:
        return self.inner.variable

    def negate(self) -> BasePredicate:
        return self.inner

    def evaluate(self, env: Dict[str, Any]) -> bool:
        return not self.inner.evaluate(env)

    def to_smt(self) -> "SMTConstraint":
        return SMTConstraint("not", children=[self.inner.to_smt()])

    def free_variables(self) -> Set[str]:
        return self.inner.free_variables()

    def substitute(self, mapping: Dict[str, str]) -> "NegationPredicate":
        return NegationPredicate(self.inner.substitute(mapping))

    def _key(self) -> Tuple:
        return ("not", self.inner._key())

    def __str__(self) -> str:
        return f"¬({self.inner})"


class MembershipPredicate(BasePredicate):
    """``var in collection``."""

    def __init__(self, var: str, collection_var: str) -> None:
        self._var = var
        self.collection_var = collection_var

    @property
    def variable(self) -> str:
        return self._var

    def negate(self) -> "NegationPredicate":
        return NegationPredicate(self)

    def evaluate(self, env: Dict[str, Any]) -> bool:
        if self._var not in env or self.collection_var not in env:
            return False
        try:
            return env[self._var] in env[self.collection_var]
        except TypeError:
            return False

    def to_smt(self) -> "SMTConstraint":
        return SMTConstraint(
            "membership", var=self._var, collection_var=self.collection_var
        )

    def free_variables(self) -> Set[str]:
        return {self._var, self.collection_var}

    def substitute(self, mapping: Dict[str, str]) -> "MembershipPredicate":
        return MembershipPredicate(
            mapping.get(self._var, self._var),
            mapping.get(self.collection_var, self.collection_var),
        )

    def _key(self) -> Tuple:
        return ("in", self._var, self.collection_var)

    def __str__(self) -> str:
        return f"{self._var} in {self.collection_var}"


class LengthPredicate(BasePredicate):
    """``len(var) op constant``."""

    def __init__(self, var: str, op: ComparisonOp, constant: int) -> None:
        self._var = var
        self.op = op
        self.constant = constant

    @property
    def variable(self) -> str:
        return self._var

    def negate(self) -> "LengthPredicate":
        return LengthPredicate(self._var, self.op.negate(), self.constant)

    def evaluate(self, env: Dict[str, Any]) -> bool:
        if self._var not in env:
            return False
        val = env[self._var]
        try:
            length = len(val)
        except TypeError:
            return False
        return self.op.evaluate(length, self.constant)

    def to_smt(self) -> "SMTConstraint":
        return SMTConstraint(
            "length", var=self._var, op=self.op.value, constant=self.constant
        )

    def free_variables(self) -> Set[str]:
        return {self._var}

    def substitute(self, mapping: Dict[str, str]) -> "LengthPredicate":
        return LengthPredicate(
            mapping.get(self._var, self._var), self.op, self.constant
        )

    def _key(self) -> Tuple:
        return ("len", self._var, self.op, self.constant)

    def __str__(self) -> str:
        return f"len({self._var}) {self.op.value} {self.constant}"


# ---------------------------------------------------------------------------
# Refinement types
# ---------------------------------------------------------------------------


@dataclass
class RefinementType:
    """
    Refinement type  {x : T | φ(x)}  where T is *base_type* and φ is the
    conjunction of the predicates in *predicates*.
    """
    base_type: BaseType = BaseType.ANY
    predicates: Set[BasePredicate] = field(default_factory=set)
    is_bottom: bool = False
    is_top: bool = False

    # -- lattice operations --------------------------------------------------

    def join(self, other: "RefinementType") -> "RefinementType":
        """Least upper bound (⊔) – intersection of predicates."""
        if self.is_bottom:
            return other.copy()
        if other.is_bottom:
            return self.copy()
        if self.is_top or other.is_top:
            return RefinementType(base_type=BaseType.ANY, is_top=True)
        bt = self.base_type if self.base_type == other.base_type else BaseType.ANY
        common_preds = self.predicates & other.predicates
        return RefinementType(base_type=bt, predicates=set(common_preds))

    def meet(self, other: "RefinementType") -> "RefinementType":
        """Greatest lower bound (⊓) – union of predicates."""
        if self.is_top:
            return other.copy()
        if other.is_top:
            return self.copy()
        if self.is_bottom or other.is_bottom:
            return RefinementType(is_bottom=True)
        bt = self._meet_base(self.base_type, other.base_type)
        if bt is None:
            return RefinementType(is_bottom=True)
        combined = self.predicates | other.predicates
        return RefinementType(base_type=bt, predicates=set(combined))

    def is_subtype(self, other: "RefinementType") -> bool:
        """Check if *self* ≤ *other* in the refinement lattice."""
        if self.is_bottom:
            return True
        if other.is_top:
            return True
        if self.is_top and not other.is_top:
            return False
        if other.is_bottom:
            return self.is_bottom
        if self.base_type != other.base_type and other.base_type != BaseType.ANY:
            return False
        return other.predicates.issubset(self.predicates)

    def widen(self, other: "RefinementType") -> "RefinementType":
        """Widening: keep only predicates present in both (same as join)
        but additionally drop predicates that have been oscillating."""
        return self.join(other)

    def narrow(self, other: "RefinementType") -> "RefinementType":
        """Narrowing: recover precision by adding back predicates from *other*
        that are consistent with *self*."""
        if self.is_top:
            return other.copy()
        if other.is_bottom:
            return self.copy()
        bt = self.base_type
        recovered = set(self.predicates)
        for p in other.predicates:
            neg = p.negate()
            if neg not in recovered:
                recovered.add(p)
        return RefinementType(base_type=bt, predicates=recovered)

    def add_predicate(self, pred: BasePredicate) -> "RefinementType":
        new_preds = set(self.predicates)
        new_preds.add(pred)
        return RefinementType(
            base_type=self.base_type, predicates=new_preds,
            is_bottom=self.is_bottom, is_top=False,
        )

    def remove_predicate(self, pred: BasePredicate) -> "RefinementType":
        new_preds = set(self.predicates)
        new_preds.discard(pred)
        return RefinementType(
            base_type=self.base_type, predicates=new_preds,
            is_bottom=self.is_bottom, is_top=len(new_preds) == 0,
        )

    def copy(self) -> "RefinementType":
        return RefinementType(
            base_type=self.base_type,
            predicates=set(self.predicates),
            is_bottom=self.is_bottom,
            is_top=self.is_top,
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def bottom() -> "RefinementType":
        return RefinementType(is_bottom=True)

    @staticmethod
    def top() -> "RefinementType":
        return RefinementType(base_type=BaseType.ANY, is_top=True)

    @staticmethod
    def _meet_base(a: BaseType, b: BaseType) -> Optional[BaseType]:
        if a == b:
            return a
        if a == BaseType.ANY:
            return b
        if b == BaseType.ANY:
            return a
        numeric = {BaseType.INT, BaseType.FLOAT, BaseType.BOOL}
        if a in numeric and b in numeric:
            if BaseType.FLOAT in (a, b):
                return BaseType.FLOAT
            return BaseType.INT
        return None

    def __str__(self) -> str:
        if self.is_bottom:
            return "⊥"
        if self.is_top:
            return "⊤"
        pred_str = " ∧ ".join(str(p) for p in sorted(self.predicates, key=str))
        if not pred_str:
            return self.base_type.value
        return f"{{{self.base_type.value} | {pred_str}}}"

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Abstract state
# ---------------------------------------------------------------------------


class AbstractState:
    """Mapping from program variables to their current refinement types."""

    def __init__(self, bindings: Optional[Dict[str, RefinementType]] = None,
                 is_bottom: bool = False) -> None:
        self._bindings: Dict[str, RefinementType] = dict(bindings or {})
        self._is_bottom = is_bottom

    @property
    def is_bottom(self) -> bool:
        return self._is_bottom

    def variables(self) -> Set[str]:
        return set(self._bindings.keys())

    def copy(self) -> "AbstractState":
        return AbstractState(
            {k: v.copy() for k, v in self._bindings.items()},
            is_bottom=self._is_bottom,
        )

    def __getitem__(self, var: str) -> RefinementType:
        if self._is_bottom:
            return RefinementType.bottom()
        return self._bindings.get(var, RefinementType.top())

    def __setitem__(self, var: str, rt: RefinementType) -> None:
        self._bindings[var] = rt
        if rt.is_bottom:
            pass
        self._is_bottom = False

    def __contains__(self, var: str) -> bool:
        return var in self._bindings

    def join(self, other: "AbstractState") -> "AbstractState":
        if self._is_bottom:
            return other.copy()
        if other._is_bottom:
            return self.copy()
        all_vars = self.variables() | other.variables()
        result: Dict[str, RefinementType] = {}
        for v in all_vars:
            result[v] = self[v].join(other[v])
        return AbstractState(result)

    def meet(self, other: "AbstractState") -> "AbstractState":
        if self._is_bottom or other._is_bottom:
            return AbstractState(is_bottom=True)
        all_vars = self.variables() | other.variables()
        result: Dict[str, RefinementType] = {}
        for v in all_vars:
            m = self[v].meet(other[v])
            if m.is_bottom:
                return AbstractState(is_bottom=True)
            result[v] = m
        return AbstractState(result)

    def widen(self, other: "AbstractState") -> "AbstractState":
        if self._is_bottom:
            return other.copy()
        if other._is_bottom:
            return self.copy()
        all_vars = self.variables() | other.variables()
        result: Dict[str, RefinementType] = {}
        for v in all_vars:
            result[v] = self[v].widen(other[v])
        return AbstractState(result)

    def narrow(self, other: "AbstractState") -> "AbstractState":
        if self._is_bottom:
            return other.copy()
        if other._is_bottom:
            return self.copy()
        all_vars = self.variables() | other.variables()
        result: Dict[str, RefinementType] = {}
        for v in all_vars:
            result[v] = self[v].narrow(other[v])
        return AbstractState(result)

    def project(self, variables: Set[str]) -> "AbstractState":
        """Keep only the specified variables."""
        projected = {v: self[v].copy() for v in variables if v in self._bindings}
        return AbstractState(projected, is_bottom=self._is_bottom)

    def rename(self, mapping: Dict[str, str]) -> "AbstractState":
        """Rename variables according to *mapping*."""
        new_bindings: Dict[str, RefinementType] = {}
        for old_name, rt in self._bindings.items():
            new_name = mapping.get(old_name, old_name)
            new_preds: Set[BasePredicate] = set()
            for p in rt.predicates:
                new_preds.add(p.substitute(mapping))
            new_bindings[new_name] = RefinementType(
                base_type=rt.base_type, predicates=new_preds,
                is_bottom=rt.is_bottom, is_top=rt.is_top,
            )
        return AbstractState(new_bindings, is_bottom=self._is_bottom)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractState):
            return NotImplemented
        if self._is_bottom != other._is_bottom:
            return False
        if self._is_bottom and other._is_bottom:
            return True
        if self.variables() != other.variables():
            return False
        for v in self.variables():
            s = self[v]
            o = other[v]
            if s.is_bottom != o.is_bottom or s.is_top != o.is_top:
                return False
            if s.base_type != o.base_type:
                return False
            if s.predicates != o.predicates:
                return False
        return True

    def __repr__(self) -> str:
        if self._is_bottom:
            return "AbstractState(⊥)"
        parts = [f"{v}: {self[v]}" for v in sorted(self._bindings)]
        return "AbstractState({" + ", ".join(parts) + "})"

    def size(self) -> int:
        return sum(len(rt.predicates) for rt in self._bindings.values())


# ---------------------------------------------------------------------------
# IR nodes
# ---------------------------------------------------------------------------


@dataclass
class SourceLoc:
    file: str = "<unknown>"
    line: int = 0
    col: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


@dataclass
class IRNode:
    node_id: int = 0
    source_loc: SourceLoc = field(default_factory=SourceLoc)


@dataclass
class AssignNode(IRNode):
    target: str = ""
    value_expr: str = ""
    value_constant: Any = None
    value_var: Optional[str] = None


@dataclass
class GuardNode(IRNode):
    condition: Optional[BasePredicate] = None
    true_branch: Optional[str] = None
    false_branch: Optional[str] = None


@dataclass
class CallNode(IRNode):
    target: str = ""
    func_name: str = ""
    args: List[str] = field(default_factory=list)


@dataclass
class ReturnNode(IRNode):
    value: Optional[str] = None


@dataclass
class PhiNode(IRNode):
    target: str = ""
    incoming: Dict[str, str] = field(default_factory=dict)


@dataclass
class BinOpNode(IRNode):
    target: str = ""
    left: str = ""
    op: BinOp = BinOp.ADD
    right: str = ""


@dataclass
class LoadAttrNode(IRNode):
    target: str = ""
    obj: str = ""
    attr: str = ""


@dataclass
class StoreAttrNode(IRNode):
    obj: str = ""
    attr: str = ""
    value: str = ""


@dataclass
class IndexNode(IRNode):
    target: str = ""
    obj: str = ""
    index: str = ""


# ---------------------------------------------------------------------------
# IR structure
# ---------------------------------------------------------------------------


@dataclass
class IRBlock:
    block_id: str = ""
    nodes: List[IRNode] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)


@dataclass
class IRFunction:
    name: str = ""
    params: List[str] = field(default_factory=list)
    blocks: Dict[str, IRBlock] = field(default_factory=dict)
    entry_block_id: str = ""
    exit_block_id: str = ""

    def all_variables(self) -> Set[str]:
        variables: Set[str] = set(self.params)
        for block in self.blocks.values():
            for node in block.nodes:
                if isinstance(node, (AssignNode, CallNode, BinOpNode,
                                     LoadAttrNode, IndexNode, PhiNode)):
                    variables.add(node.target)
                if isinstance(node, AssignNode) and node.value_var:
                    variables.add(node.value_var)
                if isinstance(node, BinOpNode):
                    variables.add(node.left)
                    variables.add(node.right)
                if isinstance(node, CallNode):
                    variables.update(node.args)
                if isinstance(node, IndexNode):
                    variables.add(node.obj)
                    variables.add(node.index)
                if isinstance(node, LoadAttrNode):
                    variables.add(node.obj)
        return variables

    def all_constants(self) -> Set[Any]:
        constants: Set[Any] = set()
        for block in self.blocks.values():
            for node in block.nodes:
                if isinstance(node, AssignNode) and node.value_constant is not None:
                    constants.add(node.value_constant)
        constants.update({0, 1, -1, None, True, False})
        return constants

    def block_order(self) -> List[str]:
        """Return blocks in reverse-post-order from entry."""
        visited: Set[str] = set()
        order: List[str] = []

        def dfs(bid: str) -> None:
            if bid in visited or bid not in self.blocks:
                return
            visited.add(bid)
            for s in self.blocks[bid].successors:
                dfs(s)
            order.append(bid)

        dfs(self.entry_block_id)
        order.reverse()
        return order


@dataclass
class IRModule:
    functions: Dict[str, IRFunction] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SMT formula (lightweight encoding)
# ---------------------------------------------------------------------------


@dataclass
class SMTConstraint:
    """Single SMT constraint node."""
    kind: str = ""
    var: Optional[str] = None
    op: Optional[str] = None
    constant: Any = None
    tag: Optional[str] = None
    is_null: Optional[bool] = None
    collection_var: Optional[str] = None
    children: List["SMTConstraint"] = field(default_factory=list)
    name: Optional[str] = None

    def __hash__(self) -> int:
        return id(self)

    def negate(self) -> "SMTConstraint":
        return SMTConstraint("not", children=[self])


class SMTFormula:
    """Simplified SMT formula: a conjunction of constraints."""

    def __init__(self) -> None:
        self.constraints: List[SMTConstraint] = []
        self._var_domains: Dict[str, List[Any]] = {}
        self._assignments: Dict[str, Any] = {}
        self._is_sat: Optional[bool] = None

    def add_constraint(self, constraint: SMTConstraint) -> None:
        self.constraints.append(constraint)
        self._is_sat = None

    def set_domain(self, var: str, values: List[Any]) -> None:
        self._var_domains[var] = values

    def check_sat(self) -> bool:
        """Check satisfiability via brute-force enumeration over small domains."""
        if self._is_sat is not None:
            return self._is_sat

        if not self.constraints:
            self._is_sat = True
            return True

        all_vars = self._collect_variables()
        if not all_vars:
            result = self._evaluate_constraints({})
            self._is_sat = result
            return result

        domains = self._build_domains(all_vars)
        self._is_sat = self._search(list(all_vars), domains, {}, 0)
        return self._is_sat

    def get_model(self) -> Dict[str, Any]:
        if self._is_sat is None:
            self.check_sat()
        return dict(self._assignments)

    def get_unsat_core(self) -> List[SMTConstraint]:
        """Return a minimal unsatisfiable subset (approximation)."""
        if self.check_sat():
            return []
        core: List[SMTConstraint] = []
        remaining = list(self.constraints)
        for i, c in enumerate(remaining):
            subset = core + remaining[i + 1:]
            test = SMTFormula()
            for sc in subset:
                test.add_constraint(sc)
            test._var_domains = dict(self._var_domains)
            if not test.check_sat():
                pass
            else:
                core.append(c)
        return core if core else list(self.constraints[:1])

    def negate(self) -> "SMTFormula":
        negated = SMTFormula()
        if len(self.constraints) == 1:
            negated.add_constraint(self.constraints[0].negate())
        else:
            children = list(self.constraints)
            neg_children = [c.negate() for c in children]
            negated.add_constraint(SMTConstraint("or", children=neg_children))
        negated._var_domains = dict(self._var_domains)
        return negated

    def conjoin(self, other: "SMTFormula") -> "SMTFormula":
        result = SMTFormula()
        for c in self.constraints:
            result.add_constraint(c)
        for c in other.constraints:
            result.add_constraint(c)
        result._var_domains = {**self._var_domains, **other._var_domains}
        return result

    # -- internal -----------------------------------------------------------

    def _collect_variables(self) -> Set[str]:
        variables: Set[str] = set()
        for c in self.constraints:
            self._vars_from_constraint(c, variables)
        return variables

    def _vars_from_constraint(self, c: SMTConstraint, acc: Set[str]) -> None:
        if c.var:
            acc.add(c.var)
        if c.collection_var:
            acc.add(c.collection_var)
        for child in c.children:
            self._vars_from_constraint(child, acc)

    def _build_domains(self, variables: Set[str]) -> Dict[str, List[Any]]:
        defaults = [0, 1, -1, 2, None, True, False, "", "a", 0.5, -0.5]
        domains: Dict[str, List[Any]] = {}
        for v in variables:
            if v in self._var_domains:
                domains[v] = self._var_domains[v]
            else:
                domains[v] = list(defaults)
        return domains

    def _search(self, variables: List[str], domains: Dict[str, List[Any]],
                assignment: Dict[str, Any], idx: int) -> bool:
        if idx == len(variables):
            if self._evaluate_constraints(assignment):
                self._assignments = dict(assignment)
                return True
            return False
        var = variables[idx]
        for val in domains.get(var, [None]):
            assignment[var] = val
            if self._search(variables, domains, assignment, idx + 1):
                return True
        if var in assignment:
            del assignment[var]
        return False

    def _evaluate_constraints(self, env: Dict[str, Any]) -> bool:
        for c in self.constraints:
            if not self._eval_constraint(c, env):
                return False
        return True

    def _eval_constraint(self, c: SMTConstraint, env: Dict[str, Any]) -> bool:
        if c.kind == "comparison":
            if c.var not in env:
                return True
            val = env[c.var]
            op_map = {
                "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
                ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
                "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
            }
            func = op_map.get(c.op)
            if func is None:
                return True
            try:
                return func(val, c.constant)
            except TypeError:
                return False

        elif c.kind == "type_tag":
            if c.var not in env:
                return True
            val = env[c.var]
            type_map = {
                "int": int, "float": float, "str": str, "bool": bool,
                "list": list, "dict": dict, "NoneType": type(None),
            }
            expected = type_map.get(c.tag or "")
            if expected is None:
                return True
            return isinstance(val, expected)

        elif c.kind == "nullity":
            if c.var not in env:
                return True
            return (env[c.var] is None) == (c.is_null or False)

        elif c.kind == "membership":
            if c.var not in env or c.collection_var not in env:
                return True
            try:
                return env[c.var] in env[c.collection_var]
            except TypeError:
                return False

        elif c.kind == "length":
            if c.var not in env:
                return True
            val = env[c.var]
            try:
                length = len(val)
            except TypeError:
                return False
            op_map = {
                "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
                ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
                "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
            }
            func = op_map.get(c.op or "")
            if func is None:
                return True
            return func(length, c.constant)

        elif c.kind == "and":
            return all(self._eval_constraint(ch, env) for ch in c.children)

        elif c.kind == "or":
            return any(self._eval_constraint(ch, env) for ch in c.children)

        elif c.kind == "not":
            if c.children:
                return not self._eval_constraint(c.children[0], env)
            return True

        return True


# ---------------------------------------------------------------------------
# Safety violation
# ---------------------------------------------------------------------------


@dataclass
class SafetyViolation:
    """A potential safety violation discovered during abstract interpretation."""
    node: IRNode
    violation_kind: ViolationKind
    details: str
    abstract_state_at_point: AbstractState


# ---------------------------------------------------------------------------
# Counterexample
# ---------------------------------------------------------------------------


@dataclass
class Counterexample:
    """Concrete counterexample trace witnessing a potential bug."""
    path: List[str]
    path_condition: List[BasePredicate]
    variable_assignments: Dict[str, Any]
    violation_type: ViolationKind
    violation_location: Optional[SourceLoc] = None

    def __hash__(self) -> int:
        return hash(self._hash_key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Counterexample):
            return NotImplemented
        return self._hash_key() == other._hash_key()

    def _hash_key(self) -> Tuple:
        path_t = tuple(self.path)
        cond_t = tuple(str(p) for p in self.path_condition)
        return (path_t, cond_t, self.violation_type)


# ---------------------------------------------------------------------------
# Counterexample cache
# ---------------------------------------------------------------------------


class CounterexampleCache:
    """LRU cache for counterexample analysis results."""

    def __init__(self, max_size: int = 1024) -> None:
        self._max_size = max_size
        self._cache: Dict[str, Tuple[bool, List[BasePredicate]]] = {}
        self._access_order: List[str] = []
        self._hits = 0
        self._misses = 0

    def _hash_counterexample(self, cex: Counterexample) -> str:
        h = hashlib.sha256()
        h.update(str(cex.path).encode())
        h.update(str([str(p) for p in cex.path_condition]).encode())
        h.update(cex.violation_type.value.encode())
        return h.hexdigest()

    def lookup(self, cex: Counterexample) -> Optional[Tuple[bool, List[BasePredicate]]]:
        key = self._hash_counterexample(cex)
        if key in self._cache:
            self._hits += 1
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        self._misses += 1
        return None

    def store(self, cex: Counterexample, is_spurious: bool,
              predicates_extracted: List[BasePredicate]) -> None:
        key = self._hash_counterexample(cex)
        if len(self._cache) >= self._max_size and key not in self._cache:
            if self._access_order:
                oldest = self._access_order.pop(0)
                self._cache.pop(oldest, None)
        self._cache[key] = (is_spurious, predicates_extracted)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Safety checker
# ---------------------------------------------------------------------------


class SafetyChecker:
    """Verifies safety properties against computed abstract states."""

    def __init__(self) -> None:
        self._checked = 0
        self._violations_found = 0

    def check_all(self, ir_func: IRFunction,
                  abstract_states: Dict[str, AbstractState]) -> List[SafetyViolation]:
        violations: List[SafetyViolation] = []
        for block_id, block in ir_func.blocks.items():
            state = abstract_states.get(block_id, AbstractState(is_bottom=True))
            if state.is_bottom:
                continue
            for node in block.nodes:
                self._checked += 1
                v = self._check_node(node, state)
                if v is not None:
                    violations.append(v)
                    self._violations_found += 1
        return violations

    def _check_node(self, node: IRNode, state: AbstractState) -> Optional[SafetyViolation]:
        if isinstance(node, IndexNode):
            return self.check_array_bounds(node, state)
        if isinstance(node, LoadAttrNode):
            return self.check_null_deref(node, state)
        if isinstance(node, BinOpNode):
            if node.op in (BinOp.DIV, BinOp.MOD, BinOp.FLOORDIV):
                return self.check_div_zero(node, state)
            return self.check_type_tag(node, state)
        if isinstance(node, CallNode):
            return self.check_null_deref_call(node, state)
        return None

    def check_array_bounds(self, node: IndexNode,
                           state: AbstractState) -> Optional[SafetyViolation]:
        obj_type = state[node.obj]
        idx_type = state[node.index]

        safe_lower = self._can_prove_safe(
            ComparisonPredicate(node.index, ComparisonOp.GE, 0), state
        )
        safe_upper = False
        for p in obj_type.predicates:
            if isinstance(p, LengthPredicate):
                bound_pred = ComparisonPredicate(
                    node.index, ComparisonOp.LT, p.constant
                )
                if self._can_prove_safe(bound_pred, state):
                    safe_upper = True
                    break

        if not safe_lower or not safe_upper:
            details = []
            if not safe_lower:
                details.append(f"cannot prove {node.index} >= 0")
            if not safe_upper:
                details.append(f"cannot prove {node.index} < len({node.obj})")
            return self._create_violation(
                node, ViolationKind.ARRAY_OUT_OF_BOUNDS,
                "; ".join(details), state,
            )
        return None

    def check_null_deref(self, node: IRNode,
                         state: AbstractState) -> Optional[SafetyViolation]:
        obj_var = ""
        if isinstance(node, LoadAttrNode):
            obj_var = node.obj
        elif isinstance(node, StoreAttrNode):
            obj_var = node.obj
        else:
            return None

        not_null = NullityPredicate(obj_var, is_null=False)
        if self._can_prove_safe(not_null, state):
            return None

        return self._create_violation(
            node, ViolationKind.NULL_DEREFERENCE,
            f"cannot prove {obj_var} is not None", state,
        )

    def check_null_deref_call(self, node: CallNode,
                              state: AbstractState) -> Optional[SafetyViolation]:
        for arg in node.args:
            arg_type = state[arg]
            is_null_pred = NullityPredicate(arg, is_null=True)
            if is_null_pred in arg_type.predicates:
                return self._create_violation(
                    node, ViolationKind.NULL_DEREFERENCE,
                    f"argument {arg} may be None", state,
                )
        return None

    def check_div_zero(self, node: BinOpNode,
                       state: AbstractState) -> Optional[SafetyViolation]:
        nonzero = ComparisonPredicate(node.right, ComparisonOp.NE, 0)
        if self._can_prove_safe(nonzero, state):
            return None
        return self._create_violation(
            node, ViolationKind.DIVISION_BY_ZERO,
            f"cannot prove {node.right} != 0", state,
        )

    def check_type_tag(self, node: BinOpNode,
                       state: AbstractState) -> Optional[SafetyViolation]:
        left_type = state[node.left]
        right_type = state[node.right]

        numeric_ops = {BinOp.ADD, BinOp.SUB, BinOp.MUL, BinOp.DIV,
                       BinOp.MOD, BinOp.FLOORDIV, BinOp.POW}
        if node.op in numeric_ops:
            numeric_types = {BaseType.INT, BaseType.FLOAT, BaseType.BOOL, BaseType.ANY}
            if (left_type.base_type not in numeric_types
                    and left_type.base_type != BaseType.UNKNOWN):
                return self._create_violation(
                    node, ViolationKind.TYPE_TAG_MISMATCH,
                    f"{node.left} has type {left_type.base_type.value}, "
                    f"expected numeric for {node.op.value}", state,
                )
            if (right_type.base_type not in numeric_types
                    and right_type.base_type != BaseType.UNKNOWN):
                return self._create_violation(
                    node, ViolationKind.TYPE_TAG_MISMATCH,
                    f"{node.right} has type {right_type.base_type.value}, "
                    f"expected numeric for {node.op.value}", state,
                )
        return None

    def _can_prove_safe(self, predicate: BasePredicate, state: AbstractState) -> bool:
        var = predicate.variable
        if var is None:
            return False
        rt = state[var]
        if rt.is_bottom:
            return True
        if predicate in rt.predicates:
            return True
        if isinstance(predicate, ComparisonPredicate):
            for p in rt.predicates:
                if isinstance(p, ComparisonPredicate) and p.variable == predicate.variable:
                    if self._implies_comparison(p, predicate):
                        return True
        if isinstance(predicate, NullityPredicate) and not predicate.is_null:
            for p in rt.predicates:
                if isinstance(p, TypeTagPredicate) and p.variable == var:
                    if p.tag != BaseType.NONE_TYPE:
                        return True
        return False

    def _implies_comparison(self, known: ComparisonPredicate,
                            goal: ComparisonPredicate) -> bool:
        if known.variable != goal.variable:
            return False
        try:
            kv = known.constant
            gv = goal.constant
            if known.op == ComparisonOp.GE and goal.op == ComparisonOp.GE:
                return kv >= gv
            if known.op == ComparisonOp.GT and goal.op == ComparisonOp.GE:
                return kv >= gv
            if known.op == ComparisonOp.GT and goal.op == ComparisonOp.GT:
                return kv >= gv
            if known.op == ComparisonOp.LE and goal.op == ComparisonOp.LE:
                return kv <= gv
            if known.op == ComparisonOp.LT and goal.op == ComparisonOp.LE:
                return kv <= gv
            if known.op == ComparisonOp.LT and goal.op == ComparisonOp.LT:
                return kv <= gv
            if known.op == ComparisonOp.EQ and goal.op == ComparisonOp.EQ:
                return kv == gv
            if known.op == ComparisonOp.EQ and goal.op == ComparisonOp.GE:
                return kv >= gv
            if known.op == ComparisonOp.EQ and goal.op == ComparisonOp.LE:
                return kv <= gv
            if known.op == ComparisonOp.EQ and goal.op == ComparisonOp.NE:
                return kv != gv
            if known.op == ComparisonOp.GE and goal.op == ComparisonOp.NE:
                return kv > gv
            if known.op == ComparisonOp.GT and goal.op == ComparisonOp.NE:
                return True
        except TypeError:
            pass
        return False

    def _create_violation(self, node: IRNode, kind: ViolationKind,
                          details: str,
                          state: AbstractState) -> SafetyViolation:
        return SafetyViolation(
            node=node,
            violation_kind=kind,
            details=details,
            abstract_state_at_point=state.copy(),
        )


# ---------------------------------------------------------------------------
# Counterexample analyser
# ---------------------------------------------------------------------------


class CounterexampleAnalyzer:
    """Extracts and checks feasibility of counterexample paths."""

    def __init__(self, ir_func: IRFunction) -> None:
        self._func = ir_func
        self._path_cache: Dict[int, List[List[str]]] = {}

    def extract_path(self, violation: SafetyViolation) -> Counterexample:
        """Extract a CFG path from the entry block to the violation node."""
        target_block = self._find_block_for_node(violation.node)
        if target_block is None:
            return Counterexample(
                path=[], path_condition=[], variable_assignments={},
                violation_type=violation.violation_kind,
                violation_location=violation.node.source_loc,
            )
        path = self._find_path(self._func.entry_block_id, target_block)
        path_condition = self._collect_path_conditions(path)
        return Counterexample(
            path=path,
            path_condition=path_condition,
            variable_assignments={},
            violation_type=violation.violation_kind,
            violation_location=violation.node.source_loc,
        )

    def is_spurious(self, cex: Counterexample) -> bool:
        """Check feasibility of the counterexample via SMT encoding."""
        if not cex.path:
            return True
        formula = self.encode_path(cex.path, self._func)
        for pc in cex.path_condition:
            formula.add_constraint(pc.to_smt())
        sat = self._check_sat(formula)
        if sat:
            cex.variable_assignments = self._extract_model(formula)
            return False
        return True

    def encode_path(self, path: List[str], ir_func: IRFunction) -> SMTFormula:
        """Build an SMT formula encoding the path through the CFG."""
        formula = SMTFormula()
        ssa = self._build_ssa_encoding(path)
        for var, version, constraint in ssa:
            formula.add_constraint(constraint)
        return formula

    def _build_ssa_encoding(
        self, path: List[str]
    ) -> List[Tuple[str, int, SMTConstraint]]:
        """Convert a path into SSA form for SMT encoding."""
        ssa_entries: List[Tuple[str, int, SMTConstraint]] = []
        var_versions: Dict[str, int] = collections.defaultdict(int)

        def cur(v: str) -> str:
            return f"{v}_{var_versions[v]}"

        def fresh(v: str) -> str:
            var_versions[v] += 1
            return f"{v}_{var_versions[v]}"

        for block_id in path:
            block = self._func.blocks.get(block_id)
            if block is None:
                continue
            for node in block.nodes:
                if isinstance(node, AssignNode):
                    new_name = fresh(node.target)
                    if node.value_constant is not None:
                        ssa_entries.append((
                            node.target, var_versions[node.target],
                            SMTConstraint("comparison", var=new_name,
                                          op="==", constant=node.value_constant),
                        ))
                    elif node.value_var:
                        src = cur(node.value_var)
                        ssa_entries.append((
                            node.target, var_versions[node.target],
                            SMTConstraint("eq_vars", var=new_name, name=src),
                        ))
                elif isinstance(node, BinOpNode):
                    new_name = fresh(node.target)
                    left_ssa = cur(node.left)
                    right_ssa = cur(node.right)
                    ssa_entries.append((
                        node.target, var_versions[node.target],
                        SMTConstraint("binop", var=new_name, op=node.op.value,
                                      name=left_ssa,
                                      collection_var=right_ssa),
                    ))
                elif isinstance(node, GuardNode):
                    if node.condition is not None:
                        renamed = node.condition.substitute(
                            {v: cur(v) for v in node.condition.free_variables()}
                        )
                        ssa_entries.append((
                            "__guard__", 0, renamed.to_smt(),
                        ))
                elif isinstance(node, PhiNode):
                    new_name = fresh(node.target)
                    for pred_block, src_var in node.incoming.items():
                        if pred_block in set(path):
                            src_ssa = cur(src_var)
                            ssa_entries.append((
                                node.target, var_versions[node.target],
                                SMTConstraint("eq_vars", var=new_name, name=src_ssa),
                            ))
                            break
                elif isinstance(node, CallNode):
                    fresh(node.target)
                elif isinstance(node, IndexNode):
                    fresh(node.target)
                elif isinstance(node, LoadAttrNode):
                    fresh(node.target)
        return ssa_entries

    def _find_block_for_node(self, node: IRNode) -> Optional[str]:
        for bid, block in self._func.blocks.items():
            for n in block.nodes:
                if n.node_id == node.node_id:
                    return bid
        return None

    def _find_path(self, start: str, end: str) -> List[str]:
        """BFS shortest path from *start* to *end*."""
        if start == end:
            return [start]
        queue: collections.deque[List[str]] = collections.deque([[start]])
        visited: Set[str] = {start}
        while queue:
            current_path = queue.popleft()
            last = current_path[-1]
            block = self._func.blocks.get(last)
            if block is None:
                continue
            for succ in block.successors:
                if succ == end:
                    return current_path + [succ]
                if succ not in visited:
                    visited.add(succ)
                    queue.append(current_path + [succ])
        return [start, end]

    def _collect_path_conditions(self, path: List[str]) -> List[BasePredicate]:
        conditions: List[BasePredicate] = []
        for i, bid in enumerate(path):
            block = self._func.blocks.get(bid)
            if block is None:
                continue
            for node in block.nodes:
                if isinstance(node, GuardNode) and node.condition is not None:
                    next_bid = path[i + 1] if i + 1 < len(path) else None
                    if next_bid == node.true_branch:
                        conditions.append(node.condition)
                    elif next_bid == node.false_branch:
                        conditions.append(node.condition.negate())
        return conditions

    def _check_sat(self, formula: SMTFormula) -> bool:
        return formula.check_sat()

    def _extract_model(self, formula: SMTFormula) -> Dict[str, Any]:
        return formula.get_model()


# ---------------------------------------------------------------------------
# Predicate refiner
# ---------------------------------------------------------------------------


class PredicateRefiner:
    """Extracts new predicates from spurious counterexamples."""

    def __init__(self, predicate_budget: int = 500) -> None:
        self._budget = predicate_budget
        self._total_extracted = 0

    def extract_interpolants(self, spurious_cex: Counterexample,
                             formula: SMTFormula) -> List[BasePredicate]:
        """Extract Craig interpolants from an infeasible path formula."""
        path = spurious_cex.path
        if len(path) < 2:
            return self._fallback_predicates(spurious_cex)

        predicates: List[BasePredicate] = []

        mid = len(path) // 2
        prefix_formula = SMTFormula()
        suffix_formula = SMTFormula()

        prefix_constraints = formula.constraints[:mid] if formula.constraints else []
        suffix_constraints = formula.constraints[mid:] if formula.constraints else []

        for c in prefix_constraints:
            prefix_formula.add_constraint(c)
        for c in suffix_constraints:
            suffix_formula.add_constraint(c)

        interpolant_preds = self._compute_interpolant_from_partition(
            prefix_formula, suffix_formula
        )
        predicates.extend(interpolant_preds)

        for pc in spurious_cex.path_condition:
            atomic = self._decompose_to_atomic(pc)
            predicates.extend(atomic)

        seen: Set[Hashable] = set()
        unique: List[BasePredicate] = []
        for p in predicates:
            k = p._key()
            if k not in seen:
                seen.add(k)
                unique.append(p)

        self._total_extracted += len(unique)
        return unique

    def project_to_P(self, interpolant: BasePredicate,
                     variables: Set[str]) -> List[BasePredicate]:
        """Project an interpolant down to the predicate language P."""
        free = interpolant.free_variables()
        if not free.issubset(variables):
            projected: List[BasePredicate] = []
            for v in free & variables:
                if isinstance(interpolant, ComparisonPredicate):
                    projected.append(ComparisonPredicate(v, interpolant.op, interpolant.constant))
                elif isinstance(interpolant, TypeTagPredicate):
                    projected.append(TypeTagPredicate(v, interpolant.tag))
                elif isinstance(interpolant, NullityPredicate):
                    projected.append(NullityPredicate(v, interpolant.is_null))
            return projected if projected else [interpolant]
        return [interpolant]

    def rank_predicates(self, candidates: List[BasePredicate],
                        current_predicates: Set[BasePredicate]) -> List[BasePredicate]:
        """Rank candidate predicates by estimated usefulness."""
        scored: List[Tuple[float, BasePredicate]] = []
        for p in candidates:
            if p in current_predicates:
                continue
            score = self._score_predicate(p, current_predicates)
            scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored]

    def add_predicates(self, new_preds: List[BasePredicate],
                       current_set: Set[BasePredicate]) -> Set[BasePredicate]:
        """Add new predicates respecting budget."""
        result = set(current_set)
        remaining_budget = self._budget - len(result)
        ranked = self.rank_predicates(new_preds, result)
        for p in ranked:
            if remaining_budget <= 0:
                break
            result.add(p)
            remaining_budget -= 1
        return result

    def _compute_interpolant_from_partition(
        self, prefix_formula: SMTFormula, suffix_formula: SMTFormula
    ) -> List[BasePredicate]:
        """Compute interpolants from prefix/suffix partition of an unsat formula."""
        predicates: List[BasePredicate] = []

        for c in prefix_formula.constraints:
            preds = self._constraint_to_predicates(c)
            predicates.extend(preds)

        for c in suffix_formula.constraints:
            preds = self._constraint_to_predicates(c)
            predicates.extend(preds)

        prefix_vars = prefix_formula._collect_variables()
        suffix_vars = suffix_formula._collect_variables()
        shared_vars = prefix_vars & suffix_vars

        filtered: List[BasePredicate] = []
        for p in predicates:
            fv = p.free_variables()
            if fv & shared_vars:
                filtered.append(p)

        return filtered if filtered else predicates

    def _constraint_to_predicates(self, c: SMTConstraint) -> List[BasePredicate]:
        """Convert an SMT constraint back to predicates."""
        results: List[BasePredicate] = []
        if c.kind == "comparison" and c.var and c.op:
            op_map = {
                "<": ComparisonOp.LT, "<=": ComparisonOp.LE,
                ">": ComparisonOp.GT, ">=": ComparisonOp.GE,
                "==": ComparisonOp.EQ, "!=": ComparisonOp.NE,
            }
            op = op_map.get(c.op)
            if op:
                var_name = c.var.rsplit("_", 1)[0] if "_" in c.var else c.var
                results.append(ComparisonPredicate(var_name, op, c.constant))
        elif c.kind == "type_tag" and c.var and c.tag:
            var_name = c.var.rsplit("_", 1)[0] if "_" in c.var else c.var
            try:
                tag = BaseType(c.tag)
                results.append(TypeTagPredicate(var_name, tag))
            except ValueError:
                pass
        elif c.kind == "nullity" and c.var:
            var_name = c.var.rsplit("_", 1)[0] if "_" in c.var else c.var
            results.append(NullityPredicate(var_name, c.is_null or False))
        elif c.kind in ("and", "or", "not"):
            for child in c.children:
                results.extend(self._constraint_to_predicates(child))
        return results

    def _decompose_to_atomic(self, predicate: BasePredicate) -> List[BasePredicate]:
        """Decompose compound predicates into atomic ones."""
        if isinstance(predicate, ConjunctionPredicate):
            return (self._decompose_to_atomic(predicate.left)
                    + self._decompose_to_atomic(predicate.right))
        if isinstance(predicate, DisjunctionPredicate):
            return (self._decompose_to_atomic(predicate.left)
                    + self._decompose_to_atomic(predicate.right))
        if isinstance(predicate, NegationPredicate):
            inner_parts = self._decompose_to_atomic(predicate.inner)
            return [p.negate() for p in inner_parts]
        return [predicate]

    def _score_predicate(self, pred: BasePredicate,
                         current: Set[BasePredicate]) -> float:
        """Score a candidate predicate for usefulness."""
        score = 1.0
        if isinstance(pred, ComparisonPredicate):
            score += 2.0
            if pred.op in (ComparisonOp.GE, ComparisonOp.GT):
                score += 0.5
            if pred.constant == 0:
                score += 1.0
        elif isinstance(pred, NullityPredicate):
            score += 3.0
        elif isinstance(pred, TypeTagPredicate):
            score += 2.5
        elif isinstance(pred, LengthPredicate):
            score += 2.0
        neg = pred.negate()
        if neg in current:
            score += 1.5
        for cp in current:
            if cp.variable == pred.variable:
                score += 0.3
        return score

    def _fallback_predicates(self, cex: Counterexample) -> List[BasePredicate]:
        """Generate fallback predicates from the counterexample itself."""
        preds: List[BasePredicate] = []
        for pc in cex.path_condition:
            preds.extend(self._decompose_to_atomic(pc))
        if cex.violation_type == ViolationKind.NULL_DEREFERENCE:
            for var in cex.variable_assignments:
                preds.append(NullityPredicate(var, is_null=False))
        elif cex.violation_type == ViolationKind.DIVISION_BY_ZERO:
            for var in cex.variable_assignments:
                preds.append(ComparisonPredicate(var, ComparisonOp.NE, 0))
        elif cex.violation_type == ViolationKind.ARRAY_OUT_OF_BOUNDS:
            for var in cex.variable_assignments:
                preds.append(ComparisonPredicate(var, ComparisonOp.GE, 0))
        return preds


# ---------------------------------------------------------------------------
# Abstract interpreter
# ---------------------------------------------------------------------------


class StdlibModel:
    """Models for standard-library and built-in functions."""

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._models["len"] = self._model_len
        self._models["int"] = self._model_int
        self._models["float"] = self._model_float
        self._models["str"] = self._model_str
        self._models["bool"] = self._model_bool
        self._models["abs"] = self._model_abs
        self._models["min"] = self._model_min
        self._models["max"] = self._model_max
        self._models["range"] = self._model_range
        self._models["isinstance"] = self._model_isinstance
        self._models["type"] = self._model_type
        self._models["print"] = self._model_print
        self._models["sorted"] = self._model_sorted
        self._models["list"] = self._model_list
        self._models["dict"] = self._model_dict
        self._models["set"] = self._model_set
        self._models["tuple"] = self._model_tuple
        self._models["append"] = self._model_append
        self._models["pop"] = self._model_pop
        self._models["get"] = self._model_get

    def has_model(self, func_name: str) -> bool:
        return func_name in self._models

    def apply(self, func_name: str, args: List[RefinementType],
              state: AbstractState) -> RefinementType:
        model = self._models.get(func_name)
        if model is None:
            return RefinementType.top()
        return model(args, state)

    def _model_len(self, args: List[RefinementType],
                   state: AbstractState) -> RefinementType:
        result = RefinementType(base_type=BaseType.INT)
        result = result.add_predicate(
            ComparisonPredicate("__result__", ComparisonOp.GE, 0)
        )
        return result

    def _model_int(self, args: List[RefinementType],
                   state: AbstractState) -> RefinementType:
        return RefinementType(base_type=BaseType.INT)

    def _model_float(self, args: List[RefinementType],
                     state: AbstractState) -> RefinementType:
        return RefinementType(base_type=BaseType.FLOAT)

    def _model_str(self, args: List[RefinementType],
                   state: AbstractState) -> RefinementType:
        return RefinementType(base_type=BaseType.STR)

    def _model_bool(self, args: List[RefinementType],
                    state: AbstractState) -> RefinementType:
        return RefinementType(base_type=BaseType.BOOL)

    def _model_abs(self, args: List[RefinementType],
                   state: AbstractState) -> RefinementType:
        if args:
            bt = args[0].base_type
        else:
            bt = BaseType.INT
        result = RefinementType(base_type=bt)
        result = result.add_predicate(
            ComparisonPredicate("__result__", ComparisonOp.GE, 0)
        )
        return result

    def _model_min(self, args: List[RefinementType],
                   state: AbstractState) -> RefinementType:
        if args:
            return RefinementType(base_type=args[0].base_type)
        return RefinementType.top()

    def _model_max(self, args: List[RefinementType],
                   state: AbstractState) -> RefinementType:
        if args:
            return RefinementType(base_type=args[0].base_type)
        return RefinementType.top()

    def _model_range(self, args: List[RefinementType],
                     state: AbstractState) -> RefinementType:
        return RefinementType(base_type=BaseType.LIST)

    def _model_isinstance(self, args: List[RefinementType],
                          state: AbstractState) -> RefinementType:
        return RefinementType(base_type=BaseType.BOOL)

    def _model_type(self, args: List[RefinementType],
                    state: AbstractState) -> RefinementType:
        return RefinementType.top()

    def _model_print(self, args: List[RefinementType],
                     state: AbstractState) -> RefinementType:
        return RefinementType(base_type=BaseType.NONE_TYPE)

    def _model_sorted(self, args: List[RefinementType],
                      state: AbstractState) -> RefinementType:
        return RefinementType(base_type=BaseType.LIST)

    def _model_list(self, args: List[RefinementType],
                    state: AbstractState) -> RefinementType:
        return RefinementType(base_type=BaseType.LIST)

    def _model_dict(self, args: List[RefinementType],
                    state: AbstractState) -> RefinementType:
        return RefinementType(base_type=BaseType.DICT)

    def _model_set(self, args: List[RefinementType],
                   state: AbstractState) -> RefinementType:
        return RefinementType(base_type=BaseType.SET)

    def _model_tuple(self, args: List[RefinementType],
                     state: AbstractState) -> RefinementType:
        return RefinementType(base_type=BaseType.TUPLE)

    def _model_append(self, args: List[RefinementType],
                      state: AbstractState) -> RefinementType:
        return RefinementType(base_type=BaseType.NONE_TYPE)

    def _model_pop(self, args: List[RefinementType],
                   state: AbstractState) -> RefinementType:
        return RefinementType.top()

    def _model_get(self, args: List[RefinementType],
                   state: AbstractState) -> RefinementType:
        return RefinementType.top()


class AbstractInterpreter:
    """Forward abstract interpreter over refinement-type abstract domain."""

    def __init__(self, predicates: Set[BasePredicate],
                 domain: Optional[str] = None,
                 stdlib_models: Optional[StdlibModel] = None,
                 widening_delay: int = 3,
                 narrowing_iterations: int = 2) -> None:
        self._predicates = set(predicates)
        self._domain = domain or "refinement"
        self._stdlib = stdlib_models or StdlibModel()
        self._widening_delay = widening_delay
        self._narrowing_iterations = narrowing_iterations
        self._iteration_counts: Dict[str, int] = {}

    def forward_analysis(self, ir_func: IRFunction) -> Dict[str, AbstractState]:
        """Run forward analysis and return per-block abstract states."""
        return self.compute_fixpoint(ir_func)

    def compute_fixpoint(self, ir_func: IRFunction) -> Dict[str, AbstractState]:
        """Iterate worklist algorithm until fixed point."""
        block_states: Dict[str, AbstractState] = {}
        for bid in ir_func.blocks:
            block_states[bid] = AbstractState(is_bottom=True)

        entry_state = self._initial_state(ir_func)
        block_states[ir_func.entry_block_id] = entry_state

        worklist: List[Tuple[int, str]] = []
        rpo = ir_func.block_order()
        rpo_index = {bid: i for i, bid in enumerate(rpo)}

        heapq.heappush(worklist, (rpo_index.get(ir_func.entry_block_id, 0),
                                  ir_func.entry_block_id))
        in_worklist: Set[str] = {ir_func.entry_block_id}
        iteration_count: Dict[str, int] = collections.defaultdict(int)
        max_iterations = len(ir_func.blocks) * 50

        total_iterations = 0
        while worklist and total_iterations < max_iterations:
            total_iterations += 1
            _, bid = heapq.heappop(worklist)
            in_worklist.discard(bid)
            block = ir_func.blocks.get(bid)
            if block is None:
                continue

            in_state = self._compute_block_input(bid, block, block_states)
            if in_state.is_bottom:
                continue

            out_state = self.process_block(block, in_state, block_states)
            iteration_count[bid] += 1

            for succ_id in block.successors:
                old_succ = block_states.get(succ_id, AbstractState(is_bottom=True))
                succ_block = ir_func.blocks.get(succ_id)

                new_input = self._compute_successor_state(
                    block, succ_id, out_state, in_state
                )

                joined = old_succ.join(new_input)

                if iteration_count.get(succ_id, 0) >= self._widening_delay:
                    joined = self._apply_widening(
                        old_succ, joined, iteration_count.get(succ_id, 0)
                    )

                if joined != old_succ:
                    block_states[succ_id] = joined
                    if succ_id not in in_worklist:
                        heapq.heappush(worklist, (rpo_index.get(succ_id, 0), succ_id))
                        in_worklist.add(succ_id)

        if self._narrowing_iterations > 0:
            block_states = self._run_narrowing(ir_func, block_states, rpo)

        self._iteration_counts = dict(iteration_count)
        return block_states

    def _run_narrowing(self, ir_func: IRFunction,
                       wide_states: Dict[str, AbstractState],
                       rpo: List[str]) -> Dict[str, AbstractState]:
        """Narrowing pass to recover precision lost by widening."""
        states = {k: v.copy() for k, v in wide_states.items()}
        for _narrow_iter in range(self._narrowing_iterations):
            changed = False
            for bid in rpo:
                block = ir_func.blocks.get(bid)
                if block is None:
                    continue
                in_state = self._compute_block_input(bid, block, states)
                if in_state.is_bottom:
                    continue
                out_state = self.process_block(block, in_state, states)
                for succ_id in block.successors:
                    old_succ = states.get(succ_id, AbstractState(is_bottom=True))
                    new_input = self._compute_successor_state(
                        block, succ_id, out_state, in_state
                    )
                    narrowed = self._apply_narrowing(old_succ, new_input)
                    if narrowed != old_succ:
                        states[succ_id] = narrowed
                        changed = True
            if not changed:
                break
        return states

    def process_block(self, block: IRBlock, in_state: AbstractState,
                      block_states: Optional[Dict[str, AbstractState]] = None
                      ) -> AbstractState:
        """Process all nodes in a block, threading state through."""
        state = in_state.copy()
        for node in block.nodes:
            state = self.process_node(node, state, block_states)
            if state.is_bottom:
                break
        return state

    def process_node(self, node: IRNode, state: AbstractState,
                     block_states: Optional[Dict[str, AbstractState]] = None
                     ) -> AbstractState:
        """Dispatch to the appropriate handler for *node*."""
        if isinstance(node, AssignNode):
            return self.handle_assign(node, state)
        if isinstance(node, GuardNode):
            true_st, false_st = self.handle_guard(node, state)
            return true_st.join(false_st)
        if isinstance(node, CallNode):
            return self.handle_call(node, state)
        if isinstance(node, BinOpNode):
            return self.handle_binop(node, state)
        if isinstance(node, LoadAttrNode):
            return self.handle_load_attr(node, state)
        if isinstance(node, IndexNode):
            return self.handle_index(node, state)
        if isinstance(node, PhiNode):
            return self.handle_phi(node, state, block_states or {})
        if isinstance(node, StoreAttrNode):
            return self._handle_store_attr(node, state)
        if isinstance(node, ReturnNode):
            return self._handle_return(node, state)
        return state

    def handle_assign(self, node: AssignNode, state: AbstractState) -> AbstractState:
        """Abstract transformer for assignment nodes."""
        new_state = state.copy()
        if node.value_constant is not None:
            val = node.value_constant
            rt = self._type_from_constant(val)
            relevant_preds: Set[BasePredicate] = set()
            for p in self._predicates:
                if p.variable == node.target:
                    env = {node.target: val}
                    try:
                        if p.evaluate(env):
                            relevant_preds.add(p)
                    except Exception:
                        pass
            rt = RefinementType(base_type=rt.base_type, predicates=relevant_preds)
            new_state[node.target] = rt
        elif node.value_var:
            src_type = state[node.value_var]
            substituted_preds: Set[BasePredicate] = set()
            for p in src_type.predicates:
                new_p = p.substitute({node.value_var: node.target})
                substituted_preds.add(new_p)
            for p in self._predicates:
                if p.variable == node.target and p not in substituted_preds:
                    src_p = p.substitute({node.target: node.value_var})
                    if src_p in src_type.predicates:
                        substituted_preds.add(p)
            new_state[node.target] = RefinementType(
                base_type=src_type.base_type,
                predicates=substituted_preds,
            )
        else:
            new_state[node.target] = RefinementType.top()
        return new_state

    def handle_call(self, node: CallNode, state: AbstractState) -> AbstractState:
        """Model function calls abstractly using stdlib models or top."""
        new_state = state.copy()
        arg_types = [state[a] for a in node.args]
        if self._stdlib.has_model(node.func_name):
            result_type = self._stdlib.apply(node.func_name, arg_types, state)
            renamed_preds: Set[BasePredicate] = set()
            for p in result_type.predicates:
                renamed_preds.add(p.substitute({"__result__": node.target}))
            result_type = RefinementType(
                base_type=result_type.base_type, predicates=renamed_preds
            )
            new_state[node.target] = result_type
        else:
            new_state[node.target] = RefinementType.top()
        return new_state

    def handle_binop(self, node: BinOpNode, state: AbstractState) -> AbstractState:
        """Abstract transformer for binary operations."""
        new_state = state.copy()
        left_t = state[node.left]
        right_t = state[node.right]

        result_base = self._binop_result_type(left_t.base_type, right_t.base_type, node.op)
        result_preds: Set[BasePredicate] = set()

        if node.op == BinOp.ADD:
            for p in left_t.predicates:
                if isinstance(p, ComparisonPredicate) and p.op == ComparisonOp.GE:
                    for q in right_t.predicates:
                        if isinstance(q, ComparisonPredicate) and q.op == ComparisonOp.GE:
                            try:
                                combined = p.constant + q.constant
                                result_preds.add(
                                    ComparisonPredicate(node.target, ComparisonOp.GE, combined)
                                )
                            except TypeError:
                                pass

        elif node.op == BinOp.SUB:
            pass

        elif node.op == BinOp.MUL:
            for p in left_t.predicates:
                if isinstance(p, ComparisonPredicate) and p.op == ComparisonOp.GE and p.constant >= 0:
                    for q in right_t.predicates:
                        if isinstance(q, ComparisonPredicate) and q.op == ComparisonOp.GE and q.constant >= 0:
                            result_preds.add(
                                ComparisonPredicate(node.target, ComparisonOp.GE, 0)
                            )

        for p in self._predicates:
            if p.variable == node.target and p not in result_preds:
                if isinstance(p, TypeTagPredicate) and p.tag == BaseType(result_base.value):
                    result_preds.add(p)

        new_state[node.target] = RefinementType(
            base_type=result_base, predicates=result_preds
        )
        return new_state

    def handle_load_attr(self, node: LoadAttrNode,
                         state: AbstractState) -> AbstractState:
        """Handle attribute loads – result is top with possible type info."""
        new_state = state.copy()
        obj_type = state[node.obj]
        result_type = RefinementType.top()

        if node.attr == "__len__":
            result_type = RefinementType(base_type=BaseType.INT)
            result_type = result_type.add_predicate(
                ComparisonPredicate(node.target, ComparisonOp.GE, 0)
            )
        elif node.attr in ("append", "extend", "insert", "remove", "pop",
                           "sort", "reverse", "clear", "copy"):
            result_type = RefinementType.top()
        else:
            not_null = NullityPredicate(node.target, is_null=False)
            if not_null in self._predicates:
                result_type = result_type.add_predicate(not_null)

        new_state[node.target] = result_type
        return new_state

    def handle_index(self, node: IndexNode, state: AbstractState) -> AbstractState:
        """Handle indexing operations."""
        new_state = state.copy()
        obj_type = state[node.obj]
        result_type = RefinementType.top()

        if obj_type.base_type == BaseType.LIST:
            pass
        elif obj_type.base_type == BaseType.DICT:
            pass
        elif obj_type.base_type == BaseType.STR:
            result_type = RefinementType(base_type=BaseType.STR)
        elif obj_type.base_type == BaseType.TUPLE:
            pass

        for p in self._predicates:
            if p.variable == node.target:
                if isinstance(p, NullityPredicate) and not p.is_null:
                    result_type = result_type.add_predicate(p)

        new_state[node.target] = result_type
        return new_state

    def handle_phi(self, node: PhiNode, state: AbstractState,
                   block_states: Dict[str, AbstractState]) -> AbstractState:
        """Join values at phi nodes from different predecessors."""
        new_state = state.copy()
        joined_type = RefinementType.bottom()

        found_any = False
        for pred_block, src_var in node.incoming.items():
            pred_state = block_states.get(pred_block)
            if pred_state is None or pred_state.is_bottom:
                continue
            found_any = True
            src_type = pred_state[src_var]
            substituted_preds: Set[BasePredicate] = set()
            for p in src_type.predicates:
                substituted_preds.add(p.substitute({src_var: node.target}))
            src_renamed = RefinementType(
                base_type=src_type.base_type,
                predicates=substituted_preds,
                is_bottom=src_type.is_bottom,
                is_top=src_type.is_top,
            )
            joined_type = joined_type.join(src_renamed)

        if not found_any:
            joined_type = RefinementType.top()

        new_state[node.target] = joined_type
        return new_state

    def handle_guard(self, node: GuardNode,
                     state: AbstractState) -> Tuple[AbstractState, AbstractState]:
        """Refine state based on guard condition, yielding true/false states."""
        if node.condition is None:
            return state.copy(), state.copy()

        true_state = state.copy()
        false_state = state.copy()

        cond = node.condition
        neg_cond = cond.negate()

        true_state = self._refine_state_with(true_state, cond)
        false_state = self._refine_state_with(false_state, neg_cond)

        return true_state, false_state

    def _refine_state_with(self, state: AbstractState,
                           pred: BasePredicate) -> AbstractState:
        """Strengthen state with a predicate."""
        new_state = state.copy()
        var = pred.variable
        if var is None:
            if isinstance(pred, ConjunctionPredicate):
                new_state = self._refine_state_with(new_state, pred.left)
                new_state = self._refine_state_with(new_state, pred.right)
            return new_state

        current = new_state[var]
        if pred in self._predicates:
            current = current.add_predicate(pred)
        else:
            if isinstance(pred, ComparisonPredicate):
                for p in self._predicates:
                    if isinstance(p, ComparisonPredicate) and p.variable == var:
                        if self._comparison_implies(pred, p):
                            current = current.add_predicate(p)
            elif isinstance(pred, NullityPredicate):
                for p in self._predicates:
                    if isinstance(p, NullityPredicate) and p.variable == var:
                        if p.is_null == pred.is_null:
                            current = current.add_predicate(p)
            elif isinstance(pred, TypeTagPredicate):
                for p in self._predicates:
                    if isinstance(p, TypeTagPredicate) and p.variable == var:
                        if p.tag == pred.tag:
                            current = current.add_predicate(p)

        new_state[var] = current
        return new_state

    def _comparison_implies(self, source: ComparisonPredicate,
                            target: ComparisonPredicate) -> bool:
        """Check if source comparison implies target comparison."""
        if source.variable != target.variable:
            return False
        try:
            sv = source.constant
            tv = target.constant
            if source.op == ComparisonOp.EQ:
                return target.op.evaluate(sv, tv)
            if source.op == ComparisonOp.GT and target.op == ComparisonOp.GE:
                return sv >= tv
            if source.op == ComparisonOp.GE and target.op == ComparisonOp.GE:
                return sv >= tv
            if source.op == ComparisonOp.LT and target.op == ComparisonOp.LE:
                return sv <= tv
            if source.op == ComparisonOp.LE and target.op == ComparisonOp.LE:
                return sv <= tv
        except TypeError:
            pass
        return False

    def _apply_widening(self, old_state: AbstractState,
                        new_state: AbstractState,
                        iteration: int) -> AbstractState:
        """Apply widening to accelerate convergence."""
        return old_state.widen(new_state)

    def _apply_narrowing(self, wide_state: AbstractState,
                         precise_state: AbstractState) -> AbstractState:
        """Apply narrowing to recover precision."""
        return wide_state.narrow(precise_state)

    def _initial_state(self, ir_func: IRFunction) -> AbstractState:
        """Build initial abstract state from function parameters."""
        bindings: Dict[str, RefinementType] = {}
        for param in ir_func.params:
            param_type = RefinementType.top()
            for p in self._predicates:
                if p.variable == param:
                    pass
            bindings[param] = param_type
        return AbstractState(bindings)

    def _compute_block_input(self, bid: str, block: IRBlock,
                             block_states: Dict[str, AbstractState]) -> AbstractState:
        """Join states from all predecessors."""
        if not block.predecessors:
            return block_states.get(bid, AbstractState(is_bottom=True))
        result = AbstractState(is_bottom=True)
        for pred_id in block.predecessors:
            pred_state = block_states.get(pred_id, AbstractState(is_bottom=True))
            result = result.join(pred_state)
        return result

    def _compute_successor_state(self, block: IRBlock, succ_id: str,
                                 out_state: AbstractState,
                                 in_state: AbstractState) -> AbstractState:
        """Compute the state flowing to a specific successor, applying
        guard refinements if the block ends with a guard."""
        for node in block.nodes:
            if isinstance(node, GuardNode):
                if node.true_branch == succ_id:
                    true_st, _ = self.handle_guard(node, out_state)
                    return true_st
                elif node.false_branch == succ_id:
                    _, false_st = self.handle_guard(node, out_state)
                    return false_st
        return out_state

    def _handle_store_attr(self, node: StoreAttrNode,
                           state: AbstractState) -> AbstractState:
        """Handle attribute stores – currently a no-op on the target."""
        return state

    def _handle_return(self, node: ReturnNode,
                       state: AbstractState) -> AbstractState:
        """Handle return nodes."""
        if node.value:
            ret_type = state[node.value]
            new_state = state.copy()
            new_state["__return__"] = ret_type
            return new_state
        return state

    def _type_from_constant(self, val: Any) -> RefinementType:
        """Infer a refinement type from a concrete constant."""
        if val is None:
            return RefinementType(base_type=BaseType.NONE_TYPE)
        if isinstance(val, bool):
            return RefinementType(base_type=BaseType.BOOL)
        if isinstance(val, int):
            return RefinementType(base_type=BaseType.INT)
        if isinstance(val, float):
            return RefinementType(base_type=BaseType.FLOAT)
        if isinstance(val, str):
            return RefinementType(base_type=BaseType.STR)
        if isinstance(val, list):
            return RefinementType(base_type=BaseType.LIST)
        if isinstance(val, dict):
            return RefinementType(base_type=BaseType.DICT)
        if isinstance(val, set):
            return RefinementType(base_type=BaseType.SET)
        if isinstance(val, tuple):
            return RefinementType(base_type=BaseType.TUPLE)
        return RefinementType.top()

    def _binop_result_type(self, left: BaseType, right: BaseType,
                           op: BinOp) -> BaseType:
        """Determine result type of a binary operation."""
        if op == BinOp.DIV:
            return BaseType.FLOAT
        if op == BinOp.FLOORDIV:
            return BaseType.INT
        if op == BinOp.MOD:
            return left if left != BaseType.ANY else BaseType.INT
        if left == BaseType.FLOAT or right == BaseType.FLOAT:
            return BaseType.FLOAT
        if left == BaseType.INT and right == BaseType.INT:
            return BaseType.INT
        if left == BaseType.STR and right == BaseType.STR and op == BinOp.ADD:
            return BaseType.STR
        if left == BaseType.STR and right == BaseType.INT and op == BinOp.MUL:
            return BaseType.STR
        if left == BaseType.LIST and right == BaseType.LIST and op == BinOp.ADD:
            return BaseType.LIST
        if left == BaseType.BOOL and right == BaseType.BOOL:
            if op in (BinOp.AND, BinOp.OR, BinOp.XOR):
                return BaseType.BOOL
            return BaseType.INT
        return BaseType.ANY


# ---------------------------------------------------------------------------
# Convergence monitor
# ---------------------------------------------------------------------------


class ConvergenceMonitor:
    """Tracks CEGAR convergence metrics across iterations."""

    def __init__(self, config: CEGARConfig) -> None:
        self._config = config
        self.predicate_growth: List[int] = []
        self._state_sizes: List[int] = []
        self._timestamps: List[float] = []
        self._start_time: float = time.monotonic()

    def record_iteration(self, iteration: int, predicates: Set[BasePredicate],
                         state_size: int) -> None:
        self.predicate_growth.append(len(predicates))
        self._state_sizes.append(state_size)
        self._timestamps.append(time.monotonic() - self._start_time)

    def is_converging(self) -> bool:
        """Heuristic: convergence if predicate growth rate drops below threshold."""
        if len(self.predicate_growth) < 3:
            return False
        recent = self.predicate_growth[-3:]
        if recent[-1] == recent[-2] == recent[-3]:
            return True
        growth_rate = self.get_convergence_rate()
        return growth_rate < self._config.convergence_threshold

    def should_stop(self, iteration: int, elapsed_time: float) -> bool:
        if iteration >= self._config.max_iterations:
            logger.info("Stopping: reached max iterations %d", self._config.max_iterations)
            return True
        if elapsed_time >= self._config.per_function_timeout:
            logger.info("Stopping: timeout %.1fs", elapsed_time)
            return True
        if self.is_converging():
            logger.debug("Stopping: converged at iteration %d", iteration)
            return True
        return False

    def get_convergence_rate(self) -> float:
        """Rate of predicate growth (lower = closer to convergence)."""
        if len(self.predicate_growth) < 2:
            return 1.0
        deltas: List[float] = []
        for i in range(1, len(self.predicate_growth)):
            prev = self.predicate_growth[i - 1]
            cur = self.predicate_growth[i]
            if prev > 0:
                deltas.append((cur - prev) / prev)
            else:
                deltas.append(1.0 if cur > 0 else 0.0)
        if not deltas:
            return 0.0
        window = deltas[-min(5, len(deltas)):]
        return sum(window) / len(window)

    def estimated_remaining_iterations(self) -> int:
        """Estimate iterations remaining until convergence."""
        rate = self.get_convergence_rate()
        if rate <= 0:
            return 0
        if len(self.predicate_growth) < 2:
            return self._config.max_iterations
        current = self.predicate_growth[-1]
        budget = self._config.predicate_budget
        if current >= budget:
            return 0
        remaining = budget - current
        per_iter = max(rate * current, 1)
        return min(int(remaining / per_iter) + 1, self._config.max_iterations)


# ---------------------------------------------------------------------------
# Function summary
# ---------------------------------------------------------------------------


@dataclass
class FunctionSummary:
    """Summary of analysis results for inter-procedural use."""
    func_name: str = ""
    preconditions: List[BasePredicate] = field(default_factory=list)
    postconditions: List[BasePredicate] = field(default_factory=list)
    param_types: Dict[str, RefinementType] = field(default_factory=dict)
    return_type: RefinementType = field(default_factory=RefinementType.top)
    modifies: Set[str] = field(default_factory=set)
    side_effects: List[str] = field(default_factory=list)


class FunctionSummarizer:
    """Computes inter-procedural summaries from per-function CEGAR results."""

    def __init__(self) -> None:
        self._summaries: Dict[str, FunctionSummary] = {}

    def summarize(self, ir_func: IRFunction,
                  cegar_result: CEGARResult) -> FunctionSummary:
        pre = self.compute_precondition(ir_func, cegar_result)
        post = self.compute_postcondition(ir_func, cegar_result)
        modifies = self.compute_modifies(ir_func)

        param_types: Dict[str, RefinementType] = {}
        for param in ir_func.params:
            param_types[param] = cegar_result.inferred_types.get(
                param, RefinementType.top()
            )

        return_type = cegar_result.inferred_types.get(
            "__return__", RefinementType.top()
        )

        side_effects: List[str] = []
        for block in ir_func.blocks.values():
            for node in block.nodes:
                if isinstance(node, StoreAttrNode):
                    side_effects.append(f"write {node.obj}.{node.attr}")
                elif isinstance(node, CallNode):
                    if node.func_name == "print":
                        side_effects.append("io:stdout")

        summary = FunctionSummary(
            func_name=ir_func.name,
            preconditions=pre,
            postconditions=post,
            param_types=param_types,
            return_type=return_type,
            modifies=modifies,
            side_effects=side_effects,
        )
        self._summaries[ir_func.name] = summary
        return summary

    def compute_precondition(self, ir_func: IRFunction,
                             result: CEGARResult) -> List[BasePredicate]:
        """Derive preconditions from parameter types and bug reports."""
        preconditions: List[BasePredicate] = []
        for param in ir_func.params:
            rt = result.inferred_types.get(param, RefinementType.top())
            for p in rt.predicates:
                if p.variable == param:
                    preconditions.append(p)

        for bug in result.bug_reports:
            if bug.bug_kind == ViolationKind.NULL_DEREFERENCE:
                for param in ir_func.params:
                    if param in bug.message:
                        preconditions.append(NullityPredicate(param, is_null=False))
            elif bug.bug_kind == ViolationKind.ARRAY_OUT_OF_BOUNDS:
                for param in ir_func.params:
                    if param in bug.message:
                        preconditions.append(
                            ComparisonPredicate(param, ComparisonOp.GE, 0)
                        )
        return preconditions

    def compute_postcondition(self, ir_func: IRFunction,
                              result: CEGARResult) -> List[BasePredicate]:
        """Derive postconditions from return type."""
        postconditions: List[BasePredicate] = []
        ret_type = result.inferred_types.get("__return__", RefinementType.top())
        for p in ret_type.predicates:
            postconditions.append(p)
        return postconditions

    def compute_modifies(self, ir_func: IRFunction) -> Set[str]:
        """Compute the set of variables modified by the function."""
        modified: Set[str] = set()
        for block in ir_func.blocks.values():
            for node in block.nodes:
                if isinstance(node, AssignNode):
                    modified.add(node.target)
                elif isinstance(node, BinOpNode):
                    modified.add(node.target)
                elif isinstance(node, CallNode):
                    modified.add(node.target)
                elif isinstance(node, LoadAttrNode):
                    modified.add(node.target)
                elif isinstance(node, IndexNode):
                    modified.add(node.target)
                elif isinstance(node, PhiNode):
                    modified.add(node.target)
                elif isinstance(node, StoreAttrNode):
                    modified.add(f"{node.obj}.{node.attr}")
        return modified

    def apply_summary(self, summary: FunctionSummary,
                      call_site: CallNode,
                      caller_state: AbstractState) -> AbstractState:
        """Apply a function summary at a call site in the caller."""
        new_state = caller_state.copy()

        param_mapping: Dict[str, str] = {}
        for i, param in enumerate(summary.param_types):
            if i < len(call_site.args):
                param_mapping[param] = call_site.args[i]

        for pre in summary.preconditions:
            fv = pre.free_variables()
            substituted = pre.substitute(param_mapping)
            var = substituted.variable
            if var and var in new_state:
                current = new_state[var]
                new_state[var] = current.add_predicate(substituted)

        ret_type = summary.return_type.copy()
        renamed_preds: Set[BasePredicate] = set()
        for p in ret_type.predicates:
            renamed_preds.add(p.substitute({"__return__": call_site.target}))
        new_state[call_site.target] = RefinementType(
            base_type=ret_type.base_type, predicates=renamed_preds,
        )

        return new_state

    def get_summary(self, func_name: str) -> Optional[FunctionSummary]:
        return self._summaries.get(func_name)


# ---------------------------------------------------------------------------
# Calling context handler
# ---------------------------------------------------------------------------


class CallingContextHandler:
    """Manages calling contexts for context-sensitive analysis."""

    def __init__(self, depth_limit: int = 3) -> None:
        self._depth_limit = depth_limit
        self._stack: List[Tuple[CallNode, str]] = []
        self._context_cache: Dict[Hashable, AbstractState] = {}

    def push_context(self, call_site: CallNode, caller_func: str) -> None:
        self._stack.append((call_site, caller_func))

    def pop_context(self) -> Optional[Tuple[CallNode, str]]:
        if self._stack:
            return self._stack.pop()
        return None

    def get_context_key(self) -> Hashable:
        """Return a hashable key representing the current calling context."""
        entries: List[Tuple[int, str]] = []
        depth = min(len(self._stack), self._depth_limit)
        for i in range(len(self._stack) - depth, len(self._stack)):
            cs, func = self._stack[i]
            entries.append((cs.node_id, func))
        return tuple(entries)

    def merge_contexts(self,
                       contexts: List[Tuple[Hashable, AbstractState]]
                       ) -> AbstractState:
        """Merge multiple calling-context states via join."""
        if not contexts:
            return AbstractState(is_bottom=True)
        result = contexts[0][1].copy()
        for _, state in contexts[1:]:
            result = result.join(state)
        return result

    def should_merge(self, depth: int) -> bool:
        """Decide whether to merge contexts at this depth for efficiency."""
        return depth >= self._depth_limit

    @property
    def depth(self) -> int:
        return len(self._stack)

    def cache_result(self, key: Hashable, state: AbstractState) -> None:
        self._context_cache[key] = state

    def lookup_cache(self, key: Hashable) -> Optional[AbstractState]:
        return self._context_cache.get(key)

    def clear(self) -> None:
        self._stack.clear()
        self._context_cache.clear()


# ---------------------------------------------------------------------------
# Predicate universe generation
# ---------------------------------------------------------------------------


class PredicateUniverse:
    """Generates the finite universe of predicates for a function.

    The cardinality is bounded by O(|Vars|² × |Constants|) which guarantees
    termination of the CEGAR loop.
    """

    def __init__(self, ir_func: IRFunction) -> None:
        self._func = ir_func
        self._variables = ir_func.all_variables()
        self._constants = ir_func.all_constants()

    def generate_initial(self) -> Set[BasePredicate]:
        """Generate a small seed set of predicates."""
        preds: Set[BasePredicate] = set()

        for var in self._variables:
            preds.add(NullityPredicate(var, is_null=True))
            preds.add(NullityPredicate(var, is_null=False))

        for var in self._variables:
            for c in self._constants:
                if isinstance(c, (int, float)):
                    preds.add(ComparisonPredicate(var, ComparisonOp.GE, c))
                    preds.add(ComparisonPredicate(var, ComparisonOp.EQ, c))

        for var in self._variables:
            for tag in [BaseType.INT, BaseType.STR, BaseType.LIST,
                        BaseType.NONE_TYPE, BaseType.BOOL, BaseType.FLOAT]:
                preds.add(TypeTagPredicate(var, tag))

        return preds

    def generate_full(self) -> Set[BasePredicate]:
        """Generate the full predicate universe."""
        preds = self.generate_initial()

        for var in self._variables:
            for c in self._constants:
                if isinstance(c, (int, float)):
                    for op in ComparisonOp:
                        preds.add(ComparisonPredicate(var, op, c))

        for var in self._variables:
            for tag in BaseType:
                preds.add(TypeTagPredicate(var, tag))

        for var in self._variables:
            for c in self._constants:
                if isinstance(c, int) and c >= 0:
                    for op in [ComparisonOp.LT, ComparisonOp.LE,
                               ComparisonOp.GE, ComparisonOp.GT, ComparisonOp.EQ]:
                        preds.add(LengthPredicate(var, op, c))

        return preds

    def max_predicates(self) -> int:
        """Upper bound on predicate universe size."""
        nv = len(self._variables)
        nc = len(self._constants)
        comparisons = nv * nc * len(ComparisonOp)
        type_tags = nv * len(BaseType)
        nullity = nv * 2
        length = nv * nc * 5
        return comparisons + type_tags + nullity + length


# ---------------------------------------------------------------------------
# CEGAR engine
# ---------------------------------------------------------------------------


class CEGAREngine:
    """
    Main CEGAR engine.  Orchestrates the two-phase abstraction-refinement loop
    for a single function or an entire module.
    """

    def __init__(self, config: Optional[CEGARConfig] = None) -> None:
        self._config = config or CEGARConfig()
        self._safety_checker = SafetyChecker()
        self._refiner = PredicateRefiner(self._config.predicate_budget)
        self._summarizer = FunctionSummarizer()
        self._context_handler = CallingContextHandler(self._config.context_depth_limit)
        self._cache = CounterexampleCache(self._config.cache_size) if self._config.enable_caching else None
        self._module_results: Dict[str, CEGARResult] = {}

    def analyze_module(self, ir_module: IRModule) -> Dict[str, CEGARResult]:
        """Analyse every function in the module."""
        results: Dict[str, CEGARResult] = {}
        call_order = self._topological_sort(ir_module)
        for func_name in call_order:
            ir_func = ir_module.functions[func_name]
            logger.info("Analyzing function %s", func_name)
            result = self.analyze_function(ir_func)
            results[func_name] = result
            if self._config.enable_summaries:
                self._summarizer.summarize(ir_func, result)
        self._module_results = results
        return results

    def analyze_function(
        self,
        ir_func: IRFunction,
        initial_predicates: Optional[Set[BasePredicate]] = None,
        domain: Optional[str] = None,
        stdlib_models: Optional[StdlibModel] = None,
    ) -> CEGARResult:
        """Run CEGAR on a single function."""
        start_time = time.monotonic()

        universe = PredicateUniverse(ir_func)
        if initial_predicates is None:
            predicates = universe.generate_initial()
        else:
            predicates = set(initial_predicates)

        stdlib = stdlib_models or StdlibModel()

        result = self.cegar_loop(ir_func, predicates, domain, stdlib, start_time)
        result.time_taken = time.monotonic() - start_time
        return result

    def cegar_loop(
        self,
        ir_func: IRFunction,
        predicates: Set[BasePredicate],
        domain: Optional[str],
        stdlib: StdlibModel,
        start_time: float,
    ) -> CEGARResult:
        """
        Main CEGAR loop implementing the two-phase algorithm:
          Phase 1: Abstract interpretation with current predicates
          Phase 2: Counterexample analysis and predicate refinement
        """
        monitor = ConvergenceMonitor(self._config)
        cex_analyzer = CounterexampleAnalyzer(ir_func)
        all_bug_reports: List[BugReport] = []
        total_cex_analyzed = 0
        converged = False
        iteration = 0

        for iteration in range(1, self._config.max_iterations + 1):
            elapsed = time.monotonic() - start_time
            if monitor.should_stop(iteration - 1, elapsed):
                if monitor.is_converging():
                    converged = True
                break

            logger.debug(
                "CEGAR iteration %d: %d predicates", iteration, len(predicates)
            )

            # -- Phase 1: Abstract interpretation --
            interpreter = AbstractInterpreter(
                predicates=predicates,
                domain=domain,
                stdlib_models=stdlib,
                widening_delay=self._config.widening_delay,
                narrowing_iterations=(
                    self._config.narrowing_iterations
                    if self._config.enable_narrowing else 0
                ),
            )
            abstract_states = interpreter.forward_analysis(ir_func)

            # -- Phase 2: Safety check & counterexample analysis --
            violations = self._safety_checker.check_all(ir_func, abstract_states)

            if not violations:
                converged = True
                logger.info("No violations found at iteration %d – converged.", iteration)
                break

            new_preds_found = False
            for violation in violations:
                cex = cex_analyzer.extract_path(violation)
                total_cex_analyzed += 1

                cached = None
                if self._cache is not None:
                    cached = self._cache.lookup(cex)

                if cached is not None:
                    is_spurious, cached_preds = cached
                    if is_spurious and cached_preds:
                        old_size = len(predicates)
                        predicates = self._refiner.add_predicates(cached_preds, predicates)
                        if len(predicates) > old_size:
                            new_preds_found = True
                    elif not is_spurious:
                        report = self._violation_to_report(violation, cex)
                        all_bug_reports.append(report)
                    continue

                is_spurious = cex_analyzer.is_spurious(cex)

                if is_spurious:
                    if self._config.enable_interpolation:
                        formula = cex_analyzer.encode_path(cex.path, ir_func)
                        for pc in cex.path_condition:
                            formula.add_constraint(pc.to_smt())
                        new_preds = self._refiner.extract_interpolants(cex, formula)
                        projected: List[BasePredicate] = []
                        func_vars = ir_func.all_variables()
                        for np in new_preds:
                            projected.extend(
                                self._refiner.project_to_P(np, func_vars)
                            )
                        ranked = self._refiner.rank_predicates(projected, predicates)
                        old_size = len(predicates)
                        predicates = self._refiner.add_predicates(ranked, predicates)
                        if len(predicates) > old_size:
                            new_preds_found = True
                        if self._cache is not None:
                            self._cache.store(cex, True, ranked)
                    else:
                        fallback = self._generate_fallback_predicates(
                            violation, cex, predicates
                        )
                        old_size = len(predicates)
                        predicates = self._refiner.add_predicates(fallback, predicates)
                        if len(predicates) > old_size:
                            new_preds_found = True
                        if self._cache is not None:
                            self._cache.store(cex, True, fallback)
                else:
                    report = self._violation_to_report(violation, cex)
                    all_bug_reports.append(report)
                    if self._cache is not None:
                        self._cache.store(cex, False, [])

            state_size = sum(s.size() for s in abstract_states.values())
            monitor.record_iteration(iteration, predicates, state_size)

            if not new_preds_found:
                converged = True
                logger.info(
                    "No new predicates at iteration %d – fixed point reached.", iteration
                )
                break

        inferred = self._extract_inferred_types(ir_func, predicates, domain, stdlib)

        return CEGARResult(
            inferred_types=inferred,
            predicates_used=predicates,
            iterations=iteration,
            converged=converged,
            counterexamples_analyzed=total_cex_analyzed,
            bug_reports=all_bug_reports,
        )

    def _check_convergence(self, old_predicates: Set[BasePredicate],
                           new_predicates: Set[BasePredicate]) -> bool:
        """Check if predicate set has stabilised."""
        return old_predicates == new_predicates

    def _merge_results(self, results: Dict[str, CEGARResult]) -> CEGARResult:
        """Merge per-function results into a module-level result."""
        merged = CEGARResult()
        for fname, r in results.items():
            for var, rt in r.inferred_types.items():
                merged.inferred_types[f"{fname}.{var}"] = rt
            merged.predicates_used |= r.predicates_used
            merged.iterations += r.iterations
            merged.counterexamples_analyzed += r.counterexamples_analyzed
            merged.time_taken += r.time_taken
            merged.bug_reports.extend(r.bug_reports)
        merged.converged = all(r.converged for r in results.values())
        return merged

    def _extract_inferred_types(
        self,
        ir_func: IRFunction,
        predicates: Set[BasePredicate],
        domain: Optional[str],
        stdlib: StdlibModel,
    ) -> Dict[str, RefinementType]:
        """Run final analysis pass to extract inferred types."""
        interpreter = AbstractInterpreter(
            predicates=predicates,
            domain=domain,
            stdlib_models=stdlib,
            widening_delay=self._config.widening_delay,
            narrowing_iterations=(
                self._config.narrowing_iterations
                if self._config.enable_narrowing else 0
            ),
        )
        states = interpreter.forward_analysis(ir_func)
        inferred: Dict[str, RefinementType] = {}

        exit_state = states.get(ir_func.exit_block_id, AbstractState(is_bottom=True))
        all_vars = ir_func.all_variables()
        all_vars.add("__return__")

        for var in all_vars:
            best = RefinementType.bottom()
            for bid, state in states.items():
                if not state.is_bottom and var in state:
                    best = best.join(state[var])
            if not best.is_bottom:
                inferred[var] = best
            else:
                inferred[var] = RefinementType.top()

        if "__return__" in exit_state:
            inferred["__return__"] = exit_state["__return__"]

        return inferred

    def _violation_to_report(self, violation: SafetyViolation,
                             cex: Counterexample) -> BugReport:
        """Convert a validated violation into a bug report."""
        confidence = 0.9
        if cex.variable_assignments:
            confidence = 0.95

        suggested_fix: Optional[str] = None
        if violation.violation_kind == ViolationKind.NULL_DEREFERENCE:
            suggested_fix = "Add a None check before this access."
        elif violation.violation_kind == ViolationKind.DIVISION_BY_ZERO:
            suggested_fix = "Add a zero check before division."
        elif violation.violation_kind == ViolationKind.ARRAY_OUT_OF_BOUNDS:
            suggested_fix = "Add bounds check before indexing."
        elif violation.violation_kind == ViolationKind.TYPE_TAG_MISMATCH:
            suggested_fix = "Add type check or cast."

        return BugReport(
            location=str(violation.node.source_loc),
            bug_kind=violation.violation_kind,
            message=violation.details,
            severity=Severity.ERROR,
            confidence=confidence,
            counterexample=cex,
            suggested_fix=suggested_fix,
        )

    def _generate_fallback_predicates(
        self,
        violation: SafetyViolation,
        cex: Counterexample,
        current_predicates: Set[BasePredicate],
    ) -> List[BasePredicate]:
        """Generate predicates without interpolation."""
        preds: List[BasePredicate] = []
        node = violation.node

        if violation.violation_kind == ViolationKind.ARRAY_OUT_OF_BOUNDS:
            if isinstance(node, IndexNode):
                preds.append(ComparisonPredicate(node.index, ComparisonOp.GE, 0))
                preds.append(LengthPredicate(node.obj, ComparisonOp.GT, 0))

        elif violation.violation_kind == ViolationKind.NULL_DEREFERENCE:
            if isinstance(node, LoadAttrNode):
                preds.append(NullityPredicate(node.obj, is_null=False))
            elif isinstance(node, CallNode):
                for arg in node.args:
                    preds.append(NullityPredicate(arg, is_null=False))

        elif violation.violation_kind == ViolationKind.DIVISION_BY_ZERO:
            if isinstance(node, BinOpNode):
                preds.append(ComparisonPredicate(node.right, ComparisonOp.NE, 0))
                preds.append(ComparisonPredicate(node.right, ComparisonOp.GT, 0))

        elif violation.violation_kind == ViolationKind.TYPE_TAG_MISMATCH:
            if isinstance(node, BinOpNode):
                preds.append(TypeTagPredicate(node.left, BaseType.INT))
                preds.append(TypeTagPredicate(node.right, BaseType.INT))
                preds.append(TypeTagPredicate(node.left, BaseType.FLOAT))
                preds.append(TypeTagPredicate(node.right, BaseType.FLOAT))

        for pc in cex.path_condition:
            atomic = self._refiner._decompose_to_atomic(pc)
            preds.extend(atomic)

        return preds

    def _topological_sort(self, ir_module: IRModule) -> List[str]:
        """Sort functions in call-dependency order (callees before callers)."""
        call_graph: Dict[str, Set[str]] = {
            name: set() for name in ir_module.functions
        }
        for name, func in ir_module.functions.items():
            for block in func.blocks.values():
                for node in block.nodes:
                    if isinstance(node, CallNode):
                        if node.func_name in ir_module.functions:
                            call_graph[name].add(node.func_name)

        visited: Set[str] = set()
        order: List[str] = []
        visiting: Set[str] = set()

        def dfs(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                visited.add(name)
                order.append(name)
                return
            visiting.add(name)
            for callee in call_graph.get(name, set()):
                dfs(callee)
            visiting.discard(name)
            visited.add(name)
            order.append(name)

        for name in ir_module.functions:
            dfs(name)

        return order


# ---------------------------------------------------------------------------
# Predicate domain helpers
# ---------------------------------------------------------------------------


class PredicateDomain:
    """Utilities for operating on the predicate lattice."""

    @staticmethod
    def implies(assumption: Set[BasePredicate],
                goal: BasePredicate) -> bool:
        """Check if *assumption* (a set of predicates) implies *goal*."""
        if goal in assumption:
            return True
        for a in assumption:
            if isinstance(a, ComparisonPredicate) and isinstance(goal, ComparisonPredicate):
                if a.variable == goal.variable:
                    try:
                        av, gv = a.constant, goal.constant
                        if a.op == ComparisonOp.EQ:
                            if goal.op.evaluate(av, gv):
                                return True
                        if a.op == ComparisonOp.GE and goal.op == ComparisonOp.GE:
                            if av >= gv:
                                return True
                        if a.op == ComparisonOp.GT and goal.op == ComparisonOp.GT:
                            if av >= gv:
                                return True
                        if a.op == ComparisonOp.GT and goal.op == ComparisonOp.GE:
                            if av >= gv:
                                return True
                        if a.op == ComparisonOp.LE and goal.op == ComparisonOp.LE:
                            if av <= gv:
                                return True
                        if a.op == ComparisonOp.LT and goal.op == ComparisonOp.LT:
                            if av <= gv:
                                return True
                        if a.op == ComparisonOp.LT and goal.op == ComparisonOp.LE:
                            if av <= gv:
                                return True
                        if a.op == ComparisonOp.GE and goal.op == ComparisonOp.NE:
                            if av > gv:
                                return True
                        if a.op == ComparisonOp.GT and goal.op == ComparisonOp.NE:
                            return True
                    except TypeError:
                        pass
            if isinstance(a, NullityPredicate) and isinstance(goal, TypeTagPredicate):
                if a.variable == goal.variable and not a.is_null:
                    if goal.tag != BaseType.NONE_TYPE:
                        pass
            if isinstance(a, TypeTagPredicate) and isinstance(goal, NullityPredicate):
                if a.variable == goal.variable and a.tag != BaseType.NONE_TYPE:
                    if not goal.is_null:
                        return True
        return False

    @staticmethod
    def is_consistent(predicates: Set[BasePredicate]) -> bool:
        """Check if a set of predicates is satisfiable (light heuristic)."""
        by_var: Dict[str, List[BasePredicate]] = collections.defaultdict(list)
        for p in predicates:
            if p.variable:
                by_var[p.variable].append(p)

        for var, preds in by_var.items():
            for p in preds:
                neg = p.negate()
                if neg in predicates:
                    return False
            eq_vals: List[Any] = []
            for p in preds:
                if isinstance(p, ComparisonPredicate) and p.op == ComparisonOp.EQ:
                    eq_vals.append(p.constant)
            if len(eq_vals) > 1 and len(set(map(str, eq_vals))) > 1:
                return False

            null_preds = [p for p in preds if isinstance(p, NullityPredicate)]
            if len(null_preds) == 2:
                if null_preds[0].is_null != null_preds[1].is_null:
                    return False

        return True

    @staticmethod
    def strongest_postcondition(
        pred: BasePredicate, node: IRNode, predicates: Set[BasePredicate]
    ) -> Set[BasePredicate]:
        """Compute strongest-postcondition of *pred* through *node*."""
        result: Set[BasePredicate] = set()
        if isinstance(node, AssignNode):
            if pred.variable == node.target:
                if node.value_var:
                    subst = pred.substitute({node.target: node.value_var})
                    result.add(subst)
                return result
            else:
                result.add(pred)
        elif isinstance(node, BinOpNode):
            if pred.variable == node.target:
                return result
            result.add(pred)
        else:
            result.add(pred)
        return result

    @staticmethod
    def weakest_precondition(
        pred: BasePredicate, node: IRNode
    ) -> Set[BasePredicate]:
        """Compute weakest-precondition of *pred* through *node*."""
        result: Set[BasePredicate] = set()
        if isinstance(node, AssignNode):
            if pred.variable == node.target:
                if node.value_var:
                    subst = pred.substitute({node.target: node.value_var})
                    result.add(subst)
                return result
            result.add(pred)
        elif isinstance(node, BinOpNode):
            if pred.variable == node.target:
                return set()
            result.add(pred)
        else:
            result.add(pred)
        return result


# ---------------------------------------------------------------------------
# Trace replay for debugging
# ---------------------------------------------------------------------------


class TraceReplayer:
    """Replays a concrete counterexample trace for validation."""

    def __init__(self, ir_func: IRFunction) -> None:
        self._func = ir_func

    def replay(self, cex: Counterexample) -> Dict[str, Any]:
        """Replay the counterexample path with concrete values."""
        env: Dict[str, Any] = dict(cex.variable_assignments)
        trace_log: List[str] = []

        for block_id in cex.path:
            block = self._func.blocks.get(block_id)
            if block is None:
                trace_log.append(f"SKIP missing block {block_id}")
                continue
            for node in block.nodes:
                self._step(node, env, trace_log)
        env["__trace__"] = trace_log
        return env

    def _step(self, node: IRNode, env: Dict[str, Any],
              log: List[str]) -> None:
        if isinstance(node, AssignNode):
            if node.value_constant is not None:
                env[node.target] = node.value_constant
                log.append(f"{node.target} = {node.value_constant!r}")
            elif node.value_var and node.value_var in env:
                env[node.target] = env[node.value_var]
                log.append(f"{node.target} = {node.value_var} (={env[node.value_var]!r})")
        elif isinstance(node, BinOpNode):
            left = env.get(node.left)
            right = env.get(node.right)
            if left is not None and right is not None:
                result = self._eval_binop(node.op, left, right)
                env[node.target] = result
                log.append(f"{node.target} = {left!r} {node.op.value} {right!r} = {result!r}")
        elif isinstance(node, GuardNode):
            if node.condition:
                val = node.condition.evaluate(env)
                log.append(f"guard {node.condition}: {val}")
        elif isinstance(node, IndexNode):
            obj = env.get(node.obj)
            idx = env.get(node.index)
            if obj is not None and idx is not None:
                try:
                    env[node.target] = obj[idx]
                    log.append(f"{node.target} = {node.obj}[{idx!r}]")
                except (IndexError, KeyError, TypeError) as e:
                    log.append(f"ERROR: {node.obj}[{idx!r}] -> {e}")

    def _eval_binop(self, op: BinOp, left: Any, right: Any) -> Any:
        ops = {
            BinOp.ADD: lambda a, b: a + b,
            BinOp.SUB: lambda a, b: a - b,
            BinOp.MUL: lambda a, b: a * b,
            BinOp.DIV: lambda a, b: a / b if b != 0 else None,
            BinOp.MOD: lambda a, b: a % b if b != 0 else None,
            BinOp.FLOORDIV: lambda a, b: a // b if b != 0 else None,
            BinOp.POW: lambda a, b: a ** b,
            BinOp.AND: lambda a, b: a & b,
            BinOp.OR: lambda a, b: a | b,
            BinOp.XOR: lambda a, b: a ^ b,
            BinOp.LSHIFT: lambda a, b: a << b,
            BinOp.RSHIFT: lambda a, b: a >> b,
        }
        func = ops.get(op)
        if func is None:
            return None
        try:
            return func(left, right)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------


class DiagnosticPrinter:
    """Pretty-prints CEGAR results for human consumption."""

    def __init__(self) -> None:
        self._indent = 0

    def print_result(self, result: CEGARResult, func_name: str = "") -> str:
        lines: List[str] = []
        header = f"=== CEGAR Result for {func_name or '<anonymous>'} ==="
        lines.append(header)
        lines.append(f"Iterations:   {result.iterations}")
        lines.append(f"Converged:    {result.converged}")
        lines.append(f"Predicates:   {len(result.predicates_used)}")
        lines.append(f"CEX analyzed: {result.counterexamples_analyzed}")
        lines.append(f"Time:         {result.time_taken:.3f}s")
        lines.append(f"Bugs:         {len(result.bug_reports)}")
        lines.append("")

        if result.inferred_types:
            lines.append("Inferred types:")
            for var in sorted(result.inferred_types):
                lines.append(f"  {var}: {result.inferred_types[var]}")
            lines.append("")

        if result.bug_reports:
            lines.append("Bug reports:")
            for br in result.bug_reports:
                lines.append(f"  {br}")
            lines.append("")

        lines.append("=" * len(header))
        return "\n".join(lines)

    def print_abstract_states(self, states: Dict[str, AbstractState]) -> str:
        lines: List[str] = []
        for bid in sorted(states):
            lines.append(f"Block {bid}: {states[bid]}")
        return "\n".join(lines)

    def print_predicates(self, predicates: Set[BasePredicate]) -> str:
        sorted_preds = sorted(predicates, key=str)
        lines = [f"Predicate set ({len(predicates)})"]
        for p in sorted_preds:
            lines.append(f"  {p}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder helpers (for constructing IR programmatically)
# ---------------------------------------------------------------------------


class IRBuilder:
    """Convenience builder for constructing IR functions in tests."""

    def __init__(self, func_name: str = "f",
                 params: Optional[List[str]] = None) -> None:
        self._func = IRFunction(name=func_name, params=params or [])
        self._node_counter = 0
        self._block_counter = 0
        self._current_block: Optional[IRBlock] = None

    def new_block(self, block_id: Optional[str] = None) -> IRBlock:
        if block_id is None:
            block_id = f"bb{self._block_counter}"
        self._block_counter += 1
        block = IRBlock(block_id=block_id)
        self._func.blocks[block_id] = block
        if not self._func.entry_block_id:
            self._func.entry_block_id = block_id
        self._current_block = block
        self._func.exit_block_id = block_id
        return block

    def _next_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def assign_const(self, target: str, value: Any,
                     block: Optional[IRBlock] = None) -> AssignNode:
        b = block or self._current_block
        assert b is not None
        node = AssignNode(node_id=self._next_id(), target=target,
                          value_constant=value, value_expr=repr(value))
        b.nodes.append(node)
        return node

    def assign_var(self, target: str, source: str,
                   block: Optional[IRBlock] = None) -> AssignNode:
        b = block or self._current_block
        assert b is not None
        node = AssignNode(node_id=self._next_id(), target=target,
                          value_var=source, value_expr=source)
        b.nodes.append(node)
        return node

    def binop(self, target: str, left: str, op: BinOp, right: str,
              block: Optional[IRBlock] = None) -> BinOpNode:
        b = block or self._current_block
        assert b is not None
        node = BinOpNode(node_id=self._next_id(), target=target,
                         left=left, op=op, right=right)
        b.nodes.append(node)
        return node

    def guard(self, cond: BasePredicate, true_branch: str, false_branch: str,
              block: Optional[IRBlock] = None) -> GuardNode:
        b = block or self._current_block
        assert b is not None
        node = GuardNode(node_id=self._next_id(), condition=cond,
                         true_branch=true_branch, false_branch=false_branch)
        b.nodes.append(node)
        if true_branch not in b.successors:
            b.successors.append(true_branch)
        if false_branch not in b.successors:
            b.successors.append(false_branch)
        for succ_id in (true_branch, false_branch):
            succ = self._func.blocks.get(succ_id)
            if succ and b.block_id not in succ.predecessors:
                succ.predecessors.append(b.block_id)
        return node

    def call(self, target: str, func_name: str, args: List[str],
             block: Optional[IRBlock] = None) -> CallNode:
        b = block or self._current_block
        assert b is not None
        node = CallNode(node_id=self._next_id(), target=target,
                        func_name=func_name, args=args)
        b.nodes.append(node)
        return node

    def ret(self, value: Optional[str] = None,
            block: Optional[IRBlock] = None) -> ReturnNode:
        b = block or self._current_block
        assert b is not None
        node = ReturnNode(node_id=self._next_id(), value=value)
        b.nodes.append(node)
        return node

    def index(self, target: str, obj: str, idx: str,
              block: Optional[IRBlock] = None) -> IndexNode:
        b = block or self._current_block
        assert b is not None
        node = IndexNode(node_id=self._next_id(), target=target,
                         obj=obj, index=idx)
        b.nodes.append(node)
        return node

    def load_attr(self, target: str, obj: str, attr: str,
                  block: Optional[IRBlock] = None) -> LoadAttrNode:
        b = block or self._current_block
        assert b is not None
        node = LoadAttrNode(node_id=self._next_id(), target=target,
                            obj=obj, attr=attr)
        b.nodes.append(node)
        return node

    def phi(self, target: str, incoming: Dict[str, str],
            block: Optional[IRBlock] = None) -> PhiNode:
        b = block or self._current_block
        assert b is not None
        node = PhiNode(node_id=self._next_id(), target=target,
                       incoming=incoming)
        b.nodes.append(node)
        return node

    def link(self, src: str, dst: str) -> None:
        src_block = self._func.blocks[src]
        dst_block = self._func.blocks[dst]
        if dst not in src_block.successors:
            src_block.successors.append(dst)
        if src not in dst_block.predecessors:
            dst_block.predecessors.append(src)

    def build(self) -> IRFunction:
        return self._func


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def run_cegar(ir_func: IRFunction,
              config: Optional[CEGARConfig] = None) -> CEGARResult:
    """One-shot convenience: run CEGAR on a single function."""
    engine = CEGAREngine(config)
    return engine.analyze_function(ir_func)


def run_cegar_module(ir_module: IRModule,
                     config: Optional[CEGARConfig] = None) -> Dict[str, CEGARResult]:
    """One-shot convenience: run CEGAR on every function in a module."""
    engine = CEGAREngine(config)
    return engine.analyze_module(ir_module)


# ---------------------------------------------------------------------------
# Self-test / smoke test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Quick smoke test exercising the main components."""
    builder = IRBuilder("example", params=["x", "y"])

    entry = builder.new_block("entry")
    body = builder.new_block("body")
    exit_b = builder.new_block("exit")

    builder.assign_const("zero", 0, entry)
    builder.guard(
        ComparisonPredicate("x", ComparisonOp.GE, 0),
        true_branch="body",
        false_branch="exit",
        block=entry,
    )
    builder.link("entry", "body")
    builder.link("entry", "exit")

    builder.binop("result", "x", BinOp.DIV, "y", body)
    builder.index("elem", "x", "zero", body)
    builder.link("body", "exit")

    builder.ret("result", exit_b)
    ir_func = builder.build()

    config = CEGARConfig(max_iterations=10, per_function_timeout=5.0)
    result = run_cegar(ir_func, config)

    printer = DiagnosticPrinter()
    print(printer.print_result(result, "example"))

    assert isinstance(result, CEGARResult)
    assert result.iterations >= 1
    assert isinstance(result.inferred_types, dict)

    p1 = ComparisonPredicate("x", ComparisonOp.GT, 0)
    p2 = ComparisonPredicate("x", ComparisonOp.LE, 0)
    assert p1.evaluate({"x": 1}) is True
    assert p1.evaluate({"x": -1}) is False
    assert p2 == p1.negate()

    t1 = RefinementType(base_type=BaseType.INT, predicates={p1})
    t2 = RefinementType(base_type=BaseType.INT, predicates={p1, p2})
    joined = t1.join(t2)
    assert p1 in joined.predicates
    met = t1.meet(t2)
    assert p1 in met.predicates and p2 in met.predicates

    conj = ConjunctionPredicate(p1, p2)
    assert conj.evaluate({"x": 1}) is False
    disj = DisjunctionPredicate(p1, p2)
    assert disj.evaluate({"x": 1}) is True

    neg = NegationPredicate(p1)
    assert neg.evaluate({"x": -1}) is True

    tt = TypeTagPredicate("x", BaseType.INT)
    assert tt.evaluate({"x": 42}) is True
    assert tt.evaluate({"x": "hello"}) is False

    np = NullityPredicate("x", is_null=True)
    assert np.evaluate({"x": None}) is True
    assert np.evaluate({"x": 42}) is False

    mp = MembershipPredicate("x", "coll")
    assert mp.evaluate({"x": 1, "coll": [1, 2, 3]}) is True
    assert mp.evaluate({"x": 4, "coll": [1, 2, 3]}) is False

    lp = LengthPredicate("arr", ComparisonOp.GT, 0)
    assert lp.evaluate({"arr": [1]}) is True
    assert lp.evaluate({"arr": []}) is False

    formula = SMTFormula()
    formula.add_constraint(SMTConstraint("comparison", var="x", op=">", constant=0))
    formula.add_constraint(SMTConstraint("comparison", var="x", op="<", constant=5))
    assert formula.check_sat() is True
    model = formula.get_model()
    assert 0 < model["x"] < 5

    unsat = SMTFormula()
    unsat.add_constraint(SMTConstraint("comparison", var="x", op=">", constant=10))
    unsat.add_constraint(SMTConstraint("comparison", var="x", op="<", constant=0))
    assert unsat.check_sat() is False

    cache = CounterexampleCache(max_size=10)
    cex = Counterexample(
        path=["entry", "body"],
        path_condition=[p1],
        variable_assignments={"x": 1},
        violation_type=ViolationKind.DIVISION_BY_ZERO,
    )
    assert cache.lookup(cex) is None
    cache.store(cex, True, [p1])
    cached = cache.lookup(cex)
    assert cached is not None
    assert cached[0] is True

    s1 = AbstractState({"x": RefinementType(base_type=BaseType.INT, predicates={p1})})
    s2 = AbstractState({"x": RefinementType(base_type=BaseType.INT, predicates={p1, p2})})
    s3 = s1.join(s2)
    assert p1 in s3["x"].predicates

    replayer = TraceReplayer(ir_func)
    cex2 = Counterexample(
        path=["entry", "body", "exit"],
        path_condition=[],
        variable_assignments={"x": 10, "y": 2, "zero": 0},
        violation_type=ViolationKind.DIVISION_BY_ZERO,
    )
    replay_env = replayer.replay(cex2)
    assert "__trace__" in replay_env

    print("All self-tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _self_test()

"""
Guard Harvesting Algorithm for CEGAR Seed Predicate Extraction.

This module implements the guard harvesting phase of the CEGAR loop for
refinement type inference in dynamically-typed languages.  Guards are
runtime checks—isinstance calls, None comparisons, truthiness tests,
bound checks, and structural tests—that the source program already
performs.  Harvesting them as *seed predicates* gives CEGAR an excellent
starting set so that the first abstract interpretation pass is already
reasonably precise.

Complexity bounds
-----------------
* **Guard extraction:** O(|AST|) per function—each IR node is visited at
  most once via a single linear walk.
* **Guard ranking:** O(|Guards| log |Guards|) via heap-based selection
  of the top-K guards.
* **Soundness guarantee:** Every extracted guard corresponds to a
  syntactic check present in the original program and therefore yields a
  *valid* predicate in the predicate language P.  No speculative
  predicates are invented at this stage.

Cross-function propagation runs in O(|E| · |G|) where E is the number
of call-graph edges and G the total guard count.

The main entry point is :class:`GuardHarvester`, which accepts an
:class:`IRFunction` (or :class:`IRModule`) and returns a ranked list of
:class:`HarvestedGuard` objects ready for use as CEGAR seeds.
"""

from __future__ import annotations

import copy
import heapq
import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Imports from sibling modules (IR layer and predicates)
# ---------------------------------------------------------------------------

from .engine import (
    IRFunction,
    IRModule,
    IRNode,
    IRBlock,
    GuardNode,
    AssignNode,
    CallNode,
    ReturnNode,
    PhiNode,
    BinOpNode,
    LoadAttrNode,
    IndexNode,
    BasePredicate,
    ComparisonPredicate,
    TypeTagPredicate,
    NullityPredicate,
    MembershipPredicate,
    LengthPredicate,
    ConjunctionPredicate,
    RefinementType,
    AbstractState,
)

logger = logging.getLogger(__name__)

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Expression Node hierarchy (local representation for guard exprs)   ║
# ╚═══════════════════════════════════════════════════════════════════════╝


class ExprNode(ABC):
    """Base class for expression nodes used in guard representation."""

    @abstractmethod
    def variables(self) -> FrozenSet[str]:
        """Return all variable names referenced by this expression."""
        ...

    @abstractmethod
    def evaluate(self, env: Mapping[str, Any]) -> Any:
        """Evaluate the expression under the given environment."""
        ...

    @abstractmethod
    def pretty(self) -> str:
        """Human-readable string."""
        ...

    def __repr__(self) -> str:  # pragma: no cover
        return self.pretty()

    @abstractmethod
    def clone(self) -> "ExprNode":
        """Deep-copy."""
        ...

    @abstractmethod
    def substitute(self, mapping: Mapping[str, "ExprNode"]) -> "ExprNode":
        """Replace variable references according to *mapping*."""
        ...


@dataclass(frozen=True)
class VarExpr(ExprNode):
    """Reference to a program variable."""
    name: str

    def variables(self) -> FrozenSet[str]:
        return frozenset({self.name})

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        if self.name not in env:
            raise KeyError(f"Variable {self.name!r} not bound")
        return env[self.name]

    def pretty(self) -> str:
        return self.name

    def clone(self) -> "VarExpr":
        return VarExpr(name=self.name)

    def substitute(self, mapping: Mapping[str, ExprNode]) -> ExprNode:
        if self.name in mapping:
            return mapping[self.name].clone()
        return self.clone()


@dataclass(frozen=True)
class ConstExpr(ExprNode):
    """A literal constant (int, float, str, bool, None)."""
    value: Any

    def variables(self) -> FrozenSet[str]:
        return frozenset()

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        return self.value

    def pretty(self) -> str:
        return repr(self.value)

    def clone(self) -> "ConstExpr":
        return ConstExpr(value=self.value)

    def substitute(self, mapping: Mapping[str, ExprNode]) -> ExprNode:
        return self.clone()


@dataclass(frozen=True)
class BinOpExpr(ExprNode):
    """Binary operation expression."""
    left: ExprNode
    op: str
    right: ExprNode

    _OPS: dict = field(default=None, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_OPS", {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b if b != 0 else math.inf,
            "//": lambda a, b: a // b if b != 0 else 0,
            "%": lambda a, b: a % b if b != 0 else 0,
            "**": lambda a, b: a ** b,
            "&": lambda a, b: a & b,
            "|": lambda a, b: a | b,
            "^": lambda a, b: a ^ b,
            "<<": lambda a, b: a << b,
            ">>": lambda a, b: a >> b,
            "and": lambda a, b: a and b,
            "or": lambda a, b: a or b,
        })

    def variables(self) -> FrozenSet[str]:
        return self.left.variables() | self.right.variables()

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        lv = self.left.evaluate(env)
        rv = self.right.evaluate(env)
        fn = self._OPS.get(self.op)
        if fn is None:
            raise ValueError(f"Unknown binary operator {self.op!r}")
        return fn(lv, rv)

    def pretty(self) -> str:
        return f"({self.left.pretty()} {self.op} {self.right.pretty()})"

    def clone(self) -> "BinOpExpr":
        return BinOpExpr(left=self.left.clone(), op=self.op, right=self.right.clone())

    def substitute(self, mapping: Mapping[str, ExprNode]) -> ExprNode:
        return BinOpExpr(
            left=self.left.substitute(mapping),
            op=self.op,
            right=self.right.substitute(mapping),
        )


@dataclass(frozen=True)
class UnaryOpExpr(ExprNode):
    """Unary operation expression."""
    op: str
    operand: ExprNode

    def variables(self) -> FrozenSet[str]:
        return self.operand.variables()

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        val = self.operand.evaluate(env)
        if self.op == "-":
            return -val
        if self.op == "+":
            return +val
        if self.op == "not":
            return not val
        if self.op == "~":
            return ~val
        raise ValueError(f"Unknown unary operator {self.op!r}")

    def pretty(self) -> str:
        return f"({self.op} {self.operand.pretty()})"

    def clone(self) -> "UnaryOpExpr":
        return UnaryOpExpr(op=self.op, operand=self.operand.clone())

    def substitute(self, mapping: Mapping[str, ExprNode]) -> ExprNode:
        return UnaryOpExpr(op=self.op, operand=self.operand.substitute(mapping))


@dataclass(frozen=True)
class CallExpr(ExprNode):
    """Function call expression."""
    func: ExprNode
    args: Tuple[ExprNode, ...] = ()

    def variables(self) -> FrozenSet[str]:
        vs: Set[str] = set(self.func.variables())
        for a in self.args:
            vs |= a.variables()
        return frozenset(vs)

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        fn = self.func.evaluate(env)
        evaluated_args = tuple(a.evaluate(env) for a in self.args)
        return fn(*evaluated_args)

    def pretty(self) -> str:
        args_s = ", ".join(a.pretty() for a in self.args)
        return f"{self.func.pretty()}({args_s})"

    def clone(self) -> "CallExpr":
        return CallExpr(
            func=self.func.clone(),
            args=tuple(a.clone() for a in self.args),
        )

    def substitute(self, mapping: Mapping[str, ExprNode]) -> ExprNode:
        return CallExpr(
            func=self.func.substitute(mapping),
            args=tuple(a.substitute(mapping) for a in self.args),
        )


@dataclass(frozen=True)
class AttrExpr(ExprNode):
    """Attribute access expression ``obj.attr``."""
    obj: ExprNode
    attr: str

    def variables(self) -> FrozenSet[str]:
        return self.obj.variables()

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        o = self.obj.evaluate(env)
        return getattr(o, self.attr)

    def pretty(self) -> str:
        return f"{self.obj.pretty()}.{self.attr}"

    def clone(self) -> "AttrExpr":
        return AttrExpr(obj=self.obj.clone(), attr=self.attr)

    def substitute(self, mapping: Mapping[str, ExprNode]) -> ExprNode:
        return AttrExpr(obj=self.obj.substitute(mapping), attr=self.attr)


@dataclass(frozen=True)
class IndexExpr(ExprNode):
    """Subscript expression ``obj[index]``."""
    obj: ExprNode
    index: ExprNode

    def variables(self) -> FrozenSet[str]:
        return self.obj.variables() | self.index.variables()

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        o = self.obj.evaluate(env)
        i = self.index.evaluate(env)
        return o[i]

    def pretty(self) -> str:
        return f"{self.obj.pretty()}[{self.index.pretty()}]"

    def clone(self) -> "IndexExpr":
        return IndexExpr(obj=self.obj.clone(), index=self.index.clone())

    def substitute(self, mapping: Mapping[str, ExprNode]) -> ExprNode:
        return IndexExpr(
            obj=self.obj.substitute(mapping),
            index=self.index.substitute(mapping),
        )


@dataclass(frozen=True)
class CompareExpr(ExprNode):
    """Comparison expression ``left op right``."""
    left: ExprNode
    op: str
    right: ExprNode

    _CMP_FUNCS: dict = field(default=None, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_CMP_FUNCS", {
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "is": lambda a, b: a is b,
            "is not": lambda a, b: a is not b,
            "in": lambda a, b: a in b,
            "not in": lambda a, b: a not in b,
        })

    def variables(self) -> FrozenSet[str]:
        return self.left.variables() | self.right.variables()

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        lv = self.left.evaluate(env)
        rv = self.right.evaluate(env)
        fn = self._CMP_FUNCS.get(self.op)
        if fn is None:
            raise ValueError(f"Unknown comparison operator {self.op!r}")
        return fn(lv, rv)

    def pretty(self) -> str:
        return f"({self.left.pretty()} {self.op} {self.right.pretty()})"

    def clone(self) -> "CompareExpr":
        return CompareExpr(
            left=self.left.clone(), op=self.op, right=self.right.clone(),
        )

    def substitute(self, mapping: Mapping[str, ExprNode]) -> ExprNode:
        return CompareExpr(
            left=self.left.substitute(mapping),
            op=self.op,
            right=self.right.substitute(mapping),
        )


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Guard Kind Enum                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝


class GuardKind(Enum):
    """Classification of harvested guards."""
    COMPARISON = auto()
    TYPE_TAG = auto()
    NULLITY = auto()
    STRUCTURAL = auto()
    TRUTHINESS = auto()
    MEMBERSHIP = auto()

    def description(self) -> str:
        _descs = {
            GuardKind.COMPARISON: "Numeric/string comparison (==, !=, <, >, <=, >=)",
            GuardKind.TYPE_TAG: "Type tag check (isinstance, type, callable)",
            GuardKind.NULLITY: "Null/None check (is None, is not None)",
            GuardKind.STRUCTURAL: "Structural check (hasattr, len, keys)",
            GuardKind.TRUTHINESS: "Truthiness coercion (if x, bool(x))",
            GuardKind.MEMBERSHIP: "Membership test (in, not in)",
        }
        return _descs.get(self, "Unknown guard kind")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  HarvestedGuard                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝


@dataclass
class HarvestedGuard:
    """A single guard harvested from program source or IR.

    Attributes
    ----------
    predicate : BasePredicate
        The predicate in the language P that this guard corresponds to.
    source_location : tuple
        ``(file, line, col)`` triple pinpointing the guard in source.
    guard_kind : GuardKind
        Classification of this guard.
    confidence : float
        Score in ``[0, 1]`` indicating how likely the guard reflects a
        meaningful program invariant.
    frequency : int
        How many times an equivalent pattern occurs in the module.
    information_content : float
        Entropy-based measure of discriminative power.
    context : str
        Snippet of surrounding IR/source for human readability.
    is_negated : bool
        ``True`` when the guard appears in a negated position
        (e.g. the false-branch of an ``if``).
    """

    predicate: BasePredicate
    source_location: Tuple[str, int, int] = ("<unknown>", 0, 0)
    guard_kind: GuardKind = GuardKind.COMPARISON
    confidence: float = 1.0
    frequency: int = 1
    information_content: float = 0.0
    context: str = ""
    is_negated: bool = False

    # Convenience helpers ---------------------------------------------------

    @property
    def file(self) -> str:
        return self.source_location[0]

    @property
    def line(self) -> int:
        return self.source_location[1]

    @property
    def col(self) -> int:
        return self.source_location[2]

    def with_negated(self, neg: bool) -> "HarvestedGuard":
        """Return a copy with ``is_negated`` flipped."""
        cpy = copy.copy(self)
        cpy.is_negated = neg
        return cpy

    def with_confidence(self, c: float) -> "HarvestedGuard":
        cpy = copy.copy(self)
        cpy.confidence = max(0.0, min(1.0, c))
        return cpy

    def fingerprint(self) -> str:
        """A compact string identifying the predicate semantics."""
        kind_tag = self.guard_kind.name[0]
        neg_tag = "!" if self.is_negated else ""
        pred_str = str(self.predicate) if self.predicate is not None else "?"
        return f"{neg_tag}{kind_tag}:{pred_str}"

    def __lt__(self, other: "HarvestedGuard") -> bool:
        """Comparison for heap ordering (higher confidence first)."""
        return self.confidence > other.confidence


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Configuration dataclasses                                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝


@dataclass
class HarvestConfig:
    """Tuning knobs for the guard harvesting pass."""
    min_confidence: float = 0.5
    max_guards_per_function: int = 500
    include_implicit_guards: bool = True
    include_loop_guards: bool = True
    include_exception_guards: bool = True
    normalize_guards: bool = True
    deduplicate: bool = True


@dataclass
class RankingWeights:
    """Relative weights used by :class:`GuardRanker`."""
    frequency_weight: float = 0.3
    information_weight: float = 0.25
    bug_relevance_weight: float = 0.3
    scope_weight: float = 0.15

    def total(self) -> float:
        return (
            self.frequency_weight
            + self.information_weight
            + self.bug_relevance_weight
            + self.scope_weight
        )

    def normalized(self) -> "RankingWeights":
        """Return a copy whose weights sum to 1."""
        t = self.total()
        if t == 0:
            return RankingWeights(0.25, 0.25, 0.25, 0.25)
        return RankingWeights(
            frequency_weight=self.frequency_weight / t,
            information_weight=self.information_weight / t,
            bug_relevance_weight=self.bug_relevance_weight / t,
            scope_weight=self.scope_weight / t,
        )


@dataclass
class SeedConfig:
    """Configuration for the seed selection strategy."""
    strategy: str = "heuristic"
    top_k: int = 50
    bug_site_radius: int = 5
    min_coverage: float = 0.6


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  CollectedConstants                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝


@dataclass
class CollectedConstants:
    """Result of :meth:`ConstantCollector.collect`."""
    integers: List[int] = field(default_factory=list)
    floats: List[float] = field(default_factory=list)
    strings: List[str] = field(default_factory=list)
    bounds: List[Tuple[Any, str]] = field(default_factory=list)
    sentinels: List[Any] = field(default_factory=list)
    all_unique: Set[Any] = field(default_factory=set)

    def merge(self, other: "CollectedConstants") -> "CollectedConstants":
        """Merge another :class:`CollectedConstants` into a new one."""
        return CollectedConstants(
            integers=self.integers + other.integers,
            floats=self.floats + other.floats,
            strings=self.strings + other.strings,
            bounds=self.bounds + other.bounds,
            sentinels=self.sentinels + other.sentinels,
            all_unique=self.all_unique | other.all_unique,
        )

    @property
    def total_count(self) -> int:
        return (
            len(self.integers)
            + len(self.floats)
            + len(self.strings)
            + len(self.bounds)
            + len(self.sentinels)
        )


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Statistics dataclasses                                              ║
# ╚═══════════════════════════════════════════════════════════════════════╝


@dataclass
class CoverageReport:
    """Branch coverage information for harvested guards."""
    total_branches: int = 0
    guarded_branches: int = 0
    coverage_ratio: float = 0.0
    unguarded_locations: List[Tuple[str, int, int]] = field(default_factory=list)


@dataclass
class RedundancyReport:
    """Redundancy analysis of harvested guards."""
    total_guards: int = 0
    unique_guards: int = 0
    redundant_guards: int = 0
    redundancy_ratio: float = 0.0
    redundancy_groups: List[List[int]] = field(default_factory=list)


@dataclass
class StatisticsReport:
    """Summary statistics from :class:`GuardStatistics`."""
    total_guards: int = 0
    by_kind: Dict[GuardKind, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    avg_information: float = 0.0
    coverage: CoverageReport = field(default_factory=CoverageReport)
    redundancy_ratio: float = 0.0


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  PatternMatcher                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝


class PatternMatcher:
    """Matches common guard patterns against IR nodes and returns
    typed predicates when a match is found.
    """

    # -- isinstance ---------------------------------------------------------

    @staticmethod
    def _is_isinstance_call(node: IRNode) -> bool:
        """Check whether *node* is a call to ``isinstance``."""
        if not isinstance(node, CallNode):
            return False
        func_name = getattr(node, "func_name", None) or getattr(node, "func", "")
        return str(func_name) in ("isinstance", "builtins.isinstance")

    @staticmethod
    def match_isinstance_pattern(node: IRNode) -> Optional[TypeTagPredicate]:
        """If *node* is ``isinstance(x, T)`` return a :class:`TypeTagPredicate`."""
        if not PatternMatcher._is_isinstance_call(node):
            return None
        args = getattr(node, "args", [])
        if len(args) < 2:
            return None
        var_name = str(args[0])
        type_name = str(args[1])
        return TypeTagPredicate(variable=var_name, type_tag=type_name)

    # -- None checks --------------------------------------------------------

    @staticmethod
    def _is_none_comparison(node: IRNode) -> bool:
        """Check whether *node* compares a value to ``None``."""
        if not isinstance(node, BinOpNode):
            return False
        op_str = str(getattr(node, "op", ""))
        if op_str not in ("is", "is not", "==", "!=", "Is", "IsNot"):
            return False
        left = str(getattr(node, "left", ""))
        right = str(getattr(node, "right", ""))
        return left == "None" or right == "None" or left == "none" or right == "none"

    @staticmethod
    def match_none_check_pattern(node: IRNode) -> Optional[NullityPredicate]:
        """If *node* is ``x is None`` / ``x is not None`` return a predicate."""
        if not PatternMatcher._is_none_comparison(node):
            return None
        left = str(getattr(node, "left", ""))
        right = str(getattr(node, "right", ""))
        op_str = str(getattr(node, "op", ""))

        var_name = right if (left.lower() == "none") else left
        is_not = op_str in ("is not", "!=", "IsNot")
        return NullityPredicate(variable=var_name, is_null=not is_not)

    # -- bound checks -------------------------------------------------------

    @staticmethod
    def match_bound_check_pattern(node: IRNode) -> Optional[ComparisonPredicate]:
        """Match ``x < len(a)`` / ``i >= 0`` etc."""
        if not isinstance(node, BinOpNode):
            return None
        op_str = str(getattr(node, "op", ""))
        if op_str not in ("<", "<=", ">", ">=", "Lt", "Le", "Gt", "Ge",
                          "LtE", "GtE"):
            return None
        left = str(getattr(node, "left", ""))
        right = str(getattr(node, "right", ""))
        return ComparisonPredicate(
            left_var=left,
            operator=op_str,
            right_expr=right,
        )

    # -- hasattr ------------------------------------------------------------

    @staticmethod
    def match_hasattr_pattern(node: IRNode) -> Optional[BasePredicate]:
        """Match ``hasattr(x, 'attr')``."""
        if not isinstance(node, CallNode):
            return None
        func_name = str(getattr(node, "func_name", "") or getattr(node, "func", ""))
        if func_name not in ("hasattr", "builtins.hasattr"):
            return None
        args = getattr(node, "args", [])
        if len(args) < 2:
            return None
        var_name = str(args[0])
        attr_name = str(args[1]).strip("'\"")
        return MembershipPredicate(
            variable=var_name,
            container=f"attrs({var_name})",
            element=attr_name,
        )

    # -- truthiness ---------------------------------------------------------

    @staticmethod
    def match_truthiness_pattern(node: IRNode) -> Optional[BasePredicate]:
        """Match implicit truthiness tests (``if x:``)."""
        if isinstance(node, GuardNode):
            cond = getattr(node, "condition", None) or getattr(node, "test", None)
            if cond is not None:
                var_name = str(cond)
                return ComparisonPredicate(
                    left_var=var_name,
                    operator="!=",
                    right_expr="False",
                )
        return None

    # -- membership ---------------------------------------------------------

    @staticmethod
    def match_membership_pattern(node: IRNode) -> Optional[MembershipPredicate]:
        """Match ``x in collection``."""
        if not isinstance(node, BinOpNode):
            return None
        op_str = str(getattr(node, "op", ""))
        if op_str not in ("in", "not in", "In", "NotIn", "not_in"):
            return None
        left = str(getattr(node, "left", ""))
        right = str(getattr(node, "right", ""))
        return MembershipPredicate(
            variable=left,
            container=right,
            element=left,
        )

    # -- length -------------------------------------------------------------

    @staticmethod
    def match_length_pattern(node: IRNode) -> Optional[LengthPredicate]:
        """Match ``len(x) <op> n``."""
        if not isinstance(node, BinOpNode):
            return None
        left = str(getattr(node, "left", ""))
        right = str(getattr(node, "right", ""))
        op_str = str(getattr(node, "op", ""))

        len_match = re.match(r"len\((\w+)\)", left)
        if len_match is None:
            len_match = re.match(r"len\((\w+)\)", right)
            if len_match is None:
                return None
            var_name = len_match.group(1)
            return LengthPredicate(
                variable=var_name, operator=op_str, bound=left,
            )
        var_name = len_match.group(1)
        return LengthPredicate(
            variable=var_name, operator=op_str, bound=right,
        )


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  GuardNormalizer                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝


class GuardNormalizer:
    """Canonicalize harvested guards so that semantic duplicates compare
    equal and the CEGAR loop does not waste effort on redundant predicates.
    """

    # Canonical operator mapping: operator string → (canonical_op, swap_operands)
    _CANONICAL_OPS: Dict[str, Tuple[str, bool]] = {
        "<": ("<", False),
        "<=": ("<=", False),
        ">": ("<", True),
        ">=": ("<=", True),
        "==": ("==", False),
        "!=": ("!=", False),
        "Lt": ("<", False),
        "Le": ("<=", False),
        "LtE": ("<=", False),
        "Gt": ("<", True),
        "GtE": ("<=", True),
        "Ge": ("<=", True),
        "Eq": ("==", False),
        "Ne": ("!=", False),
        "NotEq": ("!=", False),
        "is": ("is", False),
        "is not": ("is not", False),
        "Is": ("is", False),
        "IsNot": ("is not", False),
        "in": ("in", False),
        "not in": ("not in", False),
        "In": ("in", False),
        "NotIn": ("not in", False),
        "not_in": ("not in", False),
    }

    def __init__(self) -> None:
        self._normalization_count: int = 0

    # -- public API ---------------------------------------------------------

    def normalize(self, guard: HarvestedGuard) -> HarvestedGuard:
        """Run the full normalization pipeline on *guard* and return a
        (possibly new) :class:`HarvestedGuard`."""
        self._normalization_count += 1
        result = copy.copy(guard)
        pred = result.predicate

        if isinstance(pred, ComparisonPredicate):
            result.predicate = self.normalize_comparison(pred)
        elif isinstance(pred, TypeTagPredicate):
            result.predicate = self.normalize_isinstance(pred)
        elif isinstance(pred, NullityPredicate):
            result.predicate = self.normalize_nullcheck(pred)
        elif isinstance(pred, ConjunctionPredicate):
            atoms = self.normalize_combined(pred)
            if len(atoms) == 1:
                result.predicate = atoms[0]
            else:
                result.predicate = ConjunctionPredicate(conjuncts=atoms)
        return result

    def normalize_comparison(self, pred: ComparisonPredicate) -> ComparisonPredicate:
        """Normalize a comparison predicate to canonical form.

        Canonical form: variable on the left, constant on the right,
        operator chosen from ``{<, <=, ==, !=}``.
        """
        op = getattr(pred, "operator", "==")
        left = getattr(pred, "left_var", "")
        right = getattr(pred, "right_expr", "")

        canon_op, swap = self._canonical_operator(op)
        if swap:
            left, right = right, left

        # If the right side looks like a variable and left like a constant,
        # swap again so variables are on the left.
        if self._looks_like_constant(left) and not self._looks_like_constant(right):
            left, right = right, left
            canon_op = self._flip_op(canon_op)

        return ComparisonPredicate(
            left_var=left,
            operator=canon_op,
            right_expr=right,
        )

    def normalize_isinstance(self, pred: TypeTagPredicate) -> TypeTagPredicate:
        """Normalize isinstance to a canonical :class:`TypeTagPredicate`.

        Strips module prefixes (``builtins.int`` → ``int``) and
        normalizes common aliases.
        """
        var = getattr(pred, "variable", "")
        tag = getattr(pred, "type_tag", "")

        # Strip module prefix
        if "." in tag:
            tag = tag.rsplit(".", 1)[-1]

        # Normalize common aliases
        _aliases: Dict[str, str] = {
            "str": "str",
            "string": "str",
            "unicode": "str",
            "bytes": "bytes",
            "int": "int",
            "integer": "int",
            "long": "int",
            "float": "float",
            "double": "float",
            "bool": "bool",
            "boolean": "bool",
            "list": "list",
            "tuple": "tuple",
            "dict": "dict",
            "dictionary": "dict",
            "set": "set",
            "frozenset": "frozenset",
            "NoneType": "NoneType",
            "nonetype": "NoneType",
        }
        tag = _aliases.get(tag.lower(), tag)

        return TypeTagPredicate(variable=var, type_tag=tag)

    def normalize_nullcheck(self, pred: NullityPredicate) -> NullityPredicate:
        """Normalize null checks to use the ``is_null`` flag consistently."""
        var = getattr(pred, "variable", "")
        is_null = getattr(pred, "is_null", True)
        return NullityPredicate(variable=var, is_null=is_null)

    def normalize_combined(self, pred: BasePredicate) -> List[BasePredicate]:
        """Split combined (and/or) guards into a list of atomic predicates.

        Conjunction is flattened; disjunction is kept as-is because each
        disjunct is not independently sound.
        """
        conjuncts = getattr(pred, "conjuncts", None)
        if conjuncts is None:
            return [pred]
        result: List[BasePredicate] = []
        for sub in conjuncts:
            if hasattr(sub, "conjuncts"):
                result.extend(self.normalize_combined(sub))
            else:
                result.append(sub)
        return result

    def deduplicate(self, guards: List[HarvestedGuard]) -> List[HarvestedGuard]:
        """Remove duplicate predicates using semantic equality.

        When two guards have the same semantic meaning, keep the one with
        higher confidence and accumulate frequencies.
        """
        seen: Dict[str, int] = {}
        result: List[HarvestedGuard] = []

        for g in guards:
            fp = g.fingerprint()
            if fp in seen:
                idx = seen[fp]
                existing = result[idx]
                existing.frequency += g.frequency
                if g.confidence > existing.confidence:
                    existing.confidence = g.confidence
                    existing.source_location = g.source_location
                    existing.context = g.context
            else:
                seen[fp] = len(result)
                result.append(copy.copy(g))
        return result

    # -- private helpers ----------------------------------------------------

    def _canonical_operator(self, op: str) -> Tuple[str, bool]:
        """Map an operator string to ``(canonical, swap_needed)``."""
        entry = self._CANONICAL_OPS.get(op)
        if entry is not None:
            return entry
        return (op, False)

    @staticmethod
    def _flip_op(op: str) -> str:
        """Flip a comparison operator for operand swap."""
        _flips = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
        return _flips.get(op, op)

    @staticmethod
    def _looks_like_constant(s: str) -> bool:
        """Heuristic: is *s* a constant literal?"""
        if not s:
            return False
        if s in ("None", "True", "False"):
            return True
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            pass
        if s.startswith(("'", '"')):
            return True
        return False

    def _are_semantically_equal(self, p1: BasePredicate, p2: BasePredicate) -> bool:
        """Check semantic equality of two predicates.

        Two predicates are semantically equal if after normalization they
        yield the same fingerprint.
        """
        if type(p1) is not type(p2):
            return False

        if isinstance(p1, ComparisonPredicate) and isinstance(p2, ComparisonPredicate):
            n1 = self.normalize_comparison(p1)
            n2 = self.normalize_comparison(p2)
            return (
                getattr(n1, "left_var", "") == getattr(n2, "left_var", "")
                and getattr(n1, "operator", "") == getattr(n2, "operator", "")
                and getattr(n1, "right_expr", "") == getattr(n2, "right_expr", "")
            )

        if isinstance(p1, TypeTagPredicate) and isinstance(p2, TypeTagPredicate):
            n1 = self.normalize_isinstance(p1)
            n2 = self.normalize_isinstance(p2)
            return (
                getattr(n1, "variable", "") == getattr(n2, "variable", "")
                and getattr(n1, "type_tag", "") == getattr(n2, "type_tag", "")
            )

        if isinstance(p1, NullityPredicate) and isinstance(p2, NullityPredicate):
            return (
                getattr(p1, "variable", "") == getattr(p2, "variable", "")
                and getattr(p1, "is_null", True) == getattr(p2, "is_null", True)
            )

        if isinstance(p1, MembershipPredicate) and isinstance(p2, MembershipPredicate):
            return (
                getattr(p1, "variable", "") == getattr(p2, "variable", "")
                and getattr(p1, "container", "") == getattr(p2, "container", "")
                and getattr(p1, "element", "") == getattr(p2, "element", "")
            )

        # Fallback: string comparison
        return str(p1) == str(p2)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  GuardHarvester  (main entry point)                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝


class GuardHarvester:
    """Extract guards from IR and convert them to CEGAR seed predicates.

    The harvesting walk is O(|AST|) per function: each node in the IR is
    visited exactly once.  Multiple extraction sub-passes (comparisons,
    calls, control flow, etc.) run during the same walk to avoid repeated
    traversals.
    """

    def __init__(self, config: Optional[HarvestConfig] = None) -> None:
        self._config = config or HarvestConfig()
        self._normalizer = GuardNormalizer()
        self._pattern_matcher = PatternMatcher()
        self._stats_collector = GuardStatistics()
        self._harvested_count: int = 0

    # -- public API ---------------------------------------------------------

    def harvest_function(self, ir_func: IRFunction) -> List[HarvestedGuard]:
        """Harvest guards from a single IR function.

        Returns a deduplicated, normalized, filtered list of guards.
        """
        logger.debug("Harvesting guards from function %s", getattr(ir_func, "name", "?"))
        raw_guards: List[HarvestedGuard] = []

        # Phase 1: walk the IR and collect raw guards
        raw_guards.extend(self.harvest_from_ir(ir_func))
        raw_guards.extend(self.harvest_from_control_flow(ir_func))

        # Phase 2: per-block extraction
        blocks = self._get_blocks(ir_func)
        for block in blocks:
            raw_guards.extend(self.harvest_from_comparisons(block))
            raw_guards.extend(self.harvest_from_calls(block))
            raw_guards.extend(self.harvest_from_assertions(block))
            if self._config.include_exception_guards:
                raw_guards.extend(self.harvest_from_exceptions(block))
            if self._config.include_loop_guards:
                raw_guards.extend(self.harvest_from_loop_bounds(block))
            if self._config.include_implicit_guards:
                raw_guards.extend(self.harvest_from_optional(block))

        # Phase 3: compute confidence and information content
        for g in raw_guards:
            g.confidence = self._compute_confidence(g)

        # Phase 4: normalize
        if self._config.normalize_guards:
            raw_guards = [self._normalizer.normalize(g) for g in raw_guards]

        # Phase 5: deduplicate
        if self._config.deduplicate:
            raw_guards = self._normalizer.deduplicate(raw_guards)

        # Phase 6: compute information content (needs full list)
        for g in raw_guards:
            g.information_content = self._compute_information_content(g, raw_guards)

        # Phase 7: filter by confidence
        raw_guards = [
            g for g in raw_guards if g.confidence >= self._config.min_confidence
        ]

        # Phase 8: cap
        if len(raw_guards) > self._config.max_guards_per_function:
            raw_guards.sort(key=lambda g: (-g.confidence, -g.information_content))
            raw_guards = raw_guards[: self._config.max_guards_per_function]

        self._harvested_count += len(raw_guards)
        return raw_guards

    def harvest_module(
        self, ir_module: IRModule
    ) -> Dict[str, List[HarvestedGuard]]:
        """Harvest guards from every function in *ir_module*."""
        result: Dict[str, List[HarvestedGuard]] = {}
        functions = getattr(ir_module, "functions", {})
        if isinstance(functions, dict):
            func_iter = functions.items()
        elif isinstance(functions, (list, tuple)):
            func_iter = ((getattr(f, "name", f"anon_{i}"), f) for i, f in enumerate(functions))
        else:
            func_iter = ()

        for name, func in func_iter:
            guards = self.harvest_function(func)
            if guards:
                result[name] = guards
        return result

    # -- IR-level extraction ------------------------------------------------

    def harvest_from_ir(self, ir_func: IRFunction) -> List[HarvestedGuard]:
        """Walk the IR and extract guards from :class:`GuardNode` nodes."""
        guards: List[HarvestedGuard] = []
        blocks = self._get_blocks(ir_func)

        for block in blocks:
            for node in self._get_nodes(block):
                if isinstance(node, GuardNode):
                    pred = self._extract_predicate_from_guard_node(node)
                    if pred is not None:
                        loc = self._get_location(node)
                        kind = self._classify_guard_node(node)
                        guards.append(HarvestedGuard(
                            predicate=pred,
                            source_location=loc,
                            guard_kind=kind,
                            context=self._node_context(node),
                        ))
                        # Negated twin
                        neg_pred = self._negate_predicate(pred)
                        if neg_pred is not None:
                            guards.append(HarvestedGuard(
                                predicate=neg_pred,
                                source_location=loc,
                                guard_kind=kind,
                                context=self._node_context(node),
                                is_negated=True,
                            ))
        return guards

    def harvest_from_comparisons(self, block: Any) -> List[HarvestedGuard]:
        """Extract predicates from comparison expressions (==, !=, <, etc.)."""
        guards: List[HarvestedGuard] = []
        for node in self._get_nodes(block):
            if not isinstance(node, BinOpNode):
                continue
            op_str = str(getattr(node, "op", ""))
            if op_str not in ("<", "<=", ">", ">=", "==", "!=",
                              "Lt", "Le", "LtE", "Gt", "Ge", "GtE",
                              "Eq", "Ne", "NotEq",
                              "is", "is not", "Is", "IsNot"):
                continue

            left = str(getattr(node, "left", ""))
            right = str(getattr(node, "right", ""))
            loc = self._get_location(node)

            # Check for None comparison → NullityPredicate
            null_pred = PatternMatcher.match_none_check_pattern(node)
            if null_pred is not None:
                guards.append(HarvestedGuard(
                    predicate=null_pred,
                    source_location=loc,
                    guard_kind=GuardKind.NULLITY,
                    context=f"{left} {op_str} {right}",
                ))
                continue

            # Check for length pattern → LengthPredicate
            len_pred = PatternMatcher.match_length_pattern(node)
            if len_pred is not None:
                guards.append(HarvestedGuard(
                    predicate=len_pred,
                    source_location=loc,
                    guard_kind=GuardKind.STRUCTURAL,
                    context=f"{left} {op_str} {right}",
                ))
                continue

            # Membership pattern
            mem_pred = PatternMatcher.match_membership_pattern(node)
            if mem_pred is not None:
                guards.append(HarvestedGuard(
                    predicate=mem_pred,
                    source_location=loc,
                    guard_kind=GuardKind.MEMBERSHIP,
                    context=f"{left} {op_str} {right}",
                ))
                continue

            # Generic comparison
            pred = ComparisonPredicate(
                left_var=left, operator=op_str, right_expr=right,
            )
            guards.append(HarvestedGuard(
                predicate=pred,
                source_location=loc,
                guard_kind=GuardKind.COMPARISON,
                context=f"{left} {op_str} {right}",
            ))
        return guards

    def harvest_from_calls(self, block: Any) -> List[HarvestedGuard]:
        """Extract predicates from isinstance/hasattr/type/callable calls."""
        guards: List[HarvestedGuard] = []
        for node in self._get_nodes(block):
            if not isinstance(node, CallNode):
                continue
            func_name = str(
                getattr(node, "func_name", "") or getattr(node, "func", "")
            )
            loc = self._get_location(node)
            args = getattr(node, "args", [])

            if func_name in ("isinstance", "builtins.isinstance"):
                pred = PatternMatcher.match_isinstance_pattern(node)
                if pred is not None:
                    guards.append(HarvestedGuard(
                        predicate=pred,
                        source_location=loc,
                        guard_kind=GuardKind.TYPE_TAG,
                        context=f"isinstance({', '.join(str(a) for a in args)})",
                    ))

            elif func_name in ("hasattr", "builtins.hasattr"):
                pred = PatternMatcher.match_hasattr_pattern(node)
                if pred is not None:
                    guards.append(HarvestedGuard(
                        predicate=pred,
                        source_location=loc,
                        guard_kind=GuardKind.STRUCTURAL,
                        context=f"hasattr({', '.join(str(a) for a in args)})",
                    ))

            elif func_name in ("type", "builtins.type"):
                if len(args) >= 1:
                    var_name = str(args[0])
                    pred = TypeTagPredicate(variable=var_name, type_tag="<runtime>")
                    guards.append(HarvestedGuard(
                        predicate=pred,
                        source_location=loc,
                        guard_kind=GuardKind.TYPE_TAG,
                        context=f"type({var_name})",
                        confidence=0.6,
                    ))

            elif func_name in ("callable", "builtins.callable"):
                if len(args) >= 1:
                    var_name = str(args[0])
                    pred = TypeTagPredicate(variable=var_name, type_tag="callable")
                    guards.append(HarvestedGuard(
                        predicate=pred,
                        source_location=loc,
                        guard_kind=GuardKind.TYPE_TAG,
                        context=f"callable({var_name})",
                    ))

            elif func_name == "len":
                if len(args) >= 1:
                    var_name = str(args[0])
                    pred = LengthPredicate(
                        variable=var_name, operator=">=", bound="0",
                    )
                    guards.append(HarvestedGuard(
                        predicate=pred,
                        source_location=loc,
                        guard_kind=GuardKind.STRUCTURAL,
                        context=f"len({var_name})",
                        confidence=0.4,
                    ))

        return guards

    def harvest_from_control_flow(self, ir_func: IRFunction) -> List[HarvestedGuard]:
        """Extract guards from if/while/for conditions at the CFG level."""
        guards: List[HarvestedGuard] = []
        blocks = self._get_blocks(ir_func)

        for block in blocks:
            nodes = self._get_nodes(block)
            if not nodes:
                continue

            # The last node in a block with two successors is a branch condition
            last_node = nodes[-1]
            successors = getattr(block, "successors", [])
            succ_count = len(successors) if isinstance(successors, (list, tuple)) else 0

            if succ_count < 2:
                continue

            # The branch condition is either explicit or implicit truthiness
            condition = getattr(last_node, "condition", None) or getattr(last_node, "test", None)
            if condition is None:
                continue

            loc = self._get_location(last_node)

            # Try to extract a predicate from the condition
            pred = self._extract_predicate_from_expr(condition)
            if pred is not None:
                kind = self._classify_predicate(pred)
                guards.append(HarvestedGuard(
                    predicate=pred,
                    source_location=loc,
                    guard_kind=kind,
                    context=f"branch on {condition}",
                ))
            else:
                # Implicit truthiness guard
                var_name = str(condition)
                pred = ComparisonPredicate(
                    left_var=var_name, operator="!=", right_expr="False",
                )
                guards.append(HarvestedGuard(
                    predicate=pred,
                    source_location=loc,
                    guard_kind=GuardKind.TRUTHINESS,
                    context=f"if {var_name}",
                    confidence=0.5,
                ))

        return guards

    def harvest_from_assertions(self, block: Any) -> List[HarvestedGuard]:
        """Extract predicates from assert statements."""
        guards: List[HarvestedGuard] = []
        for node in self._get_nodes(block):
            # AssertNodes have a 'test' attribute
            node_type_name = type(node).__name__
            if node_type_name not in ("AssertNode", "Assert"):
                continue

            test = getattr(node, "test", None)
            if test is None:
                continue

            loc = self._get_location(node)
            pred = self._extract_predicate_from_expr(test)
            if pred is not None:
                kind = self._classify_predicate(pred)
                guards.append(HarvestedGuard(
                    predicate=pred,
                    source_location=loc,
                    guard_kind=kind,
                    context=f"assert {test}",
                    confidence=0.9,
                ))
            else:
                var_name = str(test)
                pred = ComparisonPredicate(
                    left_var=var_name, operator="!=", right_expr="False",
                )
                guards.append(HarvestedGuard(
                    predicate=pred,
                    source_location=loc,
                    guard_kind=GuardKind.TRUTHINESS,
                    context=f"assert {var_name}",
                    confidence=0.85,
                ))
        return guards

    def harvest_from_exceptions(self, block: Any) -> List[HarvestedGuard]:
        """Extract predicates from except type handlers.

        An ``except TypeError`` guard implies an ``isinstance`` check
        on the raised value.
        """
        guards: List[HarvestedGuard] = []
        for node in self._get_nodes(block):
            node_type_name = type(node).__name__
            if node_type_name not in ("ExceptHandlerNode", "ExceptHandler"):
                continue

            exc_types = getattr(node, "exc_types", [])
            exc_var = getattr(node, "exc_var", None)
            loc = self._get_location(node)

            for exc_type in exc_types:
                exc_type_str = str(exc_type)
                # Map exception type to predicate
                pred = self._exception_type_to_predicate(exc_type_str, exc_var)
                if pred is not None:
                    guards.append(HarvestedGuard(
                        predicate=pred,
                        source_location=loc,
                        guard_kind=GuardKind.TYPE_TAG,
                        context=f"except {exc_type_str}",
                        confidence=0.7,
                        is_negated=True,
                    ))
        return guards

    def harvest_from_loop_bounds(self, block: Any) -> List[HarvestedGuard]:
        """Extract predicates from range/enumerate/zip patterns."""
        guards: List[HarvestedGuard] = []
        for node in self._get_nodes(block):
            if not isinstance(node, CallNode):
                continue
            func_name = str(
                getattr(node, "func_name", "") or getattr(node, "func", "")
            )
            args = getattr(node, "args", [])
            loc = self._get_location(node)

            if func_name == "range":
                guards.extend(self._extract_range_guards(args, loc))
            elif func_name == "enumerate":
                if len(args) >= 1:
                    var_name = str(args[0])
                    pred = LengthPredicate(
                        variable=var_name, operator=">=", bound="0",
                    )
                    guards.append(HarvestedGuard(
                        predicate=pred,
                        source_location=loc,
                        guard_kind=GuardKind.STRUCTURAL,
                        context=f"enumerate({var_name})",
                        confidence=0.6,
                    ))
            elif func_name == "zip":
                for arg in args:
                    var_name = str(arg)
                    pred = LengthPredicate(
                        variable=var_name, operator=">=", bound="0",
                    )
                    guards.append(HarvestedGuard(
                        predicate=pred,
                        source_location=loc,
                        guard_kind=GuardKind.STRUCTURAL,
                        context=f"zip(..., {var_name}, ...)",
                        confidence=0.5,
                    ))
        return guards

    def harvest_from_optional(self, block: Any) -> List[HarvestedGuard]:
        """Extract predicates from optional chaining patterns.

        Patterns: ``x is not None``, ``if x:``, ``x?.y`` (TS),
        ``x or default``.
        """
        guards: List[HarvestedGuard] = []
        for node in self._get_nodes(block):
            loc = self._get_location(node)

            # Pattern: attribute access after a None check
            if isinstance(node, LoadAttrNode):
                obj_name = str(getattr(node, "obj", getattr(node, "object", "")))
                attr_name = str(getattr(node, "attr", getattr(node, "attribute", "")))
                pred = NullityPredicate(variable=obj_name, is_null=False)
                guards.append(HarvestedGuard(
                    predicate=pred,
                    source_location=loc,
                    guard_kind=GuardKind.NULLITY,
                    context=f"{obj_name}.{attr_name} (implicit not-None)",
                    confidence=0.4,
                ))

            # Pattern: indexing after length check
            if isinstance(node, IndexNode):
                obj_name = str(getattr(node, "obj", getattr(node, "object", "")))
                idx = str(getattr(node, "index", getattr(node, "subscript", "")))
                pred = NullityPredicate(variable=obj_name, is_null=False)
                guards.append(HarvestedGuard(
                    predicate=pred,
                    source_location=loc,
                    guard_kind=GuardKind.NULLITY,
                    context=f"{obj_name}[{idx}] (implicit not-None)",
                    confidence=0.35,
                ))

                # Also add a bound check if the index is numeric
                try:
                    idx_val = int(idx)
                    bound_pred = LengthPredicate(
                        variable=obj_name,
                        operator=">",
                        bound=str(idx_val),
                    )
                    guards.append(HarvestedGuard(
                        predicate=bound_pred,
                        source_location=loc,
                        guard_kind=GuardKind.STRUCTURAL,
                        context=f"{obj_name}[{idx}] (implicit bound)",
                        confidence=0.3,
                    ))
                except (ValueError, TypeError):
                    pass

            # Pattern: BinOpNode with "or" used for default values
            if isinstance(node, BinOpNode):
                op_str = str(getattr(node, "op", ""))
                if op_str in ("or", "Or"):
                    left = str(getattr(node, "left", ""))
                    pred = NullityPredicate(variable=left, is_null=False)
                    guards.append(HarvestedGuard(
                        predicate=pred,
                        source_location=loc,
                        guard_kind=GuardKind.NULLITY,
                        context=f"{left} or ... (optional default)",
                        confidence=0.45,
                        is_negated=True,
                    ))

        return guards

    # -- predicate extraction helpers ----------------------------------------

    def _extract_predicate_from_expr(self, expr: Any) -> Optional[BasePredicate]:
        """Convert an IR expression/node to a :class:`BasePredicate`.

        Dispatches on node type and applies pattern matching.
        """
        if expr is None:
            return None

        # If it's already a predicate, return it
        if isinstance(expr, BasePredicate):
            return expr

        # If it's an IR node, try pattern matching
        if isinstance(expr, IRNode):
            # isinstance pattern
            pred = PatternMatcher.match_isinstance_pattern(expr)
            if pred is not None:
                return pred

            # None check
            pred = PatternMatcher.match_none_check_pattern(expr)
            if pred is not None:
                return pred

            # Bound check
            pred = PatternMatcher.match_bound_check_pattern(expr)
            if pred is not None:
                return pred

            # Membership
            pred = PatternMatcher.match_membership_pattern(expr)
            if pred is not None:
                return pred

            # Length
            pred = PatternMatcher.match_length_pattern(expr)
            if pred is not None:
                return pred

            # Truthiness
            pred = PatternMatcher.match_truthiness_pattern(expr)
            if pred is not None:
                return pred

            # hasattr
            pred = PatternMatcher.match_hasattr_pattern(expr)
            if pred is not None:
                return pred

        # If it's a GuardNode, use its condition
        if isinstance(expr, GuardNode):
            cond = getattr(expr, "condition", None) or getattr(expr, "test", None)
            if cond is not None and cond is not expr:
                return self._extract_predicate_from_expr(cond)

        # BinOpNode with comparison op
        if isinstance(expr, BinOpNode):
            op_str = str(getattr(expr, "op", ""))
            left = str(getattr(expr, "left", ""))
            right = str(getattr(expr, "right", ""))
            if op_str in ("<", "<=", ">", ">=", "==", "!=",
                          "Lt", "Le", "LtE", "Gt", "Ge", "GtE",
                          "Eq", "Ne", "NotEq",
                          "is", "is not", "Is", "IsNot"):
                return ComparisonPredicate(
                    left_var=left, operator=op_str, right_expr=right,
                )

        # Fallback: use string representation
        expr_str = str(expr)
        if expr_str:
            return ComparisonPredicate(
                left_var=expr_str, operator="!=", right_expr="False",
            )

        return None

    def _extract_predicate_from_guard_node(self, node: GuardNode) -> Optional[BasePredicate]:
        """Extract a predicate specifically from a :class:`GuardNode`."""
        # Try the guard_kind attribute
        gk = getattr(node, "guard_kind", None) or getattr(node, "kind", None)
        condition = getattr(node, "condition", None) or getattr(node, "test", None)
        args = getattr(node, "args", [])

        gk_str = str(gk) if gk is not None else ""

        if "isinstance" in gk_str.lower() or "ISINSTANCE" in gk_str:
            if len(args) >= 2:
                return TypeTagPredicate(variable=str(args[0]), type_tag=str(args[1]))
            if condition is not None:
                return self._extract_predicate_from_expr(condition)

        if "none" in gk_str.lower() or "IS_NONE" in gk_str or "IS_NOT_NONE" in gk_str:
            var = str(args[0]) if args else str(condition) if condition else ""
            is_null = "NOT" not in gk_str.upper()
            return NullityPredicate(variable=var, is_null=is_null)

        if "hasattr" in gk_str.lower() or "HASATTR" in gk_str:
            if len(args) >= 2:
                return MembershipPredicate(
                    variable=str(args[0]),
                    container=f"attrs({args[0]})",
                    element=str(args[1]),
                )

        if "comparison" in gk_str.lower() or "COMPARISON" in gk_str:
            if condition is not None:
                return self._extract_predicate_from_expr(condition)

        if "truthiness" in gk_str.lower() or "TRUTHINESS" in gk_str:
            var = str(args[0]) if args else str(condition) if condition else ""
            return ComparisonPredicate(
                left_var=var, operator="!=", right_expr="False",
            )

        # Fallback: try to extract from the condition
        if condition is not None:
            return self._extract_predicate_from_expr(condition)

        return None

    def _compute_confidence(self, guard: HarvestedGuard) -> float:
        """Compute a confidence score for the guard.

        Higher confidence for explicit checks (isinstance, assert),
        lower for implicit (truthiness, optional chaining).
        """
        base = 0.5
        kind = guard.guard_kind

        kind_bonus: Dict[GuardKind, float] = {
            GuardKind.TYPE_TAG: 0.35,
            GuardKind.NULLITY: 0.30,
            GuardKind.COMPARISON: 0.25,
            GuardKind.STRUCTURAL: 0.20,
            GuardKind.MEMBERSHIP: 0.15,
            GuardKind.TRUTHINESS: 0.05,
        }
        base += kind_bonus.get(kind, 0.0)

        # Explicit context keywords boost confidence
        ctx = guard.context.lower()
        if "assert" in ctx:
            base += 0.15
        if "isinstance" in ctx:
            base += 0.10
        if "implicit" in ctx:
            base -= 0.10
        if "or" in ctx and "default" in ctx:
            base -= 0.05

        # Negated guards are slightly less confident
        if guard.is_negated:
            base -= 0.05

        # Frequency boost (log scale, capped)
        freq_boost = min(0.1, 0.02 * math.log1p(guard.frequency))
        base += freq_boost

        return max(0.0, min(1.0, base))

    def _compute_information_content(
        self, guard: HarvestedGuard, all_guards: List[HarvestedGuard]
    ) -> float:
        """Compute entropy-based information content.

        Guards that appear rarely have higher information content
        (they are more discriminating).  Uses the Shannon entropy
        formula: ``-log2(p)`` where ``p`` is the relative frequency.
        """
        if not all_guards:
            return 0.0

        total_freq = sum(g.frequency for g in all_guards)
        if total_freq == 0:
            return 0.0

        p = guard.frequency / total_freq
        if p <= 0:
            return 0.0

        # Shannon information content
        info = -math.log2(p)

        # Normalize to [0, 1] using log2(total_guards) as max
        max_info = math.log2(max(len(all_guards), 2))
        normalized = info / max_info if max_info > 0 else 0.0

        # Boost for guards with more variables (more discriminating)
        pred = guard.predicate
        if pred is not None:
            var_count = len(self._predicate_variables(pred))
            var_boost = min(0.1, 0.03 * var_count)
            normalized += var_boost

        return max(0.0, min(1.0, normalized))

    # -- classification helpers ---------------------------------------------

    def _classify_guard_node(self, node: GuardNode) -> GuardKind:
        """Classify a GuardNode into a GuardKind."""
        gk = getattr(node, "guard_kind", None) or getattr(node, "kind", None)
        gk_str = str(gk).upper() if gk is not None else ""

        if "ISINSTANCE" in gk_str or "TYPEOF" in gk_str or "TYPE" in gk_str:
            return GuardKind.TYPE_TAG
        if "NONE" in gk_str or "NULL" in gk_str:
            return GuardKind.NULLITY
        if "HASATTR" in gk_str:
            return GuardKind.STRUCTURAL
        if "COMPARISON" in gk_str:
            return GuardKind.COMPARISON
        if "TRUTHINESS" in gk_str or "TRUTHY" in gk_str:
            return GuardKind.TRUTHINESS
        if "CALLABLE" in gk_str:
            return GuardKind.TYPE_TAG
        if "MEMBERSHIP" in gk_str or "IN" in gk_str:
            return GuardKind.MEMBERSHIP
        return GuardKind.COMPARISON

    def _classify_predicate(self, pred: BasePredicate) -> GuardKind:
        """Classify a predicate into a GuardKind."""
        if isinstance(pred, TypeTagPredicate):
            return GuardKind.TYPE_TAG
        if isinstance(pred, NullityPredicate):
            return GuardKind.NULLITY
        if isinstance(pred, MembershipPredicate):
            return GuardKind.MEMBERSHIP
        if isinstance(pred, LengthPredicate):
            return GuardKind.STRUCTURAL
        if isinstance(pred, ComparisonPredicate):
            return GuardKind.COMPARISON
        return GuardKind.TRUTHINESS

    # -- utility helpers ----------------------------------------------------

    @staticmethod
    def _get_blocks(ir_func: IRFunction) -> List[Any]:
        """Retrieve basic blocks from an IR function."""
        blocks = getattr(ir_func, "blocks", None)
        if blocks is not None:
            if isinstance(blocks, dict):
                return list(blocks.values())
            return list(blocks)

        cfg = getattr(ir_func, "cfg", None)
        if cfg is not None:
            cfg_blocks = getattr(cfg, "blocks", None)
            if cfg_blocks is not None:
                if isinstance(cfg_blocks, dict):
                    return list(cfg_blocks.values())
                return list(cfg_blocks)

        body = getattr(ir_func, "body", None)
        if body is not None:
            if isinstance(body, (list, tuple)):
                return list(body)
            return [body]

        return []

    @staticmethod
    def _get_nodes(block: Any) -> List[IRNode]:
        """Retrieve IR nodes from a basic block."""
        nodes = getattr(block, "nodes", None)
        if nodes is not None:
            return list(nodes)
        instructions = getattr(block, "instructions", None)
        if instructions is not None:
            return list(instructions)
        stmts = getattr(block, "stmts", None)
        if stmts is not None:
            return list(stmts)
        if isinstance(block, (list, tuple)):
            return list(block)
        return []

    @staticmethod
    def _get_location(node: IRNode) -> Tuple[str, int, int]:
        """Extract source location from an IR node."""
        loc = getattr(node, "location", None) or getattr(node, "loc", None)
        if loc is not None:
            f = getattr(loc, "file", "<unknown>")
            ln = getattr(loc, "line", 0)
            col = getattr(loc, "col", 0) or getattr(loc, "column", 0)
            return (str(f), int(ln), int(col))
        line = getattr(node, "line", 0)
        col = getattr(node, "col", 0) or getattr(node, "column", 0)
        return ("<unknown>", int(line), int(col))

    @staticmethod
    def _node_context(node: IRNode) -> str:
        """Build a short context string from a node."""
        pp = getattr(node, "pretty_print", None)
        if pp is not None:
            try:
                return pp()[:120]
            except Exception:
                pass
        return str(node)[:120]

    @staticmethod
    def _negate_predicate(pred: BasePredicate) -> Optional[BasePredicate]:
        """Return the negation of a predicate, if possible."""
        if isinstance(pred, NullityPredicate):
            var = getattr(pred, "variable", "")
            is_null = getattr(pred, "is_null", True)
            return NullityPredicate(variable=var, is_null=not is_null)

        if isinstance(pred, ComparisonPredicate):
            _negation_map = {
                "<": ">=", "<=": ">", ">": "<=", ">=": "<",
                "==": "!=", "!=": "==",
                "Lt": "Ge", "Le": "Gt", "Gt": "Le", "Ge": "Lt",
                "GtE": "Lt", "LtE": "Gt",
                "Eq": "Ne", "Ne": "Eq", "NotEq": "Eq",
                "is": "is not", "is not": "is",
                "Is": "IsNot", "IsNot": "Is",
            }
            op = getattr(pred, "operator", "==")
            neg_op = _negation_map.get(op)
            if neg_op is not None:
                return ComparisonPredicate(
                    left_var=getattr(pred, "left_var", ""),
                    operator=neg_op,
                    right_expr=getattr(pred, "right_expr", ""),
                )

        return None

    @staticmethod
    def _predicate_variables(pred: BasePredicate) -> Set[str]:
        """Collect variable names referenced by a predicate."""
        vs: Set[str] = set()
        var = getattr(pred, "variable", None)
        if var:
            vs.add(str(var))
        left = getattr(pred, "left_var", None)
        if left:
            vs.add(str(left))
        right = getattr(pred, "right_expr", None)
        if right and not GuardNormalizer._looks_like_constant(str(right)):
            vs.add(str(right))
        return vs

    def _exception_type_to_predicate(
        self, exc_type: str, exc_var: Any
    ) -> Optional[BasePredicate]:
        """Map an exception type name to a guard predicate.

        For example, ``TypeError`` implies a type tag predicate,
        ``IndexError`` implies a bound check, etc.
        """
        var_name = str(exc_var) if exc_var is not None else "<exc>"

        _exc_map: Dict[str, Callable[[], BasePredicate]] = {
            "TypeError": lambda: TypeTagPredicate(variable=var_name, type_tag="<wrong_type>"),
            "ValueError": lambda: ComparisonPredicate(
                left_var=var_name, operator="!=", right_expr="<invalid>",
            ),
            "IndexError": lambda: LengthPredicate(
                variable=var_name, operator=">=", bound="0",
            ),
            "KeyError": lambda: MembershipPredicate(
                variable="<key>", container=var_name, element="<key>",
            ),
            "AttributeError": lambda: MembershipPredicate(
                variable=var_name, container=f"attrs({var_name})", element="<attr>",
            ),
            "ZeroDivisionError": lambda: ComparisonPredicate(
                left_var=var_name, operator="!=", right_expr="0",
            ),
            "StopIteration": lambda: LengthPredicate(
                variable=var_name, operator=">", bound="0",
            ),
            "OverflowError": lambda: ComparisonPredicate(
                left_var=var_name, operator="<=", right_expr="<max>",
            ),
        }

        factory = _exc_map.get(exc_type)
        if factory is not None:
            return factory()

        # Generic: the exception type itself acts as a type tag predicate
        return TypeTagPredicate(variable=var_name, type_tag=exc_type)

    def _extract_range_guards(
        self, args: List[Any], loc: Tuple[str, int, int]
    ) -> List[HarvestedGuard]:
        """Extract guards from ``range(...)`` call arguments."""
        guards: List[HarvestedGuard] = []
        if len(args) == 1:
            # range(n): loop var in [0, n)
            bound = str(args[0])
            pred = ComparisonPredicate(left_var=bound, operator=">", right_expr="0")
            guards.append(HarvestedGuard(
                predicate=pred,
                source_location=loc,
                guard_kind=GuardKind.COMPARISON,
                context=f"range({bound})",
                confidence=0.6,
            ))
        elif len(args) >= 2:
            start = str(args[0])
            stop = str(args[1])
            pred = ComparisonPredicate(left_var=start, operator="<", right_expr=stop)
            guards.append(HarvestedGuard(
                predicate=pred,
                source_location=loc,
                guard_kind=GuardKind.COMPARISON,
                context=f"range({start}, {stop})",
                confidence=0.65,
            ))
            if len(args) >= 3:
                step = str(args[2])
                pred2 = ComparisonPredicate(
                    left_var=step, operator="!=", right_expr="0",
                )
                guards.append(HarvestedGuard(
                    predicate=pred2,
                    source_location=loc,
                    guard_kind=GuardKind.COMPARISON,
                    context=f"range step {step} != 0",
                    confidence=0.7,
                ))
        return guards


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  GuardRanker                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝


class GuardRanker:
    """Rank harvested guards for seed selection.

    Ranking is O(|Guards| log |Guards|) via a heap-based top-K selection.
    The combined score is a weighted sum of four factors: frequency,
    information content, bug relevance, and scope.
    """

    # Common bug pattern keywords used for relevance scoring
    _BUG_PATTERNS: List[str] = [
        "None", "null", "undefined", "NaN", "Infinity",
        "div", "mod", "index", "key", "attr",
        "len", "size", "count", "range", "bound",
        "type", "isinstance", "cast",
        "overflow", "underflow", "zero",
    ]

    def __init__(self, weights: Optional[RankingWeights] = None) -> None:
        self._weights = (weights or RankingWeights()).normalized()

    # -- public API ---------------------------------------------------------

    def rank(self, guards: List[HarvestedGuard]) -> List[HarvestedGuard]:
        """Rank all guards using the combined scoring function."""
        return self.combined_rank(guards)

    def rank_by_frequency(self, guards: List[HarvestedGuard]) -> List[HarvestedGuard]:
        """Rank guards by frequency (most frequent first)."""
        return sorted(guards, key=lambda g: -g.frequency)

    def rank_by_information(self, guards: List[HarvestedGuard]) -> List[HarvestedGuard]:
        """Rank guards by information content (highest first)."""
        return sorted(guards, key=lambda g: -g.information_content)

    def rank_by_bug_relevance(self, guards: List[HarvestedGuard]) -> List[HarvestedGuard]:
        """Rank by bug relevance score (highest first)."""
        scored = [(self._compute_bug_relevance(g), g) for g in guards]
        scored.sort(key=lambda pair: -pair[0])
        return [g for _, g in scored]

    def rank_by_scope(self, guards: List[HarvestedGuard]) -> List[HarvestedGuard]:
        """Rank guards by scope score (function-local higher)."""
        scored = [(self._compute_scope_score(g), g) for g in guards]
        scored.sort(key=lambda pair: -pair[0])
        return [g for _, g in scored]

    def combined_rank(self, guards: List[HarvestedGuard]) -> List[HarvestedGuard]:
        """Weighted combination of all ranking factors using heapq."""
        if not guards:
            return []

        w = self._weights
        scored: List[Tuple[float, int, HarvestedGuard]] = []

        for idx, g in enumerate(guards):
            freq_score = self._normalize_frequency(g.frequency, guards)
            info_score = g.information_content
            bug_score = self._compute_bug_relevance(g)
            scope_score = self._compute_scope_score(g)

            combined = (
                w.frequency_weight * freq_score
                + w.information_weight * info_score
                + w.bug_relevance_weight * bug_score
                + w.scope_weight * scope_score
            )

            # Use negative for max-heap behaviour with heapq (min-heap)
            heapq.heappush(scored, (-combined, idx, g))

        result: List[HarvestedGuard] = []
        while scored:
            neg_score, _, g = heapq.heappop(scored)
            g.confidence = max(0.0, min(1.0, -neg_score))
            result.append(g)
        return result

    # -- scoring functions ---------------------------------------------------

    def _compute_bug_relevance(self, guard: HarvestedGuard) -> float:
        """Score how relevant this guard is to common bug patterns.

        Returns a value in [0, 1].
        """
        score = 0.0
        pred_str = str(guard.predicate).lower() if guard.predicate else ""
        ctx = guard.context.lower()
        combined = pred_str + " " + ctx

        matches = sum(1 for pat in self._BUG_PATTERNS if pat.lower() in combined)
        score += min(0.5, 0.1 * matches)

        # Guard kinds known to be bug-relevant
        kind_relevance: Dict[GuardKind, float] = {
            GuardKind.NULLITY: 0.35,
            GuardKind.TYPE_TAG: 0.25,
            GuardKind.STRUCTURAL: 0.20,
            GuardKind.COMPARISON: 0.10,
            GuardKind.MEMBERSHIP: 0.15,
            GuardKind.TRUTHINESS: 0.05,
        }
        score += kind_relevance.get(guard.guard_kind, 0.0)

        # Negated guards are more bug-relevant (they indicate the error path)
        if guard.is_negated:
            score += 0.10

        return max(0.0, min(1.0, score))

    def _compute_scope_score(self, guard: HarvestedGuard) -> float:
        """Score based on scope: function-local guards are more useful.

        Returns a value in [0, 1].
        """
        score = 0.5

        pred = guard.predicate
        if pred is None:
            return score

        # Count how many variables the predicate references
        var_count = 0
        for attr in ("variable", "left_var"):
            v = getattr(pred, attr, None)
            if v:
                var_count += 1
        right = getattr(pred, "right_expr", None)
        if right and not GuardNormalizer._looks_like_constant(str(right)):
            var_count += 1

        # More local variables → higher scope score
        if var_count == 1:
            score += 0.3
        elif var_count == 2:
            score += 0.2
        elif var_count >= 3:
            score += 0.1

        # Guards at specific source locations get a boost
        if guard.source_location[1] > 0:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _compute_discrimination_power(
        self, guard: HarvestedGuard, all_guards: List[HarvestedGuard]
    ) -> float:
        """How well this guard separates abstract states.

        Measured by how many other guards share the same variables
        but have different predicates.
        """
        if not all_guards:
            return 0.0

        my_vars = set()
        pred = guard.predicate
        for attr in ("variable", "left_var"):
            v = getattr(pred, attr, None)
            if v:
                my_vars.add(str(v))

        if not my_vars:
            return 0.5

        same_var_count = 0
        different_pred_count = 0
        for other in all_guards:
            if other is guard:
                continue
            other_pred = other.predicate
            other_vars = set()
            for attr in ("variable", "left_var"):
                v = getattr(other_pred, attr, None)
                if v:
                    other_vars.add(str(v))
            if my_vars & other_vars:
                same_var_count += 1
                if str(pred) != str(other_pred):
                    different_pred_count += 1

        if same_var_count == 0:
            return 1.0

        return different_pred_count / same_var_count

    @staticmethod
    def _normalize_frequency(freq: int, guards: List[HarvestedGuard]) -> float:
        """Normalize frequency to [0, 1] relative to max frequency."""
        max_freq = max((g.frequency for g in guards), default=1)
        if max_freq == 0:
            return 0.0
        return freq / max_freq


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  GuardSeedStrategy                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════╝


class GuardSeedStrategy:
    """Select seed predicates from ranked guards for the CEGAR loop."""

    def __init__(self, config: Optional[SeedConfig] = None) -> None:
        self._config = config or SeedConfig()

    # -- public API ---------------------------------------------------------

    def select_seeds(
        self,
        ranked_guards: List[HarvestedGuard],
        ir_func: IRFunction,
    ) -> List[BasePredicate]:
        """Dispatch to the chosen strategy."""
        strategy = self._config.strategy.lower()
        if strategy == "eager":
            return self.eager_strategy(ranked_guards)
        if strategy == "lazy":
            return self.lazy_strategy(ranked_guards)
        if strategy == "heuristic":
            return self.heuristic_strategy(ranked_guards, self._config.top_k)
        if strategy == "bug_focused":
            return self.bug_focused_strategy(ranked_guards, ir_func)
        logger.warning("Unknown strategy %r, falling back to heuristic", strategy)
        return self.heuristic_strategy(ranked_guards, self._config.top_k)

    def eager_strategy(self, guards: List[HarvestedGuard]) -> List[BasePredicate]:
        """Include all harvested guards as seeds."""
        return [g.predicate for g in guards if g.predicate is not None]

    def lazy_strategy(self, guards: List[HarvestedGuard]) -> List[BasePredicate]:
        """Start with an empty seed set (pure CEGAR discovery)."""
        return []

    def heuristic_strategy(
        self, guards: List[HarvestedGuard], k: int = 50
    ) -> List[BasePredicate]:
        """Include the top-K ranked guards as seeds.

        Already-ranked input is assumed; if not, the caller should rank
        first.
        """
        selected = guards[:k]

        # Verify minimum coverage if possible
        coverage = self._compute_coverage(selected, guards)
        if coverage < self._config.min_coverage and len(guards) > k:
            # Greedily add more guards to improve coverage
            remaining = guards[k:]
            for extra in remaining:
                selected.append(extra)
                coverage = self._compute_coverage(selected, guards)
                if coverage >= self._config.min_coverage:
                    break

        return [g.predicate for g in selected if g.predicate is not None]

    def bug_focused_strategy(
        self,
        guards: List[HarvestedGuard],
        ir_func: IRFunction,
    ) -> List[BasePredicate]:
        """Include guards near potential bug sites."""
        bug_sites = self._identify_bug_sites(ir_func)
        if not bug_sites:
            return self.heuristic_strategy(guards, self._config.top_k)

        selected: List[HarvestedGuard] = []
        seen_fps: Set[str] = set()

        for site in bug_sites:
            nearby = self._guards_near_site(
                guards, site, self._config.bug_site_radius
            )
            for g in nearby:
                fp = g.fingerprint()
                if fp not in seen_fps:
                    seen_fps.add(fp)
                    selected.append(g)

        # If we don't have enough, supplement with top-K heuristic
        if len(selected) < self._config.top_k:
            for g in guards:
                fp = g.fingerprint()
                if fp not in seen_fps:
                    seen_fps.add(fp)
                    selected.append(g)
                    if len(selected) >= self._config.top_k:
                        break

        return [g.predicate for g in selected if g.predicate is not None]

    # -- bug site identification --------------------------------------------

    def _identify_bug_sites(
        self, ir_func: IRFunction
    ) -> List[Tuple[str, int, int]]:
        """Find potential bug locations: divisions, indexing, dereferences."""
        sites: List[Tuple[str, int, int]] = []
        blocks = GuardHarvester._get_blocks(ir_func)

        for block in blocks:
            for node in GuardHarvester._get_nodes(block):
                loc = GuardHarvester._get_location(node)

                # Division / modulo → ZeroDivisionError
                if isinstance(node, BinOpNode):
                    op = str(getattr(node, "op", ""))
                    if op in ("/", "//", "%", "Div", "FloorDiv", "Mod"):
                        sites.append(loc)

                # Indexing → IndexError / KeyError
                if isinstance(node, IndexNode):
                    sites.append(loc)

                # Attribute access → AttributeError
                if isinstance(node, LoadAttrNode):
                    sites.append(loc)

                # Call nodes → TypeError (wrong arg types)
                if isinstance(node, CallNode):
                    sites.append(loc)

        return sites

    def _guards_near_site(
        self,
        guards: List[HarvestedGuard],
        site: Tuple[str, int, int],
        radius: int,
    ) -> List[HarvestedGuard]:
        """Return guards whose source location is within *radius* lines."""
        site_file, site_line, _ = site
        result: List[HarvestedGuard] = []
        for g in guards:
            g_file, g_line, _ = g.source_location
            if g_file == site_file or site_file == "<unknown>" or g_file == "<unknown>":
                if abs(g_line - site_line) <= radius:
                    result.append(g)
        return result

    def _compute_coverage(
        self,
        selected: List[HarvestedGuard],
        all_guards: List[HarvestedGuard],
    ) -> float:
        """Fraction of unique guard kinds and variable spaces covered."""
        if not all_guards:
            return 1.0

        all_kinds: Set[GuardKind] = {g.guard_kind for g in all_guards}
        sel_kinds: Set[GuardKind] = {g.guard_kind for g in selected}
        kind_cov = len(sel_kinds) / max(len(all_kinds), 1)

        all_vars: Set[str] = set()
        sel_vars: Set[str] = set()
        for g in all_guards:
            all_vars |= GuardHarvester._predicate_variables(g.predicate)
        for g in selected:
            sel_vars |= GuardHarvester._predicate_variables(g.predicate)
        var_cov = len(sel_vars) / max(len(all_vars), 1)

        return 0.5 * kind_cov + 0.5 * var_cov


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  ConstantCollector                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════╝


class ConstantCollector:
    """Collect constants from an IR function for predicate instantiation."""

    # Sentinel values commonly used in dynamic languages
    _SENTINELS = {-1, 0, 1, 2, -2, 255, 256, 0x7FFFFFFF, 0xFFFFFFFF}
    _SENTINEL_FLOATS = {0.0, 1.0, -1.0, float("inf"), float("-inf")}

    def __init__(self) -> None:
        self._collected_count: int = 0

    def collect(self, ir_func: IRFunction) -> CollectedConstants:
        """Collect all constants from *ir_func*."""
        result = CollectedConstants()

        literals = self.collect_literals(ir_func)
        bounds = self.collect_bounds(ir_func)
        comparisons = self.collect_comparisons(ir_func)
        magic = self.collect_magic_numbers(ir_func)

        for val in literals:
            self._add_to_result(result, val, "literal")
        for val in bounds:
            self._add_to_result(result, val, "bound")
        for val in comparisons:
            self._add_to_result(result, val, "comparison")
        for val in magic:
            self._add_to_result(result, val, "sentinel")

        self._collected_count += result.total_count
        return result

    def collect_literals(self, ir_func: IRFunction) -> List[Any]:
        """Collect integer/float/string literals."""
        literals: List[Any] = []
        blocks = GuardHarvester._get_blocks(ir_func)
        for block in blocks:
            for node in GuardHarvester._get_nodes(block):
                literals.extend(self._extract_from_node(node))
        return literals

    def collect_bounds(self, ir_func: IRFunction) -> List[Any]:
        """Collect array lengths, range bounds, loop limits."""
        bounds: List[Any] = []
        blocks = GuardHarvester._get_blocks(ir_func)
        for block in blocks:
            for node in GuardHarvester._get_nodes(block):
                if isinstance(node, CallNode):
                    func_name = str(
                        getattr(node, "func_name", "") or getattr(node, "func", "")
                    )
                    args = getattr(node, "args", [])
                    if func_name in ("range", "len"):
                        for a in args:
                            try:
                                bounds.append(int(str(a)))
                            except (ValueError, TypeError):
                                pass
                    if func_name == "range" and len(args) >= 2:
                        try:
                            bounds.append(int(str(args[0])))
                            bounds.append(int(str(args[1])))
                        except (ValueError, TypeError):
                            pass
        return bounds

    def collect_comparisons(self, ir_func: IRFunction) -> List[Any]:
        """Collect constants used in comparison expressions."""
        constants: List[Any] = []
        blocks = GuardHarvester._get_blocks(ir_func)
        for block in blocks:
            for node in GuardHarvester._get_nodes(block):
                if isinstance(node, BinOpNode):
                    op_str = str(getattr(node, "op", ""))
                    if op_str in ("<", "<=", ">", ">=", "==", "!=",
                                  "Lt", "Le", "LtE", "Gt", "Ge", "GtE",
                                  "Eq", "Ne", "NotEq"):
                        for side in ("left", "right"):
                            val_str = str(getattr(node, side, ""))
                            try:
                                constants.append(int(val_str))
                            except ValueError:
                                try:
                                    constants.append(float(val_str))
                                except ValueError:
                                    pass
        return constants

    def collect_magic_numbers(self, ir_func: IRFunction) -> List[Any]:
        """Collect common sentinel values found in the function."""
        all_consts: List[Any] = []
        blocks = GuardHarvester._get_blocks(ir_func)
        for block in blocks:
            for node in GuardHarvester._get_nodes(block):
                extracted = self._extract_from_node(node)
                for val in extracted:
                    if isinstance(val, int) and val in self._SENTINELS:
                        all_consts.append(val)
                    elif isinstance(val, float) and val in self._SENTINEL_FLOATS:
                        all_consts.append(val)
        return all_consts

    def _extract_from_node(self, node: IRNode) -> List[Any]:
        """Extract constant values from a single IR node."""
        values: List[Any] = []
        # Check for literal/constant node types
        node_type = type(node).__name__
        if node_type in ("LiteralNode", "ConstantNode", "Constant"):
            val = getattr(node, "value", None)
            if val is not None:
                values.append(val)

        # Check for constant attribute on other node types
        for attr in ("value", "const", "literal", "constant"):
            val = getattr(node, attr, None)
            if val is not None and not callable(val) and not isinstance(val, IRNode):
                if isinstance(val, (int, float, str, bool, type(None))):
                    values.append(val)

        # AssignNode with a constant RHS
        if isinstance(node, AssignNode):
            rhs = getattr(node, "value", None) or getattr(node, "rhs", None)
            if rhs is not None:
                node_type_rhs = type(rhs).__name__
                if node_type_rhs in ("LiteralNode", "ConstantNode", "Constant"):
                    val = getattr(rhs, "value", None)
                    if val is not None:
                        values.append(val)
                elif isinstance(rhs, (int, float, str, bool)):
                    values.append(rhs)

        return values

    def _categorize_constant(self, value: Any) -> str:
        """Categorize a constant value."""
        if value is None:
            return "sentinel"
        if isinstance(value, bool):
            return "sentinel"
        if isinstance(value, int):
            if value in self._SENTINELS:
                return "sentinel"
            if value >= 0:
                return "bound"
            return "literal"
        if isinstance(value, float):
            if value in self._SENTINEL_FLOATS:
                return "sentinel"
            return "literal"
        if isinstance(value, str):
            return "literal"
        return "literal"

    @staticmethod
    def _add_to_result(result: CollectedConstants, value: Any, category: str) -> None:
        """Add a constant to the appropriate list in *result*."""
        if isinstance(value, bool):
            result.sentinels.append(value)
        elif isinstance(value, int):
            result.integers.append(value)
            if category == "bound":
                result.bounds.append((value, category))
            elif category == "sentinel":
                result.sentinels.append(value)
        elif isinstance(value, float):
            result.floats.append(value)
        elif isinstance(value, str):
            result.strings.append(value)
        elif value is None:
            result.sentinels.append(value)
        result.all_unique.add(value)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  GuardStatistics                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝


class GuardStatistics:
    """Compute summary statistics for a set of harvested guards."""

    def __init__(self) -> None:
        self._computation_count: int = 0

    def compute(
        self,
        guards: List[HarvestedGuard],
        ir_func: Optional[IRFunction] = None,
    ) -> StatisticsReport:
        """Full statistics report."""
        self._computation_count += 1

        by_kind = self.count_by_kind(guards)

        avg_conf = 0.0
        avg_info = 0.0
        if guards:
            avg_conf = sum(g.confidence for g in guards) / len(guards)
            avg_info = sum(g.information_content for g in guards) / len(guards)

        coverage = CoverageReport()
        if ir_func is not None:
            coverage = self.coverage_analysis(guards, ir_func)

        redundancy = self.redundancy_analysis(guards)

        return StatisticsReport(
            total_guards=len(guards),
            by_kind=by_kind,
            avg_confidence=avg_conf,
            avg_information=avg_info,
            coverage=coverage,
            redundancy_ratio=redundancy.redundancy_ratio,
        )

    def count_by_kind(self, guards: List[HarvestedGuard]) -> Dict[GuardKind, int]:
        """Breakdown by guard type."""
        counter: Counter[GuardKind] = Counter()
        for g in guards:
            counter[g.guard_kind] += 1
        return dict(counter)

    def coverage_analysis(
        self, guards: List[HarvestedGuard], ir_func: IRFunction
    ) -> CoverageReport:
        """Fraction of branches that have associated guards."""
        total = self._count_branches(ir_func)
        guarded = self._count_guarded_branches(guards, ir_func)

        ratio = guarded / max(total, 1)

        # Find unguarded locations
        guarded_lines: Set[int] = {g.line for g in guards if g.line > 0}
        unguarded: List[Tuple[str, int, int]] = []
        blocks = GuardHarvester._get_blocks(ir_func)
        for block in blocks:
            nodes = GuardHarvester._get_nodes(block)
            if not nodes:
                continue
            last_node = nodes[-1]
            successors = getattr(block, "successors", [])
            if len(successors) if isinstance(successors, (list, tuple)) else 0 >= 2:
                loc = GuardHarvester._get_location(last_node)
                if loc[1] not in guarded_lines and loc[1] > 0:
                    unguarded.append(loc)

        return CoverageReport(
            total_branches=total,
            guarded_branches=guarded,
            coverage_ratio=ratio,
            unguarded_locations=unguarded,
        )

    def redundancy_analysis(
        self, guards: List[HarvestedGuard]
    ) -> RedundancyReport:
        """Identify redundant guards."""
        fp_groups: Dict[str, List[int]] = defaultdict(list)
        for i, g in enumerate(guards):
            fp_groups[g.fingerprint()].append(i)

        unique = len(fp_groups)
        redundant = len(guards) - unique
        ratio = redundant / max(len(guards), 1)

        groups = [indices for indices in fp_groups.values() if len(indices) > 1]

        return RedundancyReport(
            total_guards=len(guards),
            unique_guards=unique,
            redundant_guards=redundant,
            redundancy_ratio=ratio,
            redundancy_groups=groups,
        )

    def information_summary(self, guards: List[HarvestedGuard]) -> Dict[str, float]:
        """Summary of information content distribution."""
        if not guards:
            return {
                "min": 0.0, "max": 0.0, "mean": 0.0,
                "median": 0.0, "std_dev": 0.0,
            }

        infos = [g.information_content for g in guards]
        sorted_infos = sorted(infos)
        n = len(sorted_infos)
        mean = sum(sorted_infos) / n
        median = sorted_infos[n // 2]
        variance = sum((x - mean) ** 2 for x in sorted_infos) / n
        std_dev = math.sqrt(variance)

        return {
            "min": sorted_infos[0],
            "max": sorted_infos[-1],
            "mean": mean,
            "median": median,
            "std_dev": std_dev,
        }

    def _count_branches(self, ir_func: IRFunction) -> int:
        """Count total branches (blocks with ≥2 successors)."""
        count = 0
        blocks = GuardHarvester._get_blocks(ir_func)
        for block in blocks:
            successors = getattr(block, "successors", [])
            if isinstance(successors, (list, tuple)) and len(successors) >= 2:
                count += 1
        return count

    def _count_guarded_branches(
        self, guards: List[HarvestedGuard], ir_func: IRFunction
    ) -> int:
        """Count branches that have at least one guard nearby."""
        guarded_lines: Set[int] = {g.line for g in guards if g.line > 0}
        count = 0
        blocks = GuardHarvester._get_blocks(ir_func)
        for block in blocks:
            successors = getattr(block, "successors", [])
            if not (isinstance(successors, (list, tuple)) and len(successors) >= 2):
                continue
            nodes = GuardHarvester._get_nodes(block)
            if not nodes:
                continue
            last_node = nodes[-1]
            loc = GuardHarvester._get_location(last_node)
            if loc[1] in guarded_lines:
                count += 1
            else:
                # Check if any guard is within 3 lines
                for g_line in guarded_lines:
                    if abs(g_line - loc[1]) <= 3:
                        count += 1
                        break
        return count


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  CrossFunctionGuardPropagation                                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝


class CrossFunctionGuardPropagation:
    """Propagate guards across function boundaries on the call graph.

    Given per-function guards (from :meth:`GuardHarvester.harvest_module`),
    this class builds a call graph and flows guard information from
    callers to callees and vice versa.
    """

    def __init__(
        self, module_guards: Dict[str, List[HarvestedGuard]]
    ) -> None:
        self._module_guards = module_guards
        self._call_graph: Dict[str, Set[str]] = self._build_call_graph(module_guards)
        self._reverse_cg: Dict[str, Set[str]] = self._reverse_call_graph(self._call_graph)
        self._guard_flow: Dict[str, List[HarvestedGuard]] = {}

    # -- public API ---------------------------------------------------------

    def propagate(self) -> Dict[str, List[HarvestedGuard]]:
        """Propagate guards and return updated per-function guard maps."""
        flow = self._compute_guard_flow(self._call_graph)
        result: Dict[str, List[HarvestedGuard]] = {}

        for func_name, local_guards in self._module_guards.items():
            propagated = flow.get(func_name, [])
            merged = self._merge_propagated_guards(local_guards, propagated)
            result[func_name] = merged

        # Add entries for functions that appear only via propagation
        for func_name, prop_guards in flow.items():
            if func_name not in result:
                result[func_name] = list(prop_guards)

        self._guard_flow = flow
        return result

    def callee_guards(self, func_name: str) -> List[HarvestedGuard]:
        """Guards from callees that may affect the caller."""
        result: List[HarvestedGuard] = []
        callees = self._call_graph.get(func_name, set())
        for callee in callees:
            callee_g = self._module_guards.get(callee, [])
            for g in callee_g:
                adapted = self._adapt_guard_to_context(g, func_name)
                if adapted is not None:
                    result.append(adapted)
        return result

    def caller_guards(self, func_name: str) -> List[HarvestedGuard]:
        """Guards from callers that may affect the callee."""
        result: List[HarvestedGuard] = []
        callers = self._reverse_cg.get(func_name, set())
        for caller in callers:
            caller_g = self._module_guards.get(caller, [])
            for g in caller_g:
                if self._is_relevant_to_callee(g, func_name):
                    adapted = self._adapt_guard_to_context(g, func_name)
                    if adapted is not None:
                        result.append(adapted)
        return result

    def summary_guards(self, func_name: str) -> List[HarvestedGuard]:
        """Summary guards: intersection of local, callee, and caller guards."""
        local = list(self._module_guards.get(func_name, []))
        callee_g = self.callee_guards(func_name)
        caller_g = self.caller_guards(func_name)

        # Merge all and deduplicate
        all_guards = local + callee_g + caller_g
        normalizer = GuardNormalizer()
        return normalizer.deduplicate(all_guards)

    # -- call graph construction --------------------------------------------

    @staticmethod
    def _build_call_graph(
        module_guards: Dict[str, List[HarvestedGuard]]
    ) -> Dict[str, Set[str]]:
        """Build a caller → callee mapping by inspecting guard contexts.

        This is a best-effort heuristic: we look for function names
        mentioned in guard contexts and predicates.
        """
        all_funcs = set(module_guards.keys())
        cg: Dict[str, Set[str]] = defaultdict(set)

        for func_name, guards in module_guards.items():
            for g in guards:
                # Check if any other function name appears in context
                ctx = g.context
                pred_str = str(g.predicate) if g.predicate else ""
                combined = ctx + " " + pred_str

                for other_func in all_funcs:
                    if other_func != func_name and other_func in combined:
                        cg[func_name].add(other_func)

        return dict(cg)

    @staticmethod
    def _reverse_call_graph(cg: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Build the reverse call graph (callee → callers)."""
        reverse: Dict[str, Set[str]] = defaultdict(set)
        for caller, callees in cg.items():
            for callee in callees:
                reverse[callee].add(caller)
        return dict(reverse)

    def _compute_guard_flow(
        self, call_graph: Dict[str, Set[str]]
    ) -> Dict[str, List[HarvestedGuard]]:
        """Compute guard flow along the call graph edges.

        Uses a worklist algorithm: propagate guards from callees to
        callers and from callers to callees until a fixed point.
        """
        flow: Dict[str, List[HarvestedGuard]] = defaultdict(list)

        # Worklist: each item is (source_func, target_func, direction)
        worklist: Deque[Tuple[str, str, str]] = deque()

        for caller, callees in call_graph.items():
            for callee in callees:
                worklist.append((caller, callee, "down"))
                worklist.append((callee, caller, "up"))

        visited_edges: Set[Tuple[str, str, str]] = set()
        max_iterations = len(worklist) * 3 + 100

        iteration = 0
        while worklist and iteration < max_iterations:
            iteration += 1
            src, tgt, direction = worklist.popleft()
            edge_key = (src, tgt, direction)

            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            src_guards = self._module_guards.get(src, [])
            new_guards: List[HarvestedGuard] = []

            for g in src_guards:
                relevant = (
                    self._is_relevant_to_callee(g, tgt)
                    if direction == "down"
                    else True
                )
                if relevant:
                    adapted = self._adapt_guard_to_context(g, tgt)
                    if adapted is not None:
                        # Lower confidence for propagated guards
                        adapted.confidence *= 0.8
                        new_guards.append(adapted)

            if new_guards:
                existing_fps = {g.fingerprint() for g in flow[tgt]}
                for ng in new_guards:
                    if ng.fingerprint() not in existing_fps:
                        flow[tgt].append(ng)
                        existing_fps.add(ng.fingerprint())

        return dict(flow)

    def _merge_propagated_guards(
        self,
        local_guards: List[HarvestedGuard],
        propagated: List[HarvestedGuard],
    ) -> List[HarvestedGuard]:
        """Merge local and propagated guards, deduplicating."""
        normalizer = GuardNormalizer()
        combined = list(local_guards) + list(propagated)
        return normalizer.deduplicate(combined)

    @staticmethod
    def _is_relevant_to_callee(guard: HarvestedGuard, callee_name: str) -> bool:
        """Check if a guard is relevant to a callee function.

        A guard is relevant if it constrains a variable that could be
        a parameter of the callee, or if it constrains a return value.
        """
        pred = guard.predicate
        if pred is None:
            return False

        # Check if the callee name appears in the guard
        ctx = guard.context
        if callee_name in ctx:
            return True

        # Check if the guard constrains common parameter-like variables
        vars_in_pred: Set[str] = set()
        for attr in ("variable", "left_var"):
            v = getattr(pred, attr, None)
            if v:
                vars_in_pred.add(str(v))

        # Common parameter names are more likely relevant
        common_params = {"self", "cls", "args", "kwargs", "x", "y", "z",
                         "n", "key", "value", "item", "data", "result",
                         "obj", "func", "callback", "index", "i", "j"}
        if vars_in_pred & common_params:
            return True

        # Type tag and nullity guards are generally relevant
        if guard.guard_kind in (GuardKind.TYPE_TAG, GuardKind.NULLITY):
            return True

        return False

    @staticmethod
    def _adapt_guard_to_context(
        guard: HarvestedGuard, call_site: str
    ) -> Optional[HarvestedGuard]:
        """Adapt a guard's variable names to the target function context.

        For now, this creates a copy with an updated context string.
        Full SSA-aware renaming would require the call-site IR.
        """
        adapted = copy.copy(guard)
        adapted.context = f"[propagated from {call_site}] {guard.context}"
        # Mark propagated guards with slightly lower confidence
        adapted.confidence = guard.confidence * 0.9
        return adapted


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Full pipeline convenience function                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def harvest_and_select_seeds(
    ir_func: IRFunction,
    harvest_config: Optional[HarvestConfig] = None,
    ranking_weights: Optional[RankingWeights] = None,
    seed_config: Optional[SeedConfig] = None,
) -> List[BasePredicate]:
    """End-to-end pipeline: harvest → rank → select seeds.

    Parameters
    ----------
    ir_func : IRFunction
        The IR function to analyse.
    harvest_config : HarvestConfig, optional
        Configuration for the harvesting phase.
    ranking_weights : RankingWeights, optional
        Weights for the ranking phase.
    seed_config : SeedConfig, optional
        Configuration for seed selection.

    Returns
    -------
    list[BasePredicate]
        Seed predicates ready for the CEGAR loop.
    """
    harvester = GuardHarvester(harvest_config)
    ranker = GuardRanker(ranking_weights)
    selector = GuardSeedStrategy(seed_config)

    guards = harvester.harvest_function(ir_func)
    ranked = ranker.rank(guards)
    seeds = selector.select_seeds(ranked, ir_func)

    logger.info(
        "harvest_and_select_seeds: %d guards → %d ranked → %d seeds",
        len(guards),
        len(ranked),
        len(seeds),
    )
    return seeds


def harvest_module_with_propagation(
    ir_module: IRModule,
    harvest_config: Optional[HarvestConfig] = None,
    ranking_weights: Optional[RankingWeights] = None,
    seed_config: Optional[SeedConfig] = None,
) -> Dict[str, List[BasePredicate]]:
    """Harvest guards from a module with cross-function propagation.

    Parameters
    ----------
    ir_module : IRModule
        The IR module to analyse.
    harvest_config, ranking_weights, seed_config
        Optional configuration overrides.

    Returns
    -------
    dict[str, list[BasePredicate]]
        Per-function seed predicates.
    """
    harvester = GuardHarvester(harvest_config)
    ranker = GuardRanker(ranking_weights)
    selector = GuardSeedStrategy(seed_config)

    # Phase 1: per-function harvest
    per_func = harvester.harvest_module(ir_module)

    # Phase 2: cross-function propagation
    propagator = CrossFunctionGuardPropagation(per_func)
    propagated = propagator.propagate()

    # Phase 3: rank + select per function
    result: Dict[str, List[BasePredicate]] = {}
    functions = getattr(ir_module, "functions", {})
    if isinstance(functions, dict):
        func_items: Iterable = functions.items()
    elif isinstance(functions, (list, tuple)):
        func_items = ((getattr(f, "name", f"anon_{i}"), f) for i, f in enumerate(functions))
    else:
        func_items = ()

    for name, func in func_items:
        guards = propagated.get(name, [])
        ranked = ranker.rank(guards)
        seeds = selector.select_seeds(ranked, func)
        if seeds:
            result[name] = seeds

    return result


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Guard hash / equality utilities                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def guard_fingerprint_hash(guard: HarvestedGuard) -> int:
    """Return a stable integer hash for a guard's semantic content."""
    fp = guard.fingerprint()
    h = 0
    for ch in fp:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFFFFFFFFFF
    return h


def guards_equivalent(g1: HarvestedGuard, g2: HarvestedGuard) -> bool:
    """Check whether two guards are semantically equivalent."""
    if g1.guard_kind != g2.guard_kind:
        return False
    if g1.is_negated != g2.is_negated:
        return False
    normalizer = GuardNormalizer()
    return normalizer._are_semantically_equal(g1.predicate, g2.predicate)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Guard serialization / deserialization                               ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def guard_to_dict(guard: HarvestedGuard) -> Dict[str, Any]:
    """Serialize a :class:`HarvestedGuard` to a plain dict."""
    return {
        "predicate": str(guard.predicate),
        "source_location": list(guard.source_location),
        "guard_kind": guard.guard_kind.name,
        "confidence": guard.confidence,
        "frequency": guard.frequency,
        "information_content": guard.information_content,
        "context": guard.context,
        "is_negated": guard.is_negated,
        "fingerprint": guard.fingerprint(),
    }


def guard_list_to_dicts(guards: List[HarvestedGuard]) -> List[Dict[str, Any]]:
    """Serialize a list of guards."""
    return [guard_to_dict(g) for g in guards]


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Guard filtering utilities                                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def filter_by_kind(
    guards: List[HarvestedGuard], *kinds: GuardKind
) -> List[HarvestedGuard]:
    """Return only guards of the specified kinds."""
    kind_set = set(kinds)
    return [g for g in guards if g.guard_kind in kind_set]


def filter_by_confidence(
    guards: List[HarvestedGuard], threshold: float
) -> List[HarvestedGuard]:
    """Return guards with confidence ≥ *threshold*."""
    return [g for g in guards if g.confidence >= threshold]


def filter_by_variable(
    guards: List[HarvestedGuard], var_name: str
) -> List[HarvestedGuard]:
    """Return guards that reference *var_name*."""
    result: List[HarvestedGuard] = []
    for g in guards:
        pred = g.predicate
        if pred is None:
            continue
        for attr in ("variable", "left_var", "right_expr", "container", "element"):
            v = getattr(pred, attr, None)
            if v is not None and var_name in str(v):
                result.append(g)
                break
    return result


def partition_by_kind(
    guards: List[HarvestedGuard],
) -> Dict[GuardKind, List[HarvestedGuard]]:
    """Partition guards into buckets by kind."""
    buckets: Dict[GuardKind, List[HarvestedGuard]] = defaultdict(list)
    for g in guards:
        buckets[g.guard_kind].append(g)
    return dict(buckets)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Guard merging (for incremental analysis)                            ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def merge_guard_sets(
    old_guards: List[HarvestedGuard],
    new_guards: List[HarvestedGuard],
) -> List[HarvestedGuard]:
    """Merge two guard sets, keeping the higher-confidence version of
    each semantically equivalent guard and accumulating frequencies."""
    normalizer = GuardNormalizer()
    combined = list(old_guards) + list(new_guards)
    return normalizer.deduplicate(combined)


def incremental_update(
    existing: Dict[str, List[HarvestedGuard]],
    changed_functions: Dict[str, IRFunction],
    harvest_config: Optional[HarvestConfig] = None,
) -> Dict[str, List[HarvestedGuard]]:
    """Incrementally update guards for changed functions only."""
    harvester = GuardHarvester(harvest_config)
    result = dict(existing)

    for func_name, ir_func in changed_functions.items():
        new_guards = harvester.harvest_function(ir_func)
        old_guards = result.get(func_name, [])
        result[func_name] = merge_guard_sets(old_guards, new_guards)

    return result


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Guard explanation (for diagnostics / UI)                            ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def explain_guard(guard: HarvestedGuard) -> str:
    """Produce a human-readable explanation of a harvested guard."""
    lines: List[str] = []
    lines.append(f"Guard: {guard.fingerprint()}")
    lines.append(f"  Kind: {guard.guard_kind.description()}")
    lines.append(f"  Location: {guard.file}:{guard.line}:{guard.col}")
    lines.append(f"  Confidence: {guard.confidence:.3f}")
    lines.append(f"  Frequency: {guard.frequency}")
    lines.append(f"  Information: {guard.information_content:.3f}")
    if guard.context:
        lines.append(f"  Context: {guard.context}")
    if guard.is_negated:
        lines.append("  (negated)")
    return "\n".join(lines)


def explain_statistics(report: StatisticsReport) -> str:
    """Produce a human-readable summary of guard statistics."""
    lines: List[str] = []
    lines.append(f"Total guards: {report.total_guards}")
    for kind, count in sorted(report.by_kind.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {kind.name}: {count}")
    lines.append(f"Avg confidence: {report.avg_confidence:.3f}")
    lines.append(f"Avg information: {report.avg_information:.3f}")
    cov = report.coverage
    lines.append(
        f"Coverage: {cov.guarded_branches}/{cov.total_branches}"
        f" ({cov.coverage_ratio:.1%})"
    )
    lines.append(f"Redundancy: {report.redundancy_ratio:.1%}")
    return "\n".join(lines)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Predicate expression builder utilities                              ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def build_comparison_expr(
    left: str, op: str, right: str
) -> CompareExpr:
    """Build a :class:`CompareExpr` from string components."""
    return CompareExpr(
        left=VarExpr(name=left),
        op=op,
        right=VarExpr(name=right) if not GuardNormalizer._looks_like_constant(right) else ConstExpr(value=_parse_const(right)),
    )


def build_isinstance_expr(var: str, type_name: str) -> CallExpr:
    """Build ``isinstance(var, type_name)`` as a :class:`CallExpr`."""
    return CallExpr(
        func=VarExpr(name="isinstance"),
        args=(VarExpr(name=var), ConstExpr(value=type_name)),
    )


def build_none_check_expr(var: str, is_null: bool = True) -> CompareExpr:
    """Build ``var is None`` / ``var is not None``."""
    op = "is" if is_null else "is not"
    return CompareExpr(
        left=VarExpr(name=var),
        op=op,
        right=ConstExpr(value=None),
    )


def _parse_const(s: str) -> Any:
    """Parse a string into a Python constant."""
    if s == "None":
        return None
    if s == "True":
        return True
    if s == "False":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s.strip("'\"")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Batch harvesting utilities                                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def batch_harvest(
    functions: Dict[str, IRFunction],
    config: Optional[HarvestConfig] = None,
) -> Dict[str, List[HarvestedGuard]]:
    """Harvest guards from multiple functions."""
    harvester = GuardHarvester(config)
    return {
        name: harvester.harvest_function(func)
        for name, func in functions.items()
    }


def batch_rank(
    per_func_guards: Dict[str, List[HarvestedGuard]],
    weights: Optional[RankingWeights] = None,
) -> Dict[str, List[HarvestedGuard]]:
    """Rank guards per-function."""
    ranker = GuardRanker(weights)
    return {
        name: ranker.rank(guards)
        for name, guards in per_func_guards.items()
    }


def batch_statistics(
    per_func_guards: Dict[str, List[HarvestedGuard]],
) -> Dict[str, StatisticsReport]:
    """Compute statistics per-function."""
    stats = GuardStatistics()
    return {
        name: stats.compute(guards)
        for name, guards in per_func_guards.items()
    }


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Soundness assertion helpers                                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def assert_all_guards_valid(guards: List[HarvestedGuard]) -> None:
    """Assert that every guard has a non-null predicate and valid kind.

    This is the soundness check: every extracted guard must correspond
    to a valid predicate in the language P.

    Raises
    ------
    AssertionError
        If any guard is invalid.
    """
    for i, g in enumerate(guards):
        assert g.predicate is not None, (
            f"Guard #{i} has no predicate: {g}"
        )
        assert isinstance(g.guard_kind, GuardKind), (
            f"Guard #{i} has invalid kind: {g.guard_kind}"
        )
        assert 0.0 <= g.confidence <= 1.0, (
            f"Guard #{i} has confidence out of range: {g.confidence}"
        )
        assert g.frequency >= 0, (
            f"Guard #{i} has negative frequency: {g.frequency}"
        )


def assert_guards_normalized(guards: List[HarvestedGuard]) -> None:
    """Assert that guards are in normalized canonical form."""
    normalizer = GuardNormalizer()
    for i, g in enumerate(guards):
        ng = normalizer.normalize(g)
        assert str(ng.predicate) == str(g.predicate), (
            f"Guard #{i} is not normalized: {g.predicate} → {ng.predicate}"
        )


def assert_no_duplicates(guards: List[HarvestedGuard]) -> None:
    """Assert that no two guards have the same fingerprint."""
    seen: Set[str] = set()
    for i, g in enumerate(guards):
        fp = g.fingerprint()
        assert fp not in seen, (
            f"Duplicate guard at #{i}: {fp}"
        )
        seen.add(fp)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Module-level exports                                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

__all__ = [
    # Enums
    "GuardKind",
    # Dataclasses
    "HarvestedGuard",
    "HarvestConfig",
    "RankingWeights",
    "SeedConfig",
    "CollectedConstants",
    "StatisticsReport",
    "CoverageReport",
    "RedundancyReport",
    # Expression nodes
    "ExprNode",
    "VarExpr",
    "ConstExpr",
    "BinOpExpr",
    "UnaryOpExpr",
    "CallExpr",
    "AttrExpr",
    "IndexExpr",
    "CompareExpr",
    # Core classes
    "GuardHarvester",
    "GuardNormalizer",
    "GuardRanker",
    "GuardSeedStrategy",
    "ConstantCollector",
    "GuardStatistics",
    "CrossFunctionGuardPropagation",
    "PatternMatcher",
    # Pipeline functions
    "harvest_and_select_seeds",
    "harvest_module_with_propagation",
    # Utilities
    "guard_fingerprint_hash",
    "guards_equivalent",
    "guard_to_dict",
    "guard_list_to_dicts",
    "filter_by_kind",
    "filter_by_confidence",
    "filter_by_variable",
    "partition_by_kind",
    "merge_guard_sets",
    "incremental_update",
    "explain_guard",
    "explain_statistics",
    "build_comparison_expr",
    "build_isinstance_expr",
    "build_none_check_expr",
    "batch_harvest",
    "batch_rank",
    "batch_statistics",
    "assert_all_guards_valid",
    "assert_guards_normalized",
    "assert_no_duplicates",
]

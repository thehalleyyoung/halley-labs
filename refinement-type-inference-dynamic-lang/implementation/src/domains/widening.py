from __future__ import annotations

import math
import time
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
# Constants
# ---------------------------------------------------------------------------

MAX_INT: int = 2**63 - 1
MIN_INT: int = -(2**63)
_POS_INF: float = math.inf
_NEG_INF: float = -math.inf

T = TypeVar("T")

# ===================================================================
# Local domain value types (no cross-module imports)
# ===================================================================


@dataclass(frozen=True)
class Interval:
    """Closed integer interval [lo, hi].  Use ±inf for unbounded."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            object.__setattr__(self, "lo", _POS_INF)
            object.__setattr__(self, "hi", _NEG_INF)

    @staticmethod
    def bottom() -> Interval:
        return Interval(_POS_INF, _NEG_INF)

    @staticmethod
    def top() -> Interval:
        return Interval(_NEG_INF, _POS_INF)

    @property
    def is_bottom(self) -> bool:
        return self.lo > self.hi

    @property
    def is_top(self) -> bool:
        return self.lo == _NEG_INF and self.hi == _POS_INF

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    def join(self, other: Interval) -> Interval:
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def meet(self, other: Interval) -> Interval:
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        return Interval(lo, hi)

    def is_subset_of(self, other: Interval) -> bool:
        if self.is_bottom:
            return True
        if other.is_bottom:
            return False
        return other.lo <= self.lo and self.hi <= other.hi

    def width(self) -> float:
        if self.is_bottom:
            return 0.0
        if self.lo == _NEG_INF or self.hi == _POS_INF:
            return _POS_INF
        return self.hi - self.lo

    def __repr__(self) -> str:
        if self.is_bottom:
            return "⊥"
        lo_s = "-∞" if self.lo == _NEG_INF else str(int(self.lo))
        hi_s = "+∞" if self.hi == _POS_INF else str(int(self.hi))
        return f"[{lo_s}, {hi_s}]"


class TypeTag(Enum):
    INT = auto()
    FLOAT = auto()
    COMPLEX = auto()
    BOOL = auto()
    STR = auto()
    BYTES = auto()
    NONE = auto()
    LIST = auto()
    TUPLE = auto()
    SET = auto()
    DICT = auto()
    CALLABLE = auto()
    OBJECT = auto()
    TOP = auto()


@dataclass(frozen=True)
class TypeTagSet:
    """A set of type tags — finite lattice."""

    tags: FrozenSet[TypeTag]

    @staticmethod
    def bottom() -> TypeTagSet:
        return TypeTagSet(frozenset())

    @staticmethod
    def top() -> TypeTagSet:
        return TypeTagSet(frozenset(TypeTag))

    @property
    def is_bottom(self) -> bool:
        return len(self.tags) == 0

    @property
    def is_top(self) -> bool:
        return TypeTag.TOP in self.tags or self.tags == frozenset(TypeTag)

    def join(self, other: TypeTagSet) -> TypeTagSet:
        return TypeTagSet(self.tags | other.tags)

    def meet(self, other: TypeTagSet) -> TypeTagSet:
        return TypeTagSet(self.tags & other.tags)

    def is_subset_of(self, other: TypeTagSet) -> bool:
        return self.tags <= other.tags

    def __repr__(self) -> str:
        if self.is_bottom:
            return "⊥_tag"
        if self.is_top:
            return "⊤_tag"
        return "{" + ", ".join(t.name for t in sorted(self.tags, key=lambda t: t.value)) + "}"


class Nullity(Enum):
    BOTTOM = auto()
    NOT_NULL = auto()
    NULL = auto()
    TOP = auto()

    def join(self, other: Nullity) -> Nullity:
        if self == Nullity.BOTTOM:
            return other
        if other == Nullity.BOTTOM:
            return self
        if self == other:
            return self
        return Nullity.TOP

    def meet(self, other: Nullity) -> Nullity:
        if self == Nullity.TOP:
            return other
        if other == Nullity.TOP:
            return self
        if self == other:
            return self
        return Nullity.BOTTOM

    def is_subset_of(self, other: Nullity) -> bool:
        if self == Nullity.BOTTOM:
            return True
        if other == Nullity.TOP:
            return True
        return self == other


@dataclass(frozen=True)
class StringAbstraction:
    """Abstract string domain: finite set up to threshold, then top."""

    values: Optional[FrozenSet[str]]  # None = top

    @staticmethod
    def bottom() -> StringAbstraction:
        return StringAbstraction(frozenset())

    @staticmethod
    def top() -> StringAbstraction:
        return StringAbstraction(None)

    @staticmethod
    def singleton(s: str) -> StringAbstraction:
        return StringAbstraction(frozenset({s}))

    @property
    def is_bottom(self) -> bool:
        return self.values is not None and len(self.values) == 0

    @property
    def is_top(self) -> bool:
        return self.values is None

    def join(self, other: StringAbstraction, threshold: int = 32) -> StringAbstraction:
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        if self.is_top or other.is_top:
            return StringAbstraction.top()
        assert self.values is not None and other.values is not None
        merged = self.values | other.values
        if len(merged) > threshold:
            return StringAbstraction.top()
        return StringAbstraction(merged)

    def meet(self, other: StringAbstraction) -> StringAbstraction:
        if self.is_top:
            return other
        if other.is_top:
            return self
        if self.is_bottom or other.is_bottom:
            return StringAbstraction.bottom()
        assert self.values is not None and other.values is not None
        return StringAbstraction(self.values & other.values)

    def is_subset_of(self, other: StringAbstraction) -> bool:
        if self.is_bottom:
            return True
        if other.is_top:
            return True
        if self.is_top:
            return False
        assert self.values is not None and other.values is not None
        return self.values <= other.values

    def __repr__(self) -> str:
        if self.is_bottom:
            return "⊥_str"
        if self.is_top:
            return "⊤_str"
        assert self.values is not None
        elems = ", ".join(repr(s) for s in sorted(self.values))
        return "{" + elems + "}"


@dataclass
class OctagonConstraints:
    """Octagonal constraints of the form ±x_i ± x_j ≤ c.

    Stored as a difference-bound matrix (DBM) of dimension 2n × 2n
    for n variables.  Variable x_i maps to rows/cols 2i (for +x_i) and
    2i+1 (for -x_i).
    """

    n_vars: int
    matrix: List[List[float]]

    @staticmethod
    def top(n_vars: int) -> OctagonConstraints:
        dim = 2 * n_vars
        m = [[_POS_INF] * dim for _ in range(dim)]
        for i in range(dim):
            m[i][i] = 0.0
        return OctagonConstraints(n_vars, m)

    @staticmethod
    def bottom(n_vars: int) -> OctagonConstraints:
        dim = 2 * n_vars
        m = [[-1.0] * dim for _ in range(dim)]
        for i in range(dim):
            m[i][i] = 0.0
        return OctagonConstraints(n_vars, m)

    @property
    def dim(self) -> int:
        return 2 * self.n_vars

    @property
    def is_bottom(self) -> bool:
        for i in range(self.dim):
            if self.matrix[i][i] < 0:
                return True
        return False

    def get(self, i: int, j: int) -> float:
        return self.matrix[i][j]

    def set_constraint(self, i: int, j: int, val: float) -> None:
        self.matrix[i][j] = min(self.matrix[i][j], val)

    def close(self) -> OctagonConstraints:
        """Tight closure via Floyd-Warshall + strengthening."""
        dim = self.dim
        m = [row[:] for row in self.matrix]
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    s = m[i][k] + m[k][j]
                    if s < m[i][j]:
                        m[i][j] = s
        for i in range(dim):
            if m[i][i] < 0:
                return OctagonConstraints.bottom(self.n_vars)
        # Strengthening
        for i in range(dim):
            for j in range(dim):
                bar_i = i ^ 1
                bar_j = j ^ 1
                cand = (m[i][bar_i] + m[bar_j][j]) / 2.0
                if cand < m[i][j]:
                    m[i][j] = cand
        return OctagonConstraints(self.n_vars, m)

    def join(self, other: OctagonConstraints) -> OctagonConstraints:
        assert self.n_vars == other.n_vars
        dim = self.dim
        m = [[max(self.matrix[i][j], other.matrix[i][j]) for j in range(dim)] for i in range(dim)]
        return OctagonConstraints(self.n_vars, m)

    def meet(self, other: OctagonConstraints) -> OctagonConstraints:
        assert self.n_vars == other.n_vars
        dim = self.dim
        m = [[min(self.matrix[i][j], other.matrix[i][j]) for j in range(dim)] for i in range(dim)]
        return OctagonConstraints(self.n_vars, m).close()

    def is_subset_of(self, other: OctagonConstraints) -> bool:
        if self.is_bottom:
            return True
        dim = self.dim
        for i in range(dim):
            for j in range(dim):
                if self.matrix[i][j] > other.matrix[i][j] + 1e-9:
                    return False
        return True


@dataclass(frozen=True)
class AbstractValue:
    """Product abstract value combining multiple domains."""

    interval: Interval = field(default_factory=Interval.top)
    type_tag: TypeTagSet = field(default_factory=TypeTagSet.top)
    nullity: Nullity = Nullity.TOP
    string_abs: StringAbstraction = field(default_factory=StringAbstraction.top)

    @staticmethod
    def bottom() -> AbstractValue:
        return AbstractValue(
            interval=Interval.bottom(),
            type_tag=TypeTagSet.bottom(),
            nullity=Nullity.BOTTOM,
            string_abs=StringAbstraction.bottom(),
        )

    @staticmethod
    def top() -> AbstractValue:
        return AbstractValue()

    @property
    def is_bottom(self) -> bool:
        return (
            self.interval.is_bottom
            or self.type_tag.is_bottom
            or self.nullity == Nullity.BOTTOM
        )

    def join(self, other: AbstractValue) -> AbstractValue:
        return AbstractValue(
            interval=self.interval.join(other.interval),
            type_tag=self.type_tag.join(other.type_tag),
            nullity=self.nullity.join(other.nullity),
            string_abs=self.string_abs.join(other.string_abs),
        )

    def meet(self, other: AbstractValue) -> AbstractValue:
        return AbstractValue(
            interval=self.interval.meet(other.interval),
            type_tag=self.type_tag.meet(other.type_tag),
            nullity=self.nullity.meet(other.nullity),
            string_abs=self.string_abs.meet(other.string_abs),
        )

    def is_subset_of(self, other: AbstractValue) -> bool:
        return (
            self.interval.is_subset_of(other.interval)
            and self.type_tag.is_subset_of(other.type_tag)
            and self.nullity.is_subset_of(other.nullity)
            and self.string_abs.is_subset_of(other.string_abs)
        )


# ===================================================================
# Abstract state = mapping from variables to abstract values
# ===================================================================


@dataclass
class AbstractState:
    bindings: Dict[str, AbstractValue] = field(default_factory=dict)
    octagon: Optional[OctagonConstraints] = None
    _var_indices: Dict[str, int] = field(default_factory=dict)

    def copy(self) -> AbstractState:
        return AbstractState(
            bindings=dict(self.bindings),
            octagon=OctagonConstraints(
                self.octagon.n_vars, [row[:] for row in self.octagon.matrix]
            )
            if self.octagon
            else None,
            _var_indices=dict(self._var_indices),
        )

    def get(self, var: str) -> AbstractValue:
        return self.bindings.get(var, AbstractValue.top())

    def set(self, var: str, val: AbstractValue) -> None:
        self.bindings[var] = val

    def join(self, other: AbstractState) -> AbstractState:
        all_vars = set(self.bindings) | set(other.bindings)
        result: Dict[str, AbstractValue] = {}
        for v in all_vars:
            lhs = self.bindings.get(v, AbstractValue.bottom())
            rhs = other.bindings.get(v, AbstractValue.bottom())
            result[v] = lhs.join(rhs)
        oct: Optional[OctagonConstraints] = None
        if self.octagon and other.octagon:
            oct = self.octagon.join(other.octagon)
        return AbstractState(bindings=result, octagon=oct, _var_indices=dict(self._var_indices))

    def meet(self, other: AbstractState) -> AbstractState:
        all_vars = set(self.bindings) | set(other.bindings)
        result: Dict[str, AbstractValue] = {}
        for v in all_vars:
            lhs = self.bindings.get(v, AbstractValue.top())
            rhs = other.bindings.get(v, AbstractValue.top())
            result[v] = lhs.meet(rhs)
        oct: Optional[OctagonConstraints] = None
        if self.octagon and other.octagon:
            oct = self.octagon.meet(other.octagon)
        return AbstractState(bindings=result, octagon=oct, _var_indices=dict(self._var_indices))

    def is_subset_of(self, other: AbstractState) -> bool:
        for v, val in self.bindings.items():
            other_val = other.bindings.get(v, AbstractValue.top())
            if not val.is_subset_of(other_val):
                return False
        if self.octagon and other.octagon:
            if not self.octagon.is_subset_of(other.octagon):
                return False
        return True

    @property
    def is_bottom(self) -> bool:
        return any(v.is_bottom for v in self.bindings.values())


# ===================================================================
# CFG node type for widening point selection
# ===================================================================


@dataclass
class CFGBlock:
    block_id: int
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    is_loop_header: bool = False
    is_function_entry: bool = False
    is_exception_handler: bool = False
    loop_depth: int = 0
    back_edge_sources: List[int] = field(default_factory=list)
    label: str = ""

    def __hash__(self) -> int:
        return hash(self.block_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CFGBlock):
            return NotImplemented
        return self.block_id == other.block_id


@dataclass
class CFG:
    blocks: Dict[int, CFGBlock] = field(default_factory=dict)
    entry_id: int = 0
    exit_id: int = -1

    def add_block(self, block: CFGBlock) -> None:
        self.blocks[block.block_id] = block

    def add_edge(self, src: int, dst: int) -> None:
        self.blocks[src].successors.append(dst)
        self.blocks[dst].predecessors.append(src)

    def get_block(self, block_id: int) -> CFGBlock:
        return self.blocks[block_id]

    def loop_headers(self) -> Set[int]:
        return {bid for bid, b in self.blocks.items() if b.is_loop_header}

    def function_entries(self) -> Set[int]:
        return {bid for bid, b in self.blocks.items() if b.is_function_entry}

    def exception_handlers(self) -> Set[int]:
        return {bid for bid, b in self.blocks.items() if b.is_exception_handler}

    def compute_loop_headers(self) -> None:
        """Identify loop headers via dominance and back edges."""
        dominators = self._compute_dominators()
        for bid, block in self.blocks.items():
            for succ_id in block.successors:
                if succ_id in dominators.get(bid, set()):
                    self.blocks[succ_id].is_loop_header = True
                    self.blocks[succ_id].back_edge_sources.append(bid)

    def _compute_dominators(self) -> Dict[int, Set[int]]:
        all_ids = set(self.blocks.keys())
        dom: Dict[int, Set[int]] = {}
        dom[self.entry_id] = {self.entry_id}
        for bid in all_ids:
            if bid != self.entry_id:
                dom[bid] = set(all_ids)
        changed = True
        while changed:
            changed = False
            for bid in all_ids:
                if bid == self.entry_id:
                    continue
                preds = self.blocks[bid].predecessors
                if not preds:
                    new_dom = {bid}
                else:
                    new_dom = set.intersection(*(dom[p] for p in preds)) | {bid}
                if new_dom != dom[bid]:
                    dom[bid] = new_dom
                    changed = True
        return dom

    def compute_loop_depths(self) -> None:
        """Compute loop nesting depth for each block."""
        self.compute_loop_headers()
        natural_loops: Dict[int, Set[int]] = {}
        for header_id in self.loop_headers():
            body = self._compute_natural_loop(header_id)
            natural_loops[header_id] = body
        for bid in self.blocks:
            depth = sum(1 for body in natural_loops.values() if bid in body)
            self.blocks[bid].loop_depth = depth

    def _compute_natural_loop(self, header_id: int) -> Set[int]:
        body: Set[int] = {header_id}
        worklist: List[int] = []
        for src in self.blocks[header_id].back_edge_sources:
            if src not in body:
                body.add(src)
                worklist.append(src)
        while worklist:
            n = worklist.pop()
            for pred in self.blocks[n].predecessors:
                if pred not in body:
                    body.add(pred)
                    worklist.append(pred)
        return body

    def topological_order(self) -> List[int]:
        visited: Set[int] = set()
        order: List[int] = []

        def dfs(bid: int) -> None:
            if bid in visited:
                return
            visited.add(bid)
            for succ in self.blocks[bid].successors:
                dfs(succ)
            order.append(bid)

        dfs(self.entry_id)
        return list(reversed(order))

    def reverse_topological_order(self) -> List[int]:
        return list(reversed(self.topological_order()))


# ===================================================================
# Expression types for threshold extraction
# ===================================================================


class ExprKind(Enum):
    LITERAL = auto()
    VARIABLE = auto()
    BINARY_OP = auto()
    UNARY_OP = auto()
    CALL = auto()
    ATTRIBUTE = auto()
    SUBSCRIPT = auto()
    COMPARISON = auto()


@dataclass
class Expr:
    kind: ExprKind
    value: Any = None
    children: List[Expr] = field(default_factory=list)
    op: str = ""
    name: str = ""


# ===================================================================
# 1. WideningOperator — abstract base
# ===================================================================


class WideningOperator(ABC):
    """Abstract base for widening operators.

    Guarantees:
      1. old_value ⊑ widen(old_value, new_value)
      2. Any ascending chain old₀, widen(old₀, old₁), widen(…, old₂), …
         eventually stabilizes.
    """

    @abstractmethod
    def widen(self, old_value: AbstractState, new_value: AbstractState) -> AbstractState:
        ...

    @abstractmethod
    def widen_value(self, old_value: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        ...

    def widen_interval(self, old: Interval, new: Interval) -> Interval:
        raise NotImplementedError

    def widen_type_tag(self, old: TypeTagSet, new: TypeTagSet) -> TypeTagSet:
        return old.join(new)

    def widen_nullity(self, old: Nullity, new: Nullity) -> Nullity:
        return old.join(new)

    def widen_string(
        self, old: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        raise NotImplementedError

    def widen_octagon(
        self, old: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        raise NotImplementedError


# ===================================================================
# 2. StandardWidening — drop unstable constraints
# ===================================================================


class StandardWidening(WideningOperator):
    """Standard widening: drop constraints that grew between iterations.

    • Intervals: [a,b] ▽ [c,d] = [min(a,c) if c<a else a, max(b,d) if d>b else b]
      but if a bound grew, jump to ±∞.
    • Type tags: union (always stable — finite domain).
    • Nullity: join (always stable — finite domain).
    • Strings: union up to *string_threshold*, then ⊤.
    • Octagons: drop constraints that increased.
    """

    def __init__(self, string_threshold: int = 32) -> None:
        self.string_threshold = string_threshold

    # -- interval --

    def widen_interval(self, old: Interval, new: Interval) -> Interval:
        if old.is_bottom:
            return new
        if new.is_bottom:
            return old
        lo = _NEG_INF if new.lo < old.lo else old.lo
        hi = _POS_INF if new.hi > old.hi else old.hi
        return Interval(lo, hi)

    # -- string --

    def widen_string(
        self, old: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        if old.is_bottom:
            return new
        if new.is_bottom:
            return old
        if old.is_top or new.is_top:
            return StringAbstraction.top()
        assert old.values is not None and new.values is not None
        merged = old.values | new.values
        if len(merged) > self.string_threshold:
            return StringAbstraction.top()
        return StringAbstraction(merged)

    # -- octagon --

    def widen_octagon(
        self, old: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        assert old.n_vars == new.n_vars
        dim = old.dim
        m = [row[:] for row in old.matrix]
        for i in range(dim):
            for j in range(dim):
                if new.matrix[i][j] > old.matrix[i][j]:
                    m[i][j] = _POS_INF
        return OctagonConstraints(old.n_vars, m)

    # -- product value --

    def widen_value(self, old_value: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        return AbstractValue(
            interval=self.widen_interval(old_value.interval, new_value.interval),
            type_tag=self.widen_type_tag(old_value.type_tag, new_value.type_tag),
            nullity=self.widen_nullity(old_value.nullity, new_value.nullity),
            string_abs=self.widen_string(old_value.string_abs, new_value.string_abs),
        )

    # -- full state --

    def widen(self, old_value: AbstractState, new_value: AbstractState) -> AbstractState:
        all_vars = set(old_value.bindings) | set(new_value.bindings)
        result: Dict[str, AbstractValue] = {}
        for v in all_vars:
            ov = old_value.bindings.get(v, AbstractValue.bottom())
            nv = new_value.bindings.get(v, AbstractValue.bottom())
            result[v] = self.widen_value(ov, nv)
        oct: Optional[OctagonConstraints] = None
        if old_value.octagon and new_value.octagon:
            oct = self.widen_octagon(old_value.octagon, new_value.octagon)
        return AbstractState(bindings=result, octagon=oct)


# ===================================================================
# 3. ThresholdWidening — widening with threshold sets
# ===================================================================


@dataclass
class ThresholdSet:
    """A finite set of thresholds used for widening."""

    values: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.values = sorted(set(self.values))

    def next_above(self, x: float) -> float:
        for v in self.values:
            if v >= x:
                return v
        return _POS_INF

    def next_below(self, x: float) -> float:
        for v in reversed(self.values):
            if v <= x:
                return v
        return _NEG_INF

    def merge(self, other: ThresholdSet) -> ThresholdSet:
        combined = sorted(set(self.values) | set(other.values))
        return ThresholdSet(combined)

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"ThresholdSet({self.values})"


# Default thresholds
DEFAULT_THRESHOLDS = ThresholdSet(
    [-1, 0, 1, 2, 10, 100, 256, 1024, MAX_INT, -MAX_INT]
)


class ThresholdCollectionStrategy(Enum):
    """Strategy for collecting thresholds from a program."""

    COMPARISONS = auto()
    ARRAY_SIZES = auto()
    LOOP_BOUNDS = auto()
    MAGIC_NUMBERS = auto()
    ALL = auto()


def extract_thresholds(
    exprs: List[Expr],
    strategy: ThresholdCollectionStrategy = ThresholdCollectionStrategy.ALL,
) -> ThresholdSet:
    """Extract thresholds from program expressions."""
    values: Set[float] = set()

    def visit(expr: Expr) -> None:
        if expr.kind == ExprKind.LITERAL and isinstance(expr.value, (int, float)):
            v = float(expr.value)
            if strategy in (
                ThresholdCollectionStrategy.MAGIC_NUMBERS,
                ThresholdCollectionStrategy.ALL,
            ):
                values.add(v)
                values.add(v - 1)
                values.add(v + 1)

        if expr.kind == ExprKind.COMPARISON:
            if strategy in (
                ThresholdCollectionStrategy.COMPARISONS,
                ThresholdCollectionStrategy.ALL,
            ):
                for child in expr.children:
                    if child.kind == ExprKind.LITERAL and isinstance(
                        child.value, (int, float)
                    ):
                        v = float(child.value)
                        values.add(v)
                        values.add(v - 1)
                        values.add(v + 1)

        if expr.kind == ExprKind.CALL and expr.name in ("range", "len"):
            if strategy in (
                ThresholdCollectionStrategy.LOOP_BOUNDS,
                ThresholdCollectionStrategy.ARRAY_SIZES,
                ThresholdCollectionStrategy.ALL,
            ):
                for child in expr.children:
                    if child.kind == ExprKind.LITERAL and isinstance(
                        child.value, (int, float)
                    ):
                        values.add(float(child.value))

        for child in expr.children:
            visit(child)

    for e in exprs:
        visit(e)

    return ThresholdSet(sorted(values))


class ThresholdWidening(WideningOperator):
    """Widening that uses threshold sets instead of jumping to ±∞.

    For intervals: widen to next threshold instead of ∞.
    """

    def __init__(
        self,
        thresholds: Optional[ThresholdSet] = None,
        string_threshold: int = 32,
    ) -> None:
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.string_threshold = string_threshold

    def widen_interval(self, old: Interval, new: Interval) -> Interval:
        if old.is_bottom:
            return new
        if new.is_bottom:
            return old
        if new.lo < old.lo:
            lo = self.thresholds.next_below(new.lo)
        else:
            lo = old.lo
        if new.hi > old.hi:
            hi = self.thresholds.next_above(new.hi)
        else:
            hi = old.hi
        return Interval(lo, hi)

    def widen_string(
        self, old: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        if old.is_bottom:
            return new
        if new.is_bottom:
            return old
        if old.is_top or new.is_top:
            return StringAbstraction.top()
        assert old.values is not None and new.values is not None
        merged = old.values | new.values
        if len(merged) > self.string_threshold:
            return StringAbstraction.top()
        return StringAbstraction(merged)

    def widen_octagon(
        self, old: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        assert old.n_vars == new.n_vars
        dim = old.dim
        m = [row[:] for row in old.matrix]
        for i in range(dim):
            for j in range(dim):
                if new.matrix[i][j] > old.matrix[i][j]:
                    m[i][j] = self.thresholds.next_above(new.matrix[i][j])
        return OctagonConstraints(old.n_vars, m)

    def widen_value(self, old_value: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        return AbstractValue(
            interval=self.widen_interval(old_value.interval, new_value.interval),
            type_tag=self.widen_type_tag(old_value.type_tag, new_value.type_tag),
            nullity=self.widen_nullity(old_value.nullity, new_value.nullity),
            string_abs=self.widen_string(old_value.string_abs, new_value.string_abs),
        )

    def widen(self, old_value: AbstractState, new_value: AbstractState) -> AbstractState:
        all_vars = set(old_value.bindings) | set(new_value.bindings)
        result: Dict[str, AbstractValue] = {}
        for v in all_vars:
            ov = old_value.bindings.get(v, AbstractValue.bottom())
            nv = new_value.bindings.get(v, AbstractValue.bottom())
            result[v] = self.widen_value(ov, nv)
        oct: Optional[OctagonConstraints] = None
        if old_value.octagon and new_value.octagon:
            oct = self.widen_octagon(old_value.octagon, new_value.octagon)
        return AbstractState(bindings=result, octagon=oct)


# ===================================================================
# 4. DelayedWidening — delay widening for k iterations
# ===================================================================


class DelayedWidening(WideningOperator):
    """Delay widening for *k* iterations, using join instead.

    After k iterations at a given program point, switches to the
    underlying widening operator.
    """

    def __init__(
        self,
        inner: WideningOperator,
        default_k: int = 3,
        per_point_k: Optional[Dict[int, int]] = None,
        adaptive: bool = False,
    ) -> None:
        self.inner = inner
        self.default_k = default_k
        self.per_point_k: Dict[int, int] = per_point_k or {}
        self.adaptive = adaptive
        self._iteration_count: Dict[int, int] = {}
        self._prev_values: Dict[int, AbstractState] = {}
        self._convergence_rate: Dict[int, List[float]] = {}

    def get_k(self, point: int) -> int:
        return self.per_point_k.get(point, self.default_k)

    def reset(self, point: Optional[int] = None) -> None:
        if point is not None:
            self._iteration_count.pop(point, None)
            self._prev_values.pop(point, None)
            self._convergence_rate.pop(point, None)
        else:
            self._iteration_count.clear()
            self._prev_values.clear()
            self._convergence_rate.clear()

    def _estimate_convergence_rate(self, point: int, old: AbstractState, new: AbstractState) -> float:
        """Estimate convergence rate as fraction of stable variables."""
        all_vars = set(old.bindings) | set(new.bindings)
        if not all_vars:
            return 1.0
        stable = 0
        for v in all_vars:
            ov = old.bindings.get(v, AbstractValue.bottom())
            nv = new.bindings.get(v, AbstractValue.bottom())
            if nv.is_subset_of(ov):
                stable += 1
        return stable / len(all_vars)

    def _adaptive_k(self, point: int) -> int:
        """Adapt k based on convergence rate history."""
        rates = self._convergence_rate.get(point, [])
        if not rates:
            return self.get_k(point)
        avg_rate = sum(rates) / len(rates)
        if avg_rate > 0.8:
            return max(1, self.get_k(point) - 1)
        elif avg_rate < 0.3:
            return self.get_k(point) + 2
        return self.get_k(point)

    def widen_at_point(
        self, point: int, old_value: AbstractState, new_value: AbstractState
    ) -> AbstractState:
        count = self._iteration_count.get(point, 0)
        self._iteration_count[point] = count + 1

        if self.adaptive:
            rate = self._estimate_convergence_rate(point, old_value, new_value)
            self._convergence_rate.setdefault(point, []).append(rate)
            k = self._adaptive_k(point)
        else:
            k = self.get_k(point)

        self._prev_values[point] = new_value

        if count < k:
            return old_value.join(new_value)
        return self.inner.widen(old_value, new_value)

    # Stateless interface delegates to inner
    def widen(self, old_value: AbstractState, new_value: AbstractState) -> AbstractState:
        return self.inner.widen(old_value, new_value)

    def widen_value(self, old_value: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        return self.inner.widen_value(old_value, new_value)

    def widen_interval(self, old: Interval, new: Interval) -> Interval:
        return self.inner.widen_interval(old, new)

    def widen_string(
        self, old: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        return self.inner.widen_string(old, new)

    def widen_octagon(
        self, old: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        return self.inner.widen_octagon(old, new)


# ===================================================================
# 5. GuidedWidening — widening guided by property to verify
# ===================================================================


@dataclass
class PropertySpec:
    """A property specification for guided widening."""

    critical_variables: Set[str] = field(default_factory=set)
    relevant_intervals: Dict[str, Interval] = field(default_factory=dict)
    description: str = ""

    def is_critical(self, var: str) -> bool:
        return var in self.critical_variables

    def get_relevant_interval(self, var: str) -> Optional[Interval]:
        return self.relevant_intervals.get(var)


class GuidedWidening(WideningOperator):
    """Widening guided by the property to verify.

    Less aggressive on critical variables (array indices, divisors, etc.)
    by using threshold widening or delayed widening for those, while
    using standard widening for the rest.
    """

    def __init__(
        self,
        property_spec: PropertySpec,
        aggressive: Optional[WideningOperator] = None,
        conservative: Optional[WideningOperator] = None,
    ) -> None:
        self.property_spec = property_spec
        self.aggressive = aggressive or StandardWidening()
        self.conservative = conservative or ThresholdWidening()

    def widen_interval(self, old: Interval, new: Interval) -> Interval:
        return self.aggressive.widen_interval(old, new)

    def widen_string(
        self, old: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        return self.aggressive.widen_string(old, new)

    def widen_octagon(
        self, old: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        return self.aggressive.widen_octagon(old, new)

    def widen_value_for_var(
        self, var: str, old_value: AbstractValue, new_value: AbstractValue
    ) -> AbstractValue:
        if self.property_spec.is_critical(var):
            return self.conservative.widen_value(old_value, new_value)
        return self.aggressive.widen_value(old_value, new_value)

    def widen_value(self, old_value: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        return self.aggressive.widen_value(old_value, new_value)

    def widen(self, old_value: AbstractState, new_value: AbstractState) -> AbstractState:
        all_vars = set(old_value.bindings) | set(new_value.bindings)
        result: Dict[str, AbstractValue] = {}
        for v in all_vars:
            ov = old_value.bindings.get(v, AbstractValue.bottom())
            nv = new_value.bindings.get(v, AbstractValue.bottom())
            result[v] = self.widen_value_for_var(v, ov, nv)
        oct: Optional[OctagonConstraints] = None
        if old_value.octagon and new_value.octagon:
            oct = self.widen_octagon(old_value.octagon, new_value.octagon)
        return AbstractState(bindings=result, octagon=oct)


# ===================================================================
# 6. JumpSetWidening — widening using jump set
# ===================================================================


@dataclass
class ValueSequence:
    """Records a sequence of values seen at a program point."""

    values: List[float] = field(default_factory=list)
    max_length: int = 16

    def append(self, v: float) -> None:
        self.values.append(v)
        if len(self.values) > self.max_length:
            self.values = self.values[-self.max_length :]

    def detect_linear_growth(self) -> Optional[Tuple[float, float]]:
        """Detect pattern a*n + b.  Returns (a, b) or None."""
        if len(self.values) < 3:
            return None
        diffs = [self.values[i + 1] - self.values[i] for i in range(len(self.values) - 1)]
        if all(abs(d - diffs[0]) < 1e-9 for d in diffs):
            a = diffs[0]
            b = self.values[0]
            return (a, b)
        return None

    def detect_oscillation(self) -> Optional[List[float]]:
        """Detect oscillation between a set of values."""
        if len(self.values) < 4:
            return None
        unique = list(set(self.values))
        if len(unique) <= 3 and len(self.values) >= 2 * len(unique):
            return unique
        return None

    def extrapolate_next(self) -> Optional[float]:
        linear = self.detect_linear_growth()
        if linear is not None:
            a, b = linear
            n = len(self.values)
            return a * n + b
        return None


class JumpSetWidening(WideningOperator):
    """Widening using jump set: extrapolate from value sequence."""

    def __init__(self, base: Optional[WideningOperator] = None) -> None:
        self.base = base or StandardWidening()
        self._lo_sequences: Dict[str, ValueSequence] = {}
        self._hi_sequences: Dict[str, ValueSequence] = {}

    def record_value(self, var: str, interval: Interval) -> None:
        if not interval.is_bottom:
            self._lo_sequences.setdefault(var, ValueSequence()).append(interval.lo)
            self._hi_sequences.setdefault(var, ValueSequence()).append(interval.hi)

    def widen_interval_for_var(self, var: str, old: Interval, new: Interval) -> Interval:
        if old.is_bottom:
            self.record_value(var, new)
            return new
        if new.is_bottom:
            return old

        self.record_value(var, new)

        lo_seq = self._lo_sequences.get(var)
        hi_seq = self._hi_sequences.get(var)

        lo: float
        hi: float

        # Lower bound
        if new.lo < old.lo:
            if lo_seq is not None:
                osc = lo_seq.detect_oscillation()
                if osc is not None:
                    lo = min(osc)
                else:
                    extrap = lo_seq.extrapolate_next()
                    if extrap is not None and extrap <= new.lo:
                        lo = extrap
                    else:
                        lo = _NEG_INF
            else:
                lo = _NEG_INF
        else:
            lo = old.lo

        # Upper bound
        if new.hi > old.hi:
            if hi_seq is not None:
                osc = hi_seq.detect_oscillation()
                if osc is not None:
                    hi = max(osc)
                else:
                    extrap = hi_seq.extrapolate_next()
                    if extrap is not None and extrap >= new.hi:
                        hi = extrap
                    else:
                        hi = _POS_INF
            else:
                hi = _POS_INF
        else:
            hi = old.hi

        return Interval(lo, hi)

    def widen_interval(self, old: Interval, new: Interval) -> Interval:
        return self.base.widen_interval(old, new)

    def widen_string(
        self, old: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        return self.base.widen_string(old, new)

    def widen_octagon(
        self, old: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        return self.base.widen_octagon(old, new)

    def widen_value(self, old_value: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        return self.base.widen_value(old_value, new_value)

    def widen_value_for_var(
        self, var: str, old_value: AbstractValue, new_value: AbstractValue
    ) -> AbstractValue:
        return AbstractValue(
            interval=self.widen_interval_for_var(var, old_value.interval, new_value.interval),
            type_tag=self.widen_type_tag(old_value.type_tag, new_value.type_tag),
            nullity=self.widen_nullity(old_value.nullity, new_value.nullity),
            string_abs=self.base.widen_string(old_value.string_abs, new_value.string_abs),
        )

    def widen(self, old_value: AbstractState, new_value: AbstractState) -> AbstractState:
        all_vars = set(old_value.bindings) | set(new_value.bindings)
        result: Dict[str, AbstractValue] = {}
        for v in all_vars:
            ov = old_value.bindings.get(v, AbstractValue.bottom())
            nv = new_value.bindings.get(v, AbstractValue.bottom())
            result[v] = self.widen_value_for_var(v, ov, nv)
        oct: Optional[OctagonConstraints] = None
        if old_value.octagon and new_value.octagon:
            oct = self.widen_octagon(old_value.octagon, new_value.octagon)
        return AbstractState(bindings=result, octagon=oct)


# ===================================================================
# 7. NarrowingOperator — abstract base
# ===================================================================


class NarrowingOperator(ABC):
    """Abstract base for narrowing operators.

    Guarantees:
      new_value ⊑ narrow(widened, new_value) ⊑ widened
    """

    @abstractmethod
    def narrow(self, widened: AbstractState, new_value: AbstractState) -> AbstractState:
        ...

    @abstractmethod
    def narrow_value(self, widened: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        ...

    def narrow_interval(self, widened: Interval, new: Interval) -> Interval:
        raise NotImplementedError

    def narrow_type_tag(self, widened: TypeTagSet, new: TypeTagSet) -> TypeTagSet:
        return widened.meet(new) if not new.is_bottom else widened

    def narrow_nullity(self, widened: Nullity, new: Nullity) -> Nullity:
        if new == Nullity.BOTTOM:
            return widened
        return widened.meet(new)

    def narrow_string(
        self, widened: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        raise NotImplementedError

    def narrow_octagon(
        self, widened: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        raise NotImplementedError


# ===================================================================
# 8. StandardNarrowing — recover precision after widening
# ===================================================================


class StandardNarrowing(NarrowingOperator):
    """Standard narrowing: recover precision lost during widening.

    • Intervals: [a,b] △ [c,d] = [c if a=-∞ else a, d if b=+∞ else b]
    • Octagons: tighten ∞ constraints.
    """

    def narrow_interval(self, widened: Interval, new: Interval) -> Interval:
        if widened.is_bottom or new.is_bottom:
            return Interval.bottom()
        lo = new.lo if widened.lo == _NEG_INF else widened.lo
        hi = new.hi if widened.hi == _POS_INF else widened.hi
        return Interval(lo, hi)

    def narrow_string(
        self, widened: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        if widened.is_top and not new.is_top:
            return new
        return widened

    def narrow_octagon(
        self, widened: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        assert widened.n_vars == new.n_vars
        dim = widened.dim
        m = [row[:] for row in widened.matrix]
        for i in range(dim):
            for j in range(dim):
                if widened.matrix[i][j] == _POS_INF:
                    m[i][j] = new.matrix[i][j]
        return OctagonConstraints(widened.n_vars, m)

    def narrow_value(self, widened: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        return AbstractValue(
            interval=self.narrow_interval(widened.interval, new_value.interval),
            type_tag=self.narrow_type_tag(widened.type_tag, new_value.type_tag),
            nullity=self.narrow_nullity(widened.nullity, new_value.nullity),
            string_abs=self.narrow_string(widened.string_abs, new_value.string_abs),
        )

    def narrow(self, widened: AbstractState, new_value: AbstractState) -> AbstractState:
        all_vars = set(widened.bindings) | set(new_value.bindings)
        result: Dict[str, AbstractValue] = {}
        for v in all_vars:
            wv = widened.bindings.get(v, AbstractValue.top())
            nv = new_value.bindings.get(v, AbstractValue.top())
            result[v] = self.narrow_value(wv, nv)
        oct: Optional[OctagonConstraints] = None
        if widened.octagon and new_value.octagon:
            oct = self.narrow_octagon(widened.octagon, new_value.octagon)
        return AbstractState(bindings=result, octagon=oct)


# ===================================================================
# 9. IterativeNarrowing — multiple narrowing iterations
# ===================================================================


class IterativeNarrowing:
    """Run narrowing iteratively after widening fixpoint to recover precision."""

    def __init__(
        self,
        narrowing_op: NarrowingOperator,
        max_iterations: int = 10,
        progress_threshold: float = 1e-9,
    ) -> None:
        self.narrowing_op = narrowing_op
        self.max_iterations = max_iterations
        self.progress_threshold = progress_threshold
        self._iterations_done: int = 0

    @property
    def iterations_done(self) -> int:
        return self._iterations_done

    def _estimate_progress(self, old: AbstractState, new: AbstractState) -> float:
        total_delta = 0.0
        count = 0
        for v in old.bindings:
            if v in new.bindings:
                ov = old.bindings[v]
                nv = new.bindings[v]
                old_w = ov.interval.width()
                new_w = nv.interval.width()
                if old_w != _POS_INF and new_w != _POS_INF:
                    total_delta += abs(old_w - new_w)
                    count += 1
                elif old_w == _POS_INF and new_w != _POS_INF:
                    total_delta += 1e6
                    count += 1
        return total_delta / max(count, 1)

    def narrow_fixpoint(
        self,
        widened_fixpoint: AbstractState,
        transfer: Callable[[AbstractState], AbstractState],
    ) -> AbstractState:
        """Iteratively narrow starting from widened fixpoint.

        Args:
            widened_fixpoint: The fixpoint obtained after widening.
            transfer: The abstract transfer function F.

        Returns:
            A narrowed state that is still a post-fixpoint of F.
        """
        current = widened_fixpoint
        self._iterations_done = 0

        for i in range(self.max_iterations):
            new_state = transfer(current)
            narrowed = self.narrowing_op.narrow(current, new_state)

            progress = self._estimate_progress(current, narrowed)
            current = narrowed
            self._iterations_done = i + 1

            if progress < self.progress_threshold:
                break

        return current

    def narrow_at_points(
        self,
        states: Dict[int, AbstractState],
        transfer_at: Callable[[int, AbstractState], AbstractState],
        points: Set[int],
    ) -> Dict[int, AbstractState]:
        """Narrowing at specific program points."""
        current = dict(states)
        self._iterations_done = 0

        for i in range(self.max_iterations):
            made_progress = False
            for pt in points:
                if pt not in current:
                    continue
                old_state = current[pt]
                new_state = transfer_at(pt, old_state)
                narrowed = self.narrowing_op.narrow(old_state, new_state)
                progress = self._estimate_progress(old_state, narrowed)
                if progress >= self.progress_threshold:
                    made_progress = True
                current[pt] = narrowed
            self._iterations_done = i + 1
            if not made_progress:
                break

        return current


# ===================================================================
# 10. WideningPolicy — when and where to apply widening
# ===================================================================


class WideningPolicy(ABC):
    """Determines when and where to apply widening."""

    @abstractmethod
    def should_widen(self, block_id: int, iteration: int, cfg: CFG) -> bool:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class LoopHeadPolicy(WideningPolicy):
    """Widen at loop headers only."""

    def should_widen(self, block_id: int, iteration: int, cfg: CFG) -> bool:
        block = cfg.blocks.get(block_id)
        if block is None:
            return False
        return block.is_loop_header

    def name(self) -> str:
        return "loop_head"


class AllJoinPolicy(WideningPolicy):
    """Widen at all join points (blocks with >1 predecessor)."""

    def should_widen(self, block_id: int, iteration: int, cfg: CFG) -> bool:
        block = cfg.blocks.get(block_id)
        if block is None:
            return False
        return len(block.predecessors) > 1

    def name(self) -> str:
        return "all_join"


class BackEdgePolicy(WideningPolicy):
    """Widen at back-edge targets."""

    def should_widen(self, block_id: int, iteration: int, cfg: CFG) -> bool:
        block = cfg.blocks.get(block_id)
        if block is None:
            return False
        return len(block.back_edge_sources) > 0

    def name(self) -> str:
        return "back_edge"


class NestedLoopPolicy(WideningPolicy):
    """Different strategies for different nesting depths."""

    def __init__(
        self,
        inner_policy: WideningPolicy,
        outer_policy: WideningPolicy,
        depth_threshold: int = 2,
    ) -> None:
        self._inner = inner_policy
        self._outer = outer_policy
        self._depth_threshold = depth_threshold

    def should_widen(self, block_id: int, iteration: int, cfg: CFG) -> bool:
        block = cfg.blocks.get(block_id)
        if block is None:
            return False
        if block.loop_depth >= self._depth_threshold:
            return self._inner.should_widen(block_id, iteration, cfg)
        return self._outer.should_widen(block_id, iteration, cfg)

    def name(self) -> str:
        return f"nested(inner={self._inner.name()},outer={self._outer.name()},depth={self._depth_threshold})"


class CustomPolicy(WideningPolicy):
    """User-defined widening points."""

    def __init__(self, widening_points: Set[int], label: str = "custom") -> None:
        self._points = widening_points
        self._label = label

    def should_widen(self, block_id: int, iteration: int, cfg: CFG) -> bool:
        return block_id in self._points

    def name(self) -> str:
        return self._label


# ===================================================================
# 11. WideningPointSelector — select where to widen
# ===================================================================


class WideningPointSelector:
    """Select program points where widening should be applied."""

    def __init__(
        self,
        include_loop_headers: bool = True,
        include_function_entries: bool = True,
        include_exception_handlers: bool = True,
        include_heuristic: bool = False,
    ) -> None:
        self.include_loop_headers = include_loop_headers
        self.include_function_entries = include_function_entries
        self.include_exception_handlers = include_exception_handlers
        self.include_heuristic = include_heuristic

    def select_from_cfg(self, cfg: CFG) -> Set[int]:
        """Select widening points from a CFG."""
        points: Set[int] = set()

        cfg.compute_loop_headers()

        if self.include_loop_headers:
            points |= cfg.loop_headers()

        if self.include_function_entries:
            points |= cfg.function_entries()

        if self.include_exception_handlers:
            points |= cfg.exception_handlers()

        if self.include_heuristic:
            points |= self._heuristic_selection(cfg)

        return points

    def _heuristic_selection(self, cfg: CFG) -> Set[int]:
        """Heuristic for non-natural loops: find SCCs not covered by loop headers."""
        sccs = self._compute_sccs(cfg)
        headers = cfg.loop_headers()
        extra: Set[int] = set()
        for scc in sccs:
            if len(scc) <= 1:
                continue
            if not scc & headers:
                representative = min(scc)
                extra.add(representative)
        return extra

    def _compute_sccs(self, cfg: CFG) -> List[Set[int]]:
        """Tarjan's SCC algorithm."""
        index_counter = [0]
        stack: List[int] = []
        on_stack: Set[int] = set()
        index_map: Dict[int, int] = {}
        lowlink: Dict[int, int] = {}
        result: List[Set[int]] = []

        def strongconnect(v: int) -> None:
            index_map[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in cfg.blocks[v].successors:
                if w not in index_map:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index_map[w])

            if lowlink[v] == index_map[v]:
                scc: Set[int] = set()
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.add(w)
                    if w == v:
                        break
                result.append(scc)

        for bid in cfg.blocks:
            if bid not in index_map:
                strongconnect(bid)

        return result


# ===================================================================
# 12. ConvergenceAccelerator — faster convergence
# ===================================================================


@dataclass
class IterationRecord:
    """Records one iteration's state for analysis."""

    iteration: int
    state: AbstractState
    timestamp: float = 0.0


class ConvergenceAccelerator:
    """Techniques for faster convergence of abstract interpretation."""

    def __init__(
        self,
        history_size: int = 8,
        extrapolation_enabled: bool = True,
        gradient_enabled: bool = False,
    ) -> None:
        self.history_size = history_size
        self.extrapolation_enabled = extrapolation_enabled
        self.gradient_enabled = gradient_enabled
        self._history: Dict[int, List[IterationRecord]] = {}
        self._best_delays: Dict[int, int] = {}

    def record_iteration(self, point: int, iteration: int, state: AbstractState) -> None:
        record = IterationRecord(iteration=iteration, state=state, timestamp=time.monotonic())
        history = self._history.setdefault(point, [])
        history.append(record)
        if len(history) > self.history_size:
            self._history[point] = history[-self.history_size :]

    def analyze_sequence(self, point: int, var: str) -> Optional[str]:
        """Analyze the iteration sequence for a variable at a point.

        Returns a description of the detected pattern, or None.
        """
        history = self._history.get(point, [])
        if len(history) < 3:
            return None

        lo_vals = []
        hi_vals = []
        for rec in history:
            val = rec.state.bindings.get(var)
            if val is None:
                continue
            if not val.interval.is_bottom:
                lo_vals.append(val.interval.lo)
                hi_vals.append(val.interval.hi)

        if len(lo_vals) < 3:
            return None

        # Check monotonicity
        lo_increasing = all(lo_vals[i] <= lo_vals[i + 1] for i in range(len(lo_vals) - 1))
        lo_decreasing = all(lo_vals[i] >= lo_vals[i + 1] for i in range(len(lo_vals) - 1))
        hi_increasing = all(hi_vals[i] <= hi_vals[i + 1] for i in range(len(hi_vals) - 1))
        hi_decreasing = all(hi_vals[i] >= hi_vals[i + 1] for i in range(len(hi_vals) - 1))

        # Check for stabilization
        if lo_vals[-1] == lo_vals[-2] and hi_vals[-1] == hi_vals[-2]:
            return "stabilized"

        # Check for linear growth
        lo_diffs = [lo_vals[i + 1] - lo_vals[i] for i in range(len(lo_vals) - 1)]
        hi_diffs = [hi_vals[i + 1] - hi_vals[i] for i in range(len(hi_vals) - 1)]
        if lo_diffs and all(abs(d - lo_diffs[0]) < 1e-9 for d in lo_diffs):
            return f"lo_linear(step={lo_diffs[0]})"
        if hi_diffs and all(abs(d - hi_diffs[0]) < 1e-9 for d in hi_diffs):
            return f"hi_linear(step={hi_diffs[0]})"

        if lo_decreasing:
            return "lo_decreasing"
        if hi_increasing:
            return "hi_increasing"

        return "irregular"

    def extrapolate(self, point: int, var: str, current: Interval) -> Interval:
        """Extrapolate the next value based on history."""
        if not self.extrapolation_enabled:
            return current

        history = self._history.get(point, [])
        if len(history) < 3:
            return current

        lo_vals: List[float] = []
        hi_vals: List[float] = []
        for rec in history:
            val = rec.state.bindings.get(var)
            if val and not val.interval.is_bottom:
                lo_vals.append(val.interval.lo)
                hi_vals.append(val.interval.hi)

        if len(lo_vals) < 3:
            return current

        lo = current.lo
        hi = current.hi

        # Extrapolate lower bound (should be decreasing toward fixpoint)
        lo_diffs = [lo_vals[i + 1] - lo_vals[i] for i in range(len(lo_vals) - 1)]
        if lo_diffs and all(abs(d - lo_diffs[0]) < 1e-9 for d in lo_diffs) and lo_diffs[0] != 0:
            extrapolated_lo = lo + lo_diffs[0]
            lo = min(lo, extrapolated_lo)

        # Extrapolate upper bound (should be increasing toward fixpoint)
        hi_diffs = [hi_vals[i + 1] - hi_vals[i] for i in range(len(hi_vals) - 1)]
        if hi_diffs and all(abs(d - hi_diffs[0]) < 1e-9 for d in hi_diffs) and hi_diffs[0] != 0:
            extrapolated_hi = hi + hi_diffs[0]
            hi = max(hi, extrapolated_hi)

        return Interval(lo, hi)

    def suggest_delay(self, point: int) -> int:
        """Suggest a good widening delay based on convergence behavior."""
        history = self._history.get(point, [])
        if len(history) < 2:
            return 3

        convergence_count = 0
        for i in range(1, len(history)):
            prev = history[i - 1].state
            curr = history[i].state
            if curr.is_subset_of(prev):
                convergence_count += 1

        ratio = convergence_count / (len(history) - 1)
        if ratio > 0.7:
            return 1
        elif ratio > 0.4:
            return 3
        else:
            return 5

    def update_gradient(self, point: int, delay: int, quality: float) -> None:
        """Update policy gradient for learning good widening delays."""
        if not self.gradient_enabled:
            return
        current_best = self._best_delays.get(point)
        if current_best is None or quality > 0:
            self._best_delays[point] = delay


# ===================================================================
# 13. WideningStatistics — statistics about widening behavior
# ===================================================================


@dataclass
class PointStatistics:
    widening_count: int = 0
    narrowing_count: int = 0
    precision_loss_events: int = 0
    total_interval_width_before: float = 0.0
    total_interval_width_after: float = 0.0
    convergence_iteration: Optional[int] = None


class WideningStatistics:
    """Collect and report statistics about widening behavior."""

    def __init__(self) -> None:
        self._stats: Dict[int, PointStatistics] = {}
        self._global_widening_count: int = 0
        self._global_narrowing_count: int = 0
        self._start_time: float = time.monotonic()
        self._convergence_times: Dict[int, float] = {}

    def _ensure_point(self, point: int) -> PointStatistics:
        if point not in self._stats:
            self._stats[point] = PointStatistics()
        return self._stats[point]

    def record_widening(
        self,
        point: int,
        before: AbstractState,
        after: AbstractState,
    ) -> None:
        ps = self._ensure_point(point)
        ps.widening_count += 1
        self._global_widening_count += 1

        before_width = self._total_interval_width(before)
        after_width = self._total_interval_width(after)
        ps.total_interval_width_before += before_width
        ps.total_interval_width_after += after_width
        if after_width > before_width * 1.5 + 1:
            ps.precision_loss_events += 1

    def record_narrowing(
        self,
        point: int,
        before: AbstractState,
        after: AbstractState,
    ) -> None:
        ps = self._ensure_point(point)
        ps.narrowing_count += 1
        self._global_narrowing_count += 1

    def record_convergence(self, point: int, iteration: int) -> None:
        ps = self._ensure_point(point)
        ps.convergence_iteration = iteration
        self._convergence_times[point] = time.monotonic() - self._start_time

    def _total_interval_width(self, state: AbstractState) -> float:
        total = 0.0
        for v in state.bindings.values():
            w = v.interval.width()
            if w == _POS_INF:
                total += 1e9
            else:
                total += w
        return total

    def widening_count(self, point: Optional[int] = None) -> int:
        if point is not None:
            return self._ensure_point(point).widening_count
        return self._global_widening_count

    def narrowing_count(self, point: Optional[int] = None) -> int:
        if point is not None:
            return self._ensure_point(point).narrowing_count
        return self._global_narrowing_count

    def precision_loss_count(self, point: int) -> int:
        return self._ensure_point(point).precision_loss_events

    def precision_loss_ratio(self, point: int) -> float:
        ps = self._ensure_point(point)
        if ps.widening_count == 0:
            return 0.0
        return ps.precision_loss_events / ps.widening_count

    def convergence_speed(self, point: int) -> Optional[int]:
        return self._ensure_point(point).convergence_iteration

    def average_convergence_speed(self) -> float:
        speeds = [
            ps.convergence_iteration
            for ps in self._stats.values()
            if ps.convergence_iteration is not None
        ]
        if not speeds:
            return 0.0
        return sum(speeds) / len(speeds)

    def total_time(self) -> float:
        return time.monotonic() - self._start_time

    def summary(self) -> Dict[str, Any]:
        return {
            "total_widenings": self._global_widening_count,
            "total_narrowings": self._global_narrowing_count,
            "points_analyzed": len(self._stats),
            "avg_convergence_speed": self.average_convergence_speed(),
            "total_time_s": round(self.total_time(), 4),
            "per_point": {
                pt: {
                    "widenings": ps.widening_count,
                    "narrowings": ps.narrowing_count,
                    "precision_loss": ps.precision_loss_events,
                    "convergence_iter": ps.convergence_iteration,
                }
                for pt, ps in self._stats.items()
            },
        }

    def __repr__(self) -> str:
        return (
            f"WideningStatistics(widenings={self._global_widening_count}, "
            f"narrowings={self._global_narrowing_count}, "
            f"points={len(self._stats)})"
        )


# ===================================================================
# 14. ProductWidening — widening for reduced product domain
# ===================================================================


class ProductWideningMode(Enum):
    COMPONENT_WISE = auto()
    SYNCHRONIZED = auto()
    SELECTIVE = auto()


class ProductWidening(WideningOperator):
    """Widening for a reduced product of multiple abstract domains.

    Three modes:
      - COMPONENT_WISE: widen each component independently.
      - SYNCHRONIZED: widen all components at the same program points.
      - SELECTIVE: only widen components that are unstable.
    """

    def __init__(
        self,
        component_operators: List[WideningOperator],
        mode: ProductWideningMode = ProductWideningMode.COMPONENT_WISE,
        stability_threshold: int = 3,
    ) -> None:
        self.component_operators = component_operators
        self.mode = mode
        self.stability_threshold = stability_threshold
        self._component_stability: Dict[int, List[int]] = {}

    def _check_component_stability(
        self, component_idx: int, old: AbstractValue, new: AbstractValue
    ) -> bool:
        counts = self._component_stability.setdefault(component_idx, [0])
        if new.is_subset_of(old):
            counts[0] += 1
        else:
            counts[0] = 0
        return counts[0] >= self.stability_threshold

    def widen_interval(self, old: Interval, new: Interval) -> Interval:
        if self.component_operators:
            return self.component_operators[0].widen_interval(old, new)
        return StandardWidening().widen_interval(old, new)

    def widen_string(
        self, old: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        if self.component_operators:
            return self.component_operators[0].widen_string(old, new)
        return StandardWidening().widen_string(old, new)

    def widen_octagon(
        self, old: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        if self.component_operators:
            return self.component_operators[0].widen_octagon(old, new)
        return StandardWidening().widen_octagon(old, new)

    def widen_value(self, old_value: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        if not self.component_operators:
            return StandardWidening().widen_value(old_value, new_value)

        if self.mode == ProductWideningMode.COMPONENT_WISE:
            return self._widen_component_wise(old_value, new_value)
        elif self.mode == ProductWideningMode.SYNCHRONIZED:
            return self._widen_synchronized(old_value, new_value)
        else:
            return self._widen_selective(old_value, new_value)

    def _widen_component_wise(
        self, old: AbstractValue, new: AbstractValue
    ) -> AbstractValue:
        op = self.component_operators[0] if self.component_operators else StandardWidening()
        return op.widen_value(old, new)

    def _widen_synchronized(
        self, old: AbstractValue, new: AbstractValue
    ) -> AbstractValue:
        needs_widening = not new.is_subset_of(old)
        if needs_widening:
            op = self.component_operators[0] if self.component_operators else StandardWidening()
            return op.widen_value(old, new)
        return old.join(new)

    def _widen_selective(
        self, old: AbstractValue, new: AbstractValue
    ) -> AbstractValue:
        op = self.component_operators[0] if self.component_operators else StandardWidening()

        interval_stable = new.interval.is_subset_of(old.interval)
        tag_stable = new.type_tag.is_subset_of(old.type_tag)
        null_stable = new.nullity.is_subset_of(old.nullity)
        str_stable = new.string_abs.is_subset_of(old.string_abs)

        interval = old.interval if interval_stable else op.widen_interval(old.interval, new.interval)
        type_tag = old.type_tag if tag_stable else op.widen_type_tag(old.type_tag, new.type_tag)
        nullity = old.nullity if null_stable else op.widen_nullity(old.nullity, new.nullity)
        string_abs = old.string_abs if str_stable else op.widen_string(old.string_abs, new.string_abs)

        return AbstractValue(
            interval=interval,
            type_tag=type_tag,
            nullity=nullity,
            string_abs=string_abs,
        )

    def widen(self, old_value: AbstractState, new_value: AbstractState) -> AbstractState:
        all_vars = set(old_value.bindings) | set(new_value.bindings)
        result: Dict[str, AbstractValue] = {}
        for v in all_vars:
            ov = old_value.bindings.get(v, AbstractValue.bottom())
            nv = new_value.bindings.get(v, AbstractValue.bottom())
            result[v] = self.widen_value(ov, nv)
        oct: Optional[OctagonConstraints] = None
        if old_value.octagon and new_value.octagon:
            oct = self.widen_octagon(old_value.octagon, new_value.octagon)
        return AbstractState(bindings=result, octagon=oct)


# ===================================================================
# 15. WideningWithTrace — widening that records trace for debugging
# ===================================================================


@dataclass
class WideningTraceEntry:
    """One widening step recorded for debugging."""

    step_index: int
    program_point: int
    variable: str
    old_value: AbstractValue
    new_value: AbstractValue
    widened_value: AbstractValue
    component_causing_widening: str
    precision_lost: float
    timestamp: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Step {self.step_index} @ point {self.program_point}: "
            f"{self.variable}: {self.old_value.interval} ▽ {self.new_value.interval} "
            f"= {self.widened_value.interval} "
            f"[cause={self.component_causing_widening}, loss={self.precision_lost:.2f}]"
        )


class WideningWithTrace(WideningOperator):
    """Widening operator that records every step for debugging."""

    def __init__(self, inner: WideningOperator) -> None:
        self.inner = inner
        self._trace: List[WideningTraceEntry] = []
        self._step_counter: int = 0
        self._current_point: int = -1

    def set_current_point(self, point: int) -> None:
        self._current_point = point

    @property
    def trace(self) -> List[WideningTraceEntry]:
        return list(self._trace)

    def clear_trace(self) -> None:
        self._trace.clear()
        self._step_counter = 0

    def _identify_cause(self, old: AbstractValue, new: AbstractValue) -> str:
        causes: List[str] = []
        if not new.interval.is_subset_of(old.interval):
            causes.append("interval")
        if not new.type_tag.is_subset_of(old.type_tag):
            causes.append("type_tag")
        if not new.nullity.is_subset_of(old.nullity):
            causes.append("nullity")
        if not new.string_abs.is_subset_of(old.string_abs):
            causes.append("string")
        return ",".join(causes) if causes else "none"

    def _estimate_precision_loss(
        self, old: AbstractValue, widened: AbstractValue
    ) -> float:
        old_w = old.interval.width()
        new_w = widened.interval.width()
        if old_w == 0 or old_w == _POS_INF:
            if new_w == _POS_INF:
                return 1.0
            return 0.0
        if new_w == _POS_INF:
            return 1.0
        return max(0.0, (new_w - old_w) / max(old_w, 1.0))

    def widen_interval(self, old: Interval, new: Interval) -> Interval:
        return self.inner.widen_interval(old, new)

    def widen_string(
        self, old: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        return self.inner.widen_string(old, new)

    def widen_octagon(
        self, old: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        return self.inner.widen_octagon(old, new)

    def widen_value(self, old_value: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        return self.inner.widen_value(old_value, new_value)

    def widen(self, old_value: AbstractState, new_value: AbstractState) -> AbstractState:
        all_vars = set(old_value.bindings) | set(new_value.bindings)
        result: Dict[str, AbstractValue] = {}
        for v in all_vars:
            ov = old_value.bindings.get(v, AbstractValue.bottom())
            nv = new_value.bindings.get(v, AbstractValue.bottom())
            wv = self.inner.widen_value(ov, nv)
            result[v] = wv

            if not nv.is_subset_of(ov):
                entry = WideningTraceEntry(
                    step_index=self._step_counter,
                    program_point=self._current_point,
                    variable=v,
                    old_value=ov,
                    new_value=nv,
                    widened_value=wv,
                    component_causing_widening=self._identify_cause(ov, nv),
                    precision_lost=self._estimate_precision_loss(ov, wv),
                    timestamp=time.monotonic(),
                )
                self._trace.append(entry)
                self._step_counter += 1

        oct: Optional[OctagonConstraints] = None
        if old_value.octagon and new_value.octagon:
            oct = self.inner.widen_octagon(old_value.octagon, new_value.octagon)
        return AbstractState(bindings=result, octagon=oct)

    def most_imprecise_variables(self, top_k: int = 5) -> List[Tuple[str, float]]:
        loss_by_var: Dict[str, float] = {}
        for entry in self._trace:
            loss_by_var[entry.variable] = loss_by_var.get(entry.variable, 0.0) + entry.precision_lost
        ranked = sorted(loss_by_var.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def most_imprecise_points(self, top_k: int = 5) -> List[Tuple[int, float]]:
        loss_by_point: Dict[int, float] = {}
        for entry in self._trace:
            loss_by_point[entry.program_point] = (
                loss_by_point.get(entry.program_point, 0.0) + entry.precision_lost
            )
        ranked = sorted(loss_by_point.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def trace_summary(self) -> Dict[str, Any]:
        return {
            "total_steps": self._step_counter,
            "most_imprecise_vars": self.most_imprecise_variables(),
            "most_imprecise_points": self.most_imprecise_points(),
            "total_precision_loss": sum(e.precision_lost for e in self._trace),
        }


# ===================================================================
# 16. LoopInvariantWidening — widening for loop invariant inference
# ===================================================================


@dataclass
class InductionVariable:
    """Describes a detected induction variable."""

    variable: str
    init_value: Optional[float] = None
    step: Optional[float] = None
    bound: Optional[float] = None
    direction: str = "unknown"  # "increasing", "decreasing", "unknown"

    @property
    def is_linear(self) -> bool:
        return self.step is not None

    def predicted_range(self, max_iter: int = 1000) -> Optional[Interval]:
        if self.init_value is None or self.step is None:
            return None
        if self.direction == "increasing":
            hi = self.bound if self.bound is not None else self.init_value + self.step * max_iter
            return Interval(self.init_value, hi)
        elif self.direction == "decreasing":
            lo = self.bound if self.bound is not None else self.init_value + self.step * max_iter
            return Interval(lo, self.init_value)
        return None


class InductionVariableAnalyzer:
    """Analyze loops to detect induction variables."""

    def __init__(self) -> None:
        self._detected: Dict[int, List[InductionVariable]] = {}

    def analyze_loop(
        self,
        header_id: int,
        body_states: List[AbstractState],
        var_names: List[str],
    ) -> List[InductionVariable]:
        """Detect induction variables from a sequence of loop body states."""
        result: List[InductionVariable] = []

        for var in var_names:
            values: List[float] = []
            for state in body_states:
                val = state.bindings.get(var)
                if val is not None and not val.interval.is_bottom:
                    mid = (val.interval.lo + val.interval.hi) / 2
                    if mid != _POS_INF and mid != _NEG_INF:
                        values.append(mid)

            if len(values) < 2:
                continue

            diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
            if not diffs:
                continue

            if all(abs(d - diffs[0]) < 1e-9 for d in diffs):
                step = diffs[0]
                if abs(step) < 1e-15:
                    continue
                direction = "increasing" if step > 0 else "decreasing"
                iv = InductionVariable(
                    variable=var,
                    init_value=values[0],
                    step=step,
                    direction=direction,
                )
                result.append(iv)

        self._detected[header_id] = result
        return result

    def get_detected(self, header_id: int) -> List[InductionVariable]:
        return self._detected.get(header_id, [])


class LoopInvariantWidening(WideningOperator):
    """Widening specialized for loop invariant inference.

    Strategy:
      1. Start with a strong invariant candidate (meet of first few states).
      2. Weaken only as needed when the candidate is not inductive.
      3. Use loop bound information to avoid over-widening.
      4. Use induction variable analysis for precise bounds.
    """

    def __init__(
        self,
        inner: WideningOperator,
        iv_analyzer: Optional[InductionVariableAnalyzer] = None,
        warmup_iterations: int = 3,
        use_bounds: bool = True,
    ) -> None:
        self.inner = inner
        self.iv_analyzer = iv_analyzer or InductionVariableAnalyzer()
        self.warmup_iterations = warmup_iterations
        self.use_bounds = use_bounds
        self._iteration_count: Dict[int, int] = {}
        self._candidates: Dict[int, AbstractState] = {}
        self._body_states: Dict[int, List[AbstractState]] = {}
        self._induction_vars: Dict[int, Dict[str, InductionVariable]] = {}

    def set_loop_bound(self, header_id: int, bound_var: str, bound: float) -> None:
        """Register a known loop bound."""
        ivs = self._induction_vars.setdefault(header_id, {})
        if bound_var in ivs:
            ivs[bound_var].bound = bound
        else:
            ivs[bound_var] = InductionVariable(variable=bound_var, bound=bound)

    def widen_at_loop(
        self, header_id: int, old_value: AbstractState, new_value: AbstractState
    ) -> AbstractState:
        count = self._iteration_count.get(header_id, 0)
        self._iteration_count[header_id] = count + 1

        self._body_states.setdefault(header_id, []).append(new_value)

        # Warmup: use meet to build strong candidate
        if count < self.warmup_iterations:
            if header_id not in self._candidates:
                self._candidates[header_id] = new_value
            else:
                self._candidates[header_id] = self._candidates[header_id].join(new_value)
            return self._candidates[header_id]

        # After warmup: detect induction variables
        if count == self.warmup_iterations:
            body_states = self._body_states.get(header_id, [])
            all_vars = list(set().union(*(s.bindings.keys() for s in body_states)))
            detected = self.iv_analyzer.analyze_loop(header_id, body_states, all_vars)
            iv_map: Dict[str, InductionVariable] = {}
            for iv in detected:
                iv_map[iv.variable] = iv
                existing = self._induction_vars.get(header_id, {}).get(iv.variable)
                if existing and existing.bound is not None:
                    iv.bound = existing.bound
            self._induction_vars[header_id] = iv_map

        # Widen, but use IV info for induction variables
        iv_map = self._induction_vars.get(header_id, {})
        all_vars = set(old_value.bindings) | set(new_value.bindings)
        result: Dict[str, AbstractValue] = {}

        for v in all_vars:
            ov = old_value.bindings.get(v, AbstractValue.bottom())
            nv = new_value.bindings.get(v, AbstractValue.bottom())

            iv = iv_map.get(v)
            if iv is not None and self.use_bounds:
                predicted = iv.predicted_range()
                if predicted is not None:
                    iv_interval = predicted
                    widened_interval = self.inner.widen_interval(ov.interval, nv.interval)
                    refined = widened_interval.meet(iv_interval)
                    result[v] = AbstractValue(
                        interval=refined if not refined.is_bottom else widened_interval,
                        type_tag=self.inner.widen_type_tag(ov.type_tag, nv.type_tag),
                        nullity=self.inner.widen_nullity(ov.nullity, nv.nullity),
                        string_abs=self.inner.widen_string(ov.string_abs, nv.string_abs),
                    )
                    continue

            result[v] = self.inner.widen_value(ov, nv)

        oct: Optional[OctagonConstraints] = None
        if old_value.octagon and new_value.octagon:
            oct = self.inner.widen_octagon(old_value.octagon, new_value.octagon)
        return AbstractState(bindings=result, octagon=oct)

    # Default stateless interface
    def widen_interval(self, old: Interval, new: Interval) -> Interval:
        return self.inner.widen_interval(old, new)

    def widen_string(
        self, old: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        return self.inner.widen_string(old, new)

    def widen_octagon(
        self, old: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        return self.inner.widen_octagon(old, new)

    def widen_value(self, old_value: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        return self.inner.widen_value(old_value, new_value)

    def widen(self, old_value: AbstractState, new_value: AbstractState) -> AbstractState:
        return self.inner.widen(old_value, new_value)


# ===================================================================
# 17. FixpointEngine — putting it all together
# ===================================================================


@dataclass
class FixpointConfig:
    """Configuration for the fixpoint engine."""

    widening_operator: WideningOperator = field(default_factory=StandardWidening)
    narrowing_operator: NarrowingOperator = field(default_factory=StandardNarrowing)
    widening_policy: WideningPolicy = field(default_factory=LoopHeadPolicy)
    point_selector: WideningPointSelector = field(default_factory=WideningPointSelector)
    max_widening_iterations: int = 100
    max_narrowing_iterations: int = 10
    enable_narrowing: bool = True
    enable_statistics: bool = True
    enable_trace: bool = False
    convergence_accelerator: Optional[ConvergenceAccelerator] = None


class FixpointEngine:
    """Complete fixpoint computation engine with widening and narrowing."""

    def __init__(self, config: FixpointConfig) -> None:
        self.config = config
        self.stats = WideningStatistics() if config.enable_statistics else None
        self._trace_widening: Optional[WideningWithTrace] = None
        if config.enable_trace:
            self._trace_widening = WideningWithTrace(config.widening_operator)

    @property
    def widening_op(self) -> WideningOperator:
        if self._trace_widening is not None:
            return self._trace_widening
        return self.config.widening_operator

    def compute_fixpoint(
        self,
        cfg: CFG,
        transfer: Callable[[int, AbstractState], AbstractState],
        initial_state: AbstractState,
    ) -> Dict[int, AbstractState]:
        """Compute fixpoint over a CFG using widening and narrowing."""
        widening_points = self.config.point_selector.select_from_cfg(cfg)

        # Initialize states
        states: Dict[int, AbstractState] = {}
        for bid in cfg.blocks:
            states[bid] = AbstractState()
        states[cfg.entry_id] = initial_state

        # Worklist algorithm with widening
        order = cfg.topological_order()
        converged = False
        iteration = 0

        while not converged and iteration < self.config.max_widening_iterations:
            converged = True
            iteration += 1

            for bid in order:
                block = cfg.blocks[bid]
                if not block.predecessors and bid != cfg.entry_id:
                    continue

                # Compute incoming state from predecessors
                if bid == cfg.entry_id:
                    incoming = initial_state
                else:
                    pred_states = [
                        transfer(pid, states[pid]) for pid in block.predecessors if pid in states
                    ]
                    if not pred_states:
                        continue
                    incoming = pred_states[0]
                    for ps in pred_states[1:]:
                        incoming = incoming.join(ps)

                old_state = states[bid]

                # Apply widening if needed
                if bid in widening_points and self.config.widening_policy.should_widen(
                    bid, iteration, cfg
                ):
                    if self._trace_widening is not None:
                        self._trace_widening.set_current_point(bid)
                    new_state = self.widening_op.widen(old_state, incoming)
                    if self.stats is not None:
                        self.stats.record_widening(bid, old_state, new_state)
                else:
                    new_state = old_state.join(incoming)

                # Check convergence
                if not new_state.is_subset_of(old_state):
                    converged = False
                    states[bid] = new_state
                else:
                    states[bid] = new_state

                if self.config.convergence_accelerator is not None:
                    self.config.convergence_accelerator.record_iteration(bid, iteration, new_state)

            if converged:
                for bid in widening_points:
                    if self.stats is not None:
                        self.stats.record_convergence(bid, iteration)

        # Narrowing phase
        if self.config.enable_narrowing:
            narrowing = IterativeNarrowing(
                self.config.narrowing_operator,
                max_iterations=self.config.max_narrowing_iterations,
            )
            states = narrowing.narrow_at_points(
                states,
                lambda bid, st: transfer(bid, st),
                widening_points,
            )
            if self.stats is not None:
                for bid in widening_points:
                    for _ in range(narrowing.iterations_done):
                        self.stats.record_narrowing(bid, states.get(bid, AbstractState()), states.get(bid, AbstractState()))

        return states

    def compute_fixpoint_single_block(
        self,
        transfer: Callable[[AbstractState], AbstractState],
        initial: AbstractState,
    ) -> AbstractState:
        """Compute fixpoint for a single loop body."""
        current = initial
        for i in range(self.config.max_widening_iterations):
            new_state = transfer(current)
            widened = self.widening_op.widen(current, new_state)
            if widened.is_subset_of(current):
                break
            current = widened

        if self.config.enable_narrowing:
            for _ in range(self.config.max_narrowing_iterations):
                new_state = transfer(current)
                narrowed = self.config.narrowing_operator.narrow(current, new_state)
                if current.is_subset_of(narrowed):
                    break
                current = narrowed

        return current


# ===================================================================
# 18. WideningFactory — convenient construction
# ===================================================================


class WideningStrategy(Enum):
    STANDARD = auto()
    THRESHOLD = auto()
    DELAYED = auto()
    GUIDED = auto()
    JUMP_SET = auto()
    LOOP_INVARIANT = auto()


class WideningFactory:
    """Factory for constructing widening operators from configuration."""

    @staticmethod
    def create(
        strategy: WideningStrategy,
        thresholds: Optional[ThresholdSet] = None,
        delay_k: int = 3,
        property_spec: Optional[PropertySpec] = None,
        string_threshold: int = 32,
    ) -> WideningOperator:
        if strategy == WideningStrategy.STANDARD:
            return StandardWidening(string_threshold=string_threshold)

        elif strategy == WideningStrategy.THRESHOLD:
            return ThresholdWidening(
                thresholds=thresholds or DEFAULT_THRESHOLDS,
                string_threshold=string_threshold,
            )

        elif strategy == WideningStrategy.DELAYED:
            inner = StandardWidening(string_threshold=string_threshold)
            return DelayedWidening(inner=inner, default_k=delay_k)

        elif strategy == WideningStrategy.GUIDED:
            spec = property_spec or PropertySpec()
            return GuidedWidening(property_spec=spec)

        elif strategy == WideningStrategy.JUMP_SET:
            return JumpSetWidening()

        elif strategy == WideningStrategy.LOOP_INVARIANT:
            inner = ThresholdWidening(
                thresholds=thresholds or DEFAULT_THRESHOLDS,
                string_threshold=string_threshold,
            )
            return LoopInvariantWidening(inner=inner)

        raise ValueError(f"Unknown strategy: {strategy}")

    @staticmethod
    def create_config(
        strategy: WideningStrategy = WideningStrategy.STANDARD,
        enable_narrowing: bool = True,
        enable_trace: bool = False,
        thresholds: Optional[ThresholdSet] = None,
        delay_k: int = 3,
        property_spec: Optional[PropertySpec] = None,
        max_widening_iterations: int = 100,
        max_narrowing_iterations: int = 10,
    ) -> FixpointConfig:
        widen_op = WideningFactory.create(
            strategy, thresholds=thresholds, delay_k=delay_k, property_spec=property_spec
        )
        return FixpointConfig(
            widening_operator=widen_op,
            narrowing_operator=StandardNarrowing(),
            widening_policy=LoopHeadPolicy(),
            point_selector=WideningPointSelector(),
            max_widening_iterations=max_widening_iterations,
            max_narrowing_iterations=max_narrowing_iterations,
            enable_narrowing=enable_narrowing,
            enable_trace=enable_trace,
        )


# ===================================================================
# 19. Composite operators — combining multiple strategies
# ===================================================================


class SequentialWidening(WideningOperator):
    """Apply multiple widening operators in sequence, keeping the most
    precise result at each step."""

    def __init__(self, operators: List[WideningOperator]) -> None:
        if not operators:
            raise ValueError("Need at least one operator")
        self.operators = operators

    def widen_interval(self, old: Interval, new: Interval) -> Interval:
        result = self.operators[0].widen_interval(old, new)
        for op in self.operators[1:]:
            candidate = op.widen_interval(old, new)
            if candidate.is_subset_of(result):
                result = candidate
        return result

    def widen_string(
        self, old: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        return self.operators[0].widen_string(old, new)

    def widen_octagon(
        self, old: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        return self.operators[0].widen_octagon(old, new)

    def widen_value(self, old_value: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        result = self.operators[0].widen_value(old_value, new_value)
        for op in self.operators[1:]:
            candidate = op.widen_value(old_value, new_value)
            if candidate.is_subset_of(result):
                result = candidate
        return result

    def widen(self, old_value: AbstractState, new_value: AbstractState) -> AbstractState:
        result = self.operators[0].widen(old_value, new_value)
        for op in self.operators[1:]:
            candidate = op.widen(old_value, new_value)
            if candidate.is_subset_of(result):
                result = candidate
        return result


class AdaptiveWidening(WideningOperator):
    """Automatically switch between strategies based on convergence behavior."""

    def __init__(
        self,
        conservative: WideningOperator,
        aggressive: WideningOperator,
        switch_threshold: int = 10,
    ) -> None:
        self.conservative = conservative
        self.aggressive = aggressive
        self.switch_threshold = switch_threshold
        self._iteration_counts: Dict[int, int] = {}
        self._current_point: int = -1

    def set_point(self, point: int) -> None:
        self._current_point = point
        self._iteration_counts.setdefault(point, 0)
        self._iteration_counts[point] += 1

    def _select_operator(self) -> WideningOperator:
        count = self._iteration_counts.get(self._current_point, 0)
        if count > self.switch_threshold:
            return self.aggressive
        return self.conservative

    def widen_interval(self, old: Interval, new: Interval) -> Interval:
        return self._select_operator().widen_interval(old, new)

    def widen_string(
        self, old: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        return self._select_operator().widen_string(old, new)

    def widen_octagon(
        self, old: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        return self._select_operator().widen_octagon(old, new)

    def widen_value(self, old_value: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        return self._select_operator().widen_value(old_value, new_value)

    def widen(self, old_value: AbstractState, new_value: AbstractState) -> AbstractState:
        return self._select_operator().widen(old_value, new_value)


# ===================================================================
# 20. Narrowing variants
# ===================================================================


class GradualNarrowing(NarrowingOperator):
    """Narrowing that gradually tightens bounds rather than jumping."""

    def __init__(self, step_factor: float = 0.5) -> None:
        self.step_factor = step_factor

    def narrow_interval(self, widened: Interval, new: Interval) -> Interval:
        if widened.is_bottom or new.is_bottom:
            return Interval.bottom()
        lo = widened.lo
        hi = widened.hi
        if widened.lo == _NEG_INF and new.lo != _NEG_INF:
            lo = new.lo
        elif widened.lo != _NEG_INF and new.lo > widened.lo:
            lo = widened.lo + self.step_factor * (new.lo - widened.lo)
        if widened.hi == _POS_INF and new.hi != _POS_INF:
            hi = new.hi
        elif widened.hi != _POS_INF and new.hi < widened.hi:
            hi = widened.hi + self.step_factor * (new.hi - widened.hi)
        return Interval(lo, hi)

    def narrow_string(
        self, widened: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        if widened.is_top and not new.is_top:
            return new
        return widened

    def narrow_octagon(
        self, widened: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        assert widened.n_vars == new.n_vars
        dim = widened.dim
        m = [row[:] for row in widened.matrix]
        for i in range(dim):
            for j in range(dim):
                if widened.matrix[i][j] == _POS_INF and new.matrix[i][j] != _POS_INF:
                    m[i][j] = new.matrix[i][j]
                elif (
                    widened.matrix[i][j] != _POS_INF
                    and new.matrix[i][j] < widened.matrix[i][j]
                ):
                    m[i][j] = widened.matrix[i][j] + self.step_factor * (
                        new.matrix[i][j] - widened.matrix[i][j]
                    )
        return OctagonConstraints(widened.n_vars, m)

    def narrow_value(self, widened: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        return AbstractValue(
            interval=self.narrow_interval(widened.interval, new_value.interval),
            type_tag=self.narrow_type_tag(widened.type_tag, new_value.type_tag),
            nullity=self.narrow_nullity(widened.nullity, new_value.nullity),
            string_abs=self.narrow_string(widened.string_abs, new_value.string_abs),
        )

    def narrow(self, widened: AbstractState, new_value: AbstractState) -> AbstractState:
        all_vars = set(widened.bindings) | set(new_value.bindings)
        result: Dict[str, AbstractValue] = {}
        for v in all_vars:
            wv = widened.bindings.get(v, AbstractValue.top())
            nv = new_value.bindings.get(v, AbstractValue.top())
            result[v] = self.narrow_value(wv, nv)
        oct: Optional[OctagonConstraints] = None
        if widened.octagon and new_value.octagon:
            oct = self.narrow_octagon(widened.octagon, new_value.octagon)
        return AbstractState(bindings=result, octagon=oct)


class ThresholdNarrowing(NarrowingOperator):
    """Narrowing that uses thresholds to recover precision step by step."""

    def __init__(self, thresholds: Optional[ThresholdSet] = None) -> None:
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    def narrow_interval(self, widened: Interval, new: Interval) -> Interval:
        if widened.is_bottom or new.is_bottom:
            return Interval.bottom()
        lo = widened.lo
        hi = widened.hi
        if widened.lo == _NEG_INF:
            lo = self.thresholds.next_below(new.lo)
            if lo == _NEG_INF:
                lo = new.lo
        if widened.hi == _POS_INF:
            hi = self.thresholds.next_above(new.hi)
            if hi == _POS_INF:
                hi = new.hi
        return Interval(lo, hi)

    def narrow_string(
        self, widened: StringAbstraction, new: StringAbstraction
    ) -> StringAbstraction:
        if widened.is_top and not new.is_top:
            return new
        return widened

    def narrow_octagon(
        self, widened: OctagonConstraints, new: OctagonConstraints
    ) -> OctagonConstraints:
        assert widened.n_vars == new.n_vars
        dim = widened.dim
        m = [row[:] for row in widened.matrix]
        for i in range(dim):
            for j in range(dim):
                if widened.matrix[i][j] == _POS_INF:
                    m[i][j] = new.matrix[i][j]
        return OctagonConstraints(widened.n_vars, m)

    def narrow_value(self, widened: AbstractValue, new_value: AbstractValue) -> AbstractValue:
        return AbstractValue(
            interval=self.narrow_interval(widened.interval, new_value.interval),
            type_tag=self.narrow_type_tag(widened.type_tag, new_value.type_tag),
            nullity=self.narrow_nullity(widened.nullity, new_value.nullity),
            string_abs=self.narrow_string(widened.string_abs, new_value.string_abs),
        )

    def narrow(self, widened: AbstractState, new_value: AbstractState) -> AbstractState:
        all_vars = set(widened.bindings) | set(new_value.bindings)
        result: Dict[str, AbstractValue] = {}
        for v in all_vars:
            wv = widened.bindings.get(v, AbstractValue.top())
            nv = new_value.bindings.get(v, AbstractValue.top())
            result[v] = self.narrow_value(wv, nv)
        oct: Optional[OctagonConstraints] = None
        if widened.octagon and new_value.octagon:
            oct = self.narrow_octagon(widened.octagon, new_value.octagon)
        return AbstractState(bindings=result, octagon=oct)


# ===================================================================
# 21. Specialized interval widening helpers
# ===================================================================


class IntervalWideningUtils:
    """Utility functions for interval widening strategies."""

    @staticmethod
    def widen_with_landmarks(
        old: Interval, new: Interval, landmarks: List[float]
    ) -> Interval:
        """Widen using program-specific landmarks."""
        if old.is_bottom:
            return new
        if new.is_bottom:
            return old
        sorted_lm = sorted(landmarks)
        lo: float
        hi: float
        if new.lo < old.lo:
            lo = _NEG_INF
            for lm in sorted_lm:
                if lm <= new.lo:
                    lo = lm
                    break
            if lo == _NEG_INF:
                for lm in reversed(sorted_lm):
                    if lm <= new.lo:
                        lo = lm
                        break
        else:
            lo = old.lo
        if new.hi > old.hi:
            hi = _POS_INF
            for lm in sorted_lm:
                if lm >= new.hi:
                    hi = lm
                    break
        else:
            hi = old.hi
        return Interval(lo, hi)

    @staticmethod
    def widen_exponential(old: Interval, new: Interval, base: float = 2.0) -> Interval:
        """Exponential widening: double the range on each widening step."""
        if old.is_bottom:
            return new
        if new.is_bottom:
            return old
        lo = old.lo
        hi = old.hi
        if new.lo < old.lo:
            diff = old.lo - new.lo
            lo = old.lo - diff * base
        if new.hi > old.hi:
            diff = new.hi - old.hi
            hi = old.hi + diff * base
        return Interval(lo, hi)

    @staticmethod
    def widen_with_max_range(
        old: Interval, new: Interval, max_range: float
    ) -> Interval:
        """Widen but cap the total range."""
        if old.is_bottom:
            return new
        if new.is_bottom:
            return old
        lo = min(old.lo, new.lo)
        hi = max(old.hi, new.hi)
        if hi - lo > max_range:
            return Interval(_NEG_INF, _POS_INF)
        return Interval(lo, hi)

    @staticmethod
    def interpolate(a: Interval, b: Interval, t: float) -> Interval:
        """Linear interpolation between two intervals."""
        if a.is_bottom:
            return b
        if b.is_bottom:
            return a
        lo_a = a.lo if a.lo != _NEG_INF else -1e18
        hi_a = a.hi if a.hi != _POS_INF else 1e18
        lo_b = b.lo if b.lo != _NEG_INF else -1e18
        hi_b = b.hi if b.hi != _POS_INF else 1e18
        lo = lo_a + t * (lo_b - lo_a)
        hi = hi_a + t * (hi_b - hi_a)
        return Interval(lo, hi)


# ===================================================================
# 22. Octagon widening / narrowing helpers
# ===================================================================


class OctagonWideningUtils:
    """Utility functions for octagon domain widening."""

    @staticmethod
    def widen_with_thresholds(
        old: OctagonConstraints,
        new: OctagonConstraints,
        thresholds: ThresholdSet,
    ) -> OctagonConstraints:
        assert old.n_vars == new.n_vars
        dim = old.dim
        m = [row[:] for row in old.matrix]
        for i in range(dim):
            for j in range(dim):
                if new.matrix[i][j] > old.matrix[i][j]:
                    m[i][j] = thresholds.next_above(new.matrix[i][j])
        return OctagonConstraints(old.n_vars, m)

    @staticmethod
    def project_interval(oct: OctagonConstraints, var_idx: int) -> Interval:
        """Extract an interval for a single variable from octagon constraints."""
        pos = 2 * var_idx
        neg = 2 * var_idx + 1
        hi = oct.matrix[pos][neg] / 2.0 if oct.matrix[pos][neg] != _POS_INF else _POS_INF
        lo = -oct.matrix[neg][pos] / 2.0 if oct.matrix[neg][pos] != _POS_INF else _NEG_INF
        return Interval(lo, hi)

    @staticmethod
    def inject_interval(
        oct: OctagonConstraints, var_idx: int, interval: Interval
    ) -> OctagonConstraints:
        """Inject an interval constraint for a variable into the octagon."""
        m = [row[:] for row in oct.matrix]
        pos = 2 * var_idx
        neg = 2 * var_idx + 1
        if interval.hi != _POS_INF:
            m[pos][neg] = min(m[pos][neg], 2.0 * interval.hi)
        if interval.lo != _NEG_INF:
            m[neg][pos] = min(m[neg][pos], -2.0 * interval.lo)
        return OctagonConstraints(oct.n_vars, m)

    @staticmethod
    def count_finite_constraints(oct: OctagonConstraints) -> int:
        count = 0
        dim = oct.dim
        for i in range(dim):
            for j in range(dim):
                if i != j and oct.matrix[i][j] != _POS_INF:
                    count += 1
        return count

    @staticmethod
    def constraint_density(oct: OctagonConstraints) -> float:
        dim = oct.dim
        total = dim * (dim - 1)
        finite = OctagonWideningUtils.count_finite_constraints(oct)
        return finite / total if total > 0 else 0.0


# ===================================================================
# 23. Domain-specific widening helpers
# ===================================================================


class TypeTagWideningUtils:
    """Utility functions for type tag widening — always stable (finite domain)."""

    @staticmethod
    def widen(old: TypeTagSet, new: TypeTagSet) -> TypeTagSet:
        return old.join(new)

    @staticmethod
    def narrow(widened: TypeTagSet, new: TypeTagSet) -> TypeTagSet:
        return widened.meet(new)

    @staticmethod
    def distance(a: TypeTagSet, b: TypeTagSet) -> int:
        return len(a.tags.symmetric_difference(b.tags))


class NullityWideningUtils:
    """Utility functions for nullity widening — always stable (finite domain)."""

    @staticmethod
    def widen(old: Nullity, new: Nullity) -> Nullity:
        return old.join(new)

    @staticmethod
    def narrow(widened: Nullity, new: Nullity) -> Nullity:
        return widened.meet(new)


class StringWideningUtils:
    """Utility functions for string abstraction widening."""

    @staticmethod
    def widen(
        old: StringAbstraction,
        new: StringAbstraction,
        threshold: int = 32,
    ) -> StringAbstraction:
        return old.join(new, threshold=threshold)

    @staticmethod
    def widen_with_prefix(
        old: StringAbstraction,
        new: StringAbstraction,
        max_prefix_len: int = 8,
    ) -> StringAbstraction:
        """Widen strings by keeping common prefixes."""
        if old.is_bottom:
            return new
        if new.is_bottom:
            return old
        if old.is_top or new.is_top:
            return StringAbstraction.top()
        assert old.values is not None and new.values is not None
        all_strings = old.values | new.values
        if len(all_strings) <= 32:
            return StringAbstraction(frozenset(all_strings))
        prefixes: Set[str] = set()
        for s in all_strings:
            prefixes.add(s[:max_prefix_len])
        if len(prefixes) <= 32:
            return StringAbstraction(frozenset(prefixes))
        return StringAbstraction.top()


# ===================================================================
# 24. Validation utilities
# ===================================================================


class WideningValidator:
    """Validate that widening operators satisfy their contracts."""

    @staticmethod
    def validate_widening_monotone(
        op: WideningOperator,
        old: AbstractValue,
        new: AbstractValue,
    ) -> bool:
        """Check: old ⊑ widen(old, new)."""
        widened = op.widen_value(old, new)
        return old.is_subset_of(widened)

    @staticmethod
    def validate_widening_covers_new(
        op: WideningOperator,
        old: AbstractValue,
        new: AbstractValue,
    ) -> bool:
        """Check: new ⊑ widen(old, new)."""
        widened = op.widen_value(old, new)
        return new.is_subset_of(widened)

    @staticmethod
    def validate_narrowing(
        op: NarrowingOperator,
        widened: AbstractValue,
        new: AbstractValue,
    ) -> bool:
        """Check: new ⊑ narrow(widened, new) ⊑ widened."""
        narrowed = op.narrow_value(widened, new)
        return new.is_subset_of(narrowed) and narrowed.is_subset_of(widened)

    @staticmethod
    def validate_stabilization(
        op: WideningOperator,
        initial: AbstractValue,
        step: Callable[[AbstractValue], AbstractValue],
        max_iter: int = 200,
    ) -> Tuple[bool, int]:
        """Check that widening stabilizes within max_iter steps."""
        current = initial
        for i in range(max_iter):
            next_val = step(current)
            widened = op.widen_value(current, next_val)
            if widened.is_subset_of(current):
                return True, i + 1
            current = widened
        return False, max_iter

    @staticmethod
    def validate_chain(
        op: WideningOperator,
        chain: List[AbstractValue],
    ) -> Tuple[bool, int]:
        """Validate that applying widening to a chain stabilizes."""
        if not chain:
            return True, 0
        current = chain[0]
        for i, val in enumerate(chain[1:], 1):
            widened = op.widen_value(current, val)
            if widened.is_subset_of(current):
                return True, i
            current = widened
        return False, len(chain)

    @staticmethod
    def stress_test(
        op: WideningOperator,
        n_tests: int = 100,
        max_chain_len: int = 50,
    ) -> Dict[str, Any]:
        """Stress test a widening operator."""
        import random

        results = {
            "monotone_pass": 0,
            "monotone_fail": 0,
            "covers_pass": 0,
            "covers_fail": 0,
            "stabilized": 0,
            "not_stabilized": 0,
            "avg_stabilization_iter": 0.0,
        }
        stabilization_iters: List[int] = []

        for _ in range(n_tests):
            lo1 = random.randint(-100, 100)
            hi1 = lo1 + random.randint(0, 50)
            lo2 = random.randint(-100, 100)
            hi2 = lo2 + random.randint(0, 50)
            old = AbstractValue(interval=Interval(lo1, hi1))
            new = AbstractValue(interval=Interval(lo2, hi2))

            if WideningValidator.validate_widening_monotone(op, old, new):
                results["monotone_pass"] += 1
            else:
                results["monotone_fail"] += 1

            if WideningValidator.validate_widening_covers_new(op, old, new):
                results["covers_pass"] += 1
            else:
                results["covers_fail"] += 1

        for _ in range(n_tests):
            lo = random.randint(-50, 50)
            initial = AbstractValue(interval=Interval(lo, lo))

            def random_step(v: AbstractValue) -> AbstractValue:
                delta = random.randint(1, 5)
                return AbstractValue(
                    interval=Interval(v.interval.lo - delta, v.interval.hi + delta)
                )

            stabilized, iters = WideningValidator.validate_stabilization(
                op, initial, random_step, max_iter=max_chain_len
            )
            if stabilized:
                results["stabilized"] += 1
                stabilization_iters.append(iters)
            else:
                results["not_stabilized"] += 1

        if stabilization_iters:
            results["avg_stabilization_iter"] = sum(stabilization_iters) / len(
                stabilization_iters
            )

        return results


# ===================================================================
# 25. Serialization helpers
# ===================================================================


class WideningConfigSerializer:
    """Serialize/deserialize widening configurations."""

    @staticmethod
    def to_dict(config: FixpointConfig) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "max_widening_iterations": config.max_widening_iterations,
            "max_narrowing_iterations": config.max_narrowing_iterations,
            "enable_narrowing": config.enable_narrowing,
            "enable_statistics": config.enable_statistics,
            "enable_trace": config.enable_trace,
            "widening_policy": config.widening_policy.name(),
        }
        op = config.widening_operator
        if isinstance(op, StandardWidening):
            d["widening_type"] = "standard"
            d["string_threshold"] = op.string_threshold
        elif isinstance(op, ThresholdWidening):
            d["widening_type"] = "threshold"
            d["thresholds"] = op.thresholds.values
            d["string_threshold"] = op.string_threshold
        elif isinstance(op, DelayedWidening):
            d["widening_type"] = "delayed"
            d["default_k"] = op.default_k
            d["adaptive"] = op.adaptive
        elif isinstance(op, GuidedWidening):
            d["widening_type"] = "guided"
            d["critical_variables"] = sorted(op.property_spec.critical_variables)
        elif isinstance(op, JumpSetWidening):
            d["widening_type"] = "jump_set"
        elif isinstance(op, LoopInvariantWidening):
            d["widening_type"] = "loop_invariant"
            d["warmup_iterations"] = op.warmup_iterations
        else:
            d["widening_type"] = "unknown"
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> FixpointConfig:
        wtype = d.get("widening_type", "standard")
        op: WideningOperator
        if wtype == "standard":
            op = StandardWidening(string_threshold=d.get("string_threshold", 32))
        elif wtype == "threshold":
            thresholds = ThresholdSet(d.get("thresholds", DEFAULT_THRESHOLDS.values))
            op = ThresholdWidening(
                thresholds=thresholds,
                string_threshold=d.get("string_threshold", 32),
            )
        elif wtype == "delayed":
            inner = StandardWidening()
            op = DelayedWidening(
                inner=inner,
                default_k=d.get("default_k", 3),
                adaptive=d.get("adaptive", False),
            )
        elif wtype == "guided":
            spec = PropertySpec(
                critical_variables=set(d.get("critical_variables", []))
            )
            op = GuidedWidening(property_spec=spec)
        elif wtype == "jump_set":
            op = JumpSetWidening()
        elif wtype == "loop_invariant":
            inner = ThresholdWidening()
            op = LoopInvariantWidening(
                inner=inner,
                warmup_iterations=d.get("warmup_iterations", 3),
            )
        else:
            op = StandardWidening()

        policy_name = d.get("widening_policy", "loop_head")
        policy: WideningPolicy
        if policy_name == "all_join":
            policy = AllJoinPolicy()
        elif policy_name == "back_edge":
            policy = BackEdgePolicy()
        else:
            policy = LoopHeadPolicy()

        return FixpointConfig(
            widening_operator=op,
            narrowing_operator=StandardNarrowing(),
            widening_policy=policy,
            point_selector=WideningPointSelector(),
            max_widening_iterations=d.get("max_widening_iterations", 100),
            max_narrowing_iterations=d.get("max_narrowing_iterations", 10),
            enable_narrowing=d.get("enable_narrowing", True),
            enable_statistics=d.get("enable_statistics", True),
            enable_trace=d.get("enable_trace", False),
        )


# ===================================================================
# 26. Pretty printing
# ===================================================================


class WideningPrinter:
    """Pretty-print widening-related objects."""

    @staticmethod
    def format_interval(itv: Interval) -> str:
        return repr(itv)

    @staticmethod
    def format_value(val: AbstractValue) -> str:
        parts: List[str] = []
        parts.append(f"itv={val.interval}")
        parts.append(f"tag={val.type_tag}")
        parts.append(f"null={val.nullity.name}")
        parts.append(f"str={val.string_abs}")
        return "(" + ", ".join(parts) + ")"

    @staticmethod
    def format_state(state: AbstractState) -> str:
        if not state.bindings:
            return "{}"
        lines = []
        for var in sorted(state.bindings):
            val = state.bindings[var]
            lines.append(f"  {var}: {WideningPrinter.format_value(val)}")
        return "{\n" + "\n".join(lines) + "\n}"

    @staticmethod
    def format_trace(trace: List[WideningTraceEntry]) -> str:
        lines = [repr(entry) for entry in trace]
        return "\n".join(lines)

    @staticmethod
    def format_statistics(stats: WideningStatistics) -> str:
        summary = stats.summary()
        lines = [
            f"Total widenings: {summary['total_widenings']}",
            f"Total narrowings: {summary['total_narrowings']}",
            f"Points analyzed: {summary['points_analyzed']}",
            f"Avg convergence speed: {summary['avg_convergence_speed']:.1f}",
            f"Total time: {summary['total_time_s']:.4f}s",
        ]
        for pt, info in summary.get("per_point", {}).items():
            lines.append(
                f"  Point {pt}: w={info['widenings']}, n={info['narrowings']}, "
                f"loss={info['precision_loss']}, conv={info['convergence_iter']}"
            )
        return "\n".join(lines)


# ===================================================================
# Exports
# ===================================================================

__all__ = [
    # Domain types
    "Interval",
    "TypeTag",
    "TypeTagSet",
    "Nullity",
    "StringAbstraction",
    "OctagonConstraints",
    "AbstractValue",
    "AbstractState",
    # CFG
    "CFGBlock",
    "CFG",
    # Expression
    "ExprKind",
    "Expr",
    # Widening
    "WideningOperator",
    "StandardWidening",
    "ThresholdSet",
    "ThresholdWidening",
    "ThresholdCollectionStrategy",
    "extract_thresholds",
    "DEFAULT_THRESHOLDS",
    "DelayedWidening",
    "GuidedWidening",
    "PropertySpec",
    "JumpSetWidening",
    "ValueSequence",
    "SequentialWidening",
    "AdaptiveWidening",
    "ProductWidening",
    "ProductWideningMode",
    "WideningWithTrace",
    "WideningTraceEntry",
    "LoopInvariantWidening",
    "InductionVariable",
    "InductionVariableAnalyzer",
    # Narrowing
    "NarrowingOperator",
    "StandardNarrowing",
    "GradualNarrowing",
    "ThresholdNarrowing",
    "IterativeNarrowing",
    # Policy
    "WideningPolicy",
    "LoopHeadPolicy",
    "AllJoinPolicy",
    "BackEdgePolicy",
    "NestedLoopPolicy",
    "CustomPolicy",
    "WideningPointSelector",
    # Engine
    "ConvergenceAccelerator",
    "IterationRecord",
    "FixpointConfig",
    "FixpointEngine",
    "WideningStatistics",
    "PointStatistics",
    # Factory
    "WideningStrategy",
    "WideningFactory",
    # Utils
    "IntervalWideningUtils",
    "OctagonWideningUtils",
    "TypeTagWideningUtils",
    "NullityWideningUtils",
    "StringWideningUtils",
    # Validation
    "WideningValidator",
    # Serialization
    "WideningConfigSerializer",
    # Printing
    "WideningPrinter",
]

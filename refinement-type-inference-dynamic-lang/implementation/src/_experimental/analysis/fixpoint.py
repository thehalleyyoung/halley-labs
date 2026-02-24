from __future__ import annotations

"""
Fixed-point computation engine for refinement type inference.

Provides generic fixed-point solvers, widening/narrowing strategies,
loop analysis, lattice utilities, and convergence acceleration for
abstract interpretation over dynamically-typed programs.
"""

import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterator,
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
import heapq
import math


# ---------------------------------------------------------------------------
# Local type stubs
# ---------------------------------------------------------------------------

T = TypeVar("T")
S = TypeVar("S")


@dataclass(frozen=True)
class ProgramPoint:
    """A point in the program (block label + optional instruction index)."""
    block: str
    index: int = 0

    def __str__(self) -> str:
        return f"{self.block}:{self.index}"


@dataclass
class BasicBlock:
    label: str
    instructions: List[Any] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)


@dataclass
class CFG:
    """Control-flow graph."""
    blocks: Dict[str, BasicBlock] = field(default_factory=dict)
    entry: str = "entry"
    exit: str = "exit"

    def successors(self, label: str) -> List[str]:
        bb = self.blocks.get(label)
        return bb.successors if bb else []

    def predecessors(self, label: str) -> List[str]:
        bb = self.blocks.get(label)
        return bb.predecessors if bb else []

    def all_labels(self) -> List[str]:
        return list(self.blocks.keys())


# =========================================================================
#  Lattice infrastructure
# =========================================================================

class LatticeElement(Protocol):
    """Protocol for elements of a lattice."""

    def join(self, other: LatticeElement) -> LatticeElement: ...
    def meet(self, other: LatticeElement) -> LatticeElement: ...
    def leq(self, other: LatticeElement) -> bool: ...


@dataclass(frozen=True)
class Top:
    """Top element of any lattice."""
    def join(self, other: Any) -> Top:
        return self
    def meet(self, other: Any) -> Any:
        return other
    def leq(self, other: Any) -> bool:
        return isinstance(other, Top)
    def __repr__(self) -> str:
        return "⊤"


@dataclass(frozen=True)
class Bottom:
    """Bottom element of any lattice."""
    def join(self, other: Any) -> Any:
        return other
    def meet(self, other: Any) -> Bottom:
        return self
    def leq(self, other: Any) -> bool:
        return True
    def __repr__(self) -> str:
        return "⊥"


# -------------------------------------------------------------------------
#  Interval lattice
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class Interval:
    """An integer interval [lo, hi].  None means ±∞."""
    lo: Optional[int] = None  # None → -∞
    hi: Optional[int] = None  # None → +∞

    @staticmethod
    def bottom() -> Interval:
        return Interval(lo=1, hi=0)  # empty

    @staticmethod
    def top() -> Interval:
        return Interval(lo=None, hi=None)

    @staticmethod
    def const(v: int) -> Interval:
        return Interval(lo=v, hi=v)

    def is_bottom(self) -> bool:
        if self.lo is not None and self.hi is not None:
            return self.lo > self.hi
        return False

    def is_top(self) -> bool:
        return self.lo is None and self.hi is None

    def contains(self, v: int) -> bool:
        if self.is_bottom():
            return False
        lo_ok = self.lo is None or v >= self.lo
        hi_ok = self.hi is None or v <= self.hi
        return lo_ok and hi_ok

    def join(self, other: Interval) -> Interval:
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self
        lo = min(self.lo, other.lo) if self.lo is not None and other.lo is not None else None
        hi = max(self.hi, other.hi) if self.hi is not None and other.hi is not None else None
        return Interval(lo=lo, hi=hi)

    def meet(self, other: Interval) -> Interval:
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()
        lo = max(self.lo, other.lo) if self.lo is not None and other.lo is not None else (self.lo or other.lo)
        hi = min(self.hi, other.hi) if self.hi is not None and other.hi is not None else (self.hi or other.hi)
        return Interval(lo=lo, hi=hi)

    def leq(self, other: Interval) -> bool:
        if self.is_bottom():
            return True
        if other.is_bottom():
            return False
        lo_ok = other.lo is None or (self.lo is not None and self.lo >= other.lo)
        hi_ok = other.hi is None or (self.hi is not None and self.hi <= other.hi)
        return lo_ok and hi_ok

    def widen(self, other: Interval) -> Interval:
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self
        lo = self.lo if (other.lo is not None and self.lo is not None and other.lo >= self.lo) else None
        hi = self.hi if (other.hi is not None and self.hi is not None and other.hi <= self.hi) else None
        return Interval(lo=lo, hi=hi)

    def narrow(self, other: Interval) -> Interval:
        if self.is_bottom():
            return other
        lo = other.lo if self.lo is None else self.lo
        hi = other.hi if self.hi is None else self.hi
        return Interval(lo=lo, hi=hi)

    def add(self, other: Interval) -> Interval:
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()
        lo = (self.lo + other.lo) if self.lo is not None and other.lo is not None else None
        hi = (self.hi + other.hi) if self.hi is not None and other.hi is not None else None
        return Interval(lo=lo, hi=hi)

    def neg(self) -> Interval:
        if self.is_bottom():
            return self
        new_lo = -self.hi if self.hi is not None else None
        new_hi = -self.lo if self.lo is not None else None
        return Interval(lo=new_lo, hi=new_hi)

    def sub(self, other: Interval) -> Interval:
        return self.add(other.neg())

    def mul(self, other: Interval) -> Interval:
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()
        bounds: list[Optional[int]] = []
        for a in (self.lo, self.hi):
            for b in (other.lo, other.hi):
                if a is not None and b is not None:
                    bounds.append(a * b)
                else:
                    bounds.append(None)
        finite = [b for b in bounds if b is not None]
        if not finite:
            return Interval.top()
        has_none = any(b is None for b in bounds)
        lo = None if has_none else min(finite)
        hi = None if has_none else max(finite)
        return Interval(lo=lo, hi=hi)

    def __repr__(self) -> str:
        if self.is_bottom():
            return "⊥"
        lo_s = str(self.lo) if self.lo is not None else "-∞"
        hi_s = str(self.hi) if self.hi is not None else "+∞"
        return f"[{lo_s}, {hi_s}]"


# -------------------------------------------------------------------------
#  Flat lattice
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class FlatLattice(Generic[T]):
    """Flat lattice: ⊥ < {constants} < ⊤."""
    value: Optional[T] = None
    _is_top: bool = False
    _is_bottom: bool = True

    @staticmethod
    def bot() -> FlatLattice:
        return FlatLattice(_is_bottom=True)

    @staticmethod
    def top() -> FlatLattice:
        return FlatLattice(_is_top=True, _is_bottom=False)

    @staticmethod
    def of(v: T) -> FlatLattice[T]:
        return FlatLattice(value=v, _is_top=False, _is_bottom=False)

    def is_bottom(self) -> bool:
        return self._is_bottom

    def is_top(self) -> bool:
        return self._is_top

    def join(self, other: FlatLattice[T]) -> FlatLattice[T]:
        if self._is_bottom:
            return other
        if other._is_bottom:
            return self
        if self._is_top or other._is_top:
            return FlatLattice.top()
        if self.value == other.value:
            return self
        return FlatLattice.top()

    def meet(self, other: FlatLattice[T]) -> FlatLattice[T]:
        if self._is_top:
            return other
        if other._is_top:
            return self
        if self._is_bottom or other._is_bottom:
            return FlatLattice.bot()
        if self.value == other.value:
            return self
        return FlatLattice.bot()

    def leq(self, other: FlatLattice[T]) -> bool:
        if self._is_bottom:
            return True
        if other._is_top:
            return True
        if self._is_top:
            return other._is_top
        if other._is_bottom:
            return False
        return self.value == other.value


# -------------------------------------------------------------------------
#  Lifted lattice
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class LiftedLattice(Generic[T]):
    """Adds a bottom element below an existing lattice element."""
    value: Optional[T] = None
    _is_bottom: bool = True

    @staticmethod
    def bot() -> LiftedLattice:
        return LiftedLattice(_is_bottom=True)

    @staticmethod
    def lift(v: T) -> LiftedLattice[T]:
        return LiftedLattice(value=v, _is_bottom=False)

    def is_bottom(self) -> bool:
        return self._is_bottom


# -------------------------------------------------------------------------
#  Constant propagation lattice
# -------------------------------------------------------------------------

class ConstValue(Enum):
    TOP = auto()
    BOTTOM = auto()


@dataclass(frozen=True)
class ConstLattice:
    """Constant-propagation lattice: ⊥ < {c} < ⊤."""
    kind: ConstValue = ConstValue.BOTTOM
    value: Optional[Any] = None

    @staticmethod
    def bottom() -> ConstLattice:
        return ConstLattice(kind=ConstValue.BOTTOM)

    @staticmethod
    def top() -> ConstLattice:
        return ConstLattice(kind=ConstValue.TOP)

    @staticmethod
    def const(v: Any) -> ConstLattice:
        return ConstLattice(kind=ConstValue.BOTTOM, value=v)

    def is_bottom(self) -> bool:
        return self.kind == ConstValue.BOTTOM and self.value is None

    def is_top(self) -> bool:
        return self.kind == ConstValue.TOP

    def is_const(self) -> bool:
        return self.kind == ConstValue.BOTTOM and self.value is not None

    def join(self, other: ConstLattice) -> ConstLattice:
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self
        if self.is_top() or other.is_top():
            return ConstLattice.top()
        if self.value == other.value:
            return self
        return ConstLattice.top()

    def meet(self, other: ConstLattice) -> ConstLattice:
        if self.is_top():
            return other
        if other.is_top():
            return self
        if self.is_bottom() or other.is_bottom():
            return ConstLattice.bottom()
        if self.value == other.value:
            return self
        return ConstLattice.bottom()

    def leq(self, other: ConstLattice) -> bool:
        if self.is_bottom():
            return True
        if other.is_top():
            return True
        if self.is_const() and other.is_const():
            return self.value == other.value
        return False


# -------------------------------------------------------------------------
#  Product lattice
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class ProductLattice:
    """Product of two lattice elements."""
    left: Any
    right: Any

    def join(self, other: ProductLattice) -> ProductLattice:
        return ProductLattice(
            left=self.left.join(other.left),
            right=self.right.join(other.right),
        )

    def meet(self, other: ProductLattice) -> ProductLattice:
        return ProductLattice(
            left=self.left.meet(other.left),
            right=self.right.meet(other.right),
        )

    def leq(self, other: ProductLattice) -> bool:
        return self.left.leq(other.left) and self.right.leq(other.right)


# -------------------------------------------------------------------------
#  Map lattice
# -------------------------------------------------------------------------

@dataclass
class MapLattice(Generic[T]):
    """Map from keys to lattice elements, with pointwise operations."""
    _map: Dict[str, T] = field(default_factory=dict)
    _default_factory: Callable[[], T] = field(default=lambda: Bottom())  # type: ignore

    def get(self, key: str) -> T:
        return self._map.get(key, self._default_factory())

    def set(self, key: str, value: T) -> None:
        self._map[key] = value

    def keys(self) -> Set[str]:
        return set(self._map.keys())

    def join(self, other: MapLattice[T]) -> MapLattice[T]:
        result = MapLattice(_default_factory=self._default_factory)
        all_keys = self.keys() | other.keys()
        for k in all_keys:
            a = self.get(k)
            b = other.get(k)
            result.set(k, a.join(b))  # type: ignore
        return result

    def meet(self, other: MapLattice[T]) -> MapLattice[T]:
        result = MapLattice(_default_factory=self._default_factory)
        all_keys = self.keys() & other.keys()
        for k in all_keys:
            result.set(k, self.get(k).meet(other.get(k)))  # type: ignore
        return result

    def leq(self, other: MapLattice[T]) -> bool:
        for k in self.keys():
            if not self.get(k).leq(other.get(k)):  # type: ignore
                return False
        return True

    def copy(self) -> MapLattice[T]:
        result = MapLattice(_default_factory=self._default_factory)
        result._map = dict(self._map)
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MapLattice):
            return NotImplemented
        return self._map == other._map


# -------------------------------------------------------------------------
#  Powerset lattice
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class PowersetLattice:
    """Powerset lattice: ordered by subset inclusion."""
    elements: FrozenSet[str] = frozenset()

    @staticmethod
    def empty() -> PowersetLattice:
        return PowersetLattice(frozenset())

    @staticmethod
    def of(*items: str) -> PowersetLattice:
        return PowersetLattice(frozenset(items))

    def join(self, other: PowersetLattice) -> PowersetLattice:
        return PowersetLattice(self.elements | other.elements)

    def meet(self, other: PowersetLattice) -> PowersetLattice:
        return PowersetLattice(self.elements & other.elements)

    def leq(self, other: PowersetLattice) -> bool:
        return self.elements <= other.elements

    def add(self, item: str) -> PowersetLattice:
        return PowersetLattice(self.elements | {item})

    def remove(self, item: str) -> PowersetLattice:
        return PowersetLattice(self.elements - {item})

    def contains(self, item: str) -> bool:
        return item in self.elements

    def size(self) -> int:
        return len(self.elements)


# =========================================================================
#  Transfer function and equation system
# =========================================================================

class TransferFunction(ABC, Generic[T]):
    """Abstract transfer function: maps input abstract state to output."""

    @abstractmethod
    def apply(self, state: T, point: ProgramPoint) -> T:
        ...

    def apply_edge(self, state: T, src: str, dst: str) -> T:
        """Edge transfer function (default: identity)."""
        return state


@dataclass
class Equation(Generic[T]):
    """A single abstract equation: X_i = f(X_{deps})."""
    variable: str
    dependencies: List[str] = field(default_factory=list)
    transfer: Optional[Callable[[Dict[str, T]], T]] = None
    current_value: Optional[T] = None

    def evaluate(self, env: Dict[str, T]) -> T:
        if self.transfer is None:
            raise ValueError(f"No transfer function for equation {self.variable}")
        return self.transfer(env)


@dataclass
class EquationSystem(Generic[T]):
    """System of abstract equations X = F(X)."""
    equations: Dict[str, Equation[T]] = field(default_factory=dict)
    _dep_graph: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _rdep_graph: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_equation(self, eq: Equation[T]) -> None:
        self.equations[eq.variable] = eq
        for dep in eq.dependencies:
            self._dep_graph[eq.variable].add(dep)
            self._rdep_graph[dep].add(eq.variable)

    def dependencies(self, var: str) -> Set[str]:
        return set(self._dep_graph.get(var, set()))

    def dependents(self, var: str) -> Set[str]:
        return set(self._rdep_graph.get(var, set()))

    def variables(self) -> Set[str]:
        return set(self.equations.keys())

    def evaluate(self, var: str, env: Dict[str, T]) -> T:
        eq = self.equations.get(var)
        if eq is None:
            raise KeyError(f"Unknown equation variable: {var}")
        return eq.evaluate(env)

    def all_values(self) -> Dict[str, Optional[T]]:
        return {v: eq.current_value for v, eq in self.equations.items()}


# =========================================================================
#  MonotoneFramework
# =========================================================================

class Direction(Enum):
    FORWARD = auto()
    BACKWARD = auto()


@dataclass
class MonotoneFramework(Generic[T]):
    """Setup for a monotone dataflow framework."""
    cfg: CFG = field(default_factory=CFG)
    direction: Direction = Direction.FORWARD
    transfer: Optional[TransferFunction[T]] = None
    init_value: Optional[T] = None
    bottom: Optional[T] = None
    join_op: Optional[Callable[[T, T], T]] = None
    widening_points: Set[str] = field(default_factory=set)

    def flow_successors(self, label: str) -> List[str]:
        if self.direction == Direction.FORWARD:
            return self.cfg.successors(label)
        return self.cfg.predecessors(label)

    def flow_predecessors(self, label: str) -> List[str]:
        if self.direction == Direction.FORWARD:
            return self.cfg.predecessors(label)
        return self.cfg.successors(label)

    def entry_label(self) -> str:
        if self.direction == Direction.FORWARD:
            return self.cfg.entry
        return self.cfg.exit

    def join(self, a: T, b: T) -> T:
        if self.join_op:
            return self.join_op(a, b)
        return a.join(b)  # type: ignore


# =========================================================================
#  MeetOverAllPaths / MaximalFixedPoint
# =========================================================================

@dataclass
class MeetOverAllPaths(Generic[T]):
    """Computes the MOP solution (exact but potentially expensive)."""
    framework: MonotoneFramework[T] = field(default_factory=MonotoneFramework)
    max_path_length: int = 1000

    def solve(self) -> Dict[str, T]:
        if self.framework.init_value is None or self.framework.transfer is None:
            raise ValueError("Framework not fully configured")

        results: dict[str, T] = {}
        entry = self.framework.entry_label()
        bottom = self.framework.bottom or self.framework.init_value

        for label in self.framework.cfg.all_labels():
            results[label] = bottom  # type: ignore

        # Enumerate paths from entry to each point
        for target in self.framework.cfg.all_labels():
            paths = self._enumerate_paths(entry, target)
            if not paths:
                continue
            combined: Optional[T] = None
            for path in paths:
                state = self.framework.init_value
                for i, node in enumerate(path[:-1]):
                    point = ProgramPoint(block=node)
                    state = self.framework.transfer.apply(state, point)
                if combined is None:
                    combined = state
                else:
                    combined = self.framework.join(combined, state)
            if combined is not None:
                results[target] = combined
        return results

    def _enumerate_paths(self, start: str, end: str) -> List[List[str]]:
        if start == end:
            return [[start]]
        paths: list[list[str]] = []
        stack: list[tuple[str, list[str]]] = [(start, [start])]
        while stack and len(paths) < self.max_path_length:
            node, path = stack.pop()
            for succ in self.framework.flow_successors(node):
                if succ == end:
                    paths.append(path + [succ])
                elif succ not in path:  # avoid cycles
                    stack.append((succ, path + [succ]))
        return paths


@dataclass
class MaximalFixedPoint(Generic[T]):
    """Computes the MFP solution (standard worklist-based)."""
    framework: MonotoneFramework[T] = field(default_factory=MonotoneFramework)
    max_iterations: int = 10000

    def solve(self) -> Dict[str, T]:
        if self.framework.init_value is None or self.framework.transfer is None:
            raise ValueError("Framework not fully configured")

        state: dict[str, T] = {}
        entry = self.framework.entry_label()
        for label in self.framework.cfg.all_labels():
            state[label] = self.framework.bottom if self.framework.bottom else self.framework.init_value  # type: ignore

        state[entry] = self.framework.init_value
        worklist = deque(self.framework.cfg.all_labels())
        iterations = 0

        while worklist and iterations < self.max_iterations:
            iterations += 1
            node = worklist.popleft()
            preds = self.framework.flow_predecessors(node)
            if not preds:
                incoming = state[node]
            else:
                incoming = state[preds[0]]
                for pred in preds[1:]:
                    incoming = self.framework.join(incoming, state[pred])

            point = ProgramPoint(block=node)
            new_state = self.framework.transfer.apply(incoming, point)

            old = state[node]
            if not _leq(new_state, old):
                state[node] = self.framework.join(old, new_state)
                for succ in self.framework.flow_successors(node):
                    if succ not in worklist:
                        worklist.append(succ)

        return state


def _leq(a: Any, b: Any) -> bool:
    if hasattr(a, "leq"):
        return a.leq(b)
    return a == b


# =========================================================================
#  Loop analysis
# =========================================================================

@dataclass
class LoopInfo:
    """Information about a natural loop."""
    header: str
    body: Set[str] = field(default_factory=set)
    back_edges: List[Tuple[str, str]] = field(default_factory=list)
    exit_edges: List[Tuple[str, str]] = field(default_factory=list)
    nesting_depth: int = 0
    parent: Optional[str] = None
    is_countable: bool = False
    bound: Optional[int] = None


@dataclass
class LoopAnalyzer:
    """Identifies loops, computes loop nesting, determines widening points."""

    _loops: Dict[str, LoopInfo] = field(default_factory=dict)
    _widening_points: Set[str] = field(default_factory=set)

    def analyze(self, cfg: CFG) -> Dict[str, LoopInfo]:
        self._loops.clear()
        self._widening_points.clear()

        dominators = self._compute_dominators(cfg)
        back_edges = self._find_back_edges(cfg, dominators)

        for src, header in back_edges:
            if header in self._loops:
                self._loops[header].back_edges.append((src, header))
                body = self._compute_loop_body(cfg, header, src)
                self._loops[header].body |= body
            else:
                body = self._compute_loop_body(cfg, header, src)
                self._loops[header] = LoopInfo(
                    header=header,
                    body=body,
                    back_edges=[(src, header)],
                )
            self._widening_points.add(header)

        self._compute_nesting()
        self._find_exit_edges(cfg)
        return dict(self._loops)

    def _compute_dominators(self, cfg: CFG) -> Dict[str, Set[str]]:
        labels = cfg.all_labels()
        dom: dict[str, set[str]] = {l: set(labels) for l in labels}
        dom[cfg.entry] = {cfg.entry}
        changed = True
        while changed:
            changed = False
            for label in labels:
                if label == cfg.entry:
                    continue
                preds = cfg.predecessors(label)
                if not preds:
                    new_dom = {label}
                else:
                    new_dom = set(labels)
                    for p in preds:
                        new_dom &= dom[p]
                    new_dom.add(label)
                if new_dom != dom[label]:
                    dom[label] = new_dom
                    changed = True
        return dom

    def _find_back_edges(
        self, cfg: CFG, dominators: Dict[str, Set[str]]
    ) -> List[Tuple[str, str]]:
        back_edges: list[tuple[str, str]] = []
        for label in cfg.all_labels():
            for succ in cfg.successors(label):
                if succ in dominators.get(label, set()):
                    back_edges.append((label, succ))
        return back_edges

    def _compute_loop_body(self, cfg: CFG, header: str, back_edge_src: str) -> Set[str]:
        body = {header, back_edge_src}
        if header == back_edge_src:
            return body
        stack = [back_edge_src]
        while stack:
            node = stack.pop()
            for pred in cfg.predecessors(node):
                if pred not in body:
                    body.add(pred)
                    stack.append(pred)
        return body

    def _compute_nesting(self) -> None:
        headers = list(self._loops.keys())
        for h1 in headers:
            for h2 in headers:
                if h1 != h2 and self._loops[h1].body < self._loops[h2].body:
                    self._loops[h1].nesting_depth = max(
                        self._loops[h1].nesting_depth,
                        self._loops[h2].nesting_depth + 1,
                    )
                    if self._loops[h1].parent is None:
                        self._loops[h1].parent = h2

    def _find_exit_edges(self, cfg: CFG) -> None:
        for header, loop in self._loops.items():
            for node in loop.body:
                for succ in cfg.successors(node):
                    if succ not in loop.body:
                        loop.exit_edges.append((node, succ))

    def widening_points(self) -> Set[str]:
        return set(self._widening_points)

    def get_loop(self, header: str) -> Optional[LoopInfo]:
        return self._loops.get(header)

    def is_loop_header(self, label: str) -> bool:
        return label in self._loops

    def innermost_loop(self, label: str) -> Optional[LoopInfo]:
        best: Optional[LoopInfo] = None
        for loop in self._loops.values():
            if label in loop.body:
                if best is None or len(loop.body) < len(best.body):
                    best = loop
        return best

    def loop_depth(self, label: str) -> int:
        depth = 0
        for loop in self._loops.values():
            if label in loop.body:
                depth += 1
        return depth


# =========================================================================
#  LoopInvariantInference
# =========================================================================

@dataclass
class LoopInvariantInference:
    """Infers loop invariants via abstract interpretation."""

    _invariants: Dict[str, List[str]] = field(default_factory=dict)

    def infer(
        self,
        loop: LoopInfo,
        pre_state: MapLattice,
        post_state: MapLattice,
    ) -> List[str]:
        invariants: list[str] = []
        for var in pre_state.keys() & post_state.keys():
            pre_val = pre_state.get(var)
            post_val = post_state.get(var)
            if hasattr(pre_val, "leq") and pre_val.leq(post_val) and hasattr(post_val, "leq") and post_val.leq(pre_val):  # type: ignore
                invariants.append(f"{var} is loop-invariant")
            elif isinstance(pre_val, Interval) and isinstance(post_val, Interval):
                merged = pre_val.join(post_val)
                if merged.lo is not None:
                    invariants.append(f"{var} >= {merged.lo}")
                if merged.hi is not None:
                    invariants.append(f"{var} <= {merged.hi}")
        self._invariants[loop.header] = invariants
        return invariants

    def get_invariants(self, header: str) -> List[str]:
        return list(self._invariants.get(header, []))


# =========================================================================
#  LoopUnrolling
# =========================================================================

@dataclass
class LoopUnrolling:
    """Symbolic loop unrolling for bounded loops."""

    max_unroll: int = 8

    def can_unroll(self, loop: LoopInfo) -> bool:
        return loop.is_countable and loop.bound is not None and loop.bound <= self.max_unroll

    def unroll(self, loop: LoopInfo, cfg: CFG) -> List[Dict[str, Any]]:
        if not self.can_unroll(loop):
            return []
        iterations: list[dict[str, Any]] = []
        bound = loop.bound if loop.bound is not None else 0
        for i in range(bound):
            iteration_state: dict[str, Any] = {"iteration": i, "blocks": list(loop.body)}
            iterations.append(iteration_state)
        return iterations


# =========================================================================
#  AccelerationEngine
# =========================================================================

@dataclass
class AccelerationEngine:
    """Loop acceleration for special loop patterns (linear, polynomial)."""

    def can_accelerate(self, loop: LoopInfo, transfer_fn: Optional[TransferFunction] = None) -> bool:
        return loop.is_countable and len(loop.body) <= 3

    def accelerate_linear(
        self,
        init: Interval,
        step: int,
        bound: int,
    ) -> Interval:
        if init.is_bottom():
            return Interval.bottom()
        lo = init.lo
        hi = init.hi
        if step > 0:
            final_lo = lo
            final_hi = (hi + step * bound) if hi is not None else None
        elif step < 0:
            final_lo = (lo + step * bound) if lo is not None else None
            final_hi = hi
        else:
            final_lo, final_hi = lo, hi
        return Interval(lo=final_lo, hi=final_hi)

    def accelerate_polynomial(
        self,
        init: Interval,
        coefficients: List[int],
        bound: int,
    ) -> Interval:
        if not coefficients or init.is_bottom():
            return init
        values: list[int] = []
        for i in range(bound + 1):
            val = sum(c * (i ** p) for p, c in enumerate(coefficients))
            if init.lo is not None:
                val += init.lo
            values.append(val)
        if values:
            return Interval(lo=min(values), hi=max(values))
        return init


# =========================================================================
#  Widening and narrowing strategies
# =========================================================================

@dataclass
class WideningStrategy:
    """Configurable widening: after k iterations, at loop heads, with thresholds."""
    delay: int = 0
    thresholds: List[int] = field(default_factory=list)
    only_at_loop_heads: bool = True
    _iteration_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def should_widen(self, point: str) -> bool:
        return self._iteration_counts[point] >= self.delay

    def widen(self, old: Any, new: Any, point: str) -> Any:
        self._iteration_counts[point] += 1
        if not self.should_widen(point):
            return old.join(new) if hasattr(old, "join") else new

        if self.thresholds and isinstance(old, Interval) and isinstance(new, Interval):
            return self._threshold_widen(old, new)

        if hasattr(old, "widen"):
            return old.widen(new)
        return old.join(new) if hasattr(old, "join") else new

    def _threshold_widen(self, old: Interval, new: Interval) -> Interval:
        lo = old.lo
        hi = old.hi
        if new.lo is not None and (old.lo is None or new.lo < old.lo):
            candidates = [t for t in self.thresholds if new.lo is not None and t <= new.lo]
            lo = max(candidates) if candidates else None
        if new.hi is not None and (old.hi is None or new.hi > old.hi):
            candidates = [t for t in self.thresholds if new.hi is not None and t >= new.hi]
            hi = min(candidates) if candidates else None
        return Interval(lo=lo, hi=hi)

    def reset(self) -> None:
        self._iteration_counts.clear()


@dataclass
class NarrowingStrategy:
    """Narrowing phase after widening reaches fixed point."""
    max_iterations: int = 5
    _iterations: int = 0

    def narrow(self, old: Any, new: Any) -> Any:
        self._iterations += 1
        if hasattr(old, "narrow"):
            return old.narrow(new)
        return old.meet(new) if hasattr(old, "meet") else new

    def should_continue(self) -> bool:
        return self._iterations < self.max_iterations

    def reset(self) -> None:
        self._iterations = 0


@dataclass
class ThresholdWidening:
    """Widening with thresholds from program constants."""
    thresholds: List[int] = field(default_factory=list)

    def collect_thresholds(self, cfg: CFG) -> List[int]:
        constants: set[int] = {0, 1, -1}
        for bb in cfg.blocks.values():
            for instr in bb.instructions:
                if hasattr(instr, "operands"):
                    for op in instr.operands:
                        if isinstance(op, int):
                            constants.add(op)
                            constants.add(op - 1)
                            constants.add(op + 1)
        self.thresholds = sorted(constants)
        return self.thresholds

    def widen(self, old: Interval, new: Interval) -> Interval:
        if old.is_bottom():
            return new
        lo = old.lo
        hi = old.hi
        if new.lo is not None and (old.lo is None or new.lo < old.lo):
            candidates = [t for t in self.thresholds if new.lo is not None and t <= new.lo]
            lo = max(candidates) if candidates else None
        if new.hi is not None and (old.hi is None or new.hi > old.hi):
            candidates = [t for t in self.thresholds if new.hi is not None and t >= new.hi]
            hi = min(candidates) if candidates else None
        return Interval(lo=lo, hi=hi)


@dataclass
class DelayedWidening:
    """Delay widening for k iterations before applying."""
    delay: int = 3
    _counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _inner: Optional[WideningStrategy] = None

    def widen(self, old: Any, new: Any, point: str) -> Any:
        self._counts[point] += 1
        if self._counts[point] < self.delay:
            return old.join(new) if hasattr(old, "join") else new
        if self._inner:
            return self._inner.widen(old, new, point)
        if hasattr(old, "widen"):
            return old.widen(new)
        return old.join(new) if hasattr(old, "join") else new

    def reset(self) -> None:
        self._counts.clear()


# =========================================================================
#  Fixed-point solvers
# =========================================================================

@dataclass
class FixpointStatistics:
    """Tracks iteration counts, convergence rate."""
    iterations: int = 0
    equations_evaluated: int = 0
    widening_applied: int = 0
    narrowing_applied: int = 0
    start_time: float = 0.0
    elapsed: float = 0.0
    converged: bool = False

    def start(self) -> None:
        self.start_time = time.time()

    def stop(self) -> None:
        self.elapsed = time.time() - self.start_time

    def record_iteration(self) -> None:
        self.iterations += 1

    def record_evaluation(self) -> None:
        self.equations_evaluated += 1

    def __str__(self) -> str:
        return (
            f"FixpointStats(iters={self.iterations}, evals={self.equations_evaluated}, "
            f"widen={self.widening_applied}, narrow={self.narrowing_applied}, "
            f"elapsed={self.elapsed:.3f}s, converged={self.converged})"
        )


@dataclass
class FixpointSolver(Generic[T]):
    """Generic fixed-point solver parameterized by domain."""
    max_iterations: int = 10000
    statistics: FixpointStatistics = field(default_factory=FixpointStatistics)

    def solve(
        self,
        system: EquationSystem[T],
        initial: Dict[str, T],
    ) -> Dict[str, T]:
        self.statistics.start()
        env = dict(initial)
        for var in system.variables():
            env.setdefault(var, system.equations[var].current_value)  # type: ignore

        changed = True
        while changed and self.statistics.iterations < self.max_iterations:
            changed = False
            self.statistics.record_iteration()
            for var in system.variables():
                self.statistics.record_evaluation()
                new_val = system.evaluate(var, env)
                old_val = env.get(var)
                if old_val is None or not _leq(new_val, old_val):
                    env[var] = new_val if old_val is None else _join(old_val, new_val)
                    changed = True

        self.statistics.converged = not changed
        self.statistics.stop()
        return env


def _join(a: Any, b: Any) -> Any:
    if hasattr(a, "join"):
        return a.join(b)
    return b


@dataclass
class ChaoticIteration(Generic[T]):
    """Chaotic iteration strategy for solving dataflow equations."""
    max_iterations: int = 10000
    statistics: FixpointStatistics = field(default_factory=FixpointStatistics)

    def solve(
        self,
        system: EquationSystem[T],
        initial: Dict[str, T],
    ) -> Dict[str, T]:
        self.statistics.start()
        env = dict(initial)
        for var in system.variables():
            if var not in env and system.equations[var].current_value is not None:
                env[var] = system.equations[var].current_value  # type: ignore

        changed = True
        while changed and self.statistics.iterations < self.max_iterations:
            changed = False
            self.statistics.record_iteration()
            for var in system.variables():
                self.statistics.record_evaluation()
                new_val = system.evaluate(var, env)
                old_val = env.get(var)
                if old_val is None or not _leq(new_val, old_val):
                    env[var] = _join(old_val, new_val) if old_val is not None else new_val
                    changed = True

        self.statistics.converged = not changed
        self.statistics.stop()
        return env


@dataclass
class WorklistSolver(Generic[T]):
    """Efficient worklist-based iteration with priority scheduling."""
    max_iterations: int = 10000
    statistics: FixpointStatistics = field(default_factory=FixpointStatistics)
    use_priority: bool = True

    def solve(
        self,
        system: EquationSystem[T],
        initial: Dict[str, T],
        priority: Optional[Dict[str, int]] = None,
    ) -> Dict[str, T]:
        self.statistics.start()
        env = dict(initial)
        for var in system.variables():
            if var not in env and system.equations[var].current_value is not None:
                env[var] = system.equations[var].current_value  # type: ignore

        priority = priority or {}
        counter = 0

        if self.use_priority:
            heap: list[tuple[int, int, str]] = []
            in_wl: set[str] = set()
            for var in system.variables():
                counter += 1
                heapq.heappush(heap, (priority.get(var, 0), counter, var))
                in_wl.add(var)

            while heap and self.statistics.iterations < self.max_iterations:
                _, _, var = heapq.heappop(heap)
                if var not in in_wl:
                    continue
                in_wl.discard(var)
                self.statistics.record_iteration()
                self.statistics.record_evaluation()

                new_val = system.evaluate(var, env)
                old_val = env.get(var)
                if old_val is None or not _leq(new_val, old_val):
                    env[var] = _join(old_val, new_val) if old_val is not None else new_val
                    for dep in system.dependents(var):
                        if dep not in in_wl:
                            in_wl.add(dep)
                            counter += 1
                            heapq.heappush(heap, (priority.get(dep, 0), counter, dep))
        else:
            worklist = deque(system.variables())
            in_wl = set(system.variables())

            while worklist and self.statistics.iterations < self.max_iterations:
                var = worklist.popleft()
                in_wl.discard(var)
                self.statistics.record_iteration()
                self.statistics.record_evaluation()

                new_val = system.evaluate(var, env)
                old_val = env.get(var)
                if old_val is None or not _leq(new_val, old_val):
                    env[var] = _join(old_val, new_val) if old_val is not None else new_val
                    for dep in system.dependents(var):
                        if dep not in in_wl:
                            in_wl.add(dep)
                            worklist.append(dep)

        self.statistics.converged = True
        self.statistics.stop()
        return env


@dataclass
class RoundRobinSolver(Generic[T]):
    """Round-robin iteration strategy."""
    max_iterations: int = 10000
    statistics: FixpointStatistics = field(default_factory=FixpointStatistics)

    def solve(
        self,
        system: EquationSystem[T],
        initial: Dict[str, T],
        order: Optional[List[str]] = None,
    ) -> Dict[str, T]:
        self.statistics.start()
        env = dict(initial)
        variables = order or sorted(system.variables())

        for var in variables:
            if var not in env and system.equations[var].current_value is not None:
                env[var] = system.equations[var].current_value  # type: ignore

        changed = True
        while changed and self.statistics.iterations < self.max_iterations:
            changed = False
            self.statistics.record_iteration()
            for var in variables:
                self.statistics.record_evaluation()
                new_val = system.evaluate(var, env)
                old_val = env.get(var)
                if old_val is None or not _leq(new_val, old_val):
                    env[var] = _join(old_val, new_val) if old_val is not None else new_val
                    changed = True

        self.statistics.converged = not changed
        self.statistics.stop()
        return env


@dataclass
class WideningSolver(Generic[T]):
    """Fixed-point with widening and optional narrowing."""
    max_iterations: int = 10000
    widening: WideningStrategy = field(default_factory=WideningStrategy)
    narrowing: Optional[NarrowingStrategy] = None
    statistics: FixpointStatistics = field(default_factory=FixpointStatistics)
    widening_points: Set[str] = field(default_factory=set)

    def solve(
        self,
        system: EquationSystem[T],
        initial: Dict[str, T],
    ) -> Dict[str, T]:
        self.statistics.start()
        env = dict(initial)
        for var in system.variables():
            if var not in env and system.equations[var].current_value is not None:
                env[var] = system.equations[var].current_value  # type: ignore

        # Ascending phase with widening
        changed = True
        while changed and self.statistics.iterations < self.max_iterations:
            changed = False
            self.statistics.record_iteration()
            for var in system.variables():
                self.statistics.record_evaluation()
                new_val = system.evaluate(var, env)
                old_val = env.get(var)
                if old_val is None:
                    env[var] = new_val
                    changed = True
                elif not _leq(new_val, old_val):
                    if var in self.widening_points or not self.widening_points:
                        env[var] = self.widening.widen(old_val, new_val, var)
                        self.statistics.widening_applied += 1
                    else:
                        env[var] = _join(old_val, new_val)
                    changed = True

        # Descending phase with narrowing
        if self.narrowing is not None:
            self.narrowing.reset()
            while self.narrowing.should_continue():
                self.statistics.record_iteration()
                stable = True
                for var in system.variables():
                    self.statistics.record_evaluation()
                    new_val = system.evaluate(var, env)
                    old_val = env.get(var)
                    if old_val is not None:
                        narrowed = self.narrowing.narrow(old_val, new_val)
                        if narrowed != old_val:
                            env[var] = narrowed
                            stable = False
                            self.statistics.narrowing_applied += 1
                if stable:
                    break

        self.statistics.converged = True
        self.statistics.stop()
        return env


# =========================================================================
#  Stratified and Semi-Naive solvers
# =========================================================================

@dataclass
class StratifiedSolver(Generic[T]):
    """Stratified fixed-point for negation handling."""
    strata: List[Set[str]] = field(default_factory=list)
    inner_solver: FixpointSolver[T] = field(default_factory=FixpointSolver)

    def solve(
        self,
        system: EquationSystem[T],
        initial: Dict[str, T],
    ) -> Dict[str, T]:
        env = dict(initial)
        for stratum in self.strata:
            sub_system = EquationSystem[T]()
            for var in stratum:
                if var in system.equations:
                    sub_system.add_equation(system.equations[var])
            result = self.inner_solver.solve(sub_system, env)
            env.update(result)
        return env


@dataclass
class SemiNaiveSolver(Generic[T]):
    """Semi-naive evaluation for incremental updates."""
    max_iterations: int = 10000
    statistics: FixpointStatistics = field(default_factory=FixpointStatistics)

    def solve(
        self,
        system: EquationSystem[T],
        initial: Dict[str, T],
    ) -> Dict[str, T]:
        self.statistics.start()
        env = dict(initial)
        delta: dict[str, T] = dict(initial)

        changed = True
        while changed and self.statistics.iterations < self.max_iterations:
            changed = False
            self.statistics.record_iteration()
            new_delta: dict[str, T] = {}
            for var in system.variables():
                if not any(dep in delta for dep in system.dependencies(var)):
                    continue
                self.statistics.record_evaluation()
                new_val = system.evaluate(var, env)
                old_val = env.get(var)
                if old_val is None or not _leq(new_val, old_val):
                    joined = _join(old_val, new_val) if old_val is not None else new_val
                    new_delta[var] = joined
                    env[var] = joined
                    changed = True
            delta = new_delta

        self.statistics.converged = not changed
        self.statistics.stop()
        return env


# =========================================================================
#  Incremental fixed-point
# =========================================================================

@dataclass
class IncrementalFixpoint(Generic[T]):
    """Maintains fixed point under program changes."""
    _solution: Dict[str, T] = field(default_factory=dict)
    _system: Optional[EquationSystem[T]] = None
    _solver: WorklistSolver[T] = field(default_factory=WorklistSolver)

    def initialize(self, system: EquationSystem[T], initial: Dict[str, T]) -> Dict[str, T]:
        self._system = system
        self._solution = self._solver.solve(system, initial)
        return dict(self._solution)

    def update(self, changed_vars: Set[str]) -> Dict[str, T]:
        if self._system is None:
            raise ValueError("Not initialized")
        affected: set[str] = set()
        stack = list(changed_vars)
        while stack:
            var = stack.pop()
            if var in affected:
                continue
            affected.add(var)
            stack.extend(self._system.dependents(var))

        sub_system = EquationSystem[T]()
        for var in affected:
            if var in self._system.equations:
                sub_system.add_equation(self._system.equations[var])

        result = self._solver.solve(sub_system, self._solution)
        self._solution.update(result)
        return dict(self._solution)

    def current_solution(self) -> Dict[str, T]:
        return dict(self._solution)


# =========================================================================
#  DeltaPropagation
# =========================================================================

@dataclass
class DeltaPropagation(Generic[T]):
    """Propagates only changed facts."""
    _previous: Dict[str, T] = field(default_factory=dict)

    def compute_delta(self, current: Dict[str, T]) -> Dict[str, T]:
        delta: dict[str, T] = {}
        for var, val in current.items():
            prev = self._previous.get(var)
            if prev is None or not _leq(val, prev):
                delta[var] = val
        self._previous = dict(current)
        return delta

    def apply_delta(self, base: Dict[str, T], delta: Dict[str, T]) -> Dict[str, T]:
        result = dict(base)
        for var, val in delta.items():
            old = result.get(var)
            result[var] = _join(old, val) if old is not None else val
        return result


# =========================================================================
#  AbstractEquationGraph
# =========================================================================

@dataclass
class AbstractEquationGraph:
    """Dependency graph for equations."""
    _adj: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _radj: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _nodes: Set[str] = field(default_factory=set)

    def add_node(self, node: str) -> None:
        self._nodes.add(node)

    def add_edge(self, src: str, dst: str) -> None:
        self._nodes.add(src)
        self._nodes.add(dst)
        self._adj[src].add(dst)
        self._radj[dst].add(src)

    def successors(self, node: str) -> Set[str]:
        return set(self._adj.get(node, set()))

    def predecessors(self, node: str) -> Set[str]:
        return set(self._radj.get(node, set()))

    def nodes(self) -> Set[str]:
        return set(self._nodes)

    @staticmethod
    def from_system(system: EquationSystem) -> AbstractEquationGraph:
        g = AbstractEquationGraph()
        for var in system.variables():
            g.add_node(var)
            for dep in system.dependencies(var):
                g.add_edge(dep, var)
        return g


# =========================================================================
#  StrongComponentDecomposition (Tarjan's)
# =========================================================================

@dataclass
class StrongComponentDecomposition:
    """Tarjan's algorithm for SCC computation."""

    def compute(self, graph: AbstractEquationGraph) -> List[List[str]]:
        index_counter = [0]
        stack: list[str] = []
        on_stack: set[str] = set()
        index_map: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        result: list[list[str]] = []

        def strongconnect(v: str) -> None:
            index_map[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in graph.successors(v):
                if w not in index_map:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index_map[w])

            if lowlink[v] == index_map[v]:
                component: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == v:
                        break
                result.append(component)

        for node in graph.nodes():
            if node not in index_map:
                strongconnect(node)

        return result


# =========================================================================
#  TopologicalSorter
# =========================================================================

@dataclass
class TopologicalSorter:
    """Topological ordering for acyclic portions."""

    def sort(self, graph: AbstractEquationGraph) -> List[str]:
        in_degree: dict[str, int] = {n: 0 for n in graph.nodes()}
        for n in graph.nodes():
            for s in graph.successors(n):
                in_degree[s] = in_degree.get(s, 0) + 1
        queue = deque(n for n in graph.nodes() if in_degree.get(n, 0) == 0)
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for succ in graph.successors(node):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        return order

    def reverse_postorder(self, graph: AbstractEquationGraph, start: Optional[str] = None) -> List[str]:
        visited: set[str] = set()
        order: list[str] = []

        def dfs(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            for succ in graph.successors(node):
                dfs(succ)
            order.append(node)

        if start:
            dfs(start)
        else:
            for node in graph.nodes():
                dfs(node)

        order.reverse()
        return order


# =========================================================================
#  ConvergenceAccelerator
# =========================================================================

@dataclass
class ConvergenceAccelerator(Generic[T]):
    """Techniques for faster convergence (extrapolation)."""
    history_size: int = 3
    _history: Dict[str, List[T]] = field(default_factory=lambda: defaultdict(list))

    def record(self, var: str, value: T) -> None:
        self._history[var].append(value)
        if len(self._history[var]) > self.history_size:
            self._history[var] = self._history[var][-self.history_size:]

    def extrapolate(self, var: str, current: T) -> T:
        """Try to predict the fixed point from the history."""
        history = self._history.get(var, [])
        if len(history) < 2:
            return current

        if isinstance(current, Interval) and all(isinstance(h, Interval) for h in history):
            return self._extrapolate_interval(history, current)  # type: ignore
        return current

    def _extrapolate_interval(self, history: List[Interval], current: Interval) -> Interval:
        lows = [h.lo for h in history if h.lo is not None]
        highs = [h.hi for h in history if h.hi is not None]

        if len(lows) >= 2:
            diffs = [lows[i + 1] - lows[i] for i in range(len(lows) - 1)]
            if all(d == diffs[0] for d in diffs) and diffs[0] < 0:
                predicted_lo = None  # trending to -∞
            else:
                predicted_lo = current.lo
        else:
            predicted_lo = current.lo

        if len(highs) >= 2:
            diffs = [highs[i + 1] - highs[i] for i in range(len(highs) - 1)]
            if all(d == diffs[0] for d in diffs) and diffs[0] > 0:
                predicted_hi = None  # trending to +∞
            else:
                predicted_hi = current.hi
        else:
            predicted_hi = current.hi

        return Interval(lo=predicted_lo, hi=predicted_hi)

    def reset(self) -> None:
        self._history.clear()

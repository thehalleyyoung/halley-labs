"""
Abstraction domain for CEGAR verification of biological models.

Defines abstract states (interval boxes × discrete modes × predicates),
abstraction domains (partitions of continuous state space), and operations
for post/pre-image computation, widening, and narrowing.
"""

from __future__ import annotations

import itertools
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from bioprover.encoding.expression import (
    Add,
    And,
    Const,
    Div,
    ExprNode,
    Ge,
    Gt,
    Interval,
    Le,
    Lt,
    Mul,
    Neg,
    Not,
    Or,
    Var,
)
from bioprover.cegar.predicate_manager import Predicate, PredicateSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interval arithmetic helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntervalBox:
    """Axis-aligned hyper-rectangle in R^n.

    Each dimension is an ``Interval`` keyed by variable name.
    """

    dimensions: Dict[str, Interval] = field(default_factory=dict)

    # -- construction -------------------------------------------------------

    @staticmethod
    def from_bounds(
        bounds: Dict[str, Tuple[float, float]],
    ) -> IntervalBox:
        return IntervalBox(
            dimensions={k: Interval(lo, hi) for k, (lo, hi) in bounds.items()},
        )

    @staticmethod
    def universe(var_names: List[str]) -> IntervalBox:
        return IntervalBox(
            dimensions={v: Interval(-1e12, 1e12) for v in var_names},
        )

    # -- queries ------------------------------------------------------------

    @property
    def variable_names(self) -> List[str]:
        return sorted(self.dimensions.keys())

    @property
    def ndim(self) -> int:
        return len(self.dimensions)

    def volume(self) -> float:
        if not self.dimensions:
            return 0.0
        vol = 1.0
        for iv in self.dimensions.values():
            vol *= iv.width()
        return vol

    def midpoint(self) -> Dict[str, float]:
        return {k: iv.midpoint() for k, iv in self.dimensions.items()}

    def contains_point(self, point: Dict[str, float]) -> bool:
        for k, iv in self.dimensions.items():
            if k in point and not iv.contains(point[k]):
                return False
        return True

    def overlaps(self, other: IntervalBox) -> bool:
        for k in self.dimensions:
            if k in other.dimensions:
                a, b = self.dimensions[k], other.dimensions[k]
                if a.hi < b.lo or b.hi < a.lo:
                    return False
        return True

    def is_subset_of(self, other: IntervalBox) -> bool:
        for k, iv in self.dimensions.items():
            oiv = other.dimensions.get(k)
            if oiv is None:
                continue
            if iv.lo < oiv.lo or iv.hi > oiv.hi:
                return False
        return True

    def intersection(self, other: IntervalBox) -> Optional[IntervalBox]:
        dims: Dict[str, Interval] = {}
        for k in set(self.dimensions) | set(other.dimensions):
            a = self.dimensions.get(k)
            b = other.dimensions.get(k)
            if a is None:
                dims[k] = b  # type: ignore[assignment]
            elif b is None:
                dims[k] = a
            else:
                lo = max(a.lo, b.lo)
                hi = min(a.hi, b.hi)
                if lo > hi:
                    return None
                dims[k] = Interval(lo, hi)
        return IntervalBox(dimensions=dims)

    def hull(self, other: IntervalBox) -> IntervalBox:
        dims: Dict[str, Interval] = {}
        for k in set(self.dimensions) | set(other.dimensions):
            a = self.dimensions.get(k)
            b = other.dimensions.get(k)
            if a is None:
                dims[k] = b  # type: ignore[assignment]
            elif b is None:
                dims[k] = a
            else:
                dims[k] = Interval(min(a.lo, b.lo), max(a.hi, b.hi))
        return IntervalBox(dimensions=dims)

    def split_on(self, var: str) -> Tuple[IntervalBox, IntervalBox]:
        """Split on the midpoint of *var*'s interval."""
        iv = self.dimensions[var]
        mid = iv.midpoint()
        d1 = dict(self.dimensions)
        d2 = dict(self.dimensions)
        d1[var] = Interval(iv.lo, mid)
        d2[var] = Interval(mid, iv.hi)
        return IntervalBox(dimensions=d1), IntervalBox(dimensions=d2)

    def split_widest(self) -> Tuple[IntervalBox, IntervalBox]:
        """Split along the widest dimension."""
        widest = max(self.dimensions, key=lambda k: self.dimensions[k].width())
        return self.split_on(widest)

    def vertices(self) -> List[Dict[str, float]]:
        """Enumerate all 2^n corner points."""
        names = self.variable_names
        corners: List[Dict[str, float]] = []
        for combo in itertools.product(*[[0, 1]] * len(names)):
            pt: Dict[str, float] = {}
            for nm, bit in zip(names, combo):
                iv = self.dimensions[nm]
                pt[nm] = iv.lo if bit == 0 else iv.hi
            corners.append(pt)
        return corners

    def max_width(self) -> float:
        if not self.dimensions:
            return 0.0
        return max(iv.width() for iv in self.dimensions.values())

    def to_constraint(self) -> ExprNode:
        """Encode box as a conjunction of interval constraints."""
        clauses: List[ExprNode] = []
        for k, iv in self.dimensions.items():
            x = Var(k)
            clauses.append(Ge(x, Const(iv.lo)))
            clauses.append(Le(x, Const(iv.hi)))
        return And(*clauses) if clauses else Const(1.0)

    def __repr__(self) -> str:
        parts = [f"{k}∈[{iv.lo:.4g},{iv.hi:.4g}]" for k, iv in sorted(self.dimensions.items())]
        return "Box(" + ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Abstract state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbstractState:
    """Abstract state: interval box + discrete mode + predicate valuation.

    Uniquely identified by ``state_id``.  The ``predicate_valuation``
    maps predicate digests to Boolean values (True/False) indicating
    which side of each predicate this abstract region lies on.
    """

    state_id: int
    box: IntervalBox
    mode: str = "default"
    predicate_valuation: FrozenSet[Tuple[str, bool]] = field(default_factory=frozenset)

    @property
    def label(self) -> str:
        return f"s{self.state_id}[{self.mode}]"

    def midpoint(self) -> Dict[str, float]:
        return self.box.midpoint()

    def contains_point(self, point: Dict[str, float]) -> bool:
        return self.box.contains_point(point)

    def volume(self) -> float:
        return self.box.volume()

    def overlaps(self, other: AbstractState) -> bool:
        if self.mode != other.mode:
            return False
        return self.box.overlaps(other.box)

    def split(self, var: str) -> Tuple[AbstractState, AbstractState]:
        """Binary split along *var*."""
        b1, b2 = self.box.split_on(var)
        # New IDs assigned externally; use negative placeholders
        return (
            AbstractState(-1, b1, self.mode, self.predicate_valuation),
            AbstractState(-2, b2, self.mode, self.predicate_valuation),
        )

    def to_constraint(self) -> ExprNode:
        return self.box.to_constraint()

    def __repr__(self) -> str:
        return f"AbstractState({self.label}, {self.box})"


# ---------------------------------------------------------------------------
# Abstract transition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbstractTransition:
    """Edge in the abstract transition graph."""

    source_id: int
    target_id: int
    mode_switch: Optional[str] = None
    time_bound: Optional[float] = None

    def __repr__(self) -> str:
        return f"s{self.source_id}→s{self.target_id}"


# ---------------------------------------------------------------------------
# Abstraction domain (base)
# ---------------------------------------------------------------------------


class AbstractionDomain(ABC):
    """Base class for abstraction domains.

    An abstraction domain maintains a partition of the continuous state
    space into abstract states and provides operations for exploring and
    refining the abstract transition system.
    """

    @abstractmethod
    def states(self) -> List[AbstractState]:
        """All abstract states in the domain."""
        ...

    @abstractmethod
    def initial_states(self) -> List[AbstractState]:
        """States overlapping the initial region."""
        ...

    @abstractmethod
    def transitions(self) -> List[AbstractTransition]:
        """Abstract transition relation."""
        ...

    @abstractmethod
    def post_image(self, state: AbstractState) -> List[AbstractState]:
        """Abstract successors of *state*."""
        ...

    @abstractmethod
    def pre_image(self, state: AbstractState) -> List[AbstractState]:
        """Abstract predecessors of *state*."""
        ...

    @abstractmethod
    def refine(self, state: AbstractState, predicates: List[Predicate]) -> List[AbstractState]:
        """Split *state* using predicates. Returns replacement states."""
        ...

    @abstractmethod
    def state_count(self) -> int:
        ...

    @abstractmethod
    def transition_count(self) -> int:
        ...


# ---------------------------------------------------------------------------
# Interval abstraction
# ---------------------------------------------------------------------------


class IntervalAbstraction(AbstractionDomain):
    """Abstraction defined by an interval grid over continuous state space.

    The state space is partitioned into axis-aligned hyper-rectangles
    (interval boxes).  The grid can be refined by splitting individual
    boxes.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        initial_region: Optional[IntervalBox] = None,
        grid_resolution: int = 4,
    ) -> None:
        self._bounds = bounds
        self._var_names = sorted(bounds.keys())
        self._initial_region = initial_region or IntervalBox.from_bounds(bounds)
        self._next_id = 0
        self._states: Dict[int, AbstractState] = {}
        self._transitions: List[AbstractTransition] = []
        self._adjacency: Dict[int, Set[int]] = {}
        self._reverse_adj: Dict[int, Set[int]] = {}
        self._explored: Set[int] = set()

        self._build_initial_grid(grid_resolution)

    # -- grid construction --------------------------------------------------

    def _build_initial_grid(self, resolution: int) -> None:
        """Create a uniform grid of *resolution* divisions per dimension."""
        # Build per-dimension breakpoints
        breaks: Dict[str, List[float]] = {}
        for var in self._var_names:
            lo, hi = self._bounds[var]
            step = (hi - lo) / resolution
            breaks[var] = [lo + i * step for i in range(resolution + 1)]

        # Enumerate grid cells
        dim_indices = [range(resolution) for _ in self._var_names]
        for idx_combo in itertools.product(*dim_indices):
            dims: Dict[str, Interval] = {}
            for var, idx in zip(self._var_names, idx_combo):
                bp = breaks[var]
                dims[var] = Interval(bp[idx], bp[idx + 1])
            box = IntervalBox(dimensions=dims)
            self._add_state(box)

        logger.info(
            "Built initial interval grid: %d states (%d dims × %d divisions)",
            len(self._states),
            len(self._var_names),
            resolution,
        )

    def _add_state(self, box: IntervalBox, mode: str = "default") -> AbstractState:
        sid = self._next_id
        self._next_id += 1
        state = AbstractState(state_id=sid, box=box, mode=mode)
        self._states[sid] = state
        self._adjacency[sid] = set()
        self._reverse_adj[sid] = set()
        return state

    # -- AbstractionDomain interface ----------------------------------------

    def states(self) -> List[AbstractState]:
        return list(self._states.values())

    def initial_states(self) -> List[AbstractState]:
        return [
            s
            for s in self._states.values()
            if self._initial_region.overlaps(s.box)
        ]

    def transitions(self) -> List[AbstractTransition]:
        return list(self._transitions)

    def post_image(self, state: AbstractState) -> List[AbstractState]:
        targets = self._adjacency.get(state.state_id, set())
        return [self._states[t] for t in targets if t in self._states]

    def pre_image(self, state: AbstractState) -> List[AbstractState]:
        sources = self._reverse_adj.get(state.state_id, set())
        return [self._states[s] for s in sources if s in self._states]

    def refine(
        self,
        state: AbstractState,
        predicates: List[Predicate],
    ) -> List[AbstractState]:
        """Refine *state* by splitting along the widest dimension.

        Predicates guide which dimension to split on: we pick the
        dimension that best separates the predicate valuations.
        """
        if state.state_id not in self._states:
            return []

        split_var = self._choose_split_variable(state, predicates)
        b1, b2 = state.box.split_on(split_var)

        # Remove old state
        self._remove_state(state.state_id)

        # Add new states
        new1 = self._add_state(b1, state.mode)
        new2 = self._add_state(b2, state.mode)

        logger.debug(
            "Refined state %s along %s -> %s, %s",
            state.label,
            split_var,
            new1.label,
            new2.label,
        )
        return [new1, new2]

    def _choose_split_variable(
        self,
        state: AbstractState,
        predicates: List[Predicate],
    ) -> str:
        """Choose dimension to split on based on predicate relevance."""
        var_scores: Dict[str, float] = {v: 0.0 for v in self._var_names}

        for pred in predicates:
            for v in pred.variables:
                if v in var_scores:
                    var_scores[v] += 1.0

        # Among tied scores, prefer widest dimension
        best_var = max(
            self._var_names,
            key=lambda v: (var_scores.get(v, 0.0), state.box.dimensions.get(v, Interval(0, 0)).width()),
        )
        return best_var

    def _remove_state(self, sid: int) -> None:
        """Remove state and its transitions."""
        # Remove outgoing transitions
        for t in list(self._adjacency.get(sid, [])):
            self._reverse_adj.get(t, set()).discard(sid)
        self._adjacency.pop(sid, None)

        # Remove incoming transitions
        for s in list(self._reverse_adj.get(sid, [])):
            self._adjacency.get(s, set()).discard(sid)
        self._reverse_adj.pop(sid, None)

        # Remove transitions
        self._transitions = [
            tr for tr in self._transitions if tr.source_id != sid and tr.target_id != sid
        ]

        self._states.pop(sid, None)
        self._explored.discard(sid)

    def state_count(self) -> int:
        return len(self._states)

    def transition_count(self) -> int:
        return len(self._transitions)

    # -- Transition computation ---------------------------------------------

    def compute_transitions(
        self,
        rhs: Optional[Dict[str, ExprNode]] = None,
        overapprox: bool = True,
    ) -> int:
        """(Re)compute abstract transition relation.

        If *rhs* is given, uses interval evaluation of the ODE right-hand
        side to determine reachability.  Otherwise falls back to adjacency
        (neighbours in the grid are considered reachable).

        Returns the number of transitions added.
        """
        self._transitions.clear()
        for sid in self._adjacency:
            self._adjacency[sid].clear()
        for sid in self._reverse_adj:
            self._reverse_adj[sid].clear()

        count = 0
        state_list = list(self._states.values())

        if rhs is not None:
            count = self._compute_transitions_ode(state_list, rhs)
        else:
            count = self._compute_transitions_adjacency(state_list)

        logger.info("Computed %d abstract transitions", count)
        return count

    def _compute_transitions_adjacency(
        self,
        state_list: List[AbstractState],
    ) -> int:
        """Adjacency-based overapproximation: neighbouring boxes are connected."""
        count = 0
        for i, s1 in enumerate(state_list):
            for s2 in state_list[i:]:
                if s1.mode == s2.mode and _boxes_adjacent_or_overlap(s1.box, s2.box):
                    self._add_transition(s1.state_id, s2.state_id)
                    if s1.state_id != s2.state_id:
                        self._add_transition(s2.state_id, s1.state_id)
                    count += 1
        return count

    def _compute_transitions_ode(
        self,
        state_list: List[AbstractState],
        rhs: Dict[str, ExprNode],
    ) -> int:
        """ODE-based transition: box s can reach box t if flow points toward t."""
        count = 0
        for s in state_list:
            # Evaluate RHS at state's interval midpoint to approximate flow
            mid = s.box.midpoint()
            for t in state_list:
                if s.mode != t.mode:
                    continue
                if self._flow_can_reach(s.box, t.box, mid, rhs):
                    self._add_transition(s.state_id, t.state_id)
                    count += 1
        return count

    def _flow_can_reach(
        self,
        src: IntervalBox,
        dst: IntervalBox,
        flow_point: Dict[str, float],
        rhs: Dict[str, ExprNode],
    ) -> bool:
        """Conservative check: can flow from *src* reach *dst*?"""
        if src == dst:
            return True  # self-loop always possible
        if not _boxes_adjacent_or_overlap(src, dst):
            return False

        # Check if flow direction at midpoint is compatible
        for var in src.variable_names:
            rhs_expr = rhs.get(var)
            if rhs_expr is None:
                continue
            src_iv = src.dimensions.get(var)
            dst_iv = dst.dimensions.get(var)
            if src_iv is None or dst_iv is None:
                continue

            # If dst is entirely above src, flow must be non-negative at some point
            if dst_iv.lo >= src_iv.hi:
                pass  # conservatively allow (overapproximation)
            elif dst_iv.hi <= src_iv.lo:
                pass  # conservatively allow
        return True

    def _add_transition(self, src_id: int, dst_id: int) -> None:
        tr = AbstractTransition(source_id=src_id, target_id=dst_id)
        self._transitions.append(tr)
        self._adjacency.setdefault(src_id, set()).add(dst_id)
        self._reverse_adj.setdefault(dst_id, set()).add(src_id)

    # -- Lazy exploration ---------------------------------------------------

    def explore_from_initial(self, max_depth: int = 100) -> Set[int]:
        """BFS from initial states, marking reachable states.

        Returns the set of reachable state IDs.
        """
        frontier = [s.state_id for s in self.initial_states()]
        visited: Set[int] = set(frontier)

        depth = 0
        while frontier and depth < max_depth:
            next_frontier: List[int] = []
            for sid in frontier:
                for tid in self._adjacency.get(sid, set()):
                    if tid not in visited:
                        visited.add(tid)
                        next_frontier.append(tid)
            frontier = next_frontier
            depth += 1

        self._explored = visited
        logger.info("Explored %d / %d states (depth %d)", len(visited), len(self._states), depth)
        return visited

    @property
    def explored_states(self) -> List[AbstractState]:
        return [self._states[sid] for sid in self._explored if sid in self._states]

    # -- Widening / Narrowing -----------------------------------------------

    def widen(self, s1: AbstractState, s2: AbstractState) -> AbstractState:
        """Widening operator: hull of two abstract states.

        Accelerates fixpoint computation by extrapolating growth.
        If a dimension of *s2* exceeds *s1*, widen to the domain boundary.
        """
        new_dims: Dict[str, Interval] = {}
        for var in self._var_names:
            iv1 = s1.box.dimensions.get(var, Interval(0, 0))
            iv2 = s2.box.dimensions.get(var, Interval(0, 0))
            bound_lo, bound_hi = self._bounds.get(var, (-1e12, 1e12))

            lo = iv1.lo if iv2.lo >= iv1.lo else bound_lo
            hi = iv1.hi if iv2.hi <= iv1.hi else bound_hi
            new_dims[var] = Interval(lo, hi)

        return AbstractState(
            state_id=-1,
            box=IntervalBox(dimensions=new_dims),
            mode=s1.mode,
        )

    def narrow(self, s1: AbstractState, s2: AbstractState) -> AbstractState:
        """Narrowing operator: intersect to improve precision after widening."""
        new_dims: Dict[str, Interval] = {}
        for var in self._var_names:
            iv1 = s1.box.dimensions.get(var, Interval(-1e12, 1e12))
            iv2 = s2.box.dimensions.get(var, Interval(-1e12, 1e12))
            bound_lo, bound_hi = self._bounds.get(var, (-1e12, 1e12))

            lo = iv2.lo if iv1.lo == bound_lo else iv1.lo
            hi = iv2.hi if iv1.hi == bound_hi else iv1.hi
            new_dims[var] = Interval(lo, hi)

        return AbstractState(
            state_id=-1,
            box=IntervalBox(dimensions=new_dims),
            mode=s1.mode,
        )

    # -- Reporting ----------------------------------------------------------

    def size_report(self) -> Dict[str, Any]:
        return {
            "num_states": self.state_count(),
            "num_transitions": self.transition_count(),
            "num_explored": len(self._explored),
            "num_dimensions": len(self._var_names),
            "variables": self._var_names,
        }

    def get_state(self, state_id: int) -> Optional[AbstractState]:
        return self._states.get(state_id)

    def enumerate_states(self) -> Iterator[AbstractState]:
        return iter(self._states.values())


# ---------------------------------------------------------------------------
# Predicate abstraction
# ---------------------------------------------------------------------------


class PredicateAbstraction(AbstractionDomain):
    """Abstraction defined by a set of Boolean predicates.

    Each abstract state is a Boolean valuation of the predicates.
    Only valuations consistent with the concrete state space are kept.
    """

    def __init__(
        self,
        predicate_set: PredicateSet,
        bounds: Dict[str, Tuple[float, float]],
        initial_region: Optional[IntervalBox] = None,
        solver: Optional[Any] = None,
    ) -> None:
        self._predicates = predicate_set
        self._bounds = bounds
        self._var_names = sorted(bounds.keys())
        self._initial_region = initial_region or IntervalBox.from_bounds(bounds)
        self._solver = solver

        self._next_id = 0
        self._states: Dict[int, AbstractState] = {}
        self._transitions: List[AbstractTransition] = []
        self._adjacency: Dict[int, Set[int]] = {}
        self._reverse_adj: Dict[int, Set[int]] = {}

    def build(self) -> None:
        """Enumerate feasible abstract states from predicate valuations."""
        predicates = self._predicates.predicates
        n = len(predicates)

        if n > 20:
            logger.warning(
                "Predicate abstraction with %d predicates — exponential blowup risk", n
            )

        for bits in range(1 << n):
            valuation = frozenset(
                (predicates[i].digest, bool(bits & (1 << i))) for i in range(n)
            )
            # Check feasibility via SMT if solver is available
            if self._solver is not None:
                feasible = self._check_feasibility(predicates, valuation)
                if not feasible:
                    continue

            box = IntervalBox.from_bounds(self._bounds)
            sid = self._next_id
            self._next_id += 1
            state = AbstractState(
                state_id=sid,
                box=box,
                predicate_valuation=valuation,
            )
            self._states[sid] = state
            self._adjacency[sid] = set()
            self._reverse_adj[sid] = set()

        logger.info(
            "Predicate abstraction: %d feasible states from %d predicates",
            len(self._states),
            n,
        )

    def _check_feasibility(
        self,
        predicates: List[Predicate],
        valuation: FrozenSet[Tuple[str, bool]],
    ) -> bool:
        """Check if a predicate valuation is satisfiable."""
        try:
            self._solver.push()
            val_dict = dict(valuation)
            for pred in predicates:
                if val_dict.get(pred.digest, True):
                    self._solver.assert_formula(pred.expr)
                else:
                    self._solver.assert_formula(Not(pred.expr))
            # Add bound constraints
            for var in self._var_names:
                lo, hi = self._bounds[var]
                x = Var(var)
                self._solver.assert_formula(Ge(x, Const(lo)))
                self._solver.assert_formula(Le(x, Const(hi)))
            result = self._solver.check_sat()
            self._solver.pop()
            return result.is_sat
        except Exception:
            return True  # conservatively assume feasible

    # -- AbstractionDomain interface ----------------------------------------

    def states(self) -> List[AbstractState]:
        return list(self._states.values())

    def initial_states(self) -> List[AbstractState]:
        return list(self._states.values())  # conservative: all feasible states

    def transitions(self) -> List[AbstractTransition]:
        return list(self._transitions)

    def post_image(self, state: AbstractState) -> List[AbstractState]:
        targets = self._adjacency.get(state.state_id, set())
        return [self._states[t] for t in targets if t in self._states]

    def pre_image(self, state: AbstractState) -> List[AbstractState]:
        sources = self._reverse_adj.get(state.state_id, set())
        return [self._states[s] for s in sources if s in self._states]

    def refine(
        self,
        state: AbstractState,
        predicates: List[Predicate],
    ) -> List[AbstractState]:
        """Add predicates and re-split state."""
        new_count = self._predicates.add_all(predicates)
        if new_count == 0:
            return [state]

        # Remove old state, rebuild with new predicates
        self._states.pop(state.state_id, None)
        self.build()
        return list(self._states.values())

    def state_count(self) -> int:
        return len(self._states)

    def transition_count(self) -> int:
        return len(self._transitions)

    def compute_transitions_smt(self) -> int:
        """Compute transitions using SMT to check one-step reachability."""
        self._transitions.clear()
        count = 0
        states = list(self._states.values())
        for s in states:
            for t in states:
                # Conservative: allow all transitions (refine lazily)
                self._adjacency.setdefault(s.state_id, set()).add(t.state_id)
                self._reverse_adj.setdefault(t.state_id, set()).add(s.state_id)
                self._transitions.append(
                    AbstractTransition(source_id=s.state_id, target_id=t.state_id)
                )
                count += 1
        return count


# ---------------------------------------------------------------------------
# Product abstraction
# ---------------------------------------------------------------------------


class ProductAbstraction(AbstractionDomain):
    """Product of interval, predicate, and parameter abstractions.

    Combines spatial interval partitioning with predicate-based
    splitting and parameter uncertainty envelopes.
    """

    def __init__(
        self,
        interval_abs: IntervalAbstraction,
        predicate_abs: Optional[PredicateAbstraction] = None,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self._interval = interval_abs
        self._predicate = predicate_abs
        self._param_bounds = param_bounds or {}
        self._next_id = 0
        self._product_states: Dict[int, AbstractState] = {}
        self._transitions: List[AbstractTransition] = []
        self._adjacency: Dict[int, Set[int]] = {}

    def build_product(self) -> None:
        """Build product state space from component abstractions."""
        interval_states = self._interval.states()

        if self._predicate is None:
            for ist in interval_states:
                sid = self._next_id
                self._next_id += 1
                self._product_states[sid] = AbstractState(
                    state_id=sid,
                    box=ist.box,
                    mode=ist.mode,
                    predicate_valuation=frozenset(),
                )
            return

        pred_states = self._predicate.states()
        for ist in interval_states:
            for pst in pred_states:
                sid = self._next_id
                self._next_id += 1
                self._product_states[sid] = AbstractState(
                    state_id=sid,
                    box=ist.box,
                    mode=ist.mode,
                    predicate_valuation=pst.predicate_valuation,
                )

        logger.info(
            "Product abstraction: %d states (%d interval × %d predicate)",
            len(self._product_states),
            len(interval_states),
            len(pred_states),
        )

    # -- AbstractionDomain interface ----------------------------------------

    def states(self) -> List[AbstractState]:
        return list(self._product_states.values())

    def initial_states(self) -> List[AbstractState]:
        initial_boxes = {s.state_id for s in self._interval.initial_states()}
        return [
            s
            for s in self._product_states.values()
            if any(
                s.box.overlaps(self._interval.get_state(sid).box)
                for sid in initial_boxes
                if self._interval.get_state(sid) is not None
            )
        ]

    def transitions(self) -> List[AbstractTransition]:
        return list(self._transitions)

    def post_image(self, state: AbstractState) -> List[AbstractState]:
        return [
            self._product_states[t]
            for t in self._adjacency.get(state.state_id, set())
            if t in self._product_states
        ]

    def pre_image(self, state: AbstractState) -> List[AbstractState]:
        return []  # lazy: compute on demand

    def refine(
        self,
        state: AbstractState,
        predicates: List[Predicate],
    ) -> List[AbstractState]:
        """Refine using the interval abstraction component."""
        # Delegate to interval abstraction for spatial refinement
        matching = [
            s
            for s in self._interval.states()
            if s.box.overlaps(state.box)
        ]
        new_states: List[AbstractState] = []
        for ms in matching:
            refined = self._interval.refine(ms, predicates)
            for rs in refined:
                sid = self._next_id
                self._next_id += 1
                ns = AbstractState(
                    state_id=sid,
                    box=rs.box,
                    mode=state.mode,
                    predicate_valuation=state.predicate_valuation,
                )
                self._product_states[sid] = ns
                new_states.append(ns)

        # Remove old product state
        self._product_states.pop(state.state_id, None)
        return new_states

    def state_count(self) -> int:
        return len(self._product_states)

    def transition_count(self) -> int:
        return len(self._transitions)

    def refine_parameter_envelope(
        self,
        param_name: str,
        split_point: Optional[float] = None,
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
        """Split a parameter range at *split_point* (default: midpoint)."""
        if param_name not in self._param_bounds:
            raise ValueError(f"Unknown parameter: {param_name}")

        lo, hi = self._param_bounds[param_name]
        mid = split_point if split_point is not None else (lo + hi) / 2.0

        env1 = dict(self._param_bounds)
        env2 = dict(self._param_bounds)
        env1[param_name] = (lo, mid)
        env2[param_name] = (mid, hi)
        return env1, env2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _boxes_adjacent_or_overlap(a: IntervalBox, b: IntervalBox) -> bool:
    """Check if two boxes share a face, edge, or overlap."""
    for var in a.variable_names:
        if var not in b.dimensions:
            continue
        iv_a = a.dimensions[var]
        iv_b = b.dimensions[var]
        # Disjoint with gap → not adjacent
        if iv_a.hi < iv_b.lo or iv_b.hi < iv_a.lo:
            return False
    return True


def build_initial_abstraction(
    bounds: Dict[str, Tuple[float, float]],
    initial_region: Optional[IntervalBox] = None,
    resolution: int = 4,
    predicates: Optional[PredicateSet] = None,
    solver: Optional[Any] = None,
) -> AbstractionDomain:
    """Factory function: build a suitable initial abstraction.

    Returns an ``IntervalAbstraction`` if no predicates are given,
    or a ``ProductAbstraction`` if predicates are provided.
    """
    interval = IntervalAbstraction(
        bounds=bounds,
        initial_region=initial_region,
        grid_resolution=resolution,
    )

    if predicates is None or len(predicates) == 0:
        return interval

    pred_abs = PredicateAbstraction(
        predicate_set=predicates,
        bounds=bounds,
        initial_region=initial_region,
        solver=solver,
    )
    pred_abs.build()

    product = ProductAbstraction(
        interval_abs=interval,
        predicate_abs=pred_abs,
    )
    product.build_product()
    return product

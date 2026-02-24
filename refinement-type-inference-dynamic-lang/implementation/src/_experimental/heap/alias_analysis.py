"""Alias analysis and points-to analysis for the Python-native refinement type system.

Provides flow-sensitive and flow-insensitive alias analysis, Andersen-style
points-to graphs, and pattern-based analyzers for common Python idioms such
as factory functions, builder patterns, and dependency injection.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from src.heap.heap_model import HeapAddress, AbstractValue, AbstractHeap


# ---------------------------------------------------------------------------
# AliasRelation
# ---------------------------------------------------------------------------

class AliasRelation(Enum):
    """Three-valued alias relation between two variables."""

    MUST_ALIAS = auto()
    MAY_ALIAS = auto()
    NO_ALIAS = auto()

    def join(self, other: AliasRelation) -> AliasRelation:
        """Lattice join: MUST ⊔ NO = MAY, etc."""
        if self is other:
            return self
        if self is AliasRelation.NO_ALIAS and other is AliasRelation.NO_ALIAS:
            return AliasRelation.NO_ALIAS
        if self is AliasRelation.MUST_ALIAS and other is AliasRelation.MUST_ALIAS:
            return AliasRelation.MUST_ALIAS
        return AliasRelation.MAY_ALIAS

    def meet(self, other: AliasRelation) -> AliasRelation:
        """Lattice meet (greatest lower bound)."""
        if self is other:
            return self
        if AliasRelation.NO_ALIAS in (self, other):
            return AliasRelation.NO_ALIAS
        if AliasRelation.MUST_ALIAS in (self, other):
            return AliasRelation.MAY_ALIAS
        return AliasRelation.MAY_ALIAS


# ---------------------------------------------------------------------------
# FieldPath
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldPath:
    """Represents an access path such as ``x.foo.bar``.

    Attributes:
        base: The root variable name.
        fields: Tuple of field names traversed from *base*.
    """

    base: str
    fields: Tuple[str, ...] = ()

    # -- construction helpers ------------------------------------------------

    def extend(self, field_name: str) -> FieldPath:
        """Return a new path with *field_name* appended."""
        return FieldPath(base=self.base, fields=self.fields + (field_name,))

    # -- queries -------------------------------------------------------------

    def prefix_of(self, other: FieldPath) -> bool:
        """Return ``True`` if *self* is a prefix of *other*.

        ``x.foo`` is a prefix of ``x.foo.bar`` but not of ``y.foo``.
        """
        if self.base != other.base:
            return False
        if len(self.fields) > len(other.fields):
            return False
        return other.fields[: len(self.fields)] == self.fields

    def __hash__(self) -> int:
        return hash((self.base, self.fields))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FieldPath):
            return NotImplemented
        return self.base == other.base and self.fields == other.fields

    def __repr__(self) -> str:
        if self.fields:
            return f"{self.base}.{'.'.join(self.fields)}"
        return self.base


# ---------------------------------------------------------------------------
# AliasSet
# ---------------------------------------------------------------------------

@dataclass
class AliasSet:
    """Tracks which variables may point to the same heap objects.

    Maintains both *may* and *must* points-to information so that callers can
    distinguish definite from possible aliasing.
    """

    points_to: Dict[str, Set[HeapAddress]] = field(default_factory=dict)
    must_point_to: Dict[str, Optional[HeapAddress]] = field(default_factory=dict)

    # -- internal helpers ----------------------------------------------------

    def _copy(self) -> AliasSet:
        return AliasSet(
            points_to={k: set(v) for k, v in self.points_to.items()},
            must_point_to=dict(self.must_point_to),
        )

    def _pts(self, var: str) -> Set[HeapAddress]:
        """Return the points-to set for *var*, defaulting to empty."""
        return self.points_to.get(var, set())

    # -- alias queries -------------------------------------------------------

    def may_alias(self, x: str, y: str) -> bool:
        """Return ``True`` when *x* and *y* have overlapping points-to sets."""
        sx = self._pts(x)
        sy = self._pts(y)
        if not sx or not sy:
            return False
        return bool(sx & sy)

    def must_alias(self, x: str, y: str) -> bool:
        """Return ``True`` when both *x* and *y* definitely refer to the
        same single heap address."""
        mx = self.must_point_to.get(x)
        my = self.must_point_to.get(y)
        if mx is None or my is None:
            return False
        return mx == my

    def query(self, x: str, y: str) -> AliasRelation:
        """Return the alias relation between *x* and *y*."""
        if self.must_alias(x, y):
            return AliasRelation.MUST_ALIAS
        if self.may_alias(x, y):
            return AliasRelation.MAY_ALIAS
        return AliasRelation.NO_ALIAS

    # -- transfer functions --------------------------------------------------

    def assign(self, x: str, y: str) -> AliasSet:
        """Model ``x = y``: *x* now points to everything *y* points to."""
        result = self._copy()
        src = self._pts(y)
        result.points_to[x] = set(src) if src else set()
        result.must_point_to[x] = self.must_point_to.get(y)
        return result

    def allocate(self, x: str, addr: HeapAddress) -> AliasSet:
        """Model ``x = Foo()`` – *x* points to a freshly-allocated *addr*."""
        result = self._copy()
        result.points_to[x] = {addr}
        result.must_point_to[x] = addr
        return result

    def load_attr(self, result_var: str, obj: str, attr: str,
                  heap: AbstractHeap) -> AliasSet:
        """Model ``result_var = obj.attr``.

        Looks up all addresses *obj* may point to, reads *attr* from each
        corresponding heap object, and unions the resulting addresses.
        """
        result = self._copy()
        target_addrs: Set[HeapAddress] = set()
        for addr in self._pts(obj):
            heap_obj = heap.get(addr)
            if heap_obj is not None:
                field_val = heap_obj.get_field(attr)
                if field_val is not None and hasattr(field_val, "addresses"):
                    target_addrs.update(field_val.addresses)
        result.points_to[result_var] = target_addrs
        if len(target_addrs) == 1:
            result.must_point_to[result_var] = next(iter(target_addrs))
        else:
            result.must_point_to[result_var] = None
        return result

    def store_attr(self, obj: str, attr: str, val: str) -> AliasSet:
        """Model ``obj.attr = val``.

        Because the store may affect any object *obj* may point to, we
        conservatively invalidate must-alias info for variables that loaded
        the same field from an overlapping set.
        """
        result = self._copy()
        obj_addrs = self._pts(obj)
        val_addrs = self._pts(val)
        # Invalidate must-alias for any variable whose points-to set was
        # derived from the same field on an overlapping base.
        for var, addrs in list(result.points_to.items()):
            if var == val or var == obj:
                continue
            if addrs & obj_addrs:
                result.must_point_to[var] = None
        return result

    def invalidate(self, x: str) -> AliasSet:
        """Mark *x* as escaped / Top – it may alias anything."""
        result = self._copy()
        all_addrs: Set[HeapAddress] = set()
        for addrs in result.points_to.values():
            all_addrs.update(addrs)
        result.points_to[x] = set(all_addrs)
        result.must_point_to[x] = None
        return result

    def call_effect(self, args: List[str],
                    escaping: Set[str]) -> AliasSet:
        """After a call, escaped arguments may alias anything reachable.

        Non-escaping arguments keep their existing alias sets. Escaping
        arguments are widened to include all addresses reachable from any
        other escaping argument.
        """
        result = self._copy()
        escaped_addrs: Set[HeapAddress] = set()
        for a in args:
            if a in escaping:
                escaped_addrs.update(self._pts(a))
        for a in args:
            if a in escaping:
                result.points_to[a] = set(escaped_addrs)
                result.must_point_to[a] = None
        return result

    def kill(self, x: str) -> AliasSet:
        """Model reassignment of *x* – remove old alias information."""
        result = self._copy()
        result.points_to.pop(x, None)
        result.must_point_to.pop(x, None)
        return result

    # -- lattice operations --------------------------------------------------

    def join(self, other: AliasSet) -> AliasSet:
        """Lattice join – union of points-to information."""
        all_vars = set(self.points_to.keys()) | set(other.points_to.keys())
        new_pts: Dict[str, Set[HeapAddress]] = {}
        new_must: Dict[str, Optional[HeapAddress]] = {}
        for v in all_vars:
            s1 = self._pts(v)
            s2 = other._pts(v)
            merged = s1 | s2
            new_pts[v] = merged
            m1 = self.must_point_to.get(v)
            m2 = other.must_point_to.get(v)
            if m1 is not None and m1 == m2:
                new_must[v] = m1
            else:
                new_must[v] = None
        return AliasSet(points_to=new_pts, must_point_to=new_must)

    def meet(self, other: AliasSet) -> AliasSet:
        """Lattice meet – intersection of points-to information."""
        common_vars = set(self.points_to.keys()) & set(other.points_to.keys())
        new_pts: Dict[str, Set[HeapAddress]] = {}
        new_must: Dict[str, Optional[HeapAddress]] = {}
        for v in common_vars:
            s1 = self._pts(v)
            s2 = other._pts(v)
            inter = s1 & s2
            if inter:
                new_pts[v] = inter
                m1 = self.must_point_to.get(v)
                m2 = other.must_point_to.get(v)
                if m1 is not None and m1 == m2 and m1 in inter:
                    new_must[v] = m1
                elif len(inter) == 1:
                    new_must[v] = next(iter(inter))
                else:
                    new_must[v] = None
        return AliasSet(points_to=new_pts, must_point_to=new_must)

    def widen(self, other: AliasSet) -> AliasSet:
        """Widening operator for convergence.

        Once a variable's points-to set grows beyond a threshold, it is
        promoted to Top (all known addresses).
        """
        _WIDEN_THRESHOLD = 16
        joined = self.join(other)
        all_addrs: Set[HeapAddress] = set()
        for addrs in joined.points_to.values():
            all_addrs.update(addrs)
        for v in list(joined.points_to.keys()):
            if len(joined.points_to[v]) > _WIDEN_THRESHOLD:
                joined.points_to[v] = set(all_addrs)
                joined.must_point_to[v] = None
        return joined

    # -- utility -------------------------------------------------------------

    def get_aliases_of(self, x: str) -> Set[str]:
        """Return all variables that may alias *x*."""
        sx = self._pts(x)
        if not sx:
            return set()
        aliases: Set[str] = set()
        for var, addrs in self.points_to.items():
            if var == x:
                continue
            if addrs & sx:
                aliases.add(var)
        return aliases

    def __repr__(self) -> str:
        entries = []
        for v in sorted(self.points_to.keys()):
            m = self.must_point_to.get(v)
            tag = f" [must={m}]" if m else ""
            entries.append(f"  {v} -> {self.points_to[v]}{tag}")
        return "AliasSet{\n" + "\n".join(entries) + "\n}"


# ---------------------------------------------------------------------------
# PointsToGraph  –  Andersen-style (flow-insensitive) inclusion-based
# ---------------------------------------------------------------------------

@dataclass
class PointsToGraph:
    """Andersen-style inclusion-based points-to analysis.

    This is a flow-*insensitive* analysis that computes a single summary for
    the whole function/module.  It serves as a cheap baseline when full
    flow-sensitivity is not needed.
    """

    var_edges: Dict[str, Set[HeapAddress]] = field(default_factory=dict)
    field_edges: Dict[Tuple[HeapAddress, str], Set[HeapAddress]] = field(
        default_factory=dict
    )
    reverse_edges: Dict[HeapAddress, Set[str]] = field(default_factory=dict)

    # inclusion constraints: var ⊇ var
    _subset_constraints: List[Tuple[str, str]] = field(
        default_factory=list, repr=False
    )
    # complex constraints: *x.f ⊇ y, y ⊇ *x.f
    _load_constraints: List[Tuple[str, str, str]] = field(
        default_factory=list, repr=False
    )
    _store_constraints: List[Tuple[str, str, str]] = field(
        default_factory=list, repr=False
    )

    # -- edge manipulation ---------------------------------------------------

    def add_var_edge(self, var: str, addr: HeapAddress) -> None:
        """Record that *var* may point to *addr*."""
        self.var_edges.setdefault(var, set()).add(addr)
        self.reverse_edges.setdefault(addr, set()).add(var)

    def add_field_edge(self, base_addr: HeapAddress, field_name: str,
                       target_addr: HeapAddress) -> None:
        """Record that ``base_addr.field_name`` may point to *target_addr*."""
        key = (base_addr, field_name)
        self.field_edges.setdefault(key, set()).add(target_addr)

    def add_subset_constraint(self, lhs: str, rhs: str) -> None:
        """Record ``pts(lhs) ⊇ pts(rhs)``."""
        self._subset_constraints.append((lhs, rhs))

    def add_load_constraint(self, result: str, base_var: str,
                            field_name: str) -> None:
        """Record ``result = base_var.field_name``."""
        self._load_constraints.append((result, base_var, field_name))

    def add_store_constraint(self, base_var: str, field_name: str,
                             val_var: str) -> None:
        """Record ``base_var.field_name = val_var``."""
        self._store_constraints.append((base_var, field_name, val_var))

    # -- solving -------------------------------------------------------------

    def propagate(self) -> None:
        """Compute fixpoint by iterating subset/load/store constraints.

        Uses a simple worklist algorithm: keep propagating until no points-to
        set changes.
        """
        changed = True
        iterations = 0
        max_iter = 200
        while changed and iterations < max_iter:
            changed = False
            iterations += 1

            # 1. Subset constraints: pts(lhs) ⊇ pts(rhs)
            for lhs, rhs in self._subset_constraints:
                rhs_pts = self.var_edges.get(rhs, set())
                lhs_pts = self.var_edges.setdefault(lhs, set())
                before = len(lhs_pts)
                lhs_pts |= rhs_pts
                if len(lhs_pts) != before:
                    changed = True
                    for addr in lhs_pts:
                        self.reverse_edges.setdefault(addr, set()).add(lhs)

            # 2. Load constraints: result = base.field
            for result, base_var, fld in self._load_constraints:
                base_pts = self.var_edges.get(base_var, set())
                result_pts = self.var_edges.setdefault(result, set())
                before = len(result_pts)
                for addr in base_pts:
                    key = (addr, fld)
                    targets = self.field_edges.get(key, set())
                    result_pts |= targets
                if len(result_pts) != before:
                    changed = True
                    for addr in result_pts:
                        self.reverse_edges.setdefault(addr, set()).add(result)

            # 3. Store constraints: base.field = val
            for base_var, fld, val_var in self._store_constraints:
                base_pts = self.var_edges.get(base_var, set())
                val_pts = self.var_edges.get(val_var, set())
                for addr in base_pts:
                    key = (addr, fld)
                    field_set = self.field_edges.setdefault(key, set())
                    before = len(field_set)
                    field_set |= val_pts
                    if len(field_set) != before:
                        changed = True

    # -- queries -------------------------------------------------------------

    def points_to(self, var: str) -> Set[HeapAddress]:
        """Return the set of addresses *var* may point to."""
        return set(self.var_edges.get(var, set()))

    def field_points_to(self, addr: HeapAddress,
                        field_name: str) -> Set[HeapAddress]:
        """Return the addresses reachable through ``addr.field_name``."""
        return set(self.field_edges.get((addr, field_name), set()))

    def vars_pointing_to(self, addr: HeapAddress) -> Set[str]:
        """Return all variables whose points-to set includes *addr*."""
        return set(self.reverse_edges.get(addr, set()))

    def path_points_to(self, path: FieldPath) -> Set[HeapAddress]:
        """Resolve a multi-level access path through the graph.

        For example, ``x.foo.bar`` first resolves ``x`` → addresses, then
        reads ``foo`` from each, then ``bar`` from each result.
        """
        current: Set[HeapAddress] = self.points_to(path.base)
        for fld in path.fields:
            next_addrs: Set[HeapAddress] = set()
            for addr in current:
                next_addrs |= self.field_points_to(addr, fld)
            current = next_addrs
            if not current:
                break
        return current

    def __repr__(self) -> str:
        lines = ["PointsToGraph{"]
        for v in sorted(self.var_edges):
            lines.append(f"  {v} -> {self.var_edges[v]}")
        for (addr, fld), targets in sorted(
            self.field_edges.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])
        ):
            lines.append(f"  {addr}.{fld} -> {targets}")
        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# FlowSensitivePointsTo  –  context-sensitive, flow-sensitive analysis
# ---------------------------------------------------------------------------

@dataclass
class FlowSensitivePointsTo:
    """Flow-sensitive points-to analysis with k-CFA context sensitivity.

    Maintains a separate :class:`AliasSet` per program point and propagates
    information along CFG edges until a fixpoint is reached.
    """

    states: Dict[int, AliasSet] = field(default_factory=dict)
    call_context: Tuple[str, ...] = ()
    context_depth: int = 1

    # -- helpers -------------------------------------------------------------

    def _ensure_state(self, point: int) -> AliasSet:
        """Return the state at *point*, creating an empty one if absent."""
        if point not in self.states:
            self.states[point] = AliasSet()
        return self.states[point]

    def _contextualized_addr(self, site: str) -> HeapAddress:
        """Create a context-qualified heap address for an allocation site."""
        return HeapAddress(site=site, context=self.call_context)

    # -- transfer functions --------------------------------------------------

    def analyze_assignment(self, point: int, target: str,
                           value: str) -> None:
        """Model ``target = value`` at program *point*."""
        state = self._ensure_state(point)
        new_state = state.kill(target).assign(target, value)
        self.states[point] = new_state

    def analyze_allocation(self, point: int, target: str, site: str,
                           class_addr: HeapAddress) -> None:
        """Model ``target = ClassName()`` at *point*.

        Creates a fresh heap address qualified with the current calling
        context and records the allocation.
        """
        state = self._ensure_state(point)
        addr = self._contextualized_addr(site)
        new_state = state.kill(target).allocate(target, addr)
        self.states[point] = new_state

    def analyze_load(self, point: int, result: str, obj: str,
                     field: str) -> None:
        """Model ``result = obj.field`` at *point*.

        Without a concrete heap we approximate: *result* may point to
        anything *obj* points to (field-insensitive fallback) plus any
        known field-level information.
        """
        state = self._ensure_state(point)
        new_state = state._copy()
        obj_addrs = state._pts(obj)
        new_state.points_to[result] = set(obj_addrs)
        if len(obj_addrs) == 1:
            new_state.must_point_to[result] = next(iter(obj_addrs))
        else:
            new_state.must_point_to[result] = None
        self.states[point] = new_state

    def analyze_store(self, point: int, obj: str, field: str,
                      value: str) -> None:
        """Model ``obj.field = value`` at *point*."""
        state = self._ensure_state(point)
        new_state = state.store_attr(obj, field, value)
        self.states[point] = new_state

    def analyze_call(self, point: int, func: str, args: List[str],
                     ret_var: str, call_site: str) -> None:
        """Model a function call at *point*.

        Conservatively: all arguments escape to the callee, return value
        may alias any argument.  The context is extended with *call_site*.
        """
        state = self._ensure_state(point)
        escaping = set(args)
        new_state = state.call_effect(args, escaping)
        # Return value may alias any argument
        ret_addrs: Set[HeapAddress] = set()
        for a in args:
            ret_addrs |= new_state._pts(a)
        # Also add a fresh address for possible new allocation in callee
        callee_addr = self._contextualized_addr(call_site)
        ret_addrs.add(callee_addr)
        new_state.points_to[ret_var] = ret_addrs
        if len(ret_addrs) == 1:
            new_state.must_point_to[ret_var] = next(iter(ret_addrs))
        else:
            new_state.must_point_to[ret_var] = None
        self.states[point] = new_state

    def analyze_return(self, point: int, value: str) -> None:
        """Model ``return value`` at *point*.

        Marks the special ``__return__`` variable as aliasing *value*.
        """
        state = self._ensure_state(point)
        new_state = state.assign("__return__", value)
        self.states[point] = new_state

    def analyze_branch(self, point: int, true_point: int,
                       false_point: int) -> None:
        """Propagate the state at *point* to both branch targets.

        A more sophisticated analysis could refine the state using the
        branch condition; here we simply copy.
        """
        state = self._ensure_state(point)
        self.states[true_point] = state.join(self._ensure_state(true_point))
        self.states[false_point] = state.join(self._ensure_state(false_point))

    # -- merging / queries ---------------------------------------------------

    def merge_points(self, points: List[int]) -> AliasSet:
        """Join the states at several program points."""
        if not points:
            return AliasSet()
        merged = self._ensure_state(points[0])._copy()
        for p in points[1:]:
            merged = merged.join(self._ensure_state(p))
        return merged

    def state_at(self, point: int) -> AliasSet:
        """Return the alias state at *point*."""
        return self._ensure_state(point)

    # -- context manipulation ------------------------------------------------

    def push_context(self, call_site: str) -> FlowSensitivePointsTo:
        """Return a new analysis instance with *call_site* appended to the
        context string, trimmed to ``context_depth``."""
        new_ctx = (self.call_context + (call_site,))[-self.context_depth :]
        return FlowSensitivePointsTo(
            states={k: v._copy() for k, v in self.states.items()},
            call_context=new_ctx,
            context_depth=self.context_depth,
        )

    def pop_context(self) -> FlowSensitivePointsTo:
        """Return a new analysis instance with the most recent call site
        removed from the context."""
        new_ctx = self.call_context[:-1] if self.call_context else ()
        return FlowSensitivePointsTo(
            states={k: v._copy() for k, v in self.states.items()},
            call_context=new_ctx,
            context_depth=self.context_depth,
        )

    # -- fixpoint ------------------------------------------------------------

    def fixpoint(self, cfg_edges: Dict[int, List[int]]) -> None:
        """Iterate until the alias state at every program point stabilises.

        ``cfg_edges`` maps each program point to its successor points.
        The algorithm is a standard worklist-based chaotic iteration with
        widening applied after a configurable number of visits.
        """
        _WIDEN_AFTER = 5
        visit_count: Dict[int, int] = {}
        all_points: Set[int] = set(cfg_edges.keys())
        for succs in cfg_edges.values():
            all_points.update(succs)

        worklist: List[int] = sorted(all_points)
        for p in all_points:
            self._ensure_state(p)
            visit_count[p] = 0

        while worklist:
            point = worklist.pop(0)
            visit_count[point] = visit_count.get(point, 0) + 1
            current = self.states[point]

            for succ in cfg_edges.get(point, []):
                old_succ = self.states[succ]
                if visit_count.get(succ, 0) >= _WIDEN_AFTER:
                    new_succ = old_succ.widen(current)
                else:
                    new_succ = old_succ.join(current)

                # Check if the successor state changed
                if self._state_changed(old_succ, new_succ):
                    self.states[succ] = new_succ
                    if succ not in worklist:
                        worklist.append(succ)

    @staticmethod
    def _state_changed(old: AliasSet, new: AliasSet) -> bool:
        """Return ``True`` if *new* contains strictly more information than
        *old* (i.e. at least one points-to set grew)."""
        for v, addrs in new.points_to.items():
            old_addrs = old.points_to.get(v, set())
            if not addrs <= old_addrs:
                return True
        for v in new.must_point_to:
            if v not in old.must_point_to:
                return True
            if new.must_point_to[v] != old.must_point_to.get(v):
                return True
        return False


# ---------------------------------------------------------------------------
# AliasAnalysisResult
# ---------------------------------------------------------------------------

@dataclass
class AliasAnalysisResult:
    """Summary of alias analysis for a single function body.

    Captures inter-procedural information needed by callers: which
    parameters may alias each other, what escapes, what is modified,
    and what the return value may alias.
    """

    param_aliases: Dict[str, Set[str]] = field(default_factory=dict)
    return_aliases: Set[str] = field(default_factory=set)
    escaping: Set[str] = field(default_factory=set)
    modified: Set[FieldPath] = field(default_factory=set)
    allocated: Set[HeapAddress] = field(default_factory=set)

    def may_modify(self, path: FieldPath) -> bool:
        """Return ``True`` if *path* (or any prefix of it) may be written."""
        for m in self.modified:
            if m == path or m.prefix_of(path) or path.prefix_of(m):
                return True
        return False

    def escapes(self, var: str) -> bool:
        """Return ``True`` if *var* escapes into the heap or to callers."""
        return var in self.escaping

    def aliases_return(self, var: str) -> bool:
        """Return ``True`` if *var* may alias the return value."""
        return var in self.return_aliases

    def __repr__(self) -> str:
        parts = [
            f"param_aliases={self.param_aliases}",
            f"return_aliases={self.return_aliases}",
            f"escaping={self.escaping}",
            f"modified={self.modified}",
            f"allocated={self.allocated}",
        ]
        return "AliasAnalysisResult(" + ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# PythonPatternAnalyzer
# ---------------------------------------------------------------------------

class PythonPatternAnalyzer:
    """Recognise common Python aliasing patterns and produce
    :class:`AliasAnalysisResult` summaries without full analysis.

    The *func_body* argument in each method is expected to be a sequence
    (list) of statement-level AST-like dicts with at least a ``"kind"`` key.
    Supported statement kinds:

    * ``{"kind": "assign", "target": str, "value": ...}``
    * ``{"kind": "alloc", "target": str, "class": str, "site": str}``
    * ``{"kind": "store_attr", "obj": str, "attr": str, "value": str}``
    * ``{"kind": "load_attr", "result": str, "obj": str, "attr": str}``
    * ``{"kind": "call", "target": str, "func": str, "args": [str]}``
    * ``{"kind": "return", "value": str}``
    """

    def _extract_params(self, func_body: List[dict]) -> List[str]:
        """Heuristically determine parameter names from the function body.

        If a ``params`` key is present on the first statement, use it;
        otherwise look for variables used before being assigned.
        """
        if func_body and "params" in func_body[0]:
            return list(func_body[0]["params"])
        assigned: Set[str] = set()
        used_before_assign: List[str] = []
        for stmt in func_body:
            for key in ("value", "obj"):
                val = stmt.get(key)
                if isinstance(val, str) and val not in assigned:
                    if val not in used_before_assign:
                        used_before_assign.append(val)
            target = stmt.get("target")
            if isinstance(target, str):
                assigned.add(target)
        return used_before_assign

    def _collect_allocations(self, func_body: List[dict]) -> Set[HeapAddress]:
        """Return all heap addresses allocated in the body."""
        addrs: Set[HeapAddress] = set()
        for stmt in func_body:
            if stmt.get("kind") == "alloc":
                site = stmt.get("site", stmt.get("class", "unknown"))
                addrs.add(HeapAddress(site=site, context=()))
        return addrs

    def _collect_modified(self, func_body: List[dict]) -> Set[FieldPath]:
        """Return all field paths written to in the body."""
        paths: Set[FieldPath] = set()
        for stmt in func_body:
            if stmt.get("kind") == "store_attr":
                obj = stmt["obj"]
                attr = stmt["attr"]
                paths.add(FieldPath(base=obj, fields=(attr,)))
        return paths

    def _collect_escaping(self, func_body: List[dict],
                          params: List[str]) -> Set[str]:
        """Determine which variables escape via calls or stores.

        A variable escapes if it is passed as an argument to a call or
        stored into a heap field of another object.
        """
        escaping: Set[str] = set()
        for stmt in func_body:
            if stmt.get("kind") == "call":
                for a in stmt.get("args", []):
                    escaping.add(a)
            elif stmt.get("kind") == "store_attr":
                val = stmt.get("value")
                if isinstance(val, str):
                    escaping.add(val)
        return escaping

    def _collect_return_aliases(self, func_body: List[dict]) -> Set[str]:
        """Determine which variables the return value may alias."""
        aliases: Set[str] = set()
        for stmt in func_body:
            if stmt.get("kind") == "return":
                val = stmt.get("value")
                if isinstance(val, str):
                    aliases.add(val)
        return aliases

    # -- pattern-specific analysers ------------------------------------------

    def analyze_factory(self, func_body: List[dict]) -> AliasAnalysisResult:
        """Analyse a factory function.

        A factory typically allocates a fresh object, configures it via
        attribute stores, and returns it.  The return value does *not*
        alias any parameter because the object is freshly allocated.

        Escaping analysis is still performed for arguments forwarded into
        the new object's fields.
        """
        params = self._extract_params(func_body)
        allocated = self._collect_allocations(func_body)
        modified = self._collect_modified(func_body)
        escaping = self._collect_escaping(func_body, params)
        return_aliases = self._collect_return_aliases(func_body)

        # In a factory, the returned variable is typically freshly allocated
        # so it should NOT alias any parameter.  Filter return_aliases to
        # only include things that are actually parameters.
        alloc_targets: Set[str] = set()
        for stmt in func_body:
            if stmt.get("kind") == "alloc":
                t = stmt.get("target")
                if isinstance(t, str):
                    alloc_targets.add(t)

        # If the returned variable was freshly allocated, remove it from
        # the set of parameter aliases.
        filtered_return = set()
        for ra in return_aliases:
            if ra not in alloc_targets:
                filtered_return.add(ra)
        # But we keep the alloc targets as return aliases because the
        # return value IS the freshly allocated object.
        for ra in return_aliases:
            if ra in alloc_targets:
                filtered_return.add(ra)

        # Param aliases: params only alias each other if directly assigned
        param_aliases: Dict[str, Set[str]] = {p: set() for p in params}
        for stmt in func_body:
            if stmt.get("kind") == "assign":
                tgt = stmt.get("target")
                val = stmt.get("value")
                if tgt in params and val in params:
                    param_aliases.setdefault(tgt, set()).add(val)
                    param_aliases.setdefault(val, set()).add(tgt)

        return AliasAnalysisResult(
            param_aliases=param_aliases,
            return_aliases=return_aliases,
            escaping=escaping & set(params),
            modified=modified,
            allocated=allocated,
        )

    def analyze_builder(self, func_body: List[dict]) -> AliasAnalysisResult:
        """Analyse a builder-pattern method (returns ``self``).

        Builder methods store values into ``self``'s fields and return
        ``self``.  The return value therefore MUST alias the first
        parameter.
        """
        params = self._extract_params(func_body)
        allocated = self._collect_allocations(func_body)
        modified = self._collect_modified(func_body)
        escaping = self._collect_escaping(func_body, params)
        return_aliases = self._collect_return_aliases(func_body)

        # The self parameter (conventionally the first) is aliased by the
        # return value in a builder pattern.
        self_param = params[0] if params else "self"
        return_aliases.add(self_param)

        # All store_attr on self => modified paths rooted at self
        for stmt in func_body:
            if stmt.get("kind") == "store_attr" and stmt.get("obj") == self_param:
                modified.add(
                    FieldPath(base=self_param, fields=(stmt["attr"],))
                )

        param_aliases: Dict[str, Set[str]] = {p: set() for p in params}

        # Values stored into self's fields escape
        for stmt in func_body:
            if stmt.get("kind") == "store_attr" and stmt.get("obj") == self_param:
                val = stmt.get("value")
                if isinstance(val, str) and val in params:
                    escaping.add(val)

        return AliasAnalysisResult(
            param_aliases=param_aliases,
            return_aliases=return_aliases,
            escaping=escaping & set(params),
            modified=modified,
            allocated=allocated,
        )

    def analyze_dependency_injection(
        self, func_body: List[dict]
    ) -> AliasAnalysisResult:
        """Analyse a dependency-injection style function.

        In DI, the function receives dependencies as parameters, stores
        them into an object's fields, and possibly returns that object.
        All injected parameters escape because they are stored into the
        heap.
        """
        params = self._extract_params(func_body)
        allocated = self._collect_allocations(func_body)
        modified = self._collect_modified(func_body)
        return_aliases = self._collect_return_aliases(func_body)

        # Every parameter that appears as a store_attr value escapes
        escaping: Set[str] = set()
        for stmt in func_body:
            if stmt.get("kind") == "store_attr":
                val = stmt.get("value")
                if isinstance(val, str) and val in params:
                    escaping.add(val)

        # Parameters stored into the same object's fields may alias the
        # receiver indirectly.
        param_aliases: Dict[str, Set[str]] = {p: set() for p in params}
        obj_to_stored_params: Dict[str, List[str]] = {}
        for stmt in func_body:
            if stmt.get("kind") == "store_attr":
                obj = stmt.get("obj")
                val = stmt.get("value")
                if isinstance(obj, str) and isinstance(val, str) and val in params:
                    obj_to_stored_params.setdefault(obj, []).append(val)

        # Parameters stored into the same target object are "co-stored"
        # – they don't alias each other, but they are all reachable from
        # the same root and thus considered indirectly related.
        for obj, stored in obj_to_stored_params.items():
            for i, p1 in enumerate(stored):
                for p2 in stored[i + 1 :]:
                    param_aliases.setdefault(p1, set()).add(p2)
                    param_aliases.setdefault(p2, set()).add(p1)

        return AliasAnalysisResult(
            param_aliases=param_aliases,
            return_aliases=return_aliases,
            escaping=escaping,
            modified=modified,
            allocated=allocated,
        )

    def is_pure_function(self, func_body: List[dict],
                         alias_result: AliasAnalysisResult) -> bool:
        """Return ``True`` if the function performs no observable mutations
        and no parameter escapes.

        A function is considered pure when:
        1. No field paths are modified (``alias_result.modified`` is empty).
        2. No parameters escape (``alias_result.escaping`` is empty).
        3. No ``store_attr`` or ``call`` statements appear in the body
           (calls may have side effects).
        """
        if alias_result.modified:
            return False
        if alias_result.escaping:
            return False
        for stmt in func_body:
            kind = stmt.get("kind")
            if kind == "store_attr":
                return False
            if kind == "call":
                return False
        return True

from __future__ import annotations

"""
Escape analysis for heap objects in dynamically-typed programs.

Determines whether heap-allocated objects escape their defining scope,
enabling optimisations such as stack allocation, scalar replacement,
and synchronisation elimination in the refinement type inference pipeline.
"""

import time
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


# ---------------------------------------------------------------------------
# Local type stubs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Location:
    file: str = "<unknown>"
    line: int = 0
    column: int = 0


@dataclass(frozen=True)
class Instruction:
    opcode: str
    operands: Tuple[str, ...] = ()
    result: Optional[str] = None
    location: Location = field(default_factory=Location)


@dataclass
class BasicBlock:
    label: str
    instructions: List[Instruction] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)


@dataclass
class FunctionIR:
    name: str
    params: List[str] = field(default_factory=list)
    blocks: Dict[str, BasicBlock] = field(default_factory=dict)
    entry_block: str = "entry"
    closure_vars: List[str] = field(default_factory=list)

    def all_instructions(self) -> Iterator[Tuple[str, Instruction]]:
        for label, bb in self.blocks.items():
            for instr in bb.instructions:
                yield label, instr


@dataclass
class CFG:
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
#  EscapeState
# =========================================================================

class EscapeState(Enum):
    """Escape state of a heap-allocated object."""
    NO_ESCAPE = auto()        # object does not escape the creating method
    ARG_ESCAPE = auto()       # escapes via a method argument
    RETURN_ESCAPE = auto()    # escapes via return value
    EXCEPTION_ESCAPE = auto() # escapes via thrown exception
    GLOBAL_ESCAPE = auto()    # escapes to a global / module variable

    def join(self, other: EscapeState) -> EscapeState:
        return EscapeState(max(self.value, other.value))

    def leq(self, other: EscapeState) -> bool:
        return self.value <= other.value


# =========================================================================
#  EscapeAbstraction & EscapeLattice
# =========================================================================

@dataclass(frozen=True)
class EscapeAbstraction:
    """Abstract value tracking escape state plus reason."""
    state: EscapeState = EscapeState.NO_ESCAPE
    reason: str = ""
    via: Optional[str] = None  # function / variable through which it escapes

    def join(self, other: EscapeAbstraction) -> EscapeAbstraction:
        if self.state.value >= other.state.value:
            return self
        return other

    def leq(self, other: EscapeAbstraction) -> bool:
        return self.state.leq(other.state)

    def escapes(self) -> bool:
        return self.state != EscapeState.NO_ESCAPE


@dataclass
class EscapeLattice:
    """Lattice for escape states."""

    @staticmethod
    def bottom() -> EscapeAbstraction:
        return EscapeAbstraction(state=EscapeState.NO_ESCAPE)

    @staticmethod
    def top() -> EscapeAbstraction:
        return EscapeAbstraction(state=EscapeState.GLOBAL_ESCAPE, reason="top")

    @staticmethod
    def join(a: EscapeAbstraction, b: EscapeAbstraction) -> EscapeAbstraction:
        return a.join(b)

    @staticmethod
    def leq(a: EscapeAbstraction, b: EscapeAbstraction) -> bool:
        return a.leq(b)


# =========================================================================
#  Connection Graph nodes and edges
# =========================================================================

class EdgeKind(Enum):
    """Kind of edge in the connection graph."""
    POINTS_TO = auto()
    DEFERRED = auto()
    FIELD = auto()


@dataclass(frozen=True)
class ReferenceEdge:
    """Represents a reference between objects."""
    src: str
    dst: str
    kind: EdgeKind = EdgeKind.POINTS_TO
    field_name: Optional[str] = None

    def __str__(self) -> str:
        label = self.field_name if self.field_name else self.kind.name
        return f"{self.src} --{label}--> {self.dst}"


@dataclass
class ObjectNode:
    """Represents a heap-allocated object in the connection graph."""
    name: str
    allocation_site: str = ""
    type_name: str = "object"
    escape: EscapeAbstraction = field(default_factory=EscapeAbstraction)
    fields: Dict[str, str] = field(default_factory=dict)
    is_phantom: bool = False
    is_global: bool = False
    location: Location = field(default_factory=Location)

    def set_escape(self, state: EscapeState, reason: str = "", via: Optional[str] = None) -> None:
        new = EscapeAbstraction(state=state, reason=reason, via=via)
        self.escape = self.escape.join(new)


@dataclass
class PhantomObject(ObjectNode):
    """Represents objects created outside the analyzed scope."""
    is_phantom: bool = True


@dataclass
class GlobalObject(ObjectNode):
    """Represents global/module-level objects."""
    is_global: bool = True
    escape: EscapeAbstraction = field(
        default_factory=lambda: EscapeAbstraction(state=EscapeState.GLOBAL_ESCAPE, reason="global")
    )


# =========================================================================
#  ConnectionGraph
# =========================================================================

@dataclass
class ConnectionGraph:
    """Connection graph for escape analysis (Choi et al.)."""

    _objects: Dict[str, ObjectNode] = field(default_factory=dict)
    _edges: List[ReferenceEdge] = field(default_factory=list)
    _fwd: Dict[str, List[ReferenceEdge]] = field(default_factory=lambda: defaultdict(list))
    _bwd: Dict[str, List[ReferenceEdge]] = field(default_factory=lambda: defaultdict(list))
    _var_points_to: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_object(self, obj: ObjectNode) -> None:
        self._objects[obj.name] = obj

    def get_object(self, name: str) -> Optional[ObjectNode]:
        return self._objects.get(name)

    def add_edge(self, edge: ReferenceEdge) -> None:
        self._edges.append(edge)
        self._fwd[edge.src].append(edge)
        self._bwd[edge.dst].append(edge)

    def add_points_to(self, var: str, obj_name: str) -> None:
        self._var_points_to[var].add(obj_name)
        self.add_edge(ReferenceEdge(src=var, dst=obj_name, kind=EdgeKind.POINTS_TO))

    def add_field_edge(self, base: str, field_name: str, target: str) -> None:
        self.add_edge(ReferenceEdge(
            src=base, dst=target, kind=EdgeKind.FIELD, field_name=field_name,
        ))
        obj = self._objects.get(base)
        if obj:
            obj.fields[field_name] = target

    def add_deferred(self, src: str, dst: str) -> None:
        self.add_edge(ReferenceEdge(src=src, dst=dst, kind=EdgeKind.DEFERRED))

    def points_to(self, var: str) -> Set[str]:
        return set(self._var_points_to.get(var, set()))

    def field_targets(self, obj_name: str, field_name: str) -> Set[str]:
        targets: set[str] = set()
        for edge in self._fwd.get(obj_name, []):
            if edge.kind == EdgeKind.FIELD and edge.field_name == field_name:
                targets.add(edge.dst)
        return targets

    def successors(self, name: str) -> Set[str]:
        return {e.dst for e in self._fwd.get(name, [])}

    def predecessors(self, name: str) -> Set[str]:
        return {e.src for e in self._bwd.get(name, [])}

    def all_objects(self) -> List[ObjectNode]:
        return list(self._objects.values())

    def all_edges(self) -> List[ReferenceEdge]:
        return list(self._edges)

    def reachable_from(self, roots: Set[str]) -> Set[str]:
        visited: set[str] = set()
        stack = list(roots)
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for edge in self._fwd.get(node, []):
                if edge.dst not in visited:
                    stack.append(edge.dst)
        return visited

    def propagate_escape(self) -> None:
        """Propagate escape states through the connection graph."""
        changed = True
        while changed:
            changed = False
            for edge in self._edges:
                src_obj = self._objects.get(edge.src)
                dst_obj = self._objects.get(edge.dst)
                if src_obj and dst_obj:
                    old_state = dst_obj.escape
                    dst_obj.escape = dst_obj.escape.join(src_obj.escape)
                    if dst_obj.escape != old_state:
                        changed = True
                elif src_obj and not dst_obj:
                    pass
                elif dst_obj and not src_obj:
                    pass
            # Backward propagation: if a pointed-to object escapes, the pointer escapes
            for edge in self._edges:
                if edge.kind == EdgeKind.POINTS_TO:
                    dst_obj = self._objects.get(edge.dst)
                    src_obj = self._objects.get(edge.src)
                    if dst_obj and dst_obj.escape.escapes() and src_obj:
                        old = src_obj.escape
                        src_obj.escape = src_obj.escape.join(dst_obj.escape)
                        if src_obj.escape != old:
                            changed = True

    def merge(self, other: ConnectionGraph) -> ConnectionGraph:
        result = ConnectionGraph()
        for obj in self.all_objects():
            result.add_object(obj)
        for obj in other.all_objects():
            existing = result.get_object(obj.name)
            if existing:
                existing.escape = existing.escape.join(obj.escape)
            else:
                result.add_object(obj)
        for edge in self.all_edges():
            result.add_edge(edge)
        for edge in other.all_edges():
            result.add_edge(edge)
        for var, pts in self._var_points_to.items():
            for p in pts:
                result._var_points_to[var].add(p)
        for var, pts in other._var_points_to.items():
            for p in pts:
                result._var_points_to[var].add(p)
        return result


# =========================================================================
#  AllocationSiteAnalysis
# =========================================================================

@dataclass
class AllocationSite:
    """An allocation site in the program."""
    function: str
    block: str
    index: int
    type_name: str = "object"
    location: Location = field(default_factory=Location)
    variable: str = ""

    @property
    def id(self) -> str:
        return f"{self.function}:{self.block}:{self.index}"


@dataclass
class AllocationSiteAnalysis:
    """Tracks allocation sites in the program."""
    _sites: Dict[str, AllocationSite] = field(default_factory=dict)

    def analyze(self, func: FunctionIR) -> List[AllocationSite]:
        sites: list[AllocationSite] = []
        for block_label, bb in func.blocks.items():
            for i, instr in enumerate(bb.instructions):
                if instr.opcode in ("alloc", "new", "list_new", "dict_new", "set_new", "tuple_new"):
                    type_name = "object"
                    if instr.opcode == "list_new":
                        type_name = "list"
                    elif instr.opcode == "dict_new":
                        type_name = "dict"
                    elif instr.opcode == "set_new":
                        type_name = "set"
                    elif instr.opcode == "tuple_new":
                        type_name = "tuple"
                    elif instr.operands:
                        type_name = instr.operands[0]

                    site = AllocationSite(
                        function=func.name,
                        block=block_label,
                        index=i,
                        type_name=type_name,
                        location=instr.location,
                        variable=instr.result or "",
                    )
                    sites.append(site)
                    self._sites[site.id] = site
        return sites

    def get(self, site_id: str) -> Optional[AllocationSite]:
        return self._sites.get(site_id)

    def all_sites(self) -> List[AllocationSite]:
        return list(self._sites.values())

    def sites_in(self, function: str) -> List[AllocationSite]:
        return [s for s in self._sites.values() if s.function == function]


# =========================================================================
#  FieldSensitiveEscape
# =========================================================================

@dataclass
class FieldSensitiveEscape:
    """Field-sensitive escape tracking."""
    _field_escape: Dict[Tuple[str, str], EscapeAbstraction] = field(default_factory=dict)

    def set_field_escape(self, obj: str, field_name: str, state: EscapeAbstraction) -> None:
        key = (obj, field_name)
        old = self._field_escape.get(key, EscapeLattice.bottom())
        self._field_escape[key] = old.join(state)

    def get_field_escape(self, obj: str, field_name: str) -> EscapeAbstraction:
        return self._field_escape.get((obj, field_name), EscapeLattice.bottom())

    def fields_escaping(self, obj: str) -> Dict[str, EscapeAbstraction]:
        result: dict[str, EscapeAbstraction] = {}
        for (o, f), state in self._field_escape.items():
            if o == obj and state.escapes():
                result[f] = state
        return result

    def analyze(self, func: FunctionIR, graph: ConnectionGraph) -> None:
        for _, instr in func.all_instructions():
            if instr.opcode == "setattr" and len(instr.operands) >= 3:
                base = instr.operands[0]
                field_name = instr.operands[1]
                value = instr.operands[2]
                for obj_name in graph.points_to(base):
                    obj = graph.get_object(obj_name)
                    if obj:
                        for val_target in graph.points_to(value):
                            val_obj = graph.get_object(val_target)
                            if val_obj and val_obj.escape.escapes():
                                self.set_field_escape(
                                    obj_name, field_name, val_obj.escape,
                                )


# =========================================================================
#  ArrayElementEscape
# =========================================================================

@dataclass
class ArrayElementEscape:
    """Tracks escape of array/list elements."""
    _element_escape: Dict[str, EscapeAbstraction] = field(default_factory=dict)

    def record_store(self, array: str, element: str, state: EscapeAbstraction) -> None:
        old = self._element_escape.get(array, EscapeLattice.bottom())
        self._element_escape[array] = old.join(state)

    def get_escape(self, array: str) -> EscapeAbstraction:
        return self._element_escape.get(array, EscapeLattice.bottom())

    def analyze(self, func: FunctionIR, graph: ConnectionGraph) -> None:
        for _, instr in func.all_instructions():
            if instr.opcode in ("setitem", "append") and instr.operands:
                container = instr.operands[0]
                for obj_name in graph.points_to(container):
                    obj = graph.get_object(obj_name)
                    if obj:
                        val = instr.operands[-1] if len(instr.operands) > 1 else container
                        for val_target in graph.points_to(val):
                            val_obj = graph.get_object(val_target)
                            if val_obj:
                                self.record_store(obj_name, val_target, val_obj.escape)


# =========================================================================
#  ClosureCaptureEscape
# =========================================================================

@dataclass
class ClosureCaptureEscape:
    """Tracks escape via closure capture."""
    _captured: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _escape_state: Dict[str, EscapeAbstraction] = field(default_factory=dict)

    def analyze(self, func: FunctionIR, graph: ConnectionGraph) -> Dict[str, EscapeAbstraction]:
        result: dict[str, EscapeAbstraction] = {}
        for var in func.closure_vars:
            self._captured[func.name].add(var)
            for obj_name in graph.points_to(var):
                obj = graph.get_object(obj_name)
                if obj:
                    esc = EscapeAbstraction(
                        state=EscapeState.ARG_ESCAPE,
                        reason=f"captured by closure {func.name}",
                        via=func.name,
                    )
                    obj.set_escape(esc.state, esc.reason, esc.via)
                    result[obj_name] = obj.escape
                    self._escape_state[var] = esc
        return result

    def is_captured(self, func_name: str, var: str) -> bool:
        return var in self._captured.get(func_name, set())


# =========================================================================
#  ReturnEscape
# =========================================================================

@dataclass
class ReturnEscape:
    """Tracks objects escaping via return values."""
    _return_escapes: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def analyze(self, func: FunctionIR, graph: ConnectionGraph) -> Set[str]:
        escaping: set[str] = set()
        for _, instr in func.all_instructions():
            if instr.opcode == "return" and instr.operands:
                ret_var = instr.operands[0]
                for obj_name in graph.points_to(ret_var):
                    obj = graph.get_object(obj_name)
                    if obj:
                        obj.set_escape(
                            EscapeState.RETURN_ESCAPE,
                            f"returned from {func.name}",
                            func.name,
                        )
                        escaping.add(obj_name)
                        # Also mark transitively reachable objects
                        reachable = graph.reachable_from({obj_name})
                        for r in reachable:
                            r_obj = graph.get_object(r)
                            if r_obj:
                                r_obj.set_escape(
                                    EscapeState.RETURN_ESCAPE,
                                    f"reachable from return in {func.name}",
                                    func.name,
                                )
                                escaping.add(r)
        self._return_escapes[func.name] = escaping
        return escaping

    def escaping_from(self, func_name: str) -> Set[str]:
        return set(self._return_escapes.get(func_name, set()))


# =========================================================================
#  ExceptionEscapeAnalysis
# =========================================================================

@dataclass
class ExceptionEscapeAnalysis:
    """Tracks objects escaping via exception raising."""
    _exception_escapes: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def analyze(self, func: FunctionIR, graph: ConnectionGraph) -> Set[str]:
        escaping: set[str] = set()
        for _, instr in func.all_instructions():
            if instr.opcode in ("raise", "throw") and instr.operands:
                exc_var = instr.operands[0]
                for obj_name in graph.points_to(exc_var):
                    obj = graph.get_object(obj_name)
                    if obj:
                        obj.set_escape(
                            EscapeState.EXCEPTION_ESCAPE,
                            f"raised in {func.name}",
                            func.name,
                        )
                        escaping.add(obj_name)
        self._exception_escapes[func.name] = escaping
        return escaping


# =========================================================================
#  CalleeEscapeAnalysis
# =========================================================================

@dataclass
class CalleeEscapeAnalysis:
    """Tracks escape via function arguments."""
    _arg_escapes: Dict[str, Dict[int, Set[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set))
    )

    def analyze(self, func: FunctionIR, graph: ConnectionGraph) -> None:
        for _, instr in func.all_instructions():
            if instr.opcode == "call" and instr.operands:
                callee = instr.operands[0]
                for i, arg in enumerate(instr.operands[1:]):
                    for obj_name in graph.points_to(arg):
                        obj = graph.get_object(obj_name)
                        if obj:
                            obj.set_escape(
                                EscapeState.ARG_ESCAPE,
                                f"passed to {callee} as arg {i}",
                                callee,
                            )
                            self._arg_escapes[callee][i].add(obj_name)

    def escaping_to(self, callee: str, arg_index: int) -> Set[str]:
        return set(self._arg_escapes.get(callee, {}).get(arg_index, set()))


# =========================================================================
#  ThreadEscapeAnalysis
# =========================================================================

@dataclass
class ThreadEscapeAnalysis:
    """Tracks escape to other threads (async/await, threading)."""
    _thread_escapes: Set[str] = field(default_factory=set)
    _async_ops: Set[str] = field(
        default_factory=lambda: {
            "async_call", "create_task", "submit", "start_thread",
            "spawn", "asyncio.create_task", "threading.Thread",
        }
    )

    def analyze(self, func: FunctionIR, graph: ConnectionGraph) -> Set[str]:
        escaping: set[str] = set()
        for _, instr in func.all_instructions():
            if instr.opcode == "call" and instr.operands and instr.operands[0] in self._async_ops:
                for arg in instr.operands[1:]:
                    for obj_name in graph.points_to(arg):
                        obj = graph.get_object(obj_name)
                        if obj:
                            obj.set_escape(
                                EscapeState.GLOBAL_ESCAPE,
                                f"passed to async/thread op {instr.operands[0]}",
                                instr.operands[0],
                            )
                            escaping.add(obj_name)
            elif instr.opcode in ("await", "yield"):
                if instr.operands:
                    for obj_name in graph.points_to(instr.operands[0]):
                        obj = graph.get_object(obj_name)
                        if obj:
                            obj.set_escape(
                                EscapeState.GLOBAL_ESCAPE,
                                f"awaited/yielded in {func.name}",
                                func.name,
                            )
                            escaping.add(obj_name)
        self._thread_escapes |= escaping
        return escaping


# =========================================================================
#  ScopeAnalysis
# =========================================================================

@dataclass
class VariableScope:
    """Scope information for a variable."""
    name: str
    function: str
    block: str = ""
    is_local: bool = True
    is_parameter: bool = False
    is_closure: bool = False
    is_global: bool = False
    first_def: int = 0
    last_use: int = 0


@dataclass
class ScopeAnalysis:
    """Determines variable scopes and lifetimes."""
    _scopes: Dict[str, VariableScope] = field(default_factory=dict)

    def analyze(self, func: FunctionIR) -> Dict[str, VariableScope]:
        self._scopes.clear()
        # Parameters
        for param in func.params:
            self._scopes[param] = VariableScope(
                name=param,
                function=func.name,
                is_parameter=True,
            )
        # Closure variables
        for cv in func.closure_vars:
            self._scopes[cv] = VariableScope(
                name=cv,
                function=func.name,
                is_closure=True,
                is_local=False,
            )
        # Local definitions and uses
        point = 0
        for block_label, bb in func.blocks.items():
            for instr in bb.instructions:
                point += 1
                if instr.result and instr.result not in self._scopes:
                    self._scopes[instr.result] = VariableScope(
                        name=instr.result,
                        function=func.name,
                        block=block_label,
                        first_def=point,
                    )
                if instr.result and instr.result in self._scopes:
                    self._scopes[instr.result].first_def = min(
                        self._scopes[instr.result].first_def or point, point,
                    )
                for op in instr.operands:
                    if op in self._scopes:
                        self._scopes[op].last_use = max(self._scopes[op].last_use, point)
        return dict(self._scopes)

    def is_local(self, var: str) -> bool:
        scope = self._scopes.get(var)
        return scope.is_local if scope else False

    def lifetime(self, var: str) -> Tuple[int, int]:
        scope = self._scopes.get(var)
        if scope:
            return (scope.first_def, scope.last_use)
        return (0, 0)


# =========================================================================
#  ObjectLifetime
# =========================================================================

@dataclass
class ObjectLifetime:
    """Computes object lifetimes based on allocation and last use."""
    _lifetimes: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def compute(self, func: FunctionIR, graph: ConnectionGraph) -> Dict[str, Tuple[int, int]]:
        self._lifetimes.clear()
        point = 0
        alloc_points: dict[str, int] = {}
        last_use_points: dict[str, int] = {}

        for _, instr in func.all_instructions():
            point += 1
            if instr.opcode in ("alloc", "new", "list_new", "dict_new") and instr.result:
                for obj_name in graph.points_to(instr.result):
                    alloc_points[obj_name] = point
            for op in instr.operands:
                for obj_name in graph.points_to(op):
                    last_use_points[obj_name] = max(
                        last_use_points.get(obj_name, 0), point,
                    )

        for obj in graph.all_objects():
            start = alloc_points.get(obj.name, 0)
            end = last_use_points.get(obj.name, start)
            self._lifetimes[obj.name] = (start, end)
        return dict(self._lifetimes)

    def get_lifetime(self, obj_name: str) -> Tuple[int, int]:
        return self._lifetimes.get(obj_name, (0, 0))

    def overlaps(self, a: str, b: str) -> bool:
        a_start, a_end = self.get_lifetime(a)
        b_start, b_end = self.get_lifetime(b)
        return a_start <= b_end and b_start <= a_end


# =========================================================================
#  StackAllocationCandidate
# =========================================================================

@dataclass
class StackAllocationCandidate:
    """Identifies objects that could be stack-allocated."""
    object_name: str
    allocation_site: str = ""
    type_name: str = "object"
    estimated_size: int = 0
    reason: str = ""

    def __str__(self) -> str:
        return f"StackCandidate({self.object_name}, type={self.type_name}, reason={self.reason})"


@dataclass
class StackAllocationAnalysis:
    """Identifies objects suitable for stack allocation."""
    max_stack_size: int = 256
    _candidates: List[StackAllocationCandidate] = field(default_factory=list)

    def find_candidates(self, graph: ConnectionGraph) -> List[StackAllocationCandidate]:
        self._candidates.clear()
        for obj in graph.all_objects():
            if obj.is_phantom or obj.is_global:
                continue
            if not obj.escape.escapes():
                candidate = StackAllocationCandidate(
                    object_name=obj.name,
                    allocation_site=obj.allocation_site,
                    type_name=obj.type_name,
                    reason="does not escape",
                )
                self._candidates.append(candidate)
            elif obj.escape.state == EscapeState.ARG_ESCAPE:
                # Could still stack-allocate if callee doesn't store it
                candidate = StackAllocationCandidate(
                    object_name=obj.name,
                    allocation_site=obj.allocation_site,
                    type_name=obj.type_name,
                    reason="arg-escape only (may be eligible)",
                )
                self._candidates.append(candidate)
        return list(self._candidates)


# =========================================================================
#  HeapAbstraction
# =========================================================================

@dataclass
class HeapAbstraction:
    """Abstracts the heap for analysis (allocation-site abstraction)."""
    _objects_by_site: Dict[str, ObjectNode] = field(default_factory=dict)

    def get_or_create(self, site: AllocationSite) -> ObjectNode:
        if site.id in self._objects_by_site:
            return self._objects_by_site[site.id]
        obj = ObjectNode(
            name=f"obj_{site.id}",
            allocation_site=site.id,
            type_name=site.type_name,
            location=site.location,
        )
        self._objects_by_site[site.id] = obj
        return obj

    def all_objects(self) -> List[ObjectNode]:
        return list(self._objects_by_site.values())


# =========================================================================
#  PointsToGraph / FieldPointsTo / AllocationPointsTo
# =========================================================================

@dataclass
class PointsToGraph:
    """Points-to graph representation."""
    _pts: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, var: str, obj: str) -> None:
        self._pts[var].add(obj)

    def points_to(self, var: str) -> Set[str]:
        return set(self._pts.get(var, set()))

    def variables(self) -> Set[str]:
        return set(self._pts.keys())

    def merge(self, other: PointsToGraph) -> PointsToGraph:
        result = PointsToGraph()
        for var, objs in self._pts.items():
            for obj in objs:
                result.add(var, obj)
        for var, objs in other._pts.items():
            for obj in objs:
                result.add(var, obj)
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointsToGraph):
            return NotImplemented
        return self._pts == other._pts


@dataclass
class FieldPointsTo:
    """Field-sensitive points-to."""
    _pts: Dict[Tuple[str, str], Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, base: str, field_name: str, target: str) -> None:
        self._pts[(base, field_name)].add(target)

    def points_to(self, base: str, field_name: str) -> Set[str]:
        return set(self._pts.get((base, field_name), set()))

    def all_fields(self, base: str) -> Dict[str, Set[str]]:
        result: dict[str, set[str]] = {}
        for (b, f), targets in self._pts.items():
            if b == base:
                result[f] = set(targets)
        return result


@dataclass
class AllocationPointsTo:
    """Allocation-site based points-to."""
    _pts: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, var: str, alloc_site: str) -> None:
        self._pts[var].add(alloc_site)

    def sites_of(self, var: str) -> Set[str]:
        return set(self._pts.get(var, set()))


# =========================================================================
#  EscapeSummary / EscapeSummaryTable
# =========================================================================

@dataclass
class EscapeSummary:
    """Per-function escape summary."""
    function: str
    object_states: Dict[str, EscapeAbstraction] = field(default_factory=dict)
    param_escape: Dict[int, EscapeAbstraction] = field(default_factory=dict)
    return_escape: EscapeAbstraction = field(default_factory=EscapeAbstraction)
    allocations_not_escaping: Set[str] = field(default_factory=set)
    version: int = 0

    def set_param_escape(self, index: int, state: EscapeAbstraction) -> None:
        old = self.param_escape.get(index, EscapeLattice.bottom())
        self.param_escape[index] = old.join(state)

    def join(self, other: EscapeSummary) -> EscapeSummary:
        merged_objects: dict[str, EscapeAbstraction] = {}
        for k in set(self.object_states) | set(other.object_states):
            a = self.object_states.get(k, EscapeLattice.bottom())
            b = other.object_states.get(k, EscapeLattice.bottom())
            merged_objects[k] = a.join(b)
        merged_params: dict[int, EscapeAbstraction] = {}
        for k in set(self.param_escape) | set(other.param_escape):
            a = self.param_escape.get(k, EscapeLattice.bottom())
            b = other.param_escape.get(k, EscapeLattice.bottom())
            merged_params[k] = a.join(b)
        return EscapeSummary(
            function=self.function,
            object_states=merged_objects,
            param_escape=merged_params,
            return_escape=self.return_escape.join(other.return_escape),
            allocations_not_escaping=self.allocations_not_escaping & other.allocations_not_escaping,
            version=max(self.version, other.version) + 1,
        )

    def subsumes(self, other: EscapeSummary) -> bool:
        for k, v in other.object_states.items():
            mine = self.object_states.get(k, EscapeLattice.bottom())
            if not v.leq(mine):
                return False
        return True


@dataclass
class EscapeSummaryTable:
    """Stores escape summaries per function."""
    _summaries: Dict[str, EscapeSummary] = field(default_factory=dict)

    def get(self, function: str) -> Optional[EscapeSummary]:
        return self._summaries.get(function)

    def put(self, summary: EscapeSummary) -> bool:
        old = self._summaries.get(summary.function)
        if old and old.subsumes(summary):
            return False
        self._summaries[summary.function] = summary
        return True

    def all_summaries(self) -> List[EscapeSummary]:
        return list(self._summaries.values())


# =========================================================================
#  InterproceduralEscape
# =========================================================================

@dataclass
class InterproceduralEscape:
    """Interprocedural escape analysis across function boundaries."""
    summary_table: EscapeSummaryTable = field(default_factory=EscapeSummaryTable)
    _call_graph_fwd: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def set_call_graph(self, forward: Dict[str, Set[str]]) -> None:
        self._call_graph_fwd = defaultdict(set, forward)

    def analyze(
        self,
        functions: Dict[str, FunctionIR],
        local_graphs: Dict[str, ConnectionGraph],
    ) -> EscapeSummaryTable:
        """Analyze all functions bottom-up."""
        sccs = self._compute_sccs(set(functions.keys()))
        for scc in reversed(sccs):
            self._analyze_scc(scc, functions, local_graphs)
        return self.summary_table

    def _analyze_scc(
        self,
        scc: List[str],
        functions: Dict[str, FunctionIR],
        local_graphs: Dict[str, ConnectionGraph],
    ) -> None:
        changed = True
        iteration = 0
        max_iter = 100
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            for func_name in scc:
                func = functions.get(func_name)
                graph = local_graphs.get(func_name)
                if func is None or graph is None:
                    continue
                summary = self._analyze_function(func, graph)
                if self.summary_table.put(summary):
                    changed = True

    def _analyze_function(
        self, func: FunctionIR, graph: ConnectionGraph,
    ) -> EscapeSummary:
        summary = EscapeSummary(function=func.name)

        # Apply callee summaries
        for _, instr in func.all_instructions():
            if instr.opcode == "call" and instr.operands:
                callee = instr.operands[0]
                callee_summary = self.summary_table.get(callee)
                if callee_summary:
                    for i, arg in enumerate(instr.operands[1:]):
                        param_esc = callee_summary.param_escape.get(i)
                        if param_esc and param_esc.escapes():
                            for obj_name in graph.points_to(arg):
                                obj = graph.get_object(obj_name)
                                if obj:
                                    obj.set_escape(param_esc.state, param_esc.reason, param_esc.via)

        graph.propagate_escape()

        for obj in graph.all_objects():
            summary.object_states[obj.name] = obj.escape
            if not obj.escape.escapes():
                summary.allocations_not_escaping.add(obj.name)

        for i, param in enumerate(func.params):
            for obj_name in graph.points_to(param):
                obj = graph.get_object(obj_name)
                if obj:
                    summary.set_param_escape(i, obj.escape)

        return summary

    def _compute_sccs(self, nodes: Set[str]) -> List[List[str]]:
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
            for w in self._call_graph_fwd.get(v, set()):
                if w not in nodes:
                    continue
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

        for node in nodes:
            if node not in index_map:
                strongconnect(node)
        return result


# =========================================================================
#  EscapeOptimizationHints
# =========================================================================

@dataclass
class EscapeOptimizationHints:
    """Optimisation hints from escape analysis."""
    scalar_replacement: List[str] = field(default_factory=list)
    stack_allocation: List[StackAllocationCandidate] = field(default_factory=list)
    sync_elimination: List[str] = field(default_factory=list)
    lock_elision: List[str] = field(default_factory=list)

    def compute(self, graph: ConnectionGraph) -> None:
        self.scalar_replacement.clear()
        self.stack_allocation.clear()
        self.sync_elimination.clear()

        for obj in graph.all_objects():
            if obj.is_phantom or obj.is_global:
                continue
            if not obj.escape.escapes():
                self.stack_allocation.append(StackAllocationCandidate(
                    object_name=obj.name,
                    type_name=obj.type_name,
                    reason="no escape",
                ))
                if not obj.fields:
                    self.scalar_replacement.append(obj.name)
                elif len(obj.fields) <= 4:
                    self.scalar_replacement.append(obj.name)
                self.sync_elimination.append(obj.name)

    def has_optimizations(self) -> bool:
        return bool(self.scalar_replacement or self.stack_allocation or self.sync_elimination)

    def summary(self) -> str:
        return (
            f"Hints: {len(self.scalar_replacement)} scalar replacements, "
            f"{len(self.stack_allocation)} stack allocations, "
            f"{len(self.sync_elimination)} sync eliminations"
        )


# =========================================================================
#  MutableCaptureAnalysis
# =========================================================================

@dataclass
class MutableCaptureAnalysis:
    """Tracks whether captured variables are mutated."""
    _mutated: Dict[str, bool] = field(default_factory=dict)

    def analyze(self, func: FunctionIR) -> Dict[str, bool]:
        self._mutated.clear()
        for var in func.closure_vars:
            self._mutated[var] = False

        for _, instr in func.all_instructions():
            if instr.opcode in ("store", "assign", "setattr"):
                if instr.operands:
                    target = instr.operands[0]
                    if target in self._mutated:
                        self._mutated[target] = True
            if instr.result and instr.result in self._mutated:
                self._mutated[instr.result] = True
        return dict(self._mutated)

    def is_mutated(self, var: str) -> bool:
        return self._mutated.get(var, False)

    def immutable_captures(self) -> Set[str]:
        return {v for v, m in self._mutated.items() if not m}


# =========================================================================
#  AliasSet / MayAliasAnalysis / MustAliasAnalysis
# =========================================================================

@dataclass
class AliasSet:
    """Set of potentially aliased variables."""
    variables: Set[str] = field(default_factory=set)
    representative: str = ""

    def add(self, var: str) -> None:
        self.variables.add(var)
        if not self.representative:
            self.representative = var

    def contains(self, var: str) -> bool:
        return var in self.variables

    def merge(self, other: AliasSet) -> AliasSet:
        merged = AliasSet(
            variables=self.variables | other.variables,
            representative=self.representative or other.representative,
        )
        return merged

    def size(self) -> int:
        return len(self.variables)


@dataclass
class MayAliasAnalysis:
    """May-alias analysis using escape graph."""
    _alias_sets: Dict[str, AliasSet] = field(default_factory=dict)
    _parent: Dict[str, str] = field(default_factory=dict)

    def analyze(self, graph: ConnectionGraph) -> None:
        self._alias_sets.clear()
        self._parent.clear()

        for var in graph._var_points_to:
            self._parent[var] = var
            self._alias_sets[var] = AliasSet(variables={var}, representative=var)

        # Variables pointing to the same object may alias
        obj_to_vars: dict[str, list[str]] = defaultdict(list)
        for var, objs in graph._var_points_to.items():
            for obj in objs:
                obj_to_vars[obj].append(var)

        for obj, vars_list in obj_to_vars.items():
            if len(vars_list) > 1:
                root = vars_list[0]
                for other in vars_list[1:]:
                    self._union(root, other)

    def _find(self, x: str) -> str:
        while self._parent.get(x, x) != x:
            self._parent[x] = self._parent.get(self._parent[x], self._parent[x])
            x = self._parent[x]
        return x

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[rb] = ra
            sa = self._alias_sets.get(ra, AliasSet())
            sb = self._alias_sets.get(rb, AliasSet())
            self._alias_sets[ra] = sa.merge(sb)

    def may_alias(self, a: str, b: str) -> bool:
        if a not in self._parent or b not in self._parent:
            return a == b
        return self._find(a) == self._find(b)

    def alias_set_of(self, var: str) -> AliasSet:
        root = self._find(var)
        return self._alias_sets.get(root, AliasSet(variables={var}, representative=var))


@dataclass
class MustAliasAnalysis:
    """Must-alias analysis."""
    _must_alias: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def analyze(self, graph: ConnectionGraph) -> None:
        self._must_alias.clear()
        for var, objs in graph._var_points_to.items():
            if len(objs) == 1:
                obj = next(iter(objs))
                for other_var, other_objs in graph._var_points_to.items():
                    if other_var != var and other_objs == {obj}:
                        self._must_alias[var].add(other_var)
                        self._must_alias[other_var].add(var)

    def must_alias(self, a: str, b: str) -> bool:
        return b in self._must_alias.get(a, set())


# =========================================================================
#  EscapeStatistics
# =========================================================================

@dataclass
class EscapeStatistics:
    """Analysis statistics."""
    total_objects: int = 0
    no_escape: int = 0
    arg_escape: int = 0
    return_escape: int = 0
    exception_escape: int = 0
    global_escape: int = 0
    stack_candidates: int = 0
    scalar_candidates: int = 0
    elapsed: float = 0.0

    def compute(self, graph: ConnectionGraph, hints: EscapeOptimizationHints) -> None:
        self.total_objects = len(graph.all_objects())
        self.no_escape = sum(1 for o in graph.all_objects() if o.escape.state == EscapeState.NO_ESCAPE)
        self.arg_escape = sum(1 for o in graph.all_objects() if o.escape.state == EscapeState.ARG_ESCAPE)
        self.return_escape = sum(1 for o in graph.all_objects() if o.escape.state == EscapeState.RETURN_ESCAPE)
        self.exception_escape = sum(1 for o in graph.all_objects() if o.escape.state == EscapeState.EXCEPTION_ESCAPE)
        self.global_escape = sum(1 for o in graph.all_objects() if o.escape.state == EscapeState.GLOBAL_ESCAPE)
        self.stack_candidates = len(hints.stack_allocation)
        self.scalar_candidates = len(hints.scalar_replacement)

    def __str__(self) -> str:
        return (
            f"EscapeStats(total={self.total_objects}, no_escape={self.no_escape}, "
            f"arg={self.arg_escape}, return={self.return_escape}, "
            f"exception={self.exception_escape}, global={self.global_escape}, "
            f"stack_candidates={self.stack_candidates}, elapsed={self.elapsed:.3f}s)"
        )


# =========================================================================
#  EscapeAnalysis (main driver)
# =========================================================================

@dataclass
class EscapeAnalysis:
    """Main escape analysis driver.

    Coordinates allocation-site analysis, connection-graph construction,
    intra- and interprocedural escape propagation, and optimisation hint
    generation for a set of functions.
    """

    allocation_analysis: AllocationSiteAnalysis = field(default_factory=AllocationSiteAnalysis)
    heap: HeapAbstraction = field(default_factory=HeapAbstraction)
    field_sensitive: FieldSensitiveEscape = field(default_factory=FieldSensitiveEscape)
    array_escape: ArrayElementEscape = field(default_factory=ArrayElementEscape)
    closure_escape: ClosureCaptureEscape = field(default_factory=ClosureCaptureEscape)
    return_escape_analysis: ReturnEscape = field(default_factory=ReturnEscape)
    exception_escape: ExceptionEscapeAnalysis = field(default_factory=ExceptionEscapeAnalysis)
    callee_escape: CalleeEscapeAnalysis = field(default_factory=CalleeEscapeAnalysis)
    thread_escape: ThreadEscapeAnalysis = field(default_factory=ThreadEscapeAnalysis)
    scope_analysis: ScopeAnalysis = field(default_factory=ScopeAnalysis)
    mutable_capture: MutableCaptureAnalysis = field(default_factory=MutableCaptureAnalysis)
    may_alias: MayAliasAnalysis = field(default_factory=MayAliasAnalysis)
    must_alias: MustAliasAnalysis = field(default_factory=MustAliasAnalysis)
    interprocedural: InterproceduralEscape = field(default_factory=InterproceduralEscape)
    statistics: EscapeStatistics = field(default_factory=EscapeStatistics)

    _graphs: Dict[str, ConnectionGraph] = field(default_factory=dict)
    _hints: Dict[str, EscapeOptimizationHints] = field(default_factory=dict)

    def analyze_function(self, func: FunctionIR) -> ConnectionGraph:
        """Analyse a single function and return its connection graph."""
        start = time.time()
        graph = ConnectionGraph()

        # 1. Find allocation sites and create object nodes
        sites = self.allocation_analysis.analyze(func)
        for site in sites:
            obj = self.heap.get_or_create(site)
            graph.add_object(obj)
            if site.variable:
                graph.add_points_to(site.variable, obj.name)

        # 2. Build connection graph from instructions
        self._build_graph(func, graph)

        # 3. Run sub-analyses
        self.scope_analysis.analyze(func)
        self.field_sensitive.analyze(func, graph)
        self.array_escape.analyze(func, graph)
        self.closure_escape.analyze(func, graph)
        self.return_escape_analysis.analyze(func, graph)
        self.exception_escape.analyze(func, graph)
        self.callee_escape.analyze(func, graph)
        self.thread_escape.analyze(func, graph)
        self.mutable_capture.analyze(func)

        # 4. Propagate escape states
        graph.propagate_escape()

        # 5. Alias analysis
        self.may_alias.analyze(graph)
        self.must_alias.analyze(graph)

        # 6. Optimisation hints
        hints = EscapeOptimizationHints()
        hints.compute(graph)
        self._hints[func.name] = hints

        self._graphs[func.name] = graph
        self.statistics.elapsed += time.time() - start
        return graph

    def analyze_all(
        self,
        functions: Dict[str, FunctionIR],
        call_graph_fwd: Optional[Dict[str, Set[str]]] = None,
    ) -> Dict[str, ConnectionGraph]:
        """Analyse all functions, optionally with interprocedural propagation."""
        for name, func in functions.items():
            self.analyze_function(func)

        if call_graph_fwd:
            self.interprocedural.set_call_graph(call_graph_fwd)
            self.interprocedural.analyze(functions, self._graphs)

            # Recompute hints after interprocedural analysis
            for name, graph in self._graphs.items():
                hints = EscapeOptimizationHints()
                hints.compute(graph)
                self._hints[name] = hints

        # Compute statistics from the last graph
        if self._graphs:
            last_name = list(self._graphs.keys())[-1]
            last_hints = self._hints.get(last_name, EscapeOptimizationHints())
            self.statistics.compute(self._graphs[last_name], last_hints)

        return dict(self._graphs)

    def _build_graph(self, func: FunctionIR, graph: ConnectionGraph) -> None:
        """Populate connection graph from function instructions."""
        for _, instr in func.all_instructions():
            if instr.opcode in ("assign", "copy") and instr.result and instr.operands:
                src = instr.operands[0]
                pts = graph.points_to(src)
                for obj_name in pts:
                    graph.add_points_to(instr.result, obj_name)
                if not pts:
                    graph.add_deferred(instr.result, src)

            elif instr.opcode == "load" and instr.result and instr.operands:
                base = instr.operands[0]
                for obj_name in graph.points_to(base):
                    for target in graph.successors(obj_name):
                        graph.add_points_to(instr.result, target)

            elif instr.opcode == "store" and len(instr.operands) >= 2:
                base = instr.operands[0]
                val = instr.operands[1]
                for obj_name in graph.points_to(base):
                    for val_obj in graph.points_to(val):
                        graph.add_field_edge(obj_name, "*", val_obj)

            elif instr.opcode == "getattr" and instr.result and len(instr.operands) >= 2:
                base, field_name = instr.operands[0], instr.operands[1]
                for obj_name in graph.points_to(base):
                    targets = graph.field_targets(obj_name, field_name)
                    for t in targets:
                        graph.add_points_to(instr.result, t)

            elif instr.opcode == "setattr" and len(instr.operands) >= 3:
                base, field_name, val = instr.operands[0], instr.operands[1], instr.operands[2]
                for obj_name in graph.points_to(base):
                    for val_obj in graph.points_to(val):
                        graph.add_field_edge(obj_name, field_name, val_obj)

            elif instr.opcode == "call" and instr.operands:
                # Arguments escape to callee
                for arg in instr.operands[1:]:
                    for obj_name in graph.points_to(arg):
                        obj = graph.get_object(obj_name)
                        if obj:
                            obj.set_escape(
                                EscapeState.ARG_ESCAPE,
                                f"argument to {instr.operands[0]}",
                                instr.operands[0],
                            )

            elif instr.opcode == "return" and instr.operands:
                for arg in instr.operands:
                    for obj_name in graph.points_to(arg):
                        obj = graph.get_object(obj_name)
                        if obj:
                            obj.set_escape(
                                EscapeState.RETURN_ESCAPE,
                                f"returned from {func.name}",
                            )

            elif instr.opcode in ("store_global", "global_assign") and instr.operands:
                for arg in instr.operands:
                    for obj_name in graph.points_to(arg):
                        obj = graph.get_object(obj_name)
                        if obj:
                            obj.set_escape(
                                EscapeState.GLOBAL_ESCAPE,
                                "assigned to global",
                            )

    def get_graph(self, func_name: str) -> Optional[ConnectionGraph]:
        return self._graphs.get(func_name)

    def get_hints(self, func_name: str) -> Optional[EscapeOptimizationHints]:
        return self._hints.get(func_name)

    def get_statistics(self) -> EscapeStatistics:
        return self.statistics

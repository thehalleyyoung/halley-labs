from __future__ import annotations

"""
Interprocedural analysis engine for refinement type inference.

Builds call graphs, propagates function summaries, and handles
dynamic dispatch, higher-order functions, closures, and recursion
in dynamically-typed languages.
"""

import time
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
from collections import defaultdict, deque
import heapq


# ---------------------------------------------------------------------------
# Local type stubs – we don't import from other project modules.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefinementType:
    """A refinement type {x: base | predicate}."""
    base: str
    predicate: str = "true"

    def join(self, other: RefinementType) -> RefinementType:
        if self.base != other.base:
            return RefinementType("object", "true")
        if self.predicate == other.predicate:
            return self
        return RefinementType(self.base, f"({self.predicate}) ∨ ({other.predicate})")

    def meet(self, other: RefinementType) -> RefinementType:
        if self.base != other.base:
            return RefinementType("⊥", "false")
        return RefinementType(self.base, f"({self.predicate}) ∧ ({other.predicate})")

    def is_bottom(self) -> bool:
        return self.base == "⊥"

    def is_top(self) -> bool:
        return self.base == "object" and self.predicate == "true"


@dataclass(frozen=True)
class Location:
    """Source location."""
    file: str = "<unknown>"
    line: int = 0
    column: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


@dataclass(frozen=True)
class Instruction:
    """Minimal IR instruction representation."""
    opcode: str
    operands: Tuple[str, ...] = ()
    result: Optional[str] = None
    location: Location = field(default_factory=Location)


@dataclass
class BasicBlock:
    """Basic block in a CFG."""
    label: str
    instructions: List[Instruction] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)


@dataclass
class FunctionIR:
    """Intermediate representation of a function."""
    name: str
    params: List[str] = field(default_factory=list)
    blocks: Dict[str, BasicBlock] = field(default_factory=dict)
    entry_block: str = "entry"
    return_type: Optional[RefinementType] = None
    is_method: bool = False
    class_name: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    closure_vars: List[str] = field(default_factory=list)
    module: str = "<module>"

    def all_instructions(self) -> Iterator[Tuple[str, Instruction]]:
        for label, bb in self.blocks.items():
            for instr in bb.instructions:
                yield label, instr


@dataclass
class ModuleIR:
    """Intermediate representation of a module."""
    name: str
    functions: Dict[str, FunctionIR] = field(default_factory=dict)
    classes: Dict[str, ClassIR] = field(default_factory=dict)
    globals: Dict[str, RefinementType] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)


@dataclass
class ClassIR:
    """IR for a class definition."""
    name: str
    bases: List[str] = field(default_factory=list)
    methods: Dict[str, FunctionIR] = field(default_factory=dict)
    class_vars: Dict[str, RefinementType] = field(default_factory=dict)
    module: str = "<module>"
    is_abstract: bool = False


# =========================================================================
#  CallGraphEdge
# =========================================================================

@dataclass(frozen=True)
class CallSite:
    """Identifies a specific call site in the program."""
    caller: str
    block: str
    index: int
    location: Location = field(default_factory=Location)

    def __str__(self) -> str:
        return f"{self.caller}@{self.block}:{self.index}"


class CallKind(Enum):
    """Kind of function call."""
    DIRECT = auto()
    INDIRECT = auto()
    VIRTUAL = auto()
    SUPER = auto()
    CONSTRUCTOR = auto()
    CALLBACK = auto()
    CLOSURE = auto()
    BUILTIN = auto()


@dataclass(frozen=True)
class CallGraphEdge:
    """Represents a caller→callee relationship with metadata."""
    caller: str
    callee: str
    call_site: CallSite
    kind: CallKind = CallKind.DIRECT
    is_resolved: bool = True
    context: Optional[CallingContext] = None
    argument_types: Tuple[RefinementType, ...] = ()

    def __str__(self) -> str:
        arrow = "→" if self.is_resolved else "⇢"
        return f"{self.caller} {arrow} {self.callee} [{self.kind.name}] at {self.call_site}"


# =========================================================================
#  ContextSensitivity
# =========================================================================

class ContextPolicy(Enum):
    """Context sensitivity policy."""
    CONTEXT_INSENSITIVE = auto()
    CALL_SITE_SENSITIVE = auto()  # k-CFA
    OBJECT_SENSITIVE = auto()
    TYPE_SENSITIVE = auto()
    HYBRID = auto()


@dataclass
class ContextSensitivity:
    """Configurable context-sensitivity policy for interprocedural analysis.

    Supports k-CFA (call-site sensitivity), object sensitivity, type
    sensitivity, and hybrid policies.  The *depth* parameter controls
    the length of the context string.
    """
    policy: ContextPolicy = ContextPolicy.CALL_SITE_SENSITIVE
    depth: int = 1
    selective_depth: Dict[str, int] = field(default_factory=dict)
    _heap_depth: int = 1

    def make_context(
        self,
        call_site: CallSite,
        receiver: Optional[str] = None,
        current_ctx: Optional[CallingContext] = None,
    ) -> CallingContext:
        if self.policy == ContextPolicy.CONTEXT_INSENSITIVE:
            return CallingContext.empty()

        effective_depth = self.selective_depth.get(call_site.caller, self.depth)

        if self.policy == ContextPolicy.CALL_SITE_SENSITIVE:
            elements: list[str] = []
            if current_ctx is not None:
                elements = list(current_ctx.elements)
            elements.append(str(call_site))
            if len(elements) > effective_depth:
                elements = elements[-effective_depth:]
            return CallingContext(elements=tuple(elements), policy=self.policy)

        if self.policy == ContextPolicy.OBJECT_SENSITIVE:
            elements = []
            if current_ctx is not None:
                elements = list(current_ctx.elements)
            elements.append(receiver or "<unknown>")
            if len(elements) > effective_depth:
                elements = elements[-effective_depth:]
            return CallingContext(elements=tuple(elements), policy=self.policy)

        if self.policy == ContextPolicy.TYPE_SENSITIVE:
            elements = []
            if current_ctx is not None:
                elements = list(current_ctx.elements)
            type_label = receiver.split(".")[0] if receiver else "<unknown>"
            elements.append(type_label)
            if len(elements) > effective_depth:
                elements = elements[-effective_depth:]
            return CallingContext(elements=tuple(elements), policy=self.policy)

        # HYBRID: use object sensitivity for methods, call-site for functions
        if call_site.caller.count(".") > 0 and receiver:
            elements = []
            if current_ctx:
                elements = list(current_ctx.elements)
            elements.append(receiver)
            if len(elements) > effective_depth:
                elements = elements[-effective_depth:]
            return CallingContext(elements=tuple(elements), policy=ContextPolicy.OBJECT_SENSITIVE)
        else:
            elements = []
            if current_ctx:
                elements = list(current_ctx.elements)
            elements.append(str(call_site))
            if len(elements) > effective_depth:
                elements = elements[-effective_depth:]
            return CallingContext(elements=tuple(elements), policy=ContextPolicy.CALL_SITE_SENSITIVE)

    def set_selective_depth(self, function_name: str, depth: int) -> None:
        self.selective_depth[function_name] = depth


# =========================================================================
#  CallingContext
# =========================================================================

@dataclass(frozen=True)
class CallingContext:
    """Represents a calling context (call string or allocation-site string)."""
    elements: Tuple[str, ...] = ()
    policy: ContextPolicy = ContextPolicy.CONTEXT_INSENSITIVE

    @staticmethod
    def empty() -> CallingContext:
        return CallingContext()

    def extend(self, element: str, max_depth: int = 1) -> CallingContext:
        new_elems = self.elements + (element,)
        if len(new_elems) > max_depth:
            new_elems = new_elems[-max_depth:]
        return CallingContext(elements=new_elems, policy=self.policy)

    def is_empty(self) -> bool:
        return len(self.elements) == 0

    def depth(self) -> int:
        return len(self.elements)

    def truncate(self, k: int) -> CallingContext:
        if len(self.elements) <= k:
            return self
        return CallingContext(elements=self.elements[-k:], policy=self.policy)

    def __str__(self) -> str:
        if not self.elements:
            return "∅"
        return "[" + " ← ".join(self.elements) + "]"


# =========================================================================
#  CallGraph
# =========================================================================

@dataclass
class CallGraph:
    """Call graph supporting direct, indirect, and virtual calls.

    Maintains forward (caller→callees) and backward (callee→callers)
    adjacency, plus per-call-site edge information.
    """

    _forward: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _backward: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _edges: Dict[Tuple[str, str], List[CallGraphEdge]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _call_site_edges: Dict[CallSite, List[CallGraphEdge]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _nodes: Set[str] = field(default_factory=set)
    _unresolved: Set[CallSite] = field(default_factory=set)

    # -- mutation ----------------------------------------------------------

    def add_node(self, name: str) -> None:
        self._nodes.add(name)

    def add_edge(self, edge: CallGraphEdge) -> None:
        self._nodes.add(edge.caller)
        self._nodes.add(edge.callee)
        self._forward[edge.caller].add(edge.callee)
        self._backward[edge.callee].add(edge.caller)
        self._edges[(edge.caller, edge.callee)].append(edge)
        self._call_site_edges[edge.call_site].append(edge)
        if not edge.is_resolved:
            self._unresolved.add(edge.call_site)

    def remove_edge(self, caller: str, callee: str) -> None:
        self._forward[caller].discard(callee)
        self._backward[callee].discard(caller)
        self._edges.pop((caller, callee), None)

    def resolve_call_site(self, call_site: CallSite, targets: Sequence[str]) -> List[CallGraphEdge]:
        self._unresolved.discard(call_site)
        new_edges: list[CallGraphEdge] = []
        for target in targets:
            edge = CallGraphEdge(
                caller=call_site.caller,
                callee=target,
                call_site=call_site,
                kind=CallKind.INDIRECT,
                is_resolved=True,
            )
            self.add_edge(edge)
            new_edges.append(edge)
        return new_edges

    # -- queries -----------------------------------------------------------

    def callees(self, caller: str) -> Set[str]:
        return set(self._forward.get(caller, set()))

    def callers(self, callee: str) -> Set[str]:
        return set(self._backward.get(callee, set()))

    def edges_between(self, caller: str, callee: str) -> List[CallGraphEdge]:
        return list(self._edges.get((caller, callee), []))

    def edges_at(self, call_site: CallSite) -> List[CallGraphEdge]:
        return list(self._call_site_edges.get(call_site, []))

    def all_edges(self) -> Iterator[CallGraphEdge]:
        for edge_list in self._edges.values():
            yield from edge_list

    def all_nodes(self) -> Set[str]:
        return set(self._nodes)

    def unresolved_sites(self) -> Set[CallSite]:
        return set(self._unresolved)

    def has_node(self, name: str) -> bool:
        return name in self._nodes

    def has_edge(self, caller: str, callee: str) -> bool:
        return callee in self._forward.get(caller, set())

    def is_recursive(self, func: str) -> bool:
        return func in self._forward.get(func, set())

    def reachable_from(self, root: str) -> Set[str]:
        visited: set[str] = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self._forward.get(node, set()) - visited)
        return visited

    def transitive_callers(self, func: str) -> Set[str]:
        visited: set[str] = set()
        stack = [func]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self._backward.get(node, set()) - visited)
        visited.discard(func)
        return visited

    def roots(self) -> Set[str]:
        return {n for n in self._nodes if not self._backward.get(n)}

    def leaves(self) -> Set[str]:
        return {n for n in self._nodes if not self._forward.get(n)}

    def strongly_connected_components(self) -> List[List[str]]:
        """Tarjan's SCC algorithm."""
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

            for w in self._forward.get(v, set()):
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

        for node in self._nodes:
            if node not in index_map:
                strongconnect(node)

        return result

    def topological_order(self) -> List[str]:
        """Topological sort (Kahn's algorithm). Ignores back-edges for cycles."""
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for n in self._nodes:
            for s in self._forward.get(n, set()):
                in_degree[s] = in_degree.get(s, 0) + 1
        queue = deque(n for n, d in in_degree.items() if d == 0)
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for succ in self._forward.get(node, set()):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        return order

    def edge_count(self) -> int:
        return sum(len(el) for el in self._edges.values())

    def node_count(self) -> int:
        return len(self._nodes)


# =========================================================================
#  CallGraphBuilder
# =========================================================================

@dataclass
class CallGraphBuilder:
    """Builds a call graph from IR, handling dynamic dispatch and higher-order functions."""

    graph: CallGraph = field(default_factory=CallGraph)
    _function_table: Dict[str, FunctionIR] = field(default_factory=dict)
    _class_table: Dict[str, ClassIR] = field(default_factory=dict)
    _builtin_set: Set[str] = field(
        default_factory=lambda: {
            "print", "len", "range", "int", "str", "float", "list", "dict",
            "set", "tuple", "type", "isinstance", "issubclass", "hasattr",
            "getattr", "setattr", "delattr", "id", "hash", "repr", "abs",
            "min", "max", "sum", "sorted", "reversed", "enumerate", "zip",
            "map", "filter", "any", "all", "iter", "next", "open",
        }
    )

    def build(self, modules: Sequence[ModuleIR]) -> CallGraph:
        for mod in modules:
            self._register_module(mod)
        for mod in modules:
            self._process_module(mod)
        return self.graph

    def _register_module(self, module: ModuleIR) -> None:
        for name, func in module.functions.items():
            fqn = f"{module.name}.{name}"
            self._function_table[fqn] = func
            self._function_table[name] = func
            self.graph.add_node(fqn)
        for cls_name, cls_ir in module.classes.items():
            self._class_table[cls_name] = cls_ir
            for meth_name, meth in cls_ir.methods.items():
                fqn = f"{cls_name}.{meth_name}"
                self._function_table[fqn] = meth
                self.graph.add_node(fqn)

    def _process_module(self, module: ModuleIR) -> None:
        for name, func in module.functions.items():
            caller_fqn = f"{module.name}.{name}"
            self._process_function(caller_fqn, func)
        for cls_name, cls_ir in module.classes.items():
            for meth_name, meth in cls_ir.methods.items():
                caller_fqn = f"{cls_name}.{meth_name}"
                self._process_function(caller_fqn, meth)

    def _process_function(self, caller_fqn: str, func: FunctionIR) -> None:
        for block_label, instr in func.all_instructions():
            if instr.opcode == "call":
                self._process_call(caller_fqn, block_label, instr, func)
            elif instr.opcode == "method_call":
                self._process_method_call(caller_fqn, block_label, instr)

    def _process_call(
        self,
        caller: str,
        block: str,
        instr: Instruction,
        func: FunctionIR,
    ) -> None:
        if not instr.operands:
            return
        callee_name = instr.operands[0]
        site = CallSite(caller=caller, block=block, index=0, location=instr.location)

        if callee_name in self._builtin_set:
            edge = CallGraphEdge(
                caller=caller,
                callee=callee_name,
                call_site=site,
                kind=CallKind.BUILTIN,
            )
            self.graph.add_edge(edge)
            return

        resolved = self._resolve_callee(callee_name, func)
        if resolved:
            for target in resolved:
                edge = CallGraphEdge(
                    caller=caller, callee=target, call_site=site, kind=CallKind.DIRECT,
                )
                self.graph.add_edge(edge)
        else:
            edge = CallGraphEdge(
                caller=caller,
                callee=callee_name,
                call_site=site,
                kind=CallKind.INDIRECT,
                is_resolved=False,
            )
            self.graph.add_edge(edge)

    def _process_method_call(self, caller: str, block: str, instr: Instruction) -> None:
        if len(instr.operands) < 2:
            return
        receiver, method = instr.operands[0], instr.operands[1]
        site = CallSite(caller=caller, block=block, index=0, location=instr.location)

        targets = self._resolve_virtual(receiver, method)
        if targets:
            for target in targets:
                edge = CallGraphEdge(
                    caller=caller, callee=target, call_site=site, kind=CallKind.VIRTUAL,
                )
                self.graph.add_edge(edge)
        else:
            edge = CallGraphEdge(
                caller=caller,
                callee=f"{receiver}.{method}",
                call_site=site,
                kind=CallKind.VIRTUAL,
                is_resolved=False,
            )
            self.graph.add_edge(edge)

    def _resolve_callee(self, name: str, func: FunctionIR) -> List[str]:
        if name in self._function_table:
            return [name]
        fqn = f"{func.module}.{name}"
        if fqn in self._function_table:
            return [fqn]
        if name in func.closure_vars:
            return []
        return []

    def _resolve_virtual(self, receiver_type: str, method: str) -> List[str]:
        targets: list[str] = []
        if receiver_type in self._class_table:
            cls = self._class_table[receiver_type]
            fqn = f"{receiver_type}.{method}"
            if fqn in self._function_table:
                targets.append(fqn)
            for base_name in cls.bases:
                base_fqn = f"{base_name}.{method}"
                if base_fqn in self._function_table:
                    targets.append(base_fqn)
        return targets

    def add_dynamic_edge(self, caller: str, callee: str, site: CallSite) -> CallGraphEdge:
        edge = CallGraphEdge(
            caller=caller, callee=callee, call_site=site, kind=CallKind.INDIRECT,
        )
        self.graph.add_edge(edge)
        return edge


# =========================================================================
#  FunctionSummary
# =========================================================================

@dataclass
class FunctionSummary:
    """Summarises a function's behavior: preconditions, postconditions, effects."""

    name: str
    param_types: Dict[str, RefinementType] = field(default_factory=dict)
    return_type: RefinementType = field(default_factory=lambda: RefinementType("object"))
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    side_effects: SideEffects = field(default_factory=lambda: SideEffects())
    raises: Set[str] = field(default_factory=set)
    is_pure: bool = False
    is_total: bool = True
    call_count: int = 0
    version: int = 0
    context: Optional[CallingContext] = None

    def join(self, other: FunctionSummary) -> FunctionSummary:
        merged_params: dict[str, RefinementType] = {}
        all_keys = set(self.param_types) | set(other.param_types)
        for k in all_keys:
            a = self.param_types.get(k, RefinementType("⊥"))
            b = other.param_types.get(k, RefinementType("⊥"))
            merged_params[k] = a.join(b)
        return FunctionSummary(
            name=self.name,
            param_types=merged_params,
            return_type=self.return_type.join(other.return_type),
            preconditions=list(set(self.preconditions) & set(other.preconditions)),
            postconditions=list(set(self.postconditions) & set(other.postconditions)),
            side_effects=self.side_effects.merge(other.side_effects),
            raises=self.raises | other.raises,
            is_pure=self.is_pure and other.is_pure,
            is_total=self.is_total and other.is_total,
            version=max(self.version, other.version) + 1,
            context=self.context,
        )

    def subsumes(self, other: FunctionSummary) -> bool:
        if self.return_type != other.return_type:
            return False
        for k, v in other.param_types.items():
            if k not in self.param_types:
                return False
            if self.param_types[k] != v:
                return False
        return other.raises <= self.raises

    def invalidate(self) -> None:
        self.version += 1

    def clone(self) -> FunctionSummary:
        return FunctionSummary(
            name=self.name,
            param_types=dict(self.param_types),
            return_type=self.return_type,
            preconditions=list(self.preconditions),
            postconditions=list(self.postconditions),
            side_effects=self.side_effects.clone(),
            raises=set(self.raises),
            is_pure=self.is_pure,
            is_total=self.is_total,
            call_count=self.call_count,
            version=self.version,
            context=self.context,
        )


# =========================================================================
#  SideEffects
# =========================================================================

@dataclass
class SideEffects:
    """Tracks reads, writes, and allocations performed by a function."""
    reads: Set[str] = field(default_factory=set)
    writes: Set[str] = field(default_factory=set)
    allocations: Set[str] = field(default_factory=set)
    calls_external: bool = False
    raises_exceptions: bool = False

    def merge(self, other: SideEffects) -> SideEffects:
        return SideEffects(
            reads=self.reads | other.reads,
            writes=self.writes | other.writes,
            allocations=self.allocations | other.allocations,
            calls_external=self.calls_external or other.calls_external,
            raises_exceptions=self.raises_exceptions or other.raises_exceptions,
        )

    def is_pure(self) -> bool:
        return (
            not self.writes
            and not self.allocations
            and not self.calls_external
            and not self.raises_exceptions
        )

    def clone(self) -> SideEffects:
        return SideEffects(
            reads=set(self.reads),
            writes=set(self.writes),
            allocations=set(self.allocations),
            calls_external=self.calls_external,
            raises_exceptions=self.raises_exceptions,
        )


# =========================================================================
#  SummaryTable
# =========================================================================

@dataclass
class SummaryTable:
    """Stores and retrieves function summaries, handles invalidation."""

    _summaries: Dict[Tuple[str, CallingContext], FunctionSummary] = field(
        default_factory=dict
    )
    _version: int = 0
    _invalidation_listeners: List[Callable[[str], None]] = field(default_factory=list)

    def get(
        self, name: str, context: Optional[CallingContext] = None
    ) -> Optional[FunctionSummary]:
        ctx = context or CallingContext.empty()
        return self._summaries.get((name, ctx))

    def get_all(self, name: str) -> List[FunctionSummary]:
        return [s for (n, _), s in self._summaries.items() if n == name]

    def put(self, summary: FunctionSummary) -> bool:
        ctx = summary.context or CallingContext.empty()
        key = (summary.name, ctx)
        old = self._summaries.get(key)
        if old is not None and old.subsumes(summary):
            return False
        self._summaries[key] = summary
        self._version += 1
        return True

    def invalidate(self, name: str) -> List[str]:
        invalidated: list[str] = []
        keys_to_remove = [k for k in self._summaries if k[0] == name]
        for k in keys_to_remove:
            self._summaries[k].invalidate()
            invalidated.append(k[0])
        self._version += 1
        for listener in self._invalidation_listeners:
            listener(name)
        return invalidated

    def invalidate_dependents(self, name: str, call_graph: CallGraph) -> List[str]:
        callers = call_graph.transitive_callers(name)
        all_invalidated: list[str] = [name]
        for caller in callers:
            all_invalidated.extend(self.invalidate(caller))
        return all_invalidated

    def add_invalidation_listener(self, listener: Callable[[str], None]) -> None:
        self._invalidation_listeners.append(listener)

    def all_names(self) -> Set[str]:
        return {n for n, _ in self._summaries}

    def size(self) -> int:
        return len(self._summaries)

    def version(self) -> int:
        return self._version


# =========================================================================
#  WorklistScheduler
# =========================================================================

class WorklistPriority(Enum):
    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclass
class WorklistScheduler:
    """Priority-based worklist for interprocedural iteration."""

    _heap: List[Tuple[int, int, str]] = field(default_factory=list)
    _counter: int = 0
    _in_worklist: Set[str] = field(default_factory=set)
    _priority_map: Dict[str, int] = field(default_factory=dict)

    def add(self, item: str, priority: WorklistPriority = WorklistPriority.NORMAL) -> None:
        if item in self._in_worklist:
            return
        self._in_worklist.add(item)
        self._counter += 1
        heapq.heappush(self._heap, (priority.value, self._counter, item))
        self._priority_map[item] = priority.value

    def add_all(self, items: Sequence[str], priority: WorklistPriority = WorklistPriority.NORMAL) -> None:
        for item in items:
            self.add(item, priority)

    def pop(self) -> Optional[str]:
        while self._heap:
            _, _, item = heapq.heappop(self._heap)
            if item in self._in_worklist:
                self._in_worklist.discard(item)
                return item
        return None

    def is_empty(self) -> bool:
        return len(self._in_worklist) == 0

    def size(self) -> int:
        return len(self._in_worklist)

    def contains(self, item: str) -> bool:
        return item in self._in_worklist

    def boost_priority(self, item: str) -> None:
        if item in self._in_worklist:
            self._counter += 1
            heapq.heappush(self._heap, (WorklistPriority.HIGH.value, self._counter, item))

    def clear(self) -> None:
        self._heap.clear()
        self._in_worklist.clear()
        self._counter = 0


# =========================================================================
#  ConvergenceChecker
# =========================================================================

@dataclass
class ConvergenceChecker:
    """Checks whether the interprocedural analysis has reached a fixed point."""

    _previous_versions: Dict[str, int] = field(default_factory=dict)
    _stable_count: int = 0
    _total_checks: int = 0
    tolerance: int = 0
    max_iterations: int = 1000

    def check(self, summary_table: SummaryTable) -> bool:
        self._total_checks += 1
        if self._total_checks > self.max_iterations:
            return True
        changed = False
        for name in summary_table.all_names():
            summaries = summary_table.get_all(name)
            for s in summaries:
                prev = self._previous_versions.get(s.name, -1)
                if s.version != prev:
                    changed = True
                    self._previous_versions[s.name] = s.version
        if not changed:
            self._stable_count += 1
        else:
            self._stable_count = 0
        return self._stable_count > self.tolerance

    def reset(self) -> None:
        self._previous_versions.clear()
        self._stable_count = 0
        self._total_checks = 0

    def iterations(self) -> int:
        return self._total_checks


# =========================================================================
#  AnalysisStatistics
# =========================================================================

@dataclass
class AnalysisStatistics:
    """Tracks analysis metrics."""

    functions_analyzed: int = 0
    summaries_computed: int = 0
    summaries_invalidated: int = 0
    iterations: int = 0
    edges_added: int = 0
    unresolved_calls: int = 0
    recursive_functions: int = 0
    scc_count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    def start(self) -> None:
        self.start_time = time.time()

    def stop(self) -> None:
        self.end_time = time.time()

    @property
    def elapsed(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def record_function(self) -> None:
        self.functions_analyzed += 1

    def record_summary(self) -> None:
        self.summaries_computed += 1

    def record_invalidation(self, count: int = 1) -> None:
        self.summaries_invalidated += count

    def record_iteration(self) -> None:
        self.iterations += 1

    def __str__(self) -> str:
        return (
            f"AnalysisStatistics(functions={self.functions_analyzed}, "
            f"summaries={self.summaries_computed}, iterations={self.iterations}, "
            f"elapsed={self.elapsed:.3f}s)"
        )


# =========================================================================
#  ParameterBinding
# =========================================================================

@dataclass
class ParameterBinding:
    """Binds actual arguments to formal parameters with refinement type transfer."""

    _bindings: Dict[str, RefinementType] = field(default_factory=dict)

    def bind(
        self,
        formals: Sequence[str],
        actuals: Sequence[RefinementType],
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, RefinementType]:
        self._bindings.clear()
        defaults = defaults or {}
        for i, param in enumerate(formals):
            if i < len(actuals):
                self._bindings[param] = actuals[i]
            elif param in defaults:
                self._bindings[param] = RefinementType(
                    type(defaults[param]).__name__,
                    f"v == {defaults[param]!r}",
                )
            else:
                self._bindings[param] = RefinementType("object")
        return dict(self._bindings)

    def bind_kwargs(
        self,
        formals: Sequence[str],
        kwargs: Dict[str, RefinementType],
    ) -> Dict[str, RefinementType]:
        result: dict[str, RefinementType] = {}
        for param in formals:
            if param in kwargs:
                result[param] = kwargs[param]
            else:
                result[param] = RefinementType("object")
        self._bindings.update(result)
        return result

    def get(self, name: str) -> Optional[RefinementType]:
        return self._bindings.get(name)

    def all_bindings(self) -> Dict[str, RefinementType]:
        return dict(self._bindings)


# =========================================================================
#  ReturnMerger
# =========================================================================

@dataclass
class ReturnMerger:
    """Merges return values from multiple return sites."""

    _returns: List[Tuple[str, RefinementType]] = field(default_factory=list)
    _conditions: List[str] = field(default_factory=list)

    def add_return(self, site: str, rtype: RefinementType, condition: str = "true") -> None:
        self._returns.append((site, rtype))
        self._conditions.append(condition)

    def merge(self) -> RefinementType:
        if not self._returns:
            return RefinementType("None", "v is None")
        result = self._returns[0][1]
        for _, rt in self._returns[1:]:
            result = result.join(rt)
        return result

    def merge_conditional(self) -> RefinementType:
        if not self._returns:
            return RefinementType("None", "v is None")
        parts: list[str] = []
        bases: set[str] = set()
        for (site, rt), cond in zip(self._returns, self._conditions):
            bases.add(rt.base)
            if cond != "true":
                parts.append(f"({cond} ⇒ {rt.predicate})")
            else:
                parts.append(rt.predicate)
        base = bases.pop() if len(bases) == 1 else "object"
        pred = " ∨ ".join(parts)
        return RefinementType(base, pred)

    def return_sites(self) -> List[str]:
        return [site for site, _ in self._returns]

    def clear(self) -> None:
        self._returns.clear()
        self._conditions.clear()


# =========================================================================
#  ExceptionPropagator
# =========================================================================

@dataclass
class ExceptionInfo:
    """Information about a raised exception."""
    exception_type: str
    source_function: str
    location: Location = field(default_factory=Location)
    condition: str = "true"
    message: str = ""


@dataclass
class ExceptionPropagator:
    """Propagates exception types and states interprocedurally."""

    _exceptions: Dict[str, List[ExceptionInfo]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _handlers: Dict[str, List[Tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_raise(self, function: str, info: ExceptionInfo) -> None:
        self._exceptions[function].append(info)

    def add_handler(self, function: str, exc_type: str, handler_block: str) -> None:
        self._handlers[function].append((exc_type, handler_block))

    def get_exceptions(self, function: str) -> List[ExceptionInfo]:
        return list(self._exceptions.get(function, []))

    def propagate(self, call_graph: CallGraph) -> Dict[str, Set[str]]:
        propagated: dict[str, set[str]] = defaultdict(set)
        topo = call_graph.topological_order()
        for func in reversed(topo):
            handled = {t for t, _ in self._handlers.get(func, [])}
            for exc in self._exceptions.get(func, []):
                if exc.exception_type not in handled:
                    propagated[func].add(exc.exception_type)
            for callee in call_graph.callees(func):
                for exc_type in propagated.get(callee, set()):
                    if exc_type not in handled:
                        propagated[func].add(exc_type)
        return dict(propagated)

    def unhandled_in(self, function: str) -> Set[str]:
        handled = {t for t, _ in self._handlers.get(function, [])}
        raised = {e.exception_type for e in self._exceptions.get(function, [])}
        return raised - handled

    def exception_types_for(self, function: str) -> Set[str]:
        return {e.exception_type for e in self._exceptions.get(function, [])}


# =========================================================================
#  AliasAnalysis
# =========================================================================

@dataclass(frozen=True)
class AbstractLocation:
    """An abstract heap location."""
    name: str
    field: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.name}.{self.field}" if self.field else self.name


@dataclass
class AliasAnalysis:
    """Simple alias analysis for heap objects using equivalence classes."""

    _alias_sets: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _parent: Dict[str, str] = field(default_factory=dict)

    def _find(self, x: str) -> str:
        while self._parent.get(x, x) != x:
            self._parent[x] = self._parent.get(self._parent[x], self._parent[x])
            x = self._parent[x]
        return x

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[rb] = ra

    def record_assignment(self, lhs: str, rhs: str) -> None:
        self._parent.setdefault(lhs, lhs)
        self._parent.setdefault(rhs, rhs)
        self._union(lhs, rhs)

    def may_alias(self, a: str, b: str) -> bool:
        if a not in self._parent or b not in self._parent:
            return a == b
        return self._find(a) == self._find(b)

    def must_alias(self, a: str, b: str) -> bool:
        return a == b

    def alias_set(self, var: str) -> Set[str]:
        if var not in self._parent:
            return {var}
        root = self._find(var)
        return {v for v in self._parent if self._find(v) == root}

    def analyze_function(self, func: FunctionIR) -> None:
        for _, instr in func.all_instructions():
            if instr.opcode in ("assign", "copy") and instr.result and instr.operands:
                self.record_assignment(instr.result, instr.operands[0])
            elif instr.opcode == "load" and instr.result and instr.operands:
                self.record_assignment(instr.result, instr.operands[0])


# =========================================================================
#  SideEffectAnalysis
# =========================================================================

@dataclass
class SideEffectAnalysis:
    """Tracks function side effects (reads, writes, allocations)."""

    _effects: Dict[str, SideEffects] = field(default_factory=dict)

    def analyze(self, func: FunctionIR) -> SideEffects:
        effects = SideEffects()
        for _, instr in func.all_instructions():
            if instr.opcode in ("load", "getattr", "getitem"):
                for op in instr.operands:
                    effects.reads.add(op)
            elif instr.opcode in ("store", "setattr", "setitem"):
                if instr.operands:
                    effects.writes.add(instr.operands[0])
            elif instr.opcode in ("alloc", "new", "list_new", "dict_new"):
                if instr.result:
                    effects.allocations.add(instr.result)
            elif instr.opcode == "call":
                effects.calls_external = True
            elif instr.opcode in ("raise", "throw"):
                effects.raises_exceptions = True
        self._effects[func.name] = effects
        return effects

    def get(self, name: str) -> Optional[SideEffects]:
        return self._effects.get(name)

    def propagate(self, call_graph: CallGraph) -> None:
        topo = call_graph.topological_order()
        for func in reversed(topo):
            eff = self._effects.get(func)
            if eff is None:
                continue
            for callee in call_graph.callees(func):
                callee_eff = self._effects.get(callee)
                if callee_eff:
                    eff.reads |= callee_eff.reads
                    eff.writes |= callee_eff.writes
                    eff.allocations |= callee_eff.allocations
                    if callee_eff.calls_external:
                        eff.calls_external = True
                    if callee_eff.raises_exceptions:
                        eff.raises_exceptions = True

    def modifies(self, name: str, variable: str) -> bool:
        eff = self._effects.get(name)
        return eff is not None and variable in eff.writes

    def reads_from(self, name: str, variable: str) -> bool:
        eff = self._effects.get(name)
        return eff is not None and variable in eff.reads


# =========================================================================
#  PurityAnalysis
# =========================================================================

@dataclass
class PurityAnalysis:
    """Determines if a function is pure (no observable side effects)."""

    _purity: Dict[str, bool] = field(default_factory=dict)
    _cache_valid: bool = False

    def analyze(self, func: FunctionIR, effects: SideEffects) -> bool:
        is_pure = effects.is_pure()
        self._purity[func.name] = is_pure
        return is_pure

    def analyze_all(
        self,
        functions: Dict[str, FunctionIR],
        side_effects: SideEffectAnalysis,
        call_graph: CallGraph,
    ) -> Dict[str, bool]:
        self._purity.clear()
        sccs = call_graph.strongly_connected_components()
        for scc in reversed(sccs):
            if len(scc) > 1:
                for func_name in scc:
                    self._purity[func_name] = False
                continue
            func_name = scc[0]
            if call_graph.is_recursive(func_name):
                eff = side_effects.get(func_name)
                self._purity[func_name] = eff.is_pure() if eff else False
                continue
            eff = side_effects.get(func_name)
            if eff is None:
                self._purity[func_name] = False
                continue
            is_pure = eff.is_pure()
            if is_pure:
                for callee in call_graph.callees(func_name):
                    if not self._purity.get(callee, False):
                        is_pure = False
                        break
            self._purity[func_name] = is_pure
        self._cache_valid = True
        return dict(self._purity)

    def is_pure(self, name: str) -> bool:
        return self._purity.get(name, False)

    def pure_functions(self) -> Set[str]:
        return {n for n, p in self._purity.items() if p}


# =========================================================================
#  CallbackAnalyzer
# =========================================================================

HOF_REGISTRY: Dict[str, int] = {
    "map": 0,
    "filter": 0,
    "sorted": -1,  # key kwarg
    "reduce": 0,
    "functools.reduce": 0,
    "itertools.starmap": 0,
    "apply": 0,
}


@dataclass
class CallbackAnalyzer:
    """Handles higher-order function callbacks (map, filter, etc.)."""

    _callback_edges: List[CallGraphEdge] = field(default_factory=list)
    _hof_registry: Dict[str, int] = field(default_factory=lambda: dict(HOF_REGISTRY))

    def register_hof(self, name: str, callback_arg_index: int) -> None:
        self._hof_registry[name] = callback_arg_index

    def analyze_call(
        self,
        caller: str,
        callee: str,
        arguments: Sequence[str],
        call_site: CallSite,
        function_table: Dict[str, FunctionIR],
    ) -> List[CallGraphEdge]:
        edges: list[CallGraphEdge] = []
        if callee not in self._hof_registry:
            return edges
        cb_index = self._hof_registry[callee]
        if cb_index < 0:
            return edges
        if cb_index < len(arguments):
            cb_name = arguments[cb_index]
            if cb_name in function_table:
                edge = CallGraphEdge(
                    caller=caller,
                    callee=cb_name,
                    call_site=call_site,
                    kind=CallKind.CALLBACK,
                )
                edges.append(edge)
                self._callback_edges.append(edge)
        return edges

    def get_callback_edges(self) -> List[CallGraphEdge]:
        return list(self._callback_edges)

    def infer_callback_return(
        self, hof: str, callback_summary: FunctionSummary,
    ) -> RefinementType:
        if hof in ("map", "itertools.starmap"):
            return RefinementType("list", f"all(elem: {callback_summary.return_type.predicate})")
        if hof == "filter":
            return RefinementType("list", callback_summary.preconditions[0] if callback_summary.preconditions else "true")
        return callback_summary.return_type


# =========================================================================
#  ClosureAnalyzer
# =========================================================================

@dataclass
class CapturedVariable:
    """A variable captured by a closure."""
    name: str
    defining_function: str
    rtype: RefinementType = field(default_factory=lambda: RefinementType("object"))
    is_mutable: bool = False
    capture_site: Optional[Location] = None


@dataclass
class ClosureAnalyzer:
    """Tracks closure captured variables and their refinement types."""

    _closures: Dict[str, List[CapturedVariable]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _defining_scope: Dict[str, str] = field(default_factory=dict)

    def analyze(self, func: FunctionIR, enclosing: Optional[FunctionIR] = None) -> List[CapturedVariable]:
        if not func.closure_vars or enclosing is None:
            return []
        captured: list[CapturedVariable] = []
        enclosing_locals = set(enclosing.params)
        for _, instr in enclosing.all_instructions():
            if instr.result:
                enclosing_locals.add(instr.result)

        for var in func.closure_vars:
            is_mutable = self._check_mutability(var, func)
            cv = CapturedVariable(
                name=var,
                defining_function=enclosing.name,
                is_mutable=is_mutable,
            )
            captured.append(cv)
            self._defining_scope[var] = enclosing.name

        self._closures[func.name] = captured
        return captured

    def _check_mutability(self, var: str, func: FunctionIR) -> bool:
        for _, instr in func.all_instructions():
            if instr.opcode in ("store", "assign") and instr.operands and instr.operands[0] == var:
                return True
        return False

    def get_captured(self, func_name: str) -> List[CapturedVariable]:
        return list(self._closures.get(func_name, []))

    def captures_mutable(self, func_name: str) -> bool:
        return any(cv.is_mutable for cv in self._closures.get(func_name, []))

    def closure_environment(self, func_name: str) -> Dict[str, RefinementType]:
        return {cv.name: cv.rtype for cv in self._closures.get(func_name, [])}


# =========================================================================
#  ModuleAnalyzer
# =========================================================================

@dataclass
class ModuleAnalyzer:
    """Analyzes module-level code, global variables, and imports."""

    _global_types: Dict[str, Dict[str, RefinementType]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    _import_graph: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    _analyzed: Set[str] = field(default_factory=set)

    def analyze(self, module: ModuleIR) -> Dict[str, RefinementType]:
        if module.name in self._analyzed:
            return self._global_types[module.name]

        globals_map: dict[str, RefinementType] = {}
        for name, rtype in module.globals.items():
            globals_map[name] = rtype

        for imp in module.imports:
            self._import_graph[module.name].add(imp)

        self._global_types[module.name] = globals_map
        self._analyzed.add(module.name)
        return globals_map

    def get_global_type(self, module: str, name: str) -> Optional[RefinementType]:
        return self._global_types.get(module, {}).get(name)

    def set_global_type(self, module: str, name: str, rtype: RefinementType) -> None:
        self._global_types[module][name] = rtype

    def imports_of(self, module: str) -> Set[str]:
        return set(self._import_graph.get(module, set()))

    def import_order(self) -> List[str]:
        """Topological order of modules by import dependencies."""
        in_degree: dict[str, int] = defaultdict(int)
        all_modules = set(self._import_graph.keys())
        for deps in self._import_graph.values():
            all_modules |= deps
        for m in all_modules:
            in_degree.setdefault(m, 0)
        for m, deps in self._import_graph.items():
            for d in deps:
                in_degree[m] += 1  # m depends on d
        queue = deque(m for m in all_modules if in_degree[m] == 0)
        order: list[str] = []
        while queue:
            m = queue.popleft()
            order.append(m)
            for dependent, deps in self._import_graph.items():
                if m in deps:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        return order


# =========================================================================
#  ClassHierarchyAnalysis
# =========================================================================

@dataclass
class ClassHierarchyAnalysis:
    """Builds class hierarchy for virtual dispatch resolution."""

    _parents: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    _children: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _methods: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _abstract: Set[str] = field(default_factory=set)

    def build(self, classes: Dict[str, ClassIR]) -> None:
        for name, cls in classes.items():
            self._parents[name] = list(cls.bases)
            for base in cls.bases:
                self._children[base].add(name)
            for meth in cls.methods:
                self._methods[name].add(meth)
            if cls.is_abstract:
                self._abstract.add(name)

    def add_class(self, cls: ClassIR) -> None:
        self._parents[cls.name] = list(cls.bases)
        for base in cls.bases:
            self._children[base].add(cls.name)
        for meth in cls.methods:
            self._methods[cls.name].add(meth)
        if cls.is_abstract:
            self._abstract.add(cls.name)

    def is_subclass(self, child: str, parent: str) -> bool:
        if child == parent:
            return True
        visited: set[str] = set()
        stack = [child]
        while stack:
            cls = stack.pop()
            if cls in visited:
                continue
            visited.add(cls)
            if cls == parent:
                return True
            stack.extend(self._parents.get(cls, []))
        return False

    def all_subclasses(self, cls: str) -> Set[str]:
        result: set[str] = set()
        stack = list(self._children.get(cls, set()))
        while stack:
            c = stack.pop()
            if c in result:
                continue
            result.add(c)
            stack.extend(self._children.get(c, set()))
        return result

    def all_superclasses(self, cls: str) -> Set[str]:
        result: set[str] = set()
        stack = list(self._parents.get(cls, []))
        while stack:
            c = stack.pop()
            if c in result:
                continue
            result.add(c)
            stack.extend(self._parents.get(c, []))
        return result

    def concrete_subclasses(self, cls: str) -> Set[str]:
        return {c for c in self.all_subclasses(cls) if c not in self._abstract}

    def has_method(self, cls: str, method: str) -> bool:
        return method in self._methods.get(cls, set())

    def resolve_method(self, cls: str, method: str) -> Optional[str]:
        """MRO-style resolution: check class, then parents in order."""
        if self.has_method(cls, method):
            return f"{cls}.{method}"
        for parent in self._parents.get(cls, []):
            result = self.resolve_method(parent, method)
            if result:
                return result
        return None

    def mro(self, cls: str) -> List[str]:
        """Compute C3 linearization (simplified)."""
        if not self._parents.get(cls):
            return [cls]
        result = [cls]
        for parent in self._parents[cls]:
            for c in self.mro(parent):
                if c not in result:
                    result.append(c)
        return result


# =========================================================================
#  VirtualCallResolver
# =========================================================================

@dataclass
class VirtualCallResolver:
    """Resolves virtual/dynamic method calls using class hierarchy."""

    _cha: ClassHierarchyAnalysis = field(default_factory=ClassHierarchyAnalysis)
    _receiver_types: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def set_hierarchy(self, cha: ClassHierarchyAnalysis) -> None:
        self._cha = cha

    def set_receiver_type(self, variable: str, types: Set[str]) -> None:
        self._receiver_types[variable] = types

    def resolve(self, receiver: str, method: str) -> List[str]:
        """Resolve a virtual call receiver.method to concrete targets."""
        receiver_types = self._receiver_types.get(receiver, set())
        if not receiver_types:
            return self._resolve_by_name(receiver, method)
        targets: list[str] = []
        for rtype in receiver_types:
            resolved = self._cha.resolve_method(rtype, method)
            if resolved:
                targets.append(resolved)
            for sub in self._cha.concrete_subclasses(rtype):
                resolved = self._cha.resolve_method(sub, method)
                if resolved and resolved not in targets:
                    targets.append(resolved)
        return targets

    def _resolve_by_name(self, receiver: str, method: str) -> List[str]:
        parts = receiver.split(".")
        cls_name = parts[0] if parts else receiver
        resolved = self._cha.resolve_method(cls_name, method)
        if resolved:
            return [resolved]
        return []

    def narrow_receiver(self, variable: str, rtype: str) -> None:
        self._receiver_types[variable] = {rtype}

    def widen_receiver(self, variable: str, rtype: str) -> None:
        self._receiver_types[variable].add(rtype)


# =========================================================================
#  InliningDecision
# =========================================================================

@dataclass
class InliningDecision:
    """Decides when to inline a function vs use its summary."""

    max_inline_size: int = 20
    max_inline_depth: int = 3
    inline_pure: bool = True
    inline_once_called: bool = True
    _inline_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def should_inline(
        self,
        func: FunctionIR,
        summary: Optional[FunctionSummary],
        call_graph: CallGraph,
        depth: int = 0,
    ) -> bool:
        if depth >= self.max_inline_depth:
            return False
        instr_count = sum(len(bb.instructions) for bb in func.blocks.values())
        if instr_count > self.max_inline_size:
            return False
        if call_graph.is_recursive(func.name):
            return False
        callers = call_graph.callers(func.name)
        if self.inline_once_called and len(callers) == 1:
            return True
        if self.inline_pure and summary and summary.is_pure:
            return True
        if instr_count <= 5:
            return True
        return False

    def record_inline(self, func_name: str) -> None:
        self._inline_count[func_name] += 1

    def inline_count(self, func_name: str) -> int:
        return self._inline_count.get(func_name, 0)


# =========================================================================
#  RecursionHandler
# =========================================================================

@dataclass
class RecursionHandler:
    """Handles recursive and mutually-recursive functions with widening."""

    max_unroll: int = 3
    _iteration_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _previous_summaries: Dict[str, FunctionSummary] = field(default_factory=dict)

    def is_recursive_scc(self, scc: List[str]) -> bool:
        return len(scc) > 1

    def should_widen(self, func_name: str) -> bool:
        return self._iteration_count[func_name] >= self.max_unroll

    def record_iteration(self, func_name: str, summary: FunctionSummary) -> None:
        self._iteration_count[func_name] += 1
        self._previous_summaries[func_name] = summary.clone()

    def widen_summary(self, func_name: str, current: FunctionSummary) -> FunctionSummary:
        prev = self._previous_summaries.get(func_name)
        if prev is None:
            return current
        widened_params: dict[str, RefinementType] = {}
        for k in set(current.param_types) | set(prev.param_types):
            cur_t = current.param_types.get(k, RefinementType("⊥"))
            pre_t = prev.param_types.get(k, RefinementType("⊥"))
            widened_params[k] = RefinementType(
                cur_t.base if cur_t.base == pre_t.base else "object",
                "true",
            )
        widened_return = RefinementType(
            current.return_type.base if current.return_type.base == prev.return_type.base else "object",
            "true",
        )
        return FunctionSummary(
            name=func_name,
            param_types=widened_params,
            return_type=widened_return,
            preconditions=[],
            postconditions=[],
            side_effects=current.side_effects.merge(prev.side_effects),
            raises=current.raises | prev.raises,
            is_pure=current.is_pure and prev.is_pure,
            is_total=False,
            version=current.version + 1,
        )

    def converged(self, func_name: str, current: FunctionSummary) -> bool:
        prev = self._previous_summaries.get(func_name)
        if prev is None:
            return False
        return prev.subsumes(current)

    def reset(self, func_name: str) -> None:
        self._iteration_count[func_name] = 0
        self._previous_summaries.pop(func_name, None)


# =========================================================================
#  BottomUpAnalyzer
# =========================================================================

T = TypeVar("T")


@dataclass
class BottomUpAnalyzer:
    """Analyzes strongly connected components of the call graph bottom-up."""

    summary_table: SummaryTable = field(default_factory=SummaryTable)
    recursion_handler: RecursionHandler = field(default_factory=RecursionHandler)
    statistics: AnalysisStatistics = field(default_factory=AnalysisStatistics)
    _analyze_function: Optional[Callable[[FunctionIR, Dict[str, RefinementType]], FunctionSummary]] = None

    def set_analyzer(
        self, analyzer: Callable[[FunctionIR, Dict[str, RefinementType]], FunctionSummary],
    ) -> None:
        self._analyze_function = analyzer

    def analyze(
        self,
        call_graph: CallGraph,
        functions: Dict[str, FunctionIR],
    ) -> SummaryTable:
        sccs = call_graph.strongly_connected_components()
        # Bottom-up: reverse the SCC list (Tarjan produces reverse topo)
        for scc in reversed(sccs):
            self.statistics.scc_count += 1
            if len(scc) == 1 and not call_graph.is_recursive(scc[0]):
                self._analyze_single(scc[0], functions)
            else:
                self._analyze_scc(scc, functions, call_graph)
        return self.summary_table

    def _analyze_single(self, func_name: str, functions: Dict[str, FunctionIR]) -> None:
        func = functions.get(func_name)
        if func is None or self._analyze_function is None:
            return
        self.statistics.record_function()
        callee_env = self._build_callee_env(func_name)
        summary = self._analyze_function(func, callee_env)
        self.summary_table.put(summary)
        self.statistics.record_summary()

    def _analyze_scc(
        self,
        scc: List[str],
        functions: Dict[str, FunctionIR],
        call_graph: CallGraph,
    ) -> None:
        if self._analyze_function is None:
            return
        self.statistics.recursive_functions += len(scc)
        for func_name in scc:
            init_summary = FunctionSummary(name=func_name)
            self.summary_table.put(init_summary)

        changed = True
        while changed:
            changed = False
            self.statistics.record_iteration()
            for func_name in scc:
                func = functions.get(func_name)
                if func is None:
                    continue
                self.statistics.record_function()
                callee_env = self._build_callee_env(func_name)
                new_summary = self._analyze_function(func, callee_env)
                self.recursion_handler.record_iteration(func_name, new_summary)

                if self.recursion_handler.should_widen(func_name):
                    new_summary = self.recursion_handler.widen_summary(func_name, new_summary)

                if self.summary_table.put(new_summary):
                    changed = True
                    self.statistics.record_summary()

                if self.recursion_handler.converged(func_name, new_summary):
                    continue

    def _build_callee_env(self, func_name: str) -> Dict[str, RefinementType]:
        env: dict[str, RefinementType] = {}
        for name in self.summary_table.all_names():
            summaries = self.summary_table.get_all(name)
            if summaries:
                env[name] = summaries[0].return_type
        return env


# =========================================================================
#  TopDownPropagator
# =========================================================================

@dataclass
class TopDownPropagator:
    """Propagates caller constraints to callees (top-down phase)."""

    summary_table: SummaryTable = field(default_factory=SummaryTable)
    parameter_binding: ParameterBinding = field(default_factory=ParameterBinding)

    def propagate(
        self,
        call_graph: CallGraph,
        functions: Dict[str, FunctionIR],
        entry_constraints: Optional[Dict[str, Dict[str, RefinementType]]] = None,
    ) -> None:
        entry_constraints = entry_constraints or {}
        topo = call_graph.topological_order()
        for func_name in topo:
            func = functions.get(func_name)
            if func is None:
                continue
            constraints = entry_constraints.get(func_name, {})
            callers = call_graph.callers(func_name)
            for caller in callers:
                caller_summary = self.summary_table.get(caller)
                if caller_summary is None:
                    continue
                for edge_list in [call_graph.edges_between(caller, func_name)]:
                    for edge in edge_list:
                        if edge.argument_types:
                            bindings = self.parameter_binding.bind(
                                func.params,
                                list(edge.argument_types),
                            )
                            for p, t in bindings.items():
                                if p in constraints:
                                    constraints[p] = constraints[p].meet(t)
                                else:
                                    constraints[p] = t

            if constraints:
                summary = self.summary_table.get(func_name)
                if summary:
                    for p, t in constraints.items():
                        if p in summary.param_types:
                            summary.param_types[p] = summary.param_types[p].meet(t)
                        else:
                            summary.param_types[p] = t
                    self.summary_table.put(summary)

    def propagate_return(
        self,
        caller: str,
        callee: str,
        return_type: RefinementType,
        call_graph: CallGraph,
    ) -> None:
        summary = self.summary_table.get(caller)
        if summary is None:
            return
        edges = call_graph.edges_between(caller, callee)
        for edge in edges:
            if edge.call_site:
                # Record in summary that this call returns the given type
                pass


# =========================================================================
#  InterproceduralAnalyzer
# =========================================================================

@dataclass
class InterproceduralAnalyzer:
    """Main interprocedural analysis driver.

    Combines bottom-up and top-down phases, manages the worklist, and
    propagates function summaries across the call graph.
    """

    context_sensitivity: ContextSensitivity = field(default_factory=ContextSensitivity)
    call_graph: CallGraph = field(default_factory=CallGraph)
    summary_table: SummaryTable = field(default_factory=SummaryTable)
    statistics: AnalysisStatistics = field(default_factory=AnalysisStatistics)
    convergence: ConvergenceChecker = field(default_factory=ConvergenceChecker)
    worklist: WorklistScheduler = field(default_factory=WorklistScheduler)

    # sub-analyses
    side_effect_analysis: SideEffectAnalysis = field(default_factory=SideEffectAnalysis)
    purity_analysis: PurityAnalysis = field(default_factory=PurityAnalysis)
    alias_analysis: AliasAnalysis = field(default_factory=AliasAnalysis)
    exception_propagator: ExceptionPropagator = field(default_factory=ExceptionPropagator)
    closure_analyzer: ClosureAnalyzer = field(default_factory=ClosureAnalyzer)
    callback_analyzer: CallbackAnalyzer = field(default_factory=CallbackAnalyzer)
    module_analyzer: ModuleAnalyzer = field(default_factory=ModuleAnalyzer)
    cha: ClassHierarchyAnalysis = field(default_factory=ClassHierarchyAnalysis)
    virtual_resolver: VirtualCallResolver = field(default_factory=VirtualCallResolver)
    inlining: InliningDecision = field(default_factory=InliningDecision)
    recursion_handler: RecursionHandler = field(default_factory=RecursionHandler)
    bottom_up: BottomUpAnalyzer = field(default_factory=BottomUpAnalyzer)
    top_down: TopDownPropagator = field(default_factory=TopDownPropagator)

    _functions: Dict[str, FunctionIR] = field(default_factory=dict)
    _modules: List[ModuleIR] = field(default_factory=list)
    _analyze_fn: Optional[Callable[[FunctionIR, Dict[str, RefinementType]], FunctionSummary]] = None

    def set_intraprocedural_analyzer(
        self,
        analyzer: Callable[[FunctionIR, Dict[str, RefinementType]], FunctionSummary],
    ) -> None:
        self._analyze_fn = analyzer
        self.bottom_up.set_analyzer(analyzer)

    def add_module(self, module: ModuleIR) -> None:
        self._modules.append(module)
        for name, func in module.functions.items():
            fqn = f"{module.name}.{name}"
            self._functions[fqn] = func
            self._functions[name] = func
        for cls_name, cls_ir in module.classes.items():
            self.cha.add_class(cls_ir)
            for meth_name, meth in cls_ir.methods.items():
                fqn = f"{cls_name}.{meth_name}"
                self._functions[fqn] = meth
        self.module_analyzer.analyze(module)

    def analyze(self) -> SummaryTable:
        self.statistics.start()

        # Phase 1: Build call graph
        builder = CallGraphBuilder()
        self.call_graph = builder.build(self._modules)

        # Phase 2: Run sub-analyses
        self.virtual_resolver.set_hierarchy(self.cha)
        for name, func in self._functions.items():
            self.side_effect_analysis.analyze(func)
            self.alias_analysis.analyze_function(func)

        self.side_effect_analysis.propagate(self.call_graph)
        self.purity_analysis.analyze_all(
            self._functions, self.side_effect_analysis, self.call_graph,
        )

        # Phase 3: Bottom-up analysis
        self.bottom_up.summary_table = self.summary_table
        self.bottom_up.recursion_handler = self.recursion_handler
        self.bottom_up.statistics = self.statistics
        self.bottom_up.analyze(self.call_graph, self._functions)

        # Phase 4: Top-down propagation
        self.top_down.summary_table = self.summary_table
        self.top_down.propagate(self.call_graph, self._functions)

        # Phase 5: Iterate until convergence
        self._iterate()

        # Phase 6: Exception propagation
        self.exception_propagator.propagate(self.call_graph)

        self.statistics.stop()
        self.statistics.unresolved_calls = len(self.call_graph.unresolved_sites())
        self.statistics.edges_added = self.call_graph.edge_count()
        return self.summary_table

    def _iterate(self) -> None:
        roots = self.call_graph.roots()
        self.worklist.add_all(list(roots), WorklistPriority.HIGH)
        for node in self.call_graph.all_nodes():
            self.worklist.add(node)

        while not self.worklist.is_empty():
            func_name = self.worklist.pop()
            if func_name is None:
                break
            func = self._functions.get(func_name)
            if func is None:
                continue
            if self._analyze_fn is None:
                continue

            self.statistics.record_iteration()
            callee_env = self.bottom_up._build_callee_env(func_name)
            new_summary = self._analyze_fn(func, callee_env)

            if self.summary_table.put(new_summary):
                self.statistics.record_summary()
                for caller in self.call_graph.callers(func_name):
                    self.worklist.add(caller, WorklistPriority.HIGH)

            if self.convergence.check(self.summary_table):
                break

    def get_summary(self, func_name: str) -> Optional[FunctionSummary]:
        return self.summary_table.get(func_name)

    def get_statistics(self) -> AnalysisStatistics:
        return self.statistics

    def get_call_graph(self) -> CallGraph:
        return self.call_graph

    def is_pure(self, func_name: str) -> bool:
        return self.purity_analysis.is_pure(func_name)

    def side_effects(self, func_name: str) -> Optional[SideEffects]:
        return self.side_effect_analysis.get(func_name)

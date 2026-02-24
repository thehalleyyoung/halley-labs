"""
Memory model compiler: compile memory model specifications into executable models.

Supports axiomatic models, operational models, model comparison, hierarchy checking,
fence/scope semantics, model composition, and DOT visualization.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum
import copy
import itertools


class RelationType(Enum):
    PROGRAM_ORDER = "po"
    READS_FROM = "rf"
    COHERENCE_ORDER = "co"
    FROM_READS = "fr"
    HAPPENS_BEFORE = "hb"
    MODIFICATION_ORDER = "mo"
    SYNCHRONIZES_WITH = "sw"
    SEQUENCED_BEFORE = "sb"
    DATA_DEPENDENCY = "data"
    CONTROL_DEPENDENCY = "ctrl"
    ADDRESS_DEPENDENCY = "addr"
    FENCE_ORDER = "fence"


class FenceType(Enum):
    FULL = "full"
    STORE_STORE = "store_store"
    LOAD_LOAD = "load_load"
    LOAD_STORE = "load_store"
    STORE_LOAD = "store_load"
    ACQUIRE = "acquire"
    RELEASE = "release"
    SEQ_CST = "seq_cst"


class MemoryOrder(Enum):
    RELAXED = "relaxed"
    CONSUME = "consume"
    ACQUIRE = "acquire"
    RELEASE = "release"
    ACQ_REL = "acq_rel"
    SEQ_CST = "seq_cst"


class ScopeLevel(Enum):
    THREAD = "thread"
    WARP = "warp"
    CTA = "cta"
    GPU = "gpu"
    SYSTEM = "system"


@dataclass
class Event:
    eid: int
    thread_id: int
    event_type: str  # "R", "W", "F", "RMW"
    variable: str = ""
    value: int = 0
    memory_order: MemoryOrder = MemoryOrder.SEQ_CST
    scope: ScopeLevel = ScopeLevel.SYSTEM
    label: str = ""

    def __hash__(self):
        return hash(self.eid)

    def __eq__(self, other):
        return isinstance(other, Event) and self.eid == other.eid

    def __repr__(self):
        return f"E{self.eid}(T{self.thread_id}:{self.event_type} {self.variable}={self.value})"


@dataclass
class Relation:
    rel_type: RelationType
    pairs: Set[Tuple[int, int]] = field(default_factory=set)

    def add(self, src: int, dst: int):
        self.pairs.add((src, dst))

    def contains(self, src: int, dst: int) -> bool:
        return (src, dst) in self.pairs

    def compose(self, other: 'Relation') -> 'Relation':
        result = Relation(rel_type=self.rel_type)
        for a, b in self.pairs:
            for c, d in other.pairs:
                if b == c:
                    result.add(a, d)
        return result

    def transitive_closure(self) -> 'Relation':
        result = Relation(rel_type=self.rel_type)
        result.pairs = set(self.pairs)
        changed = True
        while changed:
            changed = False
            new_pairs = set()
            for a, b in result.pairs:
                for c, d in result.pairs:
                    if b == c and (a, d) not in result.pairs:
                        new_pairs.add((a, d))
                        changed = True
            result.pairs |= new_pairs
        return result

    def union(self, other: 'Relation') -> 'Relation':
        result = Relation(rel_type=self.rel_type)
        result.pairs = self.pairs | other.pairs
        return result

    def intersect(self, other: 'Relation') -> 'Relation':
        result = Relation(rel_type=self.rel_type)
        result.pairs = self.pairs & other.pairs
        return result

    def is_acyclic(self, events: List[Event]) -> bool:
        adj: Dict[int, Set[int]] = {}
        for src, dst in self.pairs:
            adj.setdefault(src, set()).add(dst)
        visited = set()
        in_stack = set()

        def dfs(node):
            visited.add(node)
            in_stack.add(node)
            for neighbor in adj.get(node, set()):
                if neighbor in in_stack:
                    return False
                if neighbor not in visited:
                    if not dfs(neighbor):
                        return False
            in_stack.discard(node)
            return True

        for e in events:
            if e.eid not in visited:
                if not dfs(e.eid):
                    return False
        return True

    def is_total_on(self, event_ids: Set[int]) -> bool:
        for a in event_ids:
            for b in event_ids:
                if a != b:
                    if not self.contains(a, b) and not self.contains(b, a):
                        return False
        return True


@dataclass
class Axiom:
    name: str
    description: str
    check_fn: Optional[Callable] = None

    def check(self, events: List[Event], relations: Dict[str, Relation]) -> bool:
        if self.check_fn:
            return self.check_fn(events, relations)
        return True


@dataclass
class Execution:
    events: List[Event] = field(default_factory=list)
    relations: Dict[str, Relation] = field(default_factory=dict)
    initial_state: Dict[str, int] = field(default_factory=dict)
    final_state: Dict[str, int] = field(default_factory=dict)

    def add_event(self, event: Event):
        self.events.append(event)

    def add_relation(self, name: str, src: int, dst: int):
        if name not in self.relations:
            self.relations[name] = Relation(rel_type=RelationType.PROGRAM_ORDER)
        self.relations[name].add(src, dst)

    def get_writes(self, variable: str) -> List[Event]:
        return [e for e in self.events if e.event_type == "W" and e.variable == variable]

    def get_reads(self, variable: str) -> List[Event]:
        return [e for e in self.events if e.event_type == "R" and e.variable == variable]


@dataclass
class ModelSpec:
    name: str
    relations: List[str] = field(default_factory=list)
    axioms: List[Axiom] = field(default_factory=list)
    fence_types: List[FenceType] = field(default_factory=list)
    scope_levels: List[ScopeLevel] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


class ExecutableModel:
    """Compiled executable memory model."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.name = spec.name
        self.axioms = spec.axioms
        self.fence_rules: Dict[FenceType, Callable] = {}
        self.scope_rules: Dict[ScopeLevel, Callable] = {}
        self._derived_relations: Dict[str, Callable] = {}

    def check_execution(self, execution: Execution) -> Tuple[bool, List[str]]:
        violations = []
        derived = self._compute_derived_relations(execution)
        all_rels = {**execution.relations, **derived}

        for axiom in self.axioms:
            if not axiom.check(execution.events, all_rels):
                violations.append(f"Axiom '{axiom.name}' violated: {axiom.description}")

        return len(violations) == 0, violations

    def _compute_derived_relations(self, execution: Execution) -> Dict[str, Relation]:
        derived = {}
        for name, compute_fn in self._derived_relations.items():
            derived[name] = compute_fn(execution)
        return derived

    def add_derived_relation(self, name: str, compute_fn: Callable):
        self._derived_relations[name] = compute_fn

    def add_fence_rule(self, fence_type: FenceType, rule_fn: Callable):
        self.fence_rules[fence_type] = rule_fn

    def add_scope_rule(self, scope: ScopeLevel, rule_fn: Callable):
        self.scope_rules[scope] = rule_fn

    def enumerate_outcomes(self, events: List[Event],
                           po: Relation) -> List[Execution]:
        """Enumerate all valid executions of the given events under this model."""
        variables = set(e.variable for e in events if e.variable)
        writes_by_var: Dict[str, List[Event]] = {}
        reads_by_var: Dict[str, List[Event]] = {}

        for e in events:
            if e.event_type == "W":
                writes_by_var.setdefault(e.variable, []).append(e)
            elif e.event_type == "R":
                reads_by_var.setdefault(e.variable, []).append(e)

        valid_executions = []

        rf_options = self._enumerate_rf(reads_by_var, writes_by_var)

        for rf_assignment in rf_options:
            for co_perm in self._enumerate_co(writes_by_var):
                exec_ = Execution(events=events)
                exec_.relations["po"] = po
                rf_rel = Relation(rel_type=RelationType.READS_FROM)
                for read_eid, write_eid in rf_assignment.items():
                    rf_rel.add(write_eid, read_eid)
                exec_.relations["rf"] = rf_rel

                co_rel = Relation(rel_type=RelationType.COHERENCE_ORDER)
                for pair in co_perm:
                    co_rel.add(pair[0], pair[1])
                exec_.relations["co"] = co_rel

                fr_rel = self._compute_fr(rf_rel, co_rel)
                exec_.relations["fr"] = fr_rel

                allowed, _ = self.check_execution(exec_)
                if allowed:
                    valid_executions.append(exec_)

        return valid_executions

    def _enumerate_rf(self, reads_by_var, writes_by_var) -> List[Dict[int, int]]:
        if not reads_by_var:
            return [{}]

        all_reads = []
        for var, reads in reads_by_var.items():
            for r in reads:
                writes = writes_by_var.get(var, [])
                options = [w.eid for w in writes]
                if not options:
                    options = [-1]  # init write
                all_reads.append((r.eid, options))

        if not all_reads:
            return [{}]

        results = []
        option_lists = [opts for _, opts in all_reads]
        read_ids = [rid for rid, _ in all_reads]

        for combo in itertools.product(*option_lists):
            assignment = {}
            for i, rid in enumerate(read_ids):
                assignment[rid] = combo[i]
            results.append(assignment)

        return results[:1000]  # cap for performance

    def _enumerate_co(self, writes_by_var) -> List[List[Tuple[int, int]]]:
        all_pairs = []
        for var, writes in writes_by_var.items():
            if len(writes) <= 1:
                all_pairs.append([])
                continue
            write_ids = [w.eid for w in writes]
            perms = list(itertools.permutations(write_ids))
            var_orders = []
            for perm in perms[:100]:
                pairs = []
                for i in range(len(perm)):
                    for j in range(i + 1, len(perm)):
                        pairs.append((perm[i], perm[j]))
                var_orders.append(pairs)
            all_pairs.append(var_orders)

        if not all_pairs or all(not p for p in all_pairs):
            return [[]]

        non_empty = [p if p else [[]] for p in all_pairs]
        results = []
        for combo in itertools.product(*non_empty):
            merged = []
            for part in combo:
                merged.extend(part)
            results.append(merged)
        return results[:1000]

    def _compute_fr(self, rf: Relation, co: Relation) -> Relation:
        fr = Relation(rel_type=RelationType.FROM_READS)
        rf_inv: Dict[int, int] = {}
        for src, dst in rf.pairs:
            rf_inv[dst] = src

        for read_eid, write_eid in rf_inv.items():
            for co_src, co_dst in co.pairs:
                if co_src == write_eid:
                    fr.add(read_eid, co_dst)
        return fr


class StoreBuffer:
    """Operational store buffer for TSO simulation."""

    def __init__(self, n_threads: int):
        self.n_threads = n_threads
        self.buffers: Dict[int, List[Tuple[str, int]]] = {i: [] for i in range(n_threads)}
        self.memory: Dict[str, int] = {}

    def write(self, thread_id: int, variable: str, value: int):
        self.buffers[thread_id].append((variable, value))

    def read(self, thread_id: int, variable: str) -> int:
        for var, val in reversed(self.buffers[thread_id]):
            if var == variable:
                return val
        return self.memory.get(variable, 0)

    def flush(self, thread_id: int) -> bool:
        if not self.buffers[thread_id]:
            return False
        var, val = self.buffers[thread_id].pop(0)
        self.memory[var] = val
        return True

    def flush_all(self, thread_id: int):
        while self.flush(thread_id):
            pass

    def is_empty(self, thread_id: int) -> bool:
        return len(self.buffers[thread_id]) == 0

    def all_empty(self) -> bool:
        return all(self.is_empty(tid) for tid in range(self.n_threads))

    def get_state(self) -> Tuple:
        buf_state = tuple(tuple(b) for b in
                          [self.buffers[i] for i in range(self.n_threads)])
        mem_state = tuple(sorted(self.memory.items()))
        return (buf_state, mem_state)


class OperationalModel:
    """Operational model: simulates execution with store buffers and write queues."""

    def __init__(self, model_type: str = "TSO"):
        self.model_type = model_type
        self.store_buffer = None

    def simulate(self, events_by_thread: Dict[int, List[Event]],
                 initial_state: Dict[str, int] = None) -> List[Dict[str, int]]:
        n_threads = len(events_by_thread)
        self.store_buffer = StoreBuffer(n_threads)
        if initial_state:
            self.store_buffer.memory = dict(initial_state)

        outcomes = set()
        pcs = {tid: 0 for tid in events_by_thread}
        self._explore_states(events_by_thread, pcs, outcomes, set(), 0)
        return [dict(o) for o in outcomes]

    def _explore_states(self, events_by_thread: Dict[int, List[Event]],
                        pcs: Dict[int, int],
                        outcomes: Set,
                        visited: Set,
                        depth: int):
        if depth > 500:
            return

        state = self._get_state(pcs)
        if state in visited:
            return
        visited.add(state)

        all_done = all(pcs[tid] >= len(events_by_thread[tid])
                       for tid in events_by_thread)
        if all_done and self.store_buffer.all_empty():
            outcome = tuple(sorted(self.store_buffer.memory.items()))
            outcomes.add(outcome)
            return

        for tid in events_by_thread:
            if pcs[tid] < len(events_by_thread[tid]):
                event = events_by_thread[tid][pcs[tid]]
                saved = self._save_state()
                self._execute_event(event)
                pcs[tid] += 1
                self._explore_states(events_by_thread, pcs, outcomes, visited, depth + 1)
                pcs[tid] -= 1
                self._restore_state(saved)

            if not self.store_buffer.is_empty(tid):
                saved = self._save_state()
                self.store_buffer.flush(tid)
                self._explore_states(events_by_thread, pcs, outcomes, visited, depth + 1)
                self._restore_state(saved)

    def _execute_event(self, event: Event):
        if event.event_type == "W":
            if self.model_type == "TSO":
                self.store_buffer.write(event.thread_id, event.variable, event.value)
            else:
                self.store_buffer.memory[event.variable] = event.value
        elif event.event_type == "R":
            event.value = self.store_buffer.read(event.thread_id, event.variable)
        elif event.event_type == "F":
            if self.model_type == "TSO":
                self.store_buffer.flush_all(event.thread_id)

    def _get_state(self, pcs: Dict[int, int]) -> Tuple:
        return (tuple(sorted(pcs.items())), self.store_buffer.get_state())

    def _save_state(self) -> Dict:
        return {
            "buffers": {tid: list(buf) for tid, buf in self.store_buffer.buffers.items()},
            "memory": dict(self.store_buffer.memory),
        }

    def _restore_state(self, saved: Dict):
        self.store_buffer.buffers = {tid: list(buf) for tid, buf in saved["buffers"].items()}
        self.store_buffer.memory = dict(saved["memory"])


class ModelComparator:
    """Compare two memory models."""

    def __init__(self):
        pass

    def compare(self, model_a: ExecutableModel, model_b: ExecutableModel,
                test_executions: List[Execution]) -> Dict[str, Any]:
        a_only = []
        b_only = []
        both_allow = []
        both_forbid = []

        for exec_ in test_executions:
            a_ok, _ = model_a.check_execution(exec_)
            b_ok, _ = model_b.check_execution(exec_)

            if a_ok and b_ok:
                both_allow.append(exec_)
            elif a_ok and not b_ok:
                a_only.append(exec_)
            elif not a_ok and b_ok:
                b_only.append(exec_)
            else:
                both_forbid.append(exec_)

        return {
            "a_only": a_only,
            "b_only": b_only,
            "both_allow": both_allow,
            "both_forbid": both_forbid,
            "a_weaker": len(a_only) > 0 and len(b_only) == 0,
            "b_weaker": len(b_only) > 0 and len(a_only) == 0,
            "equivalent": len(a_only) == 0 and len(b_only) == 0,
            "incomparable": len(a_only) > 0 and len(b_only) > 0,
        }

    def find_distinguishing_test(self, model_a: ExecutableModel,
                                 model_b: ExecutableModel,
                                 test_executions: List[Execution]) -> Optional[Execution]:
        for exec_ in test_executions:
            a_ok, _ = model_a.check_execution(exec_)
            b_ok, _ = model_b.check_execution(exec_)
            if a_ok != b_ok:
                return exec_
        return None

    def is_weaker(self, model_a: ExecutableModel, model_b: ExecutableModel,
                  test_executions: List[Execution]) -> bool:
        for exec_ in test_executions:
            a_ok, _ = model_a.check_execution(exec_)
            b_ok, _ = model_b.check_execution(exec_)
            if b_ok and not a_ok:
                return False
        return True


class FenceSemantics:
    """Define what each fence type orders."""

    def __init__(self):
        self.fence_effects: Dict[FenceType, Dict[str, bool]] = {
            FenceType.FULL: {"WW": True, "WR": True, "RW": True, "RR": True},
            FenceType.STORE_STORE: {"WW": True, "WR": False, "RW": False, "RR": False},
            FenceType.LOAD_LOAD: {"WW": False, "WR": False, "RW": False, "RR": True},
            FenceType.LOAD_STORE: {"WW": False, "WR": False, "RW": True, "RR": False},
            FenceType.STORE_LOAD: {"WW": False, "WR": True, "RW": False, "RR": False},
            FenceType.ACQUIRE: {"WW": False, "WR": False, "RW": True, "RR": True},
            FenceType.RELEASE: {"WW": True, "WR": True, "RW": False, "RR": False},
            FenceType.SEQ_CST: {"WW": True, "WR": True, "RW": True, "RR": True},
        }

    def orders(self, fence_type: FenceType, before_type: str, after_type: str) -> bool:
        key = f"{before_type}{after_type}"
        effects = self.fence_effects.get(fence_type, {})
        return effects.get(key, False)

    def minimum_fence(self, required_orderings: Set[str]) -> Optional[FenceType]:
        for ft in [FenceType.STORE_STORE, FenceType.LOAD_LOAD,
                   FenceType.LOAD_STORE, FenceType.STORE_LOAD,
                   FenceType.ACQUIRE, FenceType.RELEASE, FenceType.FULL]:
            effects = self.fence_effects[ft]
            if all(effects.get(r, False) for r in required_orderings):
                return ft
        return FenceType.SEQ_CST

    def compute_fence_relation(self, events: List[Event],
                               fence_type: FenceType) -> Relation:
        rel = Relation(rel_type=RelationType.FENCE_ORDER)
        fences = [e for e in events if e.event_type == "F"]

        for fence in fences:
            same_thread = [e for e in events if e.thread_id == fence.thread_id]
            before = [e for e in same_thread if e.eid < fence.eid and e.event_type in ("R", "W")]
            after = [e for e in same_thread if e.eid > fence.eid and e.event_type in ("R", "W")]

            for b in before:
                for a in after:
                    b_type = b.event_type
                    a_type = a.event_type
                    if self.orders(fence_type, b_type, a_type):
                        rel.add(b.eid, a.eid)
        return rel


class ScopeSemantics:
    """Define visibility rules for scoped memory models."""

    def __init__(self):
        self.scope_hierarchy = [
            ScopeLevel.THREAD, ScopeLevel.WARP, ScopeLevel.CTA,
            ScopeLevel.GPU, ScopeLevel.SYSTEM
        ]
        self.scope_tree: Dict[int, Dict[ScopeLevel, int]] = {}

    def set_scope_tree(self, tree: Dict[int, Dict[ScopeLevel, int]]):
        self.scope_tree = tree

    def same_scope(self, tid1: int, tid2: int, scope: ScopeLevel) -> bool:
        if tid1 not in self.scope_tree or tid2 not in self.scope_tree:
            return True
        s1 = self.scope_tree[tid1].get(scope, 0)
        s2 = self.scope_tree[tid2].get(scope, 0)
        return s1 == s2

    def visible(self, write_event: Event, read_event: Event) -> bool:
        write_scope = write_event.scope
        w_idx = self.scope_hierarchy.index(write_scope)
        needed_scope = self._minimum_visibility_scope(
            write_event.thread_id, read_event.thread_id)
        n_idx = self.scope_hierarchy.index(needed_scope)
        return w_idx >= n_idx

    def _minimum_visibility_scope(self, tid1: int, tid2: int) -> ScopeLevel:
        if tid1 == tid2:
            return ScopeLevel.THREAD
        for scope in self.scope_hierarchy:
            if self.same_scope(tid1, tid2, scope):
                return scope
        return ScopeLevel.SYSTEM

    def filter_rf_by_scope(self, rf: Relation, events: List[Event]) -> Relation:
        event_map = {e.eid: e for e in events}
        filtered = Relation(rel_type=RelationType.READS_FROM)
        for src, dst in rf.pairs:
            w = event_map.get(src)
            r = event_map.get(dst)
            if w and r and self.visible(w, r):
                filtered.add(src, dst)
        return filtered


class ModelComposer:
    """Compose two memory models (e.g., CPU + GPU)."""

    def compose(self, model_a: ExecutableModel, model_b: ExecutableModel,
                boundary_spec: Dict[str, Any] = None) -> ExecutableModel:
        combined_spec = ModelSpec(
            name=f"{model_a.name}+{model_b.name}",
            relations=list(set(model_a.spec.relations + model_b.spec.relations)),
            axioms=model_a.spec.axioms + model_b.spec.axioms,
            fence_types=list(set(model_a.spec.fence_types + model_b.spec.fence_types)),
            scope_levels=list(set(model_a.spec.scope_levels + model_b.spec.scope_levels)),
        )

        composed = ExecutableModel(combined_spec)

        for ft, fn in model_a.fence_rules.items():
            composed.add_fence_rule(ft, fn)
        for ft, fn in model_b.fence_rules.items():
            if ft not in composed.fence_rules:
                composed.add_fence_rule(ft, fn)

        for sl, fn in model_a.scope_rules.items():
            composed.add_scope_rule(sl, fn)
        for sl, fn in model_b.scope_rules.items():
            if sl not in composed.scope_rules:
                composed.add_scope_rule(sl, fn)

        return composed


class DOTVisualizer:
    """Generate DOT graph visualization of memory model relations."""

    def visualize_execution(self, execution: Execution) -> str:
        lines = ["digraph execution {", '  rankdir=TB;', '  node [shape=record];']

        threads: Dict[int, List[Event]] = {}
        for e in execution.events:
            threads.setdefault(e.thread_id, []).append(e)

        for tid, events in sorted(threads.items()):
            lines.append(f'  subgraph cluster_T{tid} {{')
            lines.append(f'    label="Thread {tid}";')
            for e in sorted(events, key=lambda x: x.eid):
                label = f"{e.event_type}({e.variable},{e.value})"
                lines.append(f'    E{e.eid} [label="{label}"];')
            lines.append('  }')

        colors = {
            "po": "black", "rf": "red", "co": "blue",
            "fr": "green", "hb": "purple", "fence": "orange",
        }

        for rel_name, relation in execution.relations.items():
            color = colors.get(rel_name, "gray")
            for src, dst in relation.pairs:
                lines.append(
                    f'  E{src} -> E{dst} [color={color}, label="{rel_name}"];')

        lines.append("}")
        return "\n".join(lines)

    def visualize_model_hierarchy(self, models: Dict[str, ExecutableModel],
                                  comparisons: Dict[Tuple[str, str], str]) -> str:
        lines = ["digraph hierarchy {", '  rankdir=BT;']
        for name in models:
            lines.append(f'  "{name}";')
        for (a, b), rel in comparisons.items():
            if rel == "weaker":
                lines.append(f'  "{a}" -> "{b}" [label="weaker"];')
            elif rel == "stronger":
                lines.append(f'  "{b}" -> "{a}" [label="weaker"];')
        lines.append("}")
        return "\n".join(lines)


class MemoryModelCompiler:
    """Main compiler: takes a model specification and produces an executable model."""

    def __init__(self):
        self.fence_semantics = FenceSemantics()
        self.scope_semantics = ScopeSemantics()
        self.comparator = ModelComparator()
        self.composer = ModelComposer()
        self.visualizer = DOTVisualizer()
        self._builtin_models: Dict[str, Callable] = {
            "SC": self._build_sc,
            "TSO": self._build_tso,
            "PSO": self._build_pso,
            "ARM": self._build_arm,
            "POWER": self._build_power,
        }

    def compile(self, model_spec: ModelSpec) -> ExecutableModel:
        if model_spec.name in self._builtin_models:
            return self._builtin_models[model_spec.name]()

        model = ExecutableModel(model_spec)

        for axiom in model_spec.axioms:
            if axiom.check_fn is None:
                axiom.check_fn = self._default_axiom_check(axiom.name)

        return model

    def compile_builtin(self, name: str) -> ExecutableModel:
        if name in self._builtin_models:
            return self._builtin_models[name]()
        raise ValueError(f"Unknown builtin model: {name}")

    def _build_sc(self) -> ExecutableModel:
        def sc_axiom(events, relations):
            if "po" in relations and "rf" in relations:
                hb = relations["po"].union(relations["rf"])
                if "fr" in relations:
                    hb = hb.union(relations["fr"])
                if "co" in relations:
                    hb = hb.union(relations["co"])
                tc = hb.transitive_closure()
                return tc.is_acyclic(events)
            return True

        spec = ModelSpec(
            name="SC",
            relations=["po", "rf", "co", "fr"],
            axioms=[Axiom("sc-per-location", "Total order consistent with po, rf, co, fr",
                          sc_axiom)],
        )
        return ExecutableModel(spec)

    def _build_tso(self) -> ExecutableModel:
        def tso_axiom(events, relations):
            if "po" not in relations or "rf" not in relations:
                return True
            po = relations["po"]
            rf = relations["rf"]
            co = relations.get("co", Relation(RelationType.COHERENCE_ORDER))
            fr = relations.get("fr", Relation(RelationType.FROM_READS))

            # TSO: po without store->load reorderings is preserved
            preserved_po = Relation(RelationType.PROGRAM_ORDER)
            event_map = {e.eid: e for e in events}
            for src, dst in po.pairs:
                s = event_map.get(src)
                d = event_map.get(dst)
                if s and d:
                    if not (s.event_type == "W" and d.event_type == "R"):
                        preserved_po.add(src, dst)

            combined = preserved_po.union(rf).union(co).union(fr)
            tc = combined.transitive_closure()
            return tc.is_acyclic(events)

        spec = ModelSpec(
            name="TSO",
            relations=["po", "rf", "co", "fr"],
            axioms=[Axiom("tso", "Preserve all orders except store-load", tso_axiom)],
            fence_types=[FenceType.STORE_LOAD, FenceType.FULL],
        )
        model = ExecutableModel(spec)
        model.add_fence_rule(FenceType.STORE_LOAD,
                             lambda e, rels: self._apply_mfence(e, rels))
        return model

    def _build_pso(self) -> ExecutableModel:
        def pso_axiom(events, relations):
            if "po" not in relations or "rf" not in relations:
                return True
            po = relations["po"]
            rf = relations["rf"]
            co = relations.get("co", Relation(RelationType.COHERENCE_ORDER))
            fr = relations.get("fr", Relation(RelationType.FROM_READS))

            preserved_po = Relation(RelationType.PROGRAM_ORDER)
            event_map = {e.eid: e for e in events}
            for src, dst in po.pairs:
                s = event_map.get(src)
                d = event_map.get(dst)
                if s and d:
                    if not (s.event_type == "W" and d.event_type in ("R", "W")
                            and s.variable != d.variable):
                        preserved_po.add(src, dst)

            combined = preserved_po.union(rf).union(co).union(fr)
            tc = combined.transitive_closure()
            return tc.is_acyclic(events)

        spec = ModelSpec(
            name="PSO",
            relations=["po", "rf", "co", "fr"],
            axioms=[Axiom("pso", "Allow store-store reordering to different addresses",
                          pso_axiom)],
            fence_types=[FenceType.STORE_STORE, FenceType.STORE_LOAD, FenceType.FULL],
        )
        return ExecutableModel(spec)

    def _build_arm(self) -> ExecutableModel:
        def arm_axiom(events, relations):
            if "po" not in relations:
                return True
            po = relations["po"]
            rf = relations.get("rf", Relation(RelationType.READS_FROM))
            co = relations.get("co", Relation(RelationType.COHERENCE_ORDER))
            fr = relations.get("fr", Relation(RelationType.FROM_READS))

            preserved_po = Relation(RelationType.PROGRAM_ORDER)
            event_map = {e.eid: e for e in events}
            for src, dst in po.pairs:
                s = event_map.get(src)
                d = event_map.get(dst)
                if s and d:
                    if s.variable == d.variable:
                        preserved_po.add(src, dst)
                    if s.event_type == "R" and d.event_type in ("R", "W"):
                        # Check data dependency
                        pass

            fence_rel = relations.get("fence", Relation(RelationType.FENCE_ORDER))
            combined = preserved_po.union(rf).union(co).union(fr).union(fence_rel)
            tc = combined.transitive_closure()
            return tc.is_acyclic(events)

        spec = ModelSpec(
            name="ARM",
            relations=["po", "rf", "co", "fr", "data", "ctrl", "addr"],
            axioms=[Axiom("arm", "ARM relaxed model with dependency tracking", arm_axiom)],
            fence_types=[FenceType.FULL, FenceType.ACQUIRE, FenceType.RELEASE],
        )
        return ExecutableModel(spec)

    def _build_power(self) -> ExecutableModel:
        def power_axiom(events, relations):
            if "po" not in relations:
                return True
            rf = relations.get("rf", Relation(RelationType.READS_FROM))
            co = relations.get("co", Relation(RelationType.COHERENCE_ORDER))
            fr = relations.get("fr", Relation(RelationType.FROM_READS))
            fence = relations.get("fence", Relation(RelationType.FENCE_ORDER))

            combined = rf.union(co).union(fr).union(fence)
            tc = combined.transitive_closure()
            return tc.is_acyclic(events)

        spec = ModelSpec(
            name="POWER",
            relations=["po", "rf", "co", "fr", "sync", "lwsync", "isync"],
            axioms=[Axiom("power", "POWER relaxed model", power_axiom)],
            fence_types=[FenceType.FULL, FenceType.STORE_STORE, FenceType.LOAD_LOAD],
        )
        return ExecutableModel(spec)

    def _apply_mfence(self, events: List[Event],
                      relations: Dict[str, Relation]) -> Relation:
        fence_rel = Relation(rel_type=RelationType.FENCE_ORDER)
        fences = [e for e in events if e.event_type == "F"]
        for f in fences:
            before = [e for e in events
                      if e.thread_id == f.thread_id and e.eid < f.eid]
            after = [e for e in events
                     if e.thread_id == f.thread_id and e.eid > f.eid]
            for b in before:
                for a in after:
                    fence_rel.add(b.eid, a.eid)
        return fence_rel

    def _default_axiom_check(self, axiom_name: str) -> Callable:
        def check(events, relations):
            return True
        return check

    def compare_models(self, model_a: ExecutableModel, model_b: ExecutableModel,
                       test_executions: List[Execution]) -> Dict[str, Any]:
        return self.comparator.compare(model_a, model_b, test_executions)

    def compose_models(self, model_a: ExecutableModel,
                       model_b: ExecutableModel) -> ExecutableModel:
        return self.composer.compose(model_a, model_b)

    def visualize(self, execution: Execution) -> str:
        return self.visualizer.visualize_execution(execution)


def build_sb_execution(x_val: int = 0, y_val: int = 0) -> Execution:
    """Build store-buffer (SB) litmus test execution."""
    e0 = Event(0, 0, "W", "x", 1)
    e1 = Event(1, 0, "R", "y", y_val)
    e2 = Event(2, 1, "W", "y", 1)
    e3 = Event(3, 1, "R", "x", x_val)

    exec_ = Execution(events=[e0, e1, e2, e3])
    exec_.add_relation("po", 0, 1)
    exec_.add_relation("po", 2, 3)

    if y_val == 0:
        pass  # reads from initial
    elif y_val == 1:
        exec_.add_relation("rf", 2, 1)
    if x_val == 0:
        pass
    elif x_val == 1:
        exec_.add_relation("rf", 0, 3)

    return exec_


def build_mp_execution(data_val: int = 0, flag_val: int = 0) -> Execution:
    """Build message-passing (MP) litmus test execution."""
    e0 = Event(0, 0, "W", "data", 1)
    e1 = Event(1, 0, "W", "flag", 1)
    e2 = Event(2, 1, "R", "flag", flag_val)
    e3 = Event(3, 1, "R", "data", data_val)

    exec_ = Execution(events=[e0, e1, e2, e3])
    exec_.add_relation("po", 0, 1)
    exec_.add_relation("po", 2, 3)

    if flag_val == 1:
        exec_.add_relation("rf", 1, 2)
    if data_val == 1:
        exec_.add_relation("rf", 0, 3)

    return exec_

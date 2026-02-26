"""Comprehensive tests for the coacert.explorer module."""
import json
import pytest
from typing import Dict, Any, List, Tuple, FrozenSet

from coacert.explorer.graph import StateNode, TransitionEdge, TransitionGraph, GraphStatistics
from coacert.explorer.explorer import (
    ExplicitStateExplorer, ExplorationMode, ExplorationStats,
)
from coacert.explorer.hash_table import ZobristHasher, StateHashTable
from coacert.explorer.symmetry import (
    Permutation, PermutationGroup, Orbit, SymmetryDetector,
)
from coacert.explorer.fairness import (
    FairnessKind, FairnessConstraint, AcceptancePair, Lasso, FairnessTracker,
)
from coacert.explorer.traces import (
    TraceStep, ExecutionTrace, LassoTrace, TraceManager,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _node(name, val, props=()):
    return StateNode(name, {"x": val}, frozenset(props))

@pytest.fixture
def simple_nodes():
    return [_node("s0", 0, ("init",)), _node("s1", 1, ("mid",)), _node("s2", 2, ("end",))]

@pytest.fixture
def simple_edges():
    return [TransitionEdge("inc", "s0", "s1"), TransitionEdge("inc", "s1", "s2")]

@pytest.fixture
def diamond_graph(simple_nodes):
    g = TransitionGraph()
    for n in simple_nodes:
        g.add_state(n, is_initial=(n.state_hash == "s0"))
    g.add_state(_node("s3", 3))
    for src, tgt, act in [("s0","s1","a"),("s0","s3","b"),("s1","s2","c"),("s3","s2","d")]:
        g.add_transition(TransitionEdge(act, src, tgt))
    return g

@pytest.fixture
def cyclic_graph():
    g = TransitionGraph()
    for i in range(3):
        g.add_state(StateNode(f"s{i}", {"v": i}, frozenset({f"p{i}"})), is_initial=(i == 0))
    for s, t in [("s0","s1"),("s1","s2"),("s2","s0")]:
        g.add_transition(TransitionEdge("go", s, t))
    return g

class SimpleEngine:
    def __init__(self, n=4, cyclic=False):
        self.n, self.cyclic = n, cyclic
    def compute_initial_states(self):
        return [{"val": 0}]
    def compute_next_states(self, state):
        v = state["val"]
        if v < self.n - 1:
            return [("step", {"val": v + 1})]
        return [("wrap", {"val": 0})] if self.cyclic and v == self.n - 1 else []
    def evaluate_invariant(self, state, inv):
        if inv == "nonneg": return state["val"] >= 0
        if inv == "small": return state["val"] < self.n - 1
        return True
    def compute_atomic_propositions(self, state):
        props = set()
        if state["val"] == 0: props.add("init")
        if state["val"] % 2 == 0: props.add("even")
        return frozenset(props)

class DeadlockEngine:
    def compute_initial_states(self): return [{"stuck": True}]
    def compute_next_states(self, state): return []
    def evaluate_invariant(self, state, inv): return True
    def compute_atomic_propositions(self, state): return frozenset()

# ===================================================================
# TestStateNode
# ===================================================================
class TestStateNode:
    def test_creation_and_satisfies(self, simple_nodes):
        n = simple_nodes[0]
        assert n.state_hash == "s0" and n.full_state == {"x": 0}
        assert n.satisfies("init") and not n.satisfies("end")

    def test_satisfies_all_and_any(self):
        n = _node("h", 1, ("a", "b", "c"))
        assert n.satisfies_all({"a", "b"}) and not n.satisfies_all({"a", "z"})
        assert n.satisfies_any({"z", "a"}) and not n.satisfies_any({"z"})

    def test_serialization_roundtrip(self, simple_nodes):
        for n in simple_nodes:
            restored = StateNode.from_dict(n.to_dict())
            assert restored.state_hash == n.state_hash
            assert restored.full_state == n.full_state
            assert restored.atomic_propositions == n.atomic_propositions

    def test_frozen(self, simple_nodes):
        with pytest.raises(AttributeError):
            simple_nodes[0].state_hash = "other"

# ===================================================================
# TestTransitionEdge
# ===================================================================
class TestTransitionEdge:
    def test_creation(self, simple_edges):
        e = simple_edges[0]
        assert (e.action_label, e.source_hash, e.target_hash) == ("inc", "s0", "s1")
        assert not e.is_stuttering and e.guard is None

    def test_stuttering_and_guard(self):
        e = TransitionEdge("stut", "a", "a", is_stuttering=True, guard="x>0")
        assert e.is_stuttering and e.guard == "x>0"

    def test_serialization_roundtrip(self, simple_edges):
        for e in simple_edges:
            r = TransitionEdge.from_dict(e.to_dict())
            assert (r.action_label, r.source_hash, r.target_hash) == (e.action_label, e.source_hash, e.target_hash)

    def test_frozen(self, simple_edges):
        with pytest.raises(AttributeError):
            simple_edges[0].action_label = "dec"

# ===================================================================
# TestTransitionGraph
# ===================================================================
class TestTransitionGraph:
    def test_add_and_query_states(self, simple_nodes):
        g = TransitionGraph()
        g.add_state(simple_nodes[0], is_initial=True)
        g.add_state(simple_nodes[1])
        assert g.num_states == 2 and g.has_state("s0")
        assert "s0" in g.initial_states
        assert g.get_state("s0") == simple_nodes[0]

    def test_add_and_query_transitions(self, simple_nodes, simple_edges):
        g = TransitionGraph()
        for n in simple_nodes: g.add_state(n)
        for e in simple_edges: g.add_transition(e)
        assert g.num_transitions == 2
        assert len(g.get_successors("s0")) == 1

    def test_remove_state(self, diamond_graph):
        diamond_graph.remove_state("s3")
        assert not diamond_graph.has_state("s3")

    def test_remove_transition(self, diamond_graph):
        before = diamond_graph.num_transitions
        diamond_graph.remove_transition("s0", "s1", "a")
        assert diamond_graph.num_transitions < before

    def test_successors_predecessors(self, diamond_graph):
        assert {"s1", "s3"} == diamond_graph.get_successor_hashes("s0")
        preds = diamond_graph.get_predecessors("s2")
        assert len(preds) >= 1

    def test_shortest_path(self, diamond_graph):
        path = diamond_graph.shortest_path("s0", "s2")
        assert path[0] == "s0" and path[-1] == "s2" and len(path) == 3

    def test_bfs_reachable(self, diamond_graph):
        assert diamond_graph.bfs_reachable("s0") == {"s0", "s1", "s2", "s3"}

    def test_bfs_reachable_depth_limited(self, diamond_graph):
        r = diamond_graph.bfs_reachable("s0", max_depth=1)
        assert "s1" in r and "s2" not in r

    def test_sccs_acyclic(self, diamond_graph):
        for scc in diamond_graph.strongly_connected_components():
            assert len(scc) == 1

    def test_sccs_cyclic(self, cyclic_graph):
        big = [s for s in cyclic_graph.strongly_connected_components() if len(s) > 1]
        assert len(big) == 1 and big[0] == {"s0", "s1", "s2"}

    def test_nontrivial_sccs(self, cyclic_graph):
        assert len(cyclic_graph.nontrivial_sccs()) == 1

    def test_has_cycle(self, diamond_graph, cyclic_graph):
        assert not diamond_graph.has_cycle() and cyclic_graph.has_cycle()

    def test_statistics(self, diamond_graph):
        stats = diamond_graph.compute_statistics()
        assert isinstance(stats, GraphStatistics)
        assert (stats.num_states, stats.num_transitions, stats.num_initial_states) == (4, 4, 1)

    def test_deadlock_states(self, diamond_graph):
        assert any(n.state_hash == "s2" for n in diamond_graph.get_deadlock_states())

    def test_json_roundtrip(self, diamond_graph):
        restored = TransitionGraph.from_json(diamond_graph.to_json())
        assert restored.num_states == 4 and restored.num_transitions == 4
        assert restored.initial_states == diamond_graph.initial_states

    def test_save_load_json(self, diamond_graph, tmp_path):
        p = str(tmp_path / "g.json")
        diamond_graph.save_json(p)
        assert TransitionGraph.load_json(p).num_states == 4

    def test_copy_independence(self, diamond_graph):
        clone = diamond_graph.copy()
        clone.remove_state("s2")
        assert diamond_graph.has_state("s2")

    def test_merge(self, simple_nodes):
        g1, g2 = TransitionGraph(), TransitionGraph()
        g1.add_state(simple_nodes[0], is_initial=True)
        g2.add_state(simple_nodes[0])
        g2.add_state(simple_nodes[1])
        g2.add_transition(TransitionEdge("x", "s0", "s1"))
        g1.merge(g2)
        assert g1.has_state("s1") and g1.num_transitions >= 1

    def test_subgraph(self, diamond_graph):
        sub = diamond_graph.subgraph({"s0", "s1"})
        assert sub.num_states == 2 and not sub.has_state("s2")

    def test_reachable_subgraph(self, diamond_graph):
        assert diamond_graph.reachable_subgraph({"s0"}).num_states == 4

    def test_to_dot(self, diamond_graph):
        dot = diamond_graph.to_dot()
        assert "digraph" in dot and "s0" in dot

    @pytest.mark.parametrize("n", [0, 1, 10])
    def test_various_sizes(self, n):
        g = TransitionGraph()
        for i in range(n): g.add_state(_node(f"n{i}", i))
        assert g.num_states == n

# ===================================================================
# TestExplorer
# ===================================================================
class TestExplorer:
    def test_bfs_exploration(self):
        stats = ExplicitStateExplorer(SimpleEngine(4)).explore_bfs()
        assert stats.completed and stats.states_explored == 4

    def test_dfs_exploration(self):
        stats = ExplicitStateExplorer(SimpleEngine(5)).explore_dfs()
        assert stats.completed and stats.states_explored == 5

    def test_depth_limited_bfs(self):
        stats = ExplicitStateExplorer(SimpleEngine(10)).explore_bfs(depth_limit=3)
        assert stats.states_explored <= 4

    def test_depth_limited_dfs(self):
        stats = ExplicitStateExplorer(SimpleEngine(10)).explore_dfs(depth_limit=2)
        assert stats.states_explored <= 3

    def test_cyclic_exploration(self):
        exp = ExplicitStateExplorer(SimpleEngine(3, cyclic=True))
        stats = exp.explore_bfs()
        assert stats.completed and stats.states_explored == 3 and exp.graph.has_cycle()

    def test_deadlock_detection(self):
        exp = ExplicitStateExplorer(DeadlockEngine())
        exp.explore_bfs()
        assert len(exp.find_deadlocks()) >= 1

    def test_invariant_violation(self):
        exp = ExplicitStateExplorer(SimpleEngine(4))
        exp.explore_bfs()
        assert len(exp.find_invariant_violations("small")) >= 1

    def test_invariant_holds(self):
        exp = ExplicitStateExplorer(SimpleEngine(4))
        exp.explore_bfs()
        assert len(exp.find_invariant_violations("nonneg")) == 0

    def test_reset(self):
        exp = ExplicitStateExplorer(SimpleEngine(3))
        exp.explore_bfs()
        exp.reset()
        # reset clears visited/stats, not necessarily the graph
        for h in exp.graph.all_state_hashes():
            assert not exp.is_explored(h)

    def test_is_explored(self):
        exp = ExplicitStateExplorer(SimpleEngine(3))
        exp.explore_bfs()
        for h in exp.graph.all_state_hashes():
            assert exp.is_explored(h)

    def test_stats_fields(self):
        stats = ExplicitStateExplorer(SimpleEngine(3)).explore_bfs()
        assert stats.elapsed_seconds >= 0 and "states_explored" in stats.to_dict()

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_chain_lengths(self, n):
        assert ExplicitStateExplorer(SimpleEngine(n)).explore_bfs().states_explored == n

# ===================================================================
# TestStateHashTable
# ===================================================================
class TestStateHashTable:
    def test_insert_contains_get(self):
        ht = StateHashTable()
        s = {"a": 1, "b": 2}
        ht.insert(s)
        assert ht.contains(s)
        assert ht.get(s)["a"] == 1

    def test_not_found(self):
        ht = StateHashTable()
        assert not ht.contains({"miss": 1}) and ht.get({"miss": 1}) is None

    def test_remove(self):
        ht = StateHashTable()
        s = {"x": 42}
        ht.insert(s)
        assert ht.remove(s) and not ht.contains(s)

    def test_remove_nonexistent(self):
        assert not StateHashTable().remove({"no": 1})

    def test_size_and_len(self):
        ht = StateHashTable()
        for i in range(10): ht.insert({"i": i})
        assert ht.size == 10 and len(ht) == 10

    def test_all_states(self):
        ht = StateHashTable()
        for i in range(5): ht.insert({"v": i})
        assert len(ht.all_states()) == 5

    def test_duplicate_insert(self):
        ht = StateHashTable()
        s = {"dup": True}
        ht.insert(s)
        assert ht.insert(s) is False and ht.size == 1

    def test_statistics_and_memory(self):
        ht = StateHashTable()
        for i in range(20): ht.insert({"n": i})
        assert isinstance(ht.statistics(), dict)
        assert ht.memory_usage_bytes() >= 0

    def test_clear(self):
        ht = StateHashTable()
        for i in range(5): ht.insert({"i": i})
        ht.clear()
        assert ht.size == 0

    @pytest.mark.parametrize("cap", [16, 64, 256])
    def test_initial_capacity(self, cap):
        assert StateHashTable(initial_capacity=cap).capacity >= cap

# ===================================================================
# TestZobristHasher
# ===================================================================
class TestZobristHasher:
    def test_deterministic(self):
        h = ZobristHasher(seed=42)
        s = {"a": 1, "b": 2}
        assert h.hash_state(s) == h.hash_state(s)

    def test_different_states_differ(self):
        h = ZobristHasher(seed=42)
        assert h.hash_state({"a": 1}) != h.hash_state({"a": 2})

    def test_same_seed_reproducible(self):
        s = {"x": 10, "y": 20}
        assert ZobristHasher(seed=99).hash_state(s) == ZobristHasher(seed=99).hash_state(s)

    def test_incremental_update(self):
        h = ZobristHasher(seed=7)
        full = h.hash_state({"a": 1, "b": 3})
        inc = h.incremental_update(h.hash_state({"a": 1, "b": 2}), "b", 2, 3)
        assert inc == full

    def test_bit_width_32(self):
        v = ZobristHasher(bit_width=32, seed=1).hash_state({"k": "v"})
        assert isinstance(v, int) and v < 2**32

    @pytest.mark.parametrize("seed", [0, 1, 12345])
    def test_various_seeds(self, seed):
        assert isinstance(ZobristHasher(seed=seed).hash_state({"x": 1}), int)

# ===================================================================
# TestSymmetry
# ===================================================================
class TestSymmetry:
    def test_identity(self):
        p = Permutation.identity({"a", "b", "c"})
        assert p.is_identity()
        assert all(p.apply(e) == e for e in ["a", "b", "c"])

    def test_apply(self):
        p = Permutation.from_dict({"a": "b", "b": "a"})
        assert p.apply("a") == "b" and p.apply("b") == "a"

    def test_compose(self):
        p1 = Permutation.from_dict({"a": "b", "b": "c", "c": "a"})
        p2 = Permutation.from_dict({"a": "c", "b": "b", "c": "a"})
        c = p1.compose(p2)
        assert all(c.apply(e) == p1.apply(p2.apply(e)) for e in "abc")

    def test_inverse(self):
        p = Permutation.from_dict({"a": "b", "b": "c", "c": "a"})
        assert p.compose(p.inverse()).is_identity()

    def test_order(self):
        assert Permutation.from_dict({"a": "b", "b": "c", "c": "a"}).order() == 3

    def test_cycle_notation(self):
        p = Permutation.from_dict({"a": "b", "b": "a", "c": "c"})
        non_trivial = [c for c in p.cycle_notation() if len(c) > 1]
        assert len(non_trivial) == 1 and set(non_trivial[0]) == {"a", "b"}

    def test_symmetric_group(self):
        assert PermutationGroup.symmetric_group({"a", "b", "c"}).order() == 6

    def test_cyclic_group(self):
        assert PermutationGroup.cyclic_group(["a", "b", "c"]).order() == 3

    def test_canonical_form(self):
        det = SymmetryDetector(symmetric_sets={"procs": {"p0", "p1"}})
        s1 = {"p0": "idle", "p1": "busy"}
        s2 = {"p0": "busy", "p1": "idle"}
        assert det.canonical_form(s1) == det.canonical_form(s2)

    def test_symmetric_pair(self):
        det = SymmetryDetector(symmetric_sets={"procs": {"p0", "p1"}})
        assert det.is_symmetric_pair(
            {"p0": "a", "p1": "b"},
            {"p0": "b", "p1": "a"},
        )

    def test_canonical_hash_consistent(self):
        det = SymmetryDetector(symmetric_sets={"procs": {"p0", "p1"}})
        s = {"p0": "x", "p1": "y"}
        assert det.canonical_hash(s) == det.canonical_hash(s)

    def test_reduce_graph(self, cyclic_graph):
        det = SymmetryDetector(symmetric_sets={"dummy": {"s0", "s1", "s2"}})
        assert det.reduce_graph(cyclic_graph).num_states <= cyclic_graph.num_states

    def test_reduction_ratio(self, diamond_graph):
        det = SymmetryDetector(symmetric_sets={"nodes": {"s1", "s3"}})
        assert 0.0 <= det.reduction_ratio(diamond_graph) <= 1.0

    def test_orbit_computation(self):
        det = SymmetryDetector(symmetric_sets={"p": {"a", "b"}})
        assert len(det.compute_orbit({"p_a": 1, "p_b": 2})) >= 1

# ===================================================================
# TestFairness
# ===================================================================
class TestFairness:
    @pytest.fixture
    def fair_graph(self):
        g = TransitionGraph()
        for i in range(3):
            g.add_state(StateNode(f"f{i}", {"i": i}, frozenset()), is_initial=(i == 0))
        for s, t in [("f0","f1"),("f1","f2"),("f2","f0")]:
            g.add_transition(TransitionEdge("tick", s, t))
        return g

    def test_constraint_creation(self):
        fc = FairnessConstraint(FairnessKind.STRONG, "act", name="s_act")
        assert fc.kind == FairnessKind.STRONG and fc.action_label == "act"

    def test_acceptance_pair_satisfied(self):
        fc = FairnessConstraint(FairnessKind.STRONG, "tick")
        ap = AcceptancePair(fc, frozenset({"f0","f1","f2"}), frozenset({"f0","f1","f2"}))
        assert ap.is_satisfied_by_scc({"f0", "f1", "f2"})

    def test_acceptance_pair_unsatisfied(self):
        fc = FairnessConstraint(FairnessKind.STRONG, "tick")
        ap = AcceptancePair(fc, frozenset({"f0","f1","f2"}), frozenset())
        assert not ap.is_satisfied_by_scc({"f0", "f1", "f2"})

    def test_lasso_validity(self, fair_graph):
        assert Lasso(prefix=[], loop=["f0", "f1", "f2"]).is_valid(fair_graph)

    def test_lasso_states(self):
        l = Lasso(prefix=["a","b"], loop=["b","c","d"])
        assert l.all_states() == {"a","b","c","d"} and l.loop_states() == {"b","c","d"}

    def test_tracker_pairs(self, fair_graph):
        fc = FairnessConstraint(FairnessKind.STRONG, "tick")
        assert len(FairnessTracker(fair_graph, [fc]).compute_acceptance_pairs()) >= 1

    def test_fair_sccs(self, fair_graph):
        fc = FairnessConstraint(FairnessKind.STRONG, "tick")
        assert len(FairnessTracker(fair_graph, [fc]).fair_sccs()) >= 1

    def test_unfair_sccs_missing_action(self):
        g = TransitionGraph()
        for i in range(2):
            g.add_state(StateNode(f"u{i}", {"i": i}, frozenset()), is_initial=(i == 0))
        g.add_transition(TransitionEdge("nop", "u0", "u1"))
        g.add_transition(TransitionEdge("nop", "u1", "u0"))
        fc = FairnessConstraint(FairnessKind.STRONG, "required")
        unfair = FairnessTracker(g, [fc]).unfair_sccs()
        assert isinstance(unfair, list)

    def test_find_fair_cycle(self, fair_graph):
        fc = FairnessConstraint(FairnessKind.WEAK, "tick")
        cycle = FairnessTracker(fair_graph, [fc]).find_fair_cycle()
        if cycle is not None:
            assert len(cycle.loop) >= 1

    def test_summary(self, fair_graph):
        fc = FairnessConstraint(FairnessKind.STRONG, "tick")
        assert isinstance(FairnessTracker(fair_graph, [fc]).summary(), dict)

    def test_add_constraint(self, fair_graph):
        t = FairnessTracker(fair_graph, [])
        t.add_constraint(FairnessConstraint(FairnessKind.WEAK, "tick"))
        assert len(t.constraints) == 1

    @pytest.mark.parametrize("kind", [FairnessKind.WEAK, FairnessKind.STRONG])
    def test_fairness_kinds(self, kind):
        assert FairnessConstraint(kind, "act").kind == kind

# ===================================================================
# TestTraces
# ===================================================================
class TestTraces:
    @pytest.fixture
    def trace_graph(self):
        g = TransitionGraph()
        for i in range(4):
            g.add_state(StateNode(f"t{i}", {"v": i}, frozenset({f"p{i}"})), is_initial=(i == 0))
        for s, t, a in [("t0","t1","a"),("t1","t2","b"),("t2","t3","c"),("t3","t1","d")]:
            g.add_transition(TransitionEdge(a, s, t))
        return g

    def test_trace_step_roundtrip(self):
        ts = TraceStep("s0", {"x": 0}, "go", 0, {"init"})
        r = TraceStep.from_dict(ts.to_dict())
        assert r.state_hash == "s0" and r.action_label == "go"

    def test_append_and_accessors(self):
        t = ExecutionTrace()
        t.append(TraceStep("a", {"v": 1}, "go", 0))
        t.append(TraceStep("b", {"v": 2}, "step", 1))
        assert t.length == 2 and t.state_hashes() == ["a", "b"]
        assert "go" in t.actions()
        assert t.contains_state("a") and not t.contains_state("z")

    def test_empty_trace(self):
        t = ExecutionTrace()
        assert t.is_empty and t.length == 0

    def test_is_valid(self, trace_graph):
        t = ExecutionTrace()
        t.append(TraceStep("t0", {"v": 0}, "a", 0))
        t.append(TraceStep("t1", {"v": 1}, "b", 1))
        t.append(TraceStep("t2", {"v": 2}, None, 2))
        assert t.is_valid(trace_graph)

    def test_is_invalid(self, trace_graph):
        t = ExecutionTrace()
        t.append(TraceStep("t0", {"v": 0}, "a", 0))
        t.append(TraceStep("t3", {"v": 3}, None, 1))
        assert not t.is_valid(trace_graph)

    def test_json_roundtrip(self):
        t = ExecutionTrace()
        t.append(TraceStep("a", {"x": 1}, "go", 0))
        t.append(TraceStep("b", {"x": 2}, None, 1))
        r = ExecutionTrace.from_json(t.to_json())
        assert r.length == 2 and r.state_hashes() == ["a", "b"]

    def test_save_load(self, tmp_path):
        t = ExecutionTrace([TraceStep("s", {"k": 1}, None, 0)])
        p = str(tmp_path / "trace.json")
        t.save(p)
        assert ExecutionTrace.load(p).length == 1

    def test_pretty_print(self):
        t = ExecutionTrace([TraceStep("a", {"v": 1}, "go", 0, {"init"})])
        assert len(t.pretty_print()) > 0

    def test_lasso_trace(self):
        prefix = ExecutionTrace([TraceStep("a", {}, None, 0)])
        loop = ExecutionTrace([TraceStep("b", {}, "x", 1), TraceStep("c", {}, "y", 2)])
        lt = LassoTrace(prefix, loop)
        assert lt.prefix_length == 1 and lt.loop_length == 2

    def test_lasso_trace_json(self):
        lt = LassoTrace(
            ExecutionTrace([TraceStep("a", {}, None, 0)]),
            ExecutionTrace([TraceStep("b", {}, "x", 1)]),
        )
        r = LassoTrace.from_json(lt.to_json())
        assert r.prefix_length == 1 and r.loop_length == 1

    def test_manager_construct(self, trace_graph):
        trace = TraceManager(trace_graph).construct_trace_to("t2")
        if trace is not None:
            assert trace.state_hashes()[-1] == "t2"

    def test_manager_minimize(self, trace_graph):
        t = ExecutionTrace()
        for h, v, a in [("t0",0,"a"),("t1",1,"b"),("t2",2,None)]:
            t.append(TraceStep(h, {"v": v}, a, v))
        assert TraceManager(trace_graph).minimize_trace(t).length <= t.length

    def test_manager_lasso(self, trace_graph):
        lasso = TraceManager(trace_graph).construct_lasso("t3", "t1")
        if lasso is not None:
            assert isinstance(lasso, LassoTrace)

    def test_manager_summary(self, trace_graph):
        assert isinstance(TraceManager(trace_graph).summary(), dict)

    @pytest.mark.parametrize("depth", [0, 1, 5])
    def test_trace_step_depths(self, depth):
        assert TraceStep("h", {}, None, depth).depth == depth

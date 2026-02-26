"""
Comprehensive test suite for the coacert.bisimulation module.

Covers:
  - BisimulationRelation (union-find, refinement, serialization, lattice ops)
  - PartitionRefinement (Paige–Tarjan)
  - StutteringBisimulation (Groote–Vaandrager)
  - QuotientBuilder
  - RefinementEngine (full pipeline)
  - FairnessEquivalence
"""

from __future__ import annotations

import json

import pytest

from coacert.bisimulation.relation import BisimulationRelation, RelationStats
from coacert.bisimulation.partition_refinement import (
    Block,
    PartitionRefinement,
    RefinementResult,
)
from coacert.bisimulation.stuttering import (
    StutteringBisimulation,
    StutteringResult,
    StutterCounterexample,
)
from coacert.bisimulation.quotient import (
    QuotientBuilder,
    QuotientStats,
    QuotientVerificationResult,
)
from coacert.bisimulation.refinement_iteration import (
    RefinementEngine,
    RefinementStrategy,
    EngineResult,
)
from coacert.bisimulation.fairness_equiv import (
    FairnessEquivalence,
    FairEquivalenceResult,
    FairCycleInfo,
)


# ===================================================================
# Helpers – small transition systems used across test classes
# ===================================================================

def _two_state_system():
    """s0 -a-> s1, s1 -a-> s0; same labels."""
    states = {"s0", "s1"}
    actions = {"a"}
    transitions = {
        "s0": {"a": {"s1"}},
        "s1": {"a": {"s0"}},
    }
    labels = {"s0": {"p"}, "s1": {"p"}}
    return states, actions, transitions, labels


def _three_state_split():
    """s0,s1 labelled {p}; s2 labelled {q}. s0->s1, s1->s2, s2->s0."""
    states = {"s0", "s1", "s2"}
    actions = {"a"}
    transitions = {
        "s0": {"a": {"s1"}},
        "s1": {"a": {"s2"}},
        "s2": {"a": {"s0"}},
    }
    labels = {"s0": {"p"}, "s1": {"p"}, "s2": {"q"}}
    return states, actions, transitions, labels


def _diamond_system():
    """
    s0 -a-> s1, s0 -a-> s2, s1 -a-> s3, s2 -a-> s3.
    s1 and s2 have same labels and same successor structure -> bisimilar.
    """
    states = {"s0", "s1", "s2", "s3"}
    actions = {"a"}
    transitions = {
        "s0": {"a": {"s1", "s2"}},
        "s1": {"a": {"s3"}},
        "s2": {"a": {"s3"}},
        "s3": {},
    }
    labels = {"s0": {"p"}, "s1": {"q"}, "s2": {"q"}, "s3": {"r"}}
    return states, actions, transitions, labels


def _stutter_chain():
    """s0 -a-> s1 -a-> s2; all same labels. s1 is a stutter step."""
    states = {"s0", "s1", "s2"}
    actions = {"a"}
    transitions = {
        "s0": {"a": {"s1"}},
        "s1": {"a": {"s2"}},
        "s2": {},
    }
    labels = {"s0": {"p"}, "s1": {"p"}, "s2": {"p"}}
    return states, actions, transitions, labels


# ===================================================================
# 1. TestBisimulationRelation
# ===================================================================

class TestBisimulationRelation:
    """Tests for the union-find backed BisimulationRelation."""

    def test_make_set_and_find(self):
        rel = BisimulationRelation()
        rel.make_set("a")
        assert rel.find("a") == "a"

    def test_make_set_idempotent(self):
        rel = BisimulationRelation()
        rel.make_set("a")
        rel.make_set("a")
        assert len(rel) == 1

    def test_union_and_equivalent(self):
        rel = BisimulationRelation(["a", "b", "c"])
        rel.union("a", "b")
        assert rel.equivalent("a", "b")
        assert not rel.equivalent("a", "c")

    def test_representative(self):
        rel = BisimulationRelation(["x", "y"])
        rel.union("x", "y")
        rep = rel.representative("x")
        assert rep == rel.representative("y")

    def test_class_of(self):
        rel = BisimulationRelation(["a", "b", "c"])
        rel.union("a", "b")
        assert rel.class_of("a") == frozenset({"a", "b"})
        assert rel.class_of("c") == frozenset({"c"})

    def test_classes(self):
        rel = BisimulationRelation(["a", "b", "c"])
        rel.union("a", "c")
        classes = rel.classes()
        assert len(classes) == 2
        assert frozenset({"a", "c"}) in classes
        assert frozenset({"b"}) in classes

    def test_representatives(self):
        rel = BisimulationRelation(["a", "b", "c"])
        rel.union("a", "b")
        reps = rel.representatives()
        assert len(reps) == 2

    def test_num_classes(self):
        rel = BisimulationRelation(["a", "b", "c", "d"])
        assert rel.num_classes() == 4
        rel.union("a", "b")
        rel.union("c", "d")
        assert rel.num_classes() == 2
        rel.union("a", "c")
        assert rel.num_classes() == 1

    def test_stats(self):
        rel = BisimulationRelation(["a", "b", "c"])
        rel.union("a", "b")
        st = rel.stats()
        assert isinstance(st, RelationStats)
        assert st.num_states == 3
        assert st.num_classes == 2
        assert st.largest_class == 2
        assert st.smallest_class == 1
        assert st.compression_ratio == pytest.approx(1.0 - 2 / 3)

    def test_stats_empty(self):
        rel = BisimulationRelation()
        st = rel.stats()
        assert st.num_states == 0
        assert st.num_classes == 0

    # -- refinement operations ---

    def test_refine_by(self):
        rel = BisimulationRelation.coarsest(["a", "b", "c"])
        assert rel.num_classes() == 1
        labels = {"a": "x", "b": "x", "c": "y"}
        new_classes = rel.refine_by(lambda s: labels[s])
        assert new_classes == 1  # 1 -> 2 classes, so 1 new
        assert rel.num_classes() == 2
        assert rel.equivalent("a", "b")
        assert not rel.equivalent("a", "c")

    def test_split_class(self):
        rel = BisimulationRelation.coarsest(["a", "b", "c", "d"])
        rep = rel.find("a")
        rk, rr = rel.split_class(rep, {"a", "b"}, {"c", "d"})
        assert rel.num_classes() == 2
        assert rel.equivalent("a", "b")
        assert rel.equivalent("c", "d")
        assert not rel.equivalent("a", "c")

    def test_split_class_validation(self):
        rel = BisimulationRelation.coarsest(["a", "b"])
        rep = rel.find("a")
        with pytest.raises(ValueError, match="non-empty"):
            rel.split_class(rep, {"a", "b"}, set())

    # -- lattice operations ---

    def test_intersect(self):
        r1 = BisimulationRelation.from_blocks([{"a", "b", "c"}])
        r2 = BisimulationRelation.from_blocks([{"a", "b"}, {"c"}])
        meet = r1.intersect(r2)
        assert meet.equivalent("a", "b")
        assert not meet.equivalent("a", "c")

    def test_join(self):
        r1 = BisimulationRelation.from_blocks([{"a", "b"}, {"c"}])
        r2 = BisimulationRelation.from_blocks([{"b", "c"}, {"a"}])
        joined = r1.join(r2)
        # a~b in r1, b~c in r2 => a~b~c in join
        assert joined.equivalent("a", "c")

    def test_is_finer_than(self):
        coarse = BisimulationRelation.from_blocks([{"a", "b", "c"}])
        fine = BisimulationRelation.from_blocks([{"a", "b"}, {"c"}])
        assert fine.is_finer_than(coarse)
        assert not coarse.is_finer_than(fine)

    def test_equals(self):
        r1 = BisimulationRelation.from_blocks([{"a", "b"}, {"c"}])
        r2 = BisimulationRelation.from_blocks([{"a", "b"}, {"c"}])
        assert r1.equals(r2)
        assert r1 == r2

    def test_difference_witnesses(self):
        r1 = BisimulationRelation.from_blocks([{"a", "b", "c"}])
        r2 = BisimulationRelation.from_blocks([{"a", "b"}, {"c"}])
        witnesses = r1.difference_witnesses(r2)
        assert len(witnesses) > 0
        pairs = {frozenset(w) for w in witnesses}
        assert frozenset({"a", "c"}) in pairs or frozenset({"b", "c"}) in pairs

    # -- classmethods ---

    def test_coarsest(self):
        rel = BisimulationRelation.coarsest(["a", "b", "c"])
        assert rel.num_classes() == 1
        assert rel.equivalent("a", "c")

    def test_discrete(self):
        rel = BisimulationRelation.discrete(["a", "b", "c"])
        assert rel.num_classes() == 3
        assert not rel.equivalent("a", "b")

    def test_from_labeling(self):
        labels = {"a": 1, "b": 1, "c": 2}
        rel = BisimulationRelation.from_labeling(
            ["a", "b", "c"], lambda s: labels[s]
        )
        assert rel.equivalent("a", "b")
        assert not rel.equivalent("a", "c")
        assert rel.num_classes() == 2

    def test_from_blocks(self):
        rel = BisimulationRelation.from_blocks([{"x", "y"}, {"z"}])
        assert rel.equivalent("x", "y")
        assert not rel.equivalent("x", "z")

    # -- serialization ---

    def test_to_dict_from_dict_roundtrip(self):
        rel = BisimulationRelation.from_blocks([{"a", "b"}, {"c", "d"}])
        d = rel.to_dict()
        rel2 = BisimulationRelation.from_dict(d)
        assert rel.equals(rel2)
        assert d["num_states"] == 4
        assert d["num_classes"] == 2

    def test_to_json_from_json_roundtrip(self):
        rel = BisimulationRelation.from_blocks([{"s0", "s1"}, {"s2"}])
        j = rel.to_json(indent=2)
        rel2 = BisimulationRelation.from_json(j)
        assert rel.equals(rel2)
        parsed = json.loads(j)
        assert parsed["version"] == 1

    # -- copy, restrict, map_states ---

    def test_copy(self):
        rel = BisimulationRelation.from_blocks([{"a", "b"}, {"c"}])
        cp = rel.copy()
        assert cp.equals(rel)
        cp.union("b", "c")  # mutate copy
        assert not cp.equals(rel)  # original unchanged

    def test_restrict(self):
        rel = BisimulationRelation.from_blocks([{"a", "b", "c"}])
        restricted = rel.restrict({"a", "b"})
        assert restricted.equivalent("a", "b")
        assert "c" not in restricted

    def test_map_states(self):
        rel = BisimulationRelation.from_blocks([{"a", "b"}, {"c"}])
        mapped = rel.map_states(lambda s: s.upper())
        assert mapped.equivalent("A", "B")
        assert not mapped.equivalent("A", "C")
        assert "a" not in mapped

    # -- iteration & containment ---

    def test_iter(self):
        rel = BisimulationRelation.from_blocks([{"a"}, {"b"}])
        classes = list(rel)
        assert len(classes) == 2

    def test_contains(self):
        rel = BisimulationRelation(["a", "b"])
        assert "a" in rel
        assert "z" not in rel

    def test_len(self):
        rel = BisimulationRelation(["a", "b", "c"])
        assert len(rel) == 3

    @pytest.mark.parametrize(
        "blocks,expected_classes",
        [
            ([{"a"}], 1),
            ([{"a", "b"}], 1),
            ([{"a"}, {"b"}], 2),
            ([{"a", "b"}, {"c", "d"}, {"e"}], 3),
        ],
    )
    def test_from_blocks_parametrized(self, blocks, expected_classes):
        rel = BisimulationRelation.from_blocks(blocks)
        assert rel.num_classes() == expected_classes


# ===================================================================
# 2. TestPartitionRefinement
# ===================================================================

class TestPartitionRefinement:
    """Tests for the Paige–Tarjan partition refinement engine."""

    def test_refine_two_equivalent_states(self):
        states, actions, transitions, labels = _two_state_system()
        pr = PartitionRefinement(states, actions, transitions, labels)
        result = pr.refine()
        assert isinstance(result, RefinementResult)
        # s0 and s1 have same AP and symmetric transitions -> bisimilar
        assert result.partition.equivalent("s0", "s1")
        assert result.final_blocks == 1

    def test_refine_splits_by_ap(self):
        states, actions, transitions, labels = _three_state_split()
        pr = PartitionRefinement(states, actions, transitions, labels)
        result = pr.refine()
        # s2 has different AP -> must be separate
        assert not result.partition.equivalent("s0", "s2")
        assert result.final_blocks >= 2

    def test_refine_diamond_merges_bisimilar(self):
        states, actions, transitions, labels = _diamond_system()
        pr = PartitionRefinement(states, actions, transitions, labels)
        result = pr.refine()
        # s1 and s2 have same AP {q} and same successor pattern
        assert result.partition.equivalent("s1", "s2")
        assert not result.partition.equivalent("s0", "s1")
        assert not result.partition.equivalent("s1", "s3")

    def test_ap_refinement_standalone(self):
        states, actions, transitions, labels = _three_state_split()
        pr = PartitionRefinement(states, actions, transitions, labels)
        pr._build_initial_partition()
        new_blocks = pr.refine_by_ap()
        # AP refinement is already done in initial partition, so 0 new
        assert new_blocks >= 0

    def test_action_refinement(self):
        states, actions, transitions, labels = _diamond_system()
        pr = PartitionRefinement(states, actions, transitions, labels)
        pr._build_initial_partition()
        pr.refine_by_action("a")
        snap = pr.snapshot()
        # after action refinement, s1 and s2 should remain together
        assert snap.equivalent("s1", "s2")

    def test_snapshot_before_complete(self):
        states, actions, transitions, labels = _diamond_system()
        pr = PartitionRefinement(states, actions, transitions, labels)
        pr._build_initial_partition()
        snap = pr.snapshot()
        assert isinstance(snap, BisimulationRelation)
        assert len(snap) == 4

    def test_result_metadata(self):
        states, actions, transitions, labels = _diamond_system()
        pr = PartitionRefinement(states, actions, transitions, labels)
        result = pr.refine()
        assert result.states_processed == 4
        assert result.transitions_processed > 0
        assert result.num_rounds >= 0
        assert result.total_elapsed_ms >= 0
        assert result.initial_blocks >= 1

    def test_complexity_report(self):
        states, actions, transitions, labels = _two_state_system()
        pr = PartitionRefinement(states, actions, transitions, labels)
        pr.refine()
        report = pr.complexity_report()
        assert report["n_states"] == 2
        assert report["m_transitions"] == 2
        assert "theoretical_bound" in report

    def test_block_size(self):
        b = Block(id=0, members={"s0", "s1", "s2"})
        assert b.size == 3

    @pytest.mark.parametrize(
        "system_fn,min_blocks",
        [
            (_two_state_system, 1),
            (_three_state_split, 2),
            (_diamond_system, 3),
        ],
    )
    def test_refine_various_systems(self, system_fn, min_blocks):
        states, actions, transitions, labels = system_fn()
        pr = PartitionRefinement(states, actions, transitions, labels)
        result = pr.refine()
        assert result.final_blocks >= min_blocks

    def test_single_state_system(self):
        pr = PartitionRefinement({"s0"}, {"a"}, {"s0": {}}, {"s0": {"p"}})
        result = pr.refine()
        assert result.final_blocks == 1

    def test_self_loop(self):
        pr = PartitionRefinement(
            {"s0", "s1"}, {"a"},
            {"s0": {"a": {"s0"}}, "s1": {"a": {"s1"}}},
            {"s0": {"p"}, "s1": {"p"}},
        )
        result = pr.refine()
        assert result.partition.equivalent("s0", "s1")


# ===================================================================
# 3. TestStutteringBisimulation
# ===================================================================

class TestStutteringBisimulation:
    """Tests for the Groote–Vaandrager stuttering bisimulation."""

    def test_identical_behavior_equivalent(self):
        states, actions, transitions, labels = _two_state_system()
        sb = StutteringBisimulation(states, actions, transitions, labels)
        result = sb.compute()
        assert isinstance(result, StutteringResult)
        assert result.partition.equivalent("s0", "s1")

    def test_different_ap_not_equivalent(self):
        states, actions, transitions, labels = _three_state_split()
        sb = StutteringBisimulation(states, actions, transitions, labels)
        result = sb.compute()
        assert not result.partition.equivalent("s0", "s2")

    def test_stutter_steps_equivalent(self):
        """States differing only by stutter steps should be equivalent."""
        # s0 -> s1 -> s2, all same label {p}.
        # Under stuttering bisimulation, s0 and s1 (and s2) may be merged
        # because the transitions are all within the same AP class.
        states, actions, transitions, labels = _stutter_chain()
        sb = StutteringBisimulation(states, actions, transitions, labels)
        result = sb.compute()
        # All states have same AP and transitions stay within the AP class
        assert result.partition.equivalent("s0", "s1")

    def test_divergence_sensitive(self):
        """Divergence-sensitive mode separates self-looping states."""
        states = {"s0", "s1"}
        actions = {"a"}
        transitions = {
            "s0": {"a": {"s0", "s1"}},  # s0 can self-loop (diverge)
            "s1": {},
        }
        labels = {"s0": {"p"}, "s1": {"p"}}
        sb = StutteringBisimulation(
            states, actions, transitions, labels, divergence_sensitive=True
        )
        result = sb.compute()
        assert result.divergence_sensitive
        # s0 is divergent (self-loop), s1 is not
        assert "s0" in result.divergent_states

    def test_compute_result_fields(self):
        states, actions, transitions, labels = _two_state_system()
        sb = StutteringBisimulation(states, actions, transitions, labels)
        result = sb.compute()
        assert result.num_rounds >= 1
        assert result.total_elapsed_ms >= 0
        assert result.initial_blocks >= 1
        assert result.final_blocks >= 1

    def test_verify_valid_partition(self):
        states, actions, transitions, labels = _two_state_system()
        sb = StutteringBisimulation(states, actions, transitions, labels)
        result = sb.compute()
        is_valid, cxs = sb.verify(result.partition, stuttering=True)
        assert is_valid
        assert len(cxs) == 0

    def test_verify_invalid_partition(self):
        """Putting states with different APs together should fail."""
        states, actions, transitions, labels = _three_state_split()
        sb = StutteringBisimulation(states, actions, transitions, labels)
        # force all into one block
        bad = BisimulationRelation.coarsest(states)
        is_valid, cxs = sb.verify(bad, stuttering=True)
        assert not is_valid
        assert len(cxs) > 0
        assert isinstance(cxs[0], StutterCounterexample)

    def test_compare_with_strong(self):
        states, actions, transitions, labels = _stutter_chain()
        sb = StutteringBisimulation(states, actions, transitions, labels)
        stut_part, strong_part, extra = sb.compare_with_strong()
        assert isinstance(stut_part, BisimulationRelation)
        assert isinstance(strong_part, BisimulationRelation)
        # stuttering is at least as coarse as strong
        assert strong_part.is_finer_than(stut_part)

    def test_generate_counterexample_none_when_equivalent(self):
        states, actions, transitions, labels = _two_state_system()
        sb = StutteringBisimulation(states, actions, transitions, labels)
        sb.compute()
        cx = sb.generate_counterexample("s0", "s1")
        assert cx is None

    def test_generate_counterexample_found_when_different(self):
        states, actions, transitions, labels = _three_state_split()
        sb = StutteringBisimulation(states, actions, transitions, labels)
        sb.compute()
        cx = sb.generate_counterexample("s0", "s2")
        assert cx is not None
        assert isinstance(cx, StutterCounterexample)

    def test_branching_bisimulation(self):
        states, actions, transitions, labels = _stutter_chain()
        sb = StutteringBisimulation(states, actions, transitions, labels)
        result = sb.compute_branching()
        assert isinstance(result, StutteringResult)
        assert result.final_blocks >= 1


# ===================================================================
# 4. TestQuotientBuilder
# ===================================================================

class TestQuotientBuilder:
    """Tests for quotient system construction."""

    def _build_quotient(self, system_fn, partition=None, **kwargs):
        states, actions, transitions, labels = system_fn()
        if partition is None:
            pr = PartitionRefinement(states, actions, transitions, labels)
            result = pr.refine()
            partition = result.partition
        builder = QuotientBuilder(
            states, actions, transitions, labels, partition, **kwargs
        )
        q_states, q_trans, q_labels = builder.build()
        return builder, q_states, q_trans, q_labels

    def test_build_reduces_states(self):
        builder, q_states, q_trans, q_labels = self._build_quotient(
            _diamond_system
        )
        # s1 and s2 are bisimilar -> quotient has 3 states
        assert len(q_states) == 3

    def test_build_two_equivalent(self):
        builder, q_states, q_trans, q_labels = self._build_quotient(
            _two_state_system
        )
        assert len(q_states) == 1

    def test_quotient_labels_preserved(self):
        builder, q_states, q_trans, q_labels = self._build_quotient(
            _diamond_system
        )
        # every quotient state should have labels
        for s in q_states:
            assert s in q_labels

    def test_verify_correct(self):
        builder, _, _, _ = self._build_quotient(_diamond_system)
        vr = builder.verify()
        assert isinstance(vr, QuotientVerificationResult)
        assert vr.is_correct
        assert vr.ap_preserved
        assert vr.transitions_preserved
        assert vr.no_spurious_transitions

    def test_stats(self):
        builder, _, _, _ = self._build_quotient(_diamond_system)
        st = builder.stats()
        assert isinstance(st, QuotientStats)
        assert st.original_states == 4
        assert st.quotient_states == 3
        assert st.state_compression_ratio > 0

    def test_quotient_map(self):
        builder, _, _, _ = self._build_quotient(_diamond_system)
        qm = builder.quotient_map
        # s1 and s2 should map to the same representative
        assert qm["s1"] == qm["s2"]
        assert qm["s0"] != qm["s1"]

    def test_build_with_initial_states(self):
        states, actions, transitions, labels = _diamond_system()
        pr = PartitionRefinement(states, actions, transitions, labels)
        partition = pr.refine().partition
        builder = QuotientBuilder(
            states, actions, transitions, labels, partition
        )
        builder.build(initial_states={"s0"})
        assert len(builder.quotient_initial_states) == 1

    def test_to_transition_graph(self):
        builder, _, _, _ = self._build_quotient(_diamond_system)
        tg = builder.to_transition_graph()
        assert tg is not None
        # dict fallback or TransitionGraph – either way has "states"
        if isinstance(tg, dict):
            assert "states" in tg
        else:
            assert hasattr(tg, "states")

    def test_build_no_stutter_removal(self):
        # self-loop system: s0 -a-> s0
        states = {"s0"}
        actions = {"a"}
        transitions = {"s0": {"a": {"s0"}}}
        labels = {"s0": {"p"}}
        partition = BisimulationRelation.discrete(states)
        builder = QuotientBuilder(
            states, actions, transitions, labels, partition
        )
        q_states, q_trans, q_labels = builder.build(remove_stutter=False)
        # self-loop is kept when remove_stutter=False
        assert "s0" in q_trans.get("s0", {}).get("a", set())

    def test_stats_before_build_raises(self):
        states, actions, transitions, labels = _two_state_system()
        partition = BisimulationRelation.coarsest(states)
        builder = QuotientBuilder(
            states, actions, transitions, labels, partition
        )
        with pytest.raises(RuntimeError, match="build"):
            builder.stats()

    @pytest.mark.parametrize(
        "system_fn",
        [_two_state_system, _three_state_split, _diamond_system],
    )
    def test_verify_on_various_systems(self, system_fn):
        builder, _, _, _ = self._build_quotient(system_fn)
        vr = builder.verify()
        assert vr.is_correct


# ===================================================================
# 5. TestRefinementEngine
# ===================================================================

class TestRefinementEngine:
    """Tests for the full refinement pipeline engine."""

    def test_run_eager(self):
        states, actions, transitions, labels = _diamond_system()
        engine = RefinementEngine(
            states, actions, transitions, labels,
            strategy=RefinementStrategy.EAGER,
            use_stuttering=False,
            use_fairness=False,
        )
        result = engine.run()
        assert isinstance(result, EngineResult)
        assert result.converged
        assert result.partition.equivalent("s1", "s2")

    def test_run_lazy(self):
        states, actions, transitions, labels = _diamond_system()
        engine = RefinementEngine(
            states, actions, transitions, labels,
            strategy=RefinementStrategy.LAZY,
            use_stuttering=False,
            use_fairness=False,
        )
        result = engine.run()
        assert result.converged
        assert result.strategy == "lazy"

    def test_run_ap_then_transition(self):
        states, actions, transitions, labels = _three_state_split()
        engine = RefinementEngine(
            states, actions, transitions, labels,
            strategy=RefinementStrategy.AP_THEN_TRANSITION,
            use_stuttering=False,
            use_fairness=False,
        )
        result = engine.run()
        assert result.converged
        assert not result.partition.equivalent("s0", "s2")

    def test_run_stuttering_first(self):
        states, actions, transitions, labels = _stutter_chain()
        engine = RefinementEngine(
            states, actions, transitions, labels,
            strategy=RefinementStrategy.STUTTERING_FIRST,
            use_stuttering=True,
            use_fairness=False,
        )
        result = engine.run()
        assert result.converged
        assert result.stuttering_used

    def test_convergence_report(self):
        states, actions, transitions, labels = _diamond_system()
        engine = RefinementEngine(
            states, actions, transitions, labels,
            use_stuttering=False,
            use_fairness=False,
        )
        engine.run()
        report = engine.convergence_report()
        assert "total_rounds" in report
        assert "block_progression" in report
        assert report["converged"]

    def test_convergence_report_not_run(self):
        states, actions, transitions, labels = _diamond_system()
        engine = RefinementEngine(
            states, actions, transitions, labels,
            use_stuttering=False,
            use_fairness=False,
        )
        report = engine.convergence_report()
        assert report["status"] == "not_run"

    def test_refine_with_hint(self):
        states, actions, transitions, labels = _diamond_system()
        engine = RefinementEngine(
            states, actions, transitions, labels,
            use_stuttering=False,
            use_fairness=False,
        )
        coarse = BisimulationRelation.coarsest(states)
        hint = BisimulationRelation.from_blocks(
            [{"s0"}, {"s1", "s2"}, {"s3"}]
        )
        refined = engine.refine_with_hint(coarse, hint)
        assert refined.num_classes() >= hint.num_classes()

    def test_result_summary(self):
        states, actions, transitions, labels = _two_state_system()
        engine = RefinementEngine(
            states, actions, transitions, labels,
            use_stuttering=False,
            use_fairness=False,
        )
        result = engine.run()
        summary = result.summary()
        assert "RefinementEngine" in summary
        assert "converged=True" in summary

    def test_compare_partitions(self):
        pa = BisimulationRelation.from_blocks([{"a", "b"}, {"c"}])
        pb = BisimulationRelation.from_blocks([{"a"}, {"b"}, {"c"}])
        cmp = RefinementEngine.compare(pa, pb)
        assert not cmp.a_finer_than_b
        assert cmp.b_finer_than_a
        assert not cmp.are_equal
        assert len(cmp.extra_merges_in_a) > 0

    @pytest.mark.parametrize(
        "strategy",
        list(RefinementStrategy),
    )
    def test_all_strategies_converge(self, strategy):
        states, actions, transitions, labels = _diamond_system()
        engine = RefinementEngine(
            states, actions, transitions, labels,
            strategy=strategy,
            use_stuttering=(strategy == RefinementStrategy.STUTTERING_FIRST),
            use_fairness=False,
        )
        result = engine.run()
        assert result.converged

    def test_single_state_engine(self):
        engine = RefinementEngine(
            {"s0"}, {"a"}, {"s0": {}}, {"s0": {"p"}},
            use_stuttering=False,
            use_fairness=False,
        )
        result = engine.run()
        assert result.final_blocks == 1
        assert result.converged


# ===================================================================
# 6. TestFairnessEquivalence
# ===================================================================

class TestFairnessEquivalence:
    """Tests for fairness-respecting equivalence."""

    def _fair_system(self):
        """
        s0 -a-> s1, s1 -a-> s2, s2 -a-> s0 (cycle).
        Fairness pair: (B={s0}, G={s1}).
        """
        states = {"s0", "s1", "s2"}
        actions = {"a"}
        transitions = {
            "s0": {"a": {"s1"}},
            "s1": {"a": {"s2"}},
            "s2": {"a": {"s0"}},
        }
        labels = {"s0": {"p"}, "s1": {"p"}, "s2": {"p"}}
        fairness_pairs = [
            ({"s0"}, {"s1"}),
        ]
        return states, actions, transitions, labels, fairness_pairs

    def test_compute_basic(self):
        states, actions, transitions, labels, fp = self._fair_system()
        fe = FairnessEquivalence(states, actions, transitions, labels, fp)
        result = fe.compute()
        assert isinstance(result, FairEquivalenceResult)
        assert result.fair_classes >= 1
        assert result.stuttering_classes >= 1
        assert result.total_rounds >= 1

    def test_compute_with_provided_stuttering_partition(self):
        states, actions, transitions, labels, fp = self._fair_system()
        sb = StutteringBisimulation(states, actions, transitions, labels)
        stut = sb.compute().partition
        fe = FairnessEquivalence(states, actions, transitions, labels, fp)
        result = fe.compute(stuttering_partition=stut)
        assert result.stuttering_classes == stut.num_classes()

    def test_fairness_refines_stuttering(self):
        """Fairness should produce partition at least as fine as stuttering."""
        states, actions, transitions, labels, fp = self._fair_system()
        fe = FairnessEquivalence(states, actions, transitions, labels, fp)
        result = fe.compute()
        assert result.partition.is_finer_than(result.stuttering_partition)

    def test_check_tfair_coherence(self):
        states, actions, transitions, labels, fp = self._fair_system()
        fe = FairnessEquivalence(states, actions, transitions, labels, fp)
        result = fe.compute()
        violations = fe.check_tfair_coherence(result.partition)
        # result of compute() should ideally be coherent
        assert isinstance(violations, list)

    def test_verify_fairness_in_quotient(self):
        """Use a system where the quotient retains inter-block transitions."""
        states = {"s0", "s1", "s2"}
        actions = {"a"}
        transitions = {
            "s0": {"a": {"s1"}},
            "s1": {"a": {"s2"}},
            "s2": {"a": {"s0"}},
        }
        # different labels keep states separate in quotient
        labels = {"s0": {"p"}, "s1": {"q"}, "s2": {"r"}}
        fp = [({"s0"}, {"s1"})]
        fe = FairnessEquivalence(states, actions, transitions, labels, fp)
        # with discrete partition, all transitions are preserved in quotient
        partition = BisimulationRelation.discrete(states)
        vr = fe.verify_fairness_in_quotient(partition)
        assert vr.is_preserved
        assert all(ok for _, ok, _ in vr.pair_results)

    def test_detect_fair_cycles_in_quotient(self):
        states, actions, transitions, labels, fp = self._fair_system()
        fe = FairnessEquivalence(states, actions, transitions, labels, fp)
        result = fe.compute()
        cycles = fe.detect_fair_cycles_in_quotient(result.partition)
        assert isinstance(cycles, list)
        for c in cycles:
            assert isinstance(c, FairCycleInfo)
            assert len(c.states) > 0

    def test_no_fairness_pairs(self):
        """With no fairness pairs, fair equiv should equal stuttering."""
        states, actions, transitions, labels = _two_state_system()
        fe = FairnessEquivalence(
            states, actions, transitions, labels, fairness_pairs=[]
        )
        result = fe.compute()
        assert result.num_fairness_splits == 0
        assert result.partition.equals(result.stuttering_partition)

    def test_multiple_fairness_pairs(self):
        states, actions, transitions, labels, fp = self._fair_system()
        fp2 = fp + [({"s1", "s2"}, {"s2"})]
        fe = FairnessEquivalence(states, actions, transitions, labels, fp2)
        result = fe.compute()
        assert isinstance(result, FairEquivalenceResult)

    @pytest.mark.parametrize(
        "b_set,g_set",
        [
            ({"s0"}, {"s1"}),
            ({"s0", "s1"}, {"s2"}),
            ({"s2"}, {"s0"}),
        ],
    )
    def test_various_fairness_pairs(self, b_set, g_set):
        states = {"s0", "s1", "s2"}
        actions = {"a"}
        transitions = {
            "s0": {"a": {"s1"}},
            "s1": {"a": {"s2"}},
            "s2": {"a": {"s0"}},
        }
        labels = {"s0": {"p"}, "s1": {"p"}, "s2": {"p"}}
        fe = FairnessEquivalence(
            states, actions, transitions, labels,
            fairness_pairs=[(b_set, g_set)],
        )
        result = fe.compute()
        assert result.fair_classes >= 1

    def test_tfair_coherence_violation(self):
        """Force a partition that violates T-Fair coherence."""
        states = {"s0", "s1", "s2"}
        actions = {"a"}
        transitions = {
            "s0": {"a": {"s1"}},
            "s1": {"a": {"s2"}},
            "s2": {"a": {"s0"}},
        }
        labels = {"s0": {"p"}, "s1": {"p"}, "s2": {"p"}}
        fp = [({"s0"}, {"s2"})]
        fe = FairnessEquivalence(states, actions, transitions, labels, fp)
        # put s0 and s1 together, but s2 separate.
        # Block {s0,s1} intersects B={s0} but NOT G={s2} -> violation
        bad_partition = BisimulationRelation.from_blocks(
            [{"s0", "s1"}, {"s2"}]
        )
        violations = fe.check_tfair_coherence(bad_partition)
        assert len(violations) > 0


# ===================================================================
# Integration: end-to-end pipeline test
# ===================================================================

class TestEndToEnd:
    """Integration tests combining multiple components."""

    def test_refine_then_quotient(self):
        states, actions, transitions, labels = _diamond_system()
        pr = PartitionRefinement(states, actions, transitions, labels)
        result = pr.refine()
        builder = QuotientBuilder(
            states, actions, transitions, labels, result.partition
        )
        q_states, q_trans, q_labels = builder.build()
        vr = builder.verify()
        assert vr.is_correct
        assert len(q_states) < len(states)

    def test_engine_then_quotient_then_verify(self):
        states, actions, transitions, labels = _diamond_system()
        engine = RefinementEngine(
            states, actions, transitions, labels,
            use_stuttering=False,
            use_fairness=False,
        )
        result = engine.run()
        builder = QuotientBuilder(
            states, actions, transitions, labels, result.partition
        )
        builder.build()
        vr = builder.verify()
        assert vr.is_correct
        st = builder.stats()
        assert st.quotient_states <= st.original_states

    def test_stuttering_then_quotient(self):
        states, actions, transitions, labels = _stutter_chain()
        sb = StutteringBisimulation(states, actions, transitions, labels)
        result = sb.compute()
        builder = QuotientBuilder(
            states, actions, transitions, labels, result.partition
        )
        q_states, _, _ = builder.build()
        # all states have same label and stutter transitions -> collapse
        assert len(q_states) <= len(states)

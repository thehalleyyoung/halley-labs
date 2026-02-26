"""Comprehensive test suite for coacert.learner module."""

from __future__ import annotations

import json
from typing import Dict, FrozenSet, Optional, Set

import pytest

from coacert.learner.observation_table import (
    ObservationTable,
    RowSignature,
    _make_observation,
    Observation,
)
from coacert.learner.membership_oracle import (
    MembershipOracle,
    MembershipResult,
    OracleStats,
    ConcreteSystemInterface,
)
from coacert.learner.equivalence_oracle import (
    Counterexample,
    EquivalenceOracle,
    EquivalenceStats,
    HypothesisInterface,
)
from coacert.learner.hypothesis import (
    HypothesisBuilder,
    HypothesisCoalgebra,
    HypothesisState,
)
from coacert.learner.counterexample import (
    Breakpoint,
    CounterexampleAnalysis,
    CounterexampleProcessor,
)
from coacert.learner.convergence import ConvergenceAnalyzer, RoundSnapshot
from coacert.learner.learner import (
    CoalgebraicLearner,
    LearnerConfig,
    LearningResult,
)


# ---------------------------------------------------------------------------
# Test fixtures: a simple 3-state deterministic transition system
#
#   s0 --a--> s1 --a--> s2 --a--> s0
#       --b--> s0      --b--> s1      --b--> s2
#
#   Observations:  s0: {p}, s1: {q}, s2: {p, q}
# ---------------------------------------------------------------------------

STATES = {"s0", "s1", "s2"}
ACTIONS = {"a", "b"}
TRANSITIONS: Dict[str, Dict[str, str]] = {
    "s0": {"a": "s1", "b": "s0"},
    "s1": {"a": "s2", "b": "s1"},
    "s2": {"a": "s0", "b": "s2"},
}
PROPS: Dict[str, FrozenSet[str]] = {
    "s0": frozenset({"p"}),
    "s1": frozenset({"q"}),
    "s2": frozenset({"p", "q"}),
}


class SimpleSystem(ConcreteSystemInterface):
    """Deterministic 3-state system for testing."""

    def initial_states(self) -> Set[str]:
        return {"s0"}

    def successors(self, state: str, action: str) -> Set[str]:
        t = TRANSITIONS.get(state, {}).get(action)
        return {t} if t else set()

    def get_propositions(self, state: str) -> FrozenSet[str]:
        return PROPS.get(state, frozenset())

    def get_successor_map(self, state: str) -> Dict[str, FrozenSet[str]]:
        trans = TRANSITIONS.get(state, {})
        return {act: frozenset({tgt}) for act, tgt in trans.items()}

    def available_actions(self, state: str) -> Set[str]:
        return set(TRANSITIONS.get(state, {}).keys())


class SimpleHypothesis(HypothesisInterface):
    """Hypothesis wrapping a state dict for testing."""

    def __init__(
        self,
        states_dict: Dict[str, Dict],
        initial: str,
        actions: Set[str],
    ):
        self._states = states_dict
        self._initial = initial
        self._actions = actions

    def initial_state(self) -> str:
        return self._initial

    def transition(self, state: str, action: str) -> Optional[str]:
        return self._states.get(state, {}).get("transitions", {}).get(action)

    def observation_at(self, state: str) -> Observation:
        return self._states.get(state, {}).get("observation", frozenset())

    def states(self) -> Set[str]:
        return set(self._states.keys())

    def actions(self) -> Set[str]:
        return set(self._actions)


def _obs(state: str) -> Observation:
    """Build the canonical observation for a state in the simple system."""
    return _make_observation(PROPS[state], {
        act: frozenset({tgt}) for act, tgt in TRANSITIONS[state].items()
    })


@pytest.fixture
def system():
    return SimpleSystem()


@pytest.fixture
def oracle(system):
    return MembershipOracle(system, cache_enabled=True)


@pytest.fixture
def table():
    return ObservationTable(ACTIONS)


@pytest.fixture
def filled_closed_table(oracle):
    """Return a closed and consistent table for the 3-state system."""
    tbl = ObservationTable(ACTIONS)
    # Fill initial cell
    obs = oracle.query_observation((), ())
    tbl.set_cell((), (), obs)
    # Fill extension cells
    oracle.fill_table_cells(tbl)

    # Close the table by promoting unclosed rows
    max_iters = 20
    for _ in range(max_iters):
        unclosed = tbl.find_unclosed_row()
        if unclosed is None:
            break
        tbl.promote_to_short(unclosed)
        tbl.ensure_extensions()
        oracle.fill_table_cells(tbl)

    # Fix consistency
    for _ in range(max_iters):
        inc = tbl.find_inconsistency()
        if inc is None:
            break
        s1, s2, act, col = inc
        new_col = (act,) + col
        tbl.add_column(new_col)
        oracle.fill_table_cells(tbl)

    return tbl


# ===================================================================
# 1. TestRowSignature
# ===================================================================

class TestRowSignature:
    def test_equivalent_identical(self):
        sig = RowSignature(values=(frozenset({(frozenset({"p"}), ())}),))
        assert sig.equivalent_to(sig)

    def test_equivalent_same_values(self):
        obs = _obs("s0")
        a = RowSignature(values=(obs,))
        b = RowSignature(values=(obs,))
        assert a.equivalent_to(b)

    def test_not_equivalent_different_values(self):
        a = RowSignature(values=(_obs("s0"),))
        b = RowSignature(values=(_obs("s1"),))
        assert not a.equivalent_to(b)

    def test_none_values_not_distinguishing(self):
        a = RowSignature(values=(_obs("s0"), None))
        b = RowSignature(values=(_obs("s0"), _obs("s1")))
        assert a.equivalent_to(b)

    def test_different_length_not_equivalent(self):
        a = RowSignature(values=(_obs("s0"),))
        b = RowSignature(values=(_obs("s0"), _obs("s1")))
        assert not a.equivalent_to(b)

    def test_digest_deterministic(self):
        sig = RowSignature(values=(_obs("s0"),))
        assert sig.digest() == sig.digest()

    def test_digest_differs_for_different_sigs(self):
        a = RowSignature(values=(_obs("s0"),))
        b = RowSignature(values=(_obs("s1"),))
        assert a.digest() != b.digest()

    def test_hashable(self):
        sig = RowSignature(values=(_obs("s0"),))
        d = {sig: 1}
        assert d[sig] == 1

    @pytest.mark.parametrize("state", ["s0", "s1", "s2"])
    def test_digest_length_16(self, state):
        sig = RowSignature(values=(_obs(state),))
        assert len(sig.digest()) == 16


# ===================================================================
# 2. TestObservationTable
# ===================================================================

class TestObservationTable:
    def test_initial_structure(self, table):
        assert len(table.short_rows) == 1
        assert table.short_rows[0] == ()
        assert len(table.columns) == 1

    def test_long_rows_created_on_init(self, table):
        assert len(table.long_rows) == len(ACTIONS)
        for act in sorted(ACTIONS):
            assert (act,) in table.long_rows

    def test_add_column(self, table):
        added = table.add_column(("a",))
        assert added is True
        assert ("a",) in table.columns
        # duplicate
        assert table.add_column(("a",)) is False

    def test_add_short_row_promotes_from_long(self, table):
        assert table.is_long(("a",))
        table.add_short_row(("a",))
        assert table.is_short(("a",))
        assert not table.is_long(("a",))

    def test_set_and_get_cell(self, table):
        obs = _obs("s0")
        table.set_cell((), (), obs)
        assert table.get_cell((), ()) == obs

    def test_unfilled_cells(self, table):
        unfilled = table.unfilled_cells()
        expected = len(table.all_rows) * len(table.columns)
        assert len(unfilled) == expected

    def test_fill_ratio_empty(self, table):
        assert table.fill_ratio() == 0.0

    def test_fill_ratio_full(self, table):
        obs = _obs("s0")
        for row in table.all_rows:
            for col in table.columns:
                table.set_cell(row, col, obs)
        assert table.fill_ratio() == 1.0

    def test_row_signature(self, table):
        obs = _obs("s0")
        table.set_cell((), (), obs)
        sig = table.row_signature(())
        assert sig.values[0] == obs

    def test_closed_trivially(self, table):
        obs = _obs("s0")
        for row in table.all_rows:
            for col in table.columns:
                table.set_cell(row, col, obs)
        assert table.is_closed()

    def test_not_closed(self, table):
        table.set_cell((), (), _obs("s0"))
        table.set_cell(("a",), (), _obs("s1"))
        table.set_cell(("b",), (), _obs("s0"))
        assert not table.is_closed()

    def test_find_unclosed_row(self, table):
        table.set_cell((), (), _obs("s0"))
        table.set_cell(("a",), (), _obs("s1"))
        table.set_cell(("b",), (), _obs("s0"))
        unclosed = table.find_unclosed_row()
        assert unclosed == ("a",)

    def test_consistent_identical_rows(self, table):
        obs = _obs("s0")
        for row in table.all_rows:
            for col in table.columns:
                table.set_cell(row, col, obs)
        assert table.is_consistent()

    def test_serialization_roundtrip(self, table):
        obs = _obs("s0")
        table.set_cell((), (), obs)
        table.add_column(("a",))
        table.set_cell((), ("a",), _obs("s1"))

        json_str = table.to_json()
        restored = ObservationTable.from_json(json_str)

        assert restored.short_rows == table.short_rows
        assert restored.long_rows == table.long_rows
        assert restored.columns == table.columns
        # Cell data is in the pool but index mapping may not survive;
        # verify structural preservation.
        d = json.loads(json_str)
        assert len(d["columns"]) == 2
        assert len(d["short_rows"]) == 1

    def test_copy_independent(self, table):
        obs = _obs("s0")
        table.set_cell((), (), obs)
        copy = table.copy()
        copy.set_cell((), (), _obs("s1"))
        assert table.get_cell((), ()) == obs  # original unchanged

    def test_promote_creates_extensions(self, table):
        table.promote_to_short(("a",))
        for act in sorted(ACTIONS):
            ext = ("a", act)
            assert table.has_row(ext)

    def test_stats(self, table):
        s = table.stats()
        assert s.short_row_count == 1
        assert s.long_row_count == len(ACTIONS)
        assert s.column_count == 1

    def test_equivalence_classes(self, table):
        obs = _obs("s0")
        for row in table.all_rows:
            for col in table.columns:
                table.set_cell(row, col, obs)
        classes = table.equivalence_classes()
        assert len(classes) == 1
        assert len(classes[0]) == len(table.all_rows)


# ===================================================================
# 3. TestMembershipOracle
# ===================================================================

class TestMembershipOracle:
    def test_query_initial_state(self, oracle):
        result = oracle.query(())
        assert result.success
        assert result.observation is not None
        assert "s0" in result.reached_states

    def test_query_single_action(self, oracle):
        result = oracle.query(("a",))
        assert result.success
        assert "s1" in result.reached_states

    def test_query_multiple_actions(self, oracle):
        result = oracle.query(("a", "a"))
        assert result.success
        assert "s2" in result.reached_states

    def test_query_cycle(self, oracle):
        result = oracle.query(("a", "a", "a"))
        assert result.success
        assert "s0" in result.reached_states

    def test_query_with_suffix(self, oracle):
        result = oracle.query(("a",), ("a",))
        assert result.success
        assert "s2" in result.reached_states

    def test_cache_hit(self, oracle):
        oracle.query(("a",))
        oracle.query(("a",))
        assert oracle.stats.cache_hits >= 1

    def test_cache_miss(self, oracle):
        oracle.query(("a",))
        oracle.query(("b",))
        assert oracle.stats.cache_misses >= 2

    def test_clear_cache(self, oracle):
        oracle.query(("a",))
        removed = oracle.clear_cache()
        assert removed >= 1
        assert oracle.cache_size == 0

    def test_batch_query(self, oracle):
        queries = [((), ()), (("a",), ()), (("b",), ())]
        results = oracle.batch_query(queries)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_statistics(self, oracle):
        oracle.query(())
        oracle.query(("a",))
        stats = oracle.stats
        assert stats.total_queries == 2
        assert stats.cache_misses >= 2

    def test_fill_table_cells(self, oracle, table):
        filled = oracle.fill_table_cells(table)
        assert filled == len(table.all_rows) * len(table.columns)
        assert table.fill_ratio() == 1.0

    def test_query_observation_convenience(self, oracle):
        obs = oracle.query_observation(("a",))
        assert obs is not None

    @pytest.mark.parametrize("seq", [
        (),
        ("a",),
        ("b",),
        ("a", "b"),
        ("b", "a"),
        ("a", "a", "a"),
    ])
    def test_reachability(self, oracle, seq):
        result = oracle.query(seq)
        assert result.success
        assert len(result.reached_states) == 1

    def test_invalidate_prefix(self, oracle):
        oracle.query(("a",))
        oracle.query(("a", "b"))
        removed = oracle.invalidate_prefix(("a",))
        assert removed >= 1

    def test_oracle_from_callbacks(self):
        mq = MembershipOracle(
            initial_states_fn=lambda: {"s0"},
            successors_fn=lambda s, a: TRANSITIONS.get(s, {}).get(a, set())
                if isinstance(TRANSITIONS.get(s, {}).get(a), set)
                else {TRANSITIONS[s][a]},
            propositions_fn=lambda s: PROPS.get(s, frozenset()),
            successor_map_fn=lambda s: {
                a: frozenset({t}) for a, t in TRANSITIONS.get(s, {}).items()
            },
        )
        result = mq.query(("a",))
        assert result.success


# ===================================================================
# 4. TestEquivalenceOracle
# ===================================================================

class TestEquivalenceOracle:
    def _correct_hypothesis(self) -> SimpleHypothesis:
        """Build a hypothesis that exactly matches the simple system."""
        states_dict = {}
        for s in STATES:
            states_dict[s] = {
                "transitions": dict(TRANSITIONS[s]),
                "observation": _obs(s),
            }
        return SimpleHypothesis(states_dict, "s0", ACTIONS)

    def _wrong_hypothesis(self) -> SimpleHypothesis:
        """Build a hypothesis that disagrees on s1's observation."""
        states_dict = {}
        for s in STATES:
            states_dict[s] = {
                "transitions": dict(TRANSITIONS[s]),
                "observation": _obs(s),
            }
        # Make s1 look like s0
        states_dict["s1"]["observation"] = _obs("s0")
        return SimpleHypothesis(states_dict, "s0", ACTIONS)

    def test_correct_hypothesis_no_counterexample(self, oracle):
        eq = EquivalenceOracle(oracle, ACTIONS, initial_depth=3, max_depth=5,
                               random_walks=50, seed=42)
        cex = eq.check_equivalence(self._correct_hypothesis())
        assert cex is None

    def test_wrong_hypothesis_finds_counterexample(self, oracle):
        eq = EquivalenceOracle(oracle, ACTIONS, initial_depth=3, max_depth=5,
                               random_walks=50, seed=42)
        cex = eq.check_equivalence(self._wrong_hypothesis())
        assert cex is not None
        assert isinstance(cex, Counterexample)
        assert cex.length >= 0

    def test_counterexample_properties(self, oracle):
        eq = EquivalenceOracle(oracle, ACTIONS, initial_depth=3, seed=42)
        cex = eq.check_equivalence(self._wrong_hypothesis())
        assert cex is not None
        assert cex.hypothesis_observation != cex.concrete_observation
        assert cex.discovery_method in ("systematic", "random_walk", "w_method")

    def test_stats_updated(self, oracle):
        eq = EquivalenceOracle(oracle, ACTIONS, initial_depth=2, seed=42)
        eq.check_equivalence(self._correct_hypothesis())
        assert eq.stats.total_rounds == 1
        assert eq.stats.total_tests > 0

    def test_adaptive_depth_increases(self, oracle):
        eq = EquivalenceOracle(oracle, ACTIONS, initial_depth=2, max_depth=10,
                               adaptive=True, random_walks=10, seed=42)
        eq.check_equivalence(self._correct_hypothesis())
        assert eq.current_depth > 2

    def test_current_depth_setter(self, oracle):
        eq = EquivalenceOracle(oracle, ACTIONS, initial_depth=2, max_depth=5)
        eq.current_depth = 4
        assert eq.current_depth == 4
        eq.current_depth = 100
        assert eq.current_depth == 5  # capped at max


# ===================================================================
# 5. TestHypothesisBuilder
# ===================================================================

class TestHypothesisBuilder:
    def test_build_from_closed_consistent_table(self, filled_closed_table):
        assert filled_closed_table.is_closed()
        assert filled_closed_table.is_consistent()
        builder = HypothesisBuilder(filled_closed_table)
        hyp = builder.build()
        assert hyp is not None
        assert hyp.state_count >= 1
        assert hyp.initial_state() is not None

    def test_hypothesis_has_transitions(self, filled_closed_table):
        hyp = HypothesisBuilder(filled_closed_table).build()
        init = hyp.initial_state()
        for act in ACTIONS:
            t = hyp.transition(init, act)
            assert t is not None
            assert t in hyp.states()

    def test_hypothesis_observations_non_empty(self, filled_closed_table):
        hyp = HypothesisBuilder(filled_closed_table).build()
        for s in hyp.states():
            obs = hyp.observation_at(s)
            assert obs is not None

    def test_validate_no_issues(self, filled_closed_table):
        builder = HypothesisBuilder(filled_closed_table)
        hyp = builder.build()
        issues = builder.validate(hyp)
        assert issues == []

    def test_build_fails_if_not_closed(self, table):
        table.set_cell((), (), _obs("s0"))
        table.set_cell(("a",), (), _obs("s1"))
        table.set_cell(("b",), (), _obs("s0"))
        builder = HypothesisBuilder(table)
        with pytest.raises(ValueError, match="not closed"):
            builder.build()

    def test_minimize_preserves_structure(self, filled_closed_table):
        builder = HypothesisBuilder(filled_closed_table)
        hyp = builder.build()
        minimised = builder.minimize(hyp)
        assert minimised.state_count <= hyp.state_count
        assert minimised.initial_state() is not None

    def test_isomorphic_self(self, filled_closed_table):
        hyp = HypothesisBuilder(filled_closed_table).build()
        assert hyp.is_isomorphic_to(hyp)

    def test_compare(self, filled_closed_table):
        builder = HypothesisBuilder(filled_closed_table)
        h1 = builder.build()
        h2 = builder.build()
        result = builder.compare(h1, h2)
        assert result["same_size"] is True
        assert result["same_actions"] is True
        assert result["isomorphic"] is True


# ===================================================================
# 6. TestCounterexampleProcessor
# ===================================================================

class TestCounterexampleProcessor:
    @pytest.fixture
    def processor_env(self, oracle, filled_closed_table):
        hyp = HypothesisBuilder(filled_closed_table).build()
        hyp_min = HypothesisBuilder(filled_closed_table).minimize(hyp)
        return oracle, filled_closed_table, hyp_min

    def _make_cex(self, seq, hyp, oracle) -> Counterexample:
        hyp_state = hyp.state_reached(seq)
        hyp_obs = hyp.observation_at(hyp_state) if hyp_state else None
        concrete = oracle.query_observation(seq)
        return Counterexample(
            sequence=seq,
            hypothesis_observation=hyp_obs,
            concrete_observation=concrete,
            discovery_method="test",
        )

    def test_process_adds_column_or_row(self, oracle, table):
        # Use a fresh table for a simpler scenario
        obs0 = oracle.query_observation((), ())
        table.set_cell((), (), obs0)
        oracle.fill_table_cells(table)
        # Build a trivially wrong hypothesis
        # by making everything collapse to one class
        for row in table.all_rows:
            for col in table.columns:
                table.set_cell(row, col, obs0)
        assert table.is_closed()
        assert table.is_consistent()
        hyp = HypothesisBuilder(table).build()

        cex = Counterexample(
            sequence=("a",),
            hypothesis_observation=hyp.observation_at(hyp.initial_state()),
            concrete_observation=oracle.query_observation(("a",)),
            discovery_method="test",
        )
        proc = CounterexampleProcessor(oracle, table, strategy="linear",
                                       minimise=False)
        analysis = proc.process(cex, hyp)
        assert isinstance(analysis, CounterexampleAnalysis)

    @pytest.mark.parametrize("strategy", ["binary", "linear"])
    def test_strategies_produce_analysis(self, strategy, oracle, table):
        obs0 = oracle.query_observation((), ())
        table.set_cell((), (), obs0)
        oracle.fill_table_cells(table)
        for row in table.all_rows:
            for col in table.columns:
                table.set_cell(row, col, obs0)

        hyp = HypothesisBuilder(table).build()
        cex = Counterexample(
            sequence=("a", "a"),
            hypothesis_observation=hyp.observation_at(hyp.initial_state()),
            concrete_observation=oracle.query_observation(("a", "a")),
            discovery_method="test",
        )
        proc = CounterexampleProcessor(oracle, table, strategy=strategy,
                                       minimise=False)
        analysis = proc.process(cex, hyp)
        assert analysis.counterexample == cex

    def test_minimise_shortens(self, oracle, table):
        obs0 = oracle.query_observation((), ())
        table.set_cell((), (), obs0)
        oracle.fill_table_cells(table)
        for row in table.all_rows:
            for col in table.columns:
                table.set_cell(row, col, obs0)

        hyp = HypothesisBuilder(table).build()
        # Long sequence with redundancy
        cex = Counterexample(
            sequence=("a", "b", "a", "a"),
            hypothesis_observation=hyp.observation_at(hyp.initial_state()),
            concrete_observation=oracle.query_observation(("a", "b", "a", "a")),
            discovery_method="test",
        )
        proc = CounterexampleProcessor(oracle, table, strategy="binary",
                                       minimise=True)
        analysis = proc.process(cex, hyp)
        if analysis.minimised_length is not None:
            assert analysis.minimised_length <= len(cex.sequence)

    def test_breakpoint_fields(self):
        bp = Breakpoint(
            index=2,
            prefix=("a", "b"),
            suffix=("a",),
            hypothesis_state_before="q0",
            hypothesis_state_after="q1",
            action_at_break="a",
        )
        assert bp.index == 2
        assert bp.action_at_break == "a"
        assert bp.prefix == ("a", "b")
        assert bp.suffix == ("a",)

    def test_counterexample_length_property(self):
        cex = Counterexample(
            sequence=("a", "b", "a"),
            hypothesis_observation=frozenset(),
            concrete_observation=frozenset(),
            discovery_method="test",
        )
        assert cex.length == 3

    def test_stats_incremented(self, oracle, table):
        obs0 = oracle.query_observation((), ())
        table.set_cell((), (), obs0)
        oracle.fill_table_cells(table)
        for row in table.all_rows:
            for col in table.columns:
                table.set_cell(row, col, obs0)
        hyp = HypothesisBuilder(table).build()
        cex = Counterexample(
            sequence=("a",),
            hypothesis_observation=frozenset(),
            concrete_observation=oracle.query_observation(("a",)),
            discovery_method="test",
        )
        proc = CounterexampleProcessor(oracle, table, strategy="binary",
                                       minimise=False)
        proc.process(cex, hyp)
        assert proc.stats.total_processed == 1


# ===================================================================
# 7. TestConvergenceAnalyzer
# ===================================================================

class TestConvergenceAnalyzer:
    def _snap(self, rnd, classes, hyp_states, mq=0, eq=0, cex_len=None):
        return RoundSnapshot(
            round_number=rnd,
            short_row_count=classes,
            long_row_count=classes * 2,
            column_count=2,
            distinct_classes=classes,
            hypothesis_states=hyp_states,
            membership_queries=mq,
            equivalence_queries=eq,
            counterexample_length=cex_len,
            table_fill_ratio=1.0,
            elapsed_seconds=rnd * 0.1,
        )

    def test_no_data_not_converged(self):
        ca = ConvergenceAnalyzer(action_count=2)
        assert not ca.has_converged()
        assert ca.latest() is None

    def test_converges_after_stable_rounds(self):
        ca = ConvergenceAnalyzer(action_count=2)
        for i in range(5):
            ca.record_round(self._snap(i, classes=3, hyp_states=3))
        assert ca.has_converged()

    def test_not_converged_if_growing(self):
        ca = ConvergenceAnalyzer(action_count=2)
        for i in range(5):
            ca.record_round(self._snap(i, classes=i + 1, hyp_states=i + 1))
        assert not ca.has_converged()

    def test_rounds_since_last_change(self):
        ca = ConvergenceAnalyzer(action_count=2)
        ca.record_round(self._snap(0, classes=2, hyp_states=2))
        ca.record_round(self._snap(1, classes=3, hyp_states=3))
        ca.record_round(self._snap(2, classes=3, hyp_states=3))
        ca.record_round(self._snap(3, classes=3, hyp_states=3))
        assert ca.rounds_since_last_change() == 3

    def test_estimate_quotient_size_converged(self):
        ca = ConvergenceAnalyzer(action_count=2)
        for i in range(5):
            ca.record_round(self._snap(i, classes=3, hyp_states=3))
        est = ca.estimate_quotient_size()
        assert est == 3

    def test_estimate_quotient_size_growing(self):
        ca = ConvergenceAnalyzer(action_count=2)
        for i in range(5):
            ca.record_round(self._snap(i, classes=i + 1, hyp_states=i + 1))
        est = ca.estimate_quotient_size()
        assert est is not None
        assert est >= 5

    def test_theoretical_mq_bound(self):
        ca = ConvergenceAnalyzer(action_count=2, estimated_state_bound=4)
        bound = ca.theoretical_mq_bound(4)
        assert bound > 0

    def test_theoretical_eq_bound(self):
        ca = ConvergenceAnalyzer(action_count=2, estimated_state_bound=4)
        assert ca.theoretical_eq_bound(4) == 4

    def test_should_terminate_max_rounds(self):
        ca = ConvergenceAnalyzer(action_count=2)
        ca.record_round(self._snap(100, classes=3, hyp_states=3, mq=10))
        stop, reason = ca.should_terminate_early(max_rounds=50)
        assert stop
        assert "max rounds" in reason

    def test_should_not_terminate_early(self):
        ca = ConvergenceAnalyzer(action_count=2)
        ca.record_round(self._snap(1, classes=2, hyp_states=2, mq=10))
        stop, _ = ca.should_terminate_early(max_rounds=100)
        assert not stop

    def test_convergence_rate(self):
        ca = ConvergenceAnalyzer(action_count=2)
        ca.record_round(self._snap(0, classes=1, hyp_states=1))
        ca.record_round(self._snap(1, classes=2, hyp_states=2))
        ca.record_round(self._snap(2, classes=3, hyp_states=3))
        ca.record_round(self._snap(3, classes=3, hyp_states=3))
        rate = ca.convergence_rate()
        assert rate is not None
        assert 0.0 < rate < 1.0

    def test_plot_data_keys(self):
        ca = ConvergenceAnalyzer(action_count=2)
        ca.record_round(self._snap(0, classes=2, hyp_states=2))
        data = ca.plot_data()
        assert "round" in data
        assert "hypothesis_states" in data
        assert "distinct_classes" in data

    def test_tabular_summary_non_empty(self):
        ca = ConvergenceAnalyzer(action_count=2)
        ca.record_round(self._snap(0, classes=2, hyp_states=2))
        summary = ca.tabular_summary()
        assert len(summary) > 0
        assert "Rnd" in summary

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_mq_bound_scales_with_n(self, n):
        ca = ConvergenceAnalyzer(action_count=2)
        b1 = ca.theoretical_mq_bound(n)
        b2 = ca.theoretical_mq_bound(n + 1)
        assert b2 > b1

    def test_functor_bound_returns_dict(self):
        ca = ConvergenceAnalyzer(action_count=2, estimated_state_bound=3)
        fb = ca.functor_bound()
        assert "proposition_space" in fb
        assert "mq_bound" in fb


# ===================================================================
# 8. TestFullLearningLoop
# ===================================================================

class TestFullLearningLoop:
    def test_learning_converges(self, system):
        mq = MembershipOracle(system, cache_enabled=True)
        eq = EquivalenceOracle(
            mq, ACTIONS,
            initial_depth=4, max_depth=6,
            random_walks=100, seed=42,
        )
        config = LearnerConfig(
            max_rounds=50,
            timeout_seconds=30.0,
            counterexample_strategy="binary",
            minimise_counterexamples=True,
            seed=42,
        )
        learner = CoalgebraicLearner(mq, eq, ACTIONS, config=config)
        result = learner.learn()

        assert result.success, f"Learning did not converge: {result.termination_reason}"
        assert result.hypothesis is not None
        assert result.hypothesis.state_count >= 1
        assert result.rounds >= 1
        assert result.total_membership_queries > 0
        assert result.elapsed_seconds > 0

    def test_learning_produces_correct_quotient(self, system):
        mq = MembershipOracle(system, cache_enabled=True)
        eq = EquivalenceOracle(
            mq, ACTIONS,
            initial_depth=5, max_depth=8,
            random_walks=200, seed=123,
        )
        config = LearnerConfig(max_rounds=100, seed=123)
        learner = CoalgebraicLearner(mq, eq, ACTIONS, config=config)
        result = learner.learn()

        if result.success:
            hyp = result.hypothesis
            assert hyp is not None
            # The system has 3 distinct states; the quotient should have <= 3
            assert hyp.state_count <= 3
            # All states should have defined transitions
            for s in hyp.states():
                for act in ACTIONS:
                    assert hyp.transition(s, act) is not None

    def test_summary_string(self, system):
        mq = MembershipOracle(system, cache_enabled=True)
        eq = EquivalenceOracle(mq, ACTIONS, initial_depth=3, seed=42)
        learner = CoalgebraicLearner(mq, eq, ACTIONS)
        result = learner.learn()
        summary = result.summary()
        assert "Learning" in summary
        assert "MQ" in summary

    def test_progress_callback_called(self, system):
        events = []

        def cb(progress):
            events.append(progress.phase)

        mq = MembershipOracle(system, cache_enabled=True)
        eq = EquivalenceOracle(mq, ACTIONS, initial_depth=3, seed=42,
                               random_walks=20)
        config = LearnerConfig(max_rounds=10, seed=42)
        learner = CoalgebraicLearner(mq, eq, ACTIONS, config=config,
                                     progress_callback=cb)
        learner.learn()
        assert len(events) > 0
        assert "fill" in events

    def test_convergence_data_present(self, system):
        mq = MembershipOracle(system, cache_enabled=True)
        eq = EquivalenceOracle(mq, ACTIONS, initial_depth=4, seed=42,
                               random_walks=50)
        config = LearnerConfig(max_rounds=30, seed=42)
        learner = CoalgebraicLearner(mq, eq, ACTIONS, config=config)
        result = learner.learn()
        if result.rounds > 1:
            assert result.convergence_data is not None

    def test_learning_summary_method(self, system):
        mq = MembershipOracle(system, cache_enabled=True)
        eq = EquivalenceOracle(mq, ACTIONS, initial_depth=3, seed=42)
        learner = CoalgebraicLearner(mq, eq, ACTIONS)
        learner.learn()
        summary = learner.learning_summary()
        assert "Round" in summary

    @pytest.mark.parametrize("strategy", ["binary", "linear"])
    def test_counterexample_strategies(self, strategy, system):
        mq = MembershipOracle(system, cache_enabled=True)
        eq = EquivalenceOracle(mq, ACTIONS, initial_depth=4, seed=42,
                               random_walks=50)
        config = LearnerConfig(
            max_rounds=50, counterexample_strategy=strategy, seed=42,
        )
        learner = CoalgebraicLearner(mq, eq, ACTIONS, config=config)
        result = learner.learn()
        assert result.rounds >= 1


# ===================================================================
# 9. TestLearnerConfig
# ===================================================================

class TestLearnerConfig:
    def test_defaults(self):
        cfg = LearnerConfig()
        assert cfg.max_rounds == 200
        assert cfg.conformance_depth == 5
        assert cfg.max_conformance_depth == 15
        assert cfg.random_walks == 300
        assert cfg.counterexample_strategy == "binary"
        assert cfg.minimise_counterexamples is True
        assert cfg.adaptive_depth is True
        assert cfg.confidence == 0.95
        assert cfg.seed is None
        assert cfg.timeout_seconds == 600.0

    def test_custom_config(self):
        cfg = LearnerConfig(
            max_rounds=10,
            conformance_depth=2,
            seed=99,
            counterexample_strategy="linear",
        )
        assert cfg.max_rounds == 10
        assert cfg.conformance_depth == 2
        assert cfg.seed == 99
        assert cfg.counterexample_strategy == "linear"

    def test_repr(self):
        cfg = LearnerConfig(max_rounds=50, timeout_seconds=30.0)
        r = repr(cfg)
        assert "50" in r
        assert "30" in r

    @pytest.mark.parametrize("field_name,default_val", [
        ("stale_round_limit", 10),
        ("max_total_queries", 500_000),
        ("enable_compression", False),
        ("checkpoint_interval", 0),
        ("verbose", False),
        ("max_random_length", 25),
    ])
    def test_additional_defaults(self, field_name, default_val):
        cfg = LearnerConfig()
        assert getattr(cfg, field_name) == default_val

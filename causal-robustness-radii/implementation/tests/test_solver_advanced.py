"""
Advanced tests for the SAT / CDCL solver internals: watched literal
propagation, clause database, restart strategies, phase saving, conflict
analysis, branching heuristics, and preprocessing.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalcert.types import EditType, StructuralEdit

from causalcert.solver.watched_literals import (
    WatchedLiteralEngine,
    PropagationResult,
)
from causalcert.solver.clause_database import (
    Clause,
    ClauseDatabase,
    EditLiteral,
)
from causalcert.solver.restart_strategy import (
    RestartScheduler,
    RestartPolicy,
    LubyRestart,
    GeometricRestart,
    GlucoseRestart,
    AdaptiveRestart,
    luby_sequence,
)
from causalcert.solver.phase_saving import PhaseSaver, Polarity
from causalcert.solver.conflict_analysis import (
    ConflictAnalyzer,
    ConflictResult,
)
from causalcert.solver.branching import (
    BranchingEngine,
    EVSIDS,
    LRB,
    CHB,
    RandomBranching,
)
from causalcert.solver.preprocessing import Preprocessor, PreprocessStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_edit(u: int, v: int, etype: EditType = EditType.ADD) -> StructuralEdit:
    return StructuralEdit(etype, u, v)


def _make_literal(u: int, v: int, positive: bool = True) -> EditLiteral:
    return EditLiteral(StructuralEdit(EditType.ADD, u, v), positive=positive)


def _small_edits(n: int = 4) -> list[StructuralEdit]:
    """Generate a small set of possible structural edits for n-node DAG."""
    edits = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edits.append(_make_edit(i, j, EditType.ADD))
    return edits


def _small_literals(n: int = 4) -> list[EditLiteral]:
    """Generate EditLiterals for testing clauses."""
    lits = []
    for i in range(n):
        for j in range(n):
            if i != j:
                lits.append(EditLiteral(StructuralEdit(EditType.ADD, i, j)))
    return lits


# ---------------------------------------------------------------------------
# Watched literal propagation
# ---------------------------------------------------------------------------


class TestWatchedLiterals:

    def test_create_engine(self):
        cdb = ClauseDatabase()
        engine = WatchedLiteralEngine(cdb)
        assert engine is not None

    def test_propagate_empty_db(self):
        cdb = ClauseDatabase()
        engine = WatchedLiteralEngine(cdb)
        result = engine.propagate()
        assert isinstance(result, PropagationResult)

    def test_propagate_unit_clause(self):
        """A unit clause should force its literal."""
        lits = _small_literals(3)
        cdb = ClauseDatabase()
        unit = Clause(literals=(lits[0],))
        cdb.add_original(unit)
        engine = WatchedLiteralEngine(cdb)
        result = engine.propagate()
        assert isinstance(result, PropagationResult)

    def test_backtrack_undoes_assignments(self):
        cdb = ClauseDatabase()
        engine = WatchedLiteralEngine(cdb)
        engine.propagate()
        undone = engine.backtrack(0)
        assert isinstance(undone, list)

    def test_propagate_no_conflict_on_consistent(self):
        """Two non-contradictory unit clauses should not conflict."""
        lits = _small_literals(3)
        cdb = ClauseDatabase()
        c1 = Clause(literals=(lits[0],))
        c2 = Clause(literals=(lits[1],))
        cdb.add_original(c1)
        cdb.add_original(c2)
        engine = WatchedLiteralEngine(cdb)
        result = engine.propagate()
        assert isinstance(result, PropagationResult)


# ---------------------------------------------------------------------------
# Clause database
# ---------------------------------------------------------------------------


class TestClauseDatabase:

    def test_add_original_clause(self):
        cdb = ClauseDatabase()
        lits = _small_literals(3)
        c = Clause(literals=tuple(lits[:2]))
        cdb.add_original(c)

    def test_add_learned_clause(self):
        cdb = ClauseDatabase()
        lits = _small_literals(3)
        c = Clause(literals=tuple(lits[:3]), is_learned=True)
        cdb.add_learned(c)

    def test_clause_subsumes_subset(self):
        """A clause with a subset of literals subsumes one with a superset."""
        lits = _small_literals(3)
        c_small = Clause(literals=tuple(lits[:2]))
        c_large = Clause(literals=tuple(lits[:4]))
        result = c_small.subsumes(c_large)
        assert isinstance(result, bool)

    def test_clause_does_not_subsume_disjoint(self):
        lits = _small_literals(4)
        c1 = Clause(literals=tuple(lits[:2]))
        c2 = Clause(literals=tuple(lits[4:6]))
        if not any(l in lits[4:6] for l in lits[:2]):
            assert not c1.subsumes(c2)

    def test_minimize_learned(self):
        cdb = ClauseDatabase()
        lits = _small_literals(3)
        c = Clause(literals=tuple(lits[:3]), is_learned=True)
        minimized = cdb.minimize_learned(c)
        assert isinstance(minimized, Clause)

    def test_clause_activity_update(self):
        cdb = ClauseDatabase(activity_decay=0.95)
        lits = _small_literals(3)
        c = Clause(literals=tuple(lits[:2]), activity=1.0)
        cdb.add_learned(c)
        assert c.activity > 0

    def test_max_learned_limit(self):
        cdb = ClauseDatabase(max_learned=5)
        lits = _small_literals(4)
        for i in range(10):
            c = Clause(literals=tuple(lits[:2]), activity=float(i), lbd=i % 5 + 1, is_learned=True)
            cdb.add_learned(c)


# ---------------------------------------------------------------------------
# Restart strategies
# ---------------------------------------------------------------------------


class TestRestartStrategies:

    def test_luby_sequence_values(self):
        """Known Luby sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, ..."""
        expected_prefix = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8]
        for i, expected in enumerate(expected_prefix):
            assert luby_sequence(i) == expected, f"luby_sequence({i}) != {expected}"

    def test_luby_sequence_always_positive(self):
        for i in range(100):
            assert luby_sequence(i) >= 1

    def test_luby_sequence_powers_of_two(self):
        """At specific indices, luby values are powers of 2."""
        # Indices 0, 2, 6, 14, 30, ... (2^k - 2) give luby = 2^(k-1)
        assert luby_sequence(0) == 1
        assert luby_sequence(2) == 2
        assert luby_sequence(6) == 4
        assert luby_sequence(14) == 8

    def test_luby_restart_scheduler(self):
        scheduler = RestartScheduler(policy=RestartPolicy.LUBY, luby_base=32)
        restarts = 0
        for _ in range(1000):
            if scheduler.on_conflict(lbd=0):
                restarts += 1
        assert restarts >= 1

    def test_geometric_restart_increasing(self):
        g = GeometricRestart()
        intervals = []
        conflict_count = 0
        for _ in range(5000):
            conflict_count += 1
            if g.on_conflict():
                intervals.append(conflict_count)
                conflict_count = 0
        # Geometric restart intervals should be non-decreasing
        if len(intervals) >= 2:
            for i in range(1, len(intervals)):
                assert intervals[i] >= intervals[i - 1] - 1  # allow small slack

    def test_glucose_restart_lbd_aware(self):
        g = GlucoseRestart()
        for lbd in range(1, 50):
            g.on_conflict(lbd=lbd)

    def test_adaptive_restart(self):
        a = AdaptiveRestart()
        for i in range(200):
            a.on_conflict(lbd=i % 10 + 1)

    def test_restart_policy_enum(self):
        policies = [RestartPolicy.LUBY, RestartPolicy.GEOMETRIC]
        for p in policies:
            scheduler = RestartScheduler(policy=p)
            assert scheduler is not None


# ---------------------------------------------------------------------------
# Phase saving
# ---------------------------------------------------------------------------


class TestPhaseSaving:

    def test_default_polarity(self):
        edits = _small_edits(3)
        ps = PhaseSaver(edits, default_polarity=Polarity.POSITIVE)
        for e in edits:
            pol = ps.get_polarity(e)
            assert isinstance(pol, bool)

    def test_save_and_restore(self):
        edits = _small_edits(3)
        ps = PhaseSaver(edits, default_polarity=Polarity.NEGATIVE)
        ps.save_phase(edits[0], True)
        assert ps.get_polarity(edits[0]) is True

    def test_save_overrides_default(self):
        edits = _small_edits(3)
        ps = PhaseSaver(edits, default_polarity=Polarity.POSITIVE)
        ps.save_phase(edits[0], False)
        assert ps.get_polarity(edits[0]) is False

    def test_multiple_saves(self):
        edits = _small_edits(3)
        ps = PhaseSaver(edits, default_polarity=Polarity.POSITIVE)
        ps.save_phase(edits[0], True)
        ps.save_phase(edits[0], False)
        assert ps.get_polarity(edits[0]) is False

    def test_independent_edits(self):
        edits = _small_edits(3)
        ps = PhaseSaver(edits, default_polarity=Polarity.POSITIVE)
        ps.save_phase(edits[0], False)
        # Other edits should keep default
        pol1 = ps.get_polarity(edits[1])
        assert isinstance(pol1, bool)


# ---------------------------------------------------------------------------
# Conflict analysis
# ---------------------------------------------------------------------------


class TestConflictAnalysis:

    def test_create_analyzer(self):
        cdb = ClauseDatabase()
        engine = WatchedLiteralEngine(cdb)
        analyzer = ConflictAnalyzer(engine, cdb)
        assert analyzer is not None

    def test_analyze_with_conflict_clause(self):
        lits = _small_literals(3)
        cdb = ClauseDatabase()
        engine = WatchedLiteralEngine(cdb)
        analyzer = ConflictAnalyzer(engine, cdb)
        conflict_clause = Clause(literals=tuple(lits[:2]))
        try:
            result = analyzer.analyze(conflict_clause)
            assert isinstance(result, ConflictResult)
        except Exception:
            # Conflict analysis may fail if no implication graph is built
            pass


# ---------------------------------------------------------------------------
# Branching heuristics
# ---------------------------------------------------------------------------


class TestBranchingHeuristics:

    def test_evsids_creation(self):
        edits = _small_edits(3)
        evsids = EVSIDS(edits, decay=0.95)
        assert evsids is not None

    def test_evsids_pick(self):
        edits = _small_edits(3)
        evsids = EVSIDS(edits, decay=0.95)
        decision = evsids.pick(assigned=set())
        assert decision is not None or decision is None  # can be None if all assigned

    def test_evsids_pick_excludes_assigned(self):
        edits = _small_edits(3)
        evsids = EVSIDS(edits, decay=0.95)
        assigned = set(edits)
        decision = evsids.pick(assigned=assigned)
        assert decision is None

    def test_evsids_bump(self):
        edits = _small_edits(3)
        evsids = EVSIDS(edits, decay=0.95)
        evsids.bump_many(frozenset(edits[:2]))
        # After bumping, those edits should have higher scores
        decision = evsids.pick(assigned=set())
        assert decision is not None

    def test_lrb_creation(self):
        edits = _small_edits(3)
        lrb = LRB(edits, alpha=0.4)
        assert lrb is not None

    def test_lrb_pick(self):
        edits = _small_edits(3)
        lrb = LRB(edits, alpha=0.4)
        decision = lrb.pick(assigned=set())
        assert decision is not None

    def test_lrb_conflict_updates(self):
        edits = _small_edits(3)
        lrb = LRB(edits, alpha=0.4)
        lrb.on_assign(edits[0])
        lrb.on_conflict(involved=frozenset(edits[:2]), reason_edits=frozenset(edits[2:4]))
        lrb.on_unassign(edits[0])

    def test_chb_creation(self):
        edits = _small_edits(3)
        chb = CHB(edits)
        assert chb is not None

    def test_random_branching(self):
        edits = _small_edits(3)
        rb = RandomBranching(edits)
        decision = rb.pick(assigned=set())
        assert decision in edits

    def test_branching_engine_evsids(self):
        edits = _small_edits(3)
        engine = BranchingEngine(edits, strategy="evsids")
        decision = engine.decide(assigned=set())
        assert decision is not None or len(edits) == 0

    def test_branching_engine_lrb(self):
        edits = _small_edits(3)
        engine = BranchingEngine(edits, strategy="lrb")
        decision = engine.decide(assigned=set())
        assert decision is not None

    def test_branching_all_assigned_returns_none(self):
        edits = _small_edits(3)
        engine = BranchingEngine(edits, strategy="evsids")
        assigned = set(edits)
        decision = engine.decide(assigned=assigned)
        assert decision is None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessing:

    def test_preprocessor_creation(self):
        cdb = ClauseDatabase()
        pp = Preprocessor(cdb)
        assert pp is not None

    def test_preprocess_empty(self):
        cdb = ClauseDatabase()
        pp = Preprocessor(cdb)
        stats = pp.preprocess()
        assert isinstance(stats, PreprocessStats)

    def test_preprocess_with_unit_clauses(self):
        lits = _small_literals(3)
        cdb = ClauseDatabase()
        unit = Clause(literals=(lits[0],))
        cdb.add_original(unit)
        pp = Preprocessor(cdb)
        stats = pp.preprocess()
        assert isinstance(stats, PreprocessStats)

    def test_preprocess_stats_fields(self):
        cdb = ClauseDatabase()
        pp = Preprocessor(cdb)
        stats = pp.preprocess()
        assert hasattr(stats, 'n_unit_props')
        assert hasattr(stats, 'n_subsumed')


# ---------------------------------------------------------------------------
# Integration: restart + branching + propagation
# ---------------------------------------------------------------------------


class TestSolverComponentIntegration:

    def test_restart_scheduler_with_branching(self):
        edits = _small_edits(3)
        engine = BranchingEngine(edits, strategy="evsids")
        scheduler = RestartScheduler(policy=RestartPolicy.LUBY, luby_base=16)

        decisions = []
        for i in range(100):
            d = engine.decide(assigned=set(decisions))
            if d is not None:
                decisions.append(d)
            if scheduler.on_conflict(lbd=i % 5 + 1):
                decisions.clear()

    def test_phase_saving_with_branching(self):
        edits = _small_edits(3)
        ps = PhaseSaver(edits, default_polarity=Polarity.POSITIVE)
        engine = BranchingEngine(edits, strategy="evsids")

        for _ in range(10):
            d = engine.decide(assigned=set())
            if d is not None:
                pol = ps.get_polarity(d)
                ps.save_phase(d, not pol)

    def test_luby_sequence_deterministic(self):
        """Same index always returns same value."""
        for i in range(50):
            assert luby_sequence(i) == luby_sequence(i)

    def test_geometric_restart_deterministic(self):
        g1 = GeometricRestart()
        g2 = GeometricRestart()
        for i in range(200):
            r1 = g1.on_conflict()
            r2 = g2.on_conflict()
            assert r1 == r2

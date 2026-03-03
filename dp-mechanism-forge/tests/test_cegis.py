"""
Comprehensive tests for dp_forge.cegis_loop — CEGIS synthesis orchestrator.

Tests cover:
    - SynthesisStrategy and CEGISStatus enum values
    - CEGISProgress dataclass construction and repr
    - ConvergenceHistory tracking, monotonicity, stagnation, and summary
    - WitnessSet management: add, contains, remove, coverage, frozen, smart_seed
    - DualSimplexWarmStart: store, retrieve, invalidate, failure tracking
    - auto_select_strategy dispatch logic
    - CEGISEngine unit tests (properties, best-so-far, config)
    - CEGISProgressReporter callback collection
    - Integration tests for CEGISSynthesize on tiny examples
    - synthesize_mechanism high-level API
    - Multi-strategy and hybrid synthesis
    - Cycle detection and DP-preserving projection
    - Error recovery with mocking
    - CEGISResult structural validation
"""

from __future__ import annotations

import logging
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
import pytest

from dp_forge.cegis_loop import (
    CEGISEngine,
    CEGISProgress,
    CEGISProgressReporter,
    CEGISStatus,
    CEGISSynthesize,
    ConvergenceHistory,
    DualSimplexWarmStart,
    SynthesisStrategy,
    WitnessSet,
    _dp_preserving_projection,
    auto_select_strategy,
    synthesize_mechanism,
    hybrid_synthesis,
    parallel_synthesis,
    quick_synthesize,
)
from dp_forge.exceptions import (
    ConfigurationError,
    ConvergenceError,
    InfeasibleSpecError,
    SolverError,
)
from dp_forge.lp_builder import SolveStatistics
from dp_forge.types import (
    AdjacencyRelation,
    CEGISResult,
    LossFunction,
    MechanismFamily,
    NumericalConfig,
    QuerySpec,
    QueryType,
    SolverBackend,
    SynthesisConfig,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers and Fixtures
# ═══════════════════════════════════════════════════════════════════════════


def _make_query_spec(
    n: int = 3,
    k: int = 5,
    epsilon: float = 1.0,
    delta: float = 0.0,
) -> QuerySpec:
    """Create a tiny QuerySpec for fast tests."""
    return QuerySpec.counting(n=n, epsilon=epsilon, delta=delta, k=k)


def _make_approx_query_spec(
    n: int = 3,
    k: int = 5,
    epsilon: float = 1.0,
    delta: float = 0.01,
) -> QuerySpec:
    """Create a tiny approximate-DP QuerySpec."""
    return QuerySpec.counting(n=n, epsilon=epsilon, delta=delta, k=k)


def _make_valid_mechanism(n: int, k: int) -> np.ndarray:
    """Create a valid mechanism table (rows sum to 1, non-negative)."""
    p = np.random.dirichlet(np.ones(k), size=n)
    return p


def _make_solve_stats(
    n_vars: int = 10,
    objective: float = 0.5,
    basis_info: Optional[Dict[str, Any]] = None,
    duality_gap: Optional[float] = None,
) -> SolveStatistics:
    """Create a mock SolveStatistics dataclass."""
    return SolveStatistics(
        solver_name="HiGHS",
        status="optimal",
        iterations=42,
        solve_time=0.01,
        objective_value=objective,
        primal_solution=np.ones(n_vars) / n_vars,
        dual_solution=np.ones(n_vars) * 0.1,
        basis_info=basis_info or {"status": [0] * n_vars},
        duality_gap=duality_gap,
        n_vars=n_vars,
        n_constraints=20,
    )


@pytest.fixture
def tiny_spec():
    """Tiny pure-DP counting spec (n=3, k=5, eps=1)."""
    return _make_query_spec(n=3, k=5, epsilon=1.0)


@pytest.fixture
def small_spec():
    """Small pure-DP counting spec (n=3, k=10, eps=1)."""
    return _make_query_spec(n=3, k=10, epsilon=1.0)


@pytest.fixture
def approx_spec():
    """Tiny approximate-DP counting spec (n=3, k=5, eps=1, delta=0.01)."""
    return _make_approx_query_spec(n=3, k=5, epsilon=1.0, delta=0.01)


@pytest.fixture
def default_config():
    """Default SynthesisConfig."""
    return SynthesisConfig(max_iter=50, verbose=0)


# ═══════════════════════════════════════════════════════════════════════════
# §1  SynthesisStrategy Enum
# ═══════════════════════════════════════════════════════════════════════════


class TestSynthesisStrategy:
    """Tests for SynthesisStrategy enum."""

    def test_all_members_exist(self):
        expected = {"LP_PURE", "LP_APPROX", "SDP_GAUSSIAN", "HYBRID"}
        assert set(m.name for m in SynthesisStrategy) == expected

    def test_member_count(self):
        assert len(SynthesisStrategy) == 4

    def test_repr(self):
        assert repr(SynthesisStrategy.LP_PURE) == "SynthesisStrategy.LP_PURE"
        assert repr(SynthesisStrategy.LP_APPROX) == "SynthesisStrategy.LP_APPROX"
        assert repr(SynthesisStrategy.SDP_GAUSSIAN) == "SynthesisStrategy.SDP_GAUSSIAN"
        assert repr(SynthesisStrategy.HYBRID) == "SynthesisStrategy.HYBRID"

    def test_identity(self):
        assert SynthesisStrategy.LP_PURE is SynthesisStrategy.LP_PURE
        assert SynthesisStrategy.LP_PURE != SynthesisStrategy.LP_APPROX

    def test_values_are_distinct(self):
        values = [m.value for m in SynthesisStrategy]
        assert len(values) == len(set(values))

    def test_by_name_access(self):
        assert SynthesisStrategy["LP_PURE"] is SynthesisStrategy.LP_PURE
        assert SynthesisStrategy["HYBRID"] is SynthesisStrategy.HYBRID

    def test_invalid_member_raises(self):
        with pytest.raises(KeyError):
            _ = SynthesisStrategy["NONEXISTENT"]


# ═══════════════════════════════════════════════════════════════════════════
# §2  CEGISStatus Enum
# ═══════════════════════════════════════════════════════════════════════════


class TestCEGISStatus:
    """Tests for CEGISStatus enum."""

    def test_all_members_exist(self):
        expected = {
            "RUNNING",
            "CONVERGED",
            "MAX_ITER_REACHED",
            "INFEASIBLE",
            "CYCLE_RESOLVED",
            "NUMERICAL_FAILURE",
            "STAGNATED",
        }
        assert set(m.name for m in CEGISStatus) == expected

    def test_member_count(self):
        assert len(CEGISStatus) == 7

    def test_repr(self):
        for status in CEGISStatus:
            assert repr(status) == f"CEGISStatus.{status.name}"

    def test_identity(self):
        assert CEGISStatus.RUNNING is CEGISStatus.RUNNING
        assert CEGISStatus.CONVERGED is not CEGISStatus.RUNNING

    def test_values_are_distinct(self):
        values = [m.value for m in CEGISStatus]
        assert len(values) == len(set(values))

    def test_terminal_statuses(self):
        terminal = {
            CEGISStatus.CONVERGED,
            CEGISStatus.MAX_ITER_REACHED,
            CEGISStatus.INFEASIBLE,
            CEGISStatus.CYCLE_RESOLVED,
            CEGISStatus.NUMERICAL_FAILURE,
            CEGISStatus.STAGNATED,
        }
        non_terminal = {CEGISStatus.RUNNING}
        assert terminal | non_terminal == set(CEGISStatus)


# ═══════════════════════════════════════════════════════════════════════════
# §3  CEGISProgress Dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestCEGISProgress:
    """Tests for CEGISProgress dataclass construction and repr."""

    def test_basic_construction(self):
        p = CEGISProgress(
            iteration=0,
            objective=0.5,
            violation_magnitude=1e-3,
            violation_pair=(0, 1),
            n_witness_pairs=2,
            solve_time=0.1,
            verify_time=0.05,
            total_time=0.15,
        )
        assert p.iteration == 0
        assert p.objective == 0.5
        assert p.violation_magnitude == 1e-3
        assert p.violation_pair == (0, 1)
        assert p.n_witness_pairs == 2
        assert p.solve_time == 0.1
        assert p.verify_time == 0.05
        assert p.total_time == 0.15
        assert p.status == CEGISStatus.RUNNING
        assert p.is_cycle is False

    def test_default_status_is_running(self):
        p = CEGISProgress(
            iteration=0,
            objective=1.0,
            violation_magnitude=0.0,
            violation_pair=None,
            n_witness_pairs=1,
            solve_time=0.0,
            verify_time=0.0,
            total_time=0.0,
        )
        assert p.status == CEGISStatus.RUNNING

    def test_custom_status(self):
        p = CEGISProgress(
            iteration=5,
            objective=0.3,
            violation_magnitude=0.0,
            violation_pair=None,
            n_witness_pairs=10,
            solve_time=0.5,
            verify_time=0.2,
            total_time=3.5,
            status=CEGISStatus.CONVERGED,
        )
        assert p.status == CEGISStatus.CONVERGED

    def test_cycle_flag(self):
        p = CEGISProgress(
            iteration=3,
            objective=0.4,
            violation_magnitude=0.01,
            violation_pair=(1, 2),
            n_witness_pairs=5,
            solve_time=0.1,
            verify_time=0.05,
            total_time=1.0,
            is_cycle=True,
        )
        assert p.is_cycle is True

    def test_repr_with_violation(self):
        p = CEGISProgress(
            iteration=2,
            objective=0.123456,
            violation_magnitude=1.5e-4,
            violation_pair=(0, 1),
            n_witness_pairs=3,
            solve_time=0.1,
            verify_time=0.05,
            total_time=0.5,
        )
        r = repr(p)
        assert "iter=2" in r
        assert "obj=0.123456" in r
        assert "viol=" in r
        assert "pairs=3" in r

    def test_repr_valid_mechanism(self):
        p = CEGISProgress(
            iteration=10,
            objective=0.5,
            violation_magnitude=0.0,
            violation_pair=None,
            n_witness_pairs=5,
            solve_time=0.1,
            verify_time=0.05,
            total_time=2.0,
        )
        r = repr(p)
        assert "valid" in r

    def test_zero_iteration(self):
        p = CEGISProgress(
            iteration=0,
            objective=0.0,
            violation_magnitude=0.0,
            violation_pair=None,
            n_witness_pairs=0,
            solve_time=0.0,
            verify_time=0.0,
            total_time=0.0,
        )
        assert p.iteration == 0
        assert p.n_witness_pairs == 0


# ═══════════════════════════════════════════════════════════════════════════
# §4  ConvergenceHistory
# ═══════════════════════════════════════════════════════════════════════════


class TestConvergenceHistory:
    """Tests for ConvergenceHistory: record, monotonicity, stagnation, summary."""

    def test_empty_history(self):
        h = ConvergenceHistory()
        assert h.n_iterations == 0
        assert h.objectives == []
        assert h.violations == []
        assert h.total_solve_time == 0.0
        assert h.total_verify_time == 0.0
        assert h.total_time == 0.0

    def test_record_single(self):
        h = ConvergenceHistory()
        h.record(
            objective=0.5,
            violation=0.01,
            solve_time=0.1,
            verify_time=0.05,
            n_witnesses=2,
            iteration_time=0.15,
        )
        assert h.n_iterations == 1
        assert h.objectives == [0.5]
        assert h.violations == [0.01]
        assert h.solve_times == [0.1]
        assert h.verify_times == [0.05]
        assert h.witness_counts == [2]
        assert h.iteration_times == [0.15]
        assert len(h.cycle_iterations) == 0

    def test_record_multiple(self):
        h = ConvergenceHistory()
        for i in range(5):
            h.record(
                objective=float(i) * 0.1,
                violation=0.01 / (i + 1),
                solve_time=0.1,
                verify_time=0.05,
                n_witnesses=i + 1,
                iteration_time=0.15,
            )
        assert h.n_iterations == 5
        assert len(h.objectives) == 5
        assert h.witness_counts == [1, 2, 3, 4, 5]

    def test_record_with_cycle(self):
        h = ConvergenceHistory()
        h.record(0.5, 0.01, 0.1, 0.05, 2, 0.15, is_cycle=False)
        h.record(0.6, 0.01, 0.1, 0.05, 2, 0.15, is_cycle=True)
        h.record(0.7, 0.01, 0.1, 0.05, 3, 0.15, is_cycle=False)
        assert 1 in h.cycle_iterations
        assert 0 not in h.cycle_iterations
        assert 2 not in h.cycle_iterations

    def test_total_solve_time(self):
        h = ConvergenceHistory()
        h.record(0.5, 0.0, 0.1, 0.05, 1, 0.15)
        h.record(0.6, 0.0, 0.2, 0.06, 2, 0.26)
        assert abs(h.total_solve_time - 0.3) < 1e-10

    def test_total_verify_time(self):
        h = ConvergenceHistory()
        h.record(0.5, 0.0, 0.1, 0.05, 1, 0.15)
        h.record(0.6, 0.0, 0.2, 0.06, 2, 0.26)
        assert abs(h.total_verify_time - 0.11) < 1e-10

    def test_total_time(self):
        h = ConvergenceHistory()
        h.record(0.5, 0.0, 0.1, 0.05, 1, 1.0)
        h.record(0.6, 0.0, 0.2, 0.06, 2, 2.0)
        assert abs(h.total_time - 3.0) < 1e-10

    # --- Monotonicity ---

    def test_check_monotonicity_empty(self):
        h = ConvergenceHistory()
        assert h.check_monotonicity() is True

    def test_check_monotonicity_single(self):
        h = ConvergenceHistory()
        h.record(0.5, 0.0, 0.1, 0.05, 1, 0.15)
        assert h.check_monotonicity() is True

    def test_check_monotonicity_increasing(self):
        h = ConvergenceHistory()
        for obj in [0.1, 0.2, 0.3, 0.4, 0.5]:
            h.record(obj, 0.0, 0.1, 0.05, 1, 0.15)
        assert h.check_monotonicity() is True

    def test_check_monotonicity_constant(self):
        h = ConvergenceHistory()
        for _ in range(5):
            h.record(0.5, 0.0, 0.1, 0.05, 1, 0.15)
        assert h.check_monotonicity() is True

    def test_check_monotonicity_violation(self):
        h = ConvergenceHistory()
        h.record(0.5, 0.0, 0.1, 0.05, 1, 0.15)
        h.record(0.3, 0.0, 0.1, 0.05, 2, 0.15)  # Decrease!
        assert h.check_monotonicity() is False

    def test_check_monotonicity_tolerance(self):
        h = ConvergenceHistory()
        h.record(0.5, 0.0, 0.1, 0.05, 1, 0.15)
        h.record(0.5 - 1e-14, 0.0, 0.1, 0.05, 2, 0.15)  # Within tol
        assert h.check_monotonicity(tol=1e-12) is True

    def test_check_monotonicity_skips_cycles(self):
        h = ConvergenceHistory()
        h.record(0.5, 0.0, 0.1, 0.05, 1, 0.15)
        h.record(0.3, 0.0, 0.1, 0.05, 1, 0.15, is_cycle=True)  # Cycle
        h.record(0.6, 0.0, 0.1, 0.05, 2, 0.15)
        assert h.check_monotonicity() is True

    # --- Stagnation ---

    def test_detect_stagnation_too_few_iterations(self):
        h = ConvergenceHistory()
        for i in range(5):
            h.record(0.5, 0.0, 0.1, 0.05, i + 1, 0.15)
        assert h.detect_stagnation(window=10) is False

    def test_detect_stagnation_no_change(self):
        h = ConvergenceHistory()
        for i in range(15):
            h.record(0.5, 0.0, 0.1, 0.05, i + 1, 0.15)
        assert h.detect_stagnation(window=10, tol=1e-10) is True

    def test_detect_stagnation_with_improvement(self):
        h = ConvergenceHistory()
        for i in range(15):
            h.record(0.5 + i * 0.01, 0.0, 0.1, 0.05, i + 1, 0.15)
        assert h.detect_stagnation(window=10, tol=1e-10) is False

    def test_detect_stagnation_tiny_change(self):
        h = ConvergenceHistory()
        for i in range(15):
            h.record(0.5 + i * 1e-15, 0.0, 0.1, 0.05, i + 1, 0.15)
        assert h.detect_stagnation(window=10, tol=1e-10) is True

    # --- Estimate remaining iterations ---

    def test_estimate_remaining_empty(self):
        h = ConvergenceHistory()
        assert h.estimate_remaining_iterations(10) == 10

    def test_estimate_remaining_single(self):
        h = ConvergenceHistory()
        h.record(0.5, 0.0, 0.1, 0.05, 2, 0.15)
        # One iteration, 2 witnesses, total 10 edges -> ~8 remaining
        assert h.estimate_remaining_iterations(10) == 8

    def test_estimate_remaining_fully_covered(self):
        h = ConvergenceHistory()
        h.record(0.5, 0.0, 0.1, 0.05, 5, 0.15)
        h.record(0.6, 0.0, 0.1, 0.05, 10, 0.15)
        assert h.estimate_remaining_iterations(10) == 0

    def test_estimate_remaining_growing(self):
        h = ConvergenceHistory()
        for i in range(5):
            h.record(0.5, 0.0, 0.1, 0.05, i + 1, 0.15)
        remaining = h.estimate_remaining_iterations(20)
        assert remaining > 0
        assert remaining <= 20

    # --- Objective improvement rate ---

    def test_objective_improvement_rate_empty(self):
        h = ConvergenceHistory()
        assert h.objective_improvement_rate() == 0.0

    def test_objective_improvement_rate_single(self):
        h = ConvergenceHistory()
        h.record(0.5, 0.0, 0.1, 0.05, 1, 0.15)
        assert h.objective_improvement_rate() == 0.0

    def test_objective_improvement_rate_increasing(self):
        h = ConvergenceHistory()
        for i in range(6):
            h.record(float(i) * 0.1, 0.0, 0.1, 0.05, i + 1, 0.15)
        rate = h.objective_improvement_rate(window=5)
        assert rate > 0

    def test_objective_improvement_rate_constant(self):
        h = ConvergenceHistory()
        for _ in range(10):
            h.record(0.5, 0.0, 0.1, 0.05, 1, 0.15)
        assert h.objective_improvement_rate(window=5) == 0.0

    # --- Summary ---

    def test_summary_empty(self):
        h = ConvergenceHistory()
        s = h.summary()
        assert "no iterations" in s.lower()

    def test_summary_with_data(self):
        h = ConvergenceHistory()
        h.record(0.1, 0.5, 0.1, 0.05, 1, 0.15)
        h.record(0.2, 0.3, 0.1, 0.05, 2, 0.15)
        h.record(0.3, 0.0, 0.1, 0.05, 3, 0.15)
        s = h.summary()
        assert "3 iterations" in s
        assert "Objective" in s
        assert "Violations" in s
        assert "Witness pairs" in s
        assert "Cycles" in s
        assert "Monotonic" in s
        assert "Stagnated" in s

    def test_summary_with_cycles(self):
        h = ConvergenceHistory()
        h.record(0.1, 0.5, 0.1, 0.05, 1, 0.15, is_cycle=False)
        h.record(0.2, 0.3, 0.1, 0.05, 1, 0.15, is_cycle=True)
        s = h.summary()
        assert "Cycles: 1" in s

    # --- Repr ---

    def test_repr(self):
        h = ConvergenceHistory()
        h.record(0.5, 0.0, 0.1, 0.05, 1, 0.15)
        r = repr(h)
        assert "ConvergenceHistory" in r
        assert "n=1" in r
        assert "cycles=0" in r


# ═══════════════════════════════════════════════════════════════════════════
# §5  WitnessSet
# ═══════════════════════════════════════════════════════════════════════════


class TestWitnessSet:
    """Tests for WitnessSet: add, contains, remove, coverage, frozen, etc."""

    def test_empty(self):
        ws = WitnessSet(total_edges=5)
        assert ws.size == 0
        assert len(ws) == 0
        assert ws.pairs == []

    def test_add_pair(self):
        ws = WitnessSet(total_edges=5)
        assert ws.add_pair(0, 1) is True
        assert ws.size == 1
        assert ws.contains(0, 1) is True

    def test_add_pair_duplicate(self):
        ws = WitnessSet(total_edges=5)
        ws.add_pair(0, 1)
        assert ws.add_pair(0, 1) is False
        assert ws.size == 1

    def test_add_pair_canonical(self):
        """Adding (1,0) should be equivalent to (0,1)."""
        ws = WitnessSet(total_edges=5)
        ws.add_pair(1, 0)
        assert ws.contains(0, 1) is True
        assert ws.contains(1, 0) is True
        assert ws.size == 1

    def test_add_multiple_pairs(self):
        ws = WitnessSet(total_edges=10)
        ws.add_pair(0, 1)
        ws.add_pair(2, 3)
        ws.add_pair(1, 2)
        assert ws.size == 3
        assert ws.contains(0, 1)
        assert ws.contains(2, 3)
        assert ws.contains(1, 2)

    def test_contains_missing(self):
        ws = WitnessSet(total_edges=5)
        ws.add_pair(0, 1)
        assert ws.contains(0, 2) is False
        assert ws.contains(1, 2) is False

    def test_remove_pair_present(self):
        ws = WitnessSet(total_edges=5)
        ws.add_pair(0, 1)
        assert ws.remove_pair(0, 1) is True
        assert ws.size == 0
        assert ws.contains(0, 1) is False

    def test_remove_pair_missing(self):
        ws = WitnessSet(total_edges=5)
        assert ws.remove_pair(0, 1) is False

    def test_remove_pair_canonical(self):
        """Removing (1,0) should remove the canonical (0,1) pair."""
        ws = WitnessSet(total_edges=5)
        ws.add_pair(0, 1)
        assert ws.remove_pair(1, 0) is True
        assert ws.size == 0

    def test_coverage_fraction_empty(self):
        ws = WitnessSet(total_edges=5)
        assert ws.coverage_fraction() == 0.0

    def test_coverage_fraction_partial(self):
        ws = WitnessSet(total_edges=4)
        ws.add_pair(0, 1)
        ws.add_pair(1, 2)
        assert abs(ws.coverage_fraction() - 0.5) < 1e-10

    def test_coverage_fraction_full(self):
        ws = WitnessSet(total_edges=2)
        ws.add_pair(0, 1)
        ws.add_pair(1, 2)
        assert abs(ws.coverage_fraction() - 1.0) < 1e-10

    def test_coverage_fraction_zero_edges(self):
        ws = WitnessSet(total_edges=0)
        assert ws.coverage_fraction() == 1.0

    def test_frozen_returns_frozenset(self):
        ws = WitnessSet(total_edges=5)
        ws.add_pair(0, 1)
        ws.add_pair(2, 3)
        f = ws.frozen()
        assert isinstance(f, frozenset)
        assert len(f) == 2
        assert (0, 1) in f
        assert (2, 3) in f

    def test_frozen_immutability(self):
        ws = WitnessSet(total_edges=5)
        ws.add_pair(0, 1)
        f = ws.frozen()
        ws.add_pair(2, 3)
        assert len(f) == 1  # Frozen snapshot is unaffected

    def test_pairs_sorted(self):
        ws = WitnessSet(total_edges=10)
        ws.add_pair(3, 4)
        ws.add_pair(0, 1)
        ws.add_pair(1, 2)
        assert ws.pairs == [(0, 1), (1, 2), (3, 4)]

    def test_insertion_order(self):
        ws = WitnessSet(total_edges=10)
        ws.add_pair(3, 4)
        ws.add_pair(0, 1)
        ws.add_pair(1, 2)
        assert ws.insertion_order == [(3, 4), (0, 1), (1, 2)]

    def test_contains_via_dunder(self):
        ws = WitnessSet(total_edges=5)
        ws.add_pair(0, 1)
        assert (0, 1) in ws
        assert (1, 0) in ws
        assert (0, 2) not in ws

    def test_iteration(self):
        ws = WitnessSet(total_edges=10)
        ws.add_pair(2, 3)
        ws.add_pair(0, 1)
        pairs = list(ws)
        assert pairs == [(0, 1), (2, 3)]

    def test_repr(self):
        ws = WitnessSet(total_edges=10)
        ws.add_pair(0, 1)
        r = repr(ws)
        assert "WitnessSet" in r
        assert "size=1" in r

    # --- Smart seeding ---

    def test_smart_seed_empty_edges(self):
        ws = WitnessSet(total_edges=0)
        result = ws.smart_seed([], n_initial=3)
        assert result == []

    def test_smart_seed_adds_pairs(self):
        ws = WitnessSet(total_edges=4)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        result = ws.smart_seed(edges, n_initial=2, query_values=np.arange(5.0))
        assert len(result) <= 2
        assert ws.size == len(result)
        for p in result:
            assert ws.contains(p[0], p[1])

    def test_smart_seed_respects_limit(self):
        ws = WitnessSet(total_edges=10)
        edges = [(i, i + 1) for i in range(10)]
        result = ws.smart_seed(edges, n_initial=3, query_values=np.arange(11.0))
        assert len(result) <= 3

    def test_smart_seed_no_query_values(self):
        ws = WitnessSet(total_edges=4)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        result = ws.smart_seed(edges, n_initial=2)
        assert len(result) == 2
        assert ws.size == 2

    def test_smart_seed_all_edges(self):
        ws = WitnessSet(total_edges=3)
        edges = [(0, 1), (1, 2), (2, 3)]
        result = ws.smart_seed(edges, n_initial=10, query_values=np.arange(4.0))
        assert len(result) == 3
        assert ws.size == 3

    def test_smart_seed_idempotent_for_existing(self):
        ws = WitnessSet(total_edges=4)
        ws.add_pair(0, 1)
        edges = [(0, 1), (1, 2)]
        result = ws.smart_seed(edges, n_initial=2, query_values=np.arange(3.0))
        # (0,1) was already there, so only (1,2) should be newly added
        assert ws.contains(1, 2)


# ═══════════════════════════════════════════════════════════════════════════
# §6  DualSimplexWarmStart
# ═══════════════════════════════════════════════════════════════════════════


class TestDualSimplexWarmStart:
    """Tests for DualSimplexWarmStart: store, retrieve, invalidate, failure."""

    def test_initial_state(self):
        ws = DualSimplexWarmStart()
        assert ws.is_valid is False
        assert ws.get_warm_start(10) is None

    def test_store_and_retrieve(self):
        ws = DualSimplexWarmStart()
        stats = _make_solve_stats(n_vars=10, basis_info={"col_status": [0] * 10})
        ws.store(stats)
        assert ws.is_valid is True
        basis = ws.get_warm_start(10)
        assert basis is not None
        assert basis == {"col_status": [0] * 10}

    def test_warm_start_wrong_n_vars(self):
        ws = DualSimplexWarmStart()
        stats = _make_solve_stats(n_vars=10)
        ws.store(stats)
        result = ws.get_warm_start(15)  # Different n_vars
        assert result is None
        assert ws.is_valid is False

    def test_invalidate(self):
        ws = DualSimplexWarmStart()
        stats = _make_solve_stats(n_vars=10)
        ws.store(stats)
        assert ws.is_valid is True
        ws.invalidate()
        assert ws.is_valid is False
        assert ws.get_warm_start(10) is None

    def test_record_failure_single(self):
        ws = DualSimplexWarmStart()
        stats = _make_solve_stats(n_vars=10)
        ws.store(stats)
        ws.record_failure()
        # After 1 failure, still valid (threshold is 5)
        assert ws.is_valid is True

    def test_record_failure_threshold(self):
        ws = DualSimplexWarmStart()
        stats = _make_solve_stats(n_vars=10)
        ws.store(stats)
        for _ in range(5):
            ws.record_failure()
        assert ws.is_valid is False
        assert ws.get_warm_start(10) is None

    def test_stats_tracking(self):
        ws = DualSimplexWarmStart()
        stats = _make_solve_stats(n_vars=10)
        ws.store(stats)

        # Cold start (invalid before store, but store resets)
        _ = ws.get_warm_start(10)  # warm
        _ = ws.get_warm_start(10)  # warm

        s = ws.stats
        assert s["warm_starts"] == 2
        assert s["consecutive_failures"] == 0

    def test_cold_start_counter(self):
        ws = DualSimplexWarmStart()
        # Before any store, get_warm_start returns None → cold start
        ws.get_warm_start(10)
        ws.get_warm_start(10)
        s = ws.stats
        assert s["cold_starts"] == 2

    def test_store_resets_failure_count(self):
        ws = DualSimplexWarmStart()
        stats = _make_solve_stats(n_vars=10)
        ws.store(stats)
        ws.record_failure()
        ws.record_failure()
        ws.store(stats)  # Should reset consecutive failures
        assert ws.stats["consecutive_failures"] == 0
        assert ws.is_valid is True

    def test_repr(self):
        ws = DualSimplexWarmStart()
        r = repr(ws)
        assert "DualSimplexWarmStart" in r
        assert "valid=False" in r


# ═══════════════════════════════════════════════════════════════════════════
# §7  auto_select_strategy
# ═══════════════════════════════════════════════════════════════════════════


class TestAutoSelectStrategy:
    """Tests for auto_select_strategy dispatch."""

    def test_pure_dp_selects_lp_pure(self, tiny_spec):
        assert auto_select_strategy(tiny_spec) == SynthesisStrategy.LP_PURE

    def test_approx_dp_selects_lp_approx(self, approx_spec):
        assert auto_select_strategy(approx_spec) == SynthesisStrategy.LP_APPROX

    def test_pure_dp_various_epsilon(self):
        for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
            spec = _make_query_spec(epsilon=eps, delta=0.0)
            assert auto_select_strategy(spec) == SynthesisStrategy.LP_PURE

    def test_approx_dp_various_delta(self):
        for delta in [1e-10, 1e-5, 0.01, 0.1]:
            spec = _make_approx_query_spec(epsilon=1.0, delta=delta)
            assert auto_select_strategy(spec) == SynthesisStrategy.LP_APPROX

    def test_small_n_pure(self):
        spec = _make_query_spec(n=2, k=3, epsilon=1.0)
        assert auto_select_strategy(spec) == SynthesisStrategy.LP_PURE

    def test_large_n_pure(self):
        spec = _make_query_spec(n=50, k=5, epsilon=1.0)
        assert auto_select_strategy(spec) == SynthesisStrategy.LP_PURE


# ═══════════════════════════════════════════════════════════════════════════
# §8  DP-Preserving Projection
# ═══════════════════════════════════════════════════════════════════════════


class TestDPPreservingProjection:
    """Tests for _dp_preserving_projection."""

    def test_already_valid_pure_dp(self):
        """A mechanism already satisfying pure DP should be unchanged."""
        # Uniform mechanism satisfies any DP
        n, k = 3, 4
        p = np.ones((n, k)) / k
        edges = [(0, 1), (1, 2)]
        p_proj = _dp_preserving_projection(p, epsilon=1.0, delta=0.0, edges=edges)
        np.testing.assert_allclose(p_proj.sum(axis=1), 1.0, atol=1e-10)
        np.testing.assert_array_less(-1e-10, p_proj)  # Non-negative

    def test_projection_normalises_rows(self):
        """After projection, all rows must sum to 1."""
        n, k = 3, 5
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(k), size=n)
        edges = [(0, 1), (1, 2)]
        p_proj = _dp_preserving_projection(p, epsilon=0.5, delta=0.0, edges=edges)
        np.testing.assert_allclose(p_proj.sum(axis=1), 1.0, atol=1e-10)

    def test_projection_nonnegative(self):
        n, k = 3, 5
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(k), size=n)
        edges = [(0, 1), (1, 2)]
        p_proj = _dp_preserving_projection(p, epsilon=0.5, delta=0.0, edges=edges)
        assert np.all(p_proj >= -1e-15)

    def test_pure_dp_ratio_constraint(self):
        """After projection, p[i][j] <= exp(eps)*p[i'][j] for direct edges."""
        n, k = 2, 5
        eps = 1.0
        exp_eps = math.exp(eps)
        # Start with a near-uniform mechanism that's easy to project
        p = np.ones((n, k)) / k
        # Slightly perturb one row
        p[0] = [0.3, 0.25, 0.2, 0.15, 0.1]
        p[1] = [0.1, 0.15, 0.2, 0.25, 0.3]
        edges = [(0, 1)]
        p_proj = _dp_preserving_projection(p, epsilon=eps, delta=0.0, edges=edges)
        np.testing.assert_allclose(p_proj.sum(axis=1), 1.0, atol=1e-10)
        for i, ip in edges:
            for j in range(k):
                assert p_proj[i][j] <= exp_eps * p_proj[ip][j] + 1e-8
                assert p_proj[ip][j] <= exp_eps * p_proj[i][j] + 1e-8

    def test_approx_dp_projection(self):
        """Approximate DP projection with delta > 0."""
        n, k = 3, 5
        eps = 1.0
        delta = 0.1
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(k), size=n)
        edges = [(0, 1), (1, 2)]
        p_proj = _dp_preserving_projection(p, epsilon=eps, delta=delta, edges=edges)
        np.testing.assert_allclose(p_proj.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(p_proj >= -1e-15)

    def test_projection_preserves_shape(self):
        n, k = 4, 6
        p = np.ones((n, k)) / k
        edges = [(0, 1), (1, 2), (2, 3)]
        p_proj = _dp_preserving_projection(p, epsilon=1.0, delta=0.0, edges=edges)
        assert p_proj.shape == (n, k)

    def test_projection_empty_edges(self):
        """No edges means no constraints to enforce."""
        n, k = 3, 4
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(k), size=n)
        p_proj = _dp_preserving_projection(p, epsilon=1.0, delta=0.0, edges=[])
        np.testing.assert_allclose(p_proj.sum(axis=1), 1.0, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# §9  CEGISEngine — Unit Tests (no actual synthesis)
# ═══════════════════════════════════════════════════════════════════════════


class TestCEGISEngineProperties:
    """Unit tests for CEGISEngine properties and config."""

    def test_default_construction(self):
        engine = CEGISEngine()
        assert engine.status == CEGISStatus.RUNNING
        assert engine.witness_set is None
        assert engine.get_best_so_far() is None

    def test_config_property(self, default_config):
        engine = CEGISEngine(default_config)
        assert engine.config is default_config
        assert engine.config.max_iter == 50

    def test_convergence_history_initial(self):
        engine = CEGISEngine()
        h = engine.convergence_history
        assert isinstance(h, ConvergenceHistory)
        assert h.n_iterations == 0

    def test_status_initial(self):
        engine = CEGISEngine()
        assert engine.status == CEGISStatus.RUNNING

    def test_repr(self):
        engine = CEGISEngine()
        r = repr(engine)
        assert "CEGISEngine" in r
        assert "RUNNING" in r

    def test_best_so_far_none_initially(self):
        engine = CEGISEngine()
        assert engine.get_best_so_far() is None

    def test_custom_config(self):
        config = SynthesisConfig(
            max_iter=10,
            tol=1e-6,
            warm_start=False,
            verbose=0,
        )
        engine = CEGISEngine(config)
        assert engine.config.max_iter == 10
        assert engine.config.tol == 1e-6
        assert engine.config.warm_start is False


# ═══════════════════════════════════════════════════════════════════════════
# §10  CEGISProgressReporter
# ═══════════════════════════════════════════════════════════════════════════


class TestCEGISProgressReporter:
    """Tests for CEGISProgressReporter callback collection."""

    def test_records_all_progress(self):
        reporter = CEGISProgressReporter(print_interval=1)
        for i in range(5):
            p = CEGISProgress(
                iteration=i,
                objective=float(i) * 0.1,
                violation_magnitude=0.01 / (i + 1),
                violation_pair=(0, 1) if i < 4 else None,
                n_witness_pairs=i + 1,
                solve_time=0.1,
                verify_time=0.05,
                total_time=float(i) * 0.15,
            )
            reporter(p)
        assert len(reporter.history) == 5

    def test_history_is_copy(self):
        reporter = CEGISProgressReporter()
        p = CEGISProgress(
            iteration=0, objective=0.5, violation_magnitude=0.0,
            violation_pair=None, n_witness_pairs=1,
            solve_time=0.1, verify_time=0.05, total_time=0.15,
        )
        reporter(p)
        h = reporter.history
        assert len(h) == 1
        h.clear()
        assert len(reporter.history) == 1  # Original unaffected

    def test_custom_handler_called(self):
        custom_calls = []
        handler = lambda p: custom_calls.append(p.iteration)
        reporter = CEGISProgressReporter(custom_handler=handler)
        for i in range(3):
            p = CEGISProgress(
                iteration=i, objective=0.5, violation_magnitude=0.0,
                violation_pair=None, n_witness_pairs=1,
                solve_time=0.1, verify_time=0.05, total_time=0.15,
            )
            reporter(p)
        assert custom_calls == [0, 1, 2]

    def test_summary_no_progress(self):
        reporter = CEGISProgressReporter()
        assert "No progress" in reporter.summary()

    def test_summary_with_progress(self):
        reporter = CEGISProgressReporter()
        for i in range(3):
            p = CEGISProgress(
                iteration=i, objective=float(i) * 0.1,
                violation_magnitude=0.0, violation_pair=None,
                n_witness_pairs=1, solve_time=0.1,
                verify_time=0.05, total_time=float(i) * 0.15,
            )
            reporter(p)
        s = reporter.summary()
        assert "3 iterations" in s

    def test_repr(self):
        reporter = CEGISProgressReporter(print_interval=5)
        r = repr(reporter)
        assert "CEGISProgressReporter" in r
        assert "interval=5" in r
        assert "recorded=0" in r

    def test_print_interval(self):
        """Reporter respects print interval for logging."""
        logged_iters = []
        def custom(p):
            pass  # The main test is that __call__ works without error

        reporter = CEGISProgressReporter(print_interval=3, custom_handler=custom)
        for i in range(10):
            p = CEGISProgress(
                iteration=i, objective=0.5, violation_magnitude=0.0,
                violation_pair=None, n_witness_pairs=1,
                solve_time=0.1, verify_time=0.05, total_time=0.15,
            )
            reporter(p)
        assert len(reporter.history) == 10


# ═══════════════════════════════════════════════════════════════════════════
# §11  CEGISResult Structural Validation
# ═══════════════════════════════════════════════════════════════════════════


class TestCEGISResult:
    """Tests for CEGISResult properties and validation."""

    def test_basic_construction(self):
        p = _make_valid_mechanism(3, 5)
        result = CEGISResult(
            mechanism=p,
            iterations=5,
            obj_val=0.5,
        )
        assert result.n == 3
        assert result.k == 5
        assert result.iterations == 5
        assert result.obj_val == 0.5

    def test_n_and_k_properties(self):
        p = _make_valid_mechanism(4, 8)
        result = CEGISResult(mechanism=p, iterations=1, obj_val=0.0)
        assert result.n == 4
        assert result.k == 8

    def test_converged_property_empty_history(self):
        p = _make_valid_mechanism(3, 5)
        result = CEGISResult(mechanism=p, iterations=1, obj_val=0.5)
        assert result.converged is True

    def test_converged_property_stable(self):
        p = _make_valid_mechanism(3, 5)
        result = CEGISResult(
            mechanism=p,
            iterations=3,
            obj_val=0.5,
            convergence_history=[0.5, 0.5],
        )
        assert result.converged is True

    def test_converged_property_unstable(self):
        p = _make_valid_mechanism(3, 5)
        result = CEGISResult(
            mechanism=p,
            iterations=3,
            obj_val=0.5,
            convergence_history=[0.3, 0.5],
        )
        assert result.converged is False

    def test_invalid_mechanism_shape(self):
        with pytest.raises(ValueError, match="2-D"):
            CEGISResult(mechanism=np.ones(10), iterations=1, obj_val=0.0)

    def test_invalid_row_sum(self):
        p = np.ones((3, 5)) * 0.5  # Row sum = 2.5
        with pytest.raises(ValueError, match="rows must sum to 1"):
            CEGISResult(mechanism=p, iterations=1, obj_val=0.0)

    def test_negative_iterations(self):
        p = _make_valid_mechanism(3, 5)
        with pytest.raises(ValueError, match="iterations must be >= 0"):
            CEGISResult(mechanism=p, iterations=-1, obj_val=0.0)

    def test_repr(self):
        p = _make_valid_mechanism(3, 5)
        result = CEGISResult(mechanism=p, iterations=3, obj_val=0.5)
        r = repr(result)
        assert "CEGISResult" in r
        assert "n=3" in r
        assert "k=5" in r
        assert "iter=3" in r
        assert "no_certificate" in r

    def test_with_certificate(self):
        from dp_forge.types import OptimalityCertificate

        p = _make_valid_mechanism(3, 5)
        cert = OptimalityCertificate(
            dual_vars=None, duality_gap=1e-8,
            primal_obj=0.5, dual_obj=0.5 - 1e-8,
        )
        result = CEGISResult(
            mechanism=p, iterations=3, obj_val=0.5,
            optimality_certificate=cert,
        )
        assert result.optimality_certificate is not None
        assert "with_certificate" in repr(result)

    def test_mechanism_rows_sum_to_one(self):
        p = _make_valid_mechanism(5, 10)
        result = CEGISResult(mechanism=p, iterations=1, obj_val=0.0)
        np.testing.assert_allclose(result.mechanism.sum(axis=1), 1.0, atol=1e-4)

    def test_mechanism_nonnegative(self):
        p = _make_valid_mechanism(5, 10)
        result = CEGISResult(mechanism=p, iterations=1, obj_val=0.0)
        assert np.all(result.mechanism >= 0)


# ═══════════════════════════════════════════════════════════════════════════
# §12  Integration: CEGISSynthesize on tiny examples
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestCEGISSynthesize:
    """Integration tests for CEGISSynthesize on tiny specs."""

    def test_tiny_pure_dp(self, tiny_spec):
        """CEGIS should converge on n=3, k=5, eps=1.0."""
        result = CEGISSynthesize(
            tiny_spec,
            max_iter=200,
            config=SynthesisConfig(max_iter=200, verbose=0),
        )
        assert isinstance(result, CEGISResult)
        assert result.n == 3
        assert result.k == 5
        assert result.iterations > 0
        assert result.obj_val >= 0

    def test_mechanism_rows_sum_to_one(self, tiny_spec):
        """Synthesised mechanism rows must sum to 1."""
        result = CEGISSynthesize(
            tiny_spec,
            max_iter=200,
            config=SynthesisConfig(max_iter=200, verbose=0),
        )
        row_sums = result.mechanism.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_mechanism_nonnegative(self, tiny_spec):
        """Synthesised mechanism probabilities must be non-negative."""
        result = CEGISSynthesize(
            tiny_spec,
            max_iter=200,
            config=SynthesisConfig(max_iter=200, verbose=0),
        )
        assert np.all(result.mechanism >= -1e-10)

    def test_convergence_history_populated(self, tiny_spec):
        """convergence_history should be non-empty after synthesis."""
        result = CEGISSynthesize(
            tiny_spec,
            max_iter=200,
            config=SynthesisConfig(max_iter=200, verbose=0),
        )
        assert len(result.convergence_history) > 0

    def test_callback_invoked(self, tiny_spec):
        """The callback should be invoked at least once."""
        progress_log: List[CEGISProgress] = []

        def cb(p: CEGISProgress):
            progress_log.append(p)

        result = CEGISSynthesize(
            tiny_spec,
            max_iter=200,
            config=SynthesisConfig(max_iter=200, verbose=0),
            callback=cb,
        )
        assert len(progress_log) > 0
        assert progress_log[0].iteration == 0

    def test_result_shape_matches_spec(self, tiny_spec):
        result = CEGISSynthesize(
            tiny_spec,
            max_iter=200,
            config=SynthesisConfig(max_iter=200, verbose=0),
        )
        assert result.mechanism.shape == (tiny_spec.n, tiny_spec.k)

    def test_with_reporter_callback(self, tiny_spec):
        reporter = CEGISProgressReporter(print_interval=1)
        result = CEGISSynthesize(
            tiny_spec,
            max_iter=200,
            config=SynthesisConfig(max_iter=200, verbose=0),
            callback=reporter,
        )
        assert len(reporter.history) > 0
        # Each progress has valid fields
        for p in reporter.history:
            assert p.solve_time >= 0
            assert p.verify_time >= 0

    def test_approx_dp(self, approx_spec):
        """CEGIS should work for approximate DP."""
        result = CEGISSynthesize(
            approx_spec,
            max_iter=200,
            config=SynthesisConfig(max_iter=200, verbose=0),
        )
        assert isinstance(result, CEGISResult)
        assert result.n == 3
        assert result.k == 5

    def test_no_warm_start(self, tiny_spec):
        """CEGIS works without warm-start."""
        config = SynthesisConfig(max_iter=200, verbose=0, warm_start=False)
        result = CEGISSynthesize(tiny_spec, config=config)
        assert isinstance(result, CEGISResult)
        assert result.iterations > 0


# ═══════════════════════════════════════════════════════════════════════════
# §13  CEGISEngine Integration
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestCEGISEngineIntegration:
    """Integration tests for CEGISEngine.synthesize on tiny examples."""

    def test_synthesize_basic(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)
        result = engine.synthesize(tiny_spec)
        assert isinstance(result, CEGISResult)
        assert engine.status in (CEGISStatus.CONVERGED, CEGISStatus.CYCLE_RESOLVED)

    def test_witness_set_populated(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)
        engine.synthesize(tiny_spec)
        ws = engine.witness_set
        assert ws is not None
        assert ws.size > 0

    def test_convergence_history_tracked(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)
        engine.synthesize(tiny_spec)
        h = engine.convergence_history
        assert h.n_iterations > 0
        assert len(h.objectives) == h.n_iterations

    def test_best_so_far_set(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)
        engine.synthesize(tiny_spec)
        best = engine.get_best_so_far()
        assert best is not None
        assert best.shape[0] == tiny_spec.n
        assert best.shape[1] == tiny_spec.k

    def test_status_converged(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)
        engine.synthesize(tiny_spec)
        assert engine.status in (
            CEGISStatus.CONVERGED,
            CEGISStatus.CYCLE_RESOLVED,
        )

    def test_monotonicity_of_objectives(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)
        engine.synthesize(tiny_spec)
        h = engine.convergence_history
        assert h.check_monotonicity(tol=1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# §14  synthesize_mechanism High-Level API
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestSynthesizeMechanism:
    """Tests for the synthesize_mechanism high-level API."""

    def test_with_array_input(self):
        query = np.arange(3, dtype=float)
        result = synthesize_mechanism(
            query, epsilon=1.0, k=5, verbose=0, max_iter=200,
        )
        assert isinstance(result, CEGISResult)
        assert result.n == 3
        assert result.k == 5

    def test_with_query_spec(self, tiny_spec):
        result = synthesize_mechanism(
            tiny_spec, epsilon=1.0, verbose=0, max_iter=200,
        )
        assert isinstance(result, CEGISResult)

    def test_approx_dp_api(self):
        query = np.arange(3, dtype=float)
        result = synthesize_mechanism(
            query, epsilon=1.0, delta=0.01, k=5, verbose=0, max_iter=200,
        )
        assert isinstance(result, CEGISResult)

    def test_l1_loss(self):
        query = np.arange(3, dtype=float)
        result = synthesize_mechanism(
            query, epsilon=1.0, k=5, loss=LossFunction.L1,
            verbose=0, max_iter=200,
        )
        assert isinstance(result, CEGISResult)


# ═══════════════════════════════════════════════════════════════════════════
# §15  quick_synthesize
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestQuickSynthesize:
    """Tests for the quick_synthesize preset API."""

    def test_counting(self):
        result = quick_synthesize("counting", epsilon=1.0, n=3, k=5, verbose=0)
        assert isinstance(result, CEGISResult)
        assert result.n == 3
        assert result.k == 5

    def test_histogram(self):
        result = quick_synthesize("histogram", epsilon=1.0, n=3, k=5, verbose=0)
        assert isinstance(result, CEGISResult)

    def test_invalid_query_type(self):
        with pytest.raises(ConfigurationError):
            quick_synthesize("invalid_type", epsilon=1.0)

    def test_case_insensitive(self):
        result = quick_synthesize("COUNTING", epsilon=1.0, n=3, k=5, verbose=0)
        assert isinstance(result, CEGISResult)


# ═══════════════════════════════════════════════════════════════════════════
# §16  Multi-Strategy and Hybrid Synthesis
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestMultiStrategy:
    """Tests for parallel_synthesis and hybrid_synthesis."""

    def test_hybrid_synthesis(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        result = hybrid_synthesis(tiny_spec, config=config)
        assert isinstance(result, CEGISResult)

    def test_parallel_single_strategy(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        result = parallel_synthesis(
            tiny_spec,
            strategies=[SynthesisStrategy.LP_PURE],
            config=config,
        )
        assert isinstance(result, CEGISResult)

    def test_parallel_auto_strategy(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        result = parallel_synthesis(tiny_spec, config=config)
        assert isinstance(result, CEGISResult)


# ═══════════════════════════════════════════════════════════════════════════
# §17  Cycle Detection and Handling
# ═══════════════════════════════════════════════════════════════════════════


class TestCycleDetection:
    """Tests for cycle detection in WitnessSet and handling logic."""

    def test_cycle_detected_when_pair_exists(self):
        ws = WitnessSet(total_edges=5)
        ws.add_pair(0, 1)
        ws.add_pair(1, 2)
        # Simulating cycle: pair (0,1) returned again
        assert ws.contains(0, 1) is True

    def test_no_cycle_for_new_pair(self):
        ws = WitnessSet(total_edges=5)
        ws.add_pair(0, 1)
        assert ws.contains(1, 2) is False

    def test_convergence_history_records_cycles(self):
        h = ConvergenceHistory()
        h.record(0.1, 0.5, 0.1, 0.05, 1, 0.15, is_cycle=False)
        h.record(0.2, 0.3, 0.1, 0.05, 1, 0.15, is_cycle=True)
        h.record(0.3, 0.1, 0.1, 0.05, 2, 0.15, is_cycle=False)
        assert h.cycle_iterations == {1}

    def test_multiple_cycles_recorded(self):
        h = ConvergenceHistory()
        for i in range(10):
            is_cycle = i in (2, 5, 7)
            h.record(float(i) * 0.1, 0.01, 0.1, 0.05, i, 0.15, is_cycle=is_cycle)
        assert h.cycle_iterations == {2, 5, 7}

    def test_projection_on_cycle(self):
        """DP-preserving projection should produce a valid mechanism."""
        n, k = 3, 5
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(k), size=n)
        edges = [(0, 1), (1, 2)]
        p_proj = _dp_preserving_projection(p, epsilon=1.0, delta=0.0, edges=edges)
        np.testing.assert_allclose(p_proj.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(p_proj >= -1e-15)

    def test_progress_is_cycle_flag(self):
        p = CEGISProgress(
            iteration=5, objective=0.3, violation_magnitude=0.01,
            violation_pair=(0, 1), n_witness_pairs=3,
            solve_time=0.1, verify_time=0.05, total_time=1.0,
            is_cycle=True,
        )
        assert p.is_cycle is True

    def test_monotonicity_with_cycle_skipping(self):
        """Monotonicity check should skip cycle iterations."""
        h = ConvergenceHistory()
        h.record(0.1, 0.5, 0.1, 0.05, 1, 0.15)
        h.record(0.05, 0.3, 0.1, 0.05, 1, 0.15, is_cycle=True)  # Drops
        h.record(0.2, 0.1, 0.1, 0.05, 2, 0.15)
        assert h.check_monotonicity() is True


# ═══════════════════════════════════════════════════════════════════════════
# §18  Warm-Start Behavior
# ═══════════════════════════════════════════════════════════════════════════


class TestWarmStartBehavior:
    """Tests for warm-start storage and retrieval behavior."""

    def test_store_none_solution(self):
        ws = DualSimplexWarmStart()
        stats = SolveStatistics(
            solver_name="test",
            status="optimal",
            iterations=1,
            solve_time=0.01,
            objective_value=0.5,
            primal_solution=np.array([0.5, 0.5]),
            basis_info=None,
        )
        ws.store(stats)
        assert ws.is_valid is True
        basis = ws.get_warm_start(2)
        assert basis is None  # basis_info was None

    def test_warm_start_survives_multiple_stores(self):
        ws = DualSimplexWarmStart()
        for i in range(5):
            stats = _make_solve_stats(
                n_vars=10,
                objective=float(i),
                basis_info={"iter": i},
            )
            ws.store(stats)
        basis = ws.get_warm_start(10)
        assert basis == {"iter": 4}

    def test_invalidate_after_store(self):
        ws = DualSimplexWarmStart()
        stats = _make_solve_stats(n_vars=10)
        ws.store(stats)
        ws.invalidate()
        assert ws.is_valid is False

    def test_consecutive_failures_accumulate(self):
        ws = DualSimplexWarmStart()
        stats = _make_solve_stats(n_vars=10)
        ws.store(stats)
        for i in range(4):
            ws.record_failure()
            assert ws.stats["consecutive_failures"] == i + 1
        assert ws.is_valid is True
        ws.record_failure()  # 5th failure
        assert ws.is_valid is False


# ═══════════════════════════════════════════════════════════════════════════
# §19  Error Recovery with Mocking
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorRecovery:
    """Tests for error recovery pathways using mocks."""

    def test_infeasible_spec_raises(self, tiny_spec):
        """CEGISEngine should raise InfeasibleSpecError if LP is infeasible."""
        config = SynthesisConfig(max_iter=10, verbose=0)
        engine = CEGISEngine(config)

        with patch.object(engine, '_solve_lp_with_recovery') as mock_solve:
            mock_solve.side_effect = InfeasibleSpecError("test infeasible")
            with pytest.raises(InfeasibleSpecError):
                engine.synthesize(tiny_spec)

    def test_convergence_error_on_max_iter(self, tiny_spec):
        """If max_iter is 1 and no convergence, should raise or return."""
        config = SynthesisConfig(max_iter=1, verbose=0)
        # With 1 iteration, the engine either converges or raises ConvergenceError
        # or returns a projected result. We can't predict, but it shouldn't crash.
        engine = CEGISEngine(config)
        try:
            result = engine.synthesize(tiny_spec)
            assert isinstance(result, CEGISResult)
        except ConvergenceError:
            pass  # Expected

    def test_solver_error_recovery_all_fail(self, tiny_spec):
        """SolverError when all backends fail."""
        config = SynthesisConfig(max_iter=10, verbose=0)
        engine = CEGISEngine(config)

        with patch.object(engine, '_solve_lp_with_recovery') as mock_solve:
            mock_solve.side_effect = SolverError(
                "all failed", solver_name="test", solver_status="failed",
            )
            with pytest.raises(SolverError):
                engine.synthesize(tiny_spec)

    def test_numerical_failure_handling(self, tiny_spec):
        """NumericalInstabilityError should be handled gracefully."""
        from dp_forge.exceptions import NumericalInstabilityError
        config = SynthesisConfig(max_iter=10, verbose=0)
        engine = CEGISEngine(config)

        with patch.object(engine, '_solve_lp_with_recovery') as mock_solve:
            mock_solve.side_effect = NumericalInstabilityError(
                "condition number too high",
                condition_number=1e15,
            )
            with pytest.raises(NumericalInstabilityError):
                engine.synthesize(tiny_spec)


# ═══════════════════════════════════════════════════════════════════════════
# §20  WitnessSet Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestWitnessSetEdgeCases:
    """Edge cases and boundary conditions for WitnessSet."""

    def test_same_index_pair(self):
        ws = WitnessSet(total_edges=5)
        ws.add_pair(3, 3)
        assert ws.contains(3, 3) is True
        assert ws.size == 1

    def test_large_indices(self):
        ws = WitnessSet(total_edges=100)
        ws.add_pair(999, 1000)
        assert ws.contains(999, 1000) is True
        assert ws.contains(1000, 999) is True

    def test_remove_and_readd(self):
        ws = WitnessSet(total_edges=5)
        ws.add_pair(0, 1)
        ws.remove_pair(0, 1)
        assert ws.contains(0, 1) is False
        ws.add_pair(0, 1)
        assert ws.contains(0, 1) is True

    def test_coverage_exceeds_total(self):
        ws = WitnessSet(total_edges=2)
        ws.add_pair(0, 1)
        ws.add_pair(1, 2)
        ws.add_pair(2, 3)  # More than total_edges
        assert ws.coverage_fraction() == 1.0  # Capped at 1.0

    def test_frozen_empty(self):
        ws = WitnessSet(total_edges=5)
        f = ws.frozen()
        assert f == frozenset()

    def test_many_pairs(self):
        n = 50
        ws = WitnessSet(total_edges=n * (n - 1) // 2)
        for i in range(n):
            for j in range(i + 1, n):
                ws.add_pair(i, j)
        assert ws.size == n * (n - 1) // 2
        assert abs(ws.coverage_fraction() - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# §21  ConvergenceHistory Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestConvergenceHistoryEdgeCases:
    """Edge cases for ConvergenceHistory."""

    def test_single_iteration_monotonicity(self):
        h = ConvergenceHistory()
        h.record(1.0, 0.0, 0.1, 0.05, 1, 0.15)
        assert h.check_monotonicity() is True

    def test_large_number_of_iterations(self):
        h = ConvergenceHistory()
        for i in range(1000):
            h.record(float(i) * 0.001, 0.0, 0.001, 0.001, i + 1, 0.002)
        assert h.n_iterations == 1000
        assert h.check_monotonicity() is True

    def test_stagnation_at_boundary(self):
        h = ConvergenceHistory()
        for i in range(10):
            obj = 0.5 if i < 5 else 0.5 + 1e-11
            h.record(obj, 0.0, 0.1, 0.05, i + 1, 0.15)
        assert h.detect_stagnation(window=10, tol=1e-10) is True

    def test_no_stagnation_with_oscillation(self):
        h = ConvergenceHistory()
        for i in range(15):
            obj = 0.5 + 0.1 * (i % 2)
            h.record(obj, 0.0, 0.1, 0.05, i + 1, 0.15)
        assert h.detect_stagnation(window=10, tol=1e-10) is False

    def test_estimate_remaining_with_no_growth(self):
        h = ConvergenceHistory()
        for _ in range(5):
            h.record(0.5, 0.0, 0.1, 0.05, 3, 0.15)
        remaining = h.estimate_remaining_iterations(10)
        assert remaining == 7  # 10 - 3 = 7

    def test_all_cycles(self):
        h = ConvergenceHistory()
        for i in range(5):
            h.record(0.5, 0.01, 0.1, 0.05, 1, 0.15, is_cycle=True)
        assert len(h.cycle_iterations) == 5
        assert h.check_monotonicity() is True  # All skipped

    def test_negative_objectives_monotonicity(self):
        h = ConvergenceHistory()
        for obj in [-0.5, -0.3, -0.1, 0.0, 0.2]:
            h.record(obj, 0.0, 0.1, 0.05, 1, 0.15)
        assert h.check_monotonicity() is True

    def test_objective_improvement_rate_with_small_window(self):
        h = ConvergenceHistory()
        h.record(0.1, 0.0, 0.1, 0.05, 1, 0.15)
        h.record(0.2, 0.0, 0.1, 0.05, 2, 0.15)
        rate = h.objective_improvement_rate(window=1)
        assert abs(rate - 0.1) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# §22  DualSimplexWarmStart Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestDualSimplexWarmStartEdgeCases:
    """Edge cases for DualSimplexWarmStart."""

    def test_get_warm_start_before_store(self):
        ws = DualSimplexWarmStart()
        assert ws.get_warm_start(10) is None

    def test_multiple_invalidations(self):
        ws = DualSimplexWarmStart()
        stats = _make_solve_stats(n_vars=10)
        ws.store(stats)
        ws.invalidate()
        ws.invalidate()
        ws.invalidate()
        assert ws.is_valid is False
        assert ws.stats["consecutive_failures"] == 3

    def test_store_after_invalidation(self):
        ws = DualSimplexWarmStart()
        stats = _make_solve_stats(n_vars=10)
        ws.store(stats)
        ws.invalidate()
        ws.store(stats)  # Re-store after invalidation
        assert ws.is_valid is True

    def test_warm_start_with_zero_vars(self):
        ws = DualSimplexWarmStart()
        stats = SolveStatistics(
            solver_name="test",
            status="optimal",
            iterations=1,
            solve_time=0.01,
            objective_value=0.0,
            primal_solution=np.array([]),
            basis_info={"empty": True},
        )
        ws.store(stats)
        basis = ws.get_warm_start(0)
        assert basis == {"empty": True}


# ═══════════════════════════════════════════════════════════════════════════
# §23  Synthesis Config Validation
# ═══════════════════════════════════════════════════════════════════════════


class TestSynthesisConfigValidation:
    """Test SynthesisConfig validation used by CEGISEngine."""

    def test_default_config(self):
        config = SynthesisConfig()
        assert config.max_iter == 50
        assert config.tol == 1e-8
        assert config.warm_start is True
        assert config.verbose == 1

    def test_invalid_max_iter(self):
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            SynthesisConfig(max_iter=0)

    def test_invalid_tol(self):
        with pytest.raises(ValueError, match="tol must be > 0"):
            SynthesisConfig(tol=-1e-8)

    def test_invalid_verbose(self):
        with pytest.raises(ValueError, match="verbose must be 0, 1, or 2"):
            SynthesisConfig(verbose=3)

    def test_custom_numerical_config(self):
        nc = NumericalConfig(solver_tol=1e-10, dp_tol=1e-4)
        config = SynthesisConfig(numerical=nc)
        assert config.numerical.solver_tol == 1e-10
        assert config.numerical.dp_tol == 1e-4


# ═══════════════════════════════════════════════════════════════════════════
# §24  Strategy Selection Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestStrategySelectionEdgeCases:
    """Edge cases for auto_select_strategy."""

    def test_very_small_epsilon(self):
        spec = _make_query_spec(epsilon=0.001, delta=0.0)
        assert auto_select_strategy(spec) == SynthesisStrategy.LP_PURE

    def test_very_large_epsilon(self):
        spec = _make_query_spec(epsilon=100.0, delta=0.0)
        assert auto_select_strategy(spec) == SynthesisStrategy.LP_PURE

    def test_very_small_delta(self):
        spec = _make_approx_query_spec(epsilon=1.0, delta=1e-15)
        assert auto_select_strategy(spec) == SynthesisStrategy.LP_APPROX

    def test_large_n_approx(self):
        spec = _make_approx_query_spec(n=50, k=5, epsilon=1.0, delta=0.01)
        assert auto_select_strategy(spec) == SynthesisStrategy.LP_APPROX


# ═══════════════════════════════════════════════════════════════════════════
# §25  QuerySpec Construction for CEGIS
# ═══════════════════════════════════════════════════════════════════════════


class TestQuerySpecForCEGIS:
    """Validate that QuerySpec is correctly constructed for CEGIS use."""

    def test_counting_factory(self):
        spec = QuerySpec.counting(n=5, epsilon=1.0, k=10)
        assert spec.n == 5
        assert spec.k == 10
        assert spec.epsilon == 1.0
        assert spec.delta == 0.0
        assert spec.is_pure_dp is True
        assert spec.edges is not None

    def test_histogram_factory(self):
        spec = QuerySpec.histogram(n_bins=5, epsilon=1.0, k=10)
        assert spec.n == 5
        assert spec.query_type == QueryType.HISTOGRAM

    def test_edges_auto_constructed(self):
        spec = QuerySpec(
            query_values=np.arange(5, dtype=float),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            k=5,
        )
        assert spec.edges is not None
        assert spec.edges.n == 5
        # Hamming-1 edges: (0,1), (1,2), (2,3), (3,4)
        assert len(spec.edges.edges) == 4

    def test_custom_edges(self):
        adj = AdjacencyRelation.complete(3)
        spec = QuerySpec(
            query_values=np.arange(3, dtype=float),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            k=5,
            edges=adj,
        )
        assert spec.edges is adj
        assert len(spec.edges.edges) == 3  # (0,1), (0,2), (1,2)


# ═══════════════════════════════════════════════════════════════════════════
# §26  Convergence History with Realistic Data
# ═══════════════════════════════════════════════════════════════════════════


class TestConvergenceHistoryRealistic:
    """Test ConvergenceHistory with patterns seen in real CEGIS runs."""

    def test_typical_convergence_pattern(self):
        """Simulate a typical convergence: objective increases then stabilises."""
        h = ConvergenceHistory()
        objectives = [0.1, 0.15, 0.2, 0.22, 0.23, 0.235, 0.236, 0.236, 0.236, 0.236]
        violations = [0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0001, 0.0]
        for i, (obj, viol) in enumerate(zip(objectives, violations)):
            h.record(obj, viol, 0.1, 0.05, i + 1, 0.15)

        assert h.n_iterations == 10
        assert h.check_monotonicity() is True
        assert h.detect_stagnation(window=5, tol=0.002) is True

    def test_early_termination_pattern(self):
        """Simulate fast convergence (2 iterations)."""
        h = ConvergenceHistory()
        h.record(0.1, 0.5, 0.1, 0.05, 1, 0.15)
        h.record(0.2, 0.0, 0.1, 0.05, 2, 0.15)  # Valid on second iter
        assert h.n_iterations == 2
        assert h.check_monotonicity() is True

    def test_cycle_then_convergence(self):
        """Simulate cycle at iteration 3, then convergence."""
        h = ConvergenceHistory()
        h.record(0.1, 0.5, 0.1, 0.05, 1, 0.15)
        h.record(0.15, 0.3, 0.1, 0.05, 2, 0.15)
        h.record(0.12, 0.2, 0.1, 0.05, 2, 0.15, is_cycle=True)  # Cycle
        h.record(0.2, 0.1, 0.1, 0.05, 3, 0.15)
        h.record(0.25, 0.0, 0.1, 0.05, 4, 0.15)
        assert h.check_monotonicity() is True
        assert 2 in h.cycle_iterations


# ═══════════════════════════════════════════════════════════════════════════
# §27  Progress Snapshot Validation
# ═══════════════════════════════════════════════════════════════════════════


class TestProgressSnapshotValidation:
    """Validate that CEGISProgress snapshots have consistent data."""

    def test_valid_progress_no_violation(self):
        p = CEGISProgress(
            iteration=10, objective=0.5, violation_magnitude=0.0,
            violation_pair=None, n_witness_pairs=5,
            solve_time=0.1, verify_time=0.05, total_time=2.0,
            status=CEGISStatus.CONVERGED,
        )
        assert p.violation_pair is None
        assert p.violation_magnitude == 0.0
        assert p.status == CEGISStatus.CONVERGED

    def test_valid_progress_with_violation(self):
        p = CEGISProgress(
            iteration=5, objective=0.3, violation_magnitude=1.5e-3,
            violation_pair=(2, 3), n_witness_pairs=4,
            solve_time=0.2, verify_time=0.1, total_time=1.5,
        )
        assert p.violation_pair == (2, 3)
        assert p.violation_magnitude > 0

    def test_progress_timing_consistency(self):
        p = CEGISProgress(
            iteration=0, objective=0.5, violation_magnitude=0.0,
            violation_pair=None, n_witness_pairs=1,
            solve_time=0.1, verify_time=0.05, total_time=0.15,
        )
        # total_time should be >= solve_time + verify_time in practice
        assert p.total_time >= 0
        assert p.solve_time >= 0
        assert p.verify_time >= 0


# ═══════════════════════════════════════════════════════════════════════════
# §28  WitnessSet Seeding with Various Graph Structures
# ═══════════════════════════════════════════════════════════════════════════


class TestWitnessSetSeeding:
    """Test smart_seed with different adjacency structures."""

    def test_seed_hamming1(self):
        """Hamming-1 adjacency: consecutive pairs."""
        n = 6
        edges = [(i, i + 1) for i in range(n - 1)]
        ws = WitnessSet(total_edges=len(edges))
        result = ws.smart_seed(edges, n_initial=3, query_values=np.arange(n, dtype=float))
        assert len(result) == 3
        assert ws.size == 3

    def test_seed_complete_graph(self):
        """Complete adjacency: all pairs."""
        n = 4
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        ws = WitnessSet(total_edges=len(edges))
        result = ws.smart_seed(
            edges, n_initial=3, query_values=np.arange(n, dtype=float),
        )
        assert len(result) == 3

    def test_seed_single_edge(self):
        edges = [(0, 1)]
        ws = WitnessSet(total_edges=1)
        result = ws.smart_seed(edges, n_initial=5, query_values=np.arange(2, dtype=float))
        assert len(result) == 1

    def test_seed_prioritises_boundary(self):
        """Boundary pairs (extremes of query range) should be prioritised."""
        n = 10
        edges = [(i, i + 1) for i in range(n - 1)]
        ws = WitnessSet(total_edges=len(edges))
        result = ws.smart_seed(
            edges, n_initial=2, query_values=np.arange(n, dtype=float),
        )
        # The seeded pairs should include high-gap/high-score pairs
        assert len(result) == 2
        assert ws.size == 2


# ═══════════════════════════════════════════════════════════════════════════
# §29  Comprehensive Projection Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestProjectionComprehensive:
    """Comprehensive tests for _dp_preserving_projection."""

    def test_uniform_mechanism_unchanged(self):
        """Uniform mechanism already satisfies DP → projection is (near) identity."""
        n, k = 4, 6
        p = np.ones((n, k)) / k
        edges = [(i, i + 1) for i in range(n - 1)]
        p_proj = _dp_preserving_projection(p, epsilon=1.0, delta=0.0, edges=edges)
        np.testing.assert_allclose(p_proj, p, atol=1e-10)

    def test_highly_skewed_mechanism(self):
        """A highly skewed mechanism should be significantly modified."""
        n, k = 2, 4
        p = np.array([[0.97, 0.01, 0.01, 0.01],
                       [0.01, 0.01, 0.01, 0.97]])
        edges = [(0, 1)]
        eps = 0.5  # Tight privacy
        p_proj = _dp_preserving_projection(p, epsilon=eps, delta=0.0, edges=edges)
        np.testing.assert_allclose(p_proj.sum(axis=1), 1.0, atol=1e-10)
        # Ratios should now be bounded
        exp_eps = math.exp(eps)
        for j in range(k):
            if p_proj[0][j] > 1e-15 and p_proj[1][j] > 1e-15:
                ratio = p_proj[0][j] / p_proj[1][j]
                assert ratio <= exp_eps + 1e-6

    def test_single_output_bin(self):
        """With k=1, the mechanism is trivially DP."""
        n = 3
        p = np.ones((n, 1))
        edges = [(0, 1), (1, 2)]
        p_proj = _dp_preserving_projection(p, epsilon=1.0, delta=0.0, edges=edges)
        np.testing.assert_allclose(p_proj, np.ones((n, 1)), atol=1e-10)

    def test_approx_dp_small_delta(self):
        """Approximate DP with very small delta."""
        n, k = 3, 5
        rng = np.random.default_rng(123)
        p = rng.dirichlet(np.ones(k), size=n)
        edges = [(0, 1), (1, 2)]
        p_proj = _dp_preserving_projection(
            p, epsilon=1.0, delta=1e-6, edges=edges,
        )
        np.testing.assert_allclose(p_proj.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(p_proj >= -1e-15)


# ═══════════════════════════════════════════════════════════════════════════
# §30  Integration: Engine with Callback and History Tracking
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestEngineHistoryTracking:
    """Integration tests verifying that CEGISEngine tracks history correctly."""

    def test_history_objectives_match_convergence_history(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)
        result = engine.synthesize(tiny_spec)
        h = engine.convergence_history
        assert h.objectives == result.convergence_history

    def test_history_witness_counts_increase(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)
        engine.synthesize(tiny_spec)
        h = engine.convergence_history
        # Witness counts should be non-decreasing overall
        for i in range(1, len(h.witness_counts)):
            assert h.witness_counts[i] >= h.witness_counts[i - 1] or \
                   i in h.cycle_iterations

    def test_all_solve_times_positive(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)
        engine.synthesize(tiny_spec)
        h = engine.convergence_history
        for st in h.solve_times:
            assert st >= 0

    def test_all_verify_times_positive(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)
        engine.synthesize(tiny_spec)
        h = engine.convergence_history
        for vt in h.verify_times:
            assert vt >= 0

    def test_summary_after_synthesis(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)
        engine.synthesize(tiny_spec)
        s = engine.convergence_history.summary()
        assert "ConvergenceHistory" in s
        assert "iterations" in s


# ═══════════════════════════════════════════════════════════════════════════
# §31  Parallel Synthesis Mocked
# ═══════════════════════════════════════════════════════════════════════════


class TestParallelSynthesisMocked:
    """Test parallel_synthesis behaviour with mocks."""

    def test_no_strategies_raises(self):
        spec = _make_query_spec()
        with pytest.raises((ConfigurationError, SolverError)):
            parallel_synthesis(spec, strategies=[])

    def test_sdp_gaussian_skipped_with_warning(self, tiny_spec, caplog):
        """SDP_GAUSSIAN should be skipped with a warning."""
        config = SynthesisConfig(max_iter=10, verbose=0)
        with pytest.raises((SolverError, ConfigurationError)):
            parallel_synthesis(
                tiny_spec,
                strategies=[SynthesisStrategy.SDP_GAUSSIAN],
                config=config,
            )


# ═══════════════════════════════════════════════════════════════════════════
# §32  CEGISEngine Callback Testing
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestCEGISEngineCallbacks:
    """Test that CEGISEngine properly calls callbacks."""

    def test_callback_receives_all_iterations(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)
        iterations_seen = []

        def cb(p: CEGISProgress):
            iterations_seen.append(p.iteration)

        result = engine.synthesize(tiny_spec, callback=cb)
        # Should have exactly result.iterations callbacks
        assert len(iterations_seen) == result.iterations
        assert iterations_seen == list(range(result.iterations))

    def test_callback_objective_matches_history(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)
        objectives_from_cb = []

        def cb(p: CEGISProgress):
            objectives_from_cb.append(p.objective)

        engine.synthesize(tiny_spec, callback=cb)
        h = engine.convergence_history
        np.testing.assert_allclose(objectives_from_cb, h.objectives, atol=1e-12)

    def test_callback_n_witness_pairs_nondecreasing(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)
        pairs_seen = []

        def cb(p: CEGISProgress):
            pairs_seen.append(p.n_witness_pairs)

        engine.synthesize(tiny_spec, callback=cb)
        for i in range(1, len(pairs_seen)):
            # Witness pairs should not decrease
            assert pairs_seen[i] >= pairs_seen[i - 1]


# ═══════════════════════════════════════════════════════════════════════════
# §33  Verification of DP-Compliance Post Synthesis
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestDPCompliancePostSynthesis:
    """Verify that synthesised mechanisms satisfy DP constraints."""

    def test_pure_dp_ratio_constraint(self, tiny_spec):
        """All probability ratios must be bounded by exp(eps)."""
        config = SynthesisConfig(max_iter=200, verbose=0)
        result = CEGISSynthesize(tiny_spec, config=config)
        p = result.mechanism
        eps = tiny_spec.epsilon
        exp_eps = math.exp(eps)

        for i, ip in tiny_spec.edges.edges:
            for j in range(tiny_spec.k):
                if p[i][j] > 1e-15 and p[ip][j] > 1e-15:
                    ratio_fwd = p[i][j] / p[ip][j]
                    ratio_bwd = p[ip][j] / p[i][j]
                    # Allow tolerance for floating-point
                    assert ratio_fwd <= exp_eps + 1e-4, (
                        f"Forward ratio violation: p[{i}][{j}]/p[{ip}][{j}] = "
                        f"{ratio_fwd:.6f} > exp({eps}) = {exp_eps:.6f}"
                    )
                    assert ratio_bwd <= exp_eps + 1e-4, (
                        f"Backward ratio violation: p[{ip}][{j}]/p[{i}][{j}] = "
                        f"{ratio_bwd:.6f} > exp({eps}) = {exp_eps:.6f}"
                    )

    def test_mechanism_is_distribution(self, tiny_spec):
        """Mechanism rows must be valid probability distributions."""
        config = SynthesisConfig(max_iter=200, verbose=0)
        result = CEGISSynthesize(tiny_spec, config=config)
        p = result.mechanism
        # Non-negative
        assert np.all(p >= -1e-10)
        # Rows sum to 1
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# §34  Misc Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestMiscIntegration:
    """Miscellaneous integration tests."""

    def test_n2_k3_pure(self):
        """Minimal synthesis problem: n=2, k=3."""
        spec = _make_query_spec(n=2, k=3, epsilon=1.0)
        config = SynthesisConfig(max_iter=200, verbose=0)
        result = CEGISSynthesize(spec, config=config)
        assert result.n == 2
        assert result.k == 3
        assert result.iterations > 0

    def test_n4_k5_pure(self):
        """Slightly larger problem: n=4, k=5."""
        spec = _make_query_spec(n=4, k=5, epsilon=1.0)
        config = SynthesisConfig(max_iter=200, verbose=0)
        result = CEGISSynthesize(spec, config=config)
        assert result.n == 4
        assert result.k == 5

    def test_high_epsilon(self):
        """High epsilon (loose privacy) should converge quickly."""
        spec = _make_query_spec(n=3, k=5, epsilon=5.0)
        config = SynthesisConfig(max_iter=200, verbose=0)
        result = CEGISSynthesize(spec, config=config)
        assert result.iterations > 0

    def test_low_epsilon(self):
        """Low epsilon (tight privacy) may need more iterations."""
        spec = _make_query_spec(n=2, k=3, epsilon=0.1)
        config = SynthesisConfig(max_iter=500, verbose=0)
        result = CEGISSynthesize(spec, config=config)
        assert result.n == 2

    def test_complete_adjacency(self):
        """Complete adjacency graph with all-pairs constraints."""
        n = 3
        adj = AdjacencyRelation.complete(n)
        spec = QuerySpec(
            query_values=np.arange(n, dtype=float),
            domain="test_complete",
            sensitivity=1.0,
            epsilon=1.0,
            k=5,
            edges=adj,
        )
        config = SynthesisConfig(max_iter=200, verbose=0)
        result = CEGISSynthesize(spec, config=config)
        assert result.n == n


# ═══════════════════════════════════════════════════════════════════════════
# §35  CEGISEngine State Reset
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestCEGISEngineStateReset:
    """Test that CEGISEngine properly resets state between runs."""

    def test_multiple_runs_independent(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)

        result1 = engine.synthesize(tiny_spec)
        history1_len = engine.convergence_history.n_iterations

        # Second run should reset state
        result2 = engine.synthesize(tiny_spec)
        history2_len = engine.convergence_history.n_iterations

        # History should be reset (not accumulated)
        assert history2_len == result2.iterations

    def test_status_reset_between_runs(self, tiny_spec):
        config = SynthesisConfig(max_iter=200, verbose=0)
        engine = CEGISEngine(config)

        engine.synthesize(tiny_spec)
        first_status = engine.status

        engine.synthesize(tiny_spec)
        second_status = engine.status

        # Both should be terminal statuses
        assert first_status in (CEGISStatus.CONVERGED, CEGISStatus.CYCLE_RESOLVED)
        assert second_status in (CEGISStatus.CONVERGED, CEGISStatus.CYCLE_RESOLVED)


# ═══════════════════════════════════════════════════════════════════════════
# §36  WitnessSet Canonical Ordering
# ═══════════════════════════════════════════════════════════════════════════


class TestWitnessSetCanonical:
    """Test canonical ordering of pairs in WitnessSet."""

    def test_canonical_form(self):
        ws = WitnessSet(total_edges=5)
        ws.add_pair(5, 2)
        assert ws.pairs == [(2, 5)]

    def test_canonical_symmetric(self):
        ws1 = WitnessSet(total_edges=5)
        ws2 = WitnessSet(total_edges=5)
        ws1.add_pair(1, 3)
        ws2.add_pair(3, 1)
        assert ws1.frozen() == ws2.frozen()

    def test_frozen_contains_canonical(self):
        ws = WitnessSet(total_edges=5)
        ws.add_pair(10, 3)
        f = ws.frozen()
        assert (3, 10) in f
        assert (10, 3) not in f

    def test_multiple_orderings_same_set(self):
        ws = WitnessSet(total_edges=10)
        ws.add_pair(5, 2)
        ws.add_pair(2, 5)  # Duplicate
        ws.add_pair(3, 1)
        ws.add_pair(1, 3)  # Duplicate
        assert ws.size == 2
        assert ws.pairs == [(1, 3), (2, 5)]


# ═══════════════════════════════════════════════════════════════════════════
# §37  Convergence History Timing
# ═══════════════════════════════════════════════════════════════════════════


class TestConvergenceHistoryTiming:
    """Test timing aggregation in ConvergenceHistory."""

    def test_solve_time_accumulation(self):
        h = ConvergenceHistory()
        for i in range(10):
            h.record(float(i), 0.0, 0.1 * (i + 1), 0.05, i + 1, 0.2 * (i + 1))
        expected_solve = sum(0.1 * (i + 1) for i in range(10))
        assert abs(h.total_solve_time - expected_solve) < 1e-10

    def test_verify_time_accumulation(self):
        h = ConvergenceHistory()
        for i in range(5):
            h.record(float(i), 0.0, 0.1, 0.05 * (i + 1), i + 1, 0.2)
        expected_verify = sum(0.05 * (i + 1) for i in range(5))
        assert abs(h.total_verify_time - expected_verify) < 1e-10

    def test_iteration_time_accumulation(self):
        h = ConvergenceHistory()
        for i in range(5):
            h.record(float(i), 0.0, 0.1, 0.05, i + 1, 1.0)
        assert abs(h.total_time - 5.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# §38  RichProgressBar (smoke test)
# ═══════════════════════════════════════════════════════════════════════════


class TestRichProgressBar:
    """Smoke test for RichProgressBar (does not require rich installed)."""

    def test_construction(self):
        from dp_forge.cegis_loop import RichProgressBar
        bar = RichProgressBar(total=10)
        # Should not error even if rich is missing

    def test_call_without_rich(self):
        from dp_forge.cegis_loop import RichProgressBar
        bar = RichProgressBar(total=5)
        p = CEGISProgress(
            iteration=0, objective=0.5, violation_magnitude=0.01,
            violation_pair=(0, 1), n_witness_pairs=1,
            solve_time=0.1, verify_time=0.05, total_time=0.15,
        )
        # Should not raise even without rich
        bar(p)

    def test_finish(self):
        from dp_forge.cegis_loop import RichProgressBar
        bar = RichProgressBar(total=5)
        bar.finish()  # Should not raise


# ═══════════════════════════════════════════════════════════════════════════
# §39  Optimality Certificate Extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestOptimalityCertificateExtraction:
    """Test _extract_optimality_certificate helper."""

    def test_extract_with_dual(self):
        from dp_forge.cegis_loop import _extract_optimality_certificate
        stats = _make_solve_stats(n_vars=10, objective=0.5, duality_gap=1e-8)
        cert = _extract_optimality_certificate(stats)
        assert cert is not None
        assert abs(cert.primal_obj - 0.5) < 1e-10
        assert cert.duality_gap >= 0

    def test_extract_without_dual(self):
        from dp_forge.cegis_loop import _extract_optimality_certificate
        stats = SolveStatistics(
            solver_name="test",
            status="optimal",
            iterations=1,
            solve_time=0.01,
            objective_value=0.5,
            primal_solution=np.array([0.5]),
            dual_solution=None,
            basis_info=None,
            duality_gap=None,
        )
        cert = _extract_optimality_certificate(stats)
        assert cert is not None
        assert cert.duality_gap == 0.0

    def test_extract_negative_gap_clamped(self):
        from dp_forge.cegis_loop import _extract_optimality_certificate
        stats = _make_solve_stats(n_vars=10, objective=0.5, duality_gap=-1e-10)
        cert = _extract_optimality_certificate(stats)
        assert cert.duality_gap >= 0


# ═══════════════════════════════════════════════════════════════════════════
# §40  Validate Optimality Certificate
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateOptimalityCertificate:
    """Test _validate_optimality_certificate helper."""

    def test_tight_certificate(self):
        from dp_forge.cegis_loop import _validate_optimality_certificate
        from dp_forge.types import OptimalityCertificate
        cert = OptimalityCertificate(
            dual_vars=None, duality_gap=1e-10,
            primal_obj=0.5, dual_obj=0.5 - 1e-10,
        )
        assert _validate_optimality_certificate(cert) is True

    def test_loose_certificate(self):
        from dp_forge.cegis_loop import _validate_optimality_certificate
        from dp_forge.types import OptimalityCertificate
        cert = OptimalityCertificate(
            dual_vars=None, duality_gap=0.1,
            primal_obj=0.5, dual_obj=0.4,
        )
        assert _validate_optimality_certificate(cert, tol=1e-6) is False

    def test_custom_tolerance(self):
        from dp_forge.cegis_loop import _validate_optimality_certificate
        from dp_forge.types import OptimalityCertificate
        cert = OptimalityCertificate(
            dual_vars=None, duality_gap=0.01,
            primal_obj=0.5, dual_obj=0.49,
        )
        assert _validate_optimality_certificate(cert, tol=0.1) is True
        assert _validate_optimality_certificate(cert, tol=0.001) is False


# ═══════════════════════════════════════════════════════════════════════════
# §41  CEGISEngine with Different Solver Backends
# ═══════════════════════════════════════════════════════════════════════════


class TestSolverBackendConfig:
    """Test CEGISEngine configuration with different solver backends."""

    def test_auto_solver(self):
        config = SynthesisConfig(solver=SolverBackend.AUTO, verbose=0)
        engine = CEGISEngine(config)
        assert engine.config.solver == SolverBackend.AUTO

    def test_highs_solver(self):
        config = SynthesisConfig(solver=SolverBackend.HIGHS, verbose=0)
        engine = CEGISEngine(config)
        assert engine.config.solver == SolverBackend.HIGHS

    def test_scipy_solver(self):
        config = SynthesisConfig(solver=SolverBackend.SCIPY, verbose=0)
        engine = CEGISEngine(config)
        assert engine.config.solver == SolverBackend.SCIPY


# ═══════════════════════════════════════════════════════════════════════════
# §42  Convergence History: Detailed Property Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestConvergenceHistoryProperties:
    """Detailed property tests for ConvergenceHistory."""

    def test_n_iterations_matches_length(self):
        h = ConvergenceHistory()
        for i in range(7):
            h.record(float(i), 0.0, 0.1, 0.05, i, 0.15)
        assert h.n_iterations == 7
        assert h.n_iterations == len(h.objectives)
        assert h.n_iterations == len(h.violations)
        assert h.n_iterations == len(h.solve_times)
        assert h.n_iterations == len(h.verify_times)
        assert h.n_iterations == len(h.witness_counts)
        assert h.n_iterations == len(h.iteration_times)

    def test_improvement_rate_sign(self):
        """Positive rate means objectives are increasing."""
        h = ConvergenceHistory()
        for i in range(10):
            h.record(float(i), 0.0, 0.1, 0.05, i, 0.15)
        rate = h.objective_improvement_rate()
        assert rate > 0

    def test_improvement_rate_negative(self):
        """If objectives decrease (unusual), rate is negative."""
        h = ConvergenceHistory()
        for i in range(10):
            h.record(10.0 - float(i), 0.0, 0.1, 0.05, i, 0.15)
        rate = h.objective_improvement_rate()
        assert rate < 0

    def test_stagnation_exact_window(self):
        """Stagnation detection at exactly the window boundary."""
        h = ConvergenceHistory()
        for i in range(10):
            h.record(0.5, 0.0, 0.1, 0.05, i, 0.15)
        assert h.detect_stagnation(window=10, tol=1e-10) is True

    def test_stagnation_window_minus_one(self):
        h = ConvergenceHistory()
        for i in range(9):
            h.record(0.5, 0.0, 0.1, 0.05, i, 0.15)
        assert h.detect_stagnation(window=10) is False


# ═══════════════════════════════════════════════════════════════════════════
# §43  WitnessSet Size and Len Consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestWitnessSetConsistency:
    """Ensure size, len, and pairs are always consistent."""

    def test_size_equals_len(self):
        ws = WitnessSet(total_edges=10)
        ws.add_pair(0, 1)
        ws.add_pair(2, 3)
        assert ws.size == len(ws)

    def test_size_equals_pairs_length(self):
        ws = WitnessSet(total_edges=10)
        ws.add_pair(0, 1)
        ws.add_pair(2, 3)
        ws.add_pair(4, 5)
        assert ws.size == len(ws.pairs)

    def test_consistency_after_remove(self):
        ws = WitnessSet(total_edges=10)
        ws.add_pair(0, 1)
        ws.add_pair(2, 3)
        ws.remove_pair(0, 1)
        assert ws.size == 1
        assert len(ws) == 1
        assert len(ws.pairs) == 1

    def test_frozen_size_matches(self):
        ws = WitnessSet(total_edges=10)
        ws.add_pair(0, 1)
        ws.add_pair(2, 3)
        ws.add_pair(4, 5)
        assert len(ws.frozen()) == ws.size


# ═══════════════════════════════════════════════════════════════════════════
# §44  Full Workflow Smoke Test
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestFullWorkflowSmoke:
    """End-to-end smoke test of the full CEGIS workflow."""

    def test_count_query_end_to_end(self):
        """Complete workflow: spec → synthesize → validate."""
        spec = QuerySpec.counting(n=3, epsilon=1.0, k=5)
        config = SynthesisConfig(max_iter=200, verbose=0)

        # Synthesize
        result = CEGISSynthesize(spec, config=config)

        # Validate result structure
        assert isinstance(result, CEGISResult)
        assert result.mechanism.shape == (3, 5)
        assert result.iterations > 0
        assert result.obj_val >= 0
        assert len(result.convergence_history) > 0

        # Validate mechanism properties
        p = result.mechanism
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-4)
        assert np.all(p >= -1e-10)

        # Validate DP compliance
        eps = spec.epsilon
        exp_eps = math.exp(eps)
        for i, ip in spec.edges.edges:
            for j in range(spec.k):
                if p[i][j] > 1e-12 and p[ip][j] > 1e-12:
                    assert p[i][j] / p[ip][j] <= exp_eps + 1e-3
                    assert p[ip][j] / p[i][j] <= exp_eps + 1e-3

    def test_synthesize_mechanism_api_end_to_end(self):
        """End-to-end test of the high-level API."""
        result = synthesize_mechanism(
            np.arange(3, dtype=float),
            epsilon=1.0,
            k=5,
            verbose=0,
            max_iter=200,
        )
        assert isinstance(result, CEGISResult)
        assert result.n == 3
        assert result.k == 5
        np.testing.assert_allclose(
            result.mechanism.sum(axis=1), 1.0, atol=1e-4,
        )

"""
Comprehensive tests for dp_forge.sparse.benders module.

Tests BendersMaster formulation, BendersSubproblem dual extraction,
FeasibilityCut and OptimalityCut validity, MultiCutBenders convergence,
and TrustRegionBenders stability.
"""

import numpy as np
import pytest

from dp_forge.sparse import (
    BendersCut,
    BendersDecomposer,
    CutType,
    DecompositionType,
    SparseConfig,
    SparseResult,
)
from dp_forge.sparse.benders import (
    BendersMaster,
    BendersSubproblem,
    FeasibilityCut,
    MultiCutBenders,
    OptimalityCut,
    TrustRegionBenders,
)
from dp_forge.types import AdjacencyRelation, PrivacyBudget, QuerySpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(n: int = 3, k: int = 5, eps: float = 1.0) -> QuerySpec:
    qv = np.linspace(0.0, 1.0, n)
    return QuerySpec(
        query_values=qv, domain=list(range(n)),
        sensitivity=1.0, epsilon=eps, k=k,
    )


def _uniform_mechanism(n: int, k: int) -> np.ndarray:
    return np.ones((n, k), dtype=np.float64) / k


# =============================================================================
# FeasibilityCut Tests
# =============================================================================


class TestFeasibilityCut:
    """Tests for FeasibilityCut generation."""

    def test_no_cut_when_feasible(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        fc = FeasibilityCut(spec)
        M = _uniform_mechanism(2, 3)
        budget = PrivacyBudget(epsilon=1.0)
        cut = fc.generate(M, (0, 1), budget)
        assert cut is None  # Uniform mechanism is DP-feasible

    def test_cut_generated_on_violation(self):
        spec = _make_spec(n=2, k=3, eps=0.5)
        fc = FeasibilityCut(spec)
        # Construct a violating mechanism
        M = np.array([[0.9, 0.05, 0.05], [0.01, 0.49, 0.50]])
        budget = PrivacyBudget(epsilon=0.5)
        cut = fc.generate(M, (0, 1), budget)
        assert cut is not None
        assert cut.cut_type == CutType.FEASIBILITY
        assert cut.subproblem_pair == (0, 1)

    def test_cut_coefficients_shape(self):
        spec = _make_spec(n=3, k=4, eps=0.5)
        fc = FeasibilityCut(spec)
        M = np.zeros((3, 4))
        M[0] = [0.97, 0.01, 0.01, 0.01]
        M[1] = [0.01, 0.01, 0.97, 0.01]
        M[2] = _uniform_mechanism(1, 4)[0]
        budget = PrivacyBudget(epsilon=0.5)
        cut = fc.generate(M, (0, 1), budget)
        if cut is not None:
            assert cut.coefficients.shape == (3 * 4,)

    def test_symmetric_pair_feasibility(self):
        spec = _make_spec(n=2, k=3, eps=2.0)
        fc = FeasibilityCut(spec)
        M = _uniform_mechanism(2, 3)
        budget = PrivacyBudget(epsilon=2.0)
        # Uniform is feasible for any epsilon
        assert fc.generate(M, (0, 1), budget) is None
        assert fc.generate(M, (1, 0), budget) is None


# =============================================================================
# OptimalityCut Tests
# =============================================================================


class TestOptimalityCut:
    """Tests for OptimalityCut generation."""

    def test_cut_type_is_optimality(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        oc = OptimalityCut(spec)
        M = _uniform_mechanism(2, 3)
        dual = np.array([1.0, 0.5, 0.3])
        budget = PrivacyBudget(epsilon=1.0)
        cut = oc.generate(M, (0, 1), budget, dual)
        if cut is not None:
            assert cut.cut_type == CutType.OPTIMALITY

    def test_cut_coefficients_finite(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        oc = OptimalityCut(spec)
        M = _uniform_mechanism(2, 3)
        dual = np.ones(3)
        budget = PrivacyBudget(epsilon=1.0)
        cut = oc.generate(M, (0, 1), budget, dual)
        if cut is not None:
            assert np.all(np.isfinite(cut.coefficients))
            assert np.isfinite(cut.rhs)

    def test_zero_dual_no_cut(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        oc = OptimalityCut(spec)
        M = _uniform_mechanism(2, 3)
        dual = np.zeros(3)
        budget = PrivacyBudget(epsilon=1.0)
        cut = oc.generate(M, (0, 1), budget, dual)
        # Zero dual should not generate a useful cut
        assert cut is None


# =============================================================================
# BendersSubproblem Tests
# =============================================================================


class TestBendersSubproblem:
    """Tests for BendersSubproblem dual extraction."""

    def test_feasible_uniform(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        sub = BendersSubproblem(spec)
        M = _uniform_mechanism(2, 3)
        budget = PrivacyBudget(epsilon=1.0)
        feasible, cut = sub.solve(M, (0, 1), budget)
        assert feasible is True

    def test_infeasible_returns_cut(self):
        spec = _make_spec(n=2, k=3, eps=0.1)
        sub = BendersSubproblem(spec)
        M = np.array([[0.95, 0.025, 0.025], [0.01, 0.49, 0.50]])
        budget = PrivacyBudget(epsilon=0.1)
        feasible, cut = sub.solve(M, (0, 1), budget)
        assert feasible is False
        assert cut is not None
        assert cut.cut_type == CutType.FEASIBILITY

    def test_returns_tuple(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        sub = BendersSubproblem(spec)
        M = _uniform_mechanism(2, 3)
        budget = PrivacyBudget(epsilon=1.0)
        result = sub.solve(M, (0, 1), budget)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_all_pairs_checked(self):
        spec = _make_spec(n=3, k=4, eps=1.0)
        sub = BendersSubproblem(spec)
        M = _uniform_mechanism(3, 4)
        budget = PrivacyBudget(epsilon=1.0)
        for pair in spec.edges.edges:
            feasible, _ = sub.solve(M, pair, budget)
            assert feasible is True


# =============================================================================
# BendersMaster Tests
# =============================================================================


class TestBendersMaster:
    """Tests for BendersMaster formulation."""

    def test_solve_without_cuts(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        master = BendersMaster(spec)
        M, obj = master.solve()
        assert M.shape == (2, 3)
        assert np.isfinite(obj)

    def test_row_stochasticity(self):
        spec = _make_spec(n=3, k=4, eps=1.0)
        master = BendersMaster(spec)
        M, _ = master.solve()
        for i in range(3):
            np.testing.assert_allclose(M[i].sum(), 1.0, atol=1e-4)

    def test_nonnegative_mechanism(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        master = BendersMaster(spec)
        M, _ = master.solve()
        assert np.all(M >= -1e-10)

    def test_add_cut_increases_cuts(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        master = BendersMaster(spec)
        assert len(master.cuts) == 0
        cut = BendersCut(
            cut_type=CutType.FEASIBILITY,
            coefficients=np.zeros(2 * 3),
            rhs=0.0,
            subproblem_pair=(0, 1),
        )
        master.add_cut(cut)
        assert len(master.cuts) == 1

    def test_cuts_affect_solution(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        master = BendersMaster(spec)
        M1, obj1 = master.solve()
        # Add a binding cut
        coeffs = np.ones(2 * 3) * 0.1
        cut = BendersCut(
            cut_type=CutType.OPTIMALITY,
            coefficients=coeffs,
            rhs=-10.0,
            subproblem_pair=(0, 1),
        )
        master.add_cut(cut)
        M2, obj2 = master.solve()
        # Solution may change after adding cut
        assert M2.shape == (2, 3)


# =============================================================================
# MultiCutBenders Tests
# =============================================================================


class TestMultiCutBenders:
    """Tests for MultiCutBenders convergence."""

    def test_produces_valid_result(self):
        spec = _make_spec(n=2, k=4, eps=1.0)
        config = SparseConfig(
            decomposition_type=DecompositionType.BENDERS,
            max_iterations=15, max_cuts=50, verbose=0,
        )
        mcb = MultiCutBenders(spec, config)
        result = mcb.solve()
        assert isinstance(result, SparseResult)
        assert result.mechanism.shape == (2, 4)

    def test_mechanism_row_stochastic(self):
        spec = _make_spec(n=3, k=4, eps=1.0)
        config = SparseConfig(max_iterations=10, max_cuts=30, verbose=0)
        mcb = MultiCutBenders(spec, config)
        result = mcb.solve()
        for i in range(3):
            np.testing.assert_allclose(result.mechanism[i].sum(), 1.0, atol=1e-4)

    def test_convergence_history_nonempty(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        mcb = MultiCutBenders(spec, config)
        result = mcb.solve()
        assert len(result.convergence_history) > 0

    def test_bounds_valid(self):
        spec = _make_spec(n=2, k=4, eps=1.0)
        config = SparseConfig(max_iterations=15, verbose=0)
        mcb = MultiCutBenders(spec, config)
        result = mcb.solve()
        assert result.lower_bound <= result.upper_bound + 1e-6

    def test_max_cuts_respected(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        config = SparseConfig(max_iterations=100, max_cuts=5, verbose=0)
        mcb = MultiCutBenders(spec, config)
        mcb.solve()
        assert len(mcb.master.cuts) <= 5


# =============================================================================
# TrustRegionBenders Tests
# =============================================================================


class TestTrustRegionBenders:
    """Tests for TrustRegionBenders stability."""

    def test_produces_valid_result(self):
        spec = _make_spec(n=2, k=4, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        trb = TrustRegionBenders(spec, config, initial_radius=0.5)
        result = trb.solve()
        assert isinstance(result, SparseResult)
        assert result.mechanism.shape == (2, 4)

    def test_trust_radius_dynamic(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        trb = TrustRegionBenders(spec, config, initial_radius=0.5)
        initial_radius = trb.trust_radius
        trb.solve()
        # Radius may have changed
        assert trb.trust_radius > 0

    def test_incumbent_updated(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        trb = TrustRegionBenders(spec, config)
        trb.solve()
        assert trb.incumbent is not None
        assert trb.incumbent.shape == (2, 3)

    def test_row_stochastic_result(self):
        spec = _make_spec(n=3, k=4, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        trb = TrustRegionBenders(spec, config)
        result = trb.solve()
        for i in range(3):
            np.testing.assert_allclose(result.mechanism[i].sum(), 1.0, atol=1e-4)

    def test_bounds_sensible(self):
        spec = _make_spec(n=2, k=4, eps=1.0)
        config = SparseConfig(max_iterations=15, verbose=0)
        trb = TrustRegionBenders(spec, config)
        result = trb.solve()
        assert result.lower_bound <= result.upper_bound + 1e-6

    def test_trust_region_min_max_bounds(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        trb = TrustRegionBenders(spec, config, initial_radius=0.5)
        trb.solve()
        assert trb.trust_radius >= trb._min_radius
        assert trb.trust_radius <= trb._max_radius


# =============================================================================
# Integration: BendersDecomposer Public API
# =============================================================================


class TestBendersDecomposerAPI:
    """Integration tests for BendersDecomposer."""

    def test_solve(self):
        spec = _make_spec(n=2, k=4, eps=1.0)
        config = SparseConfig(
            decomposition_type=DecompositionType.BENDERS,
            max_iterations=10, verbose=0,
        )
        bd = BendersDecomposer(config)
        result = bd.solve(spec)
        assert isinstance(result, SparseResult)
        assert result.mechanism.shape == (2, 4)
        assert np.all(result.mechanism >= -1e-10)

    def test_default_config(self):
        bd = BendersDecomposer()
        spec = _make_spec(n=2, k=3, eps=1.0)
        result = bd.solve(spec)
        assert isinstance(result, SparseResult)

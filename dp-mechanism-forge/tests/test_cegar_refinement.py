"""
Tests for CEGAR refinement engine components.

Covers LazyAbstractionTree, CounterexampleAnalysis, CraigInterpolationRefiner,
RefinementStrategySelector, ConvergenceAccelerator, and RefinementEngine.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dp_forge.types import (
    AbstractDomainType,
    AbstractValue,
    AdjacencyRelation,
    Formula,
    InterpolantType,
    Predicate,
    PrivacyBudget,
)
from dp_forge.cegar import (
    AbstractCounterexample,
    AbstractState,
    CEGARConfig,
    RefinementResult,
    RefinementStrategy,
)
from dp_forge.cegar.abstraction import IntervalAbstraction
from dp_forge.cegar.refinement import (
    ConvergenceAccelerator,
    CounterexampleAnalysis,
    CraigInterpolationRefiner,
    LazyAbstractionTree,
    RefinementEngine,
    RefinementStrategySelector,
    TreeNode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_interval(lower, upper):
    return AbstractValue(
        domain_type=AbstractDomainType.INTERVAL,
        lower=np.asarray(lower, dtype=np.float64),
        upper=np.asarray(upper, dtype=np.float64),
    )


def _make_predicate(name: str, coeffs=None, rhs=0.0) -> Predicate:
    """Create a simple linear predicate."""
    if coeffs is None:
        coeffs = [1.0]
    return Predicate(
        name=name,
        formula=Formula(
            expr=f"{name}",
            variables=frozenset({f"x{i}" for i in range(len(coeffs))}),
            formula_type="linear_arithmetic",
            metadata={"type": "linear_le", "coeffs": coeffs, "rhs": rhs},
        ),
        is_atomic=True,
    )


def _make_counterexample(pair, magnitude, n_states=2):
    """Create an AbstractCounterexample with dummy trace."""
    trace = [
        AbstractState(
            state_id=i,
            predicates=frozenset(),
            abstract_value=_make_interval([0.0], [1.0]),
        )
        for i in range(n_states)
    ]
    return AbstractCounterexample(
        trace=trace,
        violating_pair=pair,
        violation_magnitude=magnitude,
    )


def _make_mechanism(n=3, k=2, eps=1.0):
    """Create a simple randomized response mechanism."""
    p = math.exp(eps) / (1 + math.exp(eps))
    mech = np.full((n, k), (1 - p) / (k - 1 if k > 1 else 1))
    for i in range(n):
        mech[i, i % k] = p
    return mech


# ===================================================================
# LazyAbstractionTree tests
# ===================================================================


class TestLazyAbstractionTree:
    """Tests for lazy abstraction refinement tree."""

    def setup_method(self):
        self.domain = IntervalAbstraction()
        self.tree = LazyAbstractionTree(self.domain)

    def test_initialize_creates_root(self):
        val = _make_interval([0.0, 0.0], [1.0, 1.0])
        root_id = self.tree.initialize(val)
        assert self.tree.num_nodes == 1
        assert self.tree.root is not None
        assert self.tree.root.node_id == root_id

    def test_split_node_creates_children(self):
        val = _make_interval([0.0], [1.0])
        root_id = self.tree.initialize(val)
        pred = _make_predicate("p1", [1.0], 0.5)
        true_val = _make_interval([0.0], [0.5])
        false_val = _make_interval([0.5], [1.0])
        true_id, false_id = self.tree.split_node(root_id, pred, true_val, false_val)
        assert self.tree.num_nodes == 3
        assert true_id != false_id
        true_node = self.tree.get_node(true_id)
        assert true_node is not None
        assert pred in true_node.predicates

    def test_split_preserves_parent(self):
        val = _make_interval([0.0], [1.0])
        root_id = self.tree.initialize(val)
        pred = _make_predicate("p1")
        true_val = _make_interval([0.0], [0.5])
        false_val = _make_interval([0.5], [1.0])
        true_id, false_id = self.tree.split_node(root_id, pred, true_val, false_val)
        true_node = self.tree.get_node(true_id)
        assert true_node.parent == root_id

    def test_path_to_root(self):
        val = _make_interval([0.0], [1.0])
        root_id = self.tree.initialize(val)
        pred = _make_predicate("p1")
        true_val = _make_interval([0.0], [0.5])
        false_val = _make_interval([0.5], [1.0])
        true_id, _ = self.tree.split_node(root_id, pred, true_val, false_val)
        path = self.tree.get_path_to_root(true_id)
        assert path == [true_id, root_id]

    def test_get_leaf_nodes(self):
        val = _make_interval([0.0], [1.0])
        root_id = self.tree.initialize(val)
        pred = _make_predicate("p1")
        true_val = _make_interval([0.0], [0.5])
        false_val = _make_interval([0.5], [1.0])
        self.tree.split_node(root_id, pred, true_val, false_val)
        leaves = self.tree.get_leaf_nodes()
        assert len(leaves) == 2

    def test_find_covering_node(self):
        val = _make_interval([0.0], [10.0])
        root_id = self.tree.initialize(val)
        # Root covers everything, so looking for a cover of root from root itself returns None
        covering = self.tree.find_covering_node(root_id)
        assert covering is None

    def test_refine_along_path(self):
        val = _make_interval([0.0, 0.0], [1.0, 1.0])
        root_id = self.tree.initialize(val)
        pred = _make_predicate("refine_pred", [1.0, 0.0], 0.5)
        splits = self.tree.refine_along_path([root_id], [pred], self.domain)
        assert splits >= 1
        assert self.tree.num_nodes > 1

    def test_depth_increases_on_split(self):
        val = _make_interval([0.0], [1.0])
        root_id = self.tree.initialize(val)
        pred = _make_predicate("p1")
        true_val = _make_interval([0.0], [0.5])
        false_val = _make_interval([0.5], [1.0])
        true_id, _ = self.tree.split_node(root_id, pred, true_val, false_val)
        true_node = self.tree.get_node(true_id)
        assert true_node.depth == 1

    def test_empty_tree(self):
        assert self.tree.num_nodes == 0
        assert self.tree.root is None

    def test_multiple_splits(self):
        val = _make_interval([0.0, 0.0], [4.0, 4.0])
        root_id = self.tree.initialize(val)
        pred1 = _make_predicate("p1", [1.0, 0.0], 2.0)
        true_val = _make_interval([0.0, 0.0], [2.0, 4.0])
        false_val = _make_interval([2.0, 0.0], [4.0, 4.0])
        true_id, _ = self.tree.split_node(root_id, pred1, true_val, false_val)
        pred2 = _make_predicate("p2", [0.0, 1.0], 2.0)
        true_val2 = _make_interval([0.0, 0.0], [2.0, 2.0])
        false_val2 = _make_interval([0.0, 2.0], [2.0, 4.0])
        self.tree.split_node(true_id, pred2, true_val2, false_val2)
        assert self.tree.num_nodes == 5
        leaves = self.tree.get_leaf_nodes()
        assert len(leaves) == 3


# ===================================================================
# CounterexampleAnalysis tests
# ===================================================================


class TestCounterexampleAnalysis:
    """Tests for counterexample classification."""

    def setup_method(self):
        self.analyzer = CounterexampleAnalysis()

    def test_genuine_counterexample(self):
        """A mechanism that truly violates DP should be classified as genuine."""
        mech = np.array([[0.99, 0.01], [0.01, 0.99]])
        budget = PrivacyBudget(epsilon=0.1)
        ce = _make_counterexample((0, 1), 5.0)
        is_genuine, actual = self.analyzer.classify_counterexample(ce, mech, budget)
        assert is_genuine
        assert actual > budget.epsilon

    def test_spurious_counterexample(self):
        """A mechanism that satisfies DP should have spurious counterexamples."""
        eps = 1.0
        mech = _make_mechanism(n=2, k=2, eps=eps)
        budget = PrivacyBudget(epsilon=eps)
        ce = _make_counterexample((0, 1), eps + 1.0)
        is_genuine, actual = self.analyzer.classify_counterexample(ce, mech, budget)
        assert not is_genuine

    def test_feasibility_check(self):
        mech = np.array([[0.9, 0.1], [0.1, 0.9]])
        ce = _make_counterexample((0, 1), 2.0)
        # log(0.9/0.1) ≈ 2.197 >= 2.0
        is_feasible = self.analyzer.check_feasibility(ce, mech)
        assert is_feasible

    def test_out_of_bounds_pair(self):
        mech = np.array([[0.5, 0.5], [0.5, 0.5]])
        ce = _make_counterexample((10, 11), 1.0)
        is_genuine, actual = self.analyzer.classify_counterexample(
            ce, mech, PrivacyBudget(epsilon=1.0)
        )
        assert not is_genuine

    def test_extract_concrete_witness(self):
        mech = np.array([[0.9, 0.1], [0.1, 0.9]])
        ce = _make_counterexample((0, 1), 2.0)
        witness = self.analyzer.extract_concrete_witness(ce, mech)
        assert witness is not None
        i, ip, j, mag = witness
        assert i == 0 and ip == 1
        assert mag > 0

    def test_extract_witness_out_of_range(self):
        mech = np.array([[0.5, 0.5]])
        ce = _make_counterexample((5, 6), 1.0)
        witness = self.analyzer.extract_concrete_witness(ce, mech)
        assert witness is None

    def test_analyze_spuriousness_cause(self):
        mech = np.array([[0.7, 0.3], [0.3, 0.7]])
        ce = _make_counterexample((0, 1), 10.0)  # Hugely inflated
        info = self.analyzer.analyze_spuriousness_cause(ce, mech)
        assert "gap" in info
        assert info["gap"] > 0  # Claimed > actual

    @pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 2.0])
    def test_rr_not_genuine_at_correct_eps(self, eps):
        mech = _make_mechanism(n=2, k=2, eps=eps)
        budget = PrivacyBudget(epsilon=eps)
        ce = _make_counterexample((0, 1), eps + 1.0)
        is_genuine, _ = self.analyzer.classify_counterexample(ce, mech, budget)
        assert not is_genuine


# ===================================================================
# CraigInterpolationRefiner tests
# ===================================================================


class TestCraigInterpolationRefiner:
    """Tests for Craig interpolation-based refinement."""

    def setup_method(self):
        self.refiner = CraigInterpolationRefiner()

    def test_compute_interpolants_two_states(self):
        trace = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        mech = np.eye(2)
        interpolants = self.refiner.compute_interpolants(trace, mech)
        assert len(interpolants) == 1
        assert interpolants[0].formula_type == "linear_arithmetic"

    def test_compute_interpolants_three_states(self):
        trace = [np.array([0.0]), np.array([1.0]), np.array([2.0])]
        mech = np.eye(2)
        interpolants = self.refiner.compute_interpolants(trace, mech)
        assert len(interpolants) == 2

    def test_interpolant_separates_states(self):
        trace = [np.array([0.0, 0.0]), np.array([2.0, 2.0])]
        mech = np.eye(2)
        interpolants = self.refiner.compute_interpolants(trace, mech)
        assert len(interpolants) == 1
        meta = interpolants[0].metadata
        assert "coeffs" in meta
        assert "rhs" in meta

    def test_refine_from_interpolants(self):
        trace = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        mech = np.eye(2)
        interpolants = self.refiner.compute_interpolants(trace, mech)
        predicates = self.refiner.refine_from_interpolants(interpolants)
        assert len(predicates) >= 1
        assert all(p.is_atomic for p in predicates)

    def test_refine_counterexample(self):
        mech = np.array([[0.8, 0.2], [0.2, 0.8]])
        ce = _make_counterexample((0, 1), 2.0)
        result = self.refiner.refine(ce, mech)
        assert isinstance(result, RefinementResult)
        assert result.refinement_type == RefinementStrategy.INTERPOLATION

    def test_degenerate_identical_states(self):
        trace = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
        mech = np.eye(2)
        interpolants = self.refiner.compute_interpolants(trace, mech)
        assert len(interpolants) == 1
        assert interpolants[0].expr == "true"

    def test_refinement_quality(self):
        """Predicates from interpolation should be non-trivial."""
        trace = [np.array([0.3, 0.7]), np.array([0.7, 0.3])]
        mech = np.array([[0.3, 0.7], [0.7, 0.3]])
        interpolants = self.refiner.compute_interpolants(trace, mech)
        predicates = self.refiner.refine_from_interpolants(interpolants)
        assert len(predicates) >= 1
        # Predicate should reference some variables
        for p in predicates:
            assert len(p.formula.variables) > 0


# ===================================================================
# RefinementStrategySelector tests
# ===================================================================


class TestRefinementStrategy:
    """Tests for refinement strategy selection and application."""

    def test_default_strategy(self):
        config = CEGARConfig(refinement_strategy=RefinementStrategy.INTERPOLATION)
        selector = RefinementStrategySelector(config=config)
        ce = _make_counterexample((0, 1), 1.0)
        strategy = selector.select_strategy(ce, iteration=0)
        assert strategy == RefinementStrategy.INTERPOLATION

    def test_adaptive_to_impact_after_many_iterations(self):
        config = CEGARConfig(refinement_strategy=RefinementStrategy.INTERPOLATION)
        selector = RefinementStrategySelector(config=config)
        ce = _make_counterexample((0, 1), 1.0)
        strategy = selector.select_strategy(ce, iteration=25)
        assert strategy == RefinementStrategy.IMPACT

    def test_small_violation_uses_wp(self):
        config = CEGARConfig(refinement_strategy=RefinementStrategy.INTERPOLATION)
        selector = RefinementStrategySelector(config=config)
        ce = _make_counterexample((0, 1), 1e-8)
        strategy = selector.select_strategy(ce, iteration=0)
        assert strategy == RefinementStrategy.WEAKEST_PRECONDITION

    @pytest.mark.parametrize("strategy_enum", [
        RefinementStrategy.INTERPOLATION,
        RefinementStrategy.WEAKEST_PRECONDITION,
        RefinementStrategy.STRONGEST_POSTCONDITION,
        RefinementStrategy.IMPACT,
        RefinementStrategy.LAZY,
    ])
    def test_all_strategies_produce_results(self, strategy_enum):
        config = CEGARConfig()
        selector = RefinementStrategySelector(config=config)
        mech = np.array([[0.8, 0.2], [0.2, 0.8]])
        budget = PrivacyBudget(epsilon=1.0)
        ce = _make_counterexample((0, 1), 2.0)
        result = selector.refine(ce, mech, budget, strategy=strategy_enum)
        assert isinstance(result, RefinementResult)
        assert result.refinement_type == strategy_enum

    def test_refinement_count_increments(self):
        selector = RefinementStrategySelector()
        mech = np.array([[0.8, 0.2], [0.2, 0.8]])
        budget = PrivacyBudget(epsilon=1.0)
        ce = _make_counterexample((0, 1), 2.0)
        assert selector.refinement_count == 0
        selector.refine(ce, mech, budget)
        assert selector.refinement_count == 1
        selector.refine(ce, mech, budget)
        assert selector.refinement_count == 2


# ===================================================================
# ConvergenceAccelerator tests
# ===================================================================


class TestConvergenceAccelerator:
    """Tests for convergence acceleration."""

    def setup_method(self):
        self.domain = IntervalAbstraction()
        self.accel = ConvergenceAccelerator(self.domain)

    def test_should_widen_early_iterations(self):
        assert not self.accel.should_widen(0)
        assert not self.accel.should_widen(2)
        assert self.accel.should_widen(3)
        assert self.accel.should_widen(10)

    def test_should_not_widen_if_stabilized(self):
        assert not self.accel.should_widen(10, stabilized=True)

    def test_accelerate_early_uses_join(self):
        a = _make_interval([0.0], [1.0])
        b = _make_interval([0.5], [1.5])
        result = self.accel.accelerate(a, b, iteration=1)
        # Should use join, not widen
        np.testing.assert_allclose(result.lower, [0.0])
        np.testing.assert_allclose(result.upper, [1.5])

    def test_accelerate_late_uses_widen(self):
        a = _make_interval([1.0], [2.0])
        b = _make_interval([0.5], [2.5])
        result = self.accel.accelerate(a, b, iteration=5)
        # After iteration 3, widening with thresholds should be used
        # Lower decreased: should use threshold or -inf
        assert result.lower[0] <= 0.5 + 1e-9

    def test_check_convergence_when_stable(self):
        a = _make_interval([0.0], [1.0])
        b = _make_interval([0.1], [0.9])
        assert self.accel.check_convergence(a, b)  # b ⊆ a

    def test_check_convergence_when_growing(self):
        a = _make_interval([0.5], [0.5])
        b = _make_interval([0.0], [1.0])
        assert not self.accel.check_convergence(a, b)  # b ⊄ a

    def test_narrow_result(self):
        widened = _make_interval([-np.inf], [np.inf])
        precise = _make_interval([0.0], [10.0])
        result = self.accel.narrow_result(widened, precise)
        assert result.lower[0] >= 0.0 - 1e-9
        assert result.upper[0] <= 10.0 + 1e-9

    def test_custom_thresholds(self):
        thresholds = np.array([0.0, 1.0, 5.0, 10.0])
        accel = ConvergenceAccelerator(self.domain, thresholds=thresholds)
        assert accel.should_widen(5)


# ===================================================================
# RefinementEngine tests (full loop)
# ===================================================================


class TestRefinementEngine:
    """Tests for integrated refinement engine."""

    def test_analyze_genuine(self):
        engine = RefinementEngine()
        mech = np.array([[0.99, 0.01], [0.01, 0.99]])
        budget = PrivacyBudget(epsilon=0.1)
        ce = _make_counterexample((0, 1), 5.0)
        is_genuine, result = engine.analyze_and_refine(ce, mech, budget)
        assert is_genuine
        assert result is None

    def test_analyze_spurious(self):
        eps = 1.0
        engine = RefinementEngine()
        mech = _make_mechanism(n=2, k=2, eps=eps)
        budget = PrivacyBudget(epsilon=eps)
        # Create counterexample with distinct abstract values so interpolation works
        trace = [
            AbstractState(
                state_id=0,
                predicates=frozenset(),
                abstract_value=_make_interval([0.0, 0.0], [0.5, 0.5]),
            ),
            AbstractState(
                state_id=1,
                predicates=frozenset(),
                abstract_value=_make_interval([0.5, 0.5], [1.0, 1.0]),
            ),
        ]
        ce = AbstractCounterexample(
            trace=trace,
            violating_pair=(0, 1),
            violation_magnitude=eps + 1.0,
        )
        is_genuine, result = engine.analyze_and_refine(ce, mech, budget)
        assert not is_genuine
        assert result is not None

    def test_engine_with_tree(self):
        engine = RefinementEngine()
        val = _make_interval([0.0, 0.0], [1.0, 1.0])
        engine.initialize_tree(val)
        mech = _make_mechanism(n=2, k=2, eps=1.0)
        budget = PrivacyBudget(epsilon=1.0)
        ce = _make_counterexample((0, 1), 2.0)
        is_genuine, result = engine.analyze_and_refine(ce, mech, budget)
        assert not is_genuine

    def test_refinement_count_tracking(self):
        engine = RefinementEngine()
        assert engine.refinement_count == 0
        mech = _make_mechanism(n=2, k=2, eps=1.0)
        budget = PrivacyBudget(epsilon=1.0)
        ce = _make_counterexample((0, 1), 2.0)
        engine.analyze_and_refine(ce, mech, budget)
        assert engine.refinement_count == 1

    def test_predicates_accumulated(self):
        engine = RefinementEngine()
        mech = _make_mechanism(n=2, k=2, eps=1.0)
        budget = PrivacyBudget(epsilon=1.0)
        ce1 = _make_counterexample((0, 1), 2.0)
        engine.analyze_and_refine(ce1, mech, budget)
        n1 = len(engine.predicates)
        ce2 = _make_counterexample((0, 1), 3.0)
        engine.analyze_and_refine(ce2, mech, budget)
        assert len(engine.predicates) >= n1

    @pytest.mark.parametrize("n_rows", [2, 3, 5])
    def test_engine_various_mechanism_sizes(self, n_rows):
        engine = RefinementEngine()
        mech = _make_mechanism(n=n_rows, k=2, eps=1.0)
        budget = PrivacyBudget(epsilon=1.0)
        for i in range(n_rows - 1):
            ce = _make_counterexample((i, i + 1), 2.0)
            engine.analyze_and_refine(ce, mech, budget)
        assert engine.refinement_count >= n_rows - 1

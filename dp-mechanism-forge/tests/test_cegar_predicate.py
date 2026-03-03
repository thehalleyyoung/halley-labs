"""
Tests for CEGAR predicate abstraction components.

Covers PredicateDiscovery, CartesianAbstraction, BooleanAbstraction,
and PredicateEvaluator.
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
    Predicate,
    PrivacyBudget,
)
from dp_forge.cegar import (
    AbstractCounterexample,
    AbstractState,
    RefinementResult,
    RefinementStrategy,
)
from dp_forge.cegar.predicate_abstraction import (
    BooleanAbstraction,
    CartesianAbstraction,
    PredicateDiscovery,
    PredicateEvaluator,
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


def _make_linear_predicate(name: str, coeffs, rhs: float) -> Predicate:
    """Create a linear_le predicate a·x <= b."""
    return Predicate(
        name=name,
        formula=Formula(
            expr=f"{name}",
            variables=frozenset({f"x{i}" for i in range(len(coeffs))}),
            formula_type="linear_arithmetic",
            metadata={"type": "linear_le", "coeffs": list(coeffs), "rhs": rhs},
        ),
        is_atomic=True,
    )


def _make_counterexample(pair, magnitude):
    trace = [
        AbstractState(
            state_id=i,
            predicates=frozenset(),
            abstract_value=_make_interval([0.0], [1.0]),
        )
        for i in range(2)
    ]
    return AbstractCounterexample(
        trace=trace,
        violating_pair=pair,
        violation_magnitude=magnitude,
    )


# ===================================================================
# PredicateEvaluator tests
# ===================================================================


class TestPredicateEvaluator:
    """Tests for predicate evaluation correctness."""

    def setup_method(self):
        self.eval = PredicateEvaluator()

    def test_linear_le_true(self):
        pred = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        state = np.array([0.3, 0.7])
        assert self.eval.evaluate(pred, state)

    def test_linear_le_false(self):
        pred = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        state = np.array([0.8, 0.2])
        assert not self.eval.evaluate(pred, state)

    def test_linear_le_boundary(self):
        pred = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        state = np.array([0.5, 0.5])
        assert self.eval.evaluate(pred, state)

    def test_multi_dim_predicate(self):
        pred = _make_linear_predicate("p2", [1.0, 1.0], 1.0)
        assert self.eval.evaluate(pred, np.array([0.3, 0.5]))
        assert not self.eval.evaluate(pred, np.array([0.6, 0.6]))

    def test_ratio_predicate(self):
        pred = Predicate(
            name="ratio_pred",
            formula=Formula(
                expr="ratio",
                variables=frozenset({"x0", "x1"}),
                formula_type="linear_arithmetic",
                metadata={"type": "ratio_le", "i": 0, "j": 1, "bound": 2.0},
            ),
            is_atomic=True,
        )
        assert self.eval.evaluate(pred, np.array([0.4, 0.3]))  # 0.4/0.3 ≈ 1.33 <= 2.0
        assert not self.eval.evaluate(pred, np.array([0.9, 0.1]))  # 9.0 > 2.0

    def test_ge_zero_predicate(self):
        pred = Predicate(
            name="ge_pred",
            formula=Formula(
                expr="ge_zero",
                variables=frozenset({"x0"}),
                formula_type="linear_arithmetic",
                metadata={"type": "ge_zero", "index": 0},
            ),
            is_atomic=True,
        )
        assert self.eval.evaluate(pred, np.array([0.5, -0.1]))
        assert not self.eval.evaluate(pred, np.array([-0.5, 0.1]))

    def test_le_one_predicate(self):
        pred = Predicate(
            name="le_pred",
            formula=Formula(
                expr="le_one",
                variables=frozenset({"x0"}),
                formula_type="linear_arithmetic",
                metadata={"type": "le_one", "index": 0},
            ),
            is_atomic=True,
        )
        assert self.eval.evaluate(pred, np.array([0.5]))
        assert not self.eval.evaluate(pred, np.array([1.5]))

    def test_evaluate_all(self):
        p1 = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        p2 = _make_linear_predicate("p2", [0.0, 1.0], 0.5)
        state = np.array([0.3, 0.7])
        satisfied = self.eval.evaluate_all([p1, p2], state)
        assert "p1" in satisfied
        assert "p2" not in satisfied

    def test_expression_evaluation_le(self):
        pred = Predicate(
            name="expr_pred",
            formula=Formula(
                expr="x0 <= 0.5",
                variables=frozenset({"x0"}),
                formula_type="linear_arithmetic",
                metadata={},
            ),
            is_atomic=True,
        )
        assert self.eval.evaluate(pred, np.array([0.3]))
        assert not self.eval.evaluate(pred, np.array([0.8]))

    def test_expression_evaluation_ge(self):
        pred = Predicate(
            name="expr_pred2",
            formula=Formula(
                expr="x0 >= 0.5",
                variables=frozenset({"x0"}),
                formula_type="linear_arithmetic",
                metadata={},
            ),
            is_atomic=True,
        )
        assert self.eval.evaluate(pred, np.array([0.8]))
        assert not self.eval.evaluate(pred, np.array([0.3]))

    @pytest.mark.parametrize("val,expected", [
        (0.0, True),
        (0.5, True),
        (0.50001, False),
        (1.0, False),
    ])
    def test_parametrized_threshold(self, val, expected):
        pred = _make_linear_predicate("thresh", [1.0], 0.5)
        result = self.eval.evaluate(pred, np.array([val]))
        assert result == expected


# ===================================================================
# PredicateDiscovery tests
# ===================================================================


class TestPredicateDiscovery:
    """Tests for predicate generation from counterexamples."""

    def setup_method(self):
        self.discovery = PredicateDiscovery()

    def test_discover_from_counterexample(self):
        mech = np.array([[0.8, 0.2], [0.2, 0.8]])
        budget = PrivacyBudget(epsilon=1.0)
        ce = _make_counterexample((0, 1), 2.0)
        preds = self.discovery.discover_from_counterexample(ce, mech, budget)
        assert len(preds) > 0

    def test_discovered_predicates_are_unique(self):
        mech = np.array([[0.8, 0.2], [0.2, 0.8]])
        budget = PrivacyBudget(epsilon=1.0)
        ce1 = _make_counterexample((0, 1), 2.0)
        ce2 = _make_counterexample((0, 1), 2.0)
        preds1 = self.discovery.discover_from_counterexample(ce1, mech, budget)
        preds2 = self.discovery.discover_from_counterexample(ce2, mech, budget)
        all_names = [p.name for p in preds1] + [p.name for p in preds2]
        assert len(all_names) == len(set(all_names))

    def test_discover_from_mechanism(self):
        mech = np.array([[0.7, 0.3], [0.3, 0.7]])
        adj = AdjacencyRelation.hamming_distance_1(2)
        budget = PrivacyBudget(epsilon=1.0)
        preds = self.discovery.discover_from_mechanism(mech, adj, budget)
        assert len(preds) > 0

    def test_discover_max_predicates(self):
        mech = np.array([[0.7, 0.3], [0.3, 0.7]])
        adj = AdjacencyRelation.hamming_distance_1(2)
        budget = PrivacyBudget(epsilon=1.0)
        preds = self.discovery.discover_from_mechanism(mech, adj, budget, max_predicates=3)
        assert len(preds) <= 3

    def test_discover_interpolation_predicates(self):
        pre = np.array([0.3, 0.7])
        post = np.array([0.7, 0.3])
        budget = PrivacyBudget(epsilon=1.0)
        preds = self.discovery.discover_interpolation_predicates(pre, post, budget)
        assert len(preds) >= 1
        # Should have linear_le type metadata
        for p in preds:
            assert "type" in p.formula.metadata

    def test_discover_interpolation_identical_states(self):
        state = np.array([0.5, 0.5])
        budget = PrivacyBudget(epsilon=1.0)
        preds = self.discovery.discover_interpolation_predicates(state, state, budget)
        assert len(preds) == 0

    def test_discovered_predicates_accumulate(self):
        mech = np.array([[0.8, 0.2], [0.2, 0.8]])
        budget = PrivacyBudget(epsilon=1.0)
        ce = _make_counterexample((0, 1), 2.0)
        self.discovery.discover_from_counterexample(ce, mech, budget)
        n_initial = len(self.discovery.discovered_predicates)
        assert n_initial > 0
        # Discover from mechanism should add more
        adj = AdjacencyRelation.hamming_distance_1(2)
        self.discovery.discover_from_mechanism(mech, adj, budget)
        assert len(self.discovery.discovered_predicates) >= n_initial


# ===================================================================
# CartesianAbstraction tests
# ===================================================================


class TestCartesianAbstraction:
    """Tests for Cartesian predicate abstraction."""

    def test_empty_predicates(self):
        ca = CartesianAbstraction()
        assert ca.num_predicates == 0

    def test_add_predicate(self):
        ca = CartesianAbstraction()
        pred = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        ca.add_predicate(pred)
        assert ca.num_predicates == 1

    def test_add_duplicate_predicate(self):
        ca = CartesianAbstraction()
        pred = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        ca.add_predicate(pred)
        splits = ca.add_predicate(pred)
        assert splits == 0
        assert ca.num_predicates == 1

    def test_abstract_state_creation(self):
        pred = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        ca = CartesianAbstraction(predicates=[pred])
        state = ca.abstract_state(np.array([0.3, 0.7]), concrete_index=0)
        assert isinstance(state, AbstractState)
        assert "p1" in state.predicates  # 0.3 <= 0.5

    def test_different_concrete_different_abstract(self):
        pred = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        ca = CartesianAbstraction(predicates=[pred])
        s1 = ca.abstract_state(np.array([0.3, 0.0]), concrete_index=0)
        s2 = ca.abstract_state(np.array([0.8, 0.0]), concrete_index=1)
        assert s1.state_id != s2.state_id

    def test_same_truth_value_same_state(self):
        pred = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        ca = CartesianAbstraction(predicates=[pred])
        s1 = ca.abstract_state(np.array([0.3, 0.0]), concrete_index=0)
        s2 = ca.abstract_state(np.array([0.4, 0.0]), concrete_index=1)
        assert s1.state_id == s2.state_id

    def test_cartesian_product(self):
        p1 = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        p2 = _make_linear_predicate("p2", [0.0, 1.0], 0.5)
        ca = CartesianAbstraction(predicates=[p1, p2])
        # (T, T)
        s1 = ca.abstract_state(np.array([0.3, 0.3]))
        assert "p1" in s1.predicates and "p2" in s1.predicates
        # (T, F)
        s2 = ca.abstract_state(np.array([0.3, 0.8]))
        assert "p1" in s2.predicates and "p2" not in s2.predicates
        # (F, T)
        s3 = ca.abstract_state(np.array([0.8, 0.3]))
        assert "p1" not in s3.predicates and "p2" in s3.predicates
        # (F, F)
        s4 = ca.abstract_state(np.array([0.8, 0.8]))
        assert "p1" not in s4.predicates and "p2" not in s4.predicates
        # All four should be distinct
        ids = {s1.state_id, s2.state_id, s3.state_id, s4.state_id}
        assert len(ids) == 4

    def test_check_satisfiability_found(self):
        pred = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        ca = CartesianAbstraction(predicates=[pred])
        mech = np.array([[0.3, 0.7], [0.8, 0.2]])
        result = ca.check_satisfiability(frozenset({"p1"}), frozenset(), mech)
        assert result is not None
        assert result[0] <= 0.5 + 1e-9

    def test_check_satisfiability_not_found(self):
        pred = _make_linear_predicate("p1", [1.0, 0.0], 0.1)
        ca = CartesianAbstraction(predicates=[pred])
        mech = np.array([[0.5, 0.5], [0.8, 0.2]])
        result = ca.check_satisfiability(frozenset({"p1"}), frozenset(), mech)
        assert result is None

    def test_refine_from_counterexample(self):
        ca = CartesianAbstraction()
        ce = _make_counterexample((0, 1), 1.0)
        mech = np.array([[0.8, 0.2], [0.2, 0.8]])
        new_pred = _make_linear_predicate("new_pred", [1.0, 0.0], 0.5)
        result = ca.refine_from_counterexample(ce, mech, [new_pred])
        assert isinstance(result, RefinementResult)
        assert ca.num_predicates == 1

    @pytest.mark.parametrize("n_preds", [1, 2, 3, 5])
    def test_num_abstract_states_bounded(self, n_preds):
        preds = [_make_linear_predicate(f"p{i}", [1.0, 0.0], 0.1 * (i + 1))
                 for i in range(n_preds)]
        ca = CartesianAbstraction(predicates=preds)
        mech = np.array([[0.05, 0.95], [0.15, 0.85], [0.25, 0.75],
                         [0.35, 0.65], [0.45, 0.55], [0.55, 0.45]])
        for i in range(mech.shape[0]):
            ca.abstract_state(mech[i], concrete_index=i)
        # At most 2^n_preds abstract states
        assert ca.num_abstract_states <= 2 ** n_preds


# ===================================================================
# BooleanAbstraction tests
# ===================================================================


class TestBooleanAbstraction:
    """Tests for full Boolean predicate abstraction."""

    def test_empty_predicates(self):
        ba = BooleanAbstraction()
        assert ba.num_predicates == 0

    def test_add_predicate(self):
        ba = BooleanAbstraction()
        pred = _make_linear_predicate("p1", [1.0], 0.5)
        ba.add_predicate(pred)
        assert ba.num_predicates == 1

    def test_add_duplicate_no_effect(self):
        ba = BooleanAbstraction()
        pred = _make_linear_predicate("p1", [1.0], 0.5)
        ba.add_predicate(pred)
        splits = ba.add_predicate(pred)
        assert splits == 0
        assert ba.num_predicates == 1

    def test_max_predicates_limit(self):
        ba = BooleanAbstraction(max_predicates=3)
        for i in range(3):
            ba.add_predicate(_make_linear_predicate(f"p{i}", [1.0], 0.1 * (i + 1)))
        with pytest.raises(ValueError, match="at limit"):
            ba.add_predicate(_make_linear_predicate("p4", [1.0], 0.5))

    def test_abstract_state(self):
        pred = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        ba = BooleanAbstraction(predicates=[pred])
        s = ba.abstract_state(np.array([0.3, 0.7]))
        assert "p1" in s.predicates

    def test_full_boolean_lattice(self):
        p1 = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        p2 = _make_linear_predicate("p2", [0.0, 1.0], 0.5)
        ba = BooleanAbstraction(predicates=[p1, p2])
        # Create states for all 4 truth assignments
        states = []
        for x0 in [0.3, 0.8]:
            for x1 in [0.3, 0.8]:
                s = ba.abstract_state(np.array([x0, x1]))
                states.append(s)
        ids = {s.state_id for s in states}
        assert len(ids) == 4

    def test_enumerate_reachable_states(self):
        pred = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        ba = BooleanAbstraction(predicates=[pred])
        mech = np.array([[0.3, 0.7], [0.8, 0.2]])
        states = ba.enumerate_reachable_states(mech)
        assert len(states) == 2

    def test_check_implication_true(self):
        p1 = _make_linear_predicate("p1", [1.0, 0.0], 0.3)
        p2 = _make_linear_predicate("p2", [1.0, 0.0], 0.5)
        ba = BooleanAbstraction(predicates=[p1, p2])
        # p1: x0 <= 0.3, p2: x0 <= 0.5
        # p1 implies p2 (if x0 <= 0.3, then x0 <= 0.5)
        mech = np.array([[0.1, 0.9], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4]])
        result = ba.check_implication(frozenset({"p1"}), "p2", mech)
        assert result

    def test_check_implication_false(self):
        p1 = _make_linear_predicate("p1", [1.0, 0.0], 0.5)
        p2 = _make_linear_predicate("p2", [1.0, 0.0], 0.3)
        ba = BooleanAbstraction(predicates=[p1, p2])
        mech = np.array([[0.1, 0.9], [0.4, 0.6], [0.6, 0.4]])
        result = ba.check_implication(frozenset({"p1"}), "p2", mech)
        assert not result

    def test_concrete_states_tracked(self):
        pred = _make_linear_predicate("p1", [1.0], 0.5)
        ba = BooleanAbstraction(predicates=[pred])
        s1 = ba.abstract_state(np.array([0.3]), concrete_index=0)
        s2 = ba.abstract_state(np.array([0.4]), concrete_index=1)
        # Both should map to same abstract state and track both
        assert s1.state_id == s2.state_id
        assert 0 in s2.concrete_states
        assert 1 in s2.concrete_states

    @pytest.mark.parametrize("n_preds", [1, 2, 3])
    def test_state_count_with_mechanism(self, n_preds):
        preds = [_make_linear_predicate(f"p{i}", [1.0, 0.0], 0.2 * (i + 1))
                 for i in range(n_preds)]
        ba = BooleanAbstraction(predicates=preds)
        mech = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]])
        states = ba.enumerate_reachable_states(mech)
        assert len(states) <= 2 ** n_preds


# ===================================================================
# Edge cases
# ===================================================================


class TestPredicateEdgeCases:
    """Edge case tests for predicate abstraction."""

    def test_evaluator_true_expression(self):
        pred = Predicate(
            name="true_pred",
            formula=Formula(
                expr="true",
                variables=frozenset(),
                formula_type="boolean",
                metadata={},
            ),
            is_atomic=True,
        )
        ev = PredicateEvaluator()
        assert ev.evaluate(pred, np.array([0.5, 0.5]))

    def test_empty_mechanism_discovery(self):
        disc = PredicateDiscovery()
        mech = np.array([[0.5, 0.5]])
        adj = AdjacencyRelation(edges=[], n=1, symmetric=True)
        budget = PrivacyBudget(epsilon=1.0)
        preds = disc.discover_from_mechanism(mech, adj, budget)
        assert isinstance(preds, list)

    def test_cartesian_no_predicates_same_state(self):
        ca = CartesianAbstraction()
        s1 = ca.abstract_state(np.array([0.3, 0.7]))
        s2 = ca.abstract_state(np.array([0.8, 0.2]))
        # With no predicates, everything maps to same abstract state
        assert s1.state_id == s2.state_id

    def test_boolean_single_predicate_two_states(self):
        pred = _make_linear_predicate("p1", [1.0], 0.5)
        ba = BooleanAbstraction(predicates=[pred])
        s1 = ba.abstract_state(np.array([0.3]), concrete_index=0)
        s2 = ba.abstract_state(np.array([0.8]), concrete_index=1)
        assert s1.state_id != s2.state_id

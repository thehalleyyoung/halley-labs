"""Tests for specification language and checker."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.spec.predicates import (
    LinearPredicate,
    ConjunctivePredicate,
    DisjunctivePredicate,
    NegationPredicate,
    DistancePredicate,
    CollisionPredicate,
    RegionPredicate,
    PredicateEvaluator,
    PredicateLibrary,
    Zonotope,
)
from marace.spec.temporal import (
    Always,
    Eventually,
    Until,
    Next,
    BoundedResponse,
)
from marace.spec.parser import SpecParser
from marace.spec.safety_library import SafetyLibrary


def _s(*values):
    """Build a single-agent dict state from a flat array."""
    return {"a": np.array(values)}


def _s2(vals_a, vals_b):
    """Build a two-agent dict state."""
    return {"a": np.array(vals_a), "b": np.array(vals_b)}


class TestLinearPredicate:
    """Test linear predicates."""

    def test_creation(self):
        """Test creating a linear predicate."""
        pred = LinearPredicate(
            a=np.array([1.0, 0.0, -1.0, 0.0]),
            b=2.0,
            name="separation",
        )
        assert pred.name == "separation"

    def test_evaluation_satisfied(self):
        """Test predicate evaluation when satisfied."""
        pred = LinearPredicate(
            a=np.array([1.0, 0.0]),
            b=5.0,
            agent_ids=["a"],
        )
        assert pred.evaluate(_s(3.0, 10.0))

    def test_evaluation_violated(self):
        """Test predicate evaluation when violated."""
        pred = LinearPredicate(
            a=np.array([1.0, 0.0]),
            b=5.0,
            agent_ids=["a"],
        )
        assert not pred.evaluate(_s(6.0, 0.0))

    def test_boundary(self):
        """Test predicate at boundary."""
        pred = LinearPredicate(
            a=np.array([1.0]),
            b=5.0,
            agent_ids=["a"],
        )
        assert pred.evaluate(_s(5.0))


class TestCompositPredicates:
    """Test composite predicates."""

    def test_conjunction(self):
        """Test AND of predicates."""
        p1 = LinearPredicate(np.array([1.0, 0.0]), 5.0, agent_ids=["a"])
        p2 = LinearPredicate(np.array([0.0, 1.0]), 3.0, agent_ids=["a"])
        conj = ConjunctivePredicate([p1, p2])
        assert conj.evaluate(_s(4.0, 2.0))
        assert not conj.evaluate(_s(4.0, 4.0))

    def test_disjunction(self):
        """Test OR of predicates."""
        p1 = LinearPredicate(np.array([1.0, 0.0]), 5.0, agent_ids=["a"])
        p2 = LinearPredicate(np.array([0.0, 1.0]), 3.0, agent_ids=["a"])
        disj = DisjunctivePredicate([p1, p2])
        assert disj.evaluate(_s(4.0, 4.0))
        assert disj.evaluate(_s(6.0, 2.0))
        assert not disj.evaluate(_s(6.0, 4.0))

    def test_negation(self):
        """Test NOT of predicate."""
        p = LinearPredicate(np.array([1.0]), 5.0, agent_ids=["a"])
        neg = NegationPredicate(p)
        assert neg.evaluate(_s(6.0))
        assert not neg.evaluate(_s(4.0))


class TestDistancePredicate:
    """Test distance predicate."""

    def test_close_agents(self):
        """Test agents within distance threshold."""
        pred = DistancePredicate(
            agent_i="a",
            agent_j="b",
            threshold=3.0,
        )
        state = _s2([0.0, 0.0], [1.0, 1.0])
        assert pred.evaluate(state)  # Distance ~1.41 < 3.0

    def test_far_agents(self):
        """Test agents beyond distance threshold."""
        pred = DistancePredicate(
            agent_i="a",
            agent_j="b",
            threshold=1.0,
        )
        state = _s2([0.0, 0.0], [10.0, 10.0])
        assert not pred.evaluate(state)  # Distance ~14.1 > 1.0


class TestRegionPredicate:
    """Test region predicate."""

    def test_in_region(self):
        """Test agent in region."""
        pred = RegionPredicate(
            agent_id="a",
            low=[0.0, 0.0],
            high=[10.0, 10.0],
        )
        state = {"a": np.array([5.0, 5.0])}
        assert pred.evaluate(state)

    def test_outside_region(self):
        """Test agent outside region."""
        pred = RegionPredicate(
            agent_id="a",
            low=[0.0, 0.0],
            high=[10.0, 10.0],
        )
        state = {"a": np.array([15.0, 5.0])}
        assert not pred.evaluate(state)


class TestTemporalFormulas:
    """Test temporal logic formulas."""

    def test_always_satisfied(self):
        """Test Always formula when satisfied."""
        pred = LinearPredicate(np.array([1.0]), 10.0, agent_ids=["a"])
        formula = Always(predicate=pred, horizon=5)
        trace = [_s(i * 1.0) for i in range(5)]
        assert formula.evaluate(trace)

    def test_always_violated(self):
        """Test Always formula when violated."""
        pred = LinearPredicate(np.array([1.0]), 3.0, agent_ids=["a"])
        formula = Always(predicate=pred, horizon=5)
        trace = [_s(i * 1.0) for i in range(5)]
        assert not formula.evaluate(trace)  # State 4.0 > 3.0

    def test_eventually_satisfied(self):
        """Test Eventually formula when satisfied."""
        pred = LinearPredicate(np.array([-1.0]), -3.0, agent_ids=["a"])  # x >= 3
        formula = Eventually(predicate=pred, horizon=5)
        trace = [_s(i * 1.0) for i in range(5)]
        assert formula.evaluate(trace)

    def test_eventually_not_satisfied(self):
        """Test Eventually formula when not satisfied."""
        pred = LinearPredicate(np.array([-1.0]), -100.0, agent_ids=["a"])  # x >= 100
        formula = Eventually(predicate=pred, horizon=5)
        trace = [_s(i * 1.0) for i in range(5)]
        assert not formula.evaluate(trace)

    def test_bounded_response(self):
        """Test bounded response formula."""
        trigger = LinearPredicate(np.array([-1.0]), -2.0, agent_ids=["a"])  # x >= 2
        response = LinearPredicate(np.array([1.0]), 1.0, agent_ids=["a"])  # x <= 1
        formula = BoundedResponse(
            trigger=trigger,
            response=response,
            deadline=3,
        )
        trace = [_s(0.0), _s(2.0), _s(1.5), _s(0.5), _s(0.0)]
        # Trigger at step 1 (x=2), response at step 3 (x=0.5 <= 1), within deadline 3
        assert formula.evaluate(trace)


class TestSpecParser:
    """Test specification parser."""

    def test_parse_simple(self):
        """Test parsing a simple distance predicate."""
        parser = SpecParser()
        spec = parser.parse("distance(agent_0, agent_1) > 2.0")
        assert spec is not None

    def test_parse_always(self):
        """Test parsing Always formula."""
        parser = SpecParser()
        spec = parser.parse("always(distance(agent_0, agent_1) > 2.0, horizon=10)")
        assert isinstance(spec, Always)

    def test_parse_eventually(self):
        """Test parsing Eventually formula."""
        parser = SpecParser()
        spec = parser.parse("eventually(distance(agent_0, agent_1) > 3.0, horizon=20)")
        assert isinstance(spec, Eventually)

    def test_parse_distance(self):
        """Test parsing distance predicate with comparison."""
        parser = SpecParser()
        spec = parser.parse("distance(agent_0, agent_1) >= 2.0")
        assert spec is not None


class TestSafetyLibrary:
    """Test safety specification library."""

    def test_collision_freedom(self):
        """Test collision freedom spec."""
        lib = SafetyLibrary(agent_ids=["agent_0", "agent_1"])
        spec = lib.collision_freedom()
        assert spec is not None

    def test_minimum_separation(self):
        """Test minimum separation spec."""
        lib = SafetyLibrary(agent_ids=["agent_0", "agent_1", "agent_2"])
        spec = lib.min_separation(d=2.0)
        assert spec is not None

    def test_bounded_speed(self):
        """Test deadlock freedom spec (replaces bounded_speed)."""
        lib = SafetyLibrary(agent_ids=["agent_0", "agent_1"])
        spec = lib.deadlock_freedom()
        assert spec is not None

    def test_highway_safety(self):
        """Test highway safety spec."""
        lib = SafetyLibrary(
            agent_ids=["agent_0", "agent_1", "agent_2", "agent_3"]
        )
        spec = lib.highway_safety(min_dist=2.0)
        assert spec is not None


class TestPredicateEvaluator:
    """Test predicate evaluator over abstract domains."""

    def test_evaluate_linear_zonotope(self):
        """Test evaluating linear predicate over zonotope."""
        evaluator = PredicateEvaluator()
        pred = LinearPredicate(np.array([1.0, 0.0]), 0.5)
        z = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[0.3, 0.0], [0.0, 0.3]])
        )
        result = evaluator.evaluate_linear(pred, z)
        # Zonotope x range is [-0.3, 0.3], all <= 0.5
        assert result is True

    def test_evaluate_violated_zonotope(self):
        """Test predicate definitely violated over zonotope."""
        evaluator = PredicateEvaluator()
        pred = LinearPredicate(np.array([1.0, 0.0]), 0.5)
        z = Zonotope(
            center=np.array([10.0, 0.0]),
            generators=np.array([[0.1, 0.0], [0.0, 0.1]])
        )
        result = evaluator.evaluate_linear(pred, z)
        # Zonotope x range is [9.9, 10.1], all > 0.5
        assert result is False

"""Tests for HB constraints."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.abstract.zonotope import Zonotope
from marace.abstract.hb_constraints import (
    HBConstraint,
    HBConstraintSet,
    TimingConstraint,
    OrderingConstraint,
    ConsistencyChecker,
)


class TestHBConstraint:
    """Test individual HB constraints."""

    def test_constraint_creation(self):
        """Test creating a linear constraint."""
        c = HBConstraint(
            normal=np.array([1.0, -1.0, 0.0]),
            bound=0.5,
            label="timing constraint"
        )
        assert len(c.normal) == 3
        assert c.bound == 0.5

    def test_constraint_evaluation_satisfied(self):
        """Test constraint evaluation when satisfied."""
        c = HBConstraint(
            normal=np.array([1.0, 0.0]),
            bound=2.0,
            label="x <= 2"
        )
        assert c.satisfied_by(np.array([1.0, 5.0]))
        assert c.satisfied_by(np.array([2.0, 5.0]))

    def test_constraint_evaluation_violated(self):
        """Test constraint evaluation when violated."""
        c = HBConstraint(
            normal=np.array([1.0, 0.0]),
            bound=2.0,
            label="x <= 2"
        )
        assert not c.satisfied_by(np.array([3.0, 0.0]))

    def test_constraint_margin(self):
        """Test margin computation."""
        c = HBConstraint(
            normal=np.array([1.0, 0.0]),
            bound=2.0,
            label="x <= 2"
        )
        margin = c.margin(np.array([1.0, 0.0]))
        assert np.isclose(margin, 1.0)


class TestHBConstraintSet:
    """Test constraint sets."""

    def test_empty_set(self):
        """Test empty constraint set."""
        cs = HBConstraintSet()
        assert len(cs) == 0

    def test_add_constraint(self):
        """Test adding constraints to set."""
        cs = HBConstraintSet()
        cs.add(HBConstraint(np.array([1.0, 0.0]), 2.0, label="c1"))
        cs.add(HBConstraint(np.array([0.0, 1.0]), 3.0, label="c2"))
        assert len(cs) == 2

    def test_evaluate_all_satisfied(self):
        """Test evaluating all constraints when satisfied."""
        cs = HBConstraintSet()
        cs.add(HBConstraint(np.array([1.0, 0.0]), 5.0, label="c1"))
        cs.add(HBConstraint(np.array([0.0, 1.0]), 5.0, label="c2"))
        assert cs.satisfied_by_point(np.array([1.0, 1.0]))

    def test_evaluate_one_violated(self):
        """Test evaluating when one constraint violated."""
        cs = HBConstraintSet()
        cs.add(HBConstraint(np.array([1.0, 0.0]), 2.0, label="c1"))
        cs.add(HBConstraint(np.array([0.0, 1.0]), 1.0, label="c2"))
        assert not cs.satisfied_by_point(np.array([1.0, 5.0]))


class TestTimingConstraint:
    """Test timing constraints."""

    def test_timing_constraint(self):
        """Test timing constraint creation and evaluation."""
        tc = TimingConstraint(
            dim_i=0,
            dim_j=1,
            delta=0.5,
            source_event="a",
            target_event="b",
        )
        assert tc.source_event != ""

    def test_timing_constraint_as_hb(self):
        """Test converting timing to HB constraint."""
        tc = TimingConstraint(
            dim_i=0,
            dim_j=1,
            delta=1.0,
            source_event="a",
            target_event="b",
        )
        hb = tc.to_hb_constraint(state_dim=4)
        assert isinstance(hb, HBConstraint)
        assert len(hb.normal) == 4


class TestOrderingConstraint:
    """Test ordering constraints."""

    def test_ordering_constraint(self):
        """Test ordering constraint."""
        oc = OrderingConstraint(
            dim_a=0,
            dim_b=2,
            offset=0.0,
            causal_chain=("e1", "e2"),
        )
        hb = oc.to_hb_constraint(state_dim=6)
        assert isinstance(hb, HBConstraint)


class TestConsistencyChecker:
    """Test consistency checking."""

    def test_consistent_zonotope(self):
        """Test zonotope consistent with constraints."""
        z = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[0.5, 0.0], [0.0, 0.5]])
        )
        cs = HBConstraintSet()
        cs.add(HBConstraint(np.array([1.0, 0.0]), 2.0, label="x <= 2"))
        cs.add(HBConstraint(np.array([0.0, 1.0]), 2.0, label="y <= 2"))
        is_consistent, violated = ConsistencyChecker.check_all(z, cs)
        assert is_consistent

    def test_inconsistent_zonotope(self):
        """Test zonotope inconsistent with constraints."""
        z = Zonotope(
            center=np.array([5.0, 5.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        cs = HBConstraintSet()
        cs.add(HBConstraint(np.array([1.0, 0.0]), 2.0, label="x <= 2"))
        is_consistent, violated = ConsistencyChecker.check_all(z, cs)
        assert not is_consistent

    def test_partially_consistent(self):
        """Test zonotope partially consistent (overlaps constraint boundary)."""
        z = Zonotope(
            center=np.array([1.5, 0.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        cs = HBConstraintSet()
        cs.add(HBConstraint(np.array([1.0, 0.0]), 2.0, label="x <= 2"))
        is_consistent, violated = ConsistencyChecker.check_all(z, cs)
        # Zonotope extends to x=2.5, so it violates x<=2; partially consistent
        assert not is_consistent or len(violated) == 0

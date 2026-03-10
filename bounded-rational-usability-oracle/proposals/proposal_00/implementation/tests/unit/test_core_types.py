"""
Unit tests for usability_oracle.core.types.

Covers: Point2D, BoundingBox, Interval, CostTuple, TrajectoryStep,
Trajectory, and PolicyDistribution — testing geometric helpers,
interval arithmetic, cost-algebra composition, trajectory aggregation,
and policy distribution operations.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from usability_oracle.core.types import (
    ActionId,
    BoundingBox,
    CostTuple,
    Interval,
    Point2D,
    PolicyDistribution,
    StateId,
    Trajectory,
    TrajectoryStep,
)


# ═══════════════════════════════════════════════════════════════════════════
# Point2D
# ═══════════════════════════════════════════════════════════════════════════


class TestPoint2D:
    """Tests for Point2D geometric helpers, operators, and classmethods."""

    def test_distance(self) -> None:
        """Euclidean distance for a 3-4-5 triangle is 5 and is symmetric."""
        a, b = Point2D(0, 0), Point2D(3, 4)
        assert math.isclose(a.distance(b), 5.0)
        assert math.isclose(a.distance(b), b.distance(a))

    def test_midpoint(self) -> None:
        """Midpoint of (2,4) and (6,8) should be (4,6)."""
        m = Point2D(2, 4).midpoint(Point2D(6, 8))
        assert math.isclose(m.x, 4.0) and math.isclose(m.y, 6.0)

    def test_translate(self) -> None:
        """Translating (1,2) by (3,-1) yields (4,1)."""
        p = Point2D(1, 2).translate(3, -1)
        assert math.isclose(p.x, 4.0) and math.isclose(p.y, 1.0)

    def test_scale(self) -> None:
        """Scaling (3,4) by 2 yields (6,8)."""
        p = Point2D(3, 4).scale(2)
        assert math.isclose(p.x, 6.0) and math.isclose(p.y, 8.0)

    def test_manhattan_distance(self) -> None:
        """L1 distance between (0,0) and (3,4) must be 7."""
        assert math.isclose(Point2D(0, 0).manhattan_distance(Point2D(3, 4)), 7.0)

    def test_angle_to(self) -> None:
        """Angle from origin to (1,0) is 0; to (0,1) is pi/2."""
        o = Point2D(0, 0)
        assert math.isclose(o.angle_to(Point2D(1, 0)), 0.0)
        assert math.isclose(o.angle_to(Point2D(0, 1)), math.pi / 2)

    def test_add_sub_operators(self) -> None:
        """Add sums coordinates; sub differences them."""
        assert math.isclose((Point2D(1, 2) + Point2D(3, 4)).x, 4.0)
        assert math.isclose((Point2D(5, 7) - Point2D(2, 3)).y, 4.0)

    def test_origin(self) -> None:
        """Point2D.origin() should return (0, 0)."""
        o = Point2D.origin()
        assert o.x == 0.0 and o.y == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# BoundingBox
# ═══════════════════════════════════════════════════════════════════════════


class TestBoundingBox:
    """Tests for BoundingBox properties, spatial predicates, pad, and scale."""

    def test_center_and_area(self) -> None:
        """Centre and area for a known bounding box."""
        bb = BoundingBox(10, 20, 100, 50)
        assert math.isclose(bb.center.x, 60.0)
        assert bb.area == 5000.0

    def test_negative_dimensions_raise(self) -> None:
        """Negative width or height must raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            BoundingBox(0, 0, -1, 10)

    def test_contains(self) -> None:
        """Contains is inclusive on edges; outside returns False."""
        bb = BoundingBox(0, 0, 10, 10)
        assert bb.contains(Point2D(5, 5))
        assert bb.contains(Point2D(10, 10))
        assert not bb.contains(Point2D(11, 5))

    def test_overlaps(self) -> None:
        """Overlapping boxes report True; disjoint report False."""
        a = BoundingBox(0, 0, 10, 10)
        assert a.overlaps(BoundingBox(5, 5, 10, 10))
        assert not a.overlaps(BoundingBox(20, 20, 5, 5))

    def test_intersection(self) -> None:
        """Intersection returns correct sub-box or None for disjoint."""
        inter = BoundingBox(0, 0, 10, 10).intersection(BoundingBox(5, 5, 10, 10))
        assert inter is not None and math.isclose(inter.width, 5)
        assert BoundingBox(0, 0, 5, 5).intersection(BoundingBox(10, 10, 5, 5)) is None

    def test_union(self) -> None:
        """Union of two boxes encloses both."""
        u = BoundingBox(0, 0, 5, 5).union(BoundingBox(3, 3, 10, 10))
        assert u.x == 0 and math.isclose(u.width, 13)

    def test_distance_to(self) -> None:
        """Distance between separated boxes matches expected gap."""
        assert math.isclose(
            BoundingBox(0, 0, 3, 3).distance_to(BoundingBox(6, 0, 3, 3)), 3.0
        )

    def test_iou(self) -> None:
        """IoU of a box with itself is 1.0; disjoint is 0.0."""
        bb = BoundingBox(0, 0, 10, 10)
        assert math.isclose(bb.iou(bb), 1.0)
        assert bb.iou(BoundingBox(100, 100, 5, 5)) == 0.0

    def test_pad(self) -> None:
        """Padding a box expands it on all sides."""
        padded = BoundingBox(10, 10, 20, 20).pad(5)
        assert padded.x == 5 and padded.width == 30

    def test_scale(self) -> None:
        """Scaling by 2 doubles dimensions and preserves the centre."""
        bb = BoundingBox(0, 0, 10, 10)
        scaled = bb.scale(2.0)
        assert math.isclose(scaled.width, 20.0)
        assert math.isclose(scaled.center.x, bb.center.x)


# ═══════════════════════════════════════════════════════════════════════════
# Interval
# ═══════════════════════════════════════════════════════════════════════════


class TestInterval:
    """Tests for Interval basics, arithmetic, set operations, and point."""

    def test_width_and_midpoint(self) -> None:
        """Width of [2,5] is 3 and midpoint of [2,8] is 5."""
        assert Interval(2, 5).width == 3.0
        assert Interval(2, 8).midpoint == 5.0

    def test_contains_and_overlaps(self) -> None:
        """contains and overlaps behave correctly on known inputs."""
        i = Interval(0, 10)
        assert i.contains(5.0) and not i.contains(11.0)
        assert Interval(0, 5).overlaps(Interval(3, 8))
        assert not Interval(0, 2).overlaps(Interval(3, 5))

    def test_invalid_raises(self) -> None:
        """Interval with low > high must raise ValueError."""
        with pytest.raises(ValueError, match="low <= high"):
            Interval(5, 2)

    def test_add_and_sub(self) -> None:
        """Interval addition and subtraction follow standard rules."""
        assert math.isclose((Interval(1, 2) + Interval(3, 4)).low, 4)
        assert math.isclose((Interval(1, 2) + 10).low, 11)
        r = Interval(5, 7) - Interval(1, 2)
        assert math.isclose(r.low, 3) and math.isclose(r.high, 6)

    def test_mul(self) -> None:
        """Interval multiplication including negative scalar flipping."""
        r = Interval(2, 3) * Interval(4, 5)
        assert math.isclose(r.low, 8) and math.isclose(r.high, 15)
        assert math.isclose((Interval(2, 5) * (-1)).low, -5)

    def test_div(self) -> None:
        """Division by scalar works; zero or spanning-zero raises."""
        assert math.isclose((Interval(4, 8) / 2).low, 2)
        with pytest.raises(ZeroDivisionError):
            Interval(1, 2) / 0
        with pytest.raises(ZeroDivisionError):
            Interval(1, 2) / Interval(-1, 1)

    def test_pow(self) -> None:
        """Even power on interval spanning zero clamps low to 0."""
        r = Interval(-2, 3) ** 2
        assert math.isclose(r.low, 0) and math.isclose(r.high, 9)

    def test_neg_and_abs(self) -> None:
        """-[2,5] = [-5,-2]; abs([-3,5]) = [0,5]."""
        assert (-Interval(2, 5)).low == -5
        assert abs(Interval(-3, 5)).low == 0.0

    def test_union_and_intersection(self) -> None:
        """Union and intersection of intervals."""
        assert Interval(1, 3).union(Interval(5, 7)).high == 7
        inter = Interval(1, 5).intersection(Interval(3, 7))
        assert inter is not None and inter.low == 3
        assert Interval(1, 2).intersection(Interval(5, 6)) is None

    def test_point(self) -> None:
        """Interval.point(3) creates degenerate interval [3,3]."""
        i = Interval.point(3.0)
        assert i.low == 3.0 and i.width == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# CostTuple
# ═══════════════════════════════════════════════════════════════════════════


class TestCostTuple:
    """Tests for CostTuple algebra, context application, dominance, operators."""

    def test_compose_sequential_and_parallel(self) -> None:
        """Sequential adds mu/sigma_sq, maxes lambda_; parallel maxes mu, sums lambda_."""
        a = CostTuple(mu=1.0, sigma_sq=0.1, kappa=0.2, lambda_=0.5)
        b = CostTuple(mu=2.0, sigma_sq=0.3, kappa=0.1, lambda_=0.8)
        seq = a.compose_sequential(b)
        assert math.isclose(seq.mu, 3.0) and math.isclose(seq.lambda_, 0.8)
        par = a.compose_parallel(b)
        assert math.isclose(par.mu, 2.0) and math.isclose(par.lambda_, 1.3)

    def test_apply_context(self) -> None:
        """Beta=2 halves mu, quarters sigma_sq; beta<=0 raises."""
        ct = CostTuple(mu=4.0, sigma_sq=4.0, kappa=2.0, lambda_=1.0)
        s = ct.apply_context(beta=2.0)
        assert math.isclose(s.mu, 2.0) and math.isclose(s.sigma_sq, 1.0)
        with pytest.raises(ValueError):
            CostTuple(mu=1.0).apply_context(0.0)

    def test_zero(self) -> None:
        """CostTuple.zero() is the additive identity."""
        z = CostTuple.zero()
        assert z.mu == 0.0 and z.sigma_sq == 0.0

    def test_dominates(self) -> None:
        """Pareto dominance: lower dominates higher; equal does not."""
        low = CostTuple(mu=1.0, sigma_sq=0.1, kappa=0.1, lambda_=0.1)
        high = CostTuple(mu=2.0, sigma_sq=0.2, kappa=0.2, lambda_=0.2)
        assert low.dominates(high) and not high.dominates(low)
        assert not low.dominates(low)

    def test_add_and_mul_operators(self) -> None:
        """__add__ delegates to sequential; __mul__ scales correctly."""
        assert math.isclose((CostTuple(mu=1) + CostTuple(mu=2)).mu, 3.0)
        r = CostTuple(mu=2.0, sigma_sq=1.0) * 3
        assert math.isclose(r.mu, 6.0) and math.isclose(r.sigma_sq, 9.0)
        assert math.isclose((3 * CostTuple(mu=2.0)).mu, 6.0)

    def test_negative_sigma_sq_raises(self) -> None:
        """Negative sigma_sq in constructor must raise ValueError."""
        with pytest.raises(ValueError):
            CostTuple(mu=1.0, sigma_sq=-0.1)


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory
# ═══════════════════════════════════════════════════════════════════════════


def _step(state: str, action: str, mu: float, ts: float = 0.0) -> TrajectoryStep:
    """Helper to create a TrajectoryStep."""
    return TrajectoryStep(StateId(state), ActionId(action), CostTuple(mu=mu), ts)


class TestTrajectory:
    """Tests for Trajectory.from_steps, length, duration, slice, append."""

    def test_from_steps_length_duration(self) -> None:
        """from_steps aggregates total_cost; length and duration are correct."""
        traj = Trajectory.from_steps(
            [_step("s0", "a0", 1, 0), _step("s1", "a1", 2, 3.5)]
        )
        assert math.isclose(traj.total_cost.mu, 3.0)
        assert traj.length == 2 and math.isclose(traj.duration, 3.5)

    def test_slice(self) -> None:
        """Slicing a trajectory returns the correct sub-trajectory."""
        traj = Trajectory.from_steps(
            [_step(f"s{i}", f"a{i}", float(i)) for i in range(5)]
        )
        sub = traj.slice(1, 3)
        assert sub.length == 2 and sub.steps[0].state_id == StateId("s1")

    def test_append(self) -> None:
        """Appending a step increases length and updates total_cost."""
        traj = Trajectory.from_steps([_step("s0", "a0", 1)])
        extended = traj.append(_step("s1", "a1", 2))
        assert extended.length == 2 and math.isclose(extended.total_cost.mu, 3.0)


# ═══════════════════════════════════════════════════════════════════════════
# PolicyDistribution
# ═══════════════════════════════════════════════════════════════════════════


class TestPolicyDistribution:
    """Tests for PolicyDistribution action_probs, greedy, entropy, sample."""

    def _pol(self) -> PolicyDistribution:
        """Build a small policy for testing."""
        return PolicyDistribution(mapping={
            StateId("s0"): {ActionId("a0"): 0.9, ActionId("a1"): 0.1},
            StateId("s1"): {ActionId("a2"): 1.0},
        })

    def test_action_probs_and_greedy(self) -> None:
        """action_probs returns stored distribution; greedy picks max; unknown raises."""
        pol = self._pol()
        assert math.isclose(pol.action_probs(StateId("s0"))[ActionId("a0")], 0.9)
        assert pol.action_probs(StateId("unknown")) == {}
        assert pol.greedy_action(StateId("s0")) == ActionId("a0")
        with pytest.raises(KeyError):
            pol.greedy_action(StateId("missing"))

    def test_entropy(self) -> None:
        """Deterministic state has 0 entropy; uniform over 2 is ln(2)."""
        assert math.isclose(self._pol().entropy(StateId("s1")), 0.0)
        unif = PolicyDistribution(mapping={
            StateId("s"): {ActionId("a0"): 0.5, ActionId("a1"): 0.5}
        })
        assert math.isclose(unif.entropy(StateId("s")), math.log(2), rel_tol=1e-6)

    def test_sample_and_counts(self) -> None:
        """Sampling is reproducible with fixed seed; counts are correct."""
        pol = self._pol()
        a1 = pol.sample_action(StateId("s0"), rng=np.random.default_rng(42))
        a2 = pol.sample_action(StateId("s0"), rng=np.random.default_rng(42))
        assert a1 == a2
        assert pol.num_states == 2 and pol.num_actions == 3

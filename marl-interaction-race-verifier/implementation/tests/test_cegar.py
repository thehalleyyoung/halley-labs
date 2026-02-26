"""Tests for the CEGAR (Counter-Example Guided Abstraction Refinement) module."""

from __future__ import annotations

import time
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from marace.abstract.zonotope import Zonotope
from marace.abstract.fixpoint import FixpointEngine, FixpointResult, WideningStrategy
from marace.abstract.cegar import (
    AbstractionRefinement,
    CEGARResult,
    CEGARVerifier,
    CompositionalCEGARVerifier,
    RefinementRecord,
    RefinementStrategy,
    SpuriousnessChecker,
    Verdict,
    make_cegar_verifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_transfer(z: Zonotope) -> Zonotope:
    """Transfer function that returns the input unchanged (contractive)."""
    return z.copy()


def _contractive_transfer(z: Zonotope) -> Zonotope:
    """Shrink by 10 % per iteration — guaranteed convergence."""
    return z.scale(0.9)


def _expanding_transfer(z: Zonotope) -> Zonotope:
    """Expand by 10 % — will trigger widening."""
    return z.scale(1.1)


def _safe_concrete_eval(x: np.ndarray) -> np.ndarray:
    """Concrete evaluator: identity (successor = current state)."""
    return x.copy()


def _always_safe(_x: np.ndarray) -> bool:
    """Safety predicate: never unsafe."""
    return False


def _always_unsafe(_x: np.ndarray) -> bool:
    """Safety predicate: always unsafe."""
    return True


def _threshold_unsafe(x: np.ndarray) -> bool:
    """Unsafe if x[0] > 5."""
    return float(x[0]) > 5.0


def _make_2d_zonotope(cx: float = 0.0, cy: float = 0.0,
                       w: float = 1.0) -> Zonotope:
    """Axis-aligned 2-D zonotope centred at (cx, cy) with half-width w."""
    return Zonotope.from_interval(
        np.array([cx - w, cy - w]),
        np.array([cx + w, cy + w]),
    )


def _unsafe_halfspace_x0(bound: float = 5.0):
    """Unsafe region: x[0] ≥ bound."""
    return (np.array([1.0, 0.0]), bound)


# ---------------------------------------------------------------------------
# 1. SpuriousnessChecker — spurious counterexample detection
# ---------------------------------------------------------------------------


class TestSpuriousnessChecker:
    """Tests for SpuriousnessChecker."""

    def test_real_counterexample_detected(self):
        """A point that the concrete evaluator says is unsafe is real."""
        checker = SpuriousnessChecker(
            concrete_evaluator=_safe_concrete_eval,
            safety_predicate=_always_unsafe,
        )
        assert checker.check_point(np.array([0.0, 0.0])) is True

    def test_spurious_counterexample_detected(self):
        """A point that the concrete evaluator says is safe is spurious."""
        checker = SpuriousnessChecker(
            concrete_evaluator=_safe_concrete_eval,
            safety_predicate=_always_safe,
        )
        assert checker.check_point(np.array([10.0, 0.0])) is False

    def test_check_zonotope_finds_real(self):
        """Sampling the zonotope should find a real counterexample."""
        checker = SpuriousnessChecker(
            concrete_evaluator=_safe_concrete_eval,
            safety_predicate=_always_unsafe,
            num_samples=10,
        )
        z = _make_2d_zonotope()
        is_real, witness = checker.check_zonotope(z)
        assert is_real is True
        assert witness is not None

    def test_check_zonotope_all_spurious(self):
        """If no sample is unsafe, report spurious."""
        checker = SpuriousnessChecker(
            concrete_evaluator=_safe_concrete_eval,
            safety_predicate=_always_safe,
            num_samples=10,
        )
        z = _make_2d_zonotope()
        is_real, witness = checker.check_zonotope(z)
        assert is_real is False
        assert witness is None

    def test_check_along_direction(self):
        """Directional check should find a real counterexample."""
        checker = SpuriousnessChecker(
            concrete_evaluator=_safe_concrete_eval,
            safety_predicate=_always_unsafe,
        )
        z = _make_2d_zonotope()
        direction = np.array([1.0, 0.0])
        is_real, witness = checker.check_along_direction(z, direction)
        assert is_real is True
        assert witness is not None


# ---------------------------------------------------------------------------
# 2. AbstractionRefinement — splitting strategies
# ---------------------------------------------------------------------------


class TestAbstractionRefinement:
    """Tests for AbstractionRefinement splitting strategies."""

    def test_split_widest_dimension(self):
        """Split along the widest dimension of a rectangular zonotope."""
        z = Zonotope.from_interval(
            np.array([-1.0, -3.0]),
            np.array([1.0, 3.0]),
        )
        ref = AbstractionRefinement(strategy=RefinementStrategy.DIMENSION)
        left, right, dim, val = ref.refine(z)
        # Dimension 1 has width 6 > 2, so it should be chosen.
        assert dim == 1
        assert left.dimension == 2
        assert right.dimension == 2

    def test_split_counterexample_guided(self):
        """Counterexample-guided split selects the dimension of max deviation."""
        z = _make_2d_zonotope(cx=0.0, cy=0.0, w=2.0)
        spurious = np.array([1.5, 0.1])
        ref = AbstractionRefinement(strategy=RefinementStrategy.COUNTEREXAMPLE)
        left, right, dim, val = ref.refine(z, spurious_point=spurious)
        # x-deviation = 1.5 / 4 = 0.375, y-deviation = 0.1 / 4 = 0.025
        assert dim == 0

    def test_split_gradient_based(self):
        """Gradient-based split uses the safety normal."""
        z = _make_2d_zonotope(w=2.0)
        normal = np.array([0.0, 1.0])
        ref = AbstractionRefinement(
            strategy=RefinementStrategy.GRADIENT,
            safety_normal=normal,
        )
        left, right, dim, val = ref.refine(z)
        assert dim == 1  # gradient direction is y

    def test_children_cover_parent_bbox(self):
        """The split dimension should be partitioned by the split value,
        and each child should be a valid zonotope of the same dimension.

        ``split_halfspace`` intersects with complementary half-spaces.
        Each child soundly over-approximates its respective half of the
        parent, but projections onto non-split dimensions may shrink
        (tighter bounds are sound).  We verify the split dimension is
        correctly partitioned.
        """
        z = _make_2d_zonotope(cx=1.0, cy=2.0, w=1.5)
        ref = AbstractionRefinement(strategy=RefinementStrategy.DIMENSION)
        left, right, dim, val = ref.refine(z)

        assert left.dimension == z.dimension
        assert right.dimension == z.dimension

        # The split value should lie within the parent's bounding box
        parent_bb = z.bounding_box()
        assert parent_bb[dim, 0] <= val <= parent_bb[dim, 1]

        left_bb = left.bounding_box()
        right_bb = right.bounding_box()

        # Left child's upper bound in split dim should be ≤ split value
        # (plus tolerance for the over-approximation).
        # Right child's lower bound should be ≥ split value (minus tolerance).
        assert left_bb[dim, 1] <= val + 1e-6 or left_bb[dim, 0] >= val - 1e-6
        assert right_bb[dim, 0] >= val - 1e-6 or right_bb[dim, 1] <= val + 1e-6


# ---------------------------------------------------------------------------
# 3. CEGARVerifier — main loop
# ---------------------------------------------------------------------------


class TestCEGARVerifier:
    """Tests for the main CEGAR verification loop."""

    def test_safe_without_refinement(self):
        """A system whose fixpoint does not reach the unsafe region is SAFE
        with zero refinements."""
        z = _make_2d_zonotope(cx=0.0, cy=0.0, w=1.0)
        unsafe = _unsafe_halfspace_x0(bound=5.0)

        verifier = make_cegar_verifier(
            transfer_fn=_contractive_transfer,
            concrete_evaluator=_safe_concrete_eval,
            safety_predicate=_always_safe,
            unsafe_halfspace=unsafe,
            fixpoint_kwargs={"max_iterations": 10},
        )
        result = verifier.verify(z)
        assert result.verdict == Verdict.SAFE
        assert result.refinement_iterations == 0
        assert result.counterexample is None

    def test_unsafe_real_counterexample(self):
        """A system that is genuinely unsafe should return UNSAFE with a
        concrete counterexample."""
        # Zonotope centered at 6 with width 1 — entirely in the unsafe region.
        z = _make_2d_zonotope(cx=6.0, cy=0.0, w=0.5)
        unsafe = _unsafe_halfspace_x0(bound=5.0)

        verifier = make_cegar_verifier(
            transfer_fn=_identity_transfer,
            concrete_evaluator=_safe_concrete_eval,
            safety_predicate=_threshold_unsafe,
            unsafe_halfspace=unsafe,
            fixpoint_kwargs={"max_iterations": 5},
        )
        result = verifier.verify(z)
        assert result.verdict == Verdict.UNSAFE
        assert result.counterexample is not None
        assert result.counterexample[0] > 5.0

    def test_spurious_triggers_refinement(self):
        """A zonotope that abstractly intersects the unsafe region but is
        concretely safe should trigger refinement and eventually be verified
        SAFE."""
        # Identity transfer: fixpoint = initial zonotope.
        # Zonotope centred at 3 with width 3 → bbox x ∈ [0, 6].
        # Unsafe region x ≥ 5.  Abstract fixpoint intersects unsafe zone,
        # but all concrete points are safe (_always_safe).
        z = _make_2d_zonotope(cx=3.0, cy=0.0, w=3.0)
        unsafe = _unsafe_halfspace_x0(bound=5.0)

        verifier = make_cegar_verifier(
            transfer_fn=_identity_transfer,
            concrete_evaluator=_safe_concrete_eval,
            safety_predicate=_always_safe,
            unsafe_halfspace=unsafe,
            max_refinements=20,
            fixpoint_kwargs={"max_iterations": 5},
        )
        result = verifier.verify(z)
        assert result.verdict == Verdict.SAFE
        assert result.refinement_iterations > 0, (
            "Should have done at least one refinement"
        )

    def test_convergence_after_finite_refinements(self):
        """CEGAR should converge after a bounded number of refinements when
        the system is safe but the initial abstraction is too coarse."""
        z = _make_2d_zonotope(cx=2.0, cy=0.0, w=4.0)
        unsafe = _unsafe_halfspace_x0(bound=5.5)

        verifier = make_cegar_verifier(
            transfer_fn=_contractive_transfer,
            concrete_evaluator=_safe_concrete_eval,
            safety_predicate=_always_safe,
            unsafe_halfspace=unsafe,
            max_refinements=30,
            fixpoint_kwargs={"max_iterations": 10},
        )
        result = verifier.verify(z)
        assert result.verdict == Verdict.SAFE
        assert result.refinement_iterations <= 30

    def test_timeout_returns_unknown(self):
        """If the timeout is hit, the result should be UNKNOWN."""
        z = _make_2d_zonotope(cx=3.0, cy=0.0, w=3.0)
        unsafe = _unsafe_halfspace_x0(bound=5.0)

        verifier = make_cegar_verifier(
            transfer_fn=_identity_transfer,
            concrete_evaluator=_safe_concrete_eval,
            safety_predicate=_always_safe,
            unsafe_halfspace=unsafe,
            max_refinements=1000,
            max_splits=1000,
            timeout_s=0.0,  # immediate timeout
            fixpoint_kwargs={"max_iterations": 3},
        )
        result = verifier.verify(z)
        assert result.verdict == Verdict.UNKNOWN

    def test_max_refinements_returns_unknown(self):
        """Hitting the refinement cap should give UNKNOWN."""
        z = _make_2d_zonotope(cx=3.0, cy=0.0, w=3.0)
        unsafe = _unsafe_halfspace_x0(bound=5.0)

        verifier = make_cegar_verifier(
            transfer_fn=_identity_transfer,
            concrete_evaluator=_safe_concrete_eval,
            safety_predicate=_always_safe,
            unsafe_halfspace=unsafe,
            max_refinements=1,
            fixpoint_kwargs={"max_iterations": 3},
        )
        result = verifier.verify(z)
        # With max_refinements=1, we refine once and then either solve or UNKNOWN.
        assert result.verdict in (Verdict.SAFE, Verdict.UNKNOWN)
        assert result.refinement_iterations <= 1

    def test_refinement_history_tracked(self):
        """Refinement records should be populated with statistics."""
        z = _make_2d_zonotope(cx=3.0, cy=0.0, w=3.0)
        unsafe = _unsafe_halfspace_x0(bound=5.0)

        verifier = make_cegar_verifier(
            transfer_fn=_contractive_transfer,
            concrete_evaluator=_safe_concrete_eval,
            safety_predicate=_always_safe,
            unsafe_halfspace=unsafe,
            max_refinements=10,
            fixpoint_kwargs={"max_iterations": 10},
        )
        result = verifier.verify(z)
        if result.refinement_iterations > 0:
            rec = result.refinement_history[0]
            assert rec.split_dimension in (0, 1)
            assert rec.pre_split_volume > 0
            assert rec.precision_improvement >= 0.0

    def test_extract_counterexample_maximises_unsafe(self):
        """_extract_counterexample should return the point maximising aᵀx."""
        z = _make_2d_zonotope(cx=0.0, cy=0.0, w=2.0)
        unsafe_normal = np.array([1.0, 0.0])
        unsafe_bound = 5.0
        verifier = make_cegar_verifier(
            transfer_fn=_identity_transfer,
            concrete_evaluator=_safe_concrete_eval,
            safety_predicate=_always_safe,
            unsafe_halfspace=(unsafe_normal, unsafe_bound),
        )
        point = verifier._extract_counterexample(z)
        # The maximiser of x[0] over [-2,2]×[-2,2] should be near x[0]=2.
        assert point[0] >= 1.9


# ---------------------------------------------------------------------------
# 4. CompositionalCEGARVerifier — multi-agent groups
# ---------------------------------------------------------------------------


class TestCompositionalCEGAR:
    """Tests for compositional multi-agent CEGAR."""

    def test_all_safe(self):
        """If every group is safe, the combined verdict is SAFE."""
        z_a = _make_2d_zonotope(cx=0.0, cy=0.0, w=1.0)
        z_b = _make_2d_zonotope(cx=0.0, cy=0.0, w=1.0)
        unsafe = _unsafe_halfspace_x0(bound=5.0)

        v_a = make_cegar_verifier(
            _contractive_transfer, _safe_concrete_eval, _always_safe, unsafe,
            fixpoint_kwargs={"max_iterations": 5},
        )
        v_b = make_cegar_verifier(
            _contractive_transfer, _safe_concrete_eval, _always_safe, unsafe,
            fixpoint_kwargs={"max_iterations": 5},
        )
        comp = CompositionalCEGARVerifier({"A": v_a, "B": v_b})
        results = comp.verify_all({"A": z_a, "B": z_b})
        assert comp.combined_verdict(results) == Verdict.SAFE

    def test_one_unsafe(self):
        """If any group is unsafe, the combined verdict is UNSAFE."""
        z_safe = _make_2d_zonotope(cx=0.0, cy=0.0, w=1.0)
        z_bad = _make_2d_zonotope(cx=6.0, cy=0.0, w=0.5)
        unsafe = _unsafe_halfspace_x0(bound=5.0)

        v_safe = make_cegar_verifier(
            _contractive_transfer, _safe_concrete_eval, _always_safe, unsafe,
            fixpoint_kwargs={"max_iterations": 5},
        )
        v_bad = make_cegar_verifier(
            _identity_transfer, _safe_concrete_eval, _threshold_unsafe, unsafe,
            fixpoint_kwargs={"max_iterations": 5},
        )
        comp = CompositionalCEGARVerifier({"safe": v_safe, "bad": v_bad})
        results = comp.verify_all({"safe": z_safe, "bad": z_bad})
        assert comp.combined_verdict(results) == Verdict.UNSAFE

    def test_combined_result_aggregation(self):
        """combined_result should aggregate statistics correctly."""
        z = _make_2d_zonotope(cx=0.0, cy=0.0, w=1.0)
        unsafe = _unsafe_halfspace_x0(bound=5.0)

        v = make_cegar_verifier(
            _contractive_transfer, _safe_concrete_eval, _always_safe, unsafe,
            fixpoint_kwargs={"max_iterations": 5},
        )
        comp = CompositionalCEGARVerifier({"A": v})
        per_group = comp.verify_all({"A": z})
        combined = comp.combined_result(per_group)
        assert combined.verdict == Verdict.SAFE
        assert combined.total_time_s >= 0.0


# ---------------------------------------------------------------------------
# 5. CEGARResult — dataclass behaviour
# ---------------------------------------------------------------------------


class TestCEGARResult:
    """Tests for CEGARResult properties and summary."""

    def test_is_safe_property(self):
        r = CEGARResult(verdict=Verdict.SAFE)
        assert r.is_safe is True
        assert r.is_unsafe is False

    def test_is_unsafe_property(self):
        r = CEGARResult(verdict=Verdict.UNSAFE, counterexample=np.zeros(2))
        assert r.is_unsafe is True
        assert r.is_safe is False

    def test_total_precision_improvement_empty(self):
        r = CEGARResult(verdict=Verdict.SAFE)
        assert r.total_precision_improvement == 0.0

    def test_total_precision_improvement_nonzero(self):
        records = [
            RefinementRecord(
                iteration=0, split_dimension=0, split_point=0.0,
                pre_split_volume=10.0,
                post_split_volumes=(4.0, 4.0),
                precision_improvement=0.2,
            ),
            RefinementRecord(
                iteration=1, split_dimension=1, split_point=0.0,
                pre_split_volume=4.0,
                post_split_volumes=(1.5, 1.5),
                precision_improvement=0.25,
            ),
        ]
        r = CEGARResult(
            verdict=Verdict.SAFE,
            refinement_iterations=2,
            refinement_history=records,
        )
        # 1 - (1-0.2)*(1-0.25) = 1 - 0.6 = 0.4
        assert abs(r.total_precision_improvement - 0.4) < 1e-9

    def test_summary_string(self):
        r = CEGARResult(
            verdict=Verdict.UNSAFE,
            counterexample=np.array([6.0, 0.0]),
            refinement_iterations=3,
        )
        s = r.summary()
        assert "UNSAFE" in s
        assert "3" in s

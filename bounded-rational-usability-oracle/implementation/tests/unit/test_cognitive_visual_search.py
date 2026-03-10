"""Unit tests for usability_oracle.cognitive.visual_search.VisualSearchModel.

Tests cover serial, parallel, and guided search predictions, saliency
computation, effective set size, eccentricity cost, search time distribution,
fixation counts, eccentricity-aware search, and monotonicity properties.

References
----------
Treisman, A. & Gelade, G. (1980). A feature-integration theory of attention.
    *Cognitive Psychology*, 12(1), 97-136.
Wolfe, J. M. (2007). Guided Search 4.0. In *Integrated Models of Cognitive
    Systems* (pp. 99-119). Oxford University Press.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.cognitive.visual_search import VisualSearchModel
from usability_oracle.cognitive.models import BoundingBox, Point2D


# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #


class TestVisualSearchConstants:
    """Verify class-level empirical constants."""

    def test_efficient_slope(self) -> None:
        """EFFICIENT_SLOPE should be ~5 ms/item (Treisman & Gelade, 1980)."""
        assert VisualSearchModel.EFFICIENT_SLOPE == pytest.approx(0.005)

    def test_inefficient_slope(self) -> None:
        """INEFFICIENT_SLOPE should be ~25 ms/item."""
        assert VisualSearchModel.INEFFICIENT_SLOPE == pytest.approx(0.025)

    def test_base_rt(self) -> None:
        """BASE_RT should be 400 ms."""
        assert VisualSearchModel.BASE_RT == pytest.approx(0.400)


# ------------------------------------------------------------------ #
# Serial search
# ------------------------------------------------------------------ #


class TestPredictSerial:
    """Tests for VisualSearchModel.predict_serial()."""

    def test_target_present(self) -> None:
        """Serial search, target present: RT = BASE + slope * n / 2.

        n=20, slope=0.025 → 0.400 + 0.025 * 20 / 2 = 0.400 + 0.250 = 0.650.
        """
        expected = 0.400 + 0.025 * 20 / 2.0
        result = VisualSearchModel.predict_serial(20, target_present=True)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_target_absent(self) -> None:
        """Serial search, target absent: RT = BASE + slope * n.

        When absent, all items must be inspected.
        """
        expected = 0.400 + 0.025 * 20
        result = VisualSearchModel.predict_serial(20, target_present=False)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_absent_slower_than_present(self) -> None:
        """Target-absent search should take longer than target-present.

        In serial self-terminating search, absent trials require
        exhaustive inspection.
        """
        present = VisualSearchModel.predict_serial(30, target_present=True)
        absent = VisualSearchModel.predict_serial(30, target_present=False)
        assert absent > present

    def test_minimum_one_item(self) -> None:
        """n_items < 1 should be clamped to 1."""
        result = VisualSearchModel.predict_serial(0, target_present=True)
        expected = VisualSearchModel.BASE_RT + 0.025 * 1 / 2.0
        assert result == pytest.approx(expected, rel=1e-9)


# ------------------------------------------------------------------ #
# Parallel search
# ------------------------------------------------------------------ #


class TestPredictParallel:
    """Tests for VisualSearchModel.predict_parallel()."""

    def test_parallel_near_base(self) -> None:
        """Parallel search RT should be close to BASE_RT.

        Pop-out search has only a small logarithmic set-size effect.
        """
        rt = VisualSearchModel.predict_parallel(50)
        assert rt < VisualSearchModel.BASE_RT + 0.020

    def test_parallel_log_increase(self) -> None:
        """Parallel search time increases with log2(n_items).

        RT = base + 0.002 * log2(n).
        """
        expected = VisualSearchModel.BASE_RT + 0.002 * math.log2(10)
        result = VisualSearchModel.predict_parallel(10)
        assert result == pytest.approx(expected, rel=1e-9)


# ------------------------------------------------------------------ #
# Serial is slower than parallel
# ------------------------------------------------------------------ #


class TestSerialVsParallel:
    """Serial search must be slower than parallel for same n_items."""

    def test_serial_slower_than_parallel(self) -> None:
        """For any reasonable n_items, serial > parallel.

        This reflects the fundamental prediction of Feature Integration Theory.
        """
        for n in [5, 10, 20, 50]:
            serial = VisualSearchModel.predict_serial(n, target_present=True)
            parallel = VisualSearchModel.predict_parallel(n)
            assert serial > parallel, f"Serial should be slower at n={n}"


# ------------------------------------------------------------------ #
# Guided search
# ------------------------------------------------------------------ #


class TestPredictGuided:
    """Tests for VisualSearchModel.predict_guided()."""

    def test_no_guidance_equals_serial(self) -> None:
        """guidance_factor=0 → no filtering → equivalent to serial present.

        With zero guidance the effective set size equals n.
        """
        serial = VisualSearchModel.predict_serial(20, target_present=True)
        guided = VisualSearchModel.predict_guided(20, guidance_factor=0.0)
        assert guided == pytest.approx(serial, rel=1e-6)

    def test_perfect_guidance_fast(self) -> None:
        """guidance_factor=1.0 → effective set size = 1 → fast RT.

        Perfect guidance filters all distractors pre-attentively.
        """
        result = VisualSearchModel.predict_guided(100, guidance_factor=1.0)
        expected = VisualSearchModel.BASE_RT + 0.025 * 1.0 / 2.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_guidance_reduces_time(self) -> None:
        """More guidance should reduce search time."""
        low_guidance = VisualSearchModel.predict_guided(30, 0.2)
        high_guidance = VisualSearchModel.predict_guided(30, 0.8)
        assert high_guidance < low_guidance


# ------------------------------------------------------------------ #
# Saliency from structure
# ------------------------------------------------------------------ #


class TestSaliencyFromStructure:
    """Tests for VisualSearchModel.saliency_from_structure()."""

    def test_single_element_max_saliency(self) -> None:
        """A solitary element with no siblings should have saliency 1.0."""
        bbox = BoundingBox(10, 10, 50, 50)
        result = VisualSearchModel.saliency_from_structure(bbox, [])
        assert result == pytest.approx(1.0)

    def test_saliency_in_unit_interval(self) -> None:
        """Saliency score must lie in [0, 1]."""
        element = BoundingBox(100, 100, 50, 30)
        siblings = [
            BoundingBox(10, 10, 50, 30),
            BoundingBox(200, 200, 50, 30),
            BoundingBox(50, 300, 50, 30),
        ]
        result = VisualSearchModel.saliency_from_structure(element, siblings)
        assert 0.0 <= result <= 1.0

    def test_large_element_high_saliency(self) -> None:
        """An element much larger than siblings should have higher saliency.

        Size ratio is a weighted component of the saliency score.
        """
        big = BoundingBox(100, 100, 200, 200)
        small_siblings = [
            BoundingBox(10, 10, 20, 20),
            BoundingBox(400, 10, 20, 20),
            BoundingBox(10, 400, 20, 20),
        ]
        sal_big = VisualSearchModel.saliency_from_structure(big, small_siblings)

        normal = BoundingBox(100, 100, 20, 20)
        sal_normal = VisualSearchModel.saliency_from_structure(normal, small_siblings)
        assert sal_big > sal_normal


# ------------------------------------------------------------------ #
# Effective set size
# ------------------------------------------------------------------ #


class TestEffectiveSetSize:
    """Tests for VisualSearchModel.effective_set_size()."""

    def test_all_same_label(self) -> None:
        """When all items share the target label, effective size = n."""
        boxes = [BoundingBox(i * 60, 0, 50, 50) for i in range(5)]
        labels = ["btn"] * 5
        result = VisualSearchModel.effective_set_size(boxes, labels, "btn")
        assert result == pytest.approx(5.0)

    def test_all_different_label(self) -> None:
        """When no items share the target label, guidance discount applies.

        effective = 0 * 1.0 + n * 0.3.
        """
        boxes = [BoundingBox(i * 60, 0, 50, 50) for i in range(10)]
        labels = ["icon"] * 10
        result = VisualSearchModel.effective_set_size(boxes, labels, "btn")
        expected = max(1.0, 0 + 0.3 * 10)
        assert result == pytest.approx(expected)

    def test_empty_returns_one(self) -> None:
        """Empty element list should return effective set size of 1."""
        result = VisualSearchModel.effective_set_size([], [], "btn")
        assert result == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# Eccentricity cost
# ------------------------------------------------------------------ #


class TestEccentricityCost:
    """Tests for VisualSearchModel.eccentricity_cost()."""

    def test_zero_eccentricity(self) -> None:
        """Fixation on target → cost factor = 1.0 (no penalty)."""
        p = Point2D(100, 100)
        result = VisualSearchModel.eccentricity_cost(p, p)
        assert result == pytest.approx(1.0)

    def test_positive_eccentricity(self) -> None:
        """Greater eccentricity → cost factor > 1.0."""
        fix = Point2D(0, 0)
        target = Point2D(380, 0)  # 380 px ≈ 10 degrees
        result = VisualSearchModel.eccentricity_cost(fix, target)
        assert result > 1.0

    def test_cost_increases_with_distance(self) -> None:
        """Cost factor must increase monotonically with eccentricity."""
        fix = Point2D(0, 0)
        costs = [
            VisualSearchModel.eccentricity_cost(fix, Point2D(d, 0))
            for d in [0, 100, 200, 400]
        ]
        for i in range(len(costs) - 1):
            assert costs[i] <= costs[i + 1]


# ------------------------------------------------------------------ #
# Search time distribution
# ------------------------------------------------------------------ #


class TestSearchTimeDistribution:
    """Tests for VisualSearchModel.search_time_distribution()."""

    def test_returns_tuple(self) -> None:
        """search_time_distribution() should return (mean, variance)."""
        result = VisualSearchModel.search_time_distribution(20, 0.025)
        assert isinstance(result, tuple) and len(result) == 2

    def test_mean_matches_predict_serial(self) -> None:
        """Mean from distribution should match predict_serial()."""
        mean, _ = VisualSearchModel.search_time_distribution(
            20, 0.025, target_present=True
        )
        serial = VisualSearchModel.predict_serial(20, target_present=True)
        assert mean == pytest.approx(serial, rel=1e-9)

    def test_variance_positive(self) -> None:
        """Variance should be non-negative for any valid inputs."""
        _, var = VisualSearchModel.search_time_distribution(10, 0.025)
        assert var >= 0.0


# ------------------------------------------------------------------ #
# Predict with eccentricity
# ------------------------------------------------------------------ #


class TestPredictWithEccentricity:
    """Tests for VisualSearchModel.predict_with_eccentricity()."""

    def test_returns_float(self) -> None:
        """predict_with_eccentricity() should return a float."""
        elements = [BoundingBox(i * 60, 0, 50, 50) for i in range(5)]
        fix = Point2D(0, 0)
        result = VisualSearchModel.predict_with_eccentricity(
            elements, fix, target_idx=2, slope=0.025
        )
        assert isinstance(result, float)

    def test_empty_elements(self) -> None:
        """Empty element list should return BASE_RT."""
        result = VisualSearchModel.predict_with_eccentricity(
            [], Point2D(0, 0), 0, 0.025
        )
        assert result == pytest.approx(VisualSearchModel.BASE_RT)

    def test_target_near_fixation_faster(self) -> None:
        """Target near fixation should be found faster than a distant target.

        Elements sorted by eccentricity; near target found early.
        """
        elements = [
            BoundingBox(0, 0, 50, 50),     # near
            BoundingBox(500, 500, 50, 50),  # far
        ]
        fix = Point2D(25, 25)
        near = VisualSearchModel.predict_with_eccentricity(
            elements, fix, target_idx=0, slope=0.025
        )
        far = VisualSearchModel.predict_with_eccentricity(
            elements, fix, target_idx=1, slope=0.025
        )
        assert near <= far


# ------------------------------------------------------------------ #
# Number of fixations
# ------------------------------------------------------------------ #


class TestNumberOfFixations:
    """Tests for VisualSearchModel.number_of_fixations()."""

    def test_present_formula(self) -> None:
        """Target present → (n+1)/2 fixations."""
        result = VisualSearchModel.number_of_fixations(20, target_present=True)
        assert result == pytest.approx(10.5)

    def test_absent_formula(self) -> None:
        """Target absent → n fixations (exhaustive)."""
        result = VisualSearchModel.number_of_fixations(20, target_present=False)
        assert result == pytest.approx(20.0)


# ------------------------------------------------------------------ #
# Monotonicity
# ------------------------------------------------------------------ #


class TestMonotonicity:
    """Monotonicity: more items → more time."""

    def test_serial_monotone(self) -> None:
        """Serial search time should increase with n_items."""
        times = [
            VisualSearchModel.predict_serial(n, target_present=True)
            for n in [5, 10, 20, 40]
        ]
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1]

    def test_guided_monotone(self) -> None:
        """Guided search time should increase with n_items for fixed guidance."""
        times = [
            VisualSearchModel.predict_guided(n, guidance_factor=0.5)
            for n in [5, 10, 20, 40]
        ]
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1]

    def test_fixation_count_monotone(self) -> None:
        """Number of fixations should increase with n_items."""
        counts = [
            VisualSearchModel.number_of_fixations(n, True) for n in [5, 10, 20]
        ]
        for i in range(len(counts) - 1):
            assert counts[i] < counts[i + 1]

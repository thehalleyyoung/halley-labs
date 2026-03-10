"""Unit tests for usability_oracle.cognitive.perception.PerceptionModel.

Tests cover fixation time, saccade time, reading time, icon recognition,
perceptual grouping, color discrimination, visual span, text legibility,
peripheral detection, and pixel/degree conversion utilities.

References
----------
Rayner, K. (1998). Eye movements in reading and information processing.
    *Psychological Bulletin*, 124(3), 372-422.
Ware, C. (2012). *Information Visualization: Perception for Design* (3rd ed.).
    Morgan Kaufmann.
"""

from __future__ import annotations

import math

import pytest

from usability_oracle.cognitive.perception import PerceptionModel


@pytest.fixture
def model() -> PerceptionModel:
    """Create a PerceptionModel instance for use in tests."""
    return PerceptionModel()


# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #


class TestPerceptionConstants:
    """Verify class-level perceptual constants."""

    def test_mean_fixation(self) -> None:
        """Mean fixation duration should be 250 ms (Rayner, 1998)."""
        assert PerceptionModel.MEAN_FIXATION_DURATION == pytest.approx(0.250)

    def test_min_saccade(self) -> None:
        """Minimum saccade duration should be 20 ms."""
        assert PerceptionModel.MIN_SACCADE_DURATION == pytest.approx(0.020)

    def test_max_saccade(self) -> None:
        """Maximum saccade duration should be 200 ms."""
        assert PerceptionModel.MAX_SACCADE_DURATION == pytest.approx(0.200)

    def test_reading_rate(self) -> None:
        """Typical reading rate should be 250 WPM (Rayner, 1998)."""
        assert PerceptionModel.READING_RATE_WPM == 250


# ------------------------------------------------------------------ #
# Fixation time
# ------------------------------------------------------------------ #


class TestFixationTime:
    """Tests for PerceptionModel.fixation_time()."""

    def test_default_complexity(self, model: PerceptionModel) -> None:
        """complexity=1.0 → mean fixation duration = 0.250 s."""
        result = model.fixation_time(1.0)
        assert result == pytest.approx(0.250)

    def test_high_complexity(self, model: PerceptionModel) -> None:
        """Higher complexity → longer fixation, clamped to 0.600 s."""
        result = model.fixation_time(3.0)
        assert result <= 0.600

    def test_low_complexity(self, model: PerceptionModel) -> None:
        """Very low complexity → fixation clamped to at least 0.100 s."""
        result = model.fixation_time(0.1)
        assert result >= 0.100

    def test_negative_complexity_raises(self, model: PerceptionModel) -> None:
        """Negative complexity should raise ValueError."""
        with pytest.raises(ValueError, match="complexity must be non-negative"):
            model.fixation_time(-1.0)

    def test_fixation_monotone(self, model: PerceptionModel) -> None:
        """Fixation time should increase with complexity (within clamp)."""
        t1 = model.fixation_time(0.5)
        t2 = model.fixation_time(1.0)
        t3 = model.fixation_time(2.0)
        assert t1 <= t2 <= t3


# ------------------------------------------------------------------ #
# Saccade time
# ------------------------------------------------------------------ #


class TestSaccadeTime:
    """Tests for PerceptionModel.saccade_time()."""

    def test_zero_distance(self, model: PerceptionModel) -> None:
        """Zero amplitude → minimum saccade duration (20 ms)."""
        result = model.saccade_time(0.0)
        assert result == pytest.approx(0.020)

    def test_moderate_distance(self, model: PerceptionModel) -> None:
        """10 degrees → 0.020 + 0.002 * 10 = 0.040 s."""
        result = model.saccade_time(10.0)
        assert result == pytest.approx(0.040, rel=1e-6)

    def test_clamped_max(self, model: PerceptionModel) -> None:
        """Very large saccade should be clamped to MAX_SACCADE_DURATION."""
        result = model.saccade_time(200.0)
        assert result == pytest.approx(0.200)

    def test_negative_raises(self, model: PerceptionModel) -> None:
        """Negative amplitude should raise ValueError."""
        with pytest.raises(ValueError, match="distance_degrees must be non-negative"):
            model.saccade_time(-5.0)

    def test_saccade_monotone(self, model: PerceptionModel) -> None:
        """Saccade duration should increase with amplitude (within clamp)."""
        t1 = model.saccade_time(1.0)
        t2 = model.saccade_time(5.0)
        t3 = model.saccade_time(15.0)
        assert t1 <= t2 <= t3


# ------------------------------------------------------------------ #
# Reading time
# ------------------------------------------------------------------ #


class TestReadingTime:
    """Tests for PerceptionModel.reading_time()."""

    def test_basic_reading(self, model: PerceptionModel) -> None:
        """250 words at 250 WPM → 60 seconds.

        time = (250 / 250) * 60 = 60 s.
        """
        result = model.reading_time(250)
        assert result == pytest.approx(60.0, rel=1e-6)

    def test_zero_words(self, model: PerceptionModel) -> None:
        """Zero words → 0 seconds."""
        assert model.reading_time(0) == pytest.approx(0.0)

    def test_complex_text_slower(self, model: PerceptionModel) -> None:
        """word_complexity > 1.0 should increase reading time."""
        normal = model.reading_time(100, word_complexity=1.0)
        technical = model.reading_time(100, word_complexity=1.5)
        assert technical > normal

    def test_negative_words_raises(self, model: PerceptionModel) -> None:
        """Negative word count should raise ValueError."""
        with pytest.raises(ValueError, match="n_words must be non-negative"):
            model.reading_time(-5)


# ------------------------------------------------------------------ #
# Icon recognition
# ------------------------------------------------------------------ #


class TestIconRecognition:
    """Tests for PerceptionModel.icon_recognition_time()."""

    def test_familiar_icon(self, model: PerceptionModel) -> None:
        """Highly familiar icon (1.0) → 0.200 s."""
        result = model.icon_recognition_time(1.0)
        assert result == pytest.approx(0.200, rel=1e-6)

    def test_unfamiliar_icon(self, model: PerceptionModel) -> None:
        """Completely unfamiliar icon (0.0) → 0.600 s."""
        result = model.icon_recognition_time(0.0)
        assert result == pytest.approx(0.600, rel=1e-6)

    def test_half_familiar(self, model: PerceptionModel) -> None:
        """Half familiar (0.5) → 0.200 + 0.400 * 0.5 = 0.400 s."""
        result = model.icon_recognition_time(0.5)
        assert result == pytest.approx(0.400, rel=1e-6)

    def test_invalid_familiarity_raises(self, model: PerceptionModel) -> None:
        """Familiarity outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="familiarity must be in"):
            model.icon_recognition_time(1.5)
        with pytest.raises(ValueError):
            model.icon_recognition_time(-0.1)


# ------------------------------------------------------------------ #
# Grouping time
# ------------------------------------------------------------------ #


class TestGroupingTime:
    """Tests for PerceptionModel.grouping_time()."""

    def test_zero_groups(self, model: PerceptionModel) -> None:
        """0 groups → base time = 0.100 s."""
        result = model.grouping_time(0)
        assert result == pytest.approx(0.100)

    def test_five_groups(self, model: PerceptionModel) -> None:
        """5 groups → 0.100 + 0.050 * 5 = 0.350 s."""
        result = model.grouping_time(5)
        assert result == pytest.approx(0.350, rel=1e-6)

    def test_negative_groups_raises(self, model: PerceptionModel) -> None:
        """Negative group count should raise ValueError."""
        with pytest.raises(ValueError, match="n_groups must be non-negative"):
            model.grouping_time(-1)

    def test_grouping_monotone(self, model: PerceptionModel) -> None:
        """More groups → more time."""
        times = [model.grouping_time(n) for n in [0, 2, 5, 10]]
        for i in range(len(times) - 1):
            assert times[i] <= times[i + 1]


# ------------------------------------------------------------------ #
# Color discrimination
# ------------------------------------------------------------------ #


class TestColorDiscrimination:
    """Tests for PerceptionModel.color_discrimination_time()."""

    def test_standard_contrast(self, model: PerceptionModel) -> None:
        """contrast=1.0 → 0.150 s."""
        result = model.color_discrimination_time(1.0)
        assert result == pytest.approx(0.150, rel=1e-6)

    def test_high_contrast_faster(self, model: PerceptionModel) -> None:
        """Higher contrast → faster discrimination."""
        high = model.color_discrimination_time(5.0)
        low = model.color_discrimination_time(0.5)
        assert high < low

    def test_clamped_low(self, model: PerceptionModel) -> None:
        """Very high contrast → clamped to minimum 0.100 s."""
        result = model.color_discrimination_time(10.0)
        assert result == pytest.approx(0.100)

    def test_zero_contrast_raises(self, model: PerceptionModel) -> None:
        """Zero contrast should raise ValueError."""
        with pytest.raises(ValueError, match="contrast must be positive"):
            model.color_discrimination_time(0.0)


# ------------------------------------------------------------------ #
# Visual span
# ------------------------------------------------------------------ #


class TestVisualSpan:
    """Tests for PerceptionModel.visual_span()."""

    def test_foveal_span(self, model: PerceptionModel) -> None:
        """At fovea (0 degrees) → 4 characters."""
        result = model.visual_span(0.0)
        assert result == 4

    def test_peripheral_decays(self, model: PerceptionModel) -> None:
        """Span should decrease with eccentricity."""
        foveal = model.visual_span(0.0)
        peripheral = model.visual_span(5.0)
        assert peripheral < foveal

    def test_far_periphery_zero(self, model: PerceptionModel) -> None:
        """Very large eccentricity → 0 recognizable characters."""
        result = model.visual_span(50.0)
        assert result == 0

    def test_negative_raises(self, model: PerceptionModel) -> None:
        """Negative eccentricity should raise ValueError."""
        with pytest.raises(ValueError, match="eccentricity_degrees must be non-negative"):
            model.visual_span(-1.0)


# ------------------------------------------------------------------ #
# Text legibility factor
# ------------------------------------------------------------------ #


class TestTextLegibility:
    """Tests for PerceptionModel.text_legibility_factor()."""

    def test_large_font_high_legibility(self, model: PerceptionModel) -> None:
        """A large font (e.g. 24pt) at standard distance → factor near 1.0."""
        result = model.text_legibility_factor(24.0)
        assert result > 0.9

    def test_tiny_font_low_legibility(self, model: PerceptionModel) -> None:
        """A tiny font (e.g. 4pt) → factor near 0.0."""
        result = model.text_legibility_factor(4.0)
        assert result < 0.5

    def test_legibility_in_unit_interval(self, model: PerceptionModel) -> None:
        """Legibility factor must be in [0, 1]."""
        for pt in [4, 8, 12, 16, 24, 48]:
            f = model.text_legibility_factor(float(pt))
            assert 0.0 <= f <= 1.0

    def test_invalid_font_raises(self, model: PerceptionModel) -> None:
        """Non-positive font size should raise ValueError."""
        with pytest.raises(ValueError, match="font_size_pt must be positive"):
            model.text_legibility_factor(0.0)


# ------------------------------------------------------------------ #
# Peripheral detection probability
# ------------------------------------------------------------------ #


class TestPeripheralDetection:
    """Tests for PerceptionModel.peripheral_detection_probability()."""

    def test_foveal_large_target(self, model: PerceptionModel) -> None:
        """At fovea with large target → high detection probability."""
        result = model.peripheral_detection_probability(0.0, 5.0)
        assert result > 0.9

    def test_peripheral_low_detection(self, model: PerceptionModel) -> None:
        """Far periphery with small target → low detection probability."""
        result = model.peripheral_detection_probability(30.0, 0.1)
        assert result < 0.2

    def test_probability_in_unit_interval(self, model: PerceptionModel) -> None:
        """Detection probability must be in [0, 1]."""
        for ecc in [0, 5, 15, 30]:
            for size in [0.1, 1.0, 5.0]:
                p = model.peripheral_detection_probability(float(ecc), size)
                assert 0.0 <= p <= 1.0

    def test_negative_eccentricity_raises(self, model: PerceptionModel) -> None:
        """Negative eccentricity should raise ValueError."""
        with pytest.raises(ValueError, match="eccentricity_degrees must be non-negative"):
            model.peripheral_detection_probability(-1.0, 1.0)

    def test_zero_target_size_raises(self, model: PerceptionModel) -> None:
        """Non-positive target size should raise ValueError."""
        with pytest.raises(ValueError, match="target_size_degrees must be positive"):
            model.peripheral_detection_probability(5.0, 0.0)


# ------------------------------------------------------------------ #
# Pixels ↔ degrees conversion
# ------------------------------------------------------------------ #


class TestPixelDegreeConversion:
    """Tests for pixels_to_degrees() and degrees_to_pixels() static methods."""

    def test_pixels_to_degrees_basic(self) -> None:
        """Known conversion at default viewing distance.

        96 px ≈ 1 inch ≈ 25.4 mm → arctan(25.4/(2*600)) * 2 * 180/π ≈ 2.42°.
        """
        result = PerceptionModel.pixels_to_degrees(96.0)
        expected = 2.0 * math.atan(25.4 / (2.0 * 600.0)) * (180.0 / math.pi)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_degrees_to_pixels_basic(self) -> None:
        """Inverse conversion should recover the original pixel count."""
        degrees = PerceptionModel.pixels_to_degrees(100.0)
        pixels = PerceptionModel.degrees_to_pixels(degrees)
        assert pixels == pytest.approx(100.0, rel=1e-3)

    def test_roundtrip(self) -> None:
        """pixels → degrees → pixels should be approximately identity."""
        for px in [10, 50, 200, 500]:
            deg = PerceptionModel.pixels_to_degrees(float(px))
            recovered = PerceptionModel.degrees_to_pixels(deg)
            assert recovered == pytest.approx(px, rel=1e-3)

    def test_zero_pixels(self) -> None:
        """Zero pixels → 0 degrees."""
        assert PerceptionModel.pixels_to_degrees(0.0) == pytest.approx(0.0)

    def test_invalid_distance_raises(self) -> None:
        """Non-positive viewing distance should raise ValueError."""
        with pytest.raises(ValueError, match="distance_mm must be positive"):
            PerceptionModel.pixels_to_degrees(100, distance_mm=0.0)
        with pytest.raises(ValueError, match="distance_mm must be positive"):
            PerceptionModel.degrees_to_pixels(5.0, distance_mm=0.0)


# ------------------------------------------------------------------ #
# Instance creation
# ------------------------------------------------------------------ #


class TestInstanceCreation:
    """Test that PerceptionModel can be instantiated."""

    def test_create_instance(self) -> None:
        """PerceptionModel() should create a usable instance."""
        model = PerceptionModel()
        assert isinstance(model, PerceptionModel)

    def test_instance_has_methods(self) -> None:
        """Instance should have all expected method names."""
        model = PerceptionModel()
        expected_methods = [
            "fixation_time", "saccade_time", "reading_time",
            "icon_recognition_time", "grouping_time",
            "color_discrimination_time", "visual_span",
            "text_legibility_factor", "peripheral_detection_probability",
            "pixels_to_degrees", "degrees_to_pixels",
        ]
        for name in expected_methods:
            assert hasattr(model, name), f"Missing method: {name}"

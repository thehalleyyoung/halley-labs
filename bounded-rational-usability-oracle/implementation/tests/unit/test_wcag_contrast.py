"""Unit tests for usability_oracle.wcag.contrast — colour contrast algorithms.

Tests Color creation, sRGB ↔ linear conversion, relative luminance,
contrast ratio, AA/AAA thresholds, large-text detection, closest-passing
colour search, and colour-vision deficiency simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from usability_oracle.wcag.contrast import (
    Color,
    ContrastResult,
    check_contrast,
    check_contrast_cvd,
    contrast_ratio,
    contrast_ratio_all_cvd,
    contrast_ratio_from_luminance,
    find_closest_passing_color,
    is_large_text,
    linear_to_srgb,
    relative_luminance,
    relative_luminance_from_rgb,
    simulate_cvd,
    srgb_to_linear,
)
from usability_oracle.wcag.types import ConformanceLevel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLACK = Color(0, 0, 0)
WHITE = Color(255, 255, 255)
MID_GRAY = Color(128, 128, 128)
RED = Color(255, 0, 0)
GREEN = Color(0, 128, 0)
BLUE = Color(0, 0, 255)


# ═══════════════════════════════════════════════════════════════════════════
# Color construction & validation
# ═══════════════════════════════════════════════════════════════════════════


class TestColorCreation:
    """Tests for Color dataclass creation and validation."""

    def test_valid_color(self) -> None:
        c = Color(100, 200, 50)
        assert c.r == 100
        assert c.g == 200
        assert c.b == 50
        assert c.a == 1.0

    def test_with_alpha(self) -> None:
        c = Color(0, 0, 0, 0.5)
        assert c.a == 0.5

    def test_out_of_range_r_raises(self) -> None:
        with pytest.raises(ValueError, match="Color.r"):
            Color(256, 0, 0)

    def test_negative_channel_raises(self) -> None:
        with pytest.raises(ValueError, match="Color.g"):
            Color(0, -1, 0)

    def test_alpha_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            Color(0, 0, 0, 1.5)

    def test_from_hex_6(self) -> None:
        c = Color.from_hex("#ff8800")
        assert c.r == 255
        assert c.g == 136
        assert c.b == 0

    def test_from_hex_8(self) -> None:
        c = Color.from_hex("#ff880080")
        assert c.r == 255
        assert c.a == pytest.approx(128 / 255.0, abs=0.01)

    def test_from_hex_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid hex"):
            Color.from_hex("#abc")

    def test_to_hex(self) -> None:
        assert Color(255, 0, 128).to_hex() == "#ff0080"

    def test_from_rgb_tuple(self) -> None:
        c = Color.from_rgb_tuple((10, 20, 30))
        assert c.r == 10
        assert c.g == 20
        assert c.b == 30


# ═══════════════════════════════════════════════════════════════════════════
# blend_over
# ═══════════════════════════════════════════════════════════════════════════


class TestBlendOver:
    """Tests for alpha compositing."""

    def test_fully_opaque_returns_self(self) -> None:
        c = RED.blend_over(WHITE)
        assert c.r == 255
        assert c.g == 0

    def test_fully_transparent_returns_background(self) -> None:
        c = Color(255, 0, 0, 0.0).blend_over(WHITE)
        assert c.r == 255
        assert c.g == 255
        assert c.b == 255

    def test_half_alpha_blends(self) -> None:
        c = Color(0, 0, 0, 0.5).blend_over(WHITE)
        assert 120 <= c.r <= 135  # ~128


# ═══════════════════════════════════════════════════════════════════════════
# sRGB ↔ linear round-trip
# ═══════════════════════════════════════════════════════════════════════════


class TestSRGBConversion:
    """Tests for srgb_to_linear and linear_to_srgb round-trip."""

    def test_black_roundtrip(self) -> None:
        lin = srgb_to_linear(np.array([0, 0, 0], dtype=np.float64))
        srgb = linear_to_srgb(lin)
        np.testing.assert_array_equal(srgb, [0, 0, 0])

    def test_white_roundtrip(self) -> None:
        lin = srgb_to_linear(np.array([255, 255, 255], dtype=np.float64))
        srgb = linear_to_srgb(lin)
        np.testing.assert_array_equal(srgb, [255, 255, 255])

    def test_midtone_roundtrip(self) -> None:
        original = np.array([128, 64, 200], dtype=np.float64)
        lin = srgb_to_linear(original)
        srgb = linear_to_srgb(lin)
        np.testing.assert_allclose(srgb, original, atol=1)

    def test_linear_values_in_01(self) -> None:
        lin = srgb_to_linear(np.array([128, 128, 128], dtype=np.float64))
        assert np.all(lin >= 0.0)
        assert np.all(lin <= 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Relative luminance
# ═══════════════════════════════════════════════════════════════════════════


class TestRelativeLuminance:
    """Tests for WCAG relative luminance."""

    def test_black_luminance(self) -> None:
        assert relative_luminance(BLACK) == pytest.approx(0.0, abs=1e-6)

    def test_white_luminance(self) -> None:
        assert relative_luminance(WHITE) == pytest.approx(1.0, abs=1e-4)

    def test_from_rgb_matches(self) -> None:
        l1 = relative_luminance(MID_GRAY)
        l2 = relative_luminance_from_rgb(128, 128, 128)
        assert l1 == pytest.approx(l2, abs=1e-6)

    def test_luminance_in_range(self) -> None:
        for c in [BLACK, WHITE, RED, GREEN, BLUE, MID_GRAY]:
            lum = relative_luminance(c)
            assert 0.0 <= lum <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Contrast ratio
# ═══════════════════════════════════════════════════════════════════════════


class TestContrastRatio:
    """Tests for contrast_ratio and contrast_ratio_from_luminance."""

    def test_black_on_white(self) -> None:
        ratio = contrast_ratio(BLACK, WHITE)
        assert ratio == pytest.approx(21.0, abs=0.1)

    def test_same_color(self) -> None:
        ratio = contrast_ratio(RED, RED)
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_symmetric(self) -> None:
        r1 = contrast_ratio(RED, WHITE)
        r2 = contrast_ratio(WHITE, RED)
        assert r1 == pytest.approx(r2, abs=0.01)

    def test_from_luminance(self) -> None:
        ratio = contrast_ratio_from_luminance(1.0, 0.0)
        assert ratio == pytest.approx(21.0, abs=0.1)

    def test_minimum_ratio_is_one(self) -> None:
        ratio = contrast_ratio_from_luminance(0.5, 0.5)
        assert ratio == pytest.approx(1.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# check_contrast — AA/AAA thresholds
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckContrast:
    """Tests for check_contrast with WCAG threshold evaluation."""

    def test_black_on_white_passes_all(self) -> None:
        result = check_contrast(BLACK, WHITE)
        assert isinstance(result, ContrastResult)
        assert result.passes_aa_normal is True
        assert result.passes_aaa_normal is True

    def test_low_contrast_fails_aa(self) -> None:
        light_gray = Color(200, 200, 200)
        result = check_contrast(light_gray, WHITE)
        assert result.passes_aa_normal is False

    def test_large_text_relaxed_threshold(self) -> None:
        # Ratio >= 3.0 should pass AA for large text
        fg = Color(119, 119, 119)  # ~4.48:1 against white
        result = check_contrast(fg, WHITE, font_size_px=24.0)
        assert result.passes_aa_large is True

    def test_minimum_level_property(self) -> None:
        result = check_contrast(BLACK, WHITE)
        assert result.minimum_level == ConformanceLevel.AAA

    def test_minimum_level_none_for_low_contrast(self) -> None:
        result = check_contrast(Color(200, 200, 200), WHITE)
        assert result.minimum_level is None

    def test_semi_transparent_fg_blended(self) -> None:
        fg = Color(0, 0, 0, 0.5)
        result = check_contrast(fg, WHITE)
        assert 1.0 < result.ratio < 21.0


# ═══════════════════════════════════════════════════════════════════════════
# is_large_text
# ═══════════════════════════════════════════════════════════════════════════


class TestIsLargeText:
    """Tests for is_large_text determination."""

    def test_24px_normal_is_large(self) -> None:
        assert is_large_text(24.0, is_bold=False) is True

    def test_18_67px_bold_is_large(self) -> None:
        assert is_large_text(18.67, is_bold=True) is True

    def test_16px_normal_is_not_large(self) -> None:
        assert is_large_text(16.0, is_bold=False) is False

    def test_18px_not_bold_is_not_large(self) -> None:
        assert is_large_text(18.0, is_bold=False) is False

    def test_14px_bold_is_not_large(self) -> None:
        assert is_large_text(14.0, is_bold=True) is False


# ═══════════════════════════════════════════════════════════════════════════
# find_closest_passing_color
# ═══════════════════════════════════════════════════════════════════════════


class TestFindClosestPassingColor:
    """Tests for binary-search colour adjustment."""

    def test_already_passing_returns_original(self) -> None:
        result = find_closest_passing_color(BLACK, WHITE, target_ratio=4.5)
        assert contrast_ratio(result, WHITE) >= 4.5

    def test_failing_returns_adjusted(self) -> None:
        fg = Color(200, 200, 200)
        result = find_closest_passing_color(fg, WHITE, target_ratio=4.5)
        assert contrast_ratio(result, WHITE) >= 4.5

    def test_adjusted_is_valid_color(self) -> None:
        result = find_closest_passing_color(MID_GRAY, WHITE, target_ratio=7.0)
        assert 0 <= result.r <= 255
        assert 0 <= result.g <= 255
        assert 0 <= result.b <= 255


# ═══════════════════════════════════════════════════════════════════════════
# Colour-vision deficiency simulation
# ═══════════════════════════════════════════════════════════════════════════


class TestCVDSimulation:
    """Tests for simulate_cvd, check_contrast_cvd, contrast_ratio_all_cvd."""

    def test_simulate_protanopia(self) -> None:
        sim = simulate_cvd(RED, "protanopia")
        assert isinstance(sim, Color)
        assert sim.r != RED.r or sim.g != RED.g  # should differ from original

    def test_simulate_deuteranopia(self) -> None:
        sim = simulate_cvd(GREEN, "deuteranopia")
        assert isinstance(sim, Color)

    def test_simulate_tritanopia(self) -> None:
        sim = simulate_cvd(BLUE, "tritanopia")
        assert isinstance(sim, Color)

    def test_invalid_deficiency_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown deficiency"):
            simulate_cvd(RED, "invalid_type")

    def test_black_unchanged_by_cvd(self) -> None:
        for deficiency in ("protanopia", "deuteranopia", "tritanopia"):
            sim = simulate_cvd(BLACK, deficiency)
            assert sim.r <= 5 and sim.g <= 5 and sim.b <= 5

    def test_check_contrast_cvd_returns_result(self) -> None:
        result = check_contrast_cvd(BLACK, WHITE, "protanopia")
        assert isinstance(result, ContrastResult)
        assert result.ratio > 1.0

    def test_contrast_ratio_all_cvd_keys(self) -> None:
        ratios = contrast_ratio_all_cvd(BLACK, WHITE)
        assert "normal" in ratios
        assert "protanopia" in ratios
        assert "deuteranopia" in ratios
        assert "tritanopia" in ratios

    def test_contrast_ratio_all_cvd_black_white_high(self) -> None:
        ratios = contrast_ratio_all_cvd(BLACK, WHITE)
        for key, ratio in ratios.items():
            assert ratio > 15.0, f"{key} contrast too low: {ratio}"

"""
usability_oracle.wcag.contrast — Color contrast algorithms per WCAG 2.2.

Implements relative luminance, contrast ratio, sRGB linearisation, and
color-blindness simulation matrices following the W3C WCAG 2.2 specification
and the Brettel/Viénot/Mollon (1997) simulation model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from usability_oracle.wcag.types import ConformanceLevel


# ═══════════════════════════════════════════════════════════════════════════
# sRGB colour representation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Color:
    """An sRGB colour with components in [0, 255]."""

    r: int
    g: int
    b: int
    a: float = 1.0  # alpha, 0–1

    def __post_init__(self) -> None:
        for ch, val in [("r", self.r), ("g", self.g), ("b", self.b)]:
            if not (0 <= val <= 255):
                raise ValueError(f"Color.{ch} must be in [0, 255], got {val}")
        if not (0.0 <= self.a <= 1.0):
            raise ValueError(f"Color.a must be in [0, 1], got {self.a}")

    # -- conversions --------------------------------------------------------

    def to_linear(self) -> np.ndarray:
        """Convert sRGB [0, 255] to linear-light RGB [0, 1]."""
        return srgb_to_linear(np.array([self.r, self.g, self.b], dtype=np.float64))

    @classmethod
    def from_hex(cls, hex_str: str) -> Color:
        """Parse ``#RRGGBB`` or ``#RRGGBBAA``."""
        h = hex_str.lstrip("#")
        if len(h) == 6:
            return cls(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        if len(h) == 8:
            return cls(
                int(h[0:2], 16),
                int(h[2:4], 16),
                int(h[4:6], 16),
                int(h[6:8], 16) / 255.0,
            )
        raise ValueError(f"Invalid hex colour: {hex_str!r}")

    @classmethod
    def from_rgb_tuple(cls, t: Tuple[int, int, int]) -> Color:
        return cls(r=t[0], g=t[1], b=t[2])

    def to_hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def blend_over(self, background: Color) -> Color:
        """Alpha-composite *self* over *background* (both fully opaque result)."""
        a = self.a
        inv = 1.0 - a
        return Color(
            r=int(round(self.r * a + background.r * inv)),
            g=int(round(self.g * a + background.g * inv)),
            b=int(round(self.b * a + background.b * inv)),
            a=1.0,
        )


# ═══════════════════════════════════════════════════════════════════════════
# sRGB ↔ linear conversion (IEC 61966-2-1)
# ═══════════════════════════════════════════════════════════════════════════

def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB channel values [0, 255] to linear-light [0, 1].

    Applies the piecewise transfer function specified in IEC 61966-2-1.
    """
    c = srgb.astype(np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear-light [0, 1] back to sRGB [0, 255]."""
    c = np.clip(linear, 0.0, 1.0)
    srgb_float = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)
    return np.clip(np.round(srgb_float * 255.0), 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# Relative luminance (WCAG 2.2 §1.4.3)
# ═══════════════════════════════════════════════════════════════════════════

# ITU-R BT.709 luminance coefficients
_LUMA_COEFFS = np.array([0.2126, 0.7152, 0.0722])


def relative_luminance(color: Color) -> float:
    """Compute WCAG relative luminance in [0, 1].

    .. math::
        L = 0.2126 \\cdot R_{lin} + 0.7152 \\cdot G_{lin} + 0.0722 \\cdot B_{lin}

    where each channel is first linearised from sRGB via
    :func:`srgb_to_linear`.
    """
    linear = color.to_linear()
    return float(np.dot(_LUMA_COEFFS, linear))


def relative_luminance_from_rgb(r: int, g: int, b: int) -> float:
    """Convenience wrapper accepting raw channel values."""
    return relative_luminance(Color(r, g, b))


# ═══════════════════════════════════════════════════════════════════════════
# Contrast ratio (WCAG 2.2 §1.4.3)
# ═══════════════════════════════════════════════════════════════════════════

def contrast_ratio(fg: Color, bg: Color) -> float:
    """WCAG contrast ratio between two colours.

    .. math::
        CR = \\frac{L_1 + 0.05}{L_2 + 0.05}

    where *L₁* ≥ *L₂*.  Result is in [1, 21].
    """
    l1 = relative_luminance(fg)
    l2 = relative_luminance(bg)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def contrast_ratio_from_luminance(l1: float, l2: float) -> float:
    """Contrast ratio given pre-computed luminance values."""
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


# ═══════════════════════════════════════════════════════════════════════════
# AA / AAA threshold checking
# ═══════════════════════════════════════════════════════════════════════════

# WCAG 2.2 thresholds
_AA_NORMAL_TEXT = 4.5
_AA_LARGE_TEXT = 3.0
_AAA_NORMAL_TEXT = 7.0
_AAA_LARGE_TEXT = 4.5

# Large text: ≥ 18pt or ≥ 14pt bold (CSS px approximations)
_LARGE_TEXT_PX = 24.0   # 18pt ≈ 24px
_LARGE_BOLD_PX = 18.67  # 14pt ≈ 18.67px


def is_large_text(font_size_px: float, is_bold: bool = False) -> bool:
    """Determine whether text qualifies as 'large text' per WCAG."""
    if is_bold:
        return font_size_px >= _LARGE_BOLD_PX
    return font_size_px >= _LARGE_TEXT_PX


@dataclass(frozen=True, slots=True)
class ContrastResult:
    """Result of a contrast check."""

    ratio: float
    fg: Color
    bg: Color
    passes_aa_normal: bool
    passes_aa_large: bool
    passes_aaa_normal: bool
    passes_aaa_large: bool

    @property
    def minimum_level(self) -> Optional[ConformanceLevel]:
        """Highest conformance level this contrast ratio satisfies for normal text."""
        if self.passes_aaa_normal:
            return ConformanceLevel.AAA
        if self.passes_aa_normal:
            return ConformanceLevel.AA
        return None


def check_contrast(
    fg: Color,
    bg: Color,
    font_size_px: float = 16.0,
    is_bold: bool = False,
) -> ContrastResult:
    """Full contrast check against WCAG AA and AAA thresholds.

    Parameters
    ----------
    fg : Color
        Foreground (text) colour.
    bg : Color
        Background colour.
    font_size_px : float
        Font size in CSS pixels.
    is_bold : bool
        Whether the text is bold weight (≥ 700).

    Returns
    -------
    ContrastResult
        Detailed pass/fail for each threshold.
    """
    # Alpha-blend foreground onto background if semi-transparent
    effective_fg = fg.blend_over(bg) if fg.a < 1.0 else fg

    ratio = contrast_ratio(effective_fg, bg)
    large = is_large_text(font_size_px, is_bold)

    return ContrastResult(
        ratio=ratio,
        fg=effective_fg,
        bg=bg,
        passes_aa_normal=ratio >= _AA_NORMAL_TEXT,
        passes_aa_large=ratio >= _AA_LARGE_TEXT,
        passes_aaa_normal=ratio >= _AAA_NORMAL_TEXT,
        passes_aaa_large=ratio >= _AAA_LARGE_TEXT,
    )


def find_closest_passing_color(
    fg: Color,
    bg: Color,
    target_ratio: float = _AA_NORMAL_TEXT,
    max_iterations: int = 50,
) -> Color:
    """Binary-search for the closest foreground colour meeting a target ratio.

    Adjusts luminance along the fg→black or fg→white axis until the target
    contrast ratio is met, preserving hue and saturation as closely as possible.
    Tries both directions (darken and lighten) and returns the result closest
    to the original colour.
    """
    fg_linear = fg.to_linear()

    def _search(target_end: np.ndarray) -> tuple[Color, float]:
        lo, hi = 0.0, 1.0
        best: Color = fg
        best_dist = float("inf")
        for _ in range(max_iterations):
            mid = (lo + hi) / 2.0
            candidate_linear = fg_linear * (1.0 - mid) + target_end * mid
            candidate_srgb = linear_to_srgb(candidate_linear)
            candidate = Color(int(candidate_srgb[0]), int(candidate_srgb[1]), int(candidate_srgb[2]))
            ratio = contrast_ratio(candidate, bg)
            if ratio >= target_ratio:
                best = candidate
                best_dist = mid
                hi = mid  # stay closer to original
            else:
                lo = mid
        return best, best_dist

    dark_result, dark_dist = _search(np.array([0.0, 0.0, 0.0]))
    light_result, light_dist = _search(np.array([1.0, 1.0, 1.0]))

    # Check which direction actually found a passing colour
    dark_passes = contrast_ratio(dark_result, bg) >= target_ratio
    light_passes = contrast_ratio(light_result, bg) >= target_ratio

    if dark_passes and light_passes:
        return dark_result if dark_dist <= light_dist else light_result
    if dark_passes:
        return dark_result
    if light_passes:
        return light_result
    return fg  # neither direction could achieve the target (shouldn't happen)


# ═══════════════════════════════════════════════════════════════════════════
# Color-blindness simulation
# ═══════════════════════════════════════════════════════════════════════════

# Brettel/Viénot/Mollon (1997) simulation matrices in linear sRGB space.
# These transform linear-RGB to simulate how colours appear to individuals
# with each type of colour-vision deficiency.

_PROTANOPIA_MATRIX = np.array([
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281,  0.099216],
    [-0.003882, -0.048116, 1.051998],
])

_DEUTERANOPIA_MATRIX = np.array([
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501,  0.047413],
    [-0.011820, 0.042940, 0.968881],
])

_TRITANOPIA_MATRIX = np.array([
    [1.255528, -0.076749, -0.178779],
    [-0.078411, 0.930809, 0.147602],
    [0.004733, 0.691367, 0.303900],
])

_CVD_MATRICES = {
    "protanopia": _PROTANOPIA_MATRIX,
    "deuteranopia": _DEUTERANOPIA_MATRIX,
    "tritanopia": _TRITANOPIA_MATRIX,
}


def simulate_cvd(color: Color, deficiency: str) -> Color:
    """Simulate how a colour appears under a given colour-vision deficiency.

    Parameters
    ----------
    color : Color
        Original sRGB colour.
    deficiency : str
        One of ``"protanopia"``, ``"deuteranopia"``, ``"tritanopia"``.

    Returns
    -------
    Color
        Simulated colour as perceived by someone with the deficiency.
    """
    key = deficiency.lower()
    if key not in _CVD_MATRICES:
        raise ValueError(
            f"Unknown deficiency {deficiency!r}; "
            f"expected one of {list(_CVD_MATRICES.keys())}"
        )
    mat = _CVD_MATRICES[key]
    linear = color.to_linear()
    sim_linear = mat @ linear
    sim_srgb = linear_to_srgb(np.clip(sim_linear, 0.0, 1.0))
    return Color(int(sim_srgb[0]), int(sim_srgb[1]), int(sim_srgb[2]))


def check_contrast_cvd(
    fg: Color,
    bg: Color,
    deficiency: str,
    font_size_px: float = 16.0,
    is_bold: bool = False,
) -> ContrastResult:
    """Check contrast as perceived under a colour-vision deficiency."""
    sim_fg = simulate_cvd(fg, deficiency)
    sim_bg = simulate_cvd(bg, deficiency)
    return check_contrast(sim_fg, sim_bg, font_size_px, is_bold)


def contrast_ratio_all_cvd(
    fg: Color,
    bg: Color,
) -> dict[str, float]:
    """Return contrast ratios for normal vision and each CVD type."""
    result: dict[str, float] = {"normal": contrast_ratio(fg, bg)}
    for deficiency in _CVD_MATRICES:
        sim_fg = simulate_cvd(fg, deficiency)
        sim_bg = simulate_cvd(bg, deficiency)
        result[deficiency] = contrast_ratio(sim_fg, sim_bg)
    return result


__all__ = [
    "Color",
    "ContrastResult",
    "check_contrast",
    "check_contrast_cvd",
    "contrast_ratio",
    "contrast_ratio_all_cvd",
    "contrast_ratio_from_luminance",
    "find_closest_passing_color",
    "is_large_text",
    "linear_to_srgb",
    "relative_luminance",
    "relative_luminance_from_rgb",
    "simulate_cvd",
    "srgb_to_linear",
]

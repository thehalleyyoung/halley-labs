"""
usability_oracle.visualization.colors — Accessible color schemes.

Provides WCAG 2.1 compliant color palettes for visualizations,
including schemes safe for various forms of color vision deficiency
(protanopia, deuteranopia, tritanopia).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Color:
    """RGB color with optional alpha channel."""
    r: int
    g: int
    b: int
    a: float = 1.0

    def hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def rgb_tuple(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)

    def rgba_tuple(self) -> tuple[int, int, int, float]:
        return (self.r, self.g, self.b, self.a)

    def relative_luminance(self) -> float:
        """WCAG 2.1 relative luminance."""
        def _linearize(c: int) -> float:
            s = c / 255.0
            return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4
        return 0.2126 * _linearize(self.r) + 0.7152 * _linearize(self.g) + 0.0722 * _linearize(self.b)

    def contrast_ratio(self, other: "Color") -> float:
        """WCAG 2.1 contrast ratio between two colors."""
        l1 = self.relative_luminance()
        l2 = other.relative_luminance()
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)

    def meets_aa(self, other: "Color", large_text: bool = False) -> bool:
        """Check if contrast meets WCAG AA (4.5:1 normal, 3:1 large text)."""
        threshold = 3.0 if large_text else 4.5
        return self.contrast_ratio(other) >= threshold

    def meets_aaa(self, other: "Color", large_text: bool = False) -> bool:
        """Check if contrast meets WCAG AAA (7:1 normal, 4.5:1 large text)."""
        threshold = 4.5 if large_text else 7.0
        return self.contrast_ratio(other) >= threshold

    def blend(self, other: "Color", factor: float = 0.5) -> "Color":
        """Blend this color with another."""
        f = max(0.0, min(1.0, factor))
        return Color(
            r=int(self.r * (1 - f) + other.r * f),
            g=int(self.g * (1 - f) + other.g * f),
            b=int(self.b * (1 - f) + other.b * f),
            a=self.a * (1 - f) + other.a * f,
        )

    def lighten(self, amount: float = 0.2) -> "Color":
        """Lighten the color."""
        return self.blend(Color(255, 255, 255), amount)

    def darken(self, amount: float = 0.2) -> "Color":
        """Darken the color."""
        return self.blend(Color(0, 0, 0), amount)

    def to_hsl(self) -> tuple[float, float, float]:
        """Convert to HSL (hue in degrees, saturation and lightness 0-1)."""
        r, g, b = self.r / 255.0, self.g / 255.0, self.b / 255.0
        mx, mn = max(r, g, b), min(r, g, b)
        l = (mx + mn) / 2.0
        if mx == mn:
            h = s = 0.0
        else:
            d = mx - mn
            s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
            if mx == r:
                h = (g - b) / d + (6 if g < b else 0)
            elif mx == g:
                h = (b - r) / d + 2
            else:
                h = (r - g) / d + 4
            h /= 6.0
        return (h * 360.0, s, l)


# ---------------------------------------------------------------------------
# Predefined colors
# ---------------------------------------------------------------------------

WHITE = Color(255, 255, 255)
BLACK = Color(0, 0, 0)

# Wong (2011) colorblind-safe palette
BLUE = Color(0, 114, 178)
ORANGE = Color(230, 159, 0)
GREEN = Color(0, 158, 115)
YELLOW = Color(240, 228, 66)
SKY_BLUE = Color(86, 180, 233)
VERMILLION = Color(213, 94, 0)
PURPLE = Color(204, 121, 167)

# Severity colors
SEVERITY_LOW = Color(76, 175, 80)
SEVERITY_MEDIUM = Color(255, 193, 7)
SEVERITY_HIGH = Color(244, 67, 54)
SEVERITY_CRITICAL = Color(183, 28, 28)

# Status colors
STATUS_PASS = Color(46, 125, 50)
STATUS_WARN = Color(245, 124, 0)
STATUS_FAIL = Color(198, 40, 40)
STATUS_INFO = Color(25, 118, 210)


# ---------------------------------------------------------------------------
# ColorScheme
# ---------------------------------------------------------------------------

@dataclass
class ColorScheme:
    """Named color scheme with semantic roles.

    Attributes:
        name: Scheme name.
        background: Background color.
        foreground: Primary text color.
        primary: Primary accent color.
        secondary: Secondary accent color.
        success: Success/pass color.
        warning: Warning color.
        error: Error/fail color.
        info: Informational color.
        palette: Ordered list of categorical colors.
    """
    name: str = "default"
    background: Color = WHITE
    foreground: Color = BLACK
    primary: Color = BLUE
    secondary: Color = SKY_BLUE
    success: Color = STATUS_PASS
    warning: Color = STATUS_WARN
    error: Color = STATUS_FAIL
    info: Color = STATUS_INFO
    palette: list[Color] = field(default_factory=lambda: [
        BLUE, ORANGE, GREEN, PURPLE, VERMILLION, SKY_BLUE, YELLOW,
    ])

    def get_color(self, index: int) -> Color:
        """Get a categorical color by index (wraps around)."""
        return self.palette[index % len(self.palette)]

    def severity_color(self, severity: float) -> Color:
        """Map a severity value (0-1) to a color."""
        if severity < 0.25:
            return SEVERITY_LOW
        elif severity < 0.5:
            return SEVERITY_MEDIUM
        elif severity < 0.75:
            return SEVERITY_HIGH
        return SEVERITY_CRITICAL

    def gradient(self, n: int, start: Color | None = None, end: Color | None = None) -> list[Color]:
        """Generate a gradient of n colors."""
        c1 = start or self.primary
        c2 = end or self.error
        return [c1.blend(c2, i / max(n - 1, 1)) for i in range(n)]

    def sequential_palette(self, n: int, base: Color | None = None) -> list[Color]:
        """Generate a sequential (light-to-dark) palette."""
        c = base or self.primary
        return [c.lighten(0.8 - 0.8 * i / max(n - 1, 1)) for i in range(n)]

    def diverging_palette(self, n: int) -> list[Color]:
        """Generate a diverging palette (low-neutral-high)."""
        half = n // 2
        low_colors = [self.success.lighten(0.7 - 0.7 * i / max(half, 1)) for i in range(half)]
        high_colors = [self.error.lighten(0.7 * i / max(half, 1)) for i in range(half)]
        mid = [Color(245, 245, 245)] if n % 2 == 1 else []
        return low_colors + mid + high_colors

    def validate_contrast(self) -> list[str]:
        """Check WCAG AA contrast for all scheme colors against background."""
        issues = []
        checks = {
            "foreground": self.foreground,
            "primary": self.primary,
            "secondary": self.secondary,
            "success": self.success,
            "warning": self.warning,
            "error": self.error,
            "info": self.info,
        }
        for name, color in checks.items():
            ratio = color.contrast_ratio(self.background)
            if ratio < 4.5:
                issues.append(f"{name} ({color.hex()}) has contrast ratio {ratio:.2f} < 4.5:1")
        return issues


# ---------------------------------------------------------------------------
# Predefined accessible schemes
# ---------------------------------------------------------------------------

ACCESSIBLE_PALETTE = ColorScheme(
    name="accessible",
    background=WHITE,
    foreground=Color(33, 33, 33),
    primary=BLUE,
    secondary=SKY_BLUE,
    success=STATUS_PASS,
    warning=STATUS_WARN,
    error=STATUS_FAIL,
    info=STATUS_INFO,
    palette=[BLUE, ORANGE, GREEN, PURPLE, VERMILLION, SKY_BLUE, YELLOW],
)

DARK_SCHEME = ColorScheme(
    name="dark",
    background=Color(30, 30, 30),
    foreground=Color(230, 230, 230),
    primary=Color(100, 181, 246),
    secondary=Color(128, 203, 196),
    success=Color(129, 199, 132),
    warning=Color(255, 213, 79),
    error=Color(239, 154, 154),
    info=Color(144, 202, 249),
    palette=[
        Color(100, 181, 246), Color(255, 183, 77), Color(129, 199, 132),
        Color(206, 147, 216), Color(255, 138, 101), Color(128, 203, 196),
        Color(255, 241, 118),
    ],
)

HIGH_CONTRAST_SCHEME = ColorScheme(
    name="high_contrast",
    background=Color(0, 0, 0),
    foreground=Color(255, 255, 255),
    primary=Color(0, 255, 255),
    secondary=Color(255, 255, 0),
    success=Color(0, 255, 0),
    warning=Color(255, 255, 0),
    error=Color(255, 0, 0),
    info=Color(0, 200, 255),
)

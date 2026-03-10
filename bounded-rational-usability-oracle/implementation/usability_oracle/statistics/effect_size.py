"""
usability_oracle.statistics.effect_size — Effect size estimation.

Standardised effect-size measures quantify the practical significance
of usability changes:

- cohens_d: standardised mean difference  d = (μ₁ − μ₂) / s_pooled
- hedges_g: bias-corrected Cohen's d  g = d · J(df)
- glass_delta: using control group SD  Δ = (μ₁ − μ₂) / s_control
- cliff_delta: non-parametric ordinal  δ = (#{X > Y} − #{X < Y}) / (n·m)
- common_language_effect_size: P(X > Y)
- robust_effect_size: trimmed-means version

Each function returns an EffectSize with bootstrap CI and interpretation.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats as sp_stats

from usability_oracle.statistics.types import (
    ConfidenceInterval,
    EffectSize,
    EffectSizeType,
)


# ---------------------------------------------------------------------------
# Interpretation thresholds
# ---------------------------------------------------------------------------

_THRESHOLDS_D = {"small": 0.2, "medium": 0.5, "large": 0.8}
_THRESHOLDS_CLIFF = {"small": 0.147, "medium": 0.33, "large": 0.474}
_THRESHOLDS_CL = {"small": 0.56, "medium": 0.64, "large": 0.71}


def _interpret(value: float, thresholds: dict[str, float]) -> str:
    """Qualitative interpretation given thresholds on |value|."""
    av = abs(value)
    if av < thresholds["small"]:
        return "negligible"
    if av < thresholds["medium"]:
        return "small"
    if av < thresholds["large"]:
        return "medium"
    return "large"


def _interpret_cl(value: float) -> str:
    """Interpret common-language effect size (probability scale)."""
    diff = abs(value - 0.5)
    if diff < 0.06:
        return "negligible"
    if diff < 0.14:
        return "small"
    if diff < 0.21:
        return "medium"
    return "large"


def interpret(value: float, measure: EffectSizeType) -> str:
    """Return qualitative interpretation of an effect size.

    Parameters:
        value: Numerical effect-size value.
        measure: Which metric the value corresponds to.

    Returns:
        One of "negligible", "small", "medium", "large".
    """
    if measure in (EffectSizeType.COHENS_D, EffectSizeType.HEDGES_G, EffectSizeType.GLASS_DELTA):
        return _interpret(value, _THRESHOLDS_D)
    if measure == EffectSizeType.CLIFFS_DELTA:
        return _interpret(value, _THRESHOLDS_CLIFF)
    if measure == EffectSizeType.COMMON_LANGUAGE:
        return _interpret_cl(value)
    return _interpret(value, _THRESHOLDS_D)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_arrays(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("Each sample must have at least 2 observations.")
    return a, b


def _bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Compute a bootstrap percentile CI for an effect-size statistic."""
    rng = np.random.default_rng(42)
    boot_vals = np.empty(n_bootstrap)
    nx, ny = len(x), len(y)
    for i in range(n_bootstrap):
        ix = rng.integers(0, nx, size=nx)
        iy = rng.integers(0, ny, size=ny)
        boot_vals[i] = stat_fn(x[ix], y[iy])
    alpha = 1.0 - confidence_level
    lo = float(np.percentile(boot_vals, 100 * alpha / 2))
    hi = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    point = stat_fn(x, y)
    return ConfidenceInterval(
        lower=lo, upper=hi, level=confidence_level,
        point_estimate=point, method="bootstrap percentile",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Cohen's d
# ═══════════════════════════════════════════════════════════════════════════

def _cohens_d_raw(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d = (ȳ − x̄) / s_pooled."""
    n1, n2 = len(x), len(y)
    v1, v2 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0.0:
        return 0.0
    return (float(np.mean(y)) - float(np.mean(x))) / pooled


def cohens_d(
    x: Sequence[float],
    y: Sequence[float],
    confidence_level: float = 0.95,
) -> EffectSize:
    """Compute Cohen's d with bootstrap CI.

    d = (ȳ − x̄) / s_pooled

    where s_pooled = √[((n₁−1)·s₁² + (n₂−1)·s₂²) / (n₁+n₂−2)].
    """
    a, b = _to_arrays(x, y)
    d = _cohens_d_raw(a, b)
    ci = _bootstrap_ci(a, b, _cohens_d_raw, confidence_level=confidence_level)
    return EffectSize(
        measure=EffectSizeType.COHENS_D,
        value=d,
        ci=ci,
        interpretation=interpret(d, EffectSizeType.COHENS_D),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Hedges' g
# ═══════════════════════════════════════════════════════════════════════════

def _hedges_correction(df: int) -> float:
    """Hedges' J correction factor: J(df) ≈ 1 − 3/(4·df − 1)."""
    if df <= 1:
        return 1.0
    return 1.0 - 3.0 / (4.0 * df - 1.0)


def _hedges_g_raw(x: np.ndarray, y: np.ndarray) -> float:
    d = _cohens_d_raw(x, y)
    df = len(x) + len(y) - 2
    return d * _hedges_correction(df)


def hedges_g(
    x: Sequence[float],
    y: Sequence[float],
    confidence_level: float = 0.95,
) -> EffectSize:
    """Hedges' g — bias-corrected Cohen's d.

    g = d · J(df),  J(df) ≈ 1 − 3/(4·df − 1)

    Removes the small-sample upward bias of Cohen's d.
    """
    a, b = _to_arrays(x, y)
    g = _hedges_g_raw(a, b)
    ci = _bootstrap_ci(a, b, _hedges_g_raw, confidence_level=confidence_level)
    return EffectSize(
        measure=EffectSizeType.HEDGES_G,
        value=g,
        ci=ci,
        interpretation=interpret(g, EffectSizeType.HEDGES_G),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Glass' Δ
# ═══════════════════════════════════════════════════════════════════════════

def _glass_delta_raw(x: np.ndarray, y: np.ndarray) -> float:
    """Glass' Δ = (ȳ − x̄) / s_x  (control group SD)."""
    sd_ctrl = float(np.std(x, ddof=1))
    if sd_ctrl == 0.0:
        return 0.0
    return (float(np.mean(y)) - float(np.mean(x))) / sd_ctrl


def glass_delta(
    x: Sequence[float],
    y: Sequence[float],
    confidence_level: float = 0.95,
) -> EffectSize:
    """Glass' Δ — standardised by the control group SD.

    Δ = (ȳ − x̄) / s_control

    Appropriate when variances are unequal and the control group
    variance is the appropriate standardiser.
    """
    a, b = _to_arrays(x, y)
    delta = _glass_delta_raw(a, b)
    ci = _bootstrap_ci(a, b, _glass_delta_raw, confidence_level=confidence_level)
    return EffectSize(
        measure=EffectSizeType.GLASS_DELTA,
        value=delta,
        ci=ci,
        interpretation=interpret(delta, EffectSizeType.GLASS_DELTA),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Cliff's δ
# ═══════════════════════════════════════════════════════════════════════════

def _cliff_delta_raw(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's δ = (#{Y > X} − #{Y < X}) / (n·m)."""
    n, m = len(x), len(y)
    more = 0
    less = 0
    for yi in y:
        more += int(np.sum(yi > x))
        less += int(np.sum(yi < x))
    return (more - less) / (n * m)


def cliff_delta(
    x: Sequence[float],
    y: Sequence[float],
    confidence_level: float = 0.95,
) -> EffectSize:
    """Cliff's δ — non-parametric ordinal effect size.

    δ = (#{Y > X} − #{Y < X}) / (n · m)

    Ranges from −1 to +1. Does not assume any distribution.
    """
    a, b = _to_arrays(x, y)
    d = _cliff_delta_raw(a, b)
    ci = _bootstrap_ci(a, b, _cliff_delta_raw, confidence_level=confidence_level)
    return EffectSize(
        measure=EffectSizeType.CLIFFS_DELTA,
        value=d,
        ci=ci,
        interpretation=interpret(d, EffectSizeType.CLIFFS_DELTA),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Common-language effect size
# ═══════════════════════════════════════════════════════════════════════════

def _cles_raw(x: np.ndarray, y: np.ndarray) -> float:
    """P(Y > X) + 0.5·P(Y = X)."""
    n, m = len(x), len(y)
    count = 0.0
    for yi in y:
        count += float(np.sum(yi > x)) + 0.5 * float(np.sum(yi == x))
    return count / (n * m)


def common_language_effect_size(
    x: Sequence[float],
    y: Sequence[float],
    confidence_level: float = 0.95,
) -> EffectSize:
    """Common-language effect size: P(Y > X).

    The probability that a randomly chosen observation from Y exceeds
    a randomly chosen observation from X.  0.5 indicates no effect.
    """
    a, b = _to_arrays(x, y)
    cl = _cles_raw(a, b)
    ci = _bootstrap_ci(a, b, _cles_raw, confidence_level=confidence_level)
    return EffectSize(
        measure=EffectSizeType.COMMON_LANGUAGE,
        value=cl,
        ci=ci,
        interpretation=_interpret_cl(cl),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Robust effect size (trimmed means)
# ═══════════════════════════════════════════════════════════════════════════

def _winsorized_var(arr: np.ndarray, trim: float = 0.2) -> float:
    """Winsorized variance for trimmed-mean effect size."""
    n = len(arr)
    k = int(math.floor(trim * n))
    s = np.sort(arr)
    s[:k] = s[k]
    s[n - k:] = s[n - k - 1]
    return float(np.var(s, ddof=1))


def _robust_d_raw(x: np.ndarray, y: np.ndarray, trim: float = 0.2) -> float:
    """Yuen's trimmed-mean effect size."""
    tx = float(sp_stats.trim_mean(x, trim))
    ty = float(sp_stats.trim_mean(y, trim))
    wvx = _winsorized_var(x, trim)
    wvy = _winsorized_var(y, trim)
    n1, n2 = len(x), len(y)
    pooled = math.sqrt(((n1 - 1) * wvx + (n2 - 1) * wvy) / (n1 + n2 - 2))
    if pooled == 0.0:
        return 0.0
    return (ty - tx) / pooled


def robust_effect_size(
    x: Sequence[float],
    y: Sequence[float],
    trim: float = 0.2,
    confidence_level: float = 0.95,
) -> EffectSize:
    """Robust (trimmed-mean) effect size.

    Uses 20% trimmed means and Winsorized variances for outlier
    resistance.  Interpretation thresholds follow Cohen's d conventions.
    """
    a, b = _to_arrays(x, y)
    d = _robust_d_raw(a, b, trim)
    ci = _bootstrap_ci(
        a, b, lambda xx, yy: _robust_d_raw(xx, yy, trim),
        confidence_level=confidence_level,
    )
    return EffectSize(
        measure=EffectSizeType.COHENS_D,
        value=d,
        ci=ci,
        interpretation=interpret(d, EffectSizeType.COHENS_D),
    )


# ═══════════════════════════════════════════════════════════════════════════
# EffectSizeCalculator — implements EffectSizeEstimator protocol
# ═══════════════════════════════════════════════════════════════════════════

class EffectSizeCalculator:
    """Unified effect-size estimator implementing EffectSizeEstimator protocol."""

    _DISPATCH = {
        EffectSizeType.COHENS_D: cohens_d,
        EffectSizeType.HEDGES_G: hedges_g,
        EffectSizeType.GLASS_DELTA: glass_delta,
        EffectSizeType.CLIFFS_DELTA: cliff_delta,
        EffectSizeType.COMMON_LANGUAGE: common_language_effect_size,
    }

    def estimate(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        measure: EffectSizeType = EffectSizeType.COHENS_D,
        confidence_level: float = 0.95,
    ) -> EffectSize:
        fn = self._DISPATCH.get(measure)
        if fn is None:
            raise ValueError(f"Unsupported effect size measure: {measure}")
        return fn(sample_a, sample_b, confidence_level=confidence_level)

    def interpret(self, value: float, measure: EffectSizeType) -> str:
        return interpret(value, measure)

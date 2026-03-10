"""
usability_oracle.statistics.multiple_comparison — Multiple comparison correction.

Controls the family-wise error rate (FWER) or false discovery rate (FDR)
when many usability metrics are tested simultaneously:

- BonferroniCorrection: α_adj = α / m
- HolmBonferroni: step-down Holm procedure
- BenjaminiHochberg: FDR control via BH procedure
- BenjaminiYekutieli: FDR under arbitrary dependence
- StoreyBH: adaptive FDR with π₀ estimation
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from usability_oracle.statistics.types import (
    CorrectionMethod,
    FDRResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_p_values(p_values: Sequence[float]) -> np.ndarray:
    """Validate and convert p-values to an array."""
    arr = np.asarray(p_values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("p_values must be non-empty.")
    if np.any(arr < 0) or np.any(arr > 1):
        raise ValueError("All p-values must be in [0, 1].")
    return arr


# ═══════════════════════════════════════════════════════════════════════════
# BonferroniCorrection
# ═══════════════════════════════════════════════════════════════════════════

class BonferroniCorrection:
    """Bonferroni correction for FWER control.

    Adjusted p-value:  p_adj_i = min(p_i · m, 1)

    Controls the probability of making *any* false rejection
    (family-wise error rate) at level α.
    """

    def correct(
        self,
        p_values: Sequence[float],
        alpha: float = 0.05,
        method: CorrectionMethod = CorrectionMethod.BONFERRONI,
    ) -> FDRResult:
        raw = _validate_p_values(p_values)
        m = len(raw)
        adjusted = np.minimum(raw * m, 1.0)
        rejected = adjusted < alpha
        return FDRResult(
            method=CorrectionMethod.BONFERRONI,
            original_p_values=tuple(raw.tolist()),
            adjusted_p_values=tuple(adjusted.tolist()),
            rejected=tuple(rejected.tolist()),
            alpha=alpha,
            num_tests=m,
            num_rejections=int(rejected.sum()),
        )


# ═══════════════════════════════════════════════════════════════════════════
# HolmBonferroni
# ═══════════════════════════════════════════════════════════════════════════

class HolmBonferroni:
    """Holm–Bonferroni step-down procedure for FWER control.

    Algorithm:
        1. Sort p-values: p_(1) ≤ p_(2) ≤ … ≤ p_(m).
        2. For rank k, adjusted p_(k) = max_{j ≤ k} min(p_(j) · (m − j + 1), 1).
        3. Reject H_(k) if adjusted p_(k) < α.

    More powerful than Bonferroni while still controlling FWER.
    """

    def correct(
        self,
        p_values: Sequence[float],
        alpha: float = 0.05,
        method: CorrectionMethod = CorrectionMethod.HOLM,
    ) -> FDRResult:
        raw = _validate_p_values(p_values)
        m = len(raw)
        order = np.argsort(raw)
        sorted_p = raw[order]

        # Step-down: adjusted p_k = max_{j<=k} min(p_j * (m - j), 1)
        adjusted_sorted = np.empty(m)
        for k in range(m):
            adjusted_sorted[k] = min(sorted_p[k] * (m - k), 1.0)
        # Enforce monotonicity (step-down)
        for k in range(1, m):
            adjusted_sorted[k] = max(adjusted_sorted[k], adjusted_sorted[k - 1])

        # Map back to original order
        adjusted = np.empty(m)
        adjusted[order] = adjusted_sorted
        rejected = adjusted < alpha
        return FDRResult(
            method=CorrectionMethod.HOLM,
            original_p_values=tuple(raw.tolist()),
            adjusted_p_values=tuple(adjusted.tolist()),
            rejected=tuple(rejected.tolist()),
            alpha=alpha,
            num_tests=m,
            num_rejections=int(rejected.sum()),
        )


# ═══════════════════════════════════════════════════════════════════════════
# BenjaminiHochberg
# ═══════════════════════════════════════════════════════════════════════════

class BenjaminiHochberg:
    """Benjamini–Hochberg procedure for FDR control.

    Algorithm:
        1. Sort p-values: p_(1) ≤ … ≤ p_(m).
        2. Adjusted p_(k) = min_{j ≥ k} min(p_(j) · m / j, 1).
        3. Reject H_(k) if adjusted p_(k) < α.

    Controls the expected proportion of false discoveries (FDR) at level α,
    assuming independence or positive regression dependency (PRDS).
    """

    def correct(
        self,
        p_values: Sequence[float],
        alpha: float = 0.05,
        method: CorrectionMethod = CorrectionMethod.BENJAMINI_HOCHBERG,
    ) -> FDRResult:
        raw = _validate_p_values(p_values)
        m = len(raw)
        order = np.argsort(raw)
        sorted_p = raw[order]

        # Step-up: adjusted p_(k) = min_{j>=k}(p_(j) * m / j)
        adjusted_sorted = np.empty(m)
        for k in range(m):
            adjusted_sorted[k] = min(sorted_p[k] * m / (k + 1), 1.0)
        # Enforce monotonicity (step-up: go from end to start)
        for k in range(m - 2, -1, -1):
            adjusted_sorted[k] = min(adjusted_sorted[k], adjusted_sorted[k + 1])

        adjusted = np.empty(m)
        adjusted[order] = adjusted_sorted
        rejected = adjusted < alpha
        return FDRResult(
            method=CorrectionMethod.BENJAMINI_HOCHBERG,
            original_p_values=tuple(raw.tolist()),
            adjusted_p_values=tuple(adjusted.tolist()),
            rejected=tuple(rejected.tolist()),
            alpha=alpha,
            num_tests=m,
            num_rejections=int(rejected.sum()),
        )


# ═══════════════════════════════════════════════════════════════════════════
# BenjaminiYekutieli
# ═══════════════════════════════════════════════════════════════════════════

class BenjaminiYekutieli:
    """Benjamini–Yekutieli procedure for FDR under arbitrary dependence.

    Like BH but with an additional correction factor c(m) = Σ_{k=1}^{m} 1/k
    to handle arbitrary dependence between tests.

    Adjusted p_(k) = min_{j ≥ k} min(p_(j) · m · c(m) / j, 1).
    """

    def correct(
        self,
        p_values: Sequence[float],
        alpha: float = 0.05,
        method: CorrectionMethod = CorrectionMethod.BENJAMINI_YEKUTIELI,
    ) -> FDRResult:
        raw = _validate_p_values(p_values)
        m = len(raw)
        # Harmonic number c(m) = Σ 1/k
        c_m = sum(1.0 / k for k in range(1, m + 1))

        order = np.argsort(raw)
        sorted_p = raw[order]

        adjusted_sorted = np.empty(m)
        for k in range(m):
            adjusted_sorted[k] = min(sorted_p[k] * m * c_m / (k + 1), 1.0)
        for k in range(m - 2, -1, -1):
            adjusted_sorted[k] = min(adjusted_sorted[k], adjusted_sorted[k + 1])

        adjusted = np.empty(m)
        adjusted[order] = adjusted_sorted
        rejected = adjusted < alpha
        return FDRResult(
            method=CorrectionMethod.BENJAMINI_YEKUTIELI,
            original_p_values=tuple(raw.tolist()),
            adjusted_p_values=tuple(adjusted.tolist()),
            rejected=tuple(rejected.tolist()),
            alpha=alpha,
            num_tests=m,
            num_rejections=int(rejected.sum()),
        )


# ═══════════════════════════════════════════════════════════════════════════
# StoreyBH
# ═══════════════════════════════════════════════════════════════════════════

class StoreyBH:
    """Storey's adaptive BH procedure with π₀ estimation.

    Estimates the proportion of true nulls π₀ to sharpen FDR control.

    π₀ estimation uses the bootstrap method from Storey (2002):
        π̂₀(λ) = #{p_i > λ} / (m · (1 − λ))

    Then applies BH with effective FDR level α / π̂₀.
    """

    def __init__(self, lambda_val: float = 0.5) -> None:
        """
        Parameters:
            lambda_val: Tuning parameter for π₀ estimation (default 0.5).
        """
        if not (0.0 < lambda_val < 1.0):
            raise ValueError("lambda_val must be in (0, 1).")
        self.lambda_val = lambda_val

    def _estimate_pi0(self, p_values: np.ndarray) -> float:
        """Estimate π₀ using Storey's method.

        π̂₀(λ) = #{p_i > λ} / (m · (1 − λ))
        Clamped to [1/m, 1].
        """
        m = len(p_values)
        w = float(np.sum(p_values > self.lambda_val))
        pi0 = w / (m * (1.0 - self.lambda_val))
        return max(1.0 / m, min(pi0, 1.0))

    def correct(
        self,
        p_values: Sequence[float],
        alpha: float = 0.05,
        method: CorrectionMethod = CorrectionMethod.BENJAMINI_HOCHBERG,
    ) -> FDRResult:
        raw = _validate_p_values(p_values)
        m = len(raw)
        pi0 = self._estimate_pi0(raw)

        order = np.argsort(raw)
        sorted_p = raw[order]

        # BH adjusted with π₀ scaling
        adjusted_sorted = np.empty(m)
        for k in range(m):
            adjusted_sorted[k] = min(sorted_p[k] * m * pi0 / (k + 1), 1.0)
        for k in range(m - 2, -1, -1):
            adjusted_sorted[k] = min(adjusted_sorted[k], adjusted_sorted[k + 1])

        adjusted = np.empty(m)
        adjusted[order] = adjusted_sorted
        rejected = adjusted < alpha
        return FDRResult(
            method=CorrectionMethod.BENJAMINI_HOCHBERG,
            original_p_values=tuple(raw.tolist()),
            adjusted_p_values=tuple(adjusted.tolist()),
            rejected=tuple(rejected.tolist()),
            alpha=alpha,
            num_tests=m,
            num_rejections=int(rejected.sum()),
        )


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

def correct(
    p_values: Sequence[float],
    method: CorrectionMethod = CorrectionMethod.BENJAMINI_HOCHBERG,
    alpha: float = 0.05,
) -> FDRResult:
    """Apply multiple-comparison correction.

    Parameters:
        p_values: Raw p-values.
        method: Correction method.
        alpha: Target significance level.

    Returns:
        FDRResult with adjusted p-values and rejection decisions.
    """
    dispatch = {
        CorrectionMethod.BONFERRONI: BonferroniCorrection(),
        CorrectionMethod.HOLM: HolmBonferroni(),
        CorrectionMethod.BENJAMINI_HOCHBERG: BenjaminiHochberg(),
        CorrectionMethod.BENJAMINI_YEKUTIELI: BenjaminiYekutieli(),
        CorrectionMethod.NONE: _NoneCorrection(),
    }
    corrector = dispatch.get(method)
    if corrector is None:
        raise ValueError(f"Unknown correction method: {method}")
    return corrector.correct(p_values, alpha=alpha, method=method)


class _NoneCorrection:
    """No correction — pass through raw p-values."""

    def correct(
        self,
        p_values: Sequence[float],
        alpha: float = 0.05,
        method: CorrectionMethod = CorrectionMethod.NONE,
    ) -> FDRResult:
        raw = _validate_p_values(p_values)
        rejected = raw < alpha
        return FDRResult(
            method=CorrectionMethod.NONE,
            original_p_values=tuple(raw.tolist()),
            adjusted_p_values=tuple(raw.tolist()),
            rejected=tuple(rejected.tolist()),
            alpha=alpha,
            num_tests=len(raw),
            num_rejections=int(rejected.sum()),
        )

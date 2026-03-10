"""
usability_oracle.statistics.protocols — Structural interfaces for
regression detection statistics.

Defines protocols for hypothesis testing, multiple-comparison correction,
and effect-size estimation used to determine whether observed changes in
usability metrics constitute true regressions.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usability_oracle.statistics.types import (
        AlternativeHypothesis,
        BootstrapResult,
        ConfidenceInterval,
        CorrectionMethod,
        EffectSize,
        EffectSizeType,
        FDRResult,
        HypothesisTestResult,
        PowerAnalysisResult,
        TestType,
    )


# ═══════════════════════════════════════════════════════════════════════════
# StatisticalTest
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class StatisticalTest(Protocol):
    """Execute a two-sample hypothesis test for usability regression.

    Given samples of trajectory costs from a baseline (A) and candidate
    (B) UI version, tests whether the cost distribution has shifted
    upward (regression).
    """

    def test(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        alpha: float = 0.05,
        alternative: AlternativeHypothesis = ...,  # type: ignore[assignment]
    ) -> HypothesisTestResult:
        """Run the hypothesis test.

        Parameters:
            sample_a: Cost observations from the baseline UI.
            sample_b: Cost observations from the candidate UI.
            alpha: Significance level.
            alternative: Direction of the alternative hypothesis.

        Returns:
            A :class:`HypothesisTestResult` with test statistic,
            p-value, effect size, and confidence interval.

        Raises:
            InsufficientDataError: If either sample is too small.
        """
        ...

    def power_analysis(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> PowerAnalysisResult:
        """Compute the required sample size for a given power.

        Parameters:
            effect_size: Minimum effect size to detect (Cohen's d).
            alpha: Significance level.
            power: Target statistical power (1 − β).

        Returns:
            A :class:`PowerAnalysisResult` with the minimum sample size.
        """
        ...

    def bootstrap_test(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        num_resamples: int = 10_000,
        confidence_level: float = 0.95,
        seed: Optional[int] = None,
    ) -> BootstrapResult:
        """Run a bootstrap hypothesis test on the mean difference.

        Parameters:
            sample_a: Baseline cost observations.
            sample_b: Candidate cost observations.
            num_resamples: Number of bootstrap resamples.
            confidence_level: Confidence level for the CI.
            seed: Random seed for reproducibility.

        Returns:
            A :class:`BootstrapResult` for the mean-difference statistic.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# MultipleComparisonCorrector
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class MultipleComparisonCorrector(Protocol):
    """Correct p-values for multiple simultaneous hypothesis tests.

    When many usability metrics are tested at once (per-task, per-element,
    per-cognitive-dimension), the family-wise error rate or false discovery
    rate must be controlled.
    """

    def correct(
        self,
        p_values: Sequence[float],
        alpha: float = 0.05,
        method: CorrectionMethod = ...,  # type: ignore[assignment]
    ) -> FDRResult:
        """Apply multiple-comparison correction.

        Parameters:
            p_values: Raw (uncorrected) p-values.
            alpha: Target FDR or FWER level.
            method: Correction procedure.

        Returns:
            A :class:`FDRResult` with adjusted p-values and rejection
            decisions.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# EffectSizeEstimator
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class EffectSizeEstimator(Protocol):
    """Compute standardised effect-size measures between two samples.

    Effect sizes quantify the *practical significance* of a usability
    change, complementing the *statistical significance* from the
    hypothesis test.
    """

    def estimate(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        measure: EffectSizeType = ...,  # type: ignore[assignment]
        confidence_level: float = 0.95,
    ) -> EffectSize:
        """Compute an effect-size estimate with confidence interval.

        Parameters:
            sample_a: Baseline cost observations.
            sample_b: Candidate cost observations.
            measure: Which effect-size metric to compute.
            confidence_level: Confidence level for the CI.

        Returns:
            An :class:`EffectSize` with point estimate, CI, and
            qualitative interpretation.
        """
        ...

    def interpret(
        self,
        value: float,
        measure: EffectSizeType,
    ) -> str:
        """Return a qualitative interpretation of an effect size.

        Uses conventional thresholds (e.g. Cohen's d: 0.2 = small,
        0.5 = medium, 0.8 = large).

        Parameters:
            value: Numerical effect-size value.
            measure: Which metric the value corresponds to.

        Returns:
            One of ``"negligible"``, ``"small"``, ``"medium"``,
            ``"large"``.
        """
        ...

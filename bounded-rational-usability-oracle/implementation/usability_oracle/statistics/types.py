"""
usability_oracle.statistics.types — Data types for regression detection statistics.

Provides immutable value types for hypothesis testing, confidence intervals,
effect sizes, power analysis, multiple-comparison correction, and bootstrap
resampling.  These types support the statistical comparison framework that
determines whether a usability metric change constitutes a true regression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, NewType, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

@unique
class TestType(Enum):
    """Type of statistical hypothesis test."""

    WELCH_T = "welch_t"
    """Welch's t-test (unequal variances)."""

    MANN_WHITNEY_U = "mann_whitney_u"
    """Mann–Whitney U test (non-parametric)."""

    PERMUTATION = "permutation"
    """Permutation (randomisation) test."""

    BOOTSTRAP = "bootstrap"
    """Bootstrap hypothesis test."""

    BAYESIAN = "bayesian"
    """Bayesian test with posterior credible intervals."""

    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    """Two-sample Kolmogorov–Smirnov test."""


@unique
class AlternativeHypothesis(Enum):
    """Direction of the alternative hypothesis."""

    TWO_SIDED = "two_sided"
    """H₁: μ_A ≠ μ_B."""

    GREATER = "greater"
    """H₁: μ_A > μ_B (regression — cost increased)."""

    LESS = "less"
    """H₁: μ_A < μ_B (improvement — cost decreased)."""


@unique
class CorrectionMethod(Enum):
    """Multiple-comparison p-value correction method."""

    BONFERRONI = "bonferroni"
    """Bonferroni correction: α_adj = α / m."""

    HOLM = "holm"
    """Holm–Bonferroni step-down procedure."""

    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    """Benjamini–Hochberg FDR control."""

    BENJAMINI_YEKUTIELI = "benjamini_yekutieli"
    """Benjamini–Yekutieli FDR under arbitrary dependence."""

    NONE = "none"
    """No correction applied."""


@unique
class EffectSizeType(Enum):
    """Which effect-size measure to compute."""

    COHENS_D = "cohens_d"
    """Cohen's d = (μ₁ − μ₂) / s_pooled."""

    HEDGES_G = "hedges_g"
    """Hedges' g — bias-corrected Cohen's d."""

    GLASS_DELTA = "glass_delta"
    """Glass' Δ = (μ₁ − μ₂) / s_control."""

    CLIFFS_DELTA = "cliffs_delta"
    """Cliff's δ (non-parametric ordinal effect size)."""

    COMMON_LANGUAGE = "common_language"
    """Common-language effect size P(X > Y)."""


# ═══════════════════════════════════════════════════════════════════════════
# ConfidenceInterval
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A confidence or credible interval for a scalar estimand.

    Attributes:
        lower: Lower endpoint.
        upper: Upper endpoint.
        level: Confidence level, e.g. 0.95 for a 95 % CI.
        point_estimate: Point estimate (e.g. sample mean).
        method: Description of how the interval was computed
            (e.g. ``"bootstrap percentile"``, ``"Wald"``).
    """

    lower: float
    upper: float
    level: float
    point_estimate: float
    method: str = "unknown"

    @property
    def width(self) -> float:
        """Width of the interval (upper − lower)."""
        return self.upper - self.lower

    @property
    def margin_of_error(self) -> float:
        """Half-width of the interval."""
        return self.width / 2.0

    @property
    def contains_zero(self) -> bool:
        """Whether the interval straddles zero (no significant effect)."""
        return self.lower <= 0.0 <= self.upper

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "level": self.level,
            "point_estimate": self.point_estimate,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ConfidenceInterval:
        return cls(
            lower=float(d["lower"]),
            upper=float(d["upper"]),
            level=float(d["level"]),
            point_estimate=float(d["point_estimate"]),
            method=str(d.get("method", "unknown")),
        )


# ═══════════════════════════════════════════════════════════════════════════
# EffectSize
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class EffectSize:
    """A standardised effect-size estimate with confidence interval.

    Attributes:
        measure: Which effect-size metric was used.
        value: Point estimate of the effect size.
        ci: Confidence interval for the effect size.
        interpretation: Qualitative interpretation per conventional
            thresholds (``"negligible"``, ``"small"``, ``"medium"``,
            ``"large"``).
    """

    measure: EffectSizeType
    value: float
    ci: ConfidenceInterval
    interpretation: str

    @property
    def is_negligible(self) -> bool:
        """Whether the effect is negligible (|d| < 0.2 or equivalent)."""
        return self.interpretation == "negligible"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measure": self.measure.value,
            "value": self.value,
            "ci": self.ci.to_dict(),
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EffectSize:
        return cls(
            measure=EffectSizeType(d["measure"]),
            value=float(d["value"]),
            ci=ConfidenceInterval.from_dict(d["ci"]),
            interpretation=str(d["interpretation"]),
        )


# ═══════════════════════════════════════════════════════════════════════════
# HypothesisTestResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class HypothesisTestResult:
    """Result of a single hypothesis test for usability regression.

    Tests whether the expected cognitive cost increased between two
    UI versions:  H₀: μ_B − μ_A ≤ 0  vs  H₁: μ_B − μ_A > 0.

    Attributes:
        test_type: Which statistical test was used.
        statistic: Test statistic value (t, U, etc.).
        p_value: p-value under H₀.
        alternative: Direction of the alternative hypothesis.
        alpha: Significance level.
        reject_null: Whether H₀ is rejected at level α.
        effect_size: Standardised effect-size estimate.
        ci: Confidence interval for the mean difference.
        sample_size_a: Number of observations in group A (baseline).
        sample_size_b: Number of observations in group B (candidate).
        degrees_of_freedom: Degrees of freedom (where applicable).
    """

    test_type: TestType
    statistic: float
    p_value: float
    alternative: AlternativeHypothesis
    alpha: float
    reject_null: bool
    effect_size: EffectSize
    ci: ConfidenceInterval
    sample_size_a: int
    sample_size_b: int
    degrees_of_freedom: Optional[float] = None

    @property
    def is_regression(self) -> bool:
        """Whether the test indicates a statistically significant regression."""
        return self.reject_null and self.alternative != AlternativeHypothesis.LESS

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "test_type": self.test_type.value,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "alternative": self.alternative.value,
            "alpha": self.alpha,
            "reject_null": self.reject_null,
            "effect_size": self.effect_size.to_dict(),
            "ci": self.ci.to_dict(),
            "sample_size_a": self.sample_size_a,
            "sample_size_b": self.sample_size_b,
        }
        if self.degrees_of_freedom is not None:
            d["degrees_of_freedom"] = self.degrees_of_freedom
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> HypothesisTestResult:
        return cls(
            test_type=TestType(d["test_type"]),
            statistic=float(d["statistic"]),
            p_value=float(d["p_value"]),
            alternative=AlternativeHypothesis(d["alternative"]),
            alpha=float(d["alpha"]),
            reject_null=bool(d["reject_null"]),
            effect_size=EffectSize.from_dict(d["effect_size"]),
            ci=ConfidenceInterval.from_dict(d["ci"]),
            sample_size_a=int(d["sample_size_a"]),
            sample_size_b=int(d["sample_size_b"]),
            degrees_of_freedom=d.get("degrees_of_freedom"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# PowerAnalysisResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PowerAnalysisResult:
    """Result of a statistical power analysis.

    Computes the minimum sample size (number of Monte Carlo trajectories)
    needed to detect a regression of a given effect size with desired
    power (1 − β_err).

    Attributes:
        target_effect_size: Minimum effect size to detect (Cohen's d).
        alpha: Significance level.
        power: Target statistical power (1 − β_err).
        required_sample_size: Minimum n per group to achieve the target.
        actual_power: Achieved power at the required sample size
            (may slightly exceed *power* due to discretisation).
        test_type: Assumed test type for the power calculation.
    """

    target_effect_size: float
    alpha: float
    power: float
    required_sample_size: int
    actual_power: float
    test_type: TestType

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_effect_size": self.target_effect_size,
            "alpha": self.alpha,
            "power": self.power,
            "required_sample_size": self.required_sample_size,
            "actual_power": self.actual_power,
            "test_type": self.test_type.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PowerAnalysisResult:
        return cls(
            target_effect_size=float(d["target_effect_size"]),
            alpha=float(d["alpha"]),
            power=float(d["power"]),
            required_sample_size=int(d["required_sample_size"]),
            actual_power=float(d["actual_power"]),
            test_type=TestType(d["test_type"]),
        )


# ═══════════════════════════════════════════════════════════════════════════
# FDRResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class FDRResult:
    """Result of multiple-comparison correction across usability metrics.

    When testing many metrics simultaneously (e.g. per-task, per-element),
    raw p-values must be corrected to control the false discovery rate.

    Attributes:
        method: Correction method applied.
        original_p_values: Raw (uncorrected) p-values, one per test.
        adjusted_p_values: Corrected p-values.
        rejected: Boolean mask — ``True`` where H₀ is rejected after
            correction.
        alpha: Family-wise or FDR-controlling level.
        num_tests: Total number of simultaneous tests.
        num_rejections: Number of tests that remain significant.
    """

    method: CorrectionMethod
    original_p_values: Tuple[float, ...]
    adjusted_p_values: Tuple[float, ...]
    rejected: Tuple[bool, ...]
    alpha: float
    num_tests: int
    num_rejections: int

    @property
    def rejection_rate(self) -> float:
        """Fraction of tests rejected after correction."""
        return self.num_rejections / self.num_tests if self.num_tests > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "original_p_values": list(self.original_p_values),
            "adjusted_p_values": list(self.adjusted_p_values),
            "rejected": list(self.rejected),
            "alpha": self.alpha,
            "num_tests": self.num_tests,
            "num_rejections": self.num_rejections,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FDRResult:
        return cls(
            method=CorrectionMethod(d["method"]),
            original_p_values=tuple(d["original_p_values"]),
            adjusted_p_values=tuple(d["adjusted_p_values"]),
            rejected=tuple(d["rejected"]),
            alpha=float(d["alpha"]),
            num_tests=int(d["num_tests"]),
            num_rejections=int(d["num_rejections"]),
        )


# ═══════════════════════════════════════════════════════════════════════════
# BootstrapResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Result of a bootstrap resampling procedure.

    Attributes:
        statistic_name: Name of the statistic being bootstrapped
            (e.g. ``"mean_cost_difference"``).
        observed_statistic: Value of the statistic on the original data.
        bootstrap_distribution: Sorted array of bootstrap replicates.
        ci: Bootstrap confidence interval (percentile or BCa method).
        bias: Estimated bias  E[θ̂*] − θ̂.
        standard_error: Bootstrap standard error.
        num_resamples: Number of bootstrap resamples drawn.
        seed: Random seed used for reproducibility.
    """

    statistic_name: str
    observed_statistic: float
    bootstrap_distribution: Tuple[float, ...]
    ci: ConfidenceInterval
    bias: float
    standard_error: float
    num_resamples: int
    seed: Optional[int] = None

    @property
    def bias_corrected_estimate(self) -> float:
        """Bias-corrected point estimate  θ̂ − bias."""
        return self.observed_statistic - self.bias

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statistic_name": self.statistic_name,
            "observed_statistic": self.observed_statistic,
            "bootstrap_distribution": list(self.bootstrap_distribution),
            "ci": self.ci.to_dict(),
            "bias": self.bias,
            "standard_error": self.standard_error,
            "num_resamples": self.num_resamples,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BootstrapResult:
        return cls(
            statistic_name=str(d["statistic_name"]),
            observed_statistic=float(d["observed_statistic"]),
            bootstrap_distribution=tuple(d["bootstrap_distribution"]),
            ci=ConfidenceInterval.from_dict(d["ci"]),
            bias=float(d["bias"]),
            standard_error=float(d["standard_error"]),
            num_resamples=int(d["num_resamples"]),
            seed=d.get("seed"),
        )

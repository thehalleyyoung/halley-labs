"""
usability_oracle.statistics — Regression detection statistics.

Provides hypothesis testing, effect-size estimation, multiple-comparison
correction, and bootstrap resampling for determining whether observed
changes in usability metrics constitute true regressions.

::

    from usability_oracle.statistics import HypothesisTestResult, StatisticalTest
"""

from __future__ import annotations

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

from usability_oracle.statistics.protocols import (
    EffectSizeEstimator,
    MultipleComparisonCorrector,
    StatisticalTest,
)

from usability_oracle.statistics.hypothesis_tests import (
    BootstrapTest,
    BrunnerMunzel,
    PairedTTest,
    PermutationTest,
    WilcoxonSignedRank,
)

from usability_oracle.statistics.multiple_comparison import (
    BenjaminiHochberg,
    BenjaminiYekutieli,
    BonferroniCorrection,
    HolmBonferroni,
    StoreyBH,
)

from usability_oracle.statistics.effect_size import (
    EffectSizeCalculator,
    cliff_delta,
    cohens_d,
    common_language_effect_size,
    glass_delta,
    hedges_g,
    robust_effect_size,
)

from usability_oracle.statistics.power_analysis import (
    compute_power,
    minimum_detectable_effect,
    power_curve,
    required_sample_size,
    sequential_power,
)

from usability_oracle.statistics.bootstrap import BootstrapCI

__all__ = [
    # types
    "AlternativeHypothesis",
    "BootstrapResult",
    "ConfidenceInterval",
    "CorrectionMethod",
    "EffectSize",
    "EffectSizeType",
    "FDRResult",
    "HypothesisTestResult",
    "PowerAnalysisResult",
    "TestType",
    # protocols
    "EffectSizeEstimator",
    "MultipleComparisonCorrector",
    "StatisticalTest",
    # hypothesis tests
    "PairedTTest",
    "WilcoxonSignedRank",
    "PermutationTest",
    "BrunnerMunzel",
    "BootstrapTest",
    # multiple comparison
    "BonferroniCorrection",
    "HolmBonferroni",
    "BenjaminiHochberg",
    "BenjaminiYekutieli",
    "StoreyBH",
    # effect size
    "EffectSizeCalculator",
    "cohens_d",
    "hedges_g",
    "glass_delta",
    "cliff_delta",
    "common_language_effect_size",
    "robust_effect_size",
    # power analysis
    "compute_power",
    "required_sample_size",
    "minimum_detectable_effect",
    "power_curve",
    "sequential_power",
    # bootstrap
    "BootstrapCI",
]

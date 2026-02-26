"""
MARACE Race Detection Core.

Provides interaction race definitions, epsilon-race formulation with
iterative calibration, race catalog construction, and race classification.
"""

from marace.race.definition import (
    InteractionRace,
    RaceCondition,
    HBInconsistency,
    RaceWitness,
    RaceAbsence,
    RaceClassification,
)
from marace.race.epsilon_race import (
    EpsilonRace,
    EpsilonCalibrator,
    FalsePositiveEstimator,
    EpsilonSensitivityAnalysis,
    MonotoneConvergenceProof,
)
from marace.race.catalog import (
    RaceCatalog,
    CatalogEntry,
    CatalogBuilder,
    CatalogFilter,
    CatalogMerger,
    CatalogStatistics,
    CatalogExporter,
)
from marace.race.classifier import (
    RaceClassifier,
    SeverityScorer,
    PatternMatcher,
    RootCauseAnalyzer,
    RemediationSuggester,
)
from marace.race.calibration_convergence import (
    BanachFixedPointTheorem,
    MonotonicityProof,
    ContractionCondition,
    FixedPointCertificate,
    AdaptiveCalibration,
    CalibrationSoundness,
)
from marace.race.false_positive_analysis import (
    FalsePositiveModel,
    ExperimentalFPMeasurement,
    TightnessImpactReport,
    MitigationStrategies,
    ArchitecturalSensitivity,
)

__all__ = [
    "InteractionRace",
    "RaceCondition",
    "HBInconsistency",
    "RaceWitness",
    "RaceAbsence",
    "RaceClassification",
    "EpsilonRace",
    "EpsilonCalibrator",
    "FalsePositiveEstimator",
    "EpsilonSensitivityAnalysis",
    "MonotoneConvergenceProof",
    "RaceCatalog",
    "CatalogEntry",
    "CatalogBuilder",
    "CatalogFilter",
    "CatalogMerger",
    "CatalogStatistics",
    "CatalogExporter",
    "RaceClassifier",
    "SeverityScorer",
    "PatternMatcher",
    "RootCauseAnalyzer",
    "RemediationSuggester",
    "BanachFixedPointTheorem",
    "MonotonicityProof",
    "ContractionCondition",
    "FixedPointCertificate",
    "AdaptiveCalibration",
    "CalibrationSoundness",
    "FalsePositiveModel",
    "ExperimentalFPMeasurement",
    "TightnessImpactReport",
    "MitigationStrategies",
    "ArchitecturalSensitivity",
]

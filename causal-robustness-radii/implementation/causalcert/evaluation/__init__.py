"""
Evaluation sub-package — benchmarks, DGPs, ablations, and scalability.

Provides synthetic data-generating processes with known robustness radii,
semi-synthetic benchmarks, a library of published causal DAGs, and
evaluation runners for ablation studies and scalability profiling.
"""

from causalcert.evaluation.dgp import SyntheticDGP
from causalcert.evaluation.semi_synthetic import SemiSyntheticBenchmark
from causalcert.evaluation.published_dags import get_published_dag, list_published_dags
from causalcert.evaluation.scalability import ScalabilityProfiler
from causalcert.evaluation.ablation import AblationHarness
from causalcert.evaluation.metrics import coverage_rate, interval_width, fragility_auc
from causalcert.evaluation.runner import EvaluationRunner
from causalcert.evaluation.coverage import (
    CoverageResult,
    monte_carlo_coverage,
    ci_coverage_rate,
    radius_accuracy,
    power_analysis,
)
from causalcert.evaluation.visualization_eval import (
    FigureCollection,
    PlotSpec,
    coverage_plot,
    scalability_plot,
    ablation_heatmap,
    forest_plot,
    publication_figure,
)
from causalcert.evaluation.benchmark_suite import (
    BenchmarkConfig,
    BenchmarkSuite,
    MethodSpec,
    SMALL_CONFIG,
    MEDIUM_CONFIG,
    LARGE_CONFIG,
)
from causalcert.evaluation.calibration import (
    CICalibrationResult,
    FragilityCalibrationResult,
    RadiusCalibrationResult,
    CoverageCalibrationResult,
    CalibrationPlotData,
    ci_test_calibration,
    fragility_calibration,
    radius_calibration,
    coverage_calibration,
    calibration_plot_data,
    hosmer_lemeshow_test,
    kolmogorov_smirnov_test,
)

__all__ = [
    "SyntheticDGP",
    "SemiSyntheticBenchmark",
    "get_published_dag",
    "list_published_dags",
    "ScalabilityProfiler",
    "AblationHarness",
    "coverage_rate",
    "interval_width",
    "fragility_auc",
    "EvaluationRunner",
    # coverage
    "CoverageResult",
    "monte_carlo_coverage",
    "ci_coverage_rate",
    "radius_accuracy",
    "power_analysis",
    # visualization_eval
    "FigureCollection",
    "PlotSpec",
    "coverage_plot",
    "scalability_plot",
    "ablation_heatmap",
    "forest_plot",
    "publication_figure",
    # benchmark_suite
    "BenchmarkConfig",
    "BenchmarkSuite",
    "MethodSpec",
    "SMALL_CONFIG",
    "MEDIUM_CONFIG",
    "LARGE_CONFIG",
    # calibration
    "CICalibrationResult",
    "FragilityCalibrationResult",
    "RadiusCalibrationResult",
    "CoverageCalibrationResult",
    "CalibrationPlotData",
    "ci_test_calibration",
    "fragility_calibration",
    "radius_calibration",
    "coverage_calibration",
    "calibration_plot_data",
    "hosmer_lemeshow_test",
    "kolmogorov_smirnov_test",
]

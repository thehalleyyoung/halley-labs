"""
CoaCert Evaluation Harness
==========================

Modules for benchmarking, metrics collection, compression analysis,
correctness validation, differential testing, and report generation
for the CoaCert quotient-compression pipeline.
"""

from .timing import (
    Timer,
    TimingRecord,
    TimingStats,
    MultiRunTimer,
    Phase,
    compute_timing_stats,
    estimate_overhead,
    format_duration,
    timing_table,
)

from .metrics import (
    MetricsCollector,
    PipelineMetrics,
    StateSpaceMetrics,
    MemoryMetrics,
    QueryMetrics,
    WitnessMetrics,
    ThroughputMetrics,
    AggregatedMetrics,
    aggregate_metrics,
    metrics_to_json,
    metrics_to_csv_row,
    write_csv,
    load_metrics_json,
    save_metrics_json,
)

from .compression import (
    CompressionAnalyzer,
    CompressionPoint,
    CompressionQuality,
    SymmetryType,
)

from .correctness import (
    CorrectnessValidator,
    PropertyChecker,
    ReachabilityChecker,
    PropertySpec,
    PropertyKind,
    CheckResult,
    Verdict,
    Discrepancy,
    DiscrepancyKind,
    ValidationReport,
    Mutation,
    MutationOperator,
    AddSpuriousTransition,
    RemoveTransition,
    MergeStates,
    generate_random_properties,
)

from .differential_tester import (
    DifferentialTestEngine,
    LTSSnapshot,
    QuotientMapping,
    DifferentialTestReport,
    CoverageInfo,
    StateDiff,
    TransitionDiff,
    RandomWalkResult,
    BugKind,
)

from .report_generator import (
    ReportGenerator,
    SystemInfo,
    BenchmarkSummary,
)

from .benchmark import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuiteResult,
    BenchmarkStatus,
    PipelinePhases,
    RegressionInfo,
    compare_results,
    detect_regressions,
    regression_report,
    load_suite_result,
    save_suite_result,
)

from .ablation import (
    AblationComponent,
    AblationResult,
    AblationStudyResult,
    AblationRunner,
    AblationStudy,
    ScalabilityDataPoint,
    ScalabilityResult,
    BaselineComparison,
    ComparisonSuiteResult,
)

from .baseline_comparison import (
    LTS,
    LTSState,
    PaigeTarjanBaseline,
    NaiveBisimulation,
    BaselineAlgorithm,
    AlgorithmRun,
    ComparisonReport,
    StatisticalTestResult,
    StatisticalTest,
    BaselineComparisonRunner,
    BaselineComparisonResult,
    generate_latex_table,
    generate_statistical_latex_table,
)

from .scalability import (
    SpecFamily,
    ParameterizedBenchmark,
    ScalabilityDataPoint as ScalabilityDataPointV2,
    FittedModel,
    ScalabilityReport,
    ComplexityFitter,
    ScalabilityRunner,
    generate_dining_philosophers,
    generate_token_ring,
    generate_mutual_exclusion,
    generate_scalability_latex_table,
    generate_two_phase_commit,
    generate_peterson,
    TWO_PHASE_COMMIT_EXPERIMENTS,
    PETERSON_EXPERIMENTS,
    DINING_PHILOSOPHERS_EXPERIMENTS,
    TOKEN_RING_EXPERIMENTS,
    ALL_SCALABILITY_EXPERIMENTS,
)

from .statistical import (
    StatisticalSummary,
    compute_summary,
    EffectSizeResult,
    cohens_d,
    TTestResult,
    welch_t_test,
)

from .run_all_experiments import ExperimentRunner

from .bloom_soundness import (
    BloomSoundnessAnalyzer,
    SoundnessBound,
    SoundnessReport,
    SoundnessExperiment,
    EmpiricalResult,
    AdaptiveBloomConfig,
    annotate_certificate,
    VerificationSoundnessAnalyzer,
    BloomSoundnessCertificate,
    build_soundness_certificate,
    compare_witness_sizes,
)

__all__ = [
    # timing
    "Timer", "TimingRecord", "TimingStats", "MultiRunTimer", "Phase",
    "compute_timing_stats", "estimate_overhead", "format_duration",
    "timing_table",
    # metrics
    "MetricsCollector", "PipelineMetrics", "StateSpaceMetrics",
    "MemoryMetrics", "QueryMetrics", "WitnessMetrics", "ThroughputMetrics",
    "AggregatedMetrics", "aggregate_metrics", "metrics_to_json",
    "metrics_to_csv_row", "write_csv", "load_metrics_json", "save_metrics_json",
    # compression
    "CompressionAnalyzer", "CompressionPoint", "CompressionQuality",
    "SymmetryType",
    # correctness
    "CorrectnessValidator", "PropertyChecker", "ReachabilityChecker",
    "PropertySpec", "PropertyKind", "CheckResult", "Verdict",
    "Discrepancy", "DiscrepancyKind", "ValidationReport",
    "Mutation", "MutationOperator", "AddSpuriousTransition",
    "RemoveTransition", "MergeStates", "generate_random_properties",
    # differential tester
    "DifferentialTestEngine", "LTSSnapshot", "QuotientMapping",
    "DifferentialTestReport", "CoverageInfo", "StateDiff",
    "TransitionDiff", "RandomWalkResult", "BugKind",
    # report generator
    "ReportGenerator", "SystemInfo", "BenchmarkSummary",
    # benchmark runner
    "BenchmarkRunner", "BenchmarkConfig", "BenchmarkResult",
    "BenchmarkSuiteResult", "BenchmarkStatus", "PipelinePhases",
    "RegressionInfo", "compare_results", "detect_regressions",
    "regression_report", "load_suite_result", "save_suite_result",
    # ablation
    "AblationComponent", "AblationResult", "AblationStudyResult",
    "AblationRunner", "AblationStudy",
    "ScalabilityDataPoint", "ScalabilityResult",
    "BaselineComparison", "ComparisonSuiteResult",
    # baseline comparison
    "LTS", "LTSState", "PaigeTarjanBaseline", "NaiveBisimulation",
    "BaselineAlgorithm", "AlgorithmRun", "ComparisonReport",
    "StatisticalTestResult", "StatisticalTest",
    "BaselineComparisonRunner", "BaselineComparisonResult",
    "generate_latex_table", "generate_statistical_latex_table",
    # scalability
    "SpecFamily", "ParameterizedBenchmark",
    "ScalabilityDataPointV2", "FittedModel", "ScalabilityReport",
    "ComplexityFitter", "ScalabilityRunner",
    "generate_dining_philosophers", "generate_token_ring",
    "generate_mutual_exclusion", "generate_scalability_latex_table",
    "generate_two_phase_commit", "generate_peterson",
    "TWO_PHASE_COMMIT_EXPERIMENTS", "PETERSON_EXPERIMENTS",
    "DINING_PHILOSOPHERS_EXPERIMENTS", "TOKEN_RING_EXPERIMENTS",
    "ALL_SCALABILITY_EXPERIMENTS",
    # bloom soundness
    "BloomSoundnessAnalyzer", "SoundnessBound", "SoundnessReport",
    "SoundnessExperiment", "EmpiricalResult", "AdaptiveBloomConfig",
    "annotate_certificate",
    "VerificationSoundnessAnalyzer", "BloomSoundnessCertificate",
    "build_soundness_certificate", "compare_witness_sizes",
    # statistical
    "StatisticalSummary", "compute_summary",
    "EffectSizeResult", "cohens_d",
    "TTestResult", "welch_t_test",
    # experiment runner
    "ExperimentRunner",
]

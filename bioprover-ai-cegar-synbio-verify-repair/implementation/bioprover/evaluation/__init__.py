"""
BioProver evaluation module — benchmarking, baseline comparison,
Cello re-verification, mutation testing, and performance profiling.
"""

from bioprover.evaluation.benchmark_suite import (
    BenchmarkCircuit,
    BenchmarkDifficulty,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkRunner,
)
from bioprover.evaluation.baselines import (
    BaselineResult,
    BaselineTool,
    PRISMBaseline,
    StormBaseline,
    DRealBaseline,
    BreachBaseline,
    BioProverNoAIBaseline,
    BaselineComparison,
)
from bioprover.evaluation.cello_reverification import (
    CelloCircuit,
    CelloGate,
    CelloLibrary,
    CelloReverifier,
    CelloClassification,
    CelloStudyReport,
)
from bioprover.evaluation.mutation_testing import (
    MutationOperator,
    MutantStatus,
    ParameterPerturbation,
    ReactionRemoval,
    SpeciesRemoval,
    KineticLawChange,
    StoichiometryChange,
    PropertyNegation,
    Mutant,
    MutationTestRunner,
    MutationReport,
)
from bioprover.evaluation.profiling import (
    PhaseTimer,
    MemorySnapshot,
    PhaseMetrics,
    ProfilingSession,
    ScalabilityAnalysis,
    RegressionDetector,
    ProfilingReport,
)
from bioprover.evaluation.ablation_experiment import (
    AblationResult,
    AblationReport,
    run_ablation,
    ABLATION_CONFIGS,
)
from bioprover.evaluation.ablation import (
    AblationConfig,
    AblationRunResult,
    AblationRunner,
    AblationSummary,
    generate_ablation_report,
)

__all__ = [
    # Benchmark suite
    "BenchmarkCircuit",
    "BenchmarkDifficulty",
    "BenchmarkResult",
    "BenchmarkSuite",
    "BenchmarkRunner",
    # Baselines
    "BaselineResult",
    "BaselineTool",
    "PRISMBaseline",
    "StormBaseline",
    "DRealBaseline",
    "BreachBaseline",
    "BioProverNoAIBaseline",
    "BaselineComparison",
    # Cello re-verification
    "CelloCircuit",
    "CelloGate",
    "CelloLibrary",
    "CelloReverifier",
    "CelloClassification",
    "CelloStudyReport",
    # Mutation testing
    "MutationOperator",
    "MutantStatus",
    "ParameterPerturbation",
    "ReactionRemoval",
    "SpeciesRemoval",
    "KineticLawChange",
    "StoichiometryChange",
    "PropertyNegation",
    "Mutant",
    "MutationTestRunner",
    "MutationReport",
    # Profiling
    "PhaseTimer",
    "MemorySnapshot",
    "PhaseMetrics",
    "ProfilingSession",
    "ScalabilityAnalysis",
    "RegressionDetector",
    "ProfilingReport",
    # Ablation experiment
    "AblationResult",
    "AblationReport",
    "run_ablation",
    "ABLATION_CONFIGS",
    # Ablation (new)
    "AblationConfig",
    "AblationRunResult",
    "AblationRunner",
    "AblationSummary",
    "generate_ablation_report",
]

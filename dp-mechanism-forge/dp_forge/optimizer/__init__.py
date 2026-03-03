"""
Advanced optimization backend for differential privacy mechanism synthesis.

This package provides state-of-the-art LP/convex optimization algorithms
optimized for the structure of DP mechanism synthesis problems. All
implementations use scipy.optimize.linprog(method='highs') as the core LP
solver, with custom preprocessing and structural exploitation for performance.

Modules:
    backend: Unified solver backend interface with auto-selection
    structure: Toeplitz/circulant matrix structure exploitation via FFT
    cutting_plane: Cutting plane method with HiGHS inner solver
    warm_start: CEGIS warm-start strategies preserving dual bases
    column_generation: Column generation for infinite output domains
    
Key Design Principles:
    - HiGHS via scipy/highspy is the backend LP solver (not custom IPM)
    - Structure exploitation via FFT-based LinearOperators
    - All solvers support warm-starting and timeout limits
    - Numerical tolerances are tracked explicitly throughout
    - Integration with dp_forge.verifier for CEGIS loop
"""

from dp_forge.optimizer.backend import (
    BackendSelector,
    CVXPYBackend,
    HiGHSBackend,
    OptimizationBackend,
    OptimizationResult,
    SolverConfig,
    SolverStatus,
)
from dp_forge.optimizer.column_generation import (
    Column,
    ColumnGenerationEngine,
    ColumnGenerationResult,
    DomainDiscretizer,
    PricingOracle,
    StabilizedColumnGeneration,
)
from dp_forge.optimizer.cutting_plane import (
    AnalyticCenter,
    BundleMethod,
    Cut,
    CuttingPlaneEngine,
    CuttingPlaneResult,
    SeparationOracle,
    VerifierSeparationOracle,
)
from dp_forge.optimizer.structure import (
    BandedStructureDetector,
    CirculantPreconditioner,
    SymmetryReducer,
    ToeplitzOperator,
    ToeplitzStructure,
    build_constraint_graph,
    detect_constraint_structure,
    estimate_condition_number,
    extract_constraint_blocks,
    optimize_constraint_ordering,
)
from dp_forge.optimizer.warm_start import (
    AdaptiveTolerance,
    BasisInfo,
    BasisTracker,
    CEGISWarmStartManager,
    ConstraintImportanceRanker,
    ConstraintInfo,
    ConstraintPoolManager,
    DualSimplexWarmStart,
    IncrementalUpdate,
)

__all__ = [
    # Backend interface
    "OptimizationBackend",
    "HiGHSBackend",
    "CVXPYBackend",
    "BackendSelector",
    "SolverConfig",
    "SolverStatus",
    "OptimizationResult",
    # Structure exploitation
    "ToeplitzOperator",
    "ToeplitzStructure",
    "CirculantPreconditioner",
    "SymmetryReducer",
    "BandedStructureDetector",
    "detect_constraint_structure",
    "optimize_constraint_ordering",
    "extract_constraint_blocks",
    "build_constraint_graph",
    "estimate_condition_number",
    # Cutting plane
    "CuttingPlaneEngine",
    "SeparationOracle",
    "VerifierSeparationOracle",
    "AnalyticCenter",
    "BundleMethod",
    "Cut",
    "CuttingPlaneResult",
    # Warm start
    "DualSimplexWarmStart",
    "ConstraintPoolManager",
    "BasisTracker",
    "IncrementalUpdate",
    "BasisInfo",
    "ConstraintInfo",
    "CEGISWarmStartManager",
    "AdaptiveTolerance",
    "ConstraintImportanceRanker",
    # Column generation
    "ColumnGenerationEngine",
    "PricingOracle",
    "DomainDiscretizer",
    "StabilizedColumnGeneration",
    "Column",
    "ColumnGenerationResult",
]

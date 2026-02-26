"""
BioProver Infrastructure Module.

Cross-cutting infrastructure: error handling, configuration, serialization,
parallel execution, logging, and memory management for the BioProver system.
"""

from bioprover.infrastructure.errors import (
    BioProverError,
    ModelError,
    ParseError,
    ValidationError,
    SolverError,
    TimeoutError,
    NumericalError,
    EncodingError,
    SMTError,
    VerificationError,
    RefinementError,
    RepairError,
    SynthesisError,
    CompositionError,
    ErrorFormatter,
    error_context,
)
from bioprover.infrastructure.config import (
    BioProverConfig,
    SolverConfig,
    CEGARConfig,
    RepairConfig,
    TemporalConfig,
    AIConfig,
    quick_check_config,
    standard_config,
    thorough_config,
    load_config,
)
from bioprover.infrastructure.serialization import (
    Serializer,
    CheckpointManager,
    StateSnapshot,
    VersionedFormat,
    IncrementalCheckpointer,
)
from bioprover.infrastructure.parallel import (
    TaskExecutor,
    PortfolioRunner,
    DeterministicExecutor,
    WorkStealingPool,
    TaskResult,
)
from bioprover.infrastructure.logging_config import (
    setup_logging,
    VerificationAuditLog,
    PerformanceTracer,
    CEGARProgressLog,
    get_logger,
)
from bioprover.infrastructure.arena import (
    Arena,
    ObjectPool,
    RefCounted,
    MemoryTracker,
)

__all__ = [
    # Errors
    "BioProverError",
    "ModelError",
    "ParseError",
    "ValidationError",
    "SolverError",
    "TimeoutError",
    "NumericalError",
    "EncodingError",
    "SMTError",
    "VerificationError",
    "RefinementError",
    "RepairError",
    "SynthesisError",
    "CompositionError",
    "ErrorFormatter",
    "error_context",
    # Config
    "BioProverConfig",
    "SolverConfig",
    "CEGARConfig",
    "RepairConfig",
    "TemporalConfig",
    "AIConfig",
    "quick_check_config",
    "standard_config",
    "thorough_config",
    "load_config",
    # Serialization
    "Serializer",
    "CheckpointManager",
    "StateSnapshot",
    "VersionedFormat",
    "IncrementalCheckpointer",
    # Parallel
    "TaskExecutor",
    "PortfolioRunner",
    "DeterministicExecutor",
    "WorkStealingPool",
    "TaskResult",
    # Logging
    "setup_logging",
    "VerificationAuditLog",
    "PerformanceTracer",
    "CEGARProgressLog",
    "get_logger",
    # Arena
    "Arena",
    "ObjectPool",
    "RefCounted",
    "MemoryTracker",
]

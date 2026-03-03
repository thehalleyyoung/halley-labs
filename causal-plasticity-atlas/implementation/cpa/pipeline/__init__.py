"""CPA Pipeline — Three-phase orchestrated causal-plasticity analysis.

Provides the main pipeline orchestrator that coordinates causal discovery,
alignment, descriptor computation, quality-diversity exploration,
tipping-point detection, and robustness certification into a single
end-to-end workflow.

Modules
-------
orchestrator
    CPAOrchestrator: three-phase pipeline engine.
config
    PipelineConfig and sub-configurations.
checkpointing
    CheckpointManager for save/resume.
results
    AtlasResult and phase-specific result containers.
"""

from cpa.pipeline.config import (
    PipelineConfig,
    DiscoveryConfig,
    AlignmentConfig,
    DescriptorConfig,
    SearchConfig,
    DetectionConfig,
    CertificateConfig,
    ComputationConfig,
    ConfigProfile,
)
from cpa.pipeline.results import (
    AtlasResult,
    FoundationResult,
    ExplorationResult,
    ValidationResult,
)
from cpa.pipeline.checkpointing import CheckpointManager
from cpa.pipeline.orchestrator import CPAOrchestrator

__all__ = [
    "CPAOrchestrator",
    "PipelineConfig",
    "DiscoveryConfig",
    "AlignmentConfig",
    "DescriptorConfig",
    "SearchConfig",
    "DetectionConfig",
    "CertificateConfig",
    "ComputationConfig",
    "ConfigProfile",
    "CheckpointManager",
    "AtlasResult",
    "FoundationResult",
    "ExplorationResult",
    "ValidationResult",
]

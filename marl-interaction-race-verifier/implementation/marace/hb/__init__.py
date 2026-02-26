"""
Happens-before (HB) and partial-order engine for the MARACE system.

Provides vector clocks, HB relation construction, transitive closure,
connected component extraction, causal inference, and HB graph visualization.
"""

from marace.hb.vector_clock import VectorClock, VectorClockManager
from marace.hb.hb_graph import HBGraph, HBRelation
from marace.hb.causal_inference import (
    ObservationDependencyAnalyzer,
    PhysicsMediatedCausalityDetector,
    CommunicationCausalityExtractor,
    EnvironmentMediatedCausalChain,
    CausalInferenceEngine,
    CausalChainType,
    CausalEvidence,
    SoundnessClassification,
    TransitiveCausalityClosure,
    classify_edge_soundness,
)
from marace.hb.interaction_groups import (
    InteractionGroup,
    InteractionGroupExtractor,
    GroupMerger,
    GroupPartitioner,
    InteractionStrengthEstimator,
    GroupEvolution,
)
from marace.hb.visualization import (
    HBVisualizer,
    SpaceTimeDiagram,
    InteractionGroupDiagram,
    CausalChainHighlighter,
)

__all__ = [
    "VectorClock",
    "VectorClockManager",
    "HBGraph",
    "HBRelation",
    "ObservationDependencyAnalyzer",
    "PhysicsMediatedCausalityDetector",
    "CommunicationCausalityExtractor",
    "EnvironmentMediatedCausalChain",
    "CausalInferenceEngine",
    "CausalChainType",
    "CausalEvidence",
    "SoundnessClassification",
    "TransitiveCausalityClosure",
    "classify_edge_soundness",
    "InteractionGroup",
    "InteractionGroupExtractor",
    "GroupMerger",
    "GroupPartitioner",
    "InteractionStrengthEstimator",
    "GroupEvolution",
    "HBVisualizer",
    "SpaceTimeDiagram",
    "InteractionGroupDiagram",
    "CausalChainHighlighter",
]

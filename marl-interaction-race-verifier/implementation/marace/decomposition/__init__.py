"""Compositional decomposition engine for MARACE.

Partitions multi-agent systems into interaction groups based on
happens-before graph structure, generates assume-guarantee contracts
over shared state predicates, and composes per-group verification
results into whole-system guarantees.
"""

from marace.decomposition.interaction_graph import (
    InteractionEdge,
    InteractionGraph,
    InteractionStrengthMetrics,
    GraphConstructor,
    DynamicGraphUpdater,
)
from marace.decomposition.contracts import (
    InterfaceVariable,
    Contract,
    LinearContract,
    ContractTemplate,
    ContractGenerator,
    ContractChecker,
    ContractRefinement,
    ContractComposition,
    CompositionSoundnessTheorem,
    ContractRefinementChecker,
    ContractWeakeningStrengthening,
    ProofObligation,
)
from marace.decomposition.assume_guarantee import (
    CompositionRule,
    CompositionResult,
    AssumeGuaranteeVerifier,
    CircularDependencyResolver,
    SoundnessProof,
    GroupMergingStrategy,
    AdaptiveDecomposition,
)
from marace.decomposition.partitioning import (
    SpectralPartitioner,
    MinCutPartitioner,
    HierarchicalPartitioner,
    ConstrainedPartitioner,
    PartitionQualityMetrics,
    PartitionRefinement,
)
from marace.decomposition.smt_discharge import (
    SMTTheory,
    SMTEncoder,
    LPDischarger,
    ContractDischarger,
    CompositionSoundnessProver,
    DischargeResult,
)
from marace.decomposition.group_size_theory import (
    InteractionGraphModel,
    BoundedDegreeCondition,
    SpatialLocalityCondition,
    TransitiveClosure,
    AdaptiveDegradation,
    EmpiricalGroupSizeAnalysis,
)

__all__ = [
    "InteractionEdge",
    "InteractionGraph",
    "InteractionStrengthMetrics",
    "GraphConstructor",
    "DynamicGraphUpdater",
    "InterfaceVariable",
    "Contract",
    "LinearContract",
    "ContractTemplate",
    "ContractGenerator",
    "ContractChecker",
    "ContractRefinement",
    "ContractComposition",
    "CompositionSoundnessTheorem",
    "ContractRefinementChecker",
    "ContractWeakeningStrengthening",
    "ProofObligation",
    "CompositionRule",
    "CompositionResult",
    "AssumeGuaranteeVerifier",
    "CircularDependencyResolver",
    "SoundnessProof",
    "GroupMergingStrategy",
    "AdaptiveDecomposition",
    "SpectralPartitioner",
    "MinCutPartitioner",
    "HierarchicalPartitioner",
    "ConstrainedPartitioner",
    "PartitionQualityMetrics",
    "PartitionRefinement",
    "SMTTheory",
    "SMTEncoder",
    "LPDischarger",
    "ContractDischarger",
    "CompositionSoundnessProver",
    "DischargeResult",
    "InteractionGraphModel",
    "BoundedDegreeCondition",
    "SpatialLocalityCondition",
    "TransitiveClosure",
    "AdaptiveDegradation",
    "EmpiricalGroupSizeAnalysis",
]

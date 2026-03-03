"""CPA core subpackage.

Provides core types, structural causal model operations, context
management, and multi-context causal model infrastructure.
"""

from cpa.core.types import (
    PlasticityClass,
    CertificateType,
    EdgeClassification,
    ChangeType,
    SCM,
    Context,
    MCCM,
    AlignmentMapping,
    PlasticityDescriptor,
    TippingPoint,
    RobustnessCertificate,
    QDGenome,
    QDArchiveEntry,
    CVTCell,
)
from cpa.core.scm import (
    StructuralCausalModel,
    random_dag,
    erdos_renyi_dag,
    chain_dag,
    fork_dag,
    collider_dag,
)
from cpa.core.context import ContextSpace, ContextPartition
from cpa.core.mccm import (
    MultiContextCausalModel,
    build_mccm_from_data,
    build_mccm_from_scms,
)

__all__ = [
    # Enums
    "PlasticityClass",
    "CertificateType",
    "EdgeClassification",
    "ChangeType",
    # Dataclasses
    "SCM",
    "Context",
    "MCCM",
    "AlignmentMapping",
    "PlasticityDescriptor",
    "TippingPoint",
    "RobustnessCertificate",
    "QDGenome",
    "QDArchiveEntry",
    "CVTCell",
    # SCM
    "StructuralCausalModel",
    "random_dag",
    "erdos_renyi_dag",
    "chain_dag",
    "fork_dag",
    "collider_dag",
    # Context
    "ContextSpace",
    "ContextPartition",
    # MCCM
    "MultiContextCausalModel",
    "build_mccm_from_data",
    "build_mccm_from_scms",
]

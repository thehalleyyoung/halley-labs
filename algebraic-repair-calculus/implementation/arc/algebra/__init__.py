"""
arc.algebra — Three-Sorted Delta Algebra
==========================================

Provides the algebraic framework Δ = (Δ_S, Δ_D, Δ_Q, ∘, ⁻¹, push) for
reasoning about and repairing data pipeline perturbations.

Modules:
- schema_delta:  Schema Delta Monoid (Δ_S)
- data_delta:    Data Delta Group (Δ_D)
- quality_delta: Quality Delta Lattice (Δ_Q)
- interaction:   Interaction Homomorphisms (φ, ψ)
- push:          Push Operators (push_f^X)
- composition:   Compound Perturbation
"""

from arc.algebra.schema_delta import (
    AddColumn,
    AddConstraint,
    ChangeType,
    ColumnDef,
    Conflict,
    ConflictType,
    ConstraintDef,
    ConstraintType,
    DropColumn,
    DropConstraint,
    RenameColumn,
    SQLType,
    Schema,
    SchemaDelta,
    SchemaOperation,
    can_widen_type,
    coercion_expression,
    diff_schemas,
    widest_type,
)

from arc.algebra.data_delta import (
    DataDelta,
    DataOperation,
    DeleteOp,
    InsertOp,
    MultiSet,
    TypedTuple,
    UpdateOp,
    diff_relations,
)

from arc.algebra.quality_delta import (
    ConstraintAdded,
    ConstraintRemoved,
    ConstraintStatus,
    DistributionShift,
    DistributionSummary,
    QualityDelta,
    QualityImprovement,
    QualityOperation,
    QualityState,
    QualityViolation,
    SeverityLevel,
    ViolationType,
)

from arc.algebra.interaction import (
    ConstraintViolationDetector,
    PhiHomomorphism,
    PsiHomomorphism,
    apply_schema_interaction,
)

from arc.algebra.push import (
    CTEPush,
    FilterPush,
    GroupByPush,
    JoinPush,
    JoinType,
    OperatorContext,
    PushOperator,
    SelectPush,
    SetOpPush,
    SetOpType,
    UnionPush,
    WindowPush,
    get_push_operator,
    push_all_deltas,
    push_data_delta,
    push_quality_delta,
    push_schema_delta,
    register_push_operator,
    supported_operators,
)

from arc.algebra.composition import (
    CompoundPerturbation,
    PipelineState,
    compose_chain,
    compose_parallel,
    diff_states,
    verify_composition_associativity,
    verify_identity,
    verify_inverse,
)

from arc.algebra.propagation import (
    AggregationStrategy,
    BatchPropagator,
    DeltaPropagator,
    PropagationAnalyzer,
    PropagationMode,
    PropagationPath,
    PropagationPathEntry,
    PropagationResult,
    PropagationStats,
    PropagationValidator,
    estimate_impact,
    propagate_multi,
    propagate_single,
)

from arc.algebra.annihilation import (
    AnnihilationDetector,
    AnnihilationReason,
    AnnihilationResult,
    AnnihilationType,
    annihilation_strength,
    check_annihilation,
    compute_annihilation_profile,
    find_first_annihilation,
)

from arc.algebra.index_delta import (
    IndexDelta,
    IndexOperation,
    IndexOpType,
    IndexSpec,
    IndexType,
    create_index_delta,
    drop_index_delta,
)

__all__ = [
    # Schema Delta
    "AddColumn",
    "AddConstraint",
    "ChangeType",
    "ColumnDef",
    "Conflict",
    "ConflictType",
    "ConstraintDef",
    "ConstraintType",
    "DropColumn",
    "DropConstraint",
    "RenameColumn",
    "SQLType",
    "Schema",
    "SchemaDelta",
    "SchemaOperation",
    "can_widen_type",
    "coercion_expression",
    "diff_schemas",
    "widest_type",
    # Data Delta
    "DataDelta",
    "DataOperation",
    "DeleteOp",
    "InsertOp",
    "MultiSet",
    "TypedTuple",
    "UpdateOp",
    "diff_relations",
    # Quality Delta
    "ConstraintAdded",
    "ConstraintRemoved",
    "ConstraintStatus",
    "DistributionShift",
    "DistributionSummary",
    "QualityDelta",
    "QualityImprovement",
    "QualityOperation",
    "QualityState",
    "QualityViolation",
    "SeverityLevel",
    "ViolationType",
    # Interaction
    "ConstraintViolationDetector",
    "PhiHomomorphism",
    "PsiHomomorphism",
    "apply_schema_interaction",
    # Push
    "CTEPush",
    "FilterPush",
    "GroupByPush",
    "JoinPush",
    "JoinType",
    "OperatorContext",
    "PushOperator",
    "SelectPush",
    "SetOpPush",
    "SetOpType",
    "UnionPush",
    "WindowPush",
    "get_push_operator",
    "push_all_deltas",
    "push_data_delta",
    "push_quality_delta",
    "push_schema_delta",
    "register_push_operator",
    "supported_operators",
    # Composition
    "CompoundPerturbation",
    "PipelineState",
    "compose_chain",
    "compose_parallel",
    "diff_states",
    "verify_composition_associativity",
    "verify_identity",
    "verify_inverse",
    # Propagation
    "AggregationStrategy",
    "BatchPropagator",
    "DeltaPropagator",
    "PropagationAnalyzer",
    "PropagationMode",
    "PropagationPath",
    "PropagationPathEntry",
    "PropagationResult",
    "PropagationStats",
    "PropagationValidator",
    "estimate_impact",
    "propagate_multi",
    "propagate_single",
    # Annihilation
    "AnnihilationDetector",
    "AnnihilationReason",
    "AnnihilationResult",
    "AnnihilationType",
    "annihilation_strength",
    "check_annihilation",
    "compute_annihilation_profile",
    "find_first_annihilation",
    # Index Delta
    "IndexDelta",
    "IndexOperation",
    "IndexOpType",
    "IndexSpec",
    "IndexType",
    "create_index_delta",
    "drop_index_delta",
]

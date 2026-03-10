"""
usability_oracle.smt_repair — SMT-backed repair synthesis.

Encodes usability bottlenecks as SMT constraints over UI properties and
searches for minimal mutations that eliminate the bottleneck.

Re-exports all public types and protocols::

    from usability_oracle.smt_repair import RepairResult, RepairSolver
"""

from __future__ import annotations

from usability_oracle.smt_repair.types import (
    ConstraintKind,
    ConstraintSystem,
    MutationCandidate,
    MutationType,
    RepairConstraint,
    RepairResult,
    SolverStatus,
    UIVariable,
    VariableSort,
)

from usability_oracle.smt_repair.protocols import (
    ConstraintGenerator as ConstraintGeneratorProtocol,
    MutationValidator as MutationValidatorProtocol,
    RepairSolver as RepairSolverProtocol,
)

from usability_oracle.smt_repair.encoding import (
    TreeEncoding,
    Z3Encoder,
)

from usability_oracle.smt_repair.constraints import (
    ConstraintGenerator,
)

from usability_oracle.smt_repair.solver import (
    RepairSolver,
)

from usability_oracle.smt_repair.mutations import (
    MutationOperator,
    ReorderChildren,
    MergeGroups,
    SplitGroup,
    AddLandmark,
    RemoveRedundant,
    AdjustSpacing,
    PromoteElement,
    AddShortcut,
    validate_mutation,
    apply_mutation,
    compose_mutations,
)

from usability_oracle.smt_repair.validator import (
    RepairValidator,
)

__all__ = [
    # types
    "ConstraintKind",
    "ConstraintSystem",
    "MutationCandidate",
    "MutationType",
    "RepairConstraint",
    "RepairResult",
    "SolverStatus",
    "UIVariable",
    "VariableSort",
    # protocols
    "ConstraintGeneratorProtocol",
    "MutationValidatorProtocol",
    "RepairSolverProtocol",
    # encoding
    "TreeEncoding",
    "Z3Encoder",
    # constraint generation
    "ConstraintGenerator",
    # solver
    "RepairSolver",
    # mutations
    "MutationOperator",
    "ReorderChildren",
    "MergeGroups",
    "SplitGroup",
    "AddLandmark",
    "RemoveRedundant",
    "AdjustSpacing",
    "PromoteElement",
    "AddShortcut",
    "validate_mutation",
    "apply_mutation",
    "compose_mutations",
    # validator
    "RepairValidator",
]

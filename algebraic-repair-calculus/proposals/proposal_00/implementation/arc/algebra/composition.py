"""
Compound Perturbation (Algorithm A5)
=====================================

Implements the compound perturbation type that combines all three delta sorts
with their interaction effects. This is the main entry point for the
three-sorted delta algebra.

A CompoundPerturbation is a triple (δ_S, δ_D, δ_Q) from Δ_S × Δ_D × Δ_Q
with composition that accounts for interactions:

    (σ₁, δ₁, γ₁) ∘ (σ₂, δ₂, γ₂) =
        (σ₁ ∘ σ₂,
         δ₁ ∘ φ(σ₁)(δ₂),
         γ₁ ⊔ ψ(σ₁)(γ₂))

where φ is the schema→data interaction and ψ is the schema→quality interaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from arc.algebra.schema_delta import (
    Schema,
    SchemaDelta,
    SchemaOperation,
)
from arc.algebra.data_delta import (
    DataDelta,
    MultiSet,
)
from arc.algebra.quality_delta import (
    QualityDelta,
    QualityState,
)
from arc.algebra.interaction import (
    PhiHomomorphism,
    PsiHomomorphism,
    apply_schema_interaction,
)


# ---------------------------------------------------------------------------
# Pipeline State
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """
    Complete state of a data pipeline node, including schema,
    data, and quality information.
    """
    schema: Schema
    data: MultiSet
    quality: QualityState
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> PipelineState:
        return PipelineState(
            schema=self.schema.copy(),
            data=self.data.copy(),
            quality=self.quality.copy(),
            metadata=dict(self.metadata),
        )

    def row_count(self) -> int:
        return self.data.cardinality()

    def column_names(self) -> List[str]:
        return self.schema.column_names()

    def quality_score(self) -> float:
        return self.quality.overall_score

    def has_violations(self) -> bool:
        return self.quality.has_violations()

    def __repr__(self) -> str:
        return (
            f"PipelineState(schema={self.schema.name}, "
            f"rows={self.row_count()}, "
            f"quality={self.quality_score():.2f})"
        )


# ---------------------------------------------------------------------------
# Compound Perturbation
# ---------------------------------------------------------------------------

class CompoundPerturbation:
    """
    A compound perturbation (δ_S, δ_D, δ_Q) from the three-sorted delta algebra.

    Composition accounts for the interaction homomorphisms:
    φ: Δ_S → End(Δ_D) and ψ: Δ_S → End(Δ_Q).

    Algorithm A5:
    (σ₁, δ₁, γ₁) ∘ (σ₂, δ₂, γ₂) =
        (σ₁ ∘ σ₂, δ₁ ∘ φ(σ₁)(δ₂), γ₁ ⊔ ψ(σ₁)(γ₂))
    """

    __slots__ = ("_schema_delta", "_data_delta", "_quality_delta", "_hash_cache")

    def __init__(
        self,
        schema_delta: Optional[SchemaDelta] = None,
        data_delta: Optional[DataDelta] = None,
        quality_delta: Optional[QualityDelta] = None,
    ) -> None:
        self._schema_delta = schema_delta or SchemaDelta.identity()
        self._data_delta = data_delta or DataDelta.zero()
        self._quality_delta = quality_delta or QualityDelta.bottom()
        self._hash_cache: Optional[int] = None

    @property
    def schema_delta(self) -> SchemaDelta:
        return self._schema_delta

    @property
    def data_delta(self) -> DataDelta:
        return self._data_delta

    @property
    def quality_delta(self) -> QualityDelta:
        return self._quality_delta

    @staticmethod
    def identity() -> CompoundPerturbation:
        """Return the identity compound perturbation."""
        return CompoundPerturbation(
            SchemaDelta.identity(),
            DataDelta.zero(),
            QualityDelta.bottom(),
        )

    @staticmethod
    def schema_only(schema_delta: SchemaDelta) -> CompoundPerturbation:
        """Create a perturbation with only a schema change."""
        return CompoundPerturbation(schema_delta=schema_delta)

    @staticmethod
    def data_only(data_delta: DataDelta) -> CompoundPerturbation:
        """Create a perturbation with only a data change."""
        return CompoundPerturbation(data_delta=data_delta)

    @staticmethod
    def quality_only(quality_delta: QualityDelta) -> CompoundPerturbation:
        """Create a perturbation with only a quality change."""
        return CompoundPerturbation(quality_delta=quality_delta)

    def compose(self, other: CompoundPerturbation) -> CompoundPerturbation:
        """
        Algorithm A5: Compose with interaction effects.

        (σ₁, δ₁, γ₁) ∘ (σ₂, δ₂, γ₂) =
            (σ₁ ∘ σ₂,
             δ₁ ∘ φ(σ₁)(δ₂),
             γ₁ ⊔ ψ(σ₁)(γ₂))
        """
        # Schema composition (monoid operation)
        composed_schema = self._schema_delta.compose(other._schema_delta)

        # Data composition with interaction:
        # Apply φ(σ₁) to δ₂, then compose with δ₁
        transformed_data = PhiHomomorphism.apply(
            self._schema_delta, other._data_delta
        )
        composed_data = self._data_delta.compose(transformed_data)

        # Quality composition with interaction:
        # Apply ψ(σ₁) to γ₂, then take lattice join with γ₁
        transformed_quality = PsiHomomorphism.apply(
            self._schema_delta, other._quality_delta
        )
        composed_quality = self._quality_delta.join(transformed_quality)

        return CompoundPerturbation(
            composed_schema, composed_data, composed_quality
        )

    def inverse(self) -> CompoundPerturbation:
        """
        Compute the inverse perturbation.

        The inverse undoes the effects: applying p then p⁻¹ yields identity.
        Note: This is exact for schema and data (group operations),
        but approximate for quality (lattice).
        """
        inv_schema = self._schema_delta.inverse()
        inv_data_raw = self._data_delta.inverse()
        inv_data = PhiHomomorphism.apply(inv_schema, inv_data_raw)

        inv_quality_ops = []
        for op in self._quality_delta.operations:
            inv_quality_ops.append(op.inverse())

        inv_quality = QualityDelta(inv_quality_ops)
        inv_quality = PsiHomomorphism.apply(inv_schema, inv_quality)

        return CompoundPerturbation(inv_schema, inv_data, inv_quality)

    def is_identity(self) -> bool:
        """Check if this is the identity perturbation."""
        return (
            self._schema_delta.is_identity()
            and self._data_delta.is_zero()
            and self._quality_delta.is_bottom()
        )

    def apply(self, state: PipelineState) -> PipelineState:
        """
        Apply this compound perturbation to a pipeline state.

        Order: schema first, then data, then quality.
        Schema changes affect how data and quality deltas are interpreted.
        """
        result = state.copy()

        # Apply schema delta
        result.schema = self._schema_delta.apply_to_schema(result.schema)

        # Apply data delta
        result.data = self._data_delta.apply_to_data(result.data)

        # Apply quality delta
        result.quality = self._quality_delta.apply_to_quality_state(result.quality)

        return result

    def normalize(self) -> CompoundPerturbation:
        """Normalize all three components."""
        return CompoundPerturbation(
            self._schema_delta.normalize(),
            self._data_delta.normalize(),
            self._quality_delta,
        )

    def severity(self) -> float:
        """Overall severity of this perturbation."""
        schema_severity = 0.0
        if not self._schema_delta.is_identity():
            schema_severity = min(self._schema_delta.operation_count() * 0.1, 0.5)

        data_severity = 0.0
        if not self._data_delta.is_zero():
            rows = self._data_delta.affected_rows_count()
            data_severity = min(rows / 10000.0, 0.5)

        quality_severity = self._quality_delta.severity()

        return min(1.0, schema_severity + data_severity + quality_severity)

    def affected_columns(self) -> Set[str]:
        """All columns affected by any component."""
        cols: Set[str] = set()
        cols |= self._schema_delta.affected_columns()
        cols |= self._quality_delta.affected_columns()
        return cols

    def has_schema_changes(self) -> bool:
        return not self._schema_delta.is_identity()

    def has_data_changes(self) -> bool:
        return not self._data_delta.is_zero()

    def has_quality_changes(self) -> bool:
        return not self._quality_delta.is_bottom()

    def schema_operation_count(self) -> int:
        return self._schema_delta.operation_count()

    def data_operation_count(self) -> int:
        return self._data_delta.operation_count()

    def quality_operation_count(self) -> int:
        return self._quality_delta.operation_count()

    def total_operation_count(self) -> int:
        return (
            self.schema_operation_count()
            + self.data_operation_count()
            + self.quality_operation_count()
        )

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of this perturbation."""
        return {
            "is_identity": self.is_identity(),
            "severity": self.severity(),
            "schema_ops": self.schema_operation_count(),
            "data_ops": self.data_operation_count(),
            "quality_ops": self.quality_operation_count(),
            "affected_columns": sorted(self.affected_columns()),
            "has_schema": self.has_schema_changes(),
            "has_data": self.has_data_changes(),
            "has_quality": self.has_quality_changes(),
            "data_rows_affected": self._data_delta.affected_rows_count(),
            "data_net_change": self._data_delta.net_row_change(),
            "quality_severity": self._quality_delta.severity(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "schema_delta": self._schema_delta.to_dict(),
            "data_delta": self._data_delta.to_dict(),
            "quality_delta": self._quality_delta.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> CompoundPerturbation:
        """Deserialize from dictionary."""
        return CompoundPerturbation(
            schema_delta=SchemaDelta.from_dict(data.get("schema_delta", {})),
            data_delta=DataDelta.from_dict(data.get("data_delta", {})),
            quality_delta=QualityDelta.from_dict(data.get("quality_delta", {})),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompoundPerturbation):
            return NotImplemented
        return (
            self._schema_delta == other._schema_delta
            and self._data_delta == other._data_delta
            and self._quality_delta == other._quality_delta
        )

    def __hash__(self) -> int:
        if self._hash_cache is None:
            self._hash_cache = hash(
                (
                    hash(self._schema_delta),
                    hash(self._data_delta),
                    hash(self._quality_delta),
                )
            )
        return self._hash_cache

    def __repr__(self) -> str:
        if self.is_identity():
            return "CompoundPerturbation(identity)"
        parts = []
        if self.has_schema_changes():
            parts.append(f"schema={self.schema_operation_count()} ops")
        if self.has_data_changes():
            parts.append(f"data={self.data_operation_count()} ops")
        if self.has_quality_changes():
            parts.append(f"quality={self.quality_operation_count()} ops")
        return f"CompoundPerturbation({', '.join(parts)})"

    def __bool__(self) -> bool:
        return not self.is_identity()


# ---------------------------------------------------------------------------
# Composition Chains
# ---------------------------------------------------------------------------

def compose_chain(perturbations: List[CompoundPerturbation]) -> CompoundPerturbation:
    """Compose a chain of perturbations left to right."""
    if not perturbations:
        return CompoundPerturbation.identity()
    result = perturbations[0]
    for p in perturbations[1:]:
        result = result.compose(p)
    return result


def compose_parallel(perturbations: List[CompoundPerturbation]) -> CompoundPerturbation:
    """
    Compose perturbations that apply to independent parts.
    This is valid when perturbations don't conflict.
    """
    if not perturbations:
        return CompoundPerturbation.identity()
    result = perturbations[0]
    for p in perturbations[1:]:
        new_schema = SchemaDelta(
            list(result.schema_delta.operations) + list(p.schema_delta.operations)
        )
        new_data = result.data_delta.compose(p.data_delta)
        new_quality = result.quality_delta.join(p.quality_delta)
        result = CompoundPerturbation(new_schema, new_data, new_quality)
    return result


# ---------------------------------------------------------------------------
# Perturbation Diff
# ---------------------------------------------------------------------------

def diff_states(
    old_state: PipelineState, new_state: PipelineState
) -> CompoundPerturbation:
    """
    Compute the compound perturbation that transforms old_state into new_state.
    """
    from arc.algebra.schema_delta import diff_schemas

    schema_delta = diff_schemas(old_state.schema, new_state.schema)
    data_delta = DataDelta.from_diff(old_state.data, new_state.data)
    quality_delta = _diff_quality_states(old_state.quality, new_state.quality)

    return CompoundPerturbation(schema_delta, data_delta, quality_delta)


def _diff_quality_states(
    old_quality: QualityState, new_quality: QualityState
) -> QualityDelta:
    """Compute quality delta between two quality states."""
    from arc.algebra.quality_delta import (
        QualityViolation as QV,
        QualityImprovement as QI,
        ConstraintAdded as CA,
        ConstraintRemoved as CR,
        SeverityLevel,
        ViolationType,
        ConstraintType,
    )

    ops = []

    old_cids = set(old_quality.constraint_statuses.keys())
    new_cids = set(new_quality.constraint_statuses.keys())

    for cid in new_cids - old_cids:
        status = new_quality.constraint_statuses[cid]
        ops.append(
            CA(
                constraint_id=cid,
                constraint_type=status.constraint_type,
                predicate=status.predicate,
                columns=status.columns,
            )
        )

    for cid in old_cids - new_cids:
        ops.append(
            CR(
                constraint_id=cid,
                reason="Constraint removed between states",
            )
        )

    old_violations = set()
    for vlist in old_quality.active_violations.values():
        for v in vlist:
            old_violations.add(v)

    new_violations = set()
    for vlist in new_quality.active_violations.values():
        for v in vlist:
            new_violations.add(v)

    for v in new_violations - old_violations:
        ops.append(v)

    for v in old_violations - new_violations:
        ops.append(v.inverse())

    return QualityDelta(ops)


# ---------------------------------------------------------------------------
# Verification Helpers
# ---------------------------------------------------------------------------

def verify_composition_associativity(
    p1: CompoundPerturbation,
    p2: CompoundPerturbation,
    p3: CompoundPerturbation,
) -> bool:
    """
    Verify (p1 ∘ p2) ∘ p3 = p1 ∘ (p2 ∘ p3).
    """
    lhs = p1.compose(p2).compose(p3)
    rhs = p1.compose(p2.compose(p3))
    return lhs == rhs


def verify_identity(p: CompoundPerturbation) -> bool:
    """Verify p ∘ id = id ∘ p = p."""
    identity = CompoundPerturbation.identity()
    lhs = p.compose(identity)
    rhs = identity.compose(p)
    return lhs == p and rhs == p


def verify_inverse(
    p: CompoundPerturbation, state: PipelineState
) -> bool:
    """
    Verify that applying p then p⁻¹ returns to the original state.
    Note: This may not hold exactly for quality (lattice) but should
    hold for schema and data.
    """
    p_inv = p.inverse()
    after_p = p.apply(state)
    roundtrip = p_inv.apply(after_p)

    schema_ok = roundtrip.schema == state.schema
    data_ok = roundtrip.data == state.data
    return schema_ok and data_ok

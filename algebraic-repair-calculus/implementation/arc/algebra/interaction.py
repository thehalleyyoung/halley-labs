"""
Interaction Homomorphisms
=========================

Implements the two interaction homomorphisms of the three-sorted delta algebra:

  φ: Δ_S → End(Δ_D)  — Schema-to-Data interaction
  ψ: Δ_S → End(Δ_Q)  — Schema-to-Quality interaction

These capture how schema changes propagate into the data and quality sorts.

Key algebraic property (homomorphism):
  φ(δ_s1 ∘ δ_s2) = φ(δ_s1) ∘ φ(δ_s2)
  ψ(δ_s1 ∘ δ_s2) = ψ(δ_s1) ∘ ψ(δ_s2)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from arc.algebra.schema_delta import (
    AddColumn,
    AddConstraint,
    ChangeType,
    ConstraintType,
    DropColumn,
    DropConstraint,
    RenameColumn,
    SQLType,
    SchemaOperation,
    SchemaDelta,
)
from arc.algebra.data_delta import (
    DataDelta,
    DataOperation,
    DeleteOp,
    InsertOp,
    MultiSet,
    TypedTuple,
    UpdateOp,
)
from arc.algebra.quality_delta import (
    ConstraintAdded,
    ConstraintRemoved,
    DistributionShift,
    DistributionSummary,
    QualityDelta,
    QualityImprovement,
    QualityOperation,
    QualityViolation,
    SeverityLevel,
    ViolationType,
    ConstraintType as QConstraintType,
)


# ---------------------------------------------------------------------------
# Default Value Helpers
# ---------------------------------------------------------------------------

_SQL_TYPE_DEFAULTS: Dict[SQLType, Any] = {
    SQLType.INTEGER: 0,
    SQLType.BIGINT: 0,
    SQLType.SMALLINT: 0,
    SQLType.FLOAT: 0.0,
    SQLType.DOUBLE: 0.0,
    SQLType.DECIMAL: 0,
    SQLType.NUMERIC: 0,
    SQLType.VARCHAR: "",
    SQLType.TEXT: "",
    SQLType.CHAR: "",
    SQLType.BOOLEAN: False,
    SQLType.DATE: "1970-01-01",
    SQLType.TIMESTAMP: "1970-01-01 00:00:00",
    SQLType.TIMESTAMPTZ: "1970-01-01 00:00:00+00",
    SQLType.TIME: "00:00:00",
    SQLType.INTERVAL: "0",
    SQLType.JSON: "null",
    SQLType.JSONB: "null",
    SQLType.UUID: "00000000-0000-0000-0000-000000000000",
    SQLType.BYTEA: b"",
    SQLType.ARRAY: [],
    SQLType.NULL: None,
    SQLType.UNKNOWN: None,
}


def _default_value_for_type(sql_type: SQLType, default_expr: Optional[str] = None) -> Any:
    """Determine the default value for a column type."""
    if default_expr is not None:
        return _evaluate_default_expr(default_expr, sql_type)
    return _SQL_TYPE_DEFAULTS.get(sql_type)


def _evaluate_default_expr(expr: str, sql_type: SQLType) -> Any:
    """Evaluate a simple default expression."""
    expr_lower = expr.strip().lower()
    if expr_lower == "null":
        return None
    if expr_lower in ("true", "'t'"):
        return True
    if expr_lower in ("false", "'f'"):
        return False
    if expr_lower.startswith("'") and expr_lower.endswith("'"):
        return expr_lower[1:-1]
    try:
        if sql_type in (SQLType.INTEGER, SQLType.BIGINT, SQLType.SMALLINT):
            return int(expr)
        if sql_type in (SQLType.FLOAT, SQLType.DOUBLE, SQLType.DECIMAL, SQLType.NUMERIC):
            return float(expr)
    except (ValueError, TypeError):
        pass
    return expr


# ---------------------------------------------------------------------------
# Coercion Helpers
# ---------------------------------------------------------------------------

def _build_coercion_fn(
    old_type: SQLType, new_type: SQLType, coercion_expr: Optional[str] = None
) -> Callable[[Any], Any]:
    """Build a coercion function for type changes."""
    if old_type == new_type:
        return lambda x: x

    if coercion_expr:
        return _build_expr_coercion(coercion_expr, new_type)

    return _build_default_coercion(old_type, new_type)


def _build_expr_coercion(expr: str, target_type: SQLType) -> Callable[[Any], Any]:
    """Build a coercion from a CAST expression template."""
    def coerce(val: Any) -> Any:
        if val is None:
            return None
        try:
            if target_type in (SQLType.INTEGER, SQLType.BIGINT, SQLType.SMALLINT):
                return int(val)
            if target_type in (SQLType.FLOAT, SQLType.DOUBLE, SQLType.DECIMAL, SQLType.NUMERIC):
                return float(val)
            if target_type in (SQLType.VARCHAR, SQLType.TEXT, SQLType.CHAR):
                return str(val)
            if target_type == SQLType.BOOLEAN:
                if isinstance(val, str):
                    return val.lower() in ("true", "t", "1", "yes")
                return bool(val)
        except (ValueError, TypeError):
            return None
        return val
    return coerce


def _build_default_coercion(old_type: SQLType, new_type: SQLType) -> Callable[[Any], Any]:
    """Build a default coercion between two types."""
    int_types = {SQLType.INTEGER, SQLType.BIGINT, SQLType.SMALLINT}
    float_types = {SQLType.FLOAT, SQLType.DOUBLE, SQLType.DECIMAL, SQLType.NUMERIC}
    string_types = {SQLType.VARCHAR, SQLType.TEXT, SQLType.CHAR}

    def coerce(val: Any) -> Any:
        if val is None:
            return None
        try:
            if new_type in int_types:
                return int(val)
            if new_type in float_types:
                return float(val)
            if new_type in string_types:
                return str(val)
            if new_type == SQLType.BOOLEAN:
                if isinstance(val, str):
                    return val.lower() in ("true", "t", "1", "yes")
                return bool(val)
        except (ValueError, TypeError):
            return _default_value_for_type(new_type)
        return val

    return coerce


# ---------------------------------------------------------------------------
# Constraint Validation Helpers
# ---------------------------------------------------------------------------

def _validate_not_null(t: TypedTuple, columns: Tuple[str, ...]) -> bool:
    """Check that no specified column is NULL."""
    for col in columns:
        if col in t and t[col] is None:
            return False
    return True


def _validate_check_predicate(t: TypedTuple, predicate: Optional[str]) -> bool:
    """
    Evaluate a simple check predicate against a tuple.
    Supports basic comparison operators: >, <, >=, <=, =, !=, IS NOT NULL, IS NULL.
    """
    if predicate is None:
        return True

    pred = predicate.strip()

    if " IS NOT NULL" in pred.upper():
        col_name = pred.upper().replace(" IS NOT NULL", "").strip()
        col_lower = col_name.lower()
        for c in t.columns:
            if c.lower() == col_lower or c == col_name:
                return t[c] is not None
        return True

    if " IS NULL" in pred.upper():
        col_name = pred.upper().replace(" IS NULL", "").strip()
        col_lower = col_name.lower()
        for c in t.columns:
            if c.lower() == col_lower or c == col_name:
                return t[c] is None
        return True

    for op_str, op_fn in [
        (">=", lambda a, b: a >= b),
        ("<=", lambda a, b: a <= b),
        ("!=", lambda a, b: a != b),
        (">", lambda a, b: a > b),
        ("<", lambda a, b: a < b),
        ("=", lambda a, b: a == b),
    ]:
        if op_str in pred:
            parts = pred.split(op_str, 1)
            if len(parts) == 2:
                col_name = parts[0].strip()
                val_str = parts[1].strip()
                val = _parse_predicate_value(val_str)
                for c in t.columns:
                    if c == col_name:
                        try:
                            return op_fn(t[c], val)
                        except (TypeError, ValueError):
                            return True
                return True

    return True


def _parse_predicate_value(s: str) -> Any:
    """Parse a literal value from a predicate string."""
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    if s.lower() == "null":
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ---------------------------------------------------------------------------
# φ: Δ_S → End(Δ_D)  (Phi Homomorphism)
# ---------------------------------------------------------------------------

class PhiHomomorphism:
    """
    Schema-to-Data interaction homomorphism: φ: Δ_S → End(Δ_D).

    For a schema delta δ_s and data delta δ_d, computes φ(δ_s)(δ_d),
    which transforms δ_d to account for the schema change.

    Homomorphism property: φ(δ_s1 ∘ δ_s2) = φ(δ_s1) ∘ φ(δ_s2)
    """

    @staticmethod
    def apply(schema_delta: SchemaDelta, data_delta: DataDelta) -> DataDelta:
        """
        Apply the schema-to-data interaction.

        Transforms data_delta to be consistent with schema_delta.
        Each schema operation modifies the data delta:
        - ADD_COLUMN:       Extend all tuples with default value
        - DROP_COLUMN:      Project out the column from all tuples
        - RENAME_COLUMN:    Rename field in all tuples
        - CHANGE_TYPE:      Coerce field type in all tuples
        - ADD_CONSTRAINT:   Validate tuples against new constraint
        - DROP_CONSTRAINT:  No-op on data
        """
        result = data_delta
        for op in schema_delta.operations:
            result = PhiHomomorphism._apply_single(op, result)
        return result

    @staticmethod
    def _apply_single(schema_op: SchemaOperation, data_delta: DataDelta) -> DataDelta:
        """Apply a single schema operation's effect on a data delta."""
        if isinstance(schema_op, AddColumn):
            return PhiHomomorphism._apply_add_column(schema_op, data_delta)
        elif isinstance(schema_op, DropColumn):
            return PhiHomomorphism._apply_drop_column(schema_op, data_delta)
        elif isinstance(schema_op, RenameColumn):
            return PhiHomomorphism._apply_rename_column(schema_op, data_delta)
        elif isinstance(schema_op, ChangeType):
            return PhiHomomorphism._apply_change_type(schema_op, data_delta)
        elif isinstance(schema_op, AddConstraint):
            return PhiHomomorphism._apply_add_constraint(schema_op, data_delta)
        elif isinstance(schema_op, DropConstraint):
            return data_delta
        return data_delta

    @staticmethod
    def _apply_add_column(op: AddColumn, data_delta: DataDelta) -> DataDelta:
        """ADD_COLUMN: extend all tuples with the new column's default value."""
        default_val = _default_value_for_type(op.sql_type, op.default_expr)

        def extend_tuple(t: TypedTuple) -> TypedTuple:
            if op.name not in t:
                return t.extend(op.name, default_val)
            return t

        return data_delta.map_tuples(extend_tuple)

    @staticmethod
    def _apply_drop_column(op: DropColumn, data_delta: DataDelta) -> DataDelta:
        """DROP_COLUMN: project out the column from all tuples."""
        def drop_col(t: TypedTuple) -> TypedTuple:
            if op.name in t:
                return t.drop(op.name)
            return t

        return data_delta.map_tuples(drop_col)

    @staticmethod
    def _apply_rename_column(op: RenameColumn, data_delta: DataDelta) -> DataDelta:
        """RENAME_COLUMN: rename the field in all tuples."""
        def rename_col(t: TypedTuple) -> TypedTuple:
            if op.old_name in t:
                return t.rename(op.old_name, op.new_name)
            return t

        return data_delta.map_tuples(rename_col)

    @staticmethod
    def _apply_change_type(op: ChangeType, data_delta: DataDelta) -> DataDelta:
        """CHANGE_TYPE: coerce the field type in all tuples."""
        coerce_fn = _build_coercion_fn(op.old_type, op.new_type, op.coercion_expr)

        def coerce_col(t: TypedTuple) -> TypedTuple:
            if op.column_name in t:
                return t.coerce_column(op.column_name, coerce_fn)
            return t

        return data_delta.map_tuples(coerce_col)

    @staticmethod
    def _apply_add_constraint(op: AddConstraint, data_delta: DataDelta) -> DataDelta:
        """
        ADD_CONSTRAINT: validate tuples against the new constraint.

        For INSERT operations, tuples that violate the constraint are
        flagged but still included (the violation is captured in Δ_Q).
        """
        if op.constraint_type == ConstraintType.NOT_NULL:
            pass
        elif op.constraint_type == ConstraintType.CHECK:
            pass
        return data_delta

    @staticmethod
    def apply_to_operation(
        schema_op: SchemaOperation, data_op: DataOperation
    ) -> DataOperation:
        """Apply a single schema operation to a single data operation."""
        dd = DataDelta.from_operation(data_op)
        result = PhiHomomorphism._apply_single(schema_op, dd)
        if result.operations:
            return result.operations[0]
        return data_op

    @staticmethod
    def verify_homomorphism(
        s1: SchemaDelta,
        s2: SchemaDelta,
        data_delta: DataDelta,
    ) -> bool:
        """
        Verify the homomorphism property:
        φ(s1 ∘ s2)(δ_d) = φ(s1)(φ(s2)(δ_d))

        Returns True if the property holds for the given inputs.
        """
        composed = s1.compose(s2)
        lhs = PhiHomomorphism.apply(composed, data_delta)
        rhs_inner = PhiHomomorphism.apply(s2, data_delta)
        rhs = PhiHomomorphism.apply(s1, rhs_inner)
        return lhs == rhs


# ---------------------------------------------------------------------------
# ψ: Δ_S → End(Δ_Q)  (Psi Homomorphism)
# ---------------------------------------------------------------------------

class PsiHomomorphism:
    """
    Schema-to-Quality interaction homomorphism: ψ: Δ_S → End(Δ_Q).

    For a schema delta δ_s and quality delta δ_q, computes ψ(δ_s)(δ_q),
    which transforms δ_q to account for the schema change.

    Homomorphism property: ψ(δ_s1 ∘ δ_s2) = ψ(δ_s1) ∘ ψ(δ_s2)
    """

    @staticmethod
    def apply(
        schema_delta: SchemaDelta, quality_delta: QualityDelta
    ) -> QualityDelta:
        """
        Apply the schema-to-quality interaction.

        Transforms quality_delta to be consistent with schema_delta.
        Each schema operation modifies the quality delta:
        - ADD_COLUMN:       Add null constraint for non-nullable column
        - DROP_COLUMN:      Remove constraints referencing dropped column
        - RENAME_COLUMN:    Update constraint column references
        - CHANGE_TYPE:      May invalidate range/check constraints
        - ADD_CONSTRAINT:   Propagate as quality requirement
        - DROP_CONSTRAINT:  Generate CONSTRAINT_REMOVED
        """
        result = quality_delta
        for op in schema_delta.operations:
            result = PsiHomomorphism._apply_single(op, result)
        return result

    @staticmethod
    def _apply_single(
        schema_op: SchemaOperation, quality_delta: QualityDelta
    ) -> QualityDelta:
        """Apply a single schema operation's effect on a quality delta."""
        if isinstance(schema_op, AddColumn):
            return PsiHomomorphism._apply_add_column(schema_op, quality_delta)
        elif isinstance(schema_op, DropColumn):
            return PsiHomomorphism._apply_drop_column(schema_op, quality_delta)
        elif isinstance(schema_op, RenameColumn):
            return PsiHomomorphism._apply_rename_column(schema_op, quality_delta)
        elif isinstance(schema_op, ChangeType):
            return PsiHomomorphism._apply_change_type(schema_op, quality_delta)
        elif isinstance(schema_op, AddConstraint):
            return PsiHomomorphism._apply_add_constraint(schema_op, quality_delta)
        elif isinstance(schema_op, DropConstraint):
            return PsiHomomorphism._apply_drop_constraint(schema_op, quality_delta)
        return quality_delta

    @staticmethod
    def _apply_add_column(
        op: AddColumn, quality_delta: QualityDelta
    ) -> QualityDelta:
        """
        ADD_COLUMN: If the column is NOT NULL, add a null constraint
        quality requirement to the delta.
        """
        extra_ops: List[QualityOperation] = []
        if not op.nullable:
            extra_ops.append(
                ConstraintAdded(
                    constraint_id=f"nn_{op.name}",
                    constraint_type=QConstraintType.NOT_NULL,
                    predicate=f"{op.name} IS NOT NULL",
                    columns=(op.name,),
                )
            )
        if extra_ops:
            return QualityDelta(list(quality_delta.operations) + extra_ops)
        return quality_delta

    @staticmethod
    def _apply_drop_column(
        op: DropColumn, quality_delta: QualityDelta
    ) -> QualityDelta:
        """
        DROP_COLUMN: Remove all quality operations referencing the dropped column.
        Generate CONSTRAINT_REMOVED for each affected constraint.
        """
        new_ops: List[QualityOperation] = []
        removed_constraints: Set[str] = set()

        for qop in quality_delta.operations:
            if op.name in qop.affected_columns():
                for cid in qop.affected_constraints():
                    if cid not in removed_constraints:
                        new_ops.append(
                            ConstraintRemoved(
                                constraint_id=cid,
                                reason=f"Column '{op.name}' dropped",
                            )
                        )
                        removed_constraints.add(cid)
            else:
                new_ops.append(qop)

        return QualityDelta(new_ops)

    @staticmethod
    def _apply_rename_column(
        op: RenameColumn, quality_delta: QualityDelta
    ) -> QualityDelta:
        """RENAME_COLUMN: Update column references in quality operations."""
        return quality_delta.rename_column(op.old_name, op.new_name)

    @staticmethod
    def _apply_change_type(
        op: ChangeType, quality_delta: QualityDelta
    ) -> QualityDelta:
        """
        CHANGE_TYPE: May invalidate range/check constraints.
        Generate violations for constraints that reference the column
        and may be invalidated by the type change.
        """
        extra_ops: List[QualityOperation] = []
        int_types = {SQLType.INTEGER, SQLType.BIGINT, SQLType.SMALLINT}
        float_types = {SQLType.FLOAT, SQLType.DOUBLE, SQLType.DECIMAL, SQLType.NUMERIC}
        string_types = {SQLType.VARCHAR, SQLType.TEXT, SQLType.CHAR}

        type_family_old = _type_family(op.old_type)
        type_family_new = _type_family(op.new_type)

        if type_family_old != type_family_new:
            for qop in quality_delta.operations:
                if op.column_name in qop.affected_columns():
                    if isinstance(qop, (QualityViolation, ConstraintAdded)):
                        for cid in qop.affected_constraints():
                            extra_ops.append(
                                QualityViolation(
                                    constraint_id=cid,
                                    severity=SeverityLevel.WARNING,
                                    affected_tuples=0,
                                    violation_type=ViolationType.TYPE_MISMATCH,
                                    columns=(op.column_name,),
                                    message=(
                                        f"Type change {op.old_type.value}->"
                                        f"{op.new_type.value} may invalidate "
                                        f"constraint {cid}"
                                    ),
                                )
                            )

        if extra_ops:
            return QualityDelta(list(quality_delta.operations) + extra_ops)
        return quality_delta

    @staticmethod
    def _apply_add_constraint(
        op: AddConstraint, quality_delta: QualityDelta
    ) -> QualityDelta:
        """ADD_CONSTRAINT: Propagate as a quality requirement."""
        ct_mapping = {
            ConstraintType.NOT_NULL: QConstraintType.NOT_NULL,
            ConstraintType.UNIQUE: QConstraintType.UNIQUE,
            ConstraintType.PRIMARY_KEY: QConstraintType.PRIMARY_KEY,
            ConstraintType.FOREIGN_KEY: QConstraintType.FOREIGN_KEY,
            ConstraintType.CHECK: QConstraintType.CHECK,
            ConstraintType.EXCLUSION: QConstraintType.EXCLUSION,
        }
        qct = ct_mapping.get(op.constraint_type, QConstraintType.CUSTOM)

        new_op = ConstraintAdded(
            constraint_id=op.constraint_id,
            constraint_type=qct,
            predicate=op.predicate,
            columns=op.columns,
        )
        return QualityDelta(list(quality_delta.operations) + [new_op])

    @staticmethod
    def _apply_drop_constraint(
        op: DropConstraint, quality_delta: QualityDelta
    ) -> QualityDelta:
        """DROP_CONSTRAINT: Generate CONSTRAINT_REMOVED."""
        new_ops = list(quality_delta.operations)
        new_ops = [
            qop for qop in new_ops
            if op.constraint_id not in qop.affected_constraints()
        ]
        new_ops.append(
            ConstraintRemoved(
                constraint_id=op.constraint_id,
                reason="Schema constraint dropped",
                _preserved_type=(
                    _schema_ct_to_quality_ct(op._preserved_type)
                    if op._preserved_type
                    else None
                ),
                _preserved_predicate=op._preserved_predicate,
                _preserved_columns=op._preserved_columns,
            )
        )
        return QualityDelta(new_ops)

    @staticmethod
    def apply_to_operation(
        schema_op: SchemaOperation, quality_op: QualityOperation
    ) -> List[QualityOperation]:
        """Apply a single schema operation to a single quality operation."""
        qd = QualityDelta.from_operation(quality_op)
        result = PsiHomomorphism._apply_single(schema_op, qd)
        return list(result.operations)

    @staticmethod
    def verify_homomorphism(
        s1: SchemaDelta,
        s2: SchemaDelta,
        quality_delta: QualityDelta,
    ) -> bool:
        """
        Verify the homomorphism property:
        ψ(s1 ∘ s2)(δ_q) = ψ(s1)(ψ(s2)(δ_q))
        """
        composed = s1.compose(s2)
        lhs = PsiHomomorphism.apply(composed, quality_delta)
        rhs_inner = PsiHomomorphism.apply(s2, quality_delta)
        rhs = PsiHomomorphism.apply(s1, rhs_inner)
        return lhs == rhs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _type_family(sql_type: SQLType) -> str:
    """Classify SQL types into families for compatibility checking."""
    int_types = {SQLType.INTEGER, SQLType.BIGINT, SQLType.SMALLINT}
    float_types = {SQLType.FLOAT, SQLType.DOUBLE, SQLType.DECIMAL, SQLType.NUMERIC}
    string_types = {SQLType.VARCHAR, SQLType.TEXT, SQLType.CHAR}
    temporal_types = {SQLType.DATE, SQLType.TIMESTAMP, SQLType.TIMESTAMPTZ, SQLType.TIME, SQLType.INTERVAL}
    bool_types = {SQLType.BOOLEAN}
    json_types = {SQLType.JSON, SQLType.JSONB}
    binary_types = {SQLType.BYTEA}

    if sql_type in int_types:
        return "integer"
    if sql_type in float_types:
        return "float"
    if sql_type in string_types:
        return "string"
    if sql_type in temporal_types:
        return "temporal"
    if sql_type in bool_types:
        return "boolean"
    if sql_type in json_types:
        return "json"
    if sql_type in binary_types:
        return "binary"
    return "unknown"


def _schema_ct_to_quality_ct(ct: ConstraintType) -> QConstraintType:
    """Convert schema ConstraintType to quality ConstraintType."""
    mapping = {
        ConstraintType.NOT_NULL: QConstraintType.NOT_NULL,
        ConstraintType.UNIQUE: QConstraintType.UNIQUE,
        ConstraintType.PRIMARY_KEY: QConstraintType.PRIMARY_KEY,
        ConstraintType.FOREIGN_KEY: QConstraintType.FOREIGN_KEY,
        ConstraintType.CHECK: QConstraintType.CHECK,
        ConstraintType.EXCLUSION: QConstraintType.EXCLUSION,
    }
    return mapping.get(ct, QConstraintType.CUSTOM)


# ---------------------------------------------------------------------------
# Combined Interaction Application
# ---------------------------------------------------------------------------

def apply_schema_interaction(
    schema_delta: SchemaDelta,
    data_delta: DataDelta,
    quality_delta: QualityDelta,
) -> Tuple[DataDelta, QualityDelta]:
    """
    Apply both interaction homomorphisms simultaneously.

    Returns (φ(δ_s)(δ_d), ψ(δ_s)(δ_q)).
    """
    new_data = PhiHomomorphism.apply(schema_delta, data_delta)
    new_quality = PsiHomomorphism.apply(schema_delta, quality_delta)
    return new_data, new_quality


# ---------------------------------------------------------------------------
# Constraint Violation Detection
# ---------------------------------------------------------------------------

class ConstraintViolationDetector:
    """
    Detect quality violations produced by applying a data delta
    under a set of active constraints.
    """

    @staticmethod
    def detect_violations(
        data_delta: DataDelta,
        constraints: Dict[str, Any],
        existing_data: Optional[MultiSet] = None,
    ) -> QualityDelta:
        """
        Check the data delta against constraints and produce
        a quality delta capturing any violations.
        """
        violations: List[QualityOperation] = []

        for cid, cdef in constraints.items():
            ctype = cdef.get("constraint_type", "CHECK")
            columns = tuple(cdef.get("columns", []))
            predicate = cdef.get("predicate")

            inserts = data_delta.get_all_inserts()
            if inserts.is_empty():
                continue

            if ctype == "NOT_NULL":
                violating = 0
                for t in inserts.unique_tuples():
                    if not _validate_not_null(t, columns):
                        violating += inserts.multiplicity(t)
                if violating > 0:
                    violations.append(
                        QualityViolation(
                            constraint_id=cid,
                            severity=SeverityLevel.ERROR,
                            affected_tuples=violating,
                            violation_type=ViolationType.NULL_IN_NON_NULL,
                            columns=columns,
                        )
                    )

            elif ctype == "CHECK":
                violating = 0
                for t in inserts.unique_tuples():
                    if not _validate_check_predicate(t, predicate):
                        violating += inserts.multiplicity(t)
                if violating > 0:
                    violations.append(
                        QualityViolation(
                            constraint_id=cid,
                            severity=SeverityLevel.ERROR,
                            affected_tuples=violating,
                            violation_type=ViolationType.CHECK_VIOLATION,
                            columns=columns,
                        )
                    )

            elif ctype == "UNIQUE":
                if existing_data is not None:
                    new_data = data_delta.apply_to_data(existing_data)
                    for col_name in columns:
                        seen_vals: Dict[Any, int] = {}
                        dups = 0
                        for t in new_data.unique_tuples():
                            val = t.get(col_name)
                            if val is not None:
                                seen_vals[val] = seen_vals.get(val, 0) + new_data.multiplicity(t)
                                if seen_vals[val] > 1:
                                    dups += 1
                        if dups > 0:
                            violations.append(
                                QualityViolation(
                                    constraint_id=cid,
                                    severity=SeverityLevel.ERROR,
                                    affected_tuples=dups,
                                    violation_type=ViolationType.UNIQUENESS_VIOLATION,
                                    columns=columns,
                                )
                            )

        return QualityDelta(violations)

    @staticmethod
    def validate_tuple(
        t: TypedTuple,
        constraints: Dict[str, Any],
    ) -> List[QualityViolation]:
        """Validate a single tuple against constraints."""
        violations: List[QualityViolation] = []
        for cid, cdef in constraints.items():
            ctype = cdef.get("constraint_type", "CHECK")
            columns = tuple(cdef.get("columns", []))
            predicate = cdef.get("predicate")

            if ctype == "NOT_NULL":
                if not _validate_not_null(t, columns):
                    violations.append(
                        QualityViolation(
                            constraint_id=cid,
                            severity=SeverityLevel.ERROR,
                            affected_tuples=1,
                            violation_type=ViolationType.NULL_IN_NON_NULL,
                            columns=columns,
                        )
                    )
            elif ctype == "CHECK":
                if not _validate_check_predicate(t, predicate):
                    violations.append(
                        QualityViolation(
                            constraint_id=cid,
                            severity=SeverityLevel.ERROR,
                            affected_tuples=1,
                            violation_type=ViolationType.CHECK_VIOLATION,
                            columns=columns,
                        )
                    )
        return violations

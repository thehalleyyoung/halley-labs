"""
Migration parser for ARC.

Parses Python migration files (Django and Alembic style) to extract
schema deltas. Supports:
- Django: migrations.AddField, RemoveField, AlterField, CreateModel,
  DeleteModel, RenameField, RenameModel, AddIndex, RemoveIndex,
  AddConstraint, RemoveConstraint
- Alembic: op.add_column(), op.drop_column(), op.alter_column(),
  op.create_table(), op.create_index(), op.drop_index()
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arc.types.base import (
    SchemaOpType,
    SchemaOperation,
    SchemaDelta,
    SQLType,
)


# ── Django field type → SQLType mapping ──

_DJANGO_FIELD_MAP: dict[str, SQLType] = {
    "IntegerField": SQLType.INT,
    "BigIntegerField": SQLType.BIGINT,
    "SmallIntegerField": SQLType.SMALLINT,
    "PositiveIntegerField": SQLType.INT,
    "PositiveSmallIntegerField": SQLType.SMALLINT,
    "BoundedBigAutoField": SQLType.BIGINT,
    "BoundedPositiveIntegerField": SQLType.INT,
    "AutoField": SQLType.INT,
    "BigAutoField": SQLType.BIGINT,
    "FloatField": SQLType.FLOAT,
    "DecimalField": SQLType.DECIMAL,
    "CharField": SQLType.VARCHAR,
    "TextField": SQLType.TEXT,
    "SlugField": SQLType.VARCHAR,
    "SentrySlugField": SQLType.VARCHAR,
    "URLField": SQLType.VARCHAR,
    "EmailField": SQLType.VARCHAR,
    "FilePathField": SQLType.VARCHAR,
    "BooleanField": SQLType.BOOLEAN,
    "NullBooleanField": SQLType.BOOLEAN,
    "DateField": SQLType.DATE,
    "DateTimeField": SQLType.TIMESTAMP,
    "TimeField": SQLType.TIME,
    "DurationField": SQLType.BIGINT,
    "UUIDField": SQLType.UUID,
    "JSONField": SQLType.JSONB,
    "BinaryField": SQLType.BYTEA,
    "FileField": SQLType.VARCHAR,
    "ImageField": SQLType.VARCHAR,
    "ForeignKey": SQLType.BIGINT,
    "FlexibleForeignKey": SQLType.BIGINT,
    "OneToOneField": SQLType.BIGINT,
    "ManyToManyField": SQLType.INT,
    "ArrayField": SQLType.TEXT,
    "GzippedDictField": SQLType.TEXT,
    "PickledObjectField": SQLType.BYTEA,
    "EncryptedCharField": SQLType.TEXT,
    "EncryptedJsonField": SQLType.TEXT,
    "EncryptedTextField": SQLType.TEXT,
}


@dataclass
class MigrationInfo:
    """Parsed information from a single migration file."""
    filename: str
    migration_id: str
    dependencies: list[tuple[str, str]] = field(default_factory=list)
    operations: list[MigrationOp] = field(default_factory=list)
    is_post_deployment: bool = False


@dataclass
class MigrationOp:
    """A single parsed migration operation."""
    op_type: str
    model_name: str = ""
    field_name: str = ""
    new_name: str = ""
    field_type: str = ""
    field_kwargs: dict[str, Any] = field(default_factory=dict)
    index_name: str = ""
    constraint_name: str = ""
    table_fields: list[tuple[str, str]] = field(default_factory=list)


def _resolve_field_type(node: ast.expr) -> str:
    """Extract the field class name from an AST expression."""
    if isinstance(node, ast.Call):
        return _resolve_field_type(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _extract_field_kwargs(node: ast.Call) -> dict[str, Any]:
    """Extract keyword arguments from a field constructor call."""
    kwargs: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        val = _eval_ast_literal(kw.value)
        kwargs[kw.arg] = val
    return kwargs


def _eval_ast_literal(node: ast.expr) -> Any:
    """Best-effort evaluation of an AST literal node."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Name, ast.Attribute)):
        return _resolve_field_type(node)
    if isinstance(node, ast.Call):
        return f"<call:{_resolve_field_type(node.func)}>"
    if isinstance(node, ast.List):
        return [_eval_ast_literal(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast_literal(e) for e in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_ast_literal(node.operand)
    return None


def _parse_django_operations(ops_list: ast.List) -> list[MigrationOp]:
    """Parse a Django migrations operations = [...] list."""
    results: list[MigrationOp] = []

    for elt in ops_list.elts:
        if not isinstance(elt, ast.Call):
            continue

        func_name = _resolve_field_type(elt.func)
        op = _parse_single_django_op(func_name, elt)
        if op is not None:
            results.append(op)

    return results


def _parse_single_django_op(func_name: str, call: ast.Call) -> MigrationOp | None:
    """Parse a single Django migration operation call."""
    kwargs = {
        kw.arg: kw.value for kw in call.keywords if kw.arg is not None
    }

    def _get_str_kwarg(name: str) -> str:
        node = kwargs.get(name)
        if node is None and call.args:
            # Positional arguments
            idx = {"model_name": 0, "name": 1}.get(name, -1)
            if 0 <= idx < len(call.args):
                node = call.args[idx]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return ""

    if func_name in ("AddField", "AddField"):
        model = _get_str_kwarg("model_name")
        fname = _get_str_kwarg("name")
        field_node = kwargs.get("field")
        ftype = ""
        fkwargs: dict[str, Any] = {}
        if isinstance(field_node, ast.Call):
            ftype = _resolve_field_type(field_node.func)
            fkwargs = _extract_field_kwargs(field_node)
        return MigrationOp(
            op_type="AddField",
            model_name=model,
            field_name=fname,
            field_type=ftype,
            field_kwargs=fkwargs,
        )

    if func_name == "RemoveField":
        return MigrationOp(
            op_type="RemoveField",
            model_name=_get_str_kwarg("model_name"),
            field_name=_get_str_kwarg("name"),
        )

    if func_name == "AlterField":
        model = _get_str_kwarg("model_name")
        fname = _get_str_kwarg("name")
        field_node = kwargs.get("field")
        ftype = ""
        fkwargs = {}
        if isinstance(field_node, ast.Call):
            ftype = _resolve_field_type(field_node.func)
            fkwargs = _extract_field_kwargs(field_node)
        return MigrationOp(
            op_type="AlterField",
            model_name=model,
            field_name=fname,
            field_type=ftype,
            field_kwargs=fkwargs,
        )

    if func_name == "RenameField":
        return MigrationOp(
            op_type="RenameField",
            model_name=_get_str_kwarg("model_name"),
            field_name=_get_str_kwarg("old_name"),
            new_name=_get_str_kwarg("new_name"),
        )

    if func_name == "CreateModel":
        model = _get_str_kwarg("name")
        fields_node = kwargs.get("fields")
        table_fields: list[tuple[str, str]] = []
        if isinstance(fields_node, ast.List):
            for item in fields_node.elts:
                if isinstance(item, ast.Tuple) and len(item.elts) >= 2:
                    fname_node = item.elts[0]
                    ftype_node = item.elts[1]
                    fn = ""
                    if isinstance(fname_node, ast.Constant):
                        fn = str(fname_node.value)
                    ft = ""
                    if isinstance(ftype_node, ast.Call):
                        ft = _resolve_field_type(ftype_node.func)
                    table_fields.append((fn, ft))
        return MigrationOp(
            op_type="CreateModel",
            model_name=model,
            table_fields=table_fields,
        )

    if func_name == "DeleteModel":
        return MigrationOp(
            op_type="DeleteModel",
            model_name=_get_str_kwarg("name"),
        )

    if func_name == "RenameModel":
        return MigrationOp(
            op_type="RenameModel",
            model_name=_get_str_kwarg("old_name"),
            new_name=_get_str_kwarg("new_name"),
        )

    if func_name == "AddIndex":
        model = _get_str_kwarg("model_name")
        index_node = kwargs.get("index")
        idx_name = ""
        if isinstance(index_node, ast.Call):
            idx_kwargs = _extract_field_kwargs(index_node)
            idx_name = idx_kwargs.get("name", "")
        return MigrationOp(
            op_type="AddIndex",
            model_name=model,
            index_name=idx_name if isinstance(idx_name, str) else "",
        )

    if func_name == "RemoveIndex":
        model = _get_str_kwarg("model_name")
        idx_name = _get_str_kwarg("name")
        return MigrationOp(
            op_type="RemoveIndex",
            model_name=model,
            index_name=idx_name,
        )

    if func_name == "AddConstraint":
        model = _get_str_kwarg("model_name")
        cstr_node = kwargs.get("constraint")
        cstr_name = ""
        if isinstance(cstr_node, ast.Call):
            cstr_kwargs = _extract_field_kwargs(cstr_node)
            cstr_name = cstr_kwargs.get("name", "")
        return MigrationOp(
            op_type="AddConstraint",
            model_name=model,
            constraint_name=cstr_name if isinstance(cstr_name, str) else "",
        )

    if func_name == "RemoveConstraint":
        return MigrationOp(
            op_type="RemoveConstraint",
            model_name=_get_str_kwarg("model_name"),
            constraint_name=_get_str_kwarg("name"),
        )

    return None


def parse_django_migration(filepath: str | Path) -> MigrationInfo:
    """Parse a Django migration file and extract operations.

    Parameters
    ----------
    filepath:
        Path to a Django migration Python file.

    Returns
    -------
    MigrationInfo
        Parsed migration with operations, dependencies, and metadata.
    """
    filepath = Path(filepath)
    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))

    migration_id = filepath.stem
    info = MigrationInfo(
        filename=filepath.name,
        migration_id=migration_id,
    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "Migration":
            continue

        for item in node.body:
            # Parse is_post_deployment
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == "is_post_deployment":
                        if isinstance(item.value, ast.Constant):
                            info.is_post_deployment = bool(item.value.value)

                    if isinstance(target, ast.Name) and target.id == "dependencies":
                        if isinstance(item.value, ast.List):
                            for elt in item.value.elts:
                                if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2:
                                    app = _eval_ast_literal(elt.elts[0])
                                    name = _eval_ast_literal(elt.elts[1])
                                    if isinstance(app, str) and isinstance(name, str):
                                        info.dependencies.append((app, name))

                    if isinstance(target, ast.Name) and target.id == "operations":
                        if isinstance(item.value, ast.List):
                            info.operations = _parse_django_operations(item.value)

    return info


def _django_field_to_sqltype(field_type: str) -> SQLType:
    """Map a Django field class name to a SQLType."""
    return _DJANGO_FIELD_MAP.get(field_type, SQLType.TEXT)


def migration_op_to_schema_delta(op: MigrationOp) -> SchemaDelta | None:
    """Convert a single MigrationOp to an ARC SchemaDelta.

    Returns None for operations that don't map to schema deltas
    (e.g., data migrations, RunSQL).
    """
    ops: list[SchemaOperation] = []

    if op.op_type == "AddField":
        sql_type = _django_field_to_sqltype(op.field_type)
        nullable = op.field_kwargs.get("null", False)
        ops.append(SchemaOperation(
            op_type=SchemaOpType.ADD_COLUMN,
            column_name=op.field_name,
            dtype=sql_type,
            nullable=nullable if isinstance(nullable, bool) else True,
            metadata={"model": op.model_name, "field_type": op.field_type},
        ))

    elif op.op_type == "RemoveField":
        ops.append(SchemaOperation(
            op_type=SchemaOpType.DROP_COLUMN,
            column_name=op.field_name,
            metadata={"model": op.model_name},
        ))

    elif op.op_type == "AlterField":
        sql_type = _django_field_to_sqltype(op.field_type)
        nullable = op.field_kwargs.get("null")
        max_length = op.field_kwargs.get("max_length")
        db_index = op.field_kwargs.get("db_index", False)

        ops.append(SchemaOperation(
            op_type=SchemaOpType.RETYPE_COLUMN,
            column_name=op.field_name,
            new_dtype=sql_type,
            metadata={
                "model": op.model_name,
                "field_type": op.field_type,
                "max_length": max_length,
                "db_index": db_index,
            },
        ))
        if nullable is not None and isinstance(nullable, bool):
            ops.append(SchemaOperation(
                op_type=SchemaOpType.SET_NULLABLE,
                column_name=op.field_name,
                nullable=nullable,
            ))
        if db_index:
            ops.append(SchemaOperation(
                op_type=SchemaOpType.ADD_CONSTRAINT,
                column_name=op.field_name,
                constraint=f"idx_{op.model_name}_{op.field_name}",
                metadata={"type": "index"},
            ))

    elif op.op_type == "RenameField":
        ops.append(SchemaOperation(
            op_type=SchemaOpType.RENAME_COLUMN,
            column_name=op.field_name,
            new_column_name=op.new_name,
            metadata={"model": op.model_name},
        ))

    elif op.op_type == "CreateModel":
        for fname, ftype in op.table_fields:
            sql_type = _django_field_to_sqltype(ftype)
            ops.append(SchemaOperation(
                op_type=SchemaOpType.ADD_COLUMN,
                column_name=fname,
                dtype=sql_type,
                metadata={"model": op.model_name, "field_type": ftype, "create_table": True},
            ))

    elif op.op_type == "AddIndex":
        ops.append(SchemaOperation(
            op_type=SchemaOpType.ADD_CONSTRAINT,
            column_name="",
            constraint=op.index_name or f"idx_{op.model_name}",
            metadata={"model": op.model_name, "type": "index"},
        ))

    elif op.op_type == "RemoveIndex":
        ops.append(SchemaOperation(
            op_type=SchemaOpType.DROP_CONSTRAINT,
            column_name="",
            constraint=op.index_name or f"idx_{op.model_name}",
            metadata={"model": op.model_name, "type": "index"},
        ))

    elif op.op_type == "AddConstraint":
        ops.append(SchemaOperation(
            op_type=SchemaOpType.ADD_CONSTRAINT,
            column_name="",
            constraint=op.constraint_name or f"cstr_{op.model_name}",
            metadata={"model": op.model_name, "type": "constraint"},
        ))

    elif op.op_type == "RemoveConstraint":
        ops.append(SchemaOperation(
            op_type=SchemaOpType.DROP_CONSTRAINT,
            column_name="",
            constraint=op.constraint_name or f"cstr_{op.model_name}",
            metadata={"model": op.model_name, "type": "constraint"},
        ))

    else:
        return None

    if not ops:
        return None

    return SchemaDelta(
        operations=tuple(ops),
        source_node=op.model_name,
    )


def migration_to_schema_delta(info: MigrationInfo) -> SchemaDelta:
    """Convert all operations in a MigrationInfo to a composed SchemaDelta."""
    combined = SchemaDelta()
    for op in info.operations:
        delta = migration_op_to_schema_delta(op)
        if delta is not None:
            combined = combined.compose(delta)
    return combined


def load_migration_directory(
    migration_dir: str | Path,
    limit: int | None = None,
) -> list[tuple[MigrationInfo, SchemaDelta]]:
    """Load and parse all migration files in a directory.

    Parameters
    ----------
    migration_dir:
        Path to migration directory.
    limit:
        Maximum number of migrations to parse (sorted by filename).

    Returns
    -------
    list[tuple[MigrationInfo, SchemaDelta]]
        Parsed migrations with their schema deltas, sorted by filename.
    """
    migration_dir = Path(migration_dir)
    files = sorted(migration_dir.glob("*.py"))
    # Skip __init__.py
    files = [f for f in files if f.stem != "__init__"]

    if limit is not None:
        files = files[:limit]

    results: list[tuple[MigrationInfo, SchemaDelta]] = []
    for filepath in files:
        try:
            info = parse_django_migration(filepath)
            delta = migration_to_schema_delta(info)
            results.append((info, delta))
        except Exception:
            continue

    return results


# ── Alembic-style parser (for op.add_column etc.) ──

def parse_alembic_migration(filepath: str | Path) -> list[SchemaDelta]:
    """Parse an Alembic migration file looking for op.* calls.

    Returns a list of SchemaDelta objects, one per operation found.
    """
    filepath = Path(filepath)
    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))
    deltas: list[SchemaDelta] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        # Check for op.xxx() calls
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "op":
            method = node.func.attr
            delta = _parse_alembic_op(method, node)
            if delta is not None:
                deltas.append(delta)

    return deltas


def _parse_alembic_op(method: str, call: ast.Call) -> SchemaDelta | None:
    """Parse a single Alembic op.xxx() call."""
    def _get_str_arg(idx: int) -> str:
        if idx < len(call.args):
            if isinstance(call.args[idx], ast.Constant):
                return str(call.args[idx].value)
        return ""

    if method == "add_column":
        table = _get_str_arg(0)
        col = _get_str_arg(1)
        return SchemaDelta(operations=(
            SchemaOperation(
                op_type=SchemaOpType.ADD_COLUMN,
                column_name=col,
                metadata={"table": table, "source": "alembic"},
            ),
        ))

    if method == "drop_column":
        table = _get_str_arg(0)
        col = _get_str_arg(1)
        return SchemaDelta(operations=(
            SchemaOperation(
                op_type=SchemaOpType.DROP_COLUMN,
                column_name=col,
                metadata={"table": table, "source": "alembic"},
            ),
        ))

    if method == "alter_column":
        table = _get_str_arg(0)
        col = _get_str_arg(1)
        return SchemaDelta(operations=(
            SchemaOperation(
                op_type=SchemaOpType.RETYPE_COLUMN,
                column_name=col,
                metadata={"table": table, "source": "alembic"},
            ),
        ))

    if method == "create_table":
        table = _get_str_arg(0)
        return SchemaDelta(operations=(
            SchemaOperation(
                op_type=SchemaOpType.ADD_COLUMN,
                column_name="__table__",
                metadata={"table": table, "source": "alembic", "create_table": True},
            ),
        ))

    if method == "create_index":
        idx_name = _get_str_arg(0)
        table = _get_str_arg(1)
        return SchemaDelta(operations=(
            SchemaOperation(
                op_type=SchemaOpType.ADD_CONSTRAINT,
                constraint=idx_name,
                metadata={"table": table, "source": "alembic", "type": "index"},
            ),
        ))

    if method == "drop_index":
        idx_name = _get_str_arg(0)
        return SchemaDelta(operations=(
            SchemaOperation(
                op_type=SchemaOpType.DROP_CONSTRAINT,
                constraint=idx_name,
                metadata={"source": "alembic", "type": "index"},
            ),
        ))

    return None

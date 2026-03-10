"""
JSON serialization for ARC pipeline specifications, deltas and repair plans.

Provides :class:`PipelineSpec` for loading/saving complete pipeline
definitions as JSON, including schema validation, delta serialization,
and repair plan round-tripping.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from arc.types.base import (
    AvailabilityContract,
    Column,
    CostEstimate,
    EdgeType,
    ForeignKey,
    CheckConstraint,
    NodeMetadata,
    ParameterisedType,
    QualityConstraint,
    Schema,
    SQLType,
)
from arc.types.errors import (
    ParseError,
    SchemaViolationError,
    SerializationError,
    VersionUnsupportedError,
    ErrorCode,
)
from arc.types.operators import SQLOperator
from arc.graph.pipeline import PipelineEdge, PipelineGraph, PipelineNode


# =====================================================================
# Spec version
# =====================================================================

CURRENT_SPEC_VERSION = "1.0"
SUPPORTED_SPEC_VERSIONS = ["1.0"]


# =====================================================================
# Pipeline spec
# =====================================================================

class PipelineSpec:
    """JSON pipeline specification with validation and round-trip support.

    A pipeline spec is a JSON document with the structure::

        {
          "version": "1.0",
          "name": "my_pipeline",
          "metadata": {...},
          "nodes": [...],
          "edges": [...]
        }
    """

    @staticmethod
    def validate_spec(data: dict[str, Any]) -> list[str]:
        """Validate a pipeline spec dict.  Returns list of errors."""
        errors: list[str] = []

        # Version check
        version = data.get("version")
        if version is None:
            errors.append("Missing 'version' field")
        elif version not in SUPPORTED_SPEC_VERSIONS:
            errors.append(f"Unsupported version '{version}'; supported: {SUPPORTED_SPEC_VERSIONS}")

        # Nodes
        nodes = data.get("nodes")
        if nodes is None:
            errors.append("Missing 'nodes' field")
        elif not isinstance(nodes, list):
            errors.append("'nodes' must be a list")
        else:
            node_ids: set[str] = set()
            for i, node in enumerate(nodes):
                if not isinstance(node, dict):
                    errors.append(f"nodes[{i}]: must be an object")
                    continue
                nid = node.get("node_id")
                if not nid:
                    errors.append(f"nodes[{i}]: missing 'node_id'")
                elif nid in node_ids:
                    errors.append(f"nodes[{i}]: duplicate node_id '{nid}'")
                else:
                    node_ids.add(nid)
                # Validate operator
                op = node.get("operator")
                if op:
                    try:
                        SQLOperator(op)
                    except ValueError:
                        errors.append(f"nodes[{i}]: unknown operator '{op}'")
                # Validate schemas
                for schema_key in ("input_schema", "output_schema"):
                    schema_data = node.get(schema_key)
                    if schema_data and isinstance(schema_data, dict):
                        cols = schema_data.get("columns", [])
                        for j, col in enumerate(cols):
                            if not isinstance(col, dict):
                                errors.append(f"nodes[{i}].{schema_key}.columns[{j}]: must be object")
                            elif "name" not in col:
                                errors.append(f"nodes[{i}].{schema_key}.columns[{j}]: missing 'name'")
                            elif "sql_type" not in col:
                                errors.append(f"nodes[{i}].{schema_key}.columns[{j}]: missing 'sql_type'")

        # Edges
        edges = data.get("edges")
        if edges is None:
            errors.append("Missing 'edges' field")
        elif not isinstance(edges, list):
            errors.append("'edges' must be a list")
        else:
            node_ids_set = {n.get("node_id") for n in data.get("nodes", []) if isinstance(n, dict)}
            for i, edge in enumerate(edges):
                if not isinstance(edge, dict):
                    errors.append(f"edges[{i}]: must be an object")
                    continue
                if "source" not in edge:
                    errors.append(f"edges[{i}]: missing 'source'")
                if "target" not in edge:
                    errors.append(f"edges[{i}]: missing 'target'")

        return errors

    @staticmethod
    def load(path: str | Path) -> PipelineGraph:
        """Load a pipeline graph from a JSON file."""
        p = Path(path)
        if not p.exists():
            raise SerializationError(
                f"File not found: {p}",
                code=ErrorCode.SERIALIZATION_FILE_NOT_FOUND,
            )
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ParseError(str(p), str(e), line=e.lineno)

        return PipelineSpec.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PipelineGraph:
        """Deserialize a pipeline graph from a dictionary."""
        # Validate version
        version = data.get("version", CURRENT_SPEC_VERSION)
        if version not in SUPPORTED_SPEC_VERSIONS:
            raise VersionUnsupportedError(version, SUPPORTED_SPEC_VERSIONS)

        # Validate structure
        errors = PipelineSpec.validate_spec(data)
        if errors:
            raise SchemaViolationError(errors, schema_version=version)

        return PipelineGraph.from_dict(data)

    @staticmethod
    def from_json(json_str: str) -> PipelineGraph:
        """Deserialize a pipeline graph from a JSON string."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ParseError("<string>", str(e), line=e.lineno)
        return PipelineSpec.from_dict(data)

    @staticmethod
    def save(graph: PipelineGraph, path: str | Path, indent: int = 2) -> None:
        """Serialize a pipeline graph to a JSON file."""
        data = PipelineSpec.to_dict(graph)
        with open(path, "w") as f:
            json.dump(data, f, indent=indent, default=str)

    @staticmethod
    def to_dict(graph: PipelineGraph) -> dict[str, Any]:
        """Serialize a pipeline graph to a dictionary."""
        d = graph.to_dict()
        d["version"] = CURRENT_SPEC_VERSION
        return d

    @staticmethod
    def to_json(graph: PipelineGraph, indent: int = 2) -> str:
        """Serialize a pipeline graph to a JSON string."""
        data = PipelineSpec.to_dict(graph)
        return json.dumps(data, indent=indent, default=str)


# =====================================================================
# Delta serialization
# =====================================================================

class DeltaSerializer:
    """Serialize and deserialize delta objects to/from JSON.

    Handles schema deltas, data deltas, and quality deltas with
    proper type discrimination.
    """

    DELTA_SORTS = ("schema", "data", "quality", "compound")

    @staticmethod
    def serialize_delta(delta: dict[str, Any]) -> str:
        """Serialize a delta dict to JSON."""
        return json.dumps(delta, indent=2, default=str)

    @staticmethod
    def deserialize_delta(json_str: str) -> dict[str, Any]:
        """Deserialize a delta from JSON string."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ParseError("<delta>", str(e), line=e.lineno)
        return data

    @staticmethod
    def load_delta(path: str | Path) -> dict[str, Any]:
        """Load a delta from a JSON file."""
        p = Path(path)
        if not p.exists():
            raise SerializationError(
                f"File not found: {p}",
                code=ErrorCode.SERIALIZATION_FILE_NOT_FOUND,
            )
        with open(p, "r") as f:
            return json.load(f)

    @staticmethod
    def save_delta(delta: dict[str, Any], path: str | Path, indent: int = 2) -> None:
        """Save a delta to a JSON file."""
        with open(path, "w") as f:
            json.dump(delta, f, indent=indent, default=str)

    @staticmethod
    def validate_delta(data: dict[str, Any]) -> list[str]:
        """Validate a delta spec.  Returns list of errors."""
        errors: list[str] = []
        sort = data.get("sort")
        if sort is None:
            errors.append("Missing 'sort' field")
        elif sort not in DeltaSerializer.DELTA_SORTS:
            errors.append(f"Unknown delta sort '{sort}'; expected one of {DeltaSerializer.DELTA_SORTS}")

        if sort == "schema":
            ops = data.get("operations")
            if ops is None:
                errors.append("Schema delta missing 'operations' field")
            elif not isinstance(ops, list):
                errors.append("'operations' must be a list")
            else:
                for i, op in enumerate(ops):
                    if not isinstance(op, dict):
                        errors.append(f"operations[{i}]: must be an object")
                    elif "type" not in op:
                        errors.append(f"operations[{i}]: missing 'type'")

        elif sort == "data":
            changes = data.get("changes")
            if changes is None:
                errors.append("Data delta missing 'changes' field")

        elif sort == "quality":
            constraints = data.get("constraint_changes")
            if constraints is None:
                errors.append("Quality delta missing 'constraint_changes' field")

        elif sort == "compound":
            components = data.get("components")
            if components is None:
                errors.append("Compound delta missing 'components' field")
            elif not isinstance(components, list):
                errors.append("'components' must be a list")

        return errors


# =====================================================================
# Repair plan serialization
# =====================================================================

class RepairPlanSerializer:
    """Serialize and deserialize repair plans to/from JSON."""

    @staticmethod
    def serialize(plan: dict[str, Any]) -> str:
        """Serialize a repair plan to JSON."""
        return json.dumps(plan, indent=2, default=str)

    @staticmethod
    def deserialize(json_str: str) -> dict[str, Any]:
        """Deserialize a repair plan from JSON string."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ParseError("<repair_plan>", str(e), line=e.lineno)

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """Load a repair plan from a JSON file."""
        p = Path(path)
        if not p.exists():
            raise SerializationError(
                f"File not found: {p}",
                code=ErrorCode.SERIALIZATION_FILE_NOT_FOUND,
            )
        with open(p, "r") as f:
            return json.load(f)

    @staticmethod
    def save(plan: dict[str, Any], path: str | Path, indent: int = 2) -> None:
        """Save a repair plan to a JSON file."""
        with open(path, "w") as f:
            json.dump(plan, f, indent=indent, default=str)

    @staticmethod
    def validate(plan: dict[str, Any]) -> list[str]:
        """Validate a repair plan.  Returns list of errors."""
        errors: list[str] = []

        if "actions" not in plan:
            errors.append("Missing 'actions' field")
        elif not isinstance(plan["actions"], list):
            errors.append("'actions' must be a list")
        else:
            for i, action in enumerate(plan["actions"]):
                if not isinstance(action, dict):
                    errors.append(f"actions[{i}]: must be an object")
                    continue
                if "node_id" not in action:
                    errors.append(f"actions[{i}]: missing 'node_id'")
                if "action_type" not in action:
                    errors.append(f"actions[{i}]: missing 'action_type'")

        if "cost_estimate" not in plan:
            errors.append("Missing 'cost_estimate' field")

        return errors


# =====================================================================
# JSON encoder for ARC types
# =====================================================================

class ARCJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles ARC types."""

    def default(self, obj: Any) -> Any:
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, timedelta):
            return obj.total_seconds()
        if hasattr(obj, "value"):  # enums
            return obj.value
        if isinstance(obj, frozenset):
            return sorted(obj)
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


def dumps(obj: Any, indent: int = 2) -> str:
    """JSON-serialize any ARC object."""
    return json.dumps(obj, cls=ARCJSONEncoder, indent=indent)


def loads(s: str) -> Any:
    """Parse a JSON string."""
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ParseError("<string>", str(e), line=e.lineno)


# =====================================================================
# Schema converter: SQL DDL → Schema
# =====================================================================

class SchemaConverter:
    """Convert between various schema representations."""

    @staticmethod
    def from_column_defs(
        columns: list[dict[str, Any]],
        table_name: str = "",
    ) -> "Schema":
        """Create a Schema from a list of column definition dicts.

        Each dict should have at least ``name`` and ``type`` (a SQL type string
        like ``'VARCHAR(255)'``, ``'INT'``, ``'DECIMAL(10,2)'``).
        """
        from arc.types.base import Column, ParameterisedType, Schema

        cols: list[Column] = []
        for i, cdef in enumerate(columns):
            ptype = ParameterisedType.from_string(cdef["type"])
            col = Column(
                name=cdef["name"],
                sql_type=ptype,
                nullable=cdef.get("nullable", True),
                default_expr=cdef.get("default"),
                position=i,
            )
            cols.append(col)

        pk = tuple(columns[0]["name"]) if columns and columns[0].get("primary_key") else ()

        return Schema(
            columns=tuple(cols),
            primary_key=pk,
            table_name=table_name,
        )

    @staticmethod
    def to_column_defs(schema: "Schema") -> list[dict[str, Any]]:
        """Convert a Schema to a list of simple column definition dicts."""
        result: list[dict[str, Any]] = []
        for col in schema.columns:
            d: dict[str, Any] = {
                "name": col.name,
                "type": str(col.sql_type),
                "nullable": col.nullable,
            }
            if col.default_expr:
                d["default"] = col.default_expr
            result.append(d)
        return result

    @staticmethod
    def from_dict_rows(
        rows: list[dict[str, Any]],
        table_name: str = "",
    ) -> "Schema":
        """Infer a schema from a list of dict-rows by inspecting values.

        This is a best-effort inference and may not capture all type
        nuances. Useful for quick prototyping.
        """
        from arc.types.base import Column, ParameterisedType, SQLType, Schema

        if not rows:
            return Schema.empty()

        # Collect all column names across all rows
        col_names: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    col_names.append(key)
                    seen.add(key)

        columns: list[Column] = []
        for i, name in enumerate(col_names):
            # Infer type from first non-None value
            inferred_type = SQLType.TEXT
            nullable = False
            for row in rows:
                val = row.get(name)
                if val is None:
                    nullable = True
                    continue
                if isinstance(val, bool):
                    inferred_type = SQLType.BOOLEAN
                    break
                if isinstance(val, int):
                    inferred_type = SQLType.BIGINT
                    break
                if isinstance(val, float):
                    inferred_type = SQLType.DOUBLE
                    break
                if isinstance(val, str):
                    inferred_type = SQLType.TEXT
                    break
                if isinstance(val, list):
                    inferred_type = SQLType.JSON
                    break
                if isinstance(val, dict):
                    inferred_type = SQLType.JSONB
                    break

            columns.append(Column(
                name=name,
                sql_type=ParameterisedType.simple(inferred_type),
                nullable=nullable,
                position=i,
            ))

        return Schema(columns=tuple(columns), table_name=table_name)


# =====================================================================
# Batch operations
# =====================================================================

class BatchSerializer:
    """Serialize and deserialize batches of pipeline specs or deltas."""

    @staticmethod
    def save_batch(
        items: list[dict[str, Any]],
        path: str | Path,
        indent: int = 2,
    ) -> None:
        """Save a batch of JSON objects to a file (one JSON array)."""
        with open(path, "w") as f:
            json.dump(items, f, indent=indent, default=str)

    @staticmethod
    def load_batch(path: str | Path) -> list[dict[str, Any]]:
        """Load a batch of JSON objects from a file."""
        p = Path(path)
        if not p.exists():
            raise SerializationError(
                f"File not found: {p}",
                code=ErrorCode.SERIALIZATION_FILE_NOT_FOUND,
            )
        with open(p, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SerializationError(
                f"Expected JSON array, got {type(data).__name__}",
                code=ErrorCode.SERIALIZATION_PARSE_ERROR,
            )
        return data

    @staticmethod
    def save_ndjson(
        items: list[dict[str, Any]],
        path: str | Path,
    ) -> None:
        """Save as newline-delimited JSON (one object per line)."""
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item, default=str) + "\n")

    @staticmethod
    def load_ndjson(path: str | Path) -> list[dict[str, Any]]:
        """Load from newline-delimited JSON."""
        p = Path(path)
        if not p.exists():
            raise SerializationError(
                f"File not found: {p}",
                code=ErrorCode.SERIALIZATION_FILE_NOT_FOUND,
            )
        items: list[dict[str, Any]] = []
        with open(p, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ParseError(str(p), str(e), line=line_num)
        return items

"""
JSON Schema definitions for ARC pipeline specifications.

Provides programmatic access to the validation schema for pipeline specs,
delta specs, and repair plan specs.  Includes migration helpers for
moving between spec versions.
"""

from __future__ import annotations

from typing import Any

from arc.io.json_format import CURRENT_SPEC_VERSION, SUPPORTED_SPEC_VERSIONS


# =====================================================================
# JSON Schema for pipeline spec v1.0
# =====================================================================

PIPELINE_SPEC_SCHEMA_V1: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ARC Pipeline Specification",
    "description": "Schema for Algebraic Repair Calculus pipeline definitions",
    "type": "object",
    "required": ["version", "nodes", "edges"],
    "properties": {
        "version": {
            "type": "string",
            "enum": SUPPORTED_SPEC_VERSIONS,
            "description": "Specification version",
        },
        "name": {
            "type": "string",
            "description": "Human-readable pipeline name",
        },
        "metadata": {
            "type": "object",
            "description": "Arbitrary metadata",
        },
        "nodes": {
            "type": "array",
            "items": {"$ref": "#/$defs/pipeline_node"},
            "description": "Pipeline transformation nodes",
        },
        "edges": {
            "type": "array",
            "items": {"$ref": "#/$defs/pipeline_edge"},
            "description": "Directed edges (dependencies)",
        },
    },
    "$defs": {
        "sql_type": {
            "type": "object",
            "required": ["base"],
            "properties": {
                "base": {
                    "type": "string",
                    "description": "Base SQL type name",
                },
                "params": {
                    "type": "object",
                    "properties": {
                        "length": {"type": "integer", "minimum": 0},
                        "precision": {"type": "integer", "minimum": 1},
                        "scale": {"type": "integer", "minimum": 0},
                        "element_type": {"type": "string"},
                    },
                },
            },
        },
        "column": {
            "type": "object",
            "required": ["name", "sql_type"],
            "properties": {
                "name": {"type": "string"},
                "sql_type": {"$ref": "#/$defs/sql_type"},
                "nullable": {"type": "boolean", "default": True},
                "default_expr": {"type": ["string", "null"]},
                "position": {"type": "integer", "minimum": 0},
                "constraints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {"type": "string"},
                            "expression": {"type": "string"},
                            "parameters": {"type": "object"},
                        },
                    },
                },
                "description": {"type": "string"},
            },
        },
        "schema": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/column"},
                },
                "primary_key": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "unique_constraints": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "foreign_keys": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["columns", "ref_table", "ref_columns"],
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "ref_table": {"type": "string"},
                            "ref_columns": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "on_delete": {"type": "string"},
                            "on_update": {"type": "string"},
                        },
                    },
                },
                "check_constraints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["expression"],
                        "properties": {
                            "expression": {"type": "string"},
                            "constraint_name": {"type": "string"},
                        },
                    },
                },
                "schema_name": {"type": "string"},
                "table_name": {"type": "string"},
            },
        },
        "quality_constraint": {
            "type": "object",
            "required": ["constraint_id", "predicate"],
            "properties": {
                "constraint_id": {"type": "string"},
                "predicate": {"type": "string"},
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "error", "critical"],
                },
                "severity_threshold": {"type": "number"},
                "affected_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "metric_name": {"type": "string"},
                "description": {"type": "string"},
                "enabled": {"type": "boolean"},
            },
        },
        "availability_contract": {
            "type": "object",
            "properties": {
                "sla_percentage": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                },
                "max_downtime_seconds": {
                    "type": "number",
                    "minimum": 0,
                },
                "staleness_tolerance_seconds": {
                    "type": "number",
                    "minimum": 0,
                },
                "priority": {"type": "integer"},
                "description": {"type": "string"},
            },
        },
        "cost_estimate": {
            "type": "object",
            "properties": {
                "compute_seconds": {"type": "number", "minimum": 0},
                "memory_bytes": {"type": "integer", "minimum": 0},
                "io_bytes": {"type": "integer", "minimum": 0},
                "row_estimate": {"type": "integer", "minimum": 0},
                "monetary_cost": {"type": "number", "minimum": 0},
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
        },
        "operator_properties": {
            "type": "object",
            "properties": {
                "deterministic": {"type": "boolean"},
                "commutative": {"type": "boolean"},
                "associative": {"type": "boolean"},
                "idempotent": {"type": "boolean"},
                "order_independent": {"type": "boolean"},
                "monotone": {"type": "boolean"},
                "preserves_keys": {"type": "boolean"},
                "may_change_cardinality": {"type": "boolean"},
                "has_side_effects": {"type": "boolean"},
                "requires_full_input": {"type": "boolean"},
            },
        },
        "pipeline_node": {
            "type": "object",
            "required": ["node_id"],
            "properties": {
                "node_id": {"type": "string"},
                "operator": {"type": "string"},
                "query_text": {"type": "string"},
                "input_schema": {"$ref": "#/$defs/schema"},
                "output_schema": {"$ref": "#/$defs/schema"},
                "quality_constraints": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/quality_constraint"},
                },
                "availability_contract": {"$ref": "#/$defs/availability_contract"},
                "cost_estimate": {"$ref": "#/$defs/cost_estimate"},
                "properties": {"$ref": "#/$defs/operator_properties"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "source_file": {"type": "string"},
                        "source_line": {"type": "integer"},
                        "dialect": {"type": "string"},
                    },
                },
            },
        },
        "pipeline_edge": {
            "type": "object",
            "required": ["source", "target"],
            "properties": {
                "source": {"type": "string"},
                "target": {"type": "string"},
                "column_mapping": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "edge_type": {
                    "type": "string",
                    "enum": [
                        "data_flow",
                        "schema_dependency",
                        "quality_dependency",
                        "control_flow",
                        "temporal",
                    ],
                },
                "label": {"type": "string"},
            },
        },
    },
}


# =====================================================================
# Delta spec schema
# =====================================================================

DELTA_SPEC_SCHEMA_V1: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ARC Delta Specification",
    "description": "Schema for perturbation deltas",
    "type": "object",
    "required": ["sort"],
    "properties": {
        "sort": {
            "type": "string",
            "enum": ["schema", "data", "quality", "compound"],
        },
        "target_node": {"type": "string"},
        "description": {"type": "string"},
    },
    "oneOf": [
        {
            "properties": {
                "sort": {"const": "schema"},
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "add_column",
                                    "drop_column",
                                    "rename_column",
                                    "widen_type",
                                    "set_nullable",
                                    "add_constraint",
                                    "drop_constraint",
                                ],
                            },
                        },
                    },
                },
            },
            "required": ["sort", "operations"],
        },
        {
            "properties": {
                "sort": {"const": "data"},
                "changes": {
                    "type": "object",
                    "properties": {
                        "inserts": {"type": "array"},
                        "deletes": {"type": "array"},
                        "updates": {"type": "array"},
                    },
                },
            },
            "required": ["sort", "changes"],
        },
        {
            "properties": {
                "sort": {"const": "quality"},
                "constraint_changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["constraint_id", "change_type"],
                    },
                },
            },
            "required": ["sort", "constraint_changes"],
        },
        {
            "properties": {
                "sort": {"const": "compound"},
                "components": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
            "required": ["sort", "components"],
        },
    ],
}


# =====================================================================
# Repair plan schema
# =====================================================================

REPAIR_PLAN_SCHEMA_V1: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ARC Repair Plan",
    "description": "Schema for repair plans",
    "type": "object",
    "required": ["actions", "cost_estimate"],
    "properties": {
        "version": {"type": "string"},
        "pipeline_name": {"type": "string"},
        "perturbation": {"type": "object"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["node_id", "action_type"],
                "properties": {
                    "node_id": {"type": "string"},
                    "action_type": {
                        "type": "string",
                        "enum": [
                            "recompute",
                            "schema_migrate",
                            "quality_fix",
                            "fallback_source",
                            "skip",
                            "cache_invalidate",
                        ],
                    },
                    "parameters": {"type": "object"},
                    "estimated_cost": {"$ref": "#/$defs/cost_estimate"},
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "cost_estimate": {
            "$ref": "#/$defs/cost_estimate",
        },
        "affected_nodes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "correctness_guarantee": {
            "type": "string",
            "enum": ["exact", "bounded_epsilon", "best_effort"],
        },
    },
    "$defs": {
        "cost_estimate": {
            "type": "object",
            "properties": {
                "compute_seconds": {"type": "number"},
                "memory_bytes": {"type": "integer"},
                "io_bytes": {"type": "integer"},
                "row_estimate": {"type": "integer"},
                "monetary_cost": {"type": "number"},
                "confidence": {"type": "number"},
            },
        },
    },
}


# =====================================================================
# Schema access
# =====================================================================

def get_pipeline_schema(version: str = CURRENT_SPEC_VERSION) -> dict[str, Any]:
    """Return the JSON Schema for a pipeline spec at the given version."""
    if version == "1.0":
        return PIPELINE_SPEC_SCHEMA_V1
    raise ValueError(f"Unknown schema version: {version}")


def get_delta_schema(version: str = CURRENT_SPEC_VERSION) -> dict[str, Any]:
    """Return the JSON Schema for a delta spec."""
    if version == "1.0":
        return DELTA_SPEC_SCHEMA_V1
    raise ValueError(f"Unknown schema version: {version}")


def get_repair_plan_schema(version: str = CURRENT_SPEC_VERSION) -> dict[str, Any]:
    """Return the JSON Schema for a repair plan."""
    if version == "1.0":
        return REPAIR_PLAN_SCHEMA_V1
    raise ValueError(f"Unknown schema version: {version}")


# =====================================================================
# Example specs
# =====================================================================

EXAMPLE_PIPELINE_SPEC: dict[str, Any] = {
    "version": "1.0",
    "name": "example_etl",
    "metadata": {
        "owner": "data-team",
        "description": "Example ETL pipeline for demonstration",
    },
    "nodes": [
        {
            "node_id": "raw_users",
            "operator": "SOURCE",
            "output_schema": {
                "columns": [
                    {"name": "id", "sql_type": {"base": "INT"}, "nullable": False, "position": 0},
                    {"name": "name", "sql_type": {"base": "VARCHAR", "params": {"length": 255}}, "nullable": False, "position": 1},
                    {"name": "email", "sql_type": {"base": "VARCHAR", "params": {"length": 255}}, "nullable": True, "position": 2},
                    {"name": "created_at", "sql_type": {"base": "TIMESTAMP"}, "nullable": False, "position": 3},
                ],
                "primary_key": ["id"],
            },
        },
        {
            "node_id": "clean_users",
            "operator": "FILTER",
            "query_text": "SELECT * FROM raw_users WHERE email IS NOT NULL",
        },
        {
            "node_id": "user_summary",
            "operator": "GROUP_BY",
            "query_text": "SELECT DATE(created_at) AS signup_date, COUNT(*) AS user_count FROM clean_users GROUP BY 1",
            "output_schema": {
                "columns": [
                    {"name": "signup_date", "sql_type": {"base": "DATE"}, "nullable": False, "position": 0},
                    {"name": "user_count", "sql_type": {"base": "BIGINT"}, "nullable": False, "position": 1},
                ],
            },
        },
        {
            "node_id": "output",
            "operator": "SINK",
        },
    ],
    "edges": [
        {"source": "raw_users", "target": "clean_users"},
        {"source": "clean_users", "target": "user_summary"},
        {"source": "user_summary", "target": "output"},
    ],
}

EXAMPLE_DELTA_SPEC: dict[str, Any] = {
    "sort": "schema",
    "target_node": "raw_users",
    "description": "Add phone column to users table",
    "operations": [
        {
            "type": "add_column",
            "column": {
                "name": "phone",
                "sql_type": {"base": "VARCHAR", "params": {"length": 20}},
                "nullable": True,
                "position": 4,
            },
        },
    ],
}

EXAMPLE_REPAIR_PLAN: dict[str, Any] = {
    "version": "1.0",
    "pipeline_name": "example_etl",
    "perturbation": EXAMPLE_DELTA_SPEC,
    "actions": [
        {
            "node_id": "raw_users",
            "action_type": "schema_migrate",
            "parameters": {"add_column": "phone"},
        },
        {
            "node_id": "clean_users",
            "action_type": "recompute",
            "dependencies": ["raw_users"],
        },
        {
            "node_id": "user_summary",
            "action_type": "skip",
            "parameters": {"reason": "New column not used in aggregation"},
        },
    ],
    "cost_estimate": {
        "compute_seconds": 12.5,
        "memory_bytes": 1073741824,
        "io_bytes": 536870912,
        "row_estimate": 1000000,
        "monetary_cost": 0.05,
        "confidence": 0.85,
    },
    "affected_nodes": ["raw_users", "clean_users"],
    "correctness_guarantee": "exact",
}


# =====================================================================
# Version migration
# =====================================================================

class SpecMigrator:
    """Migrate pipeline specs between versions."""

    @staticmethod
    def migrate(
        data: dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> dict[str, Any]:
        """Migrate a spec from one version to another.

        Currently only version 1.0 exists, so this is a no-op placeholder
        for future version transitions.
        """
        if from_version == to_version:
            return data

        if from_version not in SUPPORTED_SPEC_VERSIONS:
            raise ValueError(f"Unknown source version: {from_version}")
        if to_version not in SUPPORTED_SPEC_VERSIONS:
            raise ValueError(f"Unknown target version: {to_version}")

        # Future: implement migration logic between versions
        result = dict(data)
        result["version"] = to_version
        return result

    @staticmethod
    def needs_migration(data: dict[str, Any]) -> bool:
        """Check if a spec needs migration to the current version."""
        version = data.get("version", "")
        return version != CURRENT_SPEC_VERSION and version in SUPPORTED_SPEC_VERSIONS

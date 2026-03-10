"""
dbt project loader for ARC.

Parses a dbt project directory to extract:
- Model SQL files and their {{ ref('...') }} dependencies
- schema.yml column definitions and tests
- Pipeline DAG as an ARC PipelineGraph
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml

from arc.types.base import (
    Column,
    ParameterisedType,
    PipelineEdge,
    PipelineGraph,
    PipelineNode,
    Schema,
    SQLType,
)


_REF_PATTERN = re.compile(r"""\{\{\s*ref\s*\(\s*['"](\w+)['"]\s*\)\s*\}\}""")

# Maps dbt test names and schema.yml type hints to SQLType
_DBT_TYPE_MAP: dict[str, SQLType] = {
    "integer": SQLType.INT,
    "int": SQLType.INT,
    "bigint": SQLType.BIGINT,
    "smallint": SQLType.SMALLINT,
    "float": SQLType.FLOAT,
    "double": SQLType.DOUBLE,
    "numeric": SQLType.DECIMAL,
    "decimal": SQLType.DECIMAL,
    "varchar": SQLType.VARCHAR,
    "text": SQLType.TEXT,
    "string": SQLType.TEXT,
    "boolean": SQLType.BOOLEAN,
    "bool": SQLType.BOOLEAN,
    "date": SQLType.DATE,
    "timestamp": SQLType.TIMESTAMP,
    "timestamptz": SQLType.TIMESTAMPTZ,
    "json": SQLType.JSON,
    "jsonb": SQLType.JSONB,
    "uuid": SQLType.UUID,
}


def _parse_sql_type(type_str: str | None) -> SQLType:
    """Convert a dbt/YAML type hint string to a SQLType, defaulting to TEXT."""
    if not type_str:
        return SQLType.TEXT
    return _DBT_TYPE_MAP.get(type_str.strip().lower(), SQLType.TEXT)


def _extract_refs(sql_text: str) -> list[str]:
    """Extract all {{ ref('model_name') }} references from a SQL template."""
    return _REF_PATTERN.findall(sql_text)


def _infer_columns_from_sql(sql_text: str) -> list[str]:
    """Best-effort column extraction from a dbt SQL model.

    Looks for the final CTE's SELECT list or the outermost SELECT.
    Falls back to empty list if we can't parse.
    """
    cleaned = re.sub(r"\{[%#].*?[%#]\}", "", sql_text)
    cleaned = re.sub(r"\{\{.*?\}\}", "'__ref__'", cleaned)

    select_pattern = re.compile(
        r"select\s+(.*?)\s+from\s+",
        re.IGNORECASE | re.DOTALL,
    )
    matches = select_pattern.findall(cleaned)
    if not matches:
        return []

    last_select = matches[-1]
    if last_select.strip() == "*":
        for m in reversed(matches[:-1]):
            if m.strip() != "*":
                last_select = m
                break
        else:
            return []

    columns: list[str] = []
    parts = last_select.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Handle "expr AS alias"
        as_match = re.search(r"\bas\s+(\w+)\s*$", part, re.IGNORECASE)
        if as_match:
            columns.append(as_match.group(1))
        else:
            # Take last identifier
            ident = re.findall(r"(\w+)", part)
            if ident:
                col = ident[-1]
                # Skip SQL keywords
                if col.upper() not in {
                    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT",
                    "NULL", "TRUE", "FALSE", "AS", "CASE", "WHEN",
                    "THEN", "ELSE", "END", "IN", "IS", "LIKE",
                }:
                    columns.append(col)
    return columns


class DbtModel:
    """Parsed representation of a single dbt model."""

    def __init__(
        self,
        name: str,
        sql_text: str,
        refs: list[str],
        columns: list[dict[str, Any]] | None = None,
        description: str = "",
        materialized: str = "table",
    ) -> None:
        self.name = name
        self.sql_text = sql_text
        self.refs = refs
        self.columns = columns or []
        self.description = description
        self.materialized = materialized

    def __repr__(self) -> str:
        return f"DbtModel({self.name!r}, refs={self.refs}, cols={len(self.columns)})"


class DbtProject:
    """Parsed representation of a dbt project."""

    def __init__(
        self,
        project_name: str,
        models: dict[str, DbtModel],
        seeds: list[str],
    ) -> None:
        self.project_name = project_name
        self.models = models
        self.seeds = seeds

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def seed_count(self) -> int:
        return len(self.seeds)

    def dependency_edges(self) -> list[tuple[str, str]]:
        """Return (source, target) edges from ref dependencies."""
        edges = []
        for model in self.models.values():
            for ref in model.refs:
                edges.append((ref, model.name))
        return edges

    def __repr__(self) -> str:
        return f"DbtProject({self.project_name!r}, models={self.model_count}, seeds={self.seed_count})"


def load_dbt_project(project_dir: str | Path) -> DbtProject:
    """Load and parse a dbt project directory.

    Parameters
    ----------
    project_dir:
        Path to the dbt project root (containing dbt_project.yml).

    Returns
    -------
    DbtProject
        Parsed project with models, refs, and column metadata.
    """
    project_dir = Path(project_dir)

    # Read dbt_project.yml
    project_yml = project_dir / "dbt_project.yml"
    if not project_yml.exists():
        raise FileNotFoundError(f"No dbt_project.yml found in {project_dir}")

    with open(project_yml) as f:
        project_config = yaml.safe_load(f)

    project_name = project_config.get("name", "unknown")
    model_paths = project_config.get("model-paths", ["models"])
    seed_paths = project_config.get("seed-paths", ["seeds"])

    # Discover seeds
    seeds: list[str] = []
    for sp in seed_paths:
        seed_dir = project_dir / sp
        if seed_dir.exists():
            for csv_file in seed_dir.glob("*.csv"):
                seeds.append(csv_file.stem)

    # Load schema.yml files first to get column metadata
    schema_meta: dict[str, dict[str, Any]] = {}
    for mp in model_paths:
        model_dir = project_dir / mp
        if not model_dir.exists():
            continue
        for yml_file in model_dir.rglob("schema.yml"):
            _parse_schema_yml(yml_file, schema_meta)
        for yml_file in model_dir.rglob("*.yml"):
            if yml_file.name != "schema.yml":
                _parse_schema_yml(yml_file, schema_meta)

    # Discover and parse SQL models
    models: dict[str, DbtModel] = {}
    for mp in model_paths:
        model_dir = project_dir / mp
        if not model_dir.exists():
            continue
        for sql_file in model_dir.rglob("*.sql"):
            model_name = sql_file.stem
            sql_text = sql_file.read_text()
            refs = _extract_refs(sql_text)
            meta = schema_meta.get(model_name, {})
            models[model_name] = DbtModel(
                name=model_name,
                sql_text=sql_text,
                refs=refs,
                columns=meta.get("columns", []),
                description=meta.get("description", ""),
            )

    return DbtProject(
        project_name=project_name,
        models=models,
        seeds=seeds,
    )


def _parse_schema_yml(
    yml_path: Path,
    schema_meta: dict[str, dict[str, Any]],
) -> None:
    """Parse a schema.yml and merge column metadata into schema_meta."""
    try:
        with open(yml_path) as f:
            data = yaml.safe_load(f)
    except Exception:
        return

    if not isinstance(data, dict):
        return

    for model_def in data.get("models", []):
        if not isinstance(model_def, dict):
            continue
        name = model_def.get("name", "")
        if not name:
            continue
        cols = []
        for col_def in model_def.get("columns", []):
            if not isinstance(col_def, dict):
                continue
            col_info: dict[str, Any] = {
                "name": col_def.get("name", ""),
                "description": col_def.get("description", ""),
                "tests": col_def.get("tests", []),
            }
            if "data_type" in col_def:
                col_info["data_type"] = col_def["data_type"]
            cols.append(col_info)
        schema_meta[name] = {
            "columns": cols,
            "description": model_def.get("description", ""),
        }


def _has_not_null_test(tests: list) -> bool:
    """Check if a column's tests include not_null."""
    for t in tests:
        if t == "not_null":
            return True
        if isinstance(t, dict) and "not_null" in t:
            return True
    return False


def _has_unique_test(tests: list) -> bool:
    """Check if a column's tests include unique."""
    for t in tests:
        if t == "unique":
            return True
        if isinstance(t, dict) and "unique" in t:
            return True
    return False


def dbt_model_to_schema(model: DbtModel) -> Schema:
    """Convert a DbtModel's column metadata to an ARC Schema.

    Uses schema.yml columns if available, otherwise infers from SQL.
    """
    columns: list[Column] = []

    if model.columns:
        for i, col_def in enumerate(model.columns):
            col_name = col_def.get("name", f"col_{i}")
            data_type = _parse_sql_type(col_def.get("data_type"))
            tests = col_def.get("tests", [])
            nullable = not _has_not_null_test(tests)
            columns.append(Column.quick(
                name=col_name,
                base_type=data_type,
                nullable=nullable,
                position=i,
            ))
    else:
        inferred = _infer_columns_from_sql(model.sql_text)
        for i, col_name in enumerate(inferred):
            columns.append(Column.quick(
                name=col_name,
                base_type=SQLType.TEXT,
                nullable=True,
                position=i,
            ))

    if not columns:
        columns = [Column.quick("_placeholder", SQLType.TEXT, position=0)]

    return Schema(
        columns=tuple(columns),
        table_name=model.name,
    )


def dbt_project_to_pipeline(project: DbtProject) -> PipelineGraph:
    """Convert a parsed DbtProject into an ARC PipelineGraph.

    Seeds become source nodes. Models become transform nodes.
    ref() dependencies become edges.
    """
    from arc.types.base import PipelineNode as PNode, SQLOperator

    graph = PipelineGraph()

    # Add seed source nodes
    for seed_name in project.seeds:
        seed_schema = Schema(
            columns=(Column.quick("id", SQLType.INT, position=0),),
            table_name=seed_name,
        )
        node = PNode(
            node_id=seed_name,
            operator=SQLOperator.CUSTOM,
            is_source=True,
            input_schema=seed_schema,
            output_schema=seed_schema,
            estimated_row_count=100,
        )
        graph.add_node(node)

    # Add model nodes
    for model in project.models.values():
        schema = dbt_model_to_schema(model)
        is_source = len(model.refs) == 0

        node = PNode(
            node_id=model.name,
            operator=SQLOperator.CUSTOM,
            is_source=is_source,
            sql_text=model.sql_text,
            input_schema=schema,
            output_schema=schema,
            estimated_row_count=1000,
        )
        graph.add_node(node)

    # Add edges from ref dependencies
    all_known = set(graph.nodes.keys())
    for model in project.models.values():
        for ref in model.refs:
            if ref not in all_known:
                # Create a placeholder source node for unresolved refs
                placeholder_schema = Schema(
                    columns=(Column.quick("id", SQLType.INT, position=0),),
                    table_name=ref,
                )
                placeholder = PNode(
                    node_id=ref,
                    operator=SQLOperator.CUSTOM,
                    is_source=True,
                    input_schema=placeholder_schema,
                    output_schema=placeholder_schema,
                    estimated_row_count=100,
                )
                graph.add_node(placeholder)
                all_known.add(ref)

            graph.add_edge(PipelineEdge(source=ref, target=model.name))

    return graph

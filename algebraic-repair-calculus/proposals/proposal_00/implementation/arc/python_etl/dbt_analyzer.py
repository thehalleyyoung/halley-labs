"""
dbt Model Analyzer
===================

Analyze dbt (data build tool) projects to extract pipeline structure,
dependencies, materializations, schema tests, and incremental model
patterns. Builds ARC pipeline graphs from dbt project definitions.

Supports:
  - Project-level analysis from dbt_project.yml
  - Model parsing from SQL files with Jinja refs
  - Dependency extraction from ref() and source() calls
  - Schema test extraction from schema.yml
  - Materialization detection (table, view, incremental, ephemeral)
  - Lineage graph construction
  - Repair SQL generation
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    HAS_YAML = False

from arc.graph.pipeline import PipelineEdge, PipelineGraph, PipelineNode
from arc.types.base import CostEstimate, Schema
from arc.types.operators import SQLOperator

logger = logging.getLogger(__name__)


# =====================================================================
# dbt Data Types
# =====================================================================


class MaterializationType(Enum):
    """dbt materialization strategies."""
    TABLE = "table"
    VIEW = "view"
    INCREMENTAL = "incremental"
    EPHEMERAL = "ephemeral"
    SEED = "seed"
    SNAPSHOT = "snapshot"
    UNKNOWN = "unknown"


class TestType(Enum):
    """dbt schema test types."""
    NOT_NULL = "not_null"
    UNIQUE = "unique"
    ACCEPTED_VALUES = "accepted_values"
    RELATIONSHIPS = "relationships"
    CUSTOM = "custom"


@dataclass
class DBTSource:
    """A dbt source definition.

    Attributes
    ----------
    name : str
        Source name.
    schema_name : str
        Database schema.
    database : str
        Database name.
    tables : list[str]
        Table names in this source.
    description : str
        Source description.
    """
    name: str = ""
    schema_name: str = ""
    database: str = ""
    tables: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class DBTTest:
    """A dbt schema test.

    Attributes
    ----------
    test_type : TestType
        Type of test.
    column : str
        Column being tested.
    model : str
        Model the test applies to.
    parameters : dict
        Test-specific parameters.
    severity : str
        Test severity (warn, error).
    """
    test_type: TestType = TestType.NOT_NULL
    column: str = ""
    model: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"

    def to_quality_constraint(self) -> Dict[str, Any]:
        """Convert to an ARC quality constraint dict."""
        constraint: Dict[str, Any] = {
            "column": self.column,
            "severity": self.severity,
        }

        if self.test_type == TestType.NOT_NULL:
            constraint["type"] = "not_null"
        elif self.test_type == TestType.UNIQUE:
            constraint["type"] = "unique"
        elif self.test_type == TestType.ACCEPTED_VALUES:
            constraint["type"] = "enum"
            constraint["values"] = self.parameters.get("values", [])
        elif self.test_type == TestType.RELATIONSHIPS:
            constraint["type"] = "foreign_key"
            constraint["to"] = self.parameters.get("to", "")
            constraint["field"] = self.parameters.get("field", "")
        else:
            constraint["type"] = "custom"
            constraint["expression"] = str(self.parameters)

        return constraint


@dataclass
class DBTModel:
    """A parsed dbt model.

    Attributes
    ----------
    name : str
        Model name.
    path : str
        File path relative to project root.
    sql : str
        Raw SQL content (with Jinja).
    compiled_sql : str
        SQL with Jinja resolved (if available).
    materialization : MaterializationType
        How this model is materialized.
    dependencies : list[str]
        Model names referenced via ref().
    source_dependencies : list[tuple[str, str]]
        Source references via source().
    tests : list[DBTTest]
        Schema tests for this model.
    description : str
        Model description.
    columns : dict[str, dict]
        Column metadata.
    config : dict
        Model config block.
    tags : list[str]
        Model tags.
    is_incremental : bool
        Whether this is an incremental model.
    unique_key : str | None
        Unique key for incremental models.
    """
    name: str = ""
    path: str = ""
    sql: str = ""
    compiled_sql: str = ""
    materialization: MaterializationType = MaterializationType.VIEW
    dependencies: List[str] = field(default_factory=list)
    source_dependencies: List[Tuple[str, str]] = field(default_factory=list)
    tests: List[DBTTest] = field(default_factory=list)
    description: str = ""
    columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    is_incremental: bool = False
    unique_key: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"DBTModel({self.name}, {self.materialization.value}, "
            f"deps={len(self.dependencies)}, tests={len(self.tests)})"
        )


@dataclass
class IncrementalModel:
    """An incremental dbt model with its strategy.

    Attributes
    ----------
    model_name : str
        Name of the model.
    unique_key : str | None
        Unique key for merge.
    strategy : str
        Incremental strategy (merge, delete+insert, insert_overwrite).
    on_schema_change : str
        Behavior on schema changes (ignore, fail, append_new_columns, sync_all_columns).
    filter_expression : str | None
        is_incremental() filter expression.
    """
    model_name: str = ""
    unique_key: Optional[str] = None
    strategy: str = "merge"
    on_schema_change: str = "ignore"
    filter_expression: Optional[str] = None


@dataclass
class DBTProject:
    """A parsed dbt project.

    Attributes
    ----------
    name : str
        Project name.
    version : str
        Project version.
    project_dir : str
        Root directory.
    models : dict[str, DBTModel]
        All models keyed by name.
    sources : dict[str, DBTSource]
        All sources keyed by name.
    tests : list[DBTTest]
        All schema tests.
    materializations : dict[str, MaterializationType]
        Model materializations.
    config : dict
        Project-level config.
    """
    name: str = ""
    version: str = ""
    project_dir: str = ""
    models: Dict[str, DBTModel] = field(default_factory=dict)
    sources: Dict[str, DBTSource] = field(default_factory=dict)
    tests: List[DBTTest] = field(default_factory=list)
    materializations: Dict[str, MaterializationType] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def summary(self) -> str:
        mat_counts = defaultdict(int)
        for m in self.materializations.values():
            mat_counts[m.value] += 1

        lines = [
            f"DBTProject({self.name}):",
            f"  Models:  {self.model_count}",
            f"  Sources: {self.source_count}",
            f"  Tests:   {len(self.tests)}",
            f"  Materializations: {dict(mat_counts)}",
        ]
        return "\n".join(lines)


# =====================================================================
# dbt Analyzer
# =====================================================================


class DBTAnalyzer:
    """Analyze dbt projects to extract pipeline structure.

    Parses dbt project files, SQL models, and schema definitions to
    build ARC-compatible pipeline graphs.

    Parameters
    ----------
    dialect : str
        SQL dialect for parsing (default: duckdb).
    """

    def __init__(self, dialect: str = "duckdb") -> None:
        self._dialect = dialect

    # ── Project Analysis ──────────────────────────────────────────

    def analyze_project(self, project_dir: str) -> DBTProject:
        """Analyze a complete dbt project.

        Parameters
        ----------
        project_dir : str
            Path to the dbt project root.

        Returns
        -------
        DBTProject
        """
        project = DBTProject(project_dir=project_dir)

        dbt_project_path = os.path.join(project_dir, "dbt_project.yml")
        if os.path.exists(dbt_project_path):
            project_config = self._load_yaml(dbt_project_path)
            project.name = project_config.get("name", "")
            project.version = str(project_config.get("version", ""))
            project.config = project_config

        models_dir = os.path.join(project_dir, "models")
        if os.path.isdir(models_dir):
            for root, _, files in os.walk(models_dir):
                for fname in files:
                    if fname.endswith(".sql"):
                        fpath = os.path.join(root, fname)
                        model = self.parse_model(fpath)
                        project.models[model.name] = model
                        project.materializations[model.name] = model.materialization

                    elif fname.endswith(".yml") or fname.endswith(".yaml"):
                        fpath = os.path.join(root, fname)
                        self._parse_schema_file(fpath, project)

        sources_dir = os.path.join(project_dir, "models")
        if os.path.isdir(sources_dir):
            for root, _, files in os.walk(sources_dir):
                for fname in files:
                    if fname.endswith(".yml") or fname.endswith(".yaml"):
                        fpath = os.path.join(root, fname)
                        self._parse_sources_file(fpath, project)

        return project

    def parse_model(self, model_path: str) -> DBTModel:
        """Parse a single dbt model SQL file.

        Parameters
        ----------
        model_path : str
            Path to the .sql file.

        Returns
        -------
        DBTModel
        """
        name = Path(model_path).stem
        sql = ""

        try:
            with open(model_path, "r") as f:
                sql = f.read()
        except (IOError, OSError) as exc:
            logger.warning("Failed to read model %s: %s", model_path, exc)

        model = DBTModel(
            name=name,
            path=model_path,
            sql=sql,
        )

        model.dependencies = self.extract_dependencies(model)
        model.source_dependencies = self._extract_source_refs(sql)
        model.materialization = self._detect_materialization(sql, {})
        model.config = self._extract_config_block(sql)
        model.is_incremental = model.materialization == MaterializationType.INCREMENTAL

        if model.is_incremental:
            model.unique_key = model.config.get("unique_key")

        model.tags = model.config.get("tags", [])

        return model

    def extract_dependencies(self, model: DBTModel) -> List[str]:
        """Extract ref() dependencies from a model.

        Parameters
        ----------
        model : DBTModel
            The model to analyze.

        Returns
        -------
        list[str]
            List of referenced model names.
        """
        return self._extract_refs(model.sql)

    # ── Lineage Graph Construction ────────────────────────────────

    def build_lineage_graph(self, project: DBTProject) -> PipelineGraph:
        """Build an ARC PipelineGraph from a dbt project.

        Parameters
        ----------
        project : DBTProject
            The parsed dbt project.

        Returns
        -------
        PipelineGraph
        """
        graph = PipelineGraph(
            name=project.name or "dbt_pipeline",
            version=project.version or "1.0",
        )

        for source in project.sources.values():
            for table in source.tables:
                source_id = f"source.{source.name}.{table}"
                node = PipelineNode(
                    node_id=source_id,
                    operator=SQLOperator.SOURCE,
                    query_text=f"SELECT * FROM {source.schema_name}.{table}",
                )
                graph.add_node(node)

        for model in project.models.values():
            operator = self._materialization_to_operator(model.materialization)
            node = PipelineNode(
                node_id=f"model.{model.name}",
                operator=operator,
                query_text=model.sql,
            )
            graph.add_node(node)

        for model in project.models.values():
            model_id = f"model.{model.name}"

            for dep in model.dependencies:
                dep_id = f"model.{dep}"
                if graph.has_node(dep_id):
                    graph.add_edge(PipelineEdge(source=dep_id, target=model_id))

            for source_name, table_name in model.source_dependencies:
                source_id = f"source.{source_name}.{table_name}"
                if graph.has_node(source_id):
                    graph.add_edge(PipelineEdge(source=source_id, target=model_id))

        return graph

    # ── Schema Tests ──────────────────────────────────────────────

    def detect_schema_tests(
        self,
        project: DBTProject,
    ) -> List[Dict[str, Any]]:
        """Extract quality constraints from dbt schema tests.

        Parameters
        ----------
        project : DBTProject
            The parsed dbt project.

        Returns
        -------
        list[dict]
            ARC-compatible quality constraint dicts.
        """
        constraints: List[Dict[str, Any]] = []

        for test in project.tests:
            constraint = test.to_quality_constraint()
            constraint["model"] = test.model
            constraints.append(constraint)

        return constraints

    # ── Materialization Analysis ──────────────────────────────────

    def extract_materializations(
        self,
        project: DBTProject,
    ) -> Dict[str, str]:
        """Extract materialization types for all models.

        Parameters
        ----------
        project : DBTProject
            The parsed dbt project.

        Returns
        -------
        dict[str, str]
            Model name to materialization type.
        """
        return {
            name: mat.value
            for name, mat in project.materializations.items()
        }

    # ── Incremental Model Analysis ────────────────────────────────

    def analyze_incremental_models(
        self,
        project: DBTProject,
    ) -> List[IncrementalModel]:
        """Analyze incremental models for their strategies.

        Parameters
        ----------
        project : DBTProject
            The parsed dbt project.

        Returns
        -------
        list[IncrementalModel]
        """
        incremental_models: List[IncrementalModel] = []

        for model in project.models.values():
            if not model.is_incremental:
                continue

            strategy = model.config.get("incremental_strategy", "merge")
            on_schema_change = model.config.get("on_schema_change", "ignore")
            filter_expr = self._extract_incremental_filter(model.sql)

            incremental_models.append(IncrementalModel(
                model_name=model.name,
                unique_key=model.unique_key,
                strategy=strategy,
                on_schema_change=on_schema_change,
                filter_expression=filter_expr,
            ))

        return incremental_models

    # ── Repair SQL Generation ─────────────────────────────────────

    def generate_repair_sql(
        self,
        model: DBTModel,
        delta_spec: Dict[str, Any],
    ) -> str:
        """Generate SQL to repair a dbt model after a schema change.

        Parameters
        ----------
        model : DBTModel
            The dbt model to repair.
        delta_spec : dict
            Schema delta specification with keys:
            - added_columns: list of (name, type, default)
            - dropped_columns: list of column names
            - renamed_columns: list of (old, new)
            - type_changes: list of (column, new_type)

        Returns
        -------
        str
            SQL statements for the repair.
        """
        statements: List[str] = []
        table_name = model.name

        if model.materialization == MaterializationType.VIEW:
            repaired_sql = self._apply_delta_to_sql(model.sql, delta_spec)
            statements.append(
                f"CREATE OR REPLACE VIEW {table_name} AS\n{repaired_sql}"
            )
            return ";\n".join(statements) + ";"

        if model.materialization == MaterializationType.TABLE:
            for col_name, col_type, default in delta_spec.get("added_columns", []):
                stmt = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"
                if default:
                    stmt += f" DEFAULT {default}"
                statements.append(stmt)

            for col in delta_spec.get("dropped_columns", []):
                statements.append(
                    f"ALTER TABLE {table_name} DROP COLUMN {col}"
                )

            for old_name, new_name in delta_spec.get("renamed_columns", []):
                statements.append(
                    f"ALTER TABLE {table_name} RENAME COLUMN {old_name} TO {new_name}"
                )

            for col_name, new_type in delta_spec.get("type_changes", []):
                statements.append(
                    f"ALTER TABLE {table_name} ALTER COLUMN {col_name} TYPE {new_type}"
                )

        if model.is_incremental:
            if delta_spec.get("added_columns"):
                for col_name, col_type, default in delta_spec["added_columns"]:
                    stmt = f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
                    if default:
                        stmt += f" DEFAULT {default}"
                    statements.append(stmt)

        if not statements:
            statements.append(f"-- No repair needed for {table_name}")

        return ";\n".join(statements) + ";"

    # ── Internal: Ref Extraction ──────────────────────────────────

    @staticmethod
    def _extract_refs(sql: str) -> List[str]:
        """Extract ref() model references from SQL."""
        refs: List[str] = []
        for match in re.finditer(r"{{\s*ref\s*\(\s*['\"](\w+)['\"]\s*\)\s*}}", sql):
            refs.append(match.group(1))
        return refs

    @staticmethod
    def _extract_source_refs(sql: str) -> List[Tuple[str, str]]:
        """Extract source() references from SQL."""
        sources: List[Tuple[str, str]] = []
        for match in re.finditer(
            r"{{\s*source\s*\(\s*['\"](\w+)['\"]"
            r"\s*,\s*['\"](\w+)['\"]\s*\)\s*}}",
            sql,
        ):
            sources.append((match.group(1), match.group(2)))
        return sources

    @staticmethod
    def _extract_config_block(sql: str) -> Dict[str, Any]:
        """Extract dbt config() block from SQL."""
        config: Dict[str, Any] = {}

        match = re.search(
            r"{{\s*config\s*\((.*?)\)\s*}}",
            sql,
            re.DOTALL,
        )
        if not match:
            return config

        config_str = match.group(1)

        for kv_match in re.finditer(
            r"(\w+)\s*=\s*['\"]?(\w+)['\"]?",
            config_str,
        ):
            key = kv_match.group(1)
            value = kv_match.group(2)

            if value.lower() in ("true", "false"):
                config[key] = value.lower() == "true"
            else:
                config[key] = value

        return config

    @staticmethod
    def _detect_materialization(
        sql: str,
        project_config: Dict[str, Any],
    ) -> MaterializationType:
        """Detect materialization type from SQL config block."""
        match = re.search(
            r"materialized\s*=\s*['\"]?(\w+)['\"]?",
            sql,
            re.IGNORECASE,
        )
        if match:
            mat_str = match.group(1).lower()
            try:
                return MaterializationType(mat_str)
            except ValueError:
                pass

        if "is_incremental()" in sql:
            return MaterializationType.INCREMENTAL

        return MaterializationType.VIEW

    @staticmethod
    def _extract_incremental_filter(sql: str) -> Optional[str]:
        """Extract the incremental filter expression."""
        match = re.search(
            r"{%\s*if\s+is_incremental\(\)\s*%}(.*?){%\s*endif\s*%}",
            sql,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _materialization_to_operator(
        materialization: MaterializationType,
    ) -> SQLOperator:
        """Map dbt materialization to ARC SQLOperator."""
        return {
            MaterializationType.TABLE: SQLOperator.SINK,
            MaterializationType.VIEW: SQLOperator.TRANSFORM,
            MaterializationType.INCREMENTAL: SQLOperator.TRANSFORM,
            MaterializationType.EPHEMERAL: SQLOperator.TRANSFORM,
            MaterializationType.SEED: SQLOperator.SOURCE,
            MaterializationType.SNAPSHOT: SQLOperator.SINK,
        }.get(materialization, SQLOperator.TRANSFORM)

    # ── Internal: YAML Parsing ────────────────────────────────────

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        """Load a YAML file."""
        if not HAS_YAML:
            logger.warning("PyYAML not available, cannot parse %s", path)
            return {}

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return data or {}
        except Exception as exc:
            logger.warning("Failed to parse YAML %s: %s", path, exc)
            return {}

    def _parse_schema_file(
        self,
        path: str,
        project: DBTProject,
    ) -> None:
        """Parse a dbt schema.yml file for tests and columns."""
        data = self._load_yaml(path)
        if not data:
            return

        models = data.get("models", [])
        if not isinstance(models, list):
            return

        for model_def in models:
            if not isinstance(model_def, dict):
                continue

            model_name = model_def.get("name", "")
            if not model_name:
                continue

            model = project.models.get(model_name)
            if model:
                model.description = model_def.get("description", "")

            columns = model_def.get("columns", [])
            if isinstance(columns, list):
                for col_def in columns:
                    if not isinstance(col_def, dict):
                        continue

                    col_name = col_def.get("name", "")
                    if not col_name:
                        continue

                    if model:
                        model.columns[col_name] = col_def

                    tests = col_def.get("tests", [])
                    if isinstance(tests, list):
                        for test in tests:
                            dbt_test = self._parse_test(
                                test, col_name, model_name
                            )
                            if dbt_test:
                                project.tests.append(dbt_test)
                                if model:
                                    model.tests.append(dbt_test)

    def _parse_sources_file(
        self,
        path: str,
        project: DBTProject,
    ) -> None:
        """Parse a dbt sources YAML file."""
        data = self._load_yaml(path)
        if not data:
            return

        sources = data.get("sources", [])
        if not isinstance(sources, list):
            return

        for source_def in sources:
            if not isinstance(source_def, dict):
                continue

            name = source_def.get("name", "")
            if not name:
                continue

            tables = []
            for table_def in source_def.get("tables", []):
                if isinstance(table_def, dict):
                    tables.append(table_def.get("name", ""))
                elif isinstance(table_def, str):
                    tables.append(table_def)

            source = DBTSource(
                name=name,
                schema_name=source_def.get("schema", ""),
                database=source_def.get("database", ""),
                tables=tables,
                description=source_def.get("description", ""),
            )
            project.sources[name] = source

    @staticmethod
    def _parse_test(
        test_def: Any,
        column: str,
        model: str,
    ) -> Optional[DBTTest]:
        """Parse a single test definition."""
        if isinstance(test_def, str):
            test_name = test_def.lower()
            test_type_map = {
                "not_null": TestType.NOT_NULL,
                "unique": TestType.UNIQUE,
            }
            if test_name in test_type_map:
                return DBTTest(
                    test_type=test_type_map[test_name],
                    column=column,
                    model=model,
                )
            return DBTTest(
                test_type=TestType.CUSTOM,
                column=column,
                model=model,
                parameters={"name": test_name},
            )

        if isinstance(test_def, dict):
            for test_name, params in test_def.items():
                test_name_lower = test_name.lower()

                if test_name_lower == "accepted_values":
                    return DBTTest(
                        test_type=TestType.ACCEPTED_VALUES,
                        column=column,
                        model=model,
                        parameters=params if isinstance(params, dict) else {"values": params},
                    )

                if test_name_lower == "relationships":
                    return DBTTest(
                        test_type=TestType.RELATIONSHIPS,
                        column=column,
                        model=model,
                        parameters=params if isinstance(params, dict) else {},
                    )

                return DBTTest(
                    test_type=TestType.CUSTOM,
                    column=column,
                    model=model,
                    parameters=params if isinstance(params, dict) else {"value": params},
                )

        return None

    @staticmethod
    def _apply_delta_to_sql(
        sql: str,
        delta_spec: Dict[str, Any],
    ) -> str:
        """Apply schema delta to a dbt model's SQL."""
        result = sql

        for old_name, new_name in delta_spec.get("renamed_columns", []):
            result = re.sub(
                r"\b" + re.escape(old_name) + r"\b",
                new_name,
                result,
            )

        for col in delta_spec.get("dropped_columns", []):
            result = re.sub(
                r",?\s*\b" + re.escape(col) + r"\b\s*,?",
                "",
                result,
            )

        return result


# =====================================================================
# Convenience Functions
# =====================================================================


def analyze_dbt_project(project_dir: str) -> DBTProject:
    """Convenience: analyze a dbt project."""
    analyzer = DBTAnalyzer()
    return analyzer.analyze_project(project_dir)


def build_dbt_lineage(project_dir: str) -> PipelineGraph:
    """Convenience: build a lineage graph from a dbt project."""
    analyzer = DBTAnalyzer()
    project = analyzer.analyze_project(project_dir)
    return analyzer.build_lineage_graph(project)


def extract_dbt_tests(project_dir: str) -> List[Dict[str, Any]]:
    """Convenience: extract quality constraints from a dbt project."""
    analyzer = DBTAnalyzer()
    project = analyzer.analyze_project(project_dir)
    return analyzer.detect_schema_tests(project)

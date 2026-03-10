"""Shared pytest fixtures for the ARC test suite."""
import pytest
import os
import tempfile
from typing import Any

# ─────────────────────────────────────────────────────────────────────
# Graceful imports — every subsystem is optional
# ─────────────────────────────────────────────────────────────────────

try:
    from arc.types.base import (
        SQLType, ParameterisedType, TypeParameters, Column, Schema,
        QualityConstraint, AvailabilityContract, CostEstimate, NodeMetadata,
        EdgeType, Severity, ConstraintType, ColumnConstraint, WideningResult,
        RepairPlan, RepairAction, ActionType, CostBreakdown,
        ValidationResult, ExecutionResult, ActionResult, TableStats,
        CheckpointInfo, ResourceSpec, ExecutionSchedule,
        ForeignKey, CheckConstraint,
    )
    from arc.types.operators import SQLOperator as OpSQLOperator
    HAS_TYPES = True
except ImportError:
    HAS_TYPES = False

try:
    from arc.graph.pipeline import PipelineNode, PipelineEdge, PipelineGraph
    from arc.graph.builder import PipelineBuilder
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False

try:
    from arc.algebra.schema_delta import (
        SchemaDelta, AddColumn, DropColumn, RenameColumn, ChangeType,
        AddConstraint, DropConstraint, ColumnDef, Schema as AlgSchema,
        ConstraintDef, SQLType as AlgSQLType,
    )
    HAS_SCHEMA_DELTA = True
except ImportError:
    HAS_SCHEMA_DELTA = False

try:
    from arc.algebra.data_delta import (
        DataDelta, TypedTuple, MultiSet, InsertOp, DeleteOp, UpdateOp,
    )
    HAS_DATA_DELTA = True
except ImportError:
    HAS_DATA_DELTA = False

try:
    from arc.algebra.quality_delta import (
        QualityDelta, QualityViolation, QualityImprovement,
        ViolationType, SeverityLevel, ConstraintAdded, ConstraintRemoved,
        DistributionShift, QualityState, ConstraintStatus, DistributionSummary,
        ConstraintType as QConstraintType,
    )
    HAS_QUALITY_DELTA = True
except ImportError:
    HAS_QUALITY_DELTA = False

try:
    from arc.algebra.composition import CompoundPerturbation, PipelineState
    HAS_COMPOSITION = True
except ImportError:
    HAS_COMPOSITION = False

try:
    from arc.algebra.interaction import PhiHomomorphism, PsiHomomorphism
    HAS_INTERACTION = True
except ImportError:
    HAS_INTERACTION = False

try:
    from arc.algebra.push import (
        OperatorContext, SelectPush, JoinPush, GroupByPush, FilterPush,
        UnionPush, WindowPush, CTEPush, SetOpPush,
        push_schema_delta, push_data_delta, push_quality_delta,
        get_push_operator, supported_operators,
        ColumnRef as PushColumnRef, AggregateSpec as PushAggSpec,
        JoinCondition as PushJoinCond, AggregateFunction,
    )
    HAS_PUSH = True
except ImportError:
    HAS_PUSH = False

try:
    from arc.sql.parser import SQLParser, ParsedQuery
    HAS_SQL_PARSER = True
except ImportError:
    HAS_SQL_PARSER = False

try:
    from arc.sql.lineage import LineageAnalyzer, ColumnLineage
    HAS_LINEAGE = True
except ImportError:
    HAS_LINEAGE = False

try:
    from arc.sql.fragment import FragmentChecker, check_fragment_f
    HAS_FRAGMENT = True
except ImportError:
    HAS_FRAGMENT = False

try:
    from arc.planner.cost import CostModel, CostFactors
    from arc.planner.dp import DPRepairPlanner
    from arc.planner.lp import LPRepairPlanner
    from arc.planner.optimizer import PlanOptimizer
    HAS_PLANNER = True
except ImportError:
    HAS_PLANNER = False

try:
    from arc.execution.engine import ExecutionEngine
    from arc.execution.checkpoint import CheckpointManager
    from arc.execution.validation import RepairValidator
    from arc.execution.scheduler import ExecutionScheduler
    HAS_EXECUTION = True
except ImportError:
    HAS_EXECUTION = False

try:
    from arc.quality.monitor import QualityMonitor
    from arc.quality.distribution import DistributionAnalyzer
    from arc.quality.constraints import ConstraintEngine
    from arc.quality.profiler import DataProfiler
    HAS_QUALITY = True
except ImportError:
    HAS_QUALITY = False

try:
    from arc.io.json_format import PipelineSpec, DeltaSerializer, RepairPlanSerializer
    from arc.io.yaml_format import YAMLPipelineSpec
    from arc.io.schema import PIPELINE_SPEC_SCHEMA_V1
    HAS_IO = True
except ImportError:
    HAS_IO = False

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ─────────────────────────────────────────────────────────────────────
# Helper: make a quick Column
# ─────────────────────────────────────────────────────────────────────

def _col(name: str, base: "SQLType" = None, nullable: bool = True, pos: int = 0) -> "Column":
    """Build a Column with minimal ceremony."""
    if base is None:
        base = SQLType.INT
    return Column.quick(name, base, nullable=nullable, position=pos)


def _pt(base: "SQLType") -> "ParameterisedType":
    """Shorthand for ParameterisedType.simple(base)."""
    return ParameterisedType.simple(base)


# ─────────────────────────────────────────────────────────────────────
# Schema fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_schema():
    """Simple 3-column schema: id INT, name VARCHAR, active BOOLEAN."""
    if not HAS_TYPES:
        pytest.skip("arc.types not available")
    return Schema(columns=(
        _col("id", SQLType.INT, nullable=False, pos=0),
        _col("name", SQLType.VARCHAR, pos=1),
        _col("active", SQLType.BOOLEAN, pos=2),
    ))


@pytest.fixture
def complex_schema():
    """A wider schema with 8 columns spanning many SQL types."""
    if not HAS_TYPES:
        pytest.skip("arc.types not available")
    return Schema(columns=(
        _col("id", SQLType.BIGINT, nullable=False, pos=0),
        _col("user_name", SQLType.VARCHAR, pos=1),
        _col("email", SQLType.TEXT, pos=2),
        _col("age", SQLType.INT, pos=3),
        _col("salary", SQLType.DECIMAL, pos=4),
        _col("created_at", SQLType.TIMESTAMP, pos=5),
        _col("is_active", SQLType.BOOLEAN, pos=6),
        _col("score", SQLType.DOUBLE, pos=7),
    ))


@pytest.fixture
def nested_schema():
    """Schema with JSON and ARRAY types."""
    if not HAS_TYPES:
        pytest.skip("arc.types not available")
    return Schema(columns=(
        _col("id", SQLType.INT, nullable=False, pos=0),
        _col("metadata", SQLType.JSONB, pos=1),
        _col("tags", SQLType.ARRAY, pos=2),
        _col("config", SQLType.JSON, pos=3),
    ))


@pytest.fixture
def empty_schema():
    """Empty schema with no columns."""
    if not HAS_TYPES:
        pytest.skip("arc.types not available")
    return Schema(columns=())


@pytest.fixture
def orders_schema():
    """Schema for orders table."""
    if not HAS_TYPES:
        pytest.skip("arc.types not available")
    return Schema(columns=(
        _col("order_id", SQLType.INT, nullable=False, pos=0),
        _col("customer_id", SQLType.INT, nullable=False, pos=1),
        _col("amount", SQLType.DECIMAL, pos=2),
        _col("status", SQLType.VARCHAR, pos=3),
        _col("order_date", SQLType.DATE, pos=4),
    ))


@pytest.fixture
def customers_schema():
    """Schema for customers table."""
    if not HAS_TYPES:
        pytest.skip("arc.types not available")
    return Schema(columns=(
        _col("customer_id", SQLType.INT, nullable=False, pos=0),
        _col("name", SQLType.VARCHAR, pos=1),
        _col("email", SQLType.TEXT, pos=2),
        _col("country", SQLType.VARCHAR, pos=3),
    ))


# ─────────────────────────────────────────────────────────────────────
# Algebra Schema fixtures (for arc.algebra.schema_delta)
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def alg_schema():
    """Algebra-level Schema for schema_delta tests."""
    if not HAS_SCHEMA_DELTA:
        pytest.skip("arc.algebra.schema_delta not available")
    return AlgSchema(
        name="test_table",
        columns={
            "id": ColumnDef(name="id", sql_type=AlgSQLType.INTEGER, nullable=False, position=0),
            "name": ColumnDef(name="name", sql_type=AlgSQLType.VARCHAR, nullable=True, position=1),
            "active": ColumnDef(name="active", sql_type=AlgSQLType.BOOLEAN, nullable=True, position=2),
        },
        constraints={},
    )


@pytest.fixture
def alg_schema_wide():
    """Wider algebra schema for more complex delta tests."""
    if not HAS_SCHEMA_DELTA:
        pytest.skip("arc.algebra.schema_delta not available")
    return AlgSchema(
        name="wide_table",
        columns={
            "id": ColumnDef(name="id", sql_type=AlgSQLType.BIGINT, nullable=False, position=0),
            "user_name": ColumnDef(name="user_name", sql_type=AlgSQLType.VARCHAR, nullable=True, position=1),
            "email": ColumnDef(name="email", sql_type=AlgSQLType.TEXT, nullable=True, position=2),
            "age": ColumnDef(name="age", sql_type=AlgSQLType.INTEGER, nullable=True, position=3),
            "salary": ColumnDef(name="salary", sql_type=AlgSQLType.DECIMAL, nullable=True, position=4),
            "created_at": ColumnDef(name="created_at", sql_type=AlgSQLType.TIMESTAMP, nullable=True, position=5),
        },
        constraints={},
    )


# ─────────────────────────────────────────────────────────────────────
# Data fixtures (for arc.algebra.data_delta)
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_tuples():
    """A list of 5 sample TypedTuples."""
    if not HAS_DATA_DELTA:
        pytest.skip("arc.algebra.data_delta not available")
    return [
        TypedTuple.from_dict({"id": 1, "name": "Alice", "active": True}),
        TypedTuple.from_dict({"id": 2, "name": "Bob", "active": True}),
        TypedTuple.from_dict({"id": 3, "name": "Charlie", "active": False}),
        TypedTuple.from_dict({"id": 4, "name": "Diana", "active": True}),
        TypedTuple.from_dict({"id": 5, "name": "Eve", "active": False}),
    ]


@pytest.fixture
def sample_multiset(sample_tuples):
    """MultiSet with 5 elements."""
    return MultiSet.from_tuples(sample_tuples)


@pytest.fixture
def empty_multiset():
    """Empty MultiSet."""
    if not HAS_DATA_DELTA:
        pytest.skip("arc.algebra.data_delta not available")
    return MultiSet.empty()


@pytest.fixture
def duplicate_multiset():
    """MultiSet with duplicate elements."""
    if not HAS_DATA_DELTA:
        pytest.skip("arc.algebra.data_delta not available")
    t1 = TypedTuple.from_dict({"id": 1, "val": "a"})
    t2 = TypedTuple.from_dict({"id": 2, "val": "b"})
    ms = MultiSet.empty()
    ms = ms.add(t1)
    ms = ms.add(t1)  # duplicate
    ms = ms.add(t2)
    return ms


# ─────────────────────────────────────────────────────────────────────
# Quality fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_quality_constraints():
    """A list of quality constraints for testing."""
    if not HAS_TYPES:
        pytest.skip("arc.types not available")
    return [
        QualityConstraint.not_null("nn_id", "id"),
        QualityConstraint.range_check("range_age", "age", min_val=0, max_val=150),
        QualityConstraint.uniqueness("uniq_email", "email"),
    ]


@pytest.fixture
def quality_state():
    """A base QualityState for testing."""
    if not HAS_QUALITY_DELTA:
        pytest.skip("arc.algebra.quality_delta not available")
    return QualityState(
        active_violations={},
        constraint_statuses={
            "nn_id": ConstraintStatus(
                constraint_id="nn_id",
                constraint_type=QConstraintType.NOT_NULL,
                is_active=True,
                is_satisfied=True,
                violation_count=0,
                columns=("id",),
            ),
        },
        quality_scores={"completeness": 1.0, "accuracy": 0.95},
        overall_score=0.975,
        column_distributions={},
    )


# ─────────────────────────────────────────────────────────────────────
# Pipeline graph fixtures
# ─────────────────────────────────────────────────────────────────────

def _make_node(node_id: str, operator=None, query: str = "", schema=None) -> "PipelineNode":
    """Helper to construct a PipelineNode with minimal boilerplate."""
    kw: dict[str, Any] = {"node_id": node_id}
    if operator is not None:
        kw["operator"] = operator
    if query:
        kw["query_text"] = query
    if schema is not None:
        kw["output_schema"] = schema
    return PipelineNode(**kw)


@pytest.fixture
def linear_pipeline(simple_schema):
    """Linear 3-node pipeline: source → transform → sink."""
    if not HAS_GRAPH or not HAS_TYPES:
        pytest.skip("arc.graph or arc.types not available")
    from arc.types.operators import SQLOperator as BaseSQLOp
    g = PipelineGraph(name="linear")
    g.add_node(_make_node("source", BaseSQLOp.SOURCE, schema=simple_schema))
    g.add_node(_make_node("transform", BaseSQLOp.SELECT, query="SELECT id, name FROM source", schema=simple_schema))
    g.add_node(_make_node("sink", BaseSQLOp.SINK, schema=simple_schema))
    g.add_edge(PipelineEdge(source="source", target="transform"))
    g.add_edge(PipelineEdge(source="transform", target="sink"))
    return g


@pytest.fixture
def diamond_pipeline(simple_schema):
    """Diamond DAG: source → left, right → merge → sink."""
    if not HAS_GRAPH or not HAS_TYPES:
        pytest.skip("arc.graph or arc.types not available")
    from arc.types.operators import SQLOperator as BaseSQLOp
    g = PipelineGraph(name="diamond")
    g.add_node(_make_node("source", BaseSQLOp.SOURCE, schema=simple_schema))
    g.add_node(_make_node("left", BaseSQLOp.FILTER, query="SELECT * FROM source WHERE active", schema=simple_schema))
    g.add_node(_make_node("right", BaseSQLOp.SELECT, query="SELECT id, name FROM source", schema=simple_schema))
    g.add_node(_make_node("merge", BaseSQLOp.JOIN, query="SELECT * FROM left JOIN right ON left.id = right.id", schema=simple_schema))
    g.add_node(_make_node("sink", BaseSQLOp.SINK, schema=simple_schema))
    g.add_edge(PipelineEdge(source="source", target="left"))
    g.add_edge(PipelineEdge(source="source", target="right"))
    g.add_edge(PipelineEdge(source="left", target="merge"))
    g.add_edge(PipelineEdge(source="right", target="merge"))
    g.add_edge(PipelineEdge(source="merge", target="sink"))
    return g


@pytest.fixture
def fan_out_pipeline(simple_schema):
    """Fan-out: source → a, b, c (3 independent sinks)."""
    if not HAS_GRAPH or not HAS_TYPES:
        pytest.skip("arc.graph or arc.types not available")
    from arc.types.operators import SQLOperator as BaseSQLOp
    g = PipelineGraph(name="fan_out")
    g.add_node(_make_node("source", BaseSQLOp.SOURCE, schema=simple_schema))
    for suffix in ("a", "b", "c"):
        g.add_node(_make_node(f"sink_{suffix}", BaseSQLOp.SINK, schema=simple_schema))
        g.add_edge(PipelineEdge(source="source", target=f"sink_{suffix}"))
    return g


@pytest.fixture
def fan_in_pipeline(simple_schema):
    """Fan-in: a, b, c → merge → sink."""
    if not HAS_GRAPH or not HAS_TYPES:
        pytest.skip("arc.graph or arc.types not available")
    from arc.types.operators import SQLOperator as BaseSQLOp
    g = PipelineGraph(name="fan_in")
    for suffix in ("a", "b", "c"):
        g.add_node(_make_node(f"source_{suffix}", BaseSQLOp.SOURCE, schema=simple_schema))
    g.add_node(_make_node("merge", BaseSQLOp.UNION, schema=simple_schema))
    g.add_node(_make_node("sink", BaseSQLOp.SINK, schema=simple_schema))
    for suffix in ("a", "b", "c"):
        g.add_edge(PipelineEdge(source=f"source_{suffix}", target="merge"))
    g.add_edge(PipelineEdge(source="merge", target="sink"))
    return g


@pytest.fixture
def complex_dag(simple_schema):
    """Complex 10-node DAG for stress tests."""
    if not HAS_GRAPH or not HAS_TYPES:
        pytest.skip("arc.graph or arc.types not available")
    from arc.types.operators import SQLOperator as BaseSQLOp
    g = PipelineGraph(name="complex")
    nodes = [
        ("src1", BaseSQLOp.SOURCE), ("src2", BaseSQLOp.SOURCE),
        ("filter1", BaseSQLOp.FILTER), ("filter2", BaseSQLOp.FILTER),
        ("join1", BaseSQLOp.JOIN), ("agg1", BaseSQLOp.GROUP_BY),
        ("select1", BaseSQLOp.SELECT), ("join2", BaseSQLOp.JOIN),
        ("final", BaseSQLOp.SELECT), ("sink", BaseSQLOp.SINK),
    ]
    for nid, op in nodes:
        g.add_node(_make_node(nid, op, schema=simple_schema))
    edges = [
        ("src1", "filter1"), ("src2", "filter2"),
        ("filter1", "join1"), ("filter2", "join1"),
        ("join1", "agg1"), ("join1", "select1"),
        ("agg1", "join2"), ("select1", "join2"),
        ("join2", "final"), ("final", "sink"),
    ]
    for s, t in edges:
        g.add_edge(PipelineEdge(source=s, target=t))
    return g


# ─────────────────────────────────────────────────────────────────────
# Delta fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def add_column_delta():
    """SchemaDelta that adds a single column."""
    if not HAS_SCHEMA_DELTA:
        pytest.skip("arc.algebra.schema_delta not available")
    return SchemaDelta.from_operations([
        AddColumn(name="email", sql_type=AlgSQLType.VARCHAR, nullable=True, position=-1),
    ])


@pytest.fixture
def drop_column_delta():
    """SchemaDelta that drops a column."""
    if not HAS_SCHEMA_DELTA:
        pytest.skip("arc.algebra.schema_delta not available")
    return SchemaDelta.from_operations([
        DropColumn(name="active"),
    ])


@pytest.fixture
def rename_column_delta():
    """SchemaDelta that renames a column."""
    if not HAS_SCHEMA_DELTA:
        pytest.skip("arc.algebra.schema_delta not available")
    return SchemaDelta.from_operations([
        RenameColumn(old_name="name", new_name="full_name"),
    ])


@pytest.fixture
def change_type_delta():
    """SchemaDelta that changes a column type."""
    if not HAS_SCHEMA_DELTA:
        pytest.skip("arc.algebra.schema_delta not available")
    return SchemaDelta.from_operations([
        ChangeType(name="id", old_type=AlgSQLType.INTEGER, new_type=AlgSQLType.BIGINT),
    ])


@pytest.fixture
def composite_schema_delta():
    """SchemaDelta with multiple operations composed together."""
    if not HAS_SCHEMA_DELTA:
        pytest.skip("arc.algebra.schema_delta not available")
    d1 = SchemaDelta.from_operations([
        AddColumn(name="email", sql_type=AlgSQLType.VARCHAR, nullable=True, position=-1),
    ])
    d2 = SchemaDelta.from_operations([
        RenameColumn(old_name="name", new_name="full_name"),
    ])
    return d1.compose(d2)


@pytest.fixture
def identity_schema_delta():
    """Identity SchemaDelta (no-op)."""
    if not HAS_SCHEMA_DELTA:
        pytest.skip("arc.algebra.schema_delta not available")
    return SchemaDelta.identity()


@pytest.fixture
def insert_delta(sample_tuples):
    """DataDelta that inserts 3 tuples."""
    return DataDelta.insert(MultiSet.from_tuples(sample_tuples[:3]))


@pytest.fixture
def delete_delta(sample_tuples):
    """DataDelta that deletes 2 tuples."""
    return DataDelta.delete(MultiSet.from_tuples(sample_tuples[:2]))


@pytest.fixture
def update_delta(sample_tuples):
    """DataDelta that updates a tuple (delete old + insert new)."""
    if not HAS_DATA_DELTA:
        pytest.skip("arc.algebra.data_delta not available")
    old = sample_tuples[0]
    new = TypedTuple.from_dict({"id": 1, "name": "Alicia", "active": True})
    d_del = DataDelta.delete(MultiSet.from_tuples([old]))
    d_ins = DataDelta.insert(MultiSet.from_tuples([new]))
    return d_del.compose(d_ins)


@pytest.fixture
def zero_data_delta():
    """Zero DataDelta (no-op)."""
    if not HAS_DATA_DELTA:
        pytest.skip("arc.algebra.data_delta not available")
    return DataDelta.zero()


@pytest.fixture
def quality_violation_delta():
    """QualityDelta with a single violation."""
    if not HAS_QUALITY_DELTA:
        pytest.skip("arc.algebra.quality_delta not available")
    return QualityDelta.from_operation(
        QualityViolation(
            constraint_id="nn_id",
            severity=SeverityLevel.ERROR,
            affected_tuples=5,
            violation_type=ViolationType.NULL_IN_NON_NULL,
            columns=("id",),
            message="Null found in non-null column id",
        )
    )


@pytest.fixture
def quality_improvement_delta():
    """QualityDelta with a single improvement."""
    if not HAS_QUALITY_DELTA:
        pytest.skip("arc.algebra.quality_delta not available")
    return QualityDelta.from_operation(
        QualityImprovement(
            constraint_id="nn_id",
            severity=SeverityLevel.INFO,
            resolved_tuples=5,
            message="Null values repaired in column id",
        )
    )


@pytest.fixture
def compound_perturbation_identity():
    """Identity compound perturbation."""
    if not HAS_COMPOSITION:
        pytest.skip("arc.algebra.composition not available")
    return CompoundPerturbation.identity()


@pytest.fixture
def schema_only_perturbation(add_column_delta):
    """Compound perturbation with only a schema delta."""
    if not HAS_COMPOSITION:
        pytest.skip("arc.algebra.composition not available")
    return CompoundPerturbation.schema_only(add_column_delta)


# ─────────────────────────────────────────────────────────────────────
# Push-through fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def select_context():
    """OperatorContext for a SELECT operator."""
    if not HAS_PUSH:
        pytest.skip("arc.algebra.push not available")
    return OperatorContext(
        operator_type="SELECT",
        columns=[
            PushColumnRef(name="id", table="t"),
            PushColumnRef(name="name", table="t"),
        ],
    )


@pytest.fixture
def filter_context():
    """OperatorContext for a FILTER operator."""
    if not HAS_PUSH:
        pytest.skip("arc.algebra.push not available")
    return OperatorContext(
        operator_type="FILTER",
        predicate_columns=[
            PushColumnRef(name="active", table="t"),
        ],
    )


@pytest.fixture
def join_context():
    """OperatorContext for a JOIN operator."""
    if not HAS_PUSH:
        pytest.skip("arc.algebra.push not available")
    return OperatorContext(
        operator_type="JOIN",
        join_condition=PushJoinCond(
            left=PushColumnRef(name="id", table="left"),
            right=PushColumnRef(name="customer_id", table="right"),
        ),
    )


@pytest.fixture
def group_by_context():
    """OperatorContext for a GROUP_BY operator."""
    if not HAS_PUSH:
        pytest.skip("arc.algebra.push not available")
    return OperatorContext(
        operator_type="GROUP_BY",
        group_columns=[PushColumnRef(name="country", table="t")],
        aggregates=[
            PushAggSpec(func=AggregateFunction.COUNT, column=PushColumnRef(name="id", table="t"), alias="cnt"),
            PushAggSpec(func=AggregateFunction.SUM, column=PushColumnRef(name="amount", table="t"), alias="total"),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# DuckDB fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def duckdb_conn():
    """In-memory DuckDB connection for test isolation."""
    if not HAS_DUCKDB:
        pytest.skip("duckdb not installed")
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def duckdb_with_tables(duckdb_conn):
    """DuckDB connection pre-loaded with customers and orders tables."""
    duckdb_conn.execute("""
        CREATE TABLE customers (
            customer_id INT PRIMARY KEY,
            name VARCHAR,
            email VARCHAR,
            country VARCHAR
        )
    """)
    duckdb_conn.execute("""
        INSERT INTO customers VALUES
        (1, 'Alice', 'alice@test.com', 'US'),
        (2, 'Bob', 'bob@test.com', 'UK'),
        (3, 'Charlie', 'charlie@test.com', 'US'),
        (4, 'Diana', 'diana@test.com', 'DE'),
        (5, 'Eve', 'eve@test.com', 'FR')
    """)
    duckdb_conn.execute("""
        CREATE TABLE orders (
            order_id INT PRIMARY KEY,
            customer_id INT,
            amount DECIMAL(10,2),
            status VARCHAR,
            order_date DATE
        )
    """)
    duckdb_conn.execute("""
        INSERT INTO orders VALUES
        (101, 1, 99.99, 'shipped', '2024-01-15'),
        (102, 2, 149.50, 'pending', '2024-01-16'),
        (103, 1, 200.00, 'shipped', '2024-01-17'),
        (104, 3, 50.00, 'cancelled', '2024-01-18'),
        (105, 4, 300.00, 'shipped', '2024-01-19'),
        (106, 5, 75.25, 'pending', '2024-01-20'),
        (107, 2, 125.00, 'shipped', '2024-01-21')
    """)
    return duckdb_conn


@pytest.fixture
def duckdb_with_nulls(duckdb_conn):
    """DuckDB connection with a table containing NULL values for quality tests."""
    duckdb_conn.execute("""
        CREATE TABLE dirty_data (
            id INT,
            name VARCHAR,
            email VARCHAR,
            age INT,
            score DOUBLE
        )
    """)
    duckdb_conn.execute("""
        INSERT INTO dirty_data VALUES
        (1, 'Alice', 'alice@test.com', 30, 85.5),
        (2, NULL, 'bob@test.com', 25, 90.0),
        (3, 'Charlie', NULL, NULL, 78.3),
        (NULL, 'Diana', 'diana@test.com', 35, NULL),
        (5, 'Eve', 'eve@test.com', -5, 95.0),
        (6, 'Frank', 'frank@test.com', 200, 60.0)
    """)
    return duckdb_conn


# ─────────────────────────────────────────────────────────────────────
# Execution / planner fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def execution_engine():
    """ExecutionEngine with in-memory DuckDB."""
    if not HAS_EXECUTION:
        pytest.skip("arc.execution not available")
    engine = ExecutionEngine()
    yield engine
    engine.close()


@pytest.fixture
def cost_model():
    """Default CostModel for planner tests."""
    if not HAS_PLANNER:
        pytest.skip("arc.planner not available")
    return CostModel(factors=CostFactors())


@pytest.fixture
def dp_planner(cost_model):
    """DP-based repair planner."""
    return DPRepairPlanner(cost_model=cost_model)


@pytest.fixture
def plan_optimizer():
    """PlanOptimizer with default settings."""
    if not HAS_PLANNER:
        pytest.skip("arc.planner not available")
    return PlanOptimizer()


# ─────────────────────────────────────────────────────────────────────
# I/O fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pipeline_spec_dict():
    """Raw dict matching the PipelineSpec JSON schema."""
    return {
        "version": "1.0",
        "name": "test_pipeline",
        "nodes": [
            {"id": "src", "operator": "SOURCE", "query": ""},
            {"id": "xform", "operator": "SELECT", "query": "SELECT id, name FROM src"},
            {"id": "sink", "operator": "SINK", "query": ""},
        ],
        "edges": [
            {"source": "src", "target": "xform"},
            {"source": "xform", "target": "sink"},
        ],
    }


@pytest.fixture
def sample_delta_dict():
    """Raw dict for serialised SchemaDelta."""
    return {
        "type": "schema_delta",
        "operations": [
            {"op": "add_column", "name": "email", "sql_type": "VARCHAR", "nullable": True},
        ],
    }


# ─────────────────────────────────────────────────────────────────────
# Filesystem fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    """Temporary directory cleaned up after test."""
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def tmp_file(tmp_dir):
    """Temporary file path inside tmp_dir (does not create file)."""
    return os.path.join(tmp_dir, "test_output.json")


# ─────────────────────────────────────────────────────────────────────
# Availability markers for skip decorators
# ─────────────────────────────────────────────────────────────────────

requires_types = pytest.mark.skipif(not HAS_TYPES, reason="arc.types not available")
requires_graph = pytest.mark.skipif(not HAS_GRAPH, reason="arc.graph not available")
requires_schema_delta = pytest.mark.skipif(not HAS_SCHEMA_DELTA, reason="arc.algebra.schema_delta not available")
requires_data_delta = pytest.mark.skipif(not HAS_DATA_DELTA, reason="arc.algebra.data_delta not available")
requires_quality_delta = pytest.mark.skipif(not HAS_QUALITY_DELTA, reason="arc.algebra.quality_delta not available")
requires_composition = pytest.mark.skipif(not HAS_COMPOSITION, reason="arc.algebra.composition not available")
requires_interaction = pytest.mark.skipif(not HAS_INTERACTION, reason="arc.algebra.interaction not available")
requires_push = pytest.mark.skipif(not HAS_PUSH, reason="arc.algebra.push not available")
requires_sql_parser = pytest.mark.skipif(not HAS_SQL_PARSER, reason="arc.sql.parser not available")
requires_lineage = pytest.mark.skipif(not HAS_LINEAGE, reason="arc.sql.lineage not available")
requires_fragment = pytest.mark.skipif(not HAS_FRAGMENT, reason="arc.sql.fragment not available")
requires_planner = pytest.mark.skipif(not HAS_PLANNER, reason="arc.planner not available")
requires_execution = pytest.mark.skipif(not HAS_EXECUTION, reason="arc.execution not available")
requires_quality = pytest.mark.skipif(not HAS_QUALITY, reason="arc.quality not available")
requires_io = pytest.mark.skipif(not HAS_IO, reason="arc.io not available")
requires_duckdb = pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")
requires_numpy = pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")

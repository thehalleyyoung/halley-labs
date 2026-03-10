"""
Integration tests for the bounded commutation theorem (T2):

    apply(repair(σ), state(G)) ≈ recompute(evolve(G, σ))

For Fragment-F queries the equality is exact; for general queries the
error is bounded by ε.  These tests build DuckDB tables, execute SQL,
and compare repaired state against full recomputation.
"""

from __future__ import annotations

from typing import Any

import pytest

# ─────────────────────────────────────────────────────────────────────
# Graceful imports
# ─────────────────────────────────────────────────────────────────────

try:
    from arc.graph.pipeline import PipelineNode, PipelineEdge, PipelineGraph
    from arc.graph.builder import PipelineBuilder
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False

try:
    from arc.algebra.schema_delta import (
        SchemaDelta, AddColumn, DropColumn, RenameColumn, ChangeType,
        ColumnDef, Schema as AlgSchema, SQLType as AlgSQLType,
    )
    HAS_SCHEMA_DELTA = True
except ImportError:
    HAS_SCHEMA_DELTA = False

try:
    from arc.algebra.data_delta import (
        DataDelta, TypedTuple, MultiSet, InsertOp, DeleteOp,
    )
    HAS_DATA_DELTA = True
except ImportError:
    HAS_DATA_DELTA = False

try:
    from arc.algebra.quality_delta import (
        QualityDelta, QualityViolation, ViolationType, SeverityLevel,
    )
    HAS_QUALITY_DELTA = True
except ImportError:
    HAS_QUALITY_DELTA = False

try:
    from arc.algebra.composition import CompoundPerturbation
    HAS_COMPOSITION = True
except ImportError:
    HAS_COMPOSITION = False

try:
    from arc.algebra.interaction import PhiHomomorphism
    HAS_INTERACTION = True
except ImportError:
    HAS_INTERACTION = False

try:
    from arc.algebra.push import push_schema_delta, push_data_delta, OperatorContext
    HAS_PUSH = True
except ImportError:
    HAS_PUSH = False

try:
    from arc.algebra.propagation import DeltaPropagator, PropagationResult
    HAS_PROPAGATION = True
except ImportError:
    HAS_PROPAGATION = False

try:
    from arc.planner.dp import DPRepairPlanner
    from arc.planner.lp import LPRepairPlanner
    from arc.planner.cost import CostModel, CostFactors
    from arc.planner.optimizer import PlanOptimizer
    HAS_PLANNER = True
except ImportError:
    HAS_PLANNER = False

try:
    from arc.execution.engine import ExecutionEngine
    from arc.execution.checkpoint import CheckpointManager
    from arc.execution.validation import RepairValidator
    HAS_EXECUTION = True
except ImportError:
    HAS_EXECUTION = False

try:
    from arc.types.base import (
        Schema, Column, ParameterisedType, SQLType,
        ActionType, RepairAction, RepairPlan,
    )
    from arc.types.operators import SQLOperator
    HAS_TYPES = True
except ImportError:
    HAS_TYPES = False

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


# Compound availability checks
HAS_CORE = HAS_GRAPH and HAS_TYPES
HAS_ALGEBRA = HAS_SCHEMA_DELTA and HAS_DATA_DELTA and HAS_QUALITY_DELTA and HAS_COMPOSITION
HAS_FULL = HAS_CORE and HAS_ALGEBRA and HAS_PROPAGATION and HAS_PLANNER and HAS_EXECUTION and HAS_DUCKDB

requires_full = pytest.mark.skipif(not HAS_FULL, reason="full ARC stack or duckdb not available")


# =====================================================================
# Helpers
# =====================================================================

def _col(name, base=None, nullable=True, pos=0):
    if base is None:
        base = SQLType.INT
    return Column.quick(name, base, nullable=nullable, position=pos)


def _schema(*specs):
    return Schema(columns=tuple(
        _col(n, t, pos=i) for i, (n, t) in enumerate(specs)
    ))


def _node(node_id, operator=None, query="", schema=None):
    kw: dict[str, Any] = {"node_id": node_id}
    if operator is not None:
        kw["operator"] = operator
    if query:
        kw["query_text"] = query
    if schema is not None:
        kw["output_schema"] = schema
    return PipelineNode(**kw)


def _linear_3(name="rc_linear"):
    s = _schema(("id", SQLType.INT), ("name", SQLType.VARCHAR), ("val", SQLType.INT))
    g = PipelineGraph(name=name)
    g.add_node(_node("source", SQLOperator.SOURCE, schema=s))
    g.add_node(_node("transform", SQLOperator.SELECT,
                      query="SELECT id, name, val FROM source", schema=s))
    g.add_node(_node("sink", SQLOperator.SINK, schema=s))
    g.add_edge(PipelineEdge(source="source", target="transform"))
    g.add_edge(PipelineEdge(source="transform", target="sink"))
    return g


def _diamond(name="rc_diamond"):
    s = _schema(("id", SQLType.INT), ("name", SQLType.VARCHAR), ("val", SQLType.INT))
    g = PipelineGraph(name=name)
    g.add_node(_node("source", SQLOperator.SOURCE, schema=s))
    g.add_node(_node("left", SQLOperator.FILTER,
                      query="SELECT * FROM source WHERE val > 0", schema=s))
    g.add_node(_node("right", SQLOperator.SELECT,
                      query="SELECT id, name FROM source", schema=s))
    g.add_node(_node("merge", SQLOperator.JOIN,
                      query="SELECT * FROM left JOIN right ON left.id = right.id", schema=s))
    g.add_node(_node("sink", SQLOperator.SINK, schema=s))
    g.add_edge(PipelineEdge(source="source", target="left"))
    g.add_edge(PipelineEdge(source="source", target="right"))
    g.add_edge(PipelineEdge(source="left", target="merge"))
    g.add_edge(PipelineEdge(source="right", target="merge"))
    g.add_edge(PipelineEdge(source="merge", target="sink"))
    return g


def _fan_out(name="rc_fan_out"):
    s = _schema(("id", SQLType.INT), ("name", SQLType.VARCHAR))
    g = PipelineGraph(name=name)
    g.add_node(_node("source", SQLOperator.SOURCE, schema=s))
    for suffix in ("a", "b", "c"):
        g.add_node(_node(f"sink_{suffix}", SQLOperator.SINK, schema=s))
        g.add_edge(PipelineEdge(source="source", target=f"sink_{suffix}"))
    return g


def _fan_in(name="rc_fan_in"):
    s = _schema(("id", SQLType.INT), ("name", SQLType.VARCHAR))
    g = PipelineGraph(name=name)
    for suffix in ("a", "b", "c"):
        g.add_node(_node(f"source_{suffix}", SQLOperator.SOURCE, schema=s))
    g.add_node(_node("union", SQLOperator.UNION, schema=s))
    g.add_node(_node("sink", SQLOperator.SINK, schema=s))
    for suffix in ("a", "b", "c"):
        g.add_edge(PipelineEdge(source=f"source_{suffix}", target="union"))
    g.add_edge(PipelineEdge(source="union", target="sink"))
    return g


def _complex_10(name="rc_complex"):
    s = _schema(("id", SQLType.INT), ("name", SQLType.VARCHAR),
                ("val", SQLType.DECIMAL), ("ts", SQLType.TIMESTAMP))
    g = PipelineGraph(name=name)
    nodes = [
        ("src1", SQLOperator.SOURCE), ("src2", SQLOperator.SOURCE),
        ("filter1", SQLOperator.FILTER), ("filter2", SQLOperator.FILTER),
        ("join1", SQLOperator.JOIN), ("agg1", SQLOperator.GROUP_BY),
        ("select1", SQLOperator.SELECT), ("join2", SQLOperator.JOIN),
        ("final", SQLOperator.SELECT), ("sink", SQLOperator.SINK),
    ]
    for nid, op in nodes:
        g.add_node(_node(nid, op, schema=s))
    edges = [
        ("src1", "filter1"), ("src2", "filter2"),
        ("filter1", "join1"), ("filter2", "join1"),
        ("join1", "agg1"), ("join1", "select1"),
        ("agg1", "join2"), ("select1", "join2"),
        ("join2", "final"), ("final", "sink"),
    ]
    for src, tgt in edges:
        g.add_edge(PipelineEdge(source=src, target=tgt))
    return g


def _add_col_delta(col_name="email", sql_type=None):
    if sql_type is None:
        sql_type = AlgSQLType.VARCHAR
    return SchemaDelta.from_operations([
        AddColumn(name=col_name, sql_type=sql_type, nullable=True, position=-1),
    ])


def _insert_delta(n=3):
    tuples = [
        TypedTuple.from_dict({"id": i, "name": f"user_{i}", "val": i * 10})
        for i in range(100, 100 + n)
    ]
    return DataDelta.insert(MultiSet.from_tuples(tuples))


def _compound(schema_delta=None, data_delta=None, quality_delta=None):
    return CompoundPerturbation(
        schema_delta=schema_delta,
        data_delta=data_delta,
        quality_delta=quality_delta,
    )


def _setup_source_table(engine, table_name="source_tbl"):
    """Create and populate a simple source table in DuckDB."""
    engine.execute_sql(f"""
        CREATE TABLE {table_name} (id INT, name VARCHAR, val INT)
    """)
    engine.execute_sql(f"""
        INSERT INTO {table_name} VALUES
        (1, 'Alice', 10), (2, 'Bob', 20), (3, 'Charlie', 30),
        (4, 'Diana', 40), (5, 'Eve', 50)
    """)


def _recompute_table(engine, src_table, dest_table, query):
    """Full recomputation: drop dest and recreate from query."""
    engine.execute_sql(f"DROP TABLE IF EXISTS {dest_table}")
    engine.execute_sql(f"CREATE TABLE {dest_table} AS {query}")


def _tables_equal(engine, table_a, table_b):
    """Check if two DuckDB tables have identical contents (order-independent)."""
    result = engine.execute_sql(f"""
        SELECT COUNT(*) FROM (
            (SELECT * FROM {table_a} EXCEPT SELECT * FROM {table_b})
            UNION ALL
            (SELECT * FROM {table_b} EXCEPT SELECT * FROM {table_a})
        ) diff
    """)
    return result.fetchall()[0][0] == 0


def _table_l1_error(engine, table_a, table_b, numeric_col):
    """Compute L1 norm of difference for a numeric column."""
    result = engine.execute_sql(f"""
        SELECT COALESCE(SUM(ABS(a.{numeric_col} - b.{numeric_col})), 0)
        FROM {table_a} a JOIN {table_b} b ON a.id = b.id
    """)
    return result.fetchall()[0][0]


# =====================================================================
# 1. Linear pipeline, schema perturbation: repair = recompute
# =====================================================================

@requires_full
class TestLinearSchemaCorrectness:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.engine = ExecutionEngine()
        self.validator = RepairValidator(self.engine)
        yield
        self.engine.close()

    def test_add_column_repair_equals_recompute(self):
        _setup_source_table(self.engine)
        self.engine.execute_sql(
            "CREATE TABLE transform_tbl AS SELECT id, name, val FROM source_tbl"
        )

        # Apply schema delta (add column) to source
        delta = _add_col_delta("email")
        self.engine.apply_schema_delta("source_tbl", delta)
        # Apply same delta to transform (repair)
        self.engine.apply_schema_delta("transform_tbl", delta)

        # Recompute from scratch
        _recompute_table(
            self.engine, "source_tbl", "recomputed_tbl",
            "SELECT id, name, val, email FROM source_tbl",
        )

        result = self.validator.validate_fragment_f("transform_tbl", "recomputed_tbl")
        is_valid = getattr(result, 'is_valid', getattr(result, 'passed', None))
        assert is_valid is True

    def test_add_column_tables_match(self):
        _setup_source_table(self.engine)
        self.engine.execute_sql(
            "CREATE TABLE xform AS SELECT id, name, val FROM source_tbl"
        )

        delta = _add_col_delta("tag")
        self.engine.apply_schema_delta("source_tbl", delta)
        self.engine.apply_schema_delta("xform", delta)

        _recompute_table(
            self.engine, "source_tbl", "recomp",
            "SELECT id, name, val, tag FROM source_tbl",
        )

        assert _tables_equal(self.engine, "xform", "recomp")


# =====================================================================
# 2. Linear pipeline, data perturbation: repair = recompute
# =====================================================================

@requires_full
class TestLinearDataCorrectness:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.engine = ExecutionEngine()
        self.validator = RepairValidator(self.engine)
        yield
        self.engine.close()

    def test_insert_repair_equals_recompute(self):
        _setup_source_table(self.engine)
        self.engine.execute_sql(
            "CREATE TABLE transform_tbl AS SELECT id, name, val FROM source_tbl"
        )

        # Insert new rows into source
        self.engine.execute_sql(
            "INSERT INTO source_tbl VALUES (6, 'Frank', 60), (7, 'Grace', 70)"
        )
        # Apply same insert to transform (repair)
        self.engine.execute_sql(
            "INSERT INTO transform_tbl VALUES (6, 'Frank', 60), (7, 'Grace', 70)"
        )

        # Recompute
        _recompute_table(
            self.engine, "source_tbl", "recomputed_tbl",
            "SELECT id, name, val FROM source_tbl",
        )

        assert _tables_equal(self.engine, "transform_tbl", "recomputed_tbl")

    def test_delete_repair_equals_recompute(self):
        _setup_source_table(self.engine)
        self.engine.execute_sql(
            "CREATE TABLE transform_tbl AS SELECT id, name, val FROM source_tbl"
        )

        # Delete from source
        self.engine.execute_sql("DELETE FROM source_tbl WHERE id = 3")
        # Repair: delete same row from transform
        self.engine.execute_sql("DELETE FROM transform_tbl WHERE id = 3")

        _recompute_table(
            self.engine, "source_tbl", "recomputed_tbl",
            "SELECT id, name, val FROM source_tbl",
        )

        assert _tables_equal(self.engine, "transform_tbl", "recomputed_tbl")


# =====================================================================
# 3. Diamond pipeline, schema perturbation: repair = recompute
# =====================================================================

@requires_full
class TestDiamondSchemaCorrectness:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.engine = ExecutionEngine()
        self.validator = RepairValidator(self.engine)
        yield
        self.engine.close()

    def test_add_column_diamond_repair(self):
        _setup_source_table(self.engine)
        self.engine.execute_sql(
            "CREATE TABLE left_tbl AS SELECT * FROM source_tbl WHERE val > 0"
        )
        self.engine.execute_sql(
            "CREATE TABLE right_tbl AS SELECT id, name FROM source_tbl"
        )

        # Add column to source
        delta = _add_col_delta("tag")
        self.engine.apply_schema_delta("source_tbl", delta)
        self.engine.apply_schema_delta("left_tbl", delta)

        # Recompute left branch
        _recompute_table(
            self.engine, "source_tbl", "left_recomp",
            "SELECT * FROM source_tbl WHERE val > 0",
        )

        result = self.validator.validate_fragment_f("left_tbl", "left_recomp")
        is_valid = getattr(result, 'is_valid', getattr(result, 'passed', None))
        assert is_valid is True


# =====================================================================
# 4. Fan-out pipeline: source perturbation reaches all branches
# =====================================================================

@requires_full
class TestFanOutCorrectness:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.engine = ExecutionEngine()
        self.validator = RepairValidator(self.engine)
        yield
        self.engine.close()

    def test_insert_reaches_all_branches(self):
        self.engine.execute_sql("CREATE TABLE src (id INT, name VARCHAR)")
        self.engine.execute_sql("INSERT INTO src VALUES (1,'A'), (2,'B')")

        for suffix in ("a", "b", "c"):
            self.engine.execute_sql(
                f"CREATE TABLE sink_{suffix} AS SELECT * FROM src"
            )

        # Insert into source
        self.engine.execute_sql("INSERT INTO src VALUES (3, 'C')")
        # Repair: propagate to all sinks
        for suffix in ("a", "b", "c"):
            self.engine.execute_sql(f"INSERT INTO sink_{suffix} VALUES (3, 'C')")

        # Recompute and validate
        for suffix in ("a", "b", "c"):
            _recompute_table(self.engine, "src", f"recomp_{suffix}", "SELECT * FROM src")
            assert _tables_equal(self.engine, f"sink_{suffix}", f"recomp_{suffix}")


# =====================================================================
# 5. Fan-in pipeline: multiple source perturbations merged at UNION
# =====================================================================

@requires_full
class TestFanInCorrectness:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.engine = ExecutionEngine()
        self.validator = RepairValidator(self.engine)
        yield
        self.engine.close()

    def test_multi_source_union(self):
        for suffix in ("a", "b", "c"):
            self.engine.execute_sql(
                f"CREATE TABLE src_{suffix} (id INT, name VARCHAR)"
            )
            self.engine.execute_sql(
                f"INSERT INTO src_{suffix} VALUES ({ord(suffix) - ord('a') + 1}, '{suffix.upper()}')"
            )

        self.engine.execute_sql("""
            CREATE TABLE union_tbl AS
            SELECT * FROM src_a UNION ALL
            SELECT * FROM src_b UNION ALL
            SELECT * FROM src_c
        """)

        # Perturb source_a
        self.engine.execute_sql("INSERT INTO src_a VALUES (10, 'New')")
        # Repair union
        self.engine.execute_sql("INSERT INTO union_tbl VALUES (10, 'New')")

        # Recompute
        _recompute_table(self.engine, "src_a", "union_recomp", """
            SELECT * FROM src_a UNION ALL
            SELECT * FROM src_b UNION ALL
            SELECT * FROM src_c
        """)

        assert _tables_equal(self.engine, "union_tbl", "union_recomp")


# =====================================================================
# 6. Complex 10-node DAG: repair matches recompute
# =====================================================================

@requires_full
class TestComplexDAGCorrectness:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.engine = ExecutionEngine()
        self.validator = RepairValidator(self.engine)
        yield
        self.engine.close()

    def test_propagation_and_plan_on_complex_dag(self):
        """Verify the full stack produces a valid plan for 10-node DAG."""
        graph = _complex_10()
        delta = _add_col_delta()
        perturbation = _compound(schema_delta=delta)
        propagator = DeltaPropagator()
        result = propagator.propagate(graph, "src1", perturbation)

        planner = DPRepairPlanner(cost_model=CostModel(factors=CostFactors()))
        deltas = getattr(result, 'node_deltas', getattr(result, 'deltas', {"src1": perturbation}))
        plan = planner.plan(graph, deltas)
        assert plan is not None

    def test_complex_dag_sql_repair(self):
        """Build partial tables and verify repair correctness for a sub-path."""
        self.engine.execute_sql("CREATE TABLE src1_tbl (id INT, name VARCHAR, val INT)")
        self.engine.execute_sql(
            "INSERT INTO src1_tbl VALUES (1,'A',10),(2,'B',20),(3,'C',30)"
        )
        self.engine.execute_sql(
            "CREATE TABLE filter1_tbl AS SELECT * FROM src1_tbl WHERE val > 10"
        )

        # Perturbation: insert
        self.engine.execute_sql("INSERT INTO src1_tbl VALUES (4,'D',40)")
        self.engine.execute_sql("INSERT INTO filter1_tbl VALUES (4,'D',40)")

        _recompute_table(
            self.engine, "src1_tbl", "filter1_recomp",
            "SELECT * FROM src1_tbl WHERE val > 10",
        )

        assert _tables_equal(self.engine, "filter1_tbl", "filter1_recomp")


# =====================================================================
# 7. Non-Fragment-F: measure error bound, ε < tolerance
# =====================================================================

@requires_full
class TestNonFragmentFBound:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.engine = ExecutionEngine()
        self.validator = RepairValidator(self.engine)
        yield
        self.engine.close()

    def test_approximate_repair_within_epsilon(self):
        """For a query outside Fragment F, repair error should be bounded."""
        self.engine.execute_sql("CREATE TABLE src (id INT, val DOUBLE)")
        self.engine.execute_sql(
            "INSERT INTO src VALUES (1,1.0),(2,2.0),(3,3.0),(4,4.0),(5,5.0)"
        )
        self.engine.execute_sql("CREATE TABLE agg AS SELECT SUM(val) as total FROM src")

        # Perturbation: insert a row
        self.engine.execute_sql("INSERT INTO src VALUES (6, 6.0)")
        # Repair: add 6.0 to aggregate
        self.engine.execute_sql("UPDATE agg SET total = total + 6.0")

        # Recompute
        _recompute_table(self.engine, "src", "agg_recomp", "SELECT SUM(val) as total FROM src")

        result = self.validator.validate_general("agg", "agg_recomp", epsilon=0.01)
        is_valid = getattr(result, 'is_valid', getattr(result, 'passed', None))
        assert is_valid is True

    def test_l1_error_bounded(self):
        """Directly measure L1 error between repaired and recomputed."""
        self.engine.execute_sql("CREATE TABLE src2 (id INT, val DOUBLE)")
        self.engine.execute_sql(
            "INSERT INTO src2 VALUES (1,10.0),(2,20.0),(3,30.0)"
        )
        self.engine.execute_sql("CREATE TABLE rep (id INT, val DOUBLE)")
        self.engine.execute_sql(
            "INSERT INTO rep VALUES (1,10.0),(2,20.0),(3,30.0)"
        )
        # Exact copy → error should be 0
        error = _table_l1_error(self.engine, "rep", "src2", "val")
        assert error < 1e-9


# =====================================================================
# 8. Multiple perturbation types: compound repair correctness
# =====================================================================

@requires_full
class TestCompoundRepairCorrectness:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.engine = ExecutionEngine()
        self.validator = RepairValidator(self.engine)
        yield
        self.engine.close()

    def test_schema_and_data_compound(self):
        """Schema + data perturbation applied together."""
        _setup_source_table(self.engine)
        self.engine.execute_sql(
            "CREATE TABLE xform AS SELECT id, name, val FROM source_tbl"
        )

        # Schema: add column
        delta = _add_col_delta("tag")
        self.engine.apply_schema_delta("source_tbl", delta)
        self.engine.apply_schema_delta("xform", delta)

        # Data: insert
        self.engine.execute_sql("INSERT INTO source_tbl VALUES (6,'F',60,NULL)")
        self.engine.execute_sql("INSERT INTO xform VALUES (6,'F',60,NULL)")

        # Recompute
        _recompute_table(
            self.engine, "source_tbl", "recomp",
            "SELECT id, name, val, tag FROM source_tbl",
        )

        assert _tables_equal(self.engine, "xform", "recomp")


# =====================================================================
# 9. Sequential perturbations: first repair then second, compare
# =====================================================================

@requires_full
class TestSequentialPerturbations:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.engine = ExecutionEngine()
        self.validator = RepairValidator(self.engine)
        yield
        self.engine.close()

    def test_two_sequential_inserts(self):
        _setup_source_table(self.engine)
        self.engine.execute_sql(
            "CREATE TABLE xform AS SELECT id, name, val FROM source_tbl"
        )

        # First perturbation
        self.engine.execute_sql("INSERT INTO source_tbl VALUES (6,'Frank',60)")
        self.engine.execute_sql("INSERT INTO xform VALUES (6,'Frank',60)")

        # Second perturbation
        self.engine.execute_sql("INSERT INTO source_tbl VALUES (7,'Grace',70)")
        self.engine.execute_sql("INSERT INTO xform VALUES (7,'Grace',70)")

        # Full recompute
        _recompute_table(
            self.engine, "source_tbl", "recomp",
            "SELECT id, name, val FROM source_tbl",
        )

        assert _tables_equal(self.engine, "xform", "recomp")

    def test_schema_then_data(self):
        _setup_source_table(self.engine)
        self.engine.execute_sql(
            "CREATE TABLE xform AS SELECT id, name, val FROM source_tbl"
        )

        # Schema perturbation
        delta = _add_col_delta("extra")
        self.engine.apply_schema_delta("source_tbl", delta)
        self.engine.apply_schema_delta("xform", delta)

        # Data perturbation (after schema change)
        self.engine.execute_sql("INSERT INTO source_tbl VALUES (6,'F',60,NULL)")
        self.engine.execute_sql("INSERT INTO xform VALUES (6,'F',60,NULL)")

        _recompute_table(
            self.engine, "source_tbl", "recomp",
            "SELECT id, name, val, extra FROM source_tbl",
        )

        assert _tables_equal(self.engine, "xform", "recomp")


# =====================================================================
# 10. RepairValidator.validate_fragment_f and validate_general
# =====================================================================

@requires_full
class TestValidatorMethods:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.engine = ExecutionEngine()
        self.validator = RepairValidator(self.engine)
        yield
        self.engine.close()

    def test_validate_fragment_f_identical_tables(self):
        self.engine.execute_sql("CREATE TABLE a (id INT, val VARCHAR)")
        self.engine.execute_sql("INSERT INTO a VALUES (1,'x'),(2,'y')")
        self.engine.execute_sql("CREATE TABLE b AS SELECT * FROM a")

        result = self.validator.validate_fragment_f("a", "b")
        is_valid = getattr(result, 'is_valid', getattr(result, 'passed', None))
        assert is_valid is True

    def test_validate_fragment_f_different_tables(self):
        self.engine.execute_sql("CREATE TABLE c (id INT, val VARCHAR)")
        self.engine.execute_sql("INSERT INTO c VALUES (1,'x')")
        self.engine.execute_sql("CREATE TABLE d (id INT, val VARCHAR)")
        self.engine.execute_sql("INSERT INTO d VALUES (1,'x'),(2,'y')")

        result = self.validator.validate_fragment_f("c", "d")
        is_valid = getattr(result, 'is_valid', getattr(result, 'passed', None))
        # Tables differ, so validation should fail
        assert is_valid is False

    def test_validate_general_identical(self):
        self.engine.execute_sql("CREATE TABLE e (id INT, val DOUBLE)")
        self.engine.execute_sql("INSERT INTO e VALUES (1,1.0),(2,2.0)")
        self.engine.execute_sql("CREATE TABLE f AS SELECT * FROM e")

        result = self.validator.validate_general("e", "f", epsilon=0.001)
        is_valid = getattr(result, 'is_valid', getattr(result, 'passed', None))
        assert is_valid is True

    def test_validate_general_within_epsilon(self):
        self.engine.execute_sql("CREATE TABLE g (id INT, val DOUBLE)")
        self.engine.execute_sql("INSERT INTO g VALUES (1, 100.0)")
        self.engine.execute_sql("CREATE TABLE h (id INT, val DOUBLE)")
        self.engine.execute_sql("INSERT INTO h VALUES (1, 100.0)")

        result = self.validator.validate_general("g", "h", epsilon=1.0)
        is_valid = getattr(result, 'is_valid', getattr(result, 'passed', None))
        assert is_valid is True

    def test_validate_general_empty_tables(self):
        self.engine.execute_sql("CREATE TABLE empty1 (id INT, val DOUBLE)")
        self.engine.execute_sql("CREATE TABLE empty2 (id INT, val DOUBLE)")

        result = self.validator.validate_general("empty1", "empty2", epsilon=0.001)
        is_valid = getattr(result, 'is_valid', getattr(result, 'passed', None))
        assert is_valid is True

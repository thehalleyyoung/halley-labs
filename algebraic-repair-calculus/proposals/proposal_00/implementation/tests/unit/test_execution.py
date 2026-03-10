"""
Unit tests for ``arc.execution`` — DuckDB-based execution engine,
checkpointing, validation, and scheduling.

All tests use DuckDB in-memory databases.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

try:
    import duckdb
except ImportError:
    pytest.skip("duckdb not installed", allow_module_level=True)

from arc.execution.engine import ExecutionEngine
from arc.execution.checkpoint import CheckpointManager
from arc.execution.validation import RepairValidator
from arc.execution.scheduler import ExecutionScheduler
from arc.types.base import (
    ActionType,
    CheckpointInfo,
    Column,
    CompoundPerturbation,
    DataDelta,
    ExecutionResult,
    ExecutionSchedule,
    ParameterisedType,
    PipelineEdge,
    PipelineGraph,
    PipelineNode,
    RepairAction,
    RepairPlan,
    RowChange,
    RowChangeType,
    Schema,
    SchemaDelta,
    SchemaOperation,
    SchemaOpType,
    SQLOperator,
    SQLType,
    TableStats,
    ValidationResult,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def execution_engine():
    """Provide an in-memory ExecutionEngine, closed after the test."""
    engine = ExecutionEngine()
    yield engine
    engine.close()


@pytest.fixture
def duckdb_conn():
    """Provide a raw DuckDB in-memory connection."""
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


# =====================================================================
# Helper factories
# =====================================================================


def _make_schema(*col_specs: tuple[str, SQLType]) -> Schema:
    """Build a Schema from (name, SQLType) pairs."""
    cols = tuple(
        Column.quick(name, stype, nullable=True, position=i)
        for i, (name, stype) in enumerate(col_specs)
    )
    return Schema(columns=cols)


def _make_schema_with_pk(
    col_specs: list[tuple[str, SQLType]], pk: list[str]
) -> Schema:
    cols = tuple(
        Column.quick(name, stype, nullable=(name not in pk), position=i)
        for i, (name, stype) in enumerate(col_specs)
    )
    return Schema(columns=cols, primary_key=tuple(pk))


def _simple_graph(*node_ids: str) -> PipelineGraph:
    """Build a linear pipeline graph: n0 → n1 → n2 …"""
    g = PipelineGraph()
    for nid in node_ids:
        g.add_node(PipelineNode(node_id=nid, table_name=nid))
    for i in range(len(node_ids) - 1):
        g.add_edge(PipelineEdge(source=node_ids[i], target=node_ids[i + 1]))
    return g


def _make_plan(actions: list[RepairAction]) -> RepairPlan:
    order = tuple(a.node_id for a in actions)
    total_cost = sum(a.estimated_cost for a in actions)
    return RepairPlan(
        actions=tuple(actions),
        execution_order=order,
        total_cost=total_cost,
    )


def _insert_delta(rows: list[dict[str, Any]]) -> DataDelta:
    """Create a DataDelta with INSERT changes."""
    changes = tuple(
        RowChange(
            change_type=RowChangeType.INSERT,
            new_values=row,
        )
        for row in rows
    )
    return DataDelta(changes=changes)


# =====================================================================
# ExecutionEngine tests
# =====================================================================


class TestExecutionEngineBasic:
    """Core engine lifecycle and SQL execution."""

    def test_context_manager(self):
        """Engine can be used as a context manager."""
        with ExecutionEngine() as engine:
            assert engine is not None
            assert engine.connection is not None
        # After exiting, the connection should be closed
        # (attempting to use it would raise)

    def test_repr(self, execution_engine: ExecutionEngine):
        r = repr(execution_engine)
        assert "ExecutionEngine" in r
        assert ":memory:" in r

    def test_execute_sql_create_and_insert(self, execution_engine: ExecutionEngine):
        """execute_sql handles DDL and DML."""
        execution_engine.execute_sql(
            "CREATE TABLE test_tbl (id INTEGER, name VARCHAR)"
        )
        execution_engine.execute_sql(
            "INSERT INTO test_tbl VALUES (1, 'alice')"
        )
        result = execution_engine.execute_sql("SELECT * FROM test_tbl")
        assert result is not None
        rows = result.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 1
        assert rows[0][1] == "alice"

    def test_execute_sql_returns_none_for_ddl(
        self, execution_engine: ExecutionEngine
    ):
        """DDL may return a relation but never raises."""
        execution_engine.execute_sql("CREATE TABLE t (x INT)")
        # Just ensuring no exception is raised

    def test_execute_sql_error_propagates(
        self, execution_engine: ExecutionEngine
    ):
        """Invalid SQL should raise."""
        with pytest.raises(Exception):
            execution_engine.execute_sql("SELECT * FROM nonexistent_table_xyz")


class TestExecutionEngineTableOps:
    """Table CRUD operations."""

    def test_create_table_from_schema(self, execution_engine: ExecutionEngine):
        schema = _make_schema(("id", SQLType.INT), ("val", SQLType.FLOAT))
        execution_engine.create_table("my_table", schema)
        assert execution_engine.table_exists("my_table")

    def test_create_table_with_primary_key(
        self, execution_engine: ExecutionEngine
    ):
        schema = _make_schema_with_pk(
            [("id", SQLType.INT), ("name", SQLType.VARCHAR)], pk=["id"]
        )
        execution_engine.create_table("pk_table", schema)
        assert execution_engine.table_exists("pk_table")

    def test_table_exists_positive(self, execution_engine: ExecutionEngine):
        execution_engine.execute_sql("CREATE TABLE existing (x INT)")
        assert execution_engine.table_exists("existing") is True

    def test_table_exists_negative(self, execution_engine: ExecutionEngine):
        assert execution_engine.table_exists("does_not_exist") is False

    def test_drop_table(self, execution_engine: ExecutionEngine):
        execution_engine.execute_sql("CREATE TABLE to_drop (x INT)")
        assert execution_engine.table_exists("to_drop")
        execution_engine.drop_table("to_drop")
        assert execution_engine.table_exists("to_drop") is False

    def test_drop_table_if_exists_no_error(
        self, execution_engine: ExecutionEngine
    ):
        """drop_table with if_exists=True shouldn't raise for missing table."""
        execution_engine.drop_table("never_existed", if_exists=True)

    def test_get_table_schema(self, execution_engine: ExecutionEngine):
        execution_engine.execute_sql(
            "CREATE TABLE schema_test (a INTEGER, b VARCHAR, c DOUBLE)"
        )
        cols = execution_engine.get_table_schema("schema_test")
        assert len(cols) == 3
        names = [c["name"] for c in cols]
        assert "a" in names
        assert "b" in names
        assert "c" in names


class TestApplySchemaAndDataDelta:
    """Schema and data delta application."""

    def test_apply_schema_delta_add_column(
        self, execution_engine: ExecutionEngine
    ):
        execution_engine.execute_sql("CREATE TABLE sd_test (id INTEGER)")
        delta = SchemaDelta(
            operations=(
                SchemaOperation(
                    op_type=SchemaOpType.ADD_COLUMN,
                    column_name="new_col",
                    dtype=SQLType.VARCHAR,
                ),
            )
        )
        execution_engine.apply_schema_delta("sd_test", delta)
        cols = execution_engine.get_table_schema("sd_test")
        col_names = [c["name"] for c in cols]
        assert "new_col" in col_names

    def test_apply_schema_delta_drop_column(
        self, execution_engine: ExecutionEngine
    ):
        execution_engine.execute_sql(
            "CREATE TABLE sd_drop (id INTEGER, old_col VARCHAR)"
        )
        delta = SchemaDelta(
            operations=(
                SchemaOperation(
                    op_type=SchemaOpType.DROP_COLUMN,
                    column_name="old_col",
                ),
            )
        )
        execution_engine.apply_schema_delta("sd_drop", delta)
        cols = execution_engine.get_table_schema("sd_drop")
        col_names = [c["name"] for c in cols]
        assert "old_col" not in col_names

    def test_apply_schema_delta_rename_column(
        self, execution_engine: ExecutionEngine
    ):
        execution_engine.execute_sql(
            "CREATE TABLE sd_rename (id INTEGER, old_name VARCHAR)"
        )
        delta = SchemaDelta(
            operations=(
                SchemaOperation(
                    op_type=SchemaOpType.RENAME_COLUMN,
                    column_name="old_name",
                    new_column_name="new_name",
                ),
            )
        )
        execution_engine.apply_schema_delta("sd_rename", delta)
        cols = execution_engine.get_table_schema("sd_rename")
        col_names = [c["name"] for c in cols]
        assert "new_name" in col_names
        assert "old_name" not in col_names

    def test_apply_data_delta_insert(self, execution_engine: ExecutionEngine):
        execution_engine.execute_sql(
            "CREATE TABLE dd_ins (id INTEGER, name VARCHAR)"
        )
        delta = _insert_delta([
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
            {"id": 3, "name": "charlie"},
        ])
        rows_affected = execution_engine.apply_data_delta("dd_ins", delta)
        assert rows_affected == 3

        result = execution_engine.execute_sql(
            "SELECT COUNT(*) FROM dd_ins"
        )
        assert result is not None
        assert result.fetchone()[0] == 3

    def test_apply_data_delta_delete(self, execution_engine: ExecutionEngine):
        execution_engine.execute_sql(
            "CREATE TABLE dd_del (id INTEGER, name VARCHAR)"
        )
        execution_engine.execute_sql(
            "INSERT INTO dd_del VALUES (1, 'alice'), (2, 'bob')"
        )
        delta = DataDelta(
            changes=(
                RowChange(
                    change_type=RowChangeType.DELETE,
                    old_values={"id": 1, "name": "alice"},
                ),
            )
        )
        rows = execution_engine.apply_data_delta("dd_del", delta)
        assert rows == 1
        result = execution_engine.execute_sql("SELECT COUNT(*) FROM dd_del")
        assert result.fetchone()[0] == 1

    def test_apply_data_delta_update(self, execution_engine: ExecutionEngine):
        execution_engine.execute_sql(
            "CREATE TABLE dd_upd (id INTEGER, name VARCHAR)"
        )
        execution_engine.execute_sql(
            "INSERT INTO dd_upd VALUES (1, 'alice')"
        )
        delta = DataDelta(
            changes=(
                RowChange(
                    change_type=RowChangeType.UPDATE,
                    old_values={"id": 1},
                    new_values={"name": "ALICE"},
                ),
            )
        )
        rows = execution_engine.apply_data_delta("dd_upd", delta)
        assert rows == 1
        result = execution_engine.execute_sql(
            "SELECT name FROM dd_upd WHERE id = 1"
        )
        assert result.fetchone()[0] == "ALICE"


class TestGetTableStats:
    """get_table_stats correctness."""

    def test_basic_stats(self, execution_engine: ExecutionEngine):
        execution_engine.execute_sql(
            "CREATE TABLE stats_t (id INTEGER, name VARCHAR, score DOUBLE)"
        )
        execution_engine.execute_sql(
            "INSERT INTO stats_t VALUES (1, 'a', 10.0), (2, 'b', 20.0), (3, NULL, 30.0)"
        )
        stats = execution_engine.get_table_stats("stats_t")
        assert isinstance(stats, TableStats)
        assert stats.table_name == "stats_t"
        assert stats.row_count == 3
        assert stats.column_count == 3

    def test_stats_null_counts(self, execution_engine: ExecutionEngine):
        execution_engine.execute_sql(
            "CREATE TABLE stats_null (x INTEGER, y VARCHAR)"
        )
        execution_engine.execute_sql(
            "INSERT INTO stats_null VALUES (1, NULL), (NULL, 'b'), (3, NULL)"
        )
        stats = execution_engine.get_table_stats("stats_null")
        assert stats.null_counts.get("y", 0) == 2
        assert stats.null_counts.get("x", 0) == 1

    def test_stats_distinct_counts(self, execution_engine: ExecutionEngine):
        execution_engine.execute_sql("CREATE TABLE stats_dist (v INTEGER)")
        execution_engine.execute_sql(
            "INSERT INTO stats_dist VALUES (1), (1), (2), (3)"
        )
        stats = execution_engine.get_table_stats("stats_dist")
        assert stats.distinct_counts.get("v") == 3

    def test_stats_empty_table(self, execution_engine: ExecutionEngine):
        execution_engine.execute_sql("CREATE TABLE stats_empty (id INT)")
        stats = execution_engine.get_table_stats("stats_empty")
        assert stats.row_count == 0

    def test_stats_min_max(self, execution_engine: ExecutionEngine):
        execution_engine.execute_sql("CREATE TABLE stats_mm (val DOUBLE)")
        execution_engine.execute_sql(
            "INSERT INTO stats_mm VALUES (1.5), (3.7), (2.2)"
        )
        stats = execution_engine.get_table_stats("stats_mm")
        assert stats.min_values.get("val") == pytest.approx(1.5)
        assert stats.max_values.get("val") == pytest.approx(3.7)


# =====================================================================
# CheckpointManager tests
# =====================================================================


class TestCheckpointManager:
    """Checkpoint creation, save/restore, commit, and listing."""

    def test_create_checkpoint_returns_id(
        self, execution_engine: ExecutionEngine
    ):
        mgr = CheckpointManager(execution_engine)
        cp_id = mgr.create_checkpoint(metadata={"reason": "test"})
        assert isinstance(cp_id, str)
        assert len(cp_id) > 0

    def test_create_multiple_checkpoints(
        self, execution_engine: ExecutionEngine
    ):
        mgr = CheckpointManager(execution_engine)
        id1 = mgr.create_checkpoint()
        id2 = mgr.create_checkpoint()
        assert id1 != id2

    def test_save_and_restore_table_state(
        self, execution_engine: ExecutionEngine
    ):
        execution_engine.execute_sql("CREATE TABLE ckpt_t (x INTEGER)")
        execution_engine.execute_sql("INSERT INTO ckpt_t VALUES (1), (2)")

        mgr = CheckpointManager(execution_engine)
        cp_id = mgr.create_checkpoint()
        mgr.save_table_state(cp_id, "ckpt_t")

        # Modify the table
        execution_engine.execute_sql("DELETE FROM ckpt_t WHERE x = 1")
        result = execution_engine.execute_sql("SELECT COUNT(*) FROM ckpt_t")
        assert result.fetchone()[0] == 1

        # Restore
        mgr.restore_checkpoint(cp_id)
        result = execution_engine.execute_sql("SELECT COUNT(*) FROM ckpt_t")
        assert result.fetchone()[0] == 2

    def test_commit_cleans_up(self, execution_engine: ExecutionEngine):
        execution_engine.execute_sql("CREATE TABLE ckpt_commit (x INT)")
        mgr = CheckpointManager(execution_engine)
        cp_id = mgr.create_checkpoint()
        mgr.save_table_state(cp_id, "ckpt_commit")

        # Commit removes the checkpoint
        mgr.commit_checkpoint(cp_id)
        assert len(mgr.list_checkpoints()) == 0

    def test_commit_unknown_checkpoint_raises(
        self, execution_engine: ExecutionEngine
    ):
        mgr = CheckpointManager(execution_engine)
        with pytest.raises(KeyError):
            mgr.commit_checkpoint("nonexistent_id")

    def test_list_checkpoints(self, execution_engine: ExecutionEngine):
        mgr = CheckpointManager(execution_engine)
        mgr.create_checkpoint(metadata={"idx": 0})
        mgr.create_checkpoint(metadata={"idx": 1})
        cps = mgr.list_checkpoints()
        assert len(cps) == 2
        assert all(isinstance(c, CheckpointInfo) for c in cps)

    def test_get_checkpoint_size(self, execution_engine: ExecutionEngine):
        execution_engine.execute_sql("CREATE TABLE ckpt_size (id INT)")
        execution_engine.execute_sql(
            "INSERT INTO ckpt_size VALUES (1), (2), (3)"
        )
        mgr = CheckpointManager(execution_engine)
        cp_id = mgr.create_checkpoint()
        mgr.save_table_state(cp_id, "ckpt_size")
        size = mgr.get_checkpoint_size(cp_id)
        assert isinstance(size, int)
        assert size > 0

    def test_save_table_state_unknown_checkpoint(
        self, execution_engine: ExecutionEngine
    ):
        mgr = CheckpointManager(execution_engine)
        with pytest.raises(KeyError):
            mgr.save_table_state("bad_id", "some_table")

    def test_cleanup_old_checkpoints(self, execution_engine: ExecutionEngine):
        mgr = CheckpointManager(execution_engine)
        mgr.create_checkpoint()
        # The checkpoint was just created, so max_age=0 should remove it
        # (created_at ~= now, now - created_at ~= 0 which is > max_age=0)
        # We need max_age to be very small or wait; use max_age=-1 to
        # guarantee removal.
        removed = mgr.cleanup_old_checkpoints(max_age_seconds=-1)
        assert removed == 1
        assert len(mgr.list_checkpoints()) == 0

    def test_repr(self, execution_engine: ExecutionEngine):
        mgr = CheckpointManager(execution_engine)
        assert "CheckpointManager" in repr(mgr)


# =====================================================================
# RepairValidator tests
# =====================================================================


class TestRepairValidator:
    """Validation of repair correctness."""

    def test_validate_fragment_f_identical_tables(
        self, execution_engine: ExecutionEngine
    ):
        execution_engine.execute_sql(
            "CREATE TABLE vf_rep AS SELECT 1 AS id, 'a' AS val "
            "UNION ALL SELECT 2, 'b'"
        )
        execution_engine.execute_sql(
            "CREATE TABLE vf_recomp AS SELECT 1 AS id, 'a' AS val "
            "UNION ALL SELECT 2, 'b'"
        )
        validator = RepairValidator(execution_engine)
        result = validator.validate_fragment_f("vf_rep", "vf_recomp")
        assert result.is_valid is True
        assert result.exact_match is True

    def test_validate_fragment_f_different_tables(
        self, execution_engine: ExecutionEngine
    ):
        execution_engine.execute_sql(
            "CREATE TABLE vf_diff_rep AS SELECT 1 AS id, 'a' AS val"
        )
        execution_engine.execute_sql(
            "CREATE TABLE vf_diff_rec AS SELECT 1 AS id, 'DIFFERENT' AS val"
        )
        validator = RepairValidator(execution_engine)
        result = validator.validate_fragment_f("vf_diff_rep", "vf_diff_rec")
        assert result.is_valid is False

    def test_validate_fragment_f_extra_rows(
        self, execution_engine: ExecutionEngine
    ):
        execution_engine.execute_sql(
            "CREATE TABLE vf_extra_r AS SELECT 1 AS id "
            "UNION ALL SELECT 2"
        )
        execution_engine.execute_sql(
            "CREATE TABLE vf_extra_c AS SELECT 1 AS id"
        )
        validator = RepairValidator(execution_engine)
        result = validator.validate_fragment_f("vf_extra_r", "vf_extra_c")
        assert result.is_valid is False
        assert result.actual_error is not None
        assert result.actual_error > 0

    def test_validate_general_within_epsilon(
        self, execution_engine: ExecutionEngine
    ):
        execution_engine.execute_sql(
            "CREATE TABLE vg_rep (id INTEGER, score DOUBLE)"
        )
        execution_engine.execute_sql(
            "INSERT INTO vg_rep VALUES (1, 10.0), (2, 20.0)"
        )
        execution_engine.execute_sql(
            "CREATE TABLE vg_rec (id INTEGER, score DOUBLE)"
        )
        execution_engine.execute_sql(
            "INSERT INTO vg_rec VALUES (1, 10.0), (2, 20.0)"
        )
        validator = RepairValidator(execution_engine, default_epsilon=1e-3)
        result = validator.validate_general("vg_rep", "vg_rec")
        assert result.is_valid is True

    def test_validate_general_exceeds_epsilon(
        self, execution_engine: ExecutionEngine
    ):
        execution_engine.execute_sql(
            "CREATE TABLE vg_far_r (id INTEGER, score DOUBLE)"
        )
        execution_engine.execute_sql(
            "INSERT INTO vg_far_r VALUES (1, 10.0), (2, 20.0)"
        )
        execution_engine.execute_sql(
            "CREATE TABLE vg_far_c (id INTEGER, score DOUBLE)"
        )
        execution_engine.execute_sql(
            "INSERT INTO vg_far_c VALUES (1, 100.0), (2, 200.0)"
        )
        validator = RepairValidator(execution_engine, default_epsilon=1e-6)
        result = validator.validate_general("vg_far_r", "vg_far_c")
        assert result.is_valid is False
        assert result.actual_error is not None
        assert result.actual_error > 1e-6

    def test_validate_general_non_numeric_falls_back(
        self, execution_engine: ExecutionEngine
    ):
        """When only non-numeric cols, falls back to Fragment-F."""
        execution_engine.execute_sql(
            "CREATE TABLE vg_str_r AS SELECT 'hello' AS s"
        )
        execution_engine.execute_sql(
            "CREATE TABLE vg_str_c AS SELECT 'hello' AS s"
        )
        validator = RepairValidator(execution_engine)
        result = validator.validate_general("vg_str_r", "vg_str_c")
        assert result.is_valid is True

    def test_validate_plan_empty_graph(
        self, execution_engine: ExecutionEngine
    ):
        graph = PipelineGraph()
        plan = RepairPlan()
        validator = RepairValidator(execution_engine)
        result = validator.validate_plan(plan, graph)
        assert result.is_valid is True

    def test_generate_validation_report(
        self, execution_engine: ExecutionEngine
    ):
        validator = RepairValidator(execution_engine)
        vr = ValidationResult(is_valid=True, exact_match=True, message="ok")
        report = validator.generate_validation_report(vr)
        assert "PASS" in report
        assert "Repair Validation Report" in report

    def test_generate_validation_report_failed(
        self, execution_engine: ExecutionEngine
    ):
        validator = RepairValidator(execution_engine)
        vr = ValidationResult(
            is_valid=False,
            exact_match=False,
            actual_error=5.0,
            error_bound=1.0,
            message="failed",
        )
        report = validator.generate_validation_report(vr)
        assert "FAIL" in report

    def test_validate_schema_consistency_no_violations(
        self, execution_engine: ExecutionEngine
    ):
        schema_a = _make_schema(("x", SQLType.INT))
        schema_b = _make_schema(("x", SQLType.INT))
        graph = PipelineGraph()
        graph.add_node(
            PipelineNode(node_id="A", output_schema=schema_a)
        )
        graph.add_node(
            PipelineNode(node_id="B", input_schema=schema_b)
        )
        graph.add_edge(PipelineEdge(source="A", target="B"))
        validator = RepairValidator(execution_engine)
        violations = validator.validate_schema_consistency(graph)
        assert violations == []

    def test_validate_schema_consistency_missing_column(
        self, execution_engine: ExecutionEngine
    ):
        schema_a = _make_schema(("x", SQLType.INT))
        graph = PipelineGraph()
        graph.add_node(
            PipelineNode(node_id="A", output_schema=schema_a)
        )
        graph.add_node(PipelineNode(node_id="B"))
        graph.add_edge(
            PipelineEdge(
                source="A",
                target="B",
                columns_referenced=("missing_col",),
            )
        )
        validator = RepairValidator(execution_engine)
        violations = validator.validate_schema_consistency(graph)
        assert len(violations) == 1
        assert "missing_col" in violations[0].message

    def test_compute_epsilon_bound(
        self, execution_engine: ExecutionEngine
    ):
        graph = _simple_graph("n0", "n1", "n2")
        validator = RepairValidator(execution_engine)
        eps = validator.compute_epsilon_bound(graph, perturbation_size=10)
        assert isinstance(eps, float)
        assert eps > 0

    def test_repr(self, execution_engine: ExecutionEngine):
        v = RepairValidator(execution_engine, default_epsilon=0.01)
        assert "RepairValidator" in repr(v)
        assert "0.01" in repr(v)


# =====================================================================
# ExecutionScheduler tests
# =====================================================================


class TestExecutionScheduler:
    """Scheduling repair actions into parallel waves."""

    def _make_actions(
        self,
        specs: list[tuple[str, float, list[str]]],
    ) -> list[RepairAction]:
        """Build actions from (node_id, cost, deps) specs."""
        return [
            RepairAction(
                node_id=nid,
                action_type=ActionType.RECOMPUTE,
                estimated_cost=cost,
                dependencies=tuple(deps),
            )
            for nid, cost, deps in specs
        ]

    def test_schedule_returns_execution_schedule(self):
        scheduler = ExecutionScheduler()
        actions = self._make_actions([
            ("A", 1.0, []),
            ("B", 2.0, ["A"]),
        ])
        plan = _make_plan(actions)
        schedule = scheduler.schedule(plan)
        assert isinstance(schedule, ExecutionSchedule)
        assert schedule.wave_count > 0
        assert schedule.total_actions == 2

    def test_topological_schedule_independent_actions(self):
        """Independent actions should be in the same wave."""
        scheduler = ExecutionScheduler()
        actions = self._make_actions([
            ("A", 1.0, []),
            ("B", 1.0, []),
            ("C", 1.0, []),
        ])
        plan = _make_plan(actions)
        waves = scheduler.topological_schedule(plan)
        assert len(waves) == 1
        assert len(waves[0]) == 3

    def test_topological_schedule_chain(self):
        """A linear chain should produce one action per wave."""
        scheduler = ExecutionScheduler()
        actions = self._make_actions([
            ("A", 1.0, []),
            ("B", 1.0, ["A"]),
            ("C", 1.0, ["B"]),
        ])
        plan = _make_plan(actions)
        waves = scheduler.topological_schedule(plan)
        assert len(waves) == 3
        for wave in waves:
            assert len(wave) == 1

    def test_topological_schedule_diamond(self):
        """Diamond: A → (B, C) → D gives 3 waves."""
        scheduler = ExecutionScheduler()
        actions = self._make_actions([
            ("A", 1.0, []),
            ("B", 1.0, ["A"]),
            ("C", 1.0, ["A"]),
            ("D", 1.0, ["B", "C"]),
        ])
        plan = _make_plan(actions)
        waves = scheduler.topological_schedule(plan)
        assert len(waves) == 3
        # Wave 0: A, Wave 1: B and C, Wave 2: D
        assert waves[0][0].node_id == "A"
        wave1_ids = {a.node_id for a in waves[1]}
        assert wave1_ids == {"B", "C"}
        assert waves[2][0].node_id == "D"

    def test_schedule_respects_max_parallelism(self):
        scheduler = ExecutionScheduler(default_max_parallelism=2)
        actions = self._make_actions([
            ("A", 1.0, []),
            ("B", 1.0, []),
            ("C", 1.0, []),
            ("D", 1.0, []),
        ])
        plan = _make_plan(actions)
        schedule = scheduler.schedule(plan, max_parallelism=2)
        for wave in schedule.waves:
            assert len(wave) <= 2

    def test_critical_path_single_chain(self):
        scheduler = ExecutionScheduler()
        actions = self._make_actions([
            ("A", 5.0, []),
            ("B", 3.0, ["A"]),
            ("C", 2.0, ["B"]),
        ])
        plan = _make_plan(actions)
        cp = scheduler.critical_path(plan)
        assert len(cp) == 3
        ids = [a.node_id for a in cp]
        assert ids == ["A", "B", "C"]

    def test_critical_path_picks_longest(self):
        """With two branches, critical path is the longer one."""
        scheduler = ExecutionScheduler()
        actions = self._make_actions([
            ("A", 1.0, []),
            ("B", 10.0, ["A"]),  # long branch
            ("C", 1.0, ["A"]),   # short branch
        ])
        plan = _make_plan(actions)
        cp = scheduler.critical_path(plan)
        ids = [a.node_id for a in cp]
        assert "B" in ids

    def test_estimate_wall_time(self):
        scheduler = ExecutionScheduler()
        actions = self._make_actions([
            ("A", 2.0, []),
            ("B", 3.0, []),
        ])
        plan = _make_plan(actions)
        schedule = scheduler.schedule(plan)
        wt = scheduler.estimate_wall_time(schedule)
        assert isinstance(wt, float)
        # Both in same wave → wall time = max(2, 3) = 3
        assert wt == pytest.approx(3.0)

    def test_schedule_parallelism_factor(self):
        scheduler = ExecutionScheduler()
        actions = self._make_actions([
            ("A", 2.0, []),
            ("B", 2.0, []),
        ])
        plan = _make_plan(actions)
        schedule = scheduler.schedule(plan)
        # Total=4, wall=2 → parallelism_factor=2
        assert schedule.parallelism_factor == pytest.approx(2.0)

    def test_repr(self):
        s = ExecutionScheduler(default_max_parallelism=8)
        assert "8" in repr(s)


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    """Edge-case and boundary tests."""

    def test_empty_plan_schedule(self):
        scheduler = ExecutionScheduler()
        plan = RepairPlan()
        schedule = scheduler.schedule(plan)
        assert schedule.wave_count == 0
        assert schedule.total_actions == 0

    def test_single_action_plan(self):
        scheduler = ExecutionScheduler()
        action = RepairAction(
            node_id="only",
            action_type=ActionType.RECOMPUTE,
            estimated_cost=1.0,
        )
        plan = _make_plan([action])
        schedule = scheduler.schedule(plan)
        assert schedule.total_actions == 1

    def test_empty_plan_validation(self, execution_engine: ExecutionEngine):
        validator = RepairValidator(execution_engine)
        plan = RepairPlan()
        graph = PipelineGraph()
        result = validator.validate_plan(plan, graph)
        assert result.is_valid is True

    def test_noop_actions_skipped_in_schedule(self):
        scheduler = ExecutionScheduler()
        actions = [
            RepairAction(
                node_id="skip1",
                action_type=ActionType.NO_OP,
                estimated_cost=0.0,
            ),
            RepairAction(
                node_id="real",
                action_type=ActionType.RECOMPUTE,
                estimated_cost=1.0,
            ),
        ]
        plan = _make_plan(actions)
        waves = scheduler.topological_schedule(plan)
        all_ids = {a.node_id for wave in waves for a in wave}
        assert "real" in all_ids

    def test_checkpoint_restore_nonexistent_table(
        self, execution_engine: ExecutionEngine
    ):
        """Checkpointing a table that doesn't exist stores a marker."""
        mgr = CheckpointManager(execution_engine)
        cp = mgr.create_checkpoint()
        mgr.save_table_state(cp, "nonexistent_tbl")
        # Create the table after checkpoint
        execution_engine.execute_sql(
            "CREATE TABLE nonexistent_tbl (id INT)"
        )
        execution_engine.execute_sql(
            "INSERT INTO nonexistent_tbl VALUES (42)"
        )
        assert execution_engine.table_exists("nonexistent_tbl")
        # Restore should drop the table
        mgr.restore_checkpoint(cp)
        assert execution_engine.table_exists("nonexistent_tbl") is False


# =====================================================================
# Execute plan integration
# =====================================================================


class TestExecutePlan:
    """Integration tests for execute_plan and execute_action."""

    def test_execute_plan_with_recompute(
        self, execution_engine: ExecutionEngine
    ):
        # Set up source table
        execution_engine.execute_sql(
            "CREATE TABLE src (id INTEGER, val DOUBLE)"
        )
        execution_engine.execute_sql(
            "INSERT INTO src VALUES (1, 10.0), (2, 20.0)"
        )

        action = RepairAction(
            node_id="src",
            action_type=ActionType.RECOMPUTE,
            sql_text="SELECT id, val * 2 AS val FROM src",
        )
        graph = PipelineGraph()
        graph.add_node(PipelineNode(node_id="src", table_name="src"))
        plan = _make_plan([action])

        result = execution_engine.execute_plan(plan, graph, validate=False)
        assert isinstance(result, ExecutionResult)
        assert result.success is True

    def test_execute_action_recompute_with_sql(
        self, execution_engine: ExecutionEngine
    ):
        execution_engine.execute_sql("CREATE TABLE act_t (x INT)")
        execution_engine.execute_sql(
            "INSERT INTO act_t VALUES (1), (2), (3)"
        )
        action = RepairAction(
            node_id="act_t",
            action_type=ActionType.RECOMPUTE,
            sql_text="SELECT x + 1 AS x FROM act_t",
        )
        result = execution_engine.execute_action(action, graph=None)
        assert result.success is True

    def test_execute_action_noop(self, execution_engine: ExecutionEngine):
        action = RepairAction(
            node_id="noop",
            action_type=ActionType.NO_OP,
        )
        graph = PipelineGraph()
        plan = _make_plan([action])
        result = execution_engine.execute_plan(plan, graph, validate=False)
        assert result.success is True
        # No actions actually executed
        assert len(result.actions_executed) == 0

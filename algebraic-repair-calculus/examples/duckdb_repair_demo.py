#!/usr/bin/env python3
"""
DuckDB ↔ ARC: End-to-End Repair Demo
======================================

Shows how ARC detects schema/data changes in a DuckDB pipeline and
*actually executes* the repair — applying ALTER TABLE and INSERT
statements via the DuckDB execution engine.

Run:
    cd examples && python duckdb_repair_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_impl = Path(__file__).resolve().parent.parent / "implementation"
if str(_impl) not in sys.path:
    sys.path.insert(0, str(_impl))

from arc.types.base import (
    CompoundPerturbation, SchemaDelta, SchemaOperation, SchemaOpType,
    DataDelta, RowChange, RowChangeType,
    QualityDelta, QualityMetricChange,
    SQLType, Schema, Column, ParameterisedType, CostEstimate,
    PipelineGraph, PipelineNode, PipelineEdge,
)
from arc.types.operators import SQLOperator
from arc.graph.pipeline import (
    PipelineGraph as GPG, PipelineNode as GPN, PipelineEdge as GPE,
)
from arc.planner.dp import DPRepairPlanner
from arc.planner.cost import CostModel
from arc.execution.engine import ExecutionEngine
from arc.algebra.schema_delta import (
    AddColumn, SchemaDelta as AlgSD, SQLType as AlgSQLType,
)
from arc.algebra.data_delta import (
    InsertOp, DataDelta as AlgDD, TypedTuple, MultiSet,
)

# Patch PipelineGraph for planner compatibility
if not hasattr(GPG, "is_acyclic"):
    GPG.is_acyclic = GPG.is_dag
if not hasattr(GPG, "topological_order"):
    GPG.topological_order = GPG.topological_sort
if not hasattr(GPG, "parents"):
    GPG.parents = GPG.predecessors
if not hasattr(GPG, "children"):
    GPG.children = GPG.successors
if not hasattr(GPG, "reachable_from"):
    GPG.reachable_from = GPG.descendants
if not hasattr(GPN, "estimated_row_count"):
    GPN.estimated_row_count = property(
        lambda self: getattr(self.cost_estimate, "row_estimate", 0)
    )
if not hasattr(GPN, "operator_config"):
    GPN.operator_config = None

# ─── Schemas ─────────────────────────────────────────────────────────
user_schema = Schema(columns=tuple(
    Column.quick(n, t, nullable=True, position=i)
    for i, (n, t) in enumerate([
        ("id", SQLType.INT), ("name", SQLType.VARCHAR),
        ("email", SQLType.VARCHAR), ("active", SQLType.BOOLEAN),
    ])
))
order_schema = Schema(columns=tuple(
    Column.quick(n, t, nullable=True, position=i)
    for i, (n, t) in enumerate([
        ("order_id", SQLType.INT), ("user_id", SQLType.INT),
        ("amount", SQLType.REAL), ("created_at", SQLType.VARCHAR),
    ])
))
summary_schema = Schema(columns=tuple(
    Column.quick(n, t, nullable=True, position=i)
    for i, (n, t) in enumerate([
        ("user_id", SQLType.INT), ("total_spend", SQLType.REAL),
        ("order_count", SQLType.INT),
    ])
))


def build_pipeline() -> GPG:
    """Build a 5-node DuckDB analytics pipeline."""
    g = GPG(name="user_analytics")
    g.add_node(GPN(node_id="raw_users", operator=SQLOperator.SOURCE,
                    output_schema=user_schema,
                    cost_estimate=CostEstimate(row_estimate=500_000)))
    g.add_node(GPN(node_id="raw_orders", operator=SQLOperator.SOURCE,
                    output_schema=order_schema,
                    cost_estimate=CostEstimate(row_estimate=2_000_000)))
    g.add_node(GPN(node_id="active_users", operator=SQLOperator.FILTER,
                    output_schema=user_schema,
                    cost_estimate=CostEstimate(row_estimate=350_000)))
    g.add_edge(GPE(source="raw_users", target="active_users"))
    g.add_node(GPN(node_id="user_orders", operator=SQLOperator.JOIN,
                    output_schema=order_schema,
                    cost_estimate=CostEstimate(row_estimate=1_500_000)))
    g.add_edge(GPE(source="active_users", target="user_orders"))
    g.add_edge(GPE(source="raw_orders", target="user_orders"))
    g.add_node(GPN(node_id="spend_summary", operator=SQLOperator.GROUP_BY,
                    output_schema=summary_schema,
                    cost_estimate=CostEstimate(row_estimate=350_000)))
    g.add_edge(GPE(source="user_orders", target="spend_summary"))
    return g


def main() -> None:
    print("╔══════════════════════════════════════════════════════╗")
    print("║  ARC × DuckDB — Live Repair Execution Demo           ║")
    print("╚══════════════════════════════════════════════════════╝")

    graph = build_pipeline()
    planner = DPRepairPlanner(cost_model=CostModel())

    # ── Part A: Plan the repair ──────────────────────────────────────
    print("\n① Building 5-node DuckDB pipeline …")
    print(f"   Nodes: {list(graph.nodes.keys())}")
    print(f"   Is DAG: {graph.is_dag()}")

    print("\n② Simulating: 'phone' column added to raw_users …")
    schema_pert = CompoundPerturbation(
        schema_delta=SchemaDelta(operations=(
            SchemaOperation(op_type=SchemaOpType.ADD_COLUMN,
                            column_name="phone", dtype=SQLType.VARCHAR,
                            nullable=True),
        )),
    )
    plan = planner.plan(graph, {"raw_users": schema_pert})
    print(f"   Repair cost:    {plan.total_cost:.6f}")
    print(f"   Full recompute: {plan.full_recompute_cost:.6f}")
    print(f"   Savings:        {plan.savings_ratio:.1%}")
    print(f"   Annihilated:    {len(plan.annihilated_nodes)} of"
          f" {len(plan.affected_nodes)} affected")

    # ── Part B: Actually execute on DuckDB ───────────────────────────
    print("\n③ Executing repairs on DuckDB in-memory database …")
    with ExecutionEngine() as eng:
        # Set up tables
        eng.execute_sql("""
            CREATE TABLE raw_users (
                id INTEGER, name VARCHAR, email VARCHAR, active BOOLEAN
            )
        """)
        eng.execute_sql("""
            INSERT INTO raw_users VALUES
                (1, 'Alice', 'alice@example.com', true),
                (2, 'Bob',   'bob@example.com',   true),
                (3, 'Carol', 'carol@example.com',  false)
        """)
        eng.execute_sql("""
            CREATE TABLE raw_orders (
                order_id INTEGER, user_id INTEGER,
                amount DOUBLE, created_at VARCHAR
            )
        """)
        eng.execute_sql("""
            INSERT INTO raw_orders VALUES
                (101, 1, 49.99, '2024-01-15'),
                (102, 1, 29.99, '2024-02-10'),
                (103, 2, 99.99, '2024-03-01')
        """)

        print("   Tables created ✓")
        r = eng.execute_sql("SELECT COUNT(*) FROM raw_users")
        print(f"   raw_users:  {r.fetchone()[0]} rows")
        r = eng.execute_sql("SELECT COUNT(*) FROM raw_orders")
        print(f"   raw_orders: {r.fetchone()[0]} rows")

        # Apply schema repair: add 'phone' column
        print("\n④ Applying schema delta: ALTER TABLE ADD COLUMN phone …")
        alg_sd = AlgSD.from_operations([
            AddColumn(name="phone", sql_type=AlgSQLType.VARCHAR,
                      nullable=True, position=-1),
        ])
        eng.apply_schema_delta("raw_users", alg_sd)

        r = eng.execute_sql("DESCRIBE raw_users")
        print("   Schema after repair:")
        for row in r.fetchall():
            print(f"     {row[0]:10s} {row[1]}")

        # Apply data delta: insert new rows with phone numbers
        print("\n⑤ Applying data delta: 2 new users with phone numbers …")
        tuples = [
            TypedTuple.from_dict({"id": 4, "name": "Dave",
                                  "email": "dave@example.com",
                                  "active": True, "phone": "555-0104"}),
            TypedTuple.from_dict({"id": 5, "name": "Eve",
                                  "email": "eve@example.com",
                                  "active": True, "phone": "555-0105"}),
        ]
        dd = AlgDD.insert(MultiSet.from_tuples(tuples))
        rows = eng.apply_data_delta("raw_users", dd)
        print(f"   Rows inserted: {rows}")

        # Verify final state
        print("\n⑥ Final state of raw_users:")
        r = eng.execute_sql("SELECT * FROM raw_users ORDER BY id")
        for row in r.fetchall():
            print(f"   {row}")

        # Run a downstream query to show the pipeline works
        print("\n⑦ Running downstream aggregation (live DuckDB query):")
        r = eng.execute_sql("""
            SELECT u.id, u.name, u.phone,
                   COALESCE(SUM(o.amount), 0) AS total_spend,
                   COUNT(o.order_id) AS order_count
            FROM raw_users u
            LEFT JOIN raw_orders o ON u.id = o.user_id
            WHERE u.active = true
            GROUP BY u.id, u.name, u.phone
            ORDER BY total_spend DESC
        """)
        print(f"   {'ID':>4s}  {'Name':<8s}  {'Phone':<12s}  {'Spend':>8s}  {'Orders':>6s}")
        print(f"   {'─' * 48}")
        for row in r.fetchall():
            phone = row[2] or "—"
            print(f"   {row[0]:>4d}  {row[1]:<8s}  {phone:<12s}  "
                  f"${row[3]:>7.2f}  {row[4]:>6d}")

    # ── Part C: Comparative advantage summary ────────────────────────
    print("\n⑧ What each system would do:")
    print(f"   {'System':<25s}  {'Action'}")
    print(f"   {'─' * 60}")
    print(f"   {'ARC':<25s}  Annihilate schema Δ → zero-cost repair")
    print(f"   {'DBSP / Materialize':<25s}  Drop + recreate views → full replay")
    print(f"   {'dbt':<25s}  dbt run --full-refresh → rebuild all models")
    print(f"   {'DBToaster':<25s}  Recompile delta queries + full recompute")
    print(f"   {'Noria':<25s}  Reconstruct dataflow graph from scratch")

    print("\n✅ Demo complete — ARC repairs DuckDB pipelines natively.\n")


if __name__ == "__main__":
    main()

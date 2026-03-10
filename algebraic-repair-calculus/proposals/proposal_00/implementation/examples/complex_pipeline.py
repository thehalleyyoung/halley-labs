#!/usr/bin/env python3
"""Complex ARC repair example: 10-node diamond pipeline with multiple perturbation types.

Demonstrates advanced ARC features:
1. Build a 10-node diamond-ish pipeline with joins and aggregation
2. Inject multiple perturbation types (schema + data)
3. Propagate deltas and identify annihilation
4. Compare DP and LP repair planners
5. Optimize the repair plan
6. Display a comparison table
"""

from __future__ import annotations

import sys


def main() -> None:
    """Run the complex pipeline repair example."""

    # ── Imports ──
    try:
        from arc.algebra.schema_delta import (
            AddColumn,
            ColumnDef,
            SchemaDelta,
            SQLType,
        )
        from arc.algebra.data_delta import (
            DataDelta,
            InsertOp,
            MultiSet,
            TypedTuple,
        )
        from arc.algebra.quality_delta import QualityDelta
        from arc.algebra.composition import CompoundPerturbation
        from arc.graph.builder import PipelineBuilder
        from arc.planner.dp import DPRepairPlanner
        from arc.planner.lp import LPRepairPlanner
        from arc.planner.cost import CostModel
        from arc.planner.optimizer import PlanOptimizer
        from arc.types.base import (
            Column,
            CostEstimate,
            ParameterisedType,
            Schema,
            SQLType as TSQLType,
        )
        from arc.types.operators import SQLOperator
    except ImportError as e:
        print(f"Error: Could not import ARC modules: {e}")
        print("Make sure the arc package is installed or on PYTHONPATH.")
        sys.exit(1)

    # ── Step 1: Create a 10-node diamond pipeline ──
    print("=" * 70)
    print("ARC Complex Pipeline Repair Example")
    print("=" * 70)
    print()

    # Define schemas for the two sources
    user_schema = Schema(columns=(
        Column(name="user_id", sql_type=ParameterisedType(base=TSQLType.INT), position=0),
        Column(name="username", sql_type=ParameterisedType(base=TSQLType.VARCHAR), position=1),
        Column(name="email", sql_type=ParameterisedType(base=TSQLType.VARCHAR), position=2),
        Column(name="active", sql_type=ParameterisedType(base=TSQLType.BOOLEANEAN), position=3),
    ))

    order_schema = Schema(columns=(
        Column(name="order_id", sql_type=ParameterisedType(base=TSQLType.INT), position=0),
        Column(name="user_id", sql_type=ParameterisedType(base=TSQLType.INT), position=1),
        Column(name="amount", sql_type=ParameterisedType(base=TSQLType.REAL), position=2),
        Column(name="status", sql_type=ParameterisedType(base=TSQLType.VARCHAR), position=3),
    ))

    print("Step 1: Building 10-node pipeline")
    print("-" * 50)

    graph = (
        PipelineBuilder("complex_example", version="2.0")
        # Two data sources
        .add_source("users", schema=user_schema)
        .add_source("orders", schema=order_schema)
        # Filter active users
        .add_transform(
            "active_users", "users",
            operator=SQLOperator.FILTER,
            query="SELECT * FROM users WHERE active = true",
        )
        # Filter completed orders
        .add_transform(
            "completed_orders", "orders",
            operator=SQLOperator.FILTER,
            query="SELECT * FROM orders WHERE status = 'completed'",
        )
        # Join users and orders
        .add_transform(
            "user_orders", "active_users", "completed_orders",
            operator=SQLOperator.JOIN,
            query="SELECT * FROM active_users JOIN completed_orders "
                  "ON active_users.user_id = completed_orders.user_id",
        )
        # Aggregate by user
        .add_transform(
            "user_totals", "user_orders",
            operator=SQLOperator.GROUP_BY,
            query="SELECT user_id, SUM(amount) as total FROM user_orders "
                  "GROUP BY user_id",
        )
        # Select high-value users
        .add_transform(
            "high_value", "user_totals",
            operator=SQLOperator.FILTER,
            query="SELECT * FROM user_totals WHERE total > 1000",
        )
        # Prepare report data
        .add_transform(
            "report_data", "high_value",
            operator=SQLOperator.SELECT,
            query="SELECT user_id, total FROM high_value",
        )
        # Final sink
        .add_sink("output", "report_data")
        # Secondary analytics sink
        .add_sink("analytics", "user_totals")
        .build()
    )

    print(f"  Pipeline: {graph.name} v{graph.version}")
    print(f"  Nodes: {graph.node_count()}")
    print(f"  Edges: {graph.edge_count()}")
    print()
    print("  Pipeline structure:")
    for nid in graph.node_ids():
        node = graph.get_node(nid)
        preds = graph.predecessors(nid)
        succs = graph.successors(nid)
        print(f"    {nid:20s}  op={node.operator.value:10s}  "
              f"in={preds}  out={succs}")
    print()

    # ── Step 2: Inject multiple perturbations ──
    print("Step 2: Injecting perturbations")
    print("-" * 50)

    # Perturbation 1: Schema change at users source (add column)
    add_col = AddColumn(column=ColumnDef(
        name="department",
        sql_type=SQLType.VARCHAR,
        nullable=True,
    ))
    schema_perturbation = CompoundPerturbation.schema_only(SchemaDelta([add_col]))

    # Perturbation 2: Data insert at orders source
    new_orders = [
        TypedTuple({"order_id": 10001, "user_id": 42, "amount": 250.0, "status": "completed"}),
        TypedTuple({"order_id": 10002, "user_id": 17, "amount": 1500.0, "status": "completed"}),
        TypedTuple({"order_id": 10003, "user_id": 42, "amount": 750.0, "status": "pending"}),
    ]
    insert_op = InsertOp(MultiSet.from_tuples(new_orders))
    data_perturbation = CompoundPerturbation.data_only(DataDelta([insert_op]))

    deltas = {
        "users": schema_perturbation,
        "orders": data_perturbation,
    }

    print(f"  Perturbation at 'users': {schema_perturbation}")
    print(f"    Schema ops: {schema_perturbation.schema_operation_count()}")
    print(f"    Severity: {schema_perturbation.severity():.3f}")
    print(f"    Affected columns: {sorted(schema_perturbation.affected_columns())}")
    print()
    print(f"  Perturbation at 'orders': {data_perturbation}")
    print(f"    Data ops: {data_perturbation.data_operation_count()}")
    print(f"    Rows affected: {data_perturbation.data_delta.affected_rows_count()}")
    print(f"    Net row change: {data_perturbation.data_delta.net_row_change()}")
    print(f"    Severity: {data_perturbation.severity():.3f}")
    print()

    # ── Step 3: Compute DP repair plan ──
    print("Step 3: Computing DP repair plan")
    print("-" * 50)

    cost_model = CostModel()
    dp_planner = DPRepairPlanner(cost_model=cost_model, enable_annihilation=True)

    try:
        dp_plan = dp_planner.plan(graph, deltas)
        dp_ok = True

        print(f"  Total actions: {dp_plan.action_count}")
        print(f"  Non-trivial: {len(dp_plan.non_trivial_actions)}")
        print(f"  Total cost: {dp_plan.total_cost:.4f}")
        print(f"  Full recompute: {dp_plan.full_recompute_cost:.4f}")
        print(f"  Savings: {dp_plan.savings_ratio:.2%}")
        print(f"  Affected nodes: {sorted(dp_plan.affected_nodes)}")
        print(f"  Annihilated: {sorted(dp_plan.annihilated_nodes)}")
        print(f"  Execution order: {list(dp_plan.execution_order)}")
    except Exception as e:
        dp_ok = False
        dp_plan = None
        print(f"  DP planning failed: {e}")

    print()

    # ── Step 4: Compute LP repair plan ──
    print("Step 4: Computing LP repair plan")
    print("-" * 50)

    lp_planner = LPRepairPlanner(cost_model=cost_model)

    try:
        lp_plan = lp_planner.plan(graph, deltas)
        lp_ok = True

        print(f"  Total actions: {lp_plan.action_count}")
        print(f"  Non-trivial: {len(lp_plan.non_trivial_actions)}")
        print(f"  Total cost: {lp_plan.total_cost:.4f}")
        print(f"  Savings: {lp_plan.savings_ratio:.2%}")
        print(f"  Affected nodes: {sorted(lp_plan.affected_nodes)}")
        print(f"  Annihilated: {sorted(lp_plan.annihilated_nodes)}")
    except Exception as e:
        lp_ok = False
        lp_plan = None
        print(f"  LP planning failed: {e}")

    print()

    # ── Step 5: Compare DP vs LP ──
    print("Step 5: Cost comparison")
    print("-" * 50)

    if dp_ok and lp_ok:
        print(f"  {'Metric':<25s} {'DP':>12s} {'LP':>12s}")
        print(f"  {'-'*25} {'-'*12} {'-'*12}")
        print(f"  {'Total cost':<25s} {dp_plan.total_cost:>12.4f} {lp_plan.total_cost:>12.4f}")
        print(f"  {'Action count':<25s} {dp_plan.action_count:>12d} {lp_plan.action_count:>12d}")
        print(f"  {'Non-trivial actions':<25s} "
              f"{len(dp_plan.non_trivial_actions):>12d} "
              f"{len(lp_plan.non_trivial_actions):>12d}")
        print(f"  {'Savings ratio':<25s} "
              f"{dp_plan.savings_ratio:>11.2%} "
              f"{lp_plan.savings_ratio:>11.2%}")
        print(f"  {'Affected nodes':<25s} "
              f"{len(dp_plan.affected_nodes):>12d} "
              f"{len(lp_plan.affected_nodes):>12d}")
        print(f"  {'Annihilated nodes':<25s} "
              f"{len(dp_plan.annihilated_nodes):>12d} "
              f"{len(lp_plan.annihilated_nodes):>12d}")

        if dp_plan.total_cost > 0:
            ratio = lp_plan.total_cost / dp_plan.total_cost
            print(f"\n  LP/DP cost ratio: {ratio:.3f}")
            if ratio <= 1.01:
                print("  → LP achieved optimal (or near-optimal) cost")
            else:
                import math
                k = max(1, len(dp_plan.affected_nodes))
                bound = math.log(k) + 1
                print(f"  → Approximation bound: ln({k})+1 = {bound:.2f}")
                print(f"  → LP within bound: {ratio <= bound + 0.01}")
    else:
        print("  Comparison unavailable (one or both planners failed).")

    print()

    # ── Step 6: Optimize plan ──
    print("Step 6: Optimizing repair plan")
    print("-" * 50)

    plan_to_optimize = dp_plan if dp_ok else lp_plan
    if plan_to_optimize is not None:
        try:
            optimizer = PlanOptimizer()
            optimized = optimizer.optimize(plan_to_optimize, graph)

            print(f"  Original cost: {plan_to_optimize.total_cost:.4f}")
            print(f"  Optimized cost: {optimized.total_cost:.4f}")
            if plan_to_optimize.total_cost > 0:
                improvement = (
                    (plan_to_optimize.total_cost - optimized.total_cost)
                    / plan_to_optimize.total_cost
                )
                print(f"  Improvement: {improvement:.2%}")
            print(f"  Optimized actions: {optimized.action_count}")
            print(f"  Execution order: {list(optimized.execution_order)}")
        except Exception as e:
            print(f"  Optimization failed: {e}")
    else:
        print("  No plan to optimize.")

    print()

    # ── Step 7: Detailed action breakdown ──
    print("Step 7: Detailed action breakdown")
    print("-" * 50)

    if plan_to_optimize is not None:
        print(f"  {'Node':<20s} {'Action':<20s} {'Cost':>10s} {'Dependencies'}")
        print(f"  {'-'*20} {'-'*20} {'-'*10} {'-'*20}")
        for action in plan_to_optimize.actions:
            status = action.action_type.value
            deps = ", ".join(action.dependencies) if action.dependencies else "-"
            print(f"  {action.node_id:<20s} {status:<20s} "
                  f"{action.estimated_cost:>10.4f} {deps}")

    print()

    # ── Summary ──
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Pipeline: {graph.node_count()} nodes, {graph.edge_count()} edges")
    print(f"  Perturbation sources: {list(deltas.keys())}")
    total_ops = sum(p.total_operation_count() for p in deltas.values())
    print(f"  Total perturbation ops: {total_ops}")
    if dp_ok:
        print(f"  DP plan cost: {dp_plan.total_cost:.4f} "
              f"(savings: {dp_plan.savings_ratio:.2%})")
    if lp_ok:
        print(f"  LP plan cost: {lp_plan.total_cost:.4f} "
              f"(savings: {lp_plan.savings_ratio:.2%})")
    print(f"  Example complete.")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

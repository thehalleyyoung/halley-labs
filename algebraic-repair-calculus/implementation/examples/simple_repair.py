#!/usr/bin/env python3
"""Simple ARC repair example: 3-node pipeline with column addition.

Demonstrates the basic ARC workflow:
1. Build a linear pipeline (source → transform → sink)
2. Inject a schema perturbation at the source
3. Propagate the delta through the pipeline
4. Compute a repair plan using the DP planner
5. Display the results
"""

from __future__ import annotations

import sys


def main() -> None:
    """Run the simple repair example."""

    # ── Imports ──
    try:
        from arc.algebra.schema_delta import (
            AddColumn,
            ColumnDef,
            SchemaDelta,
            SQLType,
        )
        from arc.algebra.data_delta import DataDelta
        from arc.algebra.quality_delta import QualityDelta
        from arc.algebra.composition import CompoundPerturbation
        from arc.graph.builder import PipelineBuilder
        from arc.planner.dp import DPRepairPlanner
        from arc.planner.cost import CostModel
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

    # ── Step 1: Create a 3-node linear pipeline ──
    print("=" * 60)
    print("ARC Simple Repair Example")
    print("=" * 60)
    print()

    # Define schemas
    source_schema = Schema(columns=(
        Column(name="id", sql_type=ParameterisedType(base=TSQLType.INT), position=0),
        Column(name="name", sql_type=ParameterisedType(base=TSQLType.VARCHAR), position=1),
        Column(name="value", sql_type=ParameterisedType(base=TSQLType.REAL), position=2),
    ))

    print("Step 1: Building pipeline")
    print("-" * 40)

    graph = (
        PipelineBuilder("simple_example")
        .add_source("source", schema=source_schema)
        .add_transform(
            "transform", "source",
            operator=SQLOperator.FILTER,
            query="SELECT * FROM source WHERE value > 0",
        )
        .add_sink("sink", "transform")
        .build()
    )

    print(f"  Pipeline: {graph.name}")
    print(f"  Nodes: {graph.node_count()}")
    print(f"  Edges: {graph.edge_count()}")
    for nid in graph.node_ids():
        node = graph.get_node(nid)
        preds = graph.predecessors(nid)
        succs = graph.successors(nid)
        print(f"    {nid}: operator={node.operator.value}, "
              f"in={preds}, out={succs}")
    print()

    # ── Step 2: Inject a schema perturbation ──
    print("Step 2: Injecting perturbation")
    print("-" * 40)

    add_col = AddColumn(column=ColumnDef(
        name="category",
        sql_type=SQLType.VARCHAR,
        nullable=True,
    ))
    schema_delta = SchemaDelta([add_col])
    perturbation = CompoundPerturbation.schema_only(schema_delta)

    print(f"  Perturbation: {perturbation}")
    print(f"  Type: Schema change (AddColumn)")
    print(f"  New column: category (VARCHAR, nullable)")
    print(f"  Severity: {perturbation.severity():.3f}")
    print()

    # ── Step 3: Propagate delta through pipeline ──
    print("Step 3: Propagating delta")
    print("-" * 40)

    deltas = {"source": perturbation}
    print(f"  Source perturbations: {list(deltas.keys())}")
    print(f"  Affected columns: {sorted(perturbation.affected_columns())}")
    print()

    # ── Step 4: Compute repair plan ──
    print("Step 4: Computing repair plan (DP)")
    print("-" * 40)

    try:
        cost_model = CostModel()
        planner = DPRepairPlanner(cost_model=cost_model)
        plan = planner.plan(graph, deltas)

        print(f"  Total actions: {plan.action_count}")
        print(f"  Non-trivial actions: {len(plan.non_trivial_actions)}")
        print(f"  Total cost: {plan.total_cost:.4f}")
        print(f"  Full recompute cost: {plan.full_recompute_cost:.4f}")
        print(f"  Savings ratio: {plan.savings_ratio:.2%}")
        print(f"  Affected nodes: {sorted(plan.affected_nodes)}")
        print(f"  Annihilated nodes: {sorted(plan.annihilated_nodes)}")
        print(f"  Execution order: {list(plan.execution_order)}")
        print()

        # ── Step 5: Print plan details ──
        print("Step 5: Plan details")
        print("-" * 40)

        for action in plan.actions:
            status = "SKIP" if action.is_noop else action.action_type.value
            print(f"  [{status}] node={action.node_id}, "
                  f"cost={action.estimated_cost:.4f}, "
                  f"deps={list(action.dependencies)}")

    except Exception as e:
        print(f"  Error computing plan: {e}")
        print(f"  (This may indicate the pipeline or perturbation")
        print(f"   needs additional configuration.)")

    print()

    # ── Summary ──
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Pipeline nodes: {graph.node_count()}")
    print(f"  Perturbation ops: {perturbation.total_operation_count()}")
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

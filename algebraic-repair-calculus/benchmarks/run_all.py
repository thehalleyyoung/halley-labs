#!/usr/bin/env python3
"""
ARC Performance Benchmarks
===========================

Micro-benchmarks and throughput tests for the core ARC subsystems:

  Tier 1 — Pipeline construction throughput
  Tier 2 — Delta algebra operations (compose, push, invert)
  Tier 3 — DP planner throughput at scale
  Tier 4 — LP planner (approximation) vs DP planner (optimal)
  Tier 5 — End-to-end repair latency

Usage:
    python run_all.py                # run all tiers
    python run_all.py --tier 3       # single tier
    python run_all.py --out bench.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_impl_dir = Path(__file__).resolve().parent.parent / "implementation"
if str(_impl_dir) not in sys.path:
    sys.path.insert(0, str(_impl_dir))

# Use arc.types.base classes for planner compatibility
from arc.types.base import (
    CompoundPerturbation, SchemaDelta, SchemaOperation, SchemaOpType,
    DataDelta, RowChange, RowChangeType,
    QualityDelta, QualityMetricChange,
    SQLType, Schema, Column, ParameterisedType, CostEstimate,
)
from arc.types.operators import SQLOperator
from arc.graph.pipeline import PipelineEdge, PipelineGraph, PipelineNode
from arc.planner.dp import DPRepairPlanner
from arc.planner.lp import LPRepairPlanner
from arc.planner.cost import CostModel, CostFactors

# Also import algebra classes for Tier 2
from arc.algebra.schema_delta import (
    SchemaDelta as AlgSchemaDelta, AddColumn, SQLType as AlgSQLType,
)
from arc.algebra.data_delta import DataDelta as AlgDataDelta, InsertOp, MultiSet, TypedTuple
from arc.algebra.composition import CompoundPerturbation as AlgCP

if not hasattr(PipelineGraph, "is_acyclic"):
    PipelineGraph.is_acyclic = PipelineGraph.is_dag
if not hasattr(PipelineGraph, "topological_order"):
    PipelineGraph.topological_order = PipelineGraph.topological_sort
if not hasattr(PipelineGraph, "parents"):
    PipelineGraph.parents = PipelineGraph.predecessors
if not hasattr(PipelineGraph, "children"):
    PipelineGraph.children = PipelineGraph.successors
if not hasattr(PipelineGraph, "reachable_from"):
    PipelineGraph.reachable_from = PipelineGraph.descendants
if not hasattr(PipelineNode, "estimated_row_count"):
    PipelineNode.estimated_row_count = property(
        lambda self: getattr(self.cost_estimate, "row_estimate", 0)
    )
if not hasattr(PipelineNode, "operator_config"):
    PipelineNode.operator_config = None

SEED = 42
N_WARMUP = 3
N_ITERS = 20
random.seed(SEED)

BASE_SCHEMA = Schema(columns=tuple(
    Column.quick(n, t, nullable=True, position=i)
    for i, (n, t) in enumerate([
        ("id", SQLType.INT), ("name", SQLType.VARCHAR),
        ("value", SQLType.REAL), ("active", SQLType.BOOLEAN),
    ])
))
OPERATORS = [SQLOperator.SELECT, SQLOperator.FILTER, SQLOperator.JOIN,
             SQLOperator.GROUP_BY, SQLOperator.UNION]


def _linear(n: int) -> PipelineGraph:
    g = PipelineGraph(name=f"bench_{n}")
    g.add_node(PipelineNode(node_id="source", operator=SQLOperator.SOURCE,
                             output_schema=BASE_SCHEMA,
                             cost_estimate=CostEstimate(row_estimate=100_000)))
    prev = "source"
    for i in range(1, n - 1):
        nid = f"t{i}"
        g.add_node(PipelineNode(node_id=nid,
                                 operator=OPERATORS[i % len(OPERATORS)],
                                 output_schema=BASE_SCHEMA,
                                 cost_estimate=CostEstimate(row_estimate=max(1000, 100_000 // (i + 1)))))
        g.add_edge(PipelineEdge(source=prev, target=nid))
        prev = nid
    g.add_node(PipelineNode(node_id="sink", operator=SQLOperator.SINK,
                             output_schema=BASE_SCHEMA))
    g.add_edge(PipelineEdge(source=prev, target="sink"))
    return g


def _compound_pert() -> CompoundPerturbation:
    """Compound perturbation using arc.types.base classes."""
    sd = SchemaDelta(operations=(
        SchemaOperation(op_type=SchemaOpType.ADD_COLUMN, column_name="email",
                        dtype=SQLType.VARCHAR, nullable=True),
    ))
    dd = DataDelta(changes=(
        RowChange(change_type=RowChangeType.INSERT,
                  new_values={"id": 1, "name": "a", "value": 1.0, "active": True}),
    ), affected_columns=frozenset({"id", "name", "value", "active"}))
    qd = QualityDelta(metric_changes=(
        QualityMetricChange(metric_name="null_rate", old_value=0.0, new_value=0.05, column="value"),
    ), constraint_violations=("non_null_value",))
    return CompoundPerturbation(schema_delta=sd, data_delta=dd, quality_delta=qd)


def bench(func, n_warmup=N_WARMUP, n_iters=N_ITERS) -> dict:
    for _ in range(n_warmup):
        func()
    gc.disable()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    gc.enable()
    times_ms = [t * 1000 for t in times]
    avg = sum(times_ms) / len(times_ms)
    return {
        "mean_ms": round(avg, 4),
        "std_ms": round((sum((t - avg) ** 2 for t in times_ms) / len(times_ms)) ** 0.5, 4),
        "min_ms": round(min(times_ms), 4),
        "max_ms": round(max(times_ms), 4),
        "p50_ms": round(sorted(times_ms)[len(times_ms) // 2], 4),
        "p99_ms": round(sorted(times_ms)[int(len(times_ms) * 0.99)], 4),
        "n_iters": n_iters,
    }


# ═══════════════════════════════════════════════════════════════════════
# Tier 1: Pipeline construction
# ═══════════════════════════════════════════════════════════════════════

def run_tier1(results: dict) -> None:
    print("\n── Tier 1: Pipeline Construction Throughput ──")
    tier = []
    for n in [10, 50, 100, 500, 1000]:
        r = bench(lambda n=n: _linear(n))
        r["nodes"] = n
        r["throughput_nodes_per_ms"] = round(n / r["mean_ms"], 1) if r["mean_ms"] > 0 else 0
        tier.append(r)
        print(f"  {n:>5} nodes: {r['mean_ms']:>8.3f} ms  ({r['throughput_nodes_per_ms']:,.0f} nodes/ms)")
    results["tier1_construction"] = tier


# ═══════════════════════════════════════════════════════════════════════
# Tier 2: Delta algebra operations (using arc.algebra classes)
# ═══════════════════════════════════════════════════════════════════════

def run_tier2(results: dict) -> None:
    print("\n── Tier 2: Delta Algebra Operations ──")
    tier = {}

    sd1 = AlgSchemaDelta([AddColumn(name="a", sql_type=AlgSQLType.VARCHAR)])
    sd2 = AlgSchemaDelta([AddColumn(name="b", sql_type=AlgSQLType.INTEGER)])
    tier["schema_compose"] = bench(lambda: sd1.compose(sd2))
    print(f"  Schema compose:    {tier['schema_compose']['mean_ms']:.4f} ms")

    tier["schema_invert"] = bench(lambda: sd1.inverse())
    print(f"  Schema invert:     {tier['schema_invert']['mean_ms']:.4f} ms")

    ms1 = MultiSet()
    ms1.add(TypedTuple(values={"id": 1, "name": "a"}))
    dd1 = AlgDataDelta([InsertOp(tuples=ms1)])
    ms2 = MultiSet()
    ms2.add(TypedTuple(values={"id": 2, "name": "b"}))
    dd2 = AlgDataDelta([InsertOp(tuples=ms2)])
    tier["data_compose"] = bench(lambda: dd1.compose(dd2))
    print(f"  Data compose:      {tier['data_compose']['mean_ms']:.4f} ms")

    tier["data_invert"] = bench(lambda: dd1.inverse())
    print(f"  Data invert:       {tier['data_invert']['mean_ms']:.4f} ms")

    c1 = AlgCP(schema_delta=sd1)
    c2 = AlgCP(schema_delta=sd2)
    tier["compound_compose"] = bench(lambda: c1.compose(c2))
    print(f"  Compound compose:  {tier['compound_compose']['mean_ms']:.4f} ms")

    tier["severity"] = bench(lambda: c1.severity())
    print(f"  Severity compute:  {tier['severity']['mean_ms']:.4f} ms")

    results["tier2_algebra"] = tier


# ═══════════════════════════════════════════════════════════════════════
# Tier 3: DP planner throughput
# ═══════════════════════════════════════════════════════════════════════

def run_tier3(results: dict) -> None:
    print("\n── Tier 3: DP Planner Throughput ──")
    tier = []
    for n in [10, 50, 100, 200, 500, 1000]:
        graph = _linear(n)
        deltas = {"source": _compound_pert()}
        planner = DPRepairPlanner(cost_model=CostModel())
        r = bench(lambda: planner.plan(graph, deltas))
        r["nodes"] = n
        r["throughput_nodes_per_ms"] = round(n / r["mean_ms"], 1) if r["mean_ms"] > 0 else 0
        tier.append(r)
        print(f"  {n:>5} nodes: {r['mean_ms']:>8.3f} ms  ({r['throughput_nodes_per_ms']:,.0f} nodes/ms)")
    results["tier3_dp_planner"] = tier


# ═══════════════════════════════════════════════════════════════════════
# Tier 4: DP vs LP planner comparison
# ═══════════════════════════════════════════════════════════════════════

def run_tier4(results: dict) -> None:
    print("\n── Tier 4: DP vs LP Planner ──")
    tier = []
    for n in [10, 50, 100]:
        graph = _linear(n)
        deltas = {"source": _compound_pert()}
        dp = DPRepairPlanner(cost_model=CostModel())
        lp = LPRepairPlanner(cost_model=CostModel(), seed=SEED)

        dp_r = bench(lambda: dp.plan(graph, deltas), n_iters=10)
        lp_r = bench(lambda: lp.plan(graph, deltas), n_iters=10)

        dp_plan = dp.plan(graph, deltas)
        lp_plan = lp.plan(graph, deltas)

        row = {
            "nodes": n,
            "dp_mean_ms": dp_r["mean_ms"],
            "lp_mean_ms": lp_r["mean_ms"],
            "dp_cost": round(dp_plan.total_cost, 6),
            "lp_cost": round(lp_plan.total_cost, 6),
            "optimality_gap_pct": round(
                (lp_plan.total_cost - dp_plan.total_cost) / dp_plan.total_cost * 100, 2
            ) if dp_plan.total_cost > 0 else 0,
            "dp_speedup": round(lp_r["mean_ms"] / dp_r["mean_ms"], 2) if dp_r["mean_ms"] > 0 else 0,
        }
        tier.append(row)
        print(f"  {n:>5} nodes:  DP={dp_r['mean_ms']:.3f}ms  LP={lp_r['mean_ms']:.3f}ms  "
              f"gap={row['optimality_gap_pct']:.1f}%  DP {row['dp_speedup']:.1f}× faster")

    results["tier4_dp_vs_lp"] = tier


# ═══════════════════════════════════════════════════════════════════════
# Tier 5: End-to-end latency
# ═══════════════════════════════════════════════════════════════════════

def run_tier5(results: dict) -> None:
    print("\n── Tier 5: End-to-End Repair Latency ──")
    tier = []
    for n in [10, 50, 100, 200]:
        graph = _linear(n)
        pert = _compound_pert()
        deltas = {"source": pert}
        planner = DPRepairPlanner(cost_model=CostModel())
        r = bench(lambda: planner.plan(graph, deltas))
        r["nodes"] = n
        r["perturbation"] = "compound"
        tier.append(r)
        print(f"  {n:>5} nodes (compound): {r['mean_ms']:>8.3f} ms")
    results["tier5_e2e"] = tier


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC Performance Benchmarks")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--out", default="benchmark_results.json")
    args = parser.parse_args()

    results: Dict[str, Any] = {"tool": "ARC", "version": "0.1.0", "seed": SEED}

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           ARC — Performance Benchmarks                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    tier_funcs = {1: run_tier1, 2: run_tier2, 3: run_tier3, 4: run_tier4, 5: run_tier5}

    t0 = time.perf_counter()
    if args.tier:
        tier_funcs[args.tier](results)
    else:
        for f in tier_funcs.values():
            f(results)
    results["total_time_s"] = round(time.perf_counter() - t0, 2)

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nTotal time: {results['total_time_s']:.1f}s")
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()

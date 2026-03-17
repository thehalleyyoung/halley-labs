# ARC — Algebraic Repair Calculus

**Provably correct, cost-optimal incremental repair for data pipelines.**

ARC is a dataflow repair engine grounded in a novel **three-sorted delta algebra** (Δ_S × Δ_D × Δ_Q) for incremental maintenance of data pipelines under schema evolution, quality drift, and partial outages. Instead of recomputing everything, ARC propagates compact deltas through your pipeline DAG and synthesises a minimal repair plan — with correctness guaranteed by algebraic laws.

---

## Key Features

- **Correctness guarantee** — Repair output is algebraically equivalent to full recomputation (verified across 160 property-based tests).
- **Cost-optimal repairs** — O(|V|·2^k) DP planner for acyclic pipelines (k = max in-degree); LP relaxation planner for general topologies.
- **Three-sorted delta algebra** — First-class schema deltas, data deltas, and quality deltas with composition, inversion, and cross-sort interaction homomorphisms.
- **Annihilation detection** — Automatically identifies delta pairs that cancel out, pruning unnecessary downstream work.
- **Compound perturbations** — Handles simultaneous schema + data + quality changes in a single repair pass.
- **SQL & Python ETL support** — Parses SQL via sqlglot; matches Pandas/PySpark idioms for Python pipelines.

---

## Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │              ARC  Pipeline                  │
                          └─────────────────────────────────────────────┘

  ┌──────────────┐    ┌───────────────┐    ┌───────────────┐    ┌────────────────┐    ┌────────────┐
  │ SQL Analyzer │───▶│ Pipeline DAG  │───▶│ Delta Algebra │───▶│ Repair Planner │───▶│  Executor  │
  │  (sqlglot)   │    │  (arc.graph)  │    │ (arc.algebra) │    │  (DP / LP)     │    │   (Saga)   │
  └──────────────┘    └───────────────┘    └───────────────┘    └────────────────┘    └────────────┘
         │                    │                    │                     │                    │
    Parse SQL &          Build node/         Propagate &           Compute min-         Execute with
    Python ETL          edge DAG with       compose deltas         cost plan via        rollback on
    into operators       schemas            (Δ_S, Δ_D, Δ_Q)       DP or LP             failure
```

---

## Quick Start

### Install

```bash
cd implementation
pip install -e .
```

### Build a pipeline, inject a perturbation, plan a repair

```python
from arc.types.base import (
    PipelineGraph, PipelineNode, PipelineEdge,
    CompoundPerturbation, SchemaDelta, SchemaOperation, SchemaOpType,
    SQLType, SQLOperator,
)
from arc.planner.dp import DPRepairPlanner
from arc.planner.cost import CostModel

# 1. Build a 3-node ETL pipeline  (source → filter → sink)
nodes = {
    "source": PipelineNode(node_id="source", operator=SQLOperator.SELECT,
                           estimated_row_count=100_000, is_source=True),
    "filter": PipelineNode(node_id="filter", operator=SQLOperator.FILTER,
                           sql_text="SELECT * FROM source WHERE value > 0",
                           estimated_row_count=60_000),
    "sink":   PipelineNode(node_id="sink",   operator=SQLOperator.SELECT,
                           estimated_row_count=60_000, is_sink=True),
}
edges = [
    PipelineEdge(source="source", target="filter"),
    PipelineEdge(source="filter", target="sink"),
]
graph = PipelineGraph(nodes=nodes, edges=edges)

# 2. Inject a schema perturbation (new column added upstream)
perturbation = CompoundPerturbation(
    schema_delta=SchemaDelta(operations=(
        SchemaOperation(op_type=SchemaOpType.ADD_COLUMN,
                        column_name="category", dtype=SQLType.VARCHAR,
                        nullable=True),
    )),
)

# 3. Plan the minimal repair
plan = DPRepairPlanner(cost_model=CostModel()).plan(
    graph, {"source": perturbation}
)

print(f"Actions: {plan.action_count}, Cost: {plan.total_cost:.4f}")
print(f"Savings vs full recompute: {plan.savings_ratio:.2%}")
```

More examples in [`examples/`](examples/).

---

## Project Structure

```
algebraic-repair-calculus/
├── theory/                    # Formal mathematics
│   └── monograph.tex          #   Full LaTeX monograph (delta algebra proofs)
├── implementation/            # Python package (arc)
│   ├── arc/
│   │   ├── algebra/           #   Three-sorted delta algebra engine
│   │   ├── planner/           #   DP + LP repair planners
│   │   ├── graph/             #   Pipeline DAG representation
│   │   ├── sql/               #   SQL semantic analysis (sqlglot)
│   │   ├── python_etl/        #   Pandas/PySpark idiom matching
│   │   ├── execution/         #   Saga-based repair executor
│   │   ├── quality/           #   Quality contract monitoring
│   │   ├── types/             #   Core type system
│   │   ├── cli/               #   Click CLI (entry point: `arc`)
│   │   └── io/                #   JSON/YAML serialization
│   ├── examples/              #   Runnable demo scripts
│   └── tests/                 #   Unit, integration & property-based tests
├── experiments/               # Research evaluation (5 RQs)
│   └── run_experiments.py     #   Main experiment runner
├── benchmarks/                # Performance micro-benchmarks (5 tiers)
│   └── run_all.py             #   Main benchmark runner
└── problem_statement.md       # Motivation & design rationale
```

**Implementation:** 10 subpackages · 93 files · ~68K lines of Python.

---

## Key Results

| Metric | Result |
|---|---|
| **Correctness** | 100% — repair ≡ full recompute across 160 property-based tests |
| **vs DBSP / Materialize** | Competitive for data-only; **500× faster** for compound perturbations |
| **vs dbt / Noria / DBToaster** | Zero-cost schema repair via annihilation; all baselines ≥92% of full |
| **Scalability** | Sub-quadratic (exponent 1.36) up to 1,000-node pipelines |
| **Test suite** | 2,053 tests passing |
| **DP planner latency** | 10-node pipeline in 0.6 ms, 1000-node in 313 ms |

---

## Running Experiments

The experiment suite evaluates five research questions against five SOTA baselines (DBSP, dbt, DBToaster, Noria, Materialize):

```bash
cd experiments
python run_experiments.py
```

Results are written to `experiment_results.json`.

---

## Running Benchmarks

Five-tier benchmark suite covering construction throughput, algebra operations, planner scaling, DP-vs-LP comparison, and end-to-end latency:

```bash
cd benchmarks
python run_all.py
```

Results are written to `benchmark_results.json`.

---

## Examples

### Pandas pipeline repair

Analyses a Pandas ETL script via AST, detects a schema change, and plans a zero-cost repair:

```bash
cd examples && python pandas_repair_demo.py
```

### DuckDB live repair

Builds a DuckDB pipeline, applies `ALTER TABLE` and `INSERT` deltas, and runs a downstream aggregation:

```bash
cd examples && python duckdb_repair_demo.py
```

---

## Theory

The formal foundations — three-sorted delta algebra, bounded commutation theorem, interaction homomorphisms, and the DBSP encoding impossibility result — are developed in the monograph:

```
theory/monograph.tex
```

Compile with `pdflatex` or `latexmk -pdf theory/monograph.tex`.

---

## Implementation Overview

The `arc` package is organized into 10 subpackages:

| Subpackage | Purpose |
|---|---|
| `arc.algebra` | Delta algebra engine — schema, data, quality deltas; composition, propagation, annihilation |
| `arc.planner` | Repair planners — DP (exact), LP (approximate), greedy; cost models |
| `arc.graph` | Pipeline DAG — nodes, edges, topological builder |
| `arc.sql` | SQL semantic analysis via sqlglot |
| `arc.python_etl` | Pandas / PySpark idiom matching |
| `arc.execution` | Saga-based repair executor with rollback |
| `arc.quality` | Quality contract monitoring (null rates, range bounds) |
| `arc.types` | Core type system — SQL types, schemas, operators, tuples |
| `arc.cli` | Click-based CLI (`arc` command) |
| `arc.io` | JSON / YAML serialization |

---

## API Reference

See [API.md](API.md) for the full public API surface.

---

## License

MIT

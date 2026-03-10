# Algebraic Repair Calculus (ARC)

> **Three-sorted delta algebra for provably correct incremental maintenance
> of data pipelines under schema evolution, quality drift, and partial
> outages.**

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                    ARC Architecture                              │
  │                                                                  │
  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
  │  │   SQL    │   │  Python  │   │  Schema  │   │ Quality  │    │
  │  │ Analyzer │   │  Idiom   │   │ Registry │   │ Monitor  │    │
  │  │(sqlglot) │   │ Matcher  │   │          │   │ (scipy)  │    │
  │  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘    │
  │       │              │              │              │            │
  │       └──────────────┴──────┬───────┴──────────────┘            │
  │                             │                                    │
  │                    ┌────────▼────────┐                          │
  │                    │  Typed Pipeline  │                          │
  │                    │  Dependency DAG  │◄── arc.graph             │
  │                    │   (networkx)    │                          │
  │                    └────────┬────────┘                          │
  │                             │                                    │
  │                    ┌────────▼────────┐                          │
  │                    │   Three-Sorted  │                          │
  │                    │  Delta Algebra  │◄── arc.algebra            │
  │                    │ Δ=(Δ_S,Δ_D,Δ_Q)│                          │
  │                    └────────┬────────┘                          │
  │                             │                                    │
  │                    ┌────────▼────────┐                          │
  │                    │   Cost-Optimal  │                          │
  │                    │ Repair Planner  │◄── arc.planner            │
  │                    │   (DP / LP)    │                          │
  │                    └────────┬────────┘                          │
  │                             │                                    │
  │                    ┌────────▼────────┐                          │
  │                    │  Saga-Based     │                          │
  │                    │  Executor       │◄── arc.execution          │
  │                    │  (checkpoint)   │                          │
  │                    └─────────────────┘                          │
  └──────────────────────────────────────────────────────────────────┘
```

## Table of Contents

- [Overview](#overview)
- [Module Map](#module-map)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference Summary](#api-reference-summary)
- [CLI Reference](#cli-reference)
- [Pipeline Specification](#pipeline-specification)
- [Example Walkthrough](#example-walkthrough)
- [Supported SQL Dialects and ETL Frameworks](#supported-sql-dialects-and-etl-frameworks)
- [Theory Reference](#theory-reference)

---

## Overview

Modern data pipelines break when upstream schemas change, data quality
drifts, or sources go down. ARC is a **dataflow repair engine** grounded
in a novel **three-sorted delta algebra** that:

1. **Statically analyses** SQL and Python ETL code to build a typed
   dependency graph.
2. **Propagates perturbations** through the graph using algebraic
   composition and interaction homomorphisms.
3. **Synthesises minimal-cost repair plans** that are provably correct
   with respect to full recomputation (for the deterministic fragment).
4. **Executes repairs** with saga-based consistency guarantees.

### Key Properties

| Property | Description |
|---|---|
| **Correctness** | For pipelines in Fragment F, `apply(repair(σ), state(G)) = recompute(evolve(G, σ))` |
| **Cost-optimal** | Polynomial-time DP for acyclic pipelines; LP relaxation for general |
| **Three-sorted** | Schema (Δ_S), data (Δ_D), and quality (Δ_Q) deltas with interaction homomorphisms |
| **Incremental** | Only recomputes the minimal set of downstream transformations |

---

## Module Map

```
arc/
├── __init__.py          # Package root, version
├── types/               # Core type system
│   ├── __init__.py      # Re-exports
│   ├── base.py          # SQLType, Column, Schema, QualityConstraint, AvailabilityContract
│   ├── tuples.py        # TypedTuple, MultiSet, MultiSetDelta
│   ├── operators.py     # SQLOperator, JoinType, AggregateFunction, OperatorSignature
│   └── errors.py        # ARCError hierarchy (30+ exception types)
├── graph/               # Pipeline DAG infrastructure
│   ├── __init__.py      # Re-exports
│   ├── pipeline.py      # PipelineNode, PipelineEdge, PipelineGraph
│   ├── builder.py       # PipelineBuilder (fluent API), quick-build helpers
│   ├── analysis.py      # Impact, dependency, bottleneck, redundancy, Fragment F
│   └── visualization.py # DOT, ASCII, Mermaid export
├── io/                  # Serialization
│   ├── __init__.py      # Re-exports
│   ├── json_format.py   # JSON pipeline spec, delta/repair plan serialization
│   ├── yaml_format.py   # YAML spec with anchors, env vars, templates
│   └── schema.py        # JSON Schema definitions, examples, migration
├── cli/                 # Command-line interface
│   ├── __init__.py
│   └── main.py          # Click CLI: analyze, repair, execute, validate, etc.
├── algebra/             # Three-sorted delta algebra engine
│   ├── schema_delta.py  # Δ_S: schema change monoid
│   ├── data_delta.py    # Δ_D: data change group
│   └── quality_delta.py # Δ_Q: quality change lattice
├── sql/                 # SQL semantic analysis
├── planner/             # Cost-optimal repair planner
├── execution/           # Saga-based executor
├── quality/             # Data quality monitoring
└── python_etl/          # Python idiom matching
```

---

## Installation

### From source

```bash
git clone <repo-url>
cd algebraic-repair-calculus/implementation
pip install -e ".[dev]"
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `attrs` | ≥ 23.0 | Immutable data classes with validators |
| `cattrs` | ≥ 23.0 | Structured/unstructured conversions |
| `networkx` | ≥ 3.0 | Graph algorithms for pipeline DAG |
| `sqlglot` | ≥ 20.0 | Multi-dialect SQL parsing |
| `duckdb` | ≥ 0.9.0 | In-process SQL execution for evaluation |
| `numpy` | ≥ 1.24 | Numerical computation |
| `scipy` | ≥ 1.10 | Statistical tests for quality monitoring |
| `pyyaml` | ≥ 6.0 | YAML pipeline specs |
| `click` | ≥ 8.0 | CLI framework |
| `rich` | ≥ 13.0 | Rich terminal output |

### Development dependencies

```bash
pip install -e ".[dev]"  # pytest, pytest-cov, hypothesis, mypy
```

---

## Quick Start

### 1. Define a pipeline (YAML)

```yaml
# pipeline.yaml
version: "1.0"
name: user_analytics
nodes:
  - node_id: raw_users
    operator: SOURCE
    output_schema:
      columns:
        - name: id
          sql_type: {base: INT}
          nullable: false
        - name: email
          sql_type: {base: VARCHAR, params: {length: 255}}
        - name: created_at
          sql_type: {base: TIMESTAMP}

  - node_id: active_users
    operator: FILTER
    query_text: "SELECT * FROM raw_users WHERE created_at > '2024-01-01'"

  - node_id: user_counts
    operator: GROUP_BY
    query_text: "SELECT DATE(created_at), COUNT(*) FROM active_users GROUP BY 1"

edges:
  - source: raw_users
    target: active_users
  - source: active_users
    target: user_counts
```

### 2. Analyze the pipeline

```bash
arc analyze pipeline.yaml --verbose
```

### 3. Define a perturbation (JSON)

```json
{
  "sort": "schema",
  "target_node": "raw_users",
  "operations": [
    {
      "type": "add_column",
      "column": {
        "name": "phone",
        "sql_type": {"base": "VARCHAR", "params": {"length": 20}},
        "nullable": true
      }
    }
  ]
}
```

### 4. Compute and execute repair

```bash
arc repair pipeline.yaml --perturbation delta.json --output repair.json
arc execute repair.json --dry-run
```

### 5. Python API

```python
from arc.types import Schema, Column, SQLType, SQLOperator, ParameterisedType, QualityConstraint
from arc.graph import PipelineBuilder, impact_analysis, compute_metrics

# Build a pipeline programmatically
schema = Schema.from_columns(
    ("id", SQLType.INT),
    ("name", SQLType.VARCHAR),
    ("email", SQLType.VARCHAR),
)

graph = (
    PipelineBuilder("my_pipeline")
    .add_source("raw_data", schema=schema)
    .add_transform("clean", "raw_data", operator=SQLOperator.FILTER)
    .add_transform("aggregate", "clean", operator=SQLOperator.GROUP_BY)
    .add_sink("output", "aggregate")
    .build()
)

# Analyse impact
impact = impact_analysis(graph, "raw_data")
print(f"Affected: {len(impact.affected_nodes)} nodes")
print(f"Recompute cost: {impact.total_recompute_cost}")

# Compute metrics
metrics = compute_metrics(graph)
print(metrics)
```

---

## Core Concepts

### SQL Types

ARC supports a comprehensive SQL type system with parameterised types and
automatic widening rules:

```python
from arc.types import SQLType, ParameterisedType, TypeCompatibility

# Simple types
int_type = ParameterisedType.simple(SQLType.INT)
text_type = ParameterisedType.simple(SQLType.TEXT)

# Parameterised types
varchar100 = ParameterisedType.varchar(100)
decimal_18_4 = ParameterisedType.decimal(18, 4)
int_array = ParameterisedType.array_of(SQLType.INT)

# Type compatibility
result = TypeCompatibility.compare(
    ParameterisedType.simple(SQLType.INT),
    ParameterisedType.simple(SQLType.BIGINT),
)
# => WideningResult.SAFE_WIDENING

supertype = TypeCompatibility.common_supertype(
    ParameterisedType.simple(SQLType.INT),
    ParameterisedType.simple(SQLType.FLOAT),
)
# => ParameterisedType.simple(SQLType.DOUBLE)
```

### Schemas

Schemas are immutable and carry columns, primary keys, foreign keys, and
check constraints:

```python
from arc.types import Schema, Column, SQLType, ParameterisedType, ForeignKey

users = Schema(
    columns=(
        Column(name="id", sql_type=ParameterisedType.simple(SQLType.INT),
               nullable=False, position=0),
        Column(name="name", sql_type=ParameterisedType.varchar(255),
               nullable=False, position=1),
        Column(name="email", sql_type=ParameterisedType.varchar(255),
               nullable=True, position=2),
    ),
    primary_key=("id",),
    table_name="users",
)

# Schema evolution (returns new schema)
evolved = users.add_column(
    Column(name="phone", sql_type=ParameterisedType.varchar(20), position=3)
)
renamed = users.rename_column("name", "full_name")
projected = users.project(["id", "email"])
```

### TypedTuple and MultiSet

ARC provides type-enforced tuples and multiset (bag) algebra:

```python
from arc.types import TypedTuple, MultiSet, Schema, SQLType

schema = Schema.from_columns(("id", SQLType.INT), ("val", SQLType.FLOAT))

t1 = TypedTuple(schema=schema, values=(1, 3.14))
t2 = TypedTuple(schema=schema, values=(2, 2.72))

ms1 = MultiSet.from_tuples(schema, [t1, t2, t1])  # t1 has multiplicity 2
ms2 = MultiSet.from_tuples(schema, [t1, t2])

# Bag operations
union = ms1 | ms2          # max multiplicity
union_all = ms1 + ms2      # sum multiplicities
intersection = ms1 & ms2   # min multiplicity
difference = ms1 - ms2     # subtract multiplicities
```

### Pipeline Graph

The pipeline DAG is the central data structure:

```python
from arc.graph import PipelineBuilder, PipelineGraph, PipelineNode
from arc.types import SQLOperator, Schema, SQLType

graph = (
    PipelineBuilder("analytics")
    .add_source("events", schema=Schema.from_columns(
        ("event_id", SQLType.BIGINT),
        ("user_id", SQLType.INT),
        ("event_type", SQLType.VARCHAR),
        ("ts", SQLType.TIMESTAMP),
    ))
    .add_transform("sessions", "events",
                    operator=SQLOperator.WINDOW,
                    query="SELECT *, session_id(...) OVER (...) FROM events")
    .add_transform("daily_active", "sessions",
                    operator=SQLOperator.GROUP_BY,
                    query="SELECT DATE(ts), COUNT(DISTINCT user_id) FROM sessions GROUP BY 1")
    .add_sink("dashboard", "daily_active")
    .build()
)

# Traversal
print(graph.topological_sort())
print(graph.sources())
print(graph.sinks())
print(graph.ancestors("dashboard"))
print(graph.critical_path())
```

### Fragment F

Fragment F is the deterministic, order-independent fragment where the
commutation theorem holds (incremental repair = full recomputation):

```python
from arc.graph import FragmentClassifier

classifier = FragmentClassifier()
classification = classifier.classify(graph)

print(classification.fragment_f_fraction)  # e.g., 0.75
print(classification.violations)  # nodes outside F with reasons
```

Nodes outside Fragment F include those with:
- Non-deterministic operators (LIMIT, UDF, external calls)
- Order-dependent operators (certain window functions)
- Side-effect-producing operators

### Quality Constraints

Quality constraints are first-class objects in the delta algebra:

```python
from arc.types import QualityConstraint, Severity

# Predefined constraint factories
qc_nn = QualityConstraint.not_null("nn_email", "email")
qc_range = QualityConstraint.range_check("age_range", "age", min_val=0, max_val=150)
qc_unique = QualityConstraint.uniqueness("pk_id", "id")
qc_fresh = QualityConstraint.freshness("data_freshness", "updated_at", max_staleness_hours=24)
qc_dist = QualityConstraint.distribution("dist_revenue", "revenue", test="ks", threshold=0.05)
```

---

## API Reference Summary

### `arc.types`

| Class | Description |
|---|---|
| `SQLType` | Enum of all SQL types |
| `ParameterisedType` | SQL type + parameters (length, precision, scale) |
| `TypeCompatibility` | Type widening/narrowing rules |
| `Column` | Column descriptor with type, nullable, default, constraints |
| `Schema` | Immutable relational schema with PK, FK, unique, check |
| `QualityConstraint` | Quality invariant (predicate, severity, columns) |
| `AvailabilityContract` | SLA, downtime, staleness tolerance |
| `CostEstimate` | Execution cost (compute, memory, I/O, monetary) |
| `TypedTuple` | Type-enforced named tuple |
| `MultiSet` | Bag with union, intersection, difference, multiplicity |
| `MultiSetDelta` | Difference between two multisets (inserts, deletes) |
| `SQLOperator` | Operator kinds (SELECT, JOIN, GROUP_BY, …) |
| `JoinType` | Join variants (INNER, LEFT, SEMI, ANTI, …) |
| `AggregateFunction` | SQL aggregates (COUNT, SUM, AVG, STDDEV, …) |
| `OperatorProperties` | Algebraic properties (deterministic, commutative, …) |
| `OperatorSignature` | Input schemas → output schema |

### `arc.types.errors`

| Exception | When raised |
|---|---|
| `ARCError` | Base class |
| `SchemaError` | Invalid schema structure |
| `TypeMismatchError` | Expected type ≠ actual type |
| `ColumnNotFoundError` | Column not in schema |
| `DeltaCompositionError` | Two deltas cannot compose |
| `DeltaPropagationError` | Delta cannot propagate through node |
| `PlannerError` | Repair planning failed |
| `InfeasibleRepairError` | No valid repair exists |
| `ExecutionError` | Repair execution failed |
| `CheckpointError` | Checkpoint creation/restore failed |
| `RollbackError` | Rollback failed |
| `ValidationError` | Generic validation failure |
| `FragmentViolationError` | Node outside Fragment F |
| `SerializationError` | Serialization/deserialization failure |
| `ParseError` | Cannot parse input |
| `CycleDetectedError` | Pipeline contains cycle |

### `arc.graph`

| Class / Function | Description |
|---|---|
| `PipelineNode` | DAG node with schema, constraints, cost |
| `PipelineEdge` | Directed edge with column mapping |
| `PipelineGraph` | Full DAG with traversal, validation, cloning |
| `PipelineBuilder` | Fluent API for graph construction |
| `impact_analysis()` | Downstream impact of a perturbation |
| `dependency_analysis()` | Upstream dependencies of a node |
| `detect_bottlenecks()` | Nodes with highest fan-out / centrality |
| `detect_redundancies()` | Duplicate computations |
| `FragmentClassifier` | Classify nodes into Fragment F |
| `compute_metrics()` | Pipeline complexity metrics |
| `compute_repair_scope()` | Minimal set of nodes to recompute |
| `to_dot()` | Graphviz DOT export |
| `to_ascii()` | Terminal ASCII rendering |
| `to_mermaid()` | Mermaid diagram export |

### `arc.io`

| Class / Function | Description |
|---|---|
| `PipelineSpec` | JSON pipeline spec load/save/validate |
| `YAMLPipelineSpec` | YAML spec with anchors, env vars, includes |
| `DeltaSerializer` | Delta JSON round-trip |
| `RepairPlanSerializer` | Repair plan JSON round-trip |
| `from_template()` | Generate pipeline from named template |
| `get_pipeline_schema()` | JSON Schema for pipeline specs |
| `get_delta_schema()` | JSON Schema for delta specs |
| `get_repair_plan_schema()` | JSON Schema for repair plans |

---

## CLI Reference

```
Usage: arc [OPTIONS] COMMAND [ARGS]...

Commands:
  analyze    Analyze a pipeline and show dependency graph
  repair     Compute repair plan for a perturbation
  execute    Execute a repair plan
  validate   Validate a pipeline specification
  fragment   Check Fragment F membership
  visualize  Visualize the pipeline graph
  monitor    Monitor quality metrics
  template   Generate pipeline from template
  info       Show ARC system information
```

### `arc analyze`

```bash
arc analyze pipeline.yaml                    # Summary metrics
arc analyze pipeline.yaml --node raw_users   # Impact analysis from node
arc analyze pipeline.yaml --verbose          # Include dependency tree
```

### `arc repair`

```bash
arc repair pipeline.yaml -p delta.json                # Compute plan
arc repair pipeline.yaml -p delta.json -o repair.json  # Save plan
arc repair pipeline.yaml -p delta.json --dry-run       # Preview only
```

### `arc execute`

```bash
arc execute repair.json              # Execute plan
arc execute repair.json --dry-run    # Preview only
arc execute repair.json --no-checkpoint  # Skip checkpoints
```

### `arc validate`

```bash
arc validate pipeline.yaml           # Warn on issues
arc validate pipeline.yaml --strict  # Fail on any issue
```

### `arc fragment`

```bash
arc fragment pipeline.yaml                  # All nodes
arc fragment pipeline.yaml --node my_node   # Single node
```

### `arc visualize`

```bash
arc visualize pipeline.yaml                          # ASCII
arc visualize pipeline.yaml -f dot -o pipeline.dot   # Graphviz
arc visualize pipeline.yaml -f mermaid               # Mermaid
arc visualize pipeline.yaml -h affected_node         # Highlight
```

### `arc template`

```bash
arc template etl_basic                    # Print YAML
arc template star_schema -o star.yaml     # Save to file
arc template diamond -f json -o dia.json  # JSON format
```

### `arc monitor`

```bash
arc monitor pipeline.yaml                # All constraints
arc monitor pipeline.yaml --node source  # Single node
```

---

## Pipeline Specification

### JSON format

```json
{
  "version": "1.0",
  "name": "my_pipeline",
  "metadata": {"owner": "data-team"},
  "nodes": [
    {
      "node_id": "source",
      "operator": "SOURCE",
      "output_schema": {
        "columns": [
          {"name": "id", "sql_type": {"base": "INT"}, "nullable": false}
        ],
        "primary_key": ["id"]
      },
      "quality_constraints": [
        {"constraint_id": "pk", "predicate": "UNIQUE(id)", "severity": "error"}
      ],
      "availability_contract": {
        "sla_percentage": 99.9,
        "max_downtime_seconds": 300
      },
      "cost_estimate": {
        "compute_seconds": 10.0,
        "row_estimate": 1000000
      }
    }
  ],
  "edges": [
    {"source": "source", "target": "transform", "edge_type": "data_flow"}
  ]
}
```

### YAML format

```yaml
version: "1.0"
name: my_pipeline

# Anchors for DRY configuration
defaults: &default_availability
  sla_percentage: 99.0
  max_downtime_seconds: 3600

nodes:
  - node_id: source
    operator: SOURCE
    output_schema:
      columns:
        - name: id
          sql_type: {base: INT}
          nullable: false
    availability_contract:
      <<: *default_availability      # Anchor reference
      sla_percentage: 99.9           # Override

  - node_id: transform
    operator: TRANSFORM
    # Environment variable interpolation
    query_text: "SELECT * FROM ${SOURCE_TABLE:raw_data}"

edges:
  - source: source
    target: transform
```

### Delta specification

```json
{
  "sort": "schema",
  "target_node": "source",
  "operations": [
    {"type": "add_column", "column": {"name": "phone", "sql_type": {"base": "VARCHAR"}}},
    {"type": "rename_column", "old_name": "name", "new_name": "full_name"},
    {"type": "widen_type", "column": "id", "new_type": {"base": "BIGINT"}}
  ]
}
```

---

## Example Walkthrough

### Scenario: Column addition propagation

A data source adds a `phone` column. ARC computes the minimal repair:

```
 raw_users (SOURCE)  ← schema delta: ADD COLUMN phone VARCHAR(20)
     │
     ▼
 clean_users (FILTER)  ← repair: recompute (filter passes new column through)
     │
     ▼
 user_counts (GROUP_BY)  ← skip (aggregation doesn't use 'phone')
     │
     ▼
 dashboard (SINK)  ← skip (output unchanged)
```

**Result:** Only `clean_users` needs recomputation. The planner proves
that `user_counts` and `dashboard` are unaffected because the `phone`
column is not in the GROUP BY's input lineage.

### Scenario: Compound perturbation

A schema change and quality drift arrive simultaneously:

```python
# Schema delta: add column + widen type
schema_delta = {
    "sort": "schema",
    "operations": [
        {"type": "add_column", "column": {"name": "phone", ...}},
        {"type": "widen_type", "column": "id", "new_type": {"base": "BIGINT"}},
    ],
}

# Quality delta: null rate exceeded
quality_delta = {
    "sort": "quality",
    "constraint_changes": [
        {"constraint_id": "nn_email", "change_type": "violated", "observed": 0.15},
    ],
}

# Compound perturbation
compound = {
    "sort": "compound",
    "target_node": "raw_users",
    "components": [schema_delta, quality_delta],
}
```

The interaction homomorphisms φ(δ_s) and ψ(δ_s) correctly compose the
repairs — the new `phone` column gets a quality constraint, and the
type widening propagates through all downstream consumers.

---

## Supported SQL Dialects and Python ETL Frameworks

### SQL Dialects (via sqlglot)

| Dialect | Static Analysis | Delta Propagation |
|---|---|---|
| PostgreSQL | ✓ (Tier 1) | ✓ |
| Spark SQL | ✓ (Tier 1) | ✓ |
| DuckDB | ✓ (evaluation) | ✓ |
| MySQL | Tier 3 | Tier 3 |
| BigQuery | Tier 3 | Tier 3 |

### Python ETL Frameworks

| Framework | Idiom Coverage | Status |
|---|---|---|
| pandas | ~85% of column-level ops | Tier 2 |
| PySpark | ~80% of DataFrame ops | Tier 2 |
| dbt models | SQL-based analysis | Tier 1 |

### Supported SQL Features

- CTEs (Common Table Expressions)
- Correlated subqueries
- Window functions with full frame specs
- Lateral joins
- Set operations (UNION, INTERSECT, EXCEPT)
- Complex aggregations with FILTER and ORDER BY
- JSON/JSONB operations

---

## Theory Reference

### Three-sorted delta algebra

The algebra is defined as Δ = (Δ_S, Δ_D, Δ_Q, ∘, ⁻¹, push) where:

- **Δ_S** — Schema delta monoid: column additions, type widenings, renames
- **Δ_D** — Data delta group: inserts, deletes, updates (with corrections)
- **Δ_Q** — Quality delta lattice: tightened constraints, new bounds

### Interaction homomorphisms

- **φ(δ_s)**: Schema → Data transformer (new columns need defaults)
- **ψ(δ_s)**: Schema → Quality transformer (new columns need constraints)

### Algebraic laws

1. **Associativity**: (δ₁ ∘ δ₂) ∘ δ₃ = δ₁ ∘ (δ₂ ∘ δ₃)
2. **Identity**: δ ∘ ε = ε ∘ δ = δ
3. **Inversion**: δ ∘ δ⁻¹ = ε (for data deltas)
4. **Commutation theorem** (Fragment F):
   `apply(repair(σ), state(G)) = recompute(evolve(G, σ))`

### Complexity

| Topology | Algorithm | Complexity |
|---|---|---|
| Acyclic (>90% of real pipelines) | Dynamic programming | O(\|V\| · 2^k), k = max in-degree |
| General | LP relaxation + rounding | Heuristic with feasibility patch |
| Small cyclic | ILP (HiGHS) | Exact, with timeout |

### Key theorems

1. **Encoding impossibility**: No data-domain encoding of schema deltas
   into DBSP preserves both type safety and incrementality.
2. **Bounded commutation**: For Fragment F pipelines, incremental repair
   yields the same state as full recomputation.
3. **NP-hardness**: Cost-optimal repair is NP-hard for general topologies
   (reduction from weighted set cover).
4. **Polynomial tractability**: Exact polynomial solution for acyclic
   topologies via dynamic programming.

### References

- Budiu et al. "DBSP: Automatic Incremental View Maintenance" (VLDB 2023)
- McSherry et al. "Differential Dataflow" (CIDR 2013)
- Curino et al. "PRISM: Schema Evolution Management" (VLDB 2008)

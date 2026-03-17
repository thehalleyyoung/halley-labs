# IsoSpec Architecture Guide

## Overview

IsoSpec is structured as a Rust workspace with 10 crates organized in three
layers: types, engine models, and analysis/application.

## Layer 1: Foundation Types (`isospec-types`)

The types crate provides all shared data structures:

- **Transaction IR**: `Transaction`, `Operation`, `OpKind` — the intermediate
  representation for transaction programs
- **Isolation Levels**: `IsolationLevel`, `AnomalyClass` — enumeration of
  standard isolation levels and their prohibited anomaly classes
- **Predicates**: `Predicate`, `PredicateAtom`, `CompOp` — symbolic
  representation of SQL WHERE clauses
- **Dependencies**: `Dependency`, `DependencyType` — DSG edge types
  (ww, wr, rw, predicate variants)
- **SMT Constraints**: `SmtExpr`, `SmtSort`, `SmtConstraintSet` — typed
  AST for SMT-LIB2 formula construction

## Layer 2: Engine Models (`isospec-engines`)

Each engine model implements the `EngineModel` trait:

```rust
pub trait EngineModel {
    fn step(&self, state: &EngineState, action: &Action) -> Result<EngineState>;
    fn can_commit(&self, state: &EngineState, txn: TransactionId) -> bool;
    fn conflicts(&self, state: &EngineState, op1: &Operation, op2: &Operation) -> bool;
}
```

### PostgreSQL 16.x SSI Model
- Tracks SIREAD locks, write locks, rw-dependency graph
- Implements dangerous-structure detection
- Supports read-only optimization

### MySQL 8.0 InnoDB Model
- Models record locks, gap locks, next-key locks per index
- Sound over-approximation across all possible index choices
- Captures gap lock interaction with INSERT operations

### SQL Server 2022 Dual-Mode Model
- Pessimistic mode: key-range locks (RangeS-S through RangeX-X)
- Optimistic mode: row versioning + commit-time conflict detection
- Mode selected by configuration flag

## Layer 3: Analysis Engine (`isospec-core`)

### DSG Builder
Constructs the Direct Serialization Graph from transaction histories by
extracting ww, wr, and rw dependencies between committed transactions.

### Cycle Detector
Uses Tarjan's SCC algorithm to find cycles in the DSG, classifying each
cycle by the dependency types on its edges to determine anomaly class.

### SMT Encoder
Encodes engine semantics as QF_LIA SMT-LIB2 constraints for Z3. Supports
incremental solving with push/pop and MaxSMT for optimization.

### Portability Analyzer
Performs differential analysis between two engine models to find workloads
that are safe under the source but anomalous under the target.

### Refinement Checker
Verifies that engine models correctly refine Adya's standard isolation
specifications via trace inclusion checking.

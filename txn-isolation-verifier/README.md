<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-green" alt="License">
  <img src="https://img.shields.io/badge/rust-1.74%2B-orange" alt="Rust">
  <img src="https://img.shields.io/badge/Z3-4.12%2B-purple" alt="Z3">
  <img src="https://img.shields.io/badge/crates-10%20workspace-red" alt="Crates">
  <img src="https://img.shields.io/badge/LoC-~38K-informational" alt="Lines of Code">
  <img src="https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey" alt="Platform">
</p>

<h1 align="center">IsoSpec</h1>
<h3 align="center">Verified Cross-Engine Transaction Isolation Analyzer</h3>

<p align="center">
  <em>
    A formal-methods-based tool for verifying transaction isolation semantics
    across production SQL database engines. IsoSpec mechanically checks whether
    observed transaction histories satisfy declared isolation levels, detects
    cross-engine behavioral divergences, and synthesizes minimal SQL witnesses
    that expose anomalies — all backed by SMT-based verification over
    engine-specific operational semantics.
  </em>
</p>

---

> **Key Insight:** Production database engines do *not* implement textbook
> isolation levels identically. PostgreSQL's Serializable Snapshot Isolation
> (SSI) prevents write skew via conflict detection; MySQL InnoDB uses gap
> locking to prevent phantoms under REPEATABLE READ; SQL Server offers both
> pessimistic and optimistic (RCSI) concurrency modes. IsoSpec encodes each
> engine's *actual* semantics — not the SQL standard's abstract definitions —
> and formally verifies whether a workload behaves equivalently across engines.
> When it doesn't, IsoSpec tells you exactly why and produces a minimal SQL
> script to reproduce the divergence.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [CLI Reference](#cli-reference)
- [Format Support](#format-support)
- [Examples](#examples)
  - [Write Skew Detection under PostgreSQL SSI](#1-write-skew-detection-under-postgresql-ssi)
  - [MySQL InnoDB Gap Lock Phantom Prevention](#2-mysql-innodb-gap-lock-phantom-prevention)
  - [Cross-Engine Portability Analysis](#3-cross-engine-portability-pgmysql-migration)
  - [Jepsen History Import and Analysis](#4-jepsen-history-import-and-analysis)
  - [Lost Update under READ COMMITTED](#5-lost-update-under-read-committed)
- [Theory Overview](#theory-overview)
- [Benchmarks](#benchmarks)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Features

- **Engine-Specific Operational Semantics**
  - PostgreSQL Serializable Snapshot Isolation (SSI) with rw-antidependency
    cycle detection, predicate locks, and SIRead lock promotion rules
  - MySQL InnoDB gap locking, next-key locking, and implicit lock semantics
    under REPEATABLE READ and SERIALIZABLE isolation modes
  - SQL Server dual-mode concurrency: pessimistic locking (default) and
    Read Committed Snapshot Isolation (RCSI) with row-versioning semantics
  - Configurable engine models — extend with custom lock managers, MVCC
    strategies, or conflict resolution policies

- **Anomaly Detection (Adya Classification)**
  - **G0** — Dirty Write detection across concurrent transactions
  - **G1a** — Aborted Read (dirty read of uncommitted data)
  - **G1b** — Intermediate Read (reading intermediate writes)
  - **G1c** — Circular Information Flow in commit-order dependency graphs
  - **G2-item** — Item Anti-Dependency Cycle (write skew)
  - **G2** — Full predicate-level anti-dependency cycles (phantom-class)
  - All anomaly classifications follow Adya, Liskov & O'Neil's Direct
    Serialization Graph (DSG) formalism with predicate extensions

- **Cross-Engine Portability Analysis**
  - Automated migration safety verification: determine whether a workload
    that is anomaly-free on Engine A remains anomaly-free on Engine B
  - Differential anomaly reporting: surface *exactly* which anomalies
    a migration introduces, with root-cause attribution to engine semantics
  - Support for multi-hop migration chains (e.g., PostgreSQL → MySQL → SQL Server)
  - Configurable safety thresholds and risk classification per anomaly class

- **SMT-Based Verification**
  - Encoding over the quantifier-free fragment of Linear Integer Arithmetic
    with Uninterpreted Functions and Arrays (QF_LIA+UF+Arrays)
  - Incremental solver interface for iterative refinement queries
  - Predicate abstraction for scalable analysis of large transaction histories
  - Proof-producing mode: extract UNSAT cores to explain *why* an anomaly
    is impossible under a given isolation level

- **Witness Synthesis**
  - Minimal reproducible SQL scripts that trigger detected anomalies
  - Engine-specific DDL and session configuration (SET TRANSACTION ISOLATION
    LEVEL, advisory locks, connection parameters)
  - Timing-annotated scripts with configurable delay injection for
    deterministic reproduction on live database instances
  - Witness minimization via delta-debugging over the transaction history

- **Predicate-Level Conflict Theory**
  - Conjunctive inequality fragment: models WHERE clauses as conjunctions
    of range predicates over integer-typed columns
  - Predicate conflict detection: determines whether two predicate-guarded
    operations can produce an anti-dependency under the engine's lock protocol
  - Integration with the SMT backend for automated predicate satisfiability
    checks during anomaly analysis
  - **NULL-aware resolution**: three-valued logic (TRUE/FALSE/UNKNOWN) for
    nullable columns with honest complexity classification — PTIME when no
    NULLs, co-NP-complete when both predicates reference nullable columns
  - Corrected Adya cycle bounds: G1a and G2-item at k=2; predicate-level
    G2 flagged as unbounded with empirical bound (k≤8)

---

## Architecture

IsoSpec is organized as a Cargo workspace with 10 core crates and one
format-specification crate:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          isospec-cli                                │
│                    (CLI entry point & orchestration)                 │
└──────────┬──────────────────────────────────┬───────────────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐           ┌──────────────────────┐
│   isospec-adapter   │           │   isospec-bench      │
│  (database adapter, │           │  (benchmark harness,  │
│   Docker, SQL exec) │           │   workload suites)    │
└──────────┬──────────┘           └──────────┬───────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐           ┌──────────────────────┐
│  isospec-history    │           │   isospec-anomaly    │
│  (history model,    │◄─────────►│  (Adya DSG anomaly   │
│   dependency graph) │           │   classification)     │
└──────────┬──────────┘           └──────────┬───────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐           ┌──────────────────────┐
│  isospec-engines    │           │   isospec-smt        │
│  (PG SSI, MySQL     │           │  (Z3 encoding,       │
│   InnoDB, SQL Srv)  │           │   QF_LIA+UF+Arrays)  │
└──────────┬──────────┘           └──────────┬───────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐           ┌──────────────────────┐
│  isospec-witness    │           │   isospec-format     │
│  (SQL script gen,   │           │  (trace parsers: PG   │
│   witness minimize) │           │   wire, MySQL, EDN)   │
└──────────┬──────────┘           └──────────────────────┘
           │
           ▼
┌─────────────────────┐
│  isospec-core       │
│  (shared logic,     │
│   refinement bridge)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  isospec-types      │
│  (base types,       │
│   traits, errors)   │
└─────────────────────┘
```

### Crate Responsibilities

| Crate              | Purpose                                                         |
| ------------------ | --------------------------------------------------------------- |
| `isospec-types`    | Foundational types: `TransactionId`, `OpKind`, `IsolationLevel`, error types, core traits |
| `isospec-core`     | Shared analysis logic, refinement bridge between engines and SMT solver |
| `isospec-engines`  | Engine-specific operational semantics (PostgreSQL, MySQL, SQL Server) |
| `isospec-history`  | Transaction history model, dependency graph construction (DSG)  |
| `isospec-anomaly`  | Anomaly detection and classification using Adya's G0–G2 taxonomy |
| `isospec-smt`      | Z3 bindings, SMT encoding (QF_LIA+UF+Arrays), solver orchestration |
| `isospec-witness`  | SQL witness script generation, delta-debugging minimization     |
| `isospec-adapter`  | Database adapter: connection pooling, Docker management, SQL execution |
| `isospec-bench`    | Benchmark harness, Hermitage test suite integration, metric collection |
| `isospec-cli`      | CLI interface, subcommand dispatch, output formatting           |
| `isospec-format`   | Trace format parsers: PG wire, MySQL wire, SQL trace, Jepsen EDN      |

---

## Installation

### Prerequisites

| Dependency | Version  | Notes                                    |
| ---------- | -------- | ---------------------------------------- |
| Rust       | ≥ 1.74   | Edition 2021; stable toolchain           |
| Z3         | ≥ 4.12   | SMT solver; `libz3-dev` / `z3` package  |
| Cargo      | ≥ 1.74   | Ships with Rust                          |

**Install Z3:**

```bash
# Ubuntu / Debian
sudo apt-get install -y libz3-dev

# macOS (Homebrew)
brew install z3

# From source (any platform)
git clone https://github.com/Z3Prover/z3.git && cd z3
python scripts/mk_make.py && cd build && make -j$(nproc) && sudo make install
```

### Install from Source

```bash
git clone https://github.com/isospec/isospec.git
cd isospec
cd implementation
cargo build --release --bin isospec

# The binary is at target/release/isospec
# Optionally, install to your PATH:
cargo install --path crates/isospec-cli
```

### Install from crates.io *(planned)*

```bash
# Coming soon:
# cargo install isospec
```

### Verify Installation

```bash
isospec --version
# isospec 0.1.0

isospec --help
```

---

## Quickstart

### 1. Analyze a Transaction History

```bash
# Check whether a recorded history satisfies its declared isolation level
isospec analyze --workload ../examples/pg_serializable_write_skew.json
```

Audit note: build and run these commands from `implementation/`. In the current
checkout, the shipped examples now load correctly from `implementation/` and
produce non-empty analysis / portability / witness output.

**Current output (`implementation/`):**

```
═══ Anomaly Analysis ═══
Workload:     pg_serializable_write_skew
Engine:       PostgreSQL
Isolation:    Serializable
Transactions: 2
Operations:   4
Tables:       1

✗ 1 anomaly class(es) detected:

  G2-item (Item Anti-Dependency) [Medium]
    PostgreSQL SSI detects rw-dependency cycle T1→T2→T1 and aborts T2
```

### 2. Cross-Engine Portability Analysis

```bash
# Check if a PostgreSQL workload is safe to migrate to MySQL
isospec portability \
  --source-engine postgresql \
  --target-engine mysql \
  --workload ../examples/cross_engine_pg_to_mysql.json
```

**Current output (`implementation/`):**

```
═══ Portability Check ═══
Workload: cross_engine_portability_pg_to_mysql
Source:   PostgreSQL/Serializable
Target:   MySQL/RepeatableRead

✗ 2 portability violation(s) found:
  • G2-item (Item Anti-Dependency) [Medium]: G2-item (Item Anti-Dependency) is prevented by PostgreSQL/Serializable but allowed by MySQL/RepeatableRead
  • G2 (Predicate Anti-Dependency) [Low]: G2 (Predicate Anti-Dependency) is prevented by PostgreSQL/Serializable but allowed by MySQL/RepeatableRead
```

### 3. Synthesize a Witness Script

```bash
# Generate a minimal SQL script that triggers G2-item on PostgreSQL
isospec witness \
  --engine postgresql \
  --anomaly g2-item \
  --workload ../examples/pg_serializable_write_skew.json
```

**Current output (`implementation/`):**

```text
═══ Witness Generation ═══
Workload: pg_serializable_write_skew
Engine:   PostgreSQL
Isolation:ReadCommitted
Anomaly:  G2-item (Item Anti-Dependency)

Generated 1 witness(es):

── Witness #0 (G2-item (Item Anti-Dependency), 2 txns) ──
    1. BEGIN T1 (ReadCommitted) -- Alice tries to go off-call
    2. BEGIN T2 (ReadCommitted) -- Bob tries to go off-call
    3. T1: READ doctors WHERE on_call = true
    4. T2: READ doctors WHERE on_call = true
    5. T1: WRITE doctors SET on_call = FALSE WHERE id = 1
    6. T2: WRITE doctors SET on_call = FALSE WHERE id = 2
    7. COMMIT T1
    8. COMMIT T2
```

### 4. Run the Hermitage Benchmark Suite

```bash
# Run the standard benchmark suite across all supported engines
isospec benchmark --suite standard
```

**Output:**

```
Hermitage Benchmark Suite — Isolation Level Comparison
╔══════════════════╦═════════════╦══════════════╦══════════════╗
║ Anomaly          ║ PostgreSQL  ║ MySQL        ║ SQL Server   ║
╠══════════════════╬═════════════╬══════════════╬══════════════╣
║ G0 (Dirty Write) ║ Prevented   ║ Prevented    ║ Prevented    ║
║ G1a (Dirty Read) ║ Prevented   ║ Prevented    ║ Prevented    ║
║ G1b (Interm Read)║ Prevented   ║ Prevented    ║ Prevented    ║
║ G1c (Circ Info)  ║ Prevented   ║ Prevented    ║ Prevented    ║
║ G2-item (Wr Skew)║ Prevented   ║ ▲ POSSIBLE   ║ Prevented    ║
║ G2 (Phantom)     ║ Prevented   ║ ▲ POSSIBLE*  ║ Prevented    ║
╚══════════════════╩═════════════╩══════════════╩══════════════╝
* MySQL InnoDB REPEATABLE READ prevents phantoms in many but not all
  cases due to gap locking heuristics. See benchmarks/RESULTS.md.
```

---

## CLI Reference

```
isospec <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    analyze       Analyze a transaction history for anomalies
    portability   Cross-engine portability analysis
    witness       Synthesize minimal SQL witness scripts
    benchmark     Run benchmark suites (standard, TPC-C, etc.)
    validate      Validate a witness against a real engine model
    refinement    Compute refinement relations between engine models

GLOBAL OPTIONS:
    -v, --verbose     Enable verbose output (repeat for more: -vv, -vvv)
    -q, --quiet       Suppress all output except errors
        --format      Output format: text (default), json, csv, dot
    -h, --help        Print help information
    -V, --version     Print version information
```

### `analyze`

```bash
isospec analyze [OPTIONS]

OPTIONS:
    -w, --workload <PATH>       Path to the workload file (JSON, required)
    -e, --engine <ENGINE>       Target engine: postgresql (default), mysql,
                                sqlserver
    -i, --isolation <LEVEL>     Isolation level: read-uncommitted,
                                read-committed, repeatable-read,
                                serializable (default), snapshot
        --max-txns <N>          Maximum transactions to consider (default: 10)
        --predicates            Enable predicate-level analysis (slower,
                                more precise)
        --timeout <SECONDS>     SMT solver timeout (default: 60)
    -o, --output <PATH>         Output file (default: stdout)
```

### `portability`

```bash
isospec portability [OPTIONS]

OPTIONS:
    -w, --workload <PATH>           Path to the workload file (required)
        --source-engine <ENGINE>    Source engine (default: postgresql)
        --source-isolation <LEVEL>  Source isolation level (default: serializable)
        --target-engine <ENGINE>    Target engine (default: mysql)
        --target-isolation <LEVEL>  Target isolation level (default: repeatable-read)
        --witnesses                 Generate witness schedules for violations
        --max-witnesses <N>         Maximum witnesses to generate (default: 5)
    -o, --output <PATH>             Output file
```

### `witness`

```bash
isospec witness [OPTIONS]

OPTIONS:
    -w, --workload <PATH>       Path to the workload file (required)
    -e, --engine <ENGINE>       Engine for which to generate witness SQL
                                (default: postgresql)
    -i, --isolation <LEVEL>     Isolation level (default: read-committed)
    -a, --anomaly <CLASS>       Anomaly class to target (required):
                                g0, g1a, g1b, g1c, g2-item, g2
        --count <N>             Maximum witnesses to generate (default: 1)
    -o, --output <PATH>         Output file path (default: stdout)
```

### `benchmark`

```bash
isospec benchmark [OPTIONS]

OPTIONS:
    -s, --suite <NAME>          Benchmark suite: standard (default), tpcc,
                                tpce, scaling, adversarial
        --warmup <N>            Number of warm-up iterations (default: 3)
        --iterations <N>        Number of iterations per test (default: 10)
    -o, --output-dir <DIR>      Output directory for results
        --report-format <FMT>   Report format: text (default), json, csv, dot
```

### `validate`

```bash
isospec validate [OPTIONS]

OPTIONS:
    -w, --witness <PATH>        Path to a witness schedule file (JSON, required)
    -e, --engine <ENGINE>       Target engine for validation (default: postgresql)
    -i, --isolation <LEVEL>     Isolation level (default: serializable)
    -o, --output <PATH>         Output file

Validates a witness schedule against an engine model. Returns exit code 0
on success, 1 on validation failure.
```

### `refinement`

```bash
isospec refinement [OPTIONS]

OPTIONS:
    --engine-a <ENGINE>         First engine model (default: postgresql)
    --level-a <LEVEL>           Isolation level of first engine
                                (default: serializable)
    --engine-b <ENGINE>         Second engine model (default: mysql)
    --level-b <LEVEL>           Isolation level of second engine
                                (default: repeatable-read)
    -w, --workload <PATH>       Optional workload for bounded refinement check
    -o, --output <PATH>         Output file
```

---

## Format Support

IsoSpec accepts transaction histories in multiple formats, automatically
detected by file extension and content inspection.

| Format                        | Extension     | Description                                          |
| ----------------------------- | ------------- | ---------------------------------------------------- |
| IsoSpec Native (JSON)         | `.json`       | Canonical format with full operation metadata         |
| IsoSpec Native (Binary)       | `.isob`       | Compact binary encoding for large histories           |
| PostgreSQL wire protocol      | `.pgcap`      | Captured via `pgaudit` or custom proxy                |
| MySQL wire protocol           | `.mycap`      | Captured via MySQL Proxy or custom instrumentation    |
| pgAudit CSV log               | `.pgaudit`    | PostgreSQL audit log (CSV format)                     |
| MySQL general query log       | `.mylog`      | MySQL general log with timestamps                     |
| Generic SQL trace log         | `.sqltrace`   | Engine-agnostic SQL trace with transaction markers    |
| Jepsen EDN history            | `.edn`        | Jepsen test framework history files                   |

### Trace Collection

**PostgreSQL (pgAudit):**

```bash
# Enable pgAudit in postgresql.conf
# shared_preload_libraries = 'pgaudit'
# pgaudit.log = 'all'

# Convert pgAudit log to IsoSpec format (via isospec-format crate)
# Programmatic: use isospec_format::pg_wire to parse, then serialize to JSON
```

**MySQL (General Log):**

```bash
# Enable general log
# SET GLOBAL general_log = 'ON';
# SET GLOBAL log_output = 'FILE';

# Convert MySQL log to IsoSpec format (via isospec-format crate)
# Programmatic: use isospec_format::mysql_wire to parse, then serialize to JSON
```

**Jepsen History:**

```bash
# Import a Jepsen EDN history (via isospec-format crate)
# Programmatic: use isospec_format::jepsen to parse, then analyze with isospec analyze
```

---

## Examples

### 1. Write Skew Detection under PostgreSQL SSI

Write skew occurs when two transactions read overlapping data, make
disjoint updates based on stale reads, and both commit — violating an
application-level constraint.

```bash
# Detect write skew in a PostgreSQL SERIALIZABLE history
isospec analyze \
  --engine postgresql \
  --isolation serializable \
  --workload examples/pg_serializable_write_skew.json
```

The underlying DSG analysis constructs the dependency graph:

```
T1 ──ww──► T2 ──rw──► T1   (cycle detected → G2-item)
     │                 ▲
     └────────rw───────┘
```

### 2. MySQL InnoDB Gap Lock Phantom Prevention

MySQL's InnoDB engine uses gap locks under REPEATABLE READ to prevent
phantoms in *most* cases, but certain predicate patterns can escape
gap lock coverage.

```bash
# Test whether InnoDB gap locking prevents a specific phantom scenario
isospec analyze \
  --engine mysql \
  --isolation repeatable-read \
  --predicates \
  --workload examples/mysql_gap_lock_phantom.json
```

```rust
// Programmatic usage via the isospec-engines and isospec-anomaly crates
use isospec_engines::mysql::MySqlModel;
use isospec_history::history::TransactionHistory;
use isospec_anomaly::detector::{AnomalyDetector, DetectionConfig};

let model = MySqlModel::new();  // InnoDB 8.0 model
let history = TransactionHistory::new();  // populate via HistoryBuilder
let detector = AnomalyDetector::new(DetectionConfig::default());

// Build a dependency graph and detect anomalies
let results = detector.detect(&edges, &committed, &aborted);
for anomaly in &results.detected {
    println!("{:?}: {}", anomaly.anomaly_class, anomaly.explanation);
}
```

### 3. Cross-Engine Portability (PG→MySQL Migration)

When migrating from PostgreSQL to MySQL, workloads that relied on SSI's
strong guarantees may exhibit new anomalies under InnoDB's lock-based
protocol.

```bash
# Full portability report with risk assessment
isospec portability \
  --source-engine postgresql \
  --target-engine mysql \
  --workload workload.json \
  --format json | jq '.new_anomalies'
```

```json
[
  {
    "class": "G2-item",
    "severity": "high",
    "description": "Write skew possible under MySQL REPEATABLE READ",
    "affected_transactions": ["T3", "T7"],
    "mitigation": "Add SELECT ... FOR UPDATE or use SERIALIZABLE"
  }
]
```

### 4. Jepsen History Import and Analysis

IsoSpec can directly ingest Jepsen EDN history files, enabling formal
anomaly analysis of Jepsen test results.

```bash
# Import and analyze a Jepsen history (after converting to IsoSpec JSON
# via the isospec-format crate's jepsen module)
isospec analyze \
  --engine postgresql \
  --workload converted_history.json
```

```bash
# Convert Jepsen history to IsoSpec native format programmatically:
# use isospec_format::jepsen to parse the EDN file, then serialize
# the resulting history to JSON for use with the CLI.
```

### 5. Lost Update under READ COMMITTED

Lost updates occur when two transactions read the same value, compute
new values based on that read, and both write — with one overwriting
the other's result.

```bash
isospec analyze \
  --engine postgresql \
  --isolation read-committed \
  --workload examples/lost_update_rc.json
```

```
Engine:    PostgreSQL
Isolation: READ COMMITTED
Result:    ANOMALY DETECTED

  ⚠ G2-item (Lost Update variant)
    T1: R(balance=100) W(balance=150)  -- adds 50
    T2: R(balance=100) W(balance=120)  -- adds 20
    Expected: balance=170  Actual: balance=120 (T1's write lost)

  Recommendation: Use SELECT ... FOR UPDATE to acquire row-level
  locks before read-modify-write sequences.
```

---

## Theory Overview

IsoSpec's formal foundations draw on three pillars:

### Adya's Direct Serialization Graph (DSG) Formalism

IsoSpec encodes transaction histories as Direct Serialization Graphs
following the framework of Adya, Liskov, and O'Neil (2000). The DSG
captures three types of dependencies:

- **Write-Write (ww):** Transaction T2 overwrites a value written by T1
- **Write-Read (wr):** Transaction T2 reads a value written by T1
- **Read-Write (rw):** Transaction T2 overwrites a value read by T1

Anomaly classes (G0–G2) are defined as specific cycle patterns in the DSG.
IsoSpec checks for these cycles using a combination of graph algorithms and
SMT-based verification for predicate-level anomalies.

### Predicate-Level Conflict Theory

Classical DSG analysis operates at the item (row) level. IsoSpec extends
this to the *predicate* level, handling WHERE clauses and range queries.
The key insight is that two operations conflict at the predicate level if
their predicate regions overlap and at least one is a write.

IsoSpec models predicates in the *conjunctive inequality fragment*:

```
P ::= c₁ ≤ x₁ ≤ d₁ ∧ c₂ ≤ x₂ ≤ d₂ ∧ ... ∧ cₙ ≤ xₙ ≤ dₙ
```

Predicate conflict is reduced to SMT satisfiability over this fragment,
enabling efficient analysis of range-predicate interactions.

### Engine Operational Semantics

Each supported engine is modeled as a labeled transition system over
a state space of locks, versions, and visibility maps:

- **PostgreSQL SSI:** Modeled as snapshot isolation plus a rw-antidependency
  tracker. Transactions are aborted when the tracker detects a dangerous
  structure (consecutive rw-antidependencies forming a pivot).
- **MySQL InnoDB:** Modeled with a hierarchical lock manager supporting
  record locks, gap locks, next-key locks, and insert intention locks.
- **SQL Server:** Dual-mode model supporting both pessimistic (2PL with
  lock escalation) and optimistic (RCSI with version store) concurrency.

### Refinement Bridge

IsoSpec defines a *refinement relation* between engine models. Engine A
*refines* Engine B at isolation level I if every history permitted by A
at level I is also permitted by B. This relation enables the portability
analysis: if the target engine does not refine the source, IsoSpec
searches for witness histories that expose the gap.

---

## Benchmarks

Detailed benchmark results are available in
[`benchmarks/real_benchmark_results.json`](benchmarks/real_benchmark_results.json)
and [`benchmarks/distributed_results.json`](benchmarks/distributed_results.json).

### Anomaly Detection Benchmark (34 scenarios)

| Size Category | Scenarios | IsoVerify F1 | Elle-style F1 | IsoVerify Avg Time |
| ------------- | --------- | ------------ | ------------- | ------------------ |
| Small (2 txns)  | 15 | **1.000** | 0.300 | 2.49 ms |
| Medium (8–24 txns) | 9 | **1.000** | 0.000 | 290 ms |
| Large (47–110 txns) | 10 | **1.000** | 0.400 | 36.8 s |
| **Overall** | **34** | **1.000** | **0.250** | — |

*Source: `benchmarks/real_benchmark_results.json`, field `summary`.*

### Distributed Evaluation (simulated, 300 txns/cell)

| DB Model / Scenario | Injected | Detected | Rate |
| ------------------- | -------- | -------- | ---- |
| CockroachDB / Clock Skew | 17 | 16 | 94.1% |
| Spanner / Clock Skew | 31 | 27 | 87.1% |
| Vitess / Clean | 91 | 90 | 98.9% |
| Vitess / Partition | 85 | 69 | 81.2% |

*Source: `benchmarks/distributed_results.json`.*

### Isolation Level Coverage (18 scenarios)

| Isolation Level | Cases | Perfect F1 | Coverage | FP | FN |
| --------------- | ----- | ---------- | -------- | -- | -- |
| Read Uncommitted | 3 | 3 | **100%** | 0 | 0 |
| Read Committed | 5 | 5 | **100%** | 0 | 0 |
| Repeatable Read | 2 | 2 | **100%** | 0 | 0 |
| Snapshot Isolation | 2 | 2 | **100%** | 0 | 0 |
| Serializable | 6 | 6 | **100%** | 0 | 0 |

Anomaly types validated: dirty reads (G1a), non-repeatable reads (P2),
phantom reads (P3/G2), lost updates (P4), read skew (A5A), write skew
(A5B/G-SIa) — all with confidence 1.000.

*Source: `benchmarks/isolation_level_results.json`.*

---

## API Reference

Full API documentation is available via `cargo doc`:

```bash
cargo doc --open --no-deps
```

### Key Types

```rust
use isospec_types::{OpKind, IsolationLevel};
use isospec_types::identifier::TransactionId;
use isospec_types::config::EngineKind;
use isospec_history::history::TransactionHistory;
use isospec_history::analyzer::DependencyGraph;
use isospec_types::Operation;
use isospec_anomaly::detector::{AnomalyDetector, AnomalyReport};
use isospec_core::engine_traits::EngineModel;
use isospec_engines::{postgresql, mysql, sqlserver};
use isospec_core::smt_encoding::SmtEncoder;
use isospec_smt::solver::SolverConfig;
use isospec_witness::synthesizer::{WitnessSynthesizer, WitnessResult};
```

### Key Traits

```rust
/// Core trait implemented by all engine models
pub trait EngineModel: Send + Sync {
    fn engine_kind(&self) -> EngineKind;
    fn supported_isolation_levels(&self) -> Vec<IsolationLevel>;
    fn create_state(&self, isolation_level: IsolationLevel) -> Box<dyn EngineState>;
    fn encode_constraints(&self, isolation_level: IsolationLevel, txn_count: usize,
                          op_count: usize) -> IsoSpecResult<SmtConstraintSet>;
    fn version_string(&self) -> &str;
    fn validate_schedule(&self, schedule: &Schedule, level: IsolationLevel)
        -> IsoSpecResult<ValidationResult>;
}

/// Mutable engine state during schedule execution
pub trait EngineState: Send {
    fn begin_transaction(&mut self, txn_id: TransactionId, level: IsolationLevel)
        -> IsoSpecResult<()>;
    fn execute_operation(&mut self, op: &Operation) -> IsoSpecResult<OperationOutcome>;
    fn commit_transaction(&mut self, txn_id: TransactionId) -> IsoSpecResult<CommitOutcome>;
    fn abort_transaction(&mut self, txn_id: TransactionId) -> IsoSpecResult<()>;
    fn extract_dependencies(&self) -> Vec<Dependency>;
}

/// Trait for database adapters (executing SQL on real engines)
pub trait DatabaseAdapter: Send + Sync {
    fn engine(&self) -> EngineKind;
    fn execute(&self, sql: &str) -> IsoSpecResult<QueryResult>;
    fn execute_batch(&self, statements: &[String]) -> IsoSpecResult<Vec<QueryResult>>;
    fn ping(&self) -> IsoSpecResult<bool>;
    fn server_version(&self) -> IsoSpecResult<String>;
    fn close(&self) -> IsoSpecResult<()>;
}
```

---

## Project Structure

```
isospec/
├── Cargo.toml                  # Workspace root
├── Cargo.lock
├── README.md
├── LICENSE-MIT
├── LICENSE-APACHE
├── CONTRIBUTING.md
├── CHANGELOG.md
│
├── crates/
│   ├── isospec-types/          # Foundational types, traits, errors
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── identifier.rs  # TransactionId, OperationId, ItemId
│   │       ├── transaction.rs # Transaction, TransactionStatus
│   │       ├── operation.rs   # Operation, OpKind, ReadOp, WriteOp
│   │       ├── isolation.rs   # IsolationLevel, AnomalyClass
│   │       ├── config.rs      # EngineKind, AnalysisConfig
│   │       ├── error.rs       # IsoSpecError, IsoSpecResult
│   │       ├── dependency.rs  # Dependency, DependencyType
│   │       ├── predicate.rs   # Predicate types
│   │       ├── schedule.rs    # Schedule, ScheduleStep
│   │       ├── constraint.rs  # SmtConstraintSet, SmtExpr
│   │       ├── value.rs       # Value type
│   │       ├── workload.rs    # Workload definition
│   │       └── ...            # lock, schema, snapshot, ir, etc.
│   │
│   ├── isospec-core/           # Shared analysis logic, refinement
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── analyzer.rs     # BoundedAnalyzer, AnalysisResult
│   │       ├── engine_traits.rs # EngineModel, EngineState traits
│   │       ├── smt_encoding.rs # SmtEncoder for bounded model checking
│   │       ├── refinement.rs   # Refinement relation computation
│   │       ├── portability.rs  # Cross-engine portability analysis
│   │       ├── dsg.rs          # Direct Serialization Graph
│   │       ├── cycle.rs        # Cycle detection algorithms
│   │       ├── predicates.rs   # Predicate conflict analysis
│   │       ├── null_aware.rs   # NULL-aware three-valued logic
│   │       └── ...             # cache, conflict, optimizer, scheduler
│   │
│   ├── isospec-engines/        # Engine-specific models
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── postgresql.rs   # PostgreSQL SSI model
│   │       ├── mysql.rs        # MySQL InnoDB model
│   │       └── sqlserver.rs    # SQL Server dual-mode model
│   │
│   ├── isospec-history/        # History model, DSG construction
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── history.rs      # TransactionHistory, HistoryEvent
│   │       ├── analyzer.rs     # HistoryAnalyzer, DependencyGraph
│   │       ├── builder.rs      # HistoryBuilder API
│   │       ├── replay.rs       # Schedule replay engine
│   │       └── trace.rs        # Trace utilities
│   │
│   ├── isospec-anomaly/        # Anomaly detection (G0–G2)
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── detector.rs     # AnomalyDetector, AnomalyReport
│   │       ├── classifier.rs   # AnomalyClassifier, ClassificationResult
│   │       ├── catalog.rs      # Anomaly catalog
│   │       ├── hermitage.rs    # Hermitage test suite definitions
│   │       └── report.rs       # Report formatting
│   │
│   ├── isospec-smt/            # SMT encoding and solver interface
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── encoding.rs     # EncodingBounds, VarNaming
│   │       ├── solver.rs       # SmtSolver trait, SolverConfig, SolverResult
│   │       ├── formula.rs      # SMT formula representation
│   │       ├── model.rs        # Model extraction
│   │       ├── incremental.rs  # Incremental solver interface
│   │       └── optimizer.rs    # Solver optimization strategies
│   │
│   ├── isospec-witness/        # Witness synthesis and minimization
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── synthesizer.rs  # WitnessSynthesizer, WitnessResult
│   │       ├── minimizer.rs    # Delta-debugging minimization
│   │       ├── sql_gen.rs      # SQL script generation
│   │       ├── timing.rs       # Timing annotation
│   │       └── validator.rs    # Witness validation
│   │
│   ├── isospec-adapter/        # Database adapter layer
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── adapter.rs      # DatabaseAdapter trait, AdapterConfig
│   │       ├── connection.rs   # Connection pooling
│   │       ├── docker.rs       # Docker container management
│   │       ├── executor.rs     # SQL execution orchestrator
│   │       └── parser.rs       # Result parsing
│   │
│   ├── isospec-bench/          # Benchmark harness
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── harness.rs      # Benchmark runner
│   │       ├── workloads.rs    # Benchmark workload definitions
│   │       ├── metrics.rs      # Metric collection
│   │       ├── report.rs       # Report generation
│   │       ├── tpcc.rs         # TPC-C benchmark
│   │       └── tpce.rs         # TPC-E benchmark
│   │
│   ├── isospec-cli/            # CLI entry point
│   │   └── src/
│   │       ├── main.rs         # CLI entry, clap argument parsing
│   │       ├── commands.rs     # Subcommand implementations
│   │       ├── input.rs        # Input parsing and loading
│   │       ├── output.rs       # Output formatting
│   │       └── format.rs       # Format conversion helpers
│   │
│   └── isospec-format/         # Trace format parsers
│       └── src/
│           ├── lib.rs
│           ├── pg_wire.rs      # PostgreSQL wire protocol parser
│           ├── mysql_wire.rs   # MySQL wire protocol parser
│           ├── sql_trace.rs    # Generic SQL trace parser
│           ├── jepsen.rs       # Jepsen history parser
│           └── edn.rs          # EDN format support
│
├── examples/                   # Example history files
│   ├── pg_serializable_write_skew.json
│   ├── mysql_gap_lock_phantom.json
│   ├── lost_update_rc.json
│   ├── sqlserver_rcsi_anomaly.json
│   └── cross_engine_migration.json
│
├── benchmarks/                 # Benchmark data and results
│   ├── RESULTS.md
│   ├── hermitage/
│   └── synthetic/
│
└── docs/                       # Additional documentation
    ├── theory.md               # Full formal theory
    ├── engine-models.md        # Engine model specifications
    └── format-spec.md          # Trace format specification
```

---

## Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for
detailed guidelines on:

- Setting up the development environment
- Running the test suite (`cargo test --workspace`)
- Code style and formatting (`cargo fmt`, `cargo clippy`)
- Submitting pull requests
- Reporting issues

### Quick Development Setup

```bash
git clone https://github.com/isospec/isospec.git
cd isospec

# Build all crates
cargo build --workspace

# Run all tests
cargo test --workspace

# Run with verbose logging
RUST_LOG=debug cargo run -- analyze --workload examples/pg_serializable_write_skew.json

# Run clippy lints
cargo clippy --workspace -- -D warnings

# Generate documentation
cargo doc --workspace --no-deps --open
```

---

## Citation

If you use IsoSpec in academic work, please cite:

```bibtex
@software{isospec2024,
  title     = {{IsoSpec}: Verified Cross-Engine Transaction Isolation Analyzer},
  author    = {{IsoSpec Contributors}},
  year      = {2024},
  url       = {https://github.com/isospec/isospec},
  version   = {0.4.0},
  note      = {Formal-methods-based tool for verifying transaction isolation
               semantics across SQL database engines}
}
```

### Related Work

- Adya, A., Liskov, B., & O'Neil, P. (2000). *Generalized Isolation Level Definitions.*
  ICDE 2000.
- Berenson, H., Bernstein, P., et al. (1995). *A Critique of ANSI SQL Isolation Levels.*
  SIGMOD 1995.
- Crooks, N., Pu, Y., Alvisi, L., & Clement, A. (2017). *Seeing is Believing:
  A Client-Centric Specification of Database Isolation.* PODC 2017.
- Kingsbury, K. (2020). *Jepsen: Distributed Systems Safety Research.*
  https://jepsen.io

---

## License

IsoSpec is dual-licensed under your choice of:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT))
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

### Contribution Licensing

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in IsoSpec by you shall be dual-licensed as above, without any
additional terms or conditions.

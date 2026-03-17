# Penumbra Architecture Guide

## Overview

Penumbra is structured as a Rust workspace with 10 crates, organized in a
strict dependency hierarchy to prevent circular dependencies and maintain
clean separation of concerns.

## Data Flow

```
Python Script → Shadow Instrumentation → Execution Trace → EAG Builder
→ Error Amplification Graph → Diagnosis Engine → Diagnosis Report
→ Repair Synthesizer → Repair Candidates → Certification → Repair Result
→ Report Generator → Output (Human/JSON/CSV)
```

## Crate Architecture

### Foundation Layer

**fpdiag-types** — The shared vocabulary crate. Every other crate depends on
this. Contains no logic beyond type definitions, constructors, and basic
methods. Modules:

| Module | Types |
|--------|-------|
| `ieee754` | `Ieee754Format`, `Ieee754Value`, `FpClass` |
| `precision` | `Precision`, `PrecisionRequirement`, `PrecisionCost` |
| `rounding` | `RoundingMode`, `RoundingError`, `StochasticRounder` |
| `expression` | `FpOp`, `ExprNode`, `ExprTree`, `ExprBuilder`, `NodeId` |
| `error_bounds` | `ErrorMetric`, `ErrorBound`, `ErrorInterval`, `ErrorSummary` |
| `eag` | `EagNodeId`, `EagEdge`, `EagNode`, `ErrorAmplificationGraph` |
| `trace` | `TraceEvent`, `ExecutionTrace`, `TraceMetadata` |
| `diagnosis` | `DiagnosisCategory`, `Diagnosis`, `DiagnosisReport` |
| `repair` | `RepairStrategy`, `RepairCandidate`, `RepairResult` |
| `source` | `SourceSpan` |
| `config` | `PenumbraConfig` and sub-configs |
| `ulp` | ULP computation utilities |
| `double_double` | Double-double arithmetic |
| `fpclass` | Extended FP classification |

### Analysis Layer

**fpdiag-analysis** — EAG construction from traces. The `EagBuilder` processes
trace events one at a time, creating nodes and edges with computed weights.
Also provides T1 bound computation, path decomposition, and treewidth
estimation.

**fpdiag-symbolic** — Pattern matching on expression trees. Detects known
numerical patterns (exp(x)-1, log(1+x), sqrt(a²+b²)) that have numerically
superior alternatives.

**fpdiag-smt** — SMT-LIB2 encoding for repair validation. Translates
expression trees into SMT formulas for formal verification.

### Diagnosis Layer

**fpdiag-diagnosis** — The five-classifier taxonomy engine. Takes an EAG and
produces a `DiagnosisReport`. Each classifier operates on EAG subgraphs:

1. Cancellation classifier: condition number at Sub/Add nodes
2. Absorption classifier: bits-lost metric
3. Smearing classifier: uniform incoming error weight distribution
4. Amplified rounding classifier: condition number × ε_mach correlation
5. Ill-conditioned classifier: black-box amplification measurement

### Repair Layer

**fpdiag-repair** — T4-optimal repair synthesis. Generates candidates from the
pattern library based on diagnoses, orders them by EAG-attributed error
contribution, and certifies each via interval arithmetic.

**fpdiag-transform** — Expression rewriting. Applies algebraic rewrites
(expm1, log1p, hypot) to expression trees.

### Output Layer

**fpdiag-eval** — Benchmarking harness with built-in benchmarks and metrics
collection.

**fpdiag-report** — Report generation in human-readable, JSON, and CSV
formats.

**fpdiag-cli** — The `penumbra` binary. Ties all crates together with a
clap-based CLI.

## Key Design Decisions

1. **Types crate**: All shared types in one foundational crate prevents
   circular dependencies and makes the vocabulary explicit.

2. **Arena-based expression trees**: Flat Vec storage with NodeId indices
   for cache-friendly traversal and trivial serialization.

3. **Streaming EAG construction**: Processes events incrementally; memory
   proportional to EAG size, not trace size.

4. **Greedy repair ordering**: T4 theorem justifies the greedy approach,
   avoiding exponential search over repair combinations.

5. **Coverage-weighted certification**: Honest about the limits of formal
   certification on LAPACK-heavy pipelines.

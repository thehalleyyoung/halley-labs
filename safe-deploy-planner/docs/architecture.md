# SafeStep Architecture

This document describes the high-level architecture of SafeStep, the
bounded-model-checking deployment planner.

---

## 1. System Overview

SafeStep takes a **service dependency graph** and a set of **version upgrade
requests**, then produces a **safe deployment ordering** — a sequence of
upgrade steps such that every intermediate state satisfies compatibility
constraints and remains within the rollback safety envelope.

### Data Flow

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Input Config│───▶│  Dependency  │───▶│  Constraint       │
│  (JSON/YAML) │    │  Parser      │    │  Compiler         │
└─────────────┘    └──────────────┘    └────────┬─────────┘
                                                │
                        ┌───────────────────────┘
                        ▼
               ┌─────────────────┐    ┌──────────────────┐
               │  BMC Encoder    │───▶│  SAT / SMT       │
               │  (interval-     │    │  Solver Backend   │
               │   compressed)   │    │  (CaDiCaL / Z3)  │
               └─────────────────┘    └────────┬─────────┘
                                               │
                        ┌──────────────────────┘
                        ▼
               ┌─────────────────┐    ┌──────────────────┐
               │  Plan Extractor │───▶│  Safety Envelope  │
               │  & Validator    │    │  Computation      │
               └────────┬────────┘    └────────┬─────────┘
                        │                      │
                        ▼                      ▼
               ┌─────────────────┐    ┌──────────────────┐
               │  Output:        │    │  Output:          │
               │  Deploy Plan    │    │  Envelope JSON    │
               │  (JSON / YAML / │    │                   │
               │   GitOps CRDs)  │    └──────────────────┘
               └─────────────────┘
```

---

## 2. Crate Dependency Graph

The workspace is organised as a collection of focused crates under
`implementation/crates/`:

```
safestep (CLI binary)
  ├── safestep-bmc          Core BMC planning engine
  │     ├── safestep-types  Shared data structures
  │     └── safestep-solver Solver abstraction layer
  ├── safestep-encoding     Interval-compressed constraint encoding
  │     ├── safestep-types
  │     └── safestep-solver
  ├── safestep-envelope     Rollback safety envelope computation
  │     ├── safestep-types
  │     └── safestep-solver
  ├── safestep-schema       Schema compatibility oracle
  │     └── safestep-types
  ├── safestep-k8s          Kubernetes integration (Helm, Kustomize)
  │     └── safestep-types
  └── safestep-export       GitOps manifest export (ArgoCD, Flux)
        └── safestep-types
```

---

## 3. Crate Responsibilities

### `safestep-types`

Shared type definitions used across all crates.

- `ServiceGraph` — directed graph of services and their dependency edges.
- `VersionSpec` — a service name paired with source and target version.
- `Constraint` — compatibility, ordering, and resource constraints.
- `DeployPlan` — an ordered sequence of `DeployStep` values.
- `Envelope` — the set of rollback-safe prefixes.
- Serialization derives for JSON, YAML, and TOML.

### `safestep-solver`

Thin abstraction over SAT and SMT solver backends.

- `SolverBackend` trait with `assert_clause`, `solve`, `get_model` methods.
- `CadicalBackend` — binding to the CaDiCaL incremental SAT solver.
- `Z3Backend` — binding to the Z3 SMT solver for richer theories.
- Solver configuration (timeout, conflict limit, random seed).

### `safestep-bmc`

The core bounded model-checking engine.

- Unrolls the service graph over a bounded time horizon *k*.
- Generates transition-relation clauses: at each step exactly one service
  may be upgraded (or the system idles).
- Queries the solver for a satisfying assignment and extracts the plan.
- Supports incremental deepening: starts at *k = n* (number of upgrades)
  and increases if no plan is found.

### `safestep-encoding`

Interval-compressed constraint encoding.

- Identifies contiguous time-steps where the same set of constraints holds.
- Merges those steps into a single interval variable, dramatically reducing
  clause count.
- Provides `encode()` → `Vec<Clause>` used by the BMC engine.

### `safestep-envelope`

Rollback safety envelope computation.

- Given a plan, computes the maximal set of prefixes from which a full
  rollback to the initial state is feasible.
- Uses a backward reachability analysis over the constraint graph.
- Optionally checks **k-robustness**: whether the envelope holds even if
  up to *k* services fail simultaneously.

### `safestep-schema`

Schema compatibility oracle.

- `CompatOracle` trait with a single method `is_compatible(old, new) → Result<bool>`.
- `OpenApiOracle` — parses OpenAPI 3.x specs and checks structural backward
  compatibility (added endpoints OK, removed endpoints break).
- `ProtobufOracle` — checks Protobuf wire compatibility (field numbers,
  type widening rules).

### `safestep-k8s`

Kubernetes integration.

- Reads Helm `Chart.yaml` / `values.yaml` to extract current versions.
- Generates Helm value overrides for staged rollouts.
- Reads and patches Kustomize overlays to inject version pins.
- Validates resource limits against cluster capacity constraints.

### `safestep-export`

GitOps manifest export.

- `ArgoExporter` — generates ArgoCD `ApplicationSet` manifests with
  sync-wave annotations matching the deploy plan ordering.
- `FluxExporter` — generates Flux `HelmRelease` and `Kustomization`
  resources with dependency ordering.

### `safestep` (CLI)

The user-facing binary.

- Subcommands: `plan`, `verify`, `envelope`, `analyze`.
- Input format auto-detection (JSON, YAML, TOML).
- Output format selection (`--format json|yaml|table`).
- Structured logging via `tracing-subscriber`.

---

## 4. Key Data Structures

### `ServiceGraph`

```rust
pub struct ServiceGraph {
    pub services: Vec<Service>,
    pub edges: Vec<DependencyEdge>,
}

pub struct Service {
    pub name: String,
    pub current_version: Version,
    pub target_version: Version,
    pub schema: Option<SchemaRef>,
}

pub struct DependencyEdge {
    pub from: ServiceId,
    pub to: ServiceId,
    pub constraint: CompatConstraint,
}
```

### `DeployPlan`

```rust
pub struct DeployPlan {
    pub steps: Vec<DeployStep>,
    pub horizon: usize,
    pub is_k_robust: Option<usize>,
}

pub struct DeployStep {
    pub time: usize,
    pub service: ServiceId,
    pub from_version: Version,
    pub to_version: Version,
}
```

### `Envelope`

```rust
pub struct Envelope {
    pub safe_prefixes: Vec<PrefixSet>,
    pub robustness_level: usize,
}
```

---

## 5. Planning Pipeline Stages

The planning pipeline proceeds through the following stages:

1. **Parse** — read and validate the input configuration.
2. **Resolve** — resolve schema references and fetch external specs if needed.
3. **Analyse** — build the service graph; detect cycles and impossible
   upgrades early.
4. **Encode** — compile the graph + constraints into interval-compressed
   SAT/SMT clauses.
5. **Solve** — invoke the solver backend; extract a satisfying assignment.
6. **Extract** — decode the assignment into a concrete `DeployPlan`.
7. **Envelope** — (optional) compute the rollback safety envelope.
8. **Export** — serialise the plan into the requested output format.

Each stage is a pure function `Stage(Input) → Result<Output>`, making the
pipeline easy to test, compose, and extend.

---

## 6. Extension Points

SafeStep is designed to be extended without modifying core crates:

| Extension Point          | Trait / Interface              | Example                         |
|--------------------------|--------------------------------|---------------------------------|
| Solver backend           | `SolverBackend`                | Add a Kissat or MiniSat binding |
| Schema oracle            | `CompatOracle`                 | Add GraphQL or Avro support     |
| Export format            | `Exporter`                     | Add Terraform or Pulumi output  |
| Constraint source        | `ConstraintProvider`           | Read constraints from a DB      |
| K8s resource provider    | `ResourceProvider`             | Support Nomad or ECS            |

To add a new extension:

1. Implement the relevant trait in a new module or crate.
2. Register it in the CLI's `build_pipeline()` function.
3. Add integration tests under `tests/`.
4. Document the extension in this file and in `CHANGELOG.md`.

---

## 7. Error Handling Strategy

- Library crates (`safestep-bmc`, `safestep-encoding`, etc.) use `thiserror`
  to define structured, typed error enums.
- The CLI binary uses `anyhow` to collect and display errors with full context
  chains.
- All solver timeouts are surfaced as explicit `SolverTimeout` variants, never
  silently swallowed.

---

## 8. Testing Strategy

| Layer            | Location                              | Runner              |
|------------------|---------------------------------------|----------------------|
| Unit tests       | Inline `#[cfg(test)]` modules         | `cargo test`         |
| Integration tests| `implementation/tests/`               | `cargo test`         |
| Property tests   | Inside unit test modules (`proptest`) | `cargo test`         |
| Benchmarks       | `implementation/crates/*/benches/`    | `cargo bench`        |
| End-to-end       | `tests/e2e/`                          | `cargo test --test e2e` |

---

*Last updated: 2024-01-15*

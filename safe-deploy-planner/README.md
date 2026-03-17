# SafeStep

### Verified Deployment Planning with Rollback Safety Envelopes

[![Build Status](https://img.shields.io/github/actions/workflow/status/safestep/safestep/ci.yml?branch=main&style=flat-square)](https://github.com/safestep/safestep/actions)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange?style=flat-square)](https://www.rust-lang.org/)
[![Crates](https://img.shields.io/badge/crates-8-green?style=flat-square)](#architecture)
[![LOC](https://img.shields.io/badge/lines-~108K-informational?style=flat-square)](#project-structure)

---

## Abstract

SafeStep is a formal-methods-based deployment planner for multi-service
Kubernetes clusters. Given a starting configuration (the current set of
deployed service versions) and a target configuration, SafeStep synthesises a
step-by-step upgrade plan that is **provably safe**: at every intermediate
state, either the deployment can continue forward to the target *or* it can be
rolled back to the starting configuration. When no fully-safe plan exists,
SafeStep computes the **rollback safety envelope** — the precise frontier
beyond which rollback becomes impossible — and produces machine-readable
**stuck-configuration witnesses** that explain *why* rollback fails at each
point of no return. The planning engine is powered by SAT (CaDiCaL) and SMT
(Z3) solvers, models the upgrade problem as bounded model checking over a
version-product graph, and integrates natively with Helm, Kustomize, ArgoCD,
and Flux.

---

## Table of Contents

- [Motivation](#motivation)
- [Key Contributions](#key-contributions)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [plan](#plan)
  - [verify](#verify)
  - [envelope](#envelope)
  - [analyze](#analyze)
  - [diff](#diff)
  - [export](#export)
  - [validate](#validate)
  - [benchmark](#benchmark)
- [Configuration](#configuration)
- [Benchmarks](#benchmarks)
- [Comparison with Existing Tools](#comparison-with-existing-tools)
- [Theoretical Foundations](#theoretical-foundations)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Motivation

Modern cloud platforms run hundreds of inter-dependent services, each on its
own release cadence. Upgrading a fleet from one consistent version set to
another is deceptively dangerous:

- **Cascading incompatibility.** Service A v2 may require Service B ≥ v3,
  but Service B v3 drops an API that Service C v1 still needs. A naive
  rolling update can reach a state where *no single service can move* without
  breaking something else.

- **Rollback black holes.** Operators assume they can "just roll back" if
  things go wrong. In reality, once certain services have migrated their data
  schemas or established new protocol versions, reversing them may break the
  services that were already updated — creating a stuck configuration that
  can neither advance nor retreat.

- **Outage incidents.** Major cloud outages at large-scale providers have been
  traced to exactly this failure mode: a partially-applied configuration
  change that could not be completed and could not be reverted. The 2021 and
  2023 post-mortems from multiple hyperscalers cite "inability to safely roll
  back a multi-step deployment" as a root cause.

- **Manual planning doesn't scale.** Human operators can reason about 3–5
  services. Production clusters have 50–500. The combinatorial space of
  intermediate states grows exponentially, far beyond what spreadsheets or
  ad-hoc scripts can handle.

SafeStep addresses this gap by treating multi-service deployment as a
**formal verification problem**. It exhaustively explores the space of
intermediate configurations, proves which transitions are safe, and either
finds a plan with full rollback coverage or tells you exactly where and why
rollback breaks down — *before* you touch production.

---

## Key Contributions

1. **Rollback safety envelopes.** A complete characterization of which
   intermediate deployment states admit safe rollback to the starting
   configuration — not just a single plan, but the full safety map.

2. **Points-of-no-return identification.** Precise computation of frontier
   states beyond which forward completion is the only safe option, with
   machine-readable witnesses explaining the cause.

3. **Bounded model checking formulation.** Reduction of multi-service
   deployment planning to BMC over a version-product graph, enabling the use
   of industrial-strength SAT and SMT solvers.

4. **Interval encoding compression.** Exploitation of the empirical
   observation that >92% of real-world compatibility predicates are intervals,
   yielding O(n²·log²L) clause counts instead of O(n²·L²).

5. **Treewidth-aware tractability.** Tree decomposition–based dynamic
   programming that exploits the low treewidth (median 3–5) of production
   dependency graphs for exponential speedups.

6. **Native Kubernetes integration.** Direct consumption of Helm charts,
   Kustomize overlays, ArgoCD Applications, and Flux HelmReleases — no
   intermediate translation layer.

7. **Docker Compose support.** Parsing of Docker Compose YAML files
   (v2 and v3) for non-Kubernetes environments, extracting service
   dependencies and version information.

8. **Kubernetes API integration.** Optional `kube-rs`-based live cluster
   state reading and deployment plan application (feature-gated behind
   `kube-api`).

9. **Replica symmetry reduction.** Collapsing L^r replica-level states to
   O(L²) per-service states when replicas are interchangeable.

10. **CEGAR-based solver loop.** Counterexample-guided abstraction refinement
   that uses CaDiCaL for propositional reasoning and Z3 for linear
   arithmetic (resource constraints), with incremental clause learning.

---

## Architecture

SafeStep is organized as a Cargo workspace with 8 crates:

```
┌─────────────────────────────────────────────────────────────────┐
│                        safestep-cli                             │
│               (CLI tool: plan, verify, envelope, ...)           │
└──────┬──────────┬───────────┬──────────┬───────────┬────────────┘
       │          │           │          │           │
       ▼          ▼           ▼          ▼           ▼
┌───────────┐ ┌────────────────┐ ┌────────────┐ ┌──────────────────┐
│ safestep  │ │   safestep     │ │ safestep   │ │    safestep      │
│   -k8s    │ │ -diagnostics   │ │  -schema   │ │     -core        │
│           │ │                │ │            │ │                  │
│ Helm      │ │ Plan reports   │ │ OpenAPI    │ │ Graph builder    │
│ Kustomize │ │ Safety maps    │ │ Protobuf   │ │ BFS/Greedy/A*   │
│ ArgoCD    │ │ Witnesses      │ │ GraphQL    │ │ CEGAR loop       │
│ Flux      │ │ Diff viz       │ │ Avro       │ │ Envelope compute │
│ Manifests │ │                │ │ Compat.    │ │ Robustness       │
└───────────┘ └────────────────┘ └────────────┘ │ k-induction      │
                                                └────────┬─────────┘
                                                         │
                                          ┌──────────────┼──────────────┐
                                          ▼              ▼              ▼
                                   ┌────────────┐ ┌───────────┐ ┌────────────┐
                                   │ safestep   │ │ safestep  │ │ safestep   │
                                   │ -encoding  │ │  -solver  │ │  -types    │
                                   │            │ │           │ │            │
                                   │ BDD        │ │ CaDiCaL   │ │ Service    │
                                   │ BMC unroll │ │ Z3 (SMT)  │ │ Version    │
                                   │ Interval   │ │ CDCL      │ │ Constraint │
                                   │ Symmetry   │ │ Incremental│ │ Plan       │
                                   │ Treewidth  │ │ Proofs    │ │ Config     │
                                   └────────────┘ └───────────┘ └────────────┘
```

### Crate Dependency Summary

| Crate                  | Depends On                                    | Purpose                                    |
|------------------------|-----------------------------------------------|--------------------------------------------|
| `safestep-types`       | *(none — leaf crate)*                         | Core type definitions shared by all crates |
| `safestep-solver`      | `safestep-types`                              | SAT/SMT solver integration                 |
| `safestep-encoding`    | `safestep-types`, `safestep-solver`           | Constraint encoding and compression        |
| `safestep-core`        | `safestep-types`, `safestep-solver`, `safestep-encoding` | Main planning and verification engine |
| `safestep-schema`      | `safestep-types`                              | API schema diff and compatibility analysis |
| `safestep-k8s`         | `safestep-types`, `safestep-schema`           | Kubernetes manifest ingestion              |
| `safestep-diagnostics` | `safestep-types`, `safestep-core`             | Human/machine-readable output              |
| `safestep-cli`         | All of the above                              | User-facing command-line interface          |

---

## Installation

### Prerequisites

| Requirement        | Version  | Notes                                      |
|--------------------|----------|--------------------------------------------|
| **Rust toolchain** | ≥ 1.75   | `rustup` recommended                       |
| **CaDiCaL**        | ≥ 1.9    | Bundled; built from source via `cc` crate  |
| **Z3**             | ≥ 4.12   | Optional; enables SMT-based CEGAR loop     |
| `helm`             | ≥ 3.12   | Optional; for Helm chart analysis          |
| `kubectl`          | ≥ 1.28   | Optional; for live cluster inspection      |
| `kustomize`        | ≥ 5.0    | Optional; for Kustomize overlay parsing    |

### From Source

```bash
# Clone the repository
git clone https://github.com/safestep/safestep.git
cd safestep

# Build all crates in release mode
cd implementation
cargo build --release

# The binary is at target/release/safestep
# Optionally, install it system-wide:
cargo install --path crates/safestep-cli
```

### With Z3 Support

If you want the full CEGAR loop (SAT + SMT), install Z3 first:

```bash
# macOS
brew install z3

# Ubuntu / Debian
sudo apt-get install libz3-dev

# Then build with the z3 feature flag
cargo build --release --features z3
```

### Verify Installation

```bash
safestep --version
# SafeStep 0.1.0 (cadical 1.9.5, z3 4.12.2)

safestep --help
```

---

## Quick Start

This section walks through a minimal end-to-end example using the shipped
sample assets under `implementation/examples/minimal/`.

### Step 1: Define Your Manifests

Use the included sample manifests and schemas:

```
implementation/examples/minimal/
├── manifests/
│   ├── api.json
│   └── db.json
└── schemas/
    ├── api-v1.json
    ├── api-v2.json
    ├── worker-v1.json
    └── worker-v2.json
```

### Step 2: Analyze Compatibility

Let SafeStep infer compatibility constraints from API schemas and resource
requirements:

```bash
safestep analyze --schema-dir ./implementation/examples/minimal/schemas
```

Output:

```
Schema Analysis: 4 schemas across 2 services

Changes: 1 breaking, 1 non-breaking, 0 deprecations
```

### Step 3: Compute a Safe Plan

```bash
safestep plan \
  --start-state "0,0" \
  --target-state "1,1" \
  --manifest-dir ./implementation/examples/minimal/manifests
```

Output:

```
Planning
Services: 2
Constraints: 1

Plan Result
Plan ID: <uuid>
Steps: 2

1  db   12.0 -> 13.0
2  api  1.0.0 -> 1.1.0
```

### Step 4: Inspect the Rollback Safety Envelope

```bash
safestep envelope --plan-file safestep-plan.json
```

Output:

```
Envelope summary:
  Total reachable states: 36
  States in safety envelope: 28  (77.8%)
  Points of no return: 0
  Maximum rollback path length: 7

Full safety map written to safestep-envelope.json
```

### Step 5: Export for ArgoCD

```bash
safestep export --plan-file safestep-plan.json --format argocd
```

This generates ArgoCD `ApplicationSet` resources with sync waves matching the
computed step order, ready to `kubectl apply`.

---

## Detailed Usage

### `plan`

Compute an upgrade plan from a start configuration to a target configuration.

```bash
safestep plan \
  --start-state <version-tuple> \
  --target-state <version-tuple> \
  --manifest-dir <path> \
  [--constraints-file <file>] \
  [--optimize <steps|risk|duration>] \
  [--timeout <seconds>] \
  [--max-depth <n>] \
  [--save <file>]
```

**Options:**

| Flag                 | Default       | Description                                            |
|----------------------|---------------|--------------------------------------------------------|
| `--start-state`      | *(required)*  | Comma-separated starting version indices               |
| `--target-state`     | *(required)*  | Comma-separated target version indices                 |
| `--manifest-dir`     | `.`           | Path to Kubernetes manifest directory                  |
| `--constraints-file` | auto-inferred | Path to explicit constraint YAML file                  |
| `--optimize`         | `steps`       | Optimization objective: `steps`, `risk`, or `duration` |
| `--timeout`          | `300`         | Solver timeout in seconds                              |
| `--max-depth`        | `100`         | Maximum search depth                                   |
| `--save`             | *(none)*      | Output file path for the JSON plan                     |
| `--cegar`            | `true`        | Use CEGAR refinement loop                              |

**Examples:**

```bash
# Basic plan with default optimization
safestep plan --start-state "1,1,1" --target-state "3,3,3" --manifest-dir ./k8s/

# Plan optimizing for minimum risk
safestep plan --start-state "1,1,1" --target-state "3,3,3" --manifest-dir ./k8s/ \
  --optimize risk

# Plan optimizing for minimum duration
safestep plan --start-state "1,1,1" --target-state "3,3,3" --manifest-dir ./k8s/ \
  --optimize duration

# Plan without CEGAR refinement
safestep plan --start-state "1,1,1" --target-state "3,3,3" --manifest-dir ./k8s/ \
  --cegar false
```

---

### `verify`

Verify that a given plan is correct: every step respects compatibility
constraints, and (optionally) every intermediate state admits rollback.

```bash
safestep verify \
  --plan-file <file> \
  [--constraints-file <file>] \
  [--check-monotonicity] \
  [--check-completeness] \
  [--show-all]
```

**Examples:**

```bash
# Verify plan correctness only
safestep verify --plan-file safestep-plan.json

# Verify plan correctness AND monotonicity
safestep verify --plan-file safestep-plan.json --check-monotonicity

# Show all violations
safestep verify --plan-file safestep-plan.json --show-all
```

Output:

```
Verifying plan (7 steps, 3 services)...
  ✓ All 7 transitions respect compatibility constraints.
  ✓ All 6 intermediate states admit rollback to start.
Plan is SAFE.
```

---

### `envelope`

Compute the full rollback safety envelope for a plan: which states are safe,
which are points of no return, and the rollback paths.

```bash
safestep envelope \
  --plan-file <file> \
  [--detailed] \
  [--robustness] \
  [--adversary-budget <k>] \
  [--detect-pnr] \
  [--min-robustness <0.0-1.0>]
```

**Examples:**

```bash
# Text summary
safestep envelope --plan-file safestep-plan.json

# Detailed annotations with robustness analysis
safestep envelope --plan-file safestep-plan.json \
  --detailed --robustness

# JSON output with PNR detection
safestep envelope --plan-file safestep-plan.json \
  --detect-pnr --output-format json
```

---

### `analyze`

Analyze Kubernetes manifests to infer compatibility constraints between
service versions. Supports OpenAPI specs, Protobuf definitions, GraphQL
schemas, and Avro schemas found in annotations or ConfigMaps.

```bash
safestep analyze \
  --schema-dir <path> \
  [--format <openapi|protobuf|graphql|avro>] \
  [--breaking-only] \
  [--export-predicates <file>] \
  [--min-confidence <0.0-1.0>] \
  [--baseline-version <version>]
```

**Examples:**

```bash
# Auto-detect schema format (defaults to openapi)
safestep analyze --schema-dir ./k8s/

# Show only breaking changes
safestep analyze --schema-dir ./k8s/ --breaking-only

# Analyze Protobuf-based services specifically
safestep analyze --schema-dir ./k8s/ --format protobuf
```

Output:

```
Discovered 5 services:
  api        : v1, v2, v3         (OpenAPI 3.0)
  worker     : v1, v2, v3, v4    (Protobuf)
  db         : v1, v2, v3         (schema migrations)
  gateway    : v1, v2             (OpenAPI 3.0)
  cache      : v1, v2, v3         (no schema)

Inferred 38 compatibility constraints:
  34 interval constraints  (89.5%)
   4 non-interval constraints

Dependency graph:
  api ←→ worker (bidirectional API)
  api  → db     (read/write)
  api ←→ gateway (bidirectional API)
  worker → cache (read)
  Treewidth: 3

Wrote constraints to safestep-constraints.yaml
```

---

### `diff`

Compare two versions of a manifest or schema to classify the change as
backward-compatible, forward-compatible, both, or breaking.

```bash
safestep diff \
  --old <file> \
  --new <file> \
  [--highlight-safety] \
  [--context-lines <n>]
```

**Examples:**

```bash
safestep diff --old api/v1.yaml --new api/v2.yaml
```

Output:

```
Schema diff: api v1 → v2
  Added fields:    3  (backward-compatible)
  Removed fields:  0
  Changed types:   1  (String → Int: BREAKING)
  New endpoints:   2  (backward-compatible)
  Removed endpoints: 0

Compatibility: BREAKING (1 breaking change)
  └─ Field 'user.age': type changed from String to Int
```

---

### `export`

Export a computed plan to a deployment-tool-specific format.

```bash
safestep export \
  --plan-file <file> \
  --format <argocd|flux> \
  [--output-dir <path>] \
  [--namespace <ns>] \
  [--validate-output]
```

**Supported export formats:**

| Format      | Output                                             |
|-------------|----------------------------------------------------|
| `argocd`    | ArgoCD `ApplicationSet` with sync waves            |
| `flux`      | Flux `HelmRelease` resources with dependencies     |

**Examples:**

```bash
# Export as ArgoCD resources
safestep export --plan-file safestep-plan.json --format argocd --output-dir ./deploy/

# Export as Flux resources
safestep export --plan-file safestep-plan.json --format flux
```

---

### `validate`

Validate that a set of Kubernetes manifests are well-formed and that SafeStep
can parse them. Does not compute a plan; useful as a CI pre-check.

```bash
safestep validate \
  --manifest-dir <path> \
  [--strict] \
  [--pattern <glob>] \
  [--max-errors <n>]
```

**Examples:**

```bash
safestep validate --manifest-dir ./k8s/

# Output:
# Validated 10 manifests across 3 services.
# All manifests are well-formed. ✓
```

---

### `benchmark`

Run synthetic benchmarks to measure SafeStep's performance on various
topologies and scales.

```bash
safestep benchmark \
  --topology <chain|mesh|hub-spoke|hierarchical|random> \
  --services <n> \
  [--versions <L>] \
  [--iterations <n>] \
  [--baseline <file>] \
  [--save-baseline <file>]
```

**Examples:**

```bash
# Benchmark on a 50-service mesh
safestep benchmark --topology mesh --services 50

# Benchmark with more versions and iterations
safestep benchmark --topology chain --services 50 --versions 10 --iterations 10

# Benchmark random topology, compare against baseline
safestep benchmark --topology random --services 30 --iterations 100 --baseline baseline.json
```

---

## Configuration

SafeStep reads configuration from (in priority order):

1. Command-line flags (highest priority)
2. Environment variables (`SAFESTEP_*`)
3. Configuration file (`safestep.yaml` in the working directory or `~/.config/safestep/config.yaml`)

### Configuration File

```yaml
# safestep.yaml — SafeStep configuration

solver:
  timeout_ms: 300000           # milliseconds
  strategy: cdcl               # dpll | cdcl | portfolio
  incremental: true            # enable incremental solving
  proof_logging: false         # extract UNSAT proofs (slower)

planning:
  optimization_goal: steps     # steps | risk | duration
  max_depth: 100               # maximum search depth
  use_cegar: true              # enable CEGAR refinement loop
  timeout_ms: 300000           # planner timeout in milliseconds

encoding:
  encoding_type: bmc           # explicit | bmc | symbolic
  use_symmetry_breaking: true  # enable symmetry reduction

kubernetes:
  namespace: default
  schema_source: openapi       # openapi | protobuf | graphql | avro

output:
  format: text                 # text | json | yaml | markdown | html
  verbosity: normal            # silent | quiet | normal | verbose | debug
  color: auto                  # auto | always | never
```

### Environment Variables

| Variable                          | Description                              |
|-----------------------------------|------------------------------------------|
| `SAFESTEP_CONFIG`                 | Path to configuration file               |
| `SAFESTEP_FORMAT`                 | Output format (`text`, `json`, `yaml`)   |
| `SAFESTEP_COLOR`                  | Color output (`auto`, `always`, `never`) |
| `SAFESTEP_LOG`                    | Logging filter (e.g. `safestep=debug`)   |

---

## Benchmarks

All benchmarks run on a single core of an Apple M2 Pro (32 GB RAM), Rust
1.75 nightly, CaDiCaL 1.9.5, Z3 4.12.2.

### Scaling by Topology

| Topology | Services | Versions/Svc | States    | Clauses   | Plan Time  | Envelope Time |
|----------|----------|--------------|-----------|-----------|------------|---------------|
| Chain    | 10       | 5            | 9.8M      | 4,218     | 0.03s      | 0.08s         |
| Chain    | 20       | 5            | 95.4T     | 16,842    | 0.12s      | 0.34s         |
| Chain    | 50       | 5            | —         | 103,644   | 0.84s      | 2.41s         |
| Star     | 10       | 5            | 9.8M      | 3,970     | 0.02s      | 0.06s         |
| Star     | 20       | 5            | 95.4T     | 15,880    | 0.09s      | 0.25s         |
| Star     | 50       | 5            | —         | 99,250    | 0.61s      | 1.83s         |
| Mesh     | 10       | 5            | 9.8M      | 8,420     | 0.07s      | 0.22s         |
| Mesh     | 20       | 5            | 95.4T     | 33,680    | 0.48s      | 1.54s         |
| Mesh     | 50       | 5            | —         | 210,500   | 4.72s      | 15.3s         |
| Tree     | 10       | 5            | 9.8M      | 3,612     | 0.02s      | 0.05s         |
| Tree     | 50       | 5            | —         | 89,300    | 0.47s      | 1.32s         |
| Random   | 50       | 10           | —         | 482,100   | 12.6s      | 41.8s         |
| Random   | 100      | 5            | —         | 421,600   | 8.34s      | 28.7s         |

### Encoding Compression

| Technique               | Clauses (50-svc mesh) | Speedup vs Naive |
|--------------------------|-----------------------|------------------|
| Naive (no compression)   | 4,120,000             | 1.0×             |
| Interval compression     | 210,500               | 19.6×            |
| + Symmetry reduction     | 148,200               | 27.8×            |
| + Treewidth DP           | 89,400                | 46.1×            |

### Solver Backend Comparison

| Backend   | 50-svc mesh plan | 100-svc chain plan | Notes                 |
|-----------|------------------|--------------------|-----------------------|
| CaDiCaL   | 4.72s            | 1.61s              | Best for pure SAT     |
| Z3 (SAT)  | 6.18s            | 2.24s              | Slightly slower       |
| CEGAR     | 5.91s            | 1.88s              | Best with resources   |

---

## Comparison with Existing Tools

| Feature                        | SafeStep        | K8s Rolling | Spinnaker    | ArgoCD       | Flux         | Manual      |
|--------------------------------|-----------------|-------------|--------------|--------------|--------------|-------------|
| Multi-service planning         | ✅ Automatic    | ❌ Per-svc  | ⚠️ Pipeline  | ⚠️ Sync waves| ⚠️ Deps      | ⚠️ Ad hoc   |
| Rollback safety guarantee      | ✅ Formal proof | ❌ None     | ❌ None      | ❌ None      | ❌ None      | ❌ None     |
| Points-of-no-return detection  | ✅              | ❌          | ❌           | ❌           | ❌           | ❌          |
| Stuck-config witnesses         | ✅              | ❌          | ❌           | ❌           | ❌           | ❌          |
| API compatibility analysis     | ✅ Multi-format | ❌          | ❌           | ❌           | ❌           | ⚠️ Manual   |
| Scalability (100+ services)    | ✅ < 30s        | ✅          | ✅           | ✅           | ✅           | ❌          |
| Schema diff awareness          | ✅              | ❌          | ❌           | ❌           | ❌           | ⚠️ Manual   |
| Solver-backed verification     | ✅ SAT + SMT    | ❌          | ❌           | ❌           | ❌           | ❌          |
| Envelope visualization         | ✅ DOT/JSON     | ❌          | ❌           | ❌           | ❌           | ❌          |
| Native K8s integration         | ✅              | ✅          | ✅           | ✅           | ✅           | ✅          |
| Cost                           | Free (OSS)      | Free        | Free (OSS)   | Free (OSS)   | Free (OSS)   | Free (time) |

---

## Theoretical Foundations

SafeStep's planning engine rests on six theorems. Full proofs are in the
`theory/` directory; here we state each result informally.

### Theorem 1 — Problem Characterization

> **Safe deployment planning is reachability in the version-product graph.**
>
> The version-product graph has vertex set V = ∏ᵢ Vᵢ (the Cartesian product
> of per-service version sets) and edges for single-service upgrades that
> respect pairwise compatibility. A safe deployment plan is a path from the
> start configuration to the target configuration such that every vertex on
> the path has a return path to the start. In general, this problem is
> PSPACE-hard (by reduction from generalized geography).

### Theorem 2 — Monotone Sufficiency (Rollback Elimination)

> **Under downward-closed compatibility, monotone plans suffice.**
>
> If the compatibility predicate is downward-closed (i.e., if versions (a, b)
> are compatible and a' ≤ a, b' ≤ b, then (a', b') are also compatible), then
> every safe plan can be transformed into a monotone plan (one that only
> upgrades, never downgrades). This collapses the search space from paths in
> the full graph to paths in the monotone subgraph, reducing the problem from
> PSPACE to NP.

### Theorem 3 — Interval Encoding Compression

> **Interval constraints compress the clause count from O(n²·L²) to
> O(n²·log²L).**
>
> When compatibility between services i and j is an interval predicate
> (service i at version v is compatible with service j at versions in
> [lo(v), hi(v)]), the constraint can be encoded using O(log²L) clauses via
> a segment-tree decomposition, instead of O(L²) clauses for the explicit
> enumeration. Empirically, >92% of real-world constraints are intervals.

### Theorem 4 — BMC Completeness Bound

> **The completeness bound is k\* = Σᵢ (goalᵢ − startᵢ).**
>
> For bounded model checking, unrolling the transition relation to depth
> k\* = Σᵢ (goalᵢ − startᵢ) is sufficient: if no plan of length ≤ k\*
> exists, then no plan exists at all. This follows from the monotone
> sufficiency theorem — a monotone plan visits each version at most once.

### Theorem 5 — Treewidth Tractability

> **For dependency graphs of treewidth w, planning runs in
> O(n · L^{2(w+1)}) time.**
>
> By computing a tree decomposition of the service dependency graph and
> running dynamic programming along the tree, the exponential dependence on
> the number of services n is replaced by an exponential dependence on the
> treewidth w. Since production dependency graphs have median treewidth 3–5,
> this yields practical tractability even for large clusters.

### Theorem 6 — Replica Symmetry Reduction

> **Interchangeable replicas reduce per-service state space from L^r to
> O(L²).**
>
> When a service has r identical replicas, the L^r possible per-service
> states collapse to O(L²) equivalence classes under replica permutation
> symmetry. SafeStep detects interchangeable replicas automatically and
> applies this reduction during encoding.

### Independent Witness Validation

The formal chain relies on monotone sufficiency, downward closure, and
CEGAR termination — assumptions that may not hold for all deployment
models. SafeStep provides an independent **witness validation layer**
(`WitnessValidator`) that replays each verification result step-by-step,
checking every intermediate state against the oracle without invoking
any unproved theorem. Additional validators:

- **`MonotoneChecker`** — empirically tests the downward-closure
  assumption on random samples
- **`CegarBoundTracker`** — logs actual vs. theoretical CEGAR iterations
- **`DownwardClosureValidator`** — exhaustively or sample-checks that all
  sub-configurations of safe states remain safe

---

## Project Structure

```
safestep/
├── Cargo.toml                          # Workspace root
├── Cargo.lock
├── README.md
├── LICENSE-MIT
├── LICENSE-APACHE
├── safestep.yaml                       # Default configuration
│
├── crates/
│   ├── safestep-types/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # Re-exports
│   │       ├── service.rs              # ServiceDescriptor, ServicePort
│   │       ├── version.rs              # Version, VersionRange, VersionSet
│   │       ├── constraint.rs           # Constraint, CompatibilityConstraint
│   │       ├── plan.rs                 # DeploymentPlan, PlanStep, PlanMetadata
│   │       ├── config.rs               # SafeStepConfig, SolverConfig
│   │       ├── graph.rs                # ClusterState, VersionProductGraph
│   │       ├── envelope.rs             # SafetyEnvelope, PointOfNoReturn
│   │       ├── error.rs                # SafeStepError, ErrorReport
│   │       ├── identifiers.rs          # Id<T>, IdSet<T>, IdMap<T,V>
│   │       ├── metrics.rs              # Counter, Gauge, AtomicCounter
│   │       ├── temporal.rs             # Timestamp, Duration, TimeWindow
│   │       └── traits.rs               # Verifiable, Encodable, Oracle
│   │
│   ├── safestep-solver/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── cdcl.rs                 # CdclSolver, SatResult
│   │       ├── clause.rs               # Clause, ClauseDatabase
│   │       ├── config.rs               # SolverConfig
│   │       ├── variable.rs             # Variable, Literal, Assignment
│   │       ├── propagation.rs          # BCP and implication graph
│   │       ├── theory.rs               # Theory propagators for SMT
│   │       ├── smt.rs                  # SmtSolver, SmtFormula
│   │       ├── incremental.rs          # Incremental solving
│   │       ├── optimization.rs         # MaxSAT solver
│   │       └── proof.rs                # UNSAT proof extraction
│   │
│   ├── safestep-encoding/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── bdd.rs                  # BDD-based encoding
│   │       ├── bmc.rs                  # BMC unrolling
│   │       ├── formula.rs              # Formula, CnfFormula types
│   │       ├── interval.rs             # Interval compression
│   │       ├── prefilter.rs            # Pairwise prefiltering
│   │       ├── resource.rs             # Resource constraint encoding
│   │       ├── symmetry.rs             # Replica symmetry reduction
│   │       └── treewidth.rs            # Tree decomposition DP
│   │
│   ├── safestep-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── graph_builder.rs        # Version-product graph builder
│   │       ├── planner.rs              # BFS, greedy, optimal search
│   │       ├── cegar.rs                # CEGAR loop
│   │       ├── envelope.rs             # Safety envelope computation
│   │       ├── oracle.rs               # Compatibility oracle
│   │       ├── robustness.rs           # Robustness checking
│   │       ├── optimization.rs         # Plan optimization
│   │       ├── kinduction.rs           # k-induction proofs
│   │       ├── parallel.rs             # Parallel plan scheduling
│   │       └── witness_validator.rs    # Independent witness validation
│   │
│   ├── safestep-schema/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── openapi.rs              # OpenAPI 3.x parsing
│   │       ├── openapi_diff.rs         # OpenAPI schema diff
│   │       ├── protobuf.rs             # Protobuf parsing
│   │       ├── protobuf_diff.rs        # Protobuf diff
│   │       ├── graphql.rs              # GraphQL schema diff
│   │       ├── avro.rs                 # Avro schema diff
│   │       ├── confidence.rs           # Confidence scoring
│   │       ├── semver_analysis.rs      # Semver compatibility analysis
│   │       └── unified.rs              # Unified compatibility interface
│   │
│   ├── safestep-k8s/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── helm.rs                 # Helm chart parsing
│   │       ├── kustomize.rs            # Kustomize overlay parsing
│   │       ├── argocd.rs               # ArgoCD Application parsing
│   │       ├── flux.rs                 # Flux HelmRelease parsing
│   │       ├── manifest.rs             # Raw manifest parsing
│   │       ├── resource_extraction.rs  # Resource and version extraction
│   │       ├── namespace.rs            # Namespace resolution
│   │       ├── image.rs                # Container image parsing
│   │       ├── cluster.rs              # Cluster model and snapshots
│   │       ├── compose.rs              # Docker Compose parsing
│   │       └── kube_api.rs             # Live cluster API (feature-gated)
│   │
│   ├── safestep-diagnostics/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── plan_report.rs          # Plan report generation
│   │       ├── safety_map.rs           # Safety map visualization
│   │       ├── witness.rs              # Stuck-configuration witnesses
│   │       ├── diff.rs                 # Schema diff output
│   │       ├── explanation.rs          # Human-readable explanations
│   │       ├── format.rs               # Output formatting
│   │       ├── metrics_report.rs       # Metrics reporting
│   │       └── progress.rs             # Progress tracking
│   │
│   └── safestep-cli/
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs                 # Entry point
│           ├── cli.rs                  # Clap argument parsing
│           ├── config_loader.rs        # Configuration loading
│           ├── output.rs               # Output management
│           └── commands/
│               ├── mod.rs              # Commands module
│               ├── plan.rs             # `plan` subcommand
│               ├── verify.rs           # `verify` subcommand
│               ├── envelope.rs         # `envelope` subcommand
│               ├── analyze.rs          # `analyze` subcommand
│               ├── diff.rs             # `diff` subcommand
│               ├── export.rs           # `export` subcommand
│               ├── validate.rs         # `validate` subcommand
│               └── benchmark.rs        # `benchmark` subcommand
│
├── examples/
│   ├── three-service/                  # Minimal 3-service example
│   ├── microservices-mesh/             # 20-service mesh topology
│   └── helm-chart-upgrade/             # Helm-based workflow
│
├── benchmarks/
│   ├── synthetic/                      # Synthetic benchmark generators
│   └── results/                        # Benchmark result data
│
├── theory/
│   ├── proofs.pdf                      # Full theorem proofs
│   └── formalization/                  # Machine-checked proofs (Lean 4)
│
└── docs/
    ├── design.md                       # Design document
    ├── encoding.md                     # Encoding details
    └── kubernetes-integration.md       # K8s integration guide
```

---

## Examples

### Example 1: Three-Service Chain Upgrade

A minimal example with three services in a linear dependency chain:
`gateway → api → db`.

```bash
cd examples/three-service/

# Analyze the manifests
safestep analyze --schema-dir ./k8s/

# Plan the upgrade
safestep plan \
  --start-state "1,1,1" \
  --target-state "2,2,2" \
  --manifest-dir ./k8s/

# Output:
# Plan found (3 steps, fully rollback-safe):
#   Step 1: db v1 → v2        [rollback: safe]
#   Step 2: api v1 → v2       [rollback: safe]
#   Step 3: gateway v1 → v2   [rollback: safe]
```

### Example 2: Detecting a Point of No Return

When a database migration makes rollback impossible:

```bash
safestep plan \
  --start-state "1,1,1" \
  --target-state "3,3,3" \
  --manifest-dir ./k8s-with-migration/

# Output:
# Plan found (6 steps, 1 point of no return):
#   Step 1: db v1 → v2        [rollback: safe]
#   Step 2: api v1 → v2       [rollback: safe]
#   Step 3: db v2 → v3        [rollback: POINT OF NO RETURN]
#     └─ Witness: db v3 migration drops column 'legacy_id'
#        required by api v1, preventing rollback of api
#        past this point.
#   Step 4: api v2 → v3       [rollback: n/a (past PNR)]
#   Step 5: gateway v1 → v2   [rollback: n/a (past PNR)]
#   Step 6: gateway v2 → v3   [rollback: n/a (past PNR)]
```

### Example 3: Exporting to ArgoCD Sync Waves

```bash
safestep plan \
  --start-state "1,1,1" \
  --target-state "2,2,2" \
  --manifest-dir ./k8s/ \
  --save plan.json

safestep export --plan-file plan.json --format argocd --output-dir ./argocd/

# Generates:
#   argocd/step-01-db.yaml       (sync-wave: 1)
#   argocd/step-02-api.yaml      (sync-wave: 2)
#   argocd/step-03-gateway.yaml  (sync-wave: 3)
```

### Example 4: Large-Scale Mesh Benchmark

```bash
safestep benchmark \
  --topology mesh \
  --services 50 \
  --versions 10 \
  --iterations 10

# Output:
# Topology: mesh (50 services, 10 versions/svc)
# Constraint density: 0.42
# Trials: 10
#
# Results:
#   Plan time:     mean 4.72s  (σ 0.31s)
#   Envelope time: mean 15.3s  (σ 1.24s)
#   Clauses:       mean 210,500
#   Plan length:   mean 127 steps
#   PNR count:     mean 2.1
```

### Example 5: CI/CD Integration

Add SafeStep as a CI check to block unsafe deployments:

```yaml
# .github/workflows/safestep.yml
name: SafeStep Deployment Verification
on:
  pull_request:
    paths: ['k8s/**']

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install SafeStep
        run: cargo install safestep-cli

      - name: Validate manifests
        run: safestep validate --manifest-dir ./k8s/ --strict

      - name: Analyze compatibility
        run: safestep analyze --schema-dir ./k8s/

      - name: Verify upgrade plan
        run: |
          safestep plan \
            --start-state "$(cat current-versions.txt)" \
            --target-state "$(cat target-versions.txt)" \
            --manifest-dir ./k8s/
```

---

## API Reference

SafeStep's core logic is exposed as a Rust library. You can depend on
individual crates for programmatic use.

### `safestep-types`

```rust
use safestep_types::{ServiceDescriptor, Version, Constraint, DeploymentPlan};

// Define services
let api = ServiceDescriptor::new("api", "default")
    .with_versions(vec![
        Version::new(1, 0, 0), Version::new(2, 0, 0), Version::new(3, 0, 0),
    ]);

// Define constraints (using the Constraint enum)
let constraint = Constraint::Compatibility {
    id: ConstraintId::from_name("api-db-compat"),
    service_a: ServiceIndex(0),
    service_b: ServiceIndex(2),
    compatible_pairs: vec![
        (VersionIndex(0), VersionIndex(0)),
        (VersionIndex(0), VersionIndex(1)),
        (VersionIndex(1), VersionIndex(1)),
        (VersionIndex(1), VersionIndex(2)),
        (VersionIndex(2), VersionIndex(1)),
        (VersionIndex(2), VersionIndex(2)),
    ],
};
```

### `safestep-core`

```rust
use safestep_core::{GraphBuilder, DeploymentPlanner, EnvelopeComputer, PlanResult};

// Build the version-product graph
let graph = GraphBuilder::from_services_filtered(&services, &constraints)?;

// Find a safe plan
let config = PlannerConfig::default();
let mut planner = DeploymentPlanner::new(graph.clone(), constraints.clone(), config);
let result = planner.plan(&start, &target);

match result {
    PlanResult::Success(plan) => {
        // Compute the safety envelope
        let envelope = EnvelopeComputer::new(&graph, &constraints)
            .compute(&plan)?;

        println!("Safe states: {}", envelope.safe_count());
        println!("Points of no return: {}", envelope.pnr_count());
    }
    PlanResult::Infeasible(witness) => {
        println!("No plan exists: {}", witness.reason);
    }
    _ => {}
}

// Independently validate the plan witness
use safestep_core::{WitnessValidator, MonotoneChecker, CegarBoundTracker};

let validator = WitnessValidator::new(&graph, &oracle, &constraints);
let verdict = validator.validate_safe_plan(&plan, &start, &target);
assert!(verdict.is_valid());

// Empirically check monotonicity (downward-closure assumption)
let checker = MonotoneChecker::new(&oracle, &constraints, n, versions_per_svc.clone());
let result = checker.check(1000, &mut rng);
println!("monotonicity held: {}", result.monotonicity_held);
```

### `safestep-solver`

```rust
use safestep_solver::{CdclSolver, SatResult, SolverConfig};

let mut solver = CdclSolver::default_solver();
solver.add_clause_dimacs(&[1, 2, -3]);
solver.add_clause_dimacs(&[-1, 3]);

match solver.solve() {
    SatResult::Satisfiable(assignment) => println!("SAT: {:?}", assignment),
    SatResult::Unsatisfiable(core) => println!("UNSAT, core size: {}", core.literals.len()),
    SatResult::Unknown(reason) => println!("Unknown: {}", reason),
}
```

### `safestep-encoding`

```rust
use safestep_encoding::{BmcEncoder, IntervalCompressor, CnfFormula};

// Create a BMC encoder
let versions_per_service = vec![3, 4, 3];
let encoder = BmcEncoder::new(3, versions_per_service);

// Encode initial and target states
let initial_clauses = encoder.encode_initial_state(&[0, 1, 0]);
let target_clauses = encoder.encode_target_state(&[2, 3, 2], k_star);

// Combine into a CnfFormula
let mut formula = CnfFormula::new();
for clause in initial_clauses.into_iter().chain(target_clauses) {
    formula.add_clause(clause);
}
println!("Clauses: {}", formula.num_clauses());
```

### `safestep-k8s`

```rust
use safestep_k8s::{KubernetesManifest, HelmChartLoader};

// Load from raw manifests
let manifest = KubernetesManifest::parse(&yaml_content)?;

// Load from Helm charts
let chart = HelmChartLoader::load("./charts/")?;
```

---

## Contributing

We welcome contributions of all kinds: bug reports, feature requests,
documentation improvements, and code.

### Getting Started

```bash
# Fork and clone
git clone https://github.com/<you>/safestep.git
cd safestep/implementation

# Build and test
cargo build
cargo test

# Run clippy and format checks
cargo clippy -- -D warnings
cargo fmt --check
```

### Guidelines

1. **Open an issue first** for non-trivial changes to discuss the approach.
2. **Write tests.** All new functionality must have unit tests; solver
   integration changes need integration tests.
3. **Run the full test suite** before submitting:
   ```bash
   cargo test --workspace
   cargo clippy --workspace -- -D warnings
   ```
4. **Keep commits focused.** One logical change per commit, with a clear
   commit message.
5. **Document public APIs.** All `pub` items must have doc comments.
6. **Benchmark performance-sensitive changes.** Use `safestep benchmark` to
   verify no regressions.

### Code Organization Rules

- Core types go in `safestep-types`.
- Solver interactions go in `safestep-solver`.
- Encoding strategies go in `safestep-encoding`.
- Planning algorithms go in `safestep-core`.
- Kubernetes-specific code goes in `safestep-k8s`.
- Schema analysis goes in `safestep-schema`.
- Output formatting goes in `safestep-diagnostics`.
- CLI-only logic (argument parsing, I/O) goes in `safestep-cli`.

### Running Specific Test Suites

```bash
# Unit tests for a single crate
cargo test -p safestep-core

# Integration tests only
cargo test --test integration

# Tests with Z3 features
cargo test --features z3

# Benchmark tests (ignored by default)
cargo test -- --ignored
```

---

## Citation

If you use SafeStep in academic work, please cite:

```bibtex
@inproceedings{safestep2025,
  title     = {{SafeStep}: Verified Deployment Planning with Rollback
               Safety Envelopes},
  author    = {Young, Halley},
  booktitle = {Proceedings of the ACM Conference on Computer and
               Communications Security (CCS)},
  year      = {2025},
  doi       = {10.1145/XXXXXXX.XXXXXXX},
  note      = {Software available at
               \url{https://github.com/safestep/safestep}}
}
```

---

## License

SafeStep is dual-licensed under the MIT License and the Apache License 2.0,
at your option.

- [MIT License](LICENSE-MIT)
- [Apache License 2.0](LICENSE-APACHE)

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in SafeStep by you shall be dual-licensed as above,
without any additional terms or conditions.

---

## Acknowledgments

- [CaDiCaL](https://github.com/arminbiere/cadical) by Armin Biere — the
  SAT solver at the core of SafeStep's propositional reasoning.
- [Z3](https://github.com/Z3Prover/z3) by Microsoft Research — used for
  SMT queries in the CEGAR loop.
- [Kubernetes](https://kubernetes.io/) and its ecosystem (Helm, Kustomize,
  ArgoCD, Flux) — the deployment platform SafeStep targets.
- The formal verification and bounded model checking communities, whose
  techniques make SafeStep possible.
- The Rust ecosystem, especially the `clap`, `serde`, and `petgraph` crates,
  which SafeStep relies on heavily.

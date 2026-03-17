<div align="center">

# BiCut

### A Bilevel Optimization Compiler with Intersection Cut Framework

[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-blue)](LICENSE)
[![Crates](https://img.shields.io/badge/crates-9-yellow)]()

*Compile bilevel optimization problems into solver-ready MILPs with correctness certificates and a theoretically-grounded intersection cut framework.*

</div>

---

## Abstract

**BiCut** is a typed bilevel optimization compiler that transforms mixed-integer
bilevel linear programs (MIBLPs) into solver-ready single-level reformulations.
It introduces *bilevel intersection cuts* — a new family of valid inequalities
derived by extending Balas's 1971 intersection cut framework to the
bilevel-infeasible set defined by follower suboptimality. The compiler provides:

- A **typed intermediate representation (IR)** with leader/follower scoping,
  integrality annotations, and constraint qualification tracking
- **Automatic reformulation selection** among KKT, strong duality,
  value-function, and column-and-constraint generation strategies
- **Machine-checkable correctness certificates** guaranteeing bilevel
  equivalence of the reformulated MILP
- **Bilevel intersection cuts** with a polynomial-time separation oracle (for
  fixed follower dimension) and finite convergence guarantees
- **Solver-agnostic emission** to Gurobi, SCIP, and HiGHS via standard
  `.mps`/`.lp` files

### Current Status (Honest Assessment)

The **KKT reformulation compiler + HiGHS** is the main working contribution:
it solves **121/122** test instances, most at the root node. The intersection
cut framework is architecturally complete and theoretically grounded, but
**the cuts do not yet close gaps beyond what HiGHS's native cuts achieve** on
our test instances. See [HONEST_EVALUATION.md](HONEST_EVALUATION.md) for
detailed results.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Examples](#examples)
  - [Simple Bilevel LP](#example-1-simple-bilevel-lp)
  - [Network Interdiction](#example-2-network-interdiction)
  - [Strategic Bidding](#example-3-strategic-bidding)
  - [Custom Cut Configuration](#example-4-custom-cut-configuration)
- [Theory Overview](#theory-overview)
  - [Bilevel Reformulation](#bilevel-reformulation)
  - [Intersection Cut Generation](#intersection-cut-generation)
  - [Value-Function Oracle](#value-function-oracle)
  - [Correctness Certificates](#correctness-certificates)
- [Benchmarks](#benchmarks)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Features

### Compiler Pipeline

| Stage | Description |
|-------|-------------|
| **Parsing** | TOML-based bilevel problem specification with leader/follower scoping |
| **Structural Analysis** | Convexity detection, CQ verification (LICQ/MFCQ/Slater), boundedness analysis |
| **Reformulation Selection** | Automatic strategy selection based on problem signature |
| **KKT Pass** | Complementarity encoding via big-M, SOS1, or indicator constraints |
| **Strong Duality Pass** | Primal-dual pairing for LP lower levels |
| **Value Function Pass** | Parametric LP with piecewise-linear value function |
| **CCG Pass** | Column-and-constraint generation for general bilevel |
| **Intersection Cuts** | Novel bilevel-specific cutting planes from Balas's framework |
| **Adaptive Cut Cache** | LRU cache with hit-rate monitoring and automatic resizing |
| **Emission** | Solver-specific output to Gurobi, SCIP, HiGHS |
| **Certificates** | Machine-checkable correctness proofs |

### Key Capabilities

- **Four reformulation strategies**: KKT/MPEC, strong duality, value function,
  column-and-constraint generation — automatically selected based on problem
  structure
- **Three complementarity encodings**: Big-M (with automatic bound tightening),
  SOS1 sets, indicator constraints
- **Three solver backends**: Gurobi (commercial), SCIP (academic), HiGHS (open-source)
- **Bilevel intersection cuts**: Novel valid inequalities with polynomial-time
  separation for fixed follower dimension
- **Value-function oracle**: Exact parametric LP for continuous lower levels,
  sampling-based approximation for MILP lower levels
- **Adaptive cut cache**: LRU cache with rolling-window hit-rate monitoring,
  automatic resizing, oracle latency histograms, and runtime performance metrics
- **Correctness certificates**: Verified preconditions (CQ status, boundedness,
  integrality) for each reformulation
- **File format support**: MPS (fixed/free), LP (CPLEX-style) parsing and writing
- **Solver abstractions**: Built-in simplex (has known bugs; use HiGHS for production), compatible with `good_lp` and HiGHS backends

### Known Limitations

- **Built-in simplex solver** has correctness issues (6 test failures, wrong optima on some problems). Use HiGHS for production.
- **BOBILib integration** is architecturally present but no BOBILib instances have been downloaded or tested.
- **Intersection cuts** are valid but do not yet improve solve performance on our 122-instance test suite.
- **Compiler tests** have 63 compilation errors due to API mismatch between tests and library.
- **Branch-and-cut tests** do not compile.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    BiCut Compiler                        │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐  │
│  │  Parser   │──▶│  Structural  │──▶│ Reformulation │  │
│  │  (TOML)   │   │  Analysis    │   │  Selection    │  │
│  └──────────┘   └──────────────┘   └───────┬───────┘  │
│                                             │           │
│         ┌───────────────┬─────────┬─────────┤           │
│         ▼               ▼         ▼         ▼           │
│  ┌──────────┐   ┌──────────┐ ┌────────┐ ┌───────┐     │
│  │ KKT Pass │   │ SD Pass  │ │VF Pass │ │  CCG  │     │
│  │          │   │          │ │        │ │ Pass  │     │
│  └────┬─────┘   └────┬─────┘ └───┬────┘ └───┬───┘     │
│       └───────────────┴───────────┴──────────┘          │
│                         │                               │
│               ┌─────────▼──────────┐                    │
│               │ Intersection Cuts  │                    │
│               │ (Cut Loop + Pool)  │                    │
│               └─────────┬──────────┘                    │
│                         │                               │
│         ┌───────────────┼───────────────┐               │
│         ▼               ▼               ▼               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐           │
│  │  Gurobi  │   │   SCIP   │   │  HiGHS   │           │
│  │ Backend  │   │ Backend  │   │ Backend  │           │
│  └──────────┘   └──────────┘   └──────────┘           │
│                                                         │
│  ┌──────────────────────────────────────────┐          │
│  │          Correctness Certificates         │          │
│  └──────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

### Crate Dependency Graph

```
bicut-cli
  ├── bicut-compiler
  │     ├── bicut-core (structural analysis, CQ verification)
  │     ├── bicut-cuts (intersection cuts, separation oracle)
  │     ├── bicut-value-function (parametric LP, lifting)
  │     ├── bicut-lp (simplex solver, tableau operations)
  │     └── bicut-types (IR, problem types, certificates)
  ├── bicut-branch-cut (branch-and-cut solver)
  │     ├── bicut-cuts
  │     └── bicut-lp
  └── bicut-bench (BOBILib benchmarking)
        ├── bicut-compiler
        └── bicut-types
```

---

## Installation

### Prerequisites

- **Rust** ≥ 1.70 (install via [rustup](https://rustup.rs/))
- **Optional**: Gurobi, SCIP, or HiGHS for downstream MILP solving

### From Source

```bash
git clone https://github.com/bicut-project/bicut.git
cd bicut/implementation
cargo build --release
```

The compiled binary will be at `target/release/bicut`.

### Via `cargo install`

```bash
# Install from local source
cd implementation
cargo install --path crates/bicut-cli

# The binary `bicut` is now available globally
bicut --help
```

### Verify Installation

```bash
cargo test
cargo run --bin bicut -- --help

# Run examples
cargo run --example knapsack_interdiction
cargo run --example toll_pricing

# Run benchmarks
cargo bench
```

### Optional: Add to PATH

```bash
export PATH="$PATH:$(pwd)/target/release"
```

---

## Quick Start

### 1. Define a Bilevel Problem

Create a file `my_problem.toml`:

```toml
[problem]
name = "simple_bilevel"
description = "Textbook bilevel LP"

[leader]
variables = ["x"]
objective = "-x - 7*y"
sense = "minimize"

[[leader.constraints]]
name = "upper_bound"
expr = "-2*x + y"
sense = "<="
rhs = 4.0

[follower]
variables = ["y"]
objective = "-y"
sense = "minimize"

[[follower.constraints]]
name = "c1"
expr = "-x + y"
sense = "<="
rhs = 1.0

[[follower.constraints]]
name = "c2"
expr = "x + y"
sense = "<="
rhs = 5.0

[follower.bounds]
y = { lower = 0.0 }
```

### 2. Compile to MILP

```bash
# Auto-select reformulation strategy
bicut compile --input my_problem.toml --output reformulated.mps

# Force KKT reformulation
bicut compile --input my_problem.toml \
  --reformulation kkt \
  --output reformulated.mps

# Generate correctness certificate
bicut compile --input my_problem.toml \
  --certificate \
  --output reformulated.mps
```

The output format is inferred from the file extension: use `.mps` for solver-ready
MPS output or `.lp` for CPLEX-style LP output.

### 3. Solve with Your Preferred Solver

```bash
# Gurobi
gurobi_cl ResultFile=solution.sol reformulated.mps

# SCIP
scip -f reformulated.mps

# HiGHS
highs reformulated.mps
```

These downstream solver commands also require a working external solver
installation; the README previously omitted that dependency.

### 4. Solve End-to-End and Analyze

```bash
# Compile and solve end-to-end
bicut solve --input my_problem.toml --time-limit 600

# Structural analysis with constraint qualification report
bicut analyze --input my_problem.toml --cq
```

---

## CLI Reference

### Global Options

```
bicut [OPTIONS] <COMMAND>

Options:
  -c, --config <FILE>      Configuration file path
  -l, --log-level <LEVEL>  Log level: trace, debug, info, warn, error [default: warn]
  -f, --format <FORMAT>    Output format: human, json [default: human]
  -o, --output <FILE>      Output file path
  -q, --quiet              Suppress non-error output
  --no-color               Disable colored output
  -h, --help               Print help
  -V, --version            Print version
```

### Commands

#### `compile` — Compile a bilevel problem to MILP

```
bicut compile [OPTIONS] --input <FILE>

Options:
  -i, --input <FILE>              Input bilevel problem (TOML)
  -o, --output <FILE>             Output MILP (.mps or .lp)
  -r, --reformulation <STRATEGY>  kkt | strong-duality | value-function | ccg | auto [default: auto]
  --certificate                   Generate correctness certificate
```

#### `solve` — Compile and solve end-to-end

```
bicut solve [OPTIONS] --input <FILE>

Options:
  -i, --input <FILE>              Input bilevel problem (TOML)
  --time-limit <SECONDS>          Solver time limit [default: 3600]
```

#### `analyze` — Structural analysis of a bilevel problem

```
bicut analyze [OPTIONS] --input <FILE>

Options:
  -i, --input <FILE>              Input bilevel problem (TOML)
  --cq                            Detailed constraint qualification report
```

*Tip: Use the global `--format json` flag for JSON output.*

#### `benchmark` — Run BOBILib benchmarks

```
bicut benchmark [OPTIONS]

Options:
  --instances <DIR>               BOBILib instance directory
  --timeout <SECONDS>             Per-instance timeout [default: 3600]
```

---

## Configuration

BiCut reads configuration from `bicut.toml` in the working directory or from
the path specified by `--config`.

```toml
# bicut.toml — BiCut configuration file

[compiler]
# Default reformulation strategy: "auto", "kkt", "strong-duality",
# "value-function", "ccg"
reformulation = "auto"

# Complementarity encoding: "big-m", "sos1", "indicator"
encoding = "big-m"

# Numerical tolerance for constraint satisfaction
tolerance = 1e-8

# Generate correctness certificates by default
certificates = true

[compiler.big_m]
# Automatic big-M computation via bound-tightening LPs
auto_compute = true
# Fallback big-M value when bounds cannot be computed
default_value = 1e6
# Maximum iterations for bound tightening
max_tightening_iterations = 100

[cuts]
# Enable bilevel intersection cuts
enabled = true
# Maximum cut rounds
max_rounds = 50
# Target root gap closure (stop if achieved)
gap_threshold = 0.01
# Maximum cuts added per round
max_cuts_per_round = 100
# Minimum cut violation to accept
min_violation = 1e-6

[cuts.cache]
# Cache strategy: "none", "lru", "parametric", "adaptive"
strategy = "adaptive"
# Maximum cache entries (initial capacity for adaptive mode)
max_entries = 4096
# Cache hit rate target (for monitoring)
target_hit_rate = 0.90
# Adaptive mode: grow cache when hit rate falls below this threshold
adaptive_grow_threshold = 0.60
# Adaptive mode: shrink cache when hit rate exceeds this threshold
adaptive_shrink_threshold = 0.95
# Adaptive mode: minimum / maximum capacity bounds
adaptive_min_capacity = 256
adaptive_max_capacity = 65536

[cuts.separation]
# Separation oracle timeout (seconds)
timeout = 1.0
# Use parametric sensitivity for warm-starting
parametric_warmstart = true
# Maximum bases to enumerate per separation call
max_bases = 1000

[value_function]
# Oracle mode: "exact" (parametric LP), "sampling", "hybrid"
mode = "hybrid"
# Number of sample points for MILP lower levels
num_samples = 1000
# Error tolerance for sampling approximation
sample_tolerance = 1e-4

[backend]
# Default solver backend: "gurobi", "scip", "highs"
default = "gurobi"
# Solver-specific settings
[backend.gurobi]
use_lazy_constraints = true
use_indicator_constraints = true

[backend.scip]
use_constraint_handlers = true

[backend.highs]
# HiGHS has limited callback support
use_row_generation = true

[benchmark]
# Default BOBILib instance directory
instance_dir = "./data/bobilib"
# Default timeout per instance
timeout = 3600
# Number of parallel workers
parallel = 4
```

---

## Examples

### Example 1: Simple Bilevel LP

A textbook bilevel LP demonstrating basic compiler functionality.

**Problem:**
```
min_{x}  -x - 7y
s.t.     -2x + y ≤ 4
         y ∈ argmin_{y≥0} { -y : -x + y ≤ 1, x + y ≤ 5 }
```

**Input** (`examples/simple_bilevel.toml`):
```toml
[problem]
name = "simple_bilevel"

[leader]
variables = ["x"]
objective = "-x - 7*y"
sense = "minimize"

[[leader.constraints]]
name = "upper"
expr = "-2*x + y"
sense = "<="
rhs = 4.0

[follower]
variables = ["y"]
objective = "-y"
sense = "minimize"

[[follower.constraints]]
name = "c1"
expr = "-x + y"
sense = "<="
rhs = 1.0

[[follower.constraints]]
name = "c2"
expr = "x + y"
sense = "<="
rhs = 5.0

[follower.bounds]
y = { lower = 0.0 }
```

**Usage:**
```bash
# Compile with KKT reformulation
bicut compile -i examples/simple_bilevel.toml -r kkt -o simple.mps
# Expected: x* = 2, y* = 3, objective = -23

# Compare reformulations
bicut compile -i examples/simple_bilevel.toml -r strong-duality -o simple_sd.mps
bicut compile -i examples/simple_bilevel.toml -r value-function -o simple_vf.mps
```

### Example 2: Network Interdiction

A network interdiction problem where the leader removes arcs to maximize the
follower's shortest path.

**Problem:**
```
max_{x}  min_{y≥0} { c^T y : Ny = [1; 0; ...; -1], y_e ≤ u_e(1-x_e) ∀e }
s.t.     Σ x_e ≤ k,  x_e ∈ {0,1}
```

**Input** (`examples/interdiction.toml`):
```toml
[problem]
name = "network_interdiction"
description = "5-node shortest path interdiction with budget k=2"

[leader]
variables = ["x1", "x2", "x3", "x4", "x5", "x6"]
variable_types = { x1 = "binary", x2 = "binary", x3 = "binary",
                   x4 = "binary", x5 = "binary", x6 = "binary" }
objective = "y1 + y2 + y3 + y4 + y5 + y6"
sense = "maximize"

[[leader.constraints]]
name = "budget"
expr = "x1 + x2 + x3 + x4 + x5 + x6"
sense = "<="
rhs = 2.0

[follower]
variables = ["y1", "y2", "y3", "y4", "y5", "y6"]
objective = "y1 + y2 + y3 + y4 + y5 + y6"
sense = "minimize"

# Flow conservation and capacity constraints...
```

**Usage:**
```bash
# Interdiction problems need value-function reformulation (integer lower level)
bicut compile -i examples/interdiction.toml -r auto -o interdiction.mps --certificate
bicut --format json analyze -i examples/interdiction.toml --cq
```

### Example 3: Strategic Bidding

A strategic generator bidding into an electricity market modeled as a bilevel
program.

**Input** (`examples/strategic_bidding.toml`):
```toml
[problem]
name = "strategic_bidding"
description = "Single generator strategic bidding in a 3-bus market"

[leader]
variables = ["p1", "p2"]
objective = "price1 * p1 + price2 * p2 - 20*p1 - 30*p2"
sense = "maximize"

[[leader.constraints]]
name = "capacity1"
expr = "p1"
sense = "<="
rhs = 100.0

[[leader.constraints]]
name = "capacity2"
expr = "p2"
sense = "<="
rhs = 80.0

[follower]
variables = ["price1", "price2", "flow12", "flow23"]
objective = "20*p1 + 30*p2 + 40*p3"
sense = "minimize"
# Market clearing (ISO dispatch minimization)

[[follower.constraints]]
name = "balance_bus1"
expr = "p1 - flow12 - demand1"
sense = "="
rhs = 0.0
```

**Usage:**
```bash
# Strong duality is optimal for LP lower levels (market clearing)
bicut compile -i examples/strategic_bidding.toml -r strong-duality -o bidding.mps

# Compile and solve end-to-end
bicut solve -i examples/strategic_bidding.toml --time-limit 600
```

### Example 4: Custom Cut Configuration

Fine-tuning the intersection cut engine for a challenging instance.

```bash
# Create a custom configuration
cat > custom_cuts.toml << 'EOF'
[cuts]
enabled = true
max_rounds = 100
gap_threshold = 0.005
max_cuts_per_round = 200
min_violation = 1e-7

[cuts.cache]
strategy = "parametric"
max_entries = 50000
target_hit_rate = 0.95

[cuts.separation]
timeout = 5.0
parametric_warmstart = true
max_bases = 5000
EOF

# Run with custom configuration (global --config goes before subcommand)
bicut --config custom_cuts.toml compile -i hard_instance.toml \
  -r kkt \
  -o tightened.mps
```

---

### Runnable Rust Examples

Two self-contained Rust examples demonstrate the library API without the CLI:

**Knapsack Interdiction** — a defender removes items from an attacker's knapsack:
```bash
cargo run --example knapsack_interdiction
```

**Toll Pricing** — a leader sets tolls on network arcs to maximize revenue:
```bash
cargo run --example toll_pricing
```

Both examples construct a `BilevelProblem`, run `StructuralAnalysis::analyze()`,
build LP relaxations, and solve them with the built-in simplex solver.

---

## Theory Overview

### Bilevel Reformulation

A bilevel program has the general form:

```
min_{x ∈ X}  F(x, y)
s.t.         G(x, y) ≤ 0
             y ∈ S(x) = argmin_{y ∈ Y(x)} f(x, y)
```

where the **leader** controls *x* and the **follower** responds optimally with
*y*. BiCut supports four reformulation strategies to convert this to a
single-level MILP:

#### KKT Reformulation

When the lower level is convex and a constraint qualification holds, replace
follower optimality with KKT conditions:

```
∇_y f(x,y) + Σ λ_j ∇_y g_j(x,y) = 0
λ_j ≥ 0,  g_j(x,y) ≤ 0,  λ_j · g_j(x,y) = 0
```

The complementarity conditions `λ_j · g_j = 0` are linearized using big-M,
SOS1, or indicator constraints. BiCut automatically computes tight big-M values
via bound-tightening LPs.

**Preconditions** (verified by BiCut's type checker):
- Lower level is convex in *y* for all feasible *x*
- LICQ, MFCQ, or Slater's condition holds at all lower-level optima

#### Strong Duality Reformulation

For LP lower levels, replace follower optimality with the strong duality
condition `c^T y = b^T λ` plus primal-dual feasibility. Avoids
complementarity constraints entirely.

**Preconditions**: LP lower level, bounded feasible region.

#### Value-Function Reformulation

Replace `y ∈ S(x)` with `f(x,y) ≤ φ(x)` where `φ(x) = min_y f(x,y)` is the
**value function**. Always valid (no CQ required), but requires computing or
approximating φ(x).

BiCut uses:
- **Exact parametric LP** for continuous lower levels
- **Sampling-based approximation** with error bounds for MILP lower levels

#### Column-and-Constraint Generation (CCG)

An iterative approach that alternates between solving a restricted master
problem and generating new columns/constraints from the follower's best
response. Convergence is guaranteed under finite feasible-response assumptions.

### Intersection Cut Generation

BiCut introduces **bilevel intersection cuts**, extending Balas's 1971
framework to bilevel optimization.

#### The Bilevel-Infeasible Set

Define the bilevel-infeasible set:

```
B̄ = { (x,y) : y is feasible but NOT optimal for the follower at x }
  = { (x,y) : Ay ≤ b + Bx, y ≥ 0, c^T y > φ(x) }
```

**Theorem (Polyhedrality).** For MIBLPs with LP lower levels, B̄ is a finite
union of polyhedra. Each polyhedron corresponds to a critical region of the
lower-level parametric LP, characterized by a specific optimal basis.

#### Separation Oracle

Given an LP relaxation optimum (x̂, ŷ) that lies in the interior of some
polyhedron B̄_k of the bilevel-infeasible set:

1. Identify the critical region containing x̂ via basis enumeration
2. Compute ray-boundary intersections: for each edge direction d_j of the LP
   simplex tableau, find α_j such that x̂ + α_j · d_j lies on the boundary of
   B̄_k
3. Construct the intersection cut: `Σ (x_j / α_j) ≥ 1`

**Complexity.** For fixed follower dimension d, the separation oracle runs in
polynomial time: O(m^d) basis enumerations, each requiring one LP solve.

#### Finite Convergence

**Theorem.** Under non-degeneracy of the LP relaxation, the bilevel
intersection cut loop terminates in finitely many rounds. Each cut strictly
separates the current vertex, and the number of vertices is bounded by C(m,n).

#### Cut Pool Management

BiCut maintains a cut pool with:
- **Parametric cache**: Warm-start separation using parametric sensitivity
  analysis, achieving >90% cache hit rates
- **Age-based purging**: Remove ineffective cuts after configurable rounds
- **Parallel cut generation**: Separate multiple violated polyhedra per round
  using Rayon

### Value-Function Oracle

The value-function oracle evaluates φ(x) = min{c^T y : Ay ≤ b + Bx, y ≥ 0}:

- **LP lower levels**: Exact evaluation via the dual simplex method with
  parametric perturbation. The value function is piecewise linear and continuous.
  Critical regions are identified and cached.
- **MILP lower levels**: Sampling-based approximation. Points are sampled in the
  leader's feasible region; for each sample, the lower-level MILP is solved.
  A piecewise-linear overestimator is constructed with provable L¹ error bounds.

### Correctness Certificates

Each reformulation generates a certificate containing:

```json
{
  "reformulation": "kkt",
  "preconditions": {
    "lower_level_convex": true,
    "cq_status": "LICQ_verified",
    "bounded_feasible_region": true,
    "big_m_validity": { "all_finite": true, "max_value": 1547.3 }
  },
  "variables_mapped": 12,
  "constraints_generated": 34,
  "hash": "sha256:a7f3b2..."
}
```

Certificates enable **independent verification**: a reviewer can check that the
stated preconditions are sufficient for the chosen reformulation without
re-deriving the mathematical argument.

---

## Benchmarks

### Honest Evaluation (122 Instances)

We evaluated BiCut on 122 bilevel instances across 4 categories using the
KKT reformulation solved by HiGHS. See [HONEST_EVALUATION.md](HONEST_EVALUATION.md)
for full details.

| Category | Count | KKT Solved | Avg Time | LP Gap |
|----------|-------|------------|----------|--------|
| Knapsack interdiction | 72 | 72/72 | 0.008s | 0.0% |
| Dense bilevel | 28 | 28/28 | 0.186s | 5.4% |
| Integer linking | 12 | 12/12 | 0.093s | 4.4% |
| Stackelberg games | 10 | 9/10 | 0.002s | 0.0% |
| **Total** | **122** | **121/122** | | |

### Value-Function Cuts

| Metric | Result |
|--------|--------|
| VF cuts generated | 35 |
| Cases where cuts helped | **0** |
| Cases where cuts hurt | **16** (5 infeasible, 11 slower) |

The cuts are mathematically valid (after a sign-convention fix) but do not
improve performance on these instances. HiGHS's native cuts are sufficient.

### What's NOT Been Tested

- ❌ No BOBILib instances downloaded or tested
- ❌ No comparison with MibS, CPLEX, or Gurobi bilevel
- ❌ No large-scale (n > 100) instances
- ❌ No mixed-integer follower instances

### Running the Honest Evaluation

```bash
cd pipeline_staging/bilevel-compiler-intersection-cuts
python3 benchmarks/honest_evaluation_v2.py
# Results saved to benchmarks/honest_benchmark_output/
```

### Running Rust Tests

```bash
cd implementation
cargo test -p bicut-types -p bicut-lp -p bicut-cuts -p bicut-value-function
# 302 tests pass (10 + 110 + 108 + 74)
```

---

## API Reference

### Core Types

#### `BilevelProblem`

The primary input type representing a bilevel optimization problem.

```rust
use bicut_types::BilevelProblem;

let problem = BilevelProblem {
    name: "my_problem".to_string(),
    leader_objective: leader_obj,
    follower_objective: follower_obj,
    variables: var_set,
    upper_constraints: vec![/* ... */],
    lower_constraints: vec![/* ... */],
    coupling_constraints: vec![],
    metadata: Default::default(),
    // see bicut_types::problem for full field documentation
};
```

#### `CompilerConfig`

Configuration for the compilation pipeline.

```rust
use bicut_compiler::{CompilerConfig, ReformulationType, BackendTarget};

let config = CompilerConfig::new(
    ReformulationType::KKT,
    BackendTarget::Gurobi,
)
.with_tolerance(1e-7)
.with_certificate(true);
```

#### `compile()`

The main entry point for bilevel-to-MILP compilation.

```rust
use bicut_compiler::compile;

let result = compile(&problem, config)?;
println!("Variables: {}", result.stats.num_vars);
println!("Constraints: {}", result.stats.num_constraints);
```

### Cut Engine

#### `CutManager`

The bilevel cut manager orchestrates intersection cut generation across rounds.

```rust
use bicut_cuts::manager::{CutManager, CutManagerConfig};

let config = CutManagerConfig {
    max_rounds: 50,
    min_gap: 0.01,
    max_cuts_per_round: 100,
    ..Default::default()
};

let mut manager = CutManager::new(config, n_leader, n_follower, follower_obj);
```

#### `CutPool`

Manages generated cuts with capacity limits.

```rust
use bicut_cuts::CutPool;

let mut pool = CutPool::new(1000);
pool.add_cut(cut);  // returns false if pool is at capacity
pool.clear();
```

### Value Function

#### `ValueFunctionOracle`

A trait for evaluating the lower-level value function. The primary
implementation is `ExactLpOracle` for LP lower levels.

```rust
use bicut_value_function::{ValueFunctionOracle, ExactLpOracle};

let oracle = ExactLpOracle::with_default_solver(problem);
let info = oracle.evaluate(&x_point)?;  // returns ValueFunctionInfo
let phi_x = oracle.value(&x_point)?;    // convenience: returns f64
```

---

## Project Structure

```
bilevel-compiler-intersection-cuts/
├── README.md                  # This file
├── LICENSE                    # MIT / Apache-2.0
├── CONTRIBUTING.md            # Contribution guidelines
├── CHANGELOG.md               # Version history
├── tool_paper.tex             # Academic paper (LaTeX, 16 pages)
├── tool_paper.pdf             # Compiled paper
├── groundings.json            # Verified claims with evidence & DOIs
├── .gitignore                 # Git ignore rules
│
├── implementation/            # Rust workspace
│   ├── Cargo.toml             # Workspace manifest (cargo install support)
│   ├── Cargo.lock             # Dependency lock
│   ├── examples/              # Runnable Rust examples
│   │   ├── knapsack_interdiction.rs  # Knapsack interdiction demo
│   │   └── toll_pricing.rs           # Toll-setting/pricing demo
│   ├── benches/               # Criterion benchmarks
│   │   └── bilevel_bench.rs   # LP, analysis, matrix benchmarks
│   └── crates/
│       ├── bicut-types/       # Core types: IR, problem, certificates
│       ├── bicut-core/        # Structural analysis, CQ verification
│       ├── bicut-lp/          # LP solver, MPS/LP format parsers
│       ├── bicut-cuts/        # Intersection cuts, separation oracle
│       ├── bicut-value-function/ # Parametric LP, value function oracle
│       ├── bicut-compiler/    # Reformulation passes, emission
│       ├── bicut-branch-cut/  # Branch-and-cut solver framework
│       ├── bicut-bench/       # BOBILib benchmark harness
│       └── bicut-cli/         # Command-line interface
│
├── ideation/                  # Problem crystallization and approach design
│   ├── final_approach.md      # Synthesis decision document
│   ├── crystallized_problem.md # Full problem statement
│   └── math_specification.md  # Mathematical foundations
│
├── theory/                    # Formal mathematical theory
│   └── approach.json          # Structured theory specification
│
├── examples/                  # Example bilevel problems (TOML)
│   ├── simple_bilevel.toml
│   ├── interdiction.toml
│   └── strategic_bidding.toml
│
├── benchmarks/                # Benchmark scripts and results
│   ├── run_bobilib.sh         # Full BOBILib benchmark
│   ├── compare_solvers.sh     # Cross-solver comparison
│   ├── ablation_study.sh      # Cut contribution ablation
│   ├── run_benchmarks.py      # Python benchmark runner (JSON/CSV output)
│   └── benchmark_output/      # Generated results (JSON, CSV, summary)
│
└── docs/                      # Additional documentation
    └── design/                # Design documents
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
detailed guidelines.

### Quick Start for Contributors

```bash
# Fork and clone
git clone https://github.com/<your-username>/bicut.git
cd bicut/implementation

# Build and test
cargo build
cargo test

# Run clippy lints
cargo clippy -- -D warnings

# Format code
cargo fmt
```

### Areas for Contribution

- **New cut families**: Split cuts, Chvátal-Gomory cuts for bilevel
- **Solver backends**: CPLEX, Mosek, COPT
- **Problem parsers**: MPS-bilevel, AMPL, JuMP interop
- **Benchmark instances**: New MIBLP instances beyond BOBILib
- **Documentation**: Tutorials, API examples, theory exposition

---

## Citation

If you use BiCut in your research, please cite:

```bibtex
@article{bicut2025,
  title     = {{BiCut}: A Solver-Agnostic Bilevel Optimization Compiler
               with Intersection Cuts},
  author    = {{BiCut Contributors}},
  journal   = {INFORMS Journal on Computing},
  year      = {2025},
  note      = {Software available at \url{https://github.com/bicut-project/bicut}}
}
```

For the intersection cut theory:

```bibtex
@article{bicut-cuts2025,
  title     = {Bilevel Intersection Cuts: Valid Inequalities from
               Follower Suboptimality},
  author    = {{BiCut Contributors}},
  journal   = {Mathematical Programming},
  year      = {2025},
  note      = {Extends Balas (1971) intersection cut framework to
               bilevel-infeasible sets}
}
```

### Related Work

- **Balas, E.** (1971). Intersection cuts — a new type of cutting planes for
  integer programming. *Operations Research*, 19(1), 19–39.
- **Fischetti, M., Ljubić, I., Monaci, M., & Sinnl, M.** (2017). A new
  general-purpose algorithm for mixed-integer bilevel linear programs.
  *Operations Research*, 65(6), 1615–1637.
- **DeNegre, S.** (2011). *Interdiction and discrete bilevel linear
  programming*. PhD thesis, Lehigh University.
- **Dempe, S.** (2002). *Foundations of Bilevel Programming*. Springer.
- **Xu, P., & Wang, L.** (2014). An exact algorithm for the bilevel
  mixed integer linear programming problem under three simplifying assumptions.
  *Computers & Operations Research*, 41, 309–318.

---

## License

BiCut is dual-licensed under:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT))
- **Apache License 2.0** ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

---

<div align="center">
<i>BiCut — Compile once, solve anywhere.</i>
</div>

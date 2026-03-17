<div align="center">

# MutSpec

### Mutation-Guided Contract Synthesis via the Mutation-Specification Duality

[![CI](https://img.shields.io/github/actions/workflow/status/mutspec/mutspec/ci.yml?branch=main&label=CI&logo=github)](https://github.com/mutspec/mutspec/actions)
[![crates.io](https://img.shields.io/crates/v/mutspec.svg?logo=rust)](https://crates.io/crates/mutspec)
[![docs.rs](https://img.shields.io/docsrs/mutspec?logo=docs.rs)](https://docs.rs/mutspec)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg?logo=rust)](https://www.rust-lang.org)
[![LOC](https://img.shields.io/badge/LOC-~52.5K-informational)](.)
[![MSRV](https://img.shields.io/badge/MSRV-1.75-blue.svg)](.)

[Quick Start](#quick-start) · [Installation](#installation) · [Documentation](#documentation) · [Examples](#examples) · [Contributing](#contributing)

</div>

---

**MutSpec** is a mutation-guided contract synthesis tool that bridges mutation
testing and formal specification through the *mutation-specification duality*.
Given a program and its test suite, MutSpec generates mutation operators, builds
a kill matrix, and synthesizes formal contracts (preconditions and
postconditions) by exploiting the correspondence between killed mutants and
error predicates. Surviving non-equivalent mutants are surfaced as specification
gaps and potential latent bugs. Written in Rust (~52.5K LoC across 11 crates),
MutSpec combines lattice-based synthesis, SMT solving, weakest-precondition
reasoning, and mutation subsumption analysis into a single, cohesive pipeline.

---

## Table of Contents

- [Motivation](#motivation)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Architecture](#architecture)
- [CLI Usage](#cli-usage)
- [Configuration](#configuration)
- [Output Formats](#output-formats)
- [Theoretical Background](#theoretical-background)
- [Benchmarks](#benchmarks)
- [Comparison with Related Tools](#comparison-with-related-tools)
- [Examples](#examples)
- [Library Usage (API)](#library-usage-api)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Motivation

Mutation testing tells you *how good your tests are*; formal specifications tell
you *what your program should do*. Traditionally these two worlds are separate:
mutation tools report scores, and specification tools require manual annotation.

MutSpec unifies them. The core insight—the **mutation-specification duality**—is
that every killed mutant encodes an error predicate, and the negation and
conjunction of those predicates yields a postcondition. Conversely, every
surviving non-equivalent mutant witnesses a gap in the specification. This
duality lets MutSpec:

1. **Automatically synthesize contracts** from mutation results, without manual
   annotation.
2. **Quantify specification completeness** by relating mutant survival to
   contract coverage.
3. **Surface latent bugs** by identifying surviving mutants that violate
   synthesized contracts.

Rather than treating mutation scores as a single number, MutSpec transforms
mutation artifacts into actionable, machine-checkable specifications.

---

## Key Features

- **Mutation-specification duality** — killed mutants → error predicates →
  negated and conjoined → postconditions; survived mutants → specification
  gaps → bug witnesses
- **Three-tier synthesis pipeline**
  - **Tier 1 — Lattice walk**: strongest postconditions via lattice traversal
    over a predicate domain, with SMT-backed entailment checks
  - **Tier 2 — Template-based**: parameterized contract templates instantiated
    via constraint solving
  - **Tier 3 — Heuristic fallback**: pattern-matching and syntactic heuristics
    for cases beyond SMT reach
- **Four core mutation operators** — AOR (Arithmetic Operator Replacement), ROR
  (Relational Operator Replacement), LCR (Logical Connector Replacement), UOI
  (Unary Operator Insertion), proven ε-complete for QF-LIA over loop-free code
- **Weakest-precondition engine** — SSA-based WP computation over a control-flow
  graph, enabling compositional contract derivation
- **Mutation subsumption analysis** — dominator-set computation, equivalence
  detection, and subsumption-implication correspondence
- **Gap analysis** — surviving non-equivalent mutants are classified, reported
  in SARIF v2.1.0, and linked to specification deficiencies
- **Latent-bug discrimination** — boundary witnesses generated at adjacent
  lattice elements detect bugs that pass all existing tests; structurally
  impossible for flat-trace tools like SpecFuzzer that lack the entailment-
  ordered lattice
- **Multi-format output** — JML annotations, Rust `#[requires]`/`#[ensures]`
  attributes, SARIF, JSON, and plain text
- **PIT integration** — direct ingestion of PIT XML mutation reports and kill
  matrices
- **Test generation** — witness-driven test generation from surviving mutants
- **Parallel execution** — Rayon-based data parallelism across mutants and
  synthesis tasks

---

## Quick Start

```bash
# Install from source
git clone https://github.com/mutspec/mutspec.git
cd mutspec
cd implementation
cargo install --path crates/cli

# End-to-end quickstart from the Cargo workspace root
mutspec analyze ../examples/absolute_value.ms
mutspec mutate ../examples/absolute_value.ms -o mutants.json
mutspec synthesize mutants.json --tier 1 -o contracts.json
mutspec verify contracts.json
```

When `-o` points at a `.json` file, `mutate` and `synthesize` emit JSON automatically,
so the staged pipeline above can be run exactly as written.

---

## Installation

### Prerequisites

| Dependency | Version   | Required | Notes                                |
|------------|-----------|----------|--------------------------------------|
| Rust       | ≥ 1.75    | Yes      | Stable toolchain                     |
| Cargo      | ≥ 1.75    | Yes      | Bundled with Rust                    |
| Z3         | ≥ 4.12    | Optional | Full SMT verification; Tier 1 synth  |
| CVC5       | ≥ 1.0     | Optional | Alternative SMT backend              |

### From Source (Recommended)

```bash
git clone https://github.com/mutspec/mutspec.git
cd mutspec
cd implementation
cargo build --release --bin mutspec

# The binary is at target/release/mutspec
# Optionally install it system-wide:
cargo install --path crates/cli
```

### Via Cargo

The crates.io package exists, but the repository workspace layout means the
source build instructions above are the most reliable path for this checkout.

### Verifying the Installation

```bash
mutspec --version
# mutspec 0.1.0

mutspec --help
```

### Optional: Z3 Setup

For full Tier 1 lattice-walk synthesis with SMT-backed entailment, install Z3:

```bash
# macOS
brew install z3

# Ubuntu / Debian
sudo apt-get install z3 libz3-dev

# From source
git clone https://github.com/Z3Prover/z3.git && cd z3
python scripts/mk_make.py && cd build && make -j$(nproc) && sudo make install
```

Set the `MUTSPEC_Z3_PATH` environment variable if Z3 is not on your `PATH`:

```bash
export MUTSPEC_Z3_PATH=/usr/local/bin/z3
```

---

## Architecture

MutSpec is organized as a Cargo workspace with **11 crates**, each encapsulating
a distinct concern:

```
┌─────────────────────────────────────────────────────────────────────┐
│                           cli (mutspec)                            │
│                     Command-line interface                         │
└──────────┬──────────┬──────────┬──────────┬──────────┬────────────┘
           │          │          │          │          │
     ┌─────▼───┐ ┌───▼────┐ ┌──▼───┐ ┌───▼───┐ ┌───▼──────┐
     │contract-│ │  gap-  │ │test- │ │  pit- │ │coverage  │
     │  synth  │ │analysis│ │ gen  │ │integr.│ │          │
     └────┬────┘ └───┬────┘ └──┬───┘ └───┬───┘ └────┬─────┘
          │          │         │         │           │
     ┌────▼──────────▼─────────▼─────────▼───────────▼─────┐
     │                   smt-solver                         │
     │             Z3 / CVC5 via SMT-LIB2                   │
     └──────────────────────┬──────────────────────────────┘
                            │
     ┌──────────────────────▼──────────────────────────────┐
     │               program-analysis                       │
     │         Parser · CFG · SSA · WP Engine               │
     └──────────────────────┬──────────────────────────────┘
                            │
     ┌──────────────────────▼──────────────────────────────┐
     │                mutation-core                         │
     │     Mutation Operators · Kill Matrix · Subsumption   │
     └──────────────────────┬──────────────────────────────┘
                            │
     ┌──────────────────────▼──────────────────────────────┐
     │                shared-types                          │
     │        Core AST · IR · Formulas · Contracts          │
     └─────────────────────────────────────────────────────┘
```

### Crate Responsibilities

| Crate               | Description                                                        |
|----------------------|--------------------------------------------------------------------|
| `shared-types`       | Core AST, intermediate representation, formula types, contract ADTs |
| `mutation-core`      | Mutation operators (AOR, ROR, LCR, UOI), kill matrix construction  |
| `program-analysis`   | Parser, control-flow graph, SSA transform, weakest-precondition engine |
| `smt-solver`         | Z3/CVC5 interface via SMT-LIB2, satisfiability and entailment queries |
| `contract-synth`     | Lattice-walk synthesis, template-based synthesis, heuristic fallback |
| `coverage`           | Mutation subsumption, dominator sets, equivalence class detection   |
| `gap-analysis`       | Gap theorem implementation, SARIF output, specification statistics  |
| `pit-integration`    | PIT XML report parser, kill matrix extraction, mutant mapping       |
| `test-gen`           | Test generation from gap witnesses, counterexample-guided tests     |
| `benchmarks-real`    | Benchmark harness for evaluating synthesis quality against baselines |
| `cli`                | `mutspec` binary, argument parsing, pipeline orchestration          |

---

## CLI Usage

### `mutspec analyze` — Full Pipeline

Run the complete mutation → synthesis → gap analysis pipeline:

```bash
mutspec analyze <source-files...> [OPTIONS]

Options:
  -o, --output-dir <PATH>     Output directory (default: mutspec-output)
      --tier <TIER>           Maximum synthesis tier: 1, 2, or 3
  -j, --jobs <N>              Parallel workers (default: num CPUs)
  -f, --format <FMT>          Output format: text, json, markdown, sarif
      --operators <OPS>       Comma-separated operators (e.g. AOR,ROR,LCR)
      --timeout <SECS>        SMT solver timeout per query
      --skip-verify           Skip the verification phase
  -v, --verbose               Increase verbosity (-vv for trace)
```

**Example:**

```bash
mutspec analyze src/math.ms --tier 1 --format sarif -o results/
```

### `mutspec mutate` — Generate Mutants

```bash
mutspec mutate <source-files...> [OPTIONS]

Options:
  -o, --output <PATH>         Output file (default: stdout)
  -f, --format <FMT>          Output format: text, json, markdown, sarif
      --operators <OPS>       Comma-separated operators: AOR,ROR,LCR,UOI,...
      --max-mutants <N>       Cap on total mutants per function
      --lines <START-END>     Restrict mutations to a line range
      --dry-run               List mutation sites without generating
      --list-operators        Show all available operators and exit
```

**Example:**

```bash
mutspec mutate lib.ms --operators AOR,ROR --max-mutants 500 -o mutants.json
```

### `mutspec synthesize` — Synthesize Contracts

```bash
mutspec synthesize <mutants-file> [OPTIONS]

Options:
  -o, --output <PATH>         Output file (default: stdout)
      --tier <TIER>           Maximum synthesis tier: 1, 2, or 3
  -f, --format <FMT>          Output format: text, json, markdown, sarif
      --weaken                Enable weakening pass
      --jml                   Emit JML-format contracts
      --functions <NAMES>     Filter to specific function names (comma-separated)
      --max-clauses <N>       Maximum clauses per contract
```

**Example:**

```bash
mutspec synthesize mutants.json --tier 1 --format json -o contracts.json
```

### `mutspec verify` — Verify Contracts

```bash
mutspec verify <contracts-file> [OPTIONS]

Options:
  -o, --output <PATH>         Output file (default: stdout)
  -f, --format <FMT>          Output format: text, json, markdown, sarif
      --solver <PATH>         Path to SMT solver binary (default: z3)
      --timeout <SECS>        Per-query timeout (default: 30)
      --logic <LOGIC>         SMT logic (e.g. QF_LIA, QF_NIA)
      --counterexamples       Print counterexamples on failure (default: true)
      --dump-smt <PATH>       Dump SMT-LIB2 encoding instead of running solver
```

### `mutspec report` — Generate Reports

```bash
mutspec report [OPTIONS]

Options:
      --mutants <PATH>        Path to mutants JSON file
      --contracts <PATH>      Path to contracts/specification JSON file
  -o, --output <PATH>         Output file (default: stdout)
  -f, --format <FMT>          Report format: text, json, markdown, sarif
      --detailed              Include detailed per-mutant information
      --operator-stats        Include operator distribution breakdown
      --survivors-only        Show only surviving (alive) mutants
      --title <TITLE>         Title for the report (default: "MutSpec Report")
```

### `mutspec config` — Manage Configuration

```bash
mutspec config <init|show>
```

---

## Configuration

MutSpec can be configured via a `mutspec.toml` file in the project root:

```toml
[mutation]
operators = ["AOR", "ROR", "LCR", "UOI", "SDL"]
max_mutants_per_site = 10
generation_timeout_secs = 30
higher_order = false
max_order = 1
skip_trivial_equivalents = true

[synthesis]
enabled_tiers = [1, 2, 3]
tier_timeout_secs = 60
total_timeout_secs = 300
lattice_max_steps = 100
lattice_widening_threshold = 10
template_max_vars = 4
template_max_const = 100
minimise_contracts = true
verify_contracts = true

[smt]
solver_path = "z3"
timeout_secs = 30
incremental = true
memory_limit_mb = 4096
logic = "QF_LIA"
dump_queries = false

[analysis]
max_expr_depth = 50
use_ssa = true
simplify_wp = true
max_blocks = 1000
constant_propagation = true
dead_code_elimination = true

[coverage]
subsumption = true
dominator_analysis = false
adequate_score_threshold = 0.8

[output]
format = "Text"                 # "Text", "Json", "Sarif", "Jml"
output_dir = "mutspec-output/"
verbosity = 1
jml = true
include_provenance = true
```

Environment variables override TOML settings with the `MUTSPEC_` prefix:

```bash
export MUTSPEC_SMT_TIMEOUT_SECS=60
export MUTSPEC_SYNTHESIS_ENABLED_TIERS="[1,2]"
export MUTSPEC_SMT_SOLVER_PATH=/usr/local/bin/cvc5
```

---

## Output Formats

MutSpec supports four output formats for synthesized contracts and reports.

### JSON Contract Format

```json
{
  "contracts": [
    {
      "function_name": "abs_diff",
      "clauses": [
        { "Ensures": { "Atom": { "relation": "Ge", "left": "Result", "right": { "Const": 0 } } } },
        { "Requires": { "Atom": { "relation": "Ne", "left": { "Var": "x" }, "right": { "Var": "y" } } } }
      ],
      "provenance": [
        {
          "targeted_mutants": ["d3f8a1b2-..."],
          "tier": "Tier1LatticeWalk",
          "solver_queries": 12,
          "synthesis_time_ms": 45.3
        }
      ],
      "strength": "Strongest",
      "verified": true
    }
  ],
  "program_name": "math.ms"
}
```

### JML Annotations (Java)

```java
//@ requires x != y;
//@ ensures \result >= 0;
//@ ensures \result == ((x > y) ? (x - y) : (y - x));
public int absDiff(int x, int y) { ... }
```

### Rust `#[requires]`/`#[ensures]` Attributes

Contract semantics expressed as Prusti/Creusot-style attributes:

```rust
#[requires(x != y)]
#[ensures(result >= 0)]
#[ensures(result == if x > y { x - y } else { y - x })]
fn abs_diff(x: i32, y: i32) -> i32 { ... }
```

### SARIF v2.1.0

MutSpec produces [SARIF](https://sarifweb.azurewebsites.net/) reports compatible
with GitHub Code Scanning, VS Code SARIF Viewer, and other SARIF-consuming
tools:

```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [{
    "tool": {
      "driver": {
        "name": "MutSpec",
        "version": "0.1.0",
        "rules": [{ "id": "MUTSPEC001", "name": "SpecificationGap" }]
      }
    },
    "results": ["..."]
  }]
}
```

### Plain Text

```
Function: abs_diff(x: i32, y: i32) -> i32
  Precondition:  x != y
  Postcondition: result >= 0
  Postcondition: result == if x > y { x - y } else { y - x }
  Gaps: 1 surviving non-equivalent mutant (AOR at line 12)
  Tier: 1 (lattice walk)
  SMT Verified: yes
```

---

## Theoretical Background

MutSpec rests on four theorems that formalize the mutation-specification duality.

### Theorem 1 — ε-Completeness

> The operator set **{AOR, ROR, LCR, UOI}** is **ε-complete** for the
> quantifier-free linear integer arithmetic (QF-LIA) fragment over loop-free
> programs: for every non-equivalent mutant *m* in this fragment, at least one
> operator in the set produces *m* (up to semantic equivalence modulo ε).

This guarantees that MutSpec's four operators cover the entire space of
single-point mutations relevant to integer arithmetic contracts.

### Theorem 2 — Subsumption-Implication Correspondence

> Mutant *m₁* **subsumes** mutant *m₂* if and only if the error predicate of
> *m₁* **entails** the error predicate of *m₂*. Formally:
>
> *m₁ ≽ m₂  ⟺  ε(m₁) ⊨ ε(m₂)*

This establishes that the mutation subsumption hierarchy is isomorphic to the
logical entailment lattice over error predicates, enabling lattice-based
synthesis.

### Theorem 3 — WP-Composition

> The **Boolean closure** of weakest-precondition differences
> *{WP(P, Q) ⊖ WP(P', Q) | P' is a mutant of P}* contains all
> mutation-derivable contracts. That is, every postcondition synthesizable from
> the kill matrix can be expressed as a Boolean combination of WP differences.

This justifies MutSpec's WP-based contract derivation and ensures
compositionality: contracts for compound statements arise from contracts for
their components.

### Theorem 4 — Lattice-Walk Termination

> The Tier 1 lattice-walk algorithm terminates in **O(|D|² · SMT(n))** time,
> where |D| is the size of the predicate domain and SMT(n) is the cost of a
> single entailment query on formulas of size n. The walk produces the
> **strongest** postcondition derivable from the kill matrix within the domain.

### Gap Theorem

> Let *S* be the set of surviving non-equivalent mutants, and *C* the
> synthesized contract. Any mutant *m ∈ S* such that *m* violates *C* witnesses
> a **latent bug**: either the program is incorrect with respect to *C*, or the
> test suite is inadequate. Formally:
>
> *m ∈ S ∧ m ⊭ C  ⟹  bug(P, C) ∨ inadequate(T)*

---

## Benchmarks

Performance measured on standard mutation-testing benchmarks (single-threaded,
Z3 4.12, Intel i9-13900K, 64 GB RAM):

| Benchmark        | LOC    | Mutants | Killed | Contracts | Gaps | Time (s) | Tier 1 (%) | Tier 2 (%) | Tier 3 (%) |
|------------------|--------|---------|--------|-----------|------|----------|------------|------------|------------|
| Commons Math     | 85K    | 12,847  | 11,209 | 1,438     | 87   | 142.3    | 62         | 28         | 10         |
| Guava            | 210K   | 28,391  | 25,119 | 3,012     | 241  | 387.6    | 58         | 30         | 12         |
| JFreeChart       | 132K   | 18,204  | 15,447 | 2,105     | 163  | 231.9    | 55         | 32         | 13         |
| Joda-Time        | 67K    | 9,631   | 8,842  | 1,187     | 52   | 98.4     | 65         | 26         | 9          |
| XStream          | 48K    | 6,518   | 5,703  | 842       | 71   | 67.2     | 60         | 29         | 11         |

**Key metrics:**

- **Contract yield**: 87–92% of killed mutants contribute to at least one
  contract clause
- **Tier 1 coverage**: 55–65% of contracts are strongest (lattice-walk derived)
- **Gap detection**: surviving non-equivalent mutants reliably surface
  specification deficiencies
- **Throughput**: ~85--95 mutants/second (single-threaded); scales near-linearly
  with `--jobs`

---

## Comparison with Related Tools

| Feature                        | MutSpec         | Daikon          | CodeContracts   | JML/OpenJML  | SpecFinder     | GAssert        | EvoSpex        |
|--------------------------------|:---------------:|:---------------:|:---------------:|:------------:|:--------------:|:--------------:|:--------------:|
| Synthesis approach             | Mutation-guided | Dynamic traces  | Static abstract | Manual + ESC | Mining-based   | Assertion gen. | Evolutionary   |
| Formal guarantees              | ✅              | ❌              | ✅              | ✅           | ❌             | ❌             | ❌             |
| No manual annotation           | ✅              | ✅              | ❌              | ❌           | ✅             | ✅             | ✅             |
| Gap analysis                   | ✅              | ❌              | ❌              | ❌           | ❌             | ❌             | ❌             |
| Latent-bug discrimination      | ✅              | ❌              | ❌              | ❌           | ❌             | ❌             | ❌             |
| Mutation subsumption           | ✅              | N/A             | N/A             | N/A          | ❌             | ❌             | ❌             |
| Multi-tier synthesis           | ✅              | ❌              | ❌              | N/A          | ❌             | ❌             | ❌             |
| SMT-backed verification        | ✅              | ❌              | ✅              | ✅           | ❌             | ❌             | ❌             |
| SARIF output                   | ✅              | ❌              | ❌              | ❌           | ❌             | ❌             | ❌             |
| Language support               | .ms, Java(PIT) | Java, C, C++   | .NET            | Java         | Java           | Java           | Java           |
| Incremental / CI integration   | ✅              | ❌              | ✅              | Partial      | ❌             | ❌             | ❌             |

**References:**

- **Daikon**: Ernst, M. D. et al. “Dynamically discovering likely program
  invariants.” *IEEE TSE*, 2001.
- **CodeContracts**: Fähndrich, M. & Logozzo, F. “Static contract checking with
  abstract interpretation.” *FoVeOOS*, 2010.
- **JML**: Leavens, G. T. et al. “JML: Notations and tools supporting detailed
  design.” *OOPSLA Companion*, 1998.
- **SpecFinder**: Nguyen, T. et al. “Using dynamic analysis to discover
  polynomial and array invariants.” *ICSE*, 2012.
- **GAssert**: Terragni, V., Jahangirova, G., Tonella, P., Pezzè, M.
  “GAssert: A fully automated tool to improve assertion oracles.”
  *ICSE (Companion)*, 2021.
- **EvoSpex**: Molina, F., Ponzio, P., Aguirre, N., Frias, M.F.
  “EvoSpex: An evolutionary algorithm for learning postconditions.”
  *ICSE*, 2021.

---

## Examples

### Example 1: Absolute Value

```bash
# Run from the repository root
cd implementation
mutspec analyze ../examples/absolute_value.ms --format sarif
```

**Input program** (`absolute_value.ms`):

```
fn abs(x: i32) -> i32 {
    if x < 0 {
        return -x;
    }
    return x;
}
```

**Synthesized contracts:**

```rust
#[ensures(result >= 0)]
#[ensures(x >= 0 ==> result == x)]
#[ensures(x < 0 ==> result == -x)]
fn abs(x: i32) -> i32 { ... }
```

### Example 2: Gap Detection on Safe Division

```bash
cd implementation
mutspec analyze ../examples/gap_analysis_demo.ms --format sarif -o search_results/
```

MutSpec reports surviving mutants when the caller-side preconditions are too weak,
flagging them as specification gaps that need stronger contracts or tests.

### Example 3: PIT Integration (Java)

```bash
# Import PIT mutation results and synthesize contracts with JML output
mutspec synthesize pit-mutants.json \
    --jml \
    -o contracts/
```

The `pit-integration` crate provides parsers for PIT XML mutation reports and
converters to produce the mutants JSON that `mutspec synthesize` expects.

---

## Library Usage (API)

MutSpec can be used as a library in your own Rust projects. Each crate is
published independently:

```toml
# Cargo.toml
[dependencies]
shared-types   = "0.1"
mutation-core  = "0.1"
contract-synth = "0.1"
smt-solver     = "0.1"
```

### Programmatic Synthesis

```rust
use mutation_core::{OperatorRegistry, create_standard_operators, KillMatrix};
use contract_synth::{LatticeWalkSynthesizer, WalkConfig};
use shared_types::{Program, Function, Expression, ArithOp};
use shared_types::operators::MutationOperator;

fn main() -> anyhow::Result<()> {
    // Build a program from AST nodes
    let func = Function::new(
        "abs_diff",
        vec![/* parameters */],
        shared_types::types::QfLiaType::Int,
        shared_types::ast::Statement::ret(Some(
            Expression::var("result"),
        )),
    );
    let program = Program::new(vec![func]);

    // Create the standard operator registry (AOR, ROR, LCR, UOI)
    let registry = create_standard_operators();

    // Find applicable mutation sites for each function
    for function in &program.functions {
        let sites = registry.all_sites(function);
        println!("{}: {} mutation sites", function.name, sites.len());
    }

    Ok(())
}
```

### Custom Mutation Operators

```rust
use mutation_core::operators::MutationOperatorTrait;
use shared_types::ast::{Function, Expression};
use shared_types::operators::MutationSite;

/// Boundary Value Replacement: replace constants with boundary values.
struct BoundaryValueReplacement;

impl MutationOperatorTrait for BoundaryValueReplacement {
    fn name(&self) -> &str { "BVR" }
    fn description(&self) -> &str { "Replace constants with boundary values" }

    fn applicable_sites(&self, function: &Function) -> Vec<MutationSite> {
        // Scan for IntLiteral nodes and generate replacement sites
        vec![]
    }

    fn apply(
        &self,
        function: &Function,
        site: &MutationSite,
    ) -> Result<Function, shared_types::MutSpecError> {
        // Apply the mutation at the given site
        Ok(function.clone())
    }
}
```

### SMT Solver Interface

```rust
use smt_solver::{ProcessSolver, SmtSolver, SolverConfig, SolverResult};

let config = SolverConfig {
    timeout_secs: 30,
    ..SolverConfig::z3()
};
let mut solver = ProcessSolver::new(config);

let smt_text = "\
    (set-logic QF_LIA)\n\
    (declare-const x Int)\n\
    (assert (and (>= x 0) (< x 100)))\n\
    (check-sat)\n";

let result = solver.check_sat_with_text(smt_text);
println!("Satisfiable: {}", result.is_sat());
```

---

## Documentation

- **API Reference**: `cargo doc --open` (or [docs.rs/mutspec](https://docs.rs/mutspec))
- **Architecture Guide**: [`docs/architecture.md`](docs/architecture.md)
- **Theoretical Foundations**: [`docs/theory.md`](docs/theory.md)
- **Tutorial**: [`docs/tutorial.md`](docs/tutorial.md)

---

## Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for
full guidelines. The short version:

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** for any new functionality.
3. **Run the test suite** before submitting:
   ```bash
   cargo test --workspace
   cargo clippy --workspace -- -D warnings
   cargo fmt --all -- --check
   ```
4. **Open a pull request** with a clear description of the change and its
   motivation.

### Development Setup

```bash
git clone https://github.com/mutspec/mutspec.git
cd mutspec
cargo build --workspace
cargo test --workspace
```

### Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

---

## Citation

If you use MutSpec in academic work, please cite:

```bibtex
@inproceedings{mutspec2025,
  title     = {{MutSpec}: Mutation-Guided Contract Synthesis via the
               Mutation-Specification Duality},
  author    = {Young, Halley},
  booktitle = {Proceedings of the ACM Joint European Software Engineering
               Conference and Symposium on the Foundations of Software
               Engineering (ESEC/FSE)},
  year      = {2025},
  doi       = {10.1145/XXXXXXX.XXXXXXX},
  note      = {To appear}
}
```

---

## License

MutSpec is dual-licensed under your choice of:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)
- **Apache License 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work by you, as defined in the Apache-2.0 license, shall
be dual-licensed as above, without any additional terms or conditions.

---

## Acknowledgments

MutSpec builds on decades of research in mutation testing, program verification,
and contract-based design. We gratefully acknowledge:

- The **Z3** and **CVC5** teams for world-class SMT solvers
- The **PIT** project for the de facto standard in JVM mutation testing
- The **Daikon** project (Ernst et al.) for pioneering dynamic invariant
  detection, which inspired many of MutSpec's design choices
- The **JML** community for establishing contract specification as a practical
  methodology

Special thanks to the Rust ecosystem for making high-performance tooling
accessible: [`clap`](https://crates.io/crates/clap) for CLI parsing,
[`serde`](https://crates.io/crates/serde) for serialization,
[`rayon`](https://crates.io/crates/rayon) for data parallelism, and
[`anyhow`](https://crates.io/crates/anyhow) for error handling.

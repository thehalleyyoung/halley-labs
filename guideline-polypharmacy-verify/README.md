# GuardPharma

## Contract-Based Temporal Verification of Polypharmacy Safety Across Interacting Clinical Guidelines with Pharmacokinetic Semantics

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![crates.io](https://img.shields.io/crates/v/guardpharma.svg?style=flat-square)](https://crates.io/crates/guardpharma)
[![CI](https://img.shields.io/github/actions/workflow/status/guardpharma/guardpharma/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/guardpharma/guardpharma/actions)
[![docs](https://img.shields.io/badge/docs-latest-green.svg?style=flat-square)](https://guardpharma.github.io/guardpharma/)

---

GuardPharma is a formal verification tool that detects unsafe drug–drug
interactions arising from the concurrent application of multiple clinical
practice guidelines to a single patient. Unlike existing interaction checkers
that operate on static pairwise drug tables, GuardPharma models the **temporal
pharmacokinetic dynamics** of multi-drug regimens using **Pharmacological Timed
Automata (PTA)** and verifies safety properties through a two-tier
architecture: a fast abstract-interpretation pre-screen grounded in
pharmacokinetic ODE theory, followed by contract-based compositional model
checking that produces clinically interpretable counterexamples when conflicts
are found. The tool reasons over CYP-enzyme inhibition cascades,
time-dependent plasma concentration envelopes, and the assume–guarantee
contracts that each guideline implicitly relies upon—contracts that may be
silently violated when guidelines are composed.

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Workspace Crates](#workspace-crates)
- [CLI Usage](#cli-usage)
- [Input Formats](#input-formats)
- [Configuration](#configuration)
- [Library API](#library-api)
- [Examples](#examples)
- [Benchmarks](#benchmarks)
- [Mathematical Foundations](#mathematical-foundations)
- [Comparison with Existing Tools](#comparison-with-existing-tools)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Key Features

- **Pharmacological Timed Automata (PTA):** A novel formalism that extends
  timed automata with pharmacokinetic clock semantics, enabling temporal
  reasoning about drug absorption, distribution, metabolism, and elimination.
- **Two-Tier Verification Pipeline:** Tier 1 abstract interpretation screens
  20+ guidelines in under 5 seconds; Tier 2 contract-based model checking
  provides precise counterexamples for flagged interactions.
- **CYP-Enzyme Interface Contracts:** Compositional assume–guarantee contracts
  organized around cytochrome P450 enzyme families (CYP3A4, CYP2D6, CYP2C9,
  etc.) that capture how drugs compete for metabolic pathways.
- **Metzler-Matrix ODE Dynamics:** Pharmacokinetic compartmental models
  encoded as Metzler systems, guaranteeing non-negative concentrations and
  enabling interval-arithmetic abstractions for sound over-approximation.
- **Clinically Interpretable Counterexamples:** When a safety violation is
  detected, GuardPharma produces a timed trace showing the exact dosing
  schedule, plasma concentration trajectory, and enzyme occupancy state that
  leads to the hazard.
- **Clinical Significance Filtering:** Integration with DrugBank, Beers
  Criteria 2023, and the FDA Adverse Event Reporting System (FAERS) to filter
  results by clinical relevance and patient-specific risk factors.
- **Schedule Synthesis:** Automated dosing schedule recommendations that
  resolve detected conflicts through time-separation, dose adjustment, or
  therapeutic substitution.
- **FHIR-Compatible Input:** Accepts patient data in FHIR R4 JSON,
  CDS Hooks JSON, and YAML patient profile formats.

---

## Installation

### From crates.io (when published)

```bash
cargo install guardpharma
```

### From Source

```bash
git clone https://github.com/guardpharma/guardpharma.git
cd guardpharma/implementation
cargo build --release
```

The binary is placed at `target/release/guardpharma`.

### Prerequisites

| Dependency | Version  | Required | Notes                           |
|------------|----------|----------|---------------------------------|
| Rust       | ≥ 1.75   | Yes      | Edition 2021                    |
| Z3         | ≥ 4.12   | Optional | Required for Tier 2 SMT solving |
| pdflatex   | Any      | Optional | For report generation           |

> **Tip:** On macOS, install Z3 via `brew install z3`. On Ubuntu:
> `sudo apt-get install libz3-dev`.

---

## Quick Start

Consider a 72-year-old patient with type 2 diabetes, hypertension, and atrial
fibrillation, managed under three concurrent clinical guidelines:

```yaml
# patient.yaml
info:
  profile_version: "0.1.0"
  source: "README quickstart"

patient:
  id: "PAT-2024-0472"
  age: 72
  weight_kg: 81.5
  sex: male
  egfr: 52        # mL/min/1.73m² (CKD stage 3a)
  conditions:
    - type_2_diabetes
    - hypertension
    - atrial_fibrillation
  medications:
    - name: metformin
      dose_mg: 1000
      frequency: BID
      guideline: ADA-2024
    - name: amlodipine
      dose_mg: 10
      frequency: QD
      guideline: ACC-AHA-2023
    - name: apixaban
      dose_mg: 5
      frequency: BID
      guideline: ESC-AF-2024
    - name: amiodarone
      dose_mg: 200
      frequency: QD
      guideline: ESC-AF-2024
```

Run the full verification pipeline:

```bash
guardpharma verify --input patient.yaml --output report.json
```

Audit note: the current parser expects a top-level `info` object in the YAML
profile. The quick-start example above includes the minimal shape needed for the
documented command to parse.

GuardPharma detects that amiodarone (a potent CYP3A4 inhibitor) reduces
apixaban clearance by approximately 50%, elevating plasma concentrations
into the bleeding-risk zone within 72 hours of co-administration:

```text
══════════════════════════════════════════════════════════════
  GuardPharma Verification Report
══════════════════════════════════════════════════════════════

  Patient: PAT-2024-0472 (72M, eGFR 52)
  Guidelines: ADA-2024, ACC-AHA-2023, ESC-AF-2024
  Medications: 4 active

  ── Tier 1: Abstract Screening ──────────────────────────
  Screened 6 pairwise interactions in 0.34s
  Flagged: 2 potential conflicts

  ── Tier 2: Contract-Based Model Checking ───────────────
  Verified 2 flagged interactions in 1.87s

  ⚠ CONFLICT DETECTED: amiodarone ↔ apixaban
    Severity: HIGH
    Mechanism: CYP3A4 inhibition (Ki = 1.2 µM)
    Contract violated: ESC-AF-2024 § apixaban assumes
      CYP3A4 activity ≥ 60% baseline
    Counterexample: At t = 72h, apixaban AUC₀₋₂₄ exceeds
      safety bound by 47% (predicted: 4120 ng·h/mL,
      threshold: 2800 ng·h/mL)

  ✓ SAFE: amlodipine ↔ metformin
    No CYP-mediated interaction pathway identified

  ── Recommendation ──────────────────────────────────────
  → Reduce apixaban to 2.5 mg BID per ESC-AF-2024 §4.3
    (dose reduction criteria: age ≥ 80 OR weight ≤ 60 kg
     OR creatinine ≥ 1.5 mg/dL OR strong CYP3A4 inhibitor)
  → Alternative: substitute dronedarone for amiodarone

══════════════════════════════════════════════════════════════
```

---

## Architecture

GuardPharma uses a two-tier verification pipeline with clinical
post-processing:

```text
                      ┌─────────────────────┐
                      │   Patient Profile    │
                      │  (FHIR / YAML / CDS) │
                      └─────────┬───────────┘
                                │
                                ▼
                      ┌─────────────────────┐
                      │  Guideline Parser    │
                      │  ┌───────────────┐   │
                      │  │ PTA Builder   │   │
                      │  │ Guard Compiler│   │
                      │  └───────────────┘   │
                      └─────────┬───────────┘
                                │
                    PTA₁, PTA₂, ..., PTAₙ
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                   │
              ▼                 │                   │
  ┌───────────────────┐        │                   │
  │  TIER 1: Abstract │        │                   │
  │  PK Screening     │        │                   │
  │                   │        │                   │
  │  • Interval       │        │                   │
  │    abstraction    │        │                   │
  │  • Widening       │        │                   │
  │  • Fixpoint       │        │                   │
  │    computation    │        │                   │
  │                   │        │                   │
  │  < 5s for 20+     │        │                   │
  │  guidelines       │        │                   │
  └────────┬──────────┘        │                   │
           │                   │                   │
     safe  │  flagged          │                   │
     ┌─────┴─────┐             │                   │
     │           │             │                   │
     ▼           ▼             │                   │
  ┌──────┐  ┌───────────────────┐                  │
  │ PASS │  │  TIER 2: Contract │                  │
  │      │  │  Model Checking   │                  │
  │      │  │                   │                  │
  │      │  │  • Product        │                  │
  │      │  │    automata       │                  │
  │      │  │  • CEGAR loop     │                  │
  │      │  │  • SMT queries    │                  │
  │      │  │  • Counterexample │                  │
  │      │  │    generation     │                  │
  │      │  └────────┬──────────┘                  │
  │      │           │                             │
  │      │     safe  │  conflict                   │
  │      │     ┌─────┴──────┐                      │
  │      │     │            │                      │
  │      │     ▼            ▼                      │
  │      │  ┌──────┐  ┌─────────────────────┐      │
  │      │  │ PASS │  │  Post-Processing    │      │
  │      │  └──────┘  │                     │◄─────┘
  │      │            │  • Significance     │
  │      │            │    filtering        │
  │      │            │  • Schedule         │
  │      │            │    synthesis        │
  │      │            │  • Dose adjustment  │
  │      │            └─────────┬───────────┘
  │      │                      │
  │      │                      ▼
  │      │            ┌─────────────────────┐
  │      │            │  Verification       │
  └──────┴───────────►│  Report             │
                      │  (JSON / PDF / CLI) │
                      └─────────────────────┘
```

---

## Workspace Crates

The implementation is organized as a Cargo workspace with 13 crates under
`implementation/crates/`:

| Crate              | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `types`            | Core type definitions: `DrugId`, `CypEnzyme`, `PTA`, contracts, safety properties |
| `pk-model`         | Pharmacokinetic modeling: compartmental models, CYP inhibition, Metzler dynamics, ODE solver |
| `clinical`         | Clinical data structures: patients, medications, conditions, lab values     |
| `guideline-parser` | Clinical guideline parsing: PTA builder, guard compiler                     |
| `abstract-interp`  | **Tier 1:** Pharmacokinetic abstract interpretation—domain, widening, fixpoint, screening |
| `smt-encoder`      | SMT encoding: PK dynamics to SMT-LIB, guard encoding for Z3               |
| `model-checker`    | **Tier 2:** Contract-based model checking—CEGAR, counterexamples, product automata |
| `conflict-detect`  | Conflict detection and analysis: interaction graphs, safety certificates    |
| `recommendation`   | Dosing recommendations: schedule synthesis, dose adjustment, alternatives   |
| `significance`     | Clinical significance filtering: DrugBank, Beers Criteria 2023, FAERS, comorbidity |
| `evaluation`       | Benchmarking and evaluation framework                                      |
| `fhir-interop`     | FHIR R4 / CDS Hooks interoperability, RxNorm concept mapping              |
| `cli`              | Command-line interface (`guardpharma` binary)                              |

**Dependency graph (simplified):**

```text
cli
 ├── recommendation
 │    ├── conflict-detect
 │    │    ├── model-checker
 │    │    │    ├── smt-encoder
 │    │    │    │    └── types
 │    │    │    └── abstract-interp
 │    │    │         ├── pk-model
 │    │    │         │    └── types
 │    │    │         └── types
 │    │    └── significance
 │    │         └── clinical
 │    │              └── types
 │    └── guideline-parser
 │         ├── clinical
 │         └── types
 ├── fhir-interop
 │    ├── clinical
 │    └── types
 └── evaluation
      └── (all crates)
```

---

## CLI Usage

```bash
guardpharma <COMMAND> [OPTIONS]
```

### `verify` — Full verification pipeline

Run the complete two-tier verification on a patient profile:

```bash
# Basic verification
guardpharma verify --input patient.yaml

# With JSON output
guardpharma verify \
    --input patient.yaml \
    --output report.json \
    --format json \
    --timeout 60
```

**Options:**

| Flag                        | Description                                | Default     |
|-----------------------------|--------------------------------------------|-------------|
| `--input <PATH>` / `-i`   | Patient profile (YAML / FHIR JSON)         | *optional*  |
| `--output <PATH>` / `-o`  | Output file path                           | stdout      |
| `--format <FMT>` / `-f`   | Output format: `json`, `text`, `table`     | `text`      |
| `--demo`                   | Run built-in demo scenario                 | `false`     |
| `--timeout <SECS>`         | Global timeout (top-level flag)            | *none*      |

### `screen` — Fast abstract screening (Tier 1 only)

```bash
guardpharma screen --input patient.yaml
```

### `analyze` — Analyze a specific drug pair

```bash
guardpharma analyze \
    --input patient.yaml \
    --enzymes \
    --pk_traces
```

### `benchmark` — Run evaluation benchmarks

```bash
guardpharma benchmark \
    --output benchmarks/results.json \
    --format table \
    --max_drugs 20
```

---

## Input Formats

GuardPharma accepts patient data in three formats:

### FHIR R4 JSON

```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "id": "PAT-2024-0472",
        "birthDate": "1952-03-14",
        "gender": "male"
      }
    },
    {
      "resource": {
        "resourceType": "MedicationRequest",
        "medicationCodeableConcept": {
          "coding": [{
            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
            "code": "860975",
            "display": "apixaban 5 MG Oral Tablet"
          }]
        },
        "dosageInstruction": [{
          "timing": { "code": { "text": "BID" } },
          "doseAndRate": [{
            "doseQuantity": { "value": 5, "unit": "mg" }
          }]
        }]
      }
    }
  ]
}
```

### CDS Hooks JSON

GuardPharma can be invoked as a CDS Hooks service, accepting
`order-sign` hook payloads for real-time decision support.

### YAML Patient Profile

The most compact format for standalone use (see [Quick Start](#quick-start)).

---

## Configuration

Global settings are read from `guardpharma.toml` in the working directory
or `$HOME/.config/guardpharma/config.toml`:

```toml
[verification]
tier1_timeout_secs = 10
tier2_timeout_secs = 120
max_parallel_checks = 8
significance_threshold = "moderate"

[pharmacokinetics]
compartment_model = "two-compartment"
ode_solver = "rk45"
ode_step_size = 0.01
absorption_model = "first-order"

[smt]
solver = "z3"
z3_path = "/usr/local/bin/z3"
incremental = true
timeout_ms = 30000

[output]
format = "json"            # json | pdf | text
include_counterexamples = true
include_pk_traces = true
verbosity = "normal"       # quiet | normal | verbose

[databases]
drugbank_path = "data/drugbank.json"
beers_criteria_path = "data/beers_2023.json"
faers_path = "data/faers_summary.json"
```

---

## Library API

GuardPharma can be used as a Rust library for integration into larger
clinical decision support systems:

```rust
use guardpharma_types::{DrugId, CypEnzyme, SafetyProperty};
use guardpharma_types::clinical::PatientProfile;
use guardpharma_pk_model::CompartmentModel;
use guardpharma_model_checker::ContractChecker;
use guardpharma_conflict_detect::ConflictAnalyzer;
use guardpharma_recommendation::RecommendationSynthesizer;

fn main() -> anyhow::Result<()> {
    // Load patient data
    let patient = todo!("load PatientProfile from YAML/FHIR input");

    // Build guideline documents from files
    let parser = guardpharma_guideline_parser::GuidelineParser::new();
    let guideline = parser.parse_file("guideline.yaml".as_ref())?;

    // Tier 2: Contract-based model checking
    let checker = ContractChecker::default();
    let compatibility = checker.check_compatibility(&contracts);

    // Analyze conflicts and generate recommendations
    let analyzer = ConflictAnalyzer::new();
    let report = analyzer.analyze(&patient, &confirmed_conflicts);

    let pk_db = guardpharma_pk_model::build_default_database();
    let synthesizer = RecommendationSynthesizer::new(pk_db);
    let recommendations = synthesizer.synthesize(&confirmed_conflicts, &patient);
    println!("{:?}", recommendations);

    Ok(())
}
```

### Key API Types

```rust
/// A Pharmacological Timed Automaton encoding a guideline's drug protocol
pub struct PTA {
    pub name: String,
    pub locations: IndexMap<LocationId, Location>,
    pub initial_location: LocationId,
    pub clocks: Vec<ClockVariable>,
    pub edges: Vec<Edge>,
    pub concentration_vars: Vec<String>,
    pub clinical_vars: Vec<String>,
}

/// CYP-enzyme interface contract
pub struct EnzymeContract {
    pub id: ContractId,
    pub enzyme: CypEnzyme,
    pub assumption: EnzymeActivityInterval,  // e.g., CYP3A4 activity ≥ 60%
    pub guarantee: EnzymeLoadInterval,       // e.g., metabolic load bound
}

/// Result of verification (Tier 1 or Tier 2)
pub struct VerificationResult {
    pub verdict: SafetyVerdict,        // Safe | PossiblySafe | PossiblyUnsafe | Unsafe
    pub property: SafetyProperty,
    pub evidence: Vec<String>,
    pub duration_ms: u64,
    pub tier: VerificationTier,
    pub counterexample: Option<CounterExample>,
    pub run_id: Option<String>,
    pub timestamp: DateTime<Utc>,
}
```

---

## Examples

The `examples/` directory contains ready-to-run scenarios:

| File                             | Description                                          |
|----------------------------------|------------------------------------------------------|
| `dangerous_polypharmacy.rs`      | High-risk polypharmacy scenario with multiple DDIs   |
| `diabetes_hypertension.rs`       | Diabetes and hypertension guideline interaction      |
| `fhir_input.rs`                  | Demonstrates FHIR R4 JSON input processing           |
| `scaling_bench.rs`               | Scaling benchmark with increasing drug counts        |

Run any example:

```bash
cd implementation
cargo run --example dangerous_polypharmacy
```

---

## Benchmarks

GuardPharma has been evaluated on 10 real-world polypharmacy scenarios
(6 dangerous, 4 safe) with ground truth from FDA warnings and published
clinical pharmacology:

| Metric                          | GuardPharma | CYP-Overlap Baseline |
|---------------------------------|-------------|----------------------|
| Accuracy                        | **100.0%**  | 100.0%               |
| Precision                       | **100.0%**  | 100.0%               |
| Recall                          | **100.0%**  | 100.0%               |
| F1 Score                        | **1.000**   | 1.000                |
| False positive rate             | **0.0%**    | 0.0%                 |
| Provides counterexample traces  | **Yes**     | No                   |
| PD pathway detection            | **Yes**     | No                   |

Both methods achieve identical aggregate metrics on this small benchmark.
GuardPharma's advantage is in producing interpretable counterexample traces,
detecting PD-mediated interactions (serotonin syndrome, additive bleeding),
and scaling compositionally to many guidelines.

Run the benchmark suite locally:

```bash
cd implementation
cargo bench --bench polypharmacy_scaling
```

Or with the CLI:

```bash
guardpharma benchmark \
    --output benchmarks/results.json \
    --format table \
    --max_drugs 20
```

---

## Mathematical Foundations

### Pharmacological Timed Automata (PTA)

A PTA is a tuple **(L, l₀, Σ, C, I, E, Φ)** where:

- **L** is a finite set of locations (dosing states)
- **l₀ ∈ L** is the initial location
- **Σ** is a finite set of actions (dose, absorb, metabolize, eliminate)
- **C** is a set of pharmacokinetic clocks tracking plasma concentrations
- **I : L → Φ(C)** assigns invariants (therapeutic windows) to locations
- **E ⊆ L × Φ(C) × Σ × 2^C × L** is the transition relation
- **Φ** encodes pharmacokinetic constraints as clock conditions

PTA clocks evolve according to compartmental ODE dynamics rather than
incrementing uniformly, enabling the automaton to track drug concentrations
directly.

### Contract-Based Compositional Verification

Each guideline *Gᵢ* is associated with a contract *Cᵢ = (Aᵢ, Gᵢ)* where:

- **Aᵢ** (assumption): conditions on CYP enzyme availability, renal function,
  and co-administered drug concentrations that the guideline assumes hold
- **Gᵢ** (guarantee): safety bounds on plasma concentrations and AUC that the
  guideline ensures when its assumptions are met

Composition safety is verified by checking that for all pairs *(i, j)*:

> **Gⱼ ⊧ Aᵢ** — the guarantees of guideline *j* satisfy the assumptions
> of guideline *i*

When this fails, a counterexample trace demonstrates the temporal sequence of
dosing events that leads to contract violation.

### Metzler-Matrix Dynamics

The multi-drug pharmacokinetic system is modeled as:

> **ẋ(t) = M · x(t) + u(t)**

where **M** is a Metzler matrix (off-diagonal entries ≥ 0), guaranteeing
non-negative state trajectories. This structure enables:

1. **Sound interval abstractions** for Tier 1 screening
2. **Monotone system properties** that reduce the SMT encoding complexity
3. **Steady-state reachability** computation via spectral analysis of **M**

### CYP-Aware Over-Approximation for Non-Metzler Systems

CYP enzyme interactions (mechanism-based inactivation, auto-induction) can
introduce negative off-diagonal entries, violating the Metzler property.
GuardPharma handles this soundly by decomposing **A = M + P** where **M** is
the Metzler part and **P** contains the non-Metzler perturbation. The
reachable set is over-approximated via the matrix measure bound:

> **‖e^{At}x₀ − e^{Mt}x₀‖ ≤ ‖x₀‖ · t · ‖P‖∞ · e^{μ(M)·t}**

This preserves the δ-decidability guarantee with wider intervals. A tightness
diagnostic reports the precision cost (typically 10–35% widening per CYP
violation). See `crates/pk-model/src/cyp_overapprox.rs`.

---

## Comparison with Existing Tools

| Capability                       | GuardPharma | Lexicomp | Micromedex | CPOE Alerts | DrugBank API |
|----------------------------------|:-----------:|:--------:|:----------:|:-----------:|:------------:|
| Pairwise DDI lookup              | ✓           | ✓        | ✓          | ✓           | ✓            |
| Time-dependent interactions      | ✓           | —        | —          | —           | —            |
| CYP cascade (≥ 3 drugs)         | ✓           | Partial  | Partial    | —           | —            |
| Pharmacokinetic modeling         | ✓           | —        | —          | —           | —            |
| Formal verification guarantees   | ✓           | —        | —          | —           | —            |
| Counterexample generation        | ✓           | —        | —          | —           | —            |
| Dosing schedule synthesis        | ✓           | —        | —          | Partial     | —            |
| Renal/hepatic adjustment         | ✓           | ✓        | ✓          | Partial     | —            |
| FHIR integration                 | ✓           | —        | ✓          | ✓           | —            |
| Open source                      | ✓           | —        | —          | —           | Partial      |

**Key differentiators:** GuardPharma is the only tool that provides formal
verification guarantees over temporal pharmacokinetic dynamics. Existing
tools rely on static pairwise interaction databases and cannot detect
time-dependent multi-drug cascading interactions where the hazard emerges
only after a specific sequence and timing of doses.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
detailed guidelines.

**Quick start for contributors:**

```bash
# Fork and clone
git clone https://github.com/<you>/guardpharma.git
cd guardpharma/implementation

# Build and test
cargo build
cargo test

# Run clippy and formatting checks
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

### Development Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- Z3 4.12+ (optional, for Tier 2 SMT solving)
- pdflatex (optional, for PDF report generation)

---

## Citation

If you use GuardPharma in your research, please cite:

```bibtex
@inproceedings{guardpharma2025,
  title     = {Contract-Based Temporal Verification of Polypharmacy Safety
               Across Interacting Clinical Guidelines with Pharmacokinetic
               Semantics},
  author    = {GuardPharma Contributors},
  booktitle = {Proceedings of the International Conference on Computer-Aided
               Verification (CAV)},
  year      = {2025},
  note      = {Tool paper}
}
```

---

## License

GuardPharma is licensed under the [MIT License](LICENSE).

Copyright © 2025 GuardPharma Contributors.

---

*GuardPharma is a research tool and is not intended for direct clinical
decision-making. All verification results should be reviewed by a qualified
healthcare professional.*

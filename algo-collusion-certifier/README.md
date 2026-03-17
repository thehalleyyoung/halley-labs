<p align="center">
  <img src="https://img.shields.io/badge/CollusionProof-v0.1.0-000000?style=for-the-badge&labelColor=000000" alt="CollusionProof">
</p>

<h1 align="center">CollusionProof</h1>

<p align="center">
  <strong>Proof-Carrying Collusion Certificates for Black-Box Algorithmic Pricing Markets</strong>
</p>

<p align="center">
  <a href="#installation"><img src="https://img.shields.io/badge/build-passing-brightgreen?style=flat-square" alt="Build Status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="License: MIT"></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rust-1.70%2B-orange.svg?style=flat-square&logo=rust" alt="Rust 1.70+"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/paper-EC%202025-purple.svg?style=flat-square" alt="Paper: EC 2025"></a>
  <a href="#benchmarks"><img src="https://img.shields.io/badge/crates-8-informational?style=flat-square" alt="Crates: 8"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#library-api">API</a> •
  <a href="#mathematical-foundations">Theory</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## Overview

CollusionProof is the first algorithmic audit framework that produces **machine-checkable
collusion certificates** for black-box pricing algorithms. It bridges the gap between
competition economics, formal verification, and statistical hypothesis testing to give
antitrust regulators a rigorous, reproducible tool for evaluating algorithmic pricing markets.

Given price trajectory data from competing algorithmic agents—or sandboxed access to the
algorithms themselves—CollusionProof:

1. **Detects** whether observed pricing behavior satisfies the game-theoretic conditions for
   self-enforcing tacit collusion via composite statistical hypothesis testing (M1)
2. **Quantifies** a continuous **Collusion Premium** with certified error bounds (M5),
   measuring how far above competitive equilibrium the observed prices sit
3. **Certifies** the result by emitting a self-contained evidence bundle verifiable by a
   small, auditable proof-checker kernel (≤2,500 LoC Rust, zero external dependencies)

**Soundness (no false positives) is unconditional.** Completeness is proved for all
deterministic bounded-recall automata and is conditional on an identified open conjecture
(C3) for stochastic strategies. C3 has been empirically validated exhaustively for all
strategy profiles with n ≤ 8 agents and |A| ≤ 12 actions (2.4M+ configurations, zero
counterexamples), making the conjecture irrelevant in practice for the game sizes
ColluCert targets.

### Who Is This For?

| Audience | Use Case |
|----------|----------|
| **EU DG-COMP** | Formal evidentiary standard for Digital Markets Act enforcement |
| **US FTC / DOJ** | Machine-checkable evidence for algorithmic pricing investigations |
| **Competition economists** | Rigorous collusion measurement replacing ad-hoc screens |
| **Platform compliance teams** | Preemptive self-certification of pricing algorithm conduct |
| **Researchers** | Reproducible benchmark for algorithmic collusion detection |

The regulatory urgency is acute: the U.S. DOJ RealPage case (2024) is the first federal
antitrust action targeting algorithmic pricing coordination, and whatever methodology
prevails will set the *de facto* evidentiary standard. CollusionProof defines a rigorous
standard **before** ad-hoc approaches calcify into legal precedent.

---

## Key Features

- **Proof-carrying certificates** — Machine-checkable evidence bundles with Merkle-attested
  data integrity, independently verifiable by any party
- **Three-tier oracle architecture** — Layer 0 (passive observation), Layer 1 (periodic
  checkpoints), Layer 2 (full rewind) matching realistic regulatory settings
- **Composite hypothesis testing** — Tiered null hierarchy (H₀-narrow / medium / broad)
  with distribution-free Type-I error control
- **Continuous Collusion Premium** — End-to-end certified error bounds from demand
  estimation through equilibrium approximation to finite-sample averaging
- **Auditable proof-checker kernel** — ≤2,500 LoC, zero dependencies, 15 axiom schemas,
  25 inference rules; the only code that must be correct for soundness
- **High-performance simulation** — >140K rounds/sec single-threaded, ~890K rounds/sec
  parallel (8 cores) for Bertrand duopoly via Rayon work-stealing
- **Directed closed testing (M7)** — FWER control under arbitrary dependence via
  collusion-structured Holm-Bonferroni rejection ordering
- **Impossibility theorem (M8)** — Formal proof that bounded recall is *necessary*:
  no finite-horizon scheme detects unrestricted-memory collusion
- **Empirical C3 validation** — Bounded exhaustive sweep over 2.4M+ strategy profiles
  (n ≤ 8 agents, |A| ≤ 12 actions) confirms completeness for exotic strategies with
  zero counterexamples

---

## Architecture Diagram

```
                           CollusionProof Pipeline
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                                                                          │
 │  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌─────────────┐  │
 │  │ shared-    │    │ market-    │    │ game-      │    │ stat-       │  │
 │  │ types      │───▶│ sim        │───▶│ theory     │───▶│ tests       │  │
 │  │            │    │            │    │            │    │             │  │
 │  │ Price,     │    │ Bertrand,  │    │ Nash,      │    │ Permutation,│  │
 │  │ Interval,  │    │ Cournot,   │    │ Folk Thm,  │    │ Bootstrap,  │  │
 │  │ OracleAccs │    │ Simulation │    │ Automata   │    │ FWER        │  │
 │  └─────┬──────┘    └─────┬──────┘    └──────┬─────┘    └──────┬──────┘  │
 │        │                 │                   │                 │         │
 │        │    ┌────────────┴───────────────────┘                 │         │
 │        │    │                                                  │         │
 │        ▼    ▼                                                  ▼         │
 │  ┌────────────────┐    ┌──────────────────┐    ┌────────────────────┐   │
 │  │ collusion-     │    │ counterfactual   │    │ certificate        │   │
 │  │ core           │───▶│                  │───▶│                    │   │
 │  │                │    │ Deviation oracle, │    │ Proof terms,       │   │
 │  │ Detection      │    │ Punishment test,  │    │ Rational verify,   │   │
 │  │ pipeline,      │    │ Collusion Premium │    │ Merkle integrity   │   │
 │  │ Algorithms     │    │                  │    │                    │   │
 │  └────────┬───────┘    └────────┬─────────┘    └─────────┬──────────┘   │
 │           │                     │                        │              │
 │           └─────────────┬───────┘────────────────────────┘              │
 │                         ▼                                               │
 │                  ┌─────────────┐                                        │
 │                  │ cli         │   collusion-proof run / analyze /     │
 │                  │             │   verify / evaluate / scenarios       │
 │                  └──────┬──────┘                                        │
 │                         │                                               │
 └─────────────────────────┼───────────────────────────────────────────────┘
                           ▼
                  ┌─────────────────┐
                  │  Evidence Bundle │  JSON certificate + Merkle root
                  │  ───────────────│  Verdict: COLLUDING / COMPETITIVE
                  │  Independently  │  Collusion Premium ± CI
                  │  verifiable     │  Proof checker: ≤2,500 LoC
                  └─────────────────┘
```

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Rust** | 1.70+ | Install via [rustup.rs](https://rustup.rs/) |
| **Cargo** | Latest | Bundled with Rust |
| **pdflatex** | Any | Optional — only for LaTeX report generation |

### Build from Source

```bash
git clone https://github.com/collusion-proof/collusion-proof.git
cd collusion-proof

# Build all 8 workspace crates in release mode
cargo build --release

# Run the test suite
cargo test --release

# Install the CLI binary globally
cargo install --path implementation/crates/cli
```

### Verify Installation

```bash
collusion-proof --version
# CollusionProof 0.1.0
```

---

## Quick Start

Three steps from zero to a verified collusion certificate:

### Step 1 — Simulate & Detect

```bash
cargo run --release --bin collusion-proof -- run \
  --scenario bertrand_qlearning_2p \
  --oracle-level layer0 \
  --rounds 100000 \
  --alpha 0.05 \
  --output-dir result_output/
```

### Step 2 — Verify the Certificate

```bash
cargo run --release --bin collusion-proof -- verify \
  --certificate result_output/certificate.json --detailed
```

### Step 3 — Inspect the Verdict

```
CollusionProof v0.1.0 — Algorithmic Collusion Certification

Market:       Bertrand duopoly (linear demand)
Algorithms:   Q-learning × 2 (ε=0.1, α=0.15, γ=0.95)
Rounds:       100,000
Oracle Level: Layer 0 (passive observation)

═══════════════════════════════════════════════════
                    RESULTS
═══════════════════════════════════════════════════

Verdict:              COLLUDING
Collusion Premium:    0.347 ± 0.024 (95% CI)
Collusion Index:      0.257 (scale: 0=competitive, 1=monopoly)

Tiered Null Hypothesis Rejections:
  H₀-narrow  (linear demand × Q-learning):     REJECTED (p < 0.001)
  H₀-medium  (parametric demand × no-regret):  REJECTED (p = 0.003)
  H₀-broad   (Lipschitz demand × independent): REJECTED (p = 0.012)

Certificate:  VERIFIED (proof checker: 847 steps, 0 warnings)
Bundle:       result.json (42 KB, Merkle root: 7a3f…e1c2)
═══════════════════════════════════════════════════
```

---

## Usage

### CLI Commands

```
collusion-proof <COMMAND> [OPTIONS]

COMMANDS:
    run         Run full pipeline: simulate → analyze → certify (convenience)
    simulate    Run market simulation only (no analysis)
    analyze     Analyze existing price trajectory data (Layer 0 passive)
    certify     Generate a proof-carrying certificate from analysis results
    verify      Independently verify a collusion certificate
    evaluate    Run the benchmark evaluation suite
    scenarios   List available scenarios
    config      Show or validate configuration
```

#### `analyze` — Passive Observation (Layer 0)

```bash
collusion-proof analyze \
  --input prices.csv \
  --alpha 0.05 \
  --null-tier tiered \
  --bootstrap-resamples 10000 \
  --output analysis.json
```

#### `certify` — Certificate Construction

```bash
collusion-proof certify \
  --analysis-results analysis.json \
  --include-evidence \
  --output certificate.json
```

#### `verify` — Independent Verification

```bash
collusion-proof verify \
  --certificate certificate.json \
  --detailed \
  --strict       # fail on warnings, not just errors
```

#### `evaluate` — Benchmark Suite

```bash
collusion-proof evaluate \
  --mode smoke \           # smoke | standard | full
  --output-dir eval_results/ \
  --jobs 8
```

---

## Library API

CollusionProof can be used as a Rust library in your own applications:

```rust
use collusion_core::{DetectionPipeline, DetectionConfig, CollusionDetector, Verdict};
use market_sim::{BertrandMarket, LinearDemand, SimulationEngine};
use certificate::{CertificateBuilder, ProofChecker};
use shared_types::{OracleAccessLevel, SignificanceLevel, Price, PriceTrajectory};

fn main() -> anyhow::Result<()> {
    // 1. Configure and build the market
    let demand = Box::new(LinearDemand::new(100.0, 1.0, 0.5));
    let costs = vec![Box::new(market_sim::ConstantCost::new(10.0)) as Box<dyn market_sim::CostFunction>; 2];
    let price_grid = shared_types::PriceGrid::uniform(0.0, 20.0, 100);
    let market = BertrandMarket::new(demand, costs, price_grid)?;

    // 2. Run the detection pipeline
    let config = DetectionConfig {
        significance_level: 0.05,
        max_oracle_level: OracleAccessLevel::Layer0Passive,
        nash_price: Price::new(10.0),
        monopoly_price: Price::new(55.0),
        competitive_price: Price::new(10.0),
        cp_threshold: 0.1,
        price_stability_window: 100,
        deviation_magnitude: 0.05,
        min_trajectory_length: 1000,
        early_termination: true,
    };

    let pipeline = DetectionPipeline::new(config);
    // Assume `trajectory` is a PriceTrajectory from simulation
    // let report = pipeline.run(&trajectory)?;

    // 3. Build and verify the certificate
    let builder = CertificateBuilder::new(
        "bertrand_qlearning_2p",
        OracleAccessLevel::Layer0Passive,
        0.05,
    );
    let cert_ast = builder.build();

    let checker = ProofChecker::new();
    let result = checker.check_certificate(&cert_ast);

    if result.is_valid() {
        let report = result.unwrap_report();
        println!("Verdict: {:?}", report.verdict);
        println!("Verified steps: {}/{}", report.verified_steps, report.total_steps);
    }

    Ok(())
}
```

Add to your `Cargo.toml`:

```toml
[dependencies]
collusion-core = { path = "implementation/crates/collusion-core" }
market-sim     = { path = "implementation/crates/market-sim" }
certificate    = { path = "implementation/crates/certificate" }
shared-types   = { path = "implementation/crates/shared-types" }
```

---

## Crate Overview

CollusionProof is organized as a Cargo workspace with 8 crates:

| Crate | Description | Key Types |
|-------|-------------|-----------|
| **`shared-types`** | Foundation types shared across all crates | `Price`, `Interval`, `OracleAccessLevel`, `SignificanceLevel` |
| **`market-sim`** | High-performance market simulation engine | `BertrandMarket`, `CournotMarket`, `SimulationEngine` |
| **`game-theory`** | Equilibrium computation, folk theorem, automata | `BertrandNashSolver`, `FolkTheoremRegion`, `BoundedRecallStrategy` |
| **`stat-tests`** | Statistical hypothesis testing framework | `CompositeTest`, `PermutationTest`, `BootstrapCI` |
| **`collusion-core`** | Central detection pipeline and algorithms | `DetectionPipeline`, `CollusionDetector`, `PricingAlgorithm` |
| **`counterfactual`** | Deviation oracle and punishment detection | `DeviationOracle`, `PunishmentDetector`, `CounterfactualAnalyzer` |
| **`certificate`** | Proof-carrying certificate construction and verification | `CertificateBuilder`, `ProofChecker`, `CertMerkleTree` |
| **`cli`** | Command-line interface | `CollusionProofCli`, argument parsing, output formatting |

### Crate Dependency Graph

```
shared-types ─────────────────────────────────────────────┐
  ├── market-sim                                          │
  ├── game-theory                                         │
  ├── stat-tests                                          │
  ├── collusion-core ← market-sim, game-theory            │
  ├── counterfactual ← market-sim, game-theory, stat-tests│
  ├── certificate    ← stat-tests                         │
  └── cli            ← all of the above ──────────────────┘
```

---

## Mathematical Foundations

The theoretical framework rests on eight contributions (M1–M8) plus a proven converse (C3'):

### M1 — Composite Hypothesis Test over Game-Algorithm Pairs

The competitive null is parameterized by (demand system, learning algorithm tuple):

```
H₀ = { P_{D,A} : D ∈ D_L, A ∈ A_ind }
```

Organized into a **tiered null hierarchy** providing a power–generality tradeoff:

| Tier | Demand Class | Algorithm Class | Power |
|------|-------------|----------------|-------|
| H₀-narrow | Linear | Q-learning | Highest |
| H₀-medium | Parametric | No-regret learners | Medium |
| H₀-broad | L-Lipschitz | Independent learners | Broadest |

### M2 — Black-Box Deviation Oracle

Adaptive (ε, α)-correct deviation bound certificates with query complexity
O(n · polylog(|P_δ|) · log(n/α) / ε²), logarithmic in grid size.

### M3 — Punishment Detection via Controlled Perturbation

First provably powerful punishment test: J = O(σ² log(1/β) / Δ_P²) injections suffice.
Permutation framework provides exact distribution-free p-values.

### M5 — Certified Collusion Premium

End-to-end error propagation: `CP = δ ± ε_CP`, composing demand estimation error,
NE approximation tolerance, and finite-sample averaging.

### C3' — Folk Theorem Converse for Bounded-Recall Automata

Sustained supra-competitive pricing by deterministic bounded-recall automata with at
most M states **necessarily** produces detectable punishment responses within M rounds.

### M8 — Impossibility Theorem

No detection scheme can identify collusion by unrestricted-memory strategies at any
finite horizon — proving bounded recall is *necessary* for any detection framework.

### Key Soundness Guarantee

> **Theorem.** For any α ∈ (0,1), any demand system D, and any tuple of independent
> learning algorithms A:
>
> Pr[ CollusionProof outputs COLLUDING | algorithms are competitive ] ≤ α
>
> *Unconditional — no assumptions on C3, oracle level, or strategy class.*

---

## Oracle Access Tiers

| Property | Layer 0 — Passive | Layer 1 — Checkpoint | Layer 2 — Full Rewind |
|----------|------------------|---------------------|----------------------|
| **Access** | Price trajectories only | Periodic state snapshots | Arbitrary history restart |
| **Tests** | M1, M7, correlation | M2 deviation oracle | M3 punishment detection |
| **Collusion Premium** | Point estimate | Partial bounds (wider CI) | Tight certified bounds |
| **Soundness** | Unconditional | Unconditional | Unconditional |
| **Completeness** | Weaker | Moderate | Strongest (bounded-recall) |
| **Regulatory use** | Initial screening | Cooperative audit | Full sandbox investigation |

---

## Examples

### Run the Example Scripts

```bash
# Basic collusion detection (Layer 0)
./examples/basic_detection.sh

# Analyze external CSV price data
./examples/passive_analysis.sh

# Full Layer 2 sandbox audit
./examples/sandbox_audit.sh
```

### Analyze Your Own Price Data

Prepare a CSV with columns `round, firm_0_price, firm_1_price, ...`:

```csv
round,firm_0_price,firm_1_price
1,45.20,44.80
2,46.10,45.90
3,47.50,47.20
```

Then run:

```bash
collusion-proof analyze \
  --input your_prices.csv \
  --alpha 0.05 \
  --output your_analysis.json
```

### Custom Scenario via TOML

```bash
collusion-proof run --config examples/custom_scenario.toml --output-dir result_output/
```

---

## Benchmarks

### Run Benchmarks

```bash
# Quick smoke test (< 30 minutes)
cargo run --release --bin collusion-proof -- evaluate --mode smoke

# Standard evaluation (~4 days, 8 cores)
cargo run --release --bin collusion-proof -- evaluate \
  --mode standard --jobs 8

# Full evaluation (~20 days, 8 cores)
cargo run --release --bin collusion-proof -- evaluate --mode full --jobs 8
```

### Performance Summary (20-Game Synthetic Benchmark)

| Method | Accuracy | FPR | Avg Time | Proof Certs |
|--------|----------|-----|----------|-------------|
| **ColluCert** | **80%** | **0.0** | **1.04 ms** | **Yes (368±212 bits)** |
| QRE-Logit | 70% | 0.0 | 0.20 ms | No |
| Nash-Eq | 35% | 0.0 | 0.005 ms | No |
| Brute Force | 25% | 0.0 | 0.032 ms | No |

### Detection Breakdown by Game Type

| Game Type | Games | ColluCert | QRE | Nash |
|-----------|-------|-----------|-----|------|
| Prisoner's Dilemma | 4 | 100% | 50% | 50% |
| Bertrand Competition | 5 | 100% | 100% | 80% |
| Cournot Oligopoly | 5 | 80% | 80% | 20% |
| Repeated Auction | 3 | 100% | 100% | 0% |
| Asymmetric Costs | 3 | 0% | 0% | 0% |

> **Note:** All benchmark games are synthetic with fixed payoff structures.
> The 3 asymmetric-cost games are not detected by any method because coordination
> is a Nash equilibrium in those games (not supra-competitive collusion).

### Throughput

| Configuration | Rounds/sec |
|--------------|-----------|
| Bertrand 2-player (1 thread) | 142,000 |
| Bertrand 2-player (8 cores) | 890,000 |
| Cournot 2-player (1 thread) | 128,000 |

### Certificate Verification

| Size | Proof Steps | Time |
|------|-------------|------|
| Small (5 tests) | ~200 | 12 ms |
| Medium (15 tests) | ~850 | 48 ms |
| Large (30 tests) | ~2,100 | 127 ms |

---

## Configuration

CollusionProof uses TOML configuration files:

```toml
[market]
model = "bertrand"          # bertrand | cournot
players = 2
demand_type = "linear"      # linear | ces | logit
demand_intercept = 100.0
demand_slope = 1.0
marginal_cost = 10.0

[algorithm]
type = "q-learning"         # q-learning | grim-trigger | dqn | tit-for-tat | bandit | nash | myopic
learning_rate = 0.15
discount_factor = 0.95
exploration_rate = 0.1

[simulation]
rounds = 100000
seeds = 5

[analysis]
significance = 0.05
null_tiers = ["narrow", "medium", "broad"]
bootstrap_samples = 10000

[oracle]
level = 0                   # 0 = passive, 1 = checkpoint, 2 = rewind
checkpoint_interval = 1000

[certificate]
enabled = true
rational_verification = true
merkle_integrity = true

[output]
format = "json"             # json | text | table
verbose = false
```

---

## File Formats

### JSON Certificate Bundle

The primary output is a self-contained JSON evidence bundle:

```json
{
  "version": "0.1.0",
  "verdict": "COLLUDING",
  "collusion_premium": {
    "point_estimate": 0.347,
    "confidence_interval": [0.323, 0.371],
    "confidence_level": 0.95
  },
  "null_rejections": {
    "narrow": { "rejected": true, "p_value": 0.0008 },
    "medium": { "rejected": true, "p_value": 0.0031 },
    "broad":  { "rejected": true, "p_value": 0.0124 }
  },
  "proof": {
    "steps": 847,
    "axioms_used": ["supra_competitive", "correlation_excess", "punishment_response"],
    "verified": true
  },
  "merkle_root": "7a3f...e1c2",
  "metadata": {
    "market": "bertrand",
    "players": 2,
    "rounds": 100000,
    "oracle_level": 0,
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

### Protobuf Evidence Bundles

For JSON output, use the `--format json` global flag:

```bash
collusion-proof --format json run --scenario bertrand_qlearning_2p --output-dir result_output/
collusion-proof verify --certificate result_output/certificate.json
```

Protocol buffer definitions are in `implementation/crates/certificate/proto/`.

---

## Contributing

We welcome contributions at the intersection of formal verification, game theory,
and competition law.

### Getting Started

```bash
git clone https://github.com/collusion-proof/collusion-proof.git
cd collusion-proof
cargo test                    # run all tests
cargo fmt                     # format code
cargo clippy --all-targets    # lint
```

### Contribution Areas

- **Market models** — Posted-price, double auction, combinatorial markets
- **Algorithms** — PPO, A3C, multi-agent RL variants
- **Theory** — Progress on Conjecture C3 for stochastic strategies
- **Benchmarks** — Additional scenarios, adversarial red-teaming
- **Performance** — SIMD optimization, GPU-accelerated simulation

### Code Standards

- Proof-checker changes require review by **two** maintainers
- Statistical tests must include power analysis and Type-I error validation
- New algorithms must implement the `PricingAlgorithm` trait
- All public APIs require doc comments with examples

### Pull Request Process

1. Fork → branch (`git checkout -b feature/my-change`)
2. Implement with tests (`cargo test --release`)
3. Run smoke benchmark (`collusion-proof evaluate --mode smoke`)
4. Open PR with clear description of what and why

---

## Citation

```bibtex
@inproceedings{collusionproof2025,
  title     = {{CollusionProof}: Proof-Carrying Collusion Certificates via
               Compositional Statistical Testing and Counterfactual Deviation
               Analysis for Black-Box Algorithmic Pricing Markets},
  author    = {Anonymous},
  booktitle = {Proceedings of the 26th ACM Conference on Economics and
               Computation (EC '25)},
  year      = {2025},
  publisher = {ACM},
  note      = {Under review}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for the full text.

Copyright © 2025 CollusionProof Team.

---

## References

1. Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial
   intelligence, algorithmic pricing, and collusion. *American Economic Review*, 110(10),
   3267–3297.

2. Asker, J., Fershtman, C., & Pakes, A. (2022). Artificial intelligence, algorithm design,
   and pricing. *AEA Papers and Proceedings*, 112, 452–456.

3. Necula, G. C. (1997). Proof-carrying code. *Proceedings of the 24th ACM
   SIGPLAN-SIGACT Symposium on Principles of Programming Languages*, 106–119.

4. Abreu, D. (1988). On the theory of infinitely repeated games with discounting.
   *Econometrica*, 56(2), 383–396.

5. Harrington, J. E. (2008). Detecting cartels. In *Handbook of Antitrust Economics*,
   213–258. MIT Press.

6. Romano, J. P., & Wolf, M. (2005). Exact and approximate stepdown methods for
   multiple hypothesis testing. *Journal of the American Statistical Association*,
   100(469), 94–108.

7. Andrews, D. W. K., & Shi, X. (2013). Inference based on conditional moment
   inequalities. *Econometrica*, 81(2), 609–666.

8. McConnell, R. M., Mehlhorn, K., Näher, S., & Schweitzer, P. (2011). Certifying
   algorithms. *Computer Science Review*, 5(2), 119–161.

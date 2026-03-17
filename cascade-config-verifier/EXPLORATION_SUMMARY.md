# CascadeVerify Project Exploration Report

## PROJECT OVERVIEW

**CascadeVerify** is a static verification tool for detecting **retry-amplification cascades** and **timeout chain violations** in microservice configurations (Kubernetes, Istio, Envoy, Helm).

**Key Claims** (from groundings.json):
- Bounded cascade reachability in RTIG model is NP-complete (C1)
- Retry-timeout networks without circuit breakers are monotone (C2)
- Minimal unsat cores correspond to minimal failure sets (C3)
- MaxSAT repair minimizes parameter deviation (C4)
- Optimal repair synthesis is Σ₂ᴾ-complete (C6)

---

## 1. RUST IMPLEMENTATION STRUCTURE

### Workspace: `implementation/Cargo.toml`

**Edition:** 2021  
**10 Crates** (workspace members):

```
crates/
├── cascade-types          # Type definitions, domain model (RTIG, policies, services)
├── cascade-graph          # Graph construction from manifests (RtigGraph API)
├── cascade-config         # YAML/JSON config parsing (Kubernetes, Istio, Envoy)
├── cascade-bmc            # Bounded Model Checking encoder (SMT constraints)
├── cascade-maxsat         # MaxSAT repair synthesis solver
├── cascade-repair         # Repair synthesis orchestration
├── cascade-analysis       # Tier 1 & Tier 2 analysis pipeline
├── cascade-service        # gRPC/REST service interface
├── cascade-verify         # Core verification algorithms
└── cascade-cli            # Main CLI binary + command handlers
```

### CLI Entry Point: `cascade-cli/src/main.rs`

**Binary Name:** `cascade-verify`  
**Exit Codes:**
- `0`: Success (no cascade findings)
- `1`: Cascade findings detected
- `2`: Runtime/input error

**Logging:**
- Via `tracing` + `tracing-subscriber`
- `-v` → DEBUG level
- `RUST_LOG` env var → custom filters
- Default: `cascade_cli=info,warn`

### CLI Commands (7 Subcommands)

From `cascade-cli/src/commands.rs`:

1. **`verify`** - Analyze config files for cascade risks
2. **`repair`** - Synthesize repairs for detected cascades
3. **`check`** - Quick Tier 1 only (CI/CD gate, fast)
4. **`analyze`** - Deep analysis with Tier 2 BMC
5. **`diff`** - Delta analysis between two config sets
6. **`report`** - Generate report from cached results
7. **`benchmark`** - Run benchmarks on synthetic topologies

### Shared CLI Arguments

- `--verbose` / `-v` – Force DEBUG logging
- `--quiet` / `-q` – Suppress all output except errors
- `--config` – Override config file location (via `CASCADE_CONFIG` env)
- `--no-color` – Disable color output (via `NO_COLOR` env)
- `--format` – Output format (default: "table")
- `--output-file` / `-o` – Write to file instead of stdout

### Input Format

**Accepted Formats:** Kubernetes YAML manifests (multi-document streams)

**Resource Types Recognized:**
- `kind: Deployment` – Service definitions with capacity/load annotations
- `kind: Service` – Kubernetes service definitions
- `kind: VirtualService` – Istio routing policies
- `kind: DestinationRule` – Istio traffic policies
- `kind: Gateway` – Istio gateway definitions

**Annotations Used:** (from examples)
- `cascade-verify/capacity: "1000"` – Service capacity (default load units)
- `cascade-verify/baseline-load: "100"` – Baseline request load
- `cascade-verify/deadline-ms: "10000"` – Service deadline (optional)

**Load Format:** Plaintext YAML files (recursively searched in directories)

---

## 2. WHAT DOES IT VERIFY?

### Core Concepts: RTIG (Retry-Timeout-Interaction Graph)

**Cascade Risks Detected:**

1. **Retry Amplification**
   - Example: A→B→C with 3 retries each = (1+3)³ = 64× amplification
   - Detected when: product of (1 + retries) along a path exceeds threshold
   
2. **Timeout Chain Violations**
   - Child timeout exceeds parent per-try timeout budget
   - Worst-case latency: Σ(retries × per-try-timeout) + Σ(base-latencies)
   - Detected when: accumulated timeout > service deadline

3. **Fan-in Overload**
   - Multiple retry paths converging on same service
   - Load amplification from convergent retries

### Verification Pipeline (Two Tiers)

**Tier 1: Fast Static Analysis**
- Graph-based monotonicity checking
- Load propagation algorithms
- Time: O(|V|² × d*) where d* = diameter × max_retries
- Used in `check` command for CI/CD gates

**Tier 2: Bounded Model Checking (BMC)**
- SMT constraint generation (via `cascade-bmc` crate)
- Encodes load propagation into logical formulas
- Completeness bound: d* = diameter(G) × max_retries (Corollary 1)
- MinUnsat core enumeration (MARCO algorithm)
- Used in `analyze` and `repair` commands

**Tier 3: MaxSAT Repair Synthesis**
- Generates minimal parameter repairs (via `cascade-maxsat`)
- Hard constraints: cascade-freedom for each discovered MUS
- Soft constraints: minimize parameter deviation
- Optimization: weighted partial MaxSAT

---

## 3. EXISTING BENCHMARKS

### Location: `benchmarks/`

**Files:**
```
benchmarks/
├── README.md
├── deep_cascade_profiler.py      # Python profiler script
├── deep_cascade_results.json     # Scalability profiling results
├── comparison/
│   ├── istio-analyze-baseline.yaml
│   └── kubelinter-baseline.yaml
└── topologies/
    ├── chain-10.yaml, chain-50.yaml, ...
    ├── star-100.yaml, ...
    └── mesh-500.yaml, ...
```

### Benchmark Configurations

#### a) Deep Cascade Profiler (`deep_cascade_profiler.py`)

**Purpose:** Measure verification time scaling w.r.t. topology size

**Topology Parameters:**
- **Depths:** [5, 10, 15, 20, 30, 50]
- **Widths:** [2, 5, 10]
- **Topologies:** "chain", "tree", "mesh"
- **Retries:** 3 per edge
- **Per-try timeout:** 5s
- **Overall timeout:** 20s
- **Base capacity:** 1000 units
- **Base load:** 100 units
- **Timeout per run:** 600s
- **Runs per config:** 3 (for variance)

**Output Format:** `deep_cascade_results.json`

**Measured Phases:**
```json
{
  "phase_breakdown": {
    "graph_build_s": 0.0057,
    "constraint_gen_s": 13.5941,      // Tier 2 bottleneck
    "smt_solving_s": 3.9762,
    "propagation_s": 1.484,
    "total_s": 19.06
  },
  "memory_mb": 75.0,
  "bottleneck": "constraint_generation",
  "timed_out": false
}
```

**Key Observations from Results:**
- Chain-depth-50 (100 services, 98 edges): 19.06s total
  - Constraint generation: 13.59s (70% of time) ← bottleneck
  - SMT solving: 3.98s (21%)
  - Propagation: 1.48s (8%)
- Memory scales ~0.75 MB per service
- Constraint generation is **cubic** in topology size

#### b) Topology Benchmark Files (`topologies/`)

**Format:** Kubernetes YAML manifests (same as examples)

**Example:** `chain-10.yaml`
```
- 10 services: svc-0 → svc-1 → ... → svc-9
- Each hop: 3 retries, 5s per-try, 20s overall
- Retry amplification: 4^9 = 262,144×
```

**Example:** `star-100.yaml`
```
- 100 spokes radiating from hub
- Hub calls each spoke independently
- Fan-out amplification risk
```

**Example:** `mesh-500.yaml`
```
- 500 services, ~3 edges per node
- Deterministic pseudo-random edges
- Tests mesh connectivity scalability
```

#### c) Comparison Baselines

**Files:**
- `istio-analyze-baseline.yaml` – Config with issues `istioctl analyze` catches **and** CascadeVerify catches
- `kubelinter-baseline.yaml` – Config with issues KubeLinter detects

**Purpose:** Demonstrate complementary analysis (istioctl finds config hygiene, CascadeVerify finds cascade semantics)

**Example Comparison Issues:**
- istioctl finds: VirtualService → non-existent host, missing DestinationRules
- CascadeVerify finds: retry amplification, timeout violations, fan-in overload

### Rust Benchmarks: `implementation/benches/`

**Location:** `crates/cascade-analysis/benches/topology_scaling.rs`

**Framework:** Criterion (Rust benchmarking with statistical rigor, HTML reports)

**Benchmark Groups:**

```rust
// 1. RTIG Construction (3 topologies × 4 sizes)
bench_rtig_construction_chain()    // Sizes: 10, 50, 100, 500
bench_rtig_construction_star()
bench_rtig_construction_mesh()

// 2. Tier 1 Analysis (all topology types)
bench_tier1_analysis()             // Fast monotonicity checking

// 3. Path Amplification Detection
bench_path_amplification()

// 4. Topology Building + Stats
bench_topology_building()
```

**Topology Generators** (deterministic, no randomness):

```rust
// Chain: svc-0 → svc-1 → ... → svc-{n-1}
build_chain(n)
  Policy: 3 retries, 5s per-try, 20s total timeout

// Star: hub → spoke_0, spoke_1, ..., spoke_{n-1}
build_star(n)
  Policy: 2 retries, 3s per-try, 15s total timeout

// Mesh: n nodes, ~3 edges per node (deterministic pseudo-random)
build_mesh(n)
  Policy: 2 retries, 4s per-try, 15s total timeout
  Edge formula: j = (i * 7 + k * 13 + 3) % n (no rand crate)
```

**Sizes Benchmarked:** [10, 50, 100, 500]

---

## 4. TOOL PAPER (tool_paper.tex) – Evaluation Claims

### Paper Structure

- **§2 Formal Model** – RTIG definition, load propagation semantics
- **§3 Monotonicity & Pruning** – Pruning strategies for non-monotone cases
- **§4 BMC Encoding** – SMT constraint generation (Σ₁ logic)
- **§4.1 MUS Enumeration** – MARCO algorithm for minimal failure sets
- **§5 MaxSAT Repair** – Synthesis algorithms
- **§6 Implementation** – ~60,000 LOC across 10 crates + Table (crates summary)
- **§7 Evaluation** – 6 experiments (details pending)

### Experiment Claims (Status: **PENDING EMPIRICAL VALIDATION**)

#### Experiment 1: Detection Effectiveness

**11 Benchmark Configurations:**

| Category | Count | Examples |
|----------|-------|----------|
| Real open-source | 5 | Bookinfo (4 svcs), Online Boutique (11), Sock Shop (8), Hipster Shop (10), Train Ticket (41) |
| Semi-synthetic | 3 | Synth-Chain-20, Synth-Tree-30, Synth-Mesh-50 |
| Synthetic scale | 3 | Synth-Hub-30, Synth-Scale-100, Synth-Scale-1000 |

**Table: tab:detection** (placeholder values)
```
Benchmark         | Svcs |  Bugs | Precision | Recall | Time
------------------|------|-------|-----------|--------|------
Bookinfo          |  4   |   3   |   --%    |  --%   |  --
Online Boutique   | 11   |  12   |   --%    |  --%   |  --
Sock Shop         |  8   |   7   |   --%    |  --%   |  --
Hipster Shop      | 10   |   9   |   --%    |  --%   |  --
Train Ticket      | 41   |  34   |   --%    |  --%   |  --
Synth-Chain-20    | 20   |  15   |   --%    |  --%   |  --
Synth-Tree-30     | 30   |  22   |   --%    |  --%   |  --
Synth-Mesh-50     | 50   |  38   |   --%    |  --%   |  --
Synth-Hub-30      | 30   |  25   |   --%    |  --%   |  --
Synth-Scale-100   | 100  |  67   |   --%    |  --%   |  --
Synth-Scale-1000  | 1000 |  --   |   --     |  --    |  --
```
**Total:** 215 cascade bugs across benchmarks

#### Experiment 2: Comparison with Existing Tools

Tools compared:
- `istioctl analyze` (Istio config validator)
- KubeLinter (Kubernetes linter)
- Custom Kubernetes test cluster runner (2-hour runtime)

Expected result: CascadeVerify detects 215 cascade bugs that linters miss.

#### Experiment 3: Repair Quality

Expected: MaxSAT repairs minimize parameter changes, full re-verification soundness.

#### Experiment 4: Scalability (tab:scale)

**Test Sizes (single-thread Apple M2):** 10, 30, 50, 100, 500, 1000 services

**Phases Measured:**
- Tier 1 (fast check)
- Tier 2 (BMC + constraint generation)
- Enum (MUS enumeration)
- Repair (MaxSAT synthesis)
- Total end-to-end

**Design:** Tier 1 → large scales for CI/CD gates  
Tier 2 + Repair → reasonable time for moderate topologies  
1000-service case: compositional decomposition (strongly-connected components analyzed independently)

#### Experiment 5: Service Mesh Validator Comparison

Compares cascade detection vs. Envoy/Zuul route validation on Listener resources.

#### Experiment 6: Production-Trace-Derived Benchmarks

(Details in paper, not yet populated in results)

---

## 5. INPUT/EXAMPLE CONFIGS

### Location: `examples/`

**7 Example Directories:**

```
examples/
├── combined/
├── e-commerce-platform/
├── fan-in-storm/
├── istio-bookinfo/          ← Real Istio demo
├── retry-amplification/
├── service-mesh-policy/
└── timeout-chain/
```

### Detailed Example: Istio BookInfo

**File:** `examples/istio-bookinfo/manifests.yaml`

**Topology:**
```
productpage → reviews → ratings
productpage → details
```

**Cascade Risks Embedded:**

1. **Retry Amplification:** productpage→reviews→ratings
   - (1 + 3 retries) × (1 + 2 retries) = 12× amplification

2. **Timeout Chain Violation:**
   - productpage deadline: 10s
   - reviews: 3×3s = 9s worst-case
   - ratings: 2×2s = 4s worst-case
   - **Total: 13s > 10s deadline** ✗

3. **Tight Timeout on Details:**
   - 2s timeout on low-capacity service → likely to fail under load

**Annotations in Manifest:**
```yaml
metadata:
  annotations:
    cascade-verify/capacity: "500"           # Service capacity
    cascade-verify/baseline-load: "200"      # Normal load
    cascade-verify/deadline-ms: "10000"      # SLA deadline
```

**VirtualService Policies:**
```yaml
# productpage → reviews: 3 retries, 3s per-try, 15s overall
retries:
  attempts: 3
  perTryTimeout: 3s
  retryOn: 5xx,reset,connect-failure
timeout: 15s

# reviews → ratings: 2 retries, 2s per-try, 8s overall
retries:
  attempts: 2
  perTryTimeout: 2s
  retryOn: 5xx
timeout: 8s
```

---

## 6. GROUNDINGS.JSON STRUCTURE

**File:** `groundings.json` (claim-evidence mapping, NLP-compatible format)

**Top-level keys:**
```json
{
  "version": "2.0",
  "project": "CascadeVerify",
  "description": "Claim-evidence mapping for...",
  "claims": [ /* array of claim objects */ ]
}
```

**Claim Object Schema:**
```json
{
  "id": "C1",
  "statement": "Bounded cascade reachability in RTIG is NP-complete.",
  "type": "theoretical",  // "theoretical" or "empirical"
  "evidence": "Full proof explanation...",
  "references": [
    "file.md §section",
    "tool_paper.tex §page, Proposition N",
    "Published paper citation"
  ],
  "code_references": [
    "implementation/crates/cascade-bmc/src/encoder.rs"
  ],
  "verification": "formal_proof",  // or "empirical_validation"
  "confidence": "high"  // "high", "medium", "low"
}
```

**Claims in File (6+ theoretical claims):**

- **C1:** NP-completeness of cascade reachability
- **C2:** Monotonicity of retry-timeout networks (CB-free)
- **C3:** MinUnsat cores ↔ minimal failure sets correspondence
- **C4:** MaxSAT repair optimality
- **C5:** BMC completeness bound d* = diameter × max_retries
- **C6:** Repair synthesis is Σ₂ᴾ-complete

---

## 7. KEY INSIGHTS FOR REAL BENCHMARKS

### What to Measure

1. **Tier 1 (fast check) scalability:**
   - Input: Chain/Star/Mesh topologies 10–1000 services
   - Measure: Time to complete, memory usage
   - Expected: Sub-second for typical CI/CD gates

2. **Tier 2 (BMC) bottleneck:**
   - Constraint generation dominates (70% of time in profiler)
   - SMT solving next (20%)
   - Need compiler optimizations or constraint pruning

3. **Repair synthesis cost:**
   - MUS enumeration can be exponential in worst case
   - Time: enumerate MUS → MaxSAT solve → result
   - Practical limits: topologies < 200 services for interactive repair

4. **Detection effectiveness:**
   - Precision/recall vs. istioctl, KubeLinter
   - Use 11 benchmarks + 215 bug corpus
   - Measure false positives (over-conservative), false negatives (missed bugs)

5. **Real-world applicability:**
   - Time per Kubernetes manifest directory
   - Integration with CI/CD (< 5s gates)
   - Memory footprint (cloud/container-native)

### Benchmark Format Decisions

- **Config Input:** Standard Kubernetes/Istio YAML (widely adoptable)
- **Topology Size Scale:** 10→1000 services (cover CI/CD to enterprise)
- **Phases to Instrument:** graph_build, constraint_gen, smt_solving, propagation, repair
- **Comparison Baseline:** istioctl analyze (widely deployed)
- **Report Format:** JSON (deep_cascade_results.json) + HTML (Criterion reports)

---

## SUMMARY TABLE

| Aspect | Details |
|--------|---------|
| **Language** | Rust (2021 edition) |
| **CLI Binary** | `cascade-verify` (clap-derived) |
| **Core Algorithm** | Bounded Model Checking + MaxSAT repair |
| **Verification Tiers** | Tier 1 (fast), Tier 2 (complete), Tier 3 (repair) |
| **Input Format** | Kubernetes/Istio YAML manifests |
| **Key Metrics** | Retry amplification, timeout violations, fan-in overload |
| **Benchmark Scale** | 10→1000 services (chain, star, mesh topologies) |
| **Existing Benchmarks** | 11 configs (5 real + 3 semi-syn + 3 syn), 215 bugs |
| **Profiler** | Python script (`deep_cascade_profiler.py`) → JSON results |
| **Rust Benchmark Framework** | Criterion (statistical rigor) |
| **Current Status** | Empirical validation **PENDING** |
| **Crates** | 10 specialized crates (types, graph, config, bmc, maxsat, repair, analysis, service, verify, cli) |


# Architecture

This document describes the architecture of Certified Leakage Contracts,
focusing on the reduced product domain, the analysis pipeline, and the
composition and certification systems.

## Overview

The system takes an x86-64 ELF binary as input and produces per-function
leakage contracts with quantitative bounds in bits.  These contracts compose
across function boundaries and are bundled into machine-checkable certificates.

```
  ELF Binary
      │
      ▼
 ┌──────────────┐
 │ leak-transform│  Binary lifting, IR construction, CFG building
 └──────┬───────┘
        │  Analysis IR
        ▼
 ┌──────────────┐
 │ leak-analysis │  Fixpoint computation over D_spec ⊗ D_cache ⊗ D_quant
 └──────┬───────┘
        │  Per-function analysis results
        ▼
 ┌──────────────┐
 │ leak-contract │  Contract extraction, composition, validation
 └──────┬───────┘
        │  Composed contracts
        ▼
 ┌──────────────┐
 │ leak-certify  │  Certificate generation with witnesses
 └──────┬───────┘
        │  Machine-checkable certificates
        ▼
    Output (JSON)
```

## The Reduced Product Domain

The core innovation is a **reduced product** of three abstract domains that
exchange information through a reduction operator.

### D_spec — Speculative Reachability

**Purpose:** Track which program points are reachable under bounded
misspeculation (Spectre-PHT model).

**Abstraction:** Each program point is tagged with a speculative reachability
set: the set of speculation windows (sequences of transiently executed
instructions) that can reach it.

**Lattice:** Power set of speculation paths, ordered by inclusion.
- ⊥ = unreachable (empty set)
- ⊤ = reachable under all speculation scenarios

**Key types:**
- `SpecState` — speculative reachability at a program point
- `SpecTag` — individual speculation path identifier
- `SpecWindow` — bounded speculation window (≤ W μops)
- `MisspecKind` — type of misspeculation (branch direction, target)

**Transfer function (`SpecTransfer`):**
At each branch instruction, the transfer function forks the analysis into
architectural (correct) and speculative (mispredicted) successors.  The
speculative path is tracked for up to W μops, after which it is killed
(the processor would have resolved the branch by then).

### D_cache — Tainted Abstract Cache State

**Purpose:** Model the LRU cache state with secret-dependence annotations
on each cache line.

**Abstraction:** An abstract cache set maps each way to an `AbstractAge`
(the line's position in the LRU ordering) and a `TaintAnnotation` (whether
the line's presence depends on secret data).

**Lattice:** Product of abstract age lattice × taint lattice per cache way.
- ⊥ = empty cache (no lines loaded)
- ⊤ = all possible cache configurations

**Key types:**
- `CacheDomain` — complete abstract cache state (all sets)
- `AbstractCacheSet` — one cache set (ways × age × taint)
- `AbstractCacheWay` — one cache way
- `CacheLineState` — age + taint for a single line
- `TaintAnnotation` — {untainted, tainted(source), unknown}
- `TaintSource` — origin of the secret dependence
- `AbstractAge` — abstract LRU position {definite(n), range(lo, hi), unknown}

**Transfer function (`CacheTransfer`):**
- **Load/Store:** Model the LRU update.  If the accessed address depends on
  secret data (determined by taint tracking), the resulting cache line is
  marked `tainted`.
- **Eviction:** When a new line is loaded into a full set, the LRU victim
  is evicted.  The abstract age ordering determines which lines may be
  evicted.

### D_quant — Quantitative Channel Capacity

**Purpose:** Count the number of distinguishable cache configurations to
bound the information leakage in bits.

**Abstraction:** At each program point, maintain a set of distinguishable
cache observations.  The leakage bound is log₂ of the count.

**Lattice:** Ordered by the number of distinguishable configurations.
- ⊥ = 1 configuration (zero leakage)
- ⊤ = 2^n configurations (maximum leakage for n-bit cache)

**Key types:**
- `QuantState` — quantitative state at a program point
- `LeakageBits` — bound in bits (log₂ of distinguishable states)
- `SetLeakage` — leakage attributed to a specific cache set

**Transfer function (`QuantTransfer`):**
After each memory access, if the cache domain indicates the access is
tainted, the counting domain partitions the observation space.  Two
cache configurations are distinguishable if and only if they differ on
at least one observable cache set.

### The Reduction Operator ρ

The reduction operator ρ : D → D exchanges information between the three
domains to improve precision beyond what each domain achieves alone.

```
ρ(spec, cache, quant) = (spec', cache', quant')
```

**Reduction rules:**

1. **Spec → Cache (unreachability pruning):**
   If `spec` determines a program point is unreachable (both architecturally
   and speculatively), set `cache = ⊥` at that point.

2. **Cache → Quant (zero-leakage detection):**
   If `cache` shows zero tainted lines in all cache sets, set `quant` to
   zero bits of leakage.

3. **Spec → Quant (speculative amplification):**
   If a speculatively reachable point introduces tainted cache accesses not
   present in the architectural execution, `quant` must account for the
   additional distinguishable observations.

4. **Quant → Spec (observation merging):**
   If the counting domain proves two speculative paths produce identical
   cache observations, they can be merged in `spec`, reducing the
   speculative reachability set.

5. **Cache → Spec (taint-guided pruning):**
   If all cache accesses on a speculative path are untainted, the path
   does not contribute to leakage and can be deprioritized.

The reduction is applied iteratively until a fixpoint is reached (typically
2–3 iterations suffice).

## Fixpoint Computation

The analysis uses a **worklist-based forward fixpoint algorithm**:

```
Input: CFG, initial abstract state, transfer functions, reduction operator
Output: Per-block abstract state at fixpoint

1. Initialize worklist W with the entry block
2. Initialize state map S: block → D with ⊥ for all blocks
3. Set S[entry] = initial state
4. While W is not empty:
   a. Remove block b from W
   b. For each instruction i in b:
      - Apply CombinedTransfer(spec, cache, quant) to get new state
      - Apply reduction ρ
   c. For each successor b' of b:
      - Compute joined = S[b'] ⊔ transfer(S[b], edge(b, b'))
      - If joined ≠ S[b']:
        - If widening trigger: joined = S[b'] ∇ joined
        - S[b'] = joined
        - Add b' to W
5. Optional: Apply narrowing passes
6. Return S
```

### Widening Strategies

- **Standard widening:** Apply widening operator ∇ at every loop header
  from the first iteration.
- **Delayed widening:** Allow `k` iterations (default k=3) before engaging
  widening, often yielding better precision.
- **Threshold widening:** Widen toward a finite set of thresholds rather than
  jumping to ⊤, preserving more information.

## Contract Extraction and Composition

### Per-Function Contracts

After fixpoint computation, a leakage contract is extracted for each function:

```
Contract(f) = (τ_f, B_f, pre_f, post_f)
```

- `τ_f` — abstract cache transformer: maps input cache state to output cache state
- `B_f` — leakage bound: maps input cache state to bits of leakage
- `pre_f` — precondition on the input cache state
- `post_f` — postcondition on the output cache state

### Composition Rules

**Sequential composition** (f followed by g):
```
τ_{f;g}(s) = τ_g(τ_f(s))
B_{f;g}(s) = B_f(s) + B_g(τ_f(s))
```

**Parallel composition** (f and g on independent data):
```
τ_{f||g}(s) = τ_f(s) ⊔ τ_g(s)
B_{f||g}(s) = B_f(s) + B_g(s)
```

**Conditional composition** (if c then f else g):
```
τ_{if}(s) = τ_f(s) ⊔ τ_g(s)
B_{if}(s) = max(B_f(s), B_g(s)) + B_branch(s)
```

**Loop composition** (loop { f } with iteration bound n):
```
B_{loop}(s) = n · B_f(s*)     where s* = fixpoint of τ_f
```

### Validation

Composed contracts are validated for:
- **Soundness:** The composed bound is a valid upper bound.
- **Monotonicity:** Larger input cache states yield larger bounds.
- **Independence:** Observations of composed functions are independent
  given intermediate state (required for additive composition).

## Certificate Generation

Certificates provide independently verifiable evidence that the claimed
leakage bounds are correct.

### Certificate Structure

```
CertificateChain
├── FunctionCertificate (for each function f)
│   ├── Claim: "B_f(s) ≤ k bits"
│   ├── FixpointWitness: abstract state at each block
│   ├── CountingWitness: distinguishable configuration count
│   └── ReductionWitness: reduction steps applied
├── CompositionCertificate (for each composition step)
│   ├── Claim: "B_{f;g}(s) ≤ B_f(s) + B_g(τ_f(s))"
│   └── CompositionWitness: independence proof
└── LibraryCertificate (top-level)
    ├── Claim: "Library leakage ≤ K bits"
    └── Hash chain linking all sub-certificates
```

### Witness Checking

An independent checker verifies each certificate by:

1. **Fixpoint check:** Re-apply transfer functions to the witness state;
   verify it is indeed a fixpoint (or post-fixpoint).
2. **Counting check:** Enumerate the distinguishable configurations in the
   witness and verify the count matches the claimed bound.
3. **Reduction check:** Verify that the reduction operator was correctly
   applied at each step.
4. **Composition check:** Verify the independence condition and the additive
   bound calculation.
5. **Chain check:** Verify hash links between certificates.

## Crate Responsibilities

| Crate              | Responsibility                                                   |
|--------------------|------------------------------------------------------------------|
| `shared-types`     | CPU, memory, and program types used by all crates                |
| `leak-types`       | Leakage-domain-specific type definitions                         |
| `leak-abstract`    | Generic abstract interpretation machinery                        |
| `leak-analysis`    | The three domains, reduction, and fixpoint engine                |
| `leak-contract`    | Contract data structures, composition, and validation            |
| `leak-quantify`    | Entropy, channels, counting, and bound computation               |
| `leak-smt`         | SMT encoding and solver integration                              |
| `leak-transform`   | Binary lifting, IR, normalization, and CFG construction          |
| `leak-certify`     | Certificate generation, witness construction, and checking       |
| `leak-eval`        | Benchmarking, comparison, and reporting                          |
| `leak-cli`         | Command-line interface and CI integration                        |

## Data Flow

```
User invokes: leakage-contracts analyze <binary>

1. leak-cli parses arguments, loads config
2. leak-transform:
   a. BinaryAdapter reads ELF, discovers functions
   b. InstructionLifter lifts x86-64 → AnalysisIR
   c. SecurityAnnotator marks secret regions
   d. IRNormalizer applies optimization passes
   e. CfgBuilder constructs per-function CFGs
3. leak-analysis:
   a. AnalysisEngine initializes D_spec ⊗ D_cache ⊗ D_quant
   b. Fixpoint loop with CombinedTransfer + ReductionOperator
   c. Extract per-function AnalysisResult
4. leak-contract:
   a. Extract LeakageContract from each AnalysisResult
   b. Validate contracts (soundness, monotonicity)
5. leak-certify (if --certify):
   a. Generate witnesses from fixpoint traces
   b. Build certificate chain
6. leak-cli formats and writes output
```

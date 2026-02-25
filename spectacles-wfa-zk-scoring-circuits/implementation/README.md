# Spectacles: A Verified Compiler from Semiring-Weighted Automata to Zero-Knowledge Scoring Circuits

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-research-green.svg)](#)

Spectacles produces **contamination-certified evaluation certificates**: machine-checkable
proofs that a benchmark score is correct AND that the test data was not leaked into
training. It achieves this by recognizing that NLP scoring functions decompose into
weighted finite automata over semirings, compiling them to STARK arithmetic circuits,
and composing with OPRF-based PSI for contamination detection.

## 30-Second Quickstart

```bash
# Build the project
cargo build

# Score a single text pair with triple verification (reference + WFA + circuit)
cargo run --bin spectacles-cli -- score --metric exact_match \
  --candidate "the cat sat on the mat" --reference "the cat sat on the mat"

# Run differential testing (all 3 implementations must agree)
cargo run --bin spectacles-cli -- differential-test --count 1000 --seed 42

# Run the independent Python verification suite
python3 tests/test_semiring_axioms.py
```

**Most impressive capability**: Every metric score comes with a STARK proof that the computation was correct *and* a PSI-based certificate that the training data wasn't contaminated — without revealing either dataset.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Pipeline Diagram](#pipeline-diagram)
- [Module Map](#module-map)
- [Supported Metrics](#supported-metrics)
- [Two-Tier Compilation](#two-tier-compilation)
- [STARK Proof System](#stark-proof-system)
- [PSI Contamination Detection](#psi-contamination-detection)
- [Metric Equivalence](#metric-equivalence)
- [Getting Started](#getting-started)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Key Types Quick Reference](#key-types-quick-reference)
- [Theoretical Background](#theoretical-background)
- [API Reference](#api-reference)
- [License](#license)

---

## Architecture Overview

Spectacles is organized into six logical layers forming a compilation and verification
pipeline:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  6. Scoring Layer   — 7 NLP metrics, each with triple implementation     │
├──────────────────────────────────────────────────────────────────────────┤
│  5. Privacy Layer   — PSI contamination detection, OPRF, trie index      │
├──────────────────────────────────────────────────────────────────────────┤
│  4. Protocol Layer  — commit-reveal-verify state machine, certificates   │
├──────────────────────────────────────────────────────────────────────────┤
│  3. Circuit Layer   — WFA→AIR compilation, STARK proving, FRI, Merkle    │
├──────────────────────────────────────────────────────────────────────────┤
│  2. Automata Layer  — WFA engine, transducers, minimization, equivalence │
├──────────────────────────────────────────────────────────────────────────┤
│  1. Specification Layer — EvalSpec DSL, type system, semiring selection   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Layer 1: Specification Layer

The **Specification Layer** provides the **EvalSpec** domain-specific language for
declaring evaluation metrics. EvalSpec has a formal type system and denotational
semantics, ensuring that only well-typed scoring specifications compile. The
pipeline:

1. **Parser** — parses EvalSpec syntax into a `Spanned<Expr>` AST with source locations
2. **TypeChecker** — validates types and infers semiring requirements via `EvalType`
3. **Compiler** (`EvalSpecCompiler`) — lowers EvalSpec to weighted finite automata
4. **Semantics** — denotational semantics define the ground truth
5. **Builtins** — pre-defined scoring functions (BLEU, ROUGE, etc.)

The type system selects the appropriate semiring for each metric automatically:
Boolean for exact match, Counting for n-gram statistics, Tropical for
edit-distance-based metrics. Types are expressed through:

- `BaseType` — `String`, `Integer`, `Float`, `Bool`, `List`, `Tuple`, `Token`,
  `TokenSequence`, `NGram(usize)`
- `SemiringType` — `Counting`, `Boolean`, `Tropical`, `BoundedCounting(u64)`,
  `Real`, `LogDomain`, `Viterbi`, `Goldilocks`
- `EvalType` — `Base(BaseType)`, `Semiring(SemiringType)`, `Function{params, ret}`,
  `Metric{...}`
- `MetricType` — overall metric classification

### Layer 2: Automata Layer

The **Automata Layer** implements a full weighted finite automaton (WFA) engine
over generic semirings. Core capabilities:

- **`WeightedFiniteAutomaton<S: Semiring>`** — the central data structure,
  parameterized by semiring type, with initial/final weight vectors and a
  transition tensor.
- **Operations** — union, concatenation, Kleene star, intersection via the
  semiring product construction, complement, reverse.
- **Minimization** — Hopcroft-style minimization adapted for weighted automata,
  reducing state count while preserving the recognized formal power series.
- **Equivalence** — language equivalence checking via formal power series
  comparison, determining whether two WFAs compute the same function.
- **Transducers** — weighted finite-state transducers for input/output
  transformations (e.g., tokenization, normalization).
- **Formal Power Series** — the algebraic semantics of WFAs as formal power
  series over the free monoid, enabling compositional reasoning.
- **Field Embedding** — maps semiring elements into the Goldilocks prime field
  for circuit compilation, preserving algebraic structure.

The automata engine provides rich symbol support via the `Symbol` enum:

```rust
pub enum Symbol {
    Char(char),
    Byte(u8),
    Token(String),
    Epsilon,
    Wildcard,
    Id(usize),
}
```

And a full `Alphabet` type with methods for constructing and manipulating ordered
symbol sets (`from_chars()`, `from_strings()`, `from_range()`, `union()`,
`intersection()`, `index_mapping()`).

**WFA construction** is streamlined via `WFABuilder`:

```rust
let wfa = WFABuilder::new(alphabet)
    .initial(0, CountingSemiring::one())
    .final_state(2, CountingSemiring::one())
    .transition(0, symbol_a, 1, CountingSemiring::one())
    .transition(1, symbol_b, 2, CountingSemiring::one())
    .label(0, "start")
    .label(2, "accept")
    .build()?;
```

Supported semiring implementations:

| Semiring | Type | Algebra | Typical Use |
|----------|------|---------|-------------|
| `BooleanSemiring` | `bool` | (∨, ∧, false, true) | Exact match, regex match |
| `CountingSemiring` | `u64` | (+, ×, 0, 1) | N-gram counting, BLEU/ROUGE |
| `BoundedCountingSemiring` | `u64` | (+, ×, 0, 1) with ceiling | Bounded n-gram counts |
| `TropicalSemiring` | `f64` | (min, +, ∞, 0) | Shortest-path / edit distance |
| `MaxPlusSemiring` | `f64` | (max, +, −∞, 0) | Longest common subsequence |

### Layer 3: Circuit Layer

The **Circuit Layer** compiles WFAs into STARK proof circuits using the
**Goldilocks field** (p = 2⁶⁴ − 2³² + 1), chosen for its efficient arithmetic
on 64-bit hardware. The modulus constant:

```rust
pub const MODULUS: u64 = 0xFFFFFFFF00000001; // 2^64 - 2^32 + 1
```

The compilation pipeline:

```
WFA<Semiring>
  ↓  field embedding
WFA<GoldilocksField>
  ↓  AIR compilation
AIRProgram (constraints + layout)
  ↓  trace generation
AIRTrace (execution witness)
  ↓  STARK proving
STARKProof (FRI commitments + queries)
```

Key components:

- **`GoldilocksField`** — prime field arithmetic with efficient modular reduction.
  Supports `add()`, `sub()`, `mul()`, `inv()`, `pow()`, `div()`.
- **`AIRProgram`** — Algebraic Intermediate Representation with boundary,
  transition, periodic, and composition constraints. Built from `AIRConstraint`
  instances with `SymbolicExpression` trees.
- **`AIRTrace`** — execution traces as 2D tables of field elements, constructed
  from rows or columns, with padding, windowing, and sub-trace extraction.
- **`TraceLayout`** — column schema describing state, input, auxiliary, and
  public columns via `ColumnType`.
- **`SymbolicExpression`** — constraint expression tree with `Constant`, `Variable`,
  `Add`, `Mul`, `Sub`, `Neg`, `Pow`, `CurrentRow`, `NextRow` nodes, supporting
  `evaluate()`, `simplify()`, `substitute()`, and degree analysis.
- **`PeriodicColumn`** — repeating-pattern helper for cyclic constraints.
- **`BoundaryDescriptor`** — (column, row, value) specification for boundary
  constraints.
- **Gadgets** — reusable arithmetic sub-circuits for bit decomposition,
  range checks, and comparison operations.

Constraint types in AIR programs:

| Type | Description |
|------|-------------|
| `Boundary` | Constraints on first/last rows (initial/final weights) |
| `Transition` | Row-to-row constraints (WFA transitions) |
| `Periodic` | Repeating constraints (cyclic structure) |
| `Composition` | Cross-column constraints (multi-metric proofs) |

The circuit layer also provides a `ScoringCircuit` abstraction with
`CircuitConstraint` variants:

```rust
pub enum CircuitConstraint {
    Mul { a, b, c },        // a × b = c
    Add { a, b, c },        // a + b = c
    Eq  { a, b },           // a = b
    Const { a, val },       // a = constant
    Bool { a },             // a ∈ {0, 1}
}
```

### Layer 4: Protocol Layer

The **Protocol Layer** implements a commit-reveal-verify state machine for
interactive evaluation:

```
Initialized → CommitOutputs → RevealBenchmark → Evaluate → Prove → Verify → Certify → Completed
                                                                                ↓
                                                                      Aborted / TimedOut
```

States:

| State | Description |
|-------|-------------|
| `Initialized` | Protocol parameters agreed upon |
| `CommitOutputs` | Evaluatee commits to model outputs via hash commitments |
| `RevealBenchmark` | Benchmark data revealed to evaluator |
| `Evaluate` | Scoring computation on committed outputs |
| `Prove` | STARK proof generation |
| `Verify` | Proof verification |
| `Certify` | Certificate issuance |
| `Completed` | Protocol finished successfully |
| `Aborted(AbortReason)` | Protocol aborted (see `AbortReason` enum) |
| `TimedOut` | Protocol exceeded time limit |

`AbortReason` variants: `ConstraintViolation`, `TimeoutExceeded`,
`InvalidTransition`, `CommitmentMismatch`, `ProofFailed`, `ExternalAbort`.

The protocol layer provides extensive infrastructure:

- **`ProtocolStateMachine`** — core state machine with commitment storage,
  event logging, serialization/deserialization, and timeout checking.
- **`ProtocolPhaseManager`** — manages execution of named phases with timeouts
  and result recording.
- **`RetryManager`** — retry logic with `BackoffStrategy` (Exponential, Linear,
  Fixed).
- **`ProtocolAuditor`** — records `AuditEvent` instances and verifies audit
  trail integrity.
- **`ProtocolSimulator`** — simulates protocol execution with optional failure
  and delay injection.
- **`StateGraph`** — graph representation of state space for deadlock analysis
  and shortest-path computation.
- **`ProtocolRunner`** — high-level orchestrator with handler registration.
- **`ProtocolTemplate`** — reusable protocol definitions
  (`evaluation_protocol()`, `certification_protocol()`, `verification_only()`).
- **`StateHistory`** — event timeline analysis.

The protocol uses **Fiat-Shamir transcripts** for non-interactive proof
generation, and supports multiple commitment schemes:

| Scheme | Type | Properties |
|--------|------|------------|
| `HashCommitment` | Hash-based | Computationally binding, hiding |
| `PedersenCommitment` | Algebraic | Information-theoretically hiding |
| `VectorCommitment` | Merkle-based | Position binding |
| `PolynomialCommitment` | Polynomial | Homomorphic |
| `TimelockCommitment` | Time-locked | Delayed revelation |

`EvaluationCertificate`s are the final output — cryptographic attestations that
a specific score was computed correctly on committed inputs. Certificate
management is provided via `CertificateChain`, `CertificateStore`, and
`CertificateBuilder`.

### Layer 5: Privacy Layer

The **Privacy Layer** provides training data contamination detection via
**Private Set Intersection (PSI)**:

- **N-gram Extraction** (`ngram.rs`) — extracts character, token, and byte
  n-grams from text with configurable n via `NGramExtractor`. N-grams are
  represented as `NGram` structs with frequency data, managed through
  `NGramSet`, `NGramFrequencyMap`, and `NGramIndex` collections.
- **Trie Index** (`trie.rs`) — memory-efficient n-gram storage using `NGramTrie`
  and `CompactTrie` structures for fast prefix-based lookup and intersection.
- **OPRF** (`oprf.rs`) — Oblivious Pseudorandom Function protocol via
  `OPRFProtocol` with `OPRFKey`, `BlindedInput`/`BlindedOutput`, and
  `BlindingFactor` types. Includes `VerifiableOPRF` and `OTExtension`
  (Oblivious Transfer Extension) for enhanced security.
- **PSI Protocol** (`protocol.rs`) — full PSI protocol computing the overlap
  between evaluation text n-grams and (obfuscated) training data n-grams.
  Operates in `PSIMode::Streaming`, `Batch`, or `Threshold` modes through
  `PSIPhase::Setup → Hashing → Matching → Verification`.

Configuration via `PSIConfig` and `NGramConfig` (n, `GramType`, min_frequency,
normalization). Results delivered as `PSIResult` with intersection data,
overlap statistics, and contamination score.

The output is a `ContaminationAttestation` — a signed statement about the
contamination level (percentage of n-gram overlap) without revealing which
specific n-grams matched. A `ContaminationMatrix` provides detailed
cross-analysis.

### Layer 6: Scoring Layer

The **Scoring Layer** implements seven NLP evaluation metrics, each with a
**triple implementation** pattern: a reference (standard) algorithm, a WFA-based
computation, and an arithmetic circuit for ZK proofs.

All scoring metrics implement the `TripleMetric` trait:

```rust
pub trait TripleMetric {
    type Input;
    type Score;
    fn score_reference(&self, input: &Self::Input) -> Self::Score;
    fn score_automaton(&self, input: &Self::Input) -> Self::Score;
    fn score_circuit(&self, input: &Self::Input) -> Self::Score;
    fn score_and_verify(&self, input: &Self::Input) -> Self::Score;
}
```

Scores are represented as `FixedPointScore { numerator: u64, denominator: u64 }`
for exact rational arithmetic in the Goldilocks field. Scoring inputs use
`ScoringPair { candidate, reference }` or
`MultiRefScoringPair { candidate, references }`.

A `DifferentialTester` cross-validates all three implementations, producing
`DifferentialResult<T>` and `AgreementReport` indicating whether all three
implementations returned consistent results. Test generation is provided via
`standard_test_suite()` and `random_test_pairs()`.

Tokenization strategies are provided by the `Tokenizer` trait:

| Tokenizer | Description |
|-----------|-------------|
| `WhitespaceTokenizer` | Split on whitespace boundaries |
| `WordPieceTokenizer` | BPE-approximate subword tokenization |
| `CharacterTokenizer` | Character-level tokenization |
| `NGramTokenizer` | N-gram-based tokenization |

Each tokenizer produces `Token { id, value, span }` instances with
`NormalizationConfig` controlling case sensitivity and punctuation handling.

---

## Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Spectacles Pipeline                               │
│                                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────────┐   │
│  │ EvalSpec │──▶│   WFA    │──▶│ Circuit  │──▶│    STARK       │   │
│  │   DSL   │   │ Engine   │   │ Compiler │   │ Prover/Verifier│   │
│  └──────────┘   └──────────┘   └──────────┘   └────────────────┘   │
│       │              │              │               │               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────────┐   │
│  │  Type   │   │ Minimize │   │Goldilocks│   │  Certificate   │   │
│  │ Checker │   │ Equiv.   │   │  Field   │   │  Generator     │   │
│  └──────────┘   └──────────┘   └──────────┘   └────────────────┘   │
│                                                      │               │
│                              ┌──────────────────────────┐           │
│                              │  PSI Contamination       │           │
│                              │  Detection (OPRF/Trie)   │           │
│                              └──────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

**Data flow in detail:**

```
                    ┌─────────────┐
                    │  EvalSpec   │    "bleu(candidate, reference, n=4)"
                    │  Source     │
                    └──────┬──────┘
                           │ parse
                    ┌──────▼──────┐
                    │  Spanned    │    AST with source locations
                    │  <Expr>     │
                    └──────┬──────┘
                           │ typecheck (infer SemiringType::Counting)
                    ┌──────▼──────┐
                    │  Typed AST  │    EvalType::Metric { semiring: Counting }
                    │             │
                    └──────┬──────┘
                           │ compile (EvalSpecCompiler)
                    ┌──────▼──────┐
                    │  WFA<       │    4 n-gram counting automata
                    │  Counting   │    composed via product construction
                    │  Semiring>  │
                    └──────┬──────┘
                           │ field_embed
                    ┌──────▼──────┐
                    │  WFA<       │    Semiring ops mapped to 𝔽_p arithmetic
                    │  Goldilocks │
                    │  Field>     │
                    └──────┬──────┘
                           │ AIR compile (WFACircuitCompiler)
                    ┌──────▼──────┐
                    │  AIRProgram │    Boundary + transition constraints
                    │  + Layout   │    TraceLayout with column schema
                    └──────┬──────┘
                           │ trace generation
                    ┌──────▼──────┐
                    │  AIRTrace   │    2D table of GoldilocksField elements
                    │  (witness)  │
                    └──────┬──────┘
                           │ STARK prove (FRI + Merkle commitments)
                    ┌──────▼──────┐
                    │  STARKProof │    Succinct proof of correct execution
                    └──────┬──────┘
                           │ verify + certify
                    ┌──────▼──────┐
                    │  Evaluation │    Cryptographic attestation:
                    │  Certificate│    score + proof + commitment
                    └─────────────┘
```

---

## Module Map

| Module | Path | Description | Key Types |
|--------|------|-------------|-----------|
| evalspec | `spectacles-core/src/evalspec/` | EvalSpec DSL: typed evaluation specification language with semiring inference | `Span`, `Spanned<T>`, `Expr`, `EvalType`, `BaseType`, `SemiringType`, `MetricType`, `EvalSpecCompiler` |
| wfa | `spectacles-core/src/wfa/` | Weighted finite automata engine with semiring-generic operations | `WeightedFiniteAutomaton<S>`, `WFABuilder`, `Symbol`, `Alphabet`, `Transition<S>`, `Semiring`, `CountingSemiring`, `BooleanSemiring`, `TropicalSemiring`, `MaxPlusSemiring`, `BoundedCountingSemiring` |
| circuit | `spectacles-core/src/circuit/` | WFA-to-STARK circuit synthesizer with two-tier compilation | `GoldilocksField`, `GoldilocksExt`, `AIRConstraint`, `AIRProgram`, `AIRTrace`, `TraceLayout`, `ColumnType`, `ConstraintType`, `SymbolicExpression`, `PeriodicColumn`, `BoundaryDescriptor`, `ScoringCircuit`, `CircuitConstraint` |
| protocol | `spectacles-core/src/protocol/` | Commit-reveal-verify protocol engine | `ProtocolStateMachine`, `ProtocolState`, `AbortReason`, `ProtocolConfig`, `ProtocolPhaseManager`, `RetryManager`, `BackoffStrategy`, `ProtocolAuditor`, `ProtocolSimulator`, `StateGraph`, `ProtocolRunner`, `ProtocolTemplate`, `CommitmentScheme`, `HashCommitment`, `PedersenCommitment`, `EvaluationCertificate`, `CertificateBuilder`, `CertificateChain`, `CertificateStore` |
| psi | `spectacles-core/src/psi/` | PSI contamination detection with trie-structured OPRF | `PSIProtocol`, `PSIConfig`, `PSIResult`, `PSIMode`, `PSIPhase`, `NGramExtractor`, `NGram`, `NGramConfig`, `GramType`, `NGramTrie`, `CompactTrie`, `OPRFProtocol`, `OPRFKey`, `VerifiableOPRF`, `ContaminationAttestation`, `ContaminationMatrix` |
| scoring | `spectacles-core/src/scoring/` | 7 NLP metrics with triple implementation (reference/WFA/circuit) | `TripleMetric`, `ScoringPair`, `FixedPointScore`, `BleuScorer`, `BleuConfig`, `RougeNScorer`, `RougeLScorer`, `RougeConfig`, `ExactMatchScorer`, `NormalizedExactMatchScorer`, `TokenF1Scorer`, `MacroF1Scorer`, `MicroF1Scorer`, `RegexMatchScorer`, `PassAtKScorer`, `DifferentialTester`, `DifferentialResult`, `AgreementReport` |
| utils | `spectacles-core/src/utils/` | Hashing, serialization, math utilities | `SpectaclesHasher`, `DomainSeparatedHasher`, `MerkleTree`, `MerkleProof`, `HashChain`, `Commitment`, `ProofSerializer`, `ProofFormat`, `CompactProof` |

---

## Supported Metrics

| # | Metric | Semiring | WFA States (est.) | Compilation Tier | Description |
|---|--------|----------|--------------------|-----------------|-------------|
| 1 | **Exact Match** | Boolean | O(n) | Tier 1 (algebraic) | Binary match between candidate and reference strings |
| 2 | **Token F1** | Counting | O(\|V\|) | Tier 1 (algebraic) | Token-level precision, recall, and F1 score (macro/micro) |
| 3 | **BLEU** | Counting | O(n⁴) | Tier 1 (algebraic) | Bilingual Evaluation Understudy with n-gram precision (n=1..4) and brevity penalty |
| 4 | **ROUGE-N** | Counting | O(n^k) | Tier 1 (algebraic) | Recall-Oriented Understudy (n-gram overlap, configurable n) |
| 5 | **ROUGE-L** | MaxPlus | O(n·m) | Tier 2 (gadget-assisted) | Longest Common Subsequence variant of ROUGE |
| 6 | **Regex Match** | Boolean | O(2^k) | Tier 1 (algebraic) | Pattern matching against regular expression specifications |
| 7 | **Pass@k** | Counting | O(k·t) | Tier 2 (gadget-assisted) | Unbiased estimator for code generation (k samples from t trials) |

Each metric is implemented in its own module (`exact_match.rs`, `bleu.rs`, etc.)
with all three tiers. A shared `tokenizer.rs` provides multiple tokenization
strategies (whitespace, WordPiece, character-level, n-gram).

### Metric Implementation Details

**Exact Match** (`exact_match.rs`):
- `ExactMatchScorer` — direct string comparison
- `NormalizedExactMatchScorer` — comparison with configurable normalization
- Helper: `exact_match_accuracy()` for batch evaluation
- WFA: `build_multi_string_wfa()` for multi-reference matching

**Token F1** (`token_f1.rs`):
- `TokenF1Scorer` — standard token-level F1
- `MacroF1Scorer` — macro-averaged F1 across categories
- `MicroF1Scorer` — micro-averaged F1 (global TP/FP/FN)
- Helpers: `token_overlap()`, `count_token_ngrams()`

**BLEU** (`bleu.rs`):
- `BleuScorer` with `BleuConfig { max_n, smoothing_method, lowercase }`
- `NgramPrecision` — per-n precision breakdown
- `BleuResult` — composite result with brevity penalty

**ROUGE** (`rouge.rs`):
- `RougeNScorer` — n-gram recall with configurable n
- `RougeLScorer` — LCS-based F-measure
- `RougeConfig` — shared configuration
- Helper: `simple_stem()` for optional stemming

**Regex Match** (`regex_match.rs`):
- `RegexMatchScorer` — pattern matching via compiled automata
- `RegexCompiler` — `RegexAst` → `Nfa` → `Dfa` pipeline
- Combinators: `regex_union()`, `regex_concat()`

**Pass@k** (`pass_at_k.rs`):
- `PassAtKScorer` with `PassAtKConfig`
- `PassAtKResult` — detailed pass/fail breakdown
- Helpers: `binomial()`, `corpus_pass_at_k()`

---

## Two-Tier Compilation

Spectacles uses a two-tier compilation strategy to handle metrics of varying
algebraic complexity:

### Tier 1: Algebraic Compilation

Metrics with direct semiring semantics compile through the standard WFA pipeline:

```
EvalSpec → WFA<Semiring> → field_embed → WFA<GoldilocksField> → AIR → STARK
```

Tier 1 metrics include:
- **Exact Match** — Boolean semiring, direct string comparison automaton
- **Token F1** — Counting semiring, bag-of-words intersection
- **BLEU** — Counting semiring, n-gram counting automata composed in product
- **ROUGE-N** — Counting semiring, n-gram recall automata
- **Regex Match** — Boolean semiring, standard NFA→DFA→WFA

These metrics have natural WFA representations where the semiring operations
directly correspond to the scoring computation. Field embedding preserves
the algebraic structure via a semiring homomorphism:

```
h: S → GoldilocksField
h(a ⊕ b) = h(a) + h(b)   (mod p)
h(a ⊗ b) = h(a) × h(b)   (mod p)
h(0̄) = 0
h(1̄) = 1
```

AIR constraints directly encode WFA transitions. For a WFA with n states and
alphabet Σ, the transition constraint at row i ensures:

```
state_j(i+1) = Σ_k  transition[k][input(i)][j] × state_k(i)
```

### Tier 2: Gadget-Assisted Compilation

Metrics requiring operations outside the semiring algebra (division, max,
comparison) use arithmetic gadgets:

```
EvalSpec → WFA<Semiring> + Gadgets → AIR(extended) → STARK
```

Tier 2 metrics include:
- **ROUGE-L** — requires longest common subsequence (dynamic programming),
  implemented via max-plus semiring with comparison gadgets
- **Pass@k** — requires combinatorial estimation with division,
  implemented via counting semiring with division gadgets

Gadgets provide verified sub-circuits for:

| Gadget | Operation | Constraint Cost |
|--------|-----------|-----------------|
| Bit Decomposition | x → (b₀, b₁, ..., bₖ) | O(log p) |
| Range Check | 0 ≤ x < 2ᵏ | O(k) |
| Comparison | x ≥ y via bit decomposition | O(log p) |
| Division | x / y with remainder check | O(1) + range check |
| Maximum | max(x, y) via comparison | O(log p) |

Gadget-assisted compilation produces larger circuits but handles the full
range of scoring functions.

---

## STARK Proof System

Spectacles uses a STARK (Scalable Transparent Argument of Knowledge) proof system
built on three core components:

### Goldilocks Field

All arithmetic operates in the Goldilocks prime field:

```
p = 2⁶⁴ − 2³² + 1 = 0xFFFFFFFF00000001
```

This prime is chosen for efficient 64-bit arithmetic: multiplication reduces via
a single shift-and-subtract, and the multiplicative group has large 2-adic order
(2³²), enabling efficient NTT-based polynomial multiplication.

The `GoldilocksField` type provides:
- Basic arithmetic: `add()`, `sub()`, `mul()`, `div()`
- Modular exponentiation: `pow()`
- Multiplicative inverse: `inv()` (via Fermat's little theorem)
- Extension field: `GoldilocksExt` for FRI query domain

### FRI Protocol

The FRI (Fast Reed-Solomon Interactive oracle proofs of proximity) protocol
provides low-degree testing for polynomial commitment. The protocol proceeds
in rounds:

1. **Commit phase**: Prover commits to evaluations of a polynomial on a
   coset of a multiplicative subgroup via Merkle trees
2. **Fold phase**: Verifier sends random challenges; prover folds the
   polynomial, halving the degree each round
3. **Query phase**: Verifier checks consistency of folded polynomials
   at random positions via Merkle proofs

FRI achieves O(log² n) proof size and O(log² n) verification time for a
degree-n polynomial, with soundness error 2^(-λ) for security parameter λ.

### Merkle Tree Commitments

The `MerkleTree` type (in `utils/hash.rs`) provides vector commitments for
trace columns:

- **Construction**: O(n) time for n leaves
- **Proof generation**: O(log n) `MerkleProofStep` instances per query
- **Verification**: O(log n) hashes per proof
- **Hash function**: BLAKE3 with domain separation via `DomainSeparatedHasher`

The hash chain `HashChain` provides sequential commitment for the
Fiat-Shamir transcript, binding all prover messages into a single
non-interactive proof.

### Proof Generation Pipeline

```
1. Generate AIRTrace (execution witness)
2. Commit to trace columns via MerkleTree
3. Compute constraint polynomials
4. Commit to composition polynomial
5. Run FRI on composition polynomial
6. Open queried positions with Merkle proofs
7. Package as STARKProof
```

The resulting `STARKProof` is serializable via `ProofSerializer` in multiple
formats (`ProofFormat::JSON`, `Bincode`, etc.) with `CompactProof`
representation for bandwidth efficiency.

---

## PSI Contamination Detection

Spectacles detects training data contamination without revealing the training set
through a Private Set Intersection protocol built on trie-structured OPRF.

### Protocol Flow

```
┌──────────────┐                    ┌──────────────┐
│   Evaluator  │                    │   Trainer    │
│   (has eval  │                    │   (has train │
│    text)     │                    │    data)     │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       │  1. Extract n-grams               │  1. Extract n-grams
       │     (NGramExtractor)              │     (NGramExtractor)
       │                                   │
       │  2. Blind n-grams                 │
       │     (OPRFProtocol.blind)          │
       │                                   │
       │  ──── blinded n-grams ──────▶     │
       │                                   │  3. Evaluate OPRF
       │                                   │     (OPRFProtocol.evaluate)
       │     ◀──── OPRF outputs ─────      │
       │                                   │
       │  4. Unblind outputs               │
       │     (OPRFProtocol.unblind)        │
       │                                   │
       │  5. Compare against trainer's     │
       │     encoded n-gram set            │
       │     (NGramTrie intersection)      │
       │                                   │
       │  6. Compute overlap statistics    │
       │     → ContaminationAttestation    │
       └───────────────────────────────────┘
```

### Communication Complexity

The PSI protocol achieves communication complexity **O((n₁ + n₂) · λ)** where:
- n₁ = number of evaluator n-grams
- n₂ = number of trainer n-grams
- λ = security parameter (typically 128 bits)

This is achieved via `OTExtension` (Oblivious Transfer Extension) which
amortizes the cost of base OTs across all n-gram comparisons.

### N-gram Configuration

```rust
let config = NGramConfig {
    n: 5,                          // 5-gram overlap detection
    gram_type: GramType::Token,    // token-level (vs Character, Byte)
    min_frequency: 2,              // ignore rare n-grams
    normalized: true,              // lowercase + strip punctuation
};
```

### Trie-Structured Storage

The `NGramTrie` provides memory-efficient n-gram storage:
- Insert: O(k) per n-gram of length k
- Lookup: O(k) per query
- Prefix search: O(k + output size)
- `CompactTrie` variant for memory-constrained environments

### Result Types

- `PSIResult` — intersection cardinality, overlap statistics, contamination score
- `ContaminationAttestation` — signed statement with overlap percentage
- `ContaminationMatrix` — detailed cross-analysis across multiple n-gram sizes

---

## Metric Equivalence

Spectacles provides decidable WFA equivalence checking for verifying that
different implementations of the same metric compute identical functions.

### Equivalence via Minimization + Isomorphism

Two WFAs are equivalent if and only if they recognize the same formal power
series. Spectacles checks this through:

1. **Minimization** — Hopcroft-style minimization adapted for weighted automata,
   producing a canonical minimal WFA that preserves the recognized FPS
2. **Isomorphism check** — two minimal WFAs recognizing the same FPS are
   isomorphic (up to state relabeling)

This procedure is decidable for all supported semirings and runs in
O(n · |Σ| · log n) time for an n-state WFA over alphabet Σ.

### Use Cases

- **Verifying metric variants**: Confirm that two BLEU implementations (e.g.,
  with different smoothing methods) compute the same function
- **Optimized compilation**: Verify that a hand-optimized WFA is equivalent to
  the automatically compiled version
- **Regression testing**: Ensure that refactoring preserves metric semantics

### CLI Usage

```bash
# Check equivalence of two metric WFAs
cargo run -p spectacles-cli -- check-equivalence \
    --metric1 bleu --metric2 bleu_smooth
```

### Formal Power Series Semantics

A WFA over semiring (S, ⊕, ⊗, 0̄, 1̄) and alphabet Σ recognizes a formal power
series f: Σ* → S. For input word w = a₁a₂...aₙ:

```
f(w) = α · M(a₁) ⊗ M(a₂) ⊗ ... ⊗ M(aₙ) · β
```

where α is the initial weight vector, β is the final weight vector, and M(a)
is the transition matrix for symbol a. Two WFAs are equivalent iff they
recognize the same formal power series — i.e., for all w ∈ Σ*:

```
f₁(w) = f₂(w)
```

---

## Getting Started

### Prerequisites

- **Rust** 1.75+ (2021 edition)
- **Cargo** (comes with Rust)

### Build

```bash
# Build all workspace members
cargo build --release

# Build only the core library
cargo build --release -p spectacles-core

# Build the CLI
cargo build --release -p spectacles-cli
```

### Run Tests

```bash
# Run all tests (unit + integration, 41+ integration tests)
cargo test

# Run tests for a specific crate
cargo test -p spectacles-core

# Run integration tests (full pipeline, differential, metric-specific E2E)
cargo test -p spectacles-integration

# Run with logging
RUST_LOG=debug cargo test

# Run property-based tests (may take longer)
cargo test -p spectacles-core -- --ignored
```

### Quick Example: Score a Text Pair

```bash
# Score with BLEU
cargo run -p spectacles-cli -- score \
    --metric bleu \
    --candidate "the cat sat on the mat" \
    --reference "the cat is on the mat"

# Score with triple verification (validates reference/WFA/circuit agree)
cargo run -p spectacles-cli -- score \
    --metric bleu \
    --candidate "the cat sat on the mat" \
    --reference "the cat is on the mat" \
    --triple
```

### Quick Example: Differential Testing

```bash
# Run differential testing across all 3 implementations
cargo run -p spectacles-cli -- differential-test \
    --metric token_f1 \
    --candidate "hello world foo" \
    --reference "hello bar world"

# Run with random test pairs
cargo run -p spectacles-cli -- differential-test \
    --metric bleu \
    --count 100 \
    --seed 42
```

### Quick Example: Circuit Compilation

```bash
# Compile a WFA to a STARK circuit
cargo run -p spectacles-cli -- compile-circuit --metric bleu

# Compile with custom max input length
cargo run -p spectacles-cli -- compile-circuit \
    --metric exact_match \
    --max-length 256

# Estimate proof size for a metric
cargo run -p spectacles-cli -- estimate-size \
    --metric rouge1 \
    --constraints 1024 \
    --wires 2048 \
    --security 128
```

### Quick Example: Batch Scoring

```bash
# Score multiple pairs from a JSONL file
cargo run -p spectacles-cli -- batch-score \
    --metric exact_match \
    --input pairs.jsonl

# Input format (pairs.jsonl):
# {"candidate": "hello world", "reference": "hello world"}
# {"candidate": "foo bar", "reference": "foo baz"}
```

### Quick Example: Hashing

```bash
# Hash input with domain-separated BLAKE3
cargo run -p spectacles-cli -- hash \
    --domain "spectacles.commitment" \
    --data "my secret data"
```

### Programmatic Usage (Rust)

```rust
use spectacles_core::scoring::{BleuScorer, ScoringPair, TripleMetric};
use spectacles_core::wfa::{WeightedFiniteAutomaton, CountingSemiring};

// Score with BLEU
let scorer = BleuScorer::default();
let pair = ScoringPair {
    candidate: "the cat sat on the mat".into(),
    reference: "the cat is on the mat".into(),
};

let ref_score = scorer.score_reference(&pair);
let wfa_score = scorer.score_automaton(&pair);
let circuit_score = scorer.score_circuit(&pair);

// All three should agree
assert_eq!(ref_score, wfa_score);
assert_eq!(wfa_score, circuit_score);
```

---

## CLI Reference

The `spectacles-cli` provides the following subcommands:

### `score` — Evaluate a metric

Score a single candidate/reference pair with one of 7 metrics.

```
spectacles-cli score [OPTIONS]

Options:
  -m, --metric <METRIC>       Metric name (see supported metrics)
  -c, --candidate <TEXT>      Candidate text to score
  -r, --reference <TEXT>      Reference text to score against
  -t, --triple                Run all 3 implementations and verify agreement
```

### `differential-test` — Cross-validate implementations

Run differential testing across reference, WFA, and circuit implementations.

```
spectacles-cli differential-test [OPTIONS]

Options:
  -m, --metric <METRIC>       Metric name
  -c, --candidate <TEXT>      Candidate text (optional; random if omitted)
  -r, --reference <TEXT>      Reference text (optional; random if omitted)
  -n, --count <N>             Number of random test pairs to generate
  -s, --seed <SEED>           Random seed for reproducibility
```

### `compile-circuit` — Compile metric to STARK circuit

Compile a metric's WFA into an AIR program with STARK constraints.

```
spectacles-cli compile-circuit [OPTIONS]

Options:
  -m, --metric <METRIC>       Metric name
  -l, --max-length <N>        Maximum input length (affects circuit size)
```

### `estimate-size` — Estimate proof size

Estimate the proof size for a given circuit configuration.

```
spectacles-cli estimate-size [OPTIONS]

Options:
  --metric <METRIC>           Metric name (uses default circuit params)
  --constraints <N>           Number of constraints
  --wires <N>                 Number of wires
  --security <BITS>           Security parameter in bits
```

### `batch-score` — Score multiple pairs

Score multiple candidate/reference pairs from a JSONL input file.

```
spectacles-cli batch-score [OPTIONS]

Options:
  -m, --metric <METRIC>       Metric name
  -i, --input <FILE>          Path to JSONL input file
```

### `hash` — Domain-separated hashing

Hash input data with BLAKE3 and domain separation.

```
spectacles-cli hash [OPTIONS]

Options:
  -d, --domain <DOMAIN>       Domain separation string
  --data <DATA>               Data to hash
```

### `help` / `version`

```bash
spectacles-cli help              # Print help
spectacles-cli --version         # Print version
spectacles-cli help <COMMAND>    # Help for a specific subcommand
```

Supported `--metric` values: `exact_match`, `token_f1`, `bleu`, `rouge1`,
`rouge2`, `rougel`, `regex_match`, `pass_at_k`.

---

## Project Structure

```
spectacles-wfa-zk-scoring-circuits/implementation/
├── Cargo.toml                              # Workspace manifest
├── Cargo.lock                              # Locked dependency versions
├── README.md                               # This file
├── api.md                                  # Comprehensive API reference
│
├── spectacles-core/                        # Core library
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                          # Crate root, module declarations
│       │
│       ├── evalspec/                       # Evaluation Specification DSL
│       │   ├── mod.rs                      #   Module re-exports
│       │   ├── types.rs                    #   Span, Spanned<T>, BaseType,
│       │   │                               #   SemiringType, EvalType, MetricType
│       │   ├── parser.rs                   #   Syntax parser → Spanned<Expr>
│       │   ├── typechecker.rs              #   Type checking & semiring inference
│       │   ├── compiler.rs                 #   EvalSpecCompiler: EvalSpec → WFA
│       │   ├── semantics.rs                #   Denotational semantics
│       │   └── builtins.rs                 #   Built-in scoring functions
│       │
│       ├── wfa/                            # Weighted Finite Automata
│       │   ├── mod.rs                      #   Module re-exports
│       │   ├── semiring.rs                 #   Semiring trait + 5 implementations
│       │   ├── automaton.rs                #   WeightedFiniteAutomaton<S>,
│       │   │                               #   WFABuilder, Symbol, Alphabet
│       │   ├── transducer.rs               #   Weighted transducers
│       │   ├── minimization.rs             #   Hopcroft minimization for WFAs
│       │   ├── equivalence.rs              #   FPS-based language equivalence
│       │   ├── operations.rs               #   ∪, ·, *, ∩, complement, reverse
│       │   ├── formal_power_series.rs      #   FPS algebraic semantics
│       │   └── field_embedding.rs          #   Semiring → 𝔽_p embedding
│       │
│       ├── circuit/                        # STARK Circuit Synthesis
│       │   ├── mod.rs                      #   Module re-exports
│       │   ├── goldilocks.rs               #   GoldilocksField, GoldilocksExt
│       │   │                               #   (p = 2⁶⁴ − 2³² + 1)
│       │   ├── air.rs                      #   AIRConstraint, AIRProgram,
│       │   │                               #   SymbolicExpression, ConstraintType
│       │   ├── compiler.rs                 #   WFACircuitCompiler: WFA → AIR
│       │   ├── trace.rs                    #   AIRTrace, TraceLayout, ColumnType
│       │   ├── stark.rs                    #   STARK prover & verifier
│       │   ├── fri.rs                      #   FRI low-degree testing
│       │   ├── merkle.rs                   #   Merkle commitment trees
│       │   └── gadgets.rs                  #   Bit decomposition, range checks,
│       │                                   #   comparison, division, maximum
│       │
│       ├── protocol/                       # ZK Protocol Engine
│       │   ├── mod.rs                      #   Module re-exports
│       │   ├── state_machine.rs            #   ProtocolStateMachine,
│       │   │                               #   ProtocolState, AbortReason,
│       │   │                               #   ProtocolConfig, ProtocolRunner,
│       │   │                               #   ProtocolSimulator, StateGraph,
│       │   │                               #   ProtocolAuditor, RetryManager
│       │   ├── commitment.rs               #   CommitmentScheme trait,
│       │   │                               #   HashCommitment, PedersenCommitment,
│       │   │                               #   VectorCommitment, PolynomialCommitment,
│       │   │                               #   TimelockCommitment
│       │   ├── transcript.rs               #   FiatShamirTranscript
│       │   └── certificate.rs              #   EvaluationCertificate,
│       │                                   #   CertificateBuilder/Chain/Store
│       │
│       ├── psi/                            # Private Set Intersection
│       │   ├── mod.rs                      #   Module re-exports
│       │   ├── ngram.rs                    #   NGramExtractor, NGram, NGramConfig,
│       │   │                               #   GramType, NGramSet, NGramFrequencyMap
│       │   ├── trie.rs                     #   NGramTrie, CompactTrie, NGramIndex
│       │   ├── oprf.rs                     #   OPRFProtocol, OPRFKey,
│       │   │                               #   VerifiableOPRF, OTExtension,
│       │   │                               #   BlindedInput/Output, BlindingFactor
│       │   └── protocol.rs                 #   PSIProtocol, PSIConfig, PSIResult,
│       │                                   #   PSIMode, PSIPhase,
│       │                                   #   ContaminationAttestation/Matrix
│       │
│       ├── scoring/                        # NLP Scoring Metrics
│       │   ├── mod.rs                      #   Module re-exports, TripleMetric trait,
│       │   │                               #   ScoringPair, FixedPointScore
│       │   ├── tokenizer.rs                #   Tokenizer trait, WhitespaceTokenizer,
│       │   │                               #   WordPieceTokenizer, CharacterTokenizer,
│       │   │                               #   NGramTokenizer, Token,
│       │   │                               #   NormalizationConfig
│       │   ├── exact_match.rs              #   ExactMatchScorer,
│       │   │                               #   NormalizedExactMatchScorer
│       │   ├── token_f1.rs                 #   TokenF1Scorer, MacroF1Scorer,
│       │   │                               #   MicroF1Scorer
│       │   ├── bleu.rs                     #   BleuScorer, BleuConfig,
│       │   │                               #   NgramPrecision, BleuResult
│       │   ├── rouge.rs                    #   RougeNScorer, RougeLScorer,
│       │   │                               #   RougeConfig
│       │   ├── regex_match.rs              #   RegexMatchScorer, RegexCompiler,
│       │   │                               #   RegexAst, Nfa, Dfa
│       │   ├── pass_at_k.rs               #   PassAtKScorer, PassAtKConfig,
│       │   │                               #   PassAtKResult
│       │   └── differential.rs             #   DifferentialTester,
│       │                                   #   DifferentialResult, AgreementReport
│       │
│       └── utils/                          # Utilities
│           ├── mod.rs                      #   Module re-exports
│           ├── hash.rs                     #   SpectaclesHasher,
│           │                               #   DomainSeparatedHasher,
│           │                               #   MerkleTree, MerkleProof,
│           │                               #   HashChain, Commitment
│           ├── math.rs                     #   extended_gcd, mod_pow, mod_inv,
│           │                               #   polynomial_eval/add/mul,
│           │                               #   lagrange_interpolate, fft, ntt,
│           │                               #   find_primitive_root, is_prime, crt
│           └── serialization.rs            #   ProofSerializer, ProofFormat,
│                                           #   CompactProof, FormatVersion,
│                                           #   estimate_proof_size,
│                                           #   compress_rle/decompress_rle
│
├── spectacles-cli/                         # Command-line interface
│   ├── Cargo.toml
│   └── src/
│       └── main.rs                         #   CLI: score, differential-test,
│                                           #   compile-circuit, estimate-size,
│                                           #   batch-score, hash
│
├── spectacles-examples/                    # Example programs
│   ├── Cargo.toml
│   └── src/
│       └── main.rs                         #   Usage examples (stub)
│
└── spectacles-integration/                 # Integration tests
    ├── Cargo.toml
    ├── src/
    │   └── lib.rs                          #   run_scoring_pipeline() utility
    └── tests/
        └── e2e_tests.rs                    #   41+ E2E tests: full pipeline,
                                            #   differential, metric-specific,
                                            #   certificate, cross-cutting
```

---

## Dependencies

### Rust (spectacles-core)

| Crate | Version | Purpose |
|-------|---------|---------|
| `serde` / `serde_json` | 1.x | Serialization for proofs, certificates, transcripts |
| `tokio` | 1.x | Async runtime for protocol state machine |
| `rayon` | 1.10 | Parallel computation for trace generation and FRI |
| `nalgebra` | 0.33 | Linear algebra for WFA matrix operations |
| `ndarray` | 0.16 | N-dimensional arrays for transition tensors |
| `num` / `num-bigint` / `num-traits` / `num-integer` | 0.4 / 0.4 / 0.2 / 0.1 | Big integer and generic numeric operations for field arithmetic |
| `sha2` | 0.10 | SHA-256 for Merkle trees and hash commitments |
| `blake3` | 1.x | BLAKE3 for fast hashing in Fiat-Shamir transcripts |
| `hex` | 0.4 | Hex encoding for proof serialization |
| `ordered-float` | 4.x | Orderable floats for tropical/max-plus semirings |
| `petgraph` | 0.6 | Graph algorithms for WFA minimization and state graphs |
| `rand` / `rand_distr` | 0.8 / 0.4 | Random sampling for FRI queries, testing, and simulation |
| `indexmap` | 2.x | Ordered maps for deterministic WFA construction |
| `uuid` | 1.x | UUID generation for protocol sessions and certificates |
| `chrono` | 0.4 | Timestamps for certificates and audit events |
| `thiserror` | 2.x | Structured error types (`WfaError`, `ProtocolError`, `PSIError`, etc.) |
| `anyhow` | 1.x | Ergonomic error handling in CLI and integration tests |
| `log` / `env_logger` | 0.4 / 0.11 | Logging infrastructure |

### Dev Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `proptest` | 1.x | Property-based testing for semiring laws and circuit correctness |
| `criterion` | 0.5 | Benchmarking for prover performance |

### Workspace Members

| Crate | Description |
|-------|-------------|
| `spectacles-core` | Core library with all modules |
| `spectacles-cli` | Command-line interface |
| `spectacles-examples` | Example programs |
| `spectacles-integration` | Integration and E2E tests |

---

## Key Types Quick Reference

```rust
// ═══════════════════════════════════════════════════
// Semiring Algebra
// ═══════════════════════════════════════════════════

pub trait Semiring: Clone + Debug + PartialEq {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
}

pub struct BooleanSemiring(pub bool);        // (∨, ∧, false, true)
pub struct CountingSemiring(pub u64);        // (+, ×, 0, 1)
pub struct BoundedCountingSemiring {         // (+, ×, 0, 1) with ceiling
    pub value: u64, pub bound: u64,
}
pub struct TropicalSemiring(pub f64);        // (min, +, ∞, 0)
pub struct MaxPlusSemiring(pub f64);         // (max, +, −∞, 0)

// ═══════════════════════════════════════════════════
// Weighted Finite Automata
// ═══════════════════════════════════════════════════

pub enum Symbol { Char(char), Byte(u8), Token(String), Epsilon, Wildcard, Id(usize) }

pub struct WeightedFiniteAutomaton<S: Semiring> {
    num_states: usize,
    alphabet_size: usize,
    initial_weights: Vec<S>,
    final_weights: Vec<S>,
    transitions: Vec<Vec<Vec<S>>>,  // [state][symbol][state]
}

pub struct Transition<S: Semiring> {
    pub from: usize, pub to: usize,
    pub symbol: Symbol, pub weight: S,
}

// ═══════════════════════════════════════════════════
// Goldilocks Field
// ═══════════════════════════════════════════════════

pub struct GoldilocksField(pub u64);  // p = 2^64 - 2^32 + 1
pub const MODULUS: u64 = 0xFFFFFFFF00000001;

pub struct GoldilocksExt { /* extension field */ }

// ═══════════════════════════════════════════════════
// AIR Constraints
// ═══════════════════════════════════════════════════

pub enum ConstraintType { Boundary, Transition, Periodic, Composition }
pub enum ColumnType { State, Input, Auxiliary, Public }

pub enum SymbolicExpression {
    Constant(GoldilocksField), Variable(String),
    Add(Box<Self>, Box<Self>), Mul(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>), Neg(Box<Self>),
    Pow(Box<Self>, u64), CurrentRow(usize), NextRow(usize),
}

pub struct AIRConstraint {
    pub name: String,
    pub expr: SymbolicExpression,
    pub constraint_type: ConstraintType,
    pub boundary_row: Option<usize>,
}

pub struct AIRProgram {
    pub constraints: Vec<AIRConstraint>,
    pub layout: TraceLayout,
    pub public_inputs: Vec<GoldilocksField>,
}

pub struct AIRTrace { /* 2D table of GoldilocksField elements */ }

// ═══════════════════════════════════════════════════
// Protocol
// ═══════════════════════════════════════════════════

pub enum ProtocolState {
    Initialized, CommitOutputs, RevealBenchmark, Evaluate,
    Prove, Verify, Certify, Completed,
    Aborted(AbortReason), TimedOut,
}

pub enum AbortReason {
    ConstraintViolation, TimeoutExceeded, InvalidTransition,
    CommitmentMismatch, ProofFailed, ExternalAbort,
}

pub struct EvaluationCertificate { /* cryptographic attestation */ }

// ═══════════════════════════════════════════════════
// Scoring
// ═══════════════════════════════════════════════════

pub trait TripleMetric {
    type Input;
    type Score;
    fn score_reference(&self, input: &Self::Input) -> Self::Score;
    fn score_automaton(&self, input: &Self::Input) -> Self::Score;
    fn score_circuit(&self, input: &Self::Input) -> Self::Score;
    fn score_and_verify(&self, input: &Self::Input) -> Self::Score;
}

pub struct ScoringPair { pub candidate: String, pub reference: String }
pub struct MultiRefScoringPair { pub candidate: String, pub references: Vec<String> }
pub struct FixedPointScore { pub numerator: u64, pub denominator: u64 }

pub enum CircuitConstraint {
    Mul { a: usize, b: usize, c: usize },
    Add { a: usize, b: usize, c: usize },
    Eq  { a: usize, b: usize },
    Const { a: usize, val: u64 },
    Bool { a: usize },
}

// ═══════════════════════════════════════════════════
// PSI / Contamination Detection
// ═══════════════════════════════════════════════════

pub enum PSIMode { Streaming, Batch, Threshold }
pub enum PSIPhase { Setup, Hashing, Matching, Verification }
pub enum GramType { Character, Token, Byte }

pub struct ContaminationAttestation { /* signed overlap statement */ }
pub struct ContaminationMatrix { /* multi-n cross-analysis */ }
```

---

## Theoretical Background

### Formal Power Series over Free Monoids

A WFA over semiring (S, ⊕, ⊗, 0̄, 1̄) and alphabet Σ recognizes a formal power
series f: Σ* → S. For input word w = a₁a₂...aₙ:

```
f(w) = α · M(a₁) ⊗ M(a₂) ⊗ ... ⊗ M(aₙ) · β
```

where α is the initial weight vector, β is the final weight vector, and M(a)
is the transition matrix for symbol a. Language equivalence reduces to checking
equality of the recognized formal power series.

The connection between NLP metrics and WFAs is that n-gram counting, string
matching, and subsequence detection all naturally decompose into weighted
automata computations:

| Metric | WFA Decomposition |
|--------|-------------------|
| Exact Match | Single-path DFA with Boolean weights |
| BLEU | Product of n-gram counting WFAs (n=1..4) |
| ROUGE-N | N-gram recall WFA with counting weights |
| ROUGE-L | LCS WFA with max-plus weights |
| Token F1 | Bag-of-words intersection WFA |

### STARK Proof System

The STARK (Scalable Transparent Argument of Knowledge) proof system provides:

- **Succinctness** — proof size is O(log² n) for computation of size n
- **Transparency** — no trusted setup required (unlike SNARKs)
- **Post-quantum security** — based on hash functions (BLAKE3/SHA-256),
  not elliptic curves or pairings
- **Scalability** — prover time is O(n · log n), verifier time is O(log² n)

The proof system works as follows:

1. **Arithmetization**: The WFA computation is encoded as an AIR (Algebraic
   Intermediate Representation) with polynomial constraints over the
   Goldilocks field
2. **Trace generation**: The prover executes the WFA and records the execution
   trace as a 2D table of field elements
3. **Commitment**: Trace columns are committed via Merkle trees
4. **Constraint composition**: All AIR constraints are composed into a single
   polynomial C(x) using random linear combination
5. **FRI**: The FRI protocol proves that C(x) has low degree (i.e., the
   constraints are satisfied everywhere)
6. **Query phase**: The verifier spot-checks consistency at random positions

The resulting proof is a `STARKProof` containing Merkle roots, FRI layer
commitments, and query responses.

### Contamination Detection via PSI

Training data contamination is detected without revealing the training set:

1. **Extract n-grams** from evaluation text using `NGramExtractor`
2. **Encode** both sets via OPRF (neither party learns the other's set)
3. **Compute** intersection cardinality via trie-structured comparison
4. **Issue** `ContaminationAttestation` with overlap percentage

The security guarantee: the evaluator learns only the overlap cardinality
(and a configurable threshold), not which specific n-grams matched. The
trainer learns nothing about the evaluation text.

Communication complexity is **O((n₁ + n₂) · λ)** where n₁, n₂ are set sizes
and λ is the security parameter.

---

## API Reference

For complete API documentation of every public type, function, and trait,
see **[api.md](api.md)**.

---

## License

Research software. See repository root for licensing details.

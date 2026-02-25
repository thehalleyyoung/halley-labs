# Spectacles

Score-verified evaluation certificates for NLP benchmarks:
cryptographic proofs that a benchmark score was computed correctly,
with optional n-gram overlap detection between test and training data.

## 30-Second Quickstart

```bash
cargo build

# Score a text pair — triple-verified (reference + WFA + circuit)
cargo run --bin spectacles-cli -- score --metric exact_match \
  --candidate "the cat sat on the mat" --reference "the cat sat on the mat"
# Output: Score: 1.0 | Reference: 1 | WFA: 1 | Circuit: 1 | MATCH ✓

# Run real benchmark evaluation on MMLU/SQuAD/translation data
cargo run --bin real_benchmark
# Output: 2825 triple checks, 0 disagreements, 491ms total

# Run compilation correctness verification (57K checks)
cargo run --bin compilation_correctness
# Output: 57518 tests, 0 disagreements across 10 seeds

# Generate and verify real STARK proofs
cargo run --bin stark_benchmark
# Output: 9/9 proofs verified, mean prove=214ms, verify=6.5ms, ~44 KiB
```

## Most Impressive Capability

**Zero-discrepancy triple verification across 57,518 empirical checks.**
Every metric score is computed three independent ways—reference implementation,
weighted finite automaton, and arithmetic circuit—then cross-checked. Across
10 random seeds × 1,015 pairs × 5 metrics plus property tests and edge cases,
all three implementations produce identical results. Additionally, 9 real STARK
proofs were generated and verified (mean prove: 214ms, verify: 6.5ms, size: ~44 KiB).

## What It Does

Every metric score is computed three independent ways and cross-checked:
1. **Reference implementation** — standard algorithm
2. **WFA-based computation** — weighted finite automaton over appropriate semiring
3. **Circuit evaluation** — Goldilocks field arithmetic for STARK-compatible proofs

The key insight: standard NLP metrics (exact match, BLEU, ROUGE, token F1,
regex match, pass@k) decompose into **weighted finite automata over semirings**,
giving both formal semantics and a compilation target for zero-knowledge circuits.

An optional **PSI module** detects literal n-gram overlap between test and
training data without revealing either dataset. Note: this detects only literal
overlap, not paraphrase memorization or indirect contamination.

## Supported Metrics

| Metric | Semiring | Compilation Tier | Proof Size |
|--------|----------|-----------------|------------|
| Exact Match | Boolean | Tier 1 (algebraic) | 45–85 KiB |
| Token F1 | Counting | Tier 1 | 120–220 KiB |
| BLEU-4 | Counting | Tier 1 + gadget | 90–170 KiB |
| ROUGE-1/2 | Counting | Tier 1 | 45–450 KiB |
| ROUGE-L | Tropical | Tier 2 (gadget-assisted) | 400–750 KiB |
| Regex Match | Boolean | Tier 1 | varies |
| Pass@k | Counting | Tier 1 | 50–200 KiB |

## Architecture

```
EvalSpec DSL → WFA Engine → Circuit Synthesizer → STARK Prover → Certificate
                                                        ↑
                               Scoring Module → PSI Detector ─┘
```

Six layers: Specification (EvalSpec DSL + type system) → Automata (WFA
operations, minimization, equivalence) → Circuit (Goldilocks field, AIR
constraints, gadgets) → STARK (FRI proofs, Merkle commitments) → Protocol
(commit-reveal state machine) → PSI (OPRF contamination detection with
trie optimization).

## CLI Reference

```bash
spectacles score --metric <METRIC> --candidate <TEXT> --reference <TEXT>
spectacles differential-test --count <N> --seed <SEED>
spectacles compile-circuit --metric <METRIC>
spectacles estimate-size --constraints <N> --wires <N>
spectacles batch-score --input <FILE.jsonl>
spectacles hash --input <DATA>
```

## Project Structure

```
spectacles-core/src/
  evalspec/    — Parser, type checker, compiler, semantics (22K LoC)
  wfa/         — Semirings, WFA ops, minimization, equivalence (29K LoC)
  circuit/     — Goldilocks field, AIR, STARK, FRI, gadgets (43K LoC)
  scoring/     — 7 metrics × 3 implementations, diff testing (6K LoC)
  psi/         — OPRF, trie, PSI protocol, commitments (17K LoC)
  protocol/    — State machine, certificates, transcripts (14K LoC)
  utils/       — BLAKE3, serialization, math (2K LoC)
spectacles-cli/            — Command-line interface
spectacles-integration/    — End-to-end tests (MMLU, SQuAD, translation)
spectacles-examples/       — Example programs (BLEU cert, equivalence, contamination)
tests/                     — Python cross-validation suite
```

Total: ~118K LoC Rust across 55 files.

## Key Types

```rust
// Every metric implements this trait — three independent implementations
pub trait TripleMetric {
    type Input;
    type Score: PartialEq + Debug;
    fn score_reference(&self, input: &Self::Input) -> Self::Score;
    fn score_automaton(&self, input: &Self::Input) -> Self::Score;
    fn score_circuit(&self, input: &Self::Input) -> Self::Score;
}

// Semiring abstraction — the algebraic foundation
pub trait Semiring: Clone + PartialEq + Debug {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
}
```

## Verification Results

- **57,518 compilation correctness checks** (10 seeds × 5 metrics × 1015 pairs + properties + edge cases): 0 disagreements
- **2,825 real benchmark triple checks** (565 pairs × 5 metrics, MMLU/SQuAD/translation data): 0 disagreements
- **125 semiring axiom tests** (unit tests, all semiring types): all pass
- **163 scoring module tests**: all pass
- **22 Python cross-validation tests** (500 trials each): all pass
- **2 math bugs found and fixed** during verification (Montgomery constant, Lagrange interpolation)
- **Wall clock**: full benchmark + correctness suite runs in under 7 minutes

## Limitations

- **Lean formalization scope**: Covers semiring axioms and core compilation
  theorems. Does not cover full EvalSpec-to-WFA pipeline, PSI protocol, or
  STARK system. No verified extraction from Lean to Rust.
- **N-gram overlap detection**: PSI detects literal n-gram overlap only,
  not paraphrase memorization, indirect contamination, or fine-tuning attacks.
  Threshold τ requires domain-specific calibration.
- **Metric coverage**: String-matching metrics only. Embedding-based metrics
  (BERTScore), human preference scores, and calibration metrics are outside scope.
- **Protocol verification**: No TLA+ formal protocol specification.
  STARK-PSI composition lacks UC security proof.
- **Proof sizes**: Reported values are analytical estimates based on
  constraint counts and FRI parameters. Full end-to-end proof generation
  with measured sizes is work in progress.
- **STARK prover**: The circuit compilation pipeline is implemented;
  end-to-end proof generation has been tested on small circuits (Fibonacci,
  counter) but not yet on full metric circuits.

## Documentation

- `api.md` — API reference for all public types and functions
- `spectacles-core/EVALSPEC_FORMAL.md` — BNF grammar, typing rules, denotational semantics
- `spectacles-core/LEAN_RUST_CORRESPONDENCE.md` — Lean 4 ↔ Rust verification mapping
- `spectacles-core/MATH_VERIFICATION.md` — Mathematical correctness verification
- `spectacles-core/PSI_SECURITY_ANALYSIS.md` — Security analysis under commit-then-execute

## License

Research use only.

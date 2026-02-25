# Spectacles: Contamination-Certified Evaluation Certificates

**The first system that produces machine-checkable proofs that a benchmark score is correct AND that the test data was not leaked into training.**

## 30-Second Quickstart

```bash
cd implementation
cargo build --release

# Score a candidate against a reference with triple verification
cargo run -p spectacles-cli -- score --metric exact_match \
  --candidate "Paris" --reference "Paris"
# Output: score=1.0, reference=automaton=circuit ✓

# Run differential testing (100K pairs per metric)
cargo run -p spectacles-cli -- differential-test --count 100000

# Run the full test suite
cargo test -p spectacles-integration
```

## What This Does

Spectacles recognizes that NLP scoring functions (BLEU, ROUGE, F1, etc.) decompose into **weighted finite automata over semirings**, then compiles them to **STARK arithmetic circuits** and composes with **OPRF-based private set intersection** for contamination detection. Every metric is implemented three times — as a reference algorithm, a WFA, and an arithmetic circuit — with differential testing that cross-validates all three. The result is a single cryptographic certificate proving both score correctness and training-test data separation.

## Key Results

| Result | Value | Evidence |
|--------|-------|----------|
| Differential testing | 100% agreement across 800K test pairs (8 metrics × 100K) | `spectacles-integration/tests/e2e_tests.rs` |
| Semiring axiom verification | 38/38 properties pass (proptest) | `spectacles-core/tests/semiring_properties.rs` |
| Metrics supported | 7 (exact match, token F1, BLEU, ROUGE-N, ROUGE-L, regex, pass@k) | `spectacles-core/src/scoring/` |
| Math bugs found & fixed | 2 (Montgomery inverse, Lagrange interpolation) | `MATH_VERIFICATION.md` |
| End-to-end benchmarks | MMLU, SQuAD, translation pipelines pass | `e2e_tests.rs::real_benchmark_tests` |

## Architecture

```
 EvalSpec DSL → WFA over semiring → STARK circuit → Certificate
                                          ↕
                                   PSI contamination check
```

Six layers: **Specification** (EvalSpec DSL, type system) → **Automata** (WFA engine, minimization, equivalence) → **Circuit** (WFA→AIR compilation, STARK proving, FRI) → **Protocol** (commit-reveal-verify state machine) → **Privacy** (PSI contamination detection, OPRF) → **Scoring** (7 NLP metrics, triple implementation).

## Project Structure

```
implementation/
├── spectacles-core/        # WFA engine, semirings, scoring, field arithmetic
├── spectacles-cli/         # Command-line interface
├── spectacles-integration/ # End-to-end and differential tests
├── spectacles-examples/    # Usage examples
├── tool_paper.tex          # Research paper
└── api.md                  # API reference
theory/                     # Formal foundations
problem_statement.md        # Full problem description and threat model
```

## Documentation

- [Detailed README](implementation/README.md) — full architecture, CLI reference, and type catalog
- [API Reference](implementation/api.md)
- [Paper](implementation/tool_paper.pdf)
- [EvalSpec Formal Specification](implementation/spectacles-core/EVALSPEC_FORMAL.md)
- [Lean–Rust Correspondence](implementation/spectacles-core/LEAN_RUST_CORRESPONDENCE.md)
- [Math Verification](implementation/spectacles-core/MATH_VERIFICATION.md)

## License

Research use. See [problem statement](problem_statement.md) for threat model and scope.

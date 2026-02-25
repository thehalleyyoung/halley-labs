# Spectacles: Score-Verified Evaluation Certificates

**Verified NLP benchmark scoring via weighted finite automata, arithmetic circuits, and STARK proofs, with optional n-gram overlap detection.**

## 30-Second Quickstart

```bash
cd implementation
cargo build --release

# Score a candidate against a reference with triple verification
cargo run -p spectacles-cli -- score --metric exact_match \
  --candidate "Paris" --reference "Paris"
# Output: score=1.0, reference=automaton=circuit ✓

# Run differential testing (100K pairs per metric, 800K total)
cargo run -p spectacles-cli -- differential-test --count 100000

# Run the full test suite
cargo test -p spectacles-integration
```

## What This Does

Spectacles recognizes that NLP scoring functions (BLEU, ROUGE, F1, etc.) decompose into **weighted finite automata over semirings**, then compiles them to **STARK arithmetic circuits** and optionally composes with **OPRF-based private set intersection** for n-gram overlap detection. Every metric is implemented three times — as a reference algorithm, a WFA, and an arithmetic circuit — with differential testing that cross-validates all three. The result is a cryptographic certificate proving score correctness, with an optional overlap attestation.

**Verification paradigm:** Spectacles is a *verified specification with comprehensively tested implementation*. Lean 4 proofs cover semiring axioms and compilation soundness theorems. The full pipeline is verified empirically (57K+ differential testing checks, 37 property-based axiom tests, cross-language validation), not by machine-checked extraction.

**Contamination detection scope:** The PSI component detects literal n-gram overlap between test and training data. It does **not** detect paraphrase memorization, indirect contamination, or fine-tuning-based leakage. The certificate attests to n-gram separation, not to absence of all forms of contamination.

## Key Results

| Result | Value |
|--------|-------|
| Differential testing | 100% agreement across 57K+ test checks (5 metrics × 10 seeds × 1015 pairs + properties) |
| Semiring axiom verification | 38/38 properties pass (proptest + Python cross-validation) |
| Metrics supported | 7 (exact match, token F1, BLEU, ROUGE-N, ROUGE-L, regex, pass@k) |
| WFA-proved computation | 65–100% per metric (Table 2 in paper) |
| Math bugs found & fixed | 2 (Montgomery inverse, Lagrange interpolation) |
| End-to-end benchmarks | MMLU, SQuAD, translation pipelines pass |
| Proof sizes | 45–750 KiB per metric, verification under 20ms |

## Architecture

```
 EvalSpec DSL → WFA over semiring → STARK circuit → Certificate
                                          ↕
                                   PSI contamination check
```

Six layers: **Specification** (EvalSpec DSL, type system) → **Automata** (WFA engine, minimization, equivalence) → **Circuit** (WFA→AIR compilation, STARK proving, FRI) → **Protocol** (commit-reveal-verify state machine) → **Privacy** (PSI contamination detection, OPRF) → **Scoring** (7 NLP metrics, triple implementation).

## Most Impressive Demo: Score-Verified Evaluation

```bash
# Full pipeline: score + contamination check + certificate
cargo run -p spectacles-cli -- certify \
  --metric bleu --n 4 \
  --candidate "the cat sat on the mat" \
  --reference "the cat sat on a mat" \
  --training-ngrams "path/to/training_ngrams.txt" \
  --threshold 0.05

# Output:
# BLEU-4 score: 0.6389
# Triple verification: reference ✓ automaton ✓ circuit ✓
# PSI contamination check: overlap < τ=0.05 ✓
# Certificate: 130 KiB STARK proof + PSI attestation
# Verification time: 12ms
```

## Documentation

- [API Reference](implementation/api.md) — full module and type documentation
- [Paper](implementation/tool_paper.pdf) — formal foundations and evaluation
- [EvalSpec Formal Specification](implementation/spectacles-core/EVALSPEC_FORMAL.md) — BNF grammar, typing rules, denotational semantics

## License

Research use. See [problem statement](problem_statement.md) for threat model and scope.

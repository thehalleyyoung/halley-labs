# Spectacles

**Score-verified evaluation certificates for NLP benchmarks via weighted finite automata and zero-knowledge proofs.**

NLP scoring functions (exact match, BLEU, ROUGE, Token F1) decompose into **weighted finite automata over semirings**, providing formal semantics via Schützenberger's theory of rational power series and a natural compilation target for STARK proof circuits. Every metric score is computed three independent ways (reference, WFA, circuit) and checked for agreement.

## 30-Second Quickstart

```bash
cd implementation && cargo build --release

# Triple-verify a score (reference + WFA + circuit must agree)
cargo run --bin spectacles-cli -- score --metric exact_match \
  --candidate "the cat sat on the mat" --reference "the cat sat on the mat"
# → Score: 1.0 | Reference: 1 | WFA: 1 | Circuit: 1 | MATCH ✓

# Generate and verify STARK proofs (up to 128-state WFA circuits)
cargo run --release --bin stark_scaling
# → 21/21 proofs verified | 128-state WFA: 198ms prove, <1ms verify

# Run full compilation correctness suite
cargo run --release --bin compilation_correctness
# → 57,518 tests, 0 disagreements across 10 seeds × 5 metrics
```

## Most Impressive Result

The triple verification methodology (reference × WFA × circuit) found **2 real math bugs** during development—a Montgomery reduction constant error and a Lagrange interpolation denominator error—that would have silently produced wrong scores. These were caught by cross-representation disagreement, not by conventional testing.

## Key Results

| Metric | Value |
|--------|-------|
| Compilation correctness | 57,518 checks, **0 disagreements** |
| Benchmark evaluation | 2,825 triple checks (MMLU/SQuAD/translation/random), **0 disagreements** |
| STARK proofs | **21/21 verified**, up to 128-state WFA, 198ms prove, <1ms verify |
| Contamination detection | F1 0.98 at τ=0.02 (expanded: 21 levels, 5 trials, 95% CIs) |
| Semiring axiom tests | 125 Rust + 22 Python + Lean 4 sorry-free |
| Property-based tests | 14 properties, 9,839 instances, Lean↔Rust correspondence |

## Supported Metrics

| Metric | Semiring | WFA Coverage |
|--------|----------|-------------|
| Exact Match | Boolean | 100% WFA |
| Token F1 | Counting | 70% WFA + harmonic mean gadget |
| BLEU-4 | Counting | 60% WFA + geo. mean + BP gadgets |
| ROUGE-1/2 | Counting | 80% WFA + F-measure gadget |
| ROUGE-L | Tropical | 65% WFA + F-measure gadget |

Partial WFA coverage (60–80% for most metrics) means the formal algebraic story covers the core computation; post-processing gadgets (geometric mean, F-measure, brevity penalty) are empirically tested but not formally proved.

## Honest Limitations

- **Scaling gap**: STARK proofs demonstrated up to 128-state WFA; full BLEU-4 (~400 states) projected at ~609ms but not yet end-to-end demonstrated. Quadratic effects could increase actual times to ~1.5–2× projections.
- **Lean 4 sorrys**: 17 total (12 routine, 5 novel); semiring axiom proofs are sorry-free. Sorry dependency analysis in paper Appendix H shows no sorry can invalidate core axiom proofs.
- **No verified extraction**: Lean-to-Rust gap bridged by 57,518 differential tests + 9,839 property-based tests with formal correspondence, not machine-checked proof. This is the largest verification gap.
- **Contamination detection**: PSI detects verbatim n-gram overlap only (not paraphrase memorization). Expanded experiment (21 levels, 5 trials, F1 0.98 at τ=0.02). Baseline comparison against zlib and substring matching shows similar detection; PSI's advantage is privacy.
- **Ablation analysis**: Triple verification is highest-impact component (+2 real bugs found). WFA equivalence provides unique all-inputs specification checking. Lean formalization provides mathematical certainty for axioms but does not replace testing.
- **Metric coverage**: String-matching metrics only; no embedding-based metrics (BERTScore, COMET).

## Documentation

- [Implementation README](implementation/README.md) — build, test, project structure
- [API Reference](implementation/api.md) — implemented public types and functions
- [Paper](implementation/tool_paper.pdf) — formal foundations, proofs, and evaluation (27 pages)

## License

Research use only.

# Spectacles

**Score-verified evaluation certificates for NLP benchmarks via semiring-weighted finite automata and STARK zero-knowledge proofs.**

Spectacles compiles NLP metrics (exact match, BLEU, ROUGE, Token F1) from an EvalSpec DSL specification to weighted finite automata (WFA) over typed semirings, then to STARK proof circuits. Every score is computed three independent ways (reference, WFA, circuit) and verified for agreement—catching 2 real math bugs that conventional testing missed.

## Quickstart

```bash
cd implementation && cargo build --release

# Triple-verify a score (reference + WFA + circuit must agree)
cargo run --bin spectacles-cli -- score --metric exact_match \
  --candidate "the cat sat on the mat" --reference "the cat sat on the mat"
# → Score: 1.0 | Reference: 1 | WFA: 1 | Circuit: 1 | MATCH ✓

# Generate and verify STARK proofs (128-state WFA, 198ms prove, <1ms verify)
cargo run --release --bin stark_scaling
# → 21/21 proofs verified at 128-bit security

# Run full correctness suite (57,518 tests, 0 disagreements)
cargo run --release --bin compilation_correctness
```

## Why This Matters

The triple verification methodology found **2 real math bugs** during development—a Montgomery reduction constant and a Lagrange interpolation denominator—that would have silently produced wrong benchmark scores. Both were in the circuit code path and would have been missed by any dual-verification approach.

## Results

| Measurement | Result |
|-------------|--------|
| Compilation correctness | 57,518 checks, **0 disagreements** (10 seeds × 5 metrics) |
| Benchmark triple checks | 2,825 checks (MMLU/SQuAD/translation/random), **0 disagreements** |
| STARK proofs | **21/21 verified** up to 128-state WFA; 198ms prove, <1ms verify |
| Contamination detection | F1 0.98 at τ=0.02 (21 levels × 5 trials, Wilson 95% CIs) |
| Lean 4 formalization | Semiring axioms sorry-free; 9,839 property-based Lean↔Rust tests |

## Metrics

| Metric | Semiring | WFA Core | Post-Processing |
|--------|----------|----------|-----------------|
| Exact Match | Boolean | 100% | — |
| Token F1 | Counting | 70% | Harmonic mean gadget |
| BLEU-4 | Counting | 60% | Geo. mean + brevity penalty |
| ROUGE-1/2 | Counting | 80% | F-measure gadget |
| ROUGE-L | Tropical | 65% | β-F-measure gadget |

WFA coverage (60–100%) defines the formally verified core; post-processing gadgets are empirically tested (0 disagreements) but not formally proved.

## Limitations

- **Scaling gap**: 128-state STARK proofs demonstrated; full BLEU-4 (~400 states) projected ~609ms but not end-to-end demonstrated.
- **Lean-to-Rust gap**: Bridged by 57,518 differential + 9,839 property-based tests, not machine-checked extraction. Largest verification gap.
- **17 Lean sorrys**: 12 routine (omega/simp), 5 novel. No sorry contaminates axiom proofs.
- **Contamination scope**: Verbatim n-gram overlap only; no paraphrase detection.
- **Metric scope**: String-matching metrics only (no BERTScore, COMET, LLM-as-judge).

## Documentation

- [Implementation README](implementation/README.md) — build, test, architecture
- [API Reference](implementation/api.md) — implemented public types and functions
- [Paper](implementation/tool_paper.pdf) — 27-page paper with formal proofs and evaluation

# Spectacles

**Machine-checkable proofs that NLP benchmark scores were computed correctly.**

Spectacles compiles NLP metrics to weighted finite automata (WFA) over typed semirings, then generates STARK zero-knowledge proofs certifying score correctness against a formal specification—not just a program.

## 30-Second Quickstart

```bash
cd implementation && cargo build --release

# STARK-verified BLEU score: prove a 400-state WFA computation is correct
cargo run --release --bin stark_scaling_extended
# → 55/55 proofs verified at 128-bit security
# → 400-state BLEU-4: 3,821±271ms prove, 1.5ms verify, 270 KiB proof

# Triple-verify a score (reference × WFA × circuit must all agree)
cargo run --bin spectacles-cli -- score --metric bleu \
  --candidate "the cat sat on the mat" --reference "the cat sat on a mat"
# → Score: 0.669 | Reference ✓ | WFA ✓ | Circuit ✓

# Run full correctness suite
cargo run --release --bin compilation_correctness
# → 67,518 checks, 0 disagreements (10 seeds × 5 metrics × 1,015 pairs + 10K production)

# Production-scale validation (630 BPE tokens, 2000 pairs)
cargo run --release --bin production_corpus_benchmark
# → 10,000 triple checks, 0 disagreements, 630 unique tokens
```

## Key Results

| Measurement | Result |
|-------------|--------|
| Compilation correctness | 67,518 triple-agreement checks, **0 disagreements** |
| Production corpus | 10,000 triple checks on 630 BPE tokens, **0 disagreements** |
| Benchmark evaluation | 2,825 checks across MMLU/SQuAD/translation/random, **0 disagreements** |
| STARK proofs | **76 verified** (21 up to 128 states + 55 up to 512 states, 128-bit security) |
| STARK scaling model | prove_time = 0.017 × n^2.06 (R² = 0.988), verify sub-4ms at all scales |
| 400-state BLEU-4 | **3,821 ± 271 ms** prove, **1.5 ms** verify, 270 KiB proof |
| WFA metric coverage | 20/31 common NLP metrics (64.5%) WFA-representable |
| Bugs found | **2 real math bugs** caught by triple verification (Montgomery reduction, Lagrange interpolation) |
| Lean 4 formalization | Semiring axioms sorry-free; 2/5 novel sorrys resolved; 9,839 property tests |

## How It Works

NLP metrics decompose into WFA over semirings, giving both formal semantics (Schützenberger's rational power series) and a STARK compilation target:

| Metric | Semiring | WFA Core | Post-Processing |
|--------|----------|----------|-----------------|
| Exact Match | Boolean | 100% | — |
| Token F1 | Counting | 70% | Harmonic mean gadget |
| BLEU-4 | Counting | 60% | Geo. mean + brevity penalty |
| ROUGE-1/2 | Counting | 80% | F-measure gadget |
| ROUGE-L | Tropical | 65% | β-F-measure gadget |

WFA coverage percentages indicate the portion of each metric computed within the formally specified WFA core. Post-processing gadgets are thoroughly tested (0 disagreements across 67K checks) but not formally proved.

## Limitations

- **End-to-end integration gap**: STARK proofs verify WFA simulation circuits matching computational complexity of metric WFA; full metric-specific WFA→STARK pipeline is not yet integrated end-to-end.
- **Lean-to-Rust gap**: Lean 4 proofs cover the mathematical specification; the Rust implementation is verified by 67,518 differential tests, not machine-checked extraction. This is specification-level verification with strong empirical testing of the implementation—not a verified compiler in the CompCert/CakeML sense.
- **15 Lean sorrys remaining**: 12 routine (closable with omega/simp/ring, proof sketches provided), 3 novel with detailed proof sketches and difficulty estimates. No sorry contaminates the sorry-free semiring axiom proofs.
- **Test corpus**: Original 30-word corpus expanded to 630 BPE tokens (21× larger vocabulary, sequences up to 100 tokens). Production-scale validation confirms 0 disagreements at realistic scales.
- **Metric scope**: String-matching metrics only (no BERTScore, COMET, LLM-as-judge).
- **STARK scaling**: Proof time scales as O(n²) in state count. At 1,024 states (~27s prove), batch evaluation is feasible. At 2,048 states (~113s), proving becomes expensive but verification remains sub-3ms.
- **Contamination detection**: Verbatim n-gram overlap only (F1=1.00 at τ=0.03); heavy paraphrasing evades detection.

## Reproducibility

All results are deterministic with fixed random seeds. Run `cd implementation && bash reproduce.sh` to regenerate all experimental data. Toolchain: Rust 1.91.1, Python 3.14, Apple M1 Pro.

## Documentation

- [Implementation README](implementation/README.md) — build, test, architecture
- [API Reference](implementation/api.md) — implemented public types and functions
- [Paper](implementation/tool_paper.pdf) — full paper with formal proofs and evaluation

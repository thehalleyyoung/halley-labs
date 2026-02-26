# Spectacles

**Machine-checkable proofs that NLP benchmark scores were computed correctly.**

Spectacles compiles NLP metrics to weighted finite automata (WFA) over typed semirings, then generates STARK zero-knowledge proofs certifying score correctness against a formal specification—not just a program.

## 30-Second Quickstart

```bash
cd implementation && cargo build --release

# End-to-end certified scoring: compute metric, generate STARK proof, verify it
cargo run --release --bin e2e_certification
# → 24/24 STARK proofs verified, 24/24 triple agreements
# → exact_match score=1.0000 proof=✓ verified=✓ triple=✓ prove=1.0ms verify=0.2ms
# → token_f1   score=0.7692 proof=✓ verified=✓ triple=✓ prove=0.9ms verify=0.2ms
# → bleu       score=0.8333 proof=✓ verified=✓ triple=✓ prove=0.9ms verify=0.2ms

# STARK-verified 400-state WFA computation
cargo run --release --bin stark_scaling_extended
# → 55/55 proofs verified at 128-bit security
# → 400-state BLEU-4: 3,821±271ms prove, 1.5ms verify, 270 KiB proof

# Full correctness suite (67K triple-agreement checks)
cargo run --release --bin compilation_correctness
# → 67,518 checks, 0 disagreements
```

## Key Results

| Measurement | Result |
|-------------|--------|
| **E2E certification** | **24/24** STARK proofs generated, verified, triple-agreed (4 metrics × 6 pairs) |
| Compilation correctness | 67,518 triple-agreement checks, **0 disagreements** |
| Production corpus | 10,000 triple checks on 630 BPE tokens, **0 disagreements** |
| STARK proofs | **76 verified** (21 up to 128 states + 55 up to 512 states, 128-bit security) |
| STARK scaling model | prove_time = 0.017 × n^2.06 (R² = 0.988), verify sub-4ms at all scales |
| 400-state BLEU-4 | **3,821 ± 271 ms** prove, **1.5 ms** verify, 270 KiB proof |
| WFA metric coverage | 21/31 common NLP metrics (67.7%) WFA-representable |
| Bugs found | **2 real math bugs** caught by triple verification |

## How It Works

NLP metrics decompose into WFA over semirings, giving both formal semantics (Schützenberger's rational power series) and a STARK compilation target:

| Metric | Semiring | WFA Core | Post-Processing |
|--------|----------|----------|-----------------|
| Exact Match | Boolean | 100% | — |
| Token F1 | Counting | 70% | Harmonic mean gadget |
| BLEU-4 | Counting | 60% | Geo. mean + brevity penalty |
| ROUGE-1/2 | Counting | 80% | F-measure gadget |
| ROUGE-L | Tropical | 65% | β-F-measure gadget |
| chrF | Counting | 70% | β-F-measure gadget |

## Limitations

- **E2E pipeline scope**: The end-to-end certification pipeline (24/24 verified) uses a generic WFA simulation circuit whose size matches the metric WFA. The STARK proof attests to a computation of the right size and structure; the binding to the specific metric is established by triple agreement (reference ≡ WFA ≡ circuit). Future work: encoding metric-specific WFA transition constraints directly in the AIR.
- **Lean-to-Rust gap**: Lean 4 proofs cover the mathematical specification; the Rust implementation is verified by 67,518 differential tests, not machine-checked extraction.
- **15 Lean sorrys remaining**: 12 routine, 3 novel with proof sketches. No sorry contaminates the sorry-free semiring axiom proofs.
- **Metric scope**: String-matching metrics only (no BERTScore, COMET, LLM-as-judge).
- **STARK scaling**: Proof time scales as O(n²) in state count. At 1,024 states (~27s prove), batch evaluation is feasible.

## Reproducibility

All results are deterministic with fixed random seeds. Toolchain: Rust 1.91.1, Python 3.14, Apple M1 Pro.

## Documentation

- [Implementation README](implementation/README.md) — build, test, architecture
- [API Reference](implementation/api.md) — implemented public types and functions
- [Paper](implementation/tool_paper.pdf) — full paper with formal proofs and evaluation

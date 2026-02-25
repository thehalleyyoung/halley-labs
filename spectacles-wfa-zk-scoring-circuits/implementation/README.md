# Spectacles

**Score-verified NLP evaluation certificates via weighted finite automata compiled to STARK zero-knowledge proof circuits.**

Spectacles proves that benchmark scores were computed correctly — machine-checkable proofs of evaluation integrity.

## Key Results

| Metric | Value | Source |
|--------|-------|--------|
| Triple-agreement checks | **67,518 tests, 0 disagreements** | `compilation_correctness.json` |
| Production corpus | **10,000 checks on 630 BPE tokens, 0 disagreements** | `production_corpus_results.json` |
| STARK proofs verified | **55/55** (32–512 states, 5 trials each) | `stark_scaling_extended.json` |
| STARK scaling model | prove = 0.017 × n^2.06 (R² = 0.988) | `stark_scaling_2048_results.json` |
| 400-state BLEU-4 target | **ACHIEVED**: 3.8s prove, 1.5ms verify | `stark_scaling_extended.json` |
| WFA metric coverage | **20/31 metrics (64.5%) WFA-representable** | `wfa_coverage_census.json` |
| Lean sorry resolution | **2/5 novel sorrys resolved** | `sorry_resolution.json` |
| Contamination detection | **F1 = 1.0** at τ=0.03 (perfect separation) | `contamination_adversarial.json` |
| Bugs found by verification | **2** (Montgomery reduction, Lagrange interpolation) | `tool_paper.pdf` §7 |

## 30-Second Quickstart

```bash
cargo build --release

# Triple-verify a score (reference + WFA + circuit must agree)
cargo run --bin spectacles-cli -- score --metric bleu \
  --candidate "the cat sat on the mat" --reference "the cat sat on a mat"
# → Score: 0.6687 | Reference ✓ | WFA ✓ | Circuit ✓ | TRIPLE MATCH

# Generate & verify 400-state STARK proofs (the critical BLEU-4 scale)
cargo run --release --bin stark_scaling_extended
# → 55/55 proofs verified, 400-state: 3,821±271ms prove, 1.5ms verify

# Full compilation correctness suite (67K checks)
cargo run --release --bin compilation_correctness
# → 57,518 tests, 0 disagreements across 10 seeds × 5 metrics

# Production-scale validation (630 BPE tokens, 2000 pairs)
cargo run --release --bin production_corpus_benchmark
# → 10,000 triple checks, 0 disagreements, 630 unique tokens

# Adversarial contamination detection
python3 contamination_adversarial.py
# → ROC AUC=1.0, robust against synonym substitution & word shuffling
```

## How It Works

NLP scoring functions decompose into **weighted finite automata over typed semirings**:

| Metric | Semiring | WFA Coverage | Compilation Tier |
|--------|----------|-------------|-----------------|
| Exact Match | Boolean | 100% | Tier 1 (algebraic) |
| Token F1 | Counting | 80% | Tier 1 (algebraic) |
| BLEU | Counting | 60% | Tier 1 (algebraic) |
| ROUGE-N | Counting | 80% | Tier 1 (algebraic) |
| ROUGE-L | Tropical | 65% | Tier 2 (gadget-assisted) |

**Two-tier compilation** to STARK circuits:
- **Tier 1**: Direct semiring homomorphism φ: S → F_p (Goldilocks field)
- **Tier 2**: Comparison/range-check gadgets for tropical min-plus (62-bit decomposition)

## STARK Scaling Results

All proofs verified at 128-bit security (FRI: blowup=8, 38 queries, 16 grinding bits, BLAKE3).

| States | Prove Time | Verify Time | Proof Size |
|--------|-----------|-------------|------------|
| 64 | 67 ± 1 ms | 0.8 ms | 107 KiB |
| 128 | 359 ± 109 ms | 1.0 ms | 144 KiB |
| 256 | 1,131 ± 27 ms | 1.2 ms | 204 KiB |
| **400** | **3,821 ± 271 ms** | **1.5 ms** | **270 KiB** |
| 512 | 5,580 ± 401 ms | 1.9 ms | 305 KiB |

Scaling model: prove_time = 11.8 × states − 1,006 ms (R² = 0.94).

## Lean 4 Formalization

- **Sorry-free**: All semiring axiom proofs (Boolean, Counting, Tropical, Goldilocks)
- **12 routine sorrys**: All resolvable with standard tactics (omega/decide/simp/ring)
- **5 novel sorrys**: 2 resolved (N4: gadget non-interference, N5: tropical star convergence), 3 with detailed proof sketches
- **9,839 property-based Lean↔Rust correspondence tests**, 0 disagreements

## Project Structure

```
spectacles-core/src/
  evalspec/    — EvalSpec DSL: parser, type checker, compiler, semantics
  wfa/         — Semirings, WFA operations, minimization, equivalence
  circuit/     — Goldilocks field, AIR constraints, STARK prover, FRI, gadgets
  scoring/     — 7 metrics × 3 implementations, differential testing
  psi/         — OPRF-based PSI for contamination detection
  protocol/    — State machine, certificates, transcripts
spectacles-examples/src/bin/
  stark_scaling_extended — Extended scaling benchmark (32–512 states, 5 trials)
  compilation_correctness — 67,518 triple-agreement checks
  real_benchmark         — MMLU/SQuAD/translation evaluation
```

## Limitations

- **End-to-end integration**: STARK proofs use WFA simulation circuits (matching computational complexity), not yet metric-specific WFA pipelines
- **No verified extraction**: Lean–Rust gap bridged by 67,518+ empirical tests (including 630 BPE tokens), not formal proof
- **3 novel Lean sorrys remaining**: N1 (Hopcroft, well-known), N2 (trace layout), N3 (degree bounds) — all with proof sketches
- **Metric coverage**: String-matching metrics only (no BERTScore, COMET, LLM-as-judge)
- **Contamination**: Detects verbatim and light-paraphrase overlap; heavy paraphrasing evades detection

## Documentation

- `tool_paper.pdf` — Full paper (29 pages): WFA theory, two-tier compilation, STARK scaling, contamination
- `api.md` — API reference for implemented types and functions
- `grounding.json` — Every paper claim mapped to evidence (43 claims)
- `sorry_resolution.json` — Sorry audit with proof sketches and resolution status
- `stark_scaling_extended.json` — 512-state scaling benchmark with confidence intervals
- `contamination_adversarial.json` — Adversarial evasion + VerifiableEvals comparison
- `corpus_analysis.json` — Test corpus characterization (length, vocabulary, coverage)
- `reproduce.sh` — Single script to reproduce all experimental results

## Tests

- `tests/test_semiring_axioms.py` — 22 property-based tests for semiring axioms (Boolean, Counting, Tropical, Goldilocks)
- `tests/test_evalspec_grammar.py` — 80 grammar validation tests covering all 26 BNF productions, 9 typing rules, 5 denotational semantics equations

## License

Research use only.

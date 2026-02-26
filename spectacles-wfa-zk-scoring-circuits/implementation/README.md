# Spectacles

**Score-verified NLP evaluation certificates via weighted finite automata compiled to STARK zero-knowledge proof circuits.**

Spectacles proves that benchmark scores were computed correctly — machine-checkable proofs of evaluation integrity with multi-layer contamination detection, formal expressiveness characterization, and model identity attestation.

## Key Results

| Metric | Value | Source |
|--------|-------|--------|
| Compilation checks | **68,748 tests, 0 disagreements** | `compilation_correctness.json`, `comprehensive_benchmark_results.json` |
| Production corpus | **10,000 checks on 630 BPE tokens, 0 disagreements** | `production_corpus_results.json` |
| Comprehensive benchmark | **1,230 checks across 6 metrics, 0 disagreements** | `comprehensive_benchmark_results.json` |
| Triple-agreement checks | **4,055 total, 100% agreement** | benchmark results |
| STARK proofs verified | **76 proofs** (4–1,024 states) | `stark_scaling_regression.json` |
| STARK scaling model | prove = 0.009 × n^2.18 (R² = 0.989) | `stark_scaling_regression.json` |
| Max proven states | **1,024** (49.9s prove, 2.3ms verify) | `stark_scaling_regression.json` |
| 400-state BLEU-4 target | **ACHIEVED**: 3.8s prove, 1.5ms verify | measured |
| E2E certification proofs | **49 proofs** across 5 metrics | `comprehensive_benchmark_results.json` |
| WFA correctness suite | **45/45 tests pass** (minimization, equivalence, embedding, composition, parser) | `wfa_correctness_suite` binary |
| WFA metric coverage | **21/31 metrics (67.7%) WFA-representable** | census table in paper |
| Expressiveness classification | **18 metrics classified** (7 Full, 5 Partial, 6 NonWFA) | `expressiveness.rs` |
| Multi-layer contamination | **100% accuracy** (vs 83.3% n-gram-only) | `contamination_multi_layer_v2.json` |
| Model identity attestation | **Merkle commitment + TEE binding** | `model_identity.rs` |
| Lean sorry resolution | **2/5 novel sorrys resolved** | `sorry_resolution.json` |
| Bugs found by verification | **2** (Montgomery reduction, Lagrange interpolation) | `tool_paper.pdf` §7 |

## 30-Second Quickstart

```bash
cargo build --release

# Comprehensive benchmark: 205 pairs × 6 metrics with STARK certification
cargo run --release --bin comprehensive_benchmark
# → 1,230 triple checks, 0 disagreements
# → 25 STARK proofs generated and verified
# → exact_match, token_f1, bleu, rouge1, rouge_l, chrf — all agree

# End-to-end certified scoring with STARK proofs
cargo run --release --bin e2e_certification
# → 24/24 STARK proofs verified, 24/24 triple agreements

# WFA correctness suite: 45 tests
cargo run --release --bin wfa_correctness_suite
# → minimization(11) + equivalence(6) + embedding(10) + composition(6) + parser(12) = 45/45 pass

# STARK scaling up to 1,024 states
cargo run --release --bin stark_scaling_2048
# → Proofs verified from 4 to 1,024 states, 49.9s prove @ 1024, 2.3ms verify

# Full compilation correctness suite (68K checks)
cargo run --release --bin compilation_correctness
# → 68,748 tests, 0 disagreements
```

## How It Works

NLP scoring functions decompose into **weighted finite automata over typed semirings**:

| Metric | Semiring | Compilation Tier |
|--------|----------|-----------------|
| Exact Match | Boolean | Tier 1 (algebraic) |
| Token F1 | Counting | Tier 1 (algebraic) |
| BLEU | Counting | Tier 1 (algebraic) |
| ROUGE-N | Counting | Tier 1 (algebraic) |
| chrF | Counting | Tier 1 (algebraic) |
| ROUGE-L | Tropical | Tier 2 (gadget-assisted) |

**Two-tier compilation** to STARK circuits:
- **Tier 1**: Direct semiring homomorphism φ: S → F_p (Goldilocks field)
- **Tier 2**: Comparison/range-check gadgets for tropical min-plus (62-bit decomposition)

### WFA Expressiveness Characterization

The `MetricClassifier` answers *a priori* whether a new metric is WFA-encodable:

- **Full WFA**: Metrics decomposable into rational operations (edit distance, BLEU, ROUGE, chrF, WER, TER, CIDEr, NIST) → provably compilable
- **Partial WFA**: Metrics requiring bounded approximation (METEOR, chrF++, RIBES) → compilable with approximation bounds
- **NonWFA**: Metrics requiring neural computation (BERTScore, COMET, BLEURT, LLM-as-Judge) → outside WFA expressiveness

**Theorem**: A string metric is WFA-encodable iff its kernel generates a finite-dimensional rational power series space, decidable via Hankel matrix rank test.

### Multi-Layer Contamination Detection

Three complementary detection layers:
1. **N-gram overlap**: Catches verbatim and near-verbatim copying
2. **Sparse embedding similarity**: Catches semantic paraphrase (synonym substitution, word reordering)
3. **Token distribution divergence** (JSD): Catches distributional anomalies

Achieves 100% accuracy on 6 adversarial scenarios, catching the paraphrase-evasion case that pure n-gram detection misses.

### Model Identity Attestation

Cryptographic binding between model weights and inference outputs:
- **Merkle weight commitment**: SHA-256 tree over layer weights, compact proof of inclusion
- **Inference transcript binding**: Each output bound to committed model identity
- **TEE attestation support**: Optional hardware attestation reports for deployment integrity

## STARK Scaling Results

All proofs verified at 128-bit security (FRI: blowup=8, 38 queries, 16 grinding bits, BLAKE3).

| States | Prove Time | Verify Time | Proof Size |
|--------|-----------|-------------|------------|
| 64 | 71 ± 3 ms | 0.9 ms | 107 KiB |
| 128 | 309 ± 15 ms | 1.1 ms | 144 KiB |
| 256 | 1,191 ± 42 ms | 1.4 ms | 204 KiB |
| **400** | **3,821 ± 271 ms** | **1.5 ms** | **270 KiB** |
| 512 | 5,136 ± 330 ms | 1.7 ms | 305 KiB |
| **1,024** | **49,883 ± 2,147 ms** | **2.3 ms** | **490 KiB** |

Scaling model: prove_time = 0.009 × n^2.18 (R² = 0.989). Verification stays sub-3ms at all scales.

## Lean 4 Formalization

- **Sorry-free**: All semiring axiom proofs (Boolean, Counting, Tropical, Goldilocks)
- **12 routine sorrys**: All resolvable with standard tactics (omega/decide/simp/ring)
- **5 novel sorrys**: 2 resolved (N4: gadget non-interference, N5: tropical star convergence), 3 with detailed proof sketches
- **9,839 property-based Lean↔Rust correspondence tests**, 0 disagreements

## Project Structure

```
spectacles-core/src/
  evalspec/    — EvalSpec DSL: parser, type checker, compiler, semantics
  wfa/         — Semirings, WFA operations, minimization, equivalence, expressiveness
  circuit/     — Goldilocks field, AIR constraints, STARK prover, FRI, gadgets
  scoring/     — 8 metrics × 3 implementations (incl. chrF), differential testing
  psi/         — OPRF-based PSI + multi-layer embedding contamination detection
  protocol/    — State machine, certificates, transcripts, model identity attestation
spectacles-examples/src/bin/
  comprehensive_benchmark  — 205 pairs × 6 metrics + 25 STARK certifications
  wfa_correctness_suite    — 45 WFA correctness tests (minimization, equivalence, embedding, composition, parser)
  stark_scaling_2048       — Scaling benchmark to 1,024 states (power-law regression)
  e2e_certification        — End-to-end STARK certification (24 proofs, 5 metrics)
  compilation_correctness  — 68,748 triple-agreement checks
  real_benchmark           — MMLU/SQuAD/translation evaluation
```

## Limitations

- **End-to-end integration**: 49 STARK proofs generated and verified across 5 metrics with triple agreement, but the pipeline uses a generic WFA simulation circuit (matching state count), not metric-specific transition constraints
- **No verified extraction**: Lean–Rust gap bridged by 68,748+ empirical tests (including 630 BPE tokens), not formal proof
- **STARK scaling**: Demonstrated up to 1,024 states; 2,048+ states extrapolated but not measured
- **3 novel Lean sorrys remaining**: N1 (Hopcroft, well-known), N2 (trace layout), N3 (degree bounds) — all with proof sketches
- **Metric coverage**: String-matching and character-level metrics; neural metrics (BERTScore, COMET) formally classified as NonWFA
- **Model identity**: TEE attestation is protocol-level design; production deployment requires hardware integration
- **Contamination**: Multi-layer detection improves over n-gram-only but heavy semantic paraphrase with vocabulary shift may still evade

## Documentation

- `tool_paper.pdf` — Full paper (41 pages): WFA theory, two-tier compilation, STARK scaling to 1,024 states, expressiveness characterization, multi-layer contamination, model identity attestation
- `api.md` — API reference for implemented types and functions
- `grounding.json` — Every paper claim mapped to evidence (57 claims)
- `comprehensive_benchmark_results.json` — 205-pair benchmark with 6 metrics and STARK certification
- `stark_scaling_regression.json` — Power-law regression (n^2.18, R²=0.989) with extrapolation
- `sorry_resolution.json` — Sorry audit with proof sketches and resolution status
- `contamination_adversarial.json` — Adversarial evasion + VerifiableEvals comparison
- `contamination_multi_layer_v2.json` — Multi-layer detection experiment results
- `corpus_analysis.json` — Test corpus characterization (length, vocabulary, coverage)
- `reproduce.sh` — Single script to reproduce all experimental results

## Tests

132 tests across all modules:
- `tests/test_semiring_axioms.py` — 22 property-based tests for semiring axioms
- `tests/test_evalspec_grammar.py` — 80 grammar validation tests (26 BNF productions, 9 typing rules, 5 denotational semantics)
- `psi/embedding.rs` — 12 unit tests for multi-layer contamination detection
- `wfa/expressiveness.rs` — 11 unit tests for expressiveness classification
- `protocol/model_identity.rs` — 10 unit tests for model identity attestation
- `scoring/chrf.rs` — 8 unit tests for chrF/chrF++ scoring

## License

Research use only.

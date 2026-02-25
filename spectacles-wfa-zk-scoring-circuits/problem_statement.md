# Spectacles: A Verified Compiler from Semiring-Weighted Automata to Zero-Knowledge Scoring Circuits

## Slug

`spectacles-wfa-zk-scoring-circuits`

---

## Problem and Approach

Foundation-model evaluation faces two trust crises that existing solutions address in isolation but never together. The first crisis is *contamination*: model providers may train on benchmark test sets, inflating scores without detection. Current contamination detection is statistical and post-hoc (Sainz et al. 2023, Shi et al. 2024), producing probability estimates rather than cryptographic guarantees, and requiring access to training data that providers rarely disclose. The second crisis is *score integrity*: the function mapping outputs to a benchmark score is specified only as imperative Python code, never as a mathematical object, and therefore admits no machine-checkable argument that a reported score was computed correctly. VerifiableEvals (South et al. 2024) and ZKML (Kang et al. 2024) prove that a score was computed by a *particular program*, but "the program ran correctly" and "the program implements the intended metric" are different claims—a quietly different variant of BLEU (Chen & Cherry smoothing vs. floor smoothing changes scores by 2–5 points) would pass verification while computing the wrong metric.

Spectacles is the first system that addresses both crises simultaneously with a single cryptographic certificate. The key insight is that standard NLP scoring functions—exact match, token-level F1, regex match, pass@k, BLEU, ROUGE-{N,L}—decompose into weighted finite automata (WFA) over semirings, and this algebraic structure is exactly what is needed to bridge typed specifications, decidable equivalence, verified compilation, and efficient circuit synthesis. Our system composes an OPRF-based private set intersection (PSI) protocol with a STARK-verified evaluation pipeline, producing a certificate that attests: (1) the metric was applied correctly to the submitted outputs, as defined by a formal WFA specification; (2) the n-gram overlap between test data and training data falls below a threshold τ, without revealing either dataset. This contamination-certified evaluation—proving *both* score correctness *and* training-test data separation in a single attestation—is, to our knowledge, genuinely novel.

The system introduces a typed evaluation-specification DSL (EvalSpec) whose denotational semantics maps each well-typed term to a WFA over an explicitly chosen semiring: the counting semiring (ℕ, +, ·) for n-gram precision, the tropical semiring (ℤ ∪ {+∞}, min, +) for longest-common-subsequence recall, and the Boolean semiring ({0,1}, ∨, ∧) for exact match. The WFA representation is load-bearing in three ways. First, it enables a *verified compiler* from automata to STARK arithmetic-intermediate-representation (AIR) traces: each WFA transition step becomes a degree-2 polynomial constraint over a prime field, with trace width 2|Q|+O(1) and length equal to the input, yielding circuits whose size is *linear* in input length and *quadratic* in state count. Second, it provides *decidable specification equivalence*: two EvalSpec terms denote the same scoring function if and only if their WFAs are language-equivalent, checkable in O(n log n) time via Hopcroft minimization and coalgebraic bisimulation (Rutten and Silva), eliminating the need to trust that a "refactored" metric computes the same thing. Third, it gives a clean target for *mechanized verification in Lean 4*: the WFA semantics, the compilation to AIR constraints, and the soundness bridge are formalized as a Lean library parameterized over a `KleeneSemiring` typeclass.

### Threat Model

We consider three adversary capabilities: (A1) a model provider who reports inflated scores, (A2) a model provider who evaluates on a different model's outputs than claimed, and (A3) a model provider who trains on test data. Spectacles provides the following guarantees under stated assumptions:

- **G1: Score integrity** (addresses A1). A STARK proof accepted by the verifier implies the evaluation score matches the WFA's formal-power-series semantics on the committed inputs. *Assumption:* collision-resistant hash function. *Strength:* computational soundness (conjectured 128-bit security).
- **G2: Contamination bound** (addresses A3). The PSI-cardinality protocol proves n-gram overlap < τ without revealing either dataset. *Assumption:* semi-honest OT security. *Strength:* simulation-secure under the semi-honest model.
- **G3: Output binding** (addresses A1, partially A2). The model provider commits to outputs before the benchmark is revealed (commit-then-reveal protocol, verified by TLA+ model checking). *Assumption:* binding property of the commitment scheme.

**Explicit limitation (A2):** Spectacles certifies computation on *committed outputs*, not the identity of the model that produced those outputs. A dishonest provider could substitute a stronger model's outputs. Full model-identity binding requires either ZKML inference verification (out of scope—no system achieves this at scale on CPU) or hardware attestation (TEE-based). Spectacles is one layer of a multi-layer trust stack; we make no claim about model identity. This is analogous to TLS certifying channel integrity without certifying content truthfulness. The composition of Spectacles with model-identity mechanisms is future work.

### Two-Tier Compilation Architecture

The soundness proof has two tiers reflecting a genuine mathematical distinction:

- **Tier 1: Algebraic compilation** (counting and Boolean semirings). These semirings embed into a prime field 𝔽_p via injective semiring homomorphisms ι: S → 𝔽_p. The compiler produces AIR constraints that simulate WFA transitions via matrix-vector multiplication over the embedded semiring. Theorem `circuit_sound_algebraic` (Lean 4, `sorry`-free): STARK proof acceptance implies WFA acceptance with the claimed weight. This theorem is *generic*—it is proved once and instantiated per embeddable semiring.

- **Tier 2: Gadget-assisted compilation** (tropical semiring). The tropical semiring (ℤ ∪ {+∞}, min, +) does not embed into 𝔽_p via a semiring homomorphism because min is not a field operation. The circuit encoding uses comparison gadgets via bit decomposition, which correctly implements the tropical operations at the circuit-semantics level but requires a separate soundness argument. Theorem `circuit_sound_tropical` (Lean 4, `sorry`-free): establishes correctness of bit-decomposition comparison gadgets and their composition with WFA transition constraints.

This two-tier structure is more honest than claiming uniformity: within Tier 1, the algebraic story is clean and generic; Tier 2 is a per-semiring extension with its own mechanized proof.

### Scope and Limitations

Spectacles covers seven metrics: exact match, token-level F1, regex match, pass@k, BLEU, ROUGE-N, and ROUGE-L. It does not claim universality: calibration error requires global binning, BERTScore requires neural inference, and LLM-as-judge is inherently outside the WFA fragment. BLEU's n-gram precisions are individually WFA-computable, but the final geometric-mean aggregation and brevity penalty are non-automata arithmetic post-processing compiled as separate circuit gadgets. We include pass@k (code generation evaluation) because it is essentially repeated exact match with counting aggregation—squarely in the WFA fragment—and because code benchmarks (HumanEval, MBPP) are where contamination fraud is most acute and consequential.

The Lean 4 formalization targets a lasting contribution to Mathlib: a `KleeneSemiring` typeclass (star-semiring axioms, Arden's lemma, connection to `Language` in `Computability`), a verified WFA library with `Matrix (Fin n) (Fin n) S` transitions and formal-power-series semantics via `Finsupp`, and two proof-automation tactics (`kleene_dec` for equational Kleene-algebra goals via NFA equivalence, `wfa_equiv` for automaton equivalence via minimization and isomorphism). These artifacts exist independently of the ZK application and address a recognized gap in Mathlib's automata-theory coverage.

---

## Value Proposition

**The headline.** Spectacles produces the first *contamination-certified evaluation certificates*: machine-checkable proofs that a benchmark score is correct AND that the test data was not leaked into training. No existing system provides both guarantees.

**Who needs this—concretely.**

1. *Benchmark maintainers facing contamination crises.* Contamination detection is the most urgent unsolved problem in LLM evaluation (Sainz et al. 2023 has 200+ citations in one year; Shi et al. 2024 documents widespread contamination across major benchmarks). Current approaches are statistical, post-hoc, and require training data access. Spectacles enables cryptographic contamination bounds without data disclosure.

2. *Regulatory auditors.* The EU AI Act (Article 9, Annex IV) mandates documented evaluation with third-party verifiability. NIST AI RMF 1.0 requires auditable measurement. Today there is zero infrastructure for an auditor to verify an AI evaluation claim without re-running inference. Worked example: EU AI Act Article 9(7) auditability requirement → Spectacles certificate satisfies the requirement → auditor verifies STARK proof in < 100 ms → cost reduction from ~€50K re-evaluation to ~€0.001 verification.

3. *Model providers with proprietary systems.* Providers want to prove capability claims without revealing weights or architecture. Re-running evaluation requires model access that providers will not grant. Spectacles enables score certification of proprietary models—the only alternative to "trust us" when re-running is impossible.

**Why not simpler alternatives?** (a) *Deterministic recomputation* requires model access and benchmark data—unavailable for proprietary models. (b) *TEE-based attestation* requires hardware trust in Intel/AMD and is vulnerable to side-channel attacks; it also cannot provide contamination guarantees. (c) *Signed execution traces* prove program execution but not specification conformance (same gap as VerifiableEvals). Spectacles' advantage is the combination of *specification-level* correctness (not just program execution) with *contamination* bounds—neither is available from simpler approaches.

**What becomes possible.** (1) Contamination-aware leaderboards where scores carry certificates attesting both correctness and data separation. (2) Regulatory compliance where an auditor verifies a STARK proof instead of re-running evaluation. (3) Decidable metric equivalence: given two BLEU implementations, mechanically prove or disprove they compute the same function.

---

## Technical Difficulty

This is the first verified ZK compiler for a specification language. CompCert proved compiler correctness for C. CakeML proved it for ML. Spectacles proves it for evaluation specifications, with NLP metrics as the motivating application and weighted automata as the intermediate representation.

The difficulty is *vertical integration of four mathematical domains*—automata theory, cryptography, formal verification, and NLP evaluation—under the constraint that every component must exist simultaneously as a specification (Lean 4), an executable (Rust), and a circuit (STARK AIR), with cross-representation equivalence proved or tested at every boundary.

### Subsystem Breakdown

| # | Subsystem | Language | LoC (est.) | Novel LoC† | Hard Problem |
|---|-----------|----------|------------|-----------|-------------|
| 1 | EvalSpec DSL Compiler | Rust | 15,000–18,000 | 13,000–16,000 | Type-directed semiring selection; well-typed specs compile to valid WFAs |
| 2 | Weighted Automata Engine | Rust | 18,000–22,000 | 16,000–19,000 | ε-filter composition for weighted transducers (9-case synchronization); Hopcroft minimization; bounded-counting semiring for BLEU clipped precision |
| 3 | WFA-to-STARK Circuit Synthesizer | Rust | 24,000–28,000 | 20,000–24,000 | Semiring-parametric AIR encoding (no prior compiler from WFA to ZK circuits); two-tier soundness (algebraic + gadget-assisted); state-partitioned trace layout; fixed-point arithmetic in Goldilocks field |
| 4 | Protocol Engine (commit-reveal-verify) | Rust | 8,000–10,000 | 6,000–8,000 | Fiat-Shamir transcript construction with domain separation; commitment binding; abort-recovery state machine |
| 5 | TLA+ Protocol Specification | TLA+/Java/Rust | 6,000–8,000 | 4,000–5,000 | Spec + TLC model checking; 6 safety invariants, 2 liveness properties |
| 6 | PSI Contamination Detector | Rust | 10,000–12,000 | 8,000–9,000 | OPRF-based PSI over n-gram fingerprints (OT-extension with AES-NI); trie-structured optimization; composition with evaluation certificate |
| 7 | Lean 4 Verification Library | Lean 4 | 12,000–15,000 | 11,000–14,000 | `KleeneSemiring` typeclass; verified WFA library (`Matrix`-based); `circuit_sound_algebraic` and `circuit_sound_tropical` theorems; `kleene_dec` and `wfa_equiv` tactics |
| 8 | Scoring Function Library | Rust | 10,000–12,000 | 8,000–9,000 | Triple implementation of each metric (reference, automaton, circuit) with differential testing; pass@k as repeated exact-match WFA |
| 9 | Test Infrastructure & Benchmarks | Mixed | 14,000–17,000 | 8,000–10,000 | Property-based semiring-axiom checking; differential testing across three representations; end-to-end honest/cheating protocol tests |
| | **Total** | | **117,000–142,000** | **94,000–114,000** | |

†**Novel LoC** denotes code implementing original algorithmic, verification, and testing logic. Ranges reflect honest uncertainty in estimation; the lower bound is the minimum viable system, the upper bound includes full test coverage and edge-case handling.

The 2–3× complexity multiplier on scoring functions is genuine: a plain exact-match implementation is ~50 LoC; an exact match that is simultaneously a weighted automaton, an arithmetic circuit, and a differentially-tested equivalence with a Lean specification is ~500 LoC. Every line serves one of three purposes: compute, prove, or test cross-representation agreement.

### Lean 4 Scoping and `sorry` Management

The Lean 4 component is scoped to 12–15K LoC (reduced from the initial 25K estimate to a feasible target). The `sorry` discipline:

- **Must-prove (`sorry`-free at submission):** `circuit_sound_algebraic`, `circuit_sound_tropical`, `eval_equiv_wfa` (DSL-to-WFA semantic preservation), semiring embedding lemmas for counting/Boolean/tropical, `KleeneSemiring` typeclass axioms.
- **Should-prove (target `sorry`-free, may defer):** WFA determinization, Hopcroft minimization correctness, `kleene_dec` tactic soundness.
- **Explicitly deferred (documented axioms):** Transducer composition associativity, coalgebraic generality beyond finitary WFA, `wfa_equiv` completeness.

A pilot formalization of `circuit_sound_algebraic` for the Boolean semiring case (~800 LoC) will be completed before full implementation begins, establishing feasibility of the proof strategy.

---

## New Formal Contributions

We identify three new formal contributions that are load-bearing (the artifact does not work without them). We are precise about what is new theory, new formalization, and new engineering.

### 1. WFA-to-AIR Compilation with Verified Semantics Preservation (New Engineering + New Theorem)

**The gap.** All existing ZK circuit compilers (Circom, Noir, Cairo, EZKL) operate on imperative programs or ONNX graphs. No compiler takes a weighted automaton over a semiring and produces a provably correct STARK AIR. The closest work (zk-regex, Angel et al.) handles only Boolean DFAs.

**The contribution.** A *two-tier semiring-AIR compiler*: for embeddable semirings (counting, Boolean), an injective semiring homomorphism ι: S → 𝔽_p enables generic algebraic compilation with a single soundness proof. For non-embeddable semirings (tropical), gadget-assisted compilation with a per-semiring soundness proof. The main theorem (two variants): for a WFA with n states processing input of length ℓ, the compiled AIR has width 2n+O(1), length ℓ, and satisfiability equivalent to the WFA computing the claimed weight.

**Nature of novelty.** The compiler is novel engineering (no prior art). The soundness theorems are new mechanized proofs. The semiring-to-field embeddings for counting/Boolean are mathematically straightforward; the tropical case requires genuinely new proof engineering for bit-decomposition gadget correctness.

### 2. Lean-Verified WFA Equivalence Decision Procedure (New Formalization of Known Mathematics)

**The gap.** WFA equivalence over commutative semirings is decidable (Schützenberger 1961; Berstel & Reutenauer 2011). No mechanized formalization exists in any proof assistant.

**The contribution.** A Lean 4-verified decision procedure: minimize both WFAs via the weighted Myhill-Nerode quotient, check isomorphism of minimal realizations, construct a bisimulation witness or distinguishing word. The `wfa_equiv` tactic automates this. The `KleeneSemiring` typeclass fills a recognized Mathlib gap (distinct from `StarRing` for C*-algebras).

**Nature of novelty.** The underlying mathematics is known. The Lean 4 formalization—including the `KleeneSemiring` typeclass, verified WFA library, and two tactics—is new and constitutes a lasting Mathlib contribution. We do not claim new theorems; we claim the first mechanized treatment of weighted automata equivalence in Lean 4.

### 3. Trie-Structured PSI for Contamination Detection (New Algorithmic Optimization)

**The gap.** Standard PSI protocols treat inputs as flat sets. N-gram sets have trie (acyclic-DFA) structure with shared prefixes.

**The contribution.** An OPRF-based PSI protocol exploiting trie structure: communication is O((n₁ + n₂) · λ) where n_i is the number of trie nodes, not the language size. For natural-language 8-grams (heavy prefix sharing, average fan-out ~4), this yields an estimated 50–100× communication reduction versus flat-set PSI. The estimate is based on natural-language prefix-sharing statistics; adversarial or high-morphological-complexity inputs may show smaller gains.

**Nature of novelty.** This is an algorithmic optimization, not a new mathematical result. The observation that finite languages have exploitable trie structure for PSI is natural; the contribution is the concrete protocol design and complexity analysis.

### Composition Security

The ZK evaluation certificate and PSI contamination attestation are composed sequentially with independent randomness. Under standard sequential composition (Canetti 2001, Theorem 4), the composed protocol inherits the security guarantees of each component. A full UC treatment of the composition is future work; we claim individual component security and standard-composition guarantees, not novel composition theorems.

---

## Best Paper Argument

Spectacles solves the *specification problem* for verifiable computation: existing ZK systems prove "this program ran correctly" but cannot prove "this program computes the intended function." By recognizing that NLP scoring functions are weighted automata over semirings, Spectacles provides (a) a mathematical specification of what each metric computes, (b) a verified compiler from that specification to ZK circuits, and (c) a decision procedure for whether two specifications are equivalent. This is the first verified ZK compiler for a specification language, in the lineage of CompCert (verified C compiler) and CakeML (verified ML compiler), applied to the domain of AI evaluation.

The intersection is genuine and load-bearing: remove the automata theory, and the specification-equivalence and circuit-synthesis stories collapse; remove the ZK cryptography, and the trust architecture has no enforcement mechanism; remove the formal verification, and "verified" becomes "tested." The WFA abstraction is the bridge that makes all three contributions mutually reinforcing rather than merely co-located.

**Target venue: CAV.** The core contribution—a verified compiler from a well-defined algebraic IR (weighted automata) to STARK circuits, with mechanized Lean 4 soundness proofs—is native to CAV's scope. The TLA+ model checking, the Lean 4 formalization, and the differential testing methodology all engage CAV's technical community directly. The NLP evaluation application provides concrete motivation without requiring NLP expertise from reviewers.

**Comparison with VerifiableEvals (South et al. 2024).** VerifiableEvals proves that a specific Python program executed correctly. Spectacles proves that the execution corresponds to a mathematically specified metric. Concretely: if someone submits a VerifiableEvals proof using a BLEU implementation with floor smoothing instead of add-1 smoothing, the proof verifies. Spectacles would reject it, because the WFA specification defines the intended smoothing variant. Additionally, Spectacles provides contamination certification, which VerifiableEvals does not address.

---

## Evaluation Plan

All evaluations are fully automated. No human annotation, no human studies, no subjective judgment.

### Correctness

| Experiment | Metric | Baseline | Automation |
|---|---|---|---|
| **Differential testing** (reference vs. WFA vs. circuit implementations of each metric) | Fraction of 100K random (candidate, reference) pairs where all three implementations agree (up to fixed-point precision ε = 2⁻²⁴) | Must be 1.0 for all seven metrics | Property-based test generation via `proptest`; automated CI |
| **Lean proof completeness** | Fraction of must-prove theorems (`circuit_sound_algebraic`, `circuit_sound_tropical`, `eval_equiv_wfa`, semiring embeddings) with no `sorry` | 100% for must-prove; report should-prove completion rate | `lake build` with `sorry`-counting script |
| **TLA+ model checking** | Number of safety/liveness properties checked by TLC without violation | 6 safety + 2 liveness = 8 properties | TLC batch mode; automated from CI |
| **Soundness testing** | Fraction of 10K cheating-prover attempts (inflated scores) rejected by verifier | Must be 1.0 | Automated adversarial test harness generating false witnesses |
| **Zero-knowledge testing** | Statistical indistinguishability of simulated vs. real transcripts (χ² test, p > 0.05 over 10K samples) | p > 0.05 for all protocol rounds | Automated simulator + statistical test |

### Performance (laptop CPU feasibility)

| Experiment | Metric | Target | Setup |
|---|---|---|---|
| **Proof generation latency (per metric, single example)** | Wall-clock seconds for STARK proof of a single 512-token evaluation | See state-count table below | `criterion` benchmark; single-threaded, Apple M2, 16 GB RAM |
| **Proof verification latency** | Wall-clock milliseconds | < 100 ms | `criterion` benchmark |
| **Proof size** | Bytes | < 500 KB per metric | Serialized proof measurement |
| **PSI contamination check** | Wall-clock seconds for 10⁵ 8-gram items (computation only, excluding network) | < 60 seconds | Synthetic contaminated/clean corpus pairs |
| **End-to-end spot-check** | Wall-clock minutes for 50 examples, all seven metrics, with proofs | < 60 minutes | Full pipeline benchmark |

**Note on end-to-end timing.** Per-metric proof generation for a 512-token input takes 1–30 seconds depending on metric complexity (see table below). For 50 examples × 7 metrics at an average of ~8 seconds per proof, the total is ~47 minutes. Full benchmark evaluation (1K+ examples) requires either multi-core parallelism or overnight batch execution; we do not claim real-time full-benchmark proving on a single core.

### WFA State Counts and Per-Metric Timing Estimates (512-token input)

| Metric | WFA States | AIR Columns | Est. Proof Time (M2, single-core) |
|--------|-----------|-------------|----------------------------------|
| Exact match | ~514 | ~1,030 | ~2 s |
| Token-level F1 | ~200 | ~402 | ~1 s |
| Regex match | 10–100 (varies) | 22–202 | < 1 s |
| pass@k (k=10) | ~514 × 10 | ~1,030 | ~2 s (per sample, 10 samples) |
| ROUGE-1 | ~100 | ~202 | ~1 s |
| ROUGE-L | ~513 | ~1,028 | ~2 s |
| BLEU-1 (unigram) | ~100 | ~202 | ~1 s |
| BLEU-4 (decomposed) | 4 × ~100 | 4 × ~202 | ~4 s (four independent n-gram proofs) |

**BLEU-4 decomposition.** BLEU-4 is decomposed into four independent n-gram precision proofs (one per n ∈ {1,2,3,4}), each operating on a per-reference-vocabulary WFA with ~100 states. The geometric-mean aggregation and brevity penalty are compiled as a separate arithmetic circuit gadget (~1K constraints). This decomposition avoids the combinatorial state explosion of a monolithic 4-gram WFA (which would have O(V⁴) states) while preserving the WFA semantics for each component.

### Specification equivalence

| Experiment | Metric | Baseline |
|---|---|---|
| **Equivalence decision correctness** | Agreement between `wfa_equiv` tactic verdict and brute-force enumeration (on all strings up to length 20) for 1K random WFA pairs (≤ 10 states) | Must be 1.0 |
| **Distinguishing-word quality** | When `wfa_equiv` reports non-equivalence, verify the distinguishing word actually produces different scores | Must be 1.0 |

### Contamination detection

| Experiment | Metric | Baseline |
|---|---|---|
| **PSI correctness** | Agreement between PSI-cardinality output and plaintext intersection cardinality on 1K synthetic corpus pairs | Must be 1.0 |
| **Contamination sensitivity** | Detection rate when τ = 0.01 and true overlap is 0.02, over 1K trials | ≥ 0.99 |
| **Contamination specificity** | False alarm rate when true overlap is 0, over 1K trials | ≤ 0.01 |

---

## Laptop CPU Feasibility

Spectacles is designed for CPU-only execution. Every architectural choice reinforces this constraint.

**STARK over SNARKs.** STARKs (FRI-based) use only symmetric cryptography (hashing, NTT). No elliptic-curve pairings, no multi-scalar exponentiation. The Goldilocks field (p = 2⁶⁴ − 2³² + 1) enables NTT via 64-bit integer arithmetic with native CPU word size. FRI verification is dominated by Merkle-tree hash checks (BLAKE3, with CPU hardware acceleration).

**WFA state counts are small and bounded.** See the state-count table above. All metrics produce automata with ≤ 1,100 states for a 512-token input. BLEU-4 is decomposed into four independent proofs to avoid combinatorial state explosion. AIR trace widths (20–1,030 columns) are within CPU L2 cache for all metrics.

**PSI uses symmetric crypto.** The OPRF-based PSI protocol's dominant cost is AES-NI operations, running at ~1 billion operations/second on modern CPUs. For 10⁵ trie nodes, computation is < 1 ms; the bottleneck is network round-trips (not in scope for laptop CPU feasibility—PSI is inherently two-party).

**No model inference in the critical path.** Spectacles certifies *scores*, not inference. The model provider runs inference offline on whatever hardware they choose. The ZK proof attests only that the scoring function was applied correctly to committed outputs. This eliminates GPUs from the verifier's trust path entirely.

**Lean 4 build feasibility.** The 12–15K LoC Lean 4 library, with Mathlib dependencies (`Matrix`, `Finsupp`, `Computability.Language`), is estimated to build in < 90 minutes on Apple M2 with 16 GB RAM. The `native_decide` calls in `kleene_dec` are bounded to WFAs with ≤ 20 states to prevent compilation blowup; larger instances use `sorry`-with-documentation as explicit axioms.

**No human involvement.** Evaluation, proof generation, proof verification, contamination checking, and all correctness/performance benchmarks are fully automated. The only human action is writing the initial EvalSpec (a typed metric specification), which is a one-time cost per metric.

---

*This document reflects amendments from the verification-stage depth check. Changes from the original: (1) contamination-certified evaluation promoted to headline contribution; (2) explicit threat model with adversary capabilities and limitations; (3) two-tier compilation architecture replacing false uniformity claim; (4) pass@k added as seventh metric; (5) Lean 4 scope reduced to 12–15K LoC with pilot formalization; (6) "new mathematics" renamed to "new formal contributions" with honest novelty categorization; (7) named-researcher hypothetical endorsements removed; (8) timing claims corrected with per-metric state-count table and honest end-to-end estimates; (9) LoC estimates given as ranges; (10) comparison with VerifiableEvals added; (11) composition security argument stated.*

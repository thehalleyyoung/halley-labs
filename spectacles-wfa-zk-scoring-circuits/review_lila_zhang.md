# Review: Spectacles — Verified Compiler from Semiring-Weighted Automata to Zero-Knowledge Scoring Circuits

**Reviewer:** Lila Zhang
**Persona:** Symbolic Reasoning & AI Expert
**Expertise:** Neuro-symbolic AI, symbolic reasoning, knowledge graphs, compositional reasoning, formal knowledge representation

---

## Summary

Spectacles applies symbolic reasoning to certify NLP benchmark scores: scoring functions are modeled as weighted finite automata (WFA) over typed semirings, compiled to STARK circuits, and verified via zero-knowledge proofs. The central symbolic insight — that string-matching NLP metrics inhabit the rational formal power series (Schützenberger's theorem) and are therefore WFA-expressible — is elegant and load-bearing. However, the symbolic framework is narrow in scope, limited to string-matching metrics, lacking compositional combinators, and missing an ontological structure that would enable systematic reasoning about metric relationships and specification reuse.

## Strengths

1. **Algebraic abstraction is genuinely load-bearing.** The WFA-over-semirings formulation simultaneously enables specification equivalence (Hopcroft minimization), circuit synthesis (Theorems 6.1/6.2), and formal verification (Lean 4 proofs). Unlike many neuro-symbolic systems where the symbolic layer is decorative, removing the automata theory here collapses the entire pipeline. This is principled symbolic reasoning.
2. **Type-directed semiring selection implements denotational semantics.** The EvalSpec DSL uses types to select semirings — counting for n-gram precision (BLEU, ROUGE-1/2), tropical for longest common subsequence (ROUGE-L), Boolean for exact match — implementing a genuine type-directed denotational semantics. The BNF grammar and typing rules are cleanly specified.
3. **Decidable specification equivalence via coalgebraic bisimulation.** The equivalence checker with distinguishing-word generation enables mechanical proof that two metric specifications denote the same function. This has applicability far beyond NLP — any domain where scoring functions can be represented as WFA benefits.
4. **KleeneSemiring typeclass fills a Mathlib gap.** The Lean 4 formalization introduces a KleeneSemiring typeclass with star-semiring axioms, contributing to the broader symbolic reasoning infrastructure independent of the ZK application.

## Weaknesses

1. **Metric scope is fundamentally limited to the regular-language fragment.** WFA can express only rational formal power series — string-matching metrics over finite vocabularies. Neural metrics (BERTScore, COMET), model-based evaluation (LLM-as-judge), and even character-level metrics with unbounded context are outside the representable class. The paper provides no analysis of what fraction of metrics in contemporary NLP benchmarks (MMLU, HumanEval, HellaSwag, ARC) are WFA-representable. For a system claiming to address "NLP evaluation integrity," the coverage is potentially narrow. Are accuracy, exact-match, and string-F1 the only WFA-representable metrics actually used in practice? This is an essential question the paper never answers.

2. **Post-processing gadgets escape the formal specification.** The WFA coverage percentages (60% for BLEU, 80% for Token F1, 100% only for Exact Match and ROUGE-L) reveal that substantial portions of metric computation — geometric mean aggregation for BLEU, harmonic mean for F1 — occur in circuit gadgets _outside_ the formal WFA specification. These gadgets are validated by differential testing (0 disagreements) but are never given formal semantics in the EvalSpec DSL. The symbolic framework thus has a specification gap precisely where composition happens: the WFA computes n-gram counts, but the final BLEU score emerges from an unspecified aggregation step.

3. **No symbolic diagnostics for proof failure.** When a STARK proof fails (malicious or buggy prover), the system outputs a binary REJECT verdict with no symbolic explanation. A symbolically rich system should generate a _counterexample trace_: which WFA transition was violated, at which input position, with what expected vs. actual weight. The absence of diagnostic symbolic reasoning means that debugging evaluation failures requires manual circuit inspection, defeating the purpose of a formal framework.

4. **EvalSpec DSL lacks higher-order composition.** The DSL compiles individual metrics to WFA but provides no combinators for building complex metrics from simpler ones: no arithmetic-mean combinator, no weighted-sum combinator, no conditional metric selection. "Compute BLEU if length > 5, else exact-match" cannot be expressed. This prevents the specification language from scaling to real evaluation protocols, which combine multiple metrics with rules.

5. **No ontological structure captures metric relationships.** Metrics are treated as isolated specifications with no knowledge graph or inheritance hierarchy. ROUGE-1 is a special case of ROUGE-N; exact match implies Token F1 = 1.0; BLEU-4 subsumes BLEU-1 through BLEU-3 via geometric mean. None of these relationships are formalized. An ontological model would enable specification reuse (e.g., deriving ROUGE-2 from a parameterized ROUGE-N template), consistency checking (e.g., verifying that exact-match scores imply F1 = 1), and systematic coverage analysis.

6. **Proposition 4.1 (ROUGE-L WFA correctness) relies on tropical matrix-vector multiplication without a compositionality proof.** The ROUGE-L construction decomposes LCS computation into tropical matrix-vector products, but the correctness argument (that the tropical WFA output equals the LCS length) is stated as a proposition without a mechanized proof. The Lean formalization has sorrys in this area. For a symbolic reasoning system, the most novel automata-theoretic construction (tropical WFA for LCS) has the weakest formal backing.

## Questions for Authors

- What fraction of metrics used in the top-20 NLP benchmarks (by leaderboard submissions) are WFA-representable? Can you provide a concrete census?
- Can the EvalSpec DSL be extended with higher-order combinators (e.g., `mean`, `weighted_sum`, `conditional`) while preserving decidable equivalence?
- Could the WFA framework be extended via weighted tree automata to capture compositional evaluation protocols, where evaluation trees (not just flat scores) are the output?

## Overall Assessment

Spectacles identifies the right algebraic abstraction — WFA over typed semirings — and builds a substantial verification pipeline atop it. The decidable equivalence, type-directed compilation, and KleeneSemiring formalization are genuine symbolic AI contributions that advance the state of the art. However, the symbolic framework is more limited than the paper's framing suggests: it covers only string-matching metrics (an unknown and potentially small fraction of modern NLP evaluation), it lacks compositional specification combinators, it provides no symbolic diagnostics, and its most novel construction (tropical WFA for ROUGE-L, Proposition 4.1) has the weakest formal backing. The gap between the WFA specification and the circuit gadgets that compute final scores (geometric mean, harmonic mean) is a specification gap the symbolic framework should address but does not.

**Score:** 6/10
**Confidence:** 4/5

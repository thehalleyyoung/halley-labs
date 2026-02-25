# Review: Spectacles — Verified Compiler from Semiring-Weighted Automata to Zero-Knowledge Scoring Circuits

**Reviewer:** Yuan Cheng
**Persona:** Probabilistic Modeling Researcher
**Expertise:** Probabilistic modeling, uncertainty quantification, Bayesian inference, calibration, distributional robustness

---

## Summary

Spectacles compiles NLP scoring metrics into weighted finite automata (WFA) over typed semirings and then generates STARK zero-knowledge proofs certifying score correctness. The project presents impressive triple-verification (reference × WFA × circuit) with 57,518 differential tests and 0 disagreements. However, the system's probabilistic and statistical foundations are underdeveloped: the contamination detection lacks a principled probabilistic model, test-corpus coverage is statistically uncharacterized, and the embedding of counting-semiring values into the Goldilocks field (Proposition E.1) relies on NLP-scale boundedness assumptions that are never validated against real-world distributional tails.

## Strengths

1. **Cryptographic soundness with quantified probability.** The 128-bit STARK security provides soundness error ≤ 2^{-128}, a rigorously quantified probabilistic guarantee far exceeding anything achievable by statistical testing alone. The 76 verified proofs across state counts up to 512 empirically confirm prover/verifier agreement.
2. **Deterministic triple-verification eliminates calibration ambiguity.** The three-way check (reference × WFA × circuit) produces binary agreement/disagreement verdicts without soft probabilistic judgments, yielding 57,518 clean deterministic checks. Two real bugs found (Montgomery reduction, Lagrange interpolation) validate the methodology.
3. **Fixed-point arithmetic with explicit error model.** Proposition E.1 establishes that counting-semiring values embed into the Goldilocks field F_p (p = 2^64 − 2^32 + 1) for bounded NLP-scale values, providing a concrete numerical error bound that avoids the floating-point uncertainty plaguing most NLP evaluation systems.
4. **Denotational semantics for specification.** The EvalSpec DSL with BNF grammar and typing rules provides a formal denotational semantics for metric specification, enabling decidable equivalence via Hopcroft minimization — a clean algebraic foundation.

## Weaknesses

1. **No uncertainty quantification on metric scores themselves.** Spectacles proves a score was computed correctly, but NLP metrics are inherently noisy — different tokenizations, reference sets, and human judgments produce score distributions, not point values. Proving that BLEU = 0.342 is correct says nothing about whether 0.342 ± 0.05 would be the answer under reasonable perturbations. The system certifies computation fidelity without any framework for reasoning about the _meaning_ or _stability_ of the certified score. This is a fundamental conceptual gap for a system targeting evaluation trustworthiness.

2. **Contamination detection has no probabilistic sensitivity analysis.** The verbatim n-gram contamination detection reports F1 = 1.00 at threshold τ = 0.03, but no receiver operating characteristic (ROC) analysis characterizes sensitivity across τ values. What is the false-positive rate as a function of natural n-gram overlap in English? Heavy paraphrasing admittedly evades detection, but no probabilistic model of partial contamination is provided — not even a mixture model quantifying what fraction of benchmark instances would be flagged under different contamination strategies. Without this, the detection's practical value is unquantifiable.

3. **Proposition E.1 boundedness assumption is unvalidated against distributional tails.** The counting-semiring embedding into F_p requires that intermediate values stay within [0, 2^62). This bound is claimed to hold for "NLP-scale values" but no empirical distribution of intermediate accumulator values across real NLP corpora (beyond the 30-word, 9-token test corpus) is provided. Real-world BLEU and ROUGE computations on long documents could produce n-gram counts exceeding these bounds, and the system provides no overflow detection or graceful degradation — it would silently produce incorrect proofs.

4. **Test-corpus coverage is statistically uncharacterized.** The 30-word vocabulary with max 9 tokens covers a vanishingly small fraction of real NLP input distributions. No statistical argument — no coverage metric, no input-space sampling strategy, no partition testing analysis — bounds the representativeness of these 57,518 tests. The jump from toy-scale testing to production-scale correctness claims has no probabilistic bridge. Property-based testing (9.8K tests) helps but still operates on synthetic distributions.

5. **Proof-time distributional analysis is absent.** The 400-state BLEU-4 proof time of 3,821 ± 271 ms reports mean ± standard deviation but no confidence intervals, no percentile analysis (P95, P99), and no tail-behavior characterization. For a system targeting production deployment, worst-case latency dominates mean latency. If the distribution has heavy tails (common in cryptographic computations due to hash collisions or FRI query variance), the ± 271 ms could be severely misleading.

6. **No distributional robustness of the overall pipeline.** The system makes no claims about how metric scores, proof times, or verification outcomes change under distributional shift in inputs. A distributionally robust evaluation would test correctness under adversarial input distributions (e.g., all-identical tokens, Unicode edge cases, maximum-length sequences), not just random sampling with fixed seeds.

## Questions for Authors

- Can you provide empirical distributions of intermediate counting-semiring accumulator values on production NLP corpora (e.g., WMT translation, full SQuAD) to validate the [0, 2^62) bound in Proposition E.1?
- What is the P99 proof-generation latency, and does the tail behavior of proof times change with WFA state count?
- Have you considered a Bayesian or distributional model of contamination that accounts for paraphrasing and soft overlap, rather than the all-or-nothing verbatim n-gram approach?

## Overall Assessment

Spectacles makes a genuinely novel contribution by applying verified compilation and zero-knowledge proofs to NLP evaluation, and its algebraic foundations (WFA over semirings, decidable equivalence, two-tier compilation) are technically sound. However, from a probabilistic modeling perspective, the system certifies computation without characterizing the uncertainty inherent in what is being computed. The contamination detection lacks a principled probabilistic model, the field-embedding boundedness assumption is unvalidated against real distributions, the test corpus provides no statistical coverage guarantee, and performance reporting omits distributional characterization. These gaps do not invalidate the core algebraic contribution, but they significantly weaken the practical evaluation claims.

**Score:** 6/10
**Confidence:** 3/5

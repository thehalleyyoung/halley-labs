# Review: Spectacles — Verified WFA-ZK Scoring Circuits for Contamination-Certified Evaluation

**Reviewer:** Yuan Cheng (Probabilistic Modeling Researcher)  
**Expertise:** Probabilistic reasoning, statistical testing, information-theoretic privacy, randomized algorithms in NLP evaluation, cryptographic soundness analysis  
**Score:** 7/10  
**Recommendation:** Weak Accept

---

## Summary

Spectacles proposes a compelling architecture coupling weighted finite automata over semirings with STARK-based zero-knowledge proofs to produce contamination-certified evaluation certificates. The statistical and probabilistic dimensions—PSI false positive/negative rates, pass@k sampling in ZK, BLEU smoothing specification equivalence, and STARK soundness—are partially addressed but leave meaningful gaps that temper my enthusiasm.

## Strengths

**1. Principled Probabilistic Framing of Contamination Detection.** The PSI-based contamination guarantee (overlap < τ) is cast as a statistical hypothesis test with a well-defined null, which is far more rigorous than ad hoc n-gram deduplication. The trie-structure optimization achieving 50–100× communication reduction is a genuine contribution, and the OPRF construction preserves the statistical guarantees while hiding individual membership decisions. The connection to differential privacy literature is implicit but promising.

**2. Correct Semiring Decomposition for Counting Metrics.** The observation that BLEU n-gram counts, token-level precision/recall, and exact match all decompose over counting or Boolean semirings with injective homomorphisms into F_p is mathematically sound. The Tier 1 algebraic embedding preserves the monoidal structure needed for correct aggregation, meaning the STARK proof inherits specification-level correctness rather than merely attesting to program execution.

**3. STARK Soundness Aligned with Evaluation Scale.** The Goldilocks field choice (p = 2^64 − 2^32 + 1) gives soundness error 2^{-64} per query, which over realistic evaluation corpus sizes (10^3–10^5 instances) keeps the union-bound failure probability negligible. The authors correctly note that FRI proximity gaps dominate the concrete security parameter, showing awareness of the gap between asymptotic and concrete soundness.

**4. Explicit Treatment of pass@k Sampling Complexity.** The paper acknowledges that pass@k requires sampling k completions per problem and that the unbiased estimator (Chen et al., 2021) involves combinatorial terms that must be arithmetized. This is a non-trivial observation—naïve circuit encoding of binomial coefficients in F_p would blow up constraint counts—and the proposed lookup-table approach is pragmatic.

## Weaknesses

**1. PSI False Positive Rate Under Approximate Matching is Uncharacterized.** The OPRF-based PSI assumes exact string equality, but real contamination involves paraphrasing, truncation, and reformatting. The paper gestures at fuzzy matching via Jaccard-based thresholds but provides no analysis of how approximate matching inflates the false positive rate or distorts the contamination bound τ. Without this, the statistical guarantee degrades to an exact-duplicate check, which the community already knows is insufficient.

**2. BLEU Smoothing Variants Break Specification Equivalence.** The paper claims WFA equivalence for BLEU, but standard BLEU has at least four smoothing variants (add-ε, add-1, NIST geometric, exponential decay) that produce different scores on identical inputs. The WFA formalization appears to fix one variant without justifying which, and the decidable equivalence result applies only within that choice. Cross-variant equivalence is undecidable in general because smoothing introduces irrational weights outside any finite semiring embedding.

**3. Probabilistic Soundness Composition Across Protocol Phases is Missing.** The three guarantees (G1: STARK proof, G2: PSI contamination, G3: commit-then-reveal) each have independent soundness parameters, but the paper does not compose them. A certificate consumer needs a single confidence level, which requires analyzing whether the protocol phases are sequentially composable or whether adaptive adversary strategies can amplify failure probabilities across phases.

**4. pass@k Arithmetization Cost is Underestimated.** The lookup-table approach for binomial coefficients in the STARK circuit requires precomputing C(n, k) for all relevant n, and the table size grows as O(n_max · k_max). For pass@100 with 1000 problems, this is 10^5 entries in the Goldilocks field, each requiring a Plookup gate. The resulting constraint overhead may dominate the rest of the scoring circuit, but no concrete benchmarks are provided.

**5. No Confidence Intervals on Differential Testing Results.** The 100K differential testing pairs are presented as a point estimate of correctness, but without confidence intervals or coverage analysis (e.g., partition testing over input equivalence classes), the statistical power of this validation is unclear. A Clopper-Pearson interval on 0/100K failures gives an upper bound of ~3×10^{-5} defect rate at 95% confidence, which the authors should state explicitly.

## Verdict

Spectacles makes a strong conceptual contribution by unifying WFA-based metric decomposition with zero-knowledge contamination certification, and the statistical foundations are mostly sound. However, the gaps in PSI approximate matching analysis, BLEU smoothing specification, and cross-phase soundness composition prevent me from giving a strong accept—addressing these would elevate the work significantly.

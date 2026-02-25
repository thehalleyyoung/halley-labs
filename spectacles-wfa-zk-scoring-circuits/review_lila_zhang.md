# Review: Spectacles — Contamination-Certified Evaluation Certificates

**Reviewer:** Lila Zhang
**Persona:** Symbolic Reasoning and AI Expert
**Expertise:** Formal language theory, automata theory, semiring algebra, symbolic compilation, type-theoretic foundations

---

## Summary

Spectacles recognizes that NLP scoring functions decompose into weighted finite automata over semirings, then compiles them to STARK arithmetic circuits. This is a deep and elegant insight — the connection between evaluation metrics and formal language theory is genuinely surprising and well-executed. The EvalSpec DSL with its BNF grammar, typing rules, and denotational semantics provides a rigorous foundation. The two-tier compilation strategy correctly handles the algebraic differences between embeddable and tropical semirings.

## Strengths

1. **The WFA decomposition insight is the core intellectual contribution.** Recognizing that BLEU, ROUGE, F1, and other metrics can be expressed as WFA over appropriate semirings is a deep observation that connects NLP evaluation to formal language theory in a previously unexplored way. This enables the entire verification pipeline.

2. **EvalSpec DSL is a well-designed specification language.** The BNF grammar, typing rules (inference rules), and denotational semantics (mapping to formal power series) provide a rigorous foundation. The expressibility boundary analysis honestly identifies which metrics are and are not WFA-expressible.

3. **Two-tier compilation is the right architecture.** Tier 1 (direct homomorphism for F_p-embeddable semirings like Boolean and counting) and Tier 2 (comparison gadgets for tropical/min-plus semirings) correctly separate the algebraically trivial from the algebraically challenging cases.

4. **KleeneSemiring axiomatization is handled carefully.** The paper correctly addresses the subtlety that KleeneSemiring's Kleene star diverges for counting semiring values ≥ 1, and that the typeclass is used for axiom formalization only, not for Kleene star computation on counting semirings.

5. **The Boolean embedding determinism argument is correct.** The proposition that Boolean ∨-to-+ mismatch never arises in deterministic WFAs (at most one active transition per state) is a clean result that justifies the Tier 1 compilation for Boolean metrics.

## Weaknesses

1. **The WFA-to-circuit compilation is not formally verified.** The most critical step — translating WFA semantics into arithmetic circuit constraints — is verified only by differential testing. A compilation soundness theorem (WFA execution ≡ circuit execution for all inputs) should be the centerpiece of the formalization, but it is deferred.

2. **The "sorry" management is concerning.** 12 routine instances and 5 novel instances of sorry in the Lean formalization mean significant proof obligations are unresolved. The routine ones may be addressable by tactics, but the 5 novel ones (Hopcroft minimization, WFA-to-AIR compilation) are precisely the most important proofs.

3. **The EvalSpec DSL expressibility analysis is incomplete.** The boundary table marks geometric mean, corpus BLEU, and BERTScore as non-WFA-expressible, but does not characterize exactly which mathematical operations push a metric beyond WFA expressibility. A formal expressibility theorem (e.g., "a metric is WFA-expressible iff it uses only operations X, Y, Z") would be a strong contribution.

4. **Post-processing aggregation gadgets are tested but not verified.** Metrics like BLEU require n-gram counting, brevity penalties, and geometric mean — operations that go beyond pure WFA computation. These are handled by aggregation gadgets that are tested but not formally verified, creating a gap in the verification chain.

5. **The semiring abstraction may not capture all metric nuances.** Some NLP metrics involve thresholding, truncation, or normalization operations that are naturally expressed in arithmetic but awkward in semiring terms. The paper does not discuss how these operations are handled or whether they introduce approximation errors.

## Novelty Assessment

The WFA decomposition of NLP metrics is highly original. The two-tier compilation strategy is a genuine architectural contribution. The EvalSpec DSL with formal semantics adds to the novelty. This is the most intellectually novel project in the collection from a formal language theory perspective. **High novelty.**

## Suggestions

1. Prioritize the WFA-to-circuit compilation soundness theorem in Lean — this is the critical missing proof.
2. Formalize the WFA expressibility boundary as a theorem characterizing exactly which metric operations are WFA-expressible.
3. Address the 5 novel sorry instances with dedicated proof strategies.
4. Provide formal specifications for the aggregation gadgets used in post-processing.

## Overall Assessment

Spectacles has the deepest intellectual insight of the projects reviewed: NLP metrics decompose into WFA over semirings. This observation enables an elegant verification pipeline from specification to proof. The EvalSpec DSL and two-tier compilation are well-designed. However, the formal verification chain has significant gaps — the most important compilation theorem is unproved, and 17 sorry instances remain. With completion of the core proofs, this could be an outstanding contribution to both formal methods and NLP evaluation.

**Score:** 8/10
**Confidence:** 4/5

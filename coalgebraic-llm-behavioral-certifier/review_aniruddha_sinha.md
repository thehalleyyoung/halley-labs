# Review: CABER — Coalgebraic Behavioral Auditing of Foundation Models

**Reviewer:** Aniruddha Sinha
**Persona:** Model Checking and AI Applicant
**Expertise:** Temporal logic, model checking, CEGAR, state-space exploration, abstraction refinement, bisimulation

---

## Summary

CABER applies coalgebraic model checking to LLM behavioral verification, combining active learning (PCL*), counterexample-guided abstraction refinement (CoalCEGAR), and a temporal logic (QCTL_F) for specification and verification. The theoretical framework is substantially more sophisticated than TOPOS's CEGAR, with genuine abstraction lattice structure and Galois connections. However, the system has been validated only on mock Markov chains, the Lean 4 formalization is absent, and the 38 property-based tests — while valuable — cannot substitute for machine-checked proofs of the core theorems.

## Strengths

1. **CoalCEGAR has genuine CEGAR structure.** The abstraction lattice over (k, n, ε) triples with refinement operations is a proper CEGAR instantiation. The monotonicity proof for safety properties (Proposition 1) and finite-lattice termination argument (Remark 4.8) are correct and non-trivial. This is significantly more rigorous than TOPOS's mislabeled "CEGAR."

2. **QCTL_F is a well-chosen logic.** The graded satisfaction degrees in [0,1] are appropriate for probabilistic behavioral verification, and the alternation-free fragment being polynomial-time decidable is practically relevant. The specification templates (RefusalPersistence, SycophancyResistance, etc.) are well-motivated by real LLM safety concerns.

3. **Kantorovich bisimulation distance is the right metric.** Using Kantorovich lifting for quantitative behavioral distance computation is theoretically sound and provides a metric that is compatible with the probabilistic semantics. The convergence properties are well-established in the coalgebra literature.

4. **End-to-end error composition (Theorem 3) is carefully stated.** The sequential conditioning argument in the proof correctly handles dependencies between pipeline stages. The five-term additive bound is tight enough to be useful while being loose enough to be provable.

5. **38 property-based tests cover meaningful invariants.** Tests for Hoeffding bounds, bandwidth monotonicity, Galois distortion, and PAC decomposition provide computational evidence for the mathematical claims.

## Weaknesses

1. **No Lean 4 formalization.** The core theorems (PCL* convergence, bandwidth-sample bound, error composition) are paper proofs. While 38 property-based tests provide evidence, they cannot verify the proofs themselves — they verify specific instances, not the general theorems. For a paper that claims formal guarantees, this is a significant gap.

2. **Mock LLM validation is insufficient for the claims.** The stochastic mock LLMs are Markov chains with 3-6 states — precisely the systems that L*-based algorithms are designed to learn. The 92-100% accuracy on these models says nothing about the approach's viability on real LLMs, which are not finite-state Markov systems.

3. **Alphabet abstraction is acknowledged as non-functorial.** The embedding-based clustering that maps LLM outputs to a finite alphabet does not preserve the coalgebraic functor structure. This means the entire theoretical framework rests on an approximation at the input layer whose error is uncharacterized.

4. **State explosion risk is unaddressed.** PCL* learns up to 40 states on 3-6 state mock models. Real LLMs with complex behavioral landscapes could require thousands of states, making model checking intractable. No analysis of state scaling or mitigation strategies is provided.

5. **QCTL_F specifications are limited to the alternation-free fragment.** While this restriction enables polynomial-time model checking, it excludes properties involving nested path quantifiers (e.g., "for every jailbreak attempt, there exists a defense that persists"). The expressive power of the supported fragment relative to actual safety properties is not discussed.

## Novelty Assessment

The coalgebraic framing of LLM behavioral verification is genuinely novel. The PCL* algorithm, functor bandwidth concept, and the integration of CoalCEGAR with QCTL_F represent original contributions. However, the positioning relative to prior work (AALpy + PRISM) is not adequately justified — the claimed structural advantages (functor-parameterized abstraction, quantitative bisimulation) are theoretical since the implementation uses mock models where PDFA + PRISM would also succeed. **High novelty in framing, uncertain novelty in practice.**

## Suggestions

1. Prioritize Lean 4 formalization of at least Theorem 1 (PCL* convergence) to validate the core guarantee.
2. Validate on at least one real LLM API to demonstrate practical viability.
3. Characterize the error introduced by the non-functorial alphabet abstraction.
4. Analyze state scaling and provide strategies for managing state explosion on complex models.
5. Discuss the expressive limitations of the alternation-free fragment relative to practical safety properties.

## Overall Assessment

CABER is the most theoretically sophisticated project in this collection, with genuine contributions in coalgebraic LLM modeling, PCL*, and functor bandwidth. The CoalCEGAR instantiation is a proper CEGAR, unlike TOPOS's mislabeled version. However, the gap between theory and practice is wide: no Lean formalization, no real LLM validation, and an acknowledged non-functorial approximation at the input layer. The work is a strong theoretical contribution that needs substantial empirical validation to fulfill its promises.

**Score:** 7/10
**Confidence:** 4/5

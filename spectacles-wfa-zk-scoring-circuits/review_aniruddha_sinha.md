# Review: Spectacles — Contamination-Certified Evaluation Certificates

**Reviewer:** Aniruddha Sinha
**Persona:** Model Checking and AI Applicant
**Expertise:** Protocol verification, state machine verification, formal methods for security protocols, model checking methodology

---

## Summary

Spectacles implements a verification pipeline from NLP metric specification to STARK proof generation, with a commit-then-execute protocol for contamination detection. The verification architecture is well-structured with clear separation of concerns (specification, automata, circuit, protocol, privacy, scoring). However, the protocol layer lacks formal verification — no TLA+ specification, no model-checked state machine, no security proof beyond informal arguments. The Lean 4 formalization covers only semiring axioms, not the compilation pipeline or protocol correctness.

## Strengths

1. **The EvalSpec DSL with formal BNF, typing rules, and denotational semantics is well-designed.** The formal specification provides clear semantics for what each metric computes. The expressibility boundary table honestly marking geometric mean, corpus BLEU, and BERTScore as NOT WFA-expressible is commendable.

2. **Two-tier compilation is architecturally sound.** Separating F_p-embeddable semirings (direct homomorphism, Tier 1) from tropical semirings (comparison gadgets, Tier 2) is the right abstraction. The tier selection is principled and automated.

3. **Property-based testing is thorough.** 37 proptest tests covering 8 core semiring axioms × 3+ types, with 256 random inputs each, provides good coverage. Cross-language validation with 22 Python tests adds an independent verification layer.

4. **Comparison gadget soundness and completeness are proved.** The soundness proof (gadget correctly computes min(a,b)) and completeness proof (unique satisfying witness) in Appendix D are non-trivial contributions with proper precondition analysis (T·w_max < 2^63).

5. **118K lines of Rust demonstrates serious implementation effort.** The six-layer architecture with per-layer integration tests shows a well-structured codebase.

## Weaknesses

1. **No TLA+ or Promela protocol specification.** The commit-then-execute protocol is described informally but not model-checked. Concurrency bugs, race conditions, and liveness violations cannot be ruled out without formal protocol verification. This is acknowledged in limitations but is a fundamental gap for a security-sensitive protocol.

2. **Lean formalization covers only semiring axioms, not the compilation pipeline.** The Lean proofs cover KleeneSemiring axioms and circuit soundness (partially), but not the WFA-to-AIR compilation, STARK proving, or PSI protocol. The paper's three-layer empirical methodology (Lean + proptest + differential) is a reasonable mitigation but does not provide machine-checked certainty for the most critical components.

3. **The PSI protocol assumes semi-honest adversaries.** No UC security, no VOPRF, and collusion is out of scope. In a benchmark certification setting, the model provider has strong incentives to cheat, and semi-honest security may be insufficient. The commit-then-execute protocol provides some defense but relies on binding commitments that are not formally verified.

4. **No end-to-end formal verification chain.** The pipeline from EvalSpec → WFA → AIR → STARK → Certificate has formal proofs at individual steps (semiring axioms, comparison gadget) but no end-to-end composition theorem. Errors at layer boundaries could invalidate downstream guarantees.

5. **Hopcroft minimization is not formally verified.** The WFA equivalence check via Hopcroft minimization is critical for the metric equivalence theorem (Theorem 3), but the Lean proof is deferred. Brute-force testing on words up to length 6 does not cover all cases.

## Novelty Assessment

The WFA-based approach to NLP metric verification is novel. The two-tier compilation strategy is a genuine architectural contribution. However, the STARK proving and PSI components use established techniques without methodological innovation. **Moderate to high novelty overall, concentrated in the WFA decomposition and compilation.**

## Suggestions

1. Create a TLA+ specification of the commit-then-execute protocol and model-check it for safety and liveness.
2. Complete the Lean formalization for Hopcroft minimization, as it is critical for Theorem 3.
3. Formalize the end-to-end compilation theorem as a single Lean theorem composing individual layer guarantees.
4. Discuss the security model limitations more prominently and consider malicious-adversary extensions.

## Overall Assessment

Spectacles has the most ambitious verification goal of the projects reviewed — machine-checkable proofs of benchmark score correctness. The WFA decomposition and two-tier compilation are genuine contributions. However, the verification chain has significant gaps: no protocol verification, partial Lean formalization, and unverified Hopcroft minimization. The project would benefit enormously from TLA+ protocol verification and complete Lean formalization of the compilation pipeline.

**Score:** 7/10
**Confidence:** 4/5

# Review: Spectacles — Contamination-Certified Evaluation Certificates

**Reviewer:** Joseph S. Chang
**Persona:** Automated Reasoning and Logic Expert
**Expertise:** Formal verification, proof theory, decidability, compilation correctness, theorem proving methodology

---

## Summary

Spectacles compiles NLP scoring functions into STARK arithmetic circuits via WFA over semirings, aiming to produce machine-checkable proofs of score correctness. The WFA decomposition insight is theoretically interesting and the Lean 4 formalization of semiring axioms demonstrates commitment to rigor. However, the formalization covers only the foundational algebraic layer — the critical compilation correctness theorem (WFA semantics preserved by circuit translation) is unproved. The "sorry" count (12 routine + 5 novel) in the Lean development is concerning, particularly since the 5 novel instances include the most important proof obligations.

## Strengths

1. **WFA-to-circuit compilation as a verified compilation problem is well-framed.** Treating the metric computation as a formal language problem and the circuit generation as compilation creates a clean verification target. The compilation should preserve denotational semantics, and the target (arithmetic circuits) has a clear operational semantics.

2. **Two-tier compilation strategy handles algebraic complexity correctly.** The distinction between F_p-embeddable (homomorphism-based) and non-embeddable (gadget-assisted) semirings is the right decomposition. The Tier 2 soundness proof with the T·w_max < 2^63 precondition is careful and correct.

3. **Comparison gadget soundness and completeness proofs are non-trivial.** The soundness proof (gadget computes min(a,b) for values in [0, 2^63)) and completeness proof (unique satisfying witness) are proper formal results with careful precondition analysis. The edge case at 2^63-1 is explicitly handled.

4. **Sorry categorization shows self-awareness.** Distinguishing routine sorries (~12, targetable by omega/decide) from novel ones (~5, requiring new proof techniques) demonstrates understanding of the formalization challenges and enables targeted effort.

5. **Cross-language validation adds independent checking.** Python implementations of semiring axioms providing an independent verification layer beyond Lean and Rust is a sound methodology.

## Weaknesses

1. **The critical compilation correctness theorem is unproved.** The theorem "for all WFA W and inputs x, execute_wfa(W, x) = evaluate_circuit(compile(W), x)" is the linchpin of the verification chain. Without this theorem, the differential testing (800K pairs, 0 disagreements) provides strong empirical evidence but not a formal guarantee. This theorem is listed among the 5 novel sorry instances.

2. **Lean formalization scope is narrower than the presentation suggests.** Semiring axioms are foundational but algebraically simple. The hard verification targets (Hopcroft minimization correctness, WFA-to-AIR compilation, STARK proof generation) are not formalized. The paper's "verified specification" framing is honest but readers may overestimate the formalization depth.

3. **STARK proof generation correctness is taken on faith.** The STARK proving component computes proofs of arithmetic circuit satisfiability, but the STARK implementation itself is not verified. A bug in the STARK prover could produce invalid proofs that pass verification. The paper does not discuss STARK implementation trust.

4. **Proof size analysis is asymptotic, not concrete.** The O(λ·log²(T|Q|)) proof size bound is useful for scalability analysis but the constants matter for practical deployment. The 45-750 KiB estimates use a formula with specific constants, but these are not validated against a real STARK prover.

5. **No proof-theoretic analysis of the verification certificate.** What exactly does the STARK proof prove? It proves that there exists a witness (execution trace) consistent with the circuit constraints. But are the circuit constraints sound? This requires the unproved compilation correctness theorem. The proof-theoretic chain is: circuit constraints ← compilation ← WFA semantics ← metric specification. Only the last link (WFA ← metric) has some Lean coverage.

6. **No formal treatment of the Goldilocks field.** The p = 2^64 - 2^32 + 1 field is used for arithmetic circuits, and two bugs were found and fixed. The field axioms should be Lean-verified (they are straightforward), and the fixed implementations should be proved correct against the axioms.

## Novelty Assessment

The WFA decomposition is a genuinely novel insight. The compilation pipeline from specification to circuit is an original contribution. The Lean formalization, while incomplete, represents a serious attempt at formal verification. **High novelty for the WFA approach, moderate novelty for the formalization methodology.**

## Suggestions

1. Prove the compilation correctness theorem — this should be the #1 priority for the Lean development.
2. Lean-verify the Goldilocks field axioms and the fixed Montgomery/Lagrange implementations.
3. Analyze the STARK implementation trust: use a verified STARK library or provide independent verification.
4. Provide concrete proof size measurements from a real STARK prover, not just asymptotic estimates.
5. State the proof-theoretic chain explicitly and mark which links are formally verified.

## Overall Assessment

Spectacles has the most intellectually interesting insight (NLP metrics as WFA over semirings) and the most ambitious verification goal (machine-checkable proofs of score correctness). The Lean formalization demonstrates commitment to rigor, and the sorry categorization shows self-awareness. However, the critical compilation correctness theorem is unproved, reducing the formal guarantee to semiring axioms. The STARK prover and protocol layer are unverified. With the compilation theorem and Goldilocks field formalization, this would be a significantly stronger contribution.

**Score:** 7/10
**Confidence:** 5/5

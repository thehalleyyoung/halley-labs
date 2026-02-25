# Review: Spectacles — Verified Compiler from Semiring-Weighted Automata to Zero-Knowledge Scoring Circuits

**Reviewer:** Aniruddha Sinha
**Persona:** Model Checking & AI Applicant
**Expertise:** Model checking, CEGAR, temporal logic verification, abstract interpretation, state-space exploration

---

## Summary

Spectacles presents a verified compilation pipeline from NLP metric specifications (as weighted finite automata over typed semirings) to STARK arithmetic circuits, accompanied by Lean 4 formalization of soundness theorems. From a model-checking perspective, this work engages genuine verification concerns — specification languages, compilation correctness, decidable equivalence, and mechanized proofs — but falls short of the end-to-end guarantees the verification community expects. The critical finding is that the system verifies _circuit simulation of a WFA_, not _metric correctness end-to-end_: the metric-to-WFA correspondence relies on 57K differential tests rather than formal proof, creating a verification gap reminiscent of pre-extraction CompCert.

## Strengths

1. **Specification-level verification.** Unlike systems that verify program execution traces, Spectacles verifies conformance to a mathematical specification (WFA formal-power-series semantics). This addresses the specification problem in verifiable computation and elevates the contribution above typical ML-verification work.
2. **Decidable equivalence checking via Hopcroft minimization.** The WFA equivalence decision procedure (Hopcroft minimization + coalgebraic bisimulation) with distinguishing-word generation enables mechanically proving or disproving that two metric implementations compute the same function. This is a lasting formal-methods contribution.
3. **Two-tier compilation architecture.** The principled separation of Tier 1 (algebraic compilation via semiring homomorphism ι: S → F_p, Theorem 6.1) from Tier 2 (gadget-assisted tropical compilation, Theorem 6.2) demonstrates honest proof engineering. The paper does not falsely claim uniform treatment.
4. **Lean 4 formalization with sorry-free semiring axioms.** Sorry-free proofs of semiring axioms and the KleeneSemiring typeclass (filling a recognized Mathlib gap) represent real mechanized verification effort.

## Weaknesses

1. **End-to-end integration gap is the dominant flaw.** The STARK proofs verify that a circuit correctly simulates a given WFA, but the metric-to-WFA construction itself is NOT formally verified. Theorem 6.1 and 6.2 establish circuit-to-WFA soundness, but there is no Theorem establishing WFA-to-metric-specification soundness. The metric-to-WFA correspondence relies entirely on 57,518 differential tests. In model-checking terms, the _abstraction function_ (metric → WFA) is untrusted. This means the system proves "this circuit computes _something_ correctly" but the formal guarantee that the _something_ equals the intended metric is empirical, not proved.

2. **Lean-to-Rust gap undermines mechanization claims.** The Lean 4 proofs cover a mathematical model; the deployed Rust implementation is a separate codebase bridged by differential testing and 9.8K property tests, not machine-checked extraction. The verification community has established that testing cannot substitute for extraction (CompCert, CakeML, sel4 all use code extraction or verified compilation). The 57K differential tests mitigate this but fundamentally cannot guarantee absence of implementation bugs. Calling the system "verified" is technically accurate only for the Lean model, not the running code.

3. **15 Lean sorrys constitute real verification debt.** Three novel sorrys with proof sketches are particularly concerning: proof sketches are not proofs, and formal verification's entire value proposition is that _seeming correct_ differs from _being provably correct_. The 12 routine sorrys (closable by omega/simp/ring) are less concerning but their continued presence suggests the formalization is more of a proof-of-concept than a complete verification. In model checking, an incomplete model is an unreliable model.

4. **No temporal properties of the evaluation protocol.** The commit-reveal-verify protocol imposes a temporal ordering on evaluation steps, but no temporal-logic specification (LTL/CTL) of this protocol is provided. The TLA+ model checking of 6 safety + 2 liveness properties is mentioned but applies only to an abstract protocol model, not to the concrete implementation. The composition of commitment, evaluation, and certification into an end-to-end guarantee is argued informally, not verified.

5. **State-space scalability is inadequately characterized.** BLEU-4 with 400 states takes 3,821 ms per proof. The circuit width scales quadratically with state count (O(|Q|² × |Σ|) constraints per step). For production-scale BLEU with vocabularies of 30K+ tokens and sequences of 100+ tokens, state counts could reach tens of thousands. No analysis of state-count growth rates is provided, no CEGAR-like state-space reduction is attempted, and the 512-state ceiling on verified proofs raises questions about whether the approach scales to realistic inputs. The absence of any abstraction-refinement strategy is a missed opportunity.

6. **WFA coverage gaps are structurally unverified.** The WFA coverage percentages (60%–100% depending on metric) mean that significant portions of metric computation (geometric mean for BLEU, harmonic mean for F1) occur in post-processing circuit gadgets outside the formal WFA model. These gadgets are tested but not specified in the EvalSpec DSL, creating an unmodeled computation gap in the formal architecture.

## Questions for Authors

- Can you provide a formal statement (even without proof) of the metric-to-WFA correspondence for at least one metric (e.g., Exact Match), establishing that the WFA construction correctly captures the metric's denotational semantics?
- What is the projected state count for BLEU-4 on a realistic vocabulary (e.g., 5K BPE tokens) and sequence length (e.g., 50 tokens)?
- Have you considered CEGAR-style abstraction-refinement to manage state-space growth, where an abstract WFA is refined only when a spurious counterexample is found?

## Overall Assessment

Spectacles makes a genuine formal-methods contribution with its decidable WFA equivalence, two-tier compilation architecture, and Lean 4 formalization. However, the end-to-end verification story has significant gaps. The most critical is the unverified metric-to-WFA construction: the system formally proves circuit-to-WFA soundness (Theorems 6.1/6.2) but relies on testing for WFA-to-metric soundness. Combined with the Lean-to-Rust gap, the 15 remaining sorrys, and the absence of temporal-logic protocol verification, the system is better described as "partially verified with strong testing" than "verified." This is still a valuable contribution — few systems in this space achieve even this level of formalization — but the verification community should evaluate it honestly against its claims.

**Score:** 6/10
**Confidence:** 4/5

# Review: Spectacles — Verified WFA-ZK Scoring Circuits for Contamination-Certified Evaluation

**Reviewer:** Joseph S. Chang (Automated Reasoning & Logic Expert)  
**Expertise:** Interactive theorem proving, proof automation, decidability theory, arithmetic circuit complexity, proof-carrying code and certified compilation  
**Score:** 7/10  
**Recommendation:** Weak Accept

---

## Summary

Spectacles ambitiously combines mechanized proof (Lean 4), zero-knowledge proof systems (STARKs), and private set intersection into a unified evaluation certification framework. The formal verification strategy—particularly the must-prove/should-prove stratification and the decidability characterization of WFA-expressible metrics—demonstrates genuine theorem-proving sophistication, but critical proof obligations remain open, and the proof-carrying certificate design needs refinement.

## Strengths

**1. Must-Prove/Should-Prove Stratification is Methodologically Sound.** The explicit classification of proof obligations into must-prove (semiring homomorphism injectivity, WFA bisimulation soundness, STARK circuit-specification equivalence) and should-prove (Hopcroft minimization, NTT correctness, hash collision resistance) reflects mature verification engineering. This stratification acknowledges that not all components carry equal trust risk and allocates formal verification effort accordingly. The must-prove obligations are precisely the ones where a bug would silently invalidate certificates, while should-prove obligations affect performance or have well-studied cryptographic arguments.

**2. Decidability Characterization of WFA-Expressible Metrics is Novel.** The paper provides the first explicit characterization of which standard NLP metrics are decidably equivalent when expressed as WFA over specific semirings. The key insight—that decidability follows from Schützenberger's theorem when the underlying semiring is commutative and Noetherian—is not new, but its application to BLEU, ROUGE, exact match, token F1, regex, and pass@k as a systematic classification is a genuine contribution to the intersection of formal language theory and NLP evaluation.

**3. Lean 4 Formalization Strategy Targets the Right Kernel.** The 800 LoC pilot formalizes the core trust kernel: semiring axioms, WFA definitions, bisimulation relation, and the key lemma that bisimilar WFA produce equal weighted languages. This is the correct minimization of the trusted computing base—if this kernel is correct, the remaining system properties (circuit compilation, STARK soundness) can be argued from standard cryptographic assumptions without additional formalization. The choice of Lean 4 over Coq or Isabelle is justified by Lean's superior metaprogramming (needed for the `wfa_equiv` tactic) and growing Mathlib ecosystem.

**4. STARK Arithmetic Circuit Soundness Argument is Precise.** The soundness argument for the STARK circuits correctly decomposes into (a) the algebraic constraint system faithfully encodes the WFA computation, (b) FRI proximity testing ensures low-degree polynomial commitment, and (c) Fiat-Shamir heuristic provides non-interactivity in the random oracle model. The Goldilocks field choice enables efficient NTT-based polynomial arithmetic, and the paper correctly notes that soundness error is dominated by the FRI query count rather than the field characteristic.

## Weaknesses

**1. Hopcroft Minimization is Critical, Not Optional.** The classification of Hopcroft minimization as "should-prove" is the paper's most significant verification gap. The `wfa_equiv` tactic works by minimizing both WFA and checking isomorphism of the minimal forms. If minimization produces non-canonical forms, the tactic is unsound—it could declare inequivalent WFA equivalent (false positive) or equivalent WFA inequivalent (false negative). Both failure modes are dangerous: false positives mean the certificate attests to a wrong score, while false negatives mean valid metrics are rejected. This should be must-prove, and the 800 LoC Lean formalization is incomplete without it.

**2. Proof-Carrying Certificate Design Lacks Formal Specification.** The paper describes certificates as "cryptographic proofs that simultaneously attest to score correctness and training-test data separation," but never formally specifies what a certificate contains, what properties a valid certificate satisfies, or what a verifier checks. A proof-carrying code approach (Necula 1997) requires an explicit verification condition generator and a proof checker; the paper has the prover (STARK) but not the verifier specification. Without this, certificate interoperability across different verifier implementations is undefined.

**3. Decidability Characterization Excludes Important Metrics.** The WFA-expressibility characterization covers seven metrics, but notably excludes BERTScore (which involves neural embeddings), MAUVE (which involves distribution divergence), and other learned metrics increasingly used in LLM evaluation. The paper does not provide a negative result—a proof that these metrics are not WFA-expressible—which would be equally valuable. The characterization is thus an open-ended positive result without clear boundaries.

**4. STARK Circuit-Specification Equivalence is the Hardest Must-Prove.** The must-prove obligation that the STARK arithmetic circuit correctly encodes the WFA computation is stated but not addressed in the Lean formalization. This is the most complex proof obligation: it requires formalizing the circuit compiler's semantics, the WFA-to-constraint-system translation, and proving their equivalence. The paper acknowledges this but offers no strategy beyond "future work," which is unsatisfying given that this is classified as must-prove.

**5. No Formal Connection Between Lean Proofs and STARK Verification.** The Lean proofs establish properties of abstract WFA, and the STARK proofs establish properties of concrete arithmetic circuits, but there is no formal bridge connecting them. The certificate consumer must trust that the Lean-verified WFA specification corresponds to the STARK-verified circuit computation, but this correspondence is exactly the Lean-Rust semantic gap by another name. A proof-carrying certificate should bundle both the STARK proof and a Lean-checkable witness that the circuit implements the specification, but no such mechanism is proposed.

## Verdict

The formal verification strategy shows genuine sophistication—the must-prove/should-prove stratification and decidability characterization are contributions to verification methodology. However, the critical gaps in Hopcroft minimization, circuit-specification equivalence, and the absence of a formal certificate specification mean the proof-carrying architecture is more aspirational than realized. Closing these gaps would yield a landmark contribution.

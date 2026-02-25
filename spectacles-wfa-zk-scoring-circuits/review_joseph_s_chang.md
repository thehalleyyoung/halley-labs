# Review: Spectacles — Verified Compiler from Semiring-Weighted Automata to Zero-Knowledge Scoring Circuits

**Reviewer:** Joseph S. Chang
**Persona:** Automated Reasoning & Logic Expert
**Expertise:** SMT solvers, automated theorem proving, proof certificates, decision procedures, solver engineering, formal logic

---

## Summary

Spectacles constructs a verified compiler from weighted finite automata (WFA) over typed semirings to STARK arithmetic circuits, with Lean 4 formalization of compilation soundness (Theorems 6.1/6.2). From an automated reasoning perspective, this project contributes a genuine decision procedure (WFA equivalence via Hopcroft minimization), mechanized soundness proofs, and reusable Lean 4 tactics (kleene_dec, wfa_equiv). However, the formalization is incomplete (15 sorrys), the proof certificate format is non-standard (STARK-specific, not LFSC/Alethe/DRAT), and several load-bearing mathematical claims — particularly the comparison gadget correctness over F_p for [0, 2^62) and transducer composition associativity — remain unproved axioms in the formal development.

## Strengths

1. **Genuine decision procedure for WFA equivalence.** The equivalence checker via Hopcroft minimization and coalgebraic bisimulation is a real decision procedure with known complexity bounds (O(n log n) for minimization), not an ad-hoc heuristic. Validation against brute-force enumeration on all strings up to length 20 for 1K random WFA pairs provides strong empirical support.
2. **Mechanized soundness proofs (Theorems 6.1/6.2).** The sorry-free proofs of circuit_sound_algebraic and circuit_sound_tropical in Lean 4 establish that STARK proof acceptance implies WFA acceptance — a non-trivial formal guarantee connecting cryptographic verification to automata-theoretic semantics.
3. **Two-tier proof architecture with honest separation.** Tier 1 (algebraic compilation via injective semiring homomorphism ι: S → F_p) and Tier 2 (gadget-assisted compilation with bit-decomposition for tropical min) are cleanly separated, with different proof strategies for each. This avoids false uniformity claims.
4. **Reusable proof automation.** The kleene_dec tactic (equational Kleene algebra via NFA equivalence) and wfa_equiv tactic (automaton equivalence via minimization) extend Lean 4's automation capabilities and have value independent of Spectacles.

## Weaknesses

1. **15 sorrys constitute unacceptable verification debt for "verified" claims.** In automated reasoning, a sorry is semantically equivalent to an axiom: it can introduce unsoundness. The 3 novel sorrys with proof sketches are particularly troubling because proof sketches are precisely the informal arguments that mechanized verification is supposed to replace. The 12 "routine" sorrys (closable by omega/simp/ring) are less dangerous but their continued presence suggests the proof development is unfinished. A paper claiming "verified compilation" with 15 sorrys would be rejected at a top formal methods venue (CPP, ITP, POPL) without qualification. The honest framing should be "partially verified" or "verified modulo 15 axioms."

2. **Transducer composition associativity is an unproved load-bearing axiom.** Transducer composition is used in metric decomposition — BLEU-4 decomposes into four composed n-gram computations. The associativity of this composition is an explicitly deferred sorry. If associativity fails (and it can fail for certain transducer classes — cf. Berstel & Reutenauer's Rational Series, §III.3), the decomposition-based proof strategy becomes unsound. This is not a routine sorry; it is a mathematical claim that requires careful proof, and its failure would invalidate the compositional verification approach.

3. **Proof certificates are STARK-specific and non-interoperable.** The generated certificates are FRI-based STARK proofs, not in any standardized proof format (LFSC, Alethe, DRAT, CPC). This means: (a) no existing proof checker outside the STARK ecosystem can verify them, (b) no certificate translation to other proof systems is possible, and (c) the "trust base" includes the entire STARK verifier implementation, which is itself unverified Rust code. In the proof certificate community, a certificate's value depends on the simplicity and trustworthiness of its checker — a STARK verifier is vastly more complex than a DRAT checker.

4. **Comparison gadget correctness over F_p is assumed, not proved.** The comparison gadget (used in Tier 2 tropical compilation for min/max operations) is claimed correct for values in [0, 2^62) within the Goldilocks field F_p (p = 2^64 − 2^32 + 1). This correctness relies on bit-decomposition constraints enforcing range, but the formal proof that these constraints are both sound (every satisfying assignment represents a valid comparison) and complete (every valid comparison has a satisfying assignment) is not provided in the Lean formalization. The bit-decomposition proof is exactly the kind of fiddly arithmetic reasoning where bugs hide.

5. **No proof complexity analysis or tight bounds.** The circuit width scales as O(|Q|² × |Σ|) per WFA step, and total proof size scales with trace length × circuit width. No tight bounds on proof generation complexity as a function of WFA parameters are proved or empirically characterized beyond 512 states. For an automated reasoning audience, understanding the computational complexity of the proof procedure is essential — is proof generation in P? NP? PSPACE? The FRI-based STARK prover has quasi-linear proving time in circuit size, but the circuit size itself may scale super-linearly in the WFA parameters.

6. **wfa_equiv tactic completeness is unproved.** The WFA equivalence tactic is sound (validated by testing) but its completeness — that every truly equivalent WFA pair is recognized as equivalent — is formally deferred. Without completeness, the tactic may reject equivalent specifications as distinct, producing false negatives in metric equivalence checking. For a decision procedure, soundness without completeness means you have a semi-decision procedure, not a decision procedure.

## Questions for Authors

- Can the 12 routine sorrys be closed by a single automated pass (e.g., `aesop` with custom simp lemmas, or `omega` after appropriate unfolding)? If so, why hasn't this been done?
- Is transducer composition associativity provable for the specific transducer class used in Spectacles (e.g., subsequential or functional transducers), even if it fails for general transducers?
- Have you considered generating proof certificates in a standardized format (e.g., LFSC or Alethe) alongside the STARK proofs, enabling independent verification by a trusted kernel?

## Overall Assessment

Spectacles makes the most substantive automated reasoning contribution of the projects reviewed: a genuine decision procedure, mechanized soundness proofs, and reusable proof tactics. The two-tier compilation architecture and KleeneSemiring formalization demonstrate real proof engineering sophistication. However, the 15 remaining sorrys — particularly the unproved transducer composition associativity and the assumed comparison gadget correctness — create genuine soundness risks that the paper underemphasizes. The non-standard proof certificate format limits interoperability and inflates the trust base. The unproved completeness of wfa_equiv downgrades the "decision procedure" to a semi-decision procedure. The formalization is impressive _for a work in progress_ but does not yet meet the standard of "verified" as understood by the automated reasoning community.

**Score:** 7/10
**Confidence:** 5/5

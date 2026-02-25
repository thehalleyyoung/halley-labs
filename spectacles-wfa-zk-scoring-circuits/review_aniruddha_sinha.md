# Review: Spectacles — Verified WFA-ZK Scoring Circuits for Contamination-Certified Evaluation

**Reviewer:** Aniruddha Sinha (Model Checking & AI Applicant)  
**Expertise:** Temporal logic model checking, automata-theoretic verification, TLA+ specification, counterexample-guided abstraction refinement (CEGAR), state-space exploration for protocol verification  
**Score:** 7/10  
**Recommendation:** Weak Accept

---

## Summary

Spectacles presents a novel architecture connecting weighted finite automata equivalence with zero-knowledge proof systems to build verifiable evaluation certificates. From a model-checking perspective, the TLA+ specification of the commit-then-reveal protocol and the automata-theoretic foundations are sound in conception, but several verification gaps—particularly around Hopcroft minimization and state-space explosion in transducer composition—limit the formal assurance claims.

## Strengths

**1. TLA+ Specification of Commit-Then-Reveal is Well-Scoped.** The commit-then-reveal protocol (addressing threat A2: different outputs) is specified in TLA+ with explicit safety and liveness properties. The safety property—that a committed evaluation cannot be retroactively altered—is naturally expressible as an invariant, and the liveness property—that honest participants eventually receive certificates—avoids the common trap of vacuous liveness in Byzantine settings by restricting to semi-honest adversaries. The refinement mapping from the abstract specification to the BLAKE3-based implementation is a model-checking best practice.

**2. Schützenberger's Decidability Result is Correctly Applied.** The paper correctly invokes the 1961 decidability result for WFA equivalence over commutative semirings, and extends it to the specific semirings needed for NLP metrics. The decidability characterization is precise: Boolean and counting semirings are commutative and Noetherian, so minimization terminates; the tropical semiring requires the additional bit-decomposition gadget because min/max operations break the polynomial-time equivalence algorithm. This automata-theoretic precision is rare in applied ZK work.

**3. Coalgebraic Bisimulation Provides Modular Proof Structure.** The choice of coalgebraic bisimulation over classical language-equivalence proofs for the `wfa_equiv` tactic is well-motivated: bisimulation is compositional under parallel and sequential composition of automata, which matters when metric computations are built from composed WFA stages (tokenization → n-gram counting → aggregation). This compositionality would not hold for trace-equivalence-based proofs.

**4. Threat Model Decomposition Maps Cleanly to Verification Obligations.** The A1→G1, A2→G3, A3→G2 mapping creates a clean separation of verification concerns. Each guarantee can be model-checked independently: G1 as a functional correctness property of the STARK circuit, G3 as a temporal property of the commit protocol, and G2 as a privacy property of the PSI protocol. This decomposition enables incremental verification, which is the right engineering approach for a system of this scale.

## Weaknesses

**1. Hopcroft Minimization Correctness is Deferred.** The `wfa_equiv` tactic depends on Hopcroft minimization to compute canonical forms, but this is classified as "should-prove" rather than "must-prove." In model-checking terms, this is equivalent to trusting the state-space reduction without verifying the bisimulation quotient—precisely the kind of assumption that leads to spurious counterexamples in CEGAR. The Lean formalization of coalgebraic bisimulation is incomplete without this component; the gap between "equivalent up to bisimulation" and "minimized automaton is canonical" is where bugs hide.

**2. State-Space Explosion in Transducer Composition is Unaddressed.** Composing WFA stages (e.g., tokenizer × n-gram counter × aggregator) produces a product automaton whose state space is multiplicative. For ROUGE-L, which involves longest common subsequence computation, the transducer state space is O(m × n) where m, n are sequence lengths. The paper does not analyze whether the resulting STARK circuit constraint count scales polynomially or whether intermediate state-space explosion makes certain metrics computationally infeasible in practice.

**3. TLA+ Model Covers Only Semi-Honest Adversaries.** The TLA+ specification assumes semi-honest behavior, but the threat model explicitly considers adversaries who "inflate scores" (A1) or "use different outputs" (A2)—these are malicious behaviors. The semi-honest assumption means the TLA+ model cannot capture the most interesting attacks. A CEGAR-style approach that refines the adversary model based on discovered attack traces would strengthen the verification, but is not discussed.

**4. No Explicit Fairness or Liveness Verification for PSI Protocol.** The PSI protocol for contamination detection involves multi-round communication between the evaluator and a data holder. The paper provides no TLA+ specification for this protocol, meaning liveness (the PSI eventually terminates with a correct result) and fairness (neither party can stall indefinitely) are unverified. Given that PSI is the mechanism for the contamination guarantee G2, this is a significant gap.

**5. CEGAR-Style Refinement Could Address the Lean-Rust Gap.** The paper identifies the Lean-Rust semantic gap as a risk but proposes only differential testing. A CEGAR-style approach—abstracting the Rust implementation as a Lean model, checking properties, and refining based on counterexamples—would provide stronger guarantees than testing alone. The absence of this well-known technique from the verification strategy is a missed opportunity, especially given the 117–142K LoC implementation scale.

## Verdict

Spectacles demonstrates strong automata-theoretic foundations and a well-structured threat model decomposition, but the deferred Hopcroft minimization proof and semi-honest TLA+ model undercut the formal assurance claims. Addressing these with CEGAR-style refinement and explicit protocol specifications would make the verification story compelling.

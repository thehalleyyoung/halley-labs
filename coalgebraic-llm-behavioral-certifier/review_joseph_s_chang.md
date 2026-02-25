# Review: CABER — Coalgebraic Behavioral Auditing of Foundation Models

**Reviewer:** Joseph S. Chang
**Persona:** Automated Reasoning and Logic Expert
**Expertise:** Temporal logic model checking, fixed-point computation, decidability, proof systems, formal verification methodology

---

## Summary

CABER implements a QCTL_F model checker for behavioral verification of learned automata approximating LLM behavior. The logic combines quantitative temporal operators with graded satisfaction, checked via fixed-point computation on finite automata. The model checking component is technically competent but not methodologically novel — QCTL_F is correctly identified as an instantiation of the Pattinson-Schröder coalgebraic modal logic framework. The main concern is the absence of Lean 4 formalization for the core theorems, with 38 property-based tests serving as an inadequate substitute.

## Strengths

1. **QCTL_F is an appropriate logic for probabilistic behavioral verification.** Graded satisfaction degrees in [0,1] naturally model the probabilistic nature of LLM behavior. The until, globally, and finally operators are sufficient for the specified behavioral properties. The alternation-free fragment being P-complete is correctly identified and practically relevant.

2. **Fixed-point model checking is correctly implemented.** The iterative computation of greatest and least fixed points for ACTL/ECTL formulas follows the standard Tarski-Knaster construction. The convergence criterion (tolerance-based) is appropriate for the quantitative setting.

3. **Specification templates map safety concerns to temporal formulas naturally.** RefusalPersistence as AG(harmful_prompt → AX(AG(refusal_mode))) correctly formalizes multi-turn persistence. The templates demonstrate good domain understanding.

4. **End-to-end error composition is a genuine reasoning contribution.** Theorem 3's sequential conditioning argument across five pipeline stages is non-trivial and correctly handles dependencies between stages. The proof structure is sound even if not mechanized.

5. **Property-based tests cover meaningful invariants.** Tests for Hoeffding bounds, bandwidth monotonicity, Galois connection distortion, and PAC error decomposition verify specific instances of the theoretical claims.

## Weaknesses

1. **No Lean 4 formalization for a paper claiming formal guarantees.** The core theorems (PCL* convergence, bandwidth-sample bound, CoalCEGAR monotonicity, error composition) are paper proofs. Property-based tests check instances (e.g., "bandwidth is sublinear for these 5 inputs") but do not verify the proofs themselves. For a paper whose central contribution is formal behavioral guarantees, this gap is fundamental.

2. **QCTL_F is not novel — it is an acknowledged instantiation.** The paper correctly positions QCTL_F as an instantiation of the Pattinson-Schröder framework. This means the logic and its decidability/complexity properties are inherited, not contributed. The novelty lies in the application to learned automata, not in the logic itself.

3. **Witness generation is underspecified.** The ModelCheckResult includes witnesses (traces) but the paper does not describe how witnesses are computed or what guarantees they satisfy. For counterexample-guided reasoning, the witness quality is critical.

4. **Fixed-point convergence on approximate automata is not analyzed.** The model checker assumes an exact finite automaton, but the learned automaton is an approximation with PAC error bounds. How do model checking results change as the automaton approximation quality varies? The error composition (Theorem 3) addresses this at a high level, but the interaction between approximation error and fixed-point convergence is not analyzed at the model-checking level.

5. **No model-checking-specific optimizations.** The implementation does not appear to use BDD-based symbolic model checking, partial order reduction, symmetry reduction, or other standard optimizations. For automata with ≤40 states this is acceptable, but scalability to larger automata is unaddressed.

6. **The 38 property tests bridge a "gap" that should not exist.** The paper acknowledges that the tests "bridge the gap" to Lean 4 formalization. But for the specific theorems claimed (convergence, sample complexity, error composition), property testing is categorically insufficient — it can find bugs but cannot verify proofs. The framing suggests the tests substitute for formalization, which they do not.

## Novelty Assessment

The model checking component is competent but not novel. The novelty lies in applying it to learned LLM behavioral automata, not in the logic or algorithms themselves. **Low model-checking novelty, moderate application novelty.**

## Suggestions

1. Mechanize at least Theorem 1 (PCL* convergence) in Lean 4 to demonstrate feasibility and validate the proof.
2. Analyze the interaction between automaton approximation error and model-checking fixed-point convergence.
3. Describe witness generation and its guarantees.
4. Consider BDD-based or symbolic model checking for scalability.
5. Be explicit that QCTL_F is inherited from the Pattinson-Schröder framework rather than contributed.

## Overall Assessment

The model checking component is technically sound but not novel. The core contribution is applying temporal verification to learned LLM behavioral automata, which is a worthwhile direction. However, the absence of Lean 4 formalization for a paper claiming formal guarantees is a significant gap that property-based tests cannot fill. The work would be substantially strengthened by mechanizing the core theorems and demonstrating the approach on real LLMs.

**Score:** 6/10
**Confidence:** 5/5

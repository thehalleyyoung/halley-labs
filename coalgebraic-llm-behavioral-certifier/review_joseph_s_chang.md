# Review: CABER — Coalgebraic Behavioral Auditing of Foundation Models

**Reviewer:** Joseph S. Chang (Automated Reasoning & Logic Expert)  
**Expertise:** Automata learning theory, decidability and complexity of temporal logics, interactive theorem proving (Lean 4, Coq), fixed-point theory, formal language theory  
**Score:** 7/10  
**Recommendation:** Weak Accept

---

## Summary

CABER presents a theoretically ambitious framework combining probabilistic automata learning (PCL*), a novel quantitative temporal logic (QCTL_F), and coalgebraic CEGAR for LLM auditing. The automata-theoretic and logical foundations are the paper's strongest contributions, though the Lean 4 formalization claim requires significant qualification.

## Strengths

**1. PCL* is a genuine theoretical contribution to automata learning.** Extending Angluin's L* to probabilistic oracles with formal PAC convergence is non-trivial and goes beyond prior work on learning probabilistic automata (e.g., Clark & Thollard's ALERGIA, Balle & Mohri's spectral methods). The key novelty is the integration of approximate equivalence queries—using statistical hypothesis testing rather than exact membership—into the observation table framework while maintaining the loop invariant that the table is closed and consistent up to tolerance ε. The resulting algorithm correctly handles the tension between exploration (refining the table) and exploitation (terminating with a sufficiently accurate hypothesis), which is the central difficulty in learning stochastic systems from queries.

**2. QCTL_F fixed-point semantics are well-founded.** The quantitative interpretation of QCTL_F formulae as functions from states to [0,1] (rather than Boolean satisfaction) is carefully constructed. The least/greatest fixed-point operators are defined via Knaster-Tarski on the complete lattice [0,1]^S, and the paper correctly identifies that monotonicity of the semantic operator follows from the monotonicity of the Kantorovich lifting of the transition functor. This ensures that fixed-point computation terminates and yields unique fixed points, which is essential for the soundness of model checking.

**3. Decidability analysis for model checking is thorough.** The paper provides a clear complexity analysis: model checking QCTL_F over finite-state coalgebras with rational transition probabilities is in P (via LP reduction), while the full logic with nested fixed points is in NP ∩ coNP (analogous to the μ-calculus). The alternation hierarchy for quantitative fixed points is correctly identified as collapsing at level 2 for the specific functor class used, which is a useful structural result that simplifies implementation.

**4. Automata-theoretic foundations connect cleanly to verification.** The extracted probabilistic automaton is simultaneously a coalgebra for the behavioral functor and a Markov chain amenable to standard probabilistic verification. This dual perspective is well-exploited: coalgebraic bisimilarity provides the behavioral equivalence notion, while the Markov chain view enables efficient computation via matrix methods. The paper correctly identifies that the Kantorovich bisimilarity metric on the coalgebra coincides with the probabilistic bisimilarity distance of Desharnais et al. for the specific functor used, grounding the abstract theory in established results.

## Weaknesses

**1. Lean 4 formalization claim is not credible as stated.** The paper claims the framework is amenable to Lean 4 formalization but provides no formalized components. Having worked on formalizations of comparable complexity (probabilistic automata in Coq required ~15K LoC for basic definitions and ~40K LoC for key theorems), I estimate that formalizing CABER's core theory—PCL* convergence, QCTL_F soundness, CoalCEGAR termination—would require 80,000-120,000 lines of Lean 4, involving formalization of measure theory (partially available in Mathlib), Kantorovich metrics (not in Mathlib), coalgebraic bisimilarity (not in Mathlib), and PAC learning theory (not in Mathlib). This is a multi-year effort for a team of 3-4 formalization experts. The paper should either remove this claim or scope it to specific, achievable sub-goals (e.g., formalizing the observation table invariants).

**2. PCL* termination guarantee depends on an unrealistic assumption.** The proof that PCL* terminates relies on the assumption that the target system has a finite number of behaviorally distinct states up to tolerance ε. For LLMs, this assumption is plausible but unverifiable: we cannot know a priori that the model's behavior on the input space Σ*_≤n admits a finite quotient at the chosen granularity. The paper should discuss conditions under which this assumption might fail (e.g., if the model implements a counting mechanism that creates unboundedly many distinguishable states) and provide heuristic termination criteria for the non-terminating case.

**3. Connection to existing probabilistic automata learning is underexplored.** The paper cites Angluin's L* and its deterministic extensions but does not adequately compare PCL* with existing algorithms for learning probabilistic automata: ALERGIA (frequency-based merging), MDI (minimum divergence inference), and Balle & Mohri's spectral methods. Each of these has known sample complexity bounds, and a formal comparison of PCL*'s Õ(β·n·log(1/δ)) with these bounds would clarify whether the coalgebraic framing provides concrete improvements or merely repackages known results in new notation.

**4. Fixed-point alternation depth is not analyzed for specification templates.** The paper provides a general complexity analysis for QCTL_F model checking but does not characterize the alternation depth of the specification templates actually used (Refusal Persistence, Paraphrase Invariance, etc.). If all practical specifications fall within the alternation-free fragment (which I suspect they do), the P-time complexity result suffices and the NP ∩ coNP upper bound is irrelevant. Conversely, if some specifications require genuine alternation, concrete examples should be provided to motivate the full generality of the logic.

## Verdict

CABER's automata-theoretic and logical foundations are technically sound and represent genuine contributions to the intersection of formal methods and AI evaluation. The Lean 4 claim should be scoped down, and the comparison with existing probabilistic automata learning algorithms needs strengthening. With these revisions, the theoretical core merits acceptance.

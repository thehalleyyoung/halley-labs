# Review: CABER — Coalgebraic Behavioral Auditing of Foundation Models

**Reviewer:** Aniruddha Sinha (Model Checking & AI Applicant)  
**Expertise:** Probabilistic model checking, CEGAR abstraction refinement, PRISM/Storm tool development, temporal logic expressiveness, counterexample-guided verification  
**Score:** 7/10  
**Recommendation:** Weak Accept

---

## Summary

CABER introduces CoalCEGAR and QCTL_F as coalgebraic generalizations of classical CEGAR and temporal logic for auditing LLMs treated as black-box transition systems. The model checking integration is technically interesting but faces fundamental questions about scalability to learned automata with uncertain transition probabilities.

## Strengths

**1. CoalCEGAR adapts CEGAR to quantitative settings with genuine novelty.** Classical CEGAR operates over Boolean abstractions with exact counterexamples; CoalCEGAR must handle quantitative Galois connections where the abstraction-concretization pair introduces bounded metric distortion rather than exact preservation. The Kantorovich-Lipschitz degradation bounds during refinement are a clean theoretical contribution. The key insight—that refinement should minimize the Kantorovich distance between the abstract and concrete behavioral functors rather than eliminating spurious counterexamples—represents a meaningful departure from the classical framework that could influence future work in quantitative verification.

**2. Polynomial-time model checking for QCTL_F is well-established.** The paper correctly identifies that restricting QCTL_F to finite-state coalgebras with rational transition probabilities yields a model checking problem solvable in polynomial time via linear programming over the Bellman equations for the quantitative fixed-point semantics. The reduction to a sequence of LP instances, one per modal depth, is elegant and practically implementable. The connection to PRISM's value iteration engine is natural and well-described.

**3. Principled integration with existing probabilistic model checkers.** Rather than building a verification engine from scratch, CABER translates the extracted automaton into PRISM's input format and lifts the quantitative semantics to Storm's sparse engine for larger state spaces. This pragmatic choice leverages decades of engineering investment in these tools and makes the approach immediately usable.

**4. Specification templates bridge formal methods and practitioner needs.** The mapping from human-readable behavioral contracts (Refusal Persistence, Paraphrase Invariance) to QCTL_F formulae is well-designed. The template instantiation mechanism, where practitioners fill in domain-specific predicates while the temporal structure is fixed, is a practical contribution that could see adoption independent of the rest of the framework.

## Weaknesses

**1. State-space explosion is not adequately addressed for learned automata.** Classical CEGAR succeeds because the abstract model starts small and grows incrementally. In CABER, the PCL* algorithm may produce automata with hundreds of states even for moderate alphabet sizes, and each CEGAR refinement step potentially doubles the state space by splitting abstract states. The paper does not provide any symbolic encoding (BDDs, MTBDDs) or compositional reasoning strategy to manage this growth. For the claimed 60-80K LoC implementation, the absence of any discussion of symbolic model checking integration with CUDD or Sylvan is a significant gap.

**2. QCTL_F expressiveness relative to PCTL*/rPATL is unclear.** The paper claims QCTL_F subsumes PCTL by instantiating the functor appropriately, but the proof sketch is incomplete—specifically, the encoding of nested probabilistic operators P_{≥p}[φ U ψ] in the functor-parameterized modality is not shown to preserve the quantitative semantics exactly. Without this, it is unclear whether QCTL_F is strictly more expressive, equally expressive, or incomparable with PCTL* for the specific class of coalgebras arising from LLM behavioral extraction. A formal expressiveness comparison, ideally with separation examples, is needed.

**3. Counterexample interpretation is fundamentally different from classical CEGAR.** In classical CEGAR, a spurious counterexample is a concrete path that witnesses the violation in the abstract model but not in the concrete one, and refinement eliminates it. In the quantitative setting, counterexamples are graded—a property may hold to degree 0.7 in the abstract model and 0.8 in the concrete one. The paper does not adequately explain how the refinement oracle decides when to refine versus when to accept the approximation, nor does it provide formal termination guarantees for the quantitative refinement loop.

**4. PRISM/Storm integration novelty is limited.** The translation from extracted automata to PRISM's DTMC/MDP format is straightforward—it amounts to emitting a transition matrix in PRISM's guarded-command syntax. While practical, this is engineering rather than research contribution. The paper would benefit from identifying specific limitations of existing model checkers that CABER's coalgebraic framing overcomes, rather than presenting standard tool integration as a contribution.

## Verdict

The model checking contributions are technically sound but need sharper differentiation from existing quantitative verification frameworks. CoalCEGAR's theoretical novelty is real, but practical scalability and expressiveness claims need stronger evidence. A formal comparison with PCTL* and symbolic state-space management would elevate this work significantly.

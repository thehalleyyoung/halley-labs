# Review by Joseph S. Chang (automated_reasoning_and_logic_expert)

## Project: LiquidPy — Guard-Harvesting Constraint Verification of Neural Network Computation Graphs via Domain-Specific SMT Theories

**Reviewer Expertise:** Automated reasoning, logic, proof systems, decision procedures, complexity theory.

**Recommendation: Weak Accept**

---

### Summary

LiquidPy contributes domain-specific SMT theories for tensor operations, Tinelli-Zarba theory combination for mixed finite/infinite sorts, an NP-completeness result for reshape satisfiability, and SMT-LIB proof certificates. I evaluate the logical and proof-theoretic contributions.

### Strengths

1. **Tinelli-Zarba is correct and well-implemented.** theory_combination.py uses restricted growth strings for canonical partition enumeration and correctly takes the cross-product across sorts. Push/pop on the Z3 solver tests each arrangement without side effects—a detail often implemented incorrectly.

2. **Broadcast axiomatization has proper formal structure.** Axioms A1-A6 form a consistent specification of NumPy broadcast semantics. The QF_LIA reduction for concrete ranks is sound: each dimension pair generates a finite disjunction, and the conjunction is decidable.

3. **Trust boundary documentation is unusually honest.** Both the paper and Lean file explicitly list what is and is not mechanized. The certificate trust boundary is stated clearly. This honesty about trust assumptions is rare.

### Weaknesses

1. **NP-completeness claim is underspecified.** The paper states a PARTITION reduction (Proposition 4.3) but the formal decision problem is imprecise. What exactly is the input? If it's "given symbolic dimensions d₁,...,dₙ and target (t₁,...,tₘ), does ∏dᵢ = ∏tⱼ have a solution?"—this is trivially in NP and the PARTITION encoding needs clarification. Furthermore, NP-completeness of the abstract problem doesn't predict practical hardness: reshape timing shows 93.9ms average with zero timeouts. The paper should provide the full reduction or downgrade to "NP-hard in theory, polynomial in practice."

2. **The Lean mechanization proves less than it appears.** `combination_soundness` states: if all solvers agree on an arrangement (by assumption each solver is sound), then the combination is satisfiable. The proof is two lines of unpacking. The soundness of each solver is an *axiom* in the `TheorySolver` class. `broadcast_symmetric` (genuine case analysis) and `mha_head_dim_sound` (using `Nat.div_add_mod`) contain more actual proof content. A best-paper submission would mechanize completeness: if the combined theory is satisfiable, enumeration finds a satisfying arrangement.

3. **Certificates are not cross-solver verified.** The paper claims certificates work with "Z3, CVC5, or any conforming solver" but provides no evidence. If UserPropagator lemmas are `assert`ed as SMT-LIB axioms, the certificate reduces to "assuming our axioms, constraints are UNSAT"—which any solver confirms trivially. Demonstrate at least one end-to-end verification with CVC5.

4. **No completeness result for the product theory.** The 4 false negatives on Suite B demonstrate practical incompleteness, but the paper doesn't characterize sources: encoding gaps, incomplete propagation, or inherent limitations? For finite-domain theories, Theorem 7's completeness proof relies on "exhaustive enumeration" but the UserPropagator doesn't literally enumerate all n^k assignments—it relies on Z3's SAT solver.

5. **The soundness chain has unacknowledged gaps.** The argument is: Lean proves combination soundness → UserPropagators implement theories → certificates capture proofs. But Lean-to-Python and Python-to-SMT-LIB links are unformalized. Both gaps sit in the TCB.

### Grounding Assessment

All grounding.json claims are supported. The NP-completeness claim is weakest—evidence type "proof" with location "paper Section 4.3" but no detailed reduction. Not hallucination—likely correct—but unsubstantiated.

### Path to Best Paper

(1) Provide the full PARTITION → reshape reduction, ideally mechanized; (2) mechanize completeness of arrangement enumeration; (3) demonstrate cross-solver certificate verification; (4) formalize the decidable fragment precisely.

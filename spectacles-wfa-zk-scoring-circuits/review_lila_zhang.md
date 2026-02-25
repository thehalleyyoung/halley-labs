# Review: Spectacles — Verified WFA-ZK Scoring Circuits for Contamination-Certified Evaluation

**Reviewer:** Lila Zhang (Symbolic Reasoning & AI Expert)  
**Expertise:** Coalgebraic methods, categorical semantics, algebraic type theory, domain-specific language design, semiring theory and Kleene algebras  
**Score:** 7/10  
**Recommendation:** Weak Accept

---

## Summary

Spectacles proposes an algebraically principled pipeline from NLP metric specifications through weighted finite automata over semirings to STARK-verified circuits, with contamination certification via PSI. The coalgebraic and algebraic foundations are the strongest aspect; the DSL design and categorical treatment of the two-tier compilation are less mature and introduce expressiveness gaps that limit the work's formal elegance.

## Strengths

**1. Coalgebraic Bisimulation for WFA Equivalence is the Right Abstraction.** The `wfa_equiv` tactic uses coalgebraic bisimulation (Rutten 2000) rather than classical Myhill-Nerode equivalence, which is the categorically correct approach: WFA over semirings are coalgebras for the functor F(X) = S × X^Σ (where S is the semiring and Σ the alphabet), and bisimulation is the canonical notion of behavioral equivalence for coalgebras. This gives the proof automatic compatibility with semiring homomorphisms—if two WFA are bisimilar over a semiring S, their images under any semiring homomorphism h: S → T are bisimilar over T. This is precisely the property needed for the Tier 1 algebraic embedding.

**2. KleeneSemiring Typeclass Design Fills a Genuine Gap.** The proposed Lean 4 KleeneSemiring typeclass, with the Kleene star axiomatized via the least fixpoint characterization (a* = 1 + a · a* with no smaller solution), is distinct from Mathlib's StarRing and C*-algebra structures. This is not merely a naming distinction: StarRing axiomatizes involutive algebras, while KleeneSemiring axiomatizes idempotent closure operators. The inclusion of both left and right unfold lemmas (star_unfold_left, star_unfold_right) shows awareness of the asymmetry in non-commutative semirings.

**3. Two-Tier Compilation Reflects Genuine Algebraic Obstruction.** The Tier 1/Tier 2 split is not arbitrary: counting and Boolean semirings admit injective homomorphisms into F_p because they are finitely generated commutative semirings embeddable in characteristic-p arithmetic. The tropical semiring (N ∪ {∞}, min, +) fails this because min is not a ring operation—there is no injective semiring homomorphism from the tropical semiring to any field. The bit-decomposition gadget in Tier 2 is the standard workaround (encoding min as a comparison circuit), and the paper correctly identifies this as a gadget-assisted embedding rather than a homomorphism.

**4. Categorical Structure of the Pipeline is Implicit but Sound.** The pipeline (EvalSpec → WFA → circuit → STARK proof) can be read as a sequence of functors between categories: EvalSpec expressions form an initial algebra, WFA form a category with semiring-weighted morphisms, and STARK circuits form a category of arithmetic constraint systems. The compilation stages are essentially structure-preserving functors, though the paper does not use this language. The fact that Tier 1 embeddings are natural transformations (commuting with the WFA operations) while Tier 2 requires gadgets (breaking naturality) is a clean categorical distinction.

## Weaknesses

**1. EvalSpec DSL Lacks Formal Denotational Semantics.** The EvalSpec DSL is described operationally (type-directed semiring selection) but has no denotational semantics. Without a formal semantics, one cannot state—let alone prove—that EvalSpec compilation is correct: "correct with respect to what?" A denotational semantics mapping EvalSpec terms to functions over weighted languages would close this gap and enable a correctness theorem of the form "compile(e) ≅ ⟦e⟧" where ⟦·⟧ is the denotation.

**2. Algebraic vs. Gadget-Assisted Boundary is Under-Theorized.** The paper identifies which metrics fall into Tier 1 vs. Tier 2 but does not characterize the boundary algebraically. The natural question—"for which semirings S does there exist an injective semiring homomorphism S → F_p for some prime p?"—has a known answer (S must be a finite commutative ring with characteristic dividing p), but the paper does not state this, leaving the reader uncertain whether future metrics might require a hypothetical Tier 3.

**3. KleeneSemiring Axiomatization Omits the Equational Theory.** The typeclass provides star_unfold_left and star_unfold_right but does not include the full Conway/Kozen axioms for Kleene algebra (e.g., the induction axioms b + a·x ≤ x ⇒ a*·b ≤ x). Without these, the typeclass axiomatizes a weaker structure than Kleene algebra, and some WFA equivalences that hold in Kleene algebra may not be provable. The paper should clarify whether the weaker axiomatization suffices for all seven metrics or whether the full Kozen axioms are needed.

**4. Functor Composition is Not Verified End-to-End.** Each compilation stage is partially verified (WFA equivalence via bisimulation, STARK soundness via FRI), but the composition of stages—EvalSpec → WFA → circuit → proof—is not verified as a single end-to-end functor. In categorical terms, the paper verifies individual morphisms but not their composition, which is where semantic gaps (Lean-Rust, WFA-circuit) can introduce inconsistencies.

**5. No Treatment of Weighted Language Inclusion (Only Equivalence).** The `wfa_equiv` tactic decides equivalence, but many practical questions involve inclusion: "does metric A always upper-bound metric B?" Weighted language inclusion is undecidable for general semirings but decidable for certain ordered semirings (e.g., the tropical semiring). The paper's exclusive focus on equivalence misses opportunities for comparative metric analysis that practitioners would find useful.

## Verdict

The coalgebraic and algebraic foundations are genuinely strong—the KleeneSemiring typeclass and the two-tier algebraic obstruction analysis are contributions in their own right. However, the EvalSpec DSL needs formal semantics, and the end-to-end composition of compilation stages remains unverified, preventing the work from achieving the full formal elegance its algebraic foundations promise.

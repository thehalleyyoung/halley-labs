# Review: CABER — Coalgebraic Behavioral Auditing of Foundation Models

**Reviewer:** Lila Zhang (Symbolic Reasoning & AI Expert)  
**Expertise:** Category theory in computer science, coalgebraic semantics, compositional verification, sheaf theory, functorial data migration  
**Score:** 7/10  
**Recommendation:** Weak Accept

---

## Summary

CABER leverages coalgebraic semantics to abstract LLM behavior into finite-state systems amenable to formal verification. The categorical framing is the paper's most distinctive contribution, enabling a modular architecture where the behavioral functor parameterizes the entire verification pipeline. However, the alphabet abstraction layer introduces a pre-categorical discretization that partially undermines the compositional elegance.

## Strengths

**1. Categorical compositionality is a genuine architectural advantage.** The behavioral functor F_LLM(X) = (Σ_≤k × D(X))^{Σ*_≤n} is well-chosen: it cleanly factors the observable behavior into output production (Σ_≤k), probabilistic transition (D(X)), and input consumption (Σ*_≤n). Crucially, this factorization is functorial—natural transformations between different functor instantiations correspond to meaningful behavioral relationships (coarsening/refining output granularity, extending/restricting input depth). This enables a lattice of behavioral models connected by coalgebra morphisms, which is precisely the right mathematical structure for multi-granularity auditing. No competing framework (HELM, CheckList, DeepMind's model cards) offers this compositionality.

**2. Sub-distribution functor choice handles partial observations correctly.** The use of D(X) as the sub-probability distribution functor (rather than full distributions) correctly models the possibility that some inputs lead to API failures, timeouts, or out-of-distribution responses that don't map to any abstract state. This is a subtle but important modeling choice that most papers in this space get wrong by assuming total transition functions. The resulting coalgebras are partial, and the bisimilarity theory extends cleanly via the Kantorovich metric on sub-distributions.

**3. Functor parameterization enables systematic sensitivity analysis.** By varying the functor parameters (output alphabet size k, input depth n, equivalence tolerance ε), practitioners can systematically explore the trade-off between model fidelity and verification tractability. The paper demonstrates this with a parameter sweep showing how increasing k beyond ~15 yields diminishing returns in behavioral discrimination while exponentially increasing query complexity—a useful practical guideline grounded in the categorical structure.

**4. Natural transformation framework for version comparison.** The construction of a span of coalgebra morphisms between automata extracted from different API versions is categorically clean and provides a principled notion of behavioral regression. The pullback construction for computing the maximal common behavioral quotient is a nice application of standard categorical machinery to a novel problem domain.

## Weaknesses

**1. Alphabet abstraction breaks functoriality at the system boundary.** The two-phase clustering (response statistics + embedding interpolation) that constructs the abstract alphabet Σ is fundamentally a non-functorial operation: it depends on the specific sample of responses observed during Phase 0, and small changes to the sample can produce incompatible alphabets. This means the entire coalgebraic framework sits atop a non-compositional foundation. The CEGAR refinement loop partially addresses this by adapting the alphabet, but the refinement itself is not expressed as a natural transformation—it is an ad hoc re-clustering step that invalidates previous coalgebraic constructions. A cleaner approach would use a presheaf over a category of possible alphabets, with restriction maps encoding coarsening relationships.

**2. Missing connection to coalgebraic modal logic literature.** The QCTL_F logic is presented as novel, but the coalgebraic modal logic community (Pattinson, Schröder, Kupke) has extensively studied functor-parameterized modalities via predicate liftings. The paper does not position QCTL_F relative to this existing body of work, particularly Schröder's coalgebraic μ-calculus, which already provides a generic fixed-point logic parameterized by a functor. If QCTL_F is an instance of this framework, its novelty is primarily in the application domain rather than the logical foundations—which is fine, but should be stated clearly.

**3. Sheaf-theoretic extensions are mentioned but not developed.** The paper briefly suggests that the multi-granularity behavioral models could be organized as a sheaf over a site of abstraction levels, with the gluing axiom enforcing consistency across refinement levels. This is an attractive idea that would provide a genuine categorical contribution, but it remains at the level of a remark. Developing this direction—even informally—would significantly strengthen the paper's theoretical positioning within the coalgebraic community.

**4. Compositionality claim is limited to single-model analysis.** The functor framework is compositional in the sense of varying parameters, but CABER does not address compositional reasoning about systems of interacting LLMs (e.g., agent pipelines, retrieval-augmented generation). The coalgebraic approach is naturally suited to this via tensor products of coalgebras or comonadic composition, but the paper does not explore this direction, limiting the practical scope of the compositionality claim.

## Verdict

CABER's categorical foundations are its strongest differentiator and are largely well-executed. The alphabet abstraction's non-functoriality is the most significant theoretical gap, but it is addressable through presheaf-based constructions. With clearer positioning relative to coalgebraic modal logic and development of the sheaf-theoretic direction, this could become a foundational reference for categorical approaches to AI auditing.

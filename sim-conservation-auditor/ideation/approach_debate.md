# ConservationLint: Approach Debate

**Project:** sim-conservation-auditor  
**Phase:** Ideation — Adversarial Debate Synthesis  
**Date:** 2025-07-18  
**Inputs:** `approaches.md` (Domain Visionary), `math_depth_assessment.md`, `difficulty_assessment.md`, `skeptic_critique.md`

---

## Preamble

Three competing approaches to ConservationLint were proposed by the Domain Visionary:

- **Approach A ("Noether's Scalpel"):** A full static symbolic pipeline that parses Python simulation code, lifts it into a conservation-aware IR, performs Lie-symmetry and BCH analysis, and localizes conservation violations to source lines — all without executing user code.
- **Approach B ("Conservation Types"):** A graded effect type system where conservation laws are first-class types, integrator composition is type-checked against a grade lattice derived from Lie algebra structure, and well-typed programs are *guaranteed* to preserve declared conservation laws.
- **Approach C ("Shadow Oracle"):** A dynamic trace analysis tool that executes instrumented simulations, recovers the shadow Hamiltonian via sparse symbolic regression (SINDy), and localizes violations through ablation-based causal intervention on code regions.

Each approach was independently evaluated by three assessors — the Math Depth Assessor, the Difficulty Assessor, and the Adversarial Skeptic — with instructions to be critical, precise, and willing to challenge each other. This document compiles their findings into a structured adversarial debate, preserving direct quotes and genuine disagreements.

---

## Approach A: "Noether's Scalpel" — Full Static Symbolic Pipeline

### Math Depth Assessment

The Math Assessor finds Approach A's claimed mathematical contributions are thinner than presented:

- **T1 (Provenance-Tagged Modified Equation):** Scored as ~20% new math, ~80% known results with engineering annotations. The BCH expansion for operator splittings is "thoroughly established (Hairer, Lubich & Wanner 2006)." The mixed-order interaction terms are "a *refinement* of known theory, not a new theory." T1 is "weakly load-bearing" — the tool "could work with a simpler framework: compute the standard BCH expansion, then attribute terms to sub-integrators by syntactic inspection of which H_i appear in each Lie monomial."

- **T2 (Computable Obstruction Criterion):** This is the stronger claim, scored ~60% new *if* the efficient reduction works. "A *general*, efficient obstruction criterion exploiting free Lie algebra structure to avoid quantifier elimination would be a real contribution to computational algebra." But the proof faces a key obstacle: "the achievable correction terms at order n are polynomial (not linear) functions of the method coefficients. The set of achievable modified Hamiltonians is a semi-algebraic variety, and checking whether it intersects the conservation kernel is a semi-algebraic feasibility problem."

- **Differential symbolic slicing:** "New as a PL concept; mathematically shallow."

- **Overall depth: 4/10.** "T1 is below POPL/CAV threshold as a standalone theorem. T2, *if proven with the efficient reduction*, would be a solid 6/10." The math is "necessary scaffolding for the engineering contribution, not the headline act."

### Difficulty Assessment

The Difficulty Assessor rates A at **5/10 algorithmic novelty** and **8/10 integration risk**, totaling ~25K LoC for a paper-scope artifact (~8K genuinely novel):

- **Individual components are not novel:** "Tree-sitter parsing is off-the-shelf. Polynomial algebra engines exist. BCH expansion is textbook. Lie-symmetry analysis for polynomial ODEs is implemented in multiple CAS packages." The difficulty is "**integration engineering**, not algorithmic novelty."

- **Integration risk is severe:** Five pipeline stages (parse → lift → symmetry → BCH → slice) with fundamental abstraction mismatches: "The Lie-symmetry analyzer works on continuous ODEs; the BCH engine works on discrete compositions of flows; the slicer works on imperative code. These three representations use fundamentally different ontologies."

- **Paper-scope reality:** "A strong systems/PL PhD student with geometric integration background" in 6 months builds a "demo on 5–8 textbook integrators reproducing known results." They would NOT build "differential symbolic slicing to line-level granularity," "a general obstruction detector (T2) beyond hand-worked examples," or "coverage of real codebases (JAX-MD, Dedalus)."

### Adversarial Critique

**Fatal flaw: Code→math extraction will not work on real code.** Kill probability: **55%.**

The Skeptic dismantles the extraction assumption target-by-target:

- **JAX-MD:** "JAX traces Python to build `jaxpr` intermediate representations via `jit`/`vmap`/`grad`. The actual numerical kernel exists in a traced computation graph, not in syntactically parseable Python. Tree-sitter sees `jax.jit(lambda x: ...)` and learns nothing about the computation inside."

- **Dedalus:** "The mathematical semantics of an FFT-based spectral derivative is `d/dx` in Fourier space. Modeling this requires understanding that the code performs a forward FFT, multiplies by `ik`, and performs an inverse FFT — a semantic chain across multiple opaque library calls that no pattern matcher will reliably reconstruct."

- **NumPy broadcasting:** "Static analysis without shape inference is semantically unfaithful; shape inference on Python is a research problem in itself (see TorchDynamo's years-long struggle)."

- **LLM competition:** "~70% overlap on the analyzable fragment." For JAX-MD and Dedalus code ConservationLint cannot parse, "the LLM wins by default on the most important target." ConservationLint's unique value covers "**6% of the total problem space**" — the 30% rigor gap on the 20% liftable fragment.

- **Evaluation circularity:** "The benchmark suite will be written in the liftable fragment. Coverage will be reported on the benchmarks. The paper will claim '85% detection rate' while being unable to parse the first file of any real simulation framework."

### Cross-Challenges

**Skeptic → Visionary:**
> "You describe the code→math extraction as 'the make-or-break engineering challenge' and then propose a mitigation of 'restrict Phase 1 to pure-NumPy code with explicit loops.' This is an admission that the approach doesn't work on real code, wrapped in optimistic language about phasing. The 'liftable fragment' is not a research contribution — it's the fragment of code simple enough that you don't need a tool to analyze it. A researcher who writes explicit-loop Verlet in pure NumPy already understands their integrator's conservation properties."

**Math Assessor → Visionary:**
> "T1 is not a theorem — it is a data structure. Labeling Lie monomials by their generator origin is what every symbolic algebra implementation already does internally. Calling this a 'Provenance-Tagged Heterogeneous Composition Modified Equation Theorem' inflates routine bookkeeping into a mathematical contribution. The BCH formula for e^A e^B already tracks which operators contribute to which terms — that is literally what nested commutators encode."

**Difficulty Assessor → Visionary:**
> "The Visionary conflates 'hard to build' with 'algorithmically novel.' The individual algorithms (Lie symmetry, BCH, Gröbner bases, program slicing) are all well-studied. What's hard is making them work together on real code — but that's *engineering* difficulty, not *research* difficulty. The provenance-tagged BCH is clever bookkeeping, not a breakthrough."

**Skeptic → Math Assessor:**
> "You rate T1 as 'conditional value' and T2 as 'crown jewel,' but both are downstream of extraction. A crown jewel theorem implemented in a tool that can't parse its input is a pure math paper, not a systems contribution. If you believe T2 is strong enough to stand alone, recommend publishing it in *Numerische Mathematik* without the tool."

**Skeptic → Difficulty Assessor:**
> "Your 7/10 difficulty score is inflated by the extraction risk. Remove extraction (which may be impossible) and the remaining pipeline — BCH expansion, Lie symmetry analysis on polynomial systems, graph traversal for localization — is a 4/10. The difficulty is concentrated in one component that may be fatal, not distributed across the system."

**Difficulty Assessor → Math Assessor (on T2):**
> "For the cases the tool can actually handle (k ≤ 5, p ≤ 4), the obstruction check reduces to verifying ≤200 Lie bracket conditions — each a polynomial identity checkable by direct computation. This is a finite, brute-force calculation, not an elegant structural theorem."

### Consensus Verdict on A

**Genuinely strong:**
- T2 (obstruction detection) is a unique capability no other approach provides — the ability to prove a violation is *architecturally unfixable*. All assessors agree this is the crown jewel, though they disagree on its depth.
- The "code→math" bridge narrative is genuinely novel — no prior work operates in this direction for conservation properties.
- Differential symbolic slicing is new as a PL concept.

**Genuinely weak:**
- The extraction problem is existential and unvalidated. All three assessors independently flag it as the single point of failure. The Skeptic's JAX-MD/Dedalus/broadcasting analysis is unrefuted.
- T1 is consensus-weak: the Math Assessor calls it "a data structure," the Difficulty Assessor calls it "bookkeeping," and the Skeptic considers it downstream of extraction anyway.
- The 20–40% coverage estimate is fabricated. No prototype exists.

**What survives:** T2 as a standalone theorem (if the efficient reduction works), and the conceptual contribution of bridging geometric mechanics with program analysis. The *tool* survives only if extraction is replaced with framework-level tracing (the Skeptic's rescue proposal: "intercept at the framework level," making the approach semi-dynamic).

---

## Approach B: "Conservation Types" — Graded Effect System

### Math Depth Assessment

The Math Assessor scores B at **6/10 depth** — the highest ceiling but widest variance:

- **Connection Theorem (Lie algebra → effect grading):** "The bridge is genuinely new — no one has connected graded effect systems to Lie-algebraic structure of Hamiltonian mechanics." But "the novelty depends entirely on whether the isomorphism is discovered or stipulated." If the grade monoid is *defined* as the Lie algebra quotient, "there's no theorem to prove."

- **Soundness theorem:** "Load-bearing" and "tractable but laborious." Standard abstract interpretation framework applied to a non-standard domain.

- **Key vulnerability — grade lattice finitization:** "The free Lie algebra on k generators has Witt-formula dimension growing as O(k^n/n) at depth n. The type system requires a *finite* lattice for decidable type-checking. Truncating at depth p loses information about higher-order conservation properties." This truncation "may make the type system unsound for programs that compose many integrators."

- **Bimodal outcome:** "It could be a 8/10 POPL contribution or a 3/10 tautology."

### Difficulty Assessment

The Difficulty Assessor rates B at **7/10 theoretical novelty** but only **4/10 implementation difficulty**, with ~14K LoC for a paper-scope artifact:

- "The connection between Lie algebras and graded effect systems is genuinely novel. Graded monads and effect systems exist, but no one has instantiated them with a grade structure derived from BCH theory."

- The *tool* is simpler: "a Python library with grade annotations and a constraint checker — is a moderate-sized DSEL, well within standard PL-engineering practice."

- **Bimodal feasibility:** "The typed API is straightforward to build (feasibility 7/10) but nobody will use it (value 2/10). Grade inference on existing code is infeasible (feasibility 3/10) but would be valuable if it worked (value 7/10). The blended score of 4/10 hides this bimodal risk profile."

- A 6-month PhD student produces: "The Connection Theorem for a restricted setting, a typed primitive library covering Verlet/symplectic Euler/Strang splitting, 5–8 examples." They would NOT build: "Grade inference on real-world code. A usable tool that non-PL-experts would adopt."

### Adversarial Critique

**Fatal flaw: Nobody will adopt a new API to get conservation checking.** Kill probability: **70%** — highest of all three.

- **Adoption is fantasy:** "Simulation developers have invested months or years in their codebases. They will not rewrite working code in a new API to get conservation type-checking." The mitigation of partnering with framework developers "assumes that framework maintainers will voluntarily add conservation-grade annotations to their APIs. This has never happened in any domain."

- **The fallback is strictly worse Approach A:** "If grade inference on arbitrary code faces 'the same extraction challenges as Approach A,' then Conservation Types is Approach A + a type system that nobody uses."

- **LLM competition:** "An LLM + SymPy covers 85% of the practical use case at zero adoption cost." For composition queries like "does composing a symplectic integrator with a Nosé-Hoover thermostat preserve energy?" GPT-4 answers correctly from textbook knowledge.

- **Evaluation circularity:** "The evaluation demonstrates that the type system works on code written to work with the type system. This is definitionally circular."

- **Demand vacuum:** ~40 framework developers worldwide might benefit. "Approximately **zero** will rewrite their APIs to use a typed conservation library created by a research group they've never heard of."

### Cross-Challenges

**Skeptic → Visionary:**
> "You claim 'this is the only approach that makes conservation violations *impossible* for well-typed programs.' This is true and vacuous. Well-typed programs don't exist yet. Proving that a hypothetical category of programs has a nice property is a theorem about an empty set."

**Math Assessor → Visionary:**
> "The Connection Theorem is likely a tautology, and the 'beautiful' type system is a lookup table in disguise. If the grade is defined as 'the element of the truncated free Lie algebra quotient corresponding to the BCH expansion of the integrator's modified Hamiltonian,' then the Connection Theorem (grades ≅ Lie algebra quotient) is true by definition. The practical type system — for the 3-5 conservation laws that matter in practice — is a finite table of composition rules. You don't need the full Lie algebra machinery to encode 'symplectic + thermostat = energy-broken.' The proposal mistakes *generality* for *depth*."

**Difficulty Assessor → Visionary:**
> "The Visionary rates Approach B's difficulty at 8/10. I rate it 7/10 for theoretical novelty, 4/10 for implementation difficulty. The difficulty score is inflated because the Visionary is rating the *math* difficulty of the Connection Theorem, not the *software artifact* difficulty."

**Skeptic → Math Assessor:**
> "You rate this 8/10 on Potential based on the Connection Theorem. But you also rate it 4/10 on Feasibility. A result that is potentially brilliant but infeasible to demonstrate on real code is a pure math paper, not a systems contribution. If you believe the theorem is strong enough, recommend publishing it without the tool. If you believe the tool is necessary, the 4/10 feasibility kills the project."

**Skeptic → Difficulty Assessor:**
> "The 8/10 difficulty score conflates mathematical difficulty (designing the grade lattice, proving the Connection Theorem) with engineering difficulty (building a tool people use). The math is genuinely hard. The engineering is straightforward *given* user adoption of the typed API. The problem is that user adoption is ~0%, making the engineering irrelevant. Difficulty should be scored on the hardest *necessary* task, which is convincing framework developers to change their APIs — a social problem, not a technical one."

**Difficulty Assessor → Math Assessor (on the Connection Theorem):**
> "I doubt this. The BCH expansion on k generators produces an infinite-dimensional free Lie algebra; truncation at order p makes it finite, but the resulting lattice structure depends on the specific Noether pairing with each conserved quantity. For generic Hamiltonian systems, there's no reason to expect the quotient to have nice lattice-theoretic properties. More likely, the Connection Theorem will be a correct but unsurprising formalization."

### Consensus Verdict on B

**Genuinely strong:**
- The Lie algebra → graded effects connection is the only "genuinely surprising mathematical idea in the three proposals" (Difficulty Assessor). All assessors acknowledge this bridge is novel.
- Soundness theorem provides the strongest formal guarantee: well-typed programs *cannot* violate conservation. This is unique among the approaches.
- The "conservation laws as types" pitch resonates across PL and physics communities.

**Genuinely weak:**
- Adoption barrier is consensus-fatal. The Skeptic calls it "a theorem about an empty set," the Difficulty Assessor warns "nobody uses the typed API," and even the Math Assessor notes the Connection Theorem is "load-bearing for the intellectual contribution, not for the artifact."
- The Connection Theorem may be tautological. Both the Math Assessor and Difficulty Assessor flag the risk that the isomorphism is definitional rather than discovered.
- Grade inference on existing code inherits all of Approach A's extraction problems, offering no escape from the code→math bottleneck.

**What survives:** The Connection Theorem as a standalone theory paper at POPL/LICS — all three assessors independently converge on this recommendation. The Skeptic's rescue: "Abandon the typed API entirely. Publish the Connection Theorem as a pure theory paper." Or, build as a mypy plugin on existing code annotations rather than a new API.

---

## Approach C: "Shadow Oracle" — Dynamic Trace Analysis

### Math Depth Assessment

The Math Assessor scores C at **2/10 depth** — the shallowest approach:

- **Shadow Hamiltonian recovery:** "This is SINDy (Brunton, Proctor & Kutz 2016) applied to backward error analysis. The candidate library is Lie monomials instead of generic polynomial terms, but this is a choice of basis, not a new algorithm." Verdict: "~5% new math, ~95% application of SINDy + compressed sensing + elementary perturbation theory."

- **Convergence theorem:** "A specific instantiation of known compressed sensing theory." The tool "works without the theorem — SINDy works in practice without formal RIP verification on most real datasets."

- **Causal ablation sensitivity:** "This is not a theorem; it's a calculation."

- **Key vulnerability:** "RIP for deterministic Hamiltonian trajectories may fail precisely when it matters most." For integrable systems (where conservation is most important), trajectories are quasi-periodic on invariant tori — "highly correlated, violating the incoherence conditions that RIP requires. The math works where you don't need it and fails where you do."

### Difficulty Assessment

The Difficulty Assessor rates C at **4/10 difficulty** (lowest), with ~15.5K LoC for paper-scope and only ~8 person-months:

- **Every component builds on established techniques:** "SINDy, physical priors in regression, ablation-based causal attribution, Python instrumentation — the novelty is in the *combination* and *domain-specific application*."

- **Cleanest architecture:** "The pipeline is linear and each stage has well-defined inputs/outputs. There are no abstraction mismatches between continuous and discrete representations, no IR design tradeoffs, no compositionality–non-locality tension." Integration risk: **LOW.**

- **6-month PhD student ships a usable tool:** "A working `@conservation_audit` decorator for any Python time-stepper. Shadow Hamiltonian recovery for ≤20-variable systems. Ablation-based localization for integrators with explicitly separated force terms. Evaluation on 15–20 benchmark kernels including JAX-MD and SciPy ODE examples." This is "notably more complete than what 6 months buys for A or B."

### Adversarial Critique

**Fatal flaw: Sparse regression will not scale beyond toy systems.** Kill probability: **35%** — lowest of all three.

- **Scalability ceiling:** "For a 100-particle 3D system (600 DOF), 'naive polynomial regression is intractable.'" Mitigations are inadequate: particle permutation symmetry only helps homogeneous systems; multi-scale regression "multiplies computational cost by 5×"; the fallback to conservation-law-specific detection "is exactly what GROMACS `gmx energy` already does."

- **The "100% coverage" claim is misleading:** "The tool *runs* on 100% of code but *localizes* on ~5% (small systems with clean force decomposition). '100% detection, 5% localization' is a very different value proposition."

- **But the LLM cannot compete here:** "An LLM cannot execute code, collect trajectories, or perform sparse regression. The dynamic analysis capability is outside LLM reach." LLM overlap: only ~30%. "This is Approach C's genuine strength."

- **Demand vacuum is the mildest:** "The broadest audience of the three approaches." The decorator pattern is "low-friction." ~500 potential users vs. ~200 for A and ~40 for B.

### Cross-Challenges

**Skeptic → Visionary:**
> "You claim '100% coverage of runnable code' as the killer feature. This is technically true and practically misleading. The tool *runs* on 100% of code but *localizes* on ~5% (small systems with clean force decomposition). If you can't localize, you're a fancy energy monitor — and GROMACS already does that."

**Math Assessor → Visionary:**
> "There is no new math here. Call it what it is: SINDy for backward error analysis. The proposal packages a direct application of Brunton et al. (2016) with a physically motivated dictionary and calls it a 'Convergence Theorem for Shadow Hamiltonian Recovery.' Wrapping known techniques in theorem-statement formatting does not create mathematical depth."

**Difficulty Assessor → Visionary:**
> "The Visionary overrates the difficulty because the proposal makes the regression problem sound harder than it is. For the *paper-scope artifact*, the target is textbook Hamiltonian systems with ≤20 state variables — well within PySINDy's demonstrated capability. The Visionary claims 'No formal obstruction detection' as a 'genuine capability gap' for Approach C. This framing overstates T2's value. For the target audience, knowing *that* angular momentum drifts at rate 3.7×10⁻⁴/step and *where* (the thermostat coupling) is far more actionable than knowing the splitting 'architecturally cannot conserve angular momentum.'"

**Skeptic → Math Assessor:**
> "You rate Potential at 5/10, calling it 'strong practical contribution but weaker theoretical depth.' I agree — but this means the paper needs a *very* strong evaluation to compensate. The evaluation requires demonstrating localization on realistic-scale systems (not just 3-body problems). If the paper shows 'we recovered the shadow Hamiltonian for a 5-particle Lennard-Jones system,' reviewers will ask 'so what?'"

**Skeptic → Difficulty Assessor:**
> "Your 6/10 difficulty rating is appropriate only if the Convergence Theorem is waived. If the paper must prove convergence of shadow Hamiltonian recovery, the difficulty jumps to 8/10. Proving RIP for the *structured* measurement matrices arising from Hamiltonian trajectories is a genuinely hard open problem in applied mathematics."

**Math Assessor → Difficulty Assessor:**
> "The honest assessment: Approach C's value is almost entirely *engineering* — making SINDy work well for shadow Hamiltonian recovery, building a robust ablation framework. The math is window dressing that provides post-hoc theoretical justification for a tool that is fundamentally empirical."

### Consensus Verdict on C

**Genuinely strong:**
- Highest feasibility (7/10) and broadest coverage — the only approach that works on JAX, FFT-based spectral methods, GPU kernels, and opaque library calls.
- Lowest adoption barrier: `@conservation_audit` decorator requires zero code rewrite.
- Least LLM-vulnerable: dynamic execution and trajectory analysis are outside LLM capability (~30% overlap vs. ~70% for A).
- Most likely to produce a working artifact and a publishable paper (75% publication probability per the Skeptic).

**Genuinely weak:**
- Mathematical depth is consensus-thin (2/10). The Math Assessor's "there is no new math here" is uncontested. The Difficulty Assessor concurs: "4/10 difficulty."
- Scalability is the real ceiling: localization degrades beyond ~50 particles, and the "100% coverage" claim masks "~5% localization coverage" on realistic systems.
- No formal obstruction detection — the tool cannot prove a violation is architecturally unfixable.
- RIP for deterministic Hamiltonian trajectories is unproved and may fail precisely for integrable systems where conservation analysis matters most.

**What survives:** The ablation-based localizer as the killer feature (all assessors agree). The Skeptic's rescue simplification — "abandon full shadow Hamiltonian recovery, focus on conservation-law-specific violation detection + ablation" — eliminates the scalability bottleneck while preserving the core value. This produces a working tool in ~3 months.

---

## Cross-Approach Synthesis

### Surviving Strengths (elements all assessors agree are valuable)

1. **T2 obstruction detection (from A):** The ability to prove a conservation violation is *architecturally unfixable* is unique and genuinely valuable. All assessors flag it as A's crown jewel, though they disagree on whether the efficient reduction exists.

2. **Ablation-based localization (from C):** The causal intervention approach — toggle code regions, measure the effect on conservation — works on any modular code without extraction. All assessors recognize this as C's killer feature.

3. **The Lie algebra → graded effects connection (from B):** The only "genuinely surprising mathematical idea" across all three proposals. All assessors agree it is novel as a PL-theoretic construction, even as they debate whether it is deep or tautological.

4. **The bridge narrative (from A):** Connecting geometric numerical integration with program analysis is genuinely novel. No prior work operates in the code→math direction for conservation. This framing has value regardless of which approach is built.

5. **Low-friction adoption via decorators (from C):** `@conservation_audit` requires no code rewrite. All assessors agree this adoption model is the most realistic.

### Fatal Weaknesses (elements that cannot be saved)

1. **Tree-sitter-based code→math extraction on real frameworks (A):** The Skeptic's JAX-MD/Dedalus/broadcasting analysis is devastating and unrefuted. The extraction approach cannot parse the named target codebases.

2. **API adoption for Conservation Types (B):** All three assessors independently conclude no simulation developer will rewrite code for conservation type-checking. The Skeptic's estimate of ~0% voluntary adoption is uncontested.

3. **Full shadow Hamiltonian recovery at scale (C):** The dictionary explosion beyond ~50 particles is a hard physics/math constraint. The "100% coverage" claim is misleading without the "5% localization" caveat.

4. **T1 as a mathematical contribution (A):** Three-way consensus: the Math Assessor calls it "a data structure, not a theorem," the Difficulty Assessor calls it "bookkeeping," and the Skeptic considers it downstream of a fatal extraction dependency.

### The Key Insight from the Debate

**The adversarial process revealed a fundamental tradeoff that was invisible in the proposals alone: the approaches that have the deepest math cannot reach real code, and the approach that reaches real code has no deep math.**

Approach A has the richest theoretical framework (T2, differential symbolic slicing, provenance-tagged BCH) but cannot parse its named target codebases — its math operates in a vacuum. Approach B has the most elegant formalism (Lie↔effects isomorphism, soundness guarantee) but requires an adoption model that will never materialize — its types annotate an empty set. Approach C works on everything but contributes "~5% new math" — it is engineering dressed in theorem-statement formatting.

No single proposal acknowledged this tradeoff honestly. The Visionary presented A's extraction limitation as a "phasing" issue, B's adoption barrier as a "partnership opportunity," and C's scalability ceiling as a "fallback." The debate exposed each of these as critical structural problems, not implementation details.

The debate also revealed that **all three approaches solve a problem that ~500 people have** (the Skeptic's demand vacuum analysis). The project's value is as a research contribution, not a product. The paper's audience is PL researchers and geometric integration researchers — the tool is the evaluation section of a theory paper.

A final, uncomfortable consensus emerged: **a SymPy script + 100 lines of Python covers ~60% of the capabilities for single-method integrators** (the Skeptic's baseline test). The unique value of ConservationLint emerges only for heterogeneous multi-method compositions with obstruction detection — a use case that is genuinely valuable, genuinely hard, and genuinely rare.

### Recommended Strategy

The debate converges on a clear recommendation from all three assessors:

**1. Build a simplified Approach C as the practical foundation.** Strip the full shadow Hamiltonian recovery; focus on conservation-law-specific violation detection (`d/dt C_v` along trajectories) plus ablation-based localization. This eliminates the scalability bottleneck, removes the need for the Convergence Theorem, preserves the killer feature (ablation localization), and produces a usable tool in ~3 months. Target ICSE or SC.

**2. Prove T2 (obstruction criterion from A) as a standalone theorem.** If the efficient reduction from semi-algebraic feasibility to structured Lie algebra computation works, this is publishable at CAV or *Foundations of Computational Mathematics* independent of any tool. If it doesn't, the math reduces to brute-force computation on small cases (k ≤ 5, p ≤ 4), which is still useful engineering.

**3. Pursue B's Connection Theorem as a separate theory paper.** If the Lie↔effects isomorphism is discovered (not stipulated) and yields surprises, submit to POPL or LICS as a self-contained 12-page contribution. Do not build software; do not require users.

**4. If a unified tool must exist: C first, then A-lite.** Shadow Oracle provides dynamic analysis on 100% of runnable code. In parallel, develop a semi-dynamic version of Noether's Scalpel — intercepting at the framework level (JAX's jaxpr, NumPy operation tracing) rather than Tree-sitter parsing — to provide formal guarantees on the subset of code amenable to static analysis. The combined tool offers graceful degradation: formal proofs where achievable, statistical guarantees everywhere else.

**5. Do not build the typed API (B as a tool).** The adoption barrier is fatal for any venue requiring an artifact evaluation. The math is the contribution; the tool is not.

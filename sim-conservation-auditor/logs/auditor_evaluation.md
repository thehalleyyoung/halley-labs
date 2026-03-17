# Independent Auditor Evaluation: ConservationLint

**Evaluator role:** Independent auditor, best-paper verification committee  
**Document under review:** `ideation/crystallized_problem.md`  
**Date:** 2025-07-17  
**Verdict:** Mixed — strong conceptual kernel, significant feasibility risks, inflated scope estimates

---

## 1. EXTREME AND OBVIOUS VALUE — Score: 5/10

### Is the pain real?

**Partially.** The Wan et al. (2019, JAMES) citation—a conservation-violating coupling scheme persisting for three years in a major climate model—is compelling and specific. Conservation violations *are* real bugs that produce plausible-looking but wrong results. This is the strongest part of the value argument.

However, the document overstates the universality of the pain:

- **The TAM is narrow.** The stated users are: climate modelers, fusion simulation teams at ITER/national labs, MD code maintainers (LAMMPS/OpenMM/GROMACS), numerical methods researchers, and DOE V&V teams. This is perhaps **2,000–10,000 developers worldwide** who actively write and maintain simulation kernels where conservation properties matter at the code level. Many of these people already have domain expertise to catch these bugs. This is not a "million developers" problem; it's a niche expert-tool market.

- **Conservation violation is NOT the #1 pain point** for simulation developers. The actual hierarchy of pain, based on surveys (e.g., Carver et al., "Software Development Practices in Computational Science," SE-CSE workshops) and practitioner reports, is roughly: (1) performance/scalability, (2) correctness bugs in general (indexing, boundary conditions, parallelization races), (3) portability across architectures, (4) reproducibility, (5) numerical stability, then (6) conservation-specific violations. ConservationLint targets a real but not top-priority pain point.

- **The "silent killer" framing is selective.** Yes, conservation violations are insidious. But so are phase errors, dispersion errors, stability violations, and order-of-accuracy degradation—none of which ConservationLint addresses. The document frames conservation as uniquely dangerous, but a simulation that conserves energy perfectly while having O(Δt) phase errors in wave propagation is equally untrustworthy.

### The LLM question

**This is a serious threat the document ignores entirely.** A domain expert can already prompt GPT-4/Claude with "here is my leapfrog integrator code, does it conserve angular momentum?" and get a useful analysis. The LLM won't provide formal BCH expansions, but for the *practical diagnostic purpose* (finding which line breaks conservation), an LLM with simulation-domain knowledge gets you 60–70% of ConservationLint's stated value at 0% engineering cost. The document needs to explicitly address why a 162K LoC Rust tool is worth building when the LLM baseline keeps improving. The honest answer is probably "exact guarantees vs. probabilistic hints"—but that honest answer also limits the TAM to people who need formal guarantees, which is an even smaller subset.

### Verdict

The pain is real but affects a small community, is not their top priority, and faces an increasingly capable LLM-based alternative for the diagnostic (non-formal) use case. The CI/CD integration story is appealing but assumes the tool actually works on real code—a claim that remains unvalidated (see §5).

---

## 2. GENUINE DIFFICULTY AS A SOFTWARE ARTIFACT — Score: 6/10

### Subsystem-by-subsystem audit of the 162K LoC claim

| # | Subsystem | Claimed LoC | Honest Assessment | Adjusted LoC | Novel? |
|---|-----------|-------------|-------------------|-------------|--------|
| 1 | Python frontend + IR | 25K | Tree-sitter parsing is library wrapping (~5K custom). IR design is real work. Pattern lifting (recognizing numerical idioms) is the hard part (~8K). Rest is boilerplate. | ~15K | Partially — pattern lifting is novel, parsing is not |
| 2 | Symbolic algebra engine | 20K | **This is the most inflated claim.** SymPy exists. The justification for writing a custom Rust engine is performance, but the operations are polynomial manipulation + Lie brackets. A focused polynomial algebra library with Lie bracket support is ~5–8K LoC in Rust, not 20K. The extra 12K is either premature generalization or padding. | ~8K | Low — this is SymPy-in-Rust for a restricted domain |
| 3 | Variational symmetry analyzer | 18K | **The document itself admits** this reduces to "structured linear algebra" with a "restricted ansatz." A restricted Lie-symmetry solver targeting translation, rotation, scaling, and Galilean generators on polynomial systems is a sophisticated linear algebra problem, but 18K is generous. The core algorithm is sparse linear system construction + solve. | ~8K | Medium — the restricted ansatz is a good idea, but the engineering is structured LA |
| 4 | Backward error engine | 18K | BCH expansion at fixed order with provenance tagging. This is the most genuinely novel subsystem. The combinatorics of heterogeneous composition tracking are real. 18K is plausible if it includes comprehensive term management. | ~15K | **Yes** — this is where T1 lives |
| 5 | Runtime monitor | 12K | Sequential hypothesis testing (CUSUM/SPRT) is textbook. Fused instrumentation is clever but standard in profiling tools. Python instrumentation adds boilerplate. | ~8K | Low — standard monitoring with a domain twist |
| 6 | Causal localization | 15K | "Differential symbolic slicing" sounds novel but is essentially backward propagation of symbolic terms through a DAG. The core algorithm is small; most LoC would be handling edge cases in the code→math mapping. | ~10K | Medium — the concept is novel, implementation is graph traversal |
| 7 | Repair engine | 12K | Template catalog + constraint checking. The "constrained program synthesis" framing is oversold. This is pattern-matching against known repair templates (symmetrize the splitting, use a corrector step, etc.) with validation. | ~6K | Low — template matching, not synthesis |
| 8 | Benchmark suite | 20K | **Mostly test fixtures.** 20+ simulation kernels are ~200–500 lines each (4–10K total code), plus annotations, expected outputs, and harness glue. The intellectual work is choosing the right benchmarks, not writing 20K lines. | ~10K (real complexity ~4K) | No — this is test infrastructure |
| 9 | Evaluation harness | 10K | Metrics computation, comparison scripts, report generation. Standard evaluation infrastructure. | ~6K | No |
| 10 | CLI/LSP/reporting | 12K | LSP server is real work (~5K). CLI and reporting are standard. | ~8K | No — standard tooling |
| | **Totals** | **162K** | | **~94K** | |

**Honest estimate: ~90–100K LoC of real code, of which ~30–35K is genuinely novel engineering.** The 162K figure is inflated by roughly 60%, likely by counting generous margins, boilerplate, and test fixtures as complexity.

### Is the symbolic algebra engine (20K LoC) justified when SymPy exists?

**Partially, but not at 20K.** The performance argument is valid: SymPy is notoriously slow for large polynomial manipulations, and calling it from Rust via Python FFI defeats the purpose. A focused Rust-native polynomial ring library with Lie bracket support is justified. But 20K LoC for this is ~2.5× what's needed. For comparison, the `polynomial` crate in Rust is ~3K LoC and handles multivariate polynomials. Adding Lie bracket computation and rational functions might double that. 8–10K is realistic.

### Is the "restricted Lie-symmetry analysis" genuinely hard?

**No, and the document essentially admits this.** Line 155: "This reduces the overdetermined PDE system to a structured linear algebra problem of size proportional to (number of variables) × (ansatz dimension)... solvable in milliseconds." A problem solvable in milliseconds on a laptop is not a "genuine engineering breakthrough." It's a well-executed reduction of a hard general problem to an easy restricted problem. The *idea* of the restriction is clever; the *implementation* is not 18K LoC of hard engineering.

### Verdict

The project has real engineering complexity, centered on the backward error engine, causal localization, and the code→math extraction pipeline. But the 162K LoC estimate is inflated ~60%, and several subsystems (symbolic algebra, symmetry analyzer, repair engine) overstate their novelty. Honest difficulty is "hard senior engineer project," not "requires engineering breakthroughs."

---

## 3. BEST-PAPER POTENTIAL — Score: 5/10

### T1: Provenance-tagged BCH (Heterogeneous Composition Modified Equation Theorem)

**What's genuinely new:** The provenance tagging structure on the BCH expansion. This is a real contribution—knowing *which sub-integrator generates which term* in the modified Hamiltonian is useful and not present in existing literature (Blanes, Casas & Murua 2008 give the expansion but not the attribution).

**What's not new:** The BCH expansion itself, the modified equation framework, the fact that splitting methods have modified Hamiltonians. These are all in Hairer, Lubich & Wanner (2006).

**Honest assessment:** T1 is **bookkeeping on known expansions**. The mathematical content is: "label each term in a known expansion with its origin." This is useful engineering (it enables localization) but is not deep mathematics. The "explicit leading-order symmetry-breaking bounds for mixed-order compositions" are the more interesting part, but the document doesn't state whether these bounds are tight or merely order-of-magnitude.

**Novelty score: 4/10.** Useful but incremental.

### T2: Computable Obstruction Criterion

**What's genuinely new:** The reduction of "can any integrator of a given order with this splitting preserve this symmetry?" to a finite algebraic test.

**Critical concern: "computable" hides truncation.** The criterion works "at a given order"—meaning it answers "can conservation be achieved to order p?" not "can conservation be achieved exactly?" For a finite truncation, the obstruction test is a finite linear algebra computation on the BCH expansion coefficients. This is computable but **incomplete**: it's possible that conservation is obstructed at order 4 but achievable at order 6 via cancellation. The document does not discuss this incompleteness.

**Is it actually undecidable in the general case?** The exact question ("does there exist ANY method for this splitting that preserves this quantity to ALL orders?") is likely undecidable or at least requires analyzing the full infinite series. The finite-order version is interesting but must be clearly framed as an approximation, not a complete decision procedure.

**Novelty score: 6/10.** The best of the three theorems, but needs honest framing of the truncation limitation.

### T3: Liftable Fragment Formalization

**This is a definition dressed as a theorem.** The "statement" is: define the liftable fragment as programs with (no recursion, no data-dependent control flow, affine array accesses, polynomial-rational dependence). Claim: this fragment admits decidable membership and faithful extraction.

**The "40–60% coverage" claim is the actual content**, and it's unverified. The document says "covering ~40–60% of kernel lines in typical simulation codes such as Dedalus, JAX-MD, and SciPy-based solvers." But:
- No measurement is presented.
- "Kernel lines" vs. "total lines" is a crucial distinction—most code in a simulation is I/O, setup, visualization, not numerical kernels.
- Dedalus and JAX-MD are unusually clean, modern, Python-native codes. They are not representative of the "hundreds of thousands of lines" production codes mentioned in the motivation (which are mostly Fortran/C++).

**Novelty score: 3/10.** Formalizing the boundary of your tool's applicability is good engineering practice but not a theorem. PLDI papers sometimes include such characterizations, but they are never the main contribution.

### Venue assessment

- **PLDI:** Possible but a stretch. PLDI wants formal semantics contributions; T1–T3 are applied math, not PL theory. The program analysis novelty (differential symbolic slicing) is the closest fit.
- **OOPSLA:** More realistic. OOPSLA accepts "domain-specific tools with evaluation" papers. But the evaluation would need to be on real-world codes, not just a curated benchmark suite.
- **ICSE:** Unlikely. Too mathematical for ICSE reviewers.
- **SC (Supercomputing):** Actually the most natural venue, but SC doesn't have the "best paper" prestige the document aspires to.
- **NeurIPS/ICML:** No. This is not ML.
- **Realistic landing:** A solid OOPSLA or CAV paper, not a best-paper winner.

### Verdict

The theorems range from "definition-as-theorem" (T3) to "bookkeeping on known results" (T1) to "genuinely interesting but incompletely framed" (T2). This is a solid systems/tools paper, not a theoretical breakthrough. Best-paper at a top venue is aspirational; acceptance at OOPSLA or CAV is realistic.

---

## 4. LAPTOP CPU + NO HUMANS — Score: 7/10

### "Structured linear algebra" for Lie symmetry — true for target systems?

**True, but only because the ansatz is restricted.** The document targets translation, rotation, scaling, and Galilean symmetry generators—the symmetries that give rise to conservation of momentum, angular momentum, energy, and center-of-mass. For these specific symmetries on polynomial-rational systems, the determining equations *do* reduce to structured linear algebra. This is well-known in the Lie symmetry literature (e.g., Hereman's `InvariantsSymmetries` package uses similar restrictions).

**For symmetries outside the ansatz (e.g., hidden symmetries, conformal symmetries, gauge symmetries), this reduction fails.** The document does not claim to handle these, which is honest. But this means ConservationLint cannot discover *unexpected* conservation laws—only verify *expected* ones. This is a significant limitation the document underplays.

### BCH at order 4–6, k ≤ 10: tractability

**Let's verify.** At order p in the BCH expansion with k operators:
- Order 4: The number of distinct Lie bracket monomials of degree 4 in k generators is O(k^4). With k=10, that's ~10,000 terms. Each term is a polynomial operation. **Tractable in seconds.**
- Order 6: O(k^6) = O(10^6) = 1,000,000 bracket evaluations. Each on polynomials in ≤50 variables. If each bracket evaluation takes 1ms (generous for polynomial manipulation), that's ~1,000 seconds ≈ 17 minutes. **This exceeds the 10-minute budget at the high end.**

**The O(k^p) claim is technically correct but the constant matters.** For k=10, p=6, the computation is borderline tractable. For k=5, p=4 (more typical), it's easily feasible. The document should specify that order 6 analysis is only available for small k, or the 10-minute budget is violated.

### Noether's Razor CPU-only baseline

**This is a problem.** Noether's Razor (Cranmer et al., NeurIPS 2024) is a neural-network-based method that requires training on trajectory data. Running it CPU-only is technically possible but would be extremely slow (hours per kernel, not minutes). The document acknowledges this parenthetically: "Run CPU-only (if feasible) or with restricted GPU budget."

**If Noether's Razor cannot run CPU-only in a comparable time budget, the comparison is invalid** as a like-for-like baseline. The document should either commit to a GPU budget for this baseline, drop it as a baseline, or use a simpler trajectory-based method (e.g., SINDy-based conservation discovery) as a substitute.

### 10-minute budget for complex codes

**Plausible for the benchmark suite; questionable for real codes.** The benchmark kernels are 200–500 lines of clean Python. The "10K LoC kernel" target is the real test. For a 10K LoC kernel with:
- Deep call graphs (e.g., force computation calls 20 helper functions)
- Library calls to NumPy/SciPy that must be resolved via cached signatures
- Multiple levels of abstraction (classes, closures, generators)

The bottleneck is IR extraction, which requires resolving all these abstractions. The document acknowledges this: "The bottleneck is IR extraction for complex kernels with deep call graphs." **A 10-minute budget is optimistic for 10K LoC with deep call graphs.** 30 minutes is more realistic.

### Verdict

The core symbolic computations (symmetry analysis, BCH expansion) are genuinely laptop-tractable for the restricted problem. The feasibility risks are in IR extraction time for complex codes and in the BCH computation at high order with many splittings. The Noether's Razor baseline is likely invalid CPU-only.

---

## 5. FATAL FLAWS

### Flaw 1 (CRITICAL): Code → Math extraction is the load-bearing assumption, and it's the weakest link

The entire pipeline depends on "lifting" imperative code into a mathematical IR. The document describes this as a "hard subproblem" but then claims it in 25K LoC. **This is where the project lives or dies.**

Evidence for concern:
- **Real simulation codes don't look like the benchmark suite.** The benchmarks target "Dedalus, JAX-MD, and SciPy-based solvers"—modern, clean, Python-native frameworks. Production codes at national labs (LAMMPS, GROMACS, WRF, CESM, FLASH) are written in Fortran/C++ with Python wrappers, use MPI, have complex memory management, and rely on hardware-specific optimizations. ConservationLint cannot analyze these.
- **Even within Python, real numerical code is opaque.** A call to `scipy.integrate.solve_ivp` or `jax.experimental.ode.odeint` hides the integrator behind a library API. The document mentions "caching of library-function signatures" but this requires manually encoding the mathematical semantics of every library function—a maintenance burden that scales with the ecosystem.
- **The liftable fragment excludes critical patterns.** Data-dependent control flow over state variables (e.g., `if particle_distance < cutoff:`) is excluded. But cutoff-based force truncation is ubiquitous in MD codes and is itself a source of conservation violations. **The tool cannot analyze the very pattern that causes the most conservation bugs in practice.**

### Flaw 2 (SERIOUS): The 40–60% coverage claim is unverified and likely aspirational

The document states "~40–60% of kernel lines" in Dedalus, JAX-MD, and SciPy-based solvers fall in the liftable fragment. This is:
- **Not measured.** No empirical data is presented.
- **Suspiciously precise for an estimate.** A range of 40–60% suggests some analysis was done, but none is reported.
- **Likely inflated.** "Kernel lines" is undefined. If it means "lines inside time-stepping loops excluding I/O and setup," then 40–60% might be achievable for JAX-MD (which is unusually functional/compositional). For Dedalus, which uses spectral methods with complex transform logic, the polynomial-rational restriction is binding—spectral methods involve trigonometric/exponential functions, not polynomials.

### Flaw 3 (SERIOUS): Causal localization through abstraction layers is fragile

The claim that ConservationLint can trace a symmetry-breaking term "back to specific source lines" assumes that the code→math lifting preserves a clean correspondence between math terms and code lines. In practice:
- **Optimized code obscures this correspondence.** Loop fusion, vectorization, and temporary elimination (common in NumPy-heavy code) mean that one source line contributes to many math terms and vice versa.
- **Library boundaries break the trace.** If the symmetry-breaking term comes from inside `np.linalg.solve` or `scipy.fft.fft`, the localization points to the library call—which is useless. The developer needs to know *why* they're calling it wrong, not *that* they called it.

### Flaw 4 (MODERATE): The repair engine is undersold as "synthesis" but is actually template matching

The document calls repair suggestion "constrained program synthesis" but then describes it as "template catalog, constraint-based selection, validation." Template matching against known repair patterns (symmetrize splits, add corrector steps, use Yoshida composition) is useful but is not synthesis in the PL sense. The 60% repair success rate target on a curated benchmark suite with known repair templates is achievable but does not generalize to novel violations.

### Flaw 5 (MODERATE): The comparison to SLAM/Herbie is aspirational, not structural

SLAM succeeded because device drivers have a small, well-defined API surface and clear safety properties. Herbie succeeded because floating-point expressions are syntactically simple and accuracy has a scalar metric. ConservationLint's domain—arbitrary simulation codes with heterogeneous integrators—has neither a small API surface nor a scalar correctness metric. The structural analogy breaks down.

### Flaw 6 (MINOR): No discussion of false positive cost

The document targets ≤10% false positive rate but does not discuss the **cost** of false positives in this domain. A false conservation warning on a pull request to a production simulation code could block critical work and erode trust in the tool. In the AddressSanitizer analogy, false positives are rare but tolerable because the tool catches memory corruption. For conservation warnings, false positives would mean the tool says "your code breaks energy conservation" when it doesn't—a much more damaging false alarm.

---

## Summary Scores

| Axis | Score | Key Justification |
|------|-------|-------------------|
| **1. Extreme and Obvious Value** | **5/10** | Real pain, tiny TAM (~5K developers), not the #1 priority for those developers, LLM competition unaddressed |
| **2. Genuine Difficulty** | **6/10** | ~100K honest LoC (not 162K), ~30K genuinely novel. Symbolic algebra engine inflated. Symmetry analysis admitted to be "milliseconds." |
| **3. Best-Paper Potential** | **5/10** | T1 is bookkeeping, T2 is the real contribution but incompletely framed, T3 is a definition. Realistic venue: OOPSLA accept, not best paper. |
| **4. Laptop CPU Feasibility** | **7/10** | Core computation is tractable. BCH order 6 with k=10 is borderline. Noether's Razor baseline likely invalid. IR extraction is the real bottleneck. |
| **5. Fatal Flaws** | **—** | Code→math extraction on real codes is the existential risk. 40–60% coverage is unverified. Cutoff-based force truncation (a major conservation-bug source) is explicitly excluded from the liftable fragment. |

## Overall Assessment: **5.5/10**

**The conceptual kernel is strong:** bridging Noether's theorem and program analysis is a genuinely novel and elegant idea. The "physics-aware program analysis" paradigm is worth exploring.

**The execution plan is overscoped and undervalidated.** The 162K LoC estimate is inflated. The theorems range from incremental to definitional. The code→math extraction—the entire project's load-bearing assumption—is handwaved with "Tree-sitter + pattern lifting" when it is actually the hardest unsolved problem in the pipeline. The benchmark suite conveniently targets clean, modern Python codes that look nothing like the production Fortran/C++ codes described in the motivation.

**Recommendation:** Reduce scope to the core pipeline (extraction + symmetry analysis + BCH localization) on a single well-chosen domain (e.g., Hamiltonian splitting methods in Python). Prove the extraction works on 5 real codes before claiming 162K LoC of infrastructure. T2 (obstruction criterion) is the best theorem—lead with it. Drop the "best paper" aspiration and target a solid OOPSLA tools paper.

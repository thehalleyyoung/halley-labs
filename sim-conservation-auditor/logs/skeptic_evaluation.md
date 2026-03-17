# ConservationLint — Fail-Fast Skeptic Evaluation

**Evaluator role:** Fail-Fast Skeptic, Best-Paper Verification Committee
**Date:** 2026-03-08
**Document under review:** `ideation/crystallized_problem.md`
**Project phase:** Crystallize (no code, no proofs, no prototype — only a problem statement exists)

**Executive verdict: CONDITIONAL CONTINUE — with severe binding conditions.**

The idea is genuinely interesting. The execution plan is wildly overscoped, the claimed novelty is inflated in multiple places, and several "theorems" are not theorems. Below is the full damage report.

---

## 1. CLAIM-BY-CLAIM CHALLENGE

### Claim 1: "Conservation violations are the silent killers of computational science"

**Verdict: PARTIALLY SUPPORTED — but overstated.**

The Wan et al. (2019, JAMES) citation is real and compelling: a conservation-violating coupler persisted for three years in a major climate model. But one anecdote is not an epidemic. Counter-evidence:

- **Major simulation codes already have conservation monitors built in.** GROMACS has had energy conservation monitoring and drift reporting since at least version 4.x (early 2000s). LAMMPS has `thermo_style` with energy/momentum tracking as a first-class feature. OpenMM checks energy conservation as part of its validation suite. These aren't "expert manual audit" — they're standard engineering practice.
- **The V&V community exists.** DOE labs (Sandia, LLNL, LANL) have formalized Verification & Validation procedures (Oberkampf & Roy, 2010) that include conservation checks as standard protocol. These are not ad-hoc; they're systematic.
- **Most conservation violations are caught by integration tests.** A well-instrumented simulation that monitors the Hamiltonian over a long trajectory will catch energy drift. The claim that "standard testing" misses these violations is true only if the tests are poorly designed — which is a human problem, not a tool gap.

**Strongest counter-argument:** The real problem isn't lack of tools — it's lack of discipline. Adding another tool doesn't fix a culture where teams don't write conservation regression tests. A simpler intervention (a conservation-test template library for pytest) might capture 80% of the value at 1% of the cost.

**What would falsify it:** Survey 20 production simulation teams. If >80% already have automated conservation checks in CI, the "desperate need" claim collapses.

### Claim 2: "No existing tool"

**Verdict: SIGNIFICANTLY OVERSTATED.**

The problem statement acknowledges Daikon, Noether's Razor, and DIG but dismisses them too quickly:

- **Daikon + physics knowledge is closer than claimed.** Daikon discovers polynomial invariants from traces. If you seed it with templates for `sum(p_i)`, `sum(r_i × p_i)`, `H(q,p)`, you get a conservation monitor. It's not automatic Noether's theorem, but for *practical* conservation checking it may suffice. The problem statement never benchmarks against this simple baseline, which is suspicious.
- **Domain-specific testing frameworks already exist.** GROMACS has `gmx energy` and `gmx check`; LAMMPS has built-in conservation diagnostics; ESMValTool for climate models checks energy balance. These are not program-analysis tools, but they catch conservation violations in practice, which undercuts the urgency argument.
- **Property-based testing (Hypothesis/QuickCheck) with physics oracles.** You can write `@given(initial_conditions)` and assert `|H(final) - H(initial)| < tol`. This is crude but effective, and it's available today at zero research cost.

**What the problem statement gets right:** None of these existing approaches do *causal localization* (tracing a violation to a specific source line) or *obstruction detection* (proving a violation is unfixable within a given algorithm). These are genuinely novel capabilities — IF they can be built.

### Claim 3: "First bridge between Noether's theorem and program analysis"

**Verdict: LIKELY TRUE BUT NEEDS STRONGER PRIOR-ART SEARCH.**

I could not find a direct counterexample of a tool that applies Noether's theorem to imperative code. However:

- **The geometric numerical integration community has been doing this manually for decades.** Hairer, Lubich & Wanner's textbook (2006) is essentially a manual version of ConservationLint's core pipeline. The contribution here is *automation*, not the mathematical connection itself.
- **Noether's Razor (Cranmer et al., NeurIPS 2024)** already bridges Noether's theorem and computation — just from the data side rather than the code side. Calling ConservationLint "the first bridge" ignores this.
- **Work on symmetry detection in computational mechanics** (e.g., Bridges & Reich, 2006; Leimkuhler & Reich, 2004) explicitly connects Noether's theorem to numerical methods. The step from "numerical methods" to "the code implementing numerical methods" is genuinely novel, but it's smaller than the problem statement implies.

**What would falsify it:** Finding a published paper that applies symbolic Lie-symmetry analysis to source code (not equations). I haven't found one, but the search space (computational mechanics + program analysis intersection) is niche enough that something could be hiding in a workshop paper or thesis.

### Claim 4: "40-60% coverage of kernel lines"

**Verdict: COMPLETELY UNSUPPORTED — this number is fabricated.**

The problem statement says the liftable fragment covers "~40-60% of kernel lines in typical simulation codes such as Dedalus, JAX-MD, and SciPy-based solvers." But:

- **No measurement has been performed.** The theory/ directory is empty. The implementation/ directory is empty. There is no code, no analysis, no prototype. This number is a guess dressed up as an estimate.
- **The liftable fragment hasn't been formally defined yet.** T3 is stated as a theorem to be proven. You cannot claim coverage of a fragment you haven't defined.
- **The codes cited (Dedalus, JAX-MD) use opaque library calls extensively.** Dedalus is built on top of FFTW, MPI, and dense linear algebra — all opaque to Tree-sitter parsing. JAX-MD uses JAX transformations (jit, vmap, grad) that are fundamentally non-trivial to "lift" — they define computations via tracing, not syntactic structure. Claiming 40-60% coverage of these codebases without having attempted extraction is irresponsible.

**Honest estimate:** For toy simulation codes written in pure NumPy with no library calls, 40-60% might be achievable. For real production codes, 10-20% is more realistic, and even that would be an achievement.

### Claim 5: "≥90% detection rate"

**Verdict: CIRCULAR UNTIL PROVEN OTHERWISE.**

- The benchmark suite is self-constructed ("20+ kernels" to be written by the same team building the tool).
- On a self-constructed benchmark, ≥90% detection is trivially achievable by tuning the tool to the benchmarks. This is the textbook definition of overfitting to your own evaluation.
- A meaningful detection rate would be measured on **held-out** code from real projects (LAMMPS pull requests, CESM bug reports, GROMACS regression logs) where the authors did not choose the examples.

**What would make this credible:** A blind evaluation on conservation bugs sourced from real simulation code repositories, ideally compiled by an independent third party.

### Claim 6: "≤10 minutes on laptop CPU" for a 10K LoC kernel

**Verdict: PLAUSIBLE FOR THE HAPPY PATH — but worst-case is unknown.**

The analysis is reasonable *if* the kernel is in the liftable fragment:
- Tree-sitter parsing: milliseconds. Fine.
- Restricted Lie-symmetry analysis: they claim linear algebra, not Gröbner bases. For a fixed ansatz (translations, rotations, scaling), this is credible.
- BCH expansion truncated at order 4: combinatorial explosion is controlled. O(k^p) with k≤10, p≤6 = 10^6 bracket evaluations, each on small polynomial expressions. Plausible.

But:
- **The bottleneck is IR extraction, which is handwaved.** "Caching of library-function signatures amortizes this" — what library-function signatures? These haven't been written. The effort to model NumPy/SciPy/JAX semantics is enormous (thousands of functions, many with complex broadcasting/shape semantics).
- **Worst-case for non-trivial codebases is unbounded.** Deep call graphs, polymorphic library calls, and indirect array access all blow up extraction time. The problem statement doesn't discuss worst-case at all.

---

## 2. LLM OBSOLESCENCE CHECK

### Can LLMs detect conservation violations RIGHT NOW?

**Partially, yes.** I assessed this critically:

- **Simple cases:** If you paste a textbook leapfrog integrator into GPT-4/Claude and ask "does this conserve angular momentum?", you will get a *correct, detailed* answer explaining that leapfrog is symplectic and thus conserves a modified Hamiltonian but not angular momentum exactly, and identifying the O(Δt²) error terms. LLMs are already quite good at this for textbook examples.
- **Medium cases:** For a Verlet integrator composed with a thermostat, an LLM can identify that the thermostat breaks energy conservation by design but may miss that the specific coupling order introduces an O(Δt) momentum violation.
- **Hard cases:** For a 5000-line simulation code with multiple files, library calls, and non-obvious splitting structure, LLMs cannot currently trace conservation violations to specific lines with mathematical rigor. Context window limitations and lack of formal symbolic reasoning prevent this.

### Does ConservationLint survive the LLM-10x assumption?

**This is the most dangerous threat.** In 2-3 years:
- LLMs with 1M+ token context windows can ingest an entire simulation codebase.
- LLMs with tool-use (calling SymPy, running simulations) could approximate the ConservationLint pipeline.
- A fine-tuned LLM on the geometric numerical integration literature could reproduce much of the domain expertise.

**ConservationLint's defensible moat** (if it works):
1. **Formal guarantees.** An LLM gives probabilistic answers; ConservationLint (in principle) gives mathematical proofs. For V&V at national labs, formal guarantees matter.
2. **Reproducibility.** Same input → same output, deterministically. LLMs are stochastic.
3. **Obstruction proofs.** T2 (proving a violation is architecturally unfixable) is genuinely beyond what an LLM can do — it requires a formal proof, not a heuristic judgment.

**But:** If the formal guarantees only apply to 10-20% of real codebases (the actual liftable fragment), the remaining 80-90% will be handled by LLMs anyway. ConservationLint risks being a precision tool that covers too little to be the primary solution.

**Verdict: ConservationLint must clearly articulate its value as a *formal verification* tool for the analyzable fragment, with LLMs as the fallback for everything else. If it's positioned as a comprehensive solution, LLMs will eat its lunch.**

---

## 3. MATHEMATICAL SKEPTICISM

### T1: Heterogeneous Composition Modified Equation "Theorem"

**Verdict: This is bookkeeping on known mathematics, not a new theorem.**

The BCH expansion for operator splitting is well-established (Blanes, Casas & Murua 2008, 2010). The "tagged provenance" is simply labeling which terms in the BCH expansion come from which operators — this is implicit in the expansion itself. Anyone who writes out the BCH formula for `exp(A)exp(B)exp(C)` can read off which terms involve which operators.

- **What would make it a real theorem:** Proving non-trivial structural properties of the tagged terms. For example: "The leading symmetry-breaking term in any k-way splitting of a rotationally symmetric Hamiltonian always originates from at most 2 of the k sub-integrators." *That* would be a theorem. Simply tagging terms in a known expansion is a notation.
- **Charitable interpretation:** The novelty may be in the rigorous treatment of *mixed-order* compositions (combining methods of different orders), which existing BCH results don't fully cover. If the proof handles this carefully, there may be a genuine technical contribution — but it needs to be stated much more precisely.

**Publishable standalone?** Not as currently stated. As a lemma in a systems paper, fine.

### T2: Computable Obstruction Criterion

**Verdict: Potentially the most interesting result — but possibly trivial.**

The claim is: given a splitting and symmetry generator, you can computably determine whether *any* integrator of a given order on that splitting can preserve the conservation law.

- **If the order is fixed:** The BCH expansion is a finite polynomial in the sub-Hamiltonians. The symmetry condition (Lie derivative vanishes) becomes a system of polynomial equations. Checking solvability of a finite polynomial system is decidable (Tarski-Seidenberg). So in one sense, this is "trivially" computable. The question is whether the reduction to a *practical* computation (not doubly-exponential quantifier elimination) is non-trivial.
- **The interesting case:** If the criterion reduces to checking a small number of Lie bracket conditions (as the problem statement suggests: "vanishing of a finite set of Lie bracket expressions"), this is genuinely useful and elegant. But the problem statement doesn't explain *why* only finitely many brackets need to be checked, which is where the actual mathematical content would live.
- **What would make it non-trivial:** Proving that the number of bracket conditions is polynomial in k (the number of sub-integrators) and p (the order), not exponential. An exponential criterion that's "computable in principle" but useless in practice would be a hollow result.

**Publishable standalone?** Possibly, if the reduction is efficient and the proof reveals structural insight about Hamiltonian splitting. As stated, it's unclear.

### T3: Liftable Fragment Formalization

**Verdict: This is exactly "programs we can handle" dressed up as a formal result.**

Every static analysis paper defines the fragment of programs it handles. Calling it a "theorem" doesn't elevate it. The specific critique:

- **The syntactic characterization is standard.** "No recursion, no data-dependent control flow, affine array indices" — this is the polyhedral model from the compiler literature (Feautrier 1992, Bondhugula et al. 2008). The novelty would have to be in the *semantic faithfulness* guarantee (that the extracted IR preserves conservation structure), not in the syntactic characterization.
- **The coverage claim (40-60%) is unverified** (see Claim 4 above). Without measurement, T3 is a definition, not a result.
- **"Decidable membership test"** — for the syntactic conditions listed, this is trivially decidable (just check the AST). Calling this out as a feature is padding.

**Publishable standalone?** No. It's a well-stated definition section, not a theorem.

### Summary: Are T1-T3 publishable?

| Theorem | Standalone publishable? | As enabling machinery? |
|---------|------------------------|----------------------|
| T1 | No (bookkeeping on BCH) | Yes, if the mixed-order case is handled rigorously |
| T2 | Maybe (depends on efficiency of criterion) | Yes, this is the strongest mathematical claim |
| T3 | No (definition, not theorem) | Yes, as an honest scope statement |

**Net assessment:** The problem statement presents these as "three new theorems." At most one (T2) might be a genuine theorem. The other two are a notation (T1) and a definition (T3). This inflation of novelty is a red flag for a best-paper claim.

---

## 4. ENGINEERING FEASIBILITY ATTACKS

### 4.1 Code→Math Extraction: The Actual Hard Part

The problem statement treats this as one subsystem among many (~25K LoC). In reality, this is where the project lives or dies.

**Fatal difficulties:**

- **NumPy broadcasting.** `a + b` where `a` is (3, N) and `b` is (N,)` has semantics that depend on runtime shapes. You cannot lift this to a mathematical expression without shape information, which requires either concrete shapes (losing generality) or symbolic shape analysis (a research problem in itself).
- **SciPy sparse matrix operations.** `scipy.sparse.linalg.spsolve(A, b)` is an opaque library call. To understand its conservation properties, you need to know the mathematical structure of A — which requires lifting the code that constructs A, which may involve sparse assembly from element contributions, which is an entirely different pattern-matching problem.
- **JAX transformations.** `jax.jit(jax.vmap(force_fn))` — JAX works by tracing Python code to build a computation graph (jaxpr). Tree-sitter parses the *Python syntax*, not the *traced computation*. To lift JAX code, you need to either (a) intercept the jaxpr (requires running JAX, defeating the purpose of static analysis) or (b) implement a JAX semantics interpreter (a massive engineering effort with no precedent).
- **Library calls with no source.** The problem statement acknowledges this but "solves" it with "caching of library-function signatures." This means hand-writing mathematical models for hundreds of NumPy/SciPy/JAX functions. This is a person-year of work that the LoC estimate doesn't account for.

### 4.2 Pattern Lifting Brittleness

"Pattern lifting" for force accumulation, finite differences, and quadrature is pattern matching on AST structures. This is inherently brittle:

- A force accumulation loop written as `for i in range(N): f[i] += ...` matches a pattern. The same logic written as `f = np.sum(forces, axis=0)` does not match the same pattern. Both are ubiquitous.
- Finite differences: `(u[i+1] - u[i]) / dx` is recognizable; `np.diff(u) / dx` is different; `scipy.ndimage.convolve1d(u, [1, -1]) / dx` is yet another pattern. All compute the same thing.
- **This is the "compiler front-end" problem.** Every new syntactic variant requires a new pattern. Production simulation codes use all variants. The pattern library will never be complete.

### 4.3 Self-Constructed Benchmarks

**This is the Achilles' heel of the evaluation plan.**

"20+ simulation kernels" written by the tool's developers, annotated with ground-truth by the tool's developers, evaluated by the tool's developers. An adversary would (correctly) argue:

1. The benchmarks will be written in the liftable fragment (because the developers know what their tool can handle).
2. The "injected bugs" will be the kinds of bugs the tool is designed to find.
3. The coverage, detection rate, and localization accuracy numbers will all be inflated relative to real-world performance.

**What would make this credible:**
- **External benchmark:** Conservation bugs from real simulation code repositories (LAMMPS GitHub issues, CESM bug tracker, GROMACS changelogs). These exist — conservation bugs are well-documented in these communities.
- **Adversarial benchmark:** Challenge a geometric numerical integration expert to write code that violates conservation in ways ConservationLint *shouldn't* be able to detect. Measure performance on their examples.
- **Held-out evaluation:** Split benchmarks into development set and test set, with the test set sealed before tool development.

### 4.4 Runtime Monitoring: <5% Overhead

The "fused instrumentation" claim is the key question. The idea: piggyback on forces/energies the simulation already computes.

- **Problem 1:** Many simulations don't compute total energy every step (it's expensive for large systems). You'd need to add an energy computation, which is O(N²) for pairwise interactions — far more than 5% overhead for large systems.
- **Problem 2:** "Without modifying the simulation code" — but you need to instrument it somehow. The problem statement mentions "instrumentation" but doesn't say how. Compile-time instrumentation requires code modification. Runtime hooking (LD_PRELOAD, etc.) adds overhead and complexity. Python monkey-patching is fragile.
- **Problem 3:** The sequential hypothesis testing (CUSUM/SPRT) is indeed standard and O(1) per step — no issue there. But the *quantity being tested* needs to be computed, and that's where the overhead comes from.

### 4.5 Scale: 162K LoC is a Research Group, Not a Paper

The problem statement estimates 162K LoC across 10 subsystems. This is:
- More code than the entire Rust compiler was at version 1.0.
- Approximately the scope of a 5-person team working for 2-3 years.
- Completely incompatible with a single best-paper submission.

A credible scope for a best-paper would be: a proof-of-concept of subsystems 1-4 (code→math extraction + symmetry analysis + backward error analysis + localization), demonstrated on 5-10 small benchmarks, with an honest discussion of limitations. That's maybe 15-25K LoC and a publishable paper.

---

## 5. DEMAND SIGNAL CHECK

### Who has actually asked for this?

**No evidence of demand is presented.** The problem statement asserts need from "climate modelers, fusion simulation teams, molecular dynamics code maintainers, V&V groups" but:

- **No user interviews are cited.** No quotes from potential users. No survey data.
- **No letters of support or expressions of interest.**
- **The V&V community has its own tools and processes.** They may not want an external tool that imposes a new workflow.

### Would they integrate a Rust-based tool into Python/Fortran workflows?

**Significant adoption friction:**

- Climate models (CESM, E3SM) are Fortran. The problem statement relegates Fortran support to a "stretch goal." Without Fortran support, the largest potential user community is excluded.
- MD codes (LAMMPS, GROMACS) are C++. Also a stretch goal.
- The Python ecosystem (JAX-MD, Dedalus) is the initial target, but these are smaller, more academic codebases — not the "multibillion-dollar simulation ecosystem" the problem statement invokes.
- A Rust CLI tool that outputs conservation diagnostics is reasonable to integrate. An LSP server is more invasive. CI integration is the right model.

### Net demand assessment

The demand is **assumed, not validated.** The problem is real (conservation violations do occur and are hard to catch), but the proposed solution may be over-engineered relative to what users actually need. A simpler tool — a conservation-test template library for pytest, with a guide on what to monitor — might satisfy 80% of the practical need.

---

## 6. LLM OBSOLESCENCE — DEEPER CUT

Let me be specific about what an LLM can do today with the right prompt:

**Prompt:** "Here is a Python implementation of a Velocity Verlet integrator with Lennard-Jones forces. I split the force computation into short-range and long-range contributions and apply them in separate kicks. Does this integration scheme conserve (1) total energy, (2) linear momentum, (3) angular momentum? If not, identify which lines cause the violation and what order error they introduce."

**Expected LLM response (GPT-4/Claude, 2024-2026):** A correct analysis identifying that the splitting introduces an O(Δt²) energy error (from BCH), that linear momentum is conserved if forces are pairwise and Newtonian, and that angular momentum conservation depends on the force splitting preserving rotational equivariance. The LLM would likely identify the correct lines, though without formal proof.

**What ConservationLint adds over this:** Formal proof of the error order. Provenance-tagged BCH expansion. Obstruction criterion. Deterministic reproducibility. These matter for V&V — but only if the tool actually works on real code.

**The existential risk:** If the liftable fragment covers only 10-20% of real code, ConservationLint provides formal guarantees on a small subset while LLMs provide heuristic answers on everything. Most users will choose the heuristic that covers 100% over the proof that covers 10%.

---

## 7. ADDITIONAL RED FLAGS

1. **No prototype exists.** The theory/, implementation/, and proposals/ directories are empty. This is a problem statement, not a project. Every claimed capability is aspirational.

2. **The "best paper" framing is premature.** You cannot argue for best paper before you have results. The problem statement reads like a grant proposal, not a paper draft.

3. **The SLAM/Herbie comparisons are grandiose.** SLAM shipped with Windows and verified millions of lines of driver code. Herbie is a working tool used by thousands. ConservationLint is a markdown file. Comparing yourself to landmark papers before writing a line of code is a red flag for judgment.

4. **The 162K LoC estimate suggests the authors don't know what's essential.** A great paper focuses. This problem statement describes an entire research program (3-5 years, multiple PhD theses) and presents it as one project.

5. **"Unanimous signoff after 3 edits"** in the history suggests insufficient adversarial review during crystallization.

---

## 8. VERDICT: CONDITIONAL CONTINUE

Despite the extensive criticism above, I do NOT recommend ABANDON. Here's why:

**What's genuinely good:**
- The core insight (applying backward error analysis to *code* rather than *equations*, with provenance tracking) is novel and interesting.
- The obstruction criterion (T2), if it works, is a genuinely useful capability that no existing tool provides.
- The causal localization idea (tracing symmetry-breaking terms to source lines) is a real contribution to program analysis.
- The combination of program analysis and geometric mechanics is unexplored territory.

**What must change (BINDING CONDITIONS):**

### Condition 1: RADICAL SCOPE REDUCTION (non-negotiable)
Cut the project to a demonstrable core:
- **Keep:** Code→math extraction for a *tiny* fragment (pure NumPy Verlet/leapfrog integrators, <500 LoC input programs).
- **Keep:** Symmetry analysis + backward error analysis + causal localization on that fragment.
- **Keep:** T2 (obstruction criterion) — this is the strongest theoretical claim.
- **Drop:** Runtime monitoring (standard engineering, not novel).
- **Drop:** Repair synthesis (another paper entirely).
- **Drop:** LSP server, CI integration (engineering, not research).
- **Drop:** Fortran/C++ frontends (scope creep).
- **Target:** 15-25K LoC, one clear demo, one clear theorem.

### Condition 2: HONEST COVERAGE MEASUREMENT (non-negotiable)
Before claiming any coverage number:
1. Implement the liftable fragment checker on 5 real-world simulation codes.
2. Measure actual coverage. Report it honestly, even if it's 5%.
3. If coverage is <15%, reconsider whether the static-analysis approach is viable at all.

### Condition 3: EXTERNAL BENCHMARK (non-negotiable)
Include at least 5 conservation bugs sourced from real simulation code repositories (not self-constructed). LAMMPS, GROMACS, and CESM all have public bug trackers. Find real conservation bugs and test against them.

### Condition 4: LLM BASELINE (strongly recommended)
Include GPT-4/Claude as a baseline in the evaluation. Paste each benchmark kernel + conservation question into the LLM. Report detection rate, localization accuracy, and analysis time. If the LLM matches ConservationLint on >70% of cases, the formal-methods framing needs revision.

### Condition 5: VALIDATE T2 BEFORE BUILDING (strongly recommended)
T2 (the obstruction criterion) is the crown jewel. Prove it on paper for 2-3 examples before building the full tool. If T2 turns out to be trivial (just "check the Lie brackets" with no structural insight) or infeasible (the criterion is exponential), the paper's theoretical contribution evaporates.

### Condition 6: USER DEMAND VALIDATION (recommended)
Talk to 3-5 simulation developers. Show them a mock-up of ConservationLint output. Ask: "Would you use this? What would you pay for it? What conservation bugs have you encountered?" If nobody cares, stop.

---

## 9. WHAT'S WORTH SCAVENGING (if abandoned)

If the project is abandoned entirely, the following ideas are independently publishable:

1. **Provenance-tagged BCH expansion** (T1, refined): A short paper in a numerical methods journal formalizing the attribution of BCH terms to sub-integrators in heterogeneous compositions. Useful for the geometric integration community even without the code-analysis application.

2. **Conservation bug benchmark:** A curated benchmark of conservation violations in simulation code, with ground-truth annotations. Useful for the scientific software engineering community regardless of what tool is built.

3. **The "physics-aware program analysis" vision paper:** A 4-page workshop paper at PLDI or ICSE-NIER outlining the idea of using domain-specific mathematical semantics in program analysis. No tool needed — just the idea and a roadmap.

---

## 10. SUMMARY SCORECARD

| Dimension | Score (1-5) | Notes |
|-----------|-------------|-------|
| Novelty of core idea | 4 | Genuinely new combination; the "bridge" is real |
| Mathematical depth | 2.5 | T2 might be deep; T1 and T3 are shallow |
| Engineering feasibility | 2 | Code→math extraction is underestimated by 5-10x |
| Evaluation credibility | 1.5 | Self-constructed benchmarks, unverified coverage claims |
| Demand validation | 1 | Zero evidence of actual user demand |
| LLM resilience | 2.5 | Formal guarantees help, but coverage gap is existential |
| Scope realism | 1 | 162K LoC is a research program, not a paper |
| Overall best-paper potential | 2.5 | Good idea, terrible scoping, no evidence yet |

**Final word:** This is a *research direction* masquerading as a *project plan*. The core insight is worth pursuing, but the current problem statement needs to be reduced by ~80% in scope and grounded in at least a proof-of-concept before any claims of "best paper" viability are credible. The greatest risk is not that the idea is wrong — it's that the team builds 162K LoC of infrastructure around an extraction engine that works on toy examples and fails on everything else.

CONDITIONAL CONTINUE — meet the binding conditions or abandon.

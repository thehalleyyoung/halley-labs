# Scavenging Synthesizer Evaluation: ConservationLint (sim-conservation-auditor)

**Role:** Scavenging Synthesizer — Best-Paper Verification Committee
**Date:** 2025-07-18

---

## CROWN JEWELS: What's Genuinely Valuable Here?

### Crown Jewel #1: The Conservation-Aware IR and Code→Math Extraction (Subsystems 1–2)

This is the most durable idea in the proposal. The notion of a formal intermediate representation that captures not just the *computation* a simulation performs but the *mathematical structure it discretizes* is genuinely novel. No existing tool operates in the code→math direction for conservation analysis. Even if the Noether/symmetry machinery never works, a conservation-aware IR for numerical kernels has standalone value as infrastructure for any future physics-aware program analysis tool. It could support:

- Automated backward error analysis (even without the full Noether pipeline)
- Numerical method identification ("this code implements Störmer-Verlet")
- Regression testing on mathematical equivalence after refactoring

**Survivability:** High. This component is useful even if T1–T3 are never proved and the repair engine is never built.

### Crown Jewel #2: Heterogeneous Composition Modified Equation Theory (T1)

This is the strongest mathematical contribution. Classical backward error analysis handles single methods; real codes compose multiple methods. The provenance-tagged BCH expansion — where each term in the modified Hamiltonian carries attribution to specific sub-integrators — is a clean, well-motivated generalization that the geometric numerical integration community would value independent of any software artifact. The "tagged" structure is the specific novel contribution; the rest is known machinery (BCH, modified equations) applied in a new configuration.

**Survivability:** High as a standalone theory paper. The theorem statement is clean, the proof strategy (induction on composition depth, tracking tags through BCH terms) is straightforward, and the result is immediately useful for anyone analyzing operator splitting in practice.

### Crown Jewel #3: The Benchmark Suite of Conservation-Annotated Simulation Kernels (Subsystem 8)

There is currently no standard benchmark for conservation correctness in numerical simulation code. A curated set of 20+ kernels spanning N-body, MD, Hamiltonian PDEs, fluids, and climate — each annotated with ground-truth conservation properties and injected-bug variants — would be a community resource. This is the kind of artifact that gets cited for a decade regardless of whether the analysis tool that motivated it succeeds.

**Survivability:** Very high. Useful to anyone working on numerical method verification, structure-preserving integration, or scientific software testing.

### Minimum Viable Contribution

A paper presenting (1) the conservation-aware IR, (2) T1 with proof, (3) a static analysis pipeline that detects conservation violations (without repair) on the benchmark suite, compared against Daikon and naive energy monitoring. This is publishable at a strong SE/PL venue even if the repair engine, runtime monitor, and obstruction detection (T2) are never completed.

---

## HONEST SCORING

### 1. EXTREME AND OBVIOUS VALUE — Score: 6/10

**The actual demand.** The proposal paints a compelling picture of conservation violations as "silent killers," and the Wan et al. 2019 climate model example is persuasive. However, the realistic adoption story is narrower than claimed:

- **National lab V&V teams:** YES, genuine demand. These groups spend weeks on manual conservation auditing. But they primarily use Fortran/C++ codes (LAMMPS, E3SM, WRF), not Python. The Python-first scope limits immediate adoption.
- **MD code maintainers (LAMMPS, GROMACS):** These codebases are C++/Fortran. The tool wouldn't apply to them directly. OpenMM plugin developers writing Python wrappers are a narrow niche.
- **Academic simulation researchers using Python (Dedalus, JAX-MD, SciPy):** This is the realistic early-adopter pool. It's real but small — perhaps a few hundred active developers worldwide.
- **Climate modelers:** Their models are overwhelmingly Fortran. The Python ecosystem (CliMA/Oceananigans is Julia, not Python) is thin.

**Realistic adoption path:** (1) Academic researchers using JAX-MD or Dedalus adopt it for method development; (2) course instructors use the benchmark suite for teaching structure-preserving integration; (3) if the tool proves itself, justify the Fortran/C++ frontend investment. This is a 3–5 year path to meaningful adoption, not immediate.

**Amendment needed:** The demand narrative should be reframed around the *research tool* use case (numerical methods researchers verifying their own implementations) rather than the *production CI/CD* use case, which requires Fortran/C++ support that is explicitly a stretch goal.

### 2. GENUINE DIFFICULTY — Score: 7/10

**Genuinely hard subproblems (3 of 7):**

- **Subproblem 1 (Code→math extraction):** Genuinely hard. Handling aliasing, mutation, library calls, and floating-point-vs-real semantics in a faithful lifting is a well-known open problem. The "liftable fragment" approach (T3) is the right strategy — honest scope limitation rather than impossible generality — but even within the fragment, correctly handling NumPy's broadcasting, fancy indexing, and in-place mutation is significant engineering.
- **Subproblem 5 (Obstruction detection / T2):** Genuinely novel and hard. The claim that conservation-achievability reduces to a finite algebraic test is a strong claim. I suspect the proof will require significant restrictions (fixed order, fixed splitting structure, specific symmetry classes) to go through. The proposal hand-waves the details.
- **Subproblem 4 (Causal localization):** Novel framing but moderate difficulty. Once T1 provides provenance-tagged terms, tracing back through the IR to source lines is essentially a form of symbolic dependency tracking. The "differential symbolic slicing" framing is nice but the implementation is straightforward given T1.

**Well-understood subproblems (4 of 7):**

- **Subproblem 2 (Lie-symmetry analysis):** With the restricted ansatz (only physical symmetry groups), this reduces to structured linear algebra. The proposal correctly identifies this. Not easy engineering, but not a research contribution.
- **Subproblem 3 (Backward error analysis):** BCH expansions are textbook material. The provenance tagging is the novelty (T1), but the computation itself is well-understood.
- **Subproblem 6 (Runtime monitoring):** CUSUM/SPRT on conserved quantities is standard statistical process control. The fused instrumentation for <5% overhead is a nice engineering target but not novel.
- **Subproblem 7 (Repair synthesis):** Template-based repair with constraint checking. The proposal honestly describes this as a catalog of known fixes, not a general synthesis procedure. Moderate engineering, low novelty.

**Is the 162K LoC estimate honest?**

No. It's padded by approximately 40–50%. Specific inflations:

| Subsystem | Claimed | Realistic | Justification |
|-----------|---------|-----------|---------------|
| Python frontend + IR | 25K | 15K | Tree-sitter grammars are compact; IR design is a data structure, not 25K of code |
| Symbolic algebra engine | 20K | 10K | For polynomial-rational systems with the restricted scope claimed, a focused algebra engine doesn't need 20K. SymPy integration could halve this. |
| Variational symmetry analyzer | 18K | 8K | With the restricted ansatz, this is a sparse linear solver with domain-specific setup |
| Backward error engine | 18K | 10K | BCH expansion is a recursive formula; the provenance tagging adds bookkeeping but not 18K of code |
| Runtime monitor | 12K | 8K | Standard instrumentation + CUSUM. Fair estimate. |
| Causal localization | 15K | 6K | Given T1's provenance tags, this is dependency tracking through the IR |
| Repair engine | 12K | 6K | Template catalog. 30 patterns × 200 LoC/pattern = 6K. |
| Benchmark suite | 20K | 15K | 20+ kernels with annotations. Fair but kernels can be compact. |
| Evaluation harness | 10K | 6K | Standard benchmark infrastructure |
| CLI/IDE/LSP | 12K | 8K | LSP server is boilerplate-heavy. Fair-ish. |
| **Total** | **162K** | **~92K** | |

**Minimum LoC for a compelling artifact:** ~60K (IR + algebra + symmetry + backward error + localization + benchmarks + evaluation — no runtime monitor, no repair engine, no LSP).

### 3. BEST-PAPER POTENTIAL — Score: 5/10

**Target venues (realistic):**

- **PLDI or OOPSLA (PL/SE):** Best fit. "Physics-aware program analysis" is a new angle for this community. The code→math extraction + conservation checking story fits the PLDI tradition of domain-specific formal methods (cf. SLAM, Herbie). **Probability of acceptance: moderate-high. Probability of best paper: low** — the committee would need to be persuaded that the scientific computing niche justifies a best-paper award over more broadly applicable PL work.
- **SC (Supercomputing):** The scientific computing audience would appreciate the domain relevance but may find the formal methods machinery opaque. SC best papers tend to be about performance at scale or new algorithms, not program analysis.
- **ICSE (Software Engineering):** Possible. The "linting for physics" angle is appealing. But ICSE reviewers may not have the geometric mechanics background to evaluate T1–T2.
- **CAV (Verification):** If the formal results are strong enough. But CAV expects machine-checked proofs or at least very rigorous formal treatment.

**The "killer result":** The single most impressive experimental outcome would be: "ConservationLint automatically rediscovered the conservation-violating coupling scheme in [real climate model X] that took human experts 3 years to find (Wan et al. 2019), localized it to the exact code region, and suggested the published fix." If that demo can be shown *on the actual code or a faithful reproduction*, it's a showstopper. But this requires the full pipeline working end-to-end on a non-trivial codebase — a high bar.

**A more achievable killer result:** "On a benchmark of 20 simulation kernels with 50 injected conservation bugs, ConservationLint achieves 92% detection rate with 8% false positive rate and correctly localizes 75% of violations to within 3 source lines. Daikon detects 15%. Energy-only monitoring detects 40%." This is solid but incremental — it establishes utility without the drama of the real-bug discovery story.

**Honest assessment:** This is a strong workshop/symposium paper and a solid top-venue acceptance. It is a *plausible* best-paper candidate at PLDI if the narrative is crisp and the real-bug-rediscovery demo works, but it faces stiff competition from work with broader applicability.

### 4. LAPTOP CPU + NO HUMANS — Score: 8/10

**This is a genuine strength of the proposal.** The entire analysis pipeline is symbolic/algebraic, not statistical or ML-based. No GPU needed anywhere. The computational bottlenecks are:

- **IR extraction:** Linear in code size, bounded by Tree-sitter parsing speed. Fast.
- **Lie symmetry analysis:** Structured linear algebra with the restricted ansatz. Milliseconds for typical kernels.
- **BCH expansion:** Polynomial in the number of terms at each order. Seconds for practical cases (k ≤ 10 sub-integrators, order ≤ 6).
- **Benchmark evaluation:** Embarrassingly parallel across kernels. 20 kernels × 10 minutes = 200 minutes sequential, easily parallelized.

**Where "no GPU" might hurt:** Only in the comparison against Noether's Razor baseline. If Noether's Razor requires GPU for training, the comparison may need to be done with pre-trained models or on a restricted subset. The proposal acknowledges this ("Run CPU-only (if feasible) or with restricted GPU budget"). This is a minor evaluation gap, not a fundamental constraint.

**Bottleneck component:** The code→math extraction (Subsystem 1) is the pacing item, not for computational cost but for *engineering completeness*. Every NumPy function, every SciPy routine, every array operation pattern needs a lifting rule. This is a long tail of pattern coverage, and gaps will reduce the effective coverage metric.

---

## AMENDMENT PROPOSALS

### Amendment 1: Reframe Demand Around Research Tooling, Not Production CI/CD

**Weakness:** The value proposition leads with national-lab CI/CD pipelines and production climate models, but the tool only supports Python and the realistic user base is academic numerical methods researchers.

**Fix:** Lead with the "numerical methods research verification" use case: "When a researcher implements a new structure-preserving integrator, ConservationLint automatically verifies that the implementation actually preserves the claimed conservation laws." This is the use case that matches the Python scope, requires no Fortran/C++ support, and has an immediate audience at every computational math department. Reposition the national-lab CI/CD vision as a *future direction* enabled by language-agnostic IR.

### Amendment 2: Strengthen T2 or Downgrade It to a Conjecture

**Weakness:** T2 (Computable Obstruction Criterion) is the weakest of the three theorems. The claim that conservation-achievability reduces to a *finite* algebraic test on BCH structure is strong but under-specified. The proposal doesn't discuss: (a) what "given order" means precisely (is it any fixed order? the order of the existing method? all orders?), (b) whether the criterion is necessary and sufficient or only sufficient, (c) computational complexity of the test.

**Fix:** Either (a) state T2 precisely enough that a mathematician can check the proof, including all quantifiers and the exact algebraic objects being tested, or (b) downgrade T2 to a conjecture ("Obstruction Conjecture") with computational evidence from the benchmark suite. Option (b) is more honest and still publishable — the conjecture + evidence would be a contribution even unproved. The paper could then claim T1 and T3 as theorems and T2 as a validated conjecture.

### Amendment 3: Cut Repair Synthesis; Invest in Deeper Localization Evaluation

**Weakness:** The repair engine (Subproblem 7, Subsystem 7) is the weakest component — template-based repair with 30 patterns is engineering, not research. The 60% repair success rate target is modest and the contribution thin.

**Fix:** Drop repair synthesis from the core contribution. Instead, invest the saved effort into a deeper evaluation of *localization accuracy* — the genuinely novel feature. Specifically:
- Measure localization precision at different granularities (function, loop, line, expression)
- Compare against random baseline, Daikon-identified invariant violations, and manual expert localization
- Conduct a user study (or at minimum, a vignette-based evaluation) where numerical analysts receive ConservationLint's localization output and rate its usefulness

This trades a weak component (repair) for a stronger evaluation of a strong component (localization), which is much more persuasive to reviewers.

### Amendment 4: Tighten the Benchmark Suite Definition

**Weakness:** "20+ kernels" is vague. The benchmark categories (N-body, MD, Hamiltonian PDEs, fluids, climate) are listed but the specific codes aren't committed to.

**Fix:** Commit to a specific list of 25 kernels in the proposal:
- 5 from JAX-MD (Lennard-Jones, soft sphere, Stillinger-Weber, EAM, Morse)
- 5 from Dedalus (KdV, NLS, Burgers, Rayleigh-Bénard, shallow water)
- 5 hand-written N-body codes (leapfrog, Yoshida, Ruth, Forest-Ruth, McLachlan 4th-order)
- 5 SciPy-based ODE integrators applied to Hamiltonian systems
- 5 conservation-violating mutants (energy leak, momentum symmetry break, angular momentum break, mass non-conservation, symplecticity violation)

This is concrete, achievable, and reviewers can evaluate the scope.

### Amendment 5: Add a "Coverage Honesty" Section to the Paper

**Weakness:** The 40–60% coverage estimate for the liftable fragment is stated without justification. If the actual number is 25%, the tool's utility collapses.

**Fix:** Add a dedicated empirical section measuring the liftable fragment coverage on 5 real codebases:
- JAX-MD (~5K LoC kernel code)
- Dedalus (~10K LoC)
- A SciPy ODE example suite
- A simple climate kernel (gray radiation model)
- An MD simulation using ASE (Atomic Simulation Environment)

Report: (a) total kernel LoC, (b) LoC in liftable fragment, (c) failure taxonomy breakdown (opaque library call: X%, data-dependent branching: Y%, non-polynomial nonlinearity: Z%). This makes the coverage claim falsifiable and gives reviewers confidence.

### Amendment 6: Evaluation Plan Needs Stronger Baselines

**Weakness:** The baselines are soft. Daikon is a straw man (it was never designed for conservation analysis). Noether's Razor may not be runnable CPU-only. "Energy-only monitoring" is trivially weak. There's no baseline representing what a *competent numerical analyst* achieves with existing tools (SymPy + manual inspection).

**Fix:** Add two baselines:
- **SymPy-assisted manual analysis:** Give a graduate student SymPy, the code, and 2 hours per kernel. This represents the current state of practice.
- **Abstract interpretation baseline:** Run a standard numerical abstract interpreter (e.g., Fluctuat-style interval analysis) on the kernels to check if naive interval bounds on conserved quantities suffice.

These establish the gap between "what exists" and "what ConservationLint provides" much more convincingly.

---

## SALVAGE ANALYSIS (Ranked by Value)

### Rank 1: The Benchmark Suite (Option 3)
**Value: HIGH.** A curated, conservation-annotated benchmark suite of simulation kernels is a community resource with no existing equivalent. It would be cited by anyone working on numerical method verification, structure-preserving integration testing, or scientific software quality. Could be published as a standalone dataset/benchmark paper at SC, JOSS, or a SciPy conference. **Estimated effort: ~15K LoC, 2–3 months.**

### Rank 2: Conservation-Aware IR + Code→Math Extraction (Option 1)
**Value: HIGH.** Even without the Noether/symmetry pipeline, the IR is reusable infrastructure. A paper titled "A Conservation-Aware Intermediate Representation for Numerical Simulation Code" presenting the IR design, the liftable fragment formalization (T3), and the extraction pipeline would be a solid contribution to the PL or scientific computing communities. **Estimated effort: ~25K LoC, 4–5 months.**

### Rank 3: Survey/Systematization Paper (Option 6)
**Value: MEDIUM-HIGH.** A systematization-of-knowledge (SoK) paper on "Conservation in Numerical Simulation Code: Theory, Practice, and Tooling Gaps" would be genuinely useful. The space connecting geometric numerical integration, software verification, and scientific computing practice is poorly mapped. This paper would establish vocabulary, identify open problems, and position future tool-building work. Could target IEEE S&P's SoK track (if framed as security/reliability of scientific infrastructure) or a survey journal. **Estimated effort: minimal code, 2–3 months writing.**

### Rank 4: ConservationLint-Lite (Option 5)
**Value: MEDIUM.** A tool that only handles single-method integrators (no composition theory, no T1) is simpler but still useful. It could detect violations in simple leapfrog/Verlet implementations, which covers many teaching and research codes. The loss is the composition theory and causal localization across sub-integrators — which is precisely what makes the full system novel. **Risk: may be too incremental for a top venue.** Estimated effort: ~35K LoC.

### Rank 5: Runtime Conservation Monitor (Option 2)
**Value: MEDIUM-LOW.** A standalone runtime monitor that tracks conserved quantities with CUSUM/SPRT drift detection is useful but not novel. It's essentially a well-engineered version of what many simulation codes already do ad hoc (plot energy vs. time, check if it drifts). The value-add over hand-written checks is statistical rigor and automation, but the intellectual contribution is thin. **Would not be publishable as a standalone research paper.** Useful as an open-source tool, published at JOSS or as a library.

### Rank 6: Theorems T1–T3 as a Pure Math Paper (Option 4)
**Value: LOW-MEDIUM.** T1 (provenance-tagged modified equations) is publishable in a numerical methods journal (BIT, Numerische Mathematik, JCP). T2, if proved, would be a nice result. T3 is a formalization contribution that the pure math community would find uninteresting (it's a PL/SE contribution wearing math clothing). As a pure math paper, only T1 carries weight, and it's a modest extension of known BCH/modified equation theory — publishable but not exciting. **The theorems derive most of their value from being *connected to a tool*; without the tool, they're incremental.**

---

## SCOPE REDUCTION PROPOSAL: The ~80K LoC System

If the project must be cut to ~80K LoC, here is the minimal system that still tells a compelling story:

### What to Keep (Core Story: "Detect and Localize Conservation Violations in Python Simulation Code")

| # | Subsystem | LoC | Rationale |
|---|-----------|-----|-----------|
| 1 | Python frontend + conservation-aware IR | 15K | Essential — the code→math bridge |
| 2 | Symbolic algebra engine | 10K | Essential — polynomial manipulation for symmetry analysis |
| 3 | Variational symmetry analyzer | 8K | Essential — the Noether connection |
| 4 | Backward error diagnostic engine | 10K | Essential — T1, the core theorem |
| 6 | Causal localization engine | 6K | Essential — the "killer feature" |
| 8 | Benchmark suite (reduced to 15 kernels) | 12K | Essential — evaluation substrate |
| 9 | Evaluation harness | 6K | Essential — automated metrics |
| 10 | CLI + basic reporting (no LSP) | 5K | Minimal interface |
| | **Total** | **~72K** | |

### What to Cut

| # | Subsystem | Claimed LoC | Why Cut |
|---|-----------|-------------|---------|
| 5 | Runtime conservation monitor | 12K | Nice-to-have, not core contribution. Demonstrate with offline post-hoc analysis instead. |
| 7 | Repair suggestion engine | 12K | Weakest component. Report violations only; don't suggest fixes. |
| 10 | LSP server / IDE integration | ~7K | Engineering polish, not research contribution. CLI suffices. |
| — | Fortran/C++ frontends | (stretch goal) | Explicitly out of scope in reduced version |

### The Reduced Story

"ConservationLint takes Python simulation code, extracts the mathematical structure it discretizes, identifies the symmetries of the continuous system via Noether's theorem, computes provenance-tagged modified equations for the heterogeneous integrator composition, and localizes conservation-law-violating code regions to specific source lines. On a benchmark of 15 annotated simulation kernels with injected conservation bugs, it achieves >90% detection rate and >70% localization IoU, outperforming all baselines."

This is a clean, publishable story. No runtime monitoring, no repair, no IDE integration — just the static analysis pipeline that embodies the core intellectual contribution.

---

## COMPARISON TO PORTFOLIO SIBLINGS

### Direct Sibling: fp-error-audit-engine (Penumbra) — area-099-scientific-computing

**Overlap:** Both target scientific Python (NumPy/SciPy) and both perform code-level diagnosis of numerical issues. Both build a formal IR from Python code and trace error/violation back to source lines.

**Distinction:** Penumbra targets floating-point precision errors (expression-level, individual operations); ConservationLint targets conservation-law violations (system-level, discretization structure). These are genuinely complementary:
- A simulation can have perfect FP arithmetic and still violate conservation (structural discretization issue)
- A simulation can perfectly conserve energy and still have catastrophic FP cancellation

**Potential shared components:**
- **Python frontend / Tree-sitter parsing:** Both need to parse Python numerical code. A shared Tree-sitter grammar + AST walker could save 5–8K LoC across both projects.
- **NumPy/SciPy operation semantics database:** Both need to understand what NumPy functions do. A shared "NumPy operation semantics" module (type signatures, mathematical equivalents, broadcasting rules) would benefit both.
- **Source-line attribution infrastructure:** Both trace from analysis results back to source lines. The "rewriting + patch application" (Penumbra) and "causal localization" (ConservationLint) share the source-mapping plumbing.
- **Benchmark infrastructure:** Both need Python simulation code for evaluation. The benchmark kernels could overlap.

**Recommendation:** Factor out a shared `sci-python-analysis-core` library providing (1) Tree-sitter Python parsing, (2) NumPy/SciPy semantics database, (3) source-line attribution, (4) benchmark kernel collection. Both projects depend on it. This saves ~10–15K LoC total and creates a more coherent portfolio.

### Distant Sibling: fp-condition-flow-engine — area-079-scientific-computing

**Overlap:** Minimal. This project focuses on condition number tracking through computation graphs. Could potentially feed into ConservationLint's analysis (ill-conditioned conservation quantity computation), but the connection is loose.

### Other Relevant Siblings

- **behavioral-semver-verifier (area-049):** Semantic versioning verification for libraries. ConservationLint could be framed as "conservation-semantic versioning" for simulation codes — detect when a code change breaks a conservation property. Conceptual connection only; no shared code.
- **ml-pipeline-static-analyzer (area-081):** Static analysis for ML pipelines. Different domain but similar tool architecture (parse code → build IR → check domain-specific properties). Architectural patterns could be shared.

---

## SUMMARY SCORECARD

| Axis | Score | Key Factor |
|------|-------|------------|
| Extreme and Obvious Value | **6/10** | Real but narrow demand; Python-only limits adoption |
| Genuine Difficulty | **7/10** | 3 of 7 subproblems genuinely hard; LoC inflated ~40% |
| Best-Paper Potential | **5/10** | Strong venue paper; best-paper requires real-bug demo working |
| Laptop CPU + No Humans | **8/10** | Natural strength; symbolic/algebraic throughout |
| **Weighted Overall** | **6.3/10** | |

### Bottom Line

ConservationLint contains a genuinely novel idea — applying Noether's theorem to program analysis of simulation code — wrapped in an overly ambitious scope. The crown jewels (conservation-aware IR, provenance-tagged modified equation theory, benchmark suite) are each independently valuable. The full 162K LoC, 7-subsystem vision is overbuilt for a single paper or project. Reduced to ~72K LoC with the static analysis pipeline as the core story, dropping repair synthesis, runtime monitoring, and IDE integration, this becomes a focused, achievable project with a clear path to a strong venue publication. The "bridge between geometric mechanics and program analysis" narrative is compelling and genuine; it just needs to be told with a smaller, more honest system behind it.

**Strongest recommendation:** Invest heavily in making the code→math extraction (Subsystem 1) robust on real code, prove T1 cleanly, and build the benchmark suite carefully. These three components are the durable contributions. Everything else is packaging.

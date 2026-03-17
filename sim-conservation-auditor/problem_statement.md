# ConservationLint: Bridging Noether's Theorem and Program Analysis to Detect and Localize Conservation-Law Violations in Scientific Simulation Code

**Slug:** sim-conservation-auditor

> **AMENDED VERSION** — Date: 2025-07-18
>
> This document incorporates all amendments identified by the verification gate team (cross-critique §9). Key changes: (1) scope reduced to ~90K LoC with phased delivery; (2) runtime monitor, repair synthesis, and LSP/IDE integration moved to Extensions; (3) coverage claim replaced with empirical measurement plan (preliminary estimate 20–40%); (4) external benchmarks added from real bug trackers; (5) T3 renamed to Liftable Fragment Characterization; (6) T2 strengthened with quantifiers, complexity bounds, and truncation limitations; (7) LLM competition section added; (8) demand narrative reframed around numerical methods research verification; (9) 25 specific benchmark kernels committed; (10) validation phase added; (11) 10-minute budget qualified; (12) detection rate target lowered to ≥85% on combined benchmarks.

---

## Problem and Approach

Scientific simulations encode physical law in imperative code, yet the mathematical structures that guarantee physical fidelity—conservation of energy, momentum, angular momentum, charge—are nowhere explicit in that code. They live in the continuous equations the programmer had in mind, not in the loops, arrays, and floating-point operations that actually execute. When a discretization step silently breaks a conservation law, the violation is invisible to compilers, linters, and type systems. It compounds over millions of time steps until mass appears from nothing, orbits precess without cause, or climate models drift into unphysical states. Today, the only defense is expert manual audit: a physicist reads the code, reconstructs the mathematics in their head, and checks invariants by hand. This does not scale.

**The primary use case is numerical methods research verification.** When a researcher implements a new structure-preserving integrator, ConservationLint automatically verifies that the implementation actually preserves the claimed conservation laws, localizes any violation to the specific code region responsible, and determines whether the violation is fixable locally or requires algorithmic restructuring. This use case matches the tool's Python scope, requires no Fortran/C++ support, and has an immediate audience in every computational mathematics department and scientific computing group. The broader vision—conservation auditing as CI/CD infrastructure for production simulation codes at national laboratories—is a future direction enabled by the language-agnostic IR, conditional on demonstrated value in the research verification setting and eventual Fortran/C++ frontend support.

ConservationLint builds the first automated bridge from simulation *code* back to the *mathematical conservation structure* it is supposed to preserve. The tool ingests Python simulation code (with a language-agnostic IR enabling future Fortran and C++ extension), extracts the numerical kernel into a formal intermediate representation via Tree-sitter parsing and domain-specific lifting, identifies the continuous symmetries of the reconstructed system using restricted Lie-symmetry analysis, computes the corresponding conserved quantities via Noether's theorem, and then diagnoses *how* the discretization breaks each conservation law using backward error analysis on the composed integrator. Crucially, when a violation is detected, ConservationLint performs *causal localization*: it traces the dominant symmetry-breaking terms in the modified equation back to specific source lines, giving the developer a precise, actionable diagnostic—not a vague global warning, but "line 247 of your leapfrog step introduces an O(Δt²) angular momentum error because the force splitting breaks rotational equivariance."

The technical heart of the system is a new theory of *heterogeneous composition modified equations* that generalizes classical backward error analysis (McLachlan & Quispel 2002, Hairer, Lubich & Wanner 2006) from single-method integrators to the patchwork of mixed schemes that real simulation codes actually use: Verlet for bonded forces composed with Ewald summation for electrostatics composed with a thermostat, each contributing different symmetry-breaking terms. This theory yields a computable *obstruction criterion* that determines when no local modification to a code region can restore a broken conservation law—telling the developer not just that something is wrong, but whether it is *fixable* in place or requires architectural change. A formal *liftable fragment characterization* delineates exactly which subsets of imperative numerical code admit this analysis, with empirically measured coverage and a failure taxonomy for code that falls outside the fragment.

The core contribution is the static analysis pipeline: extraction → symmetry analysis → backward error diagnosis → causal localization. This pipeline runs as a CLI tool in CI, flags conservation regressions on pull requests, and provides detailed diagnostic reports. For the scientific computing community—numerical methods researchers, molecular dynamics code maintainers, and V&V groups—this transforms conservation verification from an expensive, error-prone expert activity into an automated, repeatable check. For the programming languages and software engineering communities, ConservationLint demonstrates a new paradigm: *physics-aware program analysis*, where domain-specific mathematical semantics (not just types or memory safety) become first-class properties that tools can reason about.

---

## Value Proposition

**Who needs this?** Numerical methods researchers who implement structure-preserving integrators and need automated verification that their implementations actually preserve the claimed conservation laws—the immediate, primary audience. Academic simulation researchers using Python frameworks (JAX-MD, Dedalus, SciPy) who need conservation regression checks during development. Molecular dynamics code maintainers (OpenMM plugin developers, ASE-based simulation authors) where subtle integrator composition bugs silently degrade thermodynamic ensemble properties. Course instructors teaching structure-preserving integration who need a tool that demonstrates conservation analysis concretely.

**Future audience (conditional on Fortran/C++ frontend support).** Climate modelers running century-scale Earth system simulations where energy conservation violations of 0.1 W/m² corrupt radiative balance conclusions. Fusion energy simulation teams at ITER and national laboratories. V&V teams at DOE national laboratories who currently spend weeks of expert time manually auditing conservation properties for each code release. National-lab CI/CD integration is a future direction, not a launch target.

**Why urgently?** Conservation violations are the silent killers of computational science. Unlike crashes, type errors, or even numerical overflow, a conservation violation produces *plausible-looking but quantitatively wrong* results. Energy created from nothing doesn't trigger an exception; it just makes your climate 0.3°C warmer over a century. Angular momentum leaking from a molecular system doesn't segfault; it just shifts your free energy estimate by 2 kJ/mol. These errors compound monotonically over simulation time, are invisible to standard testing (they require long-horizon integration to manifest), and corrupt qualitative scientific conclusions. A 2019 study (Wan et al., JAMES) traced a persistent warm bias in a major climate model to a conservation-violating coupling scheme that had passed all unit tests and code reviews. The bug existed for three years. No existing software tool would have caught it.

Existing conservation monitors (GROMACS `gmx energy`, LAMMPS `thermo_style`, ESMValTool) detect *that* conservation is violated, but not *why* or *where*—they are the equivalent of "your test failed" without a stack trace. Property-based testing with physics oracles can check energy drift but cannot perform causal localization, obstruction detection, or provenance-tagged backward error analysis. ConservationLint's value proposition is precisely these qualitatively different capabilities.

**What becomes possible?** Automated conservation auditing as a CI check for Python simulation codes: every pull request is automatically analyzed for conservation regressions, with source-line attribution. A benchmark-driven culture of conservation correctness, analogous to how AddressSanitizer transformed memory safety practices. Ultimately, a world where "does this code conserve what it should?" is answered by a tool, not by a month of expert audit.

**Demand validation.** During Phase 1, we will conduct 3–5 structured interviews with simulation developers (targeting JAX-MD contributors, Dedalus users, and SciPy ODE integrator maintainers). We will show mock-up ConservationLint output and document whether they would use the tool, what conservation bugs they have encountered, and what diagnostic information they most need.

---

## Technical Difficulty

### Hard Subproblems

1. **Code-to-mathematics extraction.** Lifting imperative numerical code (with aliasing, mutation, control flow, library calls) into a formal mathematical representation of the discrete dynamical system it implements. This requires defining a precise "liftable fragment" of imperative code, handling array semantics, recognizing numerical patterns (finite differences, quadrature, force accumulation), and producing a symbolic representation faithful to the floating-point semantics that actually executes—not the real-number semantics the programmer intended.

2. **Restricted Lie-symmetry analysis at scale.** Classical Lie-symmetry methods (Olver 1993, Bluman & Anco 2002) determine symmetry generators by solving overdetermined PDE systems. For the polynomial-rational systems arising from discretized physics, these reduce to large sparse polynomial systems. Solving them tractably—without the exponential blowup of general Gröbner basis computation—requires exploiting the structure of physical symmetry groups (translations, rotations, Galilean boosts) and working with a *restricted ansatz* that targets the symmetries conservation laws actually arise from.

3. **Heterogeneous backward error analysis.** Classical modified equation theory analyzes a *single* numerical method applied to a *single* ODE. Real simulation codes compose multiple methods (splitting schemes, operator-splitting, multi-rate integration, mixed implicit-explicit steps) with different orders and different symmetry properties. Computing the modified equation for such heterogeneous compositions requires extending BCH (Baker-Campbell-Hausdorff) expansion theory to non-uniform operator sequences and tracking which terms in the expansion originate from which sub-integrator—the prerequisite for causal localization.

4. **Causal localization of conservation violations.** Given that the modified equation has a symmetry-breaking term, tracing that term back through the composition structure and the code-to-math lifting to identify the *specific source lines* responsible. This is a novel form of program slicing—"differential symbolic slicing"—where the slice criterion is defined not by data dependence but by contribution to a specific term in a formal power series.

5. **Obstruction detection.** Determining computably whether a conservation violation is *locally repairable* (fixable by modifying the offending code region) or *architectural* (intrinsic to the chosen splitting/coupling strategy). This requires characterizing the image of the "discretize" map in the space of modified equations and checking whether any element with the desired symmetry lies in that image—a problem in computational algebra with no known prior solution.

### Why This Requires Genuine Engineering Breakthroughs

No existing tool operates in the code→math direction for conservation analysis. Symbolic algebra systems (SymPy, Maple) analyze *equations*, not *code*. Program analysis tools (Infer, Coverity) check memory safety and type properties, not physical invariants. Herbie optimizes floating-point accuracy but has no notion of conservation. DIG (Nguyen et al., ICSE 2012) discovers polynomial invariants from program traces but targets general-purpose invariant mining without physics-specific structure; it cannot exploit Noether's theorem, provide causal localization, or determine whether violations are architecturally repairable. Noether's Razor (Cranmer et al., NeurIPS 2024) discovers conserved quantities from *trajectory data* via learned models—it does not analyze code, requires GPU training, and provides statistical rather than exact guarantees. Devito and Firedrake go *from* equations *to* code (the forward direction); ConservationLint goes *from* code *to* equations (the inverse). The combination of program analysis, symbolic algebra, and geometric mechanics in a single tool chain is unprecedented.

**Portfolio differentiation.** ConservationLint is distinct from all sibling projects in the research portfolio. The closest neighbor is `fp-diagnosis-repair-engine`, which targets floating-point accuracy errors (Herbie-lineage: numerical precision of individual expressions); ConservationLint targets conservation-law violations (Noether-lineage: global physical invariants of dynamical systems). The two are complementary—a simulation can have perfectly accurate floating-point arithmetic and still violate conservation laws due to discretization structure, and vice versa. Other portfolio projects in formal methods and verification (e.g., `algebraic-repair-calculus`, `cross-lang-verifier`, `tensor-train-modelcheck`, `tensorguard`) operate on general program properties or domain-agnostic abstractions; none incorporate the geometric mechanics (Lie symmetry analysis, backward error analysis, Noether's theorem) that defines ConservationLint's contribution.

### Estimated Subsystem Breakdown (~90K LoC, Phased Delivery)

**Phase 1 — Validate (~20K LoC).** Prove extraction works on pure-NumPy integrators (<500 LoC input programs), prove T2 on paper for 2–3 concrete examples, measure actual liftable-fragment coverage on 5 real codebases, and build a minimal end-to-end pipeline (extraction → symmetry → BCH → localization) on Verlet/leapfrog integrators.

**Phase 2 — Build (~70K incremental LoC, conditional on Phase 1 success).** Full static analysis pipeline, complete benchmark suite, full evaluation against all baselines.

| # | Subsystem | Description | Phase 1 | Phase 2 | Total | Language |
|---|-----------|-------------|---------|---------|-------|----------|
| 1 | Python frontend + conservation-aware IR | Tree-sitter Python frontend, pattern lifting, language-agnostic IR design | ~8K | ~7K | ~15K | Rust |
| 2 | Symbolic algebra engine | Polynomial manipulation, Lie bracket computation | ~5K | ~5K | ~10K | Rust |
| 3 | Variational symmetry analyzer | Restricted Lie-symmetry solver for polynomial systems | ~3K | ~5K | ~8K | Rust |
| 4 | Backward error diagnostic engine | Modified equation computation, BCH expansion, term attribution | ~3K | ~7K | ~10K | Rust |
| 5 | Causal localization engine | Differential symbolic slicing, source-line attribution | — | ~6K | ~6K | Rust |
| 6 | Benchmark suite | 25 simulation kernels with ground-truth conservation properties | ~5K | ~10K | ~15K | Python |
| 7 | Evaluation harness | Automated metrics, comparison infrastructure | — | ~6K | ~6K | Python |
| 8 | CLI + basic reporting | Command-line interface, diagnostic output | ~1K | ~4K | ~5K | Rust |
| 9 | Liftable fragment checker + coverage measurement | Empirical coverage analysis on real codebases | ~2K | ~3K | ~5K | Rust + Python |
| 10 | NumPy/SciPy semantics database | Mathematical models of library functions | ~3K | ~7K | ~10K | Rust |
| | **Total** | | **~20K** | **~70K** | **~90K** | |

---

## Validation Phase

Before building the full pipeline (Phase 2), Phase 1 must validate three critical assumptions:

1. **Extraction feasibility.** Implement the code→math extraction for pure-NumPy Verlet/leapfrog integrators on programs of <500 LoC. Demonstrate faithful IR extraction on at least 5 integrator implementations of varying complexity.

2. **T2 validation on paper.** Prove the obstruction criterion for 2–3 concrete examples:
   - A Verlet integrator with short-range/long-range force splitting (angular momentum obstruction).
   - A symplectic Euler method with thermostat coupling (energy obstruction).
   - A multi-rate leapfrog with different time steps for fast/slow forces.
   If T2 is trivial (reduces to Tarski-Seidenberg with no structural insight) or the criterion is exponential in k or p, the theoretical contribution must be revised—either reframed as an "Obstruction Conjecture" with computational evidence, or the complexity bounds must be tightened.

3. **Coverage measurement.** Implement the liftable fragment checker and measure actual coverage on 5 real codebases (see Coverage section below). If coverage is <15% of kernel lines, reconsider whether the static-analysis approach is viable at all before proceeding to Phase 2.

4. **User demand validation.** Conduct 3–5 structured interviews with simulation developers during Phase 1 to validate that the tool's diagnostic output is useful and that conservation bugs are a real pain point in their workflows.

Phase 2 proceeds only if Phase 1 demonstrates: (a) faithful extraction on the target fragment, (b) a non-trivial and efficient T2 criterion, (c) ≥15% coverage on at least 3 of 5 real codebases, and (d) positive signal from developer interviews.

---

## New Mathematics Required

### T1: Heterogeneous Composition Modified Equation Theorem

**Statement.** Let Φ = φ_k ∘ φ_{k−1} ∘ ⋯ ∘ φ_1 be a one-step map formed by composing k numerical integrators φ_i, where each φ_i is a method of order p_i applied to a sub-Hamiltonian H_i, and the sub-Hamiltonians satisfy H = H_1 + ⋯ + H_k. Then Φ is the exact time-h flow of a *tagged* modified Hamiltonian H̃ = H + Σ_{n≥1} h^n δ_n, where each correction term δ_n is a Lie polynomial in the H_i and carries a provenance tag identifying which subset of {φ_1, …, φ_k} generates it.

**What's known.** Classical backward error analysis (Hairer, Lubich & Wanner, *Geometric Numerical Integration*, 2006, Ch. IX) establishes modified equations for *individual* methods. McLachlan & Quispel (Acta Numerica, 2002) extend this to symmetric compositions of a *single* base method. BCH expansions for operator splitting are known (McLachlan & Quispel 2002; Blanes, Casas & Murua 2008), but without provenance tracking and without analysis of the symmetry-breaking structure of individual terms for *heterogeneous* compositions of *different* methods.

**What's new.** The tagged provenance structure in the *mixed-order* case: when sub-integrators have different orders p_i, the interaction between error terms is non-trivial and not addressed by existing BCH results, which assume homogeneous compositions. The theorem provides explicit leading-order symmetry-breaking bounds for mixed-order compositions, enabling causal localization to trace symmetry-breaking corrections back through the code lifting to source lines. The value of T1 depends on whether the mixed-order generalization yields non-trivial structural insights (e.g., "the leading symmetry-breaking term always originates from the lowest-order sub-integrator"); if the proof reveals such structure, T1 is a genuine mathematical contribution; if it is mechanical BCH bookkeeping with labels, T1 is a useful engineering result but not a standalone theorem.

**How it enables the artifact.** Without provenance-tagged modified equations, ConservationLint could detect *that* a conservation law is broken but not *which part of the code* breaks it. T1 is the mathematical foundation of causal localization—the feature that transforms the tool from a global pass/fail oracle into a precise, actionable diagnostic.

### T2: Computable Obstruction Criterion for Irreparable Conservation Violations

**Statement (strengthened).** Let H = H_1 + ⋯ + H_k be a splitting of a Hamiltonian system into k sub-Hamiltonians, let v be a Lie-symmetry generator of H (i.e., {H, C_v} = 0 for the conserved quantity C_v associated with v via Noether's theorem), and let p ≥ 1 be a target order. Define the *obstruction ideal* O(v, H_1, …, H_k, p) as the ideal generated by the Lie bracket expressions {δ_n, C_v} for 1 ≤ n ≤ p, where δ_n ranges over all achievable correction terms at order n for *any* composition of integrators φ_i of orders ≥ p_i applied to the splitting H_1, …, H_k. Then:

(a) **Decidability.** Whether O(v, H_1, …, H_k, p) contains only zero—equivalently, whether there exists a composition of methods of the given orders that preserves C_v to order p—is decidable via a finite algebraic test involving at most B(k, p) independent Lie bracket conditions, where B(k, p) is bounded by the dimension of the free Lie algebra truncated at depth p on k generators.

(b) **Complexity.** The number of bracket conditions B(k, p) grows as O(k^p / p) (the Witt dimension formula). For typical cases (k ≤ 5, p ≤ 4), B(k, p) ≤ 200. Each condition is a polynomial identity in the coefficients of the sub-Hamiltonians, checkable in polynomial time in the number of terms.

(c) **Truncation limitation.** The criterion is sound at order p: if the obstruction ideal is non-trivial, no composition can preserve C_v to order p. However, the criterion is *not* complete across all orders—a splitting that is unobstructed at order p may still be obstructed at order p+1. The tool reports the order at which the analysis is performed and flags cases where higher-order analysis might change the conclusion.

(d) **Necessity.** When the obstruction is confirmed at order p, no local modification of any single φ_i can restore conservation of C_v to order p; the splitting itself is the obstruction.

**What's known.** It is well known that symplectic integrators cannot exactly preserve both the Hamiltonian and all polynomial first integrals simultaneously (Ge & Marsden 1988; Chartier, Faou & Murua 2006 for the modified equation perspective). Specific non-conservation results exist for individual methods. Decidability of the finite-order question follows in principle from Tarski-Seidenberg (quantifier elimination over the reals), but this gives doubly-exponential complexity that is useless in practice.

**What's new.** The reduction to a *structured* finite test exploiting the Lie-algebraic structure of BCH expansions, yielding a polynomial-time criterion (for fixed k, p) rather than general quantifier elimination. This is the non-trivial mathematical content: not mere decidability (which is known), but an *efficient, structured* decision procedure. If the criterion cannot be made efficient (i.e., polynomial in the sub-Hamiltonian size for fixed k, p), T2 should be reframed as the "Obstruction Conjecture" with computational evidence from the benchmark suite demonstrating the criterion's effectiveness on concrete examples.

**How it enables the artifact.** T2 is what allows ConservationLint to distinguish "this code has a bug you can fix on line 247" from "this algorithm *architecturally cannot* conserve angular momentum with this force splitting—you need to restructure." Without T2, every detected violation would suggest a local fix, many of which would be provably futile. T2 saves developer time and provides deep algorithmic insight.

### Liftable Fragment Characterization (formerly T3)

*Note: This is a formal definition and scope characterization, not a theorem. It is presented as a rigorous contribution to the PL/SE literature on domain-specific program analysis fragments.*

**Definition.** The *liftable fragment* L of imperative numerical code is the largest subset of programs that admit faithful extraction into the conservation-aware IR—specifically, programs whose numerical kernel can be represented as a finite composition of smooth maps on a phase space with polynomial-rational dependence on state variables and parameters. L is characterized syntactically (no recursion in the kernel, no data-dependent control flow over state variables within a time step, array accesses with affine index expressions, no opaque library calls outside the modeled semantics database) and admits a decidable membership test. For programs in L, the extracted IR is semantically faithful: the conserved quantities of the IR system are exactly those of the discrete map implemented by the code, up to floating-point rounding.

**Coverage.** Coverage will be measured empirically on 5 real codebases (JAX-MD, Dedalus, SciPy ODE suite, gray radiation kernel, ASE-based MD). We report honest coverage with a failure taxonomy. Preliminary estimate: 20–40% of kernel lines. The failure taxonomy classifies *why* extraction fails for code outside L: opaque library call (expected to be the dominant category for JAX-MD and Dedalus), data-dependent branching, non-polynomial nonlinearity, complex control flow, or unsupported array access pattern.

**What's known.** Lifting imperative code to mathematical representations has been studied in the verified numerics community (e.g., Boldo et al., 2015) and in abstract interpretation (Cousot & Cousot 1977). The syntactic restrictions resemble the polyhedral model from compiler literature (Feautrier 1992, Bondhugula et al. 2008). No prior work formalizes the specific fragment of numerical code that admits conservation analysis, nor provides a decidable syntactic characterization combined with a semantic faithfulness guarantee for conservation properties.

**How it enables the artifact.** The Liftable Fragment Characterization is intellectual honesty as a formal result. It tells users exactly what ConservationLint can and cannot analyze, prevents false confidence, and provides a formal foundation for extending coverage over time. It also guides the engineering: the fragment definition directly determines the IR design and the Tree-sitter extraction rules.

---

## Best Paper Argument

**The paradigm claim.** ConservationLint establishes *physics-aware program analysis* as a new category of developer tool—one that reasons about domain-specific mathematical semantics, not just types, memory, or general-purpose invariants. Just as type systems brought mathematical rigor to memory safety and abstract interpretation brought lattice theory to program correctness, ConservationLint brings geometric mechanics (Noether's theorem, symplectic geometry, backward error analysis) to numerical code quality. This is the *first bridge between Noether's theorem and program analysis*, and building it requires contributions to both sides: new mathematics (T1, T2) and new program analysis techniques (differential symbolic slicing, conservation-aware IR, liftable fragment characterization).

**Why a committee would select this.** The strongest best-paper candidates combine (a) a genuinely new idea that opens a research direction, (b) rigorous technical execution with nontrivial formal results, (c) practical impact on a real community, and (d) elegant presentation of a surprising connection. ConservationLint delivers all four. The new idea is code→math conservation analysis. The formal results (T1, T2, plus the liftable fragment characterization) are precise and falsifiable. The practical impact targets the numerical methods research community with no existing automated solution. And the core insight—that Noether's theorem can be applied not just to equations but to *the code that implements them*—is surprising and elegant.

**Comparison to landmark papers.** SLAM (Ball & Rajamani, PLDI 2001) brought model checking to device drivers and launched a decade of software verification research by showing that formal methods could target a specific, high-value domain rather than general programs. ConservationLint makes an analogous move: instead of verifying general program properties, it targets the specific, high-value property of conservation-law preservation in the specific domain of scientific simulation. Herbie (Panchekha et al., PLDI 2015) showed that domain-specific numerical reasoning (floating-point accuracy) could be automated into a practical tool; ConservationLint raises the mathematical sophistication from floating-point error bounds to geometric mechanics. Noether's Razor (Cranmer et al., NeurIPS 2024) demonstrated interest in automated conservation analysis but approached it from the data/ML side; ConservationLint provides the complementary *code-analysis* approach with exact, interpretable, GPU-free results.

**The "impossible bridge" narrative.** The deepest reason this project merits attention is that it connects two communities that have never spoken: the geometric numerical integration community (which knows *everything* about conservation laws in numerical methods but works with equations on paper) and the program analysis community (which knows *everything* about extracting properties from code but has never targeted physical invariants). Building the bridge requires fluency in both, and the bridge itself—the conservation-aware IR, the provenance-tagged modified equations, the differential symbolic slicing—is the contribution.

---

## Evaluation Plan

### Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Detection rate (recall)** | Fraction of known conservation violations correctly flagged | ≥ 85% on combined benchmark suite (self-constructed + external) |
| **False positive rate** | Fraction of flagged violations that are spurious | ≤ 10% |
| **Localization accuracy** | IoU between tool-identified violating lines and ground-truth violating lines | ≥ 0.70 mean IoU |
| **Localization granularity** | Precision at function, loop, line, and expression levels | Reported at all four granularities |
| **Coverage** | Fraction of kernel lines in the liftable fragment | Measured empirically; preliminary estimate 20–40% on target codes |
| **Analysis throughput** | Time to analyze a typical kernel (<5K LoC, k≤5, order≤4) | ≤ 10 minutes on laptop CPU |
| **Analysis throughput (extended)** | Time for larger codes or higher-order analysis (>5K LoC or k>5 or order>4) | ≤ 30 minutes on laptop CPU |

### Baselines

1. **Manual inspection (expert).** Conservation violations found by a domain expert reviewing code for 8 hours. Establishes the human ceiling.
2. **Daikon (Perkins et al., 2007).** General-purpose dynamic invariant detection. Run on simulation traces to detect invariant violations. Tests whether domain-agnostic tools can catch conservation bugs.
3. **Noether's Razor (Cranmer et al., NeurIPS 2024).** ML-based conserved quantity discovery from trajectory data. Run CPU-only (if feasible) or with restricted GPU budget. Tests whether the learned approach matches exact symbolic analysis.
4. **Energy-only monitoring.** Track total Hamiltonian at each time step with a fixed threshold. Tests whether naive energy tracking suffices (it doesn't—it misses momentum, angular momentum, and other conservation laws).
5. **LLM baseline (GPT-4/Claude).** Paste each benchmark kernel plus the conservation question ("Does this code conserve energy/momentum/angular momentum? If not, identify the violating lines and the error order.") into GPT-4 and Claude. Report detection rate, localization accuracy, and analysis time. This tests whether general-purpose AI can match formal symbolic analysis. If the LLM matches ConservationLint on >70% of cases, the formal-methods framing requires revision.
6. **SymPy-assisted manual analysis.** Give a graduate student SymPy, the code, and 2 hours per kernel. This represents the current state of practice for research verification.

### Benchmark Suite (25 kernels, committed)

| # | Category | Specific Kernels | Conservation Properties |
|---|----------|-----------------|------------------------|
| 1–5 | JAX-MD | Lennard-Jones, soft sphere, Stillinger-Weber, EAM, Morse | Energy, linear momentum, angular momentum |
| 6–10 | Dedalus | KdV, NLS, Burgers, Rayleigh-Bénard, shallow water | Energy, mass, momentum, enstrophy |
| 11–15 | Hand-written N-body | Leapfrog, Yoshida 4th-order, Ruth 3rd-order, Forest-Ruth, McLachlan 4th-order | Energy, linear momentum, angular momentum |
| 16–20 | SciPy ODE | Hamiltonian systems integrated with RK45, DOP853, Radau, BDF, LSODA | Energy (expected violations for non-symplectic methods) |
| 21–25 | Conservation-violating mutants | Energy leak, momentum symmetry break, angular momentum break, mass non-conservation, symplecticity violation | Known ground truth for all metrics |

**External benchmarks (supplementary).** In addition to the 25 self-constructed kernels, the evaluation includes ≥5 conservation bugs sourced from real simulation repositories: LAMMPS GitHub issues, GROMACS changelogs, and CESM bug tracker. These external bugs are drawn from public issue trackers where conservation violations were identified, reported, and fixed by independent developers. This provides a non-circular evaluation of detection capability on bugs the tool developers did not choose or design.

Each kernel is annotated with ground-truth conservation properties, known violating lines (for mutant and external-bug variants), and expected modified equation leading terms. The full evaluation runs from a single `make evaluate` command with zero human intervention.

---

## LLM Competition and ConservationLint's Defensive Moat

Large language models (GPT-4, Claude, and successors) can already perform conservation analysis on simple-to-medium simulation kernels. Given a textbook leapfrog integrator, an LLM can correctly identify that it is symplectic, conserves a modified Hamiltonian, and may not conserve angular momentum exactly. For medium-complexity cases (e.g., Verlet composed with a thermostat), LLMs provide partially correct heuristic answers. This is a real competitive threat that ConservationLint must address directly.

**Where LLMs fall short.** For large simulation codes (>1K LoC, multiple files, non-obvious splitting structure), LLMs cannot currently: (a) trace conservation violations to specific source lines with mathematical rigor, (b) prove that a violation is architecturally unfixable (T2's obstruction criterion), (c) provide deterministic, reproducible results, or (d) give formal error-order bounds on symmetry-breaking terms. Context window limitations and lack of formal symbolic reasoning prevent rigorous analysis.

**ConservationLint's defensive moat:**

1. **Formal guarantees.** An LLM gives probabilistic answers; ConservationLint gives mathematical proofs (within the liftable fragment). For V&V applications, formal guarantees are non-negotiable.
2. **Determinism.** Same input → same output. LLMs are stochastic; running the same analysis twice may yield different conclusions.
3. **Obstruction proofs.** T2 provides formal proof that a violation is architecturally unfixable—a capability genuinely beyond heuristic reasoning.
4. **Provenance tracking.** T1 provides mathematically rigorous attribution of symmetry-breaking terms to specific sub-integrators and source lines.

**The coverage gap is existential.** If the liftable fragment covers only 10–20% of real code, ConservationLint provides formal guarantees on a small subset while LLMs provide heuristic answers on everything. Most users will choose the heuristic that covers 100% over the proof that covers 10%. ConservationLint's viability therefore depends on achieving sufficient coverage (≥20% of kernel lines) that the formally analyzable fragment includes the most critical code regions. The tool should be positioned as a *formal verification* tool for the analyzable fragment, with explicit acknowledgment that code outside the fragment requires other approaches (including LLM-assisted review).

---

## Laptop CPU Feasibility

**Symbolic computation bounds.** The restricted Lie-symmetry analysis operates on polynomial-rational systems with a *fixed ansatz* (translation, rotation, scaling, Galilean symmetry generators). This reduces the overdetermined PDE system to a structured linear algebra problem of size proportional to (number of variables) × (ansatz dimension), not the doubly-exponential general Gröbner basis computation. For typical simulation kernels (≤50 state variables, ≤6 symmetry generator parameters), the linear system has at most a few thousand rows—solvable in milliseconds.

**Modified equation computation.** The BCH expansion for heterogeneous compositions is a formal power series truncated at a user-specified order (default: order 4, capturing leading symmetry-breaking terms). At each order, the computation involves a fixed number of Lie bracket evaluations on polynomial expressions. For a k-way splitting at order p, this requires O(k^p) bracket evaluations, each polynomial-time in the number of terms. With k ≤ 5 and p ≤ 4 (the typical case for simulation kernels), this completes in seconds.

**Analysis time budget (qualified).** ≤10 minutes for typical kernels (<5K LoC, k≤5, order≤4). For larger codes or higher-order analysis, the budget scales: k=10 at order 6 requires ~17 minutes by Auditor calculation (O(10^6) bracket evaluations). The tool targets ≤30 minutes for these extended cases. The bottleneck is IR extraction for complex kernels with deep call graphs; caching of library-function signatures (NumPy, SciPy primitives) amortizes this across analyses.

**Evaluation automation.** The full benchmark evaluation (25 kernels × 6 baselines × all metrics) runs as an automated script suite. Each kernel analysis is independent and parallelizable. Total evaluation wall time target: <6 hours on a 4-core laptop, <2 hours with 16-core workstation parallelism. No GPU required at any stage (the LLM baseline runs via API calls).

---

## Limitations and Honest Scope

This section addresses known limitations directly, informed by adversarial review.

**Coverage is uncertain and may be low.** The 20–40% preliminary coverage estimate is an honest range, not a guarantee. Opaque library calls (SciPy sparse solvers, JAX transformations, FFTW-based operations in Dedalus) are expected to be the dominant failure mode. JAX code that relies on `jit`/`vmap`/`grad` tracing semantics may be largely outside the liftable fragment, since Tree-sitter parses Python syntax, not traced computation graphs. If empirical coverage on real codebases falls below 15%, the static-analysis approach may need fundamental revision.

**The code→math extraction is the existential risk.** The entire pipeline depends on faithful extraction, and no prototype exists. NumPy broadcasting semantics, in-place mutation, and library call modeling each represent significant engineering challenges. Pattern lifting is inherently brittle: the same mathematical operation can be expressed in many syntactic forms (`for i in range(N): f[i] += ...` vs. `f = np.sum(forces, axis=0)` vs. `np.add.at(f, idx, forces)`).

**Self-constructed benchmarks are necessary but insufficient.** The 25-kernel benchmark suite provides controlled evaluation with known ground truth, but performance on self-constructed benchmarks does not guarantee performance on real-world code. The external benchmarks from LAMMPS/GROMACS/CESM bug trackers partially address this, but a truly convincing evaluation requires held-out bugs from independent sources.

**T2 may be trivial or intractable.** The obstruction criterion's value depends on whether the reduction to a finite algebraic test is efficient (polynomial in sub-Hamiltonian size for fixed k, p) and reveals structural insight. If the criterion reduces to Tarski-Seidenberg quantifier elimination (doubly exponential), it is computable in principle but useless in practice. Phase 1 validates this before Phase 2 investment.

**Cutoff-based force truncation — a common MD conservation-bug source — may be excluded** by the liftable fragment if it involves data-dependent control flow (e.g., `if r < r_cut`). This is a known gap that limits the tool's applicability to molecular dynamics codes.

**Python-only scope limits adoption.** The largest potential user communities (climate modelers, national lab V&V teams) primarily use Fortran and C++. The Python-first scope matches the academic numerical methods research audience but excludes production simulation codes. Fortran/C++ frontends are future work.

---

## Extensions

The following capabilities are valuable but are *not* part of the core contribution. They are listed here as natural extensions that build on the core pipeline.

### Runtime Conservation Monitor
Instrumenting running simulations to track conserved quantities with statistically rigorous drift detection (CUSUM/SPRT) while maintaining <5% overhead. This is standard statistical process control engineering, not a research contribution. It could be built as a separate tool that consumes ConservationLint's conservation property annotations.

### Repair Suggestion Engine
When a violation is detected and localized, suggesting concrete code patches that restore the broken conservation law via a template catalog of known fixes (e.g., symmetric force splitting, symplectic correctors, shadow Hamiltonian adjustments). This is constrained template matching, not program synthesis. Estimated scope: ~6K LoC.

### LSP/IDE Integration
An LSP server that annotates the developer's editor with conservation properties and warnings in real time. This is engineering polish that builds on the CLI diagnostic output. Estimated scope: ~8K LoC (Rust + TypeScript).

### Fortran/C++ Frontends
Language-specific Tree-sitter frontends that parse Fortran and C++ simulation code into the same conservation-aware IR. This would unlock the national-lab and production simulation audience. Estimated scope: ~15–25K LoC per language.

---

## Appendix: Cross-Critique Consensus Scores

From the verification gate team (Auditor, Skeptic, Synthesizer):

| Axis | Score |
|------|-------|
| Extreme and Obvious Value | 5/10 |
| Genuine Difficulty | 6/10 |
| Best-Paper Potential | 5/10 |
| Laptop CPU + No Humans | 7/10 |
| **Composite** | **5.75/10** |

**Verdict: CONDITIONAL CONTINUE.** The core idea (Noether → program analysis bridge) is genuinely novel and worth pursuing. The execution plan has been revised per binding conditions.

# Verification Gate: sim-conservation-auditor (ConservationLint)

## Committee
- Independent Auditor: Evidence-based scoring and challenge testing
- Fail-Fast Skeptic: Aggressively reject under-supported claims
- Scavenging Synthesizer: Salvage value from abandoned proposals
- Cross-critique phase: adversarial challenge + synthesis

## Verdict: CONDITIONAL CONTINUE

Composite: 5.75/10 (V5/D6/BP5/CPU7)

The core idea — bridging Noether's theorem and program analysis to detect, localize, and repair conservation-law violations in simulation code — is genuinely novel and unanimously confirmed as unexplored territory. However, the execution plan is wildly overscoped (162K LoC is a research program, not a paper), the load-bearing assumption (code→math extraction) is entirely unvalidated, and several quantitative claims (40–60% coverage, ≥90% detection) are fabricated or circular. The project merits continued investment only under radical scope reduction and binding feasibility validation.

---

## Pillar Scores

### 1. EXTREME AND OBVIOUS VALUE — 5/10

**The pain is real but narrow, not top-priority for its own community, and faces a rising LLM threat.**

The strongest evidence for real pain is the Wan et al. (2019, JAMES) citation: a conservation-violating coupling scheme persisted for three years in a major climate model, passed all unit tests and code reviews, and corrupted published scientific conclusions. This is compelling and specific. Conservation violations are genuinely insidious — unlike crashes or type errors, they produce plausible-looking but quantitatively wrong results that compound monotonically over simulation time.

However, all three evaluators agree the value proposition is overstated:

- **TAM is narrow.** The realistic user pool is ~2,000–10,000 developers worldwide who actively write and maintain simulation kernels where conservation properties matter at the code level. Many already have domain expertise to catch these bugs. The Auditor notes this is "not a million developers problem; it's a niche expert-tool market."
- **Conservation is not the #1 pain point.** Per surveys (Carver et al., SE-CSE workshops) and practitioner reports, the hierarchy is roughly: (1) performance/scalability, (2) general correctness bugs, (3) portability, (4) reproducibility, (5) numerical stability, then (6) conservation-specific violations. ConservationLint targets a real but lower-priority concern.
- **Existing domain-specific monitors already exist.** The Skeptic documents that GROMACS has had energy conservation monitoring since v4.x, LAMMPS has `thermo_style` with energy/momentum tracking as a first-class feature, and OpenMM validates energy conservation in its test suite. These aren't "expert manual audit" — they're standard engineering practice. They detect *that* conservation is violated (though not *why* or *where*), which partially undercuts the urgency argument.
- **The Skeptic's sharpest counter:** "A simpler intervention (a conservation-test template library for pytest) might capture 80% of the value at 1% of the cost." The Auditor counters that this 80% figure is itself fabricated, and that pytest templates cannot do causal localization, obstruction detection, or provenance-tagged backward error analysis — qualitatively different capabilities.
- **LLM competition is a genuine existential threat.** All three evaluators identify this. A domain expert can already prompt GPT-4/Claude with a leapfrog integrator and get correct analysis of its conservation properties. For simple-to-medium cases, LLMs achieve 60–70% of ConservationLint's stated diagnostic value at 0% engineering cost. ConservationLint's defensible moat is formal guarantees (exact proofs vs. probabilistic hints, deterministic reproducibility, obstruction proofs) — but this moat only protects the analyzable fragment. If the liftable fragment covers only 10–20% of real code, LLMs cover 100% heuristically and most users will choose breadth over depth.
- **Zero validated demand.** The Skeptic notes: no user interviews, no letters of support, no survey data. The demand is assumed from first principles, not measured. The Synthesizer's reframing — leading with numerical methods researchers verifying their own implementations rather than production CI/CD — improves the narrative alignment but doesn't enlarge the market.

**Why 5 and not lower:** The Wan et al. anecdote is genuinely compelling. The causal localization capability (tracing violations to specific source lines) and obstruction detection (proving violations are architecturally unfixable) are qualitatively new capabilities that no existing tool — including LLMs — can provide with formal guarantees. The "physics-aware program analysis" paradigm is genuinely novel and opens a research direction.

**Why 5 and not higher:** Zero validated demand, narrow TAM, not the target community's top priority, and an increasingly capable LLM alternative for the diagnostic (non-formal) use case. The Python-only scope excludes the largest potential users (Fortran/C++ national lab codes). The Skeptic's demand score of 1/5 (2/10) is overly harsh given the real anecdotal evidence, but the Auditor's 5/10 is already generous.

### 2. GENUINE DIFFICULTY AS A SOFTWARE ARTIFACT — 6/10

**Real engineering complexity exists, but the 162K LoC claim is inflated ~40–45%, and the genuinely novel fraction is ~30–35K LoC.**

All three evaluators independently performed subsystem-level LoC audits and converged on nearly identical estimates:

| Subsystem | Claimed LoC | Auditor Est. | Synthesizer Est. | Consensus |
|-----------|-------------|--------------|------------------|-----------|
| Python frontend + IR | 25K | 15K | 15K | ~15K |
| Symbolic algebra engine | 20K | 8K | 10K | ~9K |
| Variational symmetry analyzer | 18K | 8K | 8K | ~8K |
| Backward error engine | 18K | 15K | 10K | ~12K |
| Runtime monitor | 12K | 8K | 8K | ~8K |
| Causal localization | 15K | 10K | 6K | ~8K |
| Repair engine | 12K | 6K | 6K | ~6K |
| Benchmark suite | 20K | 10K | 15K | ~12K |
| Evaluation harness | 10K | 6K | 6K | ~6K |
| CLI/LSP/reporting | 12K | 8K | 8K | ~8K |
| **Total** | **162K** | **~94K** | **~92K** | **~92K** |

**Honest estimate: ~90–95K LoC of real code, of which ~30–35K is genuinely novel engineering.** The 162K figure is inflated by roughly 40–45%, likely by counting generous margins, boilerplate, and test fixtures as complexity. The Skeptic notes this is "more code than the entire Rust compiler at v1.0" and recommends 15–25K for a paper-scope proof-of-concept.

**Which subproblems are genuinely hard (3 of 7):**

1. **Code→math extraction (Subsystem 1):** All three identify this as the project's load-bearing assumption and its weakest link. Handling NumPy broadcasting semantics, opaque library calls (SciPy, JAX), pattern-lifting brittleness, and JAX's trace-based computation model is genuinely difficult. The Skeptic documents specific fatal difficulties: NumPy broadcasting requires shape information unavailable statically; JAX works by tracing Python code to build jaxprs, which Tree-sitter cannot parse; modeling hundreds of NumPy/SciPy/JAX functions' mathematical semantics is a person-year of work not accounted for in the LoC estimate.
2. **Obstruction detection / T2 (Subsystem 5 in the math, not a standalone subsystem):** The claim that conservation-achievability reduces to a finite algebraic test is strong and genuinely novel. The proof will require significant restrictions (fixed order, fixed splitting structure, specific symmetry classes) to go through.
3. **Causal localization (Subsystem 6):** "Differential symbolic slicing" — where the slice criterion is defined by contribution to a specific term in a formal power series — is a novel form of program slicing. The concept is genuinely new; the implementation is graph traversal given T1's provenance tags.

**Which subproblems are well-understood (4 of 7):**

- **Lie-symmetry analysis (Subsystem 3):** With the restricted ansatz (translations, rotations, scaling, Galilean generators), this reduces to structured linear algebra. The problem statement itself admits this is "solvable in milliseconds." A problem solvable in milliseconds is not an engineering breakthrough.
- **Backward error computation (Subsystem 4):** BCH expansions are textbook material (Hairer, Lubich & Wanner 2006). The provenance tagging is the novelty (T1), not the computation itself.
- **Runtime monitoring (Subsystem 5):** CUSUM/SPRT on conserved quantities is standard statistical process control.
- **Repair synthesis (Subsystem 7):** Template matching against known repair patterns (symmetrize splits, use corrector steps, Yoshida composition), not program synthesis in the PL sense.

**Integration risk (Skeptic's cross-critique challenge §4.3):** The Auditor's subsystem-by-subsystem audit misses the integration risk. The IR produced by Subsystem 1 must be consumable by Subsystems 2–4, which requires perfect semantic fidelity. A single abstraction mismatch (e.g., the IR can't represent a force splitting the BCH engine needs to analyze) kills the entire pipeline. This is the hardest part of the project — not any individual subsystem, but making them work together end-to-end.

**The symbolic algebra engine (20K claimed LoC) is the most inflated subsystem.** The Auditor notes that the `polynomial` crate in Rust is ~3K LoC for multivariate polynomials. Adding Lie bracket computation and rational functions might double that. 8–10K is realistic, not 20K. The performance argument for a custom Rust engine over SymPy is valid, but the scope estimate is ~2.5× what's needed.

### 3. BEST-PAPER POTENTIAL — 5/10

**One genuine theorem (T2), one conditional theorem (T1), one definition (T3). Realistic landing: OOPSLA/CAV acceptance, not best paper.**

**T1 (Heterogeneous Composition Modified Equation Theorem) — Conditional value:**

The sharpest disagreement across evaluators concerns T1. The Auditor and Skeptic call it "bookkeeping on known expansions" (4/10 novelty): the BCH expansion is textbook (Blanes, Casas & Murua 2008), and labeling which terms come from which operators is implicit in the expansion. The Synthesizer identifies it as "Crown Jewel #2" with high survivability as a standalone theory paper. The resolution: the Auditor/Skeptic are correct for *homogeneous* compositions (all sub-integrators same order), where tagging is indeed implicit. The Synthesizer is correct that the *mixed-order* (heterogeneous) case is under-explored. **T1's value depends entirely on whether the mixed-order generalization yields non-trivial structural insights** (e.g., "the leading symmetry-breaking term always comes from the lowest-order sub-integrator"). If the proof is mechanical BCH bookkeeping with labels, it's a notation. If it reveals unexpected structure, it's a genuine contribution.

**T2 (Computable Obstruction Criterion) — Crown jewel:**

All three evaluators agree T2 is the strongest mathematical claim. The reduction of "can any integrator of a given order with this splitting preserve this symmetry?" to a finite algebraic test is genuinely interesting. However, critical concerns remain:
- **"Computable" hides truncation.** The criterion works "at a given order" — it answers "can conservation be achieved to order p?" not "can conservation be achieved exactly?" It is possible that conservation is obstructed at order 4 but achievable at order 6 via cancellation.
- **Possibly trivial.** The Skeptic notes that finite-order obstruction is decidable via Tarski-Seidenberg (quantifier elimination over polynomial systems) by default. The question is whether T2 reduces to a *practical* computation (polynomial in k and p) rather than the doubly-exponential general decision procedure. An exponential criterion that's "computable in principle" but useless in practice would be a hollow result.
- **Must be proved on paper first.** All evaluators recommend proving T2 for 2–3 concrete examples before building the full tool. If T2 turns out to be trivial or exponential, the paper's theoretical contribution evaporates.

**T3 (Liftable Fragment Formalization) — Definition, not theorem:**

Unanimous across all three evaluators: T3 is a definition dressed as a theorem. Every static analysis paper defines the fragment of programs it handles. The syntactic characterization (no recursion, no data-dependent control flow, affine array indices) is the polyhedral model from compiler literature (Feautrier 1992, Bondhugula et al. 2008). "Decidable membership" is trivially achieved by checking the AST. The 40–60% coverage claim — the actual substantive content — is unverified. T3 is good engineering practice (honest scope statement) but not a mathematical contribution.

**Venue assessment:**
- **PLDI:** Possible but a stretch. PLDI wants formal semantics contributions; T1–T3 are applied math, not PL theory. The program analysis novelty (differential symbolic slicing) is the closest fit.
- **OOPSLA:** Most realistic. Accepts "domain-specific tools with evaluation" papers. Requires evaluation on real-world codes, not just curated benchmarks.
- **CAV:** If T2's formal treatment is rigorous enough. CAV expects very precise formal results.
- **SC (Supercomputing):** Most natural domain venue, but lacks "best paper" prestige.
- **Realistic landing:** Solid OOPSLA or CAV acceptance, not best-paper.

**The "killer result"** would be: "ConservationLint automatically rediscovered the conservation-violating coupling scheme in a real climate model that took human experts 3 years to find (Wan et al. 2019), localized it to the exact code region, and suggested the published fix." If that demo works on actual code (or a faithful reproduction), it's a showstopper. But it requires end-to-end pipeline success on non-trivial code — the highest possible bar.

**The SLAM/Herbie comparisons are aspirational, not structural.** SLAM succeeded because device drivers have a small, well-defined API surface. Herbie succeeded because floating-point expressions are syntactically simple with a scalar accuracy metric. ConservationLint's domain — arbitrary simulation codes with heterogeneous integrators — has neither. The structural analogy breaks down. SLAM shipped with Windows and verified millions of lines; Herbie is used by thousands. ConservationLint is a markdown file.

### 4. LAPTOP CPU + NO HUMANS — 7/10

**The core symbolic computations are genuinely laptop-tractable for the restricted problem. Feasibility risks are in IR extraction time and BCH at high order with many splittings.**

**Lie-symmetry analysis tractability:** True, but only because the ansatz is restricted. For translations, rotations, scaling, and Galilean generators on polynomial-rational systems, the determining equations reduce to structured linear algebra. For typical kernels (≤50 state variables, ≤6 symmetry parameters), the linear system has at most a few thousand rows — solvable in milliseconds. This is well-known in the Lie symmetry literature (e.g., Hereman's `InvariantsSymmetries` package uses similar restrictions). For symmetries outside the ansatz (hidden, conformal, gauge symmetries), this reduction fails — ConservationLint can only verify *expected* conservation laws, not discover *unexpected* ones.

**BCH tractability (Auditor's order 6/k=10 calculation):**
- Order 4, k=10: ~10,000 bracket evaluations. Each on polynomial expressions. **Tractable in seconds.**
- Order 6, k=10: O(k⁶) = ~1,000,000 bracket evaluations. At ~1ms each (generous), that's ~1,000 seconds ≈ 17 minutes. **This exceeds the 10-minute budget.**
- Order 4, k=5 (more typical): O(625) brackets. **Easily feasible.**

The 10-minute claim holds for typical cases (k≤5, p≤4) but fails for extreme parameters (k=10, p=6). The claim should be qualified as "for typical kernels under 5K LoC with ≤5-way splitting at order ≤4."

**Noether's Razor baseline validity:** Noether's Razor (Cranmer et al., NeurIPS 2024) requires training a neural network on trajectory data. Running CPU-only is technically possible but extremely slow (hours per kernel, not minutes). All three evaluators agree this baseline is likely invalid for a like-for-like CPU-only comparison. The evaluation should either commit to a GPU budget for this one baseline, drop it, or substitute a simpler trajectory-based method (e.g., SINDy-based conservation discovery).

**IR extraction is the real bottleneck.** Not computationally (Tree-sitter parsing is fast) but in engineering completeness. Every NumPy function, every SciPy routine, every array operation pattern needs a lifting rule. For a 10K LoC kernel with deep call graphs, library calls to NumPy/SciPy, and multiple levels of abstraction (classes, closures, generators), the bottleneck is resolving all these abstractions. The "caching of library-function signatures" claim requires hand-writing mathematical models for hundreds of functions — a massive effort the LoC estimate doesn't account for. A 10-minute budget is optimistic for 10K LoC with deep call graphs; 30 minutes is more realistic.

**Why 7 and not higher:** BCH order 6/k=10 exceeds the stated budget. IR extraction time is unbounded for complex codebases. The Noether's Razor baseline is invalid CPU-only. **Why 7 and not lower:** The entire analysis pipeline is symbolic/algebraic with no GPU needed anywhere. The core computations (symmetry analysis, BCH at practical parameters, localization) are genuinely fast. The "laptop CPU" constraint is a natural strength of this design.

### 5. FATAL FLAWS

1. **Code→math extraction on real code is unvalidated (CRITICAL).** All three evaluators identify this as the existential risk. The entire pipeline depends on "lifting" imperative code into a mathematical IR. No prototype exists. Real simulation codes don't look like the benchmark suite: production codes at national labs are Fortran/C++ with MPI and hardware-specific optimizations; even within Python, JAX works by tracing to jaxprs (not parseable by Tree-sitter), NumPy broadcasting requires runtime shape information unavailable statically, and SciPy sparse operations are opaque library calls. The effort to model NumPy/SciPy/JAX function semantics is a person-year of work not accounted for.

2. **40–60% coverage claim is fabricated (CRITICAL).** All three evaluators: no measurement has been performed, no code exists, the liftable fragment hasn't been formally defined yet (T3 is stated as a theorem to be proved). The cited codebases (Dedalus, JAX-MD) use opaque library calls extensively. Dedalus is built on FFTW, MPI, and dense linear algebra — all opaque to Tree-sitter. JAX-MD uses JAX transformations (jit, vmap, grad) that are fundamentally non-trivial to lift. The Skeptic's language: "This number is fabricated." Honest estimate for real production codes: 10–20%, and even that would be an achievement.

3. **Self-constructed benchmarks produce circular evaluation (SERIOUS→CRITICAL).** "20+ simulation kernels" written by the tool's developers, annotated by the developers, evaluated by the developers. The benchmarks will inevitably be written in the liftable fragment, the injected bugs will be the kinds the tool is designed to find, and all quantitative claims (≥90% detection, ≤10% FP, 60% repair) will be inflated relative to real-world performance. Combined with Flaw #2, this is near-CRITICAL: every number in the paper is circular if both the benchmarks and the coverage claim are self-serving.

4. **Cutoff-based force truncation — the most common MD conservation-bug source — is excluded by the liftable fragment (SERIOUS).** Data-dependent control flow over state variables (e.g., `if particle_distance < cutoff:`) is explicitly excluded from the liftable fragment. But cutoff-based force truncation is ubiquitous in molecular dynamics codes and is itself a major source of conservation violations. The tool cannot analyze the very pattern that causes the most conservation bugs in practice in its primary target domain.

5. **162K LoC scope is a research program, not a paper (SERIOUS).** This is approximately the scope of a 5-person team working 2–3 years. It is completely incompatible with a single best-paper submission. All three evaluators independently recommend cutting to 15–72K LoC depending on phase. The Skeptic notes this suggests the authors "don't know what's essential."

6. **Zero validated user demand (MODERATE-SERIOUS).** No user interviews, no letters of support, no survey data. The V&V community has its own tools and processes and may not want an external tool imposing a new workflow. The Skeptic scores demand 1/5; the Auditor and Synthesizer are more generous but acknowledge the gap. Fortran/C++ support — required to reach the largest potential users — is explicitly a stretch goal.

---

## Binding Conditions (8 total, 4 non-negotiable)

All conditions must be met before the project advances beyond Phase 1.

| # | Condition | Source(s) | Priority | Description |
|---|-----------|-----------|----------|-------------|
| **BC1** | Radical scope reduction to two phases | All three | **NON-NEGOTIABLE** | Phase 1 (~20K LoC): extraction + symmetry + BCH + localization on pure-NumPy Verlet/leapfrog integrators (<500 LoC input programs). Phase 2 (~70K LoC): full static pipeline with 15+ benchmarks and complete evaluation, conditional on Phase 1 success. No runtime monitor, no repair engine, no LSP/IDE integration in either phase. |
| **BC2** | Honest coverage measurement | Skeptic (C2), Synthesizer (A5) | **NON-NEGOTIABLE** | Implement the liftable fragment checker on 5 real codebases (JAX-MD, Dedalus, a SciPy ODE suite, a gray radiation kernel, an ASE-based MD simulation). Report actual coverage with failure taxonomy (opaque library call: X%, data-dependent branching: Y%, non-polynomial nonlinearity: Z%). If coverage <15%, reconsider whether the static-analysis approach is viable at all. |
| **BC3** | External benchmarks | Skeptic (C3), Auditor (Flaw 2) | **NON-NEGOTIABLE** | Include ≥5 conservation bugs sourced from real simulation code repositories (LAMMPS GitHub issues, GROMACS changelogs, CESM bug tracker). No self-constructed-only evaluation. Ideally include adversarial benchmarks from a geometric numerical integration expert writing code designed to defeat the tool. |
| **BC4** | Validate T2 on paper before building | Skeptic (C5), Auditor (§3 T2) | **NON-NEGOTIABLE** | Prove the obstruction criterion for 2–3 concrete examples (e.g., Strang splitting of rotationally symmetric Hamiltonian, Verlet with thermostat, mixed-order IMEX scheme). If T2 is trivial (just Tarski-Seidenberg with no structural insight) or the criterion is exponential in k or p, revise the theoretical contribution. State all quantifiers, complexity bounds, and truncation limitations precisely. |
| **BC5** | LLM baseline | Skeptic (C4), Auditor (§1) | **STRONGLY REC.** | Include GPT-4/Claude as a baseline in the evaluation: paste each benchmark kernel + conservation question into the LLM. Report detection rate, localization accuracy, and analysis time. If the LLM matches ConservationLint on >70% of cases, the formal-methods framing needs revision. |
| **BC6** | Reframe demand narrative | Synthesizer (A1) | **STRONGLY REC.** | Lead with the numerical methods research verification use case ("when a researcher implements a new structure-preserving integrator, ConservationLint automatically verifies the implementation preserves claimed conservation laws"). Reposition national-lab CI/CD vision as a future direction enabled by language-agnostic IR. Match the narrative to the Python scope and the realistic early-adopter pool (academic computational math departments). |
| **BC7** | User demand validation | Skeptic (C6) | **RECOMMENDED** | Talk to 3–5 simulation developers. Show them a mock-up of ConservationLint output. Document: Would they use this? What conservation bugs have they encountered? What would they pay? If nobody cares, stop. |
| **BC8** | Drop repair synthesis; deepen localization evaluation | Synthesizer (A3), Auditor (Flaw 4) | **RECOMMENDED** | Replace the repair engine (weakest component — template matching, not synthesis) with a thorough evaluation of localization accuracy at function/loop/line/expression granularity. Compare against random baseline, Daikon-identified invariant violations, and manual expert localization. This trades a weak component for a stronger evaluation of a strong component. |

## Amendments (10 unanimous + 6 majority + 3 contested)

### Unanimous (all three evaluators agree)

| # | Amendment | Rationale |
|---|-----------|-----------|
| **U1** | Cut 162K scope by ≥50%. Phase 1: ~20K LoC, single domain. | All three independently conclude the scope is a research program, not a project. |
| **U2** | Drop runtime monitor from core contribution. | All three: standard engineering (CUSUM/SPRT), not novel. |
| **U3** | Drop repair synthesis from core contribution. | All three: template matching, not synthesis; weakest component. |
| **U4** | Drop LSP/IDE integration from core contribution. | All three: engineering polish, not research. CLI suffices. |
| **U5** | Measure liftable fragment coverage empirically; report honestly. | All three flag 40–60% as unverified. Must include failure taxonomy. |
| **U6** | Include external benchmarks from real bug trackers. | All three flag self-constructed evaluation as circular. LAMMPS issues, GROMACS changelogs, CESM bug tracker as sources. |
| **U7** | State T3 as a formal definition / scope characterization, not a "theorem." | All three: it's a definition. Good engineering practice, not a mathematical contribution. |
| **U8** | Strengthen or precisely state T2 with all quantifiers, complexity bounds, and truncation limitations. | All three: T2 is the crown jewel but dangerously underspecified. Must address: (a) what "given order" means precisely, (b) whether the criterion is necessary and sufficient or only sufficient, (c) computational complexity. |
| **U9** | Address LLM competition explicitly in the paper. | All three identify this as an unaddressed existential threat. Frame ConservationLint as formal verification for the analyzable fragment, with LLMs as the fallback for everything else. |
| **U10** | Qualify the 10-minute budget to "typical kernels <5K LoC, k≤5, order≤4." | Auditor: order 6/k=10 ≈ 17 min exceeds budget. Skeptic: worst-case unknown. Synthesizer: acknowledges. |

### Majority (two of three agree)

| # | Amendment | Supporters | Dissent |
|---|-----------|------------|---------|
| **M1** | Include LLM (GPT-4/Claude) as an evaluation baseline. | Skeptic, Auditor | Synthesizer doesn't oppose but recommends SymPy-manual baseline instead; both can coexist. |
| **M2** | Reframe demand around research tooling, not production CI/CD. | Synthesizer, Auditor | Skeptic would go further: validate demand exists at all before proceeding. |
| **M3** | Commit to a specific list of benchmark kernels (25 named kernels). | Synthesizer, Skeptic | Auditor doesn't propose specific list but agrees benchmarks need tightening. |
| **M4** | Validate T2 on paper (2–3 examples) before any tool building. | Skeptic, Auditor | Synthesizer offers alternative: downgrade to "Obstruction Conjecture" with computational evidence. |
| **M5** | Factor out shared infrastructure with fp-error-audit-engine (Penumbra). | Synthesizer | Auditor/Skeptic don't discuss; not opposed, just out of scope of their review. Shared Tree-sitter parsing, NumPy/SciPy semantics database, and source-line attribution could save ~10–15K LoC. |
| **M6** | Conduct user demand validation (3–5 developer interviews). | Skeptic, Synthesizer (implicitly) | Auditor recommends focusing on delivery, not market research. |

### Contested (one evaluator proposes, others don't address or disagree)

| # | Amendment | Proposer | Status |
|---|-----------|----------|--------|
| **C1** | T1 (provenance-tagged BCH) is a crown jewel and should lead the paper. | Synthesizer | Contested — Auditor/Skeptic consider it bookkeeping. Resolution depends on whether the mixed-order proof yields non-trivial structural insights. |
| **C2** | Add abstract interpretation baseline (Fluctuat-style interval analysis). | Synthesizer | Not opposed by others; low cost to include. |
| **C3** | Publish the benchmark suite as a standalone JOSS paper regardless of tool outcome. | Synthesizer | Not addressed by others. A hedge strategy with high salvage value. |

## Scope Reduction

The three evaluators' scope recommendations are complementary, not contradictory. They map to a two-phase approach:

### Phase 1: Validate (~20K LoC, aligns with Skeptic's recommendation)

**Purpose:** Prove the extraction works, prove T2 on paper, measure actual coverage. Kill the project early if the load-bearing assumptions fail.

- Implement code→math extraction for pure-NumPy Verlet/leapfrog integrators (<500 LoC input programs)
- Restricted Lie-symmetry analysis on extracted IR
- BCH expansion with provenance tagging (T1) at order 4
- Prove T2 (obstruction criterion) for 2–3 concrete examples on paper
- Measure liftable fragment coverage on 5 real codebases
- Build 5–10 small benchmark kernels + source ≥5 external bugs
- **Gate decision:** If coverage <15% on real codes, or T2 is trivial/exponential, ABANDON or pivot to salvage options.

### Phase 2: Build for Publication (~70K LoC, aligns with Synthesizer's recommendation)

**Purpose:** Full static analysis pipeline for a publishable system paper. Only proceed if Phase 1 succeeds.

- Full Python frontend + conservation-aware IR (15K)
- Complete symbolic algebra engine (10K)
- Variational symmetry analyzer (8K)
- Backward error engine with provenance (10K)
- Causal localization engine (6K)
- Benchmark suite of 15–25 kernels with external bugs (12K)
- Evaluation harness with all baselines including LLM (6K)
- CLI + basic reporting (5K)
- **No:** runtime monitor, repair engine, LSP server, Fortran/C++ frontends

## Salvage Value (if abandoned)

Ranked by independent value, endorsed by cross-critique:

1. **Conservation-Annotated Benchmark Suite (HIGH).** No standard benchmark for conservation correctness in simulation code exists. A curated set of 20+ kernels with ground-truth conservation properties and injected-bug variants would be a community resource cited for a decade. Publishable at JOSS, SC, or SciPy conference. ~15K LoC, 2–3 months.

2. **Conservation-Aware IR + Code→Math Extraction (HIGH).** Even without the Noether pipeline, the IR is reusable infrastructure for any future physics-aware program analysis tool: automated backward error analysis, numerical method identification, refactoring equivalence checking. Publishable as "A Conservation-Aware IR for Numerical Simulation Code." ~25K LoC, 4–5 months.

3. **Survey/Systematization Paper (MEDIUM-HIGH).** A SoK paper on "Conservation in Numerical Simulation Code: Theory, Practice, and Tooling Gaps" mapping the space between geometric numerical integration, software verification, and scientific computing practice. Minimal code, 2–3 months writing. Establishes vocabulary and positions future work.

4. **ConservationLint-Lite (MEDIUM).** Single-method integrators only (no composition theory, no T1). Detects violations in simple leapfrog/Verlet implementations. Covers teaching and research codes but loses the composition theory and causal localization across sub-integrators — precisely what makes the full system novel. May be too incremental for a top venue. ~35K LoC.

5. **Runtime Conservation Monitor (MEDIUM-LOW).** Standalone runtime monitor with CUSUM/SPRT drift detection. Useful but not novel — a well-engineered version of what many simulation codes already do ad hoc. Not publishable as standalone research. Suitable for JOSS or as an open-source library.

6. **Theorems T1–T3 as Pure Math Paper (LOW-MEDIUM).** T1 is publishable in a numerical methods journal (BIT, Numerische Mathematik) as a modest extension of known BCH/modified equation theory. T2, if proved, would be a nice result. T3 is uninteresting to the pure math community. The theorems derive most of their value from being connected to a tool; without the tool, they're incremental.

## Risk Assessment

- **P(top-venue acceptance):** 35% — Realistic at OOPSLA or CAV if Phase 1 succeeds, coverage is ≥25%, T2 is non-trivial, and evaluation includes external benchmarks and LLM baseline. The "physics-aware program analysis" narrative is novel enough to clear the bar. Probability drops to ~10% if coverage is <20% or T2 is trivial.
- **P(best-paper):** 4% — Requires the "killer demo" (rediscovering the Wan et al. climate model bug on real code), a surprisingly elegant T2 proof, and a reviewer committee that values domain-specific formal methods over broadly applicable PL work. Aspirational.
- **P(ABANDON after Phase 1):** 40% — The existential risk is code→math extraction. If the liftable fragment covers <15% of real code, or T2 is trivial (just Tarski-Seidenberg) or exponential, the project should be abandoned or pivoted to salvage options. The 40% reflects honest uncertainty about whether extraction can handle real code at all.
- **P(any publication from this work):** 75% — Even if the full tool fails, the benchmark suite, IR design, survey paper, or theorems T1–T2 are independently publishable. Multiple salvage paths with genuine community value.

---

*Verification gate report. 3-expert team + adversarial cross-critique. 2025-07-18.*

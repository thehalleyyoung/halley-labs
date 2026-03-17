# ConservationLint: Final Approach — Hybrid Static-Dynamic Conservation Auditor

## 1. Approach Summary

ConservationLint is the first tool to bridge Noether's theorem and program analysis, automatically detecting and localizing conservation-law violations in Python simulation code. The core insight—that physical conservation structure encoded implicitly in imperative code can be mechanically recovered and checked—defines a new category of developer tool: *physics-aware program analysis*.

The winning approach synthesizes the strongest elements from three competing designs, informed by a brutal adversarial debate that exposed fatal flaws in each. Pure static analysis (Approach A) cannot parse real frameworks—JAX traces to `jaxpr`, Dedalus uses FFT-opaque spectral methods, NumPy broadcasting requires runtime shapes. Pure dynamic analysis (Approach C) scales detection to any system but cannot localize beyond ~50 particles or prove violations are architecturally unfixable. A typed API (Approach B) requires adoption that will never materialize.

The synthesis: **hybrid extraction** using traced computation graphs (jaxpr, NumPy dispatch) for real frameworks plus Tree-sitter for pure-NumPy code, feeding a **two-tier analysis pipeline**—formal symbolic analysis where extraction succeeds, dynamic ablation-based localization everywhere else. This dramatically expands the liftable fragment beyond what Tree-sitter alone achieves, while providing graceful degradation to statistical guarantees on code that resists any extraction. The headline mathematical contribution is the **computable obstruction criterion (T2)**, which determines whether a conservation violation is locally repairable or architecturally unfixable—a capability no existing tool, LLM, or human heuristic provides. The provenance-tagged modified equation (T1) is presented honestly as engineering scaffolding, not a deep theorem. The Lie algebra → graded effects connection (from Approach B) informs the IR design but is not load-bearing for the tool.

The tool targets ~300–500 users: numerical methods researchers verifying structure-preserving integrators, academic simulation developers using Python frameworks (JAX-MD, Dedalus, SciPy), and course instructors teaching geometric integration. This is a research contribution, not a product. The paper's audience is PL researchers and geometric integration researchers. The value is the paradigm, not the user count.

**Target venue: OOPSLA.** The "impossible bridge" narrative—connecting geometric numerical integration (which knows everything about conservation but works on paper) with program analysis (which knows everything about code but has never targeted physical invariants)—fits OOPSLA's appetite for domain-specific tools with formal results and strong evaluation.

## 2. Architecture

The system is a two-tier pipeline with a shared extraction frontend:

```
                        ┌──────────────────────────────┐
                        │     Python Simulation Code    │
                        └──────────────┬───────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │       HYBRID EXTRACTION LAYER       │
                    │                                     │
                    │  ┌─────────┐   ┌────────────────┐  │
                    │  │jaxpr    │   │ Tree-sitter +   │  │
                    │  │intercept│   │ NumPy dispatch  │  │
                    │  └────┬────┘   └───────┬────────┘  │
                    │       └───────┬────────┘           │
                    │               ▼                     │
                    │    Conservation-Aware IR             │
                    │    (polynomial-rational maps on     │
                    │     phase space + provenance tags)  │
                    └──────────┬────────────┬─────────────┘
                               │            │
              ┌────────────────┘            └────────────────┐
              ▼                                              ▼
   ┌──────────────────────┐                   ┌──────────────────────┐
   │   TIER 1: STATIC     │                   │   TIER 2: DYNAMIC    │
   │   FORMAL ANALYSIS    │                   │   ABLATION ANALYSIS  │
   │                      │                   │                      │
   │ Lie-symmetry analysis│                   │ Conservation-specific│
   │ → BCH expansion with │                   │ violation detection  │
   │   provenance tags    │                   │ (d/dt C_v along      │
   │ → Obstruction check  │                   │  trajectories)       │
   │   (T2)               │                   │ → Ablation-based     │
   │ → Differential       │                   │   localization       │
   │   symbolic slicing   │                   │   (toggle + measure) │
   │                      │                   │                      │
   │ Output: formal proof │                   │ Output: statistical  │
   │ + source-line attrib │                   │ violation + causal   │
   │ + obstruction cert   │                   │ code-region attrib   │
   └──────────────────────┘                   └──────────────────────┘
              │                                              │
              └────────────────┬─────────────────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │  UNIFIED DIAGNOSTIC  │
                    │  CLI / CI Report     │
                    │  (SARIF / JSON)      │
                    └──────────────────────┘
```

**Tier selection is automatic.** If hybrid extraction succeeds for a code region, Tier 1 runs. If extraction fails (opaque library, unsupported pattern), the region is marked as Tier 2 and analyzed dynamically. The user sees a unified report indicating which results carry formal guarantees and which carry statistical confidence intervals.

## 3. Extreme Value and Target Audience

**Who needs this:** Numerical methods researchers implementing structure-preserving integrators in Python. A PhD student publishes a "symplectic, energy-conserving" integrator for molecular dynamics. Three years later, someone discovers it silently leaks angular momentum due to an asymmetric force splitting—the exact scenario from Wan et al. (2019, JAMES). ConservationLint catches this on the first commit.

**Why urgently:** Conservation violations are silent killers. Unlike crashes or type errors, they produce plausible-looking but quantitatively wrong results. Energy created from nothing doesn't trigger an exception; angular momentum leaking doesn't segfault. These errors compound monotonically, are invisible to standard testing, and corrupt scientific conclusions. Today's defense is expert manual audit. ConservationLint automates it.

**What becomes possible:** (1) Automated conservation CI for Python simulation codes—every PR checked for conservation regressions with source-line attribution. (2) Obstruction certificates—formal proof that certain violations are architecturally unfixable, saving developers from futile debugging. (3) A benchmark-driven culture of conservation correctness, analogous to how AddressSanitizer transformed memory safety practices.

**Honest scope:** ~300–500 potential users. This is a research contribution targeting PL researchers and geometric integration researchers. The tool is the evaluation section of a paradigm-establishing paper.

## 4. Technical Approach

### 4.1 Hybrid Extraction (the key innovation)

The Skeptic's most devastating critique of pure static analysis: Tree-sitter cannot see through JAX tracing or FFT-based spectral methods. The rescue: **intercept at the framework level**.

**Traced execution path (primary).** For JAX code, intercept at the `jaxpr` level—JAX exposes its traced computation graph as a structured IR. For NumPy code, use operation dispatch interception (`__array_function__` protocol) to capture mathematical operations during a single trace execution with representative inputs. Benefits: (a) JAX-MD coverage immediately, (b) shapes known at trace time, (c) only ~50 primitives to model vs. 300+ for Tree-sitter, (d) IR faithful by construction.

**Static parsing path (supplementary).** Tree-sitter for pure-NumPy code with explicit loops—textbook integrators, research prototypes. No execution required.

**The cost:** One execution with representative inputs to produce a trace. Subsequent analysis (symmetry, BCH, obstruction) remains fully static and formal.

**Coverage expansion:** Tree-sitter alone: ~15–25%. Hybrid: ~40–60% (to be measured). Gaps: FFTW-opaque spectral methods (Dedalus core), MPI, custom C extensions → Tier 2.

### 4.2 Conservation Analysis Pipeline

For code in the liftable fragment (Tier 1):

1. **Symmetry identification.** Restricted Lie-symmetry analysis on the extracted IR with a fixed ansatz (translations, rotations, scaling, Galilean generators). This reduces overdetermined PDE systems to structured linear algebra solvable in milliseconds for ≤50 state variables.

2. **Provenance-tagged BCH expansion.** Compute the modified Hamiltonian H̃ = H + Σ hⁿ δₙ via Baker-Campbell-Hausdorff expansion truncated at order p (default: 4). Each correction term δₙ carries a bitset tag identifying which sub-integrators generate it. This is engineering bookkeeping—not deep math—but it is the prerequisite for causal localization.

3. **Conservation check.** For each symmetry generator v and corresponding conserved quantity C_v, compute {δₙ, C_v} (Poisson bracket). Non-vanishing terms identify symmetry-breaking corrections at order n.

4. **Causal localization.** Trace symmetry-breaking terms through provenance tags back to specific sub-integrators, then through the IR-to-source mapping to specific source lines. For textbook integrators with 3–5 splitting stages, this gives line-level attribution.

### 4.3 Two-Tier Localization

**Tier 1 (static formal localization):** Differential symbolic slicing—a novel form of program slicing where the criterion is "contribution to order-n term in a formal power series." Given T1's provenance tags, this traces symmetry-breaking corrections through the BCH expansion DAG back to source lines. Output: formal proof with exact error order and source-line attribution.

**Tier 2 (dynamic ablation localization):** For code outside the liftable fragment, or as validation for Tier 1 results:

1. **Conservation-specific violation detection.** Compute d/dt C_v along trajectories for each declared conserved quantity. This is a scalar time series per conservation law, scaling to *any* system size—no sparse regression, no dictionary explosion. This is the Skeptic's simplification that eliminates Approach C's scalability bottleneck.

2. **Ablation-based causal attribution.** Systematically toggle code regions (force terms, splitting stages), re-run short trajectories, measure the change in d/dt C_v. Attribution via causal intervention, not regression. Works on any modular code with separable force components.

3. **Multi-scale h-analysis.** Run at 3–5 step sizes to determine the error order O(hᵖ) of each violation. This provides quantitative error estimates without symbolic computation.

**Graceful degradation:** Tier 1 results carry formal guarantees (proofs). Tier 2 results carry statistical confidence intervals. The unified report clearly distinguishes the two. This is honest and useful—formal proofs on the analyzable fragment, statistical guarantees everywhere else.

### 4.4 Obstruction Detection (T2)

The crown jewel capability: given a splitting H = H₁ + ⋯ + Hₖ and a conservation law C_v, determine whether *any* composition of integrators of given orders can preserve C_v to order p. If the obstruction ideal is non-trivial, no local code fix can restore conservation—the splitting itself is the problem.

**How it works:** The achievable correction terms at each order form a subspace (in the best case, linear; in general, semi-algebraic) of the free Lie algebra truncated at depth p on k generators. The obstruction check tests whether any element of this space annihilates C_v under the Noether pairing. For fixed k ≤ 5 and p ≤ 4, this involves ≤200 Lie bracket conditions—a tractable finite computation.

**What it tells the developer:** Not just "this code is wrong" but "this code is *unfixably* wrong with this splitting—restructure your algorithm." This saves potentially weeks of futile debugging and is the unique capability no LLM, SymPy script, or dynamic tool can replicate.

## 5. New Mathematics Required

### 5.1 T1: Tagged Modified Equation (Engineering Specification)

**Statement.** For a composition Φ = φₖ ∘ ⋯ ∘ φ₁ of k integrators applied to a split Hamiltonian H = H₁ + ⋯ + Hₖ, the modified Hamiltonian H̃ = H + Σ hⁿ δₙ decomposes as a sum of tagged Lie monomials, each carrying the subset of sub-integrators that generate it.

**Honest assessment.** T1 is ~20% new math, ~80% known BCH theory with engineering annotations. The BCH formula already encodes which operators contribute to which terms via nested commutator structure. Provenance tagging makes this explicit as a data structure specification. The potentially new content is in the *mixed-order* case (sub-integrators with orders p₁ ≠ p₂), where cross-term interactions are underexplored. If the proof reveals structural insights (e.g., "the leading symmetry-breaking term always originates from the lowest-order sub-integrator"), T1 earns theorem status. If not, it is a useful engineering formalization.

**Role in the tool:** T1 enables causal localization—tracing violations to source lines. Without it, the tool degrades to "somewhere in the integrator." It is necessary scaffolding, not the headline contribution.

### 5.2 T2: Computable Obstruction Criterion (Crown Jewel)

**Statement (precise).** Let H = H₁ + ⋯ + Hₖ, let C_v be a conserved quantity, and let p ≥ 1. Define the obstruction ideal O(v, H₁, …, Hₖ, p) generated by {δₙ, C_v} for 1 ≤ n ≤ p, where δₙ ranges over all achievable corrections. Then:

- **(a) Decidability.** Whether any composition of methods of given orders preserves C_v to order p is decidable via ≤B(k,p) Lie bracket conditions, where B(k,p) = O(kᵖ/p) (Witt dimension).
- **(b) Complexity.** For fixed k, p, each condition is polynomial-time in the sub-Hamiltonian size.
- **(c) Truncation limitation.** Sound at order p but not complete across all orders—unobstructed at order p may be obstructed at p+1.
- **(d) Necessity.** When confirmed, no local modification of any single φᵢ can restore conservation.

**What's known.** Decidability follows from Tarski-Seidenberg (doubly-exponential, useless). Specific impossibility results exist for individual methods (Ge & Marsden 1988; Chartier, Faou & Murua 2006).

**What's genuinely new.** An *efficient, structured* decision procedure exploiting free Lie algebra structure, yielding polynomial-time complexity for fixed k, p. This is the non-trivial mathematical content.

**Key risk (from Math Assessor).** The achievable correction terms at order n are polynomial (not linear) functions of the method coefficients. The set of achievable modified Hamiltonians is a semi-algebraic variety. If the feasibility check doesn't factor into independent linear conditions, the polynomial-time claim fails. **Mitigation:** If the efficient reduction fails, T2 is reframed as an "Obstruction Conjecture" with computational evidence on all benchmark examples (brute-force for k ≤ 5, p ≤ 4 is tractable regardless). The paper reports which regime the criterion works in and where it degrades.

**Proof strategy.** Characterize the image of the composition map in the truncated free Lie algebra. Show this image has enough structure (ideally linearity in the method coefficients at each fixed Lie monomial) to reduce the intersection check with the conservation kernel to linear algebra rather than quantifier elimination. Validate on 2–3 concrete examples before building.

### 5.3 Liftable Fragment Characterization

**Definition.** The liftable fragment L is the subset of imperative numerical code admitting faithful extraction into the conservation-aware IR. Under hybrid extraction, L includes: (a) pure-NumPy code with explicit loops and affine array indexing (Tree-sitter path), (b) JAX code whose jaxpr graph contains only modeled primitives (jaxpr path), (c) NumPy/SciPy code whose dispatch-intercepted operation sequence maps to polynomial-rational phase-space operations (dispatch path).

**Honest coverage estimates.** Tree-sitter alone: 15–25%. Hybrid extraction: 40–60% (conjectured; to be measured empirically on 5 real codebases in Phase 1). Remaining gaps: FFT-opaque operations, custom C extensions, data-dependent branching over state variables (e.g., `if r < r_cut`—the most common MD conservation-bug pattern, excluded from L).

**This is a definition and scope characterization, not a theorem.** It tells users exactly what the tool can and cannot analyze. The failure taxonomy classifies why extraction fails: opaque library call, data-dependent branching, non-polynomial nonlinearity, unsupported pattern.

## 6. Hard Subproblems and Risk Mitigation

| # | Subproblem | Classification | Risk | Mitigation | Fallback |
|---|-----------|----------------|------|------------|----------|
| 1 | Hybrid extraction (jaxpr + dispatch + Tree-sitter) | **NOVEL** | HIGH — no prototype exists; jaxpr interception is a second frontend | Phase 1 validates on 3 frameworks; dispatch interception is analogous to TorchDynamo (known to work) | If jaxpr path fails, restrict to NumPy dispatch + Tree-sitter (~30% coverage); Tier 2 covers the rest |
| 2 | T2 efficient reduction | **NOVEL** | HIGH — semi-algebraic feasibility may resist polynomial-time reduction | Prove on 2–3 concrete examples first; if polynomial-time reduction fails, brute-force for k≤5, p≤4 is tractable | Reframe as "Obstruction Conjecture" with computational evidence |
| 3 | Conservation-aware IR design | **HARD-BUT-KNOWN** | MODERATE — IR must serve symmetry analyzer, BCH engine, and localizer simultaneously | Prototype IR on Verlet/leapfrog first; iteratively refine | Simplify to single-method IR (loses composition analysis) |
| 4 | BCH with provenance tags | **HARD-BUT-KNOWN** | LOW — textbook BCH + bitset labels | Use Casas & Murua (2009) recursive formulas; SymPy for Phase 1, Rust for Phase 2 | Always tractable |
| 5 | Restricted Lie-symmetry solver | **HARD-BUT-KNOWN** | LOW — fixed ansatz reduces to structured linear algebra | Olver (1993) Ch. 2–3 gives the algorithm | Always tractable for ≤50 state vars |
| 6 | Differential symbolic slicing | **NOVEL** | MODERATE — new PL concept, no prior art | For paper-scope, block-level attribution suffices (no line-level); line-level is Phase 2 | Block-level localization via provenance tags |
| 7 | Ablation-based localization | **HARD-BUT-KNOWN** | LOW — causal intervention is standard; main risk is statistical power for small effects | Use Shapley-value-style attribution; report confidence intervals | Always works for coarse-grained (force-component-level) attribution |
| 8 | Conservation-specific dynamic detection | **ROUTINE** | LOW — computing d/dt C_v along trajectories is straightforward | Standard numerical differentiation | Always works |

## 7. Evaluation Plan

### Benchmarks (25 kernels)

| Category | Kernels | Conservation Properties |
|----------|---------|------------------------|
| JAX-MD (5) | LJ, soft sphere, Stillinger-Weber, EAM, Morse | Energy, linear momentum, angular momentum |
| Dedalus (5) | KdV, NLS, Burgers, Rayleigh-Bénard, shallow water | Energy, mass, momentum, enstrophy |
| Hand-written N-body (5) | Leapfrog, Yoshida, Ruth, Forest-Ruth, McLachlan | Energy, linear/angular momentum |
| SciPy ODE (5) | RK45, DOP853, Radau, BDF, LSODA on Hamiltonian systems | Energy (expected violations) |
| Conservation-violating mutants (5) | Targeted energy/momentum/angular-momentum/mass/symplecticity breaks | Ground-truth violating lines |

**External benchmarks (≥5):** Conservation bugs from LAMMPS GitHub issues, GROMACS changelogs, CESM bug tracker—reproduced faithfully in Python. These are the only non-circular evaluation component.

**Addressing the Skeptic's circularity concern:** The self-constructed kernels include ~8 genuinely distinct conservation problems (the rest are variants). We acknowledge this. External benchmarks and the LLM baseline provide non-circular evaluation.

### Baselines (6)

1. **Manual expert inspection** (8 hours per kernel). Human ceiling.
2. **Daikon** (dynamic invariant detection on simulation traces).
3. **LLM baseline** (GPT-4/Claude: paste kernel + conservation question). If the LLM matches ConservationLint on >70% of cases, the formal-methods framing needs revision. This directly addresses the Skeptic's LLM critique.
4. **Energy-only monitoring** (total Hamiltonian tracking with threshold).
5. **SymPy-assisted manual analysis** (grad student + SymPy + 2 hours per kernel). Current state of practice.
6. **Noether's Razor** (Cranmer et al., NeurIPS 2024) if CPU-feasible; otherwise SINDy-based conservation discovery as substitute.

### Metrics

| Metric | Target |
|--------|--------|
| Detection rate (recall) | ≥85% on combined benchmark suite |
| False positive rate | ≤10% |
| Localization accuracy (IoU) | ≥0.70 mean for Tier 1; ≥0.50 for Tier 2 |
| Coverage (hybrid extraction) | Measured empirically; target ≥35% of kernel lines on 3/5 real codebases |
| Analysis time (typical: <5K LoC, k≤5, p≤4) | ≤10 minutes |
| Tier 1 vs Tier 2 breakdown | Reported per kernel |

## 8. Best-Paper Argument

**The "impossible bridge" narrative.** ConservationLint connects two communities that have never spoken: geometric numerical integration (which knows everything about conservation but works on paper) and program analysis (which knows everything about code but has never targeted physical invariants). The bridge—hybrid extraction + conservation-aware IR + provenance-tagged analysis + two-tier localization—is the contribution.

**Why OOPSLA selects this:**

1. **Genuinely new paradigm.** Physics-aware program analysis is a new category, analogous to how SLAM brought model checking to device drivers and Herbie brought numerical reasoning to floating-point expressions.
2. **Non-trivial formal result.** T2 (if the efficient reduction works) is a genuine contribution to computational algebra with immediate practical impact—the obstruction certificate is a qualitatively new diagnostic capability.
3. **Strong evaluation.** The two-tier design ensures broad evaluation coverage (Tier 2 on 100% of runnable code) while demonstrating formal depth (Tier 1 on the liftable fragment). The LLM baseline directly addresses the most obvious reviewer objection.
4. **The killer demo.** Automatically rediscovering the Wan et al. (2019) climate-model conservation bug (reproduced in Python) would be a showstopper.

**The unique contributions that emerge only from this approach:** Heterogeneous multi-method composition analysis with obstruction detection. This is where SymPy + 100 lines fails, where LLMs give probabilistic guesses, and where ConservationLint gives proofs. The paper must focus here.

## 9. Honest Limitations

**What the tool cannot do:**

- Analyze Fortran/C++ code (Python-only scope excludes 80% of production simulation codes).
- Handle cutoff-based force truncation (`if r < r_cut`)—excluded from the liftable fragment by data-dependent branching, yet this is the most common MD conservation-bug pattern.
- Discover *undeclared* conservation laws (Tier 2 checks only declared quantities; Tier 1's symmetry analysis uses a restricted ansatz).
- Provide formal guarantees on code outside the liftable fragment (Tier 2 gives statistical results only).
- Scale Tier 1 formal analysis to codes >5K LoC with deep call graphs in the initial phases.

**Where LLMs win:** For simple-to-medium cases (single-method integrators, textbook splittings), GPT-4/Claude provides ~70% of the diagnostic value at zero cost. ConservationLint's unique value emerges for heterogeneous compositions (k>2) with obstruction detection—a genuinely valuable, genuinely hard, and genuinely rare use case.

**Coverage is uncertain.** The 40–60% hybrid extraction estimate is an informed conjecture. If empirical measurement yields <20%, the static tier's contribution to the paper is thin. The dynamic tier ensures the tool still works, but formal guarantees on a small fragment may not justify the engineering investment.

## 10. Subsystem Breakdown and LoC Estimate

Calibrated against the Difficulty Assessor's analysis. Phase 1 validates assumptions; Phase 2 builds for publication.

| # | Subsystem | Phase 1 | Phase 2 | Total | Language |
|---|-----------|---------|---------|-------|----------|
| 1 | Hybrid extraction (jaxpr + dispatch + Tree-sitter) | ~6K | ~6K | ~12K | Python + Rust |
| 2 | Conservation-aware IR | ~2K | ~3K | ~5K | Rust |
| 3 | Symbolic algebra engine (polynomials, Lie brackets) | ~3K | ~4K | ~7K | Rust (SymPy for Phase 1) |
| 4 | Restricted Lie-symmetry solver | ~2K | ~3K | ~5K | Rust |
| 5 | BCH engine with provenance tags | ~2K | ~3K | ~5K | Rust |
| 6 | Obstruction checker (T2) | ~1K | ~2K | ~3K | Rust |
| 7 | Causal localization (static slicing + dynamic ablation) | ~1K | ~5K | ~6K | Rust + Python |
| 8 | Dynamic tier (trajectory collection + ablation) | ~3K | ~4K | ~7K | Python |
| 9 | Benchmark suite | ~4K | ~8K | ~12K | Python |
| 10 | Evaluation harness + baselines | ~1K | ~4K | ~5K | Python |
| 11 | CLI + reporting | ~1K | ~3K | ~4K | Rust |
| | **Total** | **~26K** | **~45K** | **~71K** | |

**Phase 1 (~26K LoC, ~6 months):** Validate hybrid extraction on JAX-MD + pure-NumPy integrators. Prove T2 on 2–3 concrete examples. Measure coverage on 5 real codebases. Build minimal end-to-end pipeline on Verlet/leapfrog. Dynamic tier on all 5 benchmark categories.

**Phase 2 (~45K LoC, conditional on Phase 1):** Full static pipeline, complete benchmark suite, all baselines, publication-ready evaluation.

## 11. Scores

| Axis | Score | Rationale |
|------|-------|-----------|
| **Value** | 6/10 | Uniquely solves localization + obstruction for heterogeneous compositions—capabilities no existing tool provides. But narrow TAM (~300–500 users), Python-only, and LLMs cover ~70% of simple cases. The two-tier design broadens coverage beyond pure static analysis. |
| **Difficulty** | 6/10 | Hybrid extraction is genuinely novel engineering. T2's efficient reduction is the hard open math problem. Individual pipeline components (BCH, Lie symmetry, ablation) are known techniques. Integration risk is the real difficulty, not algorithmic novelty. |
| **Potential** | 6/10 | T2, if elegant, elevates the paper. The "impossible bridge" narrative is genuinely novel. But T1 is bookkeeping, the Connection Theorem (from B) is not load-bearing here, and the math depth is moderate—the paper wins on systems contribution + evaluation, not theorems alone. Realistic landing: strong OOPSLA acceptance, not best paper. |
| **Feasibility** | 6/10 | Hybrid extraction eliminates the existential risk of pure Tree-sitter (Approach A's fatal flaw, 55% kill probability). Dynamic tier ensures something always works (Approach C's 35% kill probability). Combined kill probability: ~25%. Phase 1 validates before Phase 2 investment. |
| **Composite** | **6.0/10** | Up from 5.75 baseline. The hybrid approach addresses the extraction crisis, the two-tier design provides graceful degradation, and honest framing (T1 as scaffolding, T2 as crown jewel, ~300–500 users) survives adversarial scrutiny. |

## 12. Kill Gates

| Gate | Milestone | Deadline | Trigger |
|------|-----------|----------|---------|
| **G1: Extraction viability** | Hybrid extraction produces faithful IR for ≥3 JAX-MD force kernels AND ≥3 pure-NumPy integrators | Month 3 | If <3 JAX-MD kernels extract faithfully, abandon jaxpr path; restrict to NumPy dispatch + Tree-sitter |
| **G2: Coverage threshold** | Liftable fragment covers ≥20% of kernel lines on ≥3 of 5 real codebases | Month 4 | If <15% on 3/5 codebases, the static tier is not viable. Pivot to dynamic-only tool (simplified Approach C) |
| **G3: T2 validation** | Obstruction criterion proved for 2–3 concrete examples. Efficient reduction assessed. | Month 4 | If T2 is trivial (just Tarski-Seidenberg with no structural insight) or exponential in k/p, reframe as "Obstruction Conjecture" with computational evidence. If even brute-force for k=3, p=3 fails, drop T2 entirely |
| **G4: LLM differentiation** | ConservationLint provides unique value (formal proof, obstruction detection, or Tier 2 localization) beyond GPT-4/Claude on ≥30% of benchmarks | Month 5 | If LLM matches ConservationLint on >70% of cases including heterogeneous compositions, the formal-methods framing is not justified. Pivot to survey/benchmark paper |
| **G5: End-to-end demo** | Complete pipeline (extraction → symmetry → BCH → localization) runs on ≥1 heterogeneous composition (k≥3) with correct obstruction detection | Month 6 | If the end-to-end pipeline cannot handle k≥3 by month 6, the paper cannot demonstrate the unique value beyond SymPy + 100 lines. Evaluate salvage options: benchmark suite (JOSS), T2 standalone (Numerische Mathematik), survey paper (ICSE/FSE) |

**Phase 2 proceeds only if G1–G5 are all met.** Any gate failure triggers reassessment, not immediate abandonment—but the reassessment must be honest about what survives.

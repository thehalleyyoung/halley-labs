# Verification Gate: sim-conservation-auditor (ConservationLint) — Mathematician Evaluation

## Committee

Three-expert adversarial team with explicit cooperation/competition loops:
- **Independent Auditor:** Evidence-based scoring with challenge testing
- **Fail-Fast Skeptic:** Aggressive rejection of under-supported claims
- **Scavenging Synthesizer:** Salvage value identification and reframing

Process: independent proposals → adversarial cross-critiques → synthesis → signoff.

## Verdict: CONDITIONAL CONTINUE

**Composite: 4.0/10 (V4/D5/BP3/L5/F3)**

Two experts recommend ABANDON (Auditor at 70% confidence, Skeptic at 65%). One recommends CONTINUE (Synthesizer). The cross-critique resolved the key disagreements closer to the Auditor/Skeptic positions. As mathematician, I weight mathematical load-bearing analysis heavily: the math in this project is not deep enough to be the reason the artifact is hard or valuable. The engineering bridge (Noether → program analysis) is genuinely novel, but the theorems that dress it are thin.

CONDITIONAL CONTINUE because the salvage floor is high (P(any pub) ≈ 0.75–0.85), the kill gates are well-designed with early termination at months 3–4, and the paradigm creation value — while speculative — is real if the extraction works. But this is a narrow continue, gated on proving T2 on paper for at least one concrete example within 3 weeks.

---

## Pillar Scores

### 1. EXTREME VALUE — 4/10

**The Auditor and Skeptic's core argument stands: the tool serves ~300–500 users in a niche where LLMs cover 70% of diagnostic value.**

The Synthesizer's paradigm argument ("physics-aware program analysis" is worth more than the tool) has structural merit — SLAM and Herbie were paradigm papers whose citation impact far exceeded their initial user base. But the cross-critique correctly identifies the disanalogy: SLAM shipped with Windows and proved model checking works *at scale on real systems*. ConservationLint has `theory_bytes=0` and `impl_loc=0`. You cannot claim paradigm-creation credit for a paradigm that hasn't been demonstrated on a single program.

**Evidence supporting 4/10:**
- TAM is ~300–500 users (final_approach.md §3, §11) — numerical methods researchers implementing structure-preserving integrators in Python.
- Conservation is priority #6 for simulation developers behind performance, correctness, portability, reproducibility, and stability (depth_check.md §1, citing Carver et al. SE-CSE surveys).
- The liftable fragment **explicitly excludes** `if r < r_cut` — acknowledged as "the most common MD conservation-bug pattern" (final_approach.md §5.3, §9). The formal tier is blind to the bugs users most need to find.
- Python-only scope excludes ~80% of production simulation codes (Fortran/C++ at national labs).
- Zero validated demand — no user interviews, no letters of support (verification_signoff.md §1).
- The Wan et al. (2019, JAMES) anecdote — the most compelling data point — is from a Fortran climate model. The tool doesn't support Fortran.
- LLMs cover ~70% of simple-case diagnostic value at zero cost (final_approach.md §9). The unique value margin — heterogeneous compositions (k>2) — is "genuinely rare" by the project's own admission.
- Existing tools (GROMACS `gmx energy`, LAMMPS `thermo_style`, OpenMM test suites) already detect *that* conservation is violated, partially undercutting the urgency framing (verification_signoff.md §1).

**Why 4 and not 3:** The Synthesizer's PIML verification connection is a genuine insight — physics-informed ML methods (Hamiltonian NNs, Lagrangian NNs) need conservation verification, and that audience is ~10K researchers. If the tool can verify conservation in differentiable simulators (JAX-MD is both a simulation framework and differentiable), the audience multiplier is real. Additionally, the obstruction detection capability ("this violation is architecturally unfixable") is qualitatively new and unmatched by any competitor.

**Why 4 and not higher:** The PIML reframing is the Synthesizer's speculation, not the project's stated plan. Zero demand validation remains damning. The liftable fragment exclusion of cutoff-based forces is a critical scope gap the project itself acknowledges.

### 2. GENUINE SOFTWARE DIFFICULTY — 5/10

**The difficulty is in engineering integration, not algorithmic novelty. The hard part is making 5 pipeline stages across 3 ontologies work together end-to-end.**

**Evidence supporting 5/10:**
- Hybrid extraction (jaxpr interception + NumPy dispatch + Tree-sitter) has no prior art for conservation analysis (final_approach.md §4.1). The TorchDynamo analogy is encouraging but imprecise — JAX's tracing model differs from PyTorch's.
- Integration risk across 5 stages (extract → IR → symmetry → BCH → localize) is rated 8/10 by the verification signoff (verification_signoff.md §5). Three different ontologies (continuous ODEs, discrete flows, imperative code) must be bridged.
- ~71K LoC total estimate, of which ~30–35K is genuinely novel (depth_check.md §2). The rest is benchmark code (12K), evaluation harness (5K), CLI (4K), and standard wrappers.
- Individual components are standard: BCH expansion is textbook (Hairer, Lubich & Wanner 2006), Lie-symmetry analysis with restricted ansatz is "solvable in milliseconds" (verification_signoff.md §2), ablation-based localization is standard causal intervention.
- The obstruction checker (T2) is 3K LoC (final_approach.md §10) — the "crown jewel" is a small component relative to the engineering bulk.
- The symbolic algebra engine was inflated ~2.5× in original estimates (claimed 20K, realistic 8–10K per depth_check.md §2).

**As a mathematician, I note:** The difficulty here is *engineering*, not *mathematics*. A project whose difficulty is "making five known techniques work together" does not score highly on mathematical difficulty. The one genuinely difficult mathematical problem (T2's efficient reduction) is unsolved, unprototyped, and may not exist.

### 3. BEST-PAPER POTENTIAL — 3/10

**This is where the mathematician's lens is sharpest. The project has ~0.4–0.6 novel theorem-equivalents. The load-bearing math is thin.**

#### Novel Theorem-Equivalents (rigorous assessment)

| Claim | Score (0–1) | Assessment |
|-------|-------------|------------|
| **T1: Tagged Modified Equation** | **0.1** | BCH expansion is textbook (Blanes, Casas & Murua 2008). Provenance tagging is a data structure annotation (bitset labels on Lie monomials). The project itself says "~20% new math, ~80% known" and demotes T1 to "engineering specification" (final_approach.md §5.1). The mixed-order case has modest novelty — Blanes & Moan (2002) already handle mixed methods, though systematic attribution of error terms to sub-integrators is underexplored. This is a useful formalization, not a theorem. |
| **T2: Obstruction Criterion** | **0.3** | The only genuinely mathematical claim. Decidability follows from Tarski-Seidenberg (known, doubly exponential). For k≤3, p≤3: ≤27 Lie bracket conditions (raw k^p upper bound; Witt formula gives ~14) — a brute-force finite computation, not an "efficient structured decision procedure." The approach.json self-grades T2 as **C** and acknowledges "exponential complexity limits practical scope." The "efficient reduction" is the claimed novelty, but the achievable corrections are polynomial functions of method coefficients → semi-algebraic variety → intersection with conservation kernel likely requires quantifier elimination, not linear algebra. **No proof exists** (`theory_bytes=0`). The Synthesizer identifies escape routes (restrict to linear/quadratic charges, fix k=2 general p, stratify by Lie rank) that could yield clean theorems, but these are suggestions, not results. |
| **T3: Liftable Fragment** | **0.0** | A definition. Every static analysis paper defines its analyzable scope. The syntactic characterization (no recursion, no data-dependent branching, affine array indices) is the polyhedral model from compiler literature (Feautrier 1992, Bondhugula et al. 2008). "Decidable membership" is trivially achieved by checking the AST. |
| **Differential Symbolic Slicing** | **0.15** | Novel as a PL concept (program slicing where criterion is contribution to a formal power series term). But given provenance tags from T1, the implementation is graph traversal on a tagged DAG. The novelty is conceptual, not algorithmic. |
| **TOTAL** | **~0.55** | Less than one novel theorem-equivalent. |

#### Mathematical Load-Bearing Analysis

**The critical question:** Is the math the reason the artifact is hard to build and the reason it delivers extreme value?

**Answer: No.** The math is not load-bearing in the required sense.

- **T1 (provenance-tagged BCH)** is load-bearing for *localization* — without it, the tool degrades to "somewhere in the integrator." But the math is known BCH expansion with bookkeeping annotations. The *implementation* of provenance tracking is hard; the *mathematics* is not.
- **T2 (obstruction criterion)** is load-bearing for the *unique* capability (proving violations are architecturally unfixable). But T2 is unproven, self-graded C, and may be trivial in the tractable regime. If T2 degrades to brute-force (likely), the "crown jewel" is a lookup table, not a theorem.
- **T3 (liftable fragment)** is not load-bearing at all — it's scope documentation.

The system's actual difficulty and value come from *engineering*: building the extraction pipeline, integrating five heterogeneous stages, and demonstrating end-to-end operation on real code. This is a strong engineering contribution dressed in mathematical language. For a math-driven evaluation, this is a fundamental mismatch.

**Comparison to genuine best-paper math:** A best-paper-caliber mathematical contribution would be something like: "We prove that for any splitting of a Hamiltonian system into k components, conservation of a Noether charge C to order p is decidable in O(poly(k,p)) time, and we give a constructive algorithm." This would be a genuine theorem with surprising content (the polynomial-time part would be surprising given the semi-algebraic setting). The current T2 falls far short — it's either brute-force for small parameters or an unproven aspiration for the general case.

**Venue assessment:** The paper's realistic landing is OOPSLA/CAV acceptance as a domain-specific tools paper, not best paper. Best paper at OOPSLA requires either a transformative practical impact (which ~500 users doesn't provide) or a stunning formal result (which 0.55 theorem-equivalents doesn't provide). The project self-estimates P(best_paper) = 0.05 (approach.json); I concur.

### 4. LAPTOP-CPU FEASIBILITY & NO-HUMANS — 5/10

**The core computations are laptop-tractable, but phase-space annotations inject human labor.**

**Evidence supporting 5/10:**
- All computation is symbolic/algebraic — no GPU needed, no neural network training.
- BCH order 4, k=5 (typical case): tractable in seconds (depth_check.md §4).
- BCH order 6, k=10: ~17 minutes, exceeds the 10-minute budget (depth_check.md §4). Qualification needed: "typical kernels <5K LoC, k≤5, order≤4."
- Lie-symmetry analysis with restricted ansatz: milliseconds for ≤50 state variables.
- **Phase-space annotations are human labor.** Users must declare phase-space structure, coordinate systems, and which quantities to check. The approach.json lists A1: "Users can accurately annotate phase space structure" as a REASONABLE assumption, but the adversarial findings flag "Phase space annotations create significant user burden" (approach.json adversarial findings). This is a human-in-the-loop element that violates the "no human annotation" criterion in spirit.
- IR extraction time unbounded for complex codebases with deep call graphs.
- The Noether's Razor baseline requires GPU training — invalid for CPU-only comparison (verification_signoff.md §4).

**Why 5 and not 7:** The Auditor scored CPU/no-humans at 7/10, but I weight the phase-space annotation burden more heavily. "No humans" means no human annotation, and this system requires domain-expert annotation of the phase space before analysis begins. For a 10-line Verlet integrator, annotations are trivial. For a real JAX-MD simulation with multiple force components, annotations may require significant expert time (>30 minutes). This partially defeats the automation promise.

### 5. FEASIBILITY — 3/10

**Zero output after theory stage. The core extraction assumption is unvalidated. The crown jewel proof doesn't exist.**

**Evidence supporting 3/10:**
- `theory_bytes = 0`, `impl_loc = 0`, `monograph_bytes = 0` (State.json). The theory stage was designed to produce proofs. It produced a JSON planning document. **This is a process failure.**
- The approach.json P(abandon) = 0.15 contradicts the verification signoff's estimate of 25%, which in turn contradicts the Skeptic's estimate of 78%. The most adversarially tested estimate (Skeptic, ~55–70% with correlation adjustment) seems most reliable, but even the signoff's 25% is concerning.
- No code→math extraction prototype exists. No jaxpr interception has been attempted. No coverage measurement has been performed.
- The "~50 primitives" claim for jaxpr contradicts JAX's actual primitive count (hundreds).
- 71K LoC across Rust + Python is ~2–3 person-years. Phase 1 (26K in 6 months) requires ~1K LoC/week of novel, mathematically complex code.
- The Skeptic's gate-by-gate analysis: P(survive all 5 gates) ≈ 14%. Even with positive correlations, P(ABANDON) ≈ 55–70%.
- The Synthesizer's counter (extraction success raises all downstream probabilities) is structurally correct but P(extraction success) itself is only ~50–65%.

**Feasibility decomposition (as mathematician, focusing on mathematical feasibility):**
- P(T2 proof with non-trivial content): ~25% — the semi-algebraic structure may resist polynomial-time reduction, and brute-force for small parameters is not a "proof" in any meaningful sense.
- P(hybrid extraction works adequately): ~50% — novel engineering, no prototype, but jaxpr is a public API.
- P(end-to-end pipeline on k≥3 composition): ~30% — requires extraction + IR + symmetry + BCH + localization all working together.
- P(top venue paper | everything works): ~50% — narrow TAM, LLM competition, evaluation concerns.
- **P(top venue acceptance): ~0.25 × 0.50 × 0.30 × 0.50 ≈ 0.019.** Extremely pessimistic, but reflects the zero-production starting point.

**Why 3 and not lower:** The kill gates genuinely bound downside. Phase 1 is a 6-month validation with early termination at months 3–4. The salvage options (JOSS benchmark, T2 standalone, survey paper) are independently publishable. The two-tier architecture ensures something works even if extraction fails. P(any pub) ≈ 0.70–0.85.

---

## Novel Theorem-Equivalents Summary

| Claim | Score | Justification |
|-------|-------|---------------|
| T1: Tagged Modified Equation | 0.1 | BCH + data structure. Self-demoted to "engineering specification." |
| T2: Obstruction Criterion | 0.3 | Decidability known. Efficient version unproven, C-grade, likely brute-force. |
| T3: Liftable Fragment | 0.0 | Definition. Standard compiler scope characterization. |
| Differential Symbolic Slicing | 0.15 | Novel concept. Trivial implementation given provenance tags. |
| **TOTAL** | **~0.55** | Less than one novel theorem-equivalent. |

---

## Fatal Flaws

### Flaw 1: theory_bytes = 0 After Theory Stage (SERIOUS)

The theory stage produced zero bytes of mathematical content. No proofs. No paper.tex. The approach.json (34KB) is planning metadata, not research output. The T2 proof — the project's crown jewel — exists only as an aspiration. **At a theory → implementation gate, this is disqualifying absent a convincing argument that the theory can be completed during implementation.** The Skeptic's point is sharp: proceeding to 71K LoC of implementation on conjectured theorems is exactly what the gate system exists to prevent.

**Mitigation:** The verification signoff's binding condition C3 (prove T2 on 2–3 examples first) partially addresses this, and the kill gate G3 at month 4 provides an abort trigger.

### Flaw 2: The Crown Jewel Is C-Grade and May Be Trivial (SERIOUS)

T2 is self-graded C in the approach.json. It is EXPSPACE-complete in general. For the tractable regime (k≤3, p≤3), it involves ≤O(27) Lie bracket checks — a finite computation performable by a graduate student with SymPy in an afternoon. The "efficient structured decision procedure" that would elevate T2 to a genuine theorem is an open research problem with no evidence of tractability. If the efficient reduction fails, the fallback is "Obstruction Conjecture with computational evidence" — a conjecture verified computationally is not OOPSLA/PLDI headline material.

**The Synthesizer's escape routes** (restrict to linear/quadratic Noether charges, fix k=2 general p, stratify by Lie rank) are genuinely interesting directions that could yield clean, publishable results. But they are suggestions from an evaluation, not results from the project.

### Flaw 3: The Liftable Fragment Excludes the Primary Bug Pattern (MODERATE)

Cutoff-based force truncation (`if r < r_cut`) is explicitly excluded from Tier 1 analysis due to data-dependent branching over state variables, yet the project itself calls this "the most common MD conservation-bug pattern." The tool's formal analysis tier cannot handle the very bugs its users most need to find. Tier 2 (dynamic) covers these statistically, but statistical coverage of the primary use case while the formal tier handles secondary cases is an inverted value proposition.

### Flaw 4: Unvalidated Core Extraction Assumption (MODERATE)

No code→math extraction prototype exists. No jaxpr interception for conservation analysis has been attempted. The 40–60% coverage estimate is "conjectured" with no empirical data. The cross-critique identifies this as the single gating question: "Can a NumPy integrator be mechanically extracted into a symbolic IR that admits Lie-symmetry analysis?" This is answerable in ~2 weeks with a 200-line script.

### Flaw 5: Venue Incoherence (MINOR)

final_approach.md targets OOPSLA; approach.json targets PLDI 2025. These venues have different expectations (OOPSLA favors domain-specific tools with evaluation; PLDI favors formal semantics contributions). The inconsistency suggests insufficient venue analysis.

---

## Team Score Comparison

| Axis | Auditor | Skeptic | Synthesizer | Cross-Critique | **Mathematician (Final)** |
|------|---------|---------|-------------|----------------|---------------------------|
| Value | 3 | 3 | 6 | 4 | **4** |
| Difficulty | 5 | 3 (math) / 4 (overall) | 6 | — | **5** |
| Best-Paper | 3 | 4 (novelty) | 5.5 | 3.5 | **3** |
| Laptop/CPU | 7 | — | 6.5 | — | **5** |
| Feasibility | 3 | 4 | 6.5 | 4.5 | **3** |
| **Composite** | **4.2** | **3.4** | **6.0** | **4.7** | **4.0** |

The Auditor and Skeptic converge (3.4–4.2); the Synthesizer diverges upward (6.0). As mathematician, I align with the Auditor/Skeptic on mathematical depth (the math is not load-bearing) while giving modest credit to the Synthesizer's paradigm-creation argument on Value.

---

## P(ABANDON) Analysis

**Skeptic's gate-by-gate analysis (most rigorous):**

| Gate | P(fail) | Reasoning |
|------|---------|-----------|
| G1: Extraction viability (Mo 3) | 0.35 | jaxpr interception unprecedented; "~50 primitives" unsupported |
| G2: Coverage ≥15% (Mo 4) | 0.30 | Real codes use FFTW, MPI, C extensions |
| G3: T2 validated (Mo 4) | 0.40 | Semi-algebraic feasibility may resist efficient reduction |
| G4: LLM differentiation ≥30% (Mo 5) | 0.25 | LLMs improving rapidly; threshold on self-constructed benchmarks |
| G5: End-to-end k≥3 demo (Mo 6) | 0.30 | 5 stages, 3 ontologies, integration risk 8/10 |

P(survive all) ≈ 0.65 × 0.70 × 0.60 × 0.75 × 0.70 ≈ 0.14

With positive correlations (G1 failure raises G2/G5 probability): **P(ABANDON) ≈ 55–70%.**

The approach.json's P(abandon) = 0.15 is not credible. The verification signoff's 25% is optimistic. I adopt the range 45–60%.

---

## Salvage Analysis (from Synthesizer, verified)

| Artifact | Venue | Timeline | P(publish) |
|----------|-------|----------|------------|
| Conservation benchmark suite (25 kernels) | JOSS | 2–3 months | 0.85 |
| T2 obstruction tables (brute-force k≤5, p≤4) | BIT / Numerische Mathematik | 3–4 months | 0.50 |
| "Physics-Aware Program Analysis" survey | ICSE-NIER / Onward! | 2–3 months | 0.70 |
| Dynamic-only conservation localizer | ICSE / ASE | 4–5 months | 0.50 |

**P(any publication) ≈ 0.75–0.85** (salvage calculation gives 0.99 assuming full independence of artifacts, which overestimates; adopted range accounts for correlated effort and skill dependencies). The salvage floor is genuinely high. Even total failure of the headline vision produces publishable outputs.

**Minimum viable paper (MVPaper):** Conservation-aware IR + T1 on 3–5 concrete examples + ablation-based dynamic localization + LLM/SymPy baselines + honest coverage measurement. ~25K LoC, 6 months. Publishable at OOPSLA as a domain-specific tools paper even without T2 and even with only 20% coverage (if honestly reported).

---

## Key Findings from Team Disagreements

### Resolved in Favor of Auditor/Skeptic
1. **Value is narrow, not paradigmatic.** Paradigm creation requires a working demonstration; ConservationLint has none. You cannot claim paradigm credit for an unbuilt tool.
2. **T2 is not a crown jewel yet.** It's a C-grade conjecture that may be trivial (brute-force at small parameters) or intractable (EXPSPACE at large parameters). The sweet spot ("efficient structured test") is an open problem with no progress.
3. **theory_bytes = 0 is a process failure.** The theory stage should have produced at minimum a proof sketch of T2 for one concrete example.

### Resolved in Favor of Synthesizer
1. **The Synthesizer's T2 escape routes are valuable.** Restricting to linear/quadratic Noether charges, fixing k=2 for general p, or stratifying by Lie rank could yield clean theorems. These should be the project's immediate mathematical targets.
2. **The salvage floor is high.** P(any pub) ≈ 0.99 means the investment is not wasted even under full failure.
3. **Mixed-order BCH (T1) is genuinely underexplored.** The Skeptic's "bookkeeping" dismissal is correct for homogeneous compositions but incorrect for the mixed-order case that matters in practice. The mixed-order attribution of error terms to specific sub-integrators is novel, if modest.
4. **The PIML verification connection is real.** Differentiable simulators (JAX-MD) need conservation verification. This audience expansion should be explored.

### Unresolved
1. **Can extraction work?** The single most important question. Answerable in ~2 weeks with a 200-line prototype. No one can resolve this by argument.

---

## Binding Conditions for CONTINUE

### BC1: T2 Proof on Paper (Week 3 Deadline)
Prove the obstruction criterion for **one concrete example**: Strang splitting (k=2) of a Hamiltonian with SO(3) angular momentum, to order p=2. If this proof is trivial (just Tarski-Seidenberg), flag it honestly. If it reveals polynomial-time structure, document the key lemma. If it fails outright, reclassify T2 as "Obstruction Conjecture."

### BC2: Extraction Prototype (Week 4 Deadline)
Build a 200-line Python script that extracts a NumPy Verlet integrator into a symbolic IR (SymPy expressions) and runs Lie-symmetry analysis. If this fails, the static tier is dead and the project pivots to dynamic-only.

### BC3: Venue Decision (Week 1)
Pick OOPSLA or PLDI. Resolve the inconsistency between final_approach.md and approach.json. Commit.

### BC4: Coverage Discipline (Throughout)
The 40–60% coverage claim must not appear in any document until validated. All planning uses 15–25% (Tree-sitter alone) as the baseline. The abort threshold remains <15%.

### BC5: Honest P(abandon) (Week 1)
Reconcile the three conflicting estimates: approach.json (0.15), verification signoff (0.25), and Skeptic (0.78). Produce a single calibrated estimate with explicit reasoning. My estimate: **0.45–0.60**.

---

## VERDICT: CONDITIONAL CONTINUE

**Vote: 2 ABANDON (Auditor 70%, Skeptic 65%) / 1 CONTINUE (Synthesizer) / Lead: CONDITIONAL CONTINUE**

I override the majority toward CONDITIONAL CONTINUE for three reasons:

1. **The salvage floor justifies bounded investment.** P(any pub) ≈ 0.75–0.85 means the 6-month Phase 1 investment very likely produces something publishable regardless. The kill gates ensure maximum wasted investment is 6 months, not 2 years.

2. **The gating question is cheaply testable.** "Can a NumPy integrator be extracted into a symbolic IR?" is answerable in 2 weeks with 200 lines of Python. Abandoning before running this experiment wastes the entire ideation investment.

3. **T2's escape routes are mathematically promising.** The Synthesizer identifies specific paths (linear/quadratic charge restriction, k=2 fixed, Lie rank stratification) that could yield clean theorems publishable independently of the tool. These represent genuine mathematical opportunities that the Auditor/Skeptic didn't evaluate.

**However:** This is a narrow, probationary CONTINUE. The project's self-score of 6.0/10 is inflated; the calibrated composite is 4.0/10. The math is not the reason this artifact is hard — the engineering is. For a mathematician's evaluation, this means the project's intellectual core is thinner than claimed. The project survives on engineering novelty (the bridge), paradigm potential (physics-aware program analysis), and salvage value (benchmarks + survey + T2 standalone) — not on mathematical depth.

**If BC1 (T2 proof) and BC2 (extraction prototype) both fail by week 4, ABANDON immediately.** No further investment is justified if the crown jewel is trivial and the extraction doesn't work.

---

## Summary Scorecard

| Axis | Score | Key Driver |
|------|-------|------------|
| Extreme Value | 4/10 | ~300–500 users; LLMs cover 70%; excludes primary bug pattern; PIML connection adds modest upside |
| Genuine Difficulty | 5/10 | Integration of 5 stages is genuinely hard; individual components are known; ~30K novel LoC |
| Best-Paper Potential | 3/10 | ~0.55 novel theorem-equivalents; T2 C-grade; math not load-bearing; P(best-paper) ≈ 5% |
| Laptop-CPU / No-Humans | 5/10 | Symbolic computation is laptop-native; phase-space annotations are human labor; BCH breaks at high k/p |
| Feasibility | 3/10 | theory_bytes=0; impl_loc=0; extraction unvalidated; P(ABANDON) ≈ 45–60% |
| **Composite** | **4.0/10** | |

| Probability | Estimate |
|-------------|----------|
| P(top venue: OOPSLA/PLDI) | 10–20% |
| P(best paper) | 2–5% |
| P(any publication) | 75–85% |
| P(ABANDON at gate) | 45–60% |
| Novel theorem-equivalents | 0.55 |

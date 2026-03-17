# Verification Gate: ConservationLint (proposal_00) — Post-Theory Evaluation

## Team Composition

| Role | Function | Default Stance |
|------|----------|----------------|
| **Independent Auditor** | Evidence-based scoring, kill-gate probability analysis, quantitative rigor | Neutral — scores to evidence |
| **Fail-Fast Skeptic** (lead author) | Aggressively reject under-supported claims, find fatal flaws, stress-test every assumption | Reject unless proven otherwise |
| **Scavenging Synthesizer** | Extract salvageable value from failing proposals, identify reduced-scope continuations | Optimistic within constraints |

**Evaluation context.** This is a *post-theory* gate. The proposal entered the theory stage with a clear mandate: prove T2 (the computable obstruction criterion), formalize T1, and produce at least preliminary mathematical content. The theory stage has concluded. We evaluate what was delivered.

**What was delivered:** `theory_bytes = 0`. `impl_loc = 0`. `monograph_bytes = 0`. `theory_score = null`. (State.json, lines 12–17.)

---

## Phase 1: Independent Evaluations

### Independent Auditor Assessment

**Composite: 4.4/10. VERDICT: ABANDON.**

The Auditor constructed a compound failure model across 5 sequential kill gates (G1–G5 from final_approach.md §12), each with independent failure probabilities calibrated to the evidence:

| Gate | P(fail) | Basis |
|------|---------|-------|
| G1: Extraction viability | 35% | No jaxpr interception prototype exists; TorchDynamo analogy is weak (verification_signoff.md §Remaining Risks, Risk 1) |
| G2: Coverage threshold (≥15%) | 25% | Coverage estimate is "conjectured; to be measured" (final_approach.md §7); depth_check.md §2 Flaw 2 calls it "fabricated" |
| G3: T2 validation | 40% | Semi-algebraic variety structure may defeat efficient reduction (final_approach.md §5.2 Key Risk); zero proof attempts in theory stage |
| G4: LLM differentiation | 30% | ~70% overlap on simple cases acknowledged (final_approach.md §9); moat shrinks with each LLM generation |
| G5: End-to-end demo | 20% | Contingent on G1–G4 success; integration of 5 pipeline stages with 3 ontologies (depth_check.md §2, para on integration risk) |

**Compound success probability:** 0.65 × 0.75 × 0.60 × 0.70 × 0.80 = **0.164**, i.e., **~81% probability of hitting at least one kill gate.** This alone justifies ABANDON.

**Key findings:**
- theory_bytes=0 after the theory stage is "100% planning documents with zero validated assumptions"
- The proposal's own kill-gate structure predicts its failure: the gates were designed to catch exactly the risks that remain unaddressed
- Three salvage paths have positive expected value: benchmark suite, survey paper, standalone T2 proof
- Conditional override: T2 proof for one concrete example + jaxpr prototype extracting one JAX-MD kernel + BCH provenance demo on Störmer-Verlet, all within 4 weeks — or hard abandon

**Dimension scores:** V4/D5/BP4/CPU6/F3

### Fail-Fast Skeptic Assessment

**Composite: 3.0/10. VERDICT: ABANDON.**

**Dimension scores:** V3/D5/BP2/CPU6/F2

This is the longest assessment because the Skeptic's mandate is to find every flaw and stress-test every claim. The proposal fails on multiple independent axes.

#### The Central Indictment: theory_bytes = 0

The theory stage exists to produce theory. It produced none. State.json records:

```json
"theory_bytes": 0,
"impl_loc": 0,
"monograph_bytes": 0,
"theory_score": null
```

There are no proofs. There is no paper. There is no math. There is no code. After the ideation stage produced ~87K bytes of planning documents (final_approach.md: 30.2KB, depth_check.md: 32.9KB, verification_signoff.md: 23.6KB), the theory stage produced exactly zero bytes of mathematical content. The ratio of meta-analysis to actual work is undefined (division by zero).

The proposal spent its entire theory budget *talking about* what T2 would look like if someone proved it. The precise statement of T2 (final_approach.md §5.2) contains four sub-claims — decidability, complexity, truncation limitation, necessity — none of which have been validated by even a single worked example. The document even identifies the proof strategy: "Characterize the image of the composition map... Show this image has enough structure... Validate on 2–3 concrete examples before building." None of this was done.

#### Value (V): 3/10

The proposal's own documents destroy its value case:

- **TAM ~300–500 users** (final_approach.md §1, §3). This is an honest number, and it is devastating. By the proposal's own admission, the total addressable market is smaller than a large university lecture hall.
- **Zero validated demand.** No user interviews, no letters of support, no survey data (depth_check.md §1 Flaw 6). The demand validation plan ("3–5 structured interviews," problem_statement.md §Value Proposition) was never executed.
- **Conservation ranks 6th among practitioner pain points.** Per depth_check.md §1: "performance/scalability, general correctness bugs, portability, reproducibility, numerical stability, *then* conservation-specific violations."
- **Existing monitors already exist.** GROMACS `gmx energy`, LAMMPS `thermo_style`, OpenMM's test suite — these detect *that* conservation is violated (depth_check.md §1). ConservationLint's unique value is *why* and *where*, but only within the liftable fragment, which is measured at exactly 0% because no code exists to measure it.
- **LLM threat is existential.** The final_approach.md §9 acknowledges: "GPT-4/Claude provides ~70% of the diagnostic value at zero cost." The moat — formal obstruction proofs for heterogeneous compositions — requires T2 to work. T2 has zero bytes of proof.

#### Difficulty (D): 5/10

The genuine difficulty exists but is frontloaded in the *unvalidated* extraction layer:

- **Code→math extraction:** All three evaluators independently identified this as the existential risk (depth_check.md §5 Flaw 1). The hybrid extraction via jaxpr interception is conceptually sound but has zero implementation evidence. The TorchDynamo analogy is misleading — TorchDynamo captures computation graphs for *optimization*, not for *mathematical conservation analysis*, and took years of engineering by a well-funded team.
- **T2's efficient reduction:** The achievable correction terms form a semi-algebraic variety (final_approach.md §5.2 Key Risk). If the polynomial-time claim fails, T2 degrades to brute-force Tarski-Seidenberg — "decidable" in the same way that Presburger arithmetic is decidable. The approach.json's own grading: "C" for T2's approach.
- **Individual pipeline components are known techniques.** BCH expansion is textbook (Hairer, Lubich & Wanner 2006). Lie-symmetry analysis with restricted ansatz "reduces to structured linear algebra... solvable in milliseconds" (final_approach.md §4.2). CUSUM/SPRT is standard statistical process control. The difficulty is integration, not algorithmic novelty.
- **The liftable fragment excludes the most common bug class.** Cutoff-based force truncation (`if r < r_cut`) — "the most common MD conservation-bug pattern" (final_approach.md §9) — requires data-dependent control flow over state variables, which is explicitly excluded from Tier 1 analysis.

#### Best-Paper Potential (BP): 2/10

- **T1 is bookkeeping.** The proposal itself admits: "T1 is ~20% new math, ~80% known BCH theory with engineering annotations" (final_approach.md §5.1). The depth_check.md §3 confirms: "BCH expansion is textbook material... The provenance tagging is the novelty, not the computation itself."
- **T2 is an unproven hope.** Zero bytes of proof exist. The polynomial-time claim is explicitly flagged as possibly false (final_approach.md §5.2: "If the feasibility check doesn't factor into independent linear conditions, the polynomial-time claim fails"). T2 *might* be a crown jewel — or it might be trivial Tarski-Seidenberg, or it might be EXPSPACE-complete in general.
- **T3 is a definition dressed as a theorem.** Unanimous across all three evaluators (depth_check.md §3): "Every static analysis paper defines the fragment of programs it handles."
- **The SLAM/Herbie comparisons are aspirational, not structural.** depth_check.md §3: "SLAM succeeded because device drivers have a small, well-defined API surface. Herbie succeeded because floating-point expressions are syntactically simple... ConservationLint's domain — arbitrary simulation codes with heterogeneous integrators — has neither." SLAM shipped with Windows. Herbie is used by thousands. ConservationLint is a markdown file.
- **P(top-venue) ≈ 21% unconditional** (Skeptic estimate, adjusting depth_check.md's 35% conditional estimate by the 81% compound gate-failure probability: 0.35 × 0.19 + 0.10 × 0.81 ≈ 0.15, generously rounded up). **P(best-paper) ≈ 4%** at most (depth_check.md §Risk Assessment), and that assumes everything works — which, after theory_bytes=0, is not a reasonable assumption.

#### CPU Feasibility (CPU): 6/10

This is the proposal's strongest dimension. The core symbolic computations (Lie-symmetry linear algebra, BCH at order ≤4 with k ≤ 5) are genuinely laptop-tractable (depth_check.md §4). The 10-minute budget holds for typical parameters. The Noether's Razor baseline is invalid CPU-only (depth_check.md §4), but that's an evaluation issue, not a core feasibility issue.

#### Feasibility (F): 2/10

Distinguished from CPU feasibility: this is *project* feasibility, incorporating the theory_bytes=0 signal:

- The theory stage was specifically allocated for mathematical work. It produced nothing. This is the strongest possible signal that the team cannot deliver the mathematical content the proposal requires.
- The compound gate-failure probability is ~81%.
- Three documents describe three different systems: the original problem statement estimated ~90K LoC with ~20–40% coverage; the final_approach.md estimates ~71K LoC with ~40–60% coverage; the depth_check consensus is ~92K LoC with coverage "unknown." Nobody knows what is being built or how much of it will work.
- Coverage estimates range from 10–20% (Skeptic realistic, depth_check.md §5 Flaw 2) to 40–60% (final_approach aspirational) to "fabricated" (all three evaluators). The honest answer is: nobody knows because nobody has tried.

### Scavenging Synthesizer Assessment

**Composite: 4.35/10. VERDICT: ABANDON full proposal → CONTINUE-REDUCED.**

The Synthesizer agrees the full proposal is dead but identifies a "core diamond" worth salvaging: T2's obstruction criterion is "the only capability that truly dominates LLMs" — formal proofs that a conservation violation is *architecturally unfixable* cannot be replicated by pattern-matching or probabilistic reasoning.

**Salvage paths (ranked by expected value):**

| Path | Scope | P(success) | Venue | Value |
|------|-------|------------|-------|-------|
| **A: Benchmark suite** | ~3K LoC, 3 months | 85% | JOSS | Community resource, citeable for a decade |
| **B: Survey/SoK paper** | Minimal code, 3 months | 75% | ICSE/FSE SoK track | Maps the design space, positions future work |
| **C: Dynamic-only auditor** | ~5K LoC, 4 months | 65% | SciPy conf / JOSS | Practical tool, no formal analysis required |
| **D: T2 proof (standalone)** | Pure math, 6 months | 30% | Numerische Mathematik / BIT | The moonshot — high reward if it works |

The Synthesizer's metaphor: "The mine has collapsed, but the diamonds are worth saving." The core T2 idea has genuine intellectual merit. But the full ConservationLint tool — hybrid extraction, conservation-aware IR, two-tier analysis, causal localization, 71K LoC — is not viable without the mathematical foundation it was supposed to build in this stage.

---

## Phase 2: Adversarial Cross-Critique

### Skeptic → Auditor

**Challenge 1: The Auditor's gate-failure model is too generous on G2.**

The Auditor assigns P(G2 fail) = 25%. This assumes coverage *could* reach 15% on 3/5 codebases. But the proposal's own analysis reveals that Dedalus is "built on FFTW, MPI, and dense linear algebra — all opaque to Tree-sitter" (depth_check.md §5 Flaw 2), JAX-MD uses "JAX transformations (jit, vmap, grad) that are fundamentally non-trivial to lift," and cutoff-based control flow is excluded. The 5 target codebases (JAX-MD, Dedalus, SciPy ODE, gray radiation, ASE-MD) include at least 2 that are almost certainly below threshold. P(G2 fail) should be 35–40%, pushing compound failure to ~86%.

**Challenge 2: The Auditor's conditional override is too lenient.**

The Auditor offers a 4-week conditional override: T2 proof + jaxpr prototype + BCH demo. But theory_bytes=0 after a full theory stage. If the team could prove T2 for one example in 4 weeks, why didn't they prove it in the theory stage? The conditional override rewards failure with more time. It should be: demonstrate the T2 proof *first*, then discuss continuation — not the reverse.

**Challenge 3: V4 is too high.**

The Auditor scores Value at 4/10. The depth_check.md committee scored it 5/10 *before* theory_bytes=0. After the theory stage produced nothing, the value proposition is weaker, not stronger — the unique formal guarantees that justify the tool over LLMs require T2, which doesn't exist. V3 is the correct score.

### Skeptic → Synthesizer

**Challenge 1: The salvage paths assume competence that theory_bytes=0 doesn't support.**

The Synthesizer assigns P(success)=85% for the benchmark suite and 65% for the dynamic-only auditor. These are not trivial projects — the benchmark suite requires faithfully reproducing conservation bugs from Fortran/C++ codebases in Python, which verification_signoff.md §Non-Circularity Assessment flags as "another form of self-construction." The dynamic-only auditor requires implementing the ablation-based localization pipeline, which is ~5K LoC of non-trivial code. A team that produced 0 bytes in the theory stage may not deliver 3–5K LoC of working code.

**Challenge 2: "Core diamond" rhetoric inflates T2's current status.**

T2 is a *conjecture*, not a diamond. It has never been tested on a single example. The Synthesizer correctly identifies T2 as the unique capability — but calling an unvalidated conjecture a "diamond" imports confidence the evidence doesn't support. Diamonds are found by mining; this diamond hasn't been found yet.

**Challenge 3: The CONTINUE-REDUCED verdict is CONTINUE in disguise.**

By recommending salvage paths A–D, the Synthesizer effectively recommends continuing work on the project under a different name. Paths C and D are 4–6 months of effort with 30–65% success probability. This is a new research project, not a "salvage." If the full proposal is ABANDON, the reduced proposals should be evaluated as fresh proposals with their own gate reviews, not grandfathered in.

### Auditor → Skeptic

**Challenge 1: V3 undersells the Wan et al. anecdote and the paradigm novelty.**

The Skeptic assigns V3, but the depth_check.md committee — including the original Skeptic — scored Value 5/10 specifically because "The Wan et al. anecdote is genuinely compelling. The causal localization capability and obstruction detection are qualitatively new capabilities." The physics-aware program analysis paradigm *is* genuinely novel. V3 underweights the paradigm contribution, even granting that theory_bytes=0 reduces confidence in delivering it.

**Counter:** The Skeptic accepts that the *paradigm idea* has value. But ideas without execution are worth nearly nothing. The value score should reflect deliverable value, not aspirational value. After theory_bytes=0, deliverable value is near zero.

**Challenge 2: F2 double-counts the theory_bytes=0 signal.**

Feasibility (F) measures whether the project *can* be done, not whether it *has been* done. theory_bytes=0 is a signal about team execution, which should lower confidence in all dimensions, but shouldn't reduce a feasibility score to 2/10 for a project where the core computations are genuinely tractable (CPU=6). The Auditor would set F3.

**Counter:** The Skeptic defines feasibility as "probability that this specific project, by this specific team, in this specific timeline, delivers publishable results." By that definition, theory_bytes=0 is devastating evidence and F2 is justified. Abstract tractability is irrelevant if the team doesn't execute.

### Auditor → Synthesizer

**Agreement:** The benchmark suite (Path A) has genuine standalone value. A curated conservation-bug benchmark with ground-truth annotations would be the first of its kind. The Auditor endorses Path A unconditionally.

**Disagreement on Path D (T2 standalone):** P(success)=30% over 6 months is a poor expected-value bet for a standalone math paper. The result might be trivial, exponential, or wrong. Without a tool to contextualize it, a standalone T2 proof in Numerische Mathematik has modest impact (~15 citations). The Auditor recommends Path D only if someone is genuinely excited about the math problem for its own sake — not as a project continuation strategy.

### Synthesizer → Skeptic

**Challenge 1: The Skeptic ignores optionality.**

theory_bytes=0 is bad, but it does not erase the intellectual content of the ideation stage. The ideation produced a precise T2 statement (final_approach.md §5.2), a well-defined architecture, and honest kill gates. These have option value: another team, or the same team in a better phase, could pick them up. The Skeptic scores as if the idea is worthless; the Synthesizer scores the idea's option value separately from this team's execution.

**Counter:** The Skeptic evaluates proposals, not ideas. Option value for future hypothetical teams is not a factor in the current gate decision.

**Challenge 2: P(best-paper) ≈ 4% is too low if T2 works.**

The depth_check.md committee estimated P(best-paper)=4% *conditional on everything working*. If T2 yields an elegant efficient reduction, the "impossible bridge" narrative + killer demo (Wan et al. reproduction) + formal obstruction certificate is genuinely a best-paper candidate at OOPSLA. The Skeptic's unconditional estimate of ~4% (actually ~2% after adjusting for theory_bytes=0) treats "T2 works" as essentially impossible.

**Counter:** P(T2 works elegantly) × P(everything else works) × P(OOPSLA best-paper | all that) ≈ 0.20 × 0.19 × 0.15 ≈ 0.6%. The Skeptic's 4% is already generous.

### Synthesizer → Auditor

**Agreement:** The compound failure model is well-constructed and the ~81% figure is the single most important number in this evaluation.

**Disagreement on salvage ranking:** The Auditor doesn't rank salvage paths. The Synthesizer argues this is a missed opportunity — the purpose of the gate review is not just to say "stop" but to say "stop *here* and go *there*." The benchmark suite has 85% success probability and fills a genuine community gap. It should be an explicit recommendation, not an afterthought.

---

## Phase 3: Synthesis

### Score Resolution

| Dimension | Skeptic | Auditor | Synthesizer | Synthesized | Justification |
|-----------|---------|---------|-------------|-------------|---------------|
| **Value (V)** | 3 | 4 | 4 | **3** | theory_bytes=0 makes the unique formal capabilities (obstruction, provenance) undeliverable. Without T2, the value proposition reduces to "LLM-inferior dynamic checking for 300 users." The paradigm idea retains some value, but ideas without execution are V3. |
| **Difficulty (D)** | 5 | 5 | 5 | **5** | Unanimous. Genuine difficulty exists in extraction and T2, but individual components are known techniques. Integration risk is real but unexceptional for a systems project. |
| **Best-Paper (BP)** | 2 | 4 | 4 | **3** | T2 is the only best-paper-caliber result and it has zero bytes of proof. T1 is bookkeeping. T3 is a definition. Without T2, the paper lands at "practical tool paper" at best — but the practical tool also doesn't exist. Split the difference at 3: the idea has potential that the execution hasn't realized. |
| **CPU (CPU)** | 6 | 6 | 6 | **6** | Unanimous. Core symbolic computations are laptop-tractable for typical parameters (k≤5, p≤4). This is the proposal's one unambiguous strength. |
| **Feasibility (F)** | 2 | 3 | 3 | **2** | theory_bytes=0 is the strongest possible negative feasibility signal. The team had a dedicated theory stage and produced no theory. The Auditor's F3 treats feasibility abstractly; the Skeptic's F2 treats it as a prediction about *this* team's ability to deliver. Given the evidence, F2 is the correct prediction. |

### Points of Agreement (Unanimous)

1. **theory_bytes = 0 is a critical red flag.** All three evaluators weight this as the dominant signal. The theory stage was the lowest-risk opportunity to validate the mathematical foundations, and it produced nothing.

2. **T2 is the unique contribution.** The obstruction criterion is the only capability that LLMs cannot replicate, that SymPy+100 lines cannot approximate, and that justifies the formal-methods approach. Without T2, the proposal is a moderately useful dynamic monitoring tool competing with existing domain-specific monitors.

3. **The full 71K LoC tool is not viable.** The depth_check committee's own analysis puts P(abandon after Phase 1) at 40% *before* theory_bytes=0. After, it is substantially higher. No evaluator recommends proceeding with the full implementation plan.

4. **Coverage estimates are fabricated.** The 40–60% figure in final_approach.md §4.1 and the 20–40% in the problem statement are both unvalidated guesses. Nobody knows what the actual liftable fragment covers because nobody has tried.

5. **The benchmark suite has standalone value.** A curated conservation-bug benchmark with ground-truth annotations would be a genuine community contribution regardless of the tool's fate.

6. **LLM competition is an existential threat.** The ~70% overlap on simple cases (final_approach.md §9) means ConservationLint's defensible moat is only as wide as T2's capabilities — which are currently zero bytes wide.

### Points of Disagreement (Resolved)

| Disagreement | Skeptic | Auditor | Synthesizer | Resolution |
|-------------|---------|---------|-------------|------------|
| **Value score** | V3 | V4 | V4 | **V3.** Auditor and Synthesizer argue the paradigm idea has intrinsic value. Skeptic argues undelivered ideas have near-zero value. Resolved in Skeptic's favor: the post-theory gate evaluates deliverables, not aspirations. |
| **Feasibility score** | F2 | F3 | F3 | **F2.** Auditor argues abstract tractability supports F3. Skeptic argues team-specific execution evidence (theory_bytes=0) demands F2. Resolved in Skeptic's favor: feasibility is a prediction about this project, not a statement about mathematical possibility. |
| **Whether salvage paths constitute "CONTINUE"** | Yes (disguised) | Neutral | No (genuinely reduced) | **Resolved: salvage paths are SEPARATE proposals**, not continuations. Path A (benchmark) and Path B (survey) can proceed without gate review. Paths C and D require fresh evaluation. |
| **T1's potential** | Bookkeeping | Bookkeeping | Possible crown jewel | **Bookkeeping.** Two evaluators agree T1 is known BCH with labels. The Synthesizer's conditional ("if mixed-order proof yields non-trivial structural insights") is untestable because no proof has been attempted. Default to the majority assessment: T1 is engineering scaffolding. |

### Fatal Flaws (Confirmed)

These flaws survived adversarial cross-critique and are confirmed as project-terminal:

**FF1: theory_bytes = 0 after theory stage (TERMINAL).** The theory stage's sole purpose was to produce mathematical content. It produced none. This is not a partial failure; it is a complete failure of the stage's objective. The proposal's crown jewel (T2) remains an untested conjecture. The engineering scaffolding (T1) remains a described-but-unformalized specification. No binding condition or conditional override can rescue a project that failed to execute its designated stage.

**FF2: The unique value proposition requires an unproven theorem (TERMINAL).** ConservationLint's defensible moat against LLMs, SymPy scripts, and existing monitors is T2's obstruction criterion. Without T2, the tool provides: (a) causal localization via provenance tags (T1), which is engineering bookkeeping; (b) dynamic conservation monitoring, which GROMACS and LAMMPS already do; (c) formal guarantees on a liftable fragment of unknown and possibly tiny size. None of these individually or collectively justify a top-venue publication.

**FF3: Three planning documents, zero artifacts (STRUCTURAL).** The project has produced 86.7KB of planning (final_approach.md + depth_check.md + verification_signoff.md) and 0 bytes of theory, code, proofs, benchmarks, or prototypes. This is not a research project in the theory stage; it is a planning exercise that has not transitioned to research. The depth_check committee itself noted: "SLAM shipped with Windows and verified millions of lines; Herbie is used by thousands. ConservationLint is a markdown file" (depth_check.md §3).

**FF4: Compound gate-failure probability ~81–86% (QUANTITATIVE).** Even accepting the proposal's own kill gates at face value, the probability of clearing all five is 14–19%. Adjusting G2 upward per the Skeptic's cross-critique (P(G2 fail) = 35–40%), the compound success probability drops to ~14%. This means that for every 7 parallel universes where this project continues, only 1 produces a publishable result — and that result has only a 35% chance of landing at a top venue (depth_check.md §Risk Assessment).

---

## Phase 4: Final Verdict

### Composite Score: 3.4/10

**Breakdown: V3 / D5 / BP3 / CPU6 / F2**

Weighted composite: (V×0.25 + D×0.20 + BP×0.25 + CPU×0.10 + F×0.20) = (0.75 + 1.0 + 0.75 + 0.6 + 0.4) = **3.5/10** (weighted) or **3.8/10** (unweighted average). We adopt the geometric-mean-influenced floor of **3.4/10** because the F2 score represents a binding constraint — no project with near-zero execution feasibility can exceed its feasibility bound regardless of how good the other dimensions look.

For calibration: the depth_check committee scored 5.75/10 *before* the theory stage. The final_approach self-scored 6.0/10 *before* the theory stage. Both of these were conditional on the theory stage producing results. It produced nothing. The 2.35–2.6 point drop from the pre-theory scores reflects the information content of theory_bytes=0.

### VERDICT: ABANDON

All three independent evaluators recommended ABANDON or ABANDON-to-reduced. The adversarial cross-critique did not surface any evidence sufficient to override this consensus. The theory stage's complete failure to produce mathematical content eliminates the proposal's unique value proposition, confirms the feasibility concerns, and provides strong evidence that the team cannot deliver the formal results the paper requires.

### Binding Conditions

Since the verdict is ABANDON, we issue salvage recommendations rather than continuation conditions.

**Immediate (no gate review required):**

1. **Benchmark suite (Path A).** Curate 15–20 conservation-bug kernels with ground-truth annotations, sourcing ≥5 from real bug trackers (LAMMPS, GROMACS, CESM). ~3K LoC, ~3 months, target JOSS. P(success) ≈ 80%. This fills a genuine community gap and is independent of all theoretical results.

2. **Survey/SoK paper (Path B).** Write "Conservation in Numerical Simulation Code: Theory, Practice, and Tooling Gaps." Map the space between geometric numerical integration, software verification, and scientific computing. Minimal code, ~3 months, target ICSE/FSE SoK track. P(success) ≈ 70%.

**Conditional (requires separate evaluation):**

3. **T2 standalone proof (Path D).** Attempt to prove the obstruction criterion for Störmer-Verlet splitting of a rotationally symmetric Hamiltonian (the simplest non-trivial case). Pure math, no code, ~2 months for this one example. If the proof reveals genuine structure (not just Tarski-Seidenberg), evaluate continuation to a math paper (Numerische Mathematik / BIT). If the proof is trivial or the efficient reduction fails, stop. **This is the only path that could resurrect the full proposal**, but it should be pursued as speculative research, not as a project commitment.

4. **Dynamic-only auditor (Path C).** Implement conservation-specific violation detection (d/dt C_v along trajectories) + ablation-based localization. ~5K LoC, ~4 months. This produces a useful but incremental tool — it does what GROMACS monitoring does, but with causal localization. P(top-venue) is low (~10%); target SciPy conference or JOSS. **Only pursue if the team has genuine enthusiasm for building a practical tool rather than proving theorems.**

### Kill Probability Assessment

| Outcome | Probability | Basis |
|---------|------------|-------|
| P(full proposal abandoned) | **95%** | theory_bytes=0, 81% compound gate failure, 3/3 evaluators recommend ABANDON |
| P(any publication from full tool) | **8%** | Requires clearing all 5 gates after a failed theory stage; generously assumes 4-week recovery is possible |
| P(top-venue publication) | **5%** | P(all gates clear) × P(OOPSLA accept \| gates clear) ≈ 0.14 × 0.35 ≈ 5% |
| P(best-paper) | **<1%** | P(top-venue) × P(best-paper \| accept) × P(T2 elegant) ≈ 0.05 × 0.10 × 0.20 ≈ 0.1% |
| P(any publication from salvage) | **70%** | Benchmark suite (80% × target JOSS) + survey (70% × target SoK) — at least one lands |
| P(T2 standalone proof succeeds) | **25%** | Pure math with no execution dependencies, but the team's track record (theory_bytes=0) is not encouraging |

### Dissenting Opinions

**Synthesizer (partial dissent):** "The verdict is correct for the full proposal, but the report undersells the intellectual contribution of the ideation stage. The precise T2 statement, the hybrid architecture, the honest kill gates, and the two-tier design are genuine intellectual assets that could be picked up by a stronger team or at a better time. ABANDON does not mean WORTHLESS. The idea deserves a future attempt; this attempt has failed."

**Auditor (minor dissent on F score):** "F2 is defensible given theory_bytes=0, but it conflates *team execution failure* with *project infeasibility*. The core computations are tractable (CPU=6), the architecture is sound, and the kill gates are well-designed. A different team — or this team with a 4-week math sprint — could achieve F4. I accept the F2 score for *this* evaluation but note that it reflects a team-specific signal, not a structural impossibility."

**Skeptic (no dissent):** "3.4/10 is, if anything, generous. The project has produced 87KB of planning documents and 0 bytes of deliverable content. The next allocation of time should go to Path A (benchmark suite) or Path D (T2 proof on one example), not to more planning."

---

*Post-theory verification gate. 3-evaluator team + adversarial cross-critique. Composite 3.4/10. VERDICT: ABANDON.*

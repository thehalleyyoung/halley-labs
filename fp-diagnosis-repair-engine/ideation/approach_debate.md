# Approach Debate: Penumbra (fp-diagnosis-repair-engine)

> **Document type:** Multi-expert adversarial assessment  
> **Approaches under review:** A (EAG-First Foundations), B (Diagnosis-Repair Pipeline), C (Empirical Science)  
> **Assessors:** Mathematician (math depth), Engineer (difficulty), Skeptic (adversarial critique)

---

## Approach A: EAG-First Foundations

### Math Depth Assessment (Mathematician)

**Load-Bearing: 6/10.** T1 soundness is load-bearing — without it, the EAG is just a visualization. The tightness ratio τ is the most interesting mathematical object across this entire approach: it transforms T1 from a vacuously loose bound into a diagnostic signal. However, the tool works without τ. Shadow values alone diagnose. The restricted T2 proof for series-parallel EAGs is partially load-bearing. T3 (exhaustive case analysis over IEEE 754) is formalized bookkeeping — classifiers work identically whether or not T3 is stated as a theorem.

**Achievability breakdown:**

| Component | Probability | Assessment |
|-----------|------------|------------|
| T1 basic soundness | 95% | Routine |
| Tightness ratio τ (series-parallel EAGs) | 70% | Genuinely hard but tractable |
| Tightness ratio τ (bounded-treewidth EAGs) | 30% | Moonshot |
| Restricted T2 (series-parallel) | 60% | Tractable |
| T3 (IEEE 754 case analysis) | 99% | Guaranteed |

**Overall Achievability: 5/10.**

**Novelty: 7/10.** The tightness ratio τ is the novel kernel — no prior work characterizes tightness as a function of DAG topology. T3 is not novel (formalizes Higham's textbook). T1 basic soundness is standard.

**Reviewer Reception: 6/10.** PL reviewers will appreciate the tightness characterization *if τ has practical diagnostic value demonstrated empirically*. If τ ≈ 0 on real programs, the math is ornamental. PLDI reviewers will ask: "Why not Fluctuat, which gives soundness for free?"

### Difficulty Assessment (Engineer)

**Hard subproblems:**

1. **Tightness characterization** — the only genuinely *research-hard* subproblem across all three approaches. Novel analysis at the intersection of forward error analysis and graph-theoretic path enumeration. No existing template to follow.
2. **MPFR replay fidelity** — vast correctness surface. Engineering-hard, not novel-hard. Every special-function corner case is a potential silent failure.
3. **Sensitivity estimation at scale** — systems work with known patterns. Hard to get fast, not hard to get correct.

**Architecture Risk: High.** The streaming pipeline is tightly coupled. Shadow engine fidelity failures cascade to all downstream components. There is no cheap fallback — if MPFR replay is unreliable, the entire EAG is unreliable.

**Failure Modes:**
- Tightness theorem yields vacuously loose bounds (τ ≈ 0 always), rendering the core mathematical contribution worthless.
- MPFR fidelity bugs discovered late in development, requiring architectural rework.

**Novel Engineering: ~38%.** Scores: Algorithmic 8/10, Systems 7/10, Integration 6/10.

### Adversarial Critique (Skeptic)

**Fatal Flaws:**

1. **"New program representation" is smoke and mirrors.** Fluctuat's zonotopes already reify per-operation error contributions with formal soundness. What algorithm does the EAG enable that querying Satire's shadow values doesn't? The burden is on the authors to demonstrate a qualitative capability gap, not just a representational difference.
2. **Tightness characterization may be vacuous for real programs** (τ ≈ 0). If every real-world EAG has treewidth exceeding the tractable regime, the theorem is a mathematical curiosity with no practical teeth.
3. **First-order assumption breaks precisely when users need it** — ill-conditioned problems are exactly the cases where FP diagnosis matters most, and exactly where first-order error models fail.

**Killer Question:** *"How is this different from running Verificarlo, collecting per-operation error, and storing results in a graph database?"*

**"Just X" Defense:** "Just Satire's shadow values stored in a DAG." To preempt: demonstrate an algorithm on the EAG that is provably better than the equivalent algorithm on raw shadow data.

**Portfolio Overlap:** HIGH overlap with `fp-condition-flow-engine` (area-079). Moderate overlap with `algebraic-repair-calculus`. A reviewer familiar with the portfolio would notice.

**Probabilities:** P(top venue) = 35%, P(best paper) = 8%, P(abandoned) = 25%.

### Cross-Expert Disagreements

**τ: Novel kernel or vacuous curiosity?** The Mathematician identifies τ as the novel kernel (7/10 novelty) and the most interesting mathematical object. The Skeptic warns that τ ≈ 0 on real programs would make the math ornamental. The Engineer agrees τ is the only genuinely research-hard subproblem — but flags a 30% moonshot probability for bounded-treewidth generalization. All three experts converge on τ as the swing factor, but disagree on whether to bet on it: the Mathematician says yes (conditional on series-parallel restriction), the Skeptic says the conditional may never hold in practice, and the Engineer says it's the only place genuine research happens.

**Architecture coupling.** The Engineer flags high architecture risk from tight coupling. Neither the Mathematician nor the Skeptic engages with this — the Mathematician evaluates math in isolation, and the Skeptic attacks claims rather than implementation. This is a blind spot: if the architecture is fragile, the math never gets validated.

**EAG vs. existing tools.** The Skeptic's killer question ("How is this different from Verificarlo + graph database?") directly challenges the Mathematician's novelty rating of 7/10. If the answer is "it's not," then τ is the *only* contribution, and the 6/10 load-bearing score collapses to τ alone.

### Survival Verdict

Approach A survives **only if** τ for series-parallel EAGs is achievable (70% per Mathematician) **and** τ ≈ 0 is falsified empirically on real programs. Both conditions must hold. The Skeptic's killer question must be answered head-on with a concrete algorithm demonstration. Without these, A is Verificarlo-with-extra-steps.

---

## Approach B: Diagnosis-Repair Pipeline

### Math Depth Assessment (Mathematician)

**Load-Bearing: 8/10.** T4 is the most load-bearing math across all approaches — it directly answers "does diagnosis-guided repair outperform blind search?" Without T4, a reviewer asks "why not try all rewrites?" C1 is load-bearing for certification claims. Per-pattern lower bounds are moderately load-bearing.

**Achievability: 8/10.**

| Component | Probability | Assessment |
|-----------|------------|------------|
| T4 submodularity (monotone DAGs) | 85% | Sweet spot: non-trivial but tractable |
| C1 certification | 95% | Routine |
| Per-pattern lower bounds | 90% | Classical, restated in context |

**Novelty: 4/10.** T4 is a clean application of known optimization theory to a new domain. Novel *in context*, not *in technique*. C1 is standard. Per-pattern bounds are classical with new packaging. This is applied math, not new math.

**Reviewer Reception: 7/10.** SC/FSE reviewers will love this — math proportionate to claims. Best math-to-impact ratio of all three approaches. PL reviewers would find it underwhelming.

### Difficulty Assessment (Engineer)

**Hard subproblems:**

1. **Repair synthesis for real code via LibCST** — the hardest engineering subproblem. No existing library handles semantics-preserving FP transformations at the AST level. Every special case is a potential correctness bug in the repair.
2. **Finding ≥5 real pipeline-level bugs** — pure empirical risk. The bugs either exist and are findable, or they don't. No amount of engineering compensates.
3. **LAPACK wrapping** — integration-hard but not novel. Well-understood problem with tedious solutions.

**Architecture Risk: Moderate.** Adapter-per-codebase design is modular and allows independent development. The deeper risk: real bugs don't fit the 5-category taxonomy, rendering the pattern library brittle.

**Failure Modes:**
- Only shallow bugs exist in the target codebases (i.e., all Herbie-solvable), killing the contribution story.
- Generated patches are unreviewable — technically correct but no human would merge them.

**Novel Engineering: ~22%.** Difficulty is inflated through breadth (6 codebases × 30 patterns × 20 LAPACK functions), but each individual unit follows established patterns. Scores: Algorithmic 4/10, Systems 7/10, Integration 8/10.

### Adversarial Critique (Skeptic)

**Fatal Flaws:**

1. **BC4 is existential risk.** Most SciPy precision issues may be expression-level (Herbie-solvable). If pipeline-level bugs requiring cross-function repair don't exist in sufficient quantity, the entire approach has no evaluation story.
2. **The repair synthesizer is a pattern library (30 hand-coded patterns), not synthesis.** Calling 30 hand-coded rewrite rules "synthesis" is fundamentally misleading. The system is brittle — pattern 31 requires a code change, not a configuration change.
3. **Certification is overpromised.** C1 doesn't apply to LAPACK black boxes, where the most consequential errors live. The certification boundary stops exactly where it matters most.

**Killer Question:** *"You found N bugs. Herbie can repair M of them. Satire can localize all of them. What does your tool provide that `satire | herbie` doesn't?"*

**"Just X" Defense:** "Just Satire + Herbie + a taxonomy lookup table." To preempt: demonstrate ≥3 bugs where Satire localizes wrong AND Herbie can't repair across function boundaries AND Penumbra uniquely succeeds.

**Portfolio Overlap:** Moderate with `ml-pipeline-selfheal`, `dp-verify-repair`.

**Probabilities:** P(top venue) = 50%, P(best paper) = 5%, P(abandoned) = 15%.

### Cross-Expert Disagreements

**Difficulty: real or inflated?** The Engineer scores novel engineering at only 22%, arguing difficulty is inflated through breadth, not depth. The Mathematician, by contrast, gives 8/10 achievability — the math is tractable. These are consistent but reveal a tension: the project is *easy to do* (high achievability, low novel engineering) but *hard to make matter* (BC4 existential risk). The Skeptic crystallizes this: the work is feasible but possibly pointless.

**T4's value.** The Mathematician calls T4 the most load-bearing math across all approaches. The Skeptic implicitly dismisses it — the killer question is about *bugs found*, not *optimization guarantees*. For the Skeptic, T4 is moot if BC4 fails. The Engineer doesn't engage with T4 at all, suggesting it's not an engineering bottleneck. T4 matters to reviewers, not to the tool.

**Pattern library as synthesis.** The Skeptic calls the 30-pattern library "fundamentally brittle" and "not synthesis." The Engineer identifies LibCST-based repair as the hardest engineering subproblem but doesn't challenge the framing. The Mathematician ignores implementation entirely. This disagreement matters: if reviewers agree with the Skeptic that 30 patterns ≠ synthesis, the paper's framing collapses regardless of T4.

**Venue mismatch.** The Mathematician says SC/FSE reviewers love this but PL reviewers find it underwhelming. The Skeptic gives it the highest P(top venue) at 50%. Both agree B has the best chance — but for different reasons (Mathematician: proportionate math; Skeptic: lowest abandonment risk).

### Survival Verdict

Approach B survives **if** BC4 holds — i.e., ≥3 pipeline-level bugs exist that require cross-function repair beyond Herbie's scope. This is an empirical question that can be answered early with a focused scouting effort. B has the best risk profile but the least ambitious ceiling. It is the "safe bet that might bore reviewers."

---

## Approach C: Empirical Science

### Math Depth Assessment (Mathematician)

**Load-Bearing: 3/10.** Almost none of the math is load-bearing for a software artifact. Treewidth characterization is a measurement, not a mechanism. The tool works identically whether treewidth is 3 or 300. Math exists to make the paper publishable as science, not to make software work.

**Achievability: 7/10.**

| Component | Probability (publishable) | Probability (useful) |
|-----------|--------------------------|---------------------|
| Treewidth measurement | 90% | N/A (it's data) |
| Bounded-treewidth proof (restricted) | 75% | — |
| Bounded-treewidth proof (general) | 40% | — |
| Frequency model | 70% | 30% |
| Propagation decay law | 85% (empirical) | 25% (as theorem) |

**Novelty: 8/10.** Nobody has measured treewidth of FP error-flow graphs. Nobody has computed error-pattern frequency distributions across codebases. This is novel *data*, not novel *theorems*.

**Reviewer Reception: 5/10.** Empirical reviewers value measurements. Theory reviewers see dressed-up statistics. Significant risk of "so what?" reaction.

### Difficulty Assessment (Engineer)

**Hard subproblems:**

1. **Treewidth computation at scale** — borrowed difficulty from PACE solvers. The algorithmic challenge is solved; the engineering challenge is plumbing.
2. **Cross-codebase normalization** — methodology-hard, not algorithm-hard. Requires careful experimental design, not clever code.
3. **Measurement validity** — the MPFR fidelity problem returns, now with higher stakes since measurements *are* the contribution.

**Architecture Risk: Low.** Each experimental stage is independently testable. Failure in one measurement campaign doesn't invalidate others.

**Failure Modes:**
- Uninteresting findings: "It depends on the codebase." Reviewers shrug.
- Insufficient novelty: "You measured things. Cool. Now what?"

**Novel Engineering: ~18%.** Scores: Algorithmic 5/10, Systems 5/10, Integration 5/10.

### Adversarial Critique (Skeptic)

**Fatal Flaws:**

1. **"Measurements always produce data" = feasibility of the trivially publishable.** The most likely outcomes are unsurprising: sequential pipelines have low treewidth (because sequential programs are sequential); error patterns match Higham's textbook taxonomy (because Higham wrote the textbook). The approach optimizes for *not failing* rather than *succeeding*.
2. **No actionable contribution.** C measures things but doesn't build, fix, or prove anything. A reader finishes the paper and cannot do anything they couldn't do before.
3. **LAPACK black boxes mean measurements miss the most error-amplifying operations.** The same blind spot as B, but worse — at least B tries to work around LAPACK. C just measures around it.

**Killer Question:** *"Treewidth is low in sequential pipelines. Sequential programs are sequential. Did you actually discover anything?"*

**Probabilities:** P(top venue) = 40%, P(best paper) = 2%, P(abandoned) = 10%.

**Verdict:** Kill C as standalone.

### Cross-Expert Disagreements

**Novelty: high or trivial?** The Mathematician gives 8/10 novelty — nobody has measured these things before. The Skeptic counters that novel measurements of unsurprising quantities are trivially novel. The Engineer doesn't engage with novelty at all (18% novel engineering). This is the core tension: C is novel in the sense that the data doesn't exist, but potentially not novel in the sense that the data is predictable.

**Risk profile.** The Engineer gives the lowest architecture risk and the Skeptic gives the lowest abandonment probability (10%). Both agree C is the safest path. But they disagree on whether safety is a virtue here: the Engineer sees low risk as low ambition; the Skeptic sees low risk as low ceiling. The Mathematician is blunter — math exists to make the paper publishable, not to make software work.

**Standalone viability.** All three experts converge: C should not be a standalone approach. The Skeptic says "kill it." The Mathematician says the math isn't load-bearing. The Engineer says the novel engineering is minimal. This is the closest thing to expert consensus in the entire debate.

### Survival Verdict

Approach C does **not** survive as standalone. Its components (treewidth measurements, frequency distributions, decay law) survive as *empirical contributions within another approach*. Treewidth data is novel and publishable as supporting evidence; it is not a paper.

---

## Cross-Approach Synthesis and Recommendations

### Score Aggregation

| Dimension | Approach A | Approach B | Approach C |
|-----------|-----------|-----------|-----------|
| **Math Load-Bearing** (Mathematician) | 6/10 | **8/10** | 3/10 |
| **Math Achievability** (Mathematician) | 5/10 | **8/10** | 7/10 |
| **Math Novelty** (Mathematician) | 7/10 | 4/10 | **8/10** |
| **Reviewer Reception** (Mathematician) | 6/10 | **7/10** | 5/10 |
| **Novel Engineering** (Engineer) | **38%** | 22% | 18% |
| **Architecture Risk** (Engineer) | High | Moderate | **Low** |
| **Algorithmic Difficulty** (Engineer) | **8/10** | 4/10 | 5/10 |
| **P(top venue)** (Skeptic) | 35% | **50%** | 40% |
| **P(best paper)** (Skeptic) | **8%** | 5% | 2% |
| **P(abandoned)** (Skeptic) | 25% | **15%** | **10%** |

**Pareto analysis:** No approach dominates. B leads on load-bearing math, achievability, reception, and P(top venue). A leads on novelty, novel engineering, and best-paper upside. C leads on nothing except low abandonment risk and novel data. B is the Pareto-efficient safe choice; A is the Pareto-efficient ambitious choice; C is dominated except as a data source.

### Consensus Recommendations

All three experts agree on the following:

1. **Kill C as standalone.** Unanimous. Repurpose its strongest components (treewidth measurements, frequency data) as empirical contributions within another framing.
2. **B has the best risk-adjusted return.** Highest P(top venue), lowest abandonment risk, most load-bearing math, highest achievability. If forced to pick one approach, pick B.
3. **A's τ is the highest-ceiling contribution but highest-risk.** If τ works empirically (non-vacuous on real programs), it is the most publishable single result. If τ ≈ 0, A collapses.
4. **BC4 must be scouted early.** B lives or dies on whether pipeline-level bugs exist beyond Herbie's reach. This is an empirical question answerable in weeks, not months.
5. **A and B share ~70% infrastructure** (Engineer). The marginal cost of attempting both is low.

### The Hybrid Debate

The experts are split on the hybrid strategy (A's framing + B's evaluation + C's data):

**For the hybrid (Engineer + Mathematician):**
- The Engineer observes A and B share ~70% infrastructure, making the hybrid "strictly dominant" from an engineering perspective: build once, publish twice.
- The Mathematician's recommended portfolio explicitly combines elements: "T1 routine soundness, τ for series-parallel only, T4 submodularity, C1 certification, treewidth measurements as novel data, decay law as empirical conjecture. No dead weight, no moonshots."
- The marginal cost of adding C's measurements to either A or B is minimal — it's data collection on infrastructure you've already built.

**Against the hybrid (Skeptic):**
- "The hybrid risks worst of all worlds — too broad for PLDI, too theoretical for SC, too tool-focused for ICSE." The Skeptic's concern is not engineering feasibility but *narrative coherence*. A paper that claims to introduce a representation (A), build a repair tool (B), and measure empirical properties (C) has no clear identity.
- "Pick one framing and commit." Reviewers evaluate papers, not research programs. Breadth that helps a research program hurts a paper submission.

**Resolution:** The Skeptic is right about the *paper* and wrong about the *project*. The hybrid is the correct *research strategy* — build shared infrastructure, pursue both A and B, collect C's data along the way. But each *submission* must pick a single framing:

- **Submission 1 (SC/FSE, safe):** B's framing. Diagnosis-repair pipeline with T4 submodularity, C1 certification, and evaluation on real bugs. Include treewidth measurements from C as a "structural analysis of error-flow in scientific pipelines" subsection. This is a tool paper with proportionate math.
- **Submission 2 (PLDI, ambitious):** A's framing. EAG with τ tightness characterization for series-parallel graphs. Use B's bug-finding results as empirical validation that the EAG representation enables better diagnosis. This is a foundations paper with empirical grounding.
- **Fallback (ICSE/empirical track):** If both fail, C's measurements are independently publishable as an empirical study, using infrastructure already built for A and B.

### Recommended Portfolio: What to Carry Forward

**Must-build (shared infrastructure):**
- EAG construction and MPFR shadow replay engine (serves A and B)
- Pattern taxonomy and detection framework (serves B and C)
- Benchmark suite across ≥3 scientific codebases (serves all)

**Must-prove (math):**
- T1 basic soundness (routine, 95%) — required for any EAG-based claim
- T4 submodularity on monotone DAGs (85%) — B's core theorem
- C1 certification (routine, 95%) — B's certification claim

**Should-attempt (high-value, tractable):**
- τ tightness ratio for series-parallel EAGs (70%) — A's novel kernel, paper-making if it works
- Treewidth measurements across codebases (90%) — novel data, low cost

**Should-not-attempt (moonshots):**
- τ for bounded-treewidth EAGs (30%) — the Mathematician explicitly flags this as a moonshot
- Propagation decay law as a theorem (25%) — publish as empirical conjecture instead
- Bounded-treewidth proof in general (40%) — insufficient probability for the effort

**Must-answer-first (gating questions):**
1. **BC4 scout:** Do ≥3 pipeline-level bugs exist in SciPy/Astropy/scikit-learn that require cross-function repair? (Gates B's entire evaluation.)
2. **τ empirical check:** On 5 real scientific pipelines, is τ > 0.1 for any of them? (Gates A's tightness contribution.)
3. **Satire|Herbie baseline:** What fraction of known FP bugs can `satire | herbie` already handle? (Gates both A and B's novelty claims.)

If BC4 scout returns ≥3 bugs and τ > 0.1 on real programs, pursue both submissions. If only BC4 holds, commit to B. If only τ holds, commit to A. If neither holds, fall back to C's empirical study and reframe the project.

### Final Verdict

**Primary path:** B (Diagnosis-Repair Pipeline), targeting SC or FSE.  
**Parallel high-risk/high-reward:** A's τ characterization, targeting PLDI.  
**Absorbed into both:** C's treewidth measurements and frequency data.  
**Gate everything on:** BC4 scouting and τ empirical validation, both achievable within the first 4 weeks.

# Depth Check: Bounded-Rational Usability Regression Testing

**Slug:** `bounded-rational-usability-oracle`
**Evaluator:** Impartial best-paper committee chair
**Method:** Three-expert adversarial panel (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with independent proposals, cross-critiques, and synthesis
**Date:** 2026-03-04

---

## Scoring Summary

| Axis | Score | Verdict |
|------|-------|---------|
| 1. Extreme and Obvious Value | **6/10** | Real pain, compelling framing, but core validity is deferred and unvalidated |
| 2. Genuine Difficulty as Software Artifact | **7/10** | Legitimately hard; bisimulation and cost-algebra soundness are genuinely novel; ~40% is standard engineering |
| 3. Best-Paper Potential | **5/10** | Framing is excellent but TV bounds are loose, evaluation is circular, and no empirical results exist |
| 4. Laptop CPU + No Humans | **9/10** | Principled CPU-native design; zero-human eval plan is clean; minor gap on accessibility tree extraction |
| **TOTAL** | **27/40** | **AMENDMENTS REQUIRED** (Axes 1, 3 below threshold of 7) |

---

## Axis 1: EXTREME AND OBVIOUS VALUE — 6/10

### What works

The problem is real. Usability regressions escape into production because no automated oracle exists analogous to a unit test suite. The **consistency-oracle framing** — claiming only relative ordering preservation, not absolute prediction — is the proposal's strongest conceptual contribution. This scoping move is correct, well-argued, and immediately makes the system practically viable despite imperfect calibration. The analogy to compiler optimizations preserving asymptotic complexity (not predicting wall-clock time) is precise and persuasive.

The CI/CD integration angle is strong. A tool that runs on every PR and produces a quantitative cognitive cost diff addresses a genuine gap in the development toolchain between accessibility linters (axe-core, Pa11y) and manual usability studies.

### What fails

**The core validity is deferred.** The entire system's utility depends on the claim that bounded-rational information-theoretic cost orderings correlate with human-perceived usability regressions. The proposal explicitly defers this validation to "follow-up work." This is the single most damaging gap: a consistency oracle must be consistent *with something external*. If the model's orderings don't match human experience, the system produces confident but meaningless verdicts.

**The evaluation plan is circular.** The system defines "regression" as a cost increase under its own model, generates synthetic mutations that increase cost, tests whether it detects the increase, and reports F1. This is self-referential. The issue-tracker annotations are the only external signal, and the proposal acknowledges they are "biased toward severe regressions."

**The LLM competitive threat is dismissed without engagement.** While the panel ultimately found the LLM alternative inadequate for CI/CD regression detection (lacking determinism, quantitative diffing, formal error bounds, and laptop-CPU execution), the problem statement never addresses this increasingly obvious comparison. This must be addressed explicitly.

**Adoption barriers are underspecified.** Teams must have accessibility trees, define formal task specifications, calibrate thresholds, and trust information-theoretic cost metrics. The task specification requirement alone is a significant barrier.

### What would raise this to 7

1. **Commit to retrospective validation using existing published human data** (CogTool datasets, Oulasvirta group interaction logs). Compute model cost orderings on published UI pairs and compare against already-available human preference orderings. This requires zero new human studies.
2. **Explicitly scope the detection boundary:** "This system detects structural usability regressions (information architecture, interaction flows, element counts, groupings) but not visual usability regressions (color, typography, spacing, animation)." Frame this as complementary coverage alongside screenshot diff tools (Chromatic, Percy).
3. **Define an MVP that can be validated quickly:** accessibility-tree parser + additive Fitts'/Hick's cost + scalar diff threshold. Show this MVP produces useful signal on even 10 real UI pairs.
4. **Engage the LLM comparison head-on:** argue that CI/CD regression detection requires determinism, quantitative comparability, and formal error bounds that LLMs cannot provide.

---

## Axis 2: GENUINE DIFFICULTY AS A SOFTWARE ARTIFACT — 7/10

### Genuinely hard and novel

1. **Bounded-rational bisimulation (Contribution 1).** The cognitive distance metric $d_{\text{cog}}$ that weights state distinguishability by the agent's capacity to exploit that distinction is a genuinely new idea. Computing the supremum over $\beta' \leq \beta$ of TV distances between softmax policies at every state pair is a non-trivial optimization. The partition-refinement algorithm adapted to this non-standard metric is real engineering difficulty. This is the intellectual core of the project.

2. **Compositional cost algebra soundness (Contribution 2).** Proving the algebra upper-bounds mutual information of a full discrete-event cognitive simulation is non-trivial. The sequential load-amplification and parallel-interference operators are psychophysically motivated and formally tractable. However, the soundness theorem is stated without proof sketch — a significant gap.

3. **Accessibility-tree-to-MDP reduction.** Unglamorous but genuinely hard. Real accessibility trees are messy, inconsistent across platforms, and rife with implicit structure. This parser is where many previous automated usability attempts have died.

4. **SMT-based repair synthesis.** Encoding UI constraints + cognitive cost objectives as Z3 constraints is a genuine constraint-synthesis problem, especially the bi-level nature (modify input → re-run pipeline → check output).

### Standard engineering wearing a theoretical costume

- Monte Carlo trajectory sampling: textbook
- Bottleneck classification: threshold checks on known cognitive categories (Wickens' MRT repackaged)
- CI/CD integration, reporting, benchmarking: important but not novel

### The 150K LoC estimate

The panel disagrees on this. The Skeptic estimates 30-50K; the Auditor finds 150K credible. The consensus view: the *shipped artifact* (core engine + CI integration) is likely ~30-40K lines. The remaining ~110K is evaluation infrastructure, benchmark suite, and framework-specific adapters — real code but not research difficulty. The proposal should clearly separate the deployment footprint from the evaluation infrastructure.

The genuine difficulty score of 7 reflects that the bisimulation, cost algebra soundness, and MDP reduction are legitimately hard problems requiring novel work, while acknowledging that roughly 40% of the codebase is standard systems engineering.

---

## Axis 3: BEST-PAPER POTENTIAL — 5/10

### What works

**The framing is genuinely novel.** Reducing usability regression testing to constrained inference over bounded-rational cognitive models is a new problem formulation. The insight that regression detection requires strictly weaker assumptions than absolute prediction is correct, surprising, and broadly applicable. This framing alone distinguishes the work from prior automated usability evaluation.

**The scope is ambitious in the right way.** Bridging MDP theory, information theory, compositional semantics, and program synthesis under a single variational principle is impressive if executed.

### What fails

**The TV bound is too loose to be informative.** The flagship theorem produces $d_{\text{TV}} \leq 0.53$ for "typical parameters" ($\beta=5, \varepsilon=0.005, H=30$). The proposal calls this "informative." It is not. A TV distance of 0.53 means the abstract MDP's trajectory distribution could differ from the original by up to 53%. If regression detection relies on comparing trajectory statistics between two UI versions, each approximated with up to 53% error, the comparison could be meaningless.

**Critical escape hatch not formalized.** If both UI versions are abstracted using the *same* bisimulation partition, errors are correlated and the *difference* in costs may be far more accurate than absolute costs. The proposal hints at this but never formalizes it. This paired-comparison error cancellation argument is the single highest-value theorem the proposal could add — it would transform a loose absolute bound into a potentially tight relative bound, directly serving the consistency-oracle claim.

**The cost algebra's soundness theorem lacks proof.** The claim $C_{\text{alg}}(G) \geq \sum_t I(S_t; A_t)$ is stated without proof sketch, proof intuition, or even a statement of required assumptions. For a theoretical contribution, this is a critical gap.

**The bottleneck taxonomy is not novel.** The five-type classification (perceptual, choice, motor, memory, interference) is essentially Wickens' Multiple Resource Theory repackaged with information-theoretic signatures. The "exhaustiveness theorem" follows from the definitions. The "distinguishability theorem" holds generically (outside a measure-zero set) — a weak claim.

**No empirical results.** Zero worked examples, zero preliminary data, zero evidence that the pipeline produces correct verdicts on any real UI pair.

**Venue mismatch.** CHI demands human validation. UIST wants a working demo. ICSE/FSE wants empirical regression data with real ground truth. The cross-disciplinary nature is a strength for positioning but a weakness for any single venue's best-paper committee.

### What would raise this to 7

1. **Formalize the paired-comparison error cancellation theorem.** Show that for two UI versions abstracted under the same bisimulation partition, the regression detection error is $O(\varepsilon)$ rather than $O(H\beta\varepsilon)$.
2. **Provide a proof sketch for the cost algebra soundness theorem.** At minimum, state the required assumptions on the discrete-event cognitive simulation class.
3. **Produce one end-to-end worked example** on a real UI pair (e.g., Material UI v4 → v5 breaking change). Show the accessibility tree diff, the MDP, the cost comparison, the bottleneck classification, and the repair suggestion.
4. **Scope the paper to one strong contribution** (consistency oracle + bisimulation) with the cost algebra and taxonomy as supporting material. Target UIST.
5. **Include retrospective validation** using published CogTool or similar data.

---

## Axis 4: LAPTOP CPU + NO HUMANS — 9/10

### This is the proposal's strongest axis

The CPU-only design is principled and well-argued across five dimensions:

1. **Structural saliency from accessibility trees** — no vision models, no pixels. This is not just cheaper; it's arguably more appropriate for structural usability analysis. The accessibility tree captures *semantic* saliency, not just visual saliency.
2. **Monte Carlo trajectory sampling** — embarrassingly parallel on CPU cores. 10K trajectories over a 50-step horizon on a ≤10K-state MDP completes in seconds on 4 cores.
3. **SMT/ILP solvers (Z3, CBC)** — CPU-native, well-optimized.
4. **Bisimulation keeps MDPs small** — the coarsening guarantee ($|\hat{S}| \leq 10^4$ for production UIs) is the critical enabler.
5. **No training phase** — calibrated from published psychophysical parameters. This eliminates GPU-dependent ML entirely.

The "no humans" evaluation plan (issue-tracker annotations + synthetic mutations) is clean and internally consistent, even if the circularity issue limits its external validity.

### Minor concerns

- **Z3 timeout risk** for complex repair constraints. Mitigable with anytime strategies.
- **Accessibility tree extraction** requires a rendering engine (headless browser for web UIs). Not mentioned in the proposal.
- **The ≤10K states claim** is empirical but unsupported by published data.

### Score: 9/10

The constraints are genuinely turned into design principles. The one-point deduction is for the unsubstantiated state-space size claims and Z3 timeout risk.

---

## Axis 5: FATAL FLAWS

### Flaw 1: Circular Evaluation (SEVERE — must be fixed)

The evaluation loop is self-referential: define cost via the model → generate mutations that increase cost → test detection of cost increase → report F1. At no point does an external signal validate that "model cost increase" corresponds to "actual usability degradation." The issue-tracker annotations are the only break in the circle and are acknowledged to be biased.

**Required fix:** Add retrospective validation against existing published human performance data. Compute model orderings on published UI pairs (CogTool, etc.) and compare against human orderings. This breaks the circle without new human studies.

### Flaw 2: Unvalidated Ordinal Validity (SEVERE — must be addressed)

The consistency oracle must be consistent with *something external*. The rank-correlation study between model orderings and human preference orderings is described as "follow-up work" when it is the *core validation* of the system's reason for existence.

**Required fix:** Either commit to retrospective validation (as above) or explicitly scope the claim: "We demonstrate internal consistency and provide a framework for future external validation." The latter is weaker but honest.

### Flaw 3: TV Bound Looseness (MODERATE — undermines theoretical claims)

$d_{\text{TV}} \leq 0.53$ is not informative. This undermines the bisimulation's utility and the best-paper narrative.

**Required fix:** Formalize the paired-comparison argument (error cancellation under shared abstraction) to derive tighter bounds for regression detection specifically.

### Flaw 4: Unproven Soundness Theorem (MODERATE — gaps in theoretical claims)

The cost algebra's soundness theorem lacks a proof sketch, proof intuition, and statement of assumptions.

**Required fix:** Provide at minimum a proof sketch and explicit statement of the discrete-event simulation class for which the bound holds.

### Flaw 5: Novel Composition Parameters Without Calibration (MODERATE)

The parameters γ (sequential coupling) and α (parallel interference) are novel — no published psychophysical values exist. The claim of calibration from "published parameters" only covers component-level parameters, not composition parameters.

**Required fix:** Either (a) demonstrate robustness of orderings across wide parameter ranges via sensitivity analysis, (b) calibrate against existing published task-completion data, or (c) scope the initial system to additive composition (no γ, α) and add the full algebra as an enhancement.

### Flaw 6: Speculative Repair Synthesis (LOW — downscope acceptable)

The repair synthesizer is the most speculative component. The intervention-specificity theorem is stated without proof. SMT-based repair is notoriously fragile.

**Required fix:** Downscope to a stretch goal. The system has clear value with regression detection + bottleneck classification alone.

---

## Panel Disagreements

The three experts disagreed significantly, reflecting genuine uncertainty:

| Axis | Skeptic | Auditor | Synthesizer | Final |
|------|---------|---------|-------------|-------|
| Value | 4 | 6 | 8 | **6** |
| Difficulty | 5 | 8 | 7 | **7** |
| Best-Paper | 5 | 4 | 6 | **5** |
| CPU/Humans | 7 | 9 | 9 | **9** |

**On Value:** The Skeptic's LLM argument was effectively refuted (LLMs lack determinism, error bounds, and CPU execution). The Synthesizer's MVP argument is appealing but untested. The Auditor's middle ground (6) correctly weights the real pain against the deferred validation.

**On Difficulty:** The Skeptic undervalues integration complexity (their own 30-50K estimate contradicts the 5-10K "core" claim). The Auditor may overvalue it at 8. The Synthesizer's 7 correctly separates genuinely novel contributions (~60%) from standard engineering (~40%).

**On Best-Paper:** The Auditor and Skeptic converged post-critique at 4-5, recognizing the circular evaluation is devastating at any top venue. The Synthesizer argues the framing alone is a contribution. The final 5 reflects that the framing is novel but the execution gaps are too large for best-paper consideration without amendments.

---

## Amendments Required

Axes 1 (Value = 6) and 3 (Best-Paper = 5) are below the threshold of 7. The following amendments are required:

### Amendment A: Break the Evaluation Circularity
Add retrospective validation against existing published human performance data (CogTool datasets, Oulasvirta group data). Compute model cost orderings on published UI pairs and compare against human orderings. Commit to this as part of the core evaluation, not follow-up work. This requires zero new human studies.

### Amendment B: Formalize Paired-Comparison Error Cancellation
Add a theorem showing that regression detection under shared bisimulation abstraction achieves tighter error bounds than absolute trajectory approximation. This transforms the loose TV bound ($\leq 0.53$) into a compelling result: "loose bounds for general approximation, tight bounds for the regression-detection task we actually solve."

### Amendment C: Prove or Sketch the Cost Algebra Soundness
Provide a proof sketch for $C_{\text{alg}}(G) \geq \sum_t I(S_t; A_t)$, including the required assumptions on the simulation class and the role of the coupling parameter γ.

### Amendment D: Explicitly Scope Detection Boundary
State clearly: "This system detects structural usability regressions visible in accessibility tree changes. Visual regressions (CSS, typography, spacing, animation) require complementary tools (Chromatic, Percy). This is by design: structural regressions are the class most affected by code changes in pull requests and least served by existing tools."

### Amendment E: Engage the LLM Comparison
Add a paragraph arguing that CI/CD regression detection requires determinism, quantitative comparability, monotonicity, and formal error bounds — properties that LLMs fundamentally cannot provide. Position the system as complementary to LLM-based usability critique, not competitive with it.

### Amendment F: Define MVP Path
Explicitly separate a minimal viable system (accessibility-tree parser + additive Fitts'/Hick's cost + scalar regression threshold) from the research frontier (bisimulation, full cost algebra, bottleneck taxonomy, repair synthesis). Show that each theoretical contribution incrementally improves the MVP.

### Amendment G: Downscope Repair Synthesis
Relegate SMT-backed repair synthesis to a stretch goal. The core contribution is regression detection + bottleneck classification + the theoretical framework enabling these.

### Amendment H: Address Parameter Calibration
For the novel composition parameters (γ, α), either (a) include sensitivity analysis showing ordering robustness across parameter ranges, or (b) scope the initial system to additive composition and introduce the full algebra as an enhancement validated against published data.

---

## Projected Post-Amendment Scores

| Axis | Current | Projected | Change |
|------|---------|-----------|--------|
| Value | 6 | **7-8** | Retrospective validation + MVP path + explicit scoping |
| Difficulty | 7 | **7** | Unchanged (downscoping repair slightly reduces, but remaining difficulty is concentrated) |
| Best-Paper | 5 | **7** | Paired-comparison theorem + proof sketch + worked example + retrospective validation |
| CPU/Humans | 9 | **9** | Unchanged |
| **Total** | **27** | **30-31** | Viable for CONTINUE |

---

## Verdict: AMENDMENTS REQUIRED before CONTINUE

The problem statement identifies a genuine gap (automated usability regression detection in CI/CD), proposes an elegant theoretical framework (bounded-rational inference over UI state spaces), and is well-designed for its computational constraints (laptop CPU, no humans). The consistency-oracle framing is the strongest conceptual contribution.

However, the current formulation has two critical weaknesses: (1) the evaluation plan is circular with no external validation, and (2) the flagship theoretical bound is too loose to be informative without the paired-comparison formalization. Both are fixable with the amendments above.

An amended version of the problem statement follows in `ideation/crystallized_problem.md`.

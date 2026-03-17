# Depth Check: Spectral Decomposition Oracle

**Panel Chair Synthesis — Final & Binding**
**Slug:** `spectral-decomposition-oracle`
**Date:** 2026-03-08
**Status:** DEFINITIVE ASSESSMENT

---

## 0. Executive Summary

This project proposed a triple-threat contribution — a bridging theorem (T2) connecting spectral graph theory to decomposition bounds, a cross-method decomposition selection system, and the first complete MIPLIB decomposition census — packaged at 155K LoC and targeting best-paper at MPC/IPCO. After a 3-expert adversarial review, the panel finds that **the original proposal is structurally unsound** due to a vacuous theorem constant, 3–6× LoC inflation, and circular evaluation. However, **a viable paper exists inside this proposal**: a spectral-feature-engineering study for decomposition selection, anchored by the MIPLIB census artifact. The project continues only under mandatory restructuring.

**Final verdict: CONDITIONAL CONTINUE — as Amendment E (spectral feature study), not as the original proposal.**

---

## 1. Disagreement Resolution

### 1.1 Should T2 appear at all?

**Resolution: KEEP as a 2–3 page "Theoretical Motivation" section. Not a main contribution.**

*Justification:* The Auditor's position is best-supported. T2 establishes a qualitatively correct structural insight (spectral quality degrades decomposition gracefully, quadratically in coupling) that motivates the feature set. Dropping it entirely (Skeptic) discards useful intuition; promoting it (original proposal) exposes a vacuous constant to reviewer attack. The compromise — honest motivation, not headline — eliminates the fatal-if-vacuous risk while preserving the intellectual through-line.

### 1.2 Continue vs. Reject?

**Resolution: CONDITIONAL CONTINUE under Amendment E.**

*Justification:* Two of three experts recommend continuation with restructuring; the Skeptic's REJECT is contingent on the team insisting on the original framing. The Synthesizer correctly identified that the original proposal is dead but Amendment E (spectral features for decomposition selection) is a different, viable project sharing ~60% of the intellectual DNA. The condition is acceptance of the restructured scope.

### 1.3 Is the no-go certificate valuable?

**Resolution: It is a LEARNED FUTILITY PREDICTOR, not a formal certificate. Useful but not novel in kind.**

*Justification:* The Skeptic's argument is dispositive (2/3 agreement with Synthesizer conceding). The theoretical threshold γ_min(ε) inherits T2's vacuous constant C, making the formal guarantee meaningless on practical instances. An empirically calibrated threshold is a classifier with a "no" class, not a certificate. Rename throughout. The concept retains practical value — a fast negative predictor that saves wasted decomposition attempts is genuinely useful — but the framing must be honest.

### 1.4 Honest Lines of Code?

**Resolution: ~25–30K LoC new code; ~50K total with infrastructure and wrappers.**

*Justification:* The Skeptic's detailed itemization (§5 of final critique: 18.5K core + 50% buffer ≈ 28K) is the most granular and accounts for the mandatory descoping to SCIP/GCG wrappers. The Auditor's 55–70K assumed partial reimplementation that all three experts agree should not happen. The original 155K is indefensible — a 5–6× inflation that would damage credibility with reviewers.

---

## 2. Final Consensus Scores

Methodology: median of three experts' post-critique scores, adjusted ±1 only with strong justification.

| Pillar | Auditor | Skeptic | Synthesizer | **Median** | **Final** | Adjustment Rationale |
|--------|---------|---------|-------------|------------|-----------|---------------------|
| **1. Value** | 6 | 5 | 5 | 5 | **5** | No adjustment. Census is the diamond but audience is niche (decomposition researchers, not all OR practitioners). Cross-method selection is a genuine extension over Kruber et al. 2017, not a paradigm shift. |
| **2. Difficulty** | 5 | 4 | 4 | 4 | **4** | No adjustment. Post-descoping, novel work is: hypergraph Laplacian adaptation for non-square matrices (genuinely hard), spectral feature pipeline (moderate), ML classifier (standard), wrappers (routine). T2 proof (Davis-Kahan adaptation) adds proof effort but not implementation difficulty. |
| **3. Best-Paper** | 3 | 3 | 3 | 3 | **3** | No adjustment. Unanimous at 3. Publishable at JoC, not competitive for best paper at any target venue. Would need census to reveal surprising structural insights to reach 4. |
| **4. Laptop CPU** | 7 | 6 | 6 | 6 | **6** | No adjustment. Spectral analysis (30s/instance) is trivially laptop-feasible. 500-instance stratified eval ≈ 5 days on 4 cores. Full spectral annotation of 1,065 instances ≈ 9 hours. Full decomposition census is impractical for iteration but feasible as a batch artifact. |
| **5. Novelty** | 4* | 3* | 4* | 4 | **4** | *Imputed from expert arguments.* Cross-method reformulation selection is a genuine extension (not just algorithm selection). Spectral features as a continuous feature family for decomposition have theoretical grounding that AutoFolio-style systems lack. But: Kruber et al. 2017 + Bergner et al. 2015 (GCG) cover adjacent ground. The delta is real but incremental. |

**Composite assessment: A solid B-tier contribution currently packaged as an A-tier contribution.** The gap between framing and substance is the core problem.

---

## 3. Fatal and Serious Flaws

### FATAL FLAWS (require fix before any continuation)

| # | Flaw | Severity | Agreement | Fixable? | Fix |
|---|------|----------|-----------|----------|-----|
| **F1** | **T2's constant C = O(k·κ⁴·‖c‖∞) is vacuous on ~60–70% of MIPLIB** (any instance with big-M constraints, κ > 10³). A bound of "degradation ≤ 10³⁰" is informationally empty. T2 cannot be the headline contribution. | Fatal | 3/3 | Yes — demote | Demote T2 to "Theoretical Motivation" section (2–3 pages). Lead with census + spectral features. Present T2 as structural insight for well-conditioned instances, not a quantitative tool. |
| **F2** | **Evaluation circularity.** Validating method selection against own implementations is self-referential. If the DW implementation is buggy, oracle learns to avoid DW — this reflects implementation quality, not structural truth. | Fatal | 3/3 | Yes — redesign | Use GCG (DW) and SCIP-native Benders as independent baselines. Reframe evaluation as selector ablation: compare spectral vs. syntactic vs. random selectors with fixed external backends. Leave-one-family-out cross-validation. |
| **F3** | **155K LoC claim is 5–6× inflated.** Counting reimplementation of Benders/DW/Lagrangian as novel code when SCIP and GCG provide these is indefensible. Would trigger immediate credibility loss with reviewers. | Fatal | 3/3 | Yes — descope | Wrap SCIP Benders + GCG DW. Custom code only for Lagrangian (partially novel) and spectral engine. Target 25–30K LoC. |

### SERIOUS FLAWS (must be addressed but not individually fatal)

| # | Flaw | Severity | Agreement | Fixable? | Fix |
|---|------|----------|-----------|----------|-----|
| **S1** | **No-go "certificate" is a learned classifier, not a formal guarantee.** Calling it a certificate when the theoretical threshold is vacuous is misleading. | Serious | 2/3 (Skeptic + Synthesizer) | Yes — rename | Rename to "spectral futility predictor." Present as empirically calibrated learned threshold. Drop the word "certificate" from all text. |
| **S2** | **Only ~10–25% of MIPLIB has exploitable block structure** (Bergner et al. 2015). This limits the data available for meaningful method-selection evaluation and means the oracle helps on a small fraction of instances. | Serious | 2/3 (Auditor + Skeptic) | Partially | Be honest about scope of applicability. Report "coverage" (% of instances where oracle adds value). Use this as motivation for the futility predictor. |
| **S3** | **Missing ground-truth for structure labels.** No authoritative labeling of MIPLIB structure types exists. F1 ≥ 0.85 target is unmeasurable without external reference. | Serious | 2/3 (Auditor + Skeptic) | Yes | Use GCG's structure detection as reference for DW-amenable instances. For Benders, use known stochastic programming benchmarks with documented stage structure. Acknowledge missing reference for Lagrangian-amenable structure. |
| **S4** | **Paper tries to be three things** (theory/system/benchmark) and the panel finds it succeeds at none convincingly in the original framing. | Serious | 2/3 (Skeptic + Synthesizer) | Yes — focus | Restructure as a computational study: census artifact + spectral feature engineering + empirical evaluation. Theory supports, does not lead. |
| **S5** | **Census with 60% timeouts is not a census.** Full MIPLIB at 1-hour caps yields ~400 completed runs + ~665 timeouts with partial data. | Serious | 2/3 (Skeptic + Synthesizer) | Yes | 500-instance stratified sample for paper evaluation. Full spectral annotation (all 1,065, no decomposition needed, ~9 hours) as artifact. Full decomposition results as supplementary material. |

---

## 4. Required Amendments (Binding)

All amendments are MANDATORY for continuation. Failure to implement any single amendment triggers the ABANDON recommendation.

### Amendment 1: Census-First Restructuring

**Restructure the paper as a computational study, not a theorem paper.**

- **Title:** "Spectral Features for MIP Decomposition Selection: A Computational Study with MIPLIB Census"
- **Lead contribution:** The first systematic decomposition census of MIPLIB 2017 — spectral structural annotations for all 1,065 instances, decomposition evaluation on 500 stratified tractable instances
- **Supporting contributions:** Spectral feature family (continuous instance features with theoretical grounding), cross-method selection oracle, futility predictor
- **Demoted to motivation:** T2 (2–3 pages, honest about vacuousness)

### Amendment 2: External Baselines (Break Circularity)

**All decomposition evaluation must use independently maintained implementations.**

- GCG for Dantzig-Wolfe reformulation and column generation
- SCIP's native Benders (SCIPcreateBendersDefault, available since v7.0)
- Custom implementation only for Lagrangian relaxation (with honest disclosure that this is the one non-independent baseline)
- Evaluation reframed as **selector ablation**: spectral selection vs. syntactic features vs. random vs. always-DW, all using the same fixed backends

### Amendment 3: Honest Scope

**LoC and system claims must reflect reality.**

- Target: 25–30K LoC of new code
- Breakdown: spectral engine (~6K), classifier + oracle (~3K), solver wrappers (~3K), census infrastructure (~4K), Lagrangian wrapper (~5K), tests + CI (~5K), analysis + visualization (~4K)
- No claims of "155K lines" or "six major subsystems" that count reimplementation of existing solvers
- The system is a **preprocessing plugin and evaluation framework**, not a standalone decomposition suite

### Amendment 4: Honest Terminology

**Rename and reframe all overclaimed components.**

- "No-go certificate" → "spectral futility predictor"
- "Structure-to-Strategy Theorem" → retained as section title for T2, but never described as the paper's "main result"
- "Bridging theorem" → "theoretical motivation establishing qualitative connection"
- "First formal futility certificate" → "first learned decomposition futility predictor using spectral features"

### Amendment 5: Stratified Evaluation Design

**Replace the full-census evaluation with a rigorous stratified design.**

- **Tier 1 (paper, controlled):** 500 MIPLIB instances, stratified by structure type (5 categories) × size bin (5 bins) × conditioning (2 levels). Power analysis: ≥20 instances per stratum for Spearman ρ at α = 0.05.
- **Tier 2 (artifact, fast):** Spectral annotations for all 1,065 instances (~9 hours, no decomposition needed). Released as open data.
- **Tier 3 (artifact, batch):** Full decomposition results for all 1,065 instances with 1-hour cap. Run as a background batch over ~12 days on 4 cores. Supplementary material, not main evaluation.
- **Development pilot:** 50-instance pilot in first 2 weeks to validate spectral correlation hypothesis before committing to full evaluation.

### Amendment 6: Spectral Feature Ablation (The Core Experiment)

**The paper's central experiment must be a feature ablation, not a theorem validation.**

- Compare decomposition selection accuracy using:
  1. Spectral features only (spectral gaps, eigenvector localization, algebraic connectivity, partition quality)
  2. Syntactic features only (constraint density, variable degree, integrality ratio, coefficient range — as in AutoFolio/SATzilla)
  3. Spectral + syntactic combined
  4. Random baseline
  5. Trivial baseline (always choose most common method)
- Report: accuracy, Spearman ρ with bound improvement, feature importance rankings
- This ablation is what makes the paper publishable independently of T2

### Amendment 7: Venue Targeting

**Target venues that match the actual contribution.**

- **Primary:** INFORMS Journal on Computing (JoC) — explicitly values reproducible computational studies with open artifacts
- **Secondary:** Computers & Operations Research — empirical + system papers welcomed
- **Stretch:** CPAIOR — algorithm selection papers published here
- **Do NOT submit to:** MPC (requires tight theorems), IPCO (theory venue), any venue where T2 would be the evaluated contribution

---

## 5. Final Recommendation

### **CONDITIONAL CONTINUE**

The project continues under all seven amendments above. The deliverable is a well-executed spectral feature engineering study for decomposition selection, anchored by a first-of-kind MIPLIB decomposition census, targeting INFORMS JoC. This is a solid, publishable contribution with genuine community value.

**What this project IS:** A computational study introducing spectral features for MIP decomposition selection, validated by comprehensive MIPLIB evaluation, with theoretical motivation from spectral perturbation theory.

**What this project IS NOT:** A bridging theorem connecting two fields, a triple-threat theory-system-benchmark paper, or a best-paper contender at a top optimization venue.

The team must explicitly accept this reframing. If the team insists on the T2-centered framing, the original proposal, or the 155K LoC scope, the recommendation converts to **ABANDON**.

---

## 6. Probability Estimates

| Outcome | Probability | Conditions |
|---------|-------------|------------|
| **P(publication at JoC)** | **0.55** | All 7 amendments implemented; spectral features demonstrate ≥15% accuracy improvement over syntactic features; census reveals non-trivial structural findings |
| **P(publication at C&OR or CPAIOR)** | **0.70** | All amendments implemented; even if spectral advantage is modest, census artifact carries the paper |
| **P(any publication at a reputable venue)** | **0.72** | Amendment E scope; honest framing; adequate evaluation |
| **P(best-paper at any venue)** | **0.03** | Would require census to reveal a genuinely surprising structural finding (e.g., "40% of MIPLIB has unexploited staircase structure") AND spectral features to dominate syntactic by a wide margin |
| **P(publication at MPC/IPCO)** | **0.05** | Only if T2's constant can be tightened to κ² or better via entry-wise analysis, which is speculative |
| **P(abandon)** | **0.25** | Spectral correlation fails (ρ < 0.4 on pilot), OR team rejects Amendment E scope, OR SCIP/GCG integration proves infeasible |

---

## 7. Kill Gates (Binding Milestones)

Failure at any gate triggers immediate ABANDON.

| Gate | Milestone | Deadline | Kill Condition |
|------|-----------|----------|----------------|
| **G1** | 50-instance pilot: Spearman ρ between δ²/γ² and observed bound degradation | Week 2 | ρ < 0.4. If the spectral ratio does not even weakly predict decomposition benefit, the entire spectral premise is invalid. |
| **G2** | SCIP Benders + GCG DW wrappers operational | Week 4 | Cannot produce dual bounds on ≥80% of the 50-instance pilot. If external solvers cannot be wrapped reliably, the evaluation is impossible. |
| **G3** | Spectral features outperform syntactic features on 200-instance development set | Week 8 | Spectral selector accuracy ≤ syntactic selector accuracy (no improvement). If spectral features add nothing over standard instance features, the paper has no thesis. |
| **G4** | 500-instance stratified evaluation complete | Week 14 | Results do not support any of: (a) ρ ≥ 0.5, (b) selector accuracy ≥ 65%, (c) futility predictor precision ≥ 80%. If all three fail, there is no publishable finding. |
| **G5** | Draft paper submitted for internal review | Week 18 | External collaborator/advisor review returns "reject" or "fundamental problems." At this point, remaining effort is not justified. |

---

## 8. What Must Be True for This Paper to Succeed

The paper rests on one empirical hypothesis that no amount of theory can substitute for:

> **Spectral features of the constraint hypergraph Laplacian are meaningfully more predictive of decomposition benefit than standard syntactic instance features.**

If this is true, the paper has a clear thesis ("spectral features work and here's why"), a strong ablation, and a useful tool. If this is false, T2 is decorative, the oracle is no better than existing approaches, and the census is the only surviving contribution — which alone may support a shorter workshop or data paper, but not a full JoC submission.

Gate G1 (Week 2, 50-instance pilot) is designed to test this hypothesis cheaply before committing months of effort.

---

## Appendix A: Score Provenance

### Post-Critique Expert Scores

| Pillar | Auditor | Skeptic | Synthesizer | Median | Final |
|--------|---------|---------|-------------|--------|-------|
| Value | 6 | 5 | 5 | 5 | **5** |
| Difficulty | 5 | 4 | 4 | 4 | **4** |
| Best-Paper | 3 | 3 | 3 | 3 | **3** |
| Laptop CPU | 7 | 6 | 6 | 6 | **6** |
| Novelty | 4* | 3* | 4* | 4 | **4** |

*Novelty scores imputed from expert arguments on originality vs. Kruber et al. 2017 and GCG.

### Movement from Original Proposal

| Pillar | Original Claim | Panel Final | Delta | Why |
|--------|---------------|-------------|-------|-----|
| Value | "Extreme" (implied 8+) | 5 | −3+ | Census is valuable; T2 vacuousness and niche audience limit broader impact |
| Difficulty | 155K LoC (implied 8+) | 4 | −4+ | Post-descoping, novel work is moderate: spectral engine + ML pipeline + wrappers |
| Best-Paper | MPC target (implied 7+) | 3 | −4+ | Unanimous: not competitive at top venues. Solid JoC paper. |
| Laptop | "Fully feasible" (claimed 8) | 6 | −2 | Spectral analysis is fast; full census iteration is slow; stratified design mitigates |

---

## Appendix B: Dissenting Notes

**Auditor dissent (minor):** Scored Value at 6 (panel final: 5), arguing the spectral-to-decomposition bridge has broader applicability than the other experts credit. The Chair finds the Skeptic's counter — that the audience is a few dozen decomposition research groups, not the OR community at large — more convincing. The census helps *decomposition researchers*; it does not change how practitioners solve supply chain or scheduling problems.

**Auditor dissent (minor):** Scored Laptop at 7 (panel final: 6). The Chair accepts 6 as the full decomposition census remains a multi-day batch job even on 4 cores, and development iteration requires the stratified design rather than naive "run everything."

**Skeptic dissent (noted):** Maintains the project should be REJECTED outright if evaluated as originally proposed. The Chair agrees — the CONDITIONAL CONTINUE applies exclusively to the Amendment E reframing. The original proposal, with T2 as hero theorem and 155K LoC scope, is REJECTED by this panel.

---

*This depth check is final and binding. The project may proceed only under the conditions specified in §4–§7. Any material deviation from the amendments requires a new depth check.*

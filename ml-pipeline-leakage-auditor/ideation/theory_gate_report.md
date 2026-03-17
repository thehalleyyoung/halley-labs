# Theory Gate Report: TaintFlow (proposal_00)

## Meta

| Field | Value |
|-------|-------|
| **Date** | 2026-03-08 |
| **Project** | ml-pipeline-leakage-auditor |
| **Proposal** | proposal_00 — TaintFlow: Quantitative Information-Flow Auditing via Hybrid Dynamic-Static Analysis |
| **Gate** | Theory Gate (pre-implementation) |
| **Evaluator Composition** | 6 evaluators: 3 prior-phase (Skeptic, Mathematician, Community Expert) + 3 verification panel (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) |
| **Process** | Adversarial cross-critique with structured dispute resolution. Each evaluator scored independently, then critiqued each other's assessments. Disputes adjudicated by evidence weight and argument quality. |
| **Adjudicated Composite** | **6.0 / 10** |
| **Verdict** | **CONDITIONAL CONTINUE** |

---

## Executive Summary

TaintFlow proposes the first differential information auditor for ML pipelines, bridging quantitative information flow (QIF), abstract interpretation, and ML pipeline semantics — a genuinely novel intersection confirmed by all six evaluators. The adjudicated composite score is **6.0/10** (V6/D6/BP4/L9/F5), reflecting high laptop-CPU feasibility and a real problem (15–25% of Kaggle kernels contain leakage), but tempered by zero implementation progress, a proof gap in the crown-jewel theorem T1 (fit-transform channel decomposition), and ~21% self-assessment inflation. We recommend **CONDITIONAL CONTINUE** via Path B (Core Tool: 15–25K LoC Python, 10–14 months), contingent on passing a binding 4-week T1 proof gate and an 8-week working prototype gate, with five immediate-termination triggers defined below.

---

## Evaluation Process

### Methodology

Six evaluators assessed TaintFlow across five dimensions using a shared rubric:

- **V (Extreme Value):** Would this be considered extremely valuable if fully realized?
- **D (Genuine Software Difficulty):** Is this genuinely difficult software engineering and/or research?
- **BP (Best-Paper Potential):** Could this plausibly win a best-paper award at a top venue?
- **L (Laptop-CPU Feasibility):** Can this be built and run on commodity hardware?
- **F (Overall Feasibility):** What is the overall probability of successful completion?

### Phases

1. **Prior Evaluation (Phase 1):** Three domain evaluators — Skeptic, Mathematician, Community Expert — independently assessed the proposal against theory artifacts (`theory/approach.json`, ideation documents, verification reports).

2. **Verification Panel (Phase 2):** Three additional evaluators — Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer — performed a second-pass review with access to Phase 1 evaluations.

3. **Cross-Critique (Phase 3):** All evaluators critiqued each other's scores. Eight substantive disputes were identified and adjudicated:
   - **Auditor won 5 of 8 disputes** (calibration anchored closest to evidence)
   - **Synthesizer won 2 disputes** (revised downward from initial optimistic scores)
   - **Skeptic won 1 dispute outright** (on best-paper probability inflation)

4. **Adjudication:** Final scores represent evidence-weighted consensus, not arithmetic mean.

---

## Consensus Scores

| Dimension | Prior Skeptic | Mathematician | Community Expert | Ind. Auditor | Fail-Fast Skeptic | Synthesizer | **Adjudicated** |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **V (Value)** | 6 | 7 | 7 | 6 | 5 | 7.5 | **6** |
| **D (Difficulty)** | 6 | 7 | 6 | 6 | 5 | 7 | **6** |
| **BP (Best-Paper)** | 5 | 4 | 5 | 4 | 3 | 5.5 | **4** |
| **L (Laptop-CPU)** | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| **F (Feasibility)** | 5 | 5 | 5.5 | 5 | 4 | 4.5 | **5** |
| **Composite** | 5.90 | 6.40 | 6.10 | 6.00 | 5.20 | 6.70 | **6.00** |
| **Recommendation** | Cond. Continue | Cond. Continue | Cond. Continue | Cond. Continue (60%) | Cond. Continue (55%) | Cond. Continue (78%) | **Cond. Continue** |

**Score Distribution Analysis:**

- **Tight consensus** on L (9/10 unanimous) — the proposal's strongest dimension.
- **Moderate spread** on V (5–7.5) — disagreement on whether conditional soundness is "extreme" value or merely "good."
- **Widest spread** on BP (3–5.5) — reflects genuine uncertainty about venue reception of conditional soundness results.
- **F scores cluster low** (4–5.5) — all evaluators concerned about zero implementation and proof gap in T1.

---

## Pillar Analysis

### Extreme Value (V: 6/10)

**What earns 6:** TaintFlow addresses a real, quantified problem. The 15–25% leakage prevalence in Kaggle kernels is well-documented. The novel intersection of QIF × abstract interpretation × ML pipeline semantics is confirmed genuine by all evaluators — no prior work combines these three fields for ML pipeline auditing. The leakage spectrum (per-feature, per-stage decomposition in bits) would be a first-of-its-kind diagnostic.

**What prevents 7+:** Value is contingent on the soundness guarantees actually holding. Without proven T1, TaintFlow reduces to a sophisticated heuristic — still useful, but not the "provably sound" tool that distinguishes it from competitors. The 500-line detection script captures ~50–70% of detection value, meaning the marginal value of the full system is concentrated in quantification (bits of leakage per feature per stage), which depends on the hardest components.

**Key evidence:**
- LeakageDetector (pattern matching) and LeakGuard (empirical re-execution) exist but provide no information-theoretic quantities
- Target users (ML platform teams, regulators, Kaggle) have demonstrated need
- The quantification gap (how much leakage, not just whether leakage exists) is genuinely unfilled

### Genuine Software Difficulty (D: 6/10)

**What earns 6:** The tight channel capacity catalog for ~15–20 common ML operations (mean, median, PCA, groupby, target encoding) requires genuine mathematical work. The fit-transform decomposition lemma (T1) is a non-trivial research contribution. The dynamic DAG extraction via `sys.settrace` with roaring-bitmap row provenance is substantial engineering.

**What prevents 7+:** Self-assessment claimed 55–65K LoC (Rust + Python), but realistic estimates converge on 15–20K genuinely novel LoC. The remaining lines would be boilerplate, glue code, and test scaffolding. The partition-taint lattice construction is standard (height 69, finite fixpoint guaranteed). API monkey-patching for 300+ pandas methods is tedious but not difficult.

**Inflation identified:**
- Claimed LoC: 55–65K → Realistic novel LoC: 15–20K (70–75% reduction)
- Much of the "Rust abstract interpretation engine" (claimed 40K LoC) can be replaced by Python with NumPy for Path B, dramatically reducing engineering complexity

### Best-Paper Potential (BP: 4/10)

**What earns 4:** The theoretical framework is genuinely novel and could contribute to QIF literature. Conditional soundness (sound given the observed execution) is a legitimate contribution. Multiple venues could accept this: ICML/NeurIPS systems track, OOPSLA, FSE, USENIX Security.

**What prevents 5+:** All evaluators agree P(best-paper) = 2–4%, not the 12% self-assessed. Reasons:
- Conditional soundness is weaker than unconditional soundness — reviewers at PL venues may view this as an engineering shortcut rather than a theoretical contribution
- The PCA bound is vacuous for d > 50 (i.e., for most real-world datasets), which undermines the "tight bounds" narrative
- Top ML venues increasingly want empirical validation at scale; a proof-focused paper without large-scale experiments may struggle
- The 500-line script comparison undermines the novelty argument: "Why build 15K LoC when 500 lines catches most leakage?"

**Venue-dependent assessment:**
- P(acceptance | top ML venue, full system) ≈ 25–35%
- P(acceptance | PL/SE venue, restricted scope) ≈ 40–55%
- P(any publication, any scope) ≈ 55–65%

### Laptop-CPU Feasibility (L: 9/10)

**Unanimous 9/10.** This is TaintFlow's strongest dimension.

- All computation is symbolic/analytical — no GPU required
- Abstract interpretation over the partition-taint lattice has bounded height (69) — guaranteed termination
- Channel capacity formulas are closed-form for Tier 1 operations
- Roaring bitmaps are memory-efficient for row provenance tracking
- Target pipelines are scikit-learn/pandas scale (not distributed ML) — typical datasets fit in memory
- Python profiling via `sys.settrace` adds modest overhead (2–5×), acceptable for an auditing tool

**Why not 10:** Empirical refinement (Phase 3, KSG estimator) on large datasets could be slow; but this phase is optional and can be disabled.

### Overall Feasibility (F: 5/10)

**What earns 5:** The specification quality is unusually high for a pre-implementation proposal. The architecture is well-decomposed into independently testable phases. The dynamic-first approach (Approach B) eliminates the highest-risk component (Python static analysis, estimated 50–65% coverage). Multiple fallback scopes exist.

**What prevents 6+:**
- **impl_loc = 0 is GENUINE** — zero code exists. Not a single line of implementation has been written.
- **The crown jewel T1 has a proof gap** — Step 3 (explaining-away concern) lacks formal treatment of correlated channels in the fit-transform decomposition.
- **Self-assessment inflation of ~21%** — claimed composite 7.50, actual adjudicated 6.0.
- **"theory_complete" status is premature** — correct label is "theory_specified." The theory is articulated, not proven.
- **Timeline risk** — 8 person-months estimated, but T1 proof alone could consume 2–4 months with uncertain outcome.

---

## Fatal Flaw Analysis

### Genuinely Fatal (would kill the project)

| # | Flaw | Status | Assessment |
|---|------|--------|------------|
| 1 | T1 (fit-transform decomposition) is provably impossible | **Not established** | No evaluator believes T1 is impossible. The explaining-away concern is a proof gap, not a fundamental barrier. Probability of complete T1 failure: ~10–15%. Probability of restricted-scope T1 (linear estimators only): ~55–65%. |
| 2 | Conditional soundness is vacuous (no venue accepts it) | **Not established** | At least SE/PL venues (OOPSLA, FSE) have accepted conditional soundness results. ML venues are more skeptical but not uniformly rejecting. |

### Serious but Non-Fatal (require mitigation)

| # | Flaw | Mitigation |
|---|------|------------|
| 3 | PCA bound vacuous for d > 50 | Restrict claims to d ≤ 50 or provide dimension-reduced bounds. Clearly scope as limitation. |
| 4 | Zero implementation after extensive theory phase | 4-week proof gate + 8-week prototype gate enforce rapid execution. |
| 5 | Self-assessment inflation (~21%) | Corrected in this report. Future milestones use adjudicated scores. |
| 6 | 500-line script captures 50–70% of detection value | Position TaintFlow for quantification value (bits per feature), not just detection. |

### Moderate (acceptable risks)

| # | Flaw | Notes |
|---|------|-------|
| 7 | "theory_complete" mislabel | Corrected to "theory_specified" in this report. |
| 8 | theory_bytes = 0 in State.json | File-path bug; theory/approach.json exists at 42.8KB. |
| 9 | LoC overestimate (55–65K → 15–20K) | Path B already scopes to 15–25K. |

**Verdict on fatal flaws:** No genuinely fatal flaw has been identified. All serious flaws have defined mitigations with measurable success criteria. The project should continue under binding conditions.

---

## Crown Jewel Assessment (T1: Fit-Transform Channel Decomposition)

### What It Is

T1 decomposes the `fit_transform` feedback loop — where test data influences fitted parameters, which then transform training data — into two independent channels:

1. **Aggregation channel:** test rows → fitted parameters (e.g., mean, variance computed over pooled train+test data)
2. **Pointwise application channel:** fitted parameters → transformed features (e.g., standardization using the fitted mean/variance)

This decomposition enables compositional leakage analysis: the total leakage through a `fit_transform` operation is bounded by the sum of leakages through each sub-channel, leveraging the data processing inequality (DPI).

### Why It Matters

Without T1, TaintFlow cannot provide compositional soundness guarantees for the most common leakage pattern in ML pipelines: fitting a preprocessor on pooled data before splitting. This pattern accounts for an estimated 60–80% of real-world leakage incidents. T1 is the theoretical linchpin that elevates TaintFlow from "sophisticated heuristic" to "provably sound auditor."

### What the Gap Is

**Step 3 (explaining-away concern):** When the aggregation channel produces correlated statistics (e.g., a `StandardScaler` computes both mean and variance, which are correlated for non-Gaussian data), the pointwise application channel's leakage is not simply the sum of individual parameter leakages. The current proof sketch assumes independence of the aggregation statistics, which does not hold in general.

Formally: for fitted parameters $\hat{\theta} = (\hat{\mu}, \hat{\sigma}^2)$ computed over pooled data, the mutual information $I(X_{\text{test}}; f(\hat{\theta}, X_{\text{train}}))$ is not generally bounded by $I(X_{\text{test}}; \hat{\mu}) + I(X_{\text{test}}; \hat{\sigma}^2)$ due to synergistic information.

### Probability Assessment

| Outcome | Probability | Implication |
|---------|:-----------:|-------------|
| T1 proven as stated (all estimators) | 15–25% | Full compositional soundness. Strongest publication case. |
| T1 proven for restricted set (linear estimators, Gaussian assumptions) | 45–55% | Sound for StandardScaler, PCA, linear regression. Covers ~70% of common leakage patterns. Publishable with clearly stated scope. |
| T1 replaced by conservative bound (no decomposition, generic channel capacity) | 15–20% | Sound but loose bounds. Less publishable. Still useful as a tool. |
| T1 fails entirely (no sound factoring possible) | 10–15% | Fall back to empirical calibration. TaintFlow becomes a heuristic tool (still useful, but main novelty claim is lost). |

### Synthesizer's Resequencing Insight

The Scavenging Synthesizer identified a critical resequencing opportunity: **build the tight capacity catalog (Tier 1 bounds for mean, median, PCA, etc.) before attempting T1.** Rationale:

1. The catalog is lower-risk and independently publishable.
2. Working with concrete bounds may reveal the structure needed to prove T1.
3. If T1 fails, the catalog alone supports a "quantitative leakage estimation" paper.
4. The catalog implementation validates the abstract interpretation framework, reducing integration risk.

This resequencing is adopted as a **binding condition** (see below).

---

## What Was Done Well

Credit is due for several aspects of this proposal that significantly exceed typical pre-implementation quality:

1. **Specification depth:** The 42.8KB `approach.json` is unusually thorough — it defines the partition-taint lattice formally, enumerates transfer functions for 15+ operations, and includes explicit tightness factors (κ). Most proposals at this stage have hand-wavy pseudocode.

2. **Honest limitations section:** The proposal itself identifies PCA bound vacuity for d > 50, the conditional-vs-unconditional soundness distinction, and the explaining-away concern. Self-identified limitations earn credibility even when self-assessment scores are inflated.

3. **Architecture decisions:** Choosing dynamic-first (Approach B) over Python static analysis was well-justified. The three-approach synthesis (B's architecture + C's tight bounds + A's decomposition lemma) demonstrates genuine design thinking. Dropping Galois insertions, general widening, and reduced product formalism (per Mathematician's recommendation) shows appropriate scope control.

4. **Prior art coverage:** The 32KB prior art survey covers security taint analysis (FlowDroid, Phosphor), QIF theory (Smith, Alvim et al.), differential privacy, neural verification, and ML systems tools. Clear differentiation from each area.

5. **Fallback decomposition:** The proposal identifies multiple independently publishable components (capacity catalog, dynamic DAG extraction, empirical leakage estimation), reducing all-or-nothing risk.

6. **Problem quantification:** The 15–25% Kaggle kernel leakage prevalence is not merely asserted — it's traced to documented studies. The "silent failure" framing (no crash, no error, just inflated metrics) is compelling and accurate.

---

## Binding Conditions

The following 11 conditions are merged from all six evaluators and are binding for continued development. Failure to meet any condition triggers the associated fail action.

### Proof & Theory (Weeks 1–4)

| # | Condition | Deadline | Success Criterion | Fail Action |
|---|-----------|----------|-------------------|-------------|
| BC-1 | **T1 proof or scoped restriction** | Week 4 | Either: (a) full T1 proof for all estimators, OR (b) T1 proof for linear estimators with Gaussian assumptions, OR (c) formally documented impossibility result for the general case with a concrete alternative bounding strategy. | **IMMEDIATE TERMINATION** of full-scope TaintFlow. Salvage catalog + empirical tool. |
| BC-2 | **Tight bound for StandardScaler** | Week 4 | Proven channel capacity bound with tightness factor κ ≤ 2 for `StandardScaler.fit_transform()` on Gaussian data. Machine-checkable proof (Lean, Coq, or detailed LaTeX with step-by-step verification). | Extend to Week 6. If still unproven, demote StandardScaler to Tier 2 (generic bound). |
| BC-3 | **PCA bound non-vacuity plan** | Week 4 | Written document specifying either: (a) dimension-reduced bound for d ≤ 50, (b) asymptotic bound for d → ∞ with explicit constants, or (c) explicit concession that PCA is Tier 2. | Flag as open problem. No termination, but best-paper probability drops to ≤ 1%. |

### Prototype & Implementation (Weeks 5–12)

| # | Condition | Deadline | Success Criterion | Fail Action |
|---|-----------|----------|-------------------|-------------|
| BC-4 | **Working DAG extraction** | Week 8 | Python prototype that correctly extracts PI-DAG for ≥ 5 scikit-learn pipelines (StandardScaler, PCA, OneHotEncoder, SimpleImputer, TargetEncoder) with verified row provenance via roaring bitmaps. | **IMMEDIATE TERMINATION.** If DAG extraction fails, the entire approach collapses. |
| BC-5 | **Capacity catalog (≥ 5 operations)** | Week 8 | Implemented and tested channel capacity bounds for ≥ 5 Tier 1 operations with unit tests validating bounds against empirical mutual information estimates (KSG). | Extend to Week 10. If still incomplete, reduce scope to 3 operations and reassess. |
| BC-6 | **End-to-end leakage report** | Week 12 | Full pipeline: DAG extraction → abstract analysis → leakage report for ≥ 1 known-leaky Kaggle pipeline, producing per-feature leakage in bits. | **IMMEDIATE TERMINATION** of tool-building effort. Pivot to theory-only publication. |
| BC-7 | **Build catalog before T1** | Continuous | Implementation must sequence: (1) DAG extraction, (2) capacity catalog, (3) T1 integration. T1 proof work proceeds in parallel on paper but is not integrated into code until catalog is stable. | Resequencing violation → forced 1-week pause for architectural review. |

### Publication & Scope (Weeks 8–14)

| # | Condition | Deadline | Success Criterion | Fail Action |
|---|-----------|----------|-------------------|-------------|
| BC-8 | **Venue decision** | Week 8 | Written venue selection with rationale. Must choose primary venue (ICML/NeurIPS/OOPSLA/FSE/USENIX Security) and backup venue. Submission deadline must be ≥ 8 weeks after venue decision. | No termination. But forces scope reduction to ensure any-venue submission. |
| BC-9 | **Comparison with 500-line script** | Week 10 | Quantitative comparison on ≥ 3 Kaggle pipelines: TaintFlow leakage spectrum vs. simple detection script. Must demonstrate clear quantification advantage (bits per feature vs. boolean flag). | If no measurable advantage, pivot to "improved detection" framing rather than "quantification." |
| BC-10 | **Self-assessment recalibration** | Week 8 | Updated self-assessment using adjudicated methodology. Must not exceed adjudicated scores by > 10% on any dimension. | Flag for external review. No termination. |
| BC-11 | **Correct State.json labels** | Week 1 | Update `status` from `"theory_complete"` to `"theory_specified"`. Fix theory_bytes path issue. | Trivial fix. No fail action needed. |

---

## Kill Gates

Five immediate-termination triggers, any one of which halts the project:

| Gate | Deadline | Criterion | Fail Action |
|------|----------|-----------|-------------|
| **KG-1: T1 Proof Gate** | Week 4 | No proof of T1 (full or restricted) AND no formal impossibility result. Silence or hand-waving does not pass. | TERMINATE full-scope TaintFlow. Salvage: capacity catalog paper + empirical detection tool. |
| **KG-2: DAG Extraction** | Week 8 | Cannot extract correct PI-DAG for ≥ 5 standard scikit-learn operations. | TERMINATE. DAG extraction is load-bearing infrastructure; if it fails, nothing downstream works. |
| **KG-3: End-to-End Demo** | Week 12 | No working end-to-end pipeline producing quantitative leakage output. | TERMINATE tool development. Pivot to theory-only paper on capacity bounds. |
| **KG-4: Soundness Violation** | Any time | Empirical evidence that abstract analysis produces unsound bounds (reports less leakage than measured). Single confirmed violation on a standard estimator. | PAUSE. Root-cause analysis. If fundamental (not implementation bug): TERMINATE. |
| **KG-5: Scope Explosion** | Week 10 | Total LoC exceeds 30K without proportional capability gain, OR timeline extends beyond 16 months. | PAUSE. Forced scope reduction to 15K LoC / 12-month target. If impossible: TERMINATE. |

---

## Recommended Path

### Path B: Core Tool (Recommended)

All six evaluators recommend **Path B** over Path A.

| Attribute | Path A (Original) | **Path B (Recommended)** |
|-----------|-------------------|--------------------------|
| **Language** | Rust + Python (PyO3 bridge) | Python (NumPy/SciPy for performance-critical sections) |
| **LoC Estimate** | 55–65K | **15–25K** |
| **Timeline** | 14–18 months | **10–14 months** |
| **Scope** | Full abstract interpretation engine, 80+ operations, Rust core | Core operations (15–20), Python throughout, optional Cython/Numba hot paths |
| **T1 Scope** | General (all estimators) | Restricted (linear estimators, Gaussian assumptions) unless full proof achieved |
| **Risk** | High (Rust/Python FFI, 40K LoC Rust engine, full T1) | **Moderate** (known-technology stack, scoped T1) |
| **Publication** | ICML/NeurIPS best-paper attempt | Solid OOPSLA/FSE/USENIX Security submission; ICML/NeurIPS if results exceed expectations |

### Implementation Sequence (Resequenced per Synthesizer)

```
Weeks 1–4:   [PARALLEL THEORY]  T1 proof attempt (paper only)
Weeks 1–4:   [IMPL]             DAG extraction prototype (sys.settrace + roaring bitmaps)
Weeks 3–4:   [IMPL]             StandardScaler capacity bound (first Tier 1 entry)
Weeks 5–8:   [IMPL]             Capacity catalog (5+ operations: mean, median, std, PCA≤50, target encoding)
Weeks 5–8:   [IMPL]             Abstract analysis engine (Python, partition-taint lattice)
Week 8:      [GATE]             T1 decision point — full/restricted/fail
Week 8:      [GATE]             Working prototype with ≥ 5 operations
Weeks 9–12:  [IMPL]             End-to-end integration, SARIF output, CLI tool
Week 10:     [EVAL]             Comparison with 500-line script on Kaggle pipelines
Weeks 12–14: [WRITE]            Paper writing, evaluation, submission
```

### Venue Recommendation

| Priority | Venue | Track | Submission Deadline | Fit |
|:--------:|-------|-------|--------------------:|-----|
| 1 | OOPSLA 2027 | Research | ~Apr 2027 | Strong. PL community values soundness. Conditional soundness precedent exists. |
| 2 | FSE 2027 | Research | ~Sep 2026 | Good. SE community values tools. Empirical evaluation expected. |
| 3 | USENIX Security 2027 | Research | ~Feb 2027 | Good. Security community values information flow. Novel application domain. |
| 4 | ICML 2027 | Systems | ~Jan 2027 | Moderate. ML community wants scale. Limited to systems track. |
| 5 | NeurIPS 2027 | Datasets & Benchmarks | ~May 2027 | Moderate. Leakage benchmark could fit D&B track. |

---

## Salvage Analysis

If TaintFlow fails at various stages, significant value remains:

### Salvage Component Value

| Component | Survives If... | Estimated Value | Publication Potential |
|-----------|---------------|-----------------|---------------------|
| **Capacity catalog** (tight bounds for 5–15 ML operations) | T1 fails, but individual bounds proven | High | Workshop paper or short paper at ICML/NeurIPS. Independently useful for QIF community. |
| **Dynamic DAG extraction** (sys.settrace + row provenance) | Any failure after Week 8 | Medium-High | Open-source tool. Useful for pipeline visualization even without leakage quantification. |
| **Empirical leakage estimation** (KSG-based MI estimation) | Any failure | Medium | Practical tool. Less novel but immediately useful. Publishable as a tool paper. |
| **Leakage benchmark** (curated Kaggle pipelines with known leakage) | Any failure | Medium | NeurIPS D&B track or standalone dataset paper. Community asset. |
| **500-line detection script** (enhanced) | Any failure | Low-Medium | Blog post or tool release. Captures 50–70% of detection value. |
| **Survey/position paper** (leakage taxonomy + information-theoretic framework) | Complete failure | Low-Medium | Survey paper covering QIF × ML pipeline semantics. |

### Expected Value Calculation

Using the adjudicated probabilities:

```
E[Value] = P(full success) × V(full) + P(partial) × V(partial) + P(fail) × V(salvage)

P(full success: T1 proven, tool works, paper accepted) ≈ 0.20
P(partial: restricted T1, tool works, paper accepted at lower venue) ≈ 0.35
P(tool works, no T1, empirical paper) ≈ 0.20
P(catalog + DAG extraction survive) ≈ 0.15
P(complete failure, only survey/benchmark) ≈ 0.10

Weighted by publication-equivalent value (1.0 = top-venue full paper):
E[Value] = 0.20(1.0) + 0.35(0.65) + 0.20(0.35) + 0.15(0.20) + 0.10(0.10)
         = 0.200 + 0.228 + 0.070 + 0.030 + 0.010
         = 0.538

Interpretation: Expected ~0.54 publication-equivalents.
This exceeds the typical theory-gate threshold of 0.30 for continuation.
```

### Key Salvage Insight

Even in the worst case (complete T1 failure + implementation difficulties), the capacity catalog + DAG extraction + benchmark combination yields ~0.20–0.30 publication-equivalents. The project has **positive expected value under all non-catastrophic failure modes.**

---

## Score Calibration

### Self-Assessment vs. Adjudicated

| Dimension | Self-Assessed | Adjudicated | Δ | Inflation |
|-----------|:---:|:---:|:---:|:---:|
| **V (Value)** | 8.0 | 6 | −2.0 | 33% |
| **D (Difficulty)** | 8.0 | 6 | −2.0 | 33% |
| **BP (Best-Paper)** | 6.5 | 4 | −2.5 | 63% |
| **L (Laptop-CPU)** | 9.0 | 9 | 0.0 | 0% |
| **F (Feasibility)** | 6.0 | 5 | −1.0 | 20% |
| **Composite** | 7.50 | 6.0 | −1.5 | 25% |

### Sources of Inflation

1. **LoC overcounting (V, D):** Claiming 55–65K LoC when 15–20K is genuinely novel inflates both value and difficulty. Boilerplate, glue code, and test scaffolding do not constitute novel software engineering.

2. **Best-paper probability (BP):** Self-assessed P(best-paper) = 12%. Adjudicated P(best-paper) = 2–4%. The gap reflects overweighting of theoretical novelty and underweighting of empirical validation expectations at ML venues. Conditional soundness is valued differently across communities.

3. **Premature "theory_complete" label (F):** Claiming theory completion when T1 has a known proof gap inflates the feasibility perception. Theory is specified, not completed.

4. **Scope asymmetry:** The self-assessment evaluates the ambitious Path A scope while the adjudicated score reflects the realistic Path B scope. Path A's difficulty is genuine but its feasibility is lower, creating a misleading difficulty × feasibility product.

5. **Laptop-CPU (L):** Only dimension with no inflation. Correctly assessed by all parties. This is a sanity-check anchor — the proposal's hardware requirements are genuinely minimal.

**Calibration note for future gates:** All milestone self-assessments should be benchmarked against this report's adjudicated scores. A > 10% inflation trigger (BC-10) is now a binding condition.

---

## Verdict

### Final Recommendation

**CONDITIONAL CONTINUE** at adjudicated composite **6.0/10**.

**Confidence:** Moderate (65%). The recommendation is robust to:
- T1 failing for the general case (Path B restricts to linear estimators)
- Best-paper falling to zero probability (tool paper at SE venue is sufficient)
- Implementation taking 14 months instead of 10 (within Path B scope)

The recommendation would flip to **ABANDON** if:
- T1 fails entirely (no restricted version) AND the capacity catalog proves vacuous for common operations
- DAG extraction via `sys.settrace` encounters fundamental Python runtime limitations
- A competing tool publishes quantitative leakage analysis before Week 12 (unlikely but possible)

### Specific Instructions for Next Phase

1. **Immediately** fix State.json: change `"theory_complete"` → `"theory_specified"`, resolve theory_bytes path bug (BC-11).
2. **Week 1:** Begin parallel tracks — T1 proof attempt (paper) + DAG extraction prototype (code).
3. **Week 4:** T1 proof gate. Hard deadline. No extensions. Results determine Path B scope.
4. **Week 8:** Prototype gate. Working DAG + ≥ 5 capacity bounds + venue decision.
5. **Week 12:** End-to-end demo gate. Quantitative leakage report on real Kaggle pipeline.
6. **Throughout:** Sequence catalog before T1 integration (BC-7). Build from concrete to abstract.

### Rankings JSON

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 6.0,
      "verdict": "CONTINUE",
      "reason": "Genuine novelty at QIF × abstract interpretation × ML pipeline intersection. Crown jewel T1 has proof gap but is likely fixable for restricted estimator set. Zero execution is concerning but specification quality is high. Path B (15-25K LoC Python, 10-14 months) well-scoped with clear kill gates. P(any publication) ≈ 55-65%. Conditional on 4-week proof gate and 8-week prototype.",
      "scavenge_from": []
    }
  ]
}
```

---

*Report prepared by the Lead Verifier on behalf of the 6-evaluator theory gate panel. This document is the definitive gate artifact for proposal_00 and supersedes all prior individual evaluations. Binding conditions and kill gates are enforceable starting from the date of this report.*

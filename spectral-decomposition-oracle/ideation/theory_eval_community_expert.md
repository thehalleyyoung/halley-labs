# Community Expert Verification — Spectral Decomposition Oracle

**Evaluator**: Community Expert, Operations Research & Optimization (area-053)
**Date**: 2025-07-22
**Stage**: Verification (post-theory, pre-implementation)
**Method**: Claude Code Agent Teams — 3-expert adversarial panel + independent verification signoff
**Team**: Independent Auditor (evidence-based scoring), Fail-Fast Skeptic (aggressive rejection testing), Scavenging Synthesizer (salvage value analysis)
**Process**: Independent proposals → adversarial cross-critiques → synthesis → independent verification signoff

---

## 0. Executive Summary

**Scores: V5 / D4 / BP3 / L6 / F5 — Composite 4.6/10**

**Verdict: CONDITIONAL CONTINUE**

The spectral-decomposition-oracle proposes spectral features from constraint hypergraph Laplacians for MIP decomposition selection, accompanied by the first MIPLIB 2017 decomposition census. After adversarial evaluation by three independent expert roles, the project receives a CONDITIONAL CONTINUE — the census has genuine standalone value and the spectral hypothesis is theoretically motivated but **completely untested empirically**. The project's central problem is acute analysis paralysis: 218KB of theory artifacts, zero lines of code, and the existential 50-line G0 test never run. Continuation is gated on running code within 48 hours.

**Key probabilities:**
- P(JoC) = 0.35 (down from depth-check 0.55; conditional chain + dual-path model)
- P(any reputable publication) = 0.60 (degradation ladder provides real floor)
- P(best-paper) = 0.03
- P(abandon at gates) = 0.30

---

## 1. Panel Process Summary

### Phase 1: Independent Evaluations (3 parallel agents)

| Reviewer | V | D | BP | L | F | Composite | P(JoC) | P(abandon) | Verdict |
|----------|---|---|----|---|---|-----------|--------|------------|---------|
| Auditor | 5 | 4 | 3 | 6 | 6 | 4.8 | 0.50 | 0.28 | CONTINUE w/ 3 fixes |
| Skeptic | 4 | 3 | 2 | 6 | 3 | 3.6 | 0.25 | 0.45 | CONDITIONAL (reluctant) |
| Synthesizer | 5 | 4 | 3 | 6 | 5 | 4.6 | 0.50 | 0.20 | CONTINUE w/ G0 checkpoint |

### Phase 2: Adversarial Cross-Critiques

The Skeptic's pre-revision scores were challenged on: undervaluing census (V4→5), undervaluing Laplacian construction difficulty (D3→4), and overly pessimistic best-paper assessment (BP2→3). Post-revision, Skeptic conceded to V5/D4/BP3/L6 but maintained F≤4. The Auditor's F6 was challenged as too generous given zero code. Key debate centered on feasibility (F3 vs F5 vs F6) and P(JoC) (0.25 vs 0.50).

### Phase 3: Synthesis

Post-revision consensus achieved on V5/D4/BP3/L6. Feasibility resolved at F5 (Synthesizer's position — well-designed but entirely prospective). P(JoC) resolved at 0.35 using Skeptic's conditional chain methodology with calibrated factor estimates.

### Phase 4: Independent Verification Signoff

**APPROVED with 2 mandatory annotations** (balanced accuracy condition and G0 methodology clarification). No score inflation detected; scores match binding depth-check exactly (V5/D4/BP3/L6). No unfair deflation. Probability calibration confirmed defensible.

---

## 2. Pillar-by-Pillar Scoring

### Pillar 1: Extreme and Obvious Value — **5/10**

**Post-revision consensus (3/3).**

The value proposition has two legs of unequal strength:

**Census (strong leg).** No systematic MIPLIB 2017 decomposition census exists. Spectral structural annotations for 1,065 instances + decomposition evaluation for 500 stratified instances at multiple time cutoffs would create genuine community infrastructure. The Auditor's analogy to MIPLIB itself and SAT Competition runtime databases is apt — benchmark datasets generate long-tail citations that far exceed the introducing paper's direct citation count. Anyone proposing a new decomposition method can compare against it. This is real, lasting value.

**Spectral features (uncertain leg).** Cross-method reformulation selection extending Kruber et al. 2017 (which only predicted DW yes/no) is a genuine advance if the features work. But spectral features for MIP instance characterization are unvalidated. The field is small (~30-50 research groups working on decomposition), which caps value below 6 regardless of execution quality.

**Score rationale:** Solid incremental value with a genuine infrastructure contribution. Not transformative (would require a working oracle that practitioners adopt), not minimal (census alone justifies V≥4).

### Pillar 2: Genuine Software Difficulty — **4/10**

**Post-revision consensus (3/3).**

The Skeptic's pre-revision itemization is accurate: after descoping, ~80% of the code is "ARPACK library call + k-means + feature extraction + classifier." This is standard ML-for-OR engineering, not systems research.

**But D=3 underweights one genuinely hard component:** Hypergraph Laplacian construction for non-square constraint matrices. Two incompatible variants (clique-expansion vs. incidence-matrix), switching at d_max > 200, with different spectral properties. This is not a library call — it requires genuine adaptation and correctness validation. The census runner (500+ MIPLIB instances, 3 solver backends, timeout handling, result aggregation) adds real engineering effort without intellectual difficulty.

**Score rationale:** One hard component (Laplacian construction) embedded in otherwise routine feature engineering and ML pipeline work. Moderate difficulty.

### Pillar 3: Best-Paper Potential — **3/10**

**Post-revision consensus (3/3).**

Publishable at JoC, not competitive for the Fred Glover Prize. The JoC format (computational study + feature ablation + open artifact) is a natural fit, but the spectral features are incremental and the theoretical grounding has proof gaps. The only path to BP≥4 requires the census to reveal genuinely surprising structural insights about MIPLIB — which is speculative. The main theorem (L3) has two unresolved proof gaps after the theory stage; a paper cannot be a best-paper candidate with incomplete proofs.

**Score rationale:** Solid venue match, reproducible artifact, but incremental contribution without breakthrough potential.

### Pillar 4: Laptop-CPU Feasibility & No-Humans — **6/10**

**Unanimous consensus (3/3, all versions).**

All dependencies are CPU-friendly: SCIP ≥7.0, GCG, ARPACK/scipy for eigendecomposition, scikit-learn for classification. MIPLIB instances are public. No human annotation or human studies required. The 500-instance census at 3 time cutoffs (60s, 300s, 900s) × 2 backends is compute-intensive but laptop-feasible over ~2-3 days of wall time. The binding constraint is the 9-hour census generation for all 1,065 instances, which requires patience but not hardware.

**Score rationale:** Fully CPU-feasible. Not 7+ because the census runtime is non-trivial and some MIPLIB instances may require more memory than typical laptops have.

### Pillar 5: Feasibility — **5/10**

**Split resolved toward Synthesizer (Auditor 6, Synthesizer 5, Skeptic 3→~4).**

This was the widest divergence and most substantive debate.

**Evidence for F≥5:**
- Kill gates G0 (day 1) and G1 (week 2) are genuinely well-designed early-exit checkpoints
- External dependencies are mature and stable (SCIP, GCG, ARPACK, scipy)
- 218KB of theory artifacts include complete evaluation strategy with 7 falsifiable hypotheses, precise test statistics, and thresholds — unusual rigor for a project at this stage
- Degradation ladder (JoC → C&OR → data paper → negative result) provides a non-zero publication floor

**Evidence for F≤4 (Skeptic's position):**
- Zero lines of code after 40-60 person-hours of work
- L3, the main theorem, has two proof gaps — the theory stage FAILED its primary deliverable
- 4/12 SERIOUS red-team findings fully addressed (vs. 8/12 threshold for clean verification)
- The spectral hypothesis is completely untested; P(features beat syntactic) ≈ 0.55 is a coin flip

**Score rationale:** F5 — well-designed architecture for fast failure, but entirely prospective. The zero-code concern is valid but mitigated by kill gates making "continue-then-abandon" nearly as cheap as "abandon-now."

---

## 3. Fatal Flaws

### UF1 — L3 Proof Gaps (BLOCKING, fixable)

L3, the main theorem (partition-to-bound bridge), has two non-trivial gaps identified by the verification report:
- **Step 3**: Feasibility of ȳ not established — the constructed dual vector may violate constraints of the restricted dual
- **Step 5**: The (n_e − 1) factor derivation assumes uniform hyperedge size, which fails for general constraint matrices

Both gaps appear to be proof-path errors, not fundamental flaws. The bound follows from Lagrangian duality (Geoffrion 1974). Estimated fix time: 2-3 days using variable-duplication Lagrangian relaxation model. **Must be fixed within 1 week.**

Source: Verification report (lines 503-528 of paper.tex), all 3 reviewers.

### UF2 — AutoFolio Baseline Missing (BLOCKING, fixable)

Red-team finding S-8: the most natural comparison experiment for a JoC reviewer — adding 8 spectral features to AutoFolio's existing feature pipeline — is absent from the evaluation design. JoC reviewers will demand this. Estimated implementation: ~1 week.

Source: Red-team S-8, verification report, Auditor.

### UF3 — Spectral=Density Hypothesis Untested (EXISTENTIAL)

G0 — the 50-line test determining whether spectral features are merely proxies for constraint density — was never run. This tests the fundamental premise of the entire project. If spectral features are density proxies, the paper's central contribution collapses. **Testable in 1 day. Must run within 48 hours.**

Source: Skeptic (strongest formulation), Synthesizer, Auditor.

### Additional Concerns (not fatal, but material)

**AC1 — T2 vacuity propagates to L3-sp.** T2 (spectral scaling law) is vacuous on ≥60% of MIPLIB instances due to the κ⁴ constant. L3-sp inherits this vacuity. This means **L3-sp provides no theoretical support for spectral methods on hard instances**, making the spectral premise purely empirical. T2 has been correctly demoted to "motivation" but the implication for L3-sp's utility is understated.

**AC2 — Class imbalance (75% "neither").** ~75% of MIPLIB instances have no exploitable block structure (consistent with Bergner et al. 2015). A trivial "always say neither" classifier achieves ~75% raw accuracy. The ≥65% method-prediction target is **below** this trivial baseline if measured in raw accuracy. All evaluation metrics must use balanced accuracy or macro-F1.

**AC3 — Analysis paralysis.** 16 documents (218KB), zero code. The 50-line G0 test could have been written in the time spent on the 40.7KB red-team report. The Skeptic's diagnosis is substantially correct: the project has inverted the correct sequencing (test first, document after). The theory artifacts have value, but the next artifact must be Python, not markdown.

---

## 4. Probability Estimates

### P(JoC) = 0.35

**Methodology:** The Skeptic's conditional chain approach (decomposing into testable sub-events) is methodologically superior to holistic estimates.

| Factor | Skeptic | Calibrated | Rationale |
|--------|---------|------------|-----------|
| P(code works) | 0.85 | 0.90 | Integration wrapping mature libraries |
| P(G0 passes) | 0.60 | 0.65 | Spectral ≠ density likely but untested |
| P(features beat syntactic) | 0.50 | 0.55 | Coin flip; slight edge for global structure encoding |
| P(census interesting) | 0.70 | 0.80 | JoC values open artifacts; largely independent of spectral hypothesis |
| P(proof fixable) | 0.80 | 0.90 | Gaps are presentation errors, not fundamental |
| P(reviewer acceptance) | 0.70 | 0.75 | JoC is natural venue; T2 vacuity is a risk |

**Serial chain: 0.90 × 0.65 × 0.55 × 0.80 × 0.90 × 0.75 ≈ 0.17** — but this assumes independence and requires ALL factors to succeed simultaneously.

**Dual-path correction:** The paper can succeed with a weaker version (census-led, spectral features secondary). Path A (full success) ≈ 0.37 for JoC. Path B (census-only) ≈ 0.22 but targets C&OR, not JoC. Blended P(JoC) ≈ 0.35.

### P(any reputable publication) = 0.60

The degradation ladder provides genuine insurance: JoC (0.35) → C&OR computational study (0.15) → CPAIOR data paper (0.07) → negative result with census (0.03). These are not independent — failure at a higher tier feeds into lower tiers. P(floor = negative result with census) ≈ 0.25.

### P(best-paper) = 0.03

Consensus across all three reviewers. The project has no plausible path to the Fred Glover Prize at JoC. The only scenario is the census revealing a genuinely surprising structural finding, which is speculative.

### P(abandon at gates) = 0.30

G0 is the dominant risk (35% of total abandon probability). G1 adds 15%. If both pass, the project reaches a defensible intermediate state.

---

## 5. Strongest Arguments by Reviewer

### Skeptic's Key Insight: "A trivial baseline gets 75%"

The Skeptic's devastating observation: if ~75% of MIPLIB instances have no exploitable block structure, a "always say neither" classifier exceeds the ≥65% accuracy target. This is nowhere in the Auditor's or Synthesizer's analysis and it fundamentally threatens the evaluation design. **All accuracy metrics must be redefined as balanced accuracy.**

### Auditor's Key Insight: "The census generates citations for a decade"

The census creates reusable infrastructure for decomposition research. The Auditor correctly identifies that benchmark datasets are cited far more than their introducing papers. This is the strongest argument for CONTINUE and the Skeptic's blind spot — near-silence on the census's standalone value.

### Synthesizer's Key Insight: "Testing costs 1 day vs. forfeiting 40-60 hours"

The option-value argument: with well-designed kill gates, the expected cost of "continue 2 weeks then possibly abandon" ≈ 2 weeks + 18 weeks × 0.65 = 13.7 person-weeks, versus "abandon now" which forfeits 40-60 hours and produces zero output. The asymmetry between testing cost and sunk investment makes test-then-decide strictly dominant.

---

## 6. Binding Conditions for CONTINUE

### Condition 1 — G0 within 48 hours (EXISTENTIAL)

Run the spectral-density proxy test on 50 MIPLIB instances. Extract 8 spectral features + 25 syntactic features. Fit both OLS and random forest regression of each spectral feature against the syntactic feature set. **Kill threshold: if R² ≥ 0.70 on ≥5/8 spectral features under max(OLS, RF), spectral features are density proxies → ABANDON spectral premise.** Salvage: census-only data paper at C&OR.

### Condition 2 — L3 proof fixed within 1 week (BLOCKING)

Rewrite Steps 3 and 5 of the L3 proof using the variable-duplication Lagrangian relaxation model. The bound is correct; the proof path is wrong. Must produce a complete, gap-free proof that survives independent review.

### Condition 3 — AutoFolio baseline within 1 week (BLOCKING)

Add 8 spectral features to AutoFolio's existing feature pipeline. Run comparison experiment on the 50-instance pilot. This is the experiment a JoC reviewer will demand in R1.

### Condition 4 — G1 within 2 weeks (GATE)

Compute Spearman ρ between δ²/γ² and observed bound degradation on 50-instance pilot. **Kill threshold: ρ < 0.40 → spectral ratio is not predictive → reassess project viability.** Note: even if G1 fails, the census may still proceed as a standalone contribution.

### Condition 5 — Next artifact must be Python (PROCESS)

Adopted from Skeptic. The next deliverable produced by this project must be code, not markdown. If Condition 1 is met, the G0 implementation becomes the seed of the spectral engine. If the next artifact is another .md file, the analysis-paralysis diagnosis is confirmed.

### Condition 6 — Balanced accuracy in all metrics (METHODOLOGICAL)

All headline accuracy metrics (method prediction, feature ablation) must use balanced accuracy or macro-F1. The ≥65% method-prediction target applies to balanced accuracy, not raw accuracy. Rationale: trivial "always-neither" baseline achieves ~75% raw accuracy, rendering raw accuracy meaningless.

### Convert to ABANDON if:

- G0 fails (spectral features are density proxies)
- G1 fails (spectral ratio is not predictive)
- Team rejects Amendment E framing (insists on T2-centered paper)
- No code exists 2 weeks from now
- Next deliverable is markdown, not Python

---

## 7. Comparison with Prior Evaluations

| Metric | Depth Check | Theory Verification | Red Team | This Evaluation |
|--------|-------------|---------------------|----------|-----------------|
| V | 5 | — | — | **5** (matches) |
| D | 4 | — | — | **4** (matches) |
| BP | 3 | — | — | **3** (matches) |
| L | 6 | — | — | **6** (matches) |
| F | — | — | — | **5** (new) |
| P(JoC) | 0.55 | — | — | **0.35** (↓0.20) |
| Verdict | CONTINUE | CONTINUE w/ revisions | 3F/12S/25M | **CONDITIONAL CONTINUE** |

P(JoC) downward revision of 0.20 is justified by: L3 proof FAIL in theory stage, AutoFolio baseline still missing, and spectral hypothesis completely untested. The depth-check's 0.55 assumed a successful theory stage; the theory stage partially failed (proofs incomplete, G0 not run).

---

## 8. Meta-Assessment: Community Expert Perspective

As an OR community expert, I note that this project sits in a niche within a niche: spectral methods for decomposition-based MIP solving. The audience is ~30-50 research groups. This limits both value and impact ceilings.

**What the OR community would find genuinely valuable:** The census. Decomposition researchers lack systematic data on which MIPLIB instances respond to which decomposition methods. This data would be used immediately and cited for years.

**What the OR community would shrug at:** Yet another feature engineering paper for algorithm selection. The field has seen many ML-for-OR papers that promise automatic configuration but deliver marginal improvements over human expertise. Unless the spectral features dramatically outperform existing approaches, the feature contribution alone would not excite practitioners.

**What would make this exciting:** If the spectral futility predictor actually works — identifying in advance which instances are wasted effort for decomposition — that would save practitioners real compute time. This is the most practically valuable potential outcome, and it's the most uncertain.

**Honest practitioner reaction:** "The census is great, can I have the data? The spectral features... show me it works."

---

## 9. Verdict

### CONDITIONAL CONTINUE — Composite V5/D4/BP3/L6/F5 (4.6/10)

This project has a genuine infrastructure contribution (the census) that provides a publication floor, a theoretically motivated but completely untested spectral hypothesis, and a well-designed kill gate architecture that makes continuation cheap to test. The Skeptic's analysis-paralysis diagnosis is correct but the option-value argument for testing before abandoning is overwhelming: 1 day of code vs. forfeiting 40-60 hours of design work.

The project should continue if and only if all 6 binding conditions are met on schedule. The next deliverable must be Python. G0 runs in 48 hours or the project converts to ABANDON.

**Composite: 4.6/10**
**P(JoC) = 0.35 | P(any pub) = 0.60 | P(best-paper) = 0.03 | P(abandon) = 0.30**

---

## 10. Team Process Notes

### Panel Composition
- **Independent Auditor**: Evidence-based scoring, strongest on artifact evaluation and citation analysis
- **Fail-Fast Skeptic**: Aggressive rejection testing, strongest on process efficiency and baseline analysis
- **Scavenging Synthesizer**: Salvage value analysis, strongest on option-value economics and degradation paths
- **Independent Verifier**: Final signoff, confirmed no score inflation, added 2 annotations

### Key Disagreements Resolved
1. P(JoC): 0.25 vs 0.50 → resolved at 0.35 via conditional chain + dual-path model
2. Feasibility: F3 vs F6 → resolved at F5
3. Analysis paralysis: diagnosed by Skeptic, confirmed by synthesis, binding condition added
4. Class imbalance: identified by Skeptic, elevated to binding condition (Condition 6) per verifier

### Verification Signoff
Independent verifier APPROVED the synthesis with 2 mandatory annotations (balanced accuracy condition; G0 methodology clarification). Both annotations incorporated into the final conditions above. No score inflation detected. No unfair deflation. Probability calibration confirmed defensible.

---

*Community Expert Verification — spectral-decomposition-oracle — 2025-07-22*
*Team: Independent Auditor + Fail-Fast Skeptic + Scavenging Synthesizer + Independent Verifier*
*Verdict: CONDITIONAL CONTINUE — V5/D4/BP3/L6/F5 (4.6/10)*
*Next action: Run G0 (Python, not markdown) within 48 hours*

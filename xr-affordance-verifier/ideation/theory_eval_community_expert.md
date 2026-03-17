# Community Expert Verification: Coverage-Certified XR Accessibility Verifier (proposal_00)

**Project:** xr-affordance-verifier
**Stage:** Post-theory verification
**Panel:** Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer
**Date:** 2026-03-08
**Prior scores:** Depth check 20/40 (V4/D7/BP4/L5) → Final approach 25/40 (V6/D6/P6/F7)

---

## Executive Summary

A 3-expert panel evaluated proposal_00 after extensive theory development (300KB+ across 7 structured documents including formal proofs, algorithms, empirical plans, adversarial red-teaming, and cross-validated synthesis). The panel split: **Auditor CONDITIONAL CONTINUE (28/50, 55%)**, **Skeptic ABANDON (18/50, 85%)**, **Synthesizer CONTINUE with radical rescoping (30/50)**. After adversarial cross-critique, verification signoff, and three mandatory revisions from the signoff reviewer, the adjudicated verdict is:

**CONDITIONAL CONTINUE at 55% confidence.** Composite **25/50** (V4/D5/BP4/L7/F5). Gated by 9 mandatory amendments, externally enforced kill-chain at D1/D2/D3, and a binding novelty verification of Decision 7 before implementation begins. P(any publication) ≈ 20–30%. P(best paper) ≈ 1–3%. Kill probability ≈ 50%. Maximum sunk cost before critical kill gates: 2 months.

---

## Panel Scoring

| Axis | Auditor | Skeptic | Synthesizer | Adjudicated | Justification |
|------|---------|---------|-------------|-------------|---------------|
| **Extreme Value** | 5 | 2 | 6 | **4** | Empty niche is real (zero existing XR accessibility tools), but zero validated demand, microscopic market (30–50K XR developers), and the domain-general certificate angle is speculative. A 500-LoC lookup table captures a large fraction of practical value. |
| **Genuine Difficulty** | 6 | 5 | 5 | **5** | A-grade math (M2 zone abstraction, M3b treewidth decomposition) was abandoned. Remaining difficulty centers on B+ crown jewel (coverage certificate C1) and affine-arithmetic FK wrapping through 7-joint chains. ~21–37K genuinely difficult LoC. Solid engineering, moderate research novelty. |
| **Best-Paper Potential** | 5 | 2 | 5 | **4** | UIST paper dead under no-humans constraint. CAV framing weakened (B+ crown jewel, mediocre ε). ISSTA pivot ("abstract interpretation meets parameterized testing") is the strongest viable framing but contingent on Decision 7's novelty being confirmed via literature review. Acceptance probability 15–25% at best-fit venue. Best-paper ≈ 1–3%. |
| **Laptop-CPU & No-Humans** | 7 | 6 | 8 | **7** | All computation is CPU-friendly (interval arithmetic, FK evaluation, stratified sampling, Z3 on QF_LRA). No GPU needed. The no-humans constraint kills the UIST developer study (15–22 participants + IRB) but the Synthesizer's 5-part alternative evaluation (mutation testing, baseline ladder, ε convergence curves, κ characterization, cross-domain transfer) is methodologically sound for ISSTA/FSE. |
| **Feasibility** | 5 | 3 | 6 | **5** | Research success probability 8–15% (Auditor's corrected chain, adjudicated across experts' flawed calculations). P(useful Tier 1 linter) ≈ 75%. P(paper-ready system) ≈ 25–35%. 2-month kill-chain limits downside exposure. ~65% compound risk of at least one critical failure (project's own estimate). |
| **Composite** | **28/50** | **18/50** | **30/50** | **25/50** | |

---

## Theory Stage Assessment

### What the Theory Stage Accomplished (Strengths)

1. **Thorough adversarial self-critique.** The theory stage's crown achievement is its honesty. The Red-Team independently confirmed that SMT-verified volume is negligible (10⁻⁹), that ε targets were 4–6× too optimistic, and that the Lipschitz assumption fails where it matters most. These findings were incorporated into binding synthesis decisions, not buried.

2. **Novel synthesis finding (Decision 7).** Crediting Tier 1 interval-arithmetic envelopes as symbolically verified volume in the coverage certificate — identified during cross-proposal synthesis — is the project's most promising intellectual contribution. It connects abstract interpretation (conservative over-approximation produces sound accessibility proofs) with parameterized testing (verified volume tightens coverage certificates). If novel, this insight carries a paper.

3. **Well-designed kill-chain.** Gates D1 (Month 1, wrapping factor), D2 (Month 2, certificate ε), D3 (Month 2, Clopper-Pearson comparison) test the three critical uncertainties within 2 months, limiting maximum sunk cost before the project either validates its core mechanisms or terminates.

4. **Piecewise-Lipschitz formulation with κ-completeness.** The synthesis correctly restated C1 for piecewise-Lipschitz frontiers, tracking excluded volume via the κ metric. This is more honest than global Lipschitz and gives the certificate a unique structural advantage over Monte Carlo: it explicitly quantifies what it covers and what it doesn't.

5. **Dual-ε reporting.** The analytical/estimated ε split (ε_analytical from kinematic Jacobian, provably sound but loose; ε_estimated from cross-validation, tighter but not provably sound) eliminates the Lipschitz estimation circularity and gives the paper a clean formal result for the soundness theorem.

### What the Theory Stage Revealed (Weaknesses)

1. **SMT volume is negligible (10⁻⁹).** The "sampling-symbolic hybrid" story collapsed when both the Red-Team and Algorithm Designer independently confirmed that SMT-verified volume is essentially zero. The reframing to "Tier 1 envelopes + sampling + targeted SMT for frontier-resolution" is architecturally sound but represents a major scope reduction from the original vision.

2. **ε achievability is mediocre.** The original ε < 0.01 target was relaxed to ε < 0.05 (hard) / ε < 0.02 (stretch). At ε ≈ 0.04–0.06, the certificate says "at most a 4–6% chance of an undetected accessibility bug." This is both harder to explain and less impressive than Monte Carlo's "tested 4M body configurations, found zero failures." The comparison is "misleading because they answer different questions" — technically correct, rhetorically unconvincing.

3. **Frontier-resolution is unproven.** The key mechanism for achieving ≥3× improvement over Clopper-Pearson (SMT queries at the accessibility frontier provide information-theoretic value via Lipschitz interpolation) is classified as "plausible but unproven" and "empirical enhancement, not theorem." Without it, improvement from Tier 1 verified volume alone is ~1.7×.

4. **Lipschitz exemption is self-defeating.** The certificate exempts joint-limit boundary populations — wheelchair users with limited ROM, elderly individuals near joint-limit transitions — which are the exact populations most affected by spatial accessibility failures. The κ-completeness metric quantifies this honestly, but "we verified accessibility for 75% of the population, excluding the 25% most likely to have accessibility problems" is a deeply uncomfortable result for a tool claiming to verify accessibility.

5. **Best ideas are dead.** The A-grade mathematical contributions (M2 SE(3) zone abstraction, M3b bounded-treewidth decomposition, PGHA semantics) were all abandoned due to intractability. What remains is competent engineering of known techniques (affine arithmetic, stratified sampling, Hoeffding bounds) composed into a domain-specific tool with a modestly novel certificate format.

---

## Critical Disagreements and Resolutions

### Disagreement 1: Is the Certificate Contribution Alive or Dead?

**Skeptic's position:** Dead. ε ≈ 0.05 is worse than Clopper-Pearson's ε_CP ≈ 10⁻⁶ from the same samples. The "they measure different things" defense is technically correct but unconvincing. A reviewer will compute the comparison and reject.

**Synthesizer's position:** Alive but reframed. The contribution is not ε-tightness but the paradigm unification: abstract interpretation verdicts contribute verified volume to statistical testing certificates. This is structurally novel even if quantitatively modest.

**Resolution:** The Skeptic's ε-vs-CP argument is the strongest factual observation in the panel and has not been defeated. The Synthesizer's reframing is *plausible* but *contingent* on Decision 7's novelty being confirmed via literature review (Amendment A9). If prior art exists in concolic testing / symbolic-execution-sampling hybrids / abstract-interpretation-guided fuzzing, the rescue argument collapses. **Both experts are conditionally correct; the literature review resolves it.**

### Disagreement 2: ROI Calculation

**Skeptic:** 1.5% research success → catastrophic ROI.
**Synthesizer:** 35% paper-ready → positive expected value.
**Auditor:** 7–14% research success → marginal but above threshold.

**Resolution:** The Skeptic's 1.5% uses a flawed independence model (ε < 0.05 and ≥3× over CP are positively correlated) and single-venue assumption. The Synthesizer's 55% "at least one paper" conflates paper-ready with paper-accepted and assumes venue independence when both submissions share the same results. **Adjudicated: 8–15% research success probability.** This is poor but not catastrophic for a project with a 2-month kill-chain and a useful artifact floor.

### Disagreement 3: Does the Lookup Table Kill the Value Proposition?

**Skeptic:** A 500-LoC lookup table captures 90%+ of value.
**Synthesizer:** The 90% figure is fabricated; lookup tables can't handle arbitrary 3D configurations, multi-joint coupling, or continuous anthropometric parameters.

**Resolution:** The Synthesizer is correct that the Skeptic's "90%" is asserted without methodology. But the Skeptic is correct that the marginal value of formal verification over simple heuristics is uncertain. **Amendment A3 (Month 2) mandates measuring Tier 1's marginal detection rate over the lookup-table baseline.** If <10% additional bugs are caught, the engineering complexity of affine arithmetic is hard to justify.

---

## Scores (Final, Adjudicated)

| Axis | Score | Change from Final Approach | Key Factor |
|------|-------|---------------------------|------------|
| **Extreme Value** | **4/10** | −2 | Zero demand unchanged; SMT collapse weakened differentiation; domain-general angle is speculative |
| **Genuine Software Difficulty** | **5/10** | −1 | A-grade math abandoned; remaining B+ crown jewel is moderate composition of known techniques |
| **Best-Paper Potential** | **4/10** | −2 | UIST dead (no-humans); CAV weakened (mediocre ε); ISSTA pivot promising but contingent on Decision 7 novelty |
| **Laptop-CPU & No-Humans** | **7/10** | +0 | CPU solid; mutation testing + baseline ladder replaces developer study |
| **Feasibility** | **5/10** | −2 | Theory stage revealed mechanism weaknesses; 65% compound risk; UIST path blocked |
| **Composite** | **25/50** | — | Down from 25/40 (different scale: now /50 with 5 axes instead of /40 with 4) |

---

## Fatal Flaws

| # | Flaw | P(fatal) | Impact | Testable? |
|---|------|----------|--------|-----------|
| **F1** | Certificate cannot beat Monte Carlo quantitatively | 40% | Research narrative collapses; tool paper only | **YES** — D3, Month 2 |
| **F2** | Lipschitz exemption excludes target populations (κ > 0.25) | 25% | Ethical and practical credibility issue | **YES** — D4, Month 2–3 |
| **F3** | Frontier-resolution doesn't work | 45% | ε improvement capped at ~1.7× (Tier 1 volume only) | **YES** — D3, Month 2 |
| **F4** | UIST paper dead under no-humans constraint | 100% | Project output reduced to single paper | Certain |
| **F5** | Decision 7 novelty already exists in literature | 20% | ISSTA framing collapses; best-paper drops to 2–3 | **YES** — Literature review, Week 1 |
| **F6** | Wrapping factor > 10× on 7-joint chains | 20% | Tier 1 linter and Decision 7 both killed | **YES** — D1, Month 1 |
| **F7** | Analytical L_max is vacuous (L_max/L̂ > 20×) | 30% | ε_analytical > 0.5; formal guarantee meaningless | **YES** — Month 2 |
| **F8** | Zero validated user demand | 35% | Tool works but nobody uses it | **PARTIALLY** — D7, Month 3 |

**Compound risk (≥1 critical failure):** ~65% (project's own estimate, validated by panel).

---

## Mandatory Amendments

| # | Amendment | Deadline | Kill Criterion | Enforcement |
|---|-----------|----------|----------------|-------------|
| **A1** | Wrapping factor test: 4-joint ±30° (D1a) AND 7-joint realistic ranges (D1b) | **Month 1** | D1a: w > 5× → switch to Taylor models. D1b: w > 10× after subdivision → **ABANDON Tier 1** | External reviewer |
| **A2** | Certificate ε prototype on 10-object benchmark | **Month 2** | ε > 0.10 → **ABANDON certificate framework**. Fall back to Tier 1 tool paper only. | External reviewer |
| **A3** | Clopper-Pearson comparison + Tier 1 marginal detection vs. lookup table | **Month 2** | Certificate improvement < 2× over CP → **DOWNSCOPE** to tool paper. Tier 1 marginal < 10% over lookup table → document as limitation. | External reviewer |
| **A4** | Rescope Paper 2 to eliminate developer study | **Before Month 3** | Replace with mutation testing + baseline ladder evaluation. Target ISSTA/FSE (not UIST) or ICSE-Tool. | Self-enforced |
| **A5** | Lipschitz violation characterization: measure κ on 50+ benchmark scenes | **Month 3** | κ > 0.25 on >30% of scenes → **DOWNSCOPE** certificate applicability claim. Label all certificates as "partial" in publications. | External reviewer |
| **A6** | Real XR scene evaluation: ≥10 scenes, ≥5 complete full pipeline | **Month 6** | <5 real scenes → downscope to procedural benchmarks. Acknowledge limitation. | Self-enforced |
| **A7** | Anthropometric data audit: document ANSUR-II biases, incorporate disability-specific ROM data or acknowledge gap | **Month 3** | Do not claim "verifies accessibility for disabled users" without disability-specific kinematic data. | Self-enforced |
| **A8** | ε vs. budget curve: report ε(T) for T ∈ {1, 5, 10, 30, 60} minutes | **Month 4** | No kill criterion. Mandatory reporting for honest evaluation. | Self-enforced |
| **A9** | **Literature verification of Decision 7 novelty** (NEW — from verification signoff) | **Week 1, before implementation** | Search ISSTA/FSE/ICSE 2018–2025 for any work crediting abstract-interpretation/interval-analysis verdicts as verified volume in statistical testing certificates. If prior art found → Best-Paper drops to 3, re-evaluate CONTINUE. | External reviewer |

**Kill-chain enforcement:** Amendments A1, A2, A3, A5, A9 require external reviewer signoff. The project's demonstrated tendency toward optimistic reframing (ε target relaxed 5×, scope repeatedly narrowed, every setback repackaged as "pragmatic pivot") means self-enforcement is insufficient. An external reviewer receives D1/D2/D3/D4 results and has binding authority to terminate.

---

## Recommended Scope (If Continuing)

### What Stays

1. **Tier 1 interval-arithmetic linter** (~12–15K LoC). Zero-config Unity plugin. First-of-kind. 85% standalone success probability. Useful artifact regardless of research outcomes.

2. **Coverage certificate framework with Tier 1 verified volume** (~10–15K LoC). C1 soundness theorem under piecewise-Lipschitz. κ-completeness tracking. Dual-ε reporting. Decision 7 integration.

3. **Stratified sampling engine** (~5–8K LoC). Latin hypercube, frontier-adaptive allocation, Pinocchio FK evaluation.

4. **Evaluation: mutation testing + baseline ladder + ε convergence curves + κ characterization + cross-domain toy examples** (no human studies).

### What's Cut

1. **SMT component entirely.** Volume ≈ 10⁻⁹. Frontier-resolution is unproven. Cut linearization engine, Z3 integration, sampling-symbolic handoff. Saves ~10–15K LoC. **Exception:** If frontier-resolution is later validated empirically (D3 stretch target), re-add targeted SMT at Month 3.

2. **Multi-step verification (k > 1).** Curse of dimensionality makes certificates hopelessly loose at k ≥ 3. Restrict to single-step. Saves ~5K LoC.

3. **Developer study.** Violates no-humans constraint. Replace with mutation testing + baseline ladder.

4. **DSL.** Zero evidence developers would write annotations. Pattern-matched extraction for XR Interaction Toolkit idioms is sufficient.

5. **UIST paper as originally designed.** Replace with ISSTA/FSE paper (coverage certificates as parameterized testing framework) + ICSE-Tool backup (Tier 1 linter).

### Resulting Scope

- **Total LoC:** ~28–40K (down from 43–68K)
- **Difficult LoC:** ~13–22K (down from 21–37K)
- **Timeline:** 6 months (Tier 1 in 2 months, certificate by Month 4, evaluation by Month 6)
- **Primary venue:** ISSTA 2027 or FSE 2027
- **Backup venue:** ICSE 2027 Tool Track

---

## Best-Paper Strategy (Revised)

### Paper 1 (Primary): ISSTA/FSE — "Coverage Certificates for Parameterized Testing"

**Pitch:** Coverage certificates that combine abstract-interpretation verified volume with stratified sampling, bounding P(undetected failure) over continuous parameter spaces. Domain-general framework demonstrated on XR accessibility + 2 toy case studies (robotic workspace, dosage safety).

**Core contribution:** Decision 7 — interval-arithmetic soundness inverted as verified volume in parameterized testing certificates. Novel connection between abstract interpretation and statistical testing.

**Acceptance probability:** 15–25% (conditional on Decision 7 novelty confirmed, ε ≥ 2× over CP).
**Best-paper probability:** 1–3%.

### Paper 2 (Backup): ICSE Tool Track — "AccessLint-XR"

**Pitch:** First spatial accessibility linter for XR development. Zero-config Unity plugin. Mutation testing evaluation on 500+ scenes.

**Acceptance probability:** 25–35%.
**Best-paper probability:** <1% (tool track).

---

## VERDICT: CONDITIONAL CONTINUE

**Confidence: 55%** (revised down from 60% per verification signoff — the Auditor gave 55% at a higher composite score; the Lead cannot justify more confidence at a lower score)

**Rationale for CONTINUE:**

1. **2-month kill-chain limits downside.** D1 (Month 1) and D2/D3 (Month 2) test all critical mechanisms. Maximum sunk cost before kill: 2 person-months.

2. **Decision 7 is genuinely novel** (pending A9 literature verification). The abstract-interpretation ↔ parameterized-testing connection is an insight that transfers across verification domains and could carry a paper.

3. **Tier 1 linter has standalone value.** First-of-kind artifact at 85% delivery probability. Useful engineering output even on failure paths.

4. **Theory stage quality is high.** The 300KB+ of structured output with adversarial red-teaming represents thorough preparation. Problems were found before implementation, not after.

**Rationale for conditionality (what would flip to ABANDON):**

1. **A9 literature review finds prior art** for Decision 7's insight → ISSTA framing collapses → Best-Paper drops to 2–3 → re-evaluate.

2. **D1 fails** (wrapping > 10× with subdivision + Taylor models) → Tier 1 is dead, Decision 7 is dead → ABANDON.

3. **D3 fails** (certificate improvement < 2× over CP) → Certificate is a marginal result → DOWNSCOPE to Tier 1 tool paper only (ICSE-Tool).

4. **Kill-chain enforcement fails** (no external reviewer exercises binding authority) → Expected outcome degrades to Skeptic's worst case (6–9 months for mediocre paper) → ABANDON.

**The Skeptic's dissent is recorded.** At 18/50 with 85% confidence on ABANDON, the Skeptic argues the project's crown jewel is dead (certificate can't beat MC), the best ideas are abandoned (M2, M3b killed), the publication path is collapsed (UIST dead), and the ROI is catastrophic (1.5% research success). The panel majority does not accept the 1.5% figure (flawed independence + single-venue assumption; corrected to 8–15%) but acknowledges the Skeptic's core critique — that the certificate's ε comparison with Clopper-Pearson is genuinely unfavorable — as the project's most serious unresolved risk. If D3 confirms the Skeptic's prediction (improvement < 2×), the project should be killed or radically downscoped.

---

## Summary

| Metric | Value |
|--------|-------|
| **Composite score** | **25/50** (V4/D5/BP4/L7/F5) |
| **Verdict** | **CONDITIONAL CONTINUE** |
| **Confidence** | **55%** |
| **P(any publication)** | **20–30%** |
| **P(best paper)** | **1–3%** |
| **P(useful artifact)** | **~65%** |
| **Kill probability** | **~50%** |
| **Maximum sunk cost before kill** | **2 months** |
| **Mandatory amendments** | **9** (A1–A9) |
| **Externally enforced gates** | **5** (A1, A2, A3, A5, A9) |
| **LoC (rescoped)** | **~28–40K total, ~13–22K difficult** |
| **Timeline** | **6 months** |
| **Primary venue** | **ISSTA/FSE 2027** |
| **Backup venue** | **ICSE 2027 Tool Track** |

---

*Panel: Independent Auditor (28/50, CONDITIONAL CONTINUE 55%), Fail-Fast Skeptic (18/50, ABANDON 85%), Scavenging Synthesizer (30/50, CONTINUE with radical rescoping). Cross-critique adjudicated range: 22–28/50. Verification signoff: CONDITIONALLY APPROVED with 3 revisions (all incorporated). End of community expert verification.*

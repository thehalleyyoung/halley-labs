# Skeptic Verification: TaintFlow (proposal_00)

**Stage:** Verification (post-theory)
**Reviewer Persona:** Rigorous Skeptic — "I cannot recommend acceptance unless the evidence compels me."
**Panel:** 3-expert team (Independent Auditor + Fail-Fast Skeptic + Scavenging Synthesizer) with adversarial cross-critique
**Date:** 2026-03-08
**Proposal:** TaintFlow — Quantitative Information-Flow Auditing via Hybrid Dynamic-Static Analysis with Provably Tight Channel Capacity Bounds

---

## Executive Summary

TaintFlow proposes a hybrid dynamic-static ML pipeline auditor that instruments a Python pipeline once to extract a dataflow DAG, then applies provenance-parameterized channel capacity bounds to produce per-feature, per-stage leakage measurements in bits. After three independent evaluations and adversarial cross-critique, the panel consensus is:

**Composite: 5.90/10** (self-assessed 7.50; 21% inflation).
**Verdict: CONDITIONAL CONTINUE** — unanimous across all three experts, with confidence 70%.

The proposal has genuine novelty at the intersection of QIF × abstract interpretation × ML pipeline semantics. The fit-transform channel decomposition is a real contribution. The problem is real and widespread. However: zero proofs exist, zero code exists, the "theory_complete" label is false, the crown jewel theorem has a proof sketch gap, the evaluation design cannot validate the core differentiator on real data, and the self-assessment is systematically inflated. Continuation is conditional on proving T1 within 30 days and delivering a working prototype within 60.

---

## Panel Composition and Process

| Role | Function | Verdict | Composite |
|------|----------|---------|-----------|
| **Independent Auditor** | Evidence-based scoring, challenge testing | CONTINUE (65%) | 6.25 |
| **Fail-Fast Skeptic** | Aggressive rejection of under-supported claims | CONTINUE (65%) | 5.60 |
| **Scavenging Synthesizer** | Salvage value analysis, risk-adjusted path planning | CONTINUE (82%) | 7.20 |
| **Lead (Adversarial Adjudication)** | Cross-critique synthesis, consensus adjudication | CONTINUE (70%) | **5.90** |

**Process:** Independent parallel evaluations → adversarial cross-critique (each expert challenged the other two) → synthesis of strongest elements → consensus scores adjudicated by lead.

---

## 1. Measurement Anomalies

### theory_bytes=0 — File-Path Bug, Not Absence of Work

State.json records `theory_bytes: 0`, but `theory/approach.json` is **42.8KB** (624 lines) containing 6 definitions, 5 theorems with proof sketches, 5 algorithms, a 16-operation capacity catalog, complexity analysis, implementation strategy, evaluation plan, and red-team assessment. The system was looking for artifacts in `proposals/proposal_00/theory/` (empty), while theory work landed at top-level `theory/approach.json`. **This is a pipeline orchestrator file-path mismatch, not missing work.**

### impl_loc=0 — Genuine

Zero implementation code exists. The project is at the specification stage. All work product is ideation and theory planning.

### "theory_complete" — FALSE

The status should read **"theory_specified"** or **"theory_sketched."** What exists is proof *sketches* (3-8 sentence outlines), not proofs. No LaTeX drafts, no formal derivations, no counterexample analysis. The gap between a plausible sketch and a rigorous proof is where projects die. This label must be corrected immediately.

---

## 2. Three Pillars Assessment

### Pillar 1: Extreme and Obvious Value — **6/10** (self-assessed: 8)

**The problem is real.** Train-test leakage is the #1 silent failure mode in ML pipelines. 15-25% of Kaggle kernels contain leakage (Yang et al., ASE 2022). Practitioners have zero quantitative diagnostic tools.

**But the value is not "extreme."** Key limitations:
- **Execution requirement:** TaintFlow requires running the pipeline — it's a self-diagnostic, not an external audit tool. Cannot audit third-party code without data access.
- **Practitioner interpretability:** "3.7 bits of leakage" is meaningless to most ML engineers. "Is 3.7 bits bad?" requires domain knowledge that practitioners lack. A simple linter saying "you called fit_transform before train_test_split" delivers 80% of the actionable value.
- **Coverage gaps:** Custom transformers → ∞ bounds. PCA → vacuous bounds for d > 50 (and PCA is one of the most common preprocessing steps). KNNImputer, IterativeImputer → excluded entirely.
- **The 500-line script challenge (Skeptic's strongest attack):** A ~500-line Python script using sys.settrace + textbook Gaussian channel formulas catches ~65-70% of leakage cases for binary detection and ~40-50% for quantification. On the proposed evaluation (Yang et al.'s binary labels), this script's measurable performance would be similar to TaintFlow's.

**Why 6, not lower:** The formal framework provides genuine value beyond the script in: (a) compositional reasoning through long pipeline DAGs, (b) "known unknowns" reporting (tight bound → loose bound → ∞), (c) formal justification enabling CI/CD gates, (d) extensibility to new operations. The value is real but narrower than claimed — research ML platform teams, not individual practitioners.

**Panel divergence:** Auditor=6, Skeptic=6, Synthesizer=8. The Synthesizer's 8 overweights the "novel intersection" narrative and underweights adoption barriers.

### Pillar 2: Genuine Software Difficulty — **6/10** (self-assessed: 7)

**Real difficulty exists:**
- Instrumenting 300+ pandas APIs with correct provenance tracking is non-trivial (reentrancy, exception paths, version compatibility).
- 130 transfer functions under a formal soundness claim require getting each one right — a single unsound transfer function produces silent false negatives.
- Python-Rust bridge via PyO3 adds integration complexity.
- Cross-language testing and debugging across Rust + Python is harder than either language alone.

**But the Skeptic has a point about the novel core:**
- The partition-taint lattice is a product of small finite lattices with height 68. No widening needed. The Rust implementation is ~500 lines of genuinely novel code plus plumbing.
- The worklist algorithm is textbook (Tarski's fixpoint theorem + RPO ordering).
- The 130 transfer functions decompose into ~5 templates instantiated 130 times. Most operations are: (1) look up type, (2) compute origins union, (3) dispatch to catalog formula, (4) construct output.
- The channel capacity catalog is ~15 formulas, of which ~10 are textbook (mean, std, var, sum, count, groupby, Fano inequality).

**Consensus estimate of genuinely novel research code: ~8-12K LoC** out of 55-65K claimed. The rest is competent engineering that any skilled Rust developer could produce from the specification. However, "competent engineering" at this scale under correctness constraints is still a 6-8 month project.

**Panel divergence:** Auditor=7, Skeptic=4, Synthesizer=8. The Skeptic's 4 is too low — it conflates "novel" with "difficult" and dismisses engineering difficulty under soundness constraints. The Synthesizer's 8 is too high — the lattice, fixpoint, and most transfer functions are simpler than the terminology suggests.

### Pillar 3: Best-Paper Potential — **5/10** (self-assessed: 8, with P(best-paper)=12% claimed)

**Best-paper requires all three:**
1. An elegant proof of T1 (the fit-transform decomposition)
2. A "wow" empirical result (finding real leakage in published papers or deployed systems)
3. Bounds within 2-3× for common operations empirically demonstrated

**T1's elegance is debatable.** The proof sketch is 3 steps using textbook tools (sufficient statistics, DPI, chain rule). The Skeptic calls this "a textbook exercise." The Synthesizer calls it "why didn't anyone think of this before." After adversarial cross-critique, the consensus is ★★½: the *modeling insight* (recognizing fit_transform as composable sub-channels) is genuinely novel, but the *proof technique* uses only known machinery. This is a solid contribution, not a surprising one.

**The evaluation cannot currently demonstrate a "wow" result.** The Yang et al. corpus has only binary labels — TaintFlow's quantitative differentiator is invisible. Synthetic validation proves internal consistency. Without curated case studies showing quantitative output changing practitioner decisions, the empirical narrative is insufficient for best paper.

**Realistic probability re-calibration:**
- P(best-paper): **2-4%** (self-assessed 12% is 3-6× inflated)
- P(strong accept): **15-20%** (self-assessed 30%)
- P(accept, any venue): **50-60%** (self-assessed 65%)
- P(accept, top venue like OOPSLA/FSE): **35-45%**

**Venue assessment:** Strong accept plausible at OOPSLA or FSE. Accept at ICML/NeurIPS systems track with strong empirical results. Best paper possible but improbable — requires perfect execution.

**Panel divergence:** Auditor=6, Skeptic=3, Synthesizer=5. The Skeptic's 3 is too harsh — the fit-transform lemma IS novel at PL venues, even if not "surprising" by the Skeptic's elevated bar. The Auditor's 6 assumes too much goes right.

### Pillar 4: Laptop-CPU Feasibility — **9/10** (self-assessed: 7)

**Unanimous across all three experts.** This is the strongest dimension:
- Finite lattice height 68 → guaranteed termination without widening
- O(K × d × 68) fixpoint → sub-second for typical pipelines (K=50, d=200)
- Rust core → C-level performance with memory safety
- Zero GPU, zero neural components, zero human annotation
- Single pipeline execution for DAG extraction adds <20% overhead
- 30-second median analysis time target is realistic

The self-assessed 7 appears to conflate "laptop-CPU feasibility" with "overall project feasibility." Computationally, this is near-perfect.

---

## 3. Feasibility — **5/10** (blended Path A/B assessment)

| Path | LoC | Timeline | P(any pub) | P(top venue) | P(best-paper) | Schedule collapse risk |
|------|-----|----------|-----------|--------------|---------------|----------------------|
| **A (Full Vision)** | 75-90K | 18-24 months | 0.75 | 0.55 | 0.08 | 0.30 |
| **B (Core Tool) — RECOMMENDED** | 30-37K | 10-14 months | 0.85 | 0.65 | 0.05 | 0.10 |
| **C (Tool Paper)** | 10-15K | 5-7 months | 0.80 | 0.10 | 0.01 | 0.05 |

**Key feasibility concerns:**
1. **Zero execution exists.** No proof, no code, no experiment. The project is entirely specification. The "theory_complete" label is false.
2. **Math effort underestimated by ~25%.** Math Reviewer estimated 13-15 person-months; proposal claimed 10.5.
3. **Skill set rarity.** Rust + Python + information theory + abstract interpretation + ML domain expertise is unusually broad for a single researcher.
4. **T1 proof gap.** The Auditor identified a conditional independence issue in Step 3 of the proof sketch. Fixable but real — evidence that the flagship proof has not been carefully worked through.
5. **Six Math Reviewer revisions unaddressed**, including one critical (M9/M5 finite-sample gap for KSG estimator).

**Why 5, not 4:** Path B (Core Tool) is well-scoped at 30-37K LoC over 10-14 months. The specification quality is genuinely excellent — reducing proof-search risk. Two independent reviewers approved with revisions. Decision gates protect against sunk-cost fallacy.

**Why 5, not 6:** Zero current execution. The gap between excellent specification and working system is the entire project. The Auditor is right that "this is where projects die." The RobustScaler coverage overclaim and Step 3 proof gap suggest the specification itself has unidentified issues.

---

## 4. Fatal Flaw Analysis

| # | Potential Flaw | Severity | Fatal? | Assessment |
|---|----------------|----------|--------|------------|
| 1 | **Evaluation cannot validate quantitative differentiator on real data** | SERIOUS | **No — fixable** | The Skeptic's strongest attack. Yang et al.'s corpus has only binary labels; quantitative TaintFlow indistinguishable from binary detector. Fix: curate 5-10 case studies showing quantitative output changes decisions. Fix is required, not optional. |
| 2 | **Crown jewel (T1) may be weaker than claimed** | SERIOUS | No | Proof sketch has Step 3 gap. Sufficient-statistic assumption excludes RobustScaler, QuantileTransformer. Coverage is ~10-12 estimators, not 14 claimed. Provable for core set but novelty is ★★½, not ★★★. |
| 3 | **Zero proofs and zero code after "theory_complete"** | SERIOUS | No | Process anomaly, not content failure. The specification is substantive (42.8KB structured JSON). But "theory_complete" is false and creates false sense of progress. |
| 4 | **Conditional soundness perceived as weak** | MODERATE | No | Legitimate concept (cf. concolic testing), but conditions are not mechanically checkable. C1 (instrumentation faithfulness) has known violations (C-extension data flow). No runtime checker proposed. |
| 5 | **d² PCA bound vacuous for d > 50** | MODERATE | No | Acknowledged. PCA is common; vacuous bounds limit utility for ~20% of real pipelines. Empirical refinement (Phase 3) is fallback but has its own M9/M5 gap. |
| 6 | **500-line script captures 65-70% of detection value** | MODERATE | No | The formal framework provides genuine value beyond the script (compositional reasoning, known-unknowns, extensibility). But the evaluation must demonstrate this advantage. |

**No genuinely fatal flaw identified.** The evaluation gap (flaw #1) is the most dangerous — if unfixed, it will cause rejection at top venues with near certainty. But it is fixable with curated case studies.

---

## 5. What Was Done Well

Credit where due — the specification work is excellent:

1. **The approach.json is one of the most detailed theory specifications** at this stage. Every theorem has assumptions, proof sketches, effort estimates, and risk assessments. The channel capacity catalog has worked examples with numerical predictions.

2. **The graduated-precision design** (tight bound → loose bound → ∞) is intellectually honest and pragmatic. Even if only 15 operations get tight bounds, the tool degrades gracefully.

3. **The hybrid dynamic-static architecture** eliminates the #1 risk from the original proposals (Python static analysis frontend with P_fail=0.55). Dynamic DAG extraction via sys.settrace + monkey-patching is the right design.

4. **Prior reviewer feedback was incorporated.** Lattice height corrected, conditional soundness qualified, fit-transform lemma extracted as named result, ornamental math (Galois insertions, widening, type system framing) dropped.

5. **Honest self-assessment of limitations.** The red-team section correctly identifies the three highest risks. KNNImputer/IterativeImputer exclusion is acknowledged. PCA vacuity is acknowledged.

6. **The salvage value is high.** Even total failure of the soundness theorem leaves a publishable tool paper. Even total failure of tightness claims leaves a publishable lattice/decomposition theory paper. Multiple independent publication paths exist.

---

## 6. Consensus Scores

| Dimension | Self-Assessed | Auditor | Skeptic | Synthesizer | **Consensus** |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Extreme Value (V) | 8 | 6 | 6 | 8 | **6** |
| Software Difficulty (D) | 7 | 7 | 4 | 8 | **6** |
| Best-Paper Potential (BP) | 8 | 6 | 3 | 5 | **5** |
| Laptop-CPU Feasibility (L) | 7 | 9 | 9 | 9 | **9** |
| Overall Feasibility (F) | 7 | 4 | 6 | 6 | **5** |

**Composite: V6 / D6 / BP5 / L9 / F5 = 5.90**

Self-assessed composite: 7.50. **Inflation: 21.3%.**

Primary sources of inflation:
- **Value (+2):** Conflating research novelty with practitioner adoption readiness. Most ML practitioners would prefer a simpler tool.
- **Best-Paper (+3):** Assuming perfect execution without discounting for proof risks, vacuous PCA bounds, and evaluation gap. P(best-paper) is 2-4%, not 12%.
- **Feasibility (+2):** Not accounting for zero existing execution, proof gaps, and math effort underestimation.

---

## 7. Verdict

### **CONDITIONAL CONTINUE** — Confidence: 70%

All three panel experts recommend CONTINUE (unanimously). After adversarial cross-critique, the lead concurs with conditions.

### Rationale for CONTINUE

1. **The core insight is genuinely novel.** No existing tool applies QIF channel capacity models to sklearn/pandas operations. The intersection of QIF × abstract interpretation × ML pipeline semantics is confirmed unexplored by both the Math Reviewer and Prior Art Reviewer.

2. **No genuinely fatal flaw exists.** The evaluation gap is serious but fixable. The T1 proof gap is real but likely patchable. Zero execution is concerning but normal at this stage.

3. **The problem is real and unsolved.** 15-25% of Kaggle kernels have leakage. Practitioners have no quantitative diagnostic. The demand side is strong.

4. **Path B (Core Tool) is well-scoped.** 30-37K LoC, 10-14 months, P(any publication)=0.85. Clear decision gates at months 4, 8, 12 protect against sunk-cost fallacy.

5. **High salvage value floor.** Even if the full vision fails, multiple independently publishable components survive (fit-transform lemma, capacity catalog, DAG extractor).

### Rationale for 30% ABANDON Probability

1. **Zero execution after theory stage.** The gap between specification and working system is where projects die. No proof has been completed. No code has been written.

2. **Evaluation design structural weakness.** If unfixed, guarantees rejection at top venues. The core differentiator (quantitative bits) cannot be validated on real data with current plan.

3. **Skill set breadth.** Rust + Python + information theory + abstract interpretation + ML domain is unusually broad for a single researcher.

4. **Proof sketch quality concerns.** T1 Step 3 has a conditional independence gap. RobustScaler coverage is overclaimed. If the flagship proof has errors in the sketch, the other proofs may have undiscovered issues.

### Required Conditions (Kill Gates)

| Gate | Deadline | Criterion | Fail Action |
|------|----------|-----------|-------------|
| **G1: Proof of T1** | +30 days | T1 proved rigorously for ≥5 estimators (StandardScaler, MinMaxScaler, PCA, SimpleImputer, OHE) | If T1 fails for core set → pivot to Path C (tool paper) |
| **G2: Working Prototype** | +60 days | Instrument a sklearn Pipeline → extract PI-DAG → produce one leakage bound | If prototype fails → **ABANDON** (architecture unvalidated) |
| **G3: Evaluation Fix** | +90 days | 5-10 curated real-world case studies demonstrating quantitative value over binary detection | If evaluation gap persists → downscope to ASE tool track |
| **G4: Math Revisions** | +90 days | All 6 Math Reviewer revisions addressed (esp. M9/M5 finite-sample gap) | If critical gaps remain → re-evaluate theoretical contribution |
| **G5: Path Commitment** | +120 days | Based on G1-G4, commit to Path A (full), B (core), or C (tool) | — |

### Binding Amendments

1. **Rename "theory_complete" to "theory_specified"** in State.json immediately.
2. **Deflate novelty claims:** T1 = ★★½ (not ★★★). Catalog = ★★☆ (not ★★★). Conditional soundness = ★★☆.
3. **Fix T1 proof sketch:** Address Step 3 conditional independence gap. Remove RobustScaler from T1 coverage (or provide separate argument).
4. **Correct estimator coverage:** ~10-12 estimators under T1, not 14.
5. **Add evaluation case studies:** 5-10 real pipelines where quantitative output demonstrably changes decisions vs. binary detection.
6. **Build the 500-line baseline script** as a comparison tool — it strengthens the evaluation by showing what TaintFlow adds beyond simple heuristics.
7. **Revise difficulty estimate:** 13-15 person-months for math (not 10.5). 20-24 person-months total for Path B.
8. **Correct P(best-paper):** 2-4% (not 12%). P(accept, any venue): 50-60% (not 65%).

### Recommended Path

**Path B (Core Tool):** Fit-transform lemma for ~10 estimators + simplified lattice + 25 well-chosen transfer functions + conditional soundness + evaluation on 200 synthetic + 100 Kaggle pipelines + 5-10 curated case studies. Drop sensitivity types (M4), reduced product (M5), min-cut attribution (M6) to follow-up work. Target FSE/OOPSLA.

**Start Path B. Upgrade to Path A if math is clean at month 8. Downgrade to Path C if T1 fails.**

---

## 8. Summary

TaintFlow is a competent, well-specified proposal at a genuinely novel intersection of three mature fields. The problem is real, the architecture is sound, and the specification quality is high. However, the self-assessment is inflated by 21%, the crown jewel is a solid modeling contribution rather than a surprising theorem, the evaluation cannot validate the core differentiator on real data, and zero execution exists after the theory stage.

The project has genuine upside (novel intersection, real practitioner pain, publishable at top venues) and genuine downside (enormous remaining work, proof risks, evaluation gap). The risk-adjusted recommendation is CONTINUE via Path B with kill gates, not because the proposal is exceptional, but because the problem is real, the approach is sound, and the salvage value is high.

**Composite: V6 / D6 / BP5 / L9 / F5 = 5.90/10.**
**Verdict: CONDITIONAL CONTINUE (70% confidence, 3-expert consensus).**
**Path: B (Core Tool), 30-37K LoC, 10-14 months, targeting FSE/OOPSLA.**

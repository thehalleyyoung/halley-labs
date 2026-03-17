# Community Expert Verification: TaintFlow (ml-pipeline-leakage-auditor)

**Evaluator:** Area-041 Community Expert (Machine Learning & AI Systems)
**Stage:** Verification (post-theory)
**Date:** 2026-03-08
**Method:** 3-expert adversarial team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with cross-critique rounds and independent verifier signoff

---

## Executive Summary

TaintFlow proposes the first sound, quantitative information-flow auditor for ML pipelines, measuring train-test leakage in bits via channel capacity bounds over a dynamically-extracted dataflow DAG. The idea sits at a genuinely unexplored intersection of quantitative information flow (QIF), abstract interpretation, and ML pipeline semantics. The ideation is exceptionally thorough — the best-documented proposal in this evaluation batch. However, **theory_bytes = 0 and impl_loc = 0**: the crown jewel theorem (fit-transform channel decomposition) has never been attempted, no code exists, and every hard claim is speculative. The team unanimously recommends **CONDITIONAL CONTINUE** with aggressive scope reduction, a hard 6-week proof gate, and venue redirection from NeurIPS to OOPSLA.

---

## Team Process

Three independent experts evaluated the proposal in parallel, then challenged each other in an adversarial cross-critique round. An independent verifier provided final signoff.

| Expert | Role | Verdict | Composite |
|--------|------|---------|-----------|
| Independent Auditor | Evidence-based scoring | Proceed cautiously | V7/D7/P6/L9/F5 = 6.25 |
| Fail-Fast Skeptic | Reject under-supported claims | Conditional continue (radical rescope) | V5/D4/P3/L9/F4 = 5.0 |
| Scavenging Synthesizer | Salvage value assessment | Conditional continue (proof gate) | V7/D7/P6/L9/F7 = 7.2 |
| **Cross-critique consensus** | — | **CONDITIONAL CONTINUE** | **V7/D6/BP5/L9/F5.5 = 6.1** |
| Independent Verifier | Final signoff | **APPROVE WITH CHANGES** | — |

---

## Pillar-by-Pillar Assessment

### 1. Extreme Value — 7/10

**The capability gap is real.** No existing tool provides per-feature, per-stage quantitative leakage attribution in bits. LeakageDetector (Yang et al., ASE 2022) is binary and syntactic (3 patterns). LeakGuard is empirical and model-dependent. TaintFlow's output — "Feature `age_normalized` has 3.7 bits of test-set contamination from `StandardScaler.fit_transform()` at line 47" — is a genuinely novel diagnostic that addresses the #1 practitioner question on ML forums: "Why do my offline and online metrics disagree?"

**But practitioners mostly want binary detection, not bits.** The formal soundness guarantee (conditional on execution path) is invisible to users — they see severity rankings, not lattice fixpoints. The quantitative bits output enables triage (which 3 of 20 contaminated features carry 95% of leakage), but most practitioners will consume this through coarse severity labels (negligible/warning/critical), not raw bit-counts. Estimated fraction of users who need bits over binary: <10%.

**Execution requirement limits reach.** TaintFlow requires running the pipeline with data, making it a self-debugging tool, not a third-party auditing tool. The regulatory compliance and publication-certification use cases are aspirational.

**Challenge test — "500-line script":** A Python script using `ast.parse()` + 10 channel capacity formulas captures ~50% of detection value but <20% of quantification value. TaintFlow's unique contribution (quantitative attribution with formal bounds) is not achievable at 500 lines. The tool is justified — but not at 55K LoC scale.

### 2. Genuine Software Difficulty — 6/10

**Two genuinely hard subproblems:**
- **Fit-transform channel decomposition (★★★):** Factoring the feedback loop where `fit` computes statistics from input and `transform` applies them back to the same input is genuinely novel. No prior QIF work handles this pattern. Proof feasibility: 2-4 months with ~35% probability of weakening.
- **Tight capacity catalog for non-trivial operations:** Rank statistics (median, quantiles) involve order-statistic distributions without closed forms. PCA bounds require Wishart distribution analysis. 3-6 months for ~15-20 tight bounds.

**The rest is well-understood engineering:**
- Partition-taint lattice: textbook product lattice (Cousot & Cousot 1979)
- sys.settrace + monkey-patching: well-trodden (coverage.py, pytest-cov)
- Worklist fixpoint: undergraduate-level with lattice height 68
- Roaring bitmap provenance: standard database technique

**The Rust engine is over-engineering.** 690K lattice operations complete in microseconds even in Python. The 55-65K LoC Rust+Python scope is 3-5× larger than the value warrants. The necessary work is ~15-25K LoC Python.

### 3. Best-Paper Potential — 5/10

**Novel intersection:** QIF × abstract interpretation × ML pipeline semantics is genuinely unexplored. The fit-transform decomposition lemma, if proved elegantly, is a real "aha" result.

**But conditional soundness faces venue risk.** At ML venues (NeurIPS/ICML), reviewers will write: "This is dynamic analysis with extra steps. What does it buy over running the pipeline twice?" At PL venues (OOPSLA), conditional soundness is standard for dynamic-analysis-assisted tools — reviewers will appreciate the nuance.

**Zero execution devastates the best-paper assessment.** theory_bytes=0 means no proofs, no code, no experiments, no paper text. The internal reviews themselves rate best paper as "conditional." P(best paper) ≈ 4-6%.

**The cross-critique resolved:** Conditional soundness is a genuine contribution (not a tautology), but must target OOPSLA where it will be evaluated on its merits.

### 4. Laptop-CPU Feasibility & No-Humans — 9/10

**Unanimous across all experts.** Lattice height 68 with constant-time operations. Fixpoint in milliseconds for typical pipelines. Rust core with Rayon parallelism (but Python suffices). No neural components — fully deterministic and reproducible. Evaluation fully automated via synthetic benchmark suite + Kaggle corpus with existing labels. No GPU, no human annotation, no human studies required. Target ≤30s median per pipeline.

### 5. Feasibility — 5.5/10

**Evidence for feasibility:**
- Lowest P_fail (~0.30) of all three original approaches
- Well-scoped MVP at ~30K LoC with ~15 tight bounds
- Graceful degradation: full paper → profiler → tool demo → catalog paper
- No research-blocking subproblems — worst case is scope reduction, not failure

**Evidence against feasibility:**
- **theory_bytes = 0, impl_loc = 0** — zero execution progress despite completing ideation
- The theory stage failed entirely (all agents errored with 503s)
- Fit-transform proof sketch Step 3 has "explaining away" concern (conditioning on θ can *increase* MI)
- Math timeline underestimated by ~25% per prior math verification
- 8-month timeline is 2-3× optimistic; realistic: 14-20 months
- P(at least one significant compromise) ≈ 0.70

**The verifier's note:** Feasibility should measure the *stated contribution's* tractability, not project survivability via fallbacks. Strong fallback paths justify investment but don't make the hard theorems easier.

---

## Fatal Flaw Analysis

### Flaw 1: "Sound but Useless" Bounds (Severity: HIGH)

PCA bound = d²×C_cov: for d=100, this is 500-5000× loose. GroupBy with 50K groups gives ~360 bits when true leakage is ~50 bits. **Only ~15-20 of 80+ operations get actionably tight bounds.** For the remaining operations, bounds are correct in ordering but not in magnitude. The "quantitative leakage in bits" headline applies fully to only ~20-30% of real pipeline operations.

**Mitigation:** The graduated-precision narrative (tight for common ops, loose-but-sound for rest, ∞ for unknown) is honest. Severity ordering (Spearman ρ ≥ 0.80) is the more defensible claim.

### Flaw 2: Crown Jewel Theorem Unproved (Severity: HIGH)

The fit-transform decomposition has a plausible proof sketch but a known gap (Step 3 "explaining away"). The sufficient-statistic assumption cleanly covers only StandardScaler, PCA, and SimpleImputer — 3 non-trivial operations. RobustScaler (median, IQR are NOT sufficient statistics), QuantileTransformer, and most non-linear estimators are excluded.

**Probability of proving as stated:** 55-65%. **As restricted to linear estimators:** 85-90%.

### Flaw 3: Zero Execution Progress (Severity: HIGH for project, N/A for idea)

theory_bytes=0, impl_loc=0. Every claim — proofs, bounds, tightness, performance, coverage — is aspirational. The theory stage itself produced only a structured plan (approach.json), not results.

### No Fatal Flaws in the Core Idea

The team finds no flaw that makes TaintFlow fundamentally impossible. The risks are all manageable with scope reduction and honest framing. The question is not "can this be built?" but "can it be built well enough, fast enough, to the claimed spec?"

---

## Salvage Analysis

### If Full Project Continues (MVP Path)
- **Scope:** ~30K LoC, 20-25 operations, 10 tight bounds, restricted fit-transform lemma
- **Timeline:** 5 months to prototype, 14-20 months to paper
- **Venue:** OOPSLA/FSE primary, NeurIPS systems track backup
- **P(accept at top venue):** ~0.45

### If Fit-Transform Decomposition Fails
- **Fallback A:** Restricted lemma for StandardScaler+PCA only → workshop paper (P: 0.70)
- **Fallback B:** Leakage profiler without soundness claims → MLSys/AISTATS (P: 0.50)
- **Fallback C:** Empirical MI profiler via KSG → tool demo at ASE/ICSE (P: 0.85)

### Standalone Components with Publication Potential
1. **Channel capacity catalog** — "How Much Does Your Scaler Leak?" — standalone workshop/short paper (P: 0.75)
2. **PI-DAG extraction infrastructure** — reusable for any ML pipeline analysis (tool demo, P: 0.70)

### Expected Value
E[V] ≈ 55-58/100 (corrected from Synthesizer's optimistic 75). The project has positive expected value with an early exit ramp at week 6.

---

## Scores

| Dimension | Score | Justification |
|---|---|---|
| **1. Extreme Value** | **7** | Real capability gap (quantitative leakage attribution in bits). No existing tool provides this. Contingent on bound tightness for ~15-20 operations. Practitioners mostly consume via severity labels, not raw bits. |
| **2. Genuine Software Difficulty** | **6** | Fit-transform decomposition (★★★) and tight catalog (~15 entries) are genuinely hard. Lattice/fixpoint/instrumentation are standard. Rust engine is over-engineering. True necessary difficulty is 6, not 7. |
| **3. Best-Paper Potential** | **5** | Novel intersection, genuine "aha" theorem if proved. But conditional soundness faces venue risk at ML venues. OOPSLA is correct target. P(best paper) ≈ 4-6%. Zero execution means paper doesn't exist yet. |
| **4. Laptop-CPU & No-Humans** | **9** | Inherently cheap analysis (finite lattice, millisecond fixpoint). No GPU, no human annotation, no human studies. Fully automated evaluation. Strongest dimension. |
| **5. Feasibility** | **5.5** | theory_bytes=0, impl_loc=0, crown jewel unproved with known gap. Strong fallback chain. P_fail ≈ 0.30 for some publishable outcome; higher for the stated contribution. Timeline 14-20 months, not 8. |

**Composite: (7 + 6 + 5 + 9 + 5.5) / 5 = 6.5** (unweighted)
**Composite: (2×7 + 6 + 5 + 9 + 2×5.5) / 8 = 5.8** (V+F double-weighted)
**Reported composite: 6.1** (balanced)

---

## VERDICT: **CONDITIONAL CONTINUE**

**Confidence: 70% (3-expert consensus: Auditor + Synthesizer CONTINUE, Skeptic CONDITIONAL CONTINUE; Verifier APPROVE WITH CHANGES)**

### Binding Conditions

**BC-1: 6-Week Proof Gate (HARD KILL GATE)**
Prove the restricted fit-transform channel decomposition lemma for StandardScaler and PCA within 6 weeks. This is the go/no-go decision point.
- **Hard fail** (StandardScaler proof fails): ABANDON soundness claims → pivot to F1 leakage profiler (MLSys/AISTATS)
- **Soft fail** (StandardScaler succeeds, PCA fails): Reduce scope to linear estimators only; paper still viable at OOPSLA

**BC-2: Scope Reduction (MANDATORY)**
Target 15-25K LoC Python, not 55-65K LoC Rust+Python. The Rust engine is over-engineering a problem where Python suffices. Cover 20-25 core operations (not 130). This is not optional — the full scope is 3-5× the value delivered.

**BC-3: Venue Commitment by Month 3**
Choose OOPSLA (formal contribution focus) vs NeurIPS systems track (empirical impact focus) by month 3. Writing for two audiences wastes effort. Recommendation: OOPSLA primary.

**BC-4: Month 6 Implementation Checkpoint**
Working end-to-end prototype on ≥3 synthetic pipelines with bound tightness measured. Catches integration issues before full evaluation investment.

**BC-5: Oracle Methodology Pre-Commitment**
Specify the evaluation oracle methodology (how ground-truth leakage is measured) before the proof gate, since noisy oracles undermine the tightness evaluation.

### Kill Conditions (ABANDON immediately if any occur)
1. T1 (fit-transform) proved to fail even for StandardScaler alone
2. Empirical evaluation shows bounds >20× loose for >70% of Kaggle pipelines
3. A competitor publishes quantitative leakage bounds before this project ships
4. After 2 months, no proofs completed for any theorem

### Recommended Timeline
- **Weeks 1-6:** Proof gate — restricted fit-transform lemma
- **Months 2-4:** Core implementation (DAG extraction + lattice + 15 transfer functions)
- **Months 4-6:** Implementation checkpoint — end-to-end on synthetic pipelines
- **Months 6-10:** Full evaluation on synthetic + real Kaggle corpus
- **Months 10-14:** Paper writing + revision for OOPSLA
- **Total: 14-20 months** (not the proposed 8)

---

## Cross-Critique Key Findings

1. **Conditional soundness IS a real contribution** — not a tautology (Skeptic overruled), but must be presented with explicit assumption-listing and targeted at PL venues where reviewers appreciate the nuance.

2. **The 500-line script captures ~50% of detection value** but <20% of quantification value. The full tool is justified, but the scope delta (55K → 15-25K LoC) is the right correction.

3. **Bounds are actionable for ~15-20 operations only.** The "quantitative leakage in bits" headline applies fully to simple aggregates. PCA/cross-column operations get vacuously loose bounds. The severity-ordering claim (correct relative ranking) is more defensible than absolute bit-counts.

4. **The channel capacity catalog has standalone publication potential** as "How Much Does Your Scaler Leak?" — a 2-month workshop paper regardless of the full project's fate.

5. **E[V] ≈ 55-58** — positive expected value with early exit ramp at week 6. The project degrades gracefully: full paper → restricted paper → profiler → tool demo → catalog paper.

---

*Evaluation conducted via 3-expert adversarial team with independent verifier signoff.*
*Auditor: PROCEED | Skeptic: CONDITIONAL CONTINUE | Synthesizer: CONDITIONAL CONTINUE | Verifier: APPROVE WITH CHANGES*
*Consensus: CONDITIONAL CONTINUE at V7/D6/BP5/L9/F5.5 = 6.1 composite*

# Depth Check: Spectral Decomposition Oracle

**Slug:** `spectral-decomposition-oracle`
**Stage:** Verification Gate
**Date:** 2026-03-08
**Panel:** 3-expert team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer)
**Disposition:** UNANIMOUS APPROVAL of assessment after cross-critique and signoff

---

## Executive Summary

**Verdict: CONDITIONAL CONTINUE — RADICAL RESTRUCTURING REQUIRED**

The original triple-threat proposal (headline theorem T2 + 155K LoC system + MIPLIB benchmark) is **REJECTED**. T2's constant is vacuous on ≥60% of MIPLIB, the LoC is inflated ≥3×, and the evaluation protocol has circularity flaws. However, a **restructured project** — "Spectral Features for MIP Decomposition Selection: A Computational Study with the First Complete MIPLIB 2017 Decomposition Census" — is viable as an INFORMS JoC submission. The census is the primary contribution; spectral features are validated empirically; T2 is demoted to 2-page motivational analysis.

**Composite Score: 4.5/10** (V5/D4/BP3/L6)

---

## Pillar Scores

### 1. EXTREME AND OBVIOUS VALUE — 5/10

**What works:** Decomposition method selection is a real pain point for OR teams working with structured MIPs. The no-go concept (predicting when decomposition is futile) has genuine practitioner value — teams currently waste weeks attempting decompositions that cannot help. The MIPLIB census is a first-of-kind community contribution: no one has systematically answered "which of the 1,065 MIPLIB instances are amenable to Benders, DW, or Lagrangian — and by how much?" The spectral framing enables continuous, geometry-aware features that degrade gracefully, unlike GCG's combinatorial partitioning which either finds blocks or doesn't.

**What's missing for ≥7:** The addressable audience is narrow — perhaps a few hundred research groups and advanced industrial OR teams. Most practitioners submit directly to Gurobi/CPLEX; CPLEX 22.1 already has automatic Benders. The "reformulation selection" framing is genuinely novel (distinct from algorithm selection), but the proposal buries this insight rather than leading with it. LLMs don't change the calculus here (algebraic structure, not pattern-matching), but they also don't create new urgency.

**Required fix:** Reframe the entire project around the census as the primary artifact and the spectral oracle as the analytical lens. Lead with the reformulation-selection framing as the opening sentence.

### 2. GENUINE DIFFICULTY AS A SOFTWARE ARTIFACT — 4/10

**What works:** Making the spectral analysis pipeline robust across 1,065 diverse MIPLIB instances (highly degenerate, numerically ill-conditioned, 50 to 10M nonzeros) without per-instance tuning is real engineering. The Davis-Kahan adaptation to non-square, non-symmetric constraint-matrix-derived hypergraph Laplacians requires genuine mathematical work. The census infrastructure (parsing, result DB, statistical analysis, reproducibility harness) is non-trivial.

**What's missing for ≥7:** The 155K LoC estimate is inflated ≥3×. Line-by-line analysis:

| Subsystem | Claimed | Honest novel estimate | Rationale |
|-----------|---------|----------------------|-----------|
| Spectral engine | 25K | 3-5K | ARPACK/Spectra does eigensolves. Hypergraph construction ~500 LoC, spectral clustering ~200 LoC, feature extraction ~300 LoC. |
| Benders | 25K | 1-2K glue | SCIP has native Benders since v6.0 |
| DW/CG | 30K | 2-3K glue | GCG exists and does exactly this |
| Lagrangian | 20K | 3-5K | Subgradient ~50 LoC, bundle method 2-3K, primal recovery 1K |
| Strategy oracle | 20K | 1-2K | Random forest on ~20 features with sklearn |
| Census infra | 15K | 5-8K | Fair — real engineering |
| Shared infra | 20K | 5-8K | Solver abstraction, sparse matrix |
| **Total** | **155K** | **~25-35K novel** | |

The novel intellectual contribution is ~25-35K lines. Reimplementing Benders, DW/CG, and Lagrangian from scratch is unnecessary engineering risk — GCG alone took a research group over a decade for DW-only.

**Required fix:** Use SCIP Benders API, GCG for DW, existing Lagrangian library (e.g., ConicBundle). Scope to ~50-70K total LoC, ~25-35K novel.

### 3. BEST-PAPER POTENTIAL — 3/10

**What works:** The *qualitative* insight that δ²/γ² predicts decomposition quality is elegant. The bridge between spectral graph theory and decomposition theory is genuinely novel — no prior work connects spectral gaps to dual bound degradation through a quantitative (even if loose) result. Lemma L3 (partition-to-bound bridge) has standalone theoretical value. The MIPLIB census would be a genuine community service.

**What's missing for ≥7:** T2's constant C = O(k·κ⁴·‖c‖∞) is **vacuous on the majority of MIPLIB**. The arithmetic is terminal: a modest big-M of 10⁶ gives κ⁴ = 10²⁴; with k ≈ 10 and ‖c‖∞ ≈ 10⁶, the bound says "degradation ≤ 10³¹ · δ²/γ²" — i.e., nothing. The "Spearman ρ ≥ 0.6" empirical escape means if the theorem doesn't predict, the empirical correlation does, reducing T2 to a motivating heuristic. A Mathematical Programming or IPCO reviewer will note that Davis-Kahan + Hoffman is a standard composition; the novelty is in the *application*, not the proof technique. No best-paper committee will award a paper whose central theorem is admittedly vacuous on ≥60% of benchmark instances.

**Realistic venue assessment:**
- MPC / Math Programming: **Non-starter** unless C tightened to O(κ²) or better
- IPCO: **Non-starter** — needs tight approximation ratios
- **INFORMS JoC: Best fit** — rewards reproducible computational studies with open data
- CPAIOR: Good backup — integration of spectral/ML/OR techniques
- Operations Research Letters: For L3 as standalone result

**Required fix:** Target INFORMS JoC. Make census the primary contribution. Present T2 as 2-page motivational analysis (structural scaling law), not headline theorem. Allow MPC as conditional secondary target only if empirical results are exceptionally strong.

### 4. LAPTOP CPU + NO HUMANS — 6/10

**What works:** Sparse eigendecomposition (k ≤ 20 eigenvectors via ARPACK on matrices with up to 10⁷ nonzeros) completes in under 30 seconds on a modern laptop CPU — this is well-established (Lehoucq et al. 1998). The decomposition subproblems are smaller than the original by construction. Individual instance analysis targets under 10 minutes for instances with <10⁵ nonzeros (80% of MIPLIB). Memory for typical instances (~50 MB for 10⁶ nonzeros) is within laptop constraints. Everything is fully automated — no human annotation or judgment required.

**What's missing for ≥9:**
- **Census iteration time:** 45 days single-core, 12 days on 4 cores for full MIPLIB. This is technically feasible but practically blocks development iteration — you can't debug at a 12-day cycle time.
- **Label generation:** Training the strategy oracle requires running all decomposition methods on all instances. With external solvers (SCIP/GCG), this is ~3 × 1,065 hours = 133 CPU-days serial.
- **Hard instances:** 50+ MIPLIB instances remain open. Decomposition subproblems for these will hit the 1-hour timeout, producing incomplete data.
- **Memory on largest instances:** Instances like `neos-5052403-cygnet` (>30M nonzeros) approach 2-4 GB for eigenvector storage — feasible on modern laptops but tight.

**Required fix:** Implement tiered census: Tier 1 (100 curated instances, ~17 hours) for daily development, Tier 2 (500 stratified, ~4 days on 4 cores) for weekly validation, Tier 3 (full 1,065) for release. This enables rapid iteration without sacrificing final coverage.

### 5. FATAL FLAWS

| # | Flaw | Severity | Fixable? | Fix |
|---|------|----------|----------|-----|
| **F1** | **T2 constant vacuous on ≥60% MIPLIB** — C = O(k·κ⁴·‖c‖∞) means the bound says nothing on big-M instances. The paper's headline contribution collapses. | CRITICAL | PARTIALLY | Demote T2 to motivating analysis. Lead with empirical correlation (Spearman ρ). Present δ²/γ² as structural scaling law, not rigorous bound. |
| **F2** | **155K LoC inflated ≥3×** — ~75K is reimplementation of algorithms available in SCIP/GCG. Novel content is ~25-35K. Creates unnecessary engineering risk. | SERIOUS | YES | Use SCIP Benders, GCG DW, existing Lagrangian library. Scope to ~50-70K total. |
| **F3** | **Evaluation circularity** — method selection accuracy validated against own implementations. A buggy Benders makes other methods look better. Self-referential. | SERIOUS | YES | Use GCG + SCIP-native Benders as independent baselines. Cross-validate. Report results against *independent* implementations. |
| **F4** | **No ground truth for structure detection F1** — "labeled MIPLIB subset" doesn't exist. If self-labeled, F1 is circular. | SERIOUS | PARTIALLY | Use GCG's detections as one ground truth. Supplement with synthetic instances with known planted structure. Acknowledge limitations. |
| **F5** | **No-go "certificate" inherits T2 vacuousness** — threshold γ_min depends on C, which is 10²⁴. Certificate essentially never fires meaningfully on real instances. | MEDIUM | YES | Rename to "futility predictor." Calibrate threshold empirically on held-out data, not from T2's constant. |
| **F6** | **Only ~15-25% of MIPLIB has exploitable block structure** (Bergner et al. 2015) — limits data for method-selection evaluation. | MEDIUM | PARTIALLY | Supplement with structured instances from literature (vehicle routing, supply chain, stochastic programming). |
| **F7** | **Census iteration time (12+ days) blocks development** — impractical for debugging and ablation without cluster access. | MEDIUM | YES | Tiered census: 100/500/1065 instances at daily/weekly/release cadence. |

---

## Required Amendments

All seven amendments are **binding conditions** for continuation:

1. **REFRAME AS COMPUTATIONAL STUDY.** The MIPLIB decomposition census is the primary contribution. T2 is a 2-page motivational analysis explaining the structural intuition behind spectral features. The paper title should reflect this: "Spectral Features for MIP Decomposition Selection: A Computational Study with the First Complete MIPLIB 2017 Decomposition Census."

2. **CUT LoC TO ~50-70K TOTAL (~25-35K NOVEL).** Use SCIP's Benders API, GCG for DW, and an existing Lagrangian library. Focus engineering on: spectral engine, census infrastructure, oracle, solver abstraction glue. Do not reimplement textbook decomposition algorithms.

3. **FIX EVALUATION PROTOCOL.** Use GCG and SCIP-native Benders as independent baselines for method selection validation. Use GCG detections + synthetic planted-structure instances for structure detection F1. Report per-structure-type accuracy breakdowns, not just aggregates.

4. **RENAME NO-GO "CERTIFICATE" TO "FUTILITY PREDICTOR."** Calibrate the threshold empirically on held-out data using cross-validation, not from T2's theoretical constant. Evaluate precision on the held-out set.

5. **IMPLEMENT TIERED CENSUS.** Tier 1: 100 curated instances (daily development, ~17 hours). Tier 2: 500 stratified instances (weekly validation, ~4 days on 4 cores). Tier 3: full 1,065 (release, ~12 days on 4 cores).

6. **ADD SPECTRAL FEATURES FOR ALGORITHM SELECTION ABLATION.** Run AutoFolio/SMAC with and without spectral features on MIPLIB to validate the feature family independently. Report per-structure-type results. Require spectral features to outperform syntactic-only features by ≥5 percentage points or improve Spearman ρ by ≥0.1, with statistical significance (p < 0.05, paired permutation test).

7. **TARGET INFORMS JoC (primary), MPC (conditional secondary).** Drop MPC as primary aspiration. Allow MPC upgrade only if Gate G3 passes with strong effect sizes (spectral > syntactic + 10pp). CPAIOR as backup.

---

## Kill Gates

| Gate | Week | Condition | Consequence |
|------|------|-----------|-------------|
| **G1** | 2 | Spearman ρ(δ²/γ², bound degradation) ≥ 0.4 on 50-instance pilot | **ABANDON** — spectral premise is dead |
| **G2** | 4 | Spectral engine complete; SCIP/GCG wrappers operational on ≥80% of Tier 1; structure detection F1 ≥ 0.7 on synthetic instances | **ABANDON** |
| **G3** | 8 | Method prediction accuracy ≥ 60% on Tier 1 using GCG/SCIP baselines; spectral > syntactic + 5pp OR Δρ ≥ 0.1 (p < 0.05) | **PIVOT** to feature-engineering-only workshop paper |
| **G4** | 14 | Tier 2 census (500 instances) complete; futility predictor precision ≥ 80% on held-out set | Reduce scope to Tier 1 only |
| **G5** | 18 | Paper draft with full Tier 3 census submitted to JoC | — |

---

## Probability Estimates

| Outcome | Probability |
|---------|-------------|
| P(publication at INFORMS JoC) | **45-55%** |
| P(best-paper at any venue) | **<5%** |
| P(any publication including workshops/CPAIOR) | **65-75%** |
| P(abandon before publication) | **30-40%** |

---

## Panel Deliberation Record

### Phase 1: Independent Proposals

| Expert | Initial Scores (V/D/BP/L) | Initial Recommendation |
|--------|---------------------------|----------------------|
| Independent Auditor | 5/4/5/4 | CONDITIONAL CONTINUE |
| Fail-Fast Skeptic | 4/4/2/6 | REJECT |
| Scavenging Synthesizer | 5/4/3/6 | REVISE AND PROCEED |

### Phase 2: Adversarial Cross-Critiques

**Key movements:**
- Auditor moved best-paper 5→3, accepting Skeptic's argument that vacuous T2 kills the theory narrative. Moved laptop 4→7 accepting tiered census fix.
- Skeptic moved value 4→5 and difficulty 4→4 (no change), conceding census has real scholarly value and Davis-Kahan adaptation is non-trivial.
- Synthesizer moved value 7→5 (conceded audience is narrow), best-paper 5→3 (conceded MPC non-starter), conceded no-go "certificate" is really a classifier, conceded 155K LoC indefensible.

### Phase 2: Post-Critique Revised Scores

| Expert | Revised Scores (V/D/BP/L) | Revised Recommendation |
|--------|---------------------------|----------------------|
| Independent Auditor | 6/5/3/7 | REVISE AND PROCEED |
| Fail-Fast Skeptic | 5/4/3/6 | CONDITIONAL CONTINUE (shifted from REJECT) |
| Scavenging Synthesizer | 5/4/3/6 | REVISE AND PROCEED (descoped) |

### Phase 3: Consensus (Medians)

| Pillar | Auditor | Skeptic | Synthesizer | **Consensus (Median)** |
|--------|---------|---------|-------------|----------------------|
| Value | 6 | 5 | 5 | **5** |
| Difficulty | 5 | 4 | 4 | **4** |
| Best-Paper | 3 | 3 | 3 | **3** |
| Laptop CPU | 7 | 6 | 6 | **6** |

### Phase 4: Verification Signoff

All three experts **APPROVED** unanimously with minor refinements:
- **Auditor:** Report per-structure-type accuracy in ablation (folded into Amendment 6)
- **Skeptic:** Add minimum effect size to G3 (≥5pp or Δρ≥0.1, p<0.05) (folded into Gate G3)
- **Synthesizer:** Allow MPC as conditional secondary venue if G3 passes strongly (folded into Amendment 7)

---

## Dissent Record

The Fail-Fast Skeptic's initial recommendation was **REJECT**. After cross-critique, the Skeptic moved to **CONDITIONAL CONTINUE** based on:
1. The census has genuine scholarly value (conceded)
2. The restructured project (spectral features + census at JoC) is fundamentally different from the rejected triple-threat proposal
3. The kill gates (especially G1 at week 2) provide early termination if the spectral premise fails

The Skeptic explicitly notes: the *original* proposal remains REJECTED. "CONDITIONAL CONTINUE" applies only to the restructured version. The psychological risk of scope creep back toward the original framing is the Skeptic's primary concern.

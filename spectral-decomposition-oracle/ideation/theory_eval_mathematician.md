# Verification Evaluation — Deep Mathematician

**Project:** spectral-decomposition-oracle (proposal_00)  
**Title:** "Spectral Features for MIP Decomposition Selection: A Computational Study with the First MIPLIB 2017 Decomposition Census"  
**Stage:** Verification (post-theory)  
**Evaluator:** Deep Mathematician — evaluates by quantity and quality of NEW, load-bearing math  
**Date:** 2025-07-22  
**Team:** 3-expert panel (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with cross-critique

---

## Executive Summary

This project proposes spectral features from the constraint hypergraph Laplacian for MIP decomposition method selection, accompanied by the first systematic decomposition census of MIPLIB 2017. After theory development: **theory_bytes=0** (pipeline measurement bug — approach.json ~50KB and paper.tex ~64KB exist), **impl_loc=0**. The math inventory contains ~0.8–1.0 novel theorem-equivalents, none of which is load-bearing for the artifact's difficulty or its value. The census (engineering artifact) and the feature definitions (straightforward spectral algebra) are the real contributions. The math dresses up these engineering contributions but does not drive them.

**Composite: 4.0/10** (V4/D4/BP2/L6/F4)  
**Verdict: CONDITIONAL CONTINUE** — gated on G1 pilot (Week 2, ρ ≥ 0.4) with 6 mandatory fixes.

---

## Team Process

Three independent evaluations were produced, followed by adversarial cross-critique and synthesis.

| Expert | Composite | Verdict | Key Contribution |
|--------|-----------|---------|------------------|
| Independent Auditor | 3.3/10 | CONDITIONAL CONTINUE | Best-calibrated scoring; identified 4 fixable fatal flaws |
| Fail-Fast Skeptic | 2.3/10 | ABANDON | Most technically precise attack; identified statistical power and Laplacian inconsistency problems |
| Scavenging Synthesizer | 5.0/10 | CONTINUE | Best strategic framing; identified degradation ladder as key risk mitigant and census as durable floor |

**Consensus points (high confidence):**
1. Software difficulty D=4 (moderate, not high)
2. Census is the most valuable artifact
3. T2 is vacuous and correctly demoted to motivational
4. L3 proof has two non-trivial gaps that must be fixed
5. Laptop feasibility is adequate (L5–6)
6. AutoFolio baseline is missing and required

**Key disagreements resolved:**
1. L3 novelty: incremental (not trivial, not headline) — L3-C specializations carry the real novelty
2. Publication probability: P(JoC) ≈ 0.16, but P(some publication) ≈ 0.63 via degradation ladder
3. Fatal flaw count: 1 truly fatal (spectral-density proxy risk, testable at G1), 6 fixable-serious
4. Statistical power: adequate for 3-class evaluation, problematic for Benders-specific claims
5. Value: V4 (census floor V3–4, spectral methodology adds conditional V+1 if G1 passes)

---

## Mathematical Content Assessment

**This is the core of my evaluation as a deep mathematician.** The question is: does this project require NEW math that is load-bearing — math that makes the artifact hard to build AND is the reason it delivers extreme value?

### Math Inventory (Claimed vs. Actual)

| Item | Claimed Status | Actual Assessment | Novel Theorem-Equivalents |
|------|---------------|-------------------|--------------------------|
| **L3** (Partition-to-Bound Bridge) | Main theorem | Geoffrion (1974) Lagrangian duality restated in hypergraph notation. The (nₑ−1) factor is a counting argument from variable duplication. Proof has 2 gaps. | ~0.2 (restatement with minor extension) |
| **L3-C Benders** | Method specialization | Genuinely non-obvious: uses Benders reduced costs rⱼ^(t) as partition quality metric without solving monolithic LP. Bridges variable-partition (Benders) with hypergraph language. | ~0.3 (the most novel item) |
| **L3-C DW** | Method specialization | Uses column-generation duals μᵢ^(t) with tight |blocks(i)|−1 factor. Bridges constraint-partition (DW) with the same quality framework. | ~0.3 (genuinely useful) |
| **L3-sp** (Spectral Partition) | Supporting lemma | O(δ²·d_max/γ_k²). Relies on balanced-cluster assumption violated by real constraint matrices. d_max factor makes it vacuous on dense constraints. Proof sketch only. | ~0.1 (largely vacuous) |
| **T2** (Spectral Scaling Law) | Demoted to motivational | C = O(k·κ⁴·‖c‖∞) evaluates to >10²⁴ on typical instances. δ<γ/2 assumption unstated. Mathematically valid but empirically meaningless. | 0.0 (vacuous = not a result) |
| **F1** (Permutation Invariance) | Feature property | Straightforward verification that eigenvalues of symmetric matrix are permutation-invariant. | 0.05 (textbook) |
| **F2** (Scaling Sensitivity) | Feature property | κ(D) bound after correction. Covers γ₂ only, not all 8 features. | 0.05 (standard perturbation theory) |

**Total novel theorem-equivalents: ~0.8–1.0**

For comparison: a strong theory paper at MPC/IPCO contains 2–4 novel theorem-equivalents. A strong computational paper at JoC typically contains 0.5–1.5 supporting lemmas. This project sits at the low end of computational paper territory.

### Is the Math Load-Bearing?

**No.** This is the critical finding. I evaluate load-bearing on two axes:

**Axis 1: Does the math make the artifact hard to build?**
- The artifact's difficulty is: (a) constructing a robust hypergraph Laplacian from non-square, mixed-sign constraint matrices, (b) numerical stability of eigensolves on near-singular Laplacians, (c) GCG/SCIP integration and census pipeline engineering.
- None of these require L3, T2, or the L3-C specializations. The spectral engine requires standard spectral graph theory (Ng-Jordan-Weiss 2001, Davis-Kahan perturbation). L3/T2 are post-hoc justifications, not design drivers.
- **The math is not the reason the artifact is hard.**

**Axis 2: Does the math deliver extreme value?**
- The artifact's value comes from: (a) the census dataset (zero new math required), (b) the 8 spectral feature definitions (standard spectral algebra), (c) empirical validation that features predict decomposition benefit (ML, not math).
- L3 provides a partition quality metric — useful but not the reason anyone would use this tool. The oracle's recommendations come from the trained classifier, not from L3's bound.
- T2 was supposed to theoretically justify why spectral features predict decomposition quality. It fails (vacuous constant). The features must stand on empirical evidence alone.
- **The math is not the reason the artifact delivers value.**

### The Ornamental Math Problem

T2 is the clearest case of ornamental math. It was originally positioned as the headline theoretical contribution — a "spectral scaling law" explaining why δ²/γ² predicts decomposition quality. But C = O(k·κ⁴·‖c‖∞) evaluates to astronomical numbers, making the bound uninformative. The project correctly demoted T2 to motivational status, but "motivational" means ornamental: it occupies 2 pages of the paper without constraining the design, improving the system, or enabling any capability.

L3 is a borderline case. It formalizes "partitions that cut fewer/lighter constraints produce better decompositions" — a truism that every decomposition practitioner knows intuitively. The formalization has some value (it gives a computable metric) but L3 is retrospective: it evaluates a partition after solving the LP, rather than guiding partition selection prospectively. The L3-C specializations partially fix this (they use iteration-specific duals), making them more useful than L3 itself.

**Bottom line as mathematician:** ~0.8–1.0 novel theorem-equivalents, of which ~0.6 are useful (L3-C Benders and L3-C DW) and ~0.4 are ornamental (T2, L3-sp). None is load-bearing. This is a computational study wearing a thin mathematical costume.

---

## Pillar Scores

### 1. Extreme Value — 4/10

**Who benefits:** 50–100 active decomposition research groups worldwide. Not OR practitioners at large.

**Census value (floor, V3–4):** The first systematic cross-method decomposition evaluation of MIPLIB 2017 fills a genuine gap. No existing public dataset answers "which MIPLIB instances benefit from Benders vs. DW vs. neither, and by how much?" Estimated 10–20 citations over 5 years as a reference dataset.

**Spectral feature value (conditional, +1 if G1 passes):** If spectral features prove non-redundant with syntactic features, they offer a new interpretable feature family for MIP instance characterization. The reformulation-selection framing (distinct from algorithm selection) is a genuine conceptual contribution that extends Kruber et al. (2017).

**Limiting factors:**
- Only 10–25% of MIPLIB has exploitable block structure (Bergner et al. 2015)
- CPLEX 22.1 and Gurobi already ship automatic decomposition heuristics
- The oracle helps on ~50–75 instances where Benders and DW meaningfully differ
- The census covers 500/1,065 instances for decomposition evaluation (not "complete")

**Score: 4/10.** The census has real scholarly value for a niche community. The spectral methodology adds conditional value. Not "extreme and obvious."

### 2. Genuine Software Difficulty — 4/10

**What is genuinely hard:**
1. Hypergraph Laplacian construction from rectangular, mixed-sign constraint matrices (~2K LoC, genuinely novel adaptation)
2. Numerical robustness of eigensolves across MIPLIB diversity (near-singular Laplacians, silent ARPACK failures)
3. GCG integration (brittle API, version-specific compatibility)

**What is not hard:**
- ML pipeline (sklearn Random Forest/GBT): standard Kaggle-level code
- SCIP Benders wrapper: API configuration
- Census infrastructure: engineering grind
- Feature extraction: scalar arithmetic on eigenvalues

The novel intellectual content concentrates in ~6.5K of ~26.5K LoC. The difficulty is engineering (making spectral computations reliable across 1,065 diverse instances), not mathematical.

**Score: 4/10.** Moderate engineering difficulty. Low mathematical difficulty.

### 3. Best-Paper Potential — 2/10

**Evidence against:**
- T2 is vacuous (not a theoretical anchor)
- L3 is incremental (Geoffrion 1974 genealogy visible to any reviewer)
- L3 proof has 2 unfixed gaps
- Core experiment is unrun (impl_loc=0)
- AutoFolio baseline missing
- Prior panel unanimously scored BP=3/10 (P(best-paper) ≈ 0.03)
- Theory stage produced no evidence to revise upward

**Best-case scenario (P ≈ 3%):** Census reveals genuinely surprising structural finding AND spectral features dominate syntactic by ≥10pp AND C-lite oracle closes instances that GCG alone cannot. This conjunction is low-probability.

**Why below binding ceiling (3):** The depth-check ceiling of 3 assumed L3 proof gaps would be fixed and the evaluation would be complete. Neither is true. The red-team resolution rate (4–6.5/12 SERIOUS) is below the 8/12 threshold.

**Score: 2/10.** Publishable at JoC with significant revision. Not competitive for best paper at any realistic target venue.

### 4. Laptop-CPU Feasibility & No-Humans — 6/10

**Feasible on laptop:**
- Spectral annotation: ~9 hours for all 1,065 instances ✓
- ML training/evaluation: seconds ✓
- 50-instance pilot: hours ✓
- 200-instance development: 3–4 days on 4 cores ✓

**Strains laptop:**
- Full 500-instance census: 5–7 days on 4 cores (feasible as batch)
- Largest instances: 2–4 GB memory per solver run
- GCG compilation in Docker: painful but doable

**No-humans:** Fully automated. Labels derived from solver outputs.

**Score: 6/10.** Meets the binding ceiling. Tiered design makes iterative development practical.

### 5. Feasibility — 4/10

**Risk inventory:**

| Risk | P | Impact |
|------|---|--------|
| Spectral hypothesis fails (G1) | 0.40 | Fatal for spectral thesis; census survives |
| L3 proof gaps unfixable for general A | 0.10 | Serious — paper loses theoretical anchor |
| GCG integration failure | 0.15 | Serious — DW evaluation impossible |
| Eigensolve numerical instability | 0.30 | Moderate — corrupts features on 5–15% of instances |
| Class imbalance → inconclusive results | 0.40 | Moderate — constrains claim strength |
| Timeline overrun (16 weeks for proofs + 26.5K LoC + census + paper) | 0.35 | Serious — no slack |

**Compound risk:** P(at least one serious risk materializes) ≈ 63%.  
**Publication probability:** P(JoC) ≈ 0.16. P(some publication) ≈ 0.63 (corrected for correlated failures in degradation ladder).

**Score: 4/10.** Buildable but risky. The degradation ladder prevents total failure (P(zero output) ≈ 0.05) but the most likely outcome is a B-tier data/computational paper, not the headline JoC contribution.

---

## Fatal Flaws

### Truly Fatal (1)

**FF1: Spectral features may be pure density proxies (P ≈ 0.30–0.40).**
If γ₂ is linearly predictable from constraint density and degree statistics (R² ≥ 0.70), the entire spectral thesis collapses. The features become expensive ($30s/instance) ways to compute what syntactic features ($0.01s/instance) already capture. **Detection:** G1 gate at Week 2. **If G1 fails → ABANDON spectral thesis, pivot to census-only.**

### Fixable-Serious (6)

| # | Flaw | Fix | Effort |
|---|------|-----|--------|
| FS1 | L3 proof gaps (Step 3 dual feasibility, Step 5 (nₑ−1) derivation) | Rewrite proof with explicit dual construction; derive factor from variable-duplication Lagrangian | 3–5 days |
| FS2 | Two incompatible Laplacians (clique vs. incidence) with d_max=200 discontinuity | Unify to single Laplacian variant, or include Laplacian-type as classifier feature | 2–3 days |
| FS3 | Statistical power inadequate for fine-grained Benders-specific claims (~25 test instances/fold) | Reformulate as 3-class primary + continuous ρ secondary; report per-class CIs honestly | 1–2 days |
| FS4 | AutoFolio + SPEC-8 baseline missing (the most natural comparator) | Add AutoFolio with/without spectral features to evaluation plan | 3–5 days |
| FS5 | Title claims "Complete" census; only 500/1,065 evaluated for decomposition | Change "Complete" to "Systematic"; distinguish spectral annotation (1,065) from decomposition evaluation (500) | 1 day |
| FS6 | 3/12 SERIOUS red-team findings unaddressed (S-1 dual degeneracy, S-4 δ<γ/2 assumption, S-8 AutoFolio) | Address each directly: specify dual choice in L3, add assumption to T2/L3-sp, add AutoFolio baseline | 2–3 days |

**Total fix effort: ~12–18 days.** All must complete before Week 1 experiments.

### Already Fixed (by Amendment E)

- T2 demoted to motivational (was: headline theorem with vacuous constant) ✓
- 155K LoC → 26.5K LoC (honest scoping) ✓
- External baselines break evaluation circularity ✓
- "Certificate" → "predictor" throughout ✓
- JoC targeting replaces MPC/IPCO aspiration ✓

---

## Novel Theorem-Equivalents Assessment

| Item | Claimed | Actual | Reason for Discount |
|------|---------|--------|---------------------|
| L3 (Partition-to-Bound Bridge) | ~1.0 | ~0.2 | Geoffrion 1974 restatement with minor extension |
| L3-C Benders | ~0.5 | ~0.3 | Genuinely useful specialization, but straightforward once L3 exists |
| L3-C DW | ~0.5 | ~0.3 | Best novel item; |blocks(i)|−1 tightening is nice |
| L3-sp (Spectral Partition) | ~0.5 | ~0.1 | Vacuous on dense constraints; balanced-cluster assumption violated |
| T2 (Spectral Scaling Law) | ~0.5 | 0.0 | Vacuous constant = not a result |
| F1, F2 | ~0.2 | ~0.1 | Textbook spectral theory |
| **Total** | **~3.2** | **~1.0** | **~70% discount for ornamental/vacuous/standard** |

**Mathematician's assessment:** ~1.0 novel theorem-equivalents, of which ~0.6 are genuinely useful (L3-C specializations) and ~0.4 are ornamental. This is at the very low end for a JoC paper, which typically requires 0.5–1.5 supporting results. The math is adequate for a computational study but not load-bearing — it neither drives the difficulty nor delivers the value.

---

## Verdict

### CONDITIONAL CONTINUE

**Composite: 4.0/10** (V4/D4/BP2/L6/F4)

**Rationale:** The project has a viable path to a solid computational paper at JoC, but the mathematical contribution is thin and ornamental. The real value is the census artifact and (if validated) the spectral feature family. The degradation ladder prevents catastrophic failure (P(zero output) ≈ 0.05). The only truly fatal flaw (spectral-density proxy risk) has a cheap, fast test (G1 at Week 2).

### Binding Conditions

| # | Condition | Deadline | Consequence |
|---|-----------|----------|-------------|
| C1 | Fix all 6 FS-flaws | Before Week 1 experiments | Cannot start experiments with known design flaws |
| C2 | **G1 gate: ρ ≥ 0.4** (spectral features vs. decomposition benefit, after partialing out density) | **Week 2** | **HARD KILL. If ρ < 0.4 → ABANDON spectral thesis.** Evaluate census-only pivot. |
| C3 | Unify Laplacian variant | Before feature extraction | Feature space must be internally consistent |
| C4 | Add AutoFolio baseline | Before Week 4 evaluation | Reviewer will reject without this comparison |
| C5 | Report honest power analysis in paper | In draft | Non-negotiable for any venue |

### Probability Estimates

| Outcome | Probability |
|---------|-------------|
| JoC publication (full oracle + census) | ~0.16 |
| C&OR/CPAIOR publication (census + partial results) | ~0.25 |
| Workshop/data paper (census only) | ~0.22 |
| Zero output | ~0.05 |
| **P(any publication)** | **~0.63** |
| **P(best-paper at any venue)** | **~0.02–0.03** |
| **P(abandon at gate)** | **~0.30–0.40** |

### Mathematician's Final Note

As a mathematician evaluating load-bearing math: this project scores poorly. The ~1.0 novel theorem-equivalents are adequate scaffolding for a computational study but are not the reason the artifact is hard (that's engineering) or the reason it delivers value (that's the census). The math wears a spectral costume but the body underneath is standard Lagrangian duality (L3), standard perturbation theory (F2, L3-sp), and a vacuous scaling law (T2). The L3-C specializations are the lone genuinely novel mathematical contribution, and they are small results (~0.3 theorem-equivalents each).

The project should continue — not because of its math, but despite it. The census and spectral features are worth building as engineering artifacts. The math provides a thin theoretical membrane over a solid computational core. That's fine for JoC, which values computational contributions. It is not fine for any venue that values mathematical novelty.

**The Skeptic dissents ABANDON at 2.3/10.** The Skeptic's core argument — that 16 weeks for a 16% chance of the headline contribution is a poor bet — is technically correct for the top-line outcome. However, the degradation ladder creates a portfolio with E[impact] ≈ 1.4–1.6 impact-units, slightly above safe alternatives. The Skeptic would be right to abandon if a clearly superior project were available; absent that information, continuation is marginal but defensible.

---

## Score Table (Final)

| # | Pillar | Score | Notes |
|---|--------|-------|-------|
| 1 | Extreme Value | **4** | Census V3–4 floor + conditional spectral V+1. Niche audience. |
| 2 | Genuine Software Difficulty | **4** | Spectral engine + census pipeline moderately hard. Math is easy. |
| 3 | Best-Paper Potential | **2** | Incremental math, niche audience, unrun experiment. P(best-paper) ≈ 2–3%. |
| 4 | Laptop-CPU & No-Humans | **6** | Eigensolves fast. Census is batch. Fully automated. |
| 5 | Feasibility | **4** | P(any pub) ≈ 0.63. P(JoC) ≈ 0.16. 63% compound risk. Degradation ladder prevents wipeout. |
| | **Composite** | **4.0** | V4/D4/BP2/L6/F4 |
| | **Verdict** | **CONDITIONAL CONTINUE** | Gated on G1 (Week 2, ρ ≥ 0.4). 6 mandatory fixes. Skeptic dissents ABANDON at 2.3. |

---

*Evaluation produced by 3-expert team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with cross-critique synthesis, under Deep Mathematician lead. All scores evidence-based against project artifacts. Binding depth-check ceilings honored except where theory-stage evidence justifies adjustment.*

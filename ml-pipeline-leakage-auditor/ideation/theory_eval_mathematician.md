# Deep Mathematician Verification: TaintFlow (proposal_00)

**Evaluator:** Deep Mathematician (verification stage)
**Proposal:** TaintFlow — Quantitative Information-Flow Auditing via Hybrid Dynamic-Static Analysis with Provably Tight Channel Capacity Bounds
**Method:** Claude Code Agent Teams — 3-role adversarial evaluation (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with cross-critique synthesis and lead adjudication
**Date:** 2026-03-08

---

## Executive Summary

TaintFlow proposes a hybrid dynamic-static ML pipeline auditor that executes once under lightweight instrumentation to extract a dataflow DAG, then applies provenance-parameterized channel capacity bounds to produce per-feature, per-stage leakage measurements in bits. The mathematical core comprises one genuinely novel modeling insight (the fit-transform channel decomposition via sufficient statistics), a systematic capacity catalog for ~16 ML operations, and a conditional soundness theorem built on standard information-theoretic machinery.

After adversarial team evaluation: **Composite 6.4/10** (V7/D7/BP4/L9/F5). The proposal identifies a real problem and proposes a technically sound approach at a genuine intersection of QIF, abstract interpretation, and ML pipeline semantics. However, **theory_bytes = 0** after the theory stage, the crown jewel theorem has a gap in its proof sketch (Step 3), and the value proposition faces a credible 50-line competitor for the detection use case. The minimum viable paper is more likely than not (~60%) but the full vision is roughly a coin flip (~35%).

**VERDICT: CONDITIONAL CONTINUE** — with 4 hard gates, detailed below.

---

## Team Process

Three independent evaluators produced parallel assessments:

| Role | Default Stance | Key Finding |
|---|---|---|
| **Independent Auditor** | Evidence-based | Feasibility=5 (sharp divergence from self-assessed 7); Gaussian assumption silently invalidates bounds for non-sub-Gaussian data |
| **Fail-Fast Skeptic** | Presume ABANDON | The fit-transform "lemma" may reduce to one application of DPI; the 50-line accuracy-delta competitor is devastating for the detection use case |
| **Scavenging Synthesizer** | Extract max value | The capacity catalog has standalone reference value; minimum viable paper (~6 months) is publishable even with weakened T1 |

Cross-critique resolved 5 key disagreements. The lead mathematician adjudicated each dispute by examining the actual mathematical content of the proof sketches.

---

## Pillar 1: Extreme Value — Score: **7/10**

**What the team agrees on:** The problem is real and well-documented. 15–25% of Kaggle kernels contain leakage (Yang et al., ASE 2022). No existing tool produces per-feature quantitative attribution in bits. LeakageDetector is binary/syntactic (3 patterns); LeakGuard requires re-execution and gives model-dependent accuracy deltas. The gap in the prior art is genuine.

**The Skeptic's 50-line competitor attack:** Run the pipeline correctly (fit on train only) vs. incorrectly (fit on all), measure accuracy delta. This gives practitioners what they actually want — an accuracy impact in percentage points — in their native units. This attack is devastating for the *detection* use case (~70% of practitioner needs) but irrelevant for the *attribution* use case. TaintFlow's genuine incremental value is:

1. **Per-feature, per-stage attribution** without re-execution — "Feature `age_normalized` has 3.7 bits of contamination from `StandardScaler.fit_transform()` at line 47"
2. **Severity ordering across features** — correct ranking even when absolute bounds are loose
3. **CI/CD integration** — automated gate without training a model

**Why not 8+:** The tool requires pipeline execution with data access (self-debugging only, not third-party audit). The PCA bound is vacuous for d > 50 (common in real pipelines). Most practitioners would act on binary detection alone — the quantitative difference between 0.007 bits and 0.5 bits, while theoretically clean, maps to the same action ("fix it" or "ignore it") in most cases.

**Why not 5:** The attribution use case is genuine and unserved. No tool today can answer "which stage, which feature, how much?" Platform teams running CI/CD gates on feature pipelines need per-feature specificity that binary detection cannot provide. The capacity catalog has standalone reference value.

---

## Pillar 2: Genuine Software Difficulty — Score: **7/10**

### Mathematical Difficulty Assessment

The proposal claims 4 mathematical contributions. I examined each for load-bearing novelty:

| Contribution | Claimed | Actual | Load-Bearing? |
|---|---|---|---|
| **T1: Fit-Transform Channel Decomposition** | ★★★ genuinely hard | ★★½ moderate with gap | **YES** — without it, no sound bounds for any estimator stage |
| **T2: Conditional Soundness** | ★★☆ non-trivial | ★★☆ agreed | **YES** — but the DPI induction is standard; novelty is in the conditional framing |
| **Capacity Catalog (Tier 1)** | ★★★ | ★★☆ | **YES** — fuel for transfer functions; but most entries are textbook formulas |
| **T3: Sensitivity Composition** | ★½ | ★½ agreed | Marginally — standard composition |
| **T4: Min-Cut Attribution** | ★½ | ★½ agreed | Marginally — Cover & Thomas Ch. 15 |
| **T5: Worklist Termination** | ★½ | ★☆ routine | No — Tarski's theorem on a height-69 lattice |

**The fit-transform lemma examined in detail (T1):** This is the decisive item. The Skeptic claims it's "one line of DPI under an assumed Markov chain." The Synthesizer claims it's "a genuine intellectual contribution that doesn't exist anywhere." After examining the proof sketch:

- **Step 1 (Markov factorization):** Recognizing that `StandardScaler.fit()` creates a sufficient-statistic channel is a genuine *modeling* contribution — domain-specific insight that has no precedent in QIF. This is not DPI; this is identifying the right structure to apply DPI to.
- **Step 2 (DPI application):** Textbook. I(D_te; θ) ≤ C_fit is standard.
- **Step 3 (the gap):** The claim I(D_te; X_j | θ) ≤ b_{input,j} = I(D_te; X_j) is **asserted without proof and is not generally true.** By the chain rule: I(D_te; X_j | θ) = I(D_te; X_j) + I(D_te; θ | X_j) − I(D_te; θ). This can exceed I(D_te; X_j) when I(D_te; θ | X_j) > I(D_te; θ) — i.e., when knowing X_j makes θ more informative about D_te. For multi-column estimators (PCA), where θ depends on columns beyond j, this is plausible. The self-assessed 30% risk-of-weakening likely underestimates this.

**Verdict on T1:** A **moderate intellectual contribution** — a novel modeling insight dressed in standard information-theoretic machinery, with a genuine gap that may require strengthened assumptions or a looser bound. Neither trivial (Skeptic wrong) nor a deep theorem (Synthesizer overclaims).

### Engineering Difficulty

The Skeptic correctly notes that most of the 55–65K LoC is engineering:
- Monkey-patching 300+ pandas methods: well-trodden (coverage.py, pytest-cov)
- Worklist over height-69 lattice: textbook
- 130 transfer functions: ~20 distinct templates × 6–7 instantiations each

**Genuinely novel, non-repetitive code: ~15–20K LoC** (capacity catalog derivations + fit-transform handling + lattice instantiation + provenance tracking + ~20 distinct transfer function templates). The remaining 35–45K is important engineering but not intellectually novel.

---

## Pillar 3: Best-Paper Potential — Score: **4/10**

**Consensus P(best_paper):**

| Venue | P(best_paper) | P(accept) |
|---|---|---|
| ICML/NeurIPS main | 0.01–0.02 | 0.40–0.50 |
| ICML/NeurIPS systems track | 0.02–0.04 | 0.50–0.60 |
| OOPSLA/FSE | 0.04–0.06 | 0.55–0.65 |
| NeurIPS D&B (catalog as artifact) | 0.02–0.03 | 0.60–0.70 |

**Why not higher:** Best papers at ICML/NeurIPS require either a surprising empirical result on a major benchmark or a theoretical breakthrough with broad impact. T1 is novel but narrow (sufficient-statistic estimators in sklearn pipelines). T2–T5 are self-rated "straightforward" to "standard adaptation." The conditional soundness qualification is a reviewer lightning rod, especially at PL venues. The proposal has zero empirical results to demonstrate practical impact.

**Why not lower:** The unexploited intersection of QIF × abstract interpretation × ML pipeline semantics is genuine — no prior work exists here. The fit-transform insight, even weakened, is publishable. The capacity catalog has standalone reference value. At OOPSLA/FSE, the hybrid architecture is a solid systems contribution. If the evaluation produces compelling results (Spearman ρ ≥ 0.80 for severity ordering, detection of previously unknown leakage in published papers), best-paper potential improves to ~0.08 at FSE.

---

## Pillar 4: Laptop-CPU Feasibility — Score: **9/10**

**Unanimous agreement across all three evaluators.** This is a major strength.

- Phase 2 analysis: O(K × d² × H) with K≈50, d≈200, H=69 → ~138M lattice updates → <100ms in Rust
- Phase 1 (dynamic DAG extraction): bounded by pipeline execution time, which by definition already runs on the user's machine
- Phase 3 (KSG estimation): O(n²d), optional, several seconds for typical datasets
- No GPUs, no clusters, no human annotation, no human studies
- Memory: O(K×d) for abstract states + O(n/64) per provenance bitmap

**Score reduced from 10 only because:** (a) OOM risk on pathological merges (many-to-many Cartesian products: 250MB+ per merge, <5% of real pipelines, documented fallback to ρ=0.5); (b) full instrumented pipeline execution may be slow for large datasets.

---

## Pillar 5: Feasibility — Score: **5/10**

**This is where the team diverges most sharply from the self-assessment (Feasibility = 7).**

### The binding constraint: theory_bytes = 0, impl_loc = 0

After the full ideation and theory specification stages, zero proofs have been written and zero code exists. The approach.json contains 525 lines of detailed formal specification — pseudocode, proof sketches, module mappings — but a proof sketch is not a proof. The gap in T1 Step 3 demonstrates this: the sketch asserts a critical inequality that may not hold in general.

### Compound probability analysis

| Risk | P(materialized) | Impact |
|---|---|---|
| T1 must weaken (Step 3 gap + sufficiency fails for some estimators) | 0.35–0.45 | Crown jewel covers ≤5 estimators |
| Tight catalog covers only ~15 operations (not 80+) | 0.40 | Tier 2 handles the rest (sound-but-loose) |
| Timeline overrun (verification math: 25% underestimate) | 0.60 | 10–14 months instead of 8 |
| Conditional soundness rejected by reviewers | 0.25 | Must reframe or target different venue |
| Gaussian assumption violated for real data | 0.20 | Unsound bounds for heavy-tailed features |

**Compound probability of full paper success:** P(T1 holds) × P(instrumentation works) × P(enough TFs) × P(empirical results compelling) ≈ 0.60 × 0.85 × 0.80 × 0.70 ≈ **0.29–0.36**

**Minimum viable paper success probability:** ~0.55–0.65 (T1 for StandardScaler/PCA/SimpleImputer + 16 Tier 1 bounds + 200 synthetic + 50 real pipelines)

### The Math Lead's earlier verification found:
- Lattice height calculation was wrong (68 not 520) — conservative error, no impact
- Soundness theorem needs qualification to "observed execution path" — acknowledged
- Difficulty underestimated by ~25% (12–14 months vs 10.5 stated)
- Fit-transform lemma correctly identified as central novelty

---

## Fatal Flaws (surviving adversarial cross-examination)

### Flaw 1: Zero Completed Proofs After Theory Stage — **CRITICAL**

State.json shows theory_bytes = 0, impl_loc = 0. The crown jewel theorem exists only as a 3-step proof sketch with a genuine gap in Step 3. Until this is formalized, the central theorem is an aspiration, not a result. All five theorems and all 16 capacity bounds are stated but unproved.

### Flaw 2: Step 3 Gap in the Crown Jewel Theorem — **HIGH**

The fit-transform decomposition lemma's proof sketch claims I(D_te; X_j | θ) ≤ b_{input,j}. This is not generally true. For PCA (where θ = eigenvectors of the full covariance), conditioning on θ constrains the data in ways that may *increase* the mutual information between D_te and X_j. The proof must either: (a) establish an additional conditional independence structure specific to each estimator class, or (b) use a different bounding strategy (e.g., I(D_te; Y_j) ≤ I(D_te; X) which is sound but looser). The self-assessed 30% risk-of-weakening is likely an underestimate given this specific gap.

### Flaw 3: Conditional Soundness as Peer Review Target — **MODERATE**

The conditions C1 (instrumentation faithfulness) and C2 (capacity bounds hold) do substantial heavy lifting. A hostile reviewer can construct a pipeline where `if random() > 0.5: leak_badly() else: be_clean()` — the tool certifies whichever path executed. The Synthesizer's suggestion to rename to "execution-faithful soundness" is cosmetic, not substantive. The paper must defend this honestly as the correct claim for interactive debugging (not certification).

### Flaws that did NOT survive cross-examination:
- **"Tightness story collapses" (Skeptic):** Overstated. 7 operations at κ=O(1) is a real catalog. "Tight within O(log n)" for median/quantile is honest, not vacuous.
- **"Regulatory audit is fantasy" (Skeptic):** Correct but irrelevant — the paper doesn't need this use case.
- **"50-line competitor kills the project" (Skeptic):** Only partially valid — devastating for detection, irrelevant for per-feature attribution.

---

## Biggest Gap Between Claimed and Actual Novelty

The **Provenance-Parameterized Channel Capacity Catalog** is rated ★★★ in the proposal. Actual novelty: **★★☆**.

Evidence:
- The mean bound is acknowledged as "textbook Gaussian channel"
- Sum, count, SimpleImputer bounds are trivial corollaries of mean
- StandardScaler and MinMaxScaler are direct applications of T1 + mean/std
- The Math Lead in the critiques notes "a reviewer could dismiss this as a table of known results applied to a list of functions"

The genuine novelty within the catalog is concentrated in ~4 entries: median (rank-channel reduction), PCA (Wishart analysis — but self-admittedly "the weakest and likely vacuous"), target encoding (Fano bound application), and covariance entries. Calling the overall catalog ★★★ inflates what is largely a systematic application of textbook information theory to a curated list of sklearn operations. A fair rating: **★★☆** — useful reference, moderate novelty.

---

## Genuinely Novel LoC Assessment

| Component | Claimed | Honest Novel | Notes |
|---|---|---|---|
| Dynamic instrumentation + DAG | ~15K | ~5K | Monkey-patching is well-trodden; provenance tracking is the novel part |
| Lattice + abstract domain | ~10K | ~4K | Product lattice is textbook; DataFrame instantiation is genuine |
| Capacity catalog | ~8K | ~4K | ~16 distinct derivations, rest are parameterized copies |
| Transfer functions (130 ops) | ~12K | ~4K | ~20 distinct templates × 6–7 instances each |
| Propagation + attribution | ~8K | ~2K | Worklist + Ford-Fulkerson are standard |
| CLI + reports | ~5K | ~1K | SARIF serialization |
| **Total** | **~55–65K** | **~15–20K** | Remaining 35–45K is important engineering but not novel |

---

## Score Summary

| Dimension | Self-Assessment | Auditor | Skeptic | Synthesizer | **Consensus** |
|---|---|---|---|---|---|
| Extreme Value | 8 | 7 | 5 | 9 | **7** |
| Genuine Difficulty | 7 | 7 | 5 | 8 | **7** |
| Best-Paper Potential | 8 (P=0.12) | 4 (P≈0.06) | 3 (P≈0.03) | 7 (P≈0.10) | **4** (P≈0.04 at OOPSLA) |
| Laptop-CPU Feasibility | (implied 9+) | 9 | 8 | 9 | **9** |
| Feasibility | 7 | 5 | 4 | 6 | **5** |

**Composite: V7 / D7 / BP4 / L9 / F5 = 6.4/10** (self-assessed 7.5)

---

## What the Math Actually Is

As a deep mathematician, I evaluate the load-bearing math — math that is the reason the artifact is hard to build and the reason it delivers extreme value. Ornamental math that doesn't drive the system is worthless.

**Load-bearing math in this proposal:**

1. **The fit-transform Markov factorization (T1):** This is the one genuinely novel modeling insight. Recognizing that `fit_transform` creates a sufficient-statistic channel that factors into aggregation + application is not a deep theorem but it *is* a non-obvious observation that enables the entire analysis. Without it, every estimator stage gets ∞ bounds and the tool is useless. **This is load-bearing. Moderate novelty. Gap in proof.**

2. **The capacity catalog (Tier 1):** These bounds are the fuel for transfer functions. Without them, every edge gets B_max — trivially sound, completely useless. The provenance parameterization (exploiting exact observed ρ) is a clean insight. **This is load-bearing. Low-to-moderate novelty per entry, moderate as a collection.**

3. **The conditional soundness theorem (T2):** The DPI induction on the observed DAG is textbook, but the conditional framing — explicit enumeration of what the instrumentation captures and doesn't capture — is an honest contribution that makes the tool's guarantees meaningful. **Load-bearing for credibility. Low novelty.**

**Ornamental math (correctly dropped):**
- Galois insertion proofs (from Approach A) — zero operational benefit
- General-purpose widening — unnecessary for acyclic pipelines
- Reduced product theorem — tautological (min of two upper bounds ≤ either)
- Type system framing — notation without capability
- Principal type claims — open question that was never going to be solved

The proposal correctly stripped ornamental math during ideation. The remaining math is load-bearing but not deep. This is a systems paper with meaningful but not difficult theoretical backing, not a theory paper.

---

## VERDICT: CONDITIONAL CONTINUE

**Rationale:** The proposal sits at a real intersection of three mature fields (QIF, abstract interpretation, ML tooling) where no prior work exists. The fit-transform insight is genuine even if moderate. The capacity catalog has standalone reference value. The hybrid architecture makes the correct engineering tradeoff. The tiered degradation design means partial completion produces a publishable result.

However: zero proofs, zero code, a gap in the crown jewel theorem, and the 50-line competitor attack against the detection use case are concrete threats.

### Hard Gates (all must be met within 4 weeks or ABANDON):

**Gate 1 — Prove or fix T1:** Close the Step 3 gap. Either show I(D_te; X_j | θ) ≤ b_{input,j} under explicit Markov assumptions for StandardScaler and PCA, or weaken to the looser bound I(D_te; Y_j) ≤ I(D_te; X) and reframe the contribution honestly. Write up the complete proof. This is theory_bytes > 0 or abandon.

**Gate 2 — Implement 5-operation capacity catalog in Rust:** Mean, std, PCA, StandardScaler, SimpleImputer. Validate against closed-form linear-Gaussian ground truth on synthetic data. This is impl_loc > 0 or abandon.

**Gate 3 — One end-to-end pipeline:** Instrumentation → DAG extraction → abstract analysis → report for a single synthetic leakage example. Demonstrate the architecture works.

**Gate 4 — Confront the 50-line competitor:** Implement the accuracy-delta baseline. Run on 10 synthetic pipelines. Demonstrate TaintFlow provides information the baseline cannot (per-feature attribution, detection without model retraining, severity ordering). If the competitor catches >90% of what TaintFlow catches with comparable specificity, pivot to attribution-only framing.

### Binding Amendments (if CONTINUE past gates):

1. **Rename "conditional soundness" to "execution-path soundness"** — cosmetic but reduces reviewer trigger risk
2. **Recount tight bounds honestly:** κ=O(1) for ~7 operations; κ=O(log n) for ~8 more. Do not claim "15–20 tight bounds" without distinguishing the two tiers of tightness.
3. **Drop KSG empirical refinement (Phase 3) from the initial paper** — adds a proof obligation (finite-sample concentration, M9) for marginal benefit. Save for follow-up.
4. **Add "published paper audit" experiment** — scanning 20–30 ML papers with public code from ICML/NeurIPS for previously undetected leakage. Even 2–3 discoveries transforms the best-paper potential.
5. **Acknowledge the PCA d² vacuity explicitly** — do not let reviewers discover it; present it as an honest limitation with future eigenstructure-aware bounds as follow-up.
6. **Genuinely novel LoC: ~15–20K**, not 55–65K. The remaining is important engineering. Do not overclaim.

### Venue Strategy:

| Priority | Venue | Framing |
|---|---|---|
| 1 | **OOPSLA** | Lead with soundness theorem + abstract interpretation for DataFrames |
| 2 | **FSE/ICSE** | Lead with tool + evaluation. "First quantitative leakage auditor" |
| 3 | **ICML systems** | Lead with practitioner impact. Needs "found real bugs" story |
| 4 | **NeurIPS D&B** | Capacity catalog + benchmark as standalone artifact |

### Minimum Viable Paper (if crown jewel weakens):

**"Channel Capacity Bounds for ML Preprocessing: Quantifying Train-Test Leakage in Bits"**
- T1 proven for StandardScaler, PCA, SimpleImputer only (3 estimators)
- Capacity catalog for 7 κ=O(1) operations
- Soundness conditioned on A_suff + A_faithful (narrow but honest)
- Evaluation: 200 synthetic + 50 real Kaggle kernels
- Timeline: ~6 months
- P(accept at FSE/OOPSLA): ~0.45–0.55

### Kill Probability: **~0.35**

The project should be abandoned if:
- Gate 1 fails (T1 proof collapses entirely, not just weakens)
- Gate 3 fails (architecture doesn't work end-to-end)
- Gates 1+2+3 all met but Gate 4 shows the 50-line competitor achieves comparable results, and no attribution advantage can be demonstrated

---

## Minority Dissents

**Skeptic (dissenting on Value, score 5):** "The 50-line accuracy-delta script covers the high-severity cases. Quantifying leakage in bits is solving a problem practitioners don't have. The value is niche at best." *Lead response: Partially valid for detection; overruled for attribution use case.*

**Synthesizer (dissenting on Best-Paper Potential, score 7):** "The unexploited intersection of QIF × abstract interpretation × ML pipelines is category-creating. With strong empirical results, best-paper at OOPSLA is realistic." *Lead response: Conditional on empirical results that don't yet exist. The current P(best_paper) reflects zero evidence of practical impact.*

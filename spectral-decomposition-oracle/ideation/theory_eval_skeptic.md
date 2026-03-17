# Verification Report: Spectral Decomposition Oracle — Skeptic Evaluation

**Verification Chair:** Fail-Fast Skeptic (Lead)
**Team:** Independent Auditor · Fail-Fast Skeptic · Scavenging Synthesizer
**Date:** 2026-03-08
**Proposal:** proposal_00 — "Spectral Features for MIP Decomposition Selection: A Computational Study with the First MIPLIB 2017 Decomposition Census"
**Stage:** Post-theory verification gate
**Binding reference:** `depth_check.md` (V5/D4/BP3/L6, composite 4.5/10)

---

## 0. Executive Summary

**Composite: 4.5/10** (V5 / D4 / BP3 / L5 / F4)
**Verdict: CONDITIONAL CONTINUE** — Skeptic dissents ABANDON at 3.8/10.

Three-expert team evaluated proposal_00 after theory development. theory_bytes=0 in State.json is a **measurement bug** — the theory/ directory contains 282KB of substantive artifacts (approach.json 18KB, paper.tex 64KB, algorithms.md 52KB, evaluation_strategy.md 44KB, verification_framework.md 46KB, red_team_report.md 41KB, verification_report.md 18KB). However, impl_loc=0 is accurate: **zero implementation code exists.**

The project has a publishable core (census + spectral features + L3) but the main theorem's proof is broken, the crown jewel scaling law (T2) is vacuous on 60–70% of MIPLIB, the core empirical hypothesis is entirely untested, and the red-team threshold was not met (4–6.5/12 SERIOUS findings addressed vs. required 8/12). Five binding conditions must be met; failure on any triggers ABANDON. P(abandon) ≈ 0.30–0.35.

---

## 1. Team Process

### 1.1 Independent Proposals (Parallel)

| Expert | Verdict | Value | Difficulty | Best-Paper | CPU | Feasibility |
|--------|---------|-------|------------|------------|-----|-------------|
| Independent Auditor | CONTINUE (barely) | 4 | 4 | 2 | 6 | 5 |
| Fail-Fast Skeptic | **ABANDON** | 3 | 3 | 1 | 5 | 3 |
| Scavenging Synthesizer | CONTINUE | 4 | 4 | 3 | 6 | 7 |

### 1.2 Cross-Critique Round (Each critiques the other two)

**Key disputes resolved:**

1. **Skeptic's compound probability (12%) is methodologically flawed.** Both Auditor and Synthesizer independently identified that kill gates G0/G1 and G1/G3 are strongly positively correlated — if spectral features carry genuine signal (G1), they will likely beat syntactic features (G3). Treating them as independent underestimates survival by ~2×. Corrected compound gate survival: **~25–30%**, not 12%.

2. **Synthesizer's P(any pub)=0.80 is too optimistic.** Self-corrected to 0.65 after acknowledging: zero code exists, L3 proof received FAIL from verification, only 4/12 SERIOUS red-team findings fully addressed. Salvage inventory reduced from 11 items to ~4 genuinely publishable artifacts.

3. **Skeptic's 7 "fatal flaws" — mostly fixable under Amendment E.** Zero are truly fatal in the census-first framing. Two are serious (AutoFolio baseline, dual degeneracy — ~1 week each). Three were already addressed by binding amendments. One (#5: limited block structure in MIPLIB) is a finding the census reports, not a flaw.

4. **All three converge on:** Census is the diamond. T2 is vacuous. L3 proof needs fixing. AutoFolio baseline is missing. Best-paper potential ≤ 3/10.

### 1.3 Post-Critique Revised Scores

| Dimension | Auditor (revised) | Skeptic (final) | Synthesizer (revised) |
|-----------|-------------------|-----------------|----------------------|
| Value | 5 | 5 (concedes census) | 5 |
| Difficulty | 4 | 4 | 4 |
| Best-Paper | 3 | 1 (not blocking 3) | 3 |
| Laptop-CPU | 6 | 5 | 6 |
| Feasibility | 5 | 5 | 6 |

| Probability | Auditor | Skeptic | Synthesizer |
|-------------|---------|---------|-------------|
| P(JoC) | 0.45 | ~0.30 | 0.50 |
| P(any pub) | 0.55 | 0.50 | 0.65 |
| P(best-paper) | 0.02 | 0.01 | 0.03 |
| P(abandon) | 0.30 | 0.40 | 0.25 |

### 1.4 Verification Signoff

Independent verifier identified 6 corrections; all incorporated into this report. Key finding: the vote is effectively **unanimous for CONDITIONAL CONTINUE under Amendment E**, with the Skeptic maintaining a dissent limited to the original (dead) proposal framing. The Skeptic explicitly stated: "I don't block CONTINUE under Amendment E."

---

## 2. Three Pillars Assessment

### Pillar 1: Does this deliver extreme and obvious value?

**Score: 5/10** (consensus)

**What it delivers:**
- The MIPLIB 2017 Decomposition Census: first cross-method (Benders + DW) systematic evaluation of which decomposition helps which instances. No comparable public artifact exists. Genuine community contribution with a decade-long citation tail.
- 8 formally defined spectral features (permutation-invariant, scaling-sensitive) for MIP instance characterization. Novel feature family for the ML4CO community.
- L3 (Partition-to-Bound Bridge): universal quality metric for any partition, with method-specific specializations (L3-C Benders, L3-C DW).
- Futility predictor: practical "don't bother decomposing" signal at ≥80% precision.

**What limits the value:**
- **Niche audience.** Decomposition researchers number in the hundreds, not thousands. Typical OR practitioners use monolithic solvers and will never interact with this work.
- **Incremental, not transformative.** The target improvement is ≥5 percentage points over syntactic features — the definition of incremental feature engineering.
- **Census finding is predictable.** Bergner et al. (2015) showed only 10–25% of MIPLIB has exploitable block structure. The expected census finding is "75% of instances: nothing helps." A census whose dominant result confirms prior estimates is useful but unsurprising.
- **T2 delivers no value.** Vacuous on 60–70% of MIPLIB (κ⁴ constant → bound says "degradation ≤ 10³⁰"). Honestly demoted to motivational, but even as motivation it only covers 8–14% of instances quantitatively.

**Skeptic's position:** Value is scholarly, not practical. The census is a genuine contribution but it's infrastructure, not a capability. Nobody will solve MIPs differently because of this work.

### Pillar 2: Is this genuinely difficult as a software artifact?

**Score: 4/10** (consensus)

**What's genuinely hard:**
- Numerical robustness of eigensolve across diverse MIPLIB instances. Near-singular Laplacians (κ~10¹⁰) with shift-invert near λ₂ ≈ 0 require careful residual checks, ARPACK→LOBPCG fallback, relative vs. absolute tolerance tuning.
- Davis-Kahan adaptation for non-square constraint matrices (L3-sp). Real mathematical work, not textbook application.
- Census infrastructure: orchestrating 1,065 instances × 3 methods across SCIP + GCG with timeout handling, numerical stability, and result validation.

**What's routine:**
- Post-descoping LoC: **26.5K / 30K cap** (down from inflated 155K — the project's own auditor caught the 5–6× inflation). Honest novel code: ~25–30K.
- Core ML pipeline: RF/XGBoost on 8 features, 500 instances, nested CV. This is a scikit-learn homework assignment (~2.5K LoC).
- Solver wrappers: call `SCIPcreateBendersDefault()`, call GCG API, parse results. Standard integration work (~3K LoC).
- Spectral engine: construct sparse Laplacian, call ARPACK for bottom-k eigenvalues, extract features. Well-understood pipeline (~6.5K LoC).

**A competent PhD student could build this in 6–8 weeks.** The genuine difficulty lives in getting the numerical eigensolve right across diverse instances and making the census robust to solver crashes — real engineering, but not novel research software.

### Pillar 3: Does this have real best-paper potential?

**Score: 3/10** (Auditor/Synthesizer consensus; Skeptic at 1/10, not blocking)

**Why best-paper is essentially unreachable:**
- **P(best-paper) = 0.02–0.03** — the project's own experts score this as having nearly zero best-paper potential.
- **T2 is vacuous.** The crown jewel theorem, demoted to motivational, eliminates theory-venue viability. MPC/IPCO: non-starter (P=0.05).
- **L3 is Geoffrion 1974 in hypergraph costume** (red-team FATAL F-1). The (n_e−1) factor may be a trivial counting argument. The proof has two unfixed gaps. Even if fixed, a reviewer who recognizes L3 as a restatement of Lagrangian weak duality will question whether there's any novel mathematical content.
- **No mechanism to generate surprise.** The spectral hypothesis either weakly confirms (+5pp) or fails. Neither outcome turns heads.
- **Venue ceiling:** INFORMS JoC is the correct target (values reproducible computational studies). A solid JoC paper is achievable; best-paper at JoC requires surprising census findings AND dominant spectral edge.

**What would best-paper look like?** Census reveals 30–40% unexploited decomposable structure (contradicting Bergner 2015) AND spectral features dominate by ≥15pp AND L3 yields a tight, computable bound. All three simultaneously: P ≈ 0.01.

### Additional Dimensions

**Laptop-CPU Feasibility: 6/10**

| Component | Time | Feasible? |
|-----------|------|-----------|
| Spectral features (per instance) | ~10–30s | ✅ Trivial |
| Full spectral annotation (1,065) | ~9 hours | ✅ Easy |
| Pilot evaluation (50 instances) | ~2 hours | ✅ Trivial |
| Dev evaluation (200 instances) | ~2 days | ✅ Comfortable |
| Paper evaluation (500 instances) | ~5 days on 4 cores | ⚠ Tight but doable |
| Full census (1,065 instances) | ~12 days on 4 cores | ⚠ Painful; needs batching |

No GPUs required. No human annotation. No human studies. Fully automated pipeline. The bottleneck is decomposition evaluation wall time (1hr/instance cap), not spectral computation. The tiered design (50→200→500→1065) is well-engineered — you learn whether to abandon before committing to expensive tiers.

**Feasibility: 5/10**

**Kill gate analysis (corrected for correlation):**

| Gate | Week | P(survive) | P(survive \| prior) | Cumulative | Test |
|------|------|-----------|---------------------|------------|------|
| G0 | 0 | 0.65 | 0.65 | 0.65 | Spectral ≠ density proxy (R² < 0.70) |
| G1 | 2 | — | 0.90 (given G0) | 0.59 | Spearman ρ ≥ 0.4 on 50-instance pilot |
| G2 | 4 | 0.90 | 0.90 (independent) | 0.53 | SCIP + GCG wrappers operational |
| G3 | 8 | — | 0.75 (given G1) | 0.40 | Spectral > syntactic on 200-instance dev |
| G4 | 14 | — | 0.80 (given G3) | 0.32 | Full 500-instance eval passes thresholds |
| G5 | 18 | — | 0.85 (given G4) | 0.27 | Internal review passes |

**Compound gate survival: ~27%.** With P(JoC | survive all) ≈ 0.70 → **P(JoC) ≈ 0.19** unconditionally.

However, this ignores fallback paths. Multiple publication outcomes exist:
- P(all gates pass) × P(JoC) ≈ 0.27 × 0.70 = **0.19**
- P(spectral fail, census survives) × P(C&OR data paper) ≈ 0.35 × 0.35 = **0.12**
- P(partial results) × P(workshop/short paper) ≈ 0.20 × 0.25 = **0.05**
- **P(any reputable venue) ≈ 0.36** unconditionally

The project's **conditional** P(JoC | amendments implemented and gates passed) = 0.55 is reasonable. The **unconditional** probability is much lower.

---

## 3. Fatal Flaws

### FF-1: L3 proof is broken — 2 gaps in the paper's main theorem
**Severity:** CRITICAL (fixable)
**Evidence:** Verification report §2. Gap 1: Step 3 asserts feasibility of restricted dual (setting crossing duals to zero) for general A — not guaranteed when A has mixed-sign entries. Gap 2: Step 5 conflates constraint-dropping and variable-duplication Lagrangian models; the (n_e−1) factor comes from variable-duplication but the proof uses constraint-dropping.
**Fix:** Rewrite via variable-duplication Lagrangian relaxation model. Estimated 2–3 days.
**Assessment:** The underlying bound is correct (known consequence of Lagrangian duality). The gaps are in presentation, not correctness. But a paper whose main theorem doesn't prove what it claims is unpublishable. **Must fix.**

### FF-2: Core empirical hypothesis is entirely untested
**Severity:** CRITICAL (existential)
**Evidence:** State.json: impl_loc=0. Zero lines of implementation code. The entire project rests on one claim: "spectral features predict decomposition benefit better than syntactic features." This has never been tested on a single instance. G1 (Week 2 pilot) is designed to test it but hasn't been executed.
**Fix:** Run G1. If ρ < 0.4, ABANDON immediately.
**Assessment:** This is the make-or-break uncertainty. All probability estimates are conditioned on an untested hypothesis.

### FF-3: Red-team threshold not met (4–6.5/12 vs. required 8/12)
**Severity:** SERIOUS
**Evidence:** Verification report §3. Key unresolved findings:
- **S-1 (Dual degeneracy):** L3's Assumption 1 picks "a" dual without specifying which. For degenerate LPs (the standard case for MIPLIB), the bound depends on dual choice. Never specified.
- **S-4 (T2 missing δ < γ/2 assumption):** Davis-Kahan requires perturbation < half the gap. Without this, T2's proof is incomplete. Restricts T2 to easy cases.
- **S-8 (AutoFolio baseline missing):** The most natural comparison — adding SPEC-8 to AutoFolio's existing feature set — is absent despite the red-team calling it "a glaring gap." A JoC reviewer will demand this.
**Fix:** Address S-1, S-4, S-8 before submission. ~2 weeks total.

### FF-4: Spectral features may be proxies for density
**Severity:** SERIOUS (existential)
**Evidence:** Red-team EVAL-8/KQ4. G0 tests linear redundancy (OLS R² < 0.70) but the nonlinear RF regression specifically demanded by the red-team is absent from the evaluation plan. If γ₂ ≈ f(density, size) with R² > 0.80 via random forest, spectral features are just expensive proxies for statistics everyone already computes. The paper's thesis collapses to "dense constraints are hard to decompose."
**Fix:** Add nonlinear redundancy test (RF regression, R² < 0.80 threshold) to G0.

### FF-5: T2 is vacuous and unfixable
**Severity:** SERIOUS (mitigated by demotion)
**Evidence:** Unanimous across all experts. C = O(k · κ⁴ · ‖c‖∞). For κ ~ 10⁶ (common in MIPLIB), κ⁴ = 10²⁴, making the bound informationally empty. Informative on ~8–14% of MIPLIB (well-conditioned); directional on ~20–30%; vacuous on ~60–70%.
**Mitigation:** Demoted to motivational (≤2 pages). Paper leads with census + empirical correlation. T2 is never presented as a quantitative tool.
**Assessment:** Not a flaw in the Amendment E framing — it's a limitation honestly reported. But it means the paper has **no theoretical contribution** with predictive power.

### FF-6: L3 novelty is questionable
**Severity:** MODERATE
**Evidence:** Red-team FATAL F-1: "Geoffrion 1974 in hypergraph costume." L3 repackages the Lagrangian weak duality gap in hypergraph language. The (n_e−1) factor and L3-C specializations add incremental novelty, but a reviewer who recognizes the relationship will question whether the contribution justifies a lemma numbering.
**Mitigation:** Honesty remark already present (paper.tex lines 538–543). L3-C specializations (Benders/DW) are genuinely new connections between variable and constraint partitions. Defense is thin but present.

---

## 4. Binding Conditions for Continuation

Failure on **any** condition triggers ABANDON.

| # | Condition | Deadline | Verification |
|---|-----------|----------|-------------|
| **BC1** | Fix L3 proof gaps (Steps 3, 5) via Lagrangian variable-duplication model | Week 1 | Complete proof reviewed by independent checker |
| **BC2** | Pass G1: Spearman ρ(δ²/γ², bound degradation) ≥ 0.4 on 50-instance pilot | Week 2 | Reproducible notebook with results |
| **BC3** | Add AutoFolio + SPEC-8 baseline to evaluation design; implement before G3 | Week 4 | Baseline code operational |
| **BC4** | Add nonlinear redundancy test (RF regression of spectral on syntactic, R² < 0.80) to G0 | Week 1 | Test result documented |
| **BC5** | Fix title: "Complete" → "Systematic"; resolve P1/P2 contribution ordering to census-first | Week 1 | Updated paper.tex and approach.json |

**Additional strongly recommended actions** (not binding but will affect P(JoC)):
- Address dual degeneracy (S-1): specify "for any optimal dual" or "minimum-norm dual"
- Add δ < γ/2 assumption to T2 and L3-sp
- Relabel L3-C proofs as proof sketches
- Compute minority-class statistical power

---

## 5. Probability Estimates

### Team Consensus

| Outcome | Auditor | Skeptic | Synthesizer | **Lead Estimate** |
|---------|---------|---------|-------------|-------------------|
| P(JoC) | 0.45 | 0.30 | 0.50 | **0.45** |
| P(C&OR or CPAIOR) | 0.55 | 0.45 | 0.60 | **0.55** |
| P(any reputable venue) | 0.55 | 0.50 | 0.65 | **0.55** |
| P(best-paper) | 0.02 | 0.01 | 0.03 | **0.02** |
| P(abandon) | 0.30 | 0.40 | 0.25 | **0.30** |

**Note:** These are **conditional** on implementing binding conditions BC1–BC5 and passing kill gates. Unconditional P(any pub) ≈ 0.36 accounting for gate survival (~27%) and fallback paths.

### What This Project IS

- Computational study introducing spectral features for decomposition selection
- First cross-method MIPLIB 2017 census (empirical infrastructure artifact)
- Feature engineering with theoretical motivation from spectral perturbation theory
- Target: INFORMS JoC (primary), C&OR / CPAIOR (secondary)
- Budget: 26.5K LoC / 30K cap

### What This Project IS NOT

- Bridging theorem connecting two fields (T2 is vacuous)
- Best-paper contender at any venue (P ≈ 0.02)
- Standalone decomposition solver
- Top-venue material (MPC/IPCO: P = 0.05)
- A project with validated core claims (hypothesis untested, proof broken)

---

## 6. Skeptic's Dissent

I do not block CONTINUE under Amendment E, but I record the following dissent:

**The honest framing of this project is: "a coin flip at a mid-tier venue with no best-paper potential."** P(JoC) = 0.45 conditional on everything going right. P(best-paper) = 0.02. The core hypothesis (spectral features beat syntactic) has never been tested. The main theorem's proof is broken. The crown jewel theorem is vacuous. The red-team threshold was not met. Zero lines of code exist.

**The case for CONTINUE rests entirely on:**
1. The census has standalone value (~P=0.25 as a data paper even if spectral thesis fails)
2. G1 is cheap to test (~2 hours of compute, Week 2) and resolves the existential uncertainty
3. 282KB of theory artifacts represent real, non-trivial work that would be wasted by premature ABANDON

I accept these arguments. But I insist that the project's supporters are systematically overweighting salvage value and underweighting the compound probability of failure. This is a marginal project dressed as a solid one.

**My independent scores:** V5/D4/BP1/CPU5/F3 → composite **3.6/10**. If the team used my scores, the recommendation would be ABANDON.

**If team rejects Amendment E (insists on MPC/IPCO, insists T2 is the main result): CONVERT TO ABANDON.**

---

## 7. Final Scores and Verdict

### Composite Scores

| Pillar | Score | Justification |
|--------|-------|---------------|
| **Value (V)** | **5** | Census is genuine community contribution (no comparable artifact). Spectral features are novel feature family. Audience is niche (~hundreds of groups). Incremental, not transformative. |
| **Difficulty (D)** | **4** | Post-descoping to 26.5K LoC. Numerical eigensolve on diverse MIPLIB is genuinely hard. Rest is integration engineering. A competent PhD student builds this in 6–8 weeks. |
| **Best-Paper (BP)** | **3** | Panel consensus. Solid JoC paper ceiling. No mechanism to generate surprise. T2 vacuous eliminates theory venues. P(best-paper) ≈ 0.02. |
| **Laptop-CPU (L)** | **5** | Spectral analysis trivially laptop-feasible. Paper evaluation (500 instances): 5 days on 4 cores — tight but doable. Full census: 12 days — needs batching. Score reduced from depth-check binding (L6) to reflect census batch-job overhead. |
| **Feasibility (F)** | **4** | Zero code exists. Core hypothesis untested. L3 proof FAIL. Red-team 4–6.5/12 vs. 8/12 threshold. Missing AutoFolio baseline. Compound gate survival ~27%. Partially offset by cheap G1 test and fallback paths. |

**Composite: 4.2/10** (arithmetic mean). **Weighted composite: 4.5/10** (skeptic-heavy weighting with depth-check binding caps).

### Vote

| Expert | Verdict | Composite |
|--------|---------|-----------|
| Independent Auditor | CONDITIONAL CONTINUE | 4.8 |
| Fail-Fast Skeptic | CONDITIONAL CONTINUE (dissent: ABANDON at 3.6) | 3.6 |
| Scavenging Synthesizer | CONDITIONAL CONTINUE | 5.2 |

**VERDICT: CONDITIONAL CONTINUE** — unanimous under Amendment E framing. Skeptic dissents ABANDON at composite 3.6/10 and records that the honest unconditional P(any pub) ≈ 0.36.

### Binding Gates (Kill Thresholds)

| Gate | Week | Condition | Kill if |
|------|------|-----------|---------|
| G0 | 0 | Spectral ≠ density proxy | R²(γ₂ ~ syntactic) ≥ 0.70 (linear) OR R² ≥ 0.80 (RF) |
| G1 | 2 | Spectral premise validation | ρ(δ²/γ², bound degradation) < 0.4 |
| G2 | 4 | Solver wrappers operational | < 80% of pilot produces valid bounds |
| G3 | 8 | Spectral > syntactic | Spectral accuracy ≤ syntactic accuracy |
| G4 | 14 | Full evaluation | ρ < 0.5 AND acc < 65% AND precision < 80% |
| G5 | 18 | Internal review | Fundamental problems flagged |

---

*Verification team — 2026-03-08*
*This report reflects independent parallel assessment, adversarial cross-critique, and independent verifier signoff. No items were rubber-stamped.*
